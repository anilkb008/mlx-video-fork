// Scheduler.swift - Flow matching schedulers for diffusion inference
// Ported from mlx_video/models/wan_2/scheduler.py

import Foundation
import MLX
import MLXFast

// MARK: - Sigma Schedule

/// Compute shifted sigma schedule matching official Wan2.2 scheduler.
func computeSigmas(
    numSteps: Int,
    shift: Float = 1.0,
    numTrainTimesteps: Int = 1000
) -> [Float] {
    // sigma bounds from unshifted training schedule
    var alphas = [Float](repeating: 0, count: numTrainTimesteps)
    for i in 0..<numTrainTimesteps {
        alphas[i] = 1.0 - Float(i) / Float(numTrainTimesteps)
    }
    let sigmaMax = 1.0 - alphas[0]  // (N-1)/N
    let sigmaMin: Float = 0.0

    // Interpolate then apply shift
    var sigmas = [Float](repeating: 0, count: numSteps)
    for i in 0..<numSteps {
        let t = Float(i) / Float(numSteps)
        let sigma = sigmaMax + t * (sigmaMin - sigmaMax)
        sigmas[i] = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }
    sigmas.append(0.0)
    return sigmas
}

// MARK: - Scheduler Protocol

public protocol DiffusionScheduler: Sendable {
    mutating func setTimesteps(numSteps: Int, shift: Float)
    mutating func step(modelOutput: MLXArray, timestep: MLXArray, sample: MLXArray) -> MLXArray
    mutating func reset()
    var timesteps: MLXArray? { get }
    var sigmas: MLXArray? { get }
}

// MARK: - Euler Scheduler

/// 1st-order Euler scheduler for flow matching diffusion.
public struct FlowMatchEulerScheduler: DiffusionScheduler {
    public let numTrainTimesteps: Int
    public private(set) var timesteps: MLXArray?
    public private(set) var sigmas: MLXArray?
    private var sigmasFloat: [Float] = []
    private var stepIndex: Int = 0

    public init(numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
    }

    public mutating func setTimesteps(numSteps: Int, shift: Float = 1.0) {
        let s = computeSigmas(numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        sigmasFloat = s
        sigmas = MLXArray(s)
        // Integer timesteps matching reference
        let ts = s.dropLast().map { Int($0 * Float(numTrainTimesteps)) }
        timesteps = MLXArray(ts.map { Float($0) })
        stepIndex = 0
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestep: MLXArray,
        sample: MLXArray
    ) -> MLXArray {
        let dt = sigmasFloat[stepIndex + 1] - sigmasFloat[stepIndex]
        let xNext = sample + dt * modelOutput
        stepIndex += 1
        return xNext
    }

    public mutating func reset() {
        stepIndex = 0
    }
}

// MARK: - DPM++ 2M Scheduler

/// DPM-Solver++(2M) for flow matching diffusion.
public struct FlowDPMPP2MScheduler: DiffusionScheduler {
    public let numTrainTimesteps: Int
    public let lowerOrderFinal: Bool
    public private(set) var timesteps: MLXArray?
    public private(set) var sigmas: MLXArray?
    private var sigmasFloat: [Float] = []
    private var stepIndex: Int = 0
    private var numSteps: Int = 0
    private var prevX0: MLXArray?

    public init(numTrainTimesteps: Int = 1000, lowerOrderFinal: Bool = true) {
        self.numTrainTimesteps = numTrainTimesteps
        self.lowerOrderFinal = lowerOrderFinal
    }

    private static func lambda_fn(_ sigma: Float) -> Float {
        if sigma >= 1.0 { return -.infinity }
        if sigma <= 0.0 { return .infinity }
        return log((1.0 - sigma) / sigma)
    }

    public mutating func setTimesteps(numSteps: Int, shift: Float = 1.0) {
        let s = computeSigmas(numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        sigmasFloat = s
        sigmas = MLXArray(s)
        let ts = s.dropLast().map { Int($0 * Float(numTrainTimesteps)) }
        timesteps = MLXArray(ts.map { Float($0) })
        stepIndex = 0
        self.numSteps = numSteps
        prevX0 = nil
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestep: MLXArray,
        sample: MLXArray
    ) -> MLXArray {
        let i = stepIndex
        let sigmaCur = sigmasFloat[i]
        let sigmaNext = sigmasFloat[i + 1]

        // velocity -> x0: x0 = sample - sigma * v
        let x0 = sample - sigmaCur * modelOutput

        let useFirstOrder = prevX0 == nil || (
            lowerOrderFinal && i == numSteps - 1 && numSteps < 15
        )

        let xNext: MLXArray
        if useFirstOrder || sigmaNext == 0.0 {
            if sigmaNext == 0.0 {
                xNext = x0
            } else {
                let lambdaCur = Self.lambda_fn(sigmaCur)
                let lambdaNext = Self.lambda_fn(sigmaNext)
                let h = lambdaNext - lambdaCur
                let alphaNext = 1.0 - sigmaNext
                let coeffX = sigmaNext / sigmaCur
                let coeffX0 = alphaNext * expm1(-Double(h))
                xNext = Float(coeffX) * sample - Float(coeffX0) * x0
            }
        } else {
            let sigmaPrev = sigmasFloat[i - 1]
            let lambdaPrev = Self.lambda_fn(sigmaPrev)
            let lambdaCur = Self.lambda_fn(sigmaCur)
            let lambdaNext = Self.lambda_fn(sigmaNext)

            let h = lambdaNext - lambdaCur
            let h0 = lambdaCur - lambdaPrev
            let r0 = h0 / h

            let D0 = x0
            let D1 = (1.0 / r0) * (x0 - prevX0!)

            let alphaNext = 1.0 - sigmaNext
            let expNegHM1 = Float(expm1(-Double(h)))

            xNext = (sigmaNext / sigmaCur) * sample
                - (alphaNext * expNegHM1) * D0
                - 0.5 * (alphaNext * expNegHM1) * D1
        }

        prevX0 = x0
        stepIndex += 1
        return xNext
    }

    public mutating func reset() {
        stepIndex = 0
        prevX0 = nil
    }
}

// MARK: - UniPC Scheduler

/// UniPC (Unified Predictor-Corrector) for flow matching diffusion.
public struct FlowUniPCScheduler: DiffusionScheduler {
    public let numTrainTimesteps: Int
    public let solverOrder: Int
    public let lowerOrderFinal: Bool
    public let useCorrector: Bool
    public let disableCorrector: Set<Int>
    public private(set) var timesteps: MLXArray?
    public private(set) var sigmas: MLXArray?
    private var sigmasFloat: [Float] = []
    private var stepIndex: Int = 0
    private var numSteps: Int = 0
    private var lowerOrderNums: Int = 0
    private var modelOutputs: [MLXArray?] = []
    private var lastSample: MLXArray?
    private var thisOrder: Int = 1

    public init(
        numTrainTimesteps: Int = 1000,
        solverOrder: Int = 2,
        lowerOrderFinal: Bool = true,
        disableCorrector: Set<Int> = [],
        useCorrector: Bool = true
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.solverOrder = solverOrder
        self.lowerOrderFinal = lowerOrderFinal
        self.useCorrector = useCorrector
        self.disableCorrector = disableCorrector
        self.modelOutputs = Array(repeating: nil, count: solverOrder)
    }

    private static func lambda_fn(_ sigma: Float) -> Float {
        if sigma >= 1.0 { return -.infinity }
        if sigma <= 0.0 { return .infinity }
        return log((1.0 - sigma) / sigma)
    }

    public mutating func setTimesteps(numSteps: Int, shift: Float = 1.0) {
        let s = computeSigmas(numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        sigmasFloat = s
        sigmas = MLXArray(s)
        let ts = s.dropLast().map { Int($0 * Float(numTrainTimesteps)) }
        timesteps = MLXArray(ts.map { Float($0) })
        stepIndex = 0
        self.numSteps = numSteps
        lowerOrderNums = 0
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lastSample = nil
        thisOrder = 1
    }

    private func convertOutput(velocity: MLXArray, sample: MLXArray) -> MLXArray {
        let sigma = sigmasFloat[stepIndex]
        return sample - sigma * velocity
    }

    private func uniPBH2(x0: MLXArray, sample: MLXArray, order: Int) -> MLXArray {
        let i = stepIndex
        let sigmaS0 = sigmasFloat[i]
        let sigmaT = sigmasFloat[i + 1]

        if sigmaT == 0.0 { return x0 }

        let lambdaS0 = Self.lambda_fn(sigmaS0)
        let lambdaT = Self.lambda_fn(sigmaT)
        let h = lambdaT - lambdaS0
        let hh = -h

        let alphaT = 1.0 - sigmaT
        let hPhi1 = Float(expm1(Double(hh)))
        let bH = hPhi1

        let m0 = modelOutputs.last!!
        var xT = (sigmaT / sigmaS0) * sample - (alphaT * hPhi1) * m0

        if order >= 2 {
            var rks = [Float]()
            var d1s = [MLXArray]()
            for k in 1..<order {
                let siIdx = i - k
                guard siIdx >= 0, let mk = modelOutputs[modelOutputs.count - (k + 1)] else { break }
                let sigmaSk = sigmasFloat[siIdx]
                let lambdaSk = Self.lambda_fn(sigmaSk)
                let rk = (lambdaSk - lambdaS0) / h
                if rk.isInfinite { break }
                rks.append(rk)
                d1s.append((mk - m0) / rk)
            }

            if !d1s.isEmpty {
                let rhosP: [Float] = [0.5]
                var predRes = MLXArray.zeros(like: m0)
                for (idx, d) in d1s.enumerated() {
                    if idx < rhosP.count {
                        predRes = predRes + rhosP[idx] * d
                    }
                }
                xT = xT - (alphaT * bH) * predRes
            }
        }

        return xT
    }

    private func uniCBH2(
        modelX0: MLXArray,
        lastSample: MLXArray,
        thisSample: MLXArray,
        order: Int
    ) -> MLXArray {
        let i = stepIndex
        let sigmaS0 = sigmasFloat[i - 1]
        let sigmaT = sigmasFloat[i]

        if sigmaT == 0.0 { return thisSample }

        let lambdaS0 = Self.lambda_fn(sigmaS0)
        let lambdaT = Self.lambda_fn(sigmaT)
        let h = lambdaT - lambdaS0
        let hh = -h

        let alphaT = 1.0 - sigmaT
        let hPhi1 = Float(expm1(Double(hh)))
        let bH = hPhi1

        let m0 = modelOutputs.last!!
        let xT_ = (sigmaT / sigmaS0) * lastSample - (alphaT * hPhi1) * m0

        let d1T = modelX0 - m0

        // Simple order-1 correction
        let rhosC: [Float] = [0.5]
        let xT = xT_ - (alphaT * bH) * (rhosC[0] * d1T)
        return xT
    }

    public mutating func step(
        modelOutput: MLXArray,
        timestep: MLXArray,
        sample: MLXArray
    ) -> MLXArray {
        let i = stepIndex
        let x0 = convertOutput(velocity: modelOutput, sample: sample)

        var currentSample = sample

        // Corrector
        let shouldCorrect = useCorrector && i > 0 && !disableCorrector.contains(i - 1) && lastSample != nil
        if shouldCorrect {
            currentSample = uniCBH2(modelX0: x0, lastSample: lastSample!, thisSample: currentSample, order: thisOrder)
        }

        // Shift model output history
        for k in 0..<(solverOrder - 1) {
            modelOutputs[k] = modelOutputs[k + 1]
        }
        modelOutputs[solverOrder - 1] = x0

        // Determine prediction order
        var order: Int
        if lowerOrderFinal {
            order = min(solverOrder, numSteps - i)
        } else {
            order = solverOrder
        }
        thisOrder = min(order, lowerOrderNums + 1)

        // Predict
        lastSample = currentSample
        let xNext = uniPBH2(x0: x0, sample: currentSample, order: thisOrder)

        if lowerOrderNums < solverOrder {
            lowerOrderNums += 1
        }

        stepIndex += 1
        return xNext
    }

    public mutating func reset() {
        stepIndex = 0
        lowerOrderNums = 0
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lastSample = nil
        thisOrder = 1
    }
}
