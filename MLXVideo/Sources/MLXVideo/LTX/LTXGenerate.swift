// LTXGenerate.swift
// Generation pipeline for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/generate.py

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Constants

/// Distilled model sigma schedules.
public let ltxStage1Sigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
public let ltxStage2Sigmas: [Float] = [0.909375, 0.725, 0.421875, 0.0]

/// Dev model scheduling constants.
let baseShiftAnchor: Int = 1024
let maxShiftAnchor: Int = 4096

// MARK: - Scheduler

/// LTX-2 scheduler for sigma generation (dev model).
///
/// Generates a sigma schedule with token-count-dependent shifting and optional
/// stretching to a terminal value.
public func ltx2Scheduler(
    steps: Int,
    numTokens: Int? = nil,
    maxShift: Float = 2.05,
    baseShift: Float = 0.95,
    stretch: Bool = true,
    terminal: Float = 0.1
) -> [Float] {
    let tokens = numTokens ?? maxShiftAnchor

    // Linear spacing from 1.0 to 0.0
    var sigmas = (0...steps).map { i in
        1.0 - Float(i) / Float(steps)
    }

    // Compute shift based on token count
    let x1 = Float(baseShiftAnchor)
    let x2 = Float(maxShiftAnchor)
    let mm = (maxShift - baseShift) / (x2 - x1)
    let b = baseShift - mm * x1
    let sigmaShift = Float(tokens) * mm + b

    // Apply shift transformation
    sigmas = sigmas.map { s in
        if s != 0 {
            return exp(sigmaShift) / (exp(sigmaShift) + pow(1.0 / s - 1.0, 1.0))
        }
        return 0.0
    }

    // Stretch sigmas to terminal value
    if stretch {
        let nonZeroIndices = sigmas.indices.filter { sigmas[$0] != 0 }
        if let lastNonZero = nonZeroIndices.last {
            let oneMinusLast = 1.0 - sigmas[lastNonZero]
            let scaleFactor = oneMinusLast / (1.0 - terminal)
            for idx in nonZeroIndices {
                let oneMinusZ = 1.0 - sigmas[idx]
                sigmas[idx] = 1.0 - (oneMinusZ / scaleFactor)
            }
        }
    }

    return sigmas
}

// MARK: - Position Grid

/// Create position grid for RoPE in pixel space.
///
/// - Returns: Position grid of shape (B, 3, numPatches, 2) in pixel space
///   where dim 2 is [start, end) bounds for each patch.
public func createPositionGrid(
    batchSize: Int,
    numFrames: Int,
    height: Int,
    width: Int,
    temporalScale: Int = 8,
    spatialScale: Int = 32,
    fps: Float = 24.0,
    causalFix: Bool = true
) -> MLXArray {
    let numPatches = numFrames * height * width

    // Create coordinate arrays
    var positions = [Float](repeating: 0, count: batchSize * 3 * numPatches * 2)

    for b in 0..<batchSize {
        for t in 0..<numFrames {
            for h in 0..<height {
                for w in 0..<width {
                    let patchIdx = t * height * width + h * width + w
                    let baseIdx = b * 3 * numPatches * 2

                    // Temporal start/end
                    var tStart = Float(t * temporalScale)
                    var tEnd = Float((t + 1) * temporalScale)

                    if causalFix {
                        tStart = max(tStart + 1 - Float(temporalScale), 0)
                        tEnd = max(tEnd + 1 - Float(temporalScale), 0)
                    }

                    // Divide temporal by fps
                    tStart /= fps
                    tEnd /= fps

                    positions[baseIdx + 0 * numPatches * 2 + patchIdx * 2 + 0] = tStart
                    positions[baseIdx + 0 * numPatches * 2 + patchIdx * 2 + 1] = tEnd

                    // Spatial height start/end
                    let hStart = Float(h * spatialScale)
                    let hEnd = Float((h + 1) * spatialScale)
                    positions[baseIdx + 1 * numPatches * 2 + patchIdx * 2 + 0] = hStart
                    positions[baseIdx + 1 * numPatches * 2 + patchIdx * 2 + 1] = hEnd

                    // Spatial width start/end
                    let wStart = Float(w * spatialScale)
                    let wEnd = Float((w + 1) * spatialScale)
                    positions[baseIdx + 2 * numPatches * 2 + patchIdx * 2 + 0] = wStart
                    positions[baseIdx + 2 * numPatches * 2 + patchIdx * 2 + 1] = wEnd
                }
            }
        }
    }

    // Cast through bfloat16 to match PyTorch's behavior
    let posArray = MLXArray(positions, [batchSize, 3, numPatches, 2])
    let bf16 = posArray.asType(.bfloat16)
    eval(bf16)
    return bf16.asType(.float32)
}

// MARK: - CFG Delta

/// Compute CFG delta for classifier-free guidance.
func cfgDelta(cond: MLXArray, uncond: MLXArray, scale: Float) -> MLXArray {
    return (scale - 1.0) * (cond - uncond)
}

// MARK: - Denoise Distilled

/// Run denoising loop for distilled pipeline (no CFG).
public func denoiseDistilled(
    latents: MLXArray,
    positions: MLXArray,
    textEmbeddings: MLXArray,
    transformer: LTXModel,
    sigmas: [Float],
    verbose: Bool = true
) -> MLXArray {
    let dtype = latents.dtype
    var currentLatents = latents.asType(.float32)

    let numSteps = sigmas.count - 1

    for i in 0..<numSteps {
        let sigma = sigmas[i]
        let sigmaNext = sigmas[i + 1]

        let shape = currentLatents.shape
        let b = shape[0], c = shape[1], f = shape[2], h = shape[3], w = shape[4]
        let numTokens = f * h * w

        // Flatten: (B, C, F, H, W) -> (B, F*H*W, C)
        let latentsFlat = currentLatents
            .reshaped([b, c, -1])
            .transposed(axes: [0, 2, 1])
            .asType(dtype)

        let timesteps = MLXArray.full([b, numTokens], values: MLXArray(sigma), type: dtype)
        let sigmaArray = MLXArray.full([b], values: MLXArray(sigma), type: dtype)

        let videoModality = Modality(
            latent: latentsFlat,
            timesteps: timesteps,
            positions: positions,
            context: textEmbeddings,
            enabled: true,
            sigma: sigmaArray
        )

        let (velocity, _) = transformer(video: videoModality)
        eval(velocity!)

        // Compute denoised (x0) in float32
        let sigmaF32 = MLXArray(sigma)
        let latentsFlatF32 = currentLatents.reshaped([b, c, -1]).transposed(axes: [0, 2, 1])
        let timestepsF32 = timesteps.asType(.float32).expandedDimensions(axis: -1)
        let x0F32 = latentsFlatF32 - timestepsF32 * velocity!.asType(.float32)
        let denoised = x0F32.transposed(axes: [0, 2, 1]).reshaped([b, c, f, h, w])

        eval(denoised)

        // Euler step in float32
        if sigmaNext > 0 {
            let sigmaNextF32 = MLXArray(sigmaNext)
            currentLatents = denoised + sigmaNextF32 * (currentLatents - denoised) / sigmaF32
        } else {
            currentLatents = denoised
        }

        eval(currentLatents)

        if verbose {
            print("Step \(i + 1)/\(numSteps) (sigma: \(String(format: "%.4f", sigma)) -> \(String(format: "%.4f", sigmaNext)))")
        }
    }

    return currentLatents.asType(dtype)
}

// MARK: - Denoise Dev (with CFG)

/// Run denoising loop for dev pipeline with classifier-free guidance.
public func denoiseDev(
    latents: MLXArray,
    positions: MLXArray,
    textEmbeddingsPos: MLXArray,
    textEmbeddingsNeg: MLXArray,
    transformer: LTXModel,
    sigmas: [Float],
    cfgScale: Float = 4.0,
    cfgRescale: Float = 0.0,
    verbose: Bool = true,
    stgScale: Float = 0.0,
    stgBlocks: [Int]? = nil
) -> MLXArray {
    let dtype = latents.dtype
    var currentLatents = latents.asType(.float32)

    let useCfg = cfgScale != 1.0
    let useStg = stgScale != 0.0 && stgBlocks != nil
    let numSteps = sigmas.count - 1

    // Precompute RoPE once
    let precomputedRope = precomputeFreqsCis(
        indicesGrid: positions,
        dim: transformer.innerDim,
        theta: transformer.positionalEmbeddingTheta,
        maxPos: transformer.positionalEmbeddingMaxPos,
        useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
        numAttentionHeads: transformer.numAttentionHeads,
        ropeType: transformer.ropeType,
        doublePrecision: transformer.config.doublePrecisionRope
    )
    eval(precomputedRope.0, precomputedRope.1)

    for i in 0..<numSteps {
        let sigma = sigmas[i]
        let sigmaNext = sigmas[i + 1]

        let shape = currentLatents.shape
        let b = shape[0], c = shape[1], f = shape[2], h = shape[3], w = shape[4]
        let numTokens = f * h * w

        let latentsFlat = currentLatents
            .reshaped([b, c, -1])
            .transposed(axes: [0, 2, 1])
            .asType(dtype)

        let timesteps = MLXArray.full([b, numTokens], values: MLXArray(sigma), type: dtype)
        let sigmaArray = MLXArray.full([b], values: MLXArray(sigma), type: dtype)

        // Positive conditioning pass
        let videoModalityPos = Modality(
            latent: latentsFlat,
            timesteps: timesteps,
            positions: positions,
            context: textEmbeddingsPos,
            enabled: true,
            positionalEmbeddings: precomputedRope,
            sigma: sigmaArray
        )
        let (velocityPos, _) = transformer(video: videoModalityPos)

        // Convert velocity to x0
        let latentsFlatF32 = currentLatents.reshaped([b, c, -1]).transposed(axes: [0, 2, 1])
        let timestepsF32 = timesteps.asType(.float32).expandedDimensions(axis: -1)
        let x0PosF32 = latentsFlatF32 - timestepsF32 * velocityPos!.asType(.float32)

        var x0GuidedF32 = x0PosF32

        if useCfg {
            // Negative conditioning pass
            let videoModalityNeg = Modality(
                latent: latentsFlat,
                timesteps: timesteps,
                positions: positions,
                context: textEmbeddingsNeg,
                enabled: true,
                positionalEmbeddings: precomputedRope,
                sigma: sigmaArray
            )
            let (velocityNeg, _) = transformer(video: videoModalityNeg)
            let x0NegF32 = latentsFlatF32 - timestepsF32 * velocityNeg!.asType(.float32)

            // Standard CFG
            x0GuidedF32 = x0PosF32 + (cfgScale - 1.0) * (x0PosF32 - x0NegF32)
        }

        // STG pass
        if useStg {
            let (velocityPtb, _) = transformer(
                video: videoModalityPos, stgVideoBlocks: stgBlocks
            )
            eval(velocityPtb!)

            let x0PtbF32 = latentsFlatF32 - timestepsF32 * velocityPtb!.asType(.float32)
            x0GuidedF32 = x0GuidedF32 + stgScale * (x0PosF32 - x0PtbF32)
        }

        // Apply CFG rescale
        if cfgRescale > 0.0 && (useCfg || useStg) {
            let vFactor = x0PosF32.variance().sqrt() / (x0GuidedF32.variance().sqrt() + 1e-8)
            let factor = cfgRescale * vFactor + (1.0 - cfgRescale)
            x0GuidedF32 = x0GuidedF32 * factor
        }

        // Reshape x0 from token space to spatial
        let denoised = x0GuidedF32.transposed(axes: [0, 2, 1]).reshaped([b, c, f, h, w])

        let sigmaF32 = MLXArray(sigma)

        // Euler step
        if sigmaNext > 0 {
            let sigmaNextF32 = MLXArray(sigmaNext)
            currentLatents = denoised + sigmaNextF32 * (currentLatents - denoised) / sigmaF32
        } else {
            currentLatents = denoised
        }

        eval(currentLatents)

        if verbose {
            print("Step \(i + 1)/\(numSteps) (sigma: \(String(format: "%.4f", sigma)) -> \(String(format: "%.4f", sigmaNext)))")
        }
    }

    return currentLatents.asType(dtype)
}

// MARK: - LTXPipeline

/// LTX text-to-video generation pipeline.
///
/// Supports both distilled (fixed sigmas, no CFG) and dev (dynamic sigmas, CFG) modes.
public class LTXPipeline {

    public enum PipelineType: Sendable {
        case distilled
        case dev
    }

    public let transformer: LTXModel
    public let pipelineType: PipelineType

    public init(transformer: LTXModel, pipelineType: PipelineType = .distilled) {
        self.transformer = transformer
        self.pipelineType = pipelineType
    }

    /// Create a pipeline from a local model directory.
    ///
    /// Automatically handles quantized models (Q4, Q8) by reading the quantization
    /// config from config.json.
    ///
    /// - Parameters:
    ///   - modelPath: Local path to the model directory
    ///   - pipelineType: Pipeline type (distilled or dev)
    public static func fromPretrained(
        modelPath: String,
        pipelineType: PipelineType = .distilled
    ) throws -> LTXPipeline {
        print("Loading LTX model from: \(modelPath)")
        let model = try LTXModel.fromPretrained(modelPath: modelPath, strict: false)
        return LTXPipeline(transformer: model, pipelineType: pipelineType)
    }

    /// Create a pipeline from a HuggingFace repo ID.
    ///
    /// Downloads the model if not cached locally. Supports quantized models.
    ///
    /// Example:
    /// ```swift
    /// let pipeline = try await LTXPipeline.fromHub("dgrauet/ltx-2.3-mlx-q4")
    /// ```
    ///
    /// - Parameters:
    ///   - repoId: HuggingFace repo ID (e.g. `dgrauet/ltx-2.3-mlx-q4`)
    ///   - pipelineType: Pipeline type (distilled or dev)
    public static func fromHub(
        _ repoId: String,
        pipelineType: PipelineType = .distilled
    ) async throws -> LTXPipeline {
        let localPath = try await getModelPath(repoId)
        return try fromPretrained(modelPath: localPath, pipelineType: pipelineType)
    }

    /// Generate video latents from text embeddings.
    ///
    /// - Parameters:
    ///   - textEmbeddings: Text encoder output of shape (B, seqLen, dim).
    ///   - negativeTextEmbeddings: Negative text embeddings for CFG (dev pipeline only).
    ///   - numFrames: Number of latent frames to generate.
    ///   - height: Latent height.
    ///   - width: Latent width.
    ///   - numSteps: Number of denoising steps.
    ///   - cfgScale: Classifier-free guidance scale (dev pipeline only).
    ///   - seed: Random seed.
    ///   - verbose: Whether to print progress.
    /// - Returns: Denoised video latents of shape (B, C, F, H, W).
    public func generateLatents(
        textEmbeddings: MLXArray,
        negativeTextEmbeddings: MLXArray? = nil,
        numFrames: Int,
        height: Int,
        width: Int,
        numSteps: Int? = nil,
        cfgScale: Float = 4.0,
        cfgRescale: Float = 0.0,
        seed: UInt64 = 42,
        verbose: Bool = true
    ) -> MLXArray {
        let batchSize = textEmbeddings.shape[0]
        let channels = transformer.config.inChannels

        // Initialize random latents
        MLXRandom.seed(seed)
        var latents = MLXRandom.normal([batchSize, channels, numFrames, height, width])
        latents = latents.asType(.bfloat16)
        eval(latents)

        // Create position grid
        let positions = createPositionGrid(
            batchSize: batchSize,
            numFrames: numFrames,
            height: height,
            width: width
        )
        eval(positions)

        switch pipelineType {
        case .distilled:
            let sigmas = numSteps != nil
                ? ltx2Scheduler(steps: numSteps!, numTokens: numFrames * height * width)
                : ltxStage1Sigmas
            return denoiseDistilled(
                latents: latents,
                positions: positions,
                textEmbeddings: textEmbeddings,
                transformer: transformer,
                sigmas: sigmas,
                verbose: verbose
            )

        case .dev:
            let steps = numSteps ?? 30
            let sigmas = ltx2Scheduler(
                steps: steps,
                numTokens: numFrames * height * width
            )
            let negEmb = negativeTextEmbeddings ?? MLXArray.zeros(like: textEmbeddings)
            return denoiseDev(
                latents: latents,
                positions: positions,
                textEmbeddingsPos: textEmbeddings,
                textEmbeddingsNeg: negEmb,
                transformer: transformer,
                sigmas: sigmas,
                cfgScale: cfgScale,
                cfgRescale: cfgRescale,
                verbose: verbose
            )
        }
    }
}
