// WanRoPE.swift - 3-way factorized Rotary Position Embeddings for Wan2
// Ported from mlx_video/models/wan_2/rope.py

import Foundation
import MLX
import MLXFast

// MARK: - RoPE Frequency Computation

/// Precompute RoPE frequency parameters as (cos, sin) pairs.
///
/// - Parameters:
///   - maxSeqLen: Maximum sequence length.
///   - dim: Embedding dimension (must be even).
///   - theta: Base frequency (default 10000).
/// - Returns: Frequency tensor of shape [maxSeqLen, dim/2, 2] where last dim is (cos, sin).
public func ropeParams(maxSeqLen: Int, dim: Int, theta: Double = 10000.0) -> MLXArray {
    precondition(dim % 2 == 0, "dim must be even")
    let halfDim = dim / 2

    // Build position × frequency matrix in Float64 for precision, then store as Float32
    // positions: [maxSeqLen], invFreq: [halfDim]
    var cosData = [Float](repeating: 0, count: maxSeqLen * halfDim)
    var sinData = [Float](repeating: 0, count: maxSeqLen * halfDim)

    for pos in 0..<maxSeqLen {
        for j in 0..<halfDim {
            let freq = Double(pos) / Foundation.pow(theta, Double(2 * j) / Double(dim))
            cosData[pos * halfDim + j] = Float(Foundation.cos(freq))
            sinData[pos * halfDim + j] = Float(Foundation.sin(freq))
        }
    }

    let cosArr = MLXArray(cosData).reshaped(maxSeqLen, halfDim, 1)
    let sinArr = MLXArray(sinData).reshaped(maxSeqLen, halfDim, 1)
    // Stack to [maxSeqLen, halfDim, 2]
    return concatenated([cosArr, sinArr], axis: 2)
}

// MARK: - RoPE Application

/// Apply 3-way factorized RoPE to Q or K tensor.
///
/// The head dimension is split into temporal, height, and width components.
/// Each component gets its own frequency table, enabling separate position
/// encoding along each spatial/temporal axis.
///
/// - Parameters:
///   - x: Shape [B, L, numHeads, headDim]
///   - gridSizes: List of (F, H, W) tuples per batch element
///   - freqs: Precomputed cos/sin, shape [1024, d//2, 2] split into 3 parts
///   - precomputedCosSin: Optional (cos, sin) from `ropePrecomputeCosSin()`
/// - Returns: Rotated tensor same shape as x.
public func ropeApply(
    _ x: MLXArray,
    gridSizes: [(Int, Int, Int)],
    freqs: MLXArray,
    precomputedCosSin: (MLXArray, MLXArray)? = nil
) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)
    let n = x.dim(2)
    let d = x.dim(3)
    let halfD = d / 2

    // Fast path: use precomputed cos/sin
    if let (cosF, sinF) = precomputedCosSin {
        let (f0, h0, w0) = gridSizes[0]
        let seqLen = f0 * h0 * w0
        let allSameGrid = b <= 1 || gridSizes.allSatisfy { $0 == gridSizes[0] }

        if allSameGrid {
            // Vectorized path: apply RoPE to all batch elements at once
            let xSeq = x[0..., ..<seqLen].reshaped(b, seqLen, n, halfD, 2)
            let xReal = xSeq[0..., 0..., 0..., 0..., 0]
            let xImag = xSeq[0..., 0..., 0..., 0..., 1]
            let outReal = xReal * cosF - xImag * sinF
            let outImag = xReal * sinF + xImag * cosF
            var xRotated = stacked([outReal, outImag], axis: -1).reshaped(b, seqLen, n, d)
            if seqLen < s {
                xRotated = concatenated([xRotated, x[0..., seqLen...]], axis: 1)
            }
            return xRotated
        } else {
            // Per-element path for mixed grid sizes
            var outputs = [MLXArray]()
            for i in 0..<b {
                let (f, h, w) = gridSizes[i]
                let sl = f * h * w
                let xI = x[i, ..<sl].reshaped(sl, n, halfD, 2)
                let xReal = xI[0..., 0..., 0..., 0]
                let xImag = xI[0..., 0..., 0..., 1]
                let outReal = xReal * cosF - xImag * sinF
                let outImag = xReal * sinF + xImag * cosF
                var xRotated = stacked([outReal, outImag], axis: -1).reshaped(sl, n, d)
                if sl < s {
                    xRotated = concatenated([xRotated, x[i, sl...]], axis: 0)
                }
                outputs.append(xRotated)
            }
            return stacked(outputs)
        }
    }

    // Cast freqs to input dtype to prevent float32 promotion cascade
    var freqsCast = freqs
    if freqs.dtype != x.dtype {
        freqsCast = freqs.asType(x.dtype)
    }

    // Split frequency dimensions: temporal gets more capacity
    let dT = halfD - 2 * (halfD / 3)
    let dH = halfD / 3
    let dW = halfD / 3

    // Split freqs along dim axis
    let freqsT = freqsCast[0..., ..<dT]          // [1024, dT, 2]
    let freqsH = freqsCast[0..., dT..<(dT + dH)] // [1024, dH, 2]
    let freqsW = freqsCast[0..., (dT + dH)..<(dT + dH + dW)] // [1024, dW, 2]

    var outputs = [MLXArray]()
    for i in 0..<b {
        let (f, h, w) = gridSizes[i]
        let seqLen = f * h * w

        // Reshape x to pairs for rotation: [seqLen, n, halfD, 2]
        let xI = x[i, ..<seqLen].reshaped(seqLen, n, halfD, 2)

        // Build per-position frequencies by expanding along grid dims
        let ft = broadcast(
            freqsT[..<f].reshaped(f, 1, 1, dT, 2),
            to: [f, h, w, dT, 2]
        )
        let fh = broadcast(
            freqsH[..<h].reshaped(1, h, 1, dH, 2),
            to: [f, h, w, dH, 2]
        )
        let fw = broadcast(
            freqsW[..<w].reshaped(1, 1, w, dW, 2),
            to: [f, h, w, dW, 2]
        )

        // Concatenate: [f*h*w, halfD, 2]
        let freqsI = concatenated([ft, fh, fw], axis: 3).reshaped(seqLen, 1, halfD, 2)

        // Apply rotation: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
        let cosFreq = freqsI[0..., 0..., 0..., 0]  // [seqLen, 1, halfD]
        let sinFreq = freqsI[0..., 0..., 0..., 1]  // [seqLen, 1, halfD]

        let xReal = xI[0..., 0..., 0..., 0]  // [seqLen, n, halfD]
        let xImag = xI[0..., 0..., 0..., 1]  // [seqLen, n, halfD]

        let outReal = xReal * cosFreq - xImag * sinFreq
        let outImag = xReal * sinFreq + xImag * cosFreq

        // Interleave back: [seqLen, n, halfD, 2] -> [seqLen, n, d]
        var xRotated = stacked([outReal, outImag], axis: -1).reshaped(seqLen, n, d)

        // Handle padding: keep non-rotated tokens after seqLen
        if seqLen < s {
            xRotated = concatenated([xRotated, x[i, seqLen...]], axis: 0)
        }

        outputs.append(xRotated)
    }

    return stacked(outputs)
}

// MARK: - Precompute cos/sin

/// Precompute cos/sin frequency tensors for constant grid sizes.
///
/// Call once before the diffusion loop. Pass result as `precomputedCosSin`
/// to `ropeApply` to skip per-step broadcast/concat.
///
/// - Parameters:
///   - gridSizes: List of (F, H, W) tuples (must be same for all batch elements)
///   - freqs: Precomputed frequencies [1024, d//2, 2]
///   - dtype: Target dtype for the output tensors
/// - Returns: (cosF, sinF) each [seqLen, 1, halfD]
public func ropePrecomputeCosSin(
    gridSizes: [(Int, Int, Int)],
    freqs: MLXArray,
    dtype: DType = .float32
) -> (MLXArray, MLXArray) {
    var freqsCast = freqs
    if freqs.dtype != dtype {
        freqsCast = freqs.asType(dtype)
    }

    let (f, h, w) = gridSizes[0]
    let seqLen = f * h * w
    let halfD = freqsCast.dim(1)

    let dT = halfD - 2 * (halfD / 3)
    let dH = halfD / 3
    let dW = halfD / 3

    let freqsT = freqsCast[0..., ..<dT]
    let freqsH = freqsCast[0..., dT..<(dT + dH)]
    let freqsW = freqsCast[0..., (dT + dH)..<(dT + dH + dW)]

    let ft = broadcast(
        freqsT[..<f].reshaped(f, 1, 1, dT, 2),
        to: [f, h, w, dT, 2]
    )
    let fh = broadcast(
        freqsH[..<h].reshaped(1, h, 1, dH, 2),
        to: [f, h, w, dH, 2]
    )
    let fw = broadcast(
        freqsW[..<w].reshaped(1, 1, w, dW, 2),
        to: [f, h, w, dW, 2]
    )

    let freqsI = concatenated([ft, fh, fw], axis: 3).reshaped(seqLen, 1, halfD, 2)
    return (freqsI[0..., 0..., 0..., 0], freqsI[0..., 0..., 0..., 1])
}
