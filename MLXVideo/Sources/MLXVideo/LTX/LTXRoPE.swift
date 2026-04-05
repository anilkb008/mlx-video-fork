// LTXRoPE.swift
// Rotary Position Embeddings for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/rope.py

import Foundation
import MLX
import MLXNN

// MARK: - Apply Rotary Embeddings

/// Apply rotary position embeddings to input tensor.
public func applyRotaryEmb(
    _ inputTensor: MLXArray,
    _ freqsCis: (MLXArray, MLXArray),
    _ ropeType: LTXRopeType = .interleaved
) -> MLXArray {
    switch ropeType {
    case .interleaved:
        return applyInterleavedRotaryEmb(inputTensor, cosFreqs: freqsCis.0, sinFreqs: freqsCis.1)
    case .split:
        return applySplitRotaryEmb(inputTensor, cosFreqs: freqsCis.0, sinFreqs: freqsCis.1)
    case .twoD:
        fatalError("2D RoPE type is not supported")
    }
}

/// Apply interleaved rotary embeddings.
/// Pairs adjacent dimensions and applies rotation.
func applyInterleavedRotaryEmb(
    _ inputTensor: MLXArray,
    cosFreqs: MLXArray,
    sinFreqs: MLXArray
) -> MLXArray {
    let inputDtype = inputTensor.dtype
    var x = inputTensor.asType(.float32)
    let cos = cosFreqs.asType(.float32)
    let sin = sinFreqs.asType(.float32)

    // Reshape to pair adjacent dimensions: (..., dim) -> (..., dim/2, 2)
    let shape = x.shape
    let lastDim = shape[shape.count - 1]
    let newShape = Array(shape.dropLast()) + [lastDim / 2, 2]
    x = x.reshaped(newShape)

    // Extract pairs
    let t1 = x[.ellipsis, 0]  // Even indices
    let t2 = x[.ellipsis, 1]  // Odd indices

    // Apply rotation: (-t2, t1) pattern
    let tRot = MLX.stacked([-t2, t1], axis: -1)

    // Flatten back
    x = x.reshaped(shape)
    let tRotFlat = tRot.reshaped(shape)

    // Apply rotary embeddings
    let out = x * cos + tRotFlat * sin

    return out.asType(inputDtype)
}

/// Apply split rotary embeddings.
/// Splits dimensions into two halves and applies rotation.
func applySplitRotaryEmb(
    _ inputTensor: MLXArray,
    cosFreqs: MLXArray,
    sinFreqs: MLXArray
) -> MLXArray {
    let inputDtype = inputTensor.dtype
    var needsReshape = false
    let originalShape = inputTensor.shape
    var x = inputTensor

    // Handle dimension mismatch
    if x.ndim != 4 && cosFreqs.ndim == 4 {
        let b = cosFreqs.shape[0]
        let h = cosFreqs.shape[1]
        let t = cosFreqs.shape[2]
        x = x.reshaped([b, t, h, -1])
        x = x.swappedAxes(1, 2)
        needsReshape = true
    }

    x = x.asType(.float32)
    let cos = cosFreqs.asType(.float32)
    let sin = sinFreqs.asType(.float32)

    // Split into two halves
    let dim = x.shape[x.ndim - 1]
    let splitShape = Array(x.shape.dropLast()) + [2, dim / 2]
    let splitInput = x.reshaped(splitShape)

    let firstHalf = splitInput[.ellipsis, 0, 0...]   // (..., dim//2)
    let secondHalf = splitInput[.ellipsis, 1, 0...]   // (..., dim//2)

    // Apply cosine to both halves and sine cross-terms
    var outputFirst = firstHalf * cos - sin * secondHalf
    var outputSecond = secondHalf * cos + sin * firstHalf

    // Stack back together
    let output = MLX.stacked([outputFirst, outputSecond], axis: -2)

    // Flatten
    var result = output.reshaped(x.shape)

    if needsReshape {
        let b = result.shape[0]
        let h = result.shape[1]
        let t = result.shape[2]
        let d = result.shape[3]
        result = result.swappedAxes(1, 2)
        result = result.reshaped([b, t, h * d])
    }

    return result.asType(inputDtype)
}

// MARK: - Frequency Grid Generation

/// Generate frequency grid for RoPE.
func generateFreqGrid(
    positionalEmbeddingTheta theta: Float,
    positionalEmbeddingMaxPosCount: Int,
    innerDim: Int
) -> MLXArray {
    let start: Float = 1.0
    let end: Float = theta
    let nElem = 2 * positionalEmbeddingMaxPosCount

    let logStart = log(start) / log(theta)
    let logEnd = log(end) / log(theta)

    var numIndices = innerDim / nElem
    if numIndices == 0 { numIndices = 1 }

    let linSpace = MLX.linSpace(Float(logStart), Float(logEnd), count: numIndices)
    let powIndices = MLX.pow(MLXArray(theta), linSpace)

    return powIndices * Float(Float.pi / 2.0)
}

/// Convert indices to fractional positions.
func getFractionalPositions(
    indicesGrid: MLXArray,
    maxPos: [Int]
) -> MLXArray {
    let nPosDims = indicesGrid.shape[1]
    assert(nPosDims == maxPos.count)

    var fractionalPositions: [MLXArray] = []
    for i in 0..<nPosDims {
        let frac = indicesGrid[0..., i] / Float(maxPos[i])
        fractionalPositions.append(frac)
    }

    return MLX.stacked(fractionalPositions, axis: -1)
}

/// Generate frequencies from indices and position grid.
func generateFreqs(
    indices: MLXArray,
    indicesGrid: MLXArray,
    maxPos: [Int],
    useMiddleIndicesGrid: Bool
) -> MLXArray {
    var grid = indicesGrid

    if useMiddleIndicesGrid {
        assert(grid.shape.count == 4)
        assert(grid.shape[3] == 2)
        let start = grid[.ellipsis, 0]
        let end = grid[.ellipsis, 1]
        grid = (start + end) / 2.0
    } else if grid.shape.count == 4 {
        grid = grid[.ellipsis, 0]
    }

    let fractionalPositions = getFractionalPositions(indicesGrid: grid, maxPos: maxPos)
    let scaledPositions = fractionalPositions * 2.0 - 1.0  // (B, T, nDims)

    // Outer product with indices
    // scaledPositions: (B, T, nDims) -> (B, T, nDims, 1)
    // indices: (numIndices,) -> (1, 1, 1, numIndices)
    let posExpanded = scaledPositions.expandedDimensions(axis: -1)
    let idxExpanded = indices.expandedDimensions(axes: [0, 0, 0])

    var freqs = posExpanded * idxExpanded  // (B, T, nDims, numIndices)

    // Transpose and flatten
    freqs = freqs.swappedAxes(-1, -2)  // (B, T, numIndices, nDims)
    let fShape = freqs.shape
    freqs = freqs.reshaped(Array(fShape.dropLast(2)) + [-1])

    return freqs
}

// MARK: - Prepare cos/sin frequencies

/// Prepare cos/sin frequencies for split RoPE.
func splitFreqsCis(
    freqs: MLXArray,
    padSize: Int,
    numAttentionHeads: Int
) -> (MLXArray, MLXArray) {
    var cosFreq = MLX.cos(freqs)
    var sinFreq = MLX.sin(freqs)

    if padSize != 0 {
        let cosPadding = MLXArray.ones(like: cosFreq[0..., ..<padSize])
        let sinPadding = MLXArray.zeros(like: sinFreq[0..., ..<padSize])
        cosFreq = MLX.concatenated([cosPadding, cosFreq], axis: -1)
        sinFreq = MLX.concatenated([sinPadding, sinFreq], axis: -1)
    }

    let b = cosFreq.shape[0]
    let t = cosFreq.shape[1]

    cosFreq = cosFreq.reshaped([b, t, numAttentionHeads, -1])
    sinFreq = sinFreq.reshaped([b, t, numAttentionHeads, -1])

    cosFreq = cosFreq.swappedAxes(1, 2)
    sinFreq = sinFreq.swappedAxes(1, 2)

    return (cosFreq, sinFreq)
}

/// Prepare cos/sin frequencies for interleaved RoPE.
func interleavedFreqsCis(
    freqs: MLXArray,
    padSize: Int
) -> (MLXArray, MLXArray) {
    var cosFreq = MLX.cos(freqs)
    var sinFreq = MLX.sin(freqs)

    // Repeat interleave: each element repeated twice
    cosFreq = MLX.repeated(cosFreq, count: 2, axis: -1)
    sinFreq = MLX.repeated(sinFreq, count: 2, axis: -1)

    if padSize != 0 {
        let cosPadding = MLXArray.ones(like: cosFreq[0..., ..<padSize])
        let sinPadding = MLXArray.zeros(like: sinFreq[0..., ..<padSize])
        cosFreq = MLX.concatenated([cosPadding, cosFreq], axis: -1)
        sinFreq = MLX.concatenated([sinPadding, sinFreq], axis: -1)
    }

    return (cosFreq, sinFreq)
}

// MARK: - Precompute Frequencies

/// Precompute RoPE frequencies.
public func precomputeFreqsCis(
    indicesGrid: MLXArray,
    dim: Int,
    theta: Float = 10000.0,
    maxPos: [Int]? = nil,
    useMiddleIndicesGrid: Bool = false,
    numAttentionHeads: Int = 32,
    ropeType: LTXRopeType = .interleaved,
    doublePrecision: Bool = false
) -> (MLXArray, MLXArray) {
    let maxPosResolved = maxPos ?? [20, 2048, 2048]

    // Keep positions in float32 for RoPE computation
    let gridF32 = indicesGrid.asType(.float32)

    // Generate frequency indices
    let indices = generateFreqGrid(
        positionalEmbeddingTheta: theta,
        positionalEmbeddingMaxPosCount: gridF32.shape[1],
        innerDim: dim
    )

    // Generate frequencies
    let freqs = generateFreqs(
        indices: indices,
        indicesGrid: gridF32,
        maxPos: maxPosResolved,
        useMiddleIndicesGrid: useMiddleIndicesGrid
    )

    // Prepare cos/sin based on rope type
    switch ropeType {
    case .split:
        let expectedFreqs = dim / 2
        let currentFreqs = freqs.shape[freqs.ndim - 1]
        let padSize = expectedFreqs - currentFreqs
        return splitFreqsCis(freqs: freqs, padSize: padSize, numAttentionHeads: numAttentionHeads)

    case .interleaved:
        let nElem = 2 * gridF32.shape[1]
        return interleavedFreqsCis(freqs: freqs, padSize: dim % nElem)

    case .twoD:
        fatalError("2D RoPE type is not supported")
    }
}
