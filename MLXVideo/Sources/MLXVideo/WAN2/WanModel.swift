// WanModel.swift - Main WAN diffusion model
// Ported from mlx_video/models/wan_2/wan_2.py

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Sinusoidal Embedding

/// Compute sinusoidal positional embeddings.
///
/// - Parameters:
///   - dim: Embedding dimension (must be even).
///   - position: Tensor of positions, 1D [L] or 2D [B, L].
/// - Returns: Embeddings of shape [L, dim] or [B, L, dim].
func sinusoidalEmbedding1D(dim: Int, position: MLXArray) -> MLXArray {
    precondition(dim % 2 == 0)
    let half = dim / 2
    let pos = position.asType(.float32)
    let invFreq = pow(Float(10000.0), -MLXArray(0..<half).asType(.float32) / Float(half))
    let sinusoid = pos.expandedDimensions(axis: -1) * invFreq  // [..., half]
    return concatenated([cos(sinusoid), sin(sinusoid)], axis: -1)
}

// MARK: - Head

/// Output projection head with learned modulation.
public class Head: Module {
    let outDim: Int
    let patchSize: (Int, Int, Int)

    @ModuleInfo public var norm: WanLayerNorm
    @ModuleInfo public var head: Linear
    public var modulation: MLXArray

    public init(dim: Int, outDim: Int, patchSize: (Int, Int, Int), eps: Float = 1e-6) {
        self.outDim = outDim
        self.patchSize = patchSize
        let projDim = patchSize.0 * patchSize.1 * patchSize.2 * outDim
        self._norm.wrappedValue = WanLayerNorm(dim: dim, eps: eps)
        self._head.wrappedValue = Linear(dim, projDim)
        self.modulation = (MLXRandom.normal([1, 2, dim]) * pow(Float(dim), -0.5)).asType(.float32)
    }

    public func callAsFunction(_ x: MLXArray, e: MLXArray) -> MLXArray {
        var eVar = e
        if eVar.ndim == 2 {
            eVar = eVar.expandedDimensions(axis: 1)  // [B, 1, dim]
        }
        // Compute modulation in float32
        let mod = modulation.expandedDimensions(axis: 1) + eVar.expandedDimensions(axis: 2)  // float32
        let e0 = mod[0..., 0..., 0, 0...]  // shift
        let e1 = mod[0..., 0..., 1, 0...]  // scale
        let xNorm = norm(x)
        let xMod = xNorm * (1 + e1) + e0
        return head(xMod)
    }
}

// MARK: - WanModel

/// Wan2 diffusion backbone for text-to-video generation.
public class WanModel: Module {
    public let config: WanModelConfig
    let dim: Int
    let numHeads: Int
    let outDim: Int
    let patchSize: (Int, Int, Int)
    let textLen: Int
    let freqDim: Int

    @ModuleInfo public var patchEmbeddingProj: Linear
    @ModuleInfo public var textEmbedding0: Linear
    let textEmbeddingAct: GELU
    @ModuleInfo public var textEmbedding1: Linear
    @ModuleInfo public var timeEmbedding0: Linear
    let timeEmbeddingAct: SiLU
    @ModuleInfo public var timeEmbedding1: Linear
    let timeProjectionAct: SiLU
    @ModuleInfo public var timeProjection: Linear
    @ModuleInfo public var blocks: [WanAttentionBlock]
    @ModuleInfo public var head: Head

    /// Precomputed RoPE frequencies (non-parameter).
    public var freqs: MLXArray

    /// Precomputed sinusoidal inv_freq for time embedding.
    var invFreq: MLXArray

    public init(config: WanModelConfig) {
        self.config = config
        self.dim = config.dim
        self.numHeads = config.numHeads
        self.outDim = config.outDim
        self.patchSize = config.patchSize
        self.textLen = config.textLen
        self.freqDim = config.freqDim

        // Patch embedding: Conv3d implemented as reshaped linear
        let patchDim = config.inDim * config.patchSize.0 * config.patchSize.1 * config.patchSize.2
        self._patchEmbeddingProj.wrappedValue = Linear(patchDim, config.dim)

        // Text embedding MLP
        self._textEmbedding0.wrappedValue = Linear(config.textDim, config.dim)
        self.textEmbeddingAct = GELU(approximation: .tanh)
        self._textEmbedding1.wrappedValue = Linear(config.dim, config.dim)

        // Time embedding MLP
        self._timeEmbedding0.wrappedValue = Linear(config.freqDim, config.dim)
        self.timeEmbeddingAct = SiLU()
        self._timeEmbedding1.wrappedValue = Linear(config.dim, config.dim)

        // Time projection for modulation (6x dim)
        self.timeProjectionAct = SiLU()
        self._timeProjection.wrappedValue = Linear(config.dim, config.dim * 6)

        // Transformer blocks
        self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
            WanAttentionBlock(
                dim: config.dim,
                ffnDim: config.ffnDim,
                numHeads: config.numHeads,
                windowSize: config.windowSize,
                qkNorm: config.qkNorm,
                crossAttnNorm: config.crossAttnNorm,
                eps: config.eps
            )
        }

        // Output head
        self._head.wrappedValue = Head(
            dim: config.dim, outDim: config.outDim,
            patchSize: config.patchSize, eps: config.eps
        )

        // Precompute RoPE frequencies
        let d = config.dim / config.numHeads
        self.freqs = concatenated([
            ropeParams(maxSeqLen: 1024, dim: d - 4 * (d / 6)),
            ropeParams(maxSeqLen: 1024, dim: 2 * (d / 6)),
            ropeParams(maxSeqLen: 1024, dim: 2 * (d / 6)),
        ], axis: 1)

        // Precompute sinusoidal inv_freq for time embedding
        let half = config.freqDim / 2
        var invFreqData = [Float](repeating: 0, count: half)
        for i in 0..<half {
            invFreqData[i] = Float(Foundation.pow(10000.0, -Double(i) / Double(half)))
        }
        self.invFreq = MLXArray(invFreqData)
    }

    // MARK: - Patchify / Unpatchify

    /// Convert video tensor to patch embeddings.
    ///
    /// - Parameter x: Video latent [C, F, H, W]
    /// - Returns: (patches, gridSize): patches [1, L, dim], gridSize (F', H', W')
    func patchify(_ x: MLXArray) -> (MLXArray, (Int, Int, Int)) {
        let c = x.dim(0), f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let pt = patchSize.0, ph = patchSize.1, pw = patchSize.2

        let fOut = f / pt
        let hOut = h / ph
        let wOut = w / pw

        // Reshape: [C, F, H, W] -> [F', H', W', C, pt, ph, pw] -> [F'*H'*W', C*pt*ph*pw]
        var xR = x.reshaped(c, fOut, pt, hOut, ph, wOut, pw)
        xR = xR.transposed(1, 3, 5, 0, 2, 4, 6)  // [F', H', W', C, pt, ph, pw]
        xR = xR.reshaped(fOut * hOut * wOut, -1)    // [L, C*pt*ph*pw]

        // Project and cast to model dtype
        var patches = patchEmbeddingProj(xR)
        patches = patches.asType(linearDType(patchEmbeddingProj))
        patches = patches.expandedDimensions(axis: 0)  // [1, L, dim]

        return (patches, (fOut, hOut, wOut))
    }

    /// Reconstruct video from patch embeddings.
    ///
    /// - Parameters:
    ///   - x: [B, L, outDim * prod(patchSize)]
    ///   - gridSizes: List of (F', H', W') per batch element
    /// - Returns: List of tensors [C, F, H, W]
    public func unpatchify(_ x: MLXArray, gridSizes: [(Int, Int, Int)]) -> [MLXArray] {
        let c = outDim
        let pt = patchSize.0, ph = patchSize.1, pw = patchSize.2
        var out = [MLXArray]()
        for (i, (f, h, w)) in gridSizes.enumerated() {
            let seqLen = f * h * w
            var u = x[i, ..<seqLen]  // [L, outDim * pt * ph * pw]
            u = u.reshaped(f, h, w, pt, ph, pw, c)
            // Rearrange: [F', H', W', pt, ph, pw, C] -> [C, F'*pt, H'*ph, W'*pw]
            u = u.transposed(6, 0, 3, 1, 4, 2, 5)  // [C, F', pt, H', ph, W', pw]
            u = u.reshaped(c, f * pt, h * ph, w * pw)
            out.append(u)
        }
        return out
    }

    // MARK: - Text Embedding

    /// Precompute text embeddings (call once, reuse across steps).
    ///
    /// - Parameter context: List of text embeddings [L_text, textDim]
    /// - Returns: Embedded context [B, textLen, dim] in model dtype
    public func embedText(_ context: [MLXArray]) -> MLXArray {
        let modelDtype = linearDType(patchEmbeddingProj)
        var contextPadded = [MLXArray]()
        for ctx in context {
            var padded = ctx
            let padLen = textLen - ctx.dim(0)
            if padLen > 0 {
                let padding = MLXArray.zeros([padLen, ctx.dim(1)]).asType(ctx.dtype)
                padded = concatenated([padded, padding], axis: 0)
            }
            contextPadded.append(padded)
        }
        let contextBatch = stacked(contextPadded)  // [B, textLen, textDim]
        let embedded = textEmbedding1(textEmbeddingAct(textEmbedding0(contextBatch)))
        return embedded.asType(modelDtype)
    }

    // MARK: - Cross KV Cache

    /// Pre-compute cross-attention K/V for all blocks.
    ///
    /// - Parameter context: Pre-embedded text [B, textLen, dim]
    /// - Returns: List of (k, v) tuples, one per block
    public func prepareCrossKV(_ context: MLXArray) -> [(MLXArray, MLXArray)] {
        blocks.map { $0.crossAttn.prepareKV(context: context) }
    }

    // MARK: - RoPE Precompute

    /// Pre-compute RoPE cos/sin for constant grid sizes.
    ///
    /// - Parameter gridSizes: List of (F, H, W) tuples per batch element
    /// - Returns: (cosF, sinF) precomputed frequency tensors
    public func prepareRope(_ gridSizes: [(Int, Int, Int)]) -> (MLXArray, MLXArray) {
        let wDtype = linearDType(patchEmbeddingProj)
        return ropePrecomputeCosSin(gridSizes: gridSizes, freqs: freqs, dtype: wDtype)
    }

    // MARK: - Forward Pass

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - xList: List of video latent tensors [C, F, H, W]
    ///   - t: Timestep tensor [B]
    ///   - context: Pre-embedded tensor from embedText() [B, textLen, dim]
    ///   - seqLen: Maximum sequence length for padding
    ///   - crossKVCaches: Optional list of (k, v) tuples from prepareCrossKV()
    ///   - y: Optional list of conditioning tensors for I2V [C_y, F, H, W]
    ///   - ropeCosSin: Optional precomputed (cos, sin) from prepareRope()
    /// - Returns: List of denoised tensors [C, F, H, W]
    public func callAsFunction(
        xList: [MLXArray],
        t: MLXArray,
        context: MLXArray,
        seqLen: Int,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil,
        y: [MLXArray]? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil
    ) -> [MLXArray] {
        let batchSize = xList.count

        // Detect identical inputs (CFG B=2)
        let allSame = batchSize > 1 && xList.dropFirst().allSatisfy { $0.shape == xList[0].shape }
            // Note: in Swift we can't do identity check on MLXArray easily, but shape check is a proxy

        // I2V: channel-concatenate conditioning y with noise x
        var xInput = xList
        if let yList = y {
            xInput = zip(xInput, yList).map { concatenated([$0, $1], axis: 0) }
        }

        var x: MLXArray
        var gridSizes: [(Int, Int, Int)]
        var seqLensList: [Int]

        // Patchify
        var patches = [(MLXArray, (Int, Int, Int))]()
        var patchArrays = [MLXArray]()
        gridSizes = []
        seqLensList = []

        for vid in xInput {
            let (p, gs) = patchify(vid)
            patches.append((p, gs))
            patchArrays.append(p)
            gridSizes.append(gs)
            seqLensList.append(p.dim(1))
        }

        // Pad and concatenate
        x = concatenated(
            patchArrays.map { p in
                if p.dim(1) < seqLen {
                    let padding = MLXArray.zeros([1, seqLen - p.dim(1), dim]).asType(p.dtype)
                    return concatenated([p, padding], axis: 1)
                }
                return p
            },
            axis: 0
        )  // [B, seqLen, dim]

        // Time embedding
        var tVar = t
        if tVar.ndim == 0 {
            tVar = tVar.expandedDimensions(axis: 0)
        }

        let sinusoid = tVar.expandedDimensions(axis: -1).asType(.float32) * invFreq
        let sinEmb = concatenated([cos(sinusoid), sin(sinusoid)], axis: -1)

        let e0: MLXArray
        if tVar.ndim == 1 {
            // Standard T2V: scalar timestep per batch element [B]
            let e = timeEmbedding1(timeEmbeddingAct(timeEmbedding0(sinEmb)))
            let eProj = timeProjection(timeProjectionAct(e))
            e0 = eProj.reshaped(batchSize, 1, 6, dim)
        } else {
            // I2V: per-token timesteps [B, L]
            let e = timeEmbedding1(timeEmbeddingAct(timeEmbedding0(sinEmb)))
            let eProj = timeProjection(timeProjectionAct(e))
            e0 = eProj.reshaped(batchSize, -1, 6, dim)
        }

        // Context: expand to batch size if needed
        var contextBatch = context
        if contextBatch.dim(0) == 1 && batchSize > 1 {
            contextBatch = broadcast(contextBatch, to: [batchSize, contextBatch.dim(1), contextBatch.dim(2)])
        }

        // Pre-compute attention mask from seqLens
        let wDtype = linearDType(patchEmbeddingProj)
        var attnMask: MLXArray? = nil
        if seqLensList.contains(where: { $0 < seqLen }) {
            attnMask = MLXArray.zeros([batchSize, 1, 1, seqLen]).asType(wDtype)
            for (i, sl) in seqLensList.enumerated() {
                if sl < seqLen {
                    let maskSlice = MLXArray.full([1, 1, 1, seqLen - sl], values: MLXArray(-1e9)).asType(wDtype)
                    let zeroSlice = MLXArray.zeros([1, 1, 1, sl]).asType(wDtype)
                    let row = concatenated([zeroSlice, maskSlice], axis: 3)
                    attnMask![i] = row[0]
                }
            }
        }

        // Run transformer blocks
        for (i, block) in blocks.enumerated() {
            let kv = crossKVCaches?[i]
            x = block(
                x,
                e: e0,
                seqLens: seqLensList,
                gridSizes: gridSizes,
                freqs: freqs,
                context: contextBatch,
                contextLens: nil,
                crossKVCache: kv,
                ropeCosSin: ropeCosSin,
                attnMask: attnMask
            )
        }

        // Output head
        let e: MLXArray
        if tVar.ndim == 1 {
            e = timeEmbedding1(timeEmbeddingAct(timeEmbedding0(sinEmb)))
        } else {
            e = timeEmbedding1(timeEmbeddingAct(timeEmbedding0(sinEmb)))
        }
        x = head(x, e: e)

        // Unpatchify
        let outputs = unpatchify(x, gridSizes: gridSizes)
        return outputs.map { $0.asType(.float32) }
    }
}
