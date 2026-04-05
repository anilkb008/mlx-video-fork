// WanTransformer.swift - Transformer block for Wan2
// Ported from mlx_video/models/wan_2/transformer.py

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - WanFFN

/// Gated feed-forward network with GELU(tanh) activation.
public class WanFFN: Module, UnaryLayer {
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear
    let act: GELU

    public init(dim: Int, ffnDim: Int) {
        self._fc1.wrappedValue = Linear(dim, ffnDim)
        self.act = GELU(approximation: .tanh)
        self._fc2.wrappedValue = Linear(ffnDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xW = x.asType(linearDType(fc1))
        return fc2(act(fc1(xW)))
    }
}

// MARK: - WanAttentionBlock

/// Wan transformer block with learned modulation, self-attn, cross-attn, and FFN.
public class WanAttentionBlock: Module {
    @ModuleInfo public var norm1: WanLayerNorm
    @ModuleInfo public var selfAttn: WanSelfAttention
    @ModuleInfo public var norm3: WanLayerNorm?
    @ModuleInfo public var crossAttn: WanCrossAttention
    @ModuleInfo public var norm2: WanLayerNorm
    @ModuleInfo public var ffn: WanFFN

    /// Learned modulation: 6 vectors for scale/shift/gate (kept in float32 for precision).
    public var modulation: MLXArray

    public init(
        dim: Int,
        ffnDim: Int,
        numHeads: Int,
        windowSize: (Int, Int) = (-1, -1),
        qkNorm: Bool = true,
        crossAttnNorm: Bool = false,
        eps: Float = 1e-6
    ) {
        // Self-attention
        self._norm1.wrappedValue = WanLayerNorm(dim: dim, eps: eps)
        self._selfAttn.wrappedValue = WanSelfAttention(
            dim: dim, numHeads: numHeads, windowSize: windowSize, qkNorm: qkNorm, eps: eps
        )

        // Cross-attention (with optional norm on context)
        self._norm3.wrappedValue = crossAttnNorm
            ? WanLayerNorm(dim: dim, eps: eps, elementwiseAffine: true)
            : nil
        self._crossAttn.wrappedValue = WanCrossAttention(
            dim: dim, numHeads: numHeads, qkNorm: qkNorm, eps: eps
        )

        // Feed-forward
        self._norm2.wrappedValue = WanLayerNorm(dim: dim, eps: eps)
        self._ffn.wrappedValue = WanFFN(dim: dim, ffnDim: ffnDim)

        // Learned modulation: 6 vectors for scale/shift/gate
        self.modulation = (MLXRandom.normal([1, 6, dim]) * pow(Float(dim), -0.5)).asType(.float32)
    }

    public func callAsFunction(
        _ x: MLXArray,
        e: MLXArray,
        seqLens: [Int],
        gridSizes: [(Int, Int, Int)],
        freqs: MLXArray,
        context: MLXArray,
        contextLens: [Int]? = nil,
        crossKVCache: (MLXArray, MLXArray)? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        // Modulation: compute in float32 for precision
        let mod = modulation + e  // float32
        let e0 = mod[0..., 0..., 0, 0...]  // shift for self-attn
        let e1 = mod[0..., 0..., 1, 0...]  // scale for self-attn
        let e2 = mod[0..., 0..., 2, 0...]  // gate for self-attn
        let e3 = mod[0..., 0..., 3, 0...]  // shift for ffn
        let e4 = mod[0..., 0..., 4, 0...]  // scale for ffn
        let e5 = mod[0..., 0..., 5, 0...]  // gate for ffn

        // Self-attention with modulation
        var xVar = x
        var xMod = norm1(xVar) * (1 + e1) + e0
        var y = selfAttn(
            xMod,
            seqLens: seqLens,
            gridSizes: gridSizes,
            freqs: freqs,
            ropeCosSin: ropeCosSin,
            attnMask: attnMask
        )
        xVar = xVar + y * e2

        // Cross-attention (no modulation, just norm)
        let xCross = norm3 != nil ? norm3!(xVar) : xVar
        xVar = xVar + crossAttn(xCross, context: context, contextLens: contextLens, kvCache: crossKVCache)

        // FFN with modulation
        xMod = norm2(xVar) * (1 + e4) + e3
        y = ffn(xMod)
        xVar = xVar + y * e5

        return xVar
    }
}
