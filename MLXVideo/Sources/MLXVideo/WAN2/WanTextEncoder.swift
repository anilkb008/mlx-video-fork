// WanTextEncoder.swift - T5 Text Encoder (UMT5-XXL) for Wan2
// Ported from mlx_video/models/wan_2/text_encoder.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - T5LayerNorm

/// RMS-based layer normalization (T5 style).
public class T5LayerNorm: Module, UnaryLayer {
    let eps: Float
    public var weight: MLXArray

    public init(dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - T5RelativeEmbedding

/// T5-style relative position bias with bucketing.
public class T5RelativeEmbedding: Module {
    let numBuckets: Int
    let numHeads: Int
    let bidirectional: Bool
    let maxDist: Int

    @ModuleInfo public var embedding: Embedding

    public init(
        numBuckets: Int,
        numHeads: Int,
        bidirectional: Bool = true,
        maxDist: Int = 128
    ) {
        self.numBuckets = numBuckets
        self.numHeads = numHeads
        self.bidirectional = bidirectional
        self.maxDist = maxDist
        self._embedding.wrappedValue = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
    }

    private func relativePositionBucket(_ relPos: MLXArray) -> MLXArray {
        if bidirectional {
            let numBucketsHalf = numBuckets / 2
            let relBucketsBase = (relPos .> 0).asType(.int32) * Int32(numBucketsHalf)
            let relPosAbs = abs(relPos)

            let maxExact = numBucketsHalf / 2
            let isSmall = relPosAbs .< Int32(maxExact)

            let relPosF = relPosAbs.asType(.float32)
            var relPosLarge = Float(maxExact) + (
                MLX.log(relPosF / Float(maxExact))
                / Foundation.log(Float(maxDist) / Float(maxExact))
                * Float(numBucketsHalf - maxExact)
            )
            relPosLarge = minimum(
                relPosLarge.asType(.int32),
                MLXArray.full(relPosLarge.shape, values: Int32(numBucketsHalf - 1))
            ).asType(.int32)

            return relBucketsBase + which(isSmall, relPosAbs.asType(.int32), relPosLarge.asType(.int32))
        } else {
            let relBuckets = MLXArray.zeros(like: relPos).asType(.int32)
            let relPosAbs = maximum(-relPos, MLXArray.zeros(like: relPos))

            let maxExact = numBuckets / 2
            let isSmall = relPosAbs .< Int32(maxExact)

            let relPosF = relPosAbs.asType(.float32)
            var relPosLarge = Float(maxExact) + (
                MLX.log(relPosF / Float(maxExact))
                / Foundation.log(Float(maxDist) / Float(maxExact))
                * Float(numBuckets - maxExact)
            )
            relPosLarge = minimum(
                relPosLarge.asType(.int32),
                MLXArray.full(relPosLarge.shape, values: Int32(numBuckets - 1))
            ).asType(.int32)

            return relBuckets + which(isSmall, relPosAbs.asType(.int32), relPosLarge.asType(.int32))
        }
    }

    public func callAsFunction(lq: Int, lk: Int) -> MLXArray {
        let positionsK = MLXArray(0..<lk).expandedDimensions(axis: 0)  // [1, lk]
        let positionsQ = MLXArray(0..<lq).expandedDimensions(axis: 1)  // [lq, 1]
        let relPos = positionsK - positionsQ  // [lq, lk]

        let buckets = relativePositionBucket(relPos)
        let embeds = embedding(buckets)  // [lq, lk, numHeads]
        return embeds.transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, N, lq, lk]
    }
}

// MARK: - T5Attention

/// T5-style multi-head attention (no scaling).
public class T5Attention: Module {
    let dim: Int
    let dimAttn: Int
    let numHeads: Int
    let headDim: Int

    @ModuleInfo public var q: Linear
    @ModuleInfo public var k: Linear
    @ModuleInfo public var v: Linear
    @ModuleInfo public var o: Linear

    public init(dim: Int, dimAttn: Int, numHeads: Int) {
        precondition(dimAttn % numHeads == 0)
        self.dim = dim
        self.dimAttn = dimAttn
        self.numHeads = numHeads
        self.headDim = dimAttn / numHeads

        self._q.wrappedValue = Linear(dim, dimAttn, bias: false)
        self._k.wrappedValue = Linear(dim, dimAttn, bias: false)
        self._v.wrappedValue = Linear(dim, dimAttn, bias: false)
        self._o.wrappedValue = Linear(dimAttn, dim, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray? = nil,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let ctx = context ?? x
        let b = x.dim(0), n = numHeads, c = headDim

        let qProj = q(x).reshaped(b, -1, n, c)    // [B, Lq, N, C]
        let kProj = k(ctx).reshaped(b, -1, n, c)   // [B, Lk, N, C]
        let vProj = v(ctx).reshaped(b, -1, n, c)

        // T5 uses no scaling -- compute attention manually with float32 softmax
        let qT = qProj.transposed(0, 2, 1, 3)  // [B, N, Lq, C]
        let kT = kProj.transposed(0, 2, 1, 3)
        let vT = vProj.transposed(0, 2, 1, 3)

        // QK^T (no scaling) in float32 for precision
        var attn = matmul(qT.asType(.float32), kT.asType(.float32).transposed(0, 1, 3, 2))

        // Add position bias
        if let pb = posBias {
            attn = attn + pb.asType(.float32)
        }

        // Apply attention mask
        if let m = mask {
            var maskExpanded = m
            if m.ndim == 2 {
                maskExpanded = m.expandedDimensions(axes: [1, 2])  // [B, 1, 1, Lk]
            } else if m.ndim == 3 {
                maskExpanded = m.expandedDimensions(axis: 1)  // [B, 1, Lq, Lk]
            }
            let additiveMask = which(maskExpanded .== 0, MLXArray(-3.389e38), MLXArray(0.0)).asType(.float32)
            attn = attn + additiveMask
        }

        // Softmax in float32, then cast back
        let attnWeights = softmax(attn, axis: -1).asType(qT.dtype)

        // Attention @ V
        let out = matmul(attnWeights, vT).transposed(0, 2, 1, 3).reshaped(b, -1, n * c)
        return o(out)
    }
}

// MARK: - T5FeedForward

/// Gated feed-forward: gate(x) * fc1(x) -> fc2.
public class T5FeedForward: Module {
    @ModuleInfo public var gateProj: Linear
    let gateAct: GELU
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear

    public init(dim: Int, dimFfn: Int) {
        self._gateProj.wrappedValue = Linear(dim, dimFfn, bias: false)
        self.gateAct = GELU(approximation: .tanh)
        self._fc1.wrappedValue = Linear(dim, dimFfn, bias: false)
        self._fc2.wrappedValue = Linear(dimFfn, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(fc1(x) * gateAct(gateProj(x)))
    }
}

// MARK: - T5SelfAttentionBlock

/// T5 encoder block: self-attention + FFN.
public class T5SelfAttentionBlock: Module {
    let sharedPos: Bool

    @ModuleInfo public var norm1: T5LayerNorm
    @ModuleInfo public var attn: T5Attention
    @ModuleInfo public var norm2: T5LayerNorm
    @ModuleInfo public var ffn: T5FeedForward
    @ModuleInfo public var posEmbedding: T5RelativeEmbedding?

    public init(
        dim: Int,
        dimAttn: Int,
        dimFfn: Int,
        numHeads: Int,
        numBuckets: Int,
        sharedPos: Bool = true
    ) {
        self.sharedPos = sharedPos
        self._norm1.wrappedValue = T5LayerNorm(dim: dim)
        self._attn.wrappedValue = T5Attention(dim: dim, dimAttn: dimAttn, numHeads: numHeads)
        self._norm2.wrappedValue = T5LayerNorm(dim: dim)
        self._ffn.wrappedValue = T5FeedForward(dim: dim, dimFfn: dimFfn)
        self._posEmbedding.wrappedValue = sharedPos
            ? nil
            : T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let e: MLXArray?
        if sharedPos {
            e = posBias
        } else {
            e = posEmbedding?.callAsFunction(lq: x.dim(1), lk: x.dim(1))
        }
        var xVar = x + attn(norm1(x), mask: mask, posBias: e)
        xVar = xVar + ffn(norm2(xVar))
        return xVar
    }
}

// MARK: - T5Encoder

/// T5 Encoder (UMT5-XXL configuration).
public class T5Encoder: Module {
    public let dim: Int

    @ModuleInfo public var tokenEmbedding: Embedding
    @ModuleInfo public var posEmbedding: T5RelativeEmbedding?
    @ModuleInfo public var blocks: [T5SelfAttentionBlock]
    @ModuleInfo public var norm: T5LayerNorm

    public init(
        vocabSize: Int = 256384,
        dim: Int = 4096,
        dimAttn: Int = 4096,
        dimFfn: Int = 10240,
        numHeads: Int = 64,
        numLayers: Int = 24,
        numBuckets: Int = 32,
        sharedPos: Bool = false
    ) {
        self.dim = dim

        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._posEmbedding.wrappedValue = sharedPos
            ? T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
            : nil
        self._blocks.wrappedValue = (0..<numLayers).map { _ in
            T5SelfAttentionBlock(
                dim: dim, dimAttn: dimAttn, dimFfn: dimFfn,
                numHeads: numHeads, numBuckets: numBuckets, sharedPos: sharedPos
            )
        }
        self._norm.wrappedValue = T5LayerNorm(dim: dim)
    }

    /// Run T5 encoder.
    ///
    /// - Parameters:
    ///   - ids: Token IDs [B, L]
    ///   - mask: Attention mask [B, L]
    /// - Returns: Hidden states [B, L, dim]
    public func callAsFunction(ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = tokenEmbedding(ids)

        let e = posEmbedding?.callAsFunction(lq: x.dim(1), lk: x.dim(1))
        for block in blocks {
            x = block(x, mask: mask, posBias: e)
        }

        x = norm(x)
        return x
    }
}
