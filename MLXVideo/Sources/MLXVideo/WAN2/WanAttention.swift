// WanAttention.swift - Attention layers for Wan2
// Ported from mlx_video/models/wan_2/attention.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Utility

/// Get the compute dtype of a linear layer, handling QuantizedLinear.
func linearDType(_ layer: Linear) -> DType {
    return layer.weight.dtype
}

// MARK: - WanRMSNorm

/// RMS normalization with learnable scale.
public class WanRMSNorm: Module, UnaryLayer {
    let eps: Float
    public var weight: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - WanLayerNorm

/// LayerNorm computed in float32, with optional affine parameters.
public class WanLayerNorm: Module, UnaryLayer {
    let eps: Float
    let elementwiseAffine: Bool
    public var weight: MLXArray?
    public var bias: MLXArray?

    public init(dim: Int, eps: Float = 1e-6, elementwiseAffine: Bool = false) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        if elementwiseAffine {
            self.weight = MLXArray.ones([dim])
            self.bias = MLXArray.zeros([dim])
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if elementwiseAffine {
            return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
        } else {
            return MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
        }
    }
}

// MARK: - WanSelfAttention

/// Self-attention with QK normalization and 3-way factorized RoPE.
public class WanSelfAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let windowSize: (Int, Int)
    let scale: Float

    @ModuleInfo public var q: Linear
    @ModuleInfo public var k: Linear
    @ModuleInfo public var v: Linear
    @ModuleInfo public var o: Linear
    @ModuleInfo public var normQ: WanRMSNorm?
    @ModuleInfo public var normK: WanRMSNorm?

    public init(
        dim: Int,
        numHeads: Int,
        windowSize: (Int, Int) = (-1, -1),
        qkNorm: Bool = true,
        eps: Float = 1e-6
    ) {
        precondition(dim % numHeads == 0)
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.windowSize = windowSize
        self.scale = pow(Float(headDim), -0.5)

        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)

        self._normQ.wrappedValue = qkNorm ? WanRMSNorm(dim: dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? WanRMSNorm(dim: dim, eps: eps) : nil
    }

    public func callAsFunction(
        _ x: MLXArray,
        seqLens: [Int],
        gridSizes: [(Int, Int, Int)],
        freqs: MLXArray,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        let n = numHeads
        let d = headDim

        // Cast to compute dtype for efficient matmul
        let wDtype = linearDType(q)
        let xW = x.asType(wDtype)

        var qProj = q(xW)
        var kProj = k(xW)
        if let nq = normQ {
            qProj = nq(qProj)
        }
        if let nk = normK {
            kProj = nk(kProj)
        }

        var qArr = qProj.reshaped(b, s, n, d)
        var kArr = kProj.reshaped(b, s, n, d)
        var vArr = v(xW).reshaped(b, s, n, d)

        // RoPE in float32 for precision
        qArr = ropeApply(qArr.asType(.float32), gridSizes: gridSizes, freqs: freqs, precomputedCosSin: ropeCosSin)
        kArr = ropeApply(kArr.asType(.float32), gridSizes: gridSizes, freqs: freqs, precomputedCosSin: ropeCosSin)

        // Cast back to weight dtype for efficient attention
        qArr = qArr.asType(wDtype).transposed(0, 2, 1, 3)
        kArr = kArr.asType(wDtype).transposed(0, 2, 1, 3)
        vArr = vArr.transposed(0, 2, 1, 3)

        // Use precomputed mask or build from seqLens
        var mask = attnMask
        if mask == nil && seqLens.contains(where: { $0 < s }) {
            mask = MLXArray.zeros([b, 1, 1, s]).asType(qArr.dtype)
            for (i, sl) in seqLens.enumerated() {
                // Note: MLX Swift array mutation is limited; we build mask differently
                if sl < s {
                    let maskSlice = MLXArray.full([1, 1, 1, s - sl], values: MLXArray(-1e9)).asType(qArr.dtype)
                    let zeroSlice = MLXArray.zeros([1, 1, 1, sl]).asType(qArr.dtype)
                    let row = concatenated([zeroSlice, maskSlice], axis: 3)
                    // For simplicity, rebuild full mask
                    mask![i] = row[0]
                }
            }
        }

        // Memory-efficient scaled dot-product attention [B, N, L, D]
        let out: MLXArray
        if let m = mask {
            out = MLXFast.scaledDotProductAttention(
                queries: qArr, keys: kArr, values: vArr, scale: scale, mask: m
            )
        } else {
            out = MLXFast.scaledDotProductAttention(
                queries: qArr, keys: kArr, values: vArr, scale: scale
            )
        }

        let outReshaped = out.transposed(0, 2, 1, 3).reshaped(b, s, -1)
        return o(outReshaped)
    }
}

// MARK: - WanCrossAttention

/// Cross-attention: Q from hidden states, K/V from text context.
public class WanCrossAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo public var q: Linear
    @ModuleInfo public var k: Linear
    @ModuleInfo public var v: Linear
    @ModuleInfo public var o: Linear
    @ModuleInfo public var normQ: WanRMSNorm?
    @ModuleInfo public var normK: WanRMSNorm?

    public init(
        dim: Int,
        numHeads: Int,
        qkNorm: Bool = true,
        eps: Float = 1e-6
    ) {
        precondition(dim % numHeads == 0)
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)

        self._normQ.wrappedValue = qkNorm ? WanRMSNorm(dim: dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? WanRMSNorm(dim: dim, eps: eps) : nil
    }

    /// Pre-compute K and V projections for caching.
    ///
    /// - Parameter context: [B, L_ctx, dim]
    /// - Returns: (k, v) each [B, N, L_ctx, D] ready for attention.
    public func prepareKV(context: MLXArray) -> (MLXArray, MLXArray) {
        let b = context.dim(0)
        let n = numHeads
        let d = headDim

        let wDtype = linearDType(k)
        let ctx = context.asType(wDtype)
        var kProj = k(ctx)
        if let nk = normK {
            kProj = nk(kProj)
        }
        kProj = kProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        let vProj = v(ctx).reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        return (kProj, vProj)
    }

    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        contextLens: [Int]? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let n = numHeads
        let d = headDim

        let wDtype = linearDType(q)
        var qProj = q(x.asType(wDtype))
        if let nq = normQ {
            qProj = nq(qProj)
        }
        qProj = qProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)

        let kArr: MLXArray
        let vArr: MLXArray
        if let (cachedK, cachedV) = kvCache {
            kArr = cachedK
            vArr = cachedV
        } else {
            let ctx = context.asType(wDtype)
            var kProj = k(ctx)
            if let nk = normK {
                kProj = nk(kProj)
            }
            kArr = kProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
            vArr = v(ctx).reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        }

        // Optional context masking
        var mask: MLXArray? = nil
        if let cls = contextLens {
            let ctxLen = kArr.dim(2)
            mask = MLXArray.zeros([b, 1, 1, ctxLen]).asType(qProj.dtype)
            for (i, cl) in cls.enumerated() {
                if cl < ctxLen {
                    let maskSlice = MLXArray.full([1, 1, 1, ctxLen - cl], values: MLXArray(-1e9)).asType(qProj.dtype)
                    let zeroSlice = MLXArray.zeros([1, 1, 1, cl]).asType(qProj.dtype)
                    let row = concatenated([zeroSlice, maskSlice], axis: 3)
                    mask![i] = row[0]
                }
            }
        }

        let out: MLXArray
        if let m = mask {
            out = MLXFast.scaledDotProductAttention(
                queries: qProj, keys: kArr, values: vArr, scale: scale, mask: m
            )
        } else {
            out = MLXFast.scaledDotProductAttention(
                queries: qProj, keys: kArr, values: vArr, scale: scale
            )
        }

        let outReshaped = out.transposed(0, 2, 1, 3).reshaped(b, -1, n * d)
        return o(outReshaped)
    }
}
