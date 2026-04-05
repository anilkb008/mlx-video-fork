// LTXAttention.swift
// Attention module for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/attention.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Scaled Dot-Product Attention

/// Compute scaled dot-product attention.
func scaledDotProductAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    heads: Int,
    mask: MLXArray? = nil
) -> MLXArray {
    let b = q.shape[0]
    let qSeqLen = q.shape[1]
    let dim = q.shape[2]
    let kvSeqLen = k.shape[1]
    let dimHead = dim / heads

    // Reshape to (B, seqLen, heads, dimHead)
    var qr = q.reshaped([b, qSeqLen, heads, dimHead])
    var kr = k.reshaped([b, kvSeqLen, heads, dimHead])
    var vr = v.reshaped([b, kvSeqLen, heads, dimHead])

    // Transpose to (B, heads, seqLen, dimHead)
    qr = qr.swappedAxes(1, 2)
    kr = kr.swappedAxes(1, 2)
    vr = vr.swappedAxes(1, 2)

    // Handle mask dimensions
    var maskProcessed = mask
    if var m = maskProcessed {
        if m.ndim == 2 {
            m = m.expandedDimensions(axis: 0)
        }
        if m.ndim == 3 {
            m = m.expandedDimensions(axis: 1)
        }
        maskProcessed = m
    }

    let scale = 1.0 / Float(dimHead).squareRoot()

    var out = MLXFast.scaledDotProductAttention(
        queries: qr, keys: kr, values: vr, scale: scale, mask: maskProcessed
    )

    // Reshape back to (B, qSeqLen, heads * dimHead)
    out = out.swappedAxes(1, 2)
    out = out.reshaped([b, qSeqLen, heads * dimHead])

    return out
}

// MARK: - Attention Module

/// Multi-head attention with rotary position embeddings.
/// Supports both self-attention and cross-attention.
public class LTXAttention: Module {

    let ropeType: LTXRopeType
    let heads: Int
    let dimHead: Int

    let toQ: Linear
    let toK: Linear
    let toV: Linear

    let qNorm: RMSNorm
    let kNorm: RMSNorm

    let toOut: Linear

    // Per-head gating (LTX-2.3) - optional
    var toGateLogits: Linear?

    public init(
        queryDim: Int,
        contextDim: Int? = nil,
        heads: Int = 8,
        dimHead: Int = 64,
        normEps: Float = 1e-6,
        ropeType: LTXRopeType = .interleaved,
        hasGateLogits: Bool = false
    ) {
        self.ropeType = ropeType
        self.heads = heads
        self.dimHead = dimHead

        let innerDim = dimHead * heads
        let ctxDim = contextDim ?? queryDim

        self.toQ = Linear(queryDim, innerDim, bias: true)
        self.toK = Linear(ctxDim, innerDim, bias: true)
        self.toV = Linear(ctxDim, innerDim, bias: true)

        self.qNorm = RMSNorm(dimensions: innerDim, eps: normEps)
        self.kNorm = RMSNorm(dimensions: innerDim, eps: normEps)

        self.toOut = Linear(innerDim, queryDim, bias: true)

        if hasGateLogits {
            self.toGateLogits = Linear(queryDim, heads, bias: true)
        }
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Query input of shape (B, seqLen, queryDim)
    ///   - context: Context for cross-attention. If nil, uses x (self-attention)
    ///   - mask: Attention mask
    ///   - pe: Position embeddings for query (and key if kPe is nil)
    ///   - kPe: Position embeddings for key (optional)
    ///   - skipAttention: If true, bypass Q*K*V attention and use value projection only (STG)
    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray? = nil,
        mask: MLXArray? = nil,
        pe: (MLXArray, MLXArray)? = nil,
        kPe: (MLXArray, MLXArray)? = nil,
        skipAttention: Bool = false
    ) -> MLXArray {
        // Compute per-head gate early (from original input)
        var gate: MLXArray? = nil
        if let gateLinear = toGateLogits {
            gate = 2.0 * sigmoid(gateLinear(x))  // (B, seq, heads)
        }

        let ctx = context ?? x
        let v = toV(ctx)

        var out: MLXArray
        if skipAttention {
            // STG: bypass Q*K*V attention, use value projection only
            out = v
        } else {
            var q = toQ(x)
            var k = toK(ctx)

            q = qNorm(q)
            k = kNorm(k)

            if let pe = pe {
                q = applyRotaryEmb(q, pe, ropeType)
                let kPeToUse = kPe ?? pe
                k = applyRotaryEmb(k, kPeToUse, ropeType)
            }

            out = scaledDotProductAttention(q: q, k: k, v: v, heads: heads, mask: mask)
        }

        // Apply per-head gating
        if let g = gate {
            let b = out.shape[0]
            let seqLen = out.shape[1]
            out = out.reshaped([b, seqLen, heads, dimHead])
            out = out * g[.ellipsis, .newAxis]
            out = out.reshaped([b, seqLen, -1])
        }

        return toOut(out)
    }
}
