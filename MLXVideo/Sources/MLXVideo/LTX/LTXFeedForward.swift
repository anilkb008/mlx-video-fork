// LTXFeedForward.swift
// Feed-forward network for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/feed_forward.py

import Foundation
import MLX
import MLXNN

// MARK: - FeedForward

/// Feed-forward network with GELU gating.
public class LTXFeedForward: Module {

    let projIn: Linear
    let act: GELU
    let projOut: Linear

    public init(dim: Int, dimOut: Int? = nil, mult: Int = 4, bias: Bool = true) {
        let outDim = dimOut ?? dim
        let innerDim = dim * mult

        self.projIn = Linear(dim, innerDim, bias: bias)
        self.act = GELU(approximation: .tanh)
        self.projOut = Linear(innerDim, outDim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = projIn(x)
        h = act(h)
        h = projOut(h)
        return h
    }
}
