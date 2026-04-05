// LTXAdaLN.swift
// Adaptive Layer Normalization for LTX video diffusion model.
// Ported from mlx_video/models/ltx_2/adaln.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Timesteps (sinusoidal embedding)

/// Sinusoidal timestep embedding module.
public class Timesteps: Module {

    let numChannels: Int
    let flipSinToCos: Bool
    let downscaleFreqShift: Float

    public init(numChannels: Int, flipSinToCos: Bool = false, downscaleFreqShift: Float = 1.0) {
        self.numChannels = numChannels
        self.flipSinToCos = flipSinToCos
        self.downscaleFreqShift = downscaleFreqShift
    }

    public func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        return getTimestepEmbedding(
            timesteps,
            embeddingDim: numChannels,
            flipSinToCos: flipSinToCos,
            downscaleFreqShift: downscaleFreqShift
        )
    }
}

/// Create sinusoidal timestep embeddings.
public func getTimestepEmbedding(
    _ timesteps: MLXArray,
    embeddingDim: Int,
    flipSinToCos: Bool = false,
    downscaleFreqShift: Float = 1.0,
    scale: Float = 1.0,
    maxPeriod: Int = 10000
) -> MLXArray {
    let halfDim = embeddingDim / 2
    var exponent = -log(Float(maxPeriod)) * MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent / Float(halfDim - Int(downscaleFreqShift))

    var emb = MLX.exp(exponent)
    emb = (timesteps[0..., .newAxis].asType(.float32) * scale) * emb[.newAxis, 0...]

    let result: MLXArray
    if flipSinToCos {
        result = MLX.concatenated([MLX.cos(emb), MLX.sin(emb)], axis: -1)
    } else {
        result = MLX.concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
    }

    // Zero pad if odd embedding dimension
    if embeddingDim % 2 == 1 {
        return MLX.padded(result, widths: [.init(0, 0), .init(0, 1)])
    }

    return result
}

// MARK: - TimestepEmbedding

/// MLP for timestep embedding.
public class TimestepEmbedding: Module {

    let linear1: Linear
    let act: SiLU
    let linear2: Linear

    public init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        let outDimResolved = outDim ?? timeEmbedDim
        self.linear1 = Linear(inChannels, timeEmbedDim)
        self.act = SiLU()
        self.linear2 = Linear(timeEmbedDim, outDimResolved)
    }

    public func callAsFunction(_ sample: MLXArray) -> MLXArray {
        var x = linear1(sample)
        x = act(x)
        x = linear2(x)
        return x
    }
}

// MARK: - PixArtAlphaCombinedTimestepSizeEmbeddings

/// Combined timestep and size embeddings (PixArt-Alpha style).
public class PixArtAlphaCombinedTimestepSizeEmbeddings: Module {

    let embeddingDim: Int
    let sizeEmbDim: Int
    let useAdditionalConditions: Bool
    let timeProj: Timesteps
    let timestepEmbedder: TimestepEmbedding

    public init(
        embeddingDim: Int,
        sizeEmbDim: Int = 0,
        useAdditionalConditions: Bool = false,
        timestepProjDim: Int = 256
    ) {
        self.embeddingDim = embeddingDim
        self.sizeEmbDim = sizeEmbDim
        self.useAdditionalConditions = useAdditionalConditions

        self.timeProj = Timesteps(
            numChannels: timestepProjDim,
            flipSinToCos: true,
            downscaleFreqShift: 0
        )
        self.timestepEmbedder = TimestepEmbedding(
            inChannels: timestepProjDim,
            timeEmbedDim: embeddingDim,
            outDim: embeddingDim
        )
    }

    public func callAsFunction(
        _ timestep: MLXArray,
        batchSize: Int? = nil,
        hiddenDtype: DType? = nil
    ) -> MLXArray {
        var timestepsProj = timeProj(timestep)
        if let dtype = hiddenDtype {
            timestepsProj = timestepsProj.asType(dtype)
        }
        let timestepsEmb = timestepEmbedder(timestepsProj)
        return timestepsEmb
    }
}

// MARK: - AdaLayerNormSingle

/// Adaptive Layer Normalization with timestep conditioning.
public class AdaLayerNormSingle: Module {

    let emb: PixArtAlphaCombinedTimestepSizeEmbeddings
    let silu: SiLU
    let linear: Linear

    public init(
        embeddingDim: Int,
        embeddingCoefficient: Int = 6,
        useAdditionalConditions: Bool = false
    ) {
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embeddingDim: embeddingDim,
            sizeEmbDim: useAdditionalConditions ? embeddingDim / 3 : 0,
            useAdditionalConditions: useAdditionalConditions
        )
        self.silu = SiLU()
        self.linear = Linear(embeddingDim, embeddingCoefficient * embeddingDim, bias: true)
    }

    public func callAsFunction(
        _ timestep: MLXArray,
        batchSize: Int? = nil,
        hiddenDtype: DType? = nil
    ) -> (MLXArray, MLXArray) {
        let embeddedTimestep = emb(timestep, batchSize: batchSize, hiddenDtype: hiddenDtype)
        let scaleShiftParams = linear(silu(embeddedTimestep))
        return (scaleShiftParams, embeddedTimestep)
    }
}
