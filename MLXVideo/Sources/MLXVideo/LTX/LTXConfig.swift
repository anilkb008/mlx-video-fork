// LTXConfig.swift - Configuration types for LTX video diffusion model
// Ported from mlx_video/models/ltx_2/config.py

import Foundation
import MLX

// MARK: - Enums

public enum LTXModelType: String, Codable, Sendable {
    case audioVideo = "ltx av model"
    case videoOnly = "ltx video only model"
    case audioOnly = "ltx audio only model"

    public var isVideoEnabled: Bool {
        self == .audioVideo || self == .videoOnly
    }

    public var isAudioEnabled: Bool {
        self == .audioVideo || self == .audioOnly
    }
}

public enum LTXRopeType: String, Codable, Sendable {
    case interleaved = "interleaved"
    case split = "split"
    case twoD = "2d"
}

public enum LTXAttentionType: String, Codable, Sendable {
    case `default` = "default"
}

// MARK: - TransformerConfig

public struct TransformerConfig: Sendable {
    public let dim: Int
    public let heads: Int
    public let dHead: Int
    public let contextDim: Int

    public init(dim: Int, heads: Int, dHead: Int, contextDim: Int) {
        self.dim = dim
        self.heads = heads
        self.dHead = dHead
        self.contextDim = contextDim
    }
}

// MARK: - VideoVAEConfig

public struct VideoVAEConfig: Codable, Sendable {
    public var convolutionDimensions: Int
    public var inChannels: Int
    public var outChannels: Int
    public var latentChannels: Int
    public var patchSize: Int

    public init(
        convolutionDimensions: Int = 3,
        inChannels: Int = 3,
        outChannels: Int = 128,
        latentChannels: Int = 128,
        patchSize: Int = 4
    ) {
        self.convolutionDimensions = convolutionDimensions
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.latentChannels = latentChannels
        self.patchSize = patchSize
    }

    enum CodingKeys: String, CodingKey {
        case convolutionDimensions = "convolution_dimensions"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case latentChannels = "latent_channels"
        case patchSize = "patch_size"
    }
}

// MARK: - LTXModelConfig

public struct LTXModelConfig: Codable, Sendable {

    // Model type
    public var modelType: LTXModelType

    // Video transformer config
    public var numAttentionHeads: Int
    public var attentionHeadDim: Int
    public var inChannels: Int
    public var outChannels: Int
    public var numLayers: Int
    public var crossAttentionDim: Int
    public var captionChannels: Int

    // Audio transformer config
    public var audioNumAttentionHeads: Int
    public var audioAttentionHeadDim: Int
    public var audioInChannels: Int
    public var audioOutChannels: Int
    public var audioCrossAttentionDim: Int
    public var audioCaptionChannels: Int

    // Positional embedding config
    public var positionalEmbeddingTheta: Float
    public var positionalEmbeddingMaxPos: [Int]
    public var audioPositionalEmbeddingMaxPos: [Int]
    public var useMiddleIndicesGrid: Bool
    public var ropeType: LTXRopeType
    public var doublePrecisionRope: Bool

    // Timestep config
    public var timestepScaleMultiplier: Int
    public var avCaTimestepScaleMultiplier: Int

    // Normalization
    public var normEps: Float

    // Attention type
    public var attentionType: LTXAttentionType

    // LTX-2.3: prompt-conditioned adaptive layer norm
    public var hasPromptAdaln: Bool

    // VAE config
    public var vaeConfig: VideoVAEConfig?

    // Computed properties
    public var innerDim: Int {
        numAttentionHeads * attentionHeadDim
    }

    public var audioInnerDim: Int {
        audioNumAttentionHeads * audioAttentionHeadDim
    }

    public func videoConfig() -> TransformerConfig? {
        guard modelType.isVideoEnabled else { return nil }
        return TransformerConfig(
            dim: innerDim,
            heads: numAttentionHeads,
            dHead: attentionHeadDim,
            contextDim: crossAttentionDim
        )
    }

    public func audioConfig() -> TransformerConfig? {
        guard modelType.isAudioEnabled else { return nil }
        return TransformerConfig(
            dim: audioInnerDim,
            heads: audioNumAttentionHeads,
            dHead: audioAttentionHeadDim,
            contextDim: audioCrossAttentionDim
        )
    }

    public init(
        modelType: LTXModelType = .audioVideo,
        numAttentionHeads: Int = 32,
        attentionHeadDim: Int = 128,
        inChannels: Int = 128,
        outChannels: Int = 128,
        numLayers: Int = 48,
        crossAttentionDim: Int = 4096,
        captionChannels: Int = 3840,
        audioNumAttentionHeads: Int = 32,
        audioAttentionHeadDim: Int = 64,
        audioInChannels: Int = 128,
        audioOutChannels: Int = 128,
        audioCrossAttentionDim: Int = 2048,
        audioCaptionChannels: Int = 3840,
        positionalEmbeddingTheta: Float = 10000.0,
        positionalEmbeddingMaxPos: [Int]? = nil,
        audioPositionalEmbeddingMaxPos: [Int]? = nil,
        useMiddleIndicesGrid: Bool = true,
        ropeType: LTXRopeType = .interleaved,
        doublePrecisionRope: Bool = false,
        timestepScaleMultiplier: Int = 1000,
        avCaTimestepScaleMultiplier: Int = 1000,
        normEps: Float = 1e-6,
        attentionType: LTXAttentionType = .default,
        hasPromptAdaln: Bool = false,
        vaeConfig: VideoVAEConfig? = nil
    ) {
        self.modelType = modelType
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numLayers = numLayers
        self.crossAttentionDim = crossAttentionDim
        self.captionChannels = captionChannels
        self.audioNumAttentionHeads = audioNumAttentionHeads
        self.audioAttentionHeadDim = audioAttentionHeadDim
        self.audioInChannels = audioInChannels
        self.audioOutChannels = audioOutChannels
        self.audioCrossAttentionDim = audioCrossAttentionDim
        self.audioCaptionChannels = audioCaptionChannels
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos ?? [20, 2048, 2048]
        self.audioPositionalEmbeddingMaxPos = audioPositionalEmbeddingMaxPos ?? [20]
        self.useMiddleIndicesGrid = useMiddleIndicesGrid
        self.ropeType = ropeType
        self.doublePrecisionRope = hasPromptAdaln ? doublePrecisionRope : false
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.avCaTimestepScaleMultiplier = avCaTimestepScaleMultiplier
        self.normEps = normEps
        self.attentionType = attentionType
        self.hasPromptAdaln = hasPromptAdaln
        self.vaeConfig = vaeConfig
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case attentionHeadDim = "attention_head_dim"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case numLayers = "num_layers"
        case crossAttentionDim = "cross_attention_dim"
        case captionChannels = "caption_channels"
        case audioNumAttentionHeads = "audio_num_attention_heads"
        case audioAttentionHeadDim = "audio_attention_head_dim"
        case audioInChannels = "audio_in_channels"
        case audioOutChannels = "audio_out_channels"
        case audioCrossAttentionDim = "audio_cross_attention_dim"
        case audioCaptionChannels = "audio_caption_channels"
        case positionalEmbeddingTheta = "positional_embedding_theta"
        case positionalEmbeddingMaxPos = "positional_embedding_max_pos"
        case audioPositionalEmbeddingMaxPos = "audio_positional_embedding_max_pos"
        case useMiddleIndicesGrid = "use_middle_indices_grid"
        case ropeType = "rope_type"
        case doublePrecisionRope = "double_precision_rope"
        case timestepScaleMultiplier = "timestep_scale_multiplier"
        case avCaTimestepScaleMultiplier = "av_ca_timestep_scale_multiplier"
        case normEps = "norm_eps"
        case attentionType = "attention_type"
        case hasPromptAdaln = "has_prompt_adaln"
        case vaeConfig = "vae_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(LTXModelType.self, forKey: .modelType) ?? .audioVideo
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        attentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .attentionHeadDim) ?? 128
        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 128
        outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 128
        numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 48
        crossAttentionDim = try container.decodeIfPresent(Int.self, forKey: .crossAttentionDim) ?? 4096
        captionChannels = try container.decodeIfPresent(Int.self, forKey: .captionChannels) ?? 3840
        audioNumAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .audioNumAttentionHeads) ?? 32
        audioAttentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .audioAttentionHeadDim) ?? 64
        audioInChannels = try container.decodeIfPresent(Int.self, forKey: .audioInChannels) ?? 128
        audioOutChannels = try container.decodeIfPresent(Int.self, forKey: .audioOutChannels) ?? 128
        audioCrossAttentionDim = try container.decodeIfPresent(Int.self, forKey: .audioCrossAttentionDim) ?? 2048
        audioCaptionChannels = try container.decodeIfPresent(Int.self, forKey: .audioCaptionChannels) ?? 3840
        positionalEmbeddingTheta = try container.decodeIfPresent(Float.self, forKey: .positionalEmbeddingTheta) ?? 10000.0
        positionalEmbeddingMaxPos = try container.decodeIfPresent([Int].self, forKey: .positionalEmbeddingMaxPos) ?? [20, 2048, 2048]
        audioPositionalEmbeddingMaxPos = try container.decodeIfPresent([Int].self, forKey: .audioPositionalEmbeddingMaxPos) ?? [20]
        useMiddleIndicesGrid = try container.decodeIfPresent(Bool.self, forKey: .useMiddleIndicesGrid) ?? true
        ropeType = try container.decodeIfPresent(LTXRopeType.self, forKey: .ropeType) ?? .interleaved
        let rawDoublePrecision = try container.decodeIfPresent(Bool.self, forKey: .doublePrecisionRope) ?? false
        let rawHasPromptAdaln = try container.decodeIfPresent(Bool.self, forKey: .hasPromptAdaln) ?? false
        hasPromptAdaln = rawHasPromptAdaln
        doublePrecisionRope = rawHasPromptAdaln ? rawDoublePrecision : false
        timestepScaleMultiplier = try container.decodeIfPresent(Int.self, forKey: .timestepScaleMultiplier) ?? 1000
        avCaTimestepScaleMultiplier = try container.decodeIfPresent(Int.self, forKey: .avCaTimestepScaleMultiplier) ?? 1000
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-6
        attentionType = try container.decodeIfPresent(LTXAttentionType.self, forKey: .attentionType) ?? .default
        vaeConfig = try container.decodeIfPresent(VideoVAEConfig.self, forKey: .vaeConfig)
    }
}
