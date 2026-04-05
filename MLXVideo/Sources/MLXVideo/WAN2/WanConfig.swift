// WanConfig.swift - Configuration for Wan T2V/I2V models (2.1 and 2.2)
// Ported from mlx_video/models/wan_2/config.py

import Foundation

// MARK: - Wan Model Configuration

public struct WanModelConfig: ModelConfig, Sendable {
    public var modelType: String
    public var modelVersion: String
    public var patchSize: (Int, Int, Int)
    public var textLen: Int
    public var inDim: Int
    public var dim: Int
    public var ffnDim: Int
    public var freqDim: Int
    public var textDim: Int
    public var outDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var windowSize: (Int, Int)
    public var qkNorm: Bool
    public var crossAttnNorm: Bool
    public var eps: Float

    // VAE
    public var vaeStride: (Int, Int, Int)
    public var vaeZDim: Int

    // Inference
    public var dualModel: Bool
    public var boundary: Float
    public var sampleShift: Float
    public var sampleSteps: Int
    public var sampleGuideScale: GuideScale
    public var numTrainTimesteps: Int
    public var sampleFps: Int
    public var frameNum: Int
    public var sampleNegPrompt: String

    // Resolution constraints
    public var maxArea: Int
    public var t5VocabSize: Int
    public var t5Dim: Int
    public var t5DimAttn: Int
    public var t5DimFfn: Int
    public var t5NumHeads: Int
    public var t5NumLayers: Int
    public var t5NumBuckets: Int

    public var headDim: Int {
        dim / numHeads
    }

    // MARK: - Guide Scale (single float or (low, high) pair)

    public enum GuideScale: Sendable, Equatable {
        case single(Float)
        case dual(Float, Float)

        public var low: Float {
            switch self {
            case .single(let v): return v
            case .dual(let low, _): return low
            }
        }

        public var high: Float {
            switch self {
            case .single(let v): return v
            case .dual(_, let high): return high
            }
        }

        public var isCFGDisabled: Bool {
            switch self {
            case .single(let v): return v <= 1.0
            case .dual(let a, let b): return a <= 1.0 && b <= 1.0
            }
        }
    }

    // MARK: - Defaults

    public init(
        modelType: String = "t2v",
        modelVersion: String = "2.2",
        patchSize: (Int, Int, Int) = (1, 2, 2),
        textLen: Int = 512,
        inDim: Int = 16,
        dim: Int = 5120,
        ffnDim: Int = 13824,
        freqDim: Int = 256,
        textDim: Int = 4096,
        outDim: Int = 16,
        numHeads: Int = 40,
        numLayers: Int = 40,
        windowSize: (Int, Int) = (-1, -1),
        qkNorm: Bool = true,
        crossAttnNorm: Bool = true,
        eps: Float = 1e-6,
        vaeStride: (Int, Int, Int) = (4, 8, 8),
        vaeZDim: Int = 16,
        dualModel: Bool = true,
        boundary: Float = 0.875,
        sampleShift: Float = 12.0,
        sampleSteps: Int = 40,
        sampleGuideScale: GuideScale = .dual(3.0, 4.0),
        numTrainTimesteps: Int = 1000,
        sampleFps: Int = 16,
        frameNum: Int = 81,
        sampleNegPrompt: String = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        maxArea: Int = 0,
        t5VocabSize: Int = 256384,
        t5Dim: Int = 4096,
        t5DimAttn: Int = 4096,
        t5DimFfn: Int = 10240,
        t5NumHeads: Int = 64,
        t5NumLayers: Int = 24,
        t5NumBuckets: Int = 32
    ) {
        self.modelType = modelType
        self.modelVersion = modelVersion
        self.patchSize = patchSize
        self.textLen = textLen
        self.inDim = inDim
        self.dim = dim
        self.ffnDim = ffnDim
        self.freqDim = freqDim
        self.textDim = textDim
        self.outDim = outDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.windowSize = windowSize
        self.qkNorm = qkNorm
        self.crossAttnNorm = crossAttnNorm
        self.eps = eps
        self.vaeStride = vaeStride
        self.vaeZDim = vaeZDim
        self.dualModel = dualModel
        self.boundary = boundary
        self.sampleShift = sampleShift
        self.sampleSteps = sampleSteps
        self.sampleGuideScale = sampleGuideScale
        self.numTrainTimesteps = numTrainTimesteps
        self.sampleFps = sampleFps
        self.frameNum = frameNum
        self.sampleNegPrompt = sampleNegPrompt
        self.maxArea = maxArea
        self.t5VocabSize = t5VocabSize
        self.t5Dim = t5Dim
        self.t5DimAttn = t5DimAttn
        self.t5DimFfn = t5DimFfn
        self.t5NumHeads = t5NumHeads
        self.t5NumLayers = t5NumLayers
        self.t5NumBuckets = t5NumBuckets
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelVersion = "model_version"
        case patchSize = "patch_size"
        case textLen = "text_len"
        case inDim = "in_dim"
        case dim
        case ffnDim = "ffn_dim"
        case freqDim = "freq_dim"
        case textDim = "text_dim"
        case outDim = "out_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case windowSize = "window_size"
        case qkNorm = "qk_norm"
        case crossAttnNorm = "cross_attn_norm"
        case eps
        case vaeStride = "vae_stride"
        case vaeZDim = "vae_z_dim"
        case dualModel = "dual_model"
        case boundary
        case sampleShift = "sample_shift"
        case sampleSteps = "sample_steps"
        case sampleGuideScale = "sample_guide_scale"
        case numTrainTimesteps = "num_train_timesteps"
        case sampleFps = "sample_fps"
        case frameNum = "frame_num"
        case sampleNegPrompt = "sample_neg_prompt"
        case maxArea = "max_area"
        case t5VocabSize = "t5_vocab_size"
        case t5Dim = "t5_dim"
        case t5DimAttn = "t5_dim_attn"
        case t5DimFfn = "t5_dim_ffn"
        case t5NumHeads = "t5_num_heads"
        case t5NumLayers = "t5_num_layers"
        case t5NumBuckets = "t5_num_buckets"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "t2v"
        modelVersion = try c.decodeIfPresent(String.self, forKey: .modelVersion) ?? "2.2"
        let ps = try c.decodeIfPresent([Int].self, forKey: .patchSize) ?? [1, 2, 2]
        patchSize = (ps[0], ps[1], ps[2])
        textLen = try c.decodeIfPresent(Int.self, forKey: .textLen) ?? 512
        inDim = try c.decodeIfPresent(Int.self, forKey: .inDim) ?? 16
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 5120
        ffnDim = try c.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 13824
        freqDim = try c.decodeIfPresent(Int.self, forKey: .freqDim) ?? 256
        textDim = try c.decodeIfPresent(Int.self, forKey: .textDim) ?? 4096
        outDim = try c.decodeIfPresent(Int.self, forKey: .outDim) ?? 16
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 40
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 40
        let ws = try c.decodeIfPresent([Int].self, forKey: .windowSize) ?? [-1, -1]
        windowSize = (ws[0], ws[1])
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? true
        crossAttnNorm = try c.decodeIfPresent(Bool.self, forKey: .crossAttnNorm) ?? true
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
        let vs = try c.decodeIfPresent([Int].self, forKey: .vaeStride) ?? [4, 8, 8]
        vaeStride = (vs[0], vs[1], vs[2])
        vaeZDim = try c.decodeIfPresent(Int.self, forKey: .vaeZDim) ?? 16
        dualModel = try c.decodeIfPresent(Bool.self, forKey: .dualModel) ?? true
        boundary = try c.decodeIfPresent(Float.self, forKey: .boundary) ?? 0.875
        sampleShift = try c.decodeIfPresent(Float.self, forKey: .sampleShift) ?? 12.0
        sampleSteps = try c.decodeIfPresent(Int.self, forKey: .sampleSteps) ?? 40
        // Guide scale can be a float or [float, float]
        if let arr = try? c.decode([Float].self, forKey: .sampleGuideScale) {
            sampleGuideScale = arr.count >= 2 ? .dual(arr[0], arr[1]) : .single(arr[0])
        } else if let val = try? c.decode(Float.self, forKey: .sampleGuideScale) {
            sampleGuideScale = .single(val)
        } else {
            sampleGuideScale = .dual(3.0, 4.0)
        }
        numTrainTimesteps = try c.decodeIfPresent(Int.self, forKey: .numTrainTimesteps) ?? 1000
        sampleFps = try c.decodeIfPresent(Int.self, forKey: .sampleFps) ?? 16
        frameNum = try c.decodeIfPresent(Int.self, forKey: .frameNum) ?? 81
        sampleNegPrompt = try c.decodeIfPresent(String.self, forKey: .sampleNegPrompt) ?? ""
        maxArea = try c.decodeIfPresent(Int.self, forKey: .maxArea) ?? 0
        t5VocabSize = try c.decodeIfPresent(Int.self, forKey: .t5VocabSize) ?? 256384
        t5Dim = try c.decodeIfPresent(Int.self, forKey: .t5Dim) ?? 4096
        t5DimAttn = try c.decodeIfPresent(Int.self, forKey: .t5DimAttn) ?? 4096
        t5DimFfn = try c.decodeIfPresent(Int.self, forKey: .t5DimFfn) ?? 10240
        t5NumHeads = try c.decodeIfPresent(Int.self, forKey: .t5NumHeads) ?? 64
        t5NumLayers = try c.decodeIfPresent(Int.self, forKey: .t5NumLayers) ?? 24
        t5NumBuckets = try c.decodeIfPresent(Int.self, forKey: .t5NumBuckets) ?? 32
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(modelVersion, forKey: .modelVersion)
        try c.encode([patchSize.0, patchSize.1, patchSize.2], forKey: .patchSize)
        try c.encode(textLen, forKey: .textLen)
        try c.encode(inDim, forKey: .inDim)
        try c.encode(dim, forKey: .dim)
        try c.encode(ffnDim, forKey: .ffnDim)
        try c.encode(freqDim, forKey: .freqDim)
        try c.encode(textDim, forKey: .textDim)
        try c.encode(outDim, forKey: .outDim)
        try c.encode(numHeads, forKey: .numHeads)
        try c.encode(numLayers, forKey: .numLayers)
        try c.encode([windowSize.0, windowSize.1], forKey: .windowSize)
        try c.encode(qkNorm, forKey: .qkNorm)
        try c.encode(crossAttnNorm, forKey: .crossAttnNorm)
        try c.encode(eps, forKey: .eps)
        try c.encode([vaeStride.0, vaeStride.1, vaeStride.2], forKey: .vaeStride)
        try c.encode(vaeZDim, forKey: .vaeZDim)
        try c.encode(dualModel, forKey: .dualModel)
        try c.encode(boundary, forKey: .boundary)
        try c.encode(sampleShift, forKey: .sampleShift)
        try c.encode(sampleSteps, forKey: .sampleSteps)
        switch sampleGuideScale {
        case .single(let v): try c.encode(v, forKey: .sampleGuideScale)
        case .dual(let a, let b): try c.encode([a, b], forKey: .sampleGuideScale)
        }
        try c.encode(numTrainTimesteps, forKey: .numTrainTimesteps)
        try c.encode(sampleFps, forKey: .sampleFps)
        try c.encode(frameNum, forKey: .frameNum)
        try c.encode(sampleNegPrompt, forKey: .sampleNegPrompt)
        try c.encode(maxArea, forKey: .maxArea)
        try c.encode(t5VocabSize, forKey: .t5VocabSize)
        try c.encode(t5Dim, forKey: .t5Dim)
        try c.encode(t5DimAttn, forKey: .t5DimAttn)
        try c.encode(t5DimFfn, forKey: .t5DimFfn)
        try c.encode(t5NumHeads, forKey: .t5NumHeads)
        try c.encode(t5NumLayers, forKey: .t5NumLayers)
        try c.encode(t5NumBuckets, forKey: .t5NumBuckets)
    }

    // MARK: - Presets

    /// Wan2.1 T2V 14B: single model, 40 layers, dim=5120.
    public static func wan21T2V14B() -> WanModelConfig {
        WanModelConfig(
            modelVersion: "2.1",
            dualModel: false,
            boundary: 0.0,
            sampleShift: 5.0,
            sampleSteps: 50,
            sampleGuideScale: .single(5.0)
        )
    }

    /// Wan2.1 T2V 1.3B: single model, 30 layers, dim=1536.
    public static func wan21T2V1_3B() -> WanModelConfig {
        WanModelConfig(
            modelVersion: "2.1",
            dim: 1536,
            ffnDim: 8960,
            numHeads: 12,
            numLayers: 30,
            dualModel: false,
            boundary: 0.0,
            sampleShift: 5.0,
            sampleSteps: 50,
            sampleGuideScale: .single(5.0)
        )
    }

    /// Wan2.2 T2V 14B: dual model, 40 layers, dim=5120 (default).
    public static func wan22T2V14B() -> WanModelConfig {
        WanModelConfig()
    }

    /// Wan2.2 I2V 14B: dual model, image-to-video, 40 layers, dim=5120.
    public static func wan22I2V14B() -> WanModelConfig {
        WanModelConfig(
            modelType: "i2v",
            inDim: 36,
            outDim: 16,
            dualModel: true,
            boundary: 0.900,
            sampleShift: 5.0,
            sampleGuideScale: .dual(3.5, 3.5),
            maxArea: 704 * 1280
        )
    }

    /// Wan2.2 TI2V 5B: text+image to video, 30 layers, dim=3072.
    public static func wan22TI2V5B() -> WanModelConfig {
        WanModelConfig(
            modelType: "ti2v",
            dim: 3072,
            ffnDim: 14336,
            inDim: 48,
            outDim: 48,
            numHeads: 24,
            numLayers: 30,
            vaeStride: (4, 16, 16),
            vaeZDim: 48,
            dualModel: false,
            boundary: 0.0,
            sampleShift: 5.0,
            sampleSteps: 40,
            sampleGuideScale: .single(5.0),
            sampleFps: 24,
            maxArea: 704 * 1280
        )
    }
}
