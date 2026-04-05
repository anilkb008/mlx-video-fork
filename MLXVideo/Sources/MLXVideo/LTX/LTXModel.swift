// LTXModel.swift
// Main LTX model for video diffusion.
// Ported from mlx_video/models/ltx_2/ltx_2.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - PixArtAlphaTextProjection

/// Text projection layer (PixArt-Alpha style).
public class PixArtAlphaTextProjection: Module {

    let linear1: Linear
    let act: GELU
    let linear2: Linear

    public init(inFeatures: Int, hiddenSize: Int, outFeatures: Int? = nil, bias: Bool = true) {
        let outDim = outFeatures ?? hiddenSize
        self.linear1 = Linear(inFeatures, hiddenSize, bias: bias)
        self.act = GELU(approximation: .tanh)
        self.linear2 = Linear(hiddenSize, outDim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = act(h)
        h = linear2(h)
        return h
    }
}

// MARK: - TransformerArgsPreprocessor

/// Preprocesses modality inputs into TransformerArgs for the transformer blocks.
public class TransformerArgsPreprocessor {

    let patchifyProj: Linear
    let adaln: AdaLayerNormSingle
    let captionProjection: PixArtAlphaTextProjection?
    let promptAdaln: AdaLayerNormSingle?
    let innerDim: Int
    let maxPos: [Int]
    let numAttentionHeads: Int
    let useMiddleIndicesGrid: Bool
    let timestepScaleMultiplier: Int
    let positionalEmbeddingTheta: Float
    let ropeType: LTXRopeType
    let doublePrecisionRope: Bool

    public init(
        patchifyProj: Linear,
        adaln: AdaLayerNormSingle,
        captionProjection: PixArtAlphaTextProjection?,
        innerDim: Int,
        maxPos: [Int],
        numAttentionHeads: Int,
        useMiddleIndicesGrid: Bool,
        timestepScaleMultiplier: Int,
        positionalEmbeddingTheta: Float,
        ropeType: LTXRopeType,
        doublePrecisionRope: Bool = false,
        promptAdaln: AdaLayerNormSingle? = nil
    ) {
        self.patchifyProj = patchifyProj
        self.adaln = adaln
        self.captionProjection = captionProjection
        self.promptAdaln = promptAdaln
        self.innerDim = innerDim
        self.maxPos = maxPos
        self.numAttentionHeads = numAttentionHeads
        self.useMiddleIndicesGrid = useMiddleIndicesGrid
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.ropeType = ropeType
        self.doublePrecisionRope = doublePrecisionRope
    }

    func prepareTimestep(
        _ timestep: MLXArray,
        batchSize: Int,
        hiddenDtype: DType? = nil
    ) -> (MLXArray, MLXArray) {
        let scaled = timestep * Float(timestepScaleMultiplier)
        let (timestepEmb, embeddedTimestep) = adaln(
            scaled.reshaped([-1]),
            hiddenDtype: hiddenDtype
        )

        let reshapedEmb = timestepEmb.reshaped([batchSize, -1, timestepEmb.shape[timestepEmb.ndim - 1]])
        let reshapedEmbedded = embeddedTimestep.reshaped([batchSize, -1, embeddedTimestep.shape[embeddedTimestep.ndim - 1]])

        return (reshapedEmb, reshapedEmbedded)
    }

    func prepareTimestepWithAdaln(
        _ adaln: AdaLayerNormSingle,
        timestep: MLXArray,
        batchSize: Int,
        hiddenDtype: DType? = nil
    ) -> (MLXArray, MLXArray) {
        let scaled = timestep * Float(timestepScaleMultiplier)
        let (timestepEmb, embeddedTimestep) = adaln(
            scaled.reshaped([-1]),
            hiddenDtype: hiddenDtype
        )
        let reshapedEmb = timestepEmb.reshaped([batchSize, -1, timestepEmb.shape[timestepEmb.ndim - 1]])
        let reshapedEmbedded = embeddedTimestep.reshaped([batchSize, -1, embeddedTimestep.shape[embeddedTimestep.ndim - 1]])
        return (reshapedEmb, reshapedEmbedded)
    }

    func prepareContext(
        _ context: MLXArray,
        x: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        let batchSize = x.shape[0]
        var ctx = context
        if let proj = captionProjection {
            ctx = proj(ctx)
        }
        ctx = ctx.reshaped([batchSize, -1, x.shape[x.ndim - 1]])
        return (ctx, attentionMask)
    }

    func prepareAttentionMask(
        _ attentionMask: MLXArray?,
        xDtype: DType
    ) -> MLXArray? {
        guard let mask = attentionMask else { return nil }

        // Check if already float
        if mask.dtype == .float16 || mask.dtype == .float32 || mask.dtype == .bfloat16 {
            return mask
        }

        // Convert boolean/int mask to float mask: 0 -> -inf (masked), 1 -> 0 (not masked)
        var floatMask = (mask.asType(xDtype) - 1) * 1e9
        floatMask = floatMask.reshaped([mask.shape[0], 1, -1, mask.shape[mask.ndim - 1]])
        return floatMask
    }

    func preparePositionalEmbeddings(
        positions: MLXArray,
        innerDim: Int,
        maxPos: [Int],
        useMiddleIndicesGrid: Bool,
        numAttentionHeads: Int
    ) -> (MLXArray, MLXArray) {
        return precomputeFreqsCis(
            indicesGrid: positions,
            dim: innerDim,
            theta: positionalEmbeddingTheta,
            maxPos: maxPos,
            useMiddleIndicesGrid: useMiddleIndicesGrid,
            numAttentionHeads: numAttentionHeads,
            ropeType: ropeType,
            doublePrecision: doublePrecisionRope
        )
    }

    public func prepare(_ modality: Modality) -> TransformerArgs {
        let x = patchifyProj(modality.latent)
        let (timestep, embeddedTimestep) = prepareTimestep(
            modality.timesteps, batchSize: x.shape[0], hiddenDtype: x.dtype
        )
        let (context, attentionMask) = prepareContext(
            modality.context, x: x, attentionMask: modality.contextMask
        )
        let processedMask = prepareAttentionMask(attentionMask, xDtype: modality.latent.dtype)

        // Use precomputed positional embeddings if provided
        let pe: (MLXArray, MLXArray)
        if let precomputed = modality.positionalEmbeddings {
            pe = precomputed
        } else {
            pe = preparePositionalEmbeddings(
                positions: modality.positions,
                innerDim: innerDim,
                maxPos: maxPos,
                useMiddleIndicesGrid: useMiddleIndicesGrid,
                numAttentionHeads: numAttentionHeads
            )
        }

        // Prompt-conditioned timestep (LTX-2.3)
        var promptTimestep: MLXArray? = nil
        var promptEmbeddedTimestep: MLXArray? = nil
        if let pAdaln = promptAdaln, let sigma = modality.sigma {
            let result = prepareTimestepWithAdaln(
                pAdaln,
                timestep: sigma,
                batchSize: x.shape[0],
                hiddenDtype: x.dtype
            )
            promptTimestep = result.0
            promptEmbeddedTimestep = result.1
        }

        return TransformerArgs(
            x: x,
            context: context,
            contextMask: processedMask,
            timesteps: timestep,
            embeddedTimestep: embeddedTimestep,
            positionalEmbeddings: pe,
            crossPositionalEmbeddings: nil,
            crossScaleShiftTimestep: nil,
            crossGateTimestep: nil,
            enabled: modality.enabled,
            promptTimesteps: promptTimestep,
            promptEmbeddedTimestep: promptEmbeddedTimestep
        )
    }
}

// MARK: - LTXModel

/// Main LTX diffusion transformer model.
public class LTXModel: Module {

    public let config: LTXModelConfig
    public let modelType: LTXModelType
    public let useMiddleIndicesGrid: Bool
    public let ropeType: LTXRopeType
    public let timestepScaleMultiplier: Int
    public let positionalEmbeddingTheta: Float

    // Video components
    public var positionalEmbeddingMaxPos: [Int] = []
    public var numAttentionHeads: Int = 0
    public var innerDim: Int = 0

    var patchifyProj: Linear?
    var adalnSingle: AdaLayerNormSingle?
    var promptAdalnSingle: AdaLayerNormSingle?
    var captionProjection: PixArtAlphaTextProjection?
    var scaleShiftTable: MLXArray?
    var normOut: LayerNorm?
    var projOut: Linear?

    // Transformer blocks (keyed by index)
    var transformerBlocks: [String: BasicAVTransformerBlock] = [:]

    // Preprocessor
    var videoArgsPreprocessor: TransformerArgsPreprocessor?

    public init(_ config: LTXModelConfig) {
        self.config = config
        self.modelType = config.modelType
        self.useMiddleIndicesGrid = config.useMiddleIndicesGrid
        self.ropeType = config.ropeType
        self.timestepScaleMultiplier = config.timestepScaleMultiplier
        self.positionalEmbeddingTheta = config.positionalEmbeddingTheta

        if config.modelType.isVideoEnabled {
            self.positionalEmbeddingMaxPos = config.positionalEmbeddingMaxPos
            self.numAttentionHeads = config.numAttentionHeads
            self.innerDim = config.innerDim
            initVideo(config)
        }

        initPreprocessors(config)
        initTransformerBlocks(config)
    }

    private func initVideo(_ config: LTXModelConfig) {
        self.patchifyProj = Linear(config.inChannels, innerDim, bias: true)

        let adalnCoefficient = config.hasPromptAdaln ? 9 : 6
        self.adalnSingle = AdaLayerNormSingle(
            embeddingDim: innerDim, embeddingCoefficient: adalnCoefficient
        )

        if config.hasPromptAdaln {
            self.promptAdalnSingle = AdaLayerNormSingle(
                embeddingDim: innerDim, embeddingCoefficient: 2
            )
        } else {
            self.captionProjection = PixArtAlphaTextProjection(
                inFeatures: config.captionChannels, hiddenSize: innerDim
            )
        }

        self.scaleShiftTable = MLXArray.zeros([2, innerDim])
        self.normOut = LayerNorm(dimensions: innerDim, eps: config.normEps, affine: false)
        self.projOut = Linear(innerDim, config.outChannels)
    }

    private func initPreprocessors(_ config: LTXModelConfig) {
        guard config.modelType.isVideoEnabled else { return }

        self.videoArgsPreprocessor = TransformerArgsPreprocessor(
            patchifyProj: patchifyProj!,
            adaln: adalnSingle!,
            captionProjection: captionProjection,
            innerDim: innerDim,
            maxPos: config.positionalEmbeddingMaxPos,
            numAttentionHeads: numAttentionHeads,
            useMiddleIndicesGrid: config.useMiddleIndicesGrid,
            timestepScaleMultiplier: config.timestepScaleMultiplier,
            positionalEmbeddingTheta: config.positionalEmbeddingTheta,
            ropeType: config.ropeType,
            doublePrecisionRope: config.doublePrecisionRope,
            promptAdaln: promptAdalnSingle
        )
    }

    private func initTransformerBlocks(_ config: LTXModelConfig) {
        let videoConfig = config.videoConfig()

        for idx in 0..<config.numLayers {
            transformerBlocks[String(idx)] = BasicAVTransformerBlock(
                idx: idx,
                video: videoConfig,
                audio: nil,
                ropeType: config.ropeType,
                normEps: config.normEps,
                hasPromptAdaln: config.hasPromptAdaln
            )
        }
    }

    func processTransformerBlocks(
        video: TransformerArgs?,
        audio: TransformerArgs?,
        stgVideoBlocks: [Int]? = nil,
        stgAudioBlocks: [Int]? = nil,
        skipCrossModal: Bool = false
    ) -> (TransformerArgs?, TransformerArgs?) {
        let stgVSet = Set(stgVideoBlocks ?? [])
        let stgASet = Set(stgAudioBlocks ?? [])

        var videoOut = video
        var audioOut = audio

        for idx in 0..<config.numLayers {
            guard let block = transformerBlocks[String(idx)] else { continue }
            let result = block(
                video: videoOut,
                audio: audioOut,
                skipVideoSelfAttn: stgVSet.contains(idx),
                skipAudioSelfAttn: stgASet.contains(idx),
                skipCrossModal: skipCrossModal
            )
            videoOut = result.0
            audioOut = result.1
        }

        return (videoOut, audioOut)
    }

    func processOutput(
        scaleShiftTable: MLXArray,
        normOut: LayerNorm,
        projOut: Linear,
        x: MLXArray,
        embeddedTimestep: MLXArray
    ) -> MLXArray {
        // scaleShiftTable: (2, dim) -> expand to (1, 1, 2, dim)
        let tableExpanded = scaleShiftTable.expandedDimensions(axes: [0, 0])
        // embeddedTimestep: (B, 1, dim) -> expand to (B, 1, 1, dim)
        let timestepExpanded = embeddedTimestep.expandedDimensions(axis: 2)

        let scaleShiftValues = tableExpanded + timestepExpanded

        let shift = scaleShiftValues[0..., 0..., 0, 0...]
        let scale = scaleShiftValues[0..., 0..., 1, 0...]

        var result = normOut(x)
        result = result * (1 + scale) + shift
        result = projOut(result)

        return result
    }

    /// Forward pass.
    public func callAsFunction(
        video: Modality? = nil,
        audio: Modality? = nil,
        stgVideoBlocks: [Int]? = nil,
        stgAudioBlocks: [Int]? = nil,
        skipCrossModal: Bool = false
    ) -> (MLXArray?, MLXArray?) {

        // Preprocess arguments
        let videoArgs = video.map { videoArgsPreprocessor!.prepare($0) }

        // Process transformer blocks
        let (videoOut, _) = processTransformerBlocks(
            video: videoArgs,
            audio: nil,
            stgVideoBlocks: stgVideoBlocks,
            stgAudioBlocks: stgAudioBlocks,
            skipCrossModal: skipCrossModal
        )

        // Process outputs
        var vx: MLXArray? = nil
        if let videoOut = videoOut, let table = scaleShiftTable,
           let norm = normOut, let proj = projOut {
            vx = processOutput(
                scaleShiftTable: table,
                normOut: norm,
                projOut: proj,
                x: videoOut.x,
                embeddedTimestep: videoOut.embeddedTimestep
            )
        }

        return (vx, nil)
    }

    /// Sanitize weight keys from PyTorch checkpoint format.
    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasRawPrefix = weights.keys.contains { $0.hasPrefix("model.diffusion_model.") }
        guard hasRawPrefix else { return weights }

        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            guard key.hasPrefix("model.diffusion_model.") else { continue }
            if key.contains("audio_embeddings_connector") || key.contains("video_embeddings_connector") {
                continue
            }

            var newKey = key
            newKey = newKey.replacingOccurrences(of: "model.diffusion_model.", with: "")
            newKey = newKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            newKey = newKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
            newKey = newKey.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
            newKey = newKey.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
            newKey = newKey.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
            newKey = newKey.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
            newKey = newKey.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

            sanitized[newKey] = value
        }

        return sanitized
    }

    /// Load model from a pretrained checkpoint directory.
    public static func fromPretrained(modelPath: String, strict: Bool = true) throws -> LTXModel {
        let url = URL(fileURLWithPath: modelPath)
        let configURL = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(LTXModelConfig.self, from: configData)

        let model = LTXModel(config)

        // Load weights from safetensors files
        var allWeights: [String: MLXArray] = [:]
        let fm = FileManager.default
        if let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: nil) {
            while let fileURL = enumerator.nextObject() as? URL {
                if fileURL.pathExtension == "safetensors" {
                    let fileWeights = try MLX.loadArrays(url: fileURL)
                    for (k, v) in fileWeights {
                        allWeights[k] = v
                    }
                }
            }
        }

        let sanitized = model.sanitize(allWeights)
        let converted = sanitized.mapValues { v -> MLXArray in
            v.dtype == .float32 ? v.asType(.bfloat16) : v
        }

        try model.update(parameters: ModuleParameters.unflattened(converted.map { ($0.key, $0.value) }), verify: strict ? .all : .noUnused)
        eval(model.parameters())

        return model
    }
}

// MARK: - X0Model

/// Wrapper that converts velocity model output to denoised (x0) prediction.
public class X0Model: Module {

    public let velocityModel: LTXModel

    public init(velocityModel: LTXModel) {
        self.velocityModel = velocityModel
    }

    /// Convert velocity prediction to denoised output: x0 = xt - sigma * v
    static func toDenoised(noisy: MLXArray, velocity: MLXArray, sigma: MLXArray) -> MLXArray {
        let noisyF32 = noisy.asType(.float32)
        let velocityF32 = velocity.asType(.float32)
        var sigmaF32 = sigma.asType(.float32)

        // Expand sigma dimensions to match velocity
        while sigmaF32.ndim < velocityF32.ndim {
            sigmaF32 = sigmaF32.expandedDimensions(axis: -1)
        }

        let result = noisyF32 - sigmaF32 * velocityF32
        return result.asType(noisy.dtype)
    }

    public func callAsFunction(
        video: Modality? = nil,
        audio: Modality? = nil,
        stgVideoBlocks: [Int]? = nil,
        stgAudioBlocks: [Int]? = nil,
        skipCrossModal: Bool = false
    ) -> (MLXArray?, MLXArray?) {
        let (vx, ax) = velocityModel(
            video: video,
            audio: audio,
            stgVideoBlocks: stgVideoBlocks,
            stgAudioBlocks: stgAudioBlocks,
            skipCrossModal: skipCrossModal
        )

        var denoisedVideo: MLXArray? = nil
        if let vx = vx, let video = video {
            denoisedVideo = X0Model.toDenoised(
                noisy: video.latent, velocity: vx, sigma: video.timesteps
            )
        }

        return (denoisedVideo, nil)
    }
}
