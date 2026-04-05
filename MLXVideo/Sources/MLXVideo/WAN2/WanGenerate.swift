// WanGenerate.swift - Generation pipeline for Wan2 video diffusion
// Ported from mlx_video/models/wan_2/generate.py

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - I2V Utilities

/// Build temporal mask for I2V: first frame = 0, rest = 1.
///
/// - Parameters:
///   - zShape: Latent shape (C, T, H, W) in channels-first
///   - patchSize: (pt, ph, pw) patch size
/// - Returns: (mask, maskTokens) where mask is [C, T, H, W] float32 and maskTokens is [1, L] float32
public func buildI2VMask(
    zShape: (Int, Int, Int, Int),
    patchSize: (Int, Int, Int)
) -> (MLXArray, MLXArray) {
    let (c, t, h, w) = zShape
    var mask = MLXArray.ones([c, t, h, w])
    // Zero out the first temporal position
    mask = concatenated([MLXArray.zeros([c, 1, h, w]), mask[0..., 1...]], axis: 1)

    // Token-level mask for per-token timesteps
    let (pt, ph, pw) = patchSize
    let maskTokens = mask[0, (0...).stride(by: pt), (0...).stride(by: ph), (0...).stride(by: pw)]
    return (mask, maskTokens.reshaped(1, -1))
}

/// Compute best output resolution that fits within maxArea preserving aspect ratio.
func bestOutputSize(w: Int, h: Int, dw: Int, dh: Int, maxArea: Int) -> (Int, Int) {
    let ratio = Float(w) / Float(h)
    let ow = Foundation.sqrt(Float(maxArea) * ratio)
    let oh = Float(maxArea) / ow

    // Option 1: process width first
    let ow1 = Int(ow) / dw * dw
    let oh1 = (maxArea / ow1) / dh * dh
    let ratio1 = Float(ow1) / Float(oh1)

    // Option 2: process height first
    let oh2 = Int(oh) / dh * dh
    let ow2 = (maxArea / oh2) / dw * dw
    let ratio2 = Float(ow2) / Float(oh2)

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio) {
        return (ow1, oh1)
    }
    return (ow2, oh2)
}

// MARK: - WanPipeline

/// Wan2 video generation pipeline supporting Text-to-Video and Image-to-Video.
///
/// Usage:
/// ```swift
/// let pipeline = WanPipeline(config: .wan22T2V14B())
/// // Load model weights, text encoder, VAE separately
/// // pipeline.model = ...
/// // pipeline.vae = ...
/// // pipeline.textEncoder = ...
/// let video = pipeline.generateVideo(prompt: "A cat playing piano", ...)
/// ```
public class WanPipeline {
    public let config: WanModelConfig

    /// The main diffusion model (or dual models for Wan2.2)
    public var model: WanModel?
    public var highNoiseModel: WanModel?
    public var lowNoiseModel: WanModel?

    /// VAE decoder (and optionally encoder for I2V)
    public var vae: WanVAE?

    /// T5 text encoder
    public var textEncoder: T5Encoder?

    public init(config: WanModelConfig) {
        self.config = config
    }

    /// Whether this is a dual-model pipeline.
    public var isDualModel: Bool {
        config.dualModel
    }

    /// Select the appropriate model for the given timestep.
    public func selectModel(timestep: Float) -> WanModel {
        if isDualModel {
            let boundary = config.boundary * Float(config.numTrainTimesteps)
            if timestep >= boundary {
                guard let model = highNoiseModel else {
                    fatalError("High noise model not loaded for dual-model pipeline")
                }
                return model
            } else {
                guard let model = lowNoiseModel else {
                    fatalError("Low noise model not loaded for dual-model pipeline")
                }
                return model
            }
        } else {
            guard let model = model else {
                fatalError("Model not loaded for single-model pipeline")
            }
            return model
        }
    }

    /// Encode text using the T5 encoder.
    ///
    /// - Parameters:
    ///   - ids: Token IDs [B, L]
    ///   - mask: Attention mask [B, L]
    /// - Returns: Text embeddings [B, L, dim]
    public func encodeText(ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        guard let encoder = textEncoder else {
            fatalError("Text encoder not loaded")
        }
        return encoder(ids: ids, mask: mask)
    }

    /// Generate video from text (and optionally an image).
    ///
    /// This is a simplified generation loop. For full functionality including
    /// dual-model switching, I2V support, and tiling, see the Python reference.
    ///
    /// - Parameters:
    ///   - prompt: Pre-encoded text embeddings [L, textDim]
    ///   - negativePrompt: Optional negative prompt embeddings [L, textDim]
    ///   - width: Video width
    ///   - height: Video height
    ///   - numFrames: Number of frames (must be 4n+1)
    ///   - steps: Number of diffusion steps
    ///   - guideScale: Guidance scale
    ///   - shift: Noise schedule shift
    ///   - seed: Random seed
    ///   - scheduler: Diffusion scheduler to use
    ///   - yI2V: Optional I2V conditioning tensor [C_y, F, H, W]
    ///   - progressHandler: Optional callback for step progress (step, totalSteps)
    /// - Returns: Generated video latent [C, F, H, W] in float32
    public func generateLatents(
        prompt: MLXArray,
        negativePrompt: MLXArray? = nil,
        width: Int = 1280,
        height: Int = 704,
        numFrames: Int = 81,
        steps: Int? = nil,
        guideScale: Float? = nil,
        shift: Float? = nil,
        seed: UInt64 = 0,
        scheduler: inout (any DiffusionScheduler),
        yI2V: MLXArray? = nil,
        progressHandler: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let numSteps = steps ?? config.sampleSteps
        let scheduleShift = shift ?? config.sampleShift
        let gs = guideScale ?? config.sampleGuideScale.low
        let cfgDisabled = gs <= 1.0

        precondition((numFrames - 1) % 4 == 0, "numFrames must be 4n+1, got \(numFrames)")

        // Set seed
        MLXRandom.seed(seed)

        // Align dimensions
        let vaeStride = config.vaeStride
        let patchSize = config.patchSize
        let alignH = patchSize.1 * vaeStride.1
        let alignW = patchSize.2 * vaeStride.2
        let alignedHeight = (height / alignH) * alignH
        let alignedWidth = (width / alignW) * alignW

        // Compute target latent shape
        let zDim = config.vaeZDim
        let tLatent = (numFrames - 1) / vaeStride.0 + 1
        let hLatent = alignedHeight / vaeStride.1
        let wLatent = alignedWidth / vaeStride.2
        let targetShape = [zDim, tLatent, hLatent, wLatent]

        // Sequence length for transformer
        let seqLen = Int(Foundation.ceil(
            Float(hLatent * wLatent) / Float(patchSize.1 * patchSize.2) * Float(tLatent)
        ))

        // Setup scheduler
        scheduler.setTimesteps(numSteps: numSteps, shift: scheduleShift)

        // Generate initial noise
        var latents = MLXRandom.normal(targetShape)

        // Get the model to use (for single-model mode or to extract text embeddings)
        let activeModel = isDualModel ? (highNoiseModel ?? lowNoiseModel!) : model!

        // Pre-embed text
        let contextEmb: MLXArray
        if cfgDisabled {
            contextEmb = activeModel.embedText([prompt])
        } else {
            let neg = negativePrompt ?? MLXArray.zeros(like: prompt)
            contextEmb = activeModel.embedText([prompt, neg])
        }
        eval(contextEmb)

        // Pre-compute cross-attention K/V
        let crossKV = activeModel.prepareCrossKV(contextEmb)
        eval(crossKV)

        // Pre-compute RoPE
        let fGrid = tLatent / patchSize.0
        let hGrid = hLatent / patchSize.1
        let wGrid = wLatent / patchSize.2
        let ropeGridSizes: [(Int, Int, Int)] = cfgDisabled
            ? [(fGrid, hGrid, wGrid)]
            : [(fGrid, hGrid, wGrid), (fGrid, hGrid, wGrid)]
        let ropeCosSin = activeModel.prepareRope(ropeGridSizes)
        eval(ropeCosSin)

        // Pre-convert timesteps
        guard let timesteps = scheduler.timesteps else {
            fatalError("Scheduler timesteps not set")
        }
        let timestepArray = timesteps.asType(.float32)

        // Diffusion loop
        for i in 0..<numSteps {
            let timestepVal = timestepArray[i].item(Float.self)
            let currentModel = selectModel(timestep: timestepVal)

            let noisePred: MLXArray
            if cfgDisabled {
                let tBatch = MLXArray([timestepVal])
                let yArg: [MLXArray]? = yI2V != nil ? [yI2V!] : nil
                let ctx = currentModel.embedText([prompt])
                let kv = currentModel.prepareCrossKV(ctx)
                let rcs = currentModel.prepareRope([(fGrid, hGrid, wGrid)])
                let preds = currentModel(
                    xList: [latents],
                    t: tBatch,
                    context: ctx,
                    seqLen: seqLen,
                    crossKVCaches: kv,
                    y: yArg,
                    ropeCosSin: rcs
                )
                noisePred = preds[0]
            } else {
                let tBatch = MLXArray([timestepVal, timestepVal])
                let yArg: [MLXArray]? = yI2V != nil ? [yI2V!, yI2V!] : nil
                let preds = currentModel(
                    xList: [latents, latents],
                    t: tBatch,
                    context: contextEmb,
                    seqLen: seqLen,
                    crossKVCaches: crossKV,
                    y: yArg,
                    ropeCosSin: ropeCosSin
                )
                let noisePredCond = preds[0]
                let noisePredUncond = preds[1]
                noisePred = noisePredUncond + gs * (noisePredCond - noisePredUncond)
            }

            latents = scheduler.step(
                modelOutput: noisePred.expandedDimensions(axis: 0),
                timestep: MLXArray(timestepVal),
                sample: latents.expandedDimensions(axis: 0)
            ).squeezed(axis: 0)

            eval(latents)
            progressHandler?(i + 1, numSteps)
        }

        return latents
    }

    /// Decode latents to video pixels using the VAE.
    ///
    /// - Parameter latents: Latent tensor [C, F, H, W]
    /// - Returns: Video tensor [B, 3, T, H, W] clamped to [-1, 1]
    public func decodeLatents(_ latents: MLXArray) -> MLXArray {
        guard let vae = vae else {
            fatalError("VAE not loaded")
        }
        return vae.decode(latents.expandedDimensions(axis: 0))
    }

    /// Full generation pipeline: text to video pixels.
    ///
    /// - Parameters:
    ///   - prompt: Pre-encoded text embeddings [L, textDim]
    ///   - negativePrompt: Optional negative prompt embeddings
    ///   - width: Video width
    ///   - height: Video height
    ///   - numFrames: Number of frames (must be 4n+1)
    ///   - steps: Diffusion steps
    ///   - guideScale: Guidance scale
    ///   - shift: Schedule shift
    ///   - seed: Random seed
    ///   - scheduler: Diffusion scheduler
    ///   - progressHandler: Progress callback
    /// - Returns: Video [B, 3, T, H, W] clamped to [-1, 1]
    public func generateVideo(
        prompt: MLXArray,
        negativePrompt: MLXArray? = nil,
        width: Int = 1280,
        height: Int = 704,
        numFrames: Int = 81,
        steps: Int? = nil,
        guideScale: Float? = nil,
        shift: Float? = nil,
        seed: UInt64 = 0,
        scheduler: inout (any DiffusionScheduler),
        progressHandler: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let latents = generateLatents(
            prompt: prompt,
            negativePrompt: negativePrompt,
            width: width,
            height: height,
            numFrames: numFrames,
            steps: steps,
            guideScale: guideScale,
            shift: shift,
            seed: seed,
            scheduler: &scheduler,
            progressHandler: progressHandler
        )
        return decodeLatents(latents)
    }
}
