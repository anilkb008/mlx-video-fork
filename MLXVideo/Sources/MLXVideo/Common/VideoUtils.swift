// VideoUtils.swift - Video processing utilities for MLX Video
// Ported from mlx_video/utils.py and postprocess.py

import Foundation
import MLX

// MARK: - Denoising Utilities

/// Convert velocity prediction to denoised output.
/// x0 = (sample - sigma * velocity) / (1 - sigma)
/// Matches `to_denoised` from mlx_video/utils.py
public func toDenoised(sample: MLXArray, velocity: MLXArray, sigma: MLXArray) -> MLXArray {
    return (sample - sigma * velocity) / (1.0 - sigma)
}

/// RMS normalization utility (no learned parameters).
/// Matches `rms_norm` from mlx_video/utils.py
public func rmsNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * MLX.rsqrt(variance + eps)
}

// MARK: - Video Post-Processing

/// Convert raw model output tensor to pixel values in [0, 255] range.
/// Input: [-1, 1] float tensor
/// Output: [0, 255] uint8 tensor
public func tensorToPixels(_ tensor: MLXArray) -> MLXArray {
    let clamped = MLX.clip(tensor, min: -1.0, max: 1.0)
    let normalized = (clamped + 1.0) / 2.0 * 255.0
    return normalized.asType(.uint8)
}

/// Compute best output resolution preserving aspect ratio within max area.
/// Matches _best_output_size from wan_2/generate.py
public func bestOutputSize(
    width: Int, height: Int,
    alignWidth: Int, alignHeight: Int,
    maxArea: Int
) -> (width: Int, height: Int) {
    let ratio = Float(width) / Float(height)
    let ow = sqrt(Float(maxArea) * ratio)
    let oh = Float(maxArea) / ow

    // Option 1: process width first
    let ow1 = Int(ow / Float(alignWidth)) * alignWidth
    let oh1 = Int(Float(maxArea) / Float(ow1) / Float(alignHeight)) * alignHeight
    let ratio1 = Float(ow1) / Float(oh1)

    // Option 2: process height first
    let oh2 = Int(oh / Float(alignHeight)) * alignHeight
    let ow2 = Int(Float(maxArea) / Float(oh2) / Float(alignWidth)) * alignWidth
    let ratio2 = Float(ow2) / Float(oh2)

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio) {
        return (ow1, oh1)
    }
    return (ow2, oh2)
}

// MARK: - Video Frame Export

/// Represents a generated video as a sequence of frames.
public struct GeneratedVideo: Sendable {
    /// Frames as [T, H, W, 3] uint8 array
    public let frames: MLXArray
    /// Frames per second
    public let fps: Int
    /// Video width
    public let width: Int
    /// Video height
    public let height: Int
    /// Number of frames
    public let frameCount: Int

    public init(frames: MLXArray, fps: Int) {
        self.frames = frames
        self.fps = fps
        let shape = frames.shape
        self.frameCount = shape[0]
        self.height = shape[1]
        self.width = shape[2]
    }
}

// MARK: - Progress Tracking

/// Represents the progress of video generation.
public struct GenerationProgress: Sendable {
    public let step: Int
    public let totalSteps: Int
    public let phase: GenerationPhase
    public let message: String

    public var fraction: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(step) / Double(totalSteps)
    }
}

public enum GenerationPhase: String, Sendable {
    case loadingModel = "Loading Model"
    case encodingText = "Encoding Text"
    case encodingImage = "Encoding Image"
    case denoising = "Denoising"
    case decodingVideo = "Decoding Video"
    case postProcessing = "Post-Processing"
    case complete = "Complete"
}

// MARK: - Generation Parameters

/// Parameters for video generation.
public struct VideoGenerationParams: Sendable {
    public let prompt: String
    public let negativePrompt: String?
    public let imagePath: String?
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let steps: Int?
    public let guidanceScale: GuidanceScale
    public let shift: Float?
    public let seed: Int
    public let scheduler: SchedulerType

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        imagePath: String? = nil,
        width: Int = 1280,
        height: Int = 704,
        numFrames: Int = 81,
        steps: Int? = nil,
        guidanceScale: GuidanceScale = .single(5.0),
        shift: Float? = nil,
        seed: Int = -1,
        scheduler: SchedulerType = .unipc
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.imagePath = imagePath
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.seed = seed
        self.scheduler = scheduler
    }
}

public enum GuidanceScale: Sendable {
    case single(Float)
    case dual(low: Float, high: Float)

    public var isDisabled: Bool {
        switch self {
        case .single(let v): return v <= 1.0
        case .dual(let low, let high): return low <= 1.0 && high <= 1.0
        }
    }
}

public enum SchedulerType: String, CaseIterable, Sendable {
    case euler
    case dpmpp = "dpm++"
    case unipc
}

// MARK: - Model Type

public enum VideoModelType: String, CaseIterable, Sendable {
    case wan2TextToVideo = "WAN2 Text-to-Video"
    case wan2ImageToVideo = "WAN2 Image-to-Video"
    case ltxTextToVideo = "LTX Text-to-Video"
}
