// MLXVideo.swift - Public API for MLX Video generation
// Swift port of mlx-video (Python) for running LTX and WAN2 video models on Apple Silicon

import Foundation
import MLX
import MLXNN

// MARK: - MLX Video Library

/// MLX Video - Generate videos using LTX and WAN diffusion models on Apple Silicon.
///
/// This library provides Swift implementations of:
/// - **WAN 2.2**: Text-to-Video and Image-to-Video (14B and 1.3B variants)
/// - **LTX Video**: Text-to-Video generation
///
/// All models run locally using the MLX framework, optimized for Apple Silicon.
///
/// ## Usage
///
/// ```swift
/// // Create a WAN2 pipeline
/// let pipeline = try await WanPipeline(modelDirectory: "/path/to/wan2-model")
///
/// // Generate a video
/// let video = try await pipeline.generate(
///     prompt: "A cat playing with a ball in a garden",
///     width: 848,
///     height: 480,
///     numFrames: 81,
///     steps: 40
/// )
/// ```

// Re-export key types
public typealias MLXVideoArray = MLXArray

// MARK: - Version

public let mlxVideoVersion = "0.1.0"

// MARK: - Pipeline Protocol

/// Protocol for video generation pipelines.
public protocol VideoGenerationPipeline: Sendable {
    /// Generate video from text prompt.
    func generate(
        prompt: String,
        negativePrompt: String?,
        width: Int,
        height: Int,
        numFrames: Int,
        steps: Int,
        guidanceScale: GuidanceScale,
        shift: Float,
        seed: Int,
        scheduler: SchedulerType,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GeneratedVideo
}

/// Protocol for image-to-video generation.
public protocol ImageToVideoGenerationPipeline: VideoGenerationPipeline {
    /// Generate video from an input image and text prompt.
    func generateFromImage(
        prompt: String,
        imagePath: String,
        negativePrompt: String?,
        width: Int,
        height: Int,
        numFrames: Int,
        steps: Int,
        guidanceScale: GuidanceScale,
        shift: Float,
        seed: Int,
        scheduler: SchedulerType,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GeneratedVideo
}
