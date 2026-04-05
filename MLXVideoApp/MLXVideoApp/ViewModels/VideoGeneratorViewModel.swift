// VideoGeneratorViewModel.swift - Generation logic and model management

import AppKit
import SwiftUI
import MLX
import MLXNN
import MLXRandom

@Observable
@MainActor
final class VideoGeneratorViewModel {
    private var appState: AppState?

    func setAppState(_ state: AppState) {
        self.appState = state
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard let state = appState else { return }
        guard !state.modelDirectory.isEmpty else {
            state.errorMessage = "Please select a model directory first."
            return
        }

        state.isModelLoaded = false
        state.modelLoadProgress = "Loading model weights..."
        state.errorMessage = nil

        do {
            // Simulate model loading stages for the UI
            state.modelLoadProgress = "Loading configuration..."
            try await Task.sleep(for: .milliseconds(100))

            state.modelLoadProgress = "Loading transformer weights..."
            try await Task.sleep(for: .milliseconds(100))

            state.modelLoadProgress = "Loading VAE weights..."
            try await Task.sleep(for: .milliseconds(100))

            state.modelLoadProgress = "Loading text encoder..."
            try await Task.sleep(for: .milliseconds(100))

            state.isModelLoaded = true
            state.modelLoadProgress = "Model loaded successfully!"
        } catch {
            state.errorMessage = "Failed to load model: \(error.localizedDescription)"
            state.modelLoadProgress = ""
        }
    }

    // MARK: - Video Generation

    func generateVideo() async {
        guard let state = appState else { return }
        guard !state.prompt.isEmpty else {
            state.errorMessage = "Please enter a text prompt."
            return
        }

        state.isGenerating = true
        state.errorMessage = nil
        state.generatedFrames = []
        state.generatedVideoURL = nil
        state.generationStartTime = Date()

        let totalSteps = state.steps
        state.generationTotalSteps = totalSteps

        do {
            // Phase 1: Text encoding
            state.generationPhase = "Encoding text..."
            state.generationStep = 0
            state.generationProgress = 0.0
            try await Task.sleep(for: .milliseconds(200))

            // Phase 2: Image encoding (I2V only)
            if state.selectedModel.isImageToVideo {
                state.generationPhase = "Encoding image..."
                try await Task.sleep(for: .milliseconds(200))
            }

            // Phase 3: Denoising loop
            state.generationPhase = "Denoising..."
            for step in 0..<totalSteps {
                state.generationStep = step + 1
                state.generationProgress = Double(step + 1) / Double(totalSteps)
                state.generationMessage = "Step \(step + 1)/\(totalSteps)"

                // In production, this is where the actual denoising step happens:
                // let noise = scheduler.step(modelOutput: prediction, timestep: t, sample: latent)
                try await Task.sleep(for: .milliseconds(50))
            }

            // Phase 4: VAE decoding
            state.generationPhase = "Decoding video..."
            state.generationMessage = "Running VAE decoder..."
            try await Task.sleep(for: .milliseconds(300))

            // Phase 5: Post-processing
            state.generationPhase = "Post-processing..."
            state.generationMessage = "Converting to video frames..."
            try await Task.sleep(for: .milliseconds(200))

            // Complete
            state.generationPhase = "Complete!"
            state.generationProgress = 1.0
            state.generationDuration = Date().timeIntervalSince(state.generationStartTime ?? Date())
            state.generationMessage = String(format: "Generated in %.1f seconds", state.generationDuration)

        } catch {
            state.errorMessage = "Generation failed: \(error.localizedDescription)"
        }

        state.isGenerating = false
    }

    // MARK: - Cancel Generation

    func cancelGeneration() {
        guard let state = appState else { return }
        state.isGenerating = false
        state.generationPhase = "Cancelled"
        state.generationMessage = "Generation was cancelled."
    }

    // MARK: - Export Video

    func exportVideo(to url: URL) async throws {
        // In production: use AVFoundation to write frames to MP4
        // For now this is a placeholder
    }

    // MARK: - Browse Model Directory

    func browseModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select the converted MLX model directory"

        if panel.runModal() == .OK, let url = panel.url {
            appState?.modelDirectory = url.path
        }
    }
}
