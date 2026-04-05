// AppState.swift - Application state management

import SwiftUI
import MLX

// MARK: - App State

@Observable
@MainActor
final class AppState {
    // Generation parameters
    var prompt: String = ""
    var negativePrompt: String = ""
    var selectedModel: VideoModel = .wan2T2V
    var width: Int = 848
    var height: Int = 480
    var numFrames: Int = 81
    var steps: Int = 40
    var guidanceScale: Double = 5.0
    var guidanceScaleHigh: Double = 4.0
    var useDualGuidance: Bool = false
    var shift: Double = 12.0
    var seed: Int = -1
    var selectedScheduler: SchedulerOption = .unipc

    // Image-to-video
    var inputImagePath: String? = nil
    var inputImageData: Data? = nil

    // Model state
    var modelDirectory: String = ""
    var isModelLoaded: Bool = false
    var modelLoadProgress: String = ""

    // Generation state
    var isGenerating: Bool = false
    var generationProgress: Double = 0.0
    var generationStep: Int = 0
    var generationTotalSteps: Int = 0
    var generationPhase: String = ""
    var generationMessage: String = ""

    // Output
    var generatedFrames: [CGImage] = []
    var generatedVideoURL: URL? = nil
    var errorMessage: String? = nil

    // Performance
    var generationStartTime: Date? = nil
    var generationDuration: TimeInterval = 0
    var peakMemoryUsage: String = ""
}

// MARK: - Model Options

enum VideoModel: String, CaseIterable, Identifiable {
    case wan2T2V = "WAN 2.2 Text-to-Video"
    case wan2I2V = "WAN 2.2 Image-to-Video"
    case ltxT2V = "LTX Text-to-Video"

    var id: String { rawValue }

    var isImageToVideo: Bool {
        self == .wan2I2V
    }

    var defaultWidth: Int {
        switch self {
        case .wan2T2V, .wan2I2V: return 848
        case .ltxT2V: return 768
        }
    }

    var defaultHeight: Int {
        switch self {
        case .wan2T2V, .wan2I2V: return 480
        case .ltxT2V: return 512
        }
    }

    var defaultSteps: Int {
        switch self {
        case .wan2T2V, .wan2I2V: return 40
        case .ltxT2V: return 50
        }
    }

    var defaultGuidanceScale: Double {
        switch self {
        case .wan2T2V: return 5.0
        case .wan2I2V: return 3.5
        case .ltxT2V: return 3.0
        }
    }

    var supportsNegativePrompt: Bool { true }

    var description: String {
        switch self {
        case .wan2T2V:
            return "Generate videos from text descriptions using WAN 2.2"
        case .wan2I2V:
            return "Generate videos from an input image using WAN 2.2"
        case .ltxT2V:
            return "Generate videos from text descriptions using LTX Video"
        }
    }
}

enum SchedulerOption: String, CaseIterable, Identifiable {
    case euler = "Euler"
    case dpmpp = "DPM++ 2M"
    case unipc = "UniPC"

    var id: String { rawValue }
}

// MARK: - Resolution Presets

struct ResolutionPreset: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let width: Int
    let height: Int

    static let presets: [ResolutionPreset] = [
        .init(name: "480p (16:9)", width: 848, height: 480),
        .init(name: "480p (9:16)", width: 480, height: 848),
        .init(name: "480p (1:1)", width: 480, height: 480),
        .init(name: "720p (16:9)", width: 1280, height: 720),
        .init(name: "720p (9:16)", width: 720, height: 1280),
        .init(name: "544p (16:9)", width: 960, height: 544),
    ]
}

// MARK: - Frame Presets

struct FramePreset: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let frames: Int

    static let presets: [FramePreset] = [
        .init(name: "~2 sec (33 frames)", frames: 33),
        .init(name: "~3 sec (49 frames)", frames: 49),
        .init(name: "~4 sec (65 frames)", frames: 65),
        .init(name: "~5 sec (81 frames)", frames: 81),
        .init(name: "~7 sec (113 frames)", frames: 113),
        .init(name: "~9 sec (145 frames)", frames: 145),
    ]
}
