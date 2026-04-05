// ContentView.swift - Main app layout with macOS 26 features

import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) private var appState
    @State private var viewModel = VideoGeneratorViewModel()
    @State private var showingSettings = false
    @State private var selectedTab: SidebarTab = .generate

    enum SidebarTab: String, CaseIterable, Identifiable {
        case generate = "Generate"
        case gallery = "Gallery"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .generate: return "wand.and.stars"
            case .gallery: return "photo.on.rectangle.angled"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebarContent
        } detail: {
            detailContent
        }
        .onAppear {
            viewModel.setAppState(appState)
        }
        .navigationTitle("MLX Video Studio")
    }

    // MARK: - Sidebar

    @ViewBuilder
    private var sidebarContent: some View {
        List(selection: $selectedTab) {
            ForEach(SidebarTab.allCases) { tab in
                Label(tab.rawValue, systemImage: tab.icon)
                    .tag(tab)
            }
        }
        .listStyle(.sidebar)
        .navigationSplitViewColumnWidth(min: 180, ideal: 200)
    }

    // MARK: - Detail

    @ViewBuilder
    private var detailContent: some View {
        switch selectedTab {
        case .generate:
            GenerationView(viewModel: viewModel)
        case .gallery:
            GalleryView()
        }
    }
}

// MARK: - Generation View

struct GenerationView: View {
    @Environment(AppState.self) private var appState
    let viewModel: VideoGeneratorViewModel

    var body: some View {
        @Bindable var state = appState

        HSplitView {
            // Left panel: Parameters
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    modelSelectionSection
                    modelDirectorySection
                    promptSection
                    if appState.selectedModel.isImageToVideo {
                        imageInputSection
                    }
                    parametersSection
                    advancedSection
                    generateButton
                }
                .padding()
            }
            .frame(minWidth: 380, idealWidth: 420, maxWidth: 500)

            // Right panel: Preview and Progress
            VStack {
                if appState.isGenerating {
                    progressView
                } else if !appState.generatedFrames.isEmpty {
                    videoPreview
                } else {
                    placeholderView
                }
            }
            .frame(minWidth: 400)
        }
    }

    // MARK: - Model Selection

    @ViewBuilder
    private var modelSelectionSection: some View {
        @Bindable var state = appState
        GroupBox("Model") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Model", selection: $state.selectedModel) {
                    ForEach(VideoModel.allCases) { model in
                        Text(model.rawValue).tag(model)
                    }
                }
                .pickerStyle(.segmented)

                Text(appState.selectedModel.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Model Directory

    @ViewBuilder
    private var modelDirectorySection: some View {
        @Bindable var state = appState
        GroupBox("Model Directory") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Path to MLX model...", text: $state.modelDirectory)
                        .textFieldStyle(.roundedBorder)

                    Button("Browse...") {
                        viewModel.browseModelDirectory()
                    }
                }

                HStack {
                    Button("Load Model") {
                        Task { await viewModel.loadModel() }
                    }
                    .disabled(state.modelDirectory.isEmpty)

                    if state.isModelLoaded {
                        Label("Loaded", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                    }

                    if !state.modelLoadProgress.isEmpty && !state.isModelLoaded {
                        Text(state.modelLoadProgress)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Prompt Input

    @ViewBuilder
    private var promptSection: some View {
        @Bindable var state = appState
        GroupBox("Prompt") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Describe the video you want to generate")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextEditor(text: $state.prompt)
                    .frame(minHeight: 80, maxHeight: 120)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .background(Color(.textBackgroundColor))
                    .cornerRadius(6)

                if appState.selectedModel.supportsNegativePrompt {
                    DisclosureGroup("Negative Prompt") {
                        TextEditor(text: $state.negativePrompt)
                            .frame(minHeight: 40, maxHeight: 80)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .padding(4)
                            .background(Color(.textBackgroundColor))
                            .cornerRadius(6)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Image Input (I2V)

    @ViewBuilder
    private var imageInputSection: some View {
        GroupBox("Input Image") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Select an image to animate")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button("Choose Image...") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.image]
                    panel.canChooseFiles = true
                    if panel.runModal() == .OK, let url = panel.url {
                        appState.inputImagePath = url.path
                        appState.inputImageData = try? Data(contentsOf: url)
                    }
                }

                if let imagePath = appState.inputImagePath {
                    HStack {
                        Image(systemName: "photo")
                        Text(URL(fileURLWithPath: imagePath).lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Parameters

    @ViewBuilder
    private var parametersSection: some View {
        @Bindable var state = appState
        GroupBox("Parameters") {
            VStack(alignment: .leading, spacing: 12) {
                // Resolution
                VStack(alignment: .leading, spacing: 4) {
                    Text("Resolution")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        Picker("Preset", selection: Binding(
                            get: {
                                ResolutionPreset.presets.first {
                                    $0.width == state.width && $0.height == state.height
                                } ?? ResolutionPreset.presets[0]
                            },
                            set: { preset in
                                state.width = preset.width
                                state.height = preset.height
                            }
                        )) {
                            ForEach(ResolutionPreset.presets) { preset in
                                Text(preset.name).tag(preset)
                            }
                        }
                    }
                    Text("\(state.width) × \(state.height)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Frames
                VStack(alignment: .leading, spacing: 4) {
                    Text("Duration")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("Frames", selection: $state.numFrames) {
                        ForEach(FramePreset.presets) { preset in
                            Text(preset.name).tag(preset.frames)
                        }
                    }
                }

                // Steps
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Steps: \(state.steps)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    Slider(
                        value: Binding(
                            get: { Double(state.steps) },
                            set: { state.steps = Int($0) }
                        ),
                        in: 10...100,
                        step: 1
                    )
                }

                // Guidance Scale
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Guidance Scale: \(String(format: "%.1f", state.guidanceScale))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    Slider(value: $state.guidanceScale, in: 1.0...20.0, step: 0.5)
                }

                // Scheduler
                Picker("Scheduler", selection: $state.selectedScheduler) {
                    ForEach(SchedulerOption.allCases) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Advanced Settings

    @ViewBuilder
    private var advancedSection: some View {
        @Bindable var state = appState
        DisclosureGroup("Advanced") {
            VStack(alignment: .leading, spacing: 12) {
                // Shift
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Shift: \(String(format: "%.1f", state.shift))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    Slider(value: $state.shift, in: 1.0...20.0, step: 0.5)
                }

                // Seed
                HStack {
                    TextField("Seed (-1 = random)", value: $state.seed, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 150)

                    Button("Random") {
                        state.seed = -1
                    }
                }

                // Dual guidance
                Toggle("Dual Guidance Scale", isOn: $state.useDualGuidance)
                if state.useDualGuidance {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("High Noise: \(String(format: "%.1f", state.guidanceScaleHigh))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Slider(value: $state.guidanceScaleHigh, in: 1.0...20.0, step: 0.5)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Generate Button

    @ViewBuilder
    private var generateButton: some View {
        HStack {
            if appState.isGenerating {
                Button("Cancel", role: .cancel) {
                    viewModel.cancelGeneration()
                }
                .controlSize(.large)
            } else {
                Button {
                    Task { await viewModel.generateVideo() }
                } label: {
                    Label("Generate Video", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .disabled(appState.prompt.isEmpty)
            }
        }

        if let error = appState.errorMessage {
            Text(error)
                .font(.caption)
                .foregroundStyle(.red)
        }
    }

    // MARK: - Progress View

    @ViewBuilder
    private var progressView: some View {
        VStack(spacing: 20) {
            Spacer()

            ProgressView(value: appState.generationProgress) {
                Text(appState.generationPhase)
                    .font(.headline)
            } currentValueLabel: {
                Text(appState.generationMessage)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .progressViewStyle(.linear)
            .padding(.horizontal, 40)

            if appState.generationStep > 0 {
                Text("Step \(appState.generationStep) of \(appState.generationTotalSteps)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
    }

    // MARK: - Video Preview

    @ViewBuilder
    private var videoPreview: some View {
        VStack {
            Text("Video Generated!")
                .font(.title2)
                .padding()

            if let duration = appState.generationStartTime {
                Text(String(format: "Generated in %.1f seconds", appState.generationDuration))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Export Video...") {
                let panel = NSSavePanel()
                panel.allowedContentTypes = [.mpeg4Movie]
                panel.nameFieldStringValue = "generated_video.mp4"
                if panel.runModal() == .OK, let url = panel.url {
                    Task { try? await viewModel.exportVideo(to: url) }
                }
            }
            .controlSize(.large)
            .padding()
        }
    }

    // MARK: - Placeholder

    @ViewBuilder
    private var placeholderView: some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: "film")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)

            Text("MLX Video Studio")
                .font(.title)
                .foregroundStyle(.secondary)

            Text("Generate videos using LTX and WAN models\nrunning locally on Apple Silicon with MLX")
                .multilineTextAlignment(.center)
                .foregroundStyle(.tertiary)

            VStack(alignment: .leading, spacing: 8) {
                Label("1. Select a model and load weights", systemImage: "1.circle")
                Label("2. Enter a text prompt", systemImage: "2.circle")
                Label("3. Adjust parameters", systemImage: "3.circle")
                Label("4. Click Generate Video", systemImage: "4.circle")
            }
            .font(.callout)
            .foregroundStyle(.secondary)
            .padding()

            Spacer()
        }
    }
}

// MARK: - Gallery View

struct GalleryView: View {
    var body: some View {
        VStack {
            Spacer()
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("Generated videos will appear here")
                .foregroundStyle(.secondary)
            Spacer()
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gear")
                }

            PerformanceSettingsView()
                .tabItem {
                    Label("Performance", systemImage: "speedometer")
                }
        }
        .frame(width: 450, height: 300)
    }
}

struct GeneralSettingsView: View {
    @AppStorage("defaultOutputDirectory") private var defaultOutputDir = ""
    @AppStorage("autoOpenAfterGeneration") private var autoOpen = true

    var body: some View {
        Form {
            TextField("Default Output Directory", text: $defaultOutputDir)
            Toggle("Auto-open video after generation", isOn: $autoOpen)
        }
        .padding()
    }
}

struct PerformanceSettingsView: View {
    @AppStorage("enableCompilation") private var enableCompilation = true
    @AppStorage("tilingMode") private var tilingMode = "auto"

    var body: some View {
        Form {
            Toggle("Enable mx.compile", isOn: $enableCompilation)
            Picker("VAE Tiling", selection: $tilingMode) {
                Text("Auto").tag("auto")
                Text("None").tag("none")
                Text("Aggressive").tag("aggressive")
                Text("Conservative").tag("conservative")
            }
        }
        .padding()
    }
}
