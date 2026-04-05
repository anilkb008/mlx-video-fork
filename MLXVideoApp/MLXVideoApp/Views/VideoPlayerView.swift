// VideoPlayerView.swift - Video playback and frame preview
// Uses macOS 26 features where available

import SwiftUI
import AVKit

// MARK: - Video Player View

struct VideoPlayerView: View {
    let videoURL: URL?
    let frames: [CGImage]
    @State private var currentFrameIndex = 0
    @State private var isPlaying = false
    @State private var playbackTimer: Timer?

    var body: some View {
        VStack {
            if let url = videoURL {
                // Full video player
                avPlayerView(url: url)
            } else if !frames.isEmpty {
                // Frame-by-frame preview
                framePreview
            } else {
                emptyState
            }
        }
    }

    @ViewBuilder
    private func avPlayerView(url: URL) -> some View {
        let player = AVPlayer(url: url)
        VideoPlayer(player: player)
            .frame(minHeight: 300)
            .cornerRadius(8)
            .onAppear {
                player.play()
            }
    }

    @ViewBuilder
    private var framePreview: some View {
        VStack(spacing: 12) {
            // Frame display
            if currentFrameIndex < frames.count {
                Image(nsImage: NSImage(cgImage: frames[currentFrameIndex], size: .zero))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 400)
                    .cornerRadius(8)
                    .shadow(radius: 4)
            }

            // Playback controls
            HStack(spacing: 16) {
                Button {
                    currentFrameIndex = 0
                } label: {
                    Image(systemName: "backward.end.fill")
                }

                Button {
                    if currentFrameIndex > 0 {
                        currentFrameIndex -= 1
                    }
                } label: {
                    Image(systemName: "backward.frame.fill")
                }

                Button {
                    togglePlayback()
                } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                }

                Button {
                    if currentFrameIndex < frames.count - 1 {
                        currentFrameIndex += 1
                    }
                } label: {
                    Image(systemName: "forward.frame.fill")
                }

                Button {
                    currentFrameIndex = frames.count - 1
                } label: {
                    Image(systemName: "forward.end.fill")
                }
            }

            // Timeline scrubber
            if frames.count > 1 {
                Slider(
                    value: Binding(
                        get: { Double(currentFrameIndex) },
                        set: { currentFrameIndex = Int($0) }
                    ),
                    in: 0...Double(frames.count - 1),
                    step: 1
                )
                .padding(.horizontal)

                Text("Frame \(currentFrameIndex + 1) of \(frames.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack {
            Image(systemName: "play.rectangle")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("No video to display")
                .foregroundStyle(.secondary)
        }
    }

    private func togglePlayback() {
        isPlaying.toggle()
        if isPlaying {
            playbackTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 16.0, repeats: true) { _ in
                if currentFrameIndex < frames.count - 1 {
                    currentFrameIndex += 1
                } else {
                    currentFrameIndex = 0 // Loop
                }
            }
        } else {
            playbackTimer?.invalidate()
            playbackTimer = nil
        }
    }
}

// MARK: - Generation Progress Ring

struct ProgressRingView: View {
    let progress: Double
    let phase: String

    var body: some View {
        ZStack {
            Circle()
                .stroke(lineWidth: 8)
                .opacity(0.1)
                .foregroundStyle(.blue)

            Circle()
                .trim(from: 0.0, to: CGFloat(progress))
                .stroke(style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round))
                .foregroundStyle(.blue)
                .rotationEffect(.degrees(-90))
                .animation(.linear, value: progress)

            VStack(spacing: 4) {
                Text("\(Int(progress * 100))%")
                    .font(.title2.monospacedDigit())
                    .bold()
                Text(phase)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 120, height: 120)
    }
}

// MARK: - Memory Usage View

struct MemoryUsageView: View {
    @State private var memoryUsage: String = "—"

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "memorychip")
                .font(.caption)
            Text(memoryUsage)
                .font(.caption.monospacedDigit())
        }
        .foregroundStyle(.secondary)
        .onAppear {
            updateMemoryUsage()
        }
    }

    private func updateMemoryUsage() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let mb = Double(info.resident_size) / 1_048_576.0
            memoryUsage = String(format: "%.0f MB", mb)
        }
    }
}
