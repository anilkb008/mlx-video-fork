// PromptSuggestionsView.swift - AI prompt suggestions with macOS 26 styling

import SwiftUI

struct PromptSuggestionsView: View {
    @Binding var selectedPrompt: String

    static let suggestions: [(category: String, prompts: [String])] = [
        ("Nature", [
            "A serene mountain lake at sunrise with mist rolling over the water, cinematic lighting",
            "Waves crashing on a rocky coastline during golden hour, slow motion, 4K quality",
            "A field of wildflowers swaying gently in the breeze, shallow depth of field",
            "Northern lights dancing over a snowy landscape, timelapse, vibrant colors",
        ]),
        ("Animals", [
            "A golden retriever playing fetch on a sandy beach, happy expression, sunlight",
            "A butterfly emerging from its cocoon in extreme close-up, macro photography",
            "A pod of dolphins leaping through ocean waves at sunset, cinematic",
            "A cat sitting on a windowsill watching rain droplets on the glass",
        ]),
        ("Sci-Fi & Fantasy", [
            "A futuristic city with flying vehicles and neon lights at night, cyberpunk style",
            "A magical forest with glowing mushrooms and floating particles, fantasy atmosphere",
            "A spaceship entering hyperspace with stars stretching into lines, cinematic",
            "An ancient dragon soaring over a medieval castle in the mountains, epic scale",
        ]),
        ("Abstract & Artistic", [
            "Colorful paint drops falling into water creating abstract patterns, slow motion",
            "Geometric shapes morphing and transforming, smooth animation, minimalist style",
            "Ink dissolving in water creating flowing organic patterns, black and white",
            "Light painting streaks creating spiral patterns in darkness, long exposure effect",
        ]),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Prompt Ideas")
                    .font(.title3)
                    .bold()

                ForEach(Self.suggestions, id: \.category) { section in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(section.category)
                            .font(.headline)
                            .foregroundStyle(.secondary)

                        ForEach(section.prompts, id: \.self) { prompt in
                            Button {
                                selectedPrompt = prompt
                            } label: {
                                HStack {
                                    Text(prompt)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.leading)
                                        .font(.callout)
                                    Spacer()
                                    Image(systemName: "arrow.right.circle")
                                        .foregroundStyle(.secondary)
                                }
                                .padding(8)
                                .background(Color(.controlBackgroundColor))
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding()
        }
    }
}
