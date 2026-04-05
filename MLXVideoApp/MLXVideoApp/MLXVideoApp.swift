// MLXVideoApp.swift - Main app entry point
// macOS 26 SwiftUI app for MLX Video generation

import SwiftUI

@main
struct MLXVideoApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appState)
        }
        .windowStyle(.automatic)
        .defaultSize(width: 1200, height: 800)

        #if compiler(>=6.0)
        Settings {
            SettingsView()
                .environment(appState)
        }
        #endif
    }
}
