import SwiftUI

@main
struct CalibrationApp: App {
    var body: some Scene {
        WindowGroup {
            // NavigationStack wraps HomeView so the "Go to Camera" button
            // uses standard iOS stack navigation — mirrors the React Native setup.
            NavigationStack {
                HomeView()
            }
        }
    }
}
