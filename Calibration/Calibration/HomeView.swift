// HomeView.swift
//
// Mirrors the teammate's HomeScreen.js — a simple landing screen with
// a title and a "Go to Camera" button that navigates to the AR capture view.
// Drop this file into the Xcode project (same group as ARCaptureSession.swift).

import SwiftUI

struct HomeView: View {
    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Text("Home Screen")
                .font(.largeTitle)
                .fontWeight(.bold)

            NavigationLink("Go to Camera") {
                // Opens the existing AR capture + calibration view full-screen,
                // nav bar hidden so the HUD shows exactly as before.
                ContentView()
                    .navigationBarHidden(true)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .navigationBarHidden(true)
    }
}
