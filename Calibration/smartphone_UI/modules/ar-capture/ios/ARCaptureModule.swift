// ARCaptureModule.swift
// Expo native module that bridges ARKit capture functionality to React Native.
// Provides: AR session management, video recording, pitch/calibration events,
// and MeTRAbs server upload.

import ExpoModulesCore
import ARKit

public class ARCaptureModule: Module {

    private lazy var captureSession: ARCaptureSessionManager = {
        let session = ARCaptureSessionManager()
        session.delegate = self
        return session
    }()

    public func definition() -> ModuleDefinition {
        Name("ARCapture")

        // Events emitted to JavaScript
        Events(
            "onPitchUpdate",
            "onFrameCountUpdate",
            "onRecordingFinished",
            "onTrackingStateChange",
            "onUploadProgress"
        )

        OnDestroy {
            self.captureSession.stopSession()
        }

        // Check if ARKit is available (false on simulator)
        Function("isAvailable") { () -> Bool in
            return ARWorldTrackingConfiguration.isSupported
        }

        // Start the AR session
        Function("startSession") {
            self.captureSession.startSession()
        }

        // Stop the AR session
        Function("stopSession") {
            self.captureSession.stopSession()
        }

        // Start recording video
        Function("startRecording") { (fps: Int, maxDuration: Int) in
            self.captureSession.targetFPS = fps
            self.captureSession.maxDuration = maxDuration
            self.captureSession.startRecording()
        }

        // Stop recording -- result comes via onRecordingFinished event
        Function("stopRecording") {
            self.captureSession.stopRecording(completion: nil)
        }

        // Set current pitch as zero reference
        Function("calibrate") {
            self.captureSession.calibrate()
        }

        // Upload video to MeTRAbs server
        Function("uploadToServer") { (videoUri: String, serverUrl: String) in
            self.captureSession.uploadToServer(videoUri: videoUri, serverUrl: serverUrl)
        }

        // Get current tracking state
        Function("getTrackingState") { () -> String in
            return self.captureSession.currentTrackingStateString()
        }

        // Native view component
        View(ARCaptureView.self) {
            Events("onLayout")

            Prop("isActive") { (view: ARCaptureView, isActive: Bool?) in
                if isActive == true {
                    view.connectSession(self.captureSession.arSession)
                }
            }
        }
    }
}

// MARK: - ARCaptureSessionDelegate

extension ARCaptureModule: ARCaptureSessionDelegate {

    func didUpdatePitch(_ pitch: Double, isLevel: Bool) {
        sendEvent("onPitchUpdate", [
            "pitch": pitch,
            "isLevel": isLevel
        ])
    }

    func didUpdateFrameCount(_ count: Int) {
        sendEvent("onFrameCountUpdate", [
            "count": count
        ])
    }

    func didFinishRecording(_ result: [String: Any]) {
        sendEvent("onRecordingFinished", result)
    }

    func didChangeTrackingState(_ state: String, message: String) {
        sendEvent("onTrackingStateChange", [
            "state": state,
            "message": message
        ])
    }

    func didUpdateUploadProgress(_ progress: Double) {
        sendEvent("onUploadProgress", [
            "progress": progress
        ])
    }
}
