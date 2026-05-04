// ARCaptureManager.swift
// Drop this into your Xcode project.
// It handles:
//   1. ARKit session at 60fps with LiDAR
//   2. Writing frames DIRECTLY to MP4 as they come in (no JSON, no memory bloat)
//   3. Pulling camera intrinsics (fx, fy, cx, cy) from the first frame

import ARKit
import AVFoundation

class ARCaptureManager: NSObject {

    // MARK: - Public
    /// Called when recording finishes. Gives you the video file + camera numbers.
    var onRecordingFinished: ((URL, Float, Float, Float, Float) -> Void)?

    /// Camera intrinsics — available after first frame arrives
    private(set) var fx: Float = 0
    private(set) var fy: Float = 0
    private(set) var cx: Float = 0
    private(set) var cy: Float = 0

    // MARK: - Private
    private var session: ARSession!
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?

    private var isRecording = false
    private var firstFrameTimestamp: Double?
    private var outputURL: URL?
    private var intrinsicsCaptured = false

    // MARK: - AR Session Setup

    func startSession() {
        session = ARSession()
        session.delegate = self

        let config = ARWorldTrackingConfiguration()

        // Pick 60fps video format
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        if let format60 = formats.first(where: { $0.framesPerSecond == 60 }) {
            config.videoFormat = format60
            print("📷 Using format: \(format60.imageResolution) @ 60fps")
        } else {
            print("⚠️ 60fps not available, using default")
        }

        // Enable LiDAR depth if device supports it (iPhone 12 Pro and later)
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = .sceneDepth
            print("✅ LiDAR depth enabled")
        }

        session.run(config)
    }

    // MARK: - Recording Control

    func startRecording() {
        // Create output file path in Documents
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let filename = "metrabs_\(Int(Date().timeIntervalSince1970)).mp4"
        outputURL = docs.appendingPathComponent(filename)

        guard let url = outputURL else { return }

        // Remove any old file at that path
        try? FileManager.default.removeItem(at: url)

        do {
            assetWriter = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            print("❌ Could not create AVAssetWriter: \(error)")
            return
        }

        // Grab actual resolution from the current AR format
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        let format = formats.first(where: { $0.framesPerSecond == 60 })
        let width  = Int(format?.imageResolution.width  ?? 1920)
        let height = Int(format?.imageResolution.height ?? 1440)

        // HEVC (H.265) gives roughly half the file size of H.264 at same quality
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 8_000_000,        // 8 Mbps — good quality, reasonable size
                AVVideoExpectedSourceFrameRateKey: 60
            ]
        ]

        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput?.expectsMediaDataInRealTime = true

        // ARKit outputs landscape frames even when phone is portrait.
        // This rotation makes the video upright when the person is standing
        // in front of the camera in portrait orientation.
        videoInput?.transform = CGAffineTransform(rotationAngle: .pi / 2)

        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput!,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )

        if let videoInput = videoInput {
            assetWriter?.add(videoInput)
        }

        assetWriter?.startWriting()
        assetWriter?.startSession(atSourceTime: .zero)

        // Reset state
        firstFrameTimestamp = nil
        intrinsicsCaptured = false
        fx = 0; fy = 0; cx = 0; cy = 0
        isRecording = true

        print("🔴 Recording started → \(filename)")
    }

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false

        videoInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            guard let self = self, let url = self.outputURL else { return }

            if let error = self.assetWriter?.error {
                print("❌ AssetWriter error: \(error)")
                return
            }

            // Report file size so you can see the improvement
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
               let size = attrs[.size] as? Int64 {
                let mb = Double(size) / 1_048_576
                print("✅ Recording saved: \(url.lastPathComponent) (\(String(format: "%.1f", mb)) MB)")
            }

            DispatchQueue.main.async {
                // Hand off: video file URL + the 4 camera numbers MeTRAbs needs
                self.onRecordingFinished?(url, self.fx, self.fy, self.cx, self.cy)
            }
        }
    }
}

// MARK: - ARSessionDelegate

extension ARCaptureManager: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording,
              let videoInput = videoInput,
              let adaptor = pixelBufferAdaptor,
              videoInput.isReadyForMoreMediaData else { return }

        // ── Capture intrinsics from the first frame only ─────────────────────
        // ARKit's intrinsics matrix is column-major (simd_float3x3):
        //   columns.0 = [fx,  0,  0]
        //   columns.1 = [ 0, fy,  0]
        //   columns.2 = [cx, cy,  1]
        if !intrinsicsCaptured {
            let m = frame.camera.intrinsics
            fx = m.columns.0.x   // focal length X (pixels)
            fy = m.columns.1.y   // focal length Y (pixels)
            cx = m.columns.2.x   // principal point X
            cy = m.columns.2.y   // principal point Y
            intrinsicsCaptured = true
            print("📐 Intrinsics: fx=\(fx) fy=\(fy) cx=\(cx) cy=\(cy)")
        }

        // ── Write frame to MP4 ────────────────────────────────────────────────
        let timestamp = frame.timestamp
        if firstFrameTimestamp == nil { firstFrameTimestamp = timestamp }
        let elapsed = timestamp - firstFrameTimestamp!
        let presentationTime = CMTime(seconds: elapsed, preferredTimescale: 600)

        // frame.capturedImage is a CVPixelBuffer — write it directly, no copy
        adaptor.append(frame.capturedImage, withPresentationTime: presentationTime)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("❌ AR Session error: \(error)")
    }
}
