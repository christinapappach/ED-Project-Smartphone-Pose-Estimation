// ARCaptureSession.swift
// ARKit Calibration Capture Layer for Metrabs Backend
//
// Memory architecture:
//   DURING RECORDING  — each frame is written to a temp folder on disk immediately.
//                       Zero frames held in RAM. No crash risk.
//   AFTER STOP        — session appears in Sessions list as "Pending" instantly.
//   BACKGROUND        — user taps Process in Sessions list; frames are read one at
//                       a time from disk, upsampled, encoded to final JSON, then the
//                       temp folder is deleted. RAM never holds more than one frame.
//
// Info.plist required:
//   NSCameraUsageDescription     → "Required for AR capture"
//   NSMicrophoneUsageDescription → "Required for AR session"

import SwiftUI
import ARKit
import AVFoundation
import RealityKit
import Combine
import simd

// MARK: - Models

struct SessionMetadata: Codable, Sendable {
    var subjectName: String = ""
    var sessionName: String = ""
}

/// Tiny manifest written to disk when recording stops.
struct SessionManifest: Codable, Sendable {
    let id: String
    let metadata: SessionMetadata
    let capturedFPS: Int
    let frameCount: Int
    var processed: Bool = false
}

struct RecordedFrame: Codable, Sendable {
    let frameID: Int
    let imageBase64: String
    let intrinsics: [[Float]]
    let cameraTransform: [[Float]]
    let imageWidth: Int
    let imageHeight: Int
    let depthWidth: Int?
    let depthHeight: Int?
    let depthBase64: String?
}

struct SessionFile: Codable, Sendable {
    let metadata: SessionMetadata
    let capturedFPS: Int
    let frames: [RecordedFrame]
}

/// One raw frame written to disk during recording.
private struct DiskFrame: Codable, Sendable {
    let frameID: Int
    let imageWidth: Int
    let imageHeight: Int
    let intrinsics: [[Float]]
    let cameraTransform: [[Float]]
    let rawDepthWidth: Int?
    let rawDepthHeight: Int?
}

// MARK: - Tracking Quality

enum TrackingQuality {
    case normal, limited(String), notAvailable
    var isWarning: Bool { if case .normal = self { return false }; return true }
    var message: String {
        switch self {
        case .normal:           return ""
        case .limited(let r):  return "⚠️ Tracking limited: \(r)"
        case .notAvailable:    return "⚠️ Tracking unavailable"
        }
    }
}

// MARK: - Capture Manager

final class CaptureManager: NSObject, ObservableObject, ARSessionDelegate {

    @Published var isRecording        = false
    @Published var frameCount         = 0
    @Published var saveStatus         = ""
    @Published var pitchDegrees       = 0.0
    @Published var isLevel            = false
    @Published var trackingQuality    = TrackingQuality.normal
    @Published var elapsedSeconds     = 0
    @Published var targetFPS:    Int  = 30
    @Published var maxDuration:  Int  = 0
    @Published var pendingMetadata    = SessionMetadata()

    // ⚠️ Set this to your MeTRAbs server address before building
    var serverURL = "http://YOUR_SERVER_IP:8000/analyze"

    let arSession = ARSession()

    private var currentSessionID      = ""
    private var currentTempDir: URL?  = nil
    private var frameIndex            = 0
    private let targetPitchDegrees    = 90.0
    private var calibrationOffset     = 0.0
    private var lastCapturedTime      = 0.0
    private var durationTimer: Timer? = nil
    private let encodeQueue = DispatchQueue(
        label: "com.arcapture.encode", qos: .userInitiated)

    /// Shared CIContext — creating one per frame was causing ~100ms overhead each time.
    /// CIContext is thread-safe and expensive to create, so we make one and reuse it.
    nonisolated(unsafe) private let sharedCIContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Caps frames in-flight. Value of 2 means at most 2 CVPixelBuffers held
    /// simultaneously — ARKit's pool has ~3, so this leaves one free for ARKit itself.
    /// If both slots are full we drop the incoming frame — far better than crashing.
    private let encodeSemaphore = DispatchSemaphore(value: 2)

    // MARK: - Session

    func startSession() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        arSession.delegate = self
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopSession() { arSession.pause() }

    // MARK: - Recording

    func startRecording() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        currentSessionID = formatter.string(from: Date())

        // Create temp folder for this session's raw frames
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let tmp  = docs.appendingPathComponent("raw_\(currentSessionID)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        currentTempDir = tmp

        frameIndex       = 0
        frameCount       = 0
        saveStatus       = ""
        lastCapturedTime = 0
        elapsedSeconds   = 0
        isRecording      = true

        durationTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.elapsedSeconds += 1
            if self.maxDuration > 0 && self.elapsedSeconds >= self.maxDuration {
                self.stopRecordingAndSave()
            }
        }
    }

    func stopRecordingAndSave() {
        guard isRecording else { return }
        isRecording = false
        durationTimer?.invalidate()
        durationTimer = nil
        saveStatus = "Saving…"

        // Wait for encode queue to drain, then write manifest
        let sid      = currentSessionID
        let tmpDir   = currentTempDir
        let meta     = pendingMetadata
        let fps      = targetFPS
        let count    = frameIndex

        encodeQueue.async { [weak self] in
            guard let self, let tmpDir else { return }

            let manifest = SessionManifest(
                id: sid,
                metadata: meta,
                capturedFPS: fps,
                frameCount: count,
                processed: false
            )
            if let data = try? JSONEncoder().encode(manifest) {
                try? data.write(to: tmpDir.appendingPathComponent("manifest.json"))
            }

            // Save first frame as thumbnail if available
            let thumbSrc = tmpDir.appendingPathComponent("frame_00000.jpg")
            if let jpegData = try? Data(contentsOf: thumbSrc) {
                self.writeThumbnail(jpegData: jpegData, sessionID: sid)
            }

            DispatchQueue.main.async {
                self.saveStatus = "✓ Ready to process"
            }
        }
    }

    func calibrate() { calibrationOffset = pitchDegrees - targetPitchDegrees }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let t        = frame.camera.transform
        let sinPitch = -t.columns.2.y
        let cosPitch =  t.columns.2.z
        let corrected = Double(atan2(sinPitch, cosPitch)) * (180 / .pi) - calibrationOffset

        let quality: TrackingQuality
        switch frame.camera.trackingState {
        case .normal: quality = .normal
        case .limited(let r):
            let msg: String
            switch r {
            case .initializing:         msg = "initializing"
            case .relocalizing:         msg = "relocalizing"
            case .excessiveMotion:      msg = "excessive motion"
            case .insufficientFeatures: msg = "insufficient features"
            @unknown default:           msg = "unknown"
            }
            quality = .limited(msg)
        case .notAvailable: quality = .notAvailable
        @unknown default:   quality = .notAvailable
        }

        DispatchQueue.main.async {
            self.pitchDegrees    = corrected
            self.isLevel         = abs(corrected - self.targetPitchDegrees) <= 1.0
            self.trackingQuality = quality
        }

        guard isRecording, let tmpDir = currentTempDir else { return }

        let now = frame.timestamp
        guard now - lastCapturedTime >= 1.0 / Double(targetFPS) else { return }
        lastCapturedTime = now

        // Grab all data from frame on this thread
        let rgbBuffer   = frame.capturedImage
        let imageWidth  = CVPixelBufferGetWidth(rgbBuffer)
        let imageHeight = CVPixelBufferGetHeight(rgbBuffer)

        let intr = frame.camera.intrinsics
        let intrinsics: [[Float]] = [
            [intr.columns.0.x, intr.columns.1.x, intr.columns.2.x],
            [intr.columns.0.y, intr.columns.1.y, intr.columns.2.y],
            [intr.columns.0.z, intr.columns.1.z, intr.columns.2.z]
        ]
        let tf = frame.camera.transform
        let cameraTransform: [[Float]] = [
            [tf.columns.0.x, tf.columns.1.x, tf.columns.2.x, tf.columns.3.x],
            [tf.columns.0.y, tf.columns.1.y, tf.columns.2.y, tf.columns.3.y],
            [tf.columns.0.z, tf.columns.1.z, tf.columns.2.z, tf.columns.3.z],
            [tf.columns.0.w, tf.columns.1.w, tf.columns.2.w, tf.columns.3.w]
        ]

        var depthBytes:  Data? = nil
        var depthWidth:  Int?  = nil
        var depthHeight: Int?  = nil
        if let dm = frame.sceneDepth?.depthMap {
            depthWidth  = CVPixelBufferGetWidth(dm)
            depthHeight = CVPixelBufferGetHeight(dm)
            depthBytes  = extractBytes(from: dm)
        }

        // Check semaphore WITHOUT blocking — if all 3 slots are full,
        // drop this frame entirely rather than stalling the AR thread.
        // Dropping occasional frames under load is far better than crashing.
        guard encodeSemaphore.wait(timeout: .now()) == .success else { return }

        // Pass the CVPixelBuffer directly — no YUV copy.
        // Swift ARC retains it automatically; encodeQueue releases when done.
        let pixelBuffer = frame.capturedImage
        let idx         = frameIndex
        frameIndex += 1

        encodeQueue.async { [weak self] in
            guard let self else { self?.encodeSemaphore.signal(); return }
            defer { self.encodeSemaphore.signal() }

            guard let jpegData = self.pixelBufferToJPEG(pixelBuffer,
                                                         context: self.sharedCIContext)
            else { return }

            let frameID = String(format: "%05d", idx)
            try? jpegData.write(to: tmpDir.appendingPathComponent("frame_\(frameID).jpg"))

            if let db = depthBytes {
                try? db.write(to: tmpDir.appendingPathComponent("depth_\(frameID).bin"))
            }

            let meta = DiskFrame(
                frameID: idx,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                intrinsics: intrinsics,
                cameraTransform: cameraTransform,
                rawDepthWidth: depthWidth,
                rawDepthHeight: depthHeight
            )
            if let md = try? JSONEncoder().encode(meta) {
                try? md.write(to: tmpDir.appendingPathComponent("meta_\(frameID).json"))
            }

            if idx % 5 == 0 {
                let count = idx + 1
                DispatchQueue.main.async { self.frameCount = count }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.saveStatus = "AR Error: \(error.localizedDescription)" }
    }

    // MARK: - Pixel Helpers

    nonisolated private func extractBytes(from buf: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        return Data(bytes: base, count: CVPixelBufferGetDataSize(buf))
    }

    nonisolated private func pixelBufferToJPEG(_ pixelBuffer: CVPixelBuffer,
                                               context: CIContext) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cg = context.createCGImage(ci, from: ci.extent) else { return nil }
        // 0.6 quality — visually fine for pose detection, ~30% faster to encode than 0.8
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.6)
    }

    /// Converts a CGImage into a CVPixelBuffer in 32BGRA format.
    /// Used when re-encoding saved JPEG frames into the MP4.
    nonisolated private func cgImageToPixelBuffer(_ cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey     as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey               as String: width,
            kCVPixelBufferHeightKey              as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pixelBuffer) == kCVReturnSuccess,
              let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    nonisolated private func writeThumbnail(jpegData: Data, sessionID: String) {
        guard let src = UIImage(data: jpegData) else { return }
        let scale    = 320.0 / src.size.width
        let size     = CGSize(width: 320, height: src.size.height * scale)
        let renderer = UIGraphicsImageRenderer(size: size)
        let thumb    = renderer.image { _ in src.draw(in: CGRect(origin: .zero, size: size)) }
        let docs     = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try? thumb.jpegData(compressionQuality: 0.6)?
            .write(to: docs.appendingPathComponent("thumb_\(sessionID).jpg"))
    }

    // MARK: - Process Session (called from SessionListView)
    //
    // Reads one frame at a time from disk — RAM usage stays flat no matter
    // how long the session is. Yields every 10 frames so iOS can breathe.

    func processSession(
        manifest: SessionManifest,
        onProgress: @escaping (Double, String) -> Void,
        onDone: @escaping (URL?) -> Void
    ) {
        let docs   = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let tmpDir = docs.appendingPathComponent("raw_\(manifest.id)", isDirectory: true)

        Task.detached(priority: .background) { [weak self] in
            guard let self else { return }

            let total     = manifest.frameCount
            let startTime = Date()
            let fileURL   = docs.appendingPathComponent("session_\(manifest.id).json")

            // Stream-write JSON to disk frame by frame.
            // RAM never holds more than one frame at a time — no accumulation.
            do {
                // Open a file handle for writing
                FileManager.default.createFile(atPath: fileURL.path, contents: nil)
                let handle = try FileHandle(forWritingTo: fileURL)
                defer { try? handle.close() }

                // Write JSON header manually
                let metaEncoder = JSONEncoder()
                let metaJSON    = try metaEncoder.encode(manifest.metadata)
                let metaStr     = String(data: metaJSON, encoding: .utf8) ?? "{}"

                let header = """
                {
                  "metadata": \(metaStr),
                  "capturedFPS": \(manifest.capturedFPS),
                  "frames": [
                """
                handle.write(Data(header.utf8))

                var writtenCount = 0

                for i in 0..<total {

                    // Progress + ETA
                    let progress = Double(i + 1) / Double(total)
                    let elapsed  = Date().timeIntervalSince(startTime)
                    var etaStr   = "Calculating…"
                    if i > 2 && elapsed > 0 {
                        let sPerFrame = elapsed / Double(i + 1)
                        let rem       = sPerFrame * Double(total - i - 1)
                        let m = Int(rem) / 60, s = Int(rem) % 60
                        etaStr = m > 0 ? "~\(m) min \(s) sec remaining" : "~\(s) sec remaining"
                    }
                    onProgress(progress, etaStr)

                    let frameID = String(format: "%05d", i)

                    // Read frame meta — tiny JSON, fine to load
                    guard let metaData = try? Data(contentsOf: tmpDir.appendingPathComponent("meta_\(frameID).json")),
                          let meta = try? JSONDecoder().decode(DiskFrame.self, from: metaData)
                    else { continue }

                    // Read JPEG
                    guard let jpegData = try? Data(contentsOf: tmpDir.appendingPathComponent("frame_\(frameID).jpg"))
                    else { continue }

                    // Read + upsample depth
                    var finalDepthBase64: String? = nil
                    var finalDepthWidth:  Int?    = nil
                    var finalDepthHeight: Int?    = nil

                    let depthURL = tmpDir.appendingPathComponent("depth_\(frameID).bin")
                    if let depthBytes = try? Data(contentsOf: depthURL),
                       let rw = meta.rawDepthWidth, let rh = meta.rawDepthHeight {
                        if let up = self.upsampleDepth(
                            rawBytes: depthBytes, rawWidth: rw, rawHeight: rh,
                            guidanceJpeg: jpegData,
                            targetWidth: meta.imageWidth, targetHeight: meta.imageHeight
                        ) {
                            finalDepthBase64 = up.base64EncodedString()
                            finalDepthWidth  = meta.imageWidth
                            finalDepthHeight = meta.imageHeight
                        } else {
                            finalDepthBase64 = depthBytes.base64EncodedString()
                            finalDepthWidth  = rw
                            finalDepthHeight = rh
                        }
                    }

                    // Build this frame's JSON and stream it straight to disk
                    // jpegData and depth strings are released at end of this scope
                    let frame = RecordedFrame(
                        frameID:         meta.frameID,
                        imageBase64:     jpegData.base64EncodedString(),
                        intrinsics:      meta.intrinsics,
                        cameraTransform: meta.cameraTransform,
                        imageWidth:      meta.imageWidth,
                        imageHeight:     meta.imageHeight,
                        depthWidth:      finalDepthWidth,
                        depthHeight:     finalDepthHeight,
                        depthBase64:     finalDepthBase64
                    )

                    if let frameData = try? JSONEncoder().encode(frame),
                       let frameStr  = String(data: frameData, encoding: .utf8) {
                        let separator = writtenCount == 0 ? "\n    " : ",\n    "
                        handle.write(Data((separator + frameStr).utf8))
                        writtenCount += 1
                    }

                    // frame, jpegData, depth strings all go out of scope here — RAM freed
                    // Yield every 5 frames so iOS can breathe
                    if i % 5 == 4 { await Task.yield() }
                }

                // Close JSON array and object
                handle.write(Data("\n  ]\n}\n".utf8))

                // Mark manifest processed
                var updated = manifest
                updated.processed = true
                if let md = try? JSONEncoder().encode(updated) {
                    try? md.write(to: tmpDir.appendingPathComponent("manifest.json"))
                }

                // Delete raw frame files to free disk space, keep manifest
                let rawFiles = (try? FileManager.default.contentsOfDirectory(
                    at: tmpDir, includingPropertiesForKeys: nil)) ?? []
                for f in rawFiles where f.lastPathComponent != "manifest.json" {
                    try? FileManager.default.removeItem(at: f)
                }

                onDone(fileURL)

            } catch {
                onDone(nil)
            }
        }
    }

    // MARK: - Process to MP4 + Upload to MeTRAbs

    /// Replaces processSession.
    /// Reads the saved JPEG frames from disk, encodes them into an MP4,
    /// then uploads the video + camera intrinsics to your MeTRAbs server.
    /// No depth upsampling. No giant JSON. File size: ~1–3 GB instead of 57 GB.
    func processToMP4AndUpload(
        manifest: SessionManifest,
        onProgress: @escaping (Double, String) -> Void,
        onDone: @escaping (URL?) -> Void
    ) {
        let docs   = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let tmpDir = docs.appendingPathComponent("raw_\(manifest.id)", isDirectory: true)
        let mp4URL = docs.appendingPathComponent("session_\(manifest.id).mp4")

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            let total = manifest.frameCount
            guard total > 0 else { DispatchQueue.main.async { onDone(nil) }; return }

            // ── 1. Pull intrinsics from the first frame's saved metadata ───────
            // The existing capture code stored the intrinsics matrix row-major:
            //   row 0 → [fx, 0,  cx]
            //   row 1 → [0,  fy, cy]
            //   row 2 → [0,  0,  1 ]
            let firstID = String(format: "%05d", 0)
            guard let metaData  = try? Data(contentsOf: tmpDir.appendingPathComponent("meta_\(firstID).json")),
                  let firstMeta = try? JSONDecoder().decode(DiskFrame.self, from: metaData)
            else {
                print("❌ Could not read first frame metadata")
                DispatchQueue.main.async { onDone(nil) }
                return
            }

            let fx = firstMeta.intrinsics[0][0]   // focal length X
            let fy = firstMeta.intrinsics[1][1]   // focal length Y
            let cx = firstMeta.intrinsics[0][2]   // principal point X
            let cy = firstMeta.intrinsics[1][2]   // principal point Y
            let W  = firstMeta.imageWidth
            let H  = firstMeta.imageHeight
            print("📐 Intrinsics: fx=\(fx) fy=\(fy) cx=\(cx) cy=\(cy)  resolution: \(W)×\(H)")

            // ── 2. Set up AVAssetWriter ────────────────────────────────────────
            try? FileManager.default.removeItem(at: mp4URL)

            guard let writer = try? AVAssetWriter(outputURL: mp4URL, fileType: .mp4) else {
                print("❌ Could not create AVAssetWriter")
                DispatchQueue.main.async { onDone(nil) }
                return
            }

            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,   // H.265 — half the size of H.264
                AVVideoWidthKey: W,
                AVVideoHeightKey: H,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 8_000_000,  // 8 Mbps — good quality
                    AVVideoExpectedSourceFrameRateKey: manifest.capturedFPS
                ]
            ]
            let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            videoInput.expectsMediaDataInRealTime = false
            // Frames were captured landscape — rotate so video plays upright
            videoInput.transform = CGAffineTransform(rotationAngle: .pi / 2)

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: videoInput,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey          as String: W,
                    kCVPixelBufferHeightKey         as String: H
                ]
            )
            writer.add(videoInput)
            writer.startWriting()
            writer.startSession(atSourceTime: .zero)

            let frameDuration = CMTime(value: 1, timescale: CMTimeScale(manifest.capturedFPS))

            // ── 3. Encode JPEG frames → MP4 ───────────────────────────────────
            let encodeStart = Date()

            for i in 0..<total {
                let frameID = String(format: "%05d", i)
                let jpegURL = tmpDir.appendingPathComponent("frame_\(frameID).jpg")

                guard let jpegData = try? Data(contentsOf: jpegURL),
                      let uiImg    = UIImage(data: jpegData),
                      let cgImg    = uiImg.cgImage,
                      let pixBuf   = self.cgImageToPixelBuffer(cgImg, width: W, height: H)
                else { continue }

                // Wait for encoder to be ready (non-real-time, rarely blocks)
                while !videoInput.isReadyForMoreMediaData { await Task.yield() }

                let pts = CMTimeMultiply(frameDuration, multiplier: Int32(i))
                adaptor.append(pixBuf, withPresentationTime: pts)

                // ETA calculation
                let elapsed   = Date().timeIntervalSince(encodeStart)
                let sPerFrame = i > 0 ? elapsed / Double(i + 1) : 0
                let remaining = sPerFrame * Double(total - i - 1)
                let etaStr    = remaining > 60
                    ? "~\(Int(remaining / 60))m \(Int(remaining) % 60)s remaining"
                    : "~\(Int(remaining))s remaining"

                let progress = Double(i + 1) / Double(total) * 0.70   // encoding = first 70%
                DispatchQueue.main.async { onProgress(progress, "Encoding \(i+1)/\(total)  \(etaStr)") }

                if i % 5 == 4 { await Task.yield() }
            }

            videoInput.markAsFinished()
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                writer.finishWriting { c.resume() }
            }

            if let attrs = try? FileManager.default.attributesOfItem(atPath: mp4URL.path),
               let sz = attrs[.size] as? Int64 {
                print("🎬 MP4 ready: \(String(format: "%.1f", Double(sz) / 1_048_576)) MB")
            }

            // ── 4. Upload MP4 + intrinsics to MeTRAbs ─────────────────────────
            DispatchQueue.main.async { onProgress(0.72, "Uploading to MeTRAbs…") }

            guard let serverURLObj = URL(string: self.serverURL) else {
                print("❌ Bad server URL: \(self.serverURL)")
                DispatchQueue.main.async { onDone(mp4URL) }
                return
            }

            // Build multipart body as a temp file — video streams in chunks,
            // never fully loaded into RAM.
            let boundary = "MetrabsBoundary-\(UUID().uuidString)"
            let bodyURL  = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(boundary).multipart")
            FileManager.default.createFile(atPath: bodyURL.path, contents: nil)

            guard let out = try? FileHandle(forWritingTo: bodyURL) else {
                DispatchQueue.main.async { onDone(mp4URL) }
                return
            }

            func write(_ s: String) { out.write(Data(s.utf8)) }

            // Intrinsics as plain form fields — the server plugs these straight into MeTRAbs
            for (name, value): (String, String) in [
                ("focal_length_x",    String(fx)),
                ("focal_length_y",    String(fy)),
                ("principal_point_x", String(cx)),
                ("principal_point_y", String(cy))
            ] {
                write("--\(boundary)\r\n")
                write("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
                write("\(value)\r\n")
            }

            // Video file — 4 MB chunks so RAM stays flat
            write("--\(boundary)\r\n")
            write("Content-Disposition: form-data; name=\"video\"; filename=\"\(mp4URL.lastPathComponent)\"\r\n")
            write("Content-Type: video/mp4\r\n\r\n")

            if let vid = try? FileHandle(forReadingFrom: mp4URL) {
                let chunkSize = 4 * 1024 * 1024
                while true {
                    let chunk = vid.readData(ofLength: chunkSize)
                    if chunk.isEmpty { break }
                    out.write(chunk)
                }
                vid.closeFile()
            }
            write("\r\n--\(boundary)--\r\n")
            out.closeFile()

            var request = URLRequest(url: serverURLObj)
            request.httpMethod = "POST"
            request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 600  // 10 min for large files

            do {
                let (_, response) = try await URLSession.shared.upload(for: request, fromFile: bodyURL)
                try? FileManager.default.removeItem(at: bodyURL)
                if let http = response as? HTTPURLResponse {
                    print(http.statusCode == 200 ? "✅ Upload complete" : "⚠️ Server returned HTTP \(http.statusCode)")
                }
            } catch {
                print("❌ Upload failed: \(error.localizedDescription)")
                try? FileManager.default.removeItem(at: bodyURL)
            }

            // ── 5. Mark processed + clean up raw frames ────────────────────────
            var updated = manifest
            updated.processed = true
            if let md = try? JSONEncoder().encode(updated) {
                try? md.write(to: tmpDir.appendingPathComponent("manifest.json"))
            }
            let rawFiles = (try? FileManager.default.contentsOfDirectory(
                at: tmpDir, includingPropertiesForKeys: nil)) ?? []
            for f in rawFiles where f.lastPathComponent != "manifest.json" {
                try? FileManager.default.removeItem(at: f)
            }

            DispatchQueue.main.async { onDone(mp4URL) }
        }
    }

    // MARK: - Depth Upsample

    nonisolated private func upsampleDepth(
        rawBytes: Data, rawWidth: Int, rawHeight: Int,
        guidanceJpeg: Data, targetWidth: Int, targetHeight: Int
    ) -> Data? {
        let ctx = CIContext()   // local — nonisolated, no actor access needed
        var db: CVPixelBuffer?
        let da: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey    as String: kCVPixelFormatType_DepthFloat32,
            kCVPixelBufferWidthKey              as String: rawWidth,
            kCVPixelBufferHeightKey             as String: rawHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, rawWidth, rawHeight,
                                  kCVPixelFormatType_DepthFloat32,
                                  da as CFDictionary, &db) == kCVReturnSuccess,
              let depthBuf = db else { return nil }

        CVPixelBufferLockBaseAddress(depthBuf, [])
        rawBytes.withUnsafeBytes { src in
            if let dst = CVPixelBufferGetBaseAddress(depthBuf) {
                memcpy(dst, src.baseAddress!, min(rawBytes.count, CVPixelBufferGetDataSize(depthBuf)))
            }
        }
        CVPixelBufferUnlockBaseAddress(depthBuf, [])

        let depthCI = CIImage(cvPixelBuffer: depthBuf)
        guard let rgbUI = UIImage(data: guidanceJpeg), let rgbCG = rgbUI.cgImage else { return nil }
        let rgbCI   = CIImage(cgImage: rgbCG)
        let scaled  = depthCI.transformed(by: CGAffineTransform(
            scaleX: CGFloat(targetWidth)  / CGFloat(rawWidth),
            y:      CGFloat(targetHeight) / CGFloat(rawHeight)))

        var output: CIImage = scaled
        if let filter = CIFilter(name: "CIEdgePreserveUpsampleFilter") {
            filter.setValue(scaled, forKey: kCIInputImageKey)
            filter.setValue(rgbCI,  forKey: "inputSmallImage")
            if let r = filter.outputImage { output = r }
        }

        var ob: CVPixelBuffer?
        let oa: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey    as String: kCVPixelFormatType_DepthFloat32,
            kCVPixelBufferWidthKey              as String: targetWidth,
            kCVPixelBufferHeightKey             as String: targetHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, targetWidth, targetHeight,
                                  kCVPixelFormatType_DepthFloat32,
                                  oa as CFDictionary, &ob) == kCVReturnSuccess,
              let outBuf = ob else { return nil }

        ctx.render(output, to: outBuf,
                   bounds: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
                   colorSpace: nil)

        CVPixelBufferLockBaseAddress(outBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(outBuf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(outBuf) else { return nil }
        return Data(bytes: base, count: CVPixelBufferGetDataSize(outBuf))
    }
}

// MARK: - ARView Container

struct ARViewContainer: UIViewRepresentable {
    let captureManager: CaptureManager
    func makeUIView(context: Context) -> ARView {
        let v = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        v.session = captureManager.arSession
        captureManager.startSession()
        return v
    }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

// MARK: - Recording Setup Sheet

struct RecordingSetupSheet: View {
    @ObservedObject var captureManager: CaptureManager
    let onStart: () -> Void

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Session Info")) {
                    TextField("Subject name (optional)",
                              text: $captureManager.pendingMetadata.subjectName)
                        .autocorrectionDisabled()
                    TextField("Session name (e.g. Squat Front)",
                              text: $captureManager.pendingMetadata.sessionName)
                        .autocorrectionDisabled()
                }
                Section(header: Text("Frame Rate")) {
                    Picker("FPS", selection: $captureManager.targetFPS) {
                        Text("30 fps  (smaller files)").tag(30)
                        Text("60 fps  (smoother)").tag(60)
                    }
                    .pickerStyle(.wheel).frame(height: 100)
                }
                Section(header: Text("Max Duration")) {
                    Picker("Stop after", selection: $captureManager.maxDuration) {
                        Text("Unlimited").tag(0)
                        Text("30 sec").tag(30)
                        Text("60 sec").tag(60)
                        Text("120 sec").tag(120)
                    }
                    .pickerStyle(.wheel).frame(height: 120)
                }
                Section {
                    Button(action: onStart) {
                        HStack {
                            Spacer()
                            Text("Start Recording")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.white)
                            Spacer()
                        }
                        .padding(.vertical, 8).background(Color.green).cornerRadius(10)
                    }
                }
            }
            .navigationTitle("New Session")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// MARK: - Session List View

struct SessionListView: View {
    @ObservedObject var captureManager: CaptureManager
    @State private var manifests:     [SessionManifest] = []
    @State private var finalURLs:     [String: URL]     = [:]   // id → final JSON URL
    @State private var progress:      [String: Double]  = [:]   // id → 0–1
    @State private var eta:           [String: String]  = [:]   // id → ETA string
    @State private var shareItem: URL?     = nil
    @State private var showShareSheet      = false

    var body: some View {
        NavigationView {
            Group {
                if manifests.isEmpty && finalURLs.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 48)).foregroundColor(.secondary)
                        Text("No sessions recorded yet").foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        // Pending (raw, not yet processed)
                        ForEach(manifests.filter { !$0.processed }, id: \.id) { manifest in
                            PendingSessionRow(
                                manifest: manifest,
                                progress: progress[manifest.id],
                                eta: eta[manifest.id],
                                onProcess: { startProcessing(manifest) }
                            )
                        }

                        // Processed (final JSON exists)
                        ForEach(manifests.filter { $0.processed }, id: \.id) { manifest in
                            if let url = finalURLs[manifest.id] {
                                ProcessedSessionRow(manifest: manifest, url: url) {
                                    shareItem = url
                                    showShareSheet = true
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear { loadSessions() }
            .sheet(isPresented: $showShareSheet) {
                if let url = shareItem { ShareSheet(url: url) }
            }
        }
    }

    private func loadSessions() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles)) ?? []

        var found: [SessionManifest] = []
        for dir in dirs where dir.lastPathComponent.hasPrefix("raw_") {
            let mURL = dir.appendingPathComponent("manifest.json")
            if let data = try? Data(contentsOf: mURL),
               let m = try? JSONDecoder().decode(SessionManifest.self, from: data) {
                found.append(m)

                // Check if final JSON already exists
                let jsonURL = docs.appendingPathComponent("session_\(m.id).json")
                if FileManager.default.fileExists(atPath: jsonURL.path) {
                    finalURLs[m.id] = jsonURL
                }
            }
        }
        manifests = found.sorted { $0.id > $1.id }
    }

    private func startProcessing(_ manifest: SessionManifest) {
        progress[manifest.id] = 0.001
        eta[manifest.id]      = "Starting…"

        captureManager.processToMP4AndUpload(
            manifest: manifest,
            onProgress: { p, e in
                DispatchQueue.main.async {
                    self.progress[manifest.id] = p
                    self.eta[manifest.id]      = e
                }
            },
            onDone: { url in
                DispatchQueue.main.async {
                    self.progress[manifest.id] = 0
                    if let url {
                        self.finalURLs[manifest.id] = url
                    }
                    self.loadSessions()
                }
            }
        )
    }
}

// MARK: - Session Row Views

private struct PendingSessionRow: View {
    let manifest: SessionManifest
    let progress: Double?
    let eta: String?
    let onProcess: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            ThumbnailView(sessionID: manifest.id)
                .frame(width: 56, height: 56)
                .cornerRadius(8).clipped()

            VStack(alignment: .leading, spacing: 4) {
                if !manifest.metadata.sessionName.isEmpty {
                    Text(manifest.metadata.sessionName)
                        .font(.system(size: 14, weight: .semibold)).lineLimit(1)
                }
                Text("\(manifest.frameCount) frames · \(manifest.capturedFPS) fps")
                    .font(.caption).foregroundColor(.secondary)

                if let p = progress, p > 0 {
                    // Processing ring inline
                    HStack(spacing: 8) {
                        MiniRingView(progress: p)
                            .frame(width: 28, height: 28)
                        Text(eta ?? "")
                            .font(.system(size: 11)).foregroundColor(.secondary)
                    }
                } else {
                    Button("Process & Upload", action: onProcess)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Color.blue).cornerRadius(6)
                }
            }
            Spacer()
        }
        .padding(.vertical, 4)
    }
}

private struct ProcessedSessionRow: View {
    let manifest: SessionManifest
    let url: URL
    let onShare: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            ThumbnailView(sessionID: manifest.id)
                .frame(width: 56, height: 56)
                .cornerRadius(8).clipped()

            VStack(alignment: .leading, spacing: 4) {
                if !manifest.metadata.sessionName.isEmpty {
                    Text(manifest.metadata.sessionName)
                        .font(.system(size: 14, weight: .semibold)).lineLimit(1)
                }
                Text(url.lastPathComponent)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary).lineLimit(1)
                if let size = fileSize(url) {
                    Text(size).font(.caption).foregroundColor(.secondary)
                }
            }
            Spacer()
            Button(action: onShare) {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 18)).foregroundColor(.blue)
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 4)
    }

    private func fileSize(_ url: URL) -> String? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let bytes = attrs[.size] as? Int64 else { return nil }
        let mb = Double(bytes) / 1_000_000
        return mb >= 1 ? String(format: "%.1f MB", mb) : String(format: "%d KB", bytes / 1000)
    }
}

// MARK: - Mini Ring (inside session row while processing)

private struct MiniRingView: View {
    let progress: Double
    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.3), lineWidth: 3)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(Color.blue, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.15), value: progress)
            Text("\(Int(progress * 100))")
                .font(.system(size: 7, weight: .bold, design: .monospaced))
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Thumbnail View

private struct ThumbnailView: View {
    let sessionID: String
    @State private var image: UIImage? = nil

    var body: some View {
        Group {
            if let img = image {
                Image(uiImage: img).resizable().scaledToFill()
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.2))
                    .overlay(Image(systemName: "photo").foregroundColor(.secondary))
            }
        }
        .onAppear {
            let docs  = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let thumb = docs.appendingPathComponent("thumb_\(sessionID).jpg")
            if let data = try? Data(contentsOf: thumb) { image = UIImage(data: data) }
        }
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var captureManager = CaptureManager()
    @State private var showSessionList      = false
    @State private var showSetupSheet       = false

    var body: some View {
        ZStack {
            ARViewContainer(captureManager: captureManager)
                .ignoresSafeArea()

            VStack {
                // Top HUD
                VStack(spacing: 6) {
                    HStack {
                        Spacer()
                        Button { showSessionList = true } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "folder")
                                Text("Sessions").font(.system(size: 13, weight: .medium))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(.black.opacity(0.55)).clipShape(Capsule())
                        }
                        .padding(.trailing, 16)
                    }

                    Text(String(format: "Tilt: %.1f°", captureManager.pitchDegrees))
                        .font(.system(size: 16, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(.black.opacity(0.55)).clipShape(Capsule())

                    if captureManager.isLevel {
                        Text("LEVEL")
                            .font(.system(size: 14, weight: .bold)).foregroundColor(.green)
                            .padding(.horizontal, 14).padding(.vertical, 5)
                            .background(Color.green.opacity(0.2))
                            .overlay(Capsule().stroke(Color.green, lineWidth: 1.5))
                            .clipShape(Capsule())
                    }

                    AngleDial(degrees: captureManager.pitchDegrees)
                        .frame(width: 160, height: 20)
                }
                .padding(.top, 56)

                Spacer()

                // Tracking warning
                if captureManager.isRecording && captureManager.trackingQuality.isWarning {
                    Text(captureManager.trackingQuality.message)
                        .font(.system(size: 13, weight: .semibold)).foregroundColor(.white)
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .background(Color.orange.opacity(0.85)).clipShape(Capsule())
                        .padding(.bottom, 4)
                }

                // REC counter
                if captureManager.isRecording {
                    HStack(spacing: 8) {
                        Text("● REC  \(captureManager.frameCount) frames")
                        if captureManager.maxDuration > 0 {
                            Text("· \(captureManager.elapsedSeconds)s / \(captureManager.maxDuration)s")
                        } else {
                            Text("· \(captureManager.elapsedSeconds)s")
                        }
                    }
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(.red)
                    .padding(.horizontal, 14).padding(.vertical, 6)
                    .background(.black.opacity(0.6)).clipShape(Capsule())
                }

                // Save status
                if !captureManager.saveStatus.isEmpty && !captureManager.isRecording {
                    Text(captureManager.saveStatus)
                        .font(.system(size: 13)).foregroundColor(.white)
                        .padding(.horizontal, 16).padding(.vertical, 8)
                        .background(.black.opacity(0.6)).cornerRadius(10)
                        .padding(.horizontal, 20)
                }

                // Controls
                HStack(spacing: 16) {
                    ControlButton(label: "Calibrate", color: .orange) {
                        captureManager.calibrate()
                    }
                    if captureManager.isRecording {
                        ControlButton(label: "Stop & Save", color: .red) {
                            captureManager.stopRecordingAndSave()
                        }
                    } else {
                        ControlButton(label: "Start Recording", color: .green) {
                            showSetupSheet = true
                        }
                    }
                }
                .padding(.bottom, 48)
            }
        }
        .sheet(isPresented: $showSessionList) {
            SessionListView(captureManager: captureManager)
        }
        .sheet(isPresented: $showSetupSheet) {
            RecordingSetupSheet(captureManager: captureManager) {
                showSetupSheet = false
                captureManager.startRecording()
            }
        }
        .onDisappear { captureManager.stopSession() }
    }
}

// MARK: - Supporting Views

private struct AngleDial: View {
    let degrees: Double
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .center) {
                RoundedRectangle(cornerRadius: 4).fill(.white.opacity(0.2)).frame(height: 6)
                Rectangle().fill(Color.green.opacity(0.8)).frame(width: 2, height: 16)
                let offset = CGFloat(degrees - 90.0) * (geo.size.width / 60.0)
                Rectangle().fill(Color.white).frame(width: 2, height: 12)
                    .offset(x: offset.clamped(to: -geo.size.width / 2 ... geo.size.width / 2))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

private struct ControlButton: View {
    let label: String
    let color: Color
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 15, weight: .semibold)).foregroundColor(.white)
                .padding(.horizontal, 20).padding(.vertical, 12)
                .background(color.opacity(0.85)).cornerRadius(12)
                .shadow(color: color.opacity(0.5), radius: 6, x: 0, y: 3)
        }
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
