// ARCaptureSession.swift
// Adapted from the standalone Swift ARKit pipeline for use as an Expo native module.
// Combines ARCaptureManager's direct-to-MP4 recording with CaptureManager's
// pitch calculation, calibration, and tracking quality monitoring.
// Captures comprehensive camera metadata for MeTRAbs pose estimation.

import ARKit
import AVFoundation
import RealityKit

protocol ARCaptureSessionDelegate: AnyObject {
    func didUpdatePitch(_ pitch: Double, isLevel: Bool)
    func didUpdateFrameCount(_ count: Int)
    func didFinishRecording(_ result: [String: Any])
    func didChangeTrackingState(_ state: String, message: String)
    func didUpdateUploadProgress(_ progress: Double)
}

final class ARCaptureSessionManager: NSObject {

    weak var delegate: ARCaptureSessionDelegate?

    let arSession = ARSession()

    // Camera intrinsics (captured from first frame)
    private(set) var fx: Float = 0
    private(set) var fy: Float = 0
    private(set) var cx: Float = 0
    private(set) var cy: Float = 0
    private(set) var intrinsicsMatrix: [[Float]] = []

    // Full camera metadata captured during recording
    private(set) var imageResolutionWidth: Int = 0
    private(set) var imageResolutionHeight: Int = 0
    private(set) var videoFormatFPS: Int = 0
    private(set) var hasDepthData: Bool = false
    private(set) var depthWidth: Int = 0
    private(set) var depthHeight: Int = 0
    private(set) var hasLiDAR: Bool = false
    private(set) var exposureDuration: Double = 0
    private(set) var exposureOffset: Float = 0
    private(set) var lensPosition: Float = 0
    private(set) var eulerAnglesX: Float = 0
    private(set) var eulerAnglesY: Float = 0
    private(set) var eulerAnglesZ: Float = 0
    private(set) var cameraTransformMatrix: [[Float]] = []
    private(set) var projectionMatrix: [[Float]] = []
    private(set) var lightEstimateIntensity: Double = 0
    private(set) var lightEstimateTemperature: Double = 0

    // Per-frame camera transforms (for full trajectory)
    private var frameTransforms: [[String: Any]] = []
    private var capturePerFrameTransforms = true

    // Pitch / calibration
    private var calibrationOffset = 0.0
    private let targetPitchDegrees = 90.0

    // Recording state
    private(set) var isRecording = false
    var targetFPS: Int = 60
    var maxDuration: Int = 60

    // AVAssetWriter (direct-to-MP4)
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var firstFrameTimestamp: Double?
    private var outputURL: URL?
    private var intrinsicsCaptured = false
    private var frameCount = 0
    private var lastCapturedTime = 0.0
    private var elapsedSeconds = 0
    private var durationTimer: Timer?

    // Depth-data file (binary blob: 12B header [width:u32 LE][height:u32 LE][frame_count:u32 LE]
    // followed by frame_count × width × height × float32 mm depth values).
    // Only populated on devices with LiDAR; nil on iPad / non-LiDAR phones.
    private var depthURL: URL?
    private var depthFileHandle: FileHandle?
    private var depthFrameCount = 0

    // The per-recording folder (Documents/recording_<ts>/) that bundles
    // video.mp4, depth.bin, intrinsics.json (JS), and output.csv (server).
    private var recordingFolderURL: URL?

    // Audio capture (recorded into the same MP4 alongside the video track).
    // ARKit doesn't capture audio — we run a parallel AVCaptureSession with the
    // microphone only and append audio samples into the asset writer.
    private var audioInput: AVAssetWriterInput?
    private var audioCaptureSession: AVCaptureSession?
    private let audioOutputQueue = DispatchQueue(label: "metrabs.audio.output", qos: .userInitiated)
    private(set) var hasAudio = false

    // Completion handler for stopRecording async flow
    private var recordingCompletion: (([String: Any]) -> Void)?

    // Uploader
    private lazy var uploader = MetrabsUploader()

    // MARK: - Session Lifecycle

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            print("ARKit world tracking not supported on this device")
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity

        // Force 60fps format — pick highest resolution at 60fps
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        let formats60 = formats.filter { $0.framesPerSecond == 60 }
        if let best60 = formats60.max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = best60
            videoFormatFPS = 60
        } else if let best = formats.first {
            config.videoFormat = best
            videoFormatFPS = best.framesPerSecond
        }

        // Enable LiDAR depth if available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
            hasLiDAR = true
        }

        // Enable smoothed scene depth if available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }

        // Enable auto-focus for best quality
        config.isAutoFocusEnabled = true

        // Enable light estimation
        config.isLightEstimationEnabled = true

        arSession.delegate = self
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopSession() {
        if isRecording { stopRecording(completion: nil) }
        arSession.pause()
    }

    // MARK: - Recording (Direct to MP4)

    func startRecording() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        // Each recording lives in its own folder so all related files
        // (video.mp4, depth.bin, intrinsics.json, output.csv) stay grouped.
        let recordingId = "recording_\(Int(Date().timeIntervalSince1970))"
        let folderURL = docs.appendingPathComponent(recordingId, isDirectory: true)
        try? FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true, attributes: nil)
        self.recordingFolderURL = folderURL

        outputURL = folderURL.appendingPathComponent("video.mp4")

        guard let url = outputURL else { return }
        try? FileManager.default.removeItem(at: url)

        do {
            assetWriter = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            print("Could not create AVAssetWriter: \(error)")
            return
        }

        // Get resolution from current AR session format
        let currentFormat = arSession.configuration?.videoFormat
        let width  = Int(currentFormat?.imageResolution.width  ?? 1920)
        let height = Int(currentFormat?.imageResolution.height ?? 1440)
        imageResolutionWidth = width
        imageResolutionHeight = height

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 12_000_000,
                AVVideoExpectedSourceFrameRateKey: targetFPS
            ]
        ]

        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput?.expectsMediaDataInRealTime = true
        // Rotate landscape AR frames to portrait
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

        // ── Audio: configure shared audio session, attach AAC writer input,
        //     and spin up an AVCaptureSession to feed mic samples in. If any
        //     step fails we keep recording video-only rather than aborting.
        hasAudio = false
        audioCaptureSession = nil
        audioInput = nil
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .videoRecording, options: [.allowBluetooth, .defaultToSpeaker, .mixWithOthers])
            try session.setActive(true, options: [.notifyOthersOnDeactivation])

            let audioSettings: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 44100,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 64000,
            ]
            let aInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
            aInput.expectsMediaDataInRealTime = true
            if let writer = assetWriter, writer.canAdd(aInput) {
                writer.add(aInput)
                self.audioInput = aInput

                let captureSession = AVCaptureSession()
                if let mic = AVCaptureDevice.default(for: .audio),
                   let micInput = try? AVCaptureDeviceInput(device: mic),
                   captureSession.canAddInput(micInput) {
                    captureSession.addInput(micInput)
                    let audioOutput = AVCaptureAudioDataOutput()
                    audioOutput.setSampleBufferDelegate(self, queue: audioOutputQueue)
                    if captureSession.canAddOutput(audioOutput) {
                        captureSession.addOutput(audioOutput)
                        captureSession.startRunning()
                        self.audioCaptureSession = captureSession
                        self.hasAudio = true
                    }
                }
            }
        } catch {
            print("Audio setup failed (recording video-only): \(error)")
        }

        assetWriter?.startWriting()
        assetWriter?.startSession(atSourceTime: .zero)

        // Reset state
        firstFrameTimestamp = nil
        intrinsicsCaptured = false
        fx = 0; fy = 0; cx = 0; cy = 0
        intrinsicsMatrix = []
        cameraTransformMatrix = []
        projectionMatrix = []
        frameTransforms = []
        frameCount = 0
        lastCapturedTime = 0
        elapsedSeconds = 0
        isRecording = true

        // Set up depth file (only when LiDAR/scene depth is supported on this device).
        // Header (12 bytes) is written as zeros now and rewritten in stopRecording().
        depthURL = nil
        depthFileHandle = nil
        depthFrameCount = 0
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
           let folder = self.recordingFolderURL {
            let dURL = folder.appendingPathComponent("depth.bin")
            try? FileManager.default.removeItem(at: dURL)
            if FileManager.default.createFile(atPath: dURL.path, contents: nil),
               let handle = try? FileHandle(forWritingTo: dURL) {
                handle.write(Data(count: 12))   // placeholder header
                self.depthURL = dURL
                self.depthFileHandle = handle
            }
        }

        // Duration timer
        durationTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.elapsedSeconds += 1
            if self.maxDuration > 0 && self.elapsedSeconds >= self.maxDuration {
                self.stopRecording(completion: nil)
            }
        }
    }

    func stopRecording(completion: (([String: Any]) -> Void)?) {
        guard isRecording else { return }
        isRecording = false
        durationTimer?.invalidate()
        durationTimer = nil

        recordingCompletion = completion

        // Stop audio capture before finalizing — final samples may already be queued.
        audioCaptureSession?.stopRunning()
        audioCaptureSession = nil
        audioInput?.markAsFinished()

        videoInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            guard let self, let url = self.outputURL else { return }

            if let error = self.assetWriter?.error {
                print("AssetWriter error: \(error)")
                return
            }

            // Get video file size
            var fileSize: Int64 = 0
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) {
                fileSize = attrs[.size] as? Int64 ?? 0
            }

            // Finalize depth file: rewrite the 12-byte header with the actual
            // width / height / frame_count, then close the handle.
            var depthFileSize: Int64 = 0
            if let handle = self.depthFileHandle, let dURL = self.depthURL {
                handle.seek(toFileOffset: 0)
                var w = UInt32(self.depthWidth).littleEndian
                var h = UInt32(self.depthHeight).littleEndian
                var n = UInt32(self.depthFrameCount).littleEndian
                handle.write(Data(bytes: &w, count: 4))
                handle.write(Data(bytes: &h, count: 4))
                handle.write(Data(bytes: &n, count: 4))
                handle.closeFile()
                self.depthFileHandle = nil
                if let attrs = try? FileManager.default.attributesOfItem(atPath: dURL.path) {
                    depthFileSize = attrs[.size] as? Int64 ?? 0
                }
                // If we never actually captured any depth frames (e.g., no LiDAR),
                // delete the empty file so the caller doesn't try to upload it.
                if self.depthFrameCount == 0 {
                    try? FileManager.default.removeItem(at: dURL)
                    self.depthURL = nil
                    depthFileSize = 0
                }
            }

            let result: [String: Any] = [
                "videoUri": url.absoluteString,
                // Camera intrinsics
                "intrinsics": self.intrinsicsMatrix,
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                // Recording info
                "frameCount": self.frameCount,
                "durationSeconds": self.elapsedSeconds,
                "targetFPS": self.targetFPS,
                "actualFPS": self.elapsedSeconds > 0 ? Double(self.frameCount) / Double(self.elapsedSeconds) : 0,
                // Resolution
                "imageWidth": self.imageResolutionWidth,
                "imageHeight": self.imageResolutionHeight,
                "videoFormatFPS": self.videoFormatFPS,
                "fileSizeBytes": fileSize,
                // Depth / LiDAR
                "hasLiDAR": self.hasLiDAR,
                "hasDepthData": self.hasDepthData,
                "depthWidth": self.depthWidth,
                "depthHeight": self.depthHeight,
                "depthUri": self.depthURL?.absoluteString as Any? ?? NSNull(),
                "depthFrameCount": self.depthFrameCount,
                "depthFileSizeBytes": depthFileSize,
                "hasAudio": self.hasAudio,
                // Per-recording folder. Everything else (intrinsics JSON,
                // server CSV) gets written here too so the recording stays
                // self-contained under one directory.
                "recordingFolder": self.recordingFolderURL?.absoluteString as Any? ?? NSNull(),
                // Camera pose at last frame
                "eulerAngles": [self.eulerAnglesX, self.eulerAnglesY, self.eulerAnglesZ],
                "cameraTransform": self.cameraTransformMatrix,
                "projectionMatrix": self.projectionMatrix,
                // Exposure / lighting
                "exposureDuration": self.exposureDuration,
                "exposureOffset": self.exposureOffset,
                "lensPosition": self.lensPosition,
                "lightEstimateIntensity": self.lightEstimateIntensity,
                "lightEstimateTemperature": self.lightEstimateTemperature,
                // Per-frame camera transforms (full trajectory)
                "frameTransforms": self.frameTransforms
            ]

            DispatchQueue.main.async {
                self.delegate?.didFinishRecording(result)
                self.recordingCompletion?(result)
                self.recordingCompletion = nil
            }
        }
    }

    // MARK: - Calibration

    func calibrate() {
        let t = arSession.currentFrame?.camera.transform
        if let t = t {
            let sinPitch = -t.columns.2.y
            let cosPitch =  t.columns.2.z
            let currentPitch = Double(atan2(sinPitch, cosPitch)) * (180 / .pi)
            calibrationOffset = currentPitch - targetPitchDegrees
        }
    }

    // MARK: - Upload

    func uploadToServer(videoUri: String, serverUrl: String) {
        guard let videoURL = URL(string: videoUri) else { return }
        uploader.serverURL = serverUrl
        uploader.onProgress = { [weak self] progress in
            self?.delegate?.didUpdateUploadProgress(progress)
        }
        uploader.upload(videoURL: videoURL, fx: fx, fy: fy, cx: cx, cy: cy)
    }

    func currentTrackingStateString() -> String {
        guard let frame = arSession.currentFrame else { return "notAvailable" }
        switch frame.camera.trackingState {
        case .normal: return "normal"
        case .limited: return "limited"
        case .notAvailable: return "notAvailable"
        @unknown default: return "notAvailable"
        }
    }

    // MARK: - Helper: Extract 4x4 matrix as [[Float]]

    private func matrixToArray(_ m: simd_float4x4) -> [[Float]] {
        return [
            [m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x],
            [m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y],
            [m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z],
            [m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w]
        ]
    }
}

// MARK: - ARSessionDelegate

extension ARCaptureSessionManager: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Calculate pitch from camera transform
        let t = frame.camera.transform
        let sinPitch = -t.columns.2.y
        let cosPitch =  t.columns.2.z
        let corrected = Double(atan2(sinPitch, cosPitch)) * (180 / .pi) - calibrationOffset

        // Tracking quality
        let stateStr: String
        let stateMsg: String
        switch frame.camera.trackingState {
        case .normal:
            stateStr = "normal"
            stateMsg = ""
        case .limited(let reason):
            stateStr = "limited"
            switch reason {
            case .initializing:         stateMsg = "Initializing AR..."
            case .relocalizing:         stateMsg = "Relocalizing..."
            case .excessiveMotion:      stateMsg = "Too much motion"
            case .insufficientFeatures: stateMsg = "Not enough features"
            @unknown default:           stateMsg = "Limited tracking"
            }
        case .notAvailable:
            stateStr = "notAvailable"
            stateMsg = "Tracking unavailable"
        @unknown default:
            stateStr = "notAvailable"
            stateMsg = "Unknown"
        }

        let isLevel = abs(corrected - targetPitchDegrees) <= 1.0

        DispatchQueue.main.async {
            self.delegate?.didUpdatePitch(corrected, isLevel: isLevel)
            self.delegate?.didChangeTrackingState(stateStr, message: stateMsg)
        }

        // Recording: write frames to MP4
        guard isRecording,
              let videoInput = videoInput,
              let adaptor = pixelBufferAdaptor,
              videoInput.isReadyForMoreMediaData else { return }

        // FPS throttle
        let now = frame.timestamp
        guard now - lastCapturedTime >= 1.0 / Double(targetFPS) else { return }
        lastCapturedTime = now

        // Capture intrinsics and full metadata from first frame
        if !intrinsicsCaptured {
            let m = frame.camera.intrinsics
            fx = m.columns.0.x
            fy = m.columns.1.y
            cx = m.columns.2.x
            cy = m.columns.2.y
            intrinsicsMatrix = [
                [m.columns.0.x, m.columns.1.x, m.columns.2.x],
                [m.columns.0.y, m.columns.1.y, m.columns.2.y],
                [m.columns.0.z, m.columns.1.z, m.columns.2.z]
            ]

            // Image resolution
            let img = frame.capturedImage
            imageResolutionWidth = CVPixelBufferGetWidth(img)
            imageResolutionHeight = CVPixelBufferGetHeight(img)

            intrinsicsCaptured = true
        }

        // Capture depth data info AND write the raw float32 depth pixels to the
        // depth-data file (one frame's worth per write). Server uses this for the
        // Mode B z-replacement (LiDAR-grounded depth).
        if let depthMap = frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap {
            if !hasDepthData {
                hasDepthData = true
                depthWidth = CVPixelBufferGetWidth(depthMap)
                depthHeight = CVPixelBufferGetHeight(depthMap)
            }
            if let handle = depthFileHandle {
                CVPixelBufferLockBaseAddress(depthMap, .readOnly)
                if let baseAddr = CVPixelBufferGetBaseAddress(depthMap) {
                    let h = CVPixelBufferGetHeight(depthMap)
                    let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
                    let totalBytes = bytesPerRow * h
                    let data = Data(bytes: baseAddr, count: totalBytes)
                    handle.write(data)
                    depthFrameCount += 1
                }
                CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            }
        }

        // Update camera metadata (latest values)
        eulerAnglesX = frame.camera.eulerAngles.x
        eulerAnglesY = frame.camera.eulerAngles.y
        eulerAnglesZ = frame.camera.eulerAngles.z
        cameraTransformMatrix = matrixToArray(frame.camera.transform)
        projectionMatrix = matrixToArray(frame.camera.projectionMatrix(for: .portrait,
                                                                        viewportSize: CGSize(width: imageResolutionHeight,
                                                                                             height: imageResolutionWidth),
                                                                        zNear: 0.001, zFar: 1000))

        // Exposure info
        exposureDuration = frame.camera.exposureDuration
        exposureOffset = frame.camera.exposureOffset

        // Light estimation
        if let lightEstimate = frame.lightEstimate {
            lightEstimateIntensity = Double(lightEstimate.ambientIntensity)
            lightEstimateTemperature = Double(lightEstimate.ambientColorTemperature)
        }

        // Per-frame camera transform (position + orientation for trajectory)
        if capturePerFrameTransforms {
            let frameData: [String: Any] = [
                "frame": frameCount,
                "timestamp": frame.timestamp,
                "transform": matrixToArray(frame.camera.transform),
                "euler": [frame.camera.eulerAngles.x, frame.camera.eulerAngles.y, frame.camera.eulerAngles.z],
                "intrinsics_fx": frame.camera.intrinsics.columns.0.x,
                "intrinsics_fy": frame.camera.intrinsics.columns.1.y,
            ]
            frameTransforms.append(frameData)
        }

        // Write frame to MP4
        let timestamp = frame.timestamp
        if firstFrameTimestamp == nil { firstFrameTimestamp = timestamp }
        let elapsed = timestamp - firstFrameTimestamp!
        let presentationTime = CMTime(seconds: elapsed, preferredTimescale: 600)

        adaptor.append(frame.capturedImage, withPresentationTime: presentationTime)
        frameCount += 1

        if frameCount % 5 == 0 {
            DispatchQueue.main.async {
                self.delegate?.didUpdateFrameCount(self.frameCount)
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("AR Session error: \(error)")
    }
}


// MARK: - Audio capture (AVCaptureAudioDataOutputSampleBufferDelegate)

extension ARCaptureSessionManager: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard isRecording,
              let audioInput = self.audioInput,
              audioInput.isReadyForMoreMediaData,
              CMSampleBufferDataIsReady(sampleBuffer) else { return }
        audioInput.append(sampleBuffer)
    }
}
