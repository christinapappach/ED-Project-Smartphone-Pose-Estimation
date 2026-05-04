// MetrabsUploader.swift
// Uploads the MP4 video + camera intrinsics directly to your MeTRAbs server.
// The intrinsics (fx, fy, cx, cy) go as plain form fields — no JSON file needed.
// The video streams from disk so it never fully loads into memory.

import Foundation

class MetrabsUploader: NSObject {

    // MARK: - Config
    // ⚠️ Change this to your server's address
    var serverURL = "http://YOUR_SERVER_IP:8000/analyze"

    // MARK: - Callbacks
    var onProgress: ((Double) -> Void)?
    var onComplete: ((Result<String, Error>) -> Void)?

    // MARK: - Upload

    func upload(videoURL: URL, fx: Float, fy: Float, cx: Float, cy: Float) {
        print("📤 Preparing upload: \(videoURL.lastPathComponent)")
        print("   fx=\(fx)  fy=\(fy)  cx=\(cx)  cy=\(cy)")

        // Build the multipart body as a TEMP FILE so we never load the video into RAM.
        // Strategy: write [headers + intrinsic fields + video file bytes + closing boundary]
        // all to a temp file on disk, then stream that temp file to the server.
        guard let tempBodyURL = buildMultipartFile(
            videoURL: videoURL,
            fx: fx, fy: fy, cx: cx, cy: cy
        ) else {
            print("❌ Could not build multipart body")
            return
        }

        // Report temp file size
        if let attrs = try? FileManager.default.attributesOfItem(atPath: tempBodyURL.path),
           let size = attrs[.size] as? Int64 {
            let mb = Double(size) / 1_048_576
            print("   Upload payload: \(String(format: "%.1f", mb)) MB")
        }

        guard let url = URL(string: serverURL) else {
            print("❌ Invalid server URL: \(serverURL)")
            return
        }

        // Extract boundary from the temp file name (we embedded it there)
        let boundary = tempBodyURL.deletingPathExtension().lastPathComponent

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600  // 10 min — large file upload

        // Use a background URLSession so upload survives if app goes to background
        let config = URLSessionConfiguration.background(withIdentifier: "metrabs-upload-\(UUID().uuidString)")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true

        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        let task = session.uploadTask(with: request, fromFile: tempBodyURL)
        task.resume()

        print("🚀 Upload started")
    }

    // MARK: - Build Multipart Body File

    /// Writes all multipart data to a temp file and returns its URL.
    /// Using a file means the video bytes stream from disk → no RAM spike.
    private func buildMultipartFile(
        videoURL: URL,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> URL? {
        let boundary = "MeTRAbsBoundary-\(UUID().uuidString)"

        // Temp file named after the boundary so we can recover it later
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent("\(boundary).multipart")

        guard FileManager.default.createFile(atPath: tempURL.path, contents: nil) else {
            return nil
        }
        guard let handle = try? FileHandle(forWritingTo: tempURL) else { return nil }
        defer { handle.closeFile() }

        // ── Intrinsic fields ─────────────────────────────────────────────────
        // These are the numbers MeTRAbs needs to do accurate 3D pose estimation.
        // They tell MeTRAbs: "this is the camera that shot this video."
        let fields: [(String, String)] = [
            ("focal_length_x",   String(fx)),
            ("focal_length_y",   String(fy)),
            ("principal_point_x", String(cx)),
            ("principal_point_y", String(cy))
        ]

        for (name, value) in fields {
            handle.write("--\(boundary)\r\n")
            handle.write("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            handle.write("\(value)\r\n")
        }

        // ── Video file ───────────────────────────────────────────────────────
        handle.write("--\(boundary)\r\n")
        handle.write("Content-Disposition: form-data; name=\"video\"; filename=\"\(videoURL.lastPathComponent)\"\r\n")
        handle.write("Content-Type: video/mp4\r\n\r\n")

        // Stream video in 4MB chunks — never loads whole file into memory
        guard let videoHandle = try? FileHandle(forReadingFrom: videoURL) else {
            print("❌ Could not open video file for reading")
            return nil
        }
        defer { videoHandle.closeFile() }

        let chunkSize = 4 * 1024 * 1024  // 4 MB
        while true {
            let chunk = videoHandle.readData(ofLength: chunkSize)
            if chunk.isEmpty { break }
            handle.write(chunk)
        }

        handle.write("\r\n--\(boundary)--\r\n")

        print("✅ Multipart body written to temp file")
        return tempURL
    }
}

// MARK: - URLSessionDelegate (progress + completion)

extension MetrabsUploader: URLSessionTaskDelegate, URLSessionDataDelegate {

    // Upload progress
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didSendBodyData bytesSent: Int64,
        totalBytesSent: Int64,
        totalBytesExpectedToSend: Int64
    ) {
        let progress = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        let mb = Double(totalBytesSent) / 1_048_576
        print("📡 Uploading: \(String(format: "%.1f", mb)) MB (\(Int(progress * 100))%)")
        DispatchQueue.main.async {
            self.onProgress?(progress)
        }
    }

    // Task finished
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("❌ Upload failed: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.onComplete?(.failure(error))
            }
        } else {
            print("✅ Upload complete")
            DispatchQueue.main.async {
                self.onComplete?(.success("Upload complete"))
            }
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        if let response = String(data: data, encoding: .utf8) {
            print("🖥️ Server response: \(response)")
        }
    }
}

// MARK: - FileHandle convenience

private extension FileHandle {
    func write(_ string: String) {
        if let data = string.data(using: .utf8) {
            write(data)
        }
    }
}
