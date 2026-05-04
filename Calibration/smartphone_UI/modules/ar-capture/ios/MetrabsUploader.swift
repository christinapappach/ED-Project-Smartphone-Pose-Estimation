// MetrabsUploader.swift
// Uploads MP4 video + camera intrinsics to MeTRAbs server.
// Video streams from disk in 4MB chunks -- never fully loads into memory.

import Foundation

class MetrabsUploader: NSObject {

    var serverURL = "http://YOUR_SERVER_IP:8000/analyze"
    var onProgress: ((Double) -> Void)?
    var onComplete: ((Result<String, Error>) -> Void)?

    func upload(videoURL: URL, fx: Float, fy: Float, cx: Float, cy: Float) {
        guard let tempBodyURL = buildMultipartFile(
            videoURL: videoURL, fx: fx, fy: fy, cx: cx, cy: cy
        ) else {
            print("Could not build multipart body")
            return
        }

        guard let url = URL(string: serverURL) else {
            print("Invalid server URL: \(serverURL)")
            return
        }

        let boundary = tempBodyURL.deletingPathExtension().lastPathComponent

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600

        let config = URLSessionConfiguration.background(withIdentifier: "metrabs-upload-\(UUID().uuidString)")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true

        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        let task = session.uploadTask(with: request, fromFile: tempBodyURL)
        task.resume()
    }

    private func buildMultipartFile(
        videoURL: URL, fx: Float, fy: Float, cx: Float, cy: Float
    ) -> URL? {
        let boundary = "MeTRAbsBoundary-\(UUID().uuidString)"
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent("\(boundary).multipart")

        guard FileManager.default.createFile(atPath: tempURL.path, contents: nil) else { return nil }
        guard let handle = try? FileHandle(forWritingTo: tempURL) else { return nil }
        defer { handle.closeFile() }

        let fields: [(String, String)] = [
            ("focal_length_x",    String(fx)),
            ("focal_length_y",    String(fy)),
            ("principal_point_x", String(cx)),
            ("principal_point_y", String(cy))
        ]

        for (name, value) in fields {
            handle.write("--\(boundary)\r\n")
            handle.write("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            handle.write("\(value)\r\n")
        }

        handle.write("--\(boundary)\r\n")
        handle.write("Content-Disposition: form-data; name=\"video\"; filename=\"\(videoURL.lastPathComponent)\"\r\n")
        handle.write("Content-Type: video/mp4\r\n\r\n")

        guard let videoHandle = try? FileHandle(forReadingFrom: videoURL) else { return nil }
        defer { videoHandle.closeFile() }

        let chunkSize = 4 * 1024 * 1024
        while true {
            let chunk = videoHandle.readData(ofLength: chunkSize)
            if chunk.isEmpty { break }
            handle.write(chunk)
        }

        handle.write("\r\n--\(boundary)--\r\n")
        return tempURL
    }
}

extension MetrabsUploader: URLSessionTaskDelegate, URLSessionDataDelegate {

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didSendBodyData bytesSent: Int64, totalBytesSent: Int64,
                    totalBytesExpectedToSend: Int64) {
        let progress = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        DispatchQueue.main.async { self.onProgress?(progress) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            DispatchQueue.main.async { self.onComplete?(.failure(error)) }
        } else {
            DispatchQueue.main.async { self.onComplete?(.success("Upload complete")) }
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        if let response = String(data: data, encoding: .utf8) {
            print("Server response: \(response)")
        }
    }
}

private extension FileHandle {
    func write(_ string: String) {
        if let data = string.data(using: .utf8) { write(data) }
    }
}
