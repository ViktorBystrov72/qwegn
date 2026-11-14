// QWCli+PullAndQuantize.swift — Python bootstrap and HuggingFace ranged downloader.

import Foundation
import QWCore
import QWKV
import QWQuant

extension QWCli {
    /// Creates or rebuilds the lazy Python venv in `~/Library/Application Support/Qwegn/venv/`.
    @discardableResult
    public static func ensurePythonEnvironment(
        venvDir: URL = QWPaths.pythonVenvDirectoryURL,
        supportRoot: URL = QWPaths.supportDirectoryURL,
        rebuild: Bool = false,
        onProgress: (String) -> Void = { print($0) }
    ) throws -> URL {
        let fm = FileManager.default
        let py = venvDir.appendingPathComponent("bin/python")
        let pip = venvDir.appendingPathComponent("bin/pip")
        let stamp = venvDir.appendingPathComponent(".requirements.stamp")
        if !rebuild, fm.isExecutableFile(atPath: py.path), fm.fileExists(atPath: stamp.path) {
            return py
        }
        onProgress("Creating Python environment…")
        if rebuild, fm.fileExists(atPath: venvDir.path) {
            try fm.removeItem(at: venvDir)
        }
        try fm.createDirectory(at: venvDir.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bootstrap = try findBootstrapPython(fm: fm)
        try runProcess(executable: bootstrap, arguments: ["-m", "venv", venvDir.path])
        if ProcessInfo.processInfo.environment["QWEGN_SKIP_PIP_INSTALL"] != "1" {
            try runProcess(executable: pip.path, arguments: ["install", "-q", "--upgrade", "pip"])
            let reqFile = supportRoot.appendingPathComponent("requirements.txt")
            var installedFromReqs = false
            if fm.fileExists(atPath: reqFile.path),
               let reqText = try? String(contentsOf: reqFile, encoding: .utf8) {
                let torchVer = reqText.split(separator: "\n")
                    .first(where: { $0.hasPrefix("torch==") })
                    .map { String($0.dropFirst("torch==".count)).trimmingCharacters(in: .whitespacesAndNewlines) }
                let torchIdx = ProcessInfo.processInfo.environment["TORCH_INDEX"] ?? "https://download.pytorch.org/whl/cpu"
                do {
                    if let tv = torchVer, !tv.isEmpty {
                        try runProcess(executable: pip.path,
                                       arguments: ["install", "-q", "--index-url", torchIdx, "torch==\(tv)"])
                    }
                    try runProcess(executable: pip.path, arguments: ["install", "-q", "-r", reqFile.path])
                    installedFromReqs = true
                } catch {
                    installedFromReqs = false
                }
            }
            if !installedFromReqs {
                try runProcess(executable: pip.path, arguments: ["install", "-q", "numpy", "safetensors", "torch"])
            }
        }
        try Data("ok\n".utf8).write(to: stamp, options: .atomic)
        return py
    }

    private static func findBootstrapPython(fm: FileManager) throws -> String {
        if let env = ProcessInfo.processInfo.environment["QWEGN_BOOTSTRAP_PYTHON"],
           !env.isEmpty, fm.isExecutableFile(atPath: env) {
            return env
        }
        let names = ["python3.13", "python3.12", "python3.11", "python3"]
        let dirs = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
            + (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for name in names {
            for dir in dirs where !dir.isEmpty {
                let cand = (dir as NSString).appendingPathComponent(name)
                if fm.isExecutableFile(atPath: cand) { return cand }
            }
        }
        throw NSError(domain: "QWCli.python", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Python 3.11+ not found (install python3 or set QWEGN_BOOTSTRAP_PYTHON)"])
    }

    private static func runProcess(executable: String, arguments: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: executable)
        p.arguments = arguments
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        let out = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            let msg = String(data: out, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            throw NSError(domain: "QWCli.python", code: Int(p.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "Command failed (\(executable) \(arguments.joined(separator: " "))): \(msg)"])
        }
    }

    /// Splits a file byte range into `parts` chunks (at most 1 chunk per 4 MiB).
    public static func downloadPlan(size: Int, parts: Int) -> [(start: Int, end: Int)] {
        guard size > 0, parts > 0 else { return [] }
        let p = min(parts, max(1, size / (4 << 20)))
        var out: [(Int, Int)] = []
        let step = (size + p - 1) / p
        var s = 0
        while s < size {
            out.append((s, min(size, s + step) - 1))
            s += step
        }
        return out
    }

    /// Returns whether a HuggingFace repository file path is required for local model inference.
    public static func shouldDownloadRepoFile(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        if name.hasPrefix(".") { return false }
        // The root README is the hub's model card; a native pack's assets/ keeps the upstream
        // README and LICENSE next to the weights, so those are pulled.
        let upper = name.uppercased()
        if upper.hasPrefix("README") || upper.hasPrefix("LICENSE") {
            return path.contains("/")
        }
        let ext = (name as NSString).pathExtension.lowercased()
        let allowedExts: Set<String> = ["json", "safetensors", "qwp", "qlsa", "sha256", "txt", "jinja",
                                        "model", "tiktoken", "bin"]
        return allowedExts.contains(ext)
    }

    /// Resolves `relativePath` strictly inside `baseDir`, rejecting traversal (`..`) or absolute escapes.
    public static func validatedRelativeDestination(baseDir: URL, relativePath: String) throws -> URL {
        let trimmed = relativePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.hasPrefix("/"),
              !trimmed.hasPrefix("~"),
              !trimmed.contains("\0") else {
            throw NSError(domain: "QWCli.pull", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Unsafe checkpoint relative path '\(relativePath)'"])
        }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty,
              !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
            throw NSError(domain: "QWCli.pull", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Path traversal component in '\(relativePath)'"])
        }
        let canonicalBase = baseDir.resolvingSymlinksInPath().standardizedFileURL
        let candidate = canonicalBase.appendingPathComponent(trimmed).standardizedFileURL
        let basePrefix = canonicalBase.path.hasSuffix("/") ? canonicalBase.path : canonicalBase.path + "/"
        guard candidate.path.hasPrefix(basePrefix) else {
            throw NSError(domain: "QWCli.pull", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Destination escapes target directory: '\(relativePath)'"])
        }
        return candidate
    }

    /// Downloads a checkpoint from the HuggingFace Hub using parallel HTTP Range requests.
    /// Every file gets its own progress row (bar, bytes, rate) that updates while it downloads.
    @discardableResult
    public static func pullCheckpoint(repo: String, folder: URL, parts: Int = 4) throws -> URL {
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains("/"), !trimmed.contains("..") else {
            throw NSError(domain: "QWCli.pull", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid HF repo id '\(repo)' (expected org/model)"])
        }
        let repoLeaf = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
        let targetDir = folder.lastPathComponent == repoLeaf
            ? folder
            : folder.appendingPathComponent(repoLeaf, isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let hfBase = (ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
        guard let treeURL = URL(string: "\(hfBase)/api/models/\(trimmed)/tree/main?recursive=true") else {
            throw NSError(domain: "QWCli.pull", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid tree URL for '\(trimmed)'"])
        }

        let treeData = try fetchSyncData(url: treeURL, token: token, range: nil)
        guard let entries = try JSONSerialization.jsonObject(with: treeData) as? [[String: Any]] else {
            throw NSError(domain: "QWCli.pull", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Unexpected HF tree response for '\(trimmed)'"])
        }

        var files: [(path: String, size: Int)] = []
        for entry in entries {
            guard (entry["type"] as? String) == "file",
                  let path = entry["path"] as? String,
                  shouldDownloadRepoFile(path) else { continue }
            let lfsSize = (entry["lfs"] as? [String: Any])?["size"] as? Int
            let directSize = entry["size"] as? Int
            files.append((path: path, size: lfsSize ?? directSize ?? 0))
        }
        guard !files.isEmpty else {
            throw NSError(domain: "QWCli.pull", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "No model files found in '\(trimmed)'"])
        }

        let totalBytes = files.reduce(0) { $0 + $1.size }
        print(String(format: "pulling %@ -> %@ (%d files, %.2f GiB)",
                     trimmed, targetDir.path, files.count, Double(totalBytes) / 1_073_741_824.0))

        // One progress row per file, in repo order. Files already complete are shown as cached.
        let progress = PullProgress(files: files.map { PullProgress.Row(path: $0.path, total: $0.size) },
                                    color: .auto())
        progress.start()
        defer { progress.finish() }

        for (index, file) in files.enumerated() {
            let dstURL = try validatedRelativeDestination(baseDir: targetDir, relativePath: file.path)
            try fm.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if file.size > 0,
               let attrs = try? fm.attributesOfItem(atPath: dstURL.path),
               let existingSize = attrs[.size] as? Int,
               existingSize == file.size {
                progress.markCached(index)
                continue
            }
            let encodedPath = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            guard let fileURL = URL(string: "\(hfBase)/\(trimmed)/resolve/main/\(encodedPath)") else {
                throw NSError(domain: "QWCli.pull", code: 5,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid file URL for '\(file.path)'"])
            }
            try downloadFileRanged(url: fileURL, dstURL: dstURL, size: file.size, parts: max(1, parts),
                                   token: token) { received in progress.update(index, received: received) }
            progress.markDone(index)
        }

        let stFiles = files.map(\.path).filter { $0.hasSuffix(".safetensors") }
        let modelST = targetDir.appendingPathComponent("model.safetensors")
        let indexST = targetDir.appendingPathComponent("model.safetensors.index.json")
        if stFiles.count == 1, stFiles[0] != "model.safetensors",
           !fm.fileExists(atPath: modelST.path), !fm.fileExists(atPath: indexST.path) {
            try? fm.createSymbolicLink(atPath: modelST.path, withDestinationPath: stFiles[0])
        }
        return targetDir
    }

    private final class PullBox<T>: @unchecked Sendable {
        let lock = NSLock()
        var value: T
        init(_ value: T) { self.value = value }
    }

    private static func fetchSyncData(url: URL, token: String?, range: (start: Int, end: Int)?) throws -> Data {
        var req = URLRequest(url: url)
        if let token, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let range {
            req.setValue("bytes=\(range.start)-\(range.end)", forHTTPHeaderField: "Range")
        }
        let sem = DispatchSemaphore(value: 0)
        let box = PullBox<Result<Data, Error>>(.failure(NSError(domain: "QWCli.pull", code: -1)))
        URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { sem.signal() }
            if let err {
                box.lock.lock(); box.value = .failure(err); box.lock.unlock()
                return
            }
            if let http = resp as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                let msg = "HTTP \(http.statusCode) for \(url.absoluteString)"
                box.lock.lock()
                box.value = .failure(NSError(domain: "QWCli.pull", code: http.statusCode,
                                             userInfo: [NSLocalizedDescriptionKey: msg]))
                box.lock.unlock()
                return
            }
            box.lock.lock(); box.value = .success(data ?? Data()); box.lock.unlock()
        }.resume()
        sem.wait()
        return try box.value.get()
    }

    /// Streams one ranged chunk per request into a shared `.part` file, reporting every received
    /// block, so a file's progress moves while its bytes arrive rather than when a chunk completes.
    private final class RangedFileDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private struct Task { let start: Int; let expected: Int; var received: Int }
        private let lock = NSLock()
        private let handle: FileHandle
        private var tasks: [Int: Task] = [:]
        private var totalReceived = 0
        private var firstError: Error?
        private let group = DispatchGroup()
        private let progress: (Int) -> Void

        init(handle: FileHandle, progress: @escaping (Int) -> Void) {
            self.handle = handle
            self.progress = progress
        }

        func add(_ task: URLSessionDataTask, start: Int, expected: Int) {
            lock.withLock { tasks[task.taskIdentifier] = Task(start: start, expected: expected, received: 0) }
            group.enter()
        }

        /// Blocks until every chunk finished, then reports the first failure, if any.
        func wait() throws {
            group.wait()
            if let err = lock.withLock({ firstError }) { throw err }
        }

        private func fail(_ error: Error) {
            if firstError == nil { firstError = error }
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                lock.withLock {
                    fail(NSError(domain: "QWCli.pull", code: http.statusCode,
                                 userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) for \(response.url?.path ?? "file")"]))
                }
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            lock.withLock {
                guard var t = tasks[dataTask.taskIdentifier] else { return }
                let room = t.expected - t.received
                let n = min(room, data.count)
                guard n > 0 else { return }
                do {
                    try handle.seek(toOffset: UInt64(t.start + t.received))
                    try handle.write(contentsOf: n == data.count ? data : data.prefix(n))
                } catch {
                    fail(error)
                    dataTask.cancel()
                    return
                }
                t.received += n
                tasks[dataTask.taskIdentifier] = t
                totalReceived += n
                progress(totalReceived)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            lock.withLock {
                if let error { fail(error) }
                else if let t = tasks[task.taskIdentifier], t.received != t.expected {
                    fail(NSError(domain: "QWCli.pull", code: 6,
                                 userInfo: [NSLocalizedDescriptionKey:
                                    "short read at offset \(t.start): \(t.received) of \(t.expected) bytes"]))
                }
            }
            group.leave()
        }
    }

    /// Downloads one file as parallel ranges into `dstURL`. `progress` receives the running
    /// byte total of this file as data arrives.
    private static func downloadFileRanged(url: URL, dstURL: URL, size: Int, parts: Int, token: String?,
                                           progress: @escaping (Int) -> Void) throws {
        let plan = downloadPlan(size: size, parts: parts)
        let fm = FileManager.default
        if plan.count <= 1 {
            let data = try fetchSyncData(url: url, token: token, range: nil)
            try data.write(to: dstURL, options: .atomic)
            progress(data.count)
            return
        }
        let tmpURL = dstURL.appendingPathExtension("part")
        _ = fm.createFile(atPath: tmpURL.path, contents: nil)
        let initHandle = try FileHandle(forWritingTo: tmpURL)
        try initHandle.truncate(atOffset: UInt64(size))
        try initHandle.close()

        let writer = try FileHandle(forWritingTo: tmpURL)
        defer { try? writer.close() }
        let job = RangedFileDownload(handle: writer, progress: progress)
        let session = URLSession(configuration: .default, delegate: job, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        for chunk in plan {
            var req = URLRequest(url: url)
            if let token, !token.isEmpty {
                req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            req.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
            let task = session.dataTask(with: req)
            job.add(task, start: chunk.start, expected: chunk.end - chunk.start + 1)
            task.resume()
        }
        do {
            try job.wait()
        } catch {
            try? fm.removeItem(at: tmpURL)
            throw error
        }
        if fm.fileExists(atPath: dstURL.path) {
            try fm.removeItem(at: dstURL)
        }
        try fm.moveItem(at: tmpURL, to: dstURL)
    }

}
