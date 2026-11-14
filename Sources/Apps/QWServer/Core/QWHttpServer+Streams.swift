import Foundation
import Network

/// Thread-safe gate that stops SSE keepalive comments once the first token arrives.
public final class QWStreamKeepaliveGate: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    public init() {}

    public func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
    }

    public func isStopped() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }
}

/// The streaming endpoints, shared wire-protocol helpers, and the static UI.
extension QWHttpServer {

    /// Resolves a client-requested model identifier against the active engine model and configured aliases.
    /// Returns `nil` when `strictNotFound` is `true` and the requested model is not recognized.
    public static func resolveReplyModelName(
        requested rawRequested: String?,
        serverModel: String,
        settings: QWServerSettings,
        strictNotFound: Bool,
        discovered: [DiscoveredModel] = QWServerSettings.scanModelsDirectory()
    ) -> String? {
        let requested = (rawRequested ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if requested.isEmpty {
            let active = settings.alias(for: serverModel)
            return active.isEmpty ? serverModel : active
        }
        let ref = settings.resolveModelReference(requested, discovered: discovered)
        let isKnownModel = requested.caseInsensitiveCompare(serverModel) == .orderedSame
            || requested.caseInsensitiveCompare(settings.modelName) == .orderedSame
            || discovered.contains(where: {
                $0.name.caseInsensitiveCompare(ref.modelName) == .orderedSame || $0.packPath == ref.packPath
            })
            || (!ref.packPath.isEmpty && FileManager.default.fileExists(atPath: ref.packPath))
        if !ref.alias.isEmpty {
            return ref.alias
        } else if isKnownModel {
            return ref.modelName
        } else if strictNotFound {
            return nil
        } else {
            let active = settings.alias(for: serverModel)
            return active.isEmpty ? serverModel : active
        }
    }

    func resolveReplyModelName(requested: String?, strictNotFound: Bool) -> String? {
        Self.resolveReplyModelName(
            requested: requested,
            serverModel: actor.getModelName(),
            settings: QWSettingsManager.shared.loadSettings(),
            strictNotFound: strictNotFound
        )
    }

    func sendSSEHeaders(_ connection: NWConnection) {
        let sseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\(corsHeaderLines(for: connection))\r\n"
        connection.send(content: Data(sseHeaders.utf8), isComplete: false, completion: .contentProcessed({ _ in }))
    }

    func sendNDJSONHeaders(_ connection: NWConnection) {
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\(corsHeaderLines(for: connection))\r\n"
        connection.send(content: Data(headers.utf8), isComplete: false, completion: .contentProcessed({ _ in }))
    }

    func startSSEKeepalive(connection: NWConnection) -> (gate: QWStreamKeepaliveGate, task: Task<Void, Never>) {
        let gate = QWStreamKeepaliveGate()
        let keepaliveComment = Data(": keepalive\n\n".utf8)
        let task = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled, !gate.isStopped() else { break }
                if self?.isPeerGone(connection) == true { break }
                connection.send(content: keepaliveComment, isComplete: false, completion: .idempotent)
            }
        }
        return (gate, task)
    }

    func handleMetricsStream(connection: NWConnection) {
        sendSSEHeaders(connection)

        Task {
            while !Task.isCancelled {
                // The peer can leave at any time. When it does, the loop
                // stops: a stream that keeps building a payload for a
                // client that is gone is work nobody asked for.
                if self.isPeerGone(connection) { break }
                let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                if let data = try? JSONEncoder().encode(snapshot),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    let msg = "data: \(jsonStr)\n\n"
                    guard let msgData = msg.data(using: .utf8) else { break }
                    connection.send(content: msgData, isComplete: false, completion: .contentProcessed({ error in
                        if error != nil {
                            connection.cancel()
                        }
                    }))
                }
                let intervalNs: UInt64 = (snapshot.throughput.phase == "prefill" || snapshot.throughput.phase == "decode")
                    ? 150_000_000
                    : 500_000_000
                try? await Task.sleep(nanoseconds: intervalNs)
            }
            connection.cancel()
        }
    }

    /// The peer is gone: the connection is cancelled or it failed. A
    /// stream must stop building a payload for nobody.
    func isPeerGone(_ connection: NWConnection) -> Bool {
        isConnectionTerminated(connection)
    }

    /// The published view model, pushed as it changes. The renderer
    /// receives the same model the dashboard endpoint serves.
    /// `/v1/metrics/stream` still serves the raw metrics.
    func handleDashboardStream(connection: NWConnection) {
        sendSSEHeaders(connection)

        Task {
            while !Task.isCancelled {
                // The peer can leave at any time. When it does, the loop
                // stops: a stream that keeps building a model for a
                // client that is gone is work nobody asked for.
                if self.isPeerGone(connection) { break }
                let model = self.dashboardModel()
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                if let data = try? encoder.encode(model),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    let msg = "data: \(jsonStr)\n\n"
                    guard let msgData = msg.data(using: .utf8) else { break }
                    connection.send(content: msgData, isComplete: false, completion: .contentProcessed({ error in
                        if error != nil {
                            connection.cancel()
                        }
                    }))
                }
                let intervalNs: UInt64 = (model.generation == .prefill || model.generation == .decode)
                    ? 150_000_000
                    : 500_000_000
                try? await Task.sleep(nanoseconds: intervalNs)
            }
            connection.cancel()
        }
    }

    func handleLogsStream(connection: NWConnection) {
        sendSSEHeaders(connection)

        // Send recent logs first
        let recent = self.logger.getRecentLogs()
        for line in recent {
            if let json = try? JSONEncoder().encode(["line": line]),
               let str = String(data: json, encoding: .utf8) {
                let msg = "data: \(str)\n\n"
                connection.send(content: Data(msg.utf8), isComplete: false, completion: .contentProcessed({ _ in }))
            }
        }

        let subId = self.logger.subscribe { [weak connection] line in
            guard let connection else { return }
            if let json = try? JSONEncoder().encode(["line": line]),
               let str = String(data: json, encoding: .utf8) {
                let msg = "data: \(str)\n\n"
                connection.send(content: Data(msg.utf8), isComplete: false, completion: .contentProcessed({ _ in }))
            }
        }

        registerCancellationHandler(for: connection) { [weak self] in
            self?.logger.unsubscribe(subId)
        }
    }

    func handleStaticUI(connection: NWConnection, path: String) {
        let clean = path.components(separatedBy: "?").first ?? path
        let relative: String
        if clean == "/ui" || clean == "/ui/" || clean == "/ui/controlbar" || clean == "/ui/controlbar.html" {
            relative = "controlbar.html"
        } else if clean == "/ui/logo.svg" {
            relative = "logo.svg"
        } else if clean.hasPrefix("/ui/") {
            relative = String(clean.dropFirst(4))
        } else {
            relative = "controlbar.html"
        }

        guard isSafeStaticRelativePath(relative) else {
            sendSimpleResponse(connection, status: 403, body: "Forbidden\n")
            return
        }

        let fm = FileManager.default
        if let localPath = Self.resolveStaticUIResourcePath(relative: relative, fm: fm),
           let data = try? Data(contentsOf: URL(fileURLWithPath: localPath)) {
            let contentType = Self.staticUIMimeType(for: relative)
            sendRawData(
                connection,
                status: 200,
                contentType: contentType,
                data: data,
                extraHeaders: ["Cache-Control: no-cache"]
            )
        } else {
            let escapedPath = QWHttpSecurity.escapeHTML(relative)
            let body = "<html><body style='background:#111318;color:#e2e2e9;font-family:sans-serif;padding:24px;'><h2>QWEGN Controlbar</h2><p>The control bar UI resource <code>\(escapedPath)</code> is not installed.</p><p>Build it: <code>make build</code></p></body></html>"
            sendRawData(connection, status: 503, contentType: "text/html; charset=utf-8", data: Data(body.utf8))
        }
    }

    private static func staticUIMimeType(for relative: String) -> String {
        let lower = relative.lowercased()
        if lower.hasSuffix(".css") { return "text/css; charset=utf-8" }
        if lower.hasSuffix(".js") { return "application/javascript; charset=utf-8" }
        if lower.hasSuffix(".svg") { return "image/svg+xml" }
        if lower.hasSuffix(".json") { return "application/json; charset=utf-8" }
        return "text/html; charset=utf-8"
    }

    private func isSafeStaticRelativePath(_ relative: String) -> Bool {
        guard !relative.isEmpty,
              !relative.hasPrefix("/"),
              !relative.contains("\\"),
              !relative.contains("\0") else {
            return false
        }
        let components = relative.split(separator: "/", omittingEmptySubsequences: false)
        return !components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." })
    }

    private static func resolveStaticUIResourcePath(relative: String, fm: FileManager) -> String? {
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        if let bundle = activeStaticUIBundle(execDir: execDir, fm: fm) {
            return containedResourcePath(bundle: bundle, relative: relative, fm: fm)
        }
        let devStaticRoot = URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent("Sources/Apps/QWServer/Resources", isDirectory: true)
            .resolvingSymlinksInPath()
            .standardizedFileURL
        var rootIsDir: ObjCBool = false
        if fm.fileExists(atPath: devStaticRoot.path, isDirectory: &rootIsDir), rootIsDir.boolValue {
            let candidate = devStaticRoot.appendingPathComponent(relative).resolvingSymlinksInPath().standardizedFileURL
            let rootPrefix = devStaticRoot.path.hasSuffix("/") ? devStaticRoot.path : devStaticRoot.path + "/"
            var isDir: ObjCBool = false
            return candidate.path.hasPrefix(rootPrefix)
                && fm.fileExists(atPath: candidate.path, isDirectory: &isDir)
                && !isDir.boolValue
                ? candidate.path
                : nil
        }
        let fallbackCandidates = [
            URL(fileURLWithPath: "/usr/local/share/qwegn/qwegine_QWServer.bundle"),
            URL(fileURLWithPath: "/Applications/Qwegn.app/Contents/Resources/qwegine_QWServer.bundle"),
        ]
        if let fallback = firstExistingBundle(in: fallbackCandidates, fm: fm) {
            return containedResourcePath(bundle: fallback, relative: relative, fm: fm)
        }
        return nil
    }

    private static func activeStaticUIBundle(execDir: URL, fm: FileManager) -> Bundle? {
        var candidates: [URL] = []
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PACKAGE_RESOURCE_BUNDLE_PATH"] ?? env["PACKAGE_RESOURCE_BUNDLE_URL"], !raw.isEmpty {
            let url = URL(fileURLWithPath: raw)
            candidates.append(url.pathExtension == "bundle" ? url : url.appendingPathComponent("qwegine_QWServer.bundle"))
        }
        let serverBundle = Bundle(for: QWHttpServer.self)
        if let mainRes = Bundle.main.resourceURL {
            candidates.append(mainRes.appendingPathComponent("qwegine_QWServer.bundle"))
        }
        if let srvRes = serverBundle.resourceURL {
            candidates.append(srvRes.appendingPathComponent("qwegine_QWServer.bundle"))
        }
        candidates.append(serverBundle.bundleURL.deletingLastPathComponent().appendingPathComponent("qwegine_QWServer.bundle"))
        candidates.append(execDir.appendingPathComponent("qwegine_QWServer.bundle"))
        candidates.append(execDir.deletingLastPathComponent().appendingPathComponent("Resources/qwegine_QWServer.bundle"))
        candidates.append(execDir.deletingLastPathComponent().appendingPathComponent("share/qwegn/qwegine_QWServer.bundle"))
        return firstExistingBundle(in: candidates, fm: fm)
    }

    private static func firstExistingBundle(in candidates: [URL], fm: FileManager) -> Bundle? {
        for url in candidates {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue,
               let bundle = Bundle(url: url) {
                return bundle
            }
        }
        return nil
    }

    private static func containedResourcePath(bundle: Bundle, relative: String, fm: FileManager) -> String? {
        guard let root = bundle.resourceURL?.resolvingSymlinksInPath().standardizedFileURL else {
            return nil
        }
        let nsRelative = relative as NSString
        let ext = nsRelative.pathExtension
        let withoutExt = nsRelative.deletingPathExtension
        let subdir = (withoutExt as NSString).deletingLastPathComponent
        let stem = (withoutExt as NSString).lastPathComponent
        let subOpt: String? = subdir.isEmpty ? nil : subdir
        let candidate = bundle.url(forResource: stem, withExtension: ext, subdirectory: subOpt)?
            .resolvingSymlinksInPath().standardizedFileURL
            ?? root.appendingPathComponent(relative).resolvingSymlinksInPath().standardizedFileURL
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(rootPrefix) else { return nil }
        var isDir: ObjCBool = false
        return fm.fileExists(atPath: candidate.path, isDirectory: &isDir) && !isDir.boolValue ? candidate.path : nil
    }
}
