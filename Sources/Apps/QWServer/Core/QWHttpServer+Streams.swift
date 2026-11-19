// QWHttpServer+Streams.swift — Live telemetry/log SSE streams, EOF cancellation
// watcher, static WebUI bundle serving, and model-name reply resolution.

import Foundation
import Network
import os

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

    func makeSSEWriter(
        for connection: NWConnection,
        dateEncodingStrategy: JSONEncoder.DateEncodingStrategy? = nil,
        onError: (@Sendable () -> Void)? = nil
    ) -> QWSSEWriter {
        let id = ObjectIdentifier(connection)
        let flag = connectionStateLock.withLock { cancellationHubs[id]?.cancelFlag }
        return QWSSEWriter(
            connection: connection,
            cancelFlag: flag,
            dateEncodingStrategy: dateEncodingStrategy,
            onError: onError
        )
    }

    func armEOFWatcher(on connection: NWConnection, id: ObjectIdentifier) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1) { [weak self, weak connection] _, _, isComplete, error in
            guard let self, let connection else { return }
            if isComplete || error != nil {
                let hub = self.connectionStateLock.withLock { self.cancellationHubs[id] }
                hub?.fire()
                connection.cancel()
                return
            }
            if !self.isConnectionTerminated(connection) {
                self.armEOFWatcher(on: connection, id: id)
            }
        }
    }

    func handleMetricsStream(connection: NWConnection) {
        let session = makeSSESession(for: connection, format: .sse, enableKeepalive: false)
        session.begin(startKeepalive: false)

        let streamTask = Task {
            while !Task.isCancelled {
                if session.isClosed { break }
                let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                session.sendSSEDataFrame(snapshot)
                let intervalNs: UInt64 = (snapshot.throughput.phase == "prefill" || snapshot.throughput.phase == "decode")
                    ? 150_000_000
                    : 500_000_000
                try? await Task.sleep(nanoseconds: intervalNs)
            }
            connection.cancel()
        }
        registerCancellationHandler(for: connection) {
            streamTask.cancel()
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
        let session = makeSSESession(
            for: connection,
            format: .sse,
            enableKeepalive: false,
            dateEncodingStrategy: .iso8601
        )
        session.begin(startKeepalive: false)

        let streamTask = Task {
            while !Task.isCancelled {
                if session.isClosed { break }
                let model = self.dashboardModel()
                session.sendSSEDataFrame(model)
                let intervalNs: UInt64 = (model.generation == .prefill || model.generation == .decode)
                    ? 150_000_000
                    : 500_000_000
                try? await Task.sleep(nanoseconds: intervalNs)
            }
            connection.cancel()
        }
        registerCancellationHandler(for: connection) {
            streamTask.cancel()
        }
    }

    func handleLogsStream(connection: NWConnection) {
        let subState = OSAllocatedUnfairLock<(id: UUID?, unsubscribed: Bool)>(initialState: (nil, false))
        let unsubscribeOnce: @Sendable () -> Void = { [weak self] in
            let id = subState.withLock { state -> UUID? in
                state.unsubscribed = true
                let current = state.id
                state.id = nil
                return current
            }
            if let id { self?.logger.unsubscribe(id) }
        }
        let session = makeSSESession(
            for: connection,
            format: .sse,
            enableKeepalive: false,
            onError: unsubscribeOnce
        )
        session.begin(startKeepalive: false)

        for line in self.logger.getRecentLogs() {
            if session.isClosed { break }
            session.sendSSEDataFrame(["line": line])
        }

        let subId = self.logger.subscribe { line in
            session.sendSSEDataFrame(["line": line])
        }
        let alreadyUnsubscribed = subState.withLock { state -> Bool in
            if state.unsubscribed { return true }
            state.id = subId
            return false
        }
        if alreadyUnsubscribed || session.isClosed {
            unsubscribeOnce()
            self.logger.unsubscribe(subId)
        }

        registerCancellationHandler(for: connection) {
            unsubscribeOnce()
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
