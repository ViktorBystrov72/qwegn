import Foundation
import Network
import QWCore
import QWEngine
import QWHAL
import QWVision
import Metal

public final class QWHttpServer: @unchecked Sendable {
    // Internal, not private: the endpoints live in extensions across
    // several files, and a member is invisible from them.
    let actor: QWEngineLifecycleActor
    let host: String
    let port: UInt16
    let apiKey: String?
    let corsAllowedOrigins: [String]
    let allowUnauthenticatedPublicBind: Bool
    let logger: QWServerLogger
    var listener: NWListener?
    let queue = DispatchQueue(label: "qwg.server.nw", qos: .userInitiated)
    let connectionStateLock = NSLock()
    var cancellationHubs: [ObjectIdentifier: QWConnectionCancellationHub] = [:]
    var connectionOrigins: [ObjectIdentifier: String] = [:]

    public init(
        actor: QWEngineLifecycleActor,
        host: String = "127.0.0.1",
        port: Int = 8080,
        logger: QWServerLogger = .shared,
        apiKey: String? = nil,
        corsAllowedOrigins: [String] = [],
        allowUnauthenticatedPublicBind: Bool = false
    ) {
        self.actor = actor
        self.host = host
        self.port = UInt16(clamping: max(0, port))
        self.logger = logger
        let trimmedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.apiKey = (trimmedKey?.isEmpty == false) ? trimmedKey : nil
        self.corsAllowedOrigins = corsAllowedOrigins
        self.allowUnauthenticatedPublicBind = allowUnauthenticatedPublicBind
    }

    public func start() throws {
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty, !trimmedHost.contains(where: \.isWhitespace) else {
            throw QWHttpServerError.invalidListenHost(host)
        }
        if !QWHttpSecurity.isLoopbackHost(trimmedHost) && !allowUnauthenticatedPublicBind && apiKey == nil {
            throw QWHttpServerError.unauthenticatedNonLoopbackBind(host: trimmedHost)
        }

        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let nwPort: NWEndpoint.Port = port == 0 ? .any : (NWEndpoint.Port(rawValue: port) ?? .any)
        let listener: NWListener
        if trimmedHost == "0.0.0.0" || trimmedHost == "::" || trimmedHost == "*" {
            listener = try NWListener(using: params, on: nwPort)
        } else {
            params.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(trimmedHost), port: nwPort)
            listener = try NWListener(using: params)
        }
        self.listener = listener

        let readySem = DispatchSemaphore(value: 0)
        let stateLock = NSLock()
        nonisolated(unsafe) var startupError: NWError? = nil
        nonisolated(unsafe) var signaled = false

        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                let activePort = listener.port?.rawValue ?? self.port
                self.logger.log(.info, "HTTP Server listening on http://\(trimmedHost):\(activePort)")
                stateLock.lock()
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            case .failed(let error):
                self.logger.log(.error, "HTTP Server failed: \(error)")
                stateLock.lock()
                startupError = error
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            case .cancelled:
                self.logger.log(.info, "HTTP Server cancelled.")
                stateLock.lock()
                let shouldSignal = !signaled
                signaled = true
                stateLock.unlock()
                if shouldSignal { readySem.signal() }
            default:
                break
            }
        }

        listener.newConnectionHandler = { [weak self] connection in
            self?.handleConnection(connection)
        }

        listener.start(queue: queue)
        _ = readySem.wait(timeout: .now() + 3.0)
        stateLock.lock()
        let err = startupError
        stateLock.unlock()
        if let err {
            listener.cancel()
            self.listener = nil
            throw err
        }
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    private func handleConnection(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        let hub = QWConnectionCancellationHub()
        connectionStateLock.lock()
        cancellationHubs[id] = hub
        connectionStateLock.unlock()
        hub.attach(to: connection)
        hub.register { [weak self] in
            self?.cleanupConnectionState(for: id)
        }
        connection.start(queue: queue)
        receiveNextChunk(connection: connection, parser: QWHttpIncrementalParser())
    }

    private func cleanupConnectionState(for id: ObjectIdentifier) {
        connectionStateLock.lock()
        cancellationHubs.removeValue(forKey: id)
        connectionOrigins.removeValue(forKey: id)
        connectionStateLock.unlock()
    }

    func registerCancellationHandler(for connection: NWConnection, _ handler: @escaping @Sendable () -> Void) {
        let id = ObjectIdentifier(connection)
        connectionStateLock.lock()
        let hub = cancellationHubs[id]
        connectionStateLock.unlock()
        if let hub {
            hub.register(handler)
            return
        }
        switch connection.state {
        case .cancelled, .failed:
            handler()
        default:
            break
        }
    }

    func isConnectionTerminated(_ connection: NWConnection) -> Bool {
        let id = ObjectIdentifier(connection)
        connectionStateLock.lock()
        let hub = cancellationHubs[id]
        connectionStateLock.unlock()
        if let hub {
            return hub.isTerminated
        }
        switch connection.state {
        case .cancelled, .failed:
            return true
        default:
            return false
        }
    }

    func generationShouldStop(for connection: NWConnection) -> (@Sendable () -> Bool) {
        { [weak self] in
            Task.isCancelled || self?.isPeerGone(connection) == true
        }
    }

    func corsHeaderLines(for connection: NWConnection) -> String {
        let id = ObjectIdentifier(connection)
        connectionStateLock.lock()
        let origin = connectionOrigins[id]
        connectionStateLock.unlock()
        guard let allowed = QWHttpSecurity.resolvedAllowedOrigin(origin, allowlist: corsAllowedOrigins) else {
            return ""
        }
        return "Access-Control-Allow-Origin: \(allowed)\r\nVary: Origin\r\n"
    }

    private func receiveNextChunk(connection: NWConnection, parser: QWHttpIncrementalParser) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil {
                connection.cancel()
                return
            }
            guard let data, !data.isEmpty else {
                if isComplete { connection.cancel() }
                return
            }
            var nextParser = parser
            do {
                switch try nextParser.append(data) {
                case .needMoreData:
                    if isComplete {
                        self.sendSimpleResponse(connection, status: 400, body: "Incomplete HTTP request\n")
                        return
                    }
                    self.receiveNextChunk(connection: connection, parser: nextParser)
                case .complete(let request):
                    let id = ObjectIdentifier(connection)
                    if let origin = request.headers["origin"] {
                        self.connectionStateLock.lock()
                        self.connectionOrigins[id] = origin
                        self.connectionStateLock.unlock()
                    }
                    self.dispatchRoute(
                        connection: connection,
                        method: request.method.uppercased(),
                        path: request.path,
                        headers: request.headers,
                        body: request.body
                    )
                }
            } catch let parseError as QWHttpParseError {
                self.sendSimpleResponse(connection, status: parseError.httpStatus, body: "\(parseError.message)\n")
            } catch {
                self.sendSimpleResponse(connection, status: 400, body: "Malformed HTTP request\n")
            }
        }
    }

    private func dispatchRoute(
        connection: NWConnection,
        method: String,
        path: String,
        headers: [String: String],
        body: Data
    ) {
        if method == "OPTIONS" {
            var corsHeader = "HTTP/1.1 204 No Content\r\n"
            corsHeader += corsHeaderLines(for: connection)
            corsHeader += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            corsHeader += "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key\r\n"
            corsHeader += "Access-Control-Max-Age: 86400\r\n"
            corsHeader += "Content-Length: 0\r\nConnection: close\r\n\r\n"
            connection.send(content: Data(corsHeader.utf8), completion: .contentProcessed({ _ in connection.cancel() }))
            return
        }

        if let expectedKey = apiKey {
            let presented = QWHttpSecurity.extractPresentedToken(headers: headers)
            guard let presented, QWHttpSecurity.constantTimeEquals(presented, expectedKey) else {
                sendSimpleResponse(connection, status: 401, body: "Unauthorized\n")
                return
            }
        }

        // Clean path (strip query params)
        let cleanPath = path.components(separatedBy: "?").first ?? path

        let routeTask = Task {
            do {
                switch (method, cleanPath) {
                case ("POST", "/v1/chat/completions"):
                    try await self.handleChatCompletions(connection: connection, body: body)

                case ("POST", "/api/chat"):
                    try await self.handleOllamaChat(connection: connection, body: body)

                case ("POST", "/api/generate"):
                    try await self.handleOllamaGenerate(connection: connection, body: body)

                case ("POST", "/v1/messages"):
                    try await self.handleMessages(connection: connection, body: body)

                case ("POST", "/v1/models/load"), ("POST", "/api/load"):
                    if !body.isEmpty,
                       let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                       let requestedModel = payload["model"] as? String {
                        try await self.actor.selectModelIfNeeded(requestedModel)
                    }
                    _ = try await self.actor.loadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("POST", "/v1/models/unload"), ("POST", "/api/unload"):
                    let reclaimed = try await self.actor.unloadExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: false, memoryReclaimedMB: reclaimed)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/models"), ("GET", "/models"):
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    var seenIds = Set<String>()
                    var models: [ModelObject] = []
                    // Each entry carries the one fact the model really
                    // has: when its bytes last changed on disk.
                    func addModel(id: String, loaded: Bool, path: String?) {
                        guard !id.isEmpty, seenIds.insert(id).inserted else { return }
                        models.append(ModelObject(id: id, loaded: loaded,
                                                  created: QWModelDiskStats.modifiedUnix(of: path)))
                    }
                    let activeAlias = settings.alias(for: modelName)
                    if !activeAlias.isEmpty {
                        addModel(id: activeAlias, loaded: isLoaded, path: settings.modelPackPath)
                    }
                    addModel(id: modelName, loaded: isLoaded, path: settings.modelPackPath)
                    for disc in QWServerSettings.scanModelsDirectory() where !disc.packPath.isEmpty && disc.supported {
                        let loadedHere = isLoaded && disc.name == modelName
                        let alias = settings.alias(for: disc.name)
                        if !alias.isEmpty {
                            addModel(id: alias, loaded: loadedHere, path: disc.packPath)
                        }
                        addModel(id: disc.name, loaded: loadedHere, path: disc.packPath)
                    }
                    let resp = ModelListResponse(models: models)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/models/discovered"):
                    let models = QWServerSettings.scanModelsDirectory()
                    self.sendJsonResponse(connection, status: 200, object: ["models": models])

                case ("GET", let path) where path.hasPrefix("/v1/models/") || path.hasPrefix("/models/"):
                    let prefix = path.hasPrefix("/v1/models/") ? "/v1/models/" : "/models/"
                    let requestedId = String(path.dropFirst(prefix.count))
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    // The model is known when it is the resident one,
                    // or a supported model the scan found. Anything
                    // else is a 404 naming it (OpenAI's shape).
                    let activeAlias = settings.alias(for: modelName)
                    let displayModel = activeAlias.isEmpty ? modelName : activeAlias
                    var path2: String?
                    var loaded = false
                    if requestedId.isEmpty {
                        path2 = settings.modelPackPath
                        loaded = isLoaded
                    } else if requestedId == modelName || requestedId == activeAlias {
                        path2 = settings.modelPackPath
                        loaded = isLoaded
                    } else if let disc = QWServerSettings.scanModelsDirectory().first(where: {
                        $0.name == requestedId || settings.alias(for: $0.name) == requestedId
                    }), disc.supported {
                        path2 = disc.packPath
                        loaded = isLoaded && disc.name == modelName
                    } else {
                        let err = ServerErrorPayload(message: "The model '\(requestedId)' does not exist.",
                                                      type: "invalid_request_error", code: 404)
                        self.sendJsonResponse(connection, status: 404, object: err)
                        return
                    }
                    let resp = ModelObject(id: requestedId.isEmpty ? displayModel : requestedId,
                                           loaded: loaded,
                                           created: QWModelDiskStats.modifiedUnix(of: path2))
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/api/version"):
                    self.sendJsonResponse(connection, status: 200, object: ["engine": "qwegine", "version": QWVersion.current])

                case ("POST", "/api/show"):
                    let modelName = self.actor.getModelName()
                    let params = self.actor.getLoadedParamsInfo()
                    let details: [String: Any] = [
                        "format": "qwg",
                        "family": "qwen3",
                        "parameter_size": "27B",
                        "quantization_level": params?.headStorage ?? "Q4g"
                    ]
                    let resp: [String: Any] = [
                        "modelfile": "# Modelfile for Qwegine\nFROM \(modelName)",
                        "parameters": "context_length \(params?.contextBudget ?? 131072)",
                        "details": details
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: resp) {
                        self.sendRawData(connection, status: 200, contentType: "application/json", data: data)
                    } else {
                        self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])
                    }

                case ("GET", "/api/tags"), ("GET", "/api/ps"):
                    let modelName = self.actor.getModelName()
                    let isLoaded = self.actor.isLoaded()
                    let settings = QWSettingsManager.shared.loadSettings()
                    let activeAlias = settings.alias(for: modelName)
                    let displayModel = activeAlias.isEmpty ? modelName : activeAlias
                    if cleanPath == "/api/ps" {
                        // The size of a resident model is the bytes its
                        // files occupy on disk, read now; the date is
                        // when those bytes last changed.
                        let models = isLoaded ? [OllamaPsResponse.RunningModel(
                            name: displayModel, model: displayModel,
                            size: QWModelDiskStats.sizeBytes(of: settings.modelPackPath),
                            expiresAt: "never")] : []
                        self.sendJsonResponse(connection, status: 200, object: OllamaPsResponse(models: models))
                    } else {
                        var seen = Set<String>()
                        var tags: [OllamaModelTag] = []
                        func addTag(_ name: String, path: String?) {
                            guard !name.isEmpty, seen.insert(name).inserted else { return }
                            tags.append(OllamaModelTag(name: name, model: name,
                                                       modifiedAt: QWModelDiskStats.modifiedRFC3339(of: path),
                                                       size: QWModelDiskStats.sizeBytes(of: path)))
                        }
                        if !activeAlias.isEmpty { addTag(activeAlias, path: settings.modelPackPath) }
                        addTag(modelName, path: settings.modelPackPath)
                        for disc in QWServerSettings.scanModelsDirectory() where !disc.packPath.isEmpty && disc.supported {
                            let alias = settings.alias(for: disc.name)
                            if !alias.isEmpty { addTag(alias, path: disc.packPath) }
                            addTag(disc.name, path: disc.packPath)
                        }
                        self.sendJsonResponse(connection, status: 200, object: OllamaTagsResponse(models: tags))
                    }

                case ("GET", "/metrics"):
                    let body = QWEngineTelemetryRegistry.shared.formatPrometheus()
                    self.sendRawData(connection, status: 200, contentType: "text/plain; version=0.0.4; charset=utf-8", data: Data(body.utf8))

                case ("GET", "/v1/metrics"), ("GET", "/api/metrics"):
                    let snapshot = QWEngineTelemetryRegistry.shared.snapshot()
                    self.sendJsonResponse(connection, status: 200, object: snapshot)

                case ("GET", "/v1/settings"), ("GET", "/api/settings"):
                    var settings = QWSettingsManager.shared.loadSettings()
                    if !settings.apiKey.isEmpty {
                        settings.apiKey = "********"
                    }
                    self.sendJsonResponse(connection, status: 200, object: settings)

                case ("POST", "/v1/settings"), ("POST", "/api/settings"):
                    let decoder = JSONDecoder()
                    do {
                        let oldSettings = QWSettingsManager.shared.loadSettings()
                        var newSettings = try decoder.decode(QWServerSettings.self, from: body)
                        let rawObj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
                        if newSettings.apiKey == "********" || rawObj?["apiKey"] == nil {
                            newSettings.apiKey = oldSettings.apiKey
                        }
                        let modelChanged = (newSettings.modelPackPath != oldSettings.modelPackPath)
                            || (newSettings.modelName != oldSettings.modelName)
                        if modelChanged {
                            let ref = newSettings.resolveModelReference(
                                newSettings.modelPackPath.isEmpty ? newSettings.modelName : newSettings.modelPackPath
                            )
                            let targetModelName = ref.modelName.isEmpty ? newSettings.modelName : ref.modelName
                            if let profile = QWSettingsManager.shared.loadProfile(forModel: targetModelName) {
                                newSettings.applyModelConfigDefaults(packPath: ref.packPath)
                                newSettings.apply(knobProfile: profile)
                            } else if newSettings.maxContext == oldSettings.maxContext {
                                newSettings.applyModelConfigDefaults(packPath: ref.packPath)
                            }
                        }
                        if let mounted = self.actor.getMountedMemoryPackInfo(), mounted.enforcesKVMode {
                            newSettings.kvMode = mounted.enforcedKVMode
                        }
                        try newSettings.validateBeforeLaunch()
                        try QWSettingsManager.shared.saveSettings(newSettings)
                        if !modelChanged {
                            try await self.actor.setVisionLoaded(newSettings.loadVision)
                        }
                        self.sendJsonResponse(connection, status: 200, object: ["status": "ok", "message": "Settings saved successfully"])
                    } catch {
                        let err = ServerErrorPayload(message: "Failed to parse or save settings: \(error.localizedDescription)", code: 400)
                        self.sendJsonResponse(connection, status: 400, object: err)
                    }

                case ("POST", "/v1/memory/mount"), ("POST", "/api/memory/mount"):
                    guard !body.isEmpty,
                          let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                          let repoPath = payload["path"] as? String,
                          !repoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        let err = ServerErrorPayload(message: "Missing 'path' in memory mount request.", code: 400)
                        self.sendJsonResponse(connection, status: 400, object: err)
                        return
                    }
                    let mounted = try await self.actor.mountMemoryPack(at: repoPath)
                    self.sendJsonResponse(connection, status: 200, object: mounted)

                case ("POST", "/v1/memory/unmount"), ("POST", "/api/memory/unmount"):
                    await self.actor.unmountMemoryPack()
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/settings/reveal"), ("POST", "/api/settings/reveal"):
                    let path = QWPaths.settingsFilePath
                    let p = Process()
                    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                    p.arguments = ["-R", path]
                    // The reveal either happened or it did not. An
                    // unconditional 200 would report success the
                    // process never established.
                    do {
                        try p.run()
                    } catch {
                        let err = ServerErrorPayload(message: "Could not open the Finder window: \(error.localizedDescription)", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    p.waitUntilExit()
                    if p.terminationStatus != 0 {
                        let err = ServerErrorPayload(message: "Finder rejected the request (exit status \(p.terminationStatus)).", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/logs/reveal"), ("POST", "/api/logs/reveal"):
                    let settings = QWSettingsManager.shared.loadSettings()
                    let logDir = settings.logDir.expandingTildePath
                    try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
                    let p = Process()
                    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                    p.arguments = [logDir]
                    do {
                        try p.run()
                    } catch {
                        let err = ServerErrorPayload(message: "Could not open the Finder window: \(error.localizedDescription)", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    p.waitUntilExit()
                    if p.terminationStatus != 0 {
                        let err = ServerErrorPayload(message: "Finder rejected the request (exit status \(p.terminationStatus)).", code: 500)
                        self.sendJsonResponse(connection, status: 500, object: err)
                        return
                    }
                    self.sendJsonResponse(connection, status: 200, object: ["status": "ok"])

                case ("POST", "/v1/engine/restart"), ("POST", "/api/restart"):
                    _ = try await self.actor.restartExplicit()
                    let modelName = self.actor.getModelName()
                    let resp = ModelLifecycleResponse(status: "ok", model: modelName, loaded: true)
                    self.sendJsonResponse(connection, status: 200, object: resp)

                case ("GET", "/v1/engine/info"), ("GET", "/api/engine/info"):
                    self.handleEngineInfoRoute(connection: connection)

                case ("GET", "/v1/metrics/stream"):
                    self.handleMetricsStream(connection: connection)

                case ("GET", "/v1/dashboard/stream"):
                    self.handleDashboardStream(connection: connection)

                case ("GET", "/v1/logs/stream"):
                    self.handleLogsStream(connection: connection)

                case ("GET", let path) where path.hasPrefix("/ui"):
                    self.handleStaticUI(connection: connection, path: path)

                case ("GET", "/v1/dashboard"):
                    let dashboard = self.dashboardModel()
                    self.sendIsoJsonResponse(connection, status: 200, object: dashboard)

                case ("GET", "/health"), ("GET", "/"):
                    let engineState = self.actor.engineState()
                    let modelName = self.actor.getModelName()
                    // The server is alive and answering, so the status
                    // code says so. What the engine is doing is the
                    // phase — a failed engine is not an unreachable one.
                    let health = QWHealthResponse(status: engineState.lifecycle.phase.rawValue,
                                                  engine: "qwegine",
                                                  model: modelName, loaded: engineState.isLoaded,
                                                  lifecycle: engineState.lifecycle)
                    let data = try JSONEncoder().encode(health)
                    self.sendRawData(connection, status: 200, contentType: "application/json", data: data)

                default:
                    let err = ServerErrorPayload(message: "Not found: \(method) \(cleanPath)", type: "invalid_request_error", code: 404)
                    self.sendJsonResponse(connection, status: 404, object: err)
                }
            } catch let mmError as QWMultimodalError {
                let err = ServerErrorPayload(message: "\(mmError)", type: "invalid_request_error", code: 400)
                self.logger.log(.warn, "HTTP \(method) \(cleanPath) bad multimodal request: \(mmError)")
                self.sendJsonResponse(connection, status: 400, object: err)
            } catch {
                if Task.isCancelled || self.isPeerGone(connection) { return }
                let err = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
                // The reason goes to the log, not only to the response body:
                // a 500 with no recorded reason is undebuggable after the fact.
                self.logger.log(.error, "HTTP \(method) \(cleanPath) failed: \(error)")
                self.sendJsonResponse(connection, status: 500, object: err)
            }
        }
        registerCancellationHandler(for: connection) {
            routeTask.cancel()
        }
    }

    // The stream endpoints and the static UI live in QWHttpServer+Streams.swift.
    // POST /v1/chat/completions lives in QWHttpServer+ChatCompletions.swift.

}
