// QWHttpServer+Routes.swift — Engine info, dashboard view model, and JSON response helpers.

import Foundation
import Network
import QWCore
import QWEngine
import QWHAL
import QWVision
import Metal

extension QWHttpServer {
    func handleEngineInfoRoute(connection: NWConnection) {
        let hw = QWMachineHardwareInfo.current()
        let caps = QWDeviceCaps.current
        let config = self.actor.getConfig()
        let engineState = self.actor.engineState()
        let isLoaded = engineState.isLoaded
        let params = self.actor.getLoadedParamsInfo()

        let metalDevName = caps?.name ?? MTLCreateSystemDefaultDevice()?.name
        let isTensor = caps?.isMetal4TensorCapable ?? false
        // The Metal family the device really supports, from
        // the probes the HAL made. A device that supports
        // none of them has no version to report.
        let metalVer: String? = if caps?.familyApple11 == true || caps?.familyApple10 == true {
            "Metal 4"
        } else if caps?.familyApple9 == true {
            "Metal 3"
        } else {
            nil
        }
        // The HAL's DRAM bandwidth is a constant from one
        // machine, not a measurement of this one. It is not
        // published as a fact about this host.
        let dramBw: Double? = nil

        let hwFacts = QWEngineInfoResponse.HardwareFacts(
            modelIdentifier: hw.modelIdentifier,
            cpuBrand: hw.cpuBrand,
            physicalCores: hw.physicalCores,
            logicalCores: hw.logicalCores,
            physicalMemoryBytes: hw.physicalMemoryBytes,
            physicalMemoryGB: hw.physicalMemoryGB,
            metalDeviceName: metalDevName,
            metalVersion: metalVer,
            isTensorCapable: isTensor,
            dramBwGBs: dramBw
        )

        let srvParams = QWEngineInfoResponse.ServerParameters(
            modelName: self.actor.getModelName(),
            modelPackPath: config.packPath,
            // The head storage of the resident model. With no
            // model resident there is no head, and a
            // quantization that describes nothing is a claim
            // about nothing.
            headStorage: params?.headStorage,
            contextBudget: config.maxContext,
            kvMode: String(describing: config.kvMode),
            mtpDrafts: config.mtpDrafts,
            mtpBackend: config.mtpBackend,
            gdnPrefillMode: config.gdnPrefill,
            prefillScratchEviction: config.evictPrefillScratch,
            lsaMode: config.lsaConfig.mode.rawValue,
            hostLogicalCores: hw.logicalCores,
            isLoaded: isLoaded
        )

        let cacheInfo = QWEngineInfoResponse.CacheConfigInfo(
            warmLimitBytes: config.cacheConfig.warmByteLimit,
            ssdLimitBytes: config.cacheConfig.ssdByteLimit,
            cacheDir: config.cacheConfig.cacheDirectory.path
        )

        let snap = QWEngineTelemetryRegistry.shared.snapshot()
        let hwUtil = snap.hardware ?? QWSystemMetricsSampler.shared.sample()
        let info = QWEngineInfoResponse(
            status: engineState.lifecycle.phase.rawValue,
            version: QWVersion.current,
            pid: Int(ProcessInfo.processInfo.processIdentifier),
            hardware: hwFacts,
            parameters: srvParams,
            cacheConfig: cacheInfo,
            hardwareUtilization: hwUtil,
            lifecycle: engineState.lifecycle,
            sources: config.sourceSummary.mapValues { QWKnobSource.normalise($0).rawValue },
            capabilities: engineState.capabilities.map(\.rawValue).sorted(),
            mountedMemoryPack: self.actor.getMountedMemoryPackInfo(),
            endpoints: [
                "chat": "POST /v1/chat/completions",
                "responses": "POST /v1/responses",
                "completions": "POST /v1/completions",
                "tokenizer_info": "GET /tokenizer_info",
                "tokenize": "POST /tokenize",
                "detokenize": "POST /detokenize",
                "models": "GET /v1/models",
                "metrics": "GET /v1/metrics",
                "metrics_stream": "GET /v1/metrics/stream",
                "logs_stream": "GET /v1/logs/stream",
                "engine_info": "GET /v1/engine/info",
                "memory_inspect": "POST /v1/memory/inspect",
                "memory_index": "POST /v1/memory/index",
                "memory_mount": "POST /v1/memory/mount",
                "memory_unmount": "POST /v1/memory/unmount",
                "restart": "POST /v1/engine/restart",
                "update": "GET /v1/update",
                "ui": "GET /ui"
            ]
        )
        self.sendJsonResponse(connection, status: 200, object: info)

    }

    /// The published model. The route and the stream publish the same
    /// thing, so a renderer cannot see two truths.
    func dashboardModel() -> QWDashboardViewModel {
        let engineState = self.actor.engineState()
        let config = self.actor.getConfig()
        // A knob the model cannot honour is not offered: the renderer
        // must not be asked to hide what it cannot show.
        let offered = Set(engineState.capabilities)
        let knobs = config.knobValues.filter { id, _ in
            guard let descriptor = QWKnobRegistry.descriptor(id) else { return true }
            guard let capability = descriptor.requiredCapability else { return true }
            return offered.contains(capability)
        }
        return QWDashboardViewModel.build(
            lifecycle: engineState.lifecycle,
            modelId: engineState.loadedModelName,
            capabilities: engineState.capabilities,
            knobs: knobs,
            knobSources: QWKnobSource.sources(from: config.sourceSummary,
                                              for: Array(knobs.keys)),
            facts: self.dashboardFacts(),
            requests: QWRequestRecordStore.shared.recent(QWRequestRecordStore.capacity))
    }

    /// The facts about the machine and the resident model, for the
    /// renderers. A fact nobody could establish is absent.
    func dashboardFacts() -> QWDashboardFacts {
        let params = self.actor.getLoadedParamsInfo()
        let config = self.actor.getConfig()
        let caps = QWDeviceCaps.current
        let hw = QWMachineHardwareInfo.current()
        let metalFamily: String? = if caps?.familyApple11 == true || caps?.familyApple10 == true {
            "Metal 4"
        } else if caps?.familyApple9 == true {
            "Metal 3"
        } else {
            nil
        }
        let mounted = self.actor.getMountedMemoryPackInfo()
        let currentHW = QWSystemMetricsSampler.shared.current()
        return QWDashboardFacts(
            quantization: params?.headStorage,
            hostLogicalCores: hw.logicalCores,
            gdnPrefillMode: config.gdnPrefill,
            gpuName: caps?.name ?? MTLCreateSystemDefaultDevice()?.name,
            metalFamily: metalFamily,
            physicalMemoryGB: hw.physicalMemoryBytes > 0 ? hw.physicalMemoryGB : nil,
            lsaMode: config.lsaConfig.mode.rawValue,
            mountedMemoryPack: mounted.map { "\($0.packName) (\($0.enforcedKVMode.uppercased()))" },
            mountedMemoryPackRoot: mounted?.rootPath,
            mountedMemoryPackFiles: mounted?.fileCount,
            mountedMemoryPackTokens: mounted?.tokenCount,
            mountedMemoryPackDelta: mounted?.modifiedFileCount,
            mountedMemoryPackKV: mounted?.enforcedKVMode,
            coolingPolicy: QWFanController.shared.currentPolicy.rawValue,
            fanControlStatus: currentHW.fanControlStatus)
    }

    /// A response whose dates are ISO-8601 instants, so a client in any
    /// language can read them.
    func sendIsoJsonResponse<T: Encodable>(_ connection: NWConnection, status: Int, object: T) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(object)
            sendRawData(connection, status: status, contentType: "application/json", data: data)
        } catch {
            sendSimpleResponse(connection, status: 500, body: "{\"error\": \"JSON encoding error\"}\n")
        }
    }

    func sendJsonResponse<T: Encodable>(_ connection: NWConnection, status: Int, object: T, extraHeaders: [String] = []) {
        do {
            let data = try JSONEncoder().encode(object)
            sendRawData(connection, status: status, contentType: "application/json", data: data, extraHeaders: extraHeaders)
        } catch {
            sendSimpleResponse(connection, status: 500, body: "{\"error\": \"JSON encoding error\"}\n")
        }
    }

    func sendRawData(_ connection: NWConnection, status: Int, contentType: String, data: Data, extraHeaders: [String] = []) {
        var header = "HTTP/1.1 \(status) \(status == 200 ? "OK" : "Error")\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(data.count)\r\n"
        for line in extraHeaders {
            header += "\(line)\r\n"
        }
        header += corsHeaderLines(for: connection)
        header += "Connection: close\r\n\r\n"

        var packet = Data(header.utf8)
        packet.append(data)
        let id = ObjectIdentifier(connection)
        connection.send(
            content: packet,
            contentContext: .finalMessage,
            isComplete: true,
            completion: .contentProcessed { [weak self] _ in
                self?.cleanupConnectionState(for: id)
                connection.cancel()
            }
        )
    }

    func sendSimpleResponse(_ connection: NWConnection, status: Int, body: String) {
        sendRawData(connection, status: status, contentType: "text/plain", data: Data(body.utf8))
    }

    func sendOpenAIError(
        _ connection: NWConnection,
        status: Int,
        message: String,
        type: String = "invalid_request_error",
        code: String? = nil
    ) {
        var detail: [String: Any] = ["message": message, "type": type]
        if let code { detail["code"] = code }
        let payload: [String: Any] = ["error": detail]
        if let data = try? JSONSerialization.data(withJSONObject: payload) {
            sendRawData(connection, status: status, contentType: "application/json", data: data)
        } else {
            sendSimpleResponse(connection, status: status, body: message)
        }
    }

    private static let jsonRequiredPostRoutes: Set<String> = [
        "/v1/chat/completions", "/chat/completions",
        "/v1/completions", "/completions",
        "/tokenize", "/v1/tokenize",
        "/detokenize", "/v1/detokenize",
        "/v1/responses", "/responses",
        "/api/chat", "/api/generate",
        "/v1/messages/count_tokens", "/messages/count_tokens",
        "/v1/messages", "/messages",
        "/v1/settings", "/api/settings",
        "/v1/memory/inspect", "/api/memory/inspect",
        "/v1/memory/index", "/api/memory/index",
        "/v1/memory/mount", "/api/memory/mount",
        "/v1/update/skip", "/api/update/skip",
    ]

    func validateMutationRequest(
        connection: NWConnection,
        method: String,
        cleanPath: String,
        headers: [String: String],
        body: Data
    ) -> Bool {
        guard method != "GET" && method != "HEAD" && method != "OPTIONS" else {
            return true
        }
        let originHeader = headers["origin"]
        let secFetchSite = headers["sec-fetch-site"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let originAllowed: Bool = if let originHeader {
            QWHttpSecurity.resolvedAllowedOrigin(originHeader, allowlist: corsAllowedOrigins) != nil
        } else {
            secFetchSite != "cross-site"
        }
        guard originAllowed else {
            let err = ServerErrorPayload(message: "Forbidden cross-origin request", type: "invalid_request_error", code: 403)
            sendJsonResponse(connection, status: 403, object: err)
            return false
        }
        let requireJSON = Self.jsonRequiredPostRoutes.contains(cleanPath) || !body.isEmpty
        if !QWHttpSecurity.isValidMutationOrigin(
            headers: headers,
            allowlist: corsAllowedOrigins,
            requireJSONContentType: requireJSON
        ) {
            let err = ServerErrorPayload(
                message: "Unsupported Media Type: Content-Type must be application/json",
                type: "invalid_request_error",
                code: 415
            )
            sendJsonResponse(connection, status: 415, object: err)
            return false
        }
        return true
    }

    func revealDirectoryInFinder(
        connection: NWConnection,
        rawDirectoryPath: String,
        createIfInQwgRoot: Bool
    ) async {
        let resolvedDir: URL
        do {
            resolvedDir = try QWHttpSecurity.validateSafeLogDirectory(
                rawDirectoryPath,
                createIfInQwgRoot: createIfInQwgRoot
            )
        } catch let err as QWHttpServerError {
            let payload = ServerErrorPayload(message: err.localizedDescription, code: 400)
            sendJsonResponse(connection, status: 400, object: payload)
            return
        } catch {
            let payload = ServerErrorPayload(
                message: "Could not prepare directory for reveal: \(error.localizedDescription)",
                code: 500
            )
            sendJsonResponse(connection, status: 500, object: payload)
            return
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolvedDir.path, isDirectory: &isDir), isDir.boolValue else {
            let payload = ServerErrorPayload(
                message: "Directory does not exist: \(resolvedDir.path)",
                code: 404
            )
            sendJsonResponse(connection, status: 404, object: payload)
            return
        }
        do {
            let status = try await runFinderReveal(arguments: ["--", resolvedDir.path])
            if status != 0 {
                let payload = ServerErrorPayload(
                    message: "Finder rejected the request (exit status \(status)).",
                    code: 500
                )
                sendJsonResponse(connection, status: 500, object: payload)
                return
            }
            sendJsonResponse(connection, status: 200, object: ["status": "ok"])
        } catch {
            let payload = ServerErrorPayload(
                message: "Could not open the Finder window: \(error.localizedDescription)",
                code: 500
            )
            sendJsonResponse(connection, status: 500, object: payload)
        }
    }

    private func runFinderReveal(arguments: [String]) async throws -> Int32 {
        if let runner = finderRevealRunner {
            return try await runner(arguments)
        }
        return try await runProcessAsync(executablePath: "/usr/bin/open", arguments: arguments)
    }

    func runProcessAsync(executablePath: String, arguments: [String]) async throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath)
        process.arguments = arguments
        return try await withCheckedThrowingContinuation { continuation in
            let lock = NSLock()
            nonisolated(unsafe) var resumed = false
            process.terminationHandler = { proc in
                let shouldResume = lock.withLock {
                    guard !resumed else { return false }
                    resumed = true
                    return true
                }
                if shouldResume {
                    continuation.resume(returning: proc.terminationStatus)
                }
            }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                let shouldResume = lock.withLock {
                    guard !resumed else { return false }
                    resumed = true
                    return true
                }
                if shouldResume {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func handleMemoryInspectRoute(connection: NWConnection, body: Data) async throws {
        guard !body.isEmpty,
              let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let repoPath = payload["path"] as? String,
              !repoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = ServerErrorPayload(message: "Missing 'path' in memory inspect request.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let validatedRoot = try actor.resolveValidatedMemoryRootURL(at: repoPath, overrideRoots: allowedWorkspaceRoots)
        let info = try await actor.inspectMemoryPack(at: validatedRoot.path)
        sendJsonResponse(connection, status: 200, object: info)
    }

    func handleMemoryIndexRoute(connection: NWConnection, body: Data) async throws {
        guard !body.isEmpty,
              let req = try? JSONDecoder().decode(QWMemoryIndexRequest.self, from: body),
              !req.path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = ServerErrorPayload(message: "Missing or invalid JSON payload ('path' required) in memory index request.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let validatedRoot = try actor.resolveValidatedMemoryRootURL(at: req.path, overrideRoots: allowedWorkspaceRoots)
        let validatedReq = QWMemoryIndexRequest(
            path: validatedRoot.path,
            packName: req.packName,
            model: req.model,
            kvMode: req.kvMode,
            lsaMode: req.lsaMode,
            scope: req.scope,
            maxCortexTokens: req.maxCortexTokens,
            includeTests: req.includeTests,
            mountAfterIndex: req.mountAfterIndex,
            stream: req.stream,
            mtpBackend: req.mtpBackend,
            dflashDraftPackPath: req.dflashDraftPackPath
        )
        guard validatedReq.stream ?? true else {
            let result = try await actor.indexMemoryPack(request: validatedReq)
            sendJsonResponse(connection, status: 200, object: result)
            return
        }

        let session = makeSSESession(for: connection, format: .sse, enableKeepalive: true)
        session.begin()
        defer { session.stopAll() }
        let sendProgress: @Sendable (QWMemoryIndexProgressEvent) -> Void = { event in
            session.sendSSEDataFrame(event)
        }

        do {
            _ = try await actor.indexMemoryPack(request: validatedReq, onProgress: sendProgress)
        } catch {
            sendProgress(QWMemoryIndexProgressEvent(
                phase: "error",
                progressFraction: 0,
                message: error.localizedDescription,
                error: "\(error)"
            ))
        }
        session.finish(doneMarker: true)
    }

    private func catalogModelEntries() -> [(id: String, loaded: Bool, path: String?)] {
        let modelName = actor.getModelName()
        let isLoaded = actor.isLoaded()
        let settings = QWSettingsManager.shared.loadSettings()
        var seen = Set<String>()
        var entries: [(id: String, loaded: Bool, path: String?)] = []
        func appendIfNew(id: String, loaded: Bool, path: String?) {
            guard !id.isEmpty, seen.insert(id).inserted else { return }
            entries.append((id: id, loaded: loaded, path: path))
        }
        let activeAlias = settings.alias(for: modelName)
        if !activeAlias.isEmpty {
            appendIfNew(id: activeAlias, loaded: isLoaded, path: settings.modelPackPath)
        }
        appendIfNew(id: modelName, loaded: isLoaded, path: settings.modelPackPath)
        for disc in QWServerSettings.scanModelsDirectory() where !disc.packPath.isEmpty && disc.supported {
            let loadedHere = isLoaded && disc.name == modelName
            let alias = settings.alias(for: disc.name)
            if !alias.isEmpty {
                appendIfNew(id: alias, loaded: loadedHere, path: disc.packPath)
            }
            appendIfNew(id: disc.name, loaded: loadedHere, path: disc.packPath)
        }
        return entries
    }

    func handleModelsListRoute(connection: NWConnection) {
        let models = catalogModelEntries().map {
            ModelObject(id: $0.id, loaded: $0.loaded, created: QWModelDiskStats.modifiedUnix(of: $0.path))
        }
        sendJsonResponse(connection, status: 200, object: ModelListResponse(models: models))
    }

    func handleDiscoveredModelsRoute(connection: NWConnection) {
        struct DiscoveredInventoryResponse: Encodable {
            let models: [QWServerSettings.DiscoveredModel]
            let drafters: [QWServerSettings.DiscoveredDrafter]
        }
        let models = QWServerSettings.scanModelsDirectory()
        let drafters = QWServerSettings.scanDrafterSummaries(primaryModels: models)
        sendJsonResponse(
            connection,
            status: 200,
            object: DiscoveredInventoryResponse(models: models, drafters: drafters)
        )
    }

    func handleGetModelRoute(connection: NWConnection, requestedId: String) {
        let modelName = actor.getModelName()
        let isLoaded = actor.isLoaded()
        let settings = QWSettingsManager.shared.loadSettings()
        let activeAlias = settings.alias(for: modelName)
        let displayModel = activeAlias.isEmpty ? modelName : activeAlias
        let resolvedPath: String?
        let loaded: Bool
        if requestedId.isEmpty || requestedId == modelName || requestedId == activeAlias {
            resolvedPath = settings.modelPackPath
            loaded = isLoaded
        } else if let disc = QWServerSettings.scanModelsDirectory().first(where: {
            $0.name == requestedId || settings.alias(for: $0.name) == requestedId
        }), disc.supported {
            resolvedPath = disc.packPath
            loaded = isLoaded && disc.name == modelName
        } else {
            let err = ServerErrorPayload(
                message: "The model '\(requestedId)' does not exist.",
                type: "invalid_request_error",
                code: 404
            )
            sendJsonResponse(connection, status: 404, object: err)
            return
        }
        let resp = ModelObject(
            id: requestedId.isEmpty ? displayModel : requestedId,
            loaded: loaded,
            created: QWModelDiskStats.modifiedUnix(of: resolvedPath)
        )
        sendJsonResponse(connection, status: 200, object: resp)
    }

    func handleOllamaTagsOrPsRoute(connection: NWConnection, isPs: Bool) {
        if isPs {
            let modelName = actor.getModelName()
            let isLoaded = actor.isLoaded()
            let settings = QWSettingsManager.shared.loadSettings()
            let activeAlias = settings.alias(for: modelName)
            let displayModel = activeAlias.isEmpty ? modelName : activeAlias
            let models = isLoaded ? [OllamaPsResponse.RunningModel(
                name: displayModel, model: displayModel,
                size: QWModelDiskStats.sizeBytes(of: settings.modelPackPath),
                expiresAt: "never"
            )] : []
            sendJsonResponse(connection, status: 200, object: OllamaPsResponse(models: models))
            return
        }
        let tags = catalogModelEntries().map {
            OllamaModelTag(
                name: $0.id,
                model: $0.id,
                modifiedAt: QWModelDiskStats.modifiedRFC3339(of: $0.path),
                size: QWModelDiskStats.sizeBytes(of: $0.path)
            )
        }
        sendJsonResponse(connection, status: 200, object: OllamaTagsResponse(models: tags))
    }

    func handleSaveSettingsRoute(connection: NWConnection, body: Data) async {
        do {
            let oldSettings = QWSettingsManager.shared.loadSettings()
            var newSettings = try JSONDecoder().decode(QWServerSettings.self, from: body)
            let rawObj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
            if newSettings.apiKey == "********" || rawObj?["apiKey"] == nil {
                newSettings.apiKey = oldSettings.apiKey
            }
            let resolvedLogDir = try QWHttpSecurity.validateSafeLogDirectory(newSettings.logDir, createIfInQwgRoot: false)
            try QWHttpSecurity.validateSafeLogFileName(newSettings.logFileName)
            newSettings.logDir = resolvedLogDir.path
            newSettings.logFileName = newSettings.logFileName.trimmingCharacters(in: .whitespacesAndNewlines)
            let modelChanged = (newSettings.modelPackPath != oldSettings.modelPackPath)
                || (newSettings.modelName != oldSettings.modelName)
            let allowUnregistered = ProcessInfo.processInfo.environment["QWG_ALLOW_UNREGISTERED_MODEL_PATHS"] == "1"
            let resolvedPackPath: String
            if modelChanged {
                let targetRefInput = newSettings.modelPackPath.isEmpty ? newSettings.modelName : newSettings.modelPackPath
                let ref = newSettings.resolveModelReference(
                    targetRefInput,
                    allowArbitraryFilesystemPaths: allowUnregistered
                )
                if !allowUnregistered && !targetRefInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let packCand = ref.packPath.isEmpty ? targetRefInput.expandingTildePath : ref.packPath
                    let candURL = URL(fileURLWithPath: packCand)
                    let inRegistered = QWHttpSecurity.canonicalDirectoryContainment(candURL, inside: QWPaths.modelsDirectoryURL)
                    let matchesExisting = !oldSettings.modelPackPath.isEmpty
                        && candURL.resolvingSymlinksInPath().standardizedFileURL.path
                        == URL(fileURLWithPath: oldSettings.modelPackPath.expandingTildePath).resolvingSymlinksInPath().standardizedFileURL.path
                    guard inRegistered || matchesExisting else {
                        throw QWMetadataValidationError("Selected model pack must reside under ~/.qwg/models")
                    }
                }
                resolvedPackPath = ref.packPath.isEmpty ? targetRefInput.expandingTildePath : ref.packPath
                let targetModelName = ref.modelName.isEmpty ? newSettings.modelName : ref.modelName
                if let profile = QWSettingsManager.shared.loadProfile(forModel: targetModelName) {
                    newSettings.applyModelConfigDefaults(packPath: resolvedPackPath)
                    newSettings.apply(knobProfile: profile)
                } else if newSettings.maxContext == oldSettings.maxContext {
                    newSettings.applyModelConfigDefaults(packPath: resolvedPackPath)
                }
            } else {
                resolvedPackPath = newSettings.resolveModelReference(
                    newSettings.modelPackPath.isEmpty ? newSettings.modelName : newSettings.modelPackPath
                ).packPath
            }
            let backendChanged = newSettings.mtpBackend.lowercased() != oldSettings.mtpBackend.lowercased()
            let packExplicitlyChanged = newSettings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
                != oldSettings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if modelChanged || (backendChanged && !packExplicitlyChanged) {
                let desc = QWModelCatalog.loadPrimaryDescriptor(at: resolvedPackPath)
                newSettings.reconcileStaleDrafterSelection(for: desc)
            }
            if let mounted = actor.getMountedMemoryPackInfo(), mounted.enforcesKVMode {
                newSettings.kvMode = mounted.enforcedKVMode
            }
            try newSettings.validateBeforeLaunch()
            try QWSettingsManager.shared.saveSettings(newSettings)
            QWFanController.shared.setPolicy(QWCoolingPolicy(rawValue: newSettings.coolingPolicy) ?? .auto)
            let updatedSchedule = QWUpdateCheckSchedule.resolve(newSettings.updateCheckSchedule)
            let svc = self.updateService
            Task { await svc.updateSchedule(updatedSchedule) }
            if !modelChanged {
                try await actor.setVisionLoaded(newSettings.loadVision)
            }
            sendJsonResponse(connection, status: 200, object: ["status": "ok", "message": "Settings saved successfully"])
        } catch {
            let err = ServerErrorPayload(message: "Failed to parse or save settings: \(error.localizedDescription)", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        }
    }

    func handleMemoryMountRoute(connection: NWConnection, body: Data) async throws {
        guard !body.isEmpty,
              let payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let repoPath = payload["path"] as? String,
              !repoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = ServerErrorPayload(message: "Missing 'path' in memory mount request.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        let validatedRoot = try actor.resolveValidatedMemoryRootURL(
            at: repoPath,
            overrideRoots: allowedWorkspaceRoots
        )
        let mounted = try await actor.mountMemoryPack(at: validatedRoot.path)
        sendJsonResponse(connection, status: 200, object: mounted)
    }

    func handleHealthRoute(connection: NWConnection) throws {
        let engineState = actor.engineState()
        let modelName = actor.getModelName()
        let health = QWHealthResponse(
            status: engineState.lifecycle.phase.rawValue,
            engine: "qwegine",
            model: modelName,
            loaded: engineState.isLoaded,
            lifecycle: engineState.lifecycle
        )
        let data = try JSONEncoder().encode(health)
        sendRawData(connection, status: 200, contentType: "application/json", data: data)
    }
}
