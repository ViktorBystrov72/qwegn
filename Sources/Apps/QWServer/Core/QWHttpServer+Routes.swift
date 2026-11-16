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
        connection.send(content: packet, completion: .contentProcessed({ _ in
            connection.cancel()
        }))
    }

    func sendSimpleResponse(_ connection: NWConnection, status: Int, body: String) {
        sendRawData(connection, status: status, contentType: "text/plain", data: Data(body.utf8))
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
        let info = try await actor.inspectMemoryPack(at: repoPath)
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
        guard req.stream ?? true else {
            let result = try await actor.indexMemoryPack(request: req)
            sendJsonResponse(connection, status: 200, object: result)
            return
        }

        sendSSEHeaders(connection)
        let keepalive = startSSEKeepalive(connection: connection)
        defer {
            keepalive.gate.stop()
            keepalive.task.cancel()
        }
        let lock = NSLock()
        let sendProgress: @Sendable (QWMemoryIndexProgressEvent) -> Void = { [weak self] event in
            guard let self, !self.isPeerGone(connection) else { return }
            let encoder = JSONEncoder()
            guard let jsonData = try? encoder.encode(event),
                  let jsonStr = String(data: jsonData, encoding: .utf8) else {
                return
            }
            let frame = Data("data: \(jsonStr)\n\n".utf8)
            lock.withLock {
                connection.send(content: frame, isComplete: false, completion: .contentProcessed({ err in
                    if err != nil { connection.cancel() }
                }))
            }
        }

        do {
            _ = try await actor.indexMemoryPack(request: req, onProgress: sendProgress)
        } catch {
            sendProgress(QWMemoryIndexProgressEvent(
                phase: "error",
                progressFraction: 0,
                message: error.localizedDescription,
                error: "\(error)"
            ))
        }
        lock.withLock {
            connection.send(content: Data("data: [DONE]\n\n".utf8), isComplete: true, completion: .contentProcessed({ _ in
                connection.cancel()
            }))
        }
    }
}
