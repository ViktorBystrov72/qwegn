// TelemetryObservable+TypesAndModels.swift — Connection/engine status types, process stop, model switching, and live settings push

import AppKit
import Darwin
import Foundation
import QWCore
import QWServer

/// Server HTTP reachability state reported independently from engine phase.
public enum ConnectionState: String, Sendable, CaseIterable {
    case unknown
    case disconnected
    case connected
}

/// Process supervisor state for the local `qwg serve` child or external daemon.
public enum EngineState: Equatable, Sendable {
    case stopped
    case starting
    case serverUp(isLoaded: Bool)
    case stopping
    case error(String)

    public var isRunningOrUp: Bool {
        if case .serverUp = self { return true }
        return false
    }

    public var isTransitioning: Bool {
        switch self {
        case .starting, .stopping: return true
        default: return false
        }
    }
}

/// Canonical 6-state menu bar indicator derived from connection, lifecycle, and inference phase.
public enum MenuBarStatusState: Equatable, Sendable {
    case offline
    case standby
    case transitioning
    case ready
    case prefilling
    case decoding
    case failed
}

extension TelemetryObservable {
    public func stopEngineAsync() async {
        isDaemonTransitioning = true
        defer { isDaemonTransitioning = false }
        engineState = .stopping
        isStartingProcess = false
        stopStreamsOnDisconnect()

        if let proc = serverProcess, proc.isRunning {
            QWServerLogger.shared.log(.info, "Terminating child engine process…")
            proc.terminate()
        }
        let pidPath = QWPaths.pidFilePath
        var candidatePids: [Int32] = []
        if let pidVal = engineInfo?.pid, pidVal > 0 { candidatePids.append(Int32(pidVal)) }
        if let fileData = try? String(contentsOfFile: pidPath, encoding: .utf8),
           let filePid = Int32(fileData.trimmingCharacters(in: .whitespacesAndNewlines)), filePid > 0 {
            candidatePids.append(filePid)
        }
        let currentAppPid = ProcessInfo.processInfo.processIdentifier
        let verifiedPids = Set(candidatePids).filter {
            $0 != currentAppPid && kill($0, 0) == 0 && Self.isVerifiedQWGProcess(pid: $0)
        }
        for pid in verifiedPids {
            QWServerLogger.shared.log(.info, "Sending SIGTERM to engine process PID \(pid)…")
            kill(pid, SIGTERM)
        }
        let deadline = Date().addingTimeInterval(5.0)
        while Date() < deadline {
            let isHealthy = await checkServerHealth()
            let anyPidAlive = verifiedPids.contains { kill($0, 0) == 0 && Self.isVerifiedQWGProcess(pid: $0) }
            let childAlive = serverProcess?.isRunning ?? false
            if !isHealthy && !anyPidAlive && !childAlive { break }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        for pid in verifiedPids where kill(pid, 0) == 0 && Self.isVerifiedQWGProcess(pid: pid) {
            QWServerLogger.shared.log(.warn, "Process \(pid) did not exit cleanly; sending SIGKILL…")
            kill(pid, SIGKILL)
        }
        if let proc = serverProcess, proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        serverProcess = nil
        try? FileManager.default.removeItem(atPath: pidPath)
        self.serverAppliedSettings = nil
        self.parentPidWatchdogArmed = false
        self.engineState = .stopped
        self.connection = .disconnected
        self.dashboard = nil
        self.engineInfo = nil
        QWServerLogger.shared.log(.info, "Engine successfully stopped.")
    }

    static func isVerifiedQWGProcess(pid: pid_t) -> Bool {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return false }
        let path = String(decoding: buffer.prefix(Int(length)), as: UTF8.self)
        return URL(fileURLWithPath: path).lastPathComponent == "qwg"
    }

    public func refreshDiscoveredModels() {
        discoveredModels = QWServerSettings.scanModelsDirectory().filter(\.supported)
    }

    public func revealModelsFolder() {
        let url = QWPaths.modelsDirectoryURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    public func switchModel(to model: DiscoveredModel) {
        var updated = settings
        updated.modelName = model.name
        updated.modelPackPath = model.packPath
        let (effective, _) = QWSettingsManager.shared.effectiveSettings(
            forModel: model.name,
            packPath: model.packPath,
            base: updated
        )
        try? saveSettings(effective)
        refreshDiscoveredModels()

        guard isConnected,
              let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/load") else {
            return
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: [
            "model": model.packPath.isEmpty ? model.name : model.packPath
        ])
        req.timeoutInterval = 120.0
        isModelTransitioning = true
        Task {
            defer { self.isModelTransitioning = false }
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Switch model failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                self.fetchHttpEngineInfo()
                QWServerLogger.shared.log(.info, "Switched active model to '\(model.name)' via API.")
            } catch {
                self.fail("Switch model failed: \(error.localizedDescription)")
            }
        }
    }

    func fail(_ message: String) {
        self.lastError = message
        self.lastErrorPhase = enginePhase
        QWServerLogger.shared.log(.warn, message)
    }

    static func responseDetail(_ data: Data) -> String {
        let body = String(data: data, encoding: .utf8) ?? ""
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "no detail in response body" }
        return trimmed.count > 300 ? String(trimmed.prefix(300)) + "…" : trimmed
    }

    func pushLiveSettingsToServer() async {
        do {
            try await pushLiveSettingsThrowing()
        } catch {
            self.fail(error.localizedDescription)
        }
    }

    func pushLiveSettingsThrowing() async throws {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/settings") else { return }
        let body = try JSONEncoder().encode(self.settings)
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        req.timeoutInterval = 30.0
        let (data, res) = try await URLSession.shared.data(for: req)
        guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (res as? HTTPURLResponse)?.statusCode ?? 0
            let detail = Self.responseDetail(data)
            throw NSError(
                domain: "QWSettings",
                code: code,
                userInfo: [NSLocalizedDescriptionKey: "Live settings update failed (HTTP \(code)): \(detail)"]
            )
        }
        self.fetchDashboard()
        self.fetchHttpEngineInfo()
    }
}
