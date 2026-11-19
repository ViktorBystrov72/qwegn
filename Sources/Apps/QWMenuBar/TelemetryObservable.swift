// TelemetryObservable.swift — Central state store and engine lifecycle coordinator

import AppKit
import Darwin
import Foundation
import Observation
import QWCore
import QWQuant
import QWServer

/// Cached LSA sidecar detection state populated off `@MainActor`.
public struct QWCachedLSASidecarStatus: Sendable, Equatable {
    public let hasSidecar: Bool
    public let statusText: String

    public init(hasSidecar: Bool, statusText: String) {
        self.hasSidecar = hasSidecar
        self.statusText = statusText
    }
}

@MainActor
@Observable
public final class TelemetryObservable {
    public var engineState: EngineState = .stopped
    public var engineInfo: QWEngineInfoResponse?
    public var dashboard: QWDashboardViewModel?
    public var connection: ConnectionState = .unknown
    public var lastError: String?
    public var engineVersion: String = QWVersion.current
    public var isStartingProcess: Bool = false
    public var isDaemonTransitioning: Bool = false
    public var isModelTransitioning: Bool = false
    public var modelLoadedOnStart: Bool = false
    public var recentLogs: [String] = []
    public var logEntries: [QWLogEntry] = []
    public var isLogStreamConnected: Bool = false
    public var settings: QWServerSettings
    public var discoveredModels: [DiscoveredModel] = []
    public var discoveredDrafterModels: [DiscoveredModel] = []
    public var cachedPrimaryDescriptors: [String: QWModelDescriptor] = [:]
    public var cachedLSASidecarStatusByKey: [String: QWCachedLSASidecarStatus] = [:]
    public var cachedDrafterMaxDrafts: [String: Int] = [:]
    public var cachedContextEstimates: [String: QWContextMemoryEstimate] = [:]
    public var parentPidWatchdogArmed: Bool = false
    public var serverAppliedSettings: QWServerSettings?
    public var enabledMenuBarMetrics: Set<QWMenuBarMetric> {
        didSet { persistMenuBarMetrics() }
    }
    public var hideMenuBarSpeedsWhenIdle: Bool {
        didSet { persistHideMenuBarSpeedsWhenIdle() }
    }

    public var serverPortString: String { String(settings.port) }
    public var isConnected: Bool { connection == .connected }
    public var enginePhase: QWEngineLifecyclePhase? { dashboard?.lifecycle.phase }
    public var isModelLoaded: Bool { dashboard?.lifecycle.phase == .loaded }
    public var contextBudget: Int? { dashboard?.knobs["contextTokens"]?.intValue }

    public var displayedError: String? {
        if let detail = dashboard?.lifecycle.detail { return detail }
        return lastError
    }

    public var serverStatusText: String {
        switch connection {
        case .unknown: return "Not contacted yet"
        case .disconnected: return "No response"
        case .connected:
            switch enginePhase {
            case .some(.loaded): return "Loaded"
            case .some(.loading): return "Loading"
            case .some(.unloading): return "Unloading"
            case .some(.failed): return "Failed"
            case .some(.unloaded): return "Unloaded"
            case .none: return "Unknown"
            }
        }
    }

    public var menuBarStatusState: MenuBarStatusState {
        if engineState.isTransitioning || isDaemonTransitioning || isModelTransitioning {
            return .transitioning
        }
        if case .error = engineState, !isConnected { return .failed }
        guard isConnected else { return .offline }
        switch enginePhase {
        case .some(.loading), .some(.unloading): return .transitioning
        case .some(.failed): return .failed
        case .some(.unloaded), .none: return .standby
        case .some(.loaded):
            switch dashboard?.generation {
            case .some(.prefill): return .prefilling
            case .some(.decode): return .decoding
            default: return .ready
            }
        }
    }

    public var daemonActionTitle: String {
        isAnyServerRunning ? "Stop Server" : "Start Server"
    }

    public var daemonActionIcon: String {
        isAnyServerRunning ? "power" : "play.fill"
    }

    @ObservationIgnored var pollTimer: Timer?
    @ObservationIgnored var loggerSubscriptionId: UUID?
    @ObservationIgnored var serverProcess: Process?
    @ObservationIgnored var lastKnownServerPid: Int32?
    @ObservationIgnored var nextLogSequence: UInt64 = 1
    @ObservationIgnored var isPopoverVisible: Bool = false
    @ObservationIgnored var isLogWindowVisible: Bool = false
    @ObservationIgnored var dashboardStreamTask: Task<Void, Never>?
    @ObservationIgnored var logStreamTask: Task<Void, Never>?
    @ObservationIgnored var lastDashboardStreamAt: Date?
    @ObservationIgnored
    private var dashboardGeneration: UInt64 = 0
    @ObservationIgnored
    private var engineInfoGeneration: UInt64 = 0
    @ObservationIgnored var startTask: Task<Void, Never>?
    @ObservationIgnored var restartTask: Task<Void, Never>?
    @ObservationIgnored private(set) var isRestartingProcess: Bool = false
    @ObservationIgnored var lastErrorPhase: QWEngineLifecyclePhase?
    @ObservationIgnored private var modelDiscoveryTask: Task<Void, Never>?
    @ObservationIgnored private var scheduledDescriptorPaths: Set<String> = []
    @ObservationIgnored private var scheduledSidecarKeys: Set<String> = []
    @ObservationIgnored private var scheduledDrafterKeys: Set<String> = []
    @ObservationIgnored private var scheduledEstimateKeys: Set<String> = []
    @ObservationIgnored private var metadataTasks: [String: Task<Void, Never>] = [:]

    public init(startServices: Bool = true, enableAutoLifecycle: Bool = true, initialSettings: QWServerSettings? = nil) {
        let resolvedSettings = initialSettings ?? QWSettingsManager.shared.loadSettings()
        self.settings = resolvedSettings
        self.discoveredModels = []
        self.engineVersion = QWVersion.current
        self.engineState = .stopped
        self.enabledMenuBarMetrics = Self.loadEnabledMenuBarMetrics()
        self.hideMenuBarSpeedsWhenIdle = Self.loadHideMenuBarSpeedsWhenIdle()

        guard startServices else { return }
        setupLocalLoggerSubscription()
        if enableAutoLifecycle {
            refreshDiscoveredModelsAsync()
            startPolling()
            applyAutoStartDecision()
        }
    }

    isolated deinit {
        pollTimer?.invalidate()
        modelDiscoveryTask?.cancel()
        for task in metadataTasks.values {
            task.cancel()
        }
        startTask?.cancel()
        restartTask?.cancel()
        dashboardStreamTask?.cancel()
        logStreamTask?.cancel()
        if let id = loggerSubscriptionId {
            QWServerLogger.shared.unsubscribe(id)
        }
        if let proc = serverProcess, proc.isRunning {
            proc.terminate()
        }
    }

    private func setupLocalLoggerSubscription() {
        for line in QWServerLogger.shared.getRecentLogs() {
            appendLogLine(line)
        }
        self.loggerSubscriptionId = QWServerLogger.shared.subscribe { [weak self] line in
            Task { @MainActor [weak self] in
                self?.appendLogLine(line)
            }
        }
    }

    public func startPolling() {
        fetchDirectOrHttp()
        schedulePollTimer()
    }

    func schedulePollTimer() {
        pollTimer?.invalidate()

        let baseInterval = Double(max(200, settings.telemetryPollIntervalMs)) / 1000.0
        let activeGen = dashboard?.generation == .prefill || dashboard?.generation == .decode
        let needsFastPoll = isPopoverVisible || isLogWindowVisible || activeGen
            || engineState.isTransitioning || enginePhase == .loading || enginePhase == .unloading
        let interval = needsFastPoll ? min(1.0, baseInterval) : max(2.0, baseInterval * 4.0)

        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.fetchDirectOrHttp()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        pollTimer = timer
    }

    public func fetchDirectOrHttp() {
        fetchHttpEngineInfo()
        let streamFresh = dashboardStreamTask != nil
            && lastDashboardStreamAt.map { Date().timeIntervalSince($0) < 0.85 } == true
        if !streamFresh {
            fetchDashboard()
        }
    }

    /// Scans model and drafter directories off `@MainActor` and populates descriptor caches.
    public func refreshDiscoveredModelsAsync(
        completion: (@MainActor ([DiscoveredModel], [DiscoveredModel]) -> Void)? = nil
    ) {
        modelDiscoveryTask?.cancel()
        modelDiscoveryTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let scanned = await Task.detached(priority: .utility) {
                let models = QWServerSettings.scanModelsDirectory()
                let drafters = QWServerSettings.scanDrafterModelsDirectory()
                var descriptors: [String: QWModelDescriptor] = [:]
                for model in models where model.supported && !model.packPath.isEmpty {
                    if let desc = QWServerSettings.loadPrimaryDescriptor(packPath: model.packPath) {
                        descriptors[model.packPath] = desc
                    }
                }
                return (models, drafters, descriptors)
            }.value
            guard !Task.isCancelled else { return }
            self.discoveredModels = scanned.0.filter(\.supported)
            self.discoveredDrafterModels = scanned.1
            for (path, desc) in scanned.2 {
                self.cachedPrimaryDescriptors[path] = desc
                self.scheduledDescriptorPaths.insert(path)
            }
            completion?(scanned.0, scanned.1)
        }
    }

    /// Returns a cached primary model descriptor and schedules background loading if absent.
    public func cachedPrimaryDescriptor(packPath: String) -> QWModelDescriptor? {
        let key = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return nil }
        if let hit = cachedPrimaryDescriptors[key] { return hit }
        if scheduledDescriptorPaths.insert(key).inserted {
            let taskKey = "desc|\(key)"
            metadataTasks[taskKey]?.cancel()
            metadataTasks[taskKey] = Task { @MainActor [weak self] in
                let desc = await Task.detached(priority: .utility) {
                    QWServerSettings.loadPrimaryDescriptor(packPath: key)
                }.value
                guard let self, !Task.isCancelled else { return }
                if let desc { self.cachedPrimaryDescriptors[key] = desc }
                self.metadataTasks.removeValue(forKey: taskKey)
            }
        }
        return nil
    }

    /// Returns cached `.qlsa` sidecar availability and status text, loading off `@MainActor` on miss.
    public func cachedLSASidecarStatus(packPath: String, explicitPath: String) -> QWCachedLSASidecarStatus? {
        let normPack = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let normExplicit = explicitPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normPack.isEmpty || !normExplicit.isEmpty else { return nil }
        let key = "\(normPack)|\(normExplicit)"
        if let hit = cachedLSASidecarStatusByKey[key] { return hit }
        if scheduledSidecarKeys.insert(key).inserted {
            let taskKey = "lsa|\(key)"
            metadataTasks[taskKey]?.cancel()
            metadataTasks[taskKey] = Task { @MainActor [weak self] in
                let status: QWCachedLSASidecarStatus = await Task.detached(priority: .utility) {
                    let explicitArg = normExplicit.isEmpty ? nil : normExplicit
                    guard let url = QWLSAConfig.resolveSidecarURL(
                        packPath: normPack, explicitSidecarPath: explicitArg
                    ) else {
                        return QWCachedLSASidecarStatus(
                            hasSidecar: false,
                            statusText: "No .qlsa sidecar detected (Auto uses in-memory Weight-SVD)"
                        )
                    }
                    if let desc = QWServerSettings.loadPrimaryDescriptor(packPath: normPack),
                       let info = try? QWLSASidecar.peekInfo(url: url, descriptor: desc) {
                        let matchNote = info.geometryMatches ? "verified" : "digest mismatch"
                        return QWCachedLSASidecarStatus(
                            hasSidecar: true,
                            statusText: "Detected \(url.lastPathComponent) [\(info.method), d=\(info.indexDim), \(matchNote)]"
                        )
                    }
                    return QWCachedLSASidecarStatus(hasSidecar: true, statusText: "Detected \(url.lastPathComponent)")
                }.value
                guard let self, !Task.isCancelled else { return }
                self.cachedLSASidecarStatusByKey[key] = status
                self.metadataTasks.removeValue(forKey: taskKey)
            }
        }
        return nil
    }

    /// Returns cached drafter `maxDrafts`, inferring from disk off `@MainActor` on cache miss.
    public func cachedMaxDrafts(forDrafterPackPath path: String) -> Int? {
        let key = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return nil }
        if let hit = cachedDrafterMaxDrafts[key] { return hit }
        if scheduledDrafterKeys.insert(key).inserted {
            let taskKey = "draft|\(key)"
            metadataTasks[taskKey]?.cancel()
            metadataTasks[taskKey] = Task { @MainActor [weak self] in
                let maxD = await Task.detached(priority: .utility) {
                    QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: key).maxDrafts
                }.value
                guard let self, !Task.isCancelled else { return }
                self.cachedDrafterMaxDrafts[key] = maxD
                self.metadataTasks.removeValue(forKey: taskKey)
            }
        }
        return nil
    }

    /// Returns cached context memory estimate, computing off `@MainActor` on cache miss.
    public func cachedContextEstimate(
        for settings: QWServerSettings,
        descriptor: QWModelDescriptor?
    ) -> QWContextMemoryEstimate? {
        let rawPath = settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawPath.isEmpty || !settings.modelName.isEmpty else { return nil }
        let key = [
            settings.modelName, rawPath, settings.kvMode.lowercased(),
            "\(settings.maxContext)", "\(settings.speculativeDecodingEnabled)",
            "\(settings.mtpDrafts)", settings.mtpBackend.lowercased(),
            "\(settings.prefillChunkSize)", descriptor?.geometryDigestV3Hex ?? "none"
        ].joined(separator: "|")
        if let hit = cachedContextEstimates[key] { return hit }
        if scheduledEstimateKeys.insert(key).inserted {
            let taskKey = "est|\(key)"
            metadataTasks[taskKey]?.cancel()
            metadataTasks[taskKey] = Task { @MainActor [weak self] in
                let est = await Task.detached(priority: .utility) {
                    settings.estimateContextMemory(for: descriptor)
                }.value
                guard let self, !Task.isCancelled else { return }
                if let est { self.cachedContextEstimates[key] = est }
                self.metadataTasks.removeValue(forKey: taskKey)
            }
        }
        return nil
    }

    public func invalidateLSASidecarCache() {
        cachedLSASidecarStatusByKey.removeAll()
        scheduledSidecarKeys.removeAll()
    }

    public func invalidateModelMetadataCache() {
        cachedPrimaryDescriptors.removeAll()
        cachedLSASidecarStatusByKey.removeAll()
        cachedDrafterMaxDrafts.removeAll()
        cachedContextEstimates.removeAll()
        scheduledDescriptorPaths.removeAll()
        scheduledSidecarKeys.removeAll()
        scheduledDrafterKeys.removeAll()
        scheduledEstimateKeys.removeAll()
    }

    /// Fetches the published dashboard view model directly from the server.
    public func fetchDashboard() {
        guard let req = authorizedRequest(path: "/v1/dashboard", timeoutInterval: 1.5) else {
            connection = .disconnected
            dashboard = nil
            stopDashboardStream()
            return
        }
        dashboardGeneration &+= 1
        let generation = dashboardGeneration
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    guard generation == dashboardGeneration else { return }
                    self.dashboard = nil
                    self.connection = .disconnected
                    self.stopStreamsOnDisconnect()
                    return
                }
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let model = try decoder.decode(QWDashboardViewModel.self, from: data)
                guard generation == dashboardGeneration else { return }
                self.applyDashboardModel(model, fromStream: false)
            } catch {
                guard generation == dashboardGeneration else { return }
                self.dashboard = nil
                self.connection = .disconnected
                self.stopStreamsOnDisconnect()
            }
        }
    }

    func applyDashboardModel(_ model: QWDashboardViewModel, fromStream: Bool = false) {
        if !fromStream,
           self.connection == .connected,
           let currentAt = self.dashboard?.sampledAt,
           model.sampledAt < currentAt {
            return
        }
        let wasActive = self.dashboard?.generation == .prefill || self.dashboard?.generation == .decode
        if fromStream {
            self.lastDashboardStreamAt = Date()
        }
        self.dashboard = model
        self.connection = .connected
        if let failed = self.lastErrorPhase, failed != model.lifecycle.phase {
            self.lastError = nil
            self.lastErrorPhase = nil
        }
        syncStreamsToVisibility()
        let isActive = model.generation == .prefill || model.generation == .decode
        if wasActive != isActive {
            schedulePollTimer()
        }
    }

    public func fetchHttpEngineInfo() {
        guard let req = authorizedRequest(path: "/v1/engine/info", timeoutInterval: 1.0) else { return }
        engineInfoGeneration &+= 1
        let generation = engineInfoGeneration
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    guard generation == engineInfoGeneration else { return }
                    self.applyUnreachableState()
                    return
                }
                let info = try JSONDecoder().decode(QWEngineInfoResponse.self, from: data)
                guard generation == engineInfoGeneration else { return }
                self.engineInfo = info
                self.engineVersion = info.version
                if let pid = info.pid, let p32 = Int32(exactly: pid), p32 > 0 {
                    self.lastKnownServerPid = p32
                }
                if !self.engineState.isTransitioning {
                    self.engineState = .serverUp(isLoaded: info.parameters.isLoaded)
                }
            } catch {
                guard generation == engineInfoGeneration else { return }
                self.applyUnreachableState()
                self.engineInfo = nil
            }
        }
    }

    // MARK: - Server Process Management & Controls

    private var serverProcessIsAlive: Bool {
        if let proc = serverProcess, proc.isRunning { return true }
        if let lockPid = QWEngineInstanceLock.runningEnginePID(), lockPid > 0,
           Self.isVerifiedQWGProcess(pid: lockPid) {
            lastKnownServerPid = lockPid
            return true
        }
        guard let pid = lastKnownServerPid, pid > 0,
              kill(pid, 0) == 0,
              Self.isVerifiedQWGProcess(pid: pid) else {
            return false
        }
        return true
    }

    private func applyUnreachableState() {
        guard !self.engineState.isTransitioning else { return }
        let liveness = QWServerLifecycleRules.liveness(
            answered: false,
            processAlive: self.serverProcessIsAlive
        )
        self.engineState = liveness == .gone ? .stopped : .error("Server not responding")
    }

    public func pendingServerRestartKeys(for candidate: QWServerSettings) -> [String] {
        guard let applied = serverAppliedSettings else { return [] }
        return QWSettingsCoherence.pendingServerRestartKeys(from: applied, to: candidate)
    }

    private func applyAutoStartDecision() {
        let decision = QWSettingsCoherence.startDecision(
            autoStartServer: settings.autoStartServer,
            serverRunning: isAnyServerRunning
        )
        if decision == .start {
            startEngine()
        }
    }

    public func toggleDaemon() {
        if isAnyServerRunning {
            stopEngine()
        } else {
            startEngine()
        }
    }

    public func openWebDashboard() {
        guard let url = serverURL(path: "/ui") else { return }
        NSWorkspace.shared.open(url)
    }

    func resolveUnhealthyAttach(pid: Int32, processAlive: Bool) {
        connection = .disconnected
        isStartingProcess = false
        if processAlive {
            let msg = "Attached qwg daemon (PID \(pid)) failed health check"
            engineState = .error(msg)
            fail(msg)
        } else {
            if lastKnownServerPid == pid {
                lastKnownServerPid = nil
            }
            engineState = .stopped
        }
    }

    private func awaitAttachedDaemonHealth(pid: Int32) async -> Bool {
        for _ in 0..<20 {
            if Task.isCancelled { return false }
            if await self.checkServerHealth() {
                self.fetchHttpEngineInfo()
                self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                self.connection = .connected
                return true
            }
            if QWEngineInstanceLock.runningEnginePID() == nil || !Self.isVerifiedQWGProcess(pid: pid) {
                return false
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false
    }

    public func startEngine() {
        if engineState.isRunningOrUp || engineState.isTransitioning || startTask != nil { return }
        engineState = .starting
        isStartingProcess = true
        isDaemonTransitioning = true

        startTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                self.startTask = nil
                self.isDaemonTransitioning = false
            }

            if await self.checkServerHealth() {
                self.fetchHttpEngineInfo()
                self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                self.connection = .connected
                self.isStartingProcess = false
                self.parentPidWatchdogArmed = false
                self.serverAppliedSettings = self.settings
                return
            }

            if let existingPid = QWEngineInstanceLock.runningEnginePID(), existingPid > 0,
               Self.isVerifiedQWGProcess(pid: existingPid) {
                self.lastKnownServerPid = existingPid
                QWServerLogger.shared.log(.info, "Attached to existing qwg daemon (PID \(existingPid)).")
                let healthy = await self.awaitAttachedDaemonHealth(pid: existingPid)
                if !healthy {
                    let alive = QWEngineInstanceLock.runningEnginePID() != nil
                        && Self.isVerifiedQWGProcess(pid: existingPid)
                    self.resolveUnhealthyAttach(pid: existingPid, processAlive: alive)
                }
                self.isStartingProcess = false
                self.parentPidWatchdogArmed = false
                self.serverAppliedSettings = self.settings
                return
            }

            guard let binaryPath = self.findQWGBinary() else {
                QWServerLogger.shared.log(.error, "Cannot start engine: 'qwg' binary not found.")
                self.engineState = .error("Binary not found")
                self.isStartingProcess = false
                return
            }

            self.resolveConfiguredPackFallbackIfMissing()
            do {
                try self.settings.validateBeforeLaunch()
            } catch {
                let msg = "\(error)"
                QWServerLogger.shared.log(.error, "Pre-launch validation failed: \(msg)")
                self.engineState = .error(msg)
                self.fail(msg)
                self.isStartingProcess = false
                return
            }

            self.modelLoadedOnStart = !self.settings.lazyLoad
            try? QWSettingsManager.shared.saveSettings(self.settings)
            let configPath = QWSettingsManager.defaultSettingsFileURL.path
            QWServerLogger.shared.log(.info, "Starting engine via \(binaryPath) --config \(configPath)…")

            let watchdogArguments = QWServerLifecycleRules.parentPidArguments(
                keepServerRunningAfterQuit: self.settings.keepServerRunningAfterQuit,
                parentPid: ProcessInfo.processInfo.processIdentifier
            )
            self.parentPidWatchdogArmed = !watchdogArguments.isEmpty
            self.serverAppliedSettings = self.settings

            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: binaryPath)
            proc.arguments = ["serve", "--config", configPath] + watchdogArguments
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            let buffer = ChildPipeLineBuffer()
            attachChildPipe(pipe, buffer: buffer)

            proc.terminationHandler = { [weak self] p in
                pipe.fileHandleForReading.readabilityHandler = nil
                buffer.flush { line in
                    QWServerLogger.shared.logChildOutputLine(line)
                }
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.serverProcess = nil
                    self.isStartingProcess = false
                    self.isDaemonTransitioning = false
                    self.modelLoadedOnStart = false
                    if self.lastKnownServerPid == p.processIdentifier {
                        self.lastKnownServerPid = nil
                    }
                    if self.engineState != .stopping && p.terminationStatus != 0 {
                        self.engineState = .error("Exited with code \(p.terminationStatus)")
                    } else {
                        self.engineState = .stopped
                    }
                    self.connection = .disconnected
                    self.stopStreamsOnDisconnect()
                    QWServerLogger.shared.log(.warn, "Engine process exited with code \(p.terminationStatus)")
                }
            }

            do {
                try proc.run()
                self.serverProcess = proc
                var running = false
                for _ in 0..<30 {
                    if Task.isCancelled { break }
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if await self.checkServerHealth() { running = true; break }
                    if let p = self.serverProcess, !p.isRunning { break }
                }
                if running {
                    self.fetchHttpEngineInfo()
                    self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                    self.connection = .connected
                    self.isStartingProcess = false
                    if !self.settings.lazyLoad { self.modelLoadedOnStart = true }
                    self.syncStreamsToVisibility()
                    QWServerLogger.shared.log(.info, "Engine running at \(self.settings.host):\(self.settings.port)")
                } else {
                    if self.engineState == .starting {
                        self.engineState = .error("Health check timed out")
                    }
                    self.isStartingProcess = false
                }
            } catch {
                pipe.fileHandleForReading.readabilityHandler = nil
                proc.terminationHandler = nil
                try? pipe.fileHandleForWriting.close()
                try? pipe.fileHandleForReading.close()
                self.engineState = .error(error.localizedDescription)
                self.isStartingProcess = false
                QWServerLogger.shared.log(.error, "Failed to launch qwg: \(error.localizedDescription)")
            }
        }
    }

    public func stopEngine() {
        modelLoadedOnStart = false
        Task { @MainActor in await self.stopEngineAsync() }
    }

    /// Restarts the `qwg serve` daemon process so startup settings take effect.
    public func restartServerProcess() {
        guard !isRestartingProcess, !isDaemonTransitioning, restartTask == nil else { return }
        isRestartingProcess = true
        isDaemonTransitioning = true
        restartTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                self.isRestartingProcess = false
                self.isDaemonTransitioning = false
                self.restartTask = nil
            }
            self.resolveConfiguredPackFallbackIfMissing()
            do {
                try self.settings.validateBeforeLaunch()
            } catch {
                self.fail("Model / Drafter validation failed: \(error)")
                return
            }
            await self.stopEngineAsync()
            self.isDaemonTransitioning = true
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            self.startEngine()
            await self.startTask?.value
        }
    }

    public var isAnyServerRunning: Bool {
        engineState.isRunningOrUp || engineState == .starting || serverProcessIsAlive || isConnected
    }

    public func unloadModel() {
        guard let req = authorizedRequest(path: "/v1/models/unload", method: "POST", timeoutInterval: 30.0) else { return }
        isModelTransitioning = true
        Task {
            defer { self.isModelTransitioning = false }
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Unload failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Model unloaded via API. Server remains active.")
            } catch {
                self.fail("Unload failed: \(error.localizedDescription)")
            }
        }
    }

    public func reloadModel() {
        guard let req = authorizedRequest(path: "/v1/models/load", method: "POST", timeoutInterval: 120.0) else { return }
        isModelTransitioning = true
        Task {
            defer { self.isModelTransitioning = false }
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Load failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.lastError = nil
                self.lastErrorPhase = nil
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Model loaded via API.")
            } catch {
                self.fail("Load failed: \(error.localizedDescription)")
            }
        }
    }

    /// Persists settings to disk and pushes live keys to the running daemon.
    public func saveSettings(_ newSettings: QWServerSettings) throws {
        try QWSettingsManager.shared.saveSettings(newSettings)
        self.settings = QWSettingsManager.shared.loadSettings()
        self.serverAppliedSettings?.loadVision = self.settings.loadVision
        QWServerLogger.shared.configureFileSink(path: self.settings.expandedLogFilePath)
        if isConnected {
            Task { @MainActor [weak self] in
                await self?.pushLiveSettingsToServer()
            }
        }
        startPolling()
        applyAutoStartDecision()
    }

    /// Saves settings and awaits live server validation before returning.
    public func saveSettingsAndValidateLive(_ newSettings: QWServerSettings) async throws {
        try QWSettingsManager.shared.saveSettings(newSettings)
        self.settings = QWSettingsManager.shared.loadSettings()
        self.serverAppliedSettings?.loadVision = self.settings.loadVision
        QWServerLogger.shared.configureFileSink(path: self.settings.expandedLogFilePath)
        if isConnected {
            try await pushLiveSettingsThrowing()
        }
        startPolling()
        applyAutoStartDecision()
    }
}
