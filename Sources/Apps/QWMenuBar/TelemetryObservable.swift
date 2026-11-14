// TelemetryObservable.swift — Central state store and engine lifecycle coordinator

import AppKit
import Darwin
import Foundation
import Observation
import QWCore
import QWServer

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
    @ObservationIgnored var lastErrorPhase: QWEngineLifecyclePhase?

    public init(startServices: Bool = true, enableAutoLifecycle: Bool = true, initialSettings: QWServerSettings? = nil) {
        let resolvedSettings = initialSettings ?? QWSettingsManager.shared.loadSettings()
        self.settings = resolvedSettings
        self.discoveredModels = (startServices && enableAutoLifecycle)
            ? QWServerSettings.scanModelsDirectory().filter(\.supported) : []
        self.engineVersion = QWVersion.current
        self.engineState = .stopped
        self.enabledMenuBarMetrics = Self.loadEnabledMenuBarMetrics()
        self.hideMenuBarSpeedsWhenIdle = Self.loadHideMenuBarSpeedsWhenIdle()

        guard startServices else { return }
        setupLocalLoggerSubscription()
        if enableAutoLifecycle {
            startPolling()
            applyAutoStartDecision()
        }
    }

    isolated deinit {
        pollTimer?.invalidate()
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
        let snap = QWEngineTelemetryRegistry.shared.snapshot()
        let isLocalActive = snap.cache.lookupsTotal > 0
            || snap.throughput.tokensInTotal > 0
            || snap.throughput.phase != "idle"
        if isLocalActive && !self.engineState.isTransitioning {
            self.engineState = .serverUp(isLoaded: engineInfo?.parameters.isLoaded ?? false)
        }
        fetchHttpEngineInfo()
        let streamFresh = dashboardStreamTask != nil
            && lastDashboardStreamAt.map { Date().timeIntervalSince($0) < 0.85 } == true
        if !streamFresh {
            fetchDashboard()
        }
    }

    /// Fetches the published dashboard view model directly from the server.
    public func fetchDashboard() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/dashboard") else {
            connection = .disconnected
            dashboard = nil
            stopDashboardStream()
            return
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.5
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
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/engine/info") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.0
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
                if let pid = info.pid, pid > 0 {
                    self.lastKnownServerPid = Int32(pid)
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
        if let lockPid = QWEngineInstanceLock.runningEnginePID(), lockPid > 0 {
            lastKnownServerPid = lockPid
            return true
        }
        guard let pid = lastKnownServerPid, pid > 0 else { return false }
        return kill(pid, 0) == 0
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
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/ui") else { return }
        NSWorkspace.shared.open(url)
    }

    public func startEngine() {
        if engineState.isRunningOrUp || engineState.isTransitioning { return }
        engineState = .starting
        isStartingProcess = true
        isDaemonTransitioning = true

        Task { [weak self] in
            guard let self else { return }
            defer { self.isDaemonTransitioning = false }

            if await self.checkServerHealth() {
                self.fetchHttpEngineInfo()
                self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                self.connection = .connected
                self.isStartingProcess = false
                self.parentPidWatchdogArmed = false
                self.serverAppliedSettings = self.settings
                return
            }

            if let existingPid = QWEngineInstanceLock.runningEnginePID(), existingPid > 0 {
                self.lastKnownServerPid = existingPid
                QWServerLogger.shared.log(.info, "Attached to existing qwg daemon (PID \(existingPid)).")
                for _ in 0..<20 {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    if await self.checkServerHealth() {
                        self.fetchHttpEngineInfo()
                        self.engineState = .serverUp(isLoaded: self.engineInfo?.parameters.isLoaded ?? false)
                        self.connection = .connected
                        break
                    }
                    if QWEngineInstanceLock.runningEnginePID() == nil { break }
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
        Task { @MainActor in
            isDaemonTransitioning = true
            defer { isDaemonTransitioning = false }
            self.resolveConfiguredPackFallbackIfMissing()
            do {
                try self.settings.validateBeforeLaunch()
            } catch {
                self.fail("Model / Drafter validation failed: \(error)")
                return
            }
            await self.stopEngineAsync()
            try? await Task.sleep(nanoseconds: 300_000_000)
            self.startEngine()
        }
    }

    public var isAnyServerRunning: Bool {
        engineState.isRunningOrUp || engineState == .starting || serverProcessIsAlive || isConnected
    }

    public func unloadModel() {
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/unload") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 30.0
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
        guard let url = URL(string: "http://\(settings.host):\(settings.port)/v1/models/load") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 120.0
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
