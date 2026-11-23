// QWSMCClient.swift — AppleSMC sensor reader and fan override controller.
// Reads host GPU/CPU/memory temperatures, fan speeds, and system power directly
// from IOKit AppleSMC; routes fan writes through qwg-smc-helper or root IOKit.

import Darwin
import Foundation
import IOKit
import QWSMCHelper

public typealias QWSMCClient = QWSMCHelper.QWSMCClient

/// Cooling policy controlling host AppleSMC fan behavior during inference.
public enum QWCoolingPolicy: String, Codable, Sendable, CaseIterable {
    case auto = "auto"
    case proactive = "proactive"
    case max = "max"
}

/// Host thermal, fan, and power telemetry read from AppleSMC and ProcessInfo.
public struct QWSMCSnapshot: Sendable, Equatable {
    public let gpuTemperatureCelsius: Double?
    public let gpuTemperatureUnavailableReason: String?
    public let cpuTemperatureCelsius: Double?
    public let memoryTemperatureCelsius: Double?
    public let fanSpeedRPM: Double?
    public let fanSpeedUnavailableReason: String?
    public let fanMinRPM: Double?
    public let fanMaxRPM: Double?
    public let fanControlStatus: String
    public let systemPowerWatts: Double?
    public let systemPowerUnavailableReason: String?
    public let thermalStateLevel: String

    public init(
        gpuTemperatureCelsius: Double? = nil,
        gpuTemperatureUnavailableReason: String? = nil,
        cpuTemperatureCelsius: Double? = nil,
        memoryTemperatureCelsius: Double? = nil,
        fanSpeedRPM: Double? = nil,
        fanSpeedUnavailableReason: String? = nil,
        fanMinRPM: Double? = nil,
        fanMaxRPM: Double? = nil,
        fanControlStatus: String = "System Auto",
        systemPowerWatts: Double? = nil,
        systemPowerUnavailableReason: String? = nil,
        thermalStateLevel: String = "Nominal"
    ) {
        self.gpuTemperatureCelsius = gpuTemperatureCelsius
        self.gpuTemperatureUnavailableReason = gpuTemperatureUnavailableReason
        self.cpuTemperatureCelsius = cpuTemperatureCelsius
        self.memoryTemperatureCelsius = memoryTemperatureCelsius
        self.fanSpeedRPM = fanSpeedRPM
        self.fanSpeedUnavailableReason = fanSpeedUnavailableReason
        self.fanMinRPM = fanMinRPM
        self.fanMaxRPM = fanMaxRPM
        self.fanControlStatus = fanControlStatus
        self.systemPowerWatts = systemPowerWatts
        self.systemPowerUnavailableReason = systemPowerUnavailableReason
        self.thermalStateLevel = thermalStateLevel
    }
}

/// Manages AppleSMC thermal/fan sampling and cooling policy enforcement.
public final class QWFanController: @unchecked Sendable {
    public static let shared = QWFanController()
    public static let helperSocketPath = QWSMCHelperConstants.socketPath

    private static let gpuTempKeys = ["Tg08", "Tg0C", "Tg0O", "Tg0R", "Tg0U", "Tg0X", "Tg04", "Tg0D", "Tg0P"]
    private static let cpuTempKeys = ["Tp00", "Tp04", "Tp08", "Tp0C", "Tp0G", "Tp0K"]
    private static let memTempKeys = ["Tm00", "Tm04", "Tm08", "Tm0C"]
    private static let proactiveCooldownSeconds: TimeInterval = 15.0
    private static let helperPingCacheTTL: TimeInterval = 2.0
    private static let unauthorizedStatus = "System Auto (SMC write requires root — enable Fan Helper)"

    private let client = QWSMCClient()
    private let socketPath: String
    private let lock = NSLock()
    private let ioLock = NSLock()
    private let reconcileQueue = DispatchQueue(label: "com.qwegn.smc.reconcile", qos: .utility)
    private var policy: QWCoolingPolicy = .auto
    private var inferencePhase: String = "idle"
    private var lastActiveInferenceAt: Date?
    private var didApplyManualOverride = false
    private var lastWriteAuthorized: Bool?
    private var lastPingAt: Date?
    private var lastPingResult: Bool = false
    private var pendingReconcile = false

    public init(
        socketPath: String = QWSMCHelperConstants.socketPath,
        registerAtExit: Bool = true
    ) {
        self.socketPath = socketPath
        if registerAtExit && socketPath == QWSMCHelperConstants.socketPath {
            atexit { QWFanController.shared.restoreAutoOnExit() }
        }
    }

    public var currentPolicy: QWCoolingPolicy {
        lock.withLock { policy }
    }

    public func setPolicy(_ newPolicy: QWCoolingPolicy) {
        lock.withLock {
            policy = newPolicy
        }
        scheduleReconcileAsync()
    }

    public func updateInferencePhase(_ phase: String) {
        let now = Date()
        let changed = lock.withLock { () -> Bool in
            let prev = inferencePhase
            inferencePhase = phase
            if phase == "prefill" || phase == "decode" {
                lastActiveInferenceAt = now
            }
            return prev != phase
        }
        if changed {
            scheduleReconcileAsync()
        }
    }

    public func scheduleReconcileAsync() {
        let shouldDispatch = lock.withLock { () -> Bool in
            if pendingReconcile { return false }
            pendingReconcile = true
            return true
        }
        guard shouldDispatch else { return }
        reconcileQueue.async { [weak self] in
            guard let self else { return }
            self.lock.withLock { self.pendingReconcile = false }
            _ = self.sampleAndReconcile()
        }
    }

    public func sampleAndReconcile(now: Date = Date()) -> QWSMCSnapshot {
        ioLock.lock()
        defer { ioLock.unlock() }
        let gpuTemp = Self.maxSensor(keys: Self.gpuTempKeys, client: client)
        let cpuTemp = Self.maxSensor(keys: Self.cpuTempKeys, client: client)
        let memTemp = Self.maxSensor(keys: Self.memTempKeys, client: client)
        let thermalState = Self.currentThermalState()

        let fanCount = client.fanCount()
        var fanActuals: [Double] = []
        var fanMins: [Double] = []
        var fanMaxs: [Double] = []
        if fanCount > 0 {
            for idx in 0..<min(fanCount, 4) {
                if let ac = client.readValue("F\(idx)Ac"), ac >= 0 { fanActuals.append(ac) }
                if let mn = client.readValue("F\(idx)Mn"), mn > 0 { fanMins.append(mn) }
                if let mx = client.readValue("F\(idx)Mx"), mx > 0 { fanMaxs.append(mx) }
            }
        }

        let avgActual = fanActuals.isEmpty ? nil : (fanActuals.reduce(0, +) / Double(fanActuals.count))
        let avgMin = fanMins.isEmpty ? nil : (fanMins.reduce(0, +) / Double(fanMins.count))
        let avgMax = fanMaxs.isEmpty ? nil : (fanMaxs.reduce(0, +) / Double(fanMaxs.count))
        let power = client.readValue("PSTR").flatMap { ($0 >= 0.5 && $0 <= 500.0) ? $0 : nil }

        let controlStatus = reconcileFans(
            fanCount: fanCount,
            actualRPMs: fanActuals,
            minRPMs: fanMins,
            maxRPMs: fanMaxs,
            gpuTemp: gpuTemp,
            thermalState: thermalState,
            now: now
        )

        return QWSMCSnapshot(
            gpuTemperatureCelsius: gpuTemp,
            gpuTemperatureUnavailableReason: gpuTemp == nil ? "this host does not publish AppleSMC GPU thermal sensors" : nil,
            cpuTemperatureCelsius: cpuTemp,
            memoryTemperatureCelsius: memTemp,
            fanSpeedRPM: avgActual,
            fanSpeedUnavailableReason: avgActual == nil ? "this host has no active cooling fans or AppleSMC fan telemetry" : nil,
            fanMinRPM: avgMin,
            fanMaxRPM: avgMax,
            fanControlStatus: controlStatus,
            systemPowerWatts: power,
            systemPowerUnavailableReason: power == nil ? "this host does not publish AppleSMC system power (PSTR)" : nil,
            thermalStateLevel: thermalState
        )
    }

    private func reconcileFans(
        fanCount: Int,
        actualRPMs: [Double],
        minRPMs: [Double],
        maxRPMs: [Double],
        gpuTemp: Double?,
        thermalState: String,
        now: Date
    ) -> String {
        guard fanCount > 0 else { return "Unavailable" }
        let snapshotState = lock.withLock {
            (policy, inferencePhase, lastActiveInferenceAt, didApplyManualOverride)
        }
        let activePolicy = snapshotState.0
        let phase = snapshotState.1
        let lastActive = snapshotState.2
        let manualWasSet = snapshotState.3
        let isHot = (gpuTemp ?? 0) >= 85.0 || thermalState == "Serious" || thermalState == "Critical"

        switch activePolicy {
        case .auto:
            if manualWasSet { _ = writeAutoMode(fanCount: fanCount) }
            return "System Auto"
        case .proactive:
            let withinCooldown = lastActive.map { now.timeIntervalSince($0) < Self.proactiveCooldownSeconds } ?? false
            let active = phase == "prefill" || phase == "decode" || withinCooldown || (gpuTemp ?? 0) >= 72.0 || isHot
            if !active && (gpuTemp ?? 0) < 58.0 {
                if manualWasSet { _ = writeAutoMode(fanCount: fanCount) }
                return isPrivilegedOrHelperPresent(now: now) ? "Proactive (Idle · Auto)" : Self.unauthorizedStatus
            }
            let ratio = isHot ? 1.0 : 0.80
            return applyManualRatio(ratio, fanCount: fanCount, actualRPMs: actualRPMs, minRPMs: minRPMs, maxRPMs: maxRPMs, label: "Proactive Boost")
        case .max:
            return applyManualRatio(1.0, fanCount: fanCount, actualRPMs: actualRPMs, minRPMs: minRPMs, maxRPMs: maxRPMs, label: "Max Blast")
        }
    }

    private func applyManualRatio(
        _ ratio: Double,
        fanCount: Int,
        actualRPMs: [Double],
        minRPMs: [Double],
        maxRPMs: [Double],
        label: String
    ) -> String {
        var appliedTargets: [Double] = []
        var writeFailed = false
        for idx in 0..<min(fanCount, 4) {
            let maxRPM = idx < maxRPMs.count ? maxRPMs[idx] : 5000.0
            let minRPM = idx < minRPMs.count ? minRPMs[idx] : 1200.0
            let curRPM = idx < actualRPMs.count ? actualRPMs[idx] : 0.0
            let safeTarget = min(maxRPM, max(curRPM, minRPM, maxRPM * ratio))
            if writeTarget(fanIndex: idx, rpm: safeTarget) {
                appliedTargets.append(safeTarget)
            } else {
                writeFailed = true
            }
        }
        lock.withLock {
            lastWriteAuthorized = !writeFailed
            if !writeFailed { didApplyManualOverride = true }
        }
        if writeFailed || appliedTargets.isEmpty {
            return Self.unauthorizedStatus
        }
        let avg = appliedTargets.reduce(0, +) / Double(appliedTargets.count)
        return "\(label) (\(Int(avg.rounded())) RPM)"
    }

    @discardableResult
    private func writeAutoMode(fanCount: Int) -> Bool {
        if FileManager.default.fileExists(atPath: socketPath),
           QWSMCHelperClient.send("AUTO\n", socketPath: socketPath)?.ok == true {
            lock.withLock {
                didApplyManualOverride = false
                lastWriteAuthorized = true
                lastPingResult = true
                lastPingAt = Date()
            }
            return true
        }
        guard Darwin.geteuid() == 0 else {
            lock.withLock { lastWriteAuthorized = false }
            return false
        }
        var ok = true
        for idx in 0..<min(fanCount, 4) {
            if client.writeUInt8("F\(idx)md", value: 0) != KERN_SUCCESS { ok = false }
        }
        lock.withLock {
            if ok { didApplyManualOverride = false }
            lastWriteAuthorized = ok
        }
        return ok
    }

    private func writeTarget(fanIndex: Int, rpm: Double) -> Bool {
        let roundedRPM = Int(rpm.rounded())
        if FileManager.default.fileExists(atPath: socketPath) {
            let resp = QWSMCHelperClient.send("SET \(fanIndex) \(roundedRPM)\n", socketPath: socketPath)
            if resp?.ok == true {
                lock.withLock {
                    lastPingResult = true
                    lastPingAt = Date()
                }
                return true
            }
            if Darwin.geteuid() != 0 { return false }
        }
        guard Darwin.geteuid() == 0 else { return false }
        let modeRes = client.writeUInt8("F\(fanIndex)md", value: 1)
        let tgRes = client.writeFloat("F\(fanIndex)Tg", value: Float(rpm))
        return modeRes == KERN_SUCCESS && tgRes == KERN_SUCCESS
    }

    private func isPrivilegedOrHelperPresent(now: Date) -> Bool {
        if Darwin.geteuid() == 0 { return true }
        if !FileManager.default.fileExists(atPath: socketPath) {
            lock.withLock {
                lastPingResult = false
                lastPingAt = now
                lastWriteAuthorized = false
            }
            return false
        }
        let cached = lock.withLock { () -> Bool? in
            if let at = lastPingAt, now.timeIntervalSince(at) < Self.helperPingCacheTTL {
                return lastPingResult
            }
            return nil
        }
        if let cached { return cached }
        let alive = QWSMCHelperClient.ping(socketPath: socketPath)
        lock.withLock {
            lastPingResult = alive
            lastPingAt = now
            lastWriteAuthorized = alive
        }
        return alive
    }

    private func restoreAutoOnExit() {
        let shouldRestore = lock.withLock { didApplyManualOverride }
        guard shouldRestore else { return }
        _ = writeAutoMode(fanCount: 4)
    }

    private static func maxSensor(keys: [String], client: QWSMCClient) -> Double? {
        var maxVal: Double?
        for key in keys {
            if let v = client.readValue(key), v >= 15.0 && v <= 135.0 {
                maxVal = max(maxVal ?? v, v)
            }
        }
        return maxVal
    }

    private static func currentThermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Nominal"
        }
    }
}
