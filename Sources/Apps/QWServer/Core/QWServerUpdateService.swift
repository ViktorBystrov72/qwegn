// QWServerUpdateService.swift — Server-side background check scheduler and verified update coordinator.

import Foundation
import QWCore

/// Envelope returned by `/v1/update*` routes for offline WebUI and API consumers.
public struct QWUpdateServerResponse: Codable, Sendable, Equatable {
    public let snapshot: QWUpdateStatusSnapshot
    public let inferenceActive: Bool
    public let loopbackDesktopActionsAllowed: Bool

    public init(
        snapshot: QWUpdateStatusSnapshot,
        inferenceActive: Bool,
        loopbackDesktopActionsAllowed: Bool
    ) {
        self.snapshot = snapshot
        self.inferenceActive = inferenceActive
        self.loopbackDesktopActionsAllowed = loopbackDesktopActionsAllowed
    }
}

/// Errors specific to server-initiated update download or install handoff.
public enum QWServerUpdateInstallError: Error, LocalizedError, Equatable, Sendable {
    case inferenceActive

    public var errorDescription: String? {
        "Active inference or model transition is in progress; wait for completion or pass force=true."
    }
}

/// Coordinates background release checks, single-flight verified downloads, and Installer handoff.
public actor QWServerUpdateService {
    private let checker: QWUpdateChecker
    private let downloader: QWUpdateDownloader
    private let currentVersion: String
    private let stateURL: URL
    private let cacheDirectory: URL
    private let installOrigin: QWInstallOrigin
    private let installerLauncher: @Sendable (URL) async throws -> Int32

    private var schedule: QWUpdateCheckSchedule
    private var phase: QWUpdatePhase
    private var downloadedBytes: Int64?
    private var totalBytes: Int64?
    private var signatureSummary: String?
    private var lastError: String?
    private var activeCheckTask: Task<QWUpdateStatusSnapshot, Never>?
    private var activeDownloadTask: Task<Void, Never>?
    private var schedulerTask: Task<Void, Never>?

    public init(
        checker: QWUpdateChecker = QWUpdateChecker(),
        downloader: QWUpdateDownloader? = nil,
        currentVersion: String = QWVersion.current,
        stateURL: URL = QWPaths.updateStateFileURL,
        cacheDirectory: URL = QWPaths.updatesCacheDirectoryURL,
        installOrigin: QWInstallOrigin = QWInstallOrigin.detect(),
        schedule: QWUpdateCheckSchedule = .resolve(QWSettingsManager.shared.loadSettings().updateCheckSchedule),
        installerLauncher: (@Sendable (URL) async throws -> Int32)? = nil
    ) {
        self.checker = checker
        self.downloader = downloader ?? QWUpdateDownloader(
            transport: checker.transport,
            extraAllowedHosts: checker.extraAllowedHosts,
            allowLoopbackHTTP: checker.allowLoopbackHTTP
        )
        self.currentVersion = currentVersion
        self.stateURL = stateURL
        self.cacheDirectory = cacheDirectory
        self.installOrigin = installOrigin
        self.schedule = schedule
        self.installerLauncher = installerLauncher ?? Self.defaultInstallerLauncher

        let initialState = QWUpdateStateStore.load(from: stateURL)
        self.phase = Self.derivedIdlePhase(currentVersion: currentVersion, state: initialState)
        self.lastError = initialState.lastError
    }

    public func startScheduler(initialDelaySeconds: TimeInterval = 5.0) {
        schedulerTask?.cancel()
        schedulerTask = Task { [weak self] in
            if initialDelaySeconds > 0 {
                try? await Task.sleep(nanoseconds: UInt64(initialDelaySeconds * 1_000_000_000))
            }
            guard !Task.isCancelled, let self else { return }
            await self.runScheduledCheck(trigger: .launch)
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_600_000_000_000)
                guard !Task.isCancelled else { break }
                await self.runScheduledCheck(trigger: .dailyTimer)
            }
        }
    }

    public func stopScheduler() {
        schedulerTask?.cancel()
        schedulerTask = nil
        activeCheckTask?.cancel()
        activeCheckTask = nil
        activeDownloadTask?.cancel()
        activeDownloadTask = nil
    }

    public func updateSchedule(_ newSchedule: QWUpdateCheckSchedule) {
        schedule = newSchedule
    }

    public func currentSnapshot() -> QWUpdateStatusSnapshot {
        let state = QWUpdateStateStore.load(from: stateURL)
        return QWUpdateChecker.buildSnapshot(
            currentVersion: currentVersion,
            state: state,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: effectivePhase(for: state),
            downloadedBytes: downloadedBytes,
            totalBytes: totalBytes,
            signatureSummary: signatureSummary,
            errorMessage: lastError
        )
    }

    public func checkNow(force: Bool = false) async -> QWUpdateStatusSnapshot {
        if let existing = activeCheckTask {
            return await existing.value
        }
        let task = Task { () -> QWUpdateStatusSnapshot in
            await self.performCheck(force: force)
        }
        activeCheckTask = task
        let snap = await task.value
        activeCheckTask = nil
        return snap
    }

    public func setSkippedVersion(_ version: String?) throws -> QWUpdateStatusSnapshot {
        var state = QWUpdateStateStore.load(from: stateURL)
        let trimmed = version?.trimmingCharacters(in: .whitespacesAndNewlines)
        state.skippedVersion = (trimmed?.isEmpty == false) ? trimmed : nil
        try QWUpdateStateStore.save(state, to: stateURL)
        if activeDownloadTask == nil {
            phase = Self.derivedIdlePhase(currentVersion: currentVersion, state: state)
        }
        return currentSnapshot()
    }

    public func startDownload() throws -> QWUpdateStatusSnapshot {
        if activeDownloadTask != nil {
            return currentSnapshot()
        }
        let state = QWUpdateStateStore.load(from: stateURL)
        guard let release = state.cachedRelease,
              QWUpdateChecker.isUpdateAvailable(currentVersion: currentVersion, latest: release) else {
            throw QWUpdateError.missingAsset("No newer release available")
        }
        guard release.pkgAsset != nil else {
            throw QWUpdateError.missingAsset("Qwegn-\(release.version.description)-macos-arm64.pkg")
        }
        guard release.checksumsAsset != nil else {
            throw QWUpdateError.missingAsset("SHA256SUMS.txt")
        }

        phase = .downloading
        downloadedBytes = 0
        totalBytes = release.pkgAsset?.sizeBytes
        lastError = nil

        activeDownloadTask = Task { [weak self] in
            guard let self else { return }
            await self.runDownload(release: release)
        }
        return currentSnapshot()
    }

    public func cancelDownload() {
        activeDownloadTask?.cancel()
        activeDownloadTask = nil
        downloadedBytes = nil
        totalBytes = nil
        let state = QWUpdateStateStore.load(from: stateURL)
        phase = Self.derivedIdlePhase(currentVersion: currentVersion, state: state)
    }

    public func installVerifiedPackage(inferenceActive: Bool, force: Bool) async throws -> QWUpdateStatusSnapshot {
        if inferenceActive && !force {
            throw QWServerUpdateInstallError.inferenceActive
        }
        let state = QWUpdateStateStore.load(from: stateURL)
        guard let path = state.verifiedPkgPath,
              let expectedHash = state.verifiedPkgSHA256 else {
            throw QWUpdateError.missingVerifiedPackage
        }
        let pkgURL = URL(fileURLWithPath: path)
        guard try QWUpdateDownloader.verifyCachedPackage(at: pkgURL, expectedSHA256: expectedHash) else {
            throw QWUpdateError.missingVerifiedPackage
        }
        let exitCode = try await installerLauncher(pkgURL)
        guard exitCode == 0 else {
            throw QWUpdateError.installerLaunchFailed("open exited with status \(exitCode)")
        }
        return currentSnapshot()
    }

    private func runScheduledCheck(trigger: QWUpdateCheckTrigger) async {
        let state = QWUpdateStateStore.load(from: stateURL)
        guard QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: schedule,
            trigger: trigger,
            state: state
        ) else {
            return
        }
        _ = await checkNow(force: false)
    }

    private func performCheck(force: Bool) async -> QWUpdateStatusSnapshot {
        let priorPhase = phase
        if activeDownloadTask == nil {
            phase = .checking
        }
        lastError = nil
        do {
            let (state, _) = try await checker.check(
                currentVersion: currentVersion,
                stateURL: stateURL,
                force: force
            )
            if activeDownloadTask == nil {
                phase = Self.derivedIdlePhase(currentVersion: currentVersion, state: state)
            } else {
                phase = priorPhase
            }
            lastError = nil
            return currentSnapshot()
        } catch {
            if activeDownloadTask == nil {
                phase = .error
            }
            lastError = error.localizedDescription
            return currentSnapshot()
        }
    }

    private func runDownload(release: QWReleaseInfo) async {
        defer { activeDownloadTask = nil }
        do {
            let pkg = try await downloader.downloadAndVerify(
                release: release,
                currentVersion: currentVersion,
                cacheDirectory: cacheDirectory,
                stateURL: stateURL
            ) { [weak self] written, expected in
                Task {
                    await self?.updateProgress(written: written, expected: expected)
                }
            }
            downloadedBytes = pkg.sizeBytes
            totalBytes = pkg.sizeBytes
            signatureSummary = pkg.signatureSummary
            lastError = nil
            phase = .readyToInstall
        } catch is CancellationError {
            let state = QWUpdateStateStore.load(from: stateURL)
            downloadedBytes = nil
            totalBytes = nil
            phase = Self.derivedIdlePhase(currentVersion: currentVersion, state: state)
        } catch {
            downloadedBytes = nil
            totalBytes = nil
            lastError = error.localizedDescription
            phase = .error
        }
    }

    private func updateProgress(written: Int64, expected: Int64?) {
        guard activeDownloadTask != nil else { return }
        downloadedBytes = written
        if let expected, expected > 0 {
            totalBytes = expected
            if written >= expected {
                phase = .verifying
            }
        }
    }

    private func effectivePhase(for state: QWUpdateState) -> QWUpdatePhase {
        if phase == .downloading || phase == .verifying || phase == .checking || phase == .error {
            return phase
        }
        return Self.derivedIdlePhase(currentVersion: currentVersion, state: state)
    }

    private static func derivedIdlePhase(currentVersion: String, state: QWUpdateState) -> QWUpdatePhase {
        let available = QWUpdateChecker.isUpdateAvailable(currentVersion: currentVersion, latest: state.cachedRelease)
        if available,
           let release = state.cachedRelease,
           let ver = state.verifiedPkgVersion,
           let path = state.verifiedPkgPath,
           ver == release.version.description,
           FileManager.default.fileExists(atPath: path) {
            return .readyToInstall
        }
        if available {
            return .available
        }
        if state.lastCheckSucceededAt != nil {
            return .upToDate
        }
        return .idle
    }

    private static func defaultInstallerLauncher(_ pkgURL: URL) async throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Installer", pkgURL.path]
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
}
