// QWUpdateController.swift — @MainActor @Observable update coordinator for Qwegn.app.

import AppKit
import Foundation
import Observation
import QWCore

/// Observable update state, background schedule runner, and Installer.app handoff for QWMenuBar.
@MainActor
@Observable
public final class QWUpdateController {
    public private(set) var status: QWUpdateStatusSnapshot
    public private(set) var isChecking: Bool = false
    public private(set) var isDownloading: Bool = false
    public private(set) var downloadFraction: Double?
    public private(set) var downloadedBytes: Int64?
    public private(set) var totalBytes: Int64?
    public private(set) var errorMessage: String?

    public var shouldShowBadge: Bool {
        if isDownloading || status.phase == .verifying || status.phase == .readyToInstall {
            return true
        }
        return status.updateAvailable && !status.isSkipped
    }

    public var badgeText: String {
        let ver = status.latestRelease?.tagName ?? ""
        if status.phase == .verifying {
            return "Verifying…"
        }
        if isDownloading {
            if let frac = downloadFraction {
                return "↓ \(Int((frac * 100.0).rounded()))%"
            }
            return "↓ …"
        }
        if status.phase == .readyToInstall {
            return ver.isEmpty ? "✓ Ready" : "✓ \(ver)"
        }
        return ver.isEmpty ? "↑ Update" : "↑ \(ver)"
    }

    @ObservationIgnored private var schedule: QWUpdateCheckSchedule
    @ObservationIgnored private let currentVersion: String
    @ObservationIgnored private let installOrigin: QWInstallOrigin
    @ObservationIgnored private let stateURL: URL
    @ObservationIgnored private let cacheDirectory: URL
    @ObservationIgnored private let checker: QWUpdateChecker
    @ObservationIgnored private let downloader: QWUpdateDownloader
    @ObservationIgnored private let openInstallerHandler: @Sendable (URL) async throws -> Bool
    @ObservationIgnored private let revealFileHandler: @MainActor (URL) -> Void
    @ObservationIgnored private let openURLHandler: @MainActor (URL) -> Void
    @ObservationIgnored private var launchTask: Task<Void, Never>?
    @ObservationIgnored private var dailyTimer: Timer?
    @ObservationIgnored private var checkTask: Task<Void, Never>?
    @ObservationIgnored private var downloadTask: Task<Void, Never>?
    @ObservationIgnored private var signatureSummary: String?

    public init(
        schedule: QWUpdateCheckSchedule = .onLaunchAndDaily,
        currentVersion: String = QWVersion.current,
        installOrigin: QWInstallOrigin = QWInstallOrigin.detect(),
        stateURL: URL = QWPaths.updateStateFileURL,
        cacheDirectory: URL = QWPaths.updatesCacheDirectoryURL,
        checker: QWUpdateChecker = QWUpdateChecker(),
        downloader: QWUpdateDownloader? = nil,
        openInstallerHandler: (@Sendable (URL) async throws -> Bool)? = nil,
        revealFileHandler: (@MainActor (URL) -> Void)? = nil,
        openURLHandler: (@MainActor (URL) -> Void)? = nil
    ) {
        self.schedule = schedule
        self.currentVersion = currentVersion
        self.installOrigin = installOrigin
        self.stateURL = stateURL
        self.cacheDirectory = cacheDirectory
        self.checker = checker
        self.downloader = downloader ?? QWUpdateDownloader(
            transport: checker.transport,
            extraAllowedHosts: checker.extraAllowedHosts,
            allowLoopbackHTTP: checker.allowLoopbackHTTP
        )
        self.openInstallerHandler = openInstallerHandler ?? Self.defaultOpenInstaller
        self.revealFileHandler = revealFileHandler ?? { url in
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
        self.openURLHandler = openURLHandler ?? { url in
            NSWorkspace.shared.open(url)
        }

        let initialState = QWUpdateStateStore.load(from: stateURL)
        self.status = Self.makeSnapshot(
            currentVersion: currentVersion,
            state: initialState,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: Self.idlePhase(currentVersion: currentVersion, state: initialState),
            signatureSummary: nil,
            errorMessage: initialState.lastError
        )
        self.errorMessage = initialState.lastError
    }

    public func startScheduler(initialDelaySeconds: TimeInterval = 5.0) {
        stopScheduler()
        launchTask = Task { [weak self] in
            if initialDelaySeconds > 0 {
                try? await Task.sleep(nanoseconds: UInt64(initialDelaySeconds * 1_000_000_000))
            }
            guard !Task.isCancelled, let self else { return }
            let state = QWUpdateStateStore.load(from: self.stateURL)
            if QWUpdateChecker.shouldPerformAutomaticCheck(
                schedule: self.schedule,
                trigger: .launch,
                state: state
            ) {
                await self.checkNowAndAwait(force: false)
            }
        }

        let timer = Timer(timeInterval: 3600, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let state = QWUpdateStateStore.load(from: self.stateURL)
                if QWUpdateChecker.shouldPerformAutomaticCheck(
                    schedule: self.schedule,
                    trigger: .dailyTimer,
                    state: state
                ) {
                    self.checkNow(force: false)
                }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        dailyTimer = timer
    }

    public func stopScheduler() {
        launchTask?.cancel()
        launchTask = nil
        dailyTimer?.invalidate()
        dailyTimer = nil
        checkTask?.cancel()
        checkTask = nil
        downloadTask?.cancel()
        downloadTask = nil
    }

    public func updateSchedule(_ newSchedule: QWUpdateCheckSchedule) {
        schedule = newSchedule
        let state = QWUpdateStateStore.load(from: stateURL)
        refreshSnapshot(from: state, phase: status.phase, error: errorMessage)
    }

    public func checkNow(force: Bool = true) {
        guard checkTask == nil else { return }
        checkTask = Task { @MainActor [weak self] in
            await self?.runCheck(force: force)
        }
    }

    public func checkNowAndAwait(force: Bool = true) async {
        if let existing = checkTask {
            await existing.value
            return
        }
        let task = Task<Void, Never> { @MainActor [weak self] in
            guard let self else { return }
            await self.runCheck(force: force)
        }
        checkTask = task
        await task.value
    }

    public func startDownload() {
        guard downloadTask == nil else { return }
        downloadTask = Task<Void, Never> { @MainActor [weak self] in
            guard let self else { return }
            await self.runDownload()
        }
    }

    public func startDownloadAndAwait() async {
        if let existing = downloadTask {
            await existing.value
            return
        }
        let task = Task<Void, Never> { @MainActor [weak self] in
            guard let self else { return }
            await self.runDownload()
        }
        downloadTask = task
        await task.value
    }

    public func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        downloadFraction = nil
        downloadedBytes = nil
        totalBytes = nil
        let state = QWUpdateStateStore.load(from: stateURL)
        refreshSnapshot(from: state, phase: Self.idlePhase(currentVersion: currentVersion, state: state), error: nil)
    }

    public func openInstaller() {
        let state = QWUpdateStateStore.load(from: stateURL)
        guard let path = state.verifiedPkgPath, let sha = state.verifiedPkgSHA256 else {
            errorMessage = QWUpdateError.missingVerifiedPackage.localizedDescription
            refreshSnapshot(from: state, phase: .error, error: errorMessage)
            return
        }
        let pkgURL = URL(fileURLWithPath: path)
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let valid = try await Task.detached(priority: .userInitiated) {
                    try QWUpdateDownloader.verifyCachedPackage(at: pkgURL, expectedSHA256: sha)
                }.value
                guard valid else {
                    self.errorMessage = QWUpdateError.missingVerifiedPackage.localizedDescription
                    self.refreshSnapshot(from: state, phase: .error, error: self.errorMessage)
                    return
                }
                let ok = try await self.openInstallerHandler(pkgURL)
                if !ok {
                    self.errorMessage = QWUpdateError.installerLaunchFailed("Installer.app did not open").localizedDescription
                    self.refreshSnapshot(from: state, phase: .error, error: self.errorMessage)
                }
            } catch {
                self.errorMessage = error.localizedDescription
                self.refreshSnapshot(from: state, phase: .error, error: self.errorMessage)
            }
        }
    }

    public func revealDownloadedPkg() {
        guard let path = status.downloadedPkgPath else { return }
        revealFileHandler(URL(fileURLWithPath: path))
    }

    public func openReleasePage() {
        guard let url = status.latestRelease?.htmlURL,
              QWUpdateChecker.isAllowedUpdateURL(
                  url,
                  extraAllowedHosts: checker.extraAllowedHosts,
                  allowLoopbackHTTP: checker.allowLoopbackHTTP
              ) else {
            return
        }
        openURLHandler(url)
    }

    public func skipCurrentUpdate() {
        guard let version = status.latestRelease?.version.description else { return }
        var state = QWUpdateStateStore.load(from: stateURL)
        state.skippedVersion = version
        try? QWUpdateStateStore.save(state, to: stateURL)
        refreshSnapshot(from: state, phase: Self.idlePhase(currentVersion: currentVersion, state: state), error: nil)
    }

    public func clearSkippedVersion() {
        var state = QWUpdateStateStore.load(from: stateURL)
        state.skippedVersion = nil
        try? QWUpdateStateStore.save(state, to: stateURL)
        refreshSnapshot(from: state, phase: Self.idlePhase(currentVersion: currentVersion, state: state), error: nil)
    }

    private func runCheck(force: Bool) async {
        defer { checkTask = nil }
        isChecking = true
        errorMessage = nil
        let before = QWUpdateStateStore.load(from: stateURL)
        if !isDownloading {
            refreshSnapshot(from: before, phase: .checking, error: nil)
        }
        defer { isChecking = false }

        do {
            let (updated, _) = try await checker.check(
                currentVersion: currentVersion,
                stateURL: stateURL,
                force: force
            )
            let nextPhase = isDownloading ? status.phase : Self.idlePhase(currentVersion: currentVersion, state: updated)
            refreshSnapshot(from: updated, phase: nextPhase, error: nil)
        } catch is CancellationError {
            let cur = QWUpdateStateStore.load(from: stateURL)
            refreshSnapshot(from: cur, phase: Self.idlePhase(currentVersion: currentVersion, state: cur), error: nil)
        } catch {
            errorMessage = error.localizedDescription
            let cur = QWUpdateStateStore.load(from: stateURL)
            if !isDownloading {
                refreshSnapshot(from: cur, phase: .error, error: errorMessage)
            }
        }
    }

    private func runDownload() async {
        defer { downloadTask = nil }
        let state = QWUpdateStateStore.load(from: stateURL)
        guard let release = state.cachedRelease,
              QWUpdateChecker.isUpdateAvailable(currentVersion: currentVersion, latest: release) else {
            return
        }

        isDownloading = true
        errorMessage = nil
        downloadedBytes = 0
        totalBytes = release.pkgAsset?.sizeBytes
        downloadFraction = 0
        refreshSnapshot(from: state, phase: .downloading, error: nil)

        do {
            let pkg = try await downloader.downloadAndVerify(
                release: release,
                currentVersion: currentVersion,
                cacheDirectory: cacheDirectory,
                stateURL: stateURL
            ) { [weak self] written, expected in
                Task { @MainActor [weak self] in
                    guard let self, self.isDownloading else { return }
                    self.downloadedBytes = written
                    if let expected, expected > 0 {
                        self.totalBytes = expected
                        self.downloadFraction = min(1.0, max(0.0, Double(written) / Double(expected)))
                        if written >= expected {
                            let cur = QWUpdateStateStore.load(from: self.stateURL)
                            self.refreshSnapshot(from: cur, phase: .verifying, error: nil)
                        }
                    }
                }
            }
            isDownloading = false
            downloadedBytes = pkg.sizeBytes
            totalBytes = pkg.sizeBytes
            downloadFraction = 1.0
            signatureSummary = pkg.signatureSummary
            let after = QWUpdateStateStore.load(from: stateURL)
            refreshSnapshot(from: after, phase: .readyToInstall, error: nil)
        } catch is CancellationError {
            isDownloading = false
            downloadFraction = nil
            downloadedBytes = nil
            totalBytes = nil
            let cur = QWUpdateStateStore.load(from: stateURL)
            refreshSnapshot(from: cur, phase: Self.idlePhase(currentVersion: currentVersion, state: cur), error: nil)
        } catch {
            isDownloading = false
            downloadFraction = nil
            downloadedBytes = nil
            totalBytes = nil
            errorMessage = error.localizedDescription
            let cur = QWUpdateStateStore.load(from: stateURL)
            refreshSnapshot(from: cur, phase: .error, error: errorMessage)
        }
    }

    private func refreshSnapshot(from state: QWUpdateState, phase: QWUpdatePhase, error: String?) {
        status = Self.makeSnapshot(
            currentVersion: currentVersion,
            state: state,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: phase,
            downloadedBytes: downloadedBytes,
            totalBytes: totalBytes,
            signatureSummary: signatureSummary,
            errorMessage: error
        )
    }

    private static func makeSnapshot(
        currentVersion: String,
        state: QWUpdateState,
        installOrigin: QWInstallOrigin,
        schedule: QWUpdateCheckSchedule,
        phase: QWUpdatePhase,
        downloadedBytes: Int64? = nil,
        totalBytes: Int64? = nil,
        signatureSummary: String?,
        errorMessage: String?
    ) -> QWUpdateStatusSnapshot {
        QWUpdateChecker.buildSnapshot(
            currentVersion: currentVersion,
            state: state,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: phase,
            downloadedBytes: downloadedBytes,
            totalBytes: totalBytes,
            signatureSummary: signatureSummary,
            errorMessage: errorMessage
        )
    }

    private static func idlePhase(currentVersion: String, state: QWUpdateState) -> QWUpdatePhase {
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

    private static func defaultOpenInstaller(_ pkgURL: URL) async throws -> Bool {
        try await Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            proc.arguments = ["-a", "Installer", pkgURL.path]
            try proc.run()
            proc.waitUntilExit()
            return proc.terminationStatus == 0
        }.value
    }
}
