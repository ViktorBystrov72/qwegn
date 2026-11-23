// QwgUpdate.swift — `qwg update` CLI command for release checks, SHA-256 verification, and Installer launch.

import ArgumentParser
import Darwin
import Foundation
import QWCore
import QWServer

struct UpdateOptions: Sendable {
    var check: Bool = false
    var downloadOnly: Bool = false
    var notes: Bool = false
    var skip: Bool = false
    var unskip: Bool = false
    var json: Bool = false
    var yes: Bool = false
    var force: Bool = false
}

struct Update: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "update",
        abstract: "Check for Qwegn updates, verify macOS .pkg SHA-256, and open Installer.app."
    )

    @Flag(name: .customLong("check"), help: "Check release availability without downloading.")
    var check: Bool = false

    @Flag(name: .customLong("download-only"), help: "Download and verify .pkg SHA-256 without opening Installer.app.")
    var downloadOnly: Bool = false

    @Flag(name: .customLong("notes"), help: "Print release notes for the latest release.")
    var notes: Bool = false

    @Flag(name: .customLong("skip"), help: "Skip the latest release version from future nudges.")
    var skip: Bool = false

    @Flag(name: .customLong("unskip"), help: "Clear any previously skipped version.")
    var unskip: Bool = false

    @Flag(name: .customLong("json"), help: "Emit machine-readable QWUpdateStatusSnapshot JSON.")
    var json: Bool = false

    @Flag(name: .customLong("yes"), help: "Launch Installer.app without prompting if a qwg server is running.")
    var yes: Bool = false

    @Flag(name: .customLong("force"), help: "Bypass cached ETag and perform a fresh release lookup.")
    var force: Bool = false

    func run() async throws {
        let options = UpdateOptions(
            check: check, downloadOnly: downloadOnly, notes: notes,
            skip: skip, unskip: unskip, json: json, yes: yes, force: force
        )
        try await QwgUpdateRunner.run(options: options)
    }
}

enum QwgUpdateRunner {
    static func run(
        options: UpdateOptions,
        checker: QWUpdateChecker = QWUpdateChecker(),
        stateURL: URL = QWPaths.updateStateFileURL,
        cacheDir: URL = QWPaths.updatesCacheDirectoryURL,
        installOrigin: QWInstallOrigin = QWInstallOrigin.detect(),
        currentVersion: String = QWVersion.current,
        runningEnginePIDProvider: () -> Int32? = { QWEngineInstanceLock.runningEnginePID() },
        stdoutWriter: @escaping @Sendable (String) -> Void = { print($0) },
        stderrWriter: @escaping @Sendable (String) -> Void = {
            FileHandle.standardError.write(Data($0.utf8))
        },
        installerLauncher: @escaping @Sendable (URL) async throws -> Int32 = defaultLaunchInstaller
    ) async throws {
        let settings = QWSettingsManager.shared.loadSettings()
        let schedule = QWUpdateCheckSchedule.resolve(settings.updateCheckSchedule)

        if options.unskip {
            var state = QWUpdateStateStore.load(from: stateURL)
            state.skippedVersion = nil
            try QWUpdateStateStore.save(state, to: stateURL)
            let snap = QWUpdateChecker.buildSnapshot(
                currentVersion: currentVersion,
                state: state, installOrigin: installOrigin, schedule: schedule, phase: .idle
            )
            if options.json {
                try emitJSON(snap, writer: stdoutWriter)
            } else {
                stdoutWriter("Cleared skipped version.")
            }
            return
        }

        let checkedState: QWUpdateState
        do {
            (checkedState, _) = try await checker.check(
                currentVersion: currentVersion, stateURL: stateURL, force: options.force
            )
        } catch {
            let fallbackState = QWUpdateStateStore.load(from: stateURL)
            let errSnap = QWUpdateChecker.buildSnapshot(
                currentVersion: currentVersion,
                state: fallbackState,
                installOrigin: installOrigin,
                schedule: schedule,
                phase: .error,
                errorMessage: error.localizedDescription
            )
            if options.json {
                try emitJSON(errSnap, writer: stdoutWriter)
            } else {
                stderrWriter("qwg update: \(error.localizedDescription)\n")
            }
            throw ExitCode(1)
        }

        var state = checkedState
        let available = QWUpdateChecker.isUpdateAvailable(
            currentVersion: currentVersion,
            latest: state.cachedRelease
        )

        if options.skip {
            guard let release = state.cachedRelease else {
                stderrWriter("qwg update: no release metadata available to skip.\n")
                throw ExitCode(1)
            }
            state.skippedVersion = release.version.description
            try QWUpdateStateStore.save(state, to: stateURL)
            let snap = QWUpdateChecker.buildSnapshot(
                currentVersion: currentVersion,
                state: state,
                installOrigin: installOrigin,
                schedule: schedule,
                phase: available ? .available : .upToDate
            )
            if options.json {
                try emitJSON(snap, writer: stdoutWriter)
            } else {
                stdoutWriter("Skipped version \(release.version.description) from future update nudges.")
            }
            return
        }

        let initialPhase: QWUpdatePhase = available ? .available : .upToDate
        let checkSnap = QWUpdateChecker.buildSnapshot(
            currentVersion: currentVersion,
            state: state, installOrigin: installOrigin, schedule: schedule, phase: initialPhase
        )

        if options.check || options.notes {
            if options.json {
                try emitJSON(checkSnap, writer: stdoutWriter)
            } else {
                stdoutWriter(formatStatusReport(snapshot: checkSnap, showNotes: options.notes))
            }
            return
        }

        guard available, let release = state.cachedRelease else {
            if options.json {
                try emitJSON(checkSnap, writer: stdoutWriter)
            } else {
                stdoutWriter("Qwegn \(currentVersion) is up to date.")
            }
            return
        }

        if !installOrigin.canInstallPkg && !options.downloadOnly {
            if options.json {
                try emitJSON(checkSnap, writer: stdoutWriter)
            } else {
                stdoutWriter(formatStatusReport(snapshot: checkSnap, showNotes: false))
                stdoutWriter("")
                stdoutWriter("Notice: Installed via \(installOrigin.displayTitle) (no ai.qwegn.pkg receipt found).")
                stdoutWriter("Download .pkg or .tar.gz from: \(release.htmlURL.absoluteString)")
                stdoutWriter("Or run 'qwg update --download-only' to fetch the SHA-256 verified .pkg.")
            }
            return
        }

        let downloader = QWUpdateDownloader(
            transport: checker.transport,
            extraAllowedHosts: checker.extraAllowedHosts,
            allowLoopbackHTTP: checker.allowLoopbackHTTP
        )
        if !options.json {
            stdoutWriter("Found Qwegn \(release.tagName) (installed: v\(currentVersion))")
        }

        let pkg = try await downloader.downloadAndVerify(
            release: release,
            currentVersion: currentVersion,
            cacheDirectory: cacheDir,
            stateURL: stateURL
        ) { written, expected in
            guard !options.json, isatty(STDOUT_FILENO) != 0, let expected, expected > 0 else { return }
            let pct = Int((Double(written) / Double(expected) * 100.0).rounded())
            let doneMB = Double(written) / (1024 * 1024)
            let totMB = Double(expected) / (1024 * 1024)
            let line = String(
                format: "\rDownloading %@: %3d%% (%.1f / %.1f MB)",
                release.pkgAsset?.name ?? "package", pct, doneMB, totMB
            )
            FileHandle.standardOutput.write(Data(line.utf8))
        }
        if !options.json && isatty(STDOUT_FILENO) != 0 {
            stdoutWriter("")
        }

        let updatedState = QWUpdateStateStore.load(from: stateURL)
        let verifiedSnap = QWUpdateChecker.buildSnapshot(
            currentVersion: currentVersion,
            state: updatedState,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: .readyToInstall,
            downloadedBytes: pkg.sizeBytes,
            totalBytes: pkg.sizeBytes,
            signatureSummary: pkg.signatureSummary
        )

        if options.downloadOnly {
            if options.json {
                try emitJSON(verifiedSnap, writer: stdoutWriter)
            } else {
                stdoutWriter("Verified SHA-256: \(pkg.sha256Hex)")
                if let sig = pkg.signatureSummary {
                    stdoutWriter("Signature       : \(sig)")
                }
                stdoutWriter("Staged package  : \(pkg.pkgURL.path)")
            }
            return
        }

        if let runningPid = runningEnginePIDProvider(), !options.yes {
            let msg = "Warning: qwg engine (PID \(runningPid)) is running; Installer.app will replace Qwegn binaries."
            if isatty(STDIN_FILENO) != 0 && !options.json {
                stdoutWriter(msg)
                print("Continue launching Installer.app? [y/N]: ", terminator: "")
                let reply = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
                guard reply == "y" || reply == "yes" else {
                    stdoutWriter("Aborted launch; verified installer remains staged at \(pkg.pkgURL.path)")
                    return
                }
            } else {
                stderrWriter("qwg update: \(msg) Re-run with --yes to proceed.\n")
                throw ExitCode(1)
            }
        }

        let openExit = try await installerLauncher(pkg.pkgURL)
        guard openExit == 0 else {
            stderrWriter("qwg update: failed to launch Installer.app (exit \(openExit)).\n")
            throw ExitCode(1)
        }
        if options.json {
            try emitJSON(verifiedSnap, writer: stdoutWriter)
        } else {
            stdoutWriter("Verified SHA-256: \(pkg.sha256Hex)")
            stdoutWriter("Opened Installer.app for \(pkg.pkgURL.path)")
        }
    }

    private static func formatStatusReport(snapshot: QWUpdateStatusSnapshot, showNotes: Bool) -> String {
        var lines: [String] = []
        lines.append("Current version : \(snapshot.currentVersion) (\(snapshot.installOrigin.displayTitle))")
        if let rel = snapshot.latestRelease {
            let dateStr = rel.publishedAt.map { ISO8601DateFormatter().string(from: $0) } ?? "unknown date"
            lines.append("Latest release  : \(rel.version.description) (\(dateStr))")
            let statusStr = snapshot.updateAvailable
                ? (snapshot.isSkipped ? "UPDATE AVAILABLE (SKIPPED)" : "UPDATE AVAILABLE")
                : "UP TO DATE"
            lines.append("Status          : \(statusStr)")
            if let asset = rel.pkgAsset {
                let mb = asset.sizeBytes.map { String(format: " (%.1f MB)", Double($0) / (1024 * 1024)) } ?? ""
                lines.append("Package asset   : \(asset.name)\(mb)")
            }
            if showNotes && !rel.notes.isEmpty {
                lines.append("")
                lines.append("Release notes (\(rel.tagName)):")
                lines.append(rel.notes)
            } else if snapshot.updateAvailable {
                lines.append("")
                lines.append("Run 'qwg update' to download, verify SHA-256, and open Installer.app.")
            }
        } else {
            lines.append("Status          : NO RELEASE METADATA")
        }
        return lines.joined(separator: "\n")
    }

    private static func emitJSON(_ snapshot: QWUpdateStatusSnapshot, writer: (String) -> Void) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(snapshot)
        if let text = String(data: data, encoding: .utf8) {
            writer(text)
        }
    }

    static func defaultLaunchInstaller(_ pkgURL: URL) async throws -> Int32 {
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
