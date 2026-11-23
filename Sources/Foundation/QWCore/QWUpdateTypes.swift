// QWUpdateTypes.swift — Canonical updater models, schedules, state store, and errors.

import Foundation

/// Automatic update check cadence persisted in QWServerSettings.
public enum QWUpdateCheckSchedule: String, Codable, CaseIterable, Sendable {
    case onLaunchAndDaily = "onLaunchAndDaily"
    case daily = "daily"
    case onLaunch = "onLaunch"
    case manual = "manual"

    public static let dailyIntervalSeconds: TimeInterval = 86_400
    public static let launchBurstCooldownSeconds: TimeInterval = 900

    public var displayTitle: String {
        switch self {
        case .onLaunchAndDaily: return "On launch & daily"
        case .daily: return "Daily (every 24h)"
        case .onLaunch: return "On launch only"
        case .manual: return "Manual only"
        }
    }

    public static func resolve(_ raw: String?) -> QWUpdateCheckSchedule {
        guard let raw else { return .onLaunchAndDaily }
        return QWUpdateCheckSchedule(rawValue: raw) ?? .onLaunchAndDaily
    }
}

/// Origin of the check trigger used to enforce schedule rules and cooldowns.
public enum QWUpdateCheckTrigger: String, Sendable {
    case launch
    case dailyTimer
    case manual
}

/// How the running Qwegn binary was installed on this Mac.
public enum QWInstallOrigin: String, Codable, Sendable {
    case pkgReceipt
    case portableBundle
    case sourceCheckout

    public var canInstallPkg: Bool { self == .pkgReceipt }

    public var displayTitle: String {
        switch self {
        case .pkgReceipt: return "macOS Installer (.pkg receipt)"
        case .portableBundle: return "Portable archive (.tar.gz)"
        case .sourceCheckout: return "Local build / source checkout"
        }
    }

    /// Classifies executable provenance cleanly without slow subprocesses on `.build/` trees.
    public static func detect(
        executablePath: String = CommandLine.arguments.first ?? "",
        receiptChecker: () -> Bool = defaultHasPkgReceipt
    ) -> QWInstallOrigin {
        let resolved = URL(fileURLWithPath: executablePath).resolvingSymlinksInPath().path
        if resolved.contains("/.build/") {
            return .sourceCheckout
        }
        let standardPkgLocation = resolved.hasPrefix("/Applications/Qwegn.app/")
            || resolved == "/usr/local/bin/qwg"
        if standardPkgLocation && receiptChecker() {
            return .pkgReceipt
        }
        return .portableBundle
    }

    public static func defaultHasPkgReceipt() -> Bool {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/pkgutil")
        proc.arguments = ["--pkgs=ai.qwegn.pkg"]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            return proc.terminationStatus == 0
        } catch {
            return false
        }
    }
}

/// Single release artifact attached to a GitHub Release.
public struct QWReleaseAsset: Codable, Sendable, Equatable {
    public let name: String
    public let downloadURL: URL
    public let sizeBytes: Int64?

    public init(name: String, downloadURL: URL, sizeBytes: Int64? = nil) {
        self.name = name
        self.downloadURL = downloadURL
        self.sizeBytes = sizeBytes
    }
}

/// Metadata describing the latest stable Qwegn release.
public struct QWReleaseInfo: Codable, Sendable, Equatable {
    public let version: QWSemVer
    public let tagName: String
    public let title: String
    public let notes: String
    public let publishedAt: Date?
    public let htmlURL: URL
    public let pkgAsset: QWReleaseAsset?
    public let tarballAsset: QWReleaseAsset?
    public let checksumsAsset: QWReleaseAsset?

    public init(
        version: QWSemVer,
        tagName: String,
        title: String,
        notes: String,
        publishedAt: Date? = nil,
        htmlURL: URL,
        pkgAsset: QWReleaseAsset? = nil,
        tarballAsset: QWReleaseAsset? = nil,
        checksumsAsset: QWReleaseAsset? = nil
    ) {
        self.version = version
        self.tagName = tagName
        self.title = title
        self.notes = notes
        self.publishedAt = publishedAt
        self.htmlURL = htmlURL
        self.pkgAsset = pkgAsset
        self.tarballAsset = tarballAsset
        self.checksumsAsset = checksumsAsset
    }
}

/// Shared updater lifecycle phase rendered across App, WebUI, and CLI.
public enum QWUpdatePhase: String, Codable, Sendable {
    case idle
    case checking
    case upToDate
    case available
    case downloading
    case verifying
    case readyToInstall
    case error
}

/// Persisted state saved under `~/.qwg/update-state.json` (`0o600`).
public struct QWUpdateState: Codable, Sendable, Equatable {
    public var etag: String?
    public var lastCheckedAt: Date?
    public var lastCheckSucceededAt: Date?
    public var cachedRelease: QWReleaseInfo?
    public var skippedVersion: String?
    public var verifiedPkgPath: String?
    public var verifiedPkgVersion: String?
    public var verifiedPkgSHA256: String?
    public var lastError: String?

    public init(
        etag: String? = nil,
        lastCheckedAt: Date? = nil,
        lastCheckSucceededAt: Date? = nil,
        cachedRelease: QWReleaseInfo? = nil,
        skippedVersion: String? = nil,
        verifiedPkgPath: String? = nil,
        verifiedPkgVersion: String? = nil,
        verifiedPkgSHA256: String? = nil,
        lastError: String? = nil
    ) {
        self.etag = etag
        self.lastCheckedAt = lastCheckedAt
        self.lastCheckSucceededAt = lastCheckSucceededAt
        self.cachedRelease = cachedRelease
        self.skippedVersion = skippedVersion
        self.verifiedPkgPath = verifiedPkgPath
        self.verifiedPkgVersion = verifiedPkgVersion
        self.verifiedPkgSHA256 = verifiedPkgSHA256
        self.lastError = lastError
    }
}

/// Atomic `0o600` JSON persistence for `QWUpdateState`.
public enum QWUpdateStateStore {
    public static func load(from url: URL = QWPaths.updateStateFileURL) -> QWUpdateState {
        guard let data = try? Data(contentsOf: url) else {
            return QWUpdateState()
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode(QWUpdateState.self, from: data)) ?? QWUpdateState()
    }

    public static func save(_ state: QWUpdateState, to url: URL = QWPaths.updateStateFileURL) throws {
        try QWPaths.ensurePrivateDirectory(url.deletingLastPathComponent())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(state)
        try data.write(to: url, options: .atomic)
        QWPaths.setPrivateFilePermissions(atPath: url.path)
    }
}

/// Canonical snapshot DTO consumed by Menu Bar, Server `/v1/update`, WebUI, and `qwg update --json`.
public struct QWUpdateStatusSnapshot: Codable, Sendable, Equatable {
    public let currentVersion: String
    public let latestRelease: QWReleaseInfo?
    public let updateAvailable: Bool
    public let isSkipped: Bool
    public let installOrigin: QWInstallOrigin
    public let schedule: QWUpdateCheckSchedule
    public let phase: QWUpdatePhase
    public let downloadedBytes: Int64?
    public let totalBytes: Int64?
    public let verifiedSHA256: String?
    public let downloadedPkgPath: String?
    public let signatureSummary: String?
    public let lastCheckedAt: Date?
    public let errorMessage: String?
    public let canAutoCheck: Bool
    public let canInstallPkg: Bool

    public init(
        currentVersion: String,
        latestRelease: QWReleaseInfo?,
        updateAvailable: Bool,
        isSkipped: Bool,
        installOrigin: QWInstallOrigin,
        schedule: QWUpdateCheckSchedule,
        phase: QWUpdatePhase,
        downloadedBytes: Int64? = nil,
        totalBytes: Int64? = nil,
        verifiedSHA256: String? = nil,
        downloadedPkgPath: String? = nil,
        signatureSummary: String? = nil,
        lastCheckedAt: Date? = nil,
        errorMessage: String? = nil,
        canAutoCheck: Bool = true,
        canInstallPkg: Bool = true
    ) {
        self.currentVersion = currentVersion
        self.latestRelease = latestRelease
        self.updateAvailable = updateAvailable
        self.isSkipped = isSkipped
        self.installOrigin = installOrigin
        self.schedule = schedule
        self.phase = phase
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.verifiedSHA256 = verifiedSHA256
        self.downloadedPkgPath = downloadedPkgPath
        self.signatureSummary = signatureSummary
        self.lastCheckedAt = lastCheckedAt
        self.errorMessage = errorMessage
        self.canAutoCheck = canAutoCheck
        self.canInstallPkg = canInstallPkg
    }
}

/// Structured updater failures with human-readable descriptions.
public enum QWUpdateError: Error, LocalizedError, Equatable, Sendable {
    case invalidURL(String)
    case disallowedHost(String)
    case networkFailure(String)
    case rateLimitedWithoutFallback
    case invalidResponse(String)
    case missingAsset(String)
    case missingChecksum(String)
    case checksumMismatch(expected: String, actual: String)
    case fileTooLarge(limitBytes: Int64)
    case missingVerifiedPackage
    case installerLaunchFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let raw):
            return "Invalid update URL: \(raw)"
        case .disallowedHost(let host):
            return "Blocked untrusted update download host: \(host)"
        case .networkFailure(let detail):
            return "Update check failed: \(detail)"
        case .rateLimitedWithoutFallback:
            return "GitHub API rate limit exceeded and redirect fallback failed."
        case .invalidResponse(let detail):
            return "Unexpected release feed response: \(detail)"
        case .missingAsset(let name):
            return "Release does not include required asset '\(name)'."
        case .missingChecksum(let name):
            return "SHA256SUMS.txt does not list a checksum for '\(name)'."
        case .checksumMismatch(let expected, let actual):
            return "SHA-256 verification failed (expected \(expected.prefix(12))…, got \(actual.prefix(12))…)."
        case .fileTooLarge(let limitBytes):
            return "Update installer exceeds maximum size cap (\(limitBytes / (1024 * 1024)) MiB)."
        case .missingVerifiedPackage:
            return "No SHA-256 verified installer package is staged locally."
        case .installerLaunchFailed(let detail):
            return "Could not launch macOS Installer: \(detail)"
        }
    }
}
