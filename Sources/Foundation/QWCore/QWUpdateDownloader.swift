// QWUpdateDownloader.swift — Streaming .pkg downloader with CryptoKit SHA-256 verification.

import CryptoKit
import Foundation

/// Verified installer artifact cached locally after bit-exact SHA-256 verification.
public struct QWVerifiedUpdatePackage: Sendable, Equatable {
    public let pkgURL: URL
    public let version: QWSemVer
    public let sha256Hex: String
    public let sizeBytes: Int64
    public let signatureSummary: String?

    public init(
        pkgURL: URL,
        version: QWSemVer,
        sha256Hex: String,
        sizeBytes: Int64,
        signatureSummary: String? = nil
    ) {
        self.pkgURL = pkgURL
        self.version = version
        self.sha256Hex = sha256Hex
        self.sizeBytes = sizeBytes
        self.signatureSummary = signatureSummary
    }
}

/// Downloads `.pkg` updates into `~/Library/Caches/Qwegn/updates`, verifies `SHA256SUMS.txt`, and prunes old installers.
public struct QWUpdateDownloader: Sendable {
    public static let maxInstallerBytes: Int64 = 512 * 1024 * 1024

    public let transport: any QWUpdateHTTPTransport
    public let extraAllowedHosts: Set<String>
    public let allowLoopbackHTTP: Bool

    public init(
        transport: any QWUpdateHTTPTransport = QWURLSessionUpdateTransport.shared,
        extraAllowedHosts: Set<String> = [],
        allowLoopbackHTTP: Bool = ProcessInfo.processInfo.environment["QWEGN_UPDATE_API_URL"] != nil
    ) {
        self.transport = transport
        var hosts = extraAllowedHosts
        if let raw = ProcessInfo.processInfo.environment["QWEGN_UPDATE_API_URL"],
           let url = URL(string: raw),
           let host = url.host?.lowercased() {
            hosts.insert(host)
        }
        self.extraAllowedHosts = hosts
        self.allowLoopbackHTTP = allowLoopbackHTTP
    }

    public static func parseSHA256Sums(_ text: String) -> [String: String] {
        var out: [String: String] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }
            guard let firstSpace = trimmed.firstIndex(where: \.isWhitespace) else { continue }
            let hash = String(trimmed[..<firstSpace]).lowercased()
            guard hash.count == 64, hash.unicodeScalars.allSatisfy(isLowerHexScalar) else { continue }
            var rest = String(trimmed[firstSpace...]).trimmingCharacters(in: .whitespaces)
            if rest.hasPrefix("*") {
                rest.removeFirst()
            }
            let fileName = URL(fileURLWithPath: rest).lastPathComponent
            guard !fileName.isEmpty else { continue }
            out[fileName] = hash
        }
        return out
    }

    public static func sha256Hex(ofFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 65_536) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func verifyCachedPackage(at url: URL, expectedSHA256: String) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        let actual = try sha256Hex(ofFileAt: url)
        return actual == expectedSHA256.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    public func downloadAndVerify(
        release: QWReleaseInfo,
        currentVersion: String = QWVersion.current,
        cacheDirectory: URL = QWPaths.updatesCacheDirectoryURL,
        stateURL: URL = QWPaths.updateStateFileURL,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void = { _, _ in }
    ) async throws -> QWVerifiedUpdatePackage {
        guard let pkgAsset = release.pkgAsset else {
            throw QWUpdateError.missingAsset("Qwegn-\(release.version.description)-macos-arm64.pkg")
        }
        guard let checksumsAsset = release.checksumsAsset else {
            throw QWUpdateError.missingAsset("SHA256SUMS.txt")
        }
        guard QWUpdateChecker.isAllowedUpdateURL(
            pkgAsset.downloadURL,
            extraAllowedHosts: extraAllowedHosts,
            allowLoopbackHTTP: allowLoopbackHTTP
        ) else {
            throw QWUpdateError.disallowedHost(pkgAsset.downloadURL.host ?? pkgAsset.downloadURL.absoluteString)
        }
        guard QWUpdateChecker.isAllowedUpdateURL(
            checksumsAsset.downloadURL,
            extraAllowedHosts: extraAllowedHosts,
            allowLoopbackHTTP: allowLoopbackHTTP
        ) else {
            throw QWUpdateError.disallowedHost(checksumsAsset.downloadURL.host ?? checksumsAsset.downloadURL.absoluteString)
        }

        let userAgent = "Qwegn/\(currentVersion)"
        var sumReq = URLRequest(url: checksumsAsset.downloadURL, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        sumReq.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        let (sumData, sumResp) = try await transport.data(for: sumReq)
        if let finalURL = sumResp.url,
           !QWUpdateChecker.isAllowedUpdateURL(
               finalURL,
               extraAllowedHosts: extraAllowedHosts,
               allowLoopbackHTTP: allowLoopbackHTTP
           ) {
            throw QWUpdateError.disallowedHost(finalURL.host ?? finalURL.absoluteString)
        }
        guard (200...299).contains(sumResp.statusCode) else {
            throw QWUpdateError.networkFailure("SHA256SUMS.txt HTTP \(sumResp.statusCode)")
        }
        let sumText = String(data: sumData, encoding: .utf8) ?? ""
        let sums = Self.parseSHA256Sums(sumText)
        guard let expectedHash = sums[pkgAsset.name] else {
            throw QWUpdateError.missingChecksum(pkgAsset.name)
        }

        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let finalPkgURL = cacheDirectory.appendingPathComponent(pkgAsset.name, isDirectory: false)
        let partURL = cacheDirectory.appendingPathComponent("\(pkgAsset.name).part", isDirectory: false)

        if FileManager.default.fileExists(atPath: finalPkgURL.path),
           let existingHash = try? Self.sha256Hex(ofFileAt: finalPkgURL),
           existingHash == expectedHash {
            let size = (try? FileManager.default.attributesOfItem(atPath: finalPkgURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            onProgress(size, size)
            try recordVerifiedState(pkgURL: finalPkgURL, version: release.version, sha256: existingHash, stateURL: stateURL)
            let sig = Self.inspectPkgSignatureSummary(at: finalPkgURL)
            return QWVerifiedUpdatePackage(pkgURL: finalPkgURL, version: release.version, sha256Hex: existingHash, sizeBytes: size, signatureSummary: sig)
        }

        try? FileManager.default.removeItem(at: partURL)
        var pkgReq = URLRequest(url: pkgAsset.downloadURL, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 600)
        pkgReq.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let writtenBytes: Int64
        do {
            writtenBytes = try await transport.downloadToFile(
                for: pkgReq,
                destinationPartURL: partURL,
                maxBytes: Self.maxInstallerBytes,
                allowedHosts: extraAllowedHosts,
                allowLoopbackHTTP: allowLoopbackHTTP,
                onProgress: onProgress
            )
        } catch {
            try? FileManager.default.removeItem(at: partURL)
            throw error
        }

        let actualHash: String
        do {
            actualHash = try Self.sha256Hex(ofFileAt: partURL)
        } catch {
            try? FileManager.default.removeItem(at: partURL)
            throw error
        }
        guard actualHash == expectedHash else {
            try? FileManager.default.removeItem(at: partURL)
            throw QWUpdateError.checksumMismatch(expected: expectedHash, actual: actualHash)
        }

        if FileManager.default.fileExists(atPath: finalPkgURL.path) {
            try FileManager.default.removeItem(at: finalPkgURL)
        }
        try FileManager.default.moveItem(at: partURL, to: finalPkgURL)
        pruneOldInstallers(in: cacheDirectory, keeping: finalPkgURL.lastPathComponent)
        try recordVerifiedState(pkgURL: finalPkgURL, version: release.version, sha256: actualHash, stateURL: stateURL)
        let sig = Self.inspectPkgSignatureSummary(at: finalPkgURL)
        return QWVerifiedUpdatePackage(pkgURL: finalPkgURL, version: release.version, sha256Hex: actualHash, sizeBytes: writtenBytes, signatureSummary: sig)
    }

    public static func inspectPkgSignatureSummary(at pkgURL: URL) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/pkgutil")
        proc.arguments = ["--check-signature", pkgURL.path]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
            proc.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8) ?? ""
            if proc.terminationStatus == 0 {
                if let devLine = text.components(separatedBy: .newlines).first(where: { $0.contains("Developer ID Installer") }) {
                    return "Signed: \(devLine.trimmingCharacters(in: .whitespaces))"
                }
                return "Signed macOS installer package"
            }
            return "Ad-hoc / unsigned package (verify SHA-256 above)"
        } catch {
            return nil
        }
    }

    private func recordVerifiedState(
        pkgURL: URL,
        version: QWSemVer,
        sha256: String,
        stateURL: URL
    ) throws {
        var state = QWUpdateStateStore.load(from: stateURL)
        state.verifiedPkgPath = pkgURL.path
        state.verifiedPkgVersion = version.description
        state.verifiedPkgSHA256 = sha256
        state.lastError = nil
        try QWUpdateStateStore.save(state, to: stateURL)
    }

    private func pruneOldInstallers(in cacheDirectory: URL, keeping fileName: String) {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: nil) else { return }
        for item in items {
            let name = item.lastPathComponent
            if name != fileName && (name.hasSuffix(".pkg") || name.hasSuffix(".part")) {
                try? fm.removeItem(at: item)
            }
        }
    }

    private static func isLowerHexScalar(_ s: UnicodeScalar) -> Bool {
        (s.value >= 48 && s.value <= 57) || (s.value >= 97 && s.value <= 102)
    }
}
