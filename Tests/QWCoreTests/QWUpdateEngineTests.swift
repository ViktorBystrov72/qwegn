// QWUpdateEngineTests.swift — Hermetic tests for SemVer, release checker, and SHA-256 verification.

import CryptoKit
import Foundation
import XCTest
@testable import QWCore

private final class MockUpdateTransport: QWUpdateHTTPTransport, @unchecked Sendable {
    struct StubResponse: Sendable {
        let statusCode: Int
        let headers: [String: String]
        let data: Data
        let finalURL: URL?
    }

    private let lock = NSLock()
    var apiResponder: (@Sendable (URLRequest) -> StubResponse)?
    var redirectLocation: URL?
    var downloadPayload: Data = Data("fake-pkg-bytes".utf8)
    var checksumsText: String = ""
    var checksumsStatusCode: Int = 200
    var checksumsURLOverride: URL?
    var downloadStatusCode: Int = 200
    var recordedRequests: [URLRequest] = []
    var downloadCallCount: Int = 0

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        lock.withLock { recordedRequests.append(request) }
        guard let url = request.url else {
            throw QWUpdateError.invalidURL("nil")
        }
        if url.lastPathComponent == "SHA256SUMS.txt" {
            let final = checksumsURLOverride ?? url
            let resp = HTTPURLResponse(
                url: final, statusCode: checksumsStatusCode, httpVersion: "HTTP/1.1", headerFields: nil
            )!
            return (Data(checksumsText.utf8), resp)
        }
        guard let responder = apiResponder else {
            throw QWUpdateError.networkFailure("No stub responder configured")
        }
        let stub = responder(request)
        let resp = HTTPURLResponse(
            url: stub.finalURL ?? url, statusCode: stub.statusCode, httpVersion: "HTTP/1.1", headerFields: stub.headers
        )!
        return (stub.data, resp)
    }

    func resolveRedirectLocation(for url: URL, userAgent: String) async throws -> URL? {
        redirectLocation
    }

    func downloadToFile(
        for request: URLRequest,
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) async throws -> Int64 {
        lock.withLock { downloadCallCount += 1 }
        guard downloadStatusCode == 200 else {
            throw QWUpdateError.networkFailure("Download HTTP \(downloadStatusCode)")
        }
        let payload = lock.withLock { downloadPayload }
        guard Int64(payload.count) <= maxBytes else {
            throw QWUpdateError.fileTooLarge(limitBytes: maxBytes)
        }
        try FileManager.default.createDirectory(
            at: destinationPartURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let mid = max(1, payload.count / 2)
        onProgress(Int64(mid), Int64(payload.count))
        try payload.write(to: destinationPartURL, options: .atomic)
        onProgress(Int64(payload.count), Int64(payload.count))
        return Int64(payload.count)
    }
}

final class QWUpdateEngineTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-update-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func makeGitHubReleaseJSON(
        tag: String = "v0.33.0",
        draft: Bool = false,
        prerelease: Bool = false,
        pkgURL: String = "https://github.com/qwegnai/qwegn/releases/download/v0.33.0/Qwegn-0.33.0-macos-arm64.pkg",
        sumsURL: String = "https://github.com/qwegnai/qwegn/releases/download/v0.33.0/SHA256SUMS.txt"
    ) -> Data {
        let obj: [String: Any] = [
            "tag_name": tag,
            "name": "Qwegn \(tag)",
            "body": "## Highlights\n- Native Check for Updates\n- Faster NAX prefill",
            "draft": draft,
            "prerelease": prerelease,
            "published_at": "2026-11-23T12:00:00Z",
            "html_url": "https://github.com/qwegnai/qwegn/releases/tag/\(tag)",
            "assets": [
                ["name": "Qwegn-0.33.0-macos-arm64.pkg", "browser_download_url": pkgURL, "size": 42_800_000],
                ["name": "qwegn-\(tag)-macos-arm64.tar.gz", "browser_download_url": "https://github.com/qwegnai/qwegn/releases/download/\(tag)/qwegn-\(tag)-macos-arm64.tar.gz", "size": 40_000_000],
                ["name": "SHA256SUMS.txt", "browser_download_url": sumsURL, "size": 256],
            ],
        ]
        return try! JSONSerialization.data(withJSONObject: obj)
    }

    func testSemVerParsingAndOrdering() {
        let v1 = QWSemVer("0.32.2")
        let v2 = QWSemVer("v0.33.0")
        let vPre = QWSemVer("0.33.0-rc.1")
        let vPre2 = QWSemVer("0.33.0-rc.2")
        let vBuild = QWSemVer("v0.33.0+build.42")

        XCTAssertNotNil(v1)
        XCTAssertNotNil(v2)
        XCTAssertNotNil(vPre)
        XCTAssertTrue(v1! < vPre!)
        XCTAssertTrue(vPre! < vPre2!)
        XCTAssertTrue(vPre2! < v2!)
        XCTAssertEqual(v2, vBuild)
        XCTAssertEqual(v2?.description, "0.33.0")
        XCTAssertTrue(vPre?.isPrerelease == true)
        XCTAssertFalse(v2?.isPrerelease == true)

        for bad in ["", "0.32", "v1.2.3.4", "01.2.3", "1.2.3-", "abc"] {
            XCTAssertNil(QWSemVer(bad), "Expected nil for invalid semver '\(bad)'")
        }
    }

    func testSchedulePolicyAndEnvironmentKillSwitch() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let fresh = QWUpdateState(lastCheckedAt: nil)
        let fiveMinAgo = QWUpdateState(lastCheckedAt: now.addingTimeInterval(-300))
        let twentyMinAgo = QWUpdateState(lastCheckedAt: now.addingTimeInterval(-1200))
        let twelveHoursAgo = QWUpdateState(lastCheckedAt: now.addingTimeInterval(-43_200))
        let twentyFiveHoursAgo = QWUpdateState(lastCheckedAt: now.addingTimeInterval(-90_000))

        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunchAndDaily, trigger: .launch, state: fresh, now: now,
            env: ["QWEGN_NO_UPDATE_CHECK": "1"]
        ))
        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .manual, trigger: .launch, state: fresh, now: now, env: [:]
        ))
        XCTAssertTrue(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .manual, trigger: .manual, state: fiveMinAgo, now: now, env: [:]
        ))

        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunchAndDaily, trigger: .launch, state: fiveMinAgo, now: now, env: [:]
        ))
        XCTAssertTrue(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunchAndDaily, trigger: .launch, state: twentyMinAgo, now: now, env: [:]
        ))
        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunchAndDaily, trigger: .dailyTimer, state: twelveHoursAgo, now: now, env: [:]
        ))
        XCTAssertTrue(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunchAndDaily, trigger: .dailyTimer, state: twentyFiveHoursAgo, now: now, env: [:]
        ))

        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .daily, trigger: .launch, state: twelveHoursAgo, now: now, env: [:]
        ))
        XCTAssertTrue(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .daily, trigger: .launch, state: twentyFiveHoursAgo, now: now, env: [:]
        ))
        XCTAssertFalse(QWUpdateChecker.shouldPerformAutomaticCheck(
            schedule: .onLaunch, trigger: .dailyTimer, state: twentyFiveHoursAgo, now: now, env: [:]
        ))
    }

    func testCheckerETag304RedirectFallbackAndStatePermissions() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let transport = MockUpdateTransport()
        let jsonBody = Self.makeGitHubReleaseJSON(tag: "v0.33.0")

        transport.apiResponder = { req in
            if req.value(forHTTPHeaderField: "If-None-Match") == "\"etag-1\"" {
                return .init(statusCode: 304, headers: [:], data: Data(), finalURL: nil)
            }
            return .init(statusCode: 200, headers: ["ETag": "\"etag-1\""], data: jsonBody, finalURL: nil)
        }

        let checker = QWUpdateChecker(transport: transport)
        let (state1, rel1) = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL)
        XCTAssertEqual(state1.etag, "\"etag-1\"")
        XCTAssertEqual(rel1?.version.description, "0.33.0")
        XCTAssertTrue(QWUpdateChecker.isUpdateAvailable(currentVersion: "0.32.2", latest: rel1))

        let attrs = try FileManager.default.attributesOfItem(atPath: stateURL.path)
        let posix = (attrs[.posixPermissions] as? NSNumber)?.intValue
        XCTAssertEqual(posix, 0o600)

        let (state2, rel2) = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL)
        XCTAssertEqual(rel2, rel1)
        XCTAssertEqual(state2.etag, "\"etag-1\"")

        // Simulate 429 rate limit falling back to releases/latest redirect tag.
        transport.apiResponder = { _ in
            .init(statusCode: 429, headers: [:], data: Data(), finalURL: nil)
        }
        transport.redirectLocation = URL(string: "https://github.com/qwegnai/qwegn/releases/tag/v0.34.0")
        let (state3, rel3) = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL, force: true)
        XCTAssertEqual(rel3?.version.description, "0.34.0")
        XCTAssertEqual(rel3?.pkgAsset?.name, "Qwegn-0.34.0-macos-arm64.pkg")
        XCTAssertEqual(state3.cachedRelease?.version.description, "0.34.0")
    }

    func testCheckerRejectsDraftPrereleaseAndDisallowedHosts() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let transport = MockUpdateTransport()
        let checker = QWUpdateChecker(transport: transport)

        transport.apiResponder = { _ in
            .init(statusCode: 200, headers: [:], data: Self.makeGitHubReleaseJSON(tag: "v0.33.0", prerelease: true), finalURL: nil)
        }
        let (_, preRel) = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL)
        XCTAssertNil(preRel)

        transport.apiResponder = { _ in
            .init(
                statusCode: 200, headers: [:],
                data: Self.makeGitHubReleaseJSON(pkgURL: "https://evil.example.com/bad.pkg"),
                finalURL: nil
            )
        }
        do {
            _ = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL, force: true)
            XCTFail("Expected disallowedHost error")
        } catch let err as QWUpdateError {
            XCTAssertEqual(err, .disallowedHost("evil.example.com"))
        }
    }

    func testDownloaderVerifiesSHA256ReusesCacheAndRejectsCorruption() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let cacheDir = dir.appendingPathComponent("cache", isDirectory: true)
        let stalePkg = cacheDir.appendingPathComponent("Qwegn-0.31.0-macos-arm64.pkg")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        try Data("old-installer".utf8).write(to: stalePkg)

        let payload = Data("valid-pkg-binary-content-for-qwegn".utf8)
        let validHash = sha256Hex(payload)
        let transport = MockUpdateTransport()
        transport.downloadPayload = payload
        transport.checksumsText = "\(validHash)  Qwegn-0.33.0-macos-arm64.pkg\n"
        let releaseData = Self.makeGitHubReleaseJSON(tag: "v0.33.0")
        transport.apiResponder = { _ in
            .init(statusCode: 200, headers: [:], data: releaseData, finalURL: nil)
        }

        let checker = QWUpdateChecker(transport: transport)
        let (_, relOpt) = try await checker.check(currentVersion: "0.32.2", stateURL: stateURL)
        let release = try XCTUnwrap(relOpt)
        let downloader = QWUpdateDownloader(transport: transport)

        let pkg1 = try await downloader.downloadAndVerify(
            release: release, cacheDirectory: cacheDir, stateURL: stateURL
        )
        XCTAssertEqual(pkg1.sha256Hex, validHash)
        XCTAssertEqual(transport.downloadCallCount, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: stalePkg.path), "Old cached .pkg must be pruned")

        // Second invocation reuses verified cache without re-downloading.
        let pkg2 = try await downloader.downloadAndVerify(
            release: release, cacheDirectory: cacheDir, stateURL: stateURL
        )
        XCTAssertEqual(pkg2.pkgURL, pkg1.pkgURL)
        XCTAssertEqual(transport.downloadCallCount, 1)

        // Checksum mismatch removes corrupted .part and throws.
        try? FileManager.default.removeItem(at: pkg1.pkgURL)
        transport.checksumsText = "\(String(repeating: "0", count: 64))  Qwegn-0.33.0-macos-arm64.pkg\n"
        do {
            _ = try await downloader.downloadAndVerify(
                release: release, cacheDirectory: cacheDir, stateURL: stateURL
            )
            XCTFail("Expected checksumMismatch")
        } catch let err as QWUpdateError {
            guard case .checksumMismatch = err else {
                XCTFail("Unexpected error: \(err)")
                return
            }
            let partURL = cacheDir.appendingPathComponent("Qwegn-0.33.0-macos-arm64.pkg.part")
            XCTAssertFalse(FileManager.default.fileExists(atPath: partURL.path))
        }
    }

    func testInstallOriginDetection() {
        XCTAssertEqual(
            QWInstallOrigin.detect(executablePath: "/Users/dev/repo/.build/release/qwg", receiptChecker: { true }),
            .sourceCheckout
        )
        XCTAssertEqual(
            QWInstallOrigin.detect(executablePath: "/Applications/Qwegn.app/Contents/MacOS/qwg-menu", receiptChecker: { true }),
            .pkgReceipt
        )
        XCTAssertEqual(
            QWInstallOrigin.detect(executablePath: "/Applications/Qwegn.app/Contents/MacOS/qwg-menu", receiptChecker: { false }),
            .portableBundle
        )
        XCTAssertEqual(
            QWInstallOrigin.detect(executablePath: "/Users/me/Downloads/qwegn/qwg", receiptChecker: { true }),
            .portableBundle
        )
    }
}
