// QWUpdateServerTests.swift — Tests /v1/update* HTTP routes, loopback gating, and settings coherence.

import CryptoKit
import Foundation
import XCTest
@testable import QWCore
@testable import QWEngine
@testable import QWServer

private extension QWHttpServer {
    var boundPort: UInt16? {
        listener?.port?.rawValue
    }
}

private final class StubServerUpdateTransport: QWUpdateHTTPTransport, @unchecked Sendable {
    let payload = Data("server-pkg-installer-payload".utf8)
    let releaseJSON: Data
    let checksumsText: String

    init(tag: String = "v99.0.0") {
        let sum = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        self.checksumsText = "\(sum)  Qwegn-99.0.0-macos-arm64.pkg\n"
        let obj: [String: Any] = [
            "tag_name": tag,
            "name": "Qwegn \(tag)",
            "body": "### Highlights\n- Server update test release",
            "draft": false,
            "prerelease": false,
            "published_at": "2026-11-23T12:00:00Z",
            "html_url": "https://github.com/qwegnai/qwegn/releases/tag/\(tag)",
            "assets": [
                [
                    "name": "Qwegn-99.0.0-macos-arm64.pkg",
                    "browser_download_url": "https://github.com/qwegnai/qwegn/releases/download/\(tag)/Qwegn-99.0.0-macos-arm64.pkg",
                    "size": payload.count,
                ],
                [
                    "name": "SHA256SUMS.txt",
                    "browser_download_url": "https://github.com/qwegnai/qwegn/releases/download/\(tag)/SHA256SUMS.txt",
                    "size": checksumsText.utf8.count,
                ],
            ],
        ]
        self.releaseJSON = try! JSONSerialization.data(withJSONObject: obj)
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let url = request.url!
        if url.lastPathComponent == "SHA256SUMS.txt" {
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            return (Data(checksumsText.utf8), resp)
        }
        let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["ETag": "\"srv-etag\""])!
        return (releaseJSON, resp)
    }

    func resolveRedirectLocation(for url: URL, userAgent: String) async throws -> URL? { nil }

    func downloadToFile(
        for request: URLRequest,
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) async throws -> Int64 {
        try FileManager.default.createDirectory(at: destinationPartURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try payload.write(to: destinationPartURL, options: .atomic)
        onProgress(Int64(payload.count), Int64(payload.count))
        return Int64(payload.count)
    }
}

final class QWUpdateServerTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-srv-upd-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    override func tearDown() {
        QWEngineTelemetryRegistry.shared.resetToIdle()
        super.tearDown()
    }

    func testLoopbackUpdateRoutesDownloadActiveInferenceGuardAndSkip() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let cacheDir = dir.appendingPathComponent("cache", isDirectory: true)
        let transport = StubServerUpdateTransport()
        let launchedURLs = NSLock()
        nonisolated(unsafe) var launched: [URL] = []

        let service = QWServerUpdateService(
            checker: QWUpdateChecker(transport: transport),
            currentVersion: "0.32.2",
            stateURL: stateURL,
            cacheDirectory: cacheDir,
            installOrigin: .pkgReceipt,
            schedule: .onLaunchAndDaily,
            installerLauncher: { url in
                launchedURLs.withLock { launched.append(url) }
                return 0
            }
        )

        let cfg = QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test-model")
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(
            actor: actor, host: "127.0.0.1", port: 0, logger: logger,
            enableAutoUpdateChecks: false, updateService: service
        )
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.boundPort)
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        // 1. Check for updates via POST /v1/update/check
        var checkReq = URLRequest(url: baseURL.appendingPathComponent("v1/update/check"))
        checkReq.httpMethod = "POST"
        checkReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        checkReq.httpBody = Data("{\"force\":true}".utf8)
        let (checkData, checkResp) = try await URLSession.shared.data(for: checkReq)
        XCTAssertEqual((checkResp as? HTTPURLResponse)?.statusCode, 200)
        let checked = try decoder.decode(QWUpdateServerResponse.self, from: checkData)
        XCTAssertTrue(checked.snapshot.updateAvailable)
        XCTAssertEqual(checked.snapshot.phase, .available)
        XCTAssertTrue(checked.loopbackDesktopActionsAllowed)

        // 2. Skip and unskip version
        var skipReq = URLRequest(url: baseURL.appendingPathComponent("v1/update/skip"))
        skipReq.httpMethod = "POST"
        skipReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        skipReq.httpBody = Data("{\"version\":\"99.0.0\"}".utf8)
        let (skipData, skipResp) = try await URLSession.shared.data(for: skipReq)
        XCTAssertEqual((skipResp as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(try decoder.decode(QWUpdateServerResponse.self, from: skipData).snapshot.isSkipped)

        skipReq.httpBody = Data("{\"version\":null}".utf8)
        let (unskipData, _) = try await URLSession.shared.data(for: skipReq)
        XCTAssertFalse(try decoder.decode(QWUpdateServerResponse.self, from: unskipData).snapshot.isSkipped)

        // 3. Download & verify .pkg
        var dlReq = URLRequest(url: baseURL.appendingPathComponent("v1/update/download"))
        dlReq.httpMethod = "POST"
        let (_, dlResp) = try await URLSession.shared.data(for: dlReq)
        XCTAssertEqual((dlResp as? HTTPURLResponse)?.statusCode, 200)

        var ready = false
        for _ in 0..<40 {
            let (stData, _) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/update"))
            let st = try decoder.decode(QWUpdateServerResponse.self, from: stData)
            if st.snapshot.phase == .readyToInstall {
                ready = true
                XCTAssertNotNil(st.snapshot.verifiedSHA256)
                break
            }
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTAssertTrue(ready, "Expected downloaded package to verify and enter readyToInstall")

        // 4. Active inference blocks install (409) unless force=true
        QWEngineTelemetryRegistry.shared.beginDecode(promptTokens: 8)
        defer { QWEngineTelemetryRegistry.shared.resetToIdle() }

        var instReq = URLRequest(url: baseURL.appendingPathComponent("v1/update/install"))
        instReq.httpMethod = "POST"
        let (_, busyResp) = try await URLSession.shared.data(for: instReq)
        XCTAssertEqual((busyResp as? HTTPURLResponse)?.statusCode, 409)

        instReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        instReq.httpBody = Data("{\"force\":true}".utf8)
        let (_, forcedResp) = try await URLSession.shared.data(for: instReq)
        XCTAssertEqual((forcedResp as? HTTPURLResponse)?.statusCode, 200)
        let count = launchedURLs.withLock { launched.count }
        XCTAssertEqual(count, 1)
    }

    func testNonLoopbackBindRejectsDownloadAndInstall() async throws {
        let dir = try makeTempDir()
        let service = QWServerUpdateService(
            checker: QWUpdateChecker(transport: StubServerUpdateTransport()),
            stateURL: dir.appendingPathComponent("state.json"),
            cacheDirectory: dir.appendingPathComponent("cache", isDirectory: true)
        )
        let cfg = QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test-model")
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(
            actor: actor, host: "0.0.0.0", port: 0, logger: logger,
            apiKey: "test-secret", enableAutoUpdateChecks: false, updateService: service
        )
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.boundPort)
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!

        for endpoint in ["v1/update/download", "v1/update/install"] {
            var req = URLRequest(url: baseURL.appendingPathComponent(endpoint))
            req.httpMethod = "POST"
            req.setValue("Bearer test-secret", forHTTPHeaderField: "Authorization")
            let (_, resp) = try await URLSession.shared.data(for: req)
            XCTAssertEqual((resp as? HTTPURLResponse)?.statusCode, 403, "\(endpoint) must reject non-loopback server binds")
        }
    }

    func testLegacySettingsDecodeDefaultsUpdateCheckSchedule() throws {
        let legacyJSON = Data("{\"host\":\"127.0.0.1\",\"port\":8080}".utf8)
        let decoded = try JSONDecoder().decode(QWServerSettings.self, from: legacyJSON)
        XCTAssertEqual(decoded.updateCheckSchedule, QWUpdateCheckSchedule.onLaunchAndDaily.rawValue)
    }
}
