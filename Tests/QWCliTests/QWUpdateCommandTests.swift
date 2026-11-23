// QWUpdateCommandTests.swift — Tests `qwg update` CLI flags and runner flows.

import ArgumentParser
import CryptoKit
import Foundation
import Testing
@testable import qwg
@testable import QWCore

private final class StubCliUpdateTransport: QWUpdateHTTPTransport, @unchecked Sendable {
    let payload = Data("cli-pkg-installer-payload".utf8)
    let releaseJSON: Data
    let checksumsText: String

    init(tag: String = "v99.0.0") {
        let sum = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        self.checksumsText = "\(sum)  Qwegn-99.0.0-macos-arm64.pkg\n"
        let obj: [String: Any] = [
            "tag_name": tag,
            "name": "Qwegn \(tag)",
            "body": "- CLI update test notes",
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
        let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
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

private final class OutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ s: String) { lock.withLock { lines.append(s) } }
    func joined() -> String { lock.withLock { lines.joined(separator: "\n") } }
}

@Test("qwg update parses all documented flags")
func updateCommandParsesFlags() throws {
    let cmd = try Update.parse(["--check", "--download-only", "--notes", "--skip", "--unskip", "--json", "--yes", "--force"])
    #expect(cmd.check)
    #expect(cmd.downloadOnly)
    #expect(cmd.notes)
    #expect(cmd.skip)
    #expect(cmd.unskip)
    #expect(cmd.json)
    #expect(cmd.yes)
    #expect(cmd.force)
}

@Test("qwg update runner handles --check --json, --skip, portable notice, and verified pkg install")
func updateRunnerEndToEnd() async throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwg-cli-upd-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let stateURL = dir.appendingPathComponent("state.json")
    let cacheDir = dir.appendingPathComponent("cache", isDirectory: true)
    let checker = QWUpdateChecker(transport: StubCliUpdateTransport())

    let jsonOut = OutputCollector()
    try await QwgUpdateRunner.run(
        options: UpdateOptions(check: true, json: true),
        checker: checker, stateURL: stateURL, cacheDir: cacheDir,
        installOrigin: .pkgReceipt, currentVersion: "0.32.2",
        runningEnginePIDProvider: { nil },
        stdoutWriter: { jsonOut.append($0) }
    )
    let dec = JSONDecoder()
    dec.dateDecodingStrategy = .iso8601
    let snap = try dec.decode(QWUpdateStatusSnapshot.self, from: Data(jsonOut.joined().utf8))
    #expect(snap.updateAvailable)
    #expect(snap.latestRelease?.version.description == "99.0.0")

    let portableOut = OutputCollector()
    try await QwgUpdateRunner.run(
        options: UpdateOptions(),
        checker: checker, stateURL: stateURL, cacheDir: cacheDir,
        installOrigin: .portableBundle, currentVersion: "0.32.2",
        runningEnginePIDProvider: { nil },
        stdoutWriter: { portableOut.append($0) }
    )
    #expect(portableOut.joined().contains("no ai.qwegn.pkg receipt"))

    // Active server blocks launch without --yes
    let errOut = OutputCollector()
    var blockedByActiveServer = false
    do {
        try await QwgUpdateRunner.run(
            options: UpdateOptions(),
            checker: checker, stateURL: stateURL, cacheDir: cacheDir,
            installOrigin: .pkgReceipt, currentVersion: "0.32.2",
            runningEnginePIDProvider: { 4242 },
            stdoutWriter: { _ in },
            stderrWriter: { errOut.append($0) },
            installerLauncher: { _ in 0 }
        )
    } catch {
        blockedByActiveServer = true
    }
    #expect(blockedByActiveServer)
    #expect(errOut.joined().contains("PID 4242"))

    // With --yes, launches verified installer
    let launchedPkg = OutputCollector()
    try await QwgUpdateRunner.run(
        options: UpdateOptions(yes: true),
        checker: checker, stateURL: stateURL, cacheDir: cacheDir,
        installOrigin: .pkgReceipt, currentVersion: "0.32.2",
        runningEnginePIDProvider: { 4242 },
        stdoutWriter: { _ in },
        installerLauncher: { url in
            launchedPkg.append(url.lastPathComponent)
            return 0
        }
    )
    #expect(launchedPkg.joined() == "Qwegn-99.0.0-macos-arm64.pkg")
}
