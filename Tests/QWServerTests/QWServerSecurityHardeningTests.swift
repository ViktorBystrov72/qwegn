// QWServerSecurityHardeningTests.swift — Regression suite for QWServer Wave 1 security hardening.

import CoreGraphics
import Foundation
import ImageIO
import JavaScriptCore
import Network
import QWCore
import QWEngine
import QWMM
import QWVision
import UniformTypeIdentifiers
import XCTest
@testable import QWServer

private final class MockSecVisionBackend: QWVisionBackend, @unchecked Sendable {
    let outDim: Int
    var isLoaded: Bool = false

    init(outDim: Int = 16) {
        self.outDim = outDim
    }

    func prepareForPrompt(hasImages: Bool) throws {
        isLoaded = hasImages
    }

    func encode(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) async throws -> [Float] {
        isLoaded = true
        return [Float](repeating: 0.25, count: grid.nToken * outDim)
    }
}

final class QWServerSecurityHardeningTests: XCTestCase {
    private static func repoRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func startMockActorAndServer(
        allowedWorkspaceRoots: [String] = [],
        maxConcurrentConnections: Int = 64,
        requestReadTimeoutSeconds: TimeInterval = 30.0,
        finderRevealRunner: (@Sendable ([String]) async throws -> Int32)? = nil
    ) async throws -> (actor: QWEngineLifecycleActor, server: QWHttpServer, port: UInt16) {
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/test-model.qwp",
            modelName: "test-model",
            host: "127.0.0.1",
            port: 0
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: config, logger: logger)
        let server = QWHttpServer(
            actor: actor,
            host: "127.0.0.1",
            port: 0,
            logger: logger,
            allowedWorkspaceRoots: allowedWorkspaceRoots,
            maxConcurrentConnections: maxConcurrentConnections,
            requestReadTimeoutSeconds: requestReadTimeoutSeconds,
            finderRevealRunner: finderRevealRunner
        )
        try server.start()
        for _ in 0..<40 {
            if let port = server.listener?.port?.rawValue, port > 0 {
                return (actor, server, port)
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let port = try XCTUnwrap(server.listener?.port?.rawValue)
        return (actor, server, port)
    }

    private struct RawHTTPResponse: Sendable {
        let statusCode: Int
        let headers: String
        let body: Data
    }

    private final class RawHTTPRunner: @unchecked Sendable {
        private let conn: NWConnection
        private let queue = DispatchQueue(label: "qwegine.sec-test.raw-http")
        private let lock = NSLock()
        private var finished = false
        private var acc = Data()
        private var continuation: CheckedContinuation<RawHTTPResponse, Error>?

        init(port: UInt16) {
            self.conn = NWConnection(
                to: .hostPort(host: .ipv4(IPv4Address("127.0.0.1")!), port: NWEndpoint.Port(rawValue: port)!),
                using: .tcp
            )
        }

        func run(rawRequest: Data, timeoutSeconds: TimeInterval) async throws -> RawHTTPResponse {
            try await withCheckedThrowingContinuation { cont in
                lock.withLock { self.continuation = cont }
                queue.asyncAfter(deadline: .now() + timeoutSeconds) { [weak self] in
                    self?.complete(.failure(NSError(
                        domain: "sec-test", code: 1, userInfo: [NSLocalizedDescriptionKey: "timeout"]
                    )))
                }
                conn.stateUpdateHandler = { [weak self] state in
                    guard let self else { return }
                    switch state {
                    case .ready:
                        self.conn.send(content: rawRequest, completion: .contentProcessed { [weak self] err in
                            guard let self else { return }
                            if let err {
                                self.complete(.failure(err))
                                return
                            }
                            self.readMore()
                        })
                    case .failed(let err):
                        self.complete(.failure(err))
                    default:
                        break
                    }
                }
                conn.start(queue: queue)
            }
        }

        private func readMore() {
            conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, err in
                guard let self else { return }
                let outcome: Result<RawHTTPResponse, Error>? = self.lock.withLock {
                    if let data, !data.isEmpty {
                        self.acc.append(data)
                    }
                    if let sep = self.acc.range(of: Data("\r\n\r\n".utf8)) {
                        let headerData = Data(self.acc[..<sep.lowerBound])
                        let headerText = String(data: headerData, encoding: .utf8) ?? ""
                        let firstLine = headerText.components(separatedBy: "\r\n").first ?? ""
                        let code = firstLine.split(separator: " ").dropFirst().first.flatMap { Int($0) } ?? 0
                        let contentLen = headerText
                            .components(separatedBy: "\r\n")
                            .first(where: { $0.lowercased().hasPrefix("content-length:") })
                            .flatMap { $0.split(separator: ":", maxSplits: 1).last }
                            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                        let body = Data(self.acc[sep.upperBound...])
                        if let contentLen, body.count >= contentLen {
                            return .success(RawHTTPResponse(statusCode: code, headers: headerText, body: Data(body.prefix(contentLen))))
                        }
                        if isComplete || err != nil {
                            return .success(RawHTTPResponse(statusCode: code, headers: headerText, body: body))
                        }
                    } else if isComplete || err != nil {
                        return .failure(err ?? NSError(domain: "sec-test", code: 2))
                    }
                    return nil
                }
                if let outcome {
                    self.complete(outcome)
                    return
                }
                self.readMore()
            }
        }

        private func complete(_ result: Result<RawHTTPResponse, Error>) {
            let cont = lock.withLock { () -> CheckedContinuation<RawHTTPResponse, Error>? in
                guard !finished else { return nil }
                finished = true
                let c = continuation
                continuation = nil
                return c
            }
            guard let cont else { return }
            conn.stateUpdateHandler = nil
            conn.cancel()
            cont.resume(with: result)
        }
    }

    private func sendRawHTTP(
        port: UInt16,
        rawRequest: Data,
        timeoutSeconds: TimeInterval = 5.0
    ) async throws -> RawHTTPResponse {
        try await RawHTTPRunner(port: port).run(rawRequest: rawRequest, timeoutSeconds: timeoutSeconds)
    }

    private func makeSolidGrayscalePNGData(width: Int, height: Int) throws -> Data {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixels = [UInt8](repeating: 128, count: width * height)
        let ctx = try XCTUnwrap(CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ))
        let cg = try XCTUnwrap(ctx.makeImage())
        let out = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(out as CFMutableData, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return out as Data
    }

    func testStrictLoopbackHostBlocksPrefixTrick() {
        XCTAssertTrue(QWHttpSecurity.isLoopbackHost("127.0.0.1"))
        XCTAssertTrue(QWHttpSecurity.isLoopbackHost("127.1.2.3"))
        XCTAssertTrue(QWHttpSecurity.isLoopbackHost("localhost"))
        XCTAssertTrue(QWHttpSecurity.isLoopbackHost("::1"))
        XCTAssertTrue(QWHttpSecurity.isLoopbackHost("[::1]"))

        XCTAssertFalse(QWHttpSecurity.isLoopbackHost("127.evil.com"))
        XCTAssertFalse(QWHttpSecurity.isLoopbackHost("127.0.0.1.evil.com"))
        XCTAssertFalse(QWHttpSecurity.isLoopbackHost("localhost.evil.com"))
        XCTAssertFalse(QWHttpSecurity.isLoopbackHost("128.0.0.1"))
        XCTAssertFalse(QWHttpSecurity.isLoopbackHost("127.0.0.256"))
        XCTAssertFalse(QWHttpSecurity.isLoopbackHost(""))
        XCTAssertNil(QWHttpSecurity.resolvedAllowedOrigin("http://127.evil.com:3000", allowlist: []))
    }

    func testMutationOriginAndContentTypePolicyEnforcedOverHTTP() async throws {
        XCTAssertTrue(QWHttpSecurity.isValidMutationOrigin(
            headers: ["content-type": "application/json; charset=utf-8"],
            allowlist: [],
            requireJSONContentType: true
        ))
        XCTAssertFalse(QWHttpSecurity.isValidMutationOrigin(
            headers: ["origin": "https://evil.example", "content-type": "application/json"],
            allowlist: [],
            requireJSONContentType: true
        ))
        XCTAssertFalse(QWHttpSecurity.isValidMutationOrigin(
            headers: ["sec-fetch-site": "cross-site", "content-type": "application/json"],
            allowlist: [],
            requireJSONContentType: true
        ))
        XCTAssertFalse(QWHttpSecurity.isValidMutationOrigin(
            headers: ["content-type": "text/plain"],
            allowlist: [],
            requireJSONContentType: false
        ))

        let (_, server, port) = try await startMockActorAndServer()
        defer { server.stop() }
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let session = URLSession.shared

        var badOriginSettings = URLRequest(url: baseURL.appendingPathComponent("v1/settings"))
        badOriginSettings.httpMethod = "POST"
        badOriginSettings.setValue("https://evil.example", forHTTPHeaderField: "Origin")
        badOriginSettings.setValue("application/json", forHTTPHeaderField: "Content-Type")
        badOriginSettings.httpBody = Data("{}".utf8)
        let (_, r1) = try await session.data(for: badOriginSettings)
        XCTAssertEqual((r1 as? HTTPURLResponse)?.statusCode, 403)

        var badTypeSettings = URLRequest(url: baseURL.appendingPathComponent("v1/settings"))
        badTypeSettings.httpMethod = "POST"
        badTypeSettings.setValue("text/plain", forHTTPHeaderField: "Content-Type")
        badTypeSettings.httpBody = Data("{}".utf8)
        let (_, r2) = try await session.data(for: badTypeSettings)
        XCTAssertEqual((r2 as? HTTPURLResponse)?.statusCode, 415)

        for route in ["v1/models/unload", "v1/engine/restart", "v1/memory/unmount", "v1/logs/reveal", "v1/settings/reveal"] {
            var crossReq = URLRequest(url: baseURL.appendingPathComponent(route))
            crossReq.httpMethod = "POST"
            crossReq.setValue("https://evil.example", forHTTPHeaderField: "Origin")
            crossReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let (_, resp) = try await session.data(for: crossReq)
            XCTAssertEqual((resp as? HTTPURLResponse)?.statusCode, 403, "Expected 403 for \(route)")

            var formReq = URLRequest(url: baseURL.appendingPathComponent(route))
            formReq.httpMethod = "POST"
            formReq.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
            let (_, formResp) = try await session.data(for: formReq)
            XCTAssertEqual((formResp as? HTTPURLResponse)?.statusCode, 415, "Expected 415 for \(route)")
        }

        var secFetchReq = URLRequest(url: baseURL.appendingPathComponent("v1/models/unload"))
        secFetchReq.httpMethod = "POST"
        secFetchReq.setValue("cross-site", forHTTPHeaderField: "Sec-Fetch-Site")
        secFetchReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (_, r3) = try await session.data(for: secFetchReq)
        XCTAssertEqual((r3 as? HTTPURLResponse)?.statusCode, 403)

        var okUnload = URLRequest(url: baseURL.appendingPathComponent("v1/models/unload"))
        okUnload.httpMethod = "POST"
        okUnload.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (_, r4) = try await session.data(for: okUnload)
        XCTAssertEqual((r4 as? HTTPURLResponse)?.statusCode, 200)
    }

    func testSettingsLogDirAndRevealDirectoryHardening() async throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_sec_logs_\(UUID().uuidString)", isDirectory: true)
        let existingDir = tempRoot.appendingPathComponent("logs", isDirectory: true)
        let appDir = tempRoot.appendingPathComponent("Evil.app/Contents/MacOS", isDirectory: true)
        let regularFile = tempRoot.appendingPathComponent("not_a_dir.txt")
        try fm.createDirectory(at: existingDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: appDir, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: regularFile)
        defer { try? fm.removeItem(at: tempRoot) }

        XCTAssertNoThrow(try QWHttpSecurity.validateSafeLogDirectory(existingDir.path, createIfInQwgRoot: false))
        XCTAssertThrowsError(try QWHttpSecurity.validateSafeLogDirectory(regularFile.path, createIfInQwgRoot: false))
        XCTAssertThrowsError(try QWHttpSecurity.validateSafeLogDirectory(appDir.path, createIfInQwgRoot: false))
        XCTAssertThrowsError(try QWHttpSecurity.validateSafeLogDirectory(
            tempRoot.appendingPathComponent("missing_outside").path,
            createIfInQwgRoot: false
        ))
        XCTAssertThrowsError(try QWHttpSecurity.validateSafeLogFileName("../evil.log"))
        XCTAssertThrowsError(try QWHttpSecurity.validateSafeLogFileName("sub/evil.log"))

        let capturedArgsLock = NSLock()
        nonisolated(unsafe) var capturedArgs: [[String]] = []
        let (_, server, port) = try await startMockActorAndServer(finderRevealRunner: { args in
            capturedArgsLock.withLock { capturedArgs.append(args) }
            return 0
        })
        defer { server.stop() }
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let session = URLSession.shared

        var fileSettings = QWServerSettings()
        fileSettings.logDir = regularFile.path
        var settingsFileReq = URLRequest(url: baseURL.appendingPathComponent("v1/settings"))
        settingsFileReq.httpMethod = "POST"
        settingsFileReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        settingsFileReq.httpBody = try JSONEncoder().encode(fileSettings)
        let (fileBody, fileResp) = try await session.data(for: settingsFileReq)
        XCTAssertEqual((fileResp as? HTTPURLResponse)?.statusCode, 400)
        XCTAssertTrue((String(data: fileBody, encoding: .utf8) ?? "").contains("directory"))

        var appSettings = QWServerSettings()
        appSettings.logDir = appDir.path
        var settingsAppReq = URLRequest(url: baseURL.appendingPathComponent("v1/settings"))
        settingsAppReq.httpMethod = "POST"
        settingsAppReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        settingsAppReq.httpBody = try JSONEncoder().encode(appSettings)
        let (appBody, appResp) = try await session.data(for: settingsAppReq)
        XCTAssertEqual((appResp as? HTTPURLResponse)?.statusCode, 400)
        XCTAssertTrue((String(data: appBody, encoding: .utf8) ?? "").contains("application bundle"))

        var revealReq = URLRequest(url: baseURL.appendingPathComponent("v1/settings/reveal"))
        revealReq.httpMethod = "POST"
        revealReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (_, revResp) = try await session.data(for: revealReq)
        XCTAssertEqual((revResp as? HTTPURLResponse)?.statusCode, 200)
        let recorded = capturedArgsLock.withLock { capturedArgs }
        XCTAssertEqual(recorded.count, 1)
        XCTAssertEqual(recorded.first?.first, "--")
        XCTAssertEqual(recorded.first?.last, QWPaths.rootDirectoryURL.resolvingSymlinksInPath().standardizedFileURL.path)
    }

    func testConnectionCapTimeoutAndIncrementalParserHardening() async throws {
        for badReq in [
            "POST /v1/models/unload HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: +2\r\n\r\n{}",
            "POST /v1/models/unload HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: -0\r\n\r\n{}",
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nBrokenHeaderWithoutColon\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Space : bad\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Bare: a\nb\r\n\r\n"
        ] {
            var parser = QWHttpIncrementalParser()
            XCTAssertThrowsError(try parser.append(Data(badReq.utf8)), "Expected parser error for \(badReq)") { err in
                XCTAssertEqual((err as? QWHttpParseError)?.httpStatus, 400)
            }
        }

        let (_, server, port) = try await startMockActorAndServer(
            maxConcurrentConnections: 2,
            requestReadTimeoutSeconds: 0.25
        )
        defer { server.stop() }

        let q = DispatchQueue(label: "qwegine.sec-test.idle-conns")
        func openIdleConn() async throws -> NWConnection {
            try await withCheckedThrowingContinuation { cont in
                let c = NWConnection(
                    to: .hostPort(host: .ipv4(IPv4Address("127.0.0.1")!), port: NWEndpoint.Port(rawValue: port)!),
                    using: .tcp
                )
                let lock = NSLock()
                nonisolated(unsafe) var done = false
                c.stateUpdateHandler = { state in
                    if case .ready = state {
                        if lock.withLock({ let ok = !done; done = true; return ok }) {
                            cont.resume(returning: c)
                        }
                    } else if case .failed(let e) = state {
                        if lock.withLock({ let ok = !done; done = true; return ok }) {
                            cont.resume(throwing: e)
                        }
                    }
                }
                c.start(queue: q)
            }
        }

        let idle1 = try await openIdleConn()
        let idle2 = try await openIdleConn()
        defer {
            idle1.cancel()
            idle2.cancel()
        }
        try await Task.sleep(nanoseconds: 40_000_000)

        let overCap = try await sendRawHTTP(
            port: port,
            rawRequest: Data("GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".utf8)
        )
        XCTAssertEqual(overCap.statusCode, 503)

        // Wait past requestReadTimeoutSeconds (0.25s) so idle1 & idle2 receive 408 and free slots.
        try await Task.sleep(nanoseconds: 500_000_000)
        for _ in 0..<4 {
            let okResp = try await sendRawHTTP(
                port: port,
                rawRequest: Data("GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".utf8)
            )
            XCTAssertEqual(okResp.statusCode, 200)
        }
    }

    func testImagePolicyIPv6BlocksNAT64AndPinnedFetchPreservesHostAndRedirects() async throws {
        let blockedV6 = [
            "::", "::1", "::8.8.8.8", "::ffff:127.0.0.1", "::ffff:8.8.8.8",
            "64:ff9b::8.8.8.8", "64:ff9b:1::1", "fe80::1", "fc00::1", "fd12::1", "ff02::1"
        ]
        for addr in blockedV6 {
            let bytes = try XCTUnwrap(QWImageDecodePolicy.IPv6AddressBytes(addr))
            XCTAssertTrue(QWImageDecodePolicy.isBlockedIPv6(bytes), "Expected blocked IPv6 for \(addr)")
        }
        for allowedV6 in ["2606:4700:4700::1111", "2001:4860:4860::8888"] {
            let bytes = try XCTUnwrap(QWImageDecodePolicy.IPv6AddressBytes(allowedV6))
            XCTAssertFalse(QWImageDecodePolicy.isBlockedIPv6(bytes), "Expected public IPv6 allowed for \(allowedV6)")
        }

        for badURLStr in [
            "http://[fe80::1%25en0]/img.png",
            "http://localhost./img.png",
            "http://sub.localhost/img.png",
            "http://user:pass@8.8.8.8/img.png",
            "http://[64:ff9b::8.8.8.8]/img.png",
            "http://[64:ff9b:1::1]/img.png",
            "http://[::ffff:8.8.8.8]/img.png",
            "http://[::8.8.8.8]/img.png"
        ] {
            let url = try XCTUnwrap(URL(string: badURLStr))
            XCTAssertThrowsError(try QWImageDecodePolicy.validateAndResolveRemoteURL(url), "Expected SSRF throw for \(badURLStr)")
        }

        // Local loopback HTTP fixture exercising pinned IP connection, Host preservation, and 302 redirect handling.
        let listenerQueue = DispatchQueue(label: "qwegine.sec-test.pinned-server")
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = .hostPort(host: .ipv4(IPv4Address("127.0.0.1")!), port: .any)
        let listener = try NWListener(using: params)
        let capturedHostsLock = NSLock()
        nonisolated(unsafe) var capturedHosts: [String] = []
        let expectedPayload = Data("PNG_STUB_BYTES_PINNED".utf8)

        listener.newConnectionHandler = { conn in
            conn.start(queue: listenerQueue)
            conn.receive(minimumIncompleteLength: 1, maximumLength: 8192) { data, _, _, _ in
                let text = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                let firstLine = text.components(separatedBy: "\r\n").first ?? ""
                if let hostLine = text.components(separatedBy: "\r\n").first(where: { $0.lowercased().hasPrefix("host:") }) {
                    let hVal = hostLine.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    capturedHostsLock.withLock { capturedHosts.append(hVal) }
                }
                let reply: String
                if firstLine.contains(" /redirect ") {
                    reply = "HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    conn.send(content: Data(reply.utf8), isComplete: true, completion: .contentProcessed({ _ in }))
                } else {
                    reply = "HTTP/1.1 200 OK\r\nContent-Length: \(expectedPayload.count)\r\nConnection: close\r\n\r\n"
                    conn.send(content: Data(reply.utf8) + expectedPayload, isComplete: true, completion: .contentProcessed({ _ in }))
                }
            }
        }
        listener.start(queue: listenerQueue)
        defer { listener.cancel() }

        var boundPort: UInt16?
        for _ in 0..<40 {
            if let p = listener.port?.rawValue, p > 0 {
                boundPort = p
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let port = try XCTUnwrap(boundPort)

        let fetched = try await QWImageDecodePolicy.fetchPinnedRemoteDataAsync(
            hostHeader: "rebind-cdn.example:\(port)",
            sniHost: "rebind-cdn.example",
            resolvedIP: "127.0.0.1",
            port: port,
            useTLS: false,
            requestTarget: "/ok.png",
            maxBytes: 4096
        )
        XCTAssertEqual(fetched, expectedPayload)
        XCTAssertEqual(capturedHostsLock.withLock { capturedHosts.first }, "rebind-cdn.example:\(port)")

        do {
            _ = try await QWImageDecodePolicy.fetchPinnedRemoteDataAsync(
                hostHeader: "rebind-cdn.example:\(port)",
                sniHost: "rebind-cdn.example",
                resolvedIP: "127.0.0.1",
                port: port,
                useTLS: false,
                requestTarget: "/redirect",
                maxBytes: 4096
            )
            XCTFail("Expected redirect refusal")
        } catch let error as QWImageDecoderError {
            guard case .disallowedImageSource = error else {
                return XCTFail("Expected disallowedImageSource on 302, got \(error)")
            }
        }
    }

    func testImageDecoderPolicyBeforeCacheHeaderDimsAreaAndCountCaps() async throws {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent("qwg_sec_img_\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }

        // Valid 4x4 PNG written to local disk.
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 255, count: 4 * 4 * 4)
        let ctx = try XCTUnwrap(CGContext(
            data: &rgba,
            width: 4,
            height: 4,
            bitsPerComponent: 8,
            bytesPerRow: 16,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let cg = try XCTUnwrap(ctx.makeImage())
        let imgURL = tempDir.appendingPathComponent("sample.png")
        let dest = try XCTUnwrap(CGImageDestinationCreateWithURL(imgURL as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))

        let vision = MockSecVisionBackend()
        QWImageDecoder.encodedCache.clear()
        let tokens = [1, 151652, QWMMConfig.imageToken, 151653, 2]

        // Warm cache under .cliLocal first, then verify .serverDefault rejects the same path before cache lookup.
        _ = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: tokens,
            imageUrls: [imgURL.path],
            hiddenDim: 16,
            visionBackend: vision,
            policy: .cliLocal
        )
        do {
            _ = try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: tokens,
                imageUrls: [imgURL.path],
                hiddenDim: 16,
                visionBackend: vision,
                policy: .serverDefault
            )
            XCTFail("Expected .serverDefault to reject local file path even when cached under .cliLocal")
        } catch let mmErr as QWMultimodalError {
            XCTAssertTrue("\(mmErr)".contains("disallowed image source"))
        }

        // Reject image lacking header pixel dimensions instead of full decompression fallback.
        let headerlessTIFF = Data([0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        XCTAssertThrowsError(try QWImageDecoder.decode(data: headerlessTIFF)) { err in
            guard case .imageDecodeFailed(let reason) = err as? QWImageDecoderError else {
                return XCTFail("Expected imageDecodeFailed for headerless image, got \(err)")
            }
            XCTAssertTrue(reason.contains("Missing pixel dimensions"))
        }

        // Reject 4097x4097 PNG (each side <= 8192, area 16,785,409 > 16,777,216) via header before full alloc.
        let bigAreaPNG = try makeSolidGrayscalePNGData(width: 4097, height: 4097)
        XCTAssertThrowsError(try QWImageDecoder.decode(data: bigAreaPNG)) { err in
            guard case .imageTooLarge = err as? QWImageDecoderError else {
                return XCTFail("Expected imageTooLarge for area > 16_777_216, got \(err)")
            }
        }

        // Reject > 16 images in a single turn.
        let imgData = try Data(contentsOf: imgURL)
        let dataUri = "data:image/png;base64,\(imgData.base64EncodedString())"
        do {
            _ = try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: tokens,
                imageUrls: Array(repeating: dataUri, count: 17),
                hiddenDim: 16,
                visionBackend: vision,
                policy: .serverDefault
            )
            XCTFail("Expected refusal for > 16 images per turn")
        } catch let mmErr as QWMultimodalError {
            XCTAssertTrue("\(mmErr)".contains("16"))
        }
    }

    func testMemoryPackWorkspaceRootsAndChatModelPathRestrictions() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("qwg_sec_ws_\(UUID().uuidString)", isDirectory: true)
        let allowedDir = root.appendingPathComponent("allowed-repo", isDirectory: true)
        let deniedDir = root.appendingPathComponent("secret-repo", isDirectory: true)
        let fakeModelsRoot = root.appendingPathComponent("models", isDirectory: true)
        let fakeModelDir = fakeModelsRoot.appendingPathComponent("demo-model", isDirectory: true)
        try fm.createDirectory(at: allowedDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: deniedDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: fakeModelDir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: fakeModelDir.appendingPathComponent("config.json"))
        defer { try? fm.removeItem(at: root) }

        let settings = QWServerSettings()
        XCTAssertFalse(QWServerSettings.isArbitraryFilesystemModelPath("demo-model"))
        XCTAssertTrue(QWServerSettings.isArbitraryFilesystemModelPath("/tmp/evil.qwp"))
        XCTAssertTrue(QWServerSettings.isArbitraryFilesystemModelPath("~/evil.qwp"))
        XCTAssertTrue(QWServerSettings.isArbitraryFilesystemModelPath("../evil.qwp"))

        let resolvedRegistered = settings.resolveModelReference(
            "demo-model",
            allowArbitraryFilesystemPaths: false,
            registeredModelsRoot: fakeModelsRoot
        )
        XCTAssertEqual(
            URL(fileURLWithPath: resolvedRegistered.packPath).resolvingSymlinksInPath().standardizedFileURL.path,
            fakeModelDir.resolvingSymlinksInPath().standardizedFileURL.path
        )
        let blockedExternal = settings.resolveModelReference(
            deniedDir.path,
            allowArbitraryFilesystemPaths: false,
            registeredModelsRoot: fakeModelsRoot
        )
        XCTAssertEqual(blockedExternal.packPath, "")

        let (_, server, port) = try await startMockActorAndServer(allowedWorkspaceRoots: [allowedDir.path])
        defer { server.stop() }
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let session = URLSession.shared

        for route in ["v1/memory/inspect", "v1/memory/index", "v1/memory/mount"] {
            var req = URLRequest(url: baseURL.appendingPathComponent(route))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["path": deniedDir.path])
            let (_, resp) = try await session.data(for: req)
            XCTAssertEqual((resp as? HTTPURLResponse)?.statusCode, 403, "Expected 403 outside allowed root for \(route)")
        }

        var okInspect = URLRequest(url: baseURL.appendingPathComponent("v1/memory/inspect"))
        okInspect.httpMethod = "POST"
        okInspect.setValue("application/json", forHTTPHeaderField: "Content-Type")
        okInspect.httpBody = try JSONSerialization.data(withJSONObject: ["path": allowedDir.path])
        let (_, okResp) = try await session.data(for: okInspect)
        XCTAssertEqual((okResp as? HTTPURLResponse)?.statusCode, 200)

        for badModel in ["/etc/passwd", "~/evil.qwp", "../escape.qwp"] {
            var chatReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
            chatReq.httpMethod = "POST"
            chatReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
            chatReq.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": badModel,
                "messages": [["role": "user", "content": "hi"]]
            ])
            let (_, chatResp) = try await session.data(for: chatReq)
            XCTAssertEqual((chatResp as? HTTPURLResponse)?.statusCode, 400, "Expected 400 for model path \(badModel)")
        }
    }

    func testWebUIJavaScriptXSSAndUrlSanitizationHardening() throws {
        let root = Self.repoRootURL()
        let jsPaths = [
            "Sources/Apps/QWServer/Resources/js/state.js",
            "Sources/Apps/QWServer/Resources/js/markdown.js",
            "Sources/Apps/QWServer/Resources/js/actions.js",
            "Sources/Apps/QWServer/Resources/js/settings.js",
            "Sources/Apps/QWServer/Resources/js/chat-presets.js",
            "Sources/Apps/QWServer/Resources/js/chat.js"
        ]
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, ex in
            jsError = ex?.toString()
        }
        ctx.evaluateScript("""
        var elements = {};
        function stubEl() {
          return {
            hidden: false, innerHTML: "", textContent: "", value: "", disabled: false,
            style: {}, classList: { add: function(){}, remove: function(){} },
            setAttribute: function(){}, querySelector: function(){ return null; }, querySelectorAll: function(){ return []; }
          };
        }
        function fetchDashboard() {}
        function fetchEngineInfo() {}
        var document = {
          getElementById: function(id) { if (!elements[id]) elements[id] = stubEl(); return elements[id]; },
          addEventListener: function() {}
        };
        """)
        for rel in jsPaths {
            let source = try String(contentsOf: root.appendingPathComponent(rel), encoding: .utf8)
            ctx.evaluateScript(source)
            XCTAssertNil(jsError, "JS evaluation error in \(rel): \(jsError ?? "")")
        }

        XCTAssertTrue(ctx.evaluateScript("sanitizeMarkdownUrl('//evil.example/a') === null")?.toBool() == true)
        XCTAssertTrue(ctx.evaluateScript("sanitizeMarkdownUrl('/\\\\evil.example/a') === null")?.toBool() == true)
        XCTAssertTrue(ctx.evaluateScript("sanitizeMarkdownUrl('javascript:alert(1)') === null")?.toBool() == true)
        XCTAssertTrue(ctx.evaluateScript("sanitizeMarkdownUrl('/ui#top') === '/ui#top'")?.toBool() == true)

        let payload = #"a');alert(1);//<img src=x>"#
        ctx.setObject(payload, forKeyedSubscript: "xssPayload" as NSString)
        let renderedHtml = ctx.evaluateScript("""
        (function() {
          discoveredModelsList = [{ name: xssPayload, packPath: "/tmp/" + xssPayload, capabilities: ["chat"], supported: true }];
          renderModelsCards();
          var cardHtml = elements["models-cards-grid"].innerHTML;
          openMemoryPackModalWithInspection({
            status: "wrong_model",
            rootPath: "/tmp",
            availableArtifacts: [{
              cacheFileExists: true,
              matchesActiveModel: false,
              modelInstalled: true,
              installedModelName: xssPayload,
              artifact: { modelName: xssPayload, kvMode: "q4", lsaMode: "auto", tokenCount: 100, cacheFileBytes: 1024 }
            }]
          });
          var memHtml = elements["mempack-artifacts-list"].innerHTML;
          chatState.sessions = [{ id: xssPayload, title: xssPayload, presetId: "server-default", messages: [
            { id: xssPayload, role: "assistant", content: "```swift\\nlet x = 1\\n```", toolCalls: [{ id: xssPayload, function: { name: xssPayload, arguments: "{}" } }], images: [{ dataUrl: "javascript:alert(1)", name: "x" }] }
          ], updatedAt: 1 }];
          chatState.activeSessionId = xssPayload;
          renderChatSessionList();
          renderChatTranscript();
          var chatHtml = elements["chat-session-list"].innerHTML + elements["chat-transcript"].innerHTML;
          var mdHtml = renderSmokeProbeMarkdown("[bad](//evil.example) [xss](javascript:alert(1)) [ok](/ui#top)");
          return cardHtml + memHtml + chatHtml + mdHtml;
        })();
        """)?.toString() ?? ""

        XCTAssertNil(jsError, "JS render error: \(jsError ?? "")")
        XCTAssertFalse(renderedHtml.lowercased().contains("onclick="), "Interpolated onclick handler leaked into rendered HTML")
        XCTAssertFalse(renderedHtml.contains("javascript:alert"), "javascript: URL leaked into rendered HTML")
        XCTAssertFalse(renderedHtml.contains("href=\"//evil.example\""), "Protocol-relative URL leaked into markdown link")
        XCTAssertTrue(renderedHtml.contains("href=\"/ui#top\""), "Expected safe relative /ui#top markdown link")
        XCTAssertTrue(renderedHtml.contains("data-select-model-name="))
        XCTAssertTrue(renderedHtml.contains("data-mempack-switch-model="))
        XCTAssertTrue(renderedHtml.contains("data-chat-action="))
    }
}
