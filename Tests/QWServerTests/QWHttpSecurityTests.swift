// QWHttpSecurityTests.swift — Security regression tests for framing, bind/auth, CORS, SSRF, and perms.

import Foundation
import Network
import XCTest
@testable import QWServer

private extension QWHttpServer {
    var boundPort: UInt16? {
        listener?.port?.rawValue
    }
}

final class QWHttpSecurityTests: XCTestCase {
    func testIncrementalParserHandlesSplitHeadersAndBody() throws {
        var parser = QWHttpIncrementalParser()
        let chunk1 = Data("POST /v1/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 11\r\n\r".utf8)
        let chunk2 = Data("\nhello ".utf8)
        let chunk3 = Data("world".utf8)

        XCTAssertEqual(try parser.append(chunk1), .needMoreData)
        XCTAssertEqual(try parser.append(chunk2), .needMoreData)
        guard case .complete(let req) = try parser.append(chunk3) else {
            return XCTFail("Expected completed parse after final body bytes")
        }
        XCTAssertEqual(req.method, "POST")
        XCTAssertEqual(req.path, "/v1/chat")
        XCTAssertEqual(req.headers["content-length"], "11")
        XCTAssertEqual(req.body, Data("hello world".utf8))
    }

    func testIncrementalParserRejectsOversizedAndMalformedFraming() {
        var smallCap = QWHttpIncrementalParser(maxHeaderBytes: 32)
        XCTAssertThrowsError(try smallCap.append(Data(repeating: 65, count: 40))) { err in
            XCTAssertEqual(err as? QWHttpParseError, .headerTooLarge)
            XCTAssertEqual((err as? QWHttpParseError)?.httpStatus, 431)
        }

        var conflict = QWHttpIncrementalParser()
        let badConflict = Data("POST / HTTP/1.1\r\nContent-Length: 4\r\nContent-Length: 5\r\n\r\n".utf8)
        XCTAssertThrowsError(try conflict.append(badConflict)) { err in
            XCTAssertEqual(err as? QWHttpParseError, .invalidContentLength)
        }

        var negative = QWHttpIncrementalParser()
        let badNegative = Data("POST / HTTP/1.1\r\nContent-Length: -2\r\n\r\n".utf8)
        XCTAssertThrowsError(try negative.append(badNegative)) { err in
            XCTAssertEqual(err as? QWHttpParseError, .invalidContentLength)
        }

        var chunked = QWHttpIncrementalParser()
        let badChunked = Data("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".utf8)
        XCTAssertThrowsError(try chunked.append(badChunked)) { err in
            XCTAssertEqual(err as? QWHttpParseError, .unsupportedTransferEncoding)
            XCTAssertEqual((err as? QWHttpParseError)?.httpStatus, 501)
        }
    }

    func testServerRefusesUnauthenticatedNonLoopbackBind() {
        let cfg = QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test-model")
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(actor: actor, host: "0.0.0.0", port: 0, logger: logger)

        XCTAssertThrowsError(try server.start()) { err in
            XCTAssertEqual(err as? QWHttpServerError, .unauthenticatedNonLoopbackBind(host: "0.0.0.0"))
        }
    }

    func testEphemeralLoopbackServerAuthAndCors() async throws {
        let cfg = QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test-model")
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: cfg, logger: logger)
        let server = QWHttpServer(
            actor: actor,
            host: "127.0.0.1",
            port: 0,
            logger: logger,
            apiKey: "secret-test-key",
            corsAllowedOrigins: ["https://allowed.example"]
        )
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.boundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))
        let session = URLSession.shared

        let (_, unauthResp) = try await session.data(from: baseURL.appendingPathComponent("health"))
        XCTAssertEqual((unauthResp as? HTTPURLResponse)?.statusCode, 401)

        var badBearer = URLRequest(url: baseURL.appendingPathComponent("health"))
        badBearer.setValue("Bearer wrong-key", forHTTPHeaderField: "Authorization")
        let (_, badResp) = try await session.data(for: badBearer)
        XCTAssertEqual((badResp as? HTTPURLResponse)?.statusCode, 401)

        var okBearer = URLRequest(url: baseURL.appendingPathComponent("health"))
        okBearer.setValue("Bearer secret-test-key", forHTTPHeaderField: "Authorization")
        let (_, okResp) = try await session.data(for: okBearer)
        XCTAssertEqual((okResp as? HTTPURLResponse)?.statusCode, 200)

        var okHeaderKey = URLRequest(url: baseURL.appendingPathComponent("health"))
        okHeaderKey.setValue("secret-test-key", forHTTPHeaderField: "x-api-key")
        let (_, okKeyResp) = try await session.data(for: okHeaderKey)
        XCTAssertEqual((okKeyResp as? HTTPURLResponse)?.statusCode, 200)

        var allowedOpt = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        allowedOpt.httpMethod = "OPTIONS"
        allowedOpt.setValue("http://127.0.0.1:8080", forHTTPHeaderField: "Origin")
        let (_, allowedOptResp) = try await session.data(for: allowedOpt)
        let allowedHTTP = try XCTUnwrap(allowedOptResp as? HTTPURLResponse)
        XCTAssertEqual(allowedHTTP.statusCode, 204)
        XCTAssertEqual(allowedHTTP.value(forHTTPHeaderField: "Access-Control-Allow-Origin"), "http://127.0.0.1:8080")

        var deniedOpt = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        deniedOpt.httpMethod = "OPTIONS"
        deniedOpt.setValue("https://evil.example", forHTTPHeaderField: "Origin")
        let (_, deniedOptResp) = try await session.data(for: deniedOpt)
        let deniedHTTP = try XCTUnwrap(deniedOptResp as? HTTPURLResponse)
        XCTAssertEqual(deniedHTTP.statusCode, 204)
        XCTAssertNil(deniedHTTP.value(forHTTPHeaderField: "Access-Control-Allow-Origin"))
    }

    func testCorsOriginResolution() {
        XCTAssertEqual(
            QWHttpSecurity.resolvedAllowedOrigin("http://localhost:3000", allowlist: []),
            "http://localhost:3000"
        )
        XCTAssertEqual(
            QWHttpSecurity.resolvedAllowedOrigin("http://[::1]:5173", allowlist: []),
            "http://[::1]:5173"
        )
        XCTAssertNil(QWHttpSecurity.resolvedAllowedOrigin("https://evil.example", allowlist: []))
        XCTAssertEqual(
            QWHttpSecurity.resolvedAllowedOrigin("https://trusted.example", allowlist: ["https://trusted.example"]),
            "https://trusted.example"
        )
    }

    func testImageDecodePolicyBlocksPrivateAndDefaultSources() throws {
        let blockedURLs = [
            "http://169.254.169.254/latest/meta-data",
            "http://127.0.0.1/a.png",
            "http://10.0.0.1/a.png",
            "http://192.168.1.10/a.png",
            "http://[::1]/a.png",
            "http://localhost/a.png",
            "ftp://example.com/a.png",
        ]
        for raw in blockedURLs {
            let url = try XCTUnwrap(URL(string: raw))
            XCTAssertThrowsError(try QWImageDecodePolicy.validateAndResolveRemoteURL(url), "Expected SSRF refusal for \(raw)") { err in
                guard case .disallowedImageSource = err as? QWImageDecoderError else {
                    return XCTFail("Expected disallowedImageSource for \(raw), got \(err)")
                }
            }
        }

        for disallowed in ["file:///etc/hosts", "https://example.com/image.png"] {
            XCTAssertThrowsError(try QWImageDecoder.decode(urlOrDataUri: disallowed)) { err in
                guard case .disallowedImageSource = err as? QWImageDecoderError else {
                    return XCTFail("Expected disallowedImageSource for \(disallowed), got \(err)")
                }
            }
        }
    }

    func testSettingsManagerSavesPrivatePermissions() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("qwg_sec_settings_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let fileURL = root.appendingPathComponent("settings.json")
        var settings = QWServerSettings()
        settings.logDir = root.appendingPathComponent("logs").path
        try QWSettingsManager().saveSettings(settings, to: fileURL)

        let dirAttrs = try FileManager.default.attributesOfItem(atPath: root.path)
        let fileAttrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        XCTAssertEqual((dirAttrs[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        XCTAssertEqual((fileAttrs[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    }

    func testCancellationHubPropagatesToGenerationCancelFlag() async {
        let hub = QWConnectionCancellationHub()
        let cancelFlag = QWGenerationCancelFlag()
        let started = expectation(description: "generation task entered cancellation handler")

        let routeTask = Task {
            await withTaskCancellationHandler(
                operation: {
                    started.fulfill()
                    while !cancelFlag.isCancelled && !Task.isCancelled {
                        try? await Task.sleep(nanoseconds: 5_000_000)
                    }
                },
                onCancel: {
                    cancelFlag.cancel()
                }
            )
        }
        hub.register {
            routeTask.cancel()
        }

        await fulfillment(of: [started], timeout: 2.0)
        XCTAssertFalse(cancelFlag.isCancelled)
        hub.fire()
        await routeTask.value
        XCTAssertTrue(hub.isTerminated)
        XCTAssertTrue(cancelFlag.isCancelled)
    }
}
