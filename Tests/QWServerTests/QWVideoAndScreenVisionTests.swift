// QWVideoAndScreenVisionTests.swift — Synthetic MP4 AVFoundation video decode,
// SSRF/local-file security policy, multi-protocol video_url schema, and WebUI JS tests.

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import JavaScriptCore
import Network
import QWCore
import QWEngine
import QWMM
@testable import QWServer
import QWToken
import QWVision
import XCTest

private final class SyntheticVideoVisionBackend: QWVisionBackend, @unchecked Sendable {
    let visionConfig: QWVisionConfig
    private(set) var isLoaded: Bool
    let isVisionEnabled: Bool
    private let hiddenDim: Int
    private(set) var encodeCallCount = 0

    init(
        videoTokenId: Int = 248057,
        hiddenDim: Int = 16,
        isLoaded: Bool = false,
        isVisionEnabled: Bool = true
    ) {
        let base = QWVisionConfig.qwen3VLReference
        self.visionConfig = QWVisionConfig(
            dim: 64,
            heads: 4,
            headDim: 16,
            depth: 1,
            mlpInner: 128,
            patch: base.patch,
            temporal: base.temporal,
            merge: base.merge,
            numPosEmbeds: base.numPosEmbeds,
            outDim: hiddenDim,
            imageTokenId: base.imageTokenId,
            videoTokenId: videoTokenId
        )
        self.isLoaded = isLoaded
        self.isVisionEnabled = isVisionEnabled
        self.hiddenDim = hiddenDim
    }

    var residentBytes: Int { isLoaded ? 1024 : 0 }

    func prepareForPrompt(hasImages: Bool) throws {
        guard isVisionEnabled || !hasImages else {
            throw QWMultimodalError.badArgument("--no-load-vision")
        }
        isLoaded = hasImages
    }

    func setVisionLoaded(_ loaded: Bool) throws {
        isLoaded = loaded
    }

    func encode(patches: [Float], grid: QWVisionGridPlan, bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float] {
        encodeCallCount += 1
        let firstVal = patches.first ?? 0
        return [Float](repeating: firstVal + Float(encodeCallCount), count: grid.nToken * hiddenDim)
    }
}

final class QWVideoAndScreenVisionTests: XCTestCase {
    /// Synthesizes a 4-frame `64x64` H.264 `.mp4` clip (Red, Green, Blue, Yellow at `4 fps`, `1.0 s`) in temp dir.
    private static func writeSyntheticFourColorMP4(to url: URL) async throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let width = 64
        let height = 64
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        XCTAssertTrue(writer.canAdd(input))
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)

        let bgraColors: [(UInt8, UInt8, UInt8, UInt8)] = [
            (0, 0, 255, 255),   // Frame 0: Red
            (0, 255, 0, 255),   // Frame 1: Green
            (255, 0, 0, 255),   // Frame 2: Blue
            (0, 255, 255, 255), // Frame 3: Yellow
        ]
        for (idx, bgra) in bgraColors.enumerated() {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            var pbOpt: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool,
                  CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOpt) == kCVReturnSuccess,
                  let pb = pbOpt else {
                XCTFail("Failed to allocate CVPixelBuffer")
                return
            }
            CVPixelBufferLockBaseAddress(pb, [])
            if let base = CVPixelBufferGetBaseAddress(pb) {
                let bpr = CVPixelBufferGetBytesPerRow(pb)
                for y in 0..<height {
                    let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<width {
                        row[x * 4 + 0] = bgra.0
                        row[x * 4 + 1] = bgra.1
                        row[x * 4 + 2] = bgra.2
                        row[x * 4 + 3] = bgra.3
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, [])
            let pts = CMTime(value: CMTimeValue(idx), timescale: 4)
            XCTAssertTrue(adaptor.append(pb, withPresentationTime: pts))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "AVAssetWriter failed: \(String(describing: writer.error))")
    }

    func testDecodeSyntheticMP4DataURIAndEnforceSecurityPolicies() async throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let mp4URL = tmpDir.appendingPathComponent("qwg-test-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: mp4URL) }
        try await Self.writeSyntheticFourColorMP4(to: mp4URL)
        let mp4Data = try Data(contentsOf: mp4URL)
        let dataURI = "data:video/mp4;base64," + mp4Data.base64EncodedString()

        // 1. Server default allows base64 data:video/mp4 and extracts 4 temporal frames (grid.t == 2).
        let decoded = try await QWVideoDecoder.decodeSmartResized(
            input: QWVideoInputSpec(url: dataURI, fps: 4.0, maxFrames: 4),
            policy: .serverDefault,
            spec: .qwen3VLReference
        )
        XCTAssertEqual(decoded.framesRGB.count, 4)
        XCTAssertEqual(decoded.grid.t, 2)
        // Frame 0 is predominantly Red (`R > 180, B < 60`); Frame 2 is predominantly Blue (`B > 180, R < 60`).
        XCTAssertGreaterThan(Int(decoded.framesRGB[0][0]), 180)
        XCTAssertLessThan(Int(decoded.framesRGB[0][2]), 60)
        XCTAssertGreaterThan(Int(decoded.framesRGB[2][2]), 180)
        XCTAssertLessThan(Int(decoded.framesRGB[2][0]), 60)

        // 2. Server default rejects local paths and remote HTTP(S) URLs.
        do {
            _ = try await QWVideoDecoder.decodeSmartResized(
                input: QWVideoInputSpec(url: mp4URL.path),
                policy: .serverDefault
            )
            XCTFail("Expected serverDefault to block local video file path")
        } catch let err as QWImageDecoderError {
            if case .disallowedImageSource = err {} else { XCTFail("Unexpected error: \(err)") }
        }

        do {
            _ = try await QWVideoDecoder.decodeSmartResized(
                input: QWVideoInputSpec(url: "https://example.com/sample.mp4"),
                policy: .serverDefault
            )
            XCTFail("Expected serverDefault to block remote video URL")
        } catch let err as QWImageDecoderError {
            if case .disallowedImageSource = err {} else { XCTFail("Unexpected error: \(err)") }
        }

        // 3. Even when remote URLs are enabled, loopback/metadata SSRF targets are blocked before network/AVFoundation.
        let remotePolicy = QWImageDecodePolicy(allowRemoteImageURLs: true)
        for blocked in ["http://127.0.0.1:8080/v.mp4", "http://169.254.169.254/latest/v.mp4", "http://localhost/v.mp4"] {
            do {
                _ = try await QWVideoDecoder.decodeSmartResized(
                    input: QWVideoInputSpec(url: blocked),
                    policy: remotePolicy
                )
                XCTFail("Expected SSRF guard to block \(blocked)")
            } catch let err as QWImageDecoderError {
                if case .disallowedImageSource = err {} else { XCTFail("Unexpected error for \(blocked): \(err)") }
            }
        }

        // 4. Non-video magic bytes in data:video/mp4 are rejected before AVFoundation.
        let badPayload = "data:video/mp4;base64," + Data("not-a-valid-mp4-container-header".utf8).base64EncodedString()
        do {
            _ = try await QWVideoDecoder.decodeSmartResized(
                input: QWVideoInputSpec(url: badPayload),
                policy: .serverDefault
            )
            XCTFail("Expected invalid container header rejection")
        } catch is QWImageDecoderError {}
    }

    func testMultiProtocolVideoSchemasAndPrepareMultimodalTurn() async throws {
        QWVideoDecoder.encodedCache.clear()
        let tmpDir = FileManager.default.temporaryDirectory
        let mp4URL = tmpDir.appendingPathComponent("qwg-mm-turn-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: mp4URL) }
        try await Self.writeSyntheticFourColorMP4(to: mp4URL)
        let b64 = try Data(contentsOf: mp4URL).base64EncodedString()
        let dataURI = "data:video/mp4;base64," + b64

        // OpenAI /v1/chat/completions video_url + vLLM video object
        let chatJSON = """
        {"model":"stub","messages":[{"role":"user","content":[
          {"type":"video_url","video_url":{"url":"\(dataURI)","fps":4.0,"max_frames":4}},
          {"type":"text","text":"Describe the color transition."}
        ]}]}
        """
        let chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(chatJSON.utf8))
        let videos = chatReq.extractVideoInputs()
        XCTAssertEqual(videos.count, 1)
        XCTAssertEqual(videos[0].fps, 4.0)
        XCTAssertEqual(videos[0].maxFrames, 4)
        XCTAssertTrue(chatReq.messages[0].content?.templateTextValue.contains("<|video_pad|>") == true)

        // OpenAI /v1/responses input_video mapping
        let respInput = try QWJson.parse("""
        [{"role":"user","content":[{"type":"input_video","video_url":{"url":"\(dataURI)","max_frames":4}},{"type":"input_text","text":"What happens?"}]}]
        """)
        let mappedResp = try OpenAIResponsesMapper.parseInput(respInput, hasPrevious: false)
        XCTAssertEqual(mappedResp.first?.content?.videoInputs.count, 1)

        // Anthropic /v1/messages video block mapping
        let anthJSON = """
        {"model":"stub","max_tokens":32,"messages":[{"role":"user","content":[
          {"type":"video","fps":4.0,"max_frames":4,"source":{"type":"base64","media_type":"video/mp4","data":"\(b64)"}},
          {"type":"text","text":"Describe."}
        ]}]}
        """
        let anthReq = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(anthJSON.utf8))
        let mappedAnth = try anthReq.toChatCompletionRequest(replyModel: "stub", requireMaxTokens: true, headers: [:])
        XCTAssertEqual(mappedAnth.request.extractVideoInputs().count, 1)

        // Prepare multimodal turn with conv3d (`videoToken >= 0`) backend + auto-unload on text turn
        let backend = SyntheticVideoVisionBackend(videoTokenId: 248057, hiddenDim: 16)
        let mm = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 248057, 20],
            imageUrls: [],
            videoInputs: videos,
            hiddenDim: 16,
            visionBackend: backend,
            policy: .serverDefault
        )
        XCTAssertTrue(mm.loadedOnDemand)
        XCTAssertEqual(mm.visualSpans.count, 1)
        XCTAssertTrue(mm.visualSpans[0].isVideo)
        XCTAssertEqual(mm.visualSpans[0].gridT, 2)
        XCTAssertGreaterThan(mm.tokens.count, 3)

        let textFollowup = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [30, 31],
            imageUrls: [],
            videoInputs: [],
            hiddenDim: 16,
            visionBackend: backend,
            policy: .serverDefault
        )
        XCTAssertTrue(textFollowup.unloadedOnText)

        // Fallback frame-sequence path when `videoTokenId < 0`
        let fallbackBackend = SyntheticVideoVisionBackend(videoTokenId: -1, hiddenDim: 16)
        let fbTurn = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 248056, 248056, 248056, 248056, 20],
            imageUrls: [],
            videoInputs: [QWVideoInputSpec(url: dataURI, maxFrames: 4)],
            hiddenDim: 16,
            visionBackend: fallbackBackend,
            policy: .serverDefault
        )
        XCTAssertEqual(fbTurn.visualSpans.count, 4)

        // Disabled vision (`--no-load-vision`) rejects video even when cached
        let disabledBackend = SyntheticVideoVisionBackend(videoTokenId: 248057, hiddenDim: 16, isVisionEnabled: false)
        do {
            _ = try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: [10, 248057, 20],
                imageUrls: [],
                videoInputs: videos,
                hiddenDim: 16,
                visionBackend: disabledBackend,
                policy: .serverDefault
            )
            XCTFail("Expected --no-load-vision to reject video turn")
        } catch let err as QWMultimodalError {
            XCTAssertTrue("\(err)".contains("--no-load-vision"))
        }
    }

    func testLoopbackScreenCaptureEndpointGuardAndWebUIVideoSerialization() throws {
        let lo4 = NWEndpoint.hostPort(host: .ipv4(.loopback), port: 8080)
        let lo6 = NWEndpoint.hostPort(host: .ipv6(.loopback), port: 8080)
        let lan = NWEndpoint.hostPort(host: .ipv4(IPv4Address("192.168.1.42")!), port: 8080)
        XCTAssertTrue(QWHttpServer.isLoopbackClientEndpoint(lo4))
        XCTAssertTrue(QWHttpServer.isLoopbackClientEndpoint(lo6))
        XCTAssertFalse(QWHttpServer.isLoopbackClientEndpoint(lan))

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let ctx = try XCTUnwrap(JSContext())
        var jsErr: String?
        ctx.exceptionHandler = { _, ex in jsErr = ex?.toString() }
        for rel in [
            "Sources/Apps/QWServer/Resources/js/state.js",
            "Sources/Apps/QWServer/Resources/js/chat-presets.js",
            "Sources/Apps/QWServer/Resources/js/chat-media.js",
            "Sources/Apps/QWServer/Resources/js/chat.js",
        ] {
            let src = try String(contentsOf: root.appendingPathComponent(rel), encoding: .utf8)
            ctx.evaluateScript(src)
            XCTAssertNil(jsErr, "JS error in \(rel): \(jsErr ?? "")")
        }
        let check = ctx.evaluateScript("""
        (() => {
          const safeOk = safeChatVideoDataUrl("data:video/mp4;base64,AAAA");
          const safeBad = safeChatVideoDataUrl("javascript:alert(1)");
          const turn = serializeChatTurnForApi({
            role: "user",
            content: "Watch this",
            images: [{ dataUrl: "data:image/png;base64,BBBB" }],
            videos: [{ dataUrl: "data:video/mp4;base64,CCCC" }]
          });
          return { safeOk, safeBad, partCount: turn.content.length, vidType: turn.content[2].type, vidUrl: turn.content[2].video_url.url };
        })()
        """)?.toDictionary()
        XCTAssertNil(jsErr)
        XCTAssertEqual(check?["safeOk"] as? String, "data:video/mp4;base64,AAAA")
        XCTAssertEqual(check?["safeBad"] as? String, "")
        XCTAssertEqual(check?["partCount"] as? Int, 3)
        XCTAssertEqual(check?["vidType"] as? String, "video_url")
        XCTAssertEqual(check?["vidUrl"] as? String, "data:video/mp4;base64,CCCC")
    }
}
