// QWCompletionsEndpointTests.swift — Tests for legacy OpenAI /v1/completions
// (raw tokenization, token IDs, echo+logprobs, streaming) and Anthropic /v1/messages/count_tokens.

import Foundation
import Network
import QWCore
@testable import QWEngine
import QWHAL
import QWHead
@testable import QWServer
import QWToken
import QWVision
import XCTest

private extension QWHttpServer {
    var testBoundPort: UInt16? { listener?.port?.rawValue }
}

private final class CompletionsMockBackend: QWModelBackend, @unchecked Sendable {
    private let lock = NSLock()
    let descriptor: QWModelDescriptor = .qwen38_27b
    var family: QWModelDescriptor.Family { descriptor.family }
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)? = nil
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState = QWPrefixCacheState(
        isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
    )
    let capabilities: Set<QWKnobCapability> = [.lsa]
    var supportsTokenConstraints: Bool { true }

    private var _capturedRequests: [QWModelGenerationRequest] = []
    var capturedRequests: [QWModelGenerationRequest] { lock.withLock { _capturedRequests } }

    private let generator: @Sendable (QWModelGenerationRequest) throws -> (tokens: [Int], altToken: Int)

    init(
        config: QWEngineLifecycleActor.Config,
        generator: @escaping @Sendable (QWModelGenerationRequest) throws -> (tokens: [Int], altToken: Int)
    ) {
        self.generator = generator
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: .qwen38_27b,
            headStorage: "HQ4_G32_M5",
            visionStatus: "none",
            contextBudget: config.maxContext,
            kvMode: config.kvMode,
            mtpDrafts: 0,
            mtpBackend: "none",
            gdnPrefillMode: "sequential",
            prefillScratchEviction: config.evictPrefillScratch,
            prefillChunkRows: config.prefillChunkSize,
            weightContainerNote: nil
        )
    }

    func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        lock.withLock { _capturedRequests.append(request) }
        if request.maxNewTokens == 1, let c = request.constraint, let allowed = try c.allowedTokens(), let forced = allowed.first {
            let candidates: [QWTokenLogprob.Candidate] = request.topLogprobsCount > 0
                ? [QWTokenLogprob.Candidate(tokenId: forced, logprob: -0.25),
                   QWTokenLogprob.Candidate(tokenId: forced == 65 ? 66 : 65, logprob: -1.75)]
                : []
            let entry = QWTokenLogprob(tokenId: forced, logprob: -0.25, topCandidates: candidates)
            request.onTokenLogprob?(entry)
            _ = onToken?(forced)
            return QWModelCompletionMetrics(
                tokens: [forced],
                prefillSec: 0.001,
                decodeSec: 0.001,
                firstTokenSec: 0.001,
                activeContextCapacity: parametersInfo.contextBudget,
                tokenLogprobs: request.collectLogprobs ? [entry] : []
            )
        }

        let plan = try generator(request)
        var emitted: [Int] = []
        var lps: [QWTokenLogprob] = []
        for tok in plan.tokens {
            if request.collectLogprobs {
                let cands: [QWTokenLogprob.Candidate] = request.topLogprobsCount > 0
                    ? [QWTokenLogprob.Candidate(tokenId: tok, logprob: -0.5),
                       QWTokenLogprob.Candidate(tokenId: plan.altToken, logprob: -2.0)]
                    : []
                let entry = QWTokenLogprob(tokenId: tok, logprob: -0.5, topCandidates: cands)
                lps.append(entry)
                request.onTokenLogprob?(entry)
            }
            emitted.append(tok)
            if let cont = onToken, !cont(tok) { break }
        }
        return QWModelCompletionMetrics(
            tokens: emitted,
            prefillSec: 0.001,
            decodeSec: 0.001,
            firstTokenSec: 0.001,
            activeContextCapacity: parametersInfo.contextBudget,
            tokenLogprobs: lps
        )
    }

    func buildStandalonePrefixCache(
        tokens: [Int], outputURL: URL, pinAsSharedBase: Bool,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int { 0 }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) { ([], 0) }
    func unmountSharedBasePrefixes() {}
}

final class QWCompletionsEndpointTests: XCTestCase {

    override func tearDown() {
        QWTelemetryRegistry.shared.reset()
        let reg = QWEngineTelemetryRegistry.shared
        reg.resetToIdle()
        reg.lock.withLock {
            reg.tokensInTotal = 0
            reg.tokensOutTotal = 0
            reg.lastPrefillTokPerSec = 0
            reg.lastDecodeTokPerSec = 0
            reg.lastPrefillSampledAt = nil
            reg.lastDecodeSampledAt = nil
            reg.ema1mPrefillTokPerSec = 0
            reg.ema1mDecodeTokPerSec = 0
            reg.totalPrefillComputedTokens = 0
            reg.totalPrefillSeconds = 0
            reg.totalDecodeTokens = 0
            reg.totalDecodeSeconds = 0
            reg.lastTurnSampledAt = nil
            reg.lastTtftMs = 0
            reg.totalTtftMs = 0
            reg.totalRequestsWithTtft = 0
            reg.lastTtftSampledAt = nil
            reg.contextTokensActive = nil
            reg.contextSampledAt = nil
        }
        QWRequestRecordStore.shared.clear()
        super.tearDown()
    }

    private func makeTokenizerAndTemplate() throws -> (QWTokenizer, QWChatTemplate) {
        var vocab: [String: Int] = [:]
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = b
        }
        let tokObj: [String: Any] = [
            "model": [
                "type": "BPE",
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [String]()
            ],
            "pre_tokenizer": [
                "type": "Metaspace",
                "replacement": " ",
                "prepend_scheme": "never",
                "split": false
            ],
            "added_tokens": [
                ["id": 256, "content": "<|im_start|>", "special": true],
                ["id": 257, "content": "<|im_end|>", "special": true]
            ]
        ]
        let tokData = try JSONSerialization.data(withJSONObject: tokObj)
        let tok = try QWTokenizer(json: tokData)
        let src = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
        return (tok, try QWChatTemplate(source: src))
    }

    func testCompletionsRawPromptsTokenArraysAndStopSequences() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let altTok = try tok.encode("z", addSpecialTokens: false)[0]
        let backend = CompletionsMockBackend(config: config) { req in
            let text = (try? tok.decode(req.promptTokens, skipSpecialTokens: false)) ?? ""
            if text.contains("STOPME") {
                return (try tok.encode("goodHALTbad", addSpecialTokens: false), altTok)
            }
            return (try tok.encode("world", addSpecialTokens: false), altTok)
        }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        func post(_ body: Any) async throws -> (Int, [String: Any]) {
            var req = URLRequest(url: baseURL.appendingPathComponent("v1/completions"))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (code, obj)
        }

        // 1. Raw string prompt with default max_tokens == 16 and no chat template tokens
        let expectedHelloTokens = try tok.encode("Hello", addSpecialTokens: false)
        let (st1, js1) = try await post(["model": "test-model", "prompt": "Hello"])
        XCTAssertEqual(st1, 200)
        XCTAssertEqual(js1["object"] as? String, "text_completion")
        let lastReq1 = try XCTUnwrap(backend.capturedRequests.last)
        XCTAssertEqual(lastReq1.promptTokens, expectedHelloTokens)
        XCTAssertEqual(lastReq1.maxNewTokens, 16)
        let c1 = try XCTUnwrap((js1["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(c1["text"] as? String, "world")

        // 2. Batch string array + integer token array prompts
        let (st2, js2) = try await post(["model": "test-model", "prompt": ["One", "Two"], "max_tokens": 5])
        XCTAssertEqual(st2, 200)
        let choices2 = try XCTUnwrap(js2["choices"] as? [[String: Any]])
        XCTAssertEqual(choices2.count, 2)
        XCTAssertEqual(choices2[0]["index"] as? Int, 0)
        XCTAssertEqual(choices2[1]["index"] as? Int, 1)

        let rawIDs = [65, 66, 67]
        let (st3, _) = try await post(["model": "test-model", "prompt": rawIDs, "max_tokens": 3])
        XCTAssertEqual(st3, 200)
        XCTAssertEqual(backend.capturedRequests.last?.promptTokens, rawIDs)

        let (stBadId, _) = try await post(["model": "test-model", "prompt": [999_999]])
        XCTAssertEqual(stBadId, 400)

        // 3. stop as bare string trims cleanly
        let (stStop, jsStop) = try await post(["model": "test-model", "prompt": "STOPME", "stop": "HALT"])
        XCTAssertEqual(stStop, 200)
        let cStop = try XCTUnwrap((jsStop["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(cStop["text"] as? String, "good")
        XCTAssertEqual(cStop["finish_reason"] as? String, "stop")
    }

    func testCompletionsEchoLogprobsZeroMaxTokensAndStreaming() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let altTok = try tok.encode("Z", addSpecialTokens: false)[0]
        let backend = CompletionsMockBackend(config: config) { _ in
            (try tok.encode("XY", addSpecialTokens: false), altTok)
        }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        // 1. echo=true + logprobs=2 + max_tokens=2: prompt token 0 is null, rest non-null
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/completions"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model",
            "prompt": "AB",
            "echo": true,
            "logprobs": 2,
            "max_tokens": 2
        ])
        let (d1, r1) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((r1 as? HTTPURLResponse)?.statusCode, 200)
        let js1 = try XCTUnwrap(try JSONSerialization.jsonObject(with: d1) as? [String: Any])
        let choice1 = try XCTUnwrap((js1["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice1["text"] as? String, "ABXY")
        let lp = try XCTUnwrap(choice1["logprobs"] as? [String: Any])
        let tokenLPs = try XCTUnwrap(lp["token_logprobs"] as? [Any])
        let tokens = try XCTUnwrap(lp["tokens"] as? [String])
        let offsets = try XCTUnwrap(lp["text_offset"] as? [Int])
        let topLPs = try XCTUnwrap(lp["top_logprobs"] as? [Any])
        XCTAssertGreaterThanOrEqual(tokens.count, 4)
        XCTAssertEqual(tokenLPs.count, tokens.count)
        XCTAssertEqual(offsets.count, tokens.count)
        XCTAssertEqual(topLPs.count, tokens.count)
        XCTAssertTrue(tokenLPs.first is NSNull, "first echoed token logprob must be null")
        XCTAssertTrue(topLPs.first is NSNull, "first echoed top_logprobs must be null")
        XCTAssertFalse(tokenLPs[1] is NSNull)
        XCTAssertNotNil(topLPs[1] as? [String: Any])

        // 2. lm_eval prompt-scoring pattern: echo=true, max_tokens=0, logprobs=1
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model",
            "prompt": "AB",
            "echo": true,
            "logprobs": 1,
            "max_tokens": 0
        ])
        let (dZero, rZero) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((rZero as? HTTPURLResponse)?.statusCode, 200)
        let jsZero = try XCTUnwrap(try JSONSerialization.jsonObject(with: dZero) as? [String: Any])
        let cZero = try XCTUnwrap((jsZero["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(cZero["text"] as? String, "AB")
        XCTAssertEqual((jsZero["usage"] as? [String: Any])?["completion_tokens"] as? Int, 0)

        // 3. Streaming SSE emits object: "text_completion" chunks and ends with [DONE]
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model",
            "prompt": "Hi",
            "max_tokens": 4,
            "stream": true
        ])
        let (sseData, sseResp) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((sseResp as? HTTPURLResponse)?.statusCode, 200)
        let sseText = String(decoding: sseData, as: UTF8.self)
        XCTAssertTrue(sseText.contains("\"object\":\"text_completion\"") || sseText.contains("\"object\" : \"text_completion\""))
        XCTAssertTrue(sseText.contains("data: [DONE]"))
    }

    func testAnthropicCountTokensEndpoint() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let backend = CompletionsMockBackend(config: config) { _ in ([65], 66) }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        func countTokens(_ body: [String: Any]) async throws -> (Int, [String: Any]) {
            var req = URLRequest(url: baseURL.appendingPathComponent("v1/messages/count_tokens"))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (code, obj)
        }

        // max_tokens omitted on count_tokens (Anthropic SDK standard)
        let (st1, js1) = try await countTokens([
            "model": "test-model",
            "messages": [["role": "user", "content": "Hello"]]
        ])
        XCTAssertEqual(st1, 200)
        let count1 = try XCTUnwrap(js1["input_tokens"] as? Int)
        XCTAssertGreaterThan(count1, 0)

        let (st2, js2) = try await countTokens([
            "model": "test-model",
            "system": "Be super helpful and concise.",
            "messages": [
                ["role": "user", "content": "Hello"],
                ["role": "assistant", "content": "Hi there!"],
                ["role": "user", "content": "Tell me much more about everything today."]
            ]
        ])
        XCTAssertEqual(st2, 200)
        let count2 = try XCTUnwrap(js2["input_tokens"] as? Int)
        XCTAssertGreaterThan(count2, count1)

        let (st404, js404) = try await countTokens([
            "model": "nonexistent-model",
            "messages": [["role": "user", "content": "Hello"]]
        ])
        XCTAssertEqual(st404, 404)
        XCTAssertEqual(js404["type"] as? String, "error")

        // Validation error inside actor.countChatPromptTokens maps cleanly to Anthropic 400 envelope
        let (stBadImg, jsBadImg) = try await countTokens([
            "model": "test-model",
            "messages": [[
                "role": "user",
                "content": [[
                    "type": "image",
                    "source": ["type": "base64", "media_type": "image/png", "data": "!!!invalid!!!"]
                ]]
            ]]
        ])
        XCTAssertEqual(stBadImg, 400)
        XCTAssertEqual(jsBadImg["type"] as? String, "error")
        XCTAssertEqual((jsBadImg["error"] as? [String: Any])?["type"] as? String, "invalid_request_error")
    }

    func testRemoteTokenizerEndpointsAndFiniteLogprobs() async throws {
        let (tok, baseTpl) = try makeTokenizerAndTemplate()
        let tplWithCtx = try QWChatTemplate(
            source: baseTpl.rawSource,
            tokenizerContext: [
                "bos_token": .string("<|im_start|>"),
                "eos_token": .string("<|im_end|>"),
                "pad_token": .string(""),
                "unk_token": .string("")
            ]
        )
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let emitId = try tok.encode("A", addSpecialTokens: false)[0]
        let backend = CompletionsMockBackend(config: config) { _ in ([emitId], 66) }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tplWithCtx) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        // 1. /v1/engine/info advertises remote tokenizer routes.
        let (infoData, infoResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/engine/info"))
        XCTAssertEqual((infoResp as? HTTPURLResponse)?.statusCode, 200)
        let endpoints = ((try JSONSerialization.jsonObject(with: infoData) as? [String: Any])?["endpoints"] as? [String: String])
        XCTAssertEqual(endpoints?["tokenizer_info"], "GET /tokenizer_info")
        XCTAssertEqual(endpoints?["tokenize"], "POST /tokenize")
        XCTAssertEqual(endpoints?["detokenize"], "POST /detokenize")

        // 2. GET /tokenizer_info & /v1/tokenizer_info?model=... omit empty tokens cleanly.
        for path in ["tokenizer_info", "v1/tokenizer_info?model=test-model"] {
            let (d, r) = try await URLSession.shared.data(from: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/\(path)")))
            XCTAssertEqual((r as? HTTPURLResponse)?.statusCode, 200)
            let js = try XCTUnwrap(try JSONSerialization.jsonObject(with: d) as? [String: Any])
            XCTAssertEqual(js["eos_token"] as? String, "<|im_end|>")
            XCTAssertEqual(js["bos_token"] as? String, "<|im_start|>")
            XCTAssertNil(js["pad_token"])
            XCTAssertNil(js["unk_token"])
            XCTAssertEqual(js["chat_template"] as? String, baseTpl.rawSource)
        }

        // 3. POST /tokenize accepts prompt or text + special token, and POST /detokenize round-trips.
        func post(_ path: String, body: Any) async throws -> (Int, [String: Any]) {
            var req = URLRequest(url: baseURL.appendingPathComponent(path))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let st = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let js = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (st, js)
        }

        let (tokSt1, tokJs1) = try await post("tokenize", body: ["prompt": "Hello", "add_special_tokens": false])
        XCTAssertEqual(tokSt1, 200)
        let helloTokens = try XCTUnwrap(tokJs1["tokens"] as? [Int])
        XCTAssertEqual(tokJs1["count"] as? Int, helloTokens.count)
        XCTAssertEqual(tokJs1["max_model_len"] as? Int, 4096)

        let (tokSt2, tokJs2) = try await post("v1/tokenize", body: ["text": "<|im_end|>"])
        XCTAssertEqual(tokSt2, 200)
        XCTAssertEqual(tokJs2["tokens"] as? [Int], [257])

        let (detSt, detJs) = try await post("detokenize", body: ["tokens": helloTokens])
        XCTAssertEqual(detSt, 200)
        XCTAssertEqual(detJs["prompt"] as? String, "Hello")

        // 4. Error handling: missing prompt/text, out-of-vocab token ID, malformed JSON -> HTTP 400.
        let (badTokSt, _) = try await post("tokenize", body: ["model": "test-model"])
        XCTAssertEqual(badTokSt, 400)
        let (badDetSt, _) = try await post("detokenize", body: ["tokens": [-1]])
        XCTAssertEqual(badDetSt, 400)
        let (malDetSt, _) = try await post("detokenize", body: ["model": "test-model"])
        XCTAssertEqual(malDetSt, 400)

        // 5. Fallback eos_token without tokenizer_config keys + non-finite logprobs clamping.
        let emptyCtxInfo = try await actor.handleTokenizerInfo()
        XCTAssertEqual(emptyCtxInfo.eosToken, "<|im_end|>")
        let fallbackActor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: baseTpl) }
        )
        let fbInfo = try await fallbackActor.handleTokenizerInfo()
        XCTAssertFalse(fbInfo.eosToken.isEmpty)

        let nonFiniteBackend = CompletionsNonFiniteBackend(config: config, tokenId: emitId)
        let nfActor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: nonFiniteBackend, tokenizer: tok, chatTemplate: tplWithCtx) }
        )
        let nfServer = QWHttpServer(actor: nfActor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try nfServer.start()
        defer { nfServer.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)
        let nfPort = try XCTUnwrap(nfServer.testBoundPort)
        var cmplReq = URLRequest(url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(nfPort)/v1/completions")))
        cmplReq.httpMethod = "POST"
        cmplReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        cmplReq.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model", "prompt": "Hi", "max_tokens": 1, "logprobs": 1
        ])
        let (nfData, nfResp) = try await URLSession.shared.data(for: cmplReq)
        XCTAssertEqual((nfResp as? HTTPURLResponse)?.statusCode, 200)
        let nfJs = try XCTUnwrap(try JSONSerialization.jsonObject(with: nfData) as? [String: Any])
        let lpObj = try XCTUnwrap((((nfJs["choices"] as? [[String: Any]])?.first)?["logprobs"] as? [String: Any]))
        let lps = try XCTUnwrap(lpObj["token_logprobs"] as? [Double])
        XCTAssertEqual(lps.first ?? 0, -9999.0, accuracy: 1e-3)
    }
}

private final class CompletionsNonFiniteBackend: QWModelBackend, @unchecked Sendable {
    let descriptor: QWModelDescriptor = .qwen38_27b
    var family: QWModelDescriptor.Family { descriptor.family }
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)? = nil
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState = QWPrefixCacheState(
        isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
    )
    let capabilities: Set<QWKnobCapability> = [.lsa]
    var supportsTokenConstraints: Bool { true }
    let tokenId: Int

    init(config: QWEngineLifecycleActor.Config, tokenId: Int) {
        self.tokenId = tokenId
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath, modelName: config.modelName,
            modelDescriptor: .qwen38_27b, headStorage: "HQ4_G32_M5", visionStatus: "none",
            contextBudget: config.maxContext, kvMode: config.kvMode, mtpDrafts: 0, mtpBackend: "none",
            gdnPrefillMode: "sequential", prefillScratchEviction: config.evictPrefillScratch,
            prefillChunkRows: config.prefillChunkSize, weightContainerNote: nil
        )
    }

    func generate(
        request: QWModelGenerationRequest, config: QWEngineLifecycleActor.Config, onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        let lp = QWTokenLogprob(
            tokenId: tokenId, logprob: -.infinity,
            topCandidates: [QWTokenLogprob.Candidate(tokenId: tokenId, logprob: .nan)]
        )
        request.onTokenLogprob?(lp)
        _ = onToken?(tokenId)
        return QWModelCompletionMetrics(
            tokens: [tokenId], prefillSec: 0.001, decodeSec: 0.001, firstTokenSec: 0.001,
            activeContextCapacity: parametersInfo.contextBudget, tokenLogprobs: [lp]
        )
    }

    func buildStandalonePrefixCache(
        tokens: [Int], outputURL: URL, pinAsSharedBase: Bool, onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int { 0 }
    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) { ([], 0) }
    func unmountSharedBasePrefixes() {}
}
