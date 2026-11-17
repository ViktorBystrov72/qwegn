// QWContextErrorAndThinkingBudgetTests.swift — Tests for QWThinkingBudgetConstraint,
// implicit/explicit thinking budgets, pre-stream HTTP 400 context validation, and maxNew clamping.

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

private final class ContextAndBudgetMockBackend: QWModelBackend, @unchecked Sendable {
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

    private let generator: @Sendable (QWModelGenerationRequest) throws -> [Int]

    init(
        config: QWEngineLifecycleActor.Config,
        contextBudget: Int,
        generator: @escaping @Sendable (QWModelGenerationRequest) throws -> [Int]
    ) {
        self.generator = generator
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: .qwen38_27b,
            headStorage: "HQ4_G32_M5",
            visionStatus: "none",
            contextBudget: contextBudget,
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
        let tokens = try generator(request)
        var emitted: [Int] = []
        emitted.reserveCapacity(tokens.count)
        for tok in tokens {
            emitted.append(tok)
            if let cont = onToken, !cont(tok) {
                break
            }
        }
        return QWModelCompletionMetrics(
            tokens: emitted,
            prefillSec: 0.001,
            decodeSec: 0.001,
            firstTokenSec: 0.001,
            activeContextCapacity: parametersInfo.contextBudget
        )
    }

    func buildStandalonePrefixCache(
        tokens: [Int], outputURL: URL, pinAsSharedBase: Bool,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int { 0 }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) { ([], 0) }
    func unmountSharedBasePrefixes() {}
}

final class QWContextErrorAndThinkingBudgetTests: XCTestCase {

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

    private func makeTokenizerAndTemplate(enableThinkingTemplate: Bool = true) throws -> (QWTokenizer, QWChatTemplate) {
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
                ["id": 257, "content": "<|im_end|>", "special": true],
                ["id": 258, "content": "<think>", "special": false],
                ["id": 259, "content": "</think>", "special": false],
                ["id": 260, "content": "<tool_call>", "special": false],
                ["id": 261, "content": "</tool_call>", "special": false]
            ]
        ]
        let tokData = try JSONSerialization.data(withJSONObject: tokObj)
        let tok = try QWTokenizer(json: tokData)
        let source = enableThinkingTemplate
            ? "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}"
            : "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
        return (tok, try QWChatTemplate(source: source))
    }

    private func makeConstraintTable() -> QWJSONTokenTable {
        let pieces: [String?] = [
            "{", "}", "\"", "a", "b", "\":", "\"a\"", ":", ",", " ", "1", "x\"}", "true", "</think>", "ab", nil
        ]
        return QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: 16)
    }

    func testThinkingBudgetConstraintForcesEndThinkAndComposesJSON() throws {
        let table = makeConstraintTable()
        let schema = try QWJSONGrammar.compile(schema: try QWJson.parse("""
        {"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}
        """))
        let inner = QWStructuredOutputConstraint(
            grammar: schema, table: table, stopTokens: [15], startsInThinking: true, endThinkToken: 13
        )
        let c = QWThinkingBudgetConstraint(
            maxThinkingTokens: 3, endThinkTokenId: 13, vocabularySize: 16, table: table, inner: inner
        )

        XCTAssertTrue(c.isInThinking)
        XCTAssertEqual(c.thinkingTokenCount, 0)
        XCTAssertEqual(c.allowsSpeculation, inner.allowsSpeculation, "mirrors inner while thinking within budget")
        XCTAssertFalse(c.isUnconstrainedNow, "within 16 tokens of budget disables unconstrained spec")
        XCTAssertEqual(c.speculationBoundaryToken, 13)

        let freeInner = QWThinkingBudgetConstraint(
            maxThinkingTokens: 2, endThinkTokenId: 13, vocabularySize: 16, table: table, inner: nil
        )
        XCTAssertTrue(freeInner.allowsSpeculation)
        try freeInner.accept(3)
        try freeInner.accept(3)
        XCTAssertFalse(freeInner.allowsSpeculation, "must block speculation once forcing </think>")
        try freeInner.accept(13)
        XCTAssertTrue(freeInner.allowsSpeculation)

        for step in 0..<3 {
            XCTAssertNil(try c.allowedTokens(), "free thinking at step \(step)")
            try c.accept(3)
            XCTAssertEqual(c.thinkingTokenCount, step + 1)
        }

        XCTAssertTrue(c.isInThinking)
        XCTAssertFalse(c.allowsSpeculation)
        XCTAssertEqual(try c.allowedTokens(), [13], "must force </think> once budget exhausted")

        try c.accept(13)
        XCTAssertFalse(c.isInThinking)
        XCTAssertEqual(try c.allowedTokens(), [0, 9], "delegates to JSON constraint after </think>")

        try c.accept(0)
        try c.accept(6)
        try c.accept(7)
        try c.accept(2)
        try c.accept(11)
        XCTAssertTrue(inner.isComplete)
        XCTAssertEqual(try c.allowedTokens(), [15])

        let plainInner = QWStructuredOutputConstraint(grammar: schema, table: table, stopTokens: [15])
        let plainWrapped = QWThinkingBudgetConstraint(
            maxThinkingTokens: 1, endThinkTokenId: 13, vocabularySize: 16, table: table,
            inner: plainInner, forwardEndThinkToInner: false
        )
        try plainWrapped.accept(3)
        XCTAssertEqual(try plainWrapped.allowedTokens(), [13])
        try plainWrapped.accept(13)
        XCTAssertEqual(try plainWrapped.allowedTokens(), [0, 9])
    }

    func testThinkingBudgetConstraintDetectsMultiTokenEndThinkSequence() throws {
        let pieces: [String?] = ["</", "think", ">", "</think>", "{", "}", "\"", nil]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let schema = try QWJSONGrammar.compile(schema: try QWJson.parse(#"{"type":"object","additionalProperties":false}"#))
        let inner = QWStructuredOutputConstraint(
            grammar: schema, table: table, stopTokens: [7], startsInThinking: true, endThinkToken: 3
        )
        let c = QWThinkingBudgetConstraint(
            maxThinkingTokens: 20, endThinkTokenId: 3, vocabularySize: pieces.count, table: table, inner: inner
        )

        XCTAssertTrue(c.isUnconstrainedNow, "budget 20 > 16 starts unconstrained")
        for _ in 0..<4 { try c.accept(0) }
        XCTAssertFalse(c.isUnconstrainedNow, "within 16 tokens of budget disables unconstrained fast-path")

        let c2 = QWThinkingBudgetConstraint(
            maxThinkingTokens: 10, endThinkTokenId: 3, vocabularySize: pieces.count, table: table, inner: inner
        )
        try c2.accept(0)
        try c2.accept(1)
        XCTAssertTrue(c2.isInThinking)
        try c2.accept(2)
        XCTAssertFalse(c2.isInThinking, "multi-token </ + think + > exits thinking")
        XCTAssertEqual(try c2.allowedTokens(), [4])
    }

    func testThinkingBudgetConstraintComposesWithToolCallConstraint() throws {
        let pieces: [String?] = [
            "<tool_call>", "</tool_call>", "\n", "<function", "=read>", "</think>", "reason"
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}
        """)
        let toolConstraint = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: nil, table: table, toolCallToken: 0, toolCallEndToken: 1
        ))
        let c = QWThinkingBudgetConstraint(
            maxThinkingTokens: 2, endThinkTokenId: 5, vocabularySize: pieces.count, table: table, inner: toolConstraint
        )
        try c.accept(6)
        try c.accept(6)
        XCTAssertEqual(try c.allowedTokens(), [5])
        try c.accept(5)
        XCTAssertFalse(c.isInThinking)

        try c.accept(0)
        try c.accept(2)
        try c.accept(3)
        XCTAssertEqual(try c.allowedTokens(), [4], "inner tool call constraint governs after </think>")
    }

    func testResolvedThinkingPolicyImplicitVsExplicit() throws {
        let implicit32 = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}"#.utf8)
        )
        let p32 = implicit32.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertFalse(p32.shouldThink)
        XCTAssertNil(p32.thinkingBudget)

        let implicit64 = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}"#.utf8)
        )
        let p64 = implicit64.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(p64.shouldThink)
        XCTAssertEqual(p64.thinkingBudget, 48)

        let implicit256 = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256}"#.utf8)
        )
        let p256 = implicit256.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(p256.shouldThink)
        XCTAssertEqual(p256.thinkingBudget, 192)

        let implicit2048 = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2048}"#.utf8)
        )
        let p2048 = implicit2048.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(p2048.shouldThink)
        XCTAssertEqual(p2048.thinkingBudget, 1792)

        let explicitEnable = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"enable_thinking":true}"#.utf8)
        )
        let pExplicitEnable = explicitEnable.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(pExplicitEnable.shouldThink)
        XCTAssertNil(pExplicitEnable.thinkingBudget)

        let explicitBudget = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"reasoning_budget":20}"#.utf8)
        )
        let pExplicitBudget = explicitBudget.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(pExplicitBudget.shouldThink)
        XCTAssertEqual(pExplicitBudget.thinkingBudget, 20)

        let effortNone = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"reasoning_effort":"none"}"#.utf8)
        )
        let pNone = effortNone.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertFalse(pNone.shouldThink)
        XCTAssertNil(pNone.thinkingBudget)
    }

    func testImplicitAndAnthropicThinkingBudgetEndToEndOverHTTP() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate(enableThinkingTemplate: true)
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: true
        )
        let thoughtTok = try tok.encode("t", addSpecialTokens: false)[0]
        let endThinkId = try XCTUnwrap(tok.tokenId(for: "</think>"))
        let answerTokens = try tok.encode("Answer", addSpecialTokens: false)

        let backend = ContextAndBudgetMockBackend(config: config, contextBudget: 4096) { req in
            var out: [Int] = []
            if let constraint = req.constraint {
                for _ in 0..<250 {
                    if let allowed = try constraint.allowedTokens(), allowed == [endThinkId] {
                        try constraint.accept(endThinkId)
                        out.append(endThinkId)
                        break
                    }
                    try constraint.accept(thoughtTok)
                    out.append(thoughtTok)
                }
            }
            out.append(contentsOf: answerTokens)
            return out
        }

        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl)
            }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        // 1. max_tokens=32 with implicit thinking -> disabled
        var req32 = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        req32.httpMethod = "POST"
        req32.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req32.httpBody = Data(#"{"model":"test-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}"#.utf8)
        let (data32, resp32) = try await URLSession.shared.data(for: req32)
        XCTAssertEqual((resp32 as? HTTPURLResponse)?.statusCode, 200)
        let json32 = try JSONSerialization.jsonObject(with: data32) as? [String: Any]
        let choices32 = json32?["choices"] as? [[String: Any]]
        let msg32 = choices32?.first?["message"] as? [String: Any]
        XCTAssertEqual(msg32?["content"] as? String, "Answer")
        XCTAssertNil(msg32?["reasoning_content"] as? String)
        XCTAssertNil(backend.capturedRequests.last?.constraint)

        // 2. max_tokens=64 with implicit thinking -> budget 48, forces </think> then answers
        var req64 = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        req64.httpMethod = "POST"
        req64.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req64.httpBody = Data(#"{"model":"test-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}"#.utf8)
        let (data64, resp64) = try await URLSession.shared.data(for: req64)
        XCTAssertEqual((resp64 as? HTTPURLResponse)?.statusCode, 200)
        let json64 = try JSONSerialization.jsonObject(with: data64) as? [String: Any]
        let msg64 = (json64?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]
        XCTAssertEqual(msg64?["content"] as? String, "Answer")
        XCTAssertEqual(msg64?["reasoning_content"] as? String, String(repeating: "t", count: 48))

        // 3. Anthropic /v1/messages thinking.budget_tokens=20
        var reqAnt = URLRequest(url: baseURL.appendingPathComponent("v1/messages"))
        reqAnt.httpMethod = "POST"
        reqAnt.setValue("application/json", forHTTPHeaderField: "Content-Type")
        reqAnt.httpBody = Data("""
        {"model":"test-model","max_tokens":128,"thinking":{"type":"enabled","budget_tokens":20},"messages":[{"role":"user","content":"Hi"}]}
        """.utf8)
        let (dataAnt, respAnt) = try await URLSession.shared.data(for: reqAnt)
        XCTAssertEqual((respAnt as? HTTPURLResponse)?.statusCode, 200)
        let jsonAnt = try JSONSerialization.jsonObject(with: dataAnt) as? [String: Any]
        let blocks = jsonAnt?["content"] as? [[String: Any]]
        XCTAssertEqual(blocks?.first?["type"] as? String, "thinking")
        XCTAssertEqual(blocks?.first?["thinking"] as? String, String(repeating: "t", count: 20))
        XCTAssertEqual(blocks?.last?["type"] as? String, "text")
        XCTAssertEqual(blocks?.last?["text"] as? String, "Answer")
    }

    func testPreStreamContextValidationReturnsHTTP400AndClampsMaxNew() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate(enableThinkingTemplate: false)
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let outTokens = try tok.encode("OK", addSpecialTokens: false)
        // contextBudget=48 < config.maxContext=4096 tests min(config.maxContext, contextBudget)
        let backend = ContextAndBudgetMockBackend(config: config, contextBudget: 48) { req in
            Array(repeating: outTokens.first ?? 65, count: req.maxNewTokens)
        }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl)
            }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)

        let port = try XCTUnwrap(server.testBoundPort)
        let baseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))
        let longText = String(repeating: "overflow ", count: 12)

        func postJSON(_ path: String, body: [String: Any]) async throws -> (Int, String, [String: Any]) {
            var req = URLRequest(url: baseURL.appendingPathComponent(path))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            let http = try XCTUnwrap(resp as? HTTPURLResponse)
            let ct = http.value(forHTTPHeaderField: "Content-Type") ?? ""
            let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return (http.statusCode, ct, obj)
        }

        // Streaming and non-streaming must both fail with HTTP 400 (not 200 SSE)
        for streamFlag in [false, true] {
            let (stChat, ctChat, jsChat) = try await postJSON("v1/chat/completions", body: [
                "model": "test-model",
                "messages": [["role": "user", "content": longText]],
                "stream": streamFlag
            ])
            XCTAssertEqual(stChat, 400)
            XCTAssertTrue(ctChat.contains("application/json"))
            XCTAssertEqual((jsChat["error"] as? [String: Any])?["code"] as? String, "context_length_exceeded")

            let (stResp, ctResp, jsResp) = try await postJSON("v1/responses", body: [
                "model": "test-model", "input": longText, "stream": streamFlag
            ])
            XCTAssertEqual(stResp, 400)
            XCTAssertTrue(ctResp.contains("application/json"))
            XCTAssertEqual((jsResp["error"] as? [String: Any])?["code"] as? String, "context_length_exceeded")

            let (stCmpl, ctCmpl, jsCmpl) = try await postJSON("v1/completions", body: [
                "model": "test-model", "prompt": longText, "stream": streamFlag
            ])
            XCTAssertEqual(stCmpl, 400)
            XCTAssertTrue(ctCmpl.contains("application/json"))
            XCTAssertEqual((jsCmpl["error"] as? [String: Any])?["code"] as? String, "context_length_exceeded")

            let (stAnt, ctAnt, jsAnt) = try await postJSON("v1/messages", body: [
                "model": "test-model",
                "max_tokens": 16,
                "messages": [["role": "user", "content": longText]],
                "stream": streamFlag
            ])
            XCTAssertEqual(stAnt, 400)
            XCTAssertTrue(ctAnt.contains("application/json"))
            XCTAssertEqual(jsAnt["type"] as? String, "error")
            XCTAssertEqual((jsAnt["error"] as? [String: Any])?["type"] as? String, "invalid_request_error")
        }

        // Prompt < 48 with prompt + max_tokens > 48 clamps maxNew to (48 - promptTokens) -> finish_reason: "length"
        let (stOk, _, jsOk) = try await postJSON("v1/chat/completions", body: [
            "model": "test-model",
            "messages": [["role": "user", "content": "hi"]],
            "max_tokens": 100
        ])
        XCTAssertEqual(stOk, 200)
        let lastReq = try XCTUnwrap(backend.capturedRequests.last)
        XCTAssertEqual(lastReq.maxNewTokens, 48 - lastReq.promptTokens.count)
        let choiceOk = (jsOk["choices"] as? [[String: Any]])?.first
        XCTAssertEqual(choiceOk?["finish_reason"] as? String, "length")
    }

    func testDeep4096ByteTokenTrieInTaskTokenBytesAndNonThinkingTemplateGating() async throws {
        let longBytes = [UInt8](repeating: UInt8(ascii: "a"), count: 4096)
        let tableBytes: [[UInt8]?] = [
            Array("{".utf8), Array("}".utf8), Array("\"".utf8), Array(":".utf8),
            Array("k".utf8), longBytes, Array("</body>".utf8)
        ]
        // Run iterative trie build + collect inside a Swift Task (512 KiB worker stack).
        let collectedCount = try await Task.detached { () -> Int in
            let tbl = QWJSONTokenTable(tokenBytes: tableBytes, vocabulary: tableBytes.count)
            let schema = try QWJSONGrammar.compile(schema: try QWJson.parse(#"{"type":"object","properties":{"k":{"type":"string"}},"required":["k"]}"#))
            let c = QWStructuredOutputConstraint(grammar: schema, table: tbl, stopTokens: [])
            _ = try c.allowedTokens()
            return tbl.plainStringTokens.count + (tbl.fullTrie.tokens.isEmpty ? 0 : 1)
        }.value
        XCTAssertGreaterThan(collectedCount, 0)

        let (tok, thinkTpl) = try makeTokenizerAndTemplate(enableThinkingTemplate: true)
        XCTAssertTrue(thinkTpl.supportsThinkingTags)
        let ernieTpl = try QWChatTemplate(source: "{% for m in messages %}{{ m.content }}{% endfor %}<ernie:think>")
        XCTAssertTrue(ernieTpl.supportsThinkingTags)
        let (nonThinkTok, plainTpl) = try makeTokenizerAndTemplate(enableThinkingTemplate: false)
        XCTAssertFalse(plainTpl.supportsThinkingTags)

        let bytes1 = QWStructuredOutput.tokenBytes(for: nonThinkTok, vocabulary: 300)
        let bytes2 = QWStructuredOutput.tokenBytes(for: nonThinkTok, vocabulary: 300)
        XCTAssertEqual(bytes1.count, 300)
        XCTAssertEqual(bytes1.count, bytes2.count)
        let endThinkId = try XCTUnwrap(nonThinkTok.tokenId(for: "</think>"))
        let tTok = try nonThinkTok.encode("t", addSpecialTokens: false)[0]
        let budget = QWThinkingBudgetConstraint(
            maxThinkingTokens: 1, endThinkTokenId: endThinkId, vocabularySize: 300, tokenBytes: bytes1)
        XCTAssertNil(try budget.allowedTokens())
        try budget.accept(tTok)
        XCTAssertEqual(try budget.allowedTokens(), [endThinkId])

        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/ERNIE-5.5.qwp", modelName: "ernie-test",
            maxContext: 4096, host: "127.0.0.1", port: 0, enableThinking: true
        )
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(#"{"model":"ernie-test","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}"#.utf8))
        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: req, config: cfg, chatTemplate: plainTpl,
            tokenizer: nonThinkTok, vocabularySize: 300, supportsTokenConstraints: true)
        XCTAssertFalse(prepared.shouldThink)
        XCTAssertNil(prepared.activeConstraint)

        let ansTokens = try tok.encode("PlainAnswer", addSpecialTokens: false)
        let backend = ContextAndBudgetMockBackend(config: cfg, contextBudget: 4096) { genReq in
            XCTAssertNil(genReq.constraint)
            return ansTokens
        }
        let actor = QWEngineLifecycleActor(
            config: cfg, logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: nonThinkTok, chatTemplate: plainTpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 50_000_000)
        let port = try XCTUnwrap(server.testBoundPort)
        var httpReq = URLRequest(url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")))
        httpReq.httpMethod = "POST"
        httpReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpReq.httpBody = Data(#"{"model":"ernie-test","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}"#.utf8)
        let (respData, httpResp) = try await URLSession.shared.data(for: httpReq)
        XCTAssertEqual((httpResp as? HTTPURLResponse)?.statusCode, 200)
        let obj = try JSONSerialization.jsonObject(with: respData) as? [String: Any]
        let msg = (obj?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]
        XCTAssertEqual(msg?["content"] as? String, "PlainAnswer")
        XCTAssertNil(msg?["reasoning_content"] as? String)
    }

    func testResolvedThinkingPolicyCapsExplicitThinkingForJsonFormat() throws {
        let explicitJson = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"enable_thinking":true,"response_format":{"type":"json_schema"}}"#.utf8)
        )
        let pJson = explicitJson.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(pJson.shouldThink)
        XCTAssertEqual(pJson.thinkingBudget, 192)

        let tinyJson = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"enable_thinking":true,"response_format":{"type":"json_object"}}"#.utf8)
        )
        let pTiny = tinyJson.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(pTiny.shouldThink)
        XCTAssertEqual(pTiny.thinkingBudget, 16)

        let explicitJsonBudget = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"enable_thinking":true,"reasoning_budget":24,"response_format":{"type":"json_schema"}}"#.utf8)
        )
        let pBudget = explicitJsonBudget.resolvedThinkingPolicy(defaultEnableThinking: true, defaultMaxTokens: 512)
        XCTAssertTrue(pBudget.shouldThink)
        XCTAssertEqual(pBudget.thinkingBudget, 24)
    }
}
