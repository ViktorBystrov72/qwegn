// QWResponsesEndpointTests.swift — Tests for OpenAI /v1/responses,
// GET/DELETE /v1/responses/:id, previous_response_id chaining, background mode, and SSE.

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

private final class ResponsesMockBackend: QWModelBackend, @unchecked Sendable {
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
        generator: @escaping @Sendable (QWModelGenerationRequest) throws -> [Int]
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
        let tokens = try generator(request)
        var emitted: [Int] = []
        emitted.reserveCapacity(tokens.count)
        for tok in tokens {
            emitted.append(tok)
            if let cont = onToken, !cont(tok) { break }
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

final class QWResponsesEndpointTests: XCTestCase {

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
                ["id": 257, "content": "<|im_end|>", "special": true],
                ["id": 258, "content": "<think>", "special": false],
                ["id": 259, "content": "</think>", "special": false],
                ["id": 260, "content": "<tool_call>", "special": false],
                ["id": 261, "content": "</tool_call>", "special": false]
            ]
        ]
        let tokData = try JSONSerialization.data(withJSONObject: tokObj)
        let tok = try QWTokenizer(json: tokData)
        let src = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            + "{% if m.tool_calls %}<tool_call>{{ m.tool_calls[0].function.name }}</tool_call>{% endif %}"
            + "<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}"
        return (tok, try QWChatTemplate(source: src))
    }

    private func parseSSEFrames(_ data: Data) -> [(event: String?, data: String)] {
        let raw = String(decoding: data, as: UTF8.self)
        let blocks = raw.components(separatedBy: "\n\n")
        var frames: [(event: String?, data: String)] = []
        for block in blocks where !block.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            var ev: String? = nil
            var lines: [String] = []
            for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
                let s = String(line)
                if s.hasPrefix("event:") {
                    ev = String(s.dropFirst(6)).trimmingCharacters(in: .whitespaces)
                } else if s.hasPrefix("data:") {
                    lines.append(String(s.dropFirst(5)).trimmingCharacters(in: .whitespaces))
                }
            }
            if !lines.isEmpty {
                frames.append((ev, lines.joined(separator: "\n")))
            }
        }
        return frames
    }

    func testEngineInfoAndFullResourceSchemaAndEffortNone() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: true
        )
        let backend = ResponsesMockBackend(config: config) { req in
            let promptStr = (try? tok.decode(req.promptTokens, skipSpecialTokens: false)) ?? ""
            if promptStr.contains("<think>") {
                return (try tok.encode("plan</think>{\"ok\":true}", addSpecialTokens: false))
            }
            return try tok.encode("{\"ok\":true}", addSpecialTokens: false)
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

        // 1. Engine info advertises responses and completions routes
        let (infoData, infoResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/engine/info"))
        XCTAssertEqual((infoResp as? HTTPURLResponse)?.statusCode, 200)
        let infoJson = try JSONSerialization.jsonObject(with: infoData) as? [String: Any]
        let endpoints = infoJson?["endpoints"] as? [String: String]
        XCTAssertEqual(endpoints?["responses"], "POST /v1/responses")
        XCTAssertEqual(endpoints?["completions"], "POST /v1/completions")

        // 2. Non-streaming /v1/responses with reasoning.effort == "none" and json_schema
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data("""
        {
          "model": "test-model",
          "instructions": "Respond tersely.",
          "input": "Say ok",
          "reasoning": {"effort": "none"},
          "text": {
            "format": {
              "type": "json_schema",
              "name": "ok_schema",
              "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
              "strict": true
            }
          }
        }
        """.utf8)
        let (data, resp) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((resp as? HTTPURLResponse)?.statusCode, 200)
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])

        // Check all 31 ResponseResource keys are encoded (even nullable ones as NSNull)
        let expectedKeys = [
            "id", "object", "created_at", "completed_at", "status", "incomplete_details",
            "model", "previous_response_id", "instructions", "output", "error", "tools",
            "tool_choice", "truncation", "parallel_tool_calls", "text", "top_p",
            "presence_penalty", "frequency_penalty", "top_logprobs", "temperature",
            "reasoning", "usage", "max_output_tokens", "max_tool_calls", "store",
            "background", "service_tier", "metadata", "safety_identifier", "prompt_cache_key"
        ]
        for key in expectedKeys {
            XCTAssertNotNil(obj[key], "ResponseResource missing key '\(key)'")
        }
        XCTAssertEqual(obj["object"] as? String, "response")
        XCTAssertEqual(obj["status"] as? String, "completed")
        XCTAssertEqual(obj["instructions"] as? String, "Respond tersely.")

        let output = try XCTUnwrap(obj["output"] as? [[String: Any]])
        XCTAssertFalse(output.contains { ($0["type"] as? String) == "reasoning" }, "effort: none must emit 0 reasoning items")
        let msgItem = try XCTUnwrap(output.first { ($0["type"] as? String) == "message" })
        let contentParts = try XCTUnwrap(msgItem["content"] as? [[String: Any]])
        XCTAssertEqual(contentParts.first?["type"] as? String, "output_text")
        XCTAssertEqual(contentParts.first?["text"] as? String, "{\"ok\":true}")

        let usage = try XCTUnwrap(obj["usage"] as? [String: Any])
        let outDetails = try XCTUnwrap(usage["output_tokens_details"] as? [String: Any])
        XCTAssertEqual(outDetails["reasoning_tokens"] as? Int, 0)
    }

    func testResponsesStreamingSSEAndToolCallsAndIncompleteTruncation() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let backend = ResponsesMockBackend(config: config) { req in
            let promptStr = (try? tok.decode(req.promptTokens, skipSpecialTokens: false)) ?? ""
            if promptStr.contains("TRUNCATE_ME") {
                return try tok.encode("alpha beta gamma delta epsilon zeta", addSpecialTokens: false)
            }
            if promptStr.contains("CALL_TWO_TOOLS") {
                let s = "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}\n</tool_call>"
                    + "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"b\"}}\n</tool_call>"
                return try tok.encode(s, addSpecialTokens: false)
            }
            return try tok.encode("thinking step</think>Final hello!", addSpecialTokens: false)
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

        // 1. Streaming with reasoning.effort = "medium"
        var streamReq = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        streamReq.httpMethod = "POST"
        streamReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        streamReq.httpBody = Data("""
        {"model":"test-model","input":"Explain","reasoning":{"effort":"medium"},"stream":true}
        """.utf8)
        let (sseData, sseResp) = try await URLSession.shared.data(for: streamReq)
        XCTAssertEqual((sseResp as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(((sseResp as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? "").contains("text/event-stream"))

        let frames = parseSSEFrames(sseData)
        let eventTypes = frames.compactMap { $0.event }
        XCTAssertEqual(eventTypes.first, "response.created")
        XCTAssertTrue(eventTypes.contains("response.in_progress"))
        XCTAssertTrue(eventTypes.contains("response.reasoning_summary_text.delta"))
        XCTAssertFalse(eventTypes.contains("response.reasoning_text.delta"))
        XCTAssertTrue(eventTypes.contains("response.output_text.delta"))
        XCTAssertTrue(eventTypes.contains("response.output_text.done"))
        XCTAssertEqual(eventTypes.last, "response.completed")
        XCTAssertEqual(frames.last?.data, "[DONE]")

        // Verify strictly increasing sequence_number and streamed item IDs match response.completed.output
        var prevSeq = -1
        var addedIDs: [String] = []
        var terminalOutputIDs: [String] = []
        for f in frames where f.event != nil {
            let d = try XCTUnwrap(f.data.data(using: .utf8))
            let obj = try JSONSerialization.jsonObject(with: d) as? [String: Any]
            let s = try XCTUnwrap(obj?["sequence_number"] as? Int)
            XCTAssertGreaterThan(s, prevSeq)
            prevSeq = s
            if f.event == "response.output_item.added",
               let item = obj?["item"] as? [String: Any],
               let itemId = item["id"] as? String {
                addedIDs.append(itemId)
            } else if f.event == "response.completed",
                      let res = obj?["response"] as? [String: Any],
                      let outItems = res["output"] as? [[String: Any]] {
                terminalOutputIDs = outItems.compactMap { $0["id"] as? String }
            }
        }
        XCTAssertEqual(addedIDs.count, 2)
        XCTAssertEqual(addedIDs, terminalOutputIDs, "streamed item_ids must match finalized output[*].id")

        // 2. Streaming with flat tool + parallel_tool_calls: false caps to 1 function_call
        var toolReq = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        toolReq.httpMethod = "POST"
        toolReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        toolReq.httpBody = Data("""
        {
          "model": "test-model",
          "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Use tools."}]},
            {"role": "user", "content": "CALL_TWO_TOOLS"}
          ],
          "tools": [
            {"type": "function", "name": "lookup", "description": "Find", "parameters": {"type": "object"}}
          ],
          "tool_choice": "required",
          "parallel_tool_calls": false,
          "stream": true
        }
        """.utf8)
        let (toolSSE, toolResp) = try await URLSession.shared.data(for: toolReq)
        XCTAssertEqual((toolResp as? HTTPURLResponse)?.statusCode, 200)
        let toolFrames = parseSSEFrames(toolSSE)
        let toolEvents = toolFrames.compactMap { $0.event }
        XCTAssertTrue(toolEvents.contains("response.function_call_arguments.delta"))
        XCTAssertTrue(toolEvents.contains("response.function_call_arguments.done"))
        let fcDoneCount = toolEvents.filter { $0 == "response.function_call_arguments.done" }.count
        XCTAssertEqual(fcDoneCount, 1, "parallel_tool_calls=false must cap emitted function calls to 1")
        XCTAssertEqual(toolFrames.last?.data, "[DONE]")
        let toolAddedID = toolFrames.first(where: { $0.event == "response.output_item.added" }).flatMap {
            (try? JSONSerialization.jsonObject(with: Data($0.data.utf8)) as? [String: Any])?["item"] as? [String: Any]
        }?["id"] as? String
        let toolTerminalID = (toolFrames.first(where: { $0.event == "response.completed" }).flatMap {
            ((try? JSONSerialization.jsonObject(with: Data($0.data.utf8)) as? [String: Any])?["response"] as? [String: Any])?["output"] as? [[String: Any]]
        })?.first?["id"] as? String
        XCTAssertNotNil(toolAddedID)
        XCTAssertEqual(toolAddedID, toolTerminalID)

        // 3. max_output_tokens truncation produces status: "incomplete" & incomplete_details
        var truncReq = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        truncReq.httpMethod = "POST"
        truncReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        truncReq.httpBody = Data(#"{"model":"test-model","input":"TRUNCATE_ME","max_output_tokens":4}"#.utf8)
        let (truncData, _) = try await URLSession.shared.data(for: truncReq)
        let truncObj = try XCTUnwrap(try JSONSerialization.jsonObject(with: truncData) as? [String: Any])
        XCTAssertEqual(truncObj["status"] as? String, "incomplete")
        XCTAssertEqual((truncObj["incomplete_details"] as? [String: Any])?["reason"] as? String, "max_output_tokens")
    }

    func testResponsesPreviousResponseIdChainingGetDeleteAndBackgroundMode() async throws {
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: false
        )
        let bgGate = DispatchSemaphore(value: 0)
        let backend = ResponsesMockBackend(config: config) { req in
            let promptStr = (try? tok.decode(req.promptTokens, skipSpecialTokens: false)) ?? ""
            if promptStr.contains("BG_JOB") {
                bgGate.wait()
                return try tok.encode("Background done", addSpecialTokens: false)
            }
            if promptStr.contains("Ada") && promptStr.contains("What is my name?") {
                return try tok.encode("Your name is Ada.", addSpecialTokens: false)
            }
            return try tok.encode("Nice to meet you, Ada.", addSpecialTokens: false)
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

        // 1. Turn 1 stored response
        var req1 = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        req1.httpMethod = "POST"
        req1.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req1.httpBody = Data("""
        {"model":"test-model","input":[{"role":"user","content":[{"type":"input_text","text":"My name is Ada."}]}]}
        """.utf8)
        let (d1, r1) = try await URLSession.shared.data(for: req1)
        XCTAssertEqual((r1 as? HTTPURLResponse)?.statusCode, 200)
        let obj1 = try XCTUnwrap(try JSONSerialization.jsonObject(with: d1) as? [String: Any])
        let id1 = try XCTUnwrap(obj1["id"] as? String)

        // 2. Turn 2 chaining previous_response_id + function_call_output item
        var req2 = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        req2.httpMethod = "POST"
        req2.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req2.httpBody = Data("""
        {
          "model": "test-model",
          "previous_response_id": "\(id1)",
          "input": [
            {"type": "function_call", "call_id": "call_1", "name": "ping", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "pong"},
            {"role": "user", "content": "What is my name?"}
          ]
        }
        """.utf8)
        let (d2, r2) = try await URLSession.shared.data(for: req2)
        XCTAssertEqual((r2 as? HTTPURLResponse)?.statusCode, 200)
        let obj2 = try XCTUnwrap(try JSONSerialization.jsonObject(with: d2) as? [String: Any])
        XCTAssertEqual(obj2["previous_response_id"] as? String, id1)
        let out2 = try XCTUnwrap(obj2["output"] as? [[String: Any]])
        let content2 = (out2.first?["content"] as? [[String: Any]])?.first?["text"] as? String
        XCTAssertEqual(content2, "Your name is Ada.")

        // 3. Unknown previous_response_id -> 400
        var reqBad = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        reqBad.httpMethod = "POST"
        reqBad.setValue("application/json", forHTTPHeaderField: "Content-Type")
        reqBad.httpBody = Data(#"{"model":"test-model","previous_response_id":"resp_missing","input":"hi"}"#.utf8)
        let (_, rBad) = try await URLSession.shared.data(for: reqBad)
        XCTAssertEqual((rBad as? HTTPURLResponse)?.statusCode, 400)

        // 4. GET and DELETE /v1/responses/:id
        let (gData, gResp) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/responses/\(id1)"))
        XCTAssertEqual((gResp as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual((try JSONSerialization.jsonObject(with: gData) as? [String: Any])?["id"] as? String, id1)

        var delReq = URLRequest(url: baseURL.appendingPathComponent("v1/responses/\(id1)"))
        delReq.httpMethod = "DELETE"
        let (delData, delResp) = try await URLSession.shared.data(for: delReq)
        XCTAssertEqual((delResp as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertEqual((try JSONSerialization.jsonObject(with: delData) as? [String: Any])?["deleted"] as? Bool, true)

        let (_, gAfterDel) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/responses/\(id1)"))
        XCTAssertEqual((gAfterDel as? HTTPURLResponse)?.statusCode, 404)

        // 5. background: true + store: false -> 400
        var bgBad = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        bgBad.httpMethod = "POST"
        bgBad.setValue("application/json", forHTTPHeaderField: "Content-Type")
        bgBad.httpBody = Data(#"{"model":"test-model","input":"hi","background":true,"store":false}"#.utf8)
        let (_, bgBadResp) = try await URLSession.shared.data(for: bgBad)
        XCTAssertEqual((bgBadResp as? HTTPURLResponse)?.statusCode, 400)

        // 6. background: true returns immediately queued, then completes once backend finishes
        var bgReq = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        bgReq.httpMethod = "POST"
        bgReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        bgReq.httpBody = Data(#"{"model":"test-model","input":"BG_JOB","background":true}"#.utf8)
        let (bgData, bgResp) = try await URLSession.shared.data(for: bgReq)
        XCTAssertEqual((bgResp as? HTTPURLResponse)?.statusCode, 200)
        let bgObj = try XCTUnwrap(try JSONSerialization.jsonObject(with: bgData) as? [String: Any])
        let bgId = try XCTUnwrap(bgObj["id"] as? String)
        XCTAssertEqual(bgObj["status"] as? String, "queued")

        bgGate.signal()
        var finalStatus = "queued"
        for _ in 0..<200 {
            try await Task.sleep(nanoseconds: 25_000_000)
            let (pollData, _) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("v1/responses/\(bgId)"))
            let pollObj = try JSONSerialization.jsonObject(with: pollData) as? [String: Any]
            finalStatus = (pollObj?["status"] as? String) ?? "queued"
            if finalStatus == "completed" { break }
        }
        XCTAssertEqual(finalStatus, "completed")
    }

    func testResponsesStoreLRUEviction() throws {
        let store = QWResponsesStore(capacity: 2)
        let dummyFormat: QWJson = .object([("type", .string("text"))])
        let decodedReq = try JSONDecoder().decode(OpenAIResponseCreateRequest.self, from: Data(#"{"input":"x"}"#.utf8))
        func entry(_ id: String) -> QWResponsesStore.StoredResponse {
            let tpl = OpenAIResponseTemplate(
                id: id,
                createdAt: 1,
                model: "m",
                req: decodedReq,
                tools: [],
                toolChoice: .string("auto"),
                resolvedTemperature: 0.7,
                resolvedTopP: 1.0,
                textFormat: OpenAIResponseResource.TextConfig(format: dummyFormat)
            )
            let res = tpl.makeResource(status: "completed", completedAt: 2, output: [], usage: nil)
            return QWResponsesStore.StoredResponse(resource: res, historyMessages: [])
        }
        store.put(entry("r1"))
        store.put(entry("r2"))
        _ = store.get("r1")
        store.put(entry("r3"))
        XCTAssertNotNil(store.get("r1"), "r1 was touched most recently")
        XCTAssertNil(store.get("r2"), "r2 should be evicted as LRU")
        XCTAssertNotNil(store.get("r3"))
    }

    func testResponsesStructuredEffortLowAndExplicitThinkingParity() async throws {
        // 1. Mapper preserves reasoning_effort without synthesizing enable_thinking: true.
        let lowSchemaJson = Data("""
        {
          "model": "test-model",
          "input": "Return JSON",
          "reasoning": {"effort": "low"},
          "text": {
            "format": {
              "type": "json_schema",
              "name": "marker_schema",
              "schema": {
                "type": "object",
                "properties": {"note": {"enum": ["one", "to=self<|message|>"]}},
                "required": ["note"],
                "additionalProperties": false
              },
              "strict": true
            }
          }
        }
        """.utf8)
        let decoded = try JSONDecoder().decode(OpenAIResponseCreateRequest.self, from: lowSchemaJson)
        let mapped = try OpenAIResponsesMapper.buildChatRequest(
            from: decoded,
            messages: [ChatMessage(role: "user", content: ChatMessageContent(string: "Return JSON"))],
            tools: nil, toolChoice: nil, headers: [:]
        )
        XCTAssertFalse(mapped.thinkingExplicitlyDisabled)
        XCTAssertNil(mapped.request.enableThinking)
        XCTAssertNil(mapped.request.reasoning)
        XCTAssertEqual(mapped.request.reasoningEffort, "low")
        XCTAssertFalse(mapped.request.effectiveEnableThinking(defaultValue: true))

        // 2. Live server parity across /v1/responses and /v1/chat/completions.
        let (tok, tpl) = try makeTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp",
            modelName: "test-model",
            maxContext: 4096,
            host: "127.0.0.1",
            port: 0,
            enableThinking: true
        )
        let markerPayload = #"{"note":"to=self<|message|>"}"#
        let draftThinkPayload = #"I noticed {"draft":1} first.\n</think>{"note":"ok"}"#
        let backend = ResponsesMockBackend(config: config) { genReq in
            if genReq.constraint is QWThinkingBudgetConstraint {
                return try tok.encode(draftThinkPayload, addSpecialTokens: false)
            }
            return try tok.encode(markerPayload, addSpecialTokens: false)
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

        // A) /v1/responses + reasoning.effort: "low" + json_schema emits clean content and zero reasoning item.
        var rLow = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        rLow.httpMethod = "POST"
        rLow.setValue("application/json", forHTTPHeaderField: "Content-Type")
        rLow.httpBody = lowSchemaJson
        let (dLow, respLow) = try await URLSession.shared.data(for: rLow)
        XCTAssertEqual((respLow as? HTTPURLResponse)?.statusCode, 200)
        let objLow = try XCTUnwrap(try JSONSerialization.jsonObject(with: dLow) as? [String: Any])
        let outLow = try XCTUnwrap(objLow["output"] as? [[String: Any]])
        XCTAssertFalse(outLow.contains { ($0["type"] as? String) == "reasoning" })
        let msgLow = try XCTUnwrap(outLow.first { ($0["type"] as? String) == "message" })
        let txtLow = ((msgLow["content"] as? [[String: Any]])?.first?["text"] as? String)
        XCTAssertEqual(txtLow, markerPayload)

        // B) /v1/responses + explicit enable_thinking: true + json_schema keeps reasoning & does not pre-slice {draft}.
        var rThink = URLRequest(url: baseURL.appendingPathComponent("v1/responses"))
        rThink.httpMethod = "POST"
        rThink.setValue("application/json", forHTTPHeaderField: "Content-Type")
        rThink.httpBody = Data("""
        {
          "model": "test-model",
          "input": "Return JSON",
          "enable_thinking": true,
          "max_output_tokens": 256,
          "text": {
            "format": {
              "type": "json_schema",
              "name": "ok_schema",
              "schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
              "strict": true
            }
          }
        }
        """.utf8)
        let (dThink, respThink) = try await URLSession.shared.data(for: rThink)
        XCTAssertEqual((respThink as? HTTPURLResponse)?.statusCode, 200)
        let objThink = try XCTUnwrap(try JSONSerialization.jsonObject(with: dThink) as? [String: Any])
        let outThink = try XCTUnwrap(objThink["output"] as? [[String: Any]])
        let rItem = try XCTUnwrap(outThink.first { ($0["type"] as? String) == "reasoning" })
        let summaryText = ((rItem["summary"] as? [[String: Any]])?.first?["text"] as? String) ?? ""
        XCTAssertTrue(summaryText.contains(#"{"draft":1}"#))
        let mItem = try XCTUnwrap(outThink.first { ($0["type"] as? String) == "message" })
        XCTAssertEqual((mItem["content"] as? [[String: Any]])?.first?["text"] as? String, #"{"note":"ok"}"#)

        // C) /v1/chat/completions + explicit enable_thinking: true + json_schema keeps reasoning_content cleanly.
        var cThink = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        cThink.httpMethod = "POST"
        cThink.setValue("application/json", forHTTPHeaderField: "Content-Type")
        cThink.httpBody = Data("""
        {
          "model": "test-model",
          "messages": [{"role": "user", "content": "Return JSON"}],
          "enable_thinking": true,
          "max_tokens": 256,
          "response_format": {
            "type": "json_schema",
            "json_schema": {
              "name": "ok_schema",
              "schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
              "strict": true
            }
          }
        }
        """.utf8)
        let (dChat, respChat) = try await URLSession.shared.data(for: cThink)
        XCTAssertEqual((respChat as? HTTPURLResponse)?.statusCode, 200)
        let objChat = try XCTUnwrap(try JSONSerialization.jsonObject(with: dChat) as? [String: Any])
        let msgChat = try XCTUnwrap(((objChat["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]))
        XCTAssertEqual(msgChat["content"] as? String, #"{"note":"ok"}"#)
        XCTAssertTrue((msgChat["reasoning_content"] as? String)?.contains(#"{"draft":1}"#) == true)

        // D) Streaming /v1/responses + explicit enable_thinking: true + json_schema emits clean output_text.
        var rStream = rThink
        rStream.httpBody = Data("""
        {
          "model": "test-model",
          "input": "Return JSON",
          "enable_thinking": true,
          "max_output_tokens": 256,
          "stream": true,
          "text": {
            "format": {
              "type": "json_schema",
              "name": "ok_schema",
              "schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
              "strict": true
            }
          }
        }
        """.utf8)
        let (dStream, respStream) = try await URLSession.shared.data(for: rStream)
        XCTAssertEqual((respStream as? HTTPURLResponse)?.statusCode, 200)
        let frames = parseSSEFrames(dStream)
        let completedFrame = try XCTUnwrap(frames.first { $0.event == "response.completed" })
        let cObj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(completedFrame.data.utf8)) as? [String: Any])
        let cOut = try XCTUnwrap((cObj["response"] as? [String: Any])?["output"] as? [[String: Any]])
        let cMsg = try XCTUnwrap(cOut.first { ($0["type"] as? String) == "message" })
        XCTAssertEqual((cMsg["content"] as? [[String: Any]])?.first?["text"] as? String, #"{"note":"ok"}"#)
        XCTAssertTrue(cOut.contains { ($0["type"] as? String) == "reasoning" })
    }
}
