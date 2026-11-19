// QWServerDeepeningTests.swift — Tests for QWSSESession, QWChatPolicyResolver,
// direct ChatCompletionRequest construction/adapters, and QWModelCatalog discovery.

import Foundation
import QWCore
import QWEngine
import QWHead
@testable import QWServer
import QWToken
import XCTest

private final class MockSSETransport: QWSSEFrameTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var _isClosed = false
    private(set) var sentFrames: [Data] = []
    private(set) var finishCalls: [Data?] = []

    var isClosed: Bool {
        get { lock.withLock { _isClosed } }
        set { lock.withLock { _isClosed = newValue } }
    }

    var combinedOutput: String {
        lock.withLock {
            let chunks = sentFrames + finishCalls.compactMap { $0 }
            return String(decoding: chunks.reduce(into: Data(), +=), as: UTF8.self)
        }
    }

    func sendSSEDataFrame<T: Encodable>(_ object: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let json = try? encoder.encode(object) else { return }
        var frame = Data("data: ".utf8)
        frame.append(json)
        frame.append(Data("\n\n".utf8))
        lock.withLock { sentFrames.append(frame) }
    }

    func sendSSENamedEvent<T: Encodable>(event: String, object: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let json = try? encoder.encode(object) else { return }
        var frame = Data("event: \(event)\ndata: ".utf8)
        frame.append(json)
        frame.append(Data("\n\n".utf8))
        lock.withLock { sentFrames.append(frame) }
    }

    func sendSSEJsonObject(event: String?, fields: [String: Any]) {
        guard let json = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]) else { return }
        var frame = Data()
        if let event {
            frame.append(Data("event: \(event)\n".utf8))
        }
        frame.append(Data("data: ".utf8))
        frame.append(json)
        frame.append(Data("\n\n".utf8))
        lock.withLock { sentFrames.append(frame) }
    }

    func sendNDJSONLine<T: Encodable>(_ object: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let json = try? encoder.encode(object) else { return }
        var frame = json
        frame.append(Data("\n".utf8))
        lock.withLock { sentFrames.append(frame) }
    }

    func sendRaw(_ data: Data) {
        lock.withLock { sentFrames.append(data) }
    }

    func finish(with finalData: Data?) {
        lock.withLock { finishCalls.append(finalData) }
    }

    func snapshotFrames() -> [Data] {
        lock.withLock { sentFrames }
    }
}

final class QWServerDeepeningTests: XCTestCase {
    private func makeTempDir(prefix: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try data.write(to: url)
    }

    private func samplePrimaryConfig(hidden: Int = 5120, layers: Int = 64) -> [String: Any] {
        let layerTypes = (0..<layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let text: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": hidden,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_hidden_layers": layers,
            "layer_types": layerTypes,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": ["rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]],
            "attn_output_gate": true,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
            "eos_token_id": 248044,
        ]
        return ["model_type": "qwen3_5", "text_config": text]
    }

    func testSSESessionHeaderDeferralFormatsAndTerminalFraming() {
        let cancelFlag = QWGenerationCancelFlag()
        let sseTransport = MockSSETransport()
        let sseSession = QWSSESession(
            format: .sse,
            extraHeaders: ["x-request-id: req-1", "x-session-id: sess-9"],
            corsHeaders: "Access-Control-Allow-Origin: http://127.0.0.1\r\n",
            keepaliveIntervalNs: nil,
            transport: sseTransport,
            stopChecker: { cancelFlag.isCancelled }
        )

        XCTAssertFalse(sseSession.isStarted)
        XCTAssertTrue(sseTransport.snapshotFrames().isEmpty, "Pre-stream failures before begin() must emit zero headers")

        sseSession.begin()
        XCTAssertTrue(sseSession.isStarted)
        sseSession.begin()
        XCTAssertEqual(sseTransport.snapshotFrames().count, 1, "begin() must emit HTTP 200 headers only once")

        let headerText = String(decoding: sseTransport.snapshotFrames()[0], as: UTF8.self)
        XCTAssertTrue(headerText.contains("HTTP/1.1 200 OK\r\n"))
        XCTAssertTrue(headerText.contains("Content-Type: text/event-stream\r\n"))
        XCTAssertTrue(headerText.contains("x-request-id: req-1\r\n"))
        XCTAssertTrue(headerText.contains("x-session-id: sess-9\r\n"))
        XCTAssertTrue(headerText.contains("Access-Control-Allow-Origin: http://127.0.0.1\r\n"))

        sseSession.sendSSENamedEvent(event: "message_start", object: ["status": "ok"])
        sseSession.finish(doneMarker: true)
        sseSession.finishWithOpenAIError(NSError(domain: "test", code: 2))

        XCTAssertEqual(sseTransport.finishCalls.count, 1, "finish calls after completion must be ignored")
        let wire = sseTransport.combinedOutput
        XCTAssertTrue(wire.contains("event: message_start\ndata: {\"status\":\"ok\"}\n\n"))
        XCTAssertTrue(wire.hasSuffix("data: [DONE]\n\n"))

        let ndjsonTransport = MockSSETransport()
        let ndjsonSession = QWSSESession(
            format: .ndjson,
            keepaliveIntervalNs: nil,
            transport: ndjsonTransport,
            stopChecker: { false }
        )
        ndjsonSession.begin()
        ndjsonSession.sendNDJSONLine(["done": false])
        ndjsonSession.finishWithNDJSONError(NSError(domain: "stream", code: 1, userInfo: [NSLocalizedDescriptionKey: "boom"]))
        let ndWire = ndjsonTransport.combinedOutput
        XCTAssertTrue(ndWire.contains("Content-Type: application/x-ndjson\r\n"))
        XCTAssertTrue(ndWire.contains("{\"done\":false}\n"))
        XCTAssertTrue(ndWire.contains("\"error\":\""))
    }

    func testSSESessionKeepaliveStopsOnFirstTokenAndHonorsCancel() async throws {
        let cancelFlag = QWGenerationCancelFlag()
        let transport = MockSSETransport()
        let session = QWSSESession(
            format: .sse,
            keepaliveIntervalNs: 10_000_000,
            transport: transport,
            stopChecker: { cancelFlag.isCancelled }
        )
        defer { session.stopAll() }

        session.begin()
        try await Task.sleep(nanoseconds: 65_000_000)
        session.stopKeepaliveIfRunning()
        let countAtStop = transport.snapshotFrames().count
        XCTAssertGreaterThan(countAtStop, 1, "Idle stream should emit at least one : keepalive comment")
        XCTAssertTrue(transport.combinedOutput.contains(": keepalive\n\n"))

        try await Task.sleep(nanoseconds: 45_000_000)
        XCTAssertEqual(transport.snapshotFrames().count, countAtStop, "No keepalives allowed after first token stops gate")

        XCTAssertFalse(session.shouldStop)
        cancelFlag.cancel()
        XCTAssertTrue(session.shouldStop)
        XCTAssertTrue(session.isClosed)

        session.sendSSEDataFrame(["ignored": true])
        XCTAssertEqual(transport.snapshotFrames().count, countAtStop, "Frames after cancellation must be dropped")
    }

    func testSSESessionErrorFramersForOpenAIAndNamedEvents() {
        let openAITransport = MockSSETransport()
        let openAISession = QWSSESession(
            format: .sse,
            keepaliveIntervalNs: nil,
            transport: openAITransport,
            stopChecker: { false }
        )
        openAISession.begin()
        openAISession.finishWithOpenAIError(CompletionInvalidInputError("bad runtime"))
        let openAIWire = openAITransport.combinedOutput
        XCTAssertTrue(openAIWire.contains("\"message\":\"bad runtime\""))
        XCTAssertTrue(openAIWire.contains("\"type\":\"server_error\""))
        XCTAssertTrue(openAIWire.hasSuffix("data: [DONE]\n\n"))

        let namedTransport = MockSSETransport()
        let namedSession = QWSSESession(
            format: .sse,
            keepaliveIntervalNs: nil,
            transport: namedTransport,
            stopChecker: { false }
        )
        namedSession.begin()
        namedSession.finishWithNamedSSEError(
            event: "response.failed",
            fields: ["type": "response.failed", "message": "failed"],
            doneMarker: false
        )
        let namedWire = namedTransport.combinedOutput
        XCTAssertTrue(namedWire.contains("event: response.failed\n"))
        XCTAssertFalse(namedWire.contains("[DONE]"))
    }

    func testChatPolicyResolverThinkingStreamConstraintAndSampling() throws {
        XCTAssertEqual(QWChatPolicyResolver.normalizeReasoningEffort("HIGH"), "xhigh")
        XCTAssertEqual(QWChatPolicyResolver.normalizeReasoningEffort("Normal"), "medium")
        XCTAssertEqual(QWChatPolicyResolver.normalizeReasoningEffort("off"), "low")
        XCTAssertEqual(QWChatPolicyResolver.normalizeReasoningEffort("unknown", fallback: "med"), "medium")

        let jsonNoThinkReq = ChatCompletionRequest(
            messages: [ChatMessage(role: "user", text: "hi")],
            responseFormat: .object([("type", .string("json_object"))]),
            maxTokens: 1024
        )
        XCTAssertFalse(QWChatPolicyResolver.effectiveEnableThinking(for: jsonNoThinkReq, defaultValue: true))

        let effortMedReq = ChatCompletionRequest(
            messages: [ChatMessage(role: "user", text: "solve")],
            tools: [.object([("type", .string("function"))])],
            responseFormat: .object([("type", .string("json_schema"))]),
            maxTokens: 4096,
            reasoningEffort: "medium"
        )
        let jsonSuppressesEffort = QWChatPolicyResolver.resolveThinkingPolicy(
            for: effortMedReq,
            defaultEnableThinking: true,
            defaultMaxTokens: 512
        )
        XCTAssertEqual(jsonSuppressesEffort, QWChatThinkingPolicy(shouldThink: false, thinkingBudget: nil))

        let effortMedPlainReq = ChatCompletionRequest(
            messages: [ChatMessage(role: "user", text: "solve")],
            tools: [.object([("type", .string("function"))])],
            responseFormat: .object([("type", .string("json_schema"))]),
            maxTokens: 4096,
            enableThinking: true,
            reasoningEffort: "medium"
        )
        let medThinking = QWChatPolicyResolver.resolveThinkingPolicy(
            for: effortMedPlainReq,
            defaultEnableThinking: false,
            defaultMaxTokens: 512
        )
        XCTAssertEqual(medThinking, QWChatThinkingPolicy(shouldThink: true, thinkingBudget: 1024))

        let streamUnsupported = QWChatPolicyResolver.resolveStreamPolicy(
            for: effortMedPlainReq,
            defaultEnableThinking: true,
            defaultMaxTokens: 512,
            supportsThinkingTags: false
        )
        XCTAssertTrue(streamUnsupported.hasActiveTools)
        XCTAssertTrue(streamUnsupported.allowToolCalls)
        XCTAssertFalse(streamUnsupported.isJsonResponseFormat, "Active tools disable standalone JSON constraint mode")
        XCTAssertFalse(streamUnsupported.shouldThink)
        XCTAssertNil(streamUnsupported.thinkingBudget)

        let streamDisabled = QWChatPolicyResolver.resolveStreamPolicy(
            for: effortMedPlainReq,
            defaultEnableThinking: true,
            defaultMaxTokens: 512,
            supportsThinkingTags: true,
            thinkingExplicitlyDisabled: true
        )
        XCTAssertFalse(streamDisabled.shouldThink)
        XCTAssertFalse(streamDisabled.allowReasoningTags)

        let sampleCalls = [
            ToolCall(id: "1", function: ToolCallFunction(name: "a", arguments: "{}")),
            ToolCall(id: "2", function: ToolCallFunction(name: "b", arguments: "{}"))
        ]
        XCTAssertEqual(QWChatPolicyResolver.capToolCalls(sampleCalls, parallelToolCalls: false)?.count, 1)
        XCTAssertNil(QWChatPolicyResolver.capToolCalls(sampleCalls, parallelToolCalls: true, maxToolCalls: 0))

        let config = QWEngineLifecycleActor.Config(
            packPath: "/tmp/model",
            maxContext: 2048,
            kvMode: .q8,
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            maxTokens: 512,
            repetitionPenalty: 1.1,
            seed: -1,
            enableThinking: true,
            sourceSummary: ["sampling": "model default"]
        )
        XCTAssertThrowsError(try QWChatPolicyResolver.resolveChatBudget(
            for: effortMedReq,
            promptTokenCount: 2048,
            config: config,
            backendContextBudget: 1024
        )) { err in
            XCTAssertTrue(err is QWServerContextLengthError)
        }
        let budget = try QWChatPolicyResolver.resolveChatBudget(
            for: effortMedReq,
            promptTokenCount: 800,
            config: config,
            backendContextBudget: 1024
        )
        XCTAssertEqual(budget, QWChatBudgetResolution(effectiveMaxContext: 1024, requestedMaxNew: 4096, clampedMaxNew: 224))

        let defaultSampling = QWChatPolicyResolver.resolveSampling(for: jsonNoThinkReq, config: config)
        XCTAssertFalse(defaultSampling.hasRequestOverride)
        let overriddenReq = ChatCompletionRequest(
            messages: [ChatMessage(role: "user", text: "x")],
            temperature: 0.25,
            seed: 42,
            cacheSalt: "salt-a"
        )
        let overriddenSampling = QWChatPolicyResolver.resolveSampling(for: overriddenReq, config: config)
        XCTAssertTrue(overriddenSampling.hasRequestOverride)
        XCTAssertEqual(overriddenSampling.sourceLabel, "request override")
        XCTAssertEqual(Double(overriddenSampling.parameters.temperature), 0.25, accuracy: 1e-4)

        let cacheRetained = QWChatPolicyResolver.resolveCacheAndSpeculation(
            for: overriddenReq,
            systemPrefixTokenCount: 256,
            firstVisualTokenOffset: 512,
            isConstrainedJson: false,
            hasExplicitSeed: true
        )
        XCTAssertEqual(cacheRetained, QWChatCachePolicy(
            usePrefixCache: true, cacheSalt: "salt-a", retainPrefixLength: 256, exactSpeculation: true
        ))
        let cacheTooShort = QWChatPolicyResolver.resolveCacheAndSpeculation(
            for: overriddenReq,
            systemPrefixTokenCount: 120,
            firstVisualTokenOffset: 512,
            isConstrainedJson: false,
            hasExplicitSeed: false
        )
        XCTAssertNil(cacheTooShort.retainPrefixLength)
        XCTAssertFalse(cacheTooShort.exactSpeculation)
    }

    func testDirectChatRequestInitializerAndProtocolAdapters() throws {
        let directReq = ChatCompletionRequest(
            model: "qwen3",
            messages: [ChatMessage(role: "user", text: "hello")],
            parallelToolCalls: false,
            stream: true,
            maxTokens: 256,
            temperature: 0.3,
            topP: 0.85,
            topK: 15,
            repetitionPenalty: 1.05,
            seed: 7,
            stop: ["END"],
            enableThinking: true,
            reasoningEffort: "medium",
            reasoning: ChatCompletionRequest.ReasoningConfig(enabled: true, effort: "medium"),
            reasoningBudget: 512,
            chatTemplateKwargs: ChatCompletionRequest.ChatTemplateKwargs(enableThinking: true, reasoningEffort: "medium"),
            metadata: ["preset": "coding", "exact_speculation": "true"],
            user: "alice",
            sessionId: "sess-1",
            promptCacheKey: "pk-1",
            cacheSalt: "cs-1",
            cachePrompt: false,
            store: true,
            httpHeaders: ["x-request-id": "req-direct"]
        )
        XCTAssertEqual(directReq.effectiveMaxTokens, 256)
        XCTAssertEqual(directReq.effectiveCachePrompt, false)
        XCTAssertEqual(directReq.effectiveCacheSalt, "cs-1")
        XCTAssertTrue(directReq.effectiveExactSpeculation)
        XCTAssertEqual(directReq.effectiveRequestId, "req-direct")
        XCTAssertEqual(directReq.effectivePresetId, "coding")

        let anthropicJSON = #"""
        {
            "model": "claude-alias",
            "max_tokens": 300,
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"},
                    {"type": "text", "text": "Summarize"}
                ]}
            ],
            "tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "read_file", "disable_parallel_tool_use": true},
            "thinking": {"type": "enabled", "budget_tokens": 128},
            "stop_sequences": ["\nHuman:"],
            "metadata": {"memory_pack": "true"}
        }
        """#
        let anthropicReq = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(anthropicJSON.utf8))
        let (chatFromAnthropic, anthThinking) = try anthropicReq.toChatCompletionRequest(
            replyModel: "reply-model",
            requireMaxTokens: true,
            headers: ["x-session-id": "s-anth"]
        )
        XCTAssertTrue(anthThinking)
        XCTAssertEqual(chatFromAnthropic.model, "reply-model")
        XCTAssertEqual(chatFromAnthropic.maxTokens, 300)
        XCTAssertEqual(chatFromAnthropic.reasoningBudget, 128)
        XCTAssertEqual(chatFromAnthropic.parallelToolCalls, false)
        XCTAssertEqual(chatFromAnthropic.tools?.count, 1)
        XCTAssertEqual(chatFromAnthropic.messages.map(\.role), ["system", "assistant", "tool", "user"])
        XCTAssertEqual(chatFromAnthropic.messages[1].reasoningContent, "plan")
        XCTAssertEqual(chatFromAnthropic.messages[1].toolCalls?.first?.function.name, "read_file")

        let ollamaChatJSON = #"""
        {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "look", "images": ["aW1n"]}],
            "options": {"num_predict": 96, "temperature": 0.2, "top_p": 0.8, "top_k": 12, "repeat_penalty": 1.15, "seed": 9, "stop": "STOP"},
            "format": "json"
        }
        """#
        let ollamaChat = try JSONDecoder().decode(OllamaChatRequest.self, from: Data(ollamaChatJSON.utf8))
        let chatFromOllama = ollamaChat.toChatCompletionRequest()
        XCTAssertEqual(chatFromOllama.maxTokens, 96)
        XCTAssertEqual(chatFromOllama.seed, 9)
        XCTAssertEqual(chatFromOllama.stop, ["STOP"])
        XCTAssertEqual(chatFromOllama.responseFormatType, "json_object")
        XCTAssertEqual(chatFromOllama.extractImageUrls().first, "data:image/jpeg;base64,aW1n")

        let ollamaGenJSON = #"""
        {"model": "qwen3", "prompt": "hi", "system": "sys", "stream": false, "options": {"num_predict": 42}}
        """#
        let ollamaGen = try JSONDecoder().decode(OllamaGenerateRequest.self, from: Data(ollamaGenJSON.utf8))
        let chatFromGen = ollamaGen.toChatCompletionRequest(promptText: "hi")
        XCTAssertEqual(chatFromGen.messages.map(\.role), ["system", "user"])
        XCTAssertEqual(chatFromGen.isStreaming, false)
        XCTAssertEqual(chatFromGen.maxTokens, 42)

        let responsesJSON = #"""
        {
            "model": "qwen3",
            "input": [{"role": "user", "content": "emit json"}],
            "max_output_tokens": 200,
            "reasoning": {"effort": "none"},
            "safety_identifier": "user-9",
            "text": {"format": {"type": "json_schema", "name": "ans", "schema": {"type": "object"}, "strict": true}}
        }
        """#
        let resReq = try JSONDecoder().decode(OpenAIResponseCreateRequest.self, from: Data(responsesJSON.utf8))
        let resMsgs = try OpenAIResponsesMapper.parseInput(resReq.input, hasPrevious: false)
        let (chatFromResponses, disabled) = try OpenAIResponsesMapper.buildChatRequest(
            from: resReq, messages: resMsgs, tools: nil, toolChoice: nil, headers: [:]
        )
        XCTAssertTrue(disabled)
        XCTAssertEqual(chatFromResponses.enableThinking, false)
        XCTAssertEqual(chatFromResponses.maxTokens, 200)
        XCTAssertEqual(chatFromResponses.user, "user-9")
        XCTAssertEqual(chatFromResponses.responseFormatType, "json_schema")
    }

    func testUnifiedModelCatalogPrimaryAndDrafterDiscovery() throws {
        let root = try makeTempDir(prefix: "qwg-catalog-deepen")
        defer { try? FileManager.default.removeItem(at: root) }

        let localModels = root.appendingPathComponent("local-models", isDirectory: true)
        let hfHub = root.appendingPathComponent("hf-hub", isDirectory: true)

        let localPrimary = localModels.appendingPathComponent("shared-model", isDirectory: true)
        try writeJSON(samplePrimaryConfig(),
                      to: localPrimary.appendingPathComponent("assets/config.json"))

        let hubSharedSnapshot = hfHub
            .appendingPathComponent("models--org--shared-model/snapshots/rev1", isDirectory: true)
        try writeJSON(samplePrimaryConfig(),
                      to: hubSharedSnapshot.appendingPathComponent("config.json"))

        let hubExtraSnapshot = hfHub
            .appendingPathComponent("models--org--hub-only-qwen/snapshots/rev2", isDirectory: true)
        try writeJSON(samplePrimaryConfig(),
                      to: hubExtraSnapshot.appendingPathComponent("config.json"))

        let primaryDesc = try XCTUnwrap(QWModelCatalog.loadPrimaryDescriptor(at: localPrimary.path))
        XCTAssertEqual(
            QWServerSettings.loadPrimaryDescriptor(packPath: localPrimary.path)?.vocab,
            primaryDesc.vocab
        )

        let drafterDir = localModels.appendingPathComponent("my-dflash2-drafter", isDirectory: true)
        try writeJSON([
            "model_type": "dflash",
            "hidden_size": primaryDesc.hidden,
            "vocab_size": primaryDesc.vocab,
            "num_hidden_layers": 2,
            "dflash_config": [
                "target_layer_ids": [0, max(0, primaryDesc.numLayers - 1)],
                "num_target_layers": primaryDesc.numLayers,
                "block_size": 12
            ]
        ], to: drafterDir.appendingPathComponent("config.json"))

        let entries = QWModelCatalog.scanInstalledPrimaryModels(
            localDirectory: localModels.path,
            hfHubDirectory: hfHub.path,
            createLocalDirectoryIfMissing: false
        )
        let names = entries.map(\.name)
        XCTAssertTrue(names.contains("shared-model"))
        XCTAssertTrue(names.contains(where: { $0.contains("hub-only-qwen") }))
        let localMatch = entries.first(where: { $0.name == "shared-model" })
        XCTAssertEqual(localMatch?.path, localPrimary.path, "Local ~/.qwg/models pack must win on name collision")

        XCTAssertFalse(QWModelCatalog.isDrafterPack(at: localPrimary.path))
        XCTAssertTrue(QWModelCatalog.isDrafterPack(at: drafterDir.path))
        let drafterMeta = try XCTUnwrap(QWModelCatalog.drafterMetadata(at: drafterDir.path))
        XCTAssertEqual(drafterMeta.backend, "dflash2")
        XCTAssertEqual(drafterMeta.maxDrafts, 11)
        XCTAssertTrue(drafterMeta.isCompatible(with: primaryDesc))
        XCTAssertTrue(QWServerSettings.isDrafterCompatible(metadata: drafterMeta, with: primaryDesc))

        let catalogDrafters = QWModelCatalog.scanDrafters(in: localModels.path)
        let settingsDrafters = QWServerSettings.scanDrafterModelsDirectory(in: localModels.path)
        XCTAssertEqual(catalogDrafters.count, 1)
        XCTAssertEqual(catalogDrafters.first?.path, drafterDir.path)
        XCTAssertEqual(settingsDrafters.map(\.packPath), catalogDrafters.map(\.path))
    }
}
