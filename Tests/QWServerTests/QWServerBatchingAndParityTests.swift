// QWServerBatchingAndParityTests.swift — CPU-only regression tests for C3.3 batching,
// telemetry non-blocking atomics, SSE/EOF cancellation, store caps, and protocol parity.

import Darwin
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

private final class ParityMockBackend: QWModelBackend, @unchecked Sendable {
    private let lock = NSLock()
    let descriptor: QWModelDescriptor
    var family: QWModelDescriptor.Family { descriptor.family }
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)?
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState = QWPrefixCacheState(
        isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
    )
    let capabilities: Set<QWKnobCapability> = [.lsa]
    var supportsTokenConstraints: Bool { true }

    private var _generateCount = 0
    private var _promptLogprobsCount = 0
    private var _capturedRequests: [QWModelGenerationRequest] = []
    private var _mountCount = 0
    var generateCount: Int { lock.withLock { _generateCount } }
    var promptLogprobsCount: Int { lock.withLock { _promptLogprobsCount } }
    var mountCount: Int { lock.withLock { _mountCount } }
    var capturedRequests: [QWModelGenerationRequest] { lock.withLock { _capturedRequests } }

    private let generator: @Sendable (QWModelGenerationRequest) throws -> [Int]
    private let stubTokens: [Int]
    private let sleepMicrosPerToken: UInt32
    private let onGenerateStart: (@Sendable () -> Void)?
    private let onGenerateCancelled: (@Sendable () -> Void)?

    init(
        config: QWEngineLifecycleActor.Config,
        descriptor: QWModelDescriptor = .qwen38_27b,
        headStorage: String = "HQ4_G32_M5",
        visionBackend: (any QWVisionBackend)? = nil,
        stubTokens: [Int] = Array(repeating: 4, count: 64),
        sleepMicrosPerToken: UInt32 = 0,
        onGenerateStart: (@Sendable () -> Void)? = nil,
        onGenerateCancelled: (@Sendable () -> Void)? = nil,
        generator: @escaping @Sendable (QWModelGenerationRequest) throws -> [Int]
    ) {
        self.descriptor = descriptor
        self.visionBackend = visionBackend
        self.stubTokens = stubTokens
        self.sleepMicrosPerToken = sleepMicrosPerToken
        self.onGenerateStart = onGenerateStart
        self.onGenerateCancelled = onGenerateCancelled
        self.generator = generator
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: descriptor,
            headStorage: headStorage,
            visionStatus: visionBackend == nil ? "none" : "loaded",
            contextBudget: config.maxContext,
            kvMode: config.kvMode,
            mtpDrafts: 0,
            mtpBackend: "none",
            gdnPrefillMode: "chunked",
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
        lock.withLock {
            _generateCount += 1
            _capturedRequests.append(request)
        }
        onGenerateStart?()
        let tokens = try generator(request)
        var emitted: [Int] = []
        for tok in tokens {
            emitted.append(tok)
            if let cont = onToken, !cont(tok) {
                onGenerateCancelled?()
                break
            }
            if sleepMicrosPerToken > 0 { usleep(sleepMicrosPerToken) }
        }
        return QWModelCompletionMetrics(
            tokens: emitted,
            prefillSec: 0.002,
            decodeSec: 0.002,
            firstTokenSec: 0.001,
            activeContextCapacity: config.maxContext,
            cacheHitTokens: request.usePrefixCache ? stubTokens.count : 0,
            cacheTier: request.usePrefixCache ? "warm_ram" : nil,
            cacheActive: request.usePrefixCache,
            warmCacheBytes: 4096,
            ssdCacheBytes: 0
        )
    }

    func computePromptLogprobs(
        promptTokens: [Int],
        topLogprobsCount: Int,
        sampling: QWSamplingParameters,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> [QWTokenLogprob]? {
        lock.withLock { _promptLogprobsCount += 1 }
        return promptTokens.enumerated().map { idx, tok in
            QWTokenLogprob(
                tokenId: tok,
                logprob: Float(-0.25 * Double(idx + 1)),
                topCandidates: (0..<max(1, topLogprobsCount)).map {
                    QWTokenLogprob.Candidate(tokenId: tok + $0, logprob: Float(-0.25 * Double($0 + 1)))
                }
            )
        }
    }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        lock.withLock { _mountCount += 1 }
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: 4096, ssdBytes: 0, warmEntries: 1, ssdEntries: 0
        )
        return (stubTokens, 4096)
    }

    func unmountSharedBasePrefixes() {
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
        )
    }
}

private final class SerialVisionMock: QWVisionBackend, @unchecked Sendable {
    private let lock = NSLock()
    var isLoaded: Bool = true
    private var _activeCount = 0
    private var _maxConcurrent = 0
    var maxConcurrent: Int { lock.withLock { _maxConcurrent } }

    func setVisionLoaded(_ loaded: Bool) throws {
        lock.withLock { isLoaded = loaded }
    }

    func prepareForPrompt(hasImages: Bool) throws {
        lock.withLock {
            _activeCount += 1
            if _activeCount > _maxConcurrent { _maxConcurrent = _activeCount }
        }
        usleep(10_000)
        lock.withLock { _activeCount -= 1 }
    }

    func encode(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) async throws -> [Float] {
        [Float](repeating: 0.1, count: grid.nToken * 8)
    }
}

final class QWServerBatchingAndParityTests: XCTestCase {
    private func makeByteTokenizerAndTemplate() throws -> (QWTokenizer, QWChatTemplate) {
        var vocab: [String: Int] = [:]
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = b
        }
        let tokObj: [String: Any] = [
            "model": ["type": "BPE", "byte_fallback": true, "vocab": vocab, "merges": [String]()],
            "pre_tokenizer": ["type": "Metaspace", "replacement": " ", "prepend_scheme": "never", "split": false],
            "added_tokens": [
                ["id": 256, "content": "<|im_start|>", "special": true],
                ["id": 257, "content": "<|im_end|>", "special": true],
                ["id": 258, "content": "<think>", "special": false],
                ["id": 259, "content": "</think>", "special": false],
                ["id": 260, "content": "<tool_call>", "special": false],
                ["id": 261, "content": "</tool_call>", "special": false]
            ]
        ]
        let tok = try QWTokenizer(json: try JSONSerialization.data(withJSONObject: tokObj))
        let tpl = try QWChatTemplate(
            source: "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}"
        )
        return (tok, tpl)
    }

    private func withTimeout<T: Sendable>(
        seconds: TimeInterval = 5.0,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw CancellationError()
            }
            guard let first = try await group.next() else { throw CancellationError() }
            group.cancelAll()
            return first
        }
    }

    func testChatMemoryPackAutoMountDoesNotSelfDeadlockAndDrainsInFlight() async throws {
        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-c33-mempack-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot.appendingPathComponent(".qwg"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let cacheFile = tempRoot.appendingPathComponent(".qwg/pack_test_q4.qwcache")
        try Data(count: 1024).write(to: cacheFile)
        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: "test-model",
            geometryDigest: QWModelDescriptor.qwen38_27b.geometryDigestV3Hex,
            kvMode: "q4",
            lsaMode: "auto",
            tokenCount: 64,
            cacheFileName: cacheFile.lastPathComponent,
            cacheFileBytes: 1024
        )
        try QWMemoryManifest(packName: "c33-repo", rootPath: tempRoot.path, files: [], caches: [artifact]).save(to: tempRoot)

        let (tok, tpl) = try makeByteTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B.qwp", modelName: "test-model", maxContext: 4096, kvMode: .f16, host: "127.0.0.1", port: 0
        )
        let backend = ParityMockBackend(config: config) { _ in try tok.encode("ok", addSpecialTokens: false) }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        actor.configureAllowedWorkspaceRoots([tempRoot.path])

        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data("""
        {"messages":[{"role":"user","content":"hello"}],"max_tokens":2,"metadata":{"workspace_path":"\(tempRoot.path)"}}
        """.utf8))
        let res = try await withTimeout(seconds: 5.0) {
            try await actor.handleChatCompletion(request: req)
        }
        XCTAssertEqual(res.content, "ok")
        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.packName, "c33-repo")
        XCTAssertEqual(actor.getConfig().kvMode, .q4)

        // Two concurrent auto-mount requests with same root mount only once.
        let countBefore = backend.mountCount
        _ = await actor.autoMountAndPrepareMemoryState(for: req)
        XCTAssertEqual(backend.mountCount, countBefore)
    }

    func testTelemetryAtomicsNonBlockingSamplerAndLifetimeCacheCounters() async {
        let registry = QWEngineTelemetryRegistry()
        registry.beginPrefill(promptTokens: 32)
        registry.beginDecode(promptTokens: 32)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    for _ in 0..<250 { registry.recordDecodeToken() }
                }
            }
        }
        registry.recordCacheLookup(hitTier: "warm_ram", matchedTokens: 16)
        registry.recordCacheLookup(hitTier: nil, matchedTokens: 0)
        registry.resetToIdle()

        let snapAfterRequest = registry.snapshot()
        XCTAssertEqual(snapAfterRequest.cache.lookupsTotal, 2, "resetToIdle() must preserve lifetime cache lookups")
        XCTAssertEqual(snapAfterRequest.cache.hitsWarmTotal, 1)
        XCTAssertEqual(snapAfterRequest.cache.missesTotal, 1)

        registry.resetToIdle(resetCacheCounters: true)
        let snapAfterUnload = registry.snapshot()
        XCTAssertEqual(snapAfterUnload.cache.lookupsTotal, 0)

        let fan = QWFanController()
        fan.setPolicy(.proactive)
        fan.updateInferencePhase("decode")
        XCTAssertEqual(fan.currentPolicy, .proactive)
        let fanSnap = fan.sampleAndReconcile()
        XCTAssertFalse(fanSnap.thermalStateLevel.isEmpty)

        let s1 = QWSystemMetricsSampler.shared.current()
        let s2 = QWSystemMetricsSampler.shared.current()
        XCTAssertEqual(s1.sampledAt, s2.sampledAt, "cached sampler returns without blocking")
    }

    func testSettingsManagerStatCacheInvalidatesOnDiskChange() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-settings-stat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("settings.json")

        let mgr = QWSettingsManager()
        var s1 = QWServerSettings()
        s1.port = 8123
        try mgr.saveSettings(s1, to: file)
        XCTAssertEqual(mgr.loadSettings(from: file).port, 8123)

        var s2 = s1
        s2.port = 9199
        let payload = try JSONEncoder().encode(s2)
        try payload.write(to: file, options: .atomic)
        XCTAssertEqual(mgr.loadSettings(from: file).port, 9199, "atomic overwrite changing inode/size must invalidate stat cache")
    }

    func testParallelOffActorTokenPrefixEncodingAndSinglePassEchoLogprobs() async throws {
        let (tok, tpl) = try makeByteTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp", modelName: "test-model", maxContext: 4096, host: "127.0.0.1", port: 0
        )
        let vision = SerialVisionMock()
        let backend = ParityMockBackend(config: config, visionBackend: vision) { _ in [65, 66] }
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )

        let wire = "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nPing<|im_end|>\n<|im_start|>assistant\n"
        try await withThrowingTaskGroup(of: Int?.self) { group in
            for i in 0..<4 {
                group.addTask {
                    let enc = try actor.encodeWireIncrementallyWithSystemPrefix(
                        wire: wire + " \(i)", tokenizer: tok, packPath: "/dummy/model.qwp", cachePrompt: true, cacheSalt: "s\(i)"
                    )
                    return enc.systemPrefixTokenCount
                }
            }
            for try await sysCount in group { XCTAssertNotNil(sysCount) }
        }

        let echoReq = try JSONDecoder().decode(CompletionRequest.self, from: Data("""
        {"model":"test-model","prompt":"ABCDEF","echo":true,"logprobs":2,"max_tokens":0}
        """.utf8))
        let res = try await actor.handleTextCompletion(request: echoReq)
        XCTAssertEqual(res.choices.first?.text, "ABCDEF")
        XCTAssertEqual(backend.promptLogprobsCount, 1)
        XCTAssertEqual(backend.generateCount, 0, "echo max_tokens=0 should not fall back to O(N) generate loop")
    }

    func testResponsesStoreByteCapPutIfAbsentAttachTaskRaceAndDuplicate409() async throws {
        let store = QWResponsesStore(capacity: 32, maxBytes: 2200)
        let dummyCreate = try JSONDecoder().decode(
            OpenAIResponseCreateRequest.self,
            from: Data(#"{"model":"m","input":"hi"}"#.utf8)
        )
        func makeEntry(_ id: String, body: String) -> QWResponsesStore.StoredResponse {
            let tmpl = OpenAIResponseTemplate(
                id: id, createdAt: 1, model: "m", req: dummyCreate,
                tools: [], toolChoice: .string("none"),
                resolvedTemperature: 0.0, resolvedTopP: 1.0,
                textFormat: OpenAIResponseResource.TextConfig()
            )
            let res = tmpl.makeResource(
                status: "completed", completedAt: 2,
                output: [.object([("text", .string(body))])], usage: nil
            )
            return QWResponsesStore.StoredResponse(resource: res, historyMessages: [])
        }
        let a = makeEntry("resp_a", body: String(repeating: "x", count: 900))
        let b = makeEntry("resp_b", body: String(repeating: "y", count: 900))
        let c = makeEntry("resp_c", body: String(repeating: "z", count: 900))
        XCTAssertTrue(store.putIfAbsent(a))
        XCTAssertFalse(store.putIfAbsent(a))
        store.put(b)
        store.put(c)
        XCTAssertLessThanOrEqual(store.count, 2)
        XCTAssertNil(store.get("resp_a"), "oldest entry must evict to satisfy maxBytes cap")
        XCTAssertNotNil(store.get("resp_c"))

        let lateAfterComplete = Task<Void, Never> { try? await Task.sleep(nanoseconds: 5_000_000_000) }
        defer { lateAfterComplete.cancel() }
        store.attachTask(id: "resp_c", task: lateAfterComplete)
        XCTAssertNil(store.get("resp_c")?.task, "completed entry must not retain late task reference")

        let deletedTask = Task<Void, Never> { try? await Task.sleep(nanoseconds: 5_000_000_000) }
        XCTAssertTrue(store.delete("resp_c"))
        store.attachTask(id: "resp_c", task: deletedTask)
        XCTAssertTrue(deletedTask.isCancelled, "attaching task after delete must cancel immediately")

        let (tok, tpl) = try makeByteTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/m.qwp", modelName: "test-model", host: "127.0.0.1", port: 0
        )
        let backend = ParityMockBackend(config: config) { _ in try tok.encode("hi", addSpecialTokens: false) }
        let actor = QWEngineLifecycleActor(
            config: config, logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 40_000_000)
        let port = try XCTUnwrap(server.testBoundPort)
        var req = URLRequest(url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/responses")))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("dup-req-id-1", forHTTPHeaderField: "x-request-id")
        req.httpBody = Data(#"{"model":"test-model","input":"hi","store":true}"#.utf8)
        let (_, r1) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((r1 as? HTTPURLResponse)?.statusCode, 200)
        let (_, r2) = try await URLSession.shared.data(for: req)
        XCTAssertEqual((r2 as? HTTPURLResponse)?.statusCode, 409)
    }

    func testToolCallParserErnieThinkWhitespacePayloadAndThinkingBudgetTrailingBytes() throws {
        let xmlPayload = "<ernie:think>step 1</ernie:think>  Hello <tool_call><function=write><parameter=code>return \"</parameter>\";</parameter></function></tool_call> Done"
        let parser = QWChatStreamParser(startsInReasoning: false, allowToolCalls: true, allowReasoningTags: true)
        var deltas = parser.feed(xmlPayload)
        deltas.append(contentsOf: parser.finish())
        let text = deltas.compactMap { if case .content(let s) = $0 { return s }; return nil }.joined()
        XCTAssertTrue(text.hasPrefix("  Hello"), "leading whitespace after think close must be preserved")
        XCTAssertTrue(text.contains("Done"), "parser must transition toolCall -> content cleanly")

        let full = QWToolCallParser.parse(text: xmlPayload, tools: nil, startsInThinking: false, truncated: false, allowToolCalls: true)
        XCTAssertEqual(full.toolCalls?.count, 1)
        XCTAssertTrue(full.toolCalls?.first?.function.arguments.contains("</parameter>") == true)

        // ThinkingBudgetConstraint forwards trailing bytes present on </think> token to inner constraint.
        let pieces: [String?] = ["{", "\"a\":1}", "}\n", "</think>{", "</think>", nil]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let schema = try QWJSONGrammar.compile(schema: try QWJson.parse(#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}"#))
        let inner = QWStructuredOutputConstraint(grammar: schema, table: table, stopTokens: [5], startsInThinking: true, endThinkToken: 4)
        let budget = QWThinkingBudgetConstraint(
            maxThinkingTokens: 5, endThinkTokenId: 4, vocabularySize: pieces.count, table: table, inner: inner
        )
        try budget.accept(3) // "</think>{" token forwards `{` into JSON grammar
        XCTAssertEqual(try budget.allowedTokens(), [1]) // only `"a":1}` completes `{`
        try budget.accept(1) // "\"a\":1}" completes the JSON object `{` + `"a":1}`
        XCTAssertTrue(inner.isComplete)
    }

    func testPassiveEOFWatcherCancelsPrefillFlagAndSSEWriterDrainsAfterCallerDrops() async throws {
        let flag = QWGenerationCancelFlag()
        let hub = QWConnectionCancellationHub()
        XCTAssertFalse(flag.isCancelled)
        hub.cancelFlag.cancel()
        XCTAssertTrue(hub.isTerminated)

        let prefillEntered = DispatchSemaphore(value: 0)
        let prefillCancelled = DispatchSemaphore(value: 0)
        let (tok, tpl) = try makeByteTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/m.qwp", modelName: "test-model", host: "127.0.0.1", port: 0
        )
        let backend = ParityMockBackend(
            config: config,
            sleepMicrosPerToken: 15_000,
            onGenerateStart: { prefillEntered.signal() },
            onGenerateCancelled: { prefillCancelled.signal() }
        ) { _ in
            Array(repeating: 65, count: 200)
        }
        let actor = QWEngineLifecycleActor(
            config: config, logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 40_000_000)
        let port = try XCTUnwrap(server.testBoundPort)

        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { Darwin.close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let connRes = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        XCTAssertEqual(connRes, 0)
        let body = Data(#"{"model":"test-model","messages":[{"role":"user","content":"wait"}],"max_tokens":200,"stream":false}"#.utf8)
        let reqBytes = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\n\r\n".utf8) + body
        _ = reqBytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return Darwin.send(fd, base, raw.count, 0)
        }
        XCTAssertEqual(prefillEntered.wait(timeout: .now() + 3.0), .success)
        Darwin.shutdown(fd, SHUT_WR)
        XCTAssertEqual(prefillCancelled.wait(timeout: .now() + 3.0), .success, "TCP EOF must cancel active generation even before SSE")
    }

    func testOllamaAndAnthropicParityFixes() async throws {
        let (tok, tpl) = try makeByteTokenizerAndTemplate()
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Llama-3-8B.qwp", modelName: "custom-llama", maxContext: 8192,
            host: "127.0.0.1", port: 0, enableThinking: false
        )
        let customCfgJSON: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "num_hidden_layers": 32,
            "max_position_embeddings": 16384,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 248044
        ]
        let customDesc = try QWModelDescriptor.decode(
            configJSON: try JSONSerialization.data(withJSONObject: customCfgJSON)
        )
        let backend = ParityMockBackend(config: config, descriptor: customDesc, headStorage: "HQ8_G32") { _ in
            try tok.encode("hello world", addSpecialTokens: false)
        }
        let actor = QWEngineLifecycleActor(
            config: config, logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: 0, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 40_000_000)
        let port = try XCTUnwrap(server.testBoundPort)
        let base = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)"))

        func jsonPost(_ path: String, _ json: String) -> URLRequest {
            var r = URLRequest(url: base.appendingPathComponent(path))
            r.httpMethod = "POST"
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = Data(json.utf8)
            return r
        }

        // Load the mock Llama descriptor before querying /api/show (dummy pack path has no disk config).
        _ = try await actor.loadExplicit()

        // 1. /api/show sources family & parameter_size from descriptor, not hardcoded qwen3/27B.
        let showReq = jsonPost("api/show", #"{"model":"custom-llama"}"#)
        let (showData, showResp) = try await URLSession.shared.data(for: showReq)
        XCTAssertEqual((showResp as? HTTPURLResponse)?.statusCode, 200)
        let showObj = try XCTUnwrap(try JSONSerialization.jsonObject(with: showData) as? [String: Any])
        let details = try XCTUnwrap(showObj["details"] as? [String: Any])
        XCTAssertEqual(details["family"] as? String, "llama")
        XCTAssertNotEqual(details["parameter_size"] as? String, "27B")

        // 2. /api/generate defaults to stream: true and emits NDJSON lines ending done:true.
        let genReq = jsonPost("api/generate", #"{"model":"custom-llama","prompt":"Hi","system":"Be brief"}"#)
        let (genData, genResp) = try await URLSession.shared.data(for: genReq)
        XCTAssertEqual((genResp as? HTTPURLResponse)?.statusCode, 200)
        let genLines = String(decoding: genData, as: UTF8.self).split(separator: "\n")
        XCTAssertGreaterThanOrEqual(genLines.count, 2)
        XCTAssertTrue(genLines.last?.contains(#""done":true"#) == true)

        // 3. /api/chat defers 200 headers so validation errors return HTTP 400 (not 200 + 500 JSON).
        let chatErrReq = jsonPost("api/chat", #"{"model":"custom-llama","messages":[{"role":"user","content":"Hi","images":["http://invalid.remote.test/img.png"]}]}"#)
        let (_, chatErrResp) = try await URLSession.shared.data(for: chatErrReq)
        XCTAssertEqual((chatErrResp as? HTTPURLResponse)?.statusCode, 400)

        // 4. /v1/messages rejects JSON boolean for integer max_tokens and emits non-zero input_tokens in message_start.
        let anthBad = jsonPost("v1/messages", #"{"model":"custom-llama","max_tokens":true,"messages":[{"role":"user","content":"hi"}]}"#)
        let (_, anthBadResp) = try await URLSession.shared.data(for: anthBad)
        XCTAssertEqual((anthBadResp as? HTTPURLResponse)?.statusCode, 400)

        let anthStream = jsonPost("v1/messages", #"{"model":"custom-llama","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hello world"}]}"#)
        let (anthData, anthResp) = try await URLSession.shared.data(for: anthStream)
        XCTAssertEqual((anthResp as? HTTPURLResponse)?.statusCode, 200)
        let anthText = String(decoding: anthData, as: UTF8.self)
        XCTAssertTrue(anthText.contains("event: message_start"))
        let startPayloadLine = anthText.split(separator: "\n").first { $0.hasPrefix("data: ") && $0.contains("message_start") }
        let startJsonStr = String((startPayloadLine ?? "").dropFirst(6))
        let startObj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(startJsonStr.utf8)) as? [String: Any])
        let msgObj = try XCTUnwrap(startObj["message"] as? [String: Any])
        let msgUsage = try XCTUnwrap(msgObj["usage"] as? [String: Any])
        XCTAssertGreaterThan((msgUsage["input_tokens"] as? Int) ?? 0, 0)
    }
}
