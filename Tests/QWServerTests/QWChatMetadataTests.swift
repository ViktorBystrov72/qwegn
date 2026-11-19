// QWChatMetadataTests.swift — Tests for Chat-to-Engine request metadata schema,
// prefix cache bypass/salting, .qwcache per-turn routing, Request Ledger, and WebUI helpers.

import Foundation
import JavaScriptCore
import QWCore
@testable import QWEngine
import QWHAL
import QWKV
import QWToken
import QWVision
import XCTest
@testable import QWServer

private final class CapturingMetadataBackend: QWModelBackend, @unchecked Sendable {
    let descriptor: QWModelDescriptor = .qwen38_27b
    var family: QWModelDescriptor.Family { descriptor.family }
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)? = nil
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState = QWPrefixCacheState(
        isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
    )
    let capabilities: Set<QWKnobCapability> = [.lsa]
    var capturedRequests: [QWModelGenerationRequest] = []
    let stubTokens: [Int]
    let stubBytes: Int

    init(config: QWEngineLifecycleActor.Config, stubTokens: [Int], stubBytes: Int) {
        self.stubTokens = stubTokens
        self.stubBytes = stubBytes
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: config.packPath,
            modelName: config.modelName,
            modelDescriptor: .qwen38_27b,
            headStorage: "HQ4_G32_M5",
            visionStatus: "none",
            contextBudget: config.maxContext,
            kvMode: config.kvMode,
            mtpDrafts: 0,
            mtpBackend: "variational",
            gdnPrefillMode: "chunked",
            prefillScratchEviction: "auto",
            prefillChunkRows: 512,
            weightContainerNote: nil
        )
    }

    func generate(
        request: QWModelGenerationRequest,
        config: QWEngineLifecycleActor.Config,
        onToken: ((Int) -> Bool)?
    ) throws -> QWModelCompletionMetrics {
        capturedRequests.append(request)
        return QWModelCompletionMetrics(
            tokens: [1],
            prefillSec: 0.01,
            decodeSec: 0.01,
            firstTokenSec: 0.01,
            activeContextCapacity: config.maxContext,
            cacheHitTokens: request.usePrefixCache ? stubTokens.count : 0,
            cacheTier: request.usePrefixCache ? "warm_ram" : nil,
            cacheActive: request.usePrefixCache,
            warmCacheBytes: stubBytes,
            ssdCacheBytes: 0
        )
    }

    func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: stubBytes, ssdBytes: 0, warmEntries: 1, ssdEntries: 0
        )
        return (stubTokens, stubBytes)
    }

    func unmountSharedBasePrefixes() {
        prefixCache = QWPrefixCacheState(
            isConfigured: true, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0
        )
    }
}

final class QWChatMetadataTests: XCTestCase {
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

    private func makeMinimalTokenizerAndTemplate() throws -> (QWTokenizer, QWChatTemplate) {
        var vocab: [String: Int] = ["<|im_start|>": 0, "<|im_end|>": 1]
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = 2 + b
        }
        let root: [String: Any] = [
            "version": "1.0",
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": false]
                ]
            ],
            "model": [
                "type": "BPE",
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [] as [String]
            ],
            "added_tokens": [
                ["id": 0, "content": "<|im_start|>", "special": true],
                ["id": 1, "content": "<|im_end|>", "special": true]
            ]
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: root)
        let tok = try QWTokenizer(json: vocabData)
        let tpl = try QWChatTemplate(
            source: "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
        )
        return (tok, tpl)
    }

    func testSchemaDecodesCoercesAndValidatesMetadata() throws {
        let json = """
        {
          "model": "qwegn",
          "messages": [{"role": "user", "content": "hi"}],
          "user": "alice_01",
          "session_id": "sess-99",
          "prompt_cache_key": "cache-primary",
          "cache_salt": "salt-override",
          "cache_prompt": false,
          "store": false,
          "metadata": {
            "use_memory_pack": false,
            "attempt": 3,
            "weight": 0.75,
            "workspace_path": "/tmp/qwg-ws",
            "preset_id": "coding-agent",
            "exact_speculation": true
          }
        }
        """
        var req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        req.httpHeaders = ["x-request-id": "req-hdr-123", "x-session-id": "sess-hdr"]
        try req.validateMetadata()

        XCTAssertEqual(req.effectiveRequestId, "req-hdr-123")
        XCTAssertEqual(req.effectiveSessionId, "sess-99")
        XCTAssertEqual(req.effectiveUser, "alice_01")
        XCTAssertEqual(req.effectivePresetId, "coding-agent")
        XCTAssertFalse(req.effectiveCachePrompt)
        XCTAssertEqual(req.effectiveCacheSalt, "salt-override")
        XCTAssertFalse(req.effectiveUseMemoryPack)
        XCTAssertEqual(req.effectiveWorkspacePath, "/tmp/qwg-ws")
        XCTAssertFalse(req.effectiveStore)
        XCTAssertTrue(req.effectiveExactSpeculation)
        XCTAssertEqual(req.metadataValue(for: "attempt"), "3")
        XCTAssertEqual(req.metadataValue(for: "weight"), "0.75")

        let headerFallbackJSON = """
        {
          "messages": [{"role": "user", "content": "hi"}],
          "prompt_cache_key": "pk-1"
        }
        """
        var fallback = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(headerFallbackJSON.utf8))
        fallback.httpHeaders = ["x-session-id": "hdr-session-7"]
        XCTAssertEqual(fallback.effectiveSessionId, "hdr-session-7")
        XCTAssertEqual(fallback.effectiveCacheSalt, "pk-1")
        XCTAssertTrue(fallback.effectiveCachePrompt)
        XCTAssertTrue(fallback.effectiveStore)
        XCTAssertTrue(fallback.effectiveUseMemoryPack)
        XCTAssertFalse(fallback.effectiveExactSpeculation)

        // Nested metadata values are rejected at decode time.
        let nestedJSON = """
        {"messages":[{"role":"user","content":"x"}],"metadata":{"bad":[1,2]}}
        """
        XCTAssertThrowsError(try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(nestedJSON.utf8)))

        // Bounds checks (>16 keys, key >64, value >512).
        let manyKeys = (0..<17).map { "\"k\($0)\":\"v\"" }.joined(separator: ",")
        let tooManyReq = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"metadata\":{\(manyKeys)}}".utf8)
        )
        XCTAssertThrowsError(try tooManyReq.validateMetadata())

        let longKey = String(repeating: "k", count: 65)
        let longKeyReq = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"metadata\":{\"\(longKey)\":\"v\"}}".utf8)
        )
        XCTAssertThrowsError(try longKeyReq.validateMetadata())

        let longVal = String(repeating: "v", count: 513)
        let longValReq = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"metadata\":{\"ok\":\"\(longVal)\"}}".utf8)
        )
        XCTAssertThrowsError(try longValReq.validateMetadata())
    }

    func testActorRoutesCacheSaltMemoryPackBypassExactSpecAndLedger() async throws {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("qwg_meta_repo_\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        let qwgDir = QWMemoryManifest.qwgDirectoryURL(in: tempRoot)
        try fm.createDirectory(at: qwgDir, withIntermediateDirectories: true)
        let cacheFile = qwgDir.appendingPathComponent("meta_test_q4.qwcache")
        try Data([0x51, 0x57, 0x43, 0x48]).write(to: cacheFile)

        let artifact = QWMemoryManifest.CacheArtifact(
            modelName: "test-model",
            geometryDigest: QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
            kvMode: "q4",
            lsaMode: "auto",
            tokenCount: 512,
            cacheFileName: cacheFile.lastPathComponent,
            cacheFileBytes: 1_048_576
        )
        let manifest = QWMemoryManifest(packName: "meta-repo", rootPath: tempRoot.path, files: [], caches: [artifact])
        try manifest.save(to: tempRoot)

        let (tok, tpl) = try makeMinimalTokenizerAndTemplate()
        let stubTokens = Array(repeating: 4, count: 512)
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/Qwen3.8-27B-hq4g32.qwp",
            modelName: "test-model",
            maxContext: 4096,
            kvMode: .f16,
            host: "127.0.0.1",
            port: 0
        )
        let backend = CapturingMetadataBackend(config: config, stubTokens: stubTokens, stubBytes: 1_048_576)
        let actor = QWEngineLifecycleActor(
            config: config,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in
                QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl)
            }
        )
        actor.configureAllowedWorkspaceRoots([tempRoot.path])

        let req1Id = "meta-req-1-\(UUID().uuidString.prefix(8))"
        let req1JSON = """
        {
          "messages": [{"role": "user", "content": "Hello workspace"}],
          "max_tokens": 1,
          "session_id": "sess-meta-1",
          "prompt_cache_key": "ns-alpha",
          "store": true,
          "metadata": {
            "workspace_path": "\(tempRoot.path)",
            "preset_id": "coding-agent",
            "exact_speculation": "true"
          }
        }
        """
        var req1 = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(req1JSON.utf8))
        req1.httpHeaders = ["x-request-id": req1Id]
        _ = try await actor.handleChatCompletion(request: req1)

        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.packName, "meta-repo")
        let cap1 = try XCTUnwrap(backend.capturedRequests.last)
        XCTAssertTrue(cap1.usePrefixCache)
        XCTAssertEqual(cap1.cacheSalt, "ns-alpha")
        XCTAssertTrue(cap1.exactSpeculation)
        XCTAssertGreaterThan(cap1.promptTokens.count, 512)

        let rec1 = try XCTUnwrap(QWRequestRecordStore.shared.recent(100).first { $0.id == req1Id })
        XCTAssertEqual(rec1.config["cacheKey"], "ns-alpha")
        XCTAssertEqual(rec1.config["preset"], "coding-agent")
        XCTAssertEqual(rec1.config["sessionId"], "sess-meta-1")
        XCTAssertEqual(rec1.config["memPack"], "mounted")
        XCTAssertEqual(rec1.config["exactSpec"], "on")
        XCTAssertEqual(rec1.config["cachePrompt"], "on")

        // Second turn disables cache_prompt, store, and per-turn use_memory_pack.
        let req2Id = "meta-req-2-\(UUID().uuidString.prefix(8))"
        let req2JSON = """
        {
          "messages": [{"role": "user", "content": "Stateless turn"}],
          "max_tokens": 1,
          "cache_prompt": false,
          "store": false,
          "metadata": {
            "use_memory_pack": "false"
          }
        }
        """
        var req2 = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(req2JSON.utf8))
        req2.httpHeaders = ["x-request-id": req2Id]
        _ = try await actor.handleChatCompletion(request: req2)

        XCTAssertEqual(actor.getMountedMemoryPackInfo()?.packName, "meta-repo")
        let cap2 = try XCTUnwrap(backend.capturedRequests.last)
        XCTAssertFalse(cap2.usePrefixCache)
        XCTAssertNil(cap2.retainPrefixLength)
        XCTAssertLessThan(cap2.promptTokens.count, 512)
        XCTAssertFalse(QWRequestRecordStore.shared.recent(100).contains { $0.id == req2Id })
    }

    func testWirePrefixCacheSaltIsolationAndCachePromptBypass() async throws {
        let (tok, _) = try makeMinimalTokenizerAndTemplate()
        let actor = QWEngineLifecycleActor(
            config: QWEngineLifecycleActor.Config(packPath: "/dummy/model.qwp", modelName: "test", host: "127.0.0.1", port: 0),
            logger: QWServerLogger(level: .none)
        )
        let wireA = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi A<|im_end|>\n<|im_start|>assistant\n"
        let wireB = "<|im_start|>system\nYou are strict.<|im_end|>\n<|im_start|>user\nHi B<|im_end|>\n<|im_start|>assistant\n"

        let firstA = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wireA, tokenizer: tok, packPath: "/dummy/model.qwp", cachePrompt: true, cacheSalt: "tenant-a"
        )
        let secondA = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wireA + " follow-up", tokenizer: tok, packPath: "/dummy/model.qwp", cachePrompt: true, cacheSalt: "tenant-a"
        )
        let firstB = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wireB, tokenizer: tok, packPath: "/dummy/model.qwp", cachePrompt: true, cacheSalt: "tenant-b"
        )
        XCTAssertNotNil(firstA.systemPrefixTokenCount)
        XCTAssertEqual(firstA.systemPrefixTokenCount, secondA.systemPrefixTokenCount)
        XCTAssertTrue(secondA.tokens.starts(with: firstA.tokens.dropLast()))
        XCTAssertNotNil(firstB.systemPrefixTokenCount)
        XCTAssertNotEqual(firstA.tokens, firstB.tokens)

        let bypassResult = try actor.encodeWireIncrementallyWithSystemPrefix(
            wire: wireA + " stateless", tokenizer: tok, packPath: "/dummy/model.qwp", cachePrompt: false, cacheSalt: "tenant-c"
        )
        XCTAssertNil(bypassResult.systemPrefixTokenCount)
    }

    func testEnginePrefixStoreIsolatesSaltsAndPreservesSharedBase() throws {
        let baseMode = QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 4, mode: .chunked, cacheSalt: nil)
        let saltAMode = QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 4, mode: .chunked, cacheSalt: "tenant-a")
        let saltBMode = QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 4, mode: .chunked, cacheSalt: "tenant-b")
        XCTAssertEqual(QWEnginePrefixStore<Int>.baseMode(of: saltAMode), baseMode)
        XCTAssertNotEqual(saltAMode, saltBMode)

        let store = try QWEnginePrefixStore<String>(byteLimit: 65536, entryLimit: 16, vocab: 1000)
        let sharedTokens = [1, 2, 3, 4]
        XCTAssertTrue(try store.insert(tokens: sharedTokens, mode: baseMode, value: "shared-mempack", payloadBytes: 16, allowsSuffix: true, isSharedBase: true))
        XCTAssertTrue(try store.insert(tokens: [1, 2, 3, 4, 10, 11], mode: saltAMode, value: "turn-a", payloadBytes: 16, allowsSuffix: true, isSharedBase: false))

        let fullQueryA = [1, 2, 3, 4, 10, 11, 99]
        let hitA = try store.lookup(tokens: fullQueryA, mode: saltAMode)
        XCTAssertEqual(hitA?.value, "turn-a")
        XCTAssertEqual(hitA?.tokenCount, 6)

        // Different salt must not hit tenant-a's conversation prefix, only the unsalted shared base.
        let hitB = try store.lookup(tokens: fullQueryA, mode: saltBMode)
        XCTAssertEqual(hitB?.value, "shared-mempack")
        XCTAssertEqual(hitB?.tokenCount, 4)

        let sharedHit = store.sharedBasePrefix(matching: fullQueryA, mode: saltBMode)
        XCTAssertEqual(sharedHit?.value, "shared-mempack")
    }

    func testSaltedCacheHistoryIsolatesCacheSaltValues() {
        let history: [Int64] = [10, 20, 30, 40]
        let unsalted = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [], cacheSalt: nil)
        XCTAssertEqual(unsalted, history)

        let blankSalt = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [], cacheSalt: "   ")
        XCTAssertEqual(blankSalt, history)

        let saltA = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [], cacheSalt: "sess-a")
        let saltB = QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [], cacheSalt: "sess-b")
        XCTAssertNotEqual(saltA[0], history[0])
        XCTAssertNotEqual(saltA[0], saltB[0])
        XCTAssertEqual(Array(saltA.dropFirst()), Array(history.dropFirst()))
        XCTAssertEqual(saltA, QWVisualSpan.saltedCacheHistory(history: history, visualSpans: [], cacheSalt: "sess-a"))
    }

    func testWebUIChatMetadataHelpersInJSContext() throws {
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in
            jsError = exc?.toString() ?? "unknown JS error"
        }

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let presetsJS = try String(contentsOf: root.appendingPathComponent("Sources/Apps/QWServer/Resources/js/chat-presets.js"), encoding: .utf8)
        let chatJS = try String(contentsOf: root.appendingPathComponent("Sources/Apps/QWServer/Resources/js/chat.js"), encoding: .utf8)
        ctx.evaluateScript(presetsJS)
        XCTAssertNil(jsError, "JS error in chat-presets.js: \(jsError ?? "")")
        ctx.evaluateScript(chatJS)
        XCTAssertNil(jsError, "JS error in chat.js: \(jsError ?? "")")

        let res = ctx.evaluateScript("""
        (() => {
          const kv = parseCustomMetadataInput('agent=local, priority=2');
          const jsonObj = parseCustomMetadataInput('{"env":"prod","trace":true,"n":4}');
          let badThrew = false;
          try { parseCustomMetadataInput('{"arr":[1,2]}'); } catch (_) { badThrew = true; }
          const s = createBlankChatSession("Meta", "coding-agent", "Work inside /tmp/repo");
          applyPresetHarnessToSession(s, "coding-agent");
          const codingExactSpec = s.params.exactSpeculation;
          s.workspacePath = "/tmp/repo";
          s.useMemoryPackHarness = false;
          s.params.cachePrompt = false;
          s.params.promptCacheKey = "team-key";
          s.params.store = false;
          s.params.customMetadata = '{"env":"dev"}';
          s.messages.push({ id: "u1", role: "user", content: "hi" });
          const payload = buildChatCompletionPayload(s, {});
          return { kv, jsonObj, badThrew, codingExactSpec, payload };
        })()
        """)?.toDictionary()

        let kv = try XCTUnwrap(res?["kv"] as? [String: Any])
        XCTAssertEqual(kv["agent"] as? String, "local")
        XCTAssertEqual(kv["priority"] as? String, "2")

        let jsonObj = try XCTUnwrap(res?["jsonObj"] as? [String: Any])
        XCTAssertEqual(jsonObj["env"] as? String, "prod")
        XCTAssertEqual(jsonObj["trace"] as? String, "true")
        XCTAssertEqual(jsonObj["n"] as? String, "4")
        XCTAssertEqual(res?["badThrew"] as? Bool, true)
        XCTAssertEqual(res?["codingExactSpeculation"] as? String ?? res?["codingExactSpec"] as? String, "true")

        let payload = try XCTUnwrap(res?["payload"] as? [String: Any])
        XCTAssertEqual(payload["cache_prompt"] as? Bool, false)
        XCTAssertEqual(payload["prompt_cache_key"] as? String, "team-key")
        XCTAssertEqual(payload["cache_salt"] as? String, "team-key")
        XCTAssertEqual(payload["store"] as? Bool, false)
        let meta = try XCTUnwrap(payload["metadata"] as? [String: Any])
        XCTAssertEqual(meta["client"] as? String, "qwegn-webui")
        XCTAssertEqual(meta["preset_id"] as? String, "coding-agent")
        XCTAssertEqual(meta["workspace_path"] as? String, "/tmp/repo")
        XCTAssertEqual(meta["use_memory_pack"] as? String, "false")
        XCTAssertEqual(meta["exact_speculation"] as? String, "true")
        XCTAssertEqual(meta["env"] as? String, "dev")
    }
}
