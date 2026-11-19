// QWContextCeilingBoundaryTests.swift — Tests universal KV pressure, RoPE boundary step-down, and FN headroom.
import CryptoKit
import Foundation
import XCTest
@testable import QWCore
import QWFlashNext
import QWHAL
import QWHead
import QWKV
@testable import QWEngine

final class QWContextCeilingBoundaryTests: XCTestCase {
    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detVec(_ count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
        let gBytes = 20
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = Float(Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1031.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func buildSyntheticNextNPack(at rootDir: URL, maxPos: Int = 256) throws {
        let h = 256, v = 256, inter = 256
        let qHeads = 4, kvHeads = 2, headDim = 64
        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "mtp_num_hidden_layers": 1,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": \(maxPos),
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
          "layer_types": ["full_attention"],
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)
        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02)),
        ]
        func appendBlock(_ blk: String, seedBase: UInt32) {
            planes.append((blk + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase, amp: 0.05, offset: 1.0))))
            planes.append((blk + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase + 1, amp: 0.05, offset: 1.0))))
            planes.append((blk + "self_attn.q_proj.weight", .hq4(n: g.qProjOut, k: h, seed: seedBase + 2, scale: 0.008)))
            planes.append((blk + "self_attn.k_proj.weight", .hq4(n: g.kvHeads * g.headDim, k: h, seed: seedBase + 3, scale: 0.008)))
            planes.append((blk + "self_attn.v_proj.weight", .hq4(n: g.kvHeads * g.headDim, k: h, seed: seedBase + 4, scale: 0.008)))
            planes.append((blk + "self_attn.o_proj.weight", .hq4(n: h, k: g.qHeads * g.headDim, seed: seedBase + 5, scale: 0.008)))
            planes.append((blk + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 6, scale: 0.008)))
            planes.append((blk + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 7, scale: 0.008)))
            planes.append((blk + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: seedBase + 8, scale: 0.008)))
        }
        appendBlock("model.layers.0.", seedBase: 100)
        planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 200, scale: 0.006)))
        planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        appendBlock("mtp.layers.0.", seedBase: 300)

        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 8)
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        padTo(256)
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "qwen3_5",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-ctx-ceiling",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (buf: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: buf) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }

    func testFlashNextEstimateContextMemoryAddsHeadroomBelow32KAndFallsBackWhenTight() throws {
        let desc = QWModelDescriptor.qwen38_27b
        let roomy = 128 * (1 << 30)
        let est4k = FlashNextModule.estimateContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 4096, kvMode: .f16,
            mtpDrafts: 0, mtpBackend: "standard", prefillChunkSize: 1024,
            tuning: .default, physicalMemoryBytes: roomy
        )
        XCTAssertEqual(est4k.kvBytes, (4096 + 1024) * FNKVStorageMode.f16.perTokenBytes)
        XCTAssertTrue(est4k.fits)

        let est16k = FlashNextModule.estimateContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 16_384, kvMode: .f16,
            mtpDrafts: 0, mtpBackend: "standard", prefillChunkSize: 1024,
            tuning: .default, physicalMemoryBytes: roomy
        )
        XCTAssertEqual(est16k.kvBytes, (16_384 + 1024) * FNKVStorageMode.f16.perTokenBytes)
        XCTAssertTrue(est16k.fits)

        // Tight physical memory: fits 4096 but not 4096 + 1024 -> falls back cleanly to 4096.
        let reserve = Int((Double(QWCapacityBudget.osReserveBytes >> 30) * Double(1 << 30)).rounded())
        let fixedBytes = est4k.weightsBytes + est4k.scratchAndStateBytes
        let tightPhys = reserve + fixedBytes + 4096 * FNKVStorageMode.f16.perTokenBytes
        let estTight = FlashNextModule.estimateContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 4096, kvMode: .f16,
            mtpDrafts: 0, mtpBackend: "standard", prefillChunkSize: 1024,
            tuning: .default, physicalMemoryBytes: tightPhys
        )
        XCTAssertEqual(estTight.kvBytes, 4096 * FNKVStorageMode.f16.perTokenBytes)
        XCTAssertTrue(estTight.fits)
    }

    func testSpeculativeStepDownNearRopeRowsAndPromptClamping() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver boundary test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-ctx-bounds-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 256)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 256, ropeRows: 256,
            kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "mtp", prefillChunkSize: 32)
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)

        // Prompt at or beyond ropeRows must fail up front across all entry points.
        let oversized = Array(repeating: 7, count: 256)
        XCTAssertThrowsError(try driver.generateCachedGreedy(prompt: oversized, maxNew: 4, speculative: true)) { err in
            XCTAssertEqual(err as? QWEngineDriverError, .posBeyondTable(256))
        }
        let sOver = try driver.makeSession(promptTokens: 256)
        XCTAssertThrowsError(try driver.generateExperimentalGreedy(session: sOver, prompt: oversized, maxNew: 4)) { err in
            driver.finishSession(sOver)
            XCTAssertEqual(err as? QWEngineDriverError, .posBeyondTable(256))
        }

        // Prompt 6 tokens below ropeRows: speculative rounds step down cleanly to serial decode.
        let nearPrompt = (0..<250).map { 10 + ($0 % 200) }
        let baseSession = try driver.makeSession(promptTokens: nearPrompt.count)
        let baseline = try driver.generate(session: baseSession, prompt: nearPrompt, maxNew: 20)
        driver.finishSession(baseSession)
        XCTAssertEqual(baseline.count, 6)

        let expSession = try driver.makeSession(promptTokens: nearPrompt.count)
        let expOut = try driver.generateExperimentalGreedy(session: expSession, prompt: nearPrompt, maxNew: 20)
        XCTAssertFalse(driver.canRunSpeculativeRound(expSession))
        driver.finishSession(expSession)
        XCTAssertEqual(expOut, baseline)

        let cachedOut = try driver.generateCachedGreedy(prompt: nearPrompt, maxNew: 20, speculative: true)
        XCTAssertEqual(cachedOut, baseline)

        let sampler = QWSamplerConfig(temperature: 0.8, topK: 16, topP: 0.9, seed: 42, uniformVersion: .upper24)
        let stochSession = try driver.makeStochasticSession(promptTokens: nearPrompt.count, config: sampler)
        let stochOut = try driver.generateExperimentalStochastic(session: stochSession, prompt: nearPrompt, maxNew: 20, speculative: true)
        driver.finishSession(stochSession)
        XCTAssertEqual(stochOut.count, 6)
    }

    func testUncachedRequestEvictsWarmPrefixPagesViaOnPagePressure() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver page-pressure test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-kv-pressure-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 256)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 256, ropeRows: 256,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)
        XCTAssertNotNil(driver.kvCache.onPagePressure)

        // Two <60-token prompts avoid autoRetain splitting; maxNew: 1 avoids decode COW so both remain cached (2/2 pages/pool).
        let p1 = (0..<48).map { 10 + ($0 % 50) }
        let p2 = (0..<48).map { 70 + ($0 % 50) }
        _ = try driver.generateCachedGreedy(prompt: p1, maxNew: 1, speculative: false)
        _ = try driver.generateCachedGreedy(prompt: p2, maxNew: 1, speculative: false)
        XCTAssertEqual(driver.prefixCacheEntries, 2)
        XCTAssertEqual(driver.freeKVPages, 0)

        // Uncached request (`usePrefixCache: false`, simulating `cache_prompt: false`) must evict without deadlock.
        let uncachedReq = QWGenerationRequest(
            prompt: (0..<200).map { 130 + ($0 % 50) },
            maxNew: 4,
            sampling: .greedy,
            usePrefixCache: false
        )
        let res = try driver.generate(request: uncachedReq, onToken: nil)
        XCTAssertEqual(res.tokens.count, 4)
        XCTAssertLessThan(driver.prefixCacheEntries, 2)
        XCTAssertGreaterThan(driver.freeKVPages, 0)

        // Clearing prefix cache clears onPagePressure cleanly.
        try driver.configurePrefixCache(byteLimit: 0, entryLimit: 8)
        XCTAssertNil(driver.kvCache.onPagePressure)
    }

    func testGenerateCachedStochasticWritesPrefillLogitsAcrossMultiTurnAndStopTokens() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver stochastic prefill test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-stoch-prefill-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 256)

        // Fresh driver starts with zeroed sc.logits; profile-style repetitionPenalty=1.05 routes temperature=0 through generateCachedStochastic.
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 256, ropeRows: 256,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)

        let sampler = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.0, topP: 1.0, topK: 1, repetitionPenalty: 1.05, seed: 7),
            vocabulary: 256
        )
        let p1 = (0..<40).map { 10 + $0 }
        let p2 = (0..<44).map { 100 + $0 }

        // Turn 1 with maxNew=4 so decodeStochastic runs and leaves a real token in sc.logits; stop at its 2nd generated token on Turn 1 replay below.
        let turn1Warm = try driver.generate(
            request: QWGenerationRequest(prompt: p1, maxNew: 3, sampling: sampler, usePrefixCache: true),
            onToken: nil
        )
        XCTAssertEqual(turn1Warm.tokens.count, 3)

        let stopTok = turn1Warm.tokens[1]
        // Ensure Turn 1 stopping on `stopTok` leaves `stopTok` logits in `sc.logits`, then Turn 2 still prefill-samples its own first token.
        let freshDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 256, ropeRows: 256,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        try freshDriver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)

        let cachedTurn1 = try freshDriver.generate(
            request: QWGenerationRequest(prompt: p1, maxNew: 4, stopIDs: [stopTok], sampling: sampler, usePrefixCache: true),
            onToken: nil
        )
        XCTAssertEqual(cachedTurn1.tokens, [turn1Warm.tokens[0]])

        let cachedTurn2 = try freshDriver.generate(
            request: QWGenerationRequest(prompt: p2, maxNew: 4, stopIDs: [stopTok], sampling: sampler, usePrefixCache: true),
            onToken: nil
        )
        XCTAssertGreaterThan(cachedTurn2.tokens.count, 0)

        let refS1 = try freshDriver.makeStochasticSession(promptTokens: p1.count, config: sampler)
        let refTurn1 = try freshDriver.generateExperimentalStochastic(session: refS1, prompt: p1, maxNew: 1, speculative: false)
        freshDriver.finishSession(refS1)
        XCTAssertEqual(turn1Warm.tokens.first, refTurn1.first)

        let refS2 = try freshDriver.makeStochasticSession(promptTokens: p2.count, config: sampler)
        let refTurn2 = try freshDriver.generateExperimentalStochastic(session: refS2, prompt: p2, maxNew: 4, speculative: false)
        freshDriver.finishSession(refS2)
        let refTurn2Clipped = Array(refTurn2.prefix { $0 != stopTok })
        XCTAssertFalse(refTurn2Clipped.isEmpty)
        XCTAssertEqual(cachedTurn2.tokens, refTurn2Clipped)
    }

    func testGenerateCachedMultiTurnWithRetainPrefixLengthCachesSystemAndTurnPrefixes() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver multi-turn retain test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-multiturn-retain-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 1024)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 1024, ropeRows: 1024,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 16)

        let sampler = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.0, topP: 1.0, topK: 1, repetitionPenalty: 1.05, seed: 11),
            vocabulary: 256
        )
        let p1 = (0..<96).map { 10 + ($0 % 200) }
        let p2 = Array(p1.prefix(92)) + (0..<36).map { 50 + ($0 % 180) }
        let p3 = Array(p2.prefix(124)) + (0..<36).map { 90 + ($0 % 150) }

        let r1 = try driver.generate(
            request: QWGenerationRequest(prompt: p1, maxNew: 2, sampling: sampler, usePrefixCache: true, retainPrefixLength: 32),
            onToken: nil
        )
        XCTAssertEqual(r1.tokens.count, 2)
        XCTAssertEqual(driver.lastPrefixCacheStats?.hit, false)

        let r2 = try driver.generate(
            request: QWGenerationRequest(prompt: p2, maxNew: 2, sampling: sampler, usePrefixCache: true, retainPrefixLength: 32),
            onToken: nil
        )
        XCTAssertEqual(r2.tokens.count, 2)
        XCTAssertEqual(driver.lastPrefixCacheStats?.hit, true)
        XCTAssertEqual(driver.lastPrefixCacheStats?.matchedTokens, 92)

        let r3 = try driver.generate(
            request: QWGenerationRequest(prompt: p3, maxNew: 2, sampling: sampler, usePrefixCache: true, retainPrefixLength: 32),
            onToken: nil
        )
        XCTAssertEqual(r3.tokens.count, 2)
        XCTAssertEqual(driver.lastPrefixCacheStats?.hit, true)
        XCTAssertEqual(driver.lastPrefixCacheStats?.matchedTokens, 124)
    }

    func testActiveRepetitionHistoryExcludesPromptAndCapsGeneratedWindow() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver session window test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-rep-window-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 512)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let stopTok = 77
        let prompt = [stopTok, stopTok, stopTok, stopTok, 11, 12]
        let session = try driver.makeSession(promptTokens: prompt.count)
        defer { driver.finishSession(session) }
        _ = try driver.prefill(session, tokens: prompt, mode: .chunked)
        XCTAssertEqual(session.promptTokenCount, prompt.count)

        session.tokenHistory = prompt + (100..<180).map { $0 }
        XCTAssertEqual(session.activeRepetitionHistory(extraTokens: [180, 181]), [])
        session.tokenHistory = prompt + [55, 55, 55]
        XCTAssertEqual(session.activeRepetitionHistory(), [])
        session.tokenHistory = prompt + [1, 0, 2, 99, 1, 0]
        XCTAssertEqual(session.activeRepetitionHistory(), [])
        session.tokenHistory = prompt + [55, 55, 55, 55]
        XCTAssertEqual(session.activeRepetitionHistory(), [55])
        session.tokenHistory = prompt + [20, 21, 20, 21, 20, 21]
        XCTAssertEqual(session.activeRepetitionHistory(), [20])
        session.tokenHistory = prompt + [20, 21, 99, 20, 21, 99]
        XCTAssertEqual(session.activeRepetitionHistory(), [20])
        session.tokenHistory = prompt + [42, 1, 42, 2, 42, 3, 42, 4, 42, 5, 42]
        XCTAssertEqual(session.activeRepetitionHistory(), [42])
        // Loop outside the 64-token generated window drops cleanly without false positives.
        session.tokenHistory = prompt + [55, 55, 55, 55] + (100..<164).map { $0 }
        XCTAssertEqual(session.activeRepetitionHistory(), [])

        var fnSession = FNGPUSessionData(history: [Int64(stopTok), Int64(stopTok), Int64(stopTok), Int64(stopTok)], sampler: QWSampler(config: .greedy))
        for tok in 100..<180 {
            fnSession.recordSeenToken(tok, vocab: 256)
        }
        XCTAssertEqual(fnSession.recentGeneratedTokens.count, FNGPUSessionData.repetitionPenaltyWindow)
        XCTAssertEqual(fnSession.seenTokenList, [])
        for _ in 0..<3 {
            fnSession.recordSeenToken(210, vocab: 256)
        }
        XCTAssertEqual(fnSession.seenTokenList, [])
        fnSession.recordSeenToken(210, vocab: 256)
        XCTAssertEqual(fnSession.seenTokenList, [210])
        XCTAssertFalse(fnSession.seenTokens.contains(stopTok))
    }

    func testSpeculatableConstraintWithStochasticSamplingAndRepetitionPenalty() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver constrained stochastic test requires bare-metal Apple Silicon GPU"
        )
        final class BoundaryBudgetConstraint: QWTokenConstraint, @unchecked Sendable {
            private let maxBudget: Int
            private let endThinkId: Int
            private(set) var acceptedCount = 0
            private(set) var closed = false
            init(maxBudget: Int, endThinkId: Int) {
                self.maxBudget = maxBudget
                self.endThinkId = endThinkId
            }
            var allowsSpeculation: Bool { true }
            var isUnconstrainedNow: Bool { !closed && (maxBudget - acceptedCount) > 16 }
            var speculationBoundaryToken: Int? { closed ? nil : endThinkId }
            func allowedTokens() throws -> [Int]? {
                if !closed && acceptedCount >= maxBudget { return [endThinkId] }
                return nil
            }
            func accept(_ token: Int) throws {
                if !closed {
                    if token == endThinkId { closed = true }
                    else { acceptedCount += 1 }
                }
            }
        }

        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-stoch-constraint-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 512)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 2, mtpAdaptiveDepth: true, prefillChunkSize: 32)
        try driver.configurePrefixCache(byteLimit: 64 << 20, entryLimit: 8)

        let endThinkId = 201
        let constraint = BoundaryBudgetConstraint(maxBudget: 20, endThinkId: endThinkId)
        let sampling = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.4, topP: 0.95, topK: 40, repetitionPenalty: 1.5, seed: -1),
            vocabulary: 256
        )
        let prompt = (0..<24).map { 10 + $0 }
        let res = try driver.generate(
            request: QWGenerationRequest(
                prompt: prompt,
                maxNew: 26,
                sampling: sampling,
                usePrefixCache: true,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2, mtpBackend: "standard"),
                constraint: constraint
            ),
            onToken: nil
        )
        XCTAssertEqual(res.tokens.count, 26)
        XCTAssertTrue(constraint.closed)
        XCTAssertTrue(res.tokens.contains(endThinkId))
        XCTAssertLessThanOrEqual(constraint.acceptedCount, 20)
    }

    func testNaivePrefillAndUncachedStreamingStochasticSpeculation() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Synthetic driver prefill/speculative test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-naive-stoch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir, maxPos: 512)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 2, prefillChunkSize: 32)
        let prompt = [10, 11, 12, 13, 14]
        let naiveSession = try driver.makeSession(promptTokens: prompt.count)
        defer { driver.finishSession(naiveSession) }
        _ = try driver.prefill(naiveSession, tokens: prompt, mode: .naive)
        XCTAssertEqual(naiveSession.generated, 1)
        XCTAssertEqual(naiveSession.promptTokenCount, prompt.count)
        XCTAssertEqual(naiveSession.tokenHistory.count, prompt.count + 1)

        let sampling = QWSamplerConfig(
            QWSamplingParameters(temperature: 0.4, topP: 0.95, topK: 40, repetitionPenalty: 1.5, seed: -1),
            vocabulary: 256
        )
        var streamed: [Int] = []
        let res = try driver.generate(
            request: QWGenerationRequest(
                prompt: prompt,
                maxNew: 12,
                sampling: sampling,
                usePrefixCache: false,
                speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2, mtpBackend: "standard")
            ),
            onToken: { tok in streamed.append(tok); return true }
        )
        XCTAssertEqual(res.tokens.count, 12)
        XCTAssertEqual(streamed, res.tokens)
        XCTAssertNotNil(res.speculativeSummary)
        XCTAssertFalse(res.stepTokenCounts.isEmpty)
    }
}
