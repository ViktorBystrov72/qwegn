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
}
