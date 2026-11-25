// QWInterleavedBatchSchedulerTests.swift — GPU parity tests for H1.3 B=8 batching, Radix KV, and interleaved prefill.

import CryptoKit
import Darwin
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL
import QWHead
@testable import QWEngine

final class QWInterleavedBatchSchedulerTests: XCTestCase {
    private enum PlaneSpec {
        case f32(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    func testEightWayBatchedDecodeRadixSiblingDeduplicationAndInterleavedPrefillParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "GPU batch scheduler parity tests require bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-h13-hybrid-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticPack(at: rootDir, pureAttention: false)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 4096, ropeRows: 4096,
            kvMode: .q8, mtpDrafts: 0, prefillChunkSize: 256, maxConcurrentSessions: 8
        )
        try driver.configurePrefixCache(
            config: QWCacheConfig(warmByteLimit: 64 << 20, ssdByteLimit: 0, entryLimit: 32, maxConcurrentSessions: 8)
        )
        XCTAssertEqual(driver.maxBatchSlots, 8)
        XCTAssertTrue(driver.canRunBatchedDecode)

        // 1. Direct B=8 forwardBatchStep 0-ULP parity vs serial decode.
        let eightPrompts: [[Int]] = (0..<8).map { idx in
            (0..<12).map { pos in (1 + idx * 7 + pos * 13) % 250 }
        }
        var serialEight: [[Int]] = []
        for p in eightPrompts {
            let res = try driver.generate(
                request: QWGenerationRequest(prompt: p, maxNew: 5, sampling: .greedy, usePrefixCache: false),
                onToken: nil
            )
            serialEight.append(res.tokens)
        }

        do {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer { for s in sessions where !s.sessionFinished { driver.finishSession(s) } }
            for b in 0..<8 {
                let sess = try driver.makeSession(promptTokens: eightPrompts[b].count)
                sessions.append(sess)
                pending.append(try driver.prefill(sess, tokens: eightPrompts[b]))
            }
            var batched = pending.map { [$0] }
            for _ in 1..<5 {
                let (nextIds, _, _) = try driver.forwardBatchStep(sessions, tokens: pending, greedyOnly: true)
                pending = nextIds
                for b in 0..<8 { batched[b].append(nextIds[b]) }
            }
            for b in 0..<8 {
                XCTAssertEqual(batched[b], serialEight[b], "B=8 stream \(b) diverged from serial")
            }
        }

        let module = DenseModule(engine: driver)
        let unified = try QWEngine(engine: module, descriptor: driver.geometry.descriptor, plan: driver.plan)
        let scheduler = unified.batchScheduler

        // 2. 8-sibling subagent shared-prefix wave (512 shared + 32 unique): Radix hit + deduplicated page bytes + 0-ULP parity.
        let shared512 = (0..<512).map { ($0 * 5 + 11) % 240 + 1 }
        let siblingPrompts: [[Int]] = (0..<8).map { sib in
            shared512 + (0..<32).map { t in ((sib * 19 + t * 3) % 240) + 1 }
        }
        var expectedSiblings: [[Int]] = []
        for p in siblingPrompts {
            let uncached = try driver.generate(
                request: QWGenerationRequest(prompt: p, maxNew: 6, sampling: .greedy, usePrefixCache: false),
                onToken: nil
            )
            expectedSiblings.append(uncached.tokens)
        }

        let waveResults = LockedBox<[Int: QWGenerationResult]>([:])
        let waveGroup = DispatchGroup()
        QWBatchAdmissionTracker.beginPreparing()
        for sib in 0..<8 {
            waveGroup.enter()
            DispatchQueue.global().async {
                defer { waveGroup.leave() }
                let req = QWGenerationRequest(
                    prompt: siblingPrompts[sib], maxNew: 6, sampling: .greedy, usePrefixCache: true
                )
                if let res = try? scheduler.generate(request: req) {
                    waveResults.mutate { $0[sib] = res }
                }
            }
        }
        usleep(3000)
        QWBatchAdmissionTracker.endPreparing()
        waveGroup.wait()

        var hitsAt512 = 0
        for sib in 0..<8 {
            let res = try XCTUnwrap(waveResults.value[sib])
            XCTAssertEqual(res.tokens, expectedSiblings[sib], "Sibling \(sib) Radix cached output diverged")
            if res.cache?.matchedTokens == 512 { hitsAt512 += 1 }
        }
        XCTAssertGreaterThanOrEqual(hitsAt512, 7, "Expected >=7 siblings to hit 512-tok Radix wave node, got \(hitsAt512)")

        // Verify deduplicated physical page accounting: 8 siblings + 1 shared checkpoint must NOT double-charge
        // the 2 shared 256-token KV pages (saving >= 7 * shared512KVBytes compared to per-entry charging).
        let shared512KVBytes = driver.kvCache.sharedPrefixBytes(tokenCount: 512)
        let singleFullKVBytes = driver.kvCache.sharedPrefixBytes(tokenCount: 544)
        let perEntryStateAndLogits = driver.gdnLayers.count * (
            QWSlotGeometry.recElems(geometry: driver.geometry) * 4
                + QWSlotGeometry.ringElems(geometry: driver.geometry) * 2
        ) + driver.sc.logits.bytes
        let undeduplicatedEstimate = (shared512KVBytes + perEntryStateAndLogits)
            + 8 * (singleFullKVBytes + perEntryStateAndLogits)
        XCTAssertLessThanOrEqual(
            driver.prefixCacheChargedBytes,
            undeduplicatedEstimate - 7 * shared512KVBytes + 65_536,
            "Expected >= 7x shared-page deduplication savings, got charged=\(driver.prefixCacheChargedBytes) vs undedup=\(undeduplicatedEstimate)"
        )

        // 3. Concurrent foreground streaming decode + background 768-tok chunked prefill (slice = 64):
        // verify foreground emits tokens DURING background prefill and both streams are 0-ULP bit-exact.
        setenv("QW_INTERLEAVED_PREFILL_SLICE", "64", 1)
        defer { unsetenv("QW_INTERLEAVED_PREFILL_SLICE") }

        let fgPrompt = (0..<40).map { ($0 * 7 + 9) % 240 + 1 }
        let bgPrompt = (0..<768).map { ($0 * 17 + 23) % 240 + 1 }
        let expectedFG = try driver.generate(
            request: QWGenerationRequest(prompt: fgPrompt, maxNew: 16, sampling: .greedy, usePrefixCache: false),
            onToken: nil
        ).tokens
        let expectedBG = try driver.generate(
            request: QWGenerationRequest(prompt: bgPrompt, maxNew: 6, sampling: .greedy, usePrefixCache: false),
            onToken: nil
        ).tokens

        let fgTokenTimes = LockedBox<[Double]>([])
        let bgPrefillSliceTimes = LockedBox<(first: Double, last: Double)>((0, 0))
        let fgOut = LockedBox<[Int]>([])
        let bgOut = LockedBox<[Int]>([])
        let tBase = Date()
        let interleaveGroup = DispatchGroup()

        interleaveGroup.enter()
        DispatchQueue.global().async {
            defer { interleaveGroup.leave() }
            let req = QWGenerationRequest(prompt: fgPrompt, maxNew: 16, sampling: .greedy, usePrefixCache: false)
            if let res = try? scheduler.generate(request: req, onToken: { _ in
                let now = Date().timeIntervalSince(tBase) * 1000.0
                fgTokenTimes.mutate { $0.append(now) }
                usleep(1500)
                return true
            }) {
                fgOut.mutate { $0 = res.tokens }
            }
        }
        usleep(2000)
        interleaveGroup.enter()
        DispatchQueue.global().async {
            defer { interleaveGroup.leave() }
            let req = QWGenerationRequest(
                prompt: bgPrompt, maxNew: 6, sampling: .greedy, usePrefixCache: false,
                onPrefillProgress: { done, total in
                    guard total > 0, done > 0 else { return }
                    let now = Date().timeIntervalSince(tBase) * 1000.0
                    bgPrefillSliceTimes.mutate { cur in
                        let first = cur.first == 0 ? now : cur.first
                        cur = (first, now)
                    }
                }
            )
            if let res = try? scheduler.generate(request: req) {
                bgOut.mutate { $0 = res.tokens }
            }
        }
        interleaveGroup.wait()

        XCTAssertEqual(fgOut.value, expectedFG, "Foreground interleaved decode diverged from serial")
        XCTAssertEqual(bgOut.value, expectedBG, "Background interleaved prefill+decode diverged from serial")
        let bgWindow = bgPrefillSliceTimes.value
        let fgDuringBG = fgTokenTimes.value.filter { $0 >= bgWindow.first && $0 <= bgWindow.last }
        XCTAssertGreaterThan(bgWindow.last, bgWindow.first, "Background prefill should span multiple 64-tok slices")
        XCTAssertFalse(fgDuringBG.isEmpty, "Foreground decode should emit tokens between background prefill slices")
    }

    func testPureAttentionZeroCopyPageSlicingInRadixPrefixCache() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-h13-pure-attn-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticPack(at: rootDir, pureAttention: true)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q8, mtpDrafts: 0, prefillChunkSize: 256
        )
        try driver.configurePrefixCache(
            config: QWCacheConfig(warmByteLimit: 32 << 20, ssdByteLimit: 0, entryLimit: 16, maxConcurrentSessions: 8)
        )
        XCTAssertTrue(driver.canSlicePureAttentionPages)

        let shared512 = (0..<512).map { ($0 * 11 + 3) % 240 + 1 }
        let agentA = shared512 + (0..<45).map { ($0 * 3 + 7) % 240 + 1 }
        let agentB = shared512 + (0..<45).map { ($0 * 5 + 19) % 240 + 1 }

        let expectedB = try driver.generate(
            request: QWGenerationRequest(prompt: agentB, maxNew: 6, sampling: .greedy, usePrefixCache: false),
            onToken: nil
        ).tokens

        // Agent A populates a single 557-token entry (no explicit checkpoint at 512).
        _ = try driver.generate(
            request: QWGenerationRequest(prompt: agentA, maxNew: 4, sampling: .greedy, usePrefixCache: true),
            onToken: nil
        )
        // Agent B diverges at token 512, but pure-attention Radix page slicing adopts pages 0..1 (512 tokens)!
        let resB = try driver.generate(
            request: QWGenerationRequest(prompt: agentB, maxNew: 6, sampling: .greedy, usePrefixCache: true),
            onToken: nil
        )
        XCTAssertEqual(resB.cache?.hit, true)
        XCTAssertEqual(resB.cache?.matchedTokens, 512)
        XCTAssertEqual(resB.tokens, expectedB)
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

    private static func buildSyntheticPack(at rootDir: URL, pureAttention: Bool) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let gdnKHeads = 2, gdnVHeads = 4, gdnKDim = 128, gdnVDim = 128, convK = 4
        let layerTypes = pureAttention
            ? "[\"full_attention\", \"full_attention\"]"
            : "[\"linear_attention\", \"full_attention\"]"
        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 2,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "linear_num_key_heads": \(gdnKHeads),
          "linear_num_value_heads": \(gdnVHeads),
          "linear_key_head_dim": \(gdnKDim),
          "linear_value_head_dim": \(gdnVDim),
          "linear_conv_kernel_dim": \(convK),
          "layer_types": \(layerTypes),
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
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

        let qRows = g.qProjOut, kvRows = g.kvHeads * g.headDim, oCols = g.qHeads * g.headDim
        func appendFullAttnLayer(_ idx: Int, seedBase: UInt32) {
            let pfx = "model.layers.\(idx)."
            planes.append((pfx + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase, amp: 0.05, offset: 1.0))))
            planes.append((pfx + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase + 1, amp: 0.05, offset: 1.0))))
            planes.append((pfx + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: seedBase + 2, scale: 0.008)))
            planes.append((pfx + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 3, scale: 0.008)))
            planes.append((pfx + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 4, scale: 0.008)))
            planes.append((pfx + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: seedBase + 5, scale: 0.008)))
            planes.append((pfx + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((pfx + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((pfx + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 6, scale: 0.008)))
            planes.append((pfx + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 7, scale: 0.008)))
            planes.append((pfx + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: seedBase + 8, scale: 0.008)))
        }

        if pureAttention {
            appendFullAttnLayer(0, seedBase: 100)
        } else {
            let b0 = "model.layers.0."
            planes.append((b0 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 100, amp: 0.05, offset: 1.0))))
            planes.append((b0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 101, amp: 0.05, offset: 1.0))))
            planes.append((b0 + "linear_attn.in_proj_qkv.weight", .hq4(n: g.gdnQKVChannels, k: h, seed: 102, scale: 0.008)))
            planes.append((b0 + "linear_attn.in_proj_z.weight", .hq4(n: g.gdnZChannels, k: h, seed: 103, scale: 0.008)))
            planes.append((b0 + "linear_attn.in_proj_a.weight", .hq4(n: g.gdnVHeads, k: h, seed: 104, scale: 0.008)))
            planes.append((b0 + "linear_attn.in_proj_b.weight", .hq4(n: g.gdnVHeads, k: h, seed: 105, scale: 0.008)))
            planes.append((b0 + "linear_attn.conv1d.weight", .bf16(shape: [g.gdnQKVChannels, 1, convK],
                                                                   values: detVec(g.gdnQKVChannels * convK, seedInit: 106, amp: 0.05))))
            planes.append((b0 + "linear_attn.norm.weight", .bf16(shape: [g.gdnHeadDim], values: [Float](repeating: 1.0, count: g.gdnHeadDim))))
            planes.append((b0 + "linear_attn.A_log", .f32(shape: [g.gdnVHeads], values: [Float](repeating: -1.0, count: g.gdnVHeads))))
            planes.append((b0 + "linear_attn.dt_bias", .f32(shape: [g.gdnVHeads], values: [Float](repeating: 0.0, count: g.gdnVHeads))))
            planes.append((b0 + "linear_attn.out_proj.weight", .hq4(n: h, k: g.gdnZChannels, seed: 107, scale: 0.008)))
            planes.append((b0 + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 108, scale: 0.008)))
            planes.append((b0 + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 109, scale: 0.008)))
            planes.append((b0 + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 110, scale: 0.008)))
        }
        appendFullAttnLayer(1, seedBase: 200)

        var file = Data()
        file.append(Data("QWP1".utf8))
        file.append(Data(repeating: 0, count: 8))
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        padTo(256)

        for (name, spec) in planes {
            switch spec {
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 4)
                for val in vals {
                    var le = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &le) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
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
                "source": "synthetic-h13-parity",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (raw: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: raw) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }
}

private final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: T
    init(_ value: T) { self.storage = value }
    var value: T { lock.withLock { storage } }
    func mutate(_ body: (inout T) -> Void) { lock.withLock { body(&storage) } }
}
