// QWDriverBatchDecodeMHCTests.swift — Batched decode parity tests for looped mHC + short-context LSA.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL
import QWQuant
@testable import QWEngine

final class QWDriverBatchDecodeMHCTests: XCTestCase {
    private enum PlaneSpec {
        case f32(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "mHC + LSA batched decode parity tests require bare-metal Apple Silicon GPU"
        )
    }

    func testLoopedMHCBatchedDecodeAtB2B3B4AndTailParity() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-batch-mhc-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticSenseNovaLoopPack(at: rootDir)

        let lsaCfg = QWLSAConfig(mode: .svd, budget: 512)
        XCTAssertEqual(lsaCfg.denseBypassThreshold, 1028)
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 64, lsaConfig: lsaCfg)
        XCTAssertTrue(driver.canRunBatchedDecode)
        XCTAssertNotNil(driver.lsaOperator)
        let initialFreeSlots = driver.freeSlots.count

        let prompts: [[Int]] = [
            [1, 10, 20, 30, 40, 50, 60, 70],
            [1, 15, 25, 35, 45, 55, 65, 75, 85],
            [1, 22, 32, 42, 52, 62, 72],
            [1, 33, 44, 55, 66, 77, 88, 99, 111],
        ]
        let maxDecode = 10
        let serialOutputs = try prompts.map { prompt in
            try driver.generate(
                request: QWGenerationRequest(prompt: prompt, maxNew: maxDecode, sampling: .greedy),
                onToken: nil).tokens
        }

        // Direct forwardBatchStep verification at B = 2, 3, and 4 against serial decode.
        for bCount in 2...min(4, initialFreeSlots) {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer {
                for s in sessions where !s.sessionFinished {
                    driver.finishSession(s)
                }
            }
            for idx in 0..<bCount {
                let s = try driver.makeSession(promptTokens: prompts[idx].count + 4)
                let tok0 = try driver.prefill(s, tokens: prompts[idx])
                XCTAssertEqual(tok0, serialOutputs[idx][0], "B=\(bCount) prefill parity at slot \(idx)")
                sessions.append(s)
                pending.append(tok0)
            }
            for step in 1..<4 {
                let (nextIds, stepMs, _) = try driver.forwardBatchStep(sessions, tokens: pending)
                XCTAssertGreaterThan(stepMs, 0)
                for idx in 0..<bCount {
                    XCTAssertEqual(nextIds[idx], serialOutputs[idx][step],
                                   "B=\(bCount) step \(step) slot \(idx) parity mismatch")
                }
                pending = nextIds
            }
        }
        XCTAssertEqual(driver.freeSlots.count, initialFreeSlots)

        // Concurrent burst forcing B=4 -> B=2 -> B=1 stepSingleSlot tail.
        let maxNewPerReq = [3, 3, 7, 10]
        let jobs = (0..<4).map { idx in
            QWBatchScheduledJob(
                request: QWGenerationRequest(prompt: prompts[idx], maxNew: maxNewPerReq[idx], sampling: .greedy),
                onToken: nil)
        }
        driver.runScheduledBatch(initialJobs: jobs, headScratch: nil,
                                 pollPending: { _ in [] }, requeueDelayed: { _ in })
        for idx in 0..<4 {
            let res = try jobs[idx].wait()
            XCTAssertEqual(res.tokens, Array(serialOutputs[idx].prefix(maxNewPerReq[idx])),
                           "Scheduled mHC burst mismatch at job \(idx)")
        }
        XCTAssertEqual(driver.freeSlots.count, initialFreeSlots)
    }

    func testMixedShortAndLongLSAScheduledBatchAndNgramHandoff() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-batch-lsa-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticSenseNovaLoopPack(at: rootDir)

        let lsaCfg = QWLSAConfig(mode: .svd, budget: 512)
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false, mtpRelaxedMargin: 0,
            prefillChunkSize: 64, lsaConfig: lsaCfg)
        XCTAssertTrue(driver.canRunBatchedDecode)
        XCTAssertEqual(driver.lsaConfig.denseBypassThreshold, 1028)
        let initialFreeSlots = driver.freeSlots.count

        let longPrompt = [1] + (0..<1039).map { 10 + ($0 % 200) }
        let shortPromptA = [1, 12, 24, 36, 48, 60, 72, 84, 12, 24, 36, 48]
        let shortPromptB = [1, 17, 29, 41, 53, 65, 77, 89, 17, 29, 41, 53]
        XCTAssertGreaterThan(longPrompt.count, driver.lsaConfig.denseBypassThreshold)

        let serialLong = try driver.generate(
            request: QWGenerationRequest(prompt: longPrompt, maxNew: 5, sampling: .greedy),
            onToken: nil).tokens
        let serialShortA = try driver.generate(
            request: QWGenerationRequest(prompt: shortPromptA, maxNew: 6, sampling: .greedy),
            onToken: nil).tokens
        let serialShortB = try driver.generate(
            request: QWGenerationRequest(prompt: shortPromptB, maxNew: 8, sampling: .greedy),
            onToken: nil).tokens

        let jobs = [
            QWBatchScheduledJob(
                request: QWGenerationRequest(prompt: longPrompt, maxNew: 5, sampling: .greedy),
                onToken: nil),
            QWBatchScheduledJob(
                request: QWGenerationRequest(prompt: shortPromptA, maxNew: 6, sampling: .greedy),
                onToken: nil),
            QWBatchScheduledJob(
                request: QWGenerationRequest(prompt: shortPromptB, maxNew: 8, sampling: .greedy),
                onToken: nil),
        ]
        driver.runScheduledBatch(initialJobs: jobs, headScratch: nil,
                                 pollPending: { _ in [] }, requeueDelayed: { _ in })

        let longRes = try jobs[0].wait()
        let shortARes = try jobs[1].wait()
        let shortBRes = try jobs[2].wait()
        XCTAssertEqual(longRes.tokens, serialLong)
        XCTAssertEqual(shortARes.tokens, serialShortA)
        XCTAssertEqual(shortBRes.tokens, serialShortB)
        XCTAssertEqual(driver.freeSlots.count, initialFreeSlots)
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

    private static func buildSyntheticSenseNovaLoopPack(at rootDir: URL) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let gdnKHeads = 2, gdnVHeads = 4, gdnKDim = 128, gdnVDim = 128, convK = 4
        let streams = 2, mhcRows = 2 * streams + streams * streams

        let cfgJSON = """
        {
          "architectures": ["SenseNovaLoopForConditionalGeneration"],
          "model_type": "sensenova_loop",
          "model_name": "SenseNova-Loop-Synthetic",
          "loop_settings": { "start": 1, "end": 3, "eval_count": 2, "counts": [2] },
          "mhc": { "start": 1, "end": 3, "streams": \(streams), "iterations": 20, "fixed_identity": false, "pre_scale": 1.0 },
          "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": \(v),
            "hidden_size": \(h),
            "intermediate_size": \(inter),
            "num_hidden_layers": 4,
            "mtp_num_hidden_layers": 0,
            "num_attention_heads": \(qHeads),
            "num_key_value_heads": \(kvHeads),
            "head_dim": \(headDim),
            "attn_output_gate": true,
            "linear_num_key_heads": \(gdnKHeads),
            "linear_num_value_heads": \(gdnVHeads),
            "linear_key_head_dim": \(gdnKDim),
            "linear_value_head_dim": \(gdnVDim),
            "linear_conv_kernel_dim": \(convK),
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2
          }
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
        for l in 0..<4 {
            let blk = "model.layers.\(l)."
            let sb = UInt32(100 + l * 30)
            planes.append((blk + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: sb, amp: 0.05, offset: 1.0))))
            planes.append((blk + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: sb + 1, amp: 0.05, offset: 1.0))))
            if l == 1 || l == 2 {
                planes.append((blk + "mhc_attn.weight", .f32(shape: [mhcRows, streams * h], values: detVec(mhcRows * streams * h, seedInit: sb + 2, amp: 0.005))))
                planes.append((blk + "mhc_attn.bias", .f32(shape: [mhcRows], values: detVec(mhcRows, seedInit: sb + 3, amp: 0.05))))
                planes.append((blk + "mhc_ffn.weight", .f32(shape: [mhcRows, streams * h], values: detVec(mhcRows * streams * h, seedInit: sb + 4, amp: 0.005))))
                planes.append((blk + "mhc_ffn.bias", .f32(shape: [mhcRows], values: detVec(mhcRows, seedInit: sb + 5, amp: 0.05))))
            }
            if l < 2 {
                planes.append((blk + "linear_attn.in_proj_qkv.weight", .hq4(n: g.gdnQKVChannels, k: h, seed: sb + 6, scale: 0.008)))
                planes.append((blk + "linear_attn.in_proj_z.weight", .hq4(n: g.gdnZChannels, k: h, seed: sb + 7, scale: 0.008)))
                planes.append((blk + "linear_attn.in_proj_a.weight", .hq4(n: g.gdnVHeads, k: h, seed: sb + 8, scale: 0.008)))
                planes.append((blk + "linear_attn.in_proj_b.weight", .hq4(n: g.gdnVHeads, k: h, seed: sb + 9, scale: 0.008)))
                planes.append((blk + "linear_attn.conv1d.weight", .bf16(shape: [g.gdnQKVChannels, 1, convK],
                                                                        values: detVec(g.gdnQKVChannels * convK, seedInit: sb + 10, amp: 0.05))))
                planes.append((blk + "linear_attn.norm.weight", .bf16(shape: [g.gdnHeadDim], values: [Float](repeating: 1.0, count: g.gdnHeadDim))))
                planes.append((blk + "linear_attn.A_log", .f32(shape: [g.gdnVHeads], values: [Float](repeating: -1.0, count: g.gdnVHeads))))
                planes.append((blk + "linear_attn.dt_bias", .f32(shape: [g.gdnVHeads], values: [Float](repeating: 0.0, count: g.gdnVHeads))))
                planes.append((blk + "linear_attn.out_proj.weight", .hq4(n: h, k: g.gdnZChannels, seed: sb + 11, scale: 0.008)))
            } else {
                planes.append((blk + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: sb + 12, scale: 0.008)))
                planes.append((blk + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: sb + 13, scale: 0.008)))
                planes.append((blk + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: sb + 14, scale: 0.008)))
                planes.append((blk + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: sb + 15, scale: 0.008)))
                planes.append((blk + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
                planes.append((blk + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            }
            planes.append((blk + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: sb + 16, scale: 0.008)))
            planes.append((blk + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: sb + 17, scale: 0.008)))
            planes.append((blk + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: sb + 18, scale: 0.008)))
        }

        var file = Data("QWP1".utf8)
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
                "model_id": "sensenova_loop",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-mhc-batch-parity",
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
