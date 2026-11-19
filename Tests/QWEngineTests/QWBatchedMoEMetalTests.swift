// QWBatchedMoEMetalTests.swift — Batched decode (B=2..4) resident MoE parity vs serial decode.
import CryptoKit
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL
import QWQuant
@testable import QWEngine

final class QWBatchedMoEMetalTests: XCTestCase {
    private enum PlaneSpec {
        case f32(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Metal batched MoE parity tests require Apple Silicon GPU"
        )
    }

    func testBatchedDecodeB2ToB4ResidentMoEMatchesSerialDecode() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-batch-moe-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticQwen3MoEPack(at: rootDir)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertTrue(driver.canRunBatchedDecode)
        XCTAssertTrue(driver.canFusePreMoENorm(weights: driver.layersW[0], rows: 4))

        let prompts: [[Int]] = [
            [1, 10, 20, 30, 40],
            [1, 15, 25, 35, 45, 55],
            [1, 22, 32, 42, 52, 62, 72],
            [1, 33, 44, 55, 66],
        ]
        let maxDecode = 6
        let serialOutputs = try prompts.map { prompt in
            try driver.generate(
                request: QWGenerationRequest(prompt: prompt, maxNew: maxDecode, sampling: .greedy),
                onToken: nil).tokens
        }

        for bCount in 2...min(4, driver.freeSlots.count) {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer {
                for s in sessions where !s.sessionFinished { driver.finishSession(s) }
            }
            for idx in 0..<bCount {
                let s = try driver.makeSession(promptTokens: prompts[idx].count + maxDecode)
                let tok0 = try driver.prefill(s, tokens: prompts[idx])
                XCTAssertEqual(tok0, serialOutputs[idx][0], "B=\(bCount) prefill slot \(idx)")
                sessions.append(s)
                pending.append(tok0)
            }
            for step in 1..<4 {
                let (nextIds, stepMs, _) = try driver.forwardBatchStep(sessions, tokens: pending)
                XCTAssertGreaterThan(stepMs, 0)
                for idx in 0..<bCount {
                    XCTAssertEqual(nextIds[idx], serialOutputs[idx][step],
                                   "B=\(bCount) step \(step) slot \(idx) aliased MoE bridge mismatch")
                }
                pending = nextIds
            }
        }
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

    private static func makeHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
        var seed = seedInit
        var out = Data(count: n * groups * 20)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * 20
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

    private static func buildSyntheticQwen3MoEPack(at rootDir: URL) throws {
        let h = 256, v = 256, moeInter = 64, qHeads = 8, kvHeads = 2, headDim = 32
        let experts = 32, topK = 4
        let cfgJSON = """
        {
          "model_type": "qwen3_moe",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(moeInter),
          "moe_intermediate_size": \(moeInter),
          "num_experts": \(experts),
          "num_experts_per_tok": \(topK),
          "norm_topk_prob": true,
          "num_hidden_layers": 1,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": 512,
          "rms_norm_eps": 1e-6,
          "rope_theta": 500000.0,
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

        let blk = "model.layers.0."
        let planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02)),
            (blk + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 21, amp: 0.05, offset: 1.0))),
            (blk + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 22, amp: 0.05, offset: 1.0))),
            (blk + "self_attn.q_proj.weight", .hq4(n: g.qProjOut, k: h, seed: 31, scale: 0.01)),
            (blk + "self_attn.k_proj.weight", .hq4(n: kvHeads * headDim, k: h, seed: 32, scale: 0.01)),
            (blk + "self_attn.v_proj.weight", .hq4(n: kvHeads * headDim, k: h, seed: 33, scale: 0.01)),
            (blk + "self_attn.o_proj.weight", .hq4(n: h, k: qHeads * headDim, seed: 34, scale: 0.01)),
            (blk + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            (blk + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))),
            (blk + "mlp.gate.weight", .f32(shape: [experts, h], values: detVec(experts * h, seedInit: 41, amp: 0.1))),
            (blk + "mlp.experts.gate_up_proj", .hq4(n: experts * 2 * moeInter, k: h, seed: 51, scale: 0.01)),
            (blk + "mlp.experts.down_proj", .hq4(n: experts * h, k: moeInter, seed: 52, scale: 0.01)),
        ]

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
                    withUnsafeBytes(of: &le) { (b: UnsafeRawBufferPointer) in raw.append(contentsOf: b) }
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
                    withUnsafeBytes(of: bf) { (b: UnsafeRawBufferPointer) in raw.append(contentsOf: b) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "qwen3_moe",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-moe-batch-parity",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (b: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: b) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }
}
