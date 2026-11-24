// QWBailingVL2LayerE2ETests.swift — 2-layer+NextN E2E parity test for generalized KDA, MLA, Group-MoE, and 3D MRoPE.
import CryptoKit
import Foundation
@testable import QWCore
@testable import QWEngine
import QWHAL
import QWQuant
import XCTest

final class QWBailingVL2LayerE2ETests: XCTestCase {
    private enum SynthPlane {
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case hq8g32(n: Int, k: Int, seed: UInt32, scale: Float)
        case shvq3g64(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    private func requireGPU() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal GPU available")
        }
    }

    private static func detVec(_ n: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<n).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    private static func makeHQ8G32(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let gSize = 32
        precondition(k % gSize == 0)
        let nb = QWAQBlock.bytes(bits: 8, group: gSize)
        var out = Data(count: n * (k / gSize) * nb)
        var s = seedInit
        out.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<(n * (k / gSize)) {
                let p = base.advanced(by: i * nb)
                p.storeBytes(of: Float16(scale).bitPattern.littleEndian, as: UInt16.self)
                s = s &* 1664525 &+ 1013904223
                let z = Float16(Float(s & 0x3F) * 1e-4).bitPattern
                p.advanced(by: 2).storeBytes(of: z.littleEndian, as: UInt16.self)
                for q in 0..<gSize {
                    s = s &* 1664525 &+ 1013904223
                    p.storeBytes(of: UInt8(truncatingIfNeeded: s >> 24), toByteOffset: 4 + q, as: UInt8.self)
                }
            }
        }
        return out
    }

    private static func makeSHVQ3G64(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let gSize = 64
        precondition(k % gSize == 0)
        let nb = QWSHVQBlock.bytes(bits: 3, group: gSize)
        var out = Data(count: n * (k / gSize) * nb)
        var s = seedInit
        out.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<(n * (k / gSize)) {
                let p = base.advanced(by: i * nb)
                p.storeBytes(of: Float16(scale).bitPattern.littleEndian, as: UInt16.self)
                for b in 2..<nb {
                    s = s &* 1664525 &+ 1013904223
                    p.storeBytes(of: UInt8(truncatingIfNeeded: s >> 16), toByteOffset: b, as: UInt8.self)
                }
            }
        }
        return out
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func buildSyntheticBailingPack(at rootDir: URL) throws -> QWModelDescriptor {
        let h = 256
        let v = 256
        let inter = 256
        let linV = 2
        let linHd = 128
        let fgbC = linV * linHd
        let qHeads = 8
        let nopeD = 128
        let ropeD = 64
        let vD = 128
        let kvRank = 512
        let nExp = 32
        let topK = 4
        let moeInter = 64

        let cfgJSON = """
        {
          "architectures": ["BailingMoeV3VLForConditionalGeneration"],
          "model_type": "bailing_moe_v3_vl",
          "tie_word_embeddings": false,
          "image_token_id": 250,
          "video_token_id": 251,
          "vision_start_token_id": 252,
          "vision_end_token_id": 253,
          "mrope_section": [8, 12, 12],
          "vision_config": {
            "model_type": "qwen3_moe_vit",
            "depth": 1,
            "hidden_size": 64,
            "num_heads": 2,
            "intermediate_size": 128,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": \(h),
            "num_position_embeddings": 16,
            "disable_merger_proj": true
          },
          "text_config": {
            "model_type": "bailing_moe_v3_text",
            "vocab_size": \(v),
            "hidden_size": \(h),
            "intermediate_size": \(inter),
            "num_hidden_layers": 2,
            "layer_group_size": 2,
            "first_k_dense_replace": 1,
            "num_attention_heads": \(qHeads),
            "num_key_value_heads": \(qHeads),
            "head_dim": \(linHd),
            "q_lora_rank": null,
            "kv_lora_rank": \(kvRank),
            "qk_nope_head_dim": \(nopeD),
            "qk_rope_head_dim": \(ropeD),
            "v_head_dim": \(vD),
            "use_mla_nope": false,
            "gated_attention_proj_granularity_type": "head_wise",
            "no_kda_lora": true,
            "kda_safe_gate": true,
            "kda_lower_bound": -5.0,
            "linear_num_value_heads": \(linV),
            "num_kv_heads_for_linear_attn": \(linV),
            "short_conv_kernel_size": 4,
            "num_experts": \(nExp),
            "num_experts_per_tok": \(topK),
            "n_group": 4,
            "topk_group": 2,
            "moe_intermediate_size": \(moeInter),
            "moe_shared_expert_intermediate_size": \(moeInter),
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "score_function": "sigmoid",
            "moe_router_enable_expert_bias": true,
            "expert_swiglu_limit_list": [0.0, 4.0],
            "share_expert_swiglu_limit_list": [0.0, 4.0],
            "partial_rotary_factor": 0.5,
            "rope_theta": 100000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "eos_token_id": 2,
            "num_nextn_predict_layers": 1,
            "mtp_use_kda": false
          }
        }
        """

        let fm = FileManager.default
        let assetsDir = rootDir.appendingPathComponent("assets")
        try fm.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        let cfgData = Data(cfgJSON.utf8)
        try cfgData.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)

        let descriptor = try QWModelDescriptor.decode(configJSON: cfgData)
        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, SynthPlane)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 101, amp: 0.2))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 102, amp: 0.04, offset: 1.0))),
            ("lm_head.weight", .hq8g32(n: v, k: h, seed: 103, scale: 0.015)),
        ]

        // Layer 0: full-rank KDA + Dense MLP
        let b0 = "model.layers.0."
        planes.append((b0 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 201, amp: 0.03, offset: 1.0))))
        planes.append((b0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 202, amp: 0.03, offset: 1.0))))
        for (idx, p) in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"].enumerated() {
            planes.append((b0 + "attention.\(p).weight", .hq8g32(n: fgbC, k: h, seed: UInt32(210 + idx), scale: 0.004)))
        }
        planes.append((b0 + "attention.b_proj.weight", .bf16(shape: [linV, h], values: detVec(linV * h, seedInit: 215, amp: 0.1))))
        for (idx, c) in ["q_conv1d", "k_conv1d", "v_conv1d"].enumerated() {
            planes.append((b0 + "attention.\(c).weight", .bf16(shape: [fgbC, 1, 4], values: detVec(fgbC * 4, seedInit: UInt32(220 + idx), amp: 0.05))))
        }
        planes.append((b0 + "attention.o_norm.weight", .bf16(shape: [linHd], values: detVec(linHd, seedInit: 223, amp: 0.03, offset: 1.0))))
        planes.append((b0 + "attention.A_log", .f32(shape: [linV], values: detVec(linV, seedInit: 224, amp: 0.1, offset: -1.0))))
        planes.append((b0 + "attention.dt_bias", .f32(shape: [fgbC], values: detVec(fgbC, seedInit: 225, amp: 0.05, offset: -0.5))))
        planes.append((b0 + "attention.o_proj.weight", .hq8g32(n: h, k: fgbC, seed: 226, scale: 0.004)))
        planes.append((b0 + "mlp.gate_proj.weight", .hq8g32(n: inter, k: h, seed: 230, scale: 0.004)))
        planes.append((b0 + "mlp.up_proj.weight", .hq8g32(n: inter, k: h, seed: 231, scale: 0.004)))
        planes.append((b0 + "mlp.down_proj.weight", .hq8g32(n: h, k: inter, seed: 232, scale: 0.004)))

        func appendMLAAndGroupMoE(layer idx: Int, seedBase: UInt32) {
            let b = "model.layers.\(idx)."
            planes.append((b + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase, amp: 0.03, offset: 1.0))))
            planes.append((b + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase + 1, amp: 0.03, offset: 1.0))))
            planes.append((b + "attention.q_proj.weight", .hq8g32(n: qHeads * (nopeD + ropeD), k: h, seed: seedBase + 2, scale: 0.004)))
            planes.append((b + "attention.kv_a_proj_with_mqa.weight", .hq8g32(n: kvRank + ropeD, k: h, seed: seedBase + 3, scale: 0.004)))
            planes.append((b + "attention.kv_a_layernorm.weight", .bf16(shape: [kvRank], values: detVec(kvRank, seedInit: seedBase + 4, amp: 0.03, offset: 1.0))))
            planes.append((b + "attention.kv_b_proj.weight", .hq8g32(n: qHeads * (nopeD + vD), k: kvRank, seed: seedBase + 5, scale: 0.004)))
            planes.append((b + "attention.dense.weight", .hq8g32(n: h, k: qHeads * vD, seed: seedBase + 6, scale: 0.004)))
            planes.append((b + "attention.g_proj.weight", .bf16(shape: [qHeads, h], values: detVec(qHeads * h, seedInit: seedBase + 7, amp: 0.1))))
            planes.append((b + "mlp.gate.weight", .f32(shape: [nExp, h], values: detVec(nExp * h, seedInit: seedBase + 8, amp: 0.04))))
            planes.append((b + "mlp.gate.e_score_correction_bias", .f32(shape: [nExp], values: detVec(nExp, seedInit: seedBase + 9, amp: 0.1))))
            planes.append((b + "mlp.shared_experts.gate_proj.weight", .hq8g32(n: moeInter, k: h, seed: seedBase + 10, scale: 0.004)))
            planes.append((b + "mlp.shared_experts.up_proj.weight", .hq8g32(n: moeInter, k: h, seed: seedBase + 11, scale: 0.004)))
            planes.append((b + "mlp.shared_experts.down_proj.weight", .hq8g32(n: h, k: moeInter, seed: seedBase + 12, scale: 0.004)))
            for e in 0..<nExp {
                let ep = b + "mlp.experts.\(e)."
                let es = seedBase + 100 + UInt32(e * 4)
                planes.append((ep + "gate_proj.weight", .shvq3g64(n: moeInter, k: h, seed: es, scale: 0.015)))
                planes.append((ep + "up_proj.weight", .shvq3g64(n: moeInter, k: h, seed: es + 1, scale: 0.015)))
                planes.append((ep + "down_proj.weight", .shvq3g64(n: h, k: moeInter, seed: es + 2, scale: 0.015)))
            }
        }

        appendMLAAndGroupMoE(layer: 1, seedBase: 300)
        // Layer 2: NextN MTP (MLA + Group-MoE)
        let b2 = "model.layers.2."
        planes.append((b2 + "enorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 500, amp: 0.03, offset: 1.0))))
        planes.append((b2 + "hnorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 501, amp: 0.03, offset: 1.0))))
        planes.append((b2 + "eh_proj.weight", .hq8g32(n: h, k: 2 * h, seed: 502, scale: 0.004)))
        planes.append((b2 + "final_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 503, amp: 0.03, offset: 1.0))))
        appendMLAAndGroupMoE(layer: 2, seedBase: 510)

        var file = Data("QWP1".utf8)
        file.append(Data(count: 8)) // uint64 header length trailer size, patched below
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(count: align - rem)) }
        }
        var entries: [String: [String: Any]] = [:]
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(64)
                let off = file.count
                for fv in vals {
                    var bits = UInt16(truncatingIfNeeded: fv.bitPattern >> 16).littleEndian
                    file.append(Data(bytes: &bits, count: 2))
                }
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .f32(let shape, let vals):
                padTo(64)
                let off = file.count
                for fv in vals {
                    var bits = fv.bitPattern.littleEndian
                    file.append(Data(bytes: &bits, count: 4))
                }
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .hq8g32(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeHQ8G32(n: n, k: k, seedInit: seed, scale: scale))
                entries[name] = ["shape": [n, k], "hq4_off": off, "hq": ["bits": 8, "group": 32]]
            case .shvq3g64(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeSHVQ3G64(n: n, k: k, seedInit: seed, scale: scale))
                entries[name] = ["shape": [n, k], "hq4_off": off, "hq": ["bits": 3, "group": 64, "vq": "shvq1"]]
            }
        }
        padTo(64)
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "bailing_moe_v3",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-bailing-vl-e2e",
                "sha256_source": "synthetic",
                "quant_recorded": "dense_hq8_g32__expert_shvq3_g64",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        var hdrLenLE = UInt64(hdrData.count).littleEndian
        file.append(hdrData)
        withUnsafeBytes(of: &hdrLenLE) { bytes in
            file.replaceSubrange(4..<12, with: bytes)
        }
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
        return descriptor
    }

    func testBailingVL2LayerTextAndMultimodalNextNGreedyParity() throws {
        try requireGPU()
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_bailing_e2e_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        let desc = try Self.buildSyntheticBailingPack(at: rootDir)
        XCTAssertEqual(desc.kdaConfig?.gateProjection, .fullRank)
        XCTAssertTrue(desc.kdaConfig?.isFullRank == true)
        XCTAssertTrue(desc.mlaConfig?.isDirectQ == true)
        XCTAssertEqual(desc.moeRoutingConfig?.nGroup, 4)
        XCTAssertEqual(desc.swigluClampConfig?.mode, .postSilu)
        XCTAssertEqual(desc.mlaConfig?.rope?.mropeLayout, .bailingVideoRope)

        let textPrompt = [1, 7, 19, 42, 88, 123, 5, 9, 31, 64]
        let maxNew = 12
        var baseline: [Int] = []
        do {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 512, kvMode: .q4,
                mtpDrafts: 0, mtpBackend: nil, mtpAdaptiveDepth: false,
                mtpRelaxedMargin: 0.0, mtpDraftStrategy: "mtp", prefillChunkSize: 32)
            let session = try driver.makeSession(promptTokens: textPrompt.count)
            defer { driver.finishSession(session) }
            baseline = try driver.generate(session: session, prompt: textPrompt, maxNew: maxNew, stopIds: [], onToken: nil)
            XCTAssertEqual(baseline.count, maxNew)
        }

        for (drafts, strategy) in [(1, "mtp"), (2, "hybrid")] {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 512, kvMode: .q4,
                mtpDrafts: drafts, mtpBackend: .standard, mtpAdaptiveDepth: false,
                mtpRelaxedMargin: 0.0, mtpDraftStrategy: strategy, prefillChunkSize: 32)
            let session = try driver.makeSession(promptTokens: textPrompt.count)
            defer { driver.finishSession(session) }
            let out = try driver.generateExperimentalGreedy(session: session, prompt: textPrompt, maxNew: maxNew)
            XCTAssertEqual(out, baseline, "Bailing NextN parity mismatch for drafts=\(drafts), strategy=\(strategy)")
        }

        // Multimodal request with QWVisualSpan (gridT=1, gridH=4, gridW=4, spatialMerge=2 -> 4 visual tokens)
        let mmPrompt = [1, 7, 19, 250, 250, 250, 250, 42, 88, 123, 5, 9]
        let span = QWVisualSpan(
            tokenOffset: 3, tokenCount: 4,
            embeddings: Self.detVec(4 * 256, seedInit: 999, amp: 0.1),
            gridT: 1, gridH: 4, gridW: 4, spatialMergeSize: 2)

        // Verify direct prefill records expected 3D MRoPE delta (-2 for 4 merged visual tokens in 2x2 grid)
        do {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 512, kvMode: .q4,
                mtpDrafts: 0, prefillChunkSize: 32)
            driver.activeVisualSpans = [span]
            defer { driver.activeVisualSpans = [] }
            let session = try driver.makeSession(promptTokens: mmPrompt.count)
            defer { driver.finishSession(session) }
            _ = try driver.prefill(session, tokens: mmPrompt, mode: nil)
            XCTAssertEqual(session.roundRecord.mropeDelta, -3)
        }

        var mmBaseline: [Int] = []
        do {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 512, kvMode: .f16,
                mtpDrafts: 0, mtpBackend: nil, mtpRelaxedMargin: 0.0, prefillChunkSize: 32)
            let req = QWGenerationRequest(
                prompt: mmPrompt, maxNew: 8, sampling: .greedy,
                speculative: QWSpeculativeParameters(
                    enabled: false, mtpDrafts: 0, mtpBackend: nil,
                    mtpRelaxedMargin: 0.0, mtpDraftStrategy: "mtp"),
                visualSpans: [span])
            mmBaseline = try driver.generate(request: req, onToken: nil).tokens
            XCTAssertEqual(mmBaseline.count, 8)
        }
        do {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 512, kvMode: .f16,
                mtpDrafts: 1, mtpBackend: .standard, mtpAdaptiveDepth: false,
                mtpRelaxedMargin: 0.0, mtpDraftStrategy: "mtp", prefillChunkSize: 32)
            let req = QWGenerationRequest(
                prompt: mmPrompt, maxNew: 8, sampling: .greedy,
                speculative: QWSpeculativeParameters(
                    enabled: true, mtpDrafts: 1, mtpBackend: "standard",
                    mtpRelaxedMargin: 0.0, mtpDraftStrategy: "mtp", adaptiveDepth: false),
                visualSpans: [span])
            let mmMTP = try driver.generate(request: req, onToken: nil).tokens
            XCTAssertEqual(mmMTP, mmBaseline, "Multimodal MRoPE NextN greedy parity mismatch")
        }
    }

    func testBailingPyTorchOracleParityIfConfigured() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["QW_BAILING_ORACLE_DIR"], !dirPath.isEmpty else {
            return
        }
        try requireGPU()
        struct OracleRef: Decodable {
            let pack_dir: String
            let text_prompt: [Int]
            let text_expected_logits: [Float]
            let mm_prompt: [Int]?
            let mm_span_offset: Int?
            let mm_span_count: Int?
            let mm_embeddings: [Float]?
            let mm_grid_t: Int?
            let mm_grid_h: Int?
            let mm_grid_w: Int?
            let mm_spatial_merge: Int?
            let mm_expected_logits: [Float]?
        }
        let refURL = URL(fileURLWithPath: dirPath).appendingPathComponent("oracle_ref.json")
        let ref = try JSONDecoder().decode(OracleRef.self, from: Data(contentsOf: refURL))
        let driver = try QWEngineDriver(
            packPath: ref.pack_dir, maxTokens: 512, ropeRows: 512, kvMode: .f16,
            mtpDrafts: 0, prefillChunkSize: 32)

        func assertLogitsMatch(prompt: [Int], spans: [QWVisualSpan], expected: [Float], label: String) throws {
            driver.activeVisualSpans = spans
            defer { driver.activeVisualSpans = [] }
            let session = try driver.makeSession(promptTokens: prompt.count)
            defer { driver.finishSession(session) }
            _ = try driver.prefill(session, tokens: prompt, mode: nil)
            let got = driver.logitVector()
            XCTAssertEqual(got.count, expected.count, "\(label) vocab mismatch")
            var maxDiff: Float = 0
            for (a, b) in zip(got, expected) { maxDiff = max(maxDiff, abs(a - b)) }
            let argGot = got.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1
            let argExp = expected.enumerated().max(by: { $0.element < $1.element })?.offset ?? -2
            XCTAssertEqual(argGot, argExp, "\(label) greedy argmax mismatch (maxDiff=\(maxDiff))")
            XCTAssertLessThan(maxDiff, 5e-3, "\(label) max_abs_diff=\(maxDiff) >= 5e-3")
        }

        try assertLogitsMatch(prompt: ref.text_prompt, spans: [], expected: ref.text_expected_logits, label: "text")
        if let mmPrompt = ref.mm_prompt, let off = ref.mm_span_offset, let cnt = ref.mm_span_count,
           let emb = ref.mm_embeddings, let gt = ref.mm_grid_t, let gh = ref.mm_grid_h,
           let gw = ref.mm_grid_w, let ms = ref.mm_spatial_merge, let exp = ref.mm_expected_logits {
            let span = QWVisualSpan(
                tokenOffset: off, tokenCount: cnt, embeddings: emb,
                gridT: gt, gridH: gh, gridW: gw, spatialMergeSize: ms)
            try assertLogitsMatch(prompt: mmPrompt, spans: [span], expected: exp, label: "multimodal")
        }
    }
}
