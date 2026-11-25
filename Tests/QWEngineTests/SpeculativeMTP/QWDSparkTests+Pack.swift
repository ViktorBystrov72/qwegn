// QWDSparkTests+Pack.swift — Synthetic 2-target-layer + 1-draft-layer QWP1 builder for DSpark E2E tests.

import CryptoKit
import Foundation
import QWCore
import QWQuant
@testable import QWEngine

extension QWDSparkTests {
    private enum SynthPlane {
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case hq8g32(n: Int, k: Int, seed: UInt32, scale: Float)
        case shvq(bits: Int, group: Int, shape: [Int], seed: UInt32, scale: Float)
    }

    private static func writeQWP1(
        to url: URL,
        planes: [(String, SynthPlane)],
        provenance: [String: Any]
    ) throws {
        var file = Data("QWP1".utf8) + Data(repeating: 0, count: 8)
        var entries: [String: [String: Any]] = [:]
        func pad(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        pad(256)
        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                pad(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    var bf = bf16Bits(val).littleEndian
                    withUnsafeBytes(of: &bf) { (buf: UnsafeRawBufferPointer) in raw.append(contentsOf: buf) }
                }
                file.append(raw)
                pad(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .f32(let shape, let vals):
                pad(256)
                let off = file.count
                vals.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in file.append(contentsOf: buf) }
                pad(256)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .hq8g32(let n, let k, let seed, let scale):
                pad(16_384)
                let off = file.count
                file.append(makeHQ8G32Data(n: n, k: k, seedInit: seed, scale: scale))
                pad(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off, "hq": ["bits": 8, "group": 32]]
            case .shvq(let bits, let group, let shape, let seed, let scale):
                let k = shape.last ?? 1
                let n = shape.dropLast().reduce(1, *)
                pad(16_384)
                let off = file.count
                file.append(makeSHVQData(bits: bits, group: group, n: n, k: k, seedInit: seed, scale: scale))
                pad(16_384)
                entries[name] = ["shape": shape, "hq4_off": off, "hq": ["bits": bits, "group": group, "vq": "shvq1"]]
            }
        }
        let hdrData = try JSONSerialization.data(
            withJSONObject: ["provenance": provenance, "tensors": entries],
            options: [.sortedKeys]
        )
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (buf: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: buf) }
        file.append(hdrData)
        try file.write(to: url)
    }

    static func buildSyntheticTargetAndDSparkPack(at rootDir: URL) throws {
        let h = 256, v = 256, inter = 256, qHeads = 16, hd = 512, nope = 448, rope = 64
        let qLora = 128, oGroups = 2, oLora = 128, compRatio = 4, win = 8, streams = 4
        let nExp = 32, topK = 4, moeInter = 64, moeLat = 64, mhcRows = 2 * streams + streams * streams
        let markovRank = 64

        let cfgJSON = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": \(v), "hidden_size": \(h), "intermediate_size": \(inter),
          "num_hidden_layers": 2, "max_position_embeddings": 16384, "rms_norm_eps": 1e-6,
          "num_attention_heads": \(qHeads), "num_key_value_heads": 1, "head_dim": \(hd),
          "qk_nope_head_dim": \(nope), "qk_rope_head_dim": \(rope), "v_head_dim": \(hd),
          "compress_ratios": [\(compRatio), -2], "default_compress_ratio": \(compRatio),
          "q_lora_rank": \(qLora), "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": \(nope), "compress_qk_rope_head_dim": \(rope),
          "compress_v_head_dim": \(hd), "o_groups": \(oGroups), "o_lora_rank": \(oLora),
          "sliding_window": \(win), "compress_rope_theta": 160000.0,
          "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 1024, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 8000000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa",
          "hybrid_mla_num_attention_heads": \(qHeads), "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": 64, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 64,
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true, "hc_mult": \(streams), "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
          "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1, "moe_layer_freq": 1, "n_routed_experts": \(nExp),
          "num_experts_per_tok": \(topK), "moe_intermediate_size": \(moeInter), "moe_latent_size": \(moeLat),
          "latent_moe_use_norm": true, "moe_split_feature_routing": true,
          "routed_scaling_factor": 2.0, "routed_scaling_factor_learnable": true, "n_shared_experts": 1,
          "scoring_func": "sigmoid", "norm_topk_prob": true, "topk_method": "noaux_tc",
          "multimax": "lm_head", "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2,
          "dspark_block_size": 7, "dspark_markov_rank": \(markovRank), "dspark_noise_token_id": 3,
          "dspark_target_layer_ids": [0, 1], "dspark_num_hidden_layers": 1,
          "dspark_sliding_window": 32, "dspark_rope_theta": 8000000.0, "enable_confidence_head": true
        }
        """.data(using: .utf8)!
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)

        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        let qlsaURL = assetsDir.appendingPathComponent("model.qlsa")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)

        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: nope, indexDim: 64, seed: 7)
        try QWLSASidecar.write(
            to: qlsaURL,
            provenance: QWLSASidecar.Header.Provenance(
                modelId: "ernie5_5_moe", geometryDigest: descriptor.geometryDigestV2Hex,
                method: "weight_svd_v1", indexDim: 64, poolSize: compRatio,
                recommendedBudget: 64, sinkTokens: compRatio, windowTokens: win
            ),
            layerTensors: [0: (wq: basis, wk: basis)], wqShape: [64, nope], wkShape: [64, nope]
        )
        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL, qlsaURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = [
                "sha256": SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined(),
                "bytes": d.count,
            ]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))
        let mfSha = SHA256.hash(data: mfData).map { String(format: "%02x", $0) }.joined()

        func eyePlus(_ n: Int, seed: UInt32) -> [Float] {
            var m = detFloats(n * n, seedInit: seed, amp: 0.01)
            for i in 0..<n { m[i * n + i] += 1.0 }
            return m
        }
        func mhcPlanes(_ blk: String, seedBase: UInt32, into arr: inout [(String, SynthPlane)]) {
            arr.append((blk + "mhc_attn.weight", .f32(shape: [mhcRows, streams * h], values: detFloats(mhcRows * streams * h, seedInit: seedBase, amp: 0.005))))
            arr.append((blk + "mhc_attn.bias", .f32(shape: [mhcRows], values: detFloats(mhcRows, seedInit: seedBase + 1, amp: 0.05))))
            arr.append((blk + "mhc_ffn.weight", .f32(shape: [mhcRows, streams * h], values: detFloats(mhcRows * streams * h, seedInit: seedBase + 2, amp: 0.005))))
            arr.append((blk + "mhc_ffn.bias", .f32(shape: [mhcRows], values: detFloats(mhcRows, seedInit: seedBase + 3, amp: 0.05))))
        }

        var tgt: [(String, SynthPlane)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detFloats(v * h, seedInit: 101, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detFloats(h, seedInit: 102, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq8g32(n: v, k: h, seed: 103, scale: 0.015)),
            ("model.mhc_contract.weight", .f32(shape: [streams, streams * h], values: detFloats(streams * streams * h, seedInit: 104, amp: 0.01))),
            ("model.mhc_contract.bias", .f32(shape: [streams], values: [0.4, 0.3, 0.2, 0.1])),
            ("lm_head.multimax", .f32(shape: [8], values: [14.0, 15.0, 8.0, 0.5, -0.05, 0.25, -0.1, -0.02])),
        ]
        let b0 = "model.layers.0."
        tgt.append((b0 + "input_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        tgt.append((b0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        mhcPlanes(b0, seedBase: 210, into: &tgt)
        tgt.append((b0 + "self_attn.fused_in_proj.weight", .hq8g32(n: qLora + 3 * hd, k: h, seed: 220, scale: 0.004)))
        tgt.append((b0 + "self_attn.q_norm.weight", .bf16(shape: [qLora], values: [Float](repeating: 1.0, count: qLora))))
        tgt.append((b0 + "self_attn.kv_norm.weight", .bf16(shape: [hd], values: [Float](repeating: 1.0, count: hd))))
        tgt.append((b0 + "self_attn.comp_norm.weight", .bf16(shape: [hd], values: [Float](repeating: 1.0, count: hd))))
        tgt.append((b0 + "self_attn.compressor.ape", .f32(shape: [compRatio, hd], values: detFloats(compRatio * hd, seedInit: 225, amp: 0.02))))
        tgt.append((b0 + "self_attn.q_up_gate_proj.weight", .hq8g32(n: qHeads * hd + oGroups * oLora, k: qLora, seed: 226, scale: 0.004)))
        tgt.append((b0 + "self_attn.wo_a.weight", .hq8g32(n: oGroups * oLora, k: (qHeads / oGroups) * hd, seed: 227, scale: 0.003)))
        tgt.append((b0 + "self_attn.o_proj.weight", .hq8g32(n: h, k: oGroups * oLora, seed: 228, scale: 0.004)))
        tgt.append((b0 + "self_attn.vha_mix.weight", .f32(shape: [qHeads, qHeads], values: eyePlus(qHeads, seed: 229))))
        tgt.append((b0 + "self_attn.attn_sink", .f32(shape: [qHeads], values: detFloats(qHeads, seedInit: 230, amp: 0.2, offset: -1.5))))
        tgt.append((b0 + "mlp.gate_proj.weight", .hq8g32(n: inter, k: h, seed: 240, scale: 0.004)))
        tgt.append((b0 + "mlp.up_proj.weight", .hq8g32(n: inter, k: h, seed: 241, scale: 0.004)))
        tgt.append((b0 + "mlp.down_proj.weight", .hq8g32(n: h, k: inter, seed: 242, scale: 0.004)))

        let b1 = "model.layers.1."
        tgt.append((b1 + "input_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        tgt.append((b1 + "post_attention_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        mhcPlanes(b1, seedBase: 310, into: &tgt)
        tgt.append((b1 + "self_attn.q_a_proj.weight", .hq8g32(n: 64, k: h, seed: 320, scale: 0.004)))
        tgt.append((b1 + "self_attn.kv_a_proj_with_mqa.weight", .hq8g32(n: 576, k: h, seed: 321, scale: 0.004)))
        tgt.append((b1 + "self_attn.indexer.wk.weight", .hq8g32(n: 128, k: h, seed: 322, scale: 0.004)))
        tgt.append((b1 + "self_attn.q_a_layernorm.weight", .bf16(shape: [64], values: [Float](repeating: 1.0, count: 64))))
        tgt.append((b1 + "self_attn.kv_a_layernorm.weight", .bf16(shape: [512], values: [Float](repeating: 1.0, count: 512))))
        tgt.append((b1 + "self_attn.indexer.k_norm.weight", .bf16(shape: [128], values: [Float](repeating: 1.0, count: 128))))
        tgt.append((b1 + "self_attn.indexer.k_norm.bias", .bf16(shape: [128], values: [Float](repeating: 0.0, count: 128))))
        tgt.append((b1 + "self_attn.indexer.weights_proj.weight", .f32(shape: [64, h], values: detFloats(64 * h, seedInit: 325, amp: 0.02))))
        tgt.append((b1 + "self_attn.q_b_gate_idx_proj.weight", .hq8g32(n: qHeads * 512 + 64 * 128, k: 64, seed: 326, scale: 0.004)))
        tgt.append((b1 + "self_attn.kv_b_proj.weight", .hq8g32(n: qHeads * 448, k: 512, seed: 327, scale: 0.004)))
        tgt.append((b1 + "self_attn.o_proj.weight", .hq8g32(n: h, k: qHeads * 256, seed: 328, scale: 0.003)))
        tgt.append((b1 + "self_attn.vha_mix.weight", .f32(shape: [qHeads, qHeads], values: eyePlus(qHeads, seed: 329))))
        tgt.append((b1 + "self_attn.attn_sink", .f32(shape: [qHeads], values: detFloats(qHeads, seedInit: 330, amp: 0.2, offset: -1.5))))
        tgt.append((b1 + "mlp.gate.weight", .f32(shape: [2 * nExp, h], values: detFloats(2 * nExp * h, seedInit: 340, amp: 0.08))))
        tgt.append((b1 + "mlp.gate.e_score_correction_bias", .f32(shape: [nExp], values: detFloats(nExp, seedInit: 341, amp: 0.1))))
        tgt.append((b1 + "mlp.routed_scaling_factor_param", .f32(shape: [nExp], values: [Float](repeating: 1.5, count: nExp))))
        tgt.append((b1 + "mlp.fc1.weight", .hq8g32(n: moeLat, k: h, seed: 342, scale: 0.004)))
        tgt.append((b1 + "mlp.latent_norm.weight", .bf16(shape: [moeLat], values: [Float](repeating: 1.0, count: moeLat))))
        tgt.append((b1 + "mlp.fc2.weight", .hq8g32(n: h, k: moeLat, seed: 343, scale: 0.004)))
        tgt.append((b1 + "mlp.shared_experts.gate_proj.weight", .hq8g32(n: moeInter, k: h, seed: 344, scale: 0.004)))
        tgt.append((b1 + "mlp.shared_experts.up_proj.weight", .hq8g32(n: moeInter, k: h, seed: 345, scale: 0.004)))
        tgt.append((b1 + "mlp.shared_experts.down_proj.weight", .hq8g32(n: h, k: moeInter, seed: 346, scale: 0.004)))
        for e in 0..<nExp {
            let ep = b1 + "mlp.experts.\(e).", sb = UInt32(400 + e * 10)
            tgt.append((ep + "gate_proj.weight", .shvq(bits: 15, group: 64, shape: [moeInter, moeLat], seed: sb, scale: 0.03)))
            tgt.append((ep + "up_proj.weight", .shvq(bits: 15, group: 64, shape: [moeInter, moeLat], seed: sb + 1, scale: 0.03)))
            tgt.append((ep + "down_proj.weight", .shvq(bits: 3, group: 64, shape: [moeLat, moeInter], seed: sb + 2, scale: 0.03)))
        }

        let baseProv: [String: Any] = [
            "model_id": "ernie5_5_moe", "geometry_digest": descriptor.geometryDigestV2Hex,
            "source": "synthetic-dspark-e2e", "sha256_source": "synthetic",
            "quant_recorded": "dense_hq8_g32__expert_shvq15_g64__expdown_shvq3_g64",
            "assets_dir": "assets", "assets_manifest_sha256": mfSha,
        ]
        try writeQWP1(to: rootDir.appendingPathComponent("model.qwp"), planes: tgt, provenance: baseProv)

        let db = "dspark.layers.0."
        var ds: [(String, SynthPlane)] = [
            ("dspark.main_proj.weight", .hq8g32(n: h, k: 2 * h, seed: 600, scale: 0.004)),
            ("dspark.main_norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("dspark.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            ("dspark.mhc_contract.weight", .f32(shape: [streams, streams * h], values: detFloats(streams * streams * h, seedInit: 601, amp: 0.01))),
            ("dspark.mhc_contract.bias", .f32(shape: [streams], values: [0.25, 0.25, 0.25, 0.25])),
            ("dspark.markov_w1.weight", .bf16(shape: [v, markovRank], values: detFloats(v * markovRank, seedInit: 602, amp: 0.2))),
            ("dspark.markov_w2.weight", .hq8g32(n: v, k: markovRank, seed: 603, scale: 0.01)),
            ("dspark.confidence_head.weight", .bf16(shape: [1, h + markovRank], values: detFloats(h + markovRank, seedInit: 604, amp: 0.05))),
            (db + "input_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            (db + "post_attention_layernorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))),
            (db + "self_attn.linear_q_down_proj.weight", .hq8g32(n: qLora, k: h, seed: 620, scale: 0.004)),
            (db + "self_attn.linear_kv_proj.weight", .hq8g32(n: hd, k: h, seed: 621, scale: 0.004)),
            (db + "self_attn.q_norm.weight", .bf16(shape: [qLora], values: [Float](repeating: 1.0, count: qLora))),
            (db + "self_attn.kv_norm.weight", .bf16(shape: [hd], values: [Float](repeating: 1.0, count: hd))),
            (db + "self_attn.q_up_gate_proj.weight", .hq8g32(n: qHeads * hd + oGroups * oLora, k: qLora, seed: 622, scale: 0.004)),
            (db + "self_attn.wo_a.weight", .hq8g32(n: oGroups * oLora, k: (qHeads / oGroups) * hd, seed: 623, scale: 0.003)),
            (db + "self_attn.o_proj.weight", .hq8g32(n: h, k: oGroups * oLora, seed: 624, scale: 0.004)),
            (db + "self_attn.vha_mix.weight", .f32(shape: [qHeads, qHeads], values: eyePlus(qHeads, seed: 625))),
            (db + "self_attn.attn_sink", .f32(shape: [qHeads], values: detFloats(qHeads, seedInit: 626, amp: 0.2, offset: -1.5))),
            (db + "mlp.gate.weight", .f32(shape: [2 * nExp, h], values: detFloats(2 * nExp * h, seedInit: 630, amp: 0.08))),
            (db + "mlp.gate.e_score_correction_bias", .f32(shape: [nExp], values: detFloats(nExp, seedInit: 631, amp: 0.1))),
            (db + "mlp.routed_scaling_factor_param", .f32(shape: [nExp], values: [Float](repeating: 1.5, count: nExp))),
            (db + "mlp.fc1.weight", .hq8g32(n: moeLat, k: h, seed: 632, scale: 0.004)),
            (db + "mlp.latent_norm.weight", .bf16(shape: [moeLat], values: [Float](repeating: 1.0, count: moeLat))),
            (db + "mlp.fc2.weight", .hq8g32(n: h, k: moeLat, seed: 633, scale: 0.004)),
            (db + "mlp.shared_experts.gate_proj.weight", .hq8g32(n: moeInter, k: h, seed: 634, scale: 0.004)),
            (db + "mlp.shared_experts.up_proj.weight", .hq8g32(n: moeInter, k: h, seed: 635, scale: 0.004)),
            (db + "mlp.shared_experts.down_proj.weight", .hq8g32(n: h, k: moeInter, seed: 636, scale: 0.004)),
            (db + "mlp.experts.gate_up_proj", .shvq(bits: 15, group: 64, shape: [nExp, 2 * moeInter, moeLat], seed: 640, scale: 0.03)),
            (db + "mlp.experts.down_proj", .shvq(bits: 3, group: 64, shape: [nExp, moeLat, moeInter], seed: 641, scale: 0.03)),
        ]
        mhcPlanes(db, seedBase: 610, into: &ds)
        try writeQWP1(to: rootDir.appendingPathComponent("dspark.qwp"), planes: ds, provenance: baseProv)
    }
}
