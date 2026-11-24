// QWModelCapabilityTests.swift — Capability decoding, Bailing/DeepSeek/Kimi configs, and V2 parity.
import Foundation
import XCTest
@testable import QWCore

final class QWModelCapabilityTests: XCTestCase {

    func testBailingMoeV3VLSyntheticCapabilityDecode() throws {
        var expLimits = [Int](repeating: 0, count: 42)
        for i in 35..<42 { expLimits[i] = 4 }
        var shLimits = [Int](repeating: 0, count: 42)
        for i in 34..<40 { shLimits[i] = 5 }
        for i in 40..<42 { shLimits[i] = 7 }
        let root: [String: Any] = [
            "model_type": "bailing_moe_v3_vl",
            "architectures": ["BailingMoeV3VLForConditionalGeneration"],
            "tie_word_embeddings": false,
            "image_token_id": 157157,
            "video_token_id": 156909,
            "vision_start_token_id": 157158,
            "vision_end_token_id": 157159,
            "mrope_section": [8, 12, 12],
            "text_config": [
                "num_hidden_layers": 42,
                "hidden_size": 2560,
                "intermediate_size": 6144,
                "eos_token_id": 156895,
                "first_k_dense_replace": 2,
                "max_position_embeddings": 262144,
                "moe_intermediate_size": 768,
                "num_experts_per_tok": 8,
                "num_attention_heads": 32,
                "q_lora_rank": NSNull(),
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "num_experts": 512,
                "num_key_value_heads": 32,
                "rope_theta": 6_000_000,
                "rms_norm_eps": 1e-6,
                "head_dim": 128,
                "vocab_size": 157184,
                "partial_rotary_factor": 0.5,
                "moe_router_enable_expert_bias": true,
                "routed_scaling_factor": 2.5,
                "n_group": 8,
                "topk_group": 4,
                "use_qk_norm": true,
                "score_function": "sigmoid",
                "moe_shared_expert_intermediate_size": 768,
                "layer_group_size": 6,
                "num_kv_heads_for_linear_attn": 0,
                "use_mla_nope": false,
                "short_conv_kernel_size": 4,
                "gated_attention_proj_granularity_type": "head_wise",
                "no_kda_lora": true,
                "use_kda_lora": false,
                "kda_safe_gate": true,
                "kda_lower_bound": -5.0,
                "norm_topk_prob": true,
                "expert_swiglu_limit_list": expLimits,
                "share_expert_swiglu_limit_list": shLimits,
                "num_nextn_predict_layers": 1,
            ] as [String: Any],
            "vision_config": [
                "model_type": "qwen3_moe_vit",
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 4096,
                "num_position_embeddings": 2304,
                "disable_merger_proj": true,
            ] as [String: Any],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        let d = try QWModelDescriptor.decode(configJSON: data)
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.family, .bailingMoeV3)
        XCTAssertEqual(d.modelType, "bailing_moe_v3_vl")
        XCTAssertEqual(d.fullAttentionLayers, [5, 11, 17, 23, 29, 35, 41])
        XCTAssertEqual(d.linearAttentionLayers.count, 35)
        XCTAssertTrue(d.hasMLAAttention)
        XCTAssertTrue(d.ropeInterleave)

        let kda = try XCTUnwrap(d.kdaConfig)
        XCTAssertEqual(kda.gateProjection, .fullRank)
        XCTAssertTrue(kda.safeGate)
        XCTAssertEqual(kda.gateLowerBound, -5.0)
        XCTAssertEqual(kda.outputNormAct, .sigmoid)
        XCTAssertEqual(kda.qkHeads, 32)
        XCTAssertEqual(kda.vHeads, 32)
        XCTAssertEqual(kda.headDim, 128)

        let mla = try XCTUnwrap(d.mlaConfig)
        XCTAssertTrue(mla.isDirectQ)
        XCTAssertEqual(mla.kvLoraRank, 512)
        XCTAssertEqual(mla.qkNopeHeadDim, 128)
        XCTAssertEqual(mla.qkRopeHeadDim, 64)
        XCTAssertEqual(mla.vHeadDim, 128)
        XCTAssertEqual(mla.sparsity, .dense(slidingWindow: 0))
        XCTAssertEqual(mla.outputGateMode, .headWiseFromHidden)
        XCTAssertEqual(mla.rope?.interleave, true)
        XCTAssertEqual(mla.rope?.mropeLayout, .bailingVideoRope)

        let moe = try XCTUnwrap(d.moeRoutingConfig)
        XCTAssertEqual(moe.nGroup, 8)
        XCTAssertEqual(moe.topkGroup, 4)
        XCTAssertEqual(moe.groupTopM, 2)
        XCTAssertEqual(moe.sharedExpertHasGate, false)
        XCTAssertEqual(moe.scoreFunction, "sigmoid")
        XCTAssertTrue(moe.enableExpertBias)
        XCTAssertEqual(moe.routedScalingFactor, 2.5)
        XCTAssertTrue(moe.normTopKProb)

        let sw = try XCTUnwrap(d.swigluClampConfig)
        XCTAssertEqual(sw.mode, .postSilu)
        XCTAssertNil(sw.routedSpec(layer: 34))
        XCTAssertEqual(sw.routedSpec(layer: 35), QWSwiGLUClampSpec(limit: 4, postSilu: true))
        XCTAssertNil(sw.sharedSpec(layer: 33))
        XCTAssertEqual(sw.sharedSpec(layer: 34), QWSwiGLUClampSpec(limit: 5, postSilu: true))
        XCTAssertEqual(sw.sharedSpec(layer: 40), QWSwiGLUClampSpec(limit: 7, postSilu: true))
        XCTAssertEqual(sw.distinctSpecs().count, 3)

        let vis = try XCTUnwrap(d.visionTowerConfig)
        XCTAssertEqual(vis.outDim, 2560)
        XCTAssertEqual(vis.mergerIntermediateDim, 2560)
        XCTAssertTrue(vis.disableInternalMergerProj)
        XCTAssertEqual(vis.imageMean.first ?? 0, 0.48145466, accuracy: 1e-6)

        let mtp = try XCTUnwrap(d.mtpConfig)
        XCTAssertEqual(mtp.layers, 1)
        XCTAssertEqual(mtp.fusionMode, .concatEHProj)

        let geom = QWModelGeometry(descriptor: d)
        XCTAssertTrue(geom.mlaHasDirectQ)
        XCTAssertFalse(geom.mlaHasIndexer)
        XCTAssertFalse(geom.hasSubquadraticAttention)
        XCTAssertEqual(geom.kdaFgbChannels, 2 * 4096)
        XCTAssertTrue(geom.supportsGroupedNAXMoEPrefill)
    }

    func testLingRealSnapshotDecodeIfPresent() throws {
        let rel = ".cache/huggingface/hub/models--inclusionAI--Ling-3.2-air/snapshots/1613b1a4dcc6aa0d8f346606628cd9a92061a525/config.json"
        let path = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(rel).path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "Ling-3.2-air cache snapshot absent")
        let d = try QWModelDescriptor.decode(configJSON: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.family, .bailingMoeV3)
        XCTAssertEqual(d.hidden, 2560)
        XCTAssertEqual(d.numExperts, 512)
        XCTAssertEqual(d.visOut, 2560)
    }

    func testDeepSeekStyleGroupMoEAndLowRankKDAVariants() throws {
        let cfg: [String: Any] = [
            "model_type": "glm5_next",
            "hidden_size": 2048,
            "vocab_size": 128000,
            "intermediate_size": 4096,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 131072,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "rope_theta": 1_000_000.0,
            "eos_token_id": 1,
            "mla_use_nope": false,
            "q_lora_rank": 512,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "linear_attn_config": [
                "linear_type": "kda",
                "num_heads": 16,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "low_rank_dim": 64,
                "gate_lower_bound": -5.0,
            ],
        ]
        let d = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg))
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertEqual(d.kdaConfig?.gateProjection, .lowRank(dim: 64))
        XCTAssertEqual(d.mlaConfig?.qProjection, .factored(rank: 512))
        XCTAssertEqual(d.moeRoutingConfig?.nGroup, 8)
        XCTAssertEqual(d.moeRoutingConfig?.topkGroup, 4)
        XCTAssertEqual(d.moeRoutingConfig?.enableExpertBias, true)
    }

    func testGeometryV2IdentityPreservedOnQwen38_27B() {
        XCTAssertTrue(QWModelDescriptor.qwen38_27b.geometryDigestV2.hasPrefix("qwg-geo-v2:model_type=qwen3_5|"))
        XCTAssertEqual(
            QWModelDescriptor.qwen38_27b.geometryDigestV2Hex,
            QWModelDescriptor.qwen38_27b.geometryDigestV2Hex
        )
        XCTAssertEqual(QWModelDescriptor.qwen38_27b.verifyInternalConsistency(), [])
    }
}
