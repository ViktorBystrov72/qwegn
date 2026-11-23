// QWInkling2DescriptorTests.swift — Synthetic config decode, 48-visit execution plan,
// mixed NVFP4/MXFP8 quantization map, and admission pricing for `inkling2_loop`.
import Foundation
import XCTest
@testable import QWCore

final class QWInkling2DescriptorTests: XCTestCase {
    private func makeInkling2JSON() -> Data {
        let localLayers = (0..<24).filter { ($0 + 1) % 6 != 0 }
        var quant: [String: Any] = [
            "group_size": 16,
            "bits": 4,
            "mode": "nvfp4",
        ]
        for l in 0..<20 {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                quant["language_model.model.layers.\(l).mlp.switch_mlp.\(p)"] = [
                    "group_size": 16, "bits": 4, "mode": "nvfp4",
                ]
            }
        }
        for l in 20..<24 {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                quant["language_model.model.layers.\(l).mlp.switch_mlp.\(p)"] = [
                    "group_size": 32, "bits": 8, "mode": "mxfp8",
                ]
            }
        }
        let obj: [String: Any] = [
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "eos_token_id": 200006,
            "model_type": "inkling2_loop",
            "generation_config": ["eos_token_id": 200006, "pad_token_id": 199999],
            "mtp_config": [
                "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9216,
                "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
                "q_bias": false, "o_bias": false, "local_layer_ids": [0, 2],
                "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
                "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48],
            ],
            "text_config": [
                "model_type": "inkling2_loop_text",
                "model_max_length": 1048576,
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "d_rel": 16,
                "rel_extent": 1024,
                "q_bias": false,
                "o_bias": false,
                "log_scaling_n_floor": 8192,
                "log_scaling_alpha": 0.1,
                "rms_norm_eps": 1e-6,
                "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.627416997969522,
                "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": localLayers,
                "sconv_kernel_size": 4,
                "final_logit_softcapping": 30.0,
                "swa_head_dim": 128,
                "swa_num_attention_heads": 24,
                "swa_num_key_value_heads": 8,
                "sliding_window_size": 1024,
                "n_layers_per_attn_res_block": 6,
                "attn_res_normalize_values": true,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "n_shared_experts": 0,
                "intermediate_size": 1152,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
                "route_scale": 8.0,
                "use_gate_bias": true,
                "loop_config": [
                    "core_start": 6, "core_end": 18, "num_passes": 3,
                    "gate_rank": 64, "is_varloop": false,
                ],
            ],
            "vision_config": [
                "model_type": "inkling2_vision",
                "text_hidden_size": 2304, "temporal_patch_size": 2, "native_patch_size": 20,
                "adapter_merge_size": 2, "projection_hidden_dim": 4096, "projector_bias": false,
                "hidden_size": 1024, "num_hidden_layers": 23, "num_attention_heads": 16,
                "mlp_ratio": 4.0, "posemb_grid_size": 32, "layer_norm_eps": 1e-5,
            ],
            "quantization": quant,
        ]
        return try! JSONSerialization.data(withJSONObject: obj)
    }

    func testDecodeLoopGeometryMixedQuantAndPlan() throws {
        let d = try QWModelDescriptor.decode(configJSON: makeInkling2JSON())
        XCTAssertEqual(d.family, .inkling2Loop)
        XCTAssertEqual(d.role, .multimodalConditionalLM)
        XCTAssertTrue(d.capabilities.contains(.moe))
        XCTAssertTrue(d.capabilities.contains(.vision))
        XCTAssertTrue(d.capabilities.contains(.mtp))
        XCTAssertFalse(d.usesPlusOneRMSNorm)
        XCTAssertFalse(d.usesChatTemplateIds)
        XCTAssertEqual(d.eosTokenId, 200006)
        XCTAssertEqual(d.maxPositionEmbeddings, 1_048_576)
        XCTAssertEqual(d.moeIntermediate, 1152)
        XCTAssertEqual(d.numExperts, 128)
        XCTAssertEqual(d.expertsPerTok, 8)
        XCTAssertEqual(d.routedScalingFactor, 8.0, accuracy: 1e-6)
        XCTAssertTrue(d.moeRouterEnableExpertBias)
        XCTAssertEqual(d.executionPlan.count, 48)
        XCTAssertEqual(Array(d.executionPlan.prefix(6)), [0, 1, 2, 3, 4, 5])
        XCTAssertEqual(Array(d.executionPlan[6..<18]), Array(6..<18))
        XCTAssertEqual(Array(d.executionPlan[18..<30]), Array(6..<18))
        XCTAssertEqual(Array(d.executionPlan[30..<42]), Array(6..<18))
        XCTAssertEqual(Array(d.executionPlan.suffix(6)), Array(18..<24))

        for l in 0..<24 {
            let expectedSW = ((l + 1) % 6 == 0) ? 0 : 1024
            XCTAssertEqual(d.layerSlidingWindow(l), expectedSW, "layer \(l)")
        }

        let ink = try XCTUnwrap(d.inkling2Config)
        XCTAssertEqual(ink.unpaddedVocabSize, 200058)
        XCTAssertEqual(ink.expertModesCount(numLayers: 24).nvfp4, 20)
        XCTAssertEqual(ink.expertModesCount(numLayers: 24).mxfp8, 4)
        XCTAssertEqual(ink.expertProjSpec(layer: 0, projection: "down_proj").mode, .nvfp4)
        XCTAssertEqual(ink.expertProjSpec(layer: 23, projection: "gate_proj").mode, .mxfp8)
        XCTAssertEqual(ink.dspark?.targetLayerIds, [5, 23, 41, 48])

        let plan = try QWEnginePlan(
            geometry: QWModelGeometry(descriptor: d),
            modelId: "inkling2_loop",
            geometryDigest: d.geometryDigestV3Hex,
            prefillChunkRows: 512,
            kvStorageMode: "f16"
        )
        XCTAssertEqual(plan.layers.count, 24)
        XCTAssertEqual(d.executionPlan.count, 48)

        let resBytes = try QWAdmissionPlan.modelResidentBytes(descriptor: d)
        XCTAssertGreaterThan(resBytes, 15_000_000_000)
        XCTAssertLessThan(resBytes, 24_000_000_000)
    }

    func testDecodeOnBackground512KBStack() throws {
        let jsonData = makeInkling2JSON()
        let exp = expectation(description: "background stack decode")
        var result: Result<QWModelDescriptor, any Error>?
        DispatchQueue.global(qos: .userInitiated).async {
            result = Result { try QWModelDescriptor.decode(configJSON: jsonData) }
            exp.fulfill()
        }
        wait(for: [exp], timeout: 5.0)
        let decoded = try XCTUnwrap(result).get()
        XCTAssertEqual(decoded.family, .inkling2Loop)
    }
}
