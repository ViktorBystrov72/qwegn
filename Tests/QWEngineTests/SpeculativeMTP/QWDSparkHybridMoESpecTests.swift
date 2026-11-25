// QWDSparkHybridMoESpecTests.swift — Verify-window MoEScratchPlan and hybrid suffix edge-case tests.

import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
@testable import QWEngine

final class QWDSparkHybridMoESpecTests: XCTestCase {
    private static func makeSyntheticLatentMoEGeometry() throws -> QWModelGeometry {
        let cfgJSON = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": 4096, "intermediate_size": 4096,
          "num_hidden_layers": 2, "max_position_embeddings": 16384, "rms_norm_eps": 1e-6,
          "num_attention_heads": 32, "num_key_value_heads": 1, "head_dim": 128,
          "qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "v_head_dim": 128,
          "compress_ratios": [4, -2], "default_compress_ratio": 4,
          "q_lora_rank": 128, "kv_lora_rank": 128,
          "compress_qk_nope_head_dim": 64, "compress_qk_rope_head_dim": 64,
          "compress_v_head_dim": 128, "o_groups": 2, "o_lora_rank": 128,
          "sliding_window": 128, "compress_rope_theta": 160000.0,
          "rope_theta": 8000000.0, "rotary_percent": 1.0,
          "first_k_dense_replace": 0, "moe_layer_freq": 1,
          "n_routed_experts": 512, "num_experts_per_tok": 10,
          "moe_intermediate_size": 2048, "moe_latent_size": 2048,
          "n_shared_experts": 1, "tie_word_embeddings": false,
          "dspark_block_size": 7, "dspark_markov_rank": 64,
          "dspark_noise_token_id": 3, "dspark_target_layer_ids": [0, 1],
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfgJSON)
        return QWModelGeometry(descriptor: desc)
    }

    func testMoEScratchPlanVerifyWindowUpTo16UsesCompactQWP() throws {
        let geometry = try Self.makeSyntheticLatentMoEGeometry()
        XCTAssertTrue(geometry.supportsGroupedNAXMoEPrefill)
        XCTAssertEqual(geometry.numExperts, 512)
        XCTAssertEqual(geometry.expertsPerTok, 10)
        XCTAssertEqual(geometry.moeLatentSize, 2048)
        XCTAssertEqual(geometry.moeIntermediate, 2048)

        let plan = QWScratchArenaBuilder.MoEScratchPlan(rows: 16, geometry: geometry, useGroupedNAX: true)
        XCTAssertFalse(plan.isPrefill, "T=9..16 verify windows must stay on CompactQWP + MoE-Spec")
        XCTAssertEqual(plan.r1Rows, 16)
        XCTAssertTrue(plan.needsActiveBuckets)
        XCTAssertEqual(plan.dnAccElems, 16 * 10 * 2048)

        let plan17 = QWScratchArenaBuilder.MoEScratchPlan(rows: 17, geometry: geometry, useGroupedNAX: true)
        XCTAssertTrue(plan17.isPrefill)

        let budget = QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(geometry: geometry, verifyRows: 16, explicitBudget: 36)
        XCTAssertEqual(budget, 36)

        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true",
           let device = MTLCreateSystemDefaultDevice(),
           let buf = device.makeBuffer(length: max(256, plan.allocationBytes), options: .storageModeShared) {
            var cursor = 0
            let carved = try plan.carve({ count, dt in
                let elemB = (dt == .f16 || dt == .bf16) ? 2 : 4
                let bytes = max(256, count * elemB)
                let off = cursor
                cursor = ((cursor + bytes + 255) / 256) * 256
                return QWTensorValue(
                    region: QWMRegion(buffer: buf, base: off, length: bytes),
                    offset: 0, shape: QWTensorShape([count], dt)
                )
            })
            XCTAssertGreaterThanOrEqual(carved.act32.shape.elementCount, 16 * 10 * 2048)
            XCTAssertGreaterThanOrEqual(carved.dnAcc.shape.elementCount, 16 * 10 * 2048)
        }
    }

    func testHybridSuffixExtensionMaxTotalDraftsCapAndGuards() {
        let neural7 = [20, 30, 40, 50, 60, 70, 80]
        let history = [40, 50, 60, 70, 80, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1, 2]
        let recycler = QWTokenRecycler()
        recycler.ingestSequence(history)

        let capped12 = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: history, extraPending: 10,
            previousBeforePending: 2, pending: 10, tokenRecycler: recycler,
            requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .hybrid,
            isDeterministic: true, maxTotalDrafts: 12
        )
        XCTAssertEqual(capped12.count, 12)
        XCTAssertEqual(capped12, neural7 + [91, 92, 93, 94, 95])

        let emptyHist = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: [], extraPending: nil,
            previousBeforePending: nil, pending: 10, tokenRecycler: QWTokenRecycler(),
            requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .hybrid, isDeterministic: true
        )
        XCTAssertEqual(emptyHist, neural7)

        let neuralOnly = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: history, extraPending: 10,
            previousBeforePending: 2, pending: 10, tokenRecycler: recycler,
            requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .mtp, isDeterministic: true
        )
        XCTAssertEqual(neuralOnly, neural7)

        let stochastic = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: history, extraPending: 10,
            previousBeforePending: 2, pending: 10, tokenRecycler: recycler,
            requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .hybrid, isDeterministic: false
        )
        XCTAssertEqual(stochastic, neural7)
    }
}
