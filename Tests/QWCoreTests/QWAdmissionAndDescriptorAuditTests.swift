// QWAdmissionAndDescriptorAuditTests.swift — Audit tests for QWAdmission, QWModelDescriptor,
// QWStandardConfigDecoder, QWSamplingParameters, and thread-safe QWSlotPinnedLRUStore.

import Foundation
import XCTest
@testable import QWCore

final class QWAdmissionAndDescriptorAuditTests: XCTestCase {
    private func makeLoopGatedDescriptor(
        hidden: Int = 512,
        vocab: Int = 1024,
        numLayers: Int = 4,
        attnGate: Bool = true,
        evalCount: Int = 1,
        mtpLayers: Int = 0,
        activationKind: QWModelDescriptor.ActivationKind = .silu,
        swigluLimit: Float? = nil,
        routedScalingFactor: Float = 1.0,
        normTopKProb: Bool = false,
        tieWordEmbeddings: Bool = false,
        hasQKNorm: Bool = true,
        usesPlusOneRMSNorm: Bool = false,
        gdnSeparateConv: Bool = false,
        gdnGateLowerBound: Float? = nil,
        yarn: QWModelDescriptor.YarnRopeConfig? = nil,
        quantization: QWQuantizationSpec? = nil,
        numExperts: Int? = nil,
        expertsPerTok: Int? = nil,
        moeIntermediate: Int? = nil,
        moeLatentSize: Int? = nil,
        hybridMLA: QWModelDescriptor.HybridMLAConfig? = nil
    ) -> QWModelDescriptor {
        let types = (0..<numLayers).map { $0 % 2 == 0 ? "linear_attention" : "full_attention" }
        let loop = evalCount > 1
            ? QWModelDescriptor.LoopSettings(start: 1, end: 3, evalCount: evalCount)
            : nil
        return QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5", capabilities: [],
            hidden: hidden, vocab: vocab, denseIntermediate: 1024,
            numLayers: numLayers, layerTypes: types, fullAttentionInterval: 2,
            rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 8, kvHeads: 2, headDim: 64, attnOutputGate: attnGate,
            partialRotaryFactor: 1.0, ropeTheta: 10000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: numExperts, expertsPerTok: expertsPerTok, moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 4, gdnVHeads: 8, gdnHeadDim: 64,
            gdnQKVChannels: (2 * 4 + 8) * 64, gdnZChannels: 8 * 64, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: mtpLayers, mtpUseDedicatedEmbeddings: false,
            eosTokenId: 2, mropeSection: (0, 0, 0),
            hasQKNorm: hasQKNorm, usesPlusOneRMSNorm: usesPlusOneRMSNorm,
            quantization: quantization, hasMropeSection: false,
            tieWordEmbeddings: tieWordEmbeddings, normTopKProb: normTopKProb,
            loopSettings: loop, routedScalingFactor: routedScalingFactor,
            swigluLimit: swigluLimit, gdnGateLowerBound: gdnGateLowerBound,
            gdnSeparateConv: gdnSeparateConv, hybridMLAConfig: hybridMLA,
            yarnRopeConfig: yarn, activationKind: activationKind, moeLatentSize: moeLatentSize
        )
    }

    func testSamplingParametersMergingZeroTopPStaysNonGreedy() {
        let base = QWSamplingParameters(temperature: 0.7, topP: 0.9, topK: 40)
        let mergedZero = base.merging(topP: 0.0)
        XCTAssertFalse(mergedZero.isGreedy)
        XCTAssertEqual(mergedZero.temperature, 0.7)
        XCTAssertEqual(mergedZero.topK, 40)
        XCTAssertEqual(mergedZero.topP, 0.0)

        let mergedTiny = base.merging(topP: 1e-6)
        XCTAssertTrue(mergedTiny.isGreedy)
        XCTAssertEqual(mergedTiny.temperature, 0.0)
        XCTAssertEqual(mergedTiny.topK, 1)
    }

    func testAdmissionOProjGDNZABAndMTPBytes() throws {
        let d0 = makeLoopGatedDescriptor(numLayers: 2, attnGate: true, mtpLayers: 0)
        let dMtp = makeLoopGatedDescriptor(numLayers: 2, attnGate: true, mtpLayers: 1)
        let hq4 = { (elems: Int) in elems / QWHQ4Layout.group * QWHQ4Layout.blockBytes }
        let expectedEmbeds = hq4(d0.vocab * d0.hidden) * 2
        let expectedAttn = hq4(1 * (d0.qProjOut * d0.hidden + 2 * d0.kvHeads * d0.headDim * d0.hidden
            + d0.qHeads * d0.headDim * d0.hidden))
        let expectedGdn = hq4(1 * (d0.gdnQKVChannels * d0.hidden + d0.gdnConvKernel * d0.gdnQKVChannels
            + d0.hidden * d0.gdnVHeads * d0.gdnHeadDim + (d0.gdnZChannels + 2 * d0.gdnVHeads) * d0.hidden))
        let expectedMlp = hq4(2 * 3 * (d0.denseIntermediate! * d0.hidden))
        XCTAssertEqual(try QWAdmissionPlan.modelResidentBytes(descriptor: d0),
                       expectedEmbeds + expectedAttn + expectedGdn + expectedMlp)

        let mtpDelta = hq4(d0.qProjOut * d0.hidden + 2 * d0.kvHeads * d0.headDim * d0.hidden
            + d0.qHeads * d0.headDim * d0.hidden + 3 * d0.denseIntermediate! * d0.hidden + 2 * d0.hidden * d0.hidden)
        XCTAssertEqual(try QWAdmissionPlan.modelResidentBytes(descriptor: dMtp)
                       - QWAdmissionPlan.modelResidentBytes(descriptor: d0), mtpDelta)
    }

    func testAdmissionSHVQGroupSizesAndLoopedStateRingF16() throws {
        let q32 = QWQuantizationSpec(expertGateUp: "SHVQ2_G32", expertDown: "SHVQ3_G128")
        let q64 = QWQuantizationSpec(expertGateUp: "SHVQ2_G64", expertDown: "SHVQ3_G64")
        let d32 = makeLoopGatedDescriptor(
            attnGate: false, quantization: q32,
            numExperts: 32, expertsPerTok: 4, moeIntermediate: 512, moeLatentSize: 512
        )
        let d64 = makeLoopGatedDescriptor(
            attnGate: false, quantization: q64,
            numExperts: 32, expertsPerTok: 4, moeIntermediate: 512, moeLatentSize: 512
        )
        let b32 = try QWAdmissionPlan.modelResidentBytes(descriptor: d32)
        let b64 = try QWAdmissionPlan.modelResidentBytes(descriptor: d64)
        XCTAssertNotEqual(b32, b64)

        let baseLoop = makeLoopGatedDescriptor(numLayers: 4, evalCount: 1)
        let doubledLoop = makeLoopGatedDescriptor(numLayers: 4, evalCount: 2)
        let sBase = QWAdmissionPlan.perSequenceStateBytes(
            descriptor: baseLoop, kvDtype: .f16, contextTokens: 16, scratchPerSeq: 0
        )
        let sLoop = QWAdmissionPlan.perSequenceStateBytes(
            descriptor: doubledLoop, kvDtype: .f16, contextTokens: 16, scratchPerSeq: 0
        )
        XCTAssertEqual(sLoop.kv, (sBase.kv * 3) / 2)
        XCTAssertEqual(sLoop.gdn, (sBase.gdn * 3) / 2)

        let ring = (baseLoop.gdnConvKernel - 1) * baseLoop.gdnQKVChannels
        let rec = baseLoop.gdnVHeads * baseLoop.gdnHeadDim * baseLoop.gdnHeadDim
        XCTAssertEqual(sBase.gdn, 2 * (rec * 4 + ring * 2))
    }

    func testDescriptorConsistencyAndDigestV3Sensitivity() throws {
        let brokenLayers = QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5", capabilities: [],
            hidden: 512, vocab: 1024, denseIntermediate: 1024,
            numLayers: 4, layerTypes: ["linear_attention"], fullAttentionInterval: 1,
            rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 9, kvHeads: 2, headDim: 64, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 10000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 4, expertsPerTok: 8, moeIntermediate: 256,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0,
            gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: [1, 2], pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2
        )
        let bad = brokenLayers.verifyInternalConsistency()
        XCTAssertTrue(bad.contains("layerTypes.count"))
        XCTAssertTrue(bad.contains("qHeads%kvHeads"))
        XCTAssertTrue(bad.contains("expertsPerTok"))
        XCTAssertTrue(bad.contains("gdnDims"))

        let base = makeLoopGatedDescriptor(
            yarn: .init(factor: 2.0, originalMaxPositionEmbeddings: 4096,
                        betaFast: 32.0, betaSlow: 1.0, mscale: 1.0, mscaleAllDim: 0.0)
        )
        let variants: [QWModelDescriptor] = [
            makeLoopGatedDescriptor(activationKind: .situ(beta: 4.0, linearBeta: 25.0), yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(swigluLimit: 7.0, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(routedScalingFactor: 2.5, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(normTopKProb: true, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(tieWordEmbeddings: true, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(hasQKNorm: false, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(usesPlusOneRMSNorm: true, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(gdnSeparateConv: true, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(gdnGateLowerBound: -4.0, yarn: base.yarnRopeConfig),
            makeLoopGatedDescriptor(yarn: .init(factor: 2.0, originalMaxPositionEmbeddings: 4096,
                                              betaFast: 16.0, betaSlow: 2.0, mscale: 1.5, mscaleAllDim: 0.5)),
        ]
        for variant in variants {
            XCTAssertNotEqual(variant.geometryDigestV3Hex, base.geometryDigestV3Hex)
            XCTAssertEqual(variant.geometryDigestV2Hex, base.geometryDigestV2Hex)
        }
    }

    func testDecodeRejectsNegativeDimsAndUndocumentedFallbacks() throws {
        let badStd = #"{"model_type":"qwen3","hidden_size":0,"vocab_size":1000,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,"intermediate_size":256,"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000,"eos_token_id":1}"#
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: Data(badStd.utf8)))

        let missingHCA = """
        {"model_type":"ernie5_5_moe","hidden_size":4096,"intermediate_size":4096,"vocab_size":2048,"num_hidden_layers":1,"compress_ratios":[128],"rms_norm_eps":1e-6,"max_position_embeddings":4096,"num_attention_heads":16,"num_key_value_heads":1,"head_dim":512,"rope_theta":10000,"eos_token_id":2}
        """
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: Data(missingHCA.utf8)))

        let missingVisionPos = """
        {"model_type":"qwen3_5","vision_config":{"depth":2,"hidden_size":256,"num_heads":4,"out_hidden_size":512},"text_config":{"model_type":"qwen3_5_text","hidden_size":512,"intermediate_size":1024,"vocab_size":1024,"num_hidden_layers":1,"layer_types":["full_attention"],"rms_norm_eps":1e-6,"max_position_embeddings":1024,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,"partial_rotary_factor":1.0,"rope_theta":10000,"eos_token_id":2}}
        """
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: Data(missingVisionPos.utf8)))

        let zeroExperts = """
        {"model_type":"maxwell_moe","hidden_size":2048,"intermediate_size":6144,"vocab_size":2048,"num_hidden_layers":1,"num_experts":0,"num_attention_heads":32,"num_key_value_heads":4,"attention_head_size":128,"qk_layernorm":true,"attention_variant":"block_sparse_gqa","block_sparse_gqa_indexer_n_heads":4,"block_sparse_gqa_indexer_head_dim":128,"block_sparse_gqa_block_size":128,"block_sparse_gqa_prefix_window_size":1024,"block_sparse_gqa_local_window_size":1024,"block_sparse_gqa_num_remote_blocks":16,"enable_hyper_connections":true,"rms_norm_eps":1e-6,"rope_theta":20000000,"max_position_embeddings":8192,"eos_token_id":2}
        """
        let decodedDense = try QWModelDescriptor.decode(configJSON: Data(zeroExperts.utf8))
        XCTAssertNil(decodedDense.numExperts)
        XCTAssertFalse(decodedDense.capabilities.contains(.moe))
        XCTAssertFalse(decodedDense.supportsExpertStreaming)

        let negExperts = """
        {"model_type":"maxwell_moe","hidden_size":2048,"intermediate_size":6144,"vocab_size":2048,"num_hidden_layers":1,"num_experts":-1,"num_attention_heads":32,"num_key_value_heads":4,"attention_head_size":128,"qk_layernorm":true,"attention_variant":"block_sparse_gqa","block_sparse_gqa_indexer_n_heads":4,"block_sparse_gqa_indexer_head_dim":128,"block_sparse_gqa_block_size":128,"block_sparse_gqa_prefix_window_size":1024,"block_sparse_gqa_local_window_size":1024,"block_sparse_gqa_num_remote_blocks":16,"enable_hyper_connections":true,"rms_norm_eps":1e-6,"rope_theta":20000000,"max_position_embeddings":8192,"eos_token_id":2}
        """
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: Data(negExperts.utf8)))
    }

    func testSlotPinnedLRUStoreConcurrentThreadSafety() {
        let store = QWSlotPinnedLRUStore<Int, Data>(byteLimit: 4096, entryLimit: 32, maxSlots: 4, rootPrefixTokenCount: 4)
        let evictLock = NSLock()
        var evictedKeys: [Int] = []
        store.onEvict = { key, _ in
            // Re-enter store inside eviction callback to confirm lock is released before notifying.
            _ = store.contains(key)
            evictLock.withLock { evictedKeys.append(key) }
        }
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "test.lru.concurrent", attributes: .concurrent)
        for worker in 0..<8 {
            group.enter()
            queue.async {
                for step in 0..<100 {
                    let key = worker * 100 + step
                    store.store(key: key, value: Data(repeating: UInt8(worker), count: 256),
                                charge: 256, tokenCount: 8 + step, slot: worker % 4,
                                history: Array(0..<Int64(8 + step)))
                    _ = store.value(for: key, touchOnHit: true)
                    _ = store.entries
                }
                group.leave()
            }
        }
        group.wait()
        XCTAssertLessThanOrEqual(store.chargedBytes, store.byteLimit)
        XCTAssertFalse(evictLock.withLock { evictedKeys.isEmpty })
    }
}
