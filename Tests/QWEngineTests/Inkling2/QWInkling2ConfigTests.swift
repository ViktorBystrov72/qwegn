// QWInkling2ConfigTests.swift — Tests for 1M q4 admission, DSpark <=16K KV cap, and Inkling-2 alignment.

import Foundation
import XCTest
@testable import QWCore
@testable import QWEngine

final class QWInkling2ConfigTests: XCTestCase {
    private let giB = 1 << 30
    private let miB = 1 << 20

    private func makeInkling2Descriptor(
        maxPos: Int = 1_048_576,
        hidden: Int = 2048,
        vocab: Int = 200_064,
        numLayers: Int = 24,
        kvHeads: Int = 8,
        headDim: Int = 128,
        numExperts: Int = 64,
        expertsPerTok: Int = 4,
        moeIntermediate: Int = 1024,
        withDSpark: Bool = true
    ) -> (desc: QWModelDescriptor, ink2: QWModelDescriptor.Inkling2Config, dspark: QWModelDescriptor.Inkling2DSparkConfig?) {
        let layerTypes = (0..<numLayers).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        let ds: QWModelDescriptor.Inkling2DSparkConfig? = withDSpark ? QWModelDescriptor.Inkling2DSparkConfig(
            hiddenSize: hidden, numLayers: 3, qHeads: 16, kvHeads: kvHeads, headDim: headDim,
            intermediateSize: 2048, maxPositionEmbeddings: maxPos, rmsEps: 1e-6, ropeTheta: 1_000_000.0,
            qBias: false, oBias: false, localLayerIds: [0, 2], slidingWindowSize: 512,
            blockSize: 8, markovRank: 128, markovHeadType: "mlp", targetLayerIds: [5, 11, 17, 23]
        ) : nil
        let ink2 = QWModelDescriptor.Inkling2Config(
            unpaddedVocabSize: vocab, dRel: 16, relExtent: 1024, qBias: false, oBias: false,
            logScalingNFloor: 512, logScalingAlpha: 0.1, useValueNormNoGamma: true,
            qkL2ClipThreshold: 22.6, relL2ClipThreshold: 8.0,
            localLayerIds: Set((0..<numLayers).filter { ($0 + 1) % 6 != 0 }),
            sconvKernelSize: 4, finalLogitSoftcapping: 30.0, swaHeadDim: headDim, swaQHeads: 16, swaKVHeads: kvHeads,
            slidingWindowSize: 512, layersPerAttnResBlock: 1, attnResNormalizeValues: true,
            mlpGateSoftcap: 4.0, mlpUpSoftcap: 25.0, routeScale: 1.0, useGateBias: true,
            loopCoreStart: 6, loopCoreEnd: 18, loopNumPasses: 3, loopGateRank: 64, isVarloop: false,
            defaultQuantMode: .nvfp4, defaultQuantBits: 4, defaultQuantGroupSize: 32,
            perLayerExpertQuant: [:], dspark: ds, vision: nil, padTokenId: 0
        )
        let desc = QWModelDescriptor(
            family: .inkling2Loop, modelType: "inkling2_loop", capabilities: [.moe],
            hidden: hidden, vocab: vocab, denseIntermediate: nil, numLayers: numLayers,
            layerTypes: layerTypes, fullAttentionInterval: 6, rmsEps: 1e-6,
            maxPositionEmbeddings: maxPos, qHeads: 16, kvHeads: kvHeads, headDim: headDim,
            attnOutputGate: true, partialRotaryFactor: 1.0, ropeTheta: 1_000_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: numExperts, expertsPerTok: expertsPerTok, moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: nil, expertGateType: "sigmoid", hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: ds != nil ? 1 : 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 200_006,
            loopSettings: QWModelDescriptor.LoopSettings(start: 6, end: 18, evalCount: 3),
            inkling2Config: ink2
        )
        return (desc, ink2, ds)
    }

    func testDSparkGlobalKVCapBoundsMemoryAcrossLongContext() throws {
        let (_, _, ds) = makeInkling2Descriptor()
        let cfg = try XCTUnwrap(ds)
        XCTAssertEqual(QWInkling2DSpark.maxGlobalKVCapacity, 16_384)
        XCTAssertEqual(QWInkling2DSpark.globalKVCapacity(for: 4096), 4096)
        XCTAssertEqual(QWInkling2DSpark.globalKVCapacity(for: 16_384), 16_384)
        XCTAssertEqual(QWInkling2DSpark.globalKVCapacity(for: 131_072), 16_384)
        XCTAssertEqual(QWInkling2DSpark.globalKVCapacity(for: 1_048_576), 16_384)

        let kv16k = QWInkling2DSpark.estimateKVBytes(config: cfg, capacity: 16_384, maxPrefillChunk: 8192)
        let kv1M = QWInkling2DSpark.estimateKVBytes(config: cfg, capacity: 1_048_576, maxPrefillChunk: 8192)
        XCTAssertEqual(kv1M, kv16k, "DSpark KV must remain capped at 16K global tokens across 16K..1M")
        let rowBytes = 2 * cfg.kvHeads * cfg.headDim * 2
        let expectedLocal = cfg.localLayerIds.count * (cfg.slidingWindowSize + 8192) * rowBytes
        let expectedGlobal = 16_384 * rowBytes
        XCTAssertEqual(kv1M, expectedLocal + expectedGlobal)
        XCTAssertLessThan(expectedGlobal, 70 * miB)
    }

    func testOneMillionQ4AdmissionInside128GiBAndAlignedMaxTokens() throws {
        let (desc, ink2, ds) = makeInkling2Descriptor()
        let uma128GiB = 128 * giB
        let est1MDefault = QWInkling2Runtime.estimateContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 1_048_576, kvMode: .q4,
            mtpDrafts: 4, prefillChunkSize: 8192, lsaConfig: QWLSAConfig(),
            physicalMemoryBytes: uma128GiB
        )
        let est1MLSA = QWInkling2Runtime.estimateContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 1_048_576, kvMode: .q4,
            mtpDrafts: 4, prefillChunkSize: 8192,
            lsaConfig: QWLSAConfig(mode: .auto, budget: 2048, clusterBlock: 512),
            physicalMemoryBytes: uma128GiB
        )
        XCTAssertTrue(est1MLSA.fits, "1M q4 Inkling-2 with LSA must admit inside 128 GiB UMA")
        XCTAssertEqual(est1MLSA.maxFittingContext, 1_048_576)
        XCTAssertLessThan(est1MLSA.requiredBytes, est1MLSA.availableBytes)
        XCTAssertGreaterThan(est1MLSA.scratchAndStateBytes, est1MDefault.scratchAndStateBytes)
        // 8 global visits * (512 + 32) * 2 = 8704 B/tok (~8.5 GiB at 1M tokens)
        XCTAssertEqual(est1MLSA.kvBytes, 1_048_576 * 8 * 2 * (512 + 32))

        // DriverResidency / QWModuleRegistry must route .inkling2Loop through Inkling-2 sizing.
        let routed = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc, packPath: nil, contextTokens: 1_048_576, kvMode: .q4,
            mtpDrafts: 4, prefillChunkSize: 8192, physicalMemoryBytes: uma128GiB
        )
        XCTAssertEqual(routed, est1MDefault)

        // Verify alignedCap matches requested --context-tokens even when headroom expands finalCap.
        let avail = uma128GiB - QWCapacityBudget.osReserveBytes
        let sizing16K = try QWInkling2Runtime.computeAdmissionSizing(
            descriptor: desc, ink2: ink2, baseModelBytes: 12 * giB, dsparkWeightBytes: 1 * giB,
            dsparkCfg: ds, activeMtpDrafts: 4, rawKVMode: "q4", requestedTokens: 16_384,
            prefillChunkSize: 8192, lsaConfig: QWLSAConfig(mode: .auto), availableBytes: avail
        )
        XCTAssertEqual(sizing16K.alignedCap, 16_384)
        XCTAssertEqual(sizing16K.finalCap, 17_408)
        let logMsg = QWCapacityPlan.formatLogLine(
            weightsBytes: sizing16K.weightsEst, fixedScratchBytes: sizing16K.fixedScratchEst,
            kvMode: "q4", perTokenBytes: sizing16K.perTokBytes, maxTokens: sizing16K.alignedCap,
            totalPlannedBytes: sizing16K.totalPlannedBytes, availableBytes: avail
        )
        XCTAssertTrue(logMsg.contains("max_tokens=16384"), "Expected max_tokens=16384 in \(logMsg)")
    }
}
