// QWModelDescriptorV3Tests.swift — Tests for geometryDigestV3 sensitivity,
// speculative drafter role decoding, and live ~/.qwg/models validation.
import XCTest
import Foundation
@testable import QWCore

final class QWModelDescriptorV3Tests: XCTestCase {

    func testGeometryDigestV3Sensitivity() throws {
        let base = QWModelDescriptor.qwen38_27b
        XCTAssertNotEqual(base.geometryDigestV3Hex, base.geometryDigestV2Hex)

        let retypedFamily = QWModelDescriptor(
            family: .senseNovaLoop, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visPosEmbeds: 1024, gdnVectorDecay: true, role: base.role)
        XCTAssertEqual(base.geometryDigestV2Hex, retypedFamily.geometryDigestV2Hex)
        XCTAssertNotEqual(base.geometryDigestV3Hex, retypedFamily.geometryDigestV3Hex)

        let withMLAAndVision = QWModelDescriptor(
            family: .qwen38Dense, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visDepth: 24,
            loopSettings: .init(start: 2, end: 8, evalCount: 2),
            mhcConfig: .init(start: 0, end: 4, streams: 4, iterations: 20),
            kvLoraRank: 512)
        XCTAssertNotEqual(base.geometryDigestV3Hex, withMLAAndVision.geometryDigestV3Hex)
    }

    func testDecodeDrafterAssignsSpeculativeDrafterRole() throws {
        let drafterJSON = """
        {
          "architectures": ["DFlashDraftModel"],
          "model_type": "qwen3",
          "hidden_size": 1024,
          "vocab_size": 151936,
          "num_hidden_layers": 5,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 3072,
          "rms_norm_eps": 1e-6,
          "dflash_config": {"mask_token_id": 1},
          "eos_token_id": 151645
        }
        """.data(using: .utf8)!
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: drafterJSON)) { err in
            guard case QWModelDescriptor.DescriptorError.drafterModel = err else {
                return XCTFail("expected drafterModel error, got \(err)")
            }
        }
        let drafter = try QWModelDescriptor.decodeDrafter(configJSON: drafterJSON)
        XCTAssertEqual(drafter.role, .speculativeDrafter)
        XCTAssertEqual(drafter.family, .qwen3Dense)
        XCTAssertEqual(drafter.visDepth, 0)
        XCTAssertEqual(drafter.maxSpeculativeDrafts, 0)
    }

    func testLiveInstalledConfigsDecodeUniqueFamilyAndRole() throws {
        let inCI = ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != nil
        try XCTSkipIf(inCI, "requires bare-metal ~/.qwg/models")
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".qwg/models")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: modelsDir.path), "~/.qwg/models absent")
        let entries = try FileManager.default.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: [.isDirectoryKey])
        var checked = 0
        for dir in entries where (try? dir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            let assetsCfg = dir.appendingPathComponent("assets/config.json")
            let rootCfg = dir.appendingPathComponent("config.json")
            let cfgURL = FileManager.default.fileExists(atPath: assetsCfg.path) ? assetsCfg : rootCfg
            guard FileManager.default.fileExists(atPath: cfgURL.path) else { continue }
            let data = try Data(contentsOf: cfgURL)
            let name = dir.lastPathComponent.lowercased()
            if name.contains("dflash") || name.contains("dspine") {
                let d = try QWModelDescriptor.decodeDrafter(configJSON: data)
                XCTAssertEqual(d.role, .speculativeDrafter, dir.lastPathComponent)
                XCTAssertEqual(d.verifyInternalConsistency(), [], dir.lastPathComponent)
            } else {
                let d = try QWModelDescriptor.decode(configJSON: data)
                XCTAssertNotEqual(d.role, .speculativeDrafter, dir.lastPathComponent)
                XCTAssertEqual(d.verifyInternalConsistency(), [], dir.lastPathComponent)
            }
            checked += 1
        }
        XCTAssertGreaterThan(checked, 0)
    }
}
