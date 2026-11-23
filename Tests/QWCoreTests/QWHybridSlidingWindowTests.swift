// QWHybridSlidingWindowTests.swift — Tests for per-layer SWA config decode, digests, geometry, and admission.
import XCTest
import Foundation
@testable import QWCore

final class QWHybridSlidingWindowTests: XCTestCase {
    private func encodeJSON(_ dict: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: dict)
    }

    private func baseTextConfig(layers: Int = 6, maxPos: Int = 32768) -> [String: Any] {
        [
            "model_type": "qwen3",
            "hidden_size": 512,
            "vocab_size": 1024,
            "num_hidden_layers": layers,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 1024,
            "max_position_embeddings": maxPos,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
        ]
    }

    func testHybridSlidingAttentionLayerTypesDecodeAndConsistency() throws {
        var cfg = baseTextConfig(layers: 6, maxPos: 32768)
        cfg["layer_types"] = [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ]
        cfg["sliding_window"] = 512

        let d = try QWModelDescriptor.decode(configJSON: try encodeJSON(cfg))
        XCTAssertEqual(d.verifyInternalConsistency(), [])
        XCTAssertNil(d.slidingWindow)
        XCTAssertEqual(d.layerSlidingWindows, [512, 512, 512, 512, 512, 0])
        XCTAssertEqual(d.fullAttentionLayers, [0, 1, 2, 3, 4, 5])
        XCTAssertTrue(d.hasAnySlidingWindow)
        for l in 0..<5 { XCTAssertEqual(d.layerSlidingWindow(l), 512) }
        XCTAssertEqual(d.layerSlidingWindow(5), 0)

        let geom = QWModelGeometry(descriptor: d)
        XCTAssertTrue(geom.hasAnySlidingWindow)
        XCTAssertEqual(geom.layerSlidingWindow(0), 512)
        XCTAssertEqual(geom.layerSlidingWindow(5), 0)
        XCTAssertEqual(geom.virtualLayerSlidingWindow(2), 512)
        XCTAssertEqual(geom.virtualLayerSlidingWindow(5), 0)
    }

    func testSlidingLayerTypesAcceptsNonPeriodicAndAllSlidingConfigs() throws {
        var nonPeriodic = baseTextConfig(layers: 4, maxPos: 16384)
        nonPeriodic["layer_types"] = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
        nonPeriodic["sliding_window"] = 1024
        nonPeriodic["full_attention_interval"] = 3
        let d1 = try QWModelDescriptor.decode(configJSON: try encodeJSON(nonPeriodic))
        XCTAssertEqual(d1.verifyInternalConsistency(), [])
        XCTAssertEqual(d1.layerSlidingWindows, [0, 1024, 1024, 0])

        var allSliding = baseTextConfig(layers: 4, maxPos: 8192)
        allSliding["layer_types"] = [String](repeating: "sliding_attention", count: 4)
        allSliding["sliding_window"] = 256
        let d2 = try QWModelDescriptor.decode(configJSON: try encodeJSON(allSliding))
        XCTAssertEqual(d2.verifyInternalConsistency(), [])
        XCTAssertEqual(d2.layerSlidingWindows, [256, 256, 256, 256])

        var oversizedSW = baseTextConfig(layers: 2, maxPos: 512)
        oversizedSW["layer_types"] = ["sliding_attention", "full_attention"]
        oversizedSW["sliding_window"] = 512
        let d3 = try QWModelDescriptor.decode(configJSON: try encodeJSON(oversizedSW))
        XCTAssertEqual(d3.layerSlidingWindows, [0, 0])
        XCTAssertFalse(d3.hasAnySlidingWindow)
    }

    func testSlidingLayerTypesRequiresPositiveSlidingWindow() throws {
        var missing = baseTextConfig(layers: 4, maxPos: 8192)
        missing["layer_types"] = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try encodeJSON(missing))) { error in
            XCTAssertTrue("\(error)".contains("layer_types"))
            XCTAssertTrue("\(error)".contains("sliding_window"))
        }

        missing["sliding_window"] = 0
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try encodeJSON(missing))) { error in
            XCTAssertTrue("\(error)".contains("sliding_window"))
        }
    }

    func testMaxWindowLayersAndSlidingWindowPatternSchedules() throws {
        var maxWindow = baseTextConfig(layers: 6, maxPos: 32768)
        maxWindow["sliding_window"] = 4096
        maxWindow["max_window_layers"] = 2
        let dMax = try QWModelDescriptor.decode(configJSON: try encodeJSON(maxWindow))
        XCTAssertEqual(dMax.verifyInternalConsistency(), [])
        XCTAssertNil(dMax.slidingWindow)
        XCTAssertEqual(dMax.layerSlidingWindows, [0, 0, 4096, 4096, 4096, 4096])

        maxWindow["use_sliding_window"] = false
        let dDisabled = try QWModelDescriptor.decode(configJSON: try encodeJSON(maxWindow))
        XCTAssertNil(dDisabled.slidingWindow)
        XCTAssertNil(dDisabled.layerSlidingWindows)
        XCTAssertFalse(dDisabled.hasAnySlidingWindow)

        var pat = baseTextConfig(layers: 6, maxPos: 32768)
        pat["sliding_window"] = 1024
        pat["sliding_window_pattern"] = 3
        let dPat = try QWModelDescriptor.decode(configJSON: try encodeJSON(pat))
        XCTAssertEqual(dPat.verifyInternalConsistency(), [])
        XCTAssertNil(dPat.slidingWindow)
        XCTAssertEqual(dPat.layerSlidingWindows, [1024, 1024, 0, 1024, 1024, 0])

        pat["sliding_window_pattern"] = 1
        let dPatOne = try QWModelDescriptor.decode(configJSON: try encodeJSON(pat))
        XCTAssertNil(dPatOne.slidingWindow)
        XCTAssertNil(dPatOne.layerSlidingWindows)
        XCTAssertFalse(dPatOne.hasAnySlidingWindow)

        var badPat = pat
        badPat["sliding_window_pattern"] = 0
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try encodeJSON(badPat)))

        var badMwl = maxWindow
        badMwl["max_window_layers"] = -1
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try encodeJSON(badMwl)))
    }

    func testDigestV2GoldenPinningAndV3LayerSlidingWindowsSensitivity() throws {
        let uniform = try QWModelDescriptor.decode(configJSON: try encodeJSON(baseTextConfig(layers: 4, maxPos: 32768)))
        var swCfg = baseTextConfig(layers: 4, maxPos: 32768)
        swCfg["sliding_window"] = 1024
        swCfg["sliding_window_pattern"] = 4
        let patterned = try QWModelDescriptor.decode(configJSON: try encodeJSON(swCfg))

        // Uniform all-full shares identical V2 digest with same config, while V3 diverges when sw_layers set.
        XCTAssertEqual(uniform.geometryDigestV2Hex, patterned.geometryDigestV2Hex)
        XCTAssertNotEqual(uniform.geometryDigestV3Hex, patterned.geometryDigestV3Hex)

        // Sliding attention layer kind maps to "S" in digestFields, altering V2 digest vs "F".
        var kindCfg = baseTextConfig(layers: 4, maxPos: 32768)
        kindCfg["layer_types"] = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        kindCfg["sliding_window"] = 1024
        let kindHybrid = try QWModelDescriptor.decode(configJSON: try encodeJSON(kindCfg))
        XCTAssertNotEqual(patterned.geometryDigestV2Hex, kindHybrid.geometryDigestV2Hex)
    }

    func testAdmissionReducesKVBytesForSlidingAndHybridModels() throws {
        let fullOnly = try QWModelDescriptor.decode(configJSON: try encodeJSON(baseTextConfig(layers: 6, maxPos: 131_072)))
        var hybridCfg = baseTextConfig(layers: 6, maxPos: 131_072)
        hybridCfg["layer_types"] = [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ]
        hybridCfg["sliding_window"] = 512
        let hybrid = try QWModelDescriptor.decode(configJSON: try encodeJSON(hybridCfg))

        let ctx = 65_536
        let fullState = QWAdmissionPlan.perSequenceStateBytes(
            descriptor: fullOnly, kvDtype: .f16, contextTokens: ctx, scratchPerSeq: 0)
        let hybridState = QWAdmissionPlan.perSequenceStateBytes(
            descriptor: hybrid, kvDtype: .f16, contextTokens: ctx, scratchPerSeq: 0, prefillChunk: 512)

        // 5 SWA layers capped at (512 + 512 + 256) = 1280 tokens; 1 full layer at 65536 tokens.
        let kvElemsPerTok = 2 * hybrid.kvHeads * hybrid.headDim
        let wantHybridKV = (5 * 1280 + 65_536) * kvElemsPerTok * 2
        XCTAssertEqual(hybridState.kv, wantHybridKV)
        XCTAssertLessThan(hybridState.kv, fullState.kv / 4)

        // Legacy kvBytesPerParam overload stays in lockstep when all windows are 0.
        let fullBytesState = QWAdmissionPlan.perSequenceStateBytes(
            descriptor: fullOnly, kvBytesPerParam: 2, contextTokens: ctx, scratchPerSeq: 0)
        XCTAssertEqual(fullBytesState.kv, fullState.kv)
    }
}
