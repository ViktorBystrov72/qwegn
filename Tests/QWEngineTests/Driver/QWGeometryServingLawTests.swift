// QWGeometryServingLawTests.swift — the geometry's serving law: which
// kernels may serve a model, and which KV mode it may carry. The law is
// ONE predicate (the 27B's shape), so the writer, the compressed pool and
// the decode cannot disagree about it.
import XCTest
@testable import QWEngine
import QWCore
import QWKV

final class QWGeometryServingLawTests: XCTestCase {
    /// A geometry from a standard config, with the given head width and
    /// kv-head count.
    private func geometry(headDim: Int, qHeads: Int = 8, kvHeads: Int = 2) throws
        -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A narrow head is served across f16, q8, and q4 as long as headDim is a multiple of 32 <= 256.
    func testANarrowHeadIsSupportedAcrossKVModes() throws {
        let g = try geometry(headDim: 128)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .f16))
        for mode in [QWKVMode.q8, .q4] {
            XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: mode))
        }
    }

    /// A 256-half head at another kv-head count is also supported across KV modes.
    func testA256HeadAtAnotherKvCountIsSupportedAcrossKVModes() throws {
        let g = try geometry(headDim: 256, qHeads: 16, kvHeads: 8)
        XCTAssertFalse(g.matchesHead256GQA6Kernel)
        XCTAssertFalse(g.matchesKVStride1024Kernel)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .f16))
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .q8))
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: .q4))
    }

    /// The specialized 4×256 GQA-6 shape keeps every mode.
    func testThe27BShapeKeepsEveryMode() throws {
        let g = QWModelGeometry.qwen38_27b
        XCTAssertTrue(g.matchesHead256GQA6Kernel)
        XCTAssertTrue(g.matchesKVStride1024Kernel)
        for mode in [QWKVMode.f16, .q8, .q4] {
            XCTAssertNoThrow(try QWEngineDriver.validateGeometry(g, kvMode: mode))
        }
    }

    /// A head the lane mapping cannot serve is refused: it covers a head in
    /// groups of 8 halves, and the writers cover 256 of them.
    func testAnUnservableHeadWidthIsRefused() throws {
        let narrow = try geometry(headDim: 128)
        XCTAssertNoThrow(try QWEngineDriver.validateGeometry(narrow, kvMode: .f16))
        // head_dim 100 is refused by the loader before it can reach here.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try JSONSerialization
            .data(withJSONObject: ["model_type": "llama", "hidden_size": 1024,
                                   "num_hidden_layers": 1, "num_attention_heads": 8,
                                   "num_key_value_heads": 2, "head_dim": 100,
                                   "intermediate_size": 256,
                                   "max_position_embeddings": 128,
                                   "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                                   "eos_token_id": 2, "vocab_size": 64])))
        // Decode fails closed on head_dim > 256, and validateGeometry enforces the same > 256 guard.
        XCTAssertThrowsError(try geometry(headDim: 512, qHeads: 2, kvHeads: 1))
        let wideDescriptor = QWModelDescriptor(
            family: .llamaDense, modelType: "llama", capabilities: [],
            hidden: 1024, vocab: 64, denseIntermediate: 256, numLayers: 1,
            layerTypes: ["full_attention"], fullAttentionInterval: 1, rmsEps: 1e-5,
            maxPositionEmbeddings: 128, qHeads: 2, kvHeads: 1, headDim: 512,
            attnOutputGate: false, partialRotaryFactor: 1.0, ropeTheta: 500000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0,
            gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2
        )
        let wide = QWModelGeometry(descriptor: wideDescriptor)
        XCTAssertThrowsError(try QWEngineDriver.validateGeometry(wide, kvMode: .f16)) { e in
            guard case QWEngineDriverError.unsupportedModel = e else {
                return XCTFail("expected unsupportedModel, got \(e)")
            }
            XCTAssertTrue("\(e)".contains("256"), "the refusal names the limit: \(e)")
        }
    }
}
