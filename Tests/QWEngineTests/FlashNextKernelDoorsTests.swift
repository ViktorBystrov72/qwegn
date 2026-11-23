import Foundation
import XCTest
import QWCore
@testable import QWEngine

final class FlashNextKernelDoorsTests: XCTestCase {
    func testRequiredDoorsIncludeStridedBatchKernels() {
        let allDoors = Set(DenseModule.requiredDoors)
        XCTAssertTrue(allDoors.contains("qwg_hc_norm_strided"))
        XCTAssertTrue(allDoors.contains("qwg_ple_gate_strided"))
        XCTAssertTrue(allDoors.contains("qwg_hc_norm"))
        XCTAssertTrue(allDoors.contains("qwg_ple_gate"))
        XCTAssertTrue(allDoors.contains("qwg_attn_decode"))
        XCTAssertTrue(allDoors.contains("qwg_router_gemm"))
        XCTAssertEqual(allDoors.count, DenseModule.requiredDoors.count)

        let hyperDesc = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp", capabilities: [.mtp],
            hidden: 2560, vocab: 248320, denseIntermediate: nil, numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4, rmsEps: 1e-6, maxPositionEmbeddings: 262_144,
            qHeads: 24, kvHeads: 2, headDim: 256, attnOutputGate: false,
            partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: 16, indexerKvHeads: 1, indexerHeadDim: 128,
            indexerBudget: 2048, indexerCompressRatio: 16,
            numExperts: 256, expertsPerTok: 8, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248044
        )
        let hyperDoors = Set(DenseModule.requiredDoors(for: hyperDesc))
        XCTAssertTrue(hyperDoors.contains("qwg_hc_norm_strided"))
        XCTAssertTrue(hyperDoors.contains("qwg_ple_gate_strided"))
        XCTAssertTrue(hyperDoors.contains("qwg_hc_norm"))
        XCTAssertTrue(hyperDoors.contains("qwg_ple_gate"))
    }
}
