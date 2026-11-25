// QWRopeMRoPETableTests.swift — Unit tests for QWRopeTables.buildMRoPECosSin
// and QWRopeApply custom per-turn 3D MRoPE table binding on Metal.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
@testable import QWRope
import QWQuant
import XCTest

final class QWRopeMRoPETableTests: XCTestCase {
    private static func makeGeometry() -> QWModelGeometry {
        let desc = QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5", capabilities: [.vision],
            hidden: 1024, vocab: 32000, denseIntermediate: 2048, numLayers: 1,
            layerTypes: ["full_attention"], fullAttentionInterval: 1, rmsEps: 1e-6,
            maxPositionEmbeddings: 4096, qHeads: 4, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 10_000_000.0,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil, numExperts: 0,
            expertsPerTok: 0, moeIntermediate: 0, sharedExpertIntermediate: 0,
            expertGateType: nil, hcCount: 1, hcLowrank: 0, gdnQKHeads: 0,
            gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0,
            gdnConvKernel: 0, pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil, mtpLayers: 0,
            mtpUseDedicatedEmbeddings: false, eosTokenId: 2, visDepth: 1, hasMropeSection: true
        )
        return QWModelGeometry(descriptor: desc)
    }

    func testBuildMRoPECosSinTextTokensMatch1DBitExact() throws {
        let g = Self.makeGeometry()
        XCTAssertEqual(g.mropeLayout, .qwenInterleaved)
        let tables = QWRopeTables(rows: 64, yarn: nil, geometry: g)
        let n = 4
        let coords: [Int32] = [
            5, 6, 7, 8,
            5, 6, 7, 8,
            5, 6, 7, 8,
        ]
        let mropeBuf = try tables.buildMRoPECosSin(mropePositions: coords, tokenCount: n, layout: .qwenInterleaved)
        XCTAssertEqual(mropeBuf.count, n * tables.width)
        for r in 0..<n {
            let expected = tables.cosSinRow(5 + r)
            let actual = Array(mropeBuf[(r * tables.width)..<((r + 1) * tables.width)])
            for k in expected.indices {
                XCTAssertEqual(actual[k].bitPattern, expected[k].bitPattern)
            }
        }
    }

    func testBuildMRoPECosSinMatchesQwenInterleavedAndBailingVideoRope() throws {
        let g = Self.makeGeometry()
        let yarn = QWYaRNConfig(factor: 2.0, originalMaxPosition: 4096)
        let tables = QWRopeTables(rows: 128, yarn: yarn, geometry: g)
        let coords: [Int32] = [
            10, 12, // T
            20, 25, // H
            30, 35, // W
        ]
        let qwen = try tables.buildMRoPECosSin(mropePositions: coords, tokenCount: 2, layout: .qwenInterleaved)
        let expected0 = tables.cosSinRowMrope((t: 10, h: 20, w: 30))
        let expected1 = tables.cosSinRowMrope((t: 12, h: 25, w: 35))
        for k in 0..<tables.width {
            XCTAssertEqual(qwen[k], expected0[k], accuracy: 1e-6)
            XCTAssertEqual(qwen[tables.width + k], expected1[k], accuracy: 1e-6)
        }

        let bailing = try tables.buildMRoPECosSin(mropePositions: coords, tokenCount: 2, layout: .bailingVideoRope)
        // Bailing VideoRoPE on 32 pairs: W = 0..<12, H = 12..<24, T = 24..<32.
        let rowW = tables.cosSinRow(30)
        let rowH = tables.cosSinRow(20)
        let rowT = tables.cosSinRow(10)
        XCTAssertEqual(bailing[0], rowW[0])
        XCTAssertEqual(bailing[11], rowW[11])
        XCTAssertEqual(bailing[12], rowH[12])
        XCTAssertEqual(bailing[23], rowH[23])
        XCTAssertEqual(bailing[24], rowT[24])
        XCTAssertEqual(bailing[31], rowT[31])
    }
}
