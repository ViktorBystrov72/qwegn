// QWMoETailCompactTests.swift — Bit-exact parity for grouped-NAX MoE tail-tile compaction across HQ4_G32, AQ4_G64, and SHVQ3_G64.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWMoETailCompactTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private enum BankKind {
        case hq4g32
        case aq4g64
        case shvq3g64

        var label: String {
            switch self {
            case .hq4g32: return "HQ4_G32"
            case .aq4g64: return "AQ4_G64"
            case .shvq3g64: return "SHVQ3_G64"
            }
        }
    }

    private func makeBankData(kind: BankKind, n: Int, k: Int) throws -> Data {
        var seed: UInt32 = 0x51A7_2029
        func nextByte() -> UInt8 {
            seed = seed &* 1664525 &+ 1013904223
            return UInt8((seed >> 16) & 0xFF)
        }
        switch kind {
        case .hq4g32:
            var data = Data(capacity: n * (k / 32) * 20)
            for _ in 0..<(n * (k / 32)) {
                data.append(contentsOf: [0x1F, 0x2C, 0x00, 0x80])
                for _ in 0..<16 { data.append(nextByte()) }
            }
            return data
        case .aq4g64:
            var data = Data(capacity: n * (k / 64) * 36)
            for _ in 0..<(n * (k / 64)) {
                let vals = (0..<64).map { _ in UInt32(nextByte() & 0x0F) }
                data.append(try QWAQBlock.pack(bits: 4, group: 64, values: vals, scale: 0.03125, bias: -0.25))
            }
            return data
        case .shvq3g64:
            let gBytes = QWSHVQBlock.bytes(bits: 3, group: 64)
            var data = Data(capacity: n * (k / 64) * gBytes)
            let sc = Float16(0.0625).bitPattern
            for _ in 0..<(n * (k / 64)) {
                data.append(UInt8(sc & 0xFF))
                data.append(UInt8(sc >> 8))
                for _ in 2..<gBytes { data.append(nextByte()) }
            }
            return data
        }
    }

    private func makeView(kind: BankKind, blocks: QWTensorValue, n: Int, k: Int) throws -> QWWeightView {
        switch kind {
        case .hq4g32:
            return try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, isRotated: true))
        case .aq4g64:
            return try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, bits: 4, groupSize: 64, isRotated: true))
        case .shvq3g64:
            return try XCTUnwrap(QWWeightView(blocks: blocks, n: n, k: k, bits: 3, groupSize: 64, isRotated: true, isSHVQ: true))
        }
    }

    private func verifyTailCompaction(kind: BankKind) throws {
        // Counts spanning every bm boundary (<=16, <=32, <=64, 64+<=16, 64+<=32, 64+33, 128).
        let counts = [1, 2, 16, 17, 32, 33, 64, 65, 80, 81, 96, 97, 128]
        let experts = counts.count
        let routes = 4
        let totalPairs = counts.reduce(0, +)
        precondition(totalPairs % routes == 0)
        let tokens = totalPairs / routes
        let k = 256, inter = 64, guRows = 2 * inter, dnRows = 128

        let guData = try makeBankData(kind: kind, n: experts * guRows, k: k)
        let dnData = try makeBankData(kind: kind, n: experts * dnRows, k: inter)
        let arena = try QWArena(hal: Self.hal, weights: guData.count + dnData.count + (1 << 20),
                                activations: 16 << 20, kv: 0, gdnStates: 0)
        let wr = arena.weightsRegion(), ar = arena.activationsRegion()
        let guBuf = try wr.allocate(QWTensorShape([guData.count / 2], .f16))
        let dnBuf = try wr.allocate(QWTensorShape([dnData.count / 2], .f16))
        arena.copyIn(guBuf, from: guData)
        arena.copyIn(dnBuf, from: dnData)
        let guView = try makeView(kind: kind, blocks: guBuf, n: experts * guRows, k: k)
        let dnView = try makeView(kind: kind, blocks: dnBuf, n: experts * dnRows, k: inter)

        var idVals = [Int32]()
        idVals.reserveCapacity(totalPairs)
        for (e, c) in counts.enumerated() {
            for _ in 0..<c { idVals.append(Int32(e)) }
        }
        let ids = try ar.allocate(QWTensorShape([totalPairs], .i32))
        arena.write(ids, from: idVals)

        let xe16 = try ar.allocate(QWTensorShape([tokens * k], .f16))
        let xh16 = try ar.allocate(QWTensorShape([totalPairs * inter], .f16))
        let xl16 = try ar.allocate(QWTensorShape([totalPairs * inter], .f16))
        arena.write(xe16, from: (0..<(tokens * k)).map { Float16(Float(($0 % 23) - 11) * 0.03125) })
        arena.write(xh16, from: (0..<(totalPairs * inter)).map { Float16(Float(($0 % 19) - 9) * 0.0625) })
        arena.write(xl16, from: (0..<(totalPairs * inter)).map { Float16(Float(($0 % 13) - 6) * 1e-4) })

        let expOff = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPair = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let sortedTok = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let invPerm = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let plan = QWGemm.MoETilePlan(experts: experts, pairs: totalPairs, gateUpRows: guRows, downRows: dnRows)
        let tileE = try ar.allocate(QWTensorShape([plan.maxTiles], .i32))
        let tileR = try ar.allocate(QWTensorShape([plan.maxTiles], .i32))
        let tileCount = try ar.allocate(QWTensorShape([plan.scratchInts], .i32))
        let guCompact = try ar.allocate(QWTensorShape([totalPairs * guRows], .f16))
        let guRef = try ar.allocate(QWTensorShape([totalPairs * guRows], .f16))
        let dnCompact = try ar.allocate(QWTensorShape([totalPairs * dnRows], .f32))
        let dnRef = try ar.allocate(QWTensorShape([totalPairs * dnRows], .f32))

        func step(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            try encode(enc)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        try step { enc in
            try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: ids, expertOff: expOff,
                                              sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                              experts: experts, totalPairs: totalPairs, routes: routes,
                                              invPerm: invPerm, tileE: tileE, tileR: tileR,
                                              tileCount: tileCount, tilePlan: plan)
        }
        let tc = arena.read(tileCount, as: UInt32.self, count: 1)[0]
        let expectedTiles = counts.map { ($0 + 63) / 64 }.reduce(0, +)
        XCTAssertEqual(Int(tc), expectedTiles, "\(kind.label): compacted tile count must match ceil(c/64)")
        let rawTileR = arena.read(tileR, as: UInt32.self, count: expectedTiles)
        var legacyTileR = [Int32]()
        legacyTileR.reserveCapacity(expectedTiles)
        var tIdx = 0
        for c in counts {
            let tiles = (c + 63) / 64
            if c > 64 {
                XCTAssertEqual(rawTileR[tIdx] >> 24, 3, "\(kind.label) c=\(c) full tile must use bm=64")
                let rem = c & 63
                if rem > 0 {
                    let wantCode: UInt32 = rem <= 16 ? 1 : (rem <= 32 ? 2 : 3)
                    let wantBm: UInt32 = rem <= 16 ? 16 : (rem <= 32 ? 32 : 64)
                    XCTAssertEqual(rawTileR[tIdx + 1] >> 24, wantCode, "\(kind.label) c=\(c) tail code")
                    XCTAssertEqual(rawTileR[tIdx + 1] & 0xFFFFFF, 64 / wantBm, "\(kind.label) c=\(c) tail tidY")
                }
            }
            for r in 0..<tiles { legacyTileR.append(Int32(r)) }
            tIdx += tiles
        }

        try step { enc in
            try QWGemm.encodeMoEGateUpNax(enc, hal: Self.hal, xe: xe16, bank: guView, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount, gu16: guCompact,
                                          rowsPerExpert: guRows, tilePlan: plan, sortedTok: sortedTok)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeMoENaxSplitF32(enc, hal: Self.hal, xHi: xh16, xLo: xl16, bank: dnView,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnCompact,
                                            rowsPerExpert: dnRows, tilePlan: plan)
        }

        // Compare against unencoded legacy tileR (forced 64-row tiles whenever c > 32).
        arena.write(tileR, from: legacyTileR)
        try step { enc in
            try QWGemm.encodeMoEGateUpNax(enc, hal: Self.hal, xe: xe16, bank: guView, expertOff: expOff,
                                          tileE: tileE, tileR: tileR, tileCount: tileCount, gu16: guRef,
                                          rowsPerExpert: guRows, tilePlan: plan, sortedTok: sortedTok)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeMoENaxSplitF32(enc, hal: Self.hal, xHi: xh16, xLo: xl16, bank: dnView,
                                            expertOff: expOff, tileE: tileE, tileR: tileR,
                                            tileCount: tileCount, y32: dnRef,
                                            rowsPerExpert: dnRows, tilePlan: plan)
        }

        XCTAssertEqual(arena.read(guCompact, as: UInt16.self, count: totalPairs * guRows),
                       arena.read(guRef, as: UInt16.self, count: totalPairs * guRows),
                       "\(kind.label) gate/up compacted tail must be 0-ULP bit-exact")
        XCTAssertEqual(arena.read(dnCompact, as: UInt32.self, count: totalPairs * dnRows),
                       arena.read(dnRef, as: UInt32.self, count: totalPairs * dnRows),
                       "\(kind.label) split down compacted tail must be 0-ULP bit-exact")
    }

    func testHeterogeneousTailTileCompactionBitExact() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Grouped NAX MoE requires Apple10+ tensor hardware")
        for kind in [BankKind.hq4g32, .aq4g64, .shvq3g64] {
            try verifyTailCompaction(kind: kind)
        }
    }
}
