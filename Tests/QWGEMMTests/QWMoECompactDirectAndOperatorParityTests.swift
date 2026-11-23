// QWMoECompactDirectAndOperatorParityTests.swift — 0-ULP parity for direct compact QWP MoE
// and QWMoEOperator T=2..8 dispatch across HQ4_G32, AQ4_G32, and SHVQ3_G64.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWMoECompactDirectAndOperatorParityTests: XCTestCase {
    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func runStep(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try encode(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    private func makeBank(kind: QWWeightView.ContainerKind, rows: Int, k: Int, seedInit: UInt32) -> Data {
        let groups = k / kind.groupSize
        let gBytes = kind.rowBytes(k: k) / groups
        var seed = seedInit
        var out = Data()
        out.reserveCapacity(rows * groups * gBytes)
        for r in 0..<rows {
            for g in 0..<groups {
                var blk = [UInt8](repeating: 0, count: gBytes)
                let scale = Float16(0.03125 * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                blk[0] = UInt8(scale & 0xFF)
                blk[1] = UInt8(scale >> 8)
                if !kind.isSHVQ && gBytes >= 4 {
                    let minV = Float16(-0.015625 * Float((r * 3 + g) % 4)).bitPattern
                    blk[2] = UInt8(minV & 0xFF)
                    blk[3] = UInt8(minV >> 8)
                }
                let start = kind.isSHVQ ? 2 : 4
                for b in start..<gBytes {
                    seed = seed &* 1664525 &+ 1013904223
                    blk[b] = UInt8((seed >> 16) & 0xFF)
                }
                out.append(contentsOf: blk)
            }
        }
        return out
    }

    private func makeFloats(count: Int, seedInit: UInt32, scale: Float = 0.25) -> [Float] {
        var seed = seedInit
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return (Float(Int32(bitPattern: seed) >> 16) / 32768.0) * scale
        }
    }

    private func makeMoEScratch(
        act: QWSubAllocator, rows: Int, hidden: Int, experts: Int, routes: Int,
        intermediate: Int, sharedIntermediate: Int, includeBuckets: Bool
    ) throws -> QWMoEOperator.Scratch {
        let pairs = rows * routes
        let expOff = includeBuckets ? try act.allocate(QWTensorShape([experts + 1], .i32)) : nil
        let sPair = includeBuckets ? try act.allocate(QWTensorShape([pairs], .i32)) : nil
        let sTok = includeBuckets ? try act.allocate(QWTensorShape([pairs], .i32)) : nil
        let actExp = includeBuckets ? try act.allocate(QWTensorShape([experts], .i32)) : nil
        let actInd = includeBuckets
            ? try act.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
            : nil
        return QWMoEOperator.Scratch(
            xRot32: try act.allocate(QWTensorShape([rows * hidden], .f32)),
            xPlain32: try act.allocate(QWTensorShape([rows * hidden], .f32)),
            logits: try act.allocate(QWTensorShape([rows * experts], .f32)),
            idx: try act.allocate(QWTensorShape([pairs], .i32)),
            wts: try act.allocate(QWTensorShape([pairs], .f32)),
            gu32: try act.allocate(QWTensorShape([pairs * 2 * intermediate], .f32)),
            act32: try act.allocate(QWTensorShape([pairs * intermediate], .f32)),
            guSh32: sharedIntermediate > 0
                ? try act.allocate(QWTensorShape([rows * 2 * sharedIntermediate], .f32)) : nil,
            actSh32: sharedIntermediate > 0
                ? try act.allocate(QWTensorShape([rows * sharedIntermediate], .f32)) : nil,
            shared: try act.allocate(QWTensorShape([rows * hidden], .f32)),
            gate: try act.allocate(QWTensorShape([rows], .f32)),
            expOff: expOff, sortedPair: sPair, sortedTok: sTok,
            invPerm: nil, tileE: nil, tileR: nil, tileCount: nil,
            gu16: nil, dnHi: nil,
            dnAcc: try act.allocate(QWTensorShape([pairs * hidden], .f32)),
            naxX16: nil, maxTileWidth: max(hidden, sharedIntermediate),
            activeExperts: actExp, activeIndirect: actInd
        )
    }

    private func verifyDirectCompact(
        kind: QWWeightView.ContainerKind, tokens: Int = 6, experts: Int = 32, routes: Int = 4,
        skipOut32: Bool, seedOffset: UInt32
    ) throws {
        let hidden = 128, intermediate = 64
        let pairs = tokens * routes
        let guBytes = makeBank(kind: kind, rows: experts * 2 * intermediate, k: hidden, seedInit: 0x110 + seedOffset)
        let dnBytes = makeBank(kind: kind, rows: experts * hidden, k: intermediate, seedInit: 0x220 + seedOffset)
        let arena = try QWArena(
            hal: Self.hal, weights: guBytes.count + dnBytes.count + (1 << 20),
            activations: 16 << 20, kv: 0, gdnStates: 0)
        let wReg = arena.weightsRegion()
        let act = arena.activationsRegion()
        let guBuf = try wReg.allocate(QWTensorShape([(guBytes.count + 1) / 2], .f16))
        let dnBuf = try wReg.allocate(QWTensorShape([(dnBytes.count + 1) / 2], .f16))
        arena.copyIn(guBuf, from: guBytes)
        arena.copyIn(dnBuf, from: dnBytes)
        let gateUp = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * 2 * intermediate, k: hidden, containerKind: kind))
        let down = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * hidden, k: intermediate, containerKind: kind))

        let xT = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let idsT = try act.allocate(QWTensorShape([pairs], .i32))
        let wtsT = try act.allocate(QWTensorShape([pairs], .f32))
        let shT = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let gT = try act.allocate(QWTensorShape([tokens], .f32))
        let refActT1 = try act.allocate(QWTensorShape([routes * intermediate], .f32))
        let refOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let cmpOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let refOut16 = try act.allocate(QWTensorShape([tokens * hidden], .bf16))
        let cmpOut16 = try act.allocate(QWTensorShape([tokens * hidden], .bf16))
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let actExp = try act.allocate(QWTensorShape([experts], .i32))
        let actInd = try act.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
        let sPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sTok = try act.allocate(QWTensorShape([pairs], .i32))
        let actBuf = try act.allocate(QWTensorShape([pairs * intermediate], .f32))
        let dnAccBuf = try act.allocate(QWTensorShape([pairs * hidden], .f32))

        var ids = [Int32](repeating: 0, count: pairs)
        let activeSpan = min(experts, 11)
        for t in 0..<tokens {
            for k in 0..<routes {
                ids[t * routes + k] = (t + k) % 7 == 0 ? -1 : Int32((t * 3 + k) % max(1, min(experts, tokens * routes > 128 ? experts : activeSpan)))
            }
        }
        arena.write(xT, from: makeFloats(count: tokens * hidden, seedInit: 0x31 + seedOffset))
        arena.write(idsT, from: ids)
        arena.write(wtsT, from: makeFloats(count: pairs, seedInit: 0x41 + seedOffset, scale: 0.5))
        arena.write(shT, from: makeFloats(count: tokens * hidden, seedInit: 0x51 + seedOffset))
        arena.write(gT, from: makeFloats(count: tokens, seedInit: 0x61 + seedOffset, scale: 0.8))

        let sentinel: UInt32 = 0xDEAD_BEEF
        arena.write(cmpOut32, from: [UInt32](repeating: sentinel, count: tokens * hidden))

        for t in 0..<tokens {
            try runStep { enc in
                try QWGemm.encodeMoEGateUpF32R1(
                    enc, hal: Self.hal, x: xT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    bank: gateUp, ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32),
                    out: refActT1, rowsPerExpert: 2 * intermediate, routes: routes, tokens: 1,
                    fuseSiluFwht: true, reuseExpertsAcrossTokens: false)
                try QWGemm.encodeMoEDownSharedCombineF32(
                    enc, hal: Self.hal, x: refActT1, bank: down, rowsPerRoute: hidden, routes: routes,
                    weights: wtsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .f32),
                    shared: shT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    gate: gT.slice(byteOffset: t * 4, count: 1, dtype: .f32),
                    out: refOut32.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32), tokens: 1,
                    out16: refOut16.slice(byteOffset: t * hidden * 2, count: hidden, dtype: .bf16),
                    reuseExpertsAcrossTokens: false, skipShared: false)
            }
        }
        try runStep { enc in
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: Self.hal, x: xT, gateUp: gateUp, down: down, ids: idsT, weights: wtsT,
                shared: shT, gate: gT, out: cmpOut32, expertOff: expOff, activeExperts: actExp,
                activeIndirect: actInd, sortedPairIdx: sPair, sortedTok: sTok,
                actBuf: actBuf, dnAccBuf: dnAccBuf, hidden: hidden, experts: experts,
                routes: routes, intermediate: intermediate, tokens: tokens, out16: cmpOut16,
                applyFwht: true, skipShared: false,
                skipOut32: skipOut32, nofmaShared: true)
        }
        let r16 = arena.read(refOut16, as: UInt16.self, count: tokens * hidden)
        let c16 = arena.read(cmpOut16, as: UInt16.self, count: tokens * hidden)
        XCTAssertEqual(c16, r16, "0-ULP bf16 mismatch \(kind.formatName) skipOut32=\(skipOut32)")
        let c32 = arena.read(cmpOut32, as: UInt32.self, count: tokens * hidden)
        if skipOut32 {
            XCTAssertTrue(c32.allSatisfy { $0 == sentinel }, "skipOut32=true overwrote out32")
        } else {
            let r32 = arena.read(refOut32, as: UInt32.self, count: tokens * hidden)
            XCTAssertEqual(c32, r32, "0-ULP f32 mismatch \(kind.formatName)")
        }
    }

    private func verifyOperatorParity(
        kind: QWWeightView.ContainerKind, tokens: Int, moeSpecBudget: Int,
        sharedInter: Int = 64, vsSequentialT1: Bool,
        skipOut32: Bool = false, concurrentBarriers: Bool = false, seedOffset: UInt32
    ) throws {
        let hidden = 256, experts = 32, routes = 4, intermediate = 64
        let guBytes = makeBank(kind: kind, rows: experts * 2 * intermediate, k: hidden, seedInit: 0x300 + seedOffset)
        let dnBytes = makeBank(kind: kind, rows: experts * hidden, k: intermediate, seedInit: 0x400 + seedOffset)
        let shGuBytes = sharedInter > 0
            ? makeBank(kind: kind, rows: 2 * sharedInter, k: hidden, seedInit: 0x500 + seedOffset) : Data()
        let shDnBytes = sharedInter > 0
            ? makeBank(kind: kind, rows: hidden, k: sharedInter, seedInit: 0x600 + seedOffset) : Data()
        let wNeed = guBytes.count + dnBytes.count + shGuBytes.count + shDnBytes.count + (1 << 20)
        let arena = try QWArena(hal: Self.hal, weights: wNeed, activations: 16 << 20, kv: 0, gdnStates: 0)
        let wReg = arena.weightsRegion()
        let act = arena.activationsRegion()

        func allocBank(_ data: Data, n: Int, k: Int) throws -> QWWeightView {
            let tv = try wReg.allocate(QWTensorShape([(data.count + 1) / 2], .f16))
            arena.copyIn(tv, from: data)
            return try XCTUnwrap(QWWeightView(blocks: tv, n: n, k: k, containerKind: kind))
        }
        let gateUp = try allocBank(guBytes, n: experts * 2 * intermediate, k: hidden)
        let down = try allocBank(dnBytes, n: experts * hidden, k: intermediate)
        let shGateUp = sharedInter > 0 ? try allocBank(shGuBytes, n: 2 * sharedInter, k: hidden) : nil
        let shDown = sharedInter > 0 ? try allocBank(shDnBytes, n: hidden, k: sharedInter) : nil

        let routerTV = try wReg.allocate(QWTensorShape([experts * hidden], .f32))
        let biasTV = try wReg.allocate(QWTensorShape([experts], .f32))
        let shGateScalarTV = sharedInter > 0 ? try wReg.allocate(QWTensorShape([hidden], .f32)) : nil
        arena.write(routerTV, from: makeFloats(count: experts * hidden, seedInit: 0x700 + seedOffset, scale: 0.4))
        arena.write(biasTV, from: makeFloats(count: experts, seedInit: 0x800 + seedOffset, scale: 0.1))
        if let shGateScalarTV {
            arena.write(shGateScalarTV, from: makeFloats(count: hidden, seedInit: 0x900 + seedOffset, scale: 0.3))
        }

        let x16TV = try act.allocate(QWTensorShape([tokens * hidden], .f16))
        let x16Raw = makeFloats(count: tokens * hidden, seedInit: 0xA00 + seedOffset, scale: 0.5)
            .map { QWQuant.f32tof16Bits($0) }
        arena.write(x16TV, from: x16Raw)

        let opWeights = QWMoEOperator.Weights(
            router: routerTV, routerBias: biasTV,
            gateUp: gateUp, down: down,
            sharedGateUp: shGateUp, sharedDown: shDown, sharedGateScalar: shGateScalarTV,
            hidden: hidden, experts: experts, topK: routes,
            intermediate: intermediate, sharedIntermediate: sharedInter,
            normTopK: true, reuseExpertsAcrossTokens: true,
            moeSpecExpertBudget: moeSpecBudget)
        let refWeights = QWMoEOperator.Weights(
            router: routerTV, routerBias: biasTV,
            gateUp: gateUp, down: down,
            sharedGateUp: shGateUp, sharedDown: shDown, sharedGateScalar: shGateScalarTV,
            hidden: hidden, experts: experts, topK: routes,
            intermediate: intermediate, sharedIntermediate: sharedInter,
            normTopK: true, reuseExpertsAcrossTokens: false,
            moeSpecExpertBudget: moeSpecBudget)

        let cmpOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let cmpOut16 = try act.allocate(QWTensorShape([tokens * hidden], .f16))
        let sentinel: UInt32 = 0xCAFE_BABE
        if skipOut32 {
            arena.write(cmpOut32, from: [UInt32](repeating: sentinel, count: tokens * hidden))
        }
        let cmpSc = try makeMoEScratch(
            act: act, rows: tokens, hidden: hidden, experts: experts, routes: routes,
            intermediate: intermediate, sharedIntermediate: sharedInter, includeBuckets: true)
        if sharedInter == 0 {
            arena.write(cmpSc.shared, from: [UInt32](repeating: 0, count: tokens * hidden))
            arena.write(cmpSc.gate, from: [UInt32](repeating: 0, count: tokens))
        }

        try runStep { enc in
            var mutEnc = enc
            try QWMoEOperator.encodeResident(
                &mutEnc, hal: Self.hal, weights: opWeights, scratch: cmpSc,
                input: .f16Normed(x16TV), out32: cmpOut32, out16: cmpOut16,
                skipOut32: skipOut32, rows: tokens, concurrentBarriers: concurrentBarriers)
        }

        let refOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let refOut16 = try act.allocate(QWTensorShape([tokens * hidden], .f16))
        if vsSequentialT1 {
            let refT1Weights = QWMoEOperator.Weights(
                router: routerTV, routerBias: biasTV,
                gateUp: gateUp, down: down,
                sharedGateUp: shGateUp, sharedDown: shDown, sharedGateScalar: shGateScalarTV,
                hidden: hidden, experts: experts, topK: routes,
                intermediate: intermediate, sharedIntermediate: sharedInter,
                normTopK: true, reuseExpertsAcrossTokens: true, moeSpecExpertBudget: 0)
            let refSc = try makeMoEScratch(
                act: act, rows: 1, hidden: hidden, experts: experts, routes: routes,
                intermediate: intermediate, sharedIntermediate: sharedInter, includeBuckets: false)
            if sharedInter == 0 {
                arena.write(refSc.shared, from: [UInt32](repeating: 0, count: hidden))
                arena.write(refSc.gate, from: [UInt32](repeating: 0, count: 1))
            }
            for t in 0..<tokens {
                try runStep { enc in
                    var mutEnc = enc
                    try QWMoEOperator.encodeResident(
                        &mutEnc, hal: Self.hal, weights: refT1Weights, scratch: refSc,
                        input: .f16Normed(x16TV.slice(byteOffset: t * hidden * 2, count: hidden, dtype: .f16)),
                        out32: refOut32.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        out16: refOut16.slice(byteOffset: t * hidden * 2, count: hidden, dtype: .f16),
                        skipOut32: false, rows: 1)
                }
            }
        } else {
            let fbSc = try makeMoEScratch(
                act: act, rows: tokens, hidden: hidden, experts: experts, routes: routes,
                intermediate: intermediate, sharedIntermediate: sharedInter, includeBuckets: false)
            if sharedInter == 0 {
                arena.write(fbSc.shared, from: [UInt32](repeating: 0, count: tokens * hidden))
                arena.write(fbSc.gate, from: [UInt32](repeating: 0, count: tokens))
            }
            try runStep { enc in
                var mutEnc = enc
                try QWMoEOperator.encodeResident(
                    &mutEnc, hal: Self.hal, weights: refWeights, scratch: fbSc,
                    input: .f16Normed(x16TV), out32: refOut32, out16: refOut16,
                    skipOut32: false, rows: tokens, concurrentBarriers: concurrentBarriers)
            }
        }
        let label = "\(kind.formatName) T=\(tokens) sh=\(sharedInter) budget=\(moeSpecBudget) seqT1=\(vsSequentialT1)"
        if skipOut32 {
            XCTAssertTrue(arena.read(cmpOut32, as: UInt32.self, count: tokens * hidden).allSatisfy { $0 == sentinel },
                          "skipOut32 overwrote out32 (\(label))")
        } else {
            XCTAssertEqual(arena.read(cmpOut32, as: UInt32.self, count: tokens * hidden),
                           arena.read(refOut32, as: UInt32.self, count: tokens * hidden),
                           "0-ULP operator f32 mismatch (\(label))")
        }
        XCTAssertEqual(arena.read(cmpOut16, as: UInt16.self, count: tokens * hidden),
                       arena.read(refOut16, as: UInt16.self, count: tokens * hidden),
                       "0-ULP operator f16 mismatch (\(label))")
    }

    func testDirectCompactDispatchParityHQ4AQ4SHVQ3() throws {
        let kinds: [QWWeightView.ContainerKind] = [
            .incumbentHQ4G32,
            .nativeAffine(bits: 4, group: 32),
            .shvq(bits: 3, group: 64)
        ]
        for (i, kind) in kinds.enumerated() {
            let base = UInt32(i * 10)
            try verifyDirectCompact(kind: kind, skipOut32: false, seedOffset: base + 1)
            try verifyDirectCompact(kind: kind, skipOut32: true, seedOffset: base + 2)
        }
        try verifyDirectCompact(
            kind: .incumbentHQ4G32, tokens: 16, experts: 140, routes: 9,
            skipOut32: false, seedOffset: 90)
    }

    func testMoEOperatorCompactT1248AndBudget12Parity() throws {
        let kinds: [QWWeightView.ContainerKind] = [
            .incumbentHQ4G32,
            .nativeAffine(bits: 4, group: 32),
            .shvq(bits: 3, group: 64)
        ]
        for (i, kind) in kinds.enumerated() {
            let base = UInt32(i * 30)
            for t in [1, 2, 4, 8] {
                try verifyOperatorParity(
                    kind: kind, tokens: t, moeSpecBudget: 0, sharedInter: 64,
                    vsSequentialT1: kind.isSHVQ, skipOut32: t > 1, seedOffset: base + UInt32(t))
                try verifyOperatorParity(
                    kind: kind, tokens: t, moeSpecBudget: 0, sharedInter: 64,
                    vsSequentialT1: kind.isSHVQ, skipOut32: false, seedOffset: base + UInt32(t) + 10)
                try verifyOperatorParity(
                    kind: kind, tokens: t, moeSpecBudget: 0, sharedInter: 0,
                    vsSequentialT1: true, skipOut32: t > 1, seedOffset: base + UInt32(t) + 20)
            }
            try verifyOperatorParity(
                kind: kind, tokens: 4, moeSpecBudget: 12, sharedInter: 64,
                vsSequentialT1: false, skipOut32: true, seedOffset: base + 99)
            try verifyOperatorParity(
                kind: kind, tokens: 4, moeSpecBudget: 12, sharedInter: 64,
                vsSequentialT1: false, skipOut32: false, seedOffset: base + 98)
        }
        try verifyOperatorParity(
            kind: .shvq(bits: 3, group: 64), tokens: 4, moeSpecBudget: 0, sharedInter: 64,
            vsSequentialT1: true, skipOut32: false, concurrentBarriers: true, seedOffset: 123)
    }
}
