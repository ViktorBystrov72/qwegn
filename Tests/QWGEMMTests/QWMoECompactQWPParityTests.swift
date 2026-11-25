// QWMoECompactQWPParityTests.swift — 0-ULP parity tests for compact QWP MoE (T<=16)
// and multi-row HQ4_G32 BF16 decode GEMV.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWMoECompactQWPParityTests: XCTestCase {
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
                let scaleF = Float16(0.03125 * (1.0 + 0.125 * Float((r + g) % 5)))
                let scale = scaleF.bitPattern
                blk[0] = UInt8(scale & 0xFF)
                blk[1] = UInt8(scale >> 8)
                if !kind.isSHVQ && gBytes >= 4 {
                    let minF: Float16 = (kind == .incumbentHQ4G32)
                        ? Float16(-1031.5 * Float(scaleF))
                        : Float16(-0.015625 * Float((r * 3 + g) % 4))
                    let minV = minF.bitPattern
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

    private func verifyCompactMatchesT1(
        kind: QWWeightView.ContainerKind, tokens: Int, experts: Int, routes: Int,
        hidden: Int, intermediate: Int, outDtype: QWDType,
        gateCap: Float, upCap: Float, skipShared: Bool,
        applyFwht: Bool? = nil, seedOffset: UInt32
    ) throws {
        let guBytes = makeBank(kind: kind, rows: experts * 2 * intermediate, k: hidden, seedInit: 0x101 + seedOffset)
        let dnBytes = makeBank(kind: kind, rows: experts * hidden, k: intermediate, seedInit: 0x202 + seedOffset)
        let pairs = tokens * routes
        let actBytes = max(16 << 20, pairs * max(hidden, intermediate) * 20)
        let arena = try QWArena(
            hal: Self.hal, weights: guBytes.count + dnBytes.count + (1 << 20),
            activations: actBytes, kv: 0, gdnStates: 0)
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
        let refOut16: QWTensorValue? = (outDtype == .f16 || outDtype == .bf16)
            ? try act.allocate(QWTensorShape([tokens * hidden], outDtype)) : nil
        let cmpOut16: QWTensorValue? = (outDtype != .f32)
            ? try act.allocate(QWTensorShape([tokens * hidden], outDtype))
            : try act.allocate(QWTensorShape([tokens * hidden], .f32))

        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let actExp = try act.allocate(QWTensorShape([experts], .i32))
        let actInd = try act.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
        let sPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sTok = try act.allocate(QWTensorShape([pairs], .i32))
        let actBuf = try act.allocate(QWTensorShape([pairs * intermediate], .f32))
        let dnAccBuf = try act.allocate(QWTensorShape([pairs * hidden], .f32))

        var ids = [Int32](repeating: 0, count: pairs)
        for t in 0..<tokens {
            for k in 0..<routes {
                ids[t * routes + k] = (t + k) % 5 == 0 ? -1 : Int32((t * 2 + k) % experts)
            }
        }
        arena.write(xT, from: makeFloats(count: tokens * hidden, seedInit: 0x33 + seedOffset))
        arena.write(idsT, from: ids)
        arena.write(wtsT, from: makeFloats(count: pairs, seedInit: 0x44 + seedOffset, scale: 0.5))
        arena.write(shT, from: makeFloats(count: tokens * hidden, seedInit: 0x55 + seedOffset))
        arena.write(gT, from: makeFloats(count: tokens, seedInit: 0x66 + seedOffset, scale: 0.8))

        let elemBytes16 = (outDtype == .f32) ? 4 : 2
        for t in 0..<tokens {
            try runStep { enc in
                try QWGemm.encodeMoEGateUpF32R1(
                    enc, hal: Self.hal, x: xT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    bank: gateUp, ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32),
                    out: refActT1, rowsPerExpert: 2 * intermediate, routes: routes, tokens: 1,
                    fuseSiluFwht: true, reuseExpertsAcrossTokens: false, gateSoftcap: gateCap, upSoftcap: upCap)
                try QWGemm.encodeMoEDownSharedCombineF32(
                    enc, hal: Self.hal, x: refActT1, bank: down, rowsPerRoute: hidden, routes: routes,
                    weights: wtsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .f32),
                    shared: shT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    gate: gT.slice(byteOffset: t * 4, count: 1, dtype: .f32),
                    out: refOut32.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                    ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32), tokens: 1,
                    out16: refOut16?.slice(byteOffset: t * hidden * elemBytes16, count: hidden, dtype: outDtype),
                    reuseExpertsAcrossTokens: false, skipShared: skipShared)
            }
        }
        try runStep { enc in
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: Self.hal, x: xT, gateUp: gateUp, down: down, ids: idsT, weights: wtsT,
                shared: shT, gate: gT, out: cmpOut32, expertOff: expOff, activeExperts: actExp,
                activeIndirect: actInd, sortedPairIdx: sPair, sortedTok: sTok,
                actBuf: actBuf, dnAccBuf: dnAccBuf, hidden: hidden, experts: experts,
                routes: routes, intermediate: intermediate, tokens: tokens, out16: cmpOut16,
                gateSoftcap: gateCap, upSoftcap: upCap, applyFwht: applyFwht, skipShared: skipShared)
        }
        let r32 = arena.read(refOut32, as: UInt32.self, count: tokens * hidden)
        let c32 = arena.read(cmpOut32, as: UInt32.self, count: tokens * hidden)
        XCTAssertEqual(c32, r32, "0-ULP f32 mismatch for \(kind.formatName) T=\(tokens)")
        if let r16t = refOut16, let c16t = cmpOut16 {
            let r16 = arena.read(r16t, as: UInt16.self, count: tokens * hidden)
            let c16 = arena.read(c16t, as: UInt16.self, count: tokens * hidden)
            XCTAssertEqual(c16, r16, "0-ULP \(outDtype) mismatch for \(kind.formatName) T=\(tokens)")
        } else if let c32Extra = cmpOut16, outDtype == .f32 {
            let ce = arena.read(c32Extra, as: UInt32.self, count: tokens * hidden)
            XCTAssertEqual(ce, r32, "0-ULP f32 out16 mismatch for \(kind.formatName) T=\(tokens)")
        }
    }

    func testCompactQWPMatchesT1BitExactHQ4G32() throws {
        try verifyCompactMatchesT1(
            kind: .incumbentHQ4G32, tokens: 4, experts: 6, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .bf16, gateCap: 7.0, upCap: 5.0, skipShared: false, seedOffset: 1)
        try verifyCompactMatchesT1(
            kind: .incumbentHQ4G32, tokens: 8, experts: 8, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .f16, gateCap: 0.0, upCap: 0.0, skipShared: true, seedOffset: 2)
    }

    func testCompactQWPMatchesT1BitExactSHVQAndAQ() throws {
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 2, group: 32), tokens: 4, experts: 6, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .bf16, gateCap: 6.0, upCap: 6.0, skipShared: false, seedOffset: 3)
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 3, group: 64), tokens: 8, experts: 8, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .f32, gateCap: 0.0, upCap: 0.0, skipShared: false, seedOffset: 4)
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 4, group: 32), tokens: 4, experts: 6, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .f16, gateCap: 0.0, upCap: 0.0, skipShared: true, seedOffset: 5)
        try verifyCompactMatchesT1(
            kind: .nativeAffine(bits: 4, group: 32), tokens: 4, experts: 6, routes: 4, hidden: 128, intermediate: 64,
            outDtype: .bf16, gateCap: 0.0, upCap: 0.0, skipShared: false, applyFwht: true, seedOffset: 6)
    }

    func testCompactQWPMatchesT1BitExactInkling2GeometrySHVQ() throws {
        for (idx, bits) in [2, 3, 4].enumerated() {
            try verifyCompactMatchesT1(
                kind: .shvq(bits: bits, group: 64), tokens: 6, experts: 4, routes: 4, hidden: 2048, intermediate: 896,
                outDtype: .bf16, gateCap: 4.0, upCap: 25.0, skipShared: true, applyFwht: true, seedOffset: UInt32(20 + idx)
            )
        }
    }

    func testHQ4DecodeBF16MultiRowMatchesM1BitExact() throws {
        let n = 77, k = 256
        let bank = makeBank(kind: .incumbentHQ4G32, rows: n, k: k, seedInit: 0x7788)
        let arena = try QWArena(hal: Self.hal, weights: bank.count + (1 << 20),
                                activations: (4 << 20), kv: 0, gdnStates: 0)
        let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(bank.count + 1) / 2], .f16))
        arena.copyIn(wBuf, from: bank)
        let weight = try XCTUnwrap(QWWeightView(blocks: wBuf, n: n, k: k, containerKind: .incumbentHQ4G32))
        let act = arena.activationsRegion()
        for rows in [4, 8] {
            let xRaw = makeFloats(count: rows * k, seedInit: UInt32(0x900 + rows)).map { Float16($0) }
            let x16 = try act.allocate(QWTensorShape([rows * k], .f16))
            let yRef = try act.allocate(QWTensorShape([rows * n], .bf16))
            let yBatched = try act.allocate(QWTensorShape([rows * n], .bf16))
            arena.write(x16, from: xRaw)
            for m in 0..<rows {
                try runStep { enc in
                    try QWGemm.encodeHQ4DecodeBF16(
                        enc, hal: Self.hal, weight: weight,
                        x16: x16.slice(byteOffset: m * k * 2, count: k, dtype: .f16),
                        yBF16: yRef.slice(byteOffset: m * n * 2, count: n, dtype: .bf16), rows: 1)
                }
            }
            try runStep { enc in
                try QWGemm.encodeHQ4DecodeBF16(
                    enc, hal: Self.hal, weight: weight, x16: x16, yBF16: yBatched, rows: rows)
            }
            let rBits = arena.read(yRef, as: UInt16.self, count: rows * n)
            let bBits = arena.read(yBatched, as: UInt16.self, count: rows * n)
            XCTAssertEqual(bBits, rBits, "0-ULP HQ4 BF16 multi-row mismatch at M=\(rows)")
        }
    }

    func testCompactQWPDownTileRows32AndSHVQ15Parity() throws {
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 15, group: 32), tokens: 4, experts: 6, routes: 4, hidden: 96, intermediate: 64,
            outDtype: .bf16, gateCap: 6.0, upCap: 5.0, skipShared: false, seedOffset: 30)
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 15, group: 64), tokens: 12, experts: 8, routes: 4, hidden: 192, intermediate: 64,
            outDtype: .f32, gateCap: 0.0, upCap: 0.0, skipShared: true, applyFwht: true, seedOffset: 31)
        try verifyCompactMatchesT1(
            kind: .shvq(bits: 3, group: 64), tokens: 8, experts: 6, routes: 4, hidden: 256, intermediate: 128,
            outDtype: .f16, gateCap: 0.0, upCap: 0.0, skipShared: false, seedOffset: 32)
        try verifyCompactMatchesT1(
            kind: .incumbentHQ4G32, tokens: 8, experts: 6, routes: 4, hidden: 256, intermediate: 64,
            outDtype: .bf16, gateCap: 0.0, upCap: 0.0, skipShared: false, seedOffset: 33)
        try verifyCompactMatchesT1(
            kind: .nativeAffine(bits: 4, group: 64), tokens: 6, experts: 6, routes: 4, hidden: 256, intermediate: 64,
            outDtype: .f16, gateCap: 0.0, upCap: 0.0, skipShared: true, applyFwht: true, seedOffset: 34)
        try verifyCompactMatchesT1(
            kind: .nativeAffine(bits: 8, group: 32), tokens: 6, experts: 6, routes: 4, hidden: 256, intermediate: 64,
            outDtype: .f32, gateCap: 0.0, upCap: 0.0, skipShared: true, applyFwht: true, seedOffset: 35)
    }

    private func makeFastRepeatedBank(kind: QWWeightView.ContainerKind, experts: Int, rowsPerExpert: Int, k: Int, seedInit: UInt32) -> Data {
        let expertBytes = rowsPerExpert * kind.rowBytes(k: k)
        let templateCount = min(experts, 8)
        let template = makeBank(kind: kind, rows: templateCount * rowsPerExpert, k: k, seedInit: seedInit)
        var out = Data(count: experts * expertBytes)
        out.withUnsafeMutableBytes { dstRaw in
            guard let dst = dstRaw.baseAddress else { return }
            template.withUnsafeBytes { srcRaw in
                guard let src = srcRaw.baseAddress else { return }
                for e in 0..<experts {
                    memcpy(dst.advanced(by: e * expertBytes), src.advanced(by: (e % templateCount) * expertBytes), expertBytes)
                }
            }
        }
        return out
    }

    func testCompactQWPT16LatentPrunedIdsMatchesT1AndHonorsActiveCount() throws {
        let tokens = 16, experts = 512, routes = 10, hidden = 2048, intermediate = 2048
        let pairs = tokens * routes
        let guKind: QWWeightView.ContainerKind = .shvq(bits: 15, group: 64)
        let dnKind: QWWeightView.ContainerKind = .shvq(bits: 3, group: 64)
        let guBytes = makeFastRepeatedBank(kind: guKind, experts: experts, rowsPerExpert: 2 * intermediate, k: hidden, seedInit: 0x511)
        let dnBytes = makeFastRepeatedBank(kind: dnKind, experts: experts, rowsPerExpert: intermediate, k: intermediate, seedInit: 0x522)
        let actBytes = max(64 << 20, pairs * (hidden + intermediate) * 8)
        let arena = try QWArena(
            hal: Self.hal, weights: guBytes.count + dnBytes.count + (2 << 20),
            activations: actBytes, kv: 0, gdnStates: 0)
        let wReg = arena.weightsRegion(), act = arena.activationsRegion()
        let guBuf = try wReg.allocate(QWTensorShape([(guBytes.count + 1) / 2], .f16))
        let dnBuf = try wReg.allocate(QWTensorShape([(dnBytes.count + 1) / 2], .f16))
        arena.copyIn(guBuf, from: guBytes)
        arena.copyIn(dnBuf, from: dnBytes)
        let gateUp = try XCTUnwrap(QWWeightView(blocks: guBuf, n: experts * 2 * intermediate, k: hidden, containerKind: guKind))
        let down = try XCTUnwrap(QWWeightView(blocks: dnBuf, n: experts * intermediate, k: intermediate, containerKind: dnKind))

        let xT = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let idsT = try act.allocate(QWTensorShape([pairs], .i32))
        let wtsT = try act.allocate(QWTensorShape([pairs], .f32))
        let shT = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let gT = try act.allocate(QWTensorShape([tokens], .f32))
        let refActT1 = try act.allocate(QWTensorShape([routes * intermediate], .f32))
        let refOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let cmpOut32 = try act.allocate(QWTensorShape([tokens * hidden], .f32))
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let actExp = try act.allocate(QWTensorShape([experts], .i32))
        let actInd = try act.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
        let sPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sTok = try act.allocate(QWTensorShape([pairs], .i32))
        let actBuf = try act.allocate(QWTensorShape([pairs * intermediate], .f32))
        let dnAccBuf = try act.allocate(QWTensorShape([pairs * hidden], .f32))

        // Simulate MoE-Spec budget=36: <=36 unique active experts plus pruned `-1` slots across T=16.
        let activePool: [Int32] = (0..<32).map { Int32(($0 * 13 + 5) % experts) }
        var ids = [Int32](repeating: -1, count: pairs)
        for t in 0..<tokens {
            for k in 0..<routes where (t + k) % 4 != 0 {
                ids[t * routes + k] = activePool[(t * 3 + k) % activePool.count]
            }
        }
        arena.write(xT, from: makeFloats(count: tokens * hidden, seedInit: 0x701))
        arena.write(idsT, from: ids)
        arena.write(wtsT, from: makeFloats(count: pairs, seedInit: 0x702, scale: 0.4))
        arena.write(shT, from: [Float](repeating: 0, count: tokens * hidden))
        arena.write(gT, from: [Float](repeating: 0, count: tokens))

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
                    reuseExpertsAcrossTokens: false, skipShared: true)
            }
        }
        try runStep { enc in
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: Self.hal, x: xT, gateUp: gateUp, down: down, ids: idsT, weights: wtsT,
                shared: shT, gate: gT, out: cmpOut32, expertOff: expOff, activeExperts: actExp,
                activeIndirect: actInd, sortedPairIdx: sPair, sortedTok: sTok,
                actBuf: actBuf, dnAccBuf: dnAccBuf, hidden: hidden, experts: experts,
                routes: routes, intermediate: intermediate, tokens: tokens, skipShared: true)
        }
        let activeCount = arena.read(actInd, as: UInt32.self, count: 1)[0]
        XCTAssertGreaterThan(activeCount, 0)
        XCTAssertLessThanOrEqual(activeCount, 36, "Expected pruned T=16 active experts <= 36, got \(activeCount)")
        let r32 = arena.read(refOut32, as: UInt32.self, count: tokens * hidden)
        let c32 = arena.read(cmpOut32, as: UInt32.self, count: tokens * hidden)
        XCTAssertEqual(c32, r32, "0-ULP mismatch for T=16 E=512 K=10 SHVQ15+SHVQ3 CompactQWP")
    }
}
