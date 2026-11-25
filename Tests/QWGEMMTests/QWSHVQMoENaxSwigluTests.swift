// QWSHVQMoENaxSwigluTests.swift — Parity tests for grouped MoE NAX SwiGLU,
// dense shared SwiGLU NAX, sub-2-bit CompactQWP, and nSlices>32 SHVQ GEMV.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWSHVQMoENaxSwigluTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private enum BankSpec {
        case hq4g32
        case aq(bits: Int, group: Int)
        case shvq(bits: Int, group: Int)

        var label: String {
            switch self {
            case .hq4g32: return "HQ4_G32"
            case .aq(let b, let g): return "AQ\(b)_G\(g)"
            case .shvq(let b, let g): return "SHVQ\(b)_G\(g)"
            }
        }
    }

    private func makeBank(_ spec: BankSpec, n: Int, k: Int) -> Data {
        var seed: UInt32 = 0x51A7_9E37
        func nextRand() -> UInt32 {
            seed = seed &* 1664525 &+ 1013904223
            return seed
        }
        switch spec {
        case .hq4g32:
            let groups = k / 32
            let gBytes = 20
            var out = Data(count: n * groups * gBytes)
            out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for r in 0..<n {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let scVal = Float(Float16(0.015 * Float(1 + (r + g) % 5)))
                        let sc = Float16(scVal).bitPattern
                        let bi = Float16(-1031.5 * scVal).bitPattern
                        ptr[base + 0] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        ptr[base + 2] = UInt8(bi & 0xFF)
                        ptr[base + 3] = UInt8(bi >> 8)
                        for i in 4..<20 {
                            ptr[base + i] = UInt8((nextRand() >> 16) & 0xFF)
                        }
                    }
                }
            }
            return out
        case .aq(let bits, let group):
            let groups = k / group
            let gBytes = QWAQBlock.bytes(bits: bits, group: group)
            var out = Data(count: n * groups * gBytes)
            out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                let scaleMul = 0.04 / Float(max(1, (1 << bits) - 1))
                for r in 0..<n {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let scVal = Float(Float16(scaleMul * Float(1 + (r + g) % 4)))
                        let sc = Float16(scVal).bitPattern
                        let bi = Float16(-0.5 * Float((1 << bits) - 1) * scVal).bitPattern
                        ptr[base + 0] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        ptr[base + 2] = UInt8(bi & 0xFF)
                        ptr[base + 3] = UInt8(bi >> 8)
                        for i in 4..<gBytes {
                            ptr[base + i] = UInt8((nextRand() >> 16) & 0xFF)
                        }
                    }
                }
            }
            return out
        case .shvq(let bits, let group):
            let groups = k / group
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var out = Data(count: n * groups * gBytes)
            out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for r in 0..<n {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let sc = Float16(0.03125 * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                        ptr[base + 0] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        for i in 2..<gBytes {
                            ptr[base + i] = UInt8((nextRand() >> 16) & 0xFF)
                        }
                    }
                }
            }
            return out
        }
    }

    private func makeView(_ spec: BankSpec, buf: QWTensorValue, n: Int, k: Int) -> QWWeightView {
        switch spec {
        case .hq4g32:
            return QWWeightView(blocks: buf, n: n, k: k, isRotated: true)!
        case .aq(let bits, let group):
            return QWWeightView(blocks: buf, n: n, k: k, bits: bits,
                                groupSize: group, isRotated: true,
                                isIncumbentContainer: false, isSHVQ: false)!
        case .shvq(let bits, let group):
            return QWWeightView(blocks: buf, n: n, k: k, bits: bits,
                                groupSize: group, isRotated: true, isSHVQ: true)!
        }
    }

    private func makeActivations(count: Int, seedInit: UInt32, scale: Float = 0.25) -> [Float] {
        var seed = seedInit
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return Float(Int32(truncatingIfNeeded: seed >> 16)) * (scale / 32768.0) - (0.5 * scale)
        }
    }

    private func step(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        do {
            try f(enc)
        } catch {
            enc.endEncoding()
            throw error
        }
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    func testMoENaxSwigluParityAcrossFormatsGatherAndMixedTiles() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "NAX SwiGLU requires Metal 4 tensor hardware")
        let specs: [BankSpec] = [
            .hq4g32,
            .aq(bits: 4, group: 32), .aq(bits: 4, group: 64),
            .aq(bits: 8, group: 32), .aq(bits: 8, group: 64),
            .shvq(bits: 1, group: 32), .shvq(bits: 1, group: 64),
            .shvq(bits: 15, group: 32), .shvq(bits: 15, group: 64),
            .shvq(bits: 2, group: 32), .shvq(bits: 2, group: 64),
            .shvq(bits: 3, group: 32), .shvq(bits: 3, group: 64),
            .shvq(bits: 4, group: 32), .shvq(bits: 4, group: 64),
        ]
        // Construct skewed routing counts to hit bm=16, bm=32, and bm=64 in one indirect grid.
        let experts = 8, routes = 2, tokens = 64, inter = 64, k = 128
        let pairs = tokens * routes
        let guRows = 2 * inter
        var ids = [Int32](repeating: 0, count: pairs)
        for p in 0..<pairs {
            if p < 70 { ids[p] = 0 }          // 70 rows -> bm=64 + bm=16 tail
            else if p < 94 { ids[p] = 1 }     // 24 rows -> bm=32
            else if p < 99 { ids[p] = 2 }     // 5 rows  -> bm=16
            else { ids[p] = Int32(3 + (p % 5)) }
        }

        let xVals = makeActivations(count: tokens * k, seedInit: 0x11223344, scale: 0.3)
        let xHalfs = xVals.map { Float16($0) }
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: pairs, gateUpRows: guRows, downRows: k)

        for spec in specs {
            let testHal = spec.label == "SHVQ3_G64" ? QWErnie55MoETests.ernieHal : Self.hal
            let guData = makeBank(spec, n: experts * guRows, k: k)
            let arena = try QWArena(hal: testHal, weights: guData.count + (1 << 20),
                                    activations: 32 << 20, kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let wBuf = try wr.allocate(QWTensorShape([(guData.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: guData)
            let view = makeView(spec, buf: wBuf, n: experts * guRows, k: k)
            XCTAssertTrue(QWGemm.supportsMoEGateUpNaxSwiglu(view), "missing MoE NAX SwiGLU for \(spec.label)")

            let xe16 = try ar.allocate(QWTensorShape([tokens * k], .f16))
            let xeSorted = try ar.allocate(QWTensorShape([pairs * k], .f16))
            let idsT = try ar.allocate(QWTensorShape([pairs], .i32))
            let expOff = try ar.allocate(QWTensorShape([experts + 1], .i32))
            let sortedPair = try ar.allocate(QWTensorShape([pairs], .i32))
            let sortedTok = try ar.allocate(QWTensorShape([pairs], .i32))
            let tileE = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
            let tileR = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
            let tileCount = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
            let gu16 = try ar.allocate(QWTensorShape([pairs * guRows], .f16))
            let refOut = try ar.allocate(QWTensorShape([pairs * inter], .f16))
            let fusedGatherOut = try ar.allocate(QWTensorShape([pairs * inter], .f16))
            let fusedDirectOut = try ar.allocate(QWTensorShape([pairs * inter], .f16))
            arena.write(xe16, from: xHalfs)
            arena.write(idsT, from: ids)

            guard let pair = testHal.begin() else { throw XCTSkip("no command buffer") }
            let cb = pair.0
            let enc = pair.1
            do {
                try QWGemm.encodeMoEGroupPairsF32(enc, hal: testHal, ids: idsT, expertOff: expOff,
                                                  sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                                  experts: experts, totalPairs: pairs, routes: routes,
                                                  tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                  tilePlan: tilePlan)
            } catch {
                enc.endEncoding()
                throw error
            }
            testHal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let sTokHost = arena.read(sortedTok, as: Int32.self, count: pairs)
            var sortedX = [Float16](repeating: Float16(0), count: pairs * k)
            for s in 0..<pairs {
                let tok = Int(sTokHost[s])
                sortedX.replaceSubrange((s * k)..<((s + 1) * k),
                                        with: xHalfs[(tok * k)..<((tok + 1) * k)])
            }
            arena.write(xeSorted, from: sortedX)

            let clamp: QWSwiGLUClampSpec? = spec.label == "SHVQ15_G64"
                ? QWSwiGLUClampSpec(limit: 7.0, postSilu: true) : nil

            guard let p2 = testHal.begin() else { throw XCTSkip("no command buffer") }
            let cb2 = p2.0
            var enc2 = p2.1
            do {
                try testHal.rotateEncoder(cb2, &enc2)
                try QWGemm.encodeMoEGateUpNax(enc2, hal: testHal, xe: xe16, bank: view, expertOff: expOff,
                                              tileE: tileE, tileR: tileR, tileCount: tileCount,
                                              gu16: gu16, rowsPerExpert: guRows, tilePlan: tilePlan,
                                              sortedTok: sortedTok)
                enc2.memoryBarrier(scope: .buffers)
                try QWGemm.encodeSiluMulSplitF16(enc2, hal: testHal, gu: gu16, outHi: refOut, outLo: refOut,
                                                 rows: pairs, intermediate: inter, swigluClamp: clamp)
                try QWGemm.encodeMoEGateUpNaxSwiglu(enc2, hal: testHal, xe: xe16, bank: view, expertOff: expOff,
                                                    tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                    out16: fusedGatherOut, intermediate: inter, tilePlan: tilePlan,
                                                    sortedTok: sortedTok, swigluClamp: clamp)
                try QWGemm.encodeMoEGateUpNaxSwiglu(enc2, hal: testHal, xe: xeSorted, bank: view, expertOff: expOff,
                                                    tileE: tileE, tileR: tileR, tileCount: tileCount,
                                                    out16: fusedDirectOut, intermediate: inter, tilePlan: tilePlan,
                                                    sortedTok: nil, swigluClamp: clamp)
            } catch {
                enc2.endEncoding()
                throw error
            }
            testHal.end(cb2, enc2)
            cb2.waitUntilCompleted()
            XCTAssertNil(cb2.error, "NAX SwiGLU failed for \(spec.label): \(String(describing: cb2.error))")

            let refVals = arena.read(refOut, as: Float16.self, count: pairs * inter).map { Float($0) }
            let gatherBits = arena.read(fusedGatherOut, as: UInt16.self, count: pairs * inter)
            let directBits = arena.read(fusedDirectOut, as: UInt16.self, count: pairs * inter)
            XCTAssertEqual(gatherBits, directBits, "gatherA vs direct NAX SwiGLU 0-ULP mismatch for \(spec.label)")
            let gatherVals = gatherBits.map { Float(Float16(bitPattern: $0)) }
            for idx in 0..<refVals.count {
                XCTAssertEqual(gatherVals[idx], refVals[idx], accuracy: 2e-2,
                               "fused vs split SwiGLU mismatch for \(spec.label) idx=\(idx)")
            }
        }
    }

    func testDenseSharedSwiGLUNaxParity() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Dense SwiGLU NAX requires Metal 4 tensor hardware")
        for spec in [BankSpec.hq4g32, .aq(bits: 8, group: 32), .shvq(bits: 15, group: 64)] {
            let inter = 128, k = 256
            let wData = makeBank(spec, n: 2 * inter, k: k)
            let arena = try QWArena(hal: Self.hal, weights: wData.count + (1 << 20),
                                    activations: 16 << 20, kv: 0, gdnStates: 0)
            let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(wData.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: wData)
            let view = makeView(spec, buf: wBuf, n: 2 * inter, k: k)
            XCTAssertTrue(QWGemm.supportsDenseSwiGLUNax(view))
            let ar = arena.activationsRegion()
            for rows in [16, 32, 65] {
                let x16 = try ar.allocate(QWTensorShape([rows * k], .f16))
                let gu32 = try ar.allocate(QWTensorShape([rows * 2 * inter], .f32))
                let act32 = try ar.allocate(QWTensorShape([rows * inter], .f32))
                let ref16 = try ar.allocate(QWTensorShape([rows * inter], .f16))
                let fused16 = try ar.allocate(QWTensorShape([rows * inter], .f16))
                arena.write(x16, from: makeActivations(count: rows * k, seedInit: UInt32(0x700 + rows)).map { Float16($0) })
                try step { enc in
                    try QWGemm.encodeNaxF32(enc, hal: Self.hal, weight: view, x16: x16, y16: x16, y32: gu32, rows: rows)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeSiluMulF32(enc, hal: Self.hal, gu: gu32, act: act32, rows: rows, intermediate: inter, act16: ref16)
                    try QWGemm.encodeDenseSwiGLUNax(enc, hal: Self.hal, weight: view, x16: x16, out16: fused16, rows: rows, intermediate: inter)
                }
                let want = arena.read(ref16, as: Float16.self, count: rows * inter).map { Float($0) }
                let got = arena.read(fused16, as: Float16.self, count: rows * inter).map { Float($0) }
                for i in 0..<want.count {
                    XCTAssertEqual(got[i], want[i], accuracy: 2e-2, "dense SwiGLU \(spec.label) rows=\(rows) idx=\(i)")
                }
            }
        }
    }

    func testCompactQWPSub2AndLargeKT1GEMVParity() throws {
        // Test CompactQWP (T=4, T=16) for SHVQ1 and SHVQ15 G32/G64 plus T=1 K=2048 fast path for SHVQ15 & SHVQ3.
        for (bits, group) in [(1, 32), (1, 64), (15, 32), (15, 64)] {
            let testHal = bits == 15 ? QWErnie55MoETests.ernieHal : Self.hal
            let tokens = 16, experts = 12, routes = 2, hidden = 128, inter = 64, pairs = tokens * routes
            let guData = makeBank(.shvq(bits: bits, group: group), n: experts * 2 * inter, k: hidden)
            let dnData = makeBank(.shvq(bits: 3, group: group), n: experts * hidden, k: inter)
            let arena = try QWArena(hal: testHal, weights: guData.count + dnData.count + (2 << 20),
                                    activations: 16 << 20, kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let guBuf = try wr.allocate(QWTensorShape([(guData.count + 1) / 2], .f16))
            let dnBuf = try wr.allocate(QWTensorShape([(dnData.count + 1) / 2], .f16))
            arena.copyIn(guBuf, from: guData)
            arena.copyIn(dnBuf, from: dnData)
            let guView = makeView(.shvq(bits: bits, group: group), buf: guBuf, n: experts * 2 * inter, k: hidden)
            let dnView = makeView(.shvq(bits: 3, group: group), buf: dnBuf, n: experts * hidden, k: inter)
            XCTAssertTrue(QWGemm.supportsCompactQWP(gateUp: guView, down: dnView))

            let xT = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let idsT = try ar.allocate(QWTensorShape([pairs], .i32))
            let wtsT = try ar.allocate(QWTensorShape([pairs], .f32))
            let shT = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let gT = try ar.allocate(QWTensorShape([tokens], .f32))
            let refAct = try ar.allocate(QWTensorShape([routes * inter], .f32))
            let refOut = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let cmpOut = try ar.allocate(QWTensorShape([tokens * hidden], .f32))
            let expOff = try ar.allocate(QWTensorShape([experts + 1], .i32))
            let actExp = try ar.allocate(QWTensorShape([experts], .i32))
            let actInd = try ar.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
            let sPair = try ar.allocate(QWTensorShape([pairs], .i32))
            let sTok = try ar.allocate(QWTensorShape([pairs], .i32))
            let actBuf = try ar.allocate(QWTensorShape([pairs * inter], .f32))
            let dnAcc = try ar.allocate(QWTensorShape([pairs * hidden], .f32))

            arena.write(xT, from: makeActivations(count: tokens * hidden, seedInit: UInt32(0x31 + bits)))
            arena.write(idsT, from: (0..<pairs).map { Int32(($0 * 3 + 1) % experts) })
            arena.write(wtsT, from: makeActivations(count: pairs, seedInit: 0x41, scale: 0.4))
            arena.write(shT, from: [Float](repeating: 0, count: tokens * hidden))
            arena.write(gT, from: [Float](repeating: 0, count: tokens))

            let clamp: QWSwiGLUClampSpec? = bits == 1 && group == 64
                ? QWSwiGLUClampSpec(limit: 7.0, postSilu: true) : nil
            func runOnTestHal(_ f: (any MTLComputeCommandEncoder) throws -> Void) throws {
                guard let (cb, enc) = testHal.begin() else { throw XCTSkip("no command buffer") }
                do {
                    try f(enc)
                } catch {
                    enc.endEncoding()
                    throw error
                }
                testHal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
            }
            for t in 0..<tokens {
                try runOnTestHal { enc in
                    try QWGemm.encodeMoEGateUpF32R1(
                        enc, hal: testHal, x: xT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        bank: guView, ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32),
                        out: refAct, rowsPerExpert: 2 * inter, routes: routes, tokens: 1,
                        fuseSiluFwht: true, reuseExpertsAcrossTokens: false, swigluClamp: clamp)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeMoEDownSharedCombineF32(
                        enc, hal: testHal, x: refAct, bank: dnView, rowsPerRoute: hidden, routes: routes,
                        weights: wtsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .f32),
                        shared: shT.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        gate: gT.slice(byteOffset: t * 4, count: 1, dtype: .f32),
                        out: refOut.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        ids: idsT.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32), tokens: 1,
                        reuseExpertsAcrossTokens: false, skipShared: true)
                }
            }
            try runOnTestHal { enc in
                try QWGemm.encodeMoEQWPCompactRoutedCombine(
                    enc, hal: testHal, x: xT, gateUp: guView, down: dnView, ids: idsT, weights: wtsT,
                    shared: shT, gate: gT, out: cmpOut, expertOff: expOff, activeExperts: actExp,
                    activeIndirect: actInd, sortedPairIdx: sPair, sortedTok: sTok,
                    actBuf: actBuf, dnAccBuf: dnAcc, hidden: hidden, experts: experts,
                    routes: routes, intermediate: inter, tokens: tokens, skipShared: true,
                    swigluClamp: clamp)
            }
            XCTAssertEqual(arena.read(cmpOut, as: UInt32.self, count: tokens * hidden),
                           arena.read(refOut, as: UInt32.self, count: tokens * hidden),
                           "0-ULP CompactQWP T=16 mismatch for SHVQ\(bits)_G\(group)")
        }

        // Verify K=2048 (nSlices=64 > 32) coalesced T=1 SHVQ15_G64 + SHVQ3_G64 GEMV fast path vs T=2 CompactQWP reference.
        let experts = 4, routes = 2, hidden = 2048, inter = 2048
        let guData = makeBank(.shvq(bits: 15, group: 64), n: experts * 2 * inter, k: hidden)
        let dnData = makeBank(.shvq(bits: 3, group: 64), n: experts * hidden, k: inter)
        let arena = try QWArena(hal: Self.hal, weights: guData.count + dnData.count + (4 << 20),
                                activations: 16 << 20, kv: 0, gdnStates: 0)
        let wr = arena.weightsRegion(), ar = arena.activationsRegion()
        let guBuf = try wr.allocate(QWTensorShape([(guData.count + 1) / 2], .f16))
        let dnBuf = try wr.allocate(QWTensorShape([(dnData.count + 1) / 2], .f16))
        arena.copyIn(guBuf, from: guData)
        arena.copyIn(dnBuf, from: dnData)
        let guView = makeView(.shvq(bits: 15, group: 64), buf: guBuf, n: experts * 2 * inter, k: hidden)
        let dnView = makeView(.shvq(bits: 3, group: 64), buf: dnBuf, n: experts * hidden, k: inter)

        let x1 = try ar.allocate(QWTensorShape([hidden], .f32))
        let ids1 = try ar.allocate(QWTensorShape([routes], .i32))
        let wts1 = try ar.allocate(QWTensorShape([routes], .f32))
        let sh1 = try ar.allocate(QWTensorShape([hidden], .f32))
        let g1 = try ar.allocate(QWTensorShape([1], .f32))
        let act1 = try ar.allocate(QWTensorShape([routes * inter], .f32))
        let outFast = try ar.allocate(QWTensorShape([hidden], .f32))
        let outCompact = try ar.allocate(QWTensorShape([hidden], .f32))
        let expOff = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let actExp = try ar.allocate(QWTensorShape([experts], .i32))
        let actInd = try ar.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
        let sPair = try ar.allocate(QWTensorShape([routes], .i32))
        let sTok = try ar.allocate(QWTensorShape([routes], .i32))
        let actC = try ar.allocate(QWTensorShape([routes * inter], .f32))
        let dnAccC = try ar.allocate(QWTensorShape([routes * hidden], .f32))

        arena.write(x1, from: makeActivations(count: hidden, seedInit: 0x9988, scale: 0.2))
        arena.write(ids1, from: [Int32(1), Int32(3)])
        arena.write(wts1, from: [Float(0.6), Float(0.4)])
        arena.write(sh1, from: [Float](repeating: 0, count: hidden))
        arena.write(g1, from: [Float(0)])

        try step { enc in
            try QWGemm.encodeMoEGateUpF32R1(
                enc, hal: Self.hal, x: x1, bank: guView, ids: ids1, out: act1,
                rowsPerExpert: 2 * inter, routes: routes, tokens: 1, fuseSiluFwht: true)
            enc.memoryBarrier(scope: .buffers)
            try QWGemm.encodeMoEDownSharedCombineF32(
                enc, hal: Self.hal, x: act1, bank: dnView, rowsPerRoute: hidden, routes: routes,
                weights: wts1, shared: sh1, gate: g1, out: outFast, ids: ids1, tokens: 1, skipShared: true)
            try QWGemm.encodeMoEQWPCompactRoutedCombine(
                enc, hal: Self.hal, x: x1, gateUp: guView, down: dnView, ids: ids1, weights: wts1,
                shared: sh1, gate: g1, out: outCompact, expertOff: expOff, activeExperts: actExp,
                activeIndirect: actInd, sortedPairIdx: sPair, sortedTok: sTok,
                actBuf: actC, dnAccBuf: dnAccC, hidden: hidden, experts: experts,
                routes: routes, intermediate: inter, tokens: 1, skipShared: true)
        }
        let fastVals = arena.read(outFast, as: Float.self, count: hidden)
        let refVals = arena.read(outCompact, as: Float.self, count: hidden)
        for i in 0..<hidden {
            XCTAssertEqual(fastVals[i], refVals[i], accuracy: 1e-4, "K=2048 T=1 fast SHVQ GEMV mismatch at \(i)")
        }

        // Verify K=2048 T=2 (cnt=1,2) and T=8 (cnt>=3 slice_dot_m4_acc) CompactQWP vs T=1.
        let maxVerifyT = 8
        let xM = try ar.allocate(QWTensorShape([maxVerifyT * hidden], .f32))
        let idsM = try ar.allocate(QWTensorShape([maxVerifyT * routes], .i32))
        let wtsM = try ar.allocate(QWTensorShape([maxVerifyT * routes], .f32))
        let shM = try ar.allocate(QWTensorShape([maxVerifyT * hidden], .f32))
        let gM = try ar.allocate(QWTensorShape([maxVerifyT], .f32))
        let sPairM = try ar.allocate(QWTensorShape([maxVerifyT * routes], .i32))
        let sTokM = try ar.allocate(QWTensorShape([maxVerifyT * routes], .i32))
        let actCM = try ar.allocate(QWTensorShape([maxVerifyT * routes * inter], .f32))
        let dnAccCM = try ar.allocate(QWTensorShape([maxVerifyT * routes * hidden], .f32))
        let refOutM = try ar.allocate(QWTensorShape([maxVerifyT * hidden], .f32))
        let cmpOutM = try ar.allocate(QWTensorShape([maxVerifyT * hidden], .f32))
        arena.write(xM, from: makeActivations(count: maxVerifyT * hidden, seedInit: 0x55AA, scale: 0.2))
        arena.write(wtsM, from: makeActivations(count: maxVerifyT * routes, seedInit: 0x77CC, scale: 0.4))
        arena.write(shM, from: [Float](repeating: 0, count: maxVerifyT * hidden))
        arena.write(gM, from: [Float](repeating: 0, count: maxVerifyT))

        for (tRows, pairIds): (Int, [Int32]) in [
            (2, [0, 1, 0, 2]),
            (8, [0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 1, 3, 2, 3, 0, 1]),
        ] {
            arena.write(idsM, from: pairIds)
            for t in 0..<tRows {
                try step { enc in
                    try QWGemm.encodeMoEGateUpF32R1(
                        enc, hal: Self.hal,
                        x: xM.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        bank: guView,
                        ids: idsM.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32),
                        out: act1, rowsPerExpert: 2 * inter, routes: routes, tokens: 1, fuseSiluFwht: true)
                    enc.memoryBarrier(scope: .buffers)
                    try QWGemm.encodeMoEDownSharedCombineF32(
                        enc, hal: Self.hal, x: act1, bank: dnView, rowsPerRoute: hidden, routes: routes,
                        weights: wtsM.slice(byteOffset: t * routes * 4, count: routes, dtype: .f32),
                        shared: shM.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        gate: gM.slice(byteOffset: t * 4, count: 1, dtype: .f32),
                        out: refOutM.slice(byteOffset: t * hidden * 4, count: hidden, dtype: .f32),
                        ids: idsM.slice(byteOffset: t * routes * 4, count: routes, dtype: .i32),
                        tokens: 1, skipShared: true)
                }
            }
            try step { enc in
                try QWGemm.encodeMoEQWPCompactRoutedCombine(
                    enc, hal: Self.hal, x: xM, gateUp: guView, down: dnView, ids: idsM, weights: wtsM,
                    shared: shM, gate: gM, out: cmpOutM, expertOff: expOff, activeExperts: actExp,
                    activeIndirect: actInd, sortedPairIdx: sPairM, sortedTok: sTokM,
                    actBuf: actCM, dnAccBuf: dnAccCM, hidden: hidden, experts: experts,
                    routes: routes, intermediate: inter, tokens: tRows, skipShared: true)
            }
            let rVals = arena.read(refOutM, as: Float.self, count: tRows * hidden)
            let cVals = arena.read(cmpOutM, as: Float.self, count: tRows * hidden)
            for i in 0..<(tRows * hidden) {
                XCTAssertEqual(cVals[i], rVals[i], accuracy: 1e-5, "K=2048 CompactQWP T=\(tRows) mismatch at \(i)")
            }
        }
    }

    func testSHVQ15AndAQ8G32VectorizedNaxParity() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "NAX requires Metal 4 hardware")
        let m = 32, n = 64, k = 256
        let xVals = makeActivations(count: m * k, seedInit: 0x3141_5926, scale: 0.25).map { Float(Float16($0)) }

        for spec in [BankSpec.shvq(bits: 15, group: 64), .aq(bits: 8, group: 32)] {
            let bankData = makeBank(spec, n: n, k: k)
            let arena = try QWArena(hal: Self.hal, weights: bankData.count + (1 << 20),
                                    activations: 4 << 20, kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let wBuf = try wr.allocate(QWTensorShape([(bankData.count + 1) / 2], .f16))
            arena.copyIn(wBuf, from: bankData)
            let wView = makeView(spec, buf: wBuf, n: n, k: k)
            let x32 = try ar.allocate(QWTensorShape([m * k], .f32))
            let y32 = try ar.allocate(QWTensorShape([m * n], .f32))
            let naxX16 = try ar.allocate(QWTensorShape([m * k], .f16))
            let naxY16 = try ar.allocate(QWTensorShape([m * n], .f16))
            arena.write(x32, from: xVals)
            try step { enc in
                try QWGemm.encodeDenseProjF32(
                    enc, hal: Self.hal, weight: wView, x: x32, y: y32,
                    naxX16: naxX16, naxY16: naxY16, maxTileWidth: k, rows: m, minNaxRows: 16)
            }
            let got = arena.read(y32, as: Float.self, count: m * n)

            let group = wView.groupSize
            let groups = k / group
            let gBytes = wView.isSHVQ ? QWSHVQBlock.bytes(bits: wView.bits, group: group) : QWAQBlock.bytes(bits: wView.bits, group: group)
            var ref = [Float](repeating: 0, count: m * n)
            var rowW = [Float](repeating: 0, count: k)
            var blk = [Float]()
            for r in 0..<n {
                for g in 0..<groups {
                    let off = (r * groups + g) * gBytes
                    bankData.subdata(in: off..<(off + gBytes)).withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        if wView.isSHVQ {
                            QWSHVQBlock.unpack(group: raw, bits: wView.bits, group: group, into: &blk)
                        } else {
                            QWAQBlock.unpack(group: raw, bits: wView.bits, group: group, into: &blk)
                        }
                    }
                    rowW.replaceSubrange((g * group)..<((g + 1) * group), with: blk)
                }
                for row in 0..<m {
                    var acc: Float = 0
                    for col in 0..<k { acc += xVals[row * k + col] * rowW[col] }
                    ref[row * n + r] = acc
                }
            }
            var diffSq = 0.0, refSq = 0.0
            for i in 0..<ref.count {
                let d = Double(got[i]) - Double(ref[i])
                diffSq += d * d
                refSq += Double(ref[i]) * Double(ref[i])
            }
            let rel = (diffSq / max(refSq, 1e-30)).squareRoot()
            XCTAssertLessThan(rel, 2e-3, "Vectorized NAX parity failed for \(spec.label): rel=\(rel)")
        }
    }
}
