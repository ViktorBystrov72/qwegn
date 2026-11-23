// QWMoEGroupBuildBucketsParityTests.swift — 0-ULP parity tests for fused qwg_moe_group_build_buckets_f32 vs separate kernels.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL

final class QWMoEGroupBuildBucketsParityTests: XCTestCase {
    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func runStep(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try encode(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    private func fillSentinel(_ arena: QWArena, _ tv: QWTensorValue, count: Int, value: UInt32) {
        arena.write(tv, from: [UInt32](repeating: value, count: count))
    }

    private func assertBucketsMatch(
        label: String, arena: QWArena, ids: [Int32], experts: Int, pairs: Int, routes: Int,
        directSlotTok: Bool, writeInv: Bool,
        sepOff: QWTensorValue, sepAct: QWTensorValue, sepInd: QWTensorValue,
        sepPair: QWTensorValue, sepTok: QWTensorValue, sepInv: QWTensorValue,
        fusOff: QWTensorValue, fusAct: QWTensorValue, fusInd: QWTensorValue,
        fusPair: QWTensorValue, fusTok: QWTensorValue, fusInv: QWTensorValue
    ) {
        let sOff = arena.read(sepOff, as: Int32.self, count: experts + 1)
        let fOff = arena.read(fusOff, as: Int32.self, count: experts + 1)
        XCTAssertEqual(fOff, sOff, "\(label): expertOff mismatch")

        let sAct = arena.read(sepAct, as: Int32.self, count: experts)
        let fAct = arena.read(fusAct, as: Int32.self, count: experts)
        XCTAssertEqual(fAct, sAct, "\(label): activeExperts mismatch")

        let sInd = arena.read(sepInd, as: UInt32.self, count: 12)
        let fInd = arena.read(fusInd, as: UInt32.self, count: 12)
        XCTAssertEqual(fInd, sInd, "\(label): activeIndirect mismatch")

        let validCount = Int(fOff[experts])
        let sP = arena.read(sepPair, as: Int32.self, count: pairs)
        let fP = arena.read(fusPair, as: Int32.self, count: pairs)
        let sT = arena.read(sepTok, as: Int32.self, count: pairs)
        let fT = arena.read(fusTok, as: Int32.self, count: pairs)
        let sI = arena.read(sepInv, as: Int32.self, count: pairs)
        let fI = arena.read(fusInv, as: Int32.self, count: pairs)

        var maxBucket = 0
        for e in 0..<experts {
            let lo = Int(fOff[e]), hi = Int(fOff[e + 1])
            maxBucket = max(maxBucket, hi - lo)
            let sBucket = (lo..<hi).map { (sP[$0], sT[$0]) }.sorted { $0.0 < $1.0 }
            let fBucket = (lo..<hi).map { (fP[$0], fT[$0]) }.sorted { $0.0 < $1.0 }
            XCTAssertEqual(fBucket.map(\.0), sBucket.map(\.0), "\(label): bucket \(e) sortedPairIdx set mismatch")
            XCTAssertEqual(fBucket.map(\.1), sBucket.map(\.1), "\(label): bucket \(e) sortedTok set mismatch")
            for (p, tok) in fBucket {
                let raw = ids[Int(p)]
                XCTAssertEqual(Int(UInt32(bitPattern: raw) & 0xFFFF), e)
                if directSlotTok {
                    XCTAssertEqual(tok, p)
                } else {
                    let gIdx = Int((UInt32(bitPattern: raw) >> 16) & 0x7F)
                    let gStride = Int((UInt32(bitPattern: raw) >> 23) & 0x7F)
                    let t = Int(p) / routes
                    XCTAssertEqual(Int(tok), gStride > 0 ? (t * gStride + gIdx) : t)
                }
            }
        }
        if maxBucket <= 1 {
            XCTAssertEqual(fP, sP, "\(label): unique-bucket sortedPairIdx exact mismatch")
            XCTAssertEqual(fT, sT, "\(label): unique-bucket sortedTok exact mismatch")
            XCTAssertEqual(fI, sI, "\(label): unique-bucket invPerm exact mismatch")
        }
        if writeInv {
            for i in 0..<pairs {
                let raw = ids[i]
                if raw >= 0 && Int(UInt32(bitPattern: raw) & 0xFFFF) < experts {
                    let slot = Int(fI[i])
                    XCTAssertTrue(slot >= 0 && slot < validCount)
                    XCTAssertEqual(Int(fP[slot]), i)
                } else {
                    XCTAssertEqual(fI[i], sI[i])
                }
            }
        } else {
            XCTAssertEqual(fI, sI, "\(label): untouched invPerm mismatch")
        }
        if validCount < pairs {
            XCTAssertEqual(Array(fP[validCount..<pairs]), Array(sP[validCount..<pairs]))
            XCTAssertEqual(Array(fT[validCount..<pairs]), Array(sT[validCount..<pairs]))
        }
    }

    func testFusedMatchesSeparateAcrossShapesAndFlags() throws {
        struct Case { let e: Int; let t: Int; let k: Int; let direct: Bool; let inv: Bool; let mode: Int }
        let cases: [Case] = [
            Case(e: 128, t: 1, k: 8, direct: false, inv: false, mode: 0),
            Case(e: 128, t: 4, k: 8, direct: false, inv: true, mode: 1),
            Case(e: 128, t: 16, k: 8, direct: false, inv: true, mode: 1),
            Case(e: 128, t: 64, k: 8, direct: true, inv: true, mode: 1),
            Case(e: 256, t: 4, k: 8, direct: false, inv: false, mode: 2),
            Case(e: 512, t: 8, k: 8, direct: false, inv: true, mode: 1),
            Case(e: 128, t: 4, k: 8, direct: false, inv: true, mode: 3),
        ]
        for (idx, c) in cases.enumerated() {
            let pairs = c.t * c.k
            let arena = try QWArena(hal: Self.hal, weights: 4096, activations: 4 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let idsT = try act.allocate(QWTensorShape([pairs], .i32))
            var ids = [Int32](repeating: 0, count: pairs)
            var rng: UInt32 = UInt32(0x1357 + idx * 97)
            for i in 0..<pairs {
                rng = rng &* 1664525 &+ 1013904223
                if c.mode == 3 {
                    ids[i] = -1
                } else if c.mode == 0 {
                    ids[i] = Int32((i * 13) % c.e)
                } else if c.mode == 1 && (i % 7 == 0) {
                    ids[i] = -1
                } else if c.mode == 2 {
                    let e = Int((rng >> 16) % UInt32(c.e))
                    let gIdx = (i % 3)
                    let gStride = 4
                    ids[i] = Int32(bitPattern: UInt32(e) | (UInt32(gIdx) << 16) | (UInt32(gStride) << 23))
                } else {
                    ids[i] = Int32((rng >> 16) % UInt32(c.e))
                }
            }
            arena.write(idsT, from: ids)

            func allocSet() throws -> (QWTensorValue, QWTensorValue, QWTensorValue, QWTensorValue, QWTensorValue, QWTensorValue) {
                let o = try act.allocate(QWTensorShape([c.e + 1], .i32))
                let a = try act.allocate(QWTensorShape([c.e], .i32))
                let ind = try act.allocate(QWTensorShape([12], .i32))
                let p = try act.allocate(QWTensorShape([pairs], .i32))
                let tk = try act.allocate(QWTensorShape([pairs], .i32))
                let iv = try act.allocate(QWTensorShape([pairs], .i32))
                fillSentinel(arena, o, count: c.e + 1, value: 0xDEAD_BEEF)
                fillSentinel(arena, a, count: c.e, value: 0xCAFE_BABE)
                fillSentinel(arena, ind, count: 12, value: 0xA5A5_5A5A)
                fillSentinel(arena, p, count: pairs, value: 0x1122_3344)
                fillSentinel(arena, tk, count: pairs, value: 0x5566_7788)
                fillSentinel(arena, iv, count: pairs, value: 0x99AA_BBCC)
                return (o, a, ind, p, tk, iv)
            }
            let sep = try allocSet()
            let fus = try allocSet()
            let plan = QWGemmFP.MoEActiveBucketPlan(experts: c.e, intermediate: 896, hidden: 2304, downTileRows: 16)

            try runStep { enc in
                try QWGemm.encodeMoEGroupPairsF32(
                    enc, hal: Self.hal, ids: idsT, expertOff: sep.0,
                    sortedPairIdx: sep.3, sortedTok: sep.4,
                    experts: c.e, totalPairs: pairs, routes: c.k,
                    invPerm: c.inv ? sep.5 : nil, directSlotTok: c.direct)
                try QWGemmFP.encodeMoEBuildActiveBuckets(
                    enc, hal: Self.hal, expertOff: QWFPBufferSlice(sep.0),
                    activeExperts: QWFPBufferSlice(sep.1), indirectArgs: QWFPBufferSlice(sep.2), plan: plan)
            }
            try runStep { enc in
                try QWGemm.encodeMoEGroupBuildBucketsF32(
                    enc, hal: Self.hal, ids: idsT, expertOff: fus.0,
                    activeExperts: fus.1, activeIndirect: fus.2,
                    sortedPairIdx: fus.3, sortedTok: fus.4,
                    experts: c.e, totalPairs: pairs, routes: c.k,
                    plan: plan, invPerm: c.inv ? fus.5 : nil, directSlotTok: c.direct)
            }
            assertBucketsMatch(
                label: "case\(idx)(E=\(c.e),T=\(c.t),K=\(c.k))", arena: arena,
                ids: ids, experts: c.e, pairs: pairs, routes: c.k,
                directSlotTok: c.direct, writeInv: c.inv,
                sepOff: sep.0, sepAct: sep.1, sepInd: sep.2, sepPair: sep.3, sepTok: sep.4, sepInv: sep.5,
                fusOff: fus.0, fusAct: fus.1, fusInd: fus.2, fusPair: fus.3, fusTok: fus.4, fusInv: fus.5)
        }
    }

    func testHighExpertFusedAndLargePairFallback() throws {
        // 1. E=1024 (>512), T=4, K=4 -> pairs=16 <= 512 uses fused kernel.
        let eHigh = 1024, tHigh = 4, kHigh = 4, pairsHigh = tHigh * kHigh
        let arena = try QWArena(hal: Self.hal, weights: 4096, activations: 4 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let idsHighT = try act.allocate(QWTensorShape([pairsHigh], .i32))
        let idsHigh: [Int32] = (0..<pairsHigh).map { Int32(($0 * 131 + 17) % eHigh) }
        arena.write(idsHighT, from: idsHigh)
        let offH = try act.allocate(QWTensorShape([eHigh + 1], .i32))
        let actH = try act.allocate(QWTensorShape([eHigh], .i32))
        let indH = try act.allocate(QWTensorShape([12], .i32))
        let sPH = try act.allocate(QWTensorShape([pairsHigh], .i32))
        let sTH = try act.allocate(QWTensorShape([pairsHigh], .i32))
        let invH = try act.allocate(QWTensorShape([pairsHigh], .i32))
        let planH = QWGemmFP.MoEActiveBucketPlan(experts: eHigh, intermediate: 512, hidden: 1024)
        try runStep { enc in
            try QWGemm.encodeMoEGroupAndActiveBuckets(
                enc, hal: Self.hal, ids: idsHighT, expertOff: offH,
                activeExperts: actH, activeIndirect: indH,
                sortedPairIdx: sPH, sortedTok: sTH,
                experts: eHigh, totalPairs: pairsHigh, routes: kHigh, plan: planH, invPerm: invH)
        }
        let gotOffH = arena.read(offH, as: Int32.self, count: eHigh + 1)
        let gotIndH = arena.read(indH, as: UInt32.self, count: 12)
        let expectedActive = Set(idsHigh.map(Int.init)).sorted().map(Int32.init)
        let gotActH = arena.read(actH, as: Int32.self, count: expectedActive.count)
        XCTAssertEqual(gotActH, expectedActive)
        XCTAssertEqual(Int(gotOffH[eHigh]), pairsHigh)
        XCTAssertEqual(gotIndH[0], UInt32(expectedActive.count))
        XCTAssertEqual(gotIndH[4], UInt32(planH.gateUpGridX))
        XCTAssertEqual(gotIndH[5], UInt32(expectedActive.count))
        XCTAssertEqual(gotIndH[8], UInt32(planH.downGridX))
        XCTAssertEqual(gotIndH[9], UInt32(expectedActive.count))

        // 2. pairs=520 > 512 falls back cleanly to separate kernels.
        let eFB = 128, tFB = 65, kFB = 8, pairsFB = tFB * kFB
        let idsFBT = try act.allocate(QWTensorShape([pairsFB], .i32))
        let idsFB = (0..<pairsFB).map { Int32(($0 * 29) % eFB) }
        arena.write(idsFBT, from: idsFB)
        let offS = try act.allocate(QWTensorShape([eFB + 1], .i32))
        let actS = try act.allocate(QWTensorShape([eFB], .i32))
        let indS = try act.allocate(QWTensorShape([12], .i32))
        let spS = try act.allocate(QWTensorShape([pairsFB], .i32))
        let stS = try act.allocate(QWTensorShape([pairsFB], .i32))
        let offF = try act.allocate(QWTensorShape([eFB + 1], .i32))
        let actF = try act.allocate(QWTensorShape([eFB], .i32))
        let indF = try act.allocate(QWTensorShape([12], .i32))
        let spF = try act.allocate(QWTensorShape([pairsFB], .i32))
        let stF = try act.allocate(QWTensorShape([pairsFB], .i32))
        let planFB = QWGemmFP.MoEActiveBucketPlan(experts: eFB, intermediate: 512, hidden: 1024)
        try runStep { enc in
            try QWGemm.encodeMoEGroupPairsF32(
                enc, hal: Self.hal, ids: idsFBT, expertOff: offS,
                sortedPairIdx: spS, sortedTok: stS, experts: eFB, totalPairs: pairsFB, routes: kFB)
            try QWGemmFP.encodeMoEBuildActiveBuckets(
                enc, hal: Self.hal, expertOff: QWFPBufferSlice(offS),
                activeExperts: QWFPBufferSlice(actS), indirectArgs: QWFPBufferSlice(indS), plan: planFB)
        }
        try runStep { enc in
            try QWGemm.encodeMoEGroupAndActiveBuckets(
                enc, hal: Self.hal, ids: idsFBT, expertOff: offF,
                activeExperts: actF, activeIndirect: indF,
                sortedPairIdx: spF, sortedTok: stF,
                experts: eFB, totalPairs: pairsFB, routes: kFB, plan: planFB)
        }
        XCTAssertEqual(arena.read(offF, as: Int32.self, count: eFB + 1),
                       arena.read(offS, as: Int32.self, count: eFB + 1))
        XCTAssertEqual(arena.read(actF, as: Int32.self, count: eFB),
                       arena.read(actS, as: Int32.self, count: eFB))
        XCTAssertEqual(arena.read(indF, as: UInt32.self, count: 12),
                       arena.read(indS, as: UInt32.self, count: 12))
    }

    func testCompactMXFP8ForwardEndToEndBitExact() throws {
        let e = 16, t = 4, k = 4, h = 64, inter = 32, pairs = t * k
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 4 << 20, kv: 0, gdnStates: 0)
        let wReg = arena.weightsRegion(), act = arena.activationsRegion()

        func makeMXFP8(rows: Int, cols: Int, seedInit: UInt32) throws -> QWGemmFP.MXFP8Bank {
            let groups = cols / 32
            let wT = try wReg.allocate(QWTensorShape([e * rows * groups * 8 * 2], .f16))
            let sT = try wReg.allocate(QWTensorShape([(e * rows * groups + 1) / 2], .f16))
            var rng = seedInit
            let wPool: [UInt8] = [0x00, 0x20, 0x30, 0x38, 0x3C, 0x40, 0xB8, 0xBC]
            var wArr = [UInt32](repeating: 0, count: e * rows * groups * 8)
            var sArr = [UInt8](repeating: 127, count: e * rows * groups)
            for i in 0..<sArr.count {
                rng = rng &* 1664525 &+ 1013904223
                sArr[i] = UInt8(125 + Int((rng >> 20) % 5))
            }
            for i in 0..<wArr.count {
                var word: UInt32 = 0
                for b in 0..<4 {
                    rng = rng &* 1664525 &+ 1013904223
                    word |= UInt32(wPool[Int((rng >> 24) % UInt32(wPool.count))]) << (b * 8)
                }
                wArr[i] = word
            }
            arena.write(wT, from: wArr)
            arena.copyIn(sT, from: Data(sArr))
            return QWGemmFP.MXFP8Bank(weights: QWFPBufferSlice(wT), scales: QWFPBufferSlice(sT), experts: e, rowsPerExpert: rows, k: cols)
        }

        let guBank = try makeMXFP8(rows: 2 * inter, cols: h, seedInit: 0x21)
        let dnBank = try makeMXFP8(rows: h, cols: inter, seedInit: 0x42)
        let xT = try act.allocate(QWTensorShape([t * h], .f16))
        let idsT = try act.allocate(QWTensorShape([pairs], .i32))
        let wtsT = try act.allocate(QWTensorShape([pairs], .f32))
        let outSep = try act.allocate(QWTensorShape([t * h], .bf16))
        let outFus = try act.allocate(QWTensorShape([t * h], .bf16))
        let outCombined = try act.allocate(QWTensorShape([t * h], .bf16))
        let expOff = try act.allocate(QWTensorShape([e + 1], .i32))
        let actExp = try act.allocate(QWTensorShape([e], .i32))
        let actInd = try act.allocate(QWTensorShape([12], .i32))
        let sPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sTok = try act.allocate(QWTensorShape([pairs], .i32))
        let invP = try act.allocate(QWTensorShape([pairs], .i32))
        let guBuf = try act.allocate(QWTensorShape([pairs * 2 * inter], .f16))
        let actBuf = try act.allocate(QWTensorShape([pairs * inter], .f16))
        let dnAcc = try act.allocate(QWTensorShape([pairs * h], .f16))

        arena.write(xT, from: (0..<(t * h)).map { Float16(Float(($0 % 19) - 9) * 0.05) })
        arena.write(idsT, from: (0..<pairs).map { Int32(($0 * 3 + 1) % e) })
        arena.write(wtsT, from: (0..<pairs).map { Float($0 % 4 + 1) * 0.25 })
        let plan = QWGemmFP.MoEActiveBucketPlan(experts: e, intermediate: inter, hidden: h)

        func runCompact(fused: Bool, dst: QWTensorValue) throws {
            try runStep { enc in
                if fused {
                    try QWGemm.encodeMoEGroupBuildBucketsF32(
                        enc, hal: Self.hal, ids: idsT, expertOff: expOff,
                        activeExperts: actExp, activeIndirect: actInd,
                        sortedPairIdx: sPair, sortedTok: sTok,
                        experts: e, totalPairs: pairs, routes: k, plan: plan, invPerm: invP)
                } else {
                    try QWGemm.encodeMoEGroupPairsF32(
                        enc, hal: Self.hal, ids: idsT, expertOff: expOff,
                        sortedPairIdx: sPair, sortedTok: sTok,
                        experts: e, totalPairs: pairs, routes: k, invPerm: invP)
                    try QWGemmFP.encodeMoEBuildActiveBuckets(
                        enc, hal: Self.hal, expertOff: QWFPBufferSlice(expOff),
                        activeExperts: QWFPBufferSlice(actExp),
                        indirectArgs: QWFPBufferSlice(actInd), plan: plan)
                }
                let expOffS = QWFPBufferSlice(expOff), actExpS = QWFPBufferSlice(actExp), actIndS = QWFPBufferSlice(actInd)
                try QWGemm.encodeMoEBankCompactGateUpSwiGLU(
                    enc, hal: Self.hal, x: QWFPBufferSlice(xT), gate: .mxfp8(guBank),
                    expertOff: expOffS, activeExperts: actExpS, activeArgs: actIndS,
                    sortedTok: QWFPBufferSlice(sTok), sortedPairIdx: QWFPBufferSlice(sPair),
                    actOut: QWFPBufferSlice(actBuf), intermediate: inter, topK: k,
                    inDtype: .f16, outDtype: .f16, useIndirectDispatch: false, maxActiveExperts: pairs)
                try QWGemm.encodeMoEBankCompactDown(
                    enc, hal: Self.hal, actIn: QWFPBufferSlice(actBuf), down: .mxfp8(dnBank),
                    expertOff: expOffS, activeExperts: actExpS, activeArgs: actIndS,
                    sortedPairIdx: QWFPBufferSlice(sPair), slotOut: QWFPBufferSlice(dnAcc),
                    topK: k, inDtype: .f16, outDtype: .f16, useIndirectDispatch: false, maxActiveExperts: pairs)
                try QWGemmFP.encodeMoEDownReduceCombineBF16(
                    enc, hal: Self.hal, dnAcc: QWFPBufferSlice(dnAcc), invPerm: QWFPBufferSlice(invP),
                    weights: QWFPBufferSlice(wtsT), shared: nil, gate: nil, out: QWFPBufferSlice(dst),
                    hidden: h, topK: k, tokens: t, inDtype: .f16, outDtype: .bf16)
            }
        }
        try runCompact(fused: false, dst: outSep)
        try runCompact(fused: true, dst: outFus)
        guard let (cb, rawEnc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        var enc = rawEnc
        try QWGemm.encodeMoEBankRoutedCombine(
            &enc, hal: Self.hal, x: QWFPBufferSlice(xT), xDtype: .f16, interDtype: .f16,
            gateUp: .mxfp8(guBank), up: nil, down: .mxfp8(dnBank), idx: idsT, wts: wtsT,
            shared: nil, sharedDtype: .bf16, gate: nil, out: QWFPBufferSlice(outCombined), outDtype: .bf16,
            expOff: expOff, sortedPair: sPair, sortedTok: sTok, invPerm: invP,
            tileE: nil, tileR: nil, tileCount: nil, activeExperts: actExp, activeIndirect: actInd,
            guBuf: guBuf, actBuf: actBuf, dnAccBuf: dnAcc,
            hidden: h, experts: e, topK: k, intermediate: inter, rows: t,
            gateSoftcap: 0, upSoftcap: 0, supportsGroupedNaxPrefill: false,
            concurrentBarriers: false, rotateEncoder: nil)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let sepU16 = arena.read(outSep, as: UInt16.self, count: t * h)
        XCTAssertEqual(arena.read(outFus, as: UInt16.self, count: t * h), sepU16)
        XCTAssertEqual(arena.read(outCombined, as: UInt16.self, count: t * h), sepU16)
    }
}
