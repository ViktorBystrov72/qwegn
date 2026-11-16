// QWHCANAXAttnTests.swift — Apple10+ NAX MQA HCA attention (`qwg_hca_attn_nax`)
// parity against `qwg_hca_attn_mma` across unsplit prefill, Split-K + combine, and sparse LSA.

import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWHCANAXAttnTests: XCTestCase {
    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeData(count: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
        var s = seed
        return (0..<count).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let v = Float(Int32(truncatingIfNeeded: s >> 33)) / Float(Int32.max)
            return v * scale
        }
    }

    private func runCase(
        rows T: Int, pos0: Int, qHeads: Int, compressRatio: Int, slidingWindow: Int,
        nSplit: Int, hasSink: Bool, useBlockList: Bool, compBudget: Int, maxComp: Int
    ) throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "qwg_hca_attn_nax requires Apple10+ NAX")
        let psoMMA = try Self.hal.pipeline("qwg_hca_attn_mma")
        let psoNAX = try Self.hal.pipeline("qwg_hca_attn_nax")
        let psoCombine = try Self.hal.pipeline("qwg_hca_attn_combine")

        let headDim = 512
        let ringCap = 256
        let qUpStride = qHeads * headDim + 64
        let listStride = max(8, compBudget + 4)
        let arena = try QWArena(hal: Self.hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()

        let qUp = try ar.allocate(QWTensorShape([T * qUpStride], .f16))
        let compKV = try ar.allocate(QWTensorShape([maxComp * headDim], .f16))
        let swaKV = try ar.allocate(QWTensorShape([ringCap * headDim], .f16))
        let bList = try ar.allocate(QWTensorShape([T * listStride], .i32))
        let bCount = try ar.allocate(QWTensorShape([T], .i32))
        let sink = try ar.allocate(QWTensorShape([qHeads], .f32))
        let outMMA = try ar.allocate(QWTensorShape([T * qHeads * headDim], .f16))
        let outNAX = try ar.allocate(QWTensorShape([T * qHeads * headDim], .f16))
        let partO = try ar.allocate(QWTensorShape([T * max(1, nSplit) * qHeads * headDim], .f16))
        let partML = try ar.allocate(QWTensorShape([T * max(1, nSplit) * qHeads * 2], .f32))

        arena.write(qUp, from: makeData(count: T * qUpStride, seed: 0x1111, scale: 0.35).map { Float16($0) })
        arena.write(compKV, from: makeData(count: maxComp * headDim, seed: 0x2222, scale: 0.30).map { Float16($0) })
        arena.write(swaKV, from: makeData(count: ringCap * headDim, seed: 0x3333, scale: 0.30).map { Float16($0) })
        arena.write(sink, from: (0..<qHeads).map { h in h % 3 == 0 ? Float(-1e30) : (Float(h % 5) - 2.0) * 0.6 })

        var counts = [Int32](repeating: Int32(compBudget), count: T)
        var lists = [Int32](repeating: 0, count: T * listStride)
        for t in 0..<T {
            let totalComp = (pos0 + t + 1) / compressRatio
            counts[t] = Int32(min(compBudget, totalComp))
            for k in 0..<Int(counts[t]) { lists[t * listStride + k] = Int32((k * 3 + t) % max(1, totalComp)) }
        }
        arena.write(bCount, from: counts)
        arena.write(bList, from: lists)

        var params = QWHCAAttnOperator.AttnParams(
            T: UInt32(T), pos0: UInt32(pos0), qHeads: UInt32(qHeads), headDim: UInt32(headDim),
            qUpStride: UInt32(qUpStride), compressRatio: UInt32(compressRatio),
            slidingWindow: UInt32(slidingWindow), ringMask: UInt32(ringCap - 1),
            compBudget: UInt32(compBudget), listStride: UInt32(listStride),
            useBlockList: useBlockList ? 1 : 0, hasSink: hasSink ? 1 : 0
        )

        func runKernel(_ pso: any MTLComputePipelineState, hg: Int, threads: Int, dst: QWTensorValue) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            enc.setComputePipelineState(pso)
            qUp.bind(enc, index: 0)
            compKV.bind(enc, index: 1)
            swaKV.bind(enc, index: 2)
            bList.bind(enc, index: 3)
            bCount.bind(enc, index: 4)
            sink.bind(enc, index: 5)
            (nSplit > 1 ? partO : dst).bind(enc, index: 6)
            partML.bind(enc, index: 7)
            enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: T, height: (qHeads + hg - 1) / hg, depth: nSplit),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            if nSplit > 1 {
                enc.memoryBarrier(scope: .buffers)
                var nSplitU = UInt32(nSplit)
                enc.setComputePipelineState(psoCombine)
                partO.bind(enc, index: 0)
                partML.bind(enc, index: 1)
                sink.bind(enc, index: 2)
                dst.bind(enc, index: 3)
                enc.setBytes(&params, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 4)
                enc.setBytes(&nSplitU, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: T * qHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        try runKernel(psoMMA, hg: 16, threads: 128, dst: outMMA)
        try runKernel(psoNAX, hg: 32, threads: 256, dst: outNAX)

        let ref = arena.read(outMMA, as: Float16.self, count: T * qHeads * headDim)
        let got = arena.read(outNAX, as: Float16.self, count: T * qHeads * headDim)
        var maxDiff: Float = 0
        for i in 0..<ref.count {
            maxDiff = max(maxDiff, abs(Float(got[i]) - Float(ref[i])))
        }
        XCTAssertLessThan(maxDiff, 2e-3,
                          "HCA NAX vs MMA diff=\(maxDiff) T=\(T) nSplit=\(nSplit) sparse=\(useBlockList)")
    }

    func testHCAATTNNAXMatchesMMAUnsplitSplitAndSparse() throws {
        // Unsplit prefill (non-multiple-of-32 slots + learned sink).
        try runCase(rows: 5, pos0: 260, qHeads: 32, compressRatio: 128, slidingWindow: 128,
                    nSplit: 1, hasSink: true, useBlockList: false, compBudget: 16, maxComp: 8)
        // Split-K decode + combine (32 heads + sink).
        try runCase(rows: 2, pos0: 600, qHeads: 32, compressRatio: 128, slidingWindow: 128,
                    nSplit: 4, hasSink: true, useBlockList: false, compBudget: 16, maxComp: 16)
        // Sparse LSA blockList + odd 16-head remainder coverage.
        try runCase(rows: 3, pos0: 900, qHeads: 16, compressRatio: 128, slidingWindow: 128,
                    nSplit: 4, hasSink: true, useBlockList: true, compBudget: 3, maxComp: 16)
    }

    private func makeWoABank(inc: Int, bits: Int, group: Int, rows: Int, cols: Int) -> Data {
        if inc == 3 {
            let vals = makeData(count: rows * cols, seed: 0x5555, scale: 0.15).map { Float16($0) }
            return vals.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        if inc == 1 {
            var d = Data(capacity: rows * (cols / 32) * 20)
            var s: UInt32 = 0x1357_2468
            for r in 0..<rows {
                for g in 0..<(cols / 32) {
                    let scVal = 0.03125 * (1.0 + 0.125 * Float((r + g) % 5))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1024.0 * scVal - 0.25).bitPattern
                    d.append(contentsOf: [UInt8(sc & 0xFF), UInt8(sc >> 8), UInt8(bi & 0xFF), UInt8(bi >> 8)])
                    for _ in 0..<4 {
                        s = s &* 1664525 &+ 1013904223
                        d.append(contentsOf: [UInt8(s & 0xFF), UInt8((s >> 8) & 0xFF),
                                              UInt8((s >> 16) & 0xFF), UInt8((s >> 24) & 0xFF)])
                    }
                }
            }
            return d
        }
        if inc == 2 {
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var d = Data(count: rows * (cols / group) * gBytes)
            var s: UInt32 = 0x9E37_79B9
            d.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for r in 0..<rows {
                    for g in 0..<(cols / group) {
                        let base = (r * (cols / group) + g) * gBytes
                        let sc = Float16(0.05 * (1.0 + 0.1 * Float((r + g) % 5))).bitPattern
                        ptr[base] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        for b in 2..<gBytes {
                            s = s &* 1664525 &+ 1013904223
                            ptr[base + b] = UInt8((s >> 16) & 0xFF)
                        }
                    }
                }
            }
            return d
        }
        var d = Data()
        let sc: Float = bits >= 8 ? 0.015625 : 0.03125
        let bi: Float = -Float(1 << (bits - 1)) * sc
        for r in 0..<rows {
            for g in 0..<(cols / group) {
                let vals = (0..<group).map { i in UInt32((i * 5 + r * 7 + g * 11) % (1 << bits)) }
                d.append(try! QWAQBlock.pack(bits: bits, group: group, values: vals, scale: sc, bias: bi))
            }
        }
        return d
    }

    private func runGroupedWoACase(label: String, rows T: Int, useNAX: Bool,
                                   inc: Int, bits: Int, group: Int,
                                   oGroups: Int = 4, oLoraRank: Int = 128, groupIn: Int = 256) throws {
        if useNAX {
            try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "requires Apple10+ NAX")
        }
        let oGateOffset = 64
        let qUpStride = oGateOffset + oGroups * oLoraRank
        let wRows = oGroups * oLoraRank
        let bank = makeWoABank(inc: inc, bits: bits, group: group, rows: wRows, cols: groupIn)

        let arena = try QWArena(hal: Self.hal, weights: max(4096, bank.count + 4096),
                                activations: 16 << 20, kv: 0, gdnStates: 0)
        let wr = arena.weightsRegion()
        let ar = arena.activationsRegion()
        let wBuf = try wr.allocate(QWTensorShape([max(1, (bank.count + 1) / 2)], .f16))
        arena.copyIn(wBuf, from: bank)
        let zIn = try ar.allocate(QWTensorShape([T * oGroups * groupIn], .f16))
        let qUp = try ar.allocate(QWTensorShape([T * qUpStride], .f16))
        let outRef = try ar.allocate(QWTensorShape([T * wRows], .f16))
        let outCand = try ar.allocate(QWTensorShape([T * wRows], .f16))
        arena.write(zIn, from: makeData(count: T * oGroups * groupIn, seed: 0x7777, scale: 0.25).map { Float16($0) })
        arena.write(qUp, from: makeData(count: T * qUpStride, seed: 0x8888, scale: 0.5).map { Float16($0) })

        var prm = QWHCAAttnOperator.GroupedWoAParams(
            T: UInt32(T), oGroups: UInt32(oGroups), oLoraRank: UInt32(oLoraRank),
            groupIn: UInt32(groupIn), qUpStride: UInt32(qUpStride), oGateOffset: UInt32(oGateOffset),
            bits: UInt32(bits), group: UInt32(group), isIncumbent: UInt32(inc)
        )
        let pRef = try Self.hal.pipeline(useNAX ? "qwg_hca_grouped_wo_a_mma" : "qwg_hca_grouped_wo_a_gemv")
        let pCand = try Self.hal.pipeline(useNAX ? "qwg_hca_grouped_wo_a_nax" : "qwg_hca_grouped_wo_a_gemv_mr")

        func launch(_ pso: any MTLComputePipelineState, dst: QWTensorValue, grid: MTLSize, tg: Int) throws {
            guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
            enc.setComputePipelineState(pso)
            zIn.bind(enc, index: 0)
            wBuf.bind(enc, index: 1)
            qUp.bind(enc, index: 2)
            dst.bind(enc, index: 3)
            enc.setBytes(&prm, length: MemoryLayout<QWHCAAttnOperator.GroupedWoAParams>.stride, index: 4)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        let gRef = useNAX
            ? MTLSize(width: oGroups, height: (T + 31) / 32, depth: oLoraRank / 128)
            : MTLSize(width: oLoraRank / 32, height: oGroups, depth: T)
        let gCand = useNAX
            ? MTLSize(width: oLoraRank / 64, height: (T + 63) / 64, depth: oGroups)
            : MTLSize(width: oLoraRank / 32, height: oGroups, depth: 1)
        try launch(pRef, dst: outRef, grid: gRef, tg: useNAX ? 128 : 256)
        try launch(pCand, dst: outCand, grid: gCand, tg: useNAX ? 128 : 256)

        let ref = arena.read(outRef, as: Float16.self, count: T * wRows)
        let got = arena.read(outCand, as: Float16.self, count: T * wRows)
        var maxDiff: Float = 0
        for i in 0..<ref.count { maxDiff = max(maxDiff, abs(Float(got[i]) - Float(ref[i]))) }
        XCTAssertLessThan(maxDiff, useNAX ? 5e-3 : 2e-3, "\(label) T=\(T) diff=\(maxDiff)")
        if !useNAX {
            XCTAssertEqual(got.map(\.bitPattern), ref.map(\.bitPattern),
                           "\(label) T=\(T) gemv_mr must be bit-exact with single-row gemv")
        }
    }

    func testHCAGroupedWoANAXAndGemvMRMatchReference() throws {
        let formats: [(String, Int, Int, Int)] = [
            ("rawF16", 3, 16, 32),
            ("HQ4_G32", 1, 4, 32),
            ("AQ8_G32", 0, 8, 32),
            ("AQ4_G64", 0, 4, 64),
            ("SHVQ3_G64", 2, 3, 64),
        ]
        for (name, inc, bits, group) in formats {
            for t in [18, 64] {
                try runGroupedWoACase(label: "nax_\(name)", rows: t, useNAX: true, inc: inc, bits: bits, group: group)
            }
            for t in [4, 8] {
                try runGroupedWoACase(label: "mr_\(name)", rows: t, useNAX: false, inc: inc, bits: bits, group: group)
            }
        }
        for t in [3, 8] {
            try runGroupedWoACase(label: "mr_AQ8_G32_k4096", rows: t, useNAX: false, inc: 0, bits: 8, group: 32,
                                  oGroups: 2, oLoraRank: 128, groupIn: 4096)
        }
    }

    func testHCAAttnNAXPrefillBitExactMatchesNAX() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "qwg_hca_attn_nax_prefill requires Apple10+ NAX")
        let pRef = try Self.hal.pipeline("qwg_hca_attn_nax")
        let pPrefill = try Self.hal.pipeline("qwg_hca_attn_nax_prefill")
        let rows = 33, pos0 = 260, qHeads = 64, headDim = 512, qUpStride = qHeads * headDim
        let compressRatio = 128, slidingWindow = 128, ringMask = 255, compBudget = 8, maxComp = 16
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 64 << 20, kv: 16 << 20, gdnStates: 0)
        let ar = arena.activationsRegion()
        let qUp = try ar.allocate(QWTensorShape([rows * qUpStride], .f16))
        let compKV = try ar.allocate(QWTensorShape([maxComp * headDim], .f16))
        let swaKV = try ar.allocate(QWTensorShape([(ringMask + 1) * headDim], .f16))
        let blkList = try ar.allocate(QWTensorShape([rows * compBudget], .i32))
        let blkCnt = try ar.allocate(QWTensorShape([rows], .i32))
        let sink = try ar.allocate(QWTensorShape([qHeads], .f32))
        let partML = try ar.allocate(QWTensorShape([4096], .f32))
        let outRef = try ar.allocate(QWTensorShape([rows * qHeads * headDim], .f16))
        let outPre = try ar.allocate(QWTensorShape([rows * qHeads * headDim], .f16))
        arena.write(qUp, from: makeData(count: rows * qUpStride, seed: 0x1111).map { Float16($0) })
        arena.write(compKV, from: makeData(count: maxComp * headDim, seed: 0x2222).map { Float16($0) })
        arena.write(swaKV, from: makeData(count: (ringMask + 1) * headDim, seed: 0x3333).map { Float16($0) })
        arena.write(sink, from: (0..<qHeads).map { Float($0 % 5) * 0.25 - 1.0 })
        let bList = (0..<(rows * compBudget)).map { Int32(($0 * 3 + 1) % maxComp) }
        arena.write(blkList, from: bList)
        arena.write(blkCnt, from: (0..<rows).map { Int32(2 + ($0 % (compBudget - 1))) })
        for useBlockList: UInt32 in [0, 1] {
            var prm = QWHCAAttnOperator.AttnParams(
                T: UInt32(rows), pos0: UInt32(pos0), qHeads: UInt32(qHeads), headDim: UInt32(headDim),
                qUpStride: UInt32(qUpStride), compressRatio: UInt32(compressRatio),
                slidingWindow: UInt32(slidingWindow), ringMask: UInt32(ringMask),
                compBudget: UInt32(compBudget), listStride: UInt32(compBudget),
                useBlockList: useBlockList, hasSink: 1)
            func run(_ pso: any MTLComputePipelineState, dst: QWTensorValue, grid: MTLSize) throws {
                guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
                enc.setComputePipelineState(pso)
                for (i, buf) in [qUp, compKV, swaKV, blkList, blkCnt, sink, dst, partML].enumerated() {
                    buf.bind(enc, index: i)
                }
                enc.setBytes(&prm, length: MemoryLayout<QWHCAAttnOperator.AttnParams>.stride, index: 8)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                Self.hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
            }
            try run(pRef, dst: outRef, grid: MTLSize(width: rows, height: qHeads / 32, depth: 1))
            try run(pPrefill, dst: outPre, grid: MTLSize(width: qHeads / 32, height: rows, depth: 1))
            let a = arena.read(outRef, as: Float16.self, count: rows * qHeads * headDim).map(\.bitPattern)
            let b = arena.read(outPre, as: Float16.self, count: rows * qHeads * headDim).map(\.bitPattern)
            XCTAssertEqual(b, a, "useBlockList=\(useBlockList) nax_prefill must match nax bit-exact")
        }
    }
}
