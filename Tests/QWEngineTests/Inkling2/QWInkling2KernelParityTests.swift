// QWInkling2KernelParityTests.swift — GPU vs Float64 CPU parity across all Inkling-2 Metal 4 kernels.
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL

final class QWInkling2KernelParityTests: XCTestCase {
    private typealias R = QWInk2TestRef

    private func makeHAL() throws -> QWHal {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Metal device required")
        return try QWHal(geometry: QWModelGeometry.qwen38_27b, weightBasis: .standard)
    }

    private func buf<T>(_ hal: QWHal, _ array: [T]) -> any MTLBuffer {
        let len = max(16, array.count * MemoryLayout<T>.stride)
        let b = hal.device.makeBuffer(length: len, options: .storageModeShared)!
        if !array.isEmpty {
            _ = array.withUnsafeBytes { memcpy(b.contents(), $0.baseAddress!, $0.count) }
        }
        return b
    }

    private func emptyBuf(_ hal: QWHal, bytes: Int) -> any MTLBuffer {
        let b = hal.device.makeBuffer(length: max(16, bytes), options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }

    private func runCmd(_ hal: QWHal, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = hal.begin() else { XCTFail("cannot begin CB"); return }
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { XCTFail("GPU error: \(err)") }
    }

    func testEmbedNormAndRMSNormParity() throws {
        let hal = try makeHAL()
        let V = 16, H = 256, T = 3
        let eps: Float = 1e-6
        let embF = R.randomBF16Floats(count: V * H, seed: 101)
        let normWF = R.randomBF16Floats(count: H, seed: 102, scale: 2.0)
        let toks: [Int32] = [2, 15, 0]

        let outBuf = emptyBuf(hal, bytes: T * H * 2)
        var ep = R.EmbedParams(T: UInt32(T), H: UInt32(H), V: UInt32(V), eps: eps)
        let pEmbed = try hal.pipeline("qwg_ink2_embed_norm")
        let embBuf = buf(hal, R.bf16Array(embF))
        let wBuf = buf(hal, R.bf16Array(normWF))
        let tokBuf = buf(hal, toks)

        try runCmd(hal) { enc in
            enc.setComputePipelineState(pEmbed)
            enc.setBuffer(tokBuf, offset: 0, index: 0)
            enc.setBuffer(embBuf, offset: 0, index: 1)
            enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(outBuf, offset: 0, index: 3)
            enc.setBytes(&ep, length: MemoryLayout<R.EmbedParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotEmb = R.floatsFromBF16Buffer(outBuf, count: T * H)
        for t in 0..<T {
            let tok = Int(toks[t])
            var sq = 0.0
            for d in 0..<H { let x = Double(embF[tok * H + d]); sq += x * x }
            let inv = 1.0 / sqrt(sq / Double(H) + Double(eps))
            for d in 0..<H {
                let want = R.roundBF16(Double(embF[tok * H + d]) * inv * Double(normWF[d]))
                XCTAssertLessThanOrEqual(abs(gotEmb[t * H + d] - want), 1e-2)
            }
        }

        // RMSNorm from F32 input
        let inF32 = R.randomF32Floats(count: T * H, seed: 103, scale: 1.5)
        let rmsOut = emptyBuf(hal, bytes: T * H * 2)
        let dummyBF = emptyBuf(hal, bytes: 16)
        var np = R.NormParams(rows: UInt32(T), dim: UInt32(H), inIsF32: 1, eps: eps)
        let pNorm = try hal.pipeline("qwg_ink2_rms_norm")
        let inF32Buf = buf(hal, inF32)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pNorm)
            enc.setBuffer(dummyBF, offset: 0, index: 0)
            enc.setBuffer(inF32Buf, offset: 0, index: 1)
            enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(rmsOut, offset: 0, index: 3)
            enc.setBytes(&np, length: MemoryLayout<R.NormParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotRms = R.floatsFromBF16Buffer(rmsOut, count: T * H)
        for r in 0..<T {
            var sq = 0.0
            for d in 0..<H { let x = Double(inF32[r * H + d]); sq += x * x }
            let inv = 1.0 / sqrt(sq / Double(H) + Double(eps))
            for d in 0..<H {
                let want = R.roundBF16(Double(inF32[r * H + d]) * inv * Double(normWF[d]))
                XCTAssertLessThanOrEqual(abs(gotRms[r * H + d] - want), 1e-2)
            }
        }
    }

    func testRouterTopKParity() throws {
        let hal = try makeHAL()
        let T = 10, H = 64, E = 16, topK = 4
        let routeScale: Float = 8.0
        let xF = R.randomBF16Floats(count: T * H, seed: 301)
        let gateWF = R.randomBF16Floats(count: E * H, seed: 302)
        let biasF = R.randomF32Floats(count: E, seed: 303, scale: 0.2)

        let topIdsBuf = emptyBuf(hal, bytes: T * topK * 4)
        let topWBuf = emptyBuf(hal, bytes: T * topK * 4)
        var rp = R.RouterParams(T: UInt32(T), H: UInt32(H), numExperts: UInt32(E),
                                topK: UInt32(topK), routeScale: routeScale)
        let pRouter = try hal.pipeline("qwg_ink2_router_topk")
        let xBuf = buf(hal, R.bf16Array(xF))
        let gateWBuf = buf(hal, R.bf16Array(gateWF))
        let biasBuf = buf(hal, biasF)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pRouter)
            enc.setBuffer(xBuf, offset: 0, index: 0)
            enc.setBuffer(gateWBuf, offset: 0, index: 1)
            enc.setBuffer(biasBuf, offset: 0, index: 2)
            enc.setBuffer(topIdsBuf, offset: 0, index: 3)
            enc.setBuffer(topWBuf, offset: 0, index: 4)
            enc.setBytes(&rp, length: MemoryLayout<R.RouterParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let idsPtr = topIdsBuf.contents().assumingMemoryBound(to: Int32.self)
        let wPtr = topWBuf.contents().assumingMemoryBound(to: Float.self)
        for t in 0..<T {
            var scores = [Double](repeating: 0, count: E)
            var biased = [Double](repeating: 0, count: E)
            for e in 0..<E {
                var d = 0.0
                for k in 0..<H { d += Double(xF[t * H + k]) * Double(gateWF[e * H + k]) }
                let s = 1.0 / (1.0 + exp(-d))
                scores[e] = s
                biased[e] = s + Double(biasF[e])
            }
            let sorted = (0..<E).sorted { biased[$0] > biased[$1] }.prefix(topK)
            let denom = sorted.map { scores[$0] }.reduce(0, +) + 1e-20
            for (k, e) in sorted.enumerated() {
                XCTAssertEqual(Int(idsPtr[t * topK + k]), e)
                XCTAssertEqual(wPtr[t * topK + k], Float(scores[e] / denom * Double(routeScale)), accuracy: 1e-4)
            }
        }
    }

    func testAttnResLoopBankAndControllerParity() throws {
        let hal = try makeHAL()
        let T = 3, H = 256, rank = 32, eps: Float = 1e-6
        let b0 = R.randomBF16Floats(count: T * H, seed: 401)
        let b1 = R.randomBF16Floats(count: T * H, seed: 402)
        let partial = R.randomF32Floats(count: T * H, seed: 403)
        let queries = R.randomF32Floats(count: 3 * H, seed: 404)

        var ring = [UInt16](repeating: 0, count: 4 * T * H)
        for i in 0..<(T * H) { ring[1 * T * H + i] = R.bf16Bits(b0[i]); ring[3 * T * H + i] = R.bf16Bits(b1[i]) }
        let outARBuf = emptyBuf(hal, bytes: 3 * T * H * 4)
        var arp = R.AttnResParams(
            T: UInt32(T), H: UInt32(H), nBlocks: 2, numQ: 3, blockStride: UInt32(T * H),
            partialIsF32: 1, normalizeValues: 1, eps: eps, slotIds: (1, 3, 0, 0, 0, 0, 0, 0)
        )
        let pAR = try hal.pipeline("qwg_ink2_attn_res")
        let ringBuf = buf(hal, ring), dummyB = emptyBuf(hal, bytes: 16), partBuf = buf(hal, partial), qBuf = buf(hal, queries)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pAR)
            enc.setBuffer(ringBuf, offset: 0, index: 0); enc.setBuffer(dummyB, offset: 0, index: 1)
            enc.setBuffer(partBuf, offset: 0, index: 2); enc.setBuffer(qBuf, offset: 0, index: 3)
            enc.setBuffer(outARBuf, offset: 0, index: 4); enc.setBytes(&arp, length: MemoryLayout<R.AttnResParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 3, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let refAR = R.cpuAttnRes(
            blocks: [b0.map(Double.init), b1.map(Double.init)],
            partial: partial.map(Double.init),
            queries: (0..<3).map { q in queries[(q * H)..<((q + 1) * H)].map(Double.init) },
            tCount: T, h: H, normalizeValues: true, eps: Double(eps)
        )
        let gotAR = R.floatsFromF32Buffer(outARBuf, count: 3 * T * H)
        for i in 0..<gotAR.count { XCTAssertLessThanOrEqual(abs(Double(gotAR[i]) - refAR[i]), 1e-4) }

        // Loop bank norm + attend
        let partBF = R.randomBF16Floats(count: T * H, seed: 405)
        let msgQ = R.randomF32Floats(count: 4 * H, seed: 406)
        let bankBuf = emptyBuf(hal, bytes: T * 2 * H * 4)
        let msgOutBuf = emptyBuf(hal, bytes: 4 * T * H * 4)
        var bp = R.BankAttendParams(T: UInt32(T), H: UInt32(H), nStates: 2, numQ: 4,
                                    stateStride: UInt32(T * H), eps: eps, slotIds: (3, 0, 0, 0, 0, 0, 0, 0))
        let pBN = try hal.pipeline("qwg_ink2_loop_bank_norm")
        let pBA = try hal.pipeline("qwg_ink2_loop_bank_attend")
        let partBFBuf = buf(hal, R.bf16Array(partBF)), msgQBuf = buf(hal, msgQ)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pBN)
            enc.setBuffer(ringBuf, offset: 0, index: 0); enc.setBuffer(partBFBuf, offset: 0, index: 1)
            enc.setBuffer(bankBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<R.BankAttendParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 2, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.setComputePipelineState(pBA)
            enc.setBuffer(bankBuf, offset: 0, index: 0); enc.setBuffer(msgQBuf, offset: 0, index: 1)
            enc.setBuffer(msgOutBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<R.BankAttendParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 4, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotMsg = R.floatsFromF32Buffer(msgOutBuf, count: 4 * T * H)

        // MessageController
        let downW = R.randomBF16Floats(count: rank * 2 * H, seed: 407)
        let forgetW = R.randomBF16Floats(count: H * rank, seed: 408)
        let gainW = R.randomBF16Floats(count: H * rank, seed: 409)
        let xCtrl = R.randomF32Floats(count: T * H, seed: 410)
        let ctrlOut = emptyBuf(hal, bytes: T * H * 4)
        var cp = R.CtrlParams(T: UInt32(T), H: UInt32(H), rank: UInt32(rank), eps: eps)
        let pCtrl = try hal.pipeline("qwg_ink2_msg_controller")
        let xCtrlBuf = buf(hal, xCtrl), dWBuf = buf(hal, R.bf16Array(downW))
        let fWBuf = buf(hal, R.bf16Array(forgetW)), gWBuf = buf(hal, R.bf16Array(gainW))
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pCtrl)
            enc.setBuffer(xCtrlBuf, offset: 0, index: 0); enc.setBuffer(msgOutBuf, offset: 0, index: 1)
            enc.setBuffer(dWBuf, offset: 0, index: 2); enc.setBuffer(fWBuf, offset: 0, index: 3)
            enc.setBuffer(gWBuf, offset: 0, index: 4); enc.setBuffer(ctrlOut, offset: 0, index: 5)
            enc.setBytes(&cp, length: MemoryLayout<R.CtrlParams>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotCtrl = R.floatsFromF32Buffer(ctrlOut, count: T * H)
        for t in 0..<T {
            var sx = 0.0, sm = 0.0
            for d in 0..<H { let vx = Double(xCtrl[t * H + d]), vm = Double(gotMsg[t * H + d]); sx += vx * vx; sm += vm * vm }
            let ix = 1.0 / sqrt(sx / Double(H) + Double(eps)), im = 1.0 / sqrt(sm / Double(H) + Double(eps))
            var pair = [Double](repeating: 0, count: 2 * H)
            for d in 0..<H {
                pair[d] = Double(R.roundBF16(Double(xCtrl[t * H + d]) * ix))
                pair[H + d] = Double(R.roundBF16(Double(gotMsg[t * H + d]) * im))
            }
            var hid = [Double](repeating: 0, count: rank)
            for r in 0..<rank { for k in 0..<(2 * H) { hid[r] += pair[k] * Double(downW[r * 2 * H + k]) } }
            for d in 0..<H {
                var fd = 0.0, gd = 0.0
                for r in 0..<rank { fd += hid[r] * Double(forgetW[d * rank + r]); gd += hid[r] * Double(gainW[d * rank + r]) }
                let want = (1.0 - tanh(fd)) * Double(xCtrl[t * H + d]) + tanh(gd) * Double(gotMsg[t * H + d])
                XCTAssertLessThanOrEqual(abs(Double(gotCtrl[t * H + d]) - want), 5e-4)
            }
        }
    }

    func testShortConvStaticDynamicAndCommitParity() throws {
        let hal = try makeHAL()
        let C = 64, nKV = 2, T = 5
        let xF = R.randomBF16Floats(count: T * C, seed: 501)
        let wStatic = R.randomBF16Floats(count: C * 4, seed: 502)
        let pStat = try hal.pipeline("qwg_ink2_sconv_static")
        let pComm = try hal.pipeline("qwg_ink2_sconv_commit")
        let stSeq = emptyBuf(hal, bytes: 3 * C * 4), stBulk = emptyBuf(hal, bytes: 3 * C * 4)
        let outSeq = emptyBuf(hal, bytes: T * C * 2), outBulk = emptyBuf(hal, bytes: T * C * 2)
        let wBuf = buf(hal, R.bf16Array(wStatic)), xBulkBuf = buf(hal, R.bf16Array(xF))
        let dummyB = emptyBuf(hal, bytes: 16), dummyF = emptyBuf(hal, bytes: 16)
        let invS5 = Float(1.0 / sqrt(5.0))

        for t in 0..<T {
            let x1 = buf(hal, R.bf16Array(Array(xF[(t * C)..<((t + 1) * C)])))
            var sp = R.SConvParams(T: 1, C: UInt32(C), nKV: 1, kernelSize: 4, updateState: 1, resMode: 0, invSqrtDenom: invS5)
            try runCmd(hal) { enc in
                enc.setComputePipelineState(pStat)
                enc.setBuffer(x1, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(stSeq, offset: 0, index: 2); enc.setBuffer(dummyB, offset: 0, index: 3)
                enc.setBuffer(dummyF, offset: 0, index: 4); enc.setBuffer(outSeq, offset: t * C * 2, index: 5)
                enc.setBuffer(dummyF, offset: 0, index: 6); enc.setBytes(&sp, length: MemoryLayout<R.SConvParams>.stride, index: 7)
                enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
        }
        var spBulk = R.SConvParams(T: UInt32(T), C: UInt32(C), nKV: 1, kernelSize: 4, updateState: 1, resMode: 0, invSqrtDenom: invS5)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pStat)
            enc.setBuffer(xBulkBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(stBulk, offset: 0, index: 2); enc.setBuffer(dummyB, offset: 0, index: 3)
            enc.setBuffer(dummyF, offset: 0, index: 4); enc.setBuffer(outBulk, offset: 0, index: 5)
            enc.setBuffer(dummyF, offset: 0, index: 6); enc.setBytes(&spBulk, length: MemoryLayout<R.SConvParams>.stride, index: 7)
            enc.dispatchThreads(MTLSize(width: C, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.setComputePipelineState(pComm)
            enc.setBuffer(xBulkBuf, offset: 0, index: 0); enc.setBuffer(stBulk, offset: 0, index: 1)
            enc.setBytes(&spBulk, length: MemoryLayout<R.SConvParams>.stride, index: 2)
            enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        XCTAssertEqual(R.floatsFromBF16Buffer(outSeq, count: T * C), R.floatsFromBF16Buffer(outBulk, count: T * C))
        XCTAssertEqual(R.floatsFromF32Buffer(stSeq, count: 3 * C), R.floatsFromF32Buffer(stBulk, count: 3 * C))

        // Dynamic sconv T=1 vs T=5 parity + CPU math check
        let swF = R.randomBF16Floats(count: 4 * C, seed: 503)
        let dynWF = R.randomF32Floats(count: T * nKV * 4, seed: 504)
        let pDyn = try hal.pipeline("qwg_ink2_sconv_dynamic")
        let stDSeq = emptyBuf(hal, bytes: 3 * C * 4), stDBulk = emptyBuf(hal, bytes: 3 * C * 4)
        let outDSeq = emptyBuf(hal, bytes: T * C * 2), outDBulk = emptyBuf(hal, bytes: T * C * 2)
        let swBuf = buf(hal, R.bf16Array(swF)), dwBulkBuf = buf(hal, dynWF)
        for t in 0..<T {
            let x1 = buf(hal, R.bf16Array(Array(xF[(t * C)..<((t + 1) * C)])))
            let dw1 = buf(hal, Array(dynWF[(t * nKV * 4)..<((t + 1) * nKV * 4)]))
            var dp = R.SConvParams(T: 1, C: UInt32(C), nKV: UInt32(nKV), kernelSize: 4, updateState: 1, resMode: 0, invSqrtDenom: 1.0 / 3.0)
            try runCmd(hal) { enc in
                enc.setComputePipelineState(pDyn)
                enc.setBuffer(x1, offset: 0, index: 0); enc.setBuffer(swBuf, offset: 0, index: 1)
                enc.setBuffer(dw1, offset: 0, index: 2); enc.setBuffer(stDSeq, offset: 0, index: 3)
                enc.setBuffer(outDSeq, offset: t * C * 2, index: 4); enc.setBytes(&dp, length: MemoryLayout<R.SConvParams>.stride, index: 5)
                enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
        }
        var dpB = R.SConvParams(T: UInt32(T), C: UInt32(C), nKV: UInt32(nKV), kernelSize: 4, updateState: 1, resMode: 0, invSqrtDenom: 1.0 / 3.0)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pDyn)
            enc.setBuffer(xBulkBuf, offset: 0, index: 0); enc.setBuffer(swBuf, offset: 0, index: 1)
            enc.setBuffer(dwBulkBuf, offset: 0, index: 2); enc.setBuffer(stDBulk, offset: 0, index: 3)
            enc.setBuffer(outDBulk, offset: 0, index: 4); enc.setBytes(&dpB, length: MemoryLayout<R.SConvParams>.stride, index: 5)
            enc.dispatchThreads(MTLSize(width: C, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.setComputePipelineState(pComm)
            enc.setBuffer(xBulkBuf, offset: 0, index: 0); enc.setBuffer(stDBulk, offset: 0, index: 1)
            enc.setBytes(&dpB, length: MemoryLayout<R.SConvParams>.stride, index: 2)
            enc.dispatchThreads(MTLSize(width: C, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        XCTAssertEqual(R.floatsFromBF16Buffer(outDSeq, count: T * C), R.floatsFromBF16Buffer(outDBulk, count: T * C))
        XCTAssertEqual(R.floatsFromF32Buffer(stDSeq, count: 3 * C), R.floatsFromF32Buffer(stDBulk, count: 3 * C))
    }

    func testQKVRAndBandedSinkGQAAcrossKVModes() throws {
        let hal = try makeHAL()
        let pPrep = try hal.pipeline("qwg_ink2_qkvr_prep_kv_write")
        let pAttn = try hal.pipeline("qwg_ink2_gqa_attn")
        let pComb = try hal.pipeline("qwg_ink2_gqa_split_combine")

        let nH = 6, nKV = 2, D = 128, dRel = 16, T = 10, relExtent = 6
        let qIn = R.randomBF16Floats(count: T * nH * D, seed: 601, scale: 3.0)
        let kIn = R.randomBF16Floats(count: T * nKV * D, seed: 602, scale: 3.0)
        let vIn = R.randomBF16Floats(count: T * nKV * D, seed: 603, scale: 2.0)
        let rIn = R.randomBF16Floats(count: T * nH * dRel, seed: 604, scale: 3.0)
        let qSc = R.randomBF16Floats(count: nH * D, seed: 605, scale: 1.5)
        let rSc = R.randomBF16Floats(count: nH * dRel, seed: 606, scale: 1.5)
        let relProj = R.randomBF16Floats(count: dRel * relExtent, seed: 607, scale: 0.3)
        let sink = R.randomBF16Floats(count: T * nH, seed: 608, scale: 1.0)
        let gate = R.randomBF16Floats(count: T * nH * D, seed: 609, scale: 1.5)

        let bQIn = buf(hal, R.bf16Array(qIn)), bKIn = buf(hal, R.bf16Array(kIn)), bVIn = buf(hal, R.bf16Array(vIn))
        let bRIn = buf(hal, R.bf16Array(rIn)), bQSc = buf(hal, R.bf16Array(qSc)), bRSc = buf(hal, R.bf16Array(rSc))
        let bRelP = buf(hal, R.bf16Array(relProj)), bSink = buf(hal, R.bf16Array(sink)), bGate = buf(hal, R.bf16Array(gate))

        for (kvMode, tol) in [(UInt32(0), Float(2e-2)), (UInt32(1), Float(5e-2)), (UInt32(2), Float(0.35))] {
            for sliding in [UInt32(0), UInt32(6)] {
                let ringCap: UInt32 = (sliding > 0) ? 16 : 32
                let qOutBuf = emptyBuf(hal, bytes: T * nH * D * 2)
                let rOutBuf = emptyBuf(hal, bytes: T * nH * dRel * 2)
                let kCacheBuf = emptyBuf(hal, bytes: Int(ringCap) * nKV * D * 2)
                let vCacheBuf = emptyBuf(hal, bytes: Int(ringCap) * nKV * D * 2)
                let kMetaBuf = emptyBuf(hal, bytes: Int(ringCap) * nKV * 4)
                let vMetaBuf = emptyBuf(hal, bytes: Int(ringCap) * nKV * 4)
                let attnOutBuf = emptyBuf(hal, bytes: T * nH * D * 2)
                let splits: UInt32 = (sliding == 0 && kvMode == 0) ? 3 : 1
                let partOBuf = emptyBuf(hal, bytes: T * nH * Int(splits) * D * 4)
                let partMLBuf = emptyBuf(hal, bytes: T * nH * Int(splits) * 8)

                var qp = R.QKVRParams(
                    T: UInt32(T), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(D), dRel: UInt32(dRel),
                    isSliding: sliding > 0 ? 1 : 0, posOffset: 0, ringCap: ringCap, kvMode: kvMode,
                    nFloor: 4, qkClip: 22.627416997969522, relClip: 8.0, logAlpha: 0.1, rmsEps: 1e-6
                )
                var ap = R.AttnParams(
                    T: UInt32(T), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(D), dRel: UInt32(dRel),
                    sliding: sliding, relExtent: UInt32(relExtent), posOffset: 0, ringCap: ringCap,
                    kvMode: kvMode, kvLen: UInt32(T), splits: splits
                )
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(pPrep)
                    enc.setBuffer(bQIn, offset: 0, index: 0); enc.setBuffer(bKIn, offset: 0, index: 1)
                    enc.setBuffer(bVIn, offset: 0, index: 2); enc.setBuffer(bRIn, offset: 0, index: 3)
                    enc.setBuffer(bQSc, offset: 0, index: 4); enc.setBuffer(bRSc, offset: 0, index: 5)
                    enc.setBuffer(qOutBuf, offset: 0, index: 6); enc.setBuffer(rOutBuf, offset: 0, index: 7)
                    enc.setBuffer(kCacheBuf, offset: 0, index: 8); enc.setBuffer(vCacheBuf, offset: 0, index: 9)
                    enc.setBuffer(kMetaBuf, offset: 0, index: 10); enc.setBuffer(vMetaBuf, offset: 0, index: 11)
                    enc.setBytes(&qp, length: MemoryLayout<R.QKVRParams>.stride, index: 12)
                    enc.dispatchThreadgroups(MTLSize(width: max(nH, nKV), height: T, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                    enc.setComputePipelineState(pAttn)
                    enc.setBuffer(qOutBuf, offset: 0, index: 0); enc.setBuffer(rOutBuf, offset: 0, index: 1)
                    enc.setBuffer(bRelP, offset: 0, index: 2); enc.setBuffer(bSink, offset: 0, index: 3)
                    enc.setBuffer(bGate, offset: 0, index: 4); enc.setBuffer(kCacheBuf, offset: 0, index: 5)
                    enc.setBuffer(vCacheBuf, offset: 0, index: 6); enc.setBuffer(kMetaBuf, offset: 0, index: 7)
                    enc.setBuffer(vMetaBuf, offset: 0, index: 8); enc.setBuffer(attnOutBuf, offset: 0, index: 9)
                    enc.setBuffer(partOBuf, offset: 0, index: 10); enc.setBuffer(partMLBuf, offset: 0, index: 11)
                    enc.setBytes(&ap, length: MemoryLayout<R.AttnParams>.stride, index: 12)
                    enc.dispatchThreadgroups(MTLSize(width: Int(splits), height: nH, depth: T),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    if splits > 1 {
                        enc.setComputePipelineState(pComb)
                        enc.setBuffer(partOBuf, offset: 0, index: 0); enc.setBuffer(partMLBuf, offset: 0, index: 1)
                        enc.setBuffer(bGate, offset: 0, index: 2); enc.setBuffer(attnOutBuf, offset: 0, index: 3)
                        enc.setBytes(&ap, length: MemoryLayout<R.AttnParams>.stride, index: 4)
                        enc.dispatchThreadgroups(MTLSize(width: nH, height: T, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    }
                }
                let qPrepared = R.floatsFromBF16Buffer(qOutBuf, count: T * nH * D)
                let rPrepared = R.floatsFromBF16Buffer(rOutBuf, count: T * nH * dRel)
                // Compute CPU K & V clipped/normed (unquantized F32/BF16 reference)
                var kRef = [Float](repeating: 0, count: T * nKV * D)
                var vRef = [Float](repeating: 0, count: T * nKV * D)
                for t in 0..<T {
                    for h in 0..<nKV {
                        let base = (t * nKV + h) * D
                        var sqK = 0.0, sqV = 0.0
                        for d in 0..<D {
                            sqK += Double(kIn[base + d]) * Double(kIn[base + d])
                            sqV += Double(vIn[base + d]) * Double(vIn[base + d])
                        }
                        let scK = min(22.627416997969522 / (sqrt(sqK) + 1e-6), 1.0)
                        let invV = 1.0 / sqrt(sqV / Double(D) + 1e-6)
                        for d in 0..<D {
                            kRef[base + d] = R.roundBF16(Double(kIn[base + d]) * scK)
                            vRef[base + d] = R.roundBF16(Double(vIn[base + d]) * invV)
                        }
                    }
                }
                let want = R.cpuGQAAttention(
                    q: qPrepared, k: kRef, v: vRef, r: rPrepared,
                    relProj: relProj, sink: sink, gate: gate,
                    tCount: T, posOffset: 0, nH: nH, nKV: nKV, headDim: D,
                    dRel: dRel, sliding: Int(sliding), relExtent: relExtent
                )
                let got = R.floatsFromBF16Buffer(attnOutBuf, count: T * nH * D)
                var maxDiff: Float = 0
                for i in 0..<got.count { maxDiff = max(maxDiff, abs(got[i] - want[i])) }
                XCTAssertLessThanOrEqual(maxDiff, tol, "kvMode=\(kvMode) sliding=\(sliding)")
            }
        }
    }

    func testFastControllerAndRouterSelectParity() throws {
        let hal = try makeHAL()
        let T = 4, H = 2304, rank = 64, eps: Float = 1e-6, E = 128, topK = 8

        let logits = R.randomF32Floats(count: T * E, seed: 701, scale: 2.0)
        let bias = R.randomF32Floats(count: E, seed: 702, scale: 0.25)
        let topIdsBuf = emptyBuf(hal, bytes: T * topK * 4)
        let topWBuf = emptyBuf(hal, bytes: T * topK * 4)
        var rp = R.RouterParams(T: UInt32(T), H: UInt32(H), numExperts: UInt32(E),
                                topK: UInt32(topK), routeScale: 2.5)
        let pSel = try hal.pipeline("qwg_ink2_router_select_topk")
        let bLog = buf(hal, logits), bBias = buf(hal, bias)
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pSel)
            enc.setBuffer(bLog, offset: 0, index: 0); enc.setBuffer(bBias, offset: 0, index: 1)
            enc.setBuffer(topIdsBuf, offset: 0, index: 2); enc.setBuffer(topWBuf, offset: 0, index: 3)
            enc.setBytes(&rp, length: MemoryLayout<R.RouterParams>.stride, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let idsPtr = topIdsBuf.contents().assumingMemoryBound(to: Int32.self)
        let wPtr = topWBuf.contents().assumingMemoryBound(to: Float.self)
        for t in 0..<T {
            var sc = [Float](repeating: 0, count: E), bi = [Float](repeating: 0, count: E)
            for e in 0..<E {
                let s = 1.0 / (1.0 + exp(-logits[t * E + e]))
                sc[e] = s; bi[e] = s + bias[e]
            }
            let sorted = (0..<E).sorted { bi[$0] == bi[$1] ? $0 < $1 : bi[$0] > bi[$1] }.prefix(topK)
            let denom = sorted.map { sc[$0] }.reduce(0, +) + 1e-20
            for (k, e) in sorted.enumerated() {
                XCTAssertEqual(Int(idsPtr[t * topK + k]), e)
                XCTAssertEqual(wPtr[t * topK + k], sc[e] / denom * 2.5, accuracy: 1e-5)
            }
        }

        let x = R.randomF32Floats(count: T * H, seed: 703)
        let msg = R.randomF32Floats(count: T * H, seed: 704)
        let hid = R.randomF32Floats(count: T * rank, seed: 705)
        let forgetW = R.randomBF16Floats(count: H * rank, seed: 706)
        let gainW = R.randomBF16Floats(count: H * rank, seed: 707)
        let normW = R.randomBF16Floats(count: H, seed: 708, scale: 1.0)
        let pairBuf = emptyBuf(hal, bytes: T * 2 * H * 2)
        let applyOut = emptyBuf(hal, bytes: T * H * 4)
        let fgBuf = emptyBuf(hal, bytes: T * 2 * H * 4)
        let blendOut = emptyBuf(hal, bytes: T * H * 2)
        let bx = buf(hal, x), bmsg = buf(hal, msg), bhid = buf(hal, hid)
        let bfw = buf(hal, R.bf16Array(forgetW)), bgw = buf(hal, R.bf16Array(gainW)), bnw = buf(hal, R.bf16Array(normW))
        let pPack = try hal.pipeline("qwg_ink2_ctrl_pack_pair")
        let pApply = try hal.pipeline("qwg_ink2_ctrl_apply_f32")
        let pBlend = try hal.pipeline("qwg_ink2_ctrl_blend_norm")
        var cp = R.CtrlParams(T: UInt32(T), H: UInt32(H), rank: UInt32(rank), eps: eps)
        let fgF = R.randomF32Floats(count: T * 2 * H, seed: 709)
        let bfg = buf(hal, fgF)
        _ = fgBuf
        try runCmd(hal) { enc in
            enc.setComputePipelineState(pPack)
            enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
            enc.setBuffer(pairBuf, offset: 0, index: 2); enc.setBytes(&cp, length: MemoryLayout<R.CtrlParams>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.setComputePipelineState(pApply)
            enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
            enc.setBuffer(bhid, offset: 0, index: 2); enc.setBuffer(bfw, offset: 0, index: 3)
            enc.setBuffer(bgw, offset: 0, index: 4); enc.setBuffer(applyOut, offset: 0, index: 5)
            enc.setBytes(&cp, length: MemoryLayout<R.CtrlParams>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (H + 31) / 32, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.setComputePipelineState(pBlend)
            enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
            enc.setBuffer(bfg, offset: 0, index: 2); enc.setBuffer(bnw, offset: 0, index: 3)
            enc.setBuffer(blendOut, offset: 0, index: 4); enc.setBytes(&cp, length: MemoryLayout<R.CtrlParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let gotPair = R.floatsFromBF16Buffer(pairBuf, count: T * 2 * H)
        let gotApply = R.floatsFromF32Buffer(applyOut, count: T * H)
        let gotBlend = R.floatsFromBF16Buffer(blendOut, count: T * H)
        for t in 0..<T {
            var sx = 0.0, sm = 0.0, sb = 0.0
            var bArr = [Double](repeating: 0, count: H)
            for d in 0..<H {
                let xv = Double(x[t * H + d]), mv = Double(msg[t * H + d])
                sx += xv * xv; sm += mv * mv
                let fv = fgF[t * 2 * H + d], gv = fgF[t * 2 * H + H + d]
                let bv = (1.0 - tanh(Double(fv))) * xv + tanh(Double(gv)) * mv
                bArr[d] = bv; sb += bv * bv
                var fd = 0.0, gd = 0.0
                for r in 0..<rank {
                    fd += Double(hid[t * rank + r]) * Double(forgetW[d * rank + r])
                    gd += Double(hid[t * rank + r]) * Double(gainW[d * rank + r])
                }
                let wantApply = (1.0 - tanh(fd)) * xv + tanh(gd) * mv
                XCTAssertEqual(Double(gotApply[t * H + d]), wantApply, accuracy: 2e-4)
            }
            let ix = 1.0 / sqrt(sx / Double(H) + Double(eps)), im = 1.0 / sqrt(sm / Double(H) + Double(eps))
            let ib = 1.0 / sqrt(sb / Double(H) + Double(eps))
            for d in 0..<H {
                XCTAssertEqual(gotPair[t * 2 * H + d], R.roundBF16(Double(x[t * H + d]) * ix), accuracy: 1e-2)
                XCTAssertEqual(gotPair[t * 2 * H + H + d], R.roundBF16(Double(msg[t * H + d]) * im), accuracy: 1e-2)
                XCTAssertEqual(gotBlend[t * H + d], R.roundBF16(bArr[d] * ib * Double(normW[d])), accuracy: 1.5e-2)
            }
        }
    }

    private struct CtrlBatchParams {
        var T, H, rank, thStride: UInt32
        var eps: Float
        var numBranches, msgBaseSlot: UInt32
        var pad1: UInt32 = 0
        var arIdx: (UInt32, UInt32, UInt32, UInt32)
        var downElemOff: (UInt32, UInt32, UInt32, UInt32)
        var forgetElemOff: (UInt32, UInt32, UInt32, UInt32)
        var gainElemOff: (UInt32, UInt32, UInt32, UInt32)
        var normElemOff: (UInt32, UInt32, UInt32, UInt32)
    }

    func testBatchedControllerBranchesParity() throws {
        let hal = try makeHAL()
        let H = 256, rank = 64, eps: Float = 1e-6, nb = 3, maxT = 130
        let downElems = rank * 2 * H, partElems = H * rank
        var downSlab = [UInt16](repeating: 0, count: nb * downElems)
        var fgSlab = [UInt16](repeating: 0, count: 2 * nb * partElems)
        var normSlab = [UInt16](repeating: 0, count: nb * H)
        var downF = [[Float]](), fgF = [[Float]](), normF = [[Float]]()
        for b in 0..<nb {
            let dw = R.randomBF16Floats(count: downElems, seed: UInt64(800 + b), scale: 0.4)
            let fw = R.randomBF16Floats(count: partElems, seed: UInt64(810 + b), scale: 0.5)
            let gw = R.randomBF16Floats(count: partElems, seed: UInt64(820 + b), scale: 0.5)
            let nw = R.randomBF16Floats(count: H, seed: UInt64(830 + b), scale: 1.0)
            downF.append(dw); fgF.append(fw + gw); normF.append(nw)
            let db = R.bf16Array(dw), fb = R.bf16Array(fw), gb = R.bf16Array(gw), nbArr = R.bf16Array(nw)
            downSlab.replaceSubrange((b * downElems)..<((b + 1) * downElems), with: db)
            fgSlab.replaceSubrange((2 * b * partElems)..<((2 * b + 1) * partElems), with: fb)
            fgSlab.replaceSubrange(((2 * b + 1) * partElems)..<((2 * b + 2) * partElems), with: gb)
            normSlab.replaceSubrange((b * H)..<((b + 1) * H), with: nbArr)
        }
        let bDown = buf(hal, downSlab), bFG = buf(hal, fgSlab), bNormW = buf(hal, normSlab)
        let xAll = R.randomF32Floats(count: 3 * maxT * H, seed: 840)
        let msgAll = R.randomF32Floats(count: 4 * maxT * H, seed: 841)
        let bx = buf(hal, xAll), bmsg = buf(hal, msgAll)
        let pairBuf = emptyBuf(hal, bytes: nb * maxT * 2 * H * 2)
        let hidBuf = emptyBuf(hal, bytes: nb * maxT * rank * 4)
        let outF32Buf = emptyBuf(hal, bytes: nb * maxT * 2 * H * 4)
        let outBF16Buf = emptyBuf(hal, bytes: nb * maxT * H * 2)

        let pPackB = try hal.pipeline("qwg_ink2_ctrl_pack_pair_batch")
        let pDownB = try hal.pipeline("qwg_ink2_ctrl_down_gemv_batch")
        let pApplyB = try hal.pipeline("qwg_ink2_ctrl_apply_f32_batch")
        let pRmsB = try hal.pipeline("qwg_ink2_ctrl_rms_norm_batch")
        let pBlendB = try hal.pipeline("qwg_ink2_ctrl_blend_norm_batch")
        let pDownNaxB = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_ink2_ctrl_down_nax_batch") : nil
        let pFgNaxB = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_ink2_ctrl_fg_blend_nax_batch") : nil

        var row0T1: [Float]?
        for T in [1, 8, 16, 32, 64, 130] {
            var xT = [Float](repeating: 0, count: nb * T * H)
            var msgT = [Float](repeating: 0, count: nb * T * H)
            for b in 0..<nb {
                for t in 0..<T {
                    let srcOff = (b * maxT + t) * H
                    let dstOff = (b * T + t) * H
                    xT.replaceSubrange(dstOff..<(dstOff + H), with: xAll[srcOff..<(srcOff + H)])
                    msgT.replaceSubrange(dstOff..<(dstOff + H), with: msgAll[srcOff..<(srcOff + H)])
                }
            }
            _ = xT.withUnsafeBytes { memcpy(bx.contents(), $0.baseAddress!, $0.count) }
            _ = msgT.withUnsafeBytes { memcpy(bmsg.contents(), $0.baseAddress!, $0.count) }
            var bp = CtrlBatchParams(
                T: UInt32(T), H: UInt32(H), rank: UInt32(rank), thStride: UInt32(T * H),
                eps: eps, numBranches: UInt32(nb), msgBaseSlot: 0,
                arIdx: (0, 1, 2, 0),
                downElemOff: (0, UInt32(downElems), UInt32(2 * downElems), 0),
                forgetElemOff: (0, UInt32(2 * partElems), UInt32(4 * partElems), 0),
                gainElemOff: (UInt32(partElems), UInt32(3 * partElems), UInt32(5 * partElems), 0),
                normElemOff: (0, UInt32(H), UInt32(2 * H), 0)
            )
            if T <= 16 {
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(pPackB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(pairBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.setComputePipelineState(pDownB)
                    enc.setBuffer(pairBuf, offset: 0, index: 0); enc.setBuffer(bDown, offset: 0, index: 1)
                    enc.setBuffer(hidBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: (rank + 3) / 4, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    enc.setComputePipelineState(pApplyB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(hidBuf, offset: 0, index: 2); enc.setBuffer(bFG, offset: 0, index: 3)
                    enc.setBuffer(outF32Buf, offset: 0, index: 4); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: (rank == 64) ? ((H + 63) / 64) : ((H + 31) / 32), height: T, depth: nb),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.setComputePipelineState(pRmsB)
                    enc.setBuffer(outF32Buf, offset: 0, index: 0); enc.setBuffer(bNormW, offset: 0, index: 1)
                    enc.setBuffer(outBF16Buf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                let got = R.floatsFromBF16Buffer(outBF16Buf, count: nb * T * H)
                if T == 1 {
                    row0T1 = got
                } else if let t1 = row0T1 {
                    for b in 0..<nb {
                        for d in 0..<H {
                            XCTAssertEqual(got[(b * T) * H + d], t1[b * H + d], "T=1 vs T=\(T) bit-exact mismatch at b=\(b), d=\(d)")
                        }
                    }
                }
            } else if let pDownNaxB, let pFgNaxB {
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(pPackB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(pairBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.setComputePipelineState(pDownNaxB)
                    enc.setBuffer(pairBuf, offset: 0, index: 0); enc.setBuffer(bDown, offset: 0, index: 1)
                    enc.setBuffer(hidBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: (rank + 63) / 64, height: (T + 63) / 64, depth: nb),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    enc.setComputePipelineState(pFgNaxB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(hidBuf, offset: 0, index: 2); enc.setBuffer(bFG, offset: 0, index: 3)
                    enc.setBuffer(outF32Buf, offset: 0, index: 4); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: (H + 31) / 32, height: (T + 63) / 64, depth: nb),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    enc.setComputePipelineState(pRmsB)
                    enc.setBuffer(outF32Buf, offset: 0, index: 0); enc.setBuffer(bNormW, offset: 0, index: 1)
                    enc.setBuffer(outBF16Buf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
            } else {
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(pPackB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(pairBuf, offset: 0, index: 2); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                let pairs = R.floatsFromBF16Buffer(pairBuf, count: nb * T * 2 * H)
                var fgInputs = [Float](repeating: 0, count: nb * T * 2 * H)
                for b in 0..<nb {
                    for t in 0..<T {
                        var hVec = [Double](repeating: 0, count: rank)
                        for r in 0..<rank {
                            var s = 0.0
                            for k in 0..<(2 * H) { s += Double(pairs[(b * T + t) * 2 * H + k]) * Double(downF[b][r * 2 * H + k]) }
                            hVec[r] = s
                        }
                        for d in 0..<(2 * H) {
                            var s = 0.0
                            for r in 0..<rank { s += hVec[r] * Double(fgF[b][d * rank + r]) }
                            fgInputs[(b * T + t) * 2 * H + d] = Float(s)
                        }
                    }
                }
                _ = fgInputs.withUnsafeBytes { memcpy(outF32Buf.contents(), $0.baseAddress!, $0.count) }
                try runCmd(hal) { enc in
                    enc.setComputePipelineState(pBlendB)
                    enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bmsg, offset: 0, index: 1)
                    enc.setBuffer(outF32Buf, offset: 0, index: 2); enc.setBuffer(bNormW, offset: 0, index: 3)
                    enc.setBuffer(outBF16Buf, offset: 0, index: 4); enc.setBytes(&bp, length: MemoryLayout<CtrlBatchParams>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: T, height: nb, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
            }
            let got = R.floatsFromBF16Buffer(outBF16Buf, count: nb * T * H)
            for b in 0..<nb {
                for t in 0..<T {
                    let sBase = (b * maxT + t) * H
                    var sx = 0.0, sm = 0.0
                    for d in 0..<H {
                        let xv = Double(xAll[sBase + d]), mv = Double(msgAll[sBase + d])
                        sx += xv * xv; sm += mv * mv
                    }
                    let ix = 1.0 / sqrt(sx / Double(H) + Double(eps)), im = 1.0 / sqrt(sm / Double(H) + Double(eps))
                    var pair = [Double](repeating: 0, count: 2 * H)
                    for d in 0..<H {
                        pair[d] = Double(R.roundBF16(Double(xAll[sBase + d]) * ix))
                        pair[H + d] = Double(R.roundBF16(Double(msgAll[sBase + d]) * im))
                    }
                    var hid = [Double](repeating: 0, count: rank)
                    for r in 0..<rank {
                        var s = 0.0
                        for k in 0..<(2 * H) { s += pair[k] * Double(downF[b][r * 2 * H + k]) }
                        hid[r] = s
                    }
                    var outV = [Double](repeating: 0, count: H)
                    var sb = 0.0
                    for d in 0..<H {
                        var fd = 0.0, gd = 0.0
                        for r in 0..<rank {
                            fd += hid[r] * Double(fgF[b][d * rank + r])
                            gd += hid[r] * Double(fgF[b][(H + d) * rank + r])
                        }
                        let v = (1.0 - tanh(fd)) * Double(xAll[sBase + d]) + tanh(gd) * Double(msgAll[sBase + d])
                        outV[d] = v; sb += v * v
                    }
                    let ib = 1.0 / sqrt(sb / Double(H) + Double(eps))
                    for d in 0..<H {
                        XCTAssertEqual(got[(b * T + t) * H + d], R.roundBF16(outV[d] * ib * Double(normF[b][d])), accuracy: 2e-2)
                    }
                }
            }
        }
    }
}
