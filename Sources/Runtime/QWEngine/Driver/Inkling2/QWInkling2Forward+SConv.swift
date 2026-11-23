// QWInkling2Forward+SConv.swift — Static/dynamic short-conv taps and fused static K/V SConv + QKVR dispatch.
import Foundation
import Metal
import QWCore

typealias Ink2TreeLags16 = (
    SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>,
    SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>,
    SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>,
    SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>, SIMD4<Int32>
)

typealias Ink2TreeDepths16 = (
    UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
    UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
)

struct Ink2SConvTapP {
    var T, C, nKV, kernelSize, updateState, resMode, writeVerifyTap, verifyTapOffset: UInt32
    var invSqrtDenom: Float
}

struct Ink2SConvTreeTapP {
    var T, C, nKV, kernelSize, updateState, resMode, writeVerifyTap, verifyTapOffset: UInt32
    var invSqrtDenom: Float
    var pad0: UInt32 = 0; var pad1: UInt32 = 0; var pad2: UInt32 = 0
    var treeLags: Ink2TreeLags16
}

struct Ink2QKVRSConvFusedP {
    var T, nH, nKV, headDim, dRel: UInt32
    var isSliding, posOffset, ringCap, kvMode, nFloor: UInt32
    var qkClip, relClip, logAlpha, rmsEps: Float
    var kvInRowStride, updateState, writeVerifyTap, kVerifyTapOffset, vVerifyTapOffset, writeKConv: UInt32
    var invSqrtDenom: Float
    var pad0: UInt32 = 0; var pad1: UInt32 = 0; var pad2: UInt32 = 0
}

struct Ink2QKVRSConvTreeFusedP {
    var base: Ink2QKVRSConvFusedP
    var depths: Ink2TreeDepths16
    var treeLags: Ink2TreeLags16
}

struct Ink2SConvCommitPairP {
    var T, C, kvInRowStride: UInt32
    var pad0: UInt32 = 0
}

extension QWInkling2Forward {
    private func packSConvTreeLags(_ lags: [SIMD4<Int32>]) -> Ink2TreeLags16 {
        func g(_ i: Int) -> SIMD4<Int32> { i < lags.count ? lags[i] : SIMD4<Int32>(-1, -2, -3, 0) }
        return (g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7),
                g(8), g(9), g(10), g(11), g(12), g(13), g(14), g(15))
    }

    private func encodeSConvCommit(
        _ enc: any MTLComputeCommandEncoder,
        xBF16: any MTLBuffer, xOff: Int, convSlab: any MTLBuffer, stOff: Int,
        t: Int, c: Int, nKV: Int, resMode: UInt32, invSqrt: Float
    ) {
        var sp = SConvP(T: UInt32(t), C: UInt32(c), nKV: UInt32(nKV), kSize: 4,
                        updState: 1, resMode: resMode, invSqrt: invSqrt)
        enc.setComputePipelineState(pSConvCommit)
        enc.setBuffer(xBF16, offset: xOff, index: 0)
        enc.setBuffer(convSlab, offset: stOff, index: 1)
        enc.setBytes(&sp, length: MemoryLayout<SConvP>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: c, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(256, c), height: 1, depth: 1))
    }

    func encodeSConvStaticTap(
        _ enc: any MTLComputeCommandEncoder,
        xBF16: any MTLBuffer, xOff: Int = 0, weight: Ink2Slice, convSlot: Int,
        convSlab: (any MTLBuffer)? = nil,
        resBF16: any MTLBuffer, resBF16Off: Int = 0, resF32: any MTLBuffer, resF32Off: Int = 0,
        outBF16: any MTLBuffer, outBF16Off: Int = 0, outF32: any MTLBuffer, outF32Off: Int = 0,
        t: Int, c: Int, resMode: UInt32, denom: Float, updateState: Bool,
        recordVerifyConv: Bool, treePlan: Ink2TreeVerifyPlan? = nil
    ) {
        let slab = convSlab ?? session.convStatesSlab
        let stOff = session.convStateOffsets[convSlot]
        let writeTap = recordVerifyConv && session.convVerifyInputsBuf != nil
        let tapOff = writeTap ? UInt32(session.convVerifyOffsets[convSlot] / 2) : 0
        let tgW = min(256, c)
        enc.setBuffer(xBF16, offset: xOff, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(slab, offset: stOff, index: 2)
        enc.setBuffer(resBF16, offset: resBF16Off, index: 3)
        enc.setBuffer(resF32, offset: resF32Off, index: 4)
        enc.setBuffer(outBF16, offset: outBF16Off, index: 5)
        enc.setBuffer(outF32, offset: outF32Off, index: 6)
        enc.setBuffer(session.convVerifyInputsBuf ?? session.dummyB, offset: 0, index: 7)
        if let treePlan {
            var tp = Ink2SConvTreeTapP(
                T: UInt32(t), C: UInt32(c), nKV: 1, kernelSize: 4,
                updateState: 0, resMode: resMode,
                writeVerifyTap: writeTap ? 1 : 0, verifyTapOffset: tapOff,
                invSqrtDenom: 1.0 / denom.squareRoot(),
                treeLags: packSConvTreeLags(treePlan.sconvLags)
            )
            enc.setComputePipelineState(pSConvStatTreeTap)
            enc.setBytes(&tp, length: MemoryLayout<Ink2SConvTreeTapP>.stride, index: 8)
            enc.dispatchThreads(MTLSize(width: c, height: t, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
            return
        }
        var tp = Ink2SConvTapP(
            T: UInt32(t), C: UInt32(c), nKV: 1, kernelSize: 4,
            updateState: updateState ? 1 : 0, resMode: resMode,
            writeVerifyTap: writeTap ? 1 : 0, verifyTapOffset: tapOff,
            invSqrtDenom: 1.0 / denom.squareRoot()
        )
        enc.setComputePipelineState(pSConvStatTap)
        enc.setBytes(&tp, length: MemoryLayout<Ink2SConvTapP>.stride, index: 8)
        enc.dispatchThreads(MTLSize(width: c, height: t, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
        if t > 1 && updateState {
            encodeSConvCommit(enc, xBF16: xBF16, xOff: xOff, convSlab: slab, stOff: stOff,
                              t: t, c: c, nKV: 1, resMode: resMode, invSqrt: 1.0 / denom.squareRoot())
        }
    }

    func encodeSConvDynamicTap(
        _ enc: any MTLComputeCommandEncoder,
        projNormOff: Int, projOutBuf: any MTLBuffer, projOutOff: Int,
        staticW: Ink2Slice, projW: Ink2LinearW, convSlot: Int,
        convSlab: (any MTLBuffer)? = nil, dynWOff: Int = 0,
        outConv: any MTLBuffer, outConvOff: Int = 0,
        t: Int, kvDim: Int, nKV: Int,
        updateState: Bool, recordVerifyConv: Bool, treePlan: Ink2TreeVerifyPlan? = nil
    ) throws {
        try encodeLinearF32(enc, weight: projW, xBuf: session.normBuf, xOff: projNormOff,
                            yF32Buf: session.dynWF32Buf, yOff: dynWOff, m: t)
        let slab = convSlab ?? session.convStatesSlab
        let stOff = session.convStateOffsets[convSlot]
        let writeTap = recordVerifyConv && session.convVerifyInputsBuf != nil
        let tapOff = writeTap ? UInt32(session.convVerifyOffsets[convSlot] / 2) : 0
        let tgW = min(256, kvDim)
        enc.setBuffer(projOutBuf, offset: projOutOff, index: 0)
        enc.setBuffer(staticW.buffer, offset: staticW.offset, index: 1)
        enc.setBuffer(session.dynWF32Buf, offset: dynWOff, index: 2)
        enc.setBuffer(slab, offset: stOff, index: 3)
        enc.setBuffer(outConv, offset: outConvOff, index: 4)
        enc.setBuffer(session.convVerifyInputsBuf ?? session.dummyB, offset: 0, index: 5)
        if let treePlan {
            var tp = Ink2SConvTreeTapP(
                T: UInt32(t), C: UInt32(kvDim), nKV: UInt32(nKV), kernelSize: 4,
                updateState: 0, resMode: 0,
                writeVerifyTap: writeTap ? 1 : 0, verifyTapOffset: tapOff,
                invSqrtDenom: 1.0 / 3.0,
                treeLags: packSConvTreeLags(treePlan.sconvLags)
            )
            enc.setComputePipelineState(pSConvDynTreeTap)
            enc.setBytes(&tp, length: MemoryLayout<Ink2SConvTreeTapP>.stride, index: 6)
            enc.dispatchThreads(MTLSize(width: kvDim, height: t, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
            return
        }
        var tp = Ink2SConvTapP(
            T: UInt32(t), C: UInt32(kvDim), nKV: UInt32(nKV), kernelSize: 4,
            updateState: updateState ? 1 : 0, resMode: 0,
            writeVerifyTap: writeTap ? 1 : 0, verifyTapOffset: tapOff,
            invSqrtDenom: 1.0 / 3.0
        )
        enc.setComputePipelineState(pSConvDynTap)
        enc.setBytes(&tp, length: MemoryLayout<Ink2SConvTapP>.stride, index: 6)
        enc.dispatchThreads(MTLSize(width: kvDim, height: t, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
        if t > 1 && updateState {
            encodeSConvCommit(enc, xBF16: projOutBuf, xOff: projOutOff, convSlab: slab, stOff: stOff,
                              t: t, c: kvDim, nKV: nKV, resMode: 0, invSqrt: 1.0 / 3.0)
        }
    }

    func encodeFusedStaticSConvQKVR(
        _ enc: any MTLComputeCommandEncoder,
        lw: QWInkling2Weights.LayerWeights,
        kv: Ink2VisitKV,
        proj: Ink2AttnProjViews,
        kcW: Ink2Slice,
        vcW: Ink2Slice,
        convSlotBase: Int,
        convSlab: (any MTLBuffer)? = nil,
        qPrepOff: Int = 0,
        rPrepOff: Int = 0,
        kConvOff: Int = 0,
        t: Int,
        posOffset: Int,
        kvInRowStride: Int,
        updateState: Bool,
        recordVerifyConv: Bool,
        writeKConv: Bool = false,
        treePlan: Ink2TreeVerifyPlan? = nil
    ) {
        let d = session.descriptor, ink = session.ink2
        let nH = d.qHeads, nKV = d.kvHeads, headDim = d.headDim, kvDim = nKV * headDim
        let slab = convSlab ?? session.convStatesSlab
        let kStOff = session.convStateOffsets[convSlotBase]
        let vStOff = session.convStateOffsets[convSlotBase + 1]
        let writeTap = recordVerifyConv && session.convVerifyInputsBuf != nil
        let kTapOff = writeTap ? UInt32(session.convVerifyOffsets[convSlotBase] / 2) : 0
        let vTapOff = writeTap ? UInt32(session.convVerifyOffsets[convSlotBase + 1] / 2) : 0
        let invSqrt = Float(1.0 / 5.0.squareRoot())
        enc.setBuffer(proj.qBuf, offset: proj.qOff, index: 0)
        enc.setBuffer(proj.kBuf, offset: proj.kOff, index: 1)
        enc.setBuffer(proj.vBuf, offset: proj.vOff, index: 2)
        enc.setBuffer(proj.rBuf, offset: proj.rOff, index: 3)
        enc.setBuffer(kcW.buffer, offset: kcW.offset, index: 4)
        enc.setBuffer(vcW.buffer, offset: vcW.offset, index: 5)
        enc.setBuffer(slab, offset: kStOff, index: 6)
        enc.setBuffer(slab, offset: vStOff, index: 7)
        enc.setBuffer(lw.qScale.buffer, offset: lw.qScale.offset, index: 8)
        enc.setBuffer(lw.relScale.buffer, offset: lw.relScale.offset, index: 9)
        enc.setBuffer(session.qPrepBuf, offset: qPrepOff, index: 10)
        enc.setBuffer(session.rPrepBuf, offset: rPrepOff, index: 11)
        enc.setBuffer(kv.kCache, offset: 0, index: 12)
        enc.setBuffer(kv.vCache, offset: 0, index: 13)
        enc.setBuffer(kv.kMeta, offset: 0, index: 14)
        enc.setBuffer(kv.vMeta, offset: 0, index: 15)
        enc.setBuffer(session.convVerifyInputsBuf ?? session.dummyB, offset: 0, index: 16)
        enc.setBuffer(session.kConvBuf, offset: kConvOff, index: 17)
        var fp = Ink2QKVRSConvFusedP(
            T: UInt32(t), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(headDim), dRel: UInt32(ink.dRel),
            isSliding: lw.isSliding ? 1 : 0, posOffset: UInt32(posOffset), ringCap: UInt32(kv.ringCap),
            kvMode: session.kvModeRaw, nFloor: UInt32(ink.logScalingNFloor),
            qkClip: ink.qkL2ClipThreshold, relClip: ink.relL2ClipThreshold,
            logAlpha: ink.logScalingAlpha, rmsEps: d.rmsEps,
            kvInRowStride: UInt32(kvInRowStride), updateState: (treePlan == nil && updateState) ? 1 : 0,
            writeVerifyTap: writeTap ? 1 : 0, kVerifyTapOffset: kTapOff, vVerifyTapOffset: vTapOff,
            writeKConv: writeKConv ? 1 : 0, invSqrtDenom: invSqrt
        )
        if let treePlan {
            func depth(_ i: Int) -> UInt32 { i < treePlan.depths.count ? treePlan.depths[i] : 0 }
            var tp = Ink2QKVRSConvTreeFusedP(
                base: fp,
                depths: (
                    depth(0), depth(1), depth(2), depth(3), depth(4), depth(5), depth(6), depth(7),
                    depth(8), depth(9), depth(10), depth(11), depth(12), depth(13), depth(14), depth(15)
                ),
                treeLags: packSConvTreeLags(treePlan.sconvLags)
            )
            enc.setComputePipelineState(pQKVRSConvStaticTreeFused)
            enc.setBytes(&tp, length: MemoryLayout<Ink2QKVRSConvTreeFusedP>.stride, index: 18)
            enc.dispatchThreadgroups(MTLSize(width: max(nH, nKV), height: t, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(pQKVRSConvStaticFused)
        enc.setBytes(&fp, length: MemoryLayout<Ink2QKVRSConvFusedP>.stride, index: 18)
        enc.dispatchThreadgroups(MTLSize(width: max(nH, nKV), height: t, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if t > 1 && updateState {
            var cp = Ink2SConvCommitPairP(T: UInt32(t), C: UInt32(kvDim), kvInRowStride: UInt32(kvInRowStride))
            enc.setComputePipelineState(pSConvCommitKVPair)
            enc.setBuffer(proj.kBuf, offset: proj.kOff, index: 0)
            enc.setBuffer(proj.vBuf, offset: proj.vOff, index: 1)
            enc.setBuffer(slab, offset: kStOff, index: 2)
            enc.setBuffer(slab, offset: vStOff, index: 3)
            enc.setBytes(&cp, length: MemoryLayout<Ink2SConvCommitPairP>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: kvDim, height: 2, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, kvDim), height: 1, depth: 1))
        }
    }

    func encodeKVSConvAndQKVR(
        lw: QWInkling2Weights.LayerWeights,
        kv: Ink2VisitKV,
        proj: Ink2AttnProjViews,
        convSlotBase: Int,
        convSlab: (any MTLBuffer)? = nil,
        kNormOff: Int,
        vNormOff: Int,
        qPrepOff: Int = 0,
        rPrepOff: Int = 0,
        kConvOff: Int = 0,
        vConvOff: Int = 0,
        dynWOff: Int = 0,
        kvInRowStride: Int? = nil,
        t: Int,
        posOffset: Int,
        updateState: Bool,
        recordVerifyConv: Bool,
        writeKConv: Bool,
        treePlan: Ink2TreeVerifyPlan? = nil,
        beginEncoder: (Ink2Stage) throws -> any MTLComputeCommandEncoder
    ) throws {
        let d = session.descriptor, ink = session.ink2
        let nH = d.qHeads, nKV = d.kvHeads, headDim = d.headDim, kvDim = nKV * headDim
        switch lw.kvSConv {
        case .staticConv(let kcW, let vcW):
            let qkvrEnc = try beginEncoder(.qkvrKVWrite)
            let stride = kvInRowStride ?? ((proj.kBuf === session.fusedAttnProjBuf) ? lw.projLayout.totalRows : kvDim)
            encodeFusedStaticSConvQKVR(
                qkvrEnc, lw: lw, kv: kv, proj: proj, kcW: kcW, vcW: vcW,
                convSlotBase: convSlotBase, convSlab: convSlab,
                qPrepOff: qPrepOff, rPrepOff: rPrepOff, kConvOff: kConvOff,
                t: t, posOffset: posOffset, kvInRowStride: stride,
                updateState: updateState, recordVerifyConv: recordVerifyConv,
                writeKConv: writeKConv, treePlan: treePlan
            )
        case .dynamicConv(let ksW, let kpW, let vsW, let vpW):
            let convEnc = try beginEncoder(.sconv)
            try encodeSConvDynamicTap(
                convEnc, projNormOff: kNormOff, projOutBuf: proj.kBuf, projOutOff: proj.kOff,
                staticW: ksW, projW: kpW, convSlot: convSlotBase,
                convSlab: convSlab, dynWOff: dynWOff,
                outConv: session.kConvBuf, outConvOff: kConvOff,
                t: t, kvDim: kvDim, nKV: nKV, updateState: updateState,
                recordVerifyConv: recordVerifyConv, treePlan: treePlan
            )
            try encodeSConvDynamicTap(
                convEnc, projNormOff: vNormOff, projOutBuf: proj.vBuf, projOutOff: proj.vOff,
                staticW: vsW, projW: vpW, convSlot: convSlotBase + 1,
                convSlab: convSlab, dynWOff: dynWOff,
                outConv: session.vConvBuf, outConvOff: vConvOff,
                t: t, kvDim: kvDim, nKV: nKV, updateState: updateState,
                recordVerifyConv: recordVerifyConv, treePlan: treePlan
            )
            var qp = QKVRP(
                T: UInt32(t), nH: UInt32(nH), nKV: UInt32(nKV), headDim: UInt32(headDim), dRel: UInt32(ink.dRel),
                isSliding: lw.isSliding ? 1 : 0, posOff: UInt32(posOffset), ringCap: UInt32(kv.ringCap),
                kvMode: session.kvModeRaw, nFloor: UInt32(ink.logScalingNFloor),
                qkClip: ink.qkL2ClipThreshold, relClip: ink.relL2ClipThreshold,
                logAlpha: ink.logScalingAlpha, rmsEps: d.rmsEps
            )
            let qkvrEnc = try beginEncoder(.qkvrKVWrite)
            qkvrEnc.setComputePipelineState(treePlan != nil ? pQKVRTree : pQKVR)
            qkvrEnc.setBuffer(proj.qBuf, offset: proj.qOff, index: 0)
            qkvrEnc.setBuffer(session.kConvBuf, offset: kConvOff, index: 1)
            qkvrEnc.setBuffer(session.vConvBuf, offset: vConvOff, index: 2)
            qkvrEnc.setBuffer(proj.rBuf, offset: proj.rOff, index: 3)
            qkvrEnc.setBuffer(lw.qScale.buffer, offset: lw.qScale.offset, index: 4)
            qkvrEnc.setBuffer(lw.relScale.buffer, offset: lw.relScale.offset, index: 5)
            qkvrEnc.setBuffer(session.qPrepBuf, offset: qPrepOff, index: 6)
            qkvrEnc.setBuffer(session.rPrepBuf, offset: rPrepOff, index: 7)
            qkvrEnc.setBuffer(kv.kCache, offset: 0, index: 8)
            qkvrEnc.setBuffer(kv.vCache, offset: 0, index: 9)
            qkvrEnc.setBuffer(kv.kMeta, offset: 0, index: 10)
            qkvrEnc.setBuffer(kv.vMeta, offset: 0, index: 11)
            qkvrEnc.setBytes(&qp, length: MemoryLayout<QKVRP>.stride, index: 12)
            if let treePlan {
                func d(_ i: Int) -> UInt32 { i < treePlan.depths.count ? treePlan.depths[i] : 0 }
                var tdp = TreeDepthsP(depths: (
                    d(0), d(1), d(2), d(3), d(4), d(5), d(6), d(7),
                    d(8), d(9), d(10), d(11), d(12), d(13), d(14), d(15)
                ))
                qkvrEnc.setBytes(&tdp, length: MemoryLayout<TreeDepthsP>.stride, index: 13)
            }
            qkvrEnc.dispatchThreadgroups(MTLSize(width: max(nH, nKV), height: t, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
    }
}
