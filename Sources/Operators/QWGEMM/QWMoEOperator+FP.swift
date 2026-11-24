// QWMoEOperator+FP.swift — Pure-FP and mixed-FP bank paths for QWMoEOperator.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

extension QWMoEOperator {
    /// Pure-FP Sparse-MoE forward pass (`BF16` router/shared + `NVFP4_G16`/`MXFP8_G32` experts).
    public static func encodeFPResident(
        _ enc: inout any MTLComputeCommandEncoder,
        hal: QWHal,
        weights w: FPWeights,
        scratch s: FPScratch,
        input x: QWFPBufferSlice,
        inDtype: QWFPElemDtype,
        out: QWFPBufferSlice,
        outDtype: QWFPElemDtype,
        rows T: Int,
        supportsGroupedNaxPrefill: Bool = true,
        concurrentBarriers: Bool = false,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)? = nil
    ) throws {
        guard T > 0 else { return }
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let H = w.hidden, E = w.experts, K = w.topK, I = w.intermediate, ISh = w.sharedIntermediate
        let routerCols = w.splitFeatureRouting ? (2 * E) : E
        guard w.router.n >= routerCols, w.router.k == H else {
            throw QWGemmError.badShape("FP router shape mismatch")
        }
        try QWGemmFP.encodeBF16(
            enc, hal: hal, weight: w.router, x: x, y: QWFPBufferSlice(s.logits),
            m: T, inDtype: inDtype, outDtype: .f32)
        if ISh > 0, let shDown = w.sharedDown, let guSh = s.guSh, let actSh = s.actSh,
           let shOut = s.shared, let shGateUp = w.sharedGateUp {
            if let shUp = w.sharedUp {
                let gSlice = QWFPBufferSlice(guSh)
                let uSlice = gSlice.slice(byteOffset: ISh * s.interDtype.byteSize)
                try QWGemmFP.encodeBF16(
                    enc, hal: hal, weight: shGateUp, x: x, y: gSlice,
                    m: T, inDtype: inDtype, outDtype: s.interDtype, strideY: 2 * ISh)
                try QWGemmFP.encodeBF16(
                    enc, hal: hal, weight: shUp, x: x, y: uSlice,
                    m: T, inDtype: inDtype, outDtype: s.interDtype, strideY: 2 * ISh)
            } else {
                try QWGemmFP.encodeBF16(
                    enc, hal: hal, weight: shGateUp, x: x, y: QWFPBufferSlice(guSh),
                    m: T, inDtype: inDtype, outDtype: s.interDtype)
            }
            barrier()
            try QWGemmFP.encodeSwiGLUSoftcap(
                enc, hal: hal, gu: QWFPBufferSlice(guSh), act: QWFPBufferSlice(actSh),
                rows: T, intermediate: ISh, gateCap: w.gateSoftcap, upCap: w.upSoftcap,
                inDtype: s.interDtype, outDtype: s.interDtype)
            barrier()
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: shDown, x: QWFPBufferSlice(actSh), y: QWFPBufferSlice(shOut),
                m: T, inDtype: s.interDtype, outDtype: s.interDtype)
        }
        barrier()
        try QWGemm.encodeFnMoeRoute(
            enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts,
            rows: T, experts: E, topK: K, normTopK: w.normTopK,
            bias: w.routerBias, expertBudget: w.moeSpecExpertBudget,
            splitFeature: w.splitFeatureRouting, routedScaleParam: w.routedScaleParam,
            useSigmoid: w.useSigmoidScore, nGroup: w.nGroup, topkGroup: w.topkGroup, groupTopM: w.groupTopM)
        barrier()
        let hasShared = ISh > 0 && s.shared != nil
        try QWGemm.encodeMoEBankRoutedCombine(
            &enc, hal: hal, x: x, xDtype: inDtype, interDtype: s.interDtype,
            gateUp: w.gateUp, up: w.up, down: w.down,
            idx: s.idx, wts: s.wts,
            shared: hasShared ? s.shared.map(QWFPBufferSlice.init) : nil,
            sharedDtype: s.interDtype,
            gate: hasShared ? s.gate.map(QWFPBufferSlice.init) : nil,
            out: out, outDtype: outDtype,
            expOff: s.expOff, sortedPair: s.sortedPair, sortedTok: s.sortedTok, invPerm: s.invPerm,
            tileE: s.tileE, tileR: s.tileR, tileCount: s.tileCount,
            activeExperts: s.activeExperts, activeIndirect: s.activeIndirect,
            guBuf: s.gu, actBuf: s.act, dnAccBuf: s.dnAcc,
            hidden: H, experts: E, topK: K, intermediate: I, rows: T,
            gateSoftcap: w.gateSoftcap, upSoftcap: w.upSoftcap,
            supportsGroupedNaxPrefill: supportsGroupedNaxPrefill,
            concurrentBarriers: concurrentBarriers, rotateEncoder: rotateEncoder)
        barrier()
    }

    static func encodeResidentWithFPBanks(
        _ enc: inout any MTLComputeCommandEncoder,
        hal: QWHal, weights w: Weights, scratch s: Scratch, input: Input,
        fpGateUp: QWMoEExpertBank, fpUp: QWMoEExpertBank?, fpDown: QWMoEExpertBank,
        out32: QWTensorValue, out16: QWTensorValue?, skipOut32: Bool, rows T: Int,
        supportsGroupedNaxPrefill: Bool, concurrentBarriers: Bool,
        rotateEncoder: ((inout any MTLComputeCommandEncoder) throws -> Void)?
    ) throws {
        guard let expOff = s.expOff, let sortedPair = s.sortedPair,
              let sortedTok = s.sortedTok, let invPerm = s.invPerm else {
            throw QWGemmError.badShape("FP MoE resident dispatch requires expOff, sortedPair, sortedTok, invPerm scratch")
        }
        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }
        let H = w.hidden, E = w.experts, K = w.topK, I = w.intermediate, ISh = w.sharedIntermediate
        let routerCols = w.splitFeatureRouting ? (2 * E) : E
        let needsRot = ISh > 0 && (w.sharedGateUp?.isRotated == true)
        let xPlain: QWTensorValue, xRot: QWTensorValue, xFP: QWFPBufferSlice, xFPDtype: QWFPElemDtype
        switch input {
        case .f16Normed(let x16):
            try QWGemm.encodeMoEBridgeF16ToF32(enc, hal: hal, x: x16, xRot32: s.xRot32,
                                               xPlain32: s.xPlain32, rows: T, hidden: H, skipRot32: !needsRot)
            xPlain = s.xPlain32; xRot = s.xRot32
            xFP = QWFPBufferSlice(x16); xFPDtype = .f16
        case .f32PreRotated(let plain, let rotated, let precast16):
            xPlain = plain; xRot = rotated
            if T > 8, let x16 = s.naxX16 {
                let needPlainCast = (precast16 == nil || needsRot)
                if needPlainCast { try QWGemm.encodeCast(enc, hal: hal, x32: plain, y16: x16, count: T * H); barrier() }
                xFP = QWFPBufferSlice(needPlainCast ? x16 : (precast16 ?? x16))
                xFPDtype = .f16
            } else {
                xFP = QWFPBufferSlice(plain); xFPDtype = .f32
            }
        }
        try QWGemm.encodeFnRouterGemm(
            enc, hal: hal, weight: w.router, x: xPlain, y: s.logits, rows: T, n: routerCols, k: H,
            shGateWeight: ISh > 0 ? w.sharedGateScalar : nil, shGateOut: ISh > 0 ? s.gate : nil)
        barrier()
        try QWGemm.encodeFnMoeRoute(
            enc, hal: hal, logits: s.logits, idx: s.idx, wts: s.wts, rows: T, experts: E, topK: K,
            normTopK: w.normTopK, bias: w.routerBias, expertBudget: w.moeSpecExpertBudget,
            splitFeature: w.splitFeatureRouting, routedScaleParam: w.routedScaleParam,
            useSigmoid: w.useSigmoidScore, nGroup: w.nGroup, topkGroup: w.topkGroup, groupTopM: w.groupTopM)
        if ISh > 0, let shGU = w.sharedGateUp, let shDn = w.sharedDown,
           let guSh = s.guSh32, let actSh = s.actSh32 {
            try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shGU, x: xRot, y: guSh,
                                          naxX16: s.naxX16 ?? guSh, naxY16: s.naxY16 ?? guSh,
                                          maxTileWidth: 0, rows: T)
            barrier()
            try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: guSh, act: actSh, rows: T,
                                        intermediate: ISh, swigluClamp: w.sharedSwigluClamp)
            barrier()
            try QWGemm.encodeDenseProjF32(enc, hal: hal, weight: shDn, x: actSh, y: s.shared,
                                          naxX16: s.naxX16 ?? actSh, naxY16: s.naxY16 ?? s.shared,
                                          maxTileWidth: 0, rows: T)
        }
        barrier()
        let guBuf = s.gu16 ?? s.gu32, actBuf = s.dnHi ?? s.act32
        let write16Only = skipOut32 && out16 != nil
        let targetOut = write16Only ? (out16 ?? out32) : out32
        let targetDtype: QWFPElemDtype = write16Only ? .f16 : .f32
        try QWGemm.encodeMoEBankRoutedCombine(
            &enc, hal: hal, x: xFP, xDtype: xFPDtype, interDtype: .f16,
            gateUp: fpGateUp, up: fpUp, down: fpDown, idx: s.idx, wts: s.wts,
            shared: ISh > 0 ? QWFPBufferSlice(s.shared) : nil, sharedDtype: .f32,
            gate: ISh > 0 ? QWFPBufferSlice(s.gate) : nil,
            out: QWFPBufferSlice(targetOut), outDtype: targetDtype,
            expOff: expOff, sortedPair: sortedPair, sortedTok: sortedTok, invPerm: invPerm,
            tileE: s.tileE, tileR: s.tileR, tileCount: s.tileCount,
            activeExperts: s.activeExperts, activeIndirect: s.activeIndirect,
            guBuf: guBuf, actBuf: actBuf, dnAccBuf: s.dnAcc,
            hidden: H, experts: E, topK: K, intermediate: I, rows: T,
            gateSoftcap: w.gateSoftcap, upSoftcap: w.upSoftcap,
            supportsGroupedNaxPrefill: supportsGroupedNaxPrefill,
            concurrentBarriers: concurrentBarriers, rotateEncoder: rotateEncoder)
        if !write16Only, let y16 = out16 {
            barrier()
            try QWGemm.encodeCast(enc, hal: hal, x32: out32, y16: y16, count: T * H)
        }
        barrier()
    }
}
