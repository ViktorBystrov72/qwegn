// QWHybridLayerKernels.swift — GDN linear-attention and full-attention layer kernels.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWQuant

/// The GDN layer: linear attention with a gated norm.
final class GDNLayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .dense, ple: false)

    static func make() -> any QWLayerKernel { GDNLayerKernel() }

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) { driver.profileStage(stage); if let a = driver.activeEnc { enc = a } }
        let sc = driver.sc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.gdn!
        let H = geometry.hidden
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = sc.fgb16,
           let fB = w.fB, let gB = w.gB {
            if context.weights.mhc == nil { mark("kda.in_proj") }
            try driver.gemvH16T4(enc, n: geometry.gdnQKVChannels, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: w.inProj, y16: sc.qkv16)
            mark("kda.fgb_proj")
            try driver.gemvH16T4(enc, n: fgbProj.n, k: H, x: sc.ln1, gs: sc.gs,
                                 wQ4: fgbProj, y16: fgb16)
            mark("kda.gates")
            try driver.gdnDecode.encodeKDAGates(
                enc, fgb: fgb16, fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                betas: sc.betas, glog: sc.glog, zOut: sc.z16, rows: 1)
        } else {
            if context.weights.mhc == nil { mark("gdn.in_proj") }
            try driver.gemvH16GdnInProjT4(enc, n: geometry.gdnTotalN, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.inProj,
                                   y16: sc.gdnY16, y32: sc.gdnY,
                                   aLog: w.aLog, dtBias: w.dtBias, betas: sc.betas, glog: sc.glog)
        }
        mark("gdn.conv")
        // C11 conv + headnorm fused for T=1:
        let gi = driver.gdnIndexMap[l]
        try driver.gdnConv.encodeT1WithHeadnorm(enc, x: sc.qkv16, w: w.conv, bias: nil,
                                                 stIn: context.session.ring[gi], y: sc.convY,
                                                 stOut: context.session.ring[gi])
        mark("gdn.recurrent")
        // C12 recurrent + gated norm
        try driver.gdnDecode.encode(enc, y: sc.convY, betas: sc.betas, glog: sc.glog,
                                     state: context.session.rec[gi], z: sc.z16, w: w.normW,
                                     o: sc.oOut, oScratch: sc.oScr,
                                     outGroupSum: nil)
        mark("gdn.out_proj")
        // out_proj (K = zC)
        try driver.gemvH16T4(enc, n: H, k: geometry.gdnZChannels, x: sc.oOut, gs: sc.gs48,
                      wQ4: w.out, y16: sc.o16)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) {
            driver.profileStage(stage)
            if let a = driver.activeEnc { enc = a }
        }
        let psc = driver.psc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.gdn!
        let T = context.rows
        let pos = context.position
        let H = geometry.hidden
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels
        let gdnVH = geometry.gdnVHeads
        let glogC = geometry.gdnGlogChannels
        let gi = driver.gdnIndexMap[l]   // M6 FIX: single source of truth
        if geometry.gdnVectorDecay, let fgbProj = w.fgbProj, let fgb16 = psc.fgb16,
           let fB = w.fB, let gB = w.gB {
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            mark("kda.in_proj")
            driver.gemmBatchedH16(enc, m: T, n: qkvC, k: H, x: activeLn1, wQ4: w.inProj,
                                  y16: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16))
            mark("kda.fgb_proj")
            driver.gemmBatchedH16(enc, m: T, n: fgbProj.n, k: H, x: activeLn1, wQ4: fgbProj,
                                  y16: driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16))
            mark("kda.gates")
            try driver.gdnDecode.encodeKDAGates(
                enc,
                fgb: driver.prefixRows(fgb16, rows: T, cols: fgbProj.n, dt: .f16),
                fB: fB, gB: gB, aLog: w.aLog, dtBias: w.dtBias,
                betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                zOut: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                rows: T)
            mark("kda.conv")
            try driver.gdnConv.encodePrefillWithHeadnorm(
                enc,
                x: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16),
                w: w.conv, bias: nil,
                stIn: pos == 0 ? driver.gdnConv.zeroRing : context.session.ring[gi],
                y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                stOut: context.session.ring[gi],
                T: T)
        } else {
            if context.weights.mhc == nil { mark("gdn.in_proj") }
            driver.gemmBatchedH16GdnInProj(enc, m: T, n: geometry.gdnTotalN, k: H,
                                    x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                                    wQ4: w.inProj,
                                    qkv16: psc.qkv16, z16: psc.z16, y32: psc.gdnY)
            if context.weights.mhc == nil { mark("gdn.conv") }
            try driver.gdnConv.encodePrefillWithHeadnorm(enc,
                                                          x: driver.prefixRows(psc.qkv16, rows: T, cols: qkvC, dt: .f16),
                                                          w: w.conv, bias: nil,
                                                          stIn: pos == 0 ? driver.gdnConv.zeroRing : context.session.ring[gi],
                                                          y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                                          stOut: context.session.ring[gi],
                                                          T: T,
                                                          gdnY32: psc.gdnY, aLog: w.aLog, dtBias: w.dtBias,
                                                          betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                                          glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32))
        }
        mark("gdn.recurrence")
        switch context.mode {
        case .chunked:
            if T <= 4096 || geometry.gdnVectorDecay || psc.U.shape.elementCount < T * zC {
                try driver.gdnPrefill.encodeRegRes(enc,
                                                   y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                                   betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                                   glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                                   z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                                   w: w.normW,
                                                   state: context.session.rec[gi],
                                                   o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                                   oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                                   T: T)
            } else {
                try driver.gdnPrefill.encode(enc,
                                             y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                             betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                             glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                             z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                             w: w.normW,
                                             state: context.session.rec[gi],
                                             o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                             U: driver.prefixRows(psc.U, rows: T, cols: zC, dt: .f32),
                                             W: driver.prefixRows(psc.W, rows: T, cols: zC, dt: .f32),
                                             A: driver.prefixRows(psc.A, rows: T, cols: zC, dt: .f32),
                                             B: driver.prefixRows(psc.B, rows: T, cols: zC, dt: .f32),
                                             oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                             T: T, L: min(64, T),
                                             solveBackend: driver.gdnSolveBackend)
            }
        case .naive:
            for t in 0..<T {
                let yRow = driver.stridedRow(psc.convY, row: t, cols: qkvC, dt: .f16)
                let bRow = driver.stridedRow(psc.betas, row: t, cols: gdnVH, dt: .f32)
                let gRow = driver.stridedRow(psc.glog, row: t, cols: glogC, dt: .f32)
                let zRow = driver.stridedRow(psc.z16, row: t, cols: zC, dt: .f16)
                let oRow = driver.stridedRow(psc.oOut, row: t, cols: zC, dt: .f16)
                let sRow = driver.stridedRow(psc.oScr, row: t, cols: zC, dt: .f32)
                try driver.gdnDecode.encode(enc, y: yRow, betas: bRow, glog: gRow,
                                            state: context.session.rec[gi], z: zRow, w: w.normW,
                                            o: oRow, oScratch: sRow)
            }
        case .regres:
            // EXP-9 sub-item 1: 192-TG zero-barrier register-resident
            // recurrence + qwg_gdn_prefill_norm.
            try driver.gdnPrefill.encodeRegRes(enc,
                                               y: driver.prefixRows(psc.convY, rows: T, cols: qkvC, dt: .f16),
                                               betas: driver.prefixRows(psc.betas, rows: T, cols: gdnVH, dt: .f32),
                                               glog: driver.prefixRows(psc.glog, rows: T, cols: glogC, dt: .f32),
                                               z: driver.prefixRows(psc.z16, rows: T, cols: zC, dt: .f16),
                                               w: w.normW,
                                               state: context.session.rec[gi],
                                               o: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                                               oScratch: driver.prefixRows(psc.oScr, rows: T, cols: zC, dt: .f32),
                                               T: T)
        }
        mark("gdn.out_proj")
        driver.gemmBatchedH16(enc, m: T, n: H, k: zC,
                       x: driver.prefixRows(psc.oOut, rows: T, cols: zC, dt: .f16),
                       wQ4: w.out,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }
}

/// The full-attention layer: paged KV attention with rope.
final class FullAttentionLayerKernel: DenseLayerKernel, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .dense, ple: false)

    static func make() -> any QWLayerKernel { FullAttentionLayerKernel() }

    override func encodeAttention(_ context: QWLayerDecodeContext) throws {
        let driver = context.driver
        var enc = context.enc
        func mark(_ stage: String) { driver.profileStage(stage); if let a = driver.activeEnc { enc = a } }
        let sc = driver.sc
        let geometry = driver.geometry
        let l = context.layer
        let pos = context.position
        let w = context.weights.attn!
        let H = geometry.hidden
        let fullAttnIdx = driver.attnIndexMap[l]
        let views = context.session.attnKVViews[fullAttnIdx]
        if let mla = driver.mlaOperator, let mlaW = w.mla {
            let latRow = driver.prefixRows(mla.mlaLatentScratch, rows: 1, cols: mla.latentStride, dt: .f16)
            let qResRow = driver.prefixRows(mla.qResidScratch, rows: 1, cols: geometry.qLoraRank, dt: .f16)
            let qFullRow = driver.prefixRows(mla.mlaQFullScratch, rows: 1, cols: mla.qFullStride, dt: .f16)
            mark("mla.qkv_a_proj")
            try driver.gemvH16T4(enc, n: w.qkvProj.n, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: latRow)
            mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: sc.ln1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
            mark("mla.q_b_proj")
            try driver.gemvH16T4(enc, n: mlaW.qBProj.n, k: geometry.qLoraRank, x: qResRow, gs: sc.gs, wQ4: mlaW.qBProj, y16: qFullRow)
            mark("mla.attend")
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: sc.attOut,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1,
                stage: QWEngineDriver.decodeProfileEnabled
                    ? { name in driver.profileStage(name); return driver.activeEnc } : nil)
            mark("mla.o_proj")
            try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.vHeadDim,
                          x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
            return
        }
        if context.weights.mhc == nil {
            mark("attn.gqa")
        }
        try driver.gemvH16T4(enc, n: geometry.attnTotalN, k: H, x: sc.ln1, gs: sc.gs, wQ4: w.qkvProj, y16: sc.attnY16)
        // C07 fused per-head norm + partial rope (position [pos]).
        try driver.ropeApply.encodeQK(enc, hal: driver.hal, q: sc.q16, k: sc.k16,
                                      wq: w.wq, wk: w.wkTiled,
                                      qOut: sc.qR, kOut: sc.kR,
                                      positionStart: pos, count: 1)
        // C09 split-K decode: attends the new row from knew/vnew AND
        // materialises it into the page slot in the same launch.
        let attnDecode = driver.attnDecode!
        if let lsa = driver.lsaOperator {
            lsa.encodeIndexAndCluster(enc, qgate: sc.qR, kIn: sc.kR,
                                      fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
            if pos >= driver.lsaConfig.denseBypassThreshold {
                lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: 1)
                let activeKeys = min(pos + 1, driver.lsaConfig.alignedBudget + 640)
                try attnDecode.encodeSparse(
                    enc, qgate: sc.qR, kv: views,
                    knew: sc.kR, vnew: sc.v16,
                    partials: sc.partials, out: sc.attOut,
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    C: pos, splits: QWAttnDecode.splits(forKeys: activeKeys),
                    listStride: lsa.listStride)
            } else {
                try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                      knew: sc.kR, vnew: sc.v16,
                                      partials: sc.partials, out: sc.attOut,
                                      C: pos, splits: QWAttnDecode.splits(forKeys: pos + 1))
            }
        } else {
            try attnDecode.encode(enc, qgate: sc.qR, kv: views,
                                  knew: sc.kR, vnew: sc.v16,
                                  partials: sc.partials, out: sc.attOut,
                                  C: pos, splits: QWAttnDecode.splits(forKeys: pos + 1))
        }
        // o_proj (K = qHeads·headDim = 6144 — the C08 law, NOT 5120)
        try driver.gemvH16T4(enc, n: H, k: geometry.qHeads * geometry.headDim,
                      x: sc.attOut, gs: sc.gs48, wQ4: w.o, y16: sc.o16)
    }

    override func encodeAttentionPrefill(_ context: QWLayerPrefillContext) throws {
        let driver = context.driver
        var enc = context.enc
        let psc = driver.psc
        let geometry = driver.geometry
        let l = context.layer
        let w = context.weights.attn!
        let T = context.rows
        let pos = context.position
        let H = geometry.hidden
        let qOut = geometry.qProjOut, qC = geometry.qChannels, kvC = geometry.kvChannels
        let fullAttnIdx = driver.attnIndexMap[l]
        let views = context.session.attnKVViews[fullAttnIdx]
        if let mla = driver.mlaOperator, let mlaW = w.mla {
            let vCols = w.o.k
            let activeLn1 = driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16)
            let latRows = driver.prefixRows(mla.mlaLatentScratch, rows: T, cols: mla.latentStride, dt: .f16)
            let qResRows = driver.prefixRows(mla.qResidScratch, rows: T, cols: geometry.qLoraRank, dt: .f16)
            let qFullRows = driver.prefixRows(mla.mlaQFullScratch, rows: T, cols: mla.qFullStride, dt: .f16)
            let attOutRows = driver.prefixRows(psc.attOut, rows: T, cols: vCols, dt: .f16)
            var enc = enc
            func mark(_ stage: String) -> (any MTLComputeCommandEncoder)? {
                driver.profileStage(stage)
                if let a = driver.activeEnc { enc = a }
                return driver.activeEnc
            }
            _ = mark("mla.qkv_a_proj")
            driver.gemmBatchedH16(enc, m: T, n: w.qkvProj.n, k: H, x: activeLn1, wQ4: w.qkvProj, y16: latRows)
            _ = mark("mla.latent_prep")
            mla.encodeLatentPrepAndPool(
                enc, ln1: activeLn1,
                mlaW: mlaW, kv: views,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T)
            _ = mark("mla.q_b_proj")
            driver.gemmBatchedH16(enc, m: T, n: mlaW.qBProj.n, k: geometry.qLoraRank, x: qResRows, wQ4: mlaW.qBProj, y16: qFullRows)
            mla.encodeScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: views,
                attOut: attOutRows,
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T,
                stage: QWEngineDriver.prefillProfileEnabled ? mark : nil)
            _ = mark("mla.o_proj")
            enc = driver.activeEnc ?? enc
            driver.gemmBatchedH16(enc, m: T, n: H, k: vCols,
                           x: attOutRows,
                           wQ4: w.o,
                           y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
            return
        }
        if context.weights.mhc == nil {
            driver.profileStage("attn.gqa")
            enc = driver.activeEnc ?? enc
        }
        driver.gemmBatchedH16(enc, m: T, n: geometry.attnTotalN, k: H,
                       x: driver.prefixRows(psc.ln1, rows: T, cols: H, dt: .f16),
                       wQ4: w.qkvProj,
                       y16: driver.prefixRows(psc.attnY16, rows: T, cols: geometry.attnTotalN, dt: .f16))
        try driver.ropeApply.encodeExtractQK(enc, hal: driver.hal,
                                              attnY: driver.prefixRows(psc.attnY16, rows: T, cols: geometry.attnTotalN, dt: .f16),
                                              qOut: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                                              kOut: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                                              vOut: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                                              wq: w.wq, wk: w.wkTiled,
                                              positionStart: pos, count: T)
        if let lsa = driver.lsaOperator {
            lsa.encodeIndexAndCluster(
                enc,
                qgate: driver.prefixRows(psc.attQR16, rows: T, cols: qOut, dt: .f16),
                kIn: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                fullAttnLayer: fullAttnIdx, pos0: pos, rows: T)
        }
        // Pad to the SELECTED prefill backend's tile (32 rows, 64 for
        // the quad kernel) — a hardcoded 32 crashed quad at S=32. Only the
        // 27B's NAX prefill reads whole tiles; the standard path does not.
        let padTile = driver.attnPrefill.tileRows
        let sp = (T + padTile - 1) / padTile * padTile
        if sp > T && QWEngineDriver.zeroPadServes(geometry: geometry) {
            try driver.encodeZeroPadAttn(enc, q: psc.attQR16, k: psc.attKR16, v: psc.attV16, t: T, sp: sp)
        }
        if let lsa = driver.lsaOperator, pos + T > driver.lsaConfig.denseBypassThreshold {
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            let useNAX = driver.geometry.supportsNAXPagedAttnPrefill
            lsa.encodeSelectBlocks(enc, fullAttnLayer: fullAttnIdx, pos0: pos, rows: T, tiled32: useNAX)
            if !useNAX {
                try driver.pagedAttnPrefillGeneral.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    pos: pos, T: T, listStride: lsa.listStride)
            } else {
                try driver.pagedAttnPrefill.encodeSparse(
                    enc,
                    qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                    kv: views,
                    out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                    blockList: lsa.blockListScratch,
                    blockCount: lsa.blockCountScratch,
                    pos: pos, T: T, listStride: lsa.listStride)
            }
        } else if !driver.geometry.supportsNAXPagedAttnPrefill {
            // General scalar prefill fallback for unsupported attention shapes.
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            try driver.pagedAttnPrefillGeneral.encode(enc,
                                               qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                               kv: views,
                                               out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                               pos: pos, T: T)
        } else if pos == 0 && T < 128 && driver.geometry.is27BShape {
            try driver.attnPrefill.encode(enc,
                                          qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                          k: driver.prefixRows(psc.attKR16, rows: sp, cols: kvC, dt: .f16),
                                          v: driver.prefixRows(psc.attV16, rows: sp, cols: kvC, dt: .f16),
                                          out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                          S: T)
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
        } else {
            // NAX paged prefill: write KV pool, then attend over all cached tokens.
            driver.encodePagedKVWrite(enc, kv: views,
                               k: driver.prefixRows(psc.attKR16, rows: T, cols: kvC, dt: .f16),
                               v: driver.prefixRows(psc.attV16, rows: T, cols: kvC, dt: .f16),
                               startPos: pos, t: T)
            try driver.pagedAttnPrefill.encode(enc,
                                               qgate: driver.prefixRows(psc.attQR16, rows: sp, cols: qOut, dt: .f16),
                                               kv: views,
                                               out: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                                               pos: pos, T: T)
        }
        driver.gemmBatchedH16(enc, m: T, n: H, k: qC,
                       x: driver.prefixRows(psc.attOut, rows: T, cols: qC, dt: .f16),
                       wQ4: w.o,
                       y16: driver.prefixRows(psc.o16, rows: T, cols: H, dt: .f16))
    }
}
