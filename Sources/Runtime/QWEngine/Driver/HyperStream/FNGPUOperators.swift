// FNGPUOperators.swift — Unified Operator Seams (QWPLEOperator, QWHCOperator, QWQSAOperator).

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWGDN
import QWHAL
import QWHyperPLE

/// Unified Per-Layer N-Gram Embedding (PLE) operator seam.
public enum QWPLEOperator {
    public static func encode(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        p: FNGPUPLE,
        s: FNGPUScratch,
        stream: QWTensorValue,
        out: QWTensorValue,
        embed: QWTensorValue? = nil,
        inInject: (base: QWTensorValue, src: QWTensorValue, scale: QWTensorValue)? = nil
    ) throws {
        try FNGPULayerPasses.ple(enc, hal: hal, p: p, s: s, stream: stream, out: out, embed: embed, inInject: inInject)
    }

    public static func stageTargets(
        _ targets: [(ple: FNNgramEmbedding, addr: Int)],
        history: [Int64],
        startPos: Int,
        count: Int
    ) {
        let numPLE = targets.count
        guard numPLE > 0 && count > 0 else { return }
        let hc = targets[0].ple.heads.sizes.count
        let headsPerPLE = count * hc
        DispatchQueue.concurrentPerform(iterations: numPLE * headsPerPLE) { idx in
            let pIdx = idx / headsPerPLE
            let rem = idx - pIdx * headsPerPLE
            let t = rem / hc
            let head = rem - t * hc
            let target = targets[pIdx]
            let dst = UnsafeMutablePointer<Float>(bitPattern: target.addr)!
            target.ple.embedTokenHead(into: dst, tokens: history, startPos: startPos, t: t, head: head)
        }
    }

    public static func stageBatch(
        targets: [(ple: FNNgramEmbedding, addr: Int)],
        histories: [[Int64]],
        positions: [Int],
        parallelHeads: Bool
    ) {
        let batch = histories.count
        precondition(batch > 0 && positions.count == batch && !targets.isEmpty)
        let heads = targets[0].ple.heads.sizes.count
        if parallelHeads && heads > 0 && targets.allSatisfy({ $0.ple.heads.sizes.count == heads }) {
            let headsPerTarget = batch * heads
            DispatchQueue.concurrentPerform(iterations: targets.count * headsPerTarget) { index in
                let target = targets[index / headsPerTarget]
                let local = index % headsPerTarget
                let row = local / heads
                let head = local % heads
                let dst = UnsafeMutablePointer<Float>(bitPattern: target.addr)!
                target.ple.embedTokenHead(into: dst, tokens: histories[row],
                                          startPos: positions[row] - row, t: row, head: head)
            }
        } else {
            DispatchQueue.concurrentPerform(iterations: targets.count * batch) { index in
                let target = targets[index / batch]
                let row = index % batch
                let dst = UnsafeMutablePointer<Float>(bitPattern: target.addr)!
                target.ple.embedToken(into: dst, tokens: histories[row],
                                      startPos: positions[row] - row, t: row)
            }
        }
    }

    public static func writeRowIds(
        _ targets: [FNNgramEmbedding],
        into dst: UnsafeMutablePointer<UInt32>,
        history: [Int64],
        startPos: Int,
        count: Int
    ) {
        let numPLE = targets.count
        guard numPLE > 0 && count > 0 else { return }
        let hc = targets[0].heads.sizes.count
        let stride = count * hc
        if numPLE * count >= 64 {
            let rawAddr = Int(bitPattern: dst)
            DispatchQueue.concurrentPerform(iterations: numPLE * count) { idx in
                let pIdx = idx / count
                let t = idx - pIdx * count
                let base = UnsafeMutablePointer<UInt32>(bitPattern: rawAddr + pIdx * stride * 4)!
                targets[pIdx].writeTokenRowIds(into: base, tokens: history, startPos: startPos, t: t)
            }
        } else {
            for pIdx in 0..<numPLE {
                let base = dst.advanced(by: pIdx * stride)
                for t in 0..<count {
                    targets[pIdx].writeTokenRowIds(into: base, tokens: history, startPos: startPos, t: t)
                }
            }
        }
    }

    public static func writeBatchRowIds(
        _ targets: [FNNgramEmbedding],
        into dst: UnsafeMutablePointer<UInt32>,
        histories: [[Int64]],
        positions: [Int]
    ) {
        let numPLE = targets.count, batch = histories.count
        guard numPLE > 0 && batch > 0 else { return }
        let hc = targets[0].heads.sizes.count
        let stride = batch * hc
        for pIdx in 0..<numPLE {
            let base = dst.advanced(by: pIdx * stride)
            for row in 0..<batch {
                targets[pIdx].writeTokenRowIds(into: base, tokens: histories[row],
                                               startPos: positions[row] - row, t: row)
            }
        }
    }
}

/// Unified Hyper-Connection (`hc`-stream) carrier operator seam.
public enum QWHCOperator {
    public static func encodeMix(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        hc: FNGPUHC,
        s: FNGPUScratch,
        stream: QWTensorValue,
        mixed: QWTensorValue,
        inj: QWTensorValue,
        rotMixed: QWTensorValue? = nil,
        injectBase: QWTensorValue? = nil,
        injectSrc: QWTensorValue? = nil,
        injectScale: QWTensorValue? = nil,
        rows: Int? = nil
    ) throws {
        try FNGPULayerPasses.hcMix(
            enc, hal: hal, hc: hc, s: s, stream: stream, mixed: mixed, inj: inj,
            rotMixed: rotMixed, injectBase: injectBase, injectSrc: injectSrc,
            injectScale: injectScale, rows: rows
        )
    }

    public static func encodeInject(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        base: QWTensorValue,
        src: QWTensorValue,
        scale: QWTensorValue,
        out: QWTensorValue,
        rows: Int,
        hidden: Int,
        hc: Int
    ) throws {
        try QWGemm.encodeFnInject(enc, hal: hal, base: base, src: src, inj: scale,
                                  out: out, rows: rows, hidden: hidden, hc: hc)
        enc.memoryBarrier(scope: .buffers)
    }
}

/// Unified Query-Sparse Attention (QSA) + LSA sequence-mixer operator seam.
public enum QWQSAOperator {
    public static func encodeDecode(
        _ enc: any MTLComputeCommandEncoder,
        ctx: FNGPUContext,
        a: FNGPUAttnMixer,
        c: FNGPUAttnDecodeState,
        mixed: QWTensorValue,
        pos: Int,
        out: QWTensorValue,
        row: Int = 0
    ) throws {
        try FNGPULayerPasses.attentionDecode(enc, ctx: ctx, a: a, c: c, mixed: mixed, pos: pos, out: out, row: row)
    }

    public static func encodeDecodeRows(
        _ enc: any MTLComputeCommandEncoder,
        ctx: FNGPUContext,
        a: FNGPUAttnMixer,
        c: FNGPUAttnDecodeState,
        pos: Int,
        out: QWTensorValue,
        rows: Int
    ) throws {
        try FNGPULayerPasses.attentionDecodeRows(enc, ctx: ctx, a: a, c: c, pos: pos, out: out, rows: rows)
    }

    public static func encodePrefill(
        _ enc: any MTLComputeCommandEncoder,
        ctx: FNGPUContext,
        a: FNGPUAttnMixer,
        c: FNGPUAttnDecodeState,
        s: FNGPUScratch,
        pos: Int,
        out: QWTensorValue,
        cacheOnly: Bool = false,
        lastRowOnly: Bool = false
    ) throws {
        try FNGPULayerPasses.attentionPrefill(
            enc, ctx: ctx, a: a, c: c, s: s, pos: pos, out: out,
            cacheOnly: cacheOnly, lastRowOnly: lastRowOnly
        )
    }
}
