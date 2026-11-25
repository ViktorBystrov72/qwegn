// QWRopeTables+PerKind.swift — Per-layer-kind RoPE tables & zero-tail partial rotation encoder.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

private struct QWRopeZeroTailParams {
    var seqLen: UInt32
    var numHeads: UInt32
    var headDim: UInt32
    var rotaryDims: UInt32
    var posOffset: UInt32
    var zeroTail: UInt32
}

extension QWRopeTables {
    /// Builds a flat `[rows, rotaryDims]` FP32 `[cos(0..<pairs), sin(0..<pairs)]` table for `entry`.
    /// When `zeroTailFreqs` is true (`Lfm3RotaryEmbedding`), frequencies follow the full `headDim`
    /// progression for prefix `ropeAngles` pairs and pad trailing frequencies with 0 (unrotated tail).
    public static func makePerKindTable(
        rows: Int,
        headDim: Int,
        entry: PerKindRoPEEntry,
        zeroTailFreqs: Bool = false
    ) throws -> (cosSin: [Float], rotaryDims: Int) {
        guard rows > 0, headDim > 0, headDim % 2 == 0 else {
            throw QWRopeError.badShape("invalid rows=\(rows) or headDim=\(headDim)")
        }
        let theta = Float(entry.ropeTheta)
        if zeroTailFreqs {
            let pairs = headDim / 2
            let ropeAngles = max(1, min(pairs, Int(Double(headDim) * entry.partialRotaryFactor) / 2))
            var invFreq = [Float](repeating: 0, count: pairs)
            for p in 0..<ropeAngles {
                let expTerm = Float(2 * p) / Float(headDim)
                invFreq[p] = 1.0 / powf(theta, expTerm)
            }
            var out = [Float](repeating: 0, count: rows * headDim)
            for r in 0..<rows {
                let base = r * headDim
                let pos = Float(r)
                for p in 0..<pairs {
                    let angle = pos * invFreq[p]
                    out[base + p] = cosf(angle)
                    out[base + pairs + p] = sinf(angle)
                }
            }
            return (out, headDim)
        }
        let rawRot = Int((Double(headDim) * entry.partialRotaryFactor).rounded())
        let rotaryDims = max(2, min(headDim, rawRot - (rawRot % 2)))
        let pairs = rotaryDims / 2

        var invFreq = [Float](repeating: 0, count: pairs)
        for p in 0..<pairs {
            let expTerm = Float(2 * p) / Float(rotaryDims)
            invFreq[p] = 1.0 / powf(theta, expTerm)
        }
        var out = [Float](repeating: 0, count: rows * rotaryDims)
        for r in 0..<rows {
            let base = r * rotaryDims
            let pos = Float(r)
            for p in 0..<pairs {
                let angle = pos * invFreq[p]
                out[base + p] = cosf(angle)
                out[base + pairs + p] = sinf(angle)
            }
        }
        return (out, rotaryDims)
    }

    /// Materializes a per-token `[tokenCount, width]` FP32 `[cos, sin]` buffer from planar `[3, tokenCount]`
    /// 3D `MRoPE` positions (`pos[0..<N]` = T, `pos[N..<2N]` = H, `pos[2N..<3N]` = W) by gathering rows from `cosSin`.
    public func buildMRoPECosSin(
        mropePositions: [Int32],
        tokenCount: Int,
        layout: QWModelDescriptor.MRoPELayout
    ) throws -> [Float] {
        guard tokenCount > 0, mropePositions.count >= 3 * tokenCount else {
            throw QWRopeError.badShape("invalid 3D MRoPE positions count=\(mropePositions.count) for N=\(tokenCount)")
        }
        var out = [Float](repeating: 0, count: tokenCount * width)
        let hSec = (pairs * 11) / 32
        let wSec = (pairs * 10) / 32
        let bailingT = pairs / 4
        let bailingH = (pairs * 3) / 8
        let bailingW = pairs - bailingT - bailingH

        for r in 0..<tokenCount {
            let pT = Int(mropePositions[r])
            let pH = Int(mropePositions[tokenCount + r])
            let pW = Int(mropePositions[2 * tokenCount + r])
            guard pT >= 0, pT < rows, pH >= 0, pH < rows, pW >= 0, pW < rows else {
                throw QWRopeError.posBeyondTable(max(pT, max(pH, pW)), rows)
            }
            let dstBase = r * width
            for i in 0..<pairs {
                let axisPos: Int
                switch layout {
                case .qwenInterleaved:
                    if i % 3 == 1 && i < 3 * hSec {
                        axisPos = pH
                    } else if i % 3 == 2 && i < 3 * wSec {
                        axisPos = pW
                    } else {
                        axisPos = pT
                    }
                case .bailingVideoRope:
                    // Reverse VideoRoPE allocation: [W, H, T] across rotary pairs.
                    if i < bailingW {
                        axisPos = pW
                    } else if i < bailingW + bailingH {
                        axisPos = pH
                    } else {
                        axisPos = pT
                    }
                case .none1D:
                    axisPos = pT
                }
                let srcBase = axisPos * width
                out[dstBase + i] = cosSin[srcBase + i]
                out[dstBase + pairs + i] = cosSin[srcBase + pairs + i]
            }
        }
        return out
    }
}

extension QWRopeApply {
    /// Applies half-split RoPE over `0..<rotaryDims` and optionally zeros unrotated tail channels.
    public static func encodeZeroTail(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        x: QWTensorValue,
        out: QWTensorValue,
        table: QWTensorValue,
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        rotaryDims: Int,
        posOffset: Int = 0,
        zeroTail: Bool = false
    ) throws {
        guard seqLen > 0, numHeads > 0, headDim > 0, rotaryDims > 0, rotaryDims <= headDim, rotaryDims % 2 == 0 else {
            throw QWRopeError.badShape("invalid zero-tail RoPE shape S=\(seqLen) H=\(numHeads) D=\(headDim) R=\(rotaryDims)")
        }
        guard x.shape.dtype == .f16, out.shape.dtype == .f16, table.shape.dtype == .f32,
              x.shape.elementCount == seqLen * numHeads * headDim,
              out.shape.elementCount == seqLen * numHeads * headDim else {
            throw QWRopeError.badShape("zero-tail RoPE dtype/count mismatch")
        }
        let pso = try hal.pipeline("qwg_rope_zero_tail_f16")
        var params = QWRopeZeroTailParams(
            seqLen: UInt32(seqLen),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim),
            rotaryDims: UInt32(rotaryDims),
            posOffset: UInt32(max(0, posOffset)),
            zeroTail: zeroTail ? 1 : 0
        )
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        out.bind(enc, index: 1)
        table.bind(enc, index: 2)
        enc.setBytes(&params, length: MemoryLayout<QWRopeZeroTailParams>.stride, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
}
