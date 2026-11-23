// QWInkling2TestRef.swift — Exact C-layout params & Float64 CPU references for Inkling-2 parity tests.
import Foundation
import Metal

enum QWInk2TestRef {
    struct EmbedParams {
        var T: UInt32; var H: UInt32; var V: UInt32; var eps: Float
    }
    struct NormParams {
        var rows: UInt32; var dim: UInt32; var inIsF32: UInt32; var eps: Float
    }
    struct DenseGemmParams {
        var M: UInt32; var N: UInt32; var K: UInt32
        var strideX: UInt32; var strideY: UInt32
        var inIsF32: UInt32; var outIsF32: UInt32; var addResidual: UInt32
        var softcap: Float
    }
    struct RouterParams {
        var T: UInt32; var H: UInt32; var numExperts: UInt32; var topK: UInt32; var routeScale: Float
        var moeSpecBudget: UInt32 = 0
    }
    struct MoEParams {
        var T: UInt32; var N: UInt32; var K: UInt32; var topK: UInt32
        var numExperts: UInt32; var hidden: UInt32; var gateCap: Float; var upCap: Float
    }
    struct AttnResParams {
        var T: UInt32; var H: UInt32; var nBlocks: UInt32; var numQ: UInt32
        var blockStride: UInt32; var partialIsF32: UInt32; var normalizeValues: UInt32; var eps: Float
        var slotIds: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct BankAttendParams {
        var T: UInt32; var H: UInt32; var nStates: UInt32; var numQ: UInt32
        var stateStride: UInt32; var eps: Float
        var slotIds: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }
    struct CtrlParams {
        var T: UInt32; var H: UInt32; var rank: UInt32; var eps: Float
    }
    struct SConvParams {
        var T: UInt32; var C: UInt32; var nKV: UInt32; var kernelSize: UInt32
        var updateState: UInt32; var resMode: UInt32; var invSqrtDenom: Float
    }
    struct QKVRParams {
        var T: UInt32; var nH: UInt32; var nKV: UInt32; var headDim: UInt32; var dRel: UInt32
        var isSliding: UInt32; var posOffset: UInt32; var ringCap: UInt32; var kvMode: UInt32
        var nFloor: UInt32; var qkClip: Float; var relClip: Float; var logAlpha: Float; var rmsEps: Float
    }
    struct AttnParams {
        var T: UInt32; var nH: UInt32; var nKV: UInt32; var headDim: UInt32; var dRel: UInt32
        var sliding: UInt32; var relExtent: UInt32; var posOffset: UInt32; var ringCap: UInt32
        var kvMode: UInt32; var kvLen: UInt32; var splits: UInt32
    }

    static func bf16Bits(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    static func f32FromBF16(_ u: UInt16) -> Float {
        Float(bitPattern: UInt32(u) << 16)
    }

    static func roundBF16(_ v: Double) -> Float {
        f32FromBF16(bf16Bits(Float(v)))
    }

    static func bf16Array(_ vals: [Float]) -> [UInt16] {
        vals.map { bf16Bits($0) }
    }

    static func floatsFromBF16Buffer(_ buf: any MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { f32FromBF16(ptr[$0]) }
    }

    static func floatsFromF32Buffer(_ buf: any MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    static let fp4Table: [Double] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    static func fp8E4M3FN(_ b: UInt8) -> Double {
        let s: Double = (b & 0x80) != 0 ? -1.0 : 1.0
        let e = Int((b >> 3) & 0xF)
        let m = Int(b & 0x7)
        if e == 15 && m == 7 { return .nan }
        if e == 0 { return s * pow(2.0, -6.0) * (Double(m) / 8.0) }
        return s * pow(2.0, Double(e - 7)) * (1.0 + Double(m) / 8.0)
    }

    static func fp8E8M0(_ b: UInt8) -> Double {
        if b == 255 { return .nan }
        if b == 0 { return pow(2.0, -127.0) }
        return pow(2.0, Double(Int(b) - 127))
    }

    static func fp8E4M3Byte(for idx: Int) -> UInt8 {
        // Avoid NaN byte 0x7F/0xFF and tiny subnormals; cover positive & negative scales cleanly.
        let valid: [UInt8] = [0x38, 0x3C, 0x40, 0x44, 0xB8, 0xBC, 0xC0, 0x34, 0x30, 0x28]
        return valid[idx % valid.count]
    }

    static func fp8E8M0Byte(for idx: Int) -> UInt8 {
        let valid: [UInt8] = [125, 126, 127, 128, 129]
        return valid[idx % valid.count]
    }

    static func lcg(_ seed: inout UInt64) -> Double {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let v = Int64(bitPattern: seed) >> 40
        return Double(v) / Double(1 << 23) * 0.25
    }

    static func randomBF16Floats(count: Int, seed: UInt64, scale: Double = 1.0) -> [Float] {
        var s = seed
        return (0..<count).map { _ in roundBF16(lcg(&s) * scale) }
    }

    static func randomF32Floats(count: Int, seed: UInt64, scale: Double = 1.0) -> [Float] {
        var s = seed
        return (0..<count).map { _ in Float(lcg(&s) * scale) }
    }

    static func softcapSwiGLU(_ g: Double, _ u: Double, gateCap: Double, upCap: Double) -> Double {
        let sig = 1.0 / (1.0 + exp(-g))
        let val = tanh(g / gateCap) * gateCap * sig * tanh(u / upCap) * upCap
        return Double(roundBF16(val))
    }

    static func cpuAttnRes(
        blocks: [[Double]],
        partial: [Double],
        queries: [[Double]],
        tCount: Int,
        h: Int,
        normalizeValues: Bool,
        eps: Double
    ) -> [Double] {
        var cands = blocks
        cands.append(partial)
        let nC = cands.count
        var out = [Double](repeating: 0, count: queries.count * tCount * h)
        let scale = 1.0 / sqrt(Double(h))
        for q in 0..<queries.count {
            for t in 0..<tCount {
                var scores = [Double](repeating: 0, count: nC)
                var invRms = [Double](repeating: 0, count: nC)
                for b in 0..<nC {
                    var sq = 0.0, dot = 0.0
                    for d in 0..<h {
                        let v = cands[b][t * h + d]
                        sq += v * v
                        dot += v * queries[q][d]
                    }
                    invRms[b] = 1.0 / sqrt(sq / Double(h) + eps)
                    scores[b] = dot * invRms[b] * scale
                }
                let mx = scores.max() ?? 0
                let exps = scores.map { exp($0 - mx) }
                let sm = exps.reduce(0, +)
                for d in 0..<h {
                    var acc = 0.0
                    for b in 0..<nC {
                        let v = cands[b][t * h + d] * (normalizeValues ? invRms[b] : 1.0)
                        acc += (exps[b] / sm) * v
                    }
                    out[(q * tCount + t) * h + d] = acc
                }
            }
        }
        return out
    }

    static func cpuGQAAttention(
        q: [Float], k: [Float], v: [Float], r: [Float],
        relProj: [Float], sink: [Float], gate: [Float],
        tCount: Int, posOffset: Int, nH: Int, nKV: Int, headDim: Int,
        dRel: Int, sliding: Int, relExtent: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: tCount * nH * headDim)
        let scale = 1.0 / sqrt(Double(headDim))
        let group = nH / nKV
        for t in 0..<tCount {
            let qPos = posOffset + t
            let minKey = (sliding > 0 && qPos + 1 > sliding) ? (qPos + 1 - sliding) : 0
            for h in 0..<nH {
                let kvh = h / group
                var scores: [Double] = []
                for j in minKey...qPos {
                    var dot = 0.0
                    for d in 0..<headDim {
                        dot += Double(q[(t * nH + h) * headDim + d]) * Double(k[(j * nKV + kvh) * headDim + d])
                    }
                    var s = dot * scale
                    let dist = qPos - j
                    if dist < relExtent {
                        for dr in 0..<dRel {
                            s += Double(r[(t * nH + h) * dRel + dr]) * Double(relProj[dr * relExtent + dist])
                        }
                    }
                    scores.append(s)
                }
                let sSink = Double(sink[t * nH + h])
                let maxAll = max(sSink, scores.max() ?? -1e30)
                var sumW = exp(sSink - maxAll)
                var acc = [Double](repeating: 0, count: headDim)
                for (idx, j) in (minKey...qPos).enumerated() {
                    let w = exp(scores[idx] - maxAll)
                    sumW += w
                    for d in 0..<headDim {
                        acc[d] += w * Double(v[(j * nKV + kvh) * headDim + d])
                    }
                }
                for d in 0..<headDim {
                    let g = Double(gate[(t * nH + h) * headDim + d])
                    let sig = 1.0 / (1.0 + exp(-g))
                    out[(t * nH + h) * headDim + d] = roundBF16((acc[d] / sumW) * sig)
                }
            }
        }
        return out
    }
}
