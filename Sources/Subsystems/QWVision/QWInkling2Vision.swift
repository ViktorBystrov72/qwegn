// QWInkling2Vision.swift — CPU Accelerate FP32 23L ViT reference backend for Inkling-2.
import Accelerate
import Foundation
import QWCore

public final class QWInkling2VisionLayerW: @unchecked Sendable {
    let lnBeforeG: [Float], lnBeforeB: [Float]
    let lnAfterG: [Float], lnAfterB: [Float]
    let lambda1: [Float], lambda2: [Float]
    let qW: [Float], qB: [Float]
    let kW: [Float], kB: [Float]
    let vW: [Float], vB: [Float]
    let outW: [Float], outB: [Float]
    let fc1W: [Float], fc1B: [Float]
    let fc2W: [Float], fc2B: [Float]

    init(_ ck: QWVisionCheckpoint, _ p: String, dim: Int, mlpInner: Int) throws {
        lnBeforeG = try ck.f32("\(p).layernorm_before.weight", want: [dim])
        lnBeforeB = try ck.f32("\(p).layernorm_before.bias", want: [dim])
        lnAfterG = try ck.f32("\(p).layernorm_after.weight", want: [dim])
        lnAfterB = try ck.f32("\(p).layernorm_after.bias", want: [dim])
        lambda1 = try ck.f32("\(p).lambda_1", want: [dim])
        lambda2 = try ck.f32("\(p).lambda_2", want: [dim])
        qW = try ck.f32("\(p).self_attn.q_proj.weight", want: [dim, dim])
        qB = try ck.f32("\(p).self_attn.q_proj.bias", want: [dim])
        kW = try ck.f32("\(p).self_attn.k_proj.weight", want: [dim, dim])
        kB = try ck.f32("\(p).self_attn.k_proj.bias", want: [dim])
        vW = try ck.f32("\(p).self_attn.v_proj.weight", want: [dim, dim])
        vB = try ck.f32("\(p).self_attn.v_proj.bias", want: [dim])
        outW = try ck.f32("\(p).self_attn.out_proj.weight", want: [dim, dim])
        outB = try ck.f32("\(p).self_attn.out_proj.bias", want: [dim])
        fc1W = try ck.f32("\(p).mlp.fc1.weight", want: [mlpInner, dim])
        fc1B = try ck.f32("\(p).mlp.fc1.bias", want: [mlpInner])
        fc2W = try ck.f32("\(p).mlp.fc2.weight", want: [dim, mlpInner])
        fc2B = try ck.f32("\(p).mlp.fc2.bias", want: [dim])
    }

    var residentBytes: Int {
        var n = lnBeforeG.count + lnBeforeB.count + lnAfterG.count + lnAfterB.count
        n += lambda1.count + lambda2.count
        n += qW.count + qB.count + kW.count + kB.count
        n += vW.count + vB.count + outW.count + outB.count
        n += fc1W.count + fc1B.count + fc2W.count + fc2B.count
        return n * MemoryLayout<Float>.stride
    }
}

public final class QWInkling2VisionWeights: @unchecked Sendable {
    public let config: QWVisionConfig
    public let patchW: [Float]
    public let posEmbed: [Float]
    public let classEmbed: [Float]
    public let clsPosEmbed: [Float]
    public let preNormG: [Float]
    public let preNormB: [Float]
    public let layers: [QWInkling2VisionLayerW]
    public let projInW: [Float]
    public let projInB: [Float]?
    public let projOutW: [Float]
    public let projOutB: [Float]?
    public let invFreq: [Float]
    public let projectionHidden: Int
    public let outDim: Int

    public var residentBytes: Int {
        var n = patchW.count + posEmbed.count + classEmbed.count + clsPosEmbed.count
        n += preNormG.count + preNormB.count + projInW.count + (projInB?.count ?? 0)
        n += projOutW.count + (projOutB?.count ?? 0) + invFreq.count
        return n * MemoryLayout<Float>.stride + layers.reduce(0) { $0 + $1.residentBytes }
    }

    public init(checkpoint ck: QWVisionCheckpoint) throws {
        let cfg = ck.config
        let d = cfg.dim
        let merged = cfg.merged
        let projH = cfg.projectionHiddenDim > 0 ? cfg.projectionHiddenDim : merged
        let resolvedOut = ck.shape("proj_out.weight")?[0] ?? cfg.outDim
        self.config = cfg
        self.projectionHidden = projH
        self.outDim = resolvedOut
        self.patchW = try ck.f32("embeddings.patch_embedding.weight", want: [d, cfg.patchChannels])
        self.posEmbed = try ck.f32("embeddings.position_embedding", want: [cfg.numPosEmbeds, d])
        self.classEmbed = try ck.f32("embeddings.class_embedding", want: [d])
        self.clsPosEmbed = try ck.f32("embeddings.cls_position_embedding", want: [d])
        self.preNormG = try ck.f32("pre_layernorm.weight", want: [d])
        self.preNormB = try ck.f32("pre_layernorm.bias", want: [d])

        var depth = 0
        while ck.has("layers.\(depth).layernorm_before.weight") { depth += 1 }
        guard depth > 0 else { throw QWVisionError.missingTensor("layers.0.layernorm_before.weight") }
        self.layers = try (0..<depth).map {
            try QWInkling2VisionLayerW(ck, "layers.\($0)", dim: d, mlpInner: cfg.mlpInner)
        }
        self.projInW = try ck.f32("proj_in.weight", want: [projH, merged])
        self.projInB = cfg.projectorBias && ck.has("proj_in.bias")
            ? try ck.f32("proj_in.bias", want: [projH])
            : nil
        self.projOutW = try ck.f32("proj_out.weight", want: [resolvedOut, projH])
        self.projOutB = cfg.projectorBias && ck.has("proj_out.bias")
            ? try ck.f32("proj_out.bias", want: [resolvedOut])
            : nil
        self.invFreq = Self.computeInvFreq(config: cfg)
        ck.releaseBacking()
    }

    private static func computeInvFreq(config cfg: QWVisionConfig) -> [Float] {
        var f = [Float](repeating: 0, count: cfg.invFreqCount)
        for i in 0..<f.count {
            f[i] = Float(pow(cfg.ropeTheta, -Double(2 * i) / Double(cfg.spatial)))
        }
        return f
    }
}

@_silgen_name("cblas_sgemm$NEWLAPACK")
private func qwInk2VisSgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32, _ alpha: Float,
    _ a: UnsafePointer<Float>, _ lda: Int32,
    _ b: UnsafePointer<Float>, _ ldb: Int32, _ beta: Float,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int32
)

public final class QWInkling2VisionRef: @unchecked Sendable {
    public let w: QWInkling2VisionWeights

    public init(weights: QWInkling2VisionWeights) {
        self.w = weights
    }

    /// Encodes `[nPatch, patchChannels]` normalized HWC patches into `[nToken, outDim]`.
    public func encode(
        patches x: [Float],
        plan: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) -> [Float] {
        let c = w.config
        let nPatch = plan.nPatch
        let nToken = plan.nToken
        let d = c.dim
        precondition(x.count == nPatch * c.patchChannels && bins.nPatch == nPatch && posIds.count == 2 * nPatch)
        guard nPatch > 0, nToken > 0, d > 0 else { return [] }

        var patchOut = [Float](repeating: 0, count: nPatch * d)
        patchOut.withUnsafeMutableBufferPointer { pb in
            guard let ptr = pb.baseAddress else { return }
            qwGemmRef(y: ptr, ldY: d, yBase: 0, x: x, rows: nPatch, w: w.patchW, nOut: d, nIn: c.patchChannels, bias: nil)
        }
        foldPosEmbed(into: &patchOut, bins: bins, dim: d)

        let nSeq = nPatch + 1
        var s = [Float](repeating: 0, count: nSeq * d)
        for i in 0..<d { s[i] = w.classEmbed[i] + w.clsPosEmbed[i] }
        s.withUnsafeMutableBufferPointer { sb in
            patchOut.withUnsafeBufferPointer { pb in
                guard let dst = sb.baseAddress, let src = pb.baseAddress else { return }
                memcpy(dst + d, src, nPatch * d * MemoryLayout<Float>.stride)
            }
        }

        var normed = [Float](repeating: 0, count: nSeq * d)
        qwLayerNorm(s, into: &normed, gamma: w.preNormG, beta: w.preNormB, rows: nSeq, dim: d, eps: c.eps)
        s = normed

        let (cos, sin) = ropeCosSin(posIds: posIds, nPatch: nPatch)
        var qBuf = [Float](repeating: 0, count: nSeq * d)
        var kBuf = [Float](repeating: 0, count: nSeq * d)
        var vBuf = [Float](repeating: 0, count: nSeq * d)
        var attnOut = [Float](repeating: 0, count: nSeq * d)
        var qHead = [Float](repeating: 0, count: c.heads * nSeq * c.headDim)
        var kHead = [Float](repeating: 0, count: c.heads * nSeq * c.headDim)
        var scores = [Float](repeating: 0, count: c.heads * nSeq * nSeq)
        var mlpH = [Float](repeating: 0, count: nSeq * c.mlpInner)

        for layer in w.layers {
            runBlock(
                s: &s, layer: layer, nSeq: nSeq, cos: cos, sin: sin,
                normed: &normed, qBuf: &qBuf, kBuf: &kBuf, vBuf: &vBuf, attnOut: &attnOut,
                qHead: &qHead, kHead: &kHead, scores: &scores, mlpH: &mlpH
            )
        }

        let noCls = Array(s.dropFirst(d))
        var hidden = [Float](repeating: 0, count: nToken * w.projectionHidden)
        hidden.withUnsafeMutableBufferPointer { hb in
            guard let hp = hb.baseAddress else { return }
            qwGemmRef(y: hp, ldY: w.projectionHidden, yBase: 0, x: noCls, rows: nToken,
                      w: w.projInW, nOut: w.projectionHidden, nIn: c.merged, bias: w.projInB, act: qwGeluErf)
        }
        var out = [Float](repeating: 0, count: nToken * w.outDim)
        out.withUnsafeMutableBufferPointer { ob in
            guard let op = ob.baseAddress else { return }
            qwGemmRef(y: op, ldY: w.outDim, yBase: 0, x: hidden, rows: nToken,
                      w: w.projOutW, nOut: w.outDim, nIn: w.projectionHidden, bias: w.projOutB)
        }
        return out
    }

    private func foldPosEmbed(into s: inout [Float], bins: QWVisionRef.InterpBins, dim: Int) {
        let numPos = w.posEmbed.count / dim
        guard numPos > 0 else { return }
        let vDim = vDSP_Length(dim)
        bins.idx.withUnsafeBufferPointer { ib in
            bins.wgt.withUnsafeBufferPointer { wb in
                w.posEmbed.withUnsafeBufferPointer { tb in
                    s.withUnsafeMutableBufferPointer { sb in
                        guard let sBase = sb.baseAddress, let tBase = tb.baseAddress,
                              let iBase = ib.baseAddress, let wBase = wb.baseAddress else { return }
                        for p in 0..<bins.nPatch {
                            let sRow = sBase + p * dim
                            for c in 0..<4 {
                                let idx = Int(iBase[p * 4 + c])
                                guard idx >= 0, idx < numPos else { continue }
                                var wc = wBase[p * 4 + c]
                                vDSP_vsma(tBase + idx * dim, 1, &wc, sRow, 1, sRow, 1, vDim)
                            }
                        }
                    }
                }
            }
        }
    }

    private func ropeCosSin(posIds: [Int32], nPatch: Int) -> (cos: [Float], sin: [Float]) {
        let c = w.config
        let hd = c.headDim
        let inv = c.invFreqCount
        let nSeq = nPatch + 1
        var cos = [Float](repeating: 1, count: nSeq * hd)
        var sin = [Float](repeating: 0, count: nSeq * hd)
        let xBlock = 2 * inv
        for p in 0..<nPatch {
            let row = p + 1
            let px = Float(posIds[p * 2])
            let py = Float(posIds[p * 2 + 1])
            let off = row * hd
            for i in 0..<inv {
                let ax = px * w.invFreq[i], ay = py * w.invFreq[i]
                let cx = cosf(ax), sx = sinf(ax), cy = cosf(ay), sy = sinf(ay)
                cos[off + 2 * i] = cx;     cos[off + 2 * i + 1] = cx
                sin[off + 2 * i] = sx;     sin[off + 2 * i + 1] = sx
                cos[off + xBlock + 2 * i] = cy; cos[off + xBlock + 2 * i + 1] = cy
                sin[off + xBlock + 2 * i] = sy; sin[off + xBlock + 2 * i + 1] = sy
            }
        }
        return (cos, sin)
    }

    private func runBlock(
        s: inout [Float], layer: QWInkling2VisionLayerW, nSeq: Int,
        cos: [Float], sin: [Float],
        normed: inout [Float], qBuf: inout [Float], kBuf: inout [Float], vBuf: inout [Float],
        attnOut: inout [Float], qHead: inout [Float], kHead: inout [Float],
        scores: inout [Float], mlpH: inout [Float]
    ) {
        let cfg = w.config
        let d = cfg.dim, heads = cfg.heads, hd = cfg.headDim, mlp = cfg.mlpInner
        qwLayerNorm(s, into: &normed, gamma: layer.lnBeforeG, beta: layer.lnBeforeB, rows: nSeq, dim: d, eps: cfg.eps)
        qBuf.withUnsafeMutableBufferPointer { qb in
            guard let qp = qb.baseAddress else { return }
            qwGemmRef(y: qp, ldY: d, yBase: 0, x: normed, rows: nSeq, w: layer.qW, nOut: d, nIn: d, bias: layer.qB)
        }
        kBuf.withUnsafeMutableBufferPointer { kb in
            guard let kp = kb.baseAddress else { return }
            qwGemmRef(y: kp, ldY: d, yBase: 0, x: normed, rows: nSeq, w: layer.kW, nOut: d, nIn: d, bias: layer.kB)
        }
        vBuf.withUnsafeMutableBufferPointer { vb in
            guard let vp = vb.baseAddress else { return }
            qwGemmRef(y: vp, ldY: d, yBase: 0, x: normed, rows: nSeq, w: layer.vW, nOut: d, nIn: d, bias: layer.vB)
        }

        let scale = Float(1.0 / Double(hd).squareRoot())
        qBuf.withUnsafeBufferPointer { qb in
            kBuf.withUnsafeBufferPointer { kb in
                vBuf.withUnsafeBufferPointer { vb in
                    cos.withUnsafeBufferPointer { cb in
                        sin.withUnsafeBufferPointer { sb in
                            qHead.withUnsafeMutableBufferPointer { qhb in
                                kHead.withUnsafeMutableBufferPointer { khb in
                                    scores.withUnsafeMutableBufferPointer { scb in
                                        attnOut.withUnsafeMutableBufferPointer { ab in
                                            guard let q0 = qb.baseAddress, let k0 = kb.baseAddress, let v0 = vb.baseAddress,
                                                  let c0 = cb.baseAddress, let s0 = sb.baseAddress,
                                                  let qh0 = qhb.baseAddress, let kh0 = khb.baseAddress,
                                                  let sc0 = scb.baseAddress, let a0 = ab.baseAddress else { return }
                                            nonisolated(unsafe) let qu = q0, ku = k0, vu = v0, cu = c0, su = s0
                                            nonisolated(unsafe) let qhu = qh0, khu = kh0, scu = sc0, au = a0
                                            DispatchQueue.concurrentPerform(iterations: heads) { head in
                                                let qPtr = qhu + head * nSeq * hd
                                                let kPtr = khu + head * nSeq * hd
                                                let scPtr = scu + head * nSeq * nSeq
                                                let halfPairs = hd / 2
                                                for row in 0..<nSeq {
                                                    let qs = qu + row * d + head * hd
                                                    let ks = ku + row * d + head * hd
                                                    let cr = cu + row * hd, sr = su + row * hd
                                                    let qd = qPtr + row * hd, kd = kPtr + row * hd
                                                    for j in 0..<halfPairs {
                                                        let j0 = 2 * j, j1 = j0 + 1
                                                        let cc = cr[j0], ss = sr[j0]
                                                        qd[j0] = qs[j0] * cc - qs[j1] * ss
                                                        qd[j1] = qs[j1] * cc + qs[j0] * ss
                                                        kd[j0] = ks[j0] * cc - ks[j1] * ss
                                                        kd[j1] = ks[j1] * cc + ks[j0] * ss
                                                    }
                                                }
                                                qwInk2VisSgemm(101, 111, 112, Int32(nSeq), Int32(nSeq), Int32(hd),
                                                               scale, qPtr, Int32(hd), kPtr, Int32(hd), 0, scPtr, Int32(nSeq))
                                                var n32 = Int32(nSeq), vN = vDSP_Length(nSeq)
                                                for qi in 0..<nSeq {
                                                    let row = scPtr + qi * nSeq
                                                    var mx: Float = -.greatestFiniteMagnitude
                                                    vDSP_maxv(row, 1, &mx, vN)
                                                    var negMx = -mx
                                                    vDSP_vsadd(row, 1, &negMx, row, 1, vN)
                                                    vvexpf(row, row, &n32)
                                                    var sum: Float = 0
                                                    vDSP_sve(row, 1, &sum, vN)
                                                    var inv = sum > 0 ? (1.0 / sum) : 0
                                                    vDSP_vsmul(row, 1, &inv, row, 1, vN)
                                                }
                                                qwInk2VisSgemm(101, 111, 111, Int32(nSeq), Int32(hd), Int32(nSeq),
                                                               1.0, scPtr, Int32(nSeq), vu + head * hd, Int32(d), 0, au + head * hd, Int32(d))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        normed.withUnsafeMutableBufferPointer { nb in
            guard let np = nb.baseAddress else { return }
            qwGemmRef(y: np, ldY: d, yBase: 0, x: attnOut, rows: nSeq, w: layer.outW, nOut: d, nIn: d, bias: layer.outB)
        }
        applyLayerScaleResidual(dst: &s, branch: normed, scale: layer.lambda1, rows: nSeq, dim: d)

        qwLayerNorm(s, into: &normed, gamma: layer.lnAfterG, beta: layer.lnAfterB, rows: nSeq, dim: d, eps: cfg.eps)
        mlpH.withUnsafeMutableBufferPointer { hb in
            guard let hp = hb.baseAddress else { return }
            qwGemmRef(y: hp, ldY: mlp, yBase: 0, x: normed, rows: nSeq, w: layer.fc1W, nOut: mlp, nIn: d, bias: layer.fc1B)
            for i in 0..<(nSeq * mlp) { hp[i] = qwGeluErf(hp[i]) }
        }
        normed.withUnsafeMutableBufferPointer { nb in
            guard let np = nb.baseAddress else { return }
            qwGemmRef(y: np, ldY: d, yBase: 0, x: mlpH, rows: nSeq, w: layer.fc2W, nOut: d, nIn: mlp, bias: layer.fc2B)
        }
        applyLayerScaleResidual(dst: &s, branch: normed, scale: layer.lambda2, rows: nSeq, dim: d)
    }

    private func applyLayerScaleResidual(dst: inout [Float], branch: [Float], scale: [Float], rows: Int, dim: Int) {
        dst.withUnsafeMutableBufferPointer { db in
            branch.withUnsafeBufferPointer { bb in
                scale.withUnsafeBufferPointer { sb in
                    guard let dP = db.baseAddress, let bP = bb.baseAddress, let sP = sb.baseAddress else { return }
                    let vD = vDSP_Length(dim)
                    for r in 0..<rows {
                        let off = r * dim
                        vDSP_vma(bP + off, 1, sP, 1, dP + off, 1, dP + off, 1, vD)
                    }
                }
            }
        }
    }
}
