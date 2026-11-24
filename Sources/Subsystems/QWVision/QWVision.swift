// QWVision.swift — CPU fp32 reference ViT behind `QWVisionBackend` (spec-05).
import Accelerate
import Foundation
import QWCore
import QWQuant

// MARK: - LayerNorm (law 4: the torch law, executable)

/// torch LayerNorm: y = (x−μ)·rsqrt(var+eps)·γ + β; var BIASED (÷dim);
/// eps INSIDE the sqrt raw (not eps² — that's a different framework; the
/// golden adjacency 1.9e-3 proves the raw-eps law for THIS checkpoint).
@inline(__always)
func qwLayerNormRow(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                    _ gamma: UnsafePointer<Float>?, _ beta: UnsafePointer<Float>?,
                    dim: Int, eps: Float) {
    guard dim > 0 else { return }
    let vDim = vDSP_Length(dim)
    var m: Float = 0
    vDSP_meanv(x, 1, &m, vDim)
    var negM = -m
    vDSP_vsadd(x, 1, &negM, out, 1, vDim)
    var s2: Float = 0
    vDSP_svesq(out, 1, &s2, vDim)
    let denom = (Double(s2) / Double(dim) + Double(eps)).squareRoot()
    if denom > 0 && denom.isFinite {
        var inv = Float(1.0 / denom)
        vDSP_vsmul(out, 1, &inv, out, 1, vDim)
    } else {
        vDSP_vclr(out, 1, vDim)
    }
    if let g = gamma, let b = beta {
        vDSP_vmul(out, 1, g, 1, out, 1, vDim)
        vDSP_vadd(out, 1, b, 1, out, 1, vDim)
    }
}

func qwLayerNormRows(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                     _ gamma: UnsafePointer<Float>?, _ beta: UnsafePointer<Float>?,
                     rows: Int, dim: Int, eps: Float) {
    guard rows > 0, dim > 0 else { return }
    nonisolated(unsafe) let xPtr = x, oPtr = out, gPtr = gamma, bPtr = beta
    if rows >= 64 {
        let chunks = min(16, rows), perChunk = (rows + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let rStart = c * perChunk, rEnd = min(rows, rStart + perChunk)
            guard rStart < rEnd else { return }
            for r in rStart..<rEnd { qwLayerNormRow(xPtr + r &* dim, oPtr + r &* dim, gPtr, bPtr, dim: dim, eps: eps) }
        }
    } else {
        for r in 0..<rows { qwLayerNormRow(xPtr + r &* dim, oPtr + r &* dim, gPtr, bPtr, dim: dim, eps: eps) }
    }
}

func qwLayerNorm(_ x: [Float], into out: inout [Float], gamma: [Float], beta: [Float],
                 rows: Int, dim: Int, eps: Float) {
    guard rows > 0, dim > 0,
          x.count >= rows * dim, out.count >= rows * dim,
          gamma.count >= dim, beta.count >= dim else { return }
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            gamma.withUnsafeBufferPointer { g in
                beta.withUnsafeBufferPointer { b in
                    guard let xBase = xb.baseAddress, let oBase = ob.baseAddress,
                          let gBase = g.baseAddress, let bBase = b.baseAddress else { return }
                    qwLayerNormRows(xBase, oBase, gBase, bBase,
                                    rows: rows, dim: dim, eps: eps)
                }
            }
        }
    }
}

// MARK: - the GELU split (law 4b: two GELUs, one checkpoint)

/// nn.GELU() default = ERF form — the MERGER's act (the split is what the
/// goldens bind; mixing them diffs ~1e-3 on merger outputs, well above tol).
@inline(__always) public func qwGeluErf(_ x: Float) -> Float {
    0.5 * x * (1.0 + erf(x * 0.7071067811865476))
}

@inline(__always)
func qwApplyGeluTanhInPlace(_ ptr: UnsafeMutablePointer<Float>, count: Int, scratch: UnsafeMutablePointer<Float>) {
    let vCount = vDSP_Length(count)
    vDSP_vmul(ptr, 1, ptr, 1, scratch, 1, vCount)
    vDSP_vmul(scratch, 1, ptr, 1, scratch, 1, vCount)
    var c1: Float = 0.044715, c2: Float = 0.7978845608028654, one: Float = 1.0, half: Float = 0.5
    vDSP_vsma(scratch, 1, &c1, ptr, 1, scratch, 1, vCount)
    vDSP_vsmul(scratch, 1, &c2, scratch, 1, vCount)
    var n32 = Int32(count)
    vvtanhf(scratch, scratch, &n32)
    vDSP_vsadd(scratch, 1, &one, scratch, 1, vCount)
    vDSP_vmul(ptr, 1, scratch, 1, ptr, 1, vCount)
    vDSP_vsmul(ptr, 1, &half, ptr, 1, vCount)
}

// MARK: - fp32 GEMM (Accelerate AMX cblas_sgemm over row-major [nOut, nIn] weights)

@_silgen_name("cblas_sgemm$NEWLAPACK")
private func qw_cblas_sgemm(
    _ order: Int32, _ transA: Int32, _ transB: Int32,
    _ m: Int32, _ n: Int32, _ k: Int32, _ alpha: Float,
    _ a: UnsafePointer<Float>, _ lda: Int32,
    _ b: UnsafePointer<Float>, _ ldb: Int32, _ beta: Float,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int32
)

/// y[r,o] = Σ_k x[r,k]·w[o,k] (+bias[o]) (+add[r·ldAdd+o]); optional act.
func qwGemmRef(y: UnsafeMutablePointer<Float>, ldY: Int, yBase: Int,
               x: [Float], rows: Int, w: [Float], nOut: Int, nIn: Int,
               bias: [Float]?, add: (src: [Float], ld: Int)? = nil,
               act: ((Float) -> Float)? = nil, rowOffset: Int = 0) {
    guard rows > 0, nOut > 0, nIn > 0 else { return }
    x.withUnsafeBufferPointer { xb in
        w.withUnsafeBufferPointer { wb in
            guard let xBasePtr = xb.baseAddress, let wBasePtr = wb.baseAddress else { return }
            let yStart = y + yBase + rowOffset * ldY
            qw_cblas_sgemm(101, 111, 112, Int32(rows), Int32(nOut), Int32(nIn), 1.0,
                           xBasePtr, Int32(nIn), wBasePtr, Int32(nIn), 0.0, yStart, Int32(ldY))
            let bConst = bias?.withUnsafeBufferPointer { $0.baseAddress }
            let aConst = add?.src.withUnsafeBufferPointer { $0.baseAddress }
            let addLd = add?.ld ?? 0
            guard bConst != nil || aConst != nil || act != nil else { return }
            let vOut = vDSP_Length(nOut)
            for r in 0..<rows {
                let yRow = yStart + r * ldY
                if let b = bConst { vDSP_vadd(yRow, 1, b, 1, yRow, 1, vOut) }
                if let a = aConst { vDSP_vadd(yRow, 1, a + (r + rowOffset) * addLd, 1, yRow, 1, vOut) }
                if let f = act { for o in 0..<nOut { yRow[o] = f(yRow[o]) } }
            }
        }
    }
}

// MARK: - weights (loaded once; the checkpoint's native layouts preserved)

public final class QWVisionWeights: @unchecked Sendable {
    public let ck: QWVisionCheckpoint
    public let config: QWVisionConfig
    public let posEmbed: [Float]          // [numPosEmbeds, dim] raster row-major
    public let patchW: [Float]            // [dim, patchChannels] folded native
    public let patchB: [Float]            // [dim]
    public let blocks: [QWVisionBlockW]
    public let mergerGamma: [Float]       // LN(dim) — affine presence asserted
    public let mergerBeta: [Float]
    public let fc1Mid: Int                // merger intermediate (fc1 out / fc2 in)
    public let fc1W: [Float]              // [fc1Mid, merged]
    public let fc1B: [Float]              // [fc1Mid]
    public let fc2Out: Int                // LLM hidden size (e.g. 5120, 2048, 2560)
    public let fc2W: [Float]              // [fc2Out, fc1Mid]
    public let fc2B: [Float]              // [fc2Out]
    public let invFreq: [Float]           // [invFreqCount]

    public convenience init(dir: String) throws {
        try self.init(checkpoint: try QWVisionCheckpoint(dir: dir))
    }

    /// Pack-mode init: weights come from `.qwp` verbatim-BF16 planes.
    public convenience init(pack packPath: String) throws {
        try self.init(checkpoint: try QWVisionCheckpoint(pack: packPath))
    }

    static func loadFoldedPatchWeight(from ck: QWVisionCheckpoint, config cfg: QWVisionConfig) throws -> [Float] {
        let d = cfg.dim, patch = cfg.patch, temporal = cfg.temporal
        let pw = ck.shape("patch_embed.proj.weight")
        let baseShape = [d, 3, temporal, patch, patch]
        let legacyShape = [d, temporal, patch, patch, 3]
        guard pw == baseShape || pw == legacyShape else {
            throw QWVisionError.badLayout(
                "patch_embed.proj.weight \(pw.debugDescription) ≠ \(baseShape) nor \(legacyShape)")
        }
        let isBase = (pw == baseShape)
        let native = try ck.f32("patch_embed.proj.weight", want: isBase ? baseShape : legacyShape)
        if isBase { return native }
        // Fold legacy (o,t,h,w,c) conv3d kernels to (o,c,t,h,w) to match the C19 slab row layout.
        let patchArea = cfg.patchArea, temporalArea = cfg.temporalPatchArea
        let patchChan = cfg.patchChannels, spatialChan = patchArea * 3
        var f = [Float](repeating: 0, count: d * patchChan)
        for o in 0..<d {
            let ob = o * patchChan
            for t in 0..<temporal {
                for raster in 0..<patchArea {
                    for c in 0..<3 {
                        f[ob + c * temporalArea + t * patchArea + raster] =
                            native[ob + t * spatialChan + raster * 3 + c]
                    }
                }
            }
        }
        return f
    }

    static func countBlocks(in ck: QWVisionCheckpoint) throws -> Int {
        var blkIdx = 0
        while ck.has("blocks.\(blkIdx).norm1.weight") { blkIdx += 1 }
        guard blkIdx > 0 else { throw QWVisionError.missingTensor("blocks.0.norm1.weight") }
        return blkIdx
    }

    static func resolveFc1Mid(in ck: QWVisionCheckpoint, config cfg: QWVisionConfig) -> Int {
        ck.shape("merger.linear_fc1.bias")?[0]
            ?? ck.shape("merger.linear_fc1.weight")?[0]
            ?? cfg.effectiveMergerIntermediateDim
    }

    static func resolveFc2Out(in ck: QWVisionCheckpoint, config cfg: QWVisionConfig) -> Int {
        ck.shape("merger.linear_fc2.bias")?[0]
            ?? ck.shape("merger.linear_fc2.weight")?[0]
            ?? cfg.outDim
    }

    static func computeInvFreq(config cfg: QWVisionConfig) -> [Float] {
        var f = [Float](repeating: 0, count: cfg.invFreqCount)
        for i in 0..<f.count {
            f[i] = Float(1.0 / pow(cfg.ropeTheta, Double(2 * i) / Double(cfg.spatial)))
        }
        return f
    }

    init(checkpoint ck: QWVisionCheckpoint) throws {
        self.ck = ck
        let cfg = ck.config
        self.config = cfg
        let d = cfg.dim
        let merged = cfg.merged
        self.posEmbed = try ck.f32("pos_embed.weight", want: [cfg.numPosEmbeds, d])
        self.patchW = try Self.loadFoldedPatchWeight(from: ck, config: cfg)
        self.patchB = try ck.f32("patch_embed.proj.bias", want: [d])
        let nBlocks = try Self.countBlocks(in: ck)
        self.blocks = try (0..<nBlocks).map { try QWVisionBlockW(ck, "blocks.\($0)", config: cfg) }
        guard ck.has("merger.norm.weight"), ck.has("merger.norm.bias") else {
            throw QWVisionError.badLayout("merger.norm non-affine law violated (affine weights required)")
        }
        self.mergerGamma = try ck.f32("merger.norm.weight", want: [d])
        self.mergerBeta = try ck.f32("merger.norm.bias", want: [d])
        let fc1Mid = Self.resolveFc1Mid(in: ck, config: cfg)
        self.fc1Mid = fc1Mid
        self.fc1W = try ck.f32("merger.linear_fc1.weight", want: [fc1Mid, merged])
        self.fc1B = (cfg.projectorBias && ck.has("merger.linear_fc1.bias"))
            ? try ck.f32("merger.linear_fc1.bias", want: [fc1Mid])
            : (cfg.projectorBias
                ? try ck.f32("merger.linear_fc1.bias", want: [fc1Mid])
                : [Float](repeating: 0, count: fc1Mid))
        let outDim = Self.resolveFc2Out(in: ck, config: cfg)
        self.fc2Out = outDim
        self.fc2W = try ck.f32("merger.linear_fc2.weight", want: [outDim, fc1Mid])
        self.fc2B = (cfg.projectorBias && ck.has("merger.linear_fc2.bias"))
            ? try ck.f32("merger.linear_fc2.bias", want: [outDim])
            : (cfg.projectorBias
                ? try ck.f32("merger.linear_fc2.bias", want: [outDim])
                : [Float](repeating: 0, count: outDim))
        self.invFreq = Self.computeInvFreq(config: cfg)
        ck.releaseBacking()
    }
}

public final class QWVisionBlockW: @unchecked Sendable {
    let norm1G: [Float], norm1B: [Float], norm2G: [Float], norm2B: [Float]
    let qkvW: [Float], qkvB: [Float]        // [qkvDim, dim] fused q‖k‖v
    let projW: [Float], projB: [Float]
    let l1W: [Float], l1B: [Float]          // [mlpInner, dim]
    let l2W: [Float], l2B: [Float]          // [dim, mlpInner]
    init(_ ck: QWVisionCheckpoint, _ p: String, config: QWVisionConfig) throws {
        let d = config.dim
        let mlp = config.mlpInner
        norm1G = try ck.f32("\(p).norm1.weight", want: [d]); norm1B = try ck.f32("\(p).norm1.bias", want: [d])
        norm2G = try ck.f32("\(p).norm2.weight", want: [d]); norm2B = try ck.f32("\(p).norm2.bias", want: [d])
        qkvW = try ck.f32("\(p).attn.qkv.weight", want: [config.qkvDim, d])
        qkvB = try ck.f32("\(p).attn.qkv.bias", want: [config.qkvDim])
        projW = try ck.f32("\(p).attn.proj.weight", want: [d, d])
        projB = try ck.f32("\(p).attn.proj.bias", want: [d])
        l1W = try ck.f32("\(p).mlp.linear_fc1.weight", want: [mlp, d])
        l1B = try ck.f32("\(p).mlp.linear_fc1.bias", want: [mlp])
        l2W = try ck.f32("\(p).mlp.linear_fc2.weight", want: [d, mlp])
        l2B = try ck.f32("\(p).mlp.linear_fc2.bias", want: [d])
    }
}

// MARK: - the forward stages (each mirrors one golden stage, law-numbered)

/// Per-grid geometry the stages consume (the C19 golden bins + coords are
/// RE-READ from disk by the test harness — the reference consumes THE SAME
/// bytes the MSL kernel will gather: one table, three consumers).
public struct QWVisionGridPlan: Sendable {
    public let thw: (t: Int, h: Int, w: Int)
    public let nPatch: Int              // t·h·w
    public let nToken: Int              // nPatch / merge²
    public let frameLen: Int            // h·w (per-FRAME segment length, law 5)
    public init(t: Int, h: Int, w: Int, merge: Int = 2) {
        self.thw = (t, h, w)
        self.nPatch = t * h * w
        self.nToken = t * h * w / (merge * merge)
        self.frameLen = h * w
    }
}

public final class QWVisionRef: @unchecked Sendable {
    public let w: QWVisionWeights
    private let profileStages = ProcessInfo.processInfo.environment["QW_VISION_PROFILE"] == "1"

    public init(weights: QWVisionWeights) { self.w = weights }

    @inline(__always)
    private func profileStart() -> UInt64 {
        profileStages ? DispatchTime.now().uptimeNanoseconds : 0
    }

    @inline(__always)
    private func profileEnd(_ stage: String, since start: UInt64) {
        guard profileStages else { return }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
        print(String(format: "vision_ref_stage_ms=%.3f stage=%@", milliseconds, stage))
    }

    /// C19-golden bins re-read from disk (the strict-interp law: the
    /// reference and the Metal kernel gather identical tables).
    public struct InterpBins: Sendable {
        public let idx: [Int32]         // [nPatch, 4] i32 (fit asserted in C19)
        public let wgt: [Float]         // [nPatch, 4] f32
        public init(idx: [Int32], wgt: [Float]) {
            precondition(idx.count == Int(idx.count / 4) * 4 && wgt.count == idx.count, "idx and wgt must have matching lengths divisible by 4")
            self.idx = idx
            self.wgt = wgt
        }
        public init(outDir: String, tag: String) throws {
            guard let ib = try? Data(contentsOf: URL(fileURLWithPath: "\(outDir)/mm_interp_idx_\(tag).bin")),
                  let wb = try? Data(contentsOf: URL(fileURLWithPath: "\(outDir)/mm_interp_w_\(tag).bin"))
            else { throw QWVisionError.goldenMissing("mm_interp_\(tag)") }
            self.idx = ib.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw.bindMemory(to: Int32.self)) }
            self.wgt = wb.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw.bindMemory(to: Float.self)) }
            guard idx.count == Int(idx.count / 4) * 4, wgt.count == idx.count else {
                throw QWVisionError.goldenMissing("mm_interp_\(tag) shape")
            }
        }
        public var nPatch: Int { Int(idx.count / 4) }
    }

    /// Full reference forward: patch slab `[nPatch, patchChannels]` -> `[nToken, fc2Out]`.
    public func encode(patches x: [Float], plan: QWVisionGridPlan,
                       bins: InterpBins, posIds: [Int32]) -> [Float] {
        let c = w.config
        let n = plan.nPatch, d = c.dim
        precondition(x.count == n * c.patchChannels && bins.nPatch == n && posIds.count == 2 * n)
        guard n > 0, d > 0 else { return [] }
        let encodeStart = profileStart()
        var stageStart = profileStart()
        var s = [Float](repeating: 0, count: n * d)
        s.withUnsafeMutableBufferPointer { sb in
            guard let sPtr = sb.baseAddress else { return }
            qwGemmRef(y: sPtr, ldY: d, yBase: 0,
                      x: x, rows: n, w: w.patchW, nOut: d, nIn: c.patchChannels, bias: w.patchB)
        }
        profileEnd("patch_embed", since: stageStart)
        stageStart = profileStart()
        foldPosEmbed(into: &s, bins: bins)
        profileEnd("position_embedding", since: stageStart)
        stageStart = profileStart()
        let (cos, sin) = ropeCosSin(posIds: posIds, n: n)
        profileEnd("rope_tables", since: stageStart)
        var cu = [Int](repeating: 0, count: plan.thw.t + 1)
        for f in 0..<plan.thw.t { cu[f + 1] = cu[f] + plan.frameLen }
        let maxSeg = max(1, plan.frameLen), attnWorkers = c.heads
        var normed = [Float](repeating: 0, count: n * d)
        var qkv = [Float](repeating: 0, count: n * c.qkvDim)
        var attnOut = [Float](repeating: 0, count: n * d)
        var mlpH = [Float](repeating: 0, count: n * c.mlpInner)
        var mlpScratch = [Float](repeating: 0, count: n * c.mlpInner)
        var attnQ = [Float](repeating: 0, count: attnWorkers * maxSeg * c.headDim)
        var attnK = [Float](repeating: 0, count: attnWorkers * maxSeg * c.headDim)
        var attnScores = [Float](repeating: 0, count: attnWorkers * maxSeg * maxSeg)
        for (index, blk) in w.blocks.enumerated() {
            stageStart = profileStart()
            block(&s, blk, plan: plan, cu: cu, cos: cos, sin: sin,
                  profileTag: profileStages ? "block\(index)" : nil,
                  normedBuf: &normed, qkvBuf: &qkv, attnBuf: &attnOut,
                  mlpHBuf: &mlpH, mlpScratchBuf: &mlpScratch,
                  attnQBuf: &attnQ, attnKBuf: &attnK, attnScoresBuf: &attnScores, maxSeg: maxSeg)
            profileEnd("block\(index)", since: stageStart)
        }
        stageStart = profileStart()
        merger(&s, plan: plan)
        profileEnd("merger", since: stageStart)
        profileEnd("encode_total", since: encodeStart)
        return s
    }

    private func foldPosEmbed(into s: inout [Float], bins: InterpBins) {
        let d = w.config.dim
        let n = bins.nPatch
        guard n > 0, d > 0 else { return }
        let numPos = w.posEmbed.count / d
        guard numPos > 0 else { return }
        let vDim = vDSP_Length(d)
        bins.idx.withUnsafeBufferPointer { ib in
            bins.wgt.withUnsafeBufferPointer { wb in
                w.posEmbed.withUnsafeBufferPointer { tb in
                    s.withUnsafeMutableBufferPointer { sb in
                        guard let sBase = sb.baseAddress, let tBase = tb.baseAddress,
                              let iBase = ib.baseAddress, let wBase = wb.baseAddress else { return }
                        for p in 0..<n {
                            let sRow = sBase + p * d
                            for c in 0..<4 {
                                let rawIdx = Int(iBase[p * 4 + c])
                                guard rawIdx >= 0, rawIdx < numPos else { continue }
                                var wc = wBase[p * 4 + c]
                                let tc = tBase + rawIdx * d
                                vDSP_vsma(tc, 1, &wc, sRow, 1, sRow, 1, vDim)
                            }
                        }
                    }
                }
            }
        }
    }

    /// stage-2 rope: [n,2] coords → cos/sin [n, headDim].
    private func ropeCosSin(posIds: [Int32], n: Int) -> (cos: [Float], sin: [Float]) {
        let cfg = w.config
        let hd = cfg.headDim
        let inv = cfg.invFreqCount
        var cos = [Float](repeating: 0, count: n * hd)
        var sin = [Float](repeating: 0, count: n * hd)
        w.invFreq.withUnsafeBufferPointer { fb in
            for p in 0..<n {
                let h = Float(posIds[p * 2]), wd = Float(posIds[p * 2 + 1])
                for i in 0..<inv {
                    let ah = h * fb[i], aw = wd * fb[i]
                    let c = cosf(ah), s = sinf(ah), c2 = cosf(aw), s2 = sinf(aw)
                    let o = p * hd + i
                    cos[o] = c;    cos[o + inv] = c2;  cos[o + 2 * inv] = c;   cos[o + 3 * inv] = c2
                    sin[o] = s;    sin[o + inv] = s2;  sin[o + 2 * inv] = s;   sin[o + 3 * inv] = s2
                }
            }
        }
        return (cos, sin)
    }

    private func block(_ s: inout [Float], _ blk: QWVisionBlockW, plan: QWVisionGridPlan,
                       cu: [Int], cos: [Float], sin: [Float], profileTag: String?,
                       normedBuf: inout [Float], qkvBuf: inout [Float], attnBuf: inout [Float],
                       mlpHBuf: inout [Float], mlpScratchBuf: inout [Float],
                       attnQBuf: inout [Float], attnKBuf: inout [Float], attnScoresBuf: inout [Float], maxSeg: Int) {
        let cfg = w.config
        let n = plan.nPatch, d = cfg.dim, qkvDim = cfg.qkvDim, mlpInner = cfg.mlpInner
        let heads = cfg.heads, hd = cfg.headDim
        guard n > 0, d > 0, heads > 0, hd > 0, qkvDim > 0, mlpInner > 0 else { return }
        var stageStart = profileStart()
        qwLayerNorm(s, into: &normedBuf, gamma: blk.norm1G, beta: blk.norm1B, rows: n, dim: d, eps: cfg.eps)
        profileEnd("\(profileTag ?? "block").norm1", since: stageStart)
        stageStart = profileStart()
        qkvBuf.withUnsafeMutableBufferPointer { qb in
            guard let qPtr = qb.baseAddress else { return }
            qwGemmRef(y: qPtr, ldY: qkvDim, yBase: 0,
                      x: normedBuf, rows: n, w: blk.qkvW, nOut: qkvDim, nIn: d, bias: blk.qkvB)
        }
        profileEnd("\(profileTag ?? "block").qkv", since: stageStart)
        let scale: Float = Float(1.0 / Double(hd).squareRoot())
        let vOffset = 2 * d
        stageStart = profileStart()
        qkvBuf.withUnsafeBufferPointer { qkvB in
            cos.withUnsafeBufferPointer { cosB in
                sin.withUnsafeBufferPointer { sinB in
                    attnBuf.withUnsafeMutableBufferPointer { ab in
                        attnQBuf.withUnsafeMutableBufferPointer { aqb in
                            attnKBuf.withUnsafeMutableBufferPointer { akb in
                                attnScoresBuf.withUnsafeMutableBufferPointer { ascb in
                                    guard let qkvBaseRaw = qkvB.baseAddress, let cosBaseRaw = cosB.baseAddress,
                                          let sinBaseRaw = sinB.baseAddress, let attnBaseRaw = ab.baseAddress,
                                          let qWorkBaseRaw = aqb.baseAddress, let kWorkBaseRaw = akb.baseAddress,
                                          let scWorkBaseRaw = ascb.baseAddress else { return }
                                    nonisolated(unsafe) let qkvBase = qkvBaseRaw, cosBase = cosBaseRaw
                                    nonisolated(unsafe) let sinBase = sinBaseRaw, attnBase = attnBaseRaw
                                    nonisolated(unsafe) let qWorkBase = qWorkBaseRaw, kWorkBase = kWorkBaseRaw
                                    nonisolated(unsafe) let scWorkBase = scWorkBaseRaw
                                    for seg in 0..<(cu.count - 1) {
                                        let lo = cu[seg], hi = cu[seg + 1], m = hi - lo
                                        guard m > 0 else { continue }
                                        let runHead: @Sendable (Int, Int) -> Void = { (head: Int, slot: Int) in
                                            let qPtr = qWorkBase + slot * maxSeg * hd
                                            let kPtr = kWorkBase + slot * maxSeg * hd
                                            let scPtr = scWorkBase + slot * maxSeg * maxSeg
                                            let half = hd / 2
                                            for j in 0..<m {
                                                let p = lo + j
                                                let qSrc = qkvBase + p * qkvDim + head * hd, kSrc = qSrc + d
                                                let cRow = cosBase + p * hd, sRow = sinBase + p * hd
                                                let qDst = qPtr + j * hd, kDst = kPtr + j * hd
                                                for i in 0..<half {
                                                    let c0 = cRow[i], s0 = sRow[i], c1 = cRow[i + half], s1 = sRow[i + half]
                                                    qDst[i] = qSrc[i] * c0 - qSrc[i + half] * s0
                                                    qDst[i + half] = qSrc[i + half] * c1 + qSrc[i] * s1
                                                    kDst[i] = kSrc[i] * c0 - kSrc[i + half] * s0
                                                    kDst[i + half] = kSrc[i + half] * c1 + kSrc[i] * s1
                                                }
                                            }
                                            qw_cblas_sgemm(101, 111, 112, Int32(m), Int32(m), Int32(hd), scale,
                                                           qPtr, Int32(hd), kPtr, Int32(hd), 0.0, scPtr, Int32(m))
                                            var m32 = Int32(m), vM = vDSP_Length(m)
                                            for qi in 0..<m {
                                                let row = scPtr + qi * m
                                                var mx: Float = -.greatestFiniteMagnitude
                                                vDSP_maxv(row, 1, &mx, vM)
                                                guard mx.isFinite else {
                                                    vDSP_vclr(row, 1, vM)
                                                    continue
                                                }
                                                var negMx = -mx
                                                vDSP_vsadd(row, 1, &negMx, row, 1, vM)
                                                vvexpf(row, row, &m32)
                                                var sum: Float = 0
                                                vDSP_sve(row, 1, &sum, vM)
                                                guard sum > 0, sum.isFinite else {
                                                    vDSP_vclr(row, 1, vM)
                                                    continue
                                                }
                                                var inv = 1.0 / sum
                                                vDSP_vsmul(row, 1, &inv, row, 1, vM)
                                            }
                                            let vStart = qkvBase + lo * qkvDim + vOffset + head * hd
                                            let outStart = attnBase + lo * d + head * hd
                                            qw_cblas_sgemm(101, 111, 111, Int32(m), Int32(hd), Int32(m), 1.0,
                                                           scPtr, Int32(m), vStart, Int32(qkvDim), 0.0, outStart, Int32(d))
                                        }
                                        DispatchQueue.concurrentPerform(iterations: heads) { head in runHead(head, head) }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        profileEnd("\(profileTag ?? "block").attention", since: stageStart)
        stageStart = profileStart()
        normedBuf.withUnsafeMutableBufferPointer { pb in
            guard let pPtr = pb.baseAddress else { return }
            qwGemmRef(y: pPtr, ldY: d, yBase: 0,
                      x: attnBuf, rows: n, w: blk.projW, nOut: d, nIn: d, bias: blk.projB)
        }
        s.withUnsafeMutableBufferPointer { sb in
            normedBuf.withUnsafeBufferPointer { pb in
                guard let sPtr = sb.baseAddress, let pPtr = pb.baseAddress else { return }
                vDSP_vadd(sPtr, 1, pPtr, 1, sPtr, 1, vDSP_Length(n * d))
            }
        }
        profileEnd("\(profileTag ?? "block").attention_projection", since: stageStart)
        stageStart = profileStart()
        qwLayerNorm(s, into: &normedBuf, gamma: blk.norm2G, beta: blk.norm2B, rows: n, dim: d, eps: cfg.eps)
        profileEnd("\(profileTag ?? "block").norm2", since: stageStart)
        stageStart = profileStart()
        let mlpCount = n * mlpInner
        mlpHBuf.withUnsafeMutableBufferPointer { hb in
            guard let hPtr = hb.baseAddress else { return }
            qwGemmRef(y: hPtr, ldY: mlpInner, yBase: 0,
                      x: normedBuf, rows: n, w: blk.l1W, nOut: mlpInner, nIn: d, bias: blk.l1B)
            mlpScratchBuf.withUnsafeMutableBufferPointer { scb in
                guard let scPtr = scb.baseAddress else { return }
                qwApplyGeluTanhInPlace(hPtr, count: mlpCount, scratch: scPtr)
            }
        }
        normedBuf.withUnsafeMutableBufferPointer { yb in
            guard let yPtr = yb.baseAddress else { return }
            qwGemmRef(y: yPtr, ldY: d, yBase: 0,
                      x: mlpHBuf, rows: n, w: blk.l2W, nOut: d, nIn: mlpInner, bias: blk.l2B)
        }
        s.withUnsafeMutableBufferPointer { sb in
            normedBuf.withUnsafeBufferPointer { yb in
                guard let sPtr = sb.baseAddress, let yPtr = yb.baseAddress else { return }
                vDSP_vadd(sPtr, 1, yPtr, 1, sPtr, 1, vDSP_Length(n * d))
            }
        }
        profileEnd("\(profileTag ?? "block").mlp", since: stageStart)
    }

    /// Merger: LN per patch row -> view-merge merge² -> fc1 -> GELU-ERF -> fc2 -> [nToken, fc2Out].
    func merger(_ s: inout [Float], plan: QWVisionGridPlan) {
        let cfg = w.config
        let n = plan.nPatch, d = cfg.dim, nt = plan.nToken, merged = cfg.merged, fc1Mid = w.fc1Mid
        guard n > 0, d > 0, nt > 0, merged > 0, fc1Mid > 0, w.fc2Out > 0 else { s = []; return }
        var normed = [Float](repeating: 0, count: n * d)
        qwLayerNorm(s, into: &normed, gamma: w.mergerGamma, beta: w.mergerBeta, rows: n, dim: d, eps: cfg.eps)
        var h = [Float](repeating: 0, count: nt * fc1Mid)
        var y = [Float](repeating: 0, count: nt * w.fc2Out)
        h.withUnsafeMutableBufferPointer { hb in
            guard let hPtr = hb.baseAddress else { return }
            qwGemmRef(y: hPtr, ldY: fc1Mid, yBase: 0,
                      x: normed, rows: nt, w: w.fc1W, nOut: fc1Mid, nIn: merged,
                      bias: w.fc1B, act: qwGeluErf)
        }
        y.withUnsafeMutableBufferPointer { yb in
            guard let yPtr = yb.baseAddress else { return }
            qwGemmRef(y: yPtr, ldY: w.fc2Out, yBase: 0,
                      x: h, rows: nt, w: w.fc2W, nOut: w.fc2Out, nIn: fc1Mid,
                      bias: w.fc2B)
        }
        s = y
    }
}
