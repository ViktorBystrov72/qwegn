// QWMM.swift — Multimodal preprocessing math, grid geometry, and X0 plan (spec-05).
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWToken
import QWVision

/// Config-sourced multimodal tokenization and patch geometry.
public struct QWMMSpec: Sendable, Equatable, Hashable {
    public let patch: Int
    public let temporal: Int
    public let merge: Int
    public let posSide: Int
    public let imageToken: Int
    public let videoToken: Int
    public let minPixels: Int
    public let maxPixels: Int

    public var factor: Int { patch * merge }
    public var patchArea: Int { patch * patch }
    public var temporalPatchArea: Int { temporal * patchArea }
    public var patchChannels: Int { 3 * temporalPatchArea }

    public init(
        patch: Int,
        temporal: Int,
        merge: Int,
        posSide: Int,
        imageToken: Int,
        videoToken: Int,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216
    ) {
        self.patch = patch
        self.temporal = temporal
        self.merge = merge
        self.posSide = posSide
        self.imageToken = imageToken
        self.videoToken = videoToken
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    public init(
        visionConfig: QWVisionConfig,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216
    ) {
        self.init(
            patch: visionConfig.patch,
            temporal: visionConfig.temporal,
            merge: visionConfig.merge,
            posSide: visionConfig.posSide,
            imageToken: visionConfig.imageTokenId,
            videoToken: visionConfig.videoTokenId,
            minPixels: minPixels,
            maxPixels: maxPixels
        )
    }

    public init(
        descriptor: QWModelDescriptor,
        minPixels: Int = 65_536,
        maxPixels: Int = 16_777_216
    ) {
        self.init(
            visionConfig: QWVisionConfig(descriptor: descriptor),
            minPixels: minPixels,
            maxPixels: maxPixels
        )
    }

    public static let qwen3VLReference = QWMMSpec(visionConfig: .qwen3VLReference)
}

// MARK: - A: smart_resize (image_processing_qwen2_vl_fast.smart_resize port)

public enum QWGridError: Error, Equatable {
    case aspectRatio(max: Int, min: Int)
    case mergeShape(t: Int, h: Int, w: Int)
    case placeholderOverflow
    case pixelDimensions(expectedH: Int, expectedW: Int, actualH: Int, actualW: Int)
    case bufferLength(expected: Int, actual: Int)
}

/// Banker's rounding (round-half-to-even) matching Python's `round(x)`.
func pyRound(_ x: Double) -> Double {
    let f = x.rounded(.down)
    let frac = x - f
    if frac > 0.5 { return f + 1 }
    if frac < 0.5 { return f }
    let i = Int(f)
    return Double(i % 2 == 0 ? i : i + 1)
}

public struct QWVisionGrid: Equatable, Sendable {
    public let t: Int, h: Int, w: Int
    public let spec: QWMMSpec
    public init(t: Int, h: Int, w: Int, spec: QWMMSpec = .qwen3VLReference) throws {
        guard h % spec.merge == 0, w % spec.merge == 0 else {
            throw QWGridError.mergeShape(t: t, h: h, w: w)
        }
        self.t = t; self.h = h; self.w = w; self.spec = spec
    }
    public var patches: Int { t * h * w }
    public var tokens: Int { patches / (spec.merge * spec.merge) }
    public var plan: QWVisionGridPlan {
        QWVisionGridPlan(t: t, h: h, w: w, merge: spec.merge)
    }
    public var interpBins: QWVisionRef.InterpBins {
        let taps = qwInterpTaps(grid: self)
        return QWVisionRef.InterpBins(idx: taps.indices, wgt: taps.weights)
    }
    public var ropePositions: [Int32] {
        qwVisionRopePositions(grid: self)
    }
}

public enum QWMMConfig {
    public static let patch = QWMMSpec.qwen3VLReference.patch
    public static let temporal = QWMMSpec.qwen3VLReference.temporal
    public static let merge = QWMMSpec.qwen3VLReference.merge
    public static let factor = QWMMSpec.qwen3VLReference.factor
    public static let minPixels = QWMMSpec.qwen3VLReference.minPixels
    public static let maxPixels = QWMMSpec.qwen3VLReference.maxPixels
    public static let posSide = QWMMSpec.qwen3VLReference.posSide
    public static let imageToken = QWMMSpec.qwen3VLReference.imageToken
    public static let videoToken = QWMMSpec.qwen3VLReference.videoToken

    /// Aspect check (< 200), factor-rounding, and pixel-budget clamp.
    public static func smartResize(
        height: Int,
        width: Int,
        maxPixels: Int? = nil,
        spec: QWMMSpec = .qwen3VLReference
    ) throws -> (h: Int, w: Int) {
        guard Double(max(height, width)) / Double(min(height, width)) < 200 else {
            throw QWGridError.aspectRatio(max: max(height, width), min: min(height, width))
        }
        let factor = Double(spec.factor)
        let budgetMax = maxPixels ?? spec.maxPixels
        let budgetMin = spec.minPixels
        var hBar = pyRound(Double(height) / factor) * factor
        var wBar = pyRound(Double(width) / factor) * factor
        if hBar * wBar > Double(budgetMax) {
            let beta = (Double(height * width) / Double(budgetMax)).squareRoot()
            hBar = max(factor, (Double(height) / beta / factor).rounded(.down) * factor)
            wBar = max(factor, (Double(width) / beta / factor).rounded(.down) * factor)
        } else if hBar * wBar < Double(budgetMin) {
            let beta = (Double(budgetMin) / Double(height * width)).squareRoot()
            hBar = (Double(height) * beta / factor).rounded(.up) * factor
            wBar = (Double(width) * beta / factor).rounded(.up) * factor
        }
        return (Int(hBar), Int(wBar))
    }
}

// MARK: - D: pos-embed bilinear interpolation (vision_utils port, exact taps)

/// Bilinear align_corners=True axis taps into a `side`-element embedding grid.
func axisTaps(_ index: Int, size: Int, side: Int) -> (taps: [Int], weights: [Float]) {
    let denom = max(Float(size - 1), 1)
    let src = Float(index) * Float(side - 1) / denom
    let floorv = src.rounded(.down)
    let base = Int(floorv)
    var taps = [Int](), ws = [Float]()
    for off in [0, 1] {
        taps.append(min(max(base + off, 0), side - 1))
        let d = abs(src - floorv - Float(off))
        ws.append(max(1 - d, 0))
    }
    return (taps, ws)
}

/// Golden-law interpolation table in merge-block order into `side x side` pos_embed.
public func qwInterpTaps(grid: QWVisionGrid) -> (indices: [Int32], weights: [Float]) {
    let m = grid.spec.merge
    let side = grid.spec.posSide
    let blocksW = grid.w / m
    var indices = [Int32](); var weights = [Float]()
    indices.reserveCapacity(grid.patches * 4); weights.reserveCapacity(grid.patches * 4)
    for _ in 0..<grid.t {
        for within in 0..<(grid.h * grid.w) {
            let inCol = within % m
            let inRow = (within / m) % m
            let blockCol = (within / (m * m)) % blocksW
            let blockRow = within / (m * m * blocksW)
            let row = blockRow * m + inRow
            let col = blockCol * m + inCol
            let (ht, hw) = axisTaps(row, size: grid.h, side: side)
            let (wt, ww) = axisTaps(col, size: grid.w, side: side)
            for hh in 0..<2 { for wc in 0..<2 {
                indices.append(Int32(ht[hh] * side + wt[wc]))
                weights.append(hw[hh] * ww[wc])
            }}
        }
    }
    return (indices, weights)
}

// MARK: - C/E: vision rope positions + per-frame segments

/// 2-axis (h,w) rope coords in merge-block patch order, repeated t times.
public func qwVisionRopePositions(grid: QWVisionGrid) -> [Int32] {
    let m = grid.spec.merge
    var hw = [Int32](); hw.reserveCapacity(grid.h * grid.w * 2)
    for hb in 0..<(grid.h / m) { for wb in 0..<(grid.w / m) {
        for i in 0..<m { for j in 0..<m {
            hw.append(Int32(hb * m + i)); hw.append(Int32(wb * m + j))
        }}} }
    var out = [Int32](); out.reserveCapacity(grid.patches * 2)
    for _ in 0..<grid.t { out.append(contentsOf: hw) }
    return out
}

// MARK: - F: mRope 3D positions

public struct QWMMBlock: Equatable, Sendable {
    public enum Kind: Equatable, Sendable { case text, image, video }
    public let kind: Kind
    public let tokens: [Int]
    public let grid: QWVisionGrid?
    public init(kind: Kind, tokens: [Int], grid: QWVisionGrid?) { self.kind = kind; self.tokens = tokens; self.grid = grid }
}

/// Builds 3-plane (t, h, w) mRope coordinates over text, image, and video blocks.
public func qwMRopePositions(blocks: [QWMMBlock], merge: Int? = nil) -> (pos: [Int32], delta: Int) {
    var S = 0; for b in blocks { S += b.tokens.count }
    var pT = [Int32](), pH = [Int32](), pW = [Int32]()
    pT.reserveCapacity(S); pH.reserveCapacity(S); pW.reserveCapacity(S)
    var cur = 0
    for b in blocks {
        switch b.kind {
        case .text:
            for i in 0..<b.tokens.count {
                let p = Int32(cur + i); pT.append(p); pH.append(p); pW.append(p)
            }
            cur += b.tokens.count
        case .image:
            guard let g = b.grid else { preconditionFailure("qwMRopePositions: image block missing grid") }
            let m = merge ?? g.spec.merge
            emitCube(t: g.t, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
            cur += max(g.h, g.w) / m
        case .video:
            guard let g = b.grid else { preconditionFailure("qwMRopePositions: video block missing grid") }
            let m = merge ?? g.spec.merge
            precondition(b.tokens.count == g.tokens, "video block token count ≠ grid tokens")
            for _ in 0..<g.t {
                emitCube(t: 1, h: g.h, w: g.w, merge: m, cur: cur, &pT, &pH, &pW)
                cur += max(g.h, g.w) / m
            }
        }
    }
    precondition(pT.count == S, "span walk lost tokens")
    var mx = Int32.min
    for i in 0..<S { mx = max(mx, max(pT[i], max(pH[i], pW[i]))) }
    var pos = [Int32](); pos.reserveCapacity(3 * S)
    pos.append(contentsOf: pT); pos.append(contentsOf: pH); pos.append(contentsOf: pW)
    return (pos, Int(mx) + 1 - S)
}

private func emitCube(t lt: Int, h hh: Int, w ww: Int, merge m: Int, cur: Int,
                      _ pT: inout [Int32], _ pH: inout [Int32], _ pW: inout [Int32]) {
    let lh = hh / m, lw = ww / m
    for i in 0..<lt {
        let hArange = (0..<lh).map { Int32(cur + $0) }
        let wArange = (0..<lw).map { Int32(cur + $0) }
        for hv in hArange {
            for wv in wArange {
                pT.append(Int32(cur + i)); pH.append(hv); pW.append(wv)
            }
        }
    }
}

// MARK: - B: patch-slab builder

private func buildPatchSlab(
    grid: QWVisionGrid,
    height: Int,
    width: Int,
    sampleCount: Int,
    readValue: (Int) -> Float
) throws -> [Float] {
    let spec = grid.spec
    let expectedH = grid.h * spec.patch
    let expectedW = grid.w * spec.patch
    guard height == expectedH && width == expectedW else {
        throw QWGridError.pixelDimensions(expectedH: expectedH, expectedW: expectedW, actualH: height, actualW: width)
    }
    let expectedCount = height * width * 3
    guard sampleCount == expectedCount else {
        throw QWGridError.bufferLength(expected: expectedCount, actual: sampleCount)
    }

    let m = spec.merge
    let patch = spec.patch
    let factor = spec.factor
    let patchArea = spec.patchArea
    let temporal = spec.temporal
    let temporalArea = spec.temporalPatchArea
    let patchChannels = spec.patchChannels
    let cellsPerBlock = m * m
    let blocksW = grid.w / m
    let totalPatches = grid.patches
    var out = [Float](repeating: 0, count: totalPatches * patchChannels)

    out.withUnsafeMutableBufferPointer { dst in
        for pi in 0..<totalPatches {
            let hb = pi / (cellsPerBlock * blocksW)
            let rem = pi % (cellsPerBlock * blocksW)
            let wb = rem / cellsPerBlock
            let p = rem % cellsPerBlock
            let i = p / m
            let j = p % m
            let tp = hb * factor + i * patch
            let wp = wb * factor + j * patch

            let rowOffset = pi * patchChannels
            for c in 0..<3 {
                let channelOffset = rowOffset + c * temporalArea
                for r in 0..<patch {
                    let srcRowStart = (tp + r) * width * 3 + wp * 3 + c
                    let dstRasterOffset = channelOffset + r * patch
                    for col in 0..<patch {
                        let val = readValue(srcRowStart + col * 3)
                        for t in 0..<temporal {
                            dst[dstRasterOffset + t * patchArea + col] = val
                        }
                    }
                }
            }
        }
    }
    return out
}

/// Construct the patch-slab layout from normalized Float RGB pixels [-1, 1].
public func qwBuildPatchSlab(normalizedRGB: [Float], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    try normalizedRGB.withUnsafeBufferPointer { src in
        try buildPatchSlab(grid: grid, height: height, width: width, sampleCount: normalizedRGB.count) { idx in
            src[idx]
        }
    }
}

/// Construct the patch-slab layout from raw UInt8 RGB pixels [0, 255], normalizing to [-1, 1].
public func qwBuildPatchSlab(imageRGB: [UInt8], height: Int, width: Int, grid: QWVisionGrid) throws -> [Float] {
    try imageRGB.withUnsafeBufferPointer { src in
        try buildPatchSlab(grid: grid, height: height, width: width, sampleCount: imageRGB.count) { idx in
            (Float(src[idx]) / 255.0 - 0.5) / 0.5
        }
    }
}

// MARK: - A+G: placeholder expansion + the X₀ slab plan

/// One span of the X₀ slab: either a text-gather span or a merged vision block.
public struct QWX0Span: Equatable, Sendable {
    public enum Content: Equatable, Sendable {
        case text(tokenStart: Int, count: Int)
        case vision(grid: QWVisionGrid, patches: Int, rows: Int)
    }
    public let content: Content
    public let rowStart: Int
    public var rowCount: Int {
        switch content { case .text(_, let c): return c; case .vision(_, _, let r): return r }
    }
}

public struct QWX0Plan: Equatable, Sendable {
    public let tokens: [Int]
    public let spans: [QWX0Span]
    public let positions: [Int32]
    public let mropeDelta: Int
    public var rows: Int { spans.reduce(0) { $0 + $1.rowCount } }

    /// Expand raw media placeholder tokens to `t*h*w/merge²` rows and compute 3D mRope positions.
    public init(
        tokens: [Int],
        images: [QWVisionGrid],
        videos: [QWVisionGrid],
        spec: QWMMSpec = .qwen3VLReference
    ) throws {
        var out: [Int] = []
        var spans: [QWX0Span] = []
        var blocks: [QWMMBlock] = []
        var ii = 0, vi = 0, row = 0
        var textRun: [Int] = []
        func flushText() {
            guard !textRun.isEmpty else { return }
            spans.append(QWX0Span(content: .text(tokenStart: out.count, count: textRun.count), rowStart: row))
            blocks.append(QWMMBlock(kind: .text, tokens: textRun, grid: nil))
            out.append(contentsOf: textRun); row += textRun.count; textRun = []
        }
        for t in tokens {
            if t == spec.imageToken {
                guard ii < images.count else { throw QWGridError.placeholderOverflow }
                let g = images[ii]; ii += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                blocks.append(QWMMBlock(kind: .image, tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else if t == spec.videoToken {
                guard vi < videos.count else { throw QWGridError.placeholderOverflow }
                let g = videos[vi]; vi += 1
                flushText()
                let exp = g.tokens
                spans.append(QWX0Span(content: .vision(grid: g, patches: g.patches, rows: exp), rowStart: row))
                blocks.append(QWMMBlock(kind: .video, tokens: Array(repeating: t, count: exp), grid: g))
                out.append(contentsOf: Array(repeating: t, count: exp)); row += exp
            } else {
                textRun.append(t)
            }
        }
        flushText()
        let rope = qwMRopePositions(blocks: blocks, merge: spec.merge)
        precondition(rope.pos.count == 3 * out.count)
        self.tokens = out; self.spans = spans; self.positions = rope.pos; self.mropeDelta = rope.delta
    }
}

// MARK: - X₀ Slab Interleaving & Vision Pipeline

/// Interleave text embeddings and merged visual tokens into the contiguous X₀ slab.
public func qwInterleaveX0(
    textEmbeddings: [Float],
    visualTokens: [Float],
    plan: QWX0Plan,
    hiddenDim: Int
) throws -> [Float] {
    let totalRows = plan.rows
    var x0 = [Float](repeating: 0, count: totalRows * hiddenDim)
    try qwInterleaveX0(into: &x0, textEmbeddings: textEmbeddings, visualTokens: visualTokens, plan: plan, hiddenDim: hiddenDim)
    return x0
}

public func qwInterleaveX0(
    into x0: inout [Float],
    textEmbeddings: [Float],
    visualTokens: [Float],
    plan: QWX0Plan,
    hiddenDim: Int
) throws {
    guard x0.count == plan.rows * hiddenDim else {
        throw QWGridError.bufferLength(expected: plan.rows * hiddenDim, actual: x0.count)
    }
    x0.withUnsafeMutableBufferPointer { x0Buf in
        textEmbeddings.withUnsafeBufferPointer { textBuf in
            visualTokens.withUnsafeBufferPointer { visBuf in
                var textOffset = 0
                var visOffset = 0
                for span in plan.spans {
                    let rowStart = span.rowStart
                    let rowCount = span.rowCount
                    let dstOffset = rowStart * hiddenDim
                    let count = rowCount * hiddenDim
                    switch span.content {
                    case .text:
                        guard textOffset + count <= textBuf.count else { return }
                        for i in 0..<count {
                            x0Buf[dstOffset + i] = textBuf[textOffset + i]
                        }
                        textOffset += count
                    case .vision:
                        guard visOffset + count <= visBuf.count else { return }
                        for i in 0..<count {
                            x0Buf[dstOffset + i] = visBuf[visOffset + i]
                        }
                        visOffset += count
                    }
                }
            }
        }
    }
}

/// End-to-end multimodal image patch encoding through a vision backend.
/// Ingests normalized float RGB pixels, forms the C19 patch slab, evaluates grid
/// geometry, position IDs, and interpolation taps in-memory, and encodes visual tokens.
public func qwEncodeVision(
    normalizedRGB: [Float],
    height: Int,
    width: Int,
    grid: QWVisionGrid,
    backend: any QWVisionBackend
) async throws -> [Float] {
    let patches = try qwBuildPatchSlab(normalizedRGB: normalizedRGB, height: height, width: width, grid: grid)
    return try await backend.encode(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
}

/// End-to-end multimodal image patch encoding through a vision backend from UInt8 RGB bytes [0, 255].
public func qwEncodeVision(
    imageRGB: [UInt8],
    height: Int,
    width: Int,
    grid: QWVisionGrid,
    backend: any QWVisionBackend
) async throws -> [Float] {
    let patches = try qwBuildPatchSlab(imageRGB: imageRGB, height: height, width: width, grid: grid)
    return try await backend.encode(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
}
