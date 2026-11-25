// QWMM+Video.swift — Multi-frame temporal patch slab construction, video smart-resize
// token budgeting, and vocabulary-driven multimodal placeholder resolution.

import Accelerate
import Foundation
import QWCore
import QWToken
import QWVision

/// Default total pixel budget across all temporal tubelets of a single video (`~1,568` merged tokens).
public let qwDefaultMaxVideoTotalPixels = 8 * 392 * 392

/// Default per-frame spatial cap for video frames (`392 x 392` = `196` merged tokens/tubelet at `factor=28`).
public let qwDefaultMaxVideoFramePixels = 392 * 392

/// Computes temporal tubelet count `t` and factor-aligned `(pixelH, pixelW)` within `maxVideoTotalPixels`.
public func qwVideoSmartResize(
    height: Int,
    width: Int,
    numFrames: Int,
    spec: QWMMSpec = .qwen3VLReference,
    maxVideoTotalPixels: Int = qwDefaultMaxVideoTotalPixels,
    maxFramePixels: Int = qwDefaultMaxVideoFramePixels
) throws -> (t: Int, h: Int, w: Int, pixelH: Int, pixelW: Int, paddedFrames: Int) {
    guard height > 0, width > 0, numFrames > 0 else {
        throw QWGridError.mergeShape(t: numFrames, h: height, w: width)
    }
    let temporal = max(1, spec.temporal)
    let paddedFrames = max(temporal, ((numFrames + temporal - 1) / temporal) * temporal)
    let tubelets = max(1, paddedFrames / temporal)
    let factorArea = spec.factor * spec.factor
    let rawPerTubeletCap = max(factorArea, maxVideoTotalPixels / tubelets)
    let effectiveFrameMax = max(factorArea, min(maxFramePixels, rawPerTubeletCap))
    let effectiveFrameMin = min(max(factorArea, spec.minPixels), effectiveFrameMax)
    let frameSpec = QWMMSpec(
        patch: spec.patch,
        temporal: spec.temporal,
        merge: spec.merge,
        posSide: spec.posSide,
        imageToken: spec.imageToken,
        videoToken: spec.videoToken,
        minPixels: effectiveFrameMin,
        maxPixels: effectiveFrameMax,
        usesUnfoldedPatchEmbed: spec.usesUnfoldedPatchEmbed,
        mropeLayout: spec.mropeLayout,
        imageMean: spec.imageMean,
        imageStd: spec.imageStd
    )
    let (pixelH, pixelW) = try QWMMConfig.smartResize(height: height, width: width, spec: frameSpec)
    return (
        t: tubelets,
        h: pixelH / spec.patch,
        w: pixelW / spec.patch,
        pixelH: pixelH,
        pixelW: pixelW,
        paddedFrames: paddedFrames
    )
}

/// Builds a `[grid.t * grid.h * grid.w, 3 * temporal * patch * patch]` planar patch slab from `framesRGB`.
/// Requires `framesRGB.count == grid.t * max(1, grid.spec.temporal)` interleaved 8-bit RGB frames.
public func qwBuildVideoPatchSlab(
    framesRGB: [[UInt8]],
    height: Int,
    width: Int,
    grid: QWVisionGrid
) throws -> [Float] {
    let spec = grid.spec
    let temporal = max(1, spec.temporal)
    let expectedH = grid.h * spec.patch
    let expectedW = grid.w * spec.patch
    guard grid.t >= 1 else {
        throw QWGridError.mergeShape(t: grid.t, h: grid.h, w: grid.w)
    }
    guard height == expectedH, width == expectedW else {
        throw QWGridError.pixelDimensions(
            expectedH: expectedH, expectedW: expectedW, actualH: height, actualW: width
        )
    }
    let expectedFrames = grid.t * temporal
    guard framesRGB.count == expectedFrames else {
        throw QWGridError.bufferLength(expected: expectedFrames, actual: framesRGB.count)
    }
    let pixelCount = height * width
    let expectedFrameBytes = pixelCount * 3
    for frame in framesRGB where frame.count != expectedFrameBytes {
        throw QWGridError.bufferLength(expected: expectedFrameBytes, actual: frame.count)
    }

    let patch = spec.patch
    let m = spec.merge
    let patchArea = patch * patch
    let temporalArea = temporal * patchArea
    let patchChannels = 3 * temporalArea
    let framePatches = grid.h * grid.w
    var out = [Float](repeating: 0, count: grid.patches * patchChannels)
    let mean = spec.imageMean
    let std = spec.imageStd
    let patchBytes = patch * MemoryLayout<Float>.stride
    let vPixels = vDSP_Length(pixelCount)
    var div255: Float = 255.0

    // Pre-allocate `temporal` planar `[3][pixelCount]` channel buffers reused across tubelets.
    var tubeletPlanar = Array(
        repeating: Array(repeating: [Float](repeating: 0, count: pixelCount), count: 3),
        count: temporal
    )

    out.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for ti in 0..<grid.t {
            for subT in 0..<temporal {
                let rgb = framesRGB[ti * temporal + subT]
                rgb.withUnsafeBufferPointer { srcBuf in
                    guard let srcBase = srcBuf.baseAddress else { return }
                    for c in 0..<3 {
                        var negMean = -mean[c]
                        var invStd = 1.0 / max(1e-6, std[c])
                        tubeletPlanar[subT][c].withUnsafeMutableBufferPointer { chBuf in
                            guard let chBase = chBuf.baseAddress else { return }
                            vDSP_vfltu8(srcBase + c, 3, chBase, 1, vPixels)
                            vDSP_vsdiv(chBase, 1, &div255, chBase, 1, vPixels)
                            vDSP_vsadd(chBase, 1, &negMean, chBase, 1, vPixels)
                            vDSP_vsmul(chBase, 1, &invStd, chBase, 1, vPixels)
                        }
                    }
                }
            }
            let tubeletRowBase = ti * framePatches
            for subT in 0..<temporal {
                for c in 0..<3 {
                    tubeletPlanar[subT][c].withUnsafeBufferPointer { chBuf in
                        guard let chBase = chBuf.baseAddress else { return }
                        var within = 0
                        for hb in 0..<(grid.h / m) {
                            for wb in 0..<(grid.w / m) {
                                for i in 0..<m {
                                    for j in 0..<m {
                                        let y0 = (hb * m + i) * patch
                                        let x0 = (wb * m + j) * patch
                                        let rowOffset = (tubeletRowBase + within) * patchChannels
                                        let dstPatch = dstBase + rowOffset + c * temporalArea + subT * patchArea
                                        for py in 0..<patch {
                                            memcpy(
                                                dstPatch + py * patch,
                                                chBase + (y0 + py) * width + x0,
                                                patchBytes
                                            )
                                        }
                                        within += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}

/// Capability- and vocabulary-driven resolver for multimodal placeholder envelopes (zero family-name branches).
public enum QWMultimodalTagResolver {
    public static let canonicalImageMarker = "<|vision_start|><|image_pad|><|vision_end|>"
    public static let canonicalVideoMarker = "<|vision_start|><|video_pad|><|vision_end|>"

    public static func resolveImageBlock(
        tokenizer: QWTokenizer,
        descriptor: QWModelDescriptor
    ) -> (placeholder: String, block: String) {
        let imgPiece = (descriptor.imageTokenId >= 0 ? tokenizer.piece(descriptor.imageTokenId) : nil)
            ?? "<|image_pad|>"
        if descriptor.visionStartTokenId >= 0, let s = tokenizer.piece(descriptor.visionStartTokenId),
           descriptor.visionEndTokenId >= 0, let e = tokenizer.piece(descriptor.visionEndTokenId) {
            return (imgPiece, s + imgPiece + e)
        }
        if tokenizer.tokenId(for: "<|begin_of_image|>") != nil,
           tokenizer.tokenId(for: "<|end_of_image|>") != nil {
            return (imgPiece, "<|begin_of_image|>" + imgPiece + "<|end_of_image|>")
        }
        if tokenizer.tokenId(for: "<|vision_start|>") != nil,
           tokenizer.tokenId(for: "<|vision_end|>") != nil {
            return (imgPiece, "<|vision_start|>" + imgPiece + "<|vision_end|>")
        }
        return ("<|image_pad|>", canonicalImageMarker)
    }

    public static func resolveVideoBlock(
        tokenizer: QWTokenizer,
        descriptor: QWModelDescriptor,
        fallbackFrameCount: Int = 4
    ) -> (placeholder: String, block: String, usesImageSequenceFallback: Bool) {
        if descriptor.videoTokenId < 0 {
            let img = resolveImageBlock(tokenizer: tokenizer, descriptor: descriptor)
            let count = max(1, fallbackFrameCount)
            let seq = (1...count).map { "[Frame \($0)]" + img.block }.joined()
            return (img.placeholder, seq, true)
        }
        let vidPiece = tokenizer.piece(descriptor.videoTokenId) ?? "<|video_pad|>"
        if tokenizer.tokenId(for: "<|video_start|>") != nil,
           tokenizer.tokenId(for: "<|video_end|>") != nil {
            return (vidPiece, "<|video_start|>" + vidPiece + "<|video_end|>", false)
        }
        if tokenizer.tokenId(for: "<|begin_of_video|>") != nil,
           tokenizer.tokenId(for: "<|end_of_video|>") != nil {
            return (vidPiece, "<|begin_of_video|>" + vidPiece + "<|end_of_video|>", false)
        }
        if descriptor.visionStartTokenId >= 0, let s = tokenizer.piece(descriptor.visionStartTokenId),
           descriptor.visionEndTokenId >= 0, let e = tokenizer.piece(descriptor.visionEndTokenId) {
            return (vidPiece, s + vidPiece + e, false)
        }
        if tokenizer.tokenId(for: "<|vision_start|>") != nil,
           tokenizer.tokenId(for: "<|vision_end|>") != nil {
            return (vidPiece, "<|vision_start|>" + vidPiece + "<|vision_end|>", false)
        }
        return (vidPiece, vidPiece, false)
    }

    /// Rewrites canonical image/video markers in `text` into the active tokenizer's native media blocks.
    public static func normalizeMediaMarkers(
        in text: String,
        tokenizer: QWTokenizer,
        descriptor: QWModelDescriptor,
        videoFallbackFrameCounts: [Int] = []
    ) -> String {
        var out = text
        if out.contains(canonicalImageMarker) {
            let nativeImg = resolveImageBlock(tokenizer: tokenizer, descriptor: descriptor).block
            if nativeImg != canonicalImageMarker {
                out = out.replacingOccurrences(of: canonicalImageMarker, with: nativeImg)
            }
        }
        if out.contains(canonicalVideoMarker) {
            let segments = out.components(separatedBy: canonicalVideoMarker)
            if segments.count > 1 {
                var rebuilt = segments[0]
                for idx in 1..<segments.count {
                    let fc = (idx - 1) < videoFallbackFrameCounts.count ? videoFallbackFrameCounts[idx - 1] : 4
                    let nativeVid = resolveVideoBlock(
                        tokenizer: tokenizer, descriptor: descriptor, fallbackFrameCount: fc
                    ).block
                    rebuilt += nativeVid + segments[idx]
                }
                out = rebuilt
            }
        }
        return out
    }
}
