// QWEngineDecisionAdapter.swift — Metal 4 QWDecisionExecutionBackend adapter (ADR-002/004/006).

import Accelerate
import CoreGraphics
import Dispatch
import Foundation
import ImageIO
import QWCore
import QWDecision
import QWMM
import QWVision

/// Executes single-pass decision batches on a `QWEngine` with optional step-boundary GPU preemption.
public final class QWEngineDecisionAdapter: QWDecisionExecutionBackend, @unchecked Sendable {
    public let engine: QWEngine
    public let decisionConfig: QWDecisionModelConfig
    public let maxContextLength: Int
    public let gpuTurnArbitrator: QWGPUTurnArbitrator?
    public let isSidecar: Bool
    public let visionBackend: (any QWVisionBackend)?
    public let loadVision: Bool

    private static let evalQueue = DispatchQueue(
        label: "qwegn.decision.metal.eval",
        qos: .userInteractive
    )
    private static let maxImageSourceBytes = 20 * 1024 * 1024
    private static let maxImageDimension = 16_384
    private static let maxImageArea: Int64 = 67_108_864

    public init(
        engine: QWEngine,
        decisionConfig: QWDecisionModelConfig? = nil,
        maxContextLength: Int? = nil,
        gpuTurnArbitrator: QWGPUTurnArbitrator? = nil,
        isSidecar: Bool = true,
        visionBackend: (any QWVisionBackend)? = nil,
        loadVision: Bool = true
    ) {
        self.engine = engine
        let resolvedConfig = decisionConfig ?? engine.descriptor.decisionConfig
            ?? QWDecisionModelConfig.synthesizedVerbalizer(
                vocabSize: engine.descriptor.vocab,
                tieWordEmbeddings: engine.descriptor.tieWordEmbeddings,
                supportsPrefixCache: engine.hasPrefixCache
            )
        self.decisionConfig = resolvedConfig
        let reportedCap = engine.report.contextCapacity ?? engine.descriptor.maxPositionEmbeddings
        self.maxContextLength = max(1, min(reportedCap, maxContextLength ?? reportedCap))
        self.gpuTurnArbitrator = gpuTurnArbitrator
        self.isSidecar = isSidecar
        self.visionBackend = visionBackend
        self.loadVision = loadVision
    }

    public func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        guard case .vocabVerbalizer = decisionConfig.head else {
            throw QWDecisionError.backendFailed(
                "Metal decision head \(decisionConfig.head) requires Milestone M3 kernels"
            )
        }
        if request.slots.contains(where: { $0.isPointerSlot }) {
            throw QWDecisionError.backendFailed("pointer head evaluation requires Milestone M3 kernels")
        }

        let t0 = ContinuousClock.now
        let groupedSlots: [Int: [(slotIdx: Int, position: Int)]] = request.slots.enumerated().reduce(into: [:]) { acc, pair in
            acc[pair.element.sequenceIndex, default: []].append((pair.offset, pair.element.tokenPosition))
        }
        let sortedSeqIndices = groupedSlots.keys.sorted()

        let prepareSequences: @Sendable () async throws -> [(
            tokens: [Int],
            visualSpans: [QWVisualSpan],
            entries: [(slotIdx: Int, position: Int)]
        )] = { [self] in
            var prepared: [(tokens: [Int], visualSpans: [QWVisualSpan], entries: [(slotIdx: Int, position: Int)])] = []
            prepared.reserveCapacity(sortedSeqIndices.count)
            var encodedCache: [Data: (grid: QWVisionGrid, features: [Float])] = [:]
            for seqIdx in sortedSeqIndices {
                guard seqIdx >= 0 && seqIdx < request.sequences.count else {
                    throw QWDecisionError.invalidQuestion("sequence index \(seqIdx) out of bounds")
                }
                let seq = request.sequences[seqIdx]
                let entries = groupedSlots[seqIdx] ?? []
                if seq.images.isEmpty {
                    if let vb = self.visionBackend, vb.isLoaded {
                        try vb.prepareForPrompt(hasImages: false)
                    }
                    prepared.append((seq.tokenIds, [], entries))
                    continue
                }
                guard self.loadVision, let vb = self.visionBackend, vb.isVisionEnabled else {
                    throw QWDecisionError.backendFailed(
                        "Vision tower is disabled (--no-load-vision) or unavailable for this model"
                    )
                }
                try vb.prepareForPrompt(hasImages: true)
                let maxPixels = self.decisionConfig.visionMaxPixels ?? 1_048_576
                let spec = QWMMSpec(visionConfig: vb.visionConfig, maxPixels: maxPixels)
                var encodedImages: [(grid: QWVisionGrid, features: [Float])] = []
                encodedImages.reserveCapacity(seq.images.count)
                for imgData in seq.images {
                    if let cached = encodedCache[imgData] {
                        encodedImages.append(cached)
                    } else {
                        let encoded = try await Self.encodeImage(imgData, maxPixels: maxPixels, spec: spec, backend: vb)
                        encodedCache[imgData] = encoded
                        encodedImages.append(encoded)
                    }
                }
                let (promptTokens, shiftedPositions) = try Self.resolvePlaceholderLayout(
                    tokenIds: seq.tokenIds,
                    images: encodedImages,
                    spec: spec,
                    topology: seq.topology,
                    positions: entries.map(\.position)
                )
                let mmPrompt = try QWMultimodalPrompt(
                    promptTokens: promptTokens,
                    images: encodedImages,
                    hiddenDim: self.engine.descriptor.hidden,
                    spec: spec
                )
                let shiftedEntries = zip(entries, shiftedPositions).map { (slotIdx: $0.slotIdx, position: $1) }
                prepared.append((mmPrompt.expandedTokens, mmPrompt.visualSpans, shiftedEntries))
            }
            return prepared
        }

        let runEval: @Sendable () async throws -> (slotOutputs: [QWDecisionSlotOutput], promptTokens: Int) = { [engine, prepareSequences] in
            let prepared = try await prepareSequences()
            let totalTokens = prepared.reduce(0) { $0 + $1.tokens.count }
            let ordered: [QWDecisionSlotOutput] = try await withCheckedThrowingContinuation { continuation in
                Self.evalQueue.async {
                    do {
                        var out = [QWDecisionSlotOutput](repeating: QWDecisionSlotOutput(), count: request.slots.count)
                        for item in prepared {
                            let outputs = try engine.evaluateDecisionLogits(
                                tokens: item.tokens,
                                positions: item.entries.map(\.position),
                                candidateTokenIds: request.uniqueCandidateTokenIds,
                                visualSpans: item.visualSpans
                            )
                            for (offset, entry) in item.entries.enumerated() {
                                out[entry.slotIdx] = outputs[offset]
                            }
                        }
                        continuation.resume(returning: out)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            return (ordered, totalTokens)
        }

        let result: (slotOutputs: [QWDecisionSlotOutput], promptTokens: Int)
        if isSidecar, let arb = gpuTurnArbitrator {
            result = try await arb.runHighPrioritySidecar {
                try await runEval()
            }
        } else {
            result = try await runEval()
        }

        let elapsed = t0.duration(to: ContinuousClock.now)
        let latencyMs = Double(elapsed.components.seconds) * 1000.0
            + Double(elapsed.components.attoseconds) / 1e15
        return QWDecisionBackendOutput(
            slotOutputs: result.slotOutputs,
            promptTokens: result.promptTokens,
            latencyMs: (latencyMs * 100.0).rounded() / 100.0
        )
    }

    private static func encodeImage(
        _ data: Data,
        maxPixels: Int,
        spec: QWMMSpec,
        backend: any QWVisionBackend
    ) async throws -> (grid: QWVisionGrid, features: [Float]) {
        let decoded = try decodeAndSmartResize(data, maxPixels: maxPixels, spec: spec)
        let patches = try qwBuildPatchSlab(
            imageRGB: decoded.rgb, height: decoded.height, width: decoded.width, grid: decoded.grid
        )
        let features = try await backend.encode(
            patches: patches,
            grid: decoded.grid.plan,
            bins: decoded.grid.interpBins,
            posIds: decoded.grid.ropePositions
        )
        return (decoded.grid, features)
    }

    private static func resolvePlaceholderLayout(
        tokenIds: [Int],
        images: [(grid: QWVisionGrid, features: [Float])],
        spec: QWMMSpec,
        topology: QWDecisionModelConfig.AttentionTopology,
        positions: [Int]
    ) throws -> (promptTokens: [Int], shiftedPositions: [Int]) {
        let placeholders = tokenIds.indices.filter { tokenIds[$0] == spec.imageToken }
        if placeholders.count == images.count {
            let shifted = positions.map { p in
                var delta = 0
                for (j, phIdx) in placeholders.enumerated() where phIdx < p {
                    delta += max(0, images[j].grid.tokens - 1)
                }
                return p + delta
            }
            return (tokenIds, shifted)
        }
        guard placeholders.isEmpty else {
            throw QWDecisionError.invalidQuestion(
                "expected \(images.count) image placeholders, found \(placeholders.count)"
            )
        }
        let isBidirectional: Bool = switch topology {
        case .bidirectionalEncoder: true
        default: false
        }
        let insertAt = (isBidirectional && !tokenIds.isEmpty) ? 1 : 0
        var augmented = tokenIds
        augmented.insert(contentsOf: Array(repeating: spec.imageToken, count: images.count), at: insertAt)
        let addedTokens = images.reduce(0) { $0 + $1.grid.tokens }
        let shifted = positions.map { p in p >= insertAt ? p + addedTokens : p }
        return (augmented, shifted)
    }

    private static func decodeAndSmartResize(
        _ data: Data,
        maxPixels: Int,
        spec: QWMMSpec
    ) throws -> (rgb: [UInt8], width: Int, height: Int, grid: QWVisionGrid) {
        guard !data.isEmpty, data.count <= maxImageSourceBytes else {
            throw QWDecisionError.invalidQuestion("Invalid or oversized image buffer (\(data.count) bytes)")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              w > 0, h > 0 else {
            throw QWDecisionError.invalidQuestion("Failed to inspect image header")
        }
        let rawOrient = (props[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation: UInt32 = (1...8).contains(rawOrient) ? rawOrient : 1
        let orientedW = (5...8).contains(orientation) ? h : w
        let orientedH = (5...8).contains(orientation) ? w : h
        let (area, overflow) = Int64(orientedW).multipliedReportingOverflow(by: Int64(orientedH))
        guard orientedW <= maxImageDimension, orientedH <= maxImageDimension,
              !overflow, area <= maxImageArea else {
            throw QWDecisionError.invalidQuestion("Image dimensions \(orientedW)x\(orientedH) exceed bounds")
        }
        let resized = try QWMMConfig.smartResize(height: orientedH, width: orientedW, maxPixels: maxPixels, spec: spec)
        let grid = try QWVisionGrid(t: 1, h: resized.h / spec.patch, w: resized.w / spec.patch, spec: spec)

        let cgImage: CGImage
        let renderOrientation: UInt32
        if resized.w < orientedW || resized.h < orientedH {
            let thumbOpts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: false,
                kCGImageSourceThumbnailMaxPixelSize: max(resized.w, resized.h),
            ]
            guard let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
                throw QWDecisionError.invalidQuestion("Failed to create image thumbnail")
            }
            cgImage = thumb
            renderOrientation = 1
        } else {
            guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw QWDecisionError.invalidQuestion("Failed to decode CGImage")
            }
            cgImage = full
            renderOrientation = orientation
        }
        let rgb = try renderRGB888(cgImage: cgImage, width: resized.w, height: resized.h, orientation: renderOrientation)
        return (rgb, resized.w, resized.h, grid)
    }

    private static func renderRGB888(
        cgImage: CGImage,
        width: Int,
        height: Int,
        orientation: UInt32
    ) throws -> [UInt8] {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw QWDecisionError.backendFailed("Failed to create sRGB colorspace")
        }
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        let drew = rgba.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Bool in
            guard let ctx = CGContext(
                data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo
            ) else { return false }
            ctx.setFillColor(CGColor(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 1.0))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            let swaps = (5...8).contains(orientation)
            let w = CGFloat(width)
            let h = CGFloat(height)
            let tx: CGAffineTransform = switch orientation {
            case 2: CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
            case 3: CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
            case 4: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
            case 5: CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: w, ty: h)
            case 6: CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: h)
            case 7: CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
            case 8: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: w, ty: 0)
            default: .identity
            }
            ctx.concatenate(tx)
            ctx.interpolationQuality = .high
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: swaps ? height : width, height: swaps ? width : height))
            return true
        }
        guard drew else {
            throw QWDecisionError.backendFailed("CGContext image render failed")
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        rgba.withUnsafeBytes { srcRaw in
            rgb.withUnsafeMutableBytes { dstRaw in
                guard let sBase = srcRaw.baseAddress, let dBase = dstRaw.baseAddress else { return }
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: sBase),
                    height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * 4
                )
                var dstBuf = vImage_Buffer(
                    data: dBase, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * 3
                )
                if vImageConvert_RGBA8888toRGB888(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                    let sPtr = sBase.assumingMemoryBound(to: UInt8.self)
                    let dPtr = dBase.assumingMemoryBound(to: UInt8.self)
                    for i in 0..<(width * height) {
                        dPtr[i * 3 + 0] = sPtr[i * 4 + 0]
                        dPtr[i * 3 + 1] = sPtr[i * 4 + 1]
                        dPtr[i * 3 + 2] = sPtr[i * 4 + 2]
                    }
                }
            }
        }
        return rgb
    }
}
