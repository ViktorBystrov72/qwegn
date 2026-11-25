// QWVideoDecoder.swift — Hardware AVFoundation / VideoToolbox video frame sampler,
// SSRF/data-URI policy enforcement, and LRU-cached multi-frame ViT patch encoding.

import AVFoundation
import CoreGraphics
import CoreMedia
import CryptoKit
import Darwin
import Foundation
import QWCore
import QWEngine
import QWMM
import QWToken
import QWVision

public enum QWVideoDecoder {
    public static let defaultSamplingFPS: Double = 2.0
    public static let defaultSampledMaxFrames: Int = 12

    public struct DecodedVideoFrames: Sendable {
        public let framesRGB: [[UInt8]]
        public let width: Int
        public let height: Int
        public let grid: QWVisionGrid
        public let durationSeconds: Double
        public let sampledTimestamps: [Double]
    }

    private enum ClassifiedVideoSource: Sendable {
        case dataURI(raw: String, fileExtension: String)
        case remoteURL(url: URL, fileExtension: String)
        case localFile(path: String)
    }

    private final class SyncBridgeBox<T: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<T, Error>?
        func store(_ value: Result<T, Error>) { lock.withLock { result = value } }
        func take() throws -> T {
            try lock.withLock {
                guard let result else {
                    throw QWImageDecoderError.imageDecodeFailed("Video decode operation did not complete")
                }
                return try result.get()
            }
        }
    }

    struct VideoCacheKey: Hashable, Sendable {
        let digest: String
        let hiddenDim: Int
        let fpsMilli: Int
        let maxFrames: Int
        let maxTotalPixels: Int
        let patch: Int
        let temporal: Int
        let merge: Int
        let posSide: Int
        let videoToken: Int
        let usesUnfoldedPatchEmbed: Bool
        let allowDataURLs: Bool
        let allowLocalPaths: Bool
        let allowRemoteURLs: Bool
    }

    final class EncodedVideoLRUCache: @unchecked Sendable {
        private let lock = NSLock()
        private let maxEntries = 6
        private var order: [VideoCacheKey] = []
        private var entries: [VideoCacheKey: [(grid: QWVisionGrid, features: [Float])]] = [:]

        func get(_ key: VideoCacheKey) -> [(grid: QWVisionGrid, features: [Float])]? {
            lock.withLock {
                guard let val = entries[key] else { return nil }
                if let idx = order.firstIndex(of: key) {
                    order.remove(at: idx)
                    order.append(key)
                }
                return val
            }
        }

        func put(_ key: VideoCacheKey, value: [(grid: QWVisionGrid, features: [Float])]) {
            lock.withLock {
                if entries[key] != nil, let idx = order.firstIndex(of: key) {
                    order.remove(at: idx)
                } else if order.count >= maxEntries, let oldest = order.first {
                    order.removeFirst()
                    entries.removeValue(forKey: oldest)
                }
                order.append(key)
                entries[key] = value
            }
        }

        func clear() {
            lock.withLock {
                order.removeAll()
                entries.removeAll()
            }
        }
    }

    static let encodedCache = EncodedVideoLRUCache()

    /// Formats a CLI/REPL user prompt by prepending native tokenizer image/video blocks when not already present.
    public static func formatPromptWithMedia(
        userPrompt: String,
        imageCount: Int,
        videoInputs: [QWVideoInputSpec],
        tokenizer: QWTokenizer,
        descriptor: QWModelDescriptor
    ) -> String {
        var prefix = ""
        if imageCount > 0 {
            let img = QWMultimodalTagResolver.resolveImageBlock(tokenizer: tokenizer, descriptor: descriptor)
            if !userPrompt.contains(img.placeholder) && !userPrompt.contains("<|image_pad|>") {
                prefix += String(repeating: img.block, count: imageCount)
            }
        }
        if !videoInputs.isEmpty {
            for vid in videoInputs {
                let fb = resolvedFallbackFrameCount(for: vid, policyMaxFrames: QWImageDecodePolicy.cliLocal.maxVideoFrames)
                let resolved = QWMultimodalTagResolver.resolveVideoBlock(
                    tokenizer: tokenizer, descriptor: descriptor, fallbackFrameCount: fb
                )
                if resolved.usesImageSequenceFallback || (!userPrompt.contains(resolved.placeholder) && !userPrompt.contains("<|video_pad|>")) {
                    prefix += resolved.block
                }
            }
        }
        return prefix + userPrompt
    }

    /// Deterministic frame count (`1...8`, default `4`) used when a vision tower has `videoToken < 0`.
    public static func resolvedFallbackFrameCount(
        for input: QWVideoInputSpec,
        policyMaxFrames: Int = QWImageDecodePolicy.defaultMaxVideoFrames
    ) -> Int {
        let cap = max(1, min(8, policyMaxFrames))
        return min(cap, max(1, input.maxFrames ?? 4))
    }

    /// Computes uniform bin-midpoint timestamps in `[0, durationSeconds)` clamped to `[minFrames, maxFrames]`.
    public static func computeSampleTimestamps(
        durationSeconds: Double,
        nominalFPS: Double,
        requestedFPS: Double? = nil,
        requestedMaxFrames: Int? = nil,
        temporalPatchSize: Int = 2,
        policyMaxFrames: Int = QWImageDecodePolicy.defaultMaxVideoFrames
    ) -> [Double] {
        let minFrames = max(1, temporalPatchSize)
        let hardMax = max(minFrames, policyMaxFrames)
        let userMax = requestedMaxFrames.map { min(max($0, minFrames), hardMax) }
            ?? min(Self.defaultSampledMaxFrames, hardMax)
        guard durationSeconds.isFinite, durationSeconds > 0.05 else {
            return Array(repeating: 0.0, count: minFrames)
        }
        let targetFPS = min(max(requestedFPS ?? Self.defaultSamplingFPS, 0.1), 16.0)
        let desiredByFPS = Int((durationSeconds * targetFPS).rounded(.toNearestOrAwayFromZero))
        let nativeEstimate: Int = {
            guard nominalFPS.isFinite, nominalFPS > 0.5 else { return userMax }
            return max(minFrames, Int((durationSeconds * nominalFPS).rounded(.toNearestOrAwayFromZero)))
        }()
        let sampleCount = min(userMax, max(minFrames, min(desiredByFPS, nativeEstimate)))
        return (0..<sampleCount).map { idx in
            (Double(idx) + 0.5) * durationSeconds / Double(sampleCount)
        }
    }

    public static func validateSourcePolicy(
        urlOrDataUri: String,
        policy: QWImageDecodePolicy = .serverDefault
    ) throws {
        _ = try classifySource(urlOrDataUri: urlOrDataUri, policy: policy)
    }

    private static func classifySource(
        urlOrDataUri: String,
        policy: QWImageDecodePolicy
    ) throws -> ClassifiedVideoSource {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QWImageDecoderError.disallowedImageSource("Empty video source")
        }
        let lower = trimmed.lowercased()
        if lower.hasPrefix("data:") {
            guard policy.allowDataURLs else {
                throw QWImageDecoderError.disallowedImageSource("Video data URIs are disabled by policy")
            }
            guard let commaIdx = trimmed.firstIndex(of: ",") else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            let header = String(lower[..<commaIdx])
            guard header.hasPrefix("data:video/"), header.contains(";base64") else {
                throw QWImageDecoderError.disallowedImageSource("Only base64 data:video/* URIs are allowed")
            }
            let ext = Self.fileExtension(forDataHeader: header)
            return .dataURI(raw: trimmed, fileExtension: ext)
        }
        if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
            guard policy.allowRemoteImageURLs else {
                throw QWImageDecoderError.disallowedImageSource("Remote HTTP(S) video URLs are disabled by policy")
            }
            guard let url = URL(string: trimmed) else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            let rawExt = url.pathExtension.lowercased()
            let ext = ["mp4", "mov", "m4v", "webm", "avi"].contains(rawExt) ? rawExt : "mp4"
            return .remoteURL(url: url, fileExtension: ext)
        }
        if lower.hasPrefix("file://") || trimmed.hasPrefix("/") || trimmed.hasPrefix("~")
            || trimmed.hasPrefix("./") || trimmed.hasPrefix("../") {
            guard policy.allowLocalImagePaths else {
                throw QWImageDecoderError.disallowedImageSource("Local filesystem video paths are disabled by policy")
            }
            let path = lower.hasPrefix("file://")
                ? (URL(string: trimmed)?.path ?? "")
                : trimmed.expandingTildePath
            return .localFile(path: path)
        }
        if lower.contains("://") {
            throw QWImageDecoderError.disallowedImageSource("Rejected video URL '\(trimmed.prefix(48))'")
        }
        if policy.allowLocalImagePaths, FileManager.default.fileExists(atPath: trimmed.expandingTildePath) {
            return .localFile(path: trimmed.expandingTildePath)
        }
        throw QWImageDecoderError.disallowedImageSource("Rejected video source '\(trimmed.prefix(48))'")
    }

    private static func fileExtension(forDataHeader header: String) -> String {
        if header.contains("video/quicktime") || header.contains("video/mov") { return "mov" }
        if header.contains("video/x-m4v") || header.contains("video/m4v") { return "m4v" }
        if header.contains("video/webm") { return "webm" }
        if header.contains("video/avi") || header.contains("video/x-msvideo") { return "avi" }
        return "mp4"
    }

    private static func validateVideoMagicBytes(_ data: Data) throws {
        guard data.count >= 12 else {
            throw QWImageDecoderError.imageDecodeFailed("Video payload too short (\(data.count) bytes)")
        }
        let bytes = Array(data.prefix(12))
        let fourCC = String(bytes: bytes[4..<8], encoding: .ascii) ?? ""
        let isoBrands: Set<String> = ["ftyp", "moov", "mdat", "wide", "free", "skip", "pnot"]
        let isISOBMFF = isoBrands.contains(fourCC)
        let isEBML = bytes[0] == 0x1A && bytes[1] == 0x45 && bytes[2] == 0xDF && bytes[3] == 0xA3
        let isRIFF = bytes[0] == 0x52 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x46
        guard isISOBMFF || isEBML || isRIFF else {
            throw QWImageDecoderError.imageDecodeFailed("Unrecognized video container header")
        }
    }

    private static func writeEphemeralVideoFile(data: Data, fileExtension: String) throws -> URL {
        try validateVideoMagicBytes(data)
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("qwg-video-\(UUID().uuidString).\(fileExtension)")
        try data.write(to: url, options: .atomic)
        _ = url.path.withCString { chmod($0, S_IRUSR | S_IWUSR) }
        return url
    }

    private static func decodeBase64VideoURI(_ raw: String, maxBytes: Int) throws -> Data {
        guard let commaIdx = raw.firstIndex(of: ",") else {
            throw QWImageDecoderError.invalidDataUrl(raw)
        }
        let b64 = String(raw[raw.index(after: commaIdx)...])
        guard b64.utf8.count <= (maxBytes / 3 + 2) * 4 else {
            throw QWImageDecoderError.imageTooLarge("Base64 video payload exceeds \(maxBytes) bytes")
        }
        guard let decoded = Data(base64Encoded: b64, options: .ignoreUnknownCharacters) else {
            throw QWImageDecoderError.base64DecodeFailed
        }
        guard decoded.count <= maxBytes else {
            throw QWImageDecoderError.imageTooLarge("Decoded video exceeds \(maxBytes) bytes")
        }
        return decoded
    }

    /// Synchronously decodes and resizes sampled video frames (safe to call from CLI main thread).
    public static func decodeSmartResizedSync(
        input: QWVideoInputSpec,
        policy: QWImageDecodePolicy = .cliLocal,
        spec: QWMMSpec = .qwen3VLReference
    ) throws -> DecodedVideoFrames {
        let sem = DispatchSemaphore(value: 0)
        let box = SyncBridgeBox<DecodedVideoFrames>()
        Task.detached(priority: .userInitiated) {
            do {
                let res = try await decodeSmartResized(input: input, policy: policy, spec: spec)
                box.store(.success(res))
            } catch {
                box.store(.failure(error))
            }
            sem.signal()
        }
        sem.wait()
        return try box.take()
    }

    /// Asynchronously resolves `input` under `policy` and extracts factor-aligned RGB frames via `AVFoundation`.
    public static func decodeSmartResized(
        input: QWVideoInputSpec,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec = .qwen3VLReference
    ) async throws -> DecodedVideoFrames {
        let classified = try classifySource(urlOrDataUri: input.url, policy: policy)
        let assetURL: URL
        var cleanupURL: URL?
        switch classified {
        case .dataURI(let raw, let ext):
            let data = try await QWImageDecoder.runOffCooperativePool {
                try decodeBase64VideoURI(raw, maxBytes: policy.maxVideoBytes)
            }
            let tmp = try writeEphemeralVideoFile(data: data, fileExtension: ext)
            assetURL = tmp
            cleanupURL = tmp
        case .remoteURL(let remote, let ext):
            let data = try await QWImageDecodePolicy.fetchBoundedRemoteDataAsync(
                url: remote, maxBytes: policy.maxVideoBytes
            )
            let tmp = try writeEphemeralVideoFile(data: data, fileExtension: ext)
            assetURL = tmp
            cleanupURL = tmp
        case .localFile(let path):
            guard !path.isEmpty,
                  FileManager.default.fileExists(atPath: path),
                  let attrs = try? FileManager.default.attributesOfItem(atPath: path),
                  let size = attrs[.size] as? NSNumber else {
                throw QWImageDecoderError.fileNotFound(path.isEmpty ? input.url : path)
            }
            guard size.int64Value <= Int64(policy.maxVideoBytes) else {
                throw QWImageDecoderError.imageTooLarge("Video file exceeds \(policy.maxVideoBytes) bytes")
            }
            assetURL = URL(fileURLWithPath: path)
        }
        defer {
            if let cleanupURL {
                try? FileManager.default.removeItem(at: cleanupURL)
            }
        }
        return try await extractFramesFromLocalAsset(
            assetURL: assetURL, input: input, policy: policy, spec: spec
        )
    }

    private static func extractFramesFromLocalAsset(
        assetURL: URL,
        input: QWVideoInputSpec,
        policy: QWImageDecodePolicy,
        spec: QWMMSpec
    ) async throws -> DecodedVideoFrames {
        let asset = AVURLAsset(url: assetURL, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else {
            throw QWImageDecoderError.imageDecodeFailed("Video container has no video track")
        }
        let durationCM = try await asset.load(.duration)
        let (naturalSize, preferredTransform, nominalFPSFloat) = try await track.load(
            .naturalSize, .preferredTransform, .nominalFrameRate
        )
        let durationSeconds = CMTimeGetSeconds(durationCM)
        if durationSeconds.isFinite && durationSeconds > policy.maxVideoDurationSeconds {
            throw QWImageDecoderError.imageTooLarge(
                "Video duration \(String(format: "%.1f", durationSeconds))s exceeds max \(policy.maxVideoDurationSeconds)s"
            )
        }
        let transformed = naturalSize.applying(preferredTransform)
        let orientedW = max(1, Int(abs(transformed.width).rounded()))
        let orientedH = max(1, Int(abs(transformed.height).rounded()))
        guard orientedW <= policy.maxPixelDimension && orientedH <= policy.maxPixelDimension else {
            throw QWImageDecoderError.imageTooLarge(
                "Video frame dimensions \(orientedW)x\(orientedH) exceed max \(policy.maxPixelDimension)"
            )
        }
        let (area, overflow) = Int64(orientedW).multipliedReportingOverflow(by: Int64(orientedH))
        let maxArea = min(Int64(policy.maxPixelArea), Int64(QWImageDecodePolicy.defaultMaxPixelArea))
        guard !overflow, area <= maxArea else {
            throw QWImageDecoderError.imageTooLarge("Video frame area \(orientedW)x\(orientedH) exceeds max \(maxArea)")
        }

        let effectiveTemporal = spec.videoToken >= 0 ? max(1, spec.temporal) : 1
        let safeDur = durationSeconds.isFinite ? max(0.0, durationSeconds) : 0.0
        let timestamps: [Double]
        if spec.videoToken < 0 {
            let count = resolvedFallbackFrameCount(for: input, policyMaxFrames: policy.maxVideoFrames)
            timestamps = safeDur > 0.05
                ? (0..<count).map { (Double($0) + 0.5) * safeDur / Double(count) }
                : Array(repeating: 0.0, count: count)
        } else {
            timestamps = computeSampleTimestamps(
                durationSeconds: safeDur,
                nominalFPS: Double(nominalFPSFloat),
                requestedFPS: input.fps,
                requestedMaxFrames: input.maxFrames,
                temporalPatchSize: effectiveTemporal,
                policyMaxFrames: policy.maxVideoFrames
            )
        }
        let maxTotalPixels = input.maxPixels.map { min(max($0, spec.factor * spec.factor), QWMMConfig.maxPixels) }
            ?? qwDefaultMaxVideoTotalPixels
        let resized = try qwVideoSmartResize(
            height: orientedH,
            width: orientedW,
            numFrames: timestamps.count,
            spec: QWMMSpec(
                patch: spec.patch, temporal: effectiveTemporal, merge: spec.merge, posSide: spec.posSide,
                imageToken: spec.imageToken, videoToken: spec.videoToken,
                minPixels: spec.minPixels, maxPixels: spec.maxPixels,
                usesUnfoldedPatchEmbed: spec.usesUnfoldedPatchEmbed, mropeLayout: spec.mropeLayout,
                imageMean: spec.imageMean, imageStd: spec.imageStd
            ),
            maxVideoTotalPixels: maxTotalPixels
        )

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: CGFloat(max(resized.pixelW, resized.pixelH) * 2),
                                       height: CGFloat(max(resized.pixelW, resized.pixelH) * 2))
        let stepSec = timestamps.count > 1 ? max(0.01, (timestamps[1] - timestamps[0]) * 0.35) : 0.05
        let tol = CMTime(seconds: stepSec, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tol
        generator.requestedTimeToleranceAfter = tol

        var framesRGB: [[UInt8]] = []
        framesRGB.reserveCapacity(resized.paddedFrames)
        for ts in timestamps {
            let cmTime = CMTime(seconds: ts, preferredTimescale: 600)
            let cgImg: CGImage
            if let generated = try? await generator.image(at: cmTime).image {
                cgImg = generated
            } else if let fallback = try? await generator.image(at: .zero).image {
                cgImg = fallback
            } else {
                throw QWImageDecoderError.imageDecodeFailed("Failed to decode video frame at \(String(format: "%.2f", ts))s")
            }
            let rendered = try QWImageDecoder.renderRGB(
                cgImage: cgImg, targetWidth: resized.pixelW, targetHeight: resized.pixelH, orientation: 1
            )
            framesRGB.append(rendered.rgb)
        }
        guard let lastFrame = framesRGB.last else {
            throw QWImageDecoderError.emptyImage
        }
        while framesRGB.count < resized.paddedFrames {
            framesRGB.append(lastFrame)
        }
        let grid = try QWVisionGrid(t: resized.t, h: resized.h, w: resized.w, spec: spec)
        return DecodedVideoFrames(
            framesRGB: framesRGB,
            width: resized.pixelW,
            height: resized.pixelH,
            grid: grid,
            durationSeconds: durationSeconds.isFinite ? max(0.0, durationSeconds) : 0.0,
            sampledTimestamps: timestamps
        )
    }

    static func cacheKey(
        input: QWVideoInputSpec,
        hiddenDim: Int,
        spec: QWMMSpec,
        policy: QWImageDecodePolicy
    ) -> VideoCacheKey {
        let trimmed = input.url.trimmingCharacters(in: .whitespacesAndNewlines)
        var hasher = SHA256()
        hasher.update(data: Data(trimmed.utf8))
        if policy.allowLocalImagePaths && trimmed.count < 1024 {
            let path = trimmed.lowercased().hasPrefix("file://")
                ? (URL(string: trimmed)?.path ?? "")
                : trimmed.expandingTildePath
            if !path.isEmpty, let attrs = try? FileManager.default.attributesOfItem(atPath: path) {
                if let size = attrs[.size] as? NSNumber {
                    hasher.update(data: Data("s:\(size.uint64Value)".utf8))
                }
                if let mtime = attrs[.modificationDate] as? Date {
                    hasher.update(data: Data("m:\(mtime.timeIntervalSince1970)".utf8))
                }
            }
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return VideoCacheKey(
            digest: digest,
            hiddenDim: hiddenDim,
            fpsMilli: Int(((input.fps ?? Self.defaultSamplingFPS) * 1000.0).rounded()),
            maxFrames: input.maxFrames ?? Self.defaultSampledMaxFrames,
            maxTotalPixels: input.maxPixels ?? qwDefaultMaxVideoTotalPixels,
            patch: spec.patch,
            temporal: spec.temporal,
            merge: spec.merge,
            posSide: spec.posSide,
            videoToken: spec.videoToken,
            usesUnfoldedPatchEmbed: spec.usesUnfoldedPatchEmbed,
            allowDataURLs: policy.allowDataURLs,
            allowLocalPaths: policy.allowLocalImagePaths,
            allowRemoteURLs: policy.allowRemoteImageURLs
        )
    }

    /// Encodes a single `QWVideoInputSpec` via `visionBackend`, returning either 1 temporal video cube
    /// (`spec.videoToken >= 0`) or `K` fallback frame images (`spec.videoToken < 0`).
    static func encodeVideo(
        input: QWVideoInputSpec,
        hiddenDim: Int,
        spec: QWMMSpec,
        policy: QWImageDecodePolicy,
        visionBackend: any QWVisionBackend,
        wasLoadedBeforeTurn: Bool,
        loadedNow: inout Bool
    ) async throws -> (videos: [(grid: QWVisionGrid, features: [Float])], fallbackImages: [(grid: QWVisionGrid, features: [Float])], encodeMs: Double) {
        try validateSourcePolicy(urlOrDataUri: input.url, policy: policy)
        let key = cacheKey(input: input, hiddenDim: hiddenDim, spec: spec, policy: policy)
        if let cached = encodedCache.get(key) {
            if spec.videoToken >= 0 {
                return (videos: cached, fallbackImages: [], encodeMs: 0.0)
            } else {
                return (videos: [], fallbackImages: cached, encodeMs: 0.0)
            }
        }
        if !visionBackend.isLoaded {
            try visionBackend.prepareForPrompt(hasImages: true)
            loadedNow = loadedNow || (!wasLoadedBeforeTurn && visionBackend.isLoaded)
        }
        let decoded = try await decodeSmartResized(input: input, policy: policy, spec: spec)
        let clock = ContinuousClock()
        let t0 = clock.now
        if spec.videoToken >= 0 {
            let patches = try qwBuildVideoPatchSlab(
                framesRGB: decoded.framesRGB,
                height: decoded.height,
                width: decoded.width,
                grid: decoded.grid
            )
            let features = try await visionBackend.encode(
                patches: patches,
                grid: decoded.grid.plan,
                bins: decoded.grid.interpBins,
                posIds: decoded.grid.ropePositions
            )
            let d = t0.duration(to: clock.now)
            let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
            let result = [(grid: decoded.grid, features: features)]
            encodedCache.put(key, value: result)
            return (result, [], ms)
        } else {
            var frameResults: [(grid: QWVisionGrid, features: [Float])] = []
            for rgb in decoded.framesRGB {
                let patches: [Float]
                let frameGrid: QWVisionGrid
                if spec.isInkling2 {
                    let prep = try QWImageDecoder.normalizePadAndPatchifyInkling2(
                        rgb: rgb, height: decoded.height, width: decoded.width, spec: spec
                    )
                    patches = prep.patches
                    frameGrid = prep.grid
                } else {
                    frameGrid = try QWVisionGrid(t: 1, h: decoded.grid.h, w: decoded.grid.w, spec: spec)
                    patches = try qwBuildPatchSlab(
                        imageRGB: rgb, height: decoded.height, width: decoded.width, grid: frameGrid
                    )
                }
                let features = try await visionBackend.encode(
                    patches: patches,
                    grid: frameGrid.plan,
                    bins: frameGrid.interpBins,
                    posIds: frameGrid.ropePositions
                )
                frameResults.append((grid: frameGrid, features: features))
            }
            let d = t0.duration(to: clock.now)
            let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
            encodedCache.put(key, value: frameResults)
            return ([], frameResults, ms)
        }
    }

    /// Synchronous CLI counterpart to `encodeVideo`.
    static func encodeVideoSync(
        input: QWVideoInputSpec,
        hiddenDim: Int,
        spec: QWMMSpec,
        policy: QWImageDecodePolicy,
        visionBackend: QWVisionOnDemandBackend
    ) throws -> (videos: [(grid: QWVisionGrid, features: [Float])], fallbackImages: [(grid: QWVisionGrid, features: [Float])]) {
        try validateSourcePolicy(urlOrDataUri: input.url, policy: policy)
        let key = cacheKey(input: input, hiddenDim: hiddenDim, spec: spec, policy: policy)
        if let cached = encodedCache.get(key) {
            if spec.videoToken >= 0 {
                return (videos: cached, fallbackImages: [])
            } else {
                return (videos: [], fallbackImages: cached)
            }
        }
        if !visionBackend.isLoaded {
            try visionBackend.prepareForPrompt(hasImages: true)
        }
        let decoded = try decodeSmartResizedSync(input: input, policy: policy, spec: spec)
        if spec.videoToken >= 0 {
            let patches = try qwBuildVideoPatchSlab(
                framesRGB: decoded.framesRGB,
                height: decoded.height,
                width: decoded.width,
                grid: decoded.grid
            )
            let features = try visionBackend.encodeSync(
                patches: patches,
                grid: decoded.grid.plan,
                bins: decoded.grid.interpBins,
                posIds: decoded.grid.ropePositions
            )
            let result = [(grid: decoded.grid, features: features)]
            encodedCache.put(key, value: result)
            return (result, [])
        } else {
            var frameResults: [(grid: QWVisionGrid, features: [Float])] = []
            for rgb in decoded.framesRGB {
                let patches: [Float]
                let frameGrid: QWVisionGrid
                if spec.isInkling2 {
                    let prep = try QWImageDecoder.normalizePadAndPatchifyInkling2(
                        rgb: rgb, height: decoded.height, width: decoded.width, spec: spec
                    )
                    patches = prep.patches
                    frameGrid = prep.grid
                } else {
                    frameGrid = try QWVisionGrid(t: 1, h: decoded.grid.h, w: decoded.grid.w, spec: spec)
                    patches = try qwBuildPatchSlab(
                        imageRGB: rgb, height: decoded.height, width: decoded.width, grid: frameGrid
                    )
                }
                let features = try visionBackend.encodeSync(
                    patches: patches,
                    grid: frameGrid.plan,
                    bins: frameGrid.interpBins,
                    posIds: frameGrid.ropePositions
                )
                frameResults.append((grid: frameGrid, features: features))
            }
            encodedCache.put(key, value: frameResults)
            return ([], frameResults)
        }
    }
}
