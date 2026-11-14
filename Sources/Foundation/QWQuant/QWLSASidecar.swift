// QWLSASidecar.swift — Binary `.qlsa` (`QLSA0001`) sidecar container reader/writer and Weight-SVD synthesizer.
// Holds per-layer low-rank index projections `wq_idx` and `wk_idx` (`f16`, shape `[d_idx, in_dim]`)
// bound to a model's `geometryDigestV2Hex` for Latent Sparse Attention (arXiv:2610.42147).

import Darwin
import Foundation
import QWCore

public enum QWLSASidecarError: Error, CustomStringConvertible, Equatable {
    case fileMissing(String)
    case badMagic(String)
    case truncated(String)
    case geometryMismatch(sidecar: String, expected: String)
    case modelMismatch(sidecar: String, expected: String)
    case tensorMissing(String)
    case shapeMismatch(tensor: String, got: [Int], expected: [Int])
    case svdFailed(String)

    public var description: String {
        switch self {
        case .fileMissing(let p):
            return "QWLSASidecar: sidecar file not found at '\(p)'"
        case .badMagic(let p):
            return "QWLSASidecar: bad magic in '\(p)' (expected QLSA0001)"
        case .truncated(let p):
            return "QWLSASidecar: truncated sidecar file '\(p)'"
        case .geometryMismatch(let s, let e):
            return "QWLSASidecar: sidecar geometry_digest \(s) != model \(e)"
        case .modelMismatch(let s, let e):
            return "QWLSASidecar: sidecar model_id '\(s)' != model '\(e)'"
        case .tensorMissing(let n):
            return "QWLSASidecar: tensor '\(n)' missing from sidecar"
        case .shapeMismatch(let t, let got, let exp):
            return "QWLSASidecar: tensor '\(t)' shape \(got) != expected \(exp)"
        case .svdFailed(let msg):
            return "QWLSASidecar: SVD synthesis failed: \(msg)"
        }
    }
}

/// Mmap-backed `QLSA0001` sidecar reader and writer.
public final class QWLSASidecar: @unchecked Sendable {
    public static let magic = "QLSA0001"

    public struct TensorEntry: Codable, Sendable, Equatable {
        public let shape: [Int]
        public let dtype: String
        public let rawOff: Int
        public let nBytes: Int

        enum CodingKeys: String, CodingKey {
            case shape
            case dtype
            case rawOff = "raw_off"
            case nBytes = "n_bytes"
        }

        public init(shape: [Int], dtype: String = "f16", rawOff: Int, nBytes: Int) {
            self.shape = shape
            self.dtype = dtype
            self.rawOff = rawOff
            self.nBytes = nBytes
        }
    }

    public struct Header: Codable, Sendable, Equatable {
        public struct Provenance: Codable, Sendable, Equatable {
            public let format: String
            public let modelId: String
            public let geometryDigest: String
            public let method: String
            public let indexDim: Int
            public let poolSize: Int
            public let clusterBlockSize: Int
            public let bucketsPerBlock: Int
            public let maxBucketKeys: Int
            public let recommendedBudget: Int
            public let sinkTokens: Int
            public let windowTokens: Int

            enum CodingKeys: String, CodingKey {
                case format
                case modelId = "model_id"
                case geometryDigest = "geometry_digest"
                case method
                case indexDim = "index_dim"
                case poolSize = "pool_size"
                case clusterBlockSize = "cluster_block_size"
                case bucketsPerBlock = "buckets_per_block"
                case maxBucketKeys = "max_bucket_keys"
                case recommendedBudget = "recommended_budget"
                case sinkTokens = "sink_tokens"
                case windowTokens = "window_tokens"
            }

            public init(
                format: String = QWLSASidecar.magic,
                modelId: String,
                geometryDigest: String,
                method: String,
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                poolSize: Int = QWLSAConfig.poolSize,
                clusterBlockSize: Int = QWLSAConfig.defaultClusterBlock,
                bucketsPerBlock: Int = QWLSAConfig.bucketsPerBlock,
                maxBucketKeys: Int = QWLSAConfig.maxBucketKeys,
                recommendedBudget: Int = QWLSAConfig.defaultBudget,
                sinkTokens: Int = QWLSAConfig.sinkTokens,
                windowTokens: Int = QWLSAConfig.windowTokens
            ) {
                self.format = format
                self.modelId = modelId
                self.geometryDigest = geometryDigest
                self.method = method
                self.indexDim = indexDim
                self.poolSize = poolSize
                self.clusterBlockSize = clusterBlockSize
                self.bucketsPerBlock = bucketsPerBlock
                self.maxBucketKeys = maxBucketKeys
                self.recommendedBudget = recommendedBudget
                self.sinkTokens = sinkTokens
                self.windowTokens = windowTokens
            }
        }

        public let provenance: Provenance
        public let tensors: [String: TensorEntry]

        public init(provenance: Provenance, tensors: [String: TensorEntry]) {
            self.provenance = provenance
            self.tensors = tensors
        }
    }

    public let url: URL
    public let header: Header
    private let rawBase: UnsafeMutableRawPointer
    private let rawCount: Int

    deinit {
        if rawCount > 0 {
            _ = munmap(rawBase, rawCount)
        }
    }

    /// Fast header-only inspection of a `.qlsa` sidecar file without mapping tensor payloads.
    public static func peekInfo(url: URL, descriptor: QWModelDescriptor? = nil) throws -> QWLSASidecarInfo {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QWLSASidecarError.fileMissing(url.path)
        }
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let prefix = try fh.read(upToCount: 16), prefix.count == 16 else {
            throw QWLSASidecarError.truncated(url.path)
        }
        guard String(decoding: prefix[0..<8], as: UTF8.self) == magic else {
            throw QWLSASidecarError.badMagic(url.path)
        }
        var hdrLen: UInt64 = 0
        for i in 0..<8 {
            hdrLen |= UInt64(prefix[8 + i]) << (8 * i)
        }
        let fileSize = fh.seekToEndOfFile()
        guard fileSize >= 16 + hdrLen else {
            throw QWLSASidecarError.truncated(url.path)
        }
        try fh.seek(toOffset: fileSize - hdrLen)
        guard let hdrData = try fh.read(upToCount: Int(hdrLen)), hdrData.count == Int(hdrLen) else {
            throw QWLSASidecarError.truncated(url.path)
        }
        let decoded = try JSONDecoder().decode(Header.self, from: hdrData)
        let prov = decoded.provenance
        let matches: Bool
        if let descriptor {
            matches = (prov.geometryDigest == descriptor.geometryDigestV3Hex
                || prov.geometryDigest == descriptor.geometryDigestV2Hex)
        } else {
            matches = true
        }
        return QWLSASidecarInfo(
            path: url.path,
            modelId: prov.modelId,
            geometryDigest: prov.geometryDigest,
            method: prov.method,
            indexDim: prov.indexDim,
            poolSize: prov.poolSize,
            clusterBlockSize: prov.clusterBlockSize,
            bucketsPerBlock: prov.bucketsPerBlock,
            maxBucketKeys: prov.maxBucketKeys,
            recommendedBudget: prov.recommendedBudget,
            geometryMatches: matches
        )
    }

    /// Opens a `QLSA0001` sidecar and validates its geometry digest against `descriptor`.
    public init(url: URL, descriptor: QWModelDescriptor? = nil, skipIdentityCheck: Bool = false) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw QWLSASidecarError.fileMissing(url.path) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw QWLSASidecarError.fileMissing(url.path) }
        let size = Int(st.st_size)
        guard size > 16 else { throw QWLSASidecarError.truncated(url.path) }
        let mapped = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
        guard let ptr = mapped, ptr != MAP_FAILED else {
            throw QWLSASidecarError.fileMissing(url.path)
        }
        self.rawBase = ptr
        self.rawCount = size

        let bytes = ptr.assumingMemoryBound(to: UInt8.self)
        let magicSlice = UnsafeBufferPointer(start: bytes, count: 8)
        guard String(decoding: magicSlice, as: UTF8.self) == Self.magic else {
            _ = munmap(ptr, size)
            throw QWLSASidecarError.badMagic(url.path)
        }
        var hdrLen: UInt64 = 0
        for i in 0..<8 {
            hdrLen |= UInt64(bytes[8 + i]) << (8 * i)
        }
        let hdrStart = size - Int(hdrLen)
        guard hdrStart >= 16 else {
            _ = munmap(ptr, size)
            throw QWLSASidecarError.truncated(url.path)
        }
        let hdrData = Data(bytes: ptr.advanced(by: hdrStart), count: Int(hdrLen))
        let decoded: Header
        do {
            decoded = try JSONDecoder().decode(Header.self, from: hdrData)
        } catch {
            _ = munmap(ptr, size)
            throw error
        }
        if !skipIdentityCheck, let descriptor {
            let matchesDigest = decoded.provenance.geometryDigest == descriptor.geometryDigestV3Hex
                || decoded.provenance.geometryDigest == descriptor.geometryDigestV2Hex
            guard matchesDigest else {
                _ = munmap(ptr, size)
                throw QWLSASidecarError.geometryMismatch(
                    sidecar: decoded.provenance.geometryDigest,
                    expected: descriptor.geometryDigestV3Hex
                )
            }
        }
        self.header = decoded
    }

    public var info: QWLSASidecarInfo {
        let p = header.provenance
        return QWLSASidecarInfo(
            path: url.path,
            modelId: p.modelId,
            geometryDigest: p.geometryDigest,
            method: p.method,
            indexDim: p.indexDim,
            poolSize: p.poolSize,
            clusterBlockSize: p.clusterBlockSize,
            bucketsPerBlock: p.bucketsPerBlock,
            maxBucketKeys: p.maxBucketKeys,
            recommendedBudget: p.recommendedBudget,
            geometryMatches: true
        )
    }

    public func entry(_ name: String) throws -> TensorEntry {
        guard let e = header.tensors[name] else {
            throw QWLSASidecarError.tensorMissing(name)
        }
        return e
    }

    /// Reads a sidecar tensor as an array of `UInt16` (`Float16` bitPatterns).
    public func readF16Tensor(_ name: String, expectedShape: [Int]? = nil) throws -> [UInt16] {
        let e = try entry(name)
        if let expectedShape, e.shape != expectedShape {
            throw QWLSASidecarError.shapeMismatch(tensor: name, got: e.shape, expected: expectedShape)
        }
        let elems = e.shape.reduce(1, *)
        let byteCount = elems * 2
        guard e.rawOff >= 16, e.rawOff + byteCount <= rawCount else {
            throw QWLSASidecarError.truncated(url.path)
        }
        var out = [UInt16](repeating: 0, count: elems)
        out.withUnsafeMutableBytes { (buf: UnsafeMutableRawBufferPointer) in
            buf.baseAddress!.copyMemory(from: UnsafeRawPointer(rawBase).advanced(by: e.rawOff), byteCount: byteCount)
        }
        return out
    }

    /// Writes a `QLSA0001` sidecar file atomically.
    public static func write(
        to url: URL,
        provenance: Header.Provenance,
        layerTensors: [Int: (wq: [UInt16], wk: [UInt16])],
        wqShape: [Int],
        wkShape: [Int]
    ) throws {
        var payload = Data()
        // Reserve 16-byte prefix (8B magic + 8B trailer length) + pad to 64 bytes
        payload.append(contentsOf: Array(magic.utf8))
        payload.append(contentsOf: [UInt8](repeating: 0, count: 8))
        payload.append(contentsOf: [UInt8](repeating: 0, count: 48)) // offset 64

        var entries: [String: TensorEntry] = [:]
        for layer in layerTensors.keys.sorted() {
            guard let pair = layerTensors[layer] else { continue }
            let wqName = "layers.\(layer).wq_idx"
            let wkName = "layers.\(layer).wk_idx"

            for (name, data, fallbackShape) in [(wqName, pair.wq, wqShape), (wkName, pair.wk, wkShape)] {
                let pad = (64 - (payload.count % 64)) % 64
                if pad > 0 {
                    payload.append(contentsOf: [UInt8](repeating: 0, count: pad))
                }
                let rawOff = payload.count
                let nBytes = data.count * 2
                data.withUnsafeBytes { (rawBuf: UnsafeRawBufferPointer) in
                    payload.append(rawBuf.bindMemory(to: UInt8.self))
                }
                let resolvedShape: [Int]
                if fallbackShape.reduce(1, *) == data.count {
                    resolvedShape = fallbackShape
                } else if provenance.indexDim > 0 && data.count % provenance.indexDim == 0 {
                    resolvedShape = [provenance.indexDim, data.count / provenance.indexDim]
                } else {
                    resolvedShape = fallbackShape
                }
                entries[name] = TensorEntry(shape: resolvedShape, dtype: "f16", rawOff: rawOff, nBytes: nBytes)
            }
        }

        let header = Header(provenance: provenance, tensors: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let hdrData = try encoder.encode(header)
        let hdrLen = UInt64(hdrData.count)
        for i in 0..<8 {
            payload[8 + i] = UInt8((hdrLen >> (8 * i)) & 0xFF)
        }
        payload.append(hdrData)
        try payload.write(to: url, options: .atomic)
    }

}
