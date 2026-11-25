import Foundation
import QWCore
import QWHAL
import QWKV

/// Manages ephemeral cold Tier-3 prefix cache storage on NVMe SSD.
/// Uses unbuffered direct I/O via QWDispatchIO for high-throughput streaming.
public final class QWCacheDiskManager: @unchecked Sendable {
    enum ReadError: Error, CustomStringConvertible {
        case invalidHeader
        case metadataMismatch
        case truncated(offset: Int64, expected: Int, actual: Int)

        var description: String {
            switch self {
            case .invalidHeader: return "invalid cache metadata header"
            case .metadataMismatch: return "cache metadata differs from its index"
            case .truncated(let offset, let expected, let actual):
                return "short cache read at \(offset): expected \(expected) bytes, got \(actual)"
            }
        }
    }

    static func readExactly(_ io: QWDispatchIO, offset: Int64,
                            into destination: UnsafeMutableRawPointer, length: Int) throws {
        let actual = try io.readSync(offset: offset, into: destination, length: length)
        guard actual == length else {
            throw ReadError.truncated(offset: offset, expected: length, actual: actual)
        }
    }

    public struct StoredMetadata: Codable, Sendable, Equatable {
        public let tokens: [Int]
        public let mode: String
        public let nextToken: Int
        public let position: Int
        public let generated: Int
        public let roundRecord: QWRoundRecord
        public let recBytes: Int
        public let ringBytes: Int
        public let trunkHiddenBytes: Int
        public let trunkHiddenValid: Bool
        public let logitsBytes: Int
        public let trunkTokenCount: Int
        public let trunkPlaneSizes: [Int]
        public let hasMtp: Bool
        public let mtpTokenCount: Int
        public let mtpPlaneSizes: [Int]
        public let allowsSuffix: Bool
        public var formatVersion: Int? = nil
        public var mlaIndexerBytes: Int? = nil
        public var dflashKBytes: Int? = nil
        public var dflashVBytes: Int? = nil
        public var dflashCacheOffset: Int? = nil
        public var dsparkRingBytes: Int? = nil
        public var dsparkCacheOffset: Int? = nil
    }

    public struct DiskHit: Sendable {
        public let metadata: StoredMetadata
        public let fileURL: URL
        public let matchedTokens: Int
    }

    public let config: QWCacheConfig
    let index: QWColdCacheIndex<StoredMetadata>
    public var chargedBytes: Int { index.chargedBytes }
    public var entryCount: Int { index.entryCount }

    public init(config: QWCacheConfig) {
        self.config = config
        self.index = QWColdCacheIndex<StoredMetadata>(
            ssdByteLimit: config.ssdByteLimit,
            entryLimit: Int.max
        )
        prepareDirectory()
        if config.wipeOnStart {
            purgeAll()
        } else {
            loadDiskIndex()
        }
    }

    deinit {
        if config.wipeOnStart {
            purgeAll()
        }
    }

    public func prepareDirectory() {
        try? FileManager.default.createDirectory(at: config.cacheDirectory,
                                                 withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
    }

    /// Rebuilds the in-memory index from existing `.qwcache` files on SSD.
    public func loadDiskIndex() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: config.cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
        ) else { return }

        let decoder = JSONDecoder()
        var discovered: [(key: String, metadata: StoredMetadata, fileURL: URL, bytes: Int, date: Date)] = []

        for file in files where file.pathExtension == "qwcache" && file.lastPathComponent.hasPrefix("entry_") {
            guard let io = try? QWDispatchIO(path: file.path, mode: .readOnly, directIO: false) else { continue }
            defer { io.close() }
            var metaLen: UInt64 = 0
            let readLen = withUnsafeMutableBytes(of: &metaLen) { (raw: UnsafeMutableRawBufferPointer) in
                (try? io.readSync(offset: 0, into: raw.baseAddress!, length: MemoryLayout<UInt64>.size)) ?? 0
            }
            guard readLen == MemoryLayout<UInt64>.size, metaLen > 0, metaLen < 16 * 1024 * 1024 else { continue }
            var metaData = Data(count: Int(metaLen))
            let actualMeta = metaData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                (try? io.readSync(offset: Int64(MemoryLayout<UInt64>.size), into: raw.baseAddress!, length: Int(metaLen))) ?? 0
            }
            guard actualMeta == Int(metaLen),
                  let meta = try? decoder.decode(StoredMetadata.self, from: metaData) else { continue }

            let key = Self.cacheKey(tokens: meta.tokens, mode: meta.mode)
            let resourceValues = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            let fileSize = resourceValues?.fileSize ?? (
                MemoryLayout<UInt64>.size + Int(metaLen) + meta.recBytes + meta.ringBytes +
                meta.trunkHiddenBytes + (meta.mlaIndexerBytes ?? 0) +
                (meta.dflashKBytes ?? 0) + (meta.dflashVBytes ?? 0) + meta.logitsBytes +
                meta.trunkPlaneSizes.reduce(0, +) + meta.mtpPlaneSizes.reduce(0, +)
            )
            let modDate = resourceValues?.contentModificationDate ?? Date.distantPast
            discovered.append((key, meta, file, fileSize, modDate))
        }

        discovered.sort { $0.date < $1.date }
        index.bootstrap(
            items: discovered.map { ($0.key, $0.metadata, $0.fileURL, $0.bytes) },
            onEvict: { try? FileManager.default.removeItem(at: $0) }
        )
    }

    /// Purges all cache files and resets index.
    public func purgeAll() {
        let fm = FileManager.default
        if let files = try? fm.contentsOfDirectory(at: config.cacheDirectory, includingPropertiesForKeys: nil) {
            for file in files where file.pathExtension == "qwcache" || file.lastPathComponent.hasPrefix("entry_") {
                try? fm.removeItem(at: file)
            }
        }
        index.purgeAll { try? fm.removeItem(at: $0) }
    }

    /// Deterministic FNV-1a-64 hash across process restarts.
    public static func cacheKey(tokens: [Int], mode: String) -> String {
        let hash = QWPrefixHash.fnv1a64(tokens: tokens, mode: mode)
        return String(format: "%016llx_%d", hash, tokens.count)
    }

    static func alignUp(_ value: Int, to alignment: Int = QWDispatchIO.pageSize) -> Int {
        let mask = alignment - 1
        return (value + mask) & ~mask
    }

    static func readHeader(_ io: QWDispatchIO, expected: StoredMetadata? = nil) throws -> (meta: StoredMetadata, nextOffset: Int64) {
        var length: UInt64 = 0
        try withUnsafeMutableBytes(of: &length) { (raw: UnsafeMutableRawBufferPointer) in
            try readExactly(io, offset: 0, into: raw.baseAddress!, length: MemoryLayout<UInt64>.size)
        }
        guard length > 0, length < 16 * 1024 * 1024 else { throw ReadError.invalidHeader }
        var data = Data(count: Int(length))
        try data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            try readExactly(io, offset: Int64(MemoryLayout<UInt64>.size),
                            into: raw.baseAddress!, length: raw.count)
        }
        let decoded = try JSONDecoder().decode(StoredMetadata.self, from: data)
        if let expected, decoded != expected {
            throw ReadError.metadataMismatch
        }
        return (decoded, Int64(MemoryLayout<UInt64>.size) + Int64(length))
    }

    /// Look up a matching prefix on SSD (exact or longest prefix if allowed).
    public func lookup(tokens: [Int], mode: String) -> DiskHit? {
        let exact = Self.cacheKey(tokens: tokens, mode: mode)
        if let hit = index.lookupExact(exact, verify: { $0.tokens == tokens && $0.mode == mode }) {
            return DiskHit(metadata: hit.metadata, fileURL: hit.fileURL, matchedTokens: tokens.count)
        }
        guard let prefix = index.lookupBestPrefix(scoreMatch: { _, meta in
            guard meta.allowsSuffix, meta.mode == mode, meta.tokens.count < tokens.count,
                  tokens.starts(with: meta.tokens) else { return nil }
            return meta.tokens.count
        }) else {
            return nil
        }
        return DiskHit(metadata: prefix.metadata, fileURL: prefix.fileURL, matchedTokens: prefix.matchedTokens)
    }

    func commitWrittenEntry(
        tempURL: URL,
        fileURL: URL,
        key: String,
        metadata: StoredMetadata,
        totalBytes: Int
    ) throws -> URL? {
        let stored = try index.record(
            key: key,
            metadata: metadata,
            fileURL: fileURL,
            bytes: totalBytes,
            beforeInstall: {
                _ = try? FileManager.default.removeItem(at: fileURL)
                try FileManager.default.moveItem(at: tempURL, to: fileURL)
            },
            onEvict: { try? FileManager.default.removeItem(at: $0) }
        )
        return stored ? fileURL : nil
    }
}
