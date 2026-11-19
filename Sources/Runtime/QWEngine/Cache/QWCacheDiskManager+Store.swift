// QWCacheDiskManager+Store.swift — Direct KV pool store/restore entrypoints for QWCacheDiskManager.

import Foundation
import QWCore
import QWHAL
import QWKV

extension QWCacheDiskManager {
    /// Reads SSD cache directly into KV pool pages via F_NOCACHE DispatchIO with zero intermediate KV heap copies.
    func readDirectIntoCache(
        hit: DiskHit,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?
    ) throws -> (
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunk: QWKVSharedPrefix,
        mtp: QWKVSharedPrefix?
    ) {
        let read = try Self.readStandaloneFile(
            at: hit.fileURL,
            expectedMetadata: hit.metadata,
            kvCache: kvCache,
            mtpCache: mtpCache
        )
        return (read.state, read.nextToken, read.logits, read.trunk, read.mtp)
    }

    /// Writes prefix state to SSD via QWDispatchIO.
    func write(tokens: [Int], mode: String, nextToken: Int,
               state: QWEnginePrefixState, logits: Data,
               trunkPlanes: [Data], mtpPlanes: [Data],
               allowsSuffix: Bool) throws -> URL? {
        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)
        let tempURL = config.cacheDirectory.appendingPathComponent("\(filename).tmp.\(UUID().uuidString)")
        let built = try Self.buildMetadataAndHeader(
            tokens: tokens, mode: mode, nextToken: nextToken, state: state, logits: logits,
            trunkPlaneSizes: trunkPlanes.map { $0.count },
            mtpPlaneSizes: mtpPlanes.map { $0.count },
            allowsSuffix: allowsSuffix
        )
        guard index.canAccept(bytes: built.totalBytes) else { return nil }

        // Assemble buffer outside the index lock so NVMe I/O never stalls lookups.
        var buffer = Data(capacity: built.totalBytes)
        buffer.append(built.headerBuffer)
        for plane in trunkPlanes { buffer.append(plane) }
        for plane in mtpPlanes { buffer.append(plane) }

        defer { try? FileManager.default.removeItem(at: tempURL) }
        try FileManager.default.createDirectory(
            at: config.cacheDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        do {
            let io = try QWDispatchIO(path: tempURL.path, mode: .writeOnly, directIO: true)
            defer { io.close() }
            if !buffer.isEmpty {
                try buffer.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    if let base = raw.baseAddress {
                        try io.writeSync(offset: 0, buffer: base, length: raw.count)
                    }
                }
            }
        }
        return try commitWrittenEntry(
            tempURL: tempURL, fileURL: fileURL, key: key,
            metadata: built.metadata, totalBytes: built.totalBytes
        )
    }

    /// Reads deserialized prefix payload from SSD via QWDispatchIO.
    func read(hit: DiskHit) throws -> (
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunkPlanes: [Data],
        mtpPlanes: [Data]
    ) {
        let meta = hit.metadata
        let io = try QWDispatchIO(path: hit.fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        let header = try Self.readHeader(io, expected: meta)
        let decoded = try Self.readHeaderSections(io: io, meta: meta, startOffset: header.nextOffset)
        var offset = decoded.kvOffset

        func readPlanes(_ sizes: [Int]) throws -> [Data] {
            var planes: [Data] = []
            planes.reserveCapacity(sizes.count)
            for size in sizes {
                var plane = Data(count: size)
                if size > 0 {
                    try plane.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                        if let base = raw.baseAddress {
                            try Self.readExactly(io, offset: offset, into: base, length: size)
                        }
                    }
                }
                planes.append(plane)
                offset += Int64(size)
            }
            return planes
        }

        let trunkPlanes = try readPlanes(meta.trunkPlaneSizes)
        let mtpPlanes = meta.hasMtp ? try readPlanes(meta.mtpPlaneSizes) : []
        return (decoded.state, meta.nextToken, decoded.logits, trunkPlanes, mtpPlanes)
    }

}
