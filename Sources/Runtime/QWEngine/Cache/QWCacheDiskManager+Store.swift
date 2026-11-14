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
        lock.lock()
        defer { lock.unlock() }
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
        lock.lock()
        defer { lock.unlock() }

        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)

        let meta = StoredMetadata(
            tokens: tokens,
            mode: mode,
            nextToken: nextToken,
            position: state.position,
            generated: state.generated,
            roundRecord: state.roundRecord,
            recBytes: state.recData.count,
            ringBytes: state.ringData.count,
            trunkHiddenBytes: state.trunkHiddenData?.count ?? 0,
            trunkHiddenValid: state.trunkHiddenValid,
            logitsBytes: logits.count,
            trunkTokenCount: state.position,
            trunkPlaneSizes: trunkPlanes.map { $0.count },
            hasMtp: !mtpPlanes.isEmpty,
            mtpTokenCount: state.roundRecord.mtpPhysicalRows,
            mtpPlaneSizes: mtpPlanes.map { $0.count },
            allowsSuffix: allowsSuffix,
            formatVersion: 2,
            mlaIndexerBytes: state.mlaIndexerData?.count
        )

        let encoder = JSONEncoder()
        let metaData = try encoder.encode(meta)
        var metaLen = UInt64(metaData.count)

        let headerRawLen = MemoryLayout<UInt64>.size + metaData.count +
            state.recData.count + state.ringData.count + (state.trunkHiddenData?.count ?? 0) +
            (state.mlaIndexerData?.count ?? 0) + logits.count
        let alignedHeaderLen = Self.alignUp(headerRawLen)
        let kvBytes = trunkPlanes.reduce(0, { $0 + $1.count }) + mtpPlanes.reduce(0, { $0 + $1.count })
        let totalPayloadBytes = alignedHeaderLen + kvBytes

        // Check against SSD capacity limit
        guard totalPayloadBytes <= config.ssdByteLimit else { return nil }

        // Evict oldest until within SSD capacity
        while chargedBytes + totalPayloadBytes > config.ssdByteLimit, !lru.isEmpty {
            evictOldest()
        }

        // Assemble buffer
        var buffer = Data(capacity: totalPayloadBytes)
        withUnsafeBytes(of: &metaLen) { (raw: UnsafeRawBufferPointer) in buffer.append(contentsOf: raw) }
        buffer.append(metaData)
        buffer.append(state.recData)
        buffer.append(state.ringData)
        if let hidden = state.trunkHiddenData {
            buffer.append(hidden)
        }
        if let mla = state.mlaIndexerData {
            buffer.append(mla)
        }
        buffer.append(logits)
        if buffer.count < alignedHeaderLen {
            buffer.append(Data(count: alignedHeaderLen - buffer.count))
        }
        for plane in trunkPlanes { buffer.append(plane) }
        for plane in mtpPlanes { buffer.append(plane) }

        // Write using QWDispatchIO with F_NOCACHE directIO
        let io = try QWDispatchIO(path: fileURL.path, mode: .writeOnly, directIO: true)
        defer { io.close() }
        try buffer.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            try io.writeSync(offset: 0, buffer: raw.baseAddress!, length: raw.count)
        }

        if let prevBytes = entryBytes[key] {
            chargedBytes -= prevBytes
        }
        entries[key] = meta
        entryFiles[key] = fileURL
        entryBytes[key] = totalPayloadBytes
        chargedBytes += totalPayloadBytes
        touch(key)
        return fileURL
    }

    /// Reads deserialized prefix payload from SSD via QWDispatchIO.
    func read(hit: DiskHit) throws -> (
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunkPlanes: [Data],
        mtpPlanes: [Data]
    ) {
        lock.lock()
        defer { lock.unlock() }

        let meta = hit.metadata
        let fileURL = hit.fileURL
        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        var offset = try Self.readHeader(io, expected: meta).nextOffset

        // Read recData
        var recData = Data(count: meta.recBytes)
        if meta.recBytes > 0 {
            try recData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    _ = try io.readSync(offset: offset, into: base, length: meta.recBytes)
                }
            }
        }
        offset += Int64(meta.recBytes)

        // Read ringData
        var ringData = Data(count: meta.ringBytes)
        if meta.ringBytes > 0 {
            try ringData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    _ = try io.readSync(offset: offset, into: base, length: meta.ringBytes)
                }
            }
        }
        offset += Int64(meta.ringBytes)

        // Read trunkHiddenData
        var trunkHiddenData: Data? = nil
        if meta.trunkHiddenBytes > 0 {
            var hidden = Data(count: meta.trunkHiddenBytes)
            try hidden.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                _ = try io.readSync(offset: offset, into: raw.baseAddress!, length: meta.trunkHiddenBytes)
            }
            trunkHiddenData = hidden
            offset += Int64(meta.trunkHiddenBytes)
        }

        var mlaIndexerData: Data? = nil
        if let mlaBytes = meta.mlaIndexerBytes, mlaBytes > 0 {
            var mla = Data(count: mlaBytes)
            try mla.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                try Self.readExactly(io, offset: offset, into: raw.baseAddress!, length: mlaBytes)
            }
            mlaIndexerData = mla
            offset += Int64(mlaBytes)
        }

        // Read logits
        var logits = Data(count: meta.logitsBytes)
        try logits.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            _ = try io.readSync(offset: offset, into: raw.baseAddress!, length: meta.logitsBytes)
        }
        offset += Int64(meta.logitsBytes)

        if (meta.formatVersion ?? 1) >= 2 {
            offset = Int64(Self.alignUp(Int(offset)))
        }

        // Read trunkPlanes
        var trunkPlanes: [Data] = []
        trunkPlanes.reserveCapacity(meta.trunkPlaneSizes.count)
        for size in meta.trunkPlaneSizes {
            var plane = Data(count: size)
            try plane.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                _ = try io.readSync(offset: offset, into: raw.baseAddress!, length: size)
            }
            trunkPlanes.append(plane)
            offset += Int64(size)
        }

        // Read mtpPlanes
        var mtpPlanes: [Data] = []
        if meta.hasMtp {
            mtpPlanes.reserveCapacity(meta.mtpPlaneSizes.count)
            for size in meta.mtpPlaneSizes {
                var plane = Data(count: size)
                try plane.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    _ = try io.readSync(offset: offset, into: raw.baseAddress!, length: size)
                }
                mtpPlanes.append(plane)
                offset += Int64(size)
            }
        }

        let prefixState = QWEnginePrefixState(
            position: meta.position,
            generated: meta.generated,
            roundRecord: meta.roundRecord,
            recData: recData,
            ringData: ringData,
            trunkHiddenData: trunkHiddenData,
            trunkHiddenValid: meta.trunkHiddenValid,
            stochastic: nil,
            mlaIndexerData: mlaIndexerData
        )

        return (prefixState, meta.nextToken, logits, trunkPlanes, mtpPlanes)
    }

}
