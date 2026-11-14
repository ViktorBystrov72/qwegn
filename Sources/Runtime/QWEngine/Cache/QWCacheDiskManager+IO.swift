// QWCacheDiskManager+IO.swift — Standalone DispatchIO file serialization for QWCacheDiskManager.

import Foundation
import QWCore
import QWHAL
import QWKV

extension QWCacheDiskManager {
    /// Streams shared prefix directly from unified memory / KV pools to an arbitrary `.qwcache` file URL.
    static func writeStandaloneFile(
        to fileURL: URL,
        tokens: [Int],
        mode: String,
        nextToken: Int,
        state: QWEnginePrefixState,
        logits: Data,
        trunk: QWKVSharedPrefix,
        kvCache: QWKVCache,
        mtp: QWKVSharedPrefix?,
        mtpCache: QWKVCache?,
        allowsSuffix: Bool,
        maxBytes: Int? = nil,
        onBeforeWrite: ((Int) -> Void)? = nil
    ) throws -> (metadata: StoredMetadata, totalBytes: Int)? {
        var trunkPlaneSizes: [Int] = []
        var trunkPageCounts: [Int: Int] = [:]
        try kvCache.withSharedPrefixPagePointers(trunk) { planeIdx, _, _, byteCount in
            trunkPageCounts[planeIdx, default: 0] += byteCount
        }
        for i in 0..<(trunkPageCounts.count) {
            trunkPlaneSizes.append(trunkPageCounts[i] ?? 0)
        }

        var mtpPlaneSizes: [Int] = []
        if let mtp, let mtpCache {
            var mtpPageCounts: [Int: Int] = [:]
            try mtpCache.withSharedPrefixPagePointers(mtp) { planeIdx, _, _, byteCount in
                mtpPageCounts[planeIdx, default: 0] += byteCount
            }
            for i in 0..<(mtpPageCounts.count) {
                mtpPlaneSizes.append(mtpPageCounts[i] ?? 0)
            }
        }

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
            trunkPlaneSizes: trunkPlaneSizes,
            hasMtp: !mtpPlaneSizes.isEmpty,
            mtpTokenCount: state.roundRecord.mtpPhysicalRows,
            mtpPlaneSizes: mtpPlaneSizes,
            allowsSuffix: allowsSuffix,
            formatVersion: 2,
            mlaIndexerBytes: state.mlaIndexerData?.count
        )

        let metaData = try JSONEncoder().encode(meta)
        var metaLen = UInt64(metaData.count)
        let headerRawLen = MemoryLayout<UInt64>.size + metaData.count +
            state.recData.count + state.ringData.count + (state.trunkHiddenData?.count ?? 0) +
            (state.mlaIndexerData?.count ?? 0) + logits.count
        let alignedHeaderLen = Self.alignUp(headerRawLen)
        let kvPayloadBytes = trunkPlaneSizes.reduce(0, +) + mtpPlaneSizes.reduce(0, +)
        let totalPayloadBytes = alignedHeaderLen + kvPayloadBytes

        if let maxBytes, totalPayloadBytes > maxBytes { return nil }
        onBeforeWrite?(totalPayloadBytes)

        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let io = try QWDispatchIO(path: fileURL.path, mode: .writeOnly, directIO: true)
        defer { io.close() }

        var headerBuf = Data(capacity: alignedHeaderLen)
        withUnsafeBytes(of: &metaLen) { (raw: UnsafeRawBufferPointer) in headerBuf.append(contentsOf: raw) }
        headerBuf.append(metaData)
        headerBuf.append(state.recData)
        headerBuf.append(state.ringData)
        if let hidden = state.trunkHiddenData {
            headerBuf.append(hidden)
        }
        if let mla = state.mlaIndexerData {
            headerBuf.append(mla)
        }
        headerBuf.append(logits)
        if headerBuf.count < alignedHeaderLen {
            headerBuf.append(Data(count: alignedHeaderLen - headerBuf.count))
        }

        try headerBuf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            try io.writeSync(offset: 0, buffer: raw.baseAddress!, length: raw.count)
        }

        var writeOffset = Int64(alignedHeaderLen)
        try kvCache.withSharedPrefixPagePointers(trunk) { _, _, ptr, byteCount in
            try io.writeSync(offset: writeOffset, buffer: ptr, length: byteCount)
            writeOffset += Int64(byteCount)
        }
        if let mtp, let mtpCache {
            try mtpCache.withSharedPrefixPagePointers(mtp) { _, _, ptr, byteCount in
                try io.writeSync(offset: writeOffset, buffer: ptr, length: byteCount)
                writeOffset += Int64(byteCount)
            }
        }
        return (meta, totalPayloadBytes)
    }

    /// Streams shared prefix directly from unified memory / KV pools to SSD with F_NOCACHE direct I/O.
    func writeDirect(
        tokens: [Int],
        mode: String,
        nextToken: Int,
        state: QWEnginePrefixState,
        logits: Data,
        trunk: QWKVSharedPrefix,
        kvCache: QWKVCache,
        mtp: QWKVSharedPrefix?,
        mtpCache: QWKVCache?,
        allowsSuffix: Bool
    ) throws -> URL? {
        lock.lock()
        defer { lock.unlock() }

        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)

        guard let written = try Self.writeStandaloneFile(
            to: fileURL,
            tokens: tokens,
            mode: mode,
            nextToken: nextToken,
            state: state,
            logits: logits,
            trunk: trunk,
            kvCache: kvCache,
            mtp: mtp,
            mtpCache: mtpCache,
            allowsSuffix: allowsSuffix,
            maxBytes: config.ssdByteLimit,
            onBeforeWrite: { [weak self] totalPayloadBytes in
                guard let self else { return }
                while self.chargedBytes + totalPayloadBytes > self.config.ssdByteLimit, !self.lru.isEmpty {
                    self.evictOldest()
                }
            }
        ) else { return nil }

        if let prevBytes = entryBytes[key] {
            chargedBytes -= prevBytes
        }
        entries[key] = written.metadata
        entryFiles[key] = fileURL
        entryBytes[key] = written.totalBytes
        chargedBytes += written.totalBytes
        touch(key)
        return fileURL
    }

    /// Reads an arbitrary `.qwcache` file directly into KV pool pages via F_NOCACHE DispatchIO.
    static func readStandaloneFile(
        at fileURL: URL,
        expectedMetadata: StoredMetadata? = nil,
        kvCache: QWKVCache,
        mtpCache: QWKVCache?
    ) throws -> (
        metadata: StoredMetadata,
        state: QWEnginePrefixState,
        nextToken: Int,
        logits: Data,
        trunk: QWKVSharedPrefix,
        mtp: QWKVSharedPrefix?
    ) {
        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        let header = try Self.readHeader(io, expected: expectedMetadata)
        let meta = header.meta
        var offset = header.nextOffset

        var recData = Data(count: meta.recBytes)
        if meta.recBytes > 0 {
            try recData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    try Self.readExactly(io, offset: offset, into: base, length: meta.recBytes)
                }
            }
        }
        offset += Int64(meta.recBytes)

        var ringData = Data(count: meta.ringBytes)
        if meta.ringBytes > 0 {
            try ringData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    try Self.readExactly(io, offset: offset, into: base, length: meta.ringBytes)
                }
            }
        }
        offset += Int64(meta.ringBytes)

        var trunkHiddenData: Data? = nil
        if meta.trunkHiddenBytes > 0 {
            var hidden = Data(count: meta.trunkHiddenBytes)
            try hidden.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                try Self.readExactly(io, offset: offset, into: raw.baseAddress!, length: meta.trunkHiddenBytes)
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

        var logits = Data(count: meta.logitsBytes)
        try logits.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            if let base = raw.baseAddress {
                try Self.readExactly(io, offset: offset, into: base, length: meta.logitsBytes)
            }
        }
        offset += Int64(meta.logitsBytes)

        if (meta.formatVersion ?? 1) >= 2 {
            offset = Int64(Self.alignUp(Int(offset)))
        }

        let tokenCount = meta.position
        let tempH = try kvCache.allocate(promptTokens: tokenCount)
        defer { kvCache.finish(tempH) }
        try kvCache.populatePrivatePagesDirectly(into: tempH, tokenCount: tokenCount) { _, _, ptr, byteCount in
            try Self.readExactly(io, offset: offset, into: ptr, length: byteCount)
            offset += Int64(byteCount)
        }
        let trunkShared = try kvCache.captureSharedPrefix(tempH)

        var mtpShared: QWKVSharedPrefix? = nil
        if let mtpCache, meta.hasMtp, !meta.mtpPlaneSizes.isEmpty {
            let mtpCount = meta.roundRecord.mtpPhysicalRows
            let tempM = try mtpCache.allocate(promptTokens: mtpCount)
            defer { mtpCache.finish(tempM) }
            try mtpCache.populatePrivatePagesDirectly(into: tempM, tokenCount: mtpCount) { _, _, ptr, byteCount in
                try Self.readExactly(io, offset: offset, into: ptr, length: byteCount)
                offset += Int64(byteCount)
            }
            mtpShared = try mtpCache.captureSharedPrefix(tempM)
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

        return (meta, prefixState, meta.nextToken, logits, trunkShared, mtpShared)
    }

}
