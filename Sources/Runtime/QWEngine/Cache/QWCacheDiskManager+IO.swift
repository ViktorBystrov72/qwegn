// QWCacheDiskManager+IO.swift — Standalone DispatchIO file serialization for QWCacheDiskManager.

import Foundation
import QWCore
import QWHAL
import QWKV

extension QWCacheDiskManager {
    static func buildMetadataAndHeader(
        tokens: [Int],
        mode: String,
        nextToken: Int,
        state: QWEnginePrefixState,
        logits: Data,
        trunkPlaneSizes: [Int],
        mtpPlaneSizes: [Int],
        allowsSuffix: Bool
    ) throws -> (metadata: StoredMetadata, headerBuffer: Data, totalBytes: Int) {
        let dflashActive = state.dflashCacheOffset > 0
            && state.dflashKData != nil && state.dflashVData != nil
        let dflashK = dflashActive ? state.dflashKData : nil
        let dflashV = dflashActive ? state.dflashVData : nil
        let dflashOffset = dflashActive ? state.dflashCacheOffset : nil

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
            mlaIndexerBytes: state.mlaIndexerData?.count,
            dflashKBytes: dflashK?.count,
            dflashVBytes: dflashV?.count,
            dflashCacheOffset: dflashOffset
        )

        let metaData = try JSONEncoder().encode(meta)
        var metaLen = UInt64(metaData.count)
        let headerRawLen = MemoryLayout<UInt64>.size + metaData.count
            + state.recData.count + state.ringData.count + (state.trunkHiddenData?.count ?? 0)
            + (state.mlaIndexerData?.count ?? 0) + (dflashK?.count ?? 0) + (dflashV?.count ?? 0)
            + logits.count
        let alignedHeaderLen = Self.alignUp(headerRawLen)
        let kvPayloadBytes = trunkPlaneSizes.reduce(0, +) + mtpPlaneSizes.reduce(0, +)
        let totalPayloadBytes = alignedHeaderLen + kvPayloadBytes

        var headerBuf = Data(capacity: alignedHeaderLen)
        withUnsafeBytes(of: &metaLen) { (raw: UnsafeRawBufferPointer) in headerBuf.append(contentsOf: raw) }
        headerBuf.append(metaData)
        headerBuf.append(state.recData)
        headerBuf.append(state.ringData)
        if let hidden = state.trunkHiddenData { headerBuf.append(hidden) }
        if let mla = state.mlaIndexerData { headerBuf.append(mla) }
        if let dflashK { headerBuf.append(dflashK) }
        if let dflashV { headerBuf.append(dflashV) }
        headerBuf.append(logits)
        if headerBuf.count < alignedHeaderLen {
            headerBuf.append(Data(count: alignedHeaderLen - headerBuf.count))
        }
        return (meta, headerBuf, totalPayloadBytes)
    }

    private static func readSection(_ io: QWDispatchIO, offset: inout Int64, length: Int) throws -> Data {
        var buf = Data(count: length)
        if length > 0 {
            try buf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                if let base = raw.baseAddress {
                    try Self.readExactly(io, offset: offset, into: base, length: length)
                }
            }
        }
        offset += Int64(length)
        return buf
    }

    static func readHeaderSections(
        io: QWDispatchIO,
        meta: StoredMetadata,
        startOffset: Int64
    ) throws -> (state: QWEnginePrefixState, logits: Data, kvOffset: Int64) {
        guard meta.recBytes >= 0, meta.ringBytes >= 0, meta.trunkHiddenBytes >= 0, meta.logitsBytes >= 0,
              (meta.mlaIndexerBytes ?? 0) >= 0,
              (meta.dflashKBytes ?? 0) >= 0, (meta.dflashVBytes ?? 0) >= 0,
              meta.trunkPlaneSizes.allSatisfy({ $0 >= 0 }),
              meta.mtpPlaneSizes.allSatisfy({ $0 >= 0 }) else {
            throw ReadError.invalidHeader
        }
        var offset = startOffset
        let recData = try readSection(io, offset: &offset, length: meta.recBytes)
        let ringData = try readSection(io, offset: &offset, length: meta.ringBytes)
        let trunkHiddenData = meta.trunkHiddenBytes > 0
            ? try readSection(io, offset: &offset, length: meta.trunkHiddenBytes) : nil
        let mlaIndexerData = (meta.mlaIndexerBytes ?? 0) > 0
            ? try readSection(io, offset: &offset, length: meta.mlaIndexerBytes ?? 0) : nil
        let dflashKData = (meta.dflashKBytes ?? 0) > 0
            ? try readSection(io, offset: &offset, length: meta.dflashKBytes ?? 0) : nil
        let dflashVData = (meta.dflashVBytes ?? 0) > 0
            ? try readSection(io, offset: &offset, length: meta.dflashVBytes ?? 0) : nil
        let logits = try readSection(io, offset: &offset, length: meta.logitsBytes)
        if (meta.formatVersion ?? 1) >= 2 {
            offset = Int64(Self.alignUp(Int(offset)))
        }
        let restoredDFlashValid = (meta.dflashCacheOffset ?? 0) > 0 && dflashKData != nil && dflashVData != nil
        let state = QWEnginePrefixState(
            position: meta.position,
            generated: meta.generated,
            roundRecord: meta.roundRecord,
            recData: recData,
            ringData: ringData,
            trunkHiddenData: trunkHiddenData,
            trunkHiddenValid: meta.trunkHiddenValid,
            stochastic: nil,
            dflashKData: restoredDFlashValid ? dflashKData : nil,
            dflashVData: restoredDFlashValid ? dflashVData : nil,
            dflashCacheOffset: restoredDFlashValid ? (meta.dflashCacheOffset ?? 0) : 0,
            mlaIndexerData: mlaIndexerData
        )
        return (state, logits, offset)
    }

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
        func collectPlaneSizes(_ cache: QWKVCache, _ prefix: QWKVSharedPrefix) throws -> [Int] {
            var counts: [Int: Int] = [:]
            try cache.withSharedPrefixPagePointers(prefix) { planeIdx, _, _, byteCount in
                counts[planeIdx, default: 0] += byteCount
            }
            return (0..<counts.count).map { counts[$0] ?? 0 }
        }
        let trunkPlaneSizes = try collectPlaneSizes(kvCache, trunk)
        let mtpPlaneSizes = try (mtp.flatMap { p in try mtpCache.map { try collectPlaneSizes($0, p) } }) ?? []

        let built = try buildMetadataAndHeader(
            tokens: tokens, mode: mode, nextToken: nextToken, state: state,
            logits: logits, trunkPlaneSizes: trunkPlaneSizes, mtpPlaneSizes: mtpPlaneSizes,
            allowsSuffix: allowsSuffix
        )
        if let maxBytes, built.totalBytes > maxBytes { return nil }
        onBeforeWrite?(built.totalBytes)

        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let io = try QWDispatchIO(path: fileURL.path, mode: .writeOnly, directIO: true)
        defer { io.close() }

        try built.headerBuffer.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let base = raw.baseAddress {
                try io.writeSync(offset: 0, buffer: base, length: raw.count)
            }
        }
        var writeOffset = Int64(built.headerBuffer.count)
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
        return (built.metadata, built.totalBytes)
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
        guard config.ssdByteLimit > 0 else { return nil }
        let key = Self.cacheKey(tokens: tokens, mode: mode)
        let filename = "entry_\(key).qwcache"
        let fileURL = config.cacheDirectory.appendingPathComponent(filename)
        let tempURL = config.cacheDirectory.appendingPathComponent("\(filename).tmp.\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        guard let written = try Self.writeStandaloneFile(
            to: tempURL,
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
            onBeforeWrite: nil
        ) else { return nil }

        return try commitWrittenEntry(
            tempURL: tempURL, fileURL: fileURL, key: key,
            metadata: written.metadata, totalBytes: written.totalBytes
        )
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
        let decoded = try readHeaderSections(io: io, meta: meta, startOffset: header.nextOffset)
        var offset = decoded.kvOffset

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

        return (meta, decoded.state, meta.nextToken, decoded.logits, trunkShared, mtpShared)
    }
}
