// QWKVCache+TestHelpers.swift — Test-only KV prefix and inspection helpers for QWKVTests.
import Foundation
@testable import QWKV

extension QWKVCache {
    var totalPages: Int { poolsFlat.reduce(0) { $0 + $1.pages.pageCount } }

    func pageIDs(_ h: QWKVHandle, layer: Int, kind: QWKVKind) -> [Int32] {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq], let li = layerIndex(layer) else { return [] }
        return kind == .k ? st.tablesK[li] : st.tablesV[li]
    }

    func adoptSharedPrefix(_ prefix: QWKVSharedPrefix, into h: QWKVHandle) throws {
        lock.lock(); defer { lock.unlock() }
        try validateSharedPrefixLocked(prefix)
        guard h.cache === self, !h.finished, let st = seqs[h.seq],
              st.writtenTokens == 0, prefix.tokenCount <= h.tokenCapacity else {
            throw QWKVError.badArgument("shared prefix destination identity, state or capacity invalid")
        }
        let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        for li in layers.indices {
            guard st.tablesK[li].count >= required, st.tablesV[li].count >= required else {
                throw QWKVError.badArgument("shared prefix destination page table is short")
            }
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let ids = prefix.pages[li * 2 + ki].prefix(required)
                let limit = pool(layers[li], kind).pages.pageCount
                guard ids.allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                    throw QWKVError.badArgument("shared prefix page ID is invalid")
                }
            }
        }
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                let newIDs = Array(prefix.pages[li * 2 + ki].prefix(required))
                let oldIDs = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let oldShared = kind == .k ? st.sharedK[li] : st.sharedV[li]
                for page in newIDs { retainSharedPage(poolIndex(li, kind), page) }
                if kind == .k {
                    st.tablesK[li].replaceSubrange(0..<required, with: newIDs)
                    st.sharedK[li].replaceSubrange(0..<required, with: repeatElement(true, count: required))
                } else {
                    st.tablesV[li].replaceSubrange(0..<required, with: newIDs)
                    st.sharedV[li].replaceSubrange(0..<required, with: repeatElement(true, count: required))
                }
                for index in 0..<required {
                    if oldShared[index] { releaseSharedPage(poolIndex(li, kind), oldIDs[index]) }
                    else if !isPagePinned(poolIndex(li, kind), oldIDs[index]) {
                        pool(layer, kind).pages.freePage(Int(oldIDs[index]))
                    }
                }
                uploadSlot(h.seq, li, kind, kind == .k ? st.tablesK[li] : st.tablesV[li])
            }
        }
        st.writtenTokens = prefix.tokenCount
    }

    func exportSharedPrefixPlanes(_ prefix: QWKVSharedPrefix) -> [Data] {
        lock.lock(); defer { lock.unlock() }
        let planeBytes = serializedPlaneBytes(tokenCount: prefix.tokenCount)
        var planes: [Data] = []
        planes.reserveCapacity(layers.count * 2)
        for (li, layer) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                var data = Data(count: planeBytes)
                let ids = prefix.pages[li * 2 + ki]
                let region = pool(layer, kind).pages.region
                data.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
                    for page in 0..<ids.count {
                        let bytes = mode == .f16
                            ? min(Self.pageTokens, prefix.tokenCount - page * Self.pageTokens) * f16RowBytes
                            : bytesPerPage
                        destination.baseAddress!.advanced(by: page * bytesPerPage)
                            .copyMemory(from: region.pointer.advanced(by: Int(ids[page]) * bytesPerPage),
                                        byteCount: bytes)
                    }
                }
                planes.append(data)
            }
        }
        return planes
    }

    func makePrivatePrefix(tokenCount: Int, planes: [Data]) -> QWKVPrivatePrefix {
        let total = planes.reduce(0) { $0 + $1.count }
        return QWKVPrivatePrefix(
            tokenCount: tokenCount, memoryBytes: total, mode: mode, cache: self, planes: planes
        )
    }
}
