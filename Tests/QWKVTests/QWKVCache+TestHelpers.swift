// QWKVCache+TestHelpers.swift — Test-only KV prefix and inspection helpers for QWKVTests.
import Foundation
@testable import QWKV

extension QWKVCache {
    var totalPages: Int { poolPairs.reduce(0) { $0 + $1.totalPages } }

    func pageIDs(_ h: QWKVHandle, layer: Int, kind: QWKVKind) -> [Int32] {
        lock.withLock {
            guard let st = seqs[h.seq], let li = layerIndex(layer) else { return [] }
            return st.layerPairs[li][kind].ids
        }
    }

    func adoptSharedPrefix(_ prefix: QWKVSharedPrefix, into h: QWKVHandle) throws {
        try lock.withLock {
            try validateSharedPrefixLocked(prefix)
            guard h.cache === self, !h.finished, let st = seqs[h.seq],
                  st.writtenTokens == 0, prefix.tokenCount <= h.tokenCapacity else {
                throw QWKVError.badArgument("shared prefix destination identity, state or capacity invalid")
            }
            let required = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
            for li in layers.indices {
                guard st.layerPairs[li].k.pageCount >= required,
                      st.layerPairs[li].v.pageCount >= required else {
                    throw QWKVError.badArgument("shared prefix destination page table is short")
                }
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let ids = prefix.pages[li * 2 + ki].prefix(required)
                    let limit = pool(at: li, kind).pages.pageCount
                    guard ids.prefix(startPage).allSatisfy({ $0 < 0 }),
                          startPage >= required || ids[startPage..<required].allSatisfy({ $0 >= 0 && Int($0) < limit }) else {
                        throw QWKVError.badArgument("shared prefix page ID is invalid")
                    }
                }
            }
            for li in layers.indices {
                let startPage = min(required, liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let newIDs = (0..<required).map { idx in idx < startPage ? Int32(-1) : prefix.pages[li * 2 + ki][idx] }
                    let oldIDs = st.layerPairs[li][kind].ids
                    let oldShared = st.layerPairs[li][kind].shared
                    for page in newIDs where page >= 0 { retainSharedPage(li: li, kind: kind, page: page) }
                    st.layerPairs[li][kind].ids.replaceSubrange(0..<required, with: newIDs)
                    st.layerPairs[li][kind].shared.replaceSubrange(
                        0..<required, with: (0..<required).map { $0 >= startPage })
                    for index in 0..<required where oldIDs[index] >= 0 {
                        if oldShared[index] {
                            releaseSharedPage(li: li, kind: kind, page: oldIDs[index])
                        } else if !isPagePinned(li: li, kind: kind, page: oldIDs[index]) {
                            pool(at: li, kind).pages.freePage(Int(oldIDs[index]))
                        }
                    }
                    uploadSlot(h.seq, li, kind, st.layerPairs[li][kind].ids)
                }
            }
            st.writtenTokens = prefix.tokenCount
        }
    }

    func exportSharedPrefixPlanes(_ prefix: QWKVSharedPrefix) -> [Data] {
        lock.withLock {
            var planes: [Data] = []
            planes.reserveCapacity(layers.count * 2)
            for li in layers.indices {
                let planeBytes = serializedPlaneBytes(layerIndex: li, tokenCount: prefix.tokenCount)
                let startPage = min(
                    (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens,
                    liveStartPage(layerIndex: li, tokenCount: prefix.tokenCount))
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    var data = Data(count: planeBytes)
                    let ids = prefix.pages[li * 2 + ki]
                    let region = pool(at: li, kind).pages.region
                    data.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
                        guard let dstBase = destination.baseAddress else { return }
                        for page in startPage..<ids.count {
                            let bytes = mode == .f16
                                ? min(Self.pageTokens, prefix.tokenCount - page * Self.pageTokens) * f16RowBytes
                                : bytesPerPage
                            dstBase.advanced(by: (page - startPage) * bytesPerPage)
                                .copyMemory(from: region.pointer.advanced(by: Int(ids[page]) * bytesPerPage),
                                            byteCount: bytes)
                        }
                    }
                    planes.append(data)
                }
            }
            return planes
        }
    }

    func makePrivatePrefix(tokenCount: Int, planes: [Data]) -> QWKVPrivatePrefix {
        let total = planes.reduce(0) { $0 + $1.count }
        return QWKVPrivatePrefix(
            tokenCount: tokenCount, memoryBytes: total, mode: mode, cache: self, planes: planes
        )
    }
}
