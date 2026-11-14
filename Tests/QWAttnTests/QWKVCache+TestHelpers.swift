// QWKVCache+TestHelpers.swift — Test-only page-table inspection helper for QWAttnTests.
@testable import QWKV

extension QWKVCache {
    func pageIDs(_ h: QWKVHandle, layer: Int, kind: QWKVKind) -> [Int32] {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq], let li = layerIndex(layer) else { return [] }
        return kind == .k ? st.tablesK[li] : st.tablesV[li]
    }
}
