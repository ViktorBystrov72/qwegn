// QWKVCache.swift — Paged KV cache for full-attention layers.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

/// The cache. One instance per engine; `layers` = attn layer indices.
public final class QWKVCache: @unchecked Sendable {
    public static let pageTokens = 256
    public var f16RowBytes: Int { geometry.kvChannels * 2 }

    public static func pageBytes(mode: QWKVMode, geometry: QWModelGeometry) -> Int {
        QWKVCompressedLayout(mode: mode, geometry: geometry).totalBytesPerPage
    }

    /// One KV buffer (K+V, all layers) for `tokens` at raw payload density.
    public static func kvBytes(tokens: Int, layers: Int, mode: QWKVMode, geometry: QWModelGeometry) -> Int {
        let elements = 2 * layers * tokens * geometry.kvChannels
        switch mode {
        case .f16: return elements * 2
        case .q8: return elements * 1
        case .q4: return elements / 2
        }
    }

    public let geometry: QWModelGeometry
    let arena: QWArena?
    let layers: [Int]
    public let mode: QWKVMode
    public let bytesPerPage: Int

    let poolPairs: [QWKVLayerPoolPair]
    let layerToLi: [Int: Int]

    final class SeqState {
        var layerPairs: [QWKVLayerPair]
        var writtenTokens: Int = 0
        var rollbackGeneration: UInt64 = 0

        init(layerCount: Int) {
            layerPairs = Array(repeating: QWKVLayerPair(), count: layerCount)
        }
    }

    var seqs: [QWKVSeqID: SeqState] = [:]
    var nextSeq: QWKVSeqID = 1
    var freeSeqs: [QWKVSeqID] = []
    let lock = NSLock()
    let cacheEpoch = UUID()
    private var _onPagePressure: (@Sendable () -> Bool)?

    /// Invoked outside `lock` on `QWKVError.outOfPages`; returns `true` if pages may have freed.
    public var onPagePressure: (@Sendable () -> Bool)? {
        get { lock.withLock { _onPagePressure } }
        set { lock.withLock { _onPagePressure = newValue } }
    }

    struct SharedPage: Hashable {
        let li: Int
        let kind: QWKVKind
        let page: Int32
    }

    var sharedPageRefs: [SharedPage: Int] = [:]

    /// Device page-table slots: 16 KiB per (seq, layer, kind) = 1M token ceiling.
    public static let tableSlotBytes = 16384
    public static let maxSeqs = 64
    public static let maxTokensPerSeq = (tableSlotBytes / 4) * pageTokens
    let tableBuffer: any MTLBuffer
    let tableBase: Int

    public private(set) var steadyStateAppends: Int = 0
    var steady = false

    public convenience init(hal: QWHal, arena: QWArena, layers: [Int], mode: QWKVMode,
                            geometry: QWModelGeometry) throws {
        try self.init(hal: hal, subAllocator: arena.kvRegion(), vSubAllocator: arena.kvVSub,
                      layers: layers, mode: mode, arena: arena, geometry: geometry)
    }

    public init(hal: QWHal, subAllocator: QWSubAllocator, vSubAllocator: QWSubAllocator? = nil,
                auxSubAllocator: QWSubAllocator? = nil,
                layers: [Int], mode: QWKVMode,
                arena: QWArena? = nil, geometry: QWModelGeometry) throws {
        guard !layers.isEmpty else { throw QWKVError.badArgument("no layers") }
        self.geometry = geometry
        self.arena = arena
        self.layers = layers
        self.mode = mode
        let pageBytes = Self.pageBytes(mode: mode, geometry: geometry)
        self.bytesPerPage = pageBytes
        let kvAux = auxSubAllocator ?? ((subAllocator === arena?.kvSub) ? arena?.kvAuxSub : nil)
        let tablesTotal = Self.tableSlotBytes * 2 * layers.count * Self.maxSeqs
        let built = try QWKVHostCodec.buildPoolsAndTables(
            hal: hal, kv: subAllocator, vSubAllocator: vSubAllocator, kvAux: kvAux,
            layers: layers, pageBytes: pageBytes, tablesTotal: tablesTotal)
        self.poolPairs = built.poolPairs
        var lmap: [Int: Int] = [:]
        for (li, layer) in layers.enumerated() { lmap[layer] = li }
        self.layerToLi = lmap
        self.tableBuffer = built.tableBuffer
        self.tableBase = built.tableBase
    }

    func layerIndex(_ l: Int) -> Int? { layerToLi[l] }
    func pool(at li: Int, _ kind: QWKVKind) -> QWKVLayerPoolPair.Pool { poolPairs[li][kind] }
    func slotBase(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind) -> Int {
        tableBase + ((Int(seq) * layers.count + li) * 2 + (kind == .k ? 0 : 1)) * Self.tableSlotBytes
    }

    private func withPagePressure<T>(_ body: () throws -> T) throws -> T {
        let maxRetries = max(1024, poolPairs.reduce(0) { $0 + $1.totalPages })
        var attempts = 0
        while true {
            do {
                return try lock.withLock { try body() }
            } catch let error as QWKVError {
                guard case .outOfPages = error, attempts < maxRetries else { throw error }
                let pressure = lock.withLock { _onPagePressure }
                guard let pressure, pressure() else { throw error }
                attempts += 1
            }
        }
    }

    public func allocate(_ seq: QWKVSeqID? = nil, promptTokens: Int) throws -> QWKVHandle {
        try withPagePressure { try allocateLocked(seq, promptTokens: promptTokens) }
    }

    private func allocateLocked(_ seq: QWKVSeqID? = nil, promptTokens: Int) throws -> QWKVHandle {
        let fromFreeList = (seq == nil && !freeSeqs.isEmpty)
        let id = seq ?? (freeSeqs.first ?? nextSeq)
        guard Int(id) < Self.maxSeqs else { throw QWKVError.badArgument("seq id ≥ maxSeqs") }
        guard seqs[id] == nil else { throw QWKVError.badArgument("seq \(id) exists") }
        let pagesNeeded = (max(promptTokens, 1) + Self.pageTokens - 1) / Self.pageTokens
        guard pagesNeeded * 4 <= Self.tableSlotBytes else {
            throw QWKVError.badArgument("prompt \(promptTokens) > \(Self.maxTokensPerSeq) ceiling")
        }
        let st = SeqState(layerCount: layers.count)
        var taken: [(Int, QWKVKind, Int32)] = []
        do {
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    var ids: [Int32] = []
                    ids.reserveCapacity(pagesNeeded)
                    for _ in 0..<pagesNeeded {
                        guard let p = pool(at: li, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: pagesNeeded, free: pool(at: li, kind).pages.freePages)
                        }
                        ids.append(Int32(p))
                        taken.append((li, kind, Int32(p)))
                    }
                    st.layerPairs[li][kind].ids = ids
                    st.layerPairs[li][kind].shared = [Bool](repeating: false, count: pagesNeeded)
                }
            }
        } catch {
            for (li, kind, p) in taken { pool(at: li, kind).pages.freePage(Int(p)) }
            throw error
        }
        for li in layers.indices {
            for kind in QWKVKind.allCases {
                uploadSlot(id, li, kind, st.layerPairs[li][kind].ids)
            }
        }
        if fromFreeList { freeSeqs.removeFirst() } else { freeSeqs.removeAll { $0 == id } }
        nextSeq = max(nextSeq, id &+ 1)
        seqs[id] = st
        return QWKVHandle(seq: id, capacity: pagesNeeded * Self.pageTokens, cache: self)
    }

    public func allocate(adopting prefix: QWKVSharedPrefix, promptTokens: Int) throws -> QWKVHandle {
        try withPagePressure { try allocateLocked(adopting: prefix, promptTokens: promptTokens) }
    }

    private func allocateLocked(adopting prefix: QWKVSharedPrefix, promptTokens: Int) throws -> QWKVHandle {
        try validateSharedPrefixLocked(prefix)
        guard promptTokens >= prefix.tokenCount, promptTokens <= Self.maxTokensPerSeq else {
            throw QWKVError.badArgument("shared prefix prompt extent invalid")
        }
        let prefixPages = (prefix.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        let pagesNeeded = max(1, (promptTokens + Self.pageTokens - 1) / Self.pageTokens)
        guard pagesNeeded * 4 <= Self.tableSlotBytes, pagesNeeded >= prefixPages else {
            throw QWKVError.badArgument("shared prefix prompt exceeds page-table ceiling")
        }
        let fromFreeList = !freeSeqs.isEmpty
        let id = freeSeqs.first ?? nextSeq
        guard Int(id) < Self.maxSeqs else { throw QWKVError.badArgument("seq id ≥ maxSeqs") }
        guard seqs[id] == nil else { throw QWKVError.badArgument("seq \(id) exists") }
        let st = SeqState(layerCount: layers.count)
        var taken: [(Int, QWKVKind, Int32)] = []
        do {
            for li in layers.indices {
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let cached = prefix.pages[li * 2 + ki]
                    var ids = Array(cached.prefix(prefixPages))
                    var shared = [Bool](repeating: true, count: prefixPages)
                    for _ in prefixPages..<pagesNeeded {
                        guard let page = pool(at: li, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: pagesNeeded - prefixPages,
                                                       free: pool(at: li, kind).pages.freePages)
                        }
                        let pageID = Int32(page)
                        ids.append(pageID)
                        shared.append(false)
                        taken.append((li, kind, pageID))
                    }
                    st.layerPairs[li][kind].ids = ids
                    st.layerPairs[li][kind].shared = shared
                }
            }
        } catch {
            for (li, kind, page) in taken { pool(at: li, kind).pages.freePage(Int(page)) }
            throw error
        }
        for li in layers.indices {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                for page in prefix.pages[li * 2 + ki].prefix(prefixPages) {
                    retainSharedPage(li: li, kind: kind, page: page)
                }
                uploadSlot(id, li, kind, st.layerPairs[li][kind].ids)
            }
        }
        if fromFreeList { freeSeqs.removeFirst() }
        nextSeq = max(nextSeq, id &+ 1)
        st.writtenTokens = prefix.tokenCount
        seqs[id] = st
        return QWKVHandle(seq: id, capacity: pagesNeeded * Self.pageTokens, cache: self)
    }

    public func grow(_ h: QWKVHandle, to tokens: Int) throws {
        try withPagePressure { try growLocked(h, to: tokens) }
    }

    private func growLocked(_ h: QWKVHandle, to tokens: Int) throws {
        guard let st = seqs[h.seq] else { throw QWKVError.badArgument("unknown seq") }
        let havePages = st.layerPairs.first?.k.pageCount ?? 0
        let needPages = (tokens + Self.pageTokens - 1) / Self.pageTokens
        guard needPages > havePages else { return }
        guard needPages * 4 <= Self.tableSlotBytes else {
            throw QWKVError.badArgument("tokens \(tokens) > \(Self.maxTokensPerSeq) ceiling")
        }
        let add = needPages - havePages
        var grabbed: [(Int, QWKVKind, [Int32])] = []
        do {
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    var ids: [Int32] = []
                    ids.reserveCapacity(add)
                    for _ in 0..<add {
                        guard let p = pool(at: li, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: add, free: pool(at: li, kind).pages.freePages)
                        }
                        ids.append(Int32(p))
                    }
                    grabbed.append((li, kind, ids))
                }
            }
        } catch {
            for (li, kind, ids) in grabbed { for p in ids { pool(at: li, kind).pages.freePage(Int(p)) } }
            throw error
        }
        for (li, kind, ids) in grabbed {
            st.layerPairs[li][kind].ids.append(contentsOf: ids)
            st.layerPairs[li][kind].shared.append(contentsOf: repeatElement(false, count: ids.count))
            uploadSlot(h.seq, li, kind, st.layerPairs[li][kind].ids)
        }
        h.tokenCapacity = (st.layerPairs.first?.k.pageCount ?? 0) * Self.pageTokens
    }

    public func append(_ h: QWKVHandle, positions: Range<Int>,
                       layer: Int, kind: QWKVKind, src: [UInt16]) throws {
        try withPagePressure { try appendLocked(h, positions: positions, layer: layer, kind: kind, src: src) }
    }

    private func appendLocked(_ h: QWKVHandle, positions: Range<Int>,
                              layer: Int, kind: QWKVKind, src: [UInt16]) throws {
        guard h.cache === self, !h.finished, let st = seqs[h.seq], let li = layerIndex(layer) else {
            throw QWKVError.badArgument("unknown seq/layer")
        }
        let rowElems = geometry.kvChannels
        guard positions.lowerBound >= 0, positions.upperBound >= positions.lowerBound,
              positions.count * rowElems == src.count else {
            throw QWKVError.badArgument("src/token mismatch")
        }
        let cap = st.layerPairs[li][kind].pageCount * Self.pageTokens
        guard positions.upperBound <= cap else { throw QWKVError.badArgument("grow first (cap \(cap))") }
        try prepareWritesLocked(st, handle: h, positions: positions, only: (li, kind))
        guard !positions.isEmpty else { return }
        if steady { steadyStateAppends += 1 }
        try QWKVHostCodec.writeRows(
            mode: mode, geometry: geometry, bytesPerPage: bytesPerPage, rowElems: rowElems,
            positions: positions, table: st.layerPairs[li][kind].ids,
            pageArena: pool(at: li, kind).pages, src: src)
        st.writtenTokens = max(st.writtenTokens, positions.upperBound)
    }

    public func prepareWrites(_ h: QWKVHandle, positions: Range<Int>) throws {
        try withPagePressure { try prepareWritesEntryLocked(h, positions: positions) }
    }

    private func prepareWritesEntryLocked(_ h: QWKVHandle, positions: Range<Int>) throws {
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("unknown, foreign, or finished seq")
        }
        try prepareWritesLocked(st, handle: h, positions: positions, only: nil)
    }

    public func hasSharedPages(_ h: QWKVHandle, positions: Range<Int>) -> Bool {
        lock.withLock {
            guard !sharedPageRefs.isEmpty, h.cache === self, !h.finished,
                  let st = seqs[h.seq], !positions.isEmpty else { return false }
            let first = positions.lowerBound / Self.pageTokens
            let last = (positions.upperBound - 1) / Self.pageTokens
            for pair in st.layerPairs {
                for kind in QWKVKind.allCases {
                    let shared = pair[kind].shared
                    guard last < shared.count else { continue }
                    for index in first...last where shared[index] { return true }
                }
            }
            return false
        }
    }

    public func layerViews(_ h: QWKVHandle, layer: Int) throws -> (k: QWKVLayerView, v: QWKVLayerView) {
        try lock.withLock {
            guard let st = seqs[h.seq], let li = layerIndex(layer) else {
                throw QWKVError.badArgument("unknown seq/layer")
            }
            return makeLayerViews(seq: h.seq, li: li, pair: st.layerPairs[li])
        }
    }

    public func layerViewsAll(_ h: QWKVHandle, layers: [Int]) throws -> [(k: QWKVLayerView, v: QWKVLayerView)] {
        try lock.withLock {
            guard let st = seqs[h.seq] else { throw QWKVError.badArgument("unknown seq") }
            return try layers.map { layer in
                guard let li = layerIndex(layer) else { throw QWKVError.badArgument("unknown layer") }
                return makeLayerViews(seq: h.seq, li: li, pair: st.layerPairs[li])
            }
        }
    }

    private func makeLayerViews(seq: QWKVSeqID, li: Int, pair: QWKVLayerPair) -> (k: QWKVLayerView, v: QWKVLayerView) {
        let kView = QWKVLayerView(
            pool: pool(at: li, .k).pages.region,
            table: tableTensor(seq, li, .k, pair.k.pageCount),
            pageBytes: bytesPerPage, mode: mode)
        let vView = QWKVLayerView(
            pool: pool(at: li, .v).pages.region,
            table: tableTensor(seq, li, .v, pair.v.pageCount),
            pageBytes: bytesPerPage, mode: mode)
        return (kView, vView)
    }

    public func snapshot(_ h: QWKVHandle) throws -> QWKVSnap {
        try lock.withLock {
            guard let st = seqs[h.seq] else { throw QWKVError.badArgument("unknown seq") }
            var pins: [(Int, QWKVKind, [Int32])] = []
            for (li, l) in layers.enumerated() {
                for kind in QWKVKind.allCases {
                    let ids = st.layerPairs[li][kind].ids
                    for p in ids { st.layerPairs[li][kind].pins[p, default: 0] += 1 }
                    pins.append((l, kind, ids))
                }
            }
            return QWKVSnap(seq: h.seq, tokenCount: st.writtenTokens, cache: self, pins: pins)
        }
    }

    func prepareWritesLocked(_ st: SeqState, handle: QWKVHandle,
                             positions: Range<Int>, only: (Int, QWKVKind)?) throws {
        guard positions.lowerBound >= 0, positions.upperBound >= positions.lowerBound else {
            throw QWKVError.badArgument("write range invalid")
        }
        guard !positions.isEmpty else { return }
        let firstPage = positions.lowerBound / Self.pageTokens
        let lastPage = (positions.upperBound - 1) / Self.pageTokens
        var copies: [(li: Int, kind: QWKVKind, index: Int, old: Int32, new: Int32)] = []
        var grabbed: [(Int, QWKVKind, Int32)] = []
        do {
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    if let only, (only.0 != li || only.1 != kind) { continue }
                    let stream = st.layerPairs[li][kind]
                    guard positions.upperBound <= stream.pageCount * Self.pageTokens,
                          stream.shared.count == stream.pageCount, lastPage < stream.pageCount else {
                        throw QWKVError.badArgument("write range exceeds admitted page capacity")
                    }
                    for index in firstPage...lastPage where stream.shared[index] {
                        guard let page = pool(at: li, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: 1, free: pool(at: li, kind).pages.freePages)
                        }
                        let replacement = Int32(page)
                        grabbed.append((li, kind, replacement))
                        copies.append((li, kind, index, stream.ids[index], replacement))
                    }
                }
            }
        } catch {
            for (li, kind, page) in grabbed { pool(at: li, kind).pages.freePage(Int(page)) }
            throw error
        }
        for copy in copies {
            let pa = pool(at: copy.li, copy.kind).pages
            pa.region.pointer.advanced(by: Int(copy.new) * bytesPerPage)
                .copyMemory(from: pa.region.pointer.advanced(by: Int(copy.old) * bytesPerPage),
                            byteCount: bytesPerPage)
        }
        for copy in copies {
            st.layerPairs[copy.li][copy.kind].ids[copy.index] = copy.new
            st.layerPairs[copy.li][copy.kind].shared[copy.index] = false
            uploadSlot(handle.seq, copy.li, copy.kind, st.layerPairs[copy.li][copy.kind].ids)
            releaseSharedPage(li: copy.li, kind: copy.kind, page: copy.old)
        }
    }

    func tableTensor(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind, _ count: Int) -> QWTensorValue {
        let base = slotBase(seq, li, kind)
        return QWTensorValue(region: QWMRegion(buffer: tableBuffer, base: base, length: count * 4),
                             offset: 0, shape: QWTensorShape([max(count, 1)], .i32))
    }

    /// Rewrite a (seq, layer, kind) device slot in place and tail-zero unused entries.
    func uploadSlot(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind, _ ids: [Int32]) {
        let base = slotBase(seq, li, kind)
        let byteCount = ids.count * 4
        if byteCount > 0 {
            ids.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let addr = raw.baseAddress else { return }
                _ = memcpy(tableBuffer.contents().advanced(by: base), addr, byteCount)
            }
        }
        if byteCount < Self.tableSlotBytes {
            memset(tableBuffer.contents().advanced(by: base + byteCount), 0,
                   Self.tableSlotBytes - byteCount)
        }
    }
}
