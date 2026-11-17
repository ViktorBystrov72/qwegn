// QWKVCache.swift — C10: paged KV cache for the 16 full-attention layers.
//
// Layout law (spec 10): page = 256 tokens × [4 kv-heads × 256 head_dim] f16
// = 0.5 MiB per layer-page (Q8 halves it). K and V in SEPARATE pools per layer
// so each attention stream reads contiguous pages. Page tables are int32
// page ids that are POOL-LOCAL — a page id only means something WITH its
// (layer, kind) — so each (seq, layer, kind) owns its own table. Tables are
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
        // Layout authority = QWKVCompressedLayout: compressed pages carry
        // their C04 scale header INSIDE the page (the kernel addresses it at
        // a fixed payload offset), so the cache page size is data+header.
        return QWKVCompressedLayout(mode: mode, geometry: geometry).totalBytesPerPage
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

    final class Pool { let pages: QWPageArena; init(_ p: QWPageArena) { pages = p } }
    // P8 FIX: pools were a [String: Pool] dictionary — layerViews() built
    // TWO fresh Strings ("\(l):\(kind)") per attention layer per step under
    // a lock (≈64 string allocs/step, ~80k lookups per 64-token prefill).
    let poolsFlat: [Pool]              // index = li * 2 + kindIndex
    let layerToLi: [Int: Int]          // absolute layer → layers index

    /// Per-sequence CPU state. tablesK[li]/tablesV[li] are PARALLEL to
    /// `layers`: per-layer page ids, pool-local, one per 256 tokens of
    /// written context (grown in page granularity; every layer tracks the
    final class SeqState {
        var tablesK: [[Int32]]
        var tablesV: [[Int32]]
        var sharedK: [[Bool]]
        var sharedV: [[Bool]]
        var writtenTokens: Int = 0
        var rollbackGeneration: UInt64 = 0
        var pins: [String: [Int32: Int]] = [:]    // "layer:kind" → page → count
        init(layers: [Int]) {
            tablesK = Array(repeating: [], count: layers.count)
            tablesV = Array(repeating: [], count: layers.count)
            sharedK = Array(repeating: [], count: layers.count)
            sharedV = Array(repeating: [], count: layers.count)
        }
    }
    var seqs: [QWKVSeqID: SeqState] = [:]
    var nextSeq: QWKVSeqID = 1
    /// Finished seq ids, reusable by allocate() (FIFO — QWState's free-list
    /// law). Without recycling, a 10k-session soak would exhaust the 64
    /// table slots the arena carves (slotBase is seq-indexed) — finish()
    var freeSeqs: [QWKVSeqID] = []
    let lock = NSLock()
    let cacheEpoch = UUID()
    private var _onPagePressure: (@Sendable () -> Bool)?
    /// Invoked outside `lock` on `QWKVError.outOfPages`; returns `true` if pages may have freed.
    public var onPagePressure: (@Sendable () -> Bool)? {
        get { lock.withLock { _onPagePressure } }
        set { lock.withLock { _onPagePressure = newValue } }
    }
    /// Counts retained immutable-prefix objects and live table entries per
    /// physical page; snapshots continue to use the legacy per-sequence pins.
    struct SharedPage: Hashable { let pool: Int; let page: Int32 }
    var sharedPageRefs: [SharedPage: Int] = [:]

    /// Device page-table slots: per (seq, layer, kind) a 16 KiB int32 slot
    /// (4096 ids = the 1,048,576-token ceiling). Grow rewrites its slot in place.
    public static let tableSlotBytes = 16384
    public static let maxSeqs = 64
    public static let maxTokensPerSeq = (tableSlotBytes / 4) * pageTokens   // 1048576 (1M context)
    let tableBuffer: any MTLBuffer
    let tableBase: Int

    /// Steady-state counters (bench_kv: append host-allocs == 0 when steady).
    public private(set) var steadyStateAppends: Int = 0
    var steady = false

    public convenience init(hal: QWHal, arena: QWArena, layers: [Int], mode: QWKVMode,
                            geometry: QWModelGeometry) throws {
        try self.init(hal: hal, subAllocator: arena.kvRegion(), vSubAllocator: arena.kvVSub,
                      layers: layers, mode: mode, arena: arena, geometry: geometry)
    }

    public init(hal: QWHal, subAllocator: QWSubAllocator, vSubAllocator: QWSubAllocator? = nil,
                layers: [Int], mode: QWKVMode,
                arena: QWArena? = nil, geometry: QWModelGeometry) throws {
        guard !layers.isEmpty else { throw QWKVError.badArgument("no layers") }
        self.geometry = geometry
        self.arena = arena
        self.layers = layers
        self.mode = mode
        self.bytesPerPage = Self.pageBytes(mode: mode, geometry: geometry)
        let kv = subAllocator
        let tablesTotal = Self.tableSlotBytes * 2 * layers.count * Self.maxSeqs
        let auxLen = (vSubAllocator != nil && subAllocator === arena?.kvSub) ? (arena?.kvAuxSub?.region.length ?? 0) : 0
        let avail = kv.region.length + (vSubAllocator?.region.length ?? 0) + auxLen
        guard tablesTotal < avail else { throw QWKVError.outOfPages(asked: tablesTotal, free: avail) }
        let poolBytes = (avail - tablesTotal) / (2 * layers.count)
        guard poolBytes >= bytesPerPage else {
            throw QWKVError.outOfPages(asked: bytesPerPage, free: poolBytes)
        }
        let pagesPerPool = poolBytes / bytesPerPage
        let onePoolBytes = pagesPerPool * bytesPerPage
        var flat: [Pool] = []
        flat.reserveCapacity(layers.count * 2)
        var lmap: [Int: Int] = [:]
        for (li, l) in layers.enumerated() {
            lmap[l] = li
            for kind in QWKVKind.allCases {        // order: .k then .v (kindIndex 0/1)
                let targetSub: QWSubAllocator
                if let kvV = vSubAllocator, kind == .v {
                    targetSub = kvV
                } else {
                    targetSub = kv
                }
                let off = try targetSub.allocate(bytes: onePoolBytes)
                let region = QWMRegion(buffer: targetSub.region.buffer,
                                       base: targetSub.region.base + off, length: onePoolBytes)
                flat.append(Pool(QWPageArena(region: region, pageSize: bytesPerPage,
                                             pageCount: pagesPerPool)))
            }
        }
        self.poolsFlat = flat
        self.layerToLi = lmap
        // table slots tail: zero-initialised (an unwritten entry must never
        // read as a garbage page id).
        if kv.freeBytes >= tablesTotal {
            let tabOff = try kv.allocate(bytes: tablesTotal)
            memset(kv.region.pointer.advanced(by: tabOff), 0, tablesTotal)
            self.tableBuffer = kv.region.buffer
            self.tableBase = kv.region.base + tabOff
        } else if let kvV = vSubAllocator, kvV.freeBytes >= tablesTotal {
            let tabOff = try kvV.allocate(bytes: tablesTotal)
            memset(kvV.region.pointer.advanced(by: tabOff), 0, tablesTotal)
            self.tableBuffer = kvV.region.buffer
            self.tableBase = kvV.region.base + tabOff
        } else if let aux = arena?.kvAuxSub, aux.freeBytes >= tablesTotal {
            let tabOff = try aux.allocate(bytes: tablesTotal)
            memset(aux.region.pointer.advanced(by: tabOff), 0, tablesTotal)
            self.tableBuffer = aux.region.buffer
            self.tableBase = aux.region.base + tabOff
        } else {
            guard let tBuf = hal.device.makeBuffer(length: tablesTotal, options: .storageModeShared) else {
                throw QWKVError.outOfPages(asked: tablesTotal, free: 0)
            }
            memset(tBuf.contents(), 0, tablesTotal)
            self.tableBuffer = tBuf
            self.tableBase = 0
        }
    }

    func layerIndex(_ l: Int) -> Int? { layerToLi[l] }   // O(1), zero allocs
    func pool(_ layer: Int, _ kind: QWKVKind) -> Pool {
        poolsFlat[layerToLi[layer]! * 2 + (kind == .k ? 0 : 1)]
    }
    func slotBase(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind) -> Int {
        tableBase + ((Int(seq) * layers.count + li) * 2 + (kind == .k ? 0 : 1)) * Self.tableSlotBytes
    }

    /// Runs `body` under `lock`; on `.outOfPages`, drops `lock` before invoking `onPagePressure`.
    private func withPagePressure<T>(_ body: () throws -> T) throws -> T {
        let maxRetries = max(1024, poolsFlat.reduce(0) { $0 + $1.pages.pageCount })
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

    /// Allocate a sequence's cache for `promptTokens` (prefill grows up-front;
    /// decode grows one page per 256 tokens via grow()). All-or-nothing: on a
    /// pool shortfall every page taken in this call returns to its own pool.
    public func allocate(_ seq: QWKVSeqID? = nil, promptTokens: Int) throws -> QWKVHandle {
        try withPagePressure {
            try allocateLocked(seq, promptTokens: promptTokens)
        }
    }

    private func allocateLocked(_ seq: QWKVSeqID? = nil, promptTokens: Int) throws -> QWKVHandle {
        let id = seq ?? (freeSeqs.isEmpty ? nextSeq : freeSeqs.removeFirst())
        // H2 FIX: the id is RETURNED to freeSeqs on every early exit below.
        // Before, a failed allocate (outOfPages, oversized prompt) ate the
        // id forever — after ~63 failures every allocate threw "seq id ≥
        func recycle() { if id < nextSeq && !freeSeqs.contains(id) { freeSeqs.append(id) } }
        nextSeq = max(nextSeq, id &+ 1)
        guard seqs[id] == nil else { recycle(); throw QWKVError.badArgument("seq \(id) exists") }
        guard id < Self.maxSeqs else { recycle(); throw QWKVError.badArgument("seq id ≥ maxSeqs") }
        let pagesNeeded = (max(promptTokens, 1) + Self.pageTokens - 1) / Self.pageTokens
        guard pagesNeeded * 4 <= Self.tableSlotBytes else {
            recycle()
            throw QWKVError.badArgument("prompt \(promptTokens) > \(Self.maxTokensPerSeq) ceiling")
        }
        let st = SeqState(layers: layers)
        var taken: [(Int, QWKVKind, Int32)] = []     // every page taken in this call (unwind list)
        do {
            for (li, l) in layers.enumerated() {
                for kind in QWKVKind.allCases {
                    var ids: [Int32] = []
                    ids.reserveCapacity(pagesNeeded)
                    for _ in 0..<pagesNeeded {
                        guard let p = pool(l, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: pagesNeeded,
                                                       free: pool(l, kind).pages.freePages)
                        }
                        ids.append(Int32(p))
                        taken.append((li, kind, Int32(p)))
                    }
                    if kind == .k {
                        st.tablesK[li] = ids
                        st.sharedK[li] = [Bool](repeating: false, count: pagesNeeded)
                    } else {
                        st.tablesV[li] = ids
                        st.sharedV[li] = [Bool](repeating: false, count: pagesNeeded)
                    }
                    uploadSlot(id, li, kind, ids)
                }
            }
        } catch {
            for (li, kind, p) in taken { pool(layers[li], kind).pages.freePage(Int(p)) }
            recycle()
            throw error
        }
        seqs[id] = st
        return QWKVHandle(seq: id, capacity: pagesNeeded * Self.pageTokens, cache: self)
    }

    /// Admit a sequence directly against a shared prefix. Only pages beyond
    /// the cached prefix are allocated, so admission does not first pay for a
    /// redundant private copy of the prefix.
    public func allocate(adopting prefix: QWKVSharedPrefix, promptTokens: Int) throws -> QWKVHandle {
        try withPagePressure {
            try allocateLocked(adopting: prefix, promptTokens: promptTokens)
        }
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

        let id = freeSeqs.isEmpty ? nextSeq : freeSeqs.removeFirst()
        func recycle() { if id < nextSeq && !freeSeqs.contains(id) { freeSeqs.append(id) } }
        nextSeq = max(nextSeq, id &+ 1)
        guard seqs[id] == nil else { recycle(); throw QWKVError.badArgument("seq \(id) exists") }
        guard id < Self.maxSeqs else { recycle(); throw QWKVError.badArgument("seq id ≥ maxSeqs") }

        let st = SeqState(layers: layers)
        var taken: [(Int, QWKVKind, Int32)] = []
        do {
            for (li, layer) in layers.enumerated() {
                for (ki, kind) in QWKVKind.allCases.enumerated() {
                    let cached = prefix.pages[li * 2 + ki]
                    var ids = Array(cached.prefix(prefixPages))
                    var shared = [Bool](repeating: true, count: prefixPages)
                    for _ in prefixPages..<pagesNeeded {
                        guard let page = pool(layer, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: pagesNeeded - prefixPages,
                                                       free: pool(layer, kind).pages.freePages)
                        }
                        let pageID = Int32(page)
                        ids.append(pageID)
                        shared.append(false)
                        taken.append((li, kind, pageID))
                    }
                    if kind == .k {
                        st.tablesK[li] = ids
                        st.sharedK[li] = shared
                    } else {
                        st.tablesV[li] = ids
                        st.sharedV[li] = shared
                    }
                }
            }
        } catch {
            for (li, kind, page) in taken { pool(layers[li], kind).pages.freePage(Int(page)) }
            recycle()
            throw error
        }

        for (li, _) in layers.enumerated() {
            for (ki, kind) in QWKVKind.allCases.enumerated() {
                for page in prefix.pages[li * 2 + ki].prefix(prefixPages) {
                    retainSharedPage(poolIndex(li, kind), page)
                }
                uploadSlot(id, li, kind, kind == .k ? st.tablesK[li] : st.tablesV[li])
            }
        }
        st.writtenTokens = prefix.tokenCount
        seqs[id] = st
        return QWKVHandle(seq: id, capacity: pagesNeeded * Self.pageTokens, cache: self)
    }

    /// Grow to ≥ `tokens` (page granularity, every layer+kind). Hot decode
    /// path (once per 256 tokens); zero host allocs when steady (the table
    /// arrays are capacity-reserved at allocate()). All-or-nothing per call.
    public func grow(_ h: QWKVHandle, to tokens: Int) throws {
        try withPagePressure {
            try growLocked(h, to: tokens)
        }
    }

    private func growLocked(_ h: QWKVHandle, to tokens: Int) throws {
        guard let st = seqs[h.seq] else { throw QWKVError.badArgument("unknown seq") }
        let havePages = st.tablesK.first?.count ?? 0
        let needPages = (tokens + Self.pageTokens - 1) / Self.pageTokens
        guard needPages > havePages else { return }
        guard needPages * 4 <= Self.tableSlotBytes else {
            throw QWKVError.badArgument("tokens \(tokens) > \(Self.maxTokensPerSeq) ceiling")
        }
        let add = needPages - havePages
        var grabbed: [(Int, QWKVKind, [Int32])] = []
        do {
            for (li, l) in layers.enumerated() {
                for kind in QWKVKind.allCases {
                    var ids: [Int32] = []
                    ids.reserveCapacity(add)
                    for _ in 0..<add {
                        guard let p = pool(l, kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: add, free: pool(l, kind).pages.freePages)
                        }
                        ids.append(Int32(p))
                    }
                    grabbed.append((li, kind, ids))
                }
            }
        } catch {
            for (li, kind, ids) in grabbed { for p in ids { pool(layers[li], kind).pages.freePage(Int(p)) } }
            throw error
        }
        for (li, kind, ids) in grabbed {
            if kind == .k {
                st.tablesK[li].append(contentsOf: ids)
                st.sharedK[li].append(contentsOf: repeatElement(false, count: ids.count))
            } else {
                st.tablesV[li].append(contentsOf: ids)
                st.sharedV[li].append(contentsOf: repeatElement(false, count: ids.count))
            }
            uploadSlot(h.seq, li, kind, kind == .k ? st.tablesK[li] : st.tablesV[li])
        }
        h.tokenCapacity = (st.tablesK.first?.count ?? 0) * Self.pageTokens
    }

    /// Write a token range's K/V rows into one (layer, kind) pool.
    /// `src` = [token][kvHeads×headDim] f16 over the range (row j = token
    /// positions.lowerBound + j). The hot decode path writes the SAME rows
    public func append(_ h: QWKVHandle, positions: Range<Int>,
                       layer: Int, kind: QWKVKind, src: [UInt16]) throws {
        try withPagePressure {
            try appendLocked(h, positions: positions, layer: layer, kind: kind, src: src)
        }
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
        let cap = st.tablesK[li].count * Self.pageTokens
        guard positions.upperBound <= cap else { throw QWKVError.badArgument("grow first (cap \(cap))") }
        try prepareWritesLocked(st, handle: h, positions: positions, only: (li, kind))
        guard !positions.isEmpty else { return }
        if steady { steadyStateAppends += 1 }
        let pa = pool(layer, kind).pages
        let table = kind == .k ? st.tablesK[li] : st.tablesV[li]
        let base0 = positions.lowerBound
        if mode == .f16 {
            src.withUnsafeBufferPointer { s in
                for tok in positions {
                    let pg = tok / Self.pageTokens, off = tok % Self.pageTokens
                    let pageID = Int(table[pg])
                    let dst = pa.region.buffer.contents()
                        .advanced(by: pa.region.base + pageID * bytesPerPage)
                        .assumingMemoryBound(to: UInt16.self) + off * rowElems
                    memcpy(dst, s.baseAddress! + (tok - base0) * rowElems, rowElems * 2)
                }
            }
        } else if mode == .q8 {
            // 4c: CPU quantiser for host-side fills (tests, prefix restore).
            // Byte-identical to the GPU quantisers (qwg_kv_paged_write /
            // qwattn_decode epilogue) — all three use QWKVCompressedLayout.
            let layout = QWKVCompressedLayout(mode: .q8, geometry: geometry)
            let effHeads = geometry.usesAnyMLA ? 1 : geometry.kvHeads
            let effHeadDim = geometry.usesAnyMLA ? geometry.kvChannels : geometry.headDim
            var frow = [Float](repeating: 0, count: rowElems)
            var qrow = [Int8](repeating: 0, count: rowElems)
            var srow = [Float16](repeating: 0, count: effHeads)
            try src.withUnsafeBufferPointer { s in
                for tok in positions {
                    let pg = tok / Self.pageTokens, off = tok % Self.pageTokens
                    let pageID = Int(table[pg])
                    let pageBase = pa.region.buffer.contents()
                        .advanced(by: pa.region.base + pageID * bytesPerPage)
                    let srcRow = s.baseAddress! + (tok - base0) * rowElems
                    for i in 0..<rowElems {
                        frow[i] = Float(Float16(bitPattern: srcRow[i]))
                    }
                    try QWKVCompressedLayout.quantizeQ8(
                        input: frow, qOut: &qrow, scalesOut: &srow,
                        kvHeads: effHeads, headDim: effHeadDim)
                    _ = qrow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in memcpy(pageBase.advanced(by: off * rowElems), raw.baseAddress!, rowElems) }
                    let scaleBase = pageBase.advanced(by: layout.dataBytesPerPage + off * (effHeads * 2))
                    _ = srow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in memcpy(scaleBase, raw.baseAddress!, effHeads * 2) }
                }
            }
        } else {
            // q4 host fill: byte-identical to GPU writers + quantizeQ4
            // (group-128 asymmetric; low nibble = even element).
            let layout = QWKVCompressedLayout(mode: .q4, geometry: geometry)
            let effHeads = geometry.usesAnyMLA ? 1 : geometry.kvHeads
            let effHeadDim = geometry.usesAnyMLA ? geometry.kvChannels : geometry.headDim
            let groupsPerHead = max(1, (effHeadDim + 127) / 128)
            let totalGroups = effHeads * groupsPerHead
            var frow = [Float](repeating: 0, count: rowElems)
            var qrow = [UInt8](repeating: 0, count: rowElems / 2)
            var sc = [Float16](repeating: 0, count: totalGroups)
            var mn = [Float16](repeating: 0, count: totalGroups)
            try src.withUnsafeBufferPointer { s in
                for tok in positions {
                    let pg = tok / Self.pageTokens, off = tok % Self.pageTokens
                    let pageID = Int(table[pg])
                    let pageBase = pa.region.buffer.contents()
                        .advanced(by: pa.region.base + pageID * bytesPerPage)
                    let srcRow = s.baseAddress! + (tok - base0) * rowElems
                    for i in 0..<rowElems {
                        frow[i] = Float(Float16(bitPattern: srcRow[i]))
                    }
                    try QWKVCompressedLayout.quantizeQ4(
                        input: frow, qOut: &qrow, scalesOut: &sc, minsOut: &mn,
                        kvHeads: effHeads, headDim: effHeadDim)
                    _ = qrow.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in memcpy(pageBase.advanced(by: off * (rowElems / 2)), raw.baseAddress!, rowElems / 2) }
                    // header: totalGroups × {scale f16, min f16} interleaved
                    let hBase = pageBase.advanced(by: layout.dataBytesPerPage + off * (totalGroups * 4))
                    for g in 0..<totalGroups {
                        hBase.assumingMemoryBound(to: Float16.self)[2 * g] = sc[g]
                        hBase.assumingMemoryBound(to: Float16.self)[2 * g + 1] = mn[g]
                    }
                }
            }
        }
        st.writtenTokens = max(st.writtenTokens, positions.upperBound)
    }

    /// Call before encoding/submitting any engine GPU write into this token
    /// range. It privatizes every shared page intersecting the range. The
    /// device table's contents change, but its buffer, base, and length do
    public func prepareWrites(_ h: QWKVHandle, positions: Range<Int>) throws {
        try withPagePressure {
            try prepareWritesEntryLocked(h, positions: positions)
        }
    }

    private func prepareWritesEntryLocked(_ h: QWKVHandle, positions: Range<Int>) throws {
        guard h.cache === self, !h.finished, let st = seqs[h.seq] else {
            throw QWKVError.badArgument("unknown, foreign, or finished seq")
        }
        try prepareWritesLocked(st, handle: h, positions: positions, only: nil)
    }

    /// Kernel-facing views for one layer (pool extent + this seq's device
    /// page table for exactly this (layer, kind)).
    public func layerViews(_ h: QWKVHandle, layer: Int) throws -> (k: QWKVLayerView, v: QWKVLayerView) {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq], let li = layerIndex(layer) else {
            throw QWKVError.badArgument("unknown seq/layer")
        }
        let pk = pool(layer, .k).pages, pv = pool(layer, .v).pages
        let tk = tableTensor(h.seq, li, .k, st.tablesK[li].count)
        let tv = tableTensor(h.seq, li, .v, st.tablesV[li].count)
        return (QWKVLayerView(pool: pk.region, table: tk, pageBytes: bytesPerPage, mode: mode),
                QWKVLayerView(pool: pv.region, table: tv, pageBytes: bytesPerPage, mode: mode))
    }

    public func layerViewsAll(_ h: QWKVHandle, layers: [Int]) throws -> [(k: QWKVLayerView, v: QWKVLayerView)] {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq] else {
            throw QWKVError.badArgument("unknown seq")
        }
        var views: [(k: QWKVLayerView, v: QWKVLayerView)] = []
        views.reserveCapacity(layers.count)
        for layer in layers {
            guard let li = layerIndex(layer) else {
                throw QWKVError.badArgument("unknown layer")
            }
            let pk = pool(layer, .k).pages, pv = pool(layer, .v).pages
            let tk = tableTensor(h.seq, li, .k, st.tablesK[li].count)
            let tv = tableTensor(h.seq, li, .v, st.tablesV[li].count)
            views.append((QWKVLayerView(pool: pk.region, table: tk, pageBytes: bytesPerPage, mode: mode),
                          QWKVLayerView(pool: pv.region, table: tv, pageBytes: bytesPerPage, mode: mode)))
        }
        return views
    }

    /// Refcounted snapshot (prefix cache C20): pins every page in use across
    /// all layers. The returned view carries the page ids + counts so the
    /// consumer can rebuild any (layer, kind) table for a shared prefix.
    public func snapshot(_ h: QWKVHandle) throws -> QWKVSnap {
        lock.lock(); defer { lock.unlock() }
        guard let st = seqs[h.seq] else { throw QWKVError.badArgument("unknown seq") }
        var pins: [(Int, QWKVKind, [Int32])] = []
        for (li, l) in layers.enumerated() {
            for kind in QWKVKind.allCases {
                let ids = kind == .k ? st.tablesK[li] : st.tablesV[li]
                let key = "\(l):\(kind.rawValue)"
                var bucket = st.pins[key] ?? [:]
                for p in ids { bucket[p, default: 0] += 1 }
                st.pins[key] = bucket
                pins.append((l, kind, ids))
            }
        }
        return QWKVSnap(seq: h.seq, tokenCount: st.writtenTokens, cache: self, pins: pins)
    }

    /// Computes serialized plane byte count for `tokenCount` tokens in the current `mode`.
    func prepareWritesLocked(_ st: SeqState, handle: QWKVHandle,
                                    positions: Range<Int>, only: (Int, QWKVKind)?) throws {
        guard positions.lowerBound >= 0, positions.upperBound >= positions.lowerBound else {
            throw QWKVError.badArgument("write range invalid")
        }
        guard !positions.isEmpty else { return }
        let firstPage = positions.lowerBound / Self.pageTokens
        let lastPage = (positions.upperBound - 1) / Self.pageTokens
        var copies: [(li: Int, kind: QWKVKind, index: Int, old: Int32, new: Int32)] = []
        var grabbed: [(Int, Int32)] = []
        do {
            for li in layers.indices {
                for kind in QWKVKind.allCases {
                    if let only, (only.0 != li || only.1 != kind) { continue }
                    let ids = kind == .k ? st.tablesK[li] : st.tablesV[li]
                    let shared = kind == .k ? st.sharedK[li] : st.sharedV[li]
                    guard positions.upperBound <= ids.count * Self.pageTokens,
                          shared.count == ids.count, lastPage < ids.count else {
                        throw QWKVError.badArgument("write range exceeds admitted page capacity")
                    }
                    for index in firstPage...lastPage where shared[index] {
                        guard let page = pool(layers[li], kind).pages.allocPage() else {
                            throw QWKVError.outOfPages(asked: 1, free: pool(layers[li], kind).pages.freePages)
                        }
                        let replacement = Int32(page)
                        grabbed.append((poolIndex(li, kind), replacement))
                        copies.append((li, kind, index, ids[index], replacement))
                    }
                }
            }
        } catch {
            for (pool, page) in grabbed { poolsFlat[pool].pages.freePage(Int(page)) }
            throw error
        }

        for copy in copies {
            let pa = pool(layers[copy.li], copy.kind).pages
            pa.region.pointer.advanced(by: Int(copy.new) * bytesPerPage)
                .copyMemory(from: pa.region.pointer.advanced(by: Int(copy.old) * bytesPerPage),
                            byteCount: bytesPerPage)
        }
        for copy in copies {
            if copy.kind == .k {
                st.tablesK[copy.li][copy.index] = copy.new
                st.sharedK[copy.li][copy.index] = false
                uploadSlot(handle.seq, copy.li, .k, st.tablesK[copy.li])
            } else {
                st.tablesV[copy.li][copy.index] = copy.new
                st.sharedV[copy.li][copy.index] = false
                uploadSlot(handle.seq, copy.li, .v, st.tablesV[copy.li])
            }
            releaseSharedPage(poolIndex(copy.li, copy.kind), copy.old)
        }
    }

    // ── table plumbing (lock held by every caller) ──────────────────────────
    func tableTensor(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind, _ count: Int) -> QWTensorValue {
        let base = slotBase(seq, li, kind)
        return QWTensorValue(region: QWMRegion(buffer: tableBuffer, base: base, length: count * 4),
                             offset: 0, shape: QWTensorShape([max(count, 1)], .i32))
    }
    /// Rewrite a (seq, layer, kind) device slot in place (same base — kernel
    /// bindings never move). Zeroes any tail past the current id count so an
    /// out-of-range kernel read sees page 0, never a stale/previous-seq id.
    func uploadSlot(_ seq: QWKVSeqID, _ li: Int, _ kind: QWKVKind, _ ids: [Int32]) {
        let base = slotBase(seq, li, kind)
        let byteCount = ids.count * 4
        if byteCount > 0 {
            // Empty-array law: withUnsafeBytes on [] hands back a nil
            // baseAddress — the memcpy would trap. The memset below already
            // zeroes the whole slot in that case, so skip the copy.
            ids.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                _ = memcpy(tableBuffer.contents().advanced(by: base), raw.baseAddress!, byteCount)
            }
        }
        if byteCount < Self.tableSlotBytes {
            memset(tableBuffer.contents().advanced(by: base + byteCount), 0,
                   Self.tableSlotBytes - byteCount)
        }
    }
}
