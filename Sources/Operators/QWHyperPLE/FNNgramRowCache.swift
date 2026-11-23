// An L1 hot-row cache in front of the PLE n-gram table's row source.
//
// The n-gram ids are EXPECTED to be steeply skewed, so the same rows are
// read over and over; caching them removes the pread() and the decode.
// That premise is not yet measured — see specs/30-flashnext-architecture.md
// §10 item 8. The cache breaks even at a ~50% hit rate.

import Foundation

/// One intrusive doubly-linked node in a stripe's LRU list.
struct FNNgramLRUEntry {
    var id: Int
    var row: [Float]
    var prev: Int32
    var next: Int32
}

/// The cache refuses a row whose size is not the one it was built for: a
/// longer row would be under-charged against the budget.
public enum FNNgramRowCacheError: Error, CustomStringConvertible {
    case rowSizeMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case let .rowSizeMismatch(expected, got):
            return "ngram cache row is \(got) bytes, the cache was built for \(expected)"
        }
    }
}

/// One stripe of the cache: its own lock, its own rows, its own O(1) LRU list.
/// A **class**, not a struct — a struct stripe puts every stripe's mutable
/// state in one array's buffer, so two concurrent element writes conflict.
public final class FNNgramRowCacheStripe: @unchecked Sendable {
    public let lock = NSLock()
    public var rows: [Int: Int] = [:]
    private var nodes: [FNNgramLRUEntry] = []
    private var head: Int32 = -1
    private var tail: Int32 = -1
    private var freeHead: Int32 = -1
    public var bytes = 0
    public var hits = 0
    public var misses = 0

    /// Promotes `index` to the MRU head in O(1). Caller holds `lock`.
    @inline(__always)
    func promoteToFront(_ index: Int) {
        let idx32 = Int32(index)
        guard idx32 != head else { return }
        let prev = nodes[index].prev
        let next = nodes[index].next
        if prev >= 0 { nodes[Int(prev)].next = next }
        if next >= 0 { nodes[Int(next)].prev = prev }
        if idx32 == tail { tail = prev }
        nodes[index].prev = -1
        nodes[index].next = head
        if head >= 0 { nodes[Int(head)].prev = idx32 }
        head = idx32
        if tail < 0 { tail = idx32 }
    }

    /// Returns the row stored at node `index`. Caller holds `lock`.
    @inline(__always)
    func row(at index: Int) -> [Float] {
        nodes[index].row
    }

    /// Inserts a fresh `(id, row)` at the MRU head in O(1). Caller holds `lock`.
    @inline(__always)
    func insertFront(id: Int, row: [Float]) {
        let index: Int
        if freeHead >= 0 {
            index = Int(freeHead)
            freeHead = nodes[index].next
            nodes[index] = FNNgramLRUEntry(id: id, row: row, prev: -1, next: head)
        } else {
            index = nodes.count
            nodes.append(FNNgramLRUEntry(id: id, row: row, prev: -1, next: head))
        }
        let idx32 = Int32(index)
        if head >= 0 { nodes[Int(head)].prev = idx32 }
        head = idx32
        if tail < 0 { tail = idx32 }
        rows[id] = index
    }

    /// Evicts and returns the LRU victim id in O(1), if any. Caller holds `lock`.
    @inline(__always)
    func popLeastRecentlyUsed() -> Int? {
        let victimIdx32 = tail
        guard victimIdx32 >= 0 else { return nil }
        let victimIdx = Int(victimIdx32)
        let victimId = nodes[victimIdx].id
        let prev = nodes[victimIdx].prev
        if prev >= 0 {
            nodes[Int(prev)].next = -1
        } else {
            head = -1
        }
        tail = prev
        nodes[victimIdx].row = []
        nodes[victimIdx].prev = -1
        nodes[victimIdx].next = freeHead
        freeHead = victimIdx32
        rows.removeValue(forKey: victimId)
        return victimId
    }

    /// Clears every entry while retaining buffer capacities. Caller holds `lock`.
    func resetAll() {
        rows.removeAll(keepingCapacity: true)
        nodes.removeAll(keepingCapacity: true)
        head = -1
        tail = -1
        freeHead = -1
        bytes = 0
        hits = 0
        misses = 0
    }
}

/// An L1 hot-row cache in front of an `FNNgramRows` source.
/// Locking is **striped** — one `NSLock` per stripe — because `FNNgramEmbedding.embedLast` reads rows from
/// `DispatchQueue.concurrentPerform`; a single mutex would serialise them.
public final class FNNgramRowCache: @unchecked Sendable {
    /// Target rows per stripe used to size the stripe count cleanly for concurrency.
    static let targetRowsPerStripe = 32
    /// The most stripes the cache will ever build, whatever the budget.
    static let maxStripeCount = 1 << 16

    private let capacityBytes: Int
    private let rowBytes: Int
    public let stripes: [FNNgramRowCacheStripe]
    private let stripeBudget: Int

    /// - Parameters: `capacityBytes` is the cache's total row budget, `rowBytes` the bytes one row occupies
    ///   (`headDim * 4`), and `maxStripes` the most locks to spread the rows over. The budget is **global**:
    ///   each stripe gets `capacityBytes / stripes.count`, so the cache never exceeds it.
    public init(capacityBytes: Int, rowBytes: Int, maxStripes: Int = 4096) {
        precondition(rowBytes > 0, "the cache needs a positive row size")
        self.capacityBytes = max(0, capacityBytes)
        self.rowBytes = rowBytes
        // A stripe that cannot hold one row is useless, so there are never
        // more stripes than the budget has rows.
        let budgetRows = max(1, capacityBytes / rowBytes)
        let wanted = min(max(1, min(maxStripes, FNNgramRowCache.maxStripeCount)),
                         max(1, budgetRows / FNNgramRowCache.targetRowsPerStripe))
        // The largest power of two <= wanted, so the stripe index is a mask
        // and not a division.
        let count = 1 << (wanted.bitWidth - 1 - wanted.leadingZeroBitCount)
        self.stripes = (0..<count).map { _ in FNNgramRowCacheStripe() }
        self.stripeBudget = max(1, capacityBytes / count)
    }

    private func stripe(for id: Int) -> FNNgramRowCacheStripe {
        let h = (UInt(truncatingIfNeeded: id) &* 0x9E37_79B9) >> 32
        return stripes[Int(h & UInt(stripes.count - 1))]
    }

    /// The cached row, or nil on a miss.
    public func value(for id: Int) -> [Float]? {
        let s = stripe(for: id)
        s.lock.lock()
        defer { s.lock.unlock() }
        guard let idx = s.rows[id] else {
            s.misses += 1
            return nil
        }
        s.hits += 1
        s.promoteToFront(idx)
        return s.row(at: idx)
    }

    /// Cache a row. A row that is already cached is left alone.
    /// The row must be exactly the size the cache was built for.
    public func store(_ id: Int, _ row: [Float]) {
        precondition(row.count * MemoryLayout<Float>.size == rowBytes,
                     "the cache was built for \(rowBytes)-byte rows, got \(row.count * 4)")
        let s = stripe(for: id)
        s.lock.lock()
        defer { s.lock.unlock() }
        guard s.rows[id] == nil else { return }
        while s.bytes + rowBytes > stripeBudget, s.popLeastRecentlyUsed() != nil {
            s.bytes -= rowBytes
        }
        guard s.bytes + rowBytes <= stripeBudget else { return }
        s.insertFront(id: id, row: row)
        s.bytes += rowBytes
    }

    /// Cache a row, refusing one of the wrong size instead of trapping.
    public func storeChecked(_ id: Int, _ row: [Float]) throws {
        let got = row.count * MemoryLayout<Float>.size
        guard got == rowBytes else {
            throw FNNgramRowCacheError.rowSizeMismatch(expected: rowBytes, got: got)
        }
        store(id, row)
    }

    /// Drop every cached row and reset the hit/miss counters.
    public func clear() {
        for s in stripes {
            s.lock.lock()
            s.resetAll()
            s.lock.unlock()
        }
    }

    /// The cache's total budget, in bytes.
    public var budgetBytes: Int { capacityBytes }

    /// Whether the cache can hold anything at all. A model without a PLE
    /// layer gets a cache with no budget, and a miss must not then pay for a
    /// row copy that will be thrown away.
    public var isEnabled: Bool { capacityBytes > 0 }
}

/// An `FNNgramRows` that serves from a cache and falls through to the source.
public final class FNNgramCachedRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let cache: FNNgramRowCache
    private let backing: any FNNgramRows

    public init(backing: any FNNgramRows, capacityBytes: Int) {
        self.backing = backing
        self.headDim = backing.headDim
        self.cache = FNNgramRowCache(capacityBytes: capacityBytes,
                                     rowBytes: backing.headDim * MemoryLayout<Float>.size)
    }

    public func row(_ id: Int) -> [Float] {
        if let hit = cache.value(for: id) { return hit }
        let fresh = backing.row(id)
        cache.store(id, fresh)
        return fresh
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) throws {
        if let hit = cache.value(for: id) {
            hit.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: headDim)
            }
            return
        }
        try backing.copyRow(id, to: dst)
        // A cache with no budget stores nothing, so do not pay for a copy
        // that would be thrown away.
        guard cache.isEnabled else { return }
        let n = headDim
        let fresh = [Float](unsafeUninitializedCapacity: n) { out, count in
            out.baseAddress!.update(from: dst, count: n)
            count = n
        }
        do {
            try cache.storeChecked(id, fresh)
        } catch {
            // A row that is not the cache's row size is a broken contract.
            // The read already returned the right data, so the row is simply
            // not cached rather than the process going down over it.
        }
    }
}
