// QWStateReferenceHarness.swift — C20 golden reference harness (test-only).
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWToken

public enum QWStateError: Error, CustomStringConvertible, Equatable {
    case slotPressure(live: Int, cap: Int)
    case unknownSeq(Int)
    case dupSeq(Int)
    case slotTooSmall(asked: Int, capacity: Int)
    case badTrace(String)

    public var description: String {
        switch self {
        case .slotPressure(let l, let c): return "QWState: slot pressure (live \(l), cap \(c)) — refuse before jetsam does"
        case .unknownSeq(let s): return "QWState: unknown sequence \(s)"
        case .dupSeq(let s): return "QWState: duplicate sequence \(s)"
        case .slotTooSmall(let a, let c): return "QWState: slot capacity \(c) B < state \(a) B"
        case .badTrace(let m): return "QWState: golden trace mismatch: \(m)"
        }
    }
}

/// One sequence's mutable state record. Buffers are ARENA-owned fixed
/// slots; this record is the bookkeeping around them.
public final class QWSeqState: @unchecked Sendable {
    public let seq: Int
    public let gdnStates: [QWTensorValue]
    public let convRings: [QWTensorValue]
    public internal(set) var position: Int
    public internal(set) var generated: Int
    public internal(set) var prefixFP: String?
    public internal(set) var clonedFromPrefix: Bool = false
    public internal(set) var kv: QWKVHandle?

    init(seq: Int, gdnStates: [QWTensorValue], convRings: [QWTensorValue]) {
        self.seq = seq
        self.gdnStates = gdnStates
        self.convRings = convRings
        self.position = 0
        self.generated = 0
    }

    public struct Snap: Sendable, Equatable {
        public let seq: Int
        public let position: Int
        public let generated: Int
        public let prefixFP: String?
    }
    public func snapshot() -> Snap {
        Snap(seq: seq, position: position, generated: generated, prefixFP: prefixFP)
    }
}

/// A prefix-cache entry: an IMMUTABLE clone (state bytes + cursor).
public struct QWPrefixEntry: Sendable {
    public let fp: String
    public let state: [Data]
    public let rings: [Data]
    public let snap: QWSeqState.Snap
}

/// Prefix-cache LRU reference store for C20 family B golden trace tests.
public final class QWPrefixCache: @unchecked Sendable {
    public let lruN: Int
    private let lock = NSLock()
    private var entries: [String: QWPrefixEntry] = [:]
    private var lru: [String] = []

    public init(lruN: Int) { self.lruN = lruN }

    @discardableResult
    public func commit(_ e: QWPrefixEntry) -> [String] {
        lock.lock(); defer { lock.unlock() }
        entries[e.fp] = e
        if let i = lru.firstIndex(of: e.fp) { lru.remove(at: i) }
        lru.insert(e.fp, at: 0)
        var evicted: [String] = []
        while lru.count > lruN, let tail = lru.popLast() {
            entries[tail] = nil
            evicted.append(tail)
        }
        return evicted
    }

    public func lookup(_ fp: String) -> QWPrefixEntry? {
        lock.lock(); defer { lock.unlock() }
        guard let e = entries[fp] else { return nil }
        if let i = lru.firstIndex(of: fp) { lru.remove(at: i); lru.insert(fp, at: 0) }
        return e
    }

    public var order: [String] { lock.lock(); defer { lock.unlock() }; return lru }
}

/// Sequence-state manager reference harness for C20 golden verification.
public final class QWState: @unchecked Sendable {
    public static let module = "QWState"

    public let cap: Int
    public let prefix: QWPrefixCache
    private let lock = NSLock()
    private let slots: [QWSeqState]
    private var freeSlots: [Int]
    private var seqOf: [Int: Int] = [:]
    public let journal: QWDraftJournal

    static func zeroAll(_ st: QWSeqState) {
        for t in st.gdnStates + st.convRings {
            memset(t.region.buffer.contents().advanced(by: t.region.base + t.offset), 0, t.bytes)
        }
    }

    static func copyAll(into st: QWSeqState, from e: QWPrefixEntry) {
        for l in 0..<e.state.count where l < st.gdnStates.count {
            let d = st.gdnStates[l]
            let p = d.region.buffer.contents().advanced(by: d.region.base + d.offset)
            e.state[l].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                _ = memcpy(p, base, min(d.bytes, raw.count))
            }
        }
        for l in 0..<e.rings.count where l < st.convRings.count {
            let d = st.convRings[l]
            let p = d.region.buffer.contents().advanced(by: d.region.base + d.offset)
            e.rings[l].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                _ = memcpy(p, base, min(d.bytes, raw.count))
            }
        }
    }

    public init(hal: QWHal, arena: QWArena, cap: Int, lruN: Int, maxDrafts: Int = 2,
                geometry: QWModelGeometry) throws {
        let gdnLayers = QWSlotGeometry.gdnLayers(geometry: geometry)
        let recElems = QWSlotGeometry.recElems(geometry: geometry)
        let ringElems = QWSlotGeometry.ringElems(geometry: geometry)
        let need = cap * QWSlotGeometry.stateBytes(geometry: geometry)
        guard need <= arena.regionSizes.gdn else {
            throw QWStateError.slotTooSmall(asked: need, capacity: arena.regionSizes.gdn)
        }
        self.cap = cap
        self.prefix = QWPrefixCache(lruN: lruN)
        self.journal = QWDraftJournal(maxDrafts: maxDrafts)
        self.freeSlots = Array(0..<cap)
        let sub = arena.gdnRegion()
        var carved: [QWSeqState] = []
        carved.reserveCapacity(cap)
        for s in 0..<cap {
            var recs: [QWTensorValue] = []
            var rings: [QWTensorValue] = []
            recs.reserveCapacity(gdnLayers)
            rings.reserveCapacity(gdnLayers)
            for _ in 0..<gdnLayers {
                recs.append(try sub.allocate(QWTensorShape([recElems], .f32)))
            }
            for _ in 0..<gdnLayers {
                rings.append(try sub.allocate(QWTensorShape([ringElems], .f32)))
            }
            carved.append(QWSeqState(seq: s, gdnStates: recs, convRings: rings))
        }
        self.slots = carved
        sub.markSteadyState()
    }

    public var liveCount: Int { lock.lock(); defer { lock.unlock() }; return seqOf.count }

    public func state(for seq: Int) throws -> QWSeqState {
        lock.lock(); defer { lock.unlock() }
        guard let slot = seqOf[seq] else { throw QWStateError.unknownSeq(seq) }
        return slots[slot]
    }

    public func adopt(seq: Int, prefixFP: String? = nil) throws -> QWSeqState {
        lock.lock(); defer { lock.unlock() }
        guard seqOf[seq] == nil, !freeSlots.isEmpty else {
            throw QWStateError.slotPressure(live: seqOf.count, cap: cap)
        }
        let slot = freeSlots.removeFirst()
        seqOf[seq] = slot
        let st = slots[slot]
        st.position = 0
        st.generated = 0
        st.prefixFP = prefixFP
        var cloned = false
        if let fp = prefixFP, let e = prefix.lookup(fp) {
            Self.copyAll(into: st, from: e)
            st.position = e.snap.position
            cloned = true
        } else {
            Self.zeroAll(st)
        }
        st.clonedFromPrefix = cloned
        return st
    }

    public func finish(seq: Int) throws {
        lock.lock(); defer { lock.unlock() }
        guard let slot = seqOf.removeValue(forKey: seq) else { throw QWStateError.unknownSeq(seq) }
        freeSlots.append(slot)
    }

    public func snapshot(_ st: QWSeqState, as fp: String) -> QWPrefixEntry {
        var state: [Data] = []
        var rings: [Data] = []
        for l in 0..<st.gdnStates.count {
            let t = st.gdnStates[l]
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            state.append(Data(bytes: p, count: t.bytes))
        }
        for t in st.convRings {
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            rings.append(Data(bytes: p, count: t.bytes))
        }
        return QWPrefixEntry(fp: fp, state: state, rings: rings, snap: st.snapshot())
    }

    public func rollback(_ st: QWSeqState, to e: QWPrefixEntry) {
        for l in 0..<e.state.count {
            let dst = st.gdnStates[l]
            let p = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            e.state[l].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                _ = memcpy(p, base, min(dst.bytes, raw.count))
            }
        }
        for l in 0..<e.rings.count {
            let dst = st.convRings[l]
            let p = dst.region.buffer.contents().advanced(by: dst.region.base + dst.offset)
            e.rings[l].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                _ = memcpy(p, base, min(dst.bytes, raw.count))
            }
        }
        st.position = e.snap.position
    }

    public func advance(_ st: QWSeqState, tokens: Int) {
        lock.lock(); defer { lock.unlock() }
        st.generated += tokens
        st.position += tokens
    }
}

/// MTP draft cursor rollback reference harness for C20 tests.
public final class QWDraftJournal: @unchecked Sendable {
    public let maxDrafts: Int
    private let lock = NSLock()
    private var commits: [Int] = []

    public init(maxDrafts: Int) { self.maxDrafts = maxDrafts }

    public var pending: Int { lock.lock(); defer { lock.unlock() }; return commits.count }

    @discardableResult
    public func journal(position: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard commits.count < maxDrafts else { return false }
        commits.append(position)
        return true
    }

    public func accept(_ n: Int) {
        lock.lock(); defer { lock.unlock() }
        commits.removeFirst(min(n, commits.count))
    }

    public func reject() -> Int? {
        lock.lock(); defer { lock.unlock() }
        defer { commits.removeAll() }
        return commits.first
    }
}
