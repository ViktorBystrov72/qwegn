// QWTokenRecycler.swift — Zero-alloc n-gram adjacency drafter (arXiv:2408.08696)
// harvesting prompt tokens and rejected target argmax rows across decode rounds.

import Foundation

/// Fixed-capacity 2-way associative transition matrix (bigram + unigram) with
/// zero heap allocations after init and MRU replacement on conflict.
public final class QWTokenRecycler: @unchecked Sendable {
    private static let bucketCount = 8192
    private static let ways = 2
    private static let totalSlots = bucketCount * ways
    private static let mixConstant: UInt64 = 0x9E37_79B9_7F4A_7C15

    private var biKey0: ContiguousArray<Int32>
    private var biKey1: ContiguousArray<Int32>
    private var biNext: ContiguousArray<Int32>
    private var uniKey: ContiguousArray<Int32>
    private var uniNext: ContiguousArray<Int32>

    public init() {
        biKey0 = ContiguousArray<Int32>(repeating: -1, count: Self.totalSlots)
        biKey1 = ContiguousArray<Int32>(repeating: -1, count: Self.totalSlots)
        biNext = ContiguousArray<Int32>(repeating: -1, count: Self.totalSlots)
        uniKey = ContiguousArray<Int32>(repeating: -1, count: Self.totalSlots)
        uniNext = ContiguousArray<Int32>(repeating: -1, count: Self.totalSlots)
    }

    public func reset() {
        biKey0.withUnsafeMutableBufferPointer { $0.initialize(repeating: -1) }
        biKey1.withUnsafeMutableBufferPointer { $0.initialize(repeating: -1) }
        biNext.withUnsafeMutableBufferPointer { $0.initialize(repeating: -1) }
        uniKey.withUnsafeMutableBufferPointer { $0.initialize(repeating: -1) }
        uniNext.withUnsafeMutableBufferPointer { $0.initialize(repeating: -1) }
    }

    public func ingestSequence(_ tokens: [Int], previousToken: Int? = nil) {
        ingestSequence(tokens[...], previousToken: previousToken)
    }

    public func ingestSequence(_ tokens: ArraySlice<Int>, previousToken: Int? = nil) {
        ingestCollection(tokens, previousToken: previousToken) { Int32(exactly: $0) }
    }

    /// Zero-allocation overload ingesting `Int64` history buffers directly.
    public func ingestSequence(_ tokens: [Int64], previousToken: Int? = nil) {
        ingestSequence(tokens[...], previousToken: previousToken)
    }

    /// Zero-allocation slice overload ingesting `Int64` history slices directly.
    public func ingestSequence(_ tokens: ArraySlice<Int64>, previousToken: Int? = nil) {
        ingestCollection(tokens, previousToken: previousToken) { Int32(exactly: $0) }
    }

    private func ingestCollection<C: Collection>(
        _ tokens: C,
        previousToken: Int?,
        toToken32: (C.Element) -> Int32?
    ) {
        guard let firstIdx = tokens.indices.first else { return }
        var prevPrev32: Int32 = -1
        if let previousToken, let v = Int32(exactly: previousToken), v >= 0 {
            prevPrev32 = v
        }
        var prev32: Int32 = -1
        if let v = toToken32(tokens[firstIdx]), v >= 0 {
            prev32 = v
        }
        var idx = tokens.index(after: firstIdx)
        guard idx < tokens.endIndex else { return }
        biKey0.withUnsafeMutableBufferPointer { bk0 in
            biKey1.withUnsafeMutableBufferPointer { bk1 in
                biNext.withUnsafeMutableBufferPointer { bn in
                    uniKey.withUnsafeMutableBufferPointer { uk in
                        uniNext.withUnsafeMutableBufferPointer { un in
                            while idx < tokens.endIndex {
                                let cur32: Int32
                                if let v = toToken32(tokens[idx]), v >= 0 {
                                    cur32 = v
                                } else {
                                    cur32 = -1
                                }
                                if prev32 >= 0 && cur32 >= 0 {
                                    Self.writeUnigram(key: prev32, next: cur32, uniKey: uk, uniNext: un)
                                    if prevPrev32 >= 0 {
                                        Self.writeBigram(k0: prevPrev32, k1: prev32, next: cur32,
                                                         biKey0: bk0, biKey1: bk1, biNext: bn)
                                    }
                                }
                                prevPrev32 = prev32
                                prev32 = cur32
                                idx = tokens.index(after: idx)
                            }
                        }
                    }
                }
            }
        }
    }

    public func recordTransition(previous: Int?, current: Int, next: Int) {
        guard let cur32 = Int32(exactly: current), cur32 >= 0,
              let next32 = Int32(exactly: next), next32 >= 0 else { return }
        recordUnigram(key: cur32, next: next32)
        if let previous, let prev32 = Int32(exactly: previous), prev32 >= 0 {
            recordBigram(k0: prev32, k1: cur32, next: next32)
        }
    }

    /// Extends `drafts` in place up to `maxCount` using bigram transitions first
    /// and bounded unigram fallbacks, guarding against short cycles.
    @discardableResult
    public func appendDrafts(
        into drafts: inout [Int],
        maxCount: Int,
        previousToken: Int?,
        pendingToken: Int,
        maxUnigramFallbacks: Int = 2
    ) -> Int {
        guard maxCount > drafts.count,
              let pending32 = Int32(exactly: pendingToken), pending32 >= 0 else { return 0 }
        let initialCount = drafts.count
        if drafts.capacity < maxCount {
            drafts.reserveCapacity(maxCount)
        }
        var prev: Int?
        var cur: Int
        if drafts.isEmpty {
            prev = previousToken
            cur = pendingToken
        } else if drafts.count == 1 {
            prev = pendingToken
            cur = drafts[0]
        } else {
            prev = drafts[drafts.count - 2]
            cur = drafts[drafts.count - 1]
        }
        var unigramUsed = 0
        while drafts.count < maxCount {
            guard let k1 = Int32(exactly: cur), k1 >= 0 else { break }
            if let prevVal = prev,
               let k0 = Int32(exactly: prevVal), k0 >= 0,
               let biHit = lookupBigram(k0: k0, k1: k1) {
                let candidate = Int(biHit)
                if hasConsecutivePair(first: cur, second: candidate, pendingToken: pendingToken, drafts: drafts) {
                    break
                }
                drafts.append(candidate)
                prev = cur
                cur = candidate
                continue
            }
            if unigramUsed < maxUnigramFallbacks,
               let uniHit = lookupUnigram(key: k1) {
                let candidate = Int(uniHit)
                if candidate == cur || drafts.contains(candidate) {
                    break
                }
                drafts.append(candidate)
                unigramUsed += 1
                prev = cur
                cur = candidate
                continue
            }
            break
        }
        return drafts.count - initialCount
    }

    @inline(__always)
    private static func hashBigram(_ k0: Int32, _ k1: Int32) -> Int {
        let a = UInt64(bitPattern: Int64(k0)) &* mixConstant
        let b = UInt64(bitPattern: Int64(k1))
        return Int((a ^ b) & UInt64(bucketCount - 1))
    }

    @inline(__always)
    private static func hashUnigram(_ key: Int32) -> Int {
        let h = UInt64(bitPattern: Int64(key)) &* mixConstant
        return Int(h & UInt64(bucketCount - 1))
    }

    @inline(__always)
    private func recordBigram(k0: Int32, k1: Int32, next: Int32) {
        biKey0.withUnsafeMutableBufferPointer { bk0 in
            biKey1.withUnsafeMutableBufferPointer { bk1 in
                biNext.withUnsafeMutableBufferPointer { bn in
                    Self.writeBigram(k0: k0, k1: k1, next: next, biKey0: bk0, biKey1: bk1, biNext: bn)
                }
            }
        }
    }

    @inline(__always)
    private func recordUnigram(key: Int32, next: Int32) {
        uniKey.withUnsafeMutableBufferPointer { uk in
            uniNext.withUnsafeMutableBufferPointer { un in
                Self.writeUnigram(key: key, next: next, uniKey: uk, uniNext: un)
            }
        }
    }

    @inline(__always)
    private static func writeBigram(
        k0: Int32,
        k1: Int32,
        next: Int32,
        biKey0: UnsafeMutableBufferPointer<Int32>,
        biKey1: UnsafeMutableBufferPointer<Int32>,
        biNext: UnsafeMutableBufferPointer<Int32>
    ) {
        let slot0 = hashBigram(k0, k1) * ways
        if biKey0[slot0] == k0 && biKey1[slot0] == k1 {
            biNext[slot0] = next
            return
        }
        let slot1 = slot0 + 1
        biKey0[slot1] = biKey0[slot0]
        biKey1[slot1] = biKey1[slot0]
        biNext[slot1] = biNext[slot0]
        biKey0[slot0] = k0
        biKey1[slot0] = k1
        biNext[slot0] = next
    }

    @inline(__always)
    private static func writeUnigram(
        key: Int32,
        next: Int32,
        uniKey: UnsafeMutableBufferPointer<Int32>,
        uniNext: UnsafeMutableBufferPointer<Int32>
    ) {
        let slot0 = hashUnigram(key) * ways
        if uniKey[slot0] == key {
            uniNext[slot0] = next
            return
        }
        let slot1 = slot0 + 1
        uniKey[slot1] = uniKey[slot0]
        uniNext[slot1] = uniNext[slot0]
        uniKey[slot0] = key
        uniNext[slot0] = next
    }

    @inline(__always)
    private func lookupBigram(k0: Int32, k1: Int32) -> Int32? {
        let slot0 = Self.hashBigram(k0, k1) * Self.ways
        if biKey0[slot0] == k0 && biKey1[slot0] == k1 {
            return biNext[slot0]
        }
        let slot1 = slot0 + 1
        if biKey0[slot1] == k0 && biKey1[slot1] == k1 {
            return biNext[slot1]
        }
        return nil
    }

    @inline(__always)
    private func lookupUnigram(key: Int32) -> Int32? {
        let slot0 = Self.hashUnigram(key) * Self.ways
        if uniKey[slot0] == key {
            return uniNext[slot0]
        }
        let slot1 = slot0 + 1
        if uniKey[slot1] == key {
            return uniNext[slot1]
        }
        return nil
    }

    @inline(__always)
    private func hasConsecutivePair(first: Int, second: Int, pendingToken: Int, drafts: [Int]) -> Bool {
        guard !drafts.isEmpty else { return false }
        var prev = pendingToken
        for tok in drafts {
            if prev == first && tok == second { return true }
            prev = tok
        }
        return false
    }
}
