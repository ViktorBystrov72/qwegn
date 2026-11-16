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

    private var biKey0: [Int32]
    private var biKey1: [Int32]
    private var biNext: [Int32]
    private var uniKey: [Int32]
    private var uniNext: [Int32]

    public init() {
        biKey0 = [Int32](repeating: -1, count: Self.totalSlots)
        biKey1 = [Int32](repeating: -1, count: Self.totalSlots)
        biNext = [Int32](repeating: -1, count: Self.totalSlots)
        uniKey = [Int32](repeating: -1, count: Self.totalSlots)
        uniNext = [Int32](repeating: -1, count: Self.totalSlots)
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
        guard let firstIdx = tokens.indices.first else { return }
        var prevPrev = previousToken
        var prev = tokens[firstIdx]
        var idx = tokens.index(after: firstIdx)
        while idx < tokens.endIndex {
            let cur = tokens[idx]
            recordTransition(previous: prevPrev, current: prev, next: cur)
            prevPrev = prev
            prev = cur
            idx = tokens.index(after: idx)
        }
    }

    public func recordTransition(previous: Int?, current: Int, next: Int) {
        guard current >= 0, next >= 0,
              current <= Int(Int32.max), next <= Int(Int32.max) else { return }
        let cur32 = Int32(current)
        let next32 = Int32(next)
        recordUnigram(key: cur32, next: next32)
        if let previous, previous >= 0, previous <= Int(Int32.max) {
            recordBigram(k0: Int32(previous), k1: cur32, next: next32)
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
        guard maxCount > drafts.count, pendingToken >= 0 else { return 0 }
        let initialCount = drafts.count
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
            if let prevVal = prev, prevVal >= 0,
               let biHit = lookupBigram(k0: Int32(prevVal), k1: Int32(cur)) {
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
               let uniHit = lookupUnigram(key: Int32(cur)) {
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
        let slot0 = Self.hashBigram(k0, k1) * Self.ways
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
    private func recordUnigram(key: Int32, next: Int32) {
        let slot0 = Self.hashUnigram(key) * Self.ways
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
