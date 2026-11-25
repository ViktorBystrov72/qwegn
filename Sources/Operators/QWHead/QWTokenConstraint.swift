// QWTokenConstraint.swift — request-local token constraints for selection.
//
// A constraint restricts which token ids the sampler may select at each step
// (for example a JSON-schema grammar). It is owned by exactly one request and
// advanced once per selected token; it is never shared across requests and is
// not part of any model or KV state.

import Foundation

public protocol QWTokenConstraint: AnyObject, Sendable {
    /// The ascending, de-duplicated token ids selectable at the current step,
    /// or nil when the constraint does not restrict this step.
    /// An empty array means the constraint has no valid continuation.
    func allowedTokens() throws -> [Int]?

    /// Advances the constraint past the selected token.
    func accept(_ token: Int) throws

    /// Whether exact speculative decoding may verify drafts row-by-row with this constraint.
    var allowsSpeculation: Bool { get }

    /// Whether the constraint currently permits any token (`allowedTokens() == nil`)
    /// up to `speculationBoundaryToken`.
    var isUnconstrainedNow: Bool { get }

    /// Token ID at which an unconstrained speculative round must stop so subsequent
    /// tokens can be verified with full logits.
    var speculationBoundaryToken: Int? { get }

    /// Writes ascending selectable token IDs into `scratch` (keeping capacity),
    /// returning `false` when unconstrained (`allowedTokens() == nil`).
    func writeAllowedTokens(into scratch: inout [Int]) throws -> Bool

    /// Returns an independent state clone for tentative multi-row speculative verification.
    func makeSpeculativeProbe() -> (any QWTokenConstraint)?

    /// Proposes up to `maxCount` canonical structural draft tokens without mutating `self`.
    func proposeCanonicalDrafts(maxCount: Int) -> [Int]
}

public enum QWConstrainedSpeculationGate {
    /// Hermetic A/B gate (`QW_CONSTRAINED_SPEC=0` restores serial constrained fallback).
    public static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["QW_CONSTRAINED_SPEC"] != "0"
    }
}

extension QWTokenConstraint {
    public var allowsSpeculation: Bool { false }
    public var isUnconstrainedNow: Bool { false }
    public var speculationBoundaryToken: Int? { nil }

    public func writeAllowedTokens(into scratch: inout [Int]) throws -> Bool {
        guard let allowed = try allowedTokens() else { return false }
        scratch.removeAll(keepingCapacity: true)
        scratch.append(contentsOf: allowed)
        return true
    }

    public func makeSpeculativeProbe() -> (any QWTokenConstraint)? { nil }
    public func proposeCanonicalDrafts(maxCount: Int) -> [Int] { [] }
}

/// Reusable preallocated scratch buffer and bitset for zero-allocation token masking.
public final class QWTokenMaskScratch: @unchecked Sendable {
    private var words: [UInt64] = []
    private var dirtyWords: [Int] = []
    private var activeVocabSize: Int = 0

    public init(vocabularyCapacity: Int = 0) {
        ensureVocabularyCapacity(vocabularyCapacity)
    }

    /// O(log N) membership test on ascending de-duplicated `sorted` token IDs.
    @inline(__always)
    public static func sortedTokens(_ sorted: [Int], contains token: Int) -> Bool {
        var lo = 0
        var hi = sorted.count - 1
        while lo <= hi {
            let mid = lo + ((hi - lo) >> 1)
            let val = sorted[mid]
            if val == token { return true }
            if val < token { lo = mid + 1 } else { hi = mid - 1 }
        }
        return false
    }

    public func ensureVocabularyCapacity(_ vocabularySize: Int) {
        let clamped = max(0, vocabularySize)
        let neededWords = (clamped + 63) >> 6
        if words.count < neededWords {
            words.append(contentsOf: repeatElement(0, count: neededWords - words.count))
        }
        activeVocabSize = clamped
    }

    /// Clears previously set bits in O(dirtyWords.count) without scanning the full bitset.
    public func clearBitset() {
        for wordIdx in dirtyWords {
            words[wordIdx] = 0
        }
        dirtyWords.removeAll(keepingCapacity: true)
    }

    /// Populates the bitset from a `Set<Int>` or `[Int]` repetition history in O(history.count).
    public func markTokens<S: Sequence>(_ tokens: S, vocabularySize: Int) throws where S.Element == Int {
        clearBitset()
        ensureVocabularyCapacity(vocabularySize)
        for token in tokens {
            guard token >= 0, token < vocabularySize else {
                clearBitset()
                throw QWTokenConstraintError.rejectedToken(token)
            }
            markTokenUnchecked(token)
        }
    }

    @inline(__always)
    func markTokenUnchecked(_ token: Int) {
        let wordIdx = token >> 6
        let mask = UInt64(1) &<< UInt64(token & 63)
        if words[wordIdx] == 0 {
            dirtyWords.append(wordIdx)
        }
        words[wordIdx] |= mask
    }

    @inline(__always)
    public func contains(_ token: Int) -> Bool {
        guard token >= 0, token < activeVocabSize else { return false }
        return (words[token >> 6] & (UInt64(1) &<< UInt64(token & 63))) != 0
    }

    @inline(__always)
    func word(at index: Int) -> UInt64 { words[index] }

    var activeWordCount: Int { (activeVocabSize + 63) >> 6 }
}

public enum QWTokenConstraintError: Error, CustomStringConvertible, Equatable {
    case emptySupport
    case rejectedToken(Int)

    public var description: String {
        switch self {
        case .emptySupport:
            return "token constraint has no valid continuation"
        case .rejectedToken(let token):
            return "token constraint rejected token \(token)"
        }
    }
}
