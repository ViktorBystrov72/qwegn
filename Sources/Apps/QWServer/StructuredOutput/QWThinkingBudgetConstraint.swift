// QWThinkingBudgetConstraint.swift — Token constraint capping thinking tokens
// and forcing </think> before max_tokens exhaustion while composing cleanly
// with an inner structured-output or tool-call constraint.

import Foundation
import QWHead

public final class QWThinkingBudgetConstraint: QWTokenConstraint, @unchecked Sendable {
    private static let endThinkBytes = Array("</think>".utf8)

    private let lock = NSLock()
    private let maxThinkingTokens: Int
    private let endThinkTokenId: Int
    private let forcedEndThinkAllowed: [Int]
    private let vocabularySize: Int
    private let hasValidEndThink: Bool
    private let forwardEndThinkToInner: Bool
    private let tokenBytes: [[UInt8]?]?
    private let endThinkCloseBytesList: [[UInt8]]
    private let recentCap: Int
    private let inner: (any QWTokenConstraint)?

    private var inThinking: Bool = true
    private var thinkingTokens: Int = 0
    private var recentUTF8: [UInt8] = []

    private enum AcceptAction {
        case none
        case single(Int)
        case multiple([Int])
    }

    public init(
        maxThinkingTokens: Int,
        endThinkTokenId: Int,
        endThinkCloseTag: String? = nil,
        vocabularySize: Int,
        table: QWJSONTokenTable? = nil,
        tokenBytes: [[UInt8]?]? = nil,
        inner: (any QWTokenConstraint)? = nil,
        forwardEndThinkToInner: Bool = true
    ) {
        self.maxThinkingTokens = max(0, maxThinkingTokens)
        self.endThinkTokenId = endThinkTokenId
        self.forcedEndThinkAllowed = [endThinkTokenId]
        self.vocabularySize = vocabularySize
        let validId = endThinkTokenId >= 0 && endThinkTokenId < vocabularySize
        self.hasValidEndThink = validId
        self.forwardEndThinkToInner = forwardEndThinkToInner
        let resolvedBytes = tokenBytes ?? table?.bytes
        self.tokenBytes = resolvedBytes
        var targets: [[UInt8]] = [Self.endThinkBytes]
        if let tag = endThinkCloseTag, !tag.isEmpty {
            let b = Array(tag.utf8)
            if !targets.contains(b) { targets.append(b) }
        } else if validId, let tb = resolvedBytes, endThinkTokenId < tb.count, let bytes = tb[endThinkTokenId],
                  let s = String(bytes: bytes, encoding: .utf8), s.hasPrefix("</"), s.hasSuffix(">"),
                  !targets.contains(bytes) {
            targets.append(bytes)
        }
        self.endThinkCloseBytesList = targets
        let cap = max(64, targets.map(\.count).max() ?? 8)
        self.recentCap = cap
        self.inner = inner
        self.recentUTF8.reserveCapacity(cap * 2)
    }

    private init(
        cloning other: QWThinkingBudgetConstraint,
        inThinking: Bool,
        thinkingTokens: Int,
        recentUTF8: [UInt8],
        innerClone: (any QWTokenConstraint)?
    ) {
        self.maxThinkingTokens = other.maxThinkingTokens
        self.endThinkTokenId = other.endThinkTokenId
        self.forcedEndThinkAllowed = other.forcedEndThinkAllowed
        self.vocabularySize = other.vocabularySize
        self.hasValidEndThink = other.hasValidEndThink
        self.forwardEndThinkToInner = other.forwardEndThinkToInner
        self.tokenBytes = other.tokenBytes
        self.endThinkCloseBytesList = other.endThinkCloseBytesList
        self.recentCap = other.recentCap
        self.inner = innerClone
        self.inThinking = inThinking
        self.thinkingTokens = thinkingTokens
        self.recentUTF8 = recentUTF8
    }

    public var allowsSpeculation: Bool {
        if QWConstrainedSpeculationGate.isEnabled {
            return inner?.allowsSpeculation ?? true
        }
        let shouldBlock: Bool = lock.withLock {
            inThinking && thinkingTokens >= maxThinkingTokens && hasValidEndThink
        }
        if shouldBlock { return false }
        return inner?.allowsSpeculation ?? true
    }

    public func makeSpeculativeProbe() -> (any QWTokenConstraint)? {
        let snap = lock.withLock { (inThinking, thinkingTokens, recentUTF8) }
        let innerClone: (any QWTokenConstraint)?
        if let inner {
            guard let cloned = inner.makeSpeculativeProbe() else { return nil }
            innerClone = cloned
        } else {
            innerClone = nil
        }
        return QWThinkingBudgetConstraint(
            cloning: self,
            inThinking: snap.0,
            thinkingTokens: snap.1,
            recentUTF8: snap.2,
            innerClone: innerClone
        )
    }

    public func proposeCanonicalDrafts(maxCount: Int) -> [Int] {
        guard QWConstrainedSpeculationGate.isEnabled, maxCount > 0 else { return [] }
        let decision = currentAllowedDecision()
        switch decision {
        case .unconstrained:
            return []
        case .forwardInner:
            return inner?.proposeCanonicalDrafts(maxCount: maxCount) ?? []
        case .forceEndThink:
            guard let probe = makeSpeculativeProbe(),
                  (try? probe.accept(endThinkTokenId)) != nil else {
                return [endThinkTokenId]
            }
            var out = [endThinkTokenId]
            if maxCount > 1 {
                out.append(contentsOf: probe.proposeCanonicalDrafts(maxCount: maxCount - 1))
            }
            return out
        }
    }

    public var isUnconstrainedNow: Bool {
        let state: (inThinking: Bool, allowSpec: Bool) = lock.withLock {
            if !inThinking { return (false, true) }
            if !hasValidEndThink { return (true, false) }
            return (true, (maxThinkingTokens - thinkingTokens) > 16)
        }
        if state.inThinking { return state.allowSpec }
        return inner?.isUnconstrainedNow ?? true
    }

    public var speculationBoundaryToken: Int? {
        let thinkingBoundary: (active: Bool, token: Int?) = lock.withLock {
            if inThinking {
                return (true, hasValidEndThink ? endThinkTokenId : nil)
            }
            return (false, nil)
        }
        if thinkingBoundary.active {
            return thinkingBoundary.token ?? inner?.speculationBoundaryToken
        }
        return inner?.speculationBoundaryToken
    }

    private enum AllowedDecision { case forceEndThink, unconstrained, forwardInner }

    private func currentAllowedDecision() -> AllowedDecision {
        lock.withLock {
            guard inThinking else { return .forwardInner }
            if thinkingTokens >= maxThinkingTokens && hasValidEndThink {
                return .forceEndThink
            }
            return .unconstrained
        }
    }

    public func allowedTokens() throws -> [Int]? {
        switch currentAllowedDecision() {
        case .forceEndThink:
            return forcedEndThinkAllowed
        case .unconstrained:
            return nil
        case .forwardInner:
            return try inner?.allowedTokens()
        }
    }

    public func writeAllowedTokens(into scratch: inout [Int]) throws -> Bool {
        switch currentAllowedDecision() {
        case .forceEndThink:
            scratch.removeAll(keepingCapacity: true)
            scratch.append(endThinkTokenId)
            return true
        case .unconstrained:
            return false
        case .forwardInner:
            return try inner?.writeAllowedTokens(into: &scratch) ?? false
        }
    }

    public func accept(_ token: Int) throws {
        let action: AcceptAction = lock.withLock {
            guard inThinking else {
                return .single(token)
            }
            if hasValidEndThink && token == endThinkTokenId {
                inThinking = false
                recentUTF8.removeAll(keepingCapacity: false)
                return forwardEndThinkToInner ? .single(token) : .none
            }
            thinkingTokens += 1
            if let tokenBytes, token >= 0, token < tokenBytes.count, token < vocabularySize,
               let bytes = tokenBytes[token], !bytes.isEmpty {
                recentUTF8.append(contentsOf: bytes)
                if recentUTF8.count > recentCap {
                    let drop = recentUTF8.count - recentCap
                    recentUTF8.withUnsafeMutableBufferPointer { buf in
                        guard let base = buf.baseAddress else { return }
                        memmove(base, base.advanced(by: drop), recentCap)
                    }
                    recentUTF8.removeLast(drop)
                }
                var matchEnd: Int?
                for target in endThinkCloseBytesList {
                    if let end = Self.endOfSubslice(recentUTF8, target), matchEnd.map({ end < $0 }) ?? true {
                        matchEnd = end
                    }
                }
                if let matchEnd {
                    inThinking = false
                    let trailingBytes = Array(recentUTF8[matchEnd...])
                    recentUTF8.removeAll(keepingCapacity: false)
                    var tokens: [Int] = []
                    if hasValidEndThink && forwardEndThinkToInner {
                        tokens.append(endThinkTokenId)
                    }
                    if !trailingBytes.isEmpty {
                        tokens.append(contentsOf: decomposeBytes(trailingBytes, using: tokenBytes))
                    }
                    if tokens.isEmpty { return .none }
                    if tokens.count == 1 { return .single(tokens[0]) }
                    return .multiple(tokens)
                }
            }
            return .none
        }
        switch action {
        case .none:
            break
        case .single(let tok):
            try inner?.accept(tok)
        case .multiple(let toks):
            for tok in toks {
                try inner?.accept(tok)
            }
        }
    }

    private func decomposeBytes(_ target: [UInt8], using table: [[UInt8]?]) -> [Int] {
        var out: [Int] = []
        var offset = 0
        while offset < target.count {
            var bestLen = 0
            var bestToken = -1
            let remaining = target.count - offset
            for tok in 0..<min(vocabularySize, table.count) where tok != endThinkTokenId {
                guard let b = table[tok], !b.isEmpty, b.count <= remaining, b.count > bestLen else { continue }
                if target[offset..<(offset + b.count)].elementsEqual(b) {
                    bestLen = b.count
                    bestToken = tok
                    if bestLen == remaining { break }
                }
            }
            guard bestToken >= 0, bestLen > 0 else { break }
            out.append(bestToken)
            offset += bestLen
        }
        return out
    }

    private static func endOfSubslice(_ haystack: [UInt8], _ needle: [UInt8]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        let limit = haystack.count - needle.count
        for i in 0...limit where haystack[i] == needle[0] {
            var match = true
            for j in 1..<needle.count where haystack[i + j] != needle[j] {
                match = false
                break
            }
            if match { return i + needle.count }
        }
        return nil
    }
}
