// QWThinkingBudgetConstraint.swift — Token constraint capping thinking tokens
// and forcing </think> before max_tokens exhaustion while composing cleanly
// with an inner structured-output or tool-call constraint.

import Foundation
import QWHead

public final class QWThinkingBudgetConstraint: QWTokenConstraint, @unchecked Sendable {
    private static let endThinkBytes = Array("</think>".utf8)
    private static let ernieEndThinkBytes = Array("</ernie:think>".utf8)

    private let lock = NSLock()
    private let maxThinkingTokens: Int
    private let endThinkTokenId: Int
    private let vocabularySize: Int
    private let hasValidEndThink: Bool
    private let forwardEndThinkToInner: Bool
    private let tokenBytes: [[UInt8]?]?
    private let inner: (any QWTokenConstraint)?

    private var inThinking: Bool = true
    private var thinkingTokens: Int = 0
    private var recentUTF8: [UInt8] = []

    public init(
        maxThinkingTokens: Int,
        endThinkTokenId: Int,
        vocabularySize: Int,
        table: QWJSONTokenTable? = nil,
        tokenBytes: [[UInt8]?]? = nil,
        inner: (any QWTokenConstraint)? = nil,
        forwardEndThinkToInner: Bool = true
    ) {
        self.maxThinkingTokens = max(0, maxThinkingTokens)
        self.endThinkTokenId = endThinkTokenId
        self.vocabularySize = vocabularySize
        self.hasValidEndThink = endThinkTokenId >= 0 && endThinkTokenId < vocabularySize
        self.forwardEndThinkToInner = forwardEndThinkToInner
        self.tokenBytes = tokenBytes ?? table?.bytes
        self.inner = inner
    }

    public var isInThinking: Bool {
        lock.withLock { inThinking }
    }

    public var thinkingTokenCount: Int {
        lock.withLock { thinkingTokens }
    }

    public var allowsSpeculation: Bool {
        let shouldBlock: Bool = lock.withLock {
            inThinking && thinkingTokens >= maxThinkingTokens && hasValidEndThink
        }
        if shouldBlock { return false }
        return inner?.allowsSpeculation ?? true
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

    public func allowedTokens() throws -> [Int]? {
        enum Decision { case forceEndThink, unconstrained, forwardInner }
        let decision: Decision = lock.withLock {
            guard inThinking else { return .forwardInner }
            if thinkingTokens >= maxThinkingTokens && hasValidEndThink {
                return .forceEndThink
            }
            return .unconstrained
        }
        switch decision {
        case .forceEndThink:
            return [endThinkTokenId]
        case .unconstrained:
            return nil
        case .forwardInner:
            return try inner?.allowedTokens()
        }
    }

    public func accept(_ token: Int) throws {
        enum Action { case none, forward(Int) }
        let action: Action = lock.withLock {
            guard inThinking else {
                return .forward(token)
            }
            if hasValidEndThink && token == endThinkTokenId {
                inThinking = false
                recentUTF8.removeAll(keepingCapacity: false)
                return forwardEndThinkToInner ? .forward(token) : .none
            }
            thinkingTokens += 1
            if let tokenBytes, token >= 0, token < tokenBytes.count, token < vocabularySize,
               let bytes = tokenBytes[token], !bytes.isEmpty {
                recentUTF8.append(contentsOf: bytes)
                if recentUTF8.count > 24 {
                    recentUTF8.removeFirst(recentUTF8.count - 24)
                }
                if Self.containsSubslice(recentUTF8, Self.endThinkBytes)
                    || Self.containsSubslice(recentUTF8, Self.ernieEndThinkBytes) {
                    inThinking = false
                    recentUTF8.removeAll(keepingCapacity: false)
                    return (hasValidEndThink && forwardEndThinkToInner) ? .forward(endThinkTokenId) : .none
                }
            }
            return .none
        }
        switch action {
        case .none:
            return
        case .forward(let forwarded):
            try inner?.accept(forwarded)
        }
    }

    private static func containsSubslice(_ haystack: [UInt8], _ needle: [UInt8]) -> Bool {
        guard !needle.isEmpty, haystack.count >= needle.count else { return false }
        let limit = haystack.count - needle.count
        for i in 0...limit where haystack[i] == needle[0] {
            var match = true
            for j in 1..<needle.count where haystack[i + j] != needle[j] {
                match = false
                break
            }
            if match { return true }
        }
        return false
    }
}
