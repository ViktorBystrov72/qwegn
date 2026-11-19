// QWEnginePrefixStore.swift — Driver prefix store backed by the unified QWSlotPinnedLRUStore.
import Foundation
import QWCore

final class QWEnginePrefixStore<Value: Sendable> {
    struct Key: Hashable, Sendable {
        let tokens: [Int]
        let mode: String
    }
    struct Hit {
        let value: Value
        let tokenCount: Int
    }

    static var defaultMaxSlots: Int { QWSlotPinnedLRUStore<Key, Value>.defaultMaxSlots }
    let vocab: Int
    private let core: QWSlotPinnedLRUStore<Key, Value>
    var onEvict: ((Key, Value, Bool) -> Void)?

    init(byteLimit: Int, entryLimit: Int = 16, vocab: Int,
         maxSlots: Int = QWEnginePrefixStore.defaultMaxSlots) throws {
        guard byteLimit > 0, entryLimit > 0, vocab > 0 else {
            throw QWEngineDriverError.state("prefix store limits must be positive")
        }
        self.vocab = vocab
        let store = QWSlotPinnedLRUStore<Key, Value>(
            byteLimit: byteLimit,
            entryLimit: entryLimit,
            maxSlots: maxSlots,
            rootPrefixTokenCount: 64
        )
        self.core = store
        store.onEvict = { [weak self] key, entry in
            self?.onEvict?(key, entry.value, entry.allowsSuffix)
        }
    }

    var maxSlots: Int { core.maxSlots }
    var byteLimit: Int { core.byteLimit }
    var entryLimit: Int { core.entryLimit }
    var count: Int { core.count }
    var chargedBytes: Int { core.chargedBytes }
    var evictions: Int { core.evictions }
    var hasSharedBase: Bool { core.hasSharedBase }

    static func baseMode(of mode: String) -> String {
        guard let range = mode.range(of: "|salt=") else { return mode }
        return String(mode[..<range.lowerBound])
    }

    func sharedBasePrefix(matching tokens: [Int], mode: String) -> Hit? {
        let base = Self.baseMode(of: mode)
        var best: Key?
        for key in core.sharedBaseKeys where (key.mode == mode || key.mode == base) &&
            key.tokens.count <= tokens.count && key.tokens.count > (best?.tokens.count ?? 0) {
            if core.entries[key] != nil && tokens.starts(with: key.tokens) {
                best = key
            }
        }
        guard let best, let entry = core.value(for: best, touchOnHit: false) else { return nil }
        return Hit(value: entry.value, tokenCount: best.tokens.count)
    }

    func sharedBranchPrefixLength(for tokens: [Int]) -> Int? {
        core.sharedBranchPrefixLength(for: tokens.map(Int64.init))
    }

    func ensureByteLimit(atLeast minimumBytes: Int) {
        core.ensureByteLimit(atLeast: minimumBytes)
    }

    func removeSharedBaseEntries() {
        core.removeSharedBaseEntries()
    }

    func lookup(tokens: [Int], mode: String) throws -> Hit? {
        try validate(tokens, mode: mode)
        let exact = Key(tokens: tokens, mode: mode)
        let base = Self.baseMode(of: mode)
        let saltedBase = base != mode ? Key(tokens: tokens, mode: base) : nil
        let queryRootHash: UInt64? = tokens.count >= 64
            ? QWPrefixHash.fnv1a64(tokens: Array(tokens.prefix(64)), mode: mode)
            : nil
        let hit = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: core,
            exactKey: exact,
            saltedBaseKey: saltedBase,
            tokenCount: tokens.count,
            queryRootHash: queryRootHash,
            allowExact: true,
            allowPrefix: true,
            sharedBaseOnly: false,
            touchOnHit: true,
            isCandidate: { key, _, _ in
                key.mode == mode || (base != mode && key.mode == base && core.isSharedBaseKey(key))
            },
            matchesPrefix: { key, _, _ in
                tokens.starts(with: key.tokens)
            }
        )
        return hit.map { Hit(value: $0.value, tokenCount: $0.matchedTokens) }
    }

    /// Oversized valid entries are not retained. The caller records that result.
    @discardableResult
    func insert(tokens: [Int], mode: String, value: Value, payloadBytes: Int,
                allowsSuffix: Bool = false, isSharedBase: Bool = false) throws -> Bool {
        let keyBytes = try keyBytes(tokens: tokens, mode: mode)
        guard payloadBytes >= 0 else {
            throw QWEngineDriverError.state("invalid prefix payload charge")
        }
        guard keyBytes <= byteLimit, payloadBytes <= byteLimit - keyBytes else { return false }
        let charge = payloadBytes + keyBytes
        let key = Key(tokens: tokens, mode: mode)
        let history64 = tokens.map(Int64.init)
        let rootHash = QWPrefixHash.fnv1a64(tokens: Array(tokens.prefix(64)), mode: mode)
        return QWTwoTierPrefixCache.storeWithSlotPinning(
            in: core,
            key: key,
            value: value,
            charge: charge,
            tokenCount: tokens.count,
            allowsSuffix: allowsSuffix,
            rootHash: rootHash,
            history: history64,
            isSharedBase: isSharedBase
        ) { cand, _ in
            cand.mode == key.mode && key.tokens.starts(with: cand.tokens)
        }
    }

    func isProtectedLastKey(_ key: Key, excludingSlot: Int? = nil) -> Bool {
        core.isProtectedLastKey(key, excludingSlot: excludingSlot)
    }

    @discardableResult
    func evictOldest(excludingSlot: Int? = nil) -> Bool {
        core.evictOldest(excludingSlot: excludingSlot)
    }

    func keyBytes(tokens: [Int], mode: String) throws -> Int {
        try validate(tokens, mode: mode)
        guard tokens.count <= Int.max / MemoryLayout<Int>.stride else {
            throw QWEngineDriverError.state("prefix key byte count overflows")
        }
        let tokenBytes = tokens.count * MemoryLayout<Int>.stride
        let modeBytes = mode.utf8.count
        guard modeBytes <= Int.max - tokenBytes else {
            throw QWEngineDriverError.state("prefix policy byte count overflows")
        }
        return tokenBytes + modeBytes
    }

    private func validate(_ tokens: [Int], mode: String) throws {
        guard !tokens.isEmpty, !mode.isEmpty,
              tokens.allSatisfy({ (0..<vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid prefix token or policy key")
        }
    }
}

struct QWEngineCachedPrefix: Sendable {
    let prefix: QWEngineSharedPrefix
    let nextToken: Int
    let logits: Data
    var memoryBytes: Int { prefix.memoryBytes + logits.count }
}
