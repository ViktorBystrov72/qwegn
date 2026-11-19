// FNPrefixWarmStore.swift — Thin adapter around QWSlotPinnedLRUStore for FlashNext prefix cache entries.
// Unifies the warm RAM tier with QWEngine's prefix cache store.

import Foundation
import QWCore

public typealias FNPrefixCacheTier = QWPrefixCacheTier
public typealias FNPrefixCacheTiers = QWPrefixCacheTiers

/// The prefix cache's warm tier: backed by the unified `QWSlotPinnedLRUStore`.
public final class FNPrefixWarmStore: @unchecked Sendable {
    public static let defaultMaxSlots = QWSlotPinnedLRUStore<String, Data>.defaultMaxSlots
    public static var maxSlots: Int { defaultMaxSlots }
    public static let rootPrefixTokenCount = QWSlotPinnedLRUStore<String, Data>.defaultRootPrefixTokenCount

    let core: QWSlotPinnedLRUStore<String, Data>
    public var onEvict: ((Int, Int) -> Void)?

    public init(byteLimit: Int, maxSlots: Int = FNPrefixWarmStore.defaultMaxSlots) {
        let store = QWSlotPinnedLRUStore<String, Data>(
            byteLimit: byteLimit,
            maxSlots: maxSlots,
            rootPrefixTokenCount: Self.rootPrefixTokenCount
        )
        self.core = store
        store.onEvict = { [weak self] _, entry in
            guard entry.charge > 0 else { return }
            self?.onEvict?(entry.tokenCount, entry.charge)
        }
    }

    public var maxSlots: Int { core.maxSlots }
    public var byteLimit: Int { core.byteLimit }
    public var isEnabled: Bool { core.isEnabled }
    public var state: (entries: Int, bytes: Int) { core.state }

    public func setMaxSlots(_ count: Int) {
        core.setMaxSlots(count)
    }

    public func ensureByteLimit(atLeast minimumBytes: Int) {
        core.ensureByteLimit(atLeast: minimumBytes)
    }

    public func removeSharedBaseEntries() {
        core.removeSharedBaseEntries()
    }

    public var hasSharedBase: Bool { core.hasSharedBase }

    public func canHold(bytes: Int) -> Bool {
        core.canHoldIncludingShared(bytes: bytes)
    }

    public func isSharedBaseEntry(_ name: String) -> Bool {
        core.isSharedBaseKey(name)
    }

    public static func rootHash(history: [Int64], kvMode: FNKVStorageMode, hasMTP: Bool,
                                dflashMarker: String? = nil) -> UInt64 {
        let n = min(rootPrefixTokenCount, history.count)
        return FNGPUEngine.fnPrefixHash(tokens: history.prefix(n), kvMode: kvMode,
                                        hasMTP: hasMTP, dflashMarker: dflashMarker)
    }

    public func sharedBranchPrefixLength(for history: [Int64]) -> Int? {
        core.sharedBranchPrefixLength(for: history)
    }

    private static func matchesEntry(
        name: String, count: Int, history: [Int64], kvMode: FNKVStorageMode, hasMTP: Bool, dflashMarker: String?
    ) -> Bool {
        let expected = FNGPUEngine.fnPrefixHash(tokens: history.prefix(count), kvMode: kvMode,
                                                hasMTP: hasMTP, dflashMarker: dflashMarker)
        return Self.hash(in: name) == String(format: "%016llx", expected)
    }

    public func resolveSlot(rootHash: UInt64, history: [Int64], kvMode: FNKVStorageMode,
                            hasMTP: Bool, dflashMarker: String? = nil) -> Int {
        core.resolveSlot(rootHash: rootHash, history: history) { name, count in
            Self.matchesEntry(name: name, count: count, history: history,
                              kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker)
        }
    }

    public func prepareSlotForStore(
        slot: Int,
        rootHash: UInt64,
        history: [Int64],
        incomingBytes: Int,
        kvMode: FNKVStorageMode,
        hasMTP: Bool,
        dflashMarker: String? = nil
    ) {
        core.prepareSlotForStore(slot: slot, history: history, incomingBytes: incomingBytes) { name, count in
            Self.matchesEntry(name: name, count: count, history: history,
                              kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker)
        }
    }

    public func makeRoom(for incomingBytes: Int, activeSlot: Int? = nil) {
        core.makeRoom(for: incomingBytes, activeSlot: activeSlot)
    }

    public func store(name: String, blob: Data, slot: Int? = nil, rootHash: UInt64? = nil,
                      history: [Int64]? = nil, isSharedBase: Bool = false) {
        guard core.isEnabled else { return }
        core.forget(name)
        guard blob.count <= core.byteLimit else { return }
        let tokCount = Self.tokenCount(in: name) ?? (history?.count ?? 0)
        _ = core.store(
            key: name,
            value: blob,
            charge: blob.count,
            tokenCount: tokCount,
            allowsSuffix: true,
            slot: slot,
            rootHash: rootHash,
            history: history,
            isSharedBase: isSharedBase
        )
    }

    public func storeWithSlotPinning(
        name: String,
        blob: Data,
        history: [Int64],
        kvMode: FNKVStorageMode,
        hasMTP: Bool,
        dflashMarker: String? = nil,
        isSharedBase: Bool = false
    ) -> Bool {
        guard core.isEnabled, blob.count <= core.byteLimit else { return false }
        let rHash = Self.rootHash(history: history, kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker)
        let tokCount = Self.tokenCount(in: name) ?? history.count
        return QWTwoTierPrefixCache.storeWithSlotPinning(
            in: core,
            key: name,
            value: blob,
            charge: blob.count,
            tokenCount: tokCount,
            allowsSuffix: true,
            rootHash: rHash,
            history: history,
            isSharedBase: isSharedBase
        ) { candName, candCount in
            Self.matchesEntry(
                name: candName, count: candCount, history: history,
                kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker
            )
        }
    }

    public func blob(name: String) -> Data? {
        core.value(for: name, touchOnHit: true)?.value
    }

    public func contains(name: String) -> Bool {
        core.contains(name)
    }

    public func longestMatch(history: [Int64], kvMode: FNKVStorageMode,
                             hasMTP: Bool, dflashMarker: String? = nil) -> (name: String, tokens: Int)? {
        let queryRootHash: UInt64? = history.count >= Self.rootPrefixTokenCount
            ? Self.rootHash(history: history, kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker)
            : nil
        var matchedName: String?
        let hit = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: core,
            exactKey: "",
            saltedBaseKey: nil,
            tokenCount: history.count,
            queryRootHash: queryRootHash,
            allowExact: false,
            allowPrefix: true,
            sharedBaseOnly: false,
            touchOnHit: false,
            isCandidate: { _, _, _ in true },
            matchesPrefix: { name, count, _ in
                let ok = Self.matchesEntry(
                    name: name, count: count, history: history,
                    kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker
                )
                if ok { matchedName = name }
                return ok
            }
        )
        guard let hit, let name = matchedName else { return nil }
        core.touchSlot(for: name)
        return (name, hit.matchedTokens)
    }

    public func longestSharedBaseMatch(history: [Int64], kvMode: FNKVStorageMode,
                                       hasMTP: Bool, dflashMarker: String? = nil) -> (name: String, tokens: Int)? {
        var matchedName: String?
        let hit = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: core,
            exactKey: "",
            saltedBaseKey: nil,
            tokenCount: history.count,
            queryRootHash: nil,
            allowExact: false,
            allowPrefix: true,
            sharedBaseOnly: true,
            touchOnHit: false,
            isCandidate: { _, _, _ in true },
            matchesPrefix: { name, count, _ in
                let ok = Self.matchesEntry(
                    name: name, count: count, history: history,
                    kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker
                )
                if ok { matchedName = name }
                return ok
            }
        )
        guard let hit, let name = matchedName else { return nil }
        return (name, hit.matchedTokens)
    }

    public func lookupWarmHit(
        exactName: String,
        history: [Int64],
        rawHistory: [Int64],
        kvMode: FNKVStorageMode,
        hasMTP: Bool,
        dflashMarker: String? = nil,
        allowExact: Bool
    ) -> QWTwoTierPrefixCache.WarmLookupHit<Data>? {
        let queryRootHash: UInt64? = history.count >= Self.rootPrefixTokenCount
            ? Self.rootHash(history: history, kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker)
            : nil
        let isSalted = rawHistory.first != history.first
        return QWTwoTierPrefixCache.lookupWarmPrefix(
            in: core,
            exactKey: exactName,
            saltedBaseKey: isSalted ? exactName : nil,
            tokenCount: history.count,
            queryRootHash: queryRootHash,
            allowExact: allowExact,
            allowPrefix: true,
            sharedBaseOnly: false,
            touchOnHit: true,
            isCandidate: { _, _, _ in true },
            matchesPrefix: { name, count, sharedFallback in
                let toks = sharedFallback ? rawHistory : history
                return Self.matchesEntry(
                    name: name, count: count, history: toks,
                    kvMode: kvMode, hasMTP: hasMTP, dflashMarker: dflashMarker
                )
            }
        )
    }

    static func tokenCount(in name: String) -> Int? {
        let parts = entryParts(name)
        guard parts.count == 4 else { return nil }
        return Int(parts[3])
    }

    static func hash(in name: String) -> String? {
        let parts = entryParts(name)
        guard parts.count == 4 else { return nil }
        return String(parts[2])
    }

    private static func entryParts(_ name: String) -> [Substring] {
        var stem = Substring(name)
        if let dot = stem.lastIndex(of: ".") { stem = stem[stem.startIndex..<dot] }
        return stem.split(separator: "_")
    }
}
