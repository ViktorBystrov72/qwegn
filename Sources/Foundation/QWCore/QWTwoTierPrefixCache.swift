// QWTwoTierPrefixCache.swift — Shared L0 two-tier (Warm RAM + Cold SSD) prefix cache control seams.

import Dispatch
import Foundation

/// Bounded asynchronous cold-SSD spill queue controller with explicit byte/count caps.
public final class QWAsyncColdWriteController: @unchecked Sendable {
    private static let maxInFlightCount = 4
    private static let maxInFlightBytes = 512 << 20
    private let queue = DispatchQueue(label: "ai.qwegn.engine.prefix-cold-write", qos: .utility)
    private let group = DispatchGroup()
    private let lock = NSLock()
    private var inFlightCount = 0
    private var inFlightBytes = 0
    private var closed = false

    public init() {}

    public func closeAndDrain() {
        lock.withLock { closed = true }
        group.wait()
    }

    public func reopenAndDrain() {
        lock.withLock { closed = true }
        group.wait()
        lock.withLock { closed = false }
    }

    public func waitUntilIdle() {
        group.wait()
    }

    public func reserve(bytes: Int) -> Bool {
        lock.withLock {
            guard !closed, bytes >= 0,
                  inFlightCount < Self.maxInFlightCount,
                  inFlightBytes + bytes <= Self.maxInFlightBytes else {
                return false
            }
            inFlightCount += 1
            inFlightBytes += bytes
            group.enter()
            return true
        }
    }

    public func cancelReservation(bytes: Int) {
        lock.withLock {
            inFlightCount -= 1
            inFlightBytes -= bytes
        }
        group.leave()
    }

    public func submitReserved(bytes: Int, work: @escaping @Sendable () -> Void) {
        queue.async { [self] in
            defer { cancelReservation(bytes: bytes) }
            work()
        }
    }
}

/// Thread-safe cold-SSD LRU and byte/entry budget index shared across engines.
public final class QWColdCacheIndex<Metadata: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    public let ssdByteLimit: Int
    public let entryLimit: Int
    private var entries: [String: Metadata] = [:]
    private var entryFiles: [String: URL] = [:]
    private var entryBytes: [String: Int] = [:]
    private var lru: [String] = []
    private var _chargedBytes: Int = 0
    private var _evictions: Int = 0

    public init(ssdByteLimit: Int, entryLimit: Int = Int.max) {
        self.ssdByteLimit = max(0, ssdByteLimit)
        self.entryLimit = max(1, entryLimit)
    }

    public var entryCount: Int { lock.withLock { entries.count } }
    public var chargedBytes: Int { lock.withLock { _chargedBytes } }
    public var evictions: Int { lock.withLock { _evictions } }

    public func canAccept(bytes: Int) -> Bool {
        ssdByteLimit > 0 && bytes >= 0 && bytes <= ssdByteLimit
    }

    /// Loads discovered on-disk items (already sorted oldest-first) within budget.
    public func bootstrap(
        items: [(key: String, metadata: Metadata, fileURL: URL, bytes: Int)],
        onEvict: ((URL) -> Void)? = nil
    ) {
        lock.withLock {
            for item in items where canAccept(bytes: item.bytes) {
                recordLocked(
                    key: item.key, metadata: item.metadata, fileURL: item.fileURL,
                    bytes: item.bytes, onEvict: onEvict
                )
            }
        }
    }

    /// Records `key` under lock, evicting oldest entries first to honor `ssdByteLimit` and `entryLimit`.
    @discardableResult
    public func record(
        key: String,
        metadata: Metadata,
        fileURL: URL,
        bytes: Int,
        beforeInstall: (() throws -> Void)? = nil,
        onEvict: ((URL) -> Void)? = nil
    ) throws -> Bool {
        try lock.withLock {
            guard canAccept(bytes: bytes) else { return false }
            let priorBytes = entryBytes[key] ?? 0
            let neededDelta = bytes - priorBytes
            while (entries[key] == nil && entries.count >= entryLimit)
                || (_chargedBytes + neededDelta > ssdByteLimit && !lru.isEmpty) {
                guard evictOldestLocked(onEvict: onEvict) else { break }
            }
            let currentPrior = entryBytes[key] ?? 0
            guard _chargedBytes - currentPrior + bytes <= ssdByteLimit,
                  entries[key] != nil || entries.count < entryLimit else {
                return false
            }
            try beforeInstall?()
            recordLocked(key: key, metadata: metadata, fileURL: fileURL, bytes: bytes, onEvict: onEvict)
            return true
        }
    }

    public func lookupExact(
        _ key: String,
        verify: (Metadata) -> Bool
    ) -> (metadata: Metadata, fileURL: URL)? {
        lock.withLock {
            guard let meta = entries[key], let url = entryFiles[key], verify(meta) else { return nil }
            touchLocked(key)
            return (meta, url)
        }
    }

    public func lookupBestPrefix(
        scoreMatch: (String, Metadata) -> Int?
    ) -> (key: String, metadata: Metadata, fileURL: URL, matchedTokens: Int)? {
        lock.withLock {
            var bestKey: String?
            var bestCount = 0
            for (key, meta) in entries {
                if let matched = scoreMatch(key, meta), matched > bestCount {
                    bestKey = key
                    bestCount = matched
                }
            }
            guard let resolvedKey = bestKey,
                  let meta = entries[resolvedKey],
                  let url = entryFiles[resolvedKey] else {
                return nil
            }
            touchLocked(resolvedKey)
            return (resolvedKey, meta, url, bestCount)
        }
    }

    @discardableResult
    public func evictOldest(onEvict: ((URL) -> Void)? = nil) -> Bool {
        lock.withLock { evictOldestLocked(onEvict: onEvict) }
    }

    public func purgeAll(onRemove: ((URL) -> Void)? = nil) {
        lock.withLock {
            for url in entryFiles.values { onRemove?(url) }
            entries.removeAll()
            entryFiles.removeAll()
            entryBytes.removeAll()
            lru.removeAll()
            _chargedBytes = 0
        }
    }

    private func recordLocked(
        key: String,
        metadata: Metadata,
        fileURL: URL,
        bytes: Int,
        onEvict: ((URL) -> Void)?
    ) {
        while (entries[key] == nil && entries.count >= entryLimit)
            || (_chargedBytes - (entryBytes[key] ?? 0) + bytes > ssdByteLimit && !lru.isEmpty) {
            guard evictOldestLocked(onEvict: onEvict) else { break }
        }
        if let prev = entryBytes[key] { _chargedBytes -= prev }
        entries[key] = metadata
        entryFiles[key] = fileURL
        entryBytes[key] = bytes
        _chargedBytes += bytes
        touchLocked(key)
    }

    private func touchLocked(_ key: String) {
        lru.removeAll { $0 == key }
        lru.append(key)
    }

    @discardableResult
    private func evictOldestLocked(onEvict: ((URL) -> Void)?) -> Bool {
        guard let oldest = lru.first else { return false }
        lru.removeFirst()
        if let file = entryFiles.removeValue(forKey: oldest) {
            onEvict?(file)
        }
        entries.removeValue(forKey: oldest)
        if let bytes = entryBytes.removeValue(forKey: oldest) {
            _chargedBytes -= bytes
        }
        _evictions += 1
        return true
    }
}

/// Shared warm+cold lookup, slot-pinned storage, spill gating, and budget accounting seam.
public enum QWTwoTierPrefixCache {
    public struct WarmLookupHit<Value: Sendable>: Sendable {
        public let value: Value
        public let matchedTokens: Int
        public let isUnsaltedSharedBase: Bool

        public init(value: Value, matchedTokens: Int, isUnsaltedSharedBase: Bool) {
            self.value = value
            self.matchedTokens = matchedTokens
            self.isUnsaltedSharedBase = isUnsaltedSharedBase
        }
    }

    /// Resolves exact -> longest-prefix -> unsalted shared-base hit against `QWSlotPinnedLRUStore`.
    public static func lookupWarmPrefix<Key: Hashable & Sendable, Value: Sendable>(
        in store: QWSlotPinnedLRUStore<Key, Value>,
        exactKey: Key,
        saltedBaseKey: Key?,
        tokenCount: Int,
        queryRootHash: UInt64?,
        allowExact: Bool = true,
        allowPrefix: Bool = true,
        sharedBaseOnly: Bool = false,
        touchOnHit: Bool = true,
        isCandidate: (_ key: Key, _ entry: QWSlotPinnedLRUStore<Key, Value>.StoredEntry, _ sharedBaseOnly: Bool) -> Bool,
        matchesPrefix: (_ key: Key, _ count: Int, _ sharedBaseFallback: Bool) -> Bool
    ) -> WarmLookupHit<Value>? {
        if allowExact && !sharedBaseOnly {
            if let entry = store.value(for: exactKey, touchOnHit: touchOnHit) {
                return WarmLookupHit(value: entry.value, matchedTokens: tokenCount, isUnsaltedSharedBase: false)
            }
            if let saltedBaseKey, store.isSharedBaseKey(saltedBaseKey),
               let entry = store.value(for: saltedBaseKey, touchOnHit: touchOnHit) {
                return WarmLookupHit(value: entry.value, matchedTokens: tokenCount, isUnsaltedSharedBase: false)
            }
        }
        guard allowPrefix else { return nil }
        let snap = store.snapshotOrderedEntries()
        if !sharedBaseOnly {
            var candidates: [(key: Key, count: Int)] = []
            candidates.reserveCapacity(snap.entries.count)
            for (key, entry) in snap.entries {
                guard entry.allowsSuffix, entry.tokenCount > 0, entry.tokenCount < tokenCount,
                      isCandidate(key, entry, false) else { continue }
                if entry.tokenCount >= store.rootPrefixTokenCount,
                   let qHash = queryRootHash, let eHash = entry.rootHash, qHash != eHash {
                    continue
                }
                candidates.append((key, entry.tokenCount))
            }
            candidates.sort { $0.count > $1.count }
            for cand in candidates where matchesPrefix(cand.key, cand.count, false) {
                if let entry = store.value(for: cand.key, touchOnHit: touchOnHit) {
                    return WarmLookupHit(value: entry.value, matchedTokens: cand.count, isUnsaltedSharedBase: false)
                }
            }
        }
        if (sharedBaseOnly || saltedBaseKey != nil), store.hasSharedBase {
            var shared: [(key: Key, count: Int)] = []
            for key in snap.sharedBaseKeys {
                guard let entry = snap.entries[key], entry.allowsSuffix,
                      entry.tokenCount > 0, entry.tokenCount < tokenCount,
                      isCandidate(key, entry, true) else { continue }
                shared.append((key, entry.tokenCount))
            }
            shared.sort { $0.count > $1.count }
            for cand in shared where matchesPrefix(cand.key, cand.count, true) {
                if let entry = store.value(for: cand.key, touchOnHit: touchOnHit) {
                    return WarmLookupHit(value: entry.value, matchedTokens: cand.count, isUnsaltedSharedBase: true)
                }
            }
        }
        return nil
    }

    /// Resolves slot assignment, prepares superseded slot entries, and stores into `QWSlotPinnedLRUStore`.
    @discardableResult
    public static func storeWithSlotPinning<Key: Hashable & Sendable, Value: Sendable>(
        in store: QWSlotPinnedLRUStore<Key, Value>,
        key: Key,
        value: Value,
        charge: Int,
        tokenCount: Int,
        allowsSuffix: Bool,
        rootHash: UInt64,
        history: [Int64],
        isSharedBase: Bool,
        matchesPrefix: (Key, Int) -> Bool
    ) -> Bool {
        let slot: Int? = isSharedBase ? nil : store.resolveSlot(
            rootHash: rootHash, history: history, isContinuationOf: matchesPrefix
        )
        if let slot, tokenCount >= store.rootPrefixTokenCount {
            store.prepareSlotForStore(
                slot: slot, history: history, incomingBytes: charge, isPrefixEntry: matchesPrefix
            )
        }
        return store.store(
            key: key,
            value: value,
            charge: charge,
            tokenCount: tokenCount,
            allowsSuffix: allowsSuffix,
            slot: slot,
            rootHash: rootHash,
            history: history,
            isSharedBase: isSharedBase
        )
    }

    public static func shouldSpillToCold(
        isSharedBase: Bool,
        warmHasSharedBase: Bool,
        ssdByteLimit: Int,
        entryBytes: Int,
        maxSpillBytes: Int
    ) -> Bool {
        !isSharedBase && !warmHasSharedBase
            && ssdByteLimit > 0
            && entryBytes >= 0
            && entryBytes <= ssdByteLimit
            && entryBytes <= maxSpillBytes
    }

    public static func incrementalCharge(fullBytes: Int, sharedBaseBytes: Int) -> Int {
        max(0, fullBytes - max(0, sharedBaseBytes))
    }

    /// Computes intermediate prompt prefix checkpoints before visual spans, periodic intervals, and branch forks.
    public static func computeRetainPoints(
        explicitRetainLength: Int?,
        matchedTokens: Int,
        firstVisualTokenOffset: Int,
        promptCount: Int,
        sharedBranchPrefixLength: Int?,
        commonWavePrefixLength: Int? = nil,
        periodicCheckpointInterval: Int = 0,
        includeTrailingAutoRetain: Bool = true
    ) -> [Int] {
        let firstVis = min(promptCount, max(0, firstVisualTokenOffset))
        guard firstVis > matchedTokens else { return [] }
        let hasVisualSpan = firstVis < promptCount
        var candidates: [Int] = []
        if let explicit = explicitRetainLength, explicit > matchedTokens, explicit < firstVis {
            candidates.append(explicit)
        }
        if hasVisualSpan, firstVis >= 64 {
            let preVis = firstVis - 4
            let curBase = candidates.last ?? matchedTokens
            if preVis >= 60, preVis > curBase + 4, preVis < firstVis {
                candidates.append(preVis)
            }
        }
        let branchCheckCeiling = hasVisualSpan ? (candidates.last ?? matchedTokens) : firstVis
        if let branchLen = sharedBranchPrefixLength,
           branchLen > matchedTokens,
           (!hasVisualSpan || (candidates.last ?? matchedTokens) == matchedTokens || (candidates.last ?? matchedTokens) == explicitRetainLength),
           branchLen < (hasVisualSpan ? firstVis : promptCount),
           (!hasVisualSpan || branchLen > branchCheckCeiling) {
            candidates.append(branchLen)
        }
        if let waveLen = commonWavePrefixLength, waveLen > matchedTokens, waveLen < firstVis {
            candidates.append(waveLen)
        }
        if periodicCheckpointInterval > 0 {
            let step = periodicCheckpointInterval
            var cursor = ((matchedTokens / step) + 1) * step
            while cursor < firstVis - 64 {
                let nearExisting = candidates.contains { abs($0 - cursor) < 256 }
                if !nearExisting {
                    candidates.append(cursor)
                }
                cursor += step
            }
        }
        candidates.sort()
        var retainPoints: [Int] = []
        for pt in candidates {
            if let prev = retainPoints.last, pt <= prev { continue }
            retainPoints.append(pt)
        }
        if !hasVisualSpan && includeTrailingAutoRetain && firstVis >= 64 {
            let autoBase = retainPoints.last ?? matchedTokens
            let autoRetain = firstVis - 4
            if autoRetain >= 60, autoRetain > autoBase + 4, autoRetain < firstVis {
                retainPoints.append(autoRetain)
            }
        }
        return retainPoints
    }
}
