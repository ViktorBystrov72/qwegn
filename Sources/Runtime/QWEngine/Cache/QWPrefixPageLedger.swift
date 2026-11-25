// QWPrefixPageLedger.swift — Reference-counted physical KV page byte ledger and pure-attention slicing.

import Foundation
import QWCore
import QWKV

struct QWPrefixPhysicalPage: Hashable, Sendable {
    let cacheID: ObjectIdentifier
    let planeIndex: Int
    let pageID: Int32
    let bytesPerPage: Int

    func hash(into hasher: inout Hasher) {
        hasher.combine(cacheID)
        hasher.combine(planeIndex)
        hasher.combine(pageID)
    }

    static func == (lhs: QWPrefixPhysicalPage, rhs: QWPrefixPhysicalPage) -> Bool {
        lhs.cacheID == rhs.cacheID && lhs.planeIndex == rhs.planeIndex && lhs.pageID == rhs.pageID
    }
}

/// Tracks unique physical `QWKVCache` pages referenced across all Warm-RAM entries in a store.
final class QWPrefixPageLedger<Key: Hashable & Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var pagesByKey: [Key: Set<QWPrefixPhysicalPage>] = [:]
    private var pageRefCounts: [QWPrefixPhysicalPage: Int] = [:]
    private var _uniqueBytes: Int = 0

    var uniqueBytes: Int {
        lock.withLock { _uniqueBytes }
    }

    static func extractUniquePages(from cached: QWEngineCachedPrefix) -> Set<QWPrefixPhysicalPage> {
        var out = Set<QWPrefixPhysicalPage>()
        cached.prefix.trunk.enumerateLivePhysicalPages { cacheID, planeIdx, pageID, pageBytes in
            out.insert(QWPrefixPhysicalPage(
                cacheID: cacheID, planeIndex: planeIdx, pageID: pageID, bytesPerPage: pageBytes
            ))
        }
        cached.prefix.mtp?.enumerateLivePhysicalPages { cacheID, planeIdx, pageID, pageBytes in
            out.insert(QWPrefixPhysicalPage(
                cacheID: cacheID, planeIndex: planeIdx, pageID: pageID, bytesPerPage: pageBytes
            ))
        }
        return out
    }

    static func totalBytes(of pages: Set<QWPrefixPhysicalPage>) -> Int {
        pages.reduce(0) { $0 + $1.bytesPerPage }
    }

    func projectedUniqueBytes(adding pages: Set<QWPrefixPhysicalPage>, replacing key: Key) -> Int {
        lock.withLock {
            var projected = _uniqueBytes
            let oldPages = pagesByKey[key] ?? []
            for p in oldPages where !pages.contains(p) {
                if pageRefCounts[p, default: 0] == 1 {
                    projected -= p.bytesPerPage
                }
            }
            for p in pages where !oldPages.contains(p) {
                if pageRefCounts[p, default: 0] == 0 {
                    projected += p.bytesPerPage
                }
            }
            return max(0, projected)
        }
    }

    func insert(key: Key, pages: Set<QWPrefixPhysicalPage>) {
        lock.withLock {
            removeLocked(key: key)
            guard !pages.isEmpty else { return }
            pagesByKey[key] = pages
            for p in pages {
                let prev = pageRefCounts[p, default: 0]
                if prev == 0 {
                    _uniqueBytes += p.bytesPerPage
                }
                pageRefCounts[p] = prev + 1
            }
        }
    }

    func remove(key: Key) {
        lock.withLock {
            removeLocked(key: key)
        }
    }

    private func removeLocked(key: Key) {
        guard let oldPages = pagesByKey.removeValue(forKey: key) else { return }
        for p in oldPages {
            let cur = pageRefCounts[p, default: 0]
            if cur <= 1 {
                pageRefCounts.removeValue(forKey: p)
                _uniqueBytes = max(0, _uniqueBytes - p.bytesPerPage)
            } else {
                pageRefCounts[p] = cur - 1
            }
        }
    }
}

extension QWEngineDriver {
    /// True when cached KV pages carry the entire prefix state and can be sliced at any `256`-token page boundary.
    var canSlicePureAttentionPages: Bool {
        geometry.gdnLayerCount == 0
            && geometry.kdaConfig == nil
            && !geometry.hasAnySlidingWindow
            && mlaOperator == nil
            && lsaOperator == nil
            && hcaOperator == nil
            && mbsaOperator == nil
            && mtpCache == nil
            && dflashDrafter == nil
            && echoDrafter == nil
    }

    /// Zero-copy slices a pure-attention `QWEngineCachedPrefix` to `targetTokens` (`targetTokens % 256 == 0`).
    func slicePureAttentionCachedPrefix(
        _ donor: QWEngineCachedPrefix,
        toPageAlignedTokens targetTokens: Int
    ) throws -> QWEngineCachedPrefix? {
        guard canSlicePureAttentionPages,
              targetTokens > 0,
              targetTokens < donor.prefix.position,
              targetTokens % QWKVCache.pageTokens == 0 else {
            return nil
        }
        guard let slicedTrunk = try kvCache.sliceSharedPrefix(
            donor.prefix.trunk, toPageAlignedTokens: targetTokens
        ) else {
            return nil
        }
        var rec = donor.prefix.state.roundRecord
        rec.consumedCount = targetTokens
        rec.absolutePosition = targetTokens
        rec.outputCount = 1
        let slicedState = QWEnginePrefixState(
            position: targetTokens,
            generated: 1,
            roundRecord: rec,
            recData: donor.prefix.state.recData,
            ringData: donor.prefix.state.ringData,
            trunkHiddenData: donor.prefix.state.trunkHiddenData,
            trunkHiddenValid: false,
            stochastic: nil,
            dflashKData: nil,
            dflashVData: nil,
            dflashCacheOffset: 0,
            mlaIndexerData: nil
        )
        let shared = QWEngineSharedPrefix(state: slicedState, trunk: slicedTrunk, mtp: nil)
        return QWEngineCachedPrefix(prefix: shared, nextToken: donor.nextToken, logits: donor.logits)
    }
}
