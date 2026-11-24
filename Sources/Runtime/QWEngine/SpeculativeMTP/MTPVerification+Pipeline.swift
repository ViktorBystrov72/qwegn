// MTPVerification+Pipeline.swift — Async Commit -> Draft -> Verify single-wait pipeline helpers.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWKV
import QWMTP

extension QWEngineDriver {
    enum AsyncNeuralDraftPlan {
        case pld([Int])
        case asyncNeural(draftCBs: [any MTLCommandBuffer], draftedPairs: Int)
    }

    func waitForPriorCommitCB(_ cb: any MTLCommandBuffer, session: QWEngineSession) throws {
        cb.waitUntilCompleted()
        if cb.status != .completed {
            admission.commandBufferErrors += 1
            session.stateFailed = true
            throw QWEngineDriverError.state("greedy commit GPU failure: \(String(describing: cb.error))")
        }
    }

    func prepareAsyncNeuralDraft(
        session: QWEngineSession,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        mtpCache: QWKVCache,
        mtpHandle: QWKVHandle,
        table: QWMTPEmbeddingTable
    ) throws -> AsyncNeuralDraftPlan? {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let (extraPending, previousToken) = pendingHistoryContext(session: session, pending: pending)
        if resolvedStrategy != .mtp,
           let chosen = tryZeroCostContextDrafts(
               session: session, pending: pending, extraPending: extraPending,
               previousToken: previousToken, count: count, strategy: resolvedStrategy) {
            return .pld(chosen)
        }
        let neuralCount: Int
        if echoDrafter != nil {
            neuralCount = (resolvedStrategy == .mtp) ? count : 1
        } else {
            let neuralCap = QWMTPAdaptiveDepth.hybridNextNNeuralDepthCap(
                isStochastic: false, stageProfiling: false,
                hasDeviceEmbedding: true, numExperts: geometry.numExperts)
            var n = resolvedStrategy == .ngram ? 1 : count
            if resolvedStrategy == .hybrid, let neuralCap {
                n = min(n, max(1, neuralCap))
            }
            neuralCount = n
        }
        let hasPostContinuation = resolvedStrategy != .mtp && count > neuralCount && (
            QWContextCopyDrafter.drafts(
                history: session.tokenHistory, extraToken: extraPending,
                maxCount: 1, minMatchLen: 5) != nil
            || geometry.numExperts == 0 || resolvedStrategy == .ngram
        )
        guard !hasPostContinuation, neuralCount >= 1, neuralCount <= 64 else {
            return nil
        }
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        ids[0] = Int32(pending)
        let draftCBs: [any MTLCommandBuffer]
        if let echoDrafter {
            draftCBs = try echoDrafter.enqueueSingleCB(
                driver: self, table: table, initialStream: hidden, kv: views,
                pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
                writeLogitsOnFinalDraft: false, journalTokenOffset: 1)
        } else if let mtpDraftEncoder {
            draftCBs = try QWMTPDraftChain.enqueueDeviceChain(
                hal: hal, draft: mtpDraftEncoder, headStorage: headStorage,
                draftHead: mtpDraftHead, hidden: hidden, kv: views,
                pendingToken: pending, drafts: neuralCount, position: session.position,
                cacheRows: basePairs, mropeDelta: session.roundRecord.mropeDelta,
                table: table, journalTokenOffset: 1)
        } else {
            return nil
        }
        mtpCache.markWritten(mtpHandle, upTo: basePairs + neuralCount)
        return .asyncNeural(draftCBs: draftCBs, draftedPairs: neuralCount)
    }

    func tryZeroCostContextDrafts(
        session: QWEngineSession,
        pending: Int,
        extraPending: Int?,
        previousToken: Int?,
        count: Int,
        strategy: QWDraftStrategy
    ) -> [Int]? {
        if let pd = QWContextCopyDrafter.periodicDrafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minRepeats: 3),
           !pd.isEmpty {
            return pd
        }
        let isMoE = geometry.numExperts > 0
        let minMatch = (strategy == .ngram) ? 3 : (isMoE ? 6 : 5)
        if let match = QWContextCopyDrafter.drafts(
            history: session.tokenHistory, extraToken: extraPending,
            maxCount: count, minMatchLen: minMatch),
           !match.tokens.isEmpty {
            let take = match.matchLen >= 6
                ? count
                : min(strategy == .ngram ? count : 2, match.tokens.count)
            var out = Array(match.tokens.prefix(max(1, take)))
            if strategy == .ngram, out.count < count {
                session.tokenRecycler.appendDrafts(
                    into: &out, maxCount: count,
                    previousToken: previousToken, pendingToken: pending,
                    maxUnigramFallbacks: isMoE ? 0 : 2)
            }
            return out
        }
        if strategy == .ngram {
            var recycled: [Int] = []
            recycled.reserveCapacity(count)
            session.tokenRecycler.appendDrafts(
                into: &recycled, maxCount: count,
                previousToken: previousToken, pendingToken: pending,
                maxUnigramFallbacks: isMoE ? 0 : 2)
            if !recycled.isEmpty { return recycled }
        }
        return nil
    }
}
