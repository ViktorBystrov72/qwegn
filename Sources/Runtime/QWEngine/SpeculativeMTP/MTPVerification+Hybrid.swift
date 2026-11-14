// MTPVerification+Hybrid.swift — Hybrid shared-last-layer ECHO + suffix/periodic PLD
// speculative proposal and commit-time exact stream pair resolution for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWMTP

extension QWEngineDriver {
    static let moeSpecExpertBudgetEnv: String? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_EXPERT_BUDGET"]

    /// Returns the active MoE-Spec (arXiv:2602.16052) unique-expert budget across `rows` verification tokens,
    /// or `0` when all routed experts should be evaluated losslessly.
    func activeMoESpecExpertBudget(rows: Int) -> Int {
        guard rows > 1, rows <= 4, geometry.numExperts > 0, geometry.expertsPerTok > 0 else {
            return 0
        }
        if let raw = Self.moeSpecExpertBudgetEnv {
            if raw.lowercased() == "auto" {
                let acceptRate = mtpDraftProposals >= 4
                    ? Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
                    : 0.75
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, acceptRate: acceptRate)
            }
            if let explicit = Int(raw), explicit > 0 {
                return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                    geometry: geometry, verifyRows: rows, explicitBudget: explicit)
            }
        }
        if tuning.moeSpecBudget > 0 {
            return QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(
                geometry: geometry, verifyRows: rows, explicitBudget: tuning.moeSpecBudget)
        }
        return 0
    }

    struct MTPRoundProposal {
        let drafts: [Int]
        let proposals: [QWMTPGumbelProposal]
        let draftedPairs: Int
    }

    private func draftFullHistory(session: QWEngineSession, pending: Int) -> [Int] {
        if session.tokenHistory.count > session.position, session.tokenHistory.last == pending {
            return session.tokenHistory
        }
        return session.tokenHistory + [pending]
    }

    private func adaptiveDraftCount(count: Int, defaultTrunkBits: Int) -> Int {
        guard mtpAdaptiveDepth && geometry.numExperts > 0 && mtpDraftProposals >= 8 && count > 1 else {
            return count
        }
        let trunkBits = layersW.first?.gu.view.bits ?? defaultTrunkBits
        let expertBits = layersW.compactMap({ $0.moe?.gateUp.bits }).first ?? trunkBits
        let acceptRate = Double(mtpDraftAccepted) / Double(max(1, mtpDraftProposals))
        return QWMTPAdaptiveDepth.optimalDraftCount(
            geometry: geometry, trunkBits: trunkBits, expertBits: expertBits,
            acceptRate: acceptRate, maxDrafts: count)
    }

    private func tryZeroCostContextDrafts(
        fullHistory: [Int],
        count: Int,
        strategy: QWDraftStrategy
    ) -> [Int]? {
        if let pd = QWContextCopyDrafter.periodicDrafts(
            history: fullHistory, maxCount: count, minRepeats: 3),
           !pd.isEmpty {
            return pd
        }
        let minMatch = (strategy == .ngram) ? 3 : 5
        if let match = QWContextCopyDrafter.drafts(
            history: fullHistory, maxCount: count, minMatchLen: minMatch),
           !match.tokens.isEmpty {
            let take = match.matchLen >= 6
                ? count
                : min(strategy == .ngram ? count : 2, match.tokens.count)
            return Array(match.tokens.prefix(max(1, take)))
        }
        return nil
    }

    func proposeEchoOrPLDRound(
        session: QWEngineSession,
        echoDrafter: QWEchoDraftEncoder,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> MTPRoundProposal {
        try proposeHybridRound(
            session: session, pending: pending, count: count,
            defaultTrunkBits: 8, owner: owner, traceTag: "[echo-draft]",
            useFullNeuralDepth: { $0 == .mtp || mtpStageProfiling || owner != nil },
            proposeNeural: { neuralCount, sampledToken, stopAfterDraft in
                try echoDrafter.propose(
                    driver: self, initialStream: hidden, kv: views,
                    pendingToken: pending, cacheRows: basePairs, drafts: neuralCount,
                    sampledToken: sampledToken, shouldStopAfterDraft: stopAfterDraft,
                    writeLogitsOnFinalDraft: mtpStageProfiling, completed: draftCompleted)
            })
    }

    func proposeNextNOrPLDRound(
        session: QWEngineSession,
        draftEncoder: QWMTPDraftEncoder,
        hidden: QWTensorValue,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        pending: Int,
        count: Int,
        basePairs: Int,
        owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)?
    ) throws -> MTPRoundProposal {
        try proposeHybridRound(
            session: session, pending: pending, count: count,
            defaultTrunkBits: 4, owner: owner, traceTag: "[draft]",
            useFullNeuralDepth: { !($0 == .ngram && owner == nil && !mtpStageProfiling) },
            proposeNeural: { neuralCount, sampledToken, stopAfterDraft in
                try QWMTPDraftChain.run(
                    hal: hal, draft: draftEncoder, headStorage: headStorage,
                    draftHead: mtpDraftHead, hidden: hidden, kv: views,
                    pendingToken: pending, drafts: neuralCount, position: session.position, cacheRows: basePairs,
                    selection: .gpuChecked,
                    deviceEmbedding: owner != nil ? nil : mtpDeviceEmbedding,
                    embedding: { try self.copyEmbedRow($0, to: $1) },
                    sampledToken: sampledToken,
                    shouldStopAfterDraft: stopAfterDraft,
                    writeLogitsOnFinalDraft: mtpStageProfiling,
                    completed: draftCompleted,
                    gpuFailure: { self.commandBufferErrors += 1 })
            })
    }

    private func proposeHybridRound(
        session: QWEngineSession,
        pending: Int,
        count: Int,
        defaultTrunkBits: Int,
        owner: QWStochasticSession?,
        traceTag: String,
        useFullNeuralDepth: (QWDraftStrategy) -> Bool,
        proposeNeural: (
            _ neuralCount: Int,
            _ sampledToken: ((QWTensorValue) throws -> Int)?,
            _ stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)?
        ) throws -> [Int]
    ) throws -> MTPRoundProposal {
        let effectiveCount = adaptiveDraftCount(count: count, defaultTrunkBits: defaultTrunkBits)
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let fullHistory = draftFullHistory(session: session, pending: pending)
        if owner == nil && !mtpStageProfiling && resolvedStrategy != .mtp,
           let chosen = tryZeroCostContextDrafts(fullHistory: fullHistory, count: count, strategy: resolvedStrategy) {
            lastRoundUsedPLD = true
            lastDraftedPairs = 0
            mtpDraftProposals += chosen.count
            return MTPRoundProposal(drafts: chosen, proposals: [], draftedPairs: 0)
        }

        let neuralCount = useFullNeuralDepth(resolvedStrategy) ? effectiveCount : 1
        var proposals: [QWMTPGumbelProposal] = []
        if let owner {
            try owner.stream.beginGumbelRound(drafts: neuralCount, vocabulary: geometry.vocab)
        }
        var draftHistory = fullHistory
        let sampledToken: ((QWTensorValue) throws -> Int)? = owner.map { owner in
            { logits in
                let proposal = try owner.stream.sampleGumbelProposal(
                    owner.distribution(logits: logits, repetitionHistory: draftHistory))
                proposals.append(proposal)
                draftHistory.append(proposal.token)
                return proposal.token
            }
        }
        let needsAdaptiveStop = mtpAdaptiveDepth && (neuralCount > 1 || mtpStageProfiling)
        let stopAfterDraft: ((QWTensorValue, Int) throws -> Bool)? = needsAdaptiveStop ? { logits, completedDrafts in
            if completedDrafts >= neuralCount && !self.mtpStageProfiling {
                return false
            }
            let values = logits.region.pointer.advanced(by: logits.offset).assumingMemoryBound(to: Float.self)
            let entropy = try QWMTPAdaptiveDepth.normalizedEntropy(
                UnsafeBufferPointer(start: values, count: logits.shape.elementCount))
            self.pendingMTPAdaptiveEntropy?.append(entropy)
            return try QWMTPAdaptiveDepth.shouldStop(
                normalizedEntropy: entropy, threshold: self.mtpAdaptiveEntropyThreshold,
                completedDrafts: completedDrafts, maximumDrafts: neuralCount)
        } : nil

        let tDraft0 = Self.decodeTraceEnabled ? Date() : nil
        var drafts = try proposeNeural(neuralCount, sampledToken, stopAfterDraft)
        let draftedPairs = drafts.count
        if owner == nil, !mtpStageProfiling, resolvedStrategy != .mtp,
           count > drafts.count, let lastDraft = drafts.last,
           let cont = QWContextCopyDrafter.drafts(
               history: fullHistory + drafts.dropLast(), appendedToken: lastDraft,
               maxCount: count - drafts.count, minMatchLen: 4),
           !cont.tokens.isEmpty {
            let extra = cont.matchLen >= 6 ? (count - drafts.count) : 1
            drafts.append(contentsOf: cont.tokens.prefix(max(1, extra)))
        }
        if let tDraft0 {
            FileHandle.standardError.write(Data(String(
                format: "%@ pos=%d count=%d wall=%.2fms\n",
                traceTag, session.position, drafts.count, Date().timeIntervalSince(tDraft0) * 1000).utf8))
        }
        lastDraftedPairs = draftedPairs
        mtpDraftProposals += drafts.count
        return MTPRoundProposal(drafts: drafts, proposals: proposals, draftedPairs: draftedPairs)
    }

    func encodeEchoCommitResolution(
        _ cb: any MTLCommandBuffer,
        session: QWEngineSession,
        echoDrafter: QWEchoDraftEncoder,
        window: QWVerifyWindow,
        trunkHidden: QWTensorValue,
        basePairs: Int,
        retained: Int,
        usedPLD: Bool
    ) throws {
        guard let views = session.mtpKVViews.first, let windowStream = window.mhcStream else {
            throw QWEngineDriverError.state("ECHO commit resolution requires MTP KV views and window mHC stream")
        }
        let streamElems = echoDrafter.streams * geometry.hidden
        let streamBytes = streamElems * 4

        if usedPLD {
            guard retained > 0 else { return }
            let embSlice = try echoDrafter.stageEmbeddings(driver: self, tokens: verifiedInputs[0..<retained])
            let prevSlice: QWTensorValue
            if retained == 1 {
                prevSlice = trunkHidden
            } else {
                guard let prepBlit = cb.makeBlitCommandEncoder() else {
                    throw QWEngineDriverError.state("ECHO PLD commit stage blit encoder")
                }
                prepBlit.copy(from: trunkHidden.region.buffer,
                              sourceOffset: trunkHidden.region.base + trunkHidden.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset,
                              size: streamBytes)
                prepBlit.copy(from: windowStream.region.buffer,
                              sourceOffset: windowStream.region.base + windowStream.offset,
                              to: echoDrafter.prevStreamBuf.region.buffer,
                              destinationOffset: echoDrafter.prevStreamBuf.region.base + echoDrafter.prevStreamBuf.offset + streamBytes,
                              size: (retained - 1) * streamBytes)
                prepBlit.endEncoding()
                prevSlice = echoDrafter.prevStreamBuf.slice(byteOffset: 0, count: retained * streamElems, dtype: .f32)
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("ECHO PLD commit seed compute encoder")
            }
            do {
                bindResidentResources(enc)
                enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
                try echoDrafter.encodeSeedChunk(enc, driver: self, prevStream: prevSlice,
                                                embeddings: embSlice, kv: views,
                                                cacheRows: basePairs, rows: retained)
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            return
        }

        guard retained >= 2 else { return }
        let extraRows = retained - 1
        let prevSlice = windowStream.slice(byteOffset: 0, count: extraRows * streamElems, dtype: .f32)
        let embSlice = try echoDrafter.stageEmbeddings(driver: self, tokens: verifiedInputs[1..<retained])
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("ECHO verify refresh compute encoder")
        }
        do {
            bindResidentResources(enc)
            enc.useResource(echoDrafter.region.buffer, usage: [.read, .write])
            try echoDrafter.encodeSeedChunk(enc, driver: self, prevStream: prevSlice,
                                            embeddings: embSlice, kv: views,
                                            cacheRows: basePairs + 1, rows: extraRows)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func encodeNextNCommitResolution(
        _ cb: any MTLCommandBuffer,
        session: QWEngineSession,
        draft: QWMTPDraftEncoder,
        window: QWVerifyWindow,
        trunkHidden: QWTensorValue,
        basePairs: Int,
        retained: Int,
        usedPLD: Bool
    ) throws {
        guard let views = session.mtpKVViews.first else {
            throw QWEngineDriverError.state("NextN commit resolution requires MTP KV views")
        }
        let h = geometry.hidden
        let seedable = min(retained, max(0, draft.positionCapacity - session.position))
        if usedPLD {
            guard seedable > 0 else { return }
            for r in 0..<seedable {
                try copyEmbedRow(verifiedInputs[r], to: try window.embeddingRow(r))
            }
            let embSlice = QWTensorValue(region: window.embeddings.region, offset: window.embeddings.offset,
                                         shape: QWTensorShape([seedable * h], .f16))
            let trunkSlice = QWTensorValue(region: window.hidden.region, offset: window.hidden.offset,
                                           shape: QWTensorShape([seedable * h], .f16))
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("NextN PLD commit seed compute encoder")
            }
            do {
                bindResidentResources(enc)
                try QWMTPPromptSeeder.encode(enc, draft: draft, embeddings: embSlice, trunkHidden: trunkSlice,
                                             previousHidden: trunkHidden, position: session.position,
                                             count: seedable, cacheRows: basePairs, kv: views)
            } catch {
                enc.endEncoding()
                throw error
            }
            enc.endEncoding()
            return
        }

        guard seedable >= 2 else { return }
        let extra = seedable - 1
        for r in 1..<seedable {
            try copyEmbedRow(verifiedInputs[r], to: try window.embeddingRow(r))
        }
        let embSlice = QWTensorValue(region: window.embeddings.region, offset: window.embeddings.offset + h * 2,
                                     shape: QWTensorShape([extra * h], .f16))
        let prevHidden = try window.hiddenRow(0)
        let trunkSlice = QWTensorValue(region: window.hidden.region, offset: window.hidden.offset + h * 2,
                                       shape: QWTensorShape([extra * h], .f16))
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("NextN verify refresh compute encoder")
        }
        do {
            bindResidentResources(enc)
            try QWMTPPromptSeeder.encode(enc, draft: draft, embeddings: embSlice, trunkHidden: trunkSlice,
                                         previousHidden: prevHidden, position: session.position + 1,
                                         count: extra, cacheRows: basePairs + 1, kv: views)
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
    }

    func finalizeMTPPairCursor(
        session: QWEngineSession,
        mtpCache: QWKVCache,
        mtpHandle: QWKVHandle,
        basePairs: Int,
        draftedPairs: Int,
        retainedPairs: Int
    ) throws {
        let currentWritten = basePairs + draftedPairs
        let targetWritten = basePairs + retainedPairs
        if targetWritten > currentWritten {
            mtpCache.markWritten(mtpHandle, upTo: targetWritten)
        } else if targetWritten < currentWritten {
            try mtpCache.rollback(mtpHandle, by: currentWritten - targetWritten)
            session.mtpKVViews = [try mtpCache.layerViews(mtpHandle, layer: 0)]
        }
    }

    func greedyRoundBoundary(_ session: QWEngineSession) throws -> QWGreedyRoundBoundary {
        guard let mtpCache, let handle = session.mtpKV else {
            throw QWEngineDriverError.state("greedy round pair cache is absent")
        }
        return QWGreedyRoundBoundary(position: session.position, generated: session.generated,
                                     record: session.roundRecord, hiddenValid: session.trunkHiddenValid,
                                     trunkRewind: try kvCache.rollbackGeneration(session.kv),
                                     pairRewind: try mtpCache.rollbackGeneration(handle))
    }

    /// A guarded round publication door, not generation integration. The draft
    /// owner must have materialized N MTP pairs from the saved round boundary.
    func commitGreedyVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                  drain: Bool = true,
                                  permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round, stochastic: false, drain: drain, permit: permit)
    }

    func commitStochasticVerification(_ session: QWEngineSession, round: QWPreparedStochasticRound,
                                      drain: Bool = true,
                                      permit: QWGenerationPermit? = nil) throws {
        try commitPreparedVerification(session, round: round.authorization, stochastic: true, drain: drain, permit: permit)
    }
}
