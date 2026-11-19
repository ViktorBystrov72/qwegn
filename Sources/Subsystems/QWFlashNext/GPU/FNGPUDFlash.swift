// FNGPUDFlash.swift — DFlash 2 / DSpine speculative decoding, hybrid PLD bypass,
// sliding-window capture, prefix-cache persistence, and residency helpers for Flash-Next.
import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWMTP
import QWQuant

struct FNGPUDFlashPrefixHeader {
    var magic: UInt32
    var numLayers: UInt32
    var geometryHash: UInt64
    var cacheOffset: UInt64
    var activeRows: UInt32
    var rowBytes: UInt32
}

public enum FNGPUDFlash {
    public static let maxDrafts = QWFourSlotRollbackRing.maxVerifyRows - 1
    public static let prefixMagic: UInt32 = 0x4446_4C31
    public static let defaultMTPPackSavingsBytes = 1_440 << 20
    public static let defaultDFlashWeightBytes = 576 << 20
    public static let defaultDFlashScratchBytes = 112 << 20

    public static func isDFlashName(_ raw: String?) -> Bool {
        guard let lower = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else { return false }
        return lower == "dflash2" || lower == "dflash" || lower == "dspine"
    }

    public static func adjustedPerTokenKVBytes(_ perTokenBytes: Int, usesDFlash: Bool) -> Int {
        usesDFlash ? (512 + (((max(0, perTokenBytes - 512) * 12) + 12) / 13)) : perTokenBytes
    }

    static func resolveDrafterPackPath(targetDescriptor: QWModelDescriptor?, explicitPath: String?) -> String {
        let trimmed = (explicitPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed.expandingTildePath }
        let env = (ProcessInfo.processInfo.environment["QW_DFLASH_DRAFT_PACK"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return !env.isEmpty ? env.expandingTildePath : QWDFlashConfig.defaultPackPath(for: targetDescriptor)
    }

    /// Draft vocab rows for DFlash proposals (`QWG_DRAFT_VOCAB`, default 98304, 0 = full vocab).
    static func resolveDraftVocab(fullVocab: Int) -> Int? {
        if let raw = ProcessInfo.processInfo.environment["QWG_DRAFT_VOCAB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let value = Int(raw) {
            return (value > 0 && value < fullVocab) ? value : nil
        }
        return fullVocab > 98_304 ? 98_304 : nil
    }

    public static func drafterWeightBytes(packPath: String?, isDSpine: Bool) -> Int {
        guard let raw = packPath?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return defaultDFlashWeightBytes }
        let url = raw.expandingTildePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return defaultDFlashWeightBytes }
        if let cfgURL = QWModelRoot(url: url).configURL, let data = try? Data(contentsOf: cfgURL),
           let desc = try? QWModelDescriptor.decodeDrafter(configJSON: data),
           let pack = try? QWQuant.Pack(url: url, descriptor: desc, skipIdentityCheck: true) {
            return QWDFlashDrafter.computeWeightCapacity(pack: pack, isDSpine: isDSpine)
        }
        let fileBytes = (try? FileManager.default.attributesOfItem(atPath: QWModelRoot(url: url).weights.path)[.size] as? Int) ?? 0
        return max(64 << 20, fileBytes + (32 << 20))
    }

    static func estimateScratchBytes(config: QWDFlashConfig) -> Int {
        func aligned(_ bytes: Int) -> Int { ((bytes + 4095) / 4096) * 4096 }
        let h = config.hiddenSize, v = config.vocabSize, i = config.intermediateSize
        let maxCtx = config.maxCacheSize, maxB = max(8, config.blockSize), maxD = maxB - 1
        let qDim = config.numAttentionHeads * config.headDim, kvDim = config.numKeyValueHeads * config.headDim
        let hasConv = config.convKernelSize > 0
        let convProj = hasConv ? (2 * config.convKernelSize * (h / max(1, config.convGroupSize))) : 0
        var total = aligned(maxCtx * config.targetLayerIds.count * h * 2) + aligned(maxCtx * h * 2)
        total += aligned(maxCtx * 2 * kvDim * 2) + aligned(maxB * h * 4) * 2 + aligned(maxB * h * 2) * 3
        total += aligned(maxB * 4) * 2 + (hasConv ? aligned(maxB * convProj * 4) : 0)
        total += aligned(maxB * (qDim + 2 * kvDim) * 2) + aligned(maxB * qDim * 2) + aligned(maxB * i * 2)
        total += aligned(maxD * v * 4) + aligned(maxD * 256 * 16 * 4) * 2 + aligned(maxD * 16 * 4) * 2
        total += aligned(config.isDSpine ? (maxD * config.bridgeCodeSize * 4) : (maxD * config.selectorRank * 4))
        total += 4 * aligned(config.numLayers * maxCtx * kvDim * 2)
        return max(32 << 20, total + (16 << 20))
    }

    public static func estimateFootprint(
        targetDescriptor: QWModelDescriptor, targetPackPath: String?, dflashPackPath: String?, isDFlash: Bool
    ) -> (nativeMTPWeightBytes: Int, dflashWeightBytes: Int, dflashScratchBytes: Int) {
        guard isDFlash else { return (0, 0, 0) }
        var nativeMTP = defaultMTPPackSavingsBytes
        if let tPath = targetPackPath?.trimmingCharacters(in: .whitespacesAndNewlines), !tPath.isEmpty,
           let pack = try? QWQuant.Pack(url: tPath.expandingTildePath.canonicalFileURL,
                                        descriptor: targetDescriptor, skipIdentityCheck: true) {
            let summed = pack.tensors.reduce(0) { acc, kv in
                guard kv.key.hasPrefix("mtp.") else { return acc }
                return acc + (PlaneLayout.resolve(kv.value)?.byteCount(kv.value)
                    ?? ((kv.value.hq8PayloadBytes ?? 0) + (kv.value.hq8ScaleBytes ?? 0)))
            }
            if summed > 0 { nativeMTP = summed }
        }
        let resolved = resolveDrafterPackPath(targetDescriptor: targetDescriptor, explicitPath: dflashPackPath)
        guard FileManager.default.fileExists(atPath: resolved.canonicalFileURL.path) else {
            return (nativeMTP, defaultDFlashWeightBytes, defaultDFlashScratchBytes)
        }
        let cfg = QWDFlashConfig.load(from: resolved.canonicalFileURL)
        return (nativeMTP, drafterWeightBytes(packPath: resolved, isDSpine: cfg.isDSpine), estimateScratchBytes(config: cfg))
    }

    static func prefillWindowEnd(historyCount: Int, targetEnd: Int?) -> Int {
        min(historyCount, targetEnd ?? historyCount)
    }

    static func prefillCaptureSlice(startPos: Int, rows: Int, historyCount: Int, maxCacheSize: Int) -> (localStart: Int, captureRows: Int, ctxStartPos: Int)? {
        guard rows > 0, maxCacheSize > 0 else { return nil }
        let windowStart = max(0, max(historyCount, startPos + rows) - maxCacheSize)
        guard startPos + rows > windowStart else { return nil }
        let rawLocal = max(0, windowStart - startPos), rawCount = rows - rawLocal
        guard rawCount > 0 else { return nil }
        let captureRows = min(rawCount, maxCacheSize), localStart = rawLocal + (rawCount - captureRows)
        return (localStart, captureRows, startPos + localStart)
    }

    static func encodeLayerCapture(
        _ enc: any MTLComputeCommandEncoder, dflash: QWDFlashDrafter, layerIndex: Int, deferOut: Bool,
        base: QWTensorValue, src: QWTensorValue, inj: QWTensorValue, cur: QWTensorValue,
        srcRowOffset: Int, dstRowOffset: Int, rows: Int, hc: Int
    ) {
        guard rows > 0, (0..<dflash.targetLayerSlotMap.count).contains(layerIndex),
              case let slot = dflash.targetLayerSlotMap[layerIndex], slot >= 0 else { return }
        enc.memoryBarrier(scope: .buffers)
        dflash.encodeCaptureHCTargetHidden(
            enc, base: deferOut ? base : cur, srcRowOffset: srcRowOffset, dstRowOffset: dstRowOffset,
            rows: rows, slot: slot, hc: hc, deferredInject: deferOut ? (src: src, inj: inj, isMHC: false) : nil)
        enc.memoryBarrier(scope: .buffers)
    }

    static func dflashGeometryMarker(config: QWDFlashConfig) -> String {
        let layers = config.targetLayerIds.map(String.init).joined(separator: ",")
        return "df:\(config.hiddenSize):\(config.numLayers):\(config.numAttentionHeads):"
            + "\(config.numKeyValueHeads):\(config.headDim):\(config.maxCacheSize):"
            + "\(config.isDSpine ? 1 : 0):\(layers)"
    }

    static func dflashGeometryHash(_ marker: String) -> UInt64 {
        marker.utf8.reduce(0xcbf2_9ce4_8422_2325) { ($0 ^ UInt64($1)) &* 0x1000_0000_01b3 }
    }

    static func prefixPayloadBytes(config: QWDFlashConfig, tokenCount: Int) -> Int {
        let activeRows = min(max(0, tokenCount), config.maxCacheSize)
        return MemoryLayout<FNGPUDFlashPrefixHeader>.size + 2 * config.numLayers * activeRows * config.numKeyValueHeads * config.headDim * 2
    }

    static func appendPrefixPayload(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        dflash: QWDFlashDrafter, marker: String, tokenCount: Int
    ) {
        let cfg = dflash.config
        appendPrefixKV(to: dstBase, at: &writeOffset, slots: dflash.sessionSlots.first, config: cfg,
                       marker: marker, tokenCount: tokenCount)
    }

    static func appendPrefixKV(
        to dstBase: UnsafeMutableRawPointer, at writeOffset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, marker: String, tokenCount: Int
    ) {
        let activeRows = min(max(0, tokenCount), cfg.maxCacheSize)
        let rowBytes = cfg.numKeyValueHeads * cfg.headDim * 2, layerBytes = activeRows * rowBytes
        var ph = FNGPUDFlashPrefixHeader(
            magic: prefixMagic, numLayers: UInt32(cfg.numLayers), geometryHash: dflashGeometryHash(marker),
            cacheOffset: UInt64(tokenCount), activeRows: UInt32(activeRows), rowBytes: UInt32(rowBytes))
        withUnsafeBytes(of: &ph) { (buf: UnsafeRawBufferPointer) in
            if let p = buf.baseAddress { memcpy(dstBase.advanced(by: writeOffset), p, buf.count); writeOffset += buf.count }
        }
        guard layerBytes > 0, let slot = slots else { return }
        for l in 0..<cfg.numLayers {
            for t in [slot.kLayers[l], slot.vLayers[l]] {
                memcpy(dstBase.advanced(by: writeOffset), t.region.buffer.contents().advanced(by: t.region.base + t.offset), layerBytes)
                writeOffset += layerBytes
            }
        }
    }

    static func restorePrefixPayload(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        dflash: QWDFlashDrafter, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        restorePrefixKV(from: base, dataCount: dataCount, offset: &offset, slots: dflash.sessionSlots.first,
                        config: dflash.config, expectedMarker: expectedMarker, expectedTokens: expectedTokens)
    }

    static func restorePrefixKV(
        from base: UnsafeRawPointer, dataCount: Int, offset: inout Int,
        slots: QWDFlashSessionSlot?, config cfg: QWDFlashConfig, expectedMarker: String, expectedTokens: Int
    ) -> Int? {
        let hdrSize = MemoryLayout<FNGPUDFlashPrefixHeader>.size
        guard offset + hdrSize <= dataCount else { return nil }
        var ph = FNGPUDFlashPrefixHeader(magic: 0, numLayers: 0, geometryHash: 0, cacheOffset: 0, activeRows: 0, rowBytes: 0)
        memcpy(&ph, base.advanced(by: offset), hdrSize); offset += hdrSize
        let active = min(max(0, expectedTokens), cfg.maxCacheSize)
        let rowBytes = cfg.numKeyValueHeads * cfg.headDim * 2, layerBytes = active * rowBytes
        guard ph.magic == prefixMagic, ph.geometryHash == dflashGeometryHash(expectedMarker),
              Int(ph.numLayers) == cfg.numLayers, Int(ph.cacheOffset) == expectedTokens,
              Int(ph.activeRows) == active, Int(ph.rowBytes) == rowBytes,
              offset + 2 * cfg.numLayers * layerBytes <= dataCount, let slot = slots else { return nil }
        if layerBytes > 0 {
            for l in 0..<cfg.numLayers {
                for t in [slot.kLayers[l], slot.vLayers[l]] {
                    memcpy(t.region.buffer.contents().advanced(by: t.region.base + t.offset), base.advanced(by: offset), layerBytes)
                    offset += layerBytes
                }
            }
        }
        return Int(ph.cacheOffset)
    }

    static func validatePrefixBlob(data: Data, expectedMarker: String, expectedTokens: Int, config: QWDFlashConfig) -> Bool {
        let start = data.count - prefixPayloadBytes(config: config, tokenCount: expectedTokens)
        guard start >= 0 else { return false }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            guard let base = raw.baseAddress else { return false }
            var ph = FNGPUDFlashPrefixHeader(magic: 0, numLayers: 0, geometryHash: 0, cacheOffset: 0, activeRows: 0, rowBytes: 0)
            memcpy(&ph, base.advanced(by: start), MemoryLayout<FNGPUDFlashPrefixHeader>.size)
            return ph.magic == prefixMagic && ph.geometryHash == dflashGeometryHash(expectedMarker)
                && Int(ph.cacheOffset) == expectedTokens
        }
    }

    static func encodeHeadLogits(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal, headWeight: QWWeightView,
        xRot16: QWTensorValue, xNorm16: QWTensorValue, xRot32: QWTensorValue?,
        yLogits32: QWTensorValue, rows: Int, vocabRows: Int
    ) throws {
        let w: QWWeightView
        if vocabRows < headWeight.n {
            let sliced = headWeight.blocks.slice(byteOffset: 0, count: (vocabRows * headWeight.rowBytes) / 2, dtype: .f16)
            guard let sub = headWeight.subview(blocks: sliced, n: vocabRows, k: headWeight.k) else {
                throw QWDFlashError.state("FNGPUDFlash: failed to slice lm_head to vocabRows=\(vocabRows)")
            }
            w = sub
        } else {
            w = headWeight
        }
        if hal.caps.isMetal4TensorCapable && w.n % 64 == 0 && w.k % 64 == 0 && (!w.isRotated || rows >= 5) {
            let x16 = w.isRotated ? xRot16 : xNorm16
            try QWGemm.encodeNaxF32(enc, hal: hal, weight: w, x16: x16, y16: x16, y32: yLogits32, rows: rows)
            return
        }
        guard let xRot32 else { throw QWDFlashError.state("FNGPUDFlash: missing xRot32 for v4 head projection") }
        try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: w, x: xRot32, y: yLogits32, rows: rows)
    }

    static func selectHybridBypassDrafts(
        history: [Int64], pendingToken: Int, mode: QWDraftStrategy, maxDrafts: Int, recycler: QWTokenRecycler
    ) -> [Int] {
        guard maxDrafts > 0, mode != .mtp else { return [] }
        let minPld = (mode == .ngram) ? min(4, maxDrafts) : min(7, maxDrafts)
        if let pld = QWContextCopyDrafter.periodicDrafts(
            history: history.suffix(256).map(Int.init), extraToken: pendingToken, maxCount: maxDrafts, minRepeats: 3
        ), pld.count >= minPld { return Array(pld.prefix(maxDrafts)) }
        var drafts: [Int] = []
        if let match = QWContextCopyDrafter.drafts(
            history: history, appendedToken: Int64(pendingToken), maxCount: maxDrafts, minMatchLen: 6
        ), !match.tokens.isEmpty {
            drafts = Array(match.tokens.prefix(maxDrafts)).map { Int($0) }
        }
        if !drafts.isEmpty || mode == .ngram {
            recycler.appendDrafts(into: &drafts, maxCount: maxDrafts,
                                  previousToken: history.last.map(Int.init), pendingToken: pendingToken,
                                  maxUnigramFallbacks: 0)
            if drafts.count >= minPld { return drafts }
        }
        return []
    }

    static func maxFittingContext(activeCapacity: Int) -> Int {
        max(0, activeCapacity - 1)
    }

    static func consumePendingContextUpdate(
        pos: Int, count: inout Int, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        let pending = count
        count = 0
        guard pending > 0, let slot else { return nil }
        return (slot, pos, pending)
    }

    static func consumePendingContextUpdate(
        _ session: inout FNGPUSessionData, slot: QWDFlashSessionSlot?
    ) -> (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? {
        consumePendingContextUpdate(pos: session.dflashPendingCtxPos,
                                    count: &session.dflashPendingCtxCount,
                                    slot: slot)
    }

    static func step(
        _ engine: FNGPUEngine, session: inout FNGPUSessionData, token: Int,
        vStack: FNGPUDecodeStack, dflash: QWDFlashDrafter
    ) throws -> Int {
        guard let dSlot = dflash.sessionSlots.first else {
            throw QWDFlashError.state("FNGPUDFlash: missing DFlash session slot")
        }
        let prevCount = session.history.count
        let dflashCap = min(maxDrafts, max(1, dflash.config.blockSize - 1))
        let maxCap = max(1, min(min(session.maxDrafts, dflashCap), engine.activeCapacity - prevCount - 2))
        let stepT0 = Date(), trackRep = session.sampler.config.repetitionPenalty != 1.0
        let isUnconstrained = session.constraint?.isUnconstrainedNow ?? true, isGreedy = session.sampler.config.isGreedy
        let allowPLD = isUnconstrained && isGreedy && session.seenTokens.isEmpty && !session.sampler.config.hasExplicitSeed
        let preCtx = consumePendingContextUpdate(&session, slot: dSlot)
        var drafts = allowPLD
            ? selectHybridBypassDrafts(history: session.history, pendingToken: token,
                                       mode: session.draftMode, maxDrafts: maxCap, recycler: session.tokenRecycler)
            : []
        let usedPLD = !drafts.isEmpty
        var proposals: [QWMTPGumbelProposal] = [], draftGPUMs = 0.0
        let stream: QWMTPRandomStream? = (!usedPLD && !isGreedy) ? QWMTPRandomStream(cloning: try session.ensureDFlashStreamState()) : nil
        if !usedPLD {
            let draftCount = session.dflashPolicy.selectDraftCount(maxDrafts: maxCap)
            if let stream { try stream.beginGumbelRound(drafts: draftCount, vocabulary: engine.vocab, maxDrafts: dflashCap) }
            do {
                let proposed = try dflash.propose(
                    hal: engine.ctx.hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                    pendingToken: token, draftCount: draftCount, draftVocab: resolveDraftVocab(fullVocab: engine.vocab),
                    headStorage: engine.dflashHeadStorage,
                    encodeHeadLogits: { enc, xRot16, xNorm16, xRot32, yLogits32, rows, vocabRows in
                        try encodeHeadLogits(enc, hal: engine.ctx.hal, headWeight: engine.head.weight, xRot16: xRot16,
                                             xNorm16: xNorm16, xRot32: xRot32, yLogits32: yLogits32, rows: rows, vocabRows: vocabRows)
                    },
                    encodeAnchorMaskEmbed: { enc, propX32, anchorToken, maskToken, rows in
                        try dflash.encodeStageProp(enc, embedTable: engine.embedTable, propX: propX32,
                                                   anchorToken: anchorToken, maskToken: maskToken, rows: rows)
                    },
                    stochastic: stream.map { (Float(max(session.sampler.config.temperature, 1e-6)), $0) },
                    preContextUpdate: preCtx,
                    completed: { cb in draftGPUMs = max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1000.0) })
                drafts = proposed.drafts; proposals = proposed.proposals
            } catch { stream?.cancelActiveRound(); throw error }
        }
        let dCount = drafts.count, activeRows = 1 + dCount
        let prevBefore = prevCount > 0 ? Int(session.history[prevCount - 1]) : nil
        session.history.append(Int64(token)); drafts.forEach { session.history.append(Int64($0)) }
        if engine.residentKVHistory.count == prevCount {
            engine.residentKVHistory.append(Int64(token)); drafts.forEach { engine.residentKVHistory.append(Int64($0)) }
        }
        engine.highWaterPos = max(engine.highWaterPos, session.history.count)

        let margin: Float = (usedPLD || !isGreedy || !isUnconstrained) ? 0 : session.relaxedMargin
        _ = try vStack.forwardRows(
            history: session.history, embedTable: engine.embedTable, head: engine.head,
            readStream: false, verifyAllRows: true, mtp: nil, activeRows: activeRows,
            relaxedMargin: margin, relaxedRows: margin > 0 ? activeRows : 0,
            relaxedMinProb: session.relaxedMinProb, dflash: dflash, preContextUpdate: usedPLD ? preCtx : nil)
        session.specProposals += dCount
        let (tok0, accepted) = try resolveVerifiedTokens(
            engine, session: &session, drafts: drafts, proposals: proposals, stream: stream,
            prevCount: prevCount, prevBefore: prevBefore, trackRep: trackRep, verifyMargin: margin)
        if accepted < dCount {
            let rejected = dCount - accepted
            session.history.removeLast(rejected)
            if engine.residentKVHistory.count == prevCount + activeRows { engine.residentKVHistory.removeLast(rejected) }
            vStack.rollbackVerifyRows(rejectedCount: rejected)
        }
        let roundMs = Date().timeIntervalSince(stepT0) * 1000.0
        session.specAccepted += accepted
        if !usedPLD { session.dflashPolicy.observeRound(draftCount: dCount, accepted: accepted, roundMs: roundMs) }
        session.dflashPendingCtxPos = prevCount; session.dflashPendingCtxCount = 1 + accepted
        session.dflashCacheOffset = prevCount + 1 + accepted; session.headToken = -1
        let divisor = Double(1 + session.pendingTokens.count)
        session.stepMs = roundMs / divisor; session.lastStepGPUMillis = (draftGPUMs + vStack.lastGPUMillis) / divisor
        session.roundTokenCounts.append(1)
        return tok0
    }

    private static func resolveVerifiedTokens(
        _ engine: FNGPUEngine, session: inout FNGPUSessionData,
        drafts: [Int], proposals: [QWMTPGumbelProposal], stream: QWMTPRandomStream?,
        prevCount: Int, prevBefore: Int?, trackRep: Bool, verifyMargin: Float
    ) throws -> (tok0: Int, accepted: Int) {
        let dCount = drafts.count
        if !proposals.isEmpty, let stream, (session.constraint?.isUnconstrainedNow ?? true) {
            let logitsPtr = engine.arena.cpuPointer(of: engine.head.logits).assumingMemoryBound(to: Float.self)
            var targets: [QWSamplingDistribution] = []
            for row in 0..<(1 + dCount) {
                let rowPtr = UnsafeBufferPointer(start: logitsPtr.advanced(by: row * engine.vocab), count: engine.vocab)
                let rep = trackRep
                    ? FNGPUSessionData.loopRepetitionTokens(
                        in: Array((session.recentGeneratedTokens + drafts.prefix(row))
                            .suffix(FNGPUSessionData.repetitionPenaltyWindow)))
                    : []
                targets.append(try session.sampler.distribution(logits: rowPtr, repetitionHistory: rep))
            }
            let res = session.relaxedMargin > 0
                ? try stream.resolveRelaxedGumbelRound(proposals: proposals, targetDistributions: targets,
                                                       margin: Double(session.relaxedMargin))
                : try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets)
            session.dflashStreamState = try stream.frozenState()
            let boundaryTok = session.constraint?.speculationBoundaryToken
            var committedAccepted = res.accepted
            var finalTok = res.finalToken
            if let constraint = session.constraint {
                for idx in 0..<res.accepted {
                    let tok = drafts[idx]
                    try constraint.accept(tok)
                    if tok == boundaryTok || !constraint.isUnconstrainedNow {
                        committedAccepted = idx
                        finalTok = tok
                        break
                    }
                }
                if committedAccepted == res.accepted {
                    try constraint.accept(finalTok)
                }
            }
            if committedAccepted == 0 {
                if trackRep { session.recordSeenToken(finalTok, vocab: engine.vocab) }
                return (finalTok, 0)
            }
            let tok0 = drafts[0]
            if trackRep { session.recordSeenToken(tok0, vocab: engine.vocab) }
            for i in 1..<committedAccepted {
                session.pendingTokens.append(drafts[i]); if trackRep { session.recordSeenToken(drafts[i], vocab: engine.vocab) }
            }
            session.pendingTokens.append(finalTok)
            if trackRep { session.recordSeenToken(finalTok, vocab: engine.vocab) }
            return (tok0, committedAccepted)
        }
        let v = engine.head.argmaxTokens(count: 1 + dCount)
        if verifyMargin == 0 && session.sampler.config.isGreedy && (session.constraint?.isUnconstrainedNow ?? true) {
            for row in (0..<(1 + dCount)).reversed() {
                let prev = (row == 0) ? prevBefore : Int(session.history[prevCount + row - 1])
                session.tokenRecycler.recordTransition(previous: prev, current: Int(session.history[prevCount + row]), next: v[row])
            }
        }
        var tok0 = 0, accepted = 0
        let boundaryTok = session.constraint?.speculationBoundaryToken
        for r in 0..<dCount {
            let draftTok = drafts[r], sampled: Int
            if v[r] == draftTok && session.sampler.config.isGreedy && !session.seenTokens.contains(draftTok) && (session.constraint?.isUnconstrainedNow ?? true) {
                try session.constraint?.accept(draftTok); sampled = draftTok
            } else {
                sampled = try engine.sampleFromHead(argmaxToken: v[r], seenTokens: session.seenTokens,
                                                    seenTokenList: session.seenTokenList,
                                                    sampler: session.sampler, row: r, constraint: session.constraint)
            }
            if r == 0 { tok0 = sampled } else { session.pendingTokens.append(sampled) }
            let hitBoundary = (boundaryTok != nil && sampled == boundaryTok) || !(session.constraint?.isUnconstrainedNow ?? true)
            if sampled == draftTok && v[r] == draftTok && !hitBoundary {
                accepted += 1; if trackRep { session.recordSeenToken(draftTok, vocab: engine.vocab) }
            } else {
                if trackRep { session.recordSeenToken(sampled, vocab: engine.vocab) }; break
            }
        }
        if accepted == dCount {
            let bonus = try engine.sampleFromHead(argmaxToken: v[dCount], seenTokens: session.seenTokens,
                                                  seenTokenList: session.seenTokenList,
                                                  sampler: session.sampler, row: dCount, constraint: session.constraint)
            if trackRep { session.recordSeenToken(bonus, vocab: engine.vocab) }
            session.pendingTokens.append(bonus)
        }
        return (tok0, accepted)
    }
}
