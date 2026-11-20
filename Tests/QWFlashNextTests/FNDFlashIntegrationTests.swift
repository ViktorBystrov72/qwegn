// FNDFlashIntegrationTests.swift — Unit tests for DFlash prefill window math,
// residency/KV footprint estimation, prefix header flags, and hybrid PLD/recycler bypass.
import XCTest
import Foundation
import QWCore
@testable import QWFlashNext

final class FNDFlashIntegrationTests: XCTestCase {
    private static func makeFlashNextDescriptor(hidden: Int = 2560, numLayers: Int = 48) -> QWModelDescriptor {
        let types = (0..<numLayers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        return QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.mtp, .vision],
            hidden: hidden, vocab: 248320,
            denseIntermediate: nil, numLayers: numLayers,
            layerTypes: types, fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 262_144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 256, expertsPerTok: 8, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248044
        )
    }

    func testPrefillSlidingWindowCaptureMath() {
        // Prompt fits inside sliding window: capture entire chunk from localStart = 0.
        let small = FNGPUDFlash.prefillCaptureSlice(startPos: 0, rows: 512, historyCount: 512, maxCacheSize: 2047)
        XCTAssertEqual(small?.localStart, 0)
        XCTAssertEqual(small?.captureRows, 512)
        XCTAssertEqual(small?.ctxStartPos, 0)

        // Chunk ends before windowStart (5000 - 2047 = 2953): skip all DFlash kernels.
        XCTAssertNil(FNGPUDFlash.prefillCaptureSlice(startPos: 0, rows: 2048, historyCount: 5000, maxCacheSize: 2047))
        XCTAssertNil(FNGPUDFlash.prefillCaptureSlice(startPos: 0, rows: 2953, historyCount: 5000, maxCacheSize: 2047))

        // Chunk straddles windowStart = 2953: capture tail slice [905..<2048).
        let straddle = FNGPUDFlash.prefillCaptureSlice(startPos: 2048, rows: 2048, historyCount: 5000, maxCacheSize: 2047)
        XCTAssertEqual(straddle?.localStart, 905)
        XCTAssertEqual(straddle?.captureRows, 1143)
        XCTAssertEqual(straddle?.ctxStartPos, 2953)

        // Trailing chunk wholly inside window.
        let tail = FNGPUDFlash.prefillCaptureSlice(startPos: 4096, rows: 904, historyCount: 5000, maxCacheSize: 2047)
        XCTAssertEqual(tail?.localStart, 0)
        XCTAssertEqual(tail?.captureRows, 904)
        XCTAssertEqual(tail?.ctxStartPos, 4096)

        // Single huge chunk (> maxCacheSize) clamps to last maxCacheSize rows.
        let clamped = FNGPUDFlash.prefillCaptureSlice(startPos: 0, rows: 4096, historyCount: 4096, maxCacheSize: 2047)
        XCTAssertEqual(clamped?.localStart, 2049)
        XCTAssertEqual(clamped?.captureRows, 2047)
        XCTAssertEqual(clamped?.ctxStartPos, 2049)
    }

    func testResidencyAndPerTokenKVAdmissionDelta() {
        XCTAssertEqual(FNGPUDFlash.adjustedPerTokenKVBytes(31_296, usesDFlash: false), 31_296)
        XCTAssertEqual(FNGPUDFlash.adjustedPerTokenKVBytes(31_296, usesDFlash: true), 28_928)
        XCTAssertEqual(FNGPUDFlash.adjustedPerTokenKVBytes(18_096, usesDFlash: true), 16_744)
        XCTAssertEqual(FNGPUDFlash.adjustedPerTokenKVBytes(11_752, usesDFlash: true), 10_888)

        let desc = Self.makeFlashNextDescriptor()
        let off = FNGPUDFlash.estimateFootprint(
            targetDescriptor: desc, targetPackPath: nil, dflashPackPath: nil, isDFlash: false)
        XCTAssertEqual(off.nativeMTPWeightBytes, 0)
        XCTAssertEqual(off.dflashWeightBytes, 0)
        XCTAssertEqual(off.dflashScratchBytes, 0)

        let missingPath = "/tmp/nonexistent-dflash-\(UUID().uuidString)"
        let on = FNGPUDFlash.estimateFootprint(
            targetDescriptor: desc, targetPackPath: missingPath, dflashPackPath: missingPath, isDFlash: true)
        XCTAssertEqual(on.nativeMTPWeightBytes, FNGPUDFlash.defaultMTPPackSavingsBytes)
        XCTAssertGreaterThan(on.nativeMTPWeightBytes, on.dflashWeightBytes)
    }

    func testPrefixHeaderBitAndGeometryMarkerHashing() {
        XCTAssertEqual(FNGPUEngine.FNPrefixHeader.dflashBit, 4)
        let tokens: [Int64] = Array(10..<160)
        let baseHash = FNGPUEngine.fnPrefixHash(tokens: tokens[...], kvMode: .f16, hasMTP: false)
        let m1 = "df:2560:5:20:8:128:2047:0:5,15,25,35,45"
        let m2 = "df:2560:5:20:8:128:2047:0:1,12,23,34,45"
        let h1 = FNGPUEngine.fnPrefixHash(tokens: tokens[...], kvMode: .f16, hasMTP: false, dflashMarker: m1)
        let h2 = FNGPUEngine.fnPrefixHash(tokens: tokens[...], kvMode: .f16, hasMTP: false, dflashMarker: m2)
        XCTAssertNotEqual(baseHash, h1)
        XCTAssertNotEqual(h1, h2)
    }

    func testHybridPLDAndTokenRecyclerBypass() {
        let recycler = QWTokenRecycler()
        let periodicHistory: [Int64] = [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14]
        // Pure neural `.mtp` must never trigger PLD bypass.
        XCTAssertEqual(
            FNGPUDFlash.selectHybridBypassDrafts(
                history: periodicHistory, pendingToken: 10, mode: .mtp, maxDrafts: 7, recycler: recycler),
            []
        )
        // Periodic match (>= 3 repeats) bypasses propose across `.hybrid` and `.ngram`.
        let pldHybrid = FNGPUDFlash.selectHybridBypassDrafts(
            history: periodicHistory, pendingToken: 10, mode: .hybrid, maxDrafts: 7, recycler: recycler)
        XCTAssertEqual(pldHybrid, [11, 12, 13, 14, 10, 11, 12])

        // Context-copy suffix match + TokenRecycler extension skips propose when hitting minPld.
        recycler.ingestSequence([105, 100, 777])
        let copyHistory: [Int64] = [100, 101, 102, 103, 104, 105, 100, 101, 102, 103, 104, 105]
        let copyExt = FNGPUDFlash.selectHybridBypassDrafts(
            history: copyHistory, pendingToken: 100, mode: .hybrid, maxDrafts: 7, recycler: recycler)
        XCTAssertEqual(copyExt, [101, 102, 103, 104, 105, 100, 777])

        // Standalone TokenRecycler bigram bypass applies in `.ngram` while `.hybrid` defers to DFlash 2.
        recycler.ingestSequence([500, 501, 502, 503, 504, 505])
        let ngramHistory: [Int64] = [1, 2, 3, 500]
        let recDrafts = FNGPUDFlash.selectHybridBypassDrafts(
            history: ngramHistory, pendingToken: 501, mode: .ngram, maxDrafts: 4, recycler: recycler)
        XCTAssertEqual(recDrafts, [502, 503, 504, 505])
        XCTAssertEqual(
            FNGPUDFlash.selectHybridBypassDrafts(
                history: ngramHistory, pendingToken: 501, mode: .hybrid, maxDrafts: 7, recycler: recycler),
            []
        )

        // Unmatched novel sequence returns empty -> routes to DFlash 2 neural proposal.
        let novel: [Int64] = [1, 2, 3, 4, 5, 6, 7, 8]
        XCTAssertEqual(
            FNGPUDFlash.selectHybridBypassDrafts(
                history: novel, pendingToken: 99, mode: .hybrid, maxDrafts: 7, recycler: QWTokenRecycler()),
            []
        )
    }

    func testTargetEndCheckpointPrefillWindowSlice() {
        // Shared-base retainLen = 2048 within a 6000-token prompt (maxCacheSize = 2047):
        // checkpoint prefill must capture [1..<2048) instead of skipping against history.count = 6000.
        let winEnd = FNGPUDFlash.prefillWindowEnd(historyCount: 6000, targetEnd: 2048)
        XCTAssertEqual(winEnd, 2048)
        let ckpt = FNGPUDFlash.prefillCaptureSlice(
            startPos: 0, rows: 2048, historyCount: winEnd, maxCacheSize: 2047)
        XCTAssertEqual(ckpt?.localStart, 1)
        XCTAssertEqual(ckpt?.captureRows, 2047)
        XCTAssertEqual(ckpt?.ctxStartPos, 1)

        // Contrast: passing unclipped 6000 would have skipped the entire checkpoint chunk.
        XCTAssertNil(FNGPUDFlash.prefillCaptureSlice(
            startPos: 0, rows: 2048, historyCount: 6000, maxCacheSize: 2047))

        // Full prompt prefill (targetEnd == nil) keeps full history.count window.
        XCTAssertEqual(FNGPUDFlash.prefillWindowEnd(historyCount: 6000, targetEnd: nil), 6000)
        let tail = FNGPUDFlash.prefillCaptureSlice(
            startPos: 2048, rows: 3952, historyCount: 6000, maxCacheSize: 2047)
        XCTAssertEqual(tail?.localStart, 1905)
        XCTAssertEqual(tail?.captureRows, 2047)
        XCTAssertEqual(tail?.ctxStartPos, 3953)
    }

    func testWarmStoreSlotExtensionWithDFlashMarker() {
        let store = FNPrefixWarmStore(byteLimit: 8192, maxSlots: 4)
        let marker = "df:2560:5:20:8:128:2047:0:5,15,25,35,45"
        let otherMarker = "df:2560:5:20:8:128:2047:0:1,12,23,34,45"
        let n1 = FNPrefixWarmStore.rootPrefixTokenCount + 32
        let turn1: [Int64] = (10..<(10 + n1)).map { Int64($0) }
        let h1 = FNGPUEngine.fnPrefixHash(tokens: turn1[...], kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let name1 = String(format: "fn_entry_%016llx_%d.qwcache", h1, n1)
        let r1 = FNPrefixWarmStore.rootHash(history: turn1, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let slot1 = store.resolveSlot(rootHash: r1, history: turn1, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        store.prepareSlotForStore(slot: slot1, rootHash: r1, history: turn1, incomingBytes: 64,
                                  kvMode: .f16, hasMTP: false, dflashMarker: marker)
        store.store(name: name1, blob: Data(repeating: 1, count: 64),
                    slot: slot1, rootHash: r1, history: turn1, isSharedBase: false)

        // An unrelated prompt resolves to a distinct slot so slot reuse on extension is non-trivial.
        let otherPrompt: [Int64] = (5000..<(5000 + n1)).map { Int64($0) }
        let hOther = FNGPUEngine.fnPrefixHash(
            tokens: otherPrompt[...], kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let rOther = FNPrefixWarmStore.rootHash(
            history: otherPrompt, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let slotOther = store.resolveSlot(
            rootHash: rOther, history: otherPrompt, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        XCTAssertNotEqual(slotOther, slot1)
        store.store(name: String(format: "fn_entry_%016llx_%d.qwcache", hOther, n1),
                    blob: Data(repeating: 2, count: 64),
                    slot: slotOther, rootHash: rOther, history: otherPrompt, isSharedBase: false)

        // Multi-turn extension of turn1 resolves the same slot and matches only under matching dflashMarker.
        let turn2 = turn1 + [901, 902, 903, 904]
        XCTAssertEqual(
            store.lookupWarmHit(exactName: "", history: turn2, rawHistory: turn2,
                                kvMode: .f16, hasMTP: false, dflashMarker: marker, allowExact: false)?.value,
            Data(repeating: 1, count: 64))
        XCTAssertNil(store.lookupWarmHit(exactName: "", history: turn2, rawHistory: turn2,
                                         kvMode: .f16, hasMTP: false, dflashMarker: nil, allowExact: false))
        XCTAssertNil(store.lookupWarmHit(exactName: "", history: turn2, rawHistory: turn2,
                                         kvMode: .f16, hasMTP: false, dflashMarker: otherMarker, allowExact: false))

        let r2 = FNPrefixWarmStore.rootHash(history: turn2, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let slot2 = store.resolveSlot(rootHash: r2, history: turn2, kvMode: .f16, hasMTP: false, dflashMarker: marker)
        XCTAssertEqual(slot2, slot1)
        store.prepareSlotForStore(slot: slot2, rootHash: r2, history: turn2, incomingBytes: 64,
                                  kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let h2 = FNGPUEngine.fnPrefixHash(tokens: turn2[...], kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let name2 = String(format: "fn_entry_%016llx_%d.qwcache", h2, turn2.count)
        store.store(name: name2, blob: Data(repeating: 3, count: 64),
                    slot: slot2, rootHash: r2, history: turn2, isSharedBase: false)
        let turn3 = turn2 + [905]
        XCTAssertEqual(
            store.lookupWarmHit(exactName: "", history: turn3, rawHistory: turn3,
                                kvMode: .f16, hasMTP: false, dflashMarker: marker, allowExact: false)?.value,
            Data(repeating: 3, count: 64))

        // Shared-base matching also honors dflashMarker isolation.
        let baseHist: [Int64] = (8000..<8160).map { Int64($0) }
        let hBase = FNGPUEngine.fnPrefixHash(tokens: baseHist[...], kvMode: .f16, hasMTP: false, dflashMarker: marker)
        let baseName = String(format: "fn_entry_%016llx_%d.qwcache", hBase, baseHist.count)
        store.store(name: baseName, blob: Data(repeating: 4, count: 64),
                    slot: nil, rootHash: nil, history: baseHist, isSharedBase: true)
        let baseQuery = baseHist + [1]
        let baseHit = store.lookupWarmHit(exactName: "", history: baseQuery, rawHistory: baseQuery,
                                          kvMode: .f16, hasMTP: false, dflashMarker: marker, allowExact: false)
        XCTAssertEqual(baseHit?.value, Data(repeating: 4, count: 64))
        XCTAssertEqual(baseHit?.matchedTokens, baseHist.count)
        XCTAssertNil(store.lookupWarmHit(exactName: "", history: baseQuery, rawHistory: baseQuery,
                                         kvMode: .f16, hasMTP: false, dflashMarker: otherMarker, allowExact: false))
    }

    func testResolveDraftVocabDefaultsTo98304ForLargeVocabs() {
        XCTAssertEqual(FNGPUDFlash.resolveDraftVocab(fullVocab: 248_320), 98_304)
        XCTAssertNil(FNGPUDFlash.resolveDraftVocab(fullVocab: 32_768))
        XCTAssertNil(FNGPUDFlash.resolveDraftVocab(fullVocab: 98_304))
    }
}
