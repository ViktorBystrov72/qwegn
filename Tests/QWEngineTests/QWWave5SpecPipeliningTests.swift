// QWWave5SpecPipeliningTests.swift — Tests for O(1) FNNgramRowCache LRU, session ring/schedule, and GDN stability.
import Dispatch
import Foundation
import Metal
import XCTest
@testable import QWEngine
@testable import QWFlashNext
import QWAlloc
import QWCore
import QWGDN
import QWHAL
import QWHead

final class QWWave5SpecPipeliningTests: XCTestCase {

    func testNgramRowCacheO1LRUEvictionAndPromoteOnHit() {
        let rowBytes = 16
        let cache = FNNgramRowCache(capacityBytes: 3 * rowBytes, rowBytes: rowBytes, maxStripes: 1)
        let r1: [Float] = [1, 1, 1, 1]
        let r2: [Float] = [2, 2, 2, 2]
        let r3: [Float] = [3, 3, 3, 3]
        let r4: [Float] = [4, 4, 4, 4]
        let r5: [Float] = [5, 5, 5, 5]

        cache.store(1, r1)
        cache.store(2, r2)
        cache.store(3, r3)

        // Duplicate store must not replace or re-charge budget.
        cache.store(2, [9, 9, 9, 9])
        XCTAssertEqual(cache.value(for: 2), r2)

        // Touch 1 so LRU victim order becomes: 3 (tail), 2, 1 (head).
        XCTAssertEqual(cache.value(for: 1), r1)

        // Insert 4 -> evicts LRU victim 3.
        cache.store(4, r4)
        XCTAssertNil(cache.value(for: 3))
        XCTAssertEqual(cache.value(for: 1), r1)
        XCTAssertEqual(cache.value(for: 2), r2)
        XCTAssertEqual(cache.value(for: 4), r4)

        // Insert 5 after touching order (1 -> 2 -> 4 -> 5) -> evicts 1 and reuses freed node index.
        cache.store(5, r5)
        XCTAssertNil(cache.value(for: 1))
        XCTAssertEqual(cache.value(for: 5), r5)
        XCTAssertEqual(cache.stripes[0].rows.count, 3)

        cache.clear()
        XCTAssertEqual(cache.stripes[0].rows.count, 0)
        XCTAssertEqual(cache.stripes[0].bytes, 0)
        XCTAssertEqual(cache.stripes[0].hits, 0)
        XCTAssertEqual(cache.stripes[0].misses, 0)
    }

    func testNgramRowCacheConcurrentStressMaintainsIntegrityAndBudget() {
        let rowBytes = 32
        let cache = FNNgramRowCache(capacityBytes: 64 * rowBytes, rowBytes: rowBytes, maxStripes: 4)
        DispatchQueue.concurrentPerform(iterations: 8) { worker in
            for step in 0..<250 {
                let id = (worker * 17 + step * 13) % 120
                let expected = [Float](repeating: Float(id), count: 8)
                if step % 3 == 0 {
                    cache.store(id, expected)
                } else if let got = cache.value(for: id) {
                    XCTAssertEqual(got, expected)
                }
            }
        }
        let maxStripeBytes = cache.budgetBytes / cache.stripes.count
        for stripe in cache.stripes {
            stripe.lock.lock()
            XCTAssertLessThanOrEqual(stripe.bytes, maxStripeBytes)
            XCTAssertEqual(stripe.bytes, stripe.rows.count * rowBytes)
            stripe.lock.unlock()
        }
    }

    func testSessionRecordSeenTokenSlidingWindowParity() {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1.2))
        var session = FNGPUSessionData(history: [1, 2, 3], sampler: sampler)
        var refWindow: [Int] = []
        for step in 0..<180 {
            let tok = (step % 5 == 0) ? 42 : ((step % 2 == 0) ? 7 : 8)
            session.recordSeenToken(tok, vocab: 128)
            refWindow.append(tok)
            if refWindow.count > FNGPUSessionData.repetitionPenaltyWindow {
                refWindow.removeFirst(refWindow.count - FNGPUSessionData.repetitionPenaltyWindow)
            }
            let expected = QWSampler.loopRepetitionTokens(in: refWindow)
            XCTAssertEqual(session.recentGeneratedTokens, refWindow)
            XCTAssertEqual(session.seenTokenList, expected)
            XCTAssertEqual(session.seenTokens, Set(expected))
        }
    }

    func testDecodeStackSchedulePLECutsCollapseAndPreserveDefaultWrapper() {
        let legacyCuts = FNGPUDecodeStack.decodeCuts(slotCount: 16)
        XCTAssertEqual(legacyCuts, [0, 1, 8, 16])
        XCTAssertEqual(QWLayerStackSchedule.decodeCuts(slotCount: 16, hasPLE: true), legacyCuts)
        XCTAssertEqual(QWLayerStackSchedule.decodeCuts(slotCount: 16, hasPLE: false), [0, 16])
    }

    func testGDNRollbackStabilityVectorDecaySIMD8AndFloat16Bitmask() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
            "GPU checkpoint tests require local Apple Silicon Metal device"
        )
        let base = QWModelDescriptor.qwen38_27b
        let vecDesc = QWModelDescriptor(
            family: base.family, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visPosEmbeds: base.visPosEmbeds, gdnVectorDecay: true, role: base.role
        )
        let vecGeo = QWModelGeometry(descriptor: vecDesc)
        let hal = try QWHal(geometry: vecGeo)
        let layers = 1, rows = 4
        let cpBytes = QWGDNReplayCheckpoint.allocationBytes(geometry: vecGeo, layers: layers, rows: rows)
        let recElems = QWSlotGeometry.recElems(geometry: vecGeo)
        let ringElems = QWSlotGeometry.ringElems(geometry: vecGeo)
        let arena = try QWArena(hal: hal, weights: 256, activations: 256, kv: 256,
                                gdnStates: cpBytes + recElems * 4 + ringElems * 2 + 8192)
        let g = arena.gdnRegion()
        let cp = try QWGDNReplayCheckpoint(geometry: vecGeo, allocator: g, layers: layers, rows: rows)
        cp.sourceRec = try g.allocate(QWTensorShape([recElems], .f32))
        cp.sourceRing = try g.allocate(QWTensorShape([ringElems], .f16))
        cp.captured[0] = UInt32((1 << rows) - 1)
        memset(cp.normalized.region.pointer.advanced(by: cp.normalized.offset), 0, cp.normalized.bytes)
        memset(cp.innovations.region.pointer.advanced(by: cp.innovations.offset), 0, cp.innovations.bytes)
        let glogPtr = cp.glog.region.pointer.advanced(by: cp.glog.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<cp.glog.shape.elementCount { glogPtr[i] = 1.0 }

        XCTAssertTrue(try cp.rollbackTapeIsStable(retaining: 2, activeRows: rows))

        // Corrupt lane 7 of final SIMD8 chunk in tail row -> must detect instability.
        let gdnHD = vecGeo.gdnHeadDim
        let tailIdx = 3 * vecGeo.gdnGlogChannels + (gdnHD - 1)
        glogPtr[tailIdx] = 5.0e-4
        XCTAssertFalse(try cp.rollbackTapeIsStable(retaining: 2, activeRows: rows))

        // Restore decay, then verify Float16 Inf (0x7C00) in K-slice returns false.
        glogPtr[tailIdx] = 1.0
        XCTAssertTrue(try cp.rollbackTapeIsStable(retaining: 2, activeRows: rows))
        let normPtr = cp.normalized.region.pointer.advanced(by: cp.normalized.offset).assumingMemoryBound(to: UInt16.self)
        let kOffset = vecGeo.gdnQKHeads * gdnHD
        normPtr[3 * vecGeo.gdnQKVChannels + kOffset + gdnHD - 1] = 0x7C00
        XCTAssertFalse(try cp.rollbackTapeIsStable(retaining: 2, activeRows: rows))

        // Zero-layer checkpoint backup & skip-ring restore must succeed without allocations.
        let zCp = try QWGDNReplayCheckpoint(geometry: vecGeo, allocator: g, layers: 0, rows: rows)
        let dummyRec = try g.allocate(QWTensorShape([1], .f32))
        let dummyRing = try g.allocate(QWTensorShape([1], .f16))
        try zCp.markZeroLayerBackup(rec: dummyRec, ring: dummyRing)
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
        try zCp.encodeRestore(blit, rec: dummyRec, ring: dummyRing, restoreRing: false)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
    }
}
