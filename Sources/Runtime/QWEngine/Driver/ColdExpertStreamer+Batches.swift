// ColdExpertStreamer+Batches.swift — Async two-phase DispatchIO batching for cold experts.

import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant
import QWLoad

extension QWEngineDriver.ColdExpertStreamer {
    /// Plans LRU promotions for staged misses without mutating residency or GPU slot map until I/O succeeds.
    func promoteAndBatch(
        _ stagedItems: [(expert: Int, slot: Int)],
        now: UInt64,
        hitSlots: [Int] = [],
        hitsDelta: Int = 0,
        missesDelta: Int = 0
    ) throws -> (ColdLoadBatch, ColdLoadBatch, PromotionTransaction) {
        let (guBatch, downBatch) = try makeBatches(stagedItems)
        var simStamps = slotStamp
        for s in hitSlots where (0..<residentExperts).contains(s) {
            simStamps[s] = now
        }
        var promotions: [(expert: Int, staging: Int, bank: Int)] = []
        promotions.reserveCapacity(stagedItems.count)
        for item in stagedItems {
            var victim = -1
            var best = UInt64.max
            for s in 0..<residentExperts where simStamps[s] < now && simStamps[s] < best {
                best = simStamps[s]
                victim = s
            }
            guard victim >= 0 else { break }
            simStamps[victim] = now
            promotions.append((expert: item.expert, staging: item.slot, bank: victim))
        }
        let tx = PromotionTransaction(
            clock: now,
            dynamicHitsDelta: hitsDelta,
            dynamicMissesDelta: missesDelta,
            hitSlots: hitSlots,
            promotions: promotions
        )
        return (guBatch, downBatch, tx)
    }

    /// Commits a planned promotion transaction to host LRU state and the GPU slot map after I/O succeeds.
    func commitPromotion(_ tx: PromotionTransaction, recordPendingPromotions: Bool = true) {
        guard isDynamic else { return }
        clock = max(clock, tx.clock)
        dynamicHits += tx.dynamicHitsDelta
        dynamicMisses += tx.dynamicMissesDelta
        for s in tx.hitSlots where (0..<residentExperts).contains(s) {
            slotStamp[s] = tx.clock
        }
        for p in tx.promotions {
            let old = Int(expertInSlot[p.bank])
            if old >= 0 { slotOfExpert[old] = -1 }
            expertInSlot[p.bank] = Int32(p.expert)
            slotOfExpert[p.expert] = Int32(p.bank)
            slotStamp[p.bank] = tx.clock
            if recordPendingPromotions {
                pendingPromotions.append((staging: p.staging, bank: p.bank))
            }
            if Self.verifyPromotions {
                promotedForVerify.append((bank: p.bank, expert: p.expert))
            }
        }
        if !tx.promotions.isEmpty {
            syncGPUSlotMap()
        }
    }

    private func physicalExpertIndex(_ logicalE: Int) -> Int {
        if let perm = expertPermutation, logicalE < perm.count {
            return perm[logicalE]
        }
        return logicalE
    }

    private func resolveFusedSlice(
        entryName: String,
        expert e: Int,
        expectedBytes: Int,
        expectedRows: Int
    ) throws -> (offset: Int, count: Int) {
        let entry = try io.pack.entry(entryName)
        guard entry.rawOff == nil, let baseOff = entry.hq4Off, baseOff >= 0 else {
            throw QWEngineDriverError.weightMissing(entryName, "cold 3D expert tensor needs hq4_off plane")
        }
        let perExpertBytes: Int
        if let eHq4 = entry.eHq4 {
            perExpertBytes = eHq4
        } else if entry.shape.count == 3 {
            perExpertBytes = expectedBytes
        } else {
            throw QWEngineDriverError.weightMissing(entryName, "cold 3D expert tensor missing e_hq4 metadata")
        }
        let maxExperts = entry.experts ?? (entry.shape.count == 3 ? entry.shape[0] : totalExperts)
        guard perExpertBytes == expectedBytes, e >= 0, e < maxExperts,
              entry.shape.count != 3 || entry.shape[1] == expectedRows else {
            throw QWEngineDriverError.weightMissing(entryName, "cold 3D expert \(e) layout mismatch")
        }
        let (relOff, ov0) = e.multipliedReportingOverflow(by: perExpertBytes)
        let (absOff, ov1) = baseOff.addingReportingOverflow(relOff)
        guard !ov0, !ov1 else {
            throw QWEngineDriverError.weightMissing(entryName, "cold 3D expert \(e) offset overflow")
        }
        return (absOff, perExpertBytes)
    }

    func makeBatches(
        _ stagedItems: [(expert: Int, slot: Int)],
        guBase: UnsafeMutableRawPointer? = nil,
        downBase: UnsafeMutableRawPointer? = nil
    ) throws -> (ColdLoadBatch, ColdLoadBatch) {
        guard !stagedItems.isEmpty else { return (.named([]), .named([])) }
        let guPtr = guBase ?? guBasePtr
        let dnPtr = downBase ?? downBasePtr
        let guExpertBytes = self.guExpertBytes
        let singleGuBytes = intermediate * guRowBytes
        let downExpertBytes = self.downExpertBytes

        switch expertLayout {
        case .fused3D(let guName, let downName):
            var guItems: [RangedItem] = []
            var dnItems: [RangedItem] = []
            guItems.reserveCapacity(stagedItems.count)
            dnItems.reserveCapacity(stagedItems.count)
            for item in stagedItems {
                let e = physicalExpertIndex(item.expert)
                let guDst = guPtr.advanced(by: item.slot * guExpertBytes)
                let dnDst = dnPtr.advanced(by: item.slot * downExpertBytes)
                let (guOff, guCount) = try resolveFusedSlice(
                    entryName: guName, expert: e, expectedBytes: guExpertBytes, expectedRows: 2 * intermediate)
                let (dnOff, dnCount) = try resolveFusedSlice(
                    entryName: downName, expert: e, expectedBytes: downExpertBytes, expectedRows: hidden)
                guItems.append(RangedItem(name: guName, offset: guOff, count: guCount, dst: guDst))
                dnItems.append(RangedItem(name: downName, offset: dnOff, count: dnCount, dst: dnDst))
            }
            return (.ranged(guItems), .ranged(dnItems))

        case .mlpPerExpert, .blockSparsePerExpert:
            var guBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            var downBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            guBatch.reserveCapacity(stagedItems.count * 2)
            downBatch.reserveCapacity(stagedItems.count)
            let isBlockSparse = {
                if case .blockSparsePerExpert = expertLayout { return true }
                return false
            }()
            for item in stagedItems {
                let e = physicalExpertIndex(item.expert)
                let guDst = guPtr.advanced(by: item.slot * guExpertBytes)
                let upDst = guDst.advanced(by: singleGuBytes)
                let dnDst = dnPtr.advanced(by: item.slot * downExpertBytes)
                let gName = layerPrefix + (isBlockSparse
                    ? "block_sparse_moe.experts.\(e).w1.weight"
                    : "mlp.experts.\(e).gate_proj.weight")
                let uName = layerPrefix + (isBlockSparse
                    ? "block_sparse_moe.experts.\(e).w3.weight"
                    : "mlp.experts.\(e).up_proj.weight")
                let dName = layerPrefix + (isBlockSparse
                    ? "block_sparse_moe.experts.\(e).w2.weight"
                    : "mlp.experts.\(e).down_proj.weight")
                for (name, expected) in [(gName, singleGuBytes), (uName, singleGuBytes), (dName, downExpertBytes)] {
                    let entry: QWQTensorEntry
                    do {
                        entry = try io.pack.entry(name)
                    } catch {
                        throw QWEngineDriverError.weightMissing(name, "cold expert missing from QWP pack")
                    }
                    guard entry.rawOff == nil, entry.hq4Off != nil,
                          let actual = PlaneLayout.hq4.byteCount(entry), actual == expected else {
                        throw QWEngineDriverError.weightMissing(name, "cold expert hq4 layout invalid")
                    }
                }
                guBatch.append((name: gName, dst: guDst))
                guBatch.append((name: uName, dst: upDst))
                downBatch.append((name: dName, dst: dnDst))
            }
            return (.named(guBatch), .named(downBatch))

        case .missing:
            throw QWEngineDriverError.weightMissing(
                layerPrefix + "mlp.experts", "cold expert tensors missing from QWP pack")
        }
    }

    /// Finds the longest prefix `startRow ..< nextRow` (at least 1 row) whose distinct
    /// cold experts (`e >= residentExperts`) fit in `stagingSlots`, rewrites `idxPtr` in place,
    /// and prepares the gate/up and down load batches without executing SSD I/O or committing LRU yet.
    func prepareSpan(
        idxPtr: UnsafeMutablePointer<UInt32>,
        startRow: Int, totalRows: Int, routes: Int,
        isVerifyWindow: Bool = false,
        specBudget: Int? = QWEngineDriver.ColdExpertStreamer.draftColdExpertBudget
    ) throws -> (
        nextRow: Int,
        stagedCount: Int,
        guBatch: ColdLoadBatch,
        downBatch: ColdLoadBatch,
        promotions: [(staging: Int, bank: Int)],
        transaction: PromotionTransaction
    ) {
        guard stagingSlots > 0, startRow < totalRows else {
            return (totalRows, 0, .named([]), .named([]), [], .empty)
        }
        if isDynamic {
            return try prepareSpanDynamic(
                idxPtr: idxPtr, startRow: startRow, totalRows: totalRows,
                routes: routes, isVerifyWindow: isVerifyWindow, specBudget: specBudget)
        }
        let resU32 = UInt32(residentExperts)
        let (applySpecBudget, effectiveColdSlots) = effectiveColdBudget(
            isVerifyWindow: isVerifyWindow, specBudget: specBudget)
        var firstColdRow = totalRows
        outerScan: for row in startRow..<totalRows {
            let base = row * routes
            for k in 0..<routes {
                let raw = idxPtr[base + k]
                if Int32(bitPattern: raw) >= 0 && raw >= resU32 {
                    firstColdRow = row
                    break outerScan
                }
            }
        }
        if firstColdRow == totalRows {
            return (totalRows, 0, .named([]), .named([]), [], .empty)
        }
        var slotForExpert: [Int: Int] = [:]
        var stagedCount = 0
        var r = firstColdRow
        var stagedItems: [(expert: Int, slot: Int)] = []
        while r < totalRows {
            let rowBase = r * routes
            var neededForRow = 0
            for k in 0..<routes {
                let raw = idxPtr[rowBase + k]
                if Int32(bitPattern: raw) < 0 { continue }
                let e = Int(raw)
                guard e >= residentExperts, e < totalExperts, slotForExpert[e] == nil else { continue }
                var seenEarlierInRow = false
                for prev in 0..<k where Int(idxPtr[rowBase + prev]) == e {
                    seenEarlierInRow = true
                    break
                }
                if !seenEarlierInRow {
                    neededForRow += 1
                }
            }
            if (r == firstColdRow || !applySpecBudget) && neededForRow > stagingSlots {
                throw QWEngineDriverError.state(
                    "cold expert staging capacity \(stagingSlots) < row \(r) demand \(neededForRow)")
            }
            if r > startRow && stagedCount + neededForRow > effectiveColdSlots && !applySpecBudget {
                break
            }
            for k in 0..<routes {
                let i = rowBase + k
                let raw = idxPtr[i]
                if Int32(bitPattern: raw) < 0 { continue }
                let e = Int(raw)
                guard e < totalExperts else {
                    idxPtr[i] = 0xFFFF_FFFF
                    continue
                }
                guard e >= residentExperts else { continue }
                if let existingSlot = slotForExpert[e] {
                    idxPtr[i] = UInt32(existingSlot)
                    continue
                }
                let slotCap = (r == startRow) ? stagingSlots : effectiveColdSlots
                guard stagedCount < slotCap else {
                    idxPtr[i] = 0xFFFF_FFFF
                    continue
                }
                let coldSlot = stagedCount
                let rewrittenIdx = residentExperts + coldSlot
                stagedCount += 1
                slotForExpert[e] = rewrittenIdx
                stagedItems.append((expert: e, slot: coldSlot))
                idxPtr[i] = UInt32(rewrittenIdx)
            }
            r += 1
        }
        let (guBatch, downBatch) = try makeBatches(stagedItems)
        return (r, stagedCount, guBatch, downBatch, [], .empty)
    }

    private func readRange(_ name: String, offset: Int, count: Int, into dst: UnsafeMutableRawPointer) throws {
        if let directStreamer {
            do {
                try directStreamer.streamTensor(offset: offset, byteCount: count, destination: dst)
                return
            } catch {
                // Fall back to verified mmap copy when direct DMA cannot satisfy offset/alignment.
            }
        }
        try io.pack.withRawPointer(at: offset, byteCount: count) { src in
            dst.copyMemory(from: src, byteCount: count)
        }
    }

    private func executeRangedTwoPhase(
        firstPhase: [RangedItem],
        secondPhase: [RangedItem],
        onFirstPhaseComplete: (@Sendable () throws -> Void)?
    ) throws {
        if firstPhase.isEmpty && secondPhase.isEmpty {
            try onFirstPhaseComplete?()
            return
        }
        final class PhaseBox: @unchecked Sendable {
            let lock = NSLock()
            var firstError: (any Error)?
            var remainingFirst: Int
            init(remainingFirst: Int) { self.remainingFirst = remainingFirst }
            func record(_ err: any Error) {
                lock.withLock { if firstError == nil { firstError = err } }
            }
            func completeFirst() -> Bool {
                lock.withLock {
                    remainingFirst -= 1
                    return remainingFirst == 0 && firstError == nil
                }
            }
        }
        let firstCount = firstPhase.count
        let jobs = firstPhase + secondPhase
        if jobs.count == 1 {
            let j = jobs[0]
            try readRange(j.name, offset: j.offset, count: j.count, into: j.dst)
            if firstCount == 1 { try onFirstPhaseComplete?() }
            return
        }
        let box = PhaseBox(remainingFirst: firstCount)
        if firstCount == 0 { try onFirstPhaseComplete?() }
        DispatchQueue.concurrentPerform(iterations: jobs.count) { idx in
            let j = jobs[idx]
            do {
                try readRange(j.name, offset: j.offset, count: j.count, into: j.dst)
            } catch {
                box.record(error)
            }
            if let onFirstPhaseComplete, idx < firstCount, box.completeFirst() {
                do {
                    try onFirstPhaseComplete()
                } catch {
                    box.record(error)
                }
            }
        }
        if let err = box.firstError { throw err }
    }

    func executeBatch(guBatch: ColdLoadBatch, downBatch: ColdLoadBatch) throws {
        try executeTwoPhaseBatch(guBatch: guBatch, downBatch: downBatch, onGateUpLoaded: nil)
    }

    func executeTwoPhaseBatch(
        guBatch: ColdLoadBatch,
        downBatch: ColdLoadBatch,
        onGateUpLoaded: (@Sendable () throws -> Void)? = nil
    ) throws {
        guard !guBatch.isEmpty || !downBatch.isEmpty else { return }
        switch (guBatch, downBatch) {
        case (.named(let gu), .named(let dn)):
            try io.loadBatchTwoPhase(
                firstPhase: gu,
                secondPhase: dn,
                layout: .hq4,
                onFirstPhaseComplete: onGateUpLoaded
            )
        case (.ranged(let gu), .ranged(let dn)):
            try executeRangedTwoPhase(
                firstPhase: gu,
                secondPhase: dn,
                onFirstPhaseComplete: onGateUpLoaded
            )
        default:
            throw QWEngineDriverError.state("mismatched cold expert batch representation")
        }
    }

    /// Immediately zeroes/unloads all `stagedCount` cold expert staging slots after the MoE down-projection.
    func unload(stagedCount: Int) {
        guard stagedCount > 0, !isDynamic else { return }
        let guExpertBytes = 2 * intermediate * guRowBytes
        let downExpertBytes = hidden * downRowBytes
        memset(guBasePtr, 0, stagedCount * guExpertBytes)
        memset(downBasePtr, 0, stagedCount * downExpertBytes)
    }
}
