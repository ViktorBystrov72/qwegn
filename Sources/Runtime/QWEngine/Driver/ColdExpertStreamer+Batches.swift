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
        /// Promotes every staged miss into the least-recently-used bank slot the current span does not
        /// read (stamp < now), mirrors the residency to the GPU, and builds the SSD load batches.
        /// Initial stamps are reverse-ordered by calibration rank so cold tail slots evict first.
        func promoteAndBatch(_ stagedItems: [(expert: Int, slot: Int)], now: UInt64) throws -> (
            [(name: String, dst: UnsafeMutableRawPointer)],
            [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            for item in stagedItems {
                var victim = -1
                var best = UInt64.max
                for s in 0..<residentExperts where slotStamp[s] < now && slotStamp[s] < best {
                    best = slotStamp[s]
                    victim = s
                }
                guard victim >= 0 else { break }
                let old = Int(expertInSlot[victim])
                if old >= 0 { slotOfExpert[old] = -1 }
                expertInSlot[victim] = Int32(item.expert)
                slotOfExpert[item.expert] = Int32(victim)
                slotStamp[victim] = now
                pendingPromotions.append((staging: item.slot, bank: victim))
                if Self.verifyPromotions { promotedForVerify.append((bank: victim, expert: item.expert)) }
            }
            if !stagedItems.isEmpty { syncGPUSlotMap() }
            return try makeBatches(stagedItems)
        }

        func makeBatches(_ stagedItems: [(expert: Int, slot: Int)]) throws -> (
            [(name: String, dst: UnsafeMutableRawPointer)],
            [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            var guBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            var downBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            guard !stagedItems.isEmpty else { return (guBatch, downBatch) }
            let guExpertBytes = self.guExpertBytes
            let singleGuBytes = intermediate * guRowBytes
            let downExpertBytes = self.downExpertBytes
            let allowMissingWeights = io.pack.tensors.count < totalExperts
            guBatch.reserveCapacity(stagedItems.count * 2)
            downBatch.reserveCapacity(stagedItems.count)
            for item in stagedItems {
                let logicalE = item.expert
                let e = (expertPermutation != nil && logicalE < expertPermutation!.count)
                    ? expertPermutation![logicalE]
                    : logicalE
                let coldSlot = item.slot
                let guDst = guBasePtr.advanced(by: coldSlot * guExpertBytes)
                let upDst = guDst.advanced(by: singleGuBytes)
                let downDst = downBasePtr.advanced(by: coldSlot * downExpertBytes)
                let gName = layerPrefix + "mlp.experts.\(e).gate_proj.weight"
                let uName = layerPrefix + "mlp.experts.\(e).up_proj.weight"
                let dName = layerPrefix + "mlp.experts.\(e).down_proj.weight"
                if (try? io.pack.entry(gName)) != nil {
                    guBatch.append((name: gName, dst: guDst))
                    guBatch.append((name: uName, dst: upDst))
                    downBatch.append((name: dName, dst: downDst))
                } else if allowMissingWeights {
                    memset(guDst, 0, guExpertBytes)
                    memset(downDst, 0, downExpertBytes)
                } else {
                    throw QWEngineDriverError.weightMissing(gName, "cold expert missing from QWP pack")
                }
            }
            return (guBatch, downBatch)
        }

        /// Finds the longest prefix `startRow ..< nextRow` (at least 1 row) whose distinct
        /// cold experts (`e >= residentExperts`) fit in `stagingSlots`, rewrites `idxPtr` in place,
        /// and prepares the `PlaneIO` gate/up and down load batches without executing the synchronous SSD read yet.
        func prepareSpan(
            idxPtr: UnsafeMutablePointer<UInt32>,
            startRow: Int, totalRows: Int, routes: Int
        ) throws -> (
            nextRow: Int,
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            guard stagingSlots > 0, startRow < totalRows else {
                return (totalRows, 0, [], [])
            }
            if isDynamic {
                return try prepareSpanDynamic(idxPtr: idxPtr, startRow: startRow,
                                              totalRows: totalRows, routes: routes)
            }
            let resU32 = UInt32(residentExperts)
            let isVerifyWindow = totalRows >= 2 && totalRows <= 4
            let effectiveColdSlots = (isVerifyWindow && Self.draftColdExpertBudget != nil)
                ? max(1, min(stagingSlots, Self.draftColdExpertBudget!))
                : stagingSlots
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
                return (totalRows, 0, [], [])
            }
            var slotForExpert: [Int: Int] = [:]
            var stagedCount = 0
            let guExpertBytes = 2 * intermediate * guRowBytes
            let singleGuBytes = intermediate * guRowBytes
            let downExpertBytes = hidden * downRowBytes
            let allowMissingWeights = io.pack.tensors.count < totalExperts

            var r = firstColdRow
            var stagedItems: [(expert: Int, slot: Int)] = []
            while r < totalRows {
                let rowBase = r * routes
                var neededForRow = 0
                for k in 0..<routes {
                    let raw = idxPtr[rowBase + k]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let e = Int(raw)
                    guard e >= residentExperts, slotForExpert[e] == nil else { continue }
                    var seenEarlierInRow = false
                    for prev in 0..<k where Int(idxPtr[rowBase + prev]) == e {
                        seenEarlierInRow = true
                        break
                    }
                    if !seenEarlierInRow {
                        neededForRow += 1
                    }
                }
                if r > startRow && stagedCount + neededForRow > effectiveColdSlots && !(isVerifyWindow && Self.draftColdExpertBudget != nil) {
                    break
                }
                for k in 0..<routes {
                    let i = rowBase + k
                    let raw = idxPtr[i]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let e = Int(raw)
                    guard e >= residentExperts else { continue }
                    if let existingSlot = slotForExpert[e] {
                        idxPtr[i] = UInt32(existingSlot)
                        continue
                    }
                    let slotCap = (r == 0) ? stagingSlots : effectiveColdSlots
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
            var guBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            var downBatch: [(name: String, dst: UnsafeMutableRawPointer)] = []
            if !stagedItems.isEmpty {
                guBatch.reserveCapacity(stagedItems.count * 2)
                downBatch.reserveCapacity(stagedItems.count)
                for item in stagedItems {
                    let logicalE = item.expert
                    let e = (expertPermutation != nil && logicalE < expertPermutation!.count)
                        ? expertPermutation![logicalE]
                        : logicalE
                    let coldSlot = item.slot
                    let guDst = guBasePtr.advanced(by: coldSlot * guExpertBytes)
                    let upDst = guDst.advanced(by: singleGuBytes)
                    let downDst = downBasePtr.advanced(by: coldSlot * downExpertBytes)
                    let gName = layerPrefix + "mlp.experts.\(e).gate_proj.weight"
                    let uName = layerPrefix + "mlp.experts.\(e).up_proj.weight"
                    let dName = layerPrefix + "mlp.experts.\(e).down_proj.weight"
                    if (try? io.pack.entry(gName)) != nil {
                        guBatch.append((name: gName, dst: guDst))
                        guBatch.append((name: uName, dst: upDst))
                        downBatch.append((name: dName, dst: downDst))
                    } else if allowMissingWeights {
                        memset(guDst, 0, guExpertBytes)
                        memset(downDst, 0, downExpertBytes)
                    } else {
                        throw QWEngineDriverError.weightMissing(gName, "cold expert missing from QWP pack")
                    }
                }
            }
            return (r, stagedCount, guBatch, downBatch)
        }

        func executeBatch(_ batch: [(name: String, dst: UnsafeMutableRawPointer)]) throws {
            guard !batch.isEmpty else { return }
            try io.loadBatch(batch, layout: .hq4)
        }

        func executeTwoPhaseBatch(
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            onGateUpLoaded: (@Sendable () throws -> Void)? = nil
        ) throws {
            guard !guBatch.isEmpty || !downBatch.isEmpty else { return }
            try io.loadBatchTwoPhase(
                firstPhase: guBatch,
                secondPhase: downBatch,
                layout: .hq4,
                onFirstPhaseComplete: onGateUpLoaded
            )
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
