// ColdExpertStreamer.swift — Demand-paged cold MoE expert slot LRU & staging.

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

extension QWEngineDriver {
    final class ColdExpertStreamer: @unchecked Sendable {
        let io: PlaneIO
        let layerPrefix: String
        let residentExperts: Int
        let totalExperts: Int
        let stagingSlots: Int
        let intermediate: Int
        let hidden: Int
        let guRowBytes: Int
        let downRowBytes: Int
        let guBasePtr: UnsafeMutableRawPointer
        let downBasePtr: UnsafeMutableRawPointer
        let expertPermutation: [Int]?

        init(io: PlaneIO, layerPrefix: String, residentExperts: Int, totalExperts: Int,
             stagingSlots: Int, intermediate: Int, hidden: Int,
             guRowBytes: Int, downRowBytes: Int,
             guBasePtr: UnsafeMutableRawPointer, downBasePtr: UnsafeMutableRawPointer,
             expertPermutation: [Int]? = nil) {
            self.io = io
            self.layerPrefix = layerPrefix
            self.residentExperts = residentExperts
            self.totalExperts = totalExperts
            self.stagingSlots = stagingSlots
            self.intermediate = intermediate
            self.hidden = hidden
            self.guRowBytes = guRowBytes
            self.downRowBytes = downRowBytes
            self.guBasePtr = guBasePtr
            self.downBasePtr = downBasePtr
            self.expertPermutation = expertPermutation
        }

        // ── Dynamic residency (trunk layers only) ──────────────────────────
        // The K resident bank slots form a per-layer LRU cache over all
        // experts instead of a fixed calibration hot set. A miss is staged as
        // before (idx = K + stagingSlot) and then promoted into the LRU victim
        // slot by a GPU copy after the layer's cold compute, so the next token
        // that routes to it hits. Router ids stay physical (0..<totalExperts);
        // only the CPU rewrite maps them to bank slots.
        private(set) var isDynamic = false
        var slotOfExpert: [Int32] = []
        var expertInSlot: [Int32] = []
        var slotStamp: [UInt64] = []
        var clock: UInt64 = 0
        /// (staging slot, bank slot) copies to encode after the cold compute.
        var pendingPromotions: [(staging: Int, bank: Int)] = []
        static let verifyPromotions = ProcessInfo.processInfo.environment["QW_EXPERT_CACHE_VERIFY"] == "1"
        /// Optional MoE-Spec (arXiv:2602.16052) cap on unique cold experts staged per verify pass (`totalRows` in 2...4).
        static let draftColdExpertBudget: Int? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_COLD_BUDGET"].flatMap(Int.init)
        var promotedForVerify: [(bank: Int, expert: Int)] = []
        private(set) var dynamicHits = 0
        private(set) var dynamicMisses = 0

        /// Host-written mirror of `slotOfExpert` (Int32 per physical expert) for GPU-side id mapping.
        private(set) var gpuSlotMap: (any MTLBuffer)?

        func attachGPUSlotMap(device: any MTLDevice) -> (any MTLBuffer)? {
            guard isDynamic else { return nil }
            if gpuSlotMap == nil {
                gpuSlotMap = device.makeBuffer(length: 4 * totalExperts, options: .storageModeShared)
                syncGPUSlotMap()
            }
            return gpuSlotMap
        }

        func syncGPUSlotMap() {
            guard let map = gpuSlotMap else { return }
            slotOfExpert.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress, !raw.isEmpty else { return }
                map.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }

        func enableDynamicResidency() {
            guard !isDynamic, stagingSlots > 0, residentExperts > 0 else { return }
            slotOfExpert = [Int32](repeating: -1, count: totalExperts)
            expertInSlot = [Int32](repeating: -1, count: residentExperts)
            slotStamp = [UInt64](repeating: 0, count: residentExperts)
            for s in 0..<residentExperts {
                slotOfExpert[s] = Int32(s)
                expertInSlot[s] = Int32(s)
                slotStamp[s] = UInt64(residentExperts - s)
            }
            clock = UInt64(residentExperts)
            isDynamic = true
        }

        var guExpertBytes: Int { 2 * intermediate * guRowBytes }

        /// Debug: byte-compare promoted bank slots against the pack (QW_EXPERT_CACHE_VERIFY=1).
        /// Call only when no GPU work is in flight.
        func verifyPromoted(guBank: UnsafeMutableRawPointer, downBank: UnsafeMutableRawPointer) throws -> Int {
            defer { promotedForVerify.removeAll(keepingCapacity: true) }
            var bad = 0
            let gb = guExpertBytes, db = downExpertBytes, single = intermediate * guRowBytes
            var tmp: UnsafeMutableRawPointer?
            posix_memalign(&tmp, 16_384, gb + db)
            guard let tmp else { return 0 }
            defer { free(tmp) }
            for (bank, p) in promotedForVerify {
                let e = (expertPermutation.flatMap { p < $0.count ? $0[p] : nil }) ?? p
                try io.loadBatch([
                    (name: layerPrefix + "mlp.experts.\(e).gate_proj.weight", dst: tmp),
                    (name: layerPrefix + "mlp.experts.\(e).up_proj.weight", dst: tmp.advanced(by: single)),
                    (name: layerPrefix + "mlp.experts.\(e).down_proj.weight", dst: tmp.advanced(by: gb)),
                ], layout: .hq4)
                if memcmp(tmp, guBank.advanced(by: bank * gb), gb) != 0
                    || memcmp(tmp.advanced(by: gb), downBank.advanced(by: bank * db), db) != 0 {
                    bad += 1
                }
            }
            return bad
        }
        var downExpertBytes: Int { hidden * downRowBytes }

        func prepareSpanDynamic(
            idxPtr: UnsafeMutablePointer<UInt32>,
            startRow: Int, totalRows: Int, routes: Int
        ) throws -> (
            nextRow: Int,
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            guard startRow < totalRows else { return (totalRows, 0, [], []) }
            clock &+= 1
            let now = clock
            let resU32 = UInt32(residentExperts)
            let isVerifyWindow = totalRows >= 2 && totalRows <= 4
            let effectiveColdSlots = isVerifyWindow
                ? (Self.draftColdExpertBudget.map { max(1, min(stagingSlots, $0)) } ?? stagingSlots)
                : stagingSlots
            var stagedFor: [Int: UInt32] = [:]
            var stagedItems: [(expert: Int, slot: Int)] = []
            var r = startRow
            while r < totalRows {
                let rowBase = r * routes
                var need = 0
                for k in 0..<routes {
                    let raw = idxPtr[rowBase + k]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let p = Int(raw)
                    guard p < totalExperts, slotOfExpert[p] < 0, stagedFor[p] == nil else { continue }
                    var seen = false
                    for prev in 0..<k where Int(idxPtr[rowBase + prev]) == p { seen = true; break }
                    if !seen { need += 1 }
                }
                if r > startRow && stagedItems.count + need > effectiveColdSlots && !(isVerifyWindow && Self.draftColdExpertBudget != nil) { break }
                for k in 0..<routes {
                    let i = rowBase + k
                    let raw = idxPtr[i]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let p = Int(raw)
                    guard p < totalExperts else { continue }
                    let s = Int(slotOfExpert[p])
                    if s >= 0 {
                        idxPtr[i] = UInt32(s)
                        slotStamp[s] = now
                        dynamicHits += 1
                        continue
                    }
                    if let x = stagedFor[p] { idxPtr[i] = x; continue }
                    dynamicMisses += 1
                    let slotCap = (r == 0) ? stagingSlots : effectiveColdSlots
                    if stagedItems.count < slotCap {
                        let slot = stagedItems.count
                        let x = resU32 + UInt32(slot)
                        stagedFor[p] = x
                        stagedItems.append((expert: p, slot: slot))
                        idxPtr[i] = x
                    } else {
                        idxPtr[i] = 0xFFFF_FFFF
                    }
                }
                r += 1
            }
            let (gu, dn) = try promoteAndBatch(stagedItems, now: now)
            return (r, stagedItems.count, gu, dn)
        }

        /// Whether a span of `routes` ids can always be mapped by `qwg_moe_span_ids`: every
        /// non-resident expert fits a staging slot, so the span never splits.
        func canMapWholeSpan() -> Bool {
            isDynamic && totalExperts <= 512 && stagingSlots >= totalExperts - residentExperts
                && Self.draftColdExpertBudget == nil
        }

        /// Dynamic-residency bookkeeping for a span mapped on the GPU by `qwg_moe_span_ids` (`mapped[i]` = bank
        /// slot, `residentExperts` + staging slot, or -1) from the physical router `ids`: stamps hits, stages
        /// each miss into its GPU-chosen slot, then promotes and batches exactly like `prepareSpan`.
        func prepareSpanMapped(
            ids: UnsafePointer<UInt32>, mapped: UnsafePointer<Int32>, count: Int
        ) throws -> (
            stagedCount: Int,
            guBatch: [(name: String, dst: UnsafeMutableRawPointer)],
            downBatch: [(name: String, dst: UnsafeMutableRawPointer)]
        ) {
            precondition(canMapWholeSpan(), "span map needs a dynamic streamer whose staging holds every miss")
            clock &+= 1
            let now = clock
            let res = Int32(residentExperts)
            var expertOfSlot = [Int](repeating: -1, count: stagingSlots)
            var firstSeen = [Int](repeating: 0, count: stagingSlots)
            var staged = 0
            for i in 0..<count {
                let m = mapped[i]
                if m < 0 { continue }
                if m < res {
                    guard Int(ids[i]) < totalExperts, slotOfExpert[Int(ids[i])] == m else {
                        throw QWEngineDriverError.state("span map slot \(m) disagrees with host residency")
                    }
                    slotStamp[Int(m)] = now
                    dynamicHits += 1
                    continue
                }
                let slot = Int(m - res)
                guard slot < stagingSlots else {
                    throw QWEngineDriverError.state("span map staging slot \(slot) out of \(stagingSlots)")
                }
                if expertOfSlot[slot] < 0 {
                    expertOfSlot[slot] = Int(ids[i])
                    firstSeen[slot] = i
                    staged += 1
                    dynamicMisses += 1
                } else if expertOfSlot[slot] != Int(ids[i]) {
                    throw QWEngineDriverError.state("span map gives staging slot \(slot) two experts")
                }
            }
            var stagedItems: [(expert: Int, slot: Int)] = []
            stagedItems.reserveCapacity(staged)
            for slot in 0..<staged {
                guard expertOfSlot[slot] >= 0 else {
                    throw QWEngineDriverError.state("span map staging slots are not dense at \(slot)")
                }
                stagedItems.append((expert: expertOfSlot[slot], slot: slot))
            }
            // Promote in route first-appearance order, as `prepareSpanDynamic` does, so the
            // post-span residency (and thus decode) is identical to the host-mapped path.
            stagedItems.sort { firstSeen[$0.slot] < firstSeen[$1.slot] }
            let (gu, dn) = try promoteAndBatch(stagedItems, now: now)
            return (staged, gu, dn)
        }

    }
}
