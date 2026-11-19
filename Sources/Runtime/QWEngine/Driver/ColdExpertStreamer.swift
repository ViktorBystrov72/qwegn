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
        enum ExpertWeightLayout: Sendable {
            case mlpPerExpert
            case blockSparsePerExpert
            case fused3D(guName: String, downName: String)
            case missing
        }

        struct RangedItem: @unchecked Sendable {
            let name: String
            let offset: Int
            let count: Int
            let dst: UnsafeMutableRawPointer
        }

        enum ColdLoadBatch: @unchecked Sendable {
            case named([(name: String, dst: UnsafeMutableRawPointer)])
            case ranged([RangedItem])

            var isEmpty: Bool {
                switch self {
                case .named(let items): return items.isEmpty
                case .ranged(let items): return items.isEmpty
                }
            }
        }

        struct PromotionTransaction: @unchecked Sendable {
            let clock: UInt64
            let dynamicHitsDelta: Int
            let dynamicMissesDelta: Int
            let hitSlots: [Int]
            let promotions: [(expert: Int, staging: Int, bank: Int)]

            static let empty = PromotionTransaction(
                clock: 0, dynamicHitsDelta: 0, dynamicMissesDelta: 0, hitSlots: [], promotions: []
            )
        }

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
        let expertLayout: ExpertWeightLayout
        let directStreamer: QWWeightStreamer?

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
            self.expertLayout = Self.resolveExpertLayout(
                pack: io.pack,
                layerPrefix: layerPrefix,
                totalExperts: totalExperts,
                expertPermutation: expertPermutation
            )
            self.directStreamer = io.policy.directIO
                ? (try? QWWeightStreamer(url: io.pack.url, unbuffered: true))
                : nil
        }

        private static func resolveExpertLayout(
            pack: QWQuant.Pack,
            layerPrefix: String,
            totalExperts: Int,
            expertPermutation: [Int]?
        ) -> ExpertWeightLayout {
            let fusedCandidates: [(String, String)] = [
                ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
                ("mlp.experts.gate_up_proj.weight", "mlp.experts.down_proj.weight"),
                ("block_sparse_moe.experts.gate_up_proj", "block_sparse_moe.experts.down_proj"),
                ("block_sparse_moe.experts.gate_up_proj.weight", "block_sparse_moe.experts.down_proj.weight"),
            ]
            for (guSuffix, dnSuffix) in fusedCandidates {
                let gu = layerPrefix + guSuffix
                let dn = layerPrefix + dnSuffix
                if (try? pack.entry(gu)) != nil, (try? pack.entry(dn)) != nil {
                    return .fused3D(guName: gu, downName: dn)
                }
            }
            let sampleE = (expertPermutation.flatMap { $0.first }) ?? 0
            if (try? pack.entry(layerPrefix + "block_sparse_moe.experts.\(sampleE).w1.weight")) != nil
                || pack.tensors.keys.contains(where: {
                    $0.hasPrefix(layerPrefix + "block_sparse_moe.experts.") && $0.hasSuffix(".w1.weight")
                }) {
                return .blockSparsePerExpert
            }
            if (try? pack.entry(layerPrefix + "mlp.experts.\(sampleE).gate_proj.weight")) != nil
                || pack.tensors.keys.contains(where: {
                    $0.hasPrefix(layerPrefix + "mlp.experts.") && $0.hasSuffix(".gate_proj.weight")
                }) {
                return .mlpPerExpert
            }
            if totalExperts > 0 {
                for key in pack.tensors.keys where key.hasPrefix(layerPrefix) {
                    if key.contains("block_sparse_moe.experts.") && key.hasSuffix(".w1.weight") {
                        return .blockSparsePerExpert
                    }
                    if key.contains("mlp.experts.") && key.hasSuffix(".gate_proj.weight") {
                        return .mlpPerExpert
                    }
                }
            }
            return .missing
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
        /// Optional MoE-Spec (arXiv:2602.16052) cap on unique cold experts staged per speculative verify pass.
        static let draftColdExpertBudget: Int? = ProcessInfo.processInfo.environment["QW_MOE_SPEC_COLD_BUDGET"].flatMap(Int.init)
        var promotedForVerify: [(bank: Int, expert: Int)] = []
        var dynamicHits = 0
        var dynamicMisses = 0

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
            let gb = guExpertBytes, db = downExpertBytes
            var tmp: UnsafeMutableRawPointer?
            posix_memalign(&tmp, 16_384, gb + db)
            guard let tmp else { return 0 }
            defer { free(tmp) }
            for (bank, p) in promotedForVerify {
                let (guBatch, dnBatch) = try makeBatches(
                    [(expert: p, slot: 0)],
                    guBase: tmp,
                    downBase: tmp.advanced(by: gb)
                )
                try executeBatch(guBatch: guBatch, downBatch: dnBatch)
                if memcmp(tmp, guBank.advanced(by: bank * gb), gb) != 0
                    || memcmp(tmp.advanced(by: gb), downBank.advanced(by: bank * db), db) != 0 {
                    bad += 1
                }
            }
            return bad
        }
        var downExpertBytes: Int { hidden * downRowBytes }

        func effectiveColdBudget(
            isVerifyWindow: Bool,
            specBudget: Int? = QWEngineDriver.ColdExpertStreamer.draftColdExpertBudget
        ) -> (apply: Bool, slots: Int) {
            if isVerifyWindow, let budget = specBudget {
                return (true, max(1, min(stagingSlots, budget)))
            }
            return (false, stagingSlots)
        }

        func prepareSpanDynamic(
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
            guard startRow < totalRows else {
                return (totalRows, 0, .named([]), .named([]), [], .empty)
            }
            let now = clock &+ 1
            let resU32 = UInt32(residentExperts)
            let (applySpecBudget, effectiveColdSlots) = effectiveColdBudget(
                isVerifyWindow: isVerifyWindow, specBudget: specBudget)
            var stagedFor: [Int: UInt32] = [:]
            var stagedItems: [(expert: Int, slot: Int)] = []
            var hitSlots: [Int] = []
            var hitsDelta = 0
            var missesDelta = 0
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
                if (r == startRow || !applySpecBudget) && need > stagingSlots {
                    throw QWEngineDriverError.state(
                        "cold expert staging capacity \(stagingSlots) < row \(r) demand \(need)")
                }
                if r > startRow && stagedItems.count + need > effectiveColdSlots && !applySpecBudget { break }
                for k in 0..<routes {
                    let i = rowBase + k
                    let raw = idxPtr[i]
                    if Int32(bitPattern: raw) < 0 { continue }
                    let p = Int(raw)
                    guard p < totalExperts else {
                        idxPtr[i] = 0xFFFF_FFFF
                        continue
                    }
                    let s = Int(slotOfExpert[p])
                    if s >= 0 {
                        idxPtr[i] = UInt32(s)
                        hitSlots.append(s)
                        hitsDelta += 1
                        continue
                    }
                    if let x = stagedFor[p] { idxPtr[i] = x; continue }
                    missesDelta += 1
                    let slotCap = (r == startRow) ? stagingSlots : effectiveColdSlots
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
            let (gu, dn, tx) = try promoteAndBatch(
                stagedItems, now: now, hitSlots: hitSlots,
                hitsDelta: hitsDelta, missesDelta: missesDelta
            )
            let promos = tx.promotions.map { (staging: $0.staging, bank: $0.bank) }
            return (r, stagedItems.count, gu, dn, promos, tx)
        }

        /// Whether a span of `routes` ids can always be mapped by `qwg_moe_span_ids`: every
        /// non-resident expert fits a staging slot, so the span never splits.
        func canMapWholeSpan(
            isVerifyWindow: Bool = false,
            specBudget: Int? = QWEngineDriver.ColdExpertStreamer.draftColdExpertBudget
        ) -> Bool {
            isDynamic && totalExperts <= 512 && stagingSlots >= totalExperts - residentExperts
                && !(isVerifyWindow && specBudget != nil)
        }

        /// Dynamic-residency bookkeeping for a span mapped on the GPU by `qwg_moe_span_ids` (`mapped[i]` = bank
        /// slot, `residentExperts` + staging slot, or -1) from the physical router `ids`: computes hit/miss stamps
        /// and staging batches transactionally for `commitPromotion` after NVMe I/O succeeds.
        func prepareSpanMapped(
            ids: UnsafePointer<UInt32>, mapped: UnsafePointer<Int32>, count: Int
        ) throws -> (
            stagedCount: Int,
            guBatch: ColdLoadBatch,
            downBatch: ColdLoadBatch,
            transaction: PromotionTransaction
        ) {
            precondition(canMapWholeSpan(), "span map needs a dynamic streamer whose staging holds every miss")
            let now = clock &+ 1
            let res = Int32(residentExperts)
            var expertOfSlot = [Int](repeating: -1, count: stagingSlots)
            var firstSeen = [Int](repeating: 0, count: stagingSlots)
            var hitSlots: [Int] = []
            var hitsDelta = 0
            var missesDelta = 0
            var staged = 0
            for i in 0..<count {
                let m = mapped[i]
                if m < 0 { continue }
                if m < res {
                    guard Int(ids[i]) < totalExperts, slotOfExpert[Int(ids[i])] == m else {
                        throw QWEngineDriverError.state("span map slot \(m) disagrees with host residency")
                    }
                    hitSlots.append(Int(m))
                    hitsDelta += 1
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
                    missesDelta += 1
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
            let (gu, dn, tx) = try promoteAndBatch(
                stagedItems, now: now, hitSlots: hitSlots,
                hitsDelta: hitsDelta, missesDelta: missesDelta
            )
            return (staged, gu, dn, tx)
        }

    }
}
