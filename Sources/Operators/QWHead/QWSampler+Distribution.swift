// QWSampler+Distribution.swift — Categorical distribution filtering, top-K/top-P, and logprob.
import Foundation
import Accelerate
import QWCore

extension QWSampler {
    static func topIndices(_ weights: [Double], count: Int) -> [Int] {
        let before: (Int, Int) -> Bool = { a, b in
            weights[a] != weights[b] ? weights[a] > weights[b] : a < b
        }
        if count >= weights.count {
            return weights.indices.sorted(by: before)
        }
        // Min-heap (by `before`) of the current best `count` indices.
        var heap: [Int] = []
        heap.reserveCapacity(count)
        func siftDown(_ start: Int) {
            var p = start
            while true {
                let l = 2 * p + 1, r = l + 1
                var m = p
                if l < heap.count && before(heap[m], heap[l]) { m = l }
                if r < heap.count && before(heap[m], heap[r]) { m = r }
                if m == p { return }
                heap.swapAt(p, m); p = m
            }
        }
        for index in weights.indices {
            if heap.count < count {
                heap.append(index)
                var c = heap.count - 1
                while c > 0 {
                    let p = (c - 1) >> 1
                    if before(heap[p], heap[c]) { heap.swapAt(c, p); c = p } else { break }
                }
            } else if before(index, heap[0]) {
                heap[0] = index
                siftDown(0)
            }
        }
        return heap.sorted(by: before)
    }

    /// The greedy token of a logit row: the largest logit, ties broken by the
    /// lowest id. A repetition penalty shifts the row first.
    public static func greedyToken(logits: [Float], repetitionHistory: [Int] = [],
                                   repetitionPenalty: Float = 1.0) throws -> Int {
        let row = repetitionPenalty == 1 ? logits
            : try adjustedLogits(logits, repetitionHistory: repetitionHistory,
                                 repetitionPenalty: repetitionPenalty)
        var best = 0
        for index in 1..<row.count where row[index] > row[best] { best = index }
        return best
    }

    public func distribution(logits: [Float],
                             repetitionHistory: [Int] = []) throws -> QWSamplingDistribution {
        try logits.withUnsafeBufferPointer {
            try distribution(logits: $0, repetitionHistory: repetitionHistory)
        }
    }

    /// Reads the row synchronously; the returned distribution owns its support.
    public func distribution(logits: UnsafeBufferPointer<Float>,
                             repetitionHistory: [Int] = []) throws -> QWSamplingDistribution {
        let T = config.temperature, P = Double(config.topP)
        let n = logits.count
        guard T.isFinite, T > 0, (1 / T).isFinite, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1,
              config.repetitionPenalty.isFinite, config.repetitionPenalty > 0,
              n > 0, n <= Int(Int32.max), Self.allFiniteSIMD(logits) else {
            throw QWSamplerError.invalid("invalid temperature, top-k/top-p or logit row")
        }
        let K = min(config.topK, n)
        if config.repetitionPenalty != 1, !repetitionHistory.isEmpty {
            guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
                throw QWSamplerError.invalid("invalid repetition penalty")
            }
            if penaltyScratch.count != n { penaltyScratch = [Float](repeating: 0, count: n) }
            if seenEpoch.count != n {
                seenEpoch = [UInt32](repeating: 0, count: n)
                currentEpoch = 0
            }
            currentEpoch &+= 1
            if currentEpoch == 0 {
                for idx in 0..<n { seenEpoch[idx] = 0 }
                currentEpoch = 1
            }
            let epoch = currentEpoch
            let rep = config.repetitionPenalty
            // The penalty is applied in place: the row is copied once, into
            // the scratch the sampler already owns.
            try penaltyScratch.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: logits.baseAddress!, count: n)
                try seenEpoch.withUnsafeMutableBufferPointer { seen in
                    for token in repetitionHistory {
                        guard token >= 0 && token < n else {
                            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
                        }
                        if seen[token] != epoch {
                            seen[token] = epoch
                            let value = dst[token]
                            let nextVal = value < 0 ? value * rep : value / rep
                            guard nextVal.isFinite else {
                                throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
                            }
                            dst[token] = nextVal
                        }
                    }
                }
            }
            return try penaltyScratch.withUnsafeBufferPointer {
                try makeDistribution(logits: $0, vocabulary: n, topK: K, topP: P,
                                     temperature: T)
            }
        }
        return try makeDistribution(logits: logits, vocabulary: n, topK: K, topP: P,
                                    temperature: T)
    }

    func makeDistribution(logits: UnsafeBufferPointer<Float>, vocabulary n: Int,
                                  topK K: Int, topP P: Double,
                                  temperature: Float) throws -> QWSamplingDistribution {
        if eScratch.count != n { eScratch = [Float](repeating: 0, count: n) }
        if shiftedScratch.count != n { shiftedScratch = [Float](repeating: 0, count: n) }
        var dmax: Float = 0
        var sum: Float = 0
        return try eScratch.withUnsafeMutableBufferPointer { ep in
            try shiftedScratch.withUnsafeMutableBufferPointer { sp in
                vDSP_maxv(logits.baseAddress!, 1, &dmax, vDSP_Length(n))
                // shifted = logits − dmax  (vsub: dst = src − scalar)
                var negDmax = -dmax
                vDSP_vsadd(logits.baseAddress!, 1, &negDmax, sp.baseAddress!, 1, vDSP_Length(n))
                // shifted = shifted · (1/T) + 0  (vsmsa: src·scalar + bias)
                var invT = 1 / temperature, bias: Float = 0
                vDSP_vsmsa(sp.baseAddress!, 1, &invT, &bias,
                           sp.baseAddress!, 1, vDSP_Length(n))
                var nu = Int32(n)
                // vForce SIMD exp (NOT vexpf — the Swift overlay shadows
                // it with a scalar variant and the C array form is not
                // imported; measured). Y = e^X.
                vvexpf(ep.baseAddress!, sp.baseAddress!, &nu)   // fp32 SIMD exp
                vDSP_sve(ep.baseAddress!, 1, &sum, vDSP_Length(n))
                return try transformedSupport(e: UnsafeBufferPointer(ep), topK: K, topP: P, sum: sum)
            }
        }
    }

    func transformedSupport(e: UnsafeBufferPointer<Float>, topK K: Int,
                                    topP P: Double, sum: Float) throws -> QWSamplingDistribution {
        guard sum.isFinite, sum > 0, Self.allFiniteNonNegSIMD(e, sum: sum) else {
            throw QWSamplerError.invalid("invalid softmax mass")
        }
        // ── top-k 64: BOUNDED MIN-HEAP of the k LARGEST (O(N log k),
        //    reused scratch, e never mutated). v1/v2 median-of-medians cost
        //    2977 µs of swaps+pivot allocations (sampler_profile: exp pass
        //    33 µs, select 2977 — the select WAS the sampler). v3 first
        //    cut had the comparison BACKWARDS (e[i] < root keeps the k
        //    SMALLEST → keepK held ~N ids → the desc sort cost 5 ms and
        //    the peaked fixture's tests still passed because the nucleus
        //    cumsum breaks long before the kth boundary matters when the
        //    top-k mass ≥ p). The contract only binds on a FLAT fixture
        //    (top-64 mass < p): torch restricts candidates to the k — a
        //    superset select would draw from the whole tail. Law: test
        //    the boundary the code implements, not just the headline path.
        heapScratch.removeAll(keepingCapacity: true)
        heapScratch.reserveCapacity(K)
        var maxEvicted: Float = -1.0
        let count = e.count
        let initK = min(K, count)
        for i in 0..<initK {
            heapScratch.append(i)
            var c = heapScratch.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if e[heapScratch[c]] < e[heapScratch[p]] { heapScratch.swapAt(c, p); c = p } else { break }
            }
        }
        if initK < count, let base = e.baseAddress {
            var i = initK
            var rootVal = e[heapScratch[0]]
            while i + 16 <= count {
                let v = UnsafeRawPointer(base.advanced(by: i)).loadUnaligned(as: SIMD16<Float>.self)
                if v.max() > rootVal {
                    for j in i..<(i + 16) where e[j] > rootVal {
                        if rootVal > maxEvicted { maxEvicted = rootVal }
                        heapScratch[0] = j
                        var p = 0
                        while true {
                            let l = 2 * p + 1, r = l + 1
                            var m = p
                            if l < K && e[heapScratch[l]] < e[heapScratch[m]] { m = l }
                            if r < K && e[heapScratch[r]] < e[heapScratch[m]] { m = r }
                            if m == p { break }
                            heapScratch.swapAt(p, m); p = m
                        }
                        rootVal = e[heapScratch[0]]
                    }
                }
                i += 16
            }
            while i < count {
                if e[i] > rootVal {
                    if rootVal > maxEvicted { maxEvicted = rootVal }
                    heapScratch[0] = i
                    var p = 0
                    while true {
                        let l = 2 * p + 1, r = l + 1
                        var m = p
                        if l < K && e[heapScratch[l]] < e[heapScratch[m]] { m = l }
                        if r < K && e[heapScratch[r]] < e[heapScratch[m]] { m = r }
                        if m == p { break }
                        heapScratch.swapAt(p, m); p = m
                    }
                    rootVal = e[heapScratch[0]]
                }
                i += 1
            }
        }
        // heap now holds the K LARGEST ids (ties included by FIRST-seen
        // ascending-id order: strict < comparisons never evict an equal —
        // exactly torch's tie direction, and it is the stable law the
        let kth = e[heapScratch[0]]
        topKScratch.removeAll(keepingCapacity: true)
        topKScratch.reserveCapacity(K)
        if maxEvicted < kth {
            topKScratch.append(contentsOf: heapScratch)
        } else {
            for i in e.indices {
                if e[i] > kth { topKScratch.append(i) }
            }
            if topKScratch.count < K {
                for i in e.indices where e[i] == kth {
                    topKScratch.append(i)
                    if topKScratch.count == K { break }
                }
            }
        }
        // order keepK DESC by e (ties ascending id) for the nucleus walk
        topKScratch.sort { a, b in let ea = e[a], eb = e[b]; return ea != eb ? ea > eb : a < b }
        // ── top-p nucleus over keepK (≤ K + ties entries); the cumsum
        //    walks NORMALIZED probs — torch thresholds against Σp = 1 ──
        let sumD = Double(sum)
        var cum = 0.0
        var need = 1
        for (r, i) in topKScratch.enumerated() {
            let before = cum
            cum += Double(e[i]) / sumD
            if before < P { need = r + 1 } else { break }
        }
        let qth = e[topKScratch[need - 1]]
        var cntGtQ = 0
        for i in topKScratch where e[i] > qth { cntGtQ += 1 }
        var keep: [Int] = []
        var sl2 = need - cntGtQ
        for i in topKScratch {                        // desc walk, ties asc id
            if e[i] > qth { keep.append(i) }
            else if e[i] == qth && sl2 > 0 { keep.append(i); sl2 -= 1 }
        }
        keep.sort()                                   // ascending id (renorm)
        var massD = 0.0
        for i in keep { massD += Double(e[i]) }
        guard massD.isFinite, massD > 0, !keep.isEmpty else {
            throw QWSamplerError.invalid("invalid survivor mass")
        }
        return QWSamplingDistribution(vocabulary: e.count, ids: keep, weights: keep.map { Double(e[$0]) },
                                      mass: massD, fullMass: sumD)
    }

    /// Computes the full-vocabulary log-probability (`logit - logsumexp`) of `chosenToken`
    /// and optionally the top `topK` (`0..20`) highest-probability candidate tokens.
    public func computeTokenLogprob(logits: UnsafeBufferPointer<Float>,
                                    chosenToken: Int,
                                    topK: Int = 0) -> QWTokenLogprob {
        let n = logits.count
        guard n > 0, chosenToken >= 0, chosenToken < n, let base = logits.baseAddress else {
            return QWTokenLogprob(tokenId: chosenToken, logprob: 0)
        }
        if eScratch.count != n { eScratch = [Float](repeating: 0, count: n) }
        if shiftedScratch.count != n { shiftedScratch = [Float](repeating: 0, count: n) }
        var dmax: Float = 0
        var sum: Float = 0
        eScratch.withUnsafeMutableBufferPointer { ep in
            shiftedScratch.withUnsafeMutableBufferPointer { sp in
                vDSP_maxv(base, 1, &dmax, vDSP_Length(n))
                var negDmax = -dmax
                vDSP_vsadd(base, 1, &negDmax, sp.baseAddress!, 1, vDSP_Length(n))
                var nu = Int32(n)
                vvexpf(ep.baseAddress!, sp.baseAddress!, &nu)
                vDSP_sve(ep.baseAddress!, 1, &sum, vDSP_Length(n))
            }
        }
        let logZ = dmax + Foundation.log(max(sum, Float.leastNormalMagnitude))
        let chosenLP = logits[chosenToken] - logZ
        let k = min(max(0, topK), min(20, n))
        guard k > 0 else {
            return QWTokenLogprob(tokenId: chosenToken, logprob: chosenLP)
        }
        heapScratch.removeAll(keepingCapacity: true)
        heapScratch.reserveCapacity(k)
        for i in 0..<k {
            heapScratch.append(i)
            var c = heapScratch.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if logits[heapScratch[c]] < logits[heapScratch[p]] {
                    heapScratch.swapAt(c, p)
                    c = p
                } else {
                    break
                }
            }
        }
        var rootVal = logits[heapScratch[0]]
        for i in k..<n where logits[i] > rootVal {
            heapScratch[0] = i
            var p = 0
            while true {
                let l = 2 * p + 1, r = l + 1
                var m = p
                if l < k && logits[heapScratch[l]] < logits[heapScratch[m]] { m = l }
                if r < k && logits[heapScratch[r]] < logits[heapScratch[m]] { m = r }
                if m == p { break }
                heapScratch.swapAt(p, m)
                p = m
            }
            rootVal = logits[heapScratch[0]]
        }
        let sortedTop = heapScratch.sorted { a, b in
            let la = logits[a], lb = logits[b]
            return la != lb ? la > lb : a < b
        }
        let candidates = sortedTop.map { idx in
            QWTokenLogprob.Candidate(tokenId: idx, logprob: logits[idx] - logZ)
        }
        return QWTokenLogprob(tokenId: chosenToken, logprob: chosenLP, topCandidates: candidates)
    }
}

// (the median-of-medians helpers that lived here are GONE — the bounded
//  min-heap above replaced them; see the select comment for the numbers.)
