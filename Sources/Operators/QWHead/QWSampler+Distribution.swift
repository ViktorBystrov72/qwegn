// QWSampler+Distribution.swift — Categorical distribution filtering, top-K/top-P, and logprob.
import Foundation
import Accelerate
import QWCore

extension QWSampler {
    /// The greedy token of a logit row: the largest logit, ties broken by the
    /// lowest id. A repetition penalty shifts the row first.
    public static func greedyToken(logits: [Float], repetitionHistory: [Int] = [],
                                   repetitionPenalty: Float = 1.0) throws -> Int {
        guard !logits.isEmpty else {
            throw QWSamplerError.invalid("empty logit row")
        }
        let row = repetitionPenalty == 1 ? logits
            : try adjustedLogits(logits, repetitionHistory: repetitionHistory,
                                 repetitionPenalty: repetitionPenalty)
        if repetitionPenalty == 1 {
            guard row.withUnsafeBufferPointer({ Self.allFiniteSIMD($0) }) else {
                throw QWSamplerError.invalid("logit row contains a non-finite value")
            }
        }
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
        @inline(__always) func heapLess(_ a: Int, _ b: Int) -> Bool {
            let la = logits[a], lb = logits[b]
            return la != lb ? la < lb : a > b
        }
        for i in 0..<k {
            heapScratch.append(i)
            var c = heapScratch.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if heapLess(heapScratch[c], heapScratch[p]) {
                    heapScratch.swapAt(c, p)
                    c = p
                } else {
                    break
                }
            }
        }
        var maxEvicted = -Float.infinity
        var rootVal = logits[heapScratch[0]]
        for i in k..<n {
            let v = logits[i]
            if v > rootVal || (v == rootVal && i < heapScratch[0]) {
                if rootVal > maxEvicted { maxEvicted = rootVal }
                heapScratch[0] = i
                var p = 0
                while true {
                    let l = 2 * p + 1, r = l + 1
                    var m = p
                    if l < k && heapLess(heapScratch[l], heapScratch[m]) { m = l }
                    if r < k && heapLess(heapScratch[r], heapScratch[m]) { m = r }
                    if m == p { break }
                    heapScratch.swapAt(p, m)
                    p = m
                }
                rootVal = logits[heapScratch[0]]
            } else if v > maxEvicted {
                maxEvicted = v
            }
        }
        let kth = logits[heapScratch[0]]
        topKScratch.removeAll(keepingCapacity: true)
        topKScratch.reserveCapacity(k)
        if maxEvicted < kth {
            topKScratch.append(contentsOf: heapScratch)
        } else {
            for i in 0..<n where logits[i] > kth {
                topKScratch.append(i)
            }
            if topKScratch.count < k {
                for i in 0..<n where logits[i] == kth {
                    topKScratch.append(i)
                    if topKScratch.count == k { break }
                }
            }
        }
        topKScratch.sort { a, b in
            let la = logits[a], lb = logits[b]
            return la != lb ? la > lb : a < b
        }
        let candidates = topKScratch.map { idx in
            QWTokenLogprob.Candidate(tokenId: idx, logprob: logits[idx] - logZ)
        }
        return QWTokenLogprob(tokenId: chosenToken, logprob: chosenLP, topCandidates: candidates)
    }

    /// Identifies tokens forming consecutive periodic cycles (period <= 96), >=6x trailing bursts,
    /// or 3rd-occurrence 8-gram non-consecutive loops.
    public static func loopRepetitionTokens(in window: [Int]) -> [Int] {
        var out: [Int] = []
        loopRepetitionTokens(in: window[...], into: &out)
        return out
    }

    /// Builds a categorical distribution restricted strictly to `allowedScratch`.
    public func constrainedDistributionFromScratch(
        logits: UnsafeBufferPointer<Float>,
        repetitionHistory: [Int] = []
    ) throws -> QWSamplingDistribution {
        try allowedScratch.withUnsafeBufferPointer { buf in
            try constrainedDistribution(logits: logits, allowed: buf, repetitionHistory: repetitionHistory)
        }
    }

    /// Builds a categorical distribution over ascending `allowed` token IDs.
    public func constrainedDistribution(
        logits: UnsafeBufferPointer<Float>,
        allowed: UnsafeBufferPointer<Int>,
        repetitionHistory: [Int] = []
    ) throws -> QWSamplingDistribution {
        let n = logits.count
        guard n > 0, let logitsBase = logits.baseAddress else { throw QWSamplerError.invalid("empty logit row") }
        let count = allowed.count
        guard count > 0, let allowedBase = allowed.baseAddress else { throw QWTokenConstraintError.emptySupport }
        if config.isGreedy {
            let best = try constrainedToken(logits: logits, allowed: allowed, repetitionHistory: repetitionHistory)
            return QWSamplingDistribution(vocabulary: n, ids: [best], weights: [1.0], mass: 1.0, fullMass: 1.0)
        }
        let temperature = config.temperature
        guard temperature.isFinite, temperature > 0, (1 / temperature).isFinite, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1 else {
            throw QWSamplerError.invalid("invalid temperature, top-k or top-p")
        }
        let penalty = config.repetitionPenalty
        guard penalty.isFinite, penalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        if count == 1 {
            let only = allowedBase[0]
            guard only >= 0, only < n, logitsBase[only].isFinite else {
                throw QWSamplerError.invalid("constrained singleton token or logit is invalid")
            }
            return QWSamplingDistribution(vocabulary: n, ids: [only], weights: [1.0], mass: 1.0, fullMass: 1.0)
        }
        let applyPenalty = penalty != 1 && !repetitionHistory.isEmpty
        if applyPenalty {
            maskScratch.clearBitset()
            maskScratch.ensureVocabularyCapacity(n)
            for token in repetitionHistory where token >= 0 && token < n {
                maskScratch.markTokenUnchecked(token)
            }
        }
        defer { if applyPenalty { maskScratch.clearBitset() } }
        ensureConstrainedScratchCapacity(count: count)
        var previous = -1
        for index in 0..<count {
            let token = allowedBase[index]
            guard token > previous, token < n else {
                throw QWSamplerError.invalid("constrained support must be ascending token ids inside the vocabulary")
            }
            previous = token
            var value = logitsBase[token]
            if applyPenalty, maskScratch.contains(token) {
                value = value < 0 ? value * penalty : value / penalty
            }
            guard value.isFinite else {
                throw QWSamplerError.invalid("constrained logit is non-finite")
            }
            penaltyScratch[index] = value
        }
        var maxValue: Float = 0, sumF: Float = 0, negMax: Float = 0
        var invT = 1 / temperature, bias: Float = 0
        var nu = Int32(count)
        let len = vDSP_Length(count)
        penaltyScratch.withUnsafeMutableBufferPointer { vp in
            let vPtr = vp.baseAddress!
            vDSP_maxv(vPtr, 1, &maxValue, len)
            negMax = -maxValue
            shiftedScratch.withUnsafeMutableBufferPointer { sp in
                let sPtr = sp.baseAddress!
                vDSP_vsadd(vPtr, 1, &negMax, sPtr, 1, len)
                vDSP_vsmsa(sPtr, 1, &invT, &bias, sPtr, 1, len)
                eScratch.withUnsafeMutableBufferPointer { ep in
                    let ePtr = ep.baseAddress!
                    vvexpf(ePtr, sPtr, &nu)
                    vDSP_sve(ePtr, 1, &sumF, len)
                }
            }
        }
        let fullMass = Double(sumF)
        let eFiniteNonNeg = eScratch.withUnsafeBufferPointer {
            Self.allFiniteNonNegSIMD(UnsafeBufferPointer(start: $0.baseAddress, count: count), sum: sumF)
        }
        guard fullMass.isFinite, fullMass > 0, eFiniteNonNeg else {
            throw QWSamplerError.invalid("invalid constrained softmax mass")
        }
        for i in 0..<count { constrainedWeightsScratch[i] = Double(eScratch[i]) }
        let k = min(config.topK, count)
        fillConstrainedTopIndices(count: count, topK: k)
        let topP = Double(config.topP)
        var keepCount = 0
        var cumulative = 0.0
        for pos in 0..<k {
            if cumulative >= topP { break }
            let index = topKScratch[pos]
            constrainedKeepScratch[keepCount] = index
            keepCount += 1
            cumulative += constrainedWeightsScratch[index] / fullMass
        }
        constrainedKeepScratch[0..<keepCount].sort()
        var ids: [Int] = []
        var weights: [Double] = []
        ids.reserveCapacity(keepCount)
        weights.reserveCapacity(keepCount)
        var mass = 0.0
        for i in 0..<keepCount {
            let slot = constrainedKeepScratch[i]
            let w = constrainedWeightsScratch[slot]
            ids.append(allowedBase[slot])
            weights.append(w)
            mass += w
        }
        guard mass.isFinite, mass > 0, !ids.isEmpty else {
            throw QWSamplerError.invalid("invalid constrained survivor mass")
        }
        return QWSamplingDistribution(vocabulary: n, ids: ids, weights: weights, mass: mass, fullMass: fullMass)
    }

    /// Evaluates variational log-margin and probability floor strictly over `allowedScratch`.
    public func clearsConstrainedVariationalGateFromScratch(
        logits: UnsafeBufferPointer<Float>,
        draftToken: Int,
        targetToken: Int,
        repetitionHistory: [Int] = [],
        margin: Float,
        minProbability: Float
    ) -> Bool {
        guard QWTokenMaskScratch.sortedTokens(allowedScratch, contains: draftToken) else { return false }
        return allowedScratch.withUnsafeBufferPointer { buf in
            clearsConstrainedVariationalGate(
                logits: logits, allowed: buf, draftToken: draftToken, targetToken: targetToken,
                repetitionHistory: repetitionHistory, margin: margin, minProbability: minProbability)
        }
    }

    /// Evaluates variational log-margin and probability floor renormalized over `allowed`.
    public func clearsConstrainedVariationalGate(
        logits: UnsafeBufferPointer<Float>,
        allowed: UnsafeBufferPointer<Int>,
        draftToken: Int,
        targetToken: Int,
        repetitionHistory: [Int] = [],
        margin: Float,
        minProbability: Float
    ) -> Bool {
        let n = logits.count
        guard margin > 0, n > 0, let logitsBase = logits.baseAddress,
              allowed.count > 0, let allowedBase = allowed.baseAddress,
              (0..<n).contains(draftToken), (0..<n).contains(targetToken) else {
            return false
        }
        var lo = 0, hi = allowed.count - 1, inAllowed = false
        while lo <= hi {
            let mid = lo + ((hi - lo) >> 1)
            let v = allowedBase[mid]
            if v == draftToken { inAllowed = true; break }
            if v < draftToken { lo = mid + 1 } else { hi = mid - 1 }
        }
        guard inAllowed else { return false }
        let penalty = config.repetitionPenalty
        let applyPenalty = penalty.isFinite && penalty > 0 && penalty != 1 && !repetitionHistory.isEmpty
        if applyPenalty {
            maskScratch.clearBitset()
            maskScratch.ensureVocabularyCapacity(n)
            for token in repetitionHistory where token >= 0 && token < n {
                maskScratch.markTokenUnchecked(token)
            }
        }
        defer { if applyPenalty { maskScratch.clearBitset() } }
        @inline(__always) func score(_ tok: Int) -> Float {
            let raw = logitsBase[tok]
            if applyPenalty, maskScratch.contains(tok) {
                return raw < 0 ? raw * penalty : raw / penalty
            }
            return raw
        }
        let draftScore = score(draftToken)
        let targetScore = score(targetToken)
        guard draftScore.isFinite, targetScore.isFinite, draftScore >= targetScore - margin else {
            return false
        }
        guard minProbability > 0 else { return true }
        guard minProbability < 1 else { return false }
        var sum = 0.0
        for idx in 0..<allowed.count {
            let tok = allowedBase[idx]
            guard tok >= 0 && tok < n else { continue }
            let diff = score(tok) - targetScore
            if diff > -45.0 { sum += exp(Double(diff)) }
        }
        guard sum > 0, sum.isFinite else { return false }
        let logProb = Double(draftScore - targetScore) - log(sum)
        return logProb >= log(Double(minProbability))
    }
}

// (the median-of-medians helpers that lived here are GONE — the bounded
//  min-heap above replaced them; see the select comment for the numbers.)
