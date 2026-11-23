// QWSampler.swift — C15 sampler: the distribution path (T>0), CPU side.
//
// WHY CPU (the spec's race; DECISIONS): Apple Silicon UMA maps
// storageModeShared CPU-side (QWMRegion.pointer), so the logits hand-off
// is a memory read, not a PCIe trip — but ONLY with the exp pass on
// vecLib vexpf: a scalar fp64 exp over 61440 logits measured 3.0 ms
// (bench_head v1, attribution probe) — 12× the ≤250 µs gate. vexpf is the
import Foundation
import Accelerate
import QWCore

public final class QWSampler {
    public var config: QWSamplerConfig
    var callIndex: UInt64 = 0
    // Reused scratch (per-token call; allocator churn is C09 steady-state
    // law territory): e = exp((v − dmax)/T) fp32.
    var eScratch: [Float] = []
    var shiftedScratch: [Float] = []
    var penaltyScratch: [Float] = []
    var heapScratch: [Int] = []
    var topKScratch: [Int] = []
    var seenEpoch: [UInt32] = []
    var currentEpoch: UInt32 = 0
    public var allowedScratch: [Int] = []
    public let maskScratch = QWTokenMaskScratch()
    var constrainedWeightsScratch: [Double] = []
    var constrainedKeepScratch: [Int] = []

    public init(config: QWSamplerConfig) { self.config = config }
    public var drawsConsumed: UInt64 { callIndex }

    @inline(__always)
    static func allFiniteSIMD(_ buf: UnsafeBufferPointer<Float>) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var maxMag: Float = 0
        var rawSum: Float = 0
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        vDSP_sve(base, 1, &rawSum, vDSP_Length(n))
        if maxMag < 1e30 && rawSum.isFinite { return true }
        return buf.allSatisfy(\.isFinite)
    }

    @inline(__always)
    static func allFiniteNonNegSIMD(_ buf: UnsafeBufferPointer<Float>, sum: Float) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var minVal: Float = 0
        var maxMag: Float = 0
        vDSP_minv(base, 1, &minVal, vDSP_Length(n))
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        if minVal >= 0 && maxMag < 1e30 && sum.isFinite { return true }
        return buf.allSatisfy { $0.isFinite && $0 >= 0 }
    }

    public static func adjustedLogits(_ logits: [Float], repetitionHistory: [Int],
                                      repetitionPenalty: Float) throws -> [Float] {
        guard repetitionPenalty.isFinite, repetitionPenalty > 0,
              logits.withUnsafeBufferPointer({ Self.allFiniteSIMD($0) }) else {
            throw QWSamplerError.invalid("invalid repetition penalty or logit row")
        }
        let n = logits.count
        for token in repetitionHistory {
            guard token >= 0, token < n else {
                throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
            }
        }
        if repetitionPenalty == 1 || repetitionHistory.isEmpty { return logits }
        var adjusted = logits
        if repetitionHistory.count <= 16 {
            for i in repetitionHistory.indices {
                let token = repetitionHistory[i]
                var duplicate = false
                for j in repetitionHistory.startIndex..<i where repetitionHistory[j] == token {
                    duplicate = true
                    break
                }
                if duplicate { continue }
                let value = adjusted[token]
                let nextVal = value < 0 ? value * repetitionPenalty : value / repetitionPenalty
                guard nextVal.isFinite else {
                    throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
                }
                adjusted[token] = nextVal
            }
        } else {
            for token in Set(repetitionHistory) {
                let value = adjusted[token]
                let nextVal = value < 0 ? value * repetitionPenalty : value / repetitionPenalty
                guard nextVal.isFinite else {
                    throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
                }
                adjusted[token] = nextVal
            }
        }
        return adjusted
    }

    /// Sample one token from one LM-head logit row. O(N) SIMD exp pass +
    /// O(N log K) bounded selection; only the K survivors are sorted.
    public func sample(logits: [Float], repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// The shared entry point: one logit row in, one token id out. The draw
    /// counter lives in the sampler, so a family that owns its own sampler
    /// gets the same sequence as any other family with the same seed.
    public func sampleToken(logits: [Float], repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The same entry point over a borrowed row: the caller keeps the buffer
    /// and the sampler copies nothing.
    public func sampleToken(logits: UnsafeBufferPointer<Float>,
                             repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The greedy token of a borrowed logit row, with the repetition penalty applied without copying the row.
    public func penalisedGreedyToken(logits: UnsafeBufferPointer<Float>,
                                     repetitionHistory: Set<Int>) throws -> Int {
        try penalisedGreedyCore(logits: logits, repetitionHistory: repetitionHistory, isEmpty: repetitionHistory.isEmpty)
    }

    /// Zero-allocation variant accepting a dense repetition-token slice or array.
    public func penalisedGreedyToken(logits: UnsafeBufferPointer<Float>,
                                     repetitionHistory: [Int]) throws -> Int {
        try penalisedGreedyCore(logits: logits, repetitionHistory: repetitionHistory, isEmpty: repetitionHistory.isEmpty)
    }

    /// The same over an array.
    public func penalisedGreedyToken(logits: [Float],
                                     repetitionHistory: Set<Int>) throws -> Int {
        try logits.withUnsafeBufferPointer {
            try penalisedGreedyCore(logits: $0, repetitionHistory: repetitionHistory, isEmpty: repetitionHistory.isEmpty)
        }
    }

    /// Zero-allocation array overload matching `penalisedGreedyToken(logits:[Float],repetitionHistory:[Int])`.
    public func penalisedGreedyToken(logits: [Float],
                                     repetitionHistory: [Int]) throws -> Int {
        try logits.withUnsafeBufferPointer {
            try penalisedGreedyCore(logits: $0, repetitionHistory: repetitionHistory, isEmpty: repetitionHistory.isEmpty)
        }
    }

    private func penalisedGreedyCore<S: Sequence>(logits: UnsafeBufferPointer<Float>,
                                                  repetitionHistory: S,
                                                  isEmpty: Bool) throws -> Int where S.Element == Int {
        let n = logits.count
        guard n > 0, let base = logits.baseAddress else { throw QWSamplerError.invalid("empty logit row") }
        let penalty = config.repetitionPenalty
        guard penalty.isFinite, penalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        guard Self.allFiniteSIMD(logits) else {
            throw QWSamplerError.invalid("logit row contains a non-finite value")
        }
        if isEmpty || penalty == 1 {
            for token in repetitionHistory where token < 0 || token >= n {
                throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
            }
            return Self.scanGreedyNoPenalty(base: base, count: n)
        }
        do {
            try maskScratch.markTokens(repetitionHistory, vocabularySize: n)
        } catch {
            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
        }
        defer { maskScratch.clearBitset() }
        var best = -1
        var bestVal = -Float.infinity
        let wordCount = (n + 63) >> 6
        for wordIdx in 0..<wordCount {
            let blockStart = wordIdx &<< 6
            let blockEnd = min(blockStart + 64, n)
            let wBits = maskScratch.word(at: wordIdx)
            if wBits == 0 {
                if blockEnd - blockStart == 64 {
                    let raw = UnsafeRawPointer(base.advanced(by: blockStart))
                    let v0 = raw.loadUnaligned(as: SIMD16<Float>.self)
                    let v1 = raw.loadUnaligned(fromByteOffset: 64, as: SIMD16<Float>.self)
                    let v2 = raw.loadUnaligned(fromByteOffset: 128, as: SIMD16<Float>.self)
                    let v3 = raw.loadUnaligned(fromByteOffset: 192, as: SIMD16<Float>.self)
                    if pointwiseMax(pointwiseMax(v0, v1), pointwiseMax(v2, v3)).max() <= bestVal {
                        continue
                    }
                }
                for idx in blockStart..<blockEnd {
                    let v = base[idx]
                    if v > bestVal {
                        bestVal = v
                        best = idx
                    }
                }
            } else {
                for idx in blockStart..<blockEnd {
                    let raw = base[idx]
                    let v: Float
                    if (wBits & (UInt64(1) &<< UInt64(idx - blockStart))) != 0 {
                        v = raw < 0 ? raw * penalty : raw / penalty
                        if !v.isFinite {
                            throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
                        }
                    } else {
                        v = raw
                    }
                    if v > bestVal {
                        bestVal = v
                        best = idx
                    }
                }
            }
        }
        return best
    }

    private static func scanGreedyNoPenalty(base: UnsafePointer<Float>, count: Int) -> Int {
        var best = 0
        var bestVal = base[0]
        var idx = 0
        let aligned64 = count & ~63
        while idx < aligned64 {
            let raw = UnsafeRawPointer(base.advanced(by: idx))
            let v0 = raw.loadUnaligned(as: SIMD16<Float>.self)
            let v1 = raw.loadUnaligned(fromByteOffset: 64, as: SIMD16<Float>.self)
            let v2 = raw.loadUnaligned(fromByteOffset: 128, as: SIMD16<Float>.self)
            let v3 = raw.loadUnaligned(fromByteOffset: 192, as: SIMD16<Float>.self)
            if pointwiseMax(pointwiseMax(v0, v1), pointwiseMax(v2, v3)).max() > bestVal {
                for j in idx..<(idx + 64) {
                    let v = base[j]
                    if v > bestVal {
                        bestVal = v
                        best = j
                    }
                }
            }
            idx += 64
        }
        while idx < count {
            let v = base[idx]
            if v > bestVal {
                bestVal = v
                best = idx
            }
            idx += 1
        }
        return best
    }

    /// Samples one token from a borrowed logit row. No copy: the row stays
    /// where the caller put it.
    public func sample(logits: UnsafeBufferPointer<Float>,
                        repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// Selects one token from `allowed` only using internal scratch buffers with zero steady-state allocations.
    public func constrainedToken(logits: UnsafeBufferPointer<Float>, allowed: [Int],
                                 repetitionHistory: Set<Int> = []) throws -> Int {
        try allowed.withUnsafeBufferPointer { buf in
            try constrainedTokenCore(logits: logits, allowed: buf, repetitionHistory: repetitionHistory,
                                     isHistoryEmpty: repetitionHistory.isEmpty)
        }
    }

    /// Zero-allocation overload accepting an `[Int]` repetition list instead of a `Set<Int>`.
    public func constrainedToken(logits: UnsafeBufferPointer<Float>, allowed: UnsafeBufferPointer<Int>,
                                 repetitionHistory: [Int] = []) throws -> Int {
        try constrainedTokenCore(logits: logits, allowed: allowed, repetitionHistory: repetitionHistory,
                                 isHistoryEmpty: repetitionHistory.isEmpty)
    }

    /// Evaluates constrained selection using tokens previously written into `allowedScratch`.
    public func constrainedTokenFromScratch(logits: UnsafeBufferPointer<Float>,
                                            repetitionHistory: [Int] = []) throws -> Int {
        try allowedScratch.withUnsafeBufferPointer { buf in
            try constrainedTokenCore(logits: logits, allowed: buf, repetitionHistory: repetitionHistory,
                                     isHistoryEmpty: repetitionHistory.isEmpty)
        }
    }

    private func constrainedTokenCore<S: Sequence>(
        logits: UnsafeBufferPointer<Float>,
        allowed: UnsafeBufferPointer<Int>,
        repetitionHistory: S,
        isHistoryEmpty: Bool
    ) throws -> Int where S.Element == Int {
        let n = logits.count
        guard n > 0, let logitsBase = logits.baseAddress else { throw QWSamplerError.invalid("empty logit row") }
        let count = allowed.count
        guard count > 0, let allowedBase = allowed.baseAddress else { throw QWTokenConstraintError.emptySupport }
        let penalty = config.repetitionPenalty
        guard penalty.isFinite, penalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        let applyPenalty = penalty != 1 && !isHistoryEmpty
        if applyPenalty {
            maskScratch.clearBitset()
            maskScratch.ensureVocabularyCapacity(n)
            for token in repetitionHistory where token >= 0 && token < n {
                maskScratch.markTokenUnchecked(token)
            }
        }
        defer { if applyPenalty { maskScratch.clearBitset() } }

        if config.isGreedy {
            var previous = -1
            var bestToken = 0
            var bestValue = -Float.infinity
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
                if index == 0 || value > bestValue {
                    bestValue = value
                    bestToken = token
                }
            }
            return bestToken
        }

        let temperature = config.temperature
        guard temperature.isFinite, temperature > 0, (1 / temperature).isFinite, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1 else {
            throw QWSamplerError.invalid("invalid temperature, top-k or top-p")
        }
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
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
        var maxValue: Float = 0
        var sumF: Float = 0
        var negMax: Float = 0
        var invT = 1 / temperature
        var bias: Float = 0
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
        for i in 0..<count {
            constrainedWeightsScratch[i] = Double(eScratch[i])
        }
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
        var mass = 0.0
        for i in 0..<keepCount {
            mass += constrainedWeightsScratch[constrainedKeepScratch[i]]
        }
        guard mass.isFinite, mass > 0 else {
            throw QWSamplerError.invalid("invalid constrained survivor mass")
        }
        let uniform = Double(QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion))
        guard uniform.isFinite, uniform >= 0,
              config.uniformVersion == .legacyRounded32 ? uniform <= 1 : uniform < 1 else {
            throw QWSamplerError.invalid("uniform outside versioned interval")
        }
        var selectedPos = keepCount - 1
        var drawCum = 0.0
        let isLegacy = config.uniformVersion == .legacyRounded32
        let boundary = isLegacy ? uniform * mass : uniform
        for i in 0..<keepCount {
            let weight = constrainedWeightsScratch[constrainedKeepScratch[i]]
            drawCum += isLegacy ? weight : (weight / mass)
            if isLegacy ? drawCum >= boundary : drawCum > boundary {
                selectedPos = i
                break
            }
        }
        callIndex += 1
        return allowedBase[constrainedKeepScratch[selectedPos]]
    }

    private func ensureConstrainedScratchCapacity(count: Int) {
        if penaltyScratch.count < count { penaltyScratch = [Float](repeating: 0, count: count) }
        if shiftedScratch.count < count { shiftedScratch = [Float](repeating: 0, count: count) }
        if eScratch.count < count { eScratch = [Float](repeating: 0, count: count) }
        if constrainedWeightsScratch.count < count { constrainedWeightsScratch = [Double](repeating: 0, count: count) }
        let k = min(config.topK, count)
        if topKScratch.count < k { topKScratch = [Int](repeating: 0, count: k) }
        if heapScratch.count < k { heapScratch = [Int](repeating: 0, count: k) }
        if constrainedKeepScratch.count < k { constrainedKeepScratch = [Int](repeating: 0, count: k) }
    }

    private func fillConstrainedTopIndices(count: Int, topK k: Int) {
        constrainedWeightsScratch.withUnsafeBufferPointer { weights in
            @inline(__always) func before(_ a: Int, _ b: Int) -> Bool {
                weights[a] != weights[b] ? weights[a] > weights[b] : a < b
            }
            if k >= count {
                for i in 0..<count { topKScratch[i] = i }
                topKScratch[0..<count].sort(by: before)
                return
            }
            heapScratch.withUnsafeMutableBufferPointer { heap in
                @inline(__always) func siftDown(_ start: Int) {
                    var p = start
                    while true {
                        let l = 2 * p + 1, r = l + 1
                        var m = p
                        if l < k && before(heap[m], heap[l]) { m = l }
                        if r < k && before(heap[m], heap[r]) { m = r }
                        if m == p { return }
                        heap.swapAt(p, m)
                        p = m
                    }
                }
                for index in 0..<k {
                    heap[index] = index
                    var c = index
                    while c > 0 {
                        let p = (c - 1) >> 1
                        if before(heap[p], heap[c]) { heap.swapAt(c, p); c = p } else { break }
                    }
                }
                for index in k..<count where before(index, heap[0]) {
                    heap[0] = index
                    siftDown(0)
                }
                for i in 0..<k { topKScratch[i] = heap[i] }
            }
            topKScratch[0..<k].sort(by: before)
        }
    }

    /// In-place version of `loopRepetitionTokens(in:)` avoiding array and `Set` allocations.
    public static func loopRepetitionTokens(in window: [Int], into out: inout [Int]) {
        loopRepetitionTokens(in: window[...], into: &out)
    }

    /// Slice-accepting in-place version of `loopRepetitionTokens(in:)` avoiding allocations.
    public static func loopRepetitionTokens(in window: ArraySlice<Int>, into out: inout [Int]) {
        out.removeAll(keepingCapacity: true)
        let m = window.count
        guard m >= 4 else { return }
        let base = window.startIndex
        for period in 1...min(96, m / 2) {
            let repeats = period == 1 ? 4 : (period == 2 ? 3 : 2)
            let span = period * repeats
            guard m >= span else { continue }
            let start = base + m - span
            var matches = true
            for offset in 0..<(span - period) where window[start + offset] != window[start + period + offset] {
                matches = false
                break
            }
            if matches {
                let candidate = window[base + m - period]
                if !out.contains(candidate) { out.append(candidate) }
            }
        }
        let tailStart = base + max(0, m - 16)
        let tailEnd = base + m
        for i in tailStart..<tailEnd {
            let candidate = window[i]
            if out.contains(candidate) { continue }
            var count = 0
            for j in i..<tailEnd where window[j] == candidate {
                count += 1
                if count >= 6 {
                    out.append(candidate)
                    break
                }
            }
        }
        // Detect >= 2 earlier copies of the trailing 8-gram (3rd occurrence) without firing on 2 copies.
        let ngram = 8
        if m >= 3 * ngram {
            let tailGram = base + m - ngram
            var priorMatches = 0
            for pos in base..<tailGram {
                var same = true
                for o in 0..<ngram where window[pos + o] != window[tailGram + o] {
                    same = false
                    break
                }
                if same { priorMatches += 1 }
            }
            if priorMatches >= 2 {
                for pos in base..<tailGram {
                    var same = true
                    for o in 0..<ngram where window[pos + o] != window[tailGram + o] {
                        same = false
                        break
                    }
                    if same {
                        let candidate = window[pos + ngram]
                        if !out.contains(candidate) { out.append(candidate) }
                    }
                }
            }
        }
    }
}
