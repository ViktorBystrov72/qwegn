// QWTypedAnswerDecoder.swift — Pointer max_pool1d + RFC6901 JSON spans, quantiles, and 4D Halton box IoU.
import Foundation

public struct QWJSONStringSpan: Sendable, Hashable, Equatable {
    public let utf8Start: Int
    public let utf8End: Int
    public let path: String

    public init(utf8Start: Int, utf8End: Int, path: String) {
        self.utf8Start = utf8Start
        self.utf8End = utf8End
        self.path = path
    }
}

public enum QWJSONSpanFinder {
    /// Finds all string literal inner ranges (excluding quotes) and their RFC 6901 JSON Pointer paths.
    public static func stringPaths(in jsonText: String) -> [QWJSONStringSpan] {
        let bytes = Array(jsonText.utf8)
        var cursor = 0
        var out: [QWJSONStringSpan] = []

        func skipWS() {
            while cursor < bytes.count {
                let b = bytes[cursor]
                if b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09 { cursor += 1 } else { break }
            }
        }

        func parseStringLiteral() -> (value: String, innerStart: Int, innerEnd: Int)? {
            skipWS()
            guard cursor < bytes.count, bytes[cursor] == 0x22 else { return nil }
            cursor += 1
            let innerStart = cursor
            var escaped = false
            while cursor < bytes.count {
                let b = bytes[cursor]
                if escaped { escaped = false; cursor += 1; continue }
                if b == 0x5C { escaped = true; cursor += 1; continue }
                if b == 0x22 {
                    let innerEnd = cursor
                    cursor += 1
                    let rawLiteral = "\"" + String(decoding: bytes[innerStart..<innerEnd], as: UTF8.self) + "\""
                    let decoded = (try? JSONDecoder().decode(String.self, from: Data(rawLiteral.utf8)))
                        ?? String(decoding: bytes[innerStart..<innerEnd], as: UTF8.self)
                    return (decoded, innerStart, innerEnd)
                }
                cursor += 1
            }
            return nil
        }

        func escapeRFC6901(_ key: String) -> String {
            key.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1")
        }

        func parseValue(path: String) -> Bool {
            skipWS()
            guard cursor < bytes.count else { return false }
            let ch = bytes[cursor]
            if ch == 0x7B {
                cursor += 1
                skipWS()
                if cursor < bytes.count && bytes[cursor] == 0x7D { cursor += 1; return true }
                while cursor < bytes.count {
                    guard let keyLit = parseStringLiteral() else { return false }
                    skipWS()
                    guard cursor < bytes.count, bytes[cursor] == 0x3A else { return false }
                    cursor += 1
                    guard parseValue(path: "\(path)/\(escapeRFC6901(keyLit.value))") else { return false }
                    skipWS()
                    if cursor < bytes.count, bytes[cursor] == 0x2C { cursor += 1 }
                    else if cursor < bytes.count, bytes[cursor] == 0x7D { cursor += 1; return true }
                    else { return false }
                }
                return false
            }
            if ch == 0x5B {
                cursor += 1
                skipWS()
                if cursor < bytes.count && bytes[cursor] == 0x5D { cursor += 1; return true }
                var idx = 0
                while cursor < bytes.count {
                    guard parseValue(path: "\(path)/\(idx)") else { return false }
                    idx += 1
                    skipWS()
                    if cursor < bytes.count, bytes[cursor] == 0x2C { cursor += 1 }
                    else if cursor < bytes.count, bytes[cursor] == 0x5D { cursor += 1; return true }
                    else { return false }
                }
                return false
            }
            if ch == 0x22 {
                guard let s = parseStringLiteral() else { return false }
                out.append(QWJSONStringSpan(utf8Start: s.innerStart, utf8End: s.innerEnd, path: path))
                return true
            }
            while cursor < bytes.count {
                let b = bytes[cursor]
                if b == 0x2C || b == 0x7D || b == 0x5D || b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09 { break }
                cursor += 1
            }
            return true
        }

        guard parseValue(path: "") else { return [] }
        return out
    }

    public static func unescapeJSONStringSlice(_ slice: String) -> String {
        guard let decoded = try? JSONDecoder().decode(String.self, from: Data("\"\(slice)\"".utf8)) else {
            return slice
        }
        return decoded
    }
}

public enum QWTypedAnswerDecoder {
    private static let haltonSamples512: [(Double, Double, Double, Double)] = {
        func radicalInverse(_ index: Int, base: Int) -> Double {
            var i = index
            var val = 0.0
            var scale = 1.0
            while i > 0 {
                let digit = i % base
                i /= base
                scale /= Double(base)
                val += Double(digit) * scale
            }
            return val
        }
        return (1...512).map { i in
            (radicalInverse(i, base: 2), radicalInverse(i, base: 3),
             radicalInverse(i, base: 5), radicalInverse(i, base: 7))
        }
    }()

    public static func decode(
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        chain: String
    ) throws -> [QWDecisionAnswer] {
        var answers: [QWDecisionAnswer] = []
        answers.reserveCapacity(built.questionBindings.count)
        for bound in built.questionBindings {
            let q = bound.question
            let effectiveChain = q.chain ?? chain
            let answer: QWDecisionAnswer
            switch q.kind {
            case .noul:
                answer = try decodeNoul(bound: bound, built: built, output: output, calibration: calibration, chain: effectiveChain)
            case .choice:
                answer = try decodeChoice(bound: bound, built: built, output: output, calibration: calibration, chain: effectiveChain)
            case .multiChoice:
                answer = try decodeMultiChoice(bound: bound, built: built, output: output, calibration: calibration, chain: effectiveChain)
            case .score:
                answer = try decodeScore(
                    bound: bound, built: built, output: output, calibration: calibration,
                    chain: effectiveChain, defaultReadout: built.defaultScoreReadout
                )
            case .read:
                answer = decodeRead(bound: bound, output: output, calibration: calibration)
            case .estimate:
                answer = decodeEstimate(bound: bound, built: built, output: output, calibration: calibration)
            case .box, .boxes:
                answer = decodeBoxQuestion(
                    bound: bound, built: built, output: output, calibration: calibration,
                    edgesMode: built.boxEdgesMode,
                    numbersThousandths: built.usesNumbersThousandthsGrid
                )
            }
            answers.append(answer)
        }
        return answers
    }

    private static func resolveLogits(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput
    ) -> [Double] {
        guard let first = bound.slotBindings.first else { return [] }
        if case .pairRelevance = first.kind {
            return bound.slotBindings.map { sb in
                let so = output.slotOutputs[sb.slotIndex]
                if let p = so.pooledLogits, !p.isEmpty {
                    return p.count == 1 ? Double(p[0]) : Double(p[p.count - 1] - p[0])
                }
                return Double(so.markerLogits?.first ?? 0)
            }
        }
        let target = built.request.slots[first.slotIndex]
        let so = output.slotOutputs[first.slotIndex]
        if case .fixedLabelSubset(_, let classIds) = first.kind, let pooled = so.pooledLogits {
            return classIds.map { cid in Double(cid < pooled.count ? pooled[cid] : -1e4) }
        }
        if let markers = so.markerLogits { return markers.map(Double.init) }
        if let pooled = so.pooledLogits { return pooled.map(Double.init) }
        return so.groupScores(for: target.candidateTokenGroups)
    }

    private static func normalizedEntropy(_ probs: [Double]) -> Double {
        let k = probs.count
        guard k > 1 else { return 1.0 }
        let h = probs.reduce(0.0) { acc, p in p > 1e-15 ? acc - p * log(p) : acc }
        return max(0.0, min(1.0, 1.0 - h / log(Double(k))))
    }

    private static func decodeNoul(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        chain: String
    ) throws -> QWDecisionAnswer {
        var logits = resolveLogits(bound: bound, built: built, output: output)
        if logits.count == 1 { logits = [0.0, logits[0]] }
        if logits.count < 2 { logits = [0.0, 0.0] }
        let pair = Array(logits.prefix(2))
        let temp = QWCalibrationEngine.temperature(profile: calibration, kind: .noul, cardinality: 2, chain: chain)
        let rawProbs = QWCalibrationEngine.calibratedSoftmax(logits: pair, temperature: temp)
        let renorm = try QWCalibrationEngine.renormalize(questionName: bound.question.name, keys: ["no", "yes"], rawProbs: rawProbs)
        let pNo = renorm[0].value
        let pYes = renorm[1].value
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .noul,
            noul: pYes >= 0.5,
            confidence: max(pNo, pYes),
            entropyConfidence: normalizedEntropy([pNo, pYes]),
            probs: renorm
        )
    }

    private static func decodeChoice(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        chain: String
    ) throws -> QWDecisionAnswer {
        let keys = bound.question.criteria.map(\.key)
        guard !keys.isEmpty else {
            throw QWDecisionError.invalidQuestion("choice question '\(bound.question.name)' requires criteria")
        }
        var logits = resolveLogits(bound: bound, built: built, output: output)
        if logits.count < keys.count { logits += [Double](repeating: -1e4, count: keys.count - logits.count) }
        logits = Array(logits.prefix(keys.count))
        let temp = QWCalibrationEngine.temperature(profile: calibration, kind: .choice, cardinality: keys.count, chain: chain)
        let rawProbs = QWCalibrationEngine.calibratedSoftmax(logits: logits, temperature: temp)
        let renorm = try QWCalibrationEngine.renormalize(questionName: bound.question.name, keys: keys, rawProbs: rawProbs)
        let vals = renorm.map(\.value)
        let bestIdx = QWCalibrationEngine.argmaxIndex(vals)
        let conf = vals[bestIdx]
        let ent = normalizedEntropy(vals)
        let abstained = bound.question.abstainBelow.map { conf < $0 } ?? false
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .choice,
            choice: abstained ? nil : keys[bestIdx],
            abstained: abstained,
            confidence: conf,
            entropyConfidence: ent,
            probs: renorm
        )
    }

    private static func decodeMultiChoice(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        chain: String
    ) throws -> QWDecisionAnswer {
        let keys = bound.question.criteria.map(\.key)
        guard !keys.isEmpty else {
            throw QWDecisionError.invalidQuestion("multiple choice '\(bound.question.name)' requires criteria")
        }
        let temp = QWCalibrationEngine.temperature(profile: calibration, kind: .multiChoice, cardinality: keys.count, chain: chain)
        let threshold = bound.question.selectThreshold ?? 0.5
        var probs: [QWDecisionProbability] = []
        var selected: [String] = []
        for (idx, sb) in bound.slotBindings.enumerated() where idx < keys.count {
            let so = output.slotOutputs[sb.slotIndex]
            let target = built.request.slots[sb.slotIndex]
            let scores: [Double]
            if let m = so.markerLogits, !m.isEmpty { scores = m.map(Double.init) }
            else if let p = so.pooledLogits, !p.isEmpty { scores = p.map(Double.init) }
            else { scores = so.groupScores(for: target.candidateTokenGroups) }
            let rawYes: Double
            if scores.count >= 2 {
                // `multiple` choice slots are typed answers evaluated un-tempered before `calibration.tables["multiple"]`.
                rawYes = QWCalibrationEngine.calibratedSoftmax(logits: Array(scores.prefix(2)), temperature: 1.0)[1]
            } else {
                let s = (scores.first ?? 0) / max(1e-4, temp)
                rawYes = 1.0 / (1.0 + exp(-s))
            }
            let calYes = min(1.0, max(0.0, QWCalibrationEngine.through(rawYes, knots: calibration.tables["multiple"])))
            let rounded = QWCalibrationEngine.roundPy(calYes)
            probs.append(QWDecisionProbability(key: keys[idx], value: rounded))
            if rounded >= threshold { selected.append(keys[idx]) }
        }
        _ = temp
        let maxP = probs.map(\.value).max() ?? 0.0
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .multiChoice,
            selectedChoices: selected,
            confidence: maxP,
            probs: probs
        )
    }

    private static func decodeScore(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        chain: String,
        defaultReadout: QWScoreReadout
    ) throws -> QWDecisionAnswer {
        let count = bound.question.criteria.count
        guard count >= 2 else {
            throw QWDecisionError.invalidQuestion("score question '\(bound.question.name)' requires >= 2 levels")
        }
        let keys = (0..<count).map(String.init)
        var logits = resolveLogits(bound: bound, built: built, output: output)
        if logits.count < count { logits += [Double](repeating: -1e4, count: count - logits.count) }
        logits = Array(logits.prefix(count))
        let temp = QWCalibrationEngine.temperature(profile: calibration, kind: .score, cardinality: count, chain: chain)
        let rawProbs = QWCalibrationEngine.calibratedSoftmax(logits: logits, temperature: temp)
        let renorm = try QWCalibrationEngine.renormalize(questionName: bound.question.name, keys: keys, rawProbs: rawProbs)
        let vals = renorm.map(\.value)
        let bestIdx = QWCalibrationEngine.argmaxIndex(vals)
        let mean = vals.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
        let second = vals.enumerated().reduce(0.0) { $0 + Double($1.offset * $1.offset) * $1.element }
        let sigma = sqrt(max(0.0, second - mean * mean))
        let mode = bound.question.scoreReadout ?? defaultReadout
        let readoutVal: Double
        switch mode {
        case .mean:
            readoutVal = mean
        case .argmax:
            readoutVal = Double(bestIdx)
        case .median:
            var cum = 0.0
            var med = Double(count - 1)
            for (i, p) in vals.enumerated() {
                cum += p
                if cum >= 0.5 {
                    med = Double(i)
                    break
                }
            }
            readoutVal = med
        }
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .score,
            confidence: vals[bestIdx],
            entropyConfidence: normalizedEntropy(vals),
            probs: renorm,
            score: readoutVal,
            sigma: sigma,
            spreadReported: mode != .argmax
        )
    }

    private static func logSoftmax(_ logits: [Float]) -> [Double] {
        guard let maxVal = logits.max(), maxVal.isFinite else {
            return [Double](repeating: -Double.infinity, count: logits.count)
        }
        let m = Double(maxVal)
        let sumExp = logits.reduce(0.0) { acc, v in
            v.isFinite ? (acc + exp(Double(v) - m)) : acc
        }
        let lse = m + log(max(1e-30, sumExp))
        return logits.map { $0.isFinite ? (Double($0) - lse) : -Double.infinity }
    }

    private static func decodeRead(
        bound: QWBoundQuestion,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile
    ) -> QWDecisionAnswer {
        guard let sb = bound.slotBindings.first else {
            return QWDecisionAnswer(name: bound.question.name, kind: .read, found: false, foundProb: 0)
        }
        let so = output.slotOutputs[sb.slotIndex]
        guard let rawStart = so.pointerStartLogits, let rawEnd = so.pointerEndLogits, !rawStart.isEmpty, rawStart.count == rawEnd.count else {
            return QWDecisionAnswer(name: bound.question.name, kind: .read, found: false, foundProb: 0)
        }
        let useExplicitMask = !bound.pointerTokenSpans.isEmpty
        let maskedStart: [Float]
        let maskedEnd: [Float]
        let totalT: Int
        if useExplicitMask {
            totalT = min(rawStart.count, bound.pointerTokenSpans.count)
            maskedStart = (0..<totalT).map { i in (i == 0 || bound.pointerTokenSpans[i] != nil) ? rawStart[i] : -Float.infinity }
            maskedEnd = (0..<totalT).map { i in (i == 0 || bound.pointerTokenSpans[i] != nil) ? rawEnd[i] : -Float.infinity }
        } else {
            totalT = min(rawStart.count, bound.stateTokenSpans.count + 1)
            maskedStart = Array(rawStart.prefix(totalT))
            maskedEnd = Array(rawEnd.prefix(totalT))
        }
        let logStart = logSoftmax(maskedStart)
        let logEnd = logSoftmax(maskedEnd)
        let rawFound = max(0.0, min(1.0, -expm1(logStart[0])))
        let calFound = min(1.0, max(0.0, QWCalibrationEngine.through(
            rawFound, knots: calibration.tables["found:\(bound.question.readKind.rawValue)"]
        )))
        guard calFound >= 0.5, totalT > 1 else {
            return QWDecisionAnswer(name: bound.question.name, kind: .read, confidence: 1.0 - calFound, found: false, foundProb: calFound)
        }
        let cap = min(bound.question.readKind.spanTokenCap, totalT - 1)
        var bestScore = -Double.infinity
        var bestS = -1
        var bestE = -1
        for s in 1..<totalT {
            if useExplicitMask && bound.pointerTokenSpans[s] == nil { continue }
            guard logStart[s].isFinite else { continue }
            let eHi = min(totalT, s + cap)
            var localBestE = -1
            var localBestEndScore = -Double.infinity
            for e in s..<eHi {
                if useExplicitMask && bound.pointerTokenSpans[e] == nil { continue }
                if logEnd[e] > localBestEndScore {
                    localBestEndScore = logEnd[e]
                    localBestE = e
                }
            }
            guard localBestE >= s else { continue }
            let totalScore = logStart[s] + localBestEndScore
            if totalScore > bestScore {
                bestScore = totalScore
                bestS = s
                bestE = localBestE
            }
        }
        guard bestS >= 1, bestE >= bestS else {
            return QWDecisionAnswer(name: bound.question.name, kind: .read, confidence: 1.0 - calFound, found: false, foundProb: calFound)
        }
        let startSpan = useExplicitMask ? bound.pointerTokenSpans[bestS] : bound.stateTokenSpans[bestS - 1]
        let endSpan = useExplicitMask ? bound.pointerTokenSpans[bestE] : bound.stateTokenSpans[bestE - 1]
        guard let sSpan = startSpan, let eSpan = endSpan else {
            return QWDecisionAnswer(name: bound.question.name, kind: .read, confidence: 1.0 - calFound, found: false, foundProb: calFound)
        }
        let escLo = sSpan.utf8Range.lowerBound
        let escHi = eSpan.utf8Range.upperBound
        let origRange = bound.escapedState.originalUTF8Range(forEscapedUTF8Range: escLo..<escHi)
        let origBytes = Array(bound.escapedState.original.utf8)
        var sLo = max(0, min(origRange.lowerBound, origBytes.count))
        var sHi = max(sLo, min(origRange.upperBound, origBytes.count))
        while sLo < sHi && (origBytes[sLo] == 0x20 || origBytes[sLo] == 0x0A || origBytes[sLo] == 0x0D || origBytes[sLo] == 0x09) {
            sLo += 1
        }
        while sHi > sLo && (origBytes[sHi - 1] == 0x20 || origBytes[sHi - 1] == 0x0A || origBytes[sHi - 1] == 0x0D || origBytes[sHi - 1] == 0x09) {
            sHi -= 1
        }

        let jsonPaths = QWJSONSpanFinder.stringPaths(in: bound.escapedState.original)
        let held = jsonPaths.filter { $0.utf8Start < sHi && sLo < $0.utf8End }
        var matchedPath: String?
        let extractedText: String
        if held.count == 1 {
            sLo = max(sLo, held[0].utf8Start)
            sHi = min(sHi, held[0].utf8End)
            matchedPath = held[0].path
            let rawSlice = String(decoding: origBytes[sLo..<sHi], as: UTF8.self)
            extractedText = QWJSONSpanFinder.unescapeJSONStringSlice(rawSlice)
        } else {
            extractedText = String(decoding: origBytes[sLo..<sHi], as: UTF8.self)
        }
        var parsedNum: Double?
        if bound.question.readKind == .number {
            parsedNum = parseReadNumber(extractedText)
        }
        let parsedDate: String? = bound.question.readKind == .date ? extractedText : nil
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .read,
            confidence: calFound,
            text: extractedText,
            number: parsedNum,
            date: parsedDate,
            span: sLo..<sHi,
            jsonPath: matchedPath,
            found: true,
            foundProb: calFound
        )
    }

    private static func gridCrossing(
        probs: [Double],
        low: Double,
        high: Double,
        targetQuantile: Double
    ) -> Double {
        let size = probs.count
        guard size > 0 else { return low }
        let step = size > 1 ? (high - low) / Double(size - 1) : 1.0
        let total = probs.reduce(0, +)
        guard total > 0 else { return 0.5 * (low + high) }
        var mass = 0.0
        for i in 0..<size {
            let value = low + Double(i) * step
            let share = probs[i] / total
            if share > 0 && mass + share >= targetQuantile {
                return min(high, max(low, value - step / 2.0 + (targetQuantile - mass) / share * step))
            }
            mass += share
        }
        return high
    }

    private static func decodeEstimate(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile
    ) -> QWDecisionAnswer {
        guard let sb = bound.slotBindings.first,
              case .estimateGrid(let low, let high, let size) = sb.kind else {
            return QWDecisionAnswer(name: bound.question.name, kind: .estimate)
        }
        let target = built.request.slots[sb.slotIndex]
        let logits = output.slotOutputs[sb.slotIndex].groupScores(for: target.candidateTokenGroups)
        let probs = QWCalibrationEngine.calibratedSoftmax(logits: Array(logits.prefix(size)), temperature: 1.0)
        let q10Target = min(0.49, max(0.01, QWCalibrationEngine.level(0.10, knots: calibration.tables["estimate"])))
        let q90Target = min(0.99, max(0.51, QWCalibrationEngine.level(0.90, knots: calibration.tables["estimate"])))
        let p10 = gridCrossing(probs: probs, low: low, high: high, targetQuantile: q10Target)
        let med = gridCrossing(probs: probs, low: low, high: high, targetQuantile: 0.5)
        let p90 = gridCrossing(probs: probs, low: low, high: high, targetQuantile: q90Target)
        let peakProb = probs.max() ?? 0.0
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: .estimate,
            confidence: peakProb,
            estimate: med,
            interval: [min(p10, med), max(p90, med)]
        )
    }

    private static func decodeBoxQuestion(
        bound: QWBoundQuestion,
        built: QWBuiltDecisionBatch,
        output: QWDecisionBackendOutput,
        calibration: QWCalibrationProfile,
        edgesMode: String,
        numbersThousandths: Bool
    ) -> QWDecisionAnswer {
        let presenceSlots = bound.slotBindings.filter {
            if case .boxPresence = $0.kind { return true }
            return false
        }
        let edgeSlots = bound.slotBindings.filter {
            if case .boxEdge = $0.kind { return true }
            return false
        }
        let isMany = (bound.question.kind == .boxes)
        let foundKey = isMany ? "found:boxes" : "found:box"
        let confKey = isMany ? "boxes" : "box"
        let gridHigh = numbersThousandths ? 0.999 : 1.0
        var presence: [Double] = []
        var below = 1.0
        var boxes: [QWDecisionBox] = []

        for ps in presenceSlots {
            let target = built.request.slots[ps.slotIndex]
            let scores = output.slotOutputs[ps.slotIndex].groupScores(for: target.candidateTokenGroups)
            let rawYes = QWCalibrationEngine.calibratedSoftmax(logits: scores, temperature: 1.0).last ?? 0.0
            let calFound = min(1.0, max(0.0, QWCalibrationEngine.through(rawYes, knots: calibration.tables[foundKey])))
            below = isMany ? min(below, calFound) : calFound
            presence.append(below)
        }
        for (blkIdx, below) in presence.enumerated() {
            if below < 0.5 {
                if isMany { break }
                return QWDecisionAnswer(name: bound.question.name, kind: .box, confidence: 0.0, found: false, foundProb: below)
            }
            let blockEdges = edgeSlots.filter {
                if case .boxEdge(let b, _, _, _) = $0.kind { return b == blkIdx }
                return false
            }
            guard blockEdges.count == 4 else { continue }
            var edgeDists: [[Double]] = []
            var edgeVals: [Double] = []
            for es in blockEdges {
                let eTarget = built.request.slots[es.slotIndex]
                let eLogits = output.slotOutputs[es.slotIndex].groupScores(for: eTarget.candidateTokenGroups)
                let dist = QWCalibrationEngine.calibratedSoftmax(logits: eLogits, temperature: 1.0)
                edgeDists.append(dist)
                if edgesMode == "interp" {
                    edgeVals.append(gridCrossing(probs: dist, low: 0.0, high: gridHigh, targetQuantile: 0.5))
                } else {
                    let step = dist.count > 1 ? gridHigh / Double(dist.count - 1) : 1.0
                    var cum = 0.0
                    let tot = dist.reduce(0, +)
                    var v = gridHigh
                    for (idx, p) in dist.enumerated() {
                        cum += p / max(1e-30, tot)
                        if cum >= 0.5 { v = Double(idx) * step; break }
                    }
                    edgeVals.append(v)
                }
            }
            let gridSize = edgeDists[0].count
            let step = gridSize > 1 ? gridHigh / Double(gridSize - 1) : 1.0
            let (top, bottom) = orderedEdges(edgeVals[0], edgeVals[1], step: step)
            let (left, right) = orderedEdges(edgeVals[2], edgeVals[3], step: step)
            let rawConf = haltonConfidence(edgeDists: edgeDists, target: (top, bottom, left, right), high: gridHigh)
            let calConf = min(1.0, max(0.0, QWCalibrationEngine.through(rawConf, knots: calibration.tables[confKey])))
            boxes.append(QWDecisionBox(top: top, bottom: bottom, left: left, right: right, found: below, confidence: calConf))
        }
        if isMany {
            boxes.sort { lhs, rhs in
                if lhs.left != rhs.left { return lhs.left < rhs.left }
                return lhs.top < rhs.top
            }
        }
        let atLeast = isMany && !boxes.isEmpty && boxes.count == presence.count
        return QWDecisionAnswer(
            name: bound.question.name,
            kind: bound.question.kind,
            confidence: boxes.first?.confidence ?? 0.0,
            found: !boxes.isEmpty,
            foundProb: presence.first,
            boxes: boxes,
            atLeast: atLeast,
            presence: presence
        )
    }

    private static func orderedEdges(_ a: Double, _ b: Double, step: Double) -> (Double, Double) {
        if a < b { return (a, b) }
        let half = step * 0.5
        let middle = min(1.0 - half, max(half, (a + b) * 0.5))
        return (middle - half, middle + half)
    }

    private static func haltonConfidence(
        edgeDists: [[Double]],
        target: (top: Double, bottom: Double, left: Double, right: Double),
        high: Double
    ) -> Double {
        let cdfs: [[Double]] = edgeDists.map { dist in
            let tot = max(1e-30, dist.reduce(0, +))
            var running = 0.0
            return dist.map { p in running += p / tot; return running }
        }
        let size = edgeDists[0].count
        let step = size > 1 ? high / Double(size - 1) : 1.0
        func sampleVal(_ cdf: [Double], _ u: Double) -> Double {
            var lo = 0
            var hi = cdf.count
            while lo < hi {
                let mid = (lo + hi) >> 1
                if cdf[mid] < u { lo = mid + 1 } else { hi = mid }
            }
            return Double(min(size - 1, lo)) * step
        }
        let targetArea = max(1e-12, (target.bottom - target.top) * (target.right - target.left))
        var hits = 0
        for (u0, u1, u2, u3) in haltonSamples512 {
            let (t, b) = orderedEdges(sampleVal(cdfs[0], u0), sampleVal(cdfs[1], u1), step: step)
            let (l, r) = orderedEdges(sampleVal(cdfs[2], u2), sampleVal(cdfs[3], u3), step: step)
            let overlapH = max(0.0, min(b, target.bottom) - max(t, target.top))
            let overlapW = max(0.0, min(r, target.right) - max(l, target.left))
            let overlap = overlapH * overlapW
            let boxArea = max(1e-12, (b - t) * (r - l))
            let iou = overlap / (targetArea + boxArea - overlap)
            if iou >= 0.5 { hits += 1 }
        }
        return Double(hits) / 512.0
    }

    private static let readNumberCurrencies = [
        "USD", "EUR", "GBP", "CAD", "AUD", "NZD", "INR", "MXN", "CNY", "RMB",
        "HKD", "SGD", "CHF", "JPY", "KRW", "BRL", "RUB", "PLN", "SEK", "NOK",
        "DKK", "TRY", "IDR", "VND", "CLP", "Rs.", "Rs", "R$",
        "$", "£", "€", "¥", "￥", "₹", "₽", "₩", "₺",
    ]

    private static func stripCurrencyEdge(_ s: String, fromStart: Bool) -> String? {
        for cur in readNumberCurrencies {
            if fromStart {
                if s.count >= cur.count,
                   s.prefix(cur.count).compare(cur, options: .caseInsensitive) == .orderedSame {
                    let rest = s.dropFirst(cur.count)
                    if cur.last?.isLetter == true, let ch = rest.first, ch.isLetter { continue }
                    return String(rest).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            } else if s.count >= cur.count,
                      s.suffix(cur.count).compare(cur, options: .caseInsensitive) == .orderedSame {
                let head = s.dropLast(cur.count)
                if cur.first?.isLetter == true, let ch = head.last, ch.isLetter { continue }
                return String(head).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return nil
    }

    private static func parseReadNumber(_ raw: String) -> Double? {
        func clean(_ t: String) -> String {
            t.replacingOccurrences(of: "\u{2212}", with: "-")
                .replacingOccurrences(of: "\u{00A0}", with: " ")
                .replacingOccurrences(of: "\u{202F}", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var s = clean(raw)
        if s.hasSuffix("%") { s = clean(String(s.dropLast())) }
        var neg = false
        if s.hasPrefix("(") && s.hasSuffix(")") {
            neg = true
            s = clean(String(s.dropFirst().dropLast()))
            if s.hasPrefix("+") || s.hasPrefix("-") { return nil }
        }
        var gotCur = false
        var gotSign = neg
        for _ in 0..<2 {
            if !gotSign && (s.hasPrefix("-") || s.hasPrefix("+")) {
                neg = neg || s.hasPrefix("-")
                gotSign = true
                s = clean(String(s.dropFirst()))
            } else if !gotCur, let stripped = stripCurrencyEdge(s, fromStart: true) ?? stripCurrencyEdge(s, fromStart: false) {
                gotCur = true
                s = clean(stripped)
            }
        }
        let scaleWords: [(String, Double)] = [
            ("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3),
        ]
        var scale = 1.0
        var trimmedScale = s
        if trimmedScale.hasSuffix(".") { trimmedScale.removeLast() }
        let lower = trimmedScale.lowercased()
        if let hit = scaleWords.first(where: { lower.hasSuffix($0.0) }) {
            let cut = trimmedScale.index(trimmedScale.endIndex, offsetBy: -hit.0.count)
            s = clean(String(trimmedScale[..<cut]))
            scale = hit.1
        } else {
            let shortScales: [(String, Double)] = [("bn", 1e9), ("k", 1e3), ("m", 1e6), ("b", 1e9)]
            if let hit = shortScales.first(where: { lower.hasSuffix($0.0) }) {
                let cut = trimmedScale.index(trimmedScale.endIndex, offsetBy: -hit.0.count)
                let head = clean(String(trimmedScale[..<cut]))
                if head.last?.isNumber == true {
                    s = head
                    scale = hit.1
                }
            }
        }
        let numeric = s.replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "\u{2019}", with: "")
        guard let v = Double(numeric), v.isFinite else { return nil }
        let out = (neg ? -v : v) * scale
        return out.isFinite ? out : nil
    }
}
