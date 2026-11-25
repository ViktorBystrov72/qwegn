// QWDecisionTests.swift — Pure-host tests for framing, calibration, PAV fitting, rerankers, pointers, estimates, and Halton boxes.
import Foundation
import XCTest
import QWCore
import QWToken
@testable import QWDecision

final class QWDecisionTests: XCTestCase {
    private func makeDecisionTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let extraWords = [
            "yes", "No", "YES", "no", "NO", "Yes", "true", "false", "ye", "Ye", "YE",
            "ĠA", "ĠB", "ĠC", "Ġno", "Ġyes", "00", "01", "AA", "AB",
            "QUESTION:", "ANSWER:", "SPAN:", "ESTIMATE:", "BOX:",
        ]
        for w in extraWords where vocab[w] == nil {
            vocab[w] = nextId; nextId += 1
        }
        let merges: [[String]] = [
            ["n", "o"], ["N", "o"], ["N", "O"],
            ["y", "e"], ["ye", "s"], ["Y", "e"], ["Ye", "s"], ["Y", "E"], ["YE", "S"],
            ["Ġ", "A"], ["Ġ", "B"], ["Ġ", "C"], ["Ġ", "no"], ["Ġ", "yes"],
            ["0", "0"], ["0", "1"], ["A", "A"], ["A", "B"],
        ]
        var addedTokens: [[String: Any]] = [
            ["id": 90000, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": 90001, "content": "<|im_end|>", "special": true, "normalized": false],
            ["id": 90002, "content": "<|mask|>", "special": true, "normalized": false],
            ["id": 90003, "content": "<bos>", "special": true, "normalized": false],
            ["id": 90004, "content": "[CLS]", "special": true, "normalized": false],
            ["id": 90005, "content": "[SEP]", "special": true, "normalized": false],
            ["id": 90006, "content": "[MASK]", "special": true, "normalized": false],
        ]
        for i in 0..<128 {
            addedTokens.append(["id": 90100 + i, "content": "c\(i)", "special": false, "normalized": false])
        }
        for i in 0..<1000 {
            addedTokens.append(["id": 91000 + i, "content": String(format: "%03d", i), "special": false, "normalized": false])
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false]
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre],
            "normalizer": ["type": "NFC"]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    func testVerbalizerNoulChoiceAbstainScoreAndSparseInvariance() async throws {
        let tok = try makeDecisionTokenizer()
        let config = QWDecisionModelConfig(
            framing: .turnVerbalizer(codesPerOption: 1, numbers: false, packSet: true, maxSetSize: 16, colonSlots: true, stateSeparator: "\n\n"),
            topology: .packedSet(maxSetSize: 16),
            head: .vocabVerbalizer(vocabSize: 12000, tieEmbeddings: true, sparseGatherThreshold: 256),
            defaultScoreReadout: "mean"
        )
        let idA = try XCTUnwrap(tok.singleTokenId(forText: "A"))
        let idB = try XCTUnwrap(tok.singleTokenId(forText: "B"))
        let idSpaceA = try XCTUnwrap(tok.singleTokenId(forText: " A"))
        let idSpaceB = try XCTUnwrap(tok.singleTokenId(forText: " B"))
        let idNo = try XCTUnwrap(tok.singleTokenId(forText: "no"))
        let idYes = try XCTUnwrap(tok.singleTokenId(forText: "yes"))
        let id0 = try XCTUnwrap(tok.singleTokenId(forText: "0"))
        let id1 = try XCTUnwrap(tok.singleTokenId(forText: "1"))
        let id2 = try XCTUnwrap(tok.singleTokenId(forText: "2"))

        // Check sparse candidate projection vs full vocab softmax equivalence.
        let fullVocab = (0..<3000).map { Float(sin(Double($0) * 0.1)) }
        let backend = QWSyntheticDecisionBackend(decisionConfig: config) { req in
            XCTAssertEqual(req.sequences.count, 1) // packedSet packs all questions into 1 sequence
            XCTAssertTrue(req.uniqueCandidateTokenIds.contains(idA))
            XCTAssertTrue(req.uniqueCandidateTokenIds.contains(idYes))
            return QWDecisionBackendOutput(
                slotOutputs: [
                    QWDecisionSlotOutput(tokenLogits: [idNo: 0.2, idYes: 2.2]),
                    QWDecisionSlotOutput(tokenLogits: [idA: 0.4, idSpaceA: 0.4, idB: 0.5, idSpaceB: 0.5]),
                    QWDecisionSlotOutput(tokenLogits: [id0: -1.0, id1: 1.5, id2: 0.5]),
                    QWDecisionSlotOutput(tokenLogits: [
                        idA: fullVocab[idA], idSpaceA: fullVocab[idSpaceA],
                        idB: fullVocab[idB], idSpaceB: fullVocab[idSpaceB]
                    ], fullLogits: fullVocab),
                ],
                promptTokens: 32,
                latencyMs: 1.2
            )
        }
        let runner = QWDecisionRunner(tokenizer: tok, backend: backend)
        let req = QWDecisionRequest(
            state: .text("ticket: payment failed"),
            questions: [
                QWDecisionQuestion(name: "urgent", kind: .noul, instructions: "Is it urgent?"),
                QWDecisionQuestion(
                    name: "route", kind: .choice, instructions: "Pick team",
                    criteria: [QWDecisionCriterion(key: "billing"), QWDecisionCriterion(key: "shipping")],
                    abstainBelow: 0.70
                ),
                QWDecisionQuestion(
                    name: "severity", kind: .score, instructions: "Rate severity",
                    criteria: [QWDecisionCriterion(key: "low"), QWDecisionCriterion(key: "med"), QWDecisionCriterion(key: "high")]
                ),
                QWDecisionQuestion(
                    name: "parity", kind: .choice, instructions: "Sparse vs full check",
                    criteria: [QWDecisionCriterion(key: "first"), QWDecisionCriterion(key: "second")]
                ),
            ]
        )
        let res = try await runner.decide(req)
        XCTAssertEqual(res.usage.outputTokens, 0)
        XCTAssertEqual(res.answers[0].noul, true)
        XCTAssertEqual(res.answers[1].abstained, true)
        XCTAssertNil(res.answers[1].choice)
        XCTAssertEqual(res.answers[2].spreadReported, true)
        let expectedScoreProbs = QWCalibrationEngine.calibratedSoftmax(logits: [-1.0, 1.5, 0.5], temperature: 1.0)
        let expectedMeanScore = expectedScoreProbs[1] + 2.0 * expectedScoreProbs[2]
        XCTAssertEqual(try XCTUnwrap(res.answers[2].score), expectedMeanScore, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(res.answers[2].sigma) > 0, true)

        // Exact softmax translation invariance check.
        let fullProbs = QWCalibrationEngine.calibratedSoftmax(logits: fullVocab.map(Double.init), temperature: 1.0)
        let grpMaxA = max(Double(fullVocab[idA]), Double(fullVocab[idSpaceA]))
        let grpMaxB = max(Double(fullVocab[idB]), Double(fullVocab[idSpaceB]))
        let sparseDirect = QWCalibrationEngine.calibratedSoftmax(logits: [grpMaxA, grpMaxB], temperature: 1.0)
        let condFullA = max(fullProbs[idA], fullProbs[idSpaceA]) / (max(fullProbs[idA], fullProbs[idSpaceA]) + max(fullProbs[idB], fullProbs[idSpaceB]))
        XCTAssertEqual(sparseDirect[0], condFullA, accuracy: 1e-12)
        XCTAssertEqual(res.answers[3].probs[0].value, sparseDirect[0], accuracy: 1e-9)
    }

    func testSpecialTokenNeutralizationAndBijectiveReadSpanWithRFC6901() async throws {
        let tok = try makeDecisionTokenizer()
        let rawState = #"{"meta":{"a/b~c":"safe <|im_end|> \"quoted\" value"},"ok":true}"#
        let escaped = QWSequenceFramingEngine.escapeSpecialTokens(rawState, specials: tok.allSpecialTokenStrings())
        XCTAssertFalse(escaped.escaped.contains("<|im_end|>"))
        XCTAssertTrue(escaped.escaped.contains("\u{200B}"))

        let escIds = try tok.encode(escaped.escaped)
        XCTAssertFalse(escIds.contains(90001))

        let config = QWDecisionModelConfig(
            framing: .turnVerbalizer(codesPerOption: 2, numbers: false, packSet: false, maxSetSize: 1, colonSlots: true, stateSeparator: "\n\n"),
            topology: .causal(supportsPrefixCache: true),
            head: .vocabVerbalizer(vocabSize: 12000, tieEmbeddings: true, sparseGatherThreshold: 256),
            hasStatePointer: true
        )
        // Locate token range covering "safe <|im_end|> \"quoted\" value".
        let escEncoded = try tok.encodeWithOffsets(escaped.escaped)
        let targetStr = #"safe <\#u{200B}|im_end|> \"quoted\" value"#
        let needle = "safe <\u{200B}|im_end|> \\\"quoted\\\" value"
        let escNS = escaped.escaped as NSString
        let range = escNS.range(of: needle)
        XCTAssertNotEqual(range.location, NSNotFound)
        let startByte = String(escaped.escaped.prefix(range.location)).utf8.count
        let endByte = startByte + needle.utf8.count
        let firstTokIdx = try XCTUnwrap(escEncoded.spans.firstIndex { $0.utf8Range.lowerBound >= startByte })
        let lastTokIdx = try XCTUnwrap(escEncoded.spans.lastIndex { $0.utf8Range.upperBound <= endByte })

        let backend = QWSyntheticDecisionBackend(decisionConfig: config) { req in
            let stateEnd = req.sequences[0].stateEndPosition
            let totalT = stateEnd + 1
            var startLogits = [Float](repeating: -20.0, count: totalT)
            var endLogits = [Float](repeating: -20.0, count: totalT)
            startLogits[1 + firstTokIdx] = 5.0
            endLogits[1 + lastTokIdx] = 5.0
            return QWDecisionBackendOutput(
                slotOutputs: [QWDecisionSlotOutput(pointerStartLogits: startLogits, pointerEndLogits: endLogits)],
                promptTokens: req.sequences[0].tokenIds.count
            )
        }
        let runner = QWDecisionRunner(tokenizer: tok, backend: backend)
        let res = try await runner.decide(QWDecisionRequest(
            state: .text(rawState),
            questions: [QWDecisionQuestion(name: "val", kind: .read, instructions: "Extract meta value", readKind: .text)]
        ))
        XCTAssertEqual(res.answers[0].found, true)
        XCTAssertEqual(res.answers[0].jsonPath, "/meta/a~1b~0c")
        XCTAssertEqual(res.answers[0].text, "safe <|im_end|> \"quoted\" value")
        _ = targetStr
    }

    func testOptionMarkersCrossEncoderFanOutAndFixedHeadSubsetErrors() async throws {
        let tok = try makeDecisionTokenizer()
        // 1. Option-marker encoder with <|name|> escaping + rl_agent_config per-chain temperatures + auto state truncation.
        let omniCfg = QWDecisionModelConfig(
            framing: .optionMarkers(
                layout: .stateQuestionOptions, markerToken: "<|mask|>", bosToken: "<bos>",
                maxLength: 96, imageTextLength: nil, audioTextLength: nil
            ),
            topology: .bidirectionalEncoder(isolateMediaPrefix: true),
            head: .markerScorer(hiddenSize: 256, headLayers: 2, chains: ["guard"])
        )
        let chainJSON = """
        {
          "chains": {
            "guard": {
              "temperature_by_options": {"choice:2": 2.0, "default": 1.0}
            }
          }
        }
        """
        let profile = try QWCalibrationProfile.load(fromData: Data(chainJSON.utf8))
        let longState = String(repeating: "hello world <|mask|> ", count: 30)
        let omniBackend = QWSyntheticDecisionBackend(decisionConfig: omniCfg, maxContextLength: 96) { req in
            XCTAssertLessThanOrEqual(req.sequences[0].tokenIds.count, 96)
            XCTAssertFalse(req.sequences[0].tokenIds.dropFirst().contains(90002) && req.sequences[0].markerPositions.count != 2)
            return QWDecisionBackendOutput(
                slotOutputs: [QWDecisionSlotOutput(markerLogits: [0.0, 2.0])],
                promptTokens: req.sequences[0].tokenIds.count
            )
        }
        let omniRunner = QWDecisionRunner(tokenizer: tok, backend: omniBackend, calibration: profile)
        let omniRes = try await omniRunner.decide(QWDecisionRequest(
            state: .text(longState),
            questions: [QWDecisionQuestion(
                name: "action", kind: .choice, instructions: "Allow?",
                criteria: [QWDecisionCriterion(key: "allow"), QWDecisionCriterion(key: "deny")]
            )],
            chain: "guard"
        ))
        XCTAssertGreaterThan(omniRes.usage.truncatedTokens, 0)
        XCTAssertEqual(omniRes.answers[0].choice, "deny")
        // Temperature = 2.0 on [0, 2] -> softmax([0, 1]) = 1 / (1 + e^-1) ~ 0.7310585786.
        XCTAssertEqual(omniRes.answers[0].confidence, 1.0 / (1.0 + exp(-1.0)), accuracy: 1e-6)

        // 2. 1-logit cross-encoder reranker fan-out over K=3 candidates.
        let rerankCfg = QWDecisionModelConfig(
            framing: .sequencePair(maxLength: 512),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            head: .pooledClassifier(pooling: .firstToken, numLabels: 1, id2label: [:])
        )
        let rerankBackend = QWSyntheticDecisionBackend(decisionConfig: rerankCfg) { req in
            XCTAssertEqual(req.sequences.count, 3)
            return QWDecisionBackendOutput(
                slotOutputs: [
                    QWDecisionSlotOutput(pooledLogits: [0.1]),
                    QWDecisionSlotOutput(pooledLogits: [2.5]),
                    QWDecisionSlotOutput(pooledLogits: [-0.4]),
                ],
                promptTokens: 48
            )
        }
        let rerankRunner = QWDecisionRunner(tokenizer: tok, backend: rerankBackend)
        let rerankRes = try await rerankRunner.decide(QWDecisionRequest(
            state: .text("User asks for invoice PDF."),
            questions: [QWDecisionQuestion(
                name: "tool", kind: .choice, instructions: "Choose best tool",
                criteria: [
                    QWDecisionCriterion(key: "search"),
                    QWDecisionCriterion(key: "billing_pdf"),
                    QWDecisionCriterion(key: "logout")
                ]
            )]
        ))
        XCTAssertEqual(rerankRes.answers[0].choice, "billing_pdf")

        // 3. C >= 3 fixed id2label head: subset passes, unseen label throws unsupportedLabelSubset.
        let nliCfg = QWDecisionModelConfig(
            framing: .sequencePair(maxLength: 512),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            head: .pooledClassifier(pooling: .meanNonPad, numLabels: 3, id2label: [0: "entailment", 1: "neutral", 2: "contradiction"])
        )
        let nliBackend = QWSyntheticDecisionBackend(decisionConfig: nliCfg) { _ in
            QWDecisionBackendOutput(slotOutputs: [QWDecisionSlotOutput(pooledLogits: [0.2, 0.4, 2.0])], promptTokens: 20)
        }
        let nliRunner = QWDecisionRunner(tokenizer: tok, backend: nliBackend)
        do {
            _ = try await nliRunner.decide(QWDecisionRequest(
                state: .text("The sky is green."),
                questions: [QWDecisionQuestion(
                    name: "bad_labels", kind: .choice, instructions: "Classify",
                    criteria: [QWDecisionCriterion(key: "entailment"), QWDecisionCriterion(key: "banana")]
                )]
            ))
            XCTFail("expected unsupportedLabelSubset")
        } catch let err as QWDecisionError {
            guard case .unsupportedLabelSubset = err else { return XCTFail("wrong error: \(err)") }
        }
    }

    func testCalibrationFitterRenormEstimateAndHaltonBoxes() async throws {
        // 1. Calibration roundPy renormalization & tie-break invariant.
        let tricky = [0.3211000000000001, 0.3211000000000001, 0.3578]
        let renorm = try QWCalibrationEngine.renormalize(questionName: "q1", keys: ["a", "b", "c"], rawProbs: tricky)
        let sum = renorm.map(\.value).reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-12)
        let ties = try QWCalibrationEngine.renormalize(questionName: "tie", keys: ["first", "second"], rawProbs: [0.5, 0.5])
        XCTAssertEqual(ties[QWCalibrationEngine.argmaxIndex(ties.map(\.value))].key, "first")

        // 2. NLL temperature fitting & PAV isotonic monotonicity.
        var overconfident: [QWCalibrationSample] = []
        for i in 0..<40 {
            let correct = (i % 5 != 0)
            let logits = correct ? [4.0, 0.0] : [2.5, 0.0]
            overconfident.append(QWCalibrationSample(logits: logits, labelIndex: 0, bucket: "choice:2"))
        }
        let pavObs: [(raw: Double, target: Double)] = [
            (0.1, 0.0), (0.2, 0.5), (0.3, 0.1), (0.6, 0.7), (0.8, 0.95)
        ]
        let (fitted, metrics) = QWCalibrationFitter.fitProfile(samples: overconfident, binaryTables: ["box": pavObs])
        XCTAssertLessThanOrEqual(metrics.nllAfter, metrics.nllBefore + 1e-9)
        let knots = try XCTUnwrap(fitted.tables["box"])
        for i in 1..<knots.count {
            XCTAssertLessThanOrEqual(knots[i - 1][1], knots[i][1] + 1e-12)
        }

        // 3. Estimate + Halton 512 Boxes with THOUSANDTHS grid & interp edges.
        let tok = try makeDecisionTokenizer()
        let boxCfg = QWDecisionModelConfig(
            framing: .turnVerbalizer(codesPerOption: 2, numbers: true, packSet: false, maxSetSize: 1, colonSlots: true, stateSeparator: "\n\n"),
            topology: .causal(supportsPrefixCache: true),
            head: .vocabVerbalizer(vocabSize: 12000, tieEmbeddings: true, sparseGatherThreshold: 256),
            usesNumbersThousandthsGrid: true,
            boxEdgesMode: "interp"
        )
        let idNo = try XCTUnwrap(tok.singleTokenId(forText: "no"))
        let idYes = try XCTUnwrap(tok.singleTokenId(forText: "yes"))
        let backend = QWSyntheticDecisionBackend(decisionConfig: boxCfg) { req in
            var outputs: [QWDecisionSlotOutput] = []
            var boxEdgeIdx = 0
            for s in req.slots {
                if s.candidateTokenGroups.count == 2 {
                    outputs.append(QWDecisionSlotOutput(tokenLogits: [idNo: -5.0, idYes: 5.0]))
                } else if s.candidateTokenGroups.count == 1000 {
                    // Peak at index 200 (0.2), 600 (0.6), 100 (0.1), 500 (0.5).
                    var map: [Int: Float] = [:]
                    let peak = [200, 600, 100, 500][boxEdgeIdx % 4]
                    boxEdgeIdx += 1
                    for (idx, grp) in s.candidateTokenGroups.enumerated() {
                        map[grp[0]] = idx == peak ? 12.0 : -12.0
                    }
                    outputs.append(QWDecisionSlotOutput(tokenLogits: map))
                } else {
                    var map: [Int: Float] = [:]
                    for (idx, grp) in s.candidateTokenGroups.enumerated() {
                        map[grp[0]] = idx == 5 ? 10.0 : -10.0
                    }
                    outputs.append(QWDecisionSlotOutput(tokenLogits: map))
                }
            }
            return QWDecisionBackendOutput(slotOutputs: outputs, promptTokens: 64)
        }
        let runner = QWDecisionRunner(tokenizer: tok, backend: backend)
        let res = try await runner.decide(QWDecisionRequest(
            state: .text("Estimate count and localize object."),
            questions: [
                QWDecisionQuestion(name: "count", kind: .estimate, instructions: "How many?", estimateLow: 0, estimateHigh: 10, estimateStep: 1),
                QWDecisionQuestion(name: "obj", kind: .box, instructions: "Find object"),
            ]
        ))
        XCTAssertEqual(try XCTUnwrap(res.answers[0].estimate), 5.0, accuracy: 0.2)
        XCTAssertEqual(res.answers[1].found, true)
        XCTAssertEqual(try XCTUnwrap(res.answers[1].boxes.first?.top), 0.2, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(res.answers[1].boxes.first?.bottom), 0.6, accuracy: 0.01)
        XCTAssertEqual(res.answers[1].confidence, 1.0, accuracy: 1e-6)

        // Verify .error truncation throws contextOverflow cleanly.
        let tinyBackend = QWSyntheticDecisionBackend(decisionConfig: boxCfg, maxContextLength: 8) { _ in
            QWDecisionBackendOutput(slotOutputs: [], promptTokens: 0)
        }
        let tinyRunner = QWDecisionRunner(tokenizer: tok, backend: tinyBackend)
        do {
            _ = try await tinyRunner.decide(QWDecisionRequest(
                state: .text("one two three four five six seven eight nine ten"),
                questions: [QWDecisionQuestion(name: "q", kind: .noul, instructions: "Urgent?")]
            ))
            XCTFail("expected contextOverflow")
        } catch let err as QWDecisionError {
            guard case .contextOverflow = err else { return XCTFail("wrong error: \(err)") }
        }
    }

    private func makeChatMLNumberTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        for w in ["yes", "Yes", "YES", "no", "No", "NO", "ye", "Ye", "YE", "ĠA", "ĠB", "0", "1", "2", "10"] where vocab[w] == nil {
            vocab[w] = nextId; nextId += 1
        }
        let merges: [[String]] = [["n", "o"], ["N", "o"], ["N", "O"], ["y", "e"], ["ye", "s"], ["Y", "e"], ["Ye", "s"], ["Y", "E"], ["YE", "S"], ["Ġ", "A"], ["Ġ", "B"], ["1", "0"]]
        var addedTokens: [[String: Any]] = [
            ["id": 124894, "content": "<|startoftext|>", "special": true, "normalized": false],
            ["id": 124899, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": 124900, "content": "<|im_end|>", "special": true, "normalized": false],
            ["id": 124901, "content": "<think>", "special": true, "normalized": false],
            ["id": 124902, "content": "</think>", "special": true, "normalized": false],
            ["id": 125009, "content": "<|vision_start|>", "special": true, "normalized": false],
            ["id": 125010, "content": "<|vision_end|>", "special": true, "normalized": false],
            ["id": 125030, "content": "<|image|>", "special": true, "normalized": false],
        ]
        for i in 0..<1000 where i > 9 {
            addedTokens.append(["id": 91000 + i, "content": "\(i)", "special": false, "normalized": false])
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false]
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre],
            "normalizer": ["type": "NFC"]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    func testChatMLD1FramingPointerMaskAndTypedCalibration() async throws {
        let tok = try makeChatMLNumberTokenizer()
        let idNo = try XCTUnwrap(tok.singleTokenId(forText: "no"))
        let idYes = try XCTUnwrap(tok.singleTokenId(forText: "yes"))
        let cfg = QWDecisionModelConfig(
            framing: .turnVerbalizer(
                codesPerOption: 2, numbers: true, packSet: true,
                maxSetSize: 16, colonSlots: true, stateSeparator: "\n\n",
                assistantLead: "<think></think>", chatML: true
            ),
            topology: .packedSet(maxSetSize: 16),
            head: .vocabVerbalizer(vocabSize: 128000, tieEmbeddings: true, sparseGatherThreshold: 256),
            hasStatePointer: true,
            hasGridReadout: true,
            usesNumbersThousandthsGrid: true,
            boxEdgesMode: "median",
            defaultScoreReadout: "median"
        )
        let cal = QWCalibrationProfile(defaultTemperature: 2.0)
        let rawState = #"{"owner":"Alice Smith","amount":42}"#
        let built = try QWSequenceFramingEngine.buildBatch(
            request: QWDecisionRequest(
                state: .text(rawState),
                questions: [
                    QWDecisionQuestion(name: "flag", kind: .noul, instructions: "Approve?"),
                    QWDecisionQuestion(
                        name: "tags", kind: .multiChoice, instructions: "Tags?",
                        criteria: [QWDecisionCriterion(key: "vip"), QWDecisionCriterion(key: "urgent")]
                    ),
                    QWDecisionQuestion(name: "owner", kind: .read, instructions: "Who owns it?", readKind: .text),
                    QWDecisionQuestion(name: "count", kind: .estimate, instructions: "How many?", estimateLow: 0, estimateHigh: 10, estimateStep: 1),
                    QWDecisionQuestion(name: "objs", kind: .boxes, instructions: "Objects", maxBoxes: 2),
                ]
            ),
            config: cfg, tokenizer: tok, maxContextLength: 2048
        )
        XCTAssertEqual(built.request.sequences.count, 1)
        let seq = built.request.sequences[0]
        let rendered = try tok.decode(seq.tokenIds, skipSpecialTokens: false)
        XCTAssertTrue(rendered.hasPrefix("<|startoftext|><|im_start|>user\n{\"owner\":\"Alice Smith\",\"amount\":42}\n\n\nQUESTION:\nApprove?"))
        XCTAssertTrue(rendered.contains("<|im_end|>\n<|im_start|>assistant\n<think></think>"))
        XCTAssertTrue(rendered.contains("QUESTION:\nWho owns it?\n\nRead the text from the state, or nothing if it is absent.<|im_end|>\n<|im_start|>assistant\n<think></think>start:\nend:"))
        XCTAssertTrue(rendered.contains("\"vip\": vip\n\"urgent\": urgent"))
        XCTAssertTrue(rendered.contains("Estimate from 0 to 10, in steps of 1. Reply with the number only."))
        XCTAssertTrue(rendered.contains("Every one of them, 16 at most, in reading order"))
        // Token 0 (BOS) is valid, user/im_start template tokens are invalid, state content tokens are valid.
        XCTAssertEqual(seq.pointerValidMask.first, true)
        XCTAssertEqual(seq.pointerValidMask[1], false)
        XCTAssertTrue(seq.pointerValidMask.dropFirst().contains(true))

        // Locate "Alice Smith" tokens via pointerTokenSpans.
        let readBinding = built.questionBindings[2]
        let escapedBytes = Array(readBinding.escapedState.original.utf8)
        let aliceRange = try XCTUnwrap(readBinding.escapedState.original.range(of: "Alice Smith"))
        let aliceStartByte = readBinding.escapedState.original[..<aliceRange.lowerBound].utf8.count
        let aliceEndByte = aliceStartByte + "Alice Smith".utf8.count
        let ptrS = try XCTUnwrap(readBinding.pointerTokenSpans.enumerated().first { idx, sp in
            idx > 0 && sp != nil && sp!.utf8Range.lowerBound >= aliceStartByte
        }?.offset)
        let ptrE = try XCTUnwrap(readBinding.pointerTokenSpans.enumerated().last { idx, sp in
            idx > 0 && sp != nil && sp!.utf8Range.upperBound <= aliceEndByte
        }?.offset)
        _ = escapedBytes

        // Verify Read pointer slot tracks both start: and end: positions and masks invalid template token logits.
        let readSlot = try XCTUnwrap(built.request.slots.first(where: { $0.isPointerSlot }))
        XCTAssertNotNil(readSlot.pointerEndPosition)
        XCTAssertNotEqual(readSlot.tokenPosition, readSlot.pointerEndPosition)

        let backend = QWSyntheticDecisionBackend(decisionConfig: cfg) { req in
            var outs: [QWDecisionSlotOutput] = []
            var boxEdgeCursor = 0
            for s in req.slots {
                if s.isPointerSlot {
                    let t = req.sequences[0].pointerValidMask.count
                    var sLogits = [Float](repeating: -20.0, count: t)
                    var eLogits = [Float](repeating: -20.0, count: t)
                    // Template token 1 has +100.0 which MUST be masked out by pointerValidMask[1] == false!
                    sLogits[1] = 100.0
                    eLogits[1] = 100.0
                    sLogits[0] = -6.0
                    sLogits[ptrS] = 6.0
                    eLogits[ptrE] = 6.0
                    outs.append(QWDecisionSlotOutput(pointerStartLogits: sLogits, pointerEndLogits: eLogits))
                } else if s.candidateTokenGroups.count == 2 {
                    outs.append(QWDecisionSlotOutput(tokenLogits: [idNo: 0.0, idYes: 2.0]))
                } else if s.candidateTokenGroups.count == 11 {
                    var map: [Int: Float] = [:]
                    for (i, grp) in s.candidateTokenGroups.enumerated() { map[grp[0]] = (i == 4) ? 8.0 : -8.0 }
                    outs.append(QWDecisionSlotOutput(tokenLogits: map))
                } else {
                    let peak = [200, 500, 100, 400][boxEdgeCursor % 4]
                    boxEdgeCursor += 1
                    var map: [Int: Float] = [:]
                    for (i, grp) in s.candidateTokenGroups.enumerated() { map[grp[0]] = (i == peak) ? 8.0 : -8.0 }
                    outs.append(QWDecisionSlotOutput(tokenLogits: map))
                }
            }
            return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: req.sequences[0].tokenIds.count)
        }
        let ans = try await QWDecisionRunner(tokenizer: tok, backend: backend, calibration: cal).decide(
            QWDecisionRequest(state: .text(rawState), questions: built.questionBindings.map(\.question))
        )
        // noul IS tempered (T=2.0 -> sigmoid(1.0) ~ 0.731059) while multiChoice is untempered (T=1.0 -> sigmoid(2.0) ~ 0.880797).
        XCTAssertEqual(ans.answers[0].confidence, 1.0 / (1.0 + exp(-1.0)), accuracy: 1e-6)
        XCTAssertEqual(ans.answers[1].probs[0].value, 1.0 / (1.0 + exp(-2.0)), accuracy: 1e-6)
        XCTAssertEqual(ans.answers[2].text, "Alice Smith")
        XCTAssertEqual(ans.answers[2].jsonPath, "/owner")
        XCTAssertEqual(try XCTUnwrap(ans.answers[3].estimate), 4.0, accuracy: 0.2)
        XCTAssertEqual(ans.answers[4].presence.count, 2)
    }
}
