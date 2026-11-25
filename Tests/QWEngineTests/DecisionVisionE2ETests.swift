// DecisionVisionE2ETests.swift — Descriptor decoding and box/boxes Halton IoU qualification tests.

import Foundation
import XCTest
@testable import QWCore
@testable import QWDecision
@testable import QWToken
@testable import QWVision

final class DecisionVisionE2ETests: XCTestCase {
    private final class MockVisionDecisionBackend: QWDecisionExecutionBackend, @unchecked Sendable {
        let decisionConfig: QWDecisionModelConfig
        let maxContextLength: Int
        private let lock = NSLock()
        private var _capturedImages: [[Data]] = []
        private let handler: @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput

        var capturedImages: [[Data]] {
            lock.withLock { _capturedImages }
        }

        init(
            decisionConfig: QWDecisionModelConfig,
            maxContextLength: Int = 4096,
            handler: @escaping @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput
        ) {
            self.decisionConfig = decisionConfig
            self.maxContextLength = maxContextLength
            self.handler = handler
        }

        func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
            lock.withLock { _capturedImages = request.sequences.map(\.images) }
            return try handler(request)
        }
    }

    private func makeTestTokenizer() throws -> QWTokenizer {
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
        for w in ["yes", "no", "ye", "Yes", "No", "YES", "NO", "true", "false", "BOX:"] where vocab[w] == nil {
            vocab[w] = nextId
            nextId += 1
        }
        let merges: [[String]] = [["n", "o"], ["y", "e"], ["ye", "s"]]
        var addedTokens: [[String: Any]] = [
            ["id": 90000, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": 90001, "content": "<|im_end|>", "special": true, "normalized": false],
            ["id": 90003, "content": "<bos>", "special": true, "normalized": false],
        ]
        for i in 0..<128 {
            addedTokens.append(["id": 90100 + i, "content": "c\(i)", "special": false, "normalized": false])
        }
        for i in 0..<1000 {
            addedTokens.append(["id": 91000 + i, "content": String(format: "%03d", i), "special": false, "normalized": false])
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
            ]],
            "normalizer": ["type": "NFC"],
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func computeIoU(_ a: (top: Double, bottom: Double, left: Double, right: Double),
                            _ b: (top: Double, bottom: Double, left: Double, right: Double)) -> Double {
        let iy1 = max(a.top, b.top)
        let ix1 = max(a.left, b.left)
        let iy2 = min(a.bottom, b.bottom)
        let ix2 = min(a.right, b.right)
        let inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
        let areaA = max(0.0, a.bottom - a.top) * max(0.0, a.right - a.left)
        let areaB = max(0.0, b.bottom - b.top) * max(0.0, b.right - b.left)
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0.0
    }

    func testVisionConfigDecodingForLFM3VLAndSigLIP2() throws {
        let lfmJSON = """
        {
          "model_type": "lfm3_vl",
          "image_token_id": 125030,
          "text_config": {
            "model_type": "lfm3",
            "eos_token_id": 2,
            "hidden_size": 128,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 4,
            "max_position_embeddings": 2048,
            "norm_eps": 1e-5,
            "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"}}
          },
          "vision_config": {
            "model_type": "lfm3_vl_vision",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "patch_size": 4,
            "downsample_factor": 2,
            "out_dim": 128,
            "conv_kernel_size": 3,
            "layer_types": ["conv", "full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
            "norm_eps": 1e-6
          }
        }
        """
        let lfmDesc = try QWModelDescriptor.decode(configJSON: Data(lfmJSON.utf8))
        XCTAssertEqual(lfmDesc.visionTowerConfig?.patchEmbedKind, .lfmHybridConv)
        XCTAssertEqual(lfmDesc.visionTowerConfig?.kvHeads, 2)
        XCTAssertEqual(lfmDesc.visionTowerConfig?.layerTypes, ["conv", "full_attention"])
        XCTAssertEqual(lfmDesc.visionTowerConfig?.convKernelSize, 3)
        let lfmVisCfg = QWVisionConfig(descriptor: lfmDesc)
        XCTAssertEqual(lfmVisCfg.architecture, .lfm3VL)
        XCTAssertEqual(lfmVisCfg.numPosEmbeds, 0)
        XCTAssertEqual(lfmVisCfg.kvHeads, 2)
        XCTAssertEqual(lfmVisCfg.merge, 2)
        XCTAssertEqual(lfmVisCfg.outDim, 128)

        let siglipJSON = """
        {
          "model_type": "lfm2_vl",
          "image_token_id": 125030,
          "text_config": {
            "model_type": "lfm2",
            "eos_token_id": 2,
            "hidden_size": 128,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 4,
            "max_position_embeddings": 2048,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0
          },
          "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "patch_size": 8,
            "num_position_embeddings": 64,
            "spatial_merge_size": 2,
            "out_dim": 128,
            "layer_norm_eps": 1e-6
          }
        }
        """
        let sigDesc = try QWModelDescriptor.decode(configJSON: Data(siglipJSON.utf8))
        XCTAssertEqual(sigDesc.visionTowerConfig?.patchEmbedKind, .siglip2NaFlex)
        let sigVisCfg = QWVisionConfig(descriptor: sigDesc)
        XCTAssertEqual(sigVisCfg.architecture, .siglip2)
        XCTAssertEqual(sigVisCfg.numPosEmbeds, 64)
        XCTAssertEqual(sigVisCfg.posSide, 8)
        XCTAssertEqual(sigVisCfg.merge, 2)
        XCTAssertEqual(sigVisCfg.outDim, 128)
    }

    func testBoxAndBoxesHaltonIoUQualification() async throws {
        let tok = try makeTestTokenizer()
        let idNo = try XCTUnwrap(tok.singleTokenId(forText: "no"))
        let idYes = try XCTUnwrap(tok.singleTokenId(forText: "yes"))
        let sampleImage = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02])

        // 1. Test both "median" and "interp" modes across 128-bin and 1000-bin thousandths grids.
        let modes: [(edgesMode: String, thousandths: Bool, target: (Double, Double, Double, Double))] = [
            ("median", false, (0.2, 0.6, 0.1, 0.5)),
            ("interp", false, (0.2, 0.6, 0.1, 0.5)),
            ("median", true, (0.2, 0.6, 0.1, 0.5)),
            ("interp", true, (0.2, 0.6, 0.1, 0.5)),
        ]

        for m in modes {
            let cfg = QWDecisionModelConfig(
                framing: .turnVerbalizer(
                    codesPerOption: 2, numbers: m.thousandths, packSet: false,
                    maxSetSize: 1, colonSlots: true, stateSeparator: "\n\n"
                ),
                topology: .causal(supportsPrefixCache: true),
                head: .vocabVerbalizer(vocabSize: 95_000, tieEmbeddings: true, sparseGatherThreshold: 256),
                hasGridReadout: true,
                usesNumbersThousandthsGrid: m.thousandths,
                boxEdgesMode: m.edgesMode
            )
            let backend = MockVisionDecisionBackend(decisionConfig: cfg) { req in
                var outs: [QWDecisionSlotOutput] = []
                let targets = [m.target.0, m.target.1, m.target.2, m.target.3]
                var edgeCursor = 0
                for s in req.slots {
                    if s.candidateTokenGroups.count == 2 {
                        outs.append(QWDecisionSlotOutput(tokenLogits: [idNo: -6.0, idYes: 6.0]))
                    } else {
                        let gridSize = s.candidateTokenGroups.count
                        let high = m.thousandths ? 0.999 : 1.0
                        let peakIdx = min(gridSize - 1, max(0, Int((targets[edgeCursor % 4] / high * Double(gridSize - 1)).rounded())))
                        edgeCursor += 1
                        var logits: [Int: Float] = [:]
                        for (idx, grp) in s.candidateTokenGroups.enumerated() {
                            logits[grp[0]] = idx == peakIdx ? 12.0 : -12.0
                        }
                        outs.append(QWDecisionSlotOutput(tokenLogits: logits))
                    }
                }
                return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: 48)
            }
            let runner = QWDecisionRunner(tokenizer: tok, backend: backend)
            let resp = try await runner.decide(QWDecisionRequest(
                state: .empty,
                questions: [QWDecisionQuestion(name: "det", kind: .box, instructions: "Locate object")],
                images: [sampleImage]
            ))
            XCTAssertEqual(backend.capturedImages.first, [sampleImage], "images must propagate to QWDecisionSequenceInput")
            let ans = try XCTUnwrap(resp.answers.first)
            XCTAssertEqual(ans.found, true)
            let box = try XCTUnwrap(ans.boxes.first)
            let iou = computeIoU((box.top, box.bottom, box.left, box.right), m.target)
            XCTAssertGreaterThanOrEqual(iou, 0.95, "mode=\(m.edgesMode) thousandths=\(m.thousandths) IoU=\(iou)")
            XCTAssertGreaterThanOrEqual(box.confidence, 0.85)
        }

        // 2. Verify found == false when presence probability is below threshold.
        let boxCfg = QWDecisionModelConfig(
            framing: .turnVerbalizer(
                codesPerOption: 2, numbers: true, packSet: false,
                maxSetSize: 1, colonSlots: true, stateSeparator: "\n\n"
            ),
            topology: .causal(supportsPrefixCache: true),
            head: .vocabVerbalizer(vocabSize: 95_000, tieEmbeddings: true, sparseGatherThreshold: 256),
            hasGridReadout: true,
            usesNumbersThousandthsGrid: true,
            boxEdgesMode: "median"
        )
        let missingBackend = MockVisionDecisionBackend(decisionConfig: boxCfg) { req in
            let outs = req.slots.map { s in
                if s.candidateTokenGroups.count == 2 {
                    return QWDecisionSlotOutput(tokenLogits: [idNo: 6.0, idYes: -6.0])
                }
                return QWDecisionSlotOutput(tokenLogits: [s.candidateTokenGroups[0][0]: 5.0])
            }
            return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: 32)
        }
        let missingResp = try await QWDecisionRunner(tokenizer: tok, backend: missingBackend).decide(
            QWDecisionRequest(
                state: .empty,
                questions: [QWDecisionQuestion(name: "absent", kind: .box, instructions: "Locate missing")],
                images: [sampleImage]
            )
        )
        let missingAns = try XCTUnwrap(missingResp.answers.first)
        XCTAssertEqual(missingAns.found, false)
        XCTAssertTrue(missingAns.boxes.isEmpty)

        // 3. Multi-box (`boxes`): verify monotone presence clamping and deterministic (left, top) sorting.
        let multiBackend = MockVisionDecisionBackend(decisionConfig: boxCfg) { req in
            var outs: [QWDecisionSlotOutput] = []
            // Slot 0: prob ~0.75 (logit delta ~ +1.0986), right-side box [0.2, 0.5, 0.6, 0.9]
            // Slot 1: raw prob ~0.95 (logit delta ~ +2.9444), left-side box [0.3, 0.7, 0.1, 0.4]
            // Slot 2+: raw prob < 0.5 (-4.0)
            let boxCoords = [
                [200, 500, 600, 900],
                [300, 700, 100, 400],
            ]
            var presIdx = 0
            var edgeCount = 0
            for s in req.slots {
                if s.candidateTokenGroups.count == 2 {
                    let yesLogit: Float = switch presIdx {
                    case 0: 1.0986123
                    case 1: 2.944439
                    default: -4.0
                    }
                    presIdx += 1
                    outs.append(QWDecisionSlotOutput(tokenLogits: [idNo: 0.0, idYes: yesLogit]))
                } else {
                    let bIdx = edgeCount / 4
                    let eIdx = edgeCount % 4
                    edgeCount += 1
                    let peak = (bIdx < boxCoords.count) ? boxCoords[bIdx][eIdx] : 100
                    var logits: [Int: Float] = [:]
                    for (idx, grp) in s.candidateTokenGroups.enumerated() {
                        logits[grp[0]] = idx == peak ? 12.0 : -12.0
                    }
                    outs.append(QWDecisionSlotOutput(tokenLogits: logits))
                }
            }
            return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: 96)
        }
        let multiResp = try await QWDecisionRunner(tokenizer: tok, backend: multiBackend).decide(
            QWDecisionRequest(
                state: .empty,
                questions: [QWDecisionQuestion(name: "objs", kind: .boxes, instructions: "Locate objects", maxBoxes: 4)],
                images: [sampleImage]
            )
        )
        let multiAns = try XCTUnwrap(multiResp.answers.first)
        XCTAssertEqual(multiAns.boxes.count, 2)
        XCTAssertLessThan(multiAns.boxes[0].left, multiAns.boxes[1].left, "boxes must sort deterministically by (left, top)")
        // Slot 1 (sorted to index 0 since left=0.1 < 0.6) had raw p=0.95 but must be clamped to slot 0's p=0.75.
        XCTAssertLessThanOrEqual(multiAns.boxes[0].found, multiAns.boxes[1].found + 1e-6)
        XCTAssertEqual(multiAns.boxes[0].found, 0.75, accuracy: 0.02)
    }
}
