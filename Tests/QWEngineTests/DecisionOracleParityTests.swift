// DecisionOracleParityTests.swift — PyTorch-to-Metal parity for all 3 M3 decision archetypes.

import Foundation
import Metal
import XCTest
@testable import QWAlloc
@testable import QWAttn
@testable import QWCore
@testable import QWDecision
@testable import QWEngine
@testable import QWGDN
@testable import QWHAL
@testable import QWHead
@testable import QWNorm
@testable import QWRope

final class DecisionOracleParityTests: XCTestCase {
    private var oracleDir: URL!
    private var manifest: [String: Any]!

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Metal decision parity oracle tests require Apple Silicon GPU"
        )
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        oracleDir = repoRoot.appendingPathComponent("out/decision_oracle")
        let manifestURL = oracleDir.appendingPathComponent("manifest.json")
        if !FileManager.default.fileExists(atPath: manifestURL.path) {
            throw XCTSkip("Run Scripts/oracle/decision_oracle.py --out out/decision_oracle first")
        }
        let data = try Data(contentsOf: manifestURL)
        guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Invalid decision_oracle manifest JSON")
            return
        }
        manifest = parsed
    }

    func testArchetype1PackedSetLFM3ParityAndIsolation() throws {
        let m = try requireDict(manifest["archetype1_packed_set_lfm3"], "archetype1")
        let hal = try QWHal(geometry: QWTestGeometry.g27, weightBasis: .standard)
        let pipeline = try QWDecisionMetalPipeline(hal: hal)
        let b = QWDecisionTensorBuilder(hal: hal)

        let h = try requireInt(m["hidden"], "hidden")
        let ffn = try requireInt(m["ffnDim"], "ffnDim")
        let seqLen = try requireInt(m["seqLen"], "seqLen")
        let stateEnd = try requireInt(m["stateEndPos"], "stateEndPos")
        let groupIds = (try requireIntArray(m["groupIds"], "groupIds")).map { Int32($0) }
        let layerDicts = try requireDictArray(m["layers"], "layers")
        let layers = try layerDicts.map { try buildLayer($0, hidden: h, ffnDim: ffn, b: b) }

        let fnW = try loadF16Tensor(try requireString(m["finalNormW"], "fnW"), shape: [h], b: b)
        let finalNorm = (QWDecisionNormKind.rmsNorm(eps: 1e-6, unitOffset: false), QWDecisionNormWeights(weight: fnW))
        let hiddenIn = try loadF16Tensor(try requireString(m["hiddenIn"], "hiddenIn"), shape: [seqLen, h], b: b)
        let outTensor = try pipeline.runBackbone(
            hiddenIn: hiddenIn, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: stateEnd, mediaPrefixLen: 0
        )
        let metalFinalH = pipeline.readF16AsFloats(outTensor, count: seqLen * h)
        let refFinalH = try loadF32(try requireString(m["refFinalH"], "refFinalH"))
        XCTAssertLessThanOrEqual(maxAbs(metalFinalH, refFinalH), 4e-3) // <= 2 FP16 ULPs at |x| ~ 4.0 under 13x sandwich post-norm gain

        // Sparse BF16 verbalizer parity (exact <= 1e-4 vs full logits on matching hidden input, <= 2e-3 E2E).
        let vocab = try requireInt(m["vocabSize"], "vocabSize")
        let candIds = (try requireIntArray(m["candidateIds"], "candidateIds")).map { Int32($0) }
        let slotRows = try requireIntArray(m["slotRows"], "slotRows")
        let lmHeadBF16 = try loadBF16Tensor(try requireString(m["lmHeadBF16"], "lmHeadBF16"), shape: [vocab, h], b: b)

        var refSlotFloats: [Float] = []
        var e2eSlotFloats: [Float] = []
        for row in slotRows {
            let start = row * h
            refSlotFloats.append(contentsOf: refFinalH[start..<(start + h)])
            e2eSlotFloats.append(contentsOf: metalFinalH[start..<(start + h)])
        }
        let refSlotT = try b.makeF16FromFloats(refSlotFloats, shape: [slotRows.count, h])
        let e2eSlotT = try b.makeF16FromFloats(e2eSlotFloats, shape: [slotRows.count, h])
        let sparseOnRef = try pipeline.evaluateSparseVerbalizerBF16(
            hiddenStates: refSlotT, lmHeadBF16: lmHeadBF16,
            rows: slotRows.count, hidden: h, vocabSize: vocab, candidateIds: candIds
        )
        let sparseE2E = try pipeline.evaluateSparseVerbalizerBF16(
            hiddenStates: e2eSlotT, lmHeadBF16: lmHeadBF16,
            rows: slotRows.count, hidden: h, vocabSize: vocab, candidateIds: candIds
        )
        let refFullLogits = try loadF32(try requireString(m["refFullLogits"], "refFullLogits"))
        let refSparseLogits = try loadF32(try requireString(m["refSparseLogits"], "refSparseLogits"))
        var extractedFull: [Float] = []
        for sIdx in 0..<slotRows.count {
            for cid in candIds {
                extractedFull.append(refFullLogits[sIdx * vocab + Int(cid)])
            }
        }
        XCTAssertLessThanOrEqual(maxAbs(sparseOnRef, extractedFull), 1e-4)
        XCTAssertLessThanOrEqual(maxAbs(sparseE2E, refSparseLogits), 2e-3)
        for sIdx in 0..<slotRows.count {
            let off = sIdx * candIds.count
            XCTAssertEqual(
                argmax(Array(sparseE2E[off..<(off + candIds.count)])),
                argmax(Array(refSparseLogits[off..<(off + candIds.count)]))
            )
        }

        // State pointer start/end scoring parity.
        let stateH = try b.makeF16FromFloats(Array(metalFinalH[0..<(stateEnd * h)]), shape: [stateEnd, h])
        let qStart = try loadF16Tensor(try requireString(m["qStart"], "qStart"), shape: [h], b: b)
        let qEnd = try loadF16Tensor(try requireString(m["qEnd"], "qEnd"), shape: [h], b: b)
        let ptrOut = try pipeline.evaluateStatePointer(stateHidden: stateH, qStart: qStart, qEnd: qEnd, stateLen: stateEnd, hidden: h)
        let refPtrStart = try loadF32(try requireString(m["refPtrStart"], "refPtrStart"))
        let refPtrEnd = try loadF32(try requireString(m["refPtrEnd"], "refPtrEnd"))
        XCTAssertLessThanOrEqual(maxAbs(ptrOut.start, refPtrStart), 2e-3)
        XCTAssertLessThanOrEqual(maxAbs(ptrOut.end, refPtrEnd), 2e-3)
        XCTAssertEqual(argmax(ptrOut.start), argmax(refPtrStart))
        XCTAssertEqual(argmax(ptrOut.end), argmax(refPtrEnd))

        // PackedSet causal+question isolation: perturbing Q_B gives 0.0 drift on Q_A (rows 4..<7), while state (0..<4) sees Q_B.
        let hiddenQBPert = try loadF16Tensor(try requireString(m["hiddenQBPert"], "hiddenQBPert"), shape: [seqLen, h], b: b)
        let outQBPert = try pipeline.runBackbone(
            hiddenIn: hiddenQBPert, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: stateEnd, mediaPrefixLen: 0
        )
        let metalQBPert = pipeline.readF16AsFloats(outQBPert, count: seqLen * h)
        XCTAssertGreaterThan(maxAbs(Array(metalQBPert[0..<(stateEnd * h)]), Array(metalFinalH[0..<(stateEnd * h)])), 0.0)
        XCTAssertEqual(maxAbs(Array(metalQBPert[(stateEnd * h)..<(7 * h)]), Array(metalFinalH[(stateEnd * h)..<(7 * h)])), 0.0)
        let refPertQBRow6 = try loadF32(try requireString(m["refPertQB_Row6"], "refPertQB_Row6"))
        XCTAssertLessThanOrEqual(maxAbs(Array(metalQBPert[(6 * h)..<(7 * h)]), refPertQBRow6), 2e-3)

        // Reverse sibling isolation on .packedSet attention layer: perturbing Q_A gives 0.0 drift on Q_B (rows 7..<10).
        let attnBaseT = try pipeline.runBackbone(
            hiddenIn: hiddenIn, seqLen: seqLen, hidden: h, layers: [layers[1]],
            finalNorm: nil, groupIds: groupIds, stateEndPos: stateEnd, mediaPrefixLen: 0
        )
        let hiddenQAPert = try loadF16Tensor(try requireString(m["hiddenQAPert"], "hiddenQAPert"), shape: [seqLen, h], b: b)
        let attnPertQAT = try pipeline.runBackbone(
            hiddenIn: hiddenQAPert, seqLen: seqLen, hidden: h, layers: [layers[1]],
            finalNorm: nil, groupIds: groupIds, stateEndPos: stateEnd, mediaPrefixLen: 0
        )
        let attnBase = pipeline.readF16AsFloats(attnBaseT, count: seqLen * h)
        let attnPertQA = pipeline.readF16AsFloats(attnPertQAT, count: seqLen * h)
        let refAttnOnly = try loadF32(try requireString(m["refAttnOnly"], "refAttnOnly"))
        XCTAssertLessThanOrEqual(maxAbs(attnBase, refAttnOnly), 4e-3) // <= 2 FP16 ULPs at |x| ~ 4.0 under 13x sandwich post-norm gain
        XCTAssertEqual(maxAbs(Array(attnPertQA[(7 * h)..<(10 * h)]), Array(attnBase[(7 * h)..<(10 * h)])), 0.0)
        let refAttnPertQARow9 = try loadF32(try requireString(m["refAttnPertQA_Row9"], "refAttnPertQA_Row9"))
        XCTAssertLessThanOrEqual(maxAbs(Array(attnPertQA[(9 * h)..<(10 * h)]), refAttnPertQARow9), 2e-3)
    }

    func testArchetype2BidirMarkerOmniParityAndMediaIsolation() throws {
        let m = try requireDict(manifest["archetype2_bidir_marker_omni"], "archetype2")
        let hal = try QWHal(geometry: QWTestGeometry.g27, weightBasis: .standard)
        let pipeline = try QWDecisionMetalPipeline(hal: hal)
        let b = QWDecisionTensorBuilder(hal: hal)

        let h = try requireInt(m["hidden"], "hidden")
        let ffn = try requireInt(m["ffnDim"], "ffnDim")
        let seqLen = try requireInt(m["seqLen"], "seqLen")
        let mediaLen = try requireInt(m["mediaPrefixLen"], "mediaPrefixLen")
        let groupIds = (try requireIntArray(m["groupIds"], "groupIds")).map { Int32($0) }
        let layerDicts = try requireDictArray(m["layers"], "layers")
        let layers = try layerDicts.map { try buildLayer($0, hidden: h, ffnDim: ffn, b: b) }

        let fnW = try loadF16Tensor(try requireString(m["finalNormW"], "fnW"), shape: [h], b: b)
        let fnB = try loadF16Tensor(try requireString(m["finalNormB"], "fnB"), shape: [h], b: b)
        let finalNorm = (QWDecisionNormKind.layerNorm(eps: 1e-5), QWDecisionNormWeights(weight: fnW, bias: fnB))

        let hiddenIn = try loadF16Tensor(try requireString(m["hiddenIn"], "hiddenIn"), shape: [seqLen, h], b: b)
        let outT = try pipeline.runBackbone(
            hiddenIn: hiddenIn, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: 0, mediaPrefixLen: mediaLen
        )
        let metalFinalH = pipeline.readF16AsFloats(outT, count: seqLen * h)
        let refFinalH = try loadF32(try requireString(m["refFinalH"], "refFinalH"))
        XCTAssertLessThanOrEqual(maxAbs(metalFinalH, refFinalH), 2e-3)

        // Multi-chain marker scorer parity.
        let markerPos = (try requireIntArray(m["markerPositions"], "markerPositions")).map { Int32($0) }
        let midDim = try requireInt(m["midDim"], "midDim")
        let outDim = try requireInt(m["outDim"], "outDim")
        let typeEmb = try loadF16Tensor(try requireString(m["typeEmb"], "typeEmb"), shape: [h], b: b)
        for (idx, pfx) in [(1, "c1"), (2, "c2")] {
            let w1 = try loadF16Tensor(try requireString(m["\(pfx)W1"], "\(pfx)W1"), shape: [midDim, h], b: b)
            let b1 = try loadF16Tensor(try requireString(m["\(pfx)B1"], "\(pfx)B1"), shape: [midDim], b: b)
            let w2 = try loadF16Tensor(try requireString(m["\(pfx)W2"], "\(pfx)W2"), shape: [outDim, midDim], b: b)
            let b2 = try loadF16Tensor(try requireString(m["\(pfx)B2"], "\(pfx)B2"), shape: [outDim], b: b)
            let scores = try pipeline.evaluateMarkerScorer(
                hiddenStates: outT, seqLen: seqLen, hidden: h, positions: markerPos, typeEmb: typeEmb,
                w1: w1, b1: b1, w2: w2, b2: b2, midDim: midDim, outDim: outDim, activation: .gelu
            )
            let refScores = try loadF32(try requireString(m["refChain\(idx)"], "refChain\(idx)"))
            XCTAssertLessThanOrEqual(maxAbs(scores, refScores), 2e-3)
            XCTAssertEqual(argmax(scores), argmax(refScores))
        }

        // Strict bidirectional mediaPrefixLen isolation in both directions (0.0 drift).
        let hiddenMediaPert = try loadF16Tensor(try requireString(m["hiddenMediaPert"], "hiddenMediaPert"), shape: [seqLen, h], b: b)
        let outMediaPertT = try pipeline.runBackbone(
            hiddenIn: hiddenMediaPert, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: 0, mediaPrefixLen: mediaLen
        )
        let metalMediaPert = pipeline.readF16AsFloats(outMediaPertT, count: seqLen * h)
        XCTAssertEqual(
            maxAbs(Array(metalMediaPert[(mediaLen * h)..<(seqLen * h)]), Array(metalFinalH[(mediaLen * h)..<(seqLen * h)])),
            0.0
        )

        let hiddenTextPert = try loadF16Tensor(try requireString(m["hiddenTextPert"], "hiddenTextPert"), shape: [seqLen, h], b: b)
        let outTextPertT = try pipeline.runBackbone(
            hiddenIn: hiddenTextPert, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: 0, mediaPrefixLen: mediaLen
        )
        let metalTextPert = pipeline.readF16AsFloats(outTextPertT, count: seqLen * h)
        XCTAssertEqual(
            maxAbs(Array(metalTextPert[0..<(mediaLen * h)]), Array(metalFinalH[0..<(mediaLen * h)])),
            0.0
        )
    }

    func testArchetype3PooledRerankerParity() throws {
        let m = try requireDict(manifest["archetype3_pooled_reranker"], "archetype3")
        let hal = try QWHal(geometry: QWTestGeometry.g27, weightBasis: .standard)
        let pipeline = try QWDecisionMetalPipeline(hal: hal)
        let b = QWDecisionTensorBuilder(hal: hal)

        let h = try requireInt(m["hidden"], "hidden")
        let ffn = try requireInt(m["ffnDim"], "ffnDim")
        let seqLen = try requireInt(m["seqLen"], "seqLen")
        let validLen = try requireInt(m["validLen"], "validLen")
        let midDim = try requireInt(m["midDim"], "midDim")
        let outDim = try requireInt(m["outDim"], "outDim")
        let groupIds = (try requireIntArray(m["groupIds"], "groupIds")).map { Int32($0) }
        let layerDicts = try requireDictArray(m["layers"], "layers")
        let layers = try layerDicts.map { try buildLayer($0, hidden: h, ffnDim: ffn, b: b) }

        let fnW = try loadF16Tensor(try requireString(m["finalNormW"], "fnW"), shape: [h], b: b)
        let fnB = try loadF16Tensor(try requireString(m["finalNormB"], "fnB"), shape: [h], b: b)
        let finalNorm = (QWDecisionNormKind.layerNorm(eps: 1e-5), QWDecisionNormWeights(weight: fnW, bias: fnB))

        let hiddenIn = try loadF16Tensor(try requireString(m["hiddenIn"], "hiddenIn"), shape: [seqLen, h], b: b)
        let outT = try pipeline.runBackbone(
            hiddenIn: hiddenIn, seqLen: seqLen, hidden: h, layers: layers,
            finalNorm: finalNorm, groupIds: groupIds, stateEndPos: 0, mediaPrefixLen: 0
        )

        let w1 = try loadF16Tensor(try requireString(m["w1"], "w1"), shape: [midDim, h], b: b)
        let b1 = try loadF16Tensor(try requireString(m["b1"], "b1"), shape: [midDim], b: b)
        let w2 = try loadF16Tensor(try requireString(m["w2"], "w2"), shape: [outDim, midDim], b: b)
        let b2 = try loadF16Tensor(try requireString(m["b2"], "b2"), shape: [outDim], b: b)
        let poolRefs = try requireDict(m["refPoolLogits"], "refPoolLogits")
        let poolKinds: [(String, QWDecisionModelConfig.PoolingKind)] = [
            ("firstToken", .firstToken),
            ("lastToken", .lastToken),
            ("meanNonPad", .meanNonPad),
        ]
        for (key, kind) in poolKinds {
            let logits = try pipeline.evaluatePooledClassifier(
                hiddenStates: outT, seqLen: seqLen, validLen: validLen, hidden: h, pooling: kind,
                w1: w1, b1: b1, w2: w2, b2: b2, midDim: midDim, outDim: outDim, activation: .gelu
            )
            let ref = try loadF32(try requireString(poolRefs[key], "ref_\(key)"))
            XCTAssertLessThanOrEqual(maxAbs(logits, ref), 2e-3, "pooling=\(key)")
            XCTAssertEqual(argmax(logits), argmax(ref), "argmax pooling=\(key)")
        }
    }

    func testRealLFM3GatedConvRMSNormSlidingWindowAndZeroTailRoPEParity() throws {
        let hal = try QWHal(geometry: QWTestGeometry.g27, weightBasis: .standard)
        let pipeline = try QWDecisionMetalPipeline(hal: hal)
        let b = QWDecisionTensorBuilder(hal: hal)
        let shortConv = try QWShortConv(hal: hal)
        let topoAttn = try QWDecisionTopologyAttn(hal: hal)

        // 1. Gated causal ShortConv (K=7): bcx[T, 3D] -> C * conv(B * x)
        let seqLen = 6, ch = 16, kSize = 7
        var bcx = [Float](repeating: 0, count: seqLen * ch * 3)
        var convW = [Float](repeating: 0, count: ch * kSize)
        var convB = [Float](repeating: 0, count: ch)
        for i in 0..<bcx.count { bcx[i] = Float(sin(Double(i) * 0.17) * 0.4) }
        for i in 0..<convW.count { convW[i] = Float(cos(Double(i) * 0.23) * 0.25) }
        for d in 0..<ch { convB[d] = Float(d) * 0.01 - 0.05 }

        let bcxT = try b.makeF16FromFloats(bcx, shape: [seqLen, ch * 3])
        let cwT = try b.makeF16FromFloats(convW, shape: [ch, kSize])
        let cbT = try b.makeF16FromFloats(convB, shape: [ch])
        let convOutT = try b.makeEmptyTensor(shape: [seqLen, ch], dtype: .f16)
        guard let pair1 = hal.begin() else { throw QWDecisionError.backendFailed("CB fail") }
        try shortConv.encodeGatedCausal(
            pair1.1, bcx: bcxT, weight: cwT, bias: cbT, out: convOutT,
            seqLen: seqLen, channels: ch, kernelSize: kSize
        )
        hal.end(pair1.0, pair1.1)
        pair1.0.waitUntilCompleted()
        let gotConv = pipeline.readF16AsFloats(convOutT, count: seqLen * ch)
        var refConv = [Float](repeating: 0, count: seqLen * ch)
        for t in 0..<seqLen {
            for d in 0..<ch {
                var acc = convB[d]
                for j in 0..<kSize {
                    let st = t - (kSize - 1 - j)
                    if st >= 0 {
                        let base = st * ch * 3 + d
                        acc += (Float(Float16(bcx[base])) * Float(Float16(bcx[base + ch * 2]))) * Float(Float16(convW[d * kSize + j]))
                    }
                }
                let cVal = Float(Float16(bcx[t * ch * 3 + ch + d]))
                refConv[t * ch + d] = acc * cVal
            }
        }
        XCTAssertLessThanOrEqual(maxAbs(gotConv, refConv), 2e-3)

        // 2. Arbitrary-width unrotated RMSNorm at headDim=128
        let normRows = 4, normH = 128
        let xNorm = (0..<(normRows * normH)).map { Float(sin(Double($0) * 0.13) * 0.8) }
        let gNorm = (0..<normH).map { 1.0 + Float($0 % 7) * 0.05 }
        let xNT = try b.makeF16FromFloats(xNorm, shape: [normRows, normH])
        let gNT = try b.makeF16FromFloats(gNorm, shape: [normH])
        let outNT = try b.makeEmptyTensor(shape: [normRows, normH], dtype: .f16)
        guard let pair2 = hal.begin() else { throw QWDecisionError.backendFailed("CB fail") }
        try QWNorm.encodeArbitraryRMSNorm(pair2.1, hal: hal, x: xNT, gamma: gNT, out: outNT, rows: normRows, hidden: normH, eps: 1e-5)
        hal.end(pair2.0, pair2.1)
        pair2.0.waitUntilCompleted()
        let gotNorm = pipeline.readF16AsFloats(outNT, count: normRows * normH)
        var refNorm = [Float](repeating: 0, count: normRows * normH)
        for r in 0..<normRows {
            let off = r * normH
            var s: Float = 0
            for i in 0..<normH { let v = Float(Float16(xNorm[off + i])); s += v * v }
            let rstd = 1.0 / sqrtf(s / Float(normH) + 1e-5)
            for i in 0..<normH { refNorm[off + i] = Float(Float16(xNorm[off + i])) * rstd * Float(Float16(gNorm[i])) }
        }
        XCTAssertLessThanOrEqual(maxAbs(gotNorm, refNorm), 2e-3)

        // 3. Sliding-window override under .packedSet + zeroTailFreqs Lfm3RoPE (headDim=128, factor=0.5)
        let ropeEntry = PerKindRoPEEntry(ropeTheta: 1_000_000.0, partialRotaryFactor: 0.5)
        let built = try QWRopeTables.makePerKindTable(rows: seqLen, headDim: 128, entry: ropeEntry, zeroTailFreqs: true)
        XCTAssertEqual(built.rotaryDims, 128)
        let tableT = try b.makeF32(built.cosSin, shape: [seqLen, 128])
        let xRope = (0..<(seqLen * 128)).map { Float(cos(Double($0) * 0.07)) }
        let xRopeT = try b.makeF16FromFloats(xRope, shape: [seqLen, 1, 128])
        let outRopeT = try b.makeEmptyTensor(shape: [seqLen, 1, 128], dtype: .f16)
        guard let pair3 = hal.begin() else { throw QWDecisionError.backendFailed("CB fail") }
        try QWRopeApply.encodeZeroTail(pair3.1, hal: hal, x: xRopeT, out: outRopeT, table: tableT, seqLen: seqLen, numHeads: 1, headDim: 128, rotaryDims: 128, zeroTail: false)
        hal.end(pair3.0, pair3.1)
        pair3.0.waitUntilCompleted()
        let gotRope = pipeline.readF16AsFloats(outRopeT, count: seqLen * 128)
        for s in 0..<seqLen {
            for d in 32..<64 {
                XCTAssertEqual(gotRope[s * 128 + d], Float(Float16(xRope[s * 128 + d])), accuracy: 1e-4)
                XCTAssertEqual(gotRope[s * 128 + 64 + d], Float(Float16(xRope[s * 128 + 64 + d])), accuracy: 1e-4)
            }
        }
        // Sliding window=2 causal check under .packedSet: perturbing t=3 leaves t=1 unchanged (0.0), and t=0 hidden from t=3 (0.0).
        let qkV = (0..<(4 * 1 * 16)).map { Float(($0 + 1)) * 0.02 }
        var qkVPert3 = qkV
        for i in (3 * 16)..<(4 * 16) { qkVPert3[i] += 0.9 }
        let qT = try b.makeF16FromFloats(qkV, shape: [4, 1, 16])
        let kT0 = try b.makeF16FromFloats(qkV, shape: [4, 1, 16])
        let kT1 = try b.makeF16FromFloats(qkVPert3, shape: [4, 1, 16])
        let gT = try b.makeI32([0, 0, 1, 1], shape: [4])
        let o0 = try b.makeEmptyTensor(shape: [4, 1, 16], dtype: .f16)
        let o1 = try b.makeEmptyTensor(shape: [4, 1, 16], dtype: .f16)
        guard let pair4 = hal.begin() else { throw QWDecisionError.backendFailed("CB fail") }
        try topoAttn.encodePrefill(pair4.1, q: qT, k: kT0, v: kT0, groupIds: gT, out: o0, seqLen: 4, qHeads: 1, kvHeads: 1, headDim: 16, topology: .packedSet(maxSetSize: 4), slidingWindow: 2)
        try topoAttn.encodePrefill(pair4.1, q: qT, k: kT1, v: kT1, groupIds: gT, out: o1, seqLen: 4, qHeads: 1, kvHeads: 1, headDim: 16, topology: .packedSet(maxSetSize: 4), slidingWindow: 2)
        hal.end(pair4.0, pair4.1)
        pair4.0.waitUntilCompleted()
        let outA0 = pipeline.readF16AsFloats(o0, count: 64)
        let outA1 = pipeline.readF16AsFloats(o1, count: 64)
        XCTAssertEqual(maxAbs(Array(outA0[0..<48]), Array(outA1[0..<48])), 0.0)
    }

    private func buildLayer(_ d: [String: Any], hidden: Int, ffnDim: Int, b: QWDecisionTensorBuilder) throws -> QWDecisionLayerWeights {
        let eps = Float((d["eps"] as? Double) ?? 1e-5)
        let unitOffset = (d["unitOffset"] as? Bool) ?? false
        let preBlockKind = makeNormKind(try requireString(d["preBlockKind"], "preBlockKind"), eps: eps, unitOffset: unitOffset)
        let postBlockKind = makeNormKind(try requireString(d["postBlockKind"], "postBlockKind"), eps: eps, unitOffset: unitOffset)
        let preFFNKind = makeNormKind(try requireString(d["preFFNKind"], "preFFNKind"), eps: eps, unitOffset: unitOffset)
        let postFFNKind = makeNormKind(try requireString(d["postFFNKind"], "postFFNKind"), eps: eps, unitOffset: unitOffset)
        let act: QWDecisionHeadActivation = (try requireString(d["ffnAct"], "ffnAct")) == "silu" ? .silu : .gelu

        let blockKindStr = try requireString(d["blockKind"], "blockKind")
        let block: QWDecisionBlockKind
        var scW: QWDecisionShortConvWeights?
        var attnW: QWDecisionAttentionWeights?
        if blockKindStr == "shortConvCausal" || blockKindStr == "shortConvCentered" {
            let ch = try requireInt(d["channels"], "channels")
            let k = try requireInt(d["kernelSize"], "kernelSize")
            let inWName = d["inProjW"] as? String
            let inW = try inWName.map { try loadF16Tensor($0, shape: [ch, hidden], b: b) }
            let convW = try loadF16Tensor(try requireString(d["convW"], "convW"), shape: [ch, k], b: b)
            let convB = try (d["convB"] as? String).map { try loadF16Tensor($0, shape: [ch], b: b) }
            let outW = try loadF16Tensor(try requireString(d["outProjW"], "outProjW"), shape: [hidden, ch], b: b)
            let outB = try (d["outProjB"] as? String).map { try loadF16Tensor($0, shape: [hidden], b: b) }
            scW = QWDecisionShortConvWeights(inProjW: inW, convW: convW, convB: convB, outProjW: outW, outProjB: outB)
            block = blockKindStr == "shortConvCentered"
                ? .shortConvCentered(channels: ch, kernelSize: k, hasBias: convB != nil, hasInProj: inW != nil)
                : .shortConvCausal(channels: ch, kernelSize: k, hasBias: convB != nil, hasInProj: inW != nil)
        } else {
            let qH = try requireInt(d["qHeads"], "qHeads")
            let kvH = try requireInt(d["kvHeads"], "kvHeads")
            let hd = try requireInt(d["headDim"], "headDim")
            let topoStr = try requireString(d["topology"], "topology")
            let isolateMedia = (d["isolateMedia"] as? Bool) ?? false
            let topo: QWDecisionModelConfig.AttentionTopology = switch topoStr {
            case "packedSet": .packedSet(maxSetSize: 16)
            case "bidirectionalEncoder": .bidirectionalEncoder(isolateMediaPrefix: isolateMedia)
            case "prefixBidirectional": .prefixBidirectional
            default: .causal(supportsPrefixCache: false)
            }
            let ropeEntry: PerKindRoPEEntry? = (d["ropeTheta"] as? Double).map {
                PerKindRoPEEntry(ropeTheta: $0, partialRotaryFactor: (d["partialFactor"] as? Double) ?? 1.0)
            }
            let zeroTail = (d["zeroTail"] as? Bool) ?? false
            let hasGate = (d["hasGate"] as? Bool) ?? false
            attnW = QWDecisionAttentionWeights(
                qW: try loadF16Tensor(try requireString(d["qW"], "qW"), shape: [qH * hd, hidden], b: b),
                kW: try loadF16Tensor(try requireString(d["kW"], "kW"), shape: [kvH * hd, hidden], b: b),
                vW: try loadF16Tensor(try requireString(d["vW"], "vW"), shape: [kvH * hd, hidden], b: b),
                oW: try loadF16Tensor(try requireString(d["oW"], "oW"), shape: [hidden, qH * hd], b: b),
                gateW: try (d["gateW"] as? String).map { try loadF16Tensor($0, shape: [qH, hidden], b: b) },
                gateB: try (d["gateB"] as? String).map { try loadF16Tensor($0, shape: [qH], b: b) }
            )
            block = .attention(
                topology: topo, qHeads: qH, kvHeads: kvH, headDim: hd,
                rope: ropeEntry, zeroTail: zeroTail, hasSigmoidGate: hasGate, scale: nil
            )
        }

        let spec = QWDecisionLayerSpec(
            hidden: hidden, ffnDim: ffnDim, block: block,
            preBlockNorm: preBlockKind, postBlockNorm: postBlockKind,
            preFFNNorm: preFFNKind, postFFNNorm: postFFNKind, ffnActivation: act
        )
        return QWDecisionLayerWeights(
            spec: spec,
            preBlockNorm: try makeNormWeights(w: d["preBlockW"] as? String, bias: d["preBlockB"] as? String, hidden: hidden, b: b),
            postBlockNorm: try makeNormWeights(w: d["postBlockW"] as? String, bias: d["postBlockB"] as? String, hidden: hidden, b: b),
            preFFNNorm: try makeNormWeights(w: d["preFFNW"] as? String, bias: d["preFFNB"] as? String, hidden: hidden, b: b),
            postFFNNorm: try makeNormWeights(w: d["postFFNW"] as? String, bias: d["postFFNB"] as? String, hidden: hidden, b: b),
            shortConv: scW, attention: attnW,
            ffnUpW: try loadF16Tensor(try requireString(d["ffnUpW"], "ffnUpW"), shape: [ffnDim, hidden], b: b),
            ffnUpB: try (d["ffnUpB"] as? String).map { try loadF16Tensor($0, shape: [ffnDim], b: b) },
            ffnDownW: try loadF16Tensor(try requireString(d["ffnDownW"], "ffnDownW"), shape: [hidden, ffnDim], b: b),
            ffnDownB: try (d["ffnDownB"] as? String).map { try loadF16Tensor($0, shape: [hidden], b: b) }
        )
    }

    private func makeNormKind(_ s: String, eps: Float, unitOffset: Bool) -> QWDecisionNormKind {
        switch s {
        case "none": return .none
        case "rms": return .rmsNorm(eps: eps, unitOffset: unitOffset)
        default: return .layerNorm(eps: eps)
        }
    }

    private func makeNormWeights(w: String?, bias: String?, hidden: Int, b: QWDecisionTensorBuilder) throws -> QWDecisionNormWeights? {
        guard let w else { return nil }
        return QWDecisionNormWeights(
            weight: try loadF16Tensor(w, shape: [hidden], b: b),
            bias: try bias.map { try loadF16Tensor($0, shape: [hidden], b: b) }
        )
    }

    private func loadF16Tensor(_ name: String, shape: [Int], b: QWDecisionTensorBuilder) throws -> QWTensorValue {
        try b.makeF16FromU16(loadU16(name), shape: shape)
    }

    private func loadBF16Tensor(_ name: String, shape: [Int], b: QWDecisionTensorBuilder) throws -> QWTensorValue {
        try b.makeBF16FromU16(loadU16(name), shape: shape)
    }

    private func loadU16(_ name: String) throws -> [UInt16] {
        let data = try Data(contentsOf: oracleDir.appendingPathComponent(name))
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: UInt16.self))
        }
    }

    private func loadF32(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: oracleDir.appendingPathComponent(name))
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private func maxAbs(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).reduce(Float(0)) { acc, pair in max(acc, abs(pair.0 - pair.1)) }
    }

    private func argmax(_ a: [Float]) -> Int {
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for (i, v) in a.enumerated() where v > bestVal {
            bestVal = v
            bestIdx = i
        }
        return bestIdx
    }

    private func requireDict(_ v: Any?, _ label: String) throws -> [String: Any] {
        guard let d = v as? [String: Any] else { throw XCTSkip("Missing dict \(label)") }
        return d
    }

    private func requireDictArray(_ v: Any?, _ label: String) throws -> [[String: Any]] {
        guard let a = v as? [[String: Any]] else { throw XCTSkip("Missing dict array \(label)") }
        return a
    }

    private func requireInt(_ v: Any?, _ label: String) throws -> Int {
        guard let i = v as? Int else { throw XCTSkip("Missing int \(label)") }
        return i
    }

    private func requireIntArray(_ v: Any?, _ label: String) throws -> [Int] {
        guard let a = v as? [Int] else { throw XCTSkip("Missing int array \(label)") }
        return a
    }

    private func requireString(_ v: Any?, _ label: String) throws -> String {
        guard let s = v as? String else { throw XCTSkip("Missing string \(label)") }
        return s
    }
}
