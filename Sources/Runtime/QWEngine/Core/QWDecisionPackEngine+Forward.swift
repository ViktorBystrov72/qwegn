// QWDecisionPackEngine+Forward.swift — Single-CB Metal 4 forward pass, sparse vocab logits, and state pointer head.

import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWDecision
import QWGEMM
import QWGDN
import QWHAL
import QWHead
import QWMM
import QWNorm
import QWRope

extension QWDecisionPackEngine {
    struct PreparedDecisionSeq {
        let tokens: [Int]
        let visualSpans: [QWVisualSpan]
        let encodedImages: [(grid: QWVisionGrid, features: [Float])]
        let topology: QWDecisionModelConfig.AttentionTopology
        let stateEndPos: Int
        let questionSpans: [Range<Int>]
        let expandedStateLen: Int
        let origToExpandedPtr: [Int?]
        let origPtrLen: Int
        let slotTargets: [(slotIdx: Int, pos: Int, endPos: Int?, isPtr: Bool)]
    }

    func executeSequenceOnGPU(
        _ item: PreparedDecisionSeq,
        candidateIds: [Int]
    ) throws -> [(slotIdx: Int, output: QWDecisionSlotOutput)] {
        let seqLen = item.tokens.count
        let hidden = descriptor.hidden
        let ffnDim = descriptor.denseIntermediate ?? descriptor.hidden
        let qHeads = descriptor.qHeads
        let kvHeads = descriptor.kvHeads
        let headDim = descriptor.headDim
        let qElems = seqLen * qHeads * headDim
        let convCh = hidden

        let b = QWDecisionTensorBuilder(hal: hal)
        let x16 = try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let normed16 = try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let branch16 = try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let ffnIn16 = try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)
        let ffnGu16 = try b.makeEmptyTensor(shape: [seqLen, 2 * ffnDim], dtype: .f16)
        let ffnAct16 = try b.makeEmptyTensor(shape: [seqLen, ffnDim], dtype: .f16)
        let convBcx16 = try b.makeEmptyTensor(shape: [seqLen, 3 * convCh], dtype: .f16)
        let convOut16 = try b.makeEmptyTensor(shape: [seqLen, convCh], dtype: .f16)
        let qBuf16 = try b.makeEmptyTensor(shape: [seqLen, qHeads * headDim], dtype: .f16)
        let kBuf16 = try b.makeEmptyTensor(shape: [seqLen, kvHeads * headDim], dtype: .f16)
        let vBuf16 = try b.makeEmptyTensor(shape: [seqLen, kvHeads * headDim], dtype: .f16)
        let qRot16 = try b.makeEmptyTensor(shape: [seqLen, qHeads * headDim], dtype: .f16)
        let kRot16 = try b.makeEmptyTensor(shape: [seqLen, kvHeads * headDim], dtype: .f16)
        let attnOut16 = try b.makeEmptyTensor(shape: [seqLen, qHeads * headDim], dtype: .f16)
        let gatePadCount = max(8, qwAlignUp(seqLen * qHeads, 8))
        let gateBuf16 = try b.makeEmptyTensor(shape: [gatePadCount], dtype: .f16)
        let scratch32Count = max(seqLen * 2 * ffnDim, max(seqLen * 3 * convCh, max(qElems, seqLen * hidden)))
        let scratch32 = try b.makeEmptyTensor(shape: [scratch32Count], dtype: .f32)
        let residualF32 = weights.layers.allSatisfy({ $0.postBlockNorm != nil && $0.postFFNNorm != nil })
            ? nil : (try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f32))
        let final16 = try b.makeEmptyTensor(shape: [seqLen, hidden], dtype: .f16)

        fillInputEmbeddings(tokens: item.tokens, encodedImages: item.encodedImages, visualSpans: item.visualSpans, into: x16)
        if let residualF32 {
            let src = x16.region.pointer.advanced(by: x16.offset).assumingMemoryBound(to: Float16.self)
            let dst = residualF32.region.pointer.advanced(by: residualF32.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<(seqLen * hidden) { dst[i] = Float(src[i]) }
        }

        let groupIds = buildGroupIds(seqLen: seqLen, stateEndPos: item.stateEndPos, questionSpans: item.questionSpans)
        let groupTensor = try b.makeI32(groupIds, shape: [seqLen])
        let ropeTables = try prepareRoPETables(seqLen: seqLen, builder: b)

        let codeTargets = item.slotTargets.filter { !$0.isPtr }
        let ptrTargets = item.slotTargets.filter { $0.isPtr }
        let validCandIds = candidateIds.filter { $0 >= 0 && $0 < descriptor.vocab }
        let uniquePosList = Array(Set(codeTargets.map(\.pos))).sorted()

        var codeGathered16: QWTensorValue?
        var codePosTensor: QWTensorValue?
        var codeCandTensor: QWTensorValue?
        var codeOutTensor: QWTensorValue?
        if !uniquePosList.isEmpty && !validCandIds.isEmpty {
            codePosTensor = try b.makeI32(uniquePosList.map(Int32.init), shape: [uniquePosList.count])
            codeGathered16 = try b.makeEmptyTensor(shape: [uniquePosList.count, hidden], dtype: .f16)
            codeCandTensor = try b.makeI32(validCandIds.map(Int32.init), shape: [validCandIds.count])
            codeOutTensor = try b.makeEmptyTensor(shape: [uniquePosList.count, validCandIds.count], dtype: .f32)
        }

        var ptrOutBuffers: [(slotIdx: Int, start: QWTensorValue, end: QWTensorValue)] = []
        if !ptrTargets.isEmpty && item.expandedStateLen > 0 {
            ptrOutBuffers.reserveCapacity(ptrTargets.count)
            for pt in ptrTargets {
                let oS = try b.makeEmptyTensor(shape: [item.expandedStateLen], dtype: .f32)
                let oE = try b.makeEmptyTensor(shape: [item.expandedStateLen], dtype: .f32)
                ptrOutBuffers.append((pt.slotIdx, oS, oE))
            }
        }

        guard let (cb, enc) = hal.begin() else {
            throw QWDecisionError.backendFailed("failed to open Metal command buffer")
        }

        let eps = descriptor.rmsEps
        for layer in weights.layers {
            try QWNorm.encodeArbitraryRMSNorm(
                enc, hal: hal, x: x16, gamma: layer.preBlockNorm, out: normed16,
                rows: seqLen, hidden: hidden, eps: eps, hasUnitOffset: false
            )
            enc.memoryBarrier(scope: .buffers)

            switch layer.op {
            case .conv(let cw):
                if cw.inProj.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: normed16, count: seqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: cw.inProj, x16: normed16, y16: convBcx16, scratch32: scratch32, rows: seqLen)
                enc.memoryBarrier(scope: .buffers)
                try shortConv.encodeGatedCausal(
                    enc, bcx: convBcx16, weight: cw.convW, bias: cw.convB, out: convOut16,
                    seqLen: seqLen, channels: cw.channels, kernelSize: cw.kernelSize
                )
                enc.memoryBarrier(scope: .buffers)
                if cw.outProj.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: convOut16, count: seqLen * cw.channels)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeOutAndResidual(
                    enc: enc, w: cw.outProj, in16: convOut16, branch16: branch16, postNorm: layer.postBlockNorm,
                    x16: x16, residualF32: residualF32, scratch32: scratch32, seqLen: seqLen, hidden: hidden, eps: eps
                )

            case .attn(let aw):
                let g16 = gateBuf16.slice(byteOffset: 0, count: seqLen * qHeads, dtype: .f16)
                if let gateProj = aw.gateProj {
                    try encodeLinear(enc, w: gateProj, x16: normed16, y16: g16, scratch32: scratch32, rows: seqLen)
                }
                if (aw.qProj.needsRotation || aw.kProj.needsRotation || aw.vProj.needsRotation) && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: normed16, count: seqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: aw.qProj, x16: normed16, y16: qBuf16, scratch32: scratch32, rows: seqLen)
                try encodeLinear(enc, w: aw.kProj, x16: normed16, y16: kBuf16, scratch32: scratch32, rows: seqLen)
                try encodeLinear(enc, w: aw.vProj, x16: normed16, y16: vBuf16, scratch32: scratch32, rows: seqLen)
                enc.memoryBarrier(scope: .buffers)

                if let qN = aw.qNorm {
                    try QWNorm.encodeArbitraryRMSNorm(
                        enc, hal: hal, x: qBuf16, gamma: qN, out: qBuf16,
                        rows: seqLen * qHeads, hidden: headDim, eps: eps, hasUnitOffset: false
                    )
                }
                if let kN = aw.kNorm {
                    try QWNorm.encodeArbitraryRMSNorm(
                        enc, hal: hal, x: kBuf16, gamma: kN, out: kBuf16,
                        rows: seqLen * kvHeads, hidden: headDim, eps: eps, hasUnitOffset: false
                    )
                }
                if aw.qNorm != nil || aw.kNorm != nil {
                    enc.memoryBarrier(scope: .buffers)
                }

                let activeQ: QWTensorValue
                let activeK: QWTensorValue
                if let rt = ropeTables[aw.ropeKind] ?? ropeTables["default"] {
                    try QWRopeApply.encodeZeroTail(
                        enc, hal: hal, x: qBuf16, out: qRot16, table: rt.table,
                        seqLen: seqLen, numHeads: qHeads, headDim: headDim, rotaryDims: rt.rotaryDims,
                        posOffset: 0, zeroTail: rt.zeroTailArg
                    )
                    try QWRopeApply.encodeZeroTail(
                        enc, hal: hal, x: kBuf16, out: kRot16, table: rt.table,
                        seqLen: seqLen, numHeads: kvHeads, headDim: headDim, rotaryDims: rt.rotaryDims,
                        posOffset: 0, zeroTail: rt.zeroTailArg
                    )
                    enc.memoryBarrier(scope: .buffers)
                    activeQ = qRot16
                    activeK = kRot16
                } else {
                    activeQ = qBuf16
                    activeK = kBuf16
                }

                let attnTopo: QWDecisionModelConfig.AttentionTopology = aw.isSliding
                    ? .causal(supportsPrefixCache: false) : item.topology
                try topoAttn.encodePrefill(
                    enc, q: activeQ, k: activeK, v: vBuf16, groupIds: groupTensor, out: attnOut16,
                    seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                    topology: attnTopo, stateEndPos: aw.isSliding ? 0 : item.stateEndPos,
                    mediaPrefixLen: 0, slidingWindow: aw.isSliding ? aw.slidingWindow : 0
                )
                if aw.gateProj != nil {
                    enc.memoryBarrier(scope: .buffers)
                    try topoAttn.encodeHeadwiseSigmoidGate(
                        enc, out: attnOut16, gate: g16, seqLen: seqLen, qHeads: qHeads, headDim: headDim
                    )
                }
                enc.memoryBarrier(scope: .buffers)
                if aw.oProj.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: attnOut16, count: seqLen * qHeads * headDim)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeOutAndResidual(
                    enc: enc, w: aw.oProj, in16: attnOut16, branch16: branch16, postNorm: layer.postBlockNorm,
                    x16: x16, residualF32: residualF32, scratch32: scratch32, seqLen: seqLen, hidden: hidden, eps: eps
                )
            }
            enc.memoryBarrier(scope: .buffers)

            try QWNorm.encodeArbitraryRMSNorm(
                enc, hal: hal, x: x16, gamma: layer.preFFNNorm, out: ffnIn16,
                rows: seqLen, hidden: hidden, eps: eps, hasUnitOffset: false
            )
            enc.memoryBarrier(scope: .buffers)

            if let w13 = layer.w13 {
                if w13.needsRotation && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: ffnIn16, count: seqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                try encodeLinear(enc, w: w13, x16: ffnIn16, y16: ffnGu16, scratch32: scratch32, rows: seqLen)
            } else if let w1 = layer.w1, let w3 = layer.w3 {
                if (w1.needsRotation || w3.needsRotation) && hal.weightBasis.rotatesActivations {
                    try QWGemm.encodeFwht32(enc, hal: hal, x: ffnIn16, count: seqLen * hidden)
                    enc.memoryBarrier(scope: .buffers)
                }
                let gateOut = ffnGu16.slice(byteOffset: 0, count: seqLen * ffnDim, dtype: .f16)
                let upOut = ffnAct16
                try encodeLinear(enc, w: w1, x16: ffnIn16, y16: gateOut, scratch32: scratch32, rows: seqLen)
                try encodeLinear(enc, w: w3, x16: ffnIn16, y16: upOut, scratch32: scratch32, rows: seqLen)
                enc.memoryBarrier(scope: .buffers)
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                throw QWDecisionError.backendFailed("unfused quant SwiGLU layout requires equal container w1/w3")
            }
            enc.memoryBarrier(scope: .buffers)
            let fuseActRot = layer.w2.needsRotation && hal.weightBasis.rotatesActivations
            try QWGemmFP.encodeSwiGLUSoftcap(
                enc, hal: hal, gu: QWFPBufferSlice(ffnGu16), act: QWFPBufferSlice(ffnAct16),
                rows: seqLen, intermediate: ffnDim, gateCap: 0, upCap: 0,
                inDtype: .f16, outDtype: .f16, rotateFwht32: fuseActRot
            )
            enc.memoryBarrier(scope: .buffers)
            try encodeOutAndResidual(
                enc: enc, w: layer.w2, in16: ffnAct16, branch16: branch16, postNorm: layer.postFFNNorm,
                x16: x16, residualF32: residualF32, scratch32: scratch32, seqLen: seqLen, hidden: hidden, eps: eps
            )
            enc.memoryBarrier(scope: .buffers)
        }

        try QWNorm.encodeArbitraryRMSNorm(
            enc, hal: hal, x: x16, gamma: weights.finalNorm, out: final16,
            rows: seqLen, hidden: hidden, eps: eps, hasUnitOffset: false
        )
        enc.memoryBarrier(scope: .buffers)

        if let codePosTensor, let codeGathered16, let codeCandTensor, let codeOutTensor,
           let lmHeadBF16 = weights.lmHeadBF16 {
            try headOp.encodeMarkerGather(
                enc, hiddenState: final16, positions: codePosTensor, typeEmb: nil, out: codeGathered16,
                seqLen: seqLen, count: uniquePosList.count, hidden: hidden
            )
            enc.memoryBarrier(scope: .buffers)
            try headOp.encodeSparseVocabRowsBF16(
                enc, hiddenState: codeGathered16, lmHeadBF16: lmHeadBF16, candidateIds: codeCandTensor,
                out: codeOutTensor, rows: uniquePosList.count, candCount: validCandIds.count,
                hidden: hidden, vocabSize: descriptor.vocab
            )
        }

        if !ptrTargets.isEmpty && item.expandedStateLen > 0 {
            let stateSlice = final16.slice(byteOffset: 0, count: item.expandedStateLen * hidden, dtype: .f16)
            for (idx, pt) in ptrTargets.enumerated() {
                let sPos = max(0, min(seqLen - 1, pt.pos))
                let ePos = max(0, min(seqLen - 1, pt.endPos ?? pt.pos))
                let qStart = final16.slice(byteOffset: sPos * hidden * 2, count: hidden, dtype: .f16)
                let qEnd = final16.slice(byteOffset: ePos * hidden * 2, count: hidden, dtype: .f16)
                try headOp.encodePointerScores(
                    enc, stateHidden: stateSlice, qStart: qStart, qEnd: qEnd,
                    outStart: ptrOutBuffers[idx].start, outEnd: ptrOutBuffers[idx].end,
                    stateLen: item.expandedStateLen, hidden: hidden
                )
            }
        }

        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw QWDecisionError.backendFailed("decision forward command buffer failed (status=\(cb.status.rawValue))")
        }

        var results: [(slotIdx: Int, output: QWDecisionSlotOutput)] = []
        if let codeOutTensor {
            let posToRow = Dictionary(uniqueKeysWithValues: uniquePosList.enumerated().map { ($1, $0) })
            let outPtr = codeOutTensor.region.pointer.advanced(by: codeOutTensor.offset).assumingMemoryBound(to: Float.self)
            let stride = validCandIds.count
            for ct in codeTargets {
                guard let rowIdx = posToRow[ct.pos] else { continue }
                var map: [Int: Float] = [:]
                map.reserveCapacity(stride)
                let base = rowIdx * stride
                for (cIdx, tid) in validCandIds.enumerated() { map[tid] = outPtr[base + cIdx] }
                results.append((ct.slotIdx, QWDecisionSlotOutput(tokenLogits: map)))
            }
        } else {
            for ct in codeTargets { results.append((ct.slotIdx, QWDecisionSlotOutput())) }
        }

        for buf in ptrOutBuffers {
            let sRaw = buf.start.region.pointer.advanced(by: buf.start.offset).assumingMemoryBound(to: Float.self)
            let eRaw = buf.end.region.pointer.advanced(by: buf.end.offset).assumingMemoryBound(to: Float.self)
            let sArr: [Float]
            let eArr: [Float]
            if item.origToExpandedPtr.isEmpty {
                sArr = Array(UnsafeBufferPointer(start: sRaw, count: item.expandedStateLen))
                eArr = Array(UnsafeBufferPointer(start: eRaw, count: item.expandedStateLen))
            } else {
                sArr = item.origToExpandedPtr.map { expIdx in
                    guard let expIdx, expIdx < item.expandedStateLen else { return -Float.infinity }
                    return sRaw[expIdx]
                }
                eArr = item.origToExpandedPtr.map { expIdx in
                    guard let expIdx, expIdx < item.expandedStateLen else { return -Float.infinity }
                    return eRaw[expIdx]
                }
            }
            results.append((buf.slotIdx, QWDecisionSlotOutput(pointerStartLogits: sArr, pointerEndLogits: eArr)))
        }
        return results
    }

    private func encodeLinearF32(
        _ enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        x16: QWTensorValue,
        y32: QWTensorValue,
        rows: Int
    ) throws {
        switch w {
        case .bf16(let bank):
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank, x: QWFPBufferSlice(x16), y: QWFPBufferSlice(y32),
                m: rows, inDtype: .f16, outDtype: .f32
            )
            case .quant(let view):
            if hal.caps.isMetal4TensorCapable && (view.n % 64 == 0 || !view.isIncumbentContainer) && view.k % 64 == 0 {
                let yTmp16 = y32.slice(byteOffset: 0, count: rows * view.n, dtype: .f16)
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16, y16: yTmp16, y32: y32, rows: rows)
            } else {
                try QWGemmAQ.encode(
                    enc, hal: hal, blocks: view.blocks, x: x16, y: y32,
                    n: view.n, k: view.k, bits: view.containerKind.bits, group: view.containerKind.groupSize,
                    rows: rows, isSHVQ: view.containerKind.isSHVQ
                )
            }
        }
    }

    private func encodeLinear(
        _ enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        x16: QWTensorValue,
        y16: QWTensorValue,
        scratch32: QWTensorValue,
        rows: Int
    ) throws {
        switch w {
        case .bf16(let bank):
            try QWGemmFP.encodeBF16(
                enc, hal: hal, weight: bank, x: QWFPBufferSlice(x16), y: QWFPBufferSlice(y16),
                m: rows, inDtype: .f16, outDtype: .f16
            )
        case .quant(let view):
            if hal.caps.isMetal4TensorCapable && (view.n % 64 == 0 || !view.isIncumbentContainer) && view.k % 64 == 0 {
                try QWGemm.encodeNaxF32(enc, hal: hal, weight: view, x16: x16, y16: y16, y32: y16, rows: rows)
            } else {
                let y32 = scratch32.slice(byteOffset: 0, count: rows * view.n, dtype: .f32)
                try QWGemmAQ.encode(
                    enc, hal: hal, blocks: view.blocks, x: x16, y: y32,
                    n: view.n, k: view.k, bits: view.containerKind.bits, group: view.containerKind.groupSize,
                    rows: rows, isSHVQ: view.containerKind.isSHVQ
                )
                enc.memoryBarrier(scope: .buffers)
                try QWGemm.encodeCast(enc, hal: hal, x32: y32, y16: y16, count: rows * view.n)
            }
        }
    }

    private func encodeOutAndResidual(
        enc: any MTLComputeCommandEncoder,
        w: QWDecisionLinearW,
        in16: QWTensorValue,
        branch16: QWTensorValue,
        postNorm: QWTensorValue?,
        x16: QWTensorValue,
        residualF32: QWTensorValue?,
        scratch32: QWTensorValue,
        seqLen: Int,
        hidden: Int,
        eps: Float
    ) throws {
        if let postNorm {
            try encodeLinear(enc, w: w, x16: in16, y16: branch16, scratch32: scratch32, rows: seqLen)
            enc.memoryBarrier(scope: .buffers)
            try QWNorm.encodePostRMSNormResidualAdd(
                enc, hal: hal, residual: x16, branch: branch16, gamma: postNorm,
                rows: seqLen, hidden: hidden, eps: eps, hasUnitOffset: false
            )
            return
        }
        guard let residualF32 else { return }
        let b32 = scratch32.slice(byteOffset: 0, count: seqLen * hidden, dtype: .f32)
        try encodeLinearF32(enc, w: w, x16: in16, y32: b32, rows: seqLen)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeAddF32(enc, hal: hal, a: residualF32, b: b32, out: residualF32, count: seqLen * hidden)
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeCast(enc, hal: hal, x32: residualF32, y16: x16, count: seqLen * hidden)
    }

    private func fillInputEmbeddings(
        tokens: [Int],
        encodedImages: [(grid: QWVisionGrid, features: [Float])],
        visualSpans: [QWVisualSpan],
        into x16: QWTensorValue
    ) {
        _ = encodedImages
        let hidden = descriptor.hidden
        let embPtr = weights.embedTokens.region.pointer.advanced(by: weights.embedTokens.offset).assumingMemoryBound(to: UInt16.self)
        let dstPtr = x16.region.pointer.advanced(by: x16.offset).assumingMemoryBound(to: Float16.self)
        for (rowIdx, tok) in tokens.enumerated() {
            let clampedTok = max(0, min(descriptor.vocab - 1, tok))
            let srcRow = embPtr.advanced(by: clampedTok * hidden)
            let dstRow = dstPtr.advanced(by: rowIdx * hidden)
            for d in 0..<hidden {
                dstRow[d] = Float16(Float(bitPattern: UInt32(srcRow[d]) << 16))
            }
        }
        for span in visualSpans {
            let feats = span.embeddings
            let count = min(span.tokenCount, feats.count / hidden)
            for localRow in 0..<count {
                let dstRow = dstPtr.advanced(by: (span.tokenOffset + localRow) * hidden)
                let srcOff = localRow * hidden
                for d in 0..<hidden {
                    dstRow[d] = Float16(feats[srcOff + d])
                }
            }
        }
    }

    private func buildGroupIds(seqLen: Int, stateEndPos: Int, questionSpans: [Range<Int>]) -> [Int32] {
        if questionSpans.isEmpty && stateEndPos <= 0 {
            return [Int32](repeating: 0, count: seqLen)
        }
        var groups = [Int32](repeating: 0, count: seqLen)
        for (qIdx, span) in questionSpans.enumerated() {
            let lo = max(0, span.lowerBound)
            let hi = min(seqLen, span.upperBound)
            for p in lo..<hi { groups[p] = Int32(qIdx + 1) }
        }
        return groups
    }

    private func prepareRoPETables(
        seqLen: Int, builder: QWDecisionTensorBuilder
    ) throws -> [String: (table: QWTensorValue, rotaryDims: Int, zeroTailArg: Bool)] {
        let hd = descriptor.headDim
        var out: [String: (table: QWTensorValue, rotaryDims: Int, zeroTailArg: Bool)] = [:]
        if let perKind = descriptor.perKindRoPEConfig, !perKind.byLayerKind.isEmpty {
            let zeroTail = perKind.zeroTailPartialRotation
            for (kind, entry) in perKind.byLayerKind {
                let built = try QWRopeTables.makePerKindTable(rows: seqLen, headDim: hd, entry: entry, zeroTailFreqs: zeroTail)
                let t = try builder.makeF32(built.cosSin, shape: [seqLen, built.rotaryDims])
                out[kind] = (t, built.rotaryDims, false)
            }
        } else {
            let entry = PerKindRoPEEntry(ropeTheta: descriptor.ropeTheta, partialRotaryFactor: 1.0, ropeType: "default")
            let built = try QWRopeTables.makePerKindTable(rows: seqLen, headDim: hd, entry: entry, zeroTailFreqs: false)
            let t = try builder.makeF32(built.cosSin, shape: [seqLen, built.rotaryDims])
            out["default"] = (t, built.rotaryDims, false)
        }
        return out
    }
}
