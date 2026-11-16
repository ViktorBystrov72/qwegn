// QWDFlashDrafter+Propose.swift — DFlash 2 block-diffusion draft proposal and selection.

import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWMTP
import QWQuant

extension QWDFlashDrafter {
    func encodeCaptureTargetHidden(_ enc: any MTLComputeCommandEncoder,
                                   source: QWTensorValue,
                                   srcRowOffset: Int,
                                   dstRowOffset: Int,
                                   rows: Int,
                                   slot: Int) {
        guard rows > 0, (0..<5).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureTarget)
        source.bind(enc, index: 0)
        targetHidden.bind(enc, index: 1)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        enc.setBytes(&sOff, length: 4, index: 2)
        enc.setBytes(&dOff, length: 4, index: 3)
        enc.setBytes(&slotU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeQ4Matmul(_ enc: any MTLComputeCommandEncoder,
                        weight: QWWeightView,
                        input: QWTensorValue,
                        output: QWTensorValue,
                        rows: Int) {
        if !QWMTPBatchedMLP.usesQ4GMKernel(weight) {
            encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
            return
        }
        let n = weight.n, k = weight.k
        let useNaxSmall = isMetal4TensorCapable && rows >= 9 && n % 64 == 0 && k % 64 == 0
        if rows == 1 {
            enc.setComputePipelineState(psoGemvT4)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            output.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if !useNaxSmall, let pso = psoDownBatched[rows] {
            let isAQ4 = !weight.isIncumbentContainer
            let rowPaired = QWMTPBatchedMLP.usesRowPairs(weight, activeRows: rows)
            let resolved = isAQ4
                ? (rowPaired ? (psoDownRowPairedAQ4[rows] ?? pso) : (psoDownBatchedAQ4[rows] ?? pso))
                : ((rowPaired ? psoDownRowPaired[rows] : nil) ?? pso)
            enc.setComputePipelineState(resolved)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n) | (isAQ4 ? 0x8000_0000 : 0), kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            let tptg = rowPaired && (isAQ4 ? psoDownRowPairedAQ4[rows] != nil : psoDownRowPaired[rows] != nil) ? 64 : 128
            enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1))
        } else if !useNaxSmall && (9...16).contains(rows) {
            let c0 = rows / 2, c1 = rows - c0
            encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: 0, count: c0 * k),
                           output: output.slice(byteOffset: 0, count: c0 * n), rows: c0)
            encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: c0 * k * 2, count: c1 * k),
                           output: output.slice(byteOffset: c0 * n * 2, count: c1 * n), rows: c1)
        } else if !weight.isIncumbentContainer {
            encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
        } else {
            encodeSmallTileNAX(enc, pso: psoGemmBatched, weight: weight, input: input, output: output, rows: rows)
        }
    }

    func encodeRMSNorm(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, gamma: QWTensorValue,
                       y: QWTensorValue, yRot: QWTensorValue,
                       rows: Int, mode: UInt32, epsOverride: Float? = nil) {
        enc.setComputePipelineState(psoRMSNorm)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        y.bind(enc, index: 2)
        yRot.bind(enc, index: 3)
        var eps = epsOverride ?? config.rmsNormEps, modeU = mode
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&modeU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeF16GemvBatched(_ enc: any MTLComputeCommandEncoder,
                              w: QWTensorValue, x: QWTensorValue, y: QWTensorValue,
                              n: Int, k: Int, m: Int) {
        enc.setComputePipelineState(psoF16GemvBatched)
        w.bind(enc, index: 0)
        x.bind(enc, index: 1)
        y.bind(enc, index: 2)
        var nU = UInt32(n), kU = UInt32(k), mU = UInt32(m)
        enc.setBytes(&nU, length: 4, index: 3)
        enc.setBytes(&kU, length: 4, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (n + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeConvApply(_ enc: any MTLComputeCommandEncoder,
                                 x: QWTensorValue, dyn: QWTensorValue,
                                 baseKernel: QWTensorValue, outOrResidual: QWTensorValue,
                                 rows: Int, mode: UInt32) {
        enc.setComputePipelineState(psoConvApply)
        x.bind(enc, index: 0)
        dyn.bind(enc, index: 1)
        baseKernel.bind(enc, index: 2)
        outOrResidual.bind(enc, index: 3)
        var modeU = mode
        enc.setBytes(&modeU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func encodeContextUpdate(_ cb: any MTLCommandBuffer,
                             slot: QWDFlashSessionSlot,
                             startPosition: Int,
                             totalCount: Int) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        let posOffset = startPosition + (totalCount - captureRows)
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWEngineDriverError.state("QWDFlashDrafter: failed to create context encoder")
        }
        defer { enc.endEncoding() }

        encodeQ4Matmul(enc, weight: fc, input: targetHidden, output: hCtx, rows: captureRows)
        enc.memoryBarrier(scope: .buffers)
        encodeRMSNorm(enc, x: hCtx, gamma: hiddenNorm, y: hCtx, yRot: hCtx, rows: captureRows, mode: 1)
        enc.memoryBarrier(scope: .buffers)

        var cOff = UInt32(posOffset), eps = config.rmsNormEps, maxCacheU = UInt32(config.maxCacheSize)
        for l in 0..<config.numLayers {
            let lw = layers[l]
            encodeQ4Matmul(enc, weight: lw.kvProj, input: hCtx, output: kvCtx, rows: captureRows)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(psoRopeKVCtx)
            kvCtx.bind(enc, index: 0)
            lw.kNorm.bind(enc, index: 1)
            invFreq.bind(enc, index: 2)
            slot.kLayers[l].bind(enc, index: 3)
            slot.vLayers[l].bind(enc, index: 4)
            enc.setBytes(&cOff, length: 4, index: 5)
            enc.setBytes(&eps, length: 4, index: 6)
            enc.setBytes(&maxCacheU, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: captureRows, height: config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
    }

    func propose(hal: QWHal,
                 slot: QWDFlashSessionSlot,
                 cacheOffset: Int,
                 pendingToken: Int,
                 draftCount: Int,
                 head: QWHead,
                 headStorage: QWHeadStorage,
                 draftHead: QWMTPTDraftHead?,
                 embedding: (Int, QWTensorValue) throws -> Void,
                 stochasticOwner: QWStochasticSession? = nil,
                 completed: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let D = draftCount
        let M = D + 1
        let maxDrafts = max(7, config.blockSize - 1)
        guard (1...maxDrafts).contains(D) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: invalid draftCount \(D) (expected 1...\(maxDrafts))")
        }
        let H = config.hiddenSize
        let V = config.vocabSize
        let R = config.selectorRank
        let topK = config.selectorTopK
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0

        let row0 = QWTensorValue(region: propNorm.region, offset: propNorm.offset, shape: QWTensorShape([H], .f16))
        let row1 = QWTensorValue(region: propNorm.region, offset: propNorm.offset + H * 2, shape: QWTensorShape([H], .f16))
        try embedding(pendingToken, row0)
        try embedding(config.maskTokenId, row1)
        let src16 = propNorm.region.buffer.contents()
            .advanced(by: propNorm.region.base + propNorm.offset)
            .assumingMemoryBound(to: Float16.self)
        let dst32 = propX.region.buffer.contents()
            .advanced(by: propX.region.base + propX.offset)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<H {
            dst32[i] = Float(src16[i])
            dst32[H + i] = Float(src16[H + i])
        }
        if M > 2 {
            let maskRow32 = UnsafeRawPointer(dst32.advanced(by: H))
            for m in 2..<M {
                memcpy(UnsafeMutableRawPointer(dst32.advanced(by: m * H)), maskRow32, H * 4)
            }
        }

        guard let (cb, enc) = hal.begin(.serial) else {
            throw QWEngineDriverError.state("QWDFlashDrafter: command buffer creation failed")
        }

        if config.isDSpine {
            encodeRMSNorm(enc, x: propX, gamma: layers[0].ln1, y: propNorm, yRot: propRot, rows: M, mode: 2)
        }

        var maxCacheU = UInt32(config.maxCacheSize)
        var isCausalU = UInt32(config.draftBlockCausal ? 1 : 0)

        for l in 0..<config.numLayers {
            let lw = layers[l]
            if !config.isDSpine, let acProj = lw.attnConvProj, let acBase = lw.attnConvBase, let pDyn = propDyn {
                encodeRMSNorm(enc, x: propX, gamma: lw.ln1, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: acProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propRot, rows: M, mode: 0)
            }
            encodeQ4Matmul(enc, weight: lw.qkvProj, input: propRot, output: qkvProp, rows: M)

            enc.setComputePipelineState(psoRopeQKProp)
            qkvProp.bind(enc, index: 0)
            lw.qNorm.bind(enc, index: 1)
            lw.kNorm.bind(enc, index: 2)
            invFreq.bind(enc, index: 3)
            var pOff = UInt32(cacheOffset), eps = config.rmsNormEps
            enc.setBytes(&pOff, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: M, height: config.numAttentionHeads + config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            enc.setComputePipelineState(psoAttn)
            qkvProp.bind(enc, index: 0)
            slot.kLayers[l].bind(enc, index: 1)
            slot.vLayers[l].bind(enc, index: 2)
            attnOut.bind(enc, index: 3)
            var nOff = UInt32(cacheOffset), mU = UInt32(M)
            enc.setBytes(&nOff, length: 4, index: 4)
            enc.setBytes(&mU, length: 4, index: 5)
            enc.setBytes(&maxCacheU, length: 4, index: 6)
            enc.setBytes(&isCausalU, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: config.numAttentionHeads, height: M, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            if config.isDSpine {
                encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propDown, rows: M)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propDown, yRot: propRot, rows: M, mode: 3)
            } else if let acBase = lw.attnConvBase, let mcProj = lw.mlpConvProj,
                      let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propNorm, rows: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propX, rows: M, mode: 1)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: mcProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: mcBase, outOrResidual: propRot, rows: M, mode: 0)
            }

            encodeGateUpMatmul(enc, weight: lw.gateUpProj, input: propRot, output: mlpInter, rows: M)
            encodeQ4Matmul(enc, weight: lw.downProj, input: mlpInter, output: propDown, rows: M)
            if let bw = dspineBridge, let bs = dspineScratch {
                encodeDSpineBridgeLayer(enc, layer: l, anchorToken: pendingToken, rows: M, bw: bw, bs: bs)
            } else if let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                encodeConvApply(enc, x: propDown, dyn: pDyn, baseKernel: mcBase, outOrResidual: propX, rows: M, mode: 1)
            }
        }

        let propXDraftSlice = dspineScratch?.propXPost4
            ?? QWTensorValue(region: propX.region, offset: propX.offset + H * 4, shape: QWTensorShape([D * H], .f32))
        let propNormSlice = QWTensorValue(region: propNorm.region, offset: propNorm.offset,
                                          shape: QWTensorShape([D * H], .f16))
        let propRotSlice = QWTensorValue(region: propRot.region, offset: propRot.offset,
                                         shape: QWTensorShape([D * H], .f16))
        let logitsSlice = QWTensorValue(region: draftLogits.region, offset: draftLogits.offset,
                                        shape: QWTensorShape([D * V], .f32))

        encodeRMSNorm(enc, x: propXDraftSlice, gamma: finalNorm, y: propNormSlice, yRot: propRotSlice, rows: D, mode: 2)
        if !config.isDSpine, let selProj = selectorHiddenProj, let hProjOut = hiddenProjOut {
            encodeF16GemvBatched(enc, w: selProj, x: propNormSlice, y: hProjOut, n: R, k: H, m: D)
        }

        if let draftHead {
            if D == 1 {
                try draftHead.encode(enc, hal: hal, x: propRotSlice, y: logitsSlice)
            } else {
                try draftHead.encodeBatched(enc, hal: hal, x: propRotSlice, y: logitsSlice, m: D)
            }
        } else {
            switch headStorage {
            case .q4g(let hw):
                if isMetal4TensorCapable && D >= 9 && hw.view.n % 64 == 0 && hw.view.k % 64 == 0 {
                    try QWGemm.encodeNaxF32(
                        enc, hal: hal, weight: hw.view,
                        x16: propRotSlice, y16: propRotSlice, y32: logitsSlice, rows: D
                    )
                } else {
                    try head.encode(enc, storage: headStorage, x: propRotSlice, y: logitsSlice, M: D)
                }
            case .f16:
                try head.encode(enc, storage: headStorage, x: propNormSlice, y: logitsSlice, M: D)
            }
        }

        var vocabU = UInt32(V), topKU = UInt32(topK)
        enc.setComputePipelineState(psoTop16Stage1)
        logitsSlice.bind(enc, index: 0)
        top16PartialsVal.bind(enc, index: 1)
        top16PartialsId.bind(enc, index: 2)
        enc.setBytes(&vocabU, length: 4, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 256, height: D, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoTop16Stage2)
        top16PartialsVal.bind(enc, index: 0)
        top16PartialsId.bind(enc, index: 1)
        unaryOut.bind(enc, index: 2)
        candidatesOut.bind(enc, index: 3)
        enc.setBytes(&topKU, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        if let bw = dspineBridge, let bs = dspineScratch {
            encodeDSpineRefinementTail(
                enc, draftCount: D, topK: topK, headStorage: headStorage,
                bw: bw, bs: bs, stochastic: stochasticOwner != nil
            )
        } else if stochasticOwner == nil,
                  let hProjOut = hiddenProjOut,
                  let pBook = predecessorCodebook,
                  let sBook = successorCodebook {
            var anchorI = Int32(pendingToken), draftCountU = UInt32(D)
            enc.setComputePipelineState(psoSelectorGreedy)
            hProjOut.bind(enc, index: 0)
            unaryOut.bind(enc, index: 1)
            candidatesOut.bind(enc, index: 2)
            pBook.bind(enc, index: 3)
            sBook.bind(enc, index: 4)
            draftTokensOut.bind(enc, index: 5)
            enc.setBytes(&anchorI, length: 4, index: 6)
            enc.setBytes(&draftCountU, length: 4, index: 7)
            enc.setBytes(&topKU, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        hal.end(cb, enc)
        cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else {
            throw QWEngineDriverError.state("QWDFlashDrafter GPU failure: \(String(describing: cb.error))")
        }
        try completed?(cb)

        if let owner = stochasticOwner {
            if let bs = dspineScratch {
                return try selectDSpineStochastic(owner: owner, draftCount: D, topK: topK, bs: bs)
            }
            return try selectStochastic(owner: owner, anchorToken: pendingToken, draftCount: D, topK: topK)
        } else {
            let ptr = draftTokensOut.region.buffer.contents()
                .advanced(by: draftTokensOut.region.base + draftTokensOut.offset)
                .assumingMemoryBound(to: Int32.self)
            let drafts = (0..<D).map { Int(ptr[$0]) }
            return (drafts, [])
        }
    }

    private func selectStochastic(owner: QWStochasticSession,
                                  anchorToken: Int,
                                  draftCount: Int,
                                  topK: Int) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        guard let hiddenProjOut, let predecessorCodebook, let successorCodebook else {
            throw QWEngineDriverError.state("QWDFlashDrafter: missing CandidateSelector buffers")
        }
        let R = config.selectorRank
        let V = config.vocabSize
        let temp = max(Double(owner.config.temperature), 1e-6)
        let hPtr = hiddenProjOut.region.buffer.contents()
            .advanced(by: hiddenProjOut.region.base + hiddenProjOut.offset)
            .assumingMemoryBound(to: Float.self)
        let uPtr = unaryOut.region.buffer.contents()
            .advanced(by: unaryOut.region.base + unaryOut.offset)
            .assumingMemoryBound(to: Float.self)
        let cPtr = candidatesOut.region.buffer.contents()
            .advanced(by: candidatesOut.region.base + candidatesOut.offset)
            .assumingMemoryBound(to: Int32.self)
        let pBook = predecessorCodebook.region.buffer.contents()
            .advanced(by: predecessorCodebook.region.base + predecessorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)
        let sBook = successorCodebook.region.buffer.contents()
            .advanced(by: successorCodebook.region.base + successorCodebook.offset)
            .assumingMemoryBound(to: UInt16.self)

        func bf16(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }

        var pred = anchorToken
        var drafts: [Int] = []
        var proposals: [QWMTPGumbelProposal] = []
        for pos in 0..<draftCount {
            var scores = [Double](repeating: 0, count: topK)
            var ids = [Int](repeating: 0, count: topK)
            let hBase = pos * R
            let pBase = pred * R
            var maxScore = -Double.infinity
            for k in 0..<topK {
                let cand = Int(cPtr[pos * 16 + k])
                ids[k] = cand
                let sBase = cand * R
                var edge: Float = 0
                for r in 0..<R {
                    edge += bf16(pBook[pBase + r]) * hPtr[hBase + r] * bf16(sBook[sBase + r])
                }
                let s = Double(uPtr[pos * 16 + k] + edge)
                scores[k] = s
                if s > maxScore { maxScore = s }
            }
            let weights = scores.map { exp(($0 - maxScore) / temp) }
            let dist = try QWSamplingDistribution.sparse(vocabulary: V, ids: ids, weights: weights)
            let proposal = try owner.stream.sampleGumbelProposal(dist)
            pred = proposal.token
            drafts.append(proposal.token)
            proposals.append(proposal)
        }
        return (drafts, proposals)
    }
}
