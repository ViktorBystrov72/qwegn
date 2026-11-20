// QWDFlashDrafter+Propose.swift — DFlash 2 / DSpine block-diffusion
// context update, draft proposal, and candidate selection.

import Darwin
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHead
import QWHAL
import QWLoad
import QWQuant

extension QWDFlashDrafter {
    public func encodeCaptureTargetHidden(
        _ enc: any MTLComputeCommandEncoder,
        source: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int,
        slot: Int
    ) {
        guard rows > 0, (0..<config.targetLayerIds.count).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureTarget)
        source.bind(enc, index: 0)
        targetHidden.bind(enc, index: 1)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        var hiddenU4 = UInt32(config.hiddenSize / 8)
        var totalSlots = UInt32(config.targetLayerIds.count)
        enc.setBytes(&sOff, length: 4, index: 2)
        enc.setBytes(&dOff, length: 4, index: 3)
        enc.setBytes(&slotU, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&totalSlots, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeCaptureHCTargetHidden(
        _ enc: any MTLComputeCommandEncoder,
        base: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int,
        slot: Int,
        hc: Int,
        deferredInject: (src: QWTensorValue, inj: QWTensorValue, isMHC: Bool)? = nil
    ) {
        guard rows > 0, hc > 0, (0..<config.targetLayerIds.count).contains(slot) else { return }
        enc.setComputePipelineState(psoCaptureHC)
        base.bind(enc, index: 0)
        if let inj = deferredInject {
            inj.src.bind(enc, index: 1)
            inj.inj.bind(enc, index: 2)
        } else {
            base.bind(enc, index: 1)
            base.bind(enc, index: 2)
        }
        targetHidden.bind(enc, index: 3)
        var sOff = UInt32(srcRowOffset), dOff = UInt32(dstRowOffset), slotU = UInt32(slot)
        var hiddenU4 = UInt32(config.hiddenSize / 8)
        var totalSlots = UInt32(config.targetLayerIds.count)
        var hcU = UInt32(hc)
        var flagsU = UInt32(deferredInject.map { $0.isMHC ? 2 : 1 } ?? 0)
        enc.setBytes(&sOff, length: 4, index: 4)
        enc.setBytes(&dOff, length: 4, index: 5)
        enc.setBytes(&slotU, length: 4, index: 6)
        enc.setBytes(&hiddenU4, length: 4, index: 7)
        enc.setBytes(&totalSlots, length: 4, index: 8)
        enc.setBytes(&hcU, length: 4, index: 9)
        enc.setBytes(&flagsU, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeStagePropBF16(
        _ enc: any MTLComputeCommandEncoder,
        table: QWTensorValue,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) {
        encodeStagePropBF16(
            enc, tableBuffer: table.region.buffer, byteOffset: table.region.base + table.offset,
            propX: propX, anchorToken: anchorToken, maskToken: maskToken, rows: rows
        )
    }

    public func encodeStagePropBF16(
        _ enc: any MTLComputeCommandEncoder,
        tableBuffer: any MTLBuffer,
        byteOffset: Int,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) {
        guard rows > 0 else { return }
        let safeAnchor = Self.clampToken(anchorToken, vocab: config.vocabSize)
        let safeMask = Self.clampToken(maskToken, vocab: config.vocabSize)
        enc.setComputePipelineState(psoStagePropBF16)
        enc.setBuffer(tableBuffer, offset: byteOffset, index: 0)
        propX.bind(enc, index: 1)
        var anchorI = Int32(safeAnchor), maskI = Int32(safeMask)
        var hiddenU = UInt32(config.hiddenSize), vocabU = UInt32(config.vocabSize)
        enc.setBytes(&anchorI, length: 4, index: 2)
        enc.setBytes(&maskI, length: 4, index: 3)
        enc.setBytes(&hiddenU, length: 4, index: 4)
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (config.hiddenSize + 255) / 256, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func encodeStageProp(
        _ enc: any MTLComputeCommandEncoder,
        embedTable: QWDeviceEmbedTable,
        propX: QWTensorValue,
        anchorToken: Int,
        maskToken: Int,
        rows: Int
    ) throws {
        guard rows > 0 else { return }
        let safeAnchor = Self.clampToken(anchorToken, vocab: config.vocabSize)
        let safeMask = Self.clampToken(maskToken, vocab: config.vocabSize)
        switch embedTable.storage {
        case .bf16(let table):
            encodeStagePropBF16(enc, table: table, propX: propX,
                                anchorToken: safeAnchor, maskToken: safeMask, rows: rows)
        case .nativeAffine(let blocks, let bits, let group):
            let tokPtr = propTokens.region.buffer.contents()
                .advanced(by: propTokens.region.base + propTokens.offset)
                .assumingMemoryBound(to: Int32.self)
            tokPtr[0] = Int32(safeAnchor)
            for r in 1..<rows { tokPtr[r] = Int32(safeMask) }
            try QWGemmAQ.encodeEmbedAQ(
                enc, hal: hal, blocks: blocks,
                token: propTokens.slice(byteOffset: 0, count: rows),
                out: propX, hidden: config.hiddenSize, hc: 1, bits: bits, group: group, rows: rows
            )
        }
    }

    func encodeQ4Matmul(_ enc: any MTLComputeCommandEncoder,
                        weight: QWWeightView,
                        input: QWTensorValue,
                        output: QWTensorValue,
                        rows: Int) throws {
        if !Self.usesQ4GMKernel(weight) {
            try encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
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
            let rowPaired = Self.usesRowPairs(weight, activeRows: rows)
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
            try encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: 0, count: c0 * k),
                               output: output.slice(byteOffset: 0, count: c0 * n), rows: c0)
            try encodeQ4Matmul(enc, weight: weight, input: input.slice(byteOffset: c0 * k * 2, count: c1 * k),
                               output: output.slice(byteOffset: c0 * n * 2, count: c1 * n), rows: c1)
        } else if !weight.isIncumbentContainer {
            try encodeAQMatmul(enc, weight: weight, input: input, output: output, rows: rows)
        } else {
            encodeSmallTileNAX(enc, pso: psoGemmBatched, weight: weight, input: input, output: output, rows: rows)
        }
    }

    func encodeRMSNorm(_ enc: any MTLComputeCommandEncoder,
                       x: QWTensorValue, gamma: QWTensorValue,
                       y: QWTensorValue, yRot: QWTensorValue,
                       rows: Int, mode: UInt32, epsOverride: Float? = nil,
                       yRot32: QWTensorValue? = nil) {
        enc.setComputePipelineState(psoRMSNorm)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        y.bind(enc, index: 2)
        yRot.bind(enc, index: 3)
        var eps = epsOverride ?? config.rmsNormEps, modeU = mode, hiddenU = UInt32(config.hiddenSize)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&modeU, length: 4, index: 5)
        enc.setBytes(&hiddenU, length: 4, index: 6)
        (yRot32 ?? propRot32).bind(enc, index: 7)
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
        var hiddenU4 = UInt32(config.hiddenSize / 8)
        var convGroupsU = UInt32(config.hiddenSize / max(1, config.convGroupSize))
        enc.setBytes(&modeU, length: 4, index: 4)
        enc.setBytes(&hiddenU4, length: 4, index: 5)
        enc.setBytes(&convGroupsU, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func encodeContextUpdate(_ cb: any MTLCommandBuffer,
                                    slot: QWDFlashSessionSlot,
                                    startPosition: Int,
                                    totalCount: Int) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw QWDFlashError.state("QWDFlashDrafter: failed to create context encoder")
        }
        defer { enc.endEncoding() }
        try encodeContextUpdate(enc, slot: slot, startPosition: startPosition, totalCount: totalCount)
    }

    public func encodeContextUpdate(_ enc: any MTLComputeCommandEncoder,
                                    slot: QWDFlashSessionSlot,
                                    startPosition: Int,
                                    totalCount: Int) throws {
        let captureRows = min(totalCount, config.maxCacheSize)
        guard captureRows > 0 else { return }
        let posOffset = startPosition + (totalCount - captureRows)

        try encodeQ4Matmul(enc, weight: fc, input: targetHidden, output: hCtx, rows: captureRows)
        enc.memoryBarrier(scope: .buffers)
        encodeRMSNorm(enc, x: hCtx, gamma: hiddenNorm, y: hCtx, yRot: hCtx, rows: captureRows, mode: 1)
        enc.memoryBarrier(scope: .buffers)

        var cOff = UInt32(posOffset), eps = config.rmsNormEps
        var maxCacheU = UInt32(config.maxCacheSize), kvHeadsU = UInt32(config.numKeyValueHeads)
        for l in 0..<config.numLayers {
            let lw = layers[l]
            try encodeQ4Matmul(enc, weight: lw.kvProj, input: hCtx, output: kvCtx, rows: captureRows)
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
            enc.setBytes(&kvHeadsU, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: captureRows, height: config.numKeyValueHeads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }
    }

    public func enqueueGreedyPropose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int? = nil,
        headStorage: QWHeadStorage? = nil,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: (_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void,
        journalBuffer: any MTLBuffer,
        journalByteOffset: Int
    ) throws -> any MTLCommandBuffer {
        guard journalByteOffset >= 0, journalByteOffset % 4 == 0,
              journalByteOffset + draftCount * 4 <= journalBuffer.length,
              journalBuffer !== draftTokensOut.region.buffer else {
            throw QWDFlashError.state("QWDFlashDrafter: invalid async journal destination")
        }
        return try withoutActuallyEscaping(encodeAnchorMaskEmbed) { anchorEmbed in
            try encodeAndCommitPropose(
                hal: hal, slot: slot, cacheOffset: cacheOffset, pendingToken: pendingToken,
                draftCount: draftCount, draftVocab: draftVocab, headStorage: headStorage,
                encodeHeadLogits: encodeHeadLogits, encodeAnchorMaskEmbed: anchorEmbed,
                embedding: nil, stochastic: nil, preContextUpdate: nil,
                journalCopy: (journalBuffer, journalByteOffset)
            )
        }
    }

    public func propose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int? = nil,
        headStorage: QWHeadStorage? = nil,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: ((_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                 _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void)? = nil,
        embedding: ((Int, QWTensorValue) throws -> Void)? = nil,
        stochastic: (temperature: Float, stream: QWMTPRandomStream)? = nil,
        preContextUpdate: (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)? = nil,
        completed: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        let cb = try encodeAndCommitPropose(
            hal: hal, slot: slot, cacheOffset: cacheOffset, pendingToken: pendingToken,
            draftCount: draftCount, draftVocab: draftVocab, headStorage: headStorage,
            encodeHeadLogits: encodeHeadLogits, encodeAnchorMaskEmbed: encodeAnchorMaskEmbed,
            embedding: embedding, stochastic: stochastic, preContextUpdate: preContextUpdate,
            journalCopy: nil
        )
        cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else {
            throw QWDFlashError.state("QWDFlashDrafter GPU failure: \(String(describing: cb.error))")
        }
        try completed?(cb)

        let D = draftCount
        let V = config.vocabSize
        let topK = config.selectorTopK
        let safeAnchor = Self.clampToken(pendingToken, vocab: V)
        if let stochastic {
            if let bs = dspineScratch {
                return try selectDSpineStochastic(
                    temperature: stochastic.temperature, stream: stochastic.stream, draftCount: D, topK: topK, bs: bs
                )
            }
            return try selectStochastic(
                temperature: stochastic.temperature, stream: stochastic.stream, anchorToken: safeAnchor, draftCount: D, topK: topK
            )
        } else {
            let ptr = draftTokensOut.region.buffer.contents()
                .advanced(by: draftTokensOut.region.base + draftTokensOut.offset)
                .assumingMemoryBound(to: Int32.self)
            let drafts = (0..<D).map { Self.clampToken(Int(ptr[$0]), vocab: V) }
            return (drafts, [])
        }
    }

    private func encodeAndCommitPropose(
        hal: QWHal,
        slot: QWDFlashSessionSlot,
        cacheOffset: Int,
        pendingToken: Int,
        draftCount: Int,
        draftVocab: Int?,
        headStorage: QWHeadStorage?,
        encodeHeadLogits: (_ enc: any MTLComputeCommandEncoder, _ xRot16: QWTensorValue, _ xNorm16: QWTensorValue,
                           _ xRot32: QWTensorValue?, _ yLogits32: QWTensorValue, _ rows: Int, _ vocabRows: Int) throws -> Void,
        encodeAnchorMaskEmbed: ((_ enc: any MTLComputeCommandEncoder, _ propX32: QWTensorValue,
                                 _ anchorToken: Int, _ maskToken: Int, _ rows: Int) throws -> Void)?,
        embedding: ((Int, QWTensorValue) throws -> Void)?,
        stochastic: (temperature: Float, stream: QWMTPRandomStream)?,
        preContextUpdate: (slot: QWDFlashSessionSlot, startPosition: Int, totalCount: Int)?,
        journalCopy: (buffer: any MTLBuffer, byteOffset: Int)?
    ) throws -> any MTLCommandBuffer {
        let D = draftCount
        let M = D + 1
        let maxDrafts = max(7, config.blockSize - 1)
        guard (1...maxDrafts).contains(D) else {
            throw QWDFlashError.state("QWDFlashDrafter: invalid draftCount \(D) (expected 1...\(maxDrafts))")
        }
        let H = config.hiddenSize
        let V = config.vocabSize
        let vocabRows: Int
        if let requestedVocab = draftVocab {
            let clamped = min(V, max(1, requestedVocab))
            vocabRows = clamped < V ? min(V, ((clamped + 255) / 256) * 256) : V
        } else {
            vocabRows = V
        }
        guard vocabRows <= 393_216 else {
            throw QWDFlashError.state("QWDFlashDrafter: vocabRows \(vocabRows) exceeds 393216 limit")
        }
        let R = config.selectorRank
        let topK = config.selectorTopK
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0
        let safeAnchor = Self.clampToken(pendingToken, vocab: V)
        let safeMask = Self.clampToken(config.maskTokenId, vocab: V)

        if encodeAnchorMaskEmbed == nil {
            guard let embedding else {
                throw QWDFlashError.state("QWDFlashDrafter: missing anchor/mask embedding provider")
            }
            let row0 = QWTensorValue(region: propNorm.region, offset: propNorm.offset, shape: QWTensorShape([H], .f16))
            let row1 = QWTensorValue(region: propNorm.region, offset: propNorm.offset + H * 2, shape: QWTensorShape([H], .f16))
            try embedding(safeAnchor, row0)
            try embedding(safeMask, row1)
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
        }

        guard let (cb, enc) = hal.beginUnretained(.serial) else {
            throw QWDFlashError.state("QWDFlashDrafter: command buffer creation failed")
        }
        var encoderEnded = false
        defer {
            if !encoderEnded {
                enc.endEncoding()
            }
        }

        if let pre = preContextUpdate, min(pre.totalCount, config.maxCacheSize) > 0 {
            try encodeContextUpdate(enc, slot: pre.slot, startPosition: pre.startPosition, totalCount: pre.totalCount)
            enc.memoryBarrier(scope: .buffers)
        }
        if let encodeAnchorMaskEmbed {
            let propXSlice = QWTensorValue(region: propX.region, offset: propX.offset, shape: QWTensorShape([M * H], .f32))
            try encodeAnchorMaskEmbed(enc, propXSlice, safeAnchor, safeMask, M)
            enc.memoryBarrier(scope: .buffers)
        }

        if config.isDSpine {
            encodeRMSNorm(enc, x: propX, gamma: layers[0].ln1, y: propNorm, yRot: propRot, rows: M, mode: 2)
        }

        var maxCacheU = UInt32(config.maxCacheSize)
        var isCausalU = UInt32(config.draftBlockCausal ? 1 : 0)
        var qHeadsU = UInt32(config.numAttentionHeads)
        var kvHeadsU = UInt32(config.numKeyValueHeads)

        for l in 0..<config.numLayers {
            let lw = layers[l]
            if !config.isDSpine, let acProj = lw.attnConvProj, let acBase = lw.attnConvBase, let pDyn = propDyn {
                encodeRMSNorm(enc, x: propX, gamma: lw.ln1, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: acProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propRot, rows: M, mode: 0)
            }
            try encodeQ4Matmul(enc, weight: lw.qkvProj, input: propRot, output: qkvProp, rows: M)

            enc.setComputePipelineState(psoRopeQKProp)
            qkvProp.bind(enc, index: 0)
            lw.qNorm.bind(enc, index: 1)
            lw.kNorm.bind(enc, index: 2)
            invFreq.bind(enc, index: 3)
            var pOff = UInt32(cacheOffset), eps = config.rmsNormEps
            enc.setBytes(&pOff, length: 4, index: 4)
            enc.setBytes(&eps, length: 4, index: 5)
            enc.setBytes(&qHeadsU, length: 4, index: 6)
            enc.setBytes(&kvHeadsU, length: 4, index: 7)
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
            enc.setBytes(&qHeadsU, length: 4, index: 8)
            enc.setBytes(&kvHeadsU, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: config.numAttentionHeads, height: M, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            if config.isDSpine {
                try encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propDown, rows: M)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propDown, yRot: propRot, rows: M, mode: 3)
            } else if let acBase = lw.attnConvBase, let mcProj = lw.mlpConvProj,
                      let mcBase = lw.mlpConvBase, let pDyn = propDyn {
                try encodeQ4Matmul(enc, weight: lw.oProj, input: attnOut, output: propNorm, rows: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: acBase, outOrResidual: propX, rows: M, mode: 1)
                encodeRMSNorm(enc, x: propX, gamma: lw.ln2, y: propNorm, yRot: propNorm, rows: M, mode: 0)
                encodeF16GemvBatched(enc, w: mcProj, x: propNorm, y: pDyn, n: convProjDim, k: H, m: M)
                encodeConvApply(enc, x: propNorm, dyn: pDyn, baseKernel: mcBase, outOrResidual: propRot, rows: M, mode: 0)
            }

            try encodeGateUpMatmul(enc, weight: lw.gateUpProj, input: propRot, output: mlpInter, rows: M)
            try encodeQ4Matmul(enc, weight: lw.downProj, input: mlpInter, output: propDown, rows: M)
            if let bw = dspineBridge, let bs = dspineScratch {
                encodeDSpineBridgeLayer(enc, layer: l, anchorToken: safeAnchor, rows: M, bw: bw, bs: bs)
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
        let propRot32Slice = QWTensorValue(region: propRot32.region, offset: propRot32.offset,
                                           shape: QWTensorShape([D * H], .f32))
        let logitsSlice = QWTensorValue(region: draftLogits.region, offset: draftLogits.offset,
                                        shape: QWTensorShape([D * vocabRows], .f32))

        encodeRMSNorm(enc, x: propXDraftSlice, gamma: finalNorm,
                      y: propNormSlice, yRot: propRotSlice, rows: D, mode: 5, yRot32: propRot32Slice)
        if !config.isDSpine, let selProj = selectorHiddenProj, let hProjOut = hiddenProjOut {
            encodeF16GemvBatched(enc, w: selProj, x: propNormSlice, y: hProjOut, n: R, k: H, m: D)
        }

        try encodeHeadLogits(enc, propRotSlice, propNormSlice, propRot32Slice, logitsSlice, D, vocabRows)

        var vocabU = UInt32(vocabRows), topKU = UInt32(topK)
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
        enc.setBytes(&vocabU, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: D, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        if let bw = dspineBridge, let bs = dspineScratch {
            guard let headStorage else {
                throw QWDFlashError.state("QWDFlashDrafter: DSpine refinement requires headStorage")
            }
            encodeDSpineRefinementTail(
                enc, draftCount: D, topK: topK, headStorage: headStorage,
                bw: bw, bs: bs, stochastic: stochastic != nil
            )
        } else if stochastic == nil,
                  let hProjOut = hiddenProjOut,
                  let pBook = predecessorCodebook,
                  let sBook = successorCodebook {
            var anchorI = Int32(safeAnchor), draftCountU = UInt32(D), selectorVocabU = UInt32(V)
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
            enc.setBytes(&selectorVocabU, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        encoderEnded = true
        enc.endEncoding()
        if let journalCopy {
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw QWDFlashError.state("QWDFlashDrafter: journal copy encoder failed")
            }
            blit.copy(
                from: draftTokensOut.region.buffer,
                sourceOffset: draftTokensOut.region.base + draftTokensOut.offset,
                to: journalCopy.buffer,
                destinationOffset: journalCopy.byteOffset,
                size: D * 4
            )
            blit.endEncoding()
        }
        cb.commit()
        return cb
    }

    static func clampToken(_ token: Int, vocab: Int) -> Int {
        (0..<vocab).contains(token) ? token : 0
    }

    static func buildSparseDistribution(
        vocab: Int,
        candidates: [(id: Int, score: Double)],
        temperature: Float
    ) throws -> QWSamplingDistribution {
        let temp = max(Double(temperature), 1e-6)
        let valid = candidates.compactMap { entry -> (id: Int, score: Double)? in
            guard (0..<vocab).contains(entry.id), entry.id != 0x7fff_ffff else { return nil }
            let score = entry.score.isFinite ? max(entry.score, -1e30) : -1e30
            return (entry.id, score)
        }
        guard let maxScore = valid.map(\.score).max() else {
            return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: [0], weights: [1.0])
        }
        var ids: [Int] = []
        var weights: [Double] = []
        for entry in valid {
            let w = exp((entry.score - maxScore) / temp)
            if w.isFinite && w > 0 {
                ids.append(entry.id)
                weights.append(w)
            }
        }
        if ids.isEmpty {
            ids = [valid[0].id]
            weights = [1.0]
        }
        return try QWSamplingDistribution.sparse(vocabulary: vocab, ids: ids, weights: weights)
    }

    private func selectStochastic(temperature: Float,
                                  stream: QWMTPRandomStream,
                                  anchorToken: Int,
                                  draftCount: Int,
                                  topK: Int) throws -> (drafts: [Int], proposals: [QWMTPGumbelProposal]) {
        guard let hiddenProjOut, let predecessorCodebook, let successorCodebook else {
            throw QWDFlashError.state("QWDFlashDrafter: missing CandidateSelector buffers")
        }
        let R = config.selectorRank
        let V = config.vocabSize
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

        var pred = Self.clampToken(anchorToken, vocab: V)
        var drafts: [Int] = []
        var proposals: [QWMTPGumbelProposal] = []
        for pos in 0..<draftCount {
            var scored: [(id: Int, score: Double)] = []
            scored.reserveCapacity(topK)
            let hBase = pos * R
            let pBase = pred * R
            for k in 0..<topK {
                let rawCand = Int(cPtr[pos * 16 + k])
                guard (0..<V).contains(rawCand), rawCand != 0x7fff_ffff else { continue }
                let sBase = rawCand * R
                var edge: Float = 0
                for r in 0..<R {
                    edge += bf16(pBook[pBase + r]) * hPtr[hBase + r] * bf16(sBook[sBase + r])
                }
                let rawScore = Double(uPtr[pos * 16 + k]) + Double(edge)
                scored.append((rawCand, rawScore))
            }
            let dist = try Self.buildSparseDistribution(vocab: V, candidates: scored, temperature: temperature)
            let proposal = try stream.sampleGumbelProposal(dist)
            let tok = Self.clampToken(proposal.token, vocab: V)
            pred = tok
            drafts.append(tok)
            proposals.append(proposal)
        }
        return (drafts, proposals)
    }
}
