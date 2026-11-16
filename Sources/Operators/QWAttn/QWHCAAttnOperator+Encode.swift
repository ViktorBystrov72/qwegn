// QWHCAAttnOperator+Encode.swift — HCA compression, attention, VHA, and grouped WoA passes.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

extension QWHCAAttnOperator {
    public func encodeInPrepAndCompress(_ enc: any MTLComputeCommandEncoder,
                                        hcaW: LayerWeights,
                                        hcaLayer: Int,
                                        pos0: Int,
                                        rows: Int) {
        guard hcaLayer >= 0, hcaLayer < layers.count, rows > 0 else { return }
        let ls = layers[hcaLayer]
        var inP = InPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qLoraRank: UInt32(qLoraRank),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            compressRatio: UInt32(compressRatio),
            ringMask: UInt32(ringCap - 1),
            rmsEps: geometry.rmsEps,
            mscale: ropeMscale
        )
        enc.setComputePipelineState(psoInPrep)
        fusedInScratch.bind(enc, index: 0)
        hcaW.qNormW.bind(enc, index: 1)
        hcaW.kvNormW.bind(enc, index: 2)
        hcaW.ape.bind(enc, index: 3)
        invFreqBuf.bind(enc, index: 4)
        qrScratch.bind(enc, index: 5)
        ls.swaKV.bind(enc, index: 6)
        ls.compKvPend.bind(enc, index: 7)
        ls.compScorePend.bind(enc, index: 8)
        enc.setBytes(&inP, length: MemoryLayout<InPrepParams>.stride, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let startBlock = pos0 / compressRatio
        let endBlock = (pos0 + rows) / compressRatio
        let numBlocks = endBlock - startBlock
        let hasLsa = (lsaAllocated && hcaW.lsaWKIdx != nil && ls.pooledKeys != nil)
        if numBlocks > 0 {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            var cP = CompressParams(
                startBlock: UInt32(startBlock),
                numBlocks: UInt32(numBlocks),
                compressRatio: UInt32(compressRatio),
                headDim: UInt32(headDim),
                nopeDim: UInt32(nopeDim),
                ropeDim: UInt32(ropeDim),
                ringMask: UInt32(ringCap - 1),
                indexDim: UInt32(indexDim),
                rmsEps: geometry.rmsEps,
                mscale: ropeMscale,
                hasLSA: hasLsa ? 1 : 0
            )
            enc.setComputePipelineState(psoCompressBlocks)
            ls.compKvPend.bind(enc, index: 0)
            ls.compScorePend.bind(enc, index: 1)
            hcaW.compNormW.bind(enc, index: 2)
            invFreqBuf.bind(enc, index: 3)
            (hcaW.lsaWKIdx ?? ls.compKV).bind(enc, index: 4)
            ls.compKV.bind(enc, index: 5)
            (ls.pooledKeys ?? ls.compKV).bind(enc, index: 6)
            enc.setBytes(&cP, length: MemoryLayout<CompressParams>.stride, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if hasLsa {
            updateCoarseClusters(enc, ls: ls, hcaLayer: hcaLayer, pos0: pos0, rows: rows)
        }
    }

    public func encodeQPrepAttendVHAAndWoA(_ enc: any MTLComputeCommandEncoder,
                                           hcaW: LayerWeights,
                                           zGatedOut: QWTensorValue,
                                           hcaLayer: Int,
                                           pos0: Int,
                                           rows: Int,
                                           skipExitRotation: Bool = false,
                                           stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        guard hcaLayer >= 0, hcaLayer < layers.count, rows > 0 else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[hcaLayer]

        mark("hca.q_prep")
        var qP = QPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            qUpStride: UInt32(qUpStride),
            indexDim: UInt32(indexDim),
            rmsEps: geometry.rmsEps,
            mscale: ropeMscale
        )
        enc.setComputePipelineState(psoQPrep)
        qUpGateScratch.bind(enc, index: 0)
        invFreqBuf.bind(enc, index: 1)
        enc.setBytes(&qP, length: MemoryLayout<QPrepParams>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: qHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let maxTotalComp = (pos0 + rows) / compressRatio
        let useSparseLsa = lsaAllocated
            && maxTotalComp > compBudget
            && hcaW.lsaWQIdx != nil
            && hcaW.lsaWKIdx != nil
            && ls.pooledKeys != nil
        if useSparseLsa {
            mark("hca.lsa_select")
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            encodeLSASelection(enc, ls: ls, wqIdx: hcaW.lsaWQIdx!, qPrepParams: &qP, pos0: pos0, rows: rows)
        }

        mark("hca.attn")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        let sinkTensor = hcaW.attnSink ?? zeroSinkBuf
        var attP = AttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            qUpStride: UInt32(qUpStride),
            compressRatio: UInt32(compressRatio),
            slidingWindow: UInt32(slidingWindow),
            ringMask: UInt32(ringCap - 1),
            compBudget: UInt32(compBudget),
            listStride: UInt32(listStride),
            useBlockList: useSparseLsa ? 1 : 0,
            hasSink: hcaW.attnSink != nil ? 1 : 0
        )
        let minTotalComp = (pos0 + 1) / compressRatio
        let approxSlots = min(useSparseLsa ? compBudget : minTotalComp, minTotalComp)
            + min(pos0 + 1, slidingWindow)
        let useSplit = attnSplits > 1 && rows <= Self.splitMaxRows && approxSlots >= 64
        let nSplit = useSplit ? attnSplits : 1
        let useNAX = psoAttnNAX != nil && headDim == 512 && qHeads % 16 == 0
        let useNAXPrefill = useNAX && !useSplit && rows >= 16 && psoAttnNAXPrefill != nil
        if let naxPrefillPSO = psoAttnNAXPrefill, useNAXPrefill {
            enc.setComputePipelineState(naxPrefillPSO)
        } else if let naxPSO = psoAttnNAX, useNAX {
            enc.setComputePipelineState(naxPSO)
        } else {
            enc.setComputePipelineState(psoAttnMMA)
        }
        qUpGateScratch.bind(enc, index: 0)
        ls.compKV.bind(enc, index: 1)
        ls.swaKV.bind(enc, index: 2)
        (blockListScratch ?? zeroSinkBuf).bind(enc, index: 3)
        (blockCountScratch ?? zeroSinkBuf).bind(enc, index: 4)
        sinkTensor.bind(enc, index: 5)
        (useSplit ? partOScratch : attScratch).bind(enc, index: 6)
        partMLScratch.bind(enc, index: 7)
        enc.setBytes(&attP, length: MemoryLayout<AttnParams>.stride, index: 8)
        let hgPerTG = useNAX ? 32 : 16
        let tgThreads = useNAX ? 256 : 128
        let gridSize = useNAXPrefill
            ? MTLSize(width: (qHeads + hgPerTG - 1) / hgPerTG, height: rows, depth: 1)
            : MTLSize(width: rows, height: (qHeads + hgPerTG - 1) / hgPerTG, depth: nSplit)
        enc.dispatchThreadgroups(gridSize,
                                 threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        if useSplit {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            var nSplitU = UInt32(nSplit)
            enc.setComputePipelineState(psoAttnCombine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            sinkTensor.bind(enc, index: 2)
            attScratch.bind(enc, index: 3)
            enc.setBytes(&attP, length: MemoryLayout<AttnParams>.stride, index: 4)
            enc.setBytes(&nSplitU, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        mark("hca.vha_invrope")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        let isRawWoA = (hcaW.woA == nil && hcaW.woARawF16 != nil)
        let skipInterFwht = isRawWoA || (hcaW.woA != nil && !hcaW.woA!.isRotated)
        var vhaP = VhaInvRopeParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim),
            nopeDim: UInt32(nopeDim),
            ropeDim: UInt32(ropeDim),
            invMscale: ropeMscale > 0 ? (1.0 / ropeMscale) : 1.0,
            hasVHA: (hcaW.vhaMix != nil ? 1 : 0) | (skipInterFwht ? 0x10000 : 0)
        )
        enc.setComputePipelineState(psoVhaInvRope)
        attScratch.bind(enc, index: 0)
        (hcaW.vhaMix ?? attScratch).bind(enc, index: 1)
        invFreqBuf.bind(enc, index: 2)
        zVhaScratch.bind(enc, index: 3)
        enc.setBytes(&vhaP, length: MemoryLayout<VhaInvRopeParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: headDim / 32, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        mark("hca.grouped_wo_a")
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        encodeGroupedWoA(enc, hcaW: hcaW, zGatedOut: zGatedOut, rows: rows, skipExitRotation: skipExitRotation)
    }
}
