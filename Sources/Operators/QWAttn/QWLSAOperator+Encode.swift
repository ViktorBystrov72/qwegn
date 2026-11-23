// QWLSAOperator+Encode.swift — LSA clustering, block selection, and factory helpers.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

extension QWLSAOperator {
    /// Controls which rows project `Q_idx` in `qwg_lsa_index_prep` (`0` = K-only, `1` = all rows, `2` = tiled32 ref rows).
    public enum IndexPrepQMode: UInt32, Sendable {
        case none = 0
        case allRows = 1
        case tiled32 = 2
    }

    public func encodeIndexAndCluster(_ enc: any MTLComputeCommandEncoder,
                                      qgate: QWTensorValue,
                                      kIn: QWTensorValue,
                                      fullAttnLayer: Int,
                                      pos0: Int,
                                      rows: Int,
                                      qMode: IndexPrepQMode = .allRows) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0,
              !isSelectionFollower(fullAttnLayer: fullAttnLayer) else { return }
        let unitTok = clusterUnitTokens
        if pos0 < residentPos512 {
            residentPos512 = isMultiHeadDirect ? ((pos0 / 16) * 16) : ((pos0 / unitTok) * unitTok)
        }
        let ls = layers[fullAttnLayer]
        var tU = UInt32(rows)
        var pos0U = UInt32(pos0)
        var qHeadsU = UInt32(geometry.qHeads)
        var kvHeadsU = UInt32(geometry.kvHeads)
        var headDimU = UInt32(geometry.headDim)
        var qProjOutU = UInt32(geometry.qProjOut)
        var qStrideU = UInt32(geometry.descriptor.attnOutputGate ? 2 : 1)
        var sliceStartU = UInt32(sliceStart)
        var sliceDimU = UInt32(sliceDim)
        var idxDimU = UInt32(indexDim)
        var ringMaskU = UInt32(ringCap - 1)
        var qModeU = qMode.rawValue

        if isMultiHeadDirect {
            var kClipF = kL2Clip
            enc.setComputePipelineState(psoIndexPrepMH)
            qgate.bind(enc, index: 0)
            kIn.bind(enc, index: 1)
            qIdxScratch.bind(enc, index: 2)
            ls.kPending.bind(enc, index: 3)
            enc.setBytes(&tU, length: 4, index: 4)
            enc.setBytes(&pos0U, length: 4, index: 5)
            enc.setBytes(&qHeadsU, length: 4, index: 6)
            enc.setBytes(&kvHeadsU, length: 4, index: 7)
            enc.setBytes(&headDimU, length: 4, index: 8)
            enc.setBytes(&qProjOutU, length: 4, index: 9)
            enc.setBytes(&qStrideU, length: 4, index: 10)
            enc.setBytes(&sliceStartU, length: 4, index: 11)
            enc.setBytes(&sliceDimU, length: 4, index: 12)
            enc.setBytes(&ringMaskU, length: 4, index: 13)
            enc.setBytes(&qModeU, length: 4, index: 14)
            enc.setBytes(&kClipF, length: 4, index: 15)
        } else {
            // 1. Project Q_idx (into qIdxScratch) and K_idx (into kPending ring buffer)
            enc.setComputePipelineState(psoIndexPrep)
            qgate.bind(enc, index: 0)
            kIn.bind(enc, index: 1)
            ls.wqIdx.bind(enc, index: 2)
            ls.wkIdx.bind(enc, index: 3)
            qIdxScratch.bind(enc, index: 4)
            ls.kPending.bind(enc, index: 5)
            enc.setBytes(&tU, length: 4, index: 6)
            enc.setBytes(&pos0U, length: 4, index: 7)
            enc.setBytes(&qHeadsU, length: 4, index: 8)
            enc.setBytes(&kvHeadsU, length: 4, index: 9)
            enc.setBytes(&headDimU, length: 4, index: 10)
            enc.setBytes(&qProjOutU, length: 4, index: 11)
            enc.setBytes(&qStrideU, length: 4, index: 12)
            enc.setBytes(&sliceStartU, length: 4, index: 13)
            enc.setBytes(&sliceDimU, length: 4, index: 14)
            enc.setBytes(&idxDimU, length: 4, index: 15)
            enc.setBytes(&ringMaskU, length: 4, index: 16)
            enc.setBytes(&qModeU, length: 4, index: 17)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        encodePoolAndCluster(enc, fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
    }

    /// BF16 variant accepting separate gate-free `qIn` (`[T, qRowStride]`, `qStride = 1`) and `kIn`.
    public func encodeIndexAndClusterBF16(_ enc: any MTLComputeCommandEncoder,
                                          qIn: QWTensorValue,
                                          kIn: QWTensorValue,
                                          fullAttnLayer: Int,
                                          pos0: Int,
                                          rows: Int,
                                          qRowStride: Int? = nil,
                                          qMode: IndexPrepQMode = .allRows) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0,
              !isSelectionFollower(fullAttnLayer: fullAttnLayer) else { return }
        let unitTok = clusterUnitTokens
        if pos0 < residentPos512 {
            residentPos512 = isMultiHeadDirect ? ((pos0 / 16) * 16) : ((pos0 / unitTok) * unitTok)
        }
        let ls = layers[fullAttnLayer]
        var tU = UInt32(rows)
        var pos0U = UInt32(pos0)
        var qHeadsU = UInt32(geometry.qHeads)
        var kvHeadsU = UInt32(geometry.kvHeads)
        var headDimU = UInt32(geometry.headDim)
        var qRowStrideU = UInt32(qRowStride ?? geometry.qChannels)
        var qStride1U: UInt32 = 1
        var sliceStartU = UInt32(sliceStart)
        var sliceDimU = UInt32(sliceDim)
        var idxDimU = UInt32(indexDim)
        var ringMaskU = UInt32(ringCap - 1)
        var qModeU = qMode.rawValue

        if isMultiHeadDirect {
            var kClipF = kL2Clip
            enc.setComputePipelineState(psoIndexPrepBF16MH)
            qIn.bind(enc, index: 0)
            kIn.bind(enc, index: 1)
            qIdxScratch.bind(enc, index: 2)
            ls.kPending.bind(enc, index: 3)
            enc.setBytes(&tU, length: 4, index: 4)
            enc.setBytes(&pos0U, length: 4, index: 5)
            enc.setBytes(&qHeadsU, length: 4, index: 6)
            enc.setBytes(&kvHeadsU, length: 4, index: 7)
            enc.setBytes(&headDimU, length: 4, index: 8)
            enc.setBytes(&qRowStrideU, length: 4, index: 9)
            enc.setBytes(&qStride1U, length: 4, index: 10)
            enc.setBytes(&sliceStartU, length: 4, index: 11)
            enc.setBytes(&sliceDimU, length: 4, index: 12)
            enc.setBytes(&ringMaskU, length: 4, index: 13)
            enc.setBytes(&qModeU, length: 4, index: 14)
            enc.setBytes(&kClipF, length: 4, index: 15)
        } else {
            enc.setComputePipelineState(psoIndexPrepBF16)
            qIn.bind(enc, index: 0)
            kIn.bind(enc, index: 1)
            ls.wqIdx.bind(enc, index: 2)
            ls.wkIdx.bind(enc, index: 3)
            qIdxScratch.bind(enc, index: 4)
            ls.kPending.bind(enc, index: 5)
            enc.setBytes(&tU, length: 4, index: 6)
            enc.setBytes(&pos0U, length: 4, index: 7)
            enc.setBytes(&qHeadsU, length: 4, index: 8)
            enc.setBytes(&kvHeadsU, length: 4, index: 9)
            enc.setBytes(&headDimU, length: 4, index: 10)
            enc.setBytes(&qRowStrideU, length: 4, index: 11)
            enc.setBytes(&sliceStartU, length: 4, index: 12)
            enc.setBytes(&sliceDimU, length: 4, index: 13)
            enc.setBytes(&idxDimU, length: 4, index: 14)
            enc.setBytes(&ringMaskU, length: 4, index: 15)
            enc.setBytes(&qModeU, length: 4, index: 16)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        encodePoolAndCluster(enc, fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
    }

    private func encodePoolAndCluster(_ enc: any MTLComputeCommandEncoder,
                                      fullAttnLayer: Int,
                                      pos0: Int,
                                      rows: Int) {
        let ls = layers[fullAttnLayer]
        var idxDimU = UInt32(indexDim)
        var kvHeadsU = UInt32(geometry.kvHeads)
        var sliceDimU = UInt32(sliceDim)
        var pooledMetaU = UInt32(pooledMetaBytes)
        var pooledRowU = UInt32(pooledRowBytes)
        var ringMaskU = UInt32(ringCap - 1)

        // 2. Pool any newly completed 4-token blocks in [pos0 / 4 ..< (pos0 + rows) / 4]
        let startBlock = pos0 / 4
        let endBlock = min(maxBlocks, (pos0 + rows) / 4)
        if endBlock > startBlock {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            let numBlocks = endBlock - startBlock
            var startBlkU = UInt32(startBlock)
            var numBlkU = UInt32(numBlocks)
            if isMultiHeadDirect {
                enc.setComputePipelineState(psoPoolKeysMH)
                ls.kPending.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                enc.setBytes(&startBlkU, length: 4, index: 2)
                enc.setBytes(&numBlkU, length: 4, index: 3)
                enc.setBytes(&kvHeadsU, length: 4, index: 4)
                enc.setBytes(&sliceDimU, length: 4, index: 5)
                enc.setBytes(&pooledMetaU, length: 4, index: 6)
                enc.setBytes(&pooledRowU, length: 4, index: 7)
                enc.setBytes(&ringMaskU, length: 4, index: 8)
            } else {
                enc.setComputePipelineState(psoPoolKeys)
                ls.kPending.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                enc.setBytes(&startBlkU, length: 4, index: 2)
                enc.setBytes(&numBlkU, length: 4, index: 3)
                enc.setBytes(&idxDimU, length: 4, index: 4)
                enc.setBytes(&ringMaskU, length: 4, index: 5)
            }
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        if isMultiHeadDirect {
            let startCoarse = pos0 / 16
            let endCoarse = min(maxCoarseBlocks, (pos0 + rows) / 16)
            if endCoarse > startCoarse {
                QWCoarsePoolIndex.barrierIfConcurrent(enc)
                var startCU = UInt32(startCoarse)
                var numCU = UInt32(endCoarse - startCoarse)
                enc.setComputePipelineState(psoPoolCoarseMH)
                ls.pooledKeys.bind(enc, index: 0)
                ls.pooledCoarseKeys.bind(enc, index: 1)
                enc.setBytes(&startCU, length: 4, index: 2)
                enc.setBytes(&numCU, length: 4, index: 3)
                enc.setBytes(&kvHeadsU, length: 4, index: 4)
                enc.setBytes(&sliceDimU, length: 4, index: 5)
                enc.setBytes(&pooledMetaU, length: 4, index: 6)
                enc.setBytes(&pooledRowU, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: endCoarse - startCoarse, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            return
        }

        // 3. Cluster newly completed chunks (4096-token e-means CSR or 512-token Ward superblocks).
        if isEmeansCoarse, let coarsePipeline {
            let csr = QWCoarsePoolIndex.LayerCSR(
                centroids: ls.centroids, memberPools: ls.bucketMembers,
                clusterSpans: ls.bucketSizes, maxMacros: maxSuperblocks)
            coarsePipeline.updateClusters(
                enc, pooledKeys: ls.pooledKeys, csr: csr,
                layerIndex: fullAttnLayer, pos0: pos0, rows: rows, indexDim: indexDim)
            return
        }
        let startSB = pos0 / 512
        let endSB = min(maxSuperblocks, (pos0 + rows) / 512)
        if endSB > startSB {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            let numSB = endSB - startSB
            var startSBU = UInt32(startSB)
            var numSBU = UInt32(numSB)
            enc.setComputePipelineState(psoWardCluster)
            ls.pooledKeys.bind(enc, index: 0)
            ls.centroids.bind(enc, index: 1)
            ls.bucketMembers.bind(enc, index: 2)
            ls.bucketSizes.bind(enc, index: 3)
            enc.setBytes(&startSBU, length: 4, index: 4)
            enc.setBytes(&numSBU, length: 4, index: 5)
            enc.setBytes(&idxDimU, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: numSB, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    /// Encode LSA block selection into `blockListScratch`/`blockCountScratch` (parallel 4-stage selector in the
    /// centroid-filter regime, else `qwg_lsa_select_blocks`). With `tiled32` (NAX prefill), selects only for
    /// reference row `min(tile * 32 + 31, rows - 1)` of each 32-row tile.
    public func encodeSelectBlocks(_ enc: any MTLComputeCommandEncoder,
                                   fullAttnLayer: Int,
                                   pos0: Int,
                                   rows: Int,
                                   tiled32: Bool = false,
                                   sink: QWTensorValue? = nil) {
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0,
              !isSelectionFollower(fullAttnLayer: fullAttnLayer) else { return }
        let ls = layers[fullAttnLayer]
        var idxDimU = UInt32(indexDim)
        var budgetBlkU = UInt32(config.budgetBlocks)
        var windowBlkU = UInt32(QWLSAConfig.windowTokens / QWLSAConfig.poolSize)
        var strideU = UInt32(listStride)
        var scratchStrideU = UInt32(maxBlocks)
        let step = tiled32 ? 32 : 1
        var rowStepU = UInt32(step)
        let totalUnits = tiled32 ? ((rows + 31) / 32) : rows

        var u0 = 0
        let targetList = blockList(forFullAttnLayer: fullAttnLayer)
        let targetCount = blockCount(forFullAttnLayer: fullAttnLayer)
        while u0 < totalUnits {
            if u0 > 0 { QWCoarsePoolIndex.barrierIfConcurrent(enc) }
            let batchCap = isMultiHeadDirect ? Self.maxMHSelectBatchRows : Self.maxSelectBatchRows
            let batchUnits = min(batchCap, totalUnits - u0)
            let r0 = u0 * step
            let remRows = rows - r0
            var subUnitsU = UInt32(batchUnits)
            var subPos0U = UInt32(pos0 + r0)
            var maxRowU = UInt32(remRows)
            let qIdxSub = QWTensorValue(
                region: qIdxScratch.region,
                offset: qIdxScratch.offset + r0 * indexDim * 2,
                shape: QWTensorShape([remRows, indexDim], .f16))
            let listSub = QWTensorValue(
                region: targetList.region,
                offset: targetList.offset + r0 * listStride * 4,
                shape: QWTensorShape([remRows, listStride], .i32))
            let countSub = QWTensorValue(
                region: targetCount.region,
                offset: targetCount.offset + r0 * 4,
                shape: QWTensorShape([remRows], .i32))

            if isMultiHeadDirect {
                let mhQIdxSub = QWTensorValue(
                    region: qIdxScratch.region,
                    offset: qIdxScratch.offset + r0 * qIndexStride * 2,
                    shape: QWTensorShape([remRows, qIndexStride], .f16))
                encodeMultiHeadSelectBatch(
                    enc, ls: ls, mhQIdxSub: mhQIdxSub, listSub: listSub, countSub: countSub,
                    sink: sink, r0: r0, remRows: remRows, pos0: pos0,
                    batchUnits: batchUnits, step: step)
                u0 += batchUnits
                continue
            }

            if isEmeansCoarse, let coarsePipeline {
                let winBlk = QWLSAConfig.windowTokens / QWLSAConfig.poolSize
                let bg = QWCoarsePoolIndex.batchGeometry(
                    pos0: pos0 + r0, units: batchUnits, rowStep: step, maxRow: remRows,
                    budgetBlocks: config.budgetBlocks, windowBlocks: winBlk,
                    hasSink: true, writeCount: true, probeClusters: Self.emeansProbeClusters)
                let shouldPrune = (bg.selectionMode == .approximateCoarse) && useParallelSelect
                let probe = shouldPrune ? Self.emeansProbeClusters : 0
                var ePrm = QWCoarsePoolIndex.SelectParams(
                    T: batchUnits, pos0: pos0 + r0, indexDim: indexDim,
                    budgetBlocks: config.budgetBlocks, windowBlocks: winBlk,
                    hasSink: true, writeCount: true, listStride: listStride,
                    rowStep: step, maxRow: remRows, clusterStride: maxBuckets,
                    probeClusters: probe, candCap: Self.selectCandidateCap)
                if bg.isDense {
                    coarsePipeline.encodeDenseFinalize(
                        enc, blockList: listSub, blockCount: countSub,
                        params: &ePrm, batchUnits: batchUnits)
                    u0 += batchUnits
                    continue
                }
                let csr = QWCoarsePoolIndex.LayerCSR(
                    centroids: ls.centroids, memberPools: ls.bucketMembers,
                    clusterSpans: ls.bucketSizes, maxMacros: maxSuperblocks)
                coarsePipeline.encodeSelection(
                    enc, query: qIdxSub, pooledKeys: ls.pooledKeys, csr: csr,
                    blockList: listSub, blockCount: countSub, params: &ePrm,
                    batchUnits: batchUnits, numClusters: bg.numClusters,
                    middleEnd: bg.middleEnd, shouldPrune: shouldPrune)
                u0 += batchUnits
                continue
            }

            if useParallelSelect, let numBuckets = filterBuckets(pos0: pos0 + r0, units: batchUnits,
                                              rowStep: step, maxRow: remRows) {
                var prm = SelectParams(
                    T: subUnitsU, pos0: subPos0U, indexDim: idxDimU, budgetBlocks: budgetBlkU,
                    windowBlocks: windowBlkU, listStride: strideU, scoreStride: scratchStrideU,
                    rowStep: rowStepU, maxRow: maxRowU, bucketStride: UInt32(maxBuckets),
                    candCap: UInt32(Self.selectCandidateCap), markStride: UInt32(selMarkStride))
                let plen = MemoryLayout<SelectParams>.stride
                let grid = MTLSize(width: (numBuckets + 255) / 256, height: batchUnits, depth: 1)
                let tpg = MTLSize(width: 256, height: 1, depth: 1)

                enc.setComputePipelineState(psoSelectScore)
                qIdxSub.bind(enc, index: 0)
                ls.centroids.bind(enc, index: 1)
                selCScore.bind(enc, index: 2)
                selHist.bind(enc, index: 3)
                enc.setBytes(&prm, length: plen, index: 4)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(psoSelectHist2)
                selCScore.bind(enc, index: 0)
                selHist.bind(enc, index: 1)
                enc.setBytes(&prm, length: plen, index: 2)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(psoSelectExpand)
                qIdxSub.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                ls.bucketMembers.bind(enc, index: 2)
                ls.bucketSizes.bind(enc, index: 3)
                selCScore.bind(enc, index: 4)
                selHist.bind(enc, index: 5)
                selMarks.bind(enc, index: 6)
                selCount.bind(enc, index: 7)
                selCandM.bind(enc, index: 8)
                selCandU.bind(enc, index: 9)
                enc.setBytes(&prm, length: plen, index: 10)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(psoSelectFinalize)
                qIdxSub.bind(enc, index: 0)
                ls.pooledKeys.bind(enc, index: 1)
                ls.centroids.bind(enc, index: 2)
                ls.bucketMembers.bind(enc, index: 3)
                ls.bucketSizes.bind(enc, index: 4)
                scoreScratch.bind(enc, index: 5)
                listSub.bind(enc, index: 6)
                countSub.bind(enc, index: 7)
                selHist.bind(enc, index: 8)
                selMarks.bind(enc, index: 9)
                selCount.bind(enc, index: 10)
                selCandM.bind(enc, index: 11)
                selCandU.bind(enc, index: 12)
                enc.setBytes(&prm, length: plen, index: 13)
                enc.dispatchThreadgroups(MTLSize(width: batchUnits, height: 1, depth: 1),
                                         threadsPerThreadgroup: tpg)
                u0 += batchUnits
                continue
            }

            enc.setComputePipelineState(psoSelectBlocks)
            qIdxSub.bind(enc, index: 0)
            ls.pooledKeys.bind(enc, index: 1)
            ls.centroids.bind(enc, index: 2)
            ls.bucketMembers.bind(enc, index: 3)
            ls.bucketSizes.bind(enc, index: 4)
            scoreScratch.bind(enc, index: 5)
            listSub.bind(enc, index: 6)
            countSub.bind(enc, index: 7)
            enc.setBytes(&subUnitsU, length: 4, index: 8)
            enc.setBytes(&subPos0U, length: 4, index: 9)
            enc.setBytes(&idxDimU, length: 4, index: 10)
            enc.setBytes(&budgetBlkU, length: 4, index: 11)
            enc.setBytes(&windowBlkU, length: 4, index: 12)
            enc.setBytes(&strideU, length: 4, index: 13)
            enc.setBytes(&scratchStrideU, length: 4, index: 14)
            enc.setBytes(&rowStepU, length: 4, index: 15)
            enc.setBytes(&maxRowU, length: 4, index: 16)
            enc.dispatchThreadgroups(
                MTLSize(width: batchUnits, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            u0 += batchUnits
        }
    }

    private func encodeMHScoreAndCollect(
        _ enc: any MTLComputeCommandEncoder,
        qIdxSub: QWTensorValue,
        keyPool: QWTensorValue,
        sinkSub: QWTensorValue,
        batchUnits: Int,
        bxScore: Int,
        bxMid: Int,
        byScore: Int,
        mhPrm: inout MHSelectParams
    ) {
        let plen = MemoryLayout<MHSelectParams>.stride
        let tpg = MTLSize(width: 256, height: 1, depth: 1)

        enc.setComputePipelineState(psoMHScoreHeads)
        qIdxSub.bind(enc, index: 0)
        keyPool.bind(enc, index: 1)
        mhFineCandIdx.bind(enc, index: 2)
        mhFineCandCount.bind(enc, index: 3)
        mhHeadScores.bind(enc, index: 4)
        mhTGStats.bind(enc, index: 5)
        enc.setBytes(&mhPrm, length: plen, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: bxScore, height: byScore, depth: 1), threadsPerThreadgroup: tpg)
        QWCoarsePoolIndex.barrierIfConcurrent(enc)

        enc.setComputePipelineState(psoMHReduceLSE)
        mhTGStats.bind(enc, index: 0)
        sinkSub.bind(enc, index: 1)
        mhFineCandCount.bind(enc, index: 2)
        mhHeadLSE.bind(enc, index: 3)
        enc.setBytes(&mhPrm, length: plen, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: geometry.qHeads, height: batchUnits, depth: 1), threadsPerThreadgroup: tpg)
        QWCoarsePoolIndex.barrierIfConcurrent(enc)

        enc.setComputePipelineState(psoMHAggHist)
        mhHeadScores.bind(enc, index: 0)
        mhHeadLSE.bind(enc, index: 1)
        mhFineCandCount.bind(enc, index: 2)
        scoreScratch.bind(enc, index: 3)
        selHist.bind(enc, index: 4)
        enc.setBytes(&mhPrm, length: plen, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: bxMid, height: batchUnits, depth: 1), threadsPerThreadgroup: tpg)
        QWCoarsePoolIndex.barrierIfConcurrent(enc)

        enc.setComputePipelineState(psoMHCollect)
        scoreScratch.bind(enc, index: 0)
        mhFineCandIdx.bind(enc, index: 1)
        mhFineCandCount.bind(enc, index: 2)
        selHist.bind(enc, index: 3)
        selCount.bind(enc, index: 4)
        selCandM.bind(enc, index: 5)
        selCandU.bind(enc, index: 6)
        enc.setBytes(&mhPrm, length: plen, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: bxMid, height: batchUnits, depth: 1), threadsPerThreadgroup: tpg)
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
    }

    private func encodeMultiHeadSelectBatch(
        _ enc: any MTLComputeCommandEncoder,
        ls: LayerState,
        mhQIdxSub: QWTensorValue,
        listSub: QWTensorValue,
        countSub: QWTensorValue,
        sink: QWTensorValue?,
        r0: Int,
        remRows: Int,
        pos0: Int,
        batchUnits: Int,
        step: Int
    ) {
        let winBlk = QWLSAConfig.windowTokens / QWLSAConfig.poolSize
        let firstRow = step > 1 ? min(step - 1, remRows - 1) : 0
        let lastRow = step > 1 ? min((batchUnits - 1) * step + step - 1, remRows - 1) : (batchUnits - 1)
        let minTotalBlocks = (pos0 + r0 + firstRow + 1) >> 2
        let minMiddleEnd = max(0, minTotalBlocks - winBlk)
        let minNumMiddle = max(0, minMiddleEnd - 1)
        let minNumCoarseMid = (minMiddleEnd + 3) >> 2
        let maxTotalBlocks = (pos0 + r0 + lastRow + 1) >> 2
        let maxMiddleEnd = max(0, maxTotalBlocks - winBlk)
        let maxNumMiddle = max(0, maxMiddleEnd - 1)
        let maxTotalCoarse = maxTotalBlocks >> 2
        let maxNumCoarseMid = (maxMiddleEnd + 3) >> 2
        let useTwoStage = (minNumMiddle > mhFineCandCap && minNumCoarseMid > mhCoarseCap)
        let bqTile = (batchUnits == 1) ? 1 : min(4, batchUnits)

        let sinkSub: QWTensorValue
        let hasSink: UInt32
        if let sink {
            sinkSub = QWTensorValue(
                region: sink.region,
                offset: sink.offset + r0 * geometry.qHeads * 2,
                shape: QWTensorShape([remRows * geometry.qHeads], .bf16))
            hasSink = 1
        } else {
            sinkSub = mhDummySink
            hasSink = 0
        }

        var mhPrm = MHSelectParams(
            T: UInt32(batchUnits), pos0: UInt32(pos0 + r0), sliceDim: UInt32(sliceDim),
            kvHeads: UInt32(geometry.kvHeads), qHeads: UInt32(geometry.qHeads),
            qIndexStride: UInt32(qIndexStride),
            pooledMetaBytes: UInt32(pooledMetaBytes), pooledRowBytes: UInt32(pooledRowBytes),
            budgetBlocks: UInt32(config.budgetBlocks), windowBlocks: UInt32(winBlk),
            listStride: UInt32(listStride), scoreStride: UInt32(mhHeadScoreCols),
            headScoreStride: UInt32(mhHeadScoreCols), rowStep: UInt32(step), maxRow: UInt32(remRows),
            candCap: UInt32(Self.selectCandidateCap),
            bqTile: UInt32(bqTile), scoreTilesX: UInt32(mhScoreTilesX),
            coarseCap: UInt32(useTwoStage ? mhCoarseCap : 0),
            hasSink: hasSink, sinkRowStride: UInt32(geometry.qHeads), stageMode: useTwoStage ? 1 : 0)
        let plen = MemoryLayout<MHSelectParams>.stride
        let tpg = MTLSize(width: 256, height: 1, depth: 1)

        if maxNumMiddle > config.budgetBlocks {
            if useTwoStage {
                encodeMHScoreAndCollect(
                    enc, qIdxSub: mhQIdxSub, keyPool: ls.pooledCoarseKeys, sinkSub: sinkSub,
                    batchUnits: batchUnits,
                    bxScore: (maxTotalCoarse + 255) / 256,
                    bxMid: (maxNumCoarseMid + 255) / 256,
                    byScore: (batchUnits + bqTile - 1) / bqTile,
                    mhPrm: &mhPrm)

                enc.setComputePipelineState(psoMHCoarseExpand)
                scoreScratch.bind(enc, index: 0)
                mhFineCandIdx.bind(enc, index: 1)
                mhFineCandCount.bind(enc, index: 2)
                selHist.bind(enc, index: 3)
                selCount.bind(enc, index: 4)
                selCandM.bind(enc, index: 5)
                selCandU.bind(enc, index: 6)
                enc.setBytes(&mhPrm, length: plen, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: batchUnits, height: 1, depth: 1), threadsPerThreadgroup: tpg)
                QWCoarsePoolIndex.barrierIfConcurrent(enc)

                mhPrm.stageMode = 2
                mhPrm.bqTile = 1
                let maxFineItems = mhFineCandCap + 1 + winBlk
                encodeMHScoreAndCollect(
                    enc, qIdxSub: mhQIdxSub, keyPool: ls.pooledKeys, sinkSub: sinkSub,
                    batchUnits: batchUnits,
                    bxScore: (maxFineItems + 255) / 256,
                    bxMid: (mhFineCandCap + 255) / 256,
                    byScore: batchUnits,
                    mhPrm: &mhPrm)
            } else {
                encodeMHScoreAndCollect(
                    enc, qIdxSub: mhQIdxSub, keyPool: ls.pooledKeys, sinkSub: sinkSub,
                    batchUnits: batchUnits,
                    bxScore: (maxTotalBlocks + 255) / 256,
                    bxMid: (maxNumMiddle + 255) / 256,
                    byScore: (batchUnits + bqTile - 1) / bqTile,
                    mhPrm: &mhPrm)
            }
        }

        enc.setComputePipelineState(psoMHFinalize)
        scoreScratch.bind(enc, index: 0)
        mhFineCandIdx.bind(enc, index: 1)
        mhFineCandCount.bind(enc, index: 2)
        listSub.bind(enc, index: 3)
        countSub.bind(enc, index: 4)
        selHist.bind(enc, index: 5)
        selCount.bind(enc, index: 6)
        selCandM.bind(enc, index: 7)
        selCandU.bind(enc, index: 8)
        enc.setBytes(&mhPrm, length: plen, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: batchUnits, height: 1, depth: 1), threadsPerThreadgroup: tpg)
    }

    public static func makeIfEnabled(
        hal: QWHal,
        geometry: QWModelGeometry,
        lsaConfig: QWLSAConfig,
        packPath: String,
        pack: QWQuant.Pack?,
        safetensors: QWSafeTensorsContainer? = nil,
        attnLayers: [Int],
        maxContext: Int,
        maxPrefillChunk: Int
    ) throws -> QWLSAOperator? {
        guard lsaConfig.isEnabled && !geometry.usesAnyMLA && geometry.blockSparseGQAConfig == nil
            && !attnLayers.isEmpty && geometry.lsaInputDim > 0 else { return nil }
        let resolvedConfig = lsaConfig.resolvingLoopShareEnvironment().resolvingAdaptiveBudget(forMaxContext: maxContext)
        let emeans = (resolvedConfig.alignedClusterBlock == QWLSAConfig.emeansClusterBlock)
        let supportsFullMH = !emeans && QWLSAConfig.supportsFullSliceQ4Direct(
            qHeads: geometry.qHeads, kvHeads: geometry.kvHeads, sliceDim: geometry.lsaHeadSliceDim)
        let sidecarURL = QWLSAConfig.resolveSidecarURL(packPath: packPath, explicitSidecarPath: lsaConfig.sidecarPath)
        let inDim = geometry.lsaInputDim
        var indexDim = QWLSAConfig.defaultIndexDim
        var useFullMHDirect = false
        var layerWeights: [LayerWeights] = []
        if lsaConfig.mode != .svd, let url = sidecarURL,
           let loaded = try? QWLSASidecar(url: url, descriptor: geometry.descriptor) {
            indexDim = loaded.header.provenance.indexDim
            let trainingFree = QWLSAConfig.isTrainingFreeSidecarMethod(loaded.header.provenance.method)
            if supportsFullMH && trainingFree {
                useFullMHDirect = true
                for vLayer in attnLayers {
                    let physLayer = geometry.executionPlan[vLayer]
                    layerWeights.append(.init(
                        physicalLayer: physLayer, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true))
                }
            } else {
                let shape = [indexDim, inDim]
                for vLayer in attnLayers {
                    let physLayer = geometry.executionPlan[vLayer]
                    let wqLoaded = try? loaded.readF16Tensor("layers.\(physLayer).wq_idx", expectedShape: shape)
                    let wkLoaded = try? loaded.readF16Tensor("layers.\(physLayer).wk_idx", expectedShape: shape)
                    let wq = wqLoaded ?? QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(physLayer &* 2 &+ 1))
                    let wk = wkLoaded ?? wq
                    layerWeights.append(.init(
                        physicalLayer: physLayer, wqIdxF16: wq, wkIdxF16: wk, isHeadFactorized: false))
                }
            }
        } else if supportsFullMH {
            useFullMHDirect = true
            for vLayer in attnLayers {
                let physLayer = geometry.executionPlan[vLayer]
                layerWeights.append(.init(
                    physicalLayer: physLayer, wqIdxF16: [], wkIdxF16: [], isHeadFactorized: true))
            }
        } else {
            let synth: [Int: (wq: [UInt16], wk: [UInt16])]?
            if let pk = pack {
                synth = try? QWLSASidecar.synthesizeFromPack(
                    pack: pk, descriptor: geometry.descriptor, indexDim: indexDim)
            } else if let st = safetensors {
                synth = try? QWLSASidecar.synthesizeFromSafeTensors(
                    container: st, descriptor: geometry.descriptor, indexDim: indexDim)
            } else {
                synth = nil
            }
            for vLayer in attnLayers {
                let physLayer = geometry.executionPlan[vLayer]
                if let pair = synth?[physLayer] {
                    layerWeights.append(.init(
                        physicalLayer: physLayer, wqIdxF16: pair.wq, wkIdxF16: pair.wk, isHeadFactorized: false))
                } else {
                    let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
                        inDim: inDim, indexDim: indexDim, seed: UInt64(physLayer &* 2 &+ 1))
                    layerWeights.append(.init(
                        physicalLayer: physLayer, wqIdxF16: basis, wkIdxF16: basis, isHeadFactorized: false))
                }
            }
        }
        return try QWLSAOperator(
            hal: hal, geometry: geometry, config: resolvedConfig,
            indexDim: indexDim, layerWeights: layerWeights,
            maxContext: maxContext, maxPrefillChunk: maxPrefillChunk,
            isMultiHeadDirect: useFullMHDirect)
    }
}
