// QWMLAAttnOperator+Encode.swift — Indexer snapshot/restore and Metal encode passes for MLA.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWKV

extension QWMLAAttnOperator {
    public func indexerStateBytes(position: Int) -> Int {
        if isDecoupledDSA {
            return layers.count * position * (idxDim + ropeDim) * 2
        }
        let rowBytes = idxDim * 2
        return layers.count * ((position / 4) + 2 * (position % 4)) * rowBytes
    }

    public func captureIndexerState(position: Int) -> Data {
        if isDecoupledDSA { return captureDSAIndexerState(position: position) }
        precondition(position >= 0 && position / 4 <= maxBlocks)
        let rowBytes = idxDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let openStart = (position / 4) * 4
        var out = Data(capacity: indexerStateBytes(position: position))
        let base = region.pointer
        for ls in layers {
            out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self),
                       count: pooledBytes)
            for ring in [ls.kPending, ls.gatePending] {
                for p in openStart..<position {
                    let row = p & (ringCap - 1)
                    out.append(base.advanced(by: ring.offset + row * rowBytes).assumingMemoryBound(to: UInt8.self),
                               count: rowBytes)
                }
            }
        }
        return out
    }

    public func restoreIndexerState(_ data: Data, position: Int) -> Bool {
        if isDecoupledDSA { return restoreDSAIndexerState(data, position: position) }
        guard position >= 0, position / 4 <= maxBlocks,
              data.count == indexerStateBytes(position: position) else { return false }
        let rowBytes = idxDim * 2
        let pooledBytes = (position / 4) * rowBytes
        let openStart = (position / 4) * 4
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var src = 0
            for ls in layers {
                base.advanced(by: ls.pooledKeys.offset)
                    .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: pooledBytes)
                src += pooledBytes
                for ring in [ls.kPending, ls.gatePending] {
                    for p in openStart..<position {
                        let row = p & (ringCap - 1)
                        base.advanced(by: ring.offset + row * rowBytes)
                            .copyMemory(from: raw.baseAddress!.advanced(by: src), byteCount: rowBytes)
                        src += rowBytes
                    }
                }
            }
        }
        resetCoarseWatermarks()
        return true
    }

    /// Bytes needed to capture live multi-session indexer pools, preserving full `ringCap` open rows and optional MTP layer cursor.
    public func liveSessionStashBytes(trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Int {
        if isDecoupledDSA {
            return liveDSASessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition)
        }
        let rowBytes = idxDim * 2
        let ringBytes = 2 * ringCap * rowBytes
        var total = 0
        for i in layers.indices {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            total += (pos / 4) * rowBytes + ringBytes
        }
        return total
    }

    public func captureLiveSessionStash(trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Data {
        if isDecoupledDSA {
            return captureDSALiveSessionStash(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition)
        }
        precondition(trunkPosition >= 0 && mtpPosition >= 0)
        let rowBytes = idxDim * 2
        let ringBytes = ringCap * rowBytes
        var out = Data(capacity: liveSessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition))
        let base = region.pointer
        for (i, ls) in layers.enumerated() {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            precondition(pos / 4 <= maxBlocks)
            let pooledBytes = (pos / 4) * rowBytes
            if pooledBytes > 0 {
                out.append(base.advanced(by: ls.pooledKeys.offset).assumingMemoryBound(to: UInt8.self), count: pooledBytes)
            }
            out.append(base.advanced(by: ls.kPending.offset).assumingMemoryBound(to: UInt8.self), count: ringBytes)
            out.append(base.advanced(by: ls.gatePending.offset).assumingMemoryBound(to: UInt8.self), count: ringBytes)
        }
        return out
    }

    public func restoreLiveSessionStash(_ data: Data, trunkPosition: Int, mtpLayerIndex: Int?, mtpPosition: Int) -> Bool {
        if isDecoupledDSA {
            return restoreDSALiveSessionStash(data, trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition)
        }
        guard trunkPosition >= 0, mtpPosition >= 0,
              data.count == liveSessionStashBytes(trunkPosition: trunkPosition, mtpLayerIndex: mtpLayerIndex, mtpPosition: mtpPosition) else {
            return false
        }
        for i in layers.indices {
            let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
            if pos / 4 > maxBlocks { return false }
        }
        let rowBytes = idxDim * 2
        let ringBytes = ringCap * rowBytes
        let base = region.pointer
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let srcBase = raw.baseAddress else { return }
            var src = 0
            for (i, ls) in layers.enumerated() {
                let pos = max(0, (i == mtpLayerIndex) ? mtpPosition : trunkPosition)
                let pooledBytes = (pos / 4) * rowBytes
                if pooledBytes > 0 {
                    base.advanced(by: ls.pooledKeys.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: pooledBytes)
                    src += pooledBytes
                }
                base.advanced(by: ls.kPending.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: ringBytes)
                src += ringBytes
                base.advanced(by: ls.gatePending.offset).copyMemory(from: srcBase.advanced(by: src), byteCount: ringBytes)
                src += ringBytes
            }
        }
        resetCoarseWatermarks()
        return true
    }

    public func encodeLatentPrepAndPool(_ enc: any MTLComputeCommandEncoder,
                                        ln1: QWTensorValue,
                                        mlaW: LayerWeights,
                                        kv: (k: QWKVLayerView, v: QWKVLayerView),
                                        fullAttnLayer: Int,
                                        pos0: Int,
                                        rows: Int) {
        if isDecoupledDSA {
            encodeDSALatentPrepAndCluster(enc, ln1: ln1, mlaW: mlaW, kv: kv,
                                          fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
            return
        }
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        let ls = layers[fullAttnLayer]
        var prepP = LatentPrepParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            hidden: UInt32(geometry.hidden),
            qLoraRank: UInt32(qLoraRank),
            kvLoraRank: UInt32(kvLoraRank),
            idxDim: UInt32(idxDim),
            idxHeads: UInt32(idxHeads),
            ringMask: UInt32(ringCap - 1),
            rmsEps: geometry.rmsEps,
            kNormEps: 1e-6,
            kvMode: Self.kvModeCode(kv.k.mode)
        )

        enc.setComputePipelineState(psoLatentPrep)
        mlaLatentScratch.bind(enc, index: 0)
        ln1.bind(enc, index: 1)
        mlaW.qANormW.bind(enc, index: 2)
        mlaW.kvANormW.bind(enc, index: 3)
        mlaW.kNormW.bind(enc, index: 4)
        mlaW.kNormB.bind(enc, index: 5)
        mlaW.wGateWeights.bind(enc, index: 6)
        qResidScratch.bind(enc, index: 7)
        kv.k.pool.bindTo(enc, index: 8)
        kv.v.pool.bindTo(enc, index: 9)
        kv.k.table.bind(enc, index: 10)
        kv.v.table.bind(enc, index: 11)
        ls.kPending.bind(enc, index: 12)
        ls.gatePending.bind(enc, index: 13)
        idxWeightsScratch.bind(enc, index: 14)
        enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoGateWeightsProj)
        ln1.bind(enc, index: 0)
        mlaW.wGateWeights.bind(enc, index: 1)
        ls.gatePending.bind(enc, index: 2)
        idxWeightsScratch.bind(enc, index: 3)
        enc.setBytes(&prepP, length: MemoryLayout<LatentPrepParams>.stride, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (idxDim + idxHeads + 7) / 8, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let startBlock = pos0 / 4
        let endBlock = (pos0 + rows) / 4
        let numBlocks = endBlock - startBlock
        if numBlocks > 0, let ape = mlaW.ape {
            var sB = UInt32(startBlock)
            var nB = UInt32(numBlocks)
            var dU = UInt32(idxDim)
            var rM = UInt32(ringCap - 1)
            enc.setComputePipelineState(psoPoolKeys)
            ls.kPending.bind(enc, index: 0)
            ls.gatePending.bind(enc, index: 1)
            ape.bind(enc, index: 2)
            ls.pooledKeys.bind(enc, index: 3)
            enc.setBytes(&sB, length: 4, index: 4)
            enc.setBytes(&nB, length: 4, index: 5)
            enc.setBytes(&dU, length: 4, index: 6)
            enc.setBytes(&rM, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(128, idxDim), height: 1, depth: 1))
        }
        updateCoarseClusters(enc, ls: ls, fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows)
    }

    internal func updateCoarseClusters(_ enc: any MTLComputeCommandEncoder,
                                       ls: LayerState,
                                       fullAttnLayer: Int,
                                       pos0: Int,
                                       rows: Int) {
        guard emeansCoarseEnabled, let pipes = coarsePipelines, let csr = ls.coarseCSR else { return }
        let chunkTok = QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        let curMacro = pos0 / chunkTok
        if curMacro < clusteredMacros[fullAttnLayer] {
            clusteredMacros[fullAttnLayer] = curMacro
        }
        let targetMacros = min(csr.maxMacros, (pos0 + rows) / chunkTok)
        let prevMacros = clusteredMacros[fullAttnLayer]
        if targetMacros > prevMacros {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
            QWCoarsePoolIndex.encodeClusterMacros(
                enc, pso: pipes.cluster1024, pooledKeys: ls.pooledKeys,
                csr: csr, startMacro: prevMacros, numMacros: targetMacros - prevMacros, indexDim: idxDim)
            clusteredMacros[fullAttnLayer] = targetMacros
        }
    }

    public func encodeScoreSelectAbsorbAttendAndProjUV(_ enc: any MTLComputeCommandEncoder,
                                                       mlaW: LayerWeights,
                                                       kv: (k: QWKVLayerView, v: QWKVLayerView),
                                                       attOut: QWTensorValue,
                                                       fullAttnLayer: Int,
                                                       pos0: Int,
                                                       rows: Int,
                                                       stage: ((String) -> (any MTLComputeCommandEncoder)?)? = nil) {
        if isDecoupledDSA {
            encodeDSAScoreSelectAbsorbAttendAndProjUV(
                enc, mlaW: mlaW, kv: kv, attOut: attOut,
                fullAttnLayer: fullAttnLayer, pos0: pos0, rows: rows, stage: stage)
            return
        }
        guard fullAttnLayer >= 0, fullAttnLayer < layers.count, rows > 0 else { return }
        var enc = enc
        func mark(_ name: String) { if let e = stage?(name) { enc = e } }
        let ls = layers[fullAttnLayer]
        mark("mla.score_select")

        encodeIndexerSelection(enc, ls: ls, pos0: pos0, rows: rows,
                               probeClusters: Self.emeansProbeClusters, windowBlocks: 0)

        var absP = AbsorbParams(
            T: UInt32(rows),
            qHeads: UInt32(qHeads),
            nopeDim: UInt32(nopeDim),
            vHeadDim: UInt32(vHeadDim),
            kvLoraRank: UInt32(kvLoraRank),
            qFullStride: UInt32(qFullStride),
            bits: UInt32(mlaW.kvBProj.bits),
            group: UInt32(mlaW.kvBProj.groupSize),
            isIncumbent: mlaW.kvBProj.isIncumbentContainer ? 1 : (mlaW.kvBProj.isSHVQ ? 2 : 0)
        )
        let mTiles = (rows + 7) / 8
        mark("mla.absorb_q")
        let absorbMMA = encodeAbsorbQ(enc, kvBProj: mlaW.kvBProj, absP: &absP, rows: rows, mTiles: mTiles)

        var attnP = SparseAttnParams(
            T: UInt32(rows),
            pos0: UInt32(pos0),
            qHeads: UInt32(qHeads),
            headDim: UInt32(nopeDim),
            kvLoraRank: UInt32(kvLoraRank),
            topKPools: UInt32(topKPools),
            listStride: UInt32(listStride),
            kvMode: Self.kvModeCode(kv.k.mode)
        )
        mark("mla.sparse_attn")
        let useMMA = sparseAttnMMAEnabled && attnP.kvMode <= 2 && rows >= 16
            && kvLoraRank == 512 && qHeads % 16 == 0
        let split = (sparseAttnMMAEnabled && !useMMA && attnP.kvMode <= 2 && rows <= Self.splitMaxRows
            && qHeads % 16 == 0) ? psoSparseAttnSplit : nil
        let useH32 = split == nil && useMMA && qHeads % 32 == 0 && psoSparseAttnNAXH32 != nil
        let naxPSO = useH32 ? psoSparseAttnNAXH32 : psoSparseAttnNAX
        enc.setComputePipelineState(split?.attend ?? (useMMA ? (naxPSO ?? psoSparseAttnMMA) : psoSparseAttn))
        qAbsScratch.bind(enc, index: 0)
        kv.k.pool.bindTo(enc, index: 1)
        kv.v.pool.bindTo(enc, index: 2)
        kv.k.table.bind(enc, index: 3)
        kv.v.table.bind(enc, index: 4)
        blockListScratch.bind(enc, index: 5)
        (split != nil ? partOScratch : uLatScratch).bind(enc, index: 6)
        enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 7)
        if let split {
            partMLScratch.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: qHeads / 16, depth: attnSplits),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            var nSplitU = UInt32(attnSplits)
            enc.setComputePipelineState(split.combine)
            partOScratch.bind(enc, index: 0)
            partMLScratch.bind(enc, index: 1)
            uLatScratch.bind(enc, index: 2)
            enc.setBytes(&attnP, length: MemoryLayout<SparseAttnParams>.stride, index: 3)
            enc.setBytes(&nSplitU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: rows * qHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if useMMA {
            let gridSize = useH32
                ? MTLSize(width: qHeads / 32, height: rows, depth: 1)
                : MTLSize(width: rows, height: qHeads / 16, depth: 1)
            let tgThreads = useH32 ? 256 : 128
            enc.dispatchThreadgroups(gridSize,
                                     threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: qHeads * rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        mark("mla.proj_uv")
        encodeProjUV(enc, kvBProj: mlaW.kvBProj, dst: attOut, absP: &absP,
                     absorbMMA: absorbMMA, rows: rows, mTiles: mTiles)
    }
}
