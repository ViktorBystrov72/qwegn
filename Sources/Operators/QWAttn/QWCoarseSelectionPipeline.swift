// QWCoarseSelectionPipeline.swift — Shared e-means coarse cluster-update and batched selection pipeline.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// Shared e-means coarse indexing pipelines, CSR/scratch builders, watermark
/// state, and batched selection encoding for LSA, HCA, and MLA attention operators.
final class QWCoarseSelectionPipeline: @unchecked Sendable {
    enum Variant {
        case lsa
        case mla

        var scoreCoarseKernel: String {
            switch self {
            case .lsa: return "qwg_emeans_lsa_score_coarse"
            case .mla: return "qwg_emeans_mla_score_coarse"
            }
        }

        var scoreCandsKernel: String {
            switch self {
            case .lsa: return "qwg_emeans_lsa_score_cands"
            case .mla: return "qwg_emeans_mla_score_cands"
            }
        }

        var flatFallbackKernel: String {
            switch self {
            case .lsa: return "qwg_emeans_lsa_flat_fallback"
            case .mla: return "qwg_emeans_mla_flat_fallback"
            }
        }
    }

    typealias EmeansPSOTuple = (
        scoreCoarse: any MTLComputePipelineState,
        scoreCands: any MTLComputePipelineState,
        flatFallback: any MTLComputePipelineState
    )

    struct LoadedPipelines: @unchecked Sendable {
        let coarse: QWCoarsePoolIndex.Pipelines
        let emeans: EmeansPSOTuple

        static func load(hal: QWHal, variant: Variant) throws -> LoadedPipelines {
            LoadedPipelines(
                coarse: try QWCoarsePoolIndex.Pipelines(hal: hal),
                emeans: (
                    try hal.pipeline(variant.scoreCoarseKernel),
                    try hal.pipeline(variant.scoreCandsKernel),
                    try hal.pipeline(variant.flatFallbackKernel)
                )
            )
        }
    }

    struct CSROffsets {
        let cCent: Int
        let cMem: Int
        let cSpan: Int

        static func reserve(
            enabled: Bool,
            maxMacros: Int,
            indexDim: Int,
            reserve: (Int) -> Int
        ) -> CSROffsets {
            guard enabled else { return CSROffsets(cCent: 0, cMem: 0, cSpan: 0) }
            let csrSizes = QWCoarsePoolIndex.csrBytesPerMacro(indexDim: indexDim)
            return CSROffsets(
                cCent: reserve(maxMacros * csrSizes.centroids),
                cMem: reserve(maxMacros * csrSizes.members),
                cSpan: reserve(maxMacros * csrSizes.spans)
            )
        }

        func makeLayerCSR(
            region: QWMRegion,
            maxMacros: Int,
            indexDim: Int
        ) -> QWCoarsePoolIndex.LayerCSR {
            let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
            return QWCoarsePoolIndex.LayerCSR(
                centroids: QWTensorValue(region: region, offset: cCent,
                                         shape: QWTensorShape([maxClusters, indexDim], .f16)),
                memberPools: QWTensorValue(region: region, offset: cMem,
                                           shape: QWTensorShape([maxMacros * 512], .i32)),
                clusterSpans: QWTensorValue(region: region, offset: cSpan,
                                            shape: QWTensorShape([maxClusters], .i32)),
                maxMacros: maxMacros
            )
        }
    }

    struct ScratchOffsets {
        let cScore: Int
        let uHist: Int
        let candState: Int
        let candPools: Int
        let candU: Int

        static func reserve(
            enabled: Bool,
            maxBatchRows: Int,
            maxClusters: Int,
            candCap: Int,
            reserve: (Int) -> Int
        ) -> ScratchOffsets {
            guard enabled else {
                return ScratchOffsets(cScore: 0, uHist: 0, candState: 0, candPools: 0, candU: 0)
            }
            let sizes = QWCoarsePoolIndex.scratchByteSizes(
                maxBatchRows: maxBatchRows, maxClusters: maxClusters, candCap: candCap)
            return ScratchOffsets(
                cScore: reserve(sizes.cScore),
                uHist: reserve(sizes.uHist),
                candState: reserve(sizes.candState),
                candPools: reserve(sizes.candPools),
                candU: reserve(sizes.candU)
            )
        }

        func makeScratch(
            region: QWMRegion,
            maxBatchRows: Int,
            maxClusters: Int,
            candCap: Int
        ) -> QWCoarsePoolIndex.Scratch {
            QWCoarsePoolIndex.Scratch(
                cScore: QWTensorValue(region: region, offset: cScore,
                                      shape: QWTensorShape([maxBatchRows, maxClusters], .i32)),
                uHist: QWTensorValue(region: region, offset: uHist,
                                     shape: QWTensorShape([maxBatchRows, 512], .i32)),
                candState: QWTensorValue(region: region, offset: candState,
                                         shape: QWTensorShape([maxBatchRows * 2], .i32)),
                candPools: QWTensorValue(region: region, offset: candPools,
                                         shape: QWTensorShape([maxBatchRows, candCap], .i32)),
                candU: QWTensorValue(region: region, offset: candU,
                                     shape: QWTensorShape([maxBatchRows, candCap], .i32)),
                maxBatchRows: maxBatchRows,
                clusterStride: maxClusters,
                candCap: candCap
            )
        }
    }

    let variant: Variant
    let loaded: LoadedPipelines
    let scratch: QWCoarsePoolIndex.Scratch
    var watermarks: [Int]

    var pipelines: QWCoarsePoolIndex.Pipelines { loaded.coarse }
    var emeansTuple: EmeansPSOTuple { loaded.emeans }

    init(
        variant: Variant,
        loaded: LoadedPipelines,
        scratch: QWCoarsePoolIndex.Scratch,
        layerCount: Int
    ) {
        self.variant = variant
        self.loaded = loaded
        self.scratch = scratch
        self.watermarks = Array(repeating: 0, count: layerCount)
    }

    func resetWatermarks() {
        for i in watermarks.indices { watermarks[i] = 0 }
    }

    func truncatePosition(_ validPosition: Int, poolShift: Int) {
        let maxMacro = max(0, validPosition) / QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        for i in watermarks.indices where watermarks[i] > maxMacro {
            watermarks[i] = maxMacro
        }
    }

    /// Incrementally updates e-means coarse clusters whenever newly written tokens
    /// cross 4096-token macro boundaries; rewinds automatically on speculative rollback.
    func updateClusters(
        _ enc: any MTLComputeCommandEncoder,
        pooledKeys: QWTensorValue,
        csr: QWCoarsePoolIndex.LayerCSR,
        layerIndex: Int,
        pos0: Int,
        rows: Int,
        indexDim: Int,
        poolShift: Int = 2,
        postBarrier: Bool = false
    ) {
        guard layerIndex >= 0, layerIndex < watermarks.count else { return }
        let chunkTok = QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        let curMacro = pos0 / chunkTok
        if curMacro < watermarks[layerIndex] {
            watermarks[layerIndex] = curMacro
        }
        let targetMacros = min(csr.maxMacros, (pos0 + rows) / chunkTok)
        let prevMacros = watermarks[layerIndex]
        guard targetMacros > prevMacros else { return }
        QWCoarsePoolIndex.barrierIfConcurrent(enc)
        QWCoarsePoolIndex.encodeClusterMacros(
            enc, pso: pipelines.cluster1024, pooledKeys: pooledKeys,
            csr: csr, startMacro: prevMacros, numMacros: targetMacros - prevMacros, indexDim: indexDim)
        if postBarrier {
            QWCoarsePoolIndex.barrierIfConcurrent(enc)
        }
        watermarks[layerIndex] = targetMacros
    }

    func encodeDenseFinalize(
        _ enc: any MTLComputeCommandEncoder,
        blockList: QWTensorValue,
        blockCount: QWTensorValue,
        params: inout QWCoarsePoolIndex.SelectParams,
        batchUnits: Int
    ) {
        let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
        enc.setComputePipelineState(pipelines.finalize)
        blockList.bind(enc, index: 0)
        blockCount.bind(enc, index: 1)
        scratch.uHist.bind(enc, index: 2)
        scratch.candState.bind(enc, index: 3)
        scratch.candPools.bind(enc, index: 4)
        scratch.candU.bind(enc, index: 5)
        enc.setBytes(&params, length: plen, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: batchUnits, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dispatches optional coarse-cluster pruning followed by histogram/collect/score/finalize
    /// (or flat exact fallback when below the pruning threshold).
    func encodeSelection(
        _ enc: any MTLComputeCommandEncoder,
        query: QWTensorValue,
        idxWeights: QWTensorValue? = nil,
        pooledKeys: QWTensorValue,
        csr: QWCoarsePoolIndex.LayerCSR,
        blockList: QWTensorValue,
        blockCount: QWTensorValue,
        params: inout QWCoarsePoolIndex.SelectParams,
        batchUnits: Int,
        numClusters: Int,
        middleEnd: Int,
        shouldPrune: Bool
    ) {
        let tpg = MTLSize(width: 256, height: 1, depth: 1)
        let plen = MemoryLayout<QWCoarsePoolIndex.SelectParams>.stride
        let emeans = loaded.emeans
        let baseIdx = idxWeights != nil ? 1 : 0
        let coarseTile = idxWeights != nil ? 64 : 256
        if shouldPrune && numClusters > 0 {
            enc.setComputePipelineState(emeans.scoreCoarse)
            query.bind(enc, index: 0)
            if let idxWeights { idxWeights.bind(enc, index: 1) }
            csr.centroids.bind(enc, index: baseIdx + 1)
            csr.clusterSpans.bind(enc, index: baseIdx + 2)
            scratch.cScore.bind(enc, index: baseIdx + 3)
            scratch.uHist.bind(enc, index: baseIdx + 4)
            enc.setBytes(&params, length: plen, index: baseIdx + 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (numClusters + coarseTile - 1) / coarseTile, height: batchUnits, depth: 1),
                threadsPerThreadgroup: tpg)
        }
        QWCoarsePoolIndex.encodeHistCollectScoreFinalize(
            enc, pipelines: pipelines, csr: csr, scratch: scratch,
            blockList: blockList, blockCount: blockCount, params: &params,
            numClusters: shouldPrune ? numClusters : 0, maxMiddlePools: middleEnd,
            encodeScoreCands: { enc, maxCand, prm in
                enc.setComputePipelineState(emeans.scoreCands)
                query.bind(enc, index: 0)
                if let idxWeights { idxWeights.bind(enc, index: 1) }
                pooledKeys.bind(enc, index: baseIdx + 1)
                scratch.candState.bind(enc, index: baseIdx + 2)
                scratch.candPools.bind(enc, index: baseIdx + 3)
                scratch.candU.bind(enc, index: baseIdx + 4)
                enc.setBytes(&prm, length: plen, index: baseIdx + 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: (maxCand + 255) / 256, height: batchUnits, depth: 1),
                    threadsPerThreadgroup: tpg)
            },
            encodeFlatFallback: { enc, prm in
                enc.setComputePipelineState(emeans.flatFallback)
                query.bind(enc, index: 0)
                if let idxWeights { idxWeights.bind(enc, index: 1) }
                pooledKeys.bind(enc, index: baseIdx + 1)
                scratch.candState.bind(enc, index: baseIdx + 2)
                blockList.bind(enc, index: baseIdx + 3)
                blockCount.bind(enc, index: baseIdx + 4)
                enc.setBytes(&prm, length: plen, index: baseIdx + 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: batchUnits, height: 1, depth: 1),
                    threadsPerThreadgroup: tpg)
            }
        )
    }
}
