// QWLSAOperator.swift — Latent Sparse Attention (arXiv:2610.42147) GPU operator
// for QWEngineDriver (all-attention, hybrid GDN+Attention, and looped mHC models).
//
// Manages per-full-attention-layer LSA projection matrices (`wq_idx`, `wk_idx`),
// pooled 4-token key cache, dynamic 512-token Ward cluster centroids/members/sizes,
// and shared block-selection scratch in a single preallocated Metal region
// (preserving zero steady-state allocations).
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWQuant

public final class QWLSAOperator: @unchecked Sendable {
    public struct LayerWeights: Sendable {
        public let physicalLayer: Int
        public let wqIdxF16: [UInt16]
        public let wkIdxF16: [UInt16]
        public init(physicalLayer: Int, wqIdxF16: [UInt16], wkIdxF16: [UInt16]) {
            self.physicalLayer = physicalLayer
            self.wqIdxF16 = wqIdxF16
            self.wkIdxF16 = wkIdxF16
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let wqIdx: QWTensorValue
        public let wkIdx: QWTensorValue
        public let kPending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let centroids: QWTensorValue
        public let bucketMembers: QWTensorValue
        public let bucketSizes: QWTensorValue
    }

    public static let maxSelectBatchRows = 32
    public static let emeansProbeClusters = 320

    public let config: QWLSAConfig
    public let isEmeansCoarse: Bool
    public let geometry: QWModelGeometry
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let indexDim: Int
    public let sliceStart: Int
    public let sliceDim: Int
    public let inputDim: Int
    public let ringCap: Int
    public let maxBlocks: Int
    public let maxSuperblocks: Int
    public let maxBuckets: Int
    public let listStride: Int
    public let region: QWMRegion
    public let layers: [LayerState]
    public let loopReusePlan: QWLSALoopReusePlan
    /// Session whose indexer pools and Ward clusters the layers currently hold.
    public weak var stateOwner: AnyObject?

    public let qIdxScratch: QWTensorValue
    public let scoreScratch: QWTensorValue
    public let blockListScratch: QWTensorValue
    public let blockCountScratch: QWTensorValue
    let reuseBlockLists: [QWTensorValue]
    let reuseBlockCounts: [QWTensorValue]

    let hal: QWHal
    let psoIndexPrep: any MTLComputePipelineState
    let psoIndexPrepBF16: any MTLComputePipelineState
    let psoPoolKeys: any MTLComputePipelineState
    let psoWardCluster: any MTLComputePipelineState
    let psoSelectBlocks: any MTLComputePipelineState
    let psoSelectScore: any MTLComputePipelineState
    let psoSelectHist2: any MTLComputePipelineState
    let psoSelectExpand: any MTLComputePipelineState
    let psoSelectFinalize: any MTLComputePipelineState
    let coarsePipeline: QWCoarseSelectionPipeline?

    /// Parallel selector scratch (zero at rest; restored by `qwg_lsa_select_finalize`).
    let selCScore: QWTensorValue
    let selHist: QWTensorValue
    let selCount: QWTensorValue
    let selMarks: QWTensorValue
    let selCandM: QWTensorValue
    let selCandU: QWTensorValue
    let selMarkStride: Int
    static let selectCandidateCap = 16384
    static let deltaMagic: UInt64 = 0x31544C4441534C51 // "QLSADLT1"
    /// Route centroid-filter-regime selections through the multi-threadgroup selector
    /// (bit-identical to `qwg_lsa_select_blocks`). Disabled only by parity tests.
    var useParallelSelect = true
    var pinnedBaseData: Data? = nil
    var pinnedBaseFullPosition: Int = 0
    var pinnedBasePos512: Int = 0
    var residentPos512: Int = 0
    var clusterUnitTokens: Int { isEmeansCoarse ? QWCoarsePoolIndex.macroChunkTokens : 512 }
    var clusterMemBytesPerUnit: Int { isEmeansCoarse ? (QWCoarsePoolIndex.macroChunkPools * 2) : 256 }
    var clusterSizeBytesPerUnit: Int { isEmeansCoarse ? (QWCoarsePoolIndex.clustersPerMacro * 4) : 64 }

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                config: QWLSAConfig,
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                layerWeights: [LayerWeights],
                maxContext: Int,
                maxPrefillChunk: Int = 512) throws {
        self.hal = hal
        self.geometry = geometry
        self.config = config
        let emeans = (config.alignedClusterBlock == QWLSAConfig.emeansClusterBlock)
        self.isEmeansCoarse = emeans
        self.maxContext = max(512, maxContext)
        self.maxPrefillChunk = max(32, maxPrefillChunk)
        self.indexDim = indexDim
        let inDim = geometry.lsaInputDim
        self.sliceStart = geometry.lsaHeadSliceStart
        self.sliceDim = geometry.lsaHeadSliceDim
        self.inputDim = inDim

        var rc = 4096
        while rc < self.maxPrefillChunk * 2 { rc <<= 1 }
        let ringCap = rc
        let maxBlocks = max(128, (self.maxContext + 3) / 4)
        let maxSuperblocks = emeans
            ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext)
            : max(1, (self.maxContext + 511) / 512)
        let maxBuckets = maxSuperblocks * 64
        let listStride = max(512, config.maxTotalSelectedBlocks)
        self.ringCap = ringCap
        self.maxBlocks = maxBlocks
        self.maxSuperblocks = maxSuperblocks
        self.maxBuckets = maxBuckets
        self.listStride = listStride

        self.psoIndexPrep = try hal.pipeline("qwg_lsa_index_prep")
        self.psoIndexPrepBF16 = try hal.pipeline("qwg_lsa_index_prep_bf16")
        self.psoPoolKeys = try hal.pipeline("qwg_lsa_pool_keys")
        self.psoWardCluster = try hal.pipeline("qwg_lsa_ward_cluster_512")
        self.psoSelectBlocks = try hal.pipeline("qwg_lsa_select_blocks")
        self.psoSelectScore = try hal.pipeline("qwg_lsa_select_score")
        self.psoSelectHist2 = try hal.pipeline("qwg_lsa_select_hist2")
        self.psoSelectExpand = try hal.pipeline("qwg_lsa_select_expand")
        self.psoSelectFinalize = try hal.pipeline("qwg_lsa_select_finalize")
        let loadedCoarse = emeans
            ? try QWCoarseSelectionPipeline.LoadedPipelines.load(hal: hal, variant: .lsa)
            : nil

        let nAttnLayers = layerWeights.count
        let reusePlan = QWLSALoopReusePlan.build(
            physicalLayers: layerWeights.map(\.physicalLayer),
            enabled: config.crossLoopReuse,
            stride: config.loopShareStride
        )
        self.loopReusePlan = reusePlan

        let layout = Self.buildLayoutPlan(
            inDim: inDim, indexDim: indexDim, ringCap: ringCap,
            maxBlocks: maxBlocks, maxSuperblocks: maxSuperblocks, maxBuckets: maxBuckets,
            listStride: listStride, maxPrefillChunk: self.maxPrefillChunk,
            emeans: emeans, physicalLayers: layerWeights.map(\.physicalLayer), reusePlan: reusePlan
        )
        let layerOffsets = layout.layerOffsets

        let totalBytes = layout.totalBytes
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "lsa-cache", asked: totalBytes, free: 0)
        }
        buf.label = "lsa-cache"
        memset(buf.contents(), 0, totalBytes)
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(nAttnLayers)
        let basePtr = reg.pointer
        let wBytes = indexDim * inDim * 2
        for i in layout.uniqueWeightIndices {
            let offs = layerOffsets[i]
            let lw = layerWeights[i]
            lw.wqIdxF16.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                if let ptr = src.baseAddress {
                    memcpy(basePtr.advanced(by: offs.wqOff), ptr, min(src.count, wBytes))
                }
            }
            lw.wkIdxF16.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                if let ptr = src.baseAddress {
                    memcpy(basePtr.advanced(by: offs.wkOff), ptr, min(src.count, wBytes))
                }
            }
        }
        for i in 0..<nAttnLayers {
            let offs = layerOffsets[i]
            builtLayers.append(LayerState(
                fullAttentionIndex: i,
                wqIdx: QWTensorValue(region: reg, offset: offs.wqOff, shape: QWTensorShape([indexDim, inDim], .f16)),
                wkIdx: QWTensorValue(region: reg, offset: offs.wkOff, shape: QWTensorShape([indexDim, inDim], .f16)),
                kPending: QWTensorValue(region: reg, offset: offs.kPendOff, shape: QWTensorShape([ringCap, indexDim], .f16)),
                pooledKeys: QWTensorValue(region: reg, offset: offs.pooledOff, shape: QWTensorShape([maxBlocks, indexDim], .f16)),
                centroids: QWTensorValue(region: reg, offset: offs.centOff, shape: QWTensorShape([maxBuckets, indexDim], .f16)),
                bucketMembers: QWTensorValue(region: reg, offset: offs.memOff,
                                             shape: QWTensorShape([emeans ? maxSuperblocks * 512 : maxBuckets], .i32)),
                bucketSizes: QWTensorValue(region: reg, offset: offs.sizeOff,
                                           shape: QWTensorShape([emeans ? maxBuckets : max(1, maxBuckets / 4)], .i32))
            ))
        }
        self.layers = builtLayers
        let maxChunk = self.maxPrefillChunk
        self.reuseBlockLists = layout.reuseOffsets.map {
            QWTensorValue(region: reg, offset: $0.listOff, shape: QWTensorShape([maxChunk, listStride], .i32))
        }
        self.reuseBlockCounts = layout.reuseOffsets.map {
            QWTensorValue(region: reg, offset: $0.countOff, shape: QWTensorShape([maxChunk], .i32))
        }

        self.qIdxScratch = QWTensorValue(
            region: reg, offset: layout.qIdxOff,
            shape: QWTensorShape([self.maxPrefillChunk, indexDim], .f16))
        self.scoreScratch = QWTensorValue(
            region: reg, offset: layout.scoreOff,
            shape: QWTensorShape([layout.scoreRows, layout.scoreCols], .i32))
        self.blockListScratch = QWTensorValue(
            region: reg, offset: layout.listOff,
            shape: QWTensorShape([self.maxPrefillChunk, listStride], .i32))
        self.blockCountScratch = QWTensorValue(
            region: reg, offset: layout.countOff,
            shape: QWTensorShape([self.maxPrefillChunk], .i32))
        self.selMarkStride = layout.markStride
        self.selCScore = QWTensorValue(region: reg, offset: layout.selCScoreOff,
                                       shape: QWTensorShape([layout.selectBatch, maxBuckets], .i32))
        self.selHist = QWTensorValue(region: reg, offset: layout.selHistOff,
                                     shape: QWTensorShape([layout.selectBatch, 512], .i32))
        self.selCount = QWTensorValue(region: reg, offset: layout.selCountOff,
                                      shape: QWTensorShape([layout.selectBatch * layout.selCountCols], .i32))
        self.selMarks = QWTensorValue(region: reg, offset: layout.selMarksOff,
                                      shape: QWTensorShape([layout.selectBatch, layout.markStride], .i32))
        self.selCandM = QWTensorValue(region: reg, offset: layout.selCandMOff,
                                      shape: QWTensorShape([layout.selectBatch, layout.candCap], .i32))
        let candUVal = QWTensorValue(region: reg, offset: layout.selCandUOff,
                                     shape: QWTensorShape([layout.selectBatch, layout.candCap], .i32))
        self.selCandU = candUVal
        if let loadedCoarse {
            let scratch = QWCoarsePoolIndex.Scratch(
                cScore: selCScore, uHist: selHist, candState: selCount,
                candPools: selCandM, candU: candUVal,
                maxBatchRows: Self.maxSelectBatchRows, clusterStride: maxBuckets, candCap: layout.candCap)
            self.coarsePipeline = QWCoarseSelectionPipeline(
                variant: .lsa, loaded: loadedCoarse, scratch: scratch, layerCount: nAttnLayers)
        } else {
            self.coarsePipeline = nil
        }
    }

    struct LayerOffsets {
        let wqOff: Int
        let wkOff: Int
        let kPendOff: Int
        let pooledOff: Int
        let centOff: Int
        let memOff: Int
        let sizeOff: Int
    }

    struct LayoutPlan {
        let layerOffsets: [LayerOffsets]
        let uniqueWeightIndices: [Int]
        let reuseOffsets: [(listOff: Int, countOff: Int)]
        let qIdxOff: Int
        let scoreOff: Int
        let scoreRows: Int
        let scoreCols: Int
        let listOff: Int
        let countOff: Int
        let selectBatch: Int
        let markStride: Int
        let selCountCols: Int
        let candCap: Int
        let selCScoreOff: Int
        let selHistOff: Int
        let selCountOff: Int
        let selMarksOff: Int
        let selCandMOff: Int
        let selCandUOff: Int
        let totalBytes: Int
    }

    static func buildLayoutPlan(
        inDim: Int, indexDim: Int, ringCap: Int,
        maxBlocks: Int, maxSuperblocks: Int, maxBuckets: Int,
        listStride: Int, maxPrefillChunk: Int, emeans: Bool,
        physicalLayers: [Int], reusePlan: QWLSALoopReusePlan
    ) -> LayoutPlan {
        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }
        var physWeightOffs: [Int: (wqOff: Int, wkOff: Int)] = [:]
        var uniqueWeightIndices: [Int] = []
        var layerOffsets = Array(repeating: LayerOffsets(
            wqOff: 0, wkOff: 0, kPendOff: 0, pooledOff: 0, centOff: 0, memOff: 0, sizeOff: 0
        ), count: physicalLayers.count)
        for (i, physLayer) in physicalLayers.enumerated() {
            let leader = reusePlan.sourceForFullAttnLayer[i]
            if leader != i {
                layerOffsets[i] = layerOffsets[leader]
                continue
            }
            let wPair: (wqOff: Int, wkOff: Int)
            if let existing = physWeightOffs[physLayer] {
                wPair = existing
            } else {
                wPair = (wqOff: reserve(indexDim * inDim * 2), wkOff: reserve(indexDim * inDim * 2))
                physWeightOffs[physLayer] = wPair
                uniqueWeightIndices.append(i)
            }
            layerOffsets[i] = LayerOffsets(
                wqOff: wPair.wqOff, wkOff: wPair.wkOff,
                kPendOff: reserve(ringCap * indexDim * 2),
                pooledOff: reserve(maxBlocks * indexDim * 2),
                centOff: reserve(maxBuckets * indexDim * 2),
                memOff: reserve(emeans ? maxSuperblocks * 2048 : maxBuckets * 4),
                sizeOff: reserve(emeans ? maxBuckets * 4 : maxBuckets)
            )
        }
        var reuseOffsets: [(listOff: Int, countOff: Int)] = []
        reuseOffsets.reserveCapacity(reusePlan.reuseSlotCount)
        for _ in 0..<reusePlan.reuseSlotCount {
            reuseOffsets.append((
                listOff: reserve(maxPrefillChunk * listStride * 4),
                countOff: reserve(maxPrefillChunk * 4)
            ))
        }
        let selectBatch = min(maxPrefillChunk, Self.maxSelectBatchRows)
        let qIdxOff = reserve(maxPrefillChunk * indexDim * 2)
        let scoreRows = emeans ? 1 : selectBatch
        let scoreCols = emeans ? 64 : maxBlocks
        let scoreOff = reserve(scoreRows * scoreCols * 4)
        let listOff = reserve(maxPrefillChunk * listStride * 4)
        let countOff = reserve(maxPrefillChunk * 4)
        let markStride = emeans ? 1 : ((maxBlocks + 31) / 32)
        let candCap = Self.selectCandidateCap
        let selCScoreOff = reserve(selectBatch * maxBuckets * 4)
        let selHistOff = reserve(selectBatch * 512 * 4)
        let selCountCols = emeans ? 2 : 1
        let selCountOff = reserve(selectBatch * selCountCols * 4)
        let selMarksOff = reserve(selectBatch * markStride * 4)
        let selCandMOff = reserve(selectBatch * candCap * 4)
        let selCandUOff = reserve(selectBatch * candCap * 4)
        return LayoutPlan(
            layerOffsets: layerOffsets, uniqueWeightIndices: uniqueWeightIndices,
            reuseOffsets: reuseOffsets, qIdxOff: qIdxOff, scoreOff: scoreOff,
            scoreRows: scoreRows, scoreCols: scoreCols, listOff: listOff, countOff: countOff,
            selectBatch: selectBatch, markStride: markStride, selCountCols: selCountCols,
            candCap: candCap, selCScoreOff: selCScoreOff, selHistOff: selHistOff,
            selCountOff: selCountOff, selMarksOff: selMarksOff,
            selCandMOff: selCandMOff, selCandUOff: selCandUOff,
            totalBytes: max(4096, cursor)
        )
    }

    /// Estimates total Metal region bytes required for LSA at `(maxContext, maxPrefillChunk)`.
    public static func estimateRegionBytes(
        geometry: QWModelGeometry,
        lsaConfig: QWLSAConfig,
        attnLayers: [Int],
        maxContext: Int,
        maxPrefillChunk: Int,
        indexDim: Int = QWLSAConfig.defaultIndexDim
    ) -> Int {
        guard lsaConfig.isEnabled, !attnLayers.isEmpty else { return 0 }
        let effCfg = lsaConfig.resolvingLoopShareEnvironment()
        let emeans = (effCfg.alignedClusterBlock == QWLSAConfig.emeansClusterBlock)
        let clampedCtx = max(512, maxContext)
        let chunk = max(32, maxPrefillChunk)
        var rc = 4096
        while rc < chunk * 2 { rc <<= 1 }
        let maxBlocks = max(128, (clampedCtx + 3) / 4)
        let maxSuperblocks = emeans
            ? QWCoarsePoolIndex.maxMacros(forMaxContext: clampedCtx)
            : max(1, (clampedCtx + 511) / 512)
        let plan = geometry.executionPlan
        let physLayers = attnLayers.map { plan.indices.contains($0) ? plan[$0] : $0 }
        let reusePlan = QWLSALoopReusePlan.build(
            physicalLayers: physLayers, enabled: effCfg.crossLoopReuse, stride: effCfg.loopShareStride
        )
        let layout = buildLayoutPlan(
            inDim: geometry.lsaInputDim, indexDim: indexDim, ringCap: rc,
            maxBlocks: maxBlocks, maxSuperblocks: maxSuperblocks, maxBuckets: maxSuperblocks * 64,
            listStride: max(512, effCfg.maxTotalSelectedBlocks), maxPrefillChunk: chunk,
            emeans: emeans, physicalLayers: physLayers, reusePlan: reusePlan
        )
        let coarseExtra = emeans ? (maxSuperblocks * 4 * physLayers.count) : 0
        return layout.totalBytes + coarseExtra
    }

    /// True when `fullAttnLayer` reuses its leader's LSA block list instead of indexing/selecting.
    public func isSelectionFollower(fullAttnLayer: Int) -> Bool {
        guard fullAttnLayer >= 0, fullAttnLayer < loopReusePlan.isFollower.count else { return false }
        return loopReusePlan.isFollower[fullAttnLayer]
    }

    /// Leader `fullAttnLayer` index whose LSA state and block selections serve `fullAttnLayer`.
    public func selectionSourceLayer(forFullAttnLayer fullAttnLayer: Int) -> Int {
        guard fullAttnLayer >= 0, fullAttnLayer < loopReusePlan.sourceForFullAttnLayer.count else {
            return fullAttnLayer
        }
        return loopReusePlan.sourceForFullAttnLayer[fullAttnLayer]
    }

    /// Selected block list tensor to bind for `fullAttnLayer` (leader cache if shared, else scratch).
    public func blockList(forFullAttnLayer fullAttnLayer: Int) -> QWTensorValue {
        let src = selectionSourceLayer(forFullAttnLayer: fullAttnLayer)
        if let slot = loopReusePlan.reuseCacheSlotForLeader[src], slot < reuseBlockLists.count {
            return reuseBlockLists[slot]
        }
        return blockListScratch
    }

    /// Selected block count tensor to bind for `fullAttnLayer` (leader cache if shared, else scratch).
    public func blockCount(forFullAttnLayer fullAttnLayer: Int) -> QWTensorValue {
        let src = selectionSourceLayer(forFullAttnLayer: fullAttnLayer)
        if let slot = loopReusePlan.reuseCacheSlotForLeader[src], slot < reuseBlockCounts.count {
            return reuseBlockCounts[slot]
        }
        return blockCountScratch
    }

    /// Mirrors `LSASelParams` in qwlsa.metal.
    struct SelectParams {
        var T: UInt32
        var pos0: UInt32
        var indexDim: UInt32
        var budgetBlocks: UInt32
        var windowBlocks: UInt32
        var listStride: UInt32
        var scoreStride: UInt32
        var rowStep: UInt32
        var maxRow: UInt32
        var bucketStride: UInt32
        var candCap: UInt32
        var markStride: UInt32
    }

    /// Centroid-bucket count of the last (largest-position) unit when it is in the
    /// centroid-filter regime of the selector, else nil. Mirrors `lsa_sel_geom`.
    func filterBuckets(pos0: Int, units: Int, rowStep: Int, maxRow: Int) -> Int? {
        let lastRow = rowStep > 1 ? min((units - 1) * rowStep + rowStep - 1, maxRow - 1) : units - 1
        let totalBlocks = (pos0 + lastRow + 1) >> 2
        let budget = config.budgetBlocks
        let window = QWLSAConfig.windowTokens / QWLSAConfig.poolSize
        guard totalBlocks > 1 + budget + window else { return nil }
        let middleEnd = totalBlocks - window
        let kEff = min(budget, middleEnd - 1)
        let numBuckets = (middleEnd >> 7) * 64
        return numBuckets > max(256, kEff * 2) ? numBuckets : nil
    }

    /// Encode index Q/K projection (`qwg_lsa_index_prep`), 4-token key pooling (`qwg_lsa_pool_keys`),
    /// and 512-token dynamic Ward clustering (`qwg_lsa_ward_cluster_512`) for `rows` tokens starting at `pos0`.
}
