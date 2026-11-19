// QWHCAAttnOperator.swift — Hybrid Compressed Attention (HCA) GPU operator:
// per-HCA-layer SWA ring, 128:1 channel-wise softmax compressor cache, optional
// Weight-SVD LSA + e-means coarse pool index (>262K tokens), Split-K decode MQA,
// fused 8x8 MMA VHA + inverse YaRN RoPE, and 3D-batched grouped low-rank wo_a.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWQuant
import QWGEMM

public final class QWHCAAttnOperator: @unchecked Sendable {
    public let geometry: QWModelGeometry
    public let hcaConfig: QWModelDescriptor.HCAConfig
    public let lsaConfig: QWLSAConfig
    public let maxContext: Int
    public let maxPrefillChunk: Int
    public let qHeads: Int
    public let headDim: Int
    public let nopeDim: Int
    public let ropeDim: Int
    public let qLoraRank: Int
    public let oGroups: Int
    public let oLoraRank: Int
    public let groupIn: Int
    public let compressRatio: Int
    public let poolShift: Int
    public let slidingWindow: Int
    public let indexDim: Int
    public let lsaBudgetBlocks: Int
    public let compBudget: Int
    public let listStride: Int
    public let ringCap: Int
    public let maxCompBlocks: Int
    public let selectBatchRows: Int
    public let inStride: Int
    public let qUpStride: Int
    public let oGateOffset: Int
    public let zOutStride: Int
    public let ropeMscale: Float
    public let attnSplits: Int
    public let lsaAllocated: Bool
    public var emeansCoarseEnabled: Bool

    public let region: QWMRegion
    public let layers: [LayerState]
    public weak var stateOwner: AnyObject? {
        didSet {
            if stateOwner !== oldValue { resetCoarseWatermarks() }
        }
    }

    public let fusedInScratch: QWTensorValue
    public let qrScratch: QWTensorValue
    public let qUpGateScratch: QWTensorValue
    public let attScratch: QWTensorValue
    public let zVhaScratch: QWTensorValue
    public let zGatedScratch: QWTensorValue
    public let invFreqBuf: QWTensorValue
    public let zeroSinkBuf: QWTensorValue
    internal let partOScratch: QWTensorValue
    internal let partMLScratch: QWTensorValue
    public let qIdxScratch: QWTensorValue?
    public let blockListScratch: QWTensorValue?
    public let blockCountScratch: QWTensorValue?
    internal let coarsePipeline: QWCoarseSelectionPipeline?
    private var emptyClusteredMacros: [Int]
    internal var coarseScratch: QWCoarsePoolIndex.Scratch? { coarsePipeline?.scratch }

    internal let hal: QWHal
    internal let psoInPrep: any MTLComputePipelineState
    internal let psoCompressBlocks: any MTLComputePipelineState
    internal let psoQPrep: any MTLComputePipelineState
    internal let psoLsaQPrep: any MTLComputePipelineState
    internal let psoAttnMMA: any MTLComputePipelineState
    internal let psoAttnNAX: (any MTLComputePipelineState)?
    internal let psoAttnNAXPrefill: (any MTLComputePipelineState)?
    internal let psoAttnCombine: any MTLComputePipelineState
    internal let psoVhaInvRope: any MTLComputePipelineState
    internal let psoGroupedWoAGemv: any MTLComputePipelineState
    internal let psoGroupedWoAGemvSub2: any MTLComputePipelineState
    internal let psoGroupedWoAGemvMR: any MTLComputePipelineState
    internal let psoGroupedWoAGemvMRSub2: any MTLComputePipelineState
    internal let psoGroupedWoAMMA: any MTLComputePipelineState
    internal let psoGroupedWoAMMASub2: any MTLComputePipelineState
    internal let psoGroupedWoANAX: (any MTLComputePipelineState)?
    internal let psoGroupedWoANAXSub2: (any MTLComputePipelineState)?
    internal var coarsePipelines: QWCoarsePoolIndex.Pipelines? { coarsePipeline?.pipelines }
    internal var psoEmeansLsa: QWCoarseSelectionPipeline.EmeansPSOTuple? { coarsePipeline?.emeansTuple }
    internal var clusteredMacros: [Int] {
        get { coarsePipeline?.watermarks ?? emptyClusteredMacros }
        set {
            if let coarsePipeline { coarsePipeline.watermarks = newValue }
            else { emptyClusteredMacros = newValue }
        }
    }

    public func resetCoarseWatermarks() {
        coarsePipeline?.resetWatermarks()
        for i in emptyClusteredMacros.indices { emptyClusteredMacros[i] = 0 }
    }

    public func truncatePosition(_ validPosition: Int) {
        if let coarsePipeline {
            coarsePipeline.truncatePosition(validPosition, poolShift: poolShift)
            return
        }
        let maxMacro = max(0, validPosition) / QWCoarsePoolIndex.macroChunkTokens(poolShift: poolShift)
        for i in emptyClusteredMacros.indices where emptyClusteredMacros[i] > maxMacro {
            emptyClusteredMacros[i] = maxMacro
        }
    }

    internal struct InPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qLoraRank: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var compressRatio: UInt32
        var ringMask: UInt32
        var rmsEps: Float
        var mscale: Float
    }

    internal struct CompressParams {
        var startBlock: UInt32
        var numBlocks: UInt32
        var compressRatio: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var ringMask: UInt32
        var indexDim: UInt32
        var rmsEps: Float
        var mscale: Float
        var hasLSA: UInt32
    }

    internal struct QPrepParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var qUpStride: UInt32
        var indexDim: UInt32
        var rmsEps: Float
        var mscale: Float
    }

    internal struct AttnParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var qUpStride: UInt32
        var compressRatio: UInt32
        var slidingWindow: UInt32
        var ringMask: UInt32
        var compBudget: UInt32
        var listStride: UInt32
        var useBlockList: UInt32
        var hasSink: UInt32
    }

    internal struct VhaInvRopeParams {
        var T: UInt32
        var pos0: UInt32
        var qHeads: UInt32
        var headDim: UInt32
        var nopeDim: UInt32
        var ropeDim: UInt32
        var invMscale: Float
        var hasVHA: UInt32
    }

    internal struct GroupedWoAParams {
        var T: UInt32
        var oGroups: UInt32
        var oLoraRank: UInt32
        var groupIn: UInt32
        var qUpStride: UInt32
        var oGateOffset: UInt32
        var bits: UInt32
        var group: UInt32
        var isIncumbent: UInt32
    }

    public static func makeIfEnabled(hal: QWHal,
                                     geometry: QWModelGeometry,
                                     hcaLayers: [Int],
                                     maxContext: Int,
                                     maxPrefillChunk: Int = 512,
                                     lsaConfig: QWLSAConfig = QWLSAConfig(),
                                     indexDim: Int = QWLSAConfig.defaultIndexDim,
                                     extraHcaLayers: Int = 0) throws -> QWHCAAttnOperator? {
        guard geometry.hcaConfig != nil, !hcaLayers.isEmpty else { return nil }
        return try QWHCAAttnOperator(
            hal: hal,
            geometry: geometry,
            numHcaLayers: hcaLayers.count + max(0, extraHcaLayers),
            maxContext: maxContext,
            maxPrefillChunk: maxPrefillChunk,
            lsaConfig: lsaConfig,
            indexDim: indexDim
        )
    }

    public init(hal: QWHal,
                geometry: QWModelGeometry,
                numHcaLayers: Int,
                maxContext: Int,
                maxPrefillChunk: Int = 512,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                indexDim: Int = QWLSAConfig.defaultIndexDim,
                lsaBudgetBlocksOverride: Int? = nil) throws {
        guard let hca = geometry.hcaConfig else {
            throw QWAllocError.outOfSpace(region: "hca-cache", asked: 0, free: 0)
        }
        self.hal = hal
        self.geometry = geometry
        self.hcaConfig = hca
        self.lsaConfig = lsaConfig
        self.maxContext = max(64, maxContext)
        self.maxPrefillChunk = max(16, maxPrefillChunk)
        self.qHeads = hca.qHeads
        self.headDim = hca.headDim
        self.nopeDim = hca.qkNopeHeadDim
        self.ropeDim = hca.qkRopeHeadDim
        self.qLoraRank = hca.qLoraRank
        self.oGroups = max(1, hca.oGroups)
        self.oLoraRank = hca.oLoraRank
        self.groupIn = (self.qHeads / self.oGroups) * self.headDim
        self.compressRatio = max(1, hca.defaultCompressRatio)
        self.poolShift = self.compressRatio > 1 ? self.compressRatio.trailingZeroBitCount : 0
        self.slidingWindow = max(1, hca.slidingWindow)
        self.indexDim = max(16, min(64, indexDim))

        let defaultBlkBudget = min(QWCoarsePoolIndex.maxBudgetBlocks,
                                   max(1, lsaConfig.isEnabled ? lsaConfig.budgetBlocks : 2048))
        let blkBudget = min(QWCoarsePoolIndex.maxBudgetBlocks,
                            max(1, lsaBudgetBlocksOverride ?? defaultBlkBudget))
        self.lsaBudgetBlocks = blkBudget
        self.compBudget = 1 + blkBudget
        self.listStride = max(64, ((self.compBudget + 63) & ~63))

        let minRing = max(self.slidingWindow, self.compressRatio) + max(32, self.maxPrefillChunk)
        var rc = 256
        while rc < minRing { rc <<= 1 }
        self.ringCap = rc
        self.maxCompBlocks = max(16, (self.maxContext + self.compressRatio - 1) / self.compressRatio)
        self.selectBatchRows = min(self.maxPrefillChunk, Self.maxSelectBatchRows)
        self.inStride = self.qLoraRank + 3 * self.headDim
        self.oGateOffset = self.qHeads * self.headDim
        self.qUpStride = self.oGateOffset + self.oGroups * self.oLoraRank
        self.zOutStride = self.oGroups * self.oLoraRank

        let (invFreqArr, mscale) = QWModelDescriptor.computeRopeInvFreq(
            rotaryDim: self.ropeDim,
            base: hca.compressRopeTheta,
            yarn: hca.yarn
        )
        self.ropeMscale = mscale

        let splits = Int(ProcessInfo.processInfo.environment["QW_HCA_ATTN_SPLIT"] ?? "") ?? 8
        self.attnSplits = (self.headDim == 512 && self.qHeads % 16 == 0) ? max(1, min(32, splits)) : 1

        self.psoInPrep = try hal.pipeline("qwg_hca_in_prep")
        self.psoCompressBlocks = try hal.pipeline("qwg_hca_compress_blocks")
        self.psoQPrep = try hal.pipeline("qwg_hca_q_prep")
        self.psoLsaQPrep = try hal.pipeline("qwg_hca_lsa_q_prep")
        self.psoAttnMMA = try hal.pipeline("qwg_hca_attn_mma")
        self.psoAttnNAX = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_hca_attn_nax") : nil
        self.psoAttnNAXPrefill = hal.caps.isMetal4TensorCapable ? try hal.pipeline("qwg_hca_attn_nax_prefill") : nil
        self.psoAttnCombine = try hal.pipeline("qwg_hca_attn_combine")
        self.psoVhaInvRope = try hal.pipeline("qwg_hca_vha_invrope")
        self.psoGroupedWoAGemv = try hal.pipeline("qwg_hca_grouped_wo_a_gemv")
        self.psoGroupedWoAGemvSub2 = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_sub2")
        self.psoGroupedWoAGemvMR = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_mr")
        self.psoGroupedWoAGemvMRSub2 = try hal.pipeline("qwg_hca_grouped_wo_a_gemv_mr_sub2")
        self.psoGroupedWoAMMA = try hal.pipeline("qwg_hca_grouped_wo_a_mma")
        self.psoGroupedWoAMMASub2 = try hal.pipeline("qwg_hca_grouped_wo_a_mma_sub2")
        self.psoGroupedWoANAX = hal.caps.isMetal4TensorCapable
            ? try hal.pipeline("qwg_hca_grouped_wo_a_nax") : nil
        self.psoGroupedWoANAXSub2 = hal.caps.isMetal4TensorCapable
            ? try hal.pipeline("qwg_hca_grouped_wo_a_nax_sub2") : nil

        let needLsa = (lsaConfig.mode != .off)
            && (self.compressRatio.nonzeroBitCount == 1)
            && (self.maxCompBlocks > self.compBudget)
        self.lsaAllocated = needLsa
        let loadedCoarse = needLsa
            ? try QWCoarseSelectionPipeline.LoadedPipelines.load(hal: hal, variant: .lsa)
            : nil
        if let rawEnv = ProcessInfo.processInfo.environment["QW_HCA_EMEANS"]?.lowercased() {
            self.emeansCoarseEnabled = needLsa && rawEnv != "0" && rawEnv != "false" && rawEnv != "off"
        } else {
            self.emeansCoarseEnabled = needLsa
        }
        self.emptyClusteredMacros = Array(repeating: 0, count: numHcaLayers)

        var cursor = 0
        func align256(_ x: Int) -> Int { (x + 255) & ~255 }
        func reserve(_ bytes: Int) -> Int {
            let off = cursor
            cursor = align256(cursor + max(256, bytes))
            return off
        }

        let maxMacros = needLsa
            ? QWCoarsePoolIndex.maxMacros(forMaxContext: self.maxContext, poolShift: self.poolShift)
            : 0
        let paddedCompBlocks = (self.maxCompBlocks + 7) & ~7

        struct LayerOffs {
            let swa: Int
            let kvPend: Int
            let scPend: Int
            let comp: Int
            let pooled: Int
            let csr: QWCoarseSelectionPipeline.CSROffsets
        }
        var layerOffs: [LayerOffs] = []
        layerOffs.reserveCapacity(numHcaLayers)
        for _ in 0..<numHcaLayers {
            let swaOff = reserve(self.ringCap * self.headDim * 2)
            let kvPendOff = reserve(self.ringCap * self.headDim * 4)
            let scPendOff = reserve(self.ringCap * self.headDim * 4)
            let compOff = reserve(paddedCompBlocks * self.headDim * 2)
            let pOff = needLsa ? reserve(paddedCompBlocks * self.indexDim * 2) : 0
            let csr = QWCoarseSelectionPipeline.CSROffsets.reserve(
                enabled: needLsa, maxMacros: maxMacros, indexDim: self.indexDim, reserve: reserve)
            layerOffs.append(LayerOffs(
                swa: swaOff, kvPend: kvPendOff, scPend: scPendOff, comp: compOff,
                pooled: pOff, csr: csr
            ))
        }

        let maxT = self.maxPrefillChunk
        let fusedInOff = reserve(maxT * self.inStride * 2)
        let qrOff = reserve(maxT * self.qLoraRank * 2)
        let qUpOff = reserve(maxT * self.qUpStride * 2)
        let attOff = reserve(maxT * self.qHeads * self.headDim * 2)
        let zVhaOff = reserve(maxT * self.qHeads * self.headDim * 2)
        let zGatedOff = reserve(maxT * self.zOutStride * 2)
        let invFreqCount = max(1, invFreqArr.count)
        let invFreqOff = reserve(invFreqCount * 4)
        let zeroSinkCount = max(16, self.qHeads)
        let zeroSinkOff = reserve(zeroSinkCount * 4)
        let splitRows = min(maxT, Self.splitMaxRows) * max(1, self.attnSplits) * self.qHeads
        let partOOff = reserve(self.attnSplits > 1 ? splitRows * self.headDim * 2 : 0)
        let partMLOff = reserve(self.attnSplits > 1 ? splitRows * 8 : 0)

        let qIdxOff = needLsa ? reserve(maxT * self.indexDim * 2) : 0
        let listOff = needLsa ? reserve(maxT * self.listStride * 4) : 0
        let countOff = needLsa ? reserve(maxT * 4) : 0
        let maxClusters = maxMacros * QWCoarsePoolIndex.clustersPerMacro
        let coarseScratchOffs = QWCoarseSelectionPipeline.ScratchOffsets.reserve(
            enabled: needLsa, maxBatchRows: self.selectBatchRows,
            maxClusters: maxClusters, candCap: Self.emeansCandCap, reserve: reserve)

        let totalBytes = max(4096, cursor)
        guard let buf = hal.device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            throw QWAllocError.outOfSpace(region: "hca-cache", asked: totalBytes, free: 0)
        }
        buf.label = "hca-cache"
        memset(buf.contents(), 0, totalBytes)
        invFreqArr.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, !raw.isEmpty {
                buf.contents().advanced(by: invFreqOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let negInf = Array(repeating: -Float.infinity, count: zeroSinkCount)
        negInf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress {
                buf.contents().advanced(by: zeroSinkOff).copyMemory(from: src, byteCount: raw.count)
            }
        }
        let reg = QWMRegion(buffer: buf, base: 0, length: totalBytes)
        self.region = reg

        var builtLayers: [LayerState] = []
        builtLayers.reserveCapacity(numHcaLayers)
        for i in 0..<numHcaLayers {
            let o = layerOffs[i]
            let csr: QWCoarsePoolIndex.LayerCSR? = needLsa
                ? o.csr.makeLayerCSR(region: reg, maxMacros: maxMacros, indexDim: self.indexDim)
                : nil
            let pKeys: QWTensorValue? = needLsa
                ? QWTensorValue(region: reg, offset: o.pooled, shape: QWTensorShape([paddedCompBlocks, self.indexDim], .f16))
                : nil
            builtLayers.append(LayerState(
                hcaLayerIndex: i,
                swaKV: QWTensorValue(region: reg, offset: o.swa, shape: QWTensorShape([self.ringCap, self.headDim], .f16)),
                compKvPend: QWTensorValue(region: reg, offset: o.kvPend, shape: QWTensorShape([self.ringCap, self.headDim], .f32)),
                compScorePend: QWTensorValue(region: reg, offset: o.scPend, shape: QWTensorShape([self.ringCap, self.headDim], .f32)),
                compKV: QWTensorValue(region: reg, offset: o.comp, shape: QWTensorShape([paddedCompBlocks, self.headDim], .f16)),
                pooledKeys: pKeys,
                coarseCSR: csr
            ))
        }
        self.layers = builtLayers

        self.fusedInScratch = QWTensorValue(region: reg, offset: fusedInOff, shape: QWTensorShape([maxT, self.inStride], .f16))
        self.qrScratch = QWTensorValue(region: reg, offset: qrOff, shape: QWTensorShape([maxT, self.qLoraRank], .f16))
        self.qUpGateScratch = QWTensorValue(region: reg, offset: qUpOff, shape: QWTensorShape([maxT, self.qUpStride], .f16))
        self.attScratch = QWTensorValue(region: reg, offset: attOff, shape: QWTensorShape([maxT, self.qHeads * self.headDim], .f16))
        self.zVhaScratch = QWTensorValue(region: reg, offset: zVhaOff, shape: QWTensorShape([maxT, self.qHeads * self.headDim], .f16))
        self.zGatedScratch = QWTensorValue(region: reg, offset: zGatedOff, shape: QWTensorShape([maxT, self.zOutStride], .f16))
        self.invFreqBuf = QWTensorValue(region: reg, offset: invFreqOff, shape: QWTensorShape([invFreqCount], .f32))
        self.zeroSinkBuf = QWTensorValue(region: reg, offset: zeroSinkOff, shape: QWTensorShape([zeroSinkCount], .f32))
        self.partOScratch = QWTensorValue(region: reg, offset: partOOff, shape: QWTensorShape([max(1, splitRows), self.headDim], .f16))
        self.partMLScratch = QWTensorValue(region: reg, offset: partMLOff, shape: QWTensorShape([max(1, splitRows), 2], .f32))

        if let loadedCoarse, needLsa {
            self.qIdxScratch = QWTensorValue(region: reg, offset: qIdxOff, shape: QWTensorShape([maxT, self.indexDim], .f16))
            self.blockListScratch = QWTensorValue(region: reg, offset: listOff, shape: QWTensorShape([maxT, self.listStride], .i32))
            self.blockCountScratch = QWTensorValue(region: reg, offset: countOff, shape: QWTensorShape([maxT], .i32))
            let cScratch = coarseScratchOffs.makeScratch(
                region: reg, maxBatchRows: self.selectBatchRows,
                maxClusters: maxClusters, candCap: Self.emeansCandCap)
            self.coarsePipeline = QWCoarseSelectionPipeline(
                variant: .lsa, loaded: loadedCoarse, scratch: cScratch, layerCount: numHcaLayers)
        } else {
            self.qIdxScratch = nil
            self.blockListScratch = nil
            self.blockCountScratch = nil
            self.coarsePipeline = nil
        }
    }

}
