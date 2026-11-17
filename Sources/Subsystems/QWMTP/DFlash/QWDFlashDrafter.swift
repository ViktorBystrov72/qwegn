// QWDFlashDrafter.swift — DFlash 2 / DSpine parallel block-diffusion drafter
// with sliding-window layers, dynamic causal conv, and rank-256 selector.
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

public final class QWDFlashDrafter: @unchecked Sendable {
    public let geometry: QWModelGeometry
    public static let rowPairedMinimumRows = 4

    public static func usesQ4GMKernel(_ weight: QWWeightView) -> Bool {
        weight.isIncumbentContainer || (!weight.isSHVQ && weight.bits == 4 && weight.groupSize == 32)
    }

    public static func usesRowPairs(_ weight: QWWeightView, activeRows: Int) -> Bool {
        usesQ4GMKernel(weight) && activeRows >= rowPairedMinimumRows
    }

    public static func siluThreadgroups(i: Int) -> Int {
        (i + 511) / 512
    }

    struct LayerWeights {
        let ln1: QWTensorValue
        let ln2: QWTensorValue
        let qNorm: QWTensorValue
        let kNorm: QWTensorValue
        let attnConvBase: QWTensorValue?
        let attnConvProj: QWTensorValue?
        let mlpConvBase: QWTensorValue?
        let mlpConvProj: QWTensorValue?
        let qkvProj: QWWeightView
        let kvProj: QWWeightView
        let oProj: QWWeightView
        let gateUpProj: QWWeightView
        let downProj: QWWeightView
    }

    public static let defaultPackPath: String = QWDFlashConfig.defaultPackPath

    public let config: QWDFlashConfig
    public let packPath: String
    public let targetLayerSlotMap: [Int]
    public let sessionSlots: [QWDFlashSessionSlot]
    public let targetHidden: QWTensorValue

    let hal: QWHal
    let weightBuffer: any MTLBuffer
    let scratchBuffer: any MTLBuffer

    let fc: QWWeightView
    let hiddenNorm: QWTensorValue
    let finalNorm: QWTensorValue
    let selectorHiddenProj: QWTensorValue?
    let predecessorCodebook: QWTensorValue?
    let successorCodebook: QWTensorValue?
    let dspineBridge: DSpineBridgeWeights?
    let invFreq: QWTensorValue
    let layers: [LayerWeights]

    let hCtx: QWTensorValue
    let kvCtx: QWTensorValue
    let propX: QWTensorValue
    let propNorm: QWTensorValue
    let propRot: QWTensorValue
    let propRot32: QWTensorValue
    let propTokens: QWTensorValue
    let propDyn: QWTensorValue?
    let qkvProp: QWTensorValue
    let attnOut: QWTensorValue
    let mlpInter: QWTensorValue
    let propDown: QWTensorValue
    let draftLogits: QWTensorValue
    let top16PartialsVal: QWTensorValue
    let top16PartialsId: QWTensorValue
    let unaryOut: QWTensorValue
    let candidatesOut: QWTensorValue
    let hiddenProjOut: QWTensorValue?
    let draftTokensOut: QWTensorValue
    let dspineScratch: DSpineBridgeScratch?

    let psoCaptureTarget: any MTLComputePipelineState
    let psoCaptureHC: any MTLComputePipelineState
    let psoStagePropBF16: any MTLComputePipelineState
    let psoRMSNorm: any MTLComputePipelineState
    let psoF16GemvBatched: any MTLComputePipelineState
    let psoConvApply: any MTLComputePipelineState
    let psoRopeKVCtx: any MTLComputePipelineState
    let psoRopeQKProp: any MTLComputePipelineState
    let psoAttn: any MTLComputePipelineState
    let psoTop16Stage1: any MTLComputePipelineState
    let psoTop16Stage2: any MTLComputePipelineState
    let psoSelectorGreedy: any MTLComputePipelineState
    let psoGemvT4: any MTLComputePipelineState
    let psoGemmBatched: any MTLComputePipelineState
    let psoSiluMulF16: any MTLComputePipelineState
    let isMetal4TensorCapable: Bool
    let psoDownBatched: [Int: any MTLComputePipelineState]
    let psoDownRowPaired: [Int: any MTLComputePipelineState]
    let psoDownBatchedAQ4: [Int: any MTLComputePipelineState]
    let psoDownRowPairedAQ4: [Int: any MTLComputePipelineState]
    let psoGateUpFwht32Batched: [Int: any MTLComputePipelineState]
    let psoGateUpFwht32BatchedAQ4: [Int: any MTLComputePipelineState]
    let dspinePSO: DSpinePipelines
    let quantPSOs: [DrafterQuantKey: DrafterQuantPipelines]

    public init(
        geometry: QWModelGeometry,
        hal: QWHal,
        packPath: String,
        drafts: Int = 7,
        selectorTopK: Int? = nil,
        sessionSlotCount: Int = 2
    ) throws {
        let resolvedSlots = max(1, sessionSlotCount)
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: geometry.descriptor,
            packPath: packPath,
            drafts: drafts,
            selectorTopK: selectorTopK
        )
        self.geometry = geometry
        self.hal = hal
        let resolvedURL = packPath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw QWDFlashError.packMissing(resolvedURL.path)
        }
        self.packPath = resolvedURL.path
        self.config = QWDFlashConfig.load(from: resolvedURL, topKOverride: selectorTopK)
        var slotMap = [Int](repeating: -1, count: geometry.numLayers)
        for (slot, lid) in config.targetLayerIds.enumerated() where (0..<geometry.numLayers).contains(lid) {
            slotMap[lid] = slot
        }
        self.targetLayerSlotMap = slotMap

        guard let drafterCfgURL = QWModelRoot(url: resolvedURL).configURL else {
            throw QWDFlashError.state("DFlash2 drafter missing config.json at \(resolvedURL.path)")
        }
        let drafterDescriptor = try QWModelDescriptor.decodeDrafter(configJSON: Data(contentsOf: drafterCfgURL))
        let pack = try QWQuant.Pack(url: resolvedURL, descriptor: drafterDescriptor)
        let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: pack.descriptor))

        let wCapacity = Self.computeWeightCapacity(pack: pack, isDSpine: config.isDSpine)
        guard let wBuf = hal.device.makeBuffer(length: wCapacity, options: .storageModeShared) else {
            throw QWDFlashError.state("QWDFlashDrafter: failed to allocate weight buffer")
        }
        self.weightBuffer = wBuf
        let wSub = QWSubAllocator(region: QWMRegion(buffer: wBuf, base: 0, length: wCapacity))

        func loadQ4(_ name: String, n: Int, k: Int) throws -> QWWeightView {
            try Self.loadQuantWeight(pack: pack, io: io, wSub: wSub, hal: hal, name: name, n: n, k: k)
        }

        func concatQ4(_ specs: [(name: String, n: Int)], k: Int) throws -> (QWWeightView, [QWWeightView]) {
            try Self.concatQuantWeights(pack: pack, io: io, wSub: wSub, hal: hal, specs: specs, k: k)
        }

        let psoBf16F16 = try hal.pipeline("qwg_bf16_to_f16")
        guard let cvtTmp = hal.device.makeBuffer(length: 16 * 1024 * 1024, options: .storageModeShared) else {
            throw QWDFlashError.state("QWDFlashDrafter: failed to allocate bf16 staging buffer")
        }

        func loadF16(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let e = try pack.entry(name)
            let total = e.shape.reduce(1, *)
            guard total == elems else {
                throw QWDFlashError.weightMissing(name, "DFlash shape \(total) != \(elems)")
            }
            let t = try wSub.allocate(QWTensorShape([elems], .f16))
            if elems >= 4096, e.rawOff != nil {
                let chunk = 8 * 1024 * 1024
                var done = 0
                while done < elems {
                    let n = min(chunk, elems - done)
                    try io.loadRaw(name, elemOffset: done, elems: n, into: cvtTmp.contents())
                    guard let (cb, enc) = hal.begin() else {
                        throw QWDFlashError.state("DFlash bf16->f16 cmdbuffer")
                    }
                    var start = UInt32(0), nn = UInt32(n)
                    enc.setComputePipelineState(psoBf16F16)
                    enc.setBuffer(cvtTmp, offset: 0, index: 0)
                    enc.setBuffer(t.region.buffer, offset: t.region.base + t.offset + done * 2, index: 1)
                    enc.setBytes(&start, length: 4, index: 2)
                    enc.setBytes(&nn, length: 4, index: 3)
                    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    done += n
                }
                return t
            }
            let src = try pack.rawBf16(name)
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
                .assumingMemoryBound(to: UInt16.self)
            for i in 0..<elems { dst[i] = QWQuant.f32tof16Bits(src[i]) }
            return t
        }

        func loadRawBF16(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let t = try wSub.allocate(QWTensorShape([elems], .f16))
            let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            try io.loadRaw(name, elemOffset: 0, elems: elems, into: dst)
            return t
        }

        let H = config.hiddenSize
        let I = config.intermediateSize
        let V = config.vocabSize
        let R = config.selectorRank
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        let hasConv = config.convKernelSize > 0
        let convProjDim = hasConv ? (2 * config.convKernelSize * (H / config.convGroupSize)) : 0

        self.fc = try loadQ4("fc.weight", n: H, k: config.targetLayerIds.count * H)
        self.hiddenNorm = try loadF16("hidden_norm.weight", H)
        self.finalNorm = try loadF16("norm.weight", H)
        if config.isDSpine {
            self.selectorHiddenProj = nil
            self.predecessorCodebook = nil
            self.successorCodebook = nil
            self.dspineBridge = try Self.loadDSpineBridgeWeights(
                config: config, wSub: wSub, loadF16: loadF16, loadRawBF16: loadRawBF16
            )
        } else {
            self.selectorHiddenProj = try loadF16("candidate_selector.hidden_projection.weight", R * H)
            self.predecessorCodebook = try loadRawBF16("candidate_selector.predecessor_codebook", V * R)
            self.successorCodebook = try loadRawBF16("candidate_selector.successor_codebook", V * R)
            self.dspineBridge = nil
        }

        let halfDim = config.headDim / 2
        let invFreqT = try wSub.allocate(QWTensorShape([halfDim], .f32))
        let invPtr = invFreqT.region.buffer.contents().advanced(by: invFreqT.region.base + invFreqT.offset)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<halfDim {
            let exp = Float(2 * i) / Float(config.headDim)
            invPtr[i] = 1.0 / powf(config.ropeTheta, exp)
        }
        self.invFreq = invFreqT

        var loadedLayers: [LayerWeights] = []
        for l in 0..<config.numLayers {
            let p = "layers.\(l)."
            let ln1 = try loadF16(p + "input_layernorm.weight", H)
            let ln2 = try loadF16(p + "post_attention_layernorm.weight", H)
            let qNorm = try loadF16(p + "self_attn.q_norm.weight", config.headDim)
            let kNorm = try loadF16(p + "self_attn.k_norm.weight", config.headDim)
            let attnConvBase = hasConv ? try loadF16(p + "attention_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let attnConvProj = hasConv ? try loadF16(p + "attention_conv.kernel_projection.weight", convProjDim * H) : nil
            let mlpConvBase = hasConv ? try loadF16(p + "mlp_conv.base_kernel", 2 * config.convKernelSize * H) : nil
            let mlpConvProj = hasConv ? try loadF16(p + "mlp_conv.kernel_projection.weight", convProjDim * H) : nil

            let (qkvBig, _) = try concatQ4([
                (p + "self_attn.q_proj.weight", qDim),
                (p + "self_attn.k_proj.weight", kvDim),
                (p + "self_attn.v_proj.weight", kvDim)
            ], k: H)
            let kvBlockOffset = qDim * qkvBig.rowBytes
            let kvByteCount = 2 * kvDim * qkvBig.rowBytes
            let kvBlocks = QWTensorValue(region: qkvBig.blocks.region,
                                         offset: qkvBig.blocks.offset + kvBlockOffset,
                                         shape: QWTensorShape([kvByteCount / 2], .f16))
            guard let kvView = qkvBig.subview(blocks: kvBlocks, n: 2 * kvDim, k: H) else {
                throw QWDFlashError.weightMissing(p + "self_attn.k_proj.weight", "invalid kvView")
            }
            let oProj = try loadQ4(p + "self_attn.o_proj.weight", n: H, k: qDim)
            let (guBig, _) = try concatQ4([
                (p + "mlp.gate_proj.weight", I),
                (p + "mlp.up_proj.weight", I)
            ], k: H)
            let downProj = try loadQ4(p + "mlp.down_proj.weight", n: H, k: I)
            loadedLayers.append(LayerWeights(
                ln1: ln1, ln2: ln2, qNorm: qNorm, kNorm: kNorm,
                attnConvBase: attnConvBase, attnConvProj: attnConvProj,
                mlpConvBase: mlpConvBase, mlpConvProj: mlpConvProj,
                qkvProj: qkvBig, kvProj: kvView, oProj: oProj,
                gateUpProj: guBig, downProj: downProj
            ))
        }
        self.layers = loadedLayers
        wSub.markSteadyState()
        pack.adviseDontNeed()

        let maxCtx = config.maxCacheSize
        let maxB = max(8, config.blockSize)
        let maxD = maxB - 1
        let sCapacity = Self.computeScratchCapacity(
            config: config, maxCtx: maxCtx, maxB: maxB, maxD: maxD,
            qDim: qDim, kvDim: kvDim, hasConv: hasConv, convProjDim: convProjDim,
            sessionSlotCount: resolvedSlots
        )
        guard let sBuf = hal.device.makeBuffer(length: sCapacity, options: .storageModeShared) else {
            throw QWDFlashError.state("QWDFlashDrafter: failed to allocate scratch buffer")
        }
        self.scratchBuffer = sBuf
        let sSub = QWSubAllocator(region: QWMRegion(buffer: sBuf, base: 0, length: sCapacity))

        self.targetHidden = try sSub.allocate(QWTensorShape([maxCtx * config.targetLayerIds.count * H], .f16))
        self.hCtx = try sSub.allocate(QWTensorShape([maxCtx * H], .f16))
        self.kvCtx = try sSub.allocate(QWTensorShape([maxCtx * 2 * kvDim], .f16))
        self.propX = try sSub.allocate(QWTensorShape([maxB * H], .f32))
        self.propNorm = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.propRot = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.propRot32 = try sSub.allocate(QWTensorShape([maxB * H], .f32))
        self.propTokens = try sSub.allocate(QWTensorShape([maxB], .i32))
        self.propDyn = hasConv ? try sSub.allocate(QWTensorShape([maxB * convProjDim], .f32)) : nil
        self.qkvProp = try sSub.allocate(QWTensorShape([maxB * (qDim + 2 * kvDim)], .f16))
        self.attnOut = try sSub.allocate(QWTensorShape([maxB * qDim], .f16))
        self.mlpInter = try sSub.allocate(QWTensorShape([maxB * I], .f16))
        self.propDown = try sSub.allocate(QWTensorShape([maxB * H], .f16))
        self.draftLogits = try sSub.allocate(QWTensorShape([maxD * V], .f32))
        self.top16PartialsVal = try sSub.allocate(QWTensorShape([maxD * 256 * 16], .f32))
        self.top16PartialsId = try sSub.allocate(QWTensorShape([maxD * 256 * 16], .i32))
        self.unaryOut = try sSub.allocate(QWTensorShape([maxD * 16], .f32))
        self.candidatesOut = try sSub.allocate(QWTensorShape([maxD * 16], .i32))
        self.hiddenProjOut = config.isDSpine ? nil : try sSub.allocate(QWTensorShape([maxD * R], .f32))
        self.draftTokensOut = try sSub.allocate(QWTensorShape([maxB], .i32))
        self.dspineScratch = config.isDSpine
            ? try Self.allocateDSpineScratch(config: config, sSub: sSub, maxB: maxB, maxD: maxD)
            : nil

        let layerKVElements = maxCtx * kvDim
        var slots: [QWDFlashSessionSlot] = []
        for _ in 0..<resolvedSlots {
            let kSlab = try sSub.allocate(QWTensorShape([config.numLayers * layerKVElements], .f16))
            let vSlab = try sSub.allocate(QWTensorShape([config.numLayers * layerKVElements], .f16))
            let kViews = (0..<config.numLayers).map { l in
                QWTensorValue(region: kSlab.region, offset: kSlab.offset + l * layerKVElements * 2,
                              shape: QWTensorShape([layerKVElements], .f16))
            }
            let vViews = (0..<config.numLayers).map { l in
                QWTensorValue(region: vSlab.region, offset: vSlab.offset + l * layerKVElements * 2,
                              shape: QWTensorShape([layerKVElements], .f16))
            }
            slots.append(QWDFlashSessionSlot(kSlab: kSlab, vSlab: vSlab, kLayers: kViews, vLayers: vViews))
        }
        self.sessionSlots = slots
        sSub.markSteadyState()

        self.psoCaptureTarget = try hal.pipeline("qwg_dflash_capture_target_hidden")
        self.psoCaptureHC = try hal.pipeline("qwg_dflash_capture_hc_hidden")
        self.psoStagePropBF16 = try hal.pipeline("qwg_dflash_stage_prop_bf16")
        self.psoRMSNorm = try hal.pipeline("qwg_dflash_rmsnorm")
        self.psoF16GemvBatched = try hal.pipeline("qwg_dflash_f16_gemv_batched")
        self.psoConvApply = try hal.pipeline("qwg_dflash_conv_apply")
        self.psoRopeKVCtx = try hal.pipeline("qwg_dflash_rope_kv_ctx")
        self.psoRopeQKProp = try hal.pipeline("qwg_dflash_rope_qk_prop")
        self.psoAttn = try hal.pipeline("qwg_dflash_attn")
        self.psoTop16Stage1 = try hal.pipeline("qwg_dflash_top16_stage1")
        self.psoTop16Stage2 = try hal.pipeline("qwg_dflash_top16_stage2")
        self.psoSelectorGreedy = try hal.pipeline("qwg_dflash_selector_greedy")
        self.psoGemvT4 = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
        self.psoGemmBatched = try hal.pipeline("qwg_gemm_q4g_h16")
        self.psoSiluMulF16 = try hal.pipeline("qwg_silu_mul_f16")
        self.isMetal4TensorCapable = hal.caps.isMetal4TensorCapable
        self.dspinePSO = try DSpinePipelines(hal: hal)

        var downMap: [Int: any MTLComputePipelineState] = [:]
        var guMap: [Int: any MTLComputePipelineState] = [:]
        var guAQ4Map: [Int: any MTLComputePipelineState] = [:]
        var downRowPairedMap: [Int: any MTLComputePipelineState] = [:]
        var downAQ4Map: [Int: any MTLComputePipelineState] = [:]
        var downRowPairedAQ4Map: [Int: any MTLComputePipelineState] = [:]
        let needsAQ4 = (!self.fc.isIncumbentContainer && !self.fc.isSHVQ && self.fc.bits == 4 && self.fc.groupSize == 32)
            || loadedLayers.contains { w in
                (!w.qkvProj.isIncumbentContainer && !w.qkvProj.isSHVQ && w.qkvProj.bits == 4 && w.qkvProj.groupSize == 32)
                || (!w.oProj.isIncumbentContainer && !w.oProj.isSHVQ && w.oProj.bits == 4 && w.oProj.groupSize == 32)
                || (!w.gateUpProj.isIncumbentContainer && !w.gateUpProj.isSHVQ && w.gateUpProj.bits == 4 && w.gateUpProj.groupSize == 32)
                || (!w.downProj.isIncumbentContainer && !w.downProj.isSHVQ && w.downProj.bits == 4 && w.downProj.groupSize == 32)
            }
        for m in 2...8 {
            downMap[m] = try hal.pipeline("qwm_mlp_down_m\(m)")
            guMap[m] = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)")
            if needsAQ4 {
                downAQ4Map[m] = try hal.pipeline("qwm_mlp_down_m\(m)", isAQ4: true)
                guAQ4Map[m] = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)", isAQ4: true)
            }
            if m >= Self.rowPairedMinimumRows {
                downRowPairedMap[m] = try hal.pipeline("qwm_mlp_down_rp_m\(m)")
                if needsAQ4 {
                    downRowPairedAQ4Map[m] = try hal.pipeline("qwm_mlp_down_rp_m\(m)", isAQ4: true)
                }
            }
        }
        self.psoDownBatched = downMap
        self.psoDownRowPaired = downRowPairedMap
        self.psoDownBatchedAQ4 = downAQ4Map
        self.psoDownRowPairedAQ4 = downRowPairedAQ4Map
        self.psoGateUpFwht32Batched = guMap
        self.psoGateUpFwht32BatchedAQ4 = guAQ4Map
        self.quantPSOs = try Self.loadQuantPipelines(hal: hal, fc: self.fc, layers: loadedLayers)
    }

    private static func computeScratchCapacity(
        config: QWDFlashConfig, maxCtx: Int, maxB: Int, maxD: Int,
        qDim: Int, kvDim: Int, hasConv: Bool, convProjDim: Int, sessionSlotCount: Int = 2
    ) -> Int {
        func aligned(_ bytes: Int) -> Int { ((bytes + 4095) / 4096) * 4096 }
        let H = config.hiddenSize
        let V = config.vocabSize
        let I = config.intermediateSize
        let R = config.selectorRank
        var total = 0
        total += aligned(maxCtx * config.targetLayerIds.count * H * 2)
        total += aligned(maxCtx * H * 2)
        total += aligned(maxCtx * 2 * kvDim * 2)
        total += aligned(maxB * H * 4) * 2
        total += aligned(maxB * H * 2) * 3
        total += aligned(maxB * 4) * 2
        if hasConv { total += aligned(maxB * convProjDim * 4) }
        total += aligned(maxB * (qDim + 2 * kvDim) * 2)
        total += aligned(maxB * qDim * 2)
        total += aligned(maxB * I * 2)
        total += aligned(maxD * V * 4)
        total += aligned(maxD * 256 * 16 * 4) * 2
        total += aligned(maxD * 16 * 4) * 2
        if !config.isDSpine { total += aligned(maxD * R * 4) }
        if config.isDSpine {
            let bc = config.bridgeCodeSize, br = config.bridgeRank
            total += aligned(maxB * 4) + aligned(maxD * bc * 4) + aligned(maxB * bc * 2)
            total += aligned(maxB * 3 * br * 4) + aligned(maxD * br * 2) + aligned(maxD * H * 4)
            total += aligned(maxD * 16 * br * 2) + aligned(maxD * 16 * H * 4)
            total += aligned(maxD * 16 * H * 2) * 2 + aligned(maxD * 16 * 16 * 4)
        }
        total += (2 * max(1, sessionSlotCount)) * aligned(config.numLayers * maxCtx * kvDim * 2)
        return max(32 * 1024 * 1024, total + 16 * 1024 * 1024)
    }
}
