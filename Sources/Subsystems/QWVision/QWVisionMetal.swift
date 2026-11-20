// QWVisionMetal.swift — Native Metal FP16 ViT backend with compact UMA weights
// and lazy per-grid scratch (0 B when unloaded, single-CB execution).
import Accelerate
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

public final class QWVisionMetalBackend: QWVisionBackend, @unchecked Sendable {
    struct BlockTensors: Sendable {
        let norm1G: QWTensorValue, norm1B: QWTensorValue
        let qkvW: QWTensorValue, qkvB: QWTensorValue
        let projW: QWTensorValue, projB: QWTensorValue
        let norm2G: QWTensorValue, norm2B: QWTensorValue
        let l1W: QWTensorValue, l1B: QWTensorValue
        let l2W: QWTensorValue, l2B: QWTensorValue
    }

    struct MetalWeights: @unchecked Sendable {
        let buffer: any MTLBuffer
        let posEmbed: QWTensorValue
        let patchW: QWTensorValue
        let patchB: QWTensorValue
        let blocks: [BlockTensors]
        let mergerG: QWTensorValue
        let mergerB: QWTensorValue
        let fc1W: QWTensorValue
        let fc1B: QWTensorValue
        let fc2W: QWTensorValue
        let fc2B: QWTensorValue
        let invFreq: QWTensorValue
        let fc2Out: Int
    }

    struct Pipelines: @unchecked Sendable {
        let gemm: any MTLComputePipelineState
        let posEmbed: any MTLComputePipelineState
        let layernorm: any MTLComputePipelineState
        let ropeQkv: any MTLComputePipelineState
        let flashAttn: any MTLComputePipelineState
        let geluTanh: any MTLComputePipelineState
        let geluErf: any MTLComputePipelineState
    }

    private let lock = NSLock()
    private let config: QWVisionConfig
    private let fc2Out: Int
    let hal: QWHal?
    let pipelines: Pipelines?
    private var weights: MetalWeights?
    private var scratchBuffer: (any MTLBuffer)?
    private var cpuRef: QWVisionRef?
    private var keepResidentFlag: Bool

    public var visionConfig: QWVisionConfig { config }
    public var isLoaded: Bool { lock.withLock { weights != nil || cpuRef != nil } }
    public var keepResident: Bool { lock.withLock { keepResidentFlag } }
    public var allowOnDemand: Bool { false }
    public var isVisionEnabled: Bool { isLoaded }

    public var residentBytes: Int {
        lock.withLock {
            if let w = weights {
                return w.buffer.length + (scratchBuffer?.length ?? 0)
            }
            return cpuRef?.w.residentBytes ?? 0
        }
    }

    public init(weights w: QWVisionWeights, hal: QWHal? = nil, keepResident: Bool = true) throws {
        self.config = w.config
        self.fc2Out = w.fc2Out
        self.keepResidentFlag = keepResident
        if ProcessInfo.processInfo.environment["QWG_VISION_CPU"] == "1" {
            self.hal = hal
            self.pipelines = nil
            self.cpuRef = QWVisionRef(weights: w)
            return
        }
        let resolvedHal = try hal ?? Self.createHal(config: w.config, packDir: w.ck.dir)
        self.hal = resolvedHal
        self.pipelines = try Self.loadPipelines(hal: resolvedHal)
        self.weights = try Self.packFromWeights(w, device: resolvedHal.device)
    }

    public init(checkpoint ck: QWVisionCheckpoint, hal: QWHal? = nil, keepResident: Bool = true) throws {
        let cfg = ck.config
        let resolvedOut = QWVisionWeights.resolveFc2Out(in: ck, config: cfg)
        self.config = cfg
        self.fc2Out = resolvedOut
        self.keepResidentFlag = keepResident
        if ProcessInfo.processInfo.environment["QWG_VISION_CPU"] == "1" {
            self.hal = hal
            self.pipelines = nil
            self.cpuRef = QWVisionRef(weights: try QWVisionWeights(checkpoint: ck))
            return
        }
        let resolvedHal = try hal ?? Self.createHal(config: cfg, packDir: ck.dir)
        self.hal = resolvedHal
        self.pipelines = try Self.loadPipelines(hal: resolvedHal)
        self.weights = try Self.packFromCheckpoint(ck, fc2Out: resolvedOut, device: resolvedHal.device)
        ck.releaseBacking()
    }

    public func prepareForPrompt(hasImages: Bool) throws {
        lock.withLock {
            if !hasImages {
                scratchBuffer = nil
                if !keepResidentFlag {
                    weights = nil
                    cpuRef = nil
                }
            }
        }
    }

    public func setVisionLoaded(_ loaded: Bool) throws {
        lock.withLock {
            keepResidentFlag = loaded
            if !loaded {
                scratchBuffer = nil
                weights = nil
                cpuRef = nil
            }
        }
    }

    public func releaseScratch() {
        lock.withLock { scratchBuffer = nil }
    }

    public func encode(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) async throws -> [Float] {
        try encodeSync(patches: patches, grid: grid, bins: bins, posIds: posIds)
    }

    public func encodeSync(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) throws -> [Float] {
        if let ref = lock.withLock({ cpuRef }) {
            return ref.encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
        }
        var cu = [Int32](repeating: 0, count: grid.thw.t + 1)
        for f in 0..<grid.thw.t {
            cu[f + 1] = cu[f] + Int32(grid.frameLen)
        }
        return try encodeSegments(
            patches: patches,
            nPatch: grid.nPatch,
            nToken: grid.nToken,
            cuSeqlens: cu,
            bins: bins,
            posIds: posIds
        )
    }

    func encodeSegments(
        patches: [Float],
        nPatch: Int,
        nToken: Int,
        cuSeqlens: [Int32],
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) throws -> [Float] {
        let cfg = config
        let d = cfg.dim
        precondition(patches.count == nPatch * cfg.patchChannels
                     && bins.nPatch == nPatch
                     && posIds.count == 2 * nPatch)
        guard nPatch > 0, nToken > 0, d > 0, cfg.heads > 0, cfg.headDim > 0, fc2Out > 0 else {
            return []
        }
        guard cuSeqlens.count >= 2, cuSeqlens.first == 0, cuSeqlens.last == Int32(nPatch) else {
            throw QWVisionError.badLayout("invalid cuSeqlens for nPatch=\(nPatch)")
        }
        return try lock.withLock {
            guard let hal, let pipelines, let w = weights else {
                throw QWVisionError.badLayout("QWVisionMetalBackend is unloaded")
            }
            return try runMetalEncodeLocked(
                hal: hal, pipelines: pipelines, w: w,
                patches: patches, nPatch: nPatch, nToken: nToken,
                cuSeqlens: cuSeqlens, bins: bins, posIds: posIds
            )
        }
    }

    private func ensureScratchLocked(device: any MTLDevice, totalBytes: Int) throws -> QWMRegion {
        if let existing = scratchBuffer, existing.length >= totalBytes {
            return QWMRegion(buffer: existing, base: 0, length: existing.length)
        }
        scratchBuffer = nil
        guard let buf = device.makeBuffer(length: max(256, totalBytes), options: .storageModeShared) else {
            throw QWHalError.noDevice
        }
        scratchBuffer = buf
        return QWMRegion(buffer: buf, base: 0, length: buf.length)
    }

    private func runMetalEncodeLocked(
        hal: QWHal, pipelines: Pipelines, w: MetalWeights,
        patches: [Float], nPatch: Int, nToken: Int,
        cuSeqlens: [Int32], bins: QWVisionRef.InterpBins, posIds: [Int32]
    ) throws -> [Float] {
        let cfg = config
        let d = cfg.dim, qkvD = cfg.qkvDim, mlp = cfg.mlpInner
        let merged = cfg.merged, outD = w.fc2Out, pc = cfg.patchChannels
        let workACount = max(nPatch * max(qkvD, mlp), nToken * merged)
        let outCount = nToken * outD
        let totalScratch =
            qwAlignUp(nPatch * pc * 2) +
            qwAlignUp(nPatch * 4 * 4) +
            qwAlignUp(nPatch * 4 * 4) +
            qwAlignUp(nPatch * 2 * 4) +
            qwAlignUp(cuSeqlens.count * 4) +
            qwAlignUp(nPatch * d * 2) +
            qwAlignUp(nPatch * d * 2) +
            qwAlignUp(workACount * 2) +
            qwAlignUp(nPatch * d * 2) +
            qwAlignUp(outCount * 2)

        let region = try ensureScratchLocked(device: hal.device, totalBytes: totalScratch)
        let sub = QWSubAllocator(region: region)
        let patchesT = try sub.allocate(QWTensorShape([nPatch * pc], .f16))
        let binsIdxT = try sub.allocate(QWTensorShape([nPatch * 4], .i32))
        let binsWgtT = try sub.allocate(QWTensorShape([nPatch * 4], .f32))
        let posIdsT = try sub.allocate(QWTensorShape([nPatch * 2], .i32))
        let cuT = try sub.allocate(QWTensorShape([cuSeqlens.count], .i32))
        let sT = try sub.allocate(QWTensorShape([nPatch * d], .f16))
        let normedT = try sub.allocate(QWTensorShape([nPatch * d], .f16))
        let workAT = try sub.allocate(QWTensorShape([workACount], .f16))
        let attnOutT = try sub.allocate(QWTensorShape([nPatch * d], .f16))
        let outT = try sub.allocate(QWTensorShape([outCount], .f16))

        Self.writeF16(patches, into: patchesT)
        Self.writePODArray(bins.idx, into: binsIdxT)
        Self.writePODArray(bins.wgt, into: binsWgtT)
        Self.writePODArray(posIds, into: posIdsT)
        Self.writePODArray(cuSeqlens, into: cuT)

        let maxSegLen = zip(cuSeqlens, cuSeqlens.dropFirst()).reduce(0) { max($0, max(0, Int($1.1 - $1.0))) }
        let attnScale = Float(1.0 / Double(cfg.headDim).squareRoot())
        guard let (cb, enc) = hal.begin() else { throw QWHalError.noDevice }

        encodeGemm(enc, pso: pipelines.gemm, x: patchesT, w: w.patchW, bias: w.patchB,
                   residual: nil, y: sT, m: nPatch, n: d, k: pc)
        encodePosEmbed(enc, pso: pipelines.posEmbed, s: sT, posEmbed: w.posEmbed,
                       binsIdx: binsIdxT, binsWgt: binsWgtT,
                       nPatch: nPatch, dim: d, numPos: cfg.numPosEmbeds)

        for blk in w.blocks {
            encodeLayerNorm(enc, pso: pipelines.layernorm, x: sT,
                            gamma: blk.norm1G, beta: blk.norm1B, out: normedT,
                            rows: nPatch, dim: d, eps: cfg.eps)
            encodeGemm(enc, pso: pipelines.gemm, x: normedT, w: blk.qkvW, bias: blk.qkvB,
                       residual: nil, y: workAT, m: nPatch, n: qkvD, k: d)
            encodeRoPE(enc, pso: pipelines.ropeQkv, qkv: workAT, posIds: posIdsT, invFreq: w.invFreq,
                       nPatch: nPatch, heads: cfg.heads, headDim: cfg.headDim,
                       dim: d, invFreqCount: cfg.invFreqCount)
            encodeFlashAttn(enc, pso: pipelines.flashAttn, qkv: workAT, out: attnOutT,
                            cuSeqlens: cuT, numSegs: cuSeqlens.count - 1, maxSegLen: maxSegLen,
                            heads: cfg.heads, headDim: cfg.headDim, dim: d, scale: attnScale)
            encodeGemm(enc, pso: pipelines.gemm, x: attnOutT, w: blk.projW, bias: blk.projB,
                       residual: sT, y: sT, m: nPatch, n: d, k: d)
            encodeLayerNorm(enc, pso: pipelines.layernorm, x: sT,
                            gamma: blk.norm2G, beta: blk.norm2B, out: normedT,
                            rows: nPatch, dim: d, eps: cfg.eps)
            encodeGemm(enc, pso: pipelines.gemm, x: normedT, w: blk.l1W, bias: blk.l1B,
                       residual: nil, y: workAT, m: nPatch, n: mlp, k: d)
            encodeGelu(enc, pso: pipelines.geluTanh, x: workAT, count: nPatch * mlp)
            encodeGemm(enc, pso: pipelines.gemm, x: workAT, w: blk.l2W, bias: blk.l2B,
                       residual: sT, y: sT, m: nPatch, n: d, k: mlp)
        }

        encodeLayerNorm(enc, pso: pipelines.layernorm, x: sT,
                        gamma: w.mergerG, beta: w.mergerB, out: normedT,
                        rows: nPatch, dim: d, eps: cfg.eps)
        encodeGemm(enc, pso: pipelines.gemm, x: normedT, w: w.fc1W, bias: w.fc1B,
                   residual: nil, y: workAT, m: nToken, n: merged, k: merged)
        encodeGelu(enc, pso: pipelines.geluErf, x: workAT, count: nToken * merged)
        encodeGemm(enc, pso: pipelines.gemm, x: workAT, w: w.fc2W, bias: w.fc2B,
                   residual: nil, y: outT, m: nToken, n: outD, k: merged)

        hal.end(cb, enc)
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        return Self.readF16(from: outT, count: outCount)
    }

    private static func createHal(config: QWVisionConfig, packDir: String?) throws -> QWHal {
        let geom: QWModelGeometry
        if let dir = packDir, let loaded = try? QWModelGeometry.load(packPath: dir) {
            geom = loaded
        } else {
            let desc = QWModelDescriptor(
                family: .qwen38Dense, modelType: "qwen3_5", capabilities: [.vision],
                hidden: max(64, config.outDim), vocab: 256, denseIntermediate: max(64, config.mlpInner),
                numLayers: 1, layerTypes: ["full_attention"], fullAttentionInterval: 1,
                rmsEps: config.eps, maxPositionEmbeddings: 4096,
                qHeads: max(1, config.heads), kvHeads: max(1, config.heads), headDim: max(8, config.headDim),
                attnOutputGate: false, partialRotaryFactor: 1.0, ropeTheta: config.ropeTheta,
                indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
                indexerBudget: nil, indexerCompressRatio: nil,
                numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
                sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
                gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
                pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
                ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
                makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
                mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 0,
                visDepth: config.depth, visHidden: config.dim, visHeads: config.heads,
                visIntermediate: config.mlpInner, visPatch: config.patch,
                visTemporal: config.temporal, visMerge: config.merge,
                visPosEmbeds: config.numPosEmbeds, visOut: config.outDim,
                imageTokenId: config.imageTokenId, videoTokenId: config.videoTokenId
            )
            geom = QWModelGeometry(descriptor: desc)
        }
        do {
            return try QWHal(geometry: geom, weightBasis: .rotated)
        } catch QWHalError.metallibBasisMismatch {
            return try QWHal(geometry: geom, weightBasis: .standard)
        }
    }

    private static func loadPipelines(hal: QWHal) throws -> Pipelines {
        Pipelines(
            gemm: try hal.pipeline("qwg_vis_gemm_f16"),
            posEmbed: try hal.pipeline("qwg_vis_pos_embed_f16"),
            layernorm: try hal.pipeline("qwg_vis_layernorm_f16"),
            ropeQkv: try hal.pipeline("qwg_vis_rope_qkv_f16"),
            flashAttn: try hal.pipeline("qwg_vis_flash_attn_f16"),
            geluTanh: try hal.pipeline("qwg_vis_gelu_tanh_f16"),
            geluErf: try hal.pipeline("qwg_vis_gelu_erf_f16")
        )
    }
}
