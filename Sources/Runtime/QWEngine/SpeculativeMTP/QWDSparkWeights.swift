// QWDSparkWeights.swift — Zero-copy QWP1 loader for DSpark weights and optional draft head.
import Darwin
import Foundation
import Metal
import QWAlloc
import QWAttn
import QWCore
import QWGEMM
import QWHAL
import QWLoad
import QWQuant

/// Weight store for DSpark context ingestion, draft HCA/MoE layers, Markov head, and confidence gate.
final class QWDSparkWeights: @unchecked Sendable {
    enum DraftHeadStorage: Sendable {
        case quantized(QWWeightView)
        case bf16(QWTensorValue, n: Int, k: Int)
    }

    struct DraftAttentionGeometry: Sendable {
        let qHeads: Int
        let headDim: Int
        let nopeDim: Int
        let ropeDim: Int
        let qLoraRank: Int
        let oGroups: Int
        let oLoraRank: Int
        var groupIn: Int { (qHeads / max(1, oGroups)) * headDim }
        var qUpStride: Int { qHeads * headDim + oGroups * oLoraRank }
        var zOutStride: Int { oGroups * oLoraRank }
    }

    struct DraftLayer: Sendable {
        let ln1: QWTensorValue
        let ln2: QWTensorValue
        let mhc: QWEngineDriver.MHCLayerWeights?
        let qDownProj: QWWeightView
        let kvProj: QWWeightView
        let qNormW: QWTensorValue
        let kvNormW: QWTensorValue
        let qUpGateProj: QWWeightView
        let woA: QWWeightView
        let oProj: QWWeightView
        let vhaMix: QWTensorValue?
        let attnSink: QWTensorValue?
        let hcaAdapter: QWHCAAttnOperator.LayerWeights
        let moe: QWEngineDriver.MoEW?
        let denseGateUp: QWWeightView?
        let denseDown: QWWeightView?
    }

    let config: QWModelDescriptor.DSparkConfig
    let packPath: String
    let draftHeadPath: String?
    let hidden: Int
    let vocab: Int
    let numTaps: Int
    let attnGeometry: DraftAttentionGeometry
    let moeExperts: Int
    let moeIntermediate: Int
    let moeSharedIntermediate: Int
    let moeLatentDim: Int

    let mainProj: QWWeightView
    let mainNorm: QWTensorValue
    let mhcContractW: QWTensorValue?
    let mhcContractB: QWTensorValue?
    let finalNorm: QWTensorValue
    let markovW1: QWTensorValue
    let markovW2: QWWeightView
    let confidenceHead: QWTensorValue?
    let draftHead: DraftHeadStorage?
    let layers: [DraftLayer]
    let allBuffers: [any MTLBuffer]

    init(
        geometry: QWModelGeometry,
        hal: QWHal,
        config: QWModelDescriptor.DSparkConfig,
        packURL: URL,
        draftHeadURL: URL?
    ) throws {
        self.config = config
        self.packPath = packURL.path
        self.draftHeadPath = draftHeadURL?.path
        self.hidden = geometry.hidden
        self.vocab = geometry.vocab
        self.numTaps = config.targetLayerIds.count

        let pack = try QWQuant.Pack(url: packURL, descriptor: geometry.descriptor, skipIdentityCheck: true)
        let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: pack.descriptor))
        let headPack = try draftHeadURL.map {
            try QWQuant.Pack(url: $0, descriptor: geometry.descriptor, skipIdentityCheck: true)
        }
        let headIO = headPack.map { PlaneIO(pack: $0, policy: LoadPolicy(descriptor: $0.descriptor)) }
        let rot = hal.weightBasis.rotatesActivations
        let normShift = QWCheckpointTensorLoader.normGammaShift(
            normGammaIsPlusOne: geometry.normGammaIsPlusOne,
            isMLXCheckpoint: false
        )

        var buffers: [any MTLBuffer] = []
        let scratchAlloc = ChunkedMetalSlab(device: hal.device)

        func mapOrLoadRaw(
            _ p: QWQuant.Pack,
            _ pIO: PlaneIO,
            name: String,
            offset: Int,
            bytes: Int,
            shape: QWTensorShape
        ) throws -> QWTensorValue {
            if let mapped = PlaneIO.mapZeroCopyRange(pack: p, device: hal.device, offset: offset, byteCount: bytes),
               mapped.subOffset % 16 == 0 {
                buffers.append(mapped.buffer)
                return QWTensorValue(
                    region: QWMRegion(buffer: mapped.buffer, base: mapped.subOffset, length: bytes),
                    offset: 0,
                    shape: shape
                )
            }
            guard let buf = hal.device.makeBuffer(length: max(256, bytes), options: .storageModeShared) else {
                throw QWEngineDriverError.state("DSpark buffer allocation failed for \(name) (\(bytes) B)")
            }
            try pIO.load(name, layout: .raw, into: buf.contents())
            buffers.append(buf)
            return QWTensorValue(
                region: QWMRegion(buffer: buf, base: 0, length: bytes),
                offset: 0,
                shape: shape
            )
        }

        func mapOrLoadQuantView(
            _ p: QWQuant.Pack,
            _ pIO: PlaneIO,
            name: String,
            n: Int,
            k: Int
        ) throws -> QWWeightView {
            let entry = try p.entry(name)
            guard let layout = PlaneLayout.resolve(entry), layout == .hq4,
                  let hqOff = layout.offset(entry),
                  let bytes = layout.byteCount(entry), bytes > 0 else {
                throw QWEngineDriverError.weightMissing(name, "expected quantized plane via PlaneLayout")
            }
            let bufTensor: QWTensorValue
            if let mapped = PlaneIO.mapZeroCopyRange(pack: p, device: hal.device, offset: hqOff, byteCount: bytes),
               mapped.subOffset % 16 == 0 {
                buffers.append(mapped.buffer)
                bufTensor = QWTensorValue(
                    region: QWMRegion(buffer: mapped.buffer, base: mapped.subOffset, length: bytes),
                    offset: 0,
                    shape: QWTensorShape([bytes / 2], .f16)
                )
            } else {
                guard let buf = hal.device.makeBuffer(length: max(256, bytes), options: .storageModeShared) else {
                    throw QWEngineDriverError.state("DSpark buffer allocation failed for \(name)")
                }
                try pIO.load(name, layout: .hq4, into: buf.contents())
                buffers.append(buf)
                bufTensor = QWTensorValue(
                    region: QWMRegion(buffer: buf, base: 0, length: bytes),
                    offset: 0,
                    shape: QWTensorShape([bytes / 2], .f16)
                )
            }
            let bits = entry.hqBits
            let grp = entry.hqGroup
            guard let view = QWWeightView(
                blocks: bufTensor,
                n: n,
                k: k,
                bits: bits,
                groupSize: grp,
                isRotated: rot,
                isIncumbentContainer: rot && entry.isIncumbentHQ4,
                isSHVQ: entry.isSHVQ
            ) else {
                throw QWEngineDriverError.weightMissing(name, "invalid DSpark QWWeightView [\(n), \(k)]")
            }
            return view
        }

        func resolveName(_ candidates: [String], in p: QWQuant.Pack = pack) -> String? {
            for cand in candidates {
                if (try? p.entry(cand)) != nil { return cand }
            }
            return nil
        }

        func readF32Array(_ p: QWQuant.Pack, name: String, expectedCount: Int) throws -> [Float] {
            let e = try p.entry(name)
            let count = e.shape.reduce(1, *)
            guard count == expectedCount else {
                throw QWEngineDriverError.weightMissing(name, "shape count \(count) != \(expectedCount)")
            }
            if let layout = PlaneLayout.resolve(e), layout == .hq4 {
                return try p.dequantQ4g(name)
            }
            if e.kind == "f16" {
                let raw = try io.load(name, layout: .raw)
                return raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
                    let p16 = buf.bindMemory(to: UInt16.self)
                    return (0..<expectedCount).map { Float(Float16(bitPattern: p16[$0])) }
                }
            }
            return try p.rawBf16(name)
        }

        func allocateF16(_ name: String, expected: Int, shift: Float) throws -> QWTensorValue {
            let vals = try readF32Array(pack, name: name, expectedCount: expected)
            var bits = [UInt16](repeating: 0, count: expected)
            for i in 0..<expected { bits[i] = QWQuant.f32tof16Bits(vals[i] - shift) }
            return try bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                try scratchAlloc.append(raw: raw, shape: QWTensorShape([expected], .f16))
            }
        }

        func allocateF32(_ name: String, expected: Int) throws -> QWTensorValue {
            let vals = try readF32Array(pack, name: name, expectedCount: expected)
            return try vals.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                try scratchAlloc.append(raw: raw, shape: QWTensorShape([expected], .f32))
            }
        }

        func allocateRawBF16(_ name: String, expected: Int) throws -> QWTensorValue {
            let e = try pack.entry(name)
            let count = e.shape.reduce(1, *)
            guard count == expected else {
                throw QWEngineDriverError.weightMissing(name, "shape count \(count) != \(expected)")
            }
            if e.rawOff != nil, e.kind == nil || e.kind == "bf16" {
                let raw = try io.load(name, layout: .raw)
                return try raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
                    try scratchAlloc.append(raw: buf, shape: QWTensorShape([expected], .bf16))
                }
            }
            let vals = try readF32Array(pack, name: name, expectedCount: expected)
            var bfBits = [UInt16](repeating: 0, count: expected)
            for i in 0..<expected {
                bfBits[i] = UInt16(truncatingIfNeeded: vals[i].bitPattern >> 16)
            }
            return try bfBits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                try scratchAlloc.append(raw: raw, shape: QWTensorShape([expected], .bf16))
            }
        }

        let H = geometry.hidden
        let V = geometry.vocab
        let R = config.markovRank
        let tapK = numTaps * H

        guard let mainProjName = resolveName(["dspark.main_proj.weight", "main_proj.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.main_proj.weight", "missing in dspark.qwp")
        }
        self.mainProj = try mapOrLoadQuantView(pack, io, name: mainProjName, n: H, k: tapK)

        guard let mainNormName = resolveName(["dspark.main_norm.weight", "main_norm.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.main_norm.weight", "missing in dspark.qwp")
        }
        self.mainNorm = try allocateF16(mainNormName, expected: H, shift: 0)

        guard let outNormName = resolveName(["dspark.norm.weight", "norm.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.norm.weight", "missing in dspark.qwp")
        }
        self.finalNorm = try allocateF16(outNormName, expected: H, shift: 0)

        if let hcCfg = geometry.mhcConfig {
            let C = hcCfg.streams
            if let cwName = resolveName(["dspark.mhc_contract.weight", "mhc_contract.weight"]),
               let cbName = resolveName(["dspark.mhc_contract.bias", "mhc_contract.bias"]) {
                self.mhcContractW = try allocateF32(cwName, expected: C * C * H)
                self.mhcContractB = try allocateF32(cbName, expected: C)
            } else {
                self.mhcContractW = nil
                self.mhcContractB = nil
            }
        } else {
            self.mhcContractW = nil
            self.mhcContractB = nil
        }

        guard let w1Name = resolveName(["dspark.markov_w1.weight", "markov_w1.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.markov_w1.weight", "missing in dspark.qwp")
        }
        let w1Entry = try pack.entry(w1Name)
        guard w1Entry.shape == [V, R],
              let w1Layout = PlaneLayout.resolve(w1Entry), w1Layout == .raw,
              let w1Off = w1Layout.offset(w1Entry),
              let w1Bytes = w1Layout.byteCount(w1Entry), w1Bytes == V * R * 2 else {
            throw QWEngineDriverError.weightMissing(w1Name, "expected raw BF16 [\(V), \(R)]")
        }
        self.markovW1 = try mapOrLoadRaw(
            pack, io, name: w1Name, offset: w1Off, bytes: w1Bytes,
            shape: QWTensorShape([V, R], .bf16)
        )

        guard let w2Name = resolveName(["dspark.markov_w2.weight", "markov_w2.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.markov_w2.weight", "missing in dspark.qwp")
        }
        let w2View = try mapOrLoadQuantView(pack, io, name: w2Name, n: V, k: R)
        guard w2View.bits == 8, w2View.groupSize == 32 else {
            throw QWEngineDriverError.weightMissing(w2Name, "DSpark Markov W2 requires hq8_g32 (got b=\(w2View.bits) g=\(w2View.groupSize))")
        }
        self.markovW2 = w2View

        if let confName = resolveName(["dspark.confidence_head.weight", "confidence_head.weight"]) {
            self.confidenceHead = try allocateRawBF16(confName, expected: H + R)
        } else {
            self.confidenceHead = nil
        }

        func loadDraftHead(
            from p: QWQuant.Pack,
            io pIO: PlaneIO,
            candidates: [String]
        ) throws -> DraftHeadStorage? {
            guard let dhName = resolveName(candidates, in: p) else { return nil }
            let dhEntry = try p.entry(dhName)
            guard dhEntry.shape == [V, H] else {
                throw QWEngineDriverError.weightMissing(dhName, "draft_head shape \(dhEntry.shape) != [\(V), \(H)]")
            }
            if let layout = PlaneLayout.resolve(dhEntry), layout == .hq4 {
                return .quantized(try mapOrLoadQuantView(p, pIO, name: dhName, n: V, k: H))
            }
            if let layout = PlaneLayout.resolve(dhEntry), layout == .raw,
               let off = layout.offset(dhEntry),
               let bytes = layout.byteCount(dhEntry),
               bytes == V * H * 2,
               (dhEntry.kind ?? "bf16") == "bf16" {
                let rawT = try mapOrLoadRaw(
                    p, pIO, name: dhName, offset: off, bytes: bytes,
                    shape: QWTensorShape([V, H], .bf16)
                )
                return .bf16(rawT, n: V, k: H)
            }
            throw QWEngineDriverError.weightMissing(dhName, "unsupported draft_head layout")
        }

        let siblingHead = try headPack.flatMap { hp in
            try headIO.flatMap { hIO in
                try loadDraftHead(
                    from: hp, io: hIO,
                    candidates: ["draft_head.weight", "lm_head.weight", "dspark.draft_head.weight", "dspark.lm_head.weight"]
                )
            }
        }
        self.draftHead = try siblingHead ?? loadDraftHead(
            from: pack, io: io,
            candidates: ["draft_head.weight", "dspark.draft_head.weight"]
        )

        let p0 = "dspark.layers.0."
        guard let qDown0 = resolveName([p0 + "self_attn.linear_q_down_proj.weight"]),
              let kv0 = resolveName([p0 + "self_attn.linear_kv_proj.weight"]),
              let qUp0 = resolveName([p0 + "self_attn.q_up_gate_proj.weight"]),
              let woA0 = resolveName([p0 + "self_attn.wo_a.weight"]) else {
            throw QWEngineDriverError.weightMissing("dspark.layers.0.self_attn", "missing HCA projections")
        }
        let qLoraRank = try pack.entry(qDown0).shape[0]
        let headDim = try pack.entry(kv0).shape[0]
        let qUpStride = try pack.entry(qUp0).shape[0]
        let woAEntry = try pack.entry(woA0)
        let zOutStride = woAEntry.shape[0]
        let groupIn = woAEntry.shape[1]
        let targetHCA = geometry.hcaConfig
        let qHeads = targetHCA?.qHeads ?? max(1, (qUpStride - zOutStride) / max(1, headDim))
        let ropeDim = targetHCA?.qkRopeHeadDim ?? min(64, headDim)
        let nopeDim = headDim - ropeDim
        let oGroups = max(1, (qHeads * headDim) / max(1, groupIn))
        let oLoraRank = zOutStride / oGroups
        let attnGeo = DraftAttentionGeometry(
            qHeads: qHeads, headDim: headDim, nopeDim: nopeDim, ropeDim: ropeDim,
            qLoraRank: qLoraRank, oGroups: oGroups, oLoraRank: oLoraRank
        )
        self.attnGeometry = attnGeo

        var inferredExperts = geometry.numExperts
        var inferredInter = geometry.moeIntermediate
        var inferredShared = geometry.sharedExpertIntermediate
        var inferredLatent = geometry.moeLatentSize ?? 0
        if let fc1Name = resolveName([p0 + "mlp.fc1.weight"]) {
            inferredLatent = try pack.entry(fc1Name).shape[0]
        }
        if let expName = resolveName([p0 + "mlp.experts.gate_up_proj"]) {
            let shp = try pack.entry(expName).shape
            if shp.count == 3 {
                inferredExperts = shp[0]
                inferredInter = shp[1] / 2
            }
        }
        if let shDown = resolveName([p0 + "mlp.shared_experts.down_proj.weight"]) {
            inferredShared = try pack.entry(shDown).shape[1]
        }
        self.moeExperts = inferredExperts
        self.moeIntermediate = inferredInter
        self.moeSharedIntermediate = inferredShared
        self.moeLatentDim = inferredLatent

        var loadedLayers: [DraftLayer] = []
        loadedLayers.reserveCapacity(config.numHiddenLayers)
        for idx in 0..<config.numHiddenLayers {
            let blk = "dspark.layers.\(idx)."
            let ln1 = try allocateF16(blk + "input_layernorm.weight", expected: H, shift: normShift)
            let ln2 = try allocateF16(blk + "post_attention_layernorm.weight", expected: H, shift: normShift)
            let mhcW: QWEngineDriver.MHCLayerWeights?
            if let hcCfg = geometry.mhcConfig,
               let awName = resolveName([blk + "mhc_attn.weight"]),
               let abName = resolveName([blk + "mhc_attn.bias"]),
               let fwName = resolveName([blk + "mhc_ffn.weight"]),
               let fbName = resolveName([blk + "mhc_ffn.bias"]) {
                let C = hcCfg.streams
                let ctrlCount = 2 * C + C * C
                mhcW = QWEngineDriver.MHCLayerWeights(
                    attnW: try allocateF32(awName, expected: ctrlCount * C * H),
                    attnB: try allocateF32(abName, expected: ctrlCount),
                    ffnW: try allocateF32(fwName, expected: ctrlCount * C * H),
                    ffnB: try allocateF32(fbName, expected: ctrlCount)
                )
            } else {
                mhcW = nil
            }

            let qDownProj = try mapOrLoadQuantView(pack, io, name: blk + "self_attn.linear_q_down_proj.weight", n: qLoraRank, k: H)
            let kvProj = try mapOrLoadQuantView(pack, io, name: blk + "self_attn.linear_kv_proj.weight", n: headDim, k: H)
            let qNormW = try allocateF16(blk + "self_attn.q_norm.weight", expected: qLoraRank, shift: 0)
            let kvNormW = try allocateF16(blk + "self_attn.kv_norm.weight", expected: headDim, shift: 0)
            let qUpGateProj = try mapOrLoadQuantView(pack, io, name: blk + "self_attn.q_up_gate_proj.weight", n: qUpStride, k: qLoraRank)
            let woA = try mapOrLoadQuantView(pack, io, name: blk + "self_attn.wo_a.weight", n: zOutStride, k: groupIn)
            let oProj = try mapOrLoadQuantView(pack, io, name: blk + "self_attn.o_proj.weight", n: H, k: zOutStride)

            let vhaMix: QWTensorValue? = try resolveName([blk + "self_attn.vha_mix.weight"]).map {
                try allocateF16($0, expected: qHeads * qHeads, shift: 0)
            }
            let attnSink: QWTensorValue? = try resolveName([blk + "self_attn.attn_sink"]).map {
                try allocateF32($0, expected: qHeads)
            }
            let hcaAdapter = QWHCAAttnOperator.LayerWeights(
                wFusedIn: nil,
                qNormW: qNormW,
                kvNormW: kvNormW,
                compNormW: kvNormW,
                ape: attnSink ?? kvNormW,
                wQUpGate: qUpGateProj,
                woA: woA,
                woARawF16: nil,
                woB: oProj,
                vhaMix: vhaMix,
                attnSink: attnSink,
                lsaWQIdx: nil,
                lsaWKIdx: nil
            )

            let moeW: QWEngineDriver.MoEW?
            let denseGU: QWWeightView?
            let denseDn: QWWeightView?
            if let gateName = resolveName([blk + "mlp.gate.weight"]) {
                let gateEntry = try pack.entry(gateName)
                let split = gateEntry.shape[0] == 2 * inferredExperts
                let routerRows = split ? (2 * inferredExperts) : inferredExperts
                let router = try allocateF32(gateName, expected: routerRows * H)
                let routerBias = try resolveName([blk + "mlp.gate.e_score_correction_bias"]).map {
                    try allocateF32($0, expected: inferredExperts)
                }
                let routedScale = try resolveName([blk + "mlp.routed_scaling_factor_param"]).map {
                    try allocateF32($0, expected: inferredExperts)
                }
                let latent: QWMoEOperator.LatentBottleneck?
                let routedDim: Int
                if inferredLatent > 0,
                   let fc1Name = resolveName([blk + "mlp.fc1.weight"]),
                   let lnName = resolveName([blk + "mlp.latent_norm.weight"]),
                   let fc2Name = resolveName([blk + "mlp.fc2.weight"]) {
                    latent = QWMoEOperator.LatentBottleneck(
                        fc1: try mapOrLoadQuantView(pack, io, name: fc1Name, n: inferredLatent, k: H),
                        normGamma: try allocateF16(lnName, expected: inferredLatent, shift: normShift),
                        normEps: geometry.rmsEps,
                        fc2: try mapOrLoadQuantView(pack, io, name: fc2Name, n: H, k: inferredLatent),
                        latentDim: inferredLatent
                    )
                    routedDim = inferredLatent
                } else {
                    latent = nil
                    routedDim = H
                }

                let sharedGU: QWWeightView?
                let sharedDn: QWWeightView?
                if inferredShared > 0 {
                    if let guShName = resolveName([blk + "mlp.shared_experts.gate_up_proj.weight"]) {
                        sharedGU = try mapOrLoadQuantView(pack, io, name: guShName, n: 2 * inferredShared, k: H)
                    } else if let gName = resolveName([blk + "mlp.shared_experts.gate_proj.weight"]),
                              let uName = resolveName([blk + "mlp.shared_experts.up_proj.weight"]) {
                        sharedGU = try Self.concatQuantPair(
                            pack: pack, io: io, device: hal.device, buffers: &buffers,
                            firstName: gName, secondName: uName,
                            firstRows: inferredShared, secondRows: inferredShared, k: H, rot: rot
                        )
                    } else {
                        sharedGU = nil
                    }
                    sharedDn = try resolveName([blk + "mlp.shared_experts.down_proj.weight"]).map {
                        try mapOrLoadQuantView(pack, io, name: $0, n: H, k: inferredShared)
                    }
                } else {
                    sharedGU = nil
                    sharedDn = nil
                }

                let expertsGU: QWWeightView
                let expertsDn: QWWeightView
                if let gu3Name = resolveName([blk + "mlp.experts.gate_up_proj"]),
                   let dn3Name = resolveName([blk + "mlp.experts.down_proj"]) {
                    expertsGU = try mapOrLoadQuantView(pack, io, name: gu3Name, n: inferredExperts * 2 * inferredInter, k: routedDim)
                    expertsDn = try mapOrLoadQuantView(pack, io, name: dn3Name, n: inferredExperts * routedDim, k: inferredInter)
                } else {
                    (expertsGU, expertsDn) = try Self.concatPerExpertStack(
                        pack: pack, io: io, device: hal.device, buffers: &buffers,
                        layerPrefix: blk, experts: inferredExperts, inter: inferredInter, k: routedDim, rot: rot
                    )
                }

                let routing = geometry.moeRoutingConfig
                moeW = QWEngineDriver.MoEW(
                    router: router,
                    gateUp: expertsGU,
                    down: expertsDn,
                    numExperts: inferredExperts,
                    expertsPerTok: geometry.expertsPerTok,
                    sharedGateUp: sharedGU,
                    sharedDown: sharedDn,
                    routerBias: routerBias,
                    routedScaleParam: routedScale,
                    splitFeatureRouting: split,
                    latent: latent,
                    useSigmoidScore: geometry.hasBiasCorrectedRouter || routing?.scoreFunction == "sigmoid",
                    nGroup: routing?.nGroup ?? 1,
                    topkGroup: routing?.topkGroup ?? 1,
                    groupTopM: routing?.groupTopM ?? 1,
                    routedSwigluClamp: geometry.routedSwiGLUClamp(layer: 0),
                    sharedSwigluClamp: geometry.sharedSwiGLUClamp(layer: 0)
                )
                denseGU = nil
                denseDn = nil
            } else {
                moeW = nil
                if let guName = resolveName([blk + "mlp.gate_up_proj.weight"]) {
                    denseGU = try mapOrLoadQuantView(pack, io, name: guName, n: 2 * inferredInter, k: H)
                } else {
                    let gName = blk + "mlp.gate_proj.weight"
                    let uName = blk + "mlp.up_proj.weight"
                    denseGU = try Self.concatQuantPair(
                        pack: pack, io: io, device: hal.device, buffers: &buffers,
                        firstName: gName, secondName: uName,
                        firstRows: inferredInter, secondRows: inferredInter, k: H, rot: rot
                    )
                }
                denseDn = try mapOrLoadQuantView(pack, io, name: blk + "mlp.down_proj.weight", n: H, k: inferredInter)
            }

            loadedLayers.append(DraftLayer(
                ln1: ln1, ln2: ln2, mhc: mhcW,
                qDownProj: qDownProj, kvProj: kvProj,
                qNormW: qNormW, kvNormW: kvNormW,
                qUpGateProj: qUpGateProj, woA: woA, oProj: oProj,
                vhaMix: vhaMix, attnSink: attnSink, hcaAdapter: hcaAdapter,
                moe: moeW, denseGateUp: denseGU, denseDown: denseDn
            ))
        }

        buffers.append(contentsOf: scratchAlloc.buffers)
        hal.makeResident(buffers)
        self.layers = loadedLayers
        self.allBuffers = buffers
    }

    private static func concatQuantPair(
        pack: QWQuant.Pack,
        io: PlaneIO,
        device: any MTLDevice,
        buffers: inout [any MTLBuffer],
        firstName: String,
        secondName: String,
        firstRows: Int,
        secondRows: Int,
        k: Int,
        rot: Bool
    ) throws -> QWWeightView {
        let e0 = try pack.entry(firstName)
        let e1 = try pack.entry(secondName)
        guard let l0 = PlaneLayout.resolve(e0), l0 == .hq4,
              let l1 = PlaneLayout.resolve(e1), l1 == .hq4,
              let b0 = l0.byteCount(e0), let b1 = l1.byteCount(e1),
              e0.hqBits == e1.hqBits, e0.hqGroup == e1.hqGroup,
              e0.isIncumbentHQ4 == e1.isIncumbentHQ4, e0.isSHVQ == e1.isSHVQ else {
            throw QWEngineDriverError.weightMissing(firstName, "mismatched quant pair layout")
        }
        let totalBytes = b0 + b1
        guard let buf = device.makeBuffer(length: max(256, totalBytes), options: .storageModeShared) else {
            throw QWEngineDriverError.state("DSpark concat buffer allocation failed")
        }
        try io.load(firstName, layout: .hq4, into: buf.contents())
        try io.load(secondName, layout: .hq4, into: buf.contents().advanced(by: b0))
        buffers.append(buf)
        let tensor = QWTensorValue(
            region: QWMRegion(buffer: buf, base: 0, length: totalBytes),
            offset: 0,
            shape: QWTensorShape([totalBytes / 2], .f16)
        )
        guard let view = QWWeightView(
            blocks: tensor, n: firstRows + secondRows, k: k,
            bits: e0.hqBits, groupSize: e0.hqGroup,
            isRotated: rot, isIncumbentContainer: rot && e0.isIncumbentHQ4, isSHVQ: e0.isSHVQ
        ) else {
            throw QWEngineDriverError.weightMissing(firstName, "invalid concatenated QWWeightView")
        }
        return view
    }

    private static func concatPerExpertStack(
        pack: QWQuant.Pack,
        io: PlaneIO,
        device: any MTLDevice,
        buffers: inout [any MTLBuffer],
        layerPrefix: String,
        experts: Int,
        inter: Int,
        k: Int,
        rot: Bool
    ) throws -> (gateUp: QWWeightView, down: QWWeightView) {
        var guNames: [String] = []
        var dnNames: [String] = []
        let useStackedGU = (try? pack.entry("\(layerPrefix)mlp.experts.0.gate_up_proj.weight")) != nil
        for e in 0..<experts {
            if useStackedGU {
                guNames.append("\(layerPrefix)mlp.experts.\(e).gate_up_proj.weight")
            } else {
                guNames.append("\(layerPrefix)mlp.experts.\(e).gate_proj.weight")
                guNames.append("\(layerPrefix)mlp.experts.\(e).up_proj.weight")
            }
            dnNames.append("\(layerPrefix)mlp.experts.\(e).down_proj.weight")
        }
        let guView = try concatQuantSeries(
            pack: pack, io: io, device: device, buffers: &buffers,
            names: guNames, n: experts * 2 * inter, k: k, rot: rot
        )
        let dnView = try concatQuantSeries(
            pack: pack, io: io, device: device, buffers: &buffers,
            names: dnNames, n: experts * k, k: inter, rot: rot
        )
        return (guView, dnView)
    }

    private static func concatQuantSeries(
        pack: QWQuant.Pack,
        io: PlaneIO,
        device: any MTLDevice,
        buffers: inout [any MTLBuffer],
        names: [String],
        n: Int,
        k: Int,
        rot: Bool
    ) throws -> QWWeightView {
        guard let first = names.first else {
            throw QWEngineDriverError.weightMissing("dspark.experts", "empty expert list")
        }
        let e0 = try pack.entry(first)
        var total = 0
        for name in names {
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), layout == .hq4,
                  let bytes = layout.byteCount(e) else {
                throw QWEngineDriverError.weightMissing(name, "expected quantized expert tensor")
            }
            total += bytes
        }
        guard let buf = device.makeBuffer(length: max(256, total), options: .storageModeShared) else {
            throw QWEngineDriverError.state("DSpark expert stack allocation failed")
        }
        var cursor = 0
        for name in names {
            let e = try pack.entry(name)
            let bytes = PlaneLayout.hq4.byteCount(e)!
            try io.load(name, layout: .hq4, into: buf.contents().advanced(by: cursor))
            cursor += bytes
        }
        buffers.append(buf)
        let tensor = QWTensorValue(
            region: QWMRegion(buffer: buf, base: 0, length: total),
            offset: 0,
            shape: QWTensorShape([total / 2], .f16)
        )
        guard let view = QWWeightView(
            blocks: tensor, n: n, k: k,
            bits: e0.hqBits, groupSize: e0.hqGroup,
            isRotated: rot, isIncumbentContainer: rot && e0.isIncumbentHQ4, isSHVQ: e0.isSHVQ
        ) else {
            throw QWEngineDriverError.weightMissing(first, "invalid expert stack QWWeightView")
        }
        return view
    }
}

private final class ChunkedMetalSlab {
    private let device: any MTLDevice
    private let defaultChunkBytes = 8 << 20
    private(set) var buffers: [any MTLBuffer] = []
    private var cursor = 0

    init(device: any MTLDevice) {
        self.device = device
    }

    func append(raw: UnsafeRawBufferPointer, shape: QWTensorShape) throws -> QWTensorValue {
        let len = max(1, raw.count)
        let aligned = (cursor + 255) & ~255
        if let cur = buffers.last, aligned + len <= cur.length {
            if let base = raw.baseAddress, raw.count > 0 {
                cur.contents().advanced(by: aligned).copyMemory(from: base, byteCount: raw.count)
            }
            cursor = aligned + max(256, len)
            return QWTensorValue(region: QWMRegion(buffer: cur, base: aligned, length: len), offset: 0, shape: shape)
        }
        let allocLen = max(defaultChunkBytes, (len + 4095) & ~4095)
        guard let buf = device.makeBuffer(length: allocLen, options: .storageModeShared) else {
            throw QWEngineDriverError.state("DSpark weight slab allocation failed (\(allocLen) B)")
        }
        buffers.append(buf)
        if let base = raw.baseAddress, raw.count > 0 {
            buf.contents().copyMemory(from: base, byteCount: raw.count)
        }
        cursor = max(256, len)
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: len), offset: 0, shape: shape)
    }
}
