// QWDecisionPackEngine+Weights.swift — Zero-family QWP1/SafeTensors loader for native Metal decision packs.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWDecision
import QWGEMM
import QWHAL
import QWLoad
import QWQuant

enum QWDecisionLinearW: @unchecked Sendable {
    case bf16(QWGemmFP.BF16Bank)
    case quant(QWWeightView)

    var needsRotation: Bool {
        if case .quant(let v) = self { return v.containerKind.isRotated }
        return false
    }

    var n: Int {
        switch self {
        case .bf16(let b): return b.n
        case .quant(let v): return v.n
        }
    }

    var k: Int {
        switch self {
        case .bf16(let b): return b.k
        case .quant(let v): return v.k
        }
    }
}

struct QWDecisionConvWeights: @unchecked Sendable {
    let inProj: QWDecisionLinearW
    let convW: QWTensorValue
    let convB: QWTensorValue?
    let outProj: QWDecisionLinearW
    let kernelSize: Int
    let channels: Int
}

struct QWDecisionAttnLayerWeights: @unchecked Sendable {
    let qProj: QWDecisionLinearW
    let kProj: QWDecisionLinearW
    let vProj: QWDecisionLinearW
    let oProj: QWDecisionLinearW
    let qNorm: QWTensorValue?
    let kNorm: QWTensorValue?
    let gateProj: QWDecisionLinearW?
    let isSliding: Bool
    let slidingWindow: Int
    let ropeKind: String
}

enum QWDecisionLayerOperatorWeights: @unchecked Sendable {
    case conv(QWDecisionConvWeights)
    case attn(QWDecisionAttnLayerWeights)
}

struct QWDecisionLoadedLayer: @unchecked Sendable {
    let preBlockNorm: QWTensorValue
    let postBlockNorm: QWTensorValue?
    let op: QWDecisionLayerOperatorWeights
    let preFFNNorm: QWTensorValue
    let postFFNNorm: QWTensorValue?
    let w13: QWDecisionLinearW?
    let w1: QWDecisionLinearW?
    let w3: QWDecisionLinearW?
    let w2: QWDecisionLinearW
}

struct QWDecisionPackWeights: @unchecked Sendable {
    let embedTokens: QWTensorValue
    let lmHeadBF16: QWTensorValue?
    let finalNorm: QWTensorValue
    let layers: [QWDecisionLoadedLayer]
    let usesRotatedBasis: Bool
    let retainedBuffers: [any MTLBuffer]
}

private enum QWDecisionWeightSource {
    case qwp(QWQuant.Pack)
    case safetensors(QWSafeTensorsContainer)

    var allNames: Set<String> {
        switch self {
        case .qwp(let p): return Set(p.tensors.keys)
        case .safetensors(let s): return Set(s.tensors.keys)
        }
    }

    func has(_ name: String) -> Bool {
        switch self {
        case .qwp(let p): return p.tensors[name] != nil
        case .safetensors(let s): return s.tensor(name) != nil
        }
    }

    func adviseDontNeed() {
        switch self {
        case .qwp(let p): p.adviseDontNeed()
        case .safetensors(let s): s.adviseDontNeed()
        }
    }
}

enum QWDecisionWeightLoader {
    static func probeRotatedBasis(root: QWModelRoot, descriptor: QWModelDescriptor) throws -> Bool {
        guard root.kind == .qwp1 else { return false }
        let pack = try QWQuant.Pack(url: root.weights, descriptor: descriptor)
        for (_, entry) in pack.tensors where entry.rawOff == nil && entry.hq4Off != nil {
            if entry.hq?.vq != "aq" { return true }
        }
        return false
    }

    static func load(root: QWModelRoot, descriptor: QWModelDescriptor, hal: QWHal) throws -> QWDecisionPackWeights {
        let source: QWDecisionWeightSource
        switch root.kind {
        case .qwp1:
            source = .qwp(try QWQuant.Pack(url: root.weights, descriptor: descriptor))
        case .safetensorsSingle, .safetensorsSharded:
            source = .safetensors(try QWSafeTensorsContainer(url: root.weights, onlyTensors: nil))
        default:
            throw QWDecisionError.backendFailed("unsupported weight container for native decision engine: \(String(describing: root.kind))")
        }
        defer { source.adviseDontNeed() }

        var buffers: [any MTLBuffer] = []
        let hidden = descriptor.hidden
        let vocab = descriptor.vocab
        let ffn = descriptor.denseIntermediate ?? descriptor.hidden
        let qDim = descriptor.qHeads * descriptor.headDim
        let kvDim = descriptor.kvHeads * descriptor.headDim
        let prefix = resolvePrefix(source)

        let embName = resolveName(source, [prefix + "embed_tokens.weight", prefix + "tok_embeddings.weight", "embed_tokens.weight"])
            ?? "\(prefix)embed_tokens.weight"
        let embedTokens = try loadBF16Tensor(name: embName, rows: vocab, cols: hidden, source: source, hal: hal, buffers: &buffers)

        let lmHeadBF16: QWTensorValue?
        if descriptor.tieWordEmbeddings {
            lmHeadBF16 = embedTokens
        } else if let lmName = resolveName(source, ["lm_head.weight", prefix + "lm_head.weight"]) {
            lmHeadBF16 = try loadBF16Tensor(name: lmName, rows: vocab, cols: hidden, source: source, hal: hal, buffers: &buffers)
        } else {
            lmHeadBF16 = embedTokens
        }

        let normName = resolveName(source, [prefix + "embedding_norm.weight", prefix + "norm.weight", "norm.weight"])
            ?? "\(prefix)norm.weight"
        let finalNorm = try loadF16Vector(name: normName, count: hidden, source: source, hal: hal, buffers: &buffers)

        var anyRotated = false
        var loadedLayers: [QWDecisionLoadedLayer] = []
        loadedLayers.reserveCapacity(descriptor.numLayers)

        for l in 0..<descriptor.numLayers {
            let lp = "\(prefix)layers.\(l)."
            let preBlockName = resolveName(source, [lp + "operator_norm.weight", lp + "input_layernorm.weight"])
                ?? (lp + "operator_norm.weight")
            let preBlock = try loadF16Vector(name: preBlockName, count: hidden, source: source, hal: hal, buffers: &buffers)
            let postBlock = try resolveName(source, [lp + "post_operator_norm.weight"]).map {
                try loadF16Vector(name: $0, count: hidden, source: source, hal: hal, buffers: &buffers)
            }
            let preFFNName = resolveName(source, [lp + "ffn_norm.weight", lp + "post_attention_layernorm.weight"])
                ?? (lp + "ffn_norm.weight")
            let preFFN = try loadF16Vector(name: preFFNName, count: hidden, source: source, hal: hal, buffers: &buffers)
            let postFFN = try resolveName(source, [lp + "post_ffn_norm.weight"]).map {
                try loadF16Vector(name: $0, count: hidden, source: source, hal: hal, buffers: &buffers)
            }

            let lKind = descriptor.layerTypes[l]
            let op: QWDecisionLayerOperatorWeights
            if lKind == "conv" || lKind == "shortconv" {
                let kernel = descriptor.shortConvConfig?.kernelSize ?? 7
                let inProj = try loadLinear(name: lp + "conv.in_proj.weight", n: 3 * hidden, k: hidden, source: source, hal: hal, buffers: &buffers)
                let convW = try loadF16Vector(name: lp + "conv.conv.weight", count: hidden * kernel, source: source, hal: hal, buffers: &buffers)
                let convB = try resolveName(source, [lp + "conv.conv.bias", lp + "conv.bias"]).map {
                    try loadF16Vector(name: $0, count: hidden, source: source, hal: hal, buffers: &buffers)
                }
                let outProj = try loadLinear(name: lp + "conv.out_proj.weight", n: hidden, k: hidden, source: source, hal: hal, buffers: &buffers)
                anyRotated = anyRotated || inProj.needsRotation || outProj.needsRotation
                op = .conv(QWDecisionConvWeights(
                    inProj: inProj, convW: convW, convB: convB, outProj: outProj,
                    kernelSize: kernel, channels: hidden
                ))
            } else {
                let qW = try loadLinear(name: lp + "self_attn.q_proj.weight", n: qDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let kW = try loadLinear(name: lp + "self_attn.k_proj.weight", n: kvDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let vW = try loadLinear(name: lp + "self_attn.v_proj.weight", n: kvDim, k: hidden, source: source, hal: hal, buffers: &buffers)
                let oW = try loadLinear(name: lp + "self_attn.o_proj.weight", n: hidden, k: qDim, source: source, hal: hal, buffers: &buffers)
                let qNorm = try resolveName(source, [lp + "self_attn.q_norm.weight", lp + "self_attn.q_layernorm.weight"]).map {
                    try loadF16Vector(name: $0, count: descriptor.headDim, source: source, hal: hal, buffers: &buffers)
                }
                let kNorm = try resolveName(source, [lp + "self_attn.k_norm.weight", lp + "self_attn.k_layernorm.weight"]).map {
                    try loadF16Vector(name: $0, count: descriptor.headDim, source: source, hal: hal, buffers: &buffers)
                }
                let gateW = try resolveName(source, [lp + "self_attn.gate_proj.weight"]).map {
                    try loadLinear(name: $0, n: descriptor.qHeads, k: hidden, source: source, hal: hal, buffers: &buffers)
                }
                anyRotated = anyRotated || qW.needsRotation || kW.needsRotation || vW.needsRotation
                    || oW.needsRotation || (gateW?.needsRotation ?? false)
                let sw = descriptor.layerSlidingWindow(l)
                op = .attn(QWDecisionAttnLayerWeights(
                    qProj: qW, kProj: kW, vProj: vW, oProj: oW, qNorm: qNorm, kNorm: kNorm, gateProj: gateW,
                    isSliding: (lKind == "sliding_attention") || (sw > 0), slidingWindow: sw, ropeKind: lKind
                ))
            }

            let w1Name = resolveName(source, [lp + "feed_forward.w1.weight", lp + "mlp.gate_proj.weight"])
                ?? (lp + "feed_forward.w1.weight")
            let w3Name = resolveName(source, [lp + "feed_forward.w3.weight", lp + "mlp.up_proj.weight"])
                ?? (lp + "feed_forward.w3.weight")
            let w2Name = resolveName(source, [lp + "feed_forward.w2.weight", lp + "mlp.down_proj.weight"])
                ?? (lp + "feed_forward.w2.weight")
            let (w13, w1Sep, w3Sep) = try loadFusedGateUp(
                w1Name: w1Name, w3Name: w3Name, nEach: ffn, k: hidden,
                source: source, hal: hal, buffers: &buffers
            )
            let w2 = try loadLinear(name: w2Name, n: hidden, k: ffn, source: source, hal: hal, buffers: &buffers)
            anyRotated = anyRotated || (w13?.needsRotation ?? false)
                || (w1Sep?.needsRotation ?? false) || (w3Sep?.needsRotation ?? false) || w2.needsRotation

            loadedLayers.append(QWDecisionLoadedLayer(
                preBlockNorm: preBlock, postBlockNorm: postBlock, op: op,
                preFFNNorm: preFFN, postFFNNorm: postFFN,
                w13: w13, w1: w1Sep, w3: w3Sep, w2: w2
            ))
        }

        return QWDecisionPackWeights(
            embedTokens: embedTokens, lmHeadBF16: lmHeadBF16, finalNorm: finalNorm,
            layers: loadedLayers, usesRotatedBasis: anyRotated, retainedBuffers: buffers
        )
    }

    private static func resolvePrefix(_ source: QWDecisionWeightSource) -> String {
        for cand in ["model.language_model.", "language_model.model.", "model.", ""] {
            if source.has(cand + "embed_tokens.weight") || source.has(cand + "tok_embeddings.weight") {
                return cand
            }
        }
        return "model."
    }

    private static func resolveName(_ source: QWDecisionWeightSource, _ candidates: [String]) -> String? {
        candidates.first(where: { source.has($0) })
    }

    private static func loadFusedGateUp(
        w1Name: String,
        w3Name: String,
        nEach: Int,
        k: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> (w13: QWDecisionLinearW?, w1: QWDecisionLinearW?, w3: QWDecisionLinearW?) {
        if let fused = try tryLoadFusedQWPOrST(w1Name: w1Name, w3Name: w3Name, nEach: nEach, k: k, source: source, hal: hal, buffers: &buffers) {
            return (fused, nil, nil)
        }
        let w1 = try loadLinear(name: w1Name, n: nEach, k: k, source: source, hal: hal, buffers: &buffers)
        let w3 = try loadLinear(name: w3Name, n: nEach, k: k, source: source, hal: hal, buffers: &buffers)
        return (nil, w1, w3)
    }

    private static func tryLoadFusedQWPOrST(
        w1Name: String,
        w3Name: String,
        nEach: Int,
        k: Int,
        source: QWDecisionWeightSource,
        hal: QWHal,
        buffers: inout [any MTLBuffer]
    ) throws -> QWDecisionLinearW? {
        switch source {
        case .safetensors(let st):
            guard let t1 = st.tensor(w1Name), let t3 = st.tensor(w3Name),
                  t1.dtype == .bf16, t3.dtype == .bf16,
                  t1.byteLength == nEach * k * 2, t3.byteLength == nEach * k * 2 else { return nil }
            let totalBytes = (2 * nEach) * k * 2
            let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
            try st.withBytes(of: w1Name) { (r1: UnsafeRawBufferPointer) in
                if let p1 = r1.baseAddress { memcpy(buf.contents(), p1, t1.byteLength) }
            }
            try st.withBytes(of: w3Name) { (r3: UnsafeRawBufferPointer) in
                if let p3 = r3.baseAddress { memcpy(buf.contents().advanced(by: t1.byteLength), p3, t3.byteLength) }
            }
            let slice = QWFPBufferSlice(buffer: buf, offset: 0, bytes: totalBytes)
            return .bf16(QWGemmFP.BF16Bank(weights: slice, n: 2 * nEach, k: k))
        case .qwp(let pack):
            let e1 = try pack.entry(w1Name)
            let e3 = try pack.entry(w3Name)
            guard let l1 = PlaneLayout.resolve(e1), let o1 = l1.offset(e1), let b1 = l1.byteCount(e1),
                  let l3 = PlaneLayout.resolve(e3), let o3 = l3.offset(e3), let b3 = l3.byteCount(e3) else {
                return nil
            }
            if e1.rawOff != nil && e3.rawOff != nil {
                guard e1.kind != "f32", e3.kind != "f32", e1.kind != "f16", e3.kind != "f16",
                      b1 == nEach * k * 2, b3 == nEach * k * 2 else { return nil }
                let totalBytes = b1 + b3
                let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
                _ = try pack.withRawPointer(at: o1, byteCount: b1) { (p1: UnsafeRawPointer) in memcpy(buf.contents(), p1, b1) }
                _ = try pack.withRawPointer(at: o3, byteCount: b3) { (p3: UnsafeRawPointer) in memcpy(buf.contents().advanced(by: b1), p3, b3) }
                let slice = QWFPBufferSlice(buffer: buf, offset: 0, bytes: totalBytes)
                return .bf16(QWGemmFP.BF16Bank(weights: slice, n: 2 * nEach, k: k))
            }
            guard let k1 = containerKind(from: e1), let k3 = containerKind(from: e3), k1 == k3, b1 == b3 else {
                return nil
            }
            let totalBytes = b1 + b3
            let buf = try allocSharedBuffer(bytes: totalBytes, hal: hal, buffers: &buffers)
            _ = try pack.withRawPointer(at: o1, byteCount: b1) { (p1: UnsafeRawPointer) in memcpy(buf.contents(), p1, b1) }
            _ = try pack.withRawPointer(at: o3, byteCount: b3) { (p3: UnsafeRawPointer) in memcpy(buf.contents().advanced(by: b1), p3, b3) }
            let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([totalBytes / 2], .f16))
            guard let view = QWWeightView(blocks: tv, n: 2 * nEach, k: k, containerKind: k1) else { return nil }
            return .quant(view)
        }
    }

    private static func loadLinear(
        name: String, n: Int, k: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWDecisionLinearW {
        if case .qwp(let pack) = source {
            let e = try pack.entry(name)
            if e.rawOff == nil, let kind = containerKind(from: e),
               let layout = PlaneLayout.resolve(e), let off = layout.offset(e), let bytes = layout.byteCount(e) {
                let buf = try allocSharedBuffer(bytes: bytes, hal: hal, buffers: &buffers)
                try pack.withRawPointer(at: off, byteCount: bytes) { (ptr: UnsafeRawPointer) in
                    if bytes > 0 { memcpy(buf.contents(), ptr, bytes) }
                }
                let tv = QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([bytes / 2], .f16))
                guard let view = QWWeightView(blocks: tv, n: n, k: k, containerKind: kind) else {
                    throw QWDecisionError.backendFailed("invalid QWWeightView [\(n), \(k)] for \(name)")
                }
                return .quant(view)
            }
        }
        let tv = try loadBF16Tensor(name: name, rows: n, cols: k, source: source, hal: hal, buffers: &buffers)
        return .bf16(QWGemmFP.BF16Bank(weights: tv, n: n, k: k))
    }

    private static func loadBF16Tensor(
        name: String, rows: Int, cols: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue {
        let count = rows * cols
        let byteLen = count * 2
        let buf = try allocSharedBuffer(bytes: byteLen, hal: hal, buffers: &buffers)
        switch source {
        case .safetensors(let st):
            guard let t = st.tensor(name) else { throw QWDecisionError.backendFailed("missing SafeTensors tensor: \(name)") }
            try st.withBytes(of: name) { (raw: UnsafeRawBufferPointer) in
                guard let src = raw.baseAddress else { return }
                convertToBF16(src: src, srcDtype: t.dtype.rawValue, count: count, expectedSrcBytes: t.byteLength, dst: buf.contents())
            }
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let layout = PlaneLayout.resolve(e), let off = layout.offset(e), let bytes = layout.byteCount(e) else {
                throw QWDecisionError.backendFailed("invalid QWP1 layout for \(name)")
            }
            if e.rawOff != nil {
                try pack.withRawPointer(at: off, byteCount: bytes) { (src: UnsafeRawPointer) in
                    convertToBF16(src: src, srcDtype: e.kind ?? "bf16", count: count, expectedSrcBytes: bytes, dst: buf.contents())
                }
            } else {
                let f32 = try pack.dequantQ4g(name)
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<min(count, f32.count) { dst[i] = f32ToBF16(f32[i]) }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([rows, cols], .bf16))
    }

    private static func loadF16Vector(
        name: String, count: Int, source: QWDecisionWeightSource, hal: QWHal, buffers: inout [any MTLBuffer]
    ) throws -> QWTensorValue {
        let buf = try allocSharedBuffer(bytes: count * 2, hal: hal, buffers: &buffers)
        let dst = buf.contents().assumingMemoryBound(to: Float16.self)
        switch source {
        case .safetensors(let st):
            guard let t = st.tensor(name) else { throw QWDecisionError.backendFailed("missing SafeTensors tensor: \(name)") }
            try st.withBytes(of: name) { (raw: UnsafeRawBufferPointer) in
                guard let src = raw.baseAddress else { return }
                convertToF16(src: src, srcDtype: t.dtype.rawValue, count: count, expectedSrcBytes: t.byteLength, dst: dst)
            }
        case .qwp(let pack):
            let e = try pack.entry(name)
            guard let off = e.rawOff else { throw QWDecisionError.backendFailed("expected raw norm/conv tensor for \(name)") }
            let elemBytes = (e.kind == "f32") ? 4 : 2
            let byteCount = count * elemBytes
            try pack.withRawPointer(at: off, byteCount: byteCount) { (src: UnsafeRawPointer) in
                convertToF16(src: src, srcDtype: e.kind ?? "bf16", count: count, expectedSrcBytes: byteCount, dst: dst)
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length), offset: 0, shape: QWTensorShape([count], .f16))
    }

    private static func containerKind(from e: QWQTensorEntry) -> QWWeightView.ContainerKind? {
        guard e.rawOff == nil, e.hq4Off != nil else { return nil }
        let rotated = e.hq?.vq != "aq"
        if e.isSHVQ { return rotated ? .shvq(bits: e.hqBits, group: e.hqGroup) : nil }
        if e.isIncumbentHQ4 { return .incumbentHQ4G32 }
        return rotated ? .rotatedHQ(bits: e.hqBits, group: e.hqGroup) : .nativeAffine(bits: e.hqBits, group: e.hqGroup)
    }

    private static func allocSharedBuffer(bytes: Int, hal: QWHal, buffers: inout [any MTLBuffer]) throws -> any MTLBuffer {
        let len = max(64, qwAlignUp(bytes, 64))
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw QWDecisionError.backendFailed("failed to allocate Metal buffer of \(len) bytes")
        }
        if len > bytes { memset(buf.contents().advanced(by: bytes), 0, len - bytes) }
        buffers.append(buf)
        return buf
    }

    private static func convertToBF16(
        src: UnsafeRawPointer, srcDtype: String, count: Int, expectedSrcBytes: Int, dst: UnsafeMutableRawPointer
    ) {
        let low = srcDtype.lowercased()
        let out = dst.assumingMemoryBound(to: UInt16.self)
        if low == "f32" && expectedSrcBytes >= count * 4 {
            let fp = src.assumingMemoryBound(to: Float.self)
            for i in 0..<count { out[i] = f32ToBF16(fp[i]) }
        } else if low == "f16" && expectedSrcBytes >= count * 2 {
            let hp = src.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { out[i] = f32ToBF16(Float(hp[i])) }
        } else if expectedSrcBytes >= count * 2 {
            memcpy(dst, src, count * 2)
        }
    }

    private static func convertToF16(
        src: UnsafeRawPointer, srcDtype: String, count: Int, expectedSrcBytes: Int, dst: UnsafeMutablePointer<Float16>
    ) {
        let low = srcDtype.lowercased()
        if low == "f32" && expectedSrcBytes >= count * 4 {
            let fp = src.assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float16(fp[i]) }
        } else if low == "f16" && expectedSrcBytes >= count * 2 {
            memcpy(dst, src, count * 2)
        } else if expectedSrcBytes >= count * 2 {
            let bp = src.assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = Float16(Float(bitPattern: UInt32(bp[i]) << 16)) }
        }
    }

    @inline(__always)
    private static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }
}
