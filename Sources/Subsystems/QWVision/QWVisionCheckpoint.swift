// QWVisionCheckpoint.swift — Safetensors shard & QWP checkpoint reader for ViT.
import Foundation
import QWCore
import QWQuant

public enum QWVisionError: Error, CustomStringConvertible, Equatable {
    case missingTensor(String)
    case shapeMismatch(String)
    case badLayout(String)
    case goldenMissing(String)

    public var description: String {
        switch self {
        case .missingTensor(let s): return "QWVision: checkpoint tensor missing: \(s)"
        case .shapeMismatch(let s): return "QWVision: shape mismatch: \(s)"
        case .badLayout(let s): return "QWVision: layout law violated: \(s)"
        case .goldenMissing(let s): return "QWVision: golden missing: \(s)"
        }
    }
}

/// Reads ONE safetensors shard for a key prefix, f16/bf16->fp32 on demand.
struct QWSafetensors {
    enum DType: Sendable, Equatable {
        case f16, bf16, f32
        case unsupported(String)

        init(_ raw: String) {
            switch raw {
            case "F16": self = .f16
            case "BF16": self = .bf16
            case "F32": self = .f32
            default: self = .unsupported(raw)
            }
        }

        var elemBytes: Int {
            switch self {
            case .f16, .bf16: return 2
            case .f32: return 4
            case .unsupported: return 0
            }
        }

        var label: String {
            switch self {
            case .f16: return "F16"
            case .bf16: return "BF16"
            case .f32: return "F32"
            case .unsupported(let raw): return raw
            }
        }
    }

    struct Entry: Sendable {
        let offset: Int
        let endOffset: Int
        let shape: [Int]
        let dtype: DType
    }

    let data: Data
    var dataStart = 0
    let tensors: [String: Entry]

    init(path: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        self.data = d
        guard d.count > 8 else { throw QWVisionError.badLayout("st too small") }
        var lenU64: UInt64 = 0
        d.withUnsafeBytes { (r: UnsafeRawBufferPointer) in
            for i in 0..<8 { lenU64 |= UInt64(r[i]) << (8 * i) }
        }
        guard lenU64 > 0, lenU64 <= UInt64(d.count - 8) else {
            throw QWVisionError.badLayout("invalid safetensors header length in \(path)")
        }
        let len = Int(lenU64)
        guard let hdr = try JSONSerialization.jsonObject(with: d.subdata(in: 8..<(8 + len))) as? [String: Any] else {
            throw QWVisionError.badLayout("invalid safetensors header in \(path)")
        }
        // Data offsets are relative to 8 + len (after header JSON).
        self.dataStart = 8 + len
        var t: [String: Entry] = [:]
        for (k, v) in hdr where k != "__metadata__" {
            guard let m = v as? [String: Any],
                  let offsets = m["data_offsets"] as? [Int], offsets.count == 2,
                  let shape = m["shape"] as? [Int],
                  let dtypeRaw = m["dtype"] as? String else {
                throw QWVisionError.badLayout("malformed safetensors header entry '\(k)'")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0], shape.allSatisfy({ $0 >= 0 }) else {
                throw QWVisionError.badLayout("negative offset or shape in safetensors entry '\(k)'")
            }
            t[k] = Entry(offset: offsets[0], endOffset: offsets[1], shape: shape, dtype: DType(dtypeRaw))
        }
        self.tensors = t
    }

    func f32(_ key: String, want: [Int]?) throws -> [Float] {
        guard let t = tensors[key] else { throw QWVisionError.missingTensor(key) }
        if let w = want, t.shape != w {
            throw QWVisionError.shapeMismatch("\(key): \(t.shape) ≠ \(w)")
        }
        let elemBytes: Int
        switch t.dtype {
        case .f16, .bf16, .f32:
            elemBytes = t.dtype.elemBytes
        case .unsupported(let raw):
            throw QWVisionError.badLayout("QWVision: unsupported dtype \(raw) for '\(key)'")
        }
        var n = 1
        for dim in t.shape {
            guard dim >= 0 else {
                throw QWVisionError.badLayout("negative dimension in tensor '\(key)'")
            }
            let (prod, overflow) = n.multipliedReportingOverflow(by: dim)
            guard !overflow else {
                throw QWVisionError.badLayout("element count overflow in tensor '\(key)'")
            }
            n = prod
        }
        let (byteCount, bytesOverflow) = n.multipliedReportingOverflow(by: elemBytes)
        let (absStart, startOverflow) = dataStart.addingReportingOverflow(t.offset)
        let (absEnd, endOverflow) = absStart.addingReportingOverflow(byteCount)
        let spanBytes = t.endOffset - t.offset
        guard t.offset >= 0,
              !bytesOverflow, !startOverflow, !endOverflow,
              spanBytes == byteCount,
              absEnd <= data.count else {
            throw QWVisionError.badLayout("safetensors data_offsets out of bounds for '\(key)'")
        }
        if n == 0 { return [] }
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { (r: UnsafeRawBufferPointer) in
            out.withUnsafeMutableBufferPointer { ob in
                guard let rawBase = r.baseAddress, let outBase = ob.baseAddress else { return }
                let src = rawBase.advanced(by: absStart).assumingMemoryBound(to: UInt8.self)
                let outU32 = UnsafeMutableRawPointer(outBase).assumingMemoryBound(to: UInt32.self)
                switch t.dtype {
                case .f16:
                    for i in 0..<n {
                        let lo = UInt16(src[i &* 2]), hi = UInt16(src[i &* 2 &+ 1])
                        outBase[i] = Float(Float16(bitPattern: lo | (hi << 8)))
                    }
                case .bf16:
                    for i in 0..<n {
                        let lo = UInt32(src[i &* 2]), hi = UInt32(src[i &* 2 &+ 1])
                        outU32[i] = (lo | (hi << 8)) << 16
                    }
                case .f32:
                    memcpy(outBase, src, byteCount)
                case .unsupported:
                    break
                }
            }
        }
        return out
    }

    func shape(_ key: String) -> [Int]? { tensors[key]?.shape }
    func has(_ key: String) -> Bool { tensors[key] != nil }
}

/// Multi-shard or `.qwp` view over a model's vision tower weights.
public final class QWVisionCheckpoint: @unchecked Sendable {
    private let lock = NSLock()
    let index: [String: String]
    let externalKeys: [String: (shard: String, fullKey: String)]
    private var shards: [String: QWSafetensors] = [:]
    let dir: String
    let prefix: String
    public let config: QWVisionConfig
    private var pack: QWQuant.Pack?
    let packPrefix = "model.visual."

    private static let externalMergerAliases: [(logical: String, candidates: [String])] = [
        ("merger.linear_fc1.weight", ["linear_proj.0.weight", "multi_modal_projector.linear_1.weight", "mm_projector.0.weight"]),
        ("merger.linear_fc1.bias", ["linear_proj.0.bias", "multi_modal_projector.linear_1.bias", "mm_projector.0.bias"]),
        ("merger.linear_fc2.weight", ["linear_proj.2.weight", "multi_modal_projector.linear_2.weight", "mm_projector.2.weight"]),
        ("merger.linear_fc2.bias", ["linear_proj.2.bias", "multi_modal_projector.linear_2.bias", "mm_projector.2.bias"]),
    ]

    public init(pack packPath: String) throws {
        self.dir = packPath
        self.prefix = "vision_tower."
        self.index = [:]
        self.externalKeys = [:]
        let pk = try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        self.pack = pk
        self.config = QWVisionConfig(descriptor: pk.descriptor)
    }

    public init(dir: String) throws {
        self.dir = dir
        self.pack = nil
        let indexPath = dir + "/model.safetensors.index.json"
        var ext: [String: (shard: String, fullKey: String)] = [:]
        if FileManager.default.fileExists(atPath: indexPath) {
            let idxData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let obj = try JSONSerialization.jsonObject(with: idxData) as? [String: Any],
                  let wm = obj["weight_map"] as? [String: String] else {
                throw QWVisionError.badLayout("malformed vision index json at \(indexPath)")
            }
            let detected = wm.keys.contains(where: { $0.hasPrefix("model.visual.") })
                ? "model.visual."
                : (wm.keys.contains(where: { $0.hasPrefix("vision.") }) ? "vision." : "vision_tower.")
            self.prefix = detected
            var v: [String: String] = [:]
            for (k, shard) in wm where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = shard }
            for alias in Self.externalMergerAliases where v[alias.logical] == nil {
                if let hit = alias.candidates.first(where: { wm[$0] != nil }), let shard = wm[hit] {
                    ext[alias.logical] = (shard, hit)
                }
            }
            self.index = v
            self.externalKeys = ext
        } else {
            let single = try QWSafetensors(path: dir + "/model.safetensors")
            let detected = single.tensors.keys.contains(where: { $0.hasPrefix("model.visual.") })
                ? "model.visual."
                : (single.tensors.keys.contains(where: { $0.hasPrefix("vision.") }) ? "vision." : "vision_tower.")
            self.prefix = detected
            var v: [String: String] = [:]
            for k in single.tensors.keys where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = "model.safetensors" }
            for alias in Self.externalMergerAliases where v[alias.logical] == nil {
                if let hit = alias.candidates.first(where: { single.tensors[$0] != nil }) {
                    ext[alias.logical] = ("model.safetensors", hit)
                }
            }
            self.index = v
            self.externalKeys = ext
            self.shards["model.safetensors"] = single
        }
        if let cfgURL = QWModelRoot(url: URL(fileURLWithPath: dir)).configURL,
           let cfgData = try? Data(contentsOf: cfgURL),
           let desc = try? QWModelDescriptor.decode(configJSON: cfgData),
           desc.visDepth > 0 {
            self.config = QWVisionConfig(descriptor: desc)
        } else {
            self.config = Self.inferConfig(
                dir: dir,
                prefix: self.prefix,
                index: self.index,
                externalKeys: self.externalKeys,
                shards: &self.shards
            )
        }
    }

    private static func inferConfig(
        dir: String,
        prefix: String,
        index: [String: String],
        externalKeys: [String: (shard: String, fullKey: String)],
        shards: inout [String: QWSafetensors]
    ) -> QWVisionConfig {
        func lookupShape(_ key: String) -> [Int]? {
            let resolved: (file: String, fullKey: String)
            if let file = index[key] {
                resolved = (file, prefix + key)
            } else if let ext = externalKeys[key] {
                resolved = (ext.shard, ext.fullKey)
            } else {
                return nil
            }
            if let s = shards[resolved.file] { return s.shape(resolved.fullKey) }
            guard let s = try? QWSafetensors(path: dir + "/" + resolved.file) else { return nil }
            shards[resolved.file] = s
            return s.shape(resolved.fullKey)
        }
        let posShape = lookupShape("pos_embed.weight")
        if index["embeddings.patch_embedding.weight"] != nil {
            let inkPos = lookupShape("embeddings.position_embedding")
            let inkPE = lookupShape("embeddings.patch_embedding.weight")
            let dim = inkPos?.dropFirst().first ?? inkPE?.first ?? 1024
            let numPos = inkPos?.first ?? 1024
            let patchChan = inkPE?.dropFirst().first ?? 1200
            let patch = max(1, Int(Double(patchChan / 3).squareRoot().rounded()))
            let mlpInner = lookupShape("layers.0.mlp.fc1.weight")?.first ?? (dim * 4)
            let projInW = lookupShape("proj_in.weight")
            let merged = projInW?.last ?? (dim * 4)
            let projHidden = projInW?.first ?? merged
            let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
            let outDim = lookupShape("proj_out.weight")?.first ?? 2304
            var depth = 0
            while lookupShape("layers.\(depth).layernorm_before.weight") != nil { depth += 1 }
            let heads = max(1, dim / 64)
            return QWVisionConfig(
                dim: dim, heads: heads, headDim: dim / heads, depth: max(1, depth),
                mlpInner: mlpInner, patch: patch, temporal: 1, merge: merge,
                numPosEmbeds: numPos, outDim: outDim, imageTokenId: 200054, videoTokenId: -1,
                ropeTheta: 10_000.0, eps: 1e-5, architecture: .inkling2,
                projectionHiddenDim: projHidden, projectorBias: lookupShape("proj_in.bias") != nil
            )
        }
        let pwShape = lookupShape("patch_embed.proj.weight")
        let ref = QWVisionConfig.qwen3VLReference
        let dim = posShape?.dropFirst().first ?? pwShape?.first ?? ref.dim
        let numPosEmbeds = posShape?.first ?? 0
        var patch = ref.patch
        var temporal = ref.temporal
        if let pwShape, pwShape.count == 5 {
            if pwShape[1] == 3 {
                temporal = pwShape[2]
                patch = pwShape[3]
            } else if pwShape[4] == 3 {
                temporal = pwShape[1]
                patch = pwShape[2]
            }
        }
        let mlpInner = lookupShape("blocks.0.mlp.linear_fc1.weight")?.first ?? (dim * 4)
        let fc1Shape = lookupShape("merger.linear_fc1.weight")
        let merged = fc1Shape?.last ?? (dim * 4)
        let fc1Mid = fc1Shape?.first ?? merged
        let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
        let outDim = lookupShape("merger.linear_fc2.bias")?.first
            ?? lookupShape("merger.linear_fc2.weight")?.first
            ?? ref.outDim
        let hasProjBias = (lookupShape("merger.linear_fc1.bias") != nil)
            || (lookupShape("merger.linear_fc2.bias") != nil)
        var depth = 0
        while lookupShape("blocks.\(depth).norm1.weight") != nil { depth += 1 }
        let heads = (dim > 0 && dim % 72 == 0) ? (dim / 72) : max(1, dim / 64)
        let headDim = heads > 0 ? dim / heads : 0
        return QWVisionConfig(
            dim: dim,
            heads: heads,
            headDim: headDim,
            depth: max(depth, 1),
            mlpInner: mlpInner,
            patch: patch,
            temporal: temporal,
            merge: merge,
            numPosEmbeds: numPosEmbeds,
            outDim: outDim,
            imageTokenId: ref.imageTokenId,
            videoTokenId: ref.videoTokenId,
            projectionHiddenDim: fc1Mid,
            projectorBias: hasProjBias,
            mergerIntermediateDim: (fc1Mid != merged ? fc1Mid : nil)
        )
    }

    func releaseBacking() {
        lock.withLock {
            shards.removeAll()
            pack?.adviseDontNeed()
            pack = nil
        }
    }

    private func shardLocked(_ file: String) throws -> QWSafetensors {
        if let s = shards[file] { return s }
        let s = try QWSafetensors(path: dir + "/" + file)
        shards[file] = s
        return s
    }

    public func f32(_ key: String, want: [Int]? = nil) throws -> [Float] {
        if let p = lock.withLock({ pack }) {
            let f = try p.rawBf16(packPrefix + key)
            if let w = want, f.count != w.reduce(1, *) {
                throw QWVisionError.badLayout("\(key): pack \(f.count) ≠ \(w)")
            }
            return f
        }
        if let file = index[key] {
            let s = try lock.withLock { try shardLocked(file) }
            return try s.f32(prefix + key, want: want)
        }
        if let ext = externalKeys[key] {
            let s = try lock.withLock { try shardLocked(ext.shard) }
            return try s.f32(ext.fullKey, want: want)
        }
        throw QWVisionError.missingTensor(key)
    }

    public func shape(_ key: String) -> [Int]? {
        if let p = lock.withLock({ pack }) { return try? p.entry(packPrefix + key).shape }
        if let file = index[key] {
            return try? lock.withLock { try shardLocked(file) }.shape(prefix + key)
        }
        if let ext = externalKeys[key] {
            return try? lock.withLock { try shardLocked(ext.shard) }.shape(ext.fullKey)
        }
        return nil
    }

    func has(_ key: String) -> Bool { shape(key) != nil }
}
