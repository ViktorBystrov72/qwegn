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
    let data: Data
    var dataStart = 0
    let tensors: [String: (offset: Int, shape: [Int], dtype: String)]

    init(path: String) throws {
        let d = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        self.data = d
        guard d.count > 8 else { throw QWVisionError.badLayout("st too small") }
        var len = 0
        d.withUnsafeBytes { (r: UnsafeRawBufferPointer) in for i in 0..<8 { len |= Int(r[i]) << (8 * i) } }
        guard len > 0, 8 + len <= d.count,
              let hdr = try JSONSerialization.jsonObject(with: d.subdata(in: 8..<(8 + len))) as? [String: Any] else {
            throw QWVisionError.badLayout("invalid safetensors header in \(path)")
        }
        // Data offsets are relative to 8 + len (after header JSON).
        self.dataStart = 8 + len
        var t: [String: (Int, [Int], String)] = [:]
        for (k, v) in hdr where k != "__metadata__" {
            guard let m = v as? [String: Any],
                  let offsets = m["data_offsets"] as? [Int], !offsets.isEmpty,
                  let shape = m["shape"] as? [Int],
                  let dtype = m["dtype"] as? String else {
                throw QWVisionError.badLayout("malformed safetensors header entry '\(k)'")
            }
            t[k] = (offsets[0], shape, dtype)
        }
        self.tensors = t
    }

    func f32(_ key: String, want: [Int]?) throws -> [Float] {
        guard let t = tensors[key] else { throw QWVisionError.missingTensor(key) }
        if let w = want, t.shape != w {
            throw QWVisionError.shapeMismatch("\(key): \(t.shape) ≠ \(w)")
        }
        guard t.dtype == "F16" || t.dtype == "BF16" || t.dtype == "F32" else {
            throw QWVisionError.badLayout("QWVision: unsupported dtype \(t.dtype) for '\(key)'")
        }
        let n = t.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { (r: UnsafeRawBufferPointer) in
            out.withUnsafeMutableBufferPointer { ob in
                guard let rawBase = r.baseAddress, let outBase = ob.baseAddress else { return }
                let src = rawBase.advanced(by: dataStart + t.offset).assumingMemoryBound(to: UInt8.self)
                let outU32 = UnsafeMutableRawPointer(outBase).assumingMemoryBound(to: UInt32.self)
                switch t.dtype {
                case "F16":
                    for i in 0..<n {
                        let lo = UInt16(src[i &* 2]), hi = UInt16(src[i &* 2 &+ 1])
                        outBase[i] = Float(Float16(bitPattern: lo | (hi << 8)))
                    }
                case "BF16":
                    for i in 0..<n {
                        let lo = UInt32(src[i &* 2]), hi = UInt32(src[i &* 2 &+ 1])
                        outU32[i] = (lo | (hi << 8)) << 16
                    }
                case "F32":
                    memcpy(outBase, src, n * 4)
                default:
                    fatalError("QWVision: dtype \(t.dtype) not in the law")
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
    let index: [String: String]
    var shards: [String: QWSafetensors] = [:]
    let dir: String
    let prefix: String
    public let config: QWVisionConfig
    var pack: QWQuant.Pack?
    let packPrefix = "model.visual."

    public init(pack packPath: String) throws {
        self.dir = packPath
        self.prefix = "vision_tower."
        self.index = [:]
        let pk = try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        self.pack = pk
        self.config = QWVisionConfig(descriptor: pk.descriptor)
    }

    public init(dir: String) throws {
        self.dir = dir
        self.pack = nil
        let indexPath = dir + "/model.safetensors.index.json"
        if FileManager.default.fileExists(atPath: indexPath) {
            let idxData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let obj = try JSONSerialization.jsonObject(with: idxData) as? [String: Any],
                  let wm = obj["weight_map"] as? [String: String] else {
                throw QWVisionError.badLayout("malformed vision index json at \(indexPath)")
            }
            let detected = wm.keys.contains(where: { $0.hasPrefix("model.visual.") }) ? "model.visual." : "vision_tower."
            self.prefix = detected
            var v: [String: String] = [:]
            for (k, shard) in wm where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = shard }
            self.index = v
        } else {
            let single = try QWSafetensors(path: dir + "/model.safetensors")
            let detected = single.tensors.keys.contains(where: { $0.hasPrefix("model.visual.") }) ? "model.visual." : "vision_tower."
            self.prefix = detected
            var v: [String: String] = [:]
            for k in single.tensors.keys where k.hasPrefix(detected) { v[String(k.dropFirst(detected.count))] = "model.safetensors" }
            self.index = v
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
                shards: &self.shards
            )
        }
    }

    private static func inferConfig(
        dir: String,
        prefix: String,
        index: [String: String],
        shards: inout [String: QWSafetensors]
    ) -> QWVisionConfig {
        func lookupShape(_ key: String) -> [Int]? {
            guard let file = index[key] else { return nil }
            if let s = shards[file] { return s.shape(prefix + key) }
            guard let s = try? QWSafetensors(path: dir + "/" + file) else { return nil }
            shards[file] = s
            return s.shape(prefix + key)
        }
        let posShape = lookupShape("pos_embed.weight")
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
        let merged = lookupShape("merger.linear_fc1.weight")?.last ?? (dim * 4)
        let merge = dim > 0 ? max(1, Int(Double(merged / dim).squareRoot().rounded())) : 2
        let outDim = lookupShape("merger.linear_fc2.bias")?.first
            ?? lookupShape("merger.linear_fc2.weight")?.first
            ?? ref.outDim
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
            videoTokenId: ref.videoTokenId
        )
    }

    func shard(_ file: String) throws -> QWSafetensors {
        if let s = shards[file] { return s }
        let s = try QWSafetensors(path: dir + "/" + file)
        shards[file] = s
        return s
    }

    public func f32(_ key: String, want: [Int]? = nil) throws -> [Float] {
        if let p = pack {
            let f = try p.rawBf16(packPrefix + key)
            if let w = want, f.count != w.reduce(1, *) {
                throw QWVisionError.badLayout("\(key): pack \(f.count) ≠ \(w)")
            }
            return f
        }
        guard let file = index[key] else { throw QWVisionError.missingTensor(key) }
        return try shard(file).f32(prefix + key, want: want)
    }

    public func shape(_ key: String) -> [Int]? {
        if let p = pack { return try? p.entry(packPrefix + key).shape }
        guard let file = index[key] else { return nil }
        return try? shard(file).shape(prefix + key)
    }

    func has(_ key: String) -> Bool { shape(key) != nil }
}
