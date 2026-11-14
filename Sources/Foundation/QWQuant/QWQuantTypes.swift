// QWQuantTypes.swift — Quantization container errors, root locations, and tensor entry schemas.
import Foundation
import QWCore

public enum QWQuantError: Error, CustomStringConvertible {
    case badMagic
    case geometryMismatch(pack: String, expected: String)
    case modelMismatch(pack: String, expected: String)
    case geometryMissing(String)
    case tensorMissing(String)
    case truncated
    case notQuantPlane(String)
    case hq8SwitchOff(String)
    case notRawPlane(String)
    case modelFileMissing(String)

    public var description: String {
        switch self {
        case .badMagic: return "QWQuant: bad magic (want QWP1)"
        case .geometryMismatch(let p, let e): return "QWQuant: pack geometry \(p) ≠ engine \(e)"
        case .modelMismatch(let p, let e):
            return "QWQuant: pack model '\(p)' ≠ engine model '\(e)' — refuse before allocation"
        case .geometryMissing(let m): return "QWQuant: pack for model '\(m)' carries no geometry_digest"
        case .tensorMissing(let n): return "QWQuant: tensor '\(n)' not in pack"
        case .truncated: return "QWQuant: truncated pack"
        case .notQuantPlane(let n): return "QWQuant: '\(n)' is a verbatim-bf16 tensor (no hq4/q8 planes)"
        case .hq8SwitchOff(let n):
            return "QWQuant: '\(n)' is HQ8-packed but QW_DENSE_W8 is off — set QW_DENSE_W8=1"
        case .notRawPlane(let n): return "QWQuant: '\(n)' is a quantized plane (no raw_off)"
        case .modelFileMissing(let p): return "QWQuant: model weights not found: \(p)"
        }
    }
}

public struct QWModelLocation: Sendable {
    public let weights: URL
    public let assets: URL
    /// Which weight container the root resolved to.
    public let kind: QWWeightSourceKind
    /// The preference that produced `kind`.
    public let preference: QWWeightSourcePreference

    public init(url: URL,
                assetsOverride: URL? = nil,
                preference: QWWeightSourcePreference = .auto) throws {
        let root = QWModelRoot(url: url, preference: preference)
        guard let kind = root.kind else {
            throw QWQuantError.modelFileMissing(root.weights.path)
        }
        self.weights = root.weights
        self.kind = kind
        self.preference = preference
        self.assets = assetsOverride ?? root.assets
    }
}

public struct QWHQFormat: Decodable, Sendable, Equatable, Hashable {
    public let bits: Int
    public let group: Int
    public let vq: String?

    public init(bits: Int, group: Int, vq: String? = nil) {
        self.bits = bits
        self.group = group
        self.vq = vq
    }
}

public struct QWQTensorEntry: Decodable, Sendable {
    public let shape: [Int]
    public let hq4Off: Int?
    public var q4gOff: Int? { hq4Off }
    public let q8Off: Int?
    public let q8SOff: Int?
    public let hq8Off: Int?
    public let hq8SOff: Int?
    public let rawOff: Int?
    public let kind: String?
    /// Expert stacks: expert count and bytes per expert (`hq4_off + e*e_hq4`).
    public let experts: Int?
    public let eHq4: Int?
    public let hq: QWHQFormat?

    public init(shape: [Int], hq4Off: Int? = nil, q8Off: Int? = nil, q8SOff: Int? = nil,
                hq8Off: Int? = nil, hq8SOff: Int? = nil,
                rawOff: Int? = nil, kind: String? = nil, experts: Int? = nil, eHq4: Int? = nil,
                hq: QWHQFormat? = nil) {
        self.shape = shape
        self.hq4Off = hq4Off
        self.q8Off = q8Off
        self.q8SOff = q8SOff
        self.hq8Off = hq8Off
        self.hq8SOff = hq8SOff
        self.rawOff = rawOff
        self.kind = kind
        self.experts = experts
        self.eHq4 = eHq4
        self.hq = hq
    }

    enum CodingKeys: String, CodingKey {
        case shape
        case hq4Off = "hq4_off"
        case q8Off = "q8_off"
        case q8SOff = "q8_s_off"
        case hq8Off = "hq8_off"
        case hq8SOff = "hq8_s_off"
        case rawOff = "raw_off"
        case kind
        case experts
        case eHq4 = "e_hq4"
        case hq
    }

    public var hqBits: Int { hq?.bits ?? 4 }
    public var hqGroup: Int { hq?.group ?? QWQuant.group }
    public var isIncumbentHQ4: Bool { hq == nil }
    public var isSHVQ: Bool { hq?.vq == "shvq1" }
    public var hqBlockBytes: Int {
        if let hq {
            if hq.vq == "shvq1" {
                return QWSHVQBlock.bytes(bits: hq.bits, group: hq.group)
            }
            return QWAQBlock.bytes(bits: hq.bits, group: hq.group)
        }
        return QWQuant.blockBytes
    }
    public var nRows: Int? { shape.count == 2 ? shape[0] : nil }
    public var kCols: Int? { shape.count == 2 ? shape[1] : nil }
    public var groups: Int? { shape.count == 2 ? shape[1] / hqGroup : nil }
    public var isRawBf16: Bool { rawOff != nil }
    public var rawElemBytes: Int { kind == "f32" ? 4 : 2 }
    public var quantized: Bool { rawOff == nil && hq4Off != nil }
    public var hq8Quantized: Bool { rawOff == nil && hq8Off != nil && hq4Off == nil }

    /// Validated HQ8 plane extents `(payloadBytes, scaleBytes)` for a 2-D
    /// `hq8_g32_m5` entry (payload `N*K` int8, scale `N*(K/32)*2` f16).
    public var hq8Extents: (payload: Int, scale: Int)? {
        guard let n = nRows, let k = kCols, k % QWQuant.group == 0,
              hq8Off != nil, hq8SOff != nil else { return nil }
        let payload = n * k, scale = n * (k / QWQuant.group) * 2
        guard payload > 0, scale > 0 else { return nil }
        return (payload, scale)
    }
    public var hq8PayloadBytes: Int? { hq8Extents?.payload }
    public var hq8ScaleBytes: Int? { hq8Extents?.scale }


}

