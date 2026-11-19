// QWWeightView.swift — Quantized weight view (format + shape + rotation basis).
import Foundation
import Metal
import QWCore
import QWQuant

/// Read-side view of one quantized weight tensor: it names the FORMAT as well
/// as the shape, so the arena can be sized at the pair's NATIVE bytes.
public struct QWWeightView: Sendable {
    /// Canonical quantized container + rotation-basis discriminant.
    public enum ContainerKind: Sendable, Equatable, Hashable {
        case incumbentHQ4G32
        case shvq(bits: Int, group: Int)
        case rotatedHQ(bits: Int, group: Int)
        case nativeAffine(bits: Int, group: Int)

        public var bits: Int {
            switch self {
            case .incumbentHQ4G32: return 4
            case .shvq(let bits, _), .rotatedHQ(let bits, _), .nativeAffine(let bits, _):
                return bits
            }
        }

        public var groupSize: Int {
            switch self {
            case .incumbentHQ4G32: return 32
            case .shvq(_, let group), .rotatedHQ(_, let group), .nativeAffine(_, let group):
                return group
            }
        }

        public var isRotated: Bool {
            switch self {
            case .incumbentHQ4G32, .shvq, .rotatedHQ: return true
            case .nativeAffine: return false
            }
        }

        public var isIncumbentContainer: Bool { self == .incumbentHQ4G32 }

        public var isSHVQ: Bool {
            if case .shvq = self { return true }
            return false
        }

        public var formatName: String {
            switch self {
            case .incumbentHQ4G32:
                return "HQ4_G32"
            case .shvq(let bits, let group):
                return QWSHVQBlock.formatName(bits: bits, group: group)
            case .rotatedHQ(let bits, let group):
                return "HQ\(bits)_G\(group)"
            case .nativeAffine(let bits, let group):
                return QWAQBlock.formatName(bits: bits, group: group)
            }
        }

        public func rowBytes(k: Int) -> Int {
            switch self {
            case .shvq(let bits, let group):
                return QWSHVQBlock.rowBytes(bits: bits, group: group, k: k)
            case .incumbentHQ4G32, .rotatedHQ, .nativeAffine:
                return QWAQBlock.rowBytes(bits: bits, group: groupSize, k: k)
            }
        }

        internal var validatedGroupBytes: Int? {
            switch self {
            case .incumbentHQ4G32:
                return QWAQBlock.bytesIfBlockSupported(bits: 4, group: 32)
            case .shvq(let bits, let group):
                guard QWSHVQBlock.isSupported(bits: bits, group: group) else { return nil }
                return QWSHVQBlock.bytesIfSupported(bits: bits, group: group)
            case .rotatedHQ(let bits, let group):
                guard QWAQBlock.isHQSupported(bits: bits, group: group) else { return nil }
                return QWAQBlock.bytesIfBlockSupported(bits: bits, group: group)
            case .nativeAffine(let bits, let group):
                guard QWAQBlock.isSupported(bits: bits, group: group) else { return nil }
                return QWAQBlock.bytesIfBlockSupported(bits: bits, group: group)
            }
        }
    }

    public let blocks: QWTensorValue
    public var nibbles: QWTensorValue { blocks }
    public var scales: QWTensorValue { blocks }
    public var mins: QWTensorValue { blocks }
    public let n: Int          // output rows
    public let k: Int          // inner dim (must be % groupSize)
    public let containerKind: ContainerKind

    public var bits: Int { containerKind.bits }
    public var groupSize: Int { containerKind.groupSize }
    public var isRotated: Bool { containerKind.isRotated }
    public var isIncumbentContainer: Bool { containerKind.isIncumbentContainer }
    public var isSHVQ: Bool { containerKind.isSHVQ }

    /// The bytes this bank's rows take at the pair's native size.
    public var rowBytes: Int { containerKind.rowBytes(k: k) }
    /// The bytes the whole bank takes at the pair's native size.
    public var requiredBytes: Int { rowBytes * n }
    /// Canonical container format name for logging and kernel routing.
    public var formatName: String { containerKind.formatName }

    /// Designated container-typed initializer validating shape, group alignment, and backing bytes.
    public init?(blocks: QWTensorValue, n: Int, k: Int, containerKind: ContainerKind) {
        guard n > 0, k > 0, k % containerKind.groupSize == 0,
              let groupBytes = containerKind.validatedGroupBytes,
              blocks.bytes >= n * (k / containerKind.groupSize) * groupBytes
        else { return nil }
        self.blocks = blocks
        self.n = n
        self.k = k
        self.containerKind = containerKind
    }

    /// `isRotated` has no default so every construction explicitly states the basis.
    public init?(blocks: QWTensorValue, n: Int, k: Int,
                 bits: Int = 4, groupSize: Int = 32, isRotated: Bool,
                 isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        let incumbent = isIncumbentContainer ?? (!isSHVQ && isRotated && bits == 4 && groupSize == 32)
        let kind: ContainerKind
        if isSHVQ {
            guard isRotated && !incumbent else { return nil }
            kind = .shvq(bits: bits, group: groupSize)
        } else if incumbent {
            guard isRotated && bits == 4 && groupSize == 32 else { return nil }
            kind = .incumbentHQ4G32
        } else if isRotated {
            kind = .rotatedHQ(bits: bits, group: groupSize)
        } else {
            kind = .nativeAffine(bits: bits, group: groupSize)
        }
        self.init(blocks: blocks, n: n, k: k, containerKind: kind)
    }

    public init?(nibbles: QWTensorValue, scales: QWTensorValue? = nil, mins: QWTensorValue? = nil,
                 n: Int, k: Int,
                 bits: Int = 4, groupSize: Int = 32, isRotated: Bool,
                 isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        self.init(blocks: nibbles, n: n, k: k,
                  bits: bits, groupSize: groupSize, isRotated: isRotated,
                  isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }

    /// A sub-view over other bytes of the same bank inheriting format and basis.
    public func subview(blocks: QWTensorValue, n: Int, k: Int) -> QWWeightView? {
        QWWeightView(blocks: blocks, n: n, k: k, containerKind: containerKind)
    }
}

extension QWWeightView: Equatable {
    /// Two views are equal when format, shape, and underlying buffer slice match.
    public static func == (a: QWWeightView, b: QWWeightView) -> Bool {
        a.n == b.n && a.k == b.k
            && a.containerKind == b.containerKind
            && a.blocks.bytes == b.blocks.bytes
            && a.blocks.offset == b.blocks.offset
            && a.blocks.region.base == b.blocks.region.base
            && a.blocks.region.buffer === b.blocks.region.buffer
    }
}
