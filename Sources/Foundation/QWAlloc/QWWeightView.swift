// QWWeightView.swift — Quantized weight view (format + shape + rotation basis).
import Foundation
import Metal
import QWCore
import QWQuant

/// Read-side view of one quantized weight tensor: it names the FORMAT as well
/// as the shape, so the arena can be sized at the pair's NATIVE bytes. It also
/// names the BASIS, and every construction states it (specs/04-qwquant.md).
public struct QWWeightView: Sendable {
    public let blocks: QWTensorValue
    public var nibbles: QWTensorValue { blocks }
    public var scales: QWTensorValue { blocks }
    public var mins: QWTensorValue { blocks }
    public let n: Int          // output rows
    public let k: Int          // inner dim (must be % groupSize)
    public let bits: Int       // 2, 3, 4, 6, 8
    public let groupSize: Int  // 32, 64, 128, 256
    public let isRotated: Bool // FWHT-rotated basis (HQ4_G32 or HQ*)
    public let isIncumbentContainer: Bool
    public let isSHVQ: Bool

    /// The bytes this bank's rows take at the pair's native size.
    public var rowBytes: Int {
        isSHVQ
            ? QWSHVQBlock.rowBytes(bits: bits, group: groupSize, k: k)
            : QWAQBlock.rowBytes(bits: bits, group: groupSize, k: k)
    }
    /// The bytes the whole bank takes at the pair's native size.
    public var requiredBytes: Int { rowBytes * n }
    /// Canonical container format name for logging and kernel routing.
    public var formatName: String {
        if isIncumbentContainer { return "HQ4_G32" }
        if isSHVQ { return QWSHVQBlock.formatName(bits: bits, group: groupSize) }
        if isRotated { return "HQ\(bits)_G\(groupSize)" }
        return QWAQBlock.formatName(bits: bits, group: groupSize)
    }

    /// `isRotated` has no default so every construction explicitly states the basis.
    public init?(blocks: QWTensorValue, n: Int, k: Int,
                 bits: Int = 4, groupSize: Int = 32, isRotated: Bool,
                 isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        let incumbent = isIncumbentContainer ?? (!isSHVQ && isRotated && bits == 4 && groupSize == 32)
        if isSHVQ {
            guard isRotated && !incumbent && QWSHVQBlock.isSupported(bits: bits, group: groupSize) else { return nil }
        } else if incumbent {
            guard isRotated && bits == 4 && groupSize == 32 else { return nil }
        } else if isRotated {
            guard QWAQBlock.isHQSupported(bits: bits, group: groupSize) else { return nil }
        } else {
            guard QWAQBlock.isSupported(bits: bits, group: groupSize) else { return nil }
        }
        let maybeGroupBytes = isSHVQ
            ? QWSHVQBlock.bytesIfSupported(bits: bits, group: groupSize)
            : QWAQBlock.bytesIfBlockSupported(bits: bits, group: groupSize)
        guard n > 0, k > 0, k % groupSize == 0,
              let groupBytes = maybeGroupBytes,
              blocks.bytes >= n * (k / groupSize) * groupBytes
        else { return nil }
        self.blocks = blocks
        self.n = n; self.k = k
        self.bits = bits; self.groupSize = groupSize
        self.isRotated = isRotated
        self.isIncumbentContainer = incumbent
        self.isSHVQ = isSHVQ
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
        QWWeightView(blocks: blocks, n: n, k: k,
                     bits: bits, groupSize: groupSize, isRotated: isRotated,
                     isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }
}

extension QWWeightView: Equatable {
    /// Two views are equal when format, shape, and underlying buffer slice match.
    public static func == (a: QWWeightView, b: QWWeightView) -> Bool {
        a.n == b.n && a.k == b.k
            && a.bits == b.bits && a.groupSize == b.groupSize
            && a.isRotated == b.isRotated
            && a.isIncumbentContainer == b.isIncumbentContainer
            && a.isSHVQ == b.isSHVQ
            && a.blocks.bytes == b.blocks.bytes
            && a.blocks.offset == b.blocks.offset
            && a.blocks.region.base == b.blocks.region.base
            && a.blocks.region.buffer === b.blocks.region.buffer
    }
}
