// QWAQTestHelpers.swift — Test-only QWAQProducer & QWBF16 helpers for QWGEMMTests.
import Foundation
import QWQuant

extension QWAQProducer {
    static func repackMLXGroup(
        bits: Int, group: Int, packed: [UInt8], scale: Float, bias: Float
    ) throws -> Data {
        let payload = group * bits / 8
        precondition(packed.count == payload, "a group holds \(payload) payload bytes")
        guard headerHolds(scale: scale, bias: bias) else {
            throw QWAQProducerError.headerNotRepresentable(scale: scale, bias: bias)
        }
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: Float16(scale).bitPattern.littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(contentsOf: withUnsafeBytes(of: Float16(bias).bitPattern.littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(contentsOf: packed)
        return out
    }
}

extension QWBF16 {
    static func toFloat(_ u16: UInt16) -> Float {
        Float(bitPattern: UInt32(u16) << 16)
    }
}

extension QWAQBlock {
    static func bitsPerWeight(bits: Int, group: Int) -> Double {
        Double(bytes(bits: bits, group: group) * 8) / Double(group)
    }
}
