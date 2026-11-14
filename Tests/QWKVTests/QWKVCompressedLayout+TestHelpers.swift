// QWKVCompressedLayout+TestHelpers.swift — Test-only KV Q8/Q4 dequantization helpers.
@testable import QWKV

extension QWKVCompressedLayout {
    static func dequantizeQ8(
        qIn: [Int8],
        scalesIn: [Float16],
        out: inout [Float],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        guard kvHeads > 0, headDim > 0,
              qIn.count == total, scalesIn.count == kvHeads, out.count == total else {
            throw QWKVError.badArgument(
                "q8 dequantization expects \(total) elements and \(kvHeads) scales (got \(qIn.count), \(scalesIn.count), \(out.count))"
            )
        }
        for h in 0..<kvHeads {
            let offset = h * headDim
            let sc = Float(scalesIn[h])
            for d in 0..<headDim {
                out[offset + d] = Float(qIn[offset + d]) * sc
            }
        }
    }

    static func dequantizeQ4(
        qIn: [UInt8],
        scalesIn: [Float16],
        minsIn: [Float16],
        out: inout [Float],
        kvHeads: Int = 4,
        headDim: Int = 256
    ) throws {
        let total = kvHeads * headDim
        let groupsPerHead = max(1, (headDim + 127) / 128)
        let totalGroups = kvHeads * groupsPerHead
        guard kvHeads > 0, headDim > 0, headDim % 2 == 0,
              qIn.count == total / 2, scalesIn.count == totalGroups,
              minsIn.count == totalGroups, out.count == total else {
            throw QWKVError.badArgument(
                "q4 dequantization expects \(total / 2) bytes, \(totalGroups) scales and mins, \(total) elements (got \(qIn.count), \(scalesIn.count), \(minsIn.count), \(out.count))"
            )
        }
        for h in 0..<kvHeads {
            for g in 0..<groupsPerHead {
                let gIdx = h * groupsPerHead + g
                let groupStart = g * 128
                let groupLen = min(128, headDim - groupStart)
                let outOffset = h * headDim + groupStart
                let inByteOffset = outOffset / 2
                let sc = Float(scalesIn[gIdx])
                let mn = Float(minsIn[gIdx])
                for i in 0..<(groupLen / 2) {
                    let byte = qIn[inByteOffset + i]
                    let q0 = Float(byte & 0x0F)
                    let q1 = Float(byte >> 4)
                    out[outOffset + 2 * i] = q0 * sc + mn
                    out[outOffset + 2 * i + 1] = q1 * sc + mn
                }
            }
        }
    }
}
