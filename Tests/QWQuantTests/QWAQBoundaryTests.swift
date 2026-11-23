// QWAQBoundaryTests.swift — CPU boundary and round-trip tests for QWAQBlock, QWAQProducer,
// and 2D/3D MLX affine QWSafeTensorsContainer decoding.
import Foundation
import XCTest
@testable import QWQuant
import QWCore

final class QWAQBoundaryTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwaq-bounds-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func writeSafetensors(_ name: String, tensors: [(String, [Int], String, [UInt8])]) throws -> URL {
        var header: [String: Any] = [:]
        var payload = Data()
        for (tname, shape, dtype, bytes) in tensors {
            header[tname] = ["dtype": dtype, "shape": shape,
                             "data_offsets": [payload.count, payload.count + bytes.count]]
            payload.append(contentsOf: bytes)
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(headerJSON)
        out.append(payload)
        let url = dir.appendingPathComponent(name)
        try out.write(to: url)
        return url
    }

    private static func packMLXWords(codes: [UInt32], bits: Int) -> [UInt8] {
        let totalBits = codes.count * bits
        precondition(totalBits % 32 == 0)
        var out = [UInt8](repeating: 0, count: totalBits / 8)
        for (i, v) in codes.enumerated() {
            var bit = i * bits
            let end = bit + bits
            while bit < end {
                let byte = bit >> 3
                let off = UInt32(bit & 7)
                let take = min(8 - Int(off), end - bit)
                let mask = UInt32((1 << take) - 1)
                let chunk = (v >> UInt32(bit - i * bits)) & mask
                out[byte] = UInt8((UInt32(out[byte]) & ~(mask << off)) | (chunk << off))
                bit += take
            }
        }
        return out
    }

    func testAQBlockAllPairsSizeRoundTripAndDotParity() throws {
        XCTAssertEqual(QWAQBlock.supported.count, 20)
        for pair in QWAQBlock.supported {
            let expectedBytes = pair.group * pair.bits / 8 + 4
            XCTAssertEqual(QWAQBlock.bytesIfSupported(bits: pair.bits, group: pair.group), expectedBytes)
            XCTAssertEqual(QWAQBlock.bytesIfBlockSupported(bits: pair.bits, group: pair.group), expectedBytes)
            XCTAssertEqual(QWAQBlock.rowBytes(bits: pair.bits, group: pair.group, k: 512),
                           (512 / pair.group) * expectedBytes)
            XCTAssertEqual(QWAQBlock.formatName(bits: pair.bits, group: pair.group), "AQ\(pair.bits)_G\(pair.group)")

            let mask = UInt32((1 << pair.bits) - 1)
            let codes = (0..<pair.group).map { UInt32(($0 * 7 + pair.bits) & Int(mask)) }
            let scale: Float = 0.25
            let bias: Float = -1.5
            let block = try QWAQBlock.pack(bits: pair.bits, group: pair.group,
                                           values: codes, scale: scale, bias: bias)
            XCTAssertEqual(block.count, expectedBytes)

            var unpacked = [Float]()
            let x = (0..<pair.group).map { Float(($0 % 9) - 4) * 0.125 }
            let dot = block.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Float in
                QWAQBlock.unpack(group: raw, bits: pair.bits, group: pair.group, into: &unpacked)
                let payload = UnsafeRawBufferPointer(rebasing: raw.dropFirst(4))
                for i in 0..<pair.group {
                    XCTAssertEqual(QWAQBlock.readBits(payload, bits: pair.bits, index: i), codes[i])
                }
                return QWAQBlock.dot(group: raw, bits: pair.bits, group: pair.group, x: x)
            }
            XCTAssertEqual(unpacked.count, pair.group)
            let refDot = zip(unpacked, x).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            XCTAssertEqual(dot, refDot, accuracy: 1e-4)
        }
        for bad in [(1, 64), (5, 32), (4, 16), (4, 48), (9, 32), (0, 32)] {
            XCTAssertNil(QWAQBlock.bytesIfSupported(bits: bad.0, group: bad.1))
            XCTAssertNil(QWAQBlock.bytesIfBlockSupported(bits: bad.0, group: bad.1))
        }
    }

    func testAQProducerHeaderRepresentabilityAndHQ4Block() throws {
        XCTAssertTrue(QWAQProducer.headerHolds(scale: 0.25, bias: -1.5))
        let f16Subnormal = Float(Float16(bitPattern: 0x0001))
        XCTAssertTrue(QWAQProducer.headerHolds(scale: f16Subnormal, bias: -f16Subnormal))
        XCTAssertFalse(QWAQProducer.headerHolds(scale: 0.1, bias: 0.0))
        XCTAssertFalse(QWAQProducer.headerHolds(scale: 70000.0, bias: 0.0))
        XCTAssertFalse(QWAQProducer.headerHolds(scale: Float.nan, bias: 0.0))

        XCTAssertThrowsError(try QWAQBlock.pack(
            bits: 4, group: 32, values: [UInt32](repeating: 0, count: 32), scale: 0.1, bias: 0.0
        )) { err in
            guard case QWAQProducerError.headerNotRepresentable = err else {
                return XCTFail("expected headerNotRepresentable, got \(err)")
            }
        }
        let rounded = QWAQBlock.packRoundedHeader(
            bits: 4, group: 32, values: [UInt32](repeating: 3, count: 32), scale: 0.1, bias: 0.0)
        XCTAssertEqual(rounded.count, 20)
        XCTAssertEqual(QWAQProducer.hq4G32Block([Float](repeating: 0.25, count: 32)).count, 20)
        XCTAssertEqual(QWAQProducer.hq4G32Block(codes: [UInt32](repeating: 8, count: 32), scale: 0.5).count, 20)
    }

    func testSafeTensors2DAnd3DMLXAffineAndBoundsValidation() throws {
        let bits = 4, experts = 3, inter = 2, cols = 64
        let rows3D = experts * inter
        let codes3D = (0..<(rows3D * cols)).map { UInt32(($0 * 5 + 3) & 0xF) }
        let wBytes3D = Self.packMLXWords(codes: codes3D, bits: bits)
        var sBytes3D = [UInt8](), bBytes3D = [UInt8]()
        var scalesF3D = [Float](), biasesF3D = [Float]()
        for r in 0..<rows3D {
            let s = Float(Float16(0.25 * Float(r + 1)))
            let b = Float(Float16(-0.5 * Float(r)))
            scalesF3D.append(s); biasesF3D.append(b)
            withUnsafeBytes(of: Float16(s).bitPattern.littleEndian) { sBytes3D.append(contentsOf: $0) }
            withUnsafeBytes(of: Float16(b).bitPattern.littleEndian) { bBytes3D.append(contentsOf: $0) }
        }
        let url3D = try writeSafetensors("moe3d.safetensors", tensors: [
            ("switch_mlp.down_proj.weight", [experts, inter, cols * bits / 32], "U32", wBytes3D),
            ("switch_mlp.down_proj.scales", [experts, inter, 1], "F16", sBytes3D),
            ("switch_mlp.down_proj.biases", [experts, inter, 1], "F16", bBytes3D),
        ])
        let st3D = try QWSafeTensorsContainer(url: url3D)
        let decoded3D = try st3D.readFloats("switch_mlp.down_proj.weight", expectedElementCount: rows3D * cols)
        XCTAssertEqual(decoded3D.count, rows3D * cols)
        for r in 0..<rows3D {
            for c in 0..<cols {
                let want = Float(codes3D[r * cols + c]) * scalesF3D[r] + biasesF3D[r]
                XCTAssertEqual(decoded3D[r * cols + c], want, accuracy: 1e-6)
            }
        }

        // Mismatched scales/biases shape must fail closed instead of misreading memory.
        let badShapeURL = try writeSafetensors("bad_shape.safetensors", tensors: [
            ("w.weight", [2, 8], "U32", [UInt8](repeating: 0, count: 64)),
            ("w.scales", [2, 1], "F16", [0x00, 0x3C, 0x00, 0x3C]),
            ("w.biases", [1, 2], "F16", [0x00, 0x00, 0x00, 0x00]),
        ])
        let stBad = try QWSafeTensorsContainer(url: badShapeURL)
        XCTAssertThrowsError(try stBad.readFloats("w.weight", expectedElementCount: 128))
    }
}
