// QWQuantSafetyHardeningTests.swift — Regression tests for QWQuant memory/integer/traversal safety.
import Foundation
import XCTest
@testable import QWQuant
import QWCore

final class QWQuantSafetyHardeningTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwquant-sec-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func writeSafetensors(_ name: String, headerPadding: Int = 0,
                                  tensors: [(String, [Int], String, [UInt8])]) throws {
        var header: [String: Any] = [:]
        var payload = Data()
        for (tname, shape, dtype, bytes) in tensors {
            header[tname] = ["dtype": dtype, "shape": shape,
                             "data_offsets": [payload.count, payload.count + bytes.count]]
            payload.append(contentsOf: bytes)
        }
        var headerJSON = try JSONSerialization.data(withJSONObject: header)
        if headerPadding > 0 {
            headerJSON.append(Data(repeating: 0x20, count: headerPadding))
        }
        var out = Data()
        out.append(contentsOf: withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(headerJSON)
        out.append(payload)
        try out.write(to: dir.appendingPathComponent(name))
    }

    private func writeSyntheticQWP(
        name: String,
        payload: Data,
        tensors: [String: Any],
        magic: String = QWQuant.magic,
        headerLenOverride: UInt64? = nil
    ) throws -> URL {
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": QWModelDescriptor.qwen38_27b.family.rawValue,
                "geometry_digest": QWModelDescriptor.qwen38_27b.geometryDigestV3Hex,
            ],
            "tensors": tensors,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj)
        let hdrLen = headerLenOverride ?? UInt64(hdrData.count)
        var out = Data(magic.utf8)
        out.append(contentsOf: withUnsafeBytes(of: hdrLen.littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(payload)
        out.append(hdrData)
        let url = dir.appendingPathComponent(name)
        try out.write(to: url)
        return url
    }

    func testComputeTopLeftSingularVectorsRejectsNonPositiveRank() throws {
        XCTAssertThrowsError(try QWLSASidecar.computeTopLeftSingularVectorsF16(
            matrix: [1, 2, 3, 4], rows: 2, cols: 2, rank: 0
        ))
        XCTAssertThrowsError(try QWLSASidecar.computeTopLeftSingularVectorsF16(
            matrix: [1, 2, 3, 4], rows: 2, cols: 2, rank: -1
        ))
        XCTAssertTrue(QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: 0, indexDim: 8).isEmpty)
        XCTAssertTrue(QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: 8, indexDim: 0).isEmpty)
        let rank1 = try QWLSASidecar.computeTopLeftSingularVectorsF16(
            matrix: [3, 0, 0, 1], rows: 2, cols: 2, rank: 1
        )
        XCTAssertEqual(rank1.count, 2)
    }

    func testPackDequantQ8Group32AndUnalignedRawBf16() throws {
        // Construct a pack with odd header/payload offset (12 + 1 = 13) for alignment-safety
        // and a 2x64 Q8 tensor with group=32 (4 groups total, not hardcoded 128).
        var payload = Data([0xAA]) // 1 byte pad so offset 13 is unaligned
        let bfOff = 12 + payload.count
        let bfVals: [Float] = [1.5, -2.25, 0.5, 4.0]
        for v in bfVals {
            let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16).littleEndian
            payload.append(contentsOf: withUnsafeBytes(of: bits) { (raw: UnsafeRawBufferPointer) in Array(raw) })
        }
        let q8Off = 12 + payload.count
        let nRows = 2, kCols = 64
        for r in 0..<nRows {
            for c in 0..<kCols {
                let q = Int8(((r * kCols + c) % 11) - 5)
                payload.append(UInt8(bitPattern: q))
            }
        }
        let q8sOff = 12 + payload.count
        let scales: [Float16] = [0.5, 1.0, 2.0, 0.25] // 2 rows * 2 groups/row = 4 scales
        for s in scales {
            payload.append(contentsOf: withUnsafeBytes(of: s.bitPattern.littleEndian) {
                (raw: UnsafeRawBufferPointer) in Array(raw)
            })
        }

        let tensors: [String: Any] = [
            "raw_unaligned": ["shape": [4], "raw_off": bfOff],
            "q8_g32": ["shape": [nRows, kCols], "q8_off": q8Off, "q8_s_off": q8sOff],
            "q8_neg": ["shape": [nRows, kCols], "q8_off": -4, "q8_s_off": q8sOff],
            "q8_oob": ["shape": [nRows, kCols], "q8_off": q8Off + 10000, "q8_s_off": q8sOff],
        ]
        let packURL = try writeSyntheticQWP(name: "test.qwp", payload: payload, tensors: tensors)
        let pack = try QWQuant.Pack(url: packURL, descriptor: .qwen38_27b)

        let rawOut = try pack.rawBf16("raw_unaligned")
        XCTAssertEqual(rawOut, bfVals)

        let q8Out = try pack.dequantQ8("q8_g32")
        XCTAssertEqual(q8Out.count, nRows * kCols)
        for r in 0..<nRows {
            for g in 0..<2 {
                let s = Float(scales[r * 2 + g])
                for j in 0..<32 {
                    let q = Float(Int8(((r * kCols + g * 32 + j) % 11) - 5))
                    XCTAssertEqual(q8Out[r * kCols + g * 32 + j], q * s, accuracy: 1e-6)
                }
            }
        }
        XCTAssertThrowsError(try pack.dequantQ8("q8_neg"))
        XCTAssertThrowsError(try pack.dequantQ8("q8_oob"))
        XCTAssertThrowsError(try pack.withRawPointer(at: Int.max - 4, byteCount: 16) { _ in })
    }

    func testPackAndSidecarRejectOverflowingHeaderLengthWithoutTrapping() throws {
        let badQWP = try writeSyntheticQWP(
            name: "overflow.qwp", payload: Data(repeating: 0, count: 32),
            tensors: [:], headerLenOverride: UInt64.max
        )
        XCTAssertThrowsError(try QWQuant.Pack.peekProvenance(url: badQWP))
        XCTAssertThrowsError(try QWQuant.Pack(url: badQWP, descriptor: .qwen38_27b))

        var qlsa = Data("QLSA0001".utf8)
        qlsa.append(contentsOf: withUnsafeBytes(of: UInt64.max.littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        qlsa.append(Data(repeating: 0, count: 32))
        let qlsaURL = dir.appendingPathComponent("bad.qlsa")
        try qlsa.write(to: qlsaURL)
        XCTAssertThrowsError(try QWLSASidecar.peekInfo(url: qlsaURL))
        XCTAssertThrowsError(try QWLSASidecar(url: qlsaURL))
    }

    func testSafeTensorsRejectsIndexPathTraversalAndMismatchedBiases() throws {
        try writeSafetensors("shard.safetensors", tensors: [
            ("w", [4], "I8", [1, 2, 3, 4]),
        ])
        for badShard in ["../shard.safetensors", "/etc/passwd", "a/../../shard.safetensors", ""] {
            let idx: [String: Any] = ["weight_map": ["w": badShard]]
            let idxURL = dir.appendingPathComponent("bad.safetensors.index.json")
            try JSONSerialization.data(withJSONObject: idx).write(to: idxURL)
            XCTAssertThrowsError(try QWSafeTensorsContainer(url: idxURL)) { err in
                guard case QWSafeTensorsError.badHeader = err else {
                    return XCTFail("Expected badHeader for '\(badShard)', got \(err)")
                }
            }
        }

        // Mismatched bReader dtype (U8 vs F16) must throw instead of OOB bindMemory.
        try writeSafetensors("bad_bias.safetensors", tensors: [
            ("t.weight", [1, 4], "U32", Array(repeating: 0, count: 16)),
            ("t.scales", [1, 1], "F16", [0x00, 0x3C]),
            ("t.biases", [1, 1], "U8", [0]),
        ])
        let badC = try QWSafeTensorsContainer(url: dir.appendingPathComponent("bad_bias.safetensors"))
        XCTAssertThrowsError(try badC.readFloats("t.weight"))
    }

    func testSafeTensorsMLX4BitG32InferenceAndUnalignedPayloads() throws {
        // 1 row, k=32, 4-bit g32 -> wCols=4 (16 bytes), gCols=1.
        // Without config or expectedElementCount, old `% (gCols * 64)` decoded this as 2-bit g64!
        var wBytes = [UInt8]()
        for w in 0..<4 {
            var word: UInt32 = 0
            for nib in 0..<8 {
                let val = UInt32((w * 8 + nib) & 0xF)
                word |= (val << (nib * 4))
            }
            wBytes.append(contentsOf: withUnsafeBytes(of: word.littleEndian) {
                (raw: UnsafeRawBufferPointer) in Array(raw)
            })
        }
        let f16One = withUnsafeBytes(of: Float16(1.0).bitPattern.littleEndian) { (raw: UnsafeRawBufferPointer) in Array(raw) }
        let f16Zero = withUnsafeBytes(of: Float16(0.0).bitPattern.littleEndian) { (raw: UnsafeRawBufferPointer) in Array(raw) }
        let f32Two = withUnsafeBytes(of: Float(2.0).bitPattern.littleEndian) { (raw: UnsafeRawBufferPointer) in Array(raw) }

        // Pad JSON header by 3 bytes so payload starts at unaligned offset (not divisible by 2 or 4).
        try writeSafetensors("model.safetensors", headerPadding: 3, tensors: [
            ("unaligned_pad", [1], "U8", [0x7F]),
            ("m.weight", [1, 4], "U32", wBytes),
            ("m.scales", [1, 1], "F16", f16One),
            ("m.biases", [1, 1], "F16", f16Zero),
            ("f32_unaligned", [1], "F32", f32Two),
        ])
        let cfg: [String: Any] = ["quantization": ["bits": 4, "group_size": 32]]
        try JSONSerialization.data(withJSONObject: cfg).write(to: dir.appendingPathComponent("config.json"))

        let c = try QWSafeTensorsContainer(url: dir.appendingPathComponent("model.safetensors"))
        let vals = try c.readFloats("m.weight")
        XCTAssertEqual(vals.count, 32)
        for i in 0..<32 {
            XCTAssertEqual(vals[i], Float(i & 0xF), accuracy: 1e-5)
        }
        XCTAssertEqual(try c.readFloats("f32_unaligned"), [2.0])
    }

    func testNgramFormatRejectsOverflowingHeaderFields() throws {
        var hdr = [UInt8](repeating: 0, count: QWNgramFormat.headerBytes)
        let magic = QWNgramFormat.magic.littleEndian
        withUnsafeBytes(of: magic) { (raw: UnsafeRawBufferPointer) in
            for i in 0..<8 { hdr[i] = raw[i] }
        }
        hdr[8] = 1 // version 1
        hdr[24] = 32 // headDim = 32
        hdr[26] = 64 // rowBytes = 64
        for i in 0..<8 { hdr[16 + i] = 0xFF } // entryCount = UInt64.max
        for i in 0..<8 { hdr[32 + i] = 0xFF } // dataOffset = UInt64.max
        XCTAssertThrowsError(try QWNgramFormat.parse(hdr))
        XCTAssertThrowsError(try QWNgramDType.bf16.rowBytes(headDim: Int.max))
    }
}
