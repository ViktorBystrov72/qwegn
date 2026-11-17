import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
@testable import QWEngine

final class QWMTPEmbeddingTableTests: XCTestCase {
    func testMappedGatherMatchesCPUAndRejectsInvalidPlane() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("qw-embed-\(UUID().uuidString)").path
        defer {
            do { try FileManager.default.removeItem(atPath: path) }
            catch { XCTFail("mapped embedding cleanup failed: \(error)") }
        }
        let words: [UInt16] = [0, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x3eaa, 0xc100,
                               0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110]
        var data = Data(repeating: 0, count: 256)
        words.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in data.append(contentsOf: raw) }
        try data.write(to: URL(fileURLWithPath: path))
        let table = try QWMTPEmbeddingTable(hal: hal, path: path, offset: 256, vocabulary: 2, hidden: 8)
        XCTAssertThrowsError(try QWMTPEmbeddingTable(hal: hal, path: path, offset: 257, vocabulary: 2, hidden: 8))
        XCTAssertThrowsError(try QWMTPEmbeddingTable(hal: hal, path: path, offset: 256, vocabulary: 3, hidden: 8))
        let buffer = try XCTUnwrap(hal.device.makeBuffer(length: 256, options: .storageModeShared))
        let region = QWMRegion(buffer: buffer, base: 0, length: 256)
        let token = QWTensorValue(region: region, offset: 0, shape: QWTensorShape([1], .i32))
        let output = QWTensorValue(region: region, offset: 64, shape: QWTensorShape([8], .f16))
        for id in [-1, 0, 1, 2] {
            buffer.contents().assumingMemoryBound(to: Int32.self).pointee = Int32(id)
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            try table.encode(enc, token: token, destination: output)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)
            guard cb.status == .completed else { continue }
            let bits = output.region.pointer.advanced(by: output.offset).assumingMemoryBound(to: UInt16.self)
            for i in 0..<8 {
                let expected = (0..<2).contains(id)
                    ? QWQuant.f32tof16Bits(Float(bitPattern: UInt32(words[id * 8 + i]) << 16))
                    : 0
                XCTAssertEqual(bits[i], expected, "row \(id) element \(i)")
            }
        }
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: path)), data, "gather must not modify pack")
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { enc.endEncoding() }
        for bad in [
            QWTensorValue(region: region, offset: -4, shape: QWTensorShape([1], .i32)),
            QWTensorValue(region: region, offset: 254, shape: QWTensorShape([1], .i32)),
            QWTensorValue(region: region, offset: 1, shape: QWTensorShape([1], .i32)),
            QWTensorValue(region: QWMRegion(buffer: buffer, base: 256, length: 4),
                          offset: 0, shape: QWTensorShape([1], .i32))
        ] {
            XCTAssertThrowsError(try table.encode(enc, token: bad, destination: output))
        }
        let aliased = QWTensorValue(region: region, offset: 0, shape: QWTensorShape([8], .f16))
        XCTAssertThrowsError(try table.encode(enc, token: token, destination: aliased))
        let mappedOutput = QWTensorValue(region: QWMRegion(buffer: table.buffer, base: 0, length: table.buffer.length),
                                        offset: table.byteOffset, shape: QWTensorShape([8], .f16))
        XCTAssertThrowsError(try table.encode(enc, token: token, destination: mappedOutput))
        XCTAssertThrowsError(try QWMTPEmbeddingTable(hal: hal, path: path, offset: 0,
                                                    vocabulary: Int(Int32.max) + 1, hidden: 1))
        XCTAssertThrowsError(try QWMTPEmbeddingTable(hal: hal, path: path, offset: 0,
                                                    vocabulary: 1, hidden: Int(UInt32.max) + 1))
        let multiTokens = buffer.contents().assumingMemoryBound(to: Int32.self)
        multiTokens[0] = 1
        multiTokens[1] = 0
        multiTokens[2] = -1
        let multiOut = QWTensorValue(region: region, offset: 64, shape: QWTensorShape([3 * 8], .f16))
        let cb2 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc2 = try XCTUnwrap(cb2.makeComputeCommandEncoder())
        try table.encodeRows(enc2, tokensBuffer: buffer, tokensByteOffset: 0, rows: 3, destination: multiOut)
        enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
        XCTAssertEqual(cb2.status, .completed)
        let mbits = multiOut.region.pointer.advanced(by: multiOut.offset).assumingMemoryBound(to: UInt16.self)
        for (rowIdx, tokId) in [1, 0, -1].enumerated() {
            for i in 0..<8 {
                let expected = (0..<2).contains(tokId)
                    ? QWQuant.f32tof16Bits(Float(bitPattern: UInt32(words[tokId * 8 + i]) << 16))
                    : 0
                XCTAssertEqual(mbits[rowIdx * 8 + i], expected, "multi row \(rowIdx) elem \(i)")
            }
        }
    }
}
