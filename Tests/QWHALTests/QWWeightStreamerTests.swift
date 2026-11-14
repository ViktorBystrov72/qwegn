import XCTest
import Foundation
@testable import QWHAL
import QWCore

final class QWWeightStreamerTests: XCTestCase {

    func testWeightStreamerDirectDMARead() throws {
        // Create a temporary weight file containing known deterministic pattern
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_weights_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let totalBytes = 64 * 1024 // 64 KiB
        var sourceData = [UInt8](repeating: 0, count: totalBytes)
        for i in 0..<totalBytes {
            sourceData[i] = UInt8((i * 31 + 17) & 0xFF)
        }
        try Data(sourceData).write(to: tempFile)

        // Initialize QWWeightStreamer with unbuffered DMA enabled
        let streamer = try QWWeightStreamer(url: tempFile, unbuffered: true)
        XCTAssertEqual(streamer.fileLength, totalBytes)

        // Read a tensor slice of 16 KiB at offset 16 KiB
        let sliceOffset = 16 * 1024
        let sliceSize = 16 * 1024
        let destBuffer = UnsafeMutableRawPointer.allocate(byteCount: sliceSize, alignment: 16384)
        defer { destBuffer.deallocate() }

        try streamer.streamTensor(offset: sliceOffset, byteCount: sliceSize, destination: destBuffer)

        // Verify byte-exact match against sourceData slice
        let destBytes = destBuffer.assumingMemoryBound(to: UInt8.self)
        for i in 0..<sliceSize {
            XCTAssertEqual(destBytes[i], sourceData[sliceOffset + i])
        }
    }
}
