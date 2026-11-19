// QWGDNReplayValidationTests.swift — Validates rollback/stability journal checks.
import Foundation
import Metal
import XCTest
@testable import QWEngine
import QWAlloc
import QWCore
import QWGDN
import QWHAL

final class QWGDNReplayValidationTests: XCTestCase {
    func testRollbackRequiresFullActiveRowCaptureBeforeRewind() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
            "GPU checkpoint validation requires local Apple Silicon Metal device"
        )
        let geometry = QWTestGeometry.g27
        let hal = try QWHal(geometry: geometry)
        let layers = 2
        let rows = 4
        let cpBytes = QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: layers, rows: rows)
        let arena = try QWArena(hal: hal, weights: 256, activations: 1 << 20, kv: 256, gdnStates: cpBytes * 2 + (1 << 20))
        let g = arena.gdnRegion()
        let checkpoint = try QWGDNReplayCheckpoint(geometry: geometry, allocator: g, layers: layers, rows: rows)
        let rec = try g.allocate(QWTensorShape([layers * QWSlotGeometry.recElems(geometry: geometry)], .f32))
        let ring = try g.allocate(QWTensorShape([layers * QWSlotGeometry.ringElems(geometry: geometry)], .f16))
        let decoder = try QWGDNDecode(hal: hal)

        let cb0 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb0.makeBlitCommandEncoder())
        try checkpoint.encodeBackup(blit, rec: rec, ring: ring)
        blit.endEncoding()
        cb0.commit()
        cb0.waitUntilCompleted()

        // Capture only retained rows 0..<2 while activeRows is 4: must refuse!
        checkpoint.captured = [UInt32(0b0011), UInt32(0b0011)]
        XCTAssertThrowsError(try checkpoint.rollbackTapeIsStable(retaining: 2, activeRows: rows))

        let cb1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb1.makeComputeCommandEncoder())
        XCTAssertThrowsError(
            try checkpoint.encodeRollbackTape(enc, decoder: decoder, rec: rec, activeRows: rows, retaining: 2)
        )
        enc.endEncoding()

        // Mark all activeRows (0..<4) captured with finite values: stability succeeds.
        checkpoint.captured = [UInt32(0b1111), UInt32(0b1111)]
        XCTAssertTrue(try checkpoint.rollbackTapeIsStable(retaining: 2, activeRows: rows))

        // Corrupt decayed log-gate on row 3 (in count..<verifiedRows) -> returns false cleanly.
        let glogPtr = checkpoint.glog.region.pointer.advanced(by: checkpoint.glog.offset)
            .assumingMemoryBound(to: Float.self)
        glogPtr[(0 * rows + 3) * geometry.gdnGlogChannels] = -10.0
        XCTAssertFalse(try checkpoint.rollbackTapeIsStable(retaining: 2, activeRows: rows))
    }

    func testZeroLayerCheckpointStillRejectsOutOfRangeActiveRows() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
            "GPU checkpoint validation requires local Apple Silicon Metal device"
        )
        let geometry = QWTestGeometry.g27
        let hal = try QWHal(geometry: geometry)
        let rows = 4
        let cpBytes = QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: 0, rows: rows)
        let arena = try QWArena(hal: hal, weights: 256, activations: 256, kv: 256, gdnStates: cpBytes + 4096)
        let g = arena.gdnRegion()
        let checkpoint = try QWGDNReplayCheckpoint(geometry: geometry, allocator: g, layers: 0, rows: rows)
        let rec = try g.allocate(QWTensorShape([1], .f32))
        let ring = try g.allocate(QWTensorShape([1], .f16))
        let decoder = try QWGDNDecode(hal: hal)

        let cb0 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb0.makeBlitCommandEncoder())
        try checkpoint.encodeBackup(blit, rec: rec, ring: ring)
        blit.endEncoding()
        cb0.commit()
        cb0.waitUntilCompleted()

        XCTAssertTrue(try checkpoint.rollbackTapeIsStable(retaining: 2, activeRows: 4))
        let cb1 = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb1.makeComputeCommandEncoder())
        XCTAssertThrowsError(
            try checkpoint.encodeRollbackTape(enc, decoder: decoder, rec: rec, activeRows: 5, retaining: 2)
        )
        enc.endEncoding()
    }
}
