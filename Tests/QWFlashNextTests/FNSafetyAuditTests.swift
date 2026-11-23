// FNSafetyAuditTests.swift — Audit tests for FlashNext N-gram throwing readers,
// stream-ngram unmapped reads, FD leak prevention, and warm store thread safety.

import Darwin
import Foundation
import XCTest
@testable import QWEngine
@testable import QWHyperPLE
@testable import QWQuant

final class FNSafetyAuditTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fn-safety-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func writeMLXSafetensors(
        to url: URL,
        rows: Int = 4,
        wCols: Int = 4,
        gCols: Int = 2,
        wSpanOverride: Int? = nil
    ) throws {
        let wBytes = wSpanOverride ?? (rows * wCols * 4)
        let sBytes = rows * gCols * 2
        let bBytes = rows * gCols * 2
        let headerObj: [String: Any] = [
            "weight": ["dtype": "U32", "shape": [rows, wCols], "data_offsets": [0, wBytes]],
            "scales": ["dtype": "BF16", "shape": [rows, gCols], "data_offsets": [wBytes, wBytes + sBytes]],
            "biases": ["dtype": "BF16", "shape": [rows, gCols], "data_offsets": [wBytes + sBytes, wBytes + sBytes + bBytes]],
        ]
        let headerJSON = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        var lenLE = UInt64(headerJSON.count).littleEndian
        var data = Data()
        withUnsafeBytes(of: &lenLE) { (raw: UnsafeRawBufferPointer) in data.append(contentsOf: raw) }
        data.append(headerJSON)
        for r in 0..<rows {
            for col in 0..<wCols {
                var w: UInt32 = 0
                for nib in 0..<8 {
                    w |= UInt32((r + col * 8 + nib) % 15) << (nib * 4)
                }
                var le = w.littleEndian
                withUnsafeBytes(of: &le) { (raw: UnsafeRawBufferPointer) in data.append(contentsOf: raw) }
            }
        }
        let oneBF16 = UInt16(0x3F80).littleEndian
        for _ in 0..<(2 * rows * gCols) {
            var le = oneBF16
            withUnsafeBytes(of: &le) { (raw: UnsafeRawBufferPointer) in data.append(contentsOf: raw) }
        }
        try data.write(to: url)
    }

    func testMLXRowsStreamModeMappedFalseAndThrowingBounds() throws {
        let url = tempDir.appendingPathComponent("ngram_table.bin")
        try writeMLXSafetensors(to: url)

        let unmapped = try FNNgramMLXRows(url: url, mapped: false)
        let mapped = try FNNgramMLXRows(url: url, mapped: true)
        let loaderUnmapped = try FNNgramRowsLoader.open(assetsURL: tempDir, mapped: false)

        for r in 0..<4 {
            var a = [Float](repeating: 0, count: unmapped.headDim)
            var b = [Float](repeating: 0, count: mapped.headDim)
            try a.withUnsafeMutableBufferPointer { try unmapped.copyRow(r, to: $0.baseAddress!) }
            try b.withUnsafeMutableBufferPointer { try mapped.copyRow(r, to: $0.baseAddress!) }
            XCTAssertEqual(a, b)
            XCTAssertEqual(a, loaderUnmapped.row(r))
        }

        var scratch = [Float](repeating: 0, count: unmapped.headDim)
        XCTAssertThrowsError(try scratch.withUnsafeMutableBufferPointer {
            try unmapped.copyRow(-1, to: $0.baseAddress!)
        })
        XCTAssertThrowsError(try scratch.withUnsafeMutableBufferPointer {
            try unmapped.copyRow(4, to: $0.baseAddress!)
        })
    }

    func testMLXRowsValidationRejectsBadGroupSizesAndSpans() throws {
        let badGroup = tempDir.appendingPathComponent("bad_group.bin")
        try writeMLXSafetensors(to: badGroup, rows: 2, wCols: 1, gCols: 2)
        XCTAssertThrowsError(try FNNgramMLXRows(url: badGroup, mapped: false))

        let badNonMult8 = tempDir.appendingPathComponent("bad_nonmult8.bin")
        try writeMLXSafetensors(to: badNonMult8, rows: 2, wCols: 3, gCols: 2)
        XCTAssertThrowsError(try FNNgramMLXRows(url: badNonMult8, mapped: false))

        let badSpan = tempDir.appendingPathComponent("bad_span.bin")
        try writeMLXSafetensors(to: badSpan, rows: 4, wCols: 4, gCols: 2, wSpanOverride: 16)
        XCTAssertThrowsError(try FNNgramMLXRows(url: badSpan, mapped: false))
    }

    func testShardRowsInitDoesNotLeakOpenedFDsOnThrow() throws {
        let shard0 = tempDir.appendingPathComponent("shard0.bin")
        try Data(repeating: 0, count: 64).write(to: shard0)
        let locURL = tempDir.appendingPathComponent("ple_ngram_locations.json")
        let sidecar: [String: Any] = [
            "total_rows": 4,
            "shards": [
                ["file": "shard0.bin", "first_row": 0, "shape": [2, 8], "row_bytes": 16, "base_off": 0],
                ["file": "missing_shard1.bin", "first_row": 2, "shape": [2, 8], "row_bytes": 16, "base_off": 0],
            ],
        ]
        try JSONSerialization.data(withJSONObject: sidecar).write(to: locURL)

        let baselineOpenFDs = try FileManager.default.contentsOfDirectory(atPath: "/dev/fd").count
        for _ in 0..<24 {
            XCTAssertThrowsError(try FNNgramShardRows(locations: locURL, snapshot: tempDir))
        }
        let afterOpenFDs = try FileManager.default.contentsOfDirectory(atPath: "/dev/fd").count
        XCTAssertLessThanOrEqual(afterOpenFDs, baselineOpenFDs + 2)
    }

    func testWarmStoreConcurrentOperationsRemainCoherent() {
        let warm = FNPrefixWarmStore(byteLimit: 8192, maxSlots: 4)
        let group = DispatchGroup()
        let q = DispatchQueue(label: "test.fn.warm.concurrent", attributes: .concurrent)
        for w in 0..<6 {
            group.enter()
            q.async {
                for step in 0..<60 {
                    let history = Array(Int64(w * 1000)...Int64(w * 1000 + 130 + step))
                    let rootHash = FNPrefixWarmStore.rootHash(history: history, kvMode: .q8, hasMTP: false)
                    let name = String(format: "fn_entry_%016llx_%d.qwcache", rootHash, history.count)
                    _ = warm.canHold(bytes: 512)
                    let slot = warm.resolveSlot(rootHash: rootHash, history: history, kvMode: .q8, hasMTP: false)
                    warm.store(name: name, blob: Data(repeating: UInt8(w), count: 512),
                               slot: slot, rootHash: rootHash, history: history)
                    let query = history + [99]
                    _ = warm.lookupWarmHit(exactName: name, history: query, rawHistory: query,
                                           kvMode: .q8, hasMTP: false, allowExact: true)
                    _ = warm.blob(name: name)
                }
                group.leave()
            }
        }
        group.wait()
        XCTAssertLessThanOrEqual(warm.state.bytes, warm.byteLimit)
    }
}
