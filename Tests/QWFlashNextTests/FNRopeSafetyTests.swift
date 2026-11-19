// FNRopeSafetyTests.swift — Regression tests for QWRopeCPU table pointer lifetime, YaRN, and bounds.
import XCTest
import QWCore
@testable import QWRope

final class FNRopeSafetyTests: XCTestCase {
    func testTablesParallelCountParityWithPositionsAndEmptyCount() {
        let rope = QWRopeCPU(rotaryDim: 16, theta: 10_000.0)
        let (emptyC, emptyS) = rope.tables(count: 0, stride: 1)
        XCTAssertTrue(emptyC.isEmpty)
        XCTAssertTrue(emptyS.isEmpty)

        let count = 37
        let stride = 3
        let (parC, parS) = rope.tables(count: count, stride: stride)
        let positions = (0..<count).map { $0 * stride }
        let (refC, refS) = rope.tables(positions: positions)
        XCTAssertEqual(parC, refC)
        XCTAssertEqual(parS, refS)
    }

    func testCheckedBoundsAndQWRopeTablesCPUExtension() throws {
        let rope = QWRopeCPU(rotaryDim: 4, theta: 10_000.0, yarnFactor: 2.0, originalMaxPosition: 4096)
        let (c, s) = rope.tables(positions: [0, 3])
        var vec: [Float] = [1, 2, 3, 4, 5, 6]
        try rope.applyChecked(&vec, base: 0, dim: 6, cos: c, sin: s, row: 1)
        XCTAssertEqual(vec[4], 5)
        XCTAssertEqual(vec[5], 6)

        XCTAssertThrowsError(try rope.applyChecked(&vec, base: 4, dim: 4, cos: c, sin: s, row: 0))
        XCTAssertThrowsError(try rope.applyChecked(&vec, base: 0, dim: 6, cos: c, sin: s, row: 2))

        let tables = QWRopeTables(rows: 4, yarn: QWYaRNConfig(factor: 2.0, originalMaxPosition: 4096),
                                  geometry: .qwen38_27b)
        XCTAssertEqual(tables.cpu.attentionScaling, tables.attentionScaling, accuracy: 1e-6)
        var headVec = [Float](repeating: 1.0, count: tables.width)
        XCTAssertThrowsError(try tables.applyCPU(&headVec, base: 0, dim: tables.width, row: 4))
        try tables.applyCPU(&headVec, base: 0, dim: tables.width, row: 1)
        XCTAssertTrue(headVec.allSatisfy(\.isFinite))
    }
}
