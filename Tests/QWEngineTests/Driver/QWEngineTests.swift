// QWEngineTests — C21 leg 1: the orchestrator's planning laws.
//
// Cadence (L1), prefill chunking + head-at-last (L2), decode-step ordering
// Full note: docs/STYLE_NOTES.md#QWEngineTests.swift.
import XCTest
import Metal
import QWCore
import QWAlloc
import QWMM
import QWQuant
import QWGEMM
@testable import QWEngine

final class QWEngineTests: XCTestCase {

    // MARK: - CPU-tail ledger

    func testEncodeLedger() {
        let led = QWEncodeLedger(budgetMicros: 300)
        XCTAssertEqual(led.summary().n, 0)
        for v in [120.0, 80.0, 310.0, 90.0, 100.0] { led.record(micros: v) }
        let s = led.summary()
        XCTAssertEqual(s.n, 5)
        XCTAssertEqual(s.p50, 100.0)
        XCTAssertEqual(s.max, 310.0)
        XCTAssertEqual(s.overBudget, 1)                          // exactly the one over
    }

    func testPrefillTimingLedgerRecordsOptionalGpuTimeAndResets() {
        let ledger = QWPrefillTimingLedger()
        ledger.record(encodeMicros: 999, gpuMicros: 999)
        XCTAssertTrue(ledger.values().isEmpty)
        ledger.setEnabled(true)
        ledger.record(encodeMicros: 120, gpuMicros: 80)
        ledger.record(encodeMicros: 100, gpuMicros: nil)
        XCTAssertEqual(ledger.values().map(\.encodeMicros), [120, 100])
        XCTAssertEqual(ledger.values().map(\.gpuMicros), [80, nil])
        ledger.reset()
        ledger.setEnabled(false)
        XCTAssertTrue(ledger.values().isEmpty)
    }

    func testPrefillScratchEvictionAutoTriggersOnHighWiredArenaRatio() {
        let uma128 = UInt64(128) << 30
        XCTAssertFalse(
            QWPrefillScratchEvictionPolicy.auto.shouldEvict(
                totalMemoryBytes: uma128,
                wiredEngineBytes: 22 << 30))
        XCTAssertTrue(
            QWPrefillScratchEvictionPolicy.auto.shouldEvict(
                totalMemoryBytes: uma128,
                wiredEngineBytes: 100 << 30))
        XCTAssertTrue(
            QWPrefillScratchEvictionPolicy.always.shouldEvict(
                totalMemoryBytes: uma128,
                wiredEngineBytes: 0))
        XCTAssertFalse(
            QWPrefillScratchEvictionPolicy.never.shouldEvict(
                totalMemoryBytes: UInt64(32) << 30,
                wiredEngineBytes: 28 << 30))
    }
}
