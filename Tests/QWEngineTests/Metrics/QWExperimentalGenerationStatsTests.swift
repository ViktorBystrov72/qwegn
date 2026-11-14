import XCTest
@testable import QWEngine

final class QWExperimentalGenerationStatsTests: XCTestCase {
    func testThroughputUsesPrefillTokensAndVerifiedDecodeTokens() {
        let stats = QWExperimentalGenerationStats(
            promptTokens: 40,
            generatedTokens: 8,
            prefillSec: 2,
            decodeSec: 3,
            roundMs: [100, 200, 300],
            roundTokenCounts: [3, 3, 1])

        XCTAssertEqual(stats.prefillTokensPerSecond, 20)
        XCTAssertEqual(stats.decodeTokensPerSecond, 7.0 / 3.0, accuracy: 0.0001)
    }

    func testThroughputIsZeroWhenPhaseHasNoDuration() {
        let stats = QWExperimentalGenerationStats(
            promptTokens: 40,
            generatedTokens: 1,
            prefillSec: 0,
            decodeSec: 0,
            roundMs: [],
            roundTokenCounts: [])

        XCTAssertEqual(stats.prefillTokensPerSecond, 0)
        XCTAssertEqual(stats.decodeTokensPerSecond, 0)
    }
}

private extension QWExperimentalGenerationStats {
    var prefillTokensPerSecond: Double {
        guard prefillSec > 0, prefillSec.isFinite else { return 0 }
        return Double(promptTokens) / prefillSec
    }

    var decodeTokensPerSecond: Double {
        guard decodeSec > 0, decodeSec.isFinite else { return 0 }
        return Double(roundTokenCounts.reduce(0, +)) / decodeSec
    }
}
