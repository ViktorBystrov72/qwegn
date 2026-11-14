import XCTest
import Metal
@testable import QWHAL

/// Post-freeze pipeline cache hits must short-circuit without taking the lock.
final class QWHalDoorArenaTests: XCTestCase {
    private func makeHAL() throws -> QWHal {
        try QWHal(metallibPath: QWHALTests.anyMetallib(), geometry: .qwen38_27b)
    }

    func testFrozenPipelineHitsTakeNoLock() throws {
        let hal = try makeHAL()
        _ = try hal.pipeline("qw_probe")
        hal.freezePipelines()
        let before = hal.psoState.withLock { $0.lockCount }
        for _ in 0..<100 { _ = try hal.pipeline("qw_probe") }
        let after = hal.psoState.withLock { $0.lockCount }
        XCTAssertEqual(after, before,
                       "post-freeze hits must short-circuit before the lock")
    }
}
