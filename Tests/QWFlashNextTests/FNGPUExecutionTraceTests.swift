import Foundation
import XCTest
@testable import QWEngine

final class FNGPUExecutionTraceTests: XCTestCase {
    func testOverlappingHostAndGPUIntervalsAreNotAddedTogether() throws {
        let buffers = [
            FNGPUExecutionTrace.BufferTiming(layers: [0, 1], encodeStartMS: 1, submitMS: 3,
                                             gpuStartMS: 4, gpuEndMS: 10),
            FNGPUExecutionTrace.BufferTiming(layers: [1, 8], encodeStartMS: 3, submitMS: 11,
                                             gpuStartMS: 12, gpuEndMS: 15),
            FNGPUExecutionTrace.BufferTiming(layers: [8, 48], encodeStartMS: 11, submitMS: 18,
                                             gpuStartMS: 19, gpuEndMS: 24),
        ]
        let report = FNGPUExecutionTrace.Report(
            phase: "decode", position: 63, rows: 1, pleRoute: "host", mappedPLEBytes: 0,
            includesMTP: false, plePrepareMS: 1, pleWaitMS: 2, nextPLEPrepareMS: 0, completionWaitMS: 7,
            stackToGPUCompleteMS: 25, buffers: buffers)
        XCTAssertEqual(report.encodeMS, 15)
        XCTAssertEqual(report.gpuMS, 14)
        XCTAssertEqual(report.gpuSpanMS, 20)
        XCTAssertEqual(report.gpuIdleBetweenBuffersMS, 6)
        XCTAssertEqual(buffers[1].submitToStartMS, 1)
        let decoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any]
        XCTAssertEqual(decoded?["schema"] as? Int, 1)
        XCTAssertEqual(decoded?["pleRoute"] as? String, "host")
        XCTAssertEqual((decoded?["buffers"] as? [[String: Any]])?.count, 3)
    }

    func testOneBufferHasNoInterBufferIdleTime() {
        let report = FNGPUExecutionTrace.Report(
            phase: "verify", position: 100, rows: 4, pleRoute: "gpu", mappedPLEBytes: 4096,
            includesMTP: true, plePrepareMS: 0.1, pleWaitMS: 0, nextPLEPrepareMS: 0, completionWaitMS: 8,
            stackToGPUCompleteMS: 12, buffers: [
                .init(layers: [0, 48], encodeStartMS: 0.1, submitMS: 3, gpuStartMS: 4, gpuEndMS: 11),
            ])
        XCTAssertEqual(report.gpuIdleBetweenBuffersMS, 0)
        XCTAssertEqual(report.gpuSpanMS, report.gpuMS)
        XCTAssertEqual(report.encodeMS, 2.9)
    }
}

private extension FNGPUExecutionTrace.BufferTiming {
    var encodeMS: Double { submitMS - encodeStartMS }
    var submitToStartMS: Double { gpuStartMS - submitMS }
    var gpuMS: Double { gpuEndMS - gpuStartMS }
}

private extension FNGPUExecutionTrace.Report {
    var encodeMS: Double { buffers.reduce(0) { $0 + $1.encodeMS } - pleWaitMS }
    var gpuMS: Double { buffers.reduce(0) { $0 + $1.gpuMS } }
    var gpuSpanMS: Double { (buffers.last?.gpuEndMS ?? 0) - (buffers.first?.gpuStartMS ?? 0) }
    var gpuIdleBetweenBuffersMS: Double {
        zip(buffers, buffers.dropFirst()).reduce(0) { $0 + max(0, $1.1.gpuStartMS - $1.0.gpuEndMS) }
    }
}
