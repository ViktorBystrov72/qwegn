// QWDecisionVisionTests.swift — PyTorch oracle parity & residency tests for LFM3-VL and SigLIP2 vision towers.

import Foundation
import XCTest
@testable import QWMM
@testable import QWVision

final class QWDecisionVisionTests: XCTestCase {
    private static var oracleRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("out/vision_decision_oracle", isDirectory: true)
    }

    private func requireOracleDir(_ name: String) throws -> URL {
        let dir = Self.oracleRoot.appendingPathComponent(name, isDirectory: true)
        let manifest = dir.appendingPathComponent("meta.json")
        if !FileManager.default.fileExists(atPath: manifest.path) {
            try XCTSkipIf(
                ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                "vision_decision_oracle fixture missing on GitHub Actions"
            )
            XCTFail("Missing oracle fixture at \(dir.path); run Scripts/oracle/vision_decision_oracle.py")
        }
        return dir
    }

    private static func loadFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.stride
        return data.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(ptr.prefix(count))
        }
    }

    private static func loadMeta(_ url: URL) throws -> (h: Int, w: Int, t: Int, nToken: Int, outDim: Int) {
        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let h = obj["h"] as? Int,
              let w = obj["w"] as? Int,
              let t = obj["t"] as? Int,
              let nToken = obj["n_token"] as? Int,
              let outDim = obj["out_dim"] as? Int else {
            throw NSError(domain: "QWDecisionVisionTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad meta.json"])
        }
        return (h, w, t, nToken, outDim)
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    func testLFM3VLVisionOracleParity() throws {
        let dir = try requireOracleDir("lfm3_vl_vision")
        let meta = try Self.loadMeta(dir.appendingPathComponent("meta.json"))
        let patches = try Self.loadFloats(dir.appendingPathComponent("patches.f32"))
        let expected = try Self.loadFloats(dir.appendingPathComponent("expected.f32"))

        let backend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: true, allowOnDemand: false)
        XCTAssertEqual(backend.visionConfig.architecture, .lfm3VL)
        XCTAssertEqual(backend.visionConfig.posSide, 0)
        XCTAssertEqual(backend.visionConfig.heads, 4)
        XCTAssertEqual(backend.visionConfig.kvHeads, 2)
        XCTAssertEqual(backend.visionConfig.temporal, 2)
        XCTAssertEqual(backend.visionConfig.projectorNormEps, 1e-6, accuracy: 1e-9)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentBytes, 0)

        let spec = QWMMSpec(visionConfig: backend.visionConfig)
        let grid = try QWVisionGrid(t: meta.t, h: meta.h, w: meta.w, spec: spec)
        let actual = try backend.encodeSync(
            patches: patches,
            grid: grid.plan,
            bins: grid.interpBins,
            posIds: grid.ropePositions
        )
        XCTAssertEqual(actual.count, meta.nToken * meta.outDim)
        let diff = Self.maxAbsDiff(actual, expected)
        XCTAssertLessThanOrEqual(diff, 1e-3, "LFM3-VL parity maxAbs=\(diff)")
    }

    func testSigLIP2NaFlexVisionOracleParity() throws {
        let dir = try requireOracleDir("siglip2_vision")
        let meta = try Self.loadMeta(dir.appendingPathComponent("meta.json"))
        let patches = try Self.loadFloats(dir.appendingPathComponent("patches.f32"))
        let expected = try Self.loadFloats(dir.appendingPathComponent("expected.f32"))

        let backend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: true, allowOnDemand: false)
        XCTAssertEqual(backend.visionConfig.architecture, .siglip2)
        XCTAssertGreaterThan(backend.visionConfig.posSide, 0)
        XCTAssertEqual(backend.visionConfig.heads, 4)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentBytes, 0)

        let spec = QWMMSpec(visionConfig: backend.visionConfig)
        let grid = try QWVisionGrid(t: meta.t, h: meta.h, w: meta.w, spec: spec)
        XCTAssertNotEqual(meta.h, meta.w, "Fixture must exercise non-square NaFlex grid")
        let actual = try backend.encodeSync(
            patches: patches,
            grid: grid.plan,
            bins: grid.interpBins,
            posIds: grid.ropePositions
        )
        XCTAssertEqual(actual.count, meta.nToken * meta.outDim)
        let diff = Self.maxAbsDiff(actual, expected)
        XCTAssertLessThanOrEqual(diff, 1e-3, "SigLIP2 NaFlex parity maxAbs=\(diff)")
    }

    func testLoadVisionAndNoLoadVisionResidencyLifecycle() throws {
        let dir = try requireOracleDir("lfm3_vl_vision")
        let meta = try Self.loadMeta(dir.appendingPathComponent("meta.json"))
        let patches = try Self.loadFloats(dir.appendingPathComponent("patches.f32"))
        let expected = try Self.loadFloats(dir.appendingPathComponent("expected.f32"))

        // 1. --no-load-vision keeps residentBytes == 0 and refuses image requests.
        let disabled = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: false, allowOnDemand: false)
        XCTAssertFalse(disabled.isLoaded)
        XCTAssertFalse(disabled.isVisionEnabled)
        XCTAssertEqual(disabled.residentBytes, 0)
        XCTAssertThrowsError(try disabled.prepareForPrompt(hasImages: true))

        let spec = QWMMSpec(visionConfig: disabled.visionConfig)
        let grid = try QWVisionGrid(t: meta.t, h: meta.h, w: meta.w, spec: spec)
        XCTAssertThrowsError(try disabled.encodeSync(
            patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
        ))

        // 2. On-demand loads for image turns and unloads back to 0 B on text turns.
        let onDemand = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: false, allowOnDemand: true)
        XCTAssertFalse(onDemand.isLoaded)
        XCTAssertTrue(onDemand.isVisionEnabled)
        XCTAssertEqual(onDemand.residentBytes, 0)

        try onDemand.prepareForPrompt(hasImages: true)
        XCTAssertTrue(onDemand.isLoaded)
        XCTAssertGreaterThan(onDemand.residentBytes, 0)
        let out = try onDemand.encodeSync(
            patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
        )
        XCTAssertLessThanOrEqual(Self.maxAbsDiff(out, expected), 1e-3)

        try onDemand.prepareForPrompt(hasImages: false)
        XCTAssertFalse(onDemand.isLoaded)
        XCTAssertEqual(onDemand.residentBytes, 0)

        // 3. --load-vision keeps weights resident across text turns.
        try onDemand.setVisionLoaded(true)
        XCTAssertTrue(onDemand.isLoaded)
        try onDemand.prepareForPrompt(hasImages: false)
        XCTAssertTrue(onDemand.isLoaded)
        XCTAssertGreaterThan(onDemand.residentBytes, 0)
    }
}
