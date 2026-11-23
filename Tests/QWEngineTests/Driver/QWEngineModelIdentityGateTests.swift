import XCTest
@testable import QWEngine
import QWQuant

/// §3 identity gate (plan gate): unknown pack model IDs reach an explicit
/// unsupported-feature error before HAL/pack open, while `.qwen38FlashNext`
/// and legacy packs pass the unified driver identity gate.
final class QWEngineModelIdentityGateTests: XCTestCase {

    private static var selfpack: URL {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let cand = dir.appendingPathComponent("Tests/QWQuantTests/fixtures/selfpack.qwp")
            if FileManager.default.fileExists(atPath: cand.path) {
                return cand
            }
            dir = dir.deletingLastPathComponent()
        }
        fatalError("selfpack.qwp fixture missing — run: python3 Scripts/quantize/pack.py --selfpack Tests/QWQuantTests/fixtures/selfpack.qwp")
    }

    private static func packWithModelId(_ modelId: String, filename: String) throws -> URL {
        let data = try Data(contentsOf: selfpack)
        guard let prov = data.range(of: Data("\"provenance\": {".utf8))
            ?? data.range(of: Data("\"provenance\":{".utf8)) else {
            throw NSError(domain: "test", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "provenance object not in header"])
        }
        let insertAt = prov.upperBound
        let extra = " \"model_id\": \"\(modelId)\","
        var out = Data()
        out.append(data.subdata(in: 0..<insertAt))
        out.append(Data(extra.utf8))
        out.append(data.subdata(in: insertAt..<data.count))
        var len = out.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in raw.loadUnaligned(fromByteOffset: 4, as: UInt64.self) }
        len += UInt64(extra.utf8.count)
        withUnsafeBytes(of: len) { (raw: UnsafeRawBufferPointer) in out.replaceSubrange(4..<12, with: raw) }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try out.write(to: tmp)
        return tmp
    }

    func testUnknownModelIdRejectedBeforeEngineInit() throws {
        let pack = try Self.packWithModelId("not-a-model-family", filename: "unknown-gate.qwp")
        defer { try? FileManager.default.removeItem(at: pack) }
        do {
            _ = try QWEngineDriver(packPath: pack.path, prewarmDoors: DenseModule.requiredDoors)
            XCTFail("Unknown model_id must not reach the driver")
        } catch let error as QWEngineDriverError {
            guard case QWEngineDriverError.unsupportedModel(let m) = error else {
                return XCTFail("expected unsupportedModel, got \(error)")
            }
            XCTAssertEqual(m, "not-a-model-family")
        }
        // The error message is operator-actionable and names the mechanism.
        let msg = QWEngineDriverError.unsupportedModel("not-a-model-family").description
        XCTAssertTrue(msg.contains("no engine adapter"), msg)
    }

    func testFNPackAdmittedToHyperStreamStartup() throws {
        let pack = try Self.packWithModelId("qwen3.8-flash-next", filename: "fn-gate.qwp")
        defer { try? FileManager.default.removeItem(at: pack) }
        let cfg = try QWEngineDriver.resolveStartup(
            packPath: pack.path, kvMode: .f16, mtpDrafts: 0, mtpBackend: nil,
            prefillScratchEviction: .auto, mtpAdaptiveDepth: nil, mtpAdaptiveEntropyThreshold: nil
        )
        XCTAssertTrue(cfg.isHyperStream)
        do {
            _ = try QWEngineDriver(packPath: pack.path, prewarmDoors: DenseModule.requiredDoors)
        } catch let error as QWEngineDriverError {
            if case QWEngineDriverError.unsupportedModel = error {
                XCTFail("unified driver must admit .qwen38FlashNext: \(error)")
            }
        } catch {
            // Subsequent pack/assets errors are expected on synthetic selfpack without FN config.
        }
    }

    func testLegacyControlPackStillAdmittedToInit() throws {
        // The selfpack fixture has no model_id → the identity gate must be
        // silent for it (it proceeds into HAL/pack open; failure there is a
        // different, pre-existing behavior — we only assert the identity
        // gate did NOT reject it with unsupportedModel).
        do {
            _ = try QWEngineDriver(packPath: Self.selfpack.path, prewarmDoors: DenseModule.requiredDoors)
        } catch let error as QWEngineDriverError {
            if case QWEngineDriverError.unsupportedModel = error {
                XCTFail("legacy control pack rejected by identity gate: \(error)")
            }
        } catch {
            // non-identity errors are outside this gate's scope
        }
    }
}
