import XCTest
@testable import QWEngine

/// LighteningMTP L1 — the typed backend contract. Pure CPU: no pack, no
/// HAL, no GPU (the guard runs before any of those exist, so these tests
/// exercise the same door the CLI hits).
final class QWMTPBackendTests: XCTestCase {
    func testResolveAcceptsExactNamesOnly() throws {
        XCTAssertEqual(try QWMTPBackend.resolve(name: "standard"), .standard)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "variational"), .variational)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "relaxed"), .variational)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "plm"), .variational)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "dflash2"), .dflash2)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "dflash"), .dflash2)
        XCTAssertEqual(try QWMTPBackend.resolve(name: "dspine"), .dflash2)
        for bad in ["", "Standard", "lightening", "lightning", "LIGHTNING", " lightning",
                    "standard mtp", "true", "1", "DFlash2"] {
            XCTAssertThrowsError(try QWMTPBackend.resolve(name: bad)) { error in
                XCTAssertTrue(String(describing: error).contains("dflash2"),
                              "refusal must name the supported backends: \(error)")
            }
        }
    }

    func testBackendDraftsContract() throws {
        // No explicit selection: N0 and the implicit Standard default.
        XCTAssertNoThrow(try QWMTPBackend.resolveSelection(nil, drafts: 0))
        XCTAssertEqual(try QWMTPBackend.resolveSelection(nil, drafts: 0), .standard)
        XCTAssertEqual(try QWMTPBackend.resolveSelection(nil, drafts: 1), .standard)
        XCTAssertEqual(try QWMTPBackend.resolveSelection(nil, drafts: 2), .standard)
        XCTAssertEqual(try QWMTPBackend.resolveSelection(nil, drafts: 3), .standard)
        // Explicit backend with positive drafts: honored exactly.
        for backend in QWMTPBackend.allCases {
            for drafts in [1, 2, 3, 5, 6, 7, 11, 15] {
                XCTAssertEqual(try QWMTPBackend.resolveSelection(backend, drafts: drafts), backend)
            }
            // Explicit backend without drafts is a refusal, not a downgrade.
            XCTAssertThrowsError(try QWMTPBackend.resolveSelection(backend, drafts: 0)) { error in
                XCTAssertTrue(String(describing: error).contains(backend.rawValue))
            }
        }
        for drafts in [-1, 16, Int.max] {
            for backend in [nil, QWMTPBackend.standard, QWMTPBackend.variational, QWMTPBackend.dflash2] {
                XCTAssertThrowsError(try QWMTPBackend.resolveSelection(backend, drafts: drafts))
            }
        }
    }

    func testRouteReportingIsHonest() throws {
        XCTAssertEqual(QWMTPBackend.standard.route, "standard")
        XCTAssertEqual(QWMTPBackend.variational.route, "variational:relaxed-argmax-quadratic-v1")
        XCTAssertEqual(QWMTPBackend.dflash2.route, "dflash2:block-diffusion-5L-w2048-conv2-top16")
        XCTAssertNotEqual(QWMTPBackend.standard.qualificationProfile,
                          QWMTPBackend.dflash2.qualificationProfile)
        XCTAssertNotEqual(QWMTPBackend.standard.qualificationProfile,
                          QWMTPBackend.variational.qualificationProfile)
    }

    func testPrefixPolicyKeyCarriesBackendIdentityForSpeculation() {
        XCTAssertEqual(QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 0, mode: .chunked),
                       GDNPrefillMode.chunked.rawValue)
        XCTAssertEqual(QWEngineDriver.prefixPolicyKey(backend: .dflash2, drafts: 0, mode: .chunked),
                       GDNPrefillMode.chunked.rawValue)
        XCTAssertEqual(QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 1, mode: .chunked),
                       "standard+chunked")
        let dflash = QWEngineDriver.prefixPolicyKey(backend: .dflash2, drafts: 2, mode: .chunked)
        XCTAssertEqual(dflash, "dflash2+chunked")
        // Cross-backend and cross-mode keys are disjoint.
        XCTAssertNotEqual(dflash,
                          QWEngineDriver.prefixPolicyKey(backend: .standard, drafts: 2, mode: .chunked))
        XCTAssertNotEqual(dflash,
                          QWEngineDriver.prefixPolicyKey(backend: .dflash2, drafts: 2, mode: .regres))
        // A key with a '+' can never be produced by a plain mode rawValue.
        for mode in [GDNPrefillMode.chunked, .naive, .regres] {
            XCTAssertFalse(mode.rawValue.contains("+"))
        }
    }

    func testBackendCodableRoundTrip() throws {
        let data = try JSONEncoder().encode(QWMTPBackend.dflash2)
        XCTAssertEqual(String(decoding: data, as: UTF8.self), "\"dflash2\"")
        XCTAssertEqual(try JSONDecoder().decode(QWMTPBackend.self, from: data), .dflash2)
    }
}
