// QWProfileStoreTests.swift — The knob values one model was configured with.
// The store is the only place a knob value is chosen: explicit, then the
// profile, then the default. A knob the model cannot honour is absent —
// never a zero that reads like a setting.

import XCTest
import QWCore
@testable import QWEngine

final class QWProfileStoreTests: XCTestCase {
    private func makeStore() throws -> (QWProfileStore, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwg-profile-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (QWProfileStore(root: root), root)
    }

    func testResolutionOrderIsExplicitThenProfileThenDefault() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = QWKnobProfile(schemaVersion: 1, modelId: "m", geometryDigest: "d",
                                    updatedAt: Date(), values: ["temperature": .double(0.2)])
        try store.save(profile)
        let resolved = try store.resolve(modelId: "m",
                                         explicit: ["temperature": .double(0.9)],
                                         capabilities: [.prefixCache])
        XCTAssertEqual(resolved["temperature"], .double(0.9))
        let profileOnly = try store.resolve(modelId: "m", explicit: [:], capabilities: [.prefixCache])
        XCTAssertEqual(profileOnly["temperature"], .double(0.2))
        // A knob the profile does not hold falls back to the registry's
        // default — not to the profile's value.
        let defaulted = try store.resolve(modelId: "m", explicit: [:], capabilities: [.prefixCache])
        XCTAssertEqual(defaulted["temperature"], .double(0.2))
        XCTAssertEqual(defaulted["topK"], .int(40), "a knob the profile lacks uses the default")
    }

    func testIncapableKnobRequestedExplicitlyThrows() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(try store.resolve(modelId: "m",
                                               explicit: ["gdnPrefill": .string("chunked")],
                                               capabilities: [])) { error in
            XCTAssertEqual(error as? QWKnobError,
                           .capabilityUnavailable(knob: "gdnPrefill", model: "m",
                                                  required: .gdnPrefill))
        }
    }

    /// A profile written against another geometry is stale: the knobs it
    /// holds may not apply to this model.
    func testAProfileFromAnotherGeometryIsStale() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try store.save(QWKnobProfile(schemaVersion: 1, modelId: "m", geometryDigest: "old",
                                     updatedAt: Date(), values: ["temperature": .double(0.2)]))
        XCTAssertThrowsError(try store.resolve(modelId: "m", explicit: [:], capabilities: [],
                                                expectedDigest: "new")) { error in
            XCTAssertEqual(error as? QWKnobError, .staleProfile(model: "m", digest: "old"))
        }
        // The same profile against the geometry it was written for is fine.
        let fresh = try store.resolve(modelId: "m", explicit: [:], capabilities: [],
                                      expectedDigest: "old")
        XCTAssertEqual(fresh["temperature"], .double(0.2))
    }

    /// A knob the model cannot honour is absent from the resolution, not
    /// present with a default that would be applied.
    func testAnUnreachableKnobIsAbsent() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let resolved = try store.resolve(modelId: "m", explicit: [:], capabilities: [])
        XCTAssertNil(resolved["gdnPrefill"], "a knob the model cannot honour must be absent")
        XCTAssertNil(resolved["kvMode"])
        XCTAssertEqual(resolved["temperature"], .double(0.7), "a knob with no capability gate resolves")
    }

    func testTheProfileSurvivesARoundTrip() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let original = QWKnobProfile(schemaVersion: 1, modelId: "toy-27b",
                                     geometryDigest: "d", updatedAt: Date(),
                                     values: ["temperature": .double(0.25),
                                              "maxTokens": .int(256),
                                              "enableThinking": .bool(false),
                                              "reasoningEffort": .string("low")])
        try store.save(original)
        let loaded = try store.load(modelId: "toy-27b")
        XCTAssertEqual(loaded?.values, original.values)
        XCTAssertEqual(loaded?.geometryDigest, original.geometryDigest)
        XCTAssertEqual(loaded?.modelId, original.modelId)
        XCTAssertEqual(loaded?.schemaVersion, original.schemaVersion)
        // The store stamps the moment the file reached the disk.
        XCTAssertEqual(loaded?.updatedAt.timeIntervalSinceNow ?? -100, 0, accuracy: 5)
    }

    /// A model id may hold characters a file name cannot. The profile must
    /// stay inside the store's directory, and two ids must not collide.
    func testTheProfileFileNameIsSafe() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = store.profileURL(for: "../../etc/passwd")
        XCTAssertEqual(url.deletingLastPathComponent().standardizedFileURL.path,
                       root.standardizedFileURL.path,
                       "a model id must not escape the profile directory")
        // Two ids that reduce to the same stem must still get their own
        // file, and two that differ only in case must not share one on a
        // case-insensitive filesystem.
        XCTAssertNotEqual(store.profileURL(for: "a/b").lastPathComponent,
                          store.profileURL(for: "a_b").lastPathComponent)
        XCTAssertNotEqual(store.profileURL(for: "toy-27B").lastPathComponent,
                          store.profileURL(for: "toy-27b").lastPathComponent)
    }

    /// A profile that cannot be read is loud, not silently ignored.
    func testACorruptProfileThrows() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = store.profileURL(for: "m")
        try Data("not json".utf8).write(to: url)
        XCTAssertThrowsError(try store.load(modelId: "m"))
    }

    /// The refusal must name the file, or the user has no way to act on it.
    func testTheRefusalNamesTheFile() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = store.profileURL(for: "m")
        try Data("not json".utf8).write(to: url)
        XCTAssertThrowsError(try store.load(modelId: "m")) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains(url.lastPathComponent),
                          "the refusal must name the file it could not read: \(message)")
        }
    }

    /// A value the registry does not accept is refused loudly, in the
    /// profile and in an explicit request alike.
    func testAnInvalidValueIsRefused() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(try store.resolve(modelId: "m",
                                               explicit: ["temperature": .double(9.0)],
                                               capabilities: [])) { error in
            XCTAssertEqual(error as? QWKnobError,
                           .invalidValue(knob: "temperature", value: "9.0", expected: "0.0...2.0"))
        }
    }

    /// A value the registry refuses must not reach the disk: the profile
    /// would be unreadable the moment it was read back.
    func testAnInvalidValueIsRefusedOnTheWayIn() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = QWKnobProfile(schemaVersion: 1, modelId: "m", geometryDigest: "d",
                                    updatedAt: Date(), values: ["temperature": .double(9.0)])
        XCTAssertThrowsError(try store.save(profile))
        XCTAssertNil(try store.load(modelId: "m"), "nothing must be written")
    }

    /// The file holds one model's profile. A file whose content names
    /// another model is not that model's profile.
    func testAProfileThatNamesAnotherModelIsRefused() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let foreign = QWKnobProfile(schemaVersion: 1, modelId: "other", geometryDigest: "d",
                                    updatedAt: Date(), values: ["temperature": .double(0.2)])
        let data = try JSONEncoder().encode(foreign)
        try data.write(to: store.profileURL(for: "m"))
        XCTAssertThrowsError(try store.load(modelId: "m"))
    }

    /// One value the model cannot honour must not cost the user every
    /// other knob.
    func testOneUnhonourableValueDoesNotKillTheResolution() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try store.save(QWKnobProfile(schemaVersion: 1, modelId: "m", geometryDigest: "d",
                                     updatedAt: Date(),
                                     values: ["kvMode": .string("q4"), "temperature": .double(0.25)]))
        let resolved = try store.resolve(modelId: "m", explicit: [:], capabilities: [.kvQuantQ8])
        XCTAssertEqual(resolved["temperature"], .double(0.25),
                       "one un-honourable value must not cost the other knobs")
        XCTAssertNil(resolved["kvMode"], "a value the model cannot honour is absent")
    }

    /// An explicitly requested value the model cannot honour is refused:
    /// the user asked for it, so the answer is no, not a quiet default.
    func testAnExplicitUnhonourableValueIsRefused() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(try store.resolve(modelId: "m",
                                               explicit: ["kvMode": .string("q4")],
                                               capabilities: [.kvQuantQ8])) { error in
            XCTAssertEqual(error as? QWKnobError,
                           .capabilityUnavailable(knob: "kvMode", model: "m",
                                                  required: .kvQuantQ4))
        }
    }

    /// A profile bound to a geometry must not be stale against itself:
    /// the geometry digest covers the model's shape, not the knobs the
    /// profile holds, so changing a knob does not invalidate it.
    func testChangingAKnobDoesNotMakeTheProfileStale() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try store.save(QWKnobProfile(schemaVersion: 1, modelId: "m", geometryDigest: "geo",
                                     updatedAt: Date(), values: ["kvMode": .string("q8")]))
        var edited = try XCTUnwrap(store.load(modelId: "m"))
        edited.values["kvMode"] = .string("q4")
        try store.save(edited)
        let resolved = try store.resolve(modelId: "m", explicit: [:],
                                         capabilities: [.kvQuantQ8, .kvQuantQ4],
                                         expectedDigest: "geo")
        XCTAssertEqual(resolved["kvMode"], .string("q4"),
                       "a knob the user changed must still apply to the geometry it was written for")
    }

    /// A profile that cannot be read is loud, and there is a way back.
    func testAProfileCanBeForgotten() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("not json".utf8).write(to: store.profileURL(for: "m"))
        XCTAssertThrowsError(try store.load(modelId: "m"))
        XCTAssertTrue(try store.forget(modelId: "m"))
        XCTAssertNil(try store.load(modelId: "m"))
        XCTAssertFalse(try store.forget(modelId: "m"), "there is nothing left to forget")
    }
}
