// QWDecisionCliTests.swift — ArgumentParser & offline calibration unit tests for decide, calibrate, and mcp.

import ArgumentParser
import Foundation
import Testing
import QWDecision
@testable import qwg

@Test("qwg decide parses decision flags with mean score readout default")
func decideParsesFlags() throws {
    let cmd = try Decide.parse([
        "--model", "/tmp/d1-2b",
        "--state", "Server returned 503 under load.",
        "--question", "Should we alert on-call?",
        "--kind", "choice",
        "--options", "ignore",
        "--options", "page",
        "--compute-unit", "metal",
    ])
    #expect(cmd.model == "/tmp/d1-2b")
    #expect(cmd.kind == "choice")
    #expect(cmd.options == ["ignore", "page"])
    #expect(cmd.scoreReadout == "mean")
    #expect(cmd.computeUnit == "metal")
}

@Test("qwg mcp parses model, compute-unit, and server proxy flags")
func mcpParsesFlags() throws {
    let cmd = try Mcp.parse([
        "--model", "d1-omni",
        "--compute-unit", "ane",
        "--server", "http://127.0.0.1:8080",
    ])
    #expect(cmd.model == "d1-omni")
    #expect(cmd.computeUnit == "ane")
    #expect(cmd.serverURL == "http://127.0.0.1:8080")
}

@Test("qwg calibrate fits temperature + PAV isotonic tables and writes valid calibration.json")
func calibrateFitsJsonlAndRoundTrips() throws {
    let tmpDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwg-cal-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpDir) }

    let inputURL = tmpDir.appendingPathComponent("samples.jsonl")
    let outputURL = tmpDir.appendingPathComponent("calibration.json")

    // Overconfident logits (scale ~4x) where optimal T > 1.0 strictly lowers NLL & ECE.
    var lines: [String] = []
    for _ in 0..<24 {
        lines.append(#"{"logits":[-3.0,3.0],"label":1,"bucket":"noul"}"#)
        lines.append(#"{"logits":[-3.0,3.0],"label":0,"bucket":"noul"}"#)
        lines.append(#"{"logits":[3.0,-3.0],"label":0,"bucket":"noul"}"#)
    }
    try (lines.joined(separator: "\n") + "\n").write(to: inputURL, atomically: true, encoding: .utf8)

    let report = try Calibrate.fitCalibrationFile(
        inputURL: inputURL,
        outputURL: outputURL,
        chain: "guardrail",
        includeIsotonic: true
    )
    #expect(report.sampleCount == 72)
    #expect(report.profile.defaultTemperature > 1.2)
    #expect(report.metrics.nllAfter < report.metrics.nllBefore)
    #expect(report.metrics.eceAfter <= report.metrics.eceBefore)
    #expect(report.profile.tables["noul"] != nil)

    let savedData = try Data(contentsOf: outputURL)
    let reloaded = try QWCalibrationProfile.load(fromData: savedData)
    #expect(abs(reloaded.defaultTemperature - report.profile.defaultTemperature) < 1e-6)
    #expect(reloaded.chainTemperatures["guardrail"]?["noul"] != nil)
}
