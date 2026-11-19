// QWServerStabilityAuditTests.swift — Regression tests for C1.2 server stability fixes.

import Foundation
import QWCore
@testable import QWServer
import QWToken
import XCTest

final class QWServerStabilityAuditTests: XCTestCase {

    // MARK: - 1. QWToolCallParser large finite float safety

    func testToolCallParserHandlesLargeFiniteFloatsWithoutSIGILL() throws {
        let tools = try [QWJson.parse("""
        {
          "type": "function",
          "function": {
            "name": "repeat_action",
            "parameters": {
              "type": "object",
              "properties": {
                "count": { "type": "integer" },
                "exact": { "type": "integer" },
                "ratio": { "type": "number" }
              }
            }
          }
        }
        """)]

        let xmlLarge = """
        <tool_call>
        <function=repeat_action>
        <parameter=count>1e30</parameter>
        <parameter=exact>42.0</parameter>
        <parameter=ratio>1e30</parameter>
        </function>
        </tool_call>
        """
        let parsedXml = QWToolCallParser.parse(text: xmlLarge, tools: tools)
        let xmlCall = try XCTUnwrap(parsedXml.toolCalls?.first)
        XCTAssertEqual(xmlCall.function.name, "repeat_action")
        let xmlArgsData = try XCTUnwrap(xmlCall.function.arguments.data(using: .utf8))
        let xmlArgs = try XCTUnwrap(try JSONSerialization.jsonObject(with: xmlArgsData) as? [String: Any])
        XCTAssertEqual(xmlArgs["count"] as? String, "1e30")
        XCTAssertEqual(xmlArgs["exact"] as? Int, 42)
        XCTAssertEqual(try XCTUnwrap(xmlArgs["ratio"] as? Double), 1e30)

        let jsonLarge = """
        <tool_call>
        {"name": "repeat_action", "arguments": {"count": "1e30", "exact": "9.0"}}
        </tool_call>
        """
        let parsedJson = QWToolCallParser.parse(text: jsonLarge, tools: tools)
        let jsonCall = try XCTUnwrap(parsedJson.toolCalls?.first)
        let jsonArgsData = try XCTUnwrap(jsonCall.function.arguments.data(using: .utf8))
        let jsonArgs = try XCTUnwrap(try JSONSerialization.jsonObject(with: jsonArgsData) as? [String: Any])
        XCTAssertEqual(jsonArgs["count"] as? String, "1e30")
        XCTAssertEqual(jsonArgs["exact"] as? Int, 9)

        let boundary = """
        <tool_call>
        <function=repeat_action>
        <parameter=count>9223372036854775808.0</parameter>
        <parameter=exact>-1e30</parameter>
        </function>
        </tool_call>
        """
        let parsedBoundary = QWToolCallParser.parse(text: boundary, tools: tools)
        XCTAssertEqual(parsedBoundary.toolCalls?.count, 1)
    }

    // MARK: - 2. QWSystemMetricsSampler gpu-core-count Data safety

    func testDecodeGPUCoreCountHandlesUInt32UInt64AndMalformedLengths() {
        var u32 = UInt32(40).littleEndian
        let data32 = Data(bytes: &u32, count: MemoryLayout<UInt32>.size)
        XCTAssertEqual(QWSystemMetricsSampler.decodeGPUCoreCount(data32), 40)

        var u64 = UInt64(80).littleEndian
        let data64 = Data(bytes: &u64, count: MemoryLayout<UInt64>.size)
        XCTAssertEqual(QWSystemMetricsSampler.decodeGPUCoreCount(data64), 80)

        XCTAssertEqual(QWSystemMetricsSampler.decodeGPUCoreCount(Int(32)), 32)
        XCTAssertEqual(QWSystemMetricsSampler.decodeGPUCoreCount(NSNumber(value: 16)), 16)

        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount(Data()))
        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount(Data([1, 2, 3])))
        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount(Data([1, 2, 3, 4, 5])))
        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount(Data(repeating: 0, count: 16)))
        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount(Data(repeating: 0, count: 4)))
        XCTAssertNil(QWSystemMetricsSampler.decodeGPUCoreCount("40"))
    }

    // MARK: - 3. QWPrometheusExposition syntax

    func testPrometheusCacheBytesUsesValidClosingBraces() {
        let registry = QWEngineTelemetryRegistry()
        registry.updateCacheUsage(
            warmBytes: 1024,
            warmLimit: 4096,
            ssdBytes: 2048,
            ssdLimit: 8192
        )
        let output = registry.formatPrometheus()
        XCTAssertTrue(output.contains("qwegine_cache_bytes{tier=\"warm_ram\",limit=\"4096\"} 1024\n"))
        XCTAssertTrue(output.contains("qwegine_cache_bytes{tier=\"cold_ssd\",limit=\"8192\"} 2048\n"))
        XCTAssertFalse(output.contains("qwegine_cache_bytes{tier=\"warm_ram\",limit=\"4096\")"))
        XCTAssertFalse(output.contains("qwegine_cache_bytes{tier=\"cold_ssd\",limit=\"8192\")"))

        for line in output.split(separator: "\n") where !line.hasPrefix("#") && line.contains("{") {
            XCTAssertTrue(line.contains("} "), "Expected valid '}' closing label set in line: \(line)")
        }
    }

    // MARK: - 4. QWServerLogger safety and lock separation

    func testLoggerHandlesExtremeIntegersAndDoublesWithoutTrapping() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-logger-safety-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let logFile = tmpDir.appendingPathComponent("audit.log")

        let logger = QWServerLogger(level: .debug)
        logger.configureFileSink(path: logFile.path)

        final class Box: @unchecked Sendable {
            var lines: [String] = []
        }
        let box = Box()
        let subId = logger.subscribe { box.lines.append($0) }
        defer { logger.unsubscribe(subId) }

        logger.logRequest(
            method: "POST",
            path: "/v1/chat/completions",
            statusCode: Int.min,
            promptTokens: Int.max,
            completionTokens: -1,
            imagesCount: 0,
            cacheHitTokens: 0,
            ttftMs: 1e30,
            decodeTokPerSec: -1e30,
            totalSec: -0.25
        )
        logger.logRequest(
            method: "GET",
            path: "/health",
            statusCode: 200,
            promptTokens: 0,
            completionTokens: 0,
            ttftMs: .nan,
            decodeTokPerSec: .infinity,
            totalSec: -.infinity
        )

        XCTAssertEqual(box.lines.count, 2)
        XCTAssertTrue(box.lines[0].contains("\(Int.min)"))
        XCTAssertTrue(box.lines[0].contains("\(Int.max)"))
        XCTAssertTrue(box.lines[0].contains("-0.25s"))

        let persisted = try String(contentsOf: logFile, encoding: .utf8)
        XCTAssertTrue(persisted.contains("\(Int.min)"))
    }

    // MARK: - 5. QWJSONSchemaGrammar zero retain cycles, depth limit, and cycle validation

    func testGrammarAnyObjectAndRecursiveRefHaveZeroRetainCycles() throws {
        weak var weakAnyGrammar: QWJSONGrammar?
        weak var weakAnyArena: QWJSGrammarArena?
        weak var weakAnyRoot: QWJSNode?

        do {
            let g = QWJSONGrammar.anyObject()
            weakAnyGrammar = g
            weakAnyArena = g.arena
            weakAnyRoot = g.root
            XCTAssertNotNil(weakAnyGrammar)
            XCTAssertNotNil(weakAnyArena)
            XCTAssertNotNil(weakAnyRoot)
        }
        XCTAssertNil(weakAnyGrammar)
        XCTAssertNil(weakAnyArena)
        XCTAssertNil(weakAnyRoot)

        weak var weakRefGrammar: QWJSONGrammar?
        weak var weakRefArena: QWJSGrammarArena?
        weak var weakRefRoot: QWJSNode?
        weak var weakConstraint: QWStructuredOutputConstraint?

        do {
            let recursiveSchema = try QWJson.parse("""
            {
              "$defs": {
                "node": {
                  "type": "object",
                  "properties": {
                    "kids": { "type": "array", "items": { "$ref": "#/$defs/node" } }
                  },
                  "additionalProperties": false
                }
              },
              "$ref": "#/$defs/node"
            }
            """)
            let g = try QWJSONGrammar.compile(schema: recursiveSchema)
            weakRefGrammar = g
            weakRefArena = g.arena
            weakRefRoot = g.root
            let table = QWJSONTokenTable(tokenBytes: [Array("{".utf8), Array("}".utf8)], vocabulary: 2)
            let c = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [])
            weakConstraint = c
            XCTAssertFalse(c.isComplete)
        }
        XCTAssertNil(weakRefGrammar)
        XCTAssertNil(weakConstraint)
        XCTAssertNil(weakRefArena)
        XCTAssertNil(weakRefRoot)

        weak var weakRootRefArena: QWJSGrammarArena?
        do {
            let selfRefAdditional = try QWJson.parse(##"{"type":"object","additionalProperties":{"$ref":"#"}}"##)
            let g = try QWJSONGrammar.compile(schema: selfRefAdditional)
            weakRootRefArena = g.arena
            var okState = QWJState(root: g.root)
            for b in #"{"a":{"b":{}}}"#.utf8 {
                XCTAssertTrue(okState.step(b))
            }
            XCTAssertTrue(okState.canEnd)
            var badState = QWJState(root: g.root)
            for b in #"{"a":{"b":"#.utf8 {
                XCTAssertTrue(badState.step(b))
            }
            XCTAssertFalse(badState.step(UInt8(ascii: "1")))
        }
        XCTAssertNil(weakRootRefArena)
    }

    func testGrammarRejectsExcessiveNestingDepthAndVacuousRefCycles() throws {
        var deepSchema: QWJson = .object([("type", .string("string"))])
        for _ in 0..<64 {
            deepSchema = .object([("type", .string("array")), ("items", deepSchema)])
        }
        XCTAssertNoThrow(try QWJSONGrammar.compile(schema: deepSchema))

        let tooDeep = QWJson.object([("type", .string("array")), ("items", deepSchema)])
        XCTAssertThrowsError(try QWJSONGrammar.compile(schema: tooDeep)) { err in
            XCTAssertTrue(err is QWJSONSchemaError)
        }

        let vacuousSchemas = [
            ##"{"$ref":"#"}"##,
            ##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
            ##"{"type":"object","properties":{"x":{"$ref":"#/$defs/loop"}},"$defs":{"loop":{"$ref":"#/$defs/loop"}}}"##,
            ##"{"allOf":[{"$ref":"#"}]}"##,
            ##"{"$defs":{"a":{"anyOf":[{"type":"string"},{"$ref":"#/$defs/a"}]}},"$ref":"#/$defs/a"}"##,
            ##"{"$ref":"#/$defs/"}"##,
            ##"{"type":[]}"##
        ]
        for raw in vacuousSchemas {
            let parsed = try QWJson.parse(raw)
            XCTAssertThrowsError(try QWJSONGrammar.compile(schema: parsed), "Expected schema rejection for \(raw)") { err in
                XCTAssertTrue(err is QWJSONSchemaError, "Expected QWJSONSchemaError for \(raw), got \(err)")
            }
        }

        // Guarded mutual $ref across an object container remains valid.
        let guardedMutual = try QWJson.parse("""
        {
          "$defs": {
            "a": { "$ref": "#/$defs/b" },
            "b": {
              "type": "object",
              "properties": { "next": { "$ref": "#/$defs/a" } },
              "additionalProperties": false
            }
          },
          "$ref": "#/$defs/a"
        }
        """)
        let compiled = try QWJSONGrammar.compile(schema: guardedMutual)
        var state = QWJState(root: compiled.root)
        for b in #"{"next":{"next":{}}}"#.utf8 {
            XCTAssertTrue(state.step(b))
        }
        XCTAssertTrue(state.canEnd)
    }
}
