// QWJsonSafetyTests.swift — Regression tests for QWJson parser bounds, escaping, and Codable order/errors.

import Foundation
@testable import QWToken
import XCTest

final class QWJsonSafetyTests: XCTestCase {
    func testNestingDepthLimitAllows128AndRejects129() throws {
        let ok = String(repeating: "[", count: 128) + "1" + String(repeating: "]", count: 128)
        _ = try QWJson.parse(ok)

        let tooDeepArray = String(repeating: "[", count: 129) + "1" + String(repeating: "]", count: 129)
        XCTAssertThrowsError(try QWJson.parse(tooDeepArray)) { err in
            XCTAssertTrue(err is QWJsonParseError)
        }

        var deepObjOpen = ""
        var deepObjClose = ""
        for _ in 0..<129 {
            deepObjOpen += "{\"k\":"
            deepObjClose += "}"
        }
        XCTAssertThrowsError(try QWJson.parse(deepObjOpen + "1" + deepObjClose)) { err in
            XCTAssertTrue(err is QWJsonParseError)
        }
    }

    func testRejectsTrailingGarbageButAllowsTrailingWhitespace() throws {
        XCTAssertEqual(try QWJson.parse("  42 \n\t "), .int(42))

        for bad in ["123 abc", "{} []", "true,", "\"ok\" 1", "nullx"] {
            XCTAssertThrowsError(try QWJson.parse(bad), "Expected failure on trailing garbage: \(bad)") { err in
                XCTAssertTrue(err is QWJsonParseError)
            }
        }
    }

    func testSurrogatesUnknownEscapesAndRawControlChars() throws {
        XCTAssertEqual(try QWJson.parse(#""\uD83D\uDE00""#), .string("😀"))
        XCTAssertEqual(try QWJson.parse(#""a\"\\\/\b\f\n\r\t""#), .string("a\"\\/\u{0008}\u{000C}\n\r\t"))

        let invalidStrings = [
            #""\uD800""#,
            #""\uD800\u0041""#,
            #""\uD800x""#,
            #""\uDC00""#,
            #""\x""#,
            #""\a""#,
            #""\'""#,
            "\"line1\nline2\"",
            "\"tab\there\"",
            "\"nul\u{0000}byte\""
        ]
        for raw in invalidStrings {
            XCTAssertThrowsError(try QWJson.parse(raw), "Expected failure on invalid string: \(raw)") { err in
                XCTAssertTrue(err is QWJsonParseError)
            }
        }
    }

    func testPyReprStringEscapesSingleQuotes() {
        let single = QWJson.string("it's a 'quote'")
        XCTAssertEqual(QWJson.array([single]).stringified, #"['it\'s a \'quote\'']"#)

        let obj = QWJson.object([("o'key", .string("v'al"))])
        XCTAssertEqual(obj.stringified, #"{'o\'key': 'v\'al'}"#)
    }

    func testCodablePreservesKeyInsertionOrderAndPropagatesErrors() throws {
        struct OrderedKeyedDecoder: Decoder {
            var codingPath: [CodingKey] = []
            var userInfo: [CodingUserInfoKey: Any] = [:]
            let pairs: [(String, QWJson)]

            func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
                KeyedDecodingContainer(KDC(pairs: pairs))
            }
            func unkeyedContainer() throws -> UnkeyedDecodingContainer {
                throw DecodingError.typeMismatch([Any].self, .init(codingPath: [], debugDescription: "not array"))
            }
            func singleValueContainer() throws -> SingleValueDecodingContainer {
                throw DecodingError.typeMismatch(String.self, .init(codingPath: [], debugDescription: "not scalar"))
            }

            struct KDC<Key: CodingKey>: KeyedDecodingContainerProtocol {
                var codingPath: [CodingKey] = []
                let pairs: [(String, QWJson)]
                var allKeys: [Key] { pairs.compactMap { Key(stringValue: $0.0) } }
                func contains(_ key: Key) -> Bool { pairs.contains { $0.0 == key.stringValue } }
                func decodeNil(forKey key: Key) throws -> Bool { false }
                func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
                    guard let entry = pairs.first(where: { $0.0 == key.stringValue }) else {
                        throw DecodingError.keyNotFound(key, .init(codingPath: codingPath, debugDescription: "missing"))
                    }
                    let data = Data(entry.1.tojson.utf8)
                    return try JSONDecoder().decode(type, from: data)
                }
                func nestedContainer<NestedKey: CodingKey>(
                    keyedBy type: NestedKey.Type, forKey key: Key
                ) throws -> KeyedDecodingContainer<NestedKey> {
                    throw DecodingError.typeMismatch([String: Any].self, .init(codingPath: codingPath, debugDescription: "unused"))
                }
                func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
                    throw DecodingError.typeMismatch([Any].self, .init(codingPath: codingPath, debugDescription: "unused"))
                }
                func superDecoder() throws -> Decoder { fatalError("unused") }
                func superDecoder(forKey key: Key) throws -> Decoder { fatalError("unused") }
            }
        }

        final class RecordingEncoder: Encoder {
            var codingPath: [CodingKey] = []
            var userInfo: [CodingUserInfoKey: Any] = [:]
            var recordedKeys: [String] = []

            func container<Key: CodingKey>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> {
                KeyedEncodingContainer(KEC(parent: self))
            }
            func unkeyedContainer() -> UnkeyedEncodingContainer { fatalError("unused") }
            func singleValueContainer() -> SingleValueEncodingContainer { SVC() }

            struct KEC<Key: CodingKey>: KeyedEncodingContainerProtocol {
                var codingPath: [CodingKey] = []
                let parent: RecordingEncoder
                mutating func encodeNil(forKey key: Key) throws { parent.recordedKeys.append(key.stringValue) }
                mutating func encode<T: Encodable>(_ value: T, forKey key: Key) throws {
                    parent.recordedKeys.append(key.stringValue)
                }
                mutating func nestedContainer<NestedKey: CodingKey>(
                    keyedBy keyType: NestedKey.Type, forKey key: Key
                ) -> KeyedEncodingContainer<NestedKey> { fatalError("unused") }
                mutating func nestedUnkeyedContainer(forKey key: Key) -> UnkeyedEncodingContainer { fatalError("unused") }
                mutating func superEncoder() -> Encoder { parent }
                mutating func superEncoder(forKey key: Key) -> Encoder { parent }
            }

            struct SVC: SingleValueEncodingContainer {
                var codingPath: [CodingKey] = []
                mutating func encodeNil() throws {}
                mutating func encode(_ value: Bool) throws {}
                mutating func encode(_ value: String) throws {}
                mutating func encode(_ value: Double) throws {}
                mutating func encode(_ value: Float) throws {}
                mutating func encode(_ value: Int) throws {}
                mutating func encode(_ value: Int8) throws {}
                mutating func encode(_ value: Int16) throws {}
                mutating func encode(_ value: Int32) throws {}
                mutating func encode(_ value: Int64) throws {}
                mutating func encode(_ value: UInt) throws {}
                mutating func encode(_ value: UInt8) throws {}
                mutating func encode(_ value: UInt16) throws {}
                mutating func encode(_ value: UInt32) throws {}
                mutating func encode(_ value: UInt64) throws {}
                mutating func encode<T: Encodable>(_ value: T) throws {}
            }
        }

        let decoder = OrderedKeyedDecoder(pairs: [("z", .int(1)), ("a", .int(2)), ("m", .int(3))])
        let decoded = try QWJson(from: decoder)
        XCTAssertEqual(decoded, .object([("z", .int(1)), ("a", .int(2)), ("m", .int(3))]))

        let encoder = RecordingEncoder()
        try decoded.encode(to: encoder)
        XCTAssertEqual(encoder.recordedKeys, ["z", "a", "m"])

        struct UnsupportedScalarDecoder: Decoder {
            var codingPath: [CodingKey] = []
            var userInfo: [CodingUserInfoKey: Any] = [:]
            func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
                throw DecodingError.typeMismatch([String: Any].self, .init(codingPath: [], debugDescription: "not dict"))
            }
            func unkeyedContainer() throws -> UnkeyedDecodingContainer {
                throw DecodingError.typeMismatch([Any].self, .init(codingPath: [], debugDescription: "not array"))
            }
            func singleValueContainer() throws -> SingleValueDecodingContainer { SVC() }
            struct SVC: SingleValueDecodingContainer {
                var codingPath: [CodingKey] = []
                func decodeNil() -> Bool { false }
                func decode<T: Decodable>(_ type: T.Type) throws -> T {
                    throw DecodingError.typeMismatch(type, .init(codingPath: [], debugDescription: "unsupported"))
                }
            }
        }
        XCTAssertThrowsError(try QWJson(from: UnsupportedScalarDecoder()))
    }

    func testOrderedRootUserInfoPreservesKeyOrderUnderFoundationJSONDecoder() throws {
        struct Payload: Decodable {
            let tools: [QWJson]
            let nested: QWJson
        }
        let raw = """
        {
          "tools": [
            {
              "z_last": 1,
              "type": "function",
              "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {
                  "type": "object",
                  "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer"}},
                  "required": ["zeta"]
                }
              }
            }
          ],
          "nested": {"x": 1},
          "nested": {"z": 10, "a": 20}
        }
        """
        let data = Data(raw.utf8)
        let root = try QWJson.parse(data)
        let expectedTool = """
        {"z_last": 1, "type": "function", "function": {"name": "get_weather", "description": "Weather lookup", "parameters": {"type": "object", "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer"}}, "required": ["zeta"]}}}
        """
        for _ in 0..<25 {
            let decoder = JSONDecoder()
            decoder.userInfo[QWJson.orderedRootUserInfoKey] = root
            let decoded = try decoder.decode(Payload.self, from: data)
            XCTAssertEqual(decoded.tools.count, 1)
            XCTAssertEqual(decoded.tools[0].tojson, expectedTool)
            XCTAssertEqual(decoded.nested, .object([("z", .int(10)), ("a", .int(20))]))
        }
    }
}
