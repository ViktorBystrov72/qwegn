// QWTokenizerOffsetsTests.swift — Tests for bijective UTF-8/UTF-16/scalar offset mapping, special token escaping list, and singleTokenId.
import Foundation
import XCTest
@testable import QWToken

final class QWTokenizerOffsetsTests: XCTestCase {
    private func makeByteLevelTokenizer(
        addedTokens: [[String: Any]],
        extraMerges: [[String]] = []
    ) throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let defaultMerges: [[String]] = [
            ["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"],
            ["Ġ", "w"], ["Ġw", "o"], ["Ġwo", "r"], ["Ġwor", "l"], ["Ġworl", "d"],
        ] + extraMerges
        for pair in defaultMerges {
            let joined = pair[0] + pair[1]
            if vocab[joined] == nil { vocab[joined] = nextId; nextId += 1 }
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false]
        ]
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": defaultMerges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre],
            "normalizer": ["type": "NFC"]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func makeMetaspaceTokenizer(prependScheme: String = "first") throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        for b in 0..<256 { vocab[String(format: "<0x%02X>", b)] = nextId; nextId += 1 }
        for s in ["▁", "h", "e", "l", "o", "w", "r", "d"] { vocab[s] = nextId; nextId += 1 }
        let merges = [["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"], ["▁", "hello"],
                      ["▁", "w"], ["w", "o"], ["wo", "r"], ["wor", "l"], ["worl", "d"]]
        for pair in merges {
            let joined = pair[0] + pair[1]
            if vocab[joined] == nil { vocab[joined] = nextId; nextId += 1 }
        }
        let root: [String: Any] = [
            "added_tokens": [["id": 5000, "content": "<|mask|>", "special": true, "normalized": false]],
            "model": ["type": "BPE", "vocab": vocab, "merges": merges, "byte_fallback": true],
            "pre_tokenizer": ["type": "Metaspace", "replacement": "▁", "prepend_scheme": prependScheme, "split": true],
            "decoder": ["decoders": [["type": "Strip", "content": " ", "start": 1, "stop": 0]]]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    func testEncodeWithOffsetsMatchesEncodeAndPreservesSubstrings() throws {
        let tok = try makeByteLevelTokenizer(addedTokens: [
            ["id": 5000, "content": "<|im_start|>", "special": true, "normalized": false],
            ["id": 5001, "content": "<|im_end|>", "special": true, "normalized": false]
        ])
        let samples = [
            "hello world",
            "café 🚀 東京",
            "<|im_start|>hello café🚀<|im_end|>",
        ]
        for text in samples {
            let direct = try tok.encode(text, addSpecialTokens: false)
            let mapped = try tok.encodeWithOffsets(text, addSpecialTokens: false)
            XCTAssertEqual(mapped.ids, direct)
            XCTAssertEqual(mapped.spans.count, direct.count)
            XCTAssertEqual(mapped.substring(forTokenRange: 0..<mapped.spans.count), text)

            var prevEnd = 0
            for span in mapped.spans {
                XCTAssertGreaterThanOrEqual(span.utf8Range.lowerBound, prevEnd)
                XCTAssertGreaterThanOrEqual(span.utf8Range.upperBound, span.utf8Range.lowerBound)
                XCTAssertLessThanOrEqual(span.utf8Range.upperBound, text.utf8.count)
                prevEnd = span.utf8Range.upperBound
            }
        }
        // Slicing exact "hello" inside "<|im_start|>hello café🚀<|im_end|>".
        let special = try tok.encodeWithOffsets("<|im_start|>hello world<|im_end|>")
        XCTAssertEqual(special.spans.first?.tokenId, 5000)
        XCTAssertEqual(special.spans.last?.tokenId, 5001)
        XCTAssertEqual(special.substring(forTokenRange: 1..<2), "hello")
        XCTAssertEqual(special.substring(forTokenRange: 2..<3), " world")
    }

    func testMetaspaceOffsetsAndSpecialLookupHelpers() throws {
        let tok = try makeMetaspaceTokenizer(prependScheme: "first")
        let mapped = try tok.encodeWithOffsets("hello world", addSpecialTokens: false)
        XCTAssertEqual(mapped.ids, try tok.encode("hello world"))
        XCTAssertEqual(mapped.substring(forTokenRange: 0..<mapped.spans.count), "hello world")

        let specials = tok.allSpecialTokenStrings()
        XCTAssertEqual(specials, ["<|mask|>"])
        XCTAssertEqual(tok.singleTokenId(forText: "<|mask|>"), 5000)
        XCTAssertNotNil(tok.singleTokenId(forText: "hello"))
        XCTAssertNil(tok.singleTokenId(forText: "hello world"))
    }
}
