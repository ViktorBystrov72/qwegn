// QWTokenHardeningTests.swift — Regression tests for Wave 2 QWToken fixes:
// streaming UTF-8 flush across specials, 16 KiB chunking & metaspace parity,
// added_tokens leftmost-longest + HF flags + raw-before-NFC, and negative IDs.
import Foundation
import XCTest
@testable import QWToken

final class QWTokenHardeningTests: XCTestCase {
    private func makeByteLevelTokenizer(
        addedTokens: [[String: Any]],
        extraMerges: [[String]] = [],
        extraVocab: [String: Int] = [:],
        usesNFC: Bool = true
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
        for (k, v) in extraVocab {
            vocab[k] = v
            nextId = max(nextId, v + 1)
        }
        let defaultMerges: [[String]] = [
            ["Ġ", "w"], ["Ġw", "o"], ["Ġwo", "r"], ["Ġwor", "d"],
            ["w", "o"], ["wo", "r"], ["wor", "d"],
            ["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"],
            ["Ġ", "h"], ["Ġh", "e"], ["Ġhe", "l"], ["Ġhel", "l"], ["Ġhell", "o"],
            ["Ċ", "Ċ"]
        ]
        var merges = defaultMerges
        merges.append(contentsOf: extraMerges)
        for pair in merges {
            let merged = pair[0] + pair[1]
            if vocab[merged] == nil {
                vocab[merged] = nextId
                nextId += 1
            }
        }
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let pre: [[String: Any]] = [
            ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false]
        ]
        var root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": pre]
        ]
        if usesNFC {
            root["normalizer"] = ["type": "NFC"]
        }
        let data = try JSONSerialization.data(withJSONObject: root)
        return try QWTokenizer(json: data)
    }

    private func makeMetaspaceTokenizer(
        prependScheme: String,
        split: Bool,
        addedTokens: [[String: Any]] = [],
        extraMerges: [[String]] = []
    ) throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = nextId
            nextId += 1
        }
        for s in ["▁", "a", "b", "c", "h", "e", "l", "o", "w", "r", "d", "f", "\n"] {
            if vocab[s] == nil {
                vocab[s] = nextId
                nextId += 1
            }
        }
        let merges = [["▁", "h"], ["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"],
                      ["▁", "w"], ["w", "o"], ["wo", "r"], ["wor", "l"], ["worl", "d"]] + extraMerges
        for m in merges {
            let joined = m[0] + m[1]
            if vocab[joined] == nil {
                vocab[joined] = nextId
                nextId += 1
            }
        }
        let root: [String: Any] = [
            "added_tokens": addedTokens,
            "model": ["type": "BPE", "vocab": vocab, "merges": merges, "byte_fallback": true],
            "pre_tokenizer": ["type": "Metaspace", "replacement": "▁", "prepend_scheme": prependScheme, "split": split],
            "decoder": ["decoders": [["type": "Strip", "content": " ", "start": 1, "stop": 0]]]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    func testStreamDecoderFlushesIncompleteUTF8BeforeSpecialToken() throws {
        let tok = try makeByteLevelTokenizer(addedTokens: [
            ["id": 5000, "content": "<|im_end|>", "special": true, "normalized": false]
        ])
        let c3Id = try XCTUnwrap(tok.vocab[String(tok.byteMap[0xC3])])
        let a9Id = try XCTUnwrap(tok.vocab[String(tok.byteMap[0xA9])])
        let ids = [c3Id, 5000, a9Id]

        let oneShotWithSpecials = try tok.decode(ids, skipSpecialTokens: false)
        var sd1 = QWStreamDecoder(tok)
        var streamedWithSpecials = ""
        for id in ids { streamedWithSpecials += try sd1.decode(id, skipSpecialTokens: false) }
        streamedWithSpecials += sd1.flush()
        XCTAssertEqual(streamedWithSpecials, oneShotWithSpecials)
        XCTAssertEqual(streamedWithSpecials, "\u{FFFD}<|im_end|>\u{FFFD}")

        let oneShotSkipSpecials = try tok.decode(ids, skipSpecialTokens: true)
        var sd2 = QWStreamDecoder(tok)
        var streamedSkipSpecials = ""
        for id in ids { streamedSkipSpecials += try sd2.decode(id, skipSpecialTokens: true) }
        streamedSkipSpecials += sd2.flush()
        XCTAssertEqual(streamedSkipSpecials, oneShotSkipSpecials)
        XCTAssertEqual(streamedSkipSpecials, "é")
    }

    func testByteLevel16KiBChunkingSplitsBeforeLeadingSpaceAndKeepsParagraphs() throws {
        let tok = try makeByteLevelTokenizer(addedTokens: [])
        let wordToken = try XCTUnwrap(tok.vocab["word"])
        let spaceWordToken = try XCTUnwrap(tok.vocab["Ġword"])
        let doubleNewlineToken = try XCTUnwrap(tok.vocab["ĊĊ"])

        // Place " word" directly straddling byte 16384: 3276 * 5 = 16380 bytes -> " word" spans 16380..16385.
        let spaceText = String(repeating: "word ", count: 3276) + "word hello"
        let spaceIds = try tok.encode(spaceText)
        XCTAssertEqual(spaceIds.first, wordToken)
        XCTAssertEqual(spaceIds.filter { $0 == spaceWordToken }.count, 3276)

        // Place "\n\n" straddling byte 16384: prefix 16383 bytes ending with "\n\n" crossing 16384.
        let paraPrefix = String(repeating: "a", count: 16383)
        let paraText = paraPrefix + "\n\nword"
        let paraIds = try tok.encode(paraText)
        XCTAssertTrue(paraIds.contains(doubleNewlineToken), "16 KiB boundary must not split \\n\\n")
    }

    func testMetaspace16KiBChunkingPrependAlwaysOnlyChunk0AndSplitWords() throws {
        let tokSplit = try makeMetaspaceTokenizer(
            prependScheme: "always",
            split: true,
            extraMerges: [["a", "▁"], ["a▁", "b"]]
        )
        let tokMerged = try makeMetaspaceTokenizer(
            prependScheme: "always",
            split: false,
            extraMerges: [["a", "▁"], ["a▁", "b"]]
        )

        let splitIds = try tokSplit.encode("a b")
        XCTAssertFalse(splitIds.contains(tokSplit.vocab["a▁b"]!), "split=true must block cross-word merge")
        XCTAssertEqual(try tokSplit.decode(splitIds, skipSpecialTokens: true), "a b")

        let mergedIds = try tokMerged.encode("a b")
        XCTAssertTrue(mergedIds.contains(tokMerged.vocab["a▁b"]!), "split=false allows cross-word merge")

        // > 20 KiB text with spaces across 16 KiB boundary must not inject extra leading ▁ at chunk 1.
        let longBody = String(repeating: "hello ", count: 3500) + "world"
        let longIds = try tokSplit.encode(longBody)
        let leadingSpaceCount = longIds.filter { $0 == tokSplit.vocab["▁"]! || $0 == tokSplit.vocab["▁h"]! || $0 == tokSplit.vocab["▁w"]! }.count
        XCTAssertEqual(leadingSpaceCount, 3501, "Only 1 initial prepend + 3500 real spaces expected across 16 KiB boundary")
    }

    func testAddedTokensLeftmostLongestHFFlagsAndRawBeforeNFC() throws {
        let added: [[String: Any]] = [
            ["id": 500, "content": "<tag>", "special": true, "normalized": false],
            ["id": 501, "content": "<tag_long>", "special": true, "normalized": false],
            ["id": 502, "content": "cat", "special": false, "single_word": true, "normalized": false],
            ["id": 503, "content": "<trim>", "special": true, "lstrip": true, "rstrip": true, "normalized": false],
            ["id": 504, "content": "e\u{0301}", "special": true, "normalized": false],
            ["id": 505, "content": "Å", "special": false, "normalized": true]
        ]
        let tok = try makeByteLevelTokenizer(addedTokens: added, usesNFC: true)

        XCTAssertEqual(try tok.encode("<tag_long><tag>"), [501, 500])

        let swIds = try tok.encode("cat catalog")
        XCTAssertEqual(swIds.first, 502)
        XCTAssertEqual(swIds.filter { $0 == 502 }.count, 1, "single_word must not match inside 'catalog'")

        let trimIds = try tok.encode("hello   <trim>   world")
        XCTAssertEqual(trimIds.filter { $0 == 503 }.count, 1)
        let spaceTok = try XCTUnwrap(tok.vocab["Ġ"])
        XCTAssertFalse(trimIds.contains(spaceTok), "lstrip/rstrip must consume surrounding spaces")

        XCTAssertEqual(try tok.encode("e\u{0301}"), [504], "non-normalized token must match before NFC")
        XCTAssertEqual(try tok.encode("\u{212B}"), [505], "normalized token must match after NFC")
    }

    func testTokenByteTableGuardsNegativeIds() throws {
        let tok = try makeByteLevelTokenizer(
            addedTokens: [["id": -1, "content": "<neg>", "special": false]],
            extraVocab: ["neg_vocab": -5]
        )
        let table = tok.tokenByteTable()
        XCTAssertFalse(table.isEmpty)
    }
}
