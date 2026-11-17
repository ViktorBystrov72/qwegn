// QWStructuredOutput.swift — maps a chat request's response_format onto a
// compiled grammar, and caches the per-tokenizer token trie.

import Foundation
import QWToken

enum QWStructuredOutput {
    /// The grammar a request asks for; nil when it has no JSON response_format.
    static func grammar(for request: ChatCompletionRequest) throws -> QWJSONGrammar? {
        switch request.responseFormatType {
        case "json_object":
            return QWJSONGrammar.anyObject()
        case "json_schema":
            guard let format = request.responseFormat else { return nil }
            let nested = format.get("json_schema").get("schema")
            let schema = nested.isUndefined ? format.get("schema") : nested
            guard !schema.isUndefined else { return QWJSONGrammar.anyObject() }
            return try QWJSONGrammar.compile(schema: schema)
        default:
            return nil
        }
    }

    /// Whether the request demands enforcement (`json_schema.strict: true`).
    static func isStrict(_ request: ChatCompletionRequest) -> Bool {
        guard request.responseFormatType == "json_schema", let format = request.responseFormat else { return false }
        if case .bool(true) = format.get("json_schema").get("strict") { return true }
        if case .bool(true) = format.get("strict") { return true }
        return false
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var tables: [ObjectIdentifier: (QWTokenizer, QWJSONTokenTable)] = [:]
    nonisolated(unsafe) private static var byteTables: [ObjectIdentifier: (QWTokenizer, Int, [[UInt8]?])] = [:]

    /// The token trie for a tokenizer, built on first use.
    static func tokenTable(for tokenizer: QWTokenizer, vocabulary: Int) -> QWJSONTokenTable {
        let key = ObjectIdentifier(tokenizer)
        return lock.withLock {
            if let (_, table) = tables[key], table.vocabulary == vocabulary { return table }
            let table = QWJSONTokenTable(tokenizer: tokenizer, vocabulary: vocabulary)
            // Holding the tokenizer keeps its identifier from being reused.
            tables = [key: (tokenizer, table)]
            return table
        }
    }

    /// Per-token byte table trimmed or padded to `vocabulary` without building JSON tries.
    static func tokenBytes(for tokenizer: QWTokenizer, vocabulary: Int) -> [[UInt8]?] {
        let key = ObjectIdentifier(tokenizer)
        return lock.withLock {
            if let (_, table) = tables[key], table.vocabulary == vocabulary {
                return table.bytes
            }
            if let (_, cachedVocab, bytes) = byteTables[key], cachedVocab == vocabulary {
                return bytes
            }
            let raw = tokenizer.tokenByteTable()
            var normalized = Array(raw.prefix(vocabulary))
            if normalized.count < vocabulary {
                normalized += [[UInt8]?](repeating: nil, count: vocabulary - normalized.count)
            }
            byteTables = [key: (tokenizer, vocabulary, normalized)]
            return normalized
        }
    }
}
