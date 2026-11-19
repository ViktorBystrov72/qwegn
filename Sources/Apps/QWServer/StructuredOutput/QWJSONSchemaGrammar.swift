// QWJSONSchemaGrammar.swift — supported JSON Schema subset compiled to a byte-level grammar.
// Supported: type, properties, required, additionalProperties, items, minItems, maxItems,
// enum, const, nullable, local $ref, and first-byte-disjoint anyOf/oneOf.
// Annotation keywords are ignored; any other keyword is rejected loudly.

import Foundation
import QWToken

public struct QWJSONSchemaError: Error, CustomStringConvertible, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { "unsupported response_format schema: \(message)" }
}

/// A literal set (enum/const values or object key names) as a byte trie.
/// Each terminal carries a payload (the key index for names, 0 for values).
final class QWJSLiteralTrie: @unchecked Sendable {
    private(set) var next: [[UInt8: Int32]] = [[:]]
    private(set) var payload: [Int32] = [-1]
    /// Bitmask of key payloads reachable from each node (key tries hold <= 64 keys).
    private(set) var reachable: [UInt64] = [0]

    func insert(_ bytes: [UInt8], payload value: Int32) {
        let bit: UInt64 = (value >= 0 && value < 64) ? (UInt64(1) << UInt64(value)) : 0
        var node: Int32 = 0
        reachable[0] |= bit
        for b in bytes {
            if let child = next[Int(node)][b] {
                node = child
            } else {
                let child = Int32(next.count)
                next.append([:])
                payload.append(-1)
                reachable.append(0)
                next[Int(node)][b] = child
                node = child
            }
            reachable[Int(node)] |= bit
        }
        payload[Int(node)] = value
    }

    @inline(__always) func child(_ node: Int32, _ b: UInt8) -> Int32? { next[Int(node)][b] }
    @inline(__always) func terminal(_ node: Int32) -> Int32 { payload[Int(node)] }
    @inline(__always) func isLeaf(_ node: Int32) -> Bool { next[Int(node)].isEmpty }
    var firstBytes: Set<UInt8> { Set(next[0].keys) }

    func words() -> [[UInt8]] {
        var out: [[UInt8]] = []
        func walk(_ node: Int32, _ prefix: [UInt8]) {
            if terminal(node) >= 0 { out.append(prefix) }
            for (b, c) in next[Int(node)] { walk(c, prefix + [b]) }
        }
        walk(0, [])
        return out
    }
}

struct QWJSObjectRecord: Sendable {
    var names: [String] = []
    var nodeIDs: [Int] = []
    var requiredMask: UInt64 = 0
    /// Schema node ID for keys outside `names`; nil forbids them.
    var additionalID: Int?
    let keyTrie = QWJSLiteralTrie()
    var allMask: UInt64 {
        names.count >= 64 ? ~0 : (UInt64(1) << UInt64(names.count)) - 1
    }
}

struct QWJSArrayRecord: Sendable {
    let itemsID: Int
    let minItems: Int
    let maxItems: Int?
}

struct QWJSNodeTypesRecord: Sendable {
    var object: QWJSObjectRecord?
    var array: QWJSArrayRecord?
    var string = false
    var number = false
    var integer = false
    var boolean = false
    var null = false

    var firstBytes: Set<UInt8> {
        var s = Set<UInt8>()
        if object != nil { s.insert(UInt8(ascii: "{")) }
        if array != nil { s.insert(UInt8(ascii: "[")) }
        if string { s.insert(UInt8(ascii: "\"")) }
        if number || integer { s.formUnion(Array("-0123456789".utf8)) }
        if boolean { s.formUnion([UInt8(ascii: "t"), UInt8(ascii: "f")]) }
        if null { s.insert(UInt8(ascii: "n")) }
        return s
    }
}

struct QWJSNodeRecord: Sendable {
    var types = QWJSNodeTypesRecord()
    var literals: QWJSLiteralTrie?
}

/// Owned value-record arena keeping all node back-edges as integer indices (zero retain cycles).
final class QWJSGrammarArena: @unchecked Sendable {
    var nodes: [QWJSNodeRecord] = []

    func append(_ record: QWJSNodeRecord = QWJSNodeRecord()) -> Int {
        let id = nodes.count
        nodes.append(record)
        return id
    }
}

struct QWJSObject: Sendable {
    struct NodesView: RandomAccessCollection, Sendable {
        let arena: QWJSGrammarArena
        let nodeIDs: [Int]
        var startIndex: Int { 0 }
        var endIndex: Int { nodeIDs.count }
        subscript(index: Int) -> QWJSNode {
            QWJSNode(arena: arena, id: nodeIDs[index])
        }
    }

    let arena: QWJSGrammarArena
    private let record: QWJSObjectRecord

    init(arena: QWJSGrammarArena, record: QWJSObjectRecord) {
        self.arena = arena
        self.record = record
    }

    var names: [String] { record.names }
    var nodes: NodesView { NodesView(arena: arena, nodeIDs: record.nodeIDs) }
    var requiredMask: UInt64 { record.requiredMask }
    var additional: QWJSNode? { record.additionalID.map { QWJSNode(arena: arena, id: $0) } }
    var keyTrie: QWJSLiteralTrie { record.keyTrie }
    var allMask: UInt64 { record.allMask }
}

struct QWJSArray: Sendable {
    let arena: QWJSGrammarArena
    let itemsID: Int
    let minItems: Int
    let maxItems: Int?

    init(arena: QWJSGrammarArena, itemsID: Int, minItems: Int, maxItems: Int?) {
        self.arena = arena
        self.itemsID = itemsID
        self.minItems = minItems
        self.maxItems = maxItems
    }

    init(items: QWJSNode, minItems: Int, maxItems: Int?) {
        self.arena = items.arena
        self.itemsID = items.id
        self.minItems = minItems
        self.maxItems = maxItems
    }

    var items: QWJSNode { QWJSNode(arena: arena, id: itemsID) }
}

struct QWJSTypes: Sendable {
    var object: QWJSObject?
    var array: QWJSArray?
    var string = false
    var number = false
    var integer = false
    var boolean = false
    var null = false

    init(arena: QWJSGrammarArena, record: QWJSNodeTypesRecord) {
        self.object = record.object.map { QWJSObject(arena: arena, record: $0) }
        self.array = record.array.map {
            QWJSArray(arena: arena, itemsID: $0.itemsID, minItems: $0.minItems, maxItems: $0.maxItems)
        }
        self.string = record.string
        self.number = record.number
        self.integer = record.integer
        self.boolean = record.boolean
        self.null = record.null
    }

    var firstBytes: Set<UInt8> {
        var s = Set<UInt8>()
        if object != nil { s.insert(UInt8(ascii: "{")) }
        if array != nil { s.insert(UInt8(ascii: "[")) }
        if string { s.insert(UInt8(ascii: "\"")) }
        if number || integer { s.formUnion(Array("-0123456789".utf8)) }
        if boolean { s.formUnion([UInt8(ascii: "t"), UInt8(ascii: "f")]) }
        if null { s.insert(UInt8(ascii: "n")) }
        return s
    }
}

/// A schema node view into an owned `QWJSGrammarArena`.
final class QWJSNode: @unchecked Sendable {
    let arena: QWJSGrammarArena
    let id: Int

    init(arena: QWJSGrammarArena, id: Int) {
        self.arena = arena
        self.id = id
    }

    convenience init() {
        let arena = QWJSGrammarArena()
        let id = arena.append()
        self.init(arena: arena, id: id)
    }

    var types: QWJSTypes { QWJSTypes(arena: arena, record: arena.nodes[id].types) }
    var literals: QWJSLiteralTrie? { arena.nodes[id].literals }
}

/// A compiled response format: the root node of the byte grammar.
public final class QWJSONGrammar: @unchecked Sendable {
    static let maxSchemaDepth = 64
    let arena: QWJSGrammarArena
    let rootID: Int
    let root: QWJSNode
    static let trueTrie = literalTrie("true")
    static let falseTrie = literalTrie("false")
    static let nullTrie = literalTrie("null")

    init(arena: QWJSGrammarArena, rootID: Int) {
        self.arena = arena
        self.rootID = rootID
        self.root = QWJSNode(arena: arena, id: rootID)
    }

    private static func literalTrie(_ word: String) -> QWJSLiteralTrie {
        let t = QWJSLiteralTrie()
        t.insert(Array(word.utf8), payload: 0)
        return t
    }

    /// Any JSON object (response_format json_object).
    public static func anyObject() -> QWJSONGrammar {
        let arena = QWJSGrammarArena()
        let anyID = allocateAnyNode(in: arena)
        let rootObject = QWJSObjectRecord(additionalID: anyID)
        let rootID = arena.append(QWJSNodeRecord(types: QWJSNodeTypesRecord(object: rootObject)))
        return QWJSONGrammar(arena: arena, rootID: rootID)
    }

    static func allocateAnyNode(in arena: QWJSGrammarArena) -> Int {
        let id = arena.append()
        let object = QWJSObjectRecord(additionalID: id)
        let array = QWJSArrayRecord(itemsID: id, minItems: 0, maxItems: nil)
        arena.nodes[id] = QWJSNodeRecord(
            types: QWJSNodeTypesRecord(
                object: object,
                array: array,
                string: true,
                number: true,
                integer: false,
                boolean: true,
                null: true
            ),
            literals: nil
        )
        return id
    }

    /// Compiles a JSON Schema (the supported subset) into a grammar.
    public static func compile(schema: QWJson) throws -> QWJSONGrammar {
        let arena = QWJSGrammarArena()
        let compiler = Compiler(root: schema, arena: arena)
        let rootID = try compiler.node(for: schema, depth: 0, unguardedRefs: [])
        return QWJSONGrammar(arena: arena, rootID: rootID)
    }

    private final class Compiler {
        let root: QWJson
        let arena: QWJSGrammarArena
        var refs: [String: Int] = [:]
        var sharedAnyNodeID: Int?
        init(root: QWJson, arena: QWJSGrammarArena) {
            self.root = root
            self.arena = arena
        }

        private func anyNodeID() -> Int {
            if let existing = sharedAnyNodeID { return existing }
            let id = QWJSONGrammar.allocateAnyNode(in: arena)
            sharedAnyNodeID = id
            return id
        }

        static let annotations: Set<String> = [
            "title", "description", "default", "examples", "$schema", "$id", "$comment",
            "deprecated", "readOnly", "writeOnly", "format", "contentMediaType",
            "contentEncoding", "$defs", "definitions",
        ]
        static let handled: Set<String> = [
            "type", "properties", "required", "additionalProperties", "items", "minItems",
            "maxItems", "enum", "const", "nullable", "$ref", "anyOf", "oneOf", "allOf",
        ]

        func node(for schema: QWJson, depth: Int, unguardedRefs: Set<String>) throws -> Int {
            guard depth <= QWJSONGrammar.maxSchemaDepth else {
                throw QWJSONSchemaError("schema nesting depth exceeds maximum of \(QWJSONGrammar.maxSchemaDepth)")
            }
            switch schema {
            case .bool(true): return anyNodeID()
            case .bool(false): throw QWJSONSchemaError("a false schema admits no value")
            case .object(let pairs): return try objectSchema(pairs, depth: depth, unguardedRefs: unguardedRefs)
            default: throw QWJSONSchemaError("a schema must be an object or a boolean")
            }
        }

        private func resolve(_ ref: String, depth: Int, unguardedRefs: Set<String>) throws -> Int {
            if unguardedRefs.contains(ref) {
                throw QWJSONSchemaError("cyclic $ref '\(ref)' without a structural container admits no value")
            }
            if let cached = refs[ref] { return cached }
            let target: QWJson
            if ref == "#" {
                target = root
            } else if ref.hasPrefix("#/$defs/") || ref.hasPrefix("#/definitions/") {
                let parts = ref.dropFirst(2).split(separator: "/", maxSplits: 1).map(String.init)
                guard parts.count == 2, !parts[1].isEmpty else {
                    throw QWJSONSchemaError("$ref \(ref) does not resolve")
                }
                let name = parts[1].replacingOccurrences(of: "~1", with: "/")
                    .replacingOccurrences(of: "~0", with: "~")
                let found = root.get(parts[0]).get(name)
                guard !found.isUndefined else { throw QWJSONSchemaError("$ref \(ref) does not resolve") }
                target = found
            } else {
                throw QWJSONSchemaError("only local $ref (#, #/$defs/..., #/definitions/...) is supported, got \(ref)")
            }
            // Allocate placeholder ID first so guarded container cycles resolve by index.
            let placeholderID = arena.append()
            refs[ref] = placeholderID
            var nextUnguarded = unguardedRefs
            nextUnguarded.insert(ref)
            let builtID = try node(for: target, depth: depth + 1, unguardedRefs: nextUnguarded)
            let builtRecord = arena.nodes[builtID]
            guard !builtRecord.types.firstBytes.isEmpty || builtRecord.literals != nil else {
                throw QWJSONSchemaError("$ref '\(ref)' resolves to an empty schema")
            }
            arena.nodes[placeholderID] = builtRecord
            return placeholderID
        }

        private func onlyAnnotationsBesides(_ key: String, _ dict: [String: QWJson]) -> Bool {
            dict.keys.allSatisfy { $0 == key || Self.annotations.contains($0) }
        }

        private func objectSchema(_ pairs: [(String, QWJson)], depth: Int, unguardedRefs: Set<String>) throws -> Int {
            var dict: [String: QWJson] = [:]
            for (k, v) in pairs { dict[k] = v }
            for key in dict.keys.sorted() where !Self.handled.contains(key) && !Self.annotations.contains(key) {
                throw QWJSONSchemaError("keyword '\(key)' is not supported")
            }
            if let ref = dict["$ref"] {
                guard case .string(let r) = ref else { throw QWJSONSchemaError("$ref must be a string") }
                guard onlyAnnotationsBesides("$ref", dict) else {
                    throw QWJSONSchemaError("$ref with sibling constraints is not supported")
                }
                return try resolve(r, depth: depth, unguardedRefs: unguardedRefs)
            }
            if let all = dict["allOf"] {
                guard case .array(let items) = all, items.count == 1, onlyAnnotationsBesides("allOf", dict) else {
                    throw QWJSONSchemaError("allOf is supported only with a single subschema and no siblings")
                }
                return try node(for: items[0], depth: depth + 1, unguardedRefs: unguardedRefs)
            }
            for key in ["anyOf", "oneOf"] {
                guard let branches = dict[key] else { continue }
                guard case .array(let items) = branches, !items.isEmpty else {
                    throw QWJSONSchemaError("\(key) must be a non-empty array")
                }
                guard onlyAnnotationsBesides(key, dict) else {
                    throw QWJSONSchemaError("\(key) with sibling constraints is not supported")
                }
                var merged = try node(for: items[0], depth: depth + 1, unguardedRefs: unguardedRefs)
                for item in items.dropFirst() {
                    let rhs = try node(for: item, depth: depth + 1, unguardedRefs: unguardedRefs)
                    merged = try union(merged, rhs)
                }
                return merged
            }
            let nullable: Bool
            if case .bool(true)? = dict["nullable"] { nullable = true } else { nullable = false }
            var result = QWJSNodeRecord()
            if let values = dict["enum"] ?? dict["const"].map({ QWJson.array([$0]) }) {
                guard case .array(let list) = values, !list.isEmpty else {
                    throw QWJSONSchemaError("enum must be a non-empty array")
                }
                let trie = QWJSLiteralTrie()
                for v in list { trie.insert(Array(QWJSONGrammar.canonical(v).utf8), payload: 0) }
                if nullable { trie.insert(Array("null".utf8), payload: 0) }
                result.literals = trie
                return arena.append(result)
            }
            var typeNames: [String]
            switch dict["type"] {
            case .string(let t)?: typeNames = [t]
            case .array(let list)?:
                typeNames = try list.map {
                    guard case .string(let t) = $0 else { throw QWJSONSchemaError("type entries must be strings") }
                    return t
                }
            case nil:
                if dict["properties"] != nil || dict["required"] != nil || dict["additionalProperties"] != nil {
                    typeNames = ["object"]
                } else if dict["items"] != nil || dict["minItems"] != nil || dict["maxItems"] != nil {
                    typeNames = ["array"]
                } else {
                    return anyNodeID()
                }
            default: throw QWJSONSchemaError("type must be a string or an array of strings")
            }
            if nullable { typeNames.append("null") }
            guard !typeNames.isEmpty else {
                throw QWJSONSchemaError("type array must not be empty")
            }
            for t in typeNames {
                switch t {
                case "object": result.types.object = try objectType(dict, depth: depth)
                case "array": result.types.array = try arrayType(dict, depth: depth)
                case "string": result.types.string = true
                case "number": result.types.number = true
                case "integer": result.types.integer = true
                case "boolean": result.types.boolean = true
                case "null": result.types.null = true
                default: throw QWJSONSchemaError("unknown type '\(t)'")
                }
            }
            guard !result.types.firstBytes.isEmpty || result.literals != nil else {
                throw QWJSONSchemaError("schema admits no valid JSON value")
            }
            return arena.append(result)
        }

        private func objectType(_ dict: [String: QWJson], depth: Int) throws -> QWJSObjectRecord {
            var object = QWJSObjectRecord()
            if let props = dict["properties"] {
                guard case .object(let pairs) = props else { throw QWJSONSchemaError("properties must be an object") }
                for (name, sub) in pairs {
                    guard !object.names.contains(name) else { throw QWJSONSchemaError("duplicate property '\(name)'") }
                    object.names.append(name)
                    object.nodeIDs.append(try node(for: sub, depth: depth + 1, unguardedRefs: []))
                }
            }
            switch dict["additionalProperties"] {
            case nil, .bool(true)?: object.additionalID = anyNodeID()
            case .bool(false)?: object.additionalID = nil
            case let sub?: object.additionalID = try node(for: sub, depth: depth + 1, unguardedRefs: [])
            }
            var required: [String] = []
            if let req = dict["required"] {
                guard case .array(let list) = req else { throw QWJSONSchemaError("required must be an array") }
                for entry in list {
                    guard case .string(let name) = entry else { throw QWJSONSchemaError("required entries must be strings") }
                    required.append(name)
                }
            }
            for name in required where !object.names.contains(name) {
                guard let extraID = object.additionalID else {
                    throw QWJSONSchemaError("required property '\(name)' is not allowed by properties")
                }
                object.names.append(name)
                object.nodeIDs.append(extraID)
            }
            guard object.names.count <= 64 else {
                throw QWJSONSchemaError("objects with more than 64 named properties are not supported")
            }
            for name in required {
                if let idx = object.names.firstIndex(of: name) {
                    object.requiredMask |= UInt64(1) << UInt64(idx)
                }
            }
            for (index, name) in object.names.enumerated() {
                object.keyTrie.insert(Array(QWJSONGrammar.escapedString(name).utf8) + [UInt8(ascii: "\"")],
                                      payload: Int32(index))
            }
            return object
        }

        private func arrayType(_ dict: [String: QWJson], depth: Int) throws -> QWJSArrayRecord {
            let itemsID: Int
            switch dict["items"] {
            case nil: itemsID = anyNodeID()
            case .array?: throw QWJSONSchemaError("tuple-form items is not supported")
            case let sub?: itemsID = try node(for: sub, depth: depth + 1, unguardedRefs: [])
            }
            func count(_ key: String) throws -> Int? {
                switch dict[key] {
                case nil: return nil
                case .int(let n)? where n >= 0: return n
                case .double(let d)? where d >= 0 && d == d.rounded() && d < 1e9: return Int(d)
                default: throw QWJSONSchemaError("\(key) must be a non-negative integer")
                }
            }
            let minItems = try count("minItems") ?? 0
            let maxItems = try count("maxItems")
            if let maxItems, maxItems < minItems { throw QWJSONSchemaError("maxItems is below minItems") }
            return QWJSArrayRecord(itemsID: itemsID, minItems: minItems, maxItems: maxItems)
        }

        private func union(_ idA: Int, _ idB: Int) throws -> Int {
            let a = arena.nodes[idA]
            let b = arena.nodes[idB]
            var out = QWJSNodeRecord()
            var types = a.types
            if let object = b.types.object {
                guard types.object == nil else { throw QWJSONSchemaError("anyOf with two object branches is not supported") }
                types.object = object
            }
            if let array = b.types.array {
                guard types.array == nil else { throw QWJSONSchemaError("anyOf with two array branches is not supported") }
                types.array = array
            }
            types.string = types.string || b.types.string
            types.number = types.number || b.types.number
            types.integer = types.integer || b.types.integer
            types.boolean = types.boolean || b.types.boolean
            types.null = types.null || b.types.null
            out.types = types
            switch (a.literals, b.literals) {
            case (nil, nil): break
            case (let l?, nil), (nil, let l?): out.literals = l
            case (let l?, let r?):
                let merged = QWJSLiteralTrie()
                for word in l.words() + r.words() { merged.insert(word, payload: 0) }
                out.literals = merged
            }
            if let lit = out.literals, !lit.firstBytes.isDisjoint(with: out.types.firstBytes) {
                throw QWJSONSchemaError("anyOf branches must be distinguishable by their first character")
            }
            return arena.append(out)
        }
    }

    /// The single canonical compact serialization the grammar accepts for a literal.
    static func canonical(_ value: QWJson) -> String {
        switch value {
        case .undefined, .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d):
            if d == d.rounded() && abs(d) < 1e15 { return String(Int64(d)) }
            return String(d)
        case .string(let s): return "\"" + escapedString(s) + "\""
        case .array(let items): return "[" + items.map(canonical).joined(separator: ",") + "]"
        case .object(let pairs):
            return "{" + pairs.map { "\"" + escapedString($0.0) + "\":" + canonical($0.1) }.joined(separator: ",") + "}"
        }
    }

    /// Minimal JSON string escaping: quote, backslash, and control characters only.
    static func escapedString(_ s: String) -> String {
        var out = ""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }
}
