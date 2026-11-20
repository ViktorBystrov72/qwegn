// Order-preserving JSON values and parser shared by model and server code.
import Foundation

// MARK: - jinja value model ------------------------------------------------
/// The template's runtime values. `.undefined` is a first-class case: the
/// template tests `is undefined` / `is defined` and `| default(...)`
public indirect enum QWJson: Equatable, Sendable {
    case undefined
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([QWJson])
    case object([(String, QWJson)])      // ORDER-PRESERVING (json.dumps law)

    public static func == (lhs: QWJson, rhs: QWJson) -> Bool {
        switch (lhs, rhs) {
        case (.undefined, .undefined), (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.double(let a), .double(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)):
            guard a.count == b.count else { return false }
            for i in 0..<a.count {
                if a[i].0 != b[i].0 || a[i].1 != b[i].1 { return false }
            }
            return true
        default: return false
        }
    }

    public var isUndefined: Bool { if case .undefined = self { return true }; return false }
    public var isString: Bool { if case .string = self { return true }; return false }

    /// jinja getattr on plain data: mapping key, else undefined.
    public func get(_ key: String) -> QWJson {
        if case .object(let pairs) = self {
            for (k, v) in pairs where k == key { return v }
        }
        return .undefined
    }

    /// jinja `needle in value`: mapping key membership, string substring,
    /// array element equality; false for anything else (Undefined contains).
    public func contains(_ needle: String) -> Bool {
        switch self {
        case .object: return !get(needle).isUndefined
        case .string(let s): return s.contains(needle)
        case .array(let a): return a.contains { $0.isString(needle) }
        default: return false
        }
    }

    public func isString(_ s: String) -> Bool {
        if case .string(let x) = self { return x == s }; return false
    }

    public subscript(key: String) -> QWJson {
        self.get(key)
    }

    public subscript(index: Int) -> QWJson {
        if case .array(let a) = self, index >= 0, index < a.count {
            return a[index]
        }
        return .undefined
    }

    public static func from(any: Any?) -> QWJson {
        guard let any = any else { return .null }
        if let j = any as? QWJson { return j }
        if let s = any as? String { return .string(s) }
        if let b = any as? Bool { return .bool(b) }
        if let i = any as? Int { return .int(i) }
        if let d = any as? Double { return .double(d) }
        if let a = any as? [Any] { return .array(a.map { from(any: $0) }) }
        if let o = any as? [(String, Any)] {
            return .object(o.map { ($0.0, from(any: $0.1)) })
        }
        if let d = any as? [String: Any] {
            return .object(d.keys.sorted().map { ($0, from(any: d[$0])) })
        }
        return .null
    }

    public static func parse(_ string: String) throws -> QWJson {
        try Parser(string).parse()
    }

    public static func parse(_ data: Data) throws -> QWJson {
        guard let s = String(data: data, encoding: .utf8) else {
            throw QWJsonParseError("Invalid UTF-8 in JSON")
        }
        return try parse(s)
    }

    private final class Parser {
        private static let maxDepth = 128
        let chars: [Unicode.Scalar]
        var pos = 0

        init(_ s: String) {
            self.chars = Array(s.unicodeScalars)
        }

        func skipWhitespace() {
            while pos < chars.count {
                let c = chars[pos].value
                if c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D {
                    pos += 1
                } else {
                    break
                }
            }
        }

        func parse() throws -> QWJson {
            skipWhitespace()
            let val = try parseValue(depth: 0)
            skipWhitespace()
            guard pos == chars.count else {
                throw QWJsonParseError("Unexpected trailing data at \(pos)")
            }
            return val
        }

        func parseValue(depth: Int) throws -> QWJson {
            skipWhitespace()
            guard pos < chars.count else {
                throw QWJsonParseError("Unexpected EOF in JSON")
            }
            let c = chars[pos].value
            switch c {
            case 0x7B: // {
                return try parseObject(depth: depth)
            case 0x5B: // [
                return try parseArray(depth: depth)
            case 0x22: // "
                return .string(try parseString())
            case 0x74: // t (true)
                if pos + 4 <= chars.count && chars[pos+1].value == 0x72 && chars[pos+2].value == 0x75 && chars[pos+3].value == 0x65 {
                    pos += 4
                    return .bool(true)
                }
                throw QWJsonParseError("Invalid token in JSON at \(pos)")
            case 0x66: // f (false)
                if pos + 5 <= chars.count && chars[pos+1].value == 0x61 && chars[pos+2].value == 0x6C && chars[pos+3].value == 0x73 && chars[pos+4].value == 0x65 {
                    pos += 5
                    return .bool(false)
                }
                throw QWJsonParseError("Invalid token in JSON at \(pos)")
            case 0x6E: // n (null)
                if pos + 4 <= chars.count && chars[pos+1].value == 0x75 && chars[pos+2].value == 0x6C && chars[pos+3].value == 0x6C {
                    pos += 4
                    return .null
                }
                throw QWJsonParseError("Invalid token in JSON at \(pos)")
            case 0x2D, 0x30...0x39: // -, 0...9
                return try parseNumber()
            default:
                throw QWJsonParseError("Unexpected character '\(chars[pos])' at \(pos)")
            }
        }

        func parseObject(depth: Int) throws -> QWJson {
            guard depth < Self.maxDepth else {
                throw QWJsonParseError("JSON nesting depth exceeds maximum of \(Self.maxDepth)")
            }
            pos += 1 // skip '{'
            skipWhitespace()
            var pairs: [(String, QWJson)] = []
            if pos < chars.count && chars[pos].value == 0x7D { // }
                pos += 1
                return .object(pairs)
            }
            while pos < chars.count {
                skipWhitespace()
                guard pos < chars.count && chars[pos].value == 0x22 else {
                    throw QWJsonParseError("Expected string key in object at \(pos)")
                }
                let key = try parseString()
                skipWhitespace()
                guard pos < chars.count && chars[pos].value == 0x3A else { // :
                    throw QWJsonParseError("Expected ':' after key in object at \(pos)")
                }
                pos += 1
                let val = try parseValue(depth: depth + 1)
                pairs.append((key, val))
                skipWhitespace()
                guard pos < chars.count else {
                    throw QWJsonParseError("Unterminated object at \(pos)")
                }
                if chars[pos].value == 0x7D { // }
                    pos += 1
                    return .object(pairs)
                }
                if chars[pos].value == 0x2C { // ,
                    pos += 1
                    continue
                }
                throw QWJsonParseError("Expected ',' or '}' in object at \(pos)")
            }
            throw QWJsonParseError("Unterminated object")
        }

        func parseArray(depth: Int) throws -> QWJson {
            guard depth < Self.maxDepth else {
                throw QWJsonParseError("JSON nesting depth exceeds maximum of \(Self.maxDepth)")
            }
            pos += 1 // skip '['
            skipWhitespace()
            var items: [QWJson] = []
            if pos < chars.count && chars[pos].value == 0x5D { // ]
                pos += 1
                return .array(items)
            }
            while pos < chars.count {
                let val = try parseValue(depth: depth + 1)
                items.append(val)
                skipWhitespace()
                guard pos < chars.count else {
                    throw QWJsonParseError("Unterminated array at \(pos)")
                }
                if chars[pos].value == 0x5D { // ]
                    pos += 1
                    return .array(items)
                }
                if chars[pos].value == 0x2C { // ,
                    pos += 1
                    continue
                }
                throw QWJsonParseError("Expected ',' or ']' in array at \(pos)")
            }
            throw QWJsonParseError("Unterminated array")
        }

        private func parseHex4() throws -> Int {
            guard pos + 4 <= chars.count else {
                throw QWJsonParseError("Short \\u escape")
            }
            var hex = 0
            for _ in 0..<4 {
                let h = chars[pos].value
                pos += 1
                let digit: Int
                switch h {
                case 0x30...0x39: digit = Int(h - 0x30)
                case 0x41...0x46: digit = Int(h - 0x41 + 10)
                case 0x61...0x66: digit = Int(h - 0x61 + 10)
                default: throw QWJsonParseError("Invalid hex in \\u escape")
                }
                hex = (hex << 4) | digit
            }
            return hex
        }

        func parseString() throws -> String {
            pos += 1 // skip opening '"'
            var out = ""
            while pos < chars.count {
                let sc = chars[pos]
                if sc.value == 0x22 { // "
                    pos += 1
                    return out
                }
                if sc.value == 0x5C { // \
                    pos += 1
                    guard pos < chars.count else {
                        throw QWJsonParseError("Unterminated escape in string")
                    }
                    let esc = chars[pos]
                    pos += 1
                    switch esc.value {
                    case 0x22: out.append("\"")
                    case 0x5C: out.append("\\")
                    case 0x2F: out.append("/")
                    case 0x62: out.append("\u{0008}")
                    case 0x66: out.append("\u{000C}")
                    case 0x6E: out.append("\n")
                    case 0x72: out.append("\r")
                    case 0x74: out.append("\t")
                    case 0x75: // uXXXX
                        let hex = try parseHex4()
                        if hex >= 0xD800 && hex <= 0xDBFF {
                            guard pos + 6 <= chars.count,
                                  chars[pos].value == 0x5C,
                                  chars[pos + 1].value == 0x75 else {
                                throw QWJsonParseError("Unpaired high surrogate in \\u escape")
                            }
                            pos += 2
                            let lowHex = try parseHex4()
                            guard lowHex >= 0xDC00 && lowHex <= 0xDFFF else {
                                throw QWJsonParseError("Invalid low surrogate in \\u escape")
                            }
                            let cp = 0x10000 + ((hex - 0xD800) << 10) + (lowHex - 0xDC00)
                            guard let u = Unicode.Scalar(cp) else {
                                throw QWJsonParseError("Invalid UTF-16 surrogate pair in \\u escape")
                            }
                            out.unicodeScalars.append(u)
                        } else if hex >= 0xDC00 && hex <= 0xDFFF {
                            throw QWJsonParseError("Unpaired low surrogate in \\u escape")
                        } else {
                            guard let u = Unicode.Scalar(hex) else {
                                throw QWJsonParseError("Invalid Unicode scalar in \\u escape")
                            }
                            out.unicodeScalars.append(u)
                        }
                    default:
                        throw QWJsonParseError("Invalid escape '\\\(esc)' in string")
                    }
                } else {
                    guard sc.value >= 0x20 else {
                        throw QWJsonParseError("Unescaped control character in string at \(pos)")
                    }
                    out.unicodeScalars.append(sc)
                    pos += 1
                }
            }
            throw QWJsonParseError("Unterminated string")
        }

        func parseNumber() throws -> QWJson {
            let start = pos
            var isFloat = false
            if chars[pos].value == 0x2D { pos += 1 } // '-'
            while pos < chars.count {
                let c = chars[pos].value
                if c >= 0x30 && c <= 0x39 {
                    pos += 1
                } else if c == 0x2E || c == 0x65 || c == 0x45 || c == 0x2B || c == 0x2D {
                    isFloat = true
                    pos += 1
                } else {
                    break
                }
            }
            let s = String(String.UnicodeScalarView(chars[start..<pos]))
            if isFloat {
                if let d = Double(s) { return .double(d) }
            } else {
                if let i = Int(s) { return .int(i) }
                if let d = Double(s) { return .double(d) }
            }
            throw QWJsonParseError("Invalid number: \(s)")
        }
    }

    /// python `str()` for `{{ name }}` interpolation sites (only the
    /// raise-message concat paths reach non-strings here).
    public var stringified: String {
        switch self {
        case .undefined: return ""
        case .null: return "None"
        case .bool(true): return "True"
        case .bool(false): return "False"
        case .int(let i): return String(i)
        case .double(let d): return QWJson.pyFloatString(d)
        case .string(let s): return s
        case .array(let a): return "[" + a.map { $0.pyRepr }.joined(separator: ", ") + "]"
        case .object(let o):
            return "{" + o.map { QWJson.pyReprKey($0.0) + ": " + $0.1.pyRepr }.joined(separator: ", ") + "}"
        }
    }

    static func pyFloatString(_ d: Double) -> String {
        if d.isNaN { return "nan" }
        if d.isInfinite { return d < 0 ? "-inf" : "inf" }
        return String(d)
    }
    static func jsonFloat(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d < 0 ? "-Infinity" : "Infinity" }
        return String(d)
    }
    static func pyReprKey(_ k: String) -> String { pyReprString(k) }
    static func pyReprString(_ s: String) -> String {
        let escaped = jsonQuote(s).dropFirst().dropLast()
            .replacingOccurrences(of: "'", with: "\\'")
        return "'" + escaped + "'"
    }

    var pyRepr: String {
        switch self {
        case .string(let s): return QWJson.pyReprString(s)
        default: return stringified
        }
    }

    /// transformers' `tojson` filter = json.dumps(x, ensure_ascii=False)
    /// with default separators (", ", ": "), insertion order preserved and
    /// non-ASCII emitted verbatim. Bound by the tok_chat tools/arguments
    public var tojson: String {
        switch self {
        case .undefined: return "undefined"        // engine never reaches (guarded)
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d): return QWJson.jsonFloat(d)
        case .string(let s): return QWJson.jsonQuote(s)
        case .array(let a): return "[" + a.map { $0.tojson }.joined(separator: ", ") + "]"
        case .object(let o):
            return "{" + o.map { QWJson.jsonQuote($0.0) + ": " + $0.1.tojson }.joined(separator: ", ") + "}"
        }
    }

    /// json.dumps string escaping (ensure_ascii=False): quote/backslash and
    /// control chars (\b \f \n \r \t short forms, other <0x20 as \\u00xx).
    public static func jsonQuote(_ s: String) -> String {
        var out = "\""
        for u in s.unicodeScalars {
            switch u.value {
            case 0x22: out += "\\\""
            case 0x5C: out += "\\\\"
            case 0x08: out += "\\b"
            case 0x0C: out += "\\f"
            case 0x0A: out += "\\n"
            case 0x0D: out += "\\r"
            case 0x09: out += "\\t"
            default:
                if u.value < 0x20 { out += String(format: "\\u%04x", u.value) }
                else { out.unicodeScalars.append(u) }
            }
        }
        out += "\""
        return out
    }
}

public struct QWJsonParseError: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    public var message: String
    public init(_ message: String) {
        self.message = message
    }
    public var description: String { "QWJson: \(message)" }
}

private struct QWJsonCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ string: String) { self.stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

extension QWJson: Codable {
    public static let orderedRootUserInfoKey = CodingUserInfoKey(rawValue: "qwegine.qwjson.orderedRoot")!

    private static func resolve(in root: QWJson, codingPath: [any CodingKey]) -> QWJson? {
        var current = root
        for key in codingPath {
            switch current {
            case .array(let items):
                guard let idx = key.intValue, idx >= 0, idx < items.count else { return nil }
                current = items[idx]
            case .object(let pairs):
                guard let match = pairs.last(where: { $0.0 == key.stringValue }) else { return nil }
                current = match.1
            default:
                return nil
            }
        }
        return current
    }

    public init(from decoder: Decoder) throws {
        if let root = decoder.userInfo[Self.orderedRootUserInfoKey] as? QWJson,
           let ordered = Self.resolve(in: root, codingPath: decoder.codingPath) {
            self = ordered
            return
        }
        if let keyed = try? decoder.container(keyedBy: QWJsonCodingKey.self) {
            var pairs: [(String, QWJson)] = []
            pairs.reserveCapacity(keyed.allKeys.count)
            for key in keyed.allKeys {
                pairs.append((key.stringValue, try keyed.decode(QWJson.self, forKey: key)))
            }
            self = .object(pairs)
            return
        }
        if var unkeyed = try? decoder.unkeyedContainer() {
            var items: [QWJson] = []
            if let count = unkeyed.count { items.reserveCapacity(count) }
            while !unkeyed.isAtEnd {
                items.append(try unkeyed.decode(QWJson.self))
            }
            self = .array(items)
            return
        }
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
            return
        }
        if let b = try? container.decode(Bool.self) {
            self = .bool(b)
            return
        }
        if let i = try? container.decode(Int.self) {
            self = .int(i)
            return
        }
        if let d = try? container.decode(Double.self) {
            self = .double(d)
            return
        }
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Unsupported JSON value for QWJson"
        )
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .undefined, .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        case .bool(let b):
            var container = encoder.singleValueContainer()
            try container.encode(b)
        case .int(let i):
            var container = encoder.singleValueContainer()
            try container.encode(i)
        case .double(let d):
            var container = encoder.singleValueContainer()
            try container.encode(d)
        case .string(let s):
            var container = encoder.singleValueContainer()
            try container.encode(s)
        case .array(let a):
            var container = encoder.unkeyedContainer()
            for item in a {
                try container.encode(item)
            }
        case .object(let pairs):
            var container = encoder.container(keyedBy: QWJsonCodingKey.self)
            for (k, v) in pairs {
                try container.encode(v, forKey: QWJsonCodingKey(k))
            }
        }
    }
}
