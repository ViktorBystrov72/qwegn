// QWSemVer.swift — Strict SemVer 2.0.0 parsing and ordering for release checks.

import Foundation

/// Semantic version (`MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`) with SemVer 2.0.0 precedence.
public struct QWSemVer: Comparable, Hashable, Codable, Sendable, CustomStringConvertible {
    public let major: Int
    public let minor: Int
    public let patch: Int
    public let prerelease: [String]

    public var isPrerelease: Bool { !prerelease.isEmpty }

    public var description: String {
        let base = "\(major).\(minor).\(patch)"
        return prerelease.isEmpty ? base : "\(base)-\(prerelease.joined(separator: "."))"
    }

    public init(major: Int, minor: Int, patch: Int, prerelease: [String] = []) {
        self.major = max(0, major)
        self.minor = max(0, minor)
        self.patch = max(0, patch)
        self.prerelease = prerelease
    }

    public init?(_ raw: String) {
        var trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("v") || trimmed.hasPrefix("V") {
            trimmed.removeFirst()
        }
        guard !trimmed.isEmpty else { return nil }

        let withoutBuild = String(trimmed.split(separator: "+", maxSplits: 1, omittingEmptySubsequences: false)[0])
        let dashParts = withoutBuild.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        let coreStr = String(dashParts[0])
        let coreParts = coreStr.split(separator: ".", omittingEmptySubsequences: false)
        guard coreParts.count == 3 else { return nil }

        guard let maj = Self.parseNonNegativeDigits(coreParts[0]),
              let min = Self.parseNonNegativeDigits(coreParts[1]),
              let pat = Self.parseNonNegativeDigits(coreParts[2]) else {
            return nil
        }

        var pre: [String] = []
        if dashParts.count == 2 {
            let preStr = String(dashParts[1])
            guard !preStr.isEmpty else { return nil }
            let ids = preStr.split(separator: ".", omittingEmptySubsequences: false)
            for id in ids {
                guard !id.isEmpty, id.unicodeScalars.allSatisfy(Self.isPrereleaseScalar) else {
                    return nil
                }
                if id.unicodeScalars.allSatisfy(Self.isASCIIDigit), id.count > 1, id.hasPrefix("0") {
                    return nil
                }
                pre.append(String(id))
            }
        }

        self.major = maj
        self.minor = min
        self.patch = pat
        self.prerelease = pre
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        guard let parsed = QWSemVer(raw) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid semantic version string: '\(raw)'"
            )
        }
        self = parsed
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(description)
    }

    public static func < (lhs: QWSemVer, rhs: QWSemVer) -> Bool {
        if lhs.major != rhs.major { return lhs.major < rhs.major }
        if lhs.minor != rhs.minor { return lhs.minor < rhs.minor }
        if lhs.patch != rhs.patch { return lhs.patch < rhs.patch }

        if lhs.prerelease.isEmpty && rhs.prerelease.isEmpty { return false }
        if lhs.prerelease.isEmpty { return false }
        if rhs.prerelease.isEmpty { return true }

        let count = min(lhs.prerelease.count, rhs.prerelease.count)
        for idx in 0..<count {
            let a = lhs.prerelease[idx]
            let b = rhs.prerelease[idx]
            if a == b { continue }
            let aNum = Int(a).flatMap { a.unicodeScalars.allSatisfy(isASCIIDigit) ? $0 : nil }
            let bNum = Int(b).flatMap { b.unicodeScalars.allSatisfy(isASCIIDigit) ? $0 : nil }
            switch (aNum, bNum) {
            case let (.some(x), .some(y)):
                return x < y
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            case (.none, .none):
                return a < b
            }
        }
        return lhs.prerelease.count < rhs.prerelease.count
    }

    private static func parseNonNegativeDigits(_ sub: Substring) -> Int? {
        guard !sub.isEmpty, sub.unicodeScalars.allSatisfy(isASCIIDigit) else { return nil }
        if sub.count > 1, sub.hasPrefix("0") { return nil }
        return Int(sub)
    }

    private static func isASCIIDigit(_ s: UnicodeScalar) -> Bool {
        s.value >= 48 && s.value <= 57
    }

    private static func isPrereleaseScalar(_ s: UnicodeScalar) -> Bool {
        isASCIIDigit(s)
            || (s.value >= 65 && s.value <= 90)
            || (s.value >= 97 && s.value <= 122)
            || s.value == 45
    }
}
