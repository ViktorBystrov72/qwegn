// QWNgramFormat+TestHelpers.swift — Test-only header constructor for QWNgram tests.
import Foundation
@testable import QWQuant

extension QWNgramFormat {
    static func makeHeader(
        version: UInt16,
        dtype: QWNgramDType,
        entryCount: UInt64,
        headDim: UInt16,
        rowBytes: UInt16,
        l1Hot: UInt32,
        note: String = "",
        codebook: [UInt16] = []
    ) throws -> [UInt8] {
        try checkVersionPairing(version: version, dtype: dtype)
        if dtype == .vq2g32 {
            _ = try validateVQ2Codebook(codebook)
        } else if dtype == .vq4g32 {
            _ = try validateVQ4Codebook(codebook)
        } else if !codebook.isEmpty {
            throw QWNgramFormatError.badHeader("codebook is only valid for v3 vq2g32 or vq4g32")
        }
        guard rowBytes == (try dtype.rowBytes(headDim: Int(headDim))) else {
            throw QWNgramFormatError.badRowSize(Int(rowBytes))
        }
        guard entryCount > 0 else {
            throw QWNgramFormatError.badHeader("entryCount is zero")
        }
        guard l1Hot <= entryCount else {
            throw QWNgramFormatError.badHeader("l1Hot \(l1Hot) exceeds entryCount \(entryCount)")
        }
        let noteBytes = Array(note.utf8)
        guard noteBytes.isEmpty || noteBytes.count == self.noteBytes else {
            throw QWNgramFormatError.badHeader("note must be \(self.noteBytes) bytes or empty")
        }
        var h = [UInt8](repeating: 0, count: headerBytes)
        func put16(_ o: Int, _ v: UInt16) { h[o] = UInt8(v & 0xFF); h[o + 1] = UInt8(v >> 8) }
        func put32(_ o: Int, _ v: UInt32) {
            for i in 0..<4 { h[o + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put64(_ o: Int, _ v: UInt64) {
            for i in 0..<8 { h[o + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        put64(0, magic)
        put16(8, version); put16(10, 0); put32(12, dtype.rawValue)
        put64(16, entryCount); put16(24, headDim); put16(26, rowBytes)
        put32(28, l1Hot); put64(32, UInt64(headerBytes))
        h[44] = UInt8(self.noteBytes)
        for (i, b) in noteBytes.enumerated() { h[45 + i] = b }
        if dtype == .vq2g32 || dtype == .vq4g32 {
            for (i, u) in codebook.enumerated() { put16(vq2CodebookOffset + i * 2, u) }
        }
        return h
    }
}
