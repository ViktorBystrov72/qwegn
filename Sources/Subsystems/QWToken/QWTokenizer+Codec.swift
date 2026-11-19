// QWTokenizer codec — encode (NFC → specials-split → regex words →
// ByteLevel → BPE merges), decode (piece join + inverse byte map), and the
// incremental (streaming) decoder with UTF-8 partial buffering.
// Every rule is bound by the 10k goldens; the String-== Unicode-canonical
// trap is handled by doing ALL merge/special matching on UTF-8 BYTES and
// keeping vocab symbol strings inside the byte-mapped range (no scalar in
// 32…126 ∪ 160…255 has a canonical composition with another member — ccc 0
import Foundation
import QWCore

/// Module marker (was Placeholder.swift).

extension QWTokenizer {
    public func piece(_ id: Int) -> String? { idToPiece[id] ?? added.first(where: { $0.id == id })?.s }

    public func tokenId(for token: String) -> Int? {
        vocab[token] ?? added.first(where: { $0.s == token })?.id
    }

    public func resolveSpecialTokens(fallback: QWModelSpecialTokens) -> QWModelSpecialTokens {
        let imStart = tokenId(for: "<|im_start|>")
            ?? tokenId(for: "<|ernie:im_start|>")
            ?? tokenId(for: "<|maxwell_start|>")
            ?? fallback.imStartTokenId
        let imEnd = tokenId(for: "<|im_end|>")
            ?? tokenId(for: "<|ernie:im_end|>")
            ?? tokenId(for: "<|maxwell_end|>")
            ?? fallback.imEndTokenId
        let think = tokenId(for: "<think>")
            ?? tokenId(for: "<ernie:think>")
            ?? fallback.thinkTokenId
        let endThink = tokenId(for: "</think>")
            ?? tokenId(for: "</ernie:think>")
            ?? fallback.endThinkTokenId
        var extra = fallback.extraStopTokenIds
        for s in ["<|im_end|>", "<|ernie:im_end|>", "<|maxwell_end|>", "<|ERNIE_endofprompt|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
            if let id = tokenId(for: s), !extra.contains(id) {
                extra.append(id)
            }
        }
        let hasChatIds = fallback.usesChatTemplateIds
            || tokenId(for: "<|im_end|>") != nil
            || tokenId(for: "<|ernie:im_end|>") != nil
            || tokenId(for: "<|maxwell_end|>") != nil
        return QWModelSpecialTokens(
            eosTokenId: fallback.eosTokenId,
            imStartTokenId: imStart,
            imEndTokenId: imEnd,
            thinkTokenId: think,
            endThinkTokenId: endThink,
            extraStopTokenIds: extra,
            usesChatTemplateIds: hasChatIds
        )
    }

    // MARK: BPE one word (already ByteLevel-mapped) → ids
    /// Merge law (engine parity): min-rank adjacent pair wins, leftmost on
    /// ties (strict `<` scan), ALL occurrences of that pair merge in one
    /// left-to-right pass; repeat until no ranked pair remains.
    func bpeWord(_ bytesSlice: ArraySlice<UInt8>) -> [Int] {
        guard !bytesSlice.isEmpty else { return [] }
        var word: [UInt32] = []
        word.reserveCapacity(bytesSlice.count)
        for b in bytesSlice {
            word.append(byteToSymId[Int(b)])
        }
        return bpeMerge(word)
    }

    func bpeMerge(_ initialWord: [UInt32]) -> [Int] {
        var word = initialWord
        guard !word.isEmpty else { return [] }
        if word.count == 1 {
            return [symTok[word[0]] ?? -1]
        }
        while word.count > 1 {
            var bestKey = UInt64.max
            var bestRank = Int.max
            let count = word.count
            for i in 0..<(count - 1) {
                let k = UInt64(word[i]) << 32 | UInt64(word[i + 1])
                if let r = ranks[k], r < bestRank {
                    bestRank = r
                    bestKey = k
                }
            }
            guard bestKey != UInt64.max, let merged = pairMerged[bestKey] else { break }
            let a = UInt32(bestKey >> 32), b = UInt32(bestKey & 0xFFFF_FFFF)
            var w = 0
            var r = 0
            while r < word.count {
                if r + 1 < word.count && word[r] == a && word[r + 1] == b {
                    word[w] = merged
                    r += 2
                } else {
                    word[w] = word[r]
                    r += 1
                }
                w += 1
            }
            word.removeSubrange(w..<word.count)
        }
        return word.map { symTok[$0] ?? -1 }
    }

    // MARK: encode
    /// tokenizers parity: addSpecialTokens is accepted but a no-op — this
    /// checkpoint's post-processor adds nothing; pinned by goldens
    public func encode(_ text: String, addSpecialTokens: Bool = false) throws -> [Int] {
        _ = addSpecialTokens   // no-op law (see above)
        var ids: [Int] = []

        func encodeChunk(
            _ bytes: [UInt8],
            _ cLo: Int,
            _ cHi: Int,
            isAtTextStart: Bool,
            isFirstChunk: Bool
        ) throws {
            guard cHi > cLo else { return }
            if case .metaspaceByteFallback(let rep, let scheme, let splitWords) = mode {
                var seg = String(decoding: bytes[cLo..<cHi], as: UTF8.self)
                seg = seg.replacingOccurrences(of: " ", with: rep)
                let shouldPrepend = !rep.isEmpty && !seg.hasPrefix(rep)
                    && ((scheme == "always" && isFirstChunk) || (scheme == "first" && isAtTextStart && isFirstChunk))
                if shouldPrepend {
                    seg = rep + seg
                }
                let wordPieces = splitWords && !rep.isEmpty
                    ? Self.splitMetaspaceMergedWithNext(seg, delimiter: rep)
                    : [seg[...]]
                for piece in wordPieces where !piece.isEmpty {
                    var syms: [UInt32] = []
                    syms.reserveCapacity(piece.unicodeScalars.count)
                    for sc in piece.unicodeScalars {
                        let s = String(sc)
                        if let sid = symId[s] {
                            syms.append(sid)
                        } else {
                            for b in s.utf8 { syms.append(byteToSymId[Int(b)]) }
                        }
                    }
                    for g in bpeMerge(syms) {
                        if g < 0 { throw Error.unresolvedByte(String(piece.prefix(16))) }
                        ids.append(g)
                    }
                }
                return
            }
            var spans: [(Int, Int)] = [(cLo, cHi)]
            for step in preSteps {
                let re: NSRegularExpression
                switch step {
                case .digits(_, let r): re = r
                case .splitIsolated(let r): re = r
                }
                var nextSpans: [(Int, Int)] = []
                nextSpans.reserveCapacity(spans.count * 2)
                for (sLo, sHi) in spans where sHi > sLo {
                    let seg = String(decoding: bytes[sLo..<sHi], as: UTF8.self)
                    let ns = seg as NSString
                    let u2b = QWTokenizer.utf16ToByteTable(seg)
                    let segBytes = sHi - sLo
                    var cursor = 0
                    re.enumerateMatches(in: seg,
                                        range: NSRange(location: 0, length: ns.length)) { mr, _, _ in
                        guard let r = mr?.range, r.length > 0 else { return }
                        let b0 = u2b[r.location]
                        let b1 = u2b[r.location + r.length]
                        if b0 > cursor {
                            nextSpans.append((sLo + cursor, sLo + b0))
                        }
                        if b1 > b0 {
                            nextSpans.append((sLo + b0, sLo + b1))
                        }
                        cursor = max(cursor, b1)
                    }
                    if cursor < segBytes {
                        nextSpans.append((sLo + cursor, sHi))
                    }
                }
                spans = nextSpans
            }
            for (pLo, pHi) in spans where pHi > pLo {
                let got = bpeWord(bytes[pLo..<pHi])
                if got.isEmpty {
                    throw Error.unresolvedByte(String(String(
                        decoding: bytes[pLo..<pHi], as: UTF8.self).prefix(16)))
                }
                for g in got {
                    if g < 0 {
                        throw Error.unresolvedByte(String(String(
                            decoding: bytes[pLo..<pHi], as: UTF8.self).prefix(16)))
                    }
                    ids.append(g)
                }
            }
        }

        func encodeSegment(_ bytes: [UInt8], _ lo: Int, _ hi: Int, isAtTextStart: Bool) throws {
            guard hi > lo else { return }
            let chunkSize = 16384
            if hi - lo <= chunkSize {
                try encodeChunk(bytes, lo, hi, isAtTextStart: isAtTextStart, isFirstChunk: true)
                return
            }
            var cur = lo
            while cur < hi {
                var nextEnd = min(cur + chunkSize, hi)
                if nextEnd < hi {
                    nextEnd = Self.chooseChunkSplitBoundary(bytes: bytes, cur: cur, limit: nextEnd, hi: hi, mode: mode)
                }
                try encodeChunk(bytes, cur, nextEnd, isAtTextStart: isAtTextStart, isFirstChunk: cur == lo)
                cur = nextEnd
            }
        }

        func encodeNormalizedSlice(_ sliceBytes: [UInt8], isAtTextStart: Bool) throws {
            guard !sliceBytes.isEmpty else { return }
            if normAddedPats.isEmpty {
                try encodeSegment(sliceBytes, 0, sliceBytes.count, isAtTextStart: isAtTextStart)
                return
            }
            let normMatches = Self.matchAddedTokens(
                in: sliceBytes, patterns: normAddedPats, firstByte: normAddedFirstByte
            )
            var pos = 0
            for m in normMatches {
                if m.loc > pos {
                    try encodeSegment(sliceBytes, pos, m.loc, isAtTextStart: isAtTextStart && pos == 0)
                }
                ids.append(m.id)
                pos = m.end
            }
            if pos < sliceBytes.count {
                try encodeSegment(sliceBytes, pos, sliceBytes.count, isAtTextStart: isAtTextStart && pos == 0)
            }
        }

        let rawBytes = Array(text.utf8)
        let rawMatches = rawAddedPats.isEmpty
            ? []
            : Self.matchAddedTokens(in: rawBytes, patterns: rawAddedPats, firstByte: rawAddedFirstByte)
        if rawMatches.isEmpty {
            let nfc = usesNFC ? text.precomposedStringWithCanonicalMapping : text
            try encodeNormalizedSlice(Array(nfc.utf8), isAtTextStart: true)
            return ids
        }

        var rawPos = 0
        for m in rawMatches {
            if m.loc > rawPos {
                let rawSub = String(decoding: rawBytes[rawPos..<m.loc], as: UTF8.self)
                let normSub = usesNFC ? rawSub.precomposedStringWithCanonicalMapping : rawSub
                try encodeNormalizedSlice(Array(normSub.utf8), isAtTextStart: rawPos == 0)
            }
            ids.append(m.id)
            rawPos = m.end
        }
        if rawPos < rawBytes.count {
            let rawSub = String(decoding: rawBytes[rawPos..<rawBytes.count], as: UTF8.self)
            let normSub = usesNFC ? rawSub.precomposedStringWithCanonicalMapping : rawSub
            try encodeNormalizedSlice(Array(normSub.utf8), isAtTextStart: false)
        }
        return ids
    }

    private static func chooseChunkSplitBoundary(
        bytes: [UInt8],
        cur: Int,
        limit: Int,
        hi: Int,
        mode: Mode
    ) -> Int {
        func findParagraphBoundary() -> Int? {
            var scan = limit
            var trailingRunStart: Int? = nil
            if bytes[scan] == 0x0A || bytes[scan] == 0x0D {
                while scan > cur && (bytes[scan - 1] == 0x0A || bytes[scan - 1] == 0x0D) {
                    scan -= 1
                }
                if scan > cur {
                    trailingRunStart = scan
                }
            }
            while scan > cur {
                if bytes[scan - 1] == 0x0A && bytes[scan] != 0x0A && bytes[scan] != 0x0D {
                    return scan
                }
                scan -= 1
            }
            return trailingRunStart
        }
        func findLeadingWhitespaceBoundary(includeTab: Bool) -> Int? {
            var ws = limit
            while ws > cur && bytes[ws - 1] != 0x20 && (!includeTab || bytes[ws - 1] != 0x09) {
                ws -= 1
            }
            guard ws > cur else { return nil }
            while ws > cur && (bytes[ws - 1] == 0x20 || (includeTab && bytes[ws - 1] == 0x09)) {
                ws -= 1
            }
            return ws > cur ? ws : nil
        }

        if case .metaspaceByteFallback = mode {
            if let ws = findLeadingWhitespaceBoundary(includeTab: false) { return ws }
            if let nl = findParagraphBoundary() { return nl }
            if let ts = findLeadingWhitespaceBoundary(includeTab: true) { return ts }
        } else {
            if let nl = findParagraphBoundary() { return nl }
            if let ws = findLeadingWhitespaceBoundary(includeTab: true) { return ws }
        }
        var end = limit
        while end < hi && (bytes[end] & 0xC0) == 0x80 {
            end += 1
        }
        return end
    }

    private static func splitMetaspaceMergedWithNext(_ s: String, delimiter: String) -> [Substring] {
        var out: [Substring] = []
        var pieceStart = s.startIndex
        var searchStart = s.startIndex
        while searchStart < s.endIndex,
              let range = s.range(of: delimiter, range: searchStart..<s.endIndex) {
            if range.lowerBound > pieceStart {
                out.append(s[pieceStart..<range.lowerBound])
                pieceStart = range.lowerBound
            }
            searchStart = range.upperBound
        }
        if pieceStart < s.endIndex {
            out.append(s[pieceStart..<s.endIndex])
        }
        return out
    }

    private static func matchAddedTokens(
        in bytes: [UInt8],
        patterns: [AddedPattern],
        firstByte: [Bool]
    ) -> [(loc: Int, end: Int, id: Int)] {
        let nBytes = bytes.count
        var matches: [(loc: Int, end: Int, id: Int)] = []
        var bi = 0
        var lastMatchEnd = 0
        while bi < nBytes {
            while bi < nBytes && !firstByte[Int(bytes[bi])] { bi += 1 }
            guard bi < nBytes else { break }
            var matched = false
            for p in patterns {
                let patEnd = bi + p.pat.count
                guard patEnd <= nBytes, bytes[bi..<patEnd].elementsEqual(p.pat) else { continue }
                if p.singleWord {
                    if bi > lastMatchEnd, let prev = utf8ScalarBefore(bytes, index: bi, minIndex: lastMatchEnd), isWordScalar(prev.scalar) {
                        continue
                    }
                    if patEnd < nBytes, let next = utf8ScalarAt(bytes, index: patEnd), isWordScalar(next.scalar) {
                        continue
                    }
                }
                var matchStart = bi
                var matchEnd = patEnd
                if p.lstrip {
                    while matchStart > lastMatchEnd,
                          let prev = utf8ScalarBefore(bytes, index: matchStart, minIndex: lastMatchEnd),
                          prev.scalar.properties.isWhitespace {
                        matchStart = prev.start
                    }
                }
                if p.rstrip {
                    while matchEnd < nBytes,
                          let next = utf8ScalarAt(bytes, index: matchEnd),
                          next.scalar.properties.isWhitespace {
                        matchEnd = next.end
                    }
                }
                matches.append((matchStart, matchEnd, p.id))
                lastMatchEnd = matchEnd
                bi = matchEnd
                matched = true
                break
            }
            if !matched {
                bi += 1
            }
        }
        return matches
    }

    private static func isWordScalar(_ sc: Unicode.Scalar) -> Bool {
        sc == "_" || sc.properties.isAlphabetic || sc.properties.numericType != nil
    }

    private static func utf8ScalarBefore(_ bytes: [UInt8], index: Int, minIndex: Int) -> (scalar: Unicode.Scalar, start: Int)? {
        guard index > minIndex else { return nil }
        var start = index - 1
        var steps = 0
        while start > minIndex && steps < 3 && (bytes[start] & 0xC0) == 0x80 {
            start -= 1
            steps += 1
        }
        var iter = bytes[start..<index].makeIterator()
        var utf8 = UTF8()
        guard case .scalarValue(let sc) = utf8.decode(&iter), iter.next() == nil else { return nil }
        return (sc, start)
    }

    private static func utf8ScalarAt(_ bytes: [UInt8], index: Int) -> (scalar: Unicode.Scalar, end: Int)? {
        guard index < bytes.count else { return nil }
        let limit = min(bytes.count, index + 4)
        var iter = bytes[index..<limit].makeIterator()
        var utf8 = UTF8()
        guard case .scalarValue(let sc) = utf8.decode(&iter) else { return nil }
        return (sc, index + qwUTF8Width(of: sc))
    }

    /// utf16-unit → byte-offset table (one entry per unit, plus a sentinel
    /// for the end position). Match ends always land on scalar boundaries,
    /// so per-scalar replication is exact.
    static func utf16ToByteTable(_ s: String) -> [Int] {
        var table = [Int]()
        table.reserveCapacity(s.utf16.count + 1)
        var b = 0
        for sc in s.unicodeScalars {
            let w = qwUTF8Width(of: sc)
            for _ in 0..<sc.utf16.count { table.append(b) }
            b += w
        }
        table.append(b)
        return table
    }

    /// The exact output bytes of every token id when decoded mid-sequence, indexed by id.
    /// Special ids and ids without a piece are nil; metaspace first-piece stripping is not modelled.
    public func tokenByteTable() -> [[UInt8]?] {
        let count = (idToPiece.keys.max() ?? -1) + 1
        var table = [[UInt8]?](repeating: nil, count: max(0, count))
        for (id, piece) in idToPiece where id >= 0 && !specialIds.contains(id) {
            var bytes: [UInt8] = []
            switch mode {
            case .byteLevel:
                for sc in piece.unicodeScalars {
                    if let b = byteUnmap[sc] {
                        bytes.append(b)
                    } else {
                        bytes.append(contentsOf: String(sc).utf8)
                    }
                }
            case .metaspaceByteFallback(let rep, _, _):
                if let fb = byteFallbackUnmap[id] {
                    bytes.append(fb)
                } else {
                    bytes.append(contentsOf: piece.replacingOccurrences(of: rep, with: " ").utf8)
                }
            }
            table[id] = bytes
        }
        return table
    }

    // MARK: decode
    /// Rust-equivalent (BPE::decode + ByteLevel post-processor): every id's
    /// VOCAB PIECE joins to one string; the joined string is mapped
    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) throws -> String {
        var out: [UInt8] = []
        var sawFirstPiece = false
        for id in ids {
            guard let piece = idToPiece[id] else {
                throw Error.unresolvedByte("id \(id) (no vocab piece at all)")
            }
            if specialIds.contains(id) {
                if skipSpecialTokens { continue }
                out.append(contentsOf: piece.utf8)     // content verbatim
                continue
            }
            switch mode {
            case .byteLevel:
                for sc in piece.unicodeScalars {
                    if let b = byteUnmap[sc] {
                        out.append(b)
                    } else {
                        out.append(contentsOf: String(sc).utf8)   // lossy path (see above)
                    }
                }
            case .metaspaceByteFallback(let rep, _, _):
                if let fb = byteFallbackUnmap[id] {
                    out.append(fb)
                } else {
                    var replaced = piece.replacingOccurrences(of: rep, with: " ")
                    if !sawFirstPiece && stripLeadingSpace && replaced.hasPrefix(" ") {
                        replaced.removeFirst()
                    }
                    out.append(contentsOf: replaced.utf8)
                }
                sawFirstPiece = true
            }
        }
        return String(decoding: out, as: UTF8.self)
    }
}

/// UTF-8 byte width of a scalar (exact encoder widths, allocation-free).
func qwUTF8Width(of sc: Unicode.Scalar) -> Int {
    switch sc.value {
    case 0x00..<0x80: return 1
    case 0x80..<0x800: return 2
    case 0x800..<0x10000: return 3
    default: return 4
    }
}

// MARK: - streaming (incremental) decoder
/// C22 needs decode-to-tokenize-while-generating: feed ids one at a time;
/// every emitted chunk is a UTF-8 scalar-boundary prefix, and the full
public struct QWStreamDecoder {
    let tok: QWTokenizer
    var pending: [UInt8] = []
    var sawFirstPiece: Bool = false

    public init(_ tok: QWTokenizer) { self.tok = tok }

    public mutating func decode(_ id: Int, skipSpecialTokens: Bool = false) throws -> String {
        if tok.specialIds.contains(id) {
            if skipSpecialTokens { return "" }
            guard let piece = tok.idToPiece[id] else {
                throw QWTokenizer.Error.unresolvedByte("id \(id) (no vocab piece at all)")
            }
            // Flush incomplete UTF-8 sequences before emitting special text so
            // replacement characters do not reorder across special token boundaries.
            var s = flush()
            s += piece
            return s
        }
        guard let piece = tok.idToPiece[id] else {
            throw QWTokenizer.Error.unresolvedByte("id \(id) (no vocab piece at all)")
        }
        switch tok.mode {
        case .byteLevel:
            for sc in piece.unicodeScalars {
                if let b = tok.byteUnmap[sc] {
                    pending.append(b)
                } else {
                    pending.append(contentsOf: String(sc).utf8)   // lossy path (see decode)
                }
            }
        case .metaspaceByteFallback(let rep, _, _):
            if let fb = tok.byteFallbackUnmap[id] {
                pending.append(fb)
            } else {
                var replaced = piece.replacingOccurrences(of: rep, with: " ")
                if !sawFirstPiece && tok.stripLeadingSpace && replaced.hasPrefix(" ") {
                    replaced.removeFirst()
                }
                pending.append(contentsOf: replaced.utf8)
            }
            sawFirstPiece = true
        }
        return drainPending()
    }

    /// End of stream: emit the lossy remainder (== one-shot lossy tail).
    public mutating func flush() -> String {
        guard !pending.isEmpty else { return "" }
        let s = String(decoding: pending, as: UTF8.self)
        pending.removeAll(keepingCapacity: false)
        return s
    }

    /// Emit the longest prefix ending on a scalar boundary; hold a
    /// structurally incomplete final sequence.
    private mutating func drainPending() -> String {
        let n = pending.count
        guard n > 0 else { return "" }
        // Walk back over trailing continuation bytes (max 3).
        var k = 0
        while k < 4, n - 1 - k >= 0, (pending[n - 1 - k] & 0xC0) == 0x80 { k += 1 }
        var hold = 0
        if n - k == 0 {
            hold = k                      // lone continuation run (orphan)
        } else {
            let lead = pending[n - k - 1]
            let w = QWStreamDecoder.utf8LeadWidth(lead)
            if k + 1 < w { hold = k + 1 }  // incomplete: lead + k continuations < w
        }
        let end = n - hold
        guard end > 0 else { return "" }
        let s = String(decoding: pending[0..<end], as: UTF8.self)
        pending.removeFirst(end)
        return s
    }

    /// Strict structural lead-byte width (0x80…0xC1 / 0xF5… are invalid
    /// leads — width 1 so they never hold continuation bytes).
    private static func utf8LeadWidth(_ lead: UInt8) -> Int {
        switch lead {
        case 0x00...0x7F: return 1
        case 0xC2...0xDF: return 2
        case 0xE0...0xEF: return 3
        case 0xF0...0xF4: return 4
        default: return 1
        }
    }
}
