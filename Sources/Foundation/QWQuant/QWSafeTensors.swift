// Zero-copy safetensors reader: maps a shard read-only and indexes its
// tensors. A single-file checkpoint and a sharded one (index.json + shards)
// are both a QWSafeTensorsContainer. No tensor is ever copied into the heap.

import Darwin
import Foundation

public enum QWSafeTensorsError: Error, Equatable, CustomStringConvertible {
    case badHeader(String)
    case missingTensor(String)
    case unsupportedDType(String)

    public var description: String {
        switch self {
        case .badHeader(let detail): return "bad safetensors header: \(detail)"
        case .missingTensor(let name): return "safetensors has no tensor '\(name)'"
        case .unsupportedDType(let raw): return "safetensors dtype '\(raw)' is not supported"
        }
    }
}

/// The safetensors dtypes, in the spec's order. `bitSize` is in BITS: `F4`
/// and `F6_*` are sub-byte, so a byte-per-element model cannot express them.
public enum QWSafeTensorsDType: String, Sendable {
    case bool = "BOOL"
    case f4 = "F4"
    case f6E2M3 = "F6_E2M3"
    case f6E3M2 = "F6_E3M2"
    case u8 = "U8"
    case i8 = "I8"
    case f8E5M2 = "F8_E5M2"
    case f8E4M3 = "F8_E4M3"
    case f8E8M0 = "F8_E8M0"
    case f8E4M3FNUZ = "F8_E4M3FNUZ"
    case f8E5M2FNUZ = "F8_E5M2FNUZ"
    case i16 = "I16"
    case u16 = "U16"
    case f16 = "F16"
    case bf16 = "BF16"
    case i32 = "I32"
    case u32 = "U32"
    case f32 = "F32"
    case c64 = "C64"
    case f64 = "F64"
    case i64 = "I64"
    case u64 = "U64"

    /// The size of one element, in BITS.
    public var bitSize: Int {
        switch self {
        case .f4: return 4
        case .f6E2M3, .f6E3M2: return 6
        case .bool, .u8, .i8, .f8E5M2, .f8E4M3, .f8E8M0, .f8E4M3FNUZ, .f8E5M2FNUZ: return 8
        case .i16, .u16, .f16, .bf16: return 16
        case .i32, .u32, .f32: return 32
        case .c64, .f64, .i64, .u64: return 64
        }
    }
}

public struct QWSafeTensorsTensor: Sendable, Hashable {
    public let name: String
    public let dtype: QWSafeTensorsDType
    public let shape: [Int]
    /// Offset within the shard's **data** region.
    public let byteOffset: Int
    public let byteLength: Int

    /// Elements per row for a 2-D tensor; `nil` for other ranks.
    public var rows: Int? { shape.count == 2 ? shape[0] : nil }
    public var cols: Int? { shape.count == 2 ? shape[1] : nil }

    /// The number of elements, or `nil` if the shape overflows.
    public var elementCount: Int? {
        shape.reduce(1) { acc, dim -> Int? in
            guard let acc else { return nil }
            let (m, over) = acc.multipliedReportingOverflow(by: dim)
            return over ? nil : m
        }
    }
}

/// One shard's header plus its mapped bytes.
public final class QWSafeTensorsShard: @unchecked Sendable {
    public let url: URL
    /// Offset of the data region within the file (`8 + header length`).
    public let dataOffset: Int
    public let tensors: [String: QWSafeTensorsTensor]
    let data: Data          // holds the mmap alive

    init(url: URL, data: Data, dataOffset: Int, tensors: [String: QWSafeTensorsTensor]) {
        self.url = url
        self.data = data
        self.dataOffset = dataOffset
        self.tensors = tensors
    }

    public func adviseDontNeed() {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            let pageSize = Int(sysconf(_SC_PAGESIZE))
            let addr = Int(bitPattern: base)
            let alignedAddr = addr & ~(pageSize - 1)
            let alignedLen = raw.count + (addr - alignedAddr)
            if let ptr = UnsafeMutableRawPointer(bitPattern: alignedAddr) {
                _ = posix_madvise(ptr, alignedLen, POSIX_MADV_DONTNEED)
            }
        }
    }
}

/// A read-only view over one safetensors file or a sharded checkpoint.
///
/// The payload is mapped read-only and the header is parsed once.
public final class QWSafeTensorsContainer: @unchecked Sendable {
    private struct QuantSpec: Sendable { let bits: Int?; let groupSize: Int? }

    private struct QuantMetadata: Sendable {
        let defaultSpec: QuantSpec
        let perPrefix: [String: QuantSpec]

        func spec(for name: String) -> QuantSpec? {
            let stem = name.hasSuffix(".weight") ? String(name.dropLast(7)) : name
            let prefixes = ["model.language_model.", "language_model.model.", "model.", "language_model."]
            var candidates = [stem]
            if let p = prefixes.first(where: { stem.hasPrefix($0) }) {
                let rest = String(stem.dropFirst(p.count))
                candidates.append(rest)
                candidates.append(contentsOf: prefixes.map { $0 + rest })
            }
            for cand in candidates {
                if let hit = perPrefix[cand] {
                    return QuantSpec(bits: hit.bits ?? defaultSpec.bits, groupSize: hit.groupSize ?? defaultSpec.groupSize)
                }
            }
            return (defaultSpec.bits != nil || defaultSpec.groupSize != nil) ? defaultSpec : nil
        }

        static func load(near weightsURL: URL) -> QuantMetadata? {
            let dir = weightsURL.deletingLastPathComponent()
            let cfgs = [dir.appendingPathComponent("config.json"),
                        dir.appendingPathComponent("assets/config.json"),
                        URL(fileURLWithPath: weightsURL.path + ".assets/config.json")]
            for cfg in cfgs {
                guard let data = try? Data(contentsOf: cfg),
                      let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                let text = root["text_config"] as? [String: Any]
                guard let qObj = (text?["quantization"] as? [String: Any])
                    ?? (text?["quantization_config"] as? [String: Any])
                    ?? (root["quantization"] as? [String: Any])
                    ?? (root["quantization_config"] as? [String: Any]) else { continue }
                var perPrefix: [String: QuantSpec] = [:]
                for (k, v) in qObj {
                    if let sub = v as? [String: Any], (sub["bits"] as? Int) != nil || (sub["group_size"] as? Int) != nil {
                        perPrefix[k] = QuantSpec(bits: sub["bits"] as? Int, groupSize: sub["group_size"] as? Int)
                    }
                }
                let db = qObj["bits"] as? Int, dg = qObj["group_size"] as? Int
                if db != nil || dg != nil || !perPrefix.isEmpty {
                    return QuantMetadata(defaultSpec: QuantSpec(bits: db, groupSize: dg), perPrefix: perPrefix)
                }
            }
            return nil
        }
    }

    public let url: URL
    public let shards: [URL: QWSafeTensorsShard]
    /// The file's own tensor names -> (shard, tensor).
    private let index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    /// Canonical names the file does not use -> the tensor they name.
    private let aliases: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    private let quantMetadata: QuantMetadata?
    /// The file's tensors. The canonical aliases are deliberately NOT here:
    /// a consumer that loads every weight must not load one twice.
    public let tensors: [String: QWSafeTensorsTensor]

    public convenience init(url: URL) throws {
        try self.init(url: url, onlyTensors: nil)
    }

    private static func validatedShardURL(base: URL, fileName: String) throws -> URL {
        guard !fileName.isEmpty,
              !fileName.hasPrefix("/"),
              !fileName.contains("\\"),
              !fileName.contains("\0") else {
            throw QWSafeTensorsError.badHeader("invalid shard path '\(fileName)' in index.json")
        }
        let parts = fileName.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.contains("."), !parts.contains("..") else {
            throw QWSafeTensorsError.badHeader("path traversal shard '\(fileName)' in index.json")
        }
        let resolvedBase = base.resolvingSymlinksInPath().standardizedFileURL
        let basePrefix = resolvedBase.path.hasSuffix("/") ? resolvedBase.path : resolvedBase.path + "/"
        let shardURL = base.appendingPathComponent(fileName).standardizedFileURL
        let resolvedShard = shardURL.resolvingSymlinksInPath().standardizedFileURL
        guard resolvedShard.path.hasPrefix(basePrefix) else {
            throw QWSafeTensorsError.badHeader("shard '\(fileName)' escapes index directory")
        }
        return shardURL
    }

    /// A container over only the named tensors. A reader that needs 33 of a
    /// checkpoint's 131 shards should not map the other 98, so a partially
    /// downloaded checkpoint stays usable.
    public init(url: URL, onlyTensors wanted: Set<String>?) throws {
        self.url = url
        var index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] = [:]
        var shards: [URL: QWSafeTensorsShard] = [:]

        if url.lastPathComponent.hasSuffix(".index.json") {
            let base = url.deletingLastPathComponent()
            let raw = try Data(contentsOf: url)
            let obj = try JSONSerialization.jsonObject(with: raw) as? [String: Any] ?? [:]
            guard let weightMap = obj["weight_map"] as? [String: String] else {
                throw QWSafeTensorsError.badHeader("index.json has no weight_map")
            }
            for (tensorName, fileName) in weightMap {
                if let wanted, !wanted.contains(tensorName) { continue }
                let shardURL = try Self.validatedShardURL(base: base, fileName: fileName)
                let shard: QWSafeTensorsShard
                if let existing = shards[shardURL] {
                    shard = existing
                } else {
                    shard = try QWSafeTensorsShard.open(shardURL)
                    shards[shardURL] = shard
                }
                guard let t = shard.tensors[tensorName] else {
                    throw QWSafeTensorsError.badHeader(
                        "index.json names '\(tensorName)' but \(fileName) does not contain it")
                }
                index[tensorName] = (shard, t)
            }
            if let wanted {
                for name in wanted where index[name] == nil {
                    throw QWSafeTensorsError.missingTensor(name)
                }
            }
        } else {
            let shard = try QWSafeTensorsShard.open(url)
            shards[url.standardizedFileURL] = shard
            for (name, t) in shard.tensors { index[name] = (shard, t) }
        }

        self.shards = shards
        self.index = index
        self.aliases = QWSafeTensorsContainer.buildAliases(index)
        self.quantMetadata = QuantMetadata.load(near: url)
        self.tensors = index.mapValues { $0.1 }
    }

    /// Bridges the `model.language_model.` / `language_model.model.` / `model.` prefixes:
    /// HF and MLX write different prefix spellings for the same tensor across checkpoints.
    /// A canonical name claimed by more than one source is NOT aliased.
    private static func buildAliases(
        _ index: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)]
    ) -> [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] {
        let prefixes = ["model.language_model.", "language_model.model.", "model."]
        var candidates: [String: Set<String>] = [:]
        for name in index.keys {
            var matched = false
            for prefix in prefixes where name.hasPrefix(prefix) {
                matched = true
                let suffix = String(name.dropFirst(prefix.count))
                for targetPrefix in prefixes {
                    let canonical = targetPrefix + suffix
                    guard canonical != name else { continue }
                    candidates[canonical, default: []].insert(name)
                }
                break
            }
            if !matched && name.hasPrefix("language_model.") {
                let canonical = String(name.dropFirst("language_model.".count))
                if canonical != name {
                    candidates[canonical, default: []].insert(name)
                }
            }
        }
        var aliases: [String: (QWSafeTensorsShard, QWSafeTensorsTensor)] = [:]
        for (canonical, sources) in candidates {
            guard index[canonical] == nil, sources.count == 1, let source = sources.first else { continue }
            if let value = index[source] { aliases[canonical] = value }
        }
        return aliases
    }

    /// The tensor under `name`, whether the file spells it that way or the
    /// canonical bridge does. Its `name` is always the file's own name.
    public func tensor(_ name: String) -> QWSafeTensorsTensor? {
        locate(name)?.1
    }

    /// The shard a tensor lives in — the byte range below is within it.
    public func shardURL(of name: String) -> URL? {
        locate(name)?.0.url
    }

    private func locate(_ name: String) -> (QWSafeTensorsShard, QWSafeTensorsTensor)? {
        if let hit = index[name] { return hit }
        return aliases[name]
    }

    /// The tensor's byte range within its shard **file** (see `shardURL(of:)`).
    public func byteRange(of name: String) -> Range<Int>? {
        guard let (shard, t) = locate(name) else { return nil }
        let (start, ov0) = shard.dataOffset.addingReportingOverflow(t.byteOffset)
        let (end, ov1) = start.addingReportingOverflow(t.byteLength)
        guard !ov0, !ov1, start >= 0, end <= shard.data.count else { return nil }
        return start..<end
    }

    /// Reads a tensor's bytes without copying them into the heap.
    ///
    /// The pointer is valid only inside `body`; it must not outlive the call.
    public func withBytes<T>(of name: String, _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        return try shard.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let (start, ov0) = shard.dataOffset.addingReportingOverflow(t.byteOffset)
            let (end, ov1) = start.addingReportingOverflow(t.byteLength)
            guard !ov0, !ov1, start >= 0, end <= raw.count else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
            }
            return try body(UnsafeRawBufferPointer(rebasing: raw[start..<end]))
        }
    }

    /// Passes the whole mapped shard buffer and the tensor's `[start..<end]` byte range
    /// inside it so callers can page-align `bytesNoCopy` Metal mappings without copying.
    public func withShardRange<T>(of name: String,
                                  _ body: (QWSafeTensorsShard, UnsafeRawBufferPointer, Range<Int>) throws -> T) throws -> T {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        return try shard.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let (start, ov0) = shard.dataOffset.addingReportingOverflow(t.byteOffset)
            let (end, ov1) = start.addingReportingOverflow(t.byteLength)
            guard !ov0, !ov1, start >= 0, end <= raw.count else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
            }
            return try body(shard, raw, start..<end)
        }
    }

    /// A reader bound to one tensor's bytes: the shard and the range are
    /// resolved once, so a hot path pays no lookup and no range check per
    /// read. The pointer handed to `body` is valid only inside it.
    public func reader(of name: String) throws -> QWSafeTensorsReader {
        guard let (shard, t) = locate(name) else { throw QWSafeTensorsError.missingTensor(name) }
        let (start, ov0) = shard.dataOffset.addingReportingOverflow(t.byteOffset)
        let (end, ov1) = start.addingReportingOverflow(t.byteLength)
        guard !ov0, !ov1, start >= 0, end <= shard.data.count else {
            throw QWSafeTensorsError.badHeader(
                "tensor \(name) runs past the end of \(shard.url.lastPathComponent)")
        }
        return QWSafeTensorsReader(shard: shard, tensor: t)
    }

    public func adviseDontNeed() {
        for shard in shards.values {
            shard.adviseDontNeed()
        }
    }

    /// Advises the kernel that the interior full pages of `name` will not be read again,
    /// without touching boundary pages shared with adjacent tensors in the same shard.
    public func adviseDontNeed(of name: String) {
        guard let (shard, t) = locate(name) else { return }
        shard.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            let pageSize = Int(sysconf(_SC_PAGESIZE))
            let (start, ov0) = shard.dataOffset.addingReportingOverflow(t.byteOffset)
            let (rawEnd, ov1) = start.addingReportingOverflow(t.byteLength)
            guard pageSize > 0, Int(bitPattern: base) % pageSize == 0, !ov0, !ov1, start >= 0 else { return }
            let end = min(raw.count, rawEnd)
            let alignedStart = ((start + pageSize - 1) / pageSize) * pageSize
            let alignedEnd = (end / pageSize) * pageSize
            guard alignedEnd > alignedStart else { return }
            _ = posix_madvise(UnsafeMutableRawPointer(mutating: base.advanced(by: alignedStart)),
                              alignedEnd - alignedStart, POSIX_MADV_DONTNEED)
        }
    }

    /// True when the checkpoint stores MLX affine-quantized (`u32` + `.scales` + `.biases`) weights.
    public var isMLXCheckpoint: Bool {
        tensor("model.language_model.embed_tokens.scales") != nil
            || tensor("embed_tokens.scales") != nil
    }

    private func resolveMLXAffineParams(
        name: String,
        n: Int,
        wCols: Int,
        gCols: Int,
        expectedElementCount: Int?
    ) throws -> (bits: Int, groupSize: Int, k: Int) {
        let (wBits, ovBits) = wCols.multipliedReportingOverflow(by: 32)
        guard !ovBits else {
            throw QWSafeTensorsError.badHeader("MLX affine width overflow for \(name)")
        }
        if let expected = expectedElementCount {
            guard expected > 0, expected % n == 0 else {
                throw QWSafeTensorsError.badHeader("invalid MLX affine layout for \(name)")
            }
            let k = expected / n
            guard k > 0, k % gCols == 0, wBits % k == 0 else {
                throw QWSafeTensorsError.badHeader("invalid MLX affine layout for \(name)")
            }
            return (bits: wBits / k, groupSize: k / gCols, k: k)
        }
        guard wBits % gCols == 0 else {
            throw QWSafeTensorsError.badHeader("invalid MLX affine group count for \(name)")
        }
        let bitsPerGroup = wBits / gCols
        if let meta = quantMetadata?.spec(for: name) {
            if let b = meta.bits, let g = meta.groupSize,
               b > 0, g > 0, b * g == bitsPerGroup, QWBF16.isSupportedMLXAffineBits(b) {
                return (b, g, gCols * g)
            }
            if let g = meta.groupSize, g > 0, bitsPerGroup % g == 0 {
                let b = bitsPerGroup / g
                if QWBF16.isSupportedMLXAffineBits(b) { return (b, g, gCols * g) }
            }
            if let b = meta.bits, QWBF16.isSupportedMLXAffineBits(b), bitsPerGroup % b == 0 {
                let g = bitsPerGroup / b
                if [32, 64, 128, 256].contains(g) { return (b, g, gCols * g) }
            }
        }
        for b in [4, 8, 6, 3, 2, 5] where bitsPerGroup % b == 0 {
            let g = bitsPerGroup / b
            if [64, 32, 128, 256].contains(g) {
                return (b, g, gCols * g)
            }
        }
        throw QWSafeTensorsError.unsupportedDType("u32 unsupported MLX affine bitsPerGroup=\(bitsPerGroup) \(name)")
    }

    private static func dequantMLXGroupRow(
        rowRaw: UnsafeRawPointer, dPtr: UnsafeMutablePointer<Float>, dstRowBase: Int,
        gCols: Int, groupSize: Int, bits: Int, scaleAt: (Int) -> Float, biasAt: (Int) -> Float
    ) {
        for g in 0..<gCols {
            let scale = scaleAt(g), bias = biasAt(g), dstGroupBase = dstRowBase + g * groupSize
            if bits == 4 && groupSize % 8 == 0 {
                let wordBase = (g * groupSize) / 8
                for wIdx in 0..<(groupSize / 8) {
                    var word = UInt32(littleEndian: rowRaw.loadUnaligned(fromByteOffset: (wordBase + wIdx) * 4, as: UInt32.self))
                    for nib in 0..<8 { dPtr[dstGroupBase + wIdx * 8 + nib] = Float(word & 0xF) * scale + bias; word >>= 4 }
                }
            } else if bits == 8 && groupSize % 4 == 0 {
                let wordBase = (g * groupSize) / 4
                for wIdx in 0..<(groupSize / 4) {
                    var word = UInt32(littleEndian: rowRaw.loadUnaligned(fromByteOffset: (wordBase + wIdx) * 4, as: UInt32.self))
                    for byteIdx in 0..<4 { dPtr[dstGroupBase + wIdx * 4 + byteIdx] = Float(word & 0xFF) * scale + bias; word >>= 8 }
                }
            } else {
                for j in 0..<groupSize {
                    dPtr[dstGroupBase + j] = QWBF16.unpackMLXAffineQ(rowRaw: rowRaw, col: g * groupSize + j, bits: bits) * scale + bias
                }
            }
        }
    }

    private func readMLXAffineFloats(
        _ name: String, r: QWSafeTensorsReader, expectedElementCount: Int?
    ) throws -> [Float] {
        let info = r.tensorInfo
        guard let count = info.elementCount else { throw QWSafeTensorsError.badHeader("shape overflow for \(name)") }
        let basePrefix = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let sReader = try reader(of: basePrefix + ".scales")
        let bReader = try reader(of: basePrefix + ".biases")
        let sInfo = sReader.tensorInfo, bInfo = bReader.tensorInfo
        guard info.shape.count >= 2, let wCols = info.shape.last,
              let sElems = sInfo.elementCount, let bElems = bInfo.elementCount, let gCols = sInfo.shape.last else {
            throw QWSafeTensorsError.unsupportedDType("u32 invalid shape \(name)")
        }
        let n = info.shape.dropLast().reduce(1, *)
        guard n > 0, wCols > 0, gCols > 0, n * wCols == count,
              sElems == n * gCols, bElems == sElems, sInfo.shape == bInfo.shape else {
            throw QWSafeTensorsError.badHeader("u32 scales/biases shape mismatch \(name)")
        }
        let (bits, groupSize, k) = try resolveMLXAffineParams(
            name: name, n: n, wCols: wCols, gCols: gCols, expectedElementCount: expectedElementCount
        )
        let (outCount, ovOut) = n.multipliedReportingOverflow(by: k)
        guard !ovOut, outCount > 0, groupSize > 0, k == gCols * groupSize,
              QWBF16.isSupportedMLXAffineBits(bits) else {
            throw QWSafeTensorsError.unsupportedDType("u32 unsupported MLX affine bits=\(bits) \(name)")
        }
        let validSBDTypes: [QWSafeTensorsDType] = [.f16, .bf16, .f32]
        guard validSBDTypes.contains(sInfo.dtype), validSBDTypes.contains(bInfo.dtype) else {
            throw QWSafeTensorsError.unsupportedDType("u32 unsupported scale/bias dtype \(name)")
        }
        if (sInfo.dtype == .f16 || sInfo.dtype == .bf16), bInfo.dtype == sInfo.dtype {
            return try r.withBytes { wRaw in
                try sReader.withBytes { sRaw in
                    try bReader.withBytes { bRaw in
                        guard let wBase = wRaw.baseAddress, wRaw.count >= n * wCols * 4,
                              let sBase = sRaw.baseAddress, sRaw.count >= sElems * 2,
                              let bBase = bRaw.baseAddress, bRaw.count >= bElems * 2 else {
                            throw QWSafeTensorsError.badHeader("truncated MLX affine tensor \(name)")
                        }
                        let isF16 = sInfo.dtype == .f16
                        return [Float](unsafeUninitializedCapacity: outCount) { dst, initialized in
                            guard let dBase = dst.baseAddress else { initialized = 0; return }
                            nonisolated(unsafe) let dPtr = dBase
                            nonisolated(unsafe) let wPtr = wBase, sPtr = sBase, bPtr = bBase
                            let workers = n >= 64 ? min(16, n) : 1, perWorker = (n + workers - 1) / workers
                            DispatchQueue.concurrentPerform(iterations: workers) { w in
                                let startR = w * perWorker, endR = min(n, startR + perWorker)
                                guard startR < endR else { return }
                                for row in startR..<endR {
                                    let sbRow = row * gCols
                                    let decode: (UnsafeRawPointer, Int) -> Float = { p, g in
                                        let u = UInt16(littleEndian: p.loadUnaligned(fromByteOffset: (sbRow + g) * 2, as: UInt16.self))
                                        return isF16 ? Float(Float16(bitPattern: u)) : Float(bitPattern: UInt32(u) << 16)
                                    }
                                    Self.dequantMLXGroupRow(
                                        rowRaw: wPtr.advanced(by: row * wCols * 4),
                                        dPtr: dPtr, dstRowBase: row * k,
                                        gCols: gCols, groupSize: groupSize, bits: bits,
                                        scaleAt: { decode(sPtr, $0) }, biasAt: { decode(bPtr, $0) }
                                    )
                                }
                            }
                            initialized = outCount
                        }
                    }
                }
            }
        }
        let scales = try readFloats(basePrefix + ".scales", expectedElementCount: sElems)
        let biases = try readFloats(basePrefix + ".biases", expectedElementCount: bElems)
        guard scales.count == n * gCols, biases.count == n * gCols else {
            throw QWSafeTensorsError.badHeader("u32 scales/biases element mismatch \(name)")
        }
        return try r.withBytes { wRaw in
            guard let wPtr = wRaw.baseAddress, wRaw.count >= n * wCols * 4 else {
                throw QWSafeTensorsError.badHeader("truncated MLX affine tensor \(name)")
            }
            return [Float](unsafeUninitializedCapacity: outCount) { dst, initialized in
                guard let dPtr = dst.baseAddress else { initialized = 0; return }
                for row in 0..<n {
                    let sbRow = row * gCols
                    Self.dequantMLXGroupRow(
                        rowRaw: wPtr.advanced(by: row * wCols * 4),
                        dPtr: dPtr, dstRowBase: row * k,
                        gCols: gCols, groupSize: groupSize, bits: bits,
                        scaleAt: { scales[sbRow + $0] }, biasAt: { biases[sbRow + $0] }
                    )
                }
                initialized = outCount
            }
        }
    }

    /// Reads and dequantizes a tensor (`bf16`, `f16`, `f32`, or MLX affine `u32` with `.scales`/`.biases`) into `[Float]`.
    public func readFloats(_ name: String, expectedElementCount: Int? = nil) throws -> [Float] {
        let r = try reader(of: name)
        let info = r.tensorInfo
        if info.dtype == .u32 {
            return try readMLXAffineFloats(name, r: r, expectedElementCount: expectedElementCount)
        }
        guard let count = info.elementCount else {
            throw QWSafeTensorsError.badHeader("shape overflow for \(name)")
        }
        if let expected = expectedElementCount, count != expected {
            throw QWSafeTensorsError.badHeader("shape \(count) != \(expected) for \(name)")
        }
        if count == 0 {
            switch info.dtype {
            case .bf16, .f16, .f32: return []
            default: throw QWSafeTensorsError.unsupportedDType(info.dtype.rawValue)
            }
        }
        return try r.withBytes { raw in
            guard let base = raw.baseAddress else { throw QWSafeTensorsError.badHeader("empty tensor buffer for \(name)") }
            let elemBytes = info.dtype == .f32 ? 4 : 2
            guard info.dtype == .bf16 || info.dtype == .f16 || info.dtype == .f32 else {
                throw QWSafeTensorsError.unsupportedDType(info.dtype.rawValue)
            }
            guard raw.count >= count * elemBytes else {
                throw QWSafeTensorsError.badHeader("truncated \(info.dtype.rawValue) tensor \(name)")
            }
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count {
                switch info.dtype {
                case .bf16:
                    let u = UInt16(littleEndian: base.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    out[i] = Float(bitPattern: UInt32(u) << 16)
                case .f16:
                    let u = UInt16(littleEndian: base.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    out[i] = Float(Float16(bitPattern: u))
                case .f32:
                    let u = UInt32(littleEndian: base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self))
                    out[i] = Float(bitPattern: u)
                default: break
                }
            }
            return out
        }
    }
}

extension QWBF16 {
    /// Alignment-safe unaligned MLX `.u32` bit-unpacker shared by `QWSafeTensors` and `QWHQ4Quantizer`.
    @inline(__always)
    public static func unpackMLXAffineQ(rowRaw: UnsafeRawPointer, col: Int, bits: Int) -> Float {
        switch bits {
        case 2, 4, 8:
            let valsPerWord = 32 / bits, mask: UInt32 = (1 << bits) - 1
            let word = UInt32(littleEndian: rowRaw.loadUnaligned(fromByteOffset: (col / valsPerWord) * 4, as: UInt32.self))
            return Float((word >> ((col % valsPerWord) * bits)) & mask)
        case 3, 6:
            let off = (bits == 3 ? (col >> 3) : (col >> 2)) * 3
            let b0 = UInt32(rowRaw.loadUnaligned(fromByteOffset: off, as: UInt8.self))
            let b1 = UInt32(rowRaw.loadUnaligned(fromByteOffset: off + 1, as: UInt8.self))
            let b2 = UInt32(rowRaw.loadUnaligned(fromByteOffset: off + 2, as: UInt8.self))
            let shift = bits == 3 ? ((col & 7) * 3) : ((col & 3) * 6)
            return Float(((b0 | (b1 << 8) | (b2 << 16)) >> shift) & UInt32((1 << bits) - 1))
        case 5:
            let off = (col >> 3) * 5
            var u40: UInt64 = 0
            for i in 0..<5 { u40 |= UInt64(rowRaw.loadUnaligned(fromByteOffset: off + i, as: UInt8.self)) << (8 * i) }
            return Float((u40 >> ((col & 7) * 5)) & 0x1F)
        default:
            return 0
        }
    }
}

/// Reads one tensor's bytes without copying them into the heap. The shard
/// and the byte range are resolved once, and the payload stays mapped: a
/// read is a memory read, not a syscall.
public final class QWSafeTensorsReader: @unchecked Sendable {
    private let shard: QWSafeTensorsShard
    private let tensor: QWSafeTensorsTensor

    init(shard: QWSafeTensorsShard, tensor: QWSafeTensorsTensor) {
        self.shard = shard
        self.tensor = tensor
    }

    /// The tensor this reader is bound to.
    public var tensorInfo: QWSafeTensorsTensor { tensor }

    /// The file the tensor lives in.
    public var shardURL: URL { shard.url }

    /// The tensor's bytes. The pointer is valid only inside `body`; it must
    /// not outlive the call.
    public func withBytes<T>(_ body: (UnsafeRawBufferPointer) throws -> T) rethrows -> T {
        try shard.data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let start = shard.dataOffset + tensor.byteOffset
            return try body(UnsafeRawBufferPointer(rebasing: raw[start..<start + tensor.byteLength]))
        }
    }
}

extension QWSafeTensorsShard {
    /// The spec's own limit on a header, in bytes.
    static let maxHeaderBytes = 100_000_000

    struct Entry {
        let name: String
        let dtype: QWSafeTensorsDType
        let shape: [Int]
        let start: Int
        let end: Int
    }

    static func open(_ url: URL) throws -> QWSafeTensorsShard {
        let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        defer { Darwin.close(fd) }
        var st = Darwin.stat()
        guard Darwin.fstat(fd, &st) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        guard st.st_size >= 8, let fileSize = Int(exactly: st.st_size) else {
            throw QWSafeTensorsError.badHeader("file too short")
        }
        let rawMap = Darwin.mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0)
        guard let mappedPtr = rawMap, mappedPtr != MAP_FAILED else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSFilePathErrorKey: url.path])
        }
        let data = Data(bytesNoCopy: mappedPtr, count: fileSize, deallocator: .custom { ptr, len in
            if len > 0 { _ = Darwin.munmap(ptr, len) }
        })
        let headerLength = data.prefix(8).withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        // The header length comes from the file, so it is untrusted: every
        // conversion of it must be checked, or a corrupt file traps the
        // process instead of throwing.
        guard headerLength <= UInt64(maxHeaderBytes) else {
            throw QWSafeTensorsError.badHeader("header is larger than the \(maxHeaderBytes)-byte limit")
        }
        guard let headerLength = Int(exactly: headerLength) else {
            throw QWSafeTensorsError.badHeader("header length does not fit")
        }
        let (headerEnd, ovHdr) = headerLength.addingReportingOverflow(8)
        guard !ovHdr, headerEnd <= data.count else { throw QWSafeTensorsError.badHeader("header length past EOF") }

        let headerData = data.subdata(in: 8..<headerEnd)
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw QWSafeTensorsError.badHeader("header is not valid JSON")
        }
        guard let header = parsed as? [String: Any] else {
            throw QWSafeTensorsError.badHeader("header is not a JSON object")
        }

        var entries: [Entry] = []
        for (name, value) in header {
            guard name != "__metadata__" else { continue }
            guard let v = value as? [String: Any] else {
                throw QWSafeTensorsError.badHeader("malformed tensor entry \(name)")
            }
            guard let dtypeRaw = v["dtype"] as? String else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no dtype")
            }
            guard let dtype = QWSafeTensorsDType(rawValue: dtypeRaw) else {
                throw QWSafeTensorsError.unsupportedDType(dtypeRaw)
            }
            guard let shape = v["shape"] as? [Int] else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no shape")
            }
            guard let offsets = v["data_offsets"] as? [Int], offsets.count == 2 else {
                throw QWSafeTensorsError.badHeader("tensor \(name) has no data_offsets")
            }
            entries.append(Entry(name: name, dtype: dtype, shape: shape,
                                 start: offsets[0], end: offsets[1]))
        }

        // The tensors must tile the data region: ordered, contiguous, from
        // zero, and ending exactly at the end of the file. A header that
        // overlaps or leaves a hole is a silent misread waiting to happen.
        entries.sort { ($0.start, $0.name) < ($1.start, $1.name) }
        var running = 0
        for e in entries {
            guard e.start >= 0, e.end >= e.start else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) has a negative or inverted range")
            }
            guard e.start == running else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(e.name) starts at \(e.start) but the previous tensor ends at \(running)")
            }
            var elements = 1
            for dim in e.shape {
                guard dim >= 0 else {
                    throw QWSafeTensorsError.badHeader("tensor \(e.name) has a negative dimension")
                }
                let (m, over) = elements.multipliedReportingOverflow(by: dim)
                guard !over else {
                    throw QWSafeTensorsError.badHeader("tensor \(e.name) has a shape that overflows")
                }
                elements = m
            }
            let (bits, over) = elements.multipliedReportingOverflow(by: e.dtype.bitSize)
            guard !over else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) overflows its byte size")
            }
            guard bits % 8 == 0 else {
                throw QWSafeTensorsError.badHeader("tensor \(e.name) is not byte-aligned")
            }
            let size = bits / 8
            guard e.end - e.start == size else {
                throw QWSafeTensorsError.badHeader(
                    "tensor \(e.name) holds \(e.end - e.start) bytes but its shape and dtype need \(size)")
            }
            running = e.end
        }
        let dataRegion = data.count - headerEnd
        guard running == dataRegion else {
            throw QWSafeTensorsError.badHeader(
                "the tensors cover \(running) bytes but the data region holds \(dataRegion)")
        }

        var tensors: [String: QWSafeTensorsTensor] = [:]
        for e in entries {
            tensors[e.name] = QWSafeTensorsTensor(name: e.name, dtype: e.dtype, shape: e.shape,
                                                   byteOffset: e.start, byteLength: e.end - e.start)
        }
        return QWSafeTensorsShard(url: url, data: data, dataOffset: headerEnd, tensors: tensors)
    }
}
