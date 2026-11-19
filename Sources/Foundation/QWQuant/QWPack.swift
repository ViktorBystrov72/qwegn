// Host-side QWP1 pack container reader and reference HQ4_G32_M5 dequantizer.
// Supports 2D HQ4_G32_M5 planes and verbatim BF16 planes for Qwen3.8-27B.

import CryptoKit
import Darwin
import Foundation
import QWCore

public enum QWQuant {
    public static let magic = "QWP1"
    public static let group = QWHQ4Layout.group
    public static let blockBytes = QWHQ4Layout.blockBytes
    public static let invSqrt32: Float = Float(bitPattern: 0x3E3504F3)

    /// Stable v1 geometry digest of the C01 numbers a 27B control pack must match.
    public static var geometryHash: String {
        sha256Hex(geometryHashString)
    }

    public static var geometryHashString: String {
        QWModelDescriptor.qwen38_27b.legacyGeoV1String
    }

    struct Header: Decodable {
        struct Prov: Decodable {
            let geometry_hash: String?
            let source: String?
            let sha256_source: String?
            let model_id: String?
            let geometry_digest: String?
        }
        let provenance: Prov
        let tensors: [String: QWQTensorEntry]
    }

    /// Mmap-backed QWP1 pack reader.
    public final class Pack {
        public let url: URL
        let header: Header
        let rawBase: UnsafeMutableRawPointer
        let rawCount: Int
        public let descriptor: QWModelDescriptor
        public private(set) var wiredBytes = 0

        public convenience init(url: URL) throws {
            let loc = try QWModelLocation(url: url)
            let cfgURL = loc.assets.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfgURL.path) else {
                throw QWModelRootError.missingConfig(cfgURL.path)
            }
            let desc = try QWModelDescriptor.decode(configJSON: Data(contentsOf: cfgURL))
            try self.init(url: url, descriptor: desc)
        }

        private static func mapFileShared(url: URL) throws -> (UnsafeMutableRawPointer, Int) {
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw QWQuantError.modelFileMissing(url.path) }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw QWQuantError.modelFileMissing(url.path) }
            guard st.st_size > 12, let size = Int(exactly: st.st_size) else {
                throw QWQuantError.truncated
            }
            let raw = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
            guard let ptr = raw, ptr != MAP_FAILED else {
                throw QWQuantError.modelFileMissing(url.path)
            }
            return (ptr, size)
        }

        deinit {
            if rawCount > 0 {
                _ = munmap(rawBase, rawCount)
            }
        }

        /// Opens a pack and validates its v1/v2 geometry identity against `descriptor`.
        public init(url: URL, descriptor: QWModelDescriptor, skipIdentityCheck: Bool = false) throws {
            let url = try QWModelLocation(url: url).weights
            let (base, count) = try Self.mapFileShared(url: url)
            var mappedOwned = true
            defer {
                if mappedOwned, count > 0 {
                    _ = munmap(base, count)
                }
            }
            guard count >= 12 else { throw QWQuantError.truncated }
            let rawBuf = UnsafeRawBufferPointer(start: base, count: count)
            guard String(decoding: rawBuf[0..<4], as: UTF8.self) == QWQuant.magic else {
                throw QWQuantError.badMagic
            }
            let hdrLenU64 = UInt64(littleEndian: rawBuf.loadUnaligned(fromByteOffset: 4, as: UInt64.self))
            guard let hdrLen = Int(exactly: hdrLenU64), hdrLen <= count - 12 else {
                throw QWQuantError.truncated
            }
            let hdrStart = count - hdrLen
            let hdrData = Data(bytes: base.advanced(by: hdrStart), count: hdrLen)
            let header = try JSONDecoder().decode(Header.self, from: hdrData)
            if !skipIdentityCheck {
                if let mid = header.provenance.model_id {
                    let midFamily = QWModelDescriptor.Family(rawValue: mid)
                        ?? QWModelFamilyRegistry.shared.spec(forModelType: mid)?.family
                    let lowerMid = mid.lowercased()
                    let isDrafterMid = descriptor.role == .speculativeDrafter
                        && (lowerMid.hasPrefix("dflash") || lowerMid.hasPrefix("dspine"))
                    let familyMatches = midFamily == descriptor.family
                        || isDrafterMid
                        || (midFamily == .qwen3MoE && descriptor.family == .qwen38MoE && (descriptor.numExperts ?? 0) > 0)
                        || (midFamily == .qwen38Dense && [.qwen38MoE, .senseNovaLoop, .glm5Next, .ernie55MoE, .maxwellMoE].contains(descriptor.family))
                    guard familyMatches else {
                        throw QWQuantError.modelMismatch(pack: mid, expected: descriptor.family.rawValue)
                    }
                    if let dg = header.provenance.geometry_digest, !dg.isEmpty {
                        let matchesDigest = dg == descriptor.geometryDigestV2Hex || dg == descriptor.geometryDigestV3Hex
                        guard matchesDigest else {
                            throw QWQuantError.geometryMismatch(pack: dg, expected: descriptor.geometryDigestV3Hex)
                        }
                    } else if descriptor.role == .speculativeDrafter, header.provenance.geometry_hash != nil {
                        // Drafter packs prior to geometry_digest carry geometry_hash only.
                    } else {
                        throw QWQuantError.geometryMissing(mid)
                    }
                } else {
                    guard descriptor.family == .qwen38Dense else {
                        throw QWQuantError.modelMismatch(pack: "legacy-v1", expected: descriptor.family.rawValue)
                    }
                    let expectedLegacy = sha256Hex(descriptor.legacyGeoV1String)
                    if let g = header.provenance.geometry_hash, g != expectedLegacy {
                        throw QWQuantError.geometryMismatch(pack: g, expected: expectedLegacy)
                    }
                }
            }
            self.url = url
            self.rawBase = base
            self.rawCount = count
            self.header = header
            self.descriptor = descriptor
            mappedOwned = false
        }

        public func entry(_ name: String) throws -> QWQTensorEntry {
            guard let e = header.tensors[name] else { throw QWQuantError.tensorMissing(name) }
            return e
        }

        public var tensorNames: [String] { header.tensors.keys.sorted() }
        public var tensors: [String: QWQTensorEntry] { header.tensors }

        public var provenance: (geometryHash: String?, source: String?, sha256Source: String?) {
            (header.provenance.geometry_hash, header.provenance.source, header.provenance.sha256_source)
        }

        /// Reads only the trailer provenance without binding a model descriptor.
        public static func peekProvenance(url: URL) throws -> (
            modelId: String?, geometryHash: String?, geometryDigest: String?,
            source: String?, sha256Source: String?, tensorCount: Int
        ) {
            let url = try QWModelLocation(url: url).weights
            let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw QWQuantError.modelFileMissing(url.path) }
            defer { Darwin.close(fd) }
            var st = Darwin.stat()
            guard Darwin.fstat(fd, &st) == 0, let fileSize = UInt64(exactly: st.st_size) else {
                throw QWQuantError.modelFileMissing(url.path)
            }
            guard fileSize >= 12 else { throw QWQuantError.truncated }
            var lenData = [UInt8](repeating: 0, count: 12)
            do {
                try lenData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    guard let base = raw.baseAddress else { throw QWQuantError.truncated }
                    try QWPOSIX.preadExact(fd: fd, into: base, count: 12, offset: 0)
                }
            } catch {
                throw QWQuantError.truncated
            }
            guard String(decoding: lenData[0..<4], as: UTF8.self) == magic else {
                throw QWQuantError.badMagic
            }
            let hdrLen = lenData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt64 in
                UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: 4, as: UInt64.self))
            }
            let (minRequired, ov) = hdrLen.addingReportingOverflow(12)
            guard !ov, fileSize >= minRequired,
                  let hdrCount = Int(exactly: hdrLen),
                  let hdrOffset = Int64(exactly: fileSize - hdrLen) else {
                throw QWQuantError.truncated
            }
            var head = Data(count: hdrCount)
            if hdrCount > 0 {
                do {
                    try head.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                        guard let base = raw.baseAddress else { throw QWQuantError.truncated }
                        try QWPOSIX.preadExact(fd: fd, into: base, count: hdrCount, offset: hdrOffset)
                    }
                } catch {
                    throw QWQuantError.truncated
                }
            }
            let h = try JSONDecoder().decode(Header.self, from: head)
            return (h.provenance.model_id, h.provenance.geometry_hash,
                    h.provenance.geometry_digest, h.provenance.source, h.provenance.sha256_source,
                    h.tensors.count)
        }

        @inline(__always)
        private static func dequantHQ4Row(
            base: UnsafeRawPointer, dstBase: UnsafeMutablePointer<Float>,
            row r: Int, groups g: Int, kCols: Int
        ) {
            for gg in 0..<g {
                let blk = base.advanced(by: (r * g + gg) * QWQuant.blockBytes)
                let dst = dstBase.advanced(by: r * kCols + gg * 32)
                QWQuant.dequantHQ4Group(blk, into: dst)
                QWQuant.fwht32OrthoBlock(dst)
            }
        }

        @inline(__always)
        public static func dequantHQ4Slab(
            base: UnsafeRawPointer, nRows: Int, kCols: Int
        ) -> [Float] {
            guard nRows > 0, kCols > 0, kCols % QWQuant.group == 0 else { return [] }
            let (total, ov) = nRows.multipliedReportingOverflow(by: kCols)
            guard !ov, total > 0 else { return [] }
            let g = kCols / QWQuant.group
            var out = [Float](repeating: 0, count: total)
            out.withUnsafeMutableBufferPointer { outBuf in
                guard let dstBase = outBuf.baseAddress else { return }
                if nRows >= 64 {
                    let rawAddr = Int(bitPattern: base)
                    let dstAddr = Int(bitPattern: dstBase)
                    let workers = min(nRows, 16)
                    let per = (nRows + workers - 1) / workers
                    DispatchQueue.concurrentPerform(iterations: workers) { w in
                        let start = w * per
                        let end = min(nRows, start + per)
                        guard start < end else { return }
                        let wBase = UnsafeRawPointer(bitPattern: rawAddr)!
                        let wDst = UnsafeMutablePointer<Float>(bitPattern: dstAddr)!
                        for r in start..<end {
                            dequantHQ4Row(base: wBase, dstBase: wDst, row: r, groups: g, kCols: kCols)
                        }
                    }
                } else {
                    for r in 0..<nRows {
                        dequantHQ4Row(base: base, dstBase: dstBase, row: r, groups: g, kCols: kCols)
                    }
                }
            }
            return out
        }

        /// Dequantizes a 2D `HQ{2,3,4,6,8}_G{32,64,128,256}` or `SHVQ` slab in the rotated layout.
        public static func dequantHQBlockSlab(
            base: UnsafeRawPointer, nRows: Int, kCols: Int, bits: Int, group: Int,
            isSHVQ: Bool = false
        ) -> [Float] {
            guard nRows > 0, kCols > 0, group > 0, kCols % group == 0 else { return [] }
            let (total, ov) = nRows.multipliedReportingOverflow(by: kCols)
            guard !ov, total > 0 else { return [] }
            let g = kCols / group
            let blkBytes = isSHVQ
                ? QWSHVQBlock.bytes(bits: bits, group: group)
                : QWAQBlock.bytes(bits: bits, group: group)
            var out = [Float](repeating: 0, count: total)
            out.withUnsafeMutableBufferPointer { outBuf in
                guard let dstBase = outBuf.baseAddress else { return }
                var groupBuf = [Float]()
                groupBuf.reserveCapacity(group)
                for r in 0..<nRows {
                    for gg in 0..<g {
                        let blkPtr = base.advanced(by: (r * g + gg) * blkBytes)
                        let rawBuf = UnsafeRawBufferPointer(start: blkPtr, count: blkBytes)
                        if isSHVQ {
                            QWSHVQBlock.unpack(group: rawBuf, bits: bits, group: group, into: &groupBuf)
                        } else {
                            QWAQBlock.unpack(group: rawBuf, bits: bits, group: group, into: &groupBuf)
                        }
                        let rowDst = dstBase.advanced(by: r * kCols + gg * group)
                        for i in 0..<group {
                            rowDst[i] = groupBuf[i]
                        }
                        var off = 0
                        while off < group {
                            QWQuant.fwht32OrthoBlock(rowDst.advanced(by: off))
                            off += 32
                        }
                    }
                }
            }
            return out
        }

        /// Dequantizes a 2D `HQ4_G32_M5`, `HQ`, or `SHVQ` tensor into fp32 floats.
        public func dequantQ4g(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let hq4 = e.hq4Off else { throw QWQuantError.notQuantPlane(name) }
            guard let n = e.nRows, let g = e.groups, let K = e.kCols,
                  n >= 0, g >= 0, K >= 0, e.hqGroup > 0 else {
                throw QWQuantError.notQuantPlane(name)
            }
            if !e.isIncumbentHQ4 {
                let valid = e.isSHVQ
                    ? QWSHVQBlock.isSupported(bits: e.hqBits, group: e.hqGroup)
                    : QWAQBlock.isBlockSupported(bits: e.hqBits, group: e.hqGroup)
                guard valid else { throw QWQuantError.notQuantPlane(name) }
            }
            let blkBytes = e.hqBlockBytes
            let (ng, ov0) = n.multipliedReportingOverflow(by: g)
            let (byteCount, ov1) = ng.multipliedReportingOverflow(by: blkBytes)
            let (endOff, ov2) = hq4.addingReportingOverflow(byteCount)
            guard hq4 >= 0, K % e.hqGroup == 0, !ov0, !ov1, !ov2, endOff <= rawCount else {
                throw QWQuantError.truncated
            }
            let ptr = UnsafeRawPointer(rawBase).advanced(by: hq4)
            if e.isIncumbentHQ4 {
                return Self.dequantHQ4Slab(base: ptr, nRows: n, kCols: K)
            }
            return Self.dequantHQBlockSlab(base: ptr, nRows: n, kCols: K,
                                           bits: e.hqBits, group: e.hqGroup, isSHVQ: e.isSHVQ)
        }


        /// HQ8_G32_M5 rot-domain dequant: `float(q) * float16(scale)` per
        /// group-32, WITHOUT the inverse FWHT (kernel-comparison domain).
        public func dequantHQ8Rot(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let p = e.hq8Off, let s = e.hq8SOff,
                  let ext = e.hq8Extents, let n = e.nRows,
                  let K = e.kCols, n >= 0, K >= 0 else { throw QWQuantError.notQuantPlane(name) }
            let g = K / QWQuant.group
            let (pEnd, ov0) = p.addingReportingOverflow(ext.payload)
            let (sEnd, ov1) = s.addingReportingOverflow(ext.scale)
            guard p >= 0, s >= 0, !ov0, !ov1,
                  pEnd <= rawCount, sEnd <= rawCount else { throw QWQuantError.truncated }
            var out = [Float](repeating: 0, count: n * K)
            let qPtr = UnsafeRawPointer(rawBase).advanced(by: p)
            let scPtr = UnsafeRawPointer(rawBase).advanced(by: s)
            for r in 0..<n {
                for gg in 0..<g {
                    let sBits = UInt16(littleEndian: scPtr.loadUnaligned(fromByteOffset: (r * g + gg) * 2, as: UInt16.self))
                    let scale = QWQuant.f16tof32Bits(sBits)
                    let row = r * K + gg * QWQuant.group
                    for j in 0..<QWQuant.group {
                        let qVal = qPtr.loadUnaligned(fromByteOffset: row + j, as: Int8.self)
                        out[row + j] = Float(qVal) * scale
                    }
                }
            }
            return out
        }

        /// HQ8_G32_M5 standard-domain dequant with inverse 32-point orthonormal FWHT.
        public func dequantHQ8(_ name: String) throws -> [Float] {
            var out = try dequantHQ8Rot(name)
            let count = out.count
            out.withUnsafeMutableBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                var off = 0
                while off + 32 <= count {
                    QWQuant.fwht32OrthoBlock(base.advanced(by: off))
                    off += 32
                }
            }
            return out
        }

        /// Legacy `q8_0` escape hatch (pre-Q8-removal packs only).
        public func dequantQ8(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let q8 = e.q8Off, let q8s = e.q8SOff else { throw QWQuantError.notQuantPlane(name) }
            guard let n = e.nRows, let K = e.kCols, n >= 0, K >= 0 else {
                throw QWQuantError.notQuantPlane(name)
            }
            let grp = e.hqGroup
            guard grp > 0, K % grp == 0 else {
                throw QWQuantError.notQuantPlane(name)
            }
            let g = K / grp
            let (elemCount, ov0) = n.multipliedReportingOverflow(by: K)
            let (qEnd, ov1) = q8.addingReportingOverflow(elemCount)
            let (ng, ov2) = n.multipliedReportingOverflow(by: g)
            let (sBytes, ov3) = ng.multipliedReportingOverflow(by: 2)
            let (sEnd, ov4) = q8s.addingReportingOverflow(sBytes)
            guard q8 >= 0, q8s >= 0, !ov0, !ov1, !ov2, !ov3, !ov4,
                  qEnd <= rawCount, sEnd <= rawCount else {
                throw QWQuantError.truncated
            }
            var out = [Float](repeating: 0, count: elemCount)
            let qPtr = UnsafeRawPointer(rawBase).advanced(by: q8)
            let scPtr = UnsafeRawPointer(rawBase).advanced(by: q8s)
            for r in 0..<n {
                for gg in 0..<g {
                    let sBits = UInt16(littleEndian: scPtr.loadUnaligned(fromByteOffset: (r * g + gg) * 2, as: UInt16.self))
                    let s = QWQuant.f16tof32Bits(sBits)
                    let row = r * K + gg * grp
                    for j in 0..<grp {
                        let qVal = qPtr.loadUnaligned(fromByteOffset: row + j, as: Int8.self)
                        out[row + j] = Float(qVal) * s
                    }
                }
            }
            return out
        }

        /// Widens a verbatim BF16 or F32 raw plane into `[Float]`.
        public func rawBf16(_ name: String) throws -> [Float] {
            let e = try entry(name)
            guard let off = e.rawOff else { throw QWQuantError.notRawPlane(name) }
            guard off >= 0 else { throw QWQuantError.truncated }
            var count = 1
            for dim in e.shape {
                guard dim >= 0 else { throw QWQuantError.truncated }
                let (m, ov) = count.multipliedReportingOverflow(by: dim)
                guard !ov else { throw QWQuantError.truncated }
                count = m
            }
            let elemBytes = e.kind == "f32" ? 4 : 2
            let (byteCount, ov0) = count.multipliedReportingOverflow(by: elemBytes)
            let (endOff, ov1) = off.addingReportingOverflow(byteCount)
            guard !ov0, !ov1, endOff <= rawCount else { throw QWQuantError.truncated }
            if count == 0 { return [] }
            var out = [Float](repeating: 0, count: count)
            let srcPtr = UnsafeRawPointer(rawBase).advanced(by: off)
            if e.kind == "f32" {
                out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
                    if let base = dst.baseAddress {
                        base.copyMemory(from: srcPtr, byteCount: byteCount)
                    }
                }
                return out
            }
            for i in 0..<count {
                let bits = UInt16(littleEndian: srcPtr.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                out[i] = Float(bitPattern: UInt32(bits) << 16)
            }
            return out
        }

        @inline(__always)
        public func withRawPointer<R>(at byteOffset: Int, byteCount: Int,
                                      _ body: (UnsafeRawPointer) throws -> R) throws -> R {
            let (endOff, ov) = byteOffset.addingReportingOverflow(byteCount)
            guard byteOffset >= 0, byteCount >= 0, !ov, endOff <= rawCount else {
                throw QWQuantError.truncated
            }
            return try body(UnsafeRawPointer(rawBase).advanced(by: byteOffset))
        }

        public func adviseDontNeed() {
            if rawCount > 0 {
                _ = madvise(rawBase, rawCount, MADV_DONTNEED)
            }
        }

        /// C30-T6: asynchronous kernel readahead over the pack mmap. Advances in
        /// Pins up to `limitBytes` of the pack mmap into RAM in 1 GiB chunks via `mlock`.
        @discardableResult
        public func wire(limitBytes: Int = .max) -> Int {
            guard wiredBytes == 0 else { return wiredBytes }
            var cap = Swift.max(0, limitBytes)
            var rl = rlimit()
            if getrlimit(RLIMIT_MEMLOCK, &rl) == 0, rl.rlim_cur < UInt64(cap) {
                cap = Int(rl.rlim_cur)
            }
            let gran = 1 << 30
            var wired = 0
            let want = Swift.min(cap, rawCount)
            if want > 0 {
                _ = madvise(rawBase, want, MADV_WILLNEED)
            }
            while wired < want {
                let chunk = Swift.min(gran, want - wired)
                if mlock(UnsafeRawPointer(rawBase).advanced(by: wired), chunk) != 0 { break }
                wired += chunk
            }
            wiredBytes = wired
            return wired
        }
    }

}
