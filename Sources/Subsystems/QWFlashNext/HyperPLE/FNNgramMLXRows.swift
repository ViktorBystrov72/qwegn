// FNNgramMLXRows.swift — MLX-Serve ngram_table.bin row reader.
import Darwin
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

/// PLE N-gram rows read on demand from an MLX-Serve `ngram_table.bin` safetensors file
/// (`weight` U32 `[rows, headDim/8]`, `scales` BF16/F16 `[rows, groups]`, `biases` BF16/F16 `[rows, groups]`),
/// using read-only `mmap` (`POSIX_MADV_RANDOM`, `pread` fallback) and SIMD8 4-bit nibble unpacking.
public final class FNNgramMLXRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let entryCount: Int
    private let wCols: Int
    private let gCols: Int
    private let groupSize: Int
    private let isF16: Bool
    private let fd: Int32
    private let fileSize: Int
    private let mappedPtr: UnsafeRawPointer?
    private let wOffset: Int
    private let sOffset: Int
    private let bOffset: Int

    public init(url: URL) throws {
        let opened = Darwin.open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw QWSafeTensorsError.badHeader("Cannot open \(url.path)")
        }
        var st = Darwin.stat()
        guard Darwin.fstat(opened, &st) == 0, st.st_size >= 8 else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Cannot stat or short file \(url.path)")
        }
        let fileSize = Int(st.st_size)
        var hdrLenLE: UInt64 = 0
        let gotLen = withUnsafeMutableBytes(of: &hdrLenLE) { (raw: UnsafeMutableRawBufferPointer) -> Int in
            guard let base = raw.baseAddress else { return -1 }
            return (try? QWPOSIX.preadUpTo(fd: opened, into: base, count: 8, offset: 0)) ?? -1
        }
        let hdrLen = Int(UInt64(littleEndian: hdrLenLE))
        guard gotLen == 8, hdrLen > 0, hdrLen <= 100_000_000, 8 + hdrLen <= fileSize else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Invalid safetensors header length in \(url.lastPathComponent)")
        }
        var headerData = Data(count: hdrLen)
        let gotHdr = headerData.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Int in
            guard let base = raw.baseAddress else { return -1 }
            return (try? QWPOSIX.preadUpTo(fd: opened, into: base, count: hdrLen, offset: 8)) ?? -1
        }
        guard gotHdr == hdrLen,
              let header = (try? JSONSerialization.jsonObject(with: headerData)) as? [String: Any] else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Invalid safetensors JSON header in \(url.lastPathComponent)")
        }
        func parseTensor(_ name: String) throws -> (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int) {
            guard let v = header[name] as? [String: Any],
                  let dtypeRaw = v["dtype"] as? String,
                  let dtype = QWSafeTensorsDType(rawValue: dtypeRaw),
                  let shape = v["shape"] as? [Int],
                  let offsets = v["data_offsets"] as? [Int], offsets.count == 2 else {
                throw QWSafeTensorsError.badHeader("Missing or malformed tensor \(name) in \(url.lastPathComponent)")
            }
            return (dtype, shape, offsets[0], offsets[1])
        }
        let wInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        let sInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        let bInfo: (dtype: QWSafeTensorsDType, shape: [Int], start: Int, end: Int)
        do {
            wInfo = try parseTensor("weight")
            sInfo = try parseTensor("scales")
            bInfo = try parseTensor("biases")
        } catch {
            Darwin.close(opened)
            throw error
        }
        let dataBase = 8 + hdrLen
        guard wInfo.dtype == .u32, wInfo.shape.count == 2,
              (sInfo.dtype == .bf16 || sInfo.dtype == .f16), sInfo.shape.count == 2,
              (bInfo.dtype == .bf16 || bInfo.dtype == .f16), bInfo.shape.count == 2,
              wInfo.shape[0] == sInfo.shape[0], wInfo.shape[0] == bInfo.shape[0],
              sInfo.shape[1] == bInfo.shape[1],
              dataBase + wInfo.end <= fileSize,
              dataBase + sInfo.end <= fileSize,
              dataBase + bInfo.end <= fileSize else {
            Darwin.close(opened)
            throw QWSafeTensorsError.badHeader("Unexpected tensor shapes/dtypes in \(url.lastPathComponent)")
        }
        self.entryCount = wInfo.shape[0]
        self.wCols = wInfo.shape[1]
        self.gCols = sInfo.shape[1]
        self.headDim = wCols * 8
        self.groupSize = (wCols * 8) / gCols
        self.isF16 = (sInfo.dtype == .f16)
        self.fd = opened
        self.fileSize = fileSize
        self.wOffset = dataBase + wInfo.start
        self.sOffset = dataBase + sInfo.start
        self.bOffset = dataBase + bInfo.start
        let mapped = Darwin.mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, opened, 0)
        if let mapped, mapped != MAP_FAILED {
            _ = Darwin.posix_madvise(mapped, fileSize, POSIX_MADV_RANDOM)
            self.mappedPtr = UnsafeRawPointer(mapped)
        } else {
            self.mappedPtr = nil
        }
    }

    deinit {
        if let mappedPtr {
            Darwin.munmap(UnsafeMutableRawPointer(mutating: mappedPtr), fileSize)
        }
        if fd >= 0 { Darwin.close(fd) }
    }

    @inline(__always)
    private static func decodeGroups(wPtr: UnsafeRawPointer, sPtr: UnsafeRawPointer, bPtr: UnsafeRawPointer,
                                     gCols: Int, wordsPerGroup: Int, groupSize: Int, isF16: Bool,
                                     into dst: UnsafeMutablePointer<Float>) {
        let shifts = SIMD8<UInt32>(0, 4, 8, 12, 16, 20, 24, 28)
        let mask = SIMD8<UInt32>(repeating: 0xF)
        for g in 0..<gCols {
            let sBits = sPtr.loadUnaligned(fromByteOffset: g * 2, as: UInt16.self).littleEndian
            let bBits = bPtr.loadUnaligned(fromByteOffset: g * 2, as: UInt16.self).littleEndian
            let scale = isF16 ? Float(Float16(bitPattern: sBits)) : Float(bitPattern: UInt32(sBits) << 16)
            let bias = isF16 ? Float(Float16(bitPattern: bBits)) : Float(bitPattern: UInt32(bBits) << 16)
            let scaleVec = SIMD8<Float>(repeating: scale)
            let biasVec = SIMD8<Float>(repeating: bias)
            let groupWordBase = g * wordsPerGroup
            let groupDstBase = g * groupSize
            for u in 0..<wordsPerGroup {
                let word = wPtr.loadUnaligned(fromByteOffset: (groupWordBase + u) * 4, as: UInt32.self).littleEndian
                let nibbles = (SIMD8<UInt32>(repeating: word) &>> shifts) & mask
                let outVec = SIMD8<Float>(nibbles) * scaleVec + biasVec
                let baseIdx = groupDstBase + u * 8
                dst[baseIdx + 0] = outVec[0]
                dst[baseIdx + 1] = outVec[1]
                dst[baseIdx + 2] = outVec[2]
                dst[baseIdx + 3] = outVec[3]
                dst[baseIdx + 4] = outVec[4]
                dst[baseIdx + 5] = outVec[5]
                dst[baseIdx + 6] = outVec[6]
                dst[baseIdx + 7] = outVec[7]
            }
        }
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) {
        precondition(id >= 0 && id < entryCount, "PLE row \(id) outside ngram_table.bin [0..<\(entryCount)]")
        let wRowBytes = wCols * 4
        let sbRowBytes = gCols * 2
        let wByteOff = wOffset + id * wRowBytes
        let sByteOff = sOffset + id * sbRowBytes
        let bByteOff = bOffset + id * sbRowBytes
        let isF16 = self.isF16
        let gCols = self.gCols
        let groupSize = self.groupSize
        let wordsPerGroup = groupSize / 8
        if let mappedPtr {
            Self.decodeGroups(wPtr: mappedPtr.advanced(by: wByteOff),
                              sPtr: mappedPtr.advanced(by: sByteOff),
                              bPtr: mappedPtr.advanced(by: bByteOff),
                              gCols: gCols, wordsPerGroup: wordsPerGroup, groupSize: groupSize, isF16: isF16,
                              into: dst)
            return
        }
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: wRowBytes + 2 * sbRowBytes) { scratch in
            guard let base = scratch.baseAddress else { preconditionFailure("empty scratch buffer") }
            let wRaw = UnsafeMutableRawPointer(base)
            let sRaw = UnsafeMutableRawPointer(base.advanced(by: wRowBytes))
            let bRaw = UnsafeMutableRawPointer(base.advanced(by: wRowBytes + sbRowBytes))
            do {
                try QWPOSIX.preadExact(fd: fd, into: wRaw, count: wRowBytes, offset: Int64(wByteOff))
                try QWPOSIX.preadExact(fd: fd, into: sRaw, count: sbRowBytes, offset: Int64(sByteOff))
                try QWPOSIX.preadExact(fd: fd, into: bRaw, count: sbRowBytes, offset: Int64(bByteOff))
            } catch {
                preconditionFailure("PLE short pread on ngram_table.bin at row \(id): \(error)")
            }
            Self.decodeGroups(
                wPtr: UnsafeRawPointer(wRaw), sPtr: UnsafeRawPointer(sRaw), bPtr: UnsafeRawPointer(bRaw),
                gCols: gCols, wordsPerGroup: wordsPerGroup, groupSize: groupSize, isF16: isF16,
                into: dst
            )
        }
    }

    public func row(_ id: Int) -> [Float] {
        let n = headDim
        return [Float](unsafeUninitializedCapacity: n) { out, count in
            copyRow(id, to: out.baseAddress!)
            count = n
        }
    }
}
