// FNNgramBinRows.swift — Packed assets/ngram.bin row reader.
import Darwin
import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

/// PLE N-gram rows read directly from packed `assets/ngram.bin` (QNGRAM v1 BF16, v2 Q8/Q4_G32, or v3 VQ2_G32).
public final class FNNgramBinRows: FNNgramRows, @unchecked Sendable {
    public let headDim: Int
    public let entryCount: Int
    public let rowBytes: Int
    public let rowsPerSector: Int
    public let dataOffset: Int
    /// The row format this file carries.
    public let dtype: QWNgramDType
    /// The number of hot rows the writer recorded (0 when unknown).
    public let l1Hot: Int
    public let codebook: [Float]
    public let fileBytes: Int
    private let fd: Int32
    private let mappedPtr: UnsafeRawPointer?
    private let mappedBytes: Int

    public init(url: URL, mapped: Bool = false) throws {
        let opened = open(url.path, O_RDONLY)
        guard opened >= 0 else {
            throw NSError(domain: "FNNgramBinRows", code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open ngram.bin at \(url.path)"])
        }
        var header = [UInt8](repeating: 0, count: QWNgramFormat.headerBytes)
        let got = header.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Int in
            guard let base = raw.baseAddress else { return -1 }
            return (try? QWPOSIX.preadUpTo(
                fd: opened,
                into: base,
                count: QWNgramFormat.headerBytes,
                offset: 0
            )) ?? -1
        }
        guard got == QWNgramFormat.headerBytes else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Short header in ngram.bin at \(url.path)"])
        }
        let hdr: QWNgramHeader
        let rps: Int
        do {
            hdr = try QWNgramFormat.parse(header)
            rps = try QWNgramFormat.rowsPerSector(rowBytes: hdr.rowBytes)
        } catch {
            close(opened)
            throw error
        }
        // The file must hold the rows the header claims. The Python auditor
        // enforces the same law; a reader and an auditor for one format
        // must not disagree about how much of the file must be there.
        var st = stat()
        guard fstat(opened, &st) == 0 else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                            "Cannot stat ngram.bin at \(url.path)"])
        }
        let size = Int(st.st_size)
        let whole = (hdr.entryCount - 1) / rps
        let tailRows = hdr.entryCount - whole * rps
        let minSize = hdr.dataOffset + whole * QWNgramFormat.sectorSize
            + tailRows * hdr.rowBytes
        let maxSize = hdr.dataOffset + (whole + 1) * QWNgramFormat.sectorSize
        guard size >= minSize, size <= maxSize else {
            close(opened)
            throw NSError(domain: "FNNgramBinRows", code: 6,
                          userInfo: [NSLocalizedDescriptionKey:
                            "ngram.bin size \(size) does not hold its \(hdr.entryCount) rows "
                            + "(expected [\(minSize), \(maxSize)]) at \(url.path)"])
        }
        let ptr: UnsafeRawPointer?
        let mappedLen: Int
        if mapped {
            let page = max(1, Int(getpagesize()))
            let aligned = ((size + page - 1) / page) * page
            let result = Darwin.mmap(nil, aligned, PROT_READ, MAP_SHARED, opened, 0)
            guard let result, result != MAP_FAILED else {
                let error = errno
                close(opened)
                throw NSError(domain: "FNNgramBinRows", code: Int(error),
                              userInfo: [NSLocalizedDescriptionKey: "Failed to map ngram.bin at \(url.path)"])
            }
            ptr = UnsafeRawPointer(result)
            mappedLen = aligned
        } else {
            ptr = nil
            mappedLen = 0
        }
        self.dtype = hdr.dtype
        self.l1Hot = hdr.l1Hot
        self.codebook = hdr.codebook
        self.fileBytes = size
        self.entryCount = hdr.entryCount
        self.headDim = hdr.headDim
        self.rowBytes = hdr.rowBytes
        self.rowsPerSector = rps
        self.dataOffset = hdr.dataOffset
        self.fd = opened
        self.mappedPtr = ptr
        self.mappedBytes = mappedLen
    }

    deinit {
        if let mappedPtr, mappedBytes > 0 {
            Darwin.munmap(UnsafeMutableRawPointer(mutating: mappedPtr), mappedBytes)
        }
        if fd >= 0 { close(fd) }
    }

    /// Maps a v3 `vq2g32` or `vq4g32` table into a page-aligned `MTLBuffer` for zero-`pread` GPU gather.
    public func makeResidentGPUBuffer(device: any MTLDevice) -> (any MTLBuffer)? {
        guard dtype == .vq2g32 || dtype == .vq4g32 else { return nil }
        let page = max(1, Int(getpagesize()))
        let alignedLen = ((fileBytes + page - 1) / page) * page
        guard alignedLen > 0, alignedLen <= Int(device.maxBufferLength) else { return nil }
        let raw = Darwin.mmap(nil, alignedLen, PROT_READ, MAP_SHARED, fd, 0)
        if let basePtr = raw, basePtr != MAP_FAILED {
            if let buf = device.makeBuffer(
                bytesNoCopy: basePtr, length: alignedLen, options: .storageModeShared,
                deallocator: { ptr, len in _ = Darwin.munmap(ptr, len) }
            ) {
                QWResidencyRegistry.note(buf)
                return buf
            }
            _ = Darwin.munmap(basePtr, alignedLen)
        }
        guard fileBytes <= (64 << 20),
              let fallback = device.makeBuffer(length: alignedLen, options: .storageModeShared) else {
            return nil
        }
        memset(fallback.contents(), 0, alignedLen)
        guard (try? QWPOSIX.preadExact(fd: fd, into: fallback.contents(), count: fileBytes, offset: 0)) != nil else {
            return nil
        }
        QWResidencyRegistry.note(fallback)
        return fallback
    }

    public func copyRow(_ id: Int, to dst: UnsafeMutablePointer<Float>) throws {
        guard id >= 0 && id < entryCount else {
            throw FNNgramRowsError.rowOutOfBounds(id: id, count: entryCount)
        }
        let off = off_t(dataOffset + (id / rowsPerSector) * QWNgramFormat.sectorSize
                        + (id % rowsPerSector) * rowBytes)
        if let mappedPtr {
            if codebook.isEmpty {
                QWNgramFormat.decodeRow(dtype: dtype, from: mappedPtr.advanced(by: Int(off)),
                                        headDim: headDim, into: dst)
            } else {
                codebook.withUnsafeBufferPointer { cb in
                    QWNgramFormat.decodeRow(dtype: dtype, from: mappedPtr.advanced(by: Int(off)),
                                            headDim: headDim, codebook: cb.baseAddress, into: dst)
                }
            }
            return
        }
        if dtype == .bf16 {
            let n = headDim
            let u16 = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: UInt16.self).advanced(by: n)
            try QWPOSIX.preadExact(fd: fd, into: UnsafeMutableRawPointer(u16), count: rowBytes, offset: Int64(off))
            for i in 0..<n {
                dst[i] = Float(bitPattern: UInt32(u16[i]) << 16)
            }
            return
        }
        // The row bytes land in a scratch buffer of their own: the
        // caller's buffer holds the row's values and nothing else.
        try withUnsafeTemporaryAllocation(of: UInt8.self, capacity: rowBytes) { scratch in
            guard let base = scratch.baseAddress else { throw FNNgramRowsError.emptyScratchBuffer }
            try QWPOSIX.preadExact(fd: fd, into: UnsafeMutableRawPointer(base), count: rowBytes, offset: Int64(off))
            let rawPtr = UnsafeRawPointer(base)
            if codebook.isEmpty {
                QWNgramFormat.decodeRow(dtype: dtype, from: rawPtr,
                                        headDim: headDim, into: dst)
            } else {
                codebook.withUnsafeBufferPointer { cb in
                    QWNgramFormat.decodeRow(dtype: dtype, from: rawPtr,
                                            headDim: headDim, codebook: cb.baseAddress, into: dst)
                }
            }
        }
    }

    public func row(_ id: Int) -> [Float] {
        let n = headDim
        return [Float](unsafeUninitializedCapacity: n) { out, count in
            qwCopyNgramRowBestEffort(self, id: id, to: out.baseAddress!, headDim: n)
            count = n
        }
    }
}
