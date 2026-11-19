import XCTest
import Foundation
@testable import QWHAL

final class QWDispatchIOTests: XCTestCase {

    private func createTempFile(size: Int) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("qw_dispatch_io_\(UUID().uuidString).bin")
        var buffer = [UInt8](repeating: 0, count: size)
        for i in 0..<size {
            buffer[i] = UInt8(i & 0xFF)
        }
        let data = Data(buffer)
        try data.write(to: fileURL)
        return fileURL
    }

    func testDispatchIODirectIOPageAligned() throws {
        let pageSize = QWDispatchIO.pageSize // 16 KB on Apple Silicon
        let totalSize = pageSize * 4
        let fileURL = try createTempFile(size: totalSize)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let io = try QWDispatchIO(path: fileURL.path, mode: .readOnly, directIO: true)
        defer { io.close() }

        var rawPtr: UnsafeMutableRawPointer? = nil
        let status = posix_memalign(&rawPtr, pageSize, pageSize)
        XCTAssertEqual(status, 0)
        guard let dst = rawPtr else { return XCTFail("posix_memalign failed") }
        defer { free(dst) }

        let bytesRead = try io.readSync(offset: Int64(pageSize), into: dst, length: pageSize)
        XCTAssertEqual(bytesRead, pageSize)

        let typed = dst.assumingMemoryBound(to: UInt8.self)
        for i in 0..<pageSize {
            let expected = UInt8((pageSize + i) & 0xFF)
            XCTAssertEqual(typed[i], expected, "Direct I/O byte mismatch at \(i)")
        }
    }

    func testDispatchIOSyncRoundtrip() throws {
        let size = 16384
        let fileURL = try createTempFile(size: size)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let io = try QWDispatchIO(path: fileURL.path, mode: .readWrite, directIO: false)
        defer { io.close() }

        var buf = [UInt8](repeating: 0, count: 512)
        let readBytes = try buf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            try io.readSync(offset: 1024, into: raw.baseAddress!, length: 512)
        }
        XCTAssertEqual(readBytes, 512)
        for i in 0..<512 {
            XCTAssertEqual(buf[i], UInt8((1024 + i) & 0xFF))
        }

        let newBuf = [UInt8](repeating: 0x55, count: 256)
        try newBuf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            try io.writeSync(offset: 4096, buffer: raw.baseAddress!, length: 256)
        }

        var verifyBuf = [UInt8](repeating: 0, count: 256)
        _ = try verifyBuf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            try io.readSync(offset: 4096, into: raw.baseAddress!, length: 256)
        }
        for i in 0..<256 {
            XCTAssertEqual(verifyBuf[i], 0x55)
        }
    }

    func testDispatchIOErrorContracts() throws {
        let nonExistentPath = "/tmp/qw_non_existent_\(UUID().uuidString).bin"
        XCTAssertThrowsError(try QWDispatchIO(path: nonExistentPath, mode: .readOnly)) { error in
            guard case QWDispatchIOError.openFailed = error else {
                return XCTFail("Expected openFailed, got \(error)")
            }
        }

        let tempFile = try createTempFile(size: 1024)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let io = try QWDispatchIO(path: tempFile.path, mode: .readOnly)
        io.close()

        var buf = [UInt8](repeating: 0, count: 128)
        XCTAssertThrowsError(try buf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            _ = try io.readSync(offset: 0, into: raw.baseAddress!, length: 128)
        }) { error in
            guard case QWDispatchIOError.channelClosed = error else {
                return XCTFail("Expected channelClosed, got \(error)")
            }
        }
    }

    func testDispatchIOCLOEXECReadExactAndWriteValidation() throws {
        let tempFile = try createTempFile(size: 512)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let io = try QWDispatchIO(path: tempFile.path, mode: .readWrite, directIO: false)
        defer { io.close() }

        let flags = fcntl(io.fd, F_GETFD)
        XCTAssertNotEqual(flags, -1)
        XCTAssertNotEqual(flags & FD_CLOEXEC, 0, "QWDispatchIO must open FD with O_CLOEXEC")

        let bufPtr = UnsafeMutableRawPointer.allocate(byteCount: 512, alignment: 1)
        bufPtr.initializeMemory(as: UInt8.self, repeating: 0, count: 512)
        defer { bufPtr.deallocate() }
        try io.readExactSync(offset: 0, into: bufPtr, length: 256)
        XCTAssertThrowsError(try io.readExactSync(offset: 400, into: bufPtr, length: 256)) { error in
            guard case QWDispatchIOError.shortRead(let exp, let act) = error else {
                return XCTFail("Expected shortRead, got \(error)")
            }
            XCTAssertEqual(exp, 256)
            XCTAssertEqual(act, 112)
        }

        let payloadPtr = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 1)
        payloadPtr.initializeMemory(as: UInt8.self, repeating: 0x11, count: 8)
        defer { payloadPtr.deallocate() }
        let rawPayload = UnsafeRawPointer(payloadPtr)
        XCTAssertThrowsError(try io.writeSync(offset: -1, buffer: rawPayload, length: 8))
        XCTAssertThrowsError(try io.writeSync(offset: 0, buffer: rawPayload, length: -1))
        XCTAssertThrowsError(try io.writeSync(offset: Int64.max - 2, buffer: rawPayload, length: 8))
    }

    func testDispatchIOConcurrentReadAndCloseHasNoFDReuseTOCTOU() throws {
        final class ErrorBox: @unchecked Sendable {
            private let lock = NSLock()
            private var stored: (any Error)?
            func record(_ error: any Error) {
                lock.withLock {
                    if stored == nil { stored = error }
                }
            }
            func get() -> (any Error)? {
                lock.withLock { stored }
            }
        }

        let size = 32 * 1024
        let tempFile = try createTempFile(size: size)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let io = try QWDispatchIO(path: tempFile.path, mode: .readOnly, directIO: false)
        let group = DispatchGroup()
        let errorBox = ErrorBox()

        for _ in 0..<8 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                var local = [UInt8](repeating: 0, count: 4096)
                for iter in 0..<20 {
                    do {
                        try local.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                            try io.readExactSync(offset: Int64((iter % 4) * 4096), into: raw.baseAddress!, length: 4096)
                        }
                    } catch QWDispatchIOError.channelClosed {
                        break
                    } catch {
                        errorBox.record(error)
                        break
                    }
                }
            }
        }
        io.close()
        group.wait()
        XCTAssertNil(errorBox.get())
    }
}
