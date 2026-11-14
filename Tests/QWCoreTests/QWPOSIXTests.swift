// QWPOSIXTests.swift — Unit tests for checked POSIX pread helpers.

import Darwin
import Foundation
import XCTest
@testable import QWCore

final class QWPOSIXTests: XCTestCase {
    func testPreadUpToAndExactAtEOF() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwposix_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let payload = Data([10, 20, 30, 40, 50])
        try payload.write(to: tmp)

        let fd = Darwin.open(tmp.path, O_RDONLY)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { Darwin.close(fd) }

        var exactBuf = [UInt8](repeating: 0, count: 3)
        try exactBuf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let base = raw.baseAddress else { return XCTFail("missing baseAddress") }
            try QWPOSIX.preadExact(fd: fd, into: base, count: 3, offset: 1)
        }
        XCTAssertEqual(exactBuf, [20, 30, 40])

        var upToBuf = [UInt8](repeating: 0, count: 8)
        let got = try upToBuf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Int in
            guard let base = raw.baseAddress else { return -1 }
            return try QWPOSIX.preadUpTo(fd: fd, into: base, count: 8, offset: 3)
        }
        XCTAssertEqual(got, 2)
        XCTAssertEqual(Array(upToBuf.prefix(2)), [40, 50])

        var shortBuf = [UInt8](repeating: 0, count: 4)
        try shortBuf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let base = raw.baseAddress else { return XCTFail("missing baseAddress") }
            XCTAssertThrowsError(try QWPOSIX.preadExact(fd: fd, into: base, count: 4, offset: 3)) { err in
                XCTAssertEqual(err as? QWPOSIXError, .shortRead(offset: 3, expected: 4, actual: 2))
            }
        }
    }

    func testPreadRejectsNegativeCountAndOffset() throws {
        var byte: UInt8 = 0
        try withUnsafeMutableBytes(of: &byte) { (raw: UnsafeMutableRawBufferPointer) in
            guard let base = raw.baseAddress else { return XCTFail("missing baseAddress") }
            XCTAssertThrowsError(try QWPOSIX.preadUpTo(fd: -1, into: base, count: 1, offset: 0))
            XCTAssertThrowsError(try QWPOSIX.preadUpTo(fd: 0, into: base, count: -1, offset: 0))
            XCTAssertThrowsError(try QWPOSIX.preadUpTo(fd: 0, into: base, count: 1, offset: -5))
        }
    }
}
