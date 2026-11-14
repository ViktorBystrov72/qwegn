// QWPack+TestHelpers.swift — Test-only expert dequant helper for QWQuantTests.
import Foundation
@testable import QWQuant

extension QWQuant.Pack {
    func dequantExpertQ4g(_ name: String, expert: Int) throws -> [Float] {
        let e = try entry(name)
        guard let hq4 = e.hq4Off, let count = e.experts, let per = e.eHq4, e.shape.count == 3 else {
            throw QWQuantError.notQuantPlane(name)
        }
        let rows = e.shape[1], cols = e.shape[2]
        let grp = e.hqGroup, blkBytes = e.hqBlockBytes
        guard (0..<count).contains(expert), cols % grp == 0,
              per == rows * (cols / grp) * blkBytes else {
            throw QWQuantError.notQuantPlane(name)
        }
        let start = hq4 + expert * per
        guard start + per <= rawCount else { throw QWQuantError.truncated }
        let ptr = UnsafeRawPointer(rawBase).advanced(by: start)
        if e.isIncumbentHQ4 {
            return Self.dequantHQ4Slab(base: ptr, nRows: rows, kCols: cols)
        }
        return Self.dequantHQBlockSlab(
            base: ptr, nRows: rows, kCols: cols, bits: e.hqBits, group: grp, isSHVQ: e.isSHVQ
        )
    }
}
