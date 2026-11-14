// QWRopeTables+TestHelpers.swift — Test-only RoPE table row helpers for QWRopeTests.
import Foundation
@testable import QWRope

extension QWRopeTables {
    func cosSinRow(_ p: Int) -> [Float] {
        precondition((0..<rows).contains(p), "position \(p) beyond table rows \(rows)")
        return Array(cosSin[p * width ..< (p + 1) * width])
    }

    func cosSinRowMrope(_ posThw: (t: Int, h: Int, w: Int)) -> [Float] {
        let invf = yarn.map { Self.invFreqYarn($0, geometry: geometry).invf }
            ?? Self.invFreqDefault(theta: Float(geometry.ropeTheta), rotaryDims: geometry.rotaryDims)
        let (t, h, w) = (Float(posThw.t), Float(posThw.h), Float(posThw.w))
        var row = [Float](repeating: 0, count: width)
        for i in 0..<pairs {
            let pos = i % 3 == 0 ? t : (i % 3 == 1 ? h : w)
            let f = pos * invf[i]
            row[i] = Foundation.cos(f) * attentionScaling
            row[pairs + i] = Foundation.sin(f) * attentionScaling
        }
        return row
    }
}
