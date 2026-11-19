// QWRopeCPU.swift — Host-side partial RoPE table generation and vector rotation with YaRN support.

import Dispatch
import Foundation
import QWCore

/// Partial rotary embedding with half-split pairing (`rotate_half` over `0..<rotaryDim`).
public struct QWRopeCPU: Sendable, Hashable {
    public let rotaryDim: Int
    public let theta: Double
    public let yarnFactor: Float?
    public let originalMaxPosition: Int
    public let betaFast: Float
    public let betaSlow: Float
    public let truncate: Bool

    public var attentionScaling: Float {
        guard let f = yarnFactor, f > 1.0 else { return 1.0 }
        return Float(0.1 * Foundation.log(Double(f)) + 1.0)
    }

    public init(
        rotaryDim: Int,
        theta: Double,
        yarnFactor: Float? = nil,
        originalMaxPosition: Int = 262_144,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        truncate: Bool = true
    ) {
        precondition(rotaryDim > 0 && rotaryDim % 2 == 0, "QWRopeCPU requires positive even rotaryDim")
        precondition(theta > 0 && theta.isFinite, "QWRopeCPU requires positive finite theta")
        precondition(originalMaxPosition > 0, "QWRopeCPU requires positive originalMaxPosition")
        self.rotaryDim = rotaryDim
        self.theta = theta
        self.yarnFactor = yarnFactor
        self.originalMaxPosition = originalMaxPosition
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.truncate = truncate
    }

    public init(rotaryDim: Int, theta: Double, yarn: QWYaRNConfig?) {
        self.init(
            rotaryDim: rotaryDim,
            theta: theta,
            yarnFactor: yarn?.factor,
            originalMaxPosition: yarn?.originalMaxPosition ?? 262_144,
            betaFast: yarn?.betaFast ?? 32,
            betaSlow: yarn?.betaSlow ?? 1,
            truncate: yarn?.truncate ?? true
        )
    }

    private func inverseFrequencies() -> [Double] {
        let half = rotaryDim / 2
        if let factor = yarnFactor, factor > 1.0 {
            let yarn = QWYaRNConfig(
                factor: factor,
                originalMaxPosition: originalMaxPosition,
                betaFast: betaFast,
                betaSlow: betaSlow,
                truncate: truncate
            )
            return QWRopeTables.invFreqYarn(rotaryDims: rotaryDim, ropeTheta: theta, yarn: yarn)
                .invf.map(Double.init)
        }
        return (0..<half).map { 1 / pow(theta, Double(2 * $0) / Double(rotaryDim)) }
    }

    /// Generates half-split duplicated `(cos, sin)` tables for positions `0..<count` scaled by `stride`.
    public func tables(count: Int, stride: Int = 1) -> (cos: [Float], sin: [Float]) {
        precondition(count >= 0 && stride >= 0, "QWRopeCPU count and stride must be non-negative")
        guard count > 0, rotaryDim > 0 else { return ([], []) }
        let (_, posOv) = (count - 1).multipliedReportingOverflow(by: stride)
        precondition(!posOv, "QWRopeCPU position overflow")
        let (total, ov) = count.multipliedReportingOverflow(by: rotaryDim)
        precondition(!ov, "QWRopeCPU table size overflow")
        let half = rotaryDim / 2
        let rDim = rotaryDim
        let inv = inverseFrequencies()
        let scale = attentionScaling
        var c = [Float](repeating: 0, count: total)
        var s = [Float](repeating: 0, count: total)
        struct PtrBox: @unchecked Sendable {
            let cp: UnsafeMutablePointer<Float>
            let sp: UnsafeMutablePointer<Float>
        }
        let chunks = max(1, min(count, ProcessInfo.processInfo.activeProcessorCount))
        let per = (count + chunks - 1) / chunks
        c.withUnsafeMutableBufferPointer { cb in
            s.withUnsafeMutableBufferPointer { sb in
                guard let cp = cb.baseAddress, let sp = sb.baseAddress else { return }
                let box = PtrBox(cp: cp, sp: sp)
                DispatchQueue.concurrentPerform(iterations: chunks) { ch in
                    let lo = ch * per, hi = min(count, lo + per)
                    guard lo < hi else { return }
                    for p in lo..<hi {
                        let pos = Double(p * stride)
                        let base = p * rDim
                        for i in 0..<half {
                            let angle = pos * inv[i]
                            let cv = Float(cos(angle)) * scale, sv = Float(sin(angle)) * scale
                            box.cp[base + i] = cv; box.cp[base + i + half] = cv
                            box.sp[base + i] = sv; box.sp[base + i + half] = sv
                        }
                    }
                }
            }
        }
        return (c, s)
    }

    /// Generates half-split duplicated `(cos, sin)` tables for explicit absolute `positions`.
    public func tables(positions: [Int]) -> (cos: [Float], sin: [Float]) {
        guard !positions.isEmpty, rotaryDim > 0 else { return ([], []) }
        let (total, ov) = positions.count.multipliedReportingOverflow(by: rotaryDim)
        precondition(!ov, "QWRopeCPU table size overflow")
        let half = rotaryDim / 2
        let inv = inverseFrequencies()
        let scale = attentionScaling
        var c = [Float](repeating: 0, count: total)
        var s = [Float](repeating: 0, count: total)
        for (p, pos) in positions.enumerated() {
            precondition(pos >= 0, "QWRopeCPU positions must be non-negative")
            for i in 0..<half {
                let angle = Double(pos) * inv[i]
                let cv = Float(cos(angle)) * scale, sv = Float(sin(angle)) * scale
                c[p * rotaryDim + i] = cv; c[p * rotaryDim + i + half] = cv
                s[p * rotaryDim + i] = sv; s[p * rotaryDim + i + half] = sv
            }
        }
        return (c, s)
    }

    /// In-place half-split rotation of one head vector (length `dim`) using row `p`.
    public func apply(_ v: inout [Float], base: Int, dim: Int, cos c: [Float], sin s: [Float], row p: Int) {
        let ok = Self.validateApplyBounds(
            vectorCount: v.count, base: base, dim: dim,
            rotaryDim: rotaryDim, cosCount: c.count, sinCount: s.count, row: p
        )
        precondition(ok, "QWRopeCPU apply bounds")
        Self.applyValidated(&v, base: base, rotaryDim: rotaryDim, cos: c, sin: s, row: p)
    }

    /// Checked in-place half-split rotation throwing `QWRopeError` on invalid vector or table bounds.
    public func applyChecked(_ v: inout [Float], base: Int, dim: Int,
                             cos c: [Float], sin s: [Float], row p: Int) throws {
        try Self.applyCPU(&v, rotaryDim: rotaryDim, base: base, dim: dim, cos: c, sin: s, row: p)
    }

    /// Static checked CPU rotation usable without constructing a `QWRopeCPU` instance.
    public static func applyCPU(_ v: inout [Float], rotaryDim: Int, base: Int, dim: Int,
                                cos c: [Float], sin s: [Float], row p: Int) throws {
        let vecSum = base.addingReportingOverflow(dim)
        guard rotaryDim > 0, rotaryDim % 2 == 0,
              base >= 0, dim >= rotaryDim,
              !vecSum.overflow, vecSum.partialValue <= v.count else {
            throw QWRopeError.badShape("invalid vector bounds base=\(base) dim=\(dim) rotaryDim=\(rotaryDim)")
        }
        let availableRows = min(c.count, s.count) / rotaryDim
        guard p >= 0, p < availableRows else {
            throw QWRopeError.posBeyondTable(p, availableRows)
        }
        applyValidated(&v, base: base, rotaryDim: rotaryDim, cos: c, sin: s, row: p)
    }

    private static func validateApplyBounds(
        vectorCount: Int, base: Int, dim: Int, rotaryDim: Int,
        cosCount: Int, sinCount: Int, row p: Int
    ) -> Bool {
        guard base >= 0, dim >= rotaryDim, rotaryDim > 0, rotaryDim % 2 == 0, p >= 0 else { return false }
        let vecSum = base.addingReportingOverflow(dim)
        guard !vecSum.overflow, vecSum.partialValue <= vectorCount else { return false }
        let rowEnd = (p + 1).multipliedReportingOverflow(by: rotaryDim)
        guard !rowEnd.overflow, rowEnd.partialValue <= cosCount, rowEnd.partialValue <= sinCount else {
            return false
        }
        return true
    }

    private static func applyValidated(_ v: inout [Float], base: Int, rotaryDim: Int,
                                         cos c: [Float], sin s: [Float], row p: Int) {
        let half = rotaryDim / 2
        let x = Array(v[base..<(base + rotaryDim)])
        let rowBase = p * rotaryDim
        for i in 0..<rotaryDim {
            let rot = i < half ? -x[i + half] : x[i - half]
            v[base + i] = x[i] * c[rowBase + i] + rot * s[rowBase + i]
        }
    }
}

public extension QWRopeTables {
    /// Host-side half-split RoPE helper matching this table's rotary geometry and YaRN config.
    var cpu: QWRopeCPU {
        QWRopeCPU(rotaryDim: geometry.rotaryDims, theta: geometry.ropeTheta, yarn: yarn)
    }

    /// Applies half-split CPU rotation using this table's `cosSin` buffer at `row`.
    func applyCPU(_ v: inout [Float], base: Int, dim: Int, row p: Int) throws {
        guard p >= 0, p < rows else {
            throw QWRopeError.posBeyondTable(p, rows)
        }
        guard base >= 0, dim >= width, !base.addingReportingOverflow(dim).overflow,
              base + dim <= v.count else {
            throw QWRopeError.badShape("invalid vector bounds base=\(base) dim=\(dim) width=\(width)")
        }
        let half = pairs
        let x = Array(v[base..<(base + width)])
        let rowBase = p * width
        for i in 0..<width {
            let pairIdx = i < half ? i : i - half
            let rot = i < half ? -x[i + half] : x[i - half]
            v[base + i] = x[i] * cosSin[rowBase + pairIdx] + rot * cosSin[rowBase + half + pairIdx]
        }
    }
}
