// QWDecisionTopologyAttnNAXParityTests.swift — Parity gate comparing NAX topological FlashAttention vs scalar reference.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc

final class QWDecisionTopologyAttnNAXParityTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Decision topology NAX requires Metal 4 tensor hardware")
    }

    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Float = 0.75) -> Float {
            let u = Float(next() >> 11) * (1.0 / 9007199254740992.0)
            return (u * 2.0 - 1.0) * scale
        }
    }

    private func makeF16Tensor(values: [Float], shape: [Int]) throws -> QWTensorValue {
        let ts = QWTensorShape(shape, .f16)
        let bytes = max(64, qwAlignUp(ts.totalBytes, 64))
        guard let buf = Self.hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate Metal buffer"); throw QWAttnError.badShape("alloc")
        }
        memset(buf.contents(), 0, bytes)
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        for (i, v) in values.enumerated() { ptr[i] = Float16(v) }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: ts)
    }

    private func makeI32Tensor(values: [Int32], shape: [Int]) throws -> QWTensorValue {
        let ts = QWTensorShape(shape, .i32)
        let bytes = max(64, qwAlignUp(ts.totalBytes, 64))
        guard let buf = Self.hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            XCTFail("Failed to allocate Metal buffer"); throw QWAttnError.badShape("alloc")
        }
        memset(buf.contents(), 0, bytes)
        let ptr = buf.contents().assumingMemoryBound(to: Int32.self)
        for (i, v) in values.enumerated() { ptr[i] = v }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: ts)
    }

    private func runPair(
        op: QWDecisionTopologyAttn,
        seqLen: Int,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        topology: QWDecisionModelConfig.AttentionTopology,
        groupIds: [Int32],
        stateEndPos: Int = 0,
        mediaPrefixLen: Int = 0,
        slidingWindow: Int = 0,
        seed: UInt64
    ) throws -> (maxDiff: Float, naxOut: [Float16]) {
        var rng = SplitMix64(state: seed)
        let qVals = (0..<(seqLen * qHeads * headDim)).map { _ in rng.nextSigned() }
        let kVals = (0..<(seqLen * kvHeads * headDim)).map { _ in rng.nextSigned() }
        let vVals = (0..<(seqLen * kvHeads * headDim)).map { _ in rng.nextSigned() }
        let qT = try makeF16Tensor(values: qVals, shape: [seqLen, qHeads * headDim])
        let kT = try makeF16Tensor(values: kVals, shape: [seqLen, kvHeads * headDim])
        let vT = try makeF16Tensor(values: vVals, shape: [seqLen, kvHeads * headDim])
        let gT = try makeI32Tensor(values: groupIds, shape: [seqLen])
        let outRef = try makeF16Tensor(values: [Float](repeating: 0, count: seqLen * qHeads * headDim), shape: [seqLen, qHeads * headDim])
        let outNax = try makeF16Tensor(values: [Float](repeating: 0, count: seqLen * qHeads * headDim), shape: [seqLen, qHeads * headDim])

        guard let (cb, enc) = Self.hal.begin() else { throw QWAttnError.badShape("cb") }
        try op.encodePrefill(
            enc, q: qT, k: kT, v: vT, groupIds: gT, out: outRef,
            seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            topology: topology, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen,
            slidingWindow: slidingWindow, forceScalar: true
        )
        try op.encodePrefill(
            enc, q: qT, k: kT, v: vT, groupIds: gT, out: outNax,
            seqLen: seqLen, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            topology: topology, stateEndPos: stateEndPos, mediaPrefixLen: mediaPrefixLen,
            slidingWindow: slidingWindow, forceScalar: false
        )
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)

        let total = seqLen * qHeads * headDim
        let rPtr = outRef.region.pointer.advanced(by: outRef.offset).assumingMemoryBound(to: Float16.self)
        let nPtr = outNax.region.pointer.advanced(by: outNax.offset).assumingMemoryBound(to: Float16.self)
        var maxDiff: Float = 0
        var naxOut = [Float16](repeating: 0, count: total)
        for i in 0..<total {
            naxOut[i] = nPtr[i]
            XCTAssertTrue(nPtr[i].isFinite)
            maxDiff = max(maxDiff, abs(Float(rPtr[i]) - Float(nPtr[i])))
        }
        return (maxDiff, naxOut)
    }

    func testAllTopologiesGQAAndPaddingParityD64AndD128() throws {
        let op = try QWDecisionTopologyAttn(hal: Self.hal)

        for headDim in [64, 128] {
            // 1. Causal S=65
            let gAll0_65 = [Int32](repeating: 0, count: 65)
            let c = try runPair(
                op: op, seqLen: 65, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .causal(supportsPrefixCache: false), groupIds: gAll0_65, seed: UInt64(10 + headDim)
            )
            XCTAssertLessThan(c.maxDiff, 2e-3, "causal hd=\(headDim)")

            // 2. Prefix bidirectional S=73, stateEndPos=37
            let p = try runPair(
                op: op, seqLen: 73, qHeads: 16, kvHeads: 4, headDim: headDim,
                topology: .prefixBidirectional,
                groupIds: [Int32](repeating: 0, count: 73), stateEndPos: 37, seed: UInt64(20 + headDim)
            )
            XCTAssertLessThan(p.maxDiff, 2e-3, "prefixBidirectional hd=\(headDim)")

            // 3. Prompt bidirectional S=17
            let pb = try runPair(
                op: op, seqLen: 17, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .promptBidirectional, groupIds: [Int32](repeating: 0, count: 17), seed: UInt64(30 + headDim)
            )
            XCTAssertLessThan(pb.maxDiff, 2e-3, "promptBidirectional hd=\(headDim)")

            // 4. PackedSet S=131 with shared prefix (0), isolated groups (1, 2, 3), and padded (-1)
            var packedGroups = [Int32](repeating: 0, count: 131)
            for i in 40..<70 { packedGroups[i] = 1 }
            for i in 70..<102 { packedGroups[i] = 2 }
            for i in 102..<120 { packedGroups[i] = 3 }
            for i in 120..<131 { packedGroups[i] = -1 }
            packedGroups[5] = -1
            let ps = try runPair(
                op: op, seqLen: 131, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .packedSet(maxSetSize: 8), groupIds: packedGroups, seed: UInt64(40 + headDim)
            )
            XCTAssertLessThan(ps.maxDiff, 2e-3, "packedSet hd=\(headDim)")
            // Padded rows (-1) must output exact zero.
            let rowStride = 8 * headDim
            for paddedRow in [5, 120, 130] {
                for d in 0..<rowStride {
                    XCTAssertEqual(Float(ps.naxOut[paddedRow * rowStride + d]), 0.0, accuracy: 0.0)
                }
            }

            // 5. BidirectionalEncoder with mediaPrefixLen=20
            let be = try runPair(
                op: op, seqLen: 67, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .bidirectionalEncoder(isolateMediaPrefix: true),
                groupIds: [Int32](repeating: 0, count: 67), mediaPrefixLen: 20, seed: UInt64(50 + headDim)
            )
            XCTAssertLessThan(be.maxDiff, 2e-3, "bidirectionalEncoder hd=\(headDim)")

            // 6. SlidingWindow=16 at S=65
            let sw = try runPair(
                op: op, seqLen: 65, qHeads: 8, kvHeads: 2, headDim: headDim,
                topology: .causal(supportsPrefixCache: false),
                groupIds: gAll0_65, slidingWindow: 16, seed: UInt64(60 + headDim)
            )
            XCTAssertLessThan(sw.maxDiff, 2e-3, "slidingWindow hd=\(headDim)")
        }
    }
}
