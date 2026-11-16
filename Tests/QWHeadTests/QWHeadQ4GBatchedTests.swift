import XCTest
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWGEMM
import QWHead

/// Phase 3 Step 3.0 (P4-AD-84): the batched q4g head kernel is the DEFAULT
/// verify route on the q4g pack — this test pins the law that makes the
/// route change TOKEN-ZERO: qwg_head_q4g_mN is BIT-EXACT row-for-row vs the
final class QWHeadQ4GBatchedTests: XCTestCase {
    func testAllRowsBitExactAgainstShippingGemv() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let h = QWModelGeometry.qwen38_27b.hidden, g = h / 128, g32 = h / 32
        let maxM = 16
        for n in [1, 257, 1024] {
            // HQ4_G32_M5 blocks: [n·g32·20] B = [n·g32·10] f16 elements.
            let arena = try QWArena(hal: hal, weights: n * g32 * 20 + 256,
                                    activations: maxM * h * 2 + maxM * g * 4 + 2 * maxM * n * 4 + 4096,
                                    kv: 256, gdnStates: 256)
            let wa = arena.weightsRegion()
            let blk = try wa.allocate(QWTensorShape([n * g32 * 10], .f16))
            let u16 = blk.region.pointer.advanced(by: blk.offset).bindMemory(to: UInt16.self, capacity: n * g32 * 10)
            var s: UInt64 = 0x5DEECE66D
            func next() -> UInt8 { s = (s &* 6364136223846793005 &+ 1442695040888963407) >> 33; return UInt8(truncatingIfNeeded: s) }
            for b in 0..<(n * g32) {
                let base = b * 10
                let sc16 = Float16(bitPattern: UInt16(0x2800 | (UInt16(next()) & 0x3F)))
                let mnVal = Float((Int(next()) - 128)) * 0.001
                let bias16 = Float16(mnVal - 1024.0 * Float(sc16))
                u16[base + 0] = sc16.bitPattern
                u16[base + 1] = bias16.bitPattern
                for w in 2..<10 {
                    u16[base + w] = UInt16(next()) | (UInt16(next()) << 8)
                }
            }
            guard let view = QWWeightView(blocks: blk, n: n, k: h, isRotated: true),
                  let aqView = QWWeightView(blocks: blk, n: n, k: h, bits: 4, groupSize: 32,
                                            isRotated: true, isIncumbentContainer: false) else {
                return XCTFail("QWWeightView layout invalid")
            }
            let a = arena.activationsRegion()
            let x = try a.allocate(QWTensorShape([maxM * h], .f16))
            let gs = try a.allocate(QWTensorShape([maxM * g], .f32))
            let gsRow = try a.allocate(QWTensorShape([g], .f32))
            let y = try a.allocate(QWTensorShape([maxM * n], .f32))
            let control = try a.allocate(QWTensorShape([maxM * n], .f32))
            let xp = x.region.pointer.advanced(by: x.offset).assumingMemoryBound(to: Float16.self)
            for i in 0..<(maxM * h) { xp[i] = Float16(Float((i * 31) % 113 - 56) / 53) }
            let plan = QWGemm.plan(QWGemmShape(m: 1, n: n, k: h), dtype: .q4g, epi: .none)
            func prefix(_ t: QWTensorValue, _ count: Int, _ dt: QWDType) -> QWTensorValue {
                QWTensorValue(region: t.region, offset: t.offset, shape: QWTensorShape([count], dt))
            }
            for m in [2, 3, 4, 7, 8, 9, 12, 16] {
                let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
                defer { if cb.status == .notEnqueued { enc.endEncoding() } }
                try QWHead.encodeBatched(enc, hal: hal, view: view,
                                         x: prefix(x, m * h, .f16),
                                         gs: prefix(gs, m * g, .f32),
                                         y: prefix(y, m * n, .f32), m: m)
                for row in 0..<m {
                    let xr = QWTensorValue(region: x.region, offset: x.offset + row * h * 2,
                                           shape: QWTensorShape([h], .f16))
                    let yr = QWTensorValue(region: control.region, offset: control.offset + row * n * 4,
                                           shape: QWTensorShape([n], .f32))
                    try QWGemm.encode(enc, hal: hal, plan: plan, x: xr,
                                      xGroupSum: gsRow, wQ4: view, y: yr)
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
                let got = y.region.pointer.advanced(by: y.offset).assumingMemoryBound(to: Float.self)
                let ref = control.region.pointer.advanced(by: control.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(m * n) {
                    XCTAssertTrue(got[i].isFinite, "M=\(m), N=\(n), output=\(i)")
                    XCTAssertEqual(got[i].bitPattern, ref[i].bitPattern, "M=\(m), N=\(n), output=\(i)")
                }
            }
            for m in [4, 9, 16] {
                let cbAQ = try XCTUnwrap(hal.queue.makeCommandBuffer())
                let encAQ = try XCTUnwrap(cbAQ.makeComputeCommandEncoder())
                try QWHead.encodeBatched(encAQ, hal: hal, view: aqView,
                                         x: prefix(x, m * h, .f16),
                                         gs: prefix(gs, m * g, .f32),
                                         y: prefix(y, m * n, .f32), m: m)
                for startRow in stride(from: 0, to: m & ~1, by: 2) {
                    let xr = QWTensorValue(region: x.region, offset: x.offset + startRow * h * 2,
                                           shape: QWTensorShape([2 * h], .f16))
                    let gr = QWTensorValue(region: gs.region, offset: gs.offset + startRow * g * 4,
                                           shape: QWTensorShape([2 * g], .f32))
                    let yr = QWTensorValue(region: control.region, offset: control.offset + startRow * n * 4,
                                           shape: QWTensorShape([2 * n], .f32))
                    try QWHead.encodeBatched(encAQ, hal: hal, view: aqView, x: xr, gs: gr, y: yr, m: 2)
                }
                if m & 1 != 0 {
                    encAQ.memoryBarrier(scope: .buffers)
                    let xr = QWTensorValue(region: x.region, offset: x.offset + (m - 2) * h * 2,
                                           shape: QWTensorShape([2 * h], .f16))
                    let gr = QWTensorValue(region: gs.region, offset: gs.offset + (m - 2) * g * 4,
                                           shape: QWTensorShape([2 * g], .f32))
                    let yr = QWTensorValue(region: control.region, offset: control.offset + (m - 2) * n * 4,
                                           shape: QWTensorShape([2 * n], .f32))
                    try QWHead.encodeBatched(encAQ, hal: hal, view: aqView, x: xr, gs: gr, y: yr, m: 2)
                }
                encAQ.endEncoding()
                cbAQ.commit()
                cbAQ.waitUntilCompleted()
                XCTAssertEqual(cbAQ.status, .completed, "\(String(describing: cbAQ.error))")
                let got = y.region.pointer.advanced(by: y.offset).assumingMemoryBound(to: Float.self)
                let ref = control.region.pointer.advanced(by: control.offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(m * n) {
                    XCTAssertTrue(got[i].isFinite, "AQ4 M=\(m), N=\(n), output=\(i)")
                    XCTAssertEqual(got[i].bitPattern, ref[i].bitPattern, "AQ4 M=\(m), N=\(n), output=\(i)")
                }
            }
        }
    }

    func testInvalidBatchedShapesAreRejectedBeforeDispatch() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let h = QWModelGeometry.qwen38_27b.hidden, g = h / 128, g32 = h / 32
        let arena = try QWArena(hal: hal, weights: 2 * g32 * 20 + 256,
                                activations: 6 * h + 2048, kv: 256, gdnStates: 256)
        let wa = arena.weightsRegion()
        let blk = try wa.allocate(QWTensorShape([2 * g32 * 10], .f16))
        guard let view = QWWeightView(blocks: blk, n: 2, k: h, isRotated: true) else {
            return XCTFail("QWWeightView layout invalid")
        }
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([h], .f16))
        let gs = try a.allocate(QWTensorShape([g], .f32))
        let y = try a.allocate(QWTensorShape([1], .f32))
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        defer { enc.endEncoding() }
        for m in [0, 1, 7] {
            XCTAssertThrowsError(try QWHead.encodeBatched(enc, hal: hal, view: view,
                                                          x: x, gs: gs, y: y, m: m))
        }
        // wrong activation/gs/y extents for m=2
        XCTAssertThrowsError(try QWHead.encodeBatched(enc, hal: hal, view: view,
                                                      x: x, gs: gs, y: y, m: 2))
    }
}
