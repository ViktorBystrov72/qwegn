import XCTest
import Metal
@testable import QWCore
@testable import QWHAL
@testable import QWEngine

/// Split mHC router (partial + finalize) and the multi-row prefill router must match the fused
/// qwg_mhc_router_norm_f16 kernel.
final class QWMHCSplitRouterTests: XCTestCase {
    private let c = 4
    private let hidden = 4096

    private func buffer(_ hal: QWHal, _ values: [Float]) throws -> any MTLBuffer {
        try XCTUnwrap(hal.device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
    }

    private func buffer(_ hal: QWHal, f16 values: [Float]) throws -> any MTLBuffer {
        let h = values.map { Float16($0) }
        return try XCTUnwrap(hal.device.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared))
    }

    private func floats(_ b: any MTLBuffer, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    private func halves(_ b: any MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().bindMemory(to: Float16.self, capacity: n)
        return (0..<n).map { Float(p[$0]) }
    }

    private struct Outputs {
        var stream: [Float]
        var norm: [Float]
        var ctrl: [Float]
    }

    private enum Kernel { case fused, split, rows }

    private func run(_ kernel: Kernel, merge: Bool, rows: Int = 3,
                     streams: Int? = nil, geometry: QWModelGeometry? = nil) throws -> Outputs {
        let activeC = streams ?? c
        let geo = try geometry ?? QWWeightAllocationPlanTests.glm53FlashGeometry()
        let hal = try QWHal(geometry: geo)
        let numRows = 2 * activeC + activeC * activeC
        let numCtrl = activeC + activeC * activeC
        let width = activeC * hidden
        let stream = try buffer(hal, (0..<(rows * width)).map { sin(Float($0) * 0.0037) * 0.9 })
        let w = try buffer(hal, (0..<(numRows * width)).map { cos(Float($0) * 0.013) * 0.05 })
        let b = try buffer(hal, (0..<numRows).map { sin(Float($0) * 0.3) * 0.2 })
        let gamma = try buffer(hal, f16: (0..<hidden).map { sin(Float($0) * 0.01) * 0.1 })
        let out = try buffer(hal, f16: [Float](repeating: 0, count: rows * hidden))
        var prev = [Float](repeating: 0, count: rows * numCtrl)
        for t in 0..<rows {
            for i in 0..<activeC { prev[t * numCtrl + i] = 0.5 + 0.1 * Float(i) }
            for m in 0..<(activeC * activeC) { prev[t * numCtrl + activeC + m] = (m % (activeC + 1) == 0 ? 0.7 : 0.1) }
        }
        let ctrl = try buffer(hal, prev)
        let branch = try buffer(hal, f16: (0..<(rows * hidden)).map { cos(Float($0) * 0.021) * 0.5 })
        var uHidden = UInt32(hidden)
        var eps: Float = 1e-6
        var preScale: Float = 1.0
        var iters: UInt32 = 20 | (merge ? 0x8000_0000 : 0)

        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder(dispatchType: .serial))
        if kernel == .split {
            let parts = (hidden + 63) / 64
            let partials = try XCTUnwrap(hal.device.makeBuffer(length: rows * parts * (numRows + 1 + activeC * (activeC + 1) / 2) * 4))
            enc.setComputePipelineState(try hal.pipeline("qwg_mhc_router_partial_f32"))
            enc.setBuffer(stream, offset: 0, index: 0)
            enc.setBuffer(w, offset: 0, index: 1)
            enc.setBuffer(ctrl, offset: 0, index: 2)
            enc.setBuffer(merge ? branch : out, offset: 0, index: 3)
            enc.setBuffer(partials, offset: 0, index: 4)
            enc.setBytes(&uHidden, length: 4, index: 5)
            enc.setBytes(&iters, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: parts, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try hal.pipeline("qwg_mhc_router_finalize_f16"))
            enc.setBuffer(stream, offset: 0, index: 0)
            enc.setBuffer(partials, offset: 0, index: 1)
            enc.setBuffer(b, offset: 0, index: 2)
            enc.setBuffer(gamma, offset: 0, index: 3)
            enc.setBuffer(out, offset: 0, index: 4)
            enc.setBuffer(ctrl, offset: 0, index: 5)
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: (hidden + 255) / 256, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(try hal.pipeline(kernel == .rows ? "qwg_mhc_router_norm_rows_f16"
                                                                         : "qwg_mhc_router_norm_f16"))
            for (i, buf) in [stream, w, b, gamma, out, ctrl].enumerated() {
                enc.setBuffer(buf, offset: 0, index: i)
            }
            enc.setBytes(&uHidden, length: 4, index: 6)
            enc.setBytes(&eps, length: 4, index: 7)
            enc.setBytes(&preScale, length: 4, index: 8)
            enc.setBytes(&iters, length: 4, index: 9)
            enc.setBuffer(merge ? branch : out, offset: 0, index: 10)
            var uRows = UInt32(rows)
            enc.setBytes(&uRows, length: 4, index: 11)
            let rpg = QWEngineDriver.mhcRouterRowsPerGroup
            let groups = kernel == .rows ? (rows + rpg - 1) / rpg : rows
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        return Outputs(stream: floats(stream, rows * width), norm: halves(out, rows * hidden),
                       ctrl: floats(ctrl, rows * numCtrl))
    }

    private func assertZeroULP(_ a: [Float], _ b: [Float], _ what: String) {
        XCTAssertEqual(a.count, b.count)
        for i in 0..<a.count {
            XCTAssertEqual(a[i].bitPattern, b[i].bitPattern, "\(what) 0-ULP mismatch at \(i): \(a[i]) vs \(b[i])")
        }
    }

    private func c2Geometry() throws -> QWModelGeometry {
        let json = """
        {
          "architectures": ["SenseNovaLoopForConditionalGeneration"],
          "model_type": "sensenova_loop",
          "loop_settings": { "start": 0, "end": 1, "eval_count": 1, "counts": [1] },
          "mhc": { "start": 0, "end": 1, "streams": 2, "iterations": 20, "fixed_identity": false, "pre_scale": 1.0 },
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "eos_token_id": 2
          }
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: json))
    }

    private func assertClose(_ a: [Float], _ b: [Float], tol: Float, _ what: String) {
        XCTAssertEqual(a.count, b.count)
        var worst: Float = 0
        for i in 0..<a.count { worst = max(worst, abs(a[i] - b[i]) / max(1, abs(b[i]))) }
        XCTAssertLessThanOrEqual(worst, tol, "\(what) max rel err \(worst)")
    }

    func testSplitMatchesFusedWithoutMerge() throws {
        let fused = try run(.fused, merge: false)
        let split = try run(.split, merge: false)
        assertClose(split.stream, fused.stream, tol: 0, "stream")
        assertClose(split.ctrl, fused.ctrl, tol: 1e-4, "ctrl")
        assertClose(split.norm, fused.norm, tol: 1e-3, "norm")
    }

    func testSplitMatchesFusedWithMerge() throws {
        let fused = try run(.fused, merge: true)
        let split = try run(.split, merge: true)
        assertClose(split.stream, fused.stream, tol: 1e-6, "stream")
        assertClose(split.ctrl, fused.ctrl, tol: 1e-4, "ctrl")
        assertClose(split.norm, fused.norm, tol: 1e-3, "norm")
    }

    func testRowsKernelMatchesFused() throws {
        for merge in [false, true] {
            let fused = try run(.fused, merge: merge, rows: 7)
            let multi = try run(.rows, merge: merge, rows: 7)
            assertClose(multi.stream, fused.stream, tol: 0, "stream merge=\(merge)")
            assertClose(multi.ctrl, fused.ctrl, tol: 1e-5, "ctrl merge=\(merge)")
            assertClose(multi.norm, fused.norm, tol: 1e-3, "norm merge=\(merge)")
        }
    }

    func testC2SplitAndRowsMatchFusedZeroULP() throws {
        let geo = try c2Geometry()
        for merge in [false, true] {
            let fused7 = try run(.fused, merge: merge, rows: 7, streams: 2, geometry: geo)
            let multi7 = try run(.rows, merge: merge, rows: 7, streams: 2, geometry: geo)
            assertZeroULP(multi7.stream, fused7.stream, "c2 rows stream merge=\(merge)")
            assertZeroULP(multi7.ctrl, fused7.ctrl, "c2 rows ctrl merge=\(merge)")
            assertZeroULP(multi7.norm, fused7.norm, "c2 rows norm merge=\(merge)")

            let fused3 = try run(.fused, merge: merge, rows: 3, streams: 2, geometry: geo)
            let split3 = try run(.split, merge: merge, rows: 3, streams: 2, geometry: geo)
            assertClose(split3.stream, fused3.stream, tol: 1e-6, "c2 split stream merge=\(merge)")
            assertClose(split3.ctrl, fused3.ctrl, tol: 1e-4, "c2 split ctrl merge=\(merge)")
            assertClose(split3.norm, fused3.norm, tol: 1e-3, "c2 split norm merge=\(merge)")
        }
    }
}
