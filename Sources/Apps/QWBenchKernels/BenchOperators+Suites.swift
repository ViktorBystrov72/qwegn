// BenchOperators+Suites.swift — GDN input-projection and LM-head operator benches.

import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

extension OperatorSuites {
    // MARK: - The GDN input projection

    /// The fused GDN projection against the two plain GEMVs it replaces. The
    /// fused kernel also writes the 96 rows of control output — `glog` and
    /// `betas` — that the plain GEMVs do not.
    static func gdnInproj(hal: QWHal, cols: Int,
                                   samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let rows = gdnRows
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let y = try BenchRunner.buffer(device, bytes: rows * 2)
        let y32 = try BenchRunner.buffer(device, bytes: rows * 4)

        // The control inputs: 48 `a` and 48 `b` rows, each with its own
        // log-scale and bias.
        let aLog = try BenchRunner.buffer(device, bytes: 48 * 4)
        let dtBias = try BenchRunner.buffer(device, bytes: 48 * 4)
        let betas = try BenchRunner.buffer(device, bytes: 48 * 4)
        let glog = try BenchRunner.buffer(device, bytes: 48 * 4)
        var aLogValues = [Float](repeating: 0, count: 48)
        var dtBiasValues = [Float](repeating: 0, count: 48)
        for h in 0..<48 {
            aLogValues[h] = Float(h % 7) * 0.125 - 0.375
            dtBiasValues[h] = Float(h % 5) * 0.25 - 0.5
        }
        BenchRunner.fillF32(aLog, aLogValues)
        BenchRunner.fillF32(dtBias, dtBiasValues)

        var expected = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                        count: rows)
        for r in 0..<rows {
            expected[r] = reference(bank: bank, row: r, cols: cols, xs: xs)
        }

        let psoGdn = try hal.pipeline("qwg_gemv_q4g_dx_h16_gdn_inproj_t4")
        let psoGemv = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
        let stride = rowBytes(cols: cols)

        func encodeGemv(_ enc: any MTLComputeCommandEncoder, row0: Int, count: Int) {
            enc.setComputePipelineState(psoGemv)
            enc.setBuffer(weights, offset: row0 * stride, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            // Production binds the **f16** tensor to this dead slot; the
            // kernel never writes it, and neither does the bench.
            enc.setBuffer(y, offset: 0, index: 3)
            var n = UInt32(count), k = UInt32(cols)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&k, length: 4, index: 5)
            enc.setBuffer(y, offset: row0 * 2, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (count + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        let fused = Racer(name: "qwg_gemv_q4g_dx_h16_gdn_inproj_t4",
                          dispatches: 1,
                          encode: { enc in
                              enc.setComputePipelineState(psoGdn)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x, offset: 0, index: 1)
                              enc.setBuffer(y32, offset: 0, index: 3)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 4)
                              enc.setBytes(&k, length: 4, index: 5)
                              enc.setBuffer(y, offset: 0, index: 8)
                              enc.setBuffer(aLog, offset: 0, index: 9)
                              enc.setBuffer(dtBias, offset: 0, index: 10)
                              enc.setBuffer(betas, offset: 0, index: 11)
                              enc.setBuffer(glog, offset: 0, index: 12)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                          })

        let fragmented = Racer(name: "qwg_gemv_q4g_dx_h16_t4 x2",
                               dispatches: 2,
                               encode: { enc in
                                   encodeGemv(enc, row0: 0, count: gdnQKVZ)
                                   encodeGemv(enc, row0: gdnQKVZ, count: rows - gdnQKVZ)
                               })

        return try race(hal: hal, suite: OperatorSuite.gdnInproj, rows: rows, cols: cols, racers: [fused, fragmented],
                        check: { which in
                            var worst = BenchRunner.worstError(y, rows: rows, reference: expected)
                            if which == 0 {
                                // The epilogue, checked against the formula
                                // applied to the kernel's **own** vx: the dot
                                // product is verified on its own, above.
                                let device = y32.contents().bindMemory(to: Float.self, capacity: rows)
                                let gotGlog = glog.contents().bindMemory(to: Float.self, capacity: 48)
                                let gotBetas = betas.contents().bindMemory(to: Float.self, capacity: 48)
                                for h in 0..<48 {
                                    let va = device[gdnQKVZ + h]
                                    let vb = device[gdnQKVZ + 48 + h]
                                    let wantGlog = -exp(aLogValues[h])
                                        * log(1.0 + exp(va + dtBiasValues[h]))
                                    let wantBetas = 1.0 / (1.0 + exp(-vb))
                                    worst = max(worst, rel(gotGlog[h], wantGlog))
                                    worst = max(worst, rel(gotBetas[h], wantBetas))
                                }
                            }
                            return worst
                        },
                        samples: samples, repeats: repeats)
    }

    /// The relative error of two values, against the larger of them and a floor.
    static func rel(_ got: Float, _ want: Float) -> Double {
        let scale = max(abs(Double(want)), 1e-6)
        return abs(Double(got) - Double(want)) / scale
    }

    // MARK: - The LM head and the argmax

    /// The dense engine's fused head against its fragmented one. The check is
    /// the **token**, not the key slots: the slot layout is an implementation
    /// detail of the fused path.
    static func lmHead(hal: QWHal, cols: Int,
                               samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let rows = headRows
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let x32 = try BenchRunner.buffer(device, bytes: cols * 4)
        BenchRunner.fillF32(x32, xs)
        let logits = try BenchRunner.buffer(device, bytes: rows * 4)
        let keysHi = try BenchRunner.buffer(device, bytes: rows * 4)
        let keysLo = try BenchRunner.buffer(device, bytes: rows * 4)
        let token = try BenchRunner.buffer(device, bytes: 4)

        // The host's answer: the largest logit, ties to the lowest row.
        var bestRow = 0
        var bestValue = -Float.infinity
        var values = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            let v = reference(bank: bank, row: r, cols: cols, xs: xs).value
            values[r] = v
            if v > bestValue { bestValue = v; bestRow = r }
        }

        let psoHead = try hal.pipeline("qwg_head_gemv_q4g_argmax")
        let psoArgmax = try hal.pipeline("qwg_head_argmax")
        let psoFin = try hal.pipeline("qwg_head_argmax_fin")
        let psoGemv = try hal.pipeline("qwg_gemv_q4g_v4")
        let psoFlash = try hal.pipeline("qwg_gemv_q4g_f32_v4_am")

        // The fused path: one key pair per threadgroup of eight rows.
        let headSlots = (rows + 7) / 8
        let fused = Racer(name: "qwg_head_gemv_q4g_argmax + qwg_head_argmax_fin",
                          dispatches: 2, slots: headSlots,
                          encode: { enc in
                              enc.setComputePipelineState(psoHead)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x, offset: 0, index: 1)
                              enc.setBuffer(logits, offset: 0, index: 3)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 4)
                              enc.setBytes(&k, length: 4, index: 5)
                              enc.setBuffer(keysHi, offset: 0, index: 6)
                              enc.setBuffer(keysLo, offset: 0, index: 7)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: headSlots, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                              enc.memoryBarrier(scope: .buffers)
                              enc.setComputePipelineState(psoFin)
                              enc.setBuffer(keysHi, offset: 0, index: 0)
                              enc.setBuffer(keysLo, offset: 0, index: 1)
                              enc.setBuffer(token, offset: 0, index: 2)
                              var slots = UInt32(headSlots)
                              enc.setBytes(&slots, length: 4, index: 3)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                          })

        // The fragmented path: the logits, then a separate argmax.
        let argmaxSlots = (rows + 255) / 256
        let fragmented = Racer(name: "qwg_gemv_q4g_v4 + qwg_head_argmax + qwg_head_argmax_fin",
                               dispatches: 3, slots: argmaxSlots,
                               encode: { enc in
                                   enc.setComputePipelineState(psoGemv)
                                   enc.setBuffer(weights, offset: 0, index: 0)
                                   enc.setBuffer(x, offset: 0, index: 1)
                                   enc.setBuffer(logits, offset: 0, index: 3)
                                   var n = UInt32(rows), k = UInt32(cols)
                                   enc.setBytes(&n, length: 4, index: 4)
                                   enc.setBytes(&k, length: 4, index: 5)
                                   // Production's `handQ4gV4` geometry: four
                                   // rows per threadgroup, 128 threads.
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                                   enc.setComputePipelineState(psoArgmax)
                                   enc.setBuffer(logits, offset: 0, index: 0)
                                   enc.setBuffer(keysHi, offset: 0, index: 1)
                                   enc.setBuffer(keysLo, offset: 0, index: 2)
                                   var n2 = UInt32(rows)
                                   enc.setBytes(&n2, length: 4, index: 3)
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: argmaxSlots, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                   enc.memoryBarrier(scope: .buffers)
                                   enc.setComputePipelineState(psoFin)
                                   enc.setBuffer(keysHi, offset: 0, index: 0)
                                   enc.setBuffer(keysLo, offset: 0, index: 1)
                                   enc.setBuffer(token, offset: 0, index: 2)
                                   var slots = UInt32(argmaxSlots)
                                   enc.setBytes(&slots, length: 4, index: 3)
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                               })

        // The FlashNext path: f32 activations, sixteen rows folded into one
        // key pair per threadgroup, and no separate argmax pass.
        let flashSlots = (rows + 15) / 16
        let flash = Racer(name: "qwg_gemv_q4g_f32_v4_am + qwg_head_argmax_fin",
                          dispatches: 2, slots: flashSlots,
                          encode: { enc in
                              enc.setComputePipelineState(psoFlash)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x32, offset: 0, index: 1)
                              enc.setBuffer(logits, offset: 0, index: 2)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 3)
                              enc.setBytes(&k, length: 4, index: 4)
                              var epi: UInt32 = 0
                              enc.setBytes(&epi, length: 4, index: 5)
                              var scale: Float = 1
                              enc.setBytes(&scale, length: 4, index: 6)
                              enc.setBuffer(keysHi, offset: 0, index: 7)
                              enc.setBuffer(keysLo, offset: 0, index: 8)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: flashSlots, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                              enc.memoryBarrier(scope: .buffers)
                              enc.setComputePipelineState(psoFin)
                              enc.setBuffer(keysHi, offset: 0, index: 0)
                              enc.setBuffer(keysLo, offset: 0, index: 1)
                              enc.setBuffer(token, offset: 0, index: 2)
                              var slots = UInt32(flashSlots)
                              enc.setBytes(&slots, length: 4, index: 3)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                          })
        return try race(hal: hal, suite: OperatorSuite.lmHead, rows: rows, cols: cols,
                        racers: [fused, fragmented, flash],
                        check: { _ in
                            let got = token.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
                            guard Int(got) < rows else { return Double.infinity }
                            if Int(got) == bestRow { return 0 }
                            // A different token is only a failure when the two
                            // logit values are not the same to rounding.
                            let scale = max(abs(Double(bestValue)), 1e-6)
                            let gap = abs(Double(bestValue) - Double(values[Int(got)])) / scale
                            return gap
                        },
                        samples: samples, repeats: repeats)
    }
}
