// QWGDNWave2Tests.swift — Tests convPrep state aliasing guard and VHeads > 64 / softplus clamp.
import Foundation
import Metal
import XCTest
@testable import QWGDN
import QWCore
import QWHAL
import QWAlloc

final class QWGDNWave2Tests: XCTestCase {
    private static func wideGdnGeometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 1024,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 96,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "eos_token_id": 2,
            "vocab_size": 64
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    func testWideGdnGeometryDecodes() throws {
        let g = try Self.wideGdnGeometry()
        XCTAssertEqual(g.gdnVHeads, 96)
    }

    func testEncodeConvPrepSeqF32RejectsAliasedStateAndZeroRows() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let g = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: g)
        let ch = g.gdnQKVChannels, vh = g.gdnVHeads
        let arena = try QWArena(hal: hal, weights: 0, activations: (ch * 16 + vh * 16) * 4 + 65536, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([ch], .f32))
        let w = try a.allocate(QWTensorShape([ch * 4], .f32))
        let st0 = try a.allocate(QWTensorShape([3 * ch], .f32))
        let st1 = try a.allocate(QWTensorShape([3 * ch], .f32))
        let b = try a.allocate(QWTensorShape([vh], .f32))
        let aIn = try a.allocate(QWTensorShape([vh], .f32))
        let aLog = try a.allocate(QWTensorShape([vh], .f32))
        let dt = try a.allocate(QWTensorShape([vh], .f32))
        let y = try a.allocate(QWTensorShape([ch], .f32))
        let glog = try a.allocate(QWTensorShape([vh], .f32))
        let beta = try a.allocate(QWTensorShape([vh], .f32))

        let (cb, enc) = try XCTUnwrap(hal.begin())
        XCTAssertThrowsError(
            try QWGDNF32.encodeConvPrepSeqF32(
                enc, hal: hal, x: x, w: w, stIn: st0, stOut: st0,
                b: b, a: aIn, aLog: aLog, dtBias: dt, y: y, glog: glog, beta: beta, rows: 1
            )
        )
        XCTAssertThrowsError(
            try QWGDNF32.encodeConvPrepSeqF32(
                enc, hal: hal, x: x, w: w, stIn: st0, stOut: st1,
                b: b, a: aIn, aLog: aLog, dtBias: dt, y: y, glog: glog, beta: beta, rows: 0
            )
        )
        enc.endEncoding()
        _ = cb
    }

    func testFusedConvHeadnormSupportsOver64VHeadsAndClampsSoftplus() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let g = try Self.wideGdnGeometry()
        XCTAssertEqual(g.gdnVHeads, 96)
        let hal = try QWHal(geometry: g)
        let conv = try QWGDNConv(hal: hal, geometry: g)
        let T = 2, ch = g.gdnQKVChannels, vh = g.gdnVHeads
        let bytes = (T * ch * 2 + ch * 4 + 3 * ch * 2) * 2 + (T * 2 * vh + T * 2 * vh + vh * 2) * 4 + 16384
        let arena = try QWArena(hal: hal, weights: 0, activations: bytes, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let x = try a.allocate(QWTensorShape([T * ch], .f16))
        let w = try a.allocate(QWTensorShape([ch * 4], .f16))
        let stIn = try a.allocate(QWTensorShape([3 * ch], .f16))
        let y = try a.allocate(QWTensorShape([T * ch], .f16))
        let stOut = try a.allocate(QWTensorShape([3 * ch], .f16))
        let ab32 = try a.allocate(QWTensorShape([T * 2 * vh], .f32))
        let aLog = try a.allocate(QWTensorShape([vh], .f32))
        let dtBias = try a.allocate(QWTensorShape([vh], .f32))
        let betas = try a.allocate(QWTensorShape([T * vh], .f32))
        let glog = try a.allocate(QWTensorShape([T * vh], .f32))

        let abPtr = arena.cpuPointer(of: ab32).assumingMemoryBound(to: Float.self)
        let alPtr = arena.cpuPointer(of: aLog).assumingMemoryBound(to: Float.self)
        let dtPtr = arena.cpuPointer(of: dtBias).assumingMemoryBound(to: Float.self)
        for h in 0..<vh {
            alPtr[h] = 0.0
            dtPtr[h] = (h >= 64) ? 120.0 : 0.5
        }
        for t in 0..<T {
            for h in 0..<vh {
                abPtr[t * 2 * vh + h] = (h >= 64) ? 10.0 : 0.25
                abPtr[t * 2 * vh + vh + h] = Float(h) * 0.01
            }
        }

        let (cb, enc) = try XCTUnwrap(hal.begin())
        try conv.encodePrefillWithHeadnorm(
            enc, x: x, w: w, bias: nil, stIn: stIn, y: y, stOut: stOut, T: T,
            gdnY32: ab32, aLog: aLog, dtBias: dtBias, betas: betas, glog: glog
        )
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)

        let bOut = arena.cpuPointer(of: betas).assumingMemoryBound(to: Float.self)
        let gOut = arena.cpuPointer(of: glog).assumingMemoryBound(to: Float.self)
        for h in 64..<vh {
            let idx = (T - 1) * vh + h
            XCTAssertTrue(bOut[idx].isFinite && bOut[idx] > 0.5)
            XCTAssertTrue(gOut[idx].isFinite)
            XCTAssertEqual(gOut[idx], -130.0, accuracy: 1e-3)
        }
    }
}
