// QWNorm+Decision.swift — Generalized LayerNorm & sandwich post-RMSNorm residual add.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

extension QWNorm {
    /// Encodes arbitrary-width FP16 `LayerNorm` with FP32 mean/variance reduction.
    public static func encodeLayerNorm(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        x: QWTensorValue,
        gamma: QWTensorValue,
        beta: QWTensorValue?,
        out: QWTensorValue,
        rows: Int,
        hidden: Int,
        eps: Float = 1e-5
    ) throws {
        guard rows > 0, hidden > 0,
              x.shape.dtype == .f16, gamma.shape.dtype == .f16, out.shape.dtype == .f16,
              x.shape.elementCount == rows * hidden,
              out.shape.elementCount == rows * hidden,
              gamma.shape.elementCount == hidden else {
            throw QWNormError.badShape("LayerNorm requires [rows, hidden] f16 tensors")
        }
        if let beta {
            guard beta.shape.dtype == .f16, beta.shape.elementCount == hidden else {
                throw QWNormError.badShape("LayerNorm beta must be [\(hidden)] f16")
            }
        }
        let pso = try hal.pipeline("qwg_layernorm_f16")
        var hiddenU = UInt32(hidden)
        var epsVal = eps
        var hasBiasU: UInt32 = beta != nil ? 1 : 0
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        (beta ?? gamma).bind(enc, index: 2)
        out.bind(enc, index: 3)
        enc.setBytes(&hiddenU, length: 4, index: 4)
        enc.setBytes(&epsVal, length: 4, index: 5)
        enc.setBytes(&hasBiasU, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    /// Encodes fused sandwich post-RMSNorm + residual add (`residual += rmsnorm(branch)`).
    public static func encodePostRMSNormResidualAdd(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        residual: QWTensorValue,
        branch: QWTensorValue,
        gamma: QWTensorValue,
        rows: Int,
        hidden: Int,
        eps: Float = 1e-6,
        hasUnitOffset: Bool = false
    ) throws {
        guard rows > 0, hidden > 0,
              residual.shape.dtype == .f16, branch.shape.dtype == .f16, gamma.shape.dtype == .f16,
              residual.shape.elementCount == rows * hidden,
              branch.shape.elementCount == rows * hidden,
              gamma.shape.elementCount == hidden else {
            throw QWNormError.badShape("postRMSNormResidualAdd requires [rows, hidden] f16 tensors")
        }
        let pso = try hal.pipeline("qwg_post_rmsnorm_residual_add_f16")
        var hiddenU = UInt32(hidden)
        var epsVal = eps
        var unitOffset: Float = hasUnitOffset ? 1.0 : 0.0
        enc.setComputePipelineState(pso)
        residual.bind(enc, index: 0)
        branch.bind(enc, index: 1)
        gamma.bind(enc, index: 2)
        enc.setBytes(&hiddenU, length: 4, index: 3)
        enc.setBytes(&epsVal, length: 4, index: 4)
        enc.setBytes(&unitOffset, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    /// Encodes arbitrary-width unrotated FP16 `RMSNorm` with FP32 variance reduction.
    public static func encodeArbitraryRMSNorm(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        x: QWTensorValue,
        gamma: QWTensorValue,
        out: QWTensorValue,
        rows: Int,
        hidden: Int,
        eps: Float = 1e-6,
        hasUnitOffset: Bool = false
    ) throws {
        guard rows > 0, hidden > 0,
              x.shape.dtype == .f16, gamma.shape.dtype == .f16, out.shape.dtype == .f16,
              x.shape.elementCount == rows * hidden,
              out.shape.elementCount == rows * hidden,
              gamma.shape.elementCount == hidden else {
            throw QWNormError.badShape("arbitraryRMSNorm requires [rows, hidden] f16 tensors")
        }
        let pso = try hal.pipeline("qwg_rmsnorm_f16")
        var hiddenU = UInt32(hidden)
        var epsVal = eps
        var unitOffset: Float = hasUnitOffset ? 1.0 : 0.0
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        gamma.bind(enc, index: 1)
        out.bind(enc, index: 2)
        enc.setBytes(&hiddenU, length: 4, index: 3)
        enc.setBytes(&epsVal, length: 4, index: 4)
        enc.setBytes(&unitOffset, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
}
