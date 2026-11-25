// QWShortConv.swift — Causal & centered depthwise 1D ShortConv encoder (k=3..7).

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWShortConvError: Error, CustomStringConvertible {
    case badShape(String)
    public var description: String {
        switch self {
        case .badShape(let msg): return "QWShortConv: \(msg)"
        }
    }
}

private struct QWShortConvMetalParams {
    var seqLen: UInt32
    var channels: UInt32
    var kernelSize: UInt32
    var hasBias: UInt32
    var mediaPrefixLen: UInt32
    var hasStateIn: UInt32
    var hasStateOut: UInt32
}

private struct QWShortConvGatedMetalParams {
    var seqLen: UInt32
    var channels: UInt32
    var kernelSize: UInt32
    var hasBias: UInt32
}

/// Depthwise 1D convolution encoder for LFM ShortConv blocks (`k=3..7`).
public final class QWShortConv: @unchecked Sendable {
    private let hal: QWHal
    private let psoCausal: any MTLComputePipelineState
    private let psoCentered: any MTLComputePipelineState
    private let psoGatedCausal: any MTLComputePipelineState
    private let dummySlab: MTLBuffer

    public init(hal: QWHal) throws {
        self.hal = hal
        self.psoCausal = try hal.pipeline("qwg_shortconv_causal_f16")
        self.psoCentered = try hal.pipeline("qwg_shortconv_centered_f16")
        self.psoGatedCausal = try hal.pipeline("qwg_shortconv_gated_causal_f16")
        guard let slab = hal.device.makeBuffer(length: 64, options: .storageModeShared) else {
            throw QWShortConvError.badShape("failed to allocate ShortConv dummy slab")
        }
        memset(slab.contents(), 0, 64)
        self.dummySlab = slab
    }

    /// Causal depthwise 1D ShortConv (`x [T, D]`, `weight [D, K]`, optional `bias [D]`, optional `stateIn/Out [K-1, D]`).
    public func encodeCausal(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        weight: QWTensorValue,
        bias: QWTensorValue?,
        stateIn: QWTensorValue?,
        out: QWTensorValue,
        stateOut: QWTensorValue?,
        seqLen: Int,
        channels: Int,
        kernelSize: Int
    ) throws {
        guard seqLen > 0, channels > 0, (3...7).contains(kernelSize) else {
            throw QWShortConvError.badShape("invalid ShortConv geometry T=\(seqLen) D=\(channels) K=\(kernelSize)")
        }
        guard x.shape.dtype == .f16, weight.shape.dtype == .f16, out.shape.dtype == .f16,
              x.shape.elementCount == seqLen * channels,
              weight.shape.elementCount == channels * kernelSize,
              out.shape.elementCount == seqLen * channels else {
            throw QWShortConvError.badShape("x, weight, out must be f16 with matching shapes")
        }
        if let bias {
            guard bias.shape.dtype == .f16, bias.shape.elementCount == channels else {
                throw QWShortConvError.badShape("bias must be [\(channels)] f16")
            }
        }
        let stateRows = kernelSize - 1
        if let stateIn {
            guard stateIn.shape.dtype == .f16, stateIn.shape.elementCount == stateRows * channels else {
                throw QWShortConvError.badShape("stateIn must be [\(stateRows), \(channels)] f16")
            }
        }
        if let stateOut {
            guard stateOut.shape.dtype == .f16, stateOut.shape.elementCount == stateRows * channels else {
                throw QWShortConvError.badShape("stateOut must be [\(stateRows), \(channels)] f16")
            }
        }

        var params = QWShortConvMetalParams(
            seqLen: UInt32(seqLen),
            channels: UInt32(channels),
            kernelSize: UInt32(kernelSize),
            hasBias: bias != nil ? 1 : 0,
            mediaPrefixLen: 0,
            hasStateIn: stateIn != nil ? 1 : 0,
            hasStateOut: stateOut != nil ? 1 : 0
        )
        enc.setComputePipelineState(psoCausal)
        x.bind(enc, index: 0)
        weight.bind(enc, index: 1)
        if let bias { bias.bind(enc, index: 2) } else { enc.setBuffer(dummySlab, offset: 0, index: 2) }
        if let stateIn { stateIn.bind(enc, index: 3) } else { enc.setBuffer(dummySlab, offset: 0, index: 3) }
        out.bind(enc, index: 4)
        if let stateOut { stateOut.bind(enc, index: 5) } else { enc.setBuffer(dummySlab, offset: 0, index: 5) }
        enc.setBytes(&params, length: MemoryLayout<QWShortConvMetalParams>.stride, index: 6)
        let tgW = min(channels, 64)
        let tgH = min(seqLen, 4)
        enc.dispatchThreads(
            MTLSize(width: channels, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
        )
    }

    /// Centered bidirectional ShortConv (`K` odd in `3...7`) with strict `mediaPrefixLen` boundary isolation.
    public func encodeCentered(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        weight: QWTensorValue,
        bias: QWTensorValue?,
        out: QWTensorValue,
        seqLen: Int,
        channels: Int,
        kernelSize: Int,
        mediaPrefixLen: Int = 0
    ) throws {
        guard seqLen > 0, channels > 0, [3, 5, 7].contains(kernelSize),
              mediaPrefixLen >= 0, mediaPrefixLen <= seqLen else {
            throw QWShortConvError.badShape("invalid centered ShortConv T=\(seqLen) D=\(channels) K=\(kernelSize) media=\(mediaPrefixLen)")
        }
        guard x.shape.dtype == .f16, weight.shape.dtype == .f16, out.shape.dtype == .f16,
              x.shape.elementCount == seqLen * channels,
              weight.shape.elementCount == channels * kernelSize,
              out.shape.elementCount == seqLen * channels else {
            throw QWShortConvError.badShape("centered ShortConv shape/dtype mismatch")
        }
        if let bias {
            guard bias.shape.dtype == .f16, bias.shape.elementCount == channels else {
                throw QWShortConvError.badShape("bias must be [\(channels)] f16")
            }
        }

        var params = QWShortConvMetalParams(
            seqLen: UInt32(seqLen),
            channels: UInt32(channels),
            kernelSize: UInt32(kernelSize),
            hasBias: bias != nil ? 1 : 0,
            mediaPrefixLen: UInt32(mediaPrefixLen),
            hasStateIn: 0,
            hasStateOut: 0
        )
        enc.setComputePipelineState(psoCentered)
        x.bind(enc, index: 0)
        weight.bind(enc, index: 1)
        if let bias { bias.bind(enc, index: 2) } else { enc.setBuffer(dummySlab, offset: 0, index: 2) }
        out.bind(enc, index: 3)
        enc.setBytes(&params, length: MemoryLayout<QWShortConvMetalParams>.stride, index: 4)
        let tgW = min(channels, 64)
        let tgH = min(seqLen, 4)
        enc.dispatchThreads(
            MTLSize(width: channels, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
        )
    }

    /// Gated Lfm3ShortConv: `bcx [T, 3*D]` (`B|C|x`), `weight [D, K]`, optional `bias [D]`, `out [T, D]`.
    public func encodeGatedCausal(
        _ enc: any MTLComputeCommandEncoder,
        bcx: QWTensorValue,
        weight: QWTensorValue,
        bias: QWTensorValue?,
        out: QWTensorValue,
        seqLen: Int,
        channels: Int,
        kernelSize: Int
    ) throws {
        guard seqLen > 0, channels > 0, (1...15).contains(kernelSize) else {
            throw QWShortConvError.badShape("invalid gated ShortConv T=\(seqLen) D=\(channels) K=\(kernelSize)")
        }
        guard bcx.shape.dtype == .f16, weight.shape.dtype == .f16, out.shape.dtype == .f16,
              bcx.shape.elementCount == seqLen * channels * 3,
              weight.shape.elementCount == channels * kernelSize,
              out.shape.elementCount == seqLen * channels else {
            throw QWShortConvError.badShape("gated ShortConv bcx/weight/out shape or dtype mismatch")
        }
        if let bias {
            guard bias.shape.dtype == .f16, bias.shape.elementCount == channels else {
                throw QWShortConvError.badShape("bias must be [\(channels)] f16")
            }
        }
        var params = QWShortConvGatedMetalParams(
            seqLen: UInt32(seqLen),
            channels: UInt32(channels),
            kernelSize: UInt32(kernelSize),
            hasBias: bias != nil ? 1 : 0
        )
        enc.setComputePipelineState(psoGatedCausal)
        bcx.bind(enc, index: 0)
        weight.bind(enc, index: 1)
        if let bias { bias.bind(enc, index: 2) } else { enc.setBuffer(dummySlab, offset: 0, index: 2) }
        out.bind(enc, index: 3)
        enc.setBytes(&params, length: MemoryLayout<QWShortConvGatedMetalParams>.stride, index: 4)
        let tgW = min(channels, 64)
        let tgH = min(seqLen, 4)
        enc.dispatchThreads(
            MTLSize(width: channels, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
        )
    }
}
