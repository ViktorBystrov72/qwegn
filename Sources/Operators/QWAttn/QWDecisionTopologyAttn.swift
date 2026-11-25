// QWDecisionTopologyAttn.swift — Generalized attention topology & head-wise sigmoid gate encoder.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

private struct QWTopoAttnMetalParams {
    var seqLen: UInt32
    var qHeads: UInt32
    var kvHeads: UInt32
    var headDim: UInt32
    var topologyMode: UInt32
    var stateEndPos: UInt32
    var mediaPrefixLen: UInt32
    var slidingWindow: UInt32
    var scale: Float
}

private struct QWSigmoidGateMetalParams {
    var seqLen: UInt32
    var qHeads: UInt32
    var headDim: UInt32
}

/// Evaluates arbitrary decision attention topologies and optional head-wise sigmoid gates on Metal.
public final class QWDecisionTopologyAttn: @unchecked Sendable {
    private let hal: QWHal
    private let psoPrefill: any MTLComputePipelineState
    private let psoPrefillNAX64: (any MTLComputePipelineState)?
    private let psoPrefillNAX128: (any MTLComputePipelineState)?
    private let psoSigmoidGate: any MTLComputePipelineState

    public init(hal: QWHal) throws {
        self.hal = hal
        self.psoPrefill = try hal.pipeline("qwg_attn_topology_prefill_f16")
        self.psoSigmoidGate = try hal.pipeline("qwg_attn_headwise_sigmoid_gate_f16")
        if hal.caps.isMetal4TensorCapable {
            self.psoPrefillNAX64 = try hal.pipeline("qwg_attn_topology_prefill_nax_d64")
            self.psoPrefillNAX128 = try hal.pipeline("qwg_attn_topology_prefill_nax_d128")
        } else {
            self.psoPrefillNAX64 = nil
            self.psoPrefillNAX128 = nil
        }
    }

    public static func topologyCode(for topology: QWDecisionModelConfig.AttentionTopology) -> UInt32 {
        switch topology {
        case .causal: return 0
        case .prefixBidirectional: return 1
        case .promptBidirectional: return 2
        case .packedSet: return 3
        case .bidirectionalEncoder: return 4
        }
    }

    public func encodePrefill(
        _ enc: any MTLComputeCommandEncoder,
        q: QWTensorValue,
        k: QWTensorValue,
        v: QWTensorValue,
        groupIds: QWTensorValue,
        out: QWTensorValue,
        seqLen: Int,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        topology: QWDecisionModelConfig.AttentionTopology,
        stateEndPos: Int = 0,
        mediaPrefixLen: Int = 0,
        slidingWindow: Int = 0,
        scale: Float? = nil,
        forceScalar: Bool = false
    ) throws {
        guard seqLen > 0, qHeads > 0, kvHeads > 0, headDim > 0, headDim <= 256,
              qHeads % kvHeads == 0, slidingWindow >= 0,
              q.shape.dtype == .f16, k.shape.dtype == .f16, v.shape.dtype == .f16,
              out.shape.dtype == .f16, groupIds.shape.dtype == .i32,
              q.shape.elementCount == seqLen * qHeads * headDim,
              out.shape.elementCount == seqLen * qHeads * headDim,
              k.shape.elementCount == seqLen * kvHeads * headDim,
              v.shape.elementCount == seqLen * kvHeads * headDim,
              groupIds.shape.elementCount == seqLen else {
            throw QWAttnError.badShape("topology attention prefill shape/dtype mismatch")
        }
        let effectiveMediaPrefixLen: Int = switch topology {
        case .bidirectionalEncoder(let isolate):
            isolate ? max(0, mediaPrefixLen) : 0
        default:
            max(0, mediaPrefixLen)
        }
        var params = QWTopoAttnMetalParams(
            seqLen: UInt32(seqLen),
            qHeads: UInt32(qHeads),
            kvHeads: UInt32(kvHeads),
            headDim: UInt32(headDim),
            topologyMode: Self.topologyCode(for: topology),
            stateEndPos: UInt32(max(0, stateEndPos)),
            mediaPrefixLen: UInt32(effectiveMediaPrefixLen),
            slidingWindow: UInt32(max(0, slidingWindow)),
            scale: scale ?? (1.0 / sqrtf(Float(headDim)))
        )
        let naxPSO: (any MTLComputePipelineState)?
        if forceScalar {
            naxPSO = nil
        } else {
            switch headDim {
            case 64: naxPSO = psoPrefillNAX64
            case 128: naxPSO = psoPrefillNAX128
            default: naxPSO = nil
            }
        }
        if let naxPSO {
            enc.setComputePipelineState(naxPSO)
            q.bind(enc, index: 0)
            k.bind(enc, index: 1)
            v.bind(enc, index: 2)
            groupIds.bind(enc, index: 3)
            out.bind(enc, index: 4)
            enc.setBytes(&params, length: MemoryLayout<QWTopoAttnMetalParams>.stride, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: qHeads, height: (seqLen + 31) / 32, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            return
        }
        enc.setComputePipelineState(psoPrefill)
        q.bind(enc, index: 0)
        k.bind(enc, index: 1)
        v.bind(enc, index: 2)
        groupIds.bind(enc, index: 3)
        out.bind(enc, index: 4)
        enc.setBytes(&params, length: MemoryLayout<QWTopoAttnMetalParams>.stride, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: qHeads, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    public func encodeHeadwiseSigmoidGate(
        _ enc: any MTLComputeCommandEncoder,
        out: QWTensorValue,
        gate: QWTensorValue,
        seqLen: Int,
        qHeads: Int,
        headDim: Int
    ) throws {
        guard seqLen > 0, qHeads > 0, headDim > 0,
              out.shape.dtype == .f16, gate.shape.dtype == .f16,
              out.shape.elementCount == seqLen * qHeads * headDim,
              gate.shape.elementCount == seqLen * qHeads else {
            throw QWAttnError.badShape("headwise sigmoid gate shape/dtype mismatch")
        }
        var params = QWSigmoidGateMetalParams(
            seqLen: UInt32(seqLen),
            qHeads: UInt32(qHeads),
            headDim: UInt32(headDim)
        )
        enc.setComputePipelineState(psoSigmoidGate)
        out.bind(enc, index: 0)
        gate.bind(enc, index: 1)
        enc.setBytes(&params, length: MemoryLayout<QWSigmoidGateMetalParams>.stride, index: 2)
        let tgX = min(headDim, 32)
        let tgY = min(qHeads, 4)
        enc.dispatchThreads(
            MTLSize(width: headDim, height: qHeads, depth: seqLen),
            threadsPerThreadgroup: MTLSize(width: tgX, height: tgY, depth: 1)
        )
    }
}
