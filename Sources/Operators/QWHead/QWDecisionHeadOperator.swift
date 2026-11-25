// QWDecisionHeadOperator.swift — Metal encoders for sparse BF16 vocab gather, marker/classifier MLPs, pooling, and pointers.

import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

public enum QWDecisionHeadActivation: UInt32, Sendable {
    case none = 0
    case gelu = 1
    case silu = 2
}

private struct QWSparseVocabMetalParams {
    var rows: UInt32
    var candCount: UInt32
    var hidden: UInt32
    var vocabSize: UInt32
}

private struct QWMarkerGatherMetalParams {
    var count: UInt32
    var hidden: UInt32
    var seqLen: UInt32
    var hasTypeEmb: UInt32
}

private struct QWDecisionLinearMetalParams {
    var rows: UInt32
    var outFeatures: UInt32
    var inFeatures: UInt32
    var hasBias: UInt32
    var actMode: UInt32
    var xIsFloat: UInt32
}

private struct QWDecisionPoolMetalParams {
    var seqLen: UInt32
    var hidden: UInt32
    var validLen: UInt32
    var poolMode: UInt32
}

private struct QWPointerMetalParams {
    var stateLen: UInt32
    var hidden: UInt32
    var invSqrtH: Float
}

/// GPU readout operator supporting all M3 System-1 decision heads cleanly.
public final class QWDecisionHeadOperator: @unchecked Sendable {
    private let hal: QWHal
    private let psoSparseVocabBF16: any MTLComputePipelineState
    private let psoMarkerGather: any MTLComputePipelineState
    private let psoLinearF32: any MTLComputePipelineState
    private let psoPoolF16: any MTLComputePipelineState
    private let psoPointerScoresF32: any MTLComputePipelineState
    private let dummySlab: MTLBuffer

    public init(hal: QWHal) throws {
        self.hal = hal
        self.psoSparseVocabBF16 = try hal.pipeline("qwg_head_sparse_vocab_rows_bf16")
        self.psoMarkerGather = try hal.pipeline("qwg_decision_marker_gather_f16")
        self.psoLinearF32 = try hal.pipeline("qwg_decision_mlp_linear_f32")
        self.psoPoolF16 = try hal.pipeline("qwg_decision_pool_f16")
        self.psoPointerScoresF32 = try hal.pipeline("qwg_decision_pointer_scores_f32")
        guard let slab = hal.device.makeBuffer(length: 64, options: .storageModeShared) else {
            throw QWHeadError.badShape("failed to allocate dummy decision head slab")
        }
        memset(slab.contents(), 0, 64)
        self.dummySlab = slab
    }

    /// Evaluates only `candCount` BF16 `lm_head` rows (`U <= 256` sparse gather, ADR-006).
    public func encodeSparseVocabRowsBF16(
        _ enc: any MTLComputeCommandEncoder,
        hiddenState: QWTensorValue,
        lmHeadBF16: QWTensorValue,
        candidateIds: QWTensorValue,
        out: QWTensorValue,
        rows: Int,
        candCount: Int,
        hidden: Int,
        vocabSize: Int
    ) throws {
        guard rows > 0, candCount > 0, hidden > 0, vocabSize > 0,
              hiddenState.shape.dtype == .f16, lmHeadBF16.shape.dtype == .bf16,
              candidateIds.shape.dtype == .i32, out.shape.dtype == .f32,
              hiddenState.shape.elementCount == rows * hidden,
              lmHeadBF16.shape.elementCount >= vocabSize * hidden,
              candidateIds.shape.elementCount == candCount,
              out.shape.elementCount >= rows * candCount else {
            throw QWHeadError.badShape("sparse BF16 vocab head tensor shape/dtype mismatch")
        }
        var params = QWSparseVocabMetalParams(
            rows: UInt32(rows),
            candCount: UInt32(candCount),
            hidden: UInt32(hidden),
            vocabSize: UInt32(vocabSize)
        )
        enc.setComputePipelineState(psoSparseVocabBF16)
        hiddenState.bind(enc, index: 0)
        lmHeadBF16.bind(enc, index: 1)
        candidateIds.bind(enc, index: 2)
        out.bind(enc, index: 3)
        enc.setBytes(&params, length: MemoryLayout<QWSparseVocabMetalParams>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: candCount, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    public func encodeMarkerGather(
        _ enc: any MTLComputeCommandEncoder,
        hiddenState: QWTensorValue,
        positions: QWTensorValue,
        typeEmb: QWTensorValue?,
        out: QWTensorValue,
        seqLen: Int,
        count: Int,
        hidden: Int
    ) throws {
        guard seqLen > 0, count > 0, hidden > 0,
              hiddenState.shape.dtype == .f16, positions.shape.dtype == .i32, out.shape.dtype == .f16,
              hiddenState.shape.elementCount == seqLen * hidden,
              positions.shape.elementCount == count,
              out.shape.elementCount == count * hidden else {
            throw QWHeadError.badShape("marker gather shape/dtype mismatch")
        }
        if let typeEmb {
            guard typeEmb.shape.dtype == .f16, typeEmb.shape.elementCount == hidden else {
                throw QWHeadError.badShape("marker typeEmb must be [\(hidden)] f16")
            }
        }
        var params = QWMarkerGatherMetalParams(
            count: UInt32(count),
            hidden: UInt32(hidden),
            seqLen: UInt32(seqLen),
            hasTypeEmb: typeEmb != nil ? 1 : 0
        )
        enc.setComputePipelineState(psoMarkerGather)
        hiddenState.bind(enc, index: 0)
        positions.bind(enc, index: 1)
        if let typeEmb { typeEmb.bind(enc, index: 2) } else { enc.setBuffer(dummySlab, offset: 0, index: 2) }
        out.bind(enc, index: 3)
        enc.setBytes(&params, length: MemoryLayout<QWMarkerGatherMetalParams>.stride, index: 4)
        let tgW = min(hidden, 64)
        let tgH = min(count, 4)
        enc.dispatchThreads(
            MTLSize(width: hidden, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
        )
    }

    public func encodeDecisionLinear(
        _ enc: any MTLComputeCommandEncoder,
        x: QWTensorValue,
        weight: QWTensorValue,
        bias: QWTensorValue?,
        out: QWTensorValue,
        rows: Int,
        inFeatures: Int,
        outFeatures: Int,
        activation: QWDecisionHeadActivation = .none
    ) throws {
        let isFloat = (x.shape.dtype == .f32)
        guard rows > 0, inFeatures > 0, outFeatures > 0,
              (x.shape.dtype == .f16 || x.shape.dtype == .f32),
              weight.shape.dtype == .f16, out.shape.dtype == .f32,
              x.shape.elementCount == rows * inFeatures,
              weight.shape.elementCount == outFeatures * inFeatures,
              out.shape.elementCount >= rows * outFeatures else {
            throw QWHeadError.badShape("decision linear shape/dtype mismatch")
        }
        if let bias {
            guard bias.shape.dtype == .f16, bias.shape.elementCount == outFeatures else {
                throw QWHeadError.badShape("decision linear bias must be [\(outFeatures)] f16")
            }
        }
        var params = QWDecisionLinearMetalParams(
            rows: UInt32(rows),
            outFeatures: UInt32(outFeatures),
            inFeatures: UInt32(inFeatures),
            hasBias: bias != nil ? 1 : 0,
            actMode: activation.rawValue,
            xIsFloat: isFloat ? 1 : 0
        )
        enc.setComputePipelineState(psoLinearF32)
        if isFloat {
            enc.setBuffer(dummySlab, offset: 0, index: 0)
            x.bind(enc, index: 1)
        } else {
            x.bind(enc, index: 0)
            enc.setBuffer(dummySlab, offset: 0, index: 1)
        }
        weight.bind(enc, index: 2)
        if let bias { bias.bind(enc, index: 3) } else { enc.setBuffer(dummySlab, offset: 0, index: 3) }
        out.bind(enc, index: 4)
        enc.setBytes(&params, length: MemoryLayout<QWDecisionLinearMetalParams>.stride, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: outFeatures, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    public func encodePool(
        _ enc: any MTLComputeCommandEncoder,
        hiddenState: QWTensorValue,
        out: QWTensorValue,
        seqLen: Int,
        validLen: Int,
        hidden: Int,
        pooling: QWDecisionModelConfig.PoolingKind
    ) throws {
        guard seqLen > 0, validLen > 0, hidden > 0,
              hiddenState.shape.dtype == .f16, out.shape.dtype == .f16,
              hiddenState.shape.elementCount == seqLen * hidden,
              out.shape.elementCount == hidden else {
            throw QWHeadError.badShape("decision pool shape/dtype mismatch")
        }
        let mode: UInt32 = switch pooling {
        case .firstToken: 0
        case .lastToken: 1
        case .meanNonPad: 2
        }
        var params = QWDecisionPoolMetalParams(
            seqLen: UInt32(seqLen),
            hidden: UInt32(hidden),
            validLen: UInt32(min(seqLen, validLen)),
            poolMode: mode
        )
        enc.setComputePipelineState(psoPoolF16)
        hiddenState.bind(enc, index: 0)
        out.bind(enc, index: 1)
        enc.setBytes(&params, length: MemoryLayout<QWDecisionPoolMetalParams>.stride, index: 2)
        enc.dispatchThreads(
            MTLSize(width: hidden, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(hidden, 64), height: 1, depth: 1)
        )
    }

    public func encodePointerScores(
        _ enc: any MTLComputeCommandEncoder,
        stateHidden: QWTensorValue,
        qStart: QWTensorValue,
        qEnd: QWTensorValue,
        outStart: QWTensorValue,
        outEnd: QWTensorValue,
        stateLen: Int,
        hidden: Int
    ) throws {
        guard stateLen > 0, hidden > 0,
              stateHidden.shape.dtype == .f16, qStart.shape.dtype == .f16, qEnd.shape.dtype == .f16,
              outStart.shape.dtype == .f32, outEnd.shape.dtype == .f32,
              stateHidden.shape.elementCount == stateLen * hidden,
              qStart.shape.elementCount == hidden, qEnd.shape.elementCount == hidden,
              outStart.shape.elementCount >= stateLen, outEnd.shape.elementCount >= stateLen else {
            throw QWHeadError.badShape("state pointer shape/dtype mismatch")
        }
        var params = QWPointerMetalParams(
            stateLen: UInt32(stateLen),
            hidden: UInt32(hidden),
            invSqrtH: 1.0 / sqrtf(Float(hidden))
        )
        enc.setComputePipelineState(psoPointerScoresF32)
        stateHidden.bind(enc, index: 0)
        qStart.bind(enc, index: 1)
        qEnd.bind(enc, index: 2)
        outStart.bind(enc, index: 3)
        outEnd.bind(enc, index: 4)
        enc.setBytes(&params, length: MemoryLayout<QWPointerMetalParams>.stride, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: stateLen, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
}
