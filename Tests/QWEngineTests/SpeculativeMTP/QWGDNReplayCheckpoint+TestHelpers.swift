// QWGDNReplayCheckpoint+TestHelpers.swift — Test-only single-row capture helpers relocated from QWGDNReplayCheckpoint.
import Metal
import QWAlloc
import QWCore
import QWHAL
@testable import QWEngine

extension QWGDNReplayCheckpoint {
    func encodeCapture(_ enc: any MTLBlitCommandEncoder, layer: Int, row: Int,
                       rawInput: QWTensorValue, postConv: QWTensorValue,
                       beta: QWTensorValue, decay: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil, !outOfPlaceVerifyActive,
              (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU replay capture index is invalid")
        }
        resetGPUBatchedStability()
        let index = layer * rows + row
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        let r = view(raw, index: index, count: qkvC)
        let n = view(normalized, index: index, count: qkvC)
        let b = view(betas, index: index, count: gdnVH)
        let g = view(glog, index: index, count: glogC)
        try validate(rawInput, like: r)
        try validate(postConv, like: n)
        try validate(beta, like: b)
        try validate(decay, like: g)
        copy(enc, rawInput, r)
        copy(enc, postConv, n)
        copy(enc, beta, b)
        copy(enc, decay, g)
        captured[layer] |= UInt32(1 << row)
    }

    func encodeCapture(_ enc: any MTLComputeCommandEncoder, hal: QWHal, layer: Int, row: Int,
                       rawInput: QWTensorValue, postConv: QWTensorValue,
                       beta: QWTensorValue, decay: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil, !outOfPlaceVerifyActive,
              (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU replay capture index is invalid")
        }
        resetGPUBatchedStability()
        let index = layer * rows + row
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        let r = view(raw, index: index, count: qkvC)
        let n = view(normalized, index: index, count: qkvC)
        let b = view(betas, index: index, count: gdnVH)
        let g = view(glog, index: index, count: glogC)
        try validate(rawInput, like: r)
        try validate(postConv, like: n)
        try validate(beta, like: b)
        try validate(decay, like: g)
        try QWVerifyWindow.encodeCopy(enc, hal: hal, source: rawInput, destination: r)
        try QWVerifyWindow.encodeCopy(enc, hal: hal, source: postConv, destination: n)
        try QWVerifyWindow.encodeCopy(enc, hal: hal, source: beta, destination: b)
        try QWVerifyWindow.encodeCopy(enc, hal: hal, source: decay, destination: g)
        captured[layer] |= UInt32(1 << row)
    }
}
