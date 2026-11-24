import Metal
import QWAlloc
import QWCore
import QWGDN
import QWHAL

/// Single-flight GPU workspace. The round owner must finish capture before
/// scratch reuse and check GPU completion before publishing restored metadata.
final class QWGDNReplayCheckpoint {
    let geometry: QWModelGeometry
    let layers: Int
    let rows: Int
    let recurrent: QWTensorValue
    let rings: QWTensorValue
    let raw: QWTensorValue
    let normalized: QWTensorValue
    let betas: QWTensorValue
    let glog: QWTensorValue
    let innovations: QWTensorValue
    let stabilityMask: QWTensorValue
    var sourceRec: QWTensorValue?
    var sourceRing: QWTensorValue?
    var captured: [UInt32]
    private(set) var outOfPlaceVerifyActive: Bool = false
    private var batchedLayerRecorded: [Bool]
    private var batchedLayerCount: Int = 0
    private var batchedActiveRows: Int = 0

    static func allocationBytes(geometry: QWModelGeometry, layers: Int, rows: Int) -> Int {
        precondition((0...128).contains(layers) && (1...16).contains(rows))
        if layers == 0 { return 8 * 256 }
        func aligned(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        return aligned(layers * QWSlotGeometry.recElems(geometry: geometry) * 4)
            + aligned(layers * QWSlotGeometry.ringElems(geometry: geometry) * 2)
            + 2 * aligned(layers * rows * qkvC * 2)
            + aligned(layers * rows * gdnVH * 4)
            + aligned(layers * rows * glogC * 4)
            + aligned(layers * rows * zC * 4)
            + 256
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, layers: Int = 48, rows: Int) throws {
        self.geometry = geometry
        guard layers >= 0, layers <= 128, (1...16).contains(rows) else {
            throw QWEngineDriverError.state("GPU checkpoint requires 0...128 layers and 1...16 rows")
        }
        self.layers = layers
        self.rows = rows
        captured = [UInt32](repeating: 0, count: layers)
        batchedLayerRecorded = [Bool](repeating: false, count: layers)
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        let recElems = max(1, layers * QWSlotGeometry.recElems(geometry: geometry))
        let ringElems = max(1, layers * QWSlotGeometry.ringElems(geometry: geometry))
        let rowQKV = max(1, layers * rows * qkvC)
        let rowVH = max(1, layers * rows * gdnVH)
        let rowGlog = max(1, layers * rows * glogC)
        let rowZ = max(1, layers * rows * zC)
        recurrent = try allocator.allocate(QWTensorShape([recElems], .f32))
        rings = try allocator.allocate(QWTensorShape([ringElems], .f16))
        raw = try allocator.allocate(QWTensorShape([rowQKV], .f16))
        normalized = try allocator.allocate(QWTensorShape([rowQKV], .f16))
        betas = try allocator.allocate(QWTensorShape([rowVH], .f32))
        glog = try allocator.allocate(QWTensorShape([rowGlog], .f32))
        innovations = try allocator.allocate(QWTensorShape([rowZ], .f32))
        stabilityMask = try allocator.allocate(QWTensorShape([1], .i32))
    }

    func view(_ tensor: QWTensorValue, index: Int, count: Int) -> QWTensorValue {
        let width = tensor.shape.dtype == .f32 ? 4 : 2
        return QWTensorValue(region: tensor.region, offset: tensor.offset + index * count * width,
                             shape: QWTensorShape([count], tensor.shape.dtype))
    }

    private func slice(_ tensor: QWTensorValue, startRow: Int, rowCount: Int, rowWidth: Int) -> QWTensorValue {
        let width = tensor.shape.dtype == .f32 ? 4 : 2
        return QWTensorValue(region: tensor.region, offset: tensor.offset + startRow * rowWidth * width,
                             shape: QWTensorShape([rowCount * rowWidth], tensor.shape.dtype))
    }

    func validate(_ t: QWTensorValue, like other: QWTensorValue) throws {
        guard t.shape.dtype == other.shape.dtype, t.bytes == other.bytes,
              t.offset >= 0, t.region.base >= 0, t.offset <= t.region.length,
              t.bytes <= t.region.length - t.offset,
              t.region.base <= t.region.buffer.length,
              t.region.length <= t.region.buffer.length - t.region.base,
              (t.region.base + t.offset) % 4 == 0 else {
            throw QWEngineDriverError.state("GPU checkpoint tensor shape or extent is invalid")
        }
        if t.region.buffer === recurrent.region.buffer {
            let start = t.region.base + t.offset
            let end = start + t.bytes
            let workspaceStart = recurrent.region.base + recurrent.offset
            let workspaceEnd = stabilityMask.region.base + stabilityMask.offset + stabilityMask.bytes
            guard end <= workspaceStart || workspaceEnd <= start else {
                throw QWEngineDriverError.state("GPU checkpoint source aliases workspace")
            }
        }
    }

    private func sameStorage(_ a: QWTensorValue, _ b: QWTensorValue?) -> Bool {
        guard let b else { return false }
        return a.region.buffer === b.region.buffer
            && a.region.base + a.offset == b.region.base + b.offset && a.bytes == b.bytes
    }

    func validateRetained(_ count: Int) throws {
        guard sourceRec != nil, sourceRing != nil, (0...rows).contains(count) else {
            throw QWEngineDriverError.state("GPU replay has no checkpoint or retained row count is invalid")
        }
        let required = UInt32((1 << count) - 1)
        for layer in 0..<layers where captured[layer] & required != required {
            throw QWEngineDriverError.state("GPU replay journal is incomplete")
        }
    }

    func validateCompleteCapture(activeRows: Int? = nil) throws {
        try validateRetained(activeRows ?? rows)
    }

    func validateResolution(rec: QWTensorValue, ring: QWTensorValue,
                            activeRows: Int? = nil) throws {
        if layers == 0 {
            try validateCompleteCapture(activeRows: activeRows)
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        try validateCompleteCapture(activeRows: activeRows)
        guard sameStorage(rec, sourceRec), sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU resolution requires original checkpoint state buffers")
        }
    }

    func copy(_ enc: any MTLBlitCommandEncoder, _ source: QWTensorValue,
              _ destination: QWTensorValue) {
        enc.copy(from: source.region.buffer, sourceOffset: source.region.base + source.offset,
                 to: destination.region.buffer, destinationOffset: destination.region.base + destination.offset,
                 size: source.bytes)
    }

    func resetGPUBatchedStability() {
        batchedLayerCount = 0
        batchedActiveRows = 0
        for layer in 0..<layers { batchedLayerRecorded[layer] = false }
    }

    private func prepareBatchedVerifyTarget(
        layer: Int, activeRows: Int, state: QWTensorValue
    ) -> (stateOut: QWTensorValue, stabilityMask: QWTensorValue?) {
        if batchedLayerCount > 0 && batchedActiveRows != activeRows {
            resetGPUBatchedStability()
        }
        batchedActiveRows = activeRows
        if !batchedLayerRecorded[layer] {
            batchedLayerRecorded[layer] = true
            batchedLayerCount += 1
        }
        let recOut = outOfPlaceVerifyActive
            ? view(recurrent, index: layer, count: QWSlotGeometry.recElems(geometry: geometry))
            : state
        return (recOut, activeRows <= 16 ? stabilityMask : nil)
    }

    func markZeroLayerBackup(rec: QWTensorValue, ring: QWTensorValue) throws {
        guard layers == 0 else {
            throw QWEngineDriverError.state("zero-layer backup requires layers == 0")
        }
        sourceRec = rec
        sourceRing = ring
        outOfPlaceVerifyActive = false
        resetGPUBatchedStability()
    }

    func encodeBackup(_ enc: any MTLBlitCommandEncoder,
                      rec: QWTensorValue, ring: QWTensorValue,
                      skipRecCopy: Bool = false) throws {
        if layers == 0 {
            sourceRec = rec
            sourceRing = ring
            outOfPlaceVerifyActive = false
            resetGPUBatchedStability()
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        if rec.region.buffer === ring.region.buffer {
            let recStart = rec.region.base + rec.offset, ringStart = ring.region.base + ring.offset
            guard recStart + rec.bytes <= ringStart || ringStart + ring.bytes <= recStart else {
                throw QWEngineDriverError.state("GPU checkpoint recurrent and ring slabs alias")
            }
        }
        let maskStart = stabilityMask.region.base + stabilityMask.offset
        enc.fill(buffer: stabilityMask.region.buffer, range: maskStart..<(maskStart + stabilityMask.bytes), value: 0)
        if !skipRecCopy {
            copy(enc, rec, recurrent)
        }
        copy(enc, ring, rings)
        sourceRec = rec
        sourceRing = ring
        outOfPlaceVerifyActive = skipRecCopy
        resetGPUBatchedStability()
        for layer in 0..<layers { captured[layer] = 0 }
    }

    func encodeRestore(_ enc: any MTLBlitCommandEncoder,
                       rec: QWTensorValue, ring: QWTensorValue,
                       restoreRing: Bool = true) throws {
        if layers == 0 {
            guard sourceRec != nil, sourceRing != nil else {
                throw QWEngineDriverError.state("GPU restore requires the checkpoint's original state buffers")
            }
            return
        }
        try validate(rec, like: recurrent)
        try validate(ring, like: rings)
        guard sameStorage(rec, sourceRec), sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU restore requires the checkpoint's original state buffers")
        }
        if !outOfPlaceVerifyActive {
            copy(enc, recurrent, rec)
        }
        if restoreRing {
            copy(enc, rings, ring)
        }
    }

    func encodeCommitFullRec(_ enc: any MTLBlitCommandEncoder, rec: QWTensorValue) throws {
        guard layers > 0 else { return }
        try validate(rec, like: recurrent)
        guard sameStorage(rec, sourceRec) else {
            throw QWEngineDriverError.state("GPU full rec commit requires original recurrent slab")
        }
        if outOfPlaceVerifyActive {
            copy(enc, recurrent, rec)
        }
    }

    func encodeBatchedKDA(_ enc: any MTLComputeCommandEncoder,
                          conv: QWGDNConv, decoder: QWGDNDecode,
                          layer: Int, activeRows: Int,
                          gdnProjection: QWTensorValue, fgbProjection: QWTensorValue,
                          fBWeight: QWTensorValue, gBWeight: QWTensorValue,
                          aLog: QWTensorValue, dtBias: QWTensorValue,
                          convWeight: QWTensorValue, normWeight: QWTensorValue,
                          zScratch: QWTensorValue,
                          ring: QWTensorValue, state: QWTensorValue,
                          output: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("GPU batched KDA capture index or active row count is invalid")
        }
        let startRow = layer * rows
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        let r = slice(raw, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let n = slice(normalized, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let b = slice(betas, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let g = slice(glog, startRow: startRow, rowCount: activeRows, rowWidth: glogC)
        let inv = slice(innovations, startRow: startRow, rowCount: activeRows, rowWidth: zC)
        let xView = QWTensorValue(region: gdnProjection.region, offset: gdnProjection.offset,
                                  shape: QWTensorShape([activeRows * qkvC], .f16))
        let zView = QWTensorValue(region: zScratch.region, offset: zScratch.offset,
                                  shape: QWTensorShape([activeRows * zC], .f16))
        let recTarget = prepareBatchedVerifyTarget(layer: layer, activeRows: activeRows, state: state)
        try decoder.encodeKDAGates(
            enc, fgb: fgbProjection, fB: fBWeight, gB: gBWeight,
            aLog: aLog, dtBias: dtBias, betas: b, glog: g, zOut: zView, rows: activeRows)
        try conv.encodeRowsWithHeadnorm(
            enc, x: xView, w: convWeight, bias: nil,
            stIn: ring, y: n, stOut: ring,
            rows: activeRows, xStride: qkvC, rawCapture: r)
        enc.memoryBarrier(scope: .buffers)
        try decoder.encodeRows(
            enc, y: n, betas: b, glog: g, state: state,
            zBase: zView, w: normWeight, o: output, innovation: inv,
            rows: activeRows, zStride: zC,
            stateOut: recTarget.stateOut, unstableSuffixMask: recTarget.stabilityMask)
        enc.memoryBarrier(scope: .buffers)
        captured[layer] |= UInt32((1 << activeRows) - 1)
    }

    func encodeBatchedKDADirect(_ enc: any MTLComputeCommandEncoder,
                                conv: QWGDNConv, decoder: QWGDNDecode,
                                layer: Int, activeRows: Int,
                                gdnProjection: QWTensorValue, fgbProjection: QWTensorValue,
                                ln1: QWTensorValue, bProjF16: QWTensorValue?,
                                aLog: QWTensorValue, dtBias: QWTensorValue,
                                convWeight: QWTensorValue, normWeight: QWTensorValue,
                                zScratch: QWTensorValue,
                                ring: QWTensorValue, state: QWTensorValue,
                                output: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("GPU batched KDA capture index or active row count is invalid")
        }
        let startRow = layer * rows
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let glogC = geometry.gdnGlogChannels
        let r = slice(raw, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let n = slice(normalized, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let b = slice(betas, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let g = slice(glog, startRow: startRow, rowCount: activeRows, rowWidth: glogC)
        let inv = slice(innovations, startRow: startRow, rowCount: activeRows, rowWidth: zC)
        let xView = QWTensorValue(region: gdnProjection.region, offset: gdnProjection.offset,
                                  shape: QWTensorShape([activeRows * qkvC], .f16))
        let zView = QWTensorValue(region: zScratch.region, offset: zScratch.offset,
                                  shape: QWTensorShape([activeRows * zC], .f16))
        let recTarget = prepareBatchedVerifyTarget(layer: layer, activeRows: activeRows, state: state)
        try decoder.encodeKDAGatesDirect(
            enc, fgb: fgbProjection, ln1: ln1, bProjF16: bProjF16,
            aLog: aLog, dtBias: dtBias, betas: b, glog: g, zOut: zView, rows: activeRows)
        try conv.encodeRowsWithHeadnorm(
            enc, x: xView, w: convWeight, bias: nil,
            stIn: ring, y: n, stOut: ring,
            rows: activeRows, xStride: qkvC, rawCapture: r)
        enc.memoryBarrier(scope: .buffers)
        try decoder.encodeRows(
            enc, y: n, betas: b, glog: g, state: state,
            zBase: zView, w: normWeight, o: output, innovation: inv,
            rows: activeRows, zStride: zC,
            stateOut: recTarget.stateOut, unstableSuffixMask: recTarget.stabilityMask)
        enc.memoryBarrier(scope: .buffers)
        captured[layer] |= UInt32((1 << activeRows) - 1)
    }

    func encodeBatchedGDN(_ enc: any MTLComputeCommandEncoder,
                          conv: QWGDNConv, decoder: QWGDNDecode,
                          layer: Int, activeRows: Int,
                          gdnProjection: QWTensorValue, controls: QWTensorValue,
                          convWeight: QWTensorValue, normWeight: QWTensorValue,
                          ring: QWTensorValue, state: QWTensorValue,
                          output: QWTensorValue) throws {
        guard sourceRec != nil, sourceRing != nil,
              (0..<layers).contains(layer), (2...rows).contains(activeRows) else {
            throw QWEngineDriverError.state("GPU batched GDN capture index or active row count is invalid")
        }
        let startRow = layer * rows
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, zC = geometry.gdnZChannels
        let r = slice(raw, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let n = slice(normalized, startRow: startRow, rowCount: activeRows, rowWidth: qkvC)
        let b = slice(betas, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let g = slice(glog, startRow: startRow, rowCount: activeRows, rowWidth: gdnVH)
        let inv = slice(innovations, startRow: startRow, rowCount: activeRows, rowWidth: zC)
        let xView = QWTensorValue(region: gdnProjection.region, offset: gdnProjection.offset,
                                  shape: QWTensorShape([activeRows * geometry.gdnTotalN], .f16))
        let cView = QWTensorValue(region: controls.region, offset: controls.offset,
                                  shape: QWTensorShape([activeRows * (2 * gdnVH)], .f32))
        let recTarget = prepareBatchedVerifyTarget(layer: layer, activeRows: activeRows, state: state)
        try conv.encodeRowsWithHeadnorm(
            enc, x: xView, w: convWeight, bias: nil,
            stIn: ring, y: n, stOut: ring,
            rows: activeRows, xStride: geometry.gdnTotalN,
            rawCapture: r, controls: cView, betasCapture: b, glogCapture: g)
        enc.memoryBarrier(scope: .buffers)
        let zSpan = (activeRows - 1) * geometry.gdnTotalN + zC
        let zBase = QWTensorValue(
            region: gdnProjection.region,
            offset: gdnProjection.offset + qkvC * 2,
            shape: QWTensorShape([zSpan], .f16))
        try decoder.encodeRows(
            enc, y: n, betas: b, glog: g, state: state,
            zBase: zBase, w: normWeight, o: output, innovation: inv,
            rows: activeRows, zStride: geometry.gdnTotalN,
            stateOut: recTarget.stateOut, unstableSuffixMask: recTarget.stabilityMask)
        enc.memoryBarrier(scope: .buffers)
        captured[layer] |= UInt32((1 << activeRows) - 1)
    }

    func encodeReplay(_ enc: any MTLComputeCommandEncoder, decoder: QWGDNDecode,
                      rec: QWTensorValue, retaining count: Int) throws {
        if layers == 0 {
            try validateRetained(count)
            return
        }
        try validate(rec, like: recurrent)
        try validateRetained(count)
        guard sameStorage(rec, sourceRec) else {
            throw QWEngineDriverError.state("GPU replay requires the checkpoint's original recurrent slab")
        }
        let qkvC = geometry.gdnQKVChannels, gdnVH = geometry.gdnVHeads, glogC = geometry.gdnGlogChannels
        for row in 0..<count {
            if row > 0 {
                enc.memoryBarrier(scope: .buffers)
            }
            for layer in 0..<layers {
                let state = view(rec, index: layer, count: QWSlotGeometry.recElems(geometry: geometry))
                let index = layer * rows + row
                try decoder.encodeStateReplay(enc, y: view(normalized, index: index, count: qkvC),
                                              betas: view(betas, index: index, count: gdnVH),
                                              glog: view(glog, index: index, count: glogC), state: state)
            }
        }
    }

    /// Returns false for rows whose inverse would amplify rounding enough to
    /// make snapshot-and-replay the safer path.
    func rollbackTapeIsStable(retaining count: Int, activeRows: Int? = nil) throws -> Bool {
        let verifiedRows = activeRows ?? rows
        guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
            throw QWEngineDriverError.state("GPU rollback active row count is invalid")
        }
        try validateRetained(verifiedRows)
        let gdnVH = geometry.gdnVHeads
        guard layers > 0, count < verifiedRows, gdnVH > 0 else { return true }
        if batchedLayerCount == layers, batchedActiveRows == verifiedRows, verifiedRows <= 16 {
            let mask = stabilityMask.region.pointer.advanced(by: stabilityMask.offset)
                .assumingMemoryBound(to: UInt32.self).pointee
            return (mask & (UInt32(1) << UInt32(count))) == 0
        }
        let decayValues = glog.region.pointer.advanced(by: glog.offset).assumingMemoryBound(to: Float.self)
        let innovationValues = innovations.region.pointer.advanced(by: innovations.offset)
            .assumingMemoryBound(to: Float.self)
        let normalizedValues = normalized.region.pointer.advanced(by: normalized.offset)
            .assumingMemoryBound(to: UInt16.self)
        let qkvC = geometry.gdnQKVChannels, gdnHD = geometry.gdnHeadDim
        let zC = geometry.gdnZChannels, kOffset = geometry.gdnQKHeads * gdnHD
        let glogC = geometry.gdnGlogChannels
        let vPerQK = max(1, gdnVH / max(1, geometry.gdnQKHeads))
        let qkHeads = min(geometry.gdnQKHeads, (gdnVH - 1) / vPerQK + 1)
        let innovSpan = gdnVH * gdnHD
        let keySpan = qkHeads * gdnHD
        let minVecDecay = SIMD8<Float>(repeating: 1.0e-3)
        let maxVecDecay = SIMD8<Float>(repeating: 1.0e38)
        for layer in 0..<layers {
            let layerRowBase = layer * rows
            if geometry.gdnVectorDecay {
                for head in 0..<gdnVH {
                    let headColBase = head * gdnHD
                    var chunkStart = 0
                    while chunkStart + 8 <= gdnHD {
                        var cumDecay = SIMD8<Float>(repeating: 1.0)
                        for row in count..<verifiedRows {
                            let base = (layerRowBase + row) * glogC + headColBase + chunkStart
                            let v = UnsafeRawPointer(decayValues.advanced(by: base)).loadUnaligned(as: SIMD8<Float>.self)
                            guard all((v .>= minVecDecay) .& (v .<= maxVecDecay)) else { return false }
                            cumDecay *= v
                            guard all((cumDecay .>= minVecDecay) .& (cumDecay .<= maxVecDecay)) else { return false }
                        }
                        chunkStart += 8
                    }
                    while chunkStart < gdnHD {
                        var cumDecay: Float = 1.0
                        for row in count..<verifiedRows {
                            let decay = decayValues[(layerRowBase + row) * glogC + headColBase + chunkStart]
                            guard decay.isFinite && decay >= 1.0e-3 && decay <= 1.0e38 else { return false }
                            cumDecay *= decay
                            guard cumDecay.isFinite && cumDecay >= 1.0e-3 && cumDecay <= 1.0e38 else { return false }
                        }
                        chunkStart += 1
                    }
                }
            } else {
                for head in 0..<gdnVH {
                    var cumLogDecay: Float = 0.0
                    for row in count..<verifiedRows {
                        let decay = decayValues[(layerRowBase + row) * glogC + head]
                        guard decay.isFinite && decay >= -6.907755 && decay <= 88.0 else { return false }
                        cumLogDecay += decay
                        guard cumLogDecay.isFinite && cumLogDecay >= -6.907755 else { return false }
                    }
                }
            }
            for row in count..<verifiedRows {
                let rowIndex = layerRowBase + row
                let innovBase = rowIndex * zC
                for idx in 0..<innovSpan where !innovationValues[innovBase + idx].isFinite {
                    return false
                }
                let keyBase = rowIndex * qkvC + kOffset
                for idx in 0..<keySpan where (normalizedValues[keyBase + idx] & 0x7C00) == 0x7C00 {
                    return false
                }
            }
        }
        return true
    }

    func encodeRollbackTape(_ enc: any MTLComputeCommandEncoder, decoder: QWGDNDecode,
                            rec: QWTensorValue, activeRows: Int? = nil,
                            retaining count: Int, verifyStability: Bool = true) throws {
        let verifiedRows = activeRows ?? rows
        if layers == 0 {
            guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
                throw QWEngineDriverError.state("GPU rollback active row count is invalid")
            }
            try validateRetained(verifiedRows)
            return
        }
        guard (2...rows).contains(verifiedRows), (0...verifiedRows).contains(count) else {
            throw QWEngineDriverError.state("GPU rollback active row count is invalid")
        }
        try validate(rec, like: recurrent)
        try validateRetained(verifiedRows)
        guard sameStorage(rec, sourceRec) else {
            throw QWEngineDriverError.state("GPU rollback requires the original recurrent slab")
        }
        if verifyStability {
            guard try rollbackTapeIsStable(retaining: count, activeRows: verifiedRows) else {
                throw QWEngineDriverError.state("GPU innovation tape is ill-conditioned; restore and replay instead")
            }
        }
        let rollbackRows = verifiedRows - count
        if rollbackRows > 0, !(outOfPlaceVerifyActive && count == 0) {
            let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels, glogC = geometry.gdnGlogChannels
            let recElems = QWSlotGeometry.recElems(geometry: geometry)
            for layer in 0..<layers {
                let dstState = view(rec, index: layer, count: recElems)
                let srcState = outOfPlaceVerifyActive ? view(recurrent, index: layer, count: recElems) : dstState
                let startRow = layer * rows + count
                try decoder.encodeStateRollbackRows(
                    enc,
                    y: slice(normalized, startRow: startRow, rowCount: rollbackRows, rowWidth: qkvC),
                    innovation: slice(innovations, startRow: startRow, rowCount: rollbackRows, rowWidth: zC),
                    glog: slice(glog, startRow: startRow, rowCount: rollbackRows, rowWidth: glogC),
                    state: srcState,
                    rows: rollbackRows,
                    stateOut: dstState)
            }
            enc.memoryBarrier(scope: .buffers)
        }
    }

    /// Coalesces each layer's 3-row history copy into 1 blit (count==0 or count>=3) or 2 blits (count==1..2).
    func encodeReplayRings(_ enc: any MTLBlitCommandEncoder,
                           ring: QWTensorValue, retaining count: Int) throws {
        if layers == 0 {
            try validateRetained(count)
            return
        }
        try validate(ring, like: rings)
        try validateRetained(count)
        guard sameStorage(ring, sourceRing) else {
            throw QWEngineDriverError.state("GPU replay requires the checkpoint's original ring slab")
        }
        if count == 0 {
            copy(enc, rings, ring)
            return
        }
        let qkvC = geometry.gdnQKVChannels
        for layer in 0..<layers {
            if count >= 3 {
                copy(enc, slice(raw, startRow: layer * rows + count - 3, rowCount: 3, rowWidth: qkvC),
                     slice(ring, startRow: layer * 3, rowCount: 3, rowWidth: qkvC))
            } else {
                let fromRings = 3 - count
                copy(enc, slice(rings, startRow: layer * 3 + count, rowCount: fromRings, rowWidth: qkvC),
                     slice(ring, startRow: layer * 3, rowCount: fromRings, rowWidth: qkvC))
                copy(enc, slice(raw, startRow: layer * rows, rowCount: count, rowWidth: qkvC),
                     slice(ring, startRow: layer * 3 + fromRings, rowCount: count, rowWidth: qkvC))
            }
        }
    }
}
