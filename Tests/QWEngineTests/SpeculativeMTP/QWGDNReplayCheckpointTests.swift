import XCTest
import Metal
import QWAlloc
import QWCore
import QWGDN
import QWHAL
@testable import QWEngine

final class QWGDNReplayCheckpointTests: XCTestCase {
    func testCheckpointBudgetIncludesEveryAlignedRequest() throws {
        XCTAssertEqual(QWGDNReplayCheckpoint.allocationBytes(geometry: QWTestGeometry.g27, layers: 48, rows: 2), 160_272_384)
        XCTAssertEqual(QWGDNReplayCheckpoint.allocationBytes(geometry: QWTestGeometry.g27, layers: 48, rows: 3), 163_436_544)
        let hal = try QWHal(geometry: .qwen38_27b)
        for rows in 1...16 {
            let budget = QWGDNReplayCheckpoint.allocationBytes(geometry: QWTestGeometry.g27, layers: 1, rows: rows)
            let arena = try QWArena(hal: hal, weights: 256, activations: 256, kv: 256, gdnStates: budget)
            let allocator = arena.gdnRegion()
            _ = try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: allocator, layers: 1, rows: rows)
            XCTAssertEqual(allocator.usedBytes, budget)
            XCTAssertThrowsError(try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: allocator, layers: 0, rows: rows))
            XCTAssertThrowsError(try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: allocator, layers: 1, rows: 17))
        }
    }

    func testDestructiveRestoreAndEveryRetainedBoundary() throws {
        try destructiveRestoreAndEveryRetainedBoundary(unsafeDecay: false)
    }

    func testIllConditionedInnovationUsesSnapshotReplayFallback() throws {
        try destructiveRestoreAndEveryRetainedBoundary(unsafeDecay: true)
    }

    func testVectorDecaySnapshotAndRowMajorReplayParity() throws {
        try destructiveRestoreAndEveryRetainedBoundary(unsafeDecay: true, geometry: Self.vectorDecayGeometry, rows: 4)
    }

    private func destructiveRestoreAndEveryRetainedBoundary(
        unsafeDecay: Bool,
        geometry: QWModelGeometry = QWTestGeometry.g27,
        rows: Int = 3
    ) throws {
        let hal = try QWHal(geometry: geometry)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable,
                          "1024-thread GDN recurrent kernel requires bare-metal Apple Silicon GPU")
        let layers = 2
        let allocatorBytes = 2 * QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: layers, rows: rows) + (1 << 20)
        let arena = try QWArena(hal: hal, weights: 256, activations: 2 << 20,
                                kv: 256, gdnStates: allocatorBytes)
        let g = arena.gdnRegion(), a = arena.activationsRegion()
        let checkpoint = try QWGDNReplayCheckpoint(geometry: geometry, allocator: g, layers: layers, rows: rows)
        let window = try QWVerifyWindow(geometry: geometry, allocator: a, rows: rows, vocabulary: 1)
        let qkvC = geometry.gdnQKVChannels, zC = geometry.gdnZChannels, vh = geometry.gdnVHeads
        let hd = geometry.gdnHeadDim, glogC = geometry.gdnGlogChannels
        let hiddenCarry = try a.allocate(QWTensorShape([geometry.hidden], .f16))
        let rec = try g.allocate(QWTensorShape([layers * QWSlotGeometry.recElems(geometry: geometry)], .f32))
        let ring = try g.allocate(QWTensorShape([layers * QWSlotGeometry.ringElems(geometry: geometry)], .f16))
        let raw = try a.allocate(QWTensorShape([layers * rows * qkvC], .f16))
        let y = try a.allocate(QWTensorShape([qkvC], .f16))
        let beta = try a.allocate(QWTensorShape([vh], .f32))
        let decay = try a.allocate(QWTensorShape([glogC], .f32))
        let convWeight = try a.allocate(QWTensorShape([4 * qkvC], .f16))
        let z = try a.allocate(QWTensorShape([zC], .f16))
        let norm = try a.allocate(QWTensorShape([hd], .f16))
        let output = try a.allocate(QWTensorShape([zC], .f16))
        let outputScratch = try a.allocate(QWTensorShape([zC], .f32))
        let rp = rec.region.pointer.advanced(by: rec.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<rec.shape.elementCount { rp[i] = Float(i % 37 - 18) / 10000 }
        func fillHalf(_ t: QWTensorValue, value: (Int) -> Float16) {
            let p = t.region.pointer.advanced(by: t.offset).assumingMemoryBound(to: Float16.self)
            for i in 0..<t.shape.elementCount { p[i] = value(i) }
        }
        fillHalf(ring) { Float16(Float($0 % 53 - 26) / 137) }
        fillHalf(raw) { Float16(Float(($0 * 13 + $0 / qkvC) % 113 - 56) / 173) }
        fillHalf(convWeight) { Float16(Float($0 % 7 - 3) / 13) }
        fillHalf(z) { _ in 0.5 }
        fillHalf(norm) { _ in 1 }
        fillHalf(window.hidden) { Float16(Float($0 / geometry.hidden + 1) / 7) }
        fillHalf(hiddenCarry) { _ in -1 }
        let bp = beta.region.pointer.advanced(by: beta.offset).assumingMemoryBound(to: Float.self)
        let dp = decay.region.pointer.advanced(by: decay.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<vh { bp[i] = Float(i + 1) / 64 }
        for i in 0..<glogC {
            if geometry.gdnVectorDecay {
                dp[i] = unsafeDecay ? exp(-5.0) : (0.5 + Float(i % 19) * 0.02)
            } else {
                dp[i] = unsafeDecay && i == 0 ? -10 : -Float(i + 1) / 97
            }
        }
        let decoder = try QWGDNDecode(hal: hal), conv = try QWGDNConv(hal: hal, geometry: geometry)
        _ = try hal.pipeline("qwg_gdn_replay_state")
        g.markSteadyState()
        a.markSteadyState()
        func view(_ t: QWTensorValue, index: Int, count: Int) -> QWTensorValue {
            QWTensorValue(region: t.region, offset: t.offset + index * count * (t.shape.dtype == .f32 ? 4 : 2),
                          shape: QWTensorShape([count], t.shape.dtype))
        }
        func bytes(_ t: QWTensorValue) -> Data {
            Data(bytes: t.region.pointer.advanced(by: t.offset), count: t.bytes)
        }
        func submit(_ body: (any MTLCommandBuffer) throws -> Void) throws {
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            try body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                throw QWEngineDriverError.state("checkpoint GPU failure: \(String(describing: cb.error))")
            }
        }
        func blit(_ cb: any MTLCommandBuffer, _ body: (any MTLBlitCommandEncoder) throws -> Void) throws {
            let enc = try XCTUnwrap(cb.makeBlitCommandEncoder())
            defer { enc.endEncoding() }
            try body(enc)
        }
        func compute(_ cb: any MTLCommandBuffer, _ body: (any MTLComputeCommandEncoder) throws -> Void) throws {
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            defer { enc.endEncoding() }
            try body(enc)
        }
        try submit { cb in
            try blit(cb) { enc in
                XCTAssertThrowsError(try checkpoint.encodeRestore(enc, rec: rec, ring: ring))
                XCTAssertThrowsError(try checkpoint.encodeCapture(enc, layer: 0, row: 0,
                                                                  rawInput: y, postConv: y, beta: beta, decay: decay))
            }
        }
        try submit { cb in
            try blit(cb) { try checkpoint.encodeBackup($0, rec: rec, ring: ring) }
            try compute(cb) { enc in
                XCTAssertThrowsError(try checkpoint.encodeReplay(enc, decoder: decoder, rec: rec, retaining: 1))
            }
        }
        var expectedRec = [bytes(rec)], expectedRing = [bytes(ring)]
        for row in 0..<rows {
            try submit { cb in
                for layer in 0..<layers {
                    let rawRow = view(raw, index: layer * rows + row, count: qkvC)
                    let state = view(rec, index: layer, count: QWSlotGeometry.recElems(geometry: geometry))
                    let history = view(ring, index: layer, count: QWSlotGeometry.ringElems(geometry: geometry))
                    try compute(cb) { enc in
                        try conv.encodeT1WithHeadnorm(enc, x: rawRow, w: convWeight, bias: nil,
                                                      stIn: history, y: y, stOut: history)
                        if row % 2 == 0 {
                            try checkpoint.encodeCapture(enc, hal: hal, layer: layer, row: row,
                                                         rawInput: rawRow, postConv: y, beta: beta, decay: decay)
                        }
                        try decoder.encode(enc, y: y, betas: beta, glog: decay, state: state,
                                           z: z, w: norm, o: output, oScratch: outputScratch,
                                           innovation: try checkpoint.innovationRow(layer: layer, row: row))
                    }
                    if row % 2 != 0 {
                        try blit(cb) {
                            try checkpoint.encodeCapture($0, layer: layer, row: row, rawInput: rawRow,
                                                         postConv: y, beta: beta, decay: decay)
                        }
                    }
                }
            }
            expectedRec.append(bytes(rec))
            expectedRing.append(bytes(ring))
        }
        let beforeInvalidRec = bytes(rec), beforeInvalidRing = bytes(ring)
        try submit { cb in
            XCTAssertThrowsError(try window.encodeRetainedPrefix(
                cb, hal: hal, checkpoint: checkpoint, decoder: decoder,
                rec: rec, ring: ring, trunkHidden: beta, retaining: 1))
            XCTAssertThrowsError(try window.encodeRetainedPrefix(
                cb, hal: hal, checkpoint: checkpoint, decoder: decoder,
                rec: rec, ring: ring, trunkHidden: window.hiddenRow(0), retaining: rows))
            XCTAssertThrowsError(try window.encodeRetainedPrefix(
                cb, hal: hal, checkpoint: checkpoint, decoder: decoder,
                rec: outputScratch, ring: ring, trunkHidden: hiddenCarry, retaining: rows))
        }
        XCTAssertEqual(bytes(rec), beforeInvalidRec, "invalid resolution must encode no state restore")
        XCTAssertEqual(bytes(ring), beforeInvalidRing, "invalid resolution must encode no ring restore")
        for retained in 0...rows {
            if geometry.gdnVectorDecay && unsafeDecay {
                XCTAssertEqual(try checkpoint.rollbackTapeIsStable(retaining: retained, activeRows: rows), retained >= rows - 1)
            }
            fillHalf(hiddenCarry) { _ in -1 }
            let originalHidden = bytes(hiddenCarry)
            try submit { cb in
                try blit(cb) { enc in
                    for t in [rec, ring] {
                        let start = t.region.base + t.offset
                        enc.fill(buffer: t.region.buffer, range: start..<(start + t.bytes), value: 0xDE)
                    }
                    try checkpoint.encodeRestore(enc, rec: rec, ring: ring)
                }
                try compute(cb) { try checkpoint.encodeReplay($0, decoder: decoder, rec: rec, retaining: rows) }
                try blit(cb) { try checkpoint.encodeReplayRings($0, ring: ring, retaining: rows) }
                try window.encodeRetainedPrefix(cb, hal: hal, checkpoint: checkpoint, decoder: decoder,
                                                rec: rec, ring: ring, trunkHidden: hiddenCarry, retaining: retained)
            }
            let actualState = rec.region.pointer.advanced(by: rec.offset).assumingMemoryBound(to: Float.self)
            let expectedState = expectedRec[retained].withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                raw.bindMemory(to: Float.self).baseAddress!
            }
            var maxStateError: Float = 0
            for index in 0..<rec.shape.elementCount {
                maxStateError = max(maxStateError, abs(actualState[index] - expectedState[index]))
            }
            XCTAssertLessThanOrEqual(maxStateError, 1.0e-4,
                "innovation rollback FP32 state error \(maxStateError) at retained boundary \(retained)")
            XCTAssertEqual(bytes(ring), expectedRing[retained], "FP16 raw ring at retained boundary \(retained)")
            XCTAssertEqual(bytes(hiddenCarry), retained == 0 ? originalHidden : bytes(try window.hiddenRow(retained - 1)),
                           "hidden seed must match the last consumed verify row")
            let p = rec.region.pointer.advanced(by: rec.offset).assumingMemoryBound(to: Float.self)
            XCTAssertTrue((0..<rec.shape.elementCount).allSatisfy { p[$0].isFinite })
        }
        XCTAssertEqual(g.steadyStateAllocs, 0)
        XCTAssertEqual(a.steadyStateAllocs, 0)
        try submit { cb in
            try blit(cb) { enc in
                XCTAssertThrowsError(try checkpoint.encodeBackup(enc, rec: checkpoint.recurrent, ring: ring))
                XCTAssertThrowsError(try checkpoint.encodeCapture(enc, layer: layers, row: 0,
                                                                  rawInput: y, postConv: y, beta: beta, decay: decay))
                XCTAssertThrowsError(try checkpoint.encodeReplayRings(enc, ring: ring, retaining: rows + 1))
            }
            try compute(cb) { enc in
                XCTAssertThrowsError(try checkpoint.encodeReplay(enc, decoder: decoder, rec: rec, retaining: -1))
                let invalid = view(y, index: 1, count: qkvC)
                XCTAssertThrowsError(try decoder.encodeStateReplay(enc, y: invalid, betas: beta, glog: decay, state: rec))
            }
        }
        try submit { cb in
            try blit(cb) { try checkpoint.encodeBackup($0, rec: rec, ring: ring) }
            try compute(cb) { enc in
                XCTAssertThrowsError(try checkpoint.encodeReplay(enc, decoder: decoder, rec: rec, retaining: rows),
                                     "a new checkpoint must invalidate the previous journal")
            }
        }
    }

    func testBatchedGDNVerifyAndRollbackParity() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable,
                          "1024-thread GDN recurrent kernel requires bare-metal Apple Silicon GPU")
        let layers = 2, rows = 4
        let allocatorBytes = 4 * QWGDNReplayCheckpoint.allocationBytes(geometry: QWTestGeometry.g27, layers: layers, rows: rows) + (16 << 20)
        let arena = try QWArena(hal: hal, weights: 256, activations: 8 << 20,
                                kv: 256, gdnStates: allocatorBytes)
        let g = arena.gdnRegion(), a = arena.activationsRegion()
        let refCheckpoint = try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: g, layers: layers, rows: rows)
        let batCheckpoint = try QWGDNReplayCheckpoint(geometry: QWTestGeometry.g27, allocator: g, layers: layers, rows: rows)
        let window = try QWVerifyWindow(geometry: QWTestGeometry.g27, allocator: a, rows: rows, vocabulary: 1)
        let hiddenCarry = try a.allocate(QWTensorShape([QWModelGeometry.qwen38_27b.hidden], .f16))
        let refRec = try g.allocate(QWTensorShape([layers * QWSlotGeometry.recElems(geometry: .qwen38_27b)], .f32))
        let batRec = try g.allocate(QWTensorShape([layers * QWSlotGeometry.recElems(geometry: .qwen38_27b)], .f32))
        let refRing = try g.allocate(QWTensorShape([layers * QWSlotGeometry.ringElems(geometry: .qwen38_27b)], .f16))
        let batRing = try g.allocate(QWTensorShape([layers * QWSlotGeometry.ringElems(geometry: .qwen38_27b)], .f16))
        let gdnProj = try a.allocate(QWTensorShape([rows * QWModelGeometry.qwen38_27b.gdnTotalN], .f16))
        let controls = try a.allocate(QWTensorShape([rows * 96], .f32))
        let convWeight = try a.allocate(QWTensorShape([4 * 10240], .f16))
        let normWeight = try a.allocate(QWTensorShape([128], .f16))
        let refOut = try a.allocate(QWTensorShape([rows * 6144], .f16))
        let batOut = try a.allocate(QWTensorShape([rows * 6144], .f16))
        let convY = try a.allocate(QWTensorShape([10240], .f16))
        let oScratch = try a.allocate(QWTensorShape([6144], .f32))

        let rp = refRec.region.pointer.advanced(by: refRec.offset).assumingMemoryBound(to: Float.self)
        let bp = batRec.region.pointer.advanced(by: batRec.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<refRec.shape.elementCount {
            let v = Float(i % 37 - 18) / 10000
            rp[i] = v
            bp[i] = v
        }
        func fillHalf(_ t: QWTensorValue, value: (Int) -> Float16) {
            let p = t.region.pointer.advanced(by: t.offset).assumingMemoryBound(to: Float16.self)
            for i in 0..<t.shape.elementCount { p[i] = value(i) }
        }
        fillHalf(refRing) { Float16(Float($0 % 53 - 26) / 137) }
        fillHalf(batRing) { Float16(Float($0 % 53 - 26) / 137) }
        fillHalf(gdnProj) { Float16(Float(($0 * 17 + $0 / 16480) % 113 - 56) / 173) }
        fillHalf(convWeight) { Float16(Float($0 % 7 - 3) / 13) }
        fillHalf(normWeight) { Float16(1.0 + Float($0 % 5) * 0.05) }
        fillHalf(window.hidden) { Float16(Float($0 / QWModelGeometry.qwen38_27b.hidden + 1) / 7) }
        fillHalf(hiddenCarry) { _ in -1 }
        let cp = controls.region.pointer.advanced(by: controls.offset).assumingMemoryBound(to: Float.self)
        for r in 0..<rows {
            for h in 0..<48 {
                cp[r * 96 + h] = Float(h + r + 1) / 64.0
                cp[r * 96 + 48 + h] = -Float(h + r + 1) / 97.0
            }
        }
        let decoder = try QWGDNDecode(hal: hal), conv = try QWGDNConv(hal: hal, geometry: .qwen38_27b)
        func view(_ t: QWTensorValue, index: Int, count: Int) -> QWTensorValue {
            QWTensorValue(region: t.region, offset: t.offset + index * count * (t.shape.dtype == .f32 ? 4 : 2),
                          shape: QWTensorShape([count], t.shape.dtype))
        }
        func bytes(_ t: QWTensorValue) -> Data {
            Data(bytes: t.region.pointer.advanced(by: t.offset), count: t.bytes)
        }
        let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
        let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
        try refCheckpoint.encodeBackup(blit, rec: refRec, ring: refRing)
        try batCheckpoint.encodeBackup(blit, rec: batRec, ring: batRing)
        blit.endEncoding()
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        for layer in 0..<layers {
            let refState = view(refRec, index: layer, count: QWSlotGeometry.recElems(geometry: .qwen38_27b))
            let refHistory = view(refRing, index: layer, count: QWSlotGeometry.ringElems(geometry: .qwen38_27b))
            for row in 0..<rows {
                let rowBase = row * QWModelGeometry.qwen38_27b.gdnTotalN
                let rawRow = QWTensorValue(region: gdnProj.region, offset: gdnProj.offset + rowBase * 2,
                                           shape: QWTensorShape([10240], .f16))
                let zRow = QWTensorValue(region: gdnProj.region, offset: gdnProj.offset + (rowBase + 10240) * 2,
                                         shape: QWTensorShape([6144], .f16))
                let betaRow = QWTensorValue(region: controls.region, offset: controls.offset + row * 96 * 4,
                                            shape: QWTensorShape([48], .f32))
                let decayRow = QWTensorValue(region: controls.region, offset: controls.offset + (row * 96 + 48) * 4,
                                             shape: QWTensorShape([48], .f32))
                let outRow = view(refOut, index: row, count: 6144)
                try conv.encodeT1WithHeadnorm(enc, x: rawRow, w: convWeight, bias: nil,
                                              stIn: refHistory, y: convY, stOut: refHistory)
                enc.memoryBarrier(scope: .buffers)
                try refCheckpoint.encodeCapture(enc, hal: hal, layer: layer, row: row,
                                                rawInput: rawRow, postConv: convY, beta: betaRow, decay: decayRow)
                try decoder.encode(enc, y: convY, betas: betaRow, glog: decayRow, state: refState,
                                   z: zRow, w: normWeight, o: outRow, oScratch: oScratch,
                                   innovation: try refCheckpoint.innovationRow(layer: layer, row: row))
                enc.memoryBarrier(scope: .buffers)
            }
            let batState = view(batRec, index: layer, count: QWSlotGeometry.recElems(geometry: .qwen38_27b))
            let batHistory = view(batRing, index: layer, count: QWSlotGeometry.ringElems(geometry: .qwen38_27b))
            try batCheckpoint.encodeBatchedGDN(
                enc, conv: conv, decoder: decoder, layer: layer, activeRows: rows,
                gdnProjection: gdnProj, controls: controls, convWeight: convWeight, normWeight: normWeight,
                ring: batHistory, state: batState, output: batOut)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)
        XCTAssertEqual(bytes(batRing), bytes(refRing), "batched GDN verify ring must be bit-exact")
        XCTAssertEqual(bytes(batRec), bytes(refRec), "batched GDN verify state must be bit-exact")
        XCTAssertEqual(bytes(batOut), bytes(refOut), "batched GDN verify output must be bit-exact")
        try batCheckpoint.validateCompleteCapture(activeRows: rows)
    }

    func testCumulativeMultiRowDecayTriggersSnapshotReplayFallback() throws {
        let rows = 8, layers = 1
        func makeCheckpoint(geometry: QWModelGeometry, fillGlog: (UnsafeMutablePointer<Float>, Int) -> Void) throws -> QWGDNReplayCheckpoint {
            let hal = try QWHal(geometry: geometry)
            let budget = QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: layers, rows: rows)
                + QWSlotGeometry.recElems(geometry: geometry) * 4 + QWSlotGeometry.ringElems(geometry: geometry) * 2 + 4096
            let arena = try QWArena(hal: hal, weights: 256, activations: 256, kv: 256, gdnStates: budget)
            let g = arena.gdnRegion()
            let cp = try QWGDNReplayCheckpoint(geometry: geometry, allocator: g, layers: layers, rows: rows)
            let rec = try g.allocate(QWTensorShape([layers * QWSlotGeometry.recElems(geometry: geometry)], .f32))
            let ring = try g.allocate(QWTensorShape([layers * QWSlotGeometry.ringElems(geometry: geometry)], .f16))
            cp.sourceRec = rec
            cp.sourceRing = ring
            cp.captured[0] = UInt32((1 << rows) - 1)
            memset(cp.normalized.region.pointer.advanced(by: cp.normalized.offset), 0, cp.normalized.bytes)
            memset(cp.innovations.region.pointer.advanced(by: cp.innovations.offset), 0, cp.innovations.bytes)
            let glogPtr = cp.glog.region.pointer.advanced(by: cp.glog.offset).assumingMemoryBound(to: Float.self)
            fillGlog(glogPtr, cp.glog.shape.elementCount)
            return cp
        }

        let scalarCp = try makeCheckpoint(geometry: QWTestGeometry.g27) { ptr, count in
            for i in 0..<count { ptr[i] = -2.0 }
        }
        XCTAssertTrue(try scalarCp.rollbackTapeIsStable(retaining: 5, activeRows: rows))
        XCTAssertFalse(try scalarCp.rollbackTapeIsStable(retaining: 4, activeRows: rows))

        let vecGeo = Self.vectorDecayGeometry
        let vecCp = try makeCheckpoint(geometry: vecGeo) { ptr, count in
            for i in 0..<count { ptr[i] = 0.2 }
        }
        XCTAssertTrue(try vecCp.rollbackTapeIsStable(retaining: 4, activeRows: rows))
        XCTAssertFalse(try vecCp.rollbackTapeIsStable(retaining: 3, activeRows: rows))

        let kdaLowerBoundCp = try makeCheckpoint(geometry: vecGeo) { ptr, count in
            for i in 0..<count { ptr[i] = 1.0 }
            ptr[6 * vecGeo.gdnGlogChannels] = exp(-5.0)
            ptr[7 * vecGeo.gdnGlogChannels] = exp(-5.0)
        }
        XCTAssertTrue(try kdaLowerBoundCp.rollbackTapeIsStable(retaining: 7, activeRows: rows))
        XCTAssertFalse(try kdaLowerBoundCp.rollbackTapeIsStable(retaining: 6, activeRows: rows))
    }

    func testCoalescedReplayRingsBitExactAcrossRetainedCounts() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let layers = 3, rows = 8
        let geo = QWTestGeometry.g27
        let qkvC = geo.gdnQKVChannels
        let recElems = QWSlotGeometry.recElems(geometry: geo)
        let ringElems = 3 * qkvC
        let budget = QWGDNReplayCheckpoint.allocationBytes(geometry: geo, layers: layers, rows: rows)
            + layers * recElems * 4 + layers * ringElems * 2 + 8192
        let arena = try QWArena(hal: hal, weights: 256, activations: 256, kv: 256, gdnStates: budget)
        let g = arena.gdnRegion()
        let cp = try QWGDNReplayCheckpoint(geometry: geo, allocator: g, layers: layers, rows: rows)
        let rec = try g.allocate(QWTensorShape([layers * recElems], .f32))
        let ring = try g.allocate(QWTensorShape([layers * ringElems], .f16))
        let ringPtr = ring.region.pointer.advanced(by: ring.offset).assumingMemoryBound(to: UInt16.self)
        let rawPtr = cp.raw.region.pointer.advanced(by: cp.raw.offset).assumingMemoryBound(to: UInt16.self)
        var initialRing = [UInt16](repeating: 0, count: layers * ringElems)
        var rawRows = [UInt16](repeating: 0, count: layers * rows * qkvC)
        for layer in 0..<layers {
            for age in 0..<3 {
                let base = (layer * 3 + age) * qkvC
                for j in 0..<qkvC {
                    initialRing[base + j] = UInt16((layer + 1) * 1000 + (age + 1) * 100 + (j % 97))
                }
            }
            for row in 0..<rows {
                let base = (layer * rows + row) * qkvC
                for j in 0..<qkvC {
                    rawRows[base + j] = UInt16(30000 + (layer + 1) * 1000 + row * 100 + (j % 89))
                }
            }
        }
        rawPtr.update(from: rawRows, count: rawRows.count)

        for count in [0, 1, 2, 3, 5, 8] {
            ringPtr.update(from: initialRing, count: initialRing.count)
            let cb = try XCTUnwrap(hal.queue.makeCommandBuffer())
            let blit = try XCTUnwrap(cb.makeBlitCommandEncoder())
            var ended = false
            defer { if !ended { blit.endEncoding() } }
            try cp.encodeBackup(blit, rec: rec, ring: ring)
            for l in 0..<layers { cp.captured[l] = UInt32((1 << rows) - 1) }
            try cp.encodeReplayRings(blit, ring: ring, retaining: count)
            blit.endEncoding()
            ended = true
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)

            for layer in 0..<layers {
                for age in 0..<3 {
                    let dstBase = (layer * 3 + age) * qkvC
                    let expectedSlice: ArraySlice<UInt16>
                    if age + count < 3 {
                        let srcBase = (layer * 3 + age + count) * qkvC
                        expectedSlice = initialRing[srcBase..<(srcBase + qkvC)]
                    } else {
                        let srcBase = (layer * rows + age + count - 3) * qkvC
                        expectedSlice = rawRows[srcBase..<(srcBase + qkvC)]
                    }
                    let actual = Array(UnsafeBufferPointer(start: ringPtr.advanced(by: dstBase), count: qkvC))
                    XCTAssertEqual(actual, Array(expectedSlice), "ring mismatch layer=\(layer) age=\(age) count=\(count)")
                }
            }
        }
    }

    private static let vectorDecayGeometry: QWModelGeometry = {
        let base = QWModelDescriptor.qwen38_27b
        let vecDesc = QWModelDescriptor(
            family: base.family, modelType: base.modelType, capabilities: base.capabilities,
            hidden: base.hidden, vocab: base.vocab, denseIntermediate: base.denseIntermediate,
            numLayers: base.numLayers, layerTypes: base.layerTypes, fullAttentionInterval: base.fullAttentionInterval,
            rmsEps: base.rmsEps, maxPositionEmbeddings: base.maxPositionEmbeddings,
            qHeads: base.qHeads, kvHeads: base.kvHeads, headDim: base.headDim,
            attnOutputGate: base.attnOutputGate, partialRotaryFactor: base.partialRotaryFactor, ropeTheta: base.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil, indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil, sharedExpertIntermediate: nil,
            expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: base.gdnQKHeads, gdnVHeads: base.gdnVHeads, gdnHeadDim: base.gdnHeadDim,
            gdnQKVChannels: base.gdnQKVChannels, gdnZChannels: base.gdnZChannels, gdnConvKernel: base.gdnConvKernel,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil, ngramSize: nil, headsPerNgram: nil,
            ngramVocabSizeBase: nil, makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: base.mtpLayers, mtpUseDedicatedEmbeddings: base.mtpUseDedicatedEmbeddings, eosTokenId: base.eosTokenId,
            visPosEmbeds: base.visPosEmbeds, gdnVectorDecay: true, role: base.role)
        return QWModelGeometry(descriptor: vecDesc)
    }()
}

private extension QWGDNReplayCheckpoint {
    func innovationRow(layer: Int, row: Int) throws -> QWTensorValue {
        guard (0..<layers).contains(layer), (0..<rows).contains(row) else {
            throw QWEngineDriverError.state("GPU innovation tape index is invalid")
        }
        return view(innovations, index: layer * rows + row, count: geometry.gdnZChannels)
    }
}
