// QWMBSAAttnOperatorTests.swift — GPU correctness & state tests for Maxwell Block-Sparse GQA.

import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
@testable import QWKV
import QWRope

final class QWMBSAAttnOperatorTests: XCTestCase {
    private static func makeGeometry() throws -> QWModelGeometry {
        let cfg = """
        {
          "model_type": "maxwell_moe",
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "vocab_size": 2048,
          "num_hidden_layers": 1,
          "num_experts": 0,
          "num_attention_heads": 32,
          "num_key_value_heads": 4,
          "attention_head_size": 128,
          "qk_layernorm": true,
          "attention_variant": "block_sparse_gqa",
          "block_sparse_gqa_indexer_n_heads": 4,
          "block_sparse_gqa_indexer_head_dim": 128,
          "block_sparse_gqa_block_size": 128,
          "block_sparse_gqa_prefix_window_size": 1024,
          "block_sparse_gqa_local_window_size": 1024,
          "block_sparse_gqa_num_remote_blocks": 16,
          "enable_hyper_connections": true,
          "num_residual_streams": 4,
          "mhc_sinkhorn_iterations": 20,
          "rms_norm_eps": 1e-6,
          "rope_theta": 20000000,
          "max_position_embeddings": 8192,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: cfg))
    }

    private func writeF16(_ buf: any MTLBuffer, byteOffset: Int, values: [Float]) {
        let dst = buf.contents().advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
        for (i, v) in values.enumerated() { dst[i] = Float16(v).bitPattern }
    }

    private func readF16(_ buf: any MTLBuffer, byteOffset: Int, count: Int) -> [Float] {
        let src = buf.contents().advanced(by: byteOffset).assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(Float16(bitPattern: src[$0])) }
    }

    private func dummyLayerWeights(hal: QWHal, geometry: QWModelGeometry) throws -> QWMBSAAttnOperator.LayerWeights {
        let headDim = geometry.headDim
        let n = (geometry.kvHeads + 1) * headDim
        let k = geometry.hidden
        let elems = n * (k / 32) * 10
        let blkBuf = try XCTUnwrap(hal.device.makeBuffer(length: elems * 2, options: .storageModeShared))
        let reg = QWMRegion(buffer: blkBuf, base: 0, length: blkBuf.length)
        let dummyBlocks = QWTensorValue(region: reg, offset: 0, shape: QWTensorShape([elems], .f16))
        let view = try XCTUnwrap(QWWeightView(
            blocks: dummyBlocks, n: n, k: k,
            bits: 4, groupSize: 32, isRotated: false))
        let normBytes = headDim * 2
        let wqBuf = try XCTUnwrap(hal.device.makeBuffer(length: normBytes, options: .storageModeShared))
        let wkBuf = try XCTUnwrap(hal.device.makeBuffer(length: normBytes, options: .storageModeShared))
        writeF16(wqBuf, byteOffset: 0, values: [Float](repeating: 0.25, count: headDim))
        writeF16(wkBuf, byteOffset: 0, values: [Float](repeating: 0.25, count: headDim))
        let wqT = QWTensorValue(
            region: QWMRegion(buffer: wqBuf, base: 0, length: normBytes),
            offset: 0, shape: QWTensorShape([headDim], .f16))
        let wkT = QWTensorValue(
            region: QWMRegion(buffer: wkBuf, base: 0, length: normBytes),
            offset: 0, shape: QWTensorShape([headDim], .f16))
        return QWMBSAAttnOperator.LayerWeights(qkvIdxProj: view, wqIdx: wqT, wkIdx: wkT)
    }

    func testTargetSplitsWaveAlignment() {
        let s1 = QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 4, maxKeys: 4096)
        XCTAssertEqual(s1, 20)
        XCTAssertEqual((4 * s1) % 40, 0)

        for rows in 1...8 {
            let splits = QWMBSAAttnOperator.targetSplits(rows: rows, kvHeads: 4, maxKeys: 4096)
            let cap = rows == 1 ? 64 : 32
            XCTAssertGreaterThanOrEqual(splits, 1)
            XCTAssertLessThanOrEqual(splits, cap)
            XCTAssertEqual((rows * 4 * splits) % 40, 0, "rows=\(rows) splits=\(splits) not 40-core wave aligned")
            let per = ((4096 + splits - 1) / splits + 15) & ~15
            XCTAssertLessThan((splits - 1) * per, 4096, "rows=\(rows) splits=\(splits) wastes an empty split")
        }

        XCTAssertEqual(QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 8, maxKeys: 4096), 20)
        XCTAssertEqual(QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 2, maxKeys: 4096), 20)
        XCTAssertEqual(QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 4, maxKeys: 8192), 40)
        XCTAssertEqual(QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 4, maxKeys: 256), 8)
        XCTAssertEqual(QWMBSAAttnOperator.targetSplits(rows: 1, kvHeads: 4, maxKeys: 4096, gpuCores: 0), 22)
    }

    func testIndexStateCaptureRestoreRoundTrip() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "GPU MBSA test skipped on CI")
        let geo = try Self.makeGeometry()
        let hal = try QWHal(geometry: geo, weightBasis: .standard)
        let op = try QWMBSAAttnOperator(
            hal: hal, geometry: geo, config: geo.blockSparseGQAConfig!,
            kvMode: .f16, layerCount: 1, maxContext: 512, maxPrefillChunk: 16)
        let pos = 137
        let ptr = (op.idxKCaches[0].region.buffer.contents()
            + op.idxKCaches[0].region.base + op.idxKCaches[0].offset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(pos * geo.headDim) { ptr[i] = UInt16(truncatingIfNeeded: i &* 17 &+ 3) }
        let snap = op.captureIndexerState(position: pos)
        XCTAssertEqual(snap.count, op.indexerStateBytes(position: pos))
        for i in 0..<(pos * geo.headDim) { ptr[i] = 0 }
        XCTAssertFalse(op.restoreIndexerState(snap.dropLast(), position: pos))
        XCTAssertTrue(op.restoreIndexerState(snap, position: pos))
        XCTAssertEqual(op.captureIndexerState(position: pos), snap)
    }

    func testDenseEquivalenceAndSparseAttentionAcrossKVModes() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "GPU MBSA test skipped on CI")
        let geo = try Self.makeGeometry()
        for mode in [QWKVMode.f16, QWKVMode.q8, QWKVMode.q4] {
            try verifyMode(mode, geometry: geo)
        }
    }

    private func verifyMode(_ mode: QWKVMode, geometry geo: QWModelGeometry) throws {
        let hal = try QWHal(geometry: geo, weightBasis: .standard)
        let maxCtx = 4608
        let pos = 4400
        let totalKeys = pos + 1
        let headDim = geo.headDim
        let kvElems = geo.kvChannels
        let pages = (totalKeys + 255) / 256 + 1
        let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let kvBytes = pageBytes * (pages * 2 + 4) + tables + (1 << 20)
        let arena = try QWArena(hal: hal, weights: 4 << 20, activations: 0, kv: kvBytes, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: mode, geometry: geo)
        let seq = try cache.allocate(promptTokens: maxCtx)
        let views = try cache.layerViews(seq, layer: 0)

        let ropeTables = QWRopeTables(rows: maxCtx, yarn: nil, geometry: geo)
        let ropeApply = try QWRopeApply(tables: ropeTables, hal: hal, weights: arena.weightsRegion(), arena: arena, geometry: geo)
        let op = try QWMBSAAttnOperator(
            hal: hal, geometry: geo, config: geo.blockSparseGQAConfig!,
            kvMode: mode, layerCount: 1, maxContext: maxCtx, maxPrefillChunk: 128)
        let lw = try dummyLayerWeights(hal: hal, geometry: geo)

        // Populate KV cache (all tokens up to pos).
        var kRows = [[Float]](repeating: [], count: totalKeys)
        var vRows = [[Float]](repeating: [], count: totalKeys)
        let kInBuf = try XCTUnwrap(hal.device.makeBuffer(length: totalKeys * kvElems * 2, options: .storageModeShared))
        let vInBuf = try XCTUnwrap(hal.device.makeBuffer(length: totalKeys * kvElems * 2, options: .storageModeShared))
        for t in 0..<totalKeys {
            var kr = [Float](repeating: 0, count: kvElems)
            var vr = [Float](repeating: 0, count: kvElems)
            for i in 0..<kvElems {
                kr[i] = sin(Float((t &* 13 &+ i) % 197) * 0.031) * 0.4
                vr[i] = cos(Float((t &* 7 &+ i) % 211) * 0.029) * 0.4
            }
            kRows[t] = kr
            vRows[t] = vr
            writeF16(kInBuf, byteOffset: t * kvElems * 2, values: kr)
            writeF16(vInBuf, byteOffset: t * kvElems * 2, values: vr)
        }
        let psoWriter = try hal.pipeline("qwg_kv_paged_write_general")
        guard let (cbW, encW) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        encW.setComputePipelineState(psoWriter)
        views.k.pool.bindTo(encW, index: 0)
        views.v.pool.bindTo(encW, index: 1)
        views.k.table.bind(encW, index: 2)
        views.v.table.bind(encW, index: 3)
        encW.setBuffer(kInBuf, offset: 0, index: 4)
        encW.setBuffer(vInBuf, offset: 0, index: 5)
        encW.setUInt32s(0, totalKeys, startIndex: 6)
        var modeVal: UInt32 = (mode == .f16 ? 0 : (mode == .q8 ? 1 : 2))
        encW.setBytes(&modeVal, length: 4, index: 8)
        encW.dispatchThreadgroups(MTLSize(width: geo.kvHeads, height: totalKeys, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encW.endEncoding(); cbW.commit(); cbW.waitUntilCompleted()
        XCTAssertEqual(cbW.status, .completed)

        // Populate idxKCaches via direct pattern: block 15 carries high positive energy.
        let idxBase = op.idxKCaches[0].region.base + op.idxKCaches[0].offset
        for t in 0..<totalKeys {
            let blk = t / 128
            let amp: Float = (blk == 15) ? 4.0 : 0.05 * Float((blk % 7) + 1)
            let row = (0..<headDim).map { Float(sin(Double($0) * 0.1 + Double(blk))) * amp }
            writeF16(op.idxKCaches[0].region.buffer, byteOffset: idxBase + t * headDim * 2, values: row)
        }
        // Test encodeIndexPrep at pos (exercises norm (1 + w) + NeoX RoPE -> idxQScratch & idxKCaches[pos]).
        let idxQKVBuf = op.idxQKVScratch.region.buffer
        let qkvOff = op.idxQKVScratch.region.base + op.idxQKVScratch.offset
        var qkvIn = [Float](repeating: 0, count: (geo.kvHeads + 1) * headDim)
        for i in 0..<qkvIn.count { qkvIn[i] = 1.0 }
        writeF16(idxQKVBuf, byteOffset: qkvOff, values: qkvIn)
        guard let (cbP, encP) = hal.begin() else { throw NSError(domain: "hal", code: 2) }
        op.encodeIndexPrep(encP, weights: lw, ropeTable: ropeApply.tableTensor, layerIndex: 0, pos0: pos, rows: 1)
        encP.endEncoding(); cbP.commit(); cbP.waitUntilCompleted()
        XCTAssertEqual(cbP.status, .completed)

        // Override idxQ to correlate specifically with block 15 vector.
        let targetK = (0..<headDim).map { Float(sin(Double($0) * 0.1 + Double(15))) }
        for h in 0..<geo.kvHeads {
            writeF16(op.idxQScratch.region.buffer,
                     byteOffset: op.idxQScratch.region.base + op.idxQScratch.offset + h * headDim * 2,
                     values: targetK)
        }

        // Dense-equivalence check at pos = 300.
        let qProjBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qProjOut * 2, options: .storageModeShared))
        let outBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qChannels * 2, options: .storageModeShared))
        let qVals = (0..<geo.qProjOut).map { sin(Float($0 % 97) * 0.07) * 0.5 }
        writeF16(qProjBuf, byteOffset: 0, values: qVals)
        let qT = QWTensorValue(region: QWMRegion(buffer: qProjBuf, base: 0, length: qProjBuf.length),
                               offset: 0, shape: QWTensorShape([geo.qProjOut], .f16))
        let outT = QWTensorValue(region: QWMRegion(buffer: outBuf, base: 0, length: outBuf.length),
                                 offset: 0, shape: QWTensorShape([geo.qChannels], .f16))

        guard let (cbD, encD) = hal.begin() else { throw NSError(domain: "hal", code: 3) }
        op.encodeSelectAndAttend(encD, qgate: qT, kv: views, out: outT, layerIndex: 0, pos0: 300, rows: 1)
        encD.endEncoding(); cbD.commit(); cbD.waitUntilCompleted()
        XCTAssertEqual(cbD.status, .completed)
        let denseCnt = op.blockCountScratch.region.buffer.contents()
            .advanced(by: op.blockCountScratch.region.base + op.blockCountScratch.offset)
            .assumingMemoryBound(to: Int32.self)[0]
        XCTAssertEqual(denseCnt, 3)

        // Sparse check at pos = 4400.
        guard let (cbS, encS) = hal.begin() else { throw NSError(domain: "hal", code: 4) }
        op.encodeSelectAndAttend(encS, qgate: qT, kv: views, out: outT, layerIndex: 0, pos0: pos, rows: 1)
        encS.endEncoding(); cbS.commit(); cbS.waitUntilCompleted()
        XCTAssertEqual(cbS.status, .completed)

        let cntPtr = op.blockCountScratch.region.buffer.contents()
            .advanced(by: op.blockCountScratch.region.base + op.blockCountScratch.offset)
            .assumingMemoryBound(to: Int32.self)
        let listPtr = op.blockListScratch.region.buffer.contents()
            .advanced(by: op.blockListScratch.region.base + op.blockListScratch.offset)
            .assumingMemoryBound(to: Int32.self)
        XCTAssertEqual(cntPtr[0], 32)
        let selected0 = (0..<Int(cntPtr[0])).map { Int(listPtr[$0]) }
        XCTAssertTrue(selected0.contains(15), "expected block 15 in selected remote blocks: \(selected0)")
        for b in 0..<8 { XCTAssertTrue(selected0.contains(b)) }
        for b in 27...34 { XCTAssertTrue(selected0.contains(b)) }

        // CPU reference over selected blocks for head 0 (kv head 0).
        let attnScale = 1.0 / sqrt(Float(headDim))
        var maxScore = -Float.infinity
        var scores: [Float] = []
        var validKeys: [Int] = []
        for blk in selected0 {
            let t0 = blk * 128
            let t1 = min(pos + 1, t0 + 128)
            for j in t0..<t1 {
                var dot: Float = 0
                for d in 0..<headDim { dot += qVals[d] * kRows[j][d] }
                let sc = dot * attnScale
                maxScore = max(maxScore, sc)
                scores.append(sc)
                validKeys.append(j)
            }
        }
        var expSum: Float = 0
        var ref = [Float](repeating: 0, count: headDim)
        for (idx, j) in validKeys.enumerated() {
            let w = exp(scores[idx] - maxScore)
            expSum += w
            for d in 0..<headDim { ref[d] += w * vRows[j][d] }
        }
        for d in 0..<headDim { ref[d] /= expSum }

        let gpuOut = readF16(outBuf, byteOffset: 0, count: headDim)
        let maxDiff = zip(gpuOut, ref).map { abs($0 - $1) }.max() ?? 0
        let tol: Float = (mode == .f16 ? 2e-3 : (mode == .q8 ? 1e-2 : 4e-2))
        XCTAssertLessThan(maxDiff, tol, "mode=\(mode) maxDiff=\(maxDiff)")

        // Helper computing exact per-(row, head) CPU sparse attention reference from per-query selected blocks.
        func cpuRefRowHead(qValsAll: [Float], qIdx: Int, qPos: Int, head: Int, blocks: [Int]) -> [Float] {
            let kvh = head / geo.gqaRatio
            let qStride = geo.attnOutputGate ? 2 : 1
            let qBase = qIdx * geo.qProjOut + head * qStride * headDim
            var m = -Float.infinity
            var scs: [Float] = []
            var keys: [Int] = []
            for blk in blocks {
                let t0 = blk * 128
                let t1 = min(qPos + 1, t0 + 128)
                for j in t0..<t1 {
                    var dot: Float = 0
                    for d in 0..<headDim { dot += qValsAll[qBase + d] * kRows[j][kvh * headDim + d] }
                    let sc = dot * attnScale
                    m = max(m, sc)
                    scs.append(sc)
                    keys.append(j)
                }
            }
            var sumE: Float = 0
            var rOut = [Float](repeating: 0, count: headDim)
            for (idx, j) in keys.enumerated() {
                let w = exp(scs[idx] - m)
                sumE += w
                for d in 0..<headDim { rOut[d] += w * vRows[j][kvh * headDim + d] }
            }
            for d in 0..<headDim { rOut[d] /= sumE }
            return rOut
        }

        // Split-K decode (rows = 1 with partials) & batched verify (rows = 3 with partials).
        let partialsBytes = 4 * geo.qHeads * 64 * (headDim + 2) * 4
        let pBuf = try XCTUnwrap(hal.device.makeBuffer(length: partialsBytes, options: .storageModeShared))
        let partialsReg = QWMRegion(buffer: pBuf, base: 0, length: partialsBytes)

        guard let (cbSp, encSp) = hal.begin() else { throw NSError(domain: "hal", code: 5) }
        op.encodeSelectAndAttend(encSp, qgate: qT, kv: views, out: outT, layerIndex: 0, pos0: pos, rows: 1, partials: partialsReg)
        encSp.endEncoding(); cbSp.commit(); cbSp.waitUntilCompleted()
        XCTAssertEqual(cbSp.status, .completed)
        let gpuSplitOut = readF16(outBuf, byteOffset: 0, count: headDim)
        let splitMaxDiff = zip(gpuSplitOut, ref).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(splitMaxDiff, tol, "split-K rows=1 mode=\(mode) maxDiff=\(splitMaxDiff)")

        // Batched verify rows=3 starting at pos0 = 4398.
        let verRows = 3
        let vPos0 = pos - 2
        let qVerBuf = try XCTUnwrap(hal.device.makeBuffer(length: verRows * geo.qProjOut * 2, options: .storageModeShared))
        let outVerBuf = try XCTUnwrap(hal.device.makeBuffer(length: verRows * geo.qChannels * 2, options: .storageModeShared))
        let qVerVals = (0..<(verRows * geo.qProjOut)).map { sin(Float($0 % 113) * 0.05) * 0.5 }
        writeF16(qVerBuf, byteOffset: 0, values: qVerVals)
        for r in 0..<verRows {
            for h in 0..<geo.kvHeads {
                writeF16(op.idxQScratch.region.buffer,
                         byteOffset: op.idxQScratch.region.base + op.idxQScratch.offset + (r * geo.kvHeads + h) * headDim * 2,
                         values: targetK)
            }
        }
        let qVerT = QWTensorValue(region: QWMRegion(buffer: qVerBuf, base: 0, length: qVerBuf.length),
                                  offset: 0, shape: QWTensorShape([verRows * geo.qProjOut], .f16))
        let outVerT = QWTensorValue(region: QWMRegion(buffer: outVerBuf, base: 0, length: outVerBuf.length),
                                    offset: 0, shape: QWTensorShape([verRows * geo.qChannels], .f16))
        guard let (cbV, encV) = hal.begin() else { throw NSError(domain: "hal", code: 6) }
        op.encodeSelectAndAttend(encV, qgate: qVerT, kv: views, out: outVerT, layerIndex: 0, pos0: vPos0, rows: verRows, partials: partialsReg)
        encV.endEncoding(); cbV.commit(); cbV.waitUntilCompleted()
        XCTAssertEqual(cbV.status, .completed)
        for r in 0..<verRows {
            let cnt = Int(cntPtr[r * geo.kvHeads])
            let blks = (0..<cnt).map { Int(listPtr[r * geo.kvHeads * op.listStride + $0]) }
            let got = readF16(outVerBuf, byteOffset: r * geo.qChannels * 2, count: headDim)
            let expR = cpuRefRowHead(qValsAll: qVerVals, qIdx: r, qPos: vPos0 + r, head: 0, blocks: blks)
            let d = zip(got, expR).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(d, tol, "verify rows=3 r=\(r) mode=\(mode) maxDiff=\(d)")
        }

        // Prefill NAX sparse check: T = 70 starting at pos0 = 4080 (crosses 4096 dense->sparse mid-tile + 6-row tail).
        let pRows = 70
        let pPos0 = 4080
        let qPreBuf = try XCTUnwrap(hal.device.makeBuffer(length: pRows * geo.qProjOut * 2, options: .storageModeShared))
        let outPreBuf = try XCTUnwrap(hal.device.makeBuffer(length: pRows * geo.qChannels * 2, options: .storageModeShared))
        let qPreVals = (0..<(pRows * geo.qProjOut)).map { cos(Float($0 % 131) * 0.04) * 0.5 }
        writeF16(qPreBuf, byteOffset: 0, values: qPreVals)
        for r in 0..<pRows {
            for h in 0..<geo.kvHeads {
                writeF16(op.idxQScratch.region.buffer,
                         byteOffset: op.idxQScratch.region.base + op.idxQScratch.offset + (r * geo.kvHeads + h) * headDim * 2,
                         values: targetK)
            }
        }
        let qPreT = QWTensorValue(region: QWMRegion(buffer: qPreBuf, base: 0, length: qPreBuf.length),
                                  offset: 0, shape: QWTensorShape([pRows * geo.qProjOut], .f16))
        let outPreT = QWTensorValue(region: QWMRegion(buffer: outPreBuf, base: 0, length: outPreBuf.length),
                                    offset: 0, shape: QWTensorShape([pRows * geo.qChannels], .f16))
        guard let (cbNax, encNax) = hal.begin() else { throw NSError(domain: "hal", code: 7) }
        op.encodeSelectAndAttend(encNax, qgate: qPreT, kv: views, out: outPreT, layerIndex: 0, pos0: pPos0, rows: pRows)
        encNax.endEncoding(); cbNax.commit(); cbNax.waitUntilCompleted()
        XCTAssertEqual(cbNax.status, .completed)

        // Compare NAX sparse prefill output against per-query reference selection (via psoSelectBlocks).
        let psoSel = op.psoSelectBlocks
        guard let (cbSel, encSel) = hal.begin() else { throw NSError(domain: "hal", code: 8) }
        encSel.setComputePipelineState(psoSel)
        op.idxQScratch.slice(byteOffset: 0, count: pRows * geo.kvHeads * headDim, dtype: .f16).bind(encSel, index: 0)
        op.idxKCaches[0].bind(encSel, index: 1)
        op.blockListScratch.bind(encSel, index: 2)
        op.blockCountScratch.bind(encSel, index: 3)
        var sp0: (UInt32, UInt32, UInt32, UInt32) = (UInt32(pPos0), UInt32(geo.kvHeads), UInt32(headDim), 128)
        var sp1: (UInt32, UInt32, UInt32, UInt32) = (4, 8, 16, UInt32(op.listStride))
        encSel.setBytes(&sp0, length: 16, index: 4)
        encSel.setBytes(&sp1, length: 16, index: 5)
        encSel.dispatchThreadgroups(MTLSize(width: geo.kvHeads, height: pRows, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encSel.endEncoding(); cbSel.commit(); cbSel.waitUntilCompleted()
        XCTAssertEqual(cbSel.status, .completed)

        let checkRows = [0, 15, 16, 31, 32, 63, 64, 69]
        for r in checkRows {
            for h in [0, 7, 15, 24, 31] {
                let kvh = h / geo.gqaRatio
                let cnt = Int(cntPtr[r * geo.kvHeads + kvh])
                let blks = (0..<cnt).map { Int(listPtr[(r * geo.kvHeads + kvh) * op.listStride + $0]) }
                let got = readF16(outPreBuf, byteOffset: (r * geo.qChannels + h * headDim) * 2, count: headDim)
                let expR = cpuRefRowHead(qValsAll: qPreVals, qIdx: r, qPos: pPos0 + r, head: h, blocks: blks)
                let d = zip(got, expR).map { abs($0 - $1) }.max() ?? 0
                XCTAssertLessThan(d, tol, "NAX prefill r=\(r) h=\(h) mode=\(mode) maxDiff=\(d)")
            }
        }
    }

    func testRotatedBasisSingleSplitMatchesSplitK() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "GPU MBSA test skipped on CI")
        let geo = try Self.makeGeometry()
        let hal = try QWHal(geometry: geo, weightBasis: .rotated)
        let maxCtx = 4608
        let pos = 4400
        let totalKeys = pos + 1
        let headDim = geo.headDim
        let kvElems = geo.kvChannels
        let pages = (totalKeys + 255) / 256 + 1
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geo)
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let kvBytes = pageBytes * (pages * 2 + 4) + tables + (1 << 20)
        let arena = try QWArena(hal: hal, weights: 4 << 20, activations: 0, kv: kvBytes, gdnStates: 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: .f16, geometry: geo)
        let seq = try cache.allocate(promptTokens: maxCtx)
        let views = try cache.layerViews(seq, layer: 0)
        let op = try QWMBSAAttnOperator(
            hal: hal, geometry: geo, config: geo.blockSparseGQAConfig!,
            kvMode: .f16, layerCount: 1, maxContext: maxCtx, maxPrefillChunk: 64)

        let kInBuf = try XCTUnwrap(hal.device.makeBuffer(length: totalKeys * kvElems * 2, options: .storageModeShared))
        let vInBuf = try XCTUnwrap(hal.device.makeBuffer(length: totalKeys * kvElems * 2, options: .storageModeShared))
        for t in 0..<totalKeys {
            let kr = (0..<kvElems).map { sin(Float((t &* 13 &+ $0) % 197) * 0.031) * 0.4 }
            let vr = (0..<kvElems).map { cos(Float((t &* 7 &+ $0) % 211) * 0.029) * 0.4 }
            writeF16(kInBuf, byteOffset: t * kvElems * 2, values: kr)
            writeF16(vInBuf, byteOffset: t * kvElems * 2, values: vr)
        }
        let psoWriter = try hal.pipeline("qwg_kv_paged_write_general")
        guard let (cbW, encW) = hal.begin() else { throw NSError(domain: "hal", code: 10) }
        encW.setComputePipelineState(psoWriter)
        views.k.pool.bindTo(encW, index: 0)
        views.v.pool.bindTo(encW, index: 1)
        views.k.table.bind(encW, index: 2)
        views.v.table.bind(encW, index: 3)
        encW.setBuffer(kInBuf, offset: 0, index: 4)
        encW.setBuffer(vInBuf, offset: 0, index: 5)
        encW.setUInt32s(0, totalKeys, startIndex: 6)
        var modeVal: UInt32 = 0
        encW.setBytes(&modeVal, length: 4, index: 8)
        encW.dispatchThreadgroups(MTLSize(width: geo.kvHeads, height: totalKeys, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encW.endEncoding(); cbW.commit(); cbW.waitUntilCompleted()
        XCTAssertEqual(cbW.status, .completed)

        let idxBase = op.idxKCaches[0].region.base + op.idxKCaches[0].offset
        for t in 0..<totalKeys {
            let blk = t / 128
            let amp: Float = (blk == 15) ? 4.0 : 0.05 * Float((blk % 7) + 1)
            let row = (0..<headDim).map { Float(sin(Double($0) * 0.1 + Double(blk))) * amp }
            writeF16(op.idxKCaches[0].region.buffer, byteOffset: idxBase + t * headDim * 2, values: row)
        }
        let targetK = (0..<headDim).map { Float(sin(Double($0) * 0.1 + Double(15))) }
        for h in 0..<geo.kvHeads {
            writeF16(op.idxQScratch.region.buffer,
                     byteOffset: op.idxQScratch.region.base + op.idxQScratch.offset + h * headDim * 2,
                     values: targetK)
        }

        let qBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qProjOut * 2, options: .storageModeShared))
        let outSingleBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qChannels * 2, options: .storageModeShared))
        let outSplitBuf = try XCTUnwrap(hal.device.makeBuffer(length: geo.qChannels * 2, options: .storageModeShared))
        writeF16(qBuf, byteOffset: 0, values: (0..<geo.qProjOut).map { sin(Float($0 % 97) * 0.07) * 0.5 })
        let qT = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                               offset: 0, shape: QWTensorShape([geo.qProjOut], .f16))
        let outSingleT = QWTensorValue(region: QWMRegion(buffer: outSingleBuf, base: 0, length: outSingleBuf.length),
                                       offset: 0, shape: QWTensorShape([geo.qChannels], .f16))
        let outSplitT = QWTensorValue(region: QWMRegion(buffer: outSplitBuf, base: 0, length: outSplitBuf.length),
                                      offset: 0, shape: QWTensorShape([geo.qChannels], .f16))
        let partialsBytes = geo.qHeads * 64 * (headDim + 2) * 4
        let pBuf = try XCTUnwrap(hal.device.makeBuffer(length: partialsBytes, options: .storageModeShared))
        let partialsReg = QWMRegion(buffer: pBuf, base: 0, length: partialsBytes)

        guard let (cb1, enc1) = hal.begin() else { throw NSError(domain: "hal", code: 11) }
        op.encodeSelectAndAttend(enc1, qgate: qT, kv: views, out: outSingleT, layerIndex: 0, pos0: pos, rows: 1, partials: nil)
        enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()
        XCTAssertEqual(cb1.status, .completed)

        guard let (cb2, enc2) = hal.begin() else { throw NSError(domain: "hal", code: 12) }
        op.encodeSelectAndAttend(enc2, qgate: qT, kv: views, out: outSplitT, layerIndex: 0, pos0: pos, rows: 1, partials: partialsReg)
        enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
        XCTAssertEqual(cb2.status, .completed)

        let gotSingle = readF16(outSingleBuf, byteOffset: 0, count: geo.qChannels)
        let gotSplit = readF16(outSplitBuf, byteOffset: 0, count: geo.qChannels)
        let maxDiff = zip(gotSingle, gotSplit).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 2e-3, "rotated splits=1 vs split-K maxDiff=\(maxDiff)")
    }
}
