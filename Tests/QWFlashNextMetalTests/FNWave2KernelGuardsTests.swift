// FNWave2KernelGuardsTests.swift — Tests for split FNGPULayer GDN, DFlash/DSpine bounds, and seed KV zero scales.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGDN
import QWGEMM
import QWHAL
import QWKV
import QWMTP
import QWQuant
@testable import QWFlashNext

final class FNWave2KernelGuardsTests: XCTestCase {
    private static func makeGDNDescriptor(hidden: Int = 256) -> QWModelDescriptor {
        QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp", capabilities: [.mtp],
            hidden: hidden, vocab: 256, denseIntermediate: nil, numLayers: 1,
            layerTypes: ["linear_attention"], fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 512, qHeads: 4, kvHeads: 2, headDim: 64,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e6,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 32, expertsPerTok: 4, moeIntermediate: 64,
            sharedExpertIntermediate: 64, expertGateType: nil,
            hcCount: 1, hcLowrank: 32,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 255
        )
    }

    private func makeHQ4(_ arena: QWArena, n: Int, k: Int, seedInit: UInt32) throws -> QWWeightView {
        let groups = k / 32
        var seed = seedInit
        var raw = Data(count: n * groups * 20)
        raw.withUnsafeMutableBytes { (buf: UnsafeMutableRawBufferPointer) in
            guard let ptr = buf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * 20
                    let sc = Float16(0.02 * (1.0 + 0.1 * Float((r + g) % 5))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF); ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = 0; ptr[base + 3] = 0
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        let tv = try arena.weightsRegion().allocate(QWTensorShape([raw.count / 2], .f16))
        arena.copyIn(tv, from: raw)
        return try XCTUnwrap(QWWeightView(blocks: tv, n: n, k: k, isRotated: true))
    }

    func testFNGPUSplitQKVAndZMatchesAcrossBatchAndSingleSlot() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        let desc = Self.makeGDNDescriptor()
        let hal = try QWHal(geometry: QWModelGeometry(descriptor: desc))
        let h = desc.hidden, conv = desc.gdnQKVChannels, value = desc.gdnZChannels
        let vh = desc.gdnVHeads, kh = desc.gdnQKHeads, hd = desc.gdnHeadDim
        let arena = try QWArena(hal: hal, weights: 16 << 20, activations: 64 << 20, kv: 256, gdnStates: 128 << 20)
        let act = arena.activationsRegion(), gdnReg = arena.gdnRegion()

        let wQkv = try makeHQ4(arena, n: conv, k: h, seedInit: 10)
        let wZ = try makeHQ4(arena, n: value, k: h, seedInit: 20)
        let wOut = try makeHQ4(arena, n: h, k: value, seedInit: 30)
        let baRows = 2 * vh
        let baZeroRaw = Data(count: baRows * (h / 32) * 20)
        let baBuf = try arena.weightsRegion().allocate(QWTensorShape([baZeroRaw.count / 2], .f16))
        arena.copyIn(baBuf, from: baZeroRaw)
        let wBA = try XCTUnwrap(QWWeightView(blocks: baBuf, n: baRows, k: h, isRotated: true))
        let dummy = try makeHQ4(arena, n: 64, k: h, seedInit: 50)
        let convW = try arena.weightsRegion().allocate(QWTensorShape([conv * 4], .f32))
        arena.write(convW, from: (0..<(conv * 4)).map { Float($0 % 5) * 0.05 })
        let normW = try arena.weightsRegion().allocate(QWTensorShape([hd], .f32))
        arena.write(normW, from: [Float](repeating: 1.0, count: hd))
        let aLog = try arena.weightsRegion().allocate(QWTensorShape([vh], .f32))
        arena.write(aLog, from: [Float](repeating: -1.0, count: vh))
        let dtBias = try arena.weightsRegion().allocate(QWTensorShape([vh], .f32))
        arena.write(dtBias, from: [Float](repeating: 0.0, count: vh))
        let baF32 = try arena.weightsRegion().allocate(QWTensorShape([baRows * h], .f32))
        arena.write(baF32, from: [Float](repeating: 0.0, count: baRows * h))
        let bF32 = baF32.slice(byteOffset: 0, count: vh * h, dtype: .f32)
        let aF32 = baF32.slice(byteOffset: vh * h * 4, count: vh * h, dtype: .f32)
        let bfZero = try arena.weightsRegion().allocate(QWTensorShape([h], .bf16))
        arena.write(bfZero, from: [UInt16](repeating: 0, count: h))
        let scZero = try arena.weightsRegion().allocate(QWTensorShape([h], .f32))
        arena.write(scZero, from: [Float](repeating: 0, count: h))

        let moe = FNGPUMoE(gateAndSharedBF16: bfZero, gateUp: dummy, down: dummy,
                           sharedGateUp: dummy, sharedDown: dummy, sharedGateScalar: scZero,
                           experts: 32, topK: 4, intermediate: 64, sharedIntermediate: 64)

        func makeMixer(slotCount: Int) throws -> (FNGPUGDNMixer, [FNBatchSlotState]) {
            var states: [QWTensorValue] = []
            var convs: [QWTensorValue] = []
            for _ in 0..<8 {
                let sT = try gdnReg.allocate(QWTensorShape([vh * hd * hd], .f32))
                memset(arena.cpuPointer(of: sT), 0, vh * hd * hd * 4)
                states.append(sT)
                let cT = try gdnReg.allocate(QWTensorShape([3 * conv], .f32))
                memset(arena.cpuPointer(of: cT), 0, 3 * conv * 4)
                convs.append(cT)
            }
            var slotStates: [FNBatchSlotState] = []
            for _ in 0..<slotCount {
                let s0 = try gdnReg.allocate(QWTensorShape([vh * hd * hd], .f32))
                let s1 = try gdnReg.allocate(QWTensorShape([vh * hd * hd], .f32))
                memset(arena.cpuPointer(of: s0), 0, vh * hd * hd * 4)
                memset(arena.cpuPointer(of: s1), 0, vh * hd * hd * 4)
                let c0 = try gdnReg.allocate(QWTensorShape([3 * conv], .f32))
                let c1 = try gdnReg.allocate(QWTensorShape([3 * conv], .f32))
                memset(arena.cpuPointer(of: c0), 0, 3 * conv * 4)
                memset(arena.cpuPointer(of: c1), 0, 3 * conv * 4)
                slotStates.append(FNBatchSlotState(gdnStates: [[s0, s1]], gdnStateParity: [0],
                                                   gdnConvState: [[c0, c1]], gdnConvParity: [0],
                                                   pleHistory: [], pleParity: [], highWaterPos: 0))
            }
            let mix = FNGPUGDNMixer(qkv: wQkv, z: wZ, qkvZ: nil, out: wOut, inProjBA: wBA,
                                    conv: convW, normWeight: normW, aLog: aLog, dtBias: dtBias,
                                    inProjB: bF32, inProjA: aF32, inProjBA32: baF32,
                                    states: states, convState: convs,
                                    vHeads: vh, kHeads: kh, kDim: hd, vDim: hd, convDim: conv)
            return (mix, slotStates)
        }

        let (gdnBatch, mutSlots) = try makeMixer(slotCount: 2)
        var slots = mutSlots
        let (gdnS0, _) = try makeMixer(slotCount: 0)
        let (gdnS1, _) = try makeMixer(slotCount: 0)

        let ctx = try FNGPUContext(hal: hal, doors: [], geometry: QWModelGeometry(descriptor: desc))
        let xVals = (0..<(2 * h)).map { Float(($0 * 7) % 19 - 9) * 0.0625 }

        let scrBatch = try FNGPUScratch(arena: arena, T: 2, hidden: h, hc: 1, lowrank: 32,
                                        moe: moe, ple: nil, gdn: gdnBatch, attn: nil)
        arena.write(scrBatch.rot32, from: xVals)
        let outBatch = try act.allocate(QWTensorShape([2 * h], .f32))

        let scrS0 = try FNGPUScratch(arena: arena, T: 1, hidden: h, hc: 1, lowrank: 32,
                                     moe: moe, ple: nil, gdn: gdnS0, attn: nil)
        arena.write(scrS0.rot32, from: Array(xVals[0..<h]))
        let outS0 = try act.allocate(QWTensorShape([h], .f32))

        let scrS1 = try FNGPUScratch(arena: arena, T: 1, hidden: h, hc: 1, lowrank: 32,
                                     moe: moe, ple: nil, gdn: gdnS1, attn: nil)
        arena.write(scrS1.rot32, from: Array(xVals[h..<(2 * h)]))
        let outS1 = try act.allocate(QWTensorShape([h], .f32))

        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try FNGPULayerPasses.gdnBatch(enc, ctx: ctx, layerIndex: 0, g: gdnBatch, s: scrBatch,
                                      out: outBatch, activeSlots: [0, 1], slotStates: &slots)
        try FNGPULayerPasses.gdn(enc, ctx: ctx, g: gdnS0, s: scrS0, mixed: scrS0.rot32, out: outS0)
        try FNGPULayerPasses.gdn(enc, ctx: ctx, g: gdnS1, s: scrS1, mixed: scrS1.rot32, out: outS1)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let bOut = arena.read(outBatch, as: Float.self, count: 2 * h)
        let ref0 = arena.read(outS0, as: Float.self, count: h)
        let ref1 = arena.read(outS1, as: Float.self, count: h)
        for i in 0..<h {
            XCTAssertTrue(bOut[i].isFinite && bOut[h + i].isFinite)
            let tol0 = Float(1e-5) * max(1.0, abs(ref0[i]))
            let tol1 = Float(1e-5) * max(1.0, abs(ref1[i]))
            XCTAssertEqual(bOut[i], ref0[i], accuracy: tol0, "slot0 ch \(i)")
            XCTAssertEqual(bOut[h + i], ref1[i], accuracy: tol1, "slot1 ch \(i)")
        }
    }

    func testDFlashTop16EdgeAndLargeVocabAndDSpineChainBounds() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        _ = dev
        let hal = try QWHal(geometry: QWModelGeometry(descriptor: Self.makeGDNDescriptor()))
        let pStage1 = try hal.pipeline("qwg_dflash_top16_stage1")
        let pStage2 = try hal.pipeline("qwg_dflash_top16_stage2")

        // 1. Small/degenerate vocabs: 0 and 19 with -inf/NaN.
        let arena = try QWArena(hal: hal, weights: 4096, activations: 4 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let pVals = try act.allocate(QWTensorShape([256 * 16], .f32))
        let pIds = try act.allocate(QWTensorShape([256 * 16], .i32))
        let unary = try act.allocate(QWTensorShape([16], .f32))
        let cands = try act.allocate(QWTensorShape([16], .i32))

        // Case A: vocab = 0
        let dummyLogits = try act.allocate(QWTensorShape([16], .f32))
        guard let (cb0, enc0) = hal.begin() else { throw XCTSkip("no cb") }
        enc0.setComputePipelineState(pStage1)
        enc0.bindTensors(dummyLogits, pVals, pIds, startIndex: 0)
        enc0.setUInt32s(0, 4, startIndex: 3)
        enc0.dispatchThreadgroups(MTLSize(width: 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc0.memoryBarrier(scope: .buffers)
        enc0.setComputePipelineState(pStage2)
        enc0.bindTensors(pVals, pIds, unary, cands, startIndex: 0)
        enc0.setUInt32s(4, 0, startIndex: 4)
        enc0.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb0, enc0)
        cb0.waitUntilCompleted()
        XCTAssertNil(cb0.error)

        // Case B: vocab = 420_000 (chunkStride > 1536 triggers large-vocab scan loop).
        let bigVocab = 420_000
        let bigLogits = try act.allocate(QWTensorShape([bigVocab], .f32))
        let bigPtr = arena.cpuPointer(of: bigLogits).assumingMemoryBound(to: Float.self)
        for i in 0..<bigVocab { bigPtr[i] = -10.0 }
        bigPtr[400_005] = 99.0
        bigPtr[419_999] = 88.0
        bigPtr[123] = 77.0
        bigPtr[395_000] = 66.0

        guard let (cb1, enc1) = hal.begin() else { throw XCTSkip("no cb") }
        enc1.setComputePipelineState(pStage1)
        enc1.bindTensors(bigLogits, pVals, pIds, startIndex: 0)
        enc1.setUInt32s(bigVocab, 4, startIndex: 3)
        enc1.dispatchThreadgroups(MTLSize(width: 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc1.memoryBarrier(scope: .buffers)
        enc1.setComputePipelineState(pStage2)
        enc1.bindTensors(pVals, pIds, unary, cands, startIndex: 0)
        enc1.setUInt32s(4, bigVocab, startIndex: 4)
        enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        hal.end(cb1, enc1)
        cb1.waitUntilCompleted()
        XCTAssertNil(cb1.error)
        XCTAssertEqual(arena.read(cands, as: Int32.self, count: 4), [400_005, 419_999, 123, 395_000])

        // DSpine refine_chain with non-finite entries in refineTable must stay bounded.
        let refineTable = try act.allocate(QWTensorShape([2 * 16 * 16], .f32))
        let tblPtr = arena.cpuPointer(of: refineTable).assumingMemoryBound(to: Float.self)
        for i in 0..<(2 * 16 * 16) { tblPtr[i] = -.infinity }
        tblPtr[3] = .nan
        tblPtr[7] = -5.0
        tblPtr[(16 + 7) * 16 + 2] = 4.0
        let chainCands = try act.allocate(QWTensorShape([2 * 16], .i32))
        let cPtr = arena.cpuPointer(of: chainCands).assumingMemoryBound(to: Int32.self)
        for i in 0..<(2 * 16) { cPtr[i] = Int32(100 + i) }
        let draftOut = try act.allocate(QWTensorShape([2], .i32))
        let pChain = try hal.pipeline("qwg_dspine_refine_chain")
        guard let (cb2, enc2) = hal.begin() else { throw XCTSkip("no cb") }
        enc2.setComputePipelineState(pChain)
        enc2.bindTensors(refineTable, chainCands, draftOut, startIndex: 0)
        enc2.setUInt32s(2, 16, startIndex: 3)
        enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)
        XCTAssertEqual(arena.read(draftOut, as: Int32.self, count: 2), [107, 118])
    }

    func testMTPSeedKVZeroRowAvoidsScaleZeroInfNaN() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        let hal = try QWHal(geometry: .qwen38_27b)
        let pso = try hal.pipeline("qwg_mtp_seed_kv")
        for mode in [QWKVMode.q8, QWKVMode.q4] {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: .qwen38_27b)
            guard let kSrc = hal.device.makeBuffer(length: 2048, options: .storageModeShared),
                  let vSrc = hal.device.makeBuffer(length: 2048, options: .storageModeShared),
                  let kPool = hal.device.makeBuffer(length: pageBytes, options: .storageModeShared),
                  let vPool = hal.device.makeBuffer(length: pageBytes, options: .storageModeShared),
                  let kTbl = hal.device.makeBuffer(length: 8, options: .storageModeShared),
                  let vTbl = hal.device.makeBuffer(length: 8, options: .storageModeShared) else {
                throw XCTSkip("buffer allocation failed")
            }
            memset(kSrc.contents(), 0, 2048)
            memset(vSrc.contents(), 0, 2048)
            memset(kPool.contents(), 0x5A, pageBytes)
            memset(vPool.contents(), 0x5A, pageBytes)
            kTbl.contents().assumingMemoryBound(to: Int32.self)[0] = 0
            vTbl.contents().assumingMemoryBound(to: Int32.self)[0] = 0
            var rowU: UInt32 = 0
            var modeU: UInt32 = (mode == .q8) ? 1 : 2
            guard let (cb, enc) = hal.begin() else { throw XCTSkip("no cb") }
            enc.setComputePipelineState(pso)
            enc.setBuffer(kSrc, offset: 0, index: 0)
            enc.setBuffer(vSrc, offset: 0, index: 1)
            enc.setBuffer(kPool, offset: 0, index: 2)
            enc.setBuffer(vPool, offset: 0, index: 3)
            enc.setBuffer(kTbl, offset: 0, index: 4)
            enc.setBuffer(vTbl, offset: 0, index: 5)
            enc.setBytes(&rowU, length: 4, index: 6)
            enc.setBytes(&modeU, length: 4, index: 7)
            enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let kp = kPool.contents().assumingMemoryBound(to: UInt8.self)
            let vp = vPool.contents().assumingMemoryBound(to: UInt8.self)
            let payload = (mode == .q8) ? 1024 : 512
            for i in 0..<payload {
                XCTAssertEqual(kp[i], 0, "q-mode \(mode) k[\(i)] should be 0")
                XCTAssertEqual(vp[i], 0, "q-mode \(mode) v[\(i)] should be 0")
            }
        }
    }

    func testFNGPUBf16FromF32AllocatesHonestBF16DtypeAndRoutes() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        let hal = try QWHal(geometry: QWModelGeometry(descriptor: Self.makeGDNDescriptor(hidden: 64)))
        let e = 64, k = 64, rows = 128
        let wF32 = (0..<(e * k)).map { Float(($0 * 11) % 23 - 11) * 0.0625 }
        let xVals = (0..<(rows * k)).map { Float(($0 * 5) % 17 - 8) * 0.125 }
        let arena = try QWArena(hal: hal, weights: e * k * 4 + 4096, activations: rows * (e + k) * 8 + 4096, kv: 0, gdnStates: 0)
        let wBF16 = try FNGPULoader.bf16FromF32(arena, wF32)
        XCTAssertEqual(wBF16.shape.dtype, .bf16)
        let xT = try arena.activationsRegion().allocate(QWTensorShape([rows * k], .f32))
        let y16T = try arena.activationsRegion().allocate(QWTensorShape([16 * e], .f32))
        let y128T = try arena.activationsRegion().allocate(QWTensorShape([rows * e], .f32))
        arena.write(xT, from: xVals)
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wBF16, x: xT, y: y16T, rows: 16, n: e, k: k)
        try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wBF16, x: xT, y: y128T, rows: rows, n: e, k: k)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let wQ = wF32.map { Float(bitPattern: UInt32(QWBF16.fromFloatRNE($0)) << 16) }
        let got16 = arena.read(y16T, as: Float.self, count: 16 * e)
        let got128 = arena.read(y128T, as: Float.self, count: rows * e)
        for r in [0, 15, 127] {
            for c in [0, 31, 63] {
                var ref: Float = 0
                for i in 0..<k { ref += wQ[c * k + i] * xVals[r * k + i] }
                if r < 16 { XCTAssertEqual(got16[r * e + c], ref, accuracy: 1e-4) }
                XCTAssertEqual(got128[r * e + c], ref, accuracy: 2e-4)
            }
        }
    }
}
