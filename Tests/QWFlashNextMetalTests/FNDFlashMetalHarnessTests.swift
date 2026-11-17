// FNDFlashMetalHarnessTests.swift — GPU parity and prefix-cache KV round-trip
// tests for Flash-Next DFlash capture, head projection, and sliding-window state.
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWMTP
import QWQuant
@testable import QWFlashNext

final class FNDFlashMetalHarnessTests: XCTestCase {
    private static func makeDescriptor(hidden: Int = 256, numLayers: Int = 4) -> QWModelDescriptor {
        let types = (0..<numLayers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        return QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.mtp],
            hidden: hidden, vocab: 256, denseIntermediate: nil, numLayers: numLayers,
            layerTypes: types, fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 4096, qHeads: 4, kvHeads: 2, headDim: 64,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 4, expertsPerTok: 2, moeIntermediate: 128,
            sharedExpertIntermediate: 128, expertGateType: nil,
            hcCount: 4, hcLowrank: 64,
            gdnQKHeads: 2, gdnVHeads: 4, gdnHeadDim: 64,
            gdnQKVChannels: 512, gdnZChannels: 256, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 255
        )
    }

    private func makeHal() throws -> QWHal {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try QWHal(geometry: QWModelGeometry(descriptor: Self.makeDescriptor()))
    }

    private func makeSlot(hal: QWHal, config: QWDFlashConfig) throws -> QWDFlashSessionSlot {
        let kvDim = config.numKeyValueHeads * config.headDim
        let layerElems = config.maxCacheSize * kvDim
        let bytes = config.numLayers * layerElems * 2
        guard let kBuf = hal.device.makeBuffer(length: bytes, options: .storageModeShared),
              let vBuf = hal.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw XCTSkip("failed to allocate synthetic KV buffers")
        }
        let kReg = QWMRegion(buffer: kBuf, base: 0, length: bytes)
        let vReg = QWMRegion(buffer: vBuf, base: 0, length: bytes)
        let kSlab = QWTensorValue(region: kReg, offset: 0, shape: QWTensorShape([config.numLayers * layerElems], .f16))
        let vSlab = QWTensorValue(region: vReg, offset: 0, shape: QWTensorShape([config.numLayers * layerElems], .f16))
        let kLayers = (0..<config.numLayers).map {
            QWTensorValue(region: kReg, offset: $0 * layerElems * 2, shape: QWTensorShape([layerElems], .f16))
        }
        let vLayers = (0..<config.numLayers).map {
            QWTensorValue(region: vReg, offset: $0 * layerElems * 2, shape: QWTensorShape([layerElems], .f16))
        }
        return QWDFlashSessionSlot(kSlab: kSlab, vSlab: vSlab, kLayers: kLayers, vLayers: vLayers)
    }

    func testDFlashSlidingWindowPrefixKVAndValidationRoundTrip() throws {
        let hal = try makeHal()
        let cfg = QWDFlashConfig(
            hiddenSize: 256, numLayers: 3, numAttentionHeads: 4, numKeyValueHeads: 2,
            headDim: 64, intermediateSize: 512, vocabSize: 256,
            slidingWindow: 33, blockSize: 8, maskTokenId: 250,
            targetLayerIds: [0, 1, 2])
        let marker = FNGPUDFlash.dflashGeometryMarker(config: cfg)
        let srcSlot = try makeSlot(hal: hal, config: cfg)
        let dstSlot = try makeSlot(hal: hal, config: cfg)

        let rowBytes = cfg.numKeyValueHeads * cfg.headDim * 2
        for (l, (kT, vT)) in zip(srcSlot.kLayers, srcSlot.vLayers).enumerated() {
            let kPtr = kT.region.buffer.contents().advanced(by: kT.region.base + kT.offset).assumingMemoryBound(to: UInt8.self)
            let vPtr = vT.region.buffer.contents().advanced(by: vT.region.base + vT.offset).assumingMemoryBound(to: UInt8.self)
            for b in 0..<(cfg.maxCacheSize * rowBytes) {
                kPtr[b] = UInt8((l * 31 + b) & 0xFF)
                vPtr[b] = UInt8((l * 17 + b * 3) & 0xFF)
            }
        }

        // Test both exact-fit (N=12 <= 32) and wrapped sliding-window (N=90 > 32).
        for tokenCount in [12, 90] {
            let expectedBytes = FNGPUDFlash.prefixPayloadBytes(config: cfg, tokenCount: tokenCount)
            var blob = Data(count: expectedBytes)
            var wOff = 0
            try blob.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                let ptr = try XCTUnwrap(raw.baseAddress)
                FNGPUDFlash.appendPrefixKV(to: ptr, at: &wOff, slots: srcSlot,
                                           config: cfg, marker: marker, tokenCount: tokenCount)
            }
            XCTAssertEqual(wOff, expectedBytes)
            XCTAssertTrue(FNGPUDFlash.validatePrefixBlob(
                data: blob, expectedMarker: marker, expectedTokens: tokenCount, config: cfg))

            var rOff = 0
            let restoredOff = try blob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
                let ptr = try XCTUnwrap(raw.baseAddress)
                return FNGPUDFlash.restorePrefixKV(from: ptr, dataCount: blob.count, offset: &rOff,
                                                   slots: dstSlot, config: cfg, expectedMarker: marker, expectedTokens: tokenCount)
            }
            XCTAssertEqual(restoredOff, tokenCount)
            XCTAssertEqual(rOff, expectedBytes)

            let activeBytes = min(tokenCount, cfg.maxCacheSize) * rowBytes
            for l in 0..<cfg.numLayers {
                let kSrc = Data(bytes: srcSlot.kLayers[l].region.buffer.contents().advanced(by: srcSlot.kLayers[l].offset), count: activeBytes)
                let kDst = Data(bytes: dstSlot.kLayers[l].region.buffer.contents().advanced(by: dstSlot.kLayers[l].offset), count: activeBytes)
                let vSrc = Data(bytes: srcSlot.vLayers[l].region.buffer.contents().advanced(by: srcSlot.vLayers[l].offset), count: activeBytes)
                let vDst = Data(bytes: dstSlot.vLayers[l].region.buffer.contents().advanced(by: dstSlot.vLayers[l].offset), count: activeBytes)
                XCTAssertEqual(kDst, kSrc)
                XCTAssertEqual(vDst, vSrc)
            }

            // Tampered geometry marker or wrong token count must fail cleanly.
            XCTAssertFalse(FNGPUDFlash.validatePrefixBlob(
                data: blob, expectedMarker: marker + "!", expectedTokens: tokenCount, config: cfg))
            var badOff = 0
            let badRestore = try blob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int? in
                let ptr = try XCTUnwrap(raw.baseAddress)
                return FNGPUDFlash.restorePrefixKV(from: ptr, dataCount: blob.count, offset: &badOff,
                                                   slots: dstSlot, config: cfg, expectedMarker: marker, expectedTokens: tokenCount + 1)
            }
            XCTAssertNil(badRestore)
        }
    }

    func testCaptureHCHiddenKernelParityForDeferredAndDirect() throws {
        let hal = try makeHal()
        let pso = try hal.pipeline("qwg_dflash_capture_hc_hidden")
        let hidden = 256, hc = 4, rows = 4, totalSlots = 2, slot = 1, srcOff = 1, dstOff = 0, capRows = rows - srcOff
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 4 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let base = try act.allocate(QWTensorShape([rows * hc * hidden], .f32))
        let src = try act.allocate(QWTensorShape([rows * hidden], .f32))
        let inj = try act.allocate(QWTensorShape([rows * hc], .f32))
        let dstDirect = try act.allocate(QWTensorShape([rows * totalSlots * hidden], .f16))
        let dstDefer = try act.allocate(QWTensorShape([rows * totalSlots * hidden], .f16))

        let baseVals = (0..<(rows * hc * hidden)).map { Float(($0 % 37) - 18) * 0.05 }
        let srcVals = (0..<(rows * hidden)).map { Float(($0 % 29) - 14) * 0.04 }
        let injVals = (0..<(rows * hc)).map { Float(($0 % 7) + 1) * 0.1 }
        arena.write(base, from: baseVals)
        arena.write(src, from: srcVals)
        arena.write(inj, from: injVals)

        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        for (flagsVal, out) in [(UInt32(0), dstDirect), (UInt32(1), dstDefer)] {
            enc.setComputePipelineState(pso)
            base.bind(enc, index: 0)
            (flagsVal == 0 ? base : src).bind(enc, index: 1)
            (flagsVal == 0 ? base : inj).bind(enc, index: 2)
            out.bind(enc, index: 3)
            var sO = UInt32(srcOff), dO = UInt32(dstOff), sl = UInt32(slot)
            var hu4 = UInt32(hidden / 8), ts = UInt32(totalSlots), hcu = UInt32(hc), fl = flagsVal
            enc.setBytes(&sO, length: 4, index: 4); enc.setBytes(&dO, length: 4, index: 5)
            enc.setBytes(&sl, length: 4, index: 6); enc.setBytes(&hu4, length: 4, index: 7)
            enc.setBytes(&ts, length: 4, index: 8); enc.setBytes(&hcu, length: 4, index: 9)
            enc.setBytes(&fl, length: 4, index: 10)
            enc.dispatchThreadgroups(MTLSize(width: capRows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()

        func fwht32(_ vec: inout [Float]) {
            let invSqrt32: Float = 1.0 / sqrt(32.0)
            for blk in stride(from: 0, to: vec.count, by: 32) {
                for len in [1, 2, 4, 8, 16] {
                    var i = 0
                    while i < 32 {
                        for j in 0..<len {
                            let u = vec[blk + i + j], v = vec[blk + i + j + len]
                            vec[blk + i + j] = u + v; vec[blk + i + j + len] = u - v
                        }
                        i += len * 2
                    }
                }
                for j in 0..<32 { vec[blk + j] *= invSqrt32 }
            }
        }
        let ptrDirect = arena.cpuPointer(of: dstDirect).assumingMemoryBound(to: Float16.self)
        let ptrDefer = arena.cpuPointer(of: dstDefer).assumingMemoryBound(to: Float16.self)
        var maxErr: Float = 0
        for r in 0..<capRows {
            let sRow = srcOff + r
            var expDirect = [Float](repeating: 0, count: hidden)
            var expDefer = [Float](repeating: 0, count: hidden)
            for h in 0..<hidden {
                var sum0: Float = 0, sum1: Float = 0
                for c in 0..<hc {
                    let b = baseVals[(sRow * hc + c) * hidden + h]
                    sum0 += b
                    sum1 += b + srcVals[sRow * hidden + h] * injVals[sRow * hc + c]
                }
                expDirect[h] = sum0 / Float(hc)
                expDefer[h] = sum1 / Float(hc)
            }
            fwht32(&expDirect); fwht32(&expDefer)
            let dstBase = (dstOff + r) * totalSlots * hidden + slot * hidden
            for h in 0..<hidden {
                maxErr = max(maxErr, abs(Float(ptrDirect[dstBase + h]) - expDirect[h]))
                maxErr = max(maxErr, abs(Float(ptrDefer[dstBase + h]) - expDefer[h]))
            }
        }
        XCTAssertLessThan(maxErr, 2e-2)
    }

    func testEncodeHeadLogitsHQ4AndAQ8FullAndSlicedVocab() throws {
        let hal = try makeHal()
        let vocab = 256, k = 256, slicedVocab = 128
        let arena = try QWArena(hal: hal, weights: 4 << 20, activations: 4 << 20, kv: 0, gdnStates: 0)
        // Build HQ4_G32 (bits 4, group 32, 20 B/blk) & AQ8_G32 (bits 8, group 32, 36 B/blk).
        func allocWeight(bits: Int, blockBytes: Int, wordFill: UInt8) throws -> QWWeightView {
            let totalBytes = vocab * (k / 32) * blockBytes
            let t = try arena.weightsRegion().allocate(QWTensorShape([totalBytes / 2], .f16))
            var raw = Data(capacity: totalBytes)
            let scale = Float16(0.02).bitPattern
            for blk in 0..<(vocab * (k / 32)) {
                raw.append(UInt8(scale & 0xFF)); raw.append(UInt8(scale >> 8))
                raw.append(0); raw.append(0)
                raw.append(contentsOf: [UInt8](repeating: UInt8((blk % 7 + Int(wordFill)) & 0x7F), count: blockBytes - 4))
            }
            arena.copyIn(t, from: raw)
            return try XCTUnwrap(QWWeightView(
                blocks: t, n: vocab, k: k, bits: bits, groupSize: 32,
                isRotated: true, isIncumbentContainer: bits == 4, isSHVQ: false))
        }
        let hq4W = try allocWeight(bits: 4, blockBytes: 20, wordFill: 0x11)
        let aq8W = try allocWeight(bits: 8, blockBytes: 36, wordFill: 0x05)
        let act = arena.activationsRegion()
        for rows in [1, 5] {
            let x16 = try act.allocate(QWTensorShape([rows * k], .f16))
            let x32 = try act.allocate(QWTensorShape([rows * k], .f32))
            let xs = (0..<(rows * k)).map { Float(($0 % 23) - 11) * 0.05 }
            arena.write(x32, from: xs)
            arena.write(x16, from: xs.map(Float16.init))
            for (w, vCount) in [(hq4W, vocab), (aq8W, vocab), (aq8W, slicedVocab)] {
                let yOut = try act.allocate(QWTensorShape([rows * vCount], .f32))
                let yRef = try act.allocate(QWTensorShape([rows * vCount], .f32))
                guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
                try FNGPUDFlash.encodeHeadLogits(enc, hal: hal, headWeight: w, xRot16: x16, xNorm16: x16,
                                                 xRot32: x32, yLogits32: yOut, rows: rows, vocabRows: vCount)
                let sub = vCount < w.n
                    ? try XCTUnwrap(w.subview(blocks: w.blocks.slice(byteOffset: 0, count: (vCount * w.rowBytes) / 2, dtype: .f16), n: vCount, k: w.k))
                    : w
                try QWGemm.encodeHQ4F32V4(enc, hal: hal, weight: sub, x: x32, y: yRef, rows: rows)
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                let a = arena.cpuPointer(of: yOut).assumingMemoryBound(to: Float.self)
                let b = arena.cpuPointer(of: yRef).assumingMemoryBound(to: Float.self)
                var maxErr: Float = 0
                for i in 0..<(rows * vCount) { maxErr = max(maxErr, abs(a[i] - b[i])) }
                XCTAssertLessThan(maxErr, 0.25, "mismatch bits=\(w.bits) rows=\(rows) vocab=\(vCount)")
            }
        }
    }

    func testConsumePendingContextUpdateClearsStateAndHonorsSlot() throws {
        let hal = try makeHal()
        let cfg = QWDFlashConfig(
            hiddenSize: 256, numLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2,
            headDim: 64, intermediateSize: 512, vocabSize: 256,
            slidingWindow: 33, blockSize: 8, maskTokenId: 250, targetLayerIds: [0, 1])
        let slot = try makeSlot(hal: hal, config: cfg)

        var count = 3
        let missing = FNGPUDFlash.consumePendingContextUpdate(pos: 17, count: &count, slot: nil)
        XCTAssertNil(missing)
        XCTAssertEqual(count, 0)

        count = 5
        let active = try XCTUnwrap(FNGPUDFlash.consumePendingContextUpdate(pos: 42, count: &count, slot: slot))
        XCTAssertEqual(active.startPosition, 42)
        XCTAssertEqual(active.totalCount, 5)
        XCTAssertEqual(count, 0)
        XCTAssertNil(FNGPUDFlash.consumePendingContextUpdate(pos: 42, count: &count, slot: slot))
        XCTAssertEqual(FNGPUDFlash.maxFittingContext(activeCapacity: 0), 0)
        XCTAssertEqual(FNGPUDFlash.maxFittingContext(activeCapacity: 4096), 4095)
    }
}
