// FNTelemetryHonestyTests.swift — Tests for Flash-Next weight and PLE residency telemetry.

import Metal
import XCTest
import QWAlloc
import QWAttn
@testable import QWCore
@testable import QWEngine
import QWHAL
import QWHyperPLE
import QWQuant
import QWRope

final class FNTelemetryHonestyTests: XCTestCase {
    func testStandaloneUniqueBufferBytesDeduplicatesAndExcludesArenaBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable")
        }
        let arenaBuf0 = try XCTUnwrap(device.makeBuffer(length: 4096, options: .storageModeShared))
        let arenaBuf1 = try XCTUnwrap(device.makeBuffer(length: 8192, options: .storageModeShared))
        let zeroCopySlab = try XCTUnwrap(device.makeBuffer(length: 16384, options: .storageModeShared))
        let pleBuffer = try XCTUnwrap(device.makeBuffer(length: 32768, options: .storageModeShared))

        let total = QWHyperStreamRuntime.standaloneUniqueBufferBytes(
            arenaBuffers: [arenaBuf0, arenaBuf1],
            candidateBuffers: [arenaBuf0, arenaBuf1, zeroCopySlab, zeroCopySlab, pleBuffer]
        )
        XCTAssertEqual(total, zeroCopySlab.length + pleBuffer.length,
                       "arena buffers must be excluded and shared non-arena buffers counted once")
    }

    func testStandaloneWeightBytesCountsZeroCopySlabsPLEAndDeduplicatesSharedHead() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable")
        }
        let hal = try QWHal(geometry: .qwen38_27b)
        let arenaBuf = try XCTUnwrap(device.makeBuffer(length: 4096, options: .storageModeShared))
        let layerSlab = try XCTUnwrap(device.makeBuffer(length: 4096, options: .storageModeShared))
        let headBuf = try XCTUnwrap(device.makeBuffer(length: 8192, options: .storageModeShared))
        let embedBuf = try XCTUnwrap(device.makeBuffer(length: 4096, options: .storageModeShared))
        let pleBuf = try XCTUnwrap(device.makeBuffer(length: 16384, options: .storageModeShared))

        func tensor(_ buf: any MTLBuffer, count: Int, dtype: QWDType = .f32) -> QWTensorValue {
            QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                          offset: 0, shape: QWTensorShape([count], dtype))
        }
        func hq4Weight(_ buf: any MTLBuffer, n: Int = 4, k: Int = 32) throws -> QWWeightView {
            let bytes = QWWeightView(blocks: tensor(buf, count: 1024, dtype: .f16),
                                     n: n, k: k, bits: 4, groupSize: 32, isRotated: true)
            return try XCTUnwrap(bytes)
        }

        let arenaWeight = try hq4Weight(arenaBuf)
        let slabWeight = try hq4Weight(layerSlab)
        let arenaTensor = tensor(arenaBuf, count: 16)
        let hc = FNGPUHC(norm: arenaTensor, injectBF16: arenaTensor,
                         down: arenaWeight, up: arenaWeight, hc: 1, hidden: 32, lowrank: 32, eps: 1e-6)
        let attn = FNGPUAttnMixer(
            bank: slabWeight, oProj: slabWeight,
            qNorm: arenaTensor, kNorm: arenaTensor, iqNorm: arenaTensor, ikNorm: arenaTensor,
            heads: 1, kvHeads: 1, headDim: 32, indexHeads: 1, indexKVHeads: 1, indexHeadDim: 32,
            rotaryDim: 32, compress: 1, blockTopK: 1, bankRows: 4,
            rope: QWRopeCPU(rotaryDim: 32, theta: 10000, yarnFactor: nil, originalMaxPosition: 4096))
        let moe = FNGPUMoE(gateAndSharedBF16: arenaTensor, gateUp: slabWeight, down: slabWeight,
                           sharedGateUp: arenaWeight, sharedDown: arenaWeight, sharedGateScalar: arenaTensor,
                           experts: 1, topK: 1, intermediate: 32, sharedIntermediate: 32)
        let ple = FNGPUPLE(keyValueBank: slabWeight, keyBank: slabWeight, valueBank: slabWeight,
                           normKey: arenaTensor, normQuery: arenaTensor, normConv: arenaTensor, convW: arenaTensor,
                           embedDim: 32, kernel: 4, dilation: 1, eps: 1e-6, history: [])
        let layer = FNGPULayerWeights(index: 0, mixer: .attention(attn), attnHC: hc, mlpHC: hc, moe: moe, ple: ple)

        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 1 << 20, kv: 1 << 20, gdnStates: 0)
        let head = try FNGPUHead(arena: arena, hal: hal, head: try hq4Weight(headBuf),
                                 mixerDown: arenaWeight, mixerUp: arenaWeight,
                                 normWeight: arenaTensor, eps: 1e-6, hidden: 32, hc: 1, lowrank: 32, maxRows: 1)
        let embed = FNGPUEmbedTable(storage: .bf16(tensor(embedBuf, count: 32, dtype: .f16)),
                                    vocab: 1, hidden: 32, hc: 1, containerNote: "test")
        let pleTable = FNGPUPLETable(buffer: pleBuf, headDim: 32, entryCount: 4,
                                     dataOffset: 0, rowsPerSector: 1, rowBytes: 32, dtype: .vq2g32)

        let total = QWHyperStreamRuntime.standaloneWeightBytes(
            arenaBuffers: [arenaBuf] + arena.allBuffers,
            layers: [layer], head: head, embedTable: embed, gpuPLETable: pleTable, mtp: nil
        )
        XCTAssertEqual(total, layerSlab.length + headBuf.length + embedBuf.length + pleBuf.length)
    }
}
