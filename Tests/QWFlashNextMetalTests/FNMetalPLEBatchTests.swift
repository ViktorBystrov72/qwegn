import Foundation
import XCTest
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
@testable import QWQuant
@testable import QWHyperPLE
@testable import QWEngine

final class FNMetalPLEBatchTests: XCTestCase {
    func testStridedKeyNormAndGateMatchPerSlot() throws {
        for batch in [1, 2, 4] {
            try check(batch: batch, hidden: 256)
            try check(batch: batch, hidden: 2560)
        }
    }

    private func check(batch: Int, hidden: Int) throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion(), hc = 4, width = hidden * hc, stride = width + hidden
        func f32(_ count: Int) throws -> QWTensorValue {
            try act.allocate(QWTensorShape([count], .f32))
        }
        let kv = try f32(batch * stride), query = try f32(batch * width), weight = try f32(width)
        let refKey = try f32(batch * width), outKey = try f32(batch * width)
        let refGate = try f32(batch * width), outGate = try f32(batch * width)
        let kvValues = (0..<(batch * stride)).map { Float(($0 % 127) - 63) / 64 }
        let queryValues = (0..<(batch * width)).map { Float(($0 % 97) - 48) / 64 }
        let weights = (0..<width).map { Float(($0 % 17) - 8) / 128 }
        arena.write(kv, from: kvValues)
        arena.write(query, from: queryValues)
        arena.write(weight, from: weights)
        func run(strided: Bool) throws -> (gpu: Double, encode: Double) {
            let started = DispatchTime.now().uptimeNanoseconds
            let pair = try XCTUnwrap(hal.begin(.concurrent))
            let enc = pair.1, key = strided ? outKey : refKey, gate = strided ? outGate : refGate
            if strided {
                try QWGemm.encodeFnHCNorm(enc, hal: hal, x: kv, weight: weight, xn: key,
                                          rows: batch, width: width, hidden: hidden, eps: 1e-5, xStride: stride)
            } else {
                for row in 0..<batch {
                    let input = kv.slice(byteOffset: row * stride * 4, count: width, dtype: .f32)
                    let output = key.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    try QWGemm.encodeFnHCNorm(enc, hal: hal, x: input, weight: weight, xn: output,
                                              rows: 1, width: width, hidden: hidden, eps: 1e-5)
                }
            }
            enc.memoryBarrier(scope: .buffers)
            if strided {
                let values = kv.slice(byteOffset: width * 4, count: (batch - 1) * stride + hidden, dtype: .f32)
                try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: key, queryNorm: query, vals: values, gv: gate,
                                           rows: batch, hidden: hidden, hc: hc, valStride: stride)
            } else {
                for row in 0..<batch {
                    let values = kv.slice(byteOffset: (row * stride + width) * 4, count: hidden, dtype: .f32)
                    let kn = key.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    let qn = query.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    let out = gate.slice(byteOffset: row * width * 4, count: width, dtype: .f32)
                    try QWGemm.encodeFnPLEGate(enc, hal: hal, keyNorm: kn, queryNorm: qn, vals: values, gv: out,
                                               rows: 1, hidden: hidden, hc: hc)
                }
            }
            let encodeMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            hal.end(pair.0, enc)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)
            return ((pair.0.gpuEndTime - pair.0.gpuStartTime) * 1000, encodeMs)
        }
        _ = try run(strided: false)
        _ = try run(strided: true)
        let keys = arena.read(refKey, as: Float.self, count: batch * width)
        let gates = arena.read(refGate, as: Float.self, count: batch * width)
        XCTAssertEqual(keys, arena.read(outKey, as: Float.self, count: batch * width))
        XCTAssertEqual(gates, arena.read(outGate, as: Float.self, count: batch * width))
        for row in 0..<batch {
            for group in 0..<hc {
                let offset = row * width + group * hidden
                let input = row * stride + group * hidden
                let sumSquares = (0..<hidden).reduce(0.0) { $0 + pow(Double(kvValues[input + $1]), 2) }
                let inverse = 1 / sqrt(sumSquares / Double(hidden) + 1e-5)
                var dot = 0.0
                for d in 0..<hidden {
                    let expected = Double(kvValues[input + d]) * inverse * (1 + Double(weights[group * hidden + d]))
                    XCTAssertEqual(Double(keys[offset + d]), expected, accuracy: 3e-5)
                    dot += Double(keys[offset + d]) * Double(queryValues[offset + d])
                }
                let scaled = dot / sqrt(Double(hidden))
                let transformed = (scaled > 0 ? 1.0 : (scaled < 0 ? -1.0 : 0.0)) * sqrt(max(abs(scaled), 1e-6))
                let sigmoid = 1 / (1 + exp(-transformed))
                for d in 0..<hidden {
                    let expected = sigmoid * Double(kvValues[row * stride + width + d])
                    XCTAssertEqual(Double(gates[offset + d]), expected, accuracy: 3e-5)
                }
            }
        }
        let dummyBlocks = try arena.weightsRegion().allocate(QWTensorShape([10], .f16))
        let dummy = try XCTUnwrap(QWWeightView(blocks: dummyBlocks, n: 1, k: 32, isRotated: true))
        let ple = FNGPUPLE(keyValueBank: dummy, keyBank: dummy, valueBank: dummy,
                           normKey: weight, normQuery: weight, normConv: weight, convW: weight,
                           embedDim: 1, kernel: 1, dilation: 1, eps: 1e-5, history: [])
        let moe = FNGPUMoE(gateAndSharedBF16: weight, gateUp: dummy, down: dummy,
                           sharedGateUp: dummy, sharedDown: dummy, sharedGateScalar: weight,
                           experts: 1, topK: 1, intermediate: 1, sharedIntermediate: 1)
        for strided in [false, true] {
            let scratch = try FNGPUScratch(arena: arena, T: batch, hidden: hidden, hc: hc, lowrank: 1,
                                           moe: moe, ple: nil, gdn: nil, attn: nil)
            arena.write(scratch.pleKV, from: kvValues)
            arena.write(scratch.pleQueryN, from: queryValues)
            let pair = try XCTUnwrap(hal.begin(.concurrent))
            try FNGPULayerPasses.pleBatchKeyGate(pair.1, hal: hal, p: ple, s: scratch,
                                                rows: batch, strided: strided)
            hal.end(pair.0, pair.1)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)
            XCTAssertEqual(keys, arena.read(scratch.pleKeyN, as: Float.self, count: batch * width))
            XCTAssertEqual(gates, arena.read(scratch.pleGV, as: Float.self, count: batch * width))
        }
        if ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1", batch == 4, hidden == 2560 {
            var baselineGPU: [Double] = [], candidateGPU: [Double] = []
            var baselineEncode: [Double] = [], candidateEncode: [Double] = []
            for sample in 0..<10 {
                let baseline: (gpu: Double, encode: Double), candidate: (gpu: Double, encode: Double)
                if sample.isMultiple(of: 2) {
                    baseline = try run(strided: false)
                    candidate = try run(strided: true)
                } else {
                    candidate = try run(strided: true)
                    baseline = try run(strided: false)
                }
                baselineGPU.append(baseline.gpu); candidateGPU.append(candidate.gpu)
                baselineEncode.append(baseline.encode); candidateEncode.append(candidate.encode)
            }
            print("Batch PLE GPU ms: per-slot \(baselineGPU), strided \(candidateGPU)")
            print("Batch PLE CPU encode ms: per-slot \(baselineEncode), strided \(candidateEncode)")
        }
    }

    func testVQ2MetalGatherMatchesCPUAndSpansSectorBoundariesBitExact() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        for entryCount in [60, 95, 4500] {
            var cbBits = [UInt16](repeating: 0, count: 512)
            for c in 1..<256 {
                let angle = Float(c) * 0.137
                let rad = 0.15 + 0.84 * Float(c % 17) / 16.0
                cbBits[2 * c] = Float16(rad * cos(angle)).bitPattern
                cbBits[2 * c + 1] = Float16(rad * sin(angle)).bitPattern
            }
            let headDim = 160, rowBytes = 90, rps = 45
            let header = try QWNgramFormat.makeHeader(version: 3, dtype: .vq2g32,
                                                      entryCount: UInt64(entryCount),
                                                      headDim: UInt16(headDim),
                                                      rowBytes: UInt16(rowBytes),
                                                      l1Hot: 0, codebook: cbBits)
            let sectors = (entryCount + rps - 1) / rps
            var fileBytes = [UInt8](repeating: 0, count: 4096 + sectors * 4096)
            fileBytes.replaceSubrange(0..<4096, with: header)
            for r in 0..<entryCount {
                let sec = r / rps, inSec = r % rps
                let rowOff = 4096 + sec * 4096 + inSec * rowBytes
                for g in 0..<5 {
                    let grpOff = rowOff + g * 18
                    let isZero = (r == 13 && g == 2) || r == 0
                    let s16 = isZero ? Float16(0.0) : Float16(0.0025 * Float(1 + ((r * 7 + g * 3) % 31)))
                    let sBits = s16.bitPattern
                    fileBytes[grpOff + 16] = UInt8(sBits & 0xFF)
                    fileBytes[grpOff + 17] = UInt8(sBits >> 8)
                    for p in 0..<16 {
                        fileBytes[grpOff + p] = isZero ? 0 : UInt8((r * 19 + g * 11 + p * 5 + 1) & 0xFF)
                    }
                }
            }
            let tmpURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwg_vq2_test_\(UUID().uuidString).bin")
            defer { try? FileManager.default.removeItem(at: tmpURL) }
            try Data(fileBytes).write(to: tmpURL)

            let preadRows = try FNNgramBinRows(url: tmpURL, mapped: false)
            let mmapRows = try FNNgramBinRows(url: tmpURL, mapped: true)
            for r in 0..<min(entryCount, 120) {
                XCTAssertEqual(preadRows.row(r), mmapRows.row(r))
            }

            let table = try XCTUnwrap(FNGPUPLETable(binRows: preadRows, device: hal.device))
            let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 16 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let numTargets = 12, rowsPerTarget = entryCount == 4500 ? 64 : 16
            let rowIds = try act.allocate(QWTensorShape([numTargets * rowsPerTarget], .i32))
            let dstOffsets = try act.allocate(QWTensorShape([numTargets * 2], .i32))
            let outs = try (0..<numTargets).map { _ in try act.allocate(QWTensorShape([rowsPerTarget * headDim], .f32)) }
            let pickedIds: [UInt32] = (0..<(numTargets * rowsPerTarget)).map { i in
                UInt32((i * 53 + 43) % entryCount)
            }
            let dstWordOffsets: [UInt64] = outs.map { out in
                FNGPUDecodeStack.pleTargetWordOffset(regionBase: out.region.base, actBase: act.region.base, tensorOffset: out.offset)
            }
            arena.write(rowIds, from: pickedIds)
            arena.write(dstOffsets, from: dstWordOffsets)

            let pair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnPLEVQ2Gather(pair.1, hal: hal, table: table, rowIds: rowIds,
                                            targetFloatOffsets: dstOffsets, outRegion: act.region,
                                            rowsPerTarget: rowsPerTarget, numTargets: numTargets)
            hal.end(pair.0, pair.1)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)

            for t in 0..<numTargets {
                let gpuVals = arena.read(outs[t], as: Float.self, count: rowsPerTarget * headDim)
                var cpuVals = [Float]()
                cpuVals.reserveCapacity(rowsPerTarget * headDim)
                for r in 0..<rowsPerTarget {
                    let id = Int(pickedIds[t * rowsPerTarget + r])
                    cpuVals.append(contentsOf: preadRows.row(id))
                }
                XCTAssertEqual(gpuVals, cpuVals, "0-ULP mismatch for entryCount=\(entryCount) target=\(t)")
            }

            let normWeight = try act.allocate(QWTensorShape([headDim], .f32))
            let normalized = try act.allocate(QWTensorShape([rowsPerTarget * headDim], .f32))
            arena.write(normWeight, from: [Float](repeating: 0, count: headDim))
            func gatherAndConsume(split: Bool) throws -> [UInt32] {
                let first = try XCTUnwrap(hal.begin(.concurrent))
                try QWGemm.encodeFnPLEVQ2Gather(first.1, hal: hal, table: table, rowIds: rowIds,
                                                targetFloatOffsets: dstOffsets, outRegion: act.region,
                                                rowsPerTarget: rowsPerTarget, numTargets: numTargets)
                var consumer = first
                if split {
                    hal.end(first.0, first.1)
                    consumer = try XCTUnwrap(hal.begin(.concurrent))
                } else {
                    first.1.memoryBarrier(scope: .buffers)
                }
                try QWGemm.encodeFnHCNorm(consumer.1, hal: hal, x: outs[0], weight: normWeight, xn: normalized,
                                          rows: rowsPerTarget, width: headDim, hidden: 32, eps: 1e-5)
                hal.end(consumer.0, consumer.1)
                consumer.0.waitUntilCompleted()
                XCTAssertNil(first.0.error)
                XCTAssertNil(consumer.0.error)
                return arena.read(normalized, as: Float.self, count: rowsPerTarget * headDim).map(\.bitPattern)
            }
            XCTAssertEqual(try gatherAndConsume(split: false), try gatherAndConsume(split: true),
                           "VQ2 gather consumers must stay bit-exact across command-buffer cuts")

            if ProcessInfo.processInfo.environment["QW_PERF_GATE"] == "1", entryCount == 4500 {
                func runCPUPread() -> Double {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    for t in 0..<numTargets {
                        var stage = [Float]()
                        stage.reserveCapacity(rowsPerTarget * headDim)
                        for r in 0..<rowsPerTarget {
                            stage.append(contentsOf: preadRows.row(Int(pickedIds[t * rowsPerTarget + r])))
                        }
                        arena.write(outs[t], from: stage)
                    }
                    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                }
                func runGPUGather() throws -> Double {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    arena.write(rowIds, from: pickedIds)
                    let cb = try XCTUnwrap(hal.begin(.concurrent))
                    try QWGemm.encodeFnPLEVQ2Gather(cb.1, hal: hal, table: table, rowIds: rowIds,
                                                    targetFloatOffsets: dstOffsets, outRegion: act.region,
                                                    rowsPerTarget: rowsPerTarget, numTargets: numTargets)
                    hal.end(cb.0, cb.1)
                    cb.0.waitUntilCompleted()
                    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                }
                _ = runCPUPread(); _ = try runGPUGather()
                var cpuMs: [Double] = [], gpuMs: [Double] = []
                for round in 0..<8 {
                    if round.isMultiple(of: 2) {
                        cpuMs.append(runCPUPread())
                        gpuMs.append(try runGPUGather())
                    } else {
                        gpuMs.append(try runGPUGather())
                        cpuMs.append(runCPUPread())
                    }
                }
                print("VQ2 PLE Gather Paired A/B (12x64=768 rows): CPU pread min=\(cpuMs.min() ?? 0)ms \(cpuMs) | GPU VQ2 min=\(gpuMs.min() ?? 0)ms \(gpuMs)")
            }
        }
    }

    func testStreamNgramsToggleControlsResidentPLEMapping() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_ple_toggle_\(UUID().uuidString)", isDirectory: true)
        let assetsDir = dir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        var cbBits = [UInt16](repeating: 0, count: 512)
        cbBits[2] = Float16(0.5).bitPattern
        cbBits[3] = Float16(-0.5).bitPattern
        let entryCount = 90, headDim = 160, rowBytes = 90, rps = 45
        let header = try QWNgramFormat.makeHeader(version: 3, dtype: .vq2g32,
                                                  entryCount: UInt64(entryCount),
                                                  headDim: UInt16(headDim),
                                                  rowBytes: UInt16(rowBytes),
                                                  l1Hot: 0, codebook: cbBits)
        let sectors = (entryCount + rps - 1) / rps
        var fileBytes = [UInt8](repeating: 0, count: 4096 + sectors * 4096)
        fileBytes.replaceSubrange(0..<4096, with: header)
        let ngramURL = assetsDir.appendingPathComponent("ngram.bin")
        let weightsURL = dir.appendingPathComponent("model.qwp")
        try Data(fileBytes).write(to: ngramURL)
        try Data([UInt8](repeating: 0, count: 64)).write(to: weightsURL)

        let resident = try FNGPUPLETable.resolveBacking(
            assetsURL: assetsDir, weightsURL: weightsURL, hasPLE: true, streamNgrams: false,
            hal: hal, weightsEstimate: 1 << 20, fixedScratchBytes: 1 << 20,
            minKVBytes: 1 << 20, availableBytes: 16 << 30)
        XCTAssertNotNil(resident.gpuTable)
        XCTAssertGreaterThan(resident.residentBytes, 0)

        let streamed = try FNGPUPLETable.resolveBacking(
            assetsURL: assetsDir, weightsURL: weightsURL, hasPLE: true, streamNgrams: true,
            hal: hal, weightsEstimate: 1 << 20, fixedScratchBytes: 1 << 20,
            minKVBytes: 1 << 20, availableBytes: 16 << 30)
        XCTAssertNil(streamed.gpuTable)
        XCTAssertEqual(streamed.residentBytes, 0)
    }

    func testVQ4MetalGatherMatchesCPUAndSpansSectorBoundariesBitExact() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        for entryCount in [100, 200, 4500] {
            var cbBits = [UInt16](repeating: 0, count: 1024)
            for c in 1..<256 {
                let base = Float(c) * 0.093
                let rad = 0.12 + 0.85 * Float(c % 19) / 18.0
                cbBits[4 * c + 0] = Float16(rad * cos(base)).bitPattern
                cbBits[4 * c + 1] = Float16(rad * sin(base)).bitPattern
                cbBits[4 * c + 2] = Float16(rad * cos(base * 1.7)).bitPattern
                cbBits[4 * c + 3] = Float16(rad * sin(base * 1.7)).bitPattern
            }
            let headDim = 160, rowBytes = 50, rps = 81
            let header = try QWNgramFormat.makeHeader(version: 3, dtype: .vq4g32,
                                                      entryCount: UInt64(entryCount),
                                                      headDim: UInt16(headDim),
                                                      rowBytes: UInt16(rowBytes),
                                                      l1Hot: 0, codebook: cbBits)
            let sectors = (entryCount + rps - 1) / rps
            var fileBytes = [UInt8](repeating: 0, count: 4096 + sectors * 4096)
            fileBytes.replaceSubrange(0..<4096, with: header)
            for r in 0..<entryCount {
                let sec = r / rps, inSec = r % rps
                let rowOff = 4096 + sec * 4096 + inSec * rowBytes
                for g in 0..<5 {
                    let grpOff = rowOff + g * 10
                    let isZero = (r == 17 && g == 3) || r == 0
                    let s16 = isZero ? Float16(0.0) : Float16(0.0025 * Float(1 + ((r * 11 + g * 5) % 31)))
                    let sBits = s16.bitPattern
                    fileBytes[grpOff + 8] = UInt8(sBits & 0xFF)
                    fileBytes[grpOff + 9] = UInt8(sBits >> 8)
                    for t in 0..<8 {
                        fileBytes[grpOff + t] = isZero ? 0 : UInt8((r * 23 + g * 13 + t * 7 + 1) & 0xFF)
                    }
                }
            }
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwg_vq4_test_\(UUID().uuidString)", isDirectory: true)
            let assetsDir = tmpDir.appendingPathComponent("assets", isDirectory: true)
            try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }
            let tmpURL = assetsDir.appendingPathComponent("ngram.bin")
            try Data(fileBytes).write(to: tmpURL)

            let preadRows = try FNNgramBinRows(url: tmpURL, mapped: false)
            let mmapRows = try FNNgramBinRows(url: tmpURL, mapped: true)
            for r in 0..<min(entryCount, 180) {
                XCTAssertEqual(preadRows.row(r), mmapRows.row(r))
            }

            let table = try XCTUnwrap(FNGPUPLETable(binRows: preadRows, device: hal.device))
            let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 16 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let numTargets = 12, rowsPerTarget = entryCount == 4500 ? 64 : 16
            let rowIds = try act.allocate(QWTensorShape([numTargets * rowsPerTarget], .i32))
            let dstOffsets = try act.allocate(QWTensorShape([numTargets * 2], .i32))
            let outs = try (0..<numTargets).map { _ in try act.allocate(QWTensorShape([rowsPerTarget * headDim], .f32)) }
            let pickedIds: [UInt32] = (0..<(numTargets * rowsPerTarget)).map { i in
                UInt32((i * 59 + 41) % entryCount)
            }
            let dstWordOffsets: [UInt64] = outs.map { out in
                FNGPUDecodeStack.pleTargetWordOffset(regionBase: out.region.base, actBase: act.region.base, tensorOffset: out.offset)
            }
            arena.write(rowIds, from: pickedIds)
            arena.write(dstOffsets, from: dstWordOffsets)

            let pair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnPLEVQ2Gather(pair.1, hal: hal, table: table, rowIds: rowIds,
                                            targetFloatOffsets: dstOffsets, outRegion: act.region,
                                            rowsPerTarget: rowsPerTarget, numTargets: numTargets)
            hal.end(pair.0, pair.1)
            pair.0.waitUntilCompleted()
            XCTAssertNil(pair.0.error)

            for t in 0..<numTargets {
                let gpuVals = arena.read(outs[t], as: Float.self, count: rowsPerTarget * headDim)
                var cpuVals = [Float]()
                cpuVals.reserveCapacity(rowsPerTarget * headDim)
                for r in 0..<rowsPerTarget {
                    let id = Int(pickedIds[t * rowsPerTarget + r])
                    cpuVals.append(contentsOf: preadRows.row(id))
                }
                XCTAssertEqual(gpuVals, cpuVals, "VQ4 0-ULP mismatch for entryCount=\(entryCount) target=\(t)")
            }

            if entryCount == 100 {
                let weightsURL = tmpDir.appendingPathComponent("model.qwp")
                try Data([UInt8](repeating: 0, count: 64)).write(to: weightsURL)
                let resident = try FNGPUPLETable.resolveBacking(
                    assetsURL: assetsDir, weightsURL: weightsURL, hasPLE: true, streamNgrams: false,
                    hal: hal, weightsEstimate: 1 << 20, fixedScratchBytes: 1 << 20,
                    minKVBytes: 1 << 20, availableBytes: 16 << 30)
                XCTAssertNotNil(resident.gpuTable)
                let streamed = try FNGPUPLETable.resolveBacking(
                    assetsURL: assetsDir, weightsURL: weightsURL, hasPLE: true, streamNgrams: true,
                    hal: hal, weightsEstimate: 1 << 20, fixedScratchBytes: 1 << 20,
                    minKVBytes: 1 << 20, availableBytes: 16 << 30)
                XCTAssertNil(streamed.gpuTable)
                XCTAssertEqual(streamed.residentBytes, 0)
            }
        }
    }

    func testFusedPLEGateNormAndResidualConvMatchesSeparateDispatches() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hc = 4, kernel = 4, dilation = 1, hl = (kernel - 1) * dilation, eps: Float = 1e-6
        for (rows, hidden) in [(9, 1280), (17, 2048), (64, 256), (97, 2560)] {
            let width = hc * hidden, kvRow = width + hidden
            let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 128 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let dummy = try XCTUnwrap(QWWeightView(blocks: try act.allocate(QWTensorShape([1024], .f16)),
                                                   n: 32, k: 32, isRotated: true))
            let normKey = try act.allocate(QWTensorShape([width], .f32))
            let normQuery = try act.allocate(QWTensorShape([width], .f32))
            let normConv = try act.allocate(QWTensorShape([width], .f32))
            let convW = try act.allocate(QWTensorShape([width * kernel], .f32))
            let histBuf = try act.allocate(QWTensorShape([hl * width], .f32))
            let streamBuf = try act.allocate(QWTensorShape([rows * width], .f32))
            let outRefBuf = try act.allocate(QWTensorShape([rows * width], .f32))
            let outFusedBuf = try act.allocate(QWTensorShape([rows * width], .f32))
            let scratch = try FNGPUScratch(arena: arena, T: rows, hidden: hidden, hc: hc, lowrank: 1,
                                           moe: FNGPUMoE(gateAndSharedBF16: normKey, gateUp: dummy, down: dummy,
                                                         sharedGateUp: dummy, sharedDown: dummy, sharedGateScalar: normKey,
                                                         experts: 1, topK: 1, intermediate: 1, sharedIntermediate: 1),
                                           ple: nil, gdn: nil, attn: nil)
            let nk = (0..<width).map { Float((($0 * 7) % 19) - 9) * 0.04 }
            let nq = (0..<width).map { Float((($0 * 11) % 17) - 8) * 0.05 }
            let nc = (0..<width).map { Float((($0 * 13) % 23) - 11) * 0.03 }
            let cw = (0..<(width * kernel)).map { Float((($0 * 5) % 13) - 6) * 0.08 }
            let hist = (0..<(hl * width)).map { Float((($0 * 3) % 29) - 14) * 0.05 }
            let stream = (0..<(rows * width)).map { Float((($0 * 17) % 31) - 15) * 0.06 }
            let kv = (0..<(rows * kvRow)).map { Float((($0 * 23) % 37) - 18) * 0.05 }
            arena.write(normKey, from: nk); arena.write(normQuery, from: nq)
            arena.write(normConv, from: nc); arena.write(convW, from: cw)
            arena.write(histBuf, from: hist); arena.write(streamBuf, from: stream)
            arena.write(scratch.pleKV, from: kv)
            // Also inject an all-zero edge row inside kv to verify sgn(g == 0) parity.
            arena.write(scratch.pleKV.slice(byteOffset: 0, count: kvRow, dtype: .f32),
                        from: [Float](repeating: 0, count: kvRow))
            let valSlice = scratch.pleKV.slice(byteOffset: width * 4, count: (rows - 1) * kvRow + hidden, dtype: .f32)

            let refPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnHCNorm(refPair.1, hal: hal, x: streamBuf, weight: normQuery, xn: scratch.pleQueryN,
                                      rows: rows, width: width, hidden: hidden, eps: eps)
            try QWGemm.encodeFnHCNorm(refPair.1, hal: hal, x: scratch.pleKV, weight: normKey, xn: scratch.pleKeyN,
                                      rows: rows, width: width, hidden: hidden, eps: eps, xStride: kvRow)
            refPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnPLEGate(refPair.1, hal: hal, keyNorm: scratch.pleKeyN, queryNorm: scratch.pleQueryN,
                                       vals: valSlice, gv: scratch.pleGV, rows: rows, hidden: hidden, hc: hc, valStride: kvRow)
            refPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnHCNorm(refPair.1, hal: hal, x: scratch.pleGV, weight: normConv, xn: scratch.pleGVN,
                                      rows: rows, width: width, hidden: hidden, eps: eps)
            refPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnPLEConv(refPair.1, hal: hal, gv: scratch.pleGV, gvNorm: scratch.pleGVN, convWeight: convW,
                                       out: scratch.pleOut, rows: rows, width: width, kernel: kernel, dilation: dilation, hist: histBuf)
            refPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeAddF32(refPair.1, hal: hal, a: streamBuf, b: scratch.pleOut, out: outRefBuf, count: rows * width)
            hal.end(refPair.0, refPair.1)
            refPair.0.waitUntilCompleted()
            XCTAssertNil(refPair.0.error)
            let refGVN = arena.read(scratch.pleGVN, as: Float.self, count: rows * width)
            let refOut = arena.read(outRefBuf, as: Float.self, count: rows * width)

            let fusedPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnPLEFusedGateNorm(fusedPair.1, hal: hal, kv: scratch.pleKV, queryNorm: scratch.pleQueryN,
                                                normKey: normKey, normConv: normConv, gv: scratch.pleGV, gvNorm: scratch.pleGVN,
                                                rows: rows, hidden: hidden, hc: hc, kvStride: kvRow, eps: eps)
            fusedPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnPLEConv(fusedPair.1, hal: hal, gv: scratch.pleGV, gvNorm: scratch.pleGVN, convWeight: convW,
                                       out: outFusedBuf, rows: rows, width: width, kernel: kernel, dilation: dilation,
                                       hist: histBuf, residual: streamBuf)
            hal.end(fusedPair.0, fusedPair.1)
            fusedPair.0.waitUntilCompleted()
            XCTAssertNil(fusedPair.0.error)
            let fusedGVN = arena.read(scratch.pleGVN, as: Float.self, count: rows * width)
            let fusedOut = arena.read(outFusedBuf, as: Float.self, count: rows * width)

            XCTAssertEqual(fusedGVN.map(\.bitPattern), refGVN.map(\.bitPattern),
                           "fused gvNorm must match unfused dispatches bit-exactly (0 ULP) for rows=\(rows) H=\(hidden)")
            XCTAssertEqual(fusedOut.map(\.bitPattern), refOut.map(\.bitPattern),
                           "fused PLE out must match unfused dispatches bit-exactly (0 ULP) for rows=\(rows) H=\(hidden)")

            // Also verify in-place residual aliasing (out === residual === streamBuf).
            let aliasPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnPLEConv(aliasPair.1, hal: hal, gv: scratch.pleGV, gvNorm: scratch.pleGVN, convWeight: convW,
                                       out: streamBuf, rows: rows, width: width, kernel: kernel, dilation: dilation,
                                       hist: histBuf, residual: streamBuf)
            hal.end(aliasPair.0, aliasPair.1)
            aliasPair.0.waitUntilCompleted()
            XCTAssertNil(aliasPair.0.error)
            let aliasOut = arena.read(streamBuf, as: Float.self, count: rows * width)
            XCTAssertEqual(aliasOut, fusedOut)
        }
    }

    func testFusedHCInjectNormMatchesSeparateDispatchesBitExact() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let hc = 4, eps: Float = 1e-6
        for (rows, hidden) in [(9, 1280), (17, 2048), (65, 2560)] {
            let width = hc * hidden
            let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 64 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let base = try act.allocate(QWTensorShape([rows * width], .f32))
            let src = try act.allocate(QWTensorShape([rows * hidden], .f32))
            let inj = try act.allocate(QWTensorShape([rows * hc], .f32))
            let w = try act.allocate(QWTensorShape([width], .f32))
            let refOut = try act.allocate(QWTensorShape([rows * width], .f32))
            let refXN = try act.allocate(QWTensorShape([rows * width], .f32))
            let fusedOut = try act.allocate(QWTensorShape([rows * width], .f32))
            let fusedXN = try act.allocate(QWTensorShape([rows * width], .f32))
            arena.write(base, from: (0..<(rows * width)).map { Float((($0 * 17) % 31) - 15) * 0.06 })
            arena.write(src, from: (0..<(rows * hidden)).map { Float((($0 * 13) % 29) - 14) * 0.05 })
            arena.write(inj, from: (0..<(rows * hc)).map { Float((($0 * 7) % 19) - 9) * 0.1 })
            arena.write(w, from: (0..<width).map { Float((($0 * 11) % 23) - 11) * 0.04 })

            let refPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnInject(refPair.1, hal: hal, base: base, src: src, inj: inj, out: refOut,
                                      rows: rows, hidden: hidden, hc: hc)
            refPair.1.memoryBarrier(scope: .buffers)
            try QWGemm.encodeFnHCNorm(refPair.1, hal: hal, x: refOut, weight: w, xn: refXN,
                                      rows: rows, width: width, hidden: hidden, eps: eps)
            hal.end(refPair.0, refPair.1)
            refPair.0.waitUntilCompleted()
            XCTAssertNil(refPair.0.error)

            let fusedPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnHCInjectNorm(fusedPair.1, hal: hal, base: base, src: src, inj: inj,
                                            xOut: fusedOut, weight: w, xn: fusedXN,
                                            rows: rows, width: width, hidden: hidden, eps: eps)
            hal.end(fusedPair.0, fusedPair.1)
            fusedPair.0.waitUntilCompleted()
            XCTAssertNil(fusedPair.0.error)

            let refOutVals = arena.read(refOut, as: Float.self, count: rows * width)
            let refXNVals = arena.read(refXN, as: Float.self, count: rows * width)
            let fusedOutVals = arena.read(fusedOut, as: Float.self, count: rows * width)
            let fusedXNVals = arena.read(fusedXN, as: Float.self, count: rows * width)
            XCTAssertEqual(fusedOutVals.map(\.bitPattern), refOutVals.map(\.bitPattern))
            XCTAssertEqual(fusedXNVals.map(\.bitPattern), refXNVals.map(\.bitPattern))

            let aliasPair = try XCTUnwrap(hal.begin(.concurrent))
            try QWGemm.encodeFnHCInjectNorm(aliasPair.1, hal: hal, base: base, src: src, inj: inj,
                                            xOut: base, weight: w, xn: fusedXN,
                                            rows: rows, width: width, hidden: hidden, eps: eps)
            hal.end(aliasPair.0, aliasPair.1)
            aliasPair.0.waitUntilCompleted()
            XCTAssertNil(aliasPair.0.error)
            XCTAssertEqual(arena.read(base, as: Float.self, count: rows * width).map(\.bitPattern), refOutVals.map(\.bitPattern))
            XCTAssertEqual(arena.read(fusedXN, as: Float.self, count: rows * width).map(\.bitPattern), refXNVals.map(\.bitPattern))
        }
    }

    func testPLETargetWordOffsetExceeding16GiBDoesNotOverflowUInt32() throws {
        let wordOffset = FNGPUDecodeStack.pleTargetWordOffset(
            regionBase: 2_684_354_560,
            actBase: 2_684_354_560,
            tensorOffset: 20 * (1 << 30)
        )
        XCTAssertEqual(wordOffset, 5_368_709_120)
        XCTAssertGreaterThan(wordOffset, UInt64(UInt32.max))
    }
}

private extension QWNgramFormat {
    static func makeHeader(
        version: UInt16,
        dtype: QWNgramDType,
        entryCount: UInt64,
        headDim: UInt16,
        rowBytes: UInt16,
        l1Hot: UInt32,
        note: String = "",
        codebook: [UInt16] = []
    ) throws -> [UInt8] {
        try checkVersionPairing(version: version, dtype: dtype)
        if dtype == .vq2g32 {
            _ = try validateVQ2Codebook(codebook)
        } else if dtype == .vq4g32 {
            _ = try validateVQ4Codebook(codebook)
        } else if !codebook.isEmpty {
            throw QWNgramFormatError.badHeader("codebook is only valid for v3 vq2g32 or vq4g32")
        }
        guard rowBytes == (try dtype.rowBytes(headDim: Int(headDim))) else {
            throw QWNgramFormatError.badRowSize(Int(rowBytes))
        }
        guard entryCount > 0 else { throw QWNgramFormatError.badHeader("entryCount is zero") }
        guard l1Hot <= entryCount else {
            throw QWNgramFormatError.badHeader("l1Hot \(l1Hot) exceeds entryCount \(entryCount)")
        }
        let noteBytes = Array(note.utf8)
        guard noteBytes.isEmpty || noteBytes.count == self.noteBytes else {
            throw QWNgramFormatError.badHeader("note must be \(self.noteBytes) bytes or empty")
        }
        var h = [UInt8](repeating: 0, count: headerBytes)
        func put16(_ o: Int, _ v: UInt16) { h[o] = UInt8(v & 0xFF); h[o + 1] = UInt8(v >> 8) }
        func put32(_ o: Int, _ v: UInt32) {
            for i in 0..<4 { h[o + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put64(_ o: Int, _ v: UInt64) {
            for i in 0..<8 { h[o + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        put64(0, magic)
        put16(8, version); put16(10, 0); put32(12, dtype.rawValue)
        put64(16, entryCount); put16(24, headDim); put16(26, rowBytes)
        put32(28, l1Hot); put64(32, UInt64(headerBytes))
        h[44] = UInt8(self.noteBytes)
        for (i, b) in noteBytes.enumerated() { h[45 + i] = b }
        if dtype == .vq2g32 || dtype == .vq4g32 {
            for (i, u) in codebook.enumerated() { put16(vq2CodebookOffset + i * 2, u) }
        }
        return h
    }
}
