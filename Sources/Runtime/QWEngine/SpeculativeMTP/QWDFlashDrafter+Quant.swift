// QWDFlashDrafter+Quant.swift — Multi-format HQ / SHVQ / AQ weight loader,
// dynamic UMA capacity sizing, and pre-warmed GEMV/GEMM dispatch for DFlash/DSpine.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWHAL
import QWLoad
import QWQuant

struct DrafterQuantKey: Hashable, Sendable {
    let bits: Int
    let group: Int
    let isSHVQ: Bool
}

struct DrafterQuantPipelines {
    let decodeT4: any MTLComputePipelineState
    let siluPair: any MTLComputePipelineState
    let gemmBatched: any MTLComputePipelineState
}

extension QWDFlashDrafter {
    static func computeWeightCapacity(pack: QWQuant.Pack, isDSpine: Bool) -> Int {
        let defaultCap = isDSpine ? 1_920_000_000 : 1_480_000_000
        let allIncumbent = pack.tensors.values.allSatisfy { !$0.quantized || $0.isIncumbentHQ4 }
        if allIncumbent {
            return defaultCap
        }
        var total = 0
        for e in pack.tensors.values {
            let bytes: Int
            if let resolved = PlaneLayout.resolve(e), let b = resolved.byteCount(e) {
                bytes = b
            } else {
                bytes = (e.hq8PayloadBytes ?? 0) + (e.hq8ScaleBytes ?? 0)
            }
            total += ((bytes + 16383) / 16384) * 16384
        }
        return max(64 * 1024 * 1024, total + 64 * 1024 * 1024)
    }

    static func loadQuantWeight(
        pack: QWQuant.Pack,
        io: PlaneIO,
        wSub: QWSubAllocator,
        hal: QWHal,
        name: String,
        n: Int,
        k: Int
    ) throws -> QWWeightView {
        let s = try QWCheckpointTensorLoader.span(pack, name)
        guard s.rows == n, s.cols == k else {
            throw QWEngineDriverError.weightMissing(name, "DFlash quant shape [\(s.rows), \(s.cols)] != [\(n), \(k)]")
        }
        let t = try wSub.allocate(QWTensorShape([s.bytes / 2], .f16))
        let dst = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
        try io.load(name, layout: .hq4, into: dst)
        let rot = hal.weightBasis.rotatesActivations
        guard let v = QWWeightView(
            blocks: t, n: n, k: k, bits: s.bits, groupSize: s.group,
            isRotated: rot, isIncumbentContainer: rot && s.isIncumbent, isSHVQ: rot && s.isSHVQ
        ) else {
            throw QWEngineDriverError.weightMissing(name, "invalid DFlash quant view")
        }
        return v
    }

    static func concatQuantWeights(
        pack: QWQuant.Pack,
        io: PlaneIO,
        wSub: QWSubAllocator,
        hal: QWHal,
        specs: [(name: String, n: Int)],
        k: Int
    ) throws -> (QWWeightView, [QWWeightView]) {
        let spans = try specs.map { spec -> (name: String, n: Int, span: QWCheckpointTensorLoader.PackSpan) in
            let s = try QWCheckpointTensorLoader.span(pack, spec.name)
            guard s.rows == spec.n, s.cols == k else {
                throw QWEngineDriverError.weightMissing(spec.name, "DFlash concat shape mismatch")
            }
            return (spec.name, spec.n, s)
        }
        guard let first = spans.first?.span,
              spans.allSatisfy({
                  $0.span.bits == first.bits && $0.span.group == first.group
                      && $0.span.isIncumbent == first.isIncumbent && $0.span.isSHVQ == first.isSHVQ
              }) else {
            throw QWEngineDriverError.weightMissing(specs.first?.name ?? "concat", "mismatched DFlash concat formats")
        }
        let totalN = specs.reduce(0) { $0 + $1.n }
        let totalBytes = spans.reduce(0) { $0 + $1.span.bytes }
        let blocks = try wSub.allocate(QWTensorShape([totalBytes / 2], .f16))
        let basePtr = blocks.region.buffer.contents().advanced(by: blocks.region.base + blocks.offset)
        let rot = hal.weightBasis.rotatesActivations
        guard let big = QWWeightView(
            blocks: blocks, n: totalN, k: k, bits: first.bits, groupSize: first.group,
            isRotated: rot, isIncumbentContainer: rot && first.isIncumbent, isSHVQ: rot && first.isSHVQ
        ) else {
            throw QWEngineDriverError.weightMissing(specs.first?.name ?? "concat", "invalid DFlash concat")
        }
        var byteOff = 0
        var subViews: [QWWeightView] = []
        for item in spans {
            try io.load(item.name, layout: .hq4, into: basePtr.advanced(by: byteOff))
            let subT = QWTensorValue(region: blocks.region, offset: blocks.offset + byteOff,
                                     shape: QWTensorShape([item.span.bytes / 2], .f16))
            guard let sv = big.subview(blocks: subT, n: item.n, k: k) else {
                throw QWEngineDriverError.weightMissing(item.name, "invalid DFlash subview")
            }
            subViews.append(sv)
            byteOff += item.span.bytes
        }
        return (big, subViews)
    }

    static func loadQuantPipelines(
        hal: QWHal,
        fc: QWWeightView,
        layers: [LayerWeights]
    ) throws -> [DrafterQuantKey: DrafterQuantPipelines] {
        var views: [QWWeightView] = [fc]
        for lw in layers {
            views.append(contentsOf: [lw.qkvProj, lw.kvProj, lw.oProj, lw.gateUpProj, lw.downProj])
        }
        var map: [DrafterQuantKey: DrafterQuantPipelines] = [:]
        for w in views where !QWMTPBatchedMLP.usesQ4GMKernel(w) {
            let key = DrafterQuantKey(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ)
            if map[key] != nil { continue }
            guard let decName = QWGemmAQ.decodeKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ),
                  let siluName = QWGemmAQ.siluPairKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ),
                  let gemmName = QWGemmAQ.gemmKernelName(bits: w.bits, group: w.groupSize, bm128: false, isSHVQ: w.isSHVQ) else {
                throw QWEngineDriverError.state("QWDFlashDrafter: unsupported quant format \(w.formatName)")
            }
            map[key] = DrafterQuantPipelines(
                decodeT4: try hal.pipeline(decName),
                siluPair: try hal.pipeline(siluName),
                gemmBatched: try hal.pipeline(gemmName)
            )
        }
        return map
    }

    func encodeSmallTileNAX(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        weight: QWWeightView,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) {
        let n = weight.n, k = weight.k
        enc.setComputePipelineState(pso)
        weight.blocks.bind(enc, index: 0)
        var nU = UInt32(n), kU = UInt32(k)
        enc.setBytes(&nU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6)
        var xs = UInt32(k), ys = UInt32(n)
        enc.setBytes(&xs, length: 4, index: 9)
        enc.setBytes(&ys, length: 4, index: 10)
        let chunk = QWGemm.naxRowChunk(rows: rows, k: k)
        var r0 = 0
        while r0 < rows {
            let r = min(chunk, rows - r0)
            let bm = isMetal4TensorCapable ? (r <= 16 ? 16 : (r <= 32 ? 32 : 64)) : 64
            let bn = (isMetal4TensorCapable && r <= 16) ? 128 : 64
            let modeBits = isMetal4TensorCapable ? (r <= 16 ? 0x2000_0000 : (r <= 32 ? 0x4000_0000 : 0)) : 0
            let xSlice = r0 == 0 ? input : input.slice(byteOffset: r0 * k * 2, count: r * k)
            let ySlice = r0 == 0 ? output : output.slice(byteOffset: r0 * n * 2, count: r * n)
            xSlice.bind(enc, index: 1)
            ySlice.bind(enc, index: 2)
            ySlice.bind(enc, index: 3)
            var mU = UInt32(r | modeBits)
            enc.setBytes(&mU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (n + bn - 1) / bn, height: (r + bm - 1) / bm, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            r0 += r
        }
    }

    func encodeSiluMulF16(
        _ enc: any MTLComputeCommandEncoder,
        gu: QWTensorValue,
        out: QWTensorValue,
        rows: Int,
        i: Int
    ) {
        enc.setComputePipelineState(psoSiluMulF16)
        gu.bind(enc, index: 0)
        out.bind(enc, index: 1)
        var mU = UInt32(rows), iU = UInt32(i)
        enc.setBytes(&mU, length: 4, index: 2)
        enc.setBytes(&iU, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: QWEngineDriver.siluThreadgroups(i: i), height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func canUseNAXGateUpScratch(rows: Int, h: Int, i: Int) -> Bool {
        (2 * i) % 64 == 0 && h % 64 == 0 && draftLogits.bytes >= rows * 2 * i * 2
    }

    func encodeAQMatmul(
        _ enc: any MTLComputeCommandEncoder,
        weight: QWWeightView,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) {
        let key = DrafterQuantKey(bits: weight.bits, group: weight.groupSize, isSHVQ: weight.isSHVQ)
        guard let qps = quantPSOs[key] else { return }
        let n = weight.n, k = weight.k
        let useNaxSmall = isMetal4TensorCapable && rows >= 8 && n % 64 == 0 && k % 64 == 0
        if rows == 1 {
            enc.setComputePipelineState(qps.decodeT4)
            weight.blocks.bind(enc, index: 0)
            input.bind(enc, index: 1)
            output.bind(enc, index: 3)
            var nU = UInt32(n) | 0x4000_0000, kU = UInt32(k)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            output.bind(enc, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else if !useNaxSmall && (2...16).contains(rows) {
            enc.setComputePipelineState(qps.decodeT4)
            weight.blocks.bind(enc, index: 0)
            var kU = UInt32(k)
            enc.setBytes(&kU, length: 4, index: 5)
            var r0 = 0
            while r0 < rows {
                let rem = rows - r0
                let chunkRows = (rem > 4 && rem - 4 == 1) ? 3 : min(4, rem)
                let xChunk = input.slice(byteOffset: r0 * k * 2, count: chunkRows * k)
                let yChunk = output.slice(byteOffset: r0 * n * 2, count: chunkRows * n)
                xChunk.bind(enc, index: 1)
                yChunk.bind(enc, index: 3)
                yChunk.bind(enc, index: 8)
                var nU = UInt32(n) | 0x4000_0000 | (UInt32(chunkRows) << 24)
                enc.setBytes(&nU, length: 4, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: (n + 15) / 16, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                r0 += chunkRows
            }
        } else {
            encodeSmallTileNAX(enc, pso: qps.gemmBatched, weight: weight, input: input, output: output, rows: rows)
        }
    }

    func encodeAQGateUpMatmul(
        _ enc: any MTLComputeCommandEncoder,
        weight: QWWeightView,
        input: QWTensorValue,
        output: QWTensorValue,
        rows: Int
    ) {
        let key = DrafterQuantKey(bits: weight.bits, group: weight.groupSize, isSHVQ: weight.isSHVQ)
        guard let qps = quantPSOs[key] else { return }
        let H = config.hiddenSize
        let I = config.intermediateSize
        if isMetal4TensorCapable && rows >= 8 && canUseNAXGateUpScratch(rows: rows, h: H, i: I) {
            let guScratch = QWTensorValue(region: draftLogits.region, offset: draftLogits.offset,
                                          shape: QWTensorShape([rows * 2 * I], .f16))
            encodeAQMatmul(enc, weight: weight, input: input, output: guScratch, rows: rows)
            encodeSiluMulF16(enc, gu: guScratch, out: output, rows: rows, i: I)
            return
        }
        let fuseFwht = !weight.isIncumbentContainer && (I % 32 == 0)
        enc.setComputePipelineState(qps.siluPair)
        weight.blocks.bind(enc, index: 0)
        var kU = UInt32(H)
        enc.setBytes(&kU, length: 4, index: 5)
        let tgWidth = (I + 31) / 32
        var r0 = 0
        while r0 < rows {
            let rem = rows - r0
            let chunkRows = (rem > 4 && rem - 4 == 1) ? 3 : min(4, rem)
            let xChunk = input.slice(byteOffset: r0 * H * 2, count: chunkRows * H)
            let yChunk = output.slice(byteOffset: r0 * I * 2, count: chunkRows * I)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var nU = UInt32(2 * I) | 0x4000_0000 | (UInt32(chunkRows) << 24) | (fuseFwht ? 0x8000_0000 : 0)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: tgWidth, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            r0 += chunkRows
        }
    }
}
