// QWVisionMetal+Ops.swift — Weight packing, FP16 conversions, and Metal
// dispatch encoders for QWVisionMetalBackend.
import Accelerate
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

extension QWVisionMetalBackend {
    static func writeF16(_ src: [Float], into dst: QWTensorValue) {
        let count = src.count
        guard count > 0 else { return }
        precondition(dst.shape.elementCount >= count)
        let dstPtr = dst.region.pointer.advanced(by: dst.offset)
        src.withUnsafeBufferPointer { sb in
            guard let srcBase = sb.baseAddress else { return }
            var vSrc = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcBase),
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )
            var vDst = vImage_Buffer(
                data: dstPtr,
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<UInt16>.stride
            )
            if vImageConvert_PlanarFtoPlanar16F(&vSrc, &vDst, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                let out16 = dstPtr.assumingMemoryBound(to: Float16.self)
                for i in 0..<count { out16[i] = Float16(srcBase[i]) }
            }
        }
    }

    static func readF16(from src: QWTensorValue, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        precondition(src.shape.elementCount >= count)
        var out = [Float](repeating: 0, count: count)
        let srcPtr = src.region.pointer.advanced(by: src.offset)
        out.withUnsafeMutableBufferPointer { ob in
            guard let outBase = ob.baseAddress else { return }
            var vSrc = vImage_Buffer(
                data: srcPtr,
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<UInt16>.stride
            )
            var vDst = vImage_Buffer(
                data: outBase,
                height: 1,
                width: vImagePixelCount(count),
                rowBytes: count * MemoryLayout<Float>.stride
            )
            if vImageConvert_Planar16FtoPlanarF(&vSrc, &vDst, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                let in16 = srcPtr.assumingMemoryBound(to: Float16.self)
                for i in 0..<count { outBase[i] = Float(in16[i]) }
            }
        }
        return out
    }

    static func writePODArray<T>(_ src: [T], into dst: QWTensorValue) {
        guard !src.isEmpty else { return }
        let bytes = src.count * MemoryLayout<T>.stride
        precondition(dst.bytes >= bytes)
        let dstPtr = dst.region.pointer.advanced(by: dst.offset)
        src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            dstPtr.copyMemory(from: base, byteCount: bytes)
        }
    }

    private static func weightBufferBytes(config cfg: QWVisionConfig, depth: Int, fc2Out: Int) -> Int {
        let d = cfg.dim, mlp = cfg.mlpInner, qkvD = cfg.qkvDim
        let merged = cfg.merged, pc = cfg.patchChannels
        let topBytes =
            qwAlignUp(cfg.numPosEmbeds * d * 2) +
            qwAlignUp(d * pc * 2) +
            qwAlignUp(d * 2) +
            qwAlignUp(d * 2) + qwAlignUp(d * 2) +
            qwAlignUp(merged * merged * 2) + qwAlignUp(merged * 2) +
            qwAlignUp(fc2Out * merged * 2) + qwAlignUp(fc2Out * 2) +
            qwAlignUp(cfg.invFreqCount * 4)
        let perBlockBytes =
            qwAlignUp(d * 2) * 4 +
            qwAlignUp(qkvD * d * 2) + qwAlignUp(qkvD * 2) +
            qwAlignUp(d * d * 2) + qwAlignUp(d * 2) +
            qwAlignUp(mlp * d * 2) + qwAlignUp(mlp * 2) +
            qwAlignUp(d * mlp * 2) + qwAlignUp(d * 2)
        return topBytes + depth * perBlockBytes
    }

    private static func packTensors(
        config cfg: QWVisionConfig,
        depth: Int,
        fc2Out: Int,
        device: any MTLDevice,
        posEmbed: () throws -> [Float],
        patchW: () throws -> [Float],
        patchB: () throws -> [Float],
        blockTensor: (Int, String, [Int]) throws -> [Float],
        mergerG: () throws -> [Float],
        mergerB: () throws -> [Float],
        fc1W: () throws -> [Float],
        fc1B: () throws -> [Float],
        fc2W: () throws -> [Float],
        fc2B: () throws -> [Float],
        invFreq: [Float]
    ) throws -> MetalWeights {
        let d = cfg.dim, mlp = cfg.mlpInner, qkvD = cfg.qkvDim
        let merged = cfg.merged, pc = cfg.patchChannels
        let totalBytes = weightBufferBytes(config: cfg, depth: depth, fc2Out: fc2Out)
        guard let buffer = device.makeBuffer(length: max(256, totalBytes), options: .storageModeShared) else {
            throw QWHalError.noDevice
        }
        let region = QWMRegion(buffer: buffer, base: 0, length: buffer.length)
        let sub = QWSubAllocator(region: region)

        func uploadF16(_ data: [Float], expectedCount: Int) throws -> QWTensorValue {
            guard data.count == expectedCount else {
                throw QWVisionError.shapeMismatch("expected \(expectedCount) elements, got \(data.count)")
            }
            let tv = try sub.allocate(QWTensorShape([expectedCount], .f16))
            writeF16(data, into: tv)
            return tv
        }

        let posEmbedT = try uploadF16(try posEmbed(), expectedCount: cfg.numPosEmbeds * d)
        let patchWT = try uploadF16(try patchW(), expectedCount: d * pc)
        let patchBT = try uploadF16(try patchB(), expectedCount: d)

        var blocks: [BlockTensors] = []
        blocks.reserveCapacity(depth)
        for b in 0..<depth {
            blocks.append(BlockTensors(
                norm1G: try uploadF16(try blockTensor(b, "norm1.weight", [d]), expectedCount: d),
                norm1B: try uploadF16(try blockTensor(b, "norm1.bias", [d]), expectedCount: d),
                qkvW: try uploadF16(try blockTensor(b, "attn.qkv.weight", [qkvD, d]), expectedCount: qkvD * d),
                qkvB: try uploadF16(try blockTensor(b, "attn.qkv.bias", [qkvD]), expectedCount: qkvD),
                projW: try uploadF16(try blockTensor(b, "attn.proj.weight", [d, d]), expectedCount: d * d),
                projB: try uploadF16(try blockTensor(b, "attn.proj.bias", [d]), expectedCount: d),
                norm2G: try uploadF16(try blockTensor(b, "norm2.weight", [d]), expectedCount: d),
                norm2B: try uploadF16(try blockTensor(b, "norm2.bias", [d]), expectedCount: d),
                l1W: try uploadF16(try blockTensor(b, "mlp.linear_fc1.weight", [mlp, d]), expectedCount: mlp * d),
                l1B: try uploadF16(try blockTensor(b, "mlp.linear_fc1.bias", [mlp]), expectedCount: mlp),
                l2W: try uploadF16(try blockTensor(b, "mlp.linear_fc2.weight", [d, mlp]), expectedCount: d * mlp),
                l2B: try uploadF16(try blockTensor(b, "mlp.linear_fc2.bias", [d]), expectedCount: d)
            ))
        }

        let mergerGT = try uploadF16(try mergerG(), expectedCount: d)
        let mergerBT = try uploadF16(try mergerB(), expectedCount: d)
        let fc1WT = try uploadF16(try fc1W(), expectedCount: merged * merged)
        let fc1BT = try uploadF16(try fc1B(), expectedCount: merged)
        let fc2WT = try uploadF16(try fc2W(), expectedCount: fc2Out * merged)
        let fc2BT = try uploadF16(try fc2B(), expectedCount: fc2Out)
        let invFreqT = try sub.allocate(QWTensorShape([invFreq.count], .f32))
        writePODArray(invFreq, into: invFreqT)

        return MetalWeights(
            buffer: buffer,
            posEmbed: posEmbedT, patchW: patchWT, patchB: patchBT,
            blocks: blocks,
            mergerG: mergerGT, mergerB: mergerBT,
            fc1W: fc1WT, fc1B: fc1BT, fc2W: fc2WT, fc2B: fc2BT,
            invFreq: invFreqT, fc2Out: fc2Out
        )
    }

    static func packFromWeights(_ w: QWVisionWeights, device: any MTLDevice) throws -> MetalWeights {
        try packTensors(
            config: w.config,
            depth: w.blocks.count,
            fc2Out: w.fc2Out,
            device: device,
            posEmbed: { w.posEmbed },
            patchW: { w.patchW },
            patchB: { w.patchB },
            blockTensor: { idx, suffix, _ in
                let blk = w.blocks[idx]
                switch suffix {
                case "norm1.weight": return blk.norm1G
                case "norm1.bias": return blk.norm1B
                case "attn.qkv.weight": return blk.qkvW
                case "attn.qkv.bias": return blk.qkvB
                case "attn.proj.weight": return blk.projW
                case "attn.proj.bias": return blk.projB
                case "norm2.weight": return blk.norm2G
                case "norm2.bias": return blk.norm2B
                case "mlp.linear_fc1.weight": return blk.l1W
                case "mlp.linear_fc1.bias": return blk.l1B
                case "mlp.linear_fc2.weight": return blk.l2W
                case "mlp.linear_fc2.bias": return blk.l2B
                default: throw QWVisionError.missingTensor("blocks.\(idx).\(suffix)")
                }
            },
            mergerG: { w.mergerGamma },
            mergerB: { w.mergerBeta },
            fc1W: { w.fc1W },
            fc1B: { w.fc1B },
            fc2W: { w.fc2W },
            fc2B: { w.fc2B },
            invFreq: w.invFreq
        )
    }

    static func packFromCheckpoint(
        _ ck: QWVisionCheckpoint,
        fc2Out: Int,
        device: any MTLDevice
    ) throws -> MetalWeights {
        let cfg = ck.config
        let d = cfg.dim, merged = cfg.merged
        let depth = try QWVisionWeights.countBlocks(in: ck)
        guard ck.has("merger.norm.weight"), ck.has("merger.norm.bias") else {
            throw QWVisionError.badLayout("merger.norm non-affine law violated (affine weights required)")
        }
        return try packTensors(
            config: cfg,
            depth: depth,
            fc2Out: fc2Out,
            device: device,
            posEmbed: { try ck.f32("pos_embed.weight", want: [cfg.numPosEmbeds, d]) },
            patchW: { try QWVisionWeights.loadFoldedPatchWeight(from: ck, config: cfg) },
            patchB: { try ck.f32("patch_embed.proj.bias", want: [d]) },
            blockTensor: { idx, suffix, shape in
                try ck.f32("blocks.\(idx).\(suffix)", want: shape)
            },
            mergerG: { try ck.f32("merger.norm.weight", want: [d]) },
            mergerB: { try ck.f32("merger.norm.bias", want: [d]) },
            fc1W: { try ck.f32("merger.linear_fc1.weight", want: [merged, merged]) },
            fc1B: { try ck.f32("merger.linear_fc1.bias", want: [merged]) },
            fc2W: { try ck.f32("merger.linear_fc2.weight", want: [fc2Out, merged]) },
            fc2B: { try ck.f32("merger.linear_fc2.bias", want: [fc2Out]) },
            invFreq: QWVisionWeights.computeInvFreq(config: cfg)
        )
    }

    // MARK: - Compute Dispatch Encoders

    func encodeGemm(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        x: QWTensorValue,
        w: QWTensorValue,
        bias: QWTensorValue?,
        residual: QWTensorValue?,
        y: QWTensorValue,
        m: Int,
        n: Int,
        k: Int
    ) {
        guard m > 0, n > 0 else { return }
        let flags: Int = (bias != nil ? 1 : 0) | (residual != nil ? 2 : 0)
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        w.bind(enc, index: 1)
        (bias ?? y).bind(enc, index: 2)
        (residual ?? y).bind(enc, index: 3)
        y.bind(enc, index: 4)
        enc.setUInt32s(m, n, k, flags, startIndex: 5)
        let grid = MTLSize(width: (n + 63) / 64, height: (m + 31) / 32, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func encodePosEmbed(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        s: QWTensorValue,
        posEmbed: QWTensorValue,
        binsIdx: QWTensorValue,
        binsWgt: QWTensorValue,
        nPatch: Int,
        dim: Int,
        numPos: Int
    ) {
        guard nPatch > 0, dim > 0 else { return }
        enc.setComputePipelineState(pso)
        enc.bindTensors(s, posEmbed, binsIdx, binsWgt, startIndex: 0)
        enc.setUInt32s(nPatch, dim, numPos, startIndex: 4)
        let grid = MTLSize(width: (dim + 255) / 256, height: nPatch, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeLayerNorm(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        x: QWTensorValue,
        gamma: QWTensorValue,
        beta: QWTensorValue,
        out: QWTensorValue,
        rows: Int,
        dim: Int,
        eps: Float
    ) {
        guard rows > 0, dim > 0 else { return }
        enc.setComputePipelineState(pso)
        enc.bindTensors(x, gamma, beta, out, startIndex: 0)
        enc.setUInt32s(rows, dim, startIndex: 4)
        enc.setPod(eps, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    func encodeRoPE(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        qkv: QWTensorValue,
        posIds: QWTensorValue,
        invFreq: QWTensorValue,
        nPatch: Int,
        heads: Int,
        headDim: Int,
        dim: Int,
        invFreqCount: Int
    ) {
        guard nPatch > 0, heads > 0, headDim > 1 else { return }
        enc.setComputePipelineState(pso)
        enc.bindTensors(qkv, posIds, invFreq, startIndex: 0)
        enc.setUInt32s(nPatch, heads, headDim, dim, invFreqCount, startIndex: 3)
        let tptg = min(128, max(32, ((headDim / 2 + 31) / 32) * 32))
        enc.dispatchThreadgroups(
            MTLSize(width: nPatch, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
    }

    func encodeFlashAttn(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        qkv: QWTensorValue,
        out: QWTensorValue,
        cuSeqlens: QWTensorValue,
        numSegs: Int,
        maxSegLen: Int,
        heads: Int,
        headDim: Int,
        dim: Int,
        scale: Float
    ) {
        guard numSegs > 0, maxSegLen > 0, heads > 0, headDim > 0 else { return }
        enc.setComputePipelineState(pso)
        enc.bindTensors(qkv, out, cuSeqlens, startIndex: 0)
        enc.setUInt32s(numSegs, heads, headDim, dim, startIndex: 3)
        enc.setPod(scale, index: 7)
        let grid = MTLSize(width: (maxSegLen + 31) / 32, height: heads, depth: numSegs)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func encodeGelu(
        _ enc: any MTLComputeCommandEncoder,
        pso: any MTLComputePipelineState,
        x: QWTensorValue,
        count: Int
    ) {
        guard count > 0 else { return }
        enc.setComputePipelineState(pso)
        x.bind(enc, index: 0)
        enc.setUInt32s(count, startIndex: 1)
        let grid = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
