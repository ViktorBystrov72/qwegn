// QWGemmMoE+CompactQWP.swift — Active-bucket compact QWP MoE (T<=16) and multi-row HQ4_G32 BF16 decode GEMV encoders.
import Foundation
import Metal
import QWAlloc
import QWHAL

private struct QWMoECompactQWPParams {
    let n: UInt32
    let k: UInt32
    let numExperts: UInt32
    let residentE: UInt32
    let flags: UInt32
    let gateCap: Float
    let upCap: Float
}

extension QWGemm {
    /// Returns the compact GateUp+SwiGLU(+FWHT32) kernel name for a QWP container, if supported.
    public static func compactGateUpKernelName(for kind: QWWeightView.ContainerKind, hasSoftcap: Bool = false) -> String? {
        switch kind {
        case .incumbentHQ4G32:
            return hasSoftcap ? "qwg_moe_compact_q4g_gateup_f32_sc" : "qwg_moe_compact_q4g_gateup_f32"
        case .nativeAffine(4, 32), .rotatedHQ(4, 32):
            return hasSoftcap ? "qwg_moe_compact_aq4_g32_gateup_f32_sc" : "qwg_moe_compact_aq4_g32_gateup_f32"
        case .nativeAffine(4, 64), .rotatedHQ(4, 64):
            return hasSoftcap ? "qwg_moe_compact_aq4_g64_gateup_f32_sc" : "qwg_moe_compact_aq4_g64_gateup_f32"
        case .nativeAffine(8, 32), .rotatedHQ(8, 32):
            return hasSoftcap ? "qwg_moe_compact_aq8_g32_gateup_f32_sc" : "qwg_moe_compact_aq8_g32_gateup_f32"
        case .shvq(2, 32):
            return hasSoftcap ? "qwg_moe_compact_shvq2_g32_gateup_f32_sc" : "qwg_moe_compact_shvq2_g32_gateup_f32"
        case .shvq(2, 64):
            return hasSoftcap ? "qwg_moe_compact_shvq2_g64_gateup_f32_sc" : "qwg_moe_compact_shvq2_g64_gateup_f32"
        case .shvq(2, 128):
            return hasSoftcap ? "qwg_moe_compact_shvq2_g128_gateup_f32_sc" : "qwg_moe_compact_shvq2_g128_gateup_f32"
        case .shvq(3, 32):
            return hasSoftcap ? "qwg_moe_compact_shvq3_g32_gateup_f32_sc" : "qwg_moe_compact_shvq3_g32_gateup_f32"
        case .shvq(3, 64):
            return hasSoftcap ? "qwg_moe_compact_shvq3_g64_gateup_f32_sc" : "qwg_moe_compact_shvq3_g64_gateup_f32"
        case .shvq(4, 32):
            return hasSoftcap ? "qwg_moe_compact_shvq4_g32_gateup_f32_sc" : "qwg_moe_compact_shvq4_g32_gateup_f32"
        case .shvq(4, 64):
            return hasSoftcap ? "qwg_moe_compact_shvq4_g64_gateup_f32_sc" : "qwg_moe_compact_shvq4_g64_gateup_f32"
        default:
            return nil
        }
    }

    /// Returns the compact Down kernel name for a QWP container, if supported.
    public static func compactDownKernelName(for kind: QWWeightView.ContainerKind) -> String? {
        switch kind {
        case .incumbentHQ4G32:                  return "qwg_moe_compact_q4g_down_f32"
        case .nativeAffine(4, 32), .rotatedHQ(4, 32): return "qwg_moe_compact_aq4_g32_down_f32"
        case .nativeAffine(4, 64), .rotatedHQ(4, 64): return "qwg_moe_compact_aq4_g64_down_f32"
        case .nativeAffine(8, 32), .rotatedHQ(8, 32): return "qwg_moe_compact_aq8_g32_down_f32"
        case .shvq(2, 32):                       return "qwg_moe_compact_shvq2_g32_down_f32"
        case .shvq(2, 64):                       return "qwg_moe_compact_shvq2_g64_down_f32"
        case .shvq(2, 128):                      return "qwg_moe_compact_shvq2_g128_down_f32"
        case .shvq(3, 32):                       return "qwg_moe_compact_shvq3_g32_down_f32"
        case .shvq(3, 64):                       return "qwg_moe_compact_shvq3_g64_down_f32"
        case .shvq(4, 32):                       return "qwg_moe_compact_shvq4_g32_down_f32"
        case .shvq(4, 64):                       return "qwg_moe_compact_shvq4_g64_down_f32"
        default:                                 return nil
        }
    }

    /// True when both GateUp and Down QWP banks have compact active-bucket kernels.
    public static func supportsCompactQWP(gateUp: QWWeightView, down: QWWeightView) -> Bool {
        compactGateUpKernelName(for: gateUp.containerKind) != nil
            && compactDownKernelName(for: down.containerKind) != nil
    }

    /// Encodes active-bucket compact QWP MoE (`T <= 16`) followed by route reduction and shared-expert combine.
    public static func encodeMoEQWPCompactRoutedCombine(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        x: QWTensorValue, gateUp: QWWeightView, down: QWWeightView,
        ids: QWTensorValue, weights: QWTensorValue,
        shared: QWTensorValue, gate: QWTensorValue, out: QWTensorValue,
        expertOff: QWTensorValue, activeExperts: QWTensorValue, activeIndirect: QWTensorValue,
        sortedPairIdx: QWTensorValue, sortedTok: QWTensorValue,
        actBuf: QWTensorValue, dnAccBuf: QWTensorValue,
        hidden: Int, experts: Int, routes: Int, intermediate: Int, tokens: Int,
        out16: QWTensorValue? = nil,
        coldGateUp: QWWeightView? = nil, coldDown: QWWeightView? = nil,
        gateSoftcap: Float = 0, upSoftcap: Float = 0,
        applyFwht: Bool? = nil, skipShared: Bool = false,
        unweightedSingle: Bool = false, numGroups: Int = 0,
        concurrentBarriers: Bool = false,
        skipOut32: Bool = false,
        nofmaShared: Bool = true
    ) throws {
        let hasSoftcap = (gateSoftcap > 0 || upSoftcap > 0)
        guard let guKernel = compactGateUpKernelName(for: gateUp.containerKind, hasSoftcap: hasSoftcap),
              let dnKernel = compactDownKernelName(for: down.containerKind) else {
            throw QWGemmError.badWeights(
                "unsupported QWP compact MoE containers: \(gateUp.formatName) / \(down.formatName)")
        }
        guard tokens >= 1, tokens <= 16, experts >= 1, experts <= 512, routes >= 1,
              hidden > 0, hidden % 32 == 0, intermediate > 0, intermediate % 32 == 0 else {
            throw QWGemmError.badShape(
                "compact QWP MoE invalid shape T=\(tokens) E=\(experts) R=\(routes) H=\(hidden) I=\(intermediate)")
        }
        guard gateUp.k == hidden, down.k == intermediate else {
            throw QWGemmError.badShape(
                "compact QWP MoE bank K mismatch: gateUp.k=\(gateUp.k) H=\(hidden), down.k=\(down.k) I=\(intermediate)")
        }
        if let cg = coldGateUp, cg.containerKind != gateUp.containerKind || cg.k != gateUp.k {
            throw QWGemmError.badWeights("compact QWP coldGateUp container/k mismatch")
        }
        if let cd = coldDown, cd.containerKind != down.containerKind || cd.k != down.k {
            throw QWGemmError.badWeights("compact QWP coldDown container/k mismatch")
        }
        let guResidentE = coldGateUp != nil ? (gateUp.n / (2 * intermediate)) : 0
        let dnResidentE = coldDown != nil ? (down.n / hidden) : 0
        let totalGuE = (gateUp.n + (coldGateUp?.n ?? 0)) / (2 * intermediate)
        let totalDnE = (down.n + (coldDown?.n ?? 0)) / hidden
        guard totalGuE >= experts, totalDnE >= experts else {
            throw QWGemmError.badWeights("compact QWP MoE banks cover \(min(totalGuE, totalDnE)) < \(experts) experts")
        }
        let pairs = tokens * routes
        guard x.shape.dtype == .f32, x.shape.elementCount >= tokens * hidden,
              ids.shape.dtype == .i32, ids.shape.elementCount >= pairs,
              weights.shape.dtype == .f32, weights.shape.elementCount >= pairs,
              out.shape.dtype == .f32, out.shape.elementCount >= tokens * hidden else {
            throw QWGemmError.badShape("compact QWP MoE input/output tensor dtype or size mismatch")
        }
        guard skipShared || (shared.shape.dtype == .f32 && shared.shape.elementCount >= tokens * hidden
            && gate.shape.dtype == .f32 && gate.shape.elementCount >= tokens) else {
            throw QWGemmError.badShape("compact QWP MoE shared/gate tensor mismatch")
        }
        guard expertOff.shape.elementCount >= experts + 1,
              activeExperts.shape.elementCount >= experts,
              activeIndirect.bytes >= QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes,
              sortedPairIdx.shape.elementCount >= pairs,
              sortedTok.shape.elementCount >= pairs,
              actBuf.shape.dtype == .f32, actBuf.shape.elementCount >= pairs * intermediate,
              dnAccBuf.shape.dtype == .f32, dnAccBuf.shape.elementCount >= pairs * hidden else {
            throw QWGemmError.badShape("compact QWP MoE scratch buffer too small")
        }
        let effectiveOut16: QWTensorValue?
        if let o16 = out16 {
            let expected = max(1, numGroups) * tokens * hidden
            guard o16.shape.elementCount >= expected else {
                throw QWGemmError.badShape("compact QWP MoE out16 needs >= \(expected) elements")
            }
            switch o16.shape.dtype {
            case .f16, .bf16:
                effectiveOut16 = o16
            case .f32:
                let aliasesOut = o16.region.buffer === out.region.buffer && o16.offset == out.offset
                effectiveOut16 = aliasesOut ? nil : o16
            default:
                throw QWGemmError.badShape("compact QWP MoE unsupported out16 dtype \(o16.shape.dtype)")
            }
        } else {
            effectiveOut16 = nil
        }

        @inline(__always) func barrier() {
            if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        }

        let bucketPlan = QWGemmFP.MoEActiveBucketPlan(
            experts: experts, intermediate: intermediate, hidden: hidden, downTileRows: 16)
        try encodeMoEGroupAndActiveBuckets(
            enc, hal: hal, ids: ids, expertOff: expertOff,
            activeExperts: activeExperts, activeIndirect: activeIndirect,
            sortedPairIdx: sortedPairIdx, sortedTok: sortedTok,
            experts: experts, totalPairs: pairs, routes: routes,
            plan: bucketPlan, concurrentBarriers: concurrentBarriers)
        barrier()

        let guParams = QWMoECompactQWPParams(
            n: UInt32(2 * intermediate), k: UInt32(hidden),
            numExperts: UInt32(experts), residentE: UInt32(guResidentE),
            flags: (applyFwht ?? (down.isRotated && (intermediate % 32 == 0))) ? 1 : 0,
            gateCap: gateSoftcap, upCap: upSoftcap)
        let indBuf = activeIndirect.region.buffer
        let indBase = activeIndirect.region.base + activeIndirect.offset
        enc.setComputePipelineState(try hal.pipeline(guKernel))
        enc.bindTensors(
            gateUp.blocks, x, expertOff, activeExperts, activeIndirect,
            sortedTok, sortedPairIdx, actBuf, (coldGateUp ?? gateUp).blocks, startIndex: 0)
        enc.setPod(guParams, index: 9)
        enc.dispatchThreadgroups(
            indirectBuffer: indBuf,
            indirectBufferOffset: indBase + QWGemmFP.MoEActiveBucketPlan.gateUpIndirectByteOffset,
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        barrier()

        let dnParams = QWMoECompactQWPParams(
            n: UInt32(hidden), k: UInt32(intermediate),
            numExperts: UInt32(experts), residentE: UInt32(dnResidentE),
            flags: 0, gateCap: 0, upCap: 0)
        enc.setComputePipelineState(try hal.pipeline(dnKernel))
        enc.bindTensors(
            down.blocks, actBuf, expertOff, activeExperts, activeIndirect,
            sortedPairIdx, dnAccBuf, (coldDown ?? down).blocks, startIndex: 0)
        enc.setPod(dnParams, index: 8)
        enc.dispatchThreadgroups(
            indirectBuffer: indBuf,
            indirectBufferOffset: indBase + QWGemmFP.MoEActiveBucketPlan.downIndirectByteOffset,
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        barrier()

        try encodeMoEDownReduceCombineF32(
            enc, hal: hal, dnAcc: dnAccBuf, invPerm: nil, weights: weights,
            shared: shared, gate: gate, out: out, hidden: hidden, routes: routes,
            tokens: tokens, out16: effectiveOut16, skipShared: skipShared,
            unweightedSingle: unweightedSingle, ids: ids, numGroups: numGroups,
            skipOut32: skipOut32, nofmaShared: nofmaShared)
    }

    /// Encodes multi-row (`M = 1..8` chunks) `HQ4_G32_M5` BF16-output decode GEMV reading weights once per tile.
    public static func encodeHQ4DecodeBF16(
        _ enc: any MTLComputeCommandEncoder,
        hal: QWHal,
        weight: QWWeightView,
        x16: QWTensorValue,
        yBF16: QWTensorValue,
        rows: Int
    ) throws {
        guard !weight.isSHVQ, weight.bits == 4, weight.groupSize == 32 else {
            throw QWGemmError.badWeights(
                "encodeHQ4DecodeBF16 requires HQ4_G32 or 4-bit g32 weights, got \(weight.formatName)")
        }
        guard weight.n > 0, weight.n < (1 << 24), weight.k > 0, weight.k % 32 == 0, rows > 0 else {
            throw QWGemmError.badShape(
                "encodeHQ4DecodeBF16 invalid shape N=\(weight.n) K=\(weight.k) rows=\(rows)")
        }
        guard x16.shape.dtype == .f16, x16.shape.elementCount >= rows * weight.k else {
            throw QWGemmError.badShape("encodeHQ4DecodeBF16 x16 requires f16 >= \(rows * weight.k) elements")
        }
        guard yBF16.shape.dtype == .bf16, yBF16.shape.elementCount >= rows * weight.n else {
            throw QWGemmError.badShape("encodeHQ4DecodeBF16 yBF16 requires bf16 >= \(rows * weight.n) elements")
        }
        let pso = try hal.pipeline("qwg_gemv_q4g_dx_bf16_t4", isAQ4: !weight.isIncumbentContainer)
        enc.setComputePipelineState(pso)
        weight.blocks.bind(enc, index: 0)
        var k32 = UInt32(weight.k)
        enc.setBytes(&k32, length: 4, index: 5)
        var rowOffset = 0
        while rowOffset < rows {
            let chunk = min(8, rows - rowOffset)
            let rowsPerTG = (chunk == 1) ? 16 : 8
            let groups = (weight.n + rowsPerTG - 1) / rowsPerTG
            let xChunk = (rowOffset == 0 && chunk == rows)
                ? x16
                : x16.slice(byteOffset: rowOffset * weight.k * 2, count: chunk * weight.k, dtype: .f16)
            let yChunk = (rowOffset == 0 && chunk == rows)
                ? yBF16
                : yBF16.slice(byteOffset: rowOffset * weight.n * 2, count: chunk * weight.n, dtype: .bf16)
            xChunk.bind(enc, index: 1)
            yChunk.bind(enc, index: 3)
            yChunk.bind(enc, index: 8)
            var n32 = UInt32(weight.n) | 0x6000_0000 | (UInt32(chunk) << 24)
            enc.setBytes(&n32, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            rowOffset += chunk
        }
    }
}
