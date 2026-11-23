// QWGemmMoE+Group.swift — Fused single-TG MoE pair grouping and active-bucket indirect dispatch builder.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL

private struct QWGemmMoEGroupBucketsParams {
    var experts: UInt32
    var totalPairs: UInt32
    var topK: UInt32
    var flags: UInt32
    var guGridX: UInt32
    var dnGridX: UInt32
}

extension QWGemm {
    /// Fused single-TG pair counting sort (`totalPairs <= 512`, `experts <= 1024`) + active-bucket indirect build.
    public static func encodeMoEGroupBuildBucketsF32(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        ids: QWTensorValue, expertOff: QWTensorValue,
        activeExperts: QWTensorValue, activeIndirect: QWTensorValue,
        sortedPairIdx: QWTensorValue, sortedTok: QWTensorValue,
        experts: Int, totalPairs: Int, routes: Int,
        plan: QWGemmFP.MoEActiveBucketPlan,
        invPerm: QWTensorValue? = nil, directSlotTok: Bool = false
    ) throws {
        guard experts >= 1, experts <= 1024, totalPairs >= 0, totalPairs <= 512, routes >= 1 else {
            throw QWGemmError.badShape("fused moe group+buckets requires E in 1...1024, pairs <= 512, routes >= 1")
        }
        guard activeIndirect.bytes >= QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes,
              activeExperts.bytes >= experts * 4 else {
            throw QWGemmError.badShape("fused moe group+buckets activeExperts/activeIndirect buffer too small")
        }
        let flags: UInt32 = (invPerm != nil ? 1 : 0) | (directSlotTok ? 2 : 0)
        let params = QWGemmMoEGroupBucketsParams(
            experts: UInt32(experts), totalPairs: UInt32(totalPairs), topK: UInt32(routes),
            flags: flags, guGridX: UInt32(plan.gateUpGridX), dnGridX: UInt32(plan.downGridX))
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_group_build_buckets_f32"))
        enc.bindTensors(ids, expertOff, sortedPairIdx, sortedTok, startIndex: 0)
        (invPerm ?? sortedPairIdx).bind(enc, index: 4)
        enc.bindTensors(activeExperts, activeIndirect, startIndex: 5)
        enc.setPod(params, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dispatches fused group+active-buckets for `totalPairs <= 512 && experts <= 1024`, else separate fallback.
    public static func encodeMoEGroupAndActiveBuckets(
        _ enc: any MTLComputeCommandEncoder, hal: QWHal,
        ids: QWTensorValue, expertOff: QWTensorValue,
        activeExperts: QWTensorValue, activeIndirect: QWTensorValue,
        sortedPairIdx: QWTensorValue, sortedTok: QWTensorValue,
        experts: Int, totalPairs: Int, routes: Int,
        plan: QWGemmFP.MoEActiveBucketPlan,
        invPerm: QWTensorValue? = nil, directSlotTok: Bool = false,
        concurrentBarriers: Bool = false
    ) throws {
        if totalPairs <= 512 && experts <= 1024 {
            try encodeMoEGroupBuildBucketsF32(
                enc, hal: hal, ids: ids, expertOff: expertOff,
                activeExperts: activeExperts, activeIndirect: activeIndirect,
                sortedPairIdx: sortedPairIdx, sortedTok: sortedTok,
                experts: experts, totalPairs: totalPairs, routes: routes,
                plan: plan, invPerm: invPerm, directSlotTok: directSlotTok)
            return
        }
        try encodeMoEGroupPairsF32(
            enc, hal: hal, ids: ids, expertOff: expertOff,
            sortedPairIdx: sortedPairIdx, sortedTok: sortedTok,
            experts: experts, totalPairs: totalPairs, routes: routes,
            invPerm: invPerm, directSlotTok: directSlotTok)
        if concurrentBarriers { enc.memoryBarrier(scope: .buffers) }
        try QWGemmFP.encodeMoEBuildActiveBuckets(
            enc, hal: hal, expertOff: QWFPBufferSlice(expertOff),
            activeExperts: QWFPBufferSlice(activeExperts),
            indirectArgs: QWFPBufferSlice(activeIndirect), plan: plan)
    }
}
