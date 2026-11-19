// qwgemm_moe_f32.metal — Routed MoE f32 GEMM kernels for HQ4_G32 weights.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_moe_f32_common.metalh"
using namespace metal;

kernel void qwg_moe_gateup_q4g_f32_r1(device const Block_HQ4_G32* blocks     [[buffer(0)]],
                                       device const float*         x          [[buffer(1)]],
                                       device const int*           ids        [[buffer(2)]],
                                       device float*               out        [[buffer(3)]],
                                       constant uint&              N          [[buffer(4)]],
                                       constant uint&              K          [[buffer(5)]],
                                       constant uint&              E          [[buffer(6)]],
                                       device const Block_HQ4_G32* coldBlocks [[buffer(7)]],
                                       uint3 tg   [[threadgroup_position_in_grid]],
                                       uint  tid  [[thread_index_in_threadgroup]],
                                       uint  sgi  [[simdgroup_index_in_threadgroup]],
                                       uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float tgAct[128];
    threadgroup QWEmptyTgLut tgLut;
    const QWMoEGateUpF32Params p = { N, K, E };
    qwg_moe_gateup_f32_r1_body<QWQuantHQ4>(blocks, x, ids, out, p, coldBlocks, tg, tid, sgi, lane, tgAct, tgLut);
}

kernel void qwg_moe_down_shared_combine_q4g_f32(device const Block_HQ4_G32* blocks     [[buffer(0)]],
                                                device const float*         x          [[buffer(1)]],
                                                device const float*         w          [[buffer(2)]],
                                                device const float*         shared     [[buffer(3)]],
                                                device const float*         gate       [[buffer(4)]],
                                                device float*               out        [[buffer(5)]],
                                                constant uint&              N          [[buffer(6)]],
                                                constant uint&              K          [[buffer(7)]],
                                                constant uint&              E          [[buffer(8)]],
                                                device const int*           ids        [[buffer(9)]],
                                                device half*                out16      [[buffer(10)]],
                                                constant uint&              hasOut16   [[buffer(11)]],
                                                device const Block_HQ4_G32* coldBlocks [[buffer(12)]],
                                                uint2 tg   [[threadgroup_position_in_grid]],
                                                uint  tid  [[thread_index_in_threadgroup]],
                                                uint  sgi  [[simdgroup_index_in_threadgroup]],
                                                uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup QWEmptyTgLut tgLut;
    const QWMoEDownF32Params p = { N, K, E, hasOut16 };
    qwg_moe_down_shared_combine_f32_body<QWQuantHQ4>(
        blocks, x, w, shared, gate, out, p, ids, out16, coldBlocks, tg, tid, sgi, lane, tgLut);
}
