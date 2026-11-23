// qwgemm_moe_aq_f32.metal — Routed MoE f32 GEMM kernels for arbitrary-bit AQ weights.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_moe_f32_common.metalh"
using namespace metal;

#define QW_AQ_MOE_F32_KERNELS(BITS, GROUP) \
kernel void qwg_moe_gateup_aq##BITS##_g##GROUP##_f32_r1(device const char*  w     [[buffer(0)]], \
                                                        device const float* x     [[buffer(1)]], \
                                                        device const int*   ids   [[buffer(2)]], \
                                                        device float*       out   [[buffer(3)]], \
                                                        constant uint&      N     [[buffer(4)]], \
                                                        constant uint&      K     [[buffer(5)]], \
                                                        constant uint&      E     [[buffer(6)]], \
                                                        device const char*  coldW [[buffer(7)]], \
                                                        uint3 tg   [[threadgroup_position_in_grid]], \
                                                        uint  tid  [[thread_index_in_threadgroup]], \
                                                        uint  sgi  [[simdgroup_index_in_threadgroup]], \
                                                        uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup QWEmptyTgLut tgLut; \
    const QWMoEGateUpF32Params p = { N, K, E }; \
    qwg_moe_gateup_f32_r1_body<QWQuantAQ<BITS, GROUP>, false>(w, x, ids, out, p, coldW, float2(0.0f), tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_gateup_aq##BITS##_g##GROUP##_f32_r1_sc(device const char*  w        [[buffer(0)]], \
                                                           device const float* x        [[buffer(1)]], \
                                                           device const int*   ids      [[buffer(2)]], \
                                                           device float*       out      [[buffer(3)]], \
                                                           constant uint&      N        [[buffer(4)]], \
                                                           constant uint&      K        [[buffer(5)]], \
                                                           constant uint&      E        [[buffer(6)]], \
                                                           device const char*  coldW    [[buffer(7)]], \
                                                           constant float2&    softcaps [[buffer(8)]], \
                                                           uint3 tg   [[threadgroup_position_in_grid]], \
                                                           uint  tid  [[thread_index_in_threadgroup]], \
                                                           uint  sgi  [[simdgroup_index_in_threadgroup]], \
                                                           uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup QWEmptyTgLut tgLut; \
    const QWMoEGateUpF32Params p = { N, K, E }; \
    qwg_moe_gateup_f32_r1_body<QWQuantAQ<BITS, GROUP>, true>(w, x, ids, out, p, coldW, softcaps, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_down_shared_combine_aq##BITS##_g##GROUP##_f32(device const char*  wBytes     [[buffer(0)]], \
                                                                  device const float* x          [[buffer(1)]], \
                                                                  device const float* w          [[buffer(2)]], \
                                                                  device const float* shared     [[buffer(3)]], \
                                                                  device const float* gate       [[buffer(4)]], \
                                                                  device float*       out        [[buffer(5)]], \
                                                                  constant uint&      N          [[buffer(6)]], \
                                                                  constant uint&      K          [[buffer(7)]], \
                                                                  constant uint&      E          [[buffer(8)]], \
                                                                  device const int*   ids        [[buffer(9)]], \
                                                                  device half*        out16      [[buffer(10)]], \
                                                                  constant uint&      hasOut16   [[buffer(11)]], \
                                                                  device const char*  coldWBytes [[buffer(12)]], \
                                                                  uint2 tg   [[threadgroup_position_in_grid]], \
                                                                  uint  tid  [[thread_index_in_threadgroup]], \
                                                                  uint  sgi  [[simdgroup_index_in_threadgroup]], \
                                                                  uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup QWEmptyTgLut tgLut; \
    const QWMoEDownF32Params p = { N, K, E, hasOut16 }; \
    qwg_moe_down_shared_combine_f32_body<QWQuantAQ<BITS, GROUP>>( \
        wBytes, x, w, shared, gate, out, p, ids, out16, coldWBytes, tg, tid, sgi, lane, tgLut); \
}

QW_AQ_MOE_F32_KERNELS(2, 32)
QW_AQ_MOE_F32_KERNELS(2, 64)
QW_AQ_MOE_F32_KERNELS(2, 128)
QW_AQ_MOE_F32_KERNELS(2, 256)
QW_AQ_MOE_F32_KERNELS(3, 32)
QW_AQ_MOE_F32_KERNELS(3, 64)
QW_AQ_MOE_F32_KERNELS(3, 128)
QW_AQ_MOE_F32_KERNELS(3, 256)
QW_AQ_MOE_F32_KERNELS(4, 32)
QW_AQ_MOE_F32_KERNELS(4, 64)
QW_AQ_MOE_F32_KERNELS(4, 128)
QW_AQ_MOE_F32_KERNELS(4, 256)
QW_AQ_MOE_F32_KERNELS(6, 32)
QW_AQ_MOE_F32_KERNELS(6, 64)
QW_AQ_MOE_F32_KERNELS(6, 128)
QW_AQ_MOE_F32_KERNELS(6, 256)
QW_AQ_MOE_F32_KERNELS(8, 32)
QW_AQ_MOE_F32_KERNELS(8, 64)
QW_AQ_MOE_F32_KERNELS(8, 128)
QW_AQ_MOE_F32_KERNELS(8, 256)
