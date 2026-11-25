// qw_moe_compact_shvq.metal — Active-bucket compact MoE decode/verify (T<=16)
// for SHVQ{2_g{32,64,128}, 3_g{32,64}, 4_g{32,64}}; LUTs isolated per Law 27.
#include <metal_stdlib>
#include <metal_simdgroup>
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include "qw_quant_traits_shvq.metalh"
#include "qw_moe_compact_shvq_fast.metalh"
using namespace metal;

struct QWMoECompactSHVQParams {
    uint N;
    uint K;
    uint numExperts;
    uint residentE;
    uint flags;
    float gateCap;
    float upCap;
};

#define QW_MOE_COMPACT_SHVQ_KERNELS(BITS, GROUP) \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_gateup_f32( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const char* coldW [[buffer(8)]], constant QWMoECompactSHVQParams& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_gateup_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQParams, false>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_gateup_f32_sc( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const char* coldW [[buffer(8)]], constant QWMoECompactSHVQParams& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_gateup_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQParams, true>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_down_f32( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedPairIdx [[buffer(5)]], \
    device float* dnSlotOut [[buffer(6)]], device const char* coldW [[buffer(7)]], \
    constant QWMoECompactSHVQParams& p [[buffer(8)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_down_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQParams>( \
        w, x, expertOff, activeExperts, activeArgs, sortedPairIdx, \
        dnSlotOut, coldW, p, tg, tid, sgi, lane, tgLut); \
}

QW_MOE_COMPACT_SHVQ_KERNELS(2, 32)
QW_MOE_COMPACT_SHVQ_KERNELS(2, 64)
QW_MOE_COMPACT_SHVQ_KERNELS(2, 128)
QW_MOE_COMPACT_SHVQ_KERNELS(3, 32)
QW_MOE_COMPACT_SHVQ_KERNELS(3, 64)
QW_MOE_COMPACT_SHVQ_KERNELS(4, 32)
QW_MOE_COMPACT_SHVQ_KERNELS(4, 64)
