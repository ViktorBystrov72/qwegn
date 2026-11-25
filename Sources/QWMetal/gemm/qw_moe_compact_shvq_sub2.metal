// qw_moe_compact_shvq_sub2.metal — Active-bucket compact MoE decode/verify (T<=16)
// for sub-2-bit SHVQ{1,15}_G{32,64} within the 16 KiB constant LUT budget.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_quant_traits_shvq.metalh"
#include "qw_moe_compact_shvq_fast.metalh"
using namespace metal;

struct QWMoECompactSHVQSub2Params {
    uint N;
    uint K;
    uint numExperts;
    uint residentE;
    uint flags;
    float gateCap;
    float upCap;
};

#define QW_MOE_COMPACT_SHVQ_SUB2_KERNELS(BITS, GROUP) \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_gateup_f32( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const char* coldW [[buffer(8)]], constant QWMoECompactSHVQSub2Params& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_gateup_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQSub2Params, false>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_gateup_f32_sc( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const char* coldW [[buffer(8)]], constant QWMoECompactSHVQSub2Params& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_gateup_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQSub2Params, true>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_shvq##BITS##_g##GROUP##_down_f32( \
    device const char* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedPairIdx [[buffer(5)]], \
    device float* dnSlotOut [[buffer(6)]], device const char* coldW [[buffer(7)]], \
    constant QWMoECompactSHVQSub2Params& p [[buffer(8)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup ShvqTgLut<BITS> tgLut; \
    qwg_moe_compact_shvq_down_impl<QWQuantSHVQ<BITS, GROUP>, QWMoECompactSHVQSub2Params>( \
        w, x, expertOff, activeExperts, activeArgs, sortedPairIdx, \
        dnSlotOut, coldW, p, tg, tid, sgi, lane, tgLut); \
}

QW_MOE_COMPACT_SHVQ_SUB2_KERNELS(1, 32)
QW_MOE_COMPACT_SHVQ_SUB2_KERNELS(1, 64)
QW_MOE_COMPACT_SHVQ_SUB2_KERNELS(15, 32)
QW_MOE_COMPACT_SHVQ_SUB2_KERNELS(15, 64)
