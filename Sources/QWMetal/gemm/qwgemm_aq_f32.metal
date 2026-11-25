// qwgemm_aq_f32.metal — Affine-quantized f32 GEMM/GEMV family over QWAQBlock.supported.
// Instantiates compile-time <BITS, GROUP> twins over QWQuantAQ and shared bodies.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_gemm_f32_common.metalh"
using namespace metal;

#define QW_AQ_F32_BM32_KERNEL(BITS, GROUP) \
kernel void qwg_gemm_aq##BITS##_g##GROUP##_f32_bm32(device const char* w [[buffer(0)]], \
                                                    device const float* x [[buffer(1)]], \
                                                    device float* y [[buffer(2)]], \
                                                    constant uint& M [[buffer(3)]], \
                                                    constant uint& N [[buffer(4)]], \
                                                    constant uint& K [[buffer(5)]], \
                                                    constant uint& epi [[buffer(6)]], \
                                                    constant float& scale [[buffer(7)]], \
                                                    uint2 tg [[threadgroup_position_in_grid]], \
                                                    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float As[32 * QWG_F32_BM_LD]; \
    threadgroup float Bs[QWG_F32_BM_ROWS * QWG_F32_BM_LD]; \
    const QWGemmF32BMParams p = { M, N, K, epi, scale }; \
    qwg_gemm_f32_bm_body<QWQuantAQ<BITS, GROUP>, 32, 128>(w, x, y, p, tg, tid, As, Bs); \
}

QW_AQ_F32_BM32_KERNEL(8, 32)
QW_AQ_F32_BM32_KERNEL(4, 32)
QW_AQ_F32_BM32_KERNEL(4, 64)

#define QW_AQ_F32_KERNELS(BITS, GROUP) \
kernel void qwg_gemm_aq##BITS##_g##GROUP##_f32_bm(device const char*  w     [[buffer(0)]], \
                                                  device const float* x     [[buffer(1)]], \
                                                  device float*       y     [[buffer(2)]], \
                                                  constant uint&      M     [[buffer(3)]], \
                                                  constant uint&      N     [[buffer(4)]], \
                                                  constant uint&      K     [[buffer(5)]], \
                                                  constant uint&      epi   [[buffer(6)]], \
                                                  constant float&     scale [[buffer(7)]], \
                                                  uint2 tg  [[threadgroup_position_in_grid]], \
                                                  uint  tid [[thread_index_in_threadgroup]]) { \
    threadgroup float As[QWG_F32_BM_TOKENS * QWG_F32_BM_LD]; \
    threadgroup float Bs[QWG_F32_BM_ROWS * QWG_F32_BM_LD]; \
    const QWGemmF32BMParams p = { M, N, K, epi, scale }; \
    qwg_gemm_f32_bm_body<QWQuantAQ<BITS, GROUP>, QWG_F32_BM_TOKENS, 256>(w, x, y, p, tg, tid, As, Bs); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4(device const char*  w     [[buffer(0)]], \
                                                  device const float* x     [[buffer(1)]], \
                                                  device float*       y     [[buffer(2)]], \
                                                  constant uint&      N     [[buffer(3)]], \
                                                  constant uint&      K     [[buffer(4)]], \
                                                  constant uint&      epi   [[buffer(5)]], \
                                                  constant float&     scale [[buffer(6)]], \
                                                  uint3 tg   [[threadgroup_position_in_grid]], \
                                                  uint  tid  [[thread_index_in_threadgroup]], \
                                                  uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                  uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgC[8 * 32]; \
    threadgroup QWEmptyTgLut tgLut; \
    const QWGemmF32V4Params p = { N, K, epi, scale }; \
    qwg_gemv_f32_v4_body<QWQuantAQ<BITS, GROUP>>(w, x, y, p, tg, tid, sg, lane, tgC, tgLut); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4_am(device const char*  w      [[buffer(0)]], \
                                                     device const float* x      [[buffer(1)]], \
                                                     device float*       y      [[buffer(2)]], \
                                                     device uint*        keysHi [[buffer(7)]], \
                                                     device uint*        keysLo [[buffer(8)]], \
                                                     constant uint&      N      [[buffer(3)]], \
                                                     constant uint&      K      [[buffer(4)]], \
                                                     constant uint&      epi    [[buffer(5)]], \
                                                     constant float&     scale  [[buffer(6)]], \
                                                     uint3 tg   [[threadgroup_position_in_grid]], \
                                                     uint  tid  [[thread_index_in_threadgroup]], \
                                                     uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                     uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint bHi[32], bLo[32]; \
    threadgroup QWEmptyTgLut tgLut; \
    const QWGemmF32V4Params p = { N, K, epi, scale }; \
    qwg_gemv_f32_v4_am_body<QWQuantAQ<BITS, GROUP>>(w, x, y, keysHi, keysLo, p, tg, tid, sg, lane, bHi, bLo, tgLut); \
} \
kernel void qwg_gemv_aq##BITS##_g##GROUP##_f32_v4_am2(device const char*  w      [[buffer(0)]], \
                                                      device const float* x      [[buffer(1)]], \
                                                      device uint*        keysHi [[buffer(2)]], \
                                                      device uint*        keysLo [[buffer(3)]], \
                                                      constant uint&      N      [[buffer(4)]], \
                                                      constant uint&      K      [[buffer(5)]], \
                                                      device float*       y      [[buffer(6)]], \
                                                      uint3 tg   [[threadgroup_position_in_grid]], \
                                                      uint  tid  [[thread_index_in_threadgroup]], \
                                                      uint  sg   [[simdgroup_index_in_threadgroup]], \
                                                      uint  lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint bHi0[4], bLo0[4], bHi1[4], bLo1[4]; \
    threadgroup QWEmptyTgLut tgLut; \
    qwg_gemv_f32_v4_am2_body<QWQuantAQ<BITS, GROUP>>(w, x, keysHi, keysLo, N, K, y, tg, tid, sg, lane, bHi0, bLo0, bHi1, bLo1, tgLut); \
}

QW_AQ_F32_KERNELS(2, 32)
QW_AQ_F32_KERNELS(2, 64)
QW_AQ_F32_KERNELS(2, 128)
QW_AQ_F32_KERNELS(2, 256)
QW_AQ_F32_KERNELS(3, 32)
QW_AQ_F32_KERNELS(3, 64)
QW_AQ_F32_KERNELS(3, 128)
QW_AQ_F32_KERNELS(3, 256)
QW_AQ_F32_KERNELS(4, 32)
QW_AQ_F32_KERNELS(4, 64)
QW_AQ_F32_KERNELS(4, 128)
QW_AQ_F32_KERNELS(4, 256)
QW_AQ_F32_KERNELS(6, 32)
QW_AQ_F32_KERNELS(6, 64)
QW_AQ_F32_KERNELS(6, 128)
QW_AQ_F32_KERNELS(6, 256)
QW_AQ_F32_KERNELS(8, 32)
QW_AQ_F32_KERNELS(8, 64)
QW_AQ_F32_KERNELS(8, 128)
QW_AQ_F32_KERNELS(8, 256)
