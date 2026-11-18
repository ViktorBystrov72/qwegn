// qwgemm_aq_swiglu_nax.metal — Paired-tile NAX gate_up + SwiGLU for AQ{2,3,4,6,8}_G{32,64,128,256}.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_tile.metalh"
#include "include/qw_gemm_nax_swiglu.metalh"

#define QW_AQ_SWIGLU_KERNEL(BITS, GROUP, NAME)                                          \
  kernel void NAME(device const char* blocks [[buffer(0)]],                             \
                   device const half* x      [[buffer(1)]],                             \
                   device half*       out16  [[buffer(2)]],                             \
                   constant uint&     M      [[buffer(3)]],                             \
                   constant uint&     I      [[buffer(4)]],                             \
                   constant uint&     K      [[buffer(5)]],                             \
                   constant uint&     xStride [[buffer(6)]],                            \
                   constant uint&     yStride [[buffer(7)]],                            \
                   uint3 tg       [[threadgroup_position_in_grid]],                     \
                   uint  lid      [[thread_index_in_threadgroup]],                      \
                   uint  simd_gid [[simdgroup_index_in_threadgroup]],                   \
                   uint  simd_lid [[thread_index_in_simdgroup]]) {                      \
    threadgroup half tgGU[4096];                                                        \
    const uint mode = (M >> 29u) & 3u;                                                  \
    const int realM = int(M & 0x1FFFFFFFu);                                             \
    if (mode == 1u) {                                                                   \
      qmm_t_nax_swiglu_impl<16, 64, 1, 4, BITS, GROUP>(                                 \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU);                                           \
    } else if (mode == 2u) {                                                            \
      qmm_t_nax_swiglu_impl<32, 32, 2, 2, BITS, GROUP>(                                 \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU);                                           \
    } else {                                                                            \
      qmm_t_nax_swiglu_impl<64, 32, 2, 2, BITS, GROUP>(                                 \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU);                                           \
    }                                                                                   \
  }

QW_AQ_SWIGLU_KERNEL(2, 32,  qwg_gemm_aq2_g32_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(2, 64,  qwg_gemm_aq2_g64_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(2, 128, qwg_gemm_aq2_g128_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(2, 256, qwg_gemm_aq2_g256_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(3, 32,  qwg_gemm_aq3_g32_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(3, 64,  qwg_gemm_aq3_g64_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(3, 128, qwg_gemm_aq3_g128_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(3, 256, qwg_gemm_aq3_g256_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(4, 32,  qwg_gemm_aq4_g32_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(4, 64,  qwg_gemm_aq4_g64_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(4, 128, qwg_gemm_aq4_g128_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(4, 256, qwg_gemm_aq4_g256_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(6, 32,  qwg_gemm_aq6_g32_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(6, 64,  qwg_gemm_aq6_g64_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(6, 128, qwg_gemm_aq6_g128_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(6, 256, qwg_gemm_aq6_g256_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(8, 32,  qwg_gemm_aq8_g32_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(8, 64,  qwg_gemm_aq8_g64_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(8, 128, qwg_gemm_aq8_g128_h16_swiglu)
QW_AQ_SWIGLU_KERNEL(8, 256, qwg_gemm_aq8_g256_h16_swiglu)
