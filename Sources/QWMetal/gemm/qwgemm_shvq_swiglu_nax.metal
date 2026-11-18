// qwgemm_shvq_swiglu_nax.metal — Paired-tile NAX gate_up + SwiGLU for SHVQ{1,15,2,3,4}_G{32,64,128,256}.
#define QW_SHVQ_ROUTED_DIRECT_LUT 1
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_moe_tile.metalh"
#include "include/qw_gemm_nax_swiglu.metalh"

#define QW_SHVQ_SWIGLU_IMPL(BITS, GROUP, HAS_TG, DECL_TG, PREP_TG, LUT_PTR)             \
  kernel void qwg_gemm_shvq##BITS##_g##GROUP##_h16_swiglu(                              \
      device const char* blocks [[buffer(0)]],                                          \
      device const half* x      [[buffer(1)]],                                          \
      device half*       out16  [[buffer(2)]],                                          \
      constant uint&     M      [[buffer(3)]],                                          \
      constant uint&     I      [[buffer(4)]],                                          \
      constant uint&     K      [[buffer(5)]],                                          \
      constant uint&     xStride [[buffer(6)]],                                         \
      constant uint&     yStride [[buffer(7)]],                                         \
      uint3 tg       [[threadgroup_position_in_grid]],                                  \
      uint  lid      [[thread_index_in_threadgroup]],                                   \
      uint  simd_gid [[simdgroup_index_in_threadgroup]],                                \
      uint  simd_lid [[thread_index_in_simdgroup]]) {                                   \
    threadgroup half tgGU[4096];                                                        \
    DECL_TG                                                                             \
    PREP_TG                                                                             \
    const uint mode = (M >> 29u) & 3u;                                                  \
    const int realM = int(M & 0x1FFFFFFFu);                                             \
    if (mode == 1u) {                                                                   \
      qmm_t_nax_swiglu_impl<16, 64, 1, 4, BITS, GROUP, true, HAS_TG>(                   \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU, LUT_PTR);                                  \
    } else if (mode == 2u) {                                                            \
      qmm_t_nax_swiglu_impl<32, 32, 2, 2, BITS, GROUP, true, HAS_TG>(                   \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU, LUT_PTR);                                  \
    } else {                                                                            \
      qmm_t_nax_swiglu_impl<64, 32, 2, 2, BITS, GROUP, true, HAS_TG>(                   \
          blocks, x, out16, int(K), int(I), realM, int(xStride), int(yStride),          \
          tg, lid, simd_gid, simd_lid, tgGU, LUT_PTR);                                  \
    }                                                                                   \
  }

#define QW_SHVQ_SWIGLU_TG(BITS, GROUP)                                                  \
  QW_SHVQ_SWIGLU_IMPL(BITS, GROUP, true,                                                \
      threadgroup ShvqTgLut<BITS> tgLut;,                                               \
      tgLut.load(lid, 128u); threadgroup_barrier(mem_flags::mem_threadgroup);,          \
      tgLut.ptr16())

#define QW_SHVQ_SWIGLU_DIRECT(BITS, GROUP)                                              \
  QW_SHVQ_SWIGLU_IMPL(BITS, GROUP, false, , , nullptr)

QW_SHVQ_SWIGLU_TG(1, 32)
QW_SHVQ_SWIGLU_TG(1, 64)
QW_SHVQ_SWIGLU_TG(1, 128)
QW_SHVQ_SWIGLU_TG(1, 256)
QW_SHVQ_SWIGLU_DIRECT(15, 32)
QW_SHVQ_SWIGLU_DIRECT(15, 64)
QW_SHVQ_SWIGLU_DIRECT(15, 128)
QW_SHVQ_SWIGLU_DIRECT(15, 256)
QW_SHVQ_SWIGLU_TG(2, 32)
QW_SHVQ_SWIGLU_TG(2, 64)
QW_SHVQ_SWIGLU_TG(2, 128)
QW_SHVQ_SWIGLU_TG(2, 256)
QW_SHVQ_SWIGLU_TG(3, 32)
QW_SHVQ_SWIGLU_TG(3, 64)
QW_SHVQ_SWIGLU_TG(3, 128)
QW_SHVQ_SWIGLU_TG(3, 256)
QW_SHVQ_SWIGLU_DIRECT(4, 32)
QW_SHVQ_SWIGLU_DIRECT(4, 64)
QW_SHVQ_SWIGLU_DIRECT(4, 128)
QW_SHVQ_SWIGLU_DIRECT(4, 256)
