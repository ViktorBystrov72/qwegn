// qwgemv_hq4_multirow_bf16.metal — Multi-row (M=1..8) HQ4_G32_M5 BF16-out
// decode GEMV reading Block_HQ4_G32 weights once per 4-row SIMDgroup tile.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant.metalh"
using namespace metal;

static inline bfloat qwg_hq4_f32_to_bf16_rne(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7F800000u) == 0x7F800000u) return bfloat(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return bfloat(as_type<float>(u & 0xFFFF0000u));
}

template <uint M, uint R, bool IS_AQ4>
static inline void qwg_gemv_q4g_bf16_m_body(
    device const Block_HQ4_G32* blocks,
    device const half* x,
    device bfloat* yBF16,
    uint realN,
    uint K,
    uint mRows,
    uint tg,
    uint ti,
    uint tptg) {
#pragma clang fp contract(off) reassociate(off)
    const uint simdPerTG = max(1u, tptg >> 5u);
    const uint sgi = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row0 = (tg * simdPerTG + sgi) * R;
    if (row0 >= realN) return;
    const uint nBlocks = K >> 5u;
    device const Block_HQ4_G32* r0 = blocks + (size_t)row0 * nBlocks;
    device const Block_HQ4_G32* rows[R];
    rows[0] = r0;
    #pragma unroll
    for (uint r = 1u; r < R; ++r) {
        rows[r] = (row0 + r < realN) ? (r0 + r * nBlocks) : r0;
    }
    float out[R][M];
    qwg_gemv_hq4_rows_r_m_t<M, R, IS_AQ4>(rows, x, K, lane, out, mRows);
    if (lane != 0u) return;
    for (uint m = 0u; m < M; ++m) {
        if (m >= mRows) break;
        device bfloat* ym = yBF16 + (size_t)m * realN;
        #pragma unroll
        for (uint r = 0u; r < R; ++r) {
            if (row0 + r < realN) ym[row0 + r] = qwg_hq4_f32_to_bf16_rne(out[r][m]);
        }
    }
}

kernel void qwg_gemv_q4g_dx_bf16_t4(
    device const Block_HQ4_G32* blocks [[buffer(0)]],
    device const half* x               [[buffer(1)]],
    device float* y32                  [[buffer(3)]],
    constant uint& N                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    device bfloat* yBF16               [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint ti   [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    (void)y32;
    const uint rawRows = (N >> 24u) & 0x1Fu;
    const uint mRows = max(1u, min(8u, rawRows));
    const uint realN = ((N & 0xFF000000u) != 0u) ? (N & 0x00FFFFFFu) : N;
    if (QW_IS_AQ4) {
        if (mRows == 1u) {
            qwg_gemv_q4g_bf16_m_body<1u, 4u, true>(blocks, x, yBF16, realN, K, 1u, tg, ti, tptg);
        } else if (mRows <= 4u) {
            qwg_gemv_q4g_bf16_m_body<4u, 2u, true>(blocks, x, yBF16, realN, K, mRows, tg, ti, tptg);
        } else {
            qwg_gemv_q4g_bf16_m_body<8u, 2u, true>(blocks, x, yBF16, realN, K, mRows, tg, ti, tptg);
        }
    } else {
        if (mRows == 1u) {
            qwg_gemv_q4g_bf16_m_body<1u, 4u, false>(blocks, x, yBF16, realN, K, 1u, tg, ti, tptg);
        } else if (mRows <= 4u) {
            qwg_gemv_q4g_bf16_m_body<4u, 2u, false>(blocks, x, yBF16, realN, K, mRows, tg, ti, tptg);
        } else {
            qwg_gemv_q4g_bf16_m_body<8u, 2u, false>(blocks, x, yBF16, realN, K, mRows, tg, ti, tptg);
        }
    }
}
