// qwgemm_aq_mlp.metal — the size-preserving family's MLP gate/up GEMV.
//
// The incumbent's `qwg_gemv_q4g_silu_pair_dx_h16` geometry over the family's
// blocks: the same silu·up epilogue, and NO rotation — the family's weights
// carry none, so the intermediate must not be rotated either.
//
// buffers: 0 blocks | 1 x f16 | 4 N | 5 K | 8 h16

#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_dequant_multi.metalh"
using namespace metal;

/// The gate and up dots of four rows, one simdgroup's worth.
struct QWAQSiluPair {
    float4 gate;
    float4 up;
};

/// Four rows a side: the gate rows r0..r0+3 and the up rows I+r0..I+r0+3,
/// the lane's 32-column block's activations shared by all of them.
template <int BITS, int GROUP>
static inline QWAQSiluPair qwg_aq_silu_pair_rows4(device const char* blocks,
                                                  device const half* x,
                                                  uint r0, uint I, uint K, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    const uint nBlocks = K >> 5u;
    const uint groupBytes = GROUP * BITS / 8 + 4u;
    const uint rowBytes = (K / GROUP) * groupBytes;
    device const char* g0 = blocks + (size_t)r0 * rowBytes;
    device const char* g1 = (r0 + 1u < I) ? (g0 + rowBytes) : g0;
    device const char* g2 = (r0 + 2u < I) ? (g0 + 2u * rowBytes) : g0;
    device const char* g3 = (r0 + 3u < I) ? (g0 + 3u * rowBytes) : g0;
    device const char* u0 = blocks + (size_t)(I + r0) * rowBytes;
    device const char* u1 = (r0 + 1u < I) ? (u0 + rowBytes) : u0;
    device const char* u2 = (r0 + 2u < I) ? (u0 + 2u * rowBytes) : u0;
    device const char* u3 = (r0 + 3u < I) ? (u0 + 3u * rowBytes) : u0;
    QWAQSiluPair acc;
    acc.gate = float4(0.0f);
    acc.up = float4(0.0f);
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        half4 xlo[4], xhi[4];
        const float sumX = qwg_aq_load_prescaled32<BITS>(x + col, xlo, xhi);
        const size_t groupOff = size_t(col / GROUP) * groupBytes;
        const uint byteInGroup = (col % GROUP) * BITS / 8u;
        acc.gate.x += qwg_aq_slice_dot<BITS>(g0 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.gate.y += qwg_aq_slice_dot<BITS>(g1 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.gate.z += qwg_aq_slice_dot<BITS>(g2 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.gate.w += qwg_aq_slice_dot<BITS>(g3 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.up.x += qwg_aq_slice_dot<BITS>(u0 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.up.y += qwg_aq_slice_dot<BITS>(u1 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.up.z += qwg_aq_slice_dot<BITS>(u2 + groupOff, byteInGroup, xlo, xhi, sumX);
        acc.up.w += qwg_aq_slice_dot<BITS>(u3 + groupOff, byteInGroup, xlo, xhi, sumX);
    }
    acc.gate += simd_shuffle_down(acc.gate, 16);
    acc.gate += simd_shuffle_down(acc.gate, 8);
    acc.gate += simd_shuffle_down(acc.gate, 4);
    acc.gate += simd_shuffle_down(acc.gate, 2);
    acc.gate += simd_shuffle_down(acc.gate, 1);
    acc.up += simd_shuffle_down(acc.up, 16);
    acc.up += simd_shuffle_down(acc.up, 8);
    acc.up += simd_shuffle_down(acc.up, 4);
    acc.up += simd_shuffle_down(acc.up, 2);
    acc.up += simd_shuffle_down(acc.up, 1);
    return acc;
}

/// Multi-row (M = 2..4) gate/up SwiGLU over 4 pairs per simdgroup (32 intermediate channels per TG),
/// reading each weight block once across all M activation rows.
template <int BITS, int GROUP>
static inline void qwg_aq_silu_pair_rows4_m(device const char* blocks,
                                            device const half* x,
                                            device half* h16,
                                            threadgroup half* tgAct,
                                            uint I, uint K, uint mRows,
                                            bool fuseFwht,
                                            uint tg, uint sgi, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    const uint rBase = tg * 32u;
    const uint r0 = rBase + sgi * 4u;
    if (r0 < I) {
        const uint nBlocks = K >> 5u;
        const uint groupBytes = GROUP * BITS / 8 + 4u;
        const uint rowBytes = (K / GROUP) * groupBytes;
        device const char* g0 = blocks + (size_t)r0 * rowBytes;
        device const char* g1 = (r0 + 1u < I) ? (g0 + rowBytes) : g0;
        device const char* g2 = (r0 + 2u < I) ? (g0 + 2u * rowBytes) : g0;
        device const char* g3 = (r0 + 3u < I) ? (g0 + 3u * rowBytes) : g0;
        device const char* u0 = blocks + (size_t)(I + r0) * rowBytes;
        device const char* u1 = (r0 + 1u < I) ? (u0 + rowBytes) : u0;
        device const char* u2 = (r0 + 2u < I) ? (u0 + 2u * rowBytes) : u0;
        device const char* u3 = (r0 + 3u < I) ? (u0 + 3u * rowBytes) : u0;
        float4 gate[8], up[8];
        #pragma unroll
        for (uint m = 0u; m < 8u; ++m) { gate[m] = float4(0.0f); up[m] = float4(0.0f); }
        for (uint g = lane; g < nBlocks; g += 32u) {
            const uint col = g << 5u;
            const size_t groupOff = size_t(col / GROUP) * groupBytes;
            const uint byteInGroup = (col % GROUP) * BITS / 8u;
            device const char* gp0 = g0 + groupOff;
            device const char* gp1 = g1 + groupOff;
            device const char* gp2 = g2 + groupOff;
            device const char* gp3 = g3 + groupOff;
            device const char* up_p0 = u0 + groupOff;
            device const char* up_p1 = u1 + groupOff;
            device const char* up_p2 = u2 + groupOff;
            device const char* up_p3 = u3 + groupOff;
            #pragma unroll
            for (uint p = 0u; p < 8u; p += 2u) {
                if (p + 1u < mRows) {
                    half4 xlo0[4], xhi0[4], xlo1[4], xhi1[4];
                    const float2 sumX = float2(
                        qwg_aq_load_prescaled32<BITS>(x + (size_t)p * K + col, xlo0, xhi0),
                        qwg_aq_load_prescaled32<BITS>(x + (size_t)(p + 1u) * K + col, xlo1, xhi1));
                    const float2 dg0 = qwg_aq_slice_dot2<BITS>(gp0, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 dg1 = qwg_aq_slice_dot2<BITS>(gp1, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 dg2 = qwg_aq_slice_dot2<BITS>(gp2, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 dg3 = qwg_aq_slice_dot2<BITS>(gp3, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 du0 = qwg_aq_slice_dot2<BITS>(up_p0, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 du1 = qwg_aq_slice_dot2<BITS>(up_p1, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 du2 = qwg_aq_slice_dot2<BITS>(up_p2, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    const float2 du3 = qwg_aq_slice_dot2<BITS>(up_p3, byteInGroup, xlo0, xhi0, xlo1, xhi1, sumX);
                    gate[p] += float4(dg0.x, dg1.x, dg2.x, dg3.x);
                    gate[p + 1u] += float4(dg0.y, dg1.y, dg2.y, dg3.y);
                    up[p] += float4(du0.x, du1.x, du2.x, du3.x);
                    up[p + 1u] += float4(du0.y, du1.y, du2.y, du3.y);
                } else if (p < mRows) {
                    half4 xlo0[4], xhi0[4];
                    const float sumX = qwg_aq_load_prescaled32<BITS>(x + (size_t)p * K + col, xlo0, xhi0);
                    gate[p].x += qwg_aq_slice_dot<BITS>(gp0, byteInGroup, xlo0, xhi0, sumX);
                    gate[p].y += qwg_aq_slice_dot<BITS>(gp1, byteInGroup, xlo0, xhi0, sumX);
                    gate[p].z += qwg_aq_slice_dot<BITS>(gp2, byteInGroup, xlo0, xhi0, sumX);
                    gate[p].w += qwg_aq_slice_dot<BITS>(gp3, byteInGroup, xlo0, xhi0, sumX);
                    up[p].x += qwg_aq_slice_dot<BITS>(up_p0, byteInGroup, xlo0, xhi0, sumX);
                    up[p].y += qwg_aq_slice_dot<BITS>(up_p1, byteInGroup, xlo0, xhi0, sumX);
                    up[p].z += qwg_aq_slice_dot<BITS>(up_p2, byteInGroup, xlo0, xhi0, sumX);
                    up[p].w += qwg_aq_slice_dot<BITS>(up_p3, byteInGroup, xlo0, xhi0, sumX);
                }
            }
        }
        #pragma unroll
        for (uint m = 0u; m < 8u; ++m) {
            if (m < mRows) {
                gate[m] += simd_shuffle_down(gate[m], 16); up[m] += simd_shuffle_down(up[m], 16);
                gate[m] += simd_shuffle_down(gate[m], 8);  up[m] += simd_shuffle_down(up[m], 8);
                gate[m] += simd_shuffle_down(gate[m], 4);  up[m] += simd_shuffle_down(up[m], 4);
                gate[m] += simd_shuffle_down(gate[m], 2);  up[m] += simd_shuffle_down(up[m], 2);
                gate[m] += simd_shuffle_down(gate[m], 1);  up[m] += simd_shuffle_down(up[m], 1);
            }
        }
        if (lane == 0u) {
            const uint c = sgi * 4u;
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) {
                if (m < mRows) {
                    #pragma unroll
                    for (uint r = 0u; r < 4u; ++r) {
                        tgAct[m * 32u + c + r] = half(qwg_swiglu_mul(gate[m][r], up[m][r]));
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi < mRows && rBase + lane < I) {
        const half val = tgAct[sgi * 32u + lane];
        h16[(size_t)sgi * I + rBase + lane] = fuseFwht ? simd_fwht32(val, lane) : val;
    }
}

/// One row's dot: the lane's 32-column block's activations against the
/// row's weights, reduced over the simdgroup.
template <int BITS, int GROUP>
static inline float qwg_aq_silu_row1(device const char* blocks,
                                     device const half* x,
                                     uint row, uint K, uint lane) {
    const uint nBlocks = K >> 5u;
    const uint groupBytes = GROUP * BITS / 8 + 4u;
    const uint rowBytes = (K / GROUP) * groupBytes;
    device const char* rowBase = blocks + (size_t)row * rowBytes;
    float acc = 0.0f;
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        device const half* xg = x + col;
        half4 lo[4], hi[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            qwg_load_half8(xg + w * 8u, lo[w], hi[w]);
        }
        const uint groupIndex = col / GROUP;
        const uint byteInGroup = (col % GROUP) * BITS / 8u;
        acc += qwg_aq_block_dot<BITS, GROUP>(rowBase, groupIndex, byteInGroup, lo, hi);
    }
    acc += simd_shuffle_down(acc, 16);
    acc += simd_shuffle_down(acc, 8);
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    return acc;
}

/// One row a side: the gate row and the up row, the lane's block's
/// activations shared by both.
template <int BITS, int GROUP>
static inline float2 qwg_aq_silu_pair_row1(device const char* blocks,
                                           device const half* x,
                                           uint row, uint I, uint K, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    const uint nBlocks = K >> 5u;
    const uint groupBytes = GROUP * BITS / 8 + 4u;
    const uint rowBytes = (K / GROUP) * groupBytes;
    device const char* gRow = blocks + (size_t)row * rowBytes;
    device const char* uRow = blocks + (size_t)(I + row) * rowBytes;
    float gAcc = 0.0f;
    float uAcc = 0.0f;
    for (uint g = lane; g < nBlocks; g += 32u) {
        const uint col = g << 5u;
        half4 xlo[4], xhi[4];
        const float sumX = qwg_aq_load_prescaled32<BITS>(x + col, xlo, xhi);
        const size_t groupOff = size_t(col / GROUP) * groupBytes;
        const uint byteInGroup = (col % GROUP) * BITS / 8u;
        gAcc += qwg_aq_slice_dot<BITS>(gRow + groupOff, byteInGroup, xlo, xhi, sumX);
        uAcc += qwg_aq_slice_dot<BITS>(uRow + groupOff, byteInGroup, xlo, xhi, sumX);
    }
    gAcc += simd_shuffle_down(gAcc, 16);
    gAcc += simd_shuffle_down(gAcc, 8);
    gAcc += simd_shuffle_down(gAcc, 4);
    gAcc += simd_shuffle_down(gAcc, 2);
    gAcc += simd_shuffle_down(gAcc, 1);
    uAcc += simd_shuffle_down(uAcc, 16);
    uAcc += simd_shuffle_down(uAcc, 8);
    uAcc += simd_shuffle_down(uAcc, 4);
    uAcc += simd_shuffle_down(uAcc, 2);
    uAcc += simd_shuffle_down(uAcc, 1);
    return float2(gAcc, uAcc);
}

#define QW_AQ_SILU_PAIR(BITS, GROUP, NAME)                                   \
    kernel void NAME(device const char* blocks [[buffer(0)]],                 \
                     device const half* x         [[buffer(1)]],             \
                     constant uint& N             [[buffer(4)]],             \
                     constant uint& K             [[buffer(5)]],             \
                     device half* h16             [[buffer(8)]],             \
                     uint tg [[threadgroup_position_in_grid]],                \
                     uint ti [[thread_index_in_threadgroup]],                 \
                     uint tptg [[threads_per_threadgroup]]) {                 \
        if ((N & 0x40000000u) != 0u) {                                       \
            const bool fuseFwht = (N & 0x80000000u) != 0u;                   \
            const uint mRows = (N >> 24u) & 0x3Fu;                           \
            if (mRows >= 2u) {                                               \
                threadgroup half tgActM[8 * 32];                             \
                const uint realN = N & 0x00FFFFFFu;                          \
                const uint I = realN >> 1u;                                  \
                qwg_aq_silu_pair_rows4_m<BITS, GROUP>(                       \
                    blocks, x, h16, tgActM, I, K, mRows, fuseFwht,           \
                    tg, ti >> 5u, ti & 31u);                                 \
                return;                                                      \
            }                                                                \
            threadgroup half tgAct[32];                                      \
            const uint realN = N & 0x3FFFFFFFu;                              \
            const uint I = realN >> 1u;                                      \
            const uint rBase = tg * 32u;                                      \
            const uint sgi = ti >> 5u;                                        \
            const uint lane = ti & 31u;                                      \
            const uint r0 = rBase + sgi * 4u;                                 \
            if (r0 < I) {                                                     \
                const QWAQSiluPair acc = qwg_aq_silu_pair_rows4<BITS, GROUP>(  \
                    blocks, x, r0, I, K, lane);                               \
                if (lane == 0u) {                                            \
                    const uint c = sgi * 4u;                                 \
                    tgAct[c + 0u] = half(qwg_swiglu_mul(acc.gate.x, acc.up.x)); \
                    tgAct[c + 1u] = half(qwg_swiglu_mul(acc.gate.y, acc.up.y)); \
                    tgAct[c + 2u] = half(qwg_swiglu_mul(acc.gate.z, acc.up.z)); \
                    tgAct[c + 3u] = half(qwg_swiglu_mul(acc.gate.w, acc.up.w)); \
                }                                                            \
            }                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
            if (sgi == 0u && rBase + lane < I) {                             \
                const half val = tgAct[lane];                                \
                h16[rBase + lane] = fuseFwht ? simd_fwht32(val, lane) : val; \
            }                                                                \
            return;                                                          \
        }                                                                    \
        if ((N & 0x80000000u) != 0u) {                                       \
            const uint realN = N & 0x7FFFFFFFu;                              \
            const uint I = realN >> 1u;                                      \
            const uint pairs_per_tg = max(1u, tptg >> 5u);                   \
            const uint sgi = ti >> 5u;                                        \
            const uint lane = ti & 31u;                                      \
            const uint row = tg * pairs_per_tg + sgi;                        \
            if (row < I) {                                                   \
                const float2 acc = qwg_aq_silu_pair_row1<BITS, GROUP>(       \
                    blocks, x, row, I, K, lane);                              \
                if (lane == 0u) {                                            \
                    h16[row] = half(qwg_swiglu_mul(acc.x, acc.y));           \
                }                                                            \
            }                                                                \
            return;                                                          \
        }                                                                    \
        threadgroup float part[16];                                          \
        const uint realN = N & 0x3FFFFFFFu;                                  \
        const uint I = realN >> 1u;                                          \
        const uint pairs_per_tg = max(1u, tptg >> 6u);                       \
        const uint pair_id = ti >> 6u;                                       \
        const uint sub = (ti >> 5u) & 1u;                                    \
        const uint lane = ti & 31u;                                          \
        const uint row = tg * pairs_per_tg + pair_id;                        \
        if (row < I) {                                                       \
            const uint outRow = (sub == 0u) ? row : row + I;                  \
            const float v = qwg_aq_silu_row1<BITS, GROUP>(blocks, x, outRow,  \
                                                          K, lane);           \
            if (lane == 0u) {                                                \
                part[pair_id * 2u + sub] = v;                                \
            }                                                                \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
        if (sub == 0u && lane == 0u && row < I) {                             \
            const float gate = part[pair_id * 2u], up = part[pair_id * 2u + 1u]; \
            h16[row] = half(qwg_swiglu_mul(gate, up));                       \
        }                                                                    \
    }

QW_AQ_SILU_PAIR(2, 32, qwg_gemv_aq2_g32_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(2, 64, qwg_gemv_aq2_g64_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(2, 128, qwg_gemv_aq2_g128_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(2, 256, qwg_gemv_aq2_g256_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(3, 32, qwg_gemv_aq3_g32_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(3, 64, qwg_gemv_aq3_g64_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(3, 128, qwg_gemv_aq3_g128_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(3, 256, qwg_gemv_aq3_g256_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(4, 32, qwg_gemv_aq4_g32_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(4, 64, qwg_gemv_aq4_g64_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(4, 128, qwg_gemv_aq4_g128_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(4, 256, qwg_gemv_aq4_g256_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(6, 32, qwg_gemv_aq6_g32_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(6, 64, qwg_gemv_aq6_g64_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(6, 128, qwg_gemv_aq6_g128_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(6, 256, qwg_gemv_aq6_g256_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(8, 32, qwg_gemv_aq8_g32_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(8, 64, qwg_gemv_aq8_g64_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(8, 128, qwg_gemv_aq8_g128_silu_pair_dx_h16)
QW_AQ_SILU_PAIR(8, 256, qwg_gemv_aq8_g256_silu_pair_dx_h16)
