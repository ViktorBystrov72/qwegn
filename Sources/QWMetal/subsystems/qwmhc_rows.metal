// qwmhc_rows.metal — prefill mHC router-norm sharing each routerW load across R rows per TG
// (R=8 fused-merge + split-Q for C=4 when hidden % 4 == 0; R=4 for C=2 or unaligned hidden).
// Stream 0-ULP merge order preserved; 4-row parallel float4 branch norm runs C=4 in <=2 passes.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_geometry_constants.metalh"
#include "qw_mhc_common.metalh"
using namespace metal;

#define QW_MHC_ROWS_C2 4u
#define QW_MHC_ROWS_C4 8u

inline float4 qwg_fwht32_8lane_f32(float4 v, uint lane) {
    if (!QW_FWHT_ROTATED) return v;
    v = v.xxzz + v.yyww * float4(1.0f, -1.0f, 1.0f, -1.0f);
    v = v.xyxy + v.zwzw * float4(1.0f, 1.0f, -1.0f, -1.0f);
    for (ushort s = 1u; s <= 4u; s <<= 1u) {
        const float4 p = simd_shuffle_xor(v, s);
        v = ((lane & s) == 0u) ? (v + p) : (p - v);
    }
    return v * 0.17677669529663689f;
}

template <uint C>
inline void qwg_mhc_branch_norm_vec4(device const float* mhcStream,
                                     device const half*  normGamma,
                                     device half*        outNorm,
                                     uint                hidden,
                                     uint                width,
                                     uint                t0,
                                     uint                nRows,
                                     float               eps,
                                     threadgroup float*  sPre,
                                     uint                sg,
                                     uint                lane,
                                     threadgroup float*  partSS) {
    const uint r = sg >> 1u;
    const uint halfSG = sg & 1u;
    const uint gIdx = halfSG * 32u + lane;
    const uint hidden4 = hidden >> 2u;
    const uint nPerThread4 = (hidden4 + 63u) >> 6u;
    float pre[C];
    float4 regB4[16];

    if (r < nRows) {
        const uint t = t0 + r;
        device const float4* s4Base = (device const float4*)(mhcStream + (size_t)t * width);
        #pragma unroll
        for (uint c = 0u; c < C; ++c) pre[c] = sPre[r * 4u + c];
        float ssBranch = 0.0f;
        if (nPerThread4 <= 16u) {
            #pragma unroll
            for (uint i = 0u; i < 16u; ++i) {
                if (i * 64u >= hidden4) break;
                const uint e4 = i * 64u + gIdx;
                if (e4 < hidden4) {
                    float4 bv = float4(0.0f);
                    #pragma unroll
                    for (uint c = 0u; c < C; ++c) bv = fma(pre[c], s4Base[(size_t)c * hidden4 + e4], bv);
                    regB4[i] = bv;
                    ssBranch += dot(bv, bv);
                }
            }
        } else {
            for (uint e4 = gIdx; e4 < hidden4; e4 += 64u) {
                float4 bv = float4(0.0f);
                #pragma unroll
                for (uint c = 0u; c < C; ++c) bv = fma(pre[c], s4Base[(size_t)c * hidden4 + e4], bv);
                ssBranch += dot(bv, bv);
            }
        }
        ssBranch = simd_sum(ssBranch);
        if (lane == 0u) partSS[r * 2u + halfSG] = ssBranch;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (r < nRows) {
        const uint t = t0 + r;
        device const float4* s4Base = (device const float4*)(mhcStream + (size_t)t * width);
        device const half4* gamma4 = (device const half4*)normGamma;
        device half4* dst4 = (device half4*)(outNorm + (size_t)t * hidden);
        const float totBranch = partSS[r * 2u] + partSS[r * 2u + 1u];
        const float rstdBranch = rsqrt(totBranch / float(hidden) + eps);
        if (nPerThread4 <= 16u) {
            #pragma unroll
            for (uint i = 0u; i < 16u; ++i) {
                if (i * 64u >= hidden4) break;
                const uint e4 = i * 64u + gIdx;
                const bool active = e4 < hidden4;
                const float4 g4 = active ? float4(gamma4[e4]) : float4(0.0f);
                float4 v = active ? (regB4[i] * rstdBranch * (1.0f + g4)) : float4(0.0f);
                v = qwg_fwht32_8lane_f32(v, lane);
                if (active) dst4[e4] = half4(v);
            }
        } else {
            const uint lim4 = (hidden4 + 63u) & ~63u;
            for (uint e4 = gIdx; e4 < lim4; e4 += 64u) {
                const bool active = e4 < hidden4;
                float4 bv = float4(0.0f);
                if (active) {
                    #pragma unroll
                    for (uint c = 0u; c < C; ++c) bv = fma(pre[c], s4Base[(size_t)c * hidden4 + e4], bv);
                }
                const float4 g4 = active ? float4(gamma4[e4]) : float4(0.0f);
                float4 v = active ? (bv * rstdBranch * (1.0f + g4)) : float4(0.0f);
                v = qwg_fwht32_8lane_f32(v, lane);
                if (active) dst4[e4] = half4(v);
            }
        }
    }
}

// Fused-merge + split-Q path for C=4, R=8, hidden % 4 == 0: SG r merges row t0+r in 32-float4 chunks
// and pair-stages streams (0,1) then (2,3) into tgV[2][8][32] (8 KiB total TG SRAM -> 4 TGs/core,
// 128 barriers vs 256); post-loop rowTots/partSS/sPre alias tgV safely after the final loop barrier.
inline void qwg_mhc_rows4_vec4(device float*       mhcStream,
                               device const float* routerW,
                               device const float* routerB,
                               device const half*  normGamma,
                               device half*        outNorm,
                               device float*       mhcControls,
                               uint                hidden,
                               float               eps,
                               float               preScale,
                               uint                realIterations,
                               bool                hasMerge,
                               device const half*  mergeBranchOut,
                               uint                rows,
                               uint                t0,
                               uint                sg,
                               uint                lane,
                               threadgroup float4  tgV[2][8][32],
                               threadgroup float*  rowTots,
                               threadgroup float*  partSS,
                               threadgroup float*  sPre) {
    constexpr uint C = 4u, R = QW_MHC_ROWS_C4, NUM_CTRL = 20u;
    if (t0 >= rows) return;
    const uint tid = sg * 32u + lane;
    const uint nRows = min(R, rows - t0);
    const uint hidden4 = hidden >> 2u;
    const uint width = C * hidden;
    const uint width4 = C * hidden4;
    const bool rowActive = sg < nRows;

    float post[C];
    float mix[C * C];
    const size_t tRow = rowActive ? (size_t)(t0 + sg) : 0u;
    device float4* s4Row = (device float4*)(mhcStream + tRow * width);
    device const half4* bo4Row = (device const half4*)(mergeBranchOut + tRow * hidden);
    if (hasMerge && rowActive) {
        device const float* prevCtrl = mhcControls + tRow * NUM_CTRL;
        #pragma unroll
        for (uint c = 0u; c < C; ++c) post[c] = prevCtrl[c];
        #pragma unroll
        for (uint m = 0u; m < C * C; ++m) mix[m] = prevCtrl[C + m];
    }

    float ss = 0.0f;
    float acc[3][R];
    #pragma unroll
    for (uint k = 0u; k < 3u; ++k) {
        #pragma unroll
        for (uint r = 0u; r < R; ++r) acc[k][r] = 0.0f;
    }
    const uint q0 = sg * 3u;
    device const float4* w4Q0 = (device const float4*)(routerW + (size_t)q0 * width);
    device const float4* w4Q1 = w4Q0 + width4;
    device const float4* w4Q2 = w4Q1 + width4;

    for (uint base = 0u; base < hidden4; base += 32u) {
        const uint h4 = base + lane;
        const bool hActive = h4 < hidden4;
        float4 ns[C];
        if (rowActive && hActive) {
            float4 v[C];
            #pragma unroll
            for (uint c = 0u; c < C; ++c) v[c] = s4Row[(size_t)c * hidden4 + h4];
            if (hasMerge) {
                const float4 b4 = float4(bo4Row[h4]);
                #pragma unroll
                for (uint i = 0u; i < C; ++i) {
                    float4 x = post[i] * b4;
                    #pragma unroll
                    for (uint j = 0u; j < C; ++j) x = fma(mix[i * C + j], v[j], x);
                    ns[i] = x;
                    s4Row[(size_t)i * hidden4 + h4] = x;
                }
            } else {
                #pragma unroll
                for (uint i = 0u; i < C; ++i) ns[i] = v[i];
            }
            #pragma unroll
            for (uint c = 0u; c < C; ++c) ss += dot(ns[c], ns[c]);
        } else {
            #pragma unroll
            for (uint i = 0u; i < C; ++i) ns[i] = float4(0.0f);
        }

        #pragma unroll
        for (uint p = 0u; p < 2u; ++p) {
            tgV[0][sg][lane] = ns[2u * p];
            tgV[1][sg][lane] = ns[2u * p + 1u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (hActive) {
                const size_t off0 = (size_t)(2u * p) * hidden4 + h4;
                const size_t off1 = off0 + hidden4;
                const float4 w0a = w4Q0[off0], w1a = w4Q1[off0], w2a = w4Q2[off0];
                const float4 w0b = w4Q0[off1], w1b = w4Q1[off1], w2b = w4Q2[off1];
                if (nRows == R) {
                    #pragma unroll
                    for (uint r = 0u; r < R; ++r) {
                        const float4 x0 = tgV[0][r][lane];
                        const float4 x1 = tgV[1][r][lane];
                        acc[0][r] += dot(x0, w0a); acc[1][r] += dot(x0, w1a); acc[2][r] += dot(x0, w2a);
                        acc[0][r] += dot(x1, w0b); acc[1][r] += dot(x1, w1b); acc[2][r] += dot(x1, w2b);
                    }
                } else {
                    for (uint r = 0u; r < nRows; ++r) {
                        const float4 x0 = tgV[0][r][lane];
                        const float4 x1 = tgV[1][r][lane];
                        acc[0][r] += dot(x0, w0a); acc[1][r] += dot(x0, w1a); acc[2][r] += dot(x0, w2a);
                        acc[0][r] += dot(x1, w0b); acc[1][r] += dot(x1, w1b); acc[2][r] += dot(x1, w2b);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    ss = simd_sum(ss);
    if (lane == 0u && rowActive) rowTots[sg * 25u] = ss;
    #pragma unroll
    for (uint r = 0u; r < R; ++r) {
        if (r >= nRows) break;
        const float d0 = simd_sum(acc[0][r]);
        const float d1 = simd_sum(acc[1][r]);
        const float d2 = simd_sum(acc[2][r]);
        if (lane == 0u) {
            rowTots[r * 25u + 1u + q0 + 0u] = d0;
            rowTots[r * 25u + 1u + q0 + 1u] = d1;
            rowTots[r * 25u + 1u + q0 + 2u] = d2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowActive) {
        qwg_mhc_controls4_simd(rowTots + sg * 25u, width, eps, preScale, realIterations,
                               routerB, mhcControls + tRow * NUM_CTRL, sPre + sg * 4u, lane, true);
    }
    // Cross-SG visibility on merged mhcStream writes and sPre before branch norm reads them.
    if (hasMerge) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((hidden & 31u) == 0u) {
        qwg_mhc_branch_norm_vec4<4u>(mhcStream, normGamma, outNorm, hidden, width, t0, min(4u, nRows), eps,
                                     sPre, sg, lane, partSS);
        if (nRows > 4u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qwg_mhc_branch_norm_vec4<4u>(mhcStream, normGamma, outNorm, hidden, width, t0 + 4u, nRows - 4u, eps,
                                         sPre + 16u, sg, lane, partSS);
        }
        return;
    }
    for (uint r = 0u; r < nRows; ++r) {
        const uint t = t0 + r;
        qwg_mhc_branch_norm<4u>(mhcStream + (size_t)t * width, normGamma, outNorm + (size_t)t * hidden,
                                hidden, eps, sPre + r * 4u, tid, sg, lane, partSS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint C, uint R>
inline void qwg_mhc_router_norm_rows_impl(device float*       mhcStream,
                                          device const float* routerW,
                                          device const float* routerB,
                                          device const half*  normGamma,
                                          device half*        outNorm,
                                          device float*       mhcControls,
                                          uint                hidden,
                                          float               eps,
                                          float               preScale,
                                          uint                iterations,
                                          device const half*  mergeBranchOut,
                                          uint                rows,
                                          uint                t0,
                                          uint                sg,
                                          uint                lane,
                                          threadgroup float*  partSS,
                                          threadgroup float*  partDots,
                                          threadgroup float*  sPre,
                                          threadgroup float*  rowTots) {
    constexpr uint NUM_CTRL = C + C * C;
    constexpr uint NUM_ROWS = 2u * C + C * C;
    if (t0 >= rows) return;
    const uint tid = sg * 32u + lane;
    const uint nRows = min(R, rows - t0);
    const uint width = C * hidden;
    const uint width4 = width >> 2u;
    const bool hasMerge = (iterations & 0x80000000u) != 0u;
    const uint realIterations = iterations & 0x7FFFFFFFu;

    if (hasMerge) {
        const bool vec4Merge = (hidden & 3u) == 0u;
        const uint hidden4 = hidden >> 2u;
        for (uint r = 0u; r < nRows; ++r) {
            const uint t = t0 + r;
            device float* sBase = mhcStream + (size_t)t * width;
            device const half* bo = mergeBranchOut + (size_t)t * hidden;
            device const float* prevCtrl = mhcControls + (size_t)t * NUM_CTRL;
            float post[C];
            float mix[C * C];
            #pragma unroll
            for (uint c = 0u; c < C; ++c) post[c] = prevCtrl[c];
            #pragma unroll
            for (uint m = 0u; m < C * C; ++m) mix[m] = prevCtrl[C + m];
            if (vec4Merge) {
                device float4* s4Base = (device float4*)sBase;
                device const half4* bo4 = (device const half4*)bo;
                for (uint h4 = tid; h4 < hidden4; h4 += 256u) {
                    float4 v[C];
                    #pragma unroll
                    for (uint c = 0u; c < C; ++c) v[c] = s4Base[(size_t)c * hidden4 + h4];
                    const float4 b4 = float4(bo4[h4]);
                    #pragma unroll
                    for (uint i = 0u; i < C; ++i) {
                        float4 ns = post[i] * b4;
                        #pragma unroll
                        for (uint j = 0u; j < C; ++j) ns = fma(mix[i * C + j], v[j], ns);
                        s4Base[(size_t)i * hidden4 + h4] = ns;
                    }
                }
            } else {
                for (uint h = tid; h < hidden; h += 256u) {
                    float v[C];
                    #pragma unroll
                    for (uint c = 0u; c < C; ++c) v[c] = sBase[c * hidden + h];
                    const float b = float(bo[h]);
                    #pragma unroll
                    for (uint i = 0u; i < C; ++i) {
                        float ns = post[i] * b;
                        #pragma unroll
                        for (uint j = 0u; j < C; ++j) ns = fma(mix[i * C + j], v[j], ns);
                        sBase[i * hidden + h] = ns;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float ss[R];
    float acc[R][NUM_ROWS];
    #pragma unroll
    for (uint r = 0u; r < R; ++r) {
        ss[r] = 0.0f;
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) acc[r][q] = 0.0f;
    }
    if ((width & 3u) == 0u) {
        device const float4* s4 = (device const float4*)(mhcStream + (size_t)t0 * width);
        for (uint k4 = tid; k4 < width4; k4 += 256u) {
            float4 v[R];
            #pragma unroll
            for (uint r = 0u; r < R; ++r) {
                v[r] = (r < nRows) ? s4[(size_t)r * width4 + k4] : float4(0.0f);
                ss[r] += dot(v[r], v[r]);
            }
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) {
                const float4 w = ((device const float4*)(routerW + (size_t)q * width))[k4];
                #pragma unroll
                for (uint r = 0u; r < R; ++r) acc[r][q] += dot(v[r], w);
            }
        }
    } else {
        device const float* s1 = mhcStream + (size_t)t0 * width;
        for (uint k = tid; k < width; k += 256u) {
            float v[R];
            #pragma unroll
            for (uint r = 0u; r < R; ++r) {
                v[r] = (r < nRows) ? s1[(size_t)r * width + k] : 0.0f;
                ss[r] = fma(v[r], v[r], ss[r]);
            }
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) {
                const float w = routerW[(size_t)q * width + k];
                #pragma unroll
                for (uint r = 0u; r < R; ++r) acc[r][q] = fma(v[r], w, acc[r][q]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < R; ++r) {
        ss[r] = simd_sum(ss[r]);
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) acc[r][q] = simd_sum(acc[r][q]);
    }
    if (lane == 0u) {
        #pragma unroll
        for (uint r = 0u; r < R; ++r) {
            partSS[r * 8u + sg] = ss[r];
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) partDots[(r * 8u + sg) * NUM_ROWS + q] = acc[r][q];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup r runs row r's controls in parallel across lanes/rows.
    if constexpr (C == 2u) {
        if (sg < nRows) {
            const uint r = sg;
            threadgroup float* tots = rowTots + r * 9u;
            if (lane < 9u) {
                float a = 0.0f;
                if (lane == 0u) {
                    for (uint i = 0u; i < 8u; ++i) a += partSS[r * 8u + i];
                } else {
                    const uint q = lane - 1u;
                    for (uint i = 0u; i < 8u; ++i) a += partDots[(r * 8u + i) * NUM_ROWS + q];
                }
                tots[lane] = a;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            qwg_mhc_controls2_simd(tots, width, eps, preScale, realIterations, routerB,
                                   mhcControls + (size_t)(t0 + r) * NUM_CTRL, sPre + r * 4u, lane, true);
        }
    } else if (lane == 0u && sg < nRows) {
        const uint r = sg;
        float totSS = 0.0f;
        float tot[NUM_ROWS];
        #pragma unroll
        for (uint q = 0u; q < NUM_ROWS; ++q) tot[q] = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            totSS += partSS[r * 8u + i];
            #pragma unroll
            for (uint q = 0u; q < NUM_ROWS; ++q) tot[q] += partDots[(r * 8u + i) * NUM_ROWS + q];
        }
        qwg_mhc_controls<C>(tot, totSS, width, eps, preScale, realIterations, routerB,
                            mhcControls + (size_t)(t0 + r) * NUM_CTRL, sPre + r * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < nRows; ++r) {
        const uint t = t0 + r;
        qwg_mhc_branch_norm<C>(mhcStream + (size_t)t * width, normGamma, outNorm + (size_t)t * hidden,
                               hidden, eps, sPre + r * 4u, tid, sg, lane, partSS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Grid: ceil(rows / 8) for C=4, ceil(rows / 4) for C=2; 256 threads per threadgroup.
kernel void qwg_mhc_router_norm_rows_f16(device float*       mhcStream      [[buffer(0)]],
                                         device const float* routerW        [[buffer(1)]],
                                         device const float* routerB        [[buffer(2)]],
                                         device const half*  normGamma      [[buffer(3)]],
                                         device half*        outNorm        [[buffer(4)]],
                                         device float*       mhcControls    [[buffer(5)]],
                                         constant uint&      hidden         [[buffer(6)]],
                                         constant float&     eps            [[buffer(7)]],
                                         constant float&     preScale       [[buffer(8)]],
                                         constant uint&      iterations     [[buffer(9)]],
                                         device const half*  mergeBranchOut [[buffer(10)]],
                                         constant uint&      rows           [[buffer(11)]],
                                         uint tg   [[threadgroup_position_in_grid]],
                                         uint sg   [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup union {
        float4 tgV[2][8][32];
        struct {
            float rowTots[8u * 25u];
            float partSS[8u];
            float sPre[8u * 4u];
        } post;
        struct {
            float partSS[QW_MHC_ROWS_C2 * 8u];
            float partDots[QW_MHC_ROWS_C2 * 8u * 24u];
            float sPre[QW_MHC_ROWS_C2 * 4u];
            float rowTots[QW_MHC_ROWS_C2 * 9u];
        } r4;
    } smem;
    if (QW_MHC_STREAMS == 4u) {
        if ((hidden & 3u) == 0u) {
            qwg_mhc_rows4_vec4(mhcStream, routerW, routerB, normGamma, outNorm,
                               mhcControls, hidden, eps, preScale,
                               iterations & 0x7FFFFFFFu, (iterations & 0x80000000u) != 0u,
                               mergeBranchOut, rows, tg * QW_MHC_ROWS_C4, sg, lane,
                               smem.tgV, smem.post.rowTots, smem.post.partSS, smem.post.sPre);
        } else {
            const uint t0 = tg * QW_MHC_ROWS_C4;
            for (uint sub = 0u; sub < 2u; ++sub) {
                const uint st = t0 + sub * QW_MHC_ROWS_C2;
                if (st >= rows) break;
                qwg_mhc_router_norm_rows_impl<4u, QW_MHC_ROWS_C2>(
                    mhcStream, routerW, routerB, normGamma, outNorm, mhcControls,
                    hidden, eps, preScale, iterations, mergeBranchOut, rows, st, sg, lane,
                    smem.r4.partSS, smem.r4.partDots, smem.r4.sPre, smem.r4.rowTots);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        qwg_mhc_router_norm_rows_impl<2u, QW_MHC_ROWS_C2>(
            mhcStream, routerW, routerB, normGamma, outNorm, mhcControls,
            hidden, eps, preScale, iterations, mergeBranchOut, rows, tg * QW_MHC_ROWS_C2, sg, lane,
            smem.r4.partSS, smem.r4.partDots, smem.r4.sPre, smem.r4.rowTots);
    }
}
