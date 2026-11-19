// qwattn_hca_out.metal — HCA output epilogue GPU kernels:
// fused 8x8 MMA VHA head mixing + GPT-J inverse YaRN RoPE, and 3D-batched
// grouped low-rank wo_a projection (GEMV decode/verify + 8x8 MMA prefill)
// with fused sigmoid(o_gate) and H_32 exit rotation (<= 6 KiB constant LUTs).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "../include/qw_mla_dequant.metalh"
using namespace metal;
using namespace mlx::steel;

struct HCAVhaInvRopeParams {
    uint T;
    uint pos0;
    uint qHeads;
    uint headDim;
    uint nopeDim;
    uint ropeDim;
    float invMscale;
    uint hasVHA;
};

struct HCAGroupedWoAParams {
    uint T;
    uint oGroups;
    uint oLoraRank;
    uint groupIn;
    uint qUpStride;
    uint oGateOffset;
    uint bits;
    uint group;
    uint isIncumbent;
};

// 5. Fused 8x8 MMA VHA head mixing + GPT-J Inverse YaRN RoPE + optional H_32 -> zVhaInv.
// Grid: (headDim / 32, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_hca_vha_invrope(
    device const half*         attIn   [[buffer(0)]],
    device const half*         vhaMix  [[buffer(1)]],
    device const float*        invFreq [[buffer(2)]],
    device half*               zOut    [[buffer(3)]],
    constant HCAVhaInvRopeParams& p    [[buffer(4)]],
    uint2                      tg      [[threadgroup_position_in_grid]],
    uint                       tid     [[thread_index_in_threadgroup]],
    uint                       sg      [[simdgroup_index_in_threadgroup]],
    uint                       lane    [[thread_index_in_simdgroup]]
) {
    const uint dBase = tg.x * 32u;
    const uint t = tg.y;
    if (t >= p.T || dBase >= p.headDim) return;

    constexpr uint YS = 40u;
    constexpr uint OS = 36u;
    threadgroup half  shY[64u * YS];
    threadgroup float shOut[64u * OS];

    const uint pos = p.pos0 + t;
    const uint rowStride = p.qHeads * p.headDim;
    device const half* yRow = attIn + (size_t)t * rowStride;
    device half* dstRow = zOut + (size_t)t * rowStride;
    const bool isRopeSlice = (dBase >= p.nopeDim);
    const bool hasVHA = (p.hasVHA & 1u) != 0u;
    const bool skipFwht = (p.hasVHA & 0x10000u) != 0u;

    if (hasVHA && (p.qHeads & 7u) == 0u && p.qHeads <= 64u) {
        for (uint i = tid; i < p.qHeads * 8u; i += 256u) {
            const uint g = i >> 3u, c4 = (i & 7u) * 4u;
            *(threadgroup half4*)(shY + g * YS + c4) = *(device const half4*)(yRow + g * p.headDim + dBase + c4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hBase = sg * 8u;
        if (hBase < p.qHeads) {
            simdgroup_float8x8 acc[4] = {
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f),
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)
            };
            for (uint g0 = 0u; g0 < p.qHeads; g0 += 8u) {
                simdgroup_half8x8 mf;
                simdgroup_load(mf, vhaMix + hBase * p.qHeads + g0, p.qHeads);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_half8x8 yf;
                    simdgroup_load(yf, shY + g0 * YS + j * 8u, YS);
                    simdgroup_multiply_accumulate(acc[j], mf, yf, acc[j]);
                }
            }
            threadgroup float* sDst = shOut + hBase * OS;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[j], sDst + j * 8u, OS);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hr = 0u; hr < 8u; ++hr) {
                const uint h = hBase + hr;
                float val = sDst[hr * OS + lane];
                if (isRopeSlice) {
                    const uint pairIdx = ((dBase - p.nopeDim) + lane) >> 1u;
                    const float ang = float(pos) * invFreq[pairIdx];
                    const float c = cos(ang) * p.invMscale, s = sin(ang) * p.invMscale;
                    const float mate = simd_shuffle_xor(val, ushort(1));
                    val = ((lane & 1u) == 0u) ? (val * c + mate * s) : (val * c - mate * s);
                }
                dstRow[h * p.headDim + dBase + lane] = half(skipFwht ? val : qwg_fwht32_lane_f32(val, lane));
            }
        }
        return;
    }

    for (uint h = sg; h < p.qHeads; h += 8u) {
        float val = 0.0f;
        if (hasVHA) {
            for (uint g = 0u; g < p.qHeads; ++g) {
                val = fma(float(vhaMix[h * p.qHeads + g]), float(yRow[g * p.headDim + dBase + lane]), val);
            }
        } else {
            val = float(yRow[h * p.headDim + dBase + lane]);
        }
        if (isRopeSlice) {
            const uint pairIdx = ((dBase - p.nopeDim) + lane) >> 1u;
            const float ang = float(pos) * invFreq[pairIdx];
            const float c = cos(ang) * p.invMscale, s = sin(ang) * p.invMscale;
            const float mate = simd_shuffle_xor(val, ushort(1));
            val = ((lane & 1u) == 0u) ? (val * c + mate * s) : (val * c - mate * s);
        }
        dstRow[h * p.headDim + dBase + lane] = half(skipFwht ? val : qwg_fwht32_lane_f32(val, lane));
    }
}

template <bool SUB2 = false>
METAL_FUNC void qw_hca_unpack_wo_a_32(device const char* woABlocks,
                                      uint row, uint col0,
                                      constant HCAGroupedWoAParams& p,
                                      threadgroup const ShvqMlaTgLut& tgLut,
                                      threadgroup half* dst, uint dstStride) {
    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 3u) {
        device const half* hSrc = (device const half*)woABlocks + (size_t)row * p.groupIn + col0;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) dst[i * dstStride] = hSrc[i];
        return;
    }
    MLAAbsorbParams mp = {p.T, p.oGroups, 0u, p.oLoraRank, p.groupIn, 0u, p.bits, p.group, inc};
    qw_mla_unpack_slice32<SUB2>(woABlocks, row, col0, mp, tgLut, dst, dstStride);
}

template <bool SUB2 = false>
METAL_FUNC void qw_hca_unpack_wo_a_reg32(device const char* woABlocks,
                                         uint row, uint col0,
                                         constant HCAGroupedWoAParams& p,
                                         threadgroup const ShvqMlaTgLut& tgLut,
                                         thread float4 (&wLo)[4],
                                         thread float4 (&wHi)[4]) {
    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 3u) {
        device const half* hSrc = (device const half*)woABlocks + (size_t)row * p.groupIn + col0;
        #pragma unroll
        for (uint w = 0u; w < 4u; ++w) {
            wLo[w] = float4(*(device const half4*)(hSrc + w * 8u));
            wHi[w] = float4(*(device const half4*)(hSrc + w * 8u + 4u));
        }
        return;
    }
    MLAAbsorbParams mp = {p.T, p.oGroups, 0u, p.oLoraRank, p.groupIn, 0u, p.bits, p.group, inc};
    qw_mla_unpack_slice32_reg<SUB2>(woABlocks, row, col0, mp, tgLut, wLo, wHi);
}

METAL_FUNC void qw_hca_accum32_row(thread float& acc,
                                   thread const float4 (&wLo)[4],
                                   thread const float4 (&wHi)[4],
                                   threadgroup const half* xPtr) {
    #pragma unroll
    for (uint w = 0u; w < 4u; ++w) {
        const float4 x0 = float4(*(threadgroup const half4*)(xPtr + w * 8u));
        const float4 x1 = float4(*(threadgroup const half4*)(xPtr + w * 8u + 4u));
        acc = precise::fma(wLo[w].x, x0.x, acc);
        acc = precise::fma(wLo[w].y, x0.y, acc);
        acc = precise::fma(wLo[w].z, x0.z, acc);
        acc = precise::fma(wLo[w].w, x0.w, acc);
        acc = precise::fma(wHi[w].x, x1.x, acc);
        acc = precise::fma(wHi[w].y, x1.y, acc);
        acc = precise::fma(wHi[w].z, x1.z, acc);
        acc = precise::fma(wHi[w].w, x1.w, acc);
    }
}

METAL_FUNC void qw_hca_reduce_t(const thread float (&acc)[4],
                                threadgroup float* shOutRow,
                                uint sg,
                                uint lane) {
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        const float sum = simd_sum(acc[r]);
        if (lane == 0u) shOutRow[sg * 4u + r] = sum;
    }
}

// 6a. 3D-batched grouped wo_a GEMV + sigmoid(o_gate) + H_32 exit (decode/verify T <= 15).
// Grid: (oLoraRank / 32, oGroups, T), Threadgroup: (256, 1, 1).
struct HCAWoAGemvTg {
    half shX[1024];
    float shOut[32];
    ShvqMlaTgLut lut;
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_gemv_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoAGemvTg&   st,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint cBase = tg.x * 32u;
    const uint g = tg.y;
    const uint t = tg.z;
    if (t >= p.T || g >= p.oGroups || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(st.lut, p.bits, tid, 256u);
    }

    const uint baseRow = g * p.oLoraRank + cBase + sg * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kBase = 0u; kBase < p.groupIn; kBase += 1024u) {
        const uint chunkCols = min(1024u, p.groupIn - kBase);
        const uint vecCols = chunkCols >> 2u;
        device const half4* src4 = (device const half4*)(
            zIn + ((size_t)t * p.oGroups + g) * p.groupIn + kBase);
        for (uint i = tid; i < vecCols; i += 256u) {
            *(threadgroup half4*)(st.shX + i * 4u) = src4[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint colOff = lane * 32u;
        if (colOff < chunkCols) {
            const uint col0 = kBase + colOff;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                float4 wLo[4], wHi[4];
                qw_hca_unpack_wo_a_reg32<SUB2>(woABlocks, baseRow + r, col0, p, st.lut, wLo, wHi);
                qw_hca_accum32_row(acc[r], wLo, wHi, st.shX + colOff);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        const float sum = simd_sum(acc[r]);
        if (lane == 0u) st.shOut[sg * 4u + r] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        const uint globalCol = g * p.oLoraRank + cBase + lane;
        const float gate = float(qUpGate[(size_t)t * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = st.shOut[lane] / (1.0f + exp(-gate));
        zGatedOut[(size_t)t * (p.oGroups * p.oLoraRank) + globalCol] = half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_gemv(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvTg st;
    qw_hca_grouped_wo_a_gemv_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_gemv_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvTg st;
    qw_hca_grouped_wo_a_gemv_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

// 6b. 3D-batched grouped wo_a 8x8 MMA + sigmoid(o_gate) + H_32 exit (prefill T >= 16).
// Grid: (oGroups, (T + 31) / 32, oLoraRank / 128), Threadgroup: (128, 1, 1).
#define HCA_WOA_TM 32u
#define HCA_WOA_TN 128u
#define HCA_WOA_TK 32u

union HCAWoATgStage {
    struct {
        half us[HCA_WOA_TM * (HCA_WOA_TK + 8u)];
        half ws[HCA_WOA_TK * (HCA_WOA_TN + 8u)];
    } tile;
    float os[HCA_WOA_TM * (HCA_WOA_TN + 4u)];
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_mma_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoATgStage&  st,
    threadgroup ShvqMlaTgLut&   tgLut,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint g = tg.x;
    const uint tBase = tg.y * HCA_WOA_TM;
    const uint cBase = tg.z * HCA_WOA_TN;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint usStride = HCA_WOA_TK + 8u, wsStride = HCA_WOA_TN + 8u, osStride = HCA_WOA_TN + 4u;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }

    const uint wRow = g * p.oLoraRank + cBase + tid;
    const uint uRow = tid >> 2u, uCol = (tid & 3u) * 8u;
    const bool uValid = tBase + uRow < p.T;
    device const half* uSrc = zIn + ((size_t)(tBase + uRow) * p.oGroups + g) * p.groupIn + uCol;

    for (uint k0 = 0u; k0 < p.groupIn; k0 += HCA_WOA_TK) {
        half4 a = half4(0.0h), b = half4(0.0h);
        if (uValid) {
            a = *(device const half4*)(uSrc + k0);
            b = *(device const half4*)(uSrc + k0 + 4u);
        }
        *(threadgroup half4*)(st.tile.us + uRow * usStride + uCol) = a;
        *(threadgroup half4*)(st.tile.us + uRow * usStride + uCol + 4u) = b;
        qw_hca_unpack_wo_a_32<SUB2>(woABlocks, wRow, k0, p, tgLut, st.tile.ws + tid, wsStride);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < HCA_WOA_TK; kk += 8u) {
            simdgroup_half8x8 af[4], bf[4];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(af[i], st.tile.us + i * 8u * usStride + kk, usStride);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(bf[j], st.tile.ws + kk * wsStride + sg * 32u + j * 8u, wsStride);
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[i][j], st.os + i * 8u * osStride + sg * 32u + j * 8u, osStride);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint outStride = p.oGroups * p.oLoraRank;
    const uint rows = min(HCA_WOA_TM, p.T - tBase);
    const uint globalCol = g * p.oLoraRank + cBase + sg * 32u + lane;
    for (uint m = 0u; m < rows; ++m) {
        const float raw = st.os[m * osStride + sg * 32u + lane];
        const float gate = float(qUpGate[(size_t)(tBase + m) * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = raw / (1.0f + exp(-gate));
        zGatedOut[(size_t)(tBase + m) * outStride + globalCol] = half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_mma(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoATgStage st;
    threadgroup ShvqMlaTgLut tgLut;
    qw_hca_grouped_wo_a_mma_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tgLut, tg, tid, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_mma_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoATgStage st;
    threadgroup ShvqMlaTgLut tgLut;
    qw_hca_grouped_wo_a_mma_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tgLut, tg, tid, sg, lane);
}

// 6c. Multi-row grouped wo_a GEMV (2 <= T <= 8 verify) reusing woABlocks across all T tokens.
// Grid: (oLoraRank / 32, oGroups, 1), Threadgroup: (256, 1, 1).
struct HCAWoAGemvMRTg {
    half shX[8][1024];
    float shOut[8][32];
    ShvqMlaTgLut lut;
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_gemv_mr_body(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup HCAWoAGemvMRTg& st,
    uint3                       tg,
    uint                        tid,
    uint                        sg,
    uint                        lane
) {
    const uint cBase = tg.x * 32u;
    const uint g = tg.y;
    const uint rowsT = min(p.T, 8u);
    if (rowsT == 0u || g >= p.oGroups || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    if (inc == 2u) {
        qw_mla_load_shvq_lut<SUB2>(st.lut, p.bits, tid, 256u);
    }

    const uint baseRow = g * p.oLoraRank + cBase + sg * 4u;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc5[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc6[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc7[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint kBase = 0u; kBase < p.groupIn; kBase += 1024u) {
        const uint chunkCols = min(1024u, p.groupIn - kBase);
        const uint vecCols = chunkCols >> 2u;
        for (uint t = 0u; t < rowsT; ++t) {
            device const half4* src4 = (device const half4*)(
                zIn + ((size_t)t * p.oGroups + g) * p.groupIn + kBase);
            for (uint i = tid; i < vecCols; i += 256u) {
                *(threadgroup half4*)(st.shX[t] + i * 4u) = src4[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint colOff = lane * 32u;
        if (colOff < chunkCols) {
            const uint col0 = kBase + colOff;
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                float4 wLo[4], wHi[4];
                qw_hca_unpack_wo_a_reg32<SUB2>(woABlocks, baseRow + r, col0, p, st.lut, wLo, wHi);
                qw_hca_accum32_row(acc0[r], wLo, wHi, st.shX[0] + colOff);
                if (rowsT > 1u) qw_hca_accum32_row(acc1[r], wLo, wHi, st.shX[1] + colOff);
                if (rowsT > 2u) qw_hca_accum32_row(acc2[r], wLo, wHi, st.shX[2] + colOff);
                if (rowsT > 3u) qw_hca_accum32_row(acc3[r], wLo, wHi, st.shX[3] + colOff);
                if (rowsT > 4u) qw_hca_accum32_row(acc4[r], wLo, wHi, st.shX[4] + colOff);
                if (rowsT > 5u) qw_hca_accum32_row(acc5[r], wLo, wHi, st.shX[5] + colOff);
                if (rowsT > 6u) qw_hca_accum32_row(acc6[r], wLo, wHi, st.shX[6] + colOff);
                if (rowsT > 7u) qw_hca_accum32_row(acc7[r], wLo, wHi, st.shX[7] + colOff);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    qw_hca_reduce_t(acc0, st.shOut[0], sg, lane);
    if (rowsT > 1u) qw_hca_reduce_t(acc1, st.shOut[1], sg, lane);
    if (rowsT > 2u) qw_hca_reduce_t(acc2, st.shOut[2], sg, lane);
    if (rowsT > 3u) qw_hca_reduce_t(acc3, st.shOut[3], sg, lane);
    if (rowsT > 4u) qw_hca_reduce_t(acc4, st.shOut[4], sg, lane);
    if (rowsT > 5u) qw_hca_reduce_t(acc5, st.shOut[5], sg, lane);
    if (rowsT > 6u) qw_hca_reduce_t(acc6, st.shOut[6], sg, lane);
    if (rowsT > 7u) qw_hca_reduce_t(acc7, st.shOut[7], sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < rowsT) {
        const uint outStride = p.oGroups * p.oLoraRank;
        const uint globalCol = g * p.oLoraRank + cBase + lane;
        const float gate = float(qUpGate[(size_t)sg * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = st.shOut[sg][lane] / (1.0f + exp(-gate));
        zGatedOut[(size_t)sg * outStride + globalCol] =
            half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

kernel void qwg_hca_grouped_wo_a_gemv_mr(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvMRTg st;
    qw_hca_grouped_wo_a_gemv_mr_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_gemv_mr_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoAGemvMRTg st;
    qw_hca_grouped_wo_a_gemv_mr_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

// 6d. Apple10+ NAX 3D-batched grouped wo_a + sigmoid(o_gate) + H_32 exit (T >= 16).
#define HCA_WOA_NAX_OS 32u

template <int MODE, bool SUB2 = false>
METAL_FUNC void qw_hca_wo_a_nax_run(
    device const half*          zIn,
    device const char*          woABlocks,
    device const half*          qUpGate,
    device half*                zGatedOut,
    constant HCAGroupedWoAParams& p,
    threadgroup float*          osBase,
    threadgroup const ShvqMlaTgLut& tgLut,
    uint                        g,
    uint                        warpT0,
    uint                        warpC0,
    int                         sgpSm,
    uint                        sg,
    uint                        lane
) {
    NAXTile<float, 2, 2> Dtile;
    Dtile.clear();

    const short2 sc = BaseNAXFrag::get_coord();
    const uint gShift = ctz(p.group);
    const uint gMask = p.group - 1u;
    const uint nGroups = p.groupIn >> gShift;

    size_t rowBytes = 0;
    uint groupBytes = 0;
    uint sliceBytes = 0;
    if constexpr (MODE == 3) {
        rowBytes = (size_t)p.groupIn * sizeof(half);
    } else if constexpr (MODE == 1) {
        rowBytes = (size_t)(p.groupIn >> 5u) * sizeof(Block_HQ4_G32);
    } else if constexpr (MODE == 2) {
        groupBytes = qwg_shvq_group_bytes(int(p.bits), int(p.group));
        sliceBytes = qwg_shvq_slice_bytes(int(p.bits));
        rowBytes = (size_t)nGroups * groupBytes;
    } else {
        groupBytes = 4u + ((p.group * p.bits) >> 3u);
        rowBytes = (size_t)nGroups * groupBytes;
    }

    device const char* wRowPtr[4];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const uint wRow = g * p.oLoraRank + warpC0 + uint(r * 16 + int(sc.y) + i * 8);
            wRowPtr[r * 2 + i] = woABlocks + (size_t)wRow * rowBytes;
        }
    }

    const int ldA = int(p.oGroups * p.groupIn);
    device const half* aBase = zIn + ((size_t)warpT0 * p.oGroups + g) * p.groupIn;
    const ushort wSub = ushort(sc.x >> 3);
    const ushort wShift = ushort((sc.x & 4) << 1);
    const bool alignedM = (sgpSm == 32);

    STEEL_PRAGMA_NO_UNROLL
    for (uint k0 = 0u; k0 < p.groupIn; k0 += 32u) {
        NAXTile<half, 2, 2> Atile;
        NAXTile<half, 2, 2> Btile;

        if (alignedM) {
            Atile.load(aBase + k0, ldA);
        } else {
            Atile.load_safe(aBase + k0, ldA, short2(32, short(sgpSm)));
        }

        const uint gIdx = k0 >> gShift;
        const uint inGroup = k0 & gMask;
        const uint payOff = (MODE == 2) ? ((inGroup >> 5u) * sliceBytes) : 0u;
        const uint q0 = uint(sc.x) >> 2u;
        const uint q1 = (uint(sc.x) + 16u) >> 2u;

        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
            thread auto& f0 = Btile.frag_at(r, 0);
            thread auto& f1 = Btile.frag_at(r, 1);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
                device const char* rPtr = wRowPtr[r * 2 + i];
                half4 vLo, vHi;
                if constexpr (MODE == 3) {
                    device const half* hPtr = (device const half*)rPtr + k0 + uint(sc.x);
                    vLo = *(device const half4*)hPtr;
                    vHi = *(device const half4*)(hPtr + 16u);
                } else if constexpr (MODE == 1) {
                    const device Block_HQ4_G32& blk = ((const device Block_HQ4_G32*)rPtr)[k0 >> 5u];
                    const half2 dm = *(const device half2*)(&blk);
                    const half2 s2 = half2(dm.x), b2 = half2(dm.y);
                    const ushort2 raw0 = as_type<ushort2>(blk.qs[wSub]) >> wShift;
                    const ushort2 raw1 = as_type<ushort2>(blk.qs[2u + wSub]) >> wShift;
                    const half2 w01 = fma(as_type<half2>((raw0 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                    const half2 w23 = fma(as_type<half2>(((raw0 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                    const half2 w45 = fma(as_type<half2>((raw1 & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                    const half2 w67 = fma(as_type<half2>(((raw1 >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)), s2, b2);
                    vLo = half4(w01.x, w01.y, w23.x, w23.y);
                    vHi = half4(w45.x, w45.y, w67.x, w67.y);
                } else if constexpr (MODE == 2) {
                    device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                    const half shvqScale = *(device const half*)gp;
                    device const uchar* pay = (device const uchar*)(gp + 2u + payOff);
                    vLo = qw_mla_shvq_quad4<SUB2>(pay, q0, p.bits, tgLut) * shvqScale;
                    vHi = qw_mla_shvq_quad4<SUB2>(pay, q1, p.bits, tgLut) * shvqScale;
                } else if constexpr (MODE == 8) {
                    device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                    const half2 sb = *(device const half2*)gp;
                    const half2 s2 = half2(sb.x), b2 = half2(sb.y), kBias = half2(-1024.0h);
                    device const uchar* pay = (device const uchar*)(gp + 4u);
                    const uint byteOff = inGroup + uint(sc.x);
                    const ushort2 rLo = as_type<ushort2>(*(device const uint*)(pay + byteOff));
                    const ushort2 rHi = as_type<ushort2>(*(device const uint*)(pay + byteOff + 16u));
                    const half2 cLoXZ = as_type<half2>((rLo & ushort2(0x00FFu)) | ushort2(0x6400u)) + kBias;
                    const half2 cLoYW = as_type<half2>((rLo >> 8u) | ushort2(0x6400u)) + kBias;
                    const half2 cHiXZ = as_type<half2>((rHi & ushort2(0x00FFu)) | ushort2(0x6400u)) + kBias;
                    const half2 cHiYW = as_type<half2>((rHi >> 8u) | ushort2(0x6400u)) + kBias;
                    const half2 wLoXZ = fma(cLoXZ, s2, b2), wLoYW = fma(cLoYW, s2, b2);
                    const half2 wHiXZ = fma(cHiXZ, s2, b2), wHiYW = fma(cHiYW, s2, b2);
                    vLo = half4(wLoXZ.x, wLoYW.x, wLoXZ.y, wLoYW.y);
                    vHi = half4(wHiXZ.x, wHiYW.x, wHiXZ.y, wHiYW.y);
                } else if constexpr (MODE == 4) {
                    device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                    const half2 sb = *(device const half2*)gp;
                    const half2 s2 = half2(sb.x), b2 = half2(sb.y), kBias = half2(-1024.0h);
                    device const uchar* pay = (device const uchar*)(gp + 4u);
                    const uint byteOff = (inGroup + uint(sc.x)) >> 1u;
                    const ushort2 raw = ushort2(*(device const ushort*)(pay + byteOff),
                                                *(device const ushort*)(pay + byteOff + 8u));
                    const half2 w0 = fma(as_type<half2>((raw & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                    const half2 w1 = fma(as_type<half2>(((raw >> 4u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                    const half2 w2 = fma(as_type<half2>(((raw >> 8u) & ushort2(0x000Fu)) | ushort2(0x6400u)) + kBias, s2, b2);
                    const half2 w3 = fma(as_type<half2>((raw >> 12u) | ushort2(0x6400u)) + kBias, s2, b2);
                    vLo = half4(w0.x, w1.x, w2.x, w3.x);
                    vHi = half4(w0.y, w1.y, w2.y, w3.y);
                } else {
                    device const char* gp = rPtr + (size_t)gIdx * groupBytes;
                    const float scAQ = float(*(device const half*)gp);
                    const float biAQ = float(*(device const half*)(gp + 2u));
                    device const uchar* pay = (device const uchar*)(gp + 4u);
                    const int bitsI = int(p.bits);
                    const int b0 = int(inGroup + uint(sc.x));
                    vLo = half4(
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 0)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 1)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 2)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 3)), scAQ, biAQ)));
                    vHi = half4(
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 16)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 17)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 18)), scAQ, biAQ)),
                        half(fma(float(qwg_aq_read_bits(pay, bitsI, b0 + 19)), scAQ, biAQ)));
                }
                f0[i * 4 + 0] = vLo.x; f0[i * 4 + 1] = vLo.y; f0[i * 4 + 2] = vLo.z; f0[i * 4 + 3] = vLo.w;
                f1[i * 4 + 0] = vHi.x; f1[i * 4 + 1] = vHi.y; f1[i * 4 + 2] = vHi.z; f1[i * 4 + 3] = vHi.w;
            }
        }

        tile_matmad_nax(Dtile, Atile, metal::false_type{}, Btile, metal::true_type{});
    }

    if constexpr (MODE == 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* osWarp = osBase + sg * (32u * HCA_WOA_NAX_OS);
    Dtile.store<float, HCA_WOA_NAX_OS, 1>(osWarp);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const bool skipFwht = (p.isIncumbent & 0x10000u) != 0u;
    const uint validRows = uint(sgpSm);
    const uint outStride = p.oGroups * p.oLoraRank;
    const uint globalCol = g * p.oLoraRank + warpC0 + lane;
    for (uint m = 0u; m < validRows; ++m) {
        const float raw = osWarp[m * HCA_WOA_NAX_OS + lane];
        const float gate = float(qUpGate[(size_t)(warpT0 + m) * p.qUpStride + p.oGateOffset + globalCol]);
        const float gated = raw / (1.0f + exp(-gate));
        zGatedOut[(size_t)(warpT0 + m) * outStride + globalCol] =
            half(skipFwht ? gated : qwg_fwht32_lane_f32(gated, lane));
    }
}

union HCAWoANaxTg {
    ShvqMlaTgLut tgLut;
    float os[4][32u * HCA_WOA_NAX_OS];
};

template <bool SUB2>
METAL_FUNC void qw_hca_grouped_wo_a_nax_body(
    device const half* zIn, device const char* woABlocks,
    device const half* qUpGate, device half* zGatedOut,
    constant HCAGroupedWoAParams& p, threadgroup HCAWoANaxTg& st,
    uint3 tg, uint tid, uint sg, uint lane
) {
    const uint cBase = tg.x * 64u, tBase = tg.y * 64u, g = tg.z;
    if (g >= p.oGroups || tBase >= p.T || cBase >= p.oLoraRank) return;

    const uint inc = p.isIncumbent & 0xFFu;
    if (inc == 2u && qw_mla_load_shvq_lut<SUB2>(st.tgLut, p.bits, tid, 128u)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint wm = sg >> 1u, wn = sg & 1u;
    const uint warpT0 = tBase + wm * 32u, warpC0 = cBase + wn * 32u;
    const int sgpSm = min(32, int(p.T) - int(warpT0));

    if (inc == 2u) {
        if (sgpSm <= 0 || warpC0 >= p.oLoraRank) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return;
        }
        qw_hca_wo_a_nax_run<2, SUB2>(zIn, woABlocks, qUpGate, zGatedOut, p,
                                     &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
        return;
    }

    if (sgpSm <= 0 || warpC0 >= p.oLoraRank) return;

    if (inc == 3u) {
        qw_hca_wo_a_nax_run<3>(zIn, woABlocks, qUpGate, zGatedOut, p,
                               &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
    } else if (inc == 1u) {
        qw_hca_wo_a_nax_run<1>(zIn, woABlocks, qUpGate, zGatedOut, p,
                               &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
    } else if (p.bits == 8u) {
        qw_hca_wo_a_nax_run<8>(zIn, woABlocks, qUpGate, zGatedOut, p,
                               &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
    } else if (p.bits == 4u) {
        qw_hca_wo_a_nax_run<4>(zIn, woABlocks, qUpGate, zGatedOut, p,
                               &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
    } else {
        qw_hca_wo_a_nax_run<0>(zIn, woABlocks, qUpGate, zGatedOut, p,
                               &st.os[0][0], st.tgLut, g, warpT0, warpC0, sgpSm, sg, lane);
    }
}

// Grid: (oLoraRank / 64, (T + 63) / 64, oGroups), Threadgroup: (128, 1, 1).
kernel void qwg_hca_grouped_wo_a_nax(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoANaxTg st;
    qw_hca_grouped_wo_a_nax_body<false>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}

kernel void qwg_hca_grouped_wo_a_nax_sub2(
    device const half* zIn [[buffer(0)]], device const char* woABlocks [[buffer(1)]],
    device const half* qUpGate [[buffer(2)]], device half* zGatedOut [[buffer(3)]],
    constant HCAGroupedWoAParams& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup HCAWoANaxTg st;
    qw_hca_grouped_wo_a_nax_body<true>(zIn, woABlocks, qUpGate, zGatedOut, p, st, tg, tid, sg, lane);
}
