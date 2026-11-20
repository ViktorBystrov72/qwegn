// qwvision.metal — Basis-independent FP16 ViT kernels with FP32 accumulators.
// Supports LayerNorm, 4-tap pos_embed, 2D RoPE, O(N) tiled FlashAttention,
// GELU (tanh & erf), and tiled simdgroup GEMM across arbitrary geometries.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "qw_fastmath_safe.metalh"
using namespace metal;

// Biased LayerNorm: y = ((x - mu) / sqrt(var + eps)) * gamma + beta in FP32.
kernel void qwg_vis_layernorm_f16(
    device const half* x     [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta  [[buffer(2)]],
    device half*       out   [[buffer(3)]],
    constant uint&     rows  [[buffer(4)]],
    constant uint&     dim   [[buffer(5)]],
    constant float&    eps   [[buffer(6)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= rows || dim == 0u) return;
    constexpr uint kThreads = 128u;
    constexpr uint kSGs = kThreads / 32u;
    threadgroup float tgPartial[kSGs];
    device const half* xRow = x + (size_t)tg * dim;
    device half* outRow = out + (size_t)tg * dim;

    float localSum = 0.0f;
    for (uint j = tid; j < dim; j += kThreads) {
        localSum += float(xRow[j]);
    }
    localSum = simd_sum(localSum);
    if (lane == 0u) tgPartial[sgi] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float totalSum = 0.0f;
    for (uint s = 0u; s < kSGs; ++s) totalSum += tgPartial[s];
    const float mean = totalSum / float(dim);

    float localVar = 0.0f;
    for (uint j = tid; j < dim; j += kThreads) {
        const float d = float(xRow[j]) - mean;
        localVar += d * d;
    }
    localVar = simd_sum(localVar);
    if (lane == 0u) tgPartial[sgi] = localVar;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float totalVar = 0.0f;
    for (uint s = 0u; s < kSGs; ++s) totalVar += tgPartial[s];
    const float denom = metal::sqrt(totalVar / float(dim) + eps);
    const bool valid = qw_is_finite_bits(mean) && qw_is_finite_bits(denom) && (denom > 0.0f);
    const float invStd = valid ? (1.0f / denom) : 0.0f;

    for (uint j = tid; j < dim; j += kThreads) {
        const float normVal = valid ? ((float(xRow[j]) - mean) * invStd) : 0.0f;
        const float res = normVal * float(gamma[j]) + float(beta[j]);
        outRow[j] = half(qw_is_finite_bits(res) ? res : float(beta[j]));
    }
}

// 4-tap bilinear positional embedding fold: s[p, :] += sum_c w[p,c] * posEmbed[idx[p,c], :].
kernel void qwg_vis_pos_embed_f16(
    device half*        s        [[buffer(0)]],
    device const half*  posEmbed [[buffer(1)]],
    device const int*   binsIdx  [[buffer(2)]],
    device const float* binsWgt  [[buffer(3)]],
    constant uint&      nPatch   [[buffer(4)]],
    constant uint&      dim      [[buffer(5)]],
    constant uint&      numPos   [[buffer(6)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]]) {
    const uint col = tg.x * 256u + tid;
    const uint p = tg.y;
    if (p >= nPatch || col >= dim) return;
    float acc = float(s[(size_t)p * dim + col]);
    const size_t bBase = (size_t)p * 4u;
    for (uint c = 0u; c < 4u; ++c) {
        const int rawIdx = binsIdx[bBase + c];
        if (rawIdx >= 0 && uint(rawIdx) < numPos) {
            acc += binsWgt[bBase + c] * float(posEmbed[(size_t)uint(rawIdx) * dim + col]);
        }
    }
    s[(size_t)p * dim + col] = half(acc);
}

// 2D vision RoPE on Q and K slices of interleaved [nPatch, 3 * dim] qkv buffer.
kernel void qwg_vis_rope_qkv_f16(
    device half*        qkv          [[buffer(0)]],
    device const int*   posIds       [[buffer(1)]],
    device const float* invFreq      [[buffer(2)]],
    constant uint&      nPatch       [[buffer(3)]],
    constant uint&      heads        [[buffer(4)]],
    constant uint&      headDim      [[buffer(5)]],
    constant uint&      dim          [[buffer(6)]],
    constant uint&      invFreqCount [[buffer(7)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    const uint p = tg;
    const uint halfDim = headDim >> 1u;
    if (p >= nPatch || halfDim == 0u) return;
    const float hPos = float(posIds[(size_t)p * 2u]);
    const float wPos = float(posIds[(size_t)p * 2u + 1u]);
    const size_t rowBase = (size_t)p * (3u * dim);

    for (uint i = tid; i < halfDim; i += tptg) {
        float c = 0.0f;
        float s = 0.0f;
        if (i < invFreqCount) {
            const float angle = hPos * invFreq[i];
            c = metal::cos(angle);
            s = metal::sin(angle);
        } else if (i < 2u * invFreqCount) {
            const float angle = wPos * invFreq[i - invFreqCount];
            c = metal::cos(angle);
            s = metal::sin(angle);
        }
        for (uint h = 0u; h < heads; ++h) {
            const size_t qOff = rowBase + (size_t)h * headDim + i;
            const size_t kOff = qOff + dim;
            const float q0 = float(qkv[qOff]);
            const float q1 = float(qkv[qOff + halfDim]);
            qkv[qOff] = half(q0 * c - q1 * s);
            qkv[qOff + halfDim] = half(q1 * c + q0 * s);

            const float k0 = float(qkv[kOff]);
            const float k1 = float(qkv[kOff + halfDim]);
            qkv[kOff] = half(k0 * c - k1 * s);
            qkv[kOff + halfDim] = half(k1 * c + k0 * s);
        }
    }
}

// GELU tanh approximation (Law 4b: ViT MLP activation).
kernel void qwg_vis_gelu_tanh_f16(
    device half*   x     [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float v = float(x[gid]);
    const float z = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    float t;
    if (z > 15.0f) {
        t = 1.0f;
    } else if (z < -15.0f) {
        t = -1.0f;
    } else {
        t = precise::tanh(z);
    }
    x[gid] = half(0.5f * v * (1.0f + t));
}

// Exact ERF GELU via A&S 7.1.26 without negative-tail cancellation (Law 4b: Merger activation).
kernel void qwg_vis_gelu_erf_f16(
    device half*   x     [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float v = float(x[gid]);
    const float u = v * 0.7071067811865476f;
    const float au = metal::abs(u);
    const float t = 1.0f / (1.0f + 0.3275911f * au);
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                         + t * (-1.453152027f + t * 1.061405429f))));
    const float erfcAu = (au > 10.0f) ? 0.0f : (poly * metal::exp(-(au * au)));
    const float onePlusErf = (u >= 0.0f) ? (2.0f - erfcAu) : erfcAu;
    x[gid] = half(0.5f * v * onePlusErf);
}

// Stage 32x64 tile of src [stride] from global (rowStart, colStart) into dst [32, 64] with zero-pad.
static inline void vis_stage_tile_32x64(
    device const half* src,
    uint rowStart, uint rowEnd,
    uint colStart, uint colEnd,
    uint stride,
    threadgroup half* dst,
    uint tid) {
    for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
        const uint r = idx >> 6u;
        const uint c = idx & 63u;
        const uint gr = rowStart + r;
        const uint gc = colStart + c;
        dst[idx] = (gr < rowEnd && gc < colEnd) ? src[(size_t)gr * stride + gc] : half(0.0h);
    }
}

// O(N) tiled online-softmax FlashAttention over cu_seqlens segments in FP16 with FP32 accumulators.
kernel void qwg_vis_flash_attn_f16(
    device const half* qkv       [[buffer(0)]],
    device half*       out       [[buffer(1)]],
    device const int*  cuSeqlens [[buffer(2)]],
    constant uint&     numSegs   [[buffer(3)]],
    constant uint&     heads     [[buffer(4)]],
    constant uint&     headDim   [[buffer(5)]],
    constant uint&     dim       [[buffer(6)]],
    constant float&    scale     [[buffer(7)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    if (tg.z >= numSegs || tg.y >= heads || headDim == 0u) return;
    const int loRaw = cuSeqlens[tg.z];
    const int hiRaw = cuSeqlens[tg.z + 1u];
    if (loRaw < 0 || hiRaw <= loRaw) return;
    const uint lo = uint(loRaw), hi = uint(hiRaw);
    const uint q0 = lo + tg.x * 32u;
    if (q0 >= hi) return;

    const uint qkvStride = 3u * dim;
    const uint hOff = tg.y * headDim;
    device const half* qBase = qkv + hOff;
    device const half* kBase = qkv + dim + hOff;
    device const half* vBase = qkv + 2u * dim + hOff;
    device half* oBase = out + hOff;

    threadgroup half  kv_tg[32u * 64u];
    threadgroup half  p_tg[32u * 32u];
    threadgroup float work_tg[32u * 68u];

    const uint rowInSg = lane >> 2u;
    const uint colGroup = lane & 3u;
    const uint sgRow = sgi * 8u + rowInSg;

    simdgroup_half8x8 qf0[8], qf1[8];
    if (headDim <= 128u) {
        vis_stage_tile_32x64(qBase, q0, hi, 0u, headDim, qkvStride, kv_tg, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kd = 0u; kd < 8u; ++kd) {
            if (kd * 8u >= headDim) break;
            simdgroup_load(qf0[kd], kv_tg + (sgi * 8u) * 64u + kd * 8u, 64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (headDim > 64u) {
            vis_stage_tile_32x64(qBase, q0, hi, 64u, headDim, qkvStride, kv_tg, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kd = 0u; kd < 8u; ++kd) {
                if (64u + kd * 8u >= headDim) break;
                simdgroup_load(qf1[kd], kv_tg + (sgi * 8u) * 64u + kd * 8u, 64, ulong2(0, 0), false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint dPass = 0u; dPass < headDim; dPass += 128u) {
        float rowMax = 0.0f;
        float rowSum = 0.0f;
        bool rowBad = false;
        float oReg[32];
        for (uint i = 0u; i < 32u; ++i) oReg[i] = 0.0f;

        for (uint k0 = lo; k0 < hi; k0 += 32u) {
            simdgroup_float8x8 sacc[4];
            for (uint kt = 0u; kt < 4u; ++kt) sacc[kt] = simdgroup_float8x8(0.0f);

            for (uint d0 = 0u; d0 < headDim; d0 += 64u) {
                if (headDim > 128u) {
                    vis_stage_tile_32x64(qBase, q0, hi, d0, headDim, qkvStride, kv_tg, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint kd = 0u; kd < 8u; ++kd) {
                        if (d0 + kd * 8u >= headDim) break;
                        simdgroup_load(qf0[kd], kv_tg + (sgi * 8u) * 64u + kd * 8u, 64, ulong2(0, 0), false);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                vis_stage_tile_32x64(kBase, k0, hi, d0, headDim, qkvStride, kv_tg, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const bool useHi = (headDim <= 128u && d0 >= 64u);
                for (uint kd = 0u; kd < 8u; ++kd) {
                    if (d0 + kd * 8u >= headDim) break;
                    const simdgroup_half8x8 qFrag = useHi ? qf1[kd] : qf0[kd];
                    for (uint kt = 0u; kt < 4u; ++kt) {
                        simdgroup_half8x8 kf;
                        simdgroup_load(kf, kv_tg + (kt * 8u) * 64u + kd * 8u, 64, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(sacc[kt], qFrag, kf, sacc[kt]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint kt = 0u; kt < 4u; ++kt) {
                simdgroup_store(sacc[kt], work_tg + (sgi * 8u) * 36u + kt * 8u, 36, ulong2(0, 0), false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float vals[8];
            float localMax = -1e30f;
            bool localBad = false;
            for (uint j = 0u; j < 8u; ++j) {
                const uint col = colGroup * 8u + j;
                if (k0 + col < hi) {
                    const float v = work_tg[sgRow * 36u + col] * scale;
                    if (!qw_is_finite_bits(v)) {
                        localBad = true;
                        vals[j] = -1e30f;
                    } else {
                        vals[j] = v;
                        localMax = max(localMax, v);
                    }
                } else {
                    vals[j] = -1e30f;
                }
            }
            localMax = max(localMax, simd_shuffle_xor(localMax, 1u));
            localMax = max(localMax, simd_shuffle_xor(localMax, 2u));
            uint badBits = localBad ? 1u : 0u;
            badBits |= simd_shuffle_xor(badBits, 1u);
            badBits |= simd_shuffle_xor(badBits, 2u);
            if (badBits != 0u) rowBad = true;

            if (k0 == lo) {
                rowMax = localMax;
            } else {
                const float mnew = max(rowMax, localMax);
                if (mnew > rowMax && qw_is_finite_bits(mnew)) {
                    const float corr = metal::exp(rowMax - mnew);
                    if (!qw_is_finite_bits(corr)) {
                        rowBad = true;
                    } else {
                        rowSum *= corr;
                        for (uint i = 0u; i < 32u; ++i) oReg[i] *= corr;
                    }
                }
                rowMax = mnew;
            }

            float localSum = 0.0f;
            for (uint j = 0u; j < 8u; ++j) {
                const uint col = colGroup * 8u + j;
                float pVal = (k0 + col < hi && !rowBad) ? metal::exp(vals[j] - rowMax) : 0.0f;
                if (!qw_is_finite_bits(pVal)) {
                    pVal = 0.0f;
                    rowBad = true;
                }
                p_tg[sgRow * 32u + col] = half(pVal);
                localSum += pVal;
            }
            localSum += simd_shuffle_xor(localSum, 1u);
            localSum += simd_shuffle_xor(localSum, 2u);
            rowSum += localSum;
            if (!qw_is_finite_bits(rowSum)) rowBad = true;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 pf[4];
            for (uint kt = 0u; kt < 4u; ++kt) {
                simdgroup_load(pf[kt], p_tg + (sgi * 8u) * 32u + kt * 8u, 32, ulong2(0, 0), false);
            }

            for (uint sub = 0u; sub < 2u; ++sub) {
                const uint c0 = dPass + sub * 64u;
                if (c0 >= headDim) break;
                const uint validCols = min(64u, headDim - c0);
                vis_stage_tile_32x64(vBase, k0, hi, c0, headDim, qkvStride, kv_tg, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 pvAcc[8];
                for (uint nd = 0u; nd < 8u; ++nd) pvAcc[nd] = simdgroup_float8x8(0.0f);
                for (uint kt = 0u; kt < 4u; ++kt) {
                    if (k0 + kt * 8u >= hi) break;
                    for (uint nd = 0u; nd < 8u; ++nd) {
                        if (nd * 8u >= validCols) break;
                        simdgroup_half8x8 vf;
                        simdgroup_load(vf, kv_tg + (kt * 8u) * 64u + nd * 8u, 64, ulong2(0, 0), false);
                        simdgroup_multiply_accumulate(pvAcc[nd], pf[kt], vf, pvAcc[nd]);
                    }
                }
                for (uint nd = 0u; nd < 8u; ++nd) {
                    simdgroup_store(pvAcc[nd], work_tg + (sgi * 8u) * 68u + nd * 8u, 68, ulong2(0, 0), false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0u; i < 16u; ++i) {
                    const uint col = colGroup + 4u * i;
                    oReg[sub * 16u + i] += work_tg[sgRow * 68u + col];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        const uint globalRow = q0 + sgRow;
        const bool ok = !rowBad && qw_is_finite_bits(rowSum) && (rowSum > 0.0f);
        const float invSum = ok ? (1.0f / rowSum) : 0.0f;
        if (globalRow < hi) {
            for (uint sub = 0u; sub < 2u; ++sub) {
                const uint c0 = dPass + sub * 64u;
                if (c0 >= headDim) break;
                for (uint i = 0u; i < 16u; ++i) {
                    const uint gc = c0 + colGroup + 4u * i;
                    if (gc < headDim) {
                        float val = ok ? (oReg[sub * 16u + i] * invSum) : 0.0f;
                        if (!qw_is_finite_bits(val)) val = 0.0f;
                        oBase[(size_t)globalRow * dim + gc] = half(val);
                    }
                }
            }
        }
        if (dPass + 128u < headDim) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Tiled FP16 GEMM (FP32 accumulate): Y = X * W^T (+ bias) (+ residual), arbitrary M/N/K.
kernel void qwg_vis_gemm_f16(
    device const half* x        [[buffer(0)]],
    device const half* w        [[buffer(1)]],
    device const half* bias     [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    device half*       y        [[buffer(4)]],
    constant uint&     M        [[buffer(5)]],
    constant uint&     N        [[buffer(6)]],
    constant uint&     K        [[buffer(7)]],
    constant uint&     flags    [[buffer(8)]],   // bit0=hasBias, bit1=hasResidual
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  sgi [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = tg.y * 32u;
    const uint col0 = tg.x * 64u;
    if (row0 >= M || col0 >= N) return;

    constexpr uint kLDA = 40u;
    constexpr uint kLDB = 40u;
    threadgroup uint4 tgRaw[32u * 16u];
    threadgroup float* tgBuf = (threadgroup float*)tgRaw;
    threadgroup half* As = (threadgroup half*)tgRaw;
    threadgroup half* Bs = As + 32u * kLDA;

    const uint sgRow = (sgi >> 1u) * 16u;
    const uint sgCol = (sgi & 1u) * 32u;
    simdgroup_float8x8 acc[2][4];
    for (uint r = 0u; r < 2u; ++r) {
        for (uint c = 0u; c < 4u; ++c) acc[r][c] = simdgroup_float8x8(0.0f);
    }

    const uint lr = tid >> 2u;
    const uint lc0 = (tid & 3u) * 8u;
    const bool vecSafe = (K & 7u) == 0u;

    for (uint k0 = 0u; k0 < K; k0 += 32u) {
        const uint gmA = row0 + lr;
        const uint gk0 = k0 + lc0;
        if (gmA < M && vecSafe && gk0 + 8u <= K) {
            *(threadgroup uint4*)(As + lr * kLDA + lc0) = *(device const uint4*)(x + (size_t)gmA * K + gk0);
        } else {
            for (uint i = 0u; i < 8u; ++i) {
                const uint gk = gk0 + i;
                As[lr * kLDA + lc0 + i] = (gmA < M && gk < K) ? x[(size_t)gmA * K + gk] : half(0.0h);
            }
        }
        for (uint pass = 0u; pass < 2u; ++pass) {
            const uint br = pass * 32u + lr;
            const uint gnB = col0 + br;
            if (gnB < N && vecSafe && gk0 + 8u <= K) {
                *(threadgroup uint4*)(Bs + br * kLDB + lc0) = *(device const uint4*)(w + (size_t)gnB * K + gk0);
            } else {
                for (uint i = 0u; i < 8u; ++i) {
                    const uint gk = gk0 + i;
                    Bs[br * kLDB + lc0 + i] = (gnB < N && gk < K) ? w[(size_t)gnB * K + gk] : half(0.0h);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint kValid = min(32u, K - k0);
        for (uint kk = 0u; kk < 4u; ++kk) {
            if (kk * 8u >= kValid) break;
            simdgroup_half8x8 af[2];
            for (uint r = 0u; r < 2u; ++r) {
                simdgroup_load(af[r], As + (sgRow + r * 8u) * kLDA + kk * 8u, kLDA, ulong2(0, 0), false);
            }
            for (uint c = 0u; c < 4u; ++c) {
                simdgroup_half8x8 bf;
                simdgroup_load(bf, Bs + (sgCol + c * 8u) * kLDB + kk * 8u, kLDB, ulong2(0, 0), true);
                for (uint r = 0u; r < 2u; ++r) {
                    simdgroup_multiply_accumulate(acc[r][c], af[r], bf, acc[r][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0u; r < 2u; ++r) {
        for (uint c = 0u; c < 4u; ++c) {
            simdgroup_store(acc[r][c], tgBuf + (sgRow + r * 8u) * 64u + (sgCol + c * 8u), 64, ulong2(0, 0), false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool hasBias = (flags & 1u) != 0u;
    const bool hasResidual = (flags & 2u) != 0u;
    for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
        const uint lm = idx >> 6u;
        const uint ln = idx & 63u;
        const uint gm = row0 + lm;
        const uint gn = col0 + ln;
        if (gm < M && gn < N) {
            float val = tgBuf[idx];
            if (hasBias) val += float(bias[gn]);
            if (hasResidual) val += float(residual[(size_t)gm * N + gn]);
            y[(size_t)gm * N + gn] = half(val);
        }
    }
}
