// qw_ink2_ctrl_fast.metal — Zero-blit copy, coalesced multi-TG controller, and top-K router select kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

inline float qw_ink2_tanh_precise(float x) {
    return precise::tanh(clamp(x, -15.0f, 15.0f));
}

kernel void qwg_ink2_copy_bf16(device const bfloat* src [[buffer(0)]],
                               device bfloat* dst [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    const uint i8 = gid * 8u;
    if (i8 + 8u <= count && ((size_t(src) & 15u) == 0u) && ((size_t(dst) & 15u) == 0u)) {
        *(device uint4*)(dst + i8) = *(device const uint4*)(src + i8);
    } else {
        for (uint j = 0u; j < 8u; ++j) {
            if (i8 + j < count) dst[i8 + j] = src[i8 + j];
        }
    }
}

kernel void qwg_ink2_ctrl_pack_pair(device const float* xF32 [[buffer(0)]],
                                    device const float* msgF32 [[buffer(1)]],
                                    device bfloat* pairBF16 [[buffer(2)]],
                                    constant QWInk2CtrlParams& p [[buffer(3)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint simd_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    const size_t base = size_t(t) * p.H;
    device const float* xp = xF32 + base;
    device const float* mp = msgF32 + base;

    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    float xv[16], mv[16];
    float sqX = 0.0f, sqM = 0.0f;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float x0 = xp[d], m0 = mp[d];
            xv[k] = x0;
            mv[k] = m0;
            sqX += x0 * x0;
            sqM += m0 * m0;
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float x0 = xp[d], m0 = mp[d];
        sqX += x0 * x0;
        sqM += m0 * m0;
    }
    sqX = simd_sum(sqX);
    sqM = simd_sum(sqM);
    threadgroup float sgX[8], sgM[8];
    if (lane == 0u) {
        sgX[simd_id] = sqX;
        sgM[simd_id] = sqM;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float vx = simd_sum((lane < 8u) ? sgX[lane] : 0.0f);
        const float vm = simd_sum((lane < 8u) ? sgM[lane] : 0.0f);
        if (lane == 0u) {
            sgX[0] = rsqrt(vx / float(p.H) + p.eps);
            sgM[0] = rsqrt(vm / float(p.H) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invX = sgX[0], invM = sgM[0];
    device bfloat* dst = pairBF16 + size_t(t) * (2u * p.H);
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            dst[d] = qw_ink2_bf16(xv[k] * invX);
            dst[p.H + d] = qw_ink2_bf16(mv[k] * invM);
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        dst[d] = qw_ink2_bf16(xp[d] * invX);
        dst[p.H + d] = qw_ink2_bf16(mp[d] * invM);
    }
}

kernel void qwg_ink2_ctrl_apply_f32(device const float* xF32 [[buffer(0)]],
                                    device const float* msgF32 [[buffer(1)]],
                                    device const float* hidF32 [[buffer(2)]],
                                    device const bfloat* forgetW [[buffer(3)]],
                                    device const bfloat* gainW [[buffer(4)]],
                                    device float* outF32 [[buffer(5)]],
                                    constant QWInk2CtrlParams& p [[buffer(6)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint simd_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint d0 = tg.x * 32u + simd_id * 4u;
    const uint t = tg.y;
    if (d0 >= p.H || t >= p.T) return;

    device const float* hp = hidF32 + size_t(t) * p.rank;
    device const bfloat* fw[4];
    device const bfloat* gw[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint rr = (d0 + j < p.H) ? (d0 + j) : d0;
        fw[j] = forgetW + size_t(rr) * p.rank;
        gw[j] = gainW + size_t(rr) * p.rank;
    }
    float4 fAcc = float4(0.0f), gAcc = float4(0.0f);
    if ((p.rank & 1u) == 0u && ((size_t(hp) & 7u) == 0u)) {
        for (uint r = lane * 2u; r < p.rank; r += 64u) {
            const float2 h2 = *(device const float2*)(hp + r);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (d0 + j >= p.H) break;
                const ushort2 fu = *(device const ushort2*)(fw[j] + r);
                const ushort2 gu = *(device const ushort2*)(gw[j] + r);
                const float2 ff = float2(as_type<bfloat>(fu.x), as_type<bfloat>(fu.y));
                const float2 gf = float2(as_type<bfloat>(gu.x), as_type<bfloat>(gu.y));
                fAcc[j] += dot(h2, ff);
                gAcc[j] += dot(h2, gf);
            }
        }
    } else {
        for (uint r = lane; r < p.rank; r += 32u) {
            const float hr = hp[r];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (d0 + j >= p.H) break;
                fAcc[j] += hr * float(fw[j][r]);
                gAcc[j] += hr * float(gw[j][r]);
            }
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float fDot = simd_sum(fAcc[j]);
        const float gDot = simd_sum(gAcc[j]);
        if (lane == 0u && d0 + j < p.H) {
            const size_t idx = size_t(t) * p.H + (d0 + j);
            const float f = qw_ink2_tanh_precise(fDot);
            const float g = qw_ink2_tanh_precise(gDot);
            outF32[idx] = (1.0f - f) * xF32[idx] + g * msgF32[idx];
        }
    }
}

kernel void qwg_ink2_ctrl_blend_norm(device const float* xF32 [[buffer(0)]],
                                     device const float* msgF32 [[buffer(1)]],
                                     device const float* fgF32 [[buffer(2)]],
                                     device const bfloat* normW [[buffer(3)]],
                                     device bfloat* outBF16 [[buffer(4)]],
                                     constant QWInk2CtrlParams& p [[buffer(5)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    const size_t base = size_t(t) * p.H;
    device const float* xp = xF32 + base;
    device const float* mp = msgF32 + base;
    device const float* fgp = fgF32 + size_t(t) * (2u * p.H);

    const uint nSteps = min(16u, (p.H + 255u) >> 8u);
    float vReg[16];
    float sq = 0.0f;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float f = qw_ink2_tanh_precise(fgp[d]);
            const float g = qw_ink2_tanh_precise(fgp[p.H + d]);
            const float v = (1.0f - f) * xp[d] + g * mp[d];
            vReg[k] = v;
            sq += v * v;
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float f = qw_ink2_tanh_precise(fgp[d]);
        const float g = qw_ink2_tanh_precise(fgp[p.H + d]);
        const float v = (1.0f - f) * xp[d] + g * mp[d];
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSq[8];
    if (lane == 0u) sgSq[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float totalSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        if (lane == 0u) sgSq[0] = rsqrt(totalSq / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSq[0];
    device bfloat* dst = outBF16 + base;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            dst[d] = qw_ink2_bf16((vReg[k] * invRms) * float(normW[d]));
        }
    }
    for (uint d = tid + 4096u; d < p.H; d += 256u) {
        const float f = qw_ink2_tanh_precise(fgp[d]);
        const float g = qw_ink2_tanh_precise(fgp[p.H + d]);
        const float v = (1.0f - f) * xp[d] + g * mp[d];
        dst[d] = qw_ink2_bf16((v * invRms) * float(normW[d]));
    }
}

kernel void qwg_ink2_router_select_topk(device const float* logitsF32 [[buffer(0)]],
                                        device const float* bias [[buffer(1)]],
                                        device int32_t* topIds [[buffer(2)]],
                                        device float* topWeights [[buffer(3)]],
                                        constant QWInk2RouterParams& p [[buffer(4)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint simd_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    device const float* lr = logitsF32 + size_t(t) * p.numExperts;
    threadgroup float scoresSh[512];
    const uint kLimit = min(p.topK, 16u);
    const uint keepK = (p.moeSpecBudget > 0u && t >= 1u && p.moeSpecBudget < kLimit)
        ? p.moeSpecBudget : kLimit;

    if (p.numExperts <= 512u) {
        const uint e0 = tid;
        const uint e1 = tid + 256u;
        float b0 = -1e30f, b1 = -1e30f;
        if (e0 < p.numExperts) {
            const float s0 = qw_ink2_sigmoid(lr[e0]);
            scoresSh[e0] = s0;
            b0 = s0 + bias[e0];
        }
        if (e1 < p.numExperts) {
            const float s1 = qw_ink2_sigmoid(lr[e1]);
            scoresSh[e1] = s1;
            b1 = s1 + bias[e1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float sgB[8];
        threadgroup int32_t sgE[8];
        threadgroup int32_t winESh;
        float sumW = 0.0f;
        device int32_t* outId = topIds + size_t(t) * p.topK;
        device float* outW = topWeights + size_t(t) * p.topK;

        for (uint k = 0u; k < keepK; ++k) {
            float locB = (e0 < p.numExperts) ? b0 : -1e30f;
            int32_t locE = (e0 < p.numExperts) ? int32_t(e0) : 0x7fffffff;
            if (e1 < p.numExperts && (b1 > locB || (b1 == locB && int32_t(e1) < locE))) {
                locB = b1;
                locE = int32_t(e1);
            }
            #pragma unroll
            for (uint off = 16u; off > 0u; off >>= 1u) {
                const float ob = simd_shuffle_down(locB, off);
                const int32_t oe = simd_shuffle_down(locE, off);
                if (ob > locB || (ob == locB && oe < locE)) {
                    locB = ob;
                    locE = oe;
                }
            }
            if (lane == 0u && simd_id < 8u) {
                sgB[simd_id] = locB;
                sgE[simd_id] = locE;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float bestB = sgB[0];
                int32_t bestE = sgE[0];
                for (uint s = 1u; s < 8u; ++s) {
                    const float cb = sgB[s];
                    const int32_t ce = sgE[s];
                    if (cb > bestB || (cb == bestB && ce < bestE)) {
                        bestB = cb;
                        bestE = ce;
                    }
                }
                winESh = bestE;
                outId[k] = bestE;
                const float w = (bestE >= 0 && uint(bestE) < p.numExperts) ? scoresSh[bestE] : 0.0f;
                outW[k] = w;
                sumW += w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int32_t wE = winESh;
            if (int32_t(e0) == wE) b0 = -1e30f;
            if (int32_t(e1) == wE) b1 = -1e30f;
        }
        if (tid == 0u) sgB[0] = p.routeScale / (sumW + 1e-20f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < kLimit) {
            if (tid < keepK) {
                outW[tid] *= sgB[0];
            } else {
                outId[tid] = -1;
                outW[tid] = 0.0f;
            }
        }
        return;
    }

    threadgroup float biasedSh[512];
    for (uint e = tid; e < p.numExperts && e < 512u; e += 256u) {
        const float sig = qw_ink2_sigmoid(lr[e]);
        scoresSh[e] = sig;
        biasedSh[e] = sig + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        int32_t chosen[16];
        float vals[16];
        float sumW = 0.0f;
        const uint limE = min(p.numExperts, 512u);
        for (uint k = 0u; k < keepK; ++k) {
            float bestB = -1e30f;
            int32_t bestE = 0;
            for (uint e = 0u; e < limE; ++e) {
                const float b = biasedSh[e];
                if (b > bestB) {
                    bestB = b;
                    bestE = int32_t(e);
                }
            }
            biasedSh[bestE] = -1e30f;
            chosen[k] = bestE;
            const float w = scoresSh[bestE];
            vals[k] = w;
            sumW += w;
        }
        const float inv = p.routeScale / (sumW + 1e-20f);
        device int32_t* outId = topIds + size_t(t) * p.topK;
        device float* outW = topWeights + size_t(t) * p.topK;
        for (uint k = 0u; k < keepK; ++k) {
            outId[k] = chosen[k];
            outW[k] = vals[k] * inv;
        }
        for (uint k = keepK; k < kLimit; ++k) {
            outId[k] = -1;
            outW[k] = 0.0f;
        }
    }
}

kernel void qwg_ink2_f16_to_bf16(device const half* src [[buffer(0)]],
                                 device bfloat* dst [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint base = gid * 4u;
    for (uint j = 0u; j < 4u; ++j) {
        const uint i = base + j;
        if (i < count) dst[i] = qw_ink2_bf16(float(src[i]));
    }
}

kernel void qwg_ink2_f32_to_bf16(device const float* src [[buffer(0)]],
                                 device bfloat* dst [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 constant float& softcap [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint base = gid * 4u;
    for (uint j = 0u; j < 4u; ++j) {
        const uint i = base + j;
        if (i < count) {
            float v = src[i];
            if (softcap > 0.0f) v = qw_ink2_tanh_precise(v / softcap) * softcap;
            dst[i] = qw_ink2_bf16(v);
        }
    }
}

kernel void qwg_ink2_softcap_f32(device float* data [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 constant float& softcap [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < count && softcap > 0.0f) {
        data[gid] = qw_ink2_tanh_precise(data[gid] / softcap) * softcap;
    }
}

kernel void qwg_ink2_bf16_to_f16_fwht32(device const bfloat* src [[buffer(0)]],
                                        device char* dst [[buffer(1)]],
                                        constant uint& count [[buffer(2)]],
                                        constant uint& flags [[buffer(3)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const uint idx = tg * 256u + tid;
    float v = (idx < count) ? float(src[idx]) : 0.0f;
#if !defined(QW_BASIS_STANDARD)
    if ((flags & 1u) != 0u) {
        #pragma unroll
        for (ushort s = 1u; s < 32u; s <<= 1u) {
            const float pr = simd_shuffle_xor(v, s);
            v = ((lane & s) == 0u) ? (v + pr) : (pr - v);
        }
        v *= 0.17677669529663689f;
    }
#else
    (void)lane;
    (void)flags;
#endif
    if (idx < count) {
        if ((flags & 2u) != 0u) ((device float*)dst)[idx] = v;
        else ((device half*)dst)[idx] = half(v);
    }
}

struct QWInk2SplitAttnProjParams {
    uint T, totalCols, qDim, gateDim, rDim, sinkDim, kvDim, hasKV;
};

kernel void qwg_ink2_split_attn_proj(device const bfloat* fused [[buffer(0)]],
                                     device bfloat* qOut [[buffer(1)]],
                                     device bfloat* gateOut [[buffer(2)]],
                                     device bfloat* rOut [[buffer(3)]],
                                     device bfloat* sinkOut [[buffer(4)]],
                                     device bfloat* kOut [[buffer(5)]],
                                     device bfloat* vOut [[buffer(6)]],
                                     constant QWInk2SplitAttnProjParams& p [[buffer(7)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint c8 = gid.x * 8u;
    const uint t = gid.y;
    if (c8 >= p.totalCols || t >= p.T) return;
    const bool vec8 = (c8 + 8u <= p.totalCols)
        && (((p.qDim | p.gateDim | p.rDim | p.sinkDim | p.kvDim | p.totalCols) & 7u) == 0u)
        && ((size_t(fused) & 15u) == 0u) && ((size_t(qOut) & 15u) == 0u)
        && ((size_t(gateOut) & 15u) == 0u) && ((size_t(rOut) & 15u) == 0u)
        && ((size_t(sinkOut) & 15u) == 0u)
        && (p.hasKV == 0u || (((size_t(kOut) & 15u) == 0u) && ((size_t(vOut) & 15u) == 0u)));
    if (vec8) {
        const uint4 v = *(device const uint4*)(fused + size_t(t) * p.totalCols + c8);
        if (c8 < p.qDim) {
            *(device uint4*)(qOut + size_t(t) * p.qDim + c8) = v;
            return;
        }
        uint rem = c8 - p.qDim;
        if (rem < p.gateDim) {
            *(device uint4*)(gateOut + size_t(t) * p.gateDim + rem) = v;
            return;
        }
        rem -= p.gateDim;
        if (rem < p.rDim) {
            *(device uint4*)(rOut + size_t(t) * p.rDim + rem) = v;
            return;
        }
        rem -= p.rDim;
        if (rem < p.sinkDim) {
            *(device uint4*)(sinkOut + size_t(t) * p.sinkDim + rem) = v;
            return;
        }
        rem -= p.sinkDim;
        if (p.hasKV != 0u) {
            if (rem < p.kvDim) {
                *(device uint4*)(kOut + size_t(t) * p.kvDim + rem) = v;
                return;
            }
            rem -= p.kvDim;
            if (rem < p.kvDim) {
                *(device uint4*)(vOut + size_t(t) * p.kvDim + rem) = v;
            }
        }
        return;
    }
    for (uint j = 0u; j < 8u; ++j) {
        const uint c = c8 + j;
        if (c >= p.totalCols) break;
        const bfloat val = fused[size_t(t) * p.totalCols + c];
        if (c < p.qDim) {
            qOut[size_t(t) * p.qDim + c] = val;
        } else {
            uint rem = c - p.qDim;
            if (rem < p.gateDim) {
                gateOut[size_t(t) * p.gateDim + rem] = val;
            } else {
                rem -= p.gateDim;
                if (rem < p.rDim) {
                    rOut[size_t(t) * p.rDim + rem] = val;
                } else {
                    rem -= p.rDim;
                    if (rem < p.sinkDim) {
                        sinkOut[size_t(t) * p.sinkDim + rem] = val;
                    } else if (p.hasKV != 0u) {
                        rem -= p.sinkDim;
                        if (rem < p.kvDim) {
                            kOut[size_t(t) * p.kvDim + rem] = val;
                        } else {
                            rem -= p.kvDim;
                            if (rem < p.kvDim) vOut[size_t(t) * p.kvDim + rem] = val;
                        }
                    }
                }
            }
        }
    }
}

struct QWInk2SConvTreeTapParams {
    uint T, C, nKV, kernelSize, updateState, resMode, writeVerifyTap, verifyTapOffset;
    float invSqrtDenom;
    uint pad0, pad1, pad2;
    int4 treeLags[16];
};

struct QWInk2SConvVerifyCommitTreeParams {
    uint numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride;
    uint keptNodes[16];
};

struct QWInk2TreeKVCompactParams {
    uint oldPos, kept, nKV, headDim, ringCap, kvMode, isSliding, pad;
    uint keptNodes[16];
};

static inline float qw_ink2_tree_lag_sample(
        device const bfloat* xBF16, device const float* stateF32,
        uint C, float s0, float s1, float s2, uint c, int idx) {
    if (idx == -3) return s0;
    if (idx == -2) return s1;
    if (idx == -1) return s2;
    if (idx < -3) return stateF32[size_t(3 + idx) * C + c];
    return float(xBF16[size_t(idx) * C + c]);
}

kernel void qwg_ink2_sconv_static_tree_tap(device const bfloat* xBF16 [[buffer(0)]],
                                           device const bfloat* weight [[buffer(1)]],
                                           device float* stateF32 [[buffer(2)]],
                                           device const bfloat* resBF16 [[buffer(3)]],
                                           device const float* resF32 [[buffer(4)]],
                                           device bfloat* outBF16 [[buffer(5)]],
                                           device float* outF32 [[buffer(6)]],
                                           device bfloat* verifyTap [[buffer(7)]],
                                           constant QWInk2SConvTreeTapParams& p [[buffer(8)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= p.C || t >= p.T || t >= 16u) return;

    device const bfloat* wc = weight + size_t(c) * 4u;
    const float w0 = float(wc[0]), w1 = float(wc[1]), w2 = float(wc[2]), w3 = float(wc[3]);
    const float s0 = stateF32[0u * p.C + c];
    const float s1 = stateF32[1u * p.C + c];
    const float s2 = stateF32[2u * p.C + c];
    const int4 lag = p.treeLags[t];

    const size_t idx = size_t(t) * p.C + c;
    const bfloat xRaw = xBF16[idx];
    if (p.writeVerifyTap != 0u) {
        verifyTap[size_t(p.verifyTapOffset) + idx] = xRaw;
    }
    const float xCur = float(xRaw);
    const float acc = w0 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.z)
                    + w1 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.y)
                    + w2 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.x)
                    + w3 * xCur;
    const bfloat convBF16 = qw_ink2_bf16((acc + xCur) * p.invSqrtDenom);
    if (p.resMode == 0u) {
        outBF16[idx] = convBF16;
    } else if (p.resMode == 1u) {
        outF32[idx] = float(convBF16);
    } else if (p.resMode == 2u) {
        outF32[idx] = float(convBF16) + float(resBF16[idx]);
    } else {
        outBF16[idx] = qw_ink2_bf16(resF32[idx] + float(convBF16));
    }
}

kernel void qwg_ink2_sconv_dynamic_tree_tap(device const bfloat* xBF16 [[buffer(0)]],
                                            device const bfloat* staticW [[buffer(1)]],
                                            device const float* dynWF32 [[buffer(2)]],
                                            device float* stateF32 [[buffer(3)]],
                                            device bfloat* outBF16 [[buffer(4)]],
                                            device bfloat* verifyTap [[buffer(5)]],
                                            constant QWInk2SConvTreeTapParams& p [[buffer(6)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= p.C || t >= p.T || t >= 16u) return;

    const uint headDim = p.C / p.nKV;
    const uint kvh = c / headDim;
    const size_t dynBase = (size_t(t) * p.nKV + kvh) * 4u;
    const float w0 = float(staticW[0u * p.C + c]) + dynWF32[dynBase + 0u];
    const float w1 = float(staticW[1u * p.C + c]) + dynWF32[dynBase + 1u];
    const float w2 = float(staticW[2u * p.C + c]) + dynWF32[dynBase + 2u];
    const float w3 = float(staticW[3u * p.C + c]) + dynWF32[dynBase + 3u];
    const float s0 = stateF32[0u * p.C + c];
    const float s1 = stateF32[1u * p.C + c];
    const float s2 = stateF32[2u * p.C + c];
    const int4 lag = p.treeLags[t];

    const size_t idx = size_t(t) * p.C + c;
    const bfloat xRaw = xBF16[idx];
    if (p.writeVerifyTap != 0u) {
        verifyTap[size_t(p.verifyTapOffset) + idx] = xRaw;
    }
    const float xCur = float(xRaw);
    const float acc = w0 * xCur
                    + w1 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.x)
                    + w2 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.y)
                    + w3 * qw_ink2_tree_lag_sample(xBF16, stateF32, p.C, s0, s1, s2, c, lag.z);
    outBF16[idx] = qw_ink2_bf16((acc + xCur) * p.invSqrtDenom);
}

kernel void qwg_ink2_sconv_verify_commit_tree(device float* convStatesSlab [[buffer(0)]],
                                               device const bfloat* verifyInputsSlab [[buffer(1)]],
                                               constant QWInk2SConvVerifyCommitTreeParams& p [[buffer(2)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    const uint flatC = gid.x;
    const uint v = gid.y;
    const uint totalCh = 2u * p.kvDim + 2u * p.H;
    if (v >= p.numVisits || flatC >= totalCh || p.kept == 0u || p.kept > 16u) return;

    const size_t visitStBase = size_t(v) * (2u * p.stateKvStride + 2u * p.stateHStride);
    const size_t visitVfBase = size_t(v) * (2u * p.verifyKvStride + 2u * p.verifyHStride);
    const bool inKV = (flatC < 2u * p.kvDim);
    const uint sub = inKV ? flatC : (flatC - 2u * p.kvDim);
    const uint ch = inKV ? p.kvDim : p.H;
    const uint slot = sub / ch;
    const uint c = sub - slot * ch;
    const size_t stOff = visitStBase + (inKV ? (slot * p.stateKvStride) : (2u * p.stateKvStride + slot * p.stateHStride));
    const size_t vfOff = visitVfBase + (inKV ? (slot * p.verifyKvStride) : (2u * p.verifyKvStride + slot * p.verifyHStride));

    device float* dst = convStatesSlab + stOff;
    device const bfloat* xIn = verifyInputsSlab + vfOff;
    float n0, n1, n2;
    if (p.kept >= 3u) {
        n0 = float(xIn[size_t(p.keptNodes[p.kept - 3u]) * ch + c]);
        n1 = float(xIn[size_t(p.keptNodes[p.kept - 2u]) * ch + c]);
        n2 = float(xIn[size_t(p.keptNodes[p.kept - 1u]) * ch + c]);
    } else if (p.kept == 2u) {
        n0 = dst[2u * ch + c];
        n1 = float(xIn[size_t(p.keptNodes[0]) * ch + c]);
        n2 = float(xIn[size_t(p.keptNodes[1]) * ch + c]);
    } else {
        n0 = dst[1u * ch + c];
        n1 = dst[2u * ch + c];
        n2 = float(xIn[size_t(p.keptNodes[0]) * ch + c]);
    }
    dst[0u * ch + c] = n0;
    dst[1u * ch + c] = n1;
    dst[2u * ch + c] = n2;
}

kernel void qwg_ink2_tree_kv_compact(device uchar* kCache [[buffer(0)]],
                                     device uchar* vCache [[buffer(1)]],
                                     device uchar* kMeta [[buffer(2)]],
                                     device uchar* vMeta [[buffer(3)]],
                                     constant QWInk2TreeKVCompactParams& p [[buffer(4)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint isV = tg.y;
    if (kvh >= p.nKV || isV >= 2u || p.kept <= 1u || p.kept > 16u) return;

    device uchar* cacheBuf = (isV == 0u) ? kCache : vCache;
    device uchar* metaBuf = (isV == 0u) ? kMeta : vMeta;
    const uint payBytes = (p.kvMode == 0u) ? (p.headDim * 2u)
                        : ((p.kvMode == 1u) ? p.headDim : (p.headDim >> 1u));
    const uint payWords = payBytes >> 2u;

    for (uint i = 1u; i < p.kept; ++i) {
        const uint srcPos = p.oldPos + p.keptNodes[i];
        const uint dstPos = p.oldPos + i;
        if (srcPos == dstPos) continue;
        const uint srcSlot = (p.isSliding != 0u) ? (srcPos % p.ringCap) : srcPos;
        const uint dstSlot = (p.isSliding != 0u) ? (dstPos % p.ringCap) : dstPos;
        if (srcSlot == dstSlot) continue;

        const size_t srcRowHead = size_t(srcSlot) * p.nKV + kvh;
        const size_t dstRowHead = size_t(dstSlot) * p.nKV + kvh;
        if ((payBytes & 3u) == 0u) {
            device const uint* sW = (device const uint*)(cacheBuf + srcRowHead * payBytes);
            device uint* dW = (device uint*)(cacheBuf + dstRowHead * payBytes);
            for (uint w = lane; w < payWords; w += 32u) dW[w] = sW[w];
        } else {
            device const uchar* sB = cacheBuf + srcRowHead * payBytes;
            device uchar* dB = cacheBuf + dstRowHead * payBytes;
            for (uint b = lane; b < payBytes; b += 32u) dB[b] = sB[b];
        }
        if (lane == 0u && p.kvMode != 0u) {
            if (p.kvMode == 1u) ((device ushort*)metaBuf)[dstRowHead] = ((device const ushort*)metaBuf)[srcRowHead];
            else ((device uint*)metaBuf)[dstRowHead] = ((device const uint*)metaBuf)[srcRowHead];
        }
        simdgroup_barrier(mem_flags::mem_device);
    }
}
