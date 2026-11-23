// qw_ink2_loop_conv.metal — Inkling-2 AttnRes, loop message bank, controllers, and short convs.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

kernel void qwg_ink2_attn_res(device const bfloat* blocksBuf [[buffer(0)]],
                              device const bfloat* partialBF16 [[buffer(1)]],
                              device const float* partialF32 [[buffer(2)]],
                              device const float* queryF32 [[buffer(3)]],
                              device float* outF32 [[buffer(4)]],
                              constant QWInk2AttnResParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint q = tg.x;
    const uint t = tg.y;
    if (q >= p.numQ || t >= p.T) return;
    const uint nCand = min(p.nBlocks + 1u, 8u);
    device const float* qRow = queryF32 + size_t(q) * p.H;
    const float hScale = rsqrt(float(p.H));

    threadgroup float sgSq[8];
    threadgroup float sgDot[8];
    threadgroup float invRmsSh[8];
    threadgroup float alphaSh[8];

    for (uint b = 0u; b < nCand; ++b) {
        float sq = 0.0f;
        float qd = 0.0f;
        if (b < p.nBlocks) {
            device const bfloat* bp = blocksBuf + size_t(p.slotIds[b]) * p.blockStride + size_t(t) * p.H;
            for (uint d = tid; d < p.H; d += 256u) {
                float v = float(bp[d]);
                sq += v * v;
                qd += v * qRow[d];
            }
        } else if (p.partialIsF32 != 0u) {
            device const float* pp = partialF32 + size_t(t) * p.H;
            for (uint d = tid; d < p.H; d += 256u) {
                float v = pp[d];
                sq += v * v;
                qd += v * qRow[d];
            }
        } else {
            device const bfloat* pp = partialBF16 + size_t(t) * p.H;
            for (uint d = tid; d < p.H; d += 256u) {
                float v = float(pp[d]);
                sq += v * v;
                qd += v * qRow[d];
            }
        }
        sq = simd_sum(sq);
        qd = simd_sum(qd);
        if (lane == 0u) {
            sgSq[simd_id] = sq;
            sgDot[simd_id] = qd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            float vSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
            float vDot = simd_sum((lane < 8u) ? sgDot[lane] : 0.0f);
            if (lane == 0u) {
                float inv = rsqrt(vSq / float(p.H) + p.eps);
                invRmsSh[b] = inv;
                alphaSh[b] = (vDot * inv) * hScale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float m = -1e30f;
        for (uint b = 0u; b < nCand; ++b) m = max(m, alphaSh[b]);
        float s = 0.0f;
        for (uint b = 0u; b < nCand; ++b) {
            float w = exp(alphaSh[b] - m);
            alphaSh[b] = w;
            s += w;
        }
        float invS = 1.0f / max(s, 1e-30f);
        for (uint b = 0u; b < nCand; ++b) alphaSh[b] *= invS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dst = outF32 + (size_t(q) * p.T + t) * p.H;
    for (uint d = tid; d < p.H; d += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < nCand; ++b) {
            float v;
            if (b < p.nBlocks) {
                v = float(blocksBuf[size_t(p.slotIds[b]) * p.blockStride + size_t(t) * p.H + d]);
            } else if (p.partialIsF32 != 0u) {
                v = partialF32[size_t(t) * p.H + d];
            } else {
                v = float(partialBF16[size_t(t) * p.H + d]);
            }
            float kv = (p.normalizeValues != 0u) ? (v * invRmsSh[b]) : v;
            acc += alphaSh[b] * kv;
        }
        dst[d] = acc;
    }
}

kernel void qwg_ink2_loop_bank_norm(device const bfloat* blocksBuf [[buffer(0)]],
                                    device const bfloat* partialBF16 [[buffer(1)]],
                                    device float* bankF32 [[buffer(2)]],
                                    constant QWInk2BankAttendParams& p [[buffer(3)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint simd_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint sIdx = tg.x;
    const uint t = tg.y;
    if (sIdx >= p.nStates || t >= p.T) return;
    const bool isLast = (sIdx + 1u == p.nStates);
    device const bfloat* src = isLast
        ? (partialBF16 + size_t(t) * p.H)
        : (blocksBuf + size_t(p.slotIds[sIdx]) * p.stateStride + size_t(t) * p.H);

    float sq = 0.0f;
    for (uint d = tid; d < p.H; d += 256u) {
        float v = float(src[d]);
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSq[8];
    if (lane == 0u) sgSq[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float val = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        if (lane == 0u) sgSq[0] = rsqrt(val / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = sgSq[0];
    device float* dst = bankF32 + (size_t(t) * p.nStates + sIdx) * p.H;
    for (uint d = tid; d < p.H; d += 256u) {
        dst[d] = float(src[d]) * inv;
    }
}

kernel void qwg_ink2_loop_bank_attend(device const float* bankF32 [[buffer(0)]],
                                      device const float* msgQueries [[buffer(1)]],
                                      device float* msgOutF32 [[buffer(2)]],
                                      constant QWInk2BankAttendParams& p [[buffer(3)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint simd_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint q = tg.x, t = tg.y;
    if (q >= p.numQ || t >= p.T) return;
    const uint M = min(p.nStates, 8u);
    device const float* qRow = msgQueries + size_t(q) * p.H;
    device const float* tBank = bankF32 + size_t(t) * p.nStates * p.H;
    const uint nSteps = min(9u, (p.H + 255u) >> 8u);
    threadgroup float sgDot[64], alphaSh[8];
    float bReg[8][9];
    float qd[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            const float qv = qRow[d];
            for (uint m = 0u; m < M; ++m) {
                const float bv = tBank[size_t(m) * p.H + d];
                bReg[m][k] = bv;
                qd[m] += qv * bv;
            }
        }
    }
    for (uint d = tid + 2304u; d < p.H; d += 256u) {
        const float qv = qRow[d];
        for (uint m = 0u; m < M; ++m) qd[m] += qv * tBank[size_t(m) * p.H + d];
    }
    for (uint m = 0u; m < M; ++m) {
        const float s = simd_sum(qd[m]);
        if (lane == 0u) sgDot[m * 8u + simd_id] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < M) {
        const float v = simd_sum((lane < 8u) ? sgDot[simd_id * 8u + lane] : 0.0f);
        if (lane == 0u) alphaSh[simd_id] = v * rsqrt(float(p.H));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float mx = -1e30f, sm = 0.0f;
        for (uint m = 0u; m < M; ++m) mx = max(mx, alphaSh[m]);
        for (uint m = 0u; m < M; ++m) { float w = exp(alphaSh[m] - mx); alphaSh[m] = w; sm += w; }
        const float invS = 1.0f / max(sm, 1e-30f);
        for (uint m = 0u; m < M; ++m) alphaSh[m] *= invS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* dst = msgOutF32 + (size_t(q) * p.T + t) * p.H;
    for (uint k = 0u; k < nSteps; ++k) {
        const uint d = tid + (k << 8u);
        if (d < p.H) {
            float acc = 0.0f;
            for (uint m = 0u; m < M; ++m) acc += alphaSh[m] * bReg[m][k];
            dst[d] = acc;
        }
    }
    for (uint d = tid + 2304u; d < p.H; d += 256u) {
        float acc = 0.0f;
        for (uint m = 0u; m < M; ++m) acc += alphaSh[m] * tBank[size_t(m) * p.H + d];
        dst[d] = acc;
    }
}

kernel void qwg_ink2_msg_controller(device const float* xF32 [[buffer(0)]],
                                    device const float* msgF32 [[buffer(1)]],
                                    device const bfloat* downW [[buffer(2)]],
                                    device const bfloat* forgetW [[buffer(3)]],
                                    device const bfloat* gainW [[buffer(4)]],
                                    device float* outF32 [[buffer(5)]],
                                    constant QWInk2CtrlParams& p [[buffer(6)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint simd_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    const size_t base = size_t(t) * p.H;
    device const float* xp = xF32 + base;
    device const float* mp = msgF32 + base;

    float sqX = 0.0f, sqM = 0.0f;
    for (uint d = tid; d < p.H; d += 256u) {
        float xv = xp[d], mv = mp[d];
        sqX += xv * xv;
        sqM += mv * mv;
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
        float vx = simd_sum((lane < 8u) ? sgX[lane] : 0.0f);
        float vm = simd_sum((lane < 8u) ? sgM[lane] : 0.0f);
        if (lane == 0u) {
            sgX[0] = rsqrt(vx / float(p.H) + p.eps);
            sgM[0] = rsqrt(vm / float(p.H) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invX = sgX[0];
    const float invM = sgM[0];

    threadgroup bfloat pairSh[8192];
    for (uint d = tid; d < p.H; d += 256u) {
        pairSh[d] = qw_ink2_bf16(xp[d] * invX);
        pairSh[p.H + d] = qw_ink2_bf16(mp[d] * invM);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float hidSh[256];
    const uint twoH = p.H * 2u;
    for (uint r = simd_id; r < p.rank; r += 8u) {
        device const bfloat* dr = downW + size_t(r) * twoH;
        float acc = 0.0f;
        for (uint k = lane; k < twoH; k += 32u) {
            acc += float(pairSh[k]) * float(dr[k]);
        }
        float v = simd_sum(acc);
        if (lane == 0u) hidSh[r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* dst = outF32 + base;
    for (uint d = tid; d < p.H; d += 256u) {
        device const bfloat* fw = forgetW + size_t(d) * p.rank;
        device const bfloat* gw = gainW + size_t(d) * p.rank;
        float fDot = 0.0f, gDot = 0.0f;
        for (uint r = 0u; r < p.rank; ++r) {
            float hr = hidSh[r];
            fDot += hr * float(fw[r]);
            gDot += hr * float(gw[r]);
        }
        float f = qw_ink2_tanh_safe(fDot);
        float g = qw_ink2_tanh_safe(gDot);
        dst[d] = (1.0f - f) * xp[d] + g * mp[d];
    }
}

struct QWInk2AttnResCtrlNormParams {
    uint T, H, nBlocks, numQ, numOut, blockStride, partialIsF32, normalizeValues;
    uint useCtrl, rank, rotateOut, outIsF16, outStride, msgStride;
    float attnResEps, rmsEps;
    uint slotIds[8], normWOffsets[4], ctrlDownOffsets[4], ctrlForgetOffsets[4], ctrlGainOffsets[4];
};

struct QWInk2SConvTapParams {
    uint T, C, nKV, kernelSize, updateState, resMode, writeVerifyTap, verifyTapOffset;
    float invSqrtDenom;
};

struct QWInk2SConvVerifyCommitParams {
    uint numVisits, kvDim, H, kept, stateKvStride, stateHStride, verifyKvStride, verifyHStride;
};

static inline void qw_ink2_sconv_static_body(
        device const bfloat* xBF16, device const bfloat* weight,
        device float* stateF32, device const bfloat* resBF16,
        device const float* resF32, device bfloat* outBF16,
        device float* outF32, device bfloat* verifyTap,
        uint T, uint C, uint updateState, uint resMode,
        uint writeVerifyTap, uint verifyTapOffset, float invSqrtDenom,
        uint2 gid) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C || t >= T) return;

    device const bfloat* wc = weight + size_t(c) * 4u;
    const float w0 = float(wc[0]), w1 = float(wc[1]), w2 = float(wc[2]), w3 = float(wc[3]);
    float s0 = stateF32[0u * C + c];
    float s1 = stateF32[1u * C + c];
    float s2 = stateF32[2u * C + c];

    auto sampleAt = [&](int idx) -> float {
        if (idx == -3) return s0;
        if (idx == -2) return s1;
        if (idx == -1) return s2;
        if (idx < -3) return stateF32[size_t(3 + idx) * C + c];
        return float(xBF16[size_t(idx) * C + c]);
    };

    const int ti = int(t);
    const size_t idx = size_t(t) * C + c;
    const bfloat xRaw = xBF16[idx];
    if (writeVerifyTap != 0u) {
        verifyTap[size_t(verifyTapOffset) + idx] = xRaw;
    }
    const float xCur = float(xRaw);
    float acc = w0 * sampleAt(ti - 3) + w1 * sampleAt(ti - 2) + w2 * sampleAt(ti - 1) + w3 * xCur;
    bfloat convBF16 = qw_ink2_bf16((acc + xCur) * invSqrtDenom);

    if (resMode == 0u) {
        outBF16[idx] = convBF16;
    } else if (resMode == 1u) {
        outF32[idx] = float(convBF16);
    } else if (resMode == 2u) {
        outF32[idx] = float(convBF16) + float(resBF16[idx]);
    } else {
        outBF16[idx] = qw_ink2_bf16(resF32[idx] + float(convBF16));
    }

    if (T == 1u && updateState != 0u) {
        stateF32[0u * C + c] = s1;
        stateF32[1u * C + c] = s2;
        stateF32[2u * C + c] = xCur;
    }
}

static inline void qw_ink2_sconv_dynamic_body(
        device const bfloat* xBF16, device const bfloat* staticW,
        device const float* dynWF32, device float* stateF32,
        device bfloat* outBF16, device bfloat* verifyTap,
        uint T, uint C, uint nKV, uint updateState,
        uint writeVerifyTap, uint verifyTapOffset, float invSqrtDenom,
        uint2 gid) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C || t >= T) return;

    const uint headDim = C / nKV;
    const uint kvh = c / headDim;
    const size_t dynBase = (size_t(t) * nKV + kvh) * 4u;
    const float w0 = float(staticW[0u * C + c]) + dynWF32[dynBase + 0u];
    const float w1 = float(staticW[1u * C + c]) + dynWF32[dynBase + 1u];
    const float w2 = float(staticW[2u * C + c]) + dynWF32[dynBase + 2u];
    const float w3 = float(staticW[3u * C + c]) + dynWF32[dynBase + 3u];

    float s0 = stateF32[0u * C + c];
    float s1 = stateF32[1u * C + c];
    float s2 = stateF32[2u * C + c];
    auto sampleAt = [&](int idx) -> float {
        if (idx == -3) return s0;
        if (idx == -2) return s1;
        if (idx == -1) return s2;
        if (idx < -3) return stateF32[size_t(3 + idx) * C + c];
        return float(xBF16[size_t(idx) * C + c]);
    };

    const int ti = int(t);
    const size_t idx = size_t(t) * C + c;
    const bfloat xRaw = xBF16[idx];
    if (writeVerifyTap != 0u) {
        verifyTap[size_t(verifyTapOffset) + idx] = xRaw;
    }
    const float xCur = float(xRaw);
    float acc = w0 * xCur + w1 * sampleAt(ti - 1) + w2 * sampleAt(ti - 2) + w3 * sampleAt(ti - 3);
    outBF16[idx] = qw_ink2_bf16((acc + xCur) * invSqrtDenom);

    if (T == 1u && updateState != 0u) {
        stateF32[0u * C + c] = s1;
        stateF32[1u * C + c] = s2;
        stateF32[2u * C + c] = xCur;
    }
}

kernel void qwg_ink2_sconv_static(device const bfloat* xBF16 [[buffer(0)]],
                                  device const bfloat* weight [[buffer(1)]],
                                  device float* stateF32 [[buffer(2)]],
                                  device const bfloat* resBF16 [[buffer(3)]],
                                  device const float* resF32 [[buffer(4)]],
                                  device bfloat* outBF16 [[buffer(5)]],
                                  device float* outF32 [[buffer(6)]],
                                  constant QWInk2SConvParams& p [[buffer(7)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    qw_ink2_sconv_static_body(xBF16, weight, stateF32, resBF16, resF32, outBF16, outF32,
                              nullptr, p.T, p.C, p.updateState, p.resMode,
                              0u, 0u, p.invSqrtDenom, gid);
}

kernel void qwg_ink2_sconv_dynamic(device const bfloat* xBF16 [[buffer(0)]],
                                   device const bfloat* staticW [[buffer(1)]],
                                   device const float* dynWF32 [[buffer(2)]],
                                   device float* stateF32 [[buffer(3)]],
                                   device bfloat* outBF16 [[buffer(4)]],
                                   constant QWInk2SConvParams& p [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    qw_ink2_sconv_dynamic_body(xBF16, staticW, dynWF32, stateF32, outBF16,
                               nullptr, p.T, p.C, p.nKV, p.updateState,
                               0u, 0u, p.invSqrtDenom, gid);
}

kernel void qwg_ink2_sconv_static_tap(device const bfloat* xBF16 [[buffer(0)]],
                                      device const bfloat* weight [[buffer(1)]],
                                      device float* stateF32 [[buffer(2)]],
                                      device const bfloat* resBF16 [[buffer(3)]],
                                      device const float* resF32 [[buffer(4)]],
                                      device bfloat* outBF16 [[buffer(5)]],
                                      device float* outF32 [[buffer(6)]],
                                      device bfloat* verifyTap [[buffer(7)]],
                                      constant QWInk2SConvTapParams& p [[buffer(8)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    qw_ink2_sconv_static_body(xBF16, weight, stateF32, resBF16, resF32, outBF16, outF32,
                              verifyTap, p.T, p.C, p.updateState, p.resMode,
                              p.writeVerifyTap, p.verifyTapOffset, p.invSqrtDenom, gid);
}

kernel void qwg_ink2_sconv_dynamic_tap(device const bfloat* xBF16 [[buffer(0)]],
                                       device const bfloat* staticW [[buffer(1)]],
                                       device const float* dynWF32 [[buffer(2)]],
                                       device float* stateF32 [[buffer(3)]],
                                       device bfloat* outBF16 [[buffer(4)]],
                                       device bfloat* verifyTap [[buffer(5)]],
                                       constant QWInk2SConvTapParams& p [[buffer(6)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    qw_ink2_sconv_dynamic_body(xBF16, staticW, dynWF32, stateF32, outBF16,
                               verifyTap, p.T, p.C, p.nKV, p.updateState,
                               p.writeVerifyTap, p.verifyTapOffset, p.invSqrtDenom, gid);
}

kernel void qwg_ink2_sconv_commit(device const bfloat* xBF16 [[buffer(0)]],
                                  device float* stateF32 [[buffer(1)]],
                                  constant QWInk2SConvParams& p [[buffer(2)]],
                                  uint c [[thread_position_in_grid]]) {
    if (c >= p.C || p.T == 0u) return;
    float s[3] = {
        stateF32[0u * p.C + c],
        stateF32[1u * p.C + c],
        stateF32[2u * p.C + c]
    };
    const int ti = int(p.T);
    float n0 = (ti >= 3) ? float(xBF16[size_t(ti - 3) * p.C + c]) : s[ti];
    float n1 = (ti >= 2) ? float(xBF16[size_t(ti - 2) * p.C + c]) : s[ti + 1];
    float n2 = float(xBF16[size_t(ti - 1) * p.C + c]);
    stateF32[0u * p.C + c] = n0;
    stateF32[1u * p.C + c] = n1;
    stateF32[2u * p.C + c] = n2;
}

kernel void qwg_ink2_sconv_verify_commit(device float* convStatesSlab [[buffer(0)]],
                                         device const bfloat* verifyInputsSlab [[buffer(1)]],
                                         constant QWInk2SConvVerifyCommitParams& p [[buffer(2)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    const uint flatC = gid.x;
    const uint v = gid.y;
    const uint totalCh = 2u * p.kvDim + 2u * p.H;
    if (v >= p.numVisits || flatC >= totalCh || p.kept == 0u) return;

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
        n0 = float(xIn[size_t(p.kept - 3u) * ch + c]);
        n1 = float(xIn[size_t(p.kept - 2u) * ch + c]);
        n2 = float(xIn[size_t(p.kept - 1u) * ch + c]);
    } else if (p.kept == 2u) {
        n0 = dst[2u * ch + c];
        n1 = float(xIn[c]);
        n2 = float(xIn[ch + c]);
    } else {
        n0 = dst[1u * ch + c];
        n1 = dst[2u * ch + c];
        n2 = float(xIn[c]);
    }
    dst[0u * ch + c] = n0;
    dst[1u * ch + c] = n1;
    dst[2u * ch + c] = n2;
}

static inline float qw_ink2_eval_ar_tail(
        device const bfloat* blocksBuf, device const float* partialF32,
        device const bfloat* partialBF16, constant QWInk2AttnResCtrlNormParams& p,
        threadgroup const float* invRmsSh, threadgroup const float* alphaSh,
        uint nBlocks, uint qIdx, uint t, uint d) {
    float acc = 0.0f;
    for (uint b = 0u; b < nBlocks; ++b) {
        const float v = float(blocksBuf[size_t(p.slotIds[b]) * p.blockStride + size_t(t) * p.H + d]);
        acc += alphaSh[qIdx * 8u + b] * ((p.normalizeValues != 0u) ? (v * invRmsSh[b]) : v);
    }
    const float vp = (p.partialIsF32 != 0u) ? partialF32[size_t(t) * p.H + d]
                                            : float(partialBF16[size_t(t) * p.H + d]);
    return acc + alphaSh[qIdx * 8u + nBlocks] * ((p.normalizeValues != 0u) ? (vp * invRmsSh[nBlocks]) : vp);
}

static inline void qw_ink2_rot32_store(
        device bfloat* outBF16, size_t outBase, uint d, uint H, uint lane,
        uint rotateOut, uint outIsF16, float val) {
#if !defined(QW_BASIS_STANDARD)
    if (rotateOut != 0u) {
        float rx = val;
        #pragma unroll
        for (ushort s = 1u; s < 32u; s <<= 1u) {
            const float pr = simd_shuffle_xor(rx, s);
            rx = ((lane & s) == 0u) ? (rx + pr) : (pr - rx);
        }
        val = rx * 0.17677669529663689f;
    }
#else
    (void)lane;
    (void)rotateOut;
#endif
    if (d < H) {
        if (outIsF16 != 0u) ((device half*)outBF16)[outBase + d] = half(val);
        else outBF16[outBase + d] = qw_ink2_bf16(val);
    }
}

static inline void qw_ink2_reduce_cand(
        float sq, float4 qd, uint numQ, uint H, float eps,
        threadgroup float* sgSq, threadgroup float* sgDot,
        threadgroup float* invRmsSh, threadgroup float* alphaSh,
        uint b, uint simd_id, uint lane) {
    sq = simd_sum(sq);
    if (lane == 0u) sgSq[simd_id] = sq;
    for (uint q = 0u; q < numQ; ++q) {
        const float dq = simd_sum(qd[q]);
        if (lane == 0u) sgDot[q * 8u + simd_id] = dq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        const float vSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
        const float inv = rsqrt(vSq / float(H) + eps);
        const float hScale = rsqrt(float(H));
        if (lane == 0u) invRmsSh[b] = inv;
        for (uint q = 0u; q < numQ; ++q) {
            const float vDot = simd_sum((lane < 8u) ? sgDot[q * 8u + lane] : 0.0f);
            if (lane == 0u) alphaSh[q * 8u + b] = (vDot * inv) * hScale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void qwg_ink2_attn_res_ctrl_norm(device const bfloat* blocksBuf [[buffer(0)]],
                                        device const bfloat* partialBF16 [[buffer(1)]],
                                        device const float* partialF32 [[buffer(2)]],
                                        device const float* queryF32 [[buffer(3)]],
                                        device const float* msgF32 [[buffer(4)]],
                                        device const bfloat* downW [[buffer(5)]],
                                        device const bfloat* forgetW [[buffer(6)]],
                                        device const bfloat* gainW [[buffer(7)]],
                                        device const bfloat* normW [[buffer(8)]],
                                        device bfloat* outBF16 [[buffer(9)]],
                                        constant QWInk2AttnResCtrlNormParams& p [[buffer(10)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint simd_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const uint t = (tg.y > 0u) ? tg.y : tg.x;
    if (t >= p.T) return;
    const uint numQ = min(max(p.numQ, 1u), 4u);
    const uint numOut = min(max(p.numOut, 1u), 4u);
    const uint nBlocks = min(p.nBlocks, 7u);
    const uint nCand = nBlocks + 1u;
    const uint nSteps = min(16u, (p.H + 255u) >> 8u);

    threadgroup float sgSq[8];
    threadgroup float sgDot[32];
    threadgroup float invRmsSh[8];
    threadgroup float alphaSh[32];
    threadgroup bfloat pairSh[8192];
    threadgroup float hidSh[256];

    uint blkReg[7][8];
    float partReg[16];

    for (uint b = 0u; b < nBlocks; ++b) {
        device const bfloat* bp = blocksBuf + size_t(p.slotIds[b]) * p.blockStride + size_t(t) * p.H;
        float sq = 0.0f;
        float4 qd = float4(0.0f);
        for (uint k = 0u; k < 8u; ++k) {
            const uint d0 = tid + (2u * k) * 256u;
            if (d0 >= p.H) break;
            const uint d1 = d0 + 256u;
            const bfloat b0 = bp[d0];
            const bfloat b1 = (d1 < p.H) ? bp[d1] : bfloat(0.0f);
            blkReg[b][k] = uint(as_type<ushort>(b0)) | (uint(as_type<ushort>(b1)) << 16u);
            const float v0 = float(b0);
            sq += v0 * v0;
            for (uint q = 0u; q < numQ; ++q) qd[q] += v0 * queryF32[size_t(q) * p.H + d0];
            if (d1 < p.H) {
                const float v1 = float(b1);
                sq += v1 * v1;
                for (uint q = 0u; q < numQ; ++q) qd[q] += v1 * queryF32[size_t(q) * p.H + d1];
            }
        }
        for (uint d = tid + 4096u; d < p.H; d += 256u) {
            const float v = float(bp[d]);
            sq += v * v;
            for (uint q = 0u; q < numQ; ++q) qd[q] += v * queryF32[size_t(q) * p.H + d];
        }
        qw_ink2_reduce_cand(sq, qd, numQ, p.H, p.attnResEps,
                            sgSq, sgDot, invRmsSh, alphaSh, b, simd_id, lane);
    }

    {
        float sq = 0.0f;
        float4 qd = float4(0.0f);
        device const float* pf = partialF32 + size_t(t) * p.H;
        device const bfloat* pb = partialBF16 + size_t(t) * p.H;
        for (uint k = 0u; k < nSteps; ++k) {
            const uint d = tid + (k << 8u);
            if (d < p.H) {
                const float v = (p.partialIsF32 != 0u) ? pf[d] : float(pb[d]);
                partReg[k] = v;
                sq += v * v;
                for (uint q = 0u; q < numQ; ++q) qd[q] += v * queryF32[size_t(q) * p.H + d];
            }
        }
        for (uint d = tid + 4096u; d < p.H; d += 256u) {
            const float v = (p.partialIsF32 != 0u) ? pf[d] : float(pb[d]);
            sq += v * v;
            for (uint q = 0u; q < numQ; ++q) qd[q] += v * queryF32[size_t(q) * p.H + d];
        }
        qw_ink2_reduce_cand(sq, qd, numQ, p.H, p.attnResEps,
                            sgSq, sgDot, invRmsSh, alphaSh, nBlocks, simd_id, lane);
    }

    if (tid < numQ) {
        float mx = -1e30f;
        for (uint b = 0u; b < nCand; ++b) mx = max(mx, alphaSh[tid * 8u + b]);
        float s = 0.0f;
        for (uint b = 0u; b < nCand; ++b) {
            const float w = exp(alphaSh[tid * 8u + b] - mx);
            alphaSh[tid * 8u + b] = w;
            s += w;
        }
        const float invS = 1.0f / max(s, 1e-30f);
        for (uint b = 0u; b < nCand; ++b) alphaSh[tid * 8u + b] *= invS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint outIdx = 0u; outIdx < numOut; ++outIdx) {
        const uint qIdx = min(outIdx, numQ - 1u);
        float arReg[16];
        float sqAr = 0.0f;
        for (uint k = 0u; k < (nSteps + 1u) / 2u; ++k) {
            const uint d0 = tid + (2u * k) * 256u;
            const uint d1 = d0 + 256u;
            float acc0 = 0.0f, acc1 = 0.0f;
            for (uint b = 0u; b < nBlocks; ++b) {
                const uint pk = blkReg[b][k];
                const float inv = invRmsSh[b];
                const float a = alphaSh[qIdx * 8u + b];
                const float v0 = float(as_type<bfloat>(ushort(pk)));
                const float v1 = float(as_type<bfloat>(ushort(pk >> 16u)));
                acc0 += a * ((p.normalizeValues != 0u) ? (v0 * inv) : v0);
                acc1 += a * ((p.normalizeValues != 0u) ? (v1 * inv) : v1);
            }
            const float invP = invRmsSh[nBlocks];
            const float aP = alphaSh[qIdx * 8u + nBlocks];
            if (d0 < p.H) {
                const float vp0 = partReg[2u * k];
                acc0 += aP * ((p.normalizeValues != 0u) ? (vp0 * invP) : vp0);
                arReg[2u * k] = acc0;
                sqAr += acc0 * acc0;
            }
            if (d1 < p.H) {
                const float vp1 = partReg[2u * k + 1u];
                acc1 += aP * ((p.normalizeValues != 0u) ? (vp1 * invP) : vp1);
                arReg[2u * k + 1u] = acc1;
                sqAr += acc1 * acc1;
            }
        }
        for (uint d = tid + 4096u; d < p.H; d += 256u) {
            const float acc = qw_ink2_eval_ar_tail(blocksBuf, partialF32, partialBF16, p,
                                                   invRmsSh, alphaSh, nBlocks, qIdx, t, d);
            sqAr += acc * acc;
        }

        if (p.useCtrl != 0u && p.H <= 4096u) {
            device const float* mp = msgF32 + size_t(outIdx) * p.msgStride + size_t(t) * p.H;
            float msgReg[16];
            float sqM = 0.0f;
            for (uint k = 0u; k < nSteps; ++k) {
                const uint d = tid + (k << 8u);
                if (d < p.H) { msgReg[k] = mp[d]; sqM += msgReg[k] * msgReg[k]; }
            }
            const float sX = simd_sum(sqAr), sM = simd_sum(sqM);
            if (lane == 0u) { sgSq[simd_id] = sX; sgDot[simd_id] = sM; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_id == 0u) {
                const float vx = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
                const float vm = simd_sum((lane < 8u) ? sgDot[lane] : 0.0f);
                if (lane == 0u) { sgSq[0] = rsqrt(vx / float(p.H) + p.rmsEps); sgDot[0] = rsqrt(vm / float(p.H) + p.rmsEps); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float invX = sgSq[0], invM = sgDot[0];
            for (uint k = 0u; k < nSteps; ++k) {
                const uint d = tid + (k << 8u);
                if (d < p.H) { pairSh[d] = qw_ink2_bf16(arReg[k] * invX); pairSh[p.H + d] = qw_ink2_bf16(msgReg[k] * invM); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint rank = min(p.rank, 256u), twoH = p.H * 2u;
            device const bfloat* dW = downW + p.ctrlDownOffsets[outIdx];
            for (uint r = simd_id; r < rank; r += 8u) {
                device const bfloat* dr = dW + size_t(r) * twoH;
                float acc = 0.0f;
                for (uint j = lane; j < twoH; j += 32u) acc += float(pairSh[j]) * float(dr[j]);
                const float v = simd_sum(acc);
                if (lane == 0u) hidSh[r] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            device const bfloat* fW = forgetW + p.ctrlForgetOffsets[outIdx];
            device const bfloat* gW = gainW + p.ctrlGainOffsets[outIdx];
            sqAr = 0.0f;
            for (uint k = 0u; k < nSteps; ++k) {
                const uint d = tid + (k << 8u);
                if (d < p.H) {
                    device const bfloat* fw = fW + size_t(d) * rank;
                    device const bfloat* gw = gW + size_t(d) * rank;
                    float fDot = 0.0f, gDot = 0.0f;
                    for (uint r = 0u; r < rank; ++r) { const float hr = hidSh[r]; fDot += hr * float(fw[r]); gDot += hr * float(gw[r]); }
                    const float val = (1.0f - qw_ink2_tanh_safe(fDot)) * arReg[k] + qw_ink2_tanh_safe(gDot) * msgReg[k];
                    arReg[k] = val;
                    sqAr += val * val;
                }
            }
        }

        const float sqTail = simd_sum(sqAr);
        if (lane == 0u) sgSq[simd_id] = sqTail;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            const float vSq = simd_sum((lane < 8u) ? sgSq[lane] : 0.0f);
            if (lane == 0u) sgSq[0] = rsqrt(vSq / float(p.H) + p.rmsEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float invNorm = sgSq[0];
        device const bfloat* nw = normW + p.normWOffsets[outIdx];
        const size_t outBase = size_t(outIdx) * p.outStride + size_t(t) * p.H;

        for (uint k = 0u; k < nSteps; ++k) {
            const uint d = tid + (k << 8u);
            const float val = (d < p.H) ? ((arReg[k] * invNorm) * float(nw[d])) : 0.0f;
            qw_ink2_rot32_store(outBF16, outBase, d, p.H, lane, p.rotateOut, p.outIsF16, val);
        }
        for (uint d = tid + 4096u; d < p.H; d += 256u) {
            const float acc = qw_ink2_eval_ar_tail(blocksBuf, partialF32, partialBF16, p,
                                                   invRmsSh, alphaSh, nBlocks, qIdx, t, d);
            qw_ink2_rot32_store(outBF16, outBase, d, p.H, lane, p.rotateOut, p.outIsF16,
                                (acc * invNorm) * float(nw[d]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
