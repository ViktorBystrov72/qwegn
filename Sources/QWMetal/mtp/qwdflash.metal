// ── QWDFlash — DFlash 2 block-diffusion drafter kernels ─────────────────────
// Implements target hidden capture, standard RMSNorm, grouped dynamic causal
// conv, 128-dim RoPE + rotating KV cache, non-causal GQA attention, top-16
// candidate extraction, and sequential pairwise CandidateSelector.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_fastmath_safe.metalh"
using namespace metal;

struct DFlashH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DFlashH8 asDFlashH8(uint4 w) { return *(thread DFlashH8*)&w; }
static inline float dflash_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

// 32-point normalized WHT in FP32 matching fwht32_f32_to_h8 butterflies and scaling.
static inline void dflash_fwht32_f32(thread float (&f)[8], uint lane_id) {
    if (!QW_FWHT_ROTATED) return;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) { float a = f[i], b = f[i + 1]; f[i] = a + b; f[i + 1] = a - b; }
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        float a0 = f[i], a1 = f[i + 1], b0 = f[i + 2], b1 = f[i + 3];
        f[i] = a0 + b0; f[i + 1] = a1 + b1; f[i + 2] = a0 - b0; f[i + 3] = a1 - b1;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) { float a = f[i], b = f[i + 4]; f[i] = a + b; f[i + 4] = a - b; }
    bool neg1 = (lane_id & 1u) != 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { float p = simd_shuffle_xor(f[i], ushort(1)); f[i] = neg1 ? (p - f[i]) : (f[i] + p); }
    bool neg2 = (lane_id & 2u) != 0;
    constexpr float inv_sqrt32 = 0.17677669529663689f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float p = simd_shuffle_xor(f[i], ushort(2));
        f[i] = (neg2 ? (p - f[i]) : (f[i] + p)) * inv_sqrt32;
    }
}

// Standard Qwen3RMSNorm (multiplies by gamma directly, NO +1).
// mode 0/1/2: RMSNorm; mode 3: x+=y -> yRot; mode 4: bridge; mode 5: y + yRot + yRot32.
kernel void qwg_dflash_rmsnorm(
        device void* x               [[buffer(0)]],
        device const uint4* gamma    [[buffer(1)]],
        device uint4* y              [[buffer(2)]],
        device uint4* yRot           [[buffer(3)]],
        constant float& eps          [[buffer(4)]],
        constant uint& mode          [[buffer(5)]],
        constant uint& hidden        [[buffer(6)]],
        device float* yRot32         [[buffer(7)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint hiddenU4 = hidden >> 3u;
    const uint hiddenF4 = hidden >> 2u;
    float acc = 0.0f;
    if (mode == 1u) {
        device const uint4* xr = (device const uint4*)x + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]);
            for (int i = 0; i < 8; i++) {
                float v = float(hv.v[i]);
                acc += v * v;
            }
        }
    } else if (mode == 3u || mode == 4u) {
        device float4* xr = (device float4*)x + (size_t)tg * hiddenF4;
        device const uint4* dr = ((mode == 4u) ? gamma : (device const uint4*)y) + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 dv = asDFlashH8(dr[u]);
            float4 v0 = xr[u * 2u + 0u] + float4(float(dv.v[0]), float(dv.v[1]), float(dv.v[2]), float(dv.v[3]));
            float4 v1 = xr[u * 2u + 1u] + float4(float(dv.v[4]), float(dv.v[5]), float(dv.v[6]), float(dv.v[7]));
            xr[u * 2u + 0u] = v0;
            xr[u * 2u + 1u] = v1;
            acc += dot(v0, v0) + dot(v1, v1);
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u];
            float4 v1 = xr[u * 2u + 1u];
            acc += dot(v0, v0) + dot(v1, v1);
        }
    }
    const float meanSq = simd_sum(acc) * (1.0f / float(hidden));
    device uint4* yr = y + (size_t)tg * hiddenU4;
    device uint4* yrr = yRot + (size_t)tg * hiddenU4;
    if (mode == 4u) {
        const float rms = sqrt(max(meanSq, 1.17549435e-38f));
        if (ti == 0u) ((device float*)yRot)[tg] = rms;
        const float invDenom = 1.0f / (rms + eps);
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u] * invDenom;
            float4 v1 = xr[u * 2u + 1u] * invDenom;
            DFlashH8 ov = {{half(v0.x), half(v0.y), half(v0.z), half(v0.w),
                            half(v1.x), half(v1.y), half(v1.z), half(v1.w)}};
            yr[u] = *(thread uint4*)&ov;
        }
        return;
    }
    float rstd = rsqrt(meanSq + eps);
    if (mode == 1u) {
        device const uint4* xr = (device const uint4*)x + (size_t)tg * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DFlashH8 hv = asDFlashH8(xr[u]), gv = asDFlashH8(gamma[u]), rv;
            float f[8];
            for (int i = 0; i < 8; i++) {
                f[i] = float(hv.v[i]) * rstd * float(gv.v[i]);
            }
            fwht32_f32_to_h8(f, rv.v, ti);
            yr[u] = *(thread uint4*)&rv;
        }
    } else {
        device const float4* xr = (device const float4*)x + (size_t)tg * hiddenF4;
        device float4* y32r = (mode == 5u) ? ((device float4*)yRot32 + (size_t)tg * hiddenF4) : nullptr;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            float4 v0 = xr[u * 2u + 0u];
            float4 v1 = xr[u * 2u + 1u];
            DFlashH8 gv = asDFlashH8(gamma[u]), ov, rv;
            float xv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float f[8];
            for (int i = 0; i < 8; i++) {
                f[i] = xv[i] * rstd * float(gv.v[i]);
                ov.v[i] = half(f[i]);
            }
            if (mode == 0u) {
                yr[u] = *(thread uint4*)&ov;
            } else if (mode == 3u) {
                fwht32_f32_to_h8(f, rv.v, ti);
                yrr[u] = *(thread uint4*)&rv;
            } else if (mode == 5u) {
                yr[u] = *(thread uint4*)&ov;
                dflash_fwht32_f32(f, ti);
                for (int i = 0; i < 8; i++) rv.v[i] = half(f[i]);
                yrr[u] = *(thread uint4*)&rv;
                y32r[u * 2u + 0u] = float4(f[0], f[1], f[2], f[3]);
                y32r[u * 2u + 1u] = float4(f[4], f[5], f[6], f[7]);
            } else {
                yr[u] = *(thread uint4*)&ov;
                fwht32_f32_to_h8(f, rv.v, ti);
                yrr[u] = *(thread uint4*)&rv;
            }
        }
    }
}

// Batched FP16 GEMV for small N (conv kernel_projection, selector, DSpine bridge), M=1..16. Emits FP32 [M, N].
kernel void qwg_dflash_f16_gemv_batched(
        device const uint4* W        [[buffer(0)]],
        device const uint4* x        [[buffer(1)]],
        device float* y              [[buffer(2)]],
        constant uint& N             [[buffer(3)]],
        constant uint& K             [[buffer(4)]],
        constant uint& M             [[buffer(5)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint row = tg * 4u + simd_id;
    if (row >= N) return;
    const uint chunks = K >> 3u;
    device const uint4* wr = W + (size_t)row * chunks;
    const uint safeM = min(M, 16u);
    float acc[16] = {0.0f};
    for (uint c = lane; c < chunks; c += 32u) {
        DFlashH8 wv = asDFlashH8(wr[c]);
        for (uint m = 0; m < safeM; ++m) {
            DFlashH8 xv = asDFlashH8(x[(size_t)m * chunks + c]);
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += float(wv.v[i]) * float(xv.v[i]);
            acc[m] += s;
        }
    }
    for (uint m = 0; m < safeM; ++m) {
        float val = simd_sum(acc[m]);
        if (lane == 0u) y[(size_t)m * N + row] = val;
    }
}

// GroupedDynamicCausalConv (kernel_size=2): mode 0 = fwht32 out [M, H]; mode 1 = f32 residual += conv.
kernel void qwg_dflash_conv_apply(
        device const uint4* x          [[buffer(0)]],
        device const float* dyn        [[buffer(1)]],
        device const uint4* baseKernel [[buffer(2)]],
        device void* outOrResidual     [[buffer(3)]],
        constant uint& mode            [[buffer(4)]],
        constant uint& hiddenU4        [[buffer(5)]],
        constant uint& convGroups      [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    if (convGroups == 0u) return;
    const uint t = tg;
    const uint hiddenF4 = hiddenU4 << 1u;
    const uint dynStride = convGroups * 4u;
    const uint uPerGroup = max(1u, (hiddenU4 + convGroups - 1u) / convGroups);
    device const uint4* xCur = x + (size_t)t * hiddenU4;
    device const uint4* xPrev = (t > 0u) ? (x + (size_t)(t - 1u) * hiddenU4) : xCur;
    device const uint4* bTap0 = baseKernel + (size_t)(mode * 2u + 0u) * hiddenU4;
    device const uint4* bTap1 = baseKernel + (size_t)(mode * 2u + 1u) * hiddenU4;
    device const float* dTap0 = dyn + (size_t)t * dynStride + (size_t)mode * (convGroups * 2u);
    device const float* dTap1 = dTap0 + convGroups;

    for (uint u = ti; u < hiddenU4; u += 32u) {
        const uint g = min(u / uPerGroup, convGroups - 1u);
        const float dw0 = dTap0[g];
        const float dw1 = dTap1[g];
        DFlashH8 xc = asDFlashH8(xCur[u]);
        DFlashH8 xp = (t > 0u) ? asDFlashH8(xPrev[u]) : DFlashH8{{0, 0, 0, 0, 0, 0, 0, 0}};
        DFlashH8 b0 = asDFlashH8(bTap0[u]);
        DFlashH8 b1 = asDFlashH8(bTap1[u]);
        float f[8];
        for (int i = 0; i < 8; ++i) {
            float w0 = float(b0.v[i]) + dw0;
            float w1 = float(b1.v[i]) + dw1;
            float v = float(xc.v[i]) * w0 + ((t > 0u) ? (float(xp.v[i]) * w1) : 0.0f);
            f[i] = v;
        }
        if (mode == 0u) {
            device uint4* dst16 = (device uint4*)outOrResidual + (size_t)t * hiddenU4;
            DFlashH8 rot;
            fwht32_f32_to_h8(f, rot.v, ti);
            dst16[u] = *(thread uint4*)&rot;
        } else {
            device float4* dst32 = (device float4*)outOrResidual + (size_t)t * hiddenF4;
            dst32[u * 2u + 0u] += float4(f[0], f[1], f[2], f[3]);
            dst32[u * 2u + 1u] += float4(f[4], f[5], f[6], f[7]);
        }
    }
}

// Context K norm + 128-dim RoPE + rotating KV cache write.
// Grid = (S, kvHeads, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_kv_ctx(
        device const half* kvCtxIn   [[buffer(0)]],
        device const half* wk        [[buffer(1)]],
        device const float* invFreq  [[buffer(2)]],
        device half* kCache          [[buffer(3)]],
        device half* vCache          [[buffer(4)]],
        constant uint& cacheOffset   [[buffer(5)]],
        constant float& eps          [[buffer(6)]],
        constant uint& maxCacheSize  [[buffer(7)]],
        constant uint& kvHeads       [[buffer(8)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const uint h = tg.y;
    const uint pos = cacheOffset + s;
    const uint slot = pos % maxCacheSize;
    const uint kvDim = kvHeads * 128u;
    const uint kvInStride = kvDim * 2u;
    device const half* kh = kvCtxIn + (size_t)s * kvInStride + h * 128u;
    device const half* vh = kvCtxIn + (size_t)s * kvInStride + kvDim + h * 128u;
    device half* kDst = kCache + (size_t)slot * kvDim + h * 128u;
    device half* vDst = vCache + (size_t)slot * kvDim + h * 128u;

    float x0 = float(kh[lane]);
    float x1 = float(kh[lane + 32u]);
    float x2 = float(kh[lane + 64u]);
    float x3 = float(kh[lane + 96u]);
    float acc = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    float rstd = rsqrt(simd_sum(acc) * (1.0f / 128.0f) + eps);

    float n0 = x0 * rstd * float(wk[lane]);
    float n1 = x1 * rstd * float(wk[lane + 32u]);
    float n2 = x2 * rstd * float(wk[lane + 64u]);
    float n3 = x3 * rstd * float(wk[lane + 96u]);

    float a0 = float(pos) * invFreq[lane];
    float a1 = float(pos) * invFreq[lane + 32u];
    float c0 = cos(a0), s0 = sin(a0);
    float c1 = cos(a1), s1 = sin(a1);

    kDst[lane]       = half(n0 * c0 - n2 * s0);
    kDst[lane + 32u] = half(n1 * c1 - n3 * s1);
    kDst[lane + 64u] = half(n2 * c0 + n0 * s0);
    kDst[lane + 96u] = half(n3 * c1 + n1 * s1);

    vDst[lane]       = vh[lane];
    vDst[lane + 32u] = vh[lane + 32u];
    vDst[lane + 64u] = vh[lane + 64u];
    vDst[lane + 96u] = vh[lane + 96u];
}

// Proposal Q (qHeads) & K (kvHeads) norm + 128-dim RoPE in-place on qkvProp.
// Grid = (M, qHeads + kvHeads, 1), threadsPerThreadgroup = (32, 1, 1).
kernel void qwg_dflash_rope_qk_prop(
        device half* qkvProp         [[buffer(0)]],
        device const half* wq        [[buffer(1)]],
        device const half* wk        [[buffer(2)]],
        device const float* invFreq  [[buffer(3)]],
        constant uint& propOffset    [[buffer(4)]],
        constant float& eps          [[buffer(5)]],
        constant uint& qHeads        [[buffer(6)]],
        constant uint& kvHeads       [[buffer(7)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint m = tg.x;
    const bool isQ = tg.y < qHeads;
    const uint h = isQ ? tg.y : (tg.y - qHeads);
    const uint pos = propOffset + m;
    const uint qDim = qHeads * 128u;
    const uint kvDim = kvHeads * 128u;
    const uint qkvStride = qDim + 2u * kvDim;
    device half* vec = isQ ? (qkvProp + (size_t)m * qkvStride + h * 128u)
                           : (qkvProp + (size_t)m * qkvStride + qDim + h * 128u);
    device const half* w = isQ ? wq : wk;

    float x0 = float(vec[lane]);
    float x1 = float(vec[lane + 32u]);
    float x2 = float(vec[lane + 64u]);
    float x3 = float(vec[lane + 96u]);
    float acc = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    float rstd = rsqrt(simd_sum(acc) * (1.0f / 128.0f) + eps);

    float n0 = x0 * rstd * float(w[lane]);
    float n1 = x1 * rstd * float(w[lane + 32u]);
    float n2 = x2 * rstd * float(w[lane + 64u]);
    float n3 = x3 * rstd * float(w[lane + 96u]);

    float a0 = float(pos) * invFreq[lane];
    float a1 = float(pos) * invFreq[lane + 32u];
    float c0 = cos(a0), s0 = sin(a0);
    float c1 = cos(a1), s1 = sin(a1);

    vec[lane]       = half(n0 * c0 - n2 * s0);
    vec[lane + 32u] = half(n1 * c1 - n3 * s1);
    vec[lane + 64u] = half(n2 * c0 + n0 * s0);
    vec[lane + 96u] = half(n3 * c1 + n1 * s1);
}

// Non-causal (DFlash2) or causal (DSpine) within-block + sliding-window context GQA attention.
// Grid = (qHeads, M, 1), threadsPerThreadgroup = (32, 1, 1). Rotates output via simd_fwht32.
kernel void qwg_dflash_attn(
        device const half* qkvProp   [[buffer(0)]],
        device const half* kCache    [[buffer(1)]],
        device const half* vCache    [[buffer(2)]],
        device half* out             [[buffer(3)]],
        constant uint& newOffset     [[buffer(4)]],
        constant uint& M             [[buffer(5)]],
        constant uint& maxCacheSize  [[buffer(6)]],
        constant uint& isCausal      [[buffer(7)]],
        constant uint& qHeads        [[buffer(8)]],
        constant uint& kvHeads       [[buffer(9)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    const uint qh = tg.x;
    const uint m = tg.y;
    const uint gqaRatio = max(1u, qHeads / kvHeads);
    const uint kvh = qh / gqaRatio;
    const uint qDim = qHeads * 128u;
    const uint kvDim = kvHeads * 128u;
    const uint qkvStride = qDim + 2u * kvDim;
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    device const half* qv = qkvProp + (size_t)m * qkvStride + qh * 128u;
    const float q0 = float(qv[lane]) * scale;
    const float q1 = float(qv[lane + 32u]) * scale;
    const float q2 = float(qv[lane + 64u]) * scale;
    const float q3 = float(qv[lane + 96u]) * scale;

    float maxLogit = -3.4e38f;
    bool hasLogit = false;
    float lsum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint absQ = newOffset + m;
    const uint startK = (absQ > maxCacheSize) ? (absQ - maxCacheSize) : 0u;
    for (uint absK = startK; absK < newOffset; ++absK) {
        const uint slot = absK % maxCacheSize;
        device const half* kv = kCache + (size_t)slot * kvDim + kvh * 128u;
        device const half* vv = vCache + (size_t)slot * kvDim + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = hasLogit ? max(maxLogit, s) : s;
        float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f;
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
        hasLogit = true;
    }

    const uint pLimit = (isCausal != 0u) ? (m + 1u) : M;
    for (uint pIdx = 0u; pIdx < pLimit; ++pIdx) {
        device const half* kv = qkvProp + (size_t)pIdx * qkvStride + qDim + kvh * 128u;
        device const half* vv = qkvProp + (size_t)pIdx * qkvStride + qDim + kvDim + kvh * 128u;
        float dot = q0 * float(kv[lane]) + q1 * float(kv[lane + 32u])
                  + q2 * float(kv[lane + 64u]) + q3 * float(kv[lane + 96u]);
        float s = simd_sum(dot);
        float newMax = hasLogit ? max(maxLogit, s) : s;
        float corr = hasLogit ? exp(maxLogit - newMax) : 0.0f;
        float p = exp(s - newMax);
        lsum = lsum * corr + p;
        o0 = o0 * corr + p * float(vv[lane]);
        o1 = o1 * corr + p * float(vv[lane + 32u]);
        o2 = o2 * corr + p * float(vv[lane + 64u]);
        o3 = o3 * corr + p * float(vv[lane + 96u]);
        maxLogit = newMax;
        hasLogit = true;
    }

    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device half* dst = out + (size_t)m * qDim + qh * 128u;
    dst[lane]       = simd_fwht32(half(o0 * invL), lane);
    dst[lane + 32u] = simd_fwht32(half(o1 * invL), lane);
    dst[lane + 64u] = simd_fwht32(half(o2 * invL), lane);
    dst[lane + 96u] = simd_fwht32(half(o3 * invL), lane);
}

static inline void dflash_tg_argmax(
        float localVal, int localId, uint ti,
        threadgroup float* sVal, threadgroup int* sId,
        thread float& outVal, thread int& outId) {
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(localVal, offset);
        int otherId = simd_shuffle_down(localId, offset);
        if (otherVal > localVal || (otherVal == localVal && otherId < localId)) {
            localVal = otherVal;
            localId = otherId;
        }
    }
    if (lane == 0u) {
        sVal[simd_id] = localVal;
        sId[simd_id] = localId;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 8u) ? sVal[lane] : -3.4e38f;
        int id = (lane < 8u) ? sId[lane] : 0x7fffffff;
        for (ushort offset = 4; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset);
            int oid = simd_shuffle_down(id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (lane == 0u) {
            sVal[0] = v;
            sId[0] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outVal = sVal[0];
    outId = sId[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Top-16 Stage 1 over vocab logits per draft row (fast register path <= 393216, general loop above).
// Grid = (256, D, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage1(
        device const float* logits   [[buffer(0)]],
        device float* partialVals    [[buffer(1)]],
        device int* partialIds       [[buffer(2)]],
        constant uint& vocab         [[buffer(3)]],
        constant uint& topK          [[buffer(4)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint chunk = tg.x;
    const uint row = tg.y;
    const uint safeTopK = min(topK, 16u);
    const uint outBase = (row * 256u + chunk) * 16u;
    if (vocab == 0u) {
        if (ti < safeTopK) {
            partialVals[outBase + ti] = -3.4e38f;
            partialIds[outBase + ti] = 0x7fffffff;
        }
        return;
    }
    const uint chunkStride = (vocab + 255u) >> 8u;
    const uint base = chunk * chunkStride;
    const uint chunkEnd = min(vocab, base + chunkStride);
    device const float* rowLogits = logits + (size_t)row * vocab;

    threadgroup float sVal[8];
    threadgroup int sId[8];

    if (chunkStride <= 1536u) {
        float regVal[6];
        int regId[6];
        #pragma unroll
        for (uint i = 0; i < 6u; ++i) {
            uint idx = base + i * 256u + ti;
            if (idx < chunkEnd) {
                float v = rowLogits[idx];
                regVal[i] = qw_is_finite_bits(v) ? max(v, -1e30f) : -1e30f;
                regId[i] = int(idx);
            } else {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        }

        for (uint k = 0; k < safeTopK; ++k) {
            float bestLocalVal = -3.4e38f;
            int bestLocalId = 0x7fffffff;
            #pragma unroll
            for (uint i = 0; i < 6u; ++i) {
                if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                    bestLocalVal = regVal[i];
                    bestLocalId = regId[i];
                }
            }
            float winVal;
            int winId;
            dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
            if (ti == 0u) {
                partialVals[outBase + k] = winVal;
                partialIds[outBase + k] = winId;
            }
            #pragma unroll
            for (uint i = 0; i < 6u; ++i) {
                if (regId[i] == winId) {
                    regVal[i] = -3.4e38f;
                    regId[i] = 0x7fffffff;
                }
            }
        }
        return;
    }

    int chosenIds[16];
    for (uint k = 0; k < safeTopK; ++k) {
        float bestLocalVal = -3.4e38f;
        int bestLocalId = 0x7fffffff;
        for (uint idx = base + ti; idx < chunkEnd; idx += 256u) {
            bool already = false;
            for (uint p = 0; p < k; ++p) {
                if (chosenIds[p] == int(idx)) { already = true; break; }
            }
            if (already) continue;
            float raw = rowLogits[idx];
            float v = qw_is_finite_bits(raw) ? max(raw, -1e30f) : -1e30f;
            if (v > bestLocalVal || (v == bestLocalVal && int(idx) < bestLocalId)) {
                bestLocalVal = v;
                bestLocalId = int(idx);
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        chosenIds[k] = winId;
        if (ti == 0u) {
            partialVals[outBase + k] = winVal;
            partialIds[outBase + k] = winId;
        }
    }
}

// Kernel 9: Top-16 Stage 2 reducing 256*16 partial candidates per draft row to topK.
// Grid = (D, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_top16_stage2(
        device const float* partialVals [[buffer(0)]],
        device const int* partialIds    [[buffer(1)]],
        device float* unaryOut          [[buffer(2)]],
        device int* candidatesOut       [[buffer(3)]],
        constant uint& topK             [[buffer(4)]],
        constant uint& vocab            [[buffer(5)]],
        uint tg [[threadgroup_position_in_grid]],
        uint ti [[thread_index_in_threadgroup]]) {
    const uint row = tg;
    const uint inBase = row * 256u * 16u;
    const uint safeTopK = min(topK, 16u);

    float regVal[16];
    int regId[16];
    for (uint i = 0; i < 16u; ++i) {
        if (i < safeTopK) {
            float pv = partialVals[inBase + ti * 16u + i];
            int pid = partialIds[inBase + ti * 16u + i];
            if (pid >= 0 && pid < int(vocab) && pid != 0x7fffffff) {
                regVal[i] = qw_is_finite_bits(pv) ? max(pv, -1e30f) : -1e30f;
                regId[i] = pid;
            } else {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        } else {
            regVal[i] = -3.4e38f;
            regId[i] = 0x7fffffff;
        }
    }

    threadgroup float sVal[8];
    threadgroup int sId[8];
    const uint outBase = row * 16u;

    for (uint k = 0; k < safeTopK; ++k) {
        float bestLocalVal = -3.4e38f;
        int bestLocalId = 0x7fffffff;
        for (uint i = 0; i < 16u; ++i) {
            if (regVal[i] > bestLocalVal || (regVal[i] == bestLocalVal && regId[i] < bestLocalId)) {
                bestLocalVal = regVal[i];
                bestLocalId = regId[i];
            }
        }
        float winVal;
        int winId;
        dflash_tg_argmax(bestLocalVal, bestLocalId, ti, sVal, sId, winVal, winId);
        if (ti == 0u) {
            bool valid = (winId >= 0 && winId < int(vocab) && winId != 0x7fffffff);
            unaryOut[outBase + k] = valid ? (qw_is_finite_bits(winVal) ? max(winVal, -1e30f) : -1e30f) : -1e30f;
            candidatesOut[outBase + k] = valid ? winId : 0;
        }
        for (uint i = 0; i < 16u; ++i) {
            if (regId[i] == winId) {
                regVal[i] = -3.4e38f;
                regId[i] = 0x7fffffff;
            }
        }
    }
}

// Greedy sequential pairwise CandidateSelector over BF16 codebooks [vocab, 256].
// Grid = (1, 1, 1), threadsPerThreadgroup = (256, 1, 1).
kernel void qwg_dflash_selector_greedy(
        device const float* hiddenProj   [[buffer(0)]],
        device const float* unary        [[buffer(1)]],
        device const int* candidates     [[buffer(2)]],
        device const ushort* predBook    [[buffer(3)]],
        device const ushort* succBook    [[buffer(4)]],
        device int* draftTokensOut       [[buffer(5)]],
        constant int& anchorToken        [[buffer(6)]],
        constant uint& draftCount        [[buffer(7)]],
        constant uint& topK              [[buffer(8)]],
        constant uint& vocab             [[buffer(9)]],
        uint ti [[thread_index_in_threadgroup]]) {
    if (vocab == 0u) {
        if (ti == 0u) {
            for (uint pos = 0; pos < draftCount; ++pos) draftTokensOut[pos] = 0;
        }
        return;
    }
    const uint simd_id = ti >> 5u;
    const uint lane = ti & 31u;
    const uint safeTopK = min(topK, 16u);
    threadgroup float sSums[8];
    threadgroup int sWinner;

    int pred = (anchorToken >= 0 && anchorToken < int(vocab) && anchorToken != 0x7fffffff) ? anchorToken : 0;
    for (uint pos = 0; pos < draftCount; ++pos) {
        float hVal = hiddenProj[pos * 256u + ti];
        float pVal = dflash_bf16_to_f32(predBook[(size_t)pred * 256u + ti]) * hVal;
        float bestScore = -3.4e38f;
        int bestToken = 0;

        for (uint k = 0; k < safeTopK; ++k) {
            int rawCand = candidates[pos * 16u + k];
            bool validCand = (rawCand >= 0 && rawCand < int(vocab) && rawCand != 0x7fffffff);
            int cand = validCand ? rawCand : 0;
            float sVal = dflash_bf16_to_f32(succBook[(size_t)cand * 256u + ti]);
            float prod = simd_sum(pVal * sVal);
            if (lane == 0u) sSums[simd_id] = prod;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ti == 0u && validCand) {
                float edge = 0.0f;
                for (uint s = 0; s < 8u; ++s) edge += sSums[s];
                float u = unary[pos * 16u + k];
                float rawScore = u + edge;
                float score = (qw_is_finite_bits(u) && qw_is_finite_bits(edge) && qw_is_finite_bits(rawScore)) ? max(rawScore, -1e30f) : -1e30f;
                if (score > bestScore) {
                    bestScore = score;
                    bestToken = cand;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ti == 0u) {
            sWinner = bestToken;
            draftTokensOut[pos] = bestToken;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pred = sWinner;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
