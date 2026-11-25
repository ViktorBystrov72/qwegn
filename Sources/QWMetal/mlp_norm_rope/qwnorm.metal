// ── C06 QWNorm — RMSNorm family (spec specs/06-qwnorm.md) ──────────────────
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_geometry_constants.metalh" // runtime-specialized geometry
#include "qw_dequant.metalh"            // fwht32_inplace register butterfly
using namespace metal;

// Register-tile size for the fused residual kernel: 5120/8 uint4 ÷ 32 lanes.
// When hidden == 5120 → 640/32 = 20 uint4/lane, fully unrolled, register-resident.
// The Swift encode() refuses any geometry where (hidden/8) % 32 != 0.
#define QW_RES2_U4_PER_LANE 20u
#define QW_RES2_U4_TOTAL (QW_RES2_U4_PER_LANE * 32u)

// uint4 == 8 halves; H8 reinterprets it for element access in registers.
struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline H8 asH8(uint4 w) { return *(thread H8*)&w; }

// PLUS-ONE law (official Qwen3_5RMSNorm): gamma is stored raw (w≈0);
// every TEXT norm multiplies by (1+w) in f32 and rotates into H_32 in f32
// before a single f16 rounding at store time.
// All variants: fp32 sumsq accumulation (never f16 — overflow law), one row
// per 32-thread simdgroup, uint4 (=8×f16) vector loads, butterfly reduce.
// eps = QWModelGeometry.rmsEps (1e-6) passed in by the launcher.
//   y = fwht32(x · rsqrt(mean(x²) + eps) · (1 + gamma))

// variant 1: standalone rmsnorm. buffers: 0 x | 1 gamma | 2 y | 3 hidden u32
kernel void qwg_rmsnorm(device const uint4* x            [[buffer(0)]],
                        device const uint4* gamma        [[buffer(1)]],
                        device uint4* y                  [[buffer(2)]],
                        constant uint& hiddenU4          [[buffer(3)]],   // hidden/8
                        constant float& eps              [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint ti [[thread_index_in_threadgroup]]) {
    device const uint4* xr = x + (size_t)tg * hiddenU4;
    device uint4* yr = y + (size_t)tg * hiddenU4;
    if (hiddenU4 <= QW_RES2_U4_TOTAL) {
        float acc = 0.0f;
        uint4 reg[QW_RES2_U4_PER_LANE];
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            uint4 xv = xr[u];
            reg[k] = xv;
            H8 hv = asH8(xv);
            for (int i = 0; i < 8; i++) { float v = float(hv.v[i]); acc += v * v; }
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            H8 hv = asH8(reg[k]), gv = asH8(gamma[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
        }
        return;
    }
    float acc = 0.0f;
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]);
        for (int i = 0; i < 8; i++) { float v = float(hv.v[i]); acc += v * v; }
    }
    float total = simd_sum(acc);
    float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]), gv = asH8(gamma[u]), ov;
        float f[8];
        for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
        fwht32_f32_to_h8(f, ov.v, ti);
        yr[u] = *(thread uint4*)&ov;
    }
}

// variant 1b: standalone rmsnorm when input x is ALREADY in rotated (H_32) basis
// (used by MTP preFcNormHidden consuming trunk finalNorm / previous draft postNorm).
// Un-rotates x in registers via fwht32_inplace (since H_32^-1 = H_32), applies
kernel void qwg_rmsnorm_from_rot(device const uint4* x            [[buffer(0)]],
                                 device const uint4* gamma        [[buffer(1)]],
                                 device uint4* y                  [[buffer(2)]],
                                 constant uint& hiddenU4          [[buffer(3)]],
                                 constant float& eps              [[buffer(4)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint ti [[thread_index_in_threadgroup]]) {
    device const uint4* xr = x + (size_t)tg * hiddenU4;
    float acc = 0.0f;
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]);
        fwht32_inplace(hv.v, ti);
        for (int i = 0; i < 8; i++) { float v = float(hv.v[i]); acc += v * v; }
    }
    float total = simd_sum(acc);
    float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
    device uint4* yr = y + (size_t)tg * hiddenU4;
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]), gv = asH8(gamma[u]), ov;
        fwht32_inplace(hv.v, ti);
        float f[8];
        for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
        fwht32_f32_to_h8(f, ov.v, ti);
        yr[u] = *(thread uint4*)&ov;
    }
}

METAL_FUNC void qwg_norm_emit_moe_bridge(thread const half (&hv)[8],
                                         device float* rotRow,
                                         device float* plainRow,
                                         uint u,
                                         uint laneId,
                                         bool skipRot32) {
    float f[8];
    for (int i = 0; i < 8; i++) {
        f[i] = float(hv[i]);
        if (!skipRot32) rotRow[u * 8u + i] = f[i];
    }
    if (QW_FWHT_ROTATED) {
        half hi[8], lo[8];
        fwht32_f32_to_split_h8(f, hi, lo, laneId);
        for (int i = 0; i < 8; i++) plainRow[u * 8u + i] = float(hi[i]) + float(lo[i]);
    } else {
        for (int i = 0; i < 8; i++) plainRow[u * 8u + i] = f[i];
    }
}

template <bool EMIT_MOE>
static inline void qwg_rmsnorm_residual_impl(device const uint4* x,
                                             device const uint4* gamma,
                                             device uint4* y,
                                             uint hiddenU4Packed,
                                             float eps,
                                             device uint4* residual,
                                             device float* xRot32,
                                             device float* xPlain32,
                                             uint tg,
                                             uint ti) {
    const bool skipRot32 = EMIT_MOE && ((hiddenU4Packed & 0x80000000u) != 0u);
    const uint hiddenU4 = hiddenU4Packed & 0x7FFFFFFFu;
    const size_t realH = (size_t)hiddenU4 * 8u;
    device const uint4* xr = x + (size_t)tg * hiddenU4;
    device uint4* rr = residual + (size_t)tg * hiddenU4;
    device uint4* yr = y + (size_t)tg * hiddenU4;
    device float* rotRow = EMIT_MOE ? (xRot32 + (size_t)tg * realH) : nullptr;
    device float* plainRow = EMIT_MOE ? (xPlain32 + (size_t)tg * realH) : nullptr;

    if (hiddenU4 <= QW_RES2_U4_TOTAL) {
        float acc = 0.0f;
        uint4 reg[QW_RES2_U4_PER_LANE];
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            uint4 xv = xr[u], rv0 = rr[u];
            H8 av = asH8(xv), bv = asH8(rv0), sv;
            for (int i = 0; i < 8; i++) { sv.v[i] = half(float(av.v[i]) + float(bv.v[i])); float t = float(sv.v[i]); acc += t * t; }
            rr[u] = *(thread uint4*)&sv;
            reg[k] = *(thread uint4*)&sv;
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            H8 bv = asH8(reg[k]), gv = asH8(gamma[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(bv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
            if (EMIT_MOE) qwg_norm_emit_moe_bridge(ov.v, rotRow, plainRow, u, ti, skipRot32);
        }
    } else {
        float acc = 0.0f;
        for (uint u = ti; u < hiddenU4; u += 32) {
            uint4 xv = xr[u], rv0 = rr[u];
            H8 av = asH8(xv), bv = asH8(rv0), sv;
            for (int i = 0; i < 8; i++) { sv.v[i] = half(float(av.v[i]) + float(bv.v[i])); float t = float(sv.v[i]); acc += t * t; }
            rr[u] = *(thread uint4*)&sv;
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
        for (uint u = ti; u < hiddenU4; u += 32) {
            H8 bv = asH8(rr[u]), gv = asH8(gamma[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(bv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
            if (EMIT_MOE) qwg_norm_emit_moe_bridge(ov.v, rotRow, plainRow, u, ti, skipRot32);
        }
    }
}

template <bool EMIT_MOE>
static inline void qwg_rmsnorm_residual_wide_impl(device const uint4* x,
                                                  device const uint4* gamma,
                                                  device uint4* y,
                                                  uint hiddenU4Packed,
                                                  float eps,
                                                  device uint4* residual,
                                                  device float* xRot32,
                                                  device float* xPlain32,
                                                  uint tg,
                                                  uint ti,
                                                  threadgroup uint4* sx,
                                                  threadgroup float* rs) {
    const bool skipRot32 = EMIT_MOE && ((hiddenU4Packed & 0x80000000u) != 0u);
    const uint hiddenU4 = hiddenU4Packed & 0x7FFFFFFFu;
    const size_t realH = (size_t)hiddenU4 * 8u;
    const size_t rowBase = (size_t)tg * hiddenU4;
    x += rowBase;
    residual += rowBase;
    y += rowBase;
    const uint u = ti;
    const uint4 gw = gamma[u];
    H8 av = asH8(x[u]), bv = asH8(residual[u]), sv;
    for (int i = 0; i < 8; i++) sv.v[i] = half(float(av.v[i]) + float(bv.v[i]));
    const uint4 sw = *(thread uint4*)&sv;
    residual[u] = sw;
    sx[u] = sw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti < 32u) {
        float acc = 0.0f;
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint v = ti + 32u * k;
            if (v >= hiddenU4) break;
            H8 hv = asH8(sx[v]);
            for (int i = 0; i < 8; i++) { float t = float(hv.v[i]); acc += t * t; }
        }
        float total = simd_sum(acc);
        if (ti == 0u) rs[0] = rsqrt(total / float(hiddenU4 * 8) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rstd = rs[0];
    H8 gv = asH8(gw), ov;
    float f[8];
    for (int i = 0; i < 8; i++) f[i] = float(sv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
    fwht32_f32_to_h8(f, ov.v, ti);
    y[u] = *(thread uint4*)&ov;
    if (EMIT_MOE) {
        qwg_norm_emit_moe_bridge(ov.v, xRot32 + (size_t)tg * realH, xPlain32 + (size_t)tg * realH,
                                 u, ti & 31u, skipRot32);
    }
}

// variant 2 (default shipping): fused residual. residual += x; y = fwht32(rmsnorm(residual)·γ)
kernel void qwg_rmsnorm_residual(device const uint4* x       [[buffer(0)]],
                                 device const uint4* gamma   [[buffer(1)]],
                                 device uint4* y             [[buffer(2)]],
                                 constant uint& hiddenU4     [[buffer(3)]],
                                 constant float& eps         [[buffer(4)]],
                                 device uint4* residual      [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint ti [[thread_index_in_threadgroup]]) {
    qwg_rmsnorm_residual_impl<false>(x, gamma, y, hiddenU4, eps, residual, nullptr, nullptr, tg, ti);
}

// variant 2m: variant 2 with fused f32 MoE bridge outputs (xRot32, xPlain32).
kernel void qwg_rmsnorm_residual_moe(device const uint4* x       [[buffer(0)]],
                                     device const uint4* gamma   [[buffer(1)]],
                                     device uint4* y             [[buffer(2)]],
                                     constant uint& hiddenU4     [[buffer(3)]],
                                     constant float& eps         [[buffer(4)]],
                                     device uint4* residual      [[buffer(5)]],
                                     device float* xRot32        [[buffer(6)]],
                                     device float* xPlain32      [[buffer(7)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint ti [[thread_index_in_threadgroup]]) {
    qwg_rmsnorm_residual_impl<true>(x, gamma, y, hiddenU4, eps, residual, xRot32, xPlain32, tg, ti);
}

// variant 2w: variant 2 with one threadgroup per row and one thread per uint4 (decode/verify).
kernel void qwg_rmsnorm_residual_wide(device const uint4* x       [[buffer(0)]],
                                      device const uint4* gamma   [[buffer(1)]],
                                      device uint4* y             [[buffer(2)]],
                                      constant uint& hiddenU4     [[buffer(3)]],
                                      constant float& eps         [[buffer(4)]],
                                      device uint4* residual      [[buffer(5)]],
                                      uint tg [[threadgroup_position_in_grid]],
                                      uint ti [[thread_index_in_threadgroup]]) {
    threadgroup uint4 sx[QW_RES2_U4_TOTAL];
    threadgroup float rs[1];
    qwg_rmsnorm_residual_wide_impl<false>(x, gamma, y, hiddenU4, eps, residual, nullptr, nullptr, tg, ti, sx, rs);
}

// variant 2wm: wide residual norm + fused f32 MoE bridge outputs (0-ULP parity with separate bridge).
kernel void qwg_rmsnorm_residual_moe_wide(device const uint4* x       [[buffer(0)]],
                                          device const uint4* gamma   [[buffer(1)]],
                                          device uint4* y             [[buffer(2)]],
                                          constant uint& hiddenU4     [[buffer(3)]],
                                          constant float& eps         [[buffer(4)]],
                                          device uint4* residual      [[buffer(5)]],
                                          device float* xRot32        [[buffer(6)]],
                                          device float* xPlain32      [[buffer(7)]],
                                          uint tg [[threadgroup_position_in_grid]],
                                          uint ti [[thread_index_in_threadgroup]]) {
    threadgroup uint4 sx[QW_RES2_U4_TOTAL];
    threadgroup float rs[1];
    qwg_rmsnorm_residual_wide_impl<true>(x, gamma, y, hiddenU4, eps, residual, xRot32, xPlain32, tg, ti, sx, rs);
}

// variant 3 (P3 race candidate): norm + write row-scale side vector instead of
// scaling into y — the GEMM prologue applies rstd during tile load.
// y = fwht32(x·γ) (unscaled by rstd), rowScales[row] = rstd.
kernel void qwg_rmsnorm_prologue_scale(device const uint4* x    [[buffer(0)]],
                                       device const uint4* gamma [[buffer(1)]],
                                       device uint4* y          [[buffer(2)]],
                                       constant uint& hiddenU4  [[buffer(3)]],
                                       constant float& eps      [[buffer(4)]],
                                       device float* rowScales  [[buffer(5)]],
                                       uint tg [[threadgroup_position_in_grid]],
                                       uint ti [[thread_index_in_threadgroup]]) {
    device const uint4* xr = x + (size_t)tg * hiddenU4;
    device uint4* yr = y + (size_t)tg * hiddenU4;
    float acc = 0.0f;
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]);
        for (int i = 0; i < 8; i++) { float t = float(hv.v[i]); acc += t * t; }
    }
    float total = simd_sum(acc);
    float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
    if (ti == 0) rowScales[tg] = rstd;
    for (uint u = ti; u < hiddenU4; u += 32) {
        H8 hv = asH8(xr[u]), gv = asH8(gamma[u]), ov;
        float f[8];
        for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * (float(gv.v[i]) + 1.0f);
        fwht32_f32_to_h8(f, ov.v, ti);
        yr[u] = *(thread uint4*)&ov;
    }
}

// variant 4 (P3 race — the COMBINED candidate): fused residual + prologue-scale.
kernel void qwg_rmsnorm_residual_pscale(device const uint4* x   [[buffer(0)]],
                                        device const uint4* gamma [[buffer(1)]],
                                        device uint4* y          [[buffer(2)]],
                                        constant uint& hiddenU4  [[buffer(3)]],
                                        constant float& eps      [[buffer(4)]],
                                        device uint4* residual   [[buffer(5)]],
                                        device float* rowScales  [[buffer(6)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint ti [[thread_index_in_threadgroup]]) {
    device const uint4* xr = x + (size_t)tg * hiddenU4;
    device uint4* rr = residual + (size_t)tg * hiddenU4;
    device uint4* yr = y + (size_t)tg * hiddenU4;

    if (hiddenU4 <= QW_RES2_U4_TOTAL) {
        float acc = 0.0f;
        uint4 reg[QW_RES2_U4_PER_LANE];
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            uint4 xv = xr[u], rv0 = rr[u];
            H8 av = asH8(xv), bv = asH8(rv0), sv;
            for (int i = 0; i < 8; i++) { sv.v[i] = half(float(av.v[i]) + float(bv.v[i])); float t = float(sv.v[i]); acc += t * t; }
            rr[u] = *(thread uint4*)&sv;
            reg[k] = *(thread uint4*)&sv;
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
        if (ti == 0) rowScales[tg] = rstd;
        // Pass 2: y = fwht32(res·(1+γ)) — NO rstd multiply (deferred to GEMV).
        #pragma unroll
        for (uint k = 0; k < QW_RES2_U4_PER_LANE; k++) {
            uint u = ti + 32u * k;
            if (u >= hiddenU4) break;
            H8 bv = asH8(reg[k]), gv = asH8(gamma[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(bv.v[i]) * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
        }
    } else {
        float acc = 0.0f;
        for (uint u = ti; u < hiddenU4; u += 32) {
            uint4 xv = xr[u], rv0 = rr[u];
            H8 av = asH8(xv), bv = asH8(rv0), sv;
            for (int i = 0; i < 8; i++) { sv.v[i] = half(float(av.v[i]) + float(bv.v[i])); float t = float(sv.v[i]); acc += t * t; }
            rr[u] = *(thread uint4*)&sv;
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8) + eps);
        if (ti == 0) rowScales[tg] = rstd;
        for (uint u = ti; u < hiddenU4; u += 32) {
            H8 bv = asH8(rr[u]), gv = asH8(gamma[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(bv.v[i]) * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
        }
    }
}

// Batched MTP pre-FC dual RMSNorm: replaces 2*pairs single-row dispatches with 1 grid=(pairs, 2, 1) dispatch.
// tg.y == 0: qwg_rmsnorm on embeddings[tStart + p] -> out[p * 2 * hiddenU4]
// tg.y == 1: qwg_rmsnorm_from_rot on (t == 0 ? previous : hidden[t - 1]) -> out[(p * 2 + 1) * hiddenU4]
kernel void qwg_mtp_pre_fc_norm(device const uint4* embeddings [[buffer(0)]],
                                device const uint4* hidden     [[buffer(1)]],
                                device const uint4* previous   [[buffer(2)]],
                                device const uint4* gammaEmb   [[buffer(3)]],
                                device const uint4* gammaHid   [[buffer(4)]],
                                device uint4* out              [[buffer(5)]],
                                constant uint& hiddenU4        [[buffer(6)]],
                                constant float& eps            [[buffer(7)]],
                                constant uint& tStart          [[buffer(8)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint ti [[thread_index_in_threadgroup]]) {
    const uint p = tg.x;
    const uint t = tStart + p;
    if (tg.y == 0u) {
        device const uint4* xr = embeddings + (size_t)t * hiddenU4;
        float acc = 0.0f;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            H8 hv = asH8(xr[u]);
            for (int i = 0; i < 8; i++) { float v = float(hv.v[i]); acc += v * v; }
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8u) + eps);
        device uint4* yr = out + (size_t)(p * 2u) * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            H8 hv = asH8(xr[u]), gv = asH8(gammaEmb[u]), ov;
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
        }
    } else {
        device const uint4* xr = (t == 0u) ? previous : (hidden + (size_t)(t - 1u) * hiddenU4);
        float acc = 0.0f;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            H8 hv = asH8(xr[u]);
            fwht32_inplace(hv.v, ti);
            for (int i = 0; i < 8; i++) { float v = float(hv.v[i]); acc += v * v; }
        }
        float total = simd_sum(acc);
        float rstd = rsqrt(total / float(hiddenU4 * 8u) + eps);
        device uint4* yr = out + (size_t)(p * 2u + 1u) * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            H8 hv = asH8(xr[u]), gv = asH8(gammaHid[u]), ov;
            fwht32_inplace(hv.v, ti);
            float f[8];
            for (int i = 0; i < 8; i++) f[i] = float(hv.v[i]) * rstd * (float(gv.v[i]) + 1.0f);
            fwht32_f32_to_h8(f, ov.v, ti);
            yr[u] = *(thread uint4*)&ov;
        }
    }
}

// Elementwise FP16 bias add in FP32 accumulation (used for optional o_proj.bias).
kernel void qwg_add_bias_f16(device uint4* x           [[buffer(0)]],
                             device const uint4* bias  [[buffer(1)]],
                             constant uint& rowU4      [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint ti [[thread_index_in_threadgroup]]) {
    const uint u = tg.x * 32u + ti;
    if (u >= rowU4) return;
    device uint4* xr = x + (size_t)tg.y * rowU4;
    H8 xv = asH8(xr[u]), bv = asH8(bias[u]), ov;
    for (int i = 0; i < 8; i++) ov.v[i] = half(float(xv.v[i]) + float(bv.v[i]));
    xr[u] = *(thread uint4*)&ov;
}

// Arbitrary-width LayerNorm in FP32 accumulation (no %256 requirement, unrotated).
kernel void qwg_layernorm_f16(device const half*  x       [[buffer(0)]],
                              device const half*  gamma   [[buffer(1)]],
                              device const half*  beta    [[buffer(2)]],
                              device half*        out     [[buffer(3)]],
                              constant uint&      hidden  [[buffer(4)]],
                              constant float&     eps     [[buffer(5)]],
                              constant uint&      hasBias [[buffer(6)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint ti [[thread_index_in_threadgroup]]) {
    device const half* xr = x + (size_t)tg * hidden;
    device half* yr = out + (size_t)tg * hidden;
    float sum = 0.0f;
    for (uint i = ti; i < hidden; i += 32u) { sum += float(xr[i]); }
    float mean = simd_sum(sum) / float(hidden);

    float varAcc = 0.0f;
    for (uint i = ti; i < hidden; i += 32u) {
        float d = float(xr[i]) - mean;
        varAcc += d * d;
    }
    float rstd = rsqrt(simd_sum(varAcc) / float(hidden) + eps);
    for (uint i = ti; i < hidden; i += 32u) {
        float val = (float(xr[i]) - mean) * rstd * float(gamma[i]);
        if (hasBias != 0u) { val += float(beta[i]); }
        yr[i] = half(val);
    }
}

// Sandwich post-norm + residual add: residual += rmsnorm(branch, gamma) in FP32.
kernel void qwg_post_rmsnorm_residual_add_f16(device half*        residual   [[buffer(0)]],
                                              device const half*  branch     [[buffer(1)]],
                                              device const half*  gamma      [[buffer(2)]],
                                              constant uint&      hidden     [[buffer(3)]],
                                              constant float&     eps        [[buffer(4)]],
                                              constant float&     unitOffset [[buffer(5)]],
                                              uint tg [[threadgroup_position_in_grid]],
                                              uint ti [[thread_index_in_threadgroup]]) {
    device half* rr = residual + (size_t)tg * hidden;
    device const half* br = branch + (size_t)tg * hidden;
    float acc = 0.0f;
    for (uint i = ti; i < hidden; i += 32u) {
        float v = float(br[i]);
        acc += v * v;
    }
    float rstd = rsqrt(simd_sum(acc) / float(hidden) + eps);
    for (uint i = ti; i < hidden; i += 32u) {
        float normed = float(br[i]) * rstd * (float(gamma[i]) + unitOffset);
        rr[i] = half(float(rr[i]) + normed);
    }
}

// Arbitrary-width unrotated RMSNorm in FP32 accumulation (for QK-norm & non-H32 backbones).
kernel void qwg_rmsnorm_f16(device const half*  x          [[buffer(0)]],
                            device const half*  gamma      [[buffer(1)]],
                            device half*        out        [[buffer(2)]],
                            constant uint&      hidden     [[buffer(3)]],
                            constant float&     eps        [[buffer(4)]],
                            constant float&     unitOffset [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint ti [[thread_index_in_threadgroup]]) {
    device const half* xr = x + (size_t)tg * hidden;
    device half* yr = out + (size_t)tg * hidden;
    float acc = 0.0f;
    for (uint i = ti; i < hidden; i += 32u) {
        float v = float(xr[i]);
        acc += v * v;
    }
    float rstd = rsqrt(simd_sum(acc) / float(hidden) + eps);
    for (uint i = ti; i < hidden; i += 32u) {
        yr[i] = half(float(xr[i]) * rstd * (float(gamma[i]) + unitOffset));
    }
}


