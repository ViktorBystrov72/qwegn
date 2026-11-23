// ── General RoPE — the standard families' path (spec specs/07-qwrope.md) ──
// The 27B's fused norm+rope kernel is frozen: generalising it in place moved
// the golden's logits. This is the general kernel — the head's data window
// is runtime, the per-head norm is optional, the rope width is the model's
// rotary dims, and the partner exchange reads the partner from memory.
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
using namespace metal;

struct H8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline H8 asH8(uint4 w) { return *(thread H8*)&w; }

kernel void qwg_rope_qk_general(device const uint4* qIn      [[buffer(0)]],
                                device const uint4* kIn      [[buffer(1)]],
                                device uint4* qOut           [[buffer(2)]],
                                device uint4* kOut           [[buffer(3)]],
                                device const uint4* wq       [[buffer(4)]],
                                device const uint4* wk       [[buffer(5)]],
                                device const float* table    [[buffer(6)]],
                                device uint4* normOut        [[buffer(7)]],
                                constant uint4& flags        [[buffer(8)]],
                                constant float& eps          [[buffer(9)]],
                                device const uint4* qkvBias  [[buffer(10)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint ti [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const bool isQ = tg.y < QW_Q_HEADS;
    const uint row = isQ ? tg.y : (tg.y - QW_Q_HEADS);
    const uint head = isQ ? (row * QW_Q_STRIDE) : row;
    const uint headU4 = flags.x;                          // headDim/8 uint4s per head
    const uint qRow = QW_QPROJ_OUT / 8u;                   // uint4s per q row
    const uint kRow = (QW_KV_HEADS * QW_HEAD_DIM) / 8u;    // uint4s per k row
    const size_t uOff = isQ ? ((size_t)s * qRow + (size_t)head * headU4)
                            : ((size_t)s * kRow + (size_t)row * headU4);
    device const uint4* src = (isQ ? qIn : kIn) + uOff;
    device uint4* dst = (isQ ? qOut : kOut) + uOff;
    device const uint4* w = (isQ ? wq : (wk + (size_t)row * headU4));
    const bool active = ti < headU4;
    const bool hasNorm = (flags.w & 1u) != 0u;
    const bool hasBias = (flags.w & 2u) != 0u;
    const size_t biasOff = isQ ? ((size_t)head * headU4)
                               : ((size_t)(QW_QPROJ_OUT / 8u) + (size_t)row * headU4);
    device const uint4* bPtr = qkvBias + biasOff;

    uint4 xw = active ? src[ti] : uint4(0u);
    H8 xv = asH8(xw);
    float xf[8];
    if (hasBias && active) {
        H8 bv = asH8(bPtr[ti]);
        for (int i = 0; i < 8; i++) xf[i] = float(xv.v[i]) + float(bv.v[i]);
    } else {
        for (int i = 0; i < 8; i++) xf[i] = float(xv.v[i]);
    }
    float xn[8];
    float rstd = 1.0f;
    if (hasNorm) {                                        // per-head norm (Qwen3 class)
        H8 wv = active ? asH8(w[ti]) : asH8(uint4(0u));
        float acc = 0.0f;
        for (int i = 0; i < 8; i++) { float v = xf[i]; acc += v * v; }
        rstd = rsqrt(simd_sum(acc) / float(headU4 * 8u) + eps);
        for (int i = 0; i < 8; i++) xn[i] = xf[i] * rstd * (float(wv.v[i]) + 1.0f);
    } else {
        for (int i = 0; i < 8; i++) xn[i] = xf[i];
    }

    if (flags.y && active) {
        H8 n16;
        for (int i = 0; i < 8; i++) n16.v[i] = half(xn[i]);
        const size_t qRowU4 = (size_t)QW_QPROJ_OUT / 8u;
        const size_t kRowU4 = (size_t)QW_KV_HEADS * headU4;
        const size_t nBase = isQ ? ((size_t)s * qRowU4 + (size_t)head * headU4)
                                 : ((size_t)flags.z + (size_t)s * kRowU4 + (size_t)row * headU4);
        normOut[nBase + ti] = *(thread uint4*)&n16;
    }

    // The roped dims pair k with k + ropeDims/2, which is ropeDims/16 uint4s
    // away — the partner lane in the 32-lane simdgroup. Exchange via simd_shuffle
    // for any rotary width (power-of-two or non-power-of-two) up to 256 dims.
    const uint ropeSlots = QW_ROPE_DIMS / 8u;
    const uint partnerHalf = QW_ROPE_DIMS / 16u;   // the partner is half the roped slots away
    const uint partner = (ti < partnerHalf) ? (ti + partnerHalf) : (ti - partnerHalf);

    float pf[8];
    if (partnerHalf > 0u && ropeSlots <= 32u) {
        const ushort srcLane = ushort(partner & 31u);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            pf[i] = simd_shuffle(xn[i], srcLane);
        }
    } else if (ti < ropeSlots) {
        H8 xvP = asH8(active && partner < headU4 ? src[partner] : uint4(0u));
        H8 bvP = asH8(hasBias && active && partner < headU4 ? bPtr[partner] : uint4(0u));
        float xfP[8];
        for (int i = 0; i < 8; i++) xfP[i] = float(xvP.v[i]) + (hasBias ? float(bvP.v[i]) : 0.0f);
        if (hasNorm) {
            H8 wvP = asH8(active && partner < headU4 ? w[partner] : uint4(0u));
            for (int i = 0; i < 8; i++) {
                pf[i] = xfP[i] * rstd * (float(wvP.v[i]) + 1.0f);
            }
        } else {
            for (int i = 0; i < 8; i++) pf[i] = xfP[i];
        }
    }

    uint4 o;
    if (ti < ropeSlots) {
        device const float* cs = table + (size_t)s * QW_ROPE_DIMS;
        const bool firstHalf = ti < partnerHalf;
        const uint pBase = (firstHalf ? ti : (ti - partnerHalf)) * 8u;
        H8 out8;
        for (int i = 0; i < 8; i++) {
            uint p = pBase + uint(i);
            float c = cs[p], sn = cs[QW_ROPE_DIMS / 2u + p];
            float rotv = firstHalf ? -pf[i] : pf[i];
            out8.v[i] = half(xn[i] * c + rotv * sn);
        }
        o = *(thread uint4*)&out8;
    } else {                                               // passthrough dims
        H8 out8;
        for (int i = 0; i < 8; i++) out8.v[i] = half(xn[i]);
        o = *(thread uint4*)&out8;
    }
    if (active) dst[ti] = o;
    // The gate half of an interleaved [Q‖gate] row is never roped.
    if (isQ && QW_HAS_Q_GATE && active) {
        if (hasBias) {
            H8 gx = asH8(src[headU4 + ti]);
            H8 gb = asH8(qkvBias[(size_t)(head + 1u) * headU4 + ti]);
            H8 go;
            for (int i = 0; i < 8; i++) go.v[i] = half(float(gx.v[i]) + float(gb.v[i]));
            dst[headU4 + ti] = *(thread uint4*)&go;
        } else {
            dst[headU4 + ti] = src[headU4 + ti];
        }
    }
}

// ── General extract + RoPE (+ optional norm) + V copy ─────────────────────
// Reads the packed attnY row (attnTotalN halfs) and emits q, k and v.
kernel void qwg_extract_rope_qk_general(device const uint4* attnY   [[buffer(0)]],
                                        device uint4* qOut           [[buffer(1)]],
                                        device uint4* kOut           [[buffer(2)]],
                                        device uint4* vOut           [[buffer(3)]],
                                        device const uint4* wq       [[buffer(4)]],
                                        device const uint4* wk       [[buffer(5)]],
                                        device const float* table    [[buffer(6)]],
                                        constant uint4& flags        [[buffer(8)]],
                                        constant float& eps          [[buffer(7)]],
                                        device const uint4* qkvBias  [[buffer(9)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint ti [[thread_index_in_threadgroup]]) {
    const uint s = tg.x;
    const bool isQ = tg.y < QW_Q_HEADS;
    const uint row = isQ ? tg.y : (tg.y - QW_Q_HEADS);
    const uint head = isQ ? (row * QW_Q_STRIDE) : row;
    const uint headU4 = flags.x;

    const uint inRowU4 = (flags.y != 0u) ? flags.y : ((QW_QPROJ_OUT + 2u * QW_KV_HEADS * QW_HEAD_DIM) / 8u);
    const size_t inRow = (size_t)s * inRowU4;
    const size_t srcOff = isQ ? (inRow + (size_t)head * headU4)
                              : (inRow + (size_t)QW_QPROJ_OUT / 8u + (size_t)row * headU4);
    device const uint4* src = attnY + srcOff;

    const size_t dstOff = isQ ? ((size_t)s * QW_QPROJ_OUT / 8u + (size_t)head * headU4)
                              : ((size_t)s * QW_KV_HEADS * QW_HEAD_DIM / 8u + (size_t)row * headU4);
    device uint4* dst = (isQ ? qOut : kOut) + dstOff;
    device const uint4* w = (isQ ? wq : (wk + (size_t)row * headU4));
    const bool active = ti < headU4;
    const bool hasNorm = (flags.w & 1u) != 0u;
    const bool hasBias = (flags.w & 2u) != 0u;
    const size_t biasOff = isQ ? ((size_t)head * headU4)
                               : ((size_t)(QW_QPROJ_OUT / 8u) + (size_t)row * headU4);
    device const uint4* bPtr = qkvBias + biasOff;

    uint4 xw = active ? src[ti] : uint4(0u);
    H8 xv = asH8(xw);
    float xf[8];
    if (hasBias && active) {
        H8 bv = asH8(bPtr[ti]);
        for (int i = 0; i < 8; i++) xf[i] = float(xv.v[i]) + float(bv.v[i]);
    } else {
        for (int i = 0; i < 8; i++) xf[i] = float(xv.v[i]);
    }
    float xn[8];
    float rstd = 1.0f;
    if (hasNorm) {
        float acc = 0.0f;
        for (int i = 0; i < 8; i++) { float v = xf[i]; acc += v * v; }
        rstd = rsqrt(simd_sum(acc) / float(headU4 * 8u) + eps);
        H8 wv = active ? asH8(w[ti]) : asH8(uint4(0u));
        for (int i = 0; i < 8; i++) xn[i] = xf[i] * rstd * (float(wv.v[i]) + 1.0f);
    } else {
        for (int i = 0; i < 8; i++) xn[i] = xf[i];
    }

    const uint ropeSlots = QW_ROPE_DIMS / 8u;
    const uint partnerHalf = QW_ROPE_DIMS / 16u;   // the partner is half the roped slots away
    const uint partner = (ti < partnerHalf) ? (ti + partnerHalf) : (ti - partnerHalf);

    float pf[8];
    if (partnerHalf > 0u && ropeSlots <= 32u) {
        const ushort srcLane = ushort(partner & 31u);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            pf[i] = simd_shuffle(xn[i], srcLane);
        }
    } else if (ti < ropeSlots) {
        H8 xvP = asH8(active && partner < headU4 ? src[partner] : uint4(0u));
        H8 bvP = asH8(hasBias && active && partner < headU4 ? bPtr[partner] : uint4(0u));
        float xfP[8];
        for (int i = 0; i < 8; i++) xfP[i] = float(xvP.v[i]) + (hasBias ? float(bvP.v[i]) : 0.0f);
        if (hasNorm) {
            H8 wvP = asH8(active && partner < headU4 ? w[partner] : uint4(0u));
            for (int i = 0; i < 8; i++) {
                pf[i] = xfP[i] * rstd * (float(wvP.v[i]) + 1.0f);
            }
        } else {
            for (int i = 0; i < 8; i++) pf[i] = xfP[i];
        }
    }

    uint4 o;
    if (ti < ropeSlots) {
        device const float* cs = table + (size_t)s * QW_ROPE_DIMS;
        const bool firstHalf = ti < partnerHalf;
        const uint pBase = (firstHalf ? ti : (ti - partnerHalf)) * 8u;
        H8 out8;
        for (int i = 0; i < 8; i++) {
            uint p = pBase + uint(i);
            float c = cs[p], sn = cs[QW_ROPE_DIMS / 2u + p];
            float rotv = firstHalf ? -pf[i] : pf[i];
            out8.v[i] = half(xn[i] * c + rotv * sn);
        }
        o = *(thread uint4*)&out8;
    } else {
        H8 out8;
        for (int i = 0; i < 8; i++) out8.v[i] = half(xn[i]);
        o = *(thread uint4*)&out8;
    }
    if (active) dst[ti] = o;
    if (isQ) {
        if (QW_HAS_Q_GATE && active) {
            if (hasBias) {
                H8 gx = asH8(src[headU4 + ti]);
                H8 gb = asH8(qkvBias[(size_t)(head + 1u) * headU4 + ti]);
                H8 go;
                for (int i = 0; i < 8; i++) go.v[i] = half(float(gx.v[i]) + float(gb.v[i]));
                dst[headU4 + ti] = *(thread uint4*)&go;
            } else {
                dst[headU4 + ti] = src[headU4 + ti];
            }
        }
    } else {
        if (active) {
            const size_t vOff = (size_t)row * headU4 + ti;
            const size_t vSrcOff = inRow + (size_t)(QW_QPROJ_OUT + QW_KV_HEADS * QW_HEAD_DIM) / 8u + vOff;
            const size_t vDstOff = (size_t)s * QW_KV_HEADS * QW_HEAD_DIM / 8u + vOff;
            if (hasBias) {
                H8 vx = asH8(attnY[vSrcOff]);
                H8 vb = asH8(qkvBias[(size_t)(QW_QPROJ_OUT + QW_KV_HEADS * QW_HEAD_DIM) / 8u + vOff]);
                H8 vo;
                for (int i = 0; i < 8; i++) vo.v[i] = half(float(vx.v[i]) + float(vb.v[i]));
                vOut[vDstOff] = *(thread uint4*)&vo;
            } else {
                vOut[vDstOff] = attnY[vSrcOff];
            }
        }
    }
}
