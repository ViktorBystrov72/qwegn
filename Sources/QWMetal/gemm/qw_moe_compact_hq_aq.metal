// qw_moe_compact_hq_aq.metal — Active-bucket compact MoE decode/verify (T<=16)
// for HQ4_G32 and AQ{4_G32,4_G64,8_G32}, 0-ULP bit-identical to T=1 r1/combine.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_moe_f32_common.metalh"
using namespace metal;

struct QWMoECompactQWPParams {
    uint N;
    uint K;
    uint numExperts;
    uint residentE;
    uint flags;
    float gateCap;
    float upCap;
};

template <typename Traits, bool HAS_SOFTCAP = false>
METAL_FUNC void qwg_moe_compact_gateup_f32_body(
    typename Traits::WeightPtr w, device const float* x,
    device const int32_t* expertOff, device const int32_t* activeExperts,
    device const uint* activeArgs, device const int32_t* sortedTok,
    device const int32_t* sortedPairIdx, device float* actOut,
    typename Traits::WeightPtr coldW, constant QWMoECompactQWPParams& p,
    uint2 tg, uint tid, uint sgi, uint lane,
    threadgroup float* tgAct, threadgroup typename Traits::TgLut& tgLut) {
    if (tg.y >= activeArgs[0]) return;
    const uint rawExpert = uint(activeExperts[tg.y]);
    if (rawExpert >= p.numExperts) return;
    const int off = expertOff[rawExpert];
    const int cnt = expertOff[rawExpert + 1] - off;
    if (cnt <= 0) return;
    const uint I = p.N >> 1u;
    const uint rBase = tg.x * 32u;
    if (rBase >= I) return;
    if constexpr (Traits::has_tg_lut()) {
        tgLut.load(tid, 256u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        (void)tid;
    }
    const bool isCold = (p.residentE > 0u) && (rawExpert >= p.residentE);
    const uint expert = isCold ? (rawExpert - p.residentE) : rawExpert;
    typename Traits::WeightPtr wSrc = isCold ? coldW : w;
    const uint r0 = rBase + sgi * 4u;
    const uint upHalf = lane >> 4u, hl = lane & 15u, nSlices = p.K >> 5u;
    const size_t rowStride = Traits::row_stride(p.K);
    const size_t eBase = size_t(expert) * p.N + (upHalf != 0u ? I : 0u);
    const uint rows = (r0 < I) ? min(4u, I - r0) : 0u;
    typename Traits::WeightPtr a0 = Traits::row_ptr(wSrc, eBase + (rows != 0u ? r0 : 0u), rowStride);
    const size_t st1 = rows > 1u ? rowStride : 0u;
    const size_t st2 = rows > 2u ? 2u * rowStride : 0u;
    const size_t st3 = rows > 3u ? 3u * rowStride : 0u;
    const bool applyFwht = (p.flags & 1u) != 0u;
    const bool sortedOut = (p.flags & 2u) != 0u;
    const float2 softcaps = float2(p.gateCap, p.upCap);
    if constexpr (!HAS_SOFTCAP) (void)softcaps;

    for (uint base = 0u; base < uint(cnt); base += 4u) {
        const uint mCount = min(4u, uint(cnt) - base);
        uint tokIds[4], pairSlots[4];
        for (uint m = 0u; m < mCount; ++m) {
            const uint idx = uint(off) + base + m;
            tokIds[m] = uint(sortedTok[idx]);
            pairSlots[m] = sortedOut ? idx : uint(sortedPairIdx[idx]);
        }
        float4 accM[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        for (uint s = hl; rows != 0u && s < nSlices; s += 16u) {
            const uint col = s << 5u;
            typename Traits::WeightPtr gp = Traits::group_ptr(a0, s);
            const uint byteInGroup = Traits::byte_in_group(s);
            if (mCount == 1u) {
                typename Traits::ActSlice a;
                Traits::load_act_f32(x + size_t(tokIds[0]) * p.K + col, a);
                accM[0].x += Traits::slice_dot(gp, byteInGroup, a, tgLut);
                accM[0].y += Traits::slice_dot(gp + st1, byteInGroup, a, tgLut);
                accM[0].z += Traits::slice_dot(gp + st2, byteInGroup, a, tgLut);
                accM[0].w += Traits::slice_dot(gp + st3, byteInGroup, a, tgLut);
            } else {
                typename Traits::ActSlice a0x, a1x;
                Traits::load_act_f32(x + size_t(tokIds[0]) * p.K + col, a0x);
                Traits::load_act_f32(x + size_t(tokIds[1]) * p.K + col, a1x);
                const float2 r0 = Traits::slice_dot2_acc(gp, byteInGroup, a0x, a1x, tgLut, float2(accM[0].x, accM[1].x));
                const float2 r1 = Traits::slice_dot2_acc(gp + st1, byteInGroup, a0x, a1x, tgLut, float2(accM[0].y, accM[1].y));
                const float2 r2 = Traits::slice_dot2_acc(gp + st2, byteInGroup, a0x, a1x, tgLut, float2(accM[0].z, accM[1].z));
                const float2 r3 = Traits::slice_dot2_acc(gp + st3, byteInGroup, a0x, a1x, tgLut, float2(accM[0].w, accM[1].w));
                accM[0] = float4(r0.x, r1.x, r2.x, r3.x);
                accM[1] = float4(r0.y, r1.y, r2.y, r3.y);
                if (mCount == 3u) {
                    Traits::load_act_f32(x + size_t(tokIds[2]) * p.K + col, a0x);
                    accM[2].x += Traits::slice_dot(gp, byteInGroup, a0x, tgLut);
                    accM[2].y += Traits::slice_dot(gp + st1, byteInGroup, a0x, tgLut);
                    accM[2].z += Traits::slice_dot(gp + st2, byteInGroup, a0x, tgLut);
                    accM[2].w += Traits::slice_dot(gp + st3, byteInGroup, a0x, tgLut);
                } else if (mCount == 4u) {
                    Traits::load_act_f32(x + size_t(tokIds[2]) * p.K + col, a0x);
                    Traits::load_act_f32(x + size_t(tokIds[3]) * p.K + col, a1x);
                    const float2 e0 = Traits::slice_dot2_acc(gp, byteInGroup, a0x, a1x, tgLut, float2(accM[2].x, accM[3].x));
                    const float2 e1 = Traits::slice_dot2_acc(gp + st1, byteInGroup, a0x, a1x, tgLut, float2(accM[2].y, accM[3].y));
                    const float2 e2 = Traits::slice_dot2_acc(gp + st2, byteInGroup, a0x, a1x, tgLut, float2(accM[2].z, accM[3].z));
                    const float2 e3 = Traits::slice_dot2_acc(gp + st3, byteInGroup, a0x, a1x, tgLut, float2(accM[2].w, accM[3].w));
                    accM[2] = float4(e0.x, e1.x, e2.x, e3.x);
                    accM[3] = float4(e0.y, e1.y, e2.y, e3.y);
                }
            }
        }
        for (uint m = 0u; m < mCount; ++m) {
            accM[m] = qwg_simd_reduce_down<16u>(accM[m]);
            const float4 uAcc = simd_shuffle_down(accM[m], 16);
            if (lane == 0u) {
                const uint c = m * 32u + sgi * 4u;
                if constexpr (HAS_SOFTCAP) {
                    tgAct[c + 0u] = rows > 0u ? qwg_moe_softcap_swiglu_val(accM[m].x, uAcc.x, softcaps) : 0.0f;
                    tgAct[c + 1u] = rows > 1u ? qwg_moe_softcap_swiglu_val(accM[m].y, uAcc.y, softcaps) : 0.0f;
                    tgAct[c + 2u] = rows > 2u ? qwg_moe_softcap_swiglu_val(accM[m].z, uAcc.z, softcaps) : 0.0f;
                    tgAct[c + 3u] = rows > 3u ? qwg_moe_softcap_swiglu_val(accM[m].w, uAcc.w, softcaps) : 0.0f;
                } else {
                    tgAct[c + 0u] = rows > 0u ? qwg_swiglu_mul(accM[m].x, uAcc.x) : 0.0f;
                    tgAct[c + 1u] = rows > 1u ? qwg_swiglu_mul(accM[m].y, uAcc.y) : 0.0f;
                    tgAct[c + 2u] = rows > 2u ? qwg_swiglu_mul(accM[m].z, uAcc.z) : 0.0f;
                    tgAct[c + 3u] = rows > 3u ? qwg_swiglu_mul(accM[m].w, uAcc.w) : 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi < mCount) {
            const float act = (rBase + lane < I) ? tgAct[sgi * 32u + lane] : 0.0f;
            const float v = applyFwht ? qwg_fwht32_lane_f32(act, lane) : act;
            if (rBase + lane < I) actOut[size_t(pairSlots[sgi]) * I + rBase + lane] = v;
        }
        if (base + 4u < uint(cnt)) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename Traits>
METAL_FUNC void qwg_moe_compact_down_f32_body(
    typename Traits::WeightPtr wBytes, device const float* x,
    device const int32_t* expertOff, device const int32_t* activeExperts,
    device const uint* activeArgs, device const int32_t* sortedPairIdx,
    device float* dnSlotOut, typename Traits::WeightPtr coldWBytes,
    constant QWMoECompactQWPParams& p,
    uint2 tg, uint tid, uint sgi, uint lane,
    threadgroup typename Traits::TgLut& tgLut) {
    if (tg.y >= activeArgs[0]) return;
    const uint rawExpert = uint(activeExperts[tg.y]);
    if (rawExpert >= p.numExperts) return;
    const int off = expertOff[rawExpert];
    const int cnt = expertOff[rawExpert + 1] - off;
    if (cnt <= 0) return;
    if (tg.x * 32u >= p.N) return;
    if constexpr (Traits::has_tg_lut()) {
        tgLut.load(tid, 256u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        (void)tid;
    }
    const uint r0 = (tg.x * 8u + sgi) * 4u;
    const uint rowsLeft = (r0 < p.N) ? (p.N - r0) : 0u;
    if (rowsLeft == 0u) return;
    const bool isCold = (p.residentE > 0u) && (rawExpert >= p.residentE);
    const uint wExp = isCold ? (rawExpert - p.residentE) : rawExpert;
    typename Traits::WeightPtr wSrc = isCold ? coldWBytes : wBytes;
    const uint nSlices = p.K >> 5u;
    const bool sortedOut = (p.flags & 2u) != 0u;

    for (uint pair = 0u; pair < uint(cnt); pair += 2u) {
        const uint idxA = uint(off) + pair;
        const uint slotA = sortedOut ? idxA : uint(sortedPairIdx[idxA]);
        if (pair + 1u < uint(cnt)) {
            const uint idxB = idxA + 1u;
            const uint slotB = sortedOut ? idxB : uint(sortedPairIdx[idxB]);
            float4 vx0, vx1;
            qwg_moe_down_rows4_2x<Traits>(
                wSrc, x + size_t(slotA) * p.K, x + size_t(slotB) * p.K,
                wExp, r0, rowsLeft, p.N, p.K, nSlices, lane, tgLut, vx0, vx1);
            if (lane == 0u) {
                qwg_moe_write_route4(dnSlotOut, slotA, p.N, r0, rowsLeft, vx0);
                qwg_moe_write_route4(dnSlotOut, slotB, p.N, r0, rowsLeft, vx1);
            }
        } else {
            const float4 vx0 = qwg_moe_down_rows4_1x<Traits>(
                wSrc, x + size_t(slotA) * p.K,
                wExp, r0, rowsLeft, p.N, p.K, nSlices, lane, tgLut);
            if (lane == 0u) qwg_moe_write_route4(dnSlotOut, slotA, p.N, r0, rowsLeft, vx0);
        }
    }
}

#define QW_MOE_COMPACT_HQ_AQ_KERNELS(NAME, TRAITS, WTYPE) \
kernel void qwg_moe_compact_##NAME##_gateup_f32( \
    device const WTYPE* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const WTYPE* coldW [[buffer(8)]], constant QWMoECompactQWPParams& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup QWEmptyTgLut tgLut; \
    qwg_moe_compact_gateup_f32_body<TRAITS, false>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_##NAME##_gateup_f32_sc( \
    device const WTYPE* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedTok [[buffer(5)]], \
    device const int32_t* sortedPairIdx [[buffer(6)]], device float* actOut [[buffer(7)]], \
    device const WTYPE* coldW [[buffer(8)]], constant QWMoECompactQWPParams& p [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float tgAct[128]; \
    threadgroup QWEmptyTgLut tgLut; \
    qwg_moe_compact_gateup_f32_body<TRAITS, true>( \
        w, x, expertOff, activeExperts, activeArgs, sortedTok, sortedPairIdx, \
        actOut, coldW, p, tg, tid, sgi, lane, tgAct, tgLut); \
} \
kernel void qwg_moe_compact_##NAME##_down_f32( \
    device const WTYPE* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device const int32_t* expertOff [[buffer(2)]], device const int32_t* activeExperts [[buffer(3)]], \
    device const uint* activeArgs [[buffer(4)]], device const int32_t* sortedPairIdx [[buffer(5)]], \
    device float* dnSlotOut [[buffer(6)]], device const WTYPE* coldW [[buffer(7)]], \
    constant QWMoECompactQWPParams& p [[buffer(8)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup QWEmptyTgLut tgLut; \
    qwg_moe_compact_down_f32_body<TRAITS>( \
        w, x, expertOff, activeExperts, activeArgs, sortedPairIdx, \
        dnSlotOut, coldW, p, tg, tid, sgi, lane, tgLut); \
}

#define QW_MOE_COMPACT_AQ_PAIR(BITS, GROUP) \
    using TraitsAQ_##BITS##_##GROUP = QWQuantAQ<BITS, GROUP>; \
    QW_MOE_COMPACT_HQ_AQ_KERNELS(aq##BITS##_g##GROUP, TraitsAQ_##BITS##_##GROUP, char)

QW_MOE_COMPACT_HQ_AQ_KERNELS(q4g, QWQuantHQ4, Block_HQ4_G32)
QW_MOE_COMPACT_AQ_PAIR(4, 32)
QW_MOE_COMPACT_AQ_PAIR(4, 64)
QW_MOE_COMPACT_AQ_PAIR(8, 32)
