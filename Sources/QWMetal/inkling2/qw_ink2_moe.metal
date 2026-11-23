// qw_ink2_moe.metal — TG-cached sigmoid top-K router with MoE-Spec pruning for Inkling-2.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"
#include "include/qw_router_union.metalh"

using namespace metal;

inline float qw_ink2_router_dot8_fma(float4 w0, float4 w1, float4 x0, float4 x1, float acc) {
#pragma clang fp contract(off) reassociate(off)
    const float a0 = fma(w0.w, x0.w, fma(w0.z, x0.z, fma(w0.y, x0.y, fma(w0.x, x0.x, acc))));
    return fma(w1.w, x1.w, fma(w1.z, x1.z, fma(w1.y, x1.y, fma(w1.x, x1.x, a0))));
}

kernel void qwg_ink2_router_topk(device const bfloat* x [[buffer(0)]],
                                 device const bfloat* gateW [[buffer(1)]],
                                 device const float* bias [[buffer(2)]],
                                 device int32_t* topIds [[buffer(3)]],
                                 device float* topWeights [[buffer(4)]],
                                 constant QWInk2RouterParams& p [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint simd_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off) reassociate(off)
    const uint t = tg;
    if (t >= p.T) return;
    const uint nExp = min(p.numExperts, 512u);
    device const bfloat* xr = x + size_t(t) * p.H;
    const bool useSh = (p.H <= 4096u);
    const bool vec8 = ((p.H & 7u) == 0u) && ((size_t(gateW) & 15u) == 0u) && ((size_t(xr) & 15u) == 0u);

    threadgroup uint4 xShRaw[512];
    threadgroup bfloat* xSh = (threadgroup bfloat*)xShRaw;
    threadgroup float scoresSh[512];
    threadgroup float biasedSh[512];

    if (useSh) {
        if (vec8) {
            for (uint k = tid * 8u; k < p.H; k += 2048u) {
                *(threadgroup uint4*)(xSh + k) = *(device const uint4*)(xr + k);
            }
        } else {
            for (uint k = tid; k < p.H; k += 256u) xSh[k] = xr[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = simd_id; e < nExp; e += 8u) {
        device const bfloat* wr = gateW + size_t(e) * p.H;
        float acc = 0.0f;
        if (vec8) {
            for (uint k = lane * 8u; k < p.H; k += 256u) {
                const uint4 wq = *(device const uint4*)(wr + k);
                const uint4 xq = useSh ? *(threadgroup const uint4*)(xSh + k)
                                       : *(device const uint4*)(xr + k);
                const float4 wf0 = float4(as_type<bfloat>(ushort(wq.x)), as_type<bfloat>(ushort(wq.x >> 16u)),
                                          as_type<bfloat>(ushort(wq.y)), as_type<bfloat>(ushort(wq.y >> 16u)));
                const float4 wf1 = float4(as_type<bfloat>(ushort(wq.z)), as_type<bfloat>(ushort(wq.z >> 16u)),
                                          as_type<bfloat>(ushort(wq.w)), as_type<bfloat>(ushort(wq.w >> 16u)));
                const float4 xf0 = float4(as_type<bfloat>(ushort(xq.x)), as_type<bfloat>(ushort(xq.x >> 16u)),
                                          as_type<bfloat>(ushort(xq.y)), as_type<bfloat>(ushort(xq.y >> 16u)));
                const float4 xf1 = float4(as_type<bfloat>(ushort(xq.z)), as_type<bfloat>(ushort(xq.z >> 16u)),
                                          as_type<bfloat>(ushort(xq.w)), as_type<bfloat>(ushort(xq.w >> 16u)));
                acc = qw_ink2_router_dot8_fma(wf0, wf1, xf0, xf1, acc);
            }
        } else {
            for (uint k = lane; k < p.H; k += 32u) {
                const float xv = useSh ? float(xSh[k]) : float(xr[k]);
                acc = fma(float(wr[k]), xv, acc);
            }
        }
        const float dotVal = simd_sum(acc);
        if (lane == 0u) {
            const float sig = qw_ink2_sigmoid(dotVal);
            scoresSh[e] = sig;
            biasedSh[e] = sig + bias[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kLimit = min(p.topK, 16u);
    const uint keepK = (p.moeSpecBudget > 0u && t >= 1u && p.moeSpecBudget < kLimit)
        ? p.moeSpecBudget : kLimit;
    const uint e0 = tid;
    const uint e1 = tid + 256u;
    float b0 = (e0 < nExp) ? biasedSh[e0] : -1e30f;
    float b1 = (e1 < nExp) ? biasedSh[e1] : -1e30f;

    threadgroup float sgB[8];
    threadgroup int32_t sgE[8];
    threadgroup int32_t winESh;
    threadgroup float invSh;
    float sumW = 0.0f;
    device int32_t* outId = topIds + size_t(t) * p.topK;
    device float* outW = topWeights + size_t(t) * p.topK;

    for (uint k = 0u; k < keepK; ++k) {
        float locB = (e0 < nExp) ? b0 : -1e30f;
        int32_t locE = (e0 < nExp) ? int32_t(e0) : 0x7fffffff;
        if (e1 < nExp && (b1 > locB || (b1 == locB && int32_t(e1) < locE))) {
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
        if (simd_id == 0u) {
            float sb = (lane < 8u) ? sgB[lane] : -1e30f;
            int32_t se = (lane < 8u) ? sgE[lane] : 0x7fffffff;
            #pragma unroll
            for (uint off = 4u; off > 0u; off >>= 1u) {
                const float ob = simd_shuffle_down(sb, off);
                const int32_t oe = simd_shuffle_down(se, off);
                if (ob > sb || (ob == sb && oe < se)) {
                    sb = ob;
                    se = oe;
                }
            }
            if (lane == 0u) {
                winESh = se;
                outId[k] = se;
                const float w = (se >= 0 && uint(se) < nExp) ? scoresSh[se] : 0.0f;
                outW[k] = w;
                sumW += w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int32_t wE = winESh;
        if (int32_t(e0) == wE) b0 = -1e30f;
        if (int32_t(e1) == wE) b1 = -1e30f;
    }
    if (tid == 0u) invSh = p.routeScale / (sumW + 1e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < kLimit) {
        if (tid < keepK) {
            outW[tid] *= invSh;
        } else {
            outId[tid] = -1;
            outW[tid] = 0.0f;
        }
    }
}

kernel void qwg_ink2_router_union_prune(device int32_t* topIds [[buffer(0)]],
                                        device float* topWeights [[buffer(1)]],
                                        constant QWInk2RouterParams& p [[buffer(2)]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint simd_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup QWRUnionScratch unionSc;
    if (p.moeSpecBudget == 0u || p.T <= 1u) return;
    qwg_router_union_prune_tg(
        topIds, topWeights, p.T, p.topK, p.moeSpecBudget, true, tid, simd_id, lane, unionSc);
}
