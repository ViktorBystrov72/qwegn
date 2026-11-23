// qw_ink2_dense.metal — Inkling-2 BF16 embed-norm and RMSNorm kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

kernel void qwg_ink2_embed_norm(device const int32_t* tokenIds [[buffer(0)]],
                                device const bfloat* embedTable [[buffer(1)]],
                                device const bfloat* normW [[buffer(2)]],
                                device bfloat* out [[buffer(3)]],
                                constant QWInk2EmbedParams& p [[buffer(4)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint t = tg;
    if (t >= p.T) return;
    int32_t tok = clamp(tokenIds[t], 0, int32_t(p.V) - 1);
    device const bfloat* row = embedTable + size_t(tok) * p.H;
    device bfloat* dst = out + size_t(t) * p.H;

    float sq = 0.0f;
    for (uint d = tid; d < p.H; d += 256u) {
        float v = float(row[d]);
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSums[8];
    if (lane == 0u) sgSums[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float val = (lane < 8u) ? sgSums[lane] : 0.0f;
        val = simd_sum(val);
        if (lane == 0u) sgSums[0] = rsqrt(val / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invRms = sgSums[0];
    for (uint d = tid; d < p.H; d += 256u) {
        float v = float(row[d]);
        dst[d] = qw_ink2_bf16((v * invRms) * float(normW[d]));
    }
}

kernel void qwg_ink2_rms_norm(device const bfloat* inBF16 [[buffer(0)]],
                              device const float* inF32 [[buffer(1)]],
                              device const bfloat* weight [[buffer(2)]],
                              device bfloat* outBF16 [[buffer(3)]],
                              constant QWInk2NormParams& p [[buffer(4)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint r = tg;
    if (r >= p.rows) return;
    const size_t base = size_t(r) * p.dim;
    float sq = 0.0f;
    if (p.inIsF32 != 0u) {
        device const float* src = inF32 + base;
        for (uint d = tid; d < p.dim; d += 256u) {
            float v = src[d];
            sq += v * v;
        }
    } else {
        device const bfloat* src = inBF16 + base;
        for (uint d = tid; d < p.dim; d += 256u) {
            float v = float(src[d]);
            sq += v * v;
        }
    }
    sq = simd_sum(sq);
    threadgroup float sgSums[8];
    if (lane == 0u) sgSums[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float val = (lane < 8u) ? sgSums[lane] : 0.0f;
        val = simd_sum(val);
        if (lane == 0u) sgSums[0] = rsqrt(val / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invRms = sgSums[0];
    device bfloat* dst = outBF16 + base;
    if (p.inIsF32 != 0u) {
        device const float* src = inF32 + base;
        for (uint d = tid; d < p.dim; d += 256u) {
            dst[d] = qw_ink2_bf16((src[d] * invRms) * float(weight[d]));
        }
    } else {
        device const bfloat* src = inBF16 + base;
        for (uint d = tid; d < p.dim; d += 256u) {
            dst[d] = qw_ink2_bf16((float(src[d]) * invRms) * float(weight[d]));
        }
    }
}
