// qwgemm_moe_nax_f16.metal: fused expert-sorted down reduction + shared expert combine for the NAX MoE path.
#include <metal_stdlib>
using namespace metal;

static inline bfloat qwg_moe_combine_f32_to_bf16(float v) {
    uint u = as_type<uint>(v);
    if ((u & 0x7F800000u) == 0x7F800000u) return bfloat(v);
    u = u + 0x7FFFu + ((u >> 16u) & 1u);
    return bfloat(as_type<float>(u & 0xFFFF0000u));
}

static inline void qwg_moe_combine_store16_4(device half* out16, size_t idx, float4 y4,
                                              bool writeBF16, bool writeF32) {
    if (writeF32) {
        *(device float4*)((device float*)out16 + idx) = y4;
    } else if (writeBF16) {
        device bfloat* dst = (device bfloat*)out16 + idx;
        dst[0] = qwg_moe_combine_f32_to_bf16(y4.x);
        dst[1] = qwg_moe_combine_f32_to_bf16(y4.y);
        dst[2] = qwg_moe_combine_f32_to_bf16(y4.z);
        dst[3] = qwg_moe_combine_f32_to_bf16(y4.w);
    } else {
        *(device half4*)(out16 + idx) = half4(y4);
    }
}

static inline void qwg_moe_combine_store16_1(device half* out16, size_t idx, float y,
                                              bool writeBF16, bool writeF32) {
    if (writeF32) {
        ((device float*)out16)[idx] = y;
    } else if (writeBF16) {
        ((device bfloat*)out16)[idx] = qwg_moe_combine_f32_to_bf16(y);
    } else {
        out16[idx] = half(y);
    }
}

template <bool DN_IS_F16>
static inline float4 qwg_moe_reduce_acc4(
    device const float* dnAcc, device const int32_t* invPerm,
    device const float* w, device const int32_t* ids,
    uint N, uint TK, uint h, uint basePair, uint targetGroup, uint numGroups,
    bool identityPerm, bool hasIdsFilter) {
    device const half* dnAcc16 = (device const half*)dnAcc;
    float4 acc = float4(0.0f);
    for (uint k = 0; k < TK; ++k) {
        if (hasIdsFilter) {
            const int rawS = ids[basePair + k];
            if (rawS < 0) continue;
            if (numGroups > 0u && ((uint(rawS) >> 16u) & 0x7Fu) != targetGroup) continue;
        }
        const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
        if (p < 0) continue;
        const float wk = w[basePair + k];
        if (wk != 0.0f) {
            const float4 dv = DN_IS_F16
                ? float4(*(device const half4*)(dnAcc16 + size_t(p) * N + h))
                : (*(device const float4*)(dnAcc + size_t(p) * N + h));
            acc += wk * dv;
        }
    }
    return acc;
}

static inline float qwg_moe_reduce_acc1(
    device const float* dnAcc, device const int32_t* invPerm,
    device const float* w, device const int32_t* ids,
    uint N, uint TK, uint col, uint basePair, uint targetGroup, uint numGroups,
    bool dnIsF16, bool identityPerm, bool hasIdsFilter) {
    device const half* dnAcc16 = (device const half*)dnAcc;
    float acc = 0.0f;
    for (uint k = 0; k < TK; ++k) {
        if (hasIdsFilter) {
            const int rawS = ids[basePair + k];
            if (rawS < 0) continue;
            if (numGroups > 0u && ((uint(rawS) >> 16u) & 0x7Fu) != targetGroup) continue;
        }
        const float wk = w[basePair + k];
        if (wk == 0.0f) continue;
        const int p = identityPerm ? int(basePair + k) : invPerm[basePair + k];
        if (p < 0) continue;
        const float dv = dnIsF16 ? float(dnAcc16[size_t(p) * N + col]) : dnAcc[size_t(p) * N + col];
        acc += wk * dv;
    }
    return acc;
}

static inline float4 qwg_moe_add_shared_nofma4(float4 acc, float g, float4 sh) {
#pragma clang fp contract(off)
    return acc + (g * sh);
}

static inline float qwg_moe_add_shared_nofma1(float acc, float g, float sh) {
#pragma clang fp contract(off)
    return acc + (g * sh);
}

// Fused expert-sorted down reduction + shared expert combine:
// out[t, h] = sum_{k=0..<TK} w[t*TK + k] * dnAcc[invPerm[t*TK + k], h] + gate[t] * shared[t, h].
template <bool NOFMA_SHARED>
static inline void qwg_moe_down_reduce_combine_f32_body(device const float* dnAcc,
                                                        device const int32_t* invPerm,
                                                        device const float* w,
                                                        device const float* shared,
                                                        device const float* gate,
                                                        device float* out,
                                                        uint N,
                                                        uint TK,
                                                        uint tokens,
                                                        device half* out16,
                                                        uint hasOut16,
                                                        device const int32_t* ids,
                                                        uint2 tg,
                                                        uint tid) {
    const uint h = (tg.x * 256u + tid) * 4u;
    if (h >= N) return;
    const bool scatterPerRoute = (hasOut16 & 64u) != 0u;
    const bool writeOutBF16 = (hasOut16 & 0x10000u) != 0u;
    const bool writeOutF32 = (hasOut16 & 0x20000u) != 0u;
    if (scatterPerRoute) {
        if (tg.y >= tokens * TK) return;
        const bool dnIsF16 = (hasOut16 & 2u) != 0u;
        const bool writeOut16 = (hasOut16 & 1u) != 0u;
        const bool identityPerm = (hasOut16 & 4u) != 0u;
        const bool skipOut32 = (hasOut16 & 128u) != 0u;
        const bool hasIdsFilter = (hasOut16 & 32u) != 0u;
        if (hasIdsFilter && ids[tg.y] < 0) return;
        const int p = identityPerm ? int(tg.y) : invPerm[tg.y];
        if (p < 0) return;
        device const half* dnAcc16 = (device const half*)dnAcc;
        if (h + 4u <= N && ((N & 3u) == 0u)) {
            const float4 y4 = dnIsF16
                ? float4(*(device const half4*)(dnAcc16 + size_t(p) * N + h))
                : (*(device const float4*)(dnAcc + size_t(p) * N + h));
            const size_t outIdx = size_t(tg.y) * N + h;
            if (!skipOut32) *(device float4*)(out + outIdx) = y4;
            if (writeOut16) qwg_moe_combine_store16_4(out16, outIdx, y4, writeOutBF16, writeOutF32);
        } else {
            for (uint i = 0u; h + i < N; ++i) {
                const float y = dnIsF16 ? float(dnAcc16[size_t(p) * N + h + i])
                                        : dnAcc[size_t(p) * N + h + i];
                const size_t outIdx = size_t(tg.y) * N + h + i;
                if (!skipOut32) out[outIdx] = y;
                if (writeOut16) qwg_moe_combine_store16_1(out16, outIdx, y, writeOutBF16, writeOutF32);
            }
        }
        return;
    }
    const uint numGroups = (hasOut16 >> 8u) & 0xFFu;
    const uint t = (numGroups > 0u) ? (tg.y / numGroups) : tg.y;
    const uint targetGroup = (numGroups > 0u) ? (tg.y % numGroups) : 0u;
    if (t >= tokens) return;
    const uint basePair = t * TK;
    const bool dnIsF16 = (hasOut16 & 2u) != 0u;
    const bool writeOut16 = (hasOut16 & 1u) != 0u;
    const bool identityPerm = (hasOut16 & 4u) != 0u;
    const bool skipShared = numGroups > 0u || (hasOut16 & 8u) != 0u;
    const bool unweightedSingle = (hasOut16 & 16u) != 0u;
    const bool hasIdsFilter = numGroups > 0u || (hasOut16 & 32u) != 0u;
    const bool skipOut32 = writeOut16 && ((hasOut16 & 128u) != 0u);
    const float g = skipShared ? 0.0f : gate[t];
    device const half* dnAcc16 = (device const half*)dnAcc;
    if (h + 4u <= N && ((N & 3u) == 0u)) {
        const size_t inIdx = size_t(t) * N + h;
        float4 acc;
        if (unweightedSingle) {
            acc = dnIsF16 ? float4(*(device const half4*)(dnAcc16 + inIdx))
                          : (*(device const float4*)(dnAcc + inIdx));
        } else if (dnIsF16) {
            acc = qwg_moe_reduce_acc4<true>(dnAcc, invPerm, w, ids, N, TK, h, basePair,
                                            targetGroup, numGroups, identityPerm, hasIdsFilter);
        } else {
            acc = qwg_moe_reduce_acc4<false>(dnAcc, invPerm, w, ids, N, TK, h, basePair,
                                             targetGroup, numGroups, identityPerm, hasIdsFilter);
        }
        const uint outRow = (numGroups > 0u) ? tg.y : t;
        const size_t idx = size_t(outRow) * N + h;
        const float4 sh4 = skipShared ? float4(0.0f) : (*(device const float4*)(shared + idx));
        float4 y4;
        if (skipShared) {
            y4 = acc;
        } else if constexpr (NOFMA_SHARED) {
            y4 = qwg_moe_add_shared_nofma4(acc, g, sh4);
        } else {
            y4 = acc + g * sh4;
        }
        if (!skipOut32) *(device float4*)(out + idx) = y4;
        if (writeOut16) qwg_moe_combine_store16_4(out16, idx, y4, writeOutBF16, writeOutF32);
    } else {
        for (uint i = 0u; h + i < N; ++i) {
            const uint outRow = (numGroups > 0u) ? tg.y : t;
            const size_t idx = size_t(outRow) * N + h + i;
            float acc;
            if (unweightedSingle) {
                acc = dnIsF16 ? float(dnAcc16[idx]) : dnAcc[idx];
            } else {
                acc = qwg_moe_reduce_acc1(dnAcc, invPerm, w, ids, N, TK, h + i, basePair,
                                          targetGroup, numGroups, dnIsF16, identityPerm, hasIdsFilter);
            }
            float y;
            if (skipShared) {
                y = acc;
            } else if constexpr (NOFMA_SHARED) {
                y = qwg_moe_add_shared_nofma1(acc, g, shared[idx]);
            } else {
                y = acc + g * shared[idx];
            }
            if (!skipOut32) out[idx] = y;
            if (writeOut16) qwg_moe_combine_store16_1(out16, idx, y, writeOutBF16, writeOutF32);
        }
    }
}

kernel void qwg_moe_down_reduce_combine_f32(device const float* dnAcc [[buffer(0)]],
                                            device const int32_t* invPerm [[buffer(1)]],
                                            device const float* w [[buffer(2)]],
                                            device const float* shared [[buffer(3)]],
                                            device const float* gate [[buffer(4)]],
                                            device float* out [[buffer(5)]],
                                            constant uint& N [[buffer(6)]],
                                            constant uint& TK [[buffer(7)]],
                                            constant uint& tokens [[buffer(8)]],
                                            device half* out16 [[buffer(9)]],
                                            constant uint& hasOut16 [[buffer(10)]],
                                            device const int32_t* ids [[buffer(11)]],
                                            uint2 tg [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]]) {
    qwg_moe_down_reduce_combine_f32_body<false>(
        dnAcc, invPerm, w, shared, gate, out, N, TK, tokens, out16, hasOut16, ids, tg, tid);
}

kernel void qwg_moe_down_reduce_combine_f32_nofma(device const float* dnAcc [[buffer(0)]],
                                                  device const int32_t* invPerm [[buffer(1)]],
                                                  device const float* w [[buffer(2)]],
                                                  device const float* shared [[buffer(3)]],
                                                  device const float* gate [[buffer(4)]],
                                                  device float* out [[buffer(5)]],
                                                  constant uint& N [[buffer(6)]],
                                                  constant uint& TK [[buffer(7)]],
                                                  constant uint& tokens [[buffer(8)]],
                                                  device half* out16 [[buffer(9)]],
                                                  constant uint& hasOut16 [[buffer(10)]],
                                                  device const int32_t* ids [[buffer(11)]],
                                                  uint2 tg [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]]) {
    qwg_moe_down_reduce_combine_f32_body<true>(
        dnAcc, invPerm, w, shared, gate, out, N, TK, tokens, out16, hasOut16, ids, tg, tid);
}
