// qw_decision_heads.metal — Sparse BF16 verbalizer gather, marker/classifier MLP, pooling, and span pointer heads.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float qw_erf_exact(float x) {
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

struct QWSparseVocabParams {
    uint rows;
    uint candCount;
    uint hidden;
    uint vocabSize;
};

kernel void qwg_head_sparse_vocab_rows_bf16(
    device const half*           hiddenState  [[buffer(0)]],
    device const ushort*         lmHeadBF16   [[buffer(1)]],
    device const uint*           candidateIds [[buffer(2)]],
    device float*                out          [[buffer(3)]],
    constant QWSparseVocabParams& p           [[buffer(4)]],
    uint2 tg                                  [[threadgroup_position_in_grid]],
    uint ti                                   [[thread_index_in_threadgroup]]
) {
    uint c = tg.x;
    uint m = tg.y;
    if (c >= p.candCount || m >= p.rows) return;
    uint tok = candidateIds[c];
    if (tok >= p.vocabSize) {
        if (ti == 0u) { out[(size_t)m * p.candCount + c] = -1e9f; }
        return;
    }
    device const half* xRow = hiddenState + (size_t)m * p.hidden;
    device const ushort* wRow = lmHeadBF16 + (size_t)tok * p.hidden;
    float acc = 0.0f;
    for (uint i = ti; i < p.hidden; i += 32u) {
        float wVal = as_type<float>(uint(wRow[i]) << 16u);
        acc += float(xRow[i]) * wVal;
    }
    float total = simd_sum(acc);
    if (ti == 0u) { out[(size_t)m * p.candCount + c] = total; }
}

struct QWMarkerGatherParams {
    uint count;
    uint hidden;
    uint seqLen;
    uint hasTypeEmb;
};

kernel void qwg_decision_marker_gather_f16(
    device const half*             hidden    [[buffer(0)]],
    device const int*              positions [[buffer(1)]],
    device const half*             typeEmb   [[buffer(2)]],
    device half*                   out       [[buffer(3)]],
    constant QWMarkerGatherParams& p         [[buffer(4)]],
    uint2 tid                                [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint k = tid.y;
    if (d >= p.hidden || k >= p.count) return;
    int pos = positions[k];
    float v = 0.0f;
    if (pos >= 0 && uint(pos) < p.seqLen) {
        v = float(hidden[(size_t)uint(pos) * p.hidden + d]);
    }
    if (p.hasTypeEmb != 0u) { v += float(typeEmb[d]); }
    out[(size_t)k * p.hidden + d] = half(v);
}

struct QWDecisionLinearParams {
    uint rows;
    uint outFeatures;
    uint inFeatures;
    uint hasBias;
    uint actMode;   // 0=none, 1=gelu, 2=silu
    uint xIsFloat;
};

kernel void qwg_decision_mlp_linear_f32(
    device const half*               xHalf   [[buffer(0)]],
    device const float*              xFloat  [[buffer(1)]],
    device const half*               w       [[buffer(2)]],
    device const half*               bias    [[buffer(3)]],
    device float*                    y       [[buffer(4)]],
    constant QWDecisionLinearParams& p       [[buffer(5)]],
    uint2 tg                                 [[threadgroup_position_in_grid]],
    uint ti                                  [[thread_index_in_threadgroup]]
) {
    uint n = tg.x;
    uint m = tg.y;
    if (n >= p.outFeatures || m >= p.rows) return;
    device const half* wRow = w + (size_t)n * p.inFeatures;
    float acc = 0.0f;
    if (p.xIsFloat != 0u) {
        device const float* xr = xFloat + (size_t)m * p.inFeatures;
        for (uint k = ti; k < p.inFeatures; k += 32u) { acc += xr[k] * float(wRow[k]); }
    } else {
        device const half* xr = xHalf + (size_t)m * p.inFeatures;
        for (uint k = ti; k < p.inFeatures; k += 32u) { acc += float(xr[k]) * float(wRow[k]); }
    }
    float total = simd_sum(acc);
    if (ti == 0u) {
        if (p.hasBias != 0u) { total += float(bias[n]); }
        if (p.actMode == 1u) {
            total = 0.5f * total * (1.0f + qw_erf_exact(total * 0.7071067811865475f));
        } else if (p.actMode == 2u) {
            total = total / (1.0f + exp(-total));
        }
        y[(size_t)m * p.outFeatures + n] = total;
    }
}

struct QWDecisionPoolParams {
    uint seqLen;
    uint hidden;
    uint validLen;
    uint poolMode;  // 0=firstToken, 1=lastToken, 2=meanNonPad
};

kernel void qwg_decision_pool_f16(
    device const half*            hidden [[buffer(0)]],
    device half*                  out    [[buffer(1)]],
    constant QWDecisionPoolParams& p     [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    if (tid >= p.hidden) return;
    uint len = min(p.seqLen, max(1u, p.validLen));
    if (p.poolMode == 0u) {
        out[tid] = hidden[tid];
    } else if (p.poolMode == 1u) {
        out[tid] = hidden[(size_t)(len - 1u) * p.hidden + tid];
    } else {
        float acc = 0.0f;
        for (uint t = 0u; t < len; t++) { acc += float(hidden[(size_t)t * p.hidden + tid]); }
        out[tid] = half(acc / float(len));
    }
}

struct QWPointerParams {
    uint stateLen;
    uint hidden;
    float invSqrtH;
};

kernel void qwg_decision_pointer_scores_f32(
    device const half*        stateH   [[buffer(0)]],
    device const half*        qStart   [[buffer(1)]],
    device const half*        qEnd     [[buffer(2)]],
    device float*             outStart [[buffer(3)]],
    device float*             outEnd   [[buffer(4)]],
    constant QWPointerParams& p        [[buffer(5)]],
    uint2 tg                           [[threadgroup_position_in_grid]],
    uint ti                            [[thread_index_in_threadgroup]]
) {
    uint s = tg.x;
    if (s >= p.stateLen || tg.y >= 2u) return;
    device const half* sr = stateH + (size_t)s * p.hidden;
    device const half* q = (tg.y == 0u) ? qStart : qEnd;
    float acc = 0.0f;
    for (uint i = ti; i < p.hidden; i += 32u) { acc += float(sr[i]) * float(q[i]); }
    float score = simd_sum(acc) * p.invSqrtH;
    if (ti == 0u) {
        if (tg.y == 0u) { outStart[s] = score; } else { outEnd[s] = score; }
    }
}
