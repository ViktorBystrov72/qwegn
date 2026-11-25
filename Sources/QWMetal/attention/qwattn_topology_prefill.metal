// qwattn_topology_prefill.metal — Topology-generalized attention & separate head-wise sigmoid gate.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct QWTopoAttnParams {
    uint seqLen;
    uint qHeads;
    uint kvHeads;
    uint headDim;
    uint topologyMode;   // 0=causal, 1=prefixBidir, 2=promptBidir, 3=packedSet, 4=bidirEncoder
    uint stateEndPos;
    uint mediaPrefixLen;
    uint slidingWindow;
    float scale;
};

kernel void qwg_attn_topology_prefill_f16(
    device const half*       q        [[buffer(0)]],
    device const half*       k        [[buffer(1)]],
    device const half*       v        [[buffer(2)]],
    device const int*        groupIds [[buffer(3)]],
    device half*             out      [[buffer(4)]],
    constant QWTopoAttnParams& p      [[buffer(5)]],
    uint2 tg                          [[threadgroup_position_in_grid]],
    uint ti                           [[thread_index_in_threadgroup]]
) {
    const uint h = tg.x;
    const uint qPos = tg.y;
    const uint lane = ti & 31u;
    if (h >= p.qHeads || qPos >= p.seqLen) return;

    const uint gqaRatio = max(1u, p.qHeads / max(1u, p.kvHeads));
    const uint kvHead = min(p.kvHeads - 1u, h / gqaRatio);
    device half* outRow = out + ((size_t)qPos * p.qHeads + h) * p.headDim;

    int qGroup = groupIds[qPos];
    if (qGroup < 0) {
        for (uint d = lane; d < p.headDim; d += 32u) { outRow[d] = 0.0h; }
        return;
    }

    const uint passes = (p.headDim + 31u) / 32u;
    device const half* qRow = q + ((size_t)qPos * p.qHeads + h) * p.headDim;
    float qReg[8];
    float oAcc[8];
    for (uint pass = 0u; pass < 8u; pass++) {
        uint d = lane + pass * 32u;
        qReg[pass] = (pass < passes && d < p.headDim) ? float(qRow[d]) : 0.0f;
        oAcc[pass] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    const size_t kvStride = (size_t)p.kvHeads * p.headDim;
    const size_t kvBase = (size_t)kvHead * p.headDim;

    for (uint kPos = 0u; kPos < p.seqLen; kPos++) {
        int kGroup = groupIds[kPos];
        bool allow = (kGroup >= 0);
        if (allow) {
            if (p.slidingWindow > 0u) {
                allow = (kPos <= qPos) && ((qPos - kPos) < p.slidingWindow);
            } else if (p.topologyMode == 0u) {
                allow = (kPos <= qPos);
            } else if (p.topologyMode == 1u) {
                allow = ((qPos < p.stateEndPos && kPos < p.stateEndPos) || (kPos <= qPos));
            } else if (p.topologyMode == 3u) {
                allow = (qGroup == 0) || (kGroup == 0) || (kGroup == qGroup);
            } else if (p.topologyMode == 4u) {
                allow = (p.mediaPrefixLen == 0u) || ((qPos < p.mediaPrefixLen) == (kPos < p.mediaPrefixLen));
            }
        }
        if (!allow) continue;

        device const half* kRow = k + (size_t)kPos * kvStride + kvBase;
        device const half* vRow = v + (size_t)kPos * kvStride + kvBase;
        float dot = 0.0f;
        for (uint pass = 0u; pass < passes; pass++) {
            uint d = lane + pass * 32u;
            if (d < p.headDim) { dot += qReg[pass] * float(kRow[d]); }
        }
        float score = simd_sum(dot) * p.scale;
        float mNew = fmax(m, score);
        float alpha = (l > 0.0f) ? exp(m - mNew) : 0.0f;
        float prob = exp(score - mNew);
        for (uint pass = 0u; pass < passes; pass++) {
            uint d = lane + pass * 32u;
            float vVal = (d < p.headDim) ? float(vRow[d]) : 0.0f;
            oAcc[pass] = oAcc[pass] * alpha + prob * vVal;
        }
        l = l * alpha + prob;
        m = mNew;
    }

    float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint pass = 0u; pass < passes; pass++) {
        uint d = lane + pass * 32u;
        if (d < p.headDim) { outRow[d] = half(oAcc[pass] * invL); }
    }
}

struct QWSigmoidGateParams {
    uint seqLen;
    uint qHeads;
    uint headDim;
};

kernel void qwg_attn_headwise_sigmoid_gate_f16(
    device half*                 out  [[buffer(0)]],
    device const half*           gate [[buffer(1)]],
    constant QWSigmoidGateParams& p   [[buffer(2)]],
    uint3 tid                         [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint h = tid.y;
    uint s = tid.z;
    if (d >= p.headDim || h >= p.qHeads || s >= p.seqLen) return;
    float g = 1.0f / (1.0f + exp(-float(gate[(size_t)s * p.qHeads + h])));
    size_t idx = ((size_t)s * p.qHeads + h) * p.headDim + d;
    out[idx] = half(float(out[idx]) * g);
}
