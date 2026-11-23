// qw_ink2_attn.metal — Inkling-2 QKVR prep, KV f16/q8/q4 write, banded rel-bias GQA, split combine.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

inline void qw_ink2_write_kv_head(half4 vals, uint lane, uint slot, uint kvh, uint nKV, uint D,
                                  uint kvMode, device uchar* cacheBuf, device uchar* metaBuf) {
    const size_t rowHead = (size_t(slot) * nKV + kvh);
    if (kvMode == 0u) {
        device half* dst = ((device half*)cacheBuf) + rowHead * D + lane * 4u;
        *(device half4*)dst = vals;
    } else if (kvMode == 1u) {
        float ma = max(max(abs(float(vals.x)), abs(float(vals.y))),
                       max(abs(float(vals.z)), abs(float(vals.w))));
        float headMax = simd_max(ma);
        float scF = max(headMax / 127.0f, 6.103515625e-5f);
        half sc16 = half(scF);
        float invSc = (float(sc16) > 0.0f) ? (1.0f / float(sc16)) : 0.0f;
        char4 qv;
        qv.x = char(clamp(int(rint(float(vals.x) * invSc)), -127, 127));
        qv.y = char(clamp(int(rint(float(vals.y) * invSc)), -127, 127));
        qv.z = char(clamp(int(rint(float(vals.z) * invSc)), -127, 127));
        qv.w = char(clamp(int(rint(float(vals.w) * invSc)), -127, 127));
        device char* dst = ((device char*)cacheBuf) + rowHead * D + lane * 4u;
        *(device char4*)dst = qv;
        if (lane == 0u) {
            ((device half*)metaBuf)[rowHead] = sc16;
        }
    } else {
        float mn = min(min(float(vals.x), float(vals.y)), min(float(vals.z), float(vals.w)));
        float mx = max(max(float(vals.x), float(vals.y)), max(float(vals.z), float(vals.w)));
        float headMin = simd_min(mn);
        float headMax = simd_max(mx);
        half mn16 = half(headMin);
        half sc16 = half(max((headMax - float(mn16)) / 15.0f, 6.103515625e-5f));
        float invSc = 1.0f / max(float(sc16), 1e-7f);
        uint n0 = clamp(uint(rint((float(vals.x) - float(mn16)) * invSc)), 0u, 15u);
        uint n1 = clamp(uint(rint((float(vals.y) - float(mn16)) * invSc)), 0u, 15u);
        uint n2 = clamp(uint(rint((float(vals.z) - float(mn16)) * invSc)), 0u, 15u);
        uint n3 = clamp(uint(rint((float(vals.w) - float(mn16)) * invSc)), 0u, 15u);
        uchar2 pb = uchar2(uchar(n0 | (n1 << 4u)), uchar(n2 | (n3 << 4u)));
        device uchar* dst = cacheBuf + rowHead * (D >> 1u) + lane * 2u;
        *(device uchar2*)dst = pb;
        if (lane == 0u) {
            ((device half2*)metaBuf)[rowHead] = half2(sc16, mn16);
        }
    }
}

inline half4 qw_ink2_read_kv_head(uint lane, uint slot, uint kvh, uint nKV, uint D, uint kvMode,
                                   device const uchar* cacheBuf, device const uchar* metaBuf) {
    const size_t rowHead = size_t(slot) * nKV + kvh;
    if (kvMode == 0u) {
        return *(device const half4*)(((device const half*)cacheBuf) + rowHead * D + lane * 4u);
    } else if (kvMode == 1u) {
        half sc = ((device const half*)metaBuf)[rowHead];
        char4 c = *(device const char4*)(((device const char*)cacheBuf) + rowHead * D + lane * 4u);
        return half4(half(c.x) * sc, half(c.y) * sc, half(c.z) * sc, half(c.w) * sc);
    } else {
        half2 sm = ((device const half2*)metaBuf)[rowHead];
        uchar2 b = *(device const uchar2*)(cacheBuf + rowHead * (D >> 1u) + lane * 2u);
        half sc = sm.x, mn = sm.y;
        return half4(fma(half(b.x & 0x0Fu), sc, mn), fma(half(b.x >> 4u), sc, mn),
                     fma(half(b.y & 0x0Fu), sc, mn), fma(half(b.y >> 4u), sc, mn));
    }
}

static inline void qw_ink2_qkvr_prep_kv_write_impl(
        device const bfloat* qIn, device const bfloat* kIn, device const bfloat* vIn,
        device const bfloat* rIn, device const bfloat* qScale, device const bfloat* relScale,
        device bfloat* qOut, device bfloat* rOut,
        device uchar* kCache, device uchar* vCache, device uchar* kMeta, device uchar* vMeta,
        constant QWInk2QKVRParams& p, uint2 tg, uint lane, uint logicalT) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);

    float tau = 1.0f;
    if (p.isSliding == 0u && p.nFloor > 0u) {
        float ratio = float(p.posOffset + logicalT + 1u) / float(p.nFloor);
        if (ratio > 1.0f) tau = 1.0f + p.logAlpha * log(ratio);
    }

    if (h < p.nH) {
        if (activeD) {
            const size_t qBase = (size_t(t) * p.nH + h) * p.headDim + dBase;
            const size_t sBase = size_t(h) * p.headDim + dBase;
            float4 qv = float4(*(device const bfloat4*)(qIn + qBase)) * float4(*(device const bfloat4*)(qScale + sBase));
            float normQ = sqrt(simd_sum(dot(qv, qv)));
            float scQ = min(p.qkClip / (normQ + 1e-6f), 1.0f);
            float4 qScaled = qv * scQ;
            float4 qClipped = float4(qw_ink2_bf16(qScaled.x), qw_ink2_bf16(qScaled.y),
                                     qw_ink2_bf16(qScaled.z), qw_ink2_bf16(qScaled.w));
            if (p.isSliding == 0u && tau != 1.0f) qClipped *= tau;
            qOut[qBase + 0u] = qw_ink2_bf16(qClipped.x);
            qOut[qBase + 1u] = qw_ink2_bf16(qClipped.y);
            qOut[qBase + 2u] = qw_ink2_bf16(qClipped.z);
            qOut[qBase + 3u] = qw_ink2_bf16(qClipped.w);
        }
        float rv = (lane < p.dRel)
            ? float(rIn[(size_t(t) * p.nH + h) * p.dRel + lane]) * float(relScale[size_t(h) * p.dRel + lane])
            : 0.0f;
        float normR = sqrt(simd_sum(rv * rv));
        float scR = min(p.relClip / (normR + 1e-6f), 1.0f);
        if (lane < p.dRel) {
            float rc = float(qw_ink2_bf16(rv * scR));
            if (p.isSliding == 0u && tau != 1.0f) rc *= tau;
            rOut[(size_t(t) * p.nH + h) * p.dRel + lane] = qw_ink2_bf16(rc);
        }
    }

    if (h < p.nKV && activeD) {
        const size_t kvBase = (size_t(t) * p.nKV + h) * p.headDim + dBase;
        float4 kv = float4(*(device const bfloat4*)(kIn + kvBase));
        float normK = sqrt(simd_sum(dot(kv, kv)));
        float scK = min(p.qkClip / (normK + 1e-6f), 1.0f);
        float4 kc = kv * scK;
        half4 k16 = half4(half(float(qw_ink2_bf16(kc.x))), half(float(qw_ink2_bf16(kc.y))),
                          half(float(qw_ink2_bf16(kc.z))), half(float(qw_ink2_bf16(kc.w))));

        float4 vv = float4(*(device const bfloat4*)(vIn + kvBase));
        float sumSqV = simd_sum(dot(vv, vv));
        float invV = rsqrt(sumSqV / float(p.headDim) + p.rmsEps);
        float4 vc = vv * invV;
        half4 v16 = half4(half(float(qw_ink2_bf16(vc.x))), half(float(qw_ink2_bf16(vc.y))),
                          half(float(qw_ink2_bf16(vc.z))), half(float(qw_ink2_bf16(vc.w))));

        const uint absPos = p.posOffset + t;
        const uint slot = (p.isSliding != 0u) ? (absPos % p.ringCap) : absPos;
        qw_ink2_write_kv_head(k16, lane, slot, h, p.nKV, p.headDim, p.kvMode, kCache, kMeta);
        qw_ink2_write_kv_head(v16, lane, slot, h, p.nKV, p.headDim, p.kvMode, vCache, vMeta);
    }
}

kernel void qwg_ink2_qkvr_prep_kv_write(device const bfloat* qIn [[buffer(0)]],
                                        device const bfloat* kIn [[buffer(1)]],
                                        device const bfloat* vIn [[buffer(2)]],
                                        device const bfloat* rIn [[buffer(3)]],
                                        device const bfloat* qScale [[buffer(4)]],
                                        device const bfloat* relScale [[buffer(5)]],
                                        device bfloat* qOut [[buffer(6)]],
                                        device bfloat* rOut [[buffer(7)]],
                                        device uchar* kCache [[buffer(8)]],
                                        device uchar* vCache [[buffer(9)]],
                                        device uchar* kMeta [[buffer(10)]],
                                        device uchar* vMeta [[buffer(11)]],
                                        constant QWInk2QKVRParams& p [[buffer(12)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= p.T) return;
    qw_ink2_qkvr_prep_kv_write_impl(qIn, kIn, vIn, rIn, qScale, relScale, qOut, rOut,
                                    kCache, vCache, kMeta, vMeta, p, tg, lane, tg.y);
}

struct QWInk2TreeDepthsParams {
    uint depths[16];
};

kernel void qwg_ink2_qkvr_prep_kv_write_tree(device const bfloat* qIn [[buffer(0)]],
                                             device const bfloat* kIn [[buffer(1)]],
                                             device const bfloat* vIn [[buffer(2)]],
                                             device const bfloat* rIn [[buffer(3)]],
                                             device const bfloat* qScale [[buffer(4)]],
                                             device const bfloat* relScale [[buffer(5)]],
                                             device bfloat* qOut [[buffer(6)]],
                                             device bfloat* rOut [[buffer(7)]],
                                             device uchar* kCache [[buffer(8)]],
                                             device uchar* vCache [[buffer(9)]],
                                             device uchar* kMeta [[buffer(10)]],
                                             device uchar* vMeta [[buffer(11)]],
                                             constant QWInk2QKVRParams& p [[buffer(12)]],
                                             constant QWInk2TreeDepthsParams& td [[buffer(13)]],
                                             uint2 tg [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= p.T || tg.y >= 16u) return;
    qw_ink2_qkvr_prep_kv_write_impl(qIn, kIn, vIn, rIn, qScale, relScale, qOut, rOut,
                                    kCache, vCache, kMeta, vMeta, p, tg, lane, td.depths[tg.y]);
}

kernel void qwg_ink2_gqa_attn(device const bfloat* qBF16 [[buffer(0)]],
                              device const bfloat* rBF16 [[buffer(1)]],
                              device const bfloat* relProj [[buffer(2)]],
                              device const bfloat* sinkBF16 [[buffer(3)]],
                              device const bfloat* gateBF16 [[buffer(4)]],
                              device const uchar* kCache [[buffer(5)]],
                              device const uchar* vCache [[buffer(6)]],
                              device const uchar* kMeta [[buffer(7)]],
                              device const uchar* vMeta [[buffer(8)]],
                              device bfloat* outBF16 [[buffer(9)]],
                              device float* partO [[buffer(10)]],
                              device float2* partML [[buffer(11)]],
                              constant QWInk2AttnParams& p [[buffer(12)]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint sp = tg.x;
    const uint h = tg.y;
    const uint t = tg.z;
    if (sp >= p.splits || h >= p.nH || t >= p.T) return;

    const uint dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);
    const uint kvh = h / (p.nH / p.nKV);
    const float qkScale = rsqrt(float(p.headDim));

    float4 q4 = activeD
        ? float4(*(device const bfloat4*)(qBF16 + (size_t(t) * p.nH + h) * p.headDim + dBase))
        : float4(0.0f);
    float rLane = (lane < p.dRel) ? float(rBF16[(size_t(t) * p.nH + h) * p.dRel + lane]) : 0.0f;

    const uint qPos = p.posOffset + t;
    const uint minKey = (p.sliding > 0u && qPos + 1u > p.sliding) ? (qPos + 1u - p.sliding) : 0u;
    const uint totalKeys = qPos - minKey + 1u;
    const uint chunk = (totalKeys + p.splits - 1u) / p.splits;
    const uint kStart = minKey + sp * chunk;
    const uint kEnd = min(qPos + 1u, kStart + chunk);

    float m = (sp == 0u) ? float(sinkBF16[size_t(t) * p.nH + h]) : -1e30f;
    float l = (sp == 0u) ? 1.0f : 0.0f;
    float4 acc = float4(0.0f);

    for (uint j = kStart; j < kEnd; ++j) {
        const uint slot = (p.sliding > 0u) ? (j % p.ringCap) : j;
        half4 k16 = activeD ? qw_ink2_read_kv_head(lane, slot, kvh, p.nKV, p.headDim, p.kvMode, kCache, kMeta)
                            : half4(0.0h);
        float score = simd_sum(dot(q4, float4(k16))) * qkScale;
        const uint dist = qPos - j;
        if (dist < p.relExtent) {
            float rb = (lane < p.dRel) ? rLane * float(relProj[size_t(lane) * p.relExtent + dist]) : 0.0f;
            score += simd_sum(rb);
        }
        half4 v16 = activeD ? qw_ink2_read_kv_head(lane, slot, kvh, p.nKV, p.headDim, p.kvMode, vCache, vMeta)
                            : half4(0.0h);
        float mNew = max(m, score);
        float a = exp(m - mNew);
        float b = exp(score - mNew);
        l = l * a + b;
        acc = acc * a + b * float4(v16);
        m = mNew;
    }

    if (p.splits == 1u) {
        if (activeD) {
            const size_t oBase = (size_t(t) * p.nH + h) * p.headDim + dBase;
            float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
            float4 res = (acc / l);
            res.x *= qw_ink2_sigmoid(g4.x);
            res.y *= qw_ink2_sigmoid(g4.y);
            res.z *= qw_ink2_sigmoid(g4.z);
            res.w *= qw_ink2_sigmoid(g4.w);
            outBF16[oBase + 0u] = qw_ink2_bf16(res.x);
            outBF16[oBase + 1u] = qw_ink2_bf16(res.y);
            outBF16[oBase + 2u] = qw_ink2_bf16(res.z);
            outBF16[oBase + 3u] = qw_ink2_bf16(res.w);
        }
    } else {
        if (activeD) {
            const size_t pBase = ((size_t(t) * p.nH + h) * p.splits + sp) * p.headDim + dBase;
            *(device float4*)(partO + pBase) = acc;
        }
        if (lane == 0u) {
            partML[(size_t(t) * p.nH + h) * p.splits + sp] = float2(m, l);
        }
    }
}

kernel void qwg_ink2_gqa_split_combine(device const float* partO [[buffer(0)]],
                                       device const float2* partML [[buffer(1)]],
                                       device const bfloat* gateBF16 [[buffer(2)]],
                                       device bfloat* outBF16 [[buffer(3)]],
                                       constant QWInk2AttnParams& p [[buffer(4)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x;
    const uint t = tg.y;
    if (h >= p.nH || t >= p.T) return;
    const uint dBase = lane * 4u;
    if (dBase >= p.headDim) return;

    const size_t mlBase = (size_t(t) * p.nH + h) * p.splits;
    float mMax = -1e30f;
    for (uint s = 0u; s < p.splits; ++s) mMax = max(mMax, partML[mlBase + s].x);

    float lSum = 0.0f;
    float4 acc = float4(0.0f);
    for (uint s = 0u; s < p.splits; ++s) {
        float2 ml = partML[mlBase + s];
        if (ml.y > 0.0f) {
            float scale = exp(ml.x - mMax);
            lSum += ml.y * scale;
            acc += (*(device const float4*)(partO + (mlBase + s) * p.headDim + dBase)) * scale;
        }
    }
    const size_t oBase = (size_t(t) * p.nH + h) * p.headDim + dBase;
    float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
    float4 res = acc / max(lSum, 1e-30f);
    res.x *= qw_ink2_sigmoid(g4.x);
    res.y *= qw_ink2_sigmoid(g4.y);
    res.z *= qw_ink2_sigmoid(g4.z);
    res.w *= qw_ink2_sigmoid(g4.w);
    outBF16[oBase + 0u] = qw_ink2_bf16(res.x);
    outBF16[oBase + 1u] = qw_ink2_bf16(res.y);
    outBF16[oBase + 2u] = qw_ink2_bf16(res.z);
    outBF16[oBase + 3u] = qw_ink2_bf16(res.w);
}
