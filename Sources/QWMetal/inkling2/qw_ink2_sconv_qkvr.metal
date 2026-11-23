// qw_ink2_sconv_qkvr.metal — Fused static K/V SConv + QKVR prep + KV cache write kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

struct QWInk2QKVRSConvFusedParams {
    uint T, nH, nKV, headDim, dRel;
    uint isSliding, posOffset, ringCap, kvMode, nFloor;
    float qkClip, relClip, logAlpha, rmsEps;
    uint kvInRowStride, updateState, writeVerifyTap, kVerifyTapOffset, vVerifyTapOffset, writeKConv;
    float invSqrtDenom;
    uint pad0, pad1, pad2;
};

struct QWInk2QKVRSConvTreeFusedParams {
    uint T, nH, nKV, headDim, dRel;
    uint isSliding, posOffset, ringCap, kvMode, nFloor;
    float qkClip, relClip, logAlpha, rmsEps;
    uint kvInRowStride, updateState, writeVerifyTap, kVerifyTapOffset, vVerifyTapOffset, writeKConv;
    float invSqrtDenom;
    uint pad0, pad1, pad2;
    uint depths[16];
    int4 treeLags[16];
};

struct QWInk2SConvCommitPairParams {
    uint T, C, kvInRowStride, pad0;
};

static inline void qw_ink2_sconv_fused_write_kv(half4 vals, uint lane, uint slot, uint kvh, uint nKV, uint D,
                                                uint kvMode, device uchar* cacheBuf, device uchar* metaBuf) {
    const size_t rowHead = size_t(slot) * nKV + kvh;
    if (kvMode == 0u) {
        device half* dst = ((device half*)cacheBuf) + rowHead * D + lane * 4u;
        *(device half4*)dst = vals;
    } else if (kvMode == 1u) {
        const float ma = max(max(abs(float(vals.x)), abs(float(vals.y))),
                             max(abs(float(vals.z)), abs(float(vals.w))));
        const float headMax = simd_max(ma);
        const half sc16 = half(max(headMax / 127.0f, 6.103515625e-5f));
        const float invSc = (float(sc16) > 0.0f) ? (1.0f / float(sc16)) : 0.0f;
        char4 qv;
        qv.x = char(clamp(int(rint(float(vals.x) * invSc)), -127, 127));
        qv.y = char(clamp(int(rint(float(vals.y) * invSc)), -127, 127));
        qv.z = char(clamp(int(rint(float(vals.z) * invSc)), -127, 127));
        qv.w = char(clamp(int(rint(float(vals.w) * invSc)), -127, 127));
        *(device char4*)(((device char*)cacheBuf) + rowHead * D + lane * 4u) = qv;
        if (lane == 0u) ((device half*)metaBuf)[rowHead] = sc16;
    } else {
        const float mn = min(min(float(vals.x), float(vals.y)), min(float(vals.z), float(vals.w)));
        const float mx = max(max(float(vals.x), float(vals.y)), max(float(vals.z), float(vals.w)));
        const half mn16 = half(simd_min(mn));
        const half sc16 = half(max((simd_max(mx) - float(mn16)) / 15.0f, 6.103515625e-5f));
        const float invSc = 1.0f / max(float(sc16), 1e-7f);
        const uint n0 = clamp(uint(rint((float(vals.x) - float(mn16)) * invSc)), 0u, 15u);
        const uint n1 = clamp(uint(rint((float(vals.y) - float(mn16)) * invSc)), 0u, 15u);
        const uint n2 = clamp(uint(rint((float(vals.z) - float(mn16)) * invSc)), 0u, 15u);
        const uint n3 = clamp(uint(rint((float(vals.w) - float(mn16)) * invSc)), 0u, 15u);
        *(device uchar2*)(cacheBuf + rowHead * (D >> 1u) + lane * 2u) =
            uchar2(uchar(n0 | (n1 << 4u)), uchar(n2 | (n3 << 4u)));
        if (lane == 0u) ((device half2*)metaBuf)[rowHead] = half2(sc16, mn16);
    }
}

static inline float qw_ink2_sconv_sample_strided(
        device const bfloat* xIn, device const float* stateF32,
        uint C, uint rowStride, float s0, float s1, float s2, uint c, int idx) {
    if (idx == -3) return s0;
    if (idx == -2) return s1;
    if (idx == -1) return s2;
    if (idx < -3) return stateF32[size_t(3 + idx) * C + c];
    return float(xIn[size_t(idx) * rowStride + c]);
}

static inline bfloat qw_ink2_sconv_static_chan(
        device const bfloat* xIn, device const bfloat* weight, device float* stateF32,
        device bfloat* verifyTap, uint C, uint rowStride, uint T, uint t, uint c,
        uint writeVerifyTap, uint verifyTapOffset, float invSqrtDenom,
        bool updateInline, bool isTree, int4 lag) {
#pragma clang fp contract(off) reassociate(off)
    device const bfloat* wc = weight + size_t(c) * 4u;
    const float w0 = float(wc[0]), w1 = float(wc[1]), w2 = float(wc[2]), w3 = float(wc[3]);
    const float s0 = stateF32[0u * C + c];
    const float s1 = stateF32[1u * C + c];
    const float s2 = stateF32[2u * C + c];
    const bfloat xRaw = xIn[size_t(t) * rowStride + c];
    if (writeVerifyTap != 0u) verifyTap[size_t(verifyTapOffset) + size_t(t) * C + c] = xRaw;
    const float xCur = float(xRaw);
    float acc;
    if (isTree) {
        acc = w0 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, lag.z)
            + w1 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, lag.y)
            + w2 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, lag.x)
            + w3 * xCur;
    } else {
        const int ti = int(t);
        acc = w0 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, ti - 3)
            + w1 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, ti - 2)
            + w2 * qw_ink2_sconv_sample_strided(xIn, stateF32, C, rowStride, s0, s1, s2, c, ti - 1)
            + w3 * xCur;
    }
    const bfloat out = qw_ink2_bf16((acc + xCur) * invSqrtDenom);
    if (updateInline && T == 1u) {
        stateF32[0u * C + c] = s1;
        stateF32[1u * C + c] = s2;
        stateF32[2u * C + c] = xCur;
    }
    return out;
}

template <typename P>
static inline void qw_ink2_qkvr_sconv_static_body(
        device const bfloat* qIn, device const bfloat* kIn, device const bfloat* vIn,
        device const bfloat* rIn, device const bfloat* kWeight, device const bfloat* vWeight,
        device float* kStateF32, device float* vStateF32,
        device const bfloat* qScale, device const bfloat* relScale,
        device bfloat* qOut, device bfloat* rOut,
        device uchar* kCache, device uchar* vCache, device uchar* kMeta, device uchar* vMeta,
        device bfloat* verifyTap, device bfloat* kConvOut,
        constant P& p, uint2 tg, uint lane, uint logicalT, bool isTree, int4 lag) {
#pragma clang fp contract(off) reassociate(off)
    const uint h = tg.x, t = tg.y;
    const uint dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);

    float tau = 1.0f;
    if (p.isSliding == 0u && p.nFloor > 0u) {
        const float ratio = float(p.posOffset + logicalT + 1u) / float(p.nFloor);
        if (ratio > 1.0f) tau = 1.0f + p.logAlpha * log(ratio);
    }

    if (h < p.nH) {
        if (activeD) {
            const size_t qBase = (size_t(t) * p.nH + h) * p.headDim + dBase;
            const size_t sBase = size_t(h) * p.headDim + dBase;
            const bool vecQ = ((p.headDim & 3u) == 0u) && ((size_t(qIn) & 7u) == 0u) && ((size_t(qScale) & 7u) == 0u);
            float4 qRaw = vecQ
                ? float4(*(device const bfloat4*)(qIn + qBase))
                : float4(float(qIn[qBase]), float(qIn[qBase + 1u]), float(qIn[qBase + 2u]), float(qIn[qBase + 3u]));
            float4 qSc = vecQ
                ? float4(*(device const bfloat4*)(qScale + sBase))
                : float4(float(qScale[sBase]), float(qScale[sBase + 1u]), float(qScale[sBase + 2u]), float(qScale[sBase + 3u]));
            const float4 qv = qRaw * qSc;
            const float normQ = sqrt(simd_sum(dot(qv, qv)));
            const float scQ = min(p.qkClip / (normQ + 1e-6f), 1.0f);
            const float4 qScaled = qv * scQ;
            float4 qClipped = float4(qw_ink2_bf16(qScaled.x), qw_ink2_bf16(qScaled.y),
                                     qw_ink2_bf16(qScaled.z), qw_ink2_bf16(qScaled.w));
            if (p.isSliding == 0u && tau != 1.0f) qClipped *= tau;
            qOut[qBase + 0u] = qw_ink2_bf16(qClipped.x);
            qOut[qBase + 1u] = qw_ink2_bf16(qClipped.y);
            qOut[qBase + 2u] = qw_ink2_bf16(qClipped.z);
            qOut[qBase + 3u] = qw_ink2_bf16(qClipped.w);
        }
        const float rv = (lane < p.dRel)
            ? float(rIn[(size_t(t) * p.nH + h) * p.dRel + lane]) * float(relScale[size_t(h) * p.dRel + lane])
            : 0.0f;
        const float normR = sqrt(simd_sum(rv * rv));
        const float scR = min(p.relClip / (normR + 1e-6f), 1.0f);
        if (lane < p.dRel) {
            float rc = float(qw_ink2_bf16(rv * scR));
            if (p.isSliding == 0u && tau != 1.0f) rc *= tau;
            rOut[(size_t(t) * p.nH + h) * p.dRel + lane] = qw_ink2_bf16(rc);
        }
    }

    if (h < p.nKV && activeD) {
        const uint C = p.nKV * p.headDim;
        const bool updInline = (!isTree && p.T == 1u && p.updateState != 0u);
        float4 kv, vv;
        for (uint i = 0u; i < 4u; ++i) {
            const uint c = h * p.headDim + dBase + i;
            const bfloat kb = qw_ink2_sconv_static_chan(
                kIn, kWeight, kStateF32, verifyTap, C, p.kvInRowStride, p.T, t, c,
                p.writeVerifyTap, p.kVerifyTapOffset, p.invSqrtDenom, updInline, isTree, lag);
            if (p.writeKConv != 0u) kConvOut[size_t(t) * C + c] = kb;
            kv[i] = float(kb);
            const bfloat vb = qw_ink2_sconv_static_chan(
                vIn, vWeight, vStateF32, verifyTap, C, p.kvInRowStride, p.T, t, c,
                p.writeVerifyTap, p.vVerifyTapOffset, p.invSqrtDenom, updInline, isTree, lag);
            vv[i] = float(vb);
        }
        const float normK = sqrt(simd_sum(dot(kv, kv)));
        const float scK = min(p.qkClip / (normK + 1e-6f), 1.0f);
        const float4 kc = kv * scK;
        const half4 k16 = half4(half(float(qw_ink2_bf16(kc.x))), half(float(qw_ink2_bf16(kc.y))),
                                half(float(qw_ink2_bf16(kc.z))), half(float(qw_ink2_bf16(kc.w))));
        const float sumSqV = simd_sum(dot(vv, vv));
        const float invV = rsqrt(sumSqV / float(p.headDim) + p.rmsEps);
        const float4 vc = vv * invV;
        const half4 v16 = half4(half(float(qw_ink2_bf16(vc.x))), half(float(qw_ink2_bf16(vc.y))),
                                half(float(qw_ink2_bf16(vc.z))), half(float(qw_ink2_bf16(vc.w))));
        const uint absPos = p.posOffset + t;
        const uint slot = (p.isSliding != 0u) ? (absPos % p.ringCap) : absPos;
        qw_ink2_sconv_fused_write_kv(k16, lane, slot, h, p.nKV, p.headDim, p.kvMode, kCache, kMeta);
        qw_ink2_sconv_fused_write_kv(v16, lane, slot, h, p.nKV, p.headDim, p.kvMode, vCache, vMeta);
    }
}

kernel void qwg_ink2_qkvr_sconv_static_fused(
        device const bfloat* qIn [[buffer(0)]], device const bfloat* kIn [[buffer(1)]],
        device const bfloat* vIn [[buffer(2)]], device const bfloat* rIn [[buffer(3)]],
        device const bfloat* kWeight [[buffer(4)]], device const bfloat* vWeight [[buffer(5)]],
        device float* kStateF32 [[buffer(6)]], device float* vStateF32 [[buffer(7)]],
        device const bfloat* qScale [[buffer(8)]], device const bfloat* relScale [[buffer(9)]],
        device bfloat* qOut [[buffer(10)]], device bfloat* rOut [[buffer(11)]],
        device uchar* kCache [[buffer(12)]], device uchar* vCache [[buffer(13)]],
        device uchar* kMeta [[buffer(14)]], device uchar* vMeta [[buffer(15)]],
        device bfloat* verifyTap [[buffer(16)]], device bfloat* kConvOut [[buffer(17)]],
        constant QWInk2QKVRSConvFusedParams& p [[buffer(18)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= p.T) return;
    qw_ink2_qkvr_sconv_static_body(
        qIn, kIn, vIn, rIn, kWeight, vWeight, kStateF32, vStateF32,
        qScale, relScale, qOut, rOut, kCache, vCache, kMeta, vMeta,
        verifyTap, kConvOut, p, tg, lane, tg.y, false, int4(0));
}

kernel void qwg_ink2_qkvr_sconv_static_tree_fused(
        device const bfloat* qIn [[buffer(0)]], device const bfloat* kIn [[buffer(1)]],
        device const bfloat* vIn [[buffer(2)]], device const bfloat* rIn [[buffer(3)]],
        device const bfloat* kWeight [[buffer(4)]], device const bfloat* vWeight [[buffer(5)]],
        device float* kStateF32 [[buffer(6)]], device float* vStateF32 [[buffer(7)]],
        device const bfloat* qScale [[buffer(8)]], device const bfloat* relScale [[buffer(9)]],
        device bfloat* qOut [[buffer(10)]], device bfloat* rOut [[buffer(11)]],
        device uchar* kCache [[buffer(12)]], device uchar* vCache [[buffer(13)]],
        device uchar* kMeta [[buffer(14)]], device uchar* vMeta [[buffer(15)]],
        device bfloat* verifyTap [[buffer(16)]], device bfloat* kConvOut [[buffer(17)]],
        constant QWInk2QKVRSConvTreeFusedParams& p [[buffer(18)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (tg.y >= p.T || tg.y >= 16u) return;
    qw_ink2_qkvr_sconv_static_body(
        qIn, kIn, vIn, rIn, kWeight, vWeight, kStateF32, vStateF32,
        qScale, relScale, qOut, rOut, kCache, vCache, kMeta, vMeta,
        verifyTap, kConvOut, p, tg, lane, p.depths[tg.y], true, p.treeLags[tg.y]);
}

kernel void qwg_ink2_sconv_commit_kv_pair(device const bfloat* kIn [[buffer(0)]],
                                          device const bfloat* vIn [[buffer(1)]],
                                          device float* kStateF32 [[buffer(2)]],
                                          device float* vStateF32 [[buffer(3)]],
                                          constant QWInk2SConvCommitPairParams& p [[buffer(4)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, isV = gid.y;
    if (c >= p.C || isV >= 2u || p.T == 0u) return;
    device const bfloat* xIn = (isV == 0u) ? kIn : vIn;
    device float* st = (isV == 0u) ? kStateF32 : vStateF32;
    const float s1 = st[1u * p.C + c], s2 = st[2u * p.C + c];
    const int ti = int(p.T);
    const float n0 = (ti >= 3) ? float(xIn[size_t(ti - 3) * p.kvInRowStride + c]) : ((ti == 2) ? s2 : s1);
    const float n1 = (ti >= 2) ? float(xIn[size_t(ti - 2) * p.kvInRowStride + c]) : s2;
    const float n2 = float(xIn[size_t(ti - 1) * p.kvInRowStride + c]);
    st[0u * p.C + c] = n0;
    st[1u * p.C + c] = n1;
    st[2u * p.C + c] = n2;
}
