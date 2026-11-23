// qwattn_sink_rel_decode.metal — High-occupancy 256-thread T=1 Split-KV decode & combine kernels
// with optional attention sink, banded relative position bias, and output gate (f16/q8/q4 KV).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "qw_attn_sink_rel.metalh"
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

template <uint MODE>
METAL_FUNC void qwg_attn_decode_sink_rel_gqa_body(
    device const bfloat* qBF16, device const bfloat* rBF16, device const bfloat* relProj,
    device const bfloat* sinkBF16, device const bfloat* gateBF16,
    device const uchar* kCache, device const uchar* vCache,
    device const uchar* kMeta, device const uchar* vMeta,
    device bfloat* outBF16, device float* partO, device float2* partML,
    constant QWSinkRelAttnParams& p, uint3 tg, uint sgi, uint lane,
    threadgroup float (*scoreSh)[8], threadgroup half4 (*vSh)[33]
) {
    #pragma clang fp contract(off) reassociate(off)
    const uint sp = tg.x, kvh = tg.y, t = tg.z;
    if (sp >= p.splits || kvh >= p.nKV || t >= p.T) return;

    const uint gqa = p.nH / p.nKV, headBase = kvh * gqa, dBase = lane * 4u;
    const bool activeD = (dBase < p.headDim);
    const float qkScale = rsqrt(float(p.headDim));

    const uint qPos = p.posOffset + t;
    const uint minKey = (p.sliding > 0u && qPos + 1u > p.sliding) ? (qPos + 1u - p.sliding) : 0u;
    const uint kvEnd = min(qPos + 1u, p.kvLen);
    uint kStart = 0u, kEnd = 0u;
    bool hasSinkHere = false;
    if (p.splitSpan > 0u) {
        const uint globalMin = (p.sliding > 0u && p.posOffset + 1u > p.sliding) ? (p.posOffset + 1u - p.sliding) : 0u;
        const uint firstBlock = min(globalMin, p.kvLen) / p.splitSpan;
        const uint blockStart = (firstBlock + sp) * p.splitSpan;
        const uint blockEnd = blockStart + p.splitSpan;
        kStart = max(blockStart, minKey);
        kEnd = min(blockEnd, kvEnd);
        hasSinkHere = (p.hasSink != 0u) && (minKey >= blockStart) && (minKey < blockEnd) && (minKey < kvEnd);
    } else {
        const uint totalKeys = (kvEnd > minKey) ? (kvEnd - minKey) : 0u;
        const uint chunk = (totalKeys + p.splits - 1u) / p.splits;
        kStart = minKey + sp * chunk;
        kEnd = min(kvEnd, kStart + chunk);
        hasSinkHere = (sp == 0u) && (p.hasSink != 0u);
    }
    const bool checkRel = (p.hasRelBias != 0u) && (p.dRel > 0u) && (p.relExtent > 0u)
                        && (kStart < kEnd) && ((qPos - (kEnd - 1u)) < p.relExtent);

    for (uint g0 = 0u; g0 < gqa; g0 += 4u) {
        const uint gCount = min(4u, gqa - g0);
        float4 q4[4];
        float rLane[4];

        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            if (g < gCount) {
                const uint h = headBase + g0 + g;
                q4[g] = activeD
                    ? float4(*(device const bfloat4*)(qBF16 + (size_t(t) * p.nH + h) * p.headDim + dBase))
                    : float4(0.0f);
                rLane[g] = (p.hasRelBias != 0u && lane < p.dRel)
                    ? float(rBF16[(size_t(t) * p.nH + h) * p.dRel + lane]) : 0.0f;
            } else {
                q4[g] = float4(0.0f);
                rLane[g] = 0.0f;
            }
        }

        const bool ownsHead = (sgi < gCount);
        const uint hOwn = ownsHead ? (headBase + g0 + sgi) : 0u;
        float m = (ownsHead && hasSinkHere) ? float(sinkBF16[size_t(t) * p.nH + hOwn]) : -1e30f;
        float l = (ownsHead && hasSinkHere) ? 1.0f : 0.0f;
        float4 acc = float4(0.0f);

        for (uint j0 = kStart; j0 < kEnd; j0 += 8u) {
            const uint cnt = min(8u, kEnd - j0);
            const uint j = j0 + sgi;
            if (sgi < cnt) {
                const uint slot = (p.sliding > 0u) ? (j % p.ringCap) : j;
                const half4 k16 = activeD
                    ? qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, kCache, kMeta) : half4(0.0h);
                const half4 v16 = activeD
                    ? qwg_sink_rel_read_kv4<MODE>(lane, slot, kvh, p.nKV, p.headDim, vCache, vMeta) : half4(0.0h);
                if (activeD) vSh[sgi][lane] = v16;
                const float4 kf = float4(k16);
                const uint dist = qPos - j;
                const bool inRel = checkRel && (dist < p.relExtent);
                const float relW = (inRel && lane < p.dRel)
                    ? float(relProj[size_t(lane) * p.relExtent + dist]) : 0.0f;

                #pragma unroll
                for (uint g = 0u; g < 4u; ++g) {
                    if (g < gCount) {
                        float score = simd_sum(qwg_sink_rel_dot4(q4[g], kf)) * qkScale;
                        if (inRel) score += simd_sum(rLane[g] * relW);
                        if (lane == 0u) scoreSh[g][sgi] = score;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (ownsHead) {
                for (uint b = 0u; b < cnt; ++b) {
                    const float score = scoreSh[sgi][b];
                    const float4 vf = activeD ? float4(vSh[b][lane]) : float4(0.0f);
                    const float mNew = max(m, score);
                    const float a = (m > -1e29f) ? exp(m - mNew) : 0.0f;
                    const float bExp = exp(score - mNew);
                    l = l * a + bExp;
                    acc = acc * a + bExp * vf;
                    m = mNew;
                }
            }
            if (j0 + 8u < kEnd || g0 + 4u < gqa) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (ownsHead) {
            if (p.splits == 1u) {
                if (activeD) {
                    const size_t oBase = (size_t(t) * p.nH + hOwn) * p.headDim + dBase;
                    float4 res = acc / max(l, 1e-30f);
                    if (p.hasOutGate != 0u) {
                        const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
                        res.x *= qw_ink2_sigmoid(g4.x); res.y *= qw_ink2_sigmoid(g4.y);
                        res.z *= qw_ink2_sigmoid(g4.z); res.w *= qw_ink2_sigmoid(g4.w);
                    }
                    *(device bfloat4*)(outBF16 + oBase) = bfloat4(
                        qw_ink2_bf16(res.x), qw_ink2_bf16(res.y), qw_ink2_bf16(res.z), qw_ink2_bf16(res.w)
                    );
                }
            } else {
                const size_t mlIdx = (size_t(t) * p.nH + hOwn) * p.splits + sp;
                if (activeD) *(device float4*)(partO + mlIdx * p.headDim + dBase) = acc;
                if (lane == 0u) partML[mlIdx] = float2(m, l);
            }
        }
    }
}

#define QW_SINK_REL_DECODE_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_sink_rel_gqa##SFX( \
    device const bfloat* qBF16 [[buffer(0)]], device const bfloat* rBF16 [[buffer(1)]], \
    device const bfloat* relProj [[buffer(2)]], device const bfloat* sinkBF16 [[buffer(3)]], \
    device const bfloat* gateBF16 [[buffer(4)]], device const uchar* kCache [[buffer(5)]], \
    device const uchar* vCache [[buffer(6)]], device const uchar* kMeta [[buffer(7)]], \
    device const uchar* vMeta [[buffer(8)]], device bfloat* outBF16 [[buffer(9)]], \
    device float* partO [[buffer(10)]], device float2* partML [[buffer(11)]], \
    constant QWSinkRelAttnParams& p [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]], \
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scoreSh[4][8]; \
    threadgroup half4 vSh[8][33]; \
    qwg_attn_decode_sink_rel_gqa_body<MODE>( \
        qBF16, rBF16, relProj, sinkBF16, gateBF16, \
        kCache, vCache, kMeta, vMeta, outBF16, partO, partML, p, tg, sgi, lane, scoreSh, vSh); \
}
QW_SINK_REL_DECODE_ENTRY(_f16, 0u)
QW_SINK_REL_DECODE_ENTRY(_q8, 1u)
QW_SINK_REL_DECODE_ENTRY(_q4, 2u)
#undef QW_SINK_REL_DECODE_ENTRY

kernel void qwg_attn_decode_sink_rel_combine(
    device const float* partO [[buffer(0)]], device const float2* partML [[buffer(1)]],
    device const bfloat* gateBF16 [[buffer(2)]], device bfloat* outBF16 [[buffer(3)]],
    constant QWSinkRelAttnParams& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    #pragma clang fp contract(off) reassociate(off)
    const uint h = tg.x, t = tg.y;
    if (h >= p.nH || t >= p.T) return;
    const uint dBase = lane * 4u;
    if (dBase >= p.headDim) return;

    const size_t mlBase = (size_t(t) * p.nH + h) * p.splits;
    float mMax = -1e30f;
    for (uint s = 0u; s < p.splits; ++s) {
        const float2 ml = partML[mlBase + s];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }

    float lSum = 0.0f;
    float4 acc = float4(0.0f);
    for (uint s = 0u; s < p.splits; ++s) {
        const float2 ml = partML[mlBase + s];
        if (ml.y > 0.0f) {
            const float scale = (ml.x == mMax) ? 1.0f : exp(ml.x - mMax);
            lSum += ml.y * scale;
            acc += (*(device const float4*)(partO + (mlBase + s) * p.headDim + dBase)) * scale;
        }
    }

    const size_t oBase = (size_t(t) * p.nH + h) * p.headDim + dBase;
    float4 res = acc / max(lSum, 1e-30f);
    if (p.hasOutGate != 0u) {
        const float4 g4 = float4(*(device const bfloat4*)(gateBF16 + oBase));
        res.x *= qw_ink2_sigmoid(g4.x); res.y *= qw_ink2_sigmoid(g4.y);
        res.z *= qw_ink2_sigmoid(g4.z); res.w *= qw_ink2_sigmoid(g4.w);
    }
    *(device bfloat4*)(outBF16 + oBase) = bfloat4(
        qw_ink2_bf16(res.x), qw_ink2_bf16(res.y), qw_ink2_bf16(res.z), qw_ink2_bf16(res.w)
    );
}
