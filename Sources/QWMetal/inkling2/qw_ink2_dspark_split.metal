// qw_ink2_dspark_split.metal — Split-KV bidirectional GQA attention for DSpark drafting.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

struct QWInk2DSparkAttnSplitParams {
    uint32_t B;
    uint32_t nH;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t ctxStart;
    uint32_t ctxEnd;
    uint32_t sliding;
    uint32_t ringCap;
    uint32_t qStride;
    uint32_t kvPropStride;
    uint32_t qRowOffset;
    uint32_t activeQRows;
    uint32_t splits;
    float scale;
};

struct QWInk2DSparkAttnCombineParams {
    uint32_t nH;
    uint32_t headDim;
    uint32_t qRowOffset;
    uint32_t activeQRows;
    uint32_t splits;
};

/// Split-KV bidirectional GQA for DSpark (`tg.y = bGrp * splits + sp`; `splits == 1` writes directOut).
kernel void qwg_ink2_dspark_attn_gqa_split(device const bfloat* qProp [[buffer(0)]],
                                           device const bfloat* kProp [[buffer(1)]],
                                           device const bfloat* vProp [[buffer(2)]],
                                           device const bfloat* kCache [[buffer(3)]],
                                           device const bfloat* vCache [[buffer(4)]],
                                           device float* partO [[buffer(5)]],
                                           device float2* partML [[buffer(6)]],
                                           device bfloat* directOut [[buffer(7)]],
                                           constant QWInk2DSparkAttnSplitParams& p [[buffer(8)]],
                                           uint3 tg [[threadgroup_position_in_grid]],
                                           uint simd_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint sp = (p.splits > 0u) ? (tg.y % p.splits) : 0u;
    const uint bStart = ((p.splits > 0u) ? (tg.y / p.splits) : tg.y) * 4u;
    const uint gqa = (p.nKV > 0u) ? (p.nH / p.nKV) : 0u;
    const uint gStart = tg.z * 2u;
    const uint chunks = (p.headDim + 31u) / 32u;
    if (kvh >= p.nKV || sp >= p.splits || bStart >= p.activeQRows || gStart >= gqa || chunks > 4u) return;

    const uint bCnt = min(4u, p.activeQRows - bStart);
    const uint gCnt = min(2u, gqa - gStart);
    const uint hBase = kvh * gqa + gStart;
    const uint qDim = p.nH * p.headDim;
    const uint kvDim = p.nKV * p.headDim;
    const uint qStride = (p.qStride > 0u) ? p.qStride : qDim;
    const uint kvPStride = (p.kvPropStride > 0u) ? p.kvPropStride : kvDim;

    float qVec[4][2][4];
    #pragma unroll
    for (uint bb = 0u; bb < 4u; ++bb) {
        if (bb >= bCnt) break;
        const size_t qTok = size_t(p.qRowOffset + bStart + bb) * qStride;
        #pragma unroll
        for (uint g = 0u; g < 2u; ++g) {
            if (g >= gCnt) break;
            device const bfloat* qRow = qProp + qTok + size_t(hBase + g) * p.headDim;
            for (uint c = 0u; c < chunks; ++c) {
                const uint d = lane + c * 32u;
                qVec[bb][g][c] = (d < p.headDim) ? float(qRow[d]) : 0.0f;
            }
        }
    }

    threadgroup float vSh[8][132];
    threadgroup float scoreSh[8][8];

    const uint qB = simd_id >> 1u;
    const uint qG = simd_id & 1u;
    const bool qActive = (qB < bCnt && qG < gCnt);
    float mReg = -INFINITY;
    float lReg = 0.0f;
    float oReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint ctxCount = (p.ctxEnd > p.ctxStart) ? (p.ctxEnd - p.ctxStart) : 0u;
    const uint chunk = (ctxCount + p.splits - 1u) / p.splits;
    const uint cStart = min(ctxCount, sp * chunk);
    const uint cEnd = min(ctxCount, cStart + chunk);
    const uint splitCtxKeys = cEnd - cStart;
    const uint totalKeys = splitCtxKeys + ((sp == 0u) ? p.B : 0u);

    for (uint tileBase = 0u; tileBase < totalKeys; tileBase += 8u) {
        const uint tileCnt = min(8u, totalKeys - tileBase);
        if (simd_id < tileCnt) {
            const uint kPos = tileBase + simd_id;
            device const bfloat* kRow;
            device const bfloat* vRow;
            if (kPos < splitCtxKeys) {
                const uint j = p.ctxStart + cStart + kPos;
                const uint slot = (p.sliding != 0u) ? (j % p.ringCap) : j;
                const size_t kvOff = size_t(slot) * kvDim + size_t(kvh) * p.headDim;
                kRow = kCache + kvOff;
                vRow = vCache + kvOff;
            } else {
                const uint bp = kPos - splitCtxKeys;
                const size_t pOff = size_t(bp) * kvPStride + size_t(kvh) * p.headDim;
                kRow = kProp + pOff;
                vRow = vProp + pOff;
            }
            float kVec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint c = 0u; c < chunks; ++c) {
                const uint d = lane + c * 32u;
                float kv = 0.0f;
                float vv = 0.0f;
                if (d < p.headDim) {
                    kv = float(kRow[d]);
                    vv = float(vRow[d]);
                }
                kVec[c] = kv;
                vSh[simd_id][lane + c * 32u] = vv;
            }
            #pragma unroll
            for (uint qb = 0u; qb < 4u; ++qb) {
                if (qb >= bCnt) break;
                #pragma unroll
                for (uint qg = 0u; qg < 2u; ++qg) {
                    if (qg >= gCnt) break;
                    float dotAcc = 0.0f;
                    for (uint c = 0u; c < chunks; ++c) dotAcc += qVec[qb][qg][c] * kVec[c];
                    const float s = simd_sum(dotAcc) * p.scale;
                    if (lane == 0u) scoreSh[simd_id][qb * 2u + qg] = s;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qActive) {
            for (uint s = 0u; s < tileCnt; ++s) {
                const float sVal = scoreSh[s][simd_id];
                const float mNew = max(mReg, sVal);
                const float expPrev = (lReg > 0.0f) ? metal::exp(mReg - mNew) : 0.0f;
                const float w = metal::exp(sVal - mNew);
                lReg = lReg * expPrev + w;
                for (uint c = 0u; c < chunks; ++c) oReg[c] = oReg[c] * expPrev + w * vSh[s][lane + c * 32u];
                mReg = mNew;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!qActive) return;
    const uint bLocal = bStart + qB;
    const uint h = hBase + qG;
    if (p.splits == 1u) {
        const size_t oTok = size_t(p.qRowOffset + bLocal) * qDim;
        device bfloat* oRow = directOut + oTok + size_t(h) * p.headDim;
        const float invL = (lReg > 0.0f) ? (1.0f / lReg) : 0.0f;
        for (uint c = 0u; c < chunks; ++c) {
            const uint d = lane + c * 32u;
            if (d < p.headDim) oRow[d] = qw_ink2_bf16(oReg[c] * invL);
        }
        return;
    }
    const size_t slotIdx = (size_t(bLocal) * p.nH + h) * p.splits + sp;
    if (lane == 0u) partML[slotIdx] = float2(mReg, lReg);
    device float* pOut = partO + slotIdx * p.headDim;
    for (uint c = 0u; c < chunks; ++c) {
        const uint d = lane + c * 32u;
        if (d < p.headDim) pOut[d] = oReg[c];
    }
}

/// Reduces split-KV attention partials across `p.splits` and writes BF16 output rows.
kernel void qwg_ink2_dspark_attn_gqa_combine(device const float* partO [[buffer(0)]],
                                             device const float2* partML [[buffer(1)]],
                                             device bfloat* outBF16 [[buffer(2)]],
                                             constant QWInk2DSparkAttnCombineParams& p [[buffer(3)]],
                                             uint2 tg [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x;
    const uint bLocal = tg.y;
    const uint chunks = (p.headDim + 31u) / 32u;
    if (h >= p.nH || bLocal >= p.activeQRows || chunks > 4u) return;

    const size_t baseML = (size_t(bLocal) * p.nH + h) * p.splits;
    float mMax = -INFINITY;
    for (uint sp = 0u; sp < p.splits; ++sp) {
        const float2 ml = partML[baseML + sp];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }

    float lSum = 0.0f;
    float oAcc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint sp = 0u; sp < p.splits; ++sp) {
        const float2 ml = partML[baseML + sp];
        if (ml.y <= 0.0f) continue;
        const float scaleSp = metal::exp(ml.x - mMax);
        lSum += ml.y * scaleSp;
        device const float* pRow = partO + (baseML + sp) * p.headDim;
        for (uint c = 0u; c < chunks; ++c) {
            const uint d = lane + c * 32u;
            if (d < p.headDim) oAcc[c] += pRow[d] * scaleSp;
        }
    }

    const size_t qDim = size_t(p.nH) * p.headDim;
    const size_t outBase = size_t(p.qRowOffset + bLocal) * qDim + size_t(h) * p.headDim;
    const float invL = (lSum > 0.0f) ? (1.0f / lSum) : 0.0f;
    for (uint c = 0u; c < chunks; ++c) {
        const uint d = lane + c * 32u;
        if (d < p.headDim) outBF16[outBase + d] = qw_ink2_bf16(oAcc[c] * invL);
    }
}
