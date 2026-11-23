// qwlsa_mh.metal — Full-Slice Q4 multi-head direct LSA prep, pooling, and coalesced scoring.
#include "qwlsa_mh_common.metalh"

// Prepares unnormalized full-slice Q and L2-clipped full-slice K (FP16 inputs).
kernel void qwg_lsa_index_prep_mh(device const half* qIn      [[buffer(0)]],
                                  device const half* kIn      [[buffer(1)]],
                                  device half*       qIdxOut  [[buffer(2)]],
                                  device half*       kPending [[buffer(3)]],
                                  constant uint&     T          [[buffer(4)]],
                                  constant uint&     pos0       [[buffer(5)]],
                                  constant uint&     qHeads     [[buffer(6)]],
                                  constant uint&     kvHeads    [[buffer(7)]],
                                  constant uint&     headDim    [[buffer(8)]],
                                  constant uint&     qRowStride [[buffer(9)]],
                                  constant uint&     qStride    [[buffer(10)]],
                                  constant uint&     sliceStart [[buffer(11)]],
                                  constant uint&     sliceDim   [[buffer(12)]],
                                  constant uint&     ringMask   [[buffer(13)]],
                                  constant uint&     qMode      [[buffer(14)]],
                                  constant float&    kL2Clip    [[buffer(15)]],
                                  uint tg  [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tg;
    if (t >= T) return;
    threadgroup float partial[4];
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint sLim = min(sliceDim, 256u);
    const uint d0 = tid, d1 = tid + 128u;
    const uint ringRow = (pos0 + t) & ringMask;

    for (uint h = 0u; h < kvHeads; h++) {
        const uint64_t kInBase = uint64_t(t) * kvHeads * headDim + uint64_t(h) * headDim + sliceStart;
        const float v0 = (d0 < sLim) ? float(kIn[kInBase + d0]) : 0.0f;
        const float v1 = (d1 < sLim) ? float(kIn[kInBase + d1]) : 0.0f;
        float scK = 1.0f;
        if (kL2Clip > 0.0f) {
            const float sgSS = simd_sum(v0 * v0 + v1 * v1);
            if (lane == 0u) partial[sg] = sgSS;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float normSq = partial[0] + partial[1] + partial[2] + partial[3];
            scK = min(kL2Clip / (sqrt(normSq) + 1e-6f), 1.0f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint64_t kOutBase = uint64_t(ringRow) * kvHeads * sliceDim + uint64_t(h) * sliceDim;
        if (d0 < sLim) kPending[kOutBase + d0] = half(v0 * scK);
        if (d1 < sLim) kPending[kOutBase + d1] = half(v1 * scK);
    }

    const bool doQ = (qMode == 1u) || (qMode == 2u && ((t & 31u) == 31u || (t + 1u) == T));
    if (!doQ) return;
    const uint qIndexStride = qHeads * sliceDim;
    for (uint qh = 0u; qh < qHeads; qh++) {
        const uint64_t qInBase = uint64_t(t) * qRowStride + uint64_t(qh) * qStride * headDim + sliceStart;
        const uint64_t qOutBase = uint64_t(t) * qIndexStride + uint64_t(qh) * sliceDim;
        if (d0 < sLim) qIdxOut[qOutBase + d0] = qIn[qInBase + d0];
        if (d1 < sLim) qIdxOut[qOutBase + d1] = qIn[qInBase + d1];
    }
}

// Prepares unnormalized full-slice Q and L2-clipped full-slice K (BF16 inputs).
kernel void qwg_lsa_index_prep_bf16_mh(device const ushort* qIn      [[buffer(0)]],
                                       device const ushort* kIn      [[buffer(1)]],
                                       device half*         qIdxOut  [[buffer(2)]],
                                       device half*         kPending [[buffer(3)]],
                                       constant uint&       T          [[buffer(4)]],
                                       constant uint&       pos0       [[buffer(5)]],
                                       constant uint&       qHeads     [[buffer(6)]],
                                       constant uint&       kvHeads    [[buffer(7)]],
                                       constant uint&       headDim    [[buffer(8)]],
                                       constant uint&       qRowStride [[buffer(9)]],
                                       constant uint&       qStride    [[buffer(10)]],
                                       constant uint&       sliceStart [[buffer(11)]],
                                       constant uint&       sliceDim   [[buffer(12)]],
                                       constant uint&       ringMask   [[buffer(13)]],
                                       constant uint&       qMode      [[buffer(14)]],
                                       constant float&      kL2Clip    [[buffer(15)]],
                                       uint tg  [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tg;
    if (t >= T) return;
    threadgroup float partial[4];
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint sLim = min(sliceDim, 256u);
    const uint d0 = tid, d1 = tid + 128u;
    const uint ringRow = (pos0 + t) & ringMask;

    for (uint h = 0u; h < kvHeads; h++) {
        const uint64_t kInBase = uint64_t(t) * kvHeads * headDim + uint64_t(h) * headDim + sliceStart;
        const float v0 = (d0 < sLim) ? lsa_mh_bf16_to_f32(kIn[kInBase + d0]) : 0.0f;
        const float v1 = (d1 < sLim) ? lsa_mh_bf16_to_f32(kIn[kInBase + d1]) : 0.0f;
        float scK = 1.0f;
        if (kL2Clip > 0.0f) {
            const float sgSS = simd_sum(v0 * v0 + v1 * v1);
            if (lane == 0u) partial[sg] = sgSS;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float normSq = partial[0] + partial[1] + partial[2] + partial[3];
            scK = min(kL2Clip / (sqrt(normSq) + 1e-6f), 1.0f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint64_t kOutBase = uint64_t(ringRow) * kvHeads * sliceDim + uint64_t(h) * sliceDim;
        if (d0 < sLim) kPending[kOutBase + d0] = half(v0 * scK);
        if (d1 < sLim) kPending[kOutBase + d1] = half(v1 * scK);
    }

    const bool doQ = (qMode == 1u) || (qMode == 2u && ((t & 31u) == 31u || (t + 1u) == T));
    if (!doQ) return;
    const uint qIndexStride = qHeads * sliceDim;
    for (uint qh = 0u; qh < qHeads; qh++) {
        const uint64_t qInBase = uint64_t(t) * qRowStride + uint64_t(qh) * qStride * headDim + sliceStart;
        const uint64_t qOutBase = uint64_t(t) * qIndexStride + uint64_t(qh) * sliceDim;
        if (d0 < sLim) qIdxOut[qOutBase + d0] = half(lsa_mh_bf16_to_f32(qIn[qInBase + d0]));
        if (d1 < sLim) qIdxOut[qOutBase + d1] = half(lsa_mh_bf16_to_f32(qIn[qInBase + d1]));
    }
}

// Pools 4 consecutive clipped K slices per KV head and packs into per-(block, kvHead) asymmetric uint4.
kernel void qwg_lsa_pool_keys_mh(device const half* kPending        [[buffer(0)]],
                                 device uchar*      pooledKeys      [[buffer(1)]],
                                 constant uint&     startBlk        [[buffer(2)]],
                                 constant uint&     numBlocks       [[buffer(3)]],
                                 constant uint&     kvHeads         [[buffer(4)]],
                                 constant uint&     sliceDim        [[buffer(5)]],
                                 constant uint&     pooledMetaBytes [[buffer(6)]],
                                 constant uint&     pooledRowBytes  [[buffer(7)]],
                                 constant uint&     ringMask        [[buffer(8)]],
                                 uint tg  [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= numBlocks) return;
    const uint blk = startBlk + tg;
    const uint p0 = blk << 2u;
    const uint r0 = (p0 + 0u) & ringMask, r1 = (p0 + 1u) & ringMask;
    const uint r2 = (p0 + 2u) & ringMask, r3 = (p0 + 3u) & ringMask;
    const uint rowStrideK = kvHeads * sliceDim;
    const uint sPairs = sliceDim >> 1u;

    device uchar* blkPtr = pooledKeys + uint64_t(blk) * pooledRowBytes;
    device half2* metaOut = (device half2*)blkPtr;
    device uchar* payOut = blkPtr + pooledMetaBytes;
    threadgroup float tgMin[4], tgMax[4];
    const uint sg = tid >> 5u, lane = tid & 31u;
    const bool active = tid < sPairs;

    for (uint h = 0u; h < kvHeads; h++) {
        float v0 = 0.0f, v1 = 0.0f;
        if (active) {
            const uint hOff = h * sliceDim + (tid << 1u);
            v0 = 0.25f * (float(kPending[uint64_t(r0) * rowStrideK + hOff])
                        + float(kPending[uint64_t(r1) * rowStrideK + hOff])
                        + float(kPending[uint64_t(r2) * rowStrideK + hOff])
                        + float(kPending[uint64_t(r3) * rowStrideK + hOff]));
            v1 = 0.25f * (float(kPending[uint64_t(r0) * rowStrideK + hOff + 1u])
                        + float(kPending[uint64_t(r1) * rowStrideK + hOff + 1u])
                        + float(kPending[uint64_t(r2) * rowStrideK + hOff + 1u])
                        + float(kPending[uint64_t(r3) * rowStrideK + hOff + 1u]));
        }
        lsa_mh_pack_q4_head(metaOut, payOut, h, sPairs, tid, sg, lane, active, v0, v1, tgMin, tgMax);
    }
}

// Pools 4 consecutive Q4 fine blocks into a 16-token Q4 coarse block per KV head.
kernel void qwg_lsa_pool_coarse_mh(device const uchar* pooledKeys       [[buffer(0)]],
                                   device uchar*       pooledCoarseKeys [[buffer(1)]],
                                   constant uint&      startCoarseBlk   [[buffer(2)]],
                                   constant uint&      numCoarseBlocks  [[buffer(3)]],
                                   constant uint&      kvHeads          [[buffer(4)]],
                                   constant uint&      sliceDim         [[buffer(5)]],
                                   constant uint&      pooledMetaBytes  [[buffer(6)]],
                                   constant uint&      pooledRowBytes   [[buffer(7)]],
                                   uint tg  [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= numCoarseBlocks) return;
    const uint cBlk = startCoarseBlk + tg;
    const uint f0 = cBlk << 2u;
    const uint sPairs = sliceDim >> 1u;
    device uchar* outPtr = pooledCoarseKeys + uint64_t(cBlk) * pooledRowBytes;
    device half2* metaOut = (device half2*)outPtr;
    device uchar* payOut = outPtr + pooledMetaBytes;
    threadgroup float tgMin[4], tgMax[4];
    const uint sg = tid >> 5u, lane = tid & 31u;
    const bool active = tid < sPairs;

    for (uint h = 0u; h < kvHeads; h++) {
        float v0 = 0.0f, v1 = 0.0f;
        for (uint b = 0u; b < 4u; b++) {
            device const uchar* fPtr = pooledKeys + uint64_t(f0 + b) * pooledRowBytes;
            const half2 hm = ((device const half2*)fPtr)[h];
            if (active) {
                const uchar pk = (fPtr + pooledMetaBytes)[h * sPairs + tid];
                const float sc = float(hm.x), mn = float(hm.y);
                v0 += mn + sc * float(pk & 0xFu);
                v1 += mn + sc * float(pk >> 4u);
            }
        }
        lsa_mh_pack_q4_head(metaOut, payOut, h, sPairs, tid, sg, lane, active,
                            v0 * 0.25f, v1 * 0.25f, tgMin, tgMax);
    }
}

template <uint U>
static inline void lsa_mh_score_body(device const half*     qIdx,
                                     device const uchar*    keyPool,
                                     device const uint*     fineCandIdx,
                                     device half*           mhHeadScores,
                                     device float2*         mhTGStats,
                                     constant LSAMHSelParams& p,
                                     threadgroup const LSAMHGeom* gArr,
                                     uint uBase, uint uCnt, uint shFineCnt,
                                     const thread uint* itemLim, uint maxItemLim,
                                     threadgroup float2 (*sgStat)[64],
                                     uint2 tg, uint tid) {
    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint gPerKV = p.qHeads / p.kvHeads;
    const uint sPairs = p.sliceDim >> 1u;
    const uint headVec4 = p.sliceDim >> 2u;
    const float scaleInv = rsqrt(float(p.sliceDim));
    const uint fineStride = p.coarseCap * 4u;

    float myQSum0[U], myMax0[U], mySum0[U];
    float myQSum1[U], myMax1[U], mySum1[U];
    half myScore0[U], myScore1[U];
    #pragma unroll
    for (uint qb = 0u; qb < U; qb++) {
        float qs0 = 0.0f, qs1 = 0.0f;
        if (qb < uCnt && gArr[qb].isSparse != 0u) {
            device const half* qRowBase = qIdx + uint64_t(gArr[qb].qRow) * p.qIndexStride;
            if (lane < p.qHeads) {
                device const half4* qPtr = (device const half4*)(qRowBase + lane * p.sliceDim);
                float4 s4 = float4(0.0f);
                for (uint k = 0u; k < headVec4; k++) s4 += float4(qPtr[k]);
                qs0 = s4.x + s4.y + s4.z + s4.w;
            }
            if ((lane + 32u) < p.qHeads) {
                device const half4* qPtr = (device const half4*)(qRowBase + (lane + 32u) * p.sliceDim);
                float4 s4 = float4(0.0f);
                for (uint k = 0u; k < headVec4; k++) s4 += float4(qPtr[k]);
                qs1 = s4.x + s4.y + s4.z + s4.w;
            }
        }
        myQSum0[qb] = qs0; myQSum1[qb] = qs1;
        myMax0[qb] = -1e30f; mySum0[qb] = 0.0f; myScore0[qb] = half(0.0h);
        myMax1[qb] = -1e30f; mySum1[qb] = 0.0f; myScore1[qb] = half(0.0h);
    }

    for (uint w = 0u; w < 32u; w++) {
        const uint item = (tg.x << 8u) + (w << 3u) + sg;
        if (item < maxItemLim) {
            uint blk = item;
            if (p.stageMode == 2u) {
                if (item < shFineCnt) blk = fineCandIdx[uint64_t(uBase) * fineStride + item];
                else if (item == shFineCnt) blk = 0u;
                else blk = gArr[0].middleEnd + (item - shFineCnt - 1u);
            }
            device const uchar* blkPtr = keyPool + uint64_t(blk) * p.pooledRowBytes;
            device const half2* metaPtr = (device const half2*)blkPtr;
            device const uchar* payPtr = blkPtr + p.pooledMetaBytes;
            for (uint g = 0u; g < p.kvHeads; g++) {
                const half2 hm = metaPtr[g];
                const float sc = float(hm.x), mn = float(hm.y);
                device const ushort* pay2 = (device const ushort*)(payPtr + g * sPairs);
                const ushort pk0 = (lane < headVec4) ? pay2[lane] : 0u;
                const ushort pk1 = ((lane + 32u) < headVec4) ? pay2[lane + 32u] : 0u;
                const float4 n0 = float4(float(pk0 & 0xFu), float((pk0 >> 4u) & 0xFu),
                                         float((pk0 >> 8u) & 0xFu), float((pk0 >> 12u) & 0xFu));
                const float4 n1 = float4(float(pk1 & 0xFu), float((pk1 >> 4u) & 0xFu),
                                         float((pk1 >> 8u) & 0xFu), float((pk1 >> 12u) & 0xFu));
                #pragma unroll
                for (uint qb = 0u; qb < U; qb++) {
                    if (qb >= uCnt || gArr[qb].isSparse == 0u || item >= itemLim[qb]) continue;
                    device const half* qRowBase = qIdx + uint64_t(gArr[qb].qRow) * p.qIndexStride;
                    for (uint sub = 0u; sub < gPerKV; sub++) {
                        const uint qh = g * gPerKV + sub;
                        device const half4* qPtr = (device const half4*)(qRowBase + qh * p.sliceDim);
                        float acc = (lane < headVec4) ? dot(float4(qPtr[lane]), n0) : 0.0f;
                        if ((lane + 32u) < headVec4) acc += dot(float4(qPtr[lane + 32u]), n1);
                        const float dotN = simd_sum(acc);
                        if (lane == qh) {
                            myScore0[qb] = half(clamp((sc * dotN + mn * myQSum0[qb]) * scaleInv, -60000.0f, 60000.0f));
                        } else if ((lane + 32u) == qh) {
                            myScore1[qb] = half(clamp((sc * dotN + mn * myQSum1[qb]) * scaleInv, -60000.0f, 60000.0f));
                        }
                    }
                }
            }
            #pragma unroll
            for (uint qb = 0u; qb < U; qb++) {
                if (qb < uCnt && gArr[qb].isSparse != 0u && item < itemLim[qb]) {
                    if (lane < p.qHeads) {
                        const float sf = float(myScore0[qb]);
                        const float oldM = myMax0[qb];
                        const float newM = max(oldM, sf);
                        mySum0[qb] = (oldM > -1e20f ? mySum0[qb] * exp(oldM - newM) : 0.0f) + exp(sf - newM);
                        myMax0[qb] = newM;
                    }
                    if ((lane + 32u) < p.qHeads) {
                        const float sf = float(myScore1[qb]);
                        const float oldM = myMax1[qb];
                        const float newM = max(oldM, sf);
                        mySum1[qb] = (oldM > -1e20f ? mySum1[qb] * exp(oldM - newM) : 0.0f) + exp(sf - newM);
                        myMax1[qb] = newM;
                    }
                    uint writeIdx = 0u;
                    bool doWrite = false;
                    if (p.stageMode == 0u) {
                        if (item >= 1u && item < gArr[qb].middleEnd) { writeIdx = item - 1u; doWrite = true; }
                    } else if (p.stageMode == 1u) {
                        if (item < gArr[qb].numCoarseMid) { writeIdx = item; doWrite = true; }
                    } else {
                        if (item < shFineCnt) { writeIdx = item; doWrite = true; }
                    }
                    if (doWrite) {
                        const uint64_t rowOff = (uint64_t(uBase + qb) * p.headScoreStride + writeIdx) * p.qHeads;
                        if (lane < p.qHeads) mhHeadScores[rowOff + lane] = myScore0[qb];
                        if ((lane + 32u) < p.qHeads) mhHeadScores[rowOff + lane + 32u] = myScore1[qb];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint qb = 0u; qb < U; qb++) {
        if (qb < uCnt && gArr[qb].isSparse != 0u) {
            if (lane < p.qHeads) sgStat[sg][lane] = float2(myMax0[qb], mySum0[qb]);
            if ((lane + 32u) < p.qHeads) sgStat[sg][lane + 32u] = float2(myMax1[qb], mySum1[qb]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qb < uCnt && gArr[qb].isSparse != 0u && tid < p.qHeads) {
            float tileM = -1e30f;
            for (uint s = 0u; s < 8u; s++) {
                const float2 st = sgStat[s][tid];
                if (st.y > 0.0f) tileM = max(tileM, st.x);
            }
            float tileS = 0.0f;
            if (tileM > -1e20f) {
                for (uint s = 0u; s < 8u; s++) {
                    const float2 st = sgStat[s][tid];
                    if (st.y > 0.0f) tileS += st.y * exp(st.x - tileM);
                }
            }
            mhTGStats[(uint64_t(uBase + qb) * p.qHeads + tid) * p.scoreTilesX + tg.x] = float2(tileM, tileS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Scores Q4 blocks across heads using register-resident LSE and coalesced [unit, item, qHead] writes.
kernel void qwg_lsa_mh_score_heads(device const half*     qIdx          [[buffer(0)]],
                                   device const uchar*    keyPool       [[buffer(1)]],
                                   device const uint*     fineCandIdx   [[buffer(2)]],
                                   device const uint*     fineCandCount [[buffer(3)]],
                                   device half*           mhHeadScores  [[buffer(4)]],
                                   device float2*         mhTGStats     [[buffer(5)]],
                                   constant LSAMHSelParams& p           [[buffer(6)]],
                                   uint2 tg  [[threadgroup_position_in_grid]],
                                   uint  tid [[thread_index_in_threadgroup]]) {
    const uint uBase = tg.y * p.bqTile;
    if (uBase >= p.T) return;
    const uint uCnt = min(p.bqTile, p.T - uBase);

    threadgroup float2 sgStat[8][64];
    threadgroup LSAMHGeom gArr[4];
    threadgroup uint shFineCnt;

    if (tid < uCnt) gArr[tid] = lsa_mh_geom(p, uBase + tid);
    if (p.stageMode == 2u && tid == 0u) shFineCnt = fineCandCount[uBase];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint itemLim[4] = {0u, 0u, 0u, 0u};
    uint maxItemLim = 0u;
    const uint fCnt = (p.stageMode == 2u) ? shFineCnt : 0u;
    for (uint qb = 0u; qb < uCnt; qb++) {
        if (gArr[qb].isSparse == 0u) continue;
        itemLim[qb] = (p.stageMode == 0u) ? gArr[qb].totalBlocks
                    : ((p.stageMode == 1u) ? gArr[qb].totalCoarse : (fCnt + 1u + p.windowBlocks));
        maxItemLim = max(maxItemLim, itemLim[qb]);
    }
    if (tg.x * 256u >= maxItemLim) return;

    if (uCnt == 1u) {
        lsa_mh_score_body<1>(qIdx, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                             uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg, tid);
    } else {
        lsa_mh_score_body<4>(qIdx, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                             uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg, tid);
    }
}

// Reduces per-tile (max, expSum) statistics into exact global LSE including optional sink logit.
kernel void qwg_lsa_mh_reduce_lse(device const float2*   mhTGStats     [[buffer(0)]],
                                  device const ushort*   sinkBF16      [[buffer(1)]],
                                  device const uint*     fineCandCount [[buffer(2)]],
                                  device float*          mhHeadLSE     [[buffer(3)]],
                                  constant LSAMHSelParams& p           [[buffer(4)]],
                                  uint2 tg  [[threadgroup_position_in_grid]],
                                  uint  tid [[thread_index_in_threadgroup]]) {
    const uint qh = tg.x, unit = tg.y;
    if (qh >= p.qHeads || unit >= p.T) return;
    const LSAMHGeom geom = lsa_mh_geom(p, unit);
    if (geom.isSparse == 0u) return;

    const uint lseItems = (p.stageMode == 0u) ? geom.totalBlocks
                        : ((p.stageMode == 1u) ? geom.totalCoarse
                                               : (fineCandCount[unit] + 1u + p.windowBlocks));
    const uint numTiles = (lseItems + 255u) >> 8u;
    device const float2* stats = mhTGStats + (uint64_t(unit) * p.qHeads + qh) * p.scoreTilesX;
    threadgroup float sgRed[8];
    const uint sg = tid >> 5u, lane = tid & 31u;

    float localM = -1e30f;
    for (uint t = tid; t < numTiles; t += 256u) {
        const float2 st = stats[t];
        if (st.y > 0.0f) localM = max(localM, st.x);
    }
    const float sgM = simd_max(localM);
    if (lane == 0u) sgRed[sg] = sgM;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mGlobal = -1e30f;
    for (uint s = 0u; s < 8u; s++) mGlobal = max(mGlobal, sgRed[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localL = 0.0f;
    if (mGlobal > -1e20f) {
        for (uint t = tid; t < numTiles; t += 256u) {
            const float2 st = stats[t];
            if (st.y > 0.0f) localL += st.y * exp(st.x - mGlobal);
        }
    }
    const float sgL = simd_sum(localL);
    if (lane == 0u) sgRed[sg] = sgL;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float lGlobal = 0.0f;
        for (uint s = 0u; s < 8u; s++) lGlobal += sgRed[s];
        if (p.hasSink != 0u) {
            const float sLogit = lsa_mh_bf16_to_f32(sinkBF16[uint64_t(geom.qRow) * p.sinkRowStride + qh]);
            const float newM = max(mGlobal, sLogit);
            lGlobal = (mGlobal > -1e20f ? lGlobal * exp(mGlobal - newM) : 0.0f) + exp(sLogit - newM);
            mGlobal = newM;
        }
        mhHeadLSE[uint64_t(unit) * p.qHeads + qh] = mGlobal + log(max(lGlobal, 1e-30f));
    }
}
