// qwlsa_mh_select.metal — Coalesced MH LSA softmax aggregation, 2-pass radix collect, and finalize kernels.
#include "qwlsa_mh_common.metalh"

template <typename OutT>
static inline void lsa_mh_fast_merge(threadgroup uint* keep,
                                     threadgroup uint& keepCount,
                                     device const uint* cm,
                                     device const uint* cu,
                                     uint rawCount,
                                     uint total,
                                     uint greater,
                                     uint needTies,
                                     uint threshU,
                                     device OutT* outBuf,
                                     uint outOffset,
                                     uint tid) {
    constexpr uint TIE_FLAG = 0x80000000u;
    uint n = 256u;
    while (n < total) n <<= 1u;
    if (tid == 0u) keepCount = 0u;
    for (uint i = tid; i < n; i += 256u) keep[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rawCount; i += 256u) {
        const uint u = cu[i];
        if (u >= threshU) {
            const uint slot = atomic_fetch_add_explicit((threadgroup atomic_uint*)&keepCount, 1u,
                                                        memory_order_relaxed);
            keep[slot] = cm[i] | (u == threshU ? TIE_FLAG : 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= n; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = tid; i < n; i += 256u) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const uint a = keep[i], c = keep[ixj];
                    if ((a > c) == ((i & k) == 0u)) { keep[i] = c; keep[ixj] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const uint bBase = total - needTies;
    for (uint i = tid; i < greater; i += 256u) {
        const uint v = keep[i];
        uint lo = 0u, hi = needTies;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1u;
            if ((keep[bBase + mid] & ~TIE_FLAG) < v) lo = mid + 1u; else hi = mid;
        }
        outBuf[outOffset + i + lo] = OutT(v);
    }
    for (uint j = tid; j < needTies; j += 256u) {
        const uint w = keep[bBase + j] & ~TIE_FLAG;
        uint lo = 0u, hi = greater;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1u;
            if (keep[mid] < w) lo = mid + 1u; else hi = mid;
        }
        outBuf[outOffset + j + lo] = OutT(w);
    }
}

// Aggregates softmax mass across heads via coalesced [unit, item, qHead] reads and fills MSB histogram.
kernel void qwg_lsa_mh_agg_hist(device const half*     mhHeadScores  [[buffer(0)]],
                                device const float*    mhHeadLSE     [[buffer(1)]],
                                device const uint*     fineCandCount [[buffer(2)]],
                                device uint*           scoreScratch  [[buffer(3)]],
                                device atomic_uint*    uHist         [[buffer(4)]],
                                constant LSAMHSelParams& p           [[buffer(5)]],
                                uint2 tg  [[threadgroup_position_in_grid]],
                                uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const LSAMHGeom geom = lsa_mh_geom(p, unit);
    if (geom.isSparse == 0u) return;

    const uint nItems = (p.stageMode == 0u) ? geom.numMiddle
                      : ((p.stageMode == 1u) ? geom.numCoarseMid : fineCandCount[unit]);
    if (tg.x * 256u >= nItems) return;

    threadgroup uint localHist[256];
    localHist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint64_t lseBase = uint64_t(unit) * p.qHeads;
    const float myLSE0 = (lane < p.qHeads) ? mhHeadLSE[lseBase + lane] : 0.0f;
    const float myLSE1 = ((lane + 32u) < p.qHeads) ? mhHeadLSE[lseBase + lane + 32u] : 0.0f;
    const uint64_t scoreUnitBase = uint64_t(unit) * p.headScoreStride;

    for (uint w = 0u; w < 32u; w++) {
        const uint idx = (tg.x << 8u) + (w << 3u) + sg;
        if (idx < nItems) {
            const uint64_t rowOff = (scoreUnitBase + idx) * p.qHeads;
            float pVal = (lane < p.qHeads) ? exp(float(mhHeadScores[rowOff + lane]) - myLSE0) : 0.0f;
            if ((lane + 32u) < p.qHeads) pVal += exp(float(mhHeadScores[rowOff + lane + 32u]) - myLSE1);
            const float probSum = simd_sum(pVal);
            if (lane == 0u) {
                const float probClamped = clamp(probSum, 0.0f, float(p.qHeads));
                const uint bits = as_type<uint>(probClamped);
                const uint u16 = (bits <= 0x33800000u) ? 0u : min(65535u, (bits - 0x33800000u) >> 12u);
                scoreScratch[uint64_t(unit) * p.scoreStride + idx] = u16;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHist[(u16 >> 8u) & 0xFFu],
                                          1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint hc = localHist[tid];
    if (hc != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], hc, memory_order_relaxed);
}

kernel void qwg_lsa_mh_collect(device const uint*     scoreScratch  [[buffer(0)]],
                               device const uint*     fineCandIdx   [[buffer(1)]],
                               device const uint*     fineCandCount [[buffer(2)]],
                               device atomic_uint*    uHist         [[buffer(3)]],
                               device atomic_uint*    candCount     [[buffer(4)]],
                               device uint*           candM         [[buffer(5)]],
                               device uint*           candU         [[buffer(6)]],
                               constant LSAMHSelParams& p           [[buffer(7)]],
                               uint2 tg  [[threadgroup_position_in_grid]],
                               uint  tid [[thread_index_in_threadgroup]]) {
    const uint unit = tg.y;
    if (unit >= p.T) return;
    const LSAMHGeom g = lsa_mh_geom(p, unit);
    if (g.isSparse == 0u) return;

    const uint nItems = (p.stageMode == 0u) ? g.numMiddle
                      : ((p.stageMode == 1u) ? g.numCoarseMid : fineCandCount[unit]);
    if (tg.x * 256u >= nItems) return;

    const uint kTarget = (p.stageMode == 1u) ? g.coarseKEff : g.kEff;
    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    const uint sg = tid >> 5u, lane = tid & 31u;
    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
    if (tid == 0u) { greaterAccum = 0u; chosenBin = 0u; sgTotal = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_mh_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kTarget, false, tid, sg, lane);
    const uint topBin = chosenBin;

    hist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint idx = tg.x * 256u + tid;
    if (idx < nItems) {
        const uint u16 = scoreScratch[uint64_t(unit) * p.scoreStride + idx];
        const uint bin = (u16 >> 8u) & 0xFFu;
        if (bin >= topBin) {
            const uint mId = (p.stageMode == 0u) ? (idx + 1u)
                           : ((p.stageMode == 1u) ? idx
                                                  : fineCandIdx[uint64_t(unit) * (p.coarseCap * 4u) + idx]);
            const uint slot = atomic_fetch_add_explicit(&candCount[unit], 1u, memory_order_relaxed);
            if (slot < p.candCap) {
                candM[uint64_t(unit) * p.candCap + slot] = mId;
                candU[uint64_t(unit) * p.candCap + slot] = u16;
            }
            if (bin == topBin) {
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[u16 & 0xFFu], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (hist[tid] != 0u) {
        atomic_fetch_add_explicit(&uHist[unit * 512u + 256u + tid], hist[tid], memory_order_relaxed);
    }
}

// Finalizes Stage 1 top-coarse blocks and expands them into sorted fine candidate block IDs.
kernel void qwg_lsa_mh_coarse_expand(device const uint*     scoreScratch  [[buffer(0)]],
                                     device uint*           fineCandIdx   [[buffer(1)]],
                                     device uint*           fineCandCount [[buffer(2)]],
                                     device atomic_uint*    uHist         [[buffer(3)]],
                                     device atomic_uint*    candCount     [[buffer(4)]],
                                     device uint*           candM         [[buffer(5)]],
                                     device const uint*     candU         [[buffer(6)]],
                                     constant LSAMHSelParams& p           [[buffer(7)]],
                                     uint unit [[threadgroup_position_in_grid]],
                                     uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    constexpr uint SORT_N = 4096u;
    const LSAMHGeom g = lsa_mh_geom(p, unit);
    if (g.isSparse == 0u) {
        if (tid == 0u) fineCandCount[unit] = 0u;
        atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
        if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
        return;
    }

    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    threadgroup uint keep[SORT_N];
    threadgroup uint keepCount;
    const uint sg = tid >> 5u, lane = tid & 31u;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
    if (tid == 0u) { greaterAccum = 0u; chosenBin = 0u; sgTotal = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_mh_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.coarseKEff, false, tid, sg, lane);
    const uint topBin = chosenBin;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + 256u + tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_mh_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.coarseKEff, true, tid, sg, lane);
    const uint threshU = (topBin << 8u) | chosenBin;
    const uint rawCount = atomic_load_explicit(&candCount[unit], memory_order_relaxed);
    const uint fineStride = p.coarseCap * 4u;
    device uint* fineOut = fineCandIdx + uint64_t(unit) * fineStride;
    const bool fast = (rawCount <= p.candCap) && (sgTotal <= SORT_N);

    if (fast) {
        const uint total = sgTotal;
        const uint greater = greaterAccum;
        const uint needTies = g.coarseKEff - greater;
        device uint* cm = candM + uint64_t(unit) * p.candCap;
        device const uint* cu = candU + uint64_t(unit) * p.candCap;
        lsa_mh_fast_merge(keep, keepCount, cm, cu, rawCount, total, greater, needTies, threshU, cm, 0u, tid);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if (tid == 0u) chosenBin = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint cChunks = (g.coarseKEff + 255u) >> 8u;
        for (uint ch = 0u; ch < cChunks; ch++) {
            const uint i = (ch << 8u) + tid;
            uint fb0 = 0u, cnt = 0u;
            if (i < g.coarseKEff) {
                const uint c = cm[i];
                fb0 = max(1u, c << 2u);
                const uint fb1 = min(g.middleEnd, (c << 2u) + 4u);
                cnt = (fb1 > fb0) ? (fb1 - fb0) : 0u;
            }
            const uint wRank = simd_prefix_exclusive_sum(cnt);
            const uint wSum = simd_sum(cnt);
            if (lane == 0u) scanSum[sg] = wSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint wBase = chosenBin;
            for (uint s = 0u; s < sg; s++) wBase += scanSum[s];
            for (uint k = 0u; k < cnt; k++) {
                const uint dst = wBase + wRank + k;
                if (dst < fineStride) fineOut[dst] = fb0 + k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkFine = 0u;
                for (uint s = 0u; s < 8u; s++) chunkFine += scanSum[s];
                chosenBin += chunkFine;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) fineCandCount[unit] = min(chosenBin, fineStride);
    } else {
        const uint skipTies = sgTotal - g.coarseKEff;
        if (tid == 0u) { chosenBin = 0u; greaterAccum = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint numChunks = (g.numCoarseMid + 255u) >> 8u;
        for (uint ch = 0u; ch < numChunks; ch++) {
            const uint idx = (ch << 8u) + tid;
            const bool valid = idx < g.numCoarseMid;
            const uint u = valid ? scoreScratch[uint64_t(unit) * p.scoreStride + idx] : 0u;
            const bool isGt = valid && (u > threshU);
            const bool isTie = valid && (u == threshU);
            const uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
            const uint tieSum = simd_sum(isTie ? 1u : 0u);
            if (lane == 0u) scanSum[sg] = tieSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieBase = greaterAccum;
            for (uint s = 0u; s < sg; s++) tieBase += scanSum[s];
            const bool keepCoarse = isGt || (isTie && ((tieBase + tieRank) >= skipTies));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkTies = 0u;
                for (uint s = 0u; s < 8u; s++) chunkTies += scanSum[s];
                greaterAccum += chunkTies;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint fb0 = 0u, cnt = 0u;
            if (keepCoarse) {
                fb0 = max(1u, idx << 2u);
                const uint fb1 = min(g.middleEnd, (idx << 2u) + 4u);
                cnt = (fb1 > fb0) ? (fb1 - fb0) : 0u;
            }
            const uint wRank = simd_prefix_exclusive_sum(cnt);
            const uint wSum = simd_sum(cnt);
            if (lane == 0u) scanSum[sg] = wSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint wBase = chosenBin;
            for (uint s = 0u; s < sg; s++) wBase += scanSum[s];
            for (uint k = 0u; k < cnt; k++) {
                const uint dst = wBase + wRank + k;
                if (dst < fineStride) fineOut[dst] = fb0 + k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkFine = 0u;
                for (uint s = 0u; s < 8u; s++) chunkFine += scanSum[s];
                chosenBin += chunkFine;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) fineCandCount[unit] = min(chosenBin, fineStride);
    }

    atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
}

kernel void qwg_lsa_mh_finalize(device const uint*     scoreScratch  [[buffer(0)]],
                                device const uint*     fineCandIdx   [[buffer(1)]],
                                device const uint*     fineCandCount [[buffer(2)]],
                                device int*            blockList     [[buffer(3)]],
                                device uint*           blockCount    [[buffer(4)]],
                                device atomic_uint*    uHist         [[buffer(5)]],
                                device atomic_uint*    candCount     [[buffer(6)]],
                                device const uint*     candM         [[buffer(7)]],
                                device const uint*     candU         [[buffer(8)]],
                                constant LSAMHSelParams& p           [[buffer(9)]],
                                uint unit [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]]) {
    if (unit >= p.T) return;
    constexpr uint SORT_N = 4096u;
    const LSAMHGeom g = lsa_mh_geom(p, unit);
    device int* outRow = blockList + uint64_t(g.qRow) * p.listStride;

    if (g.isSparse == 0u) {
        for (uint b = tid; b < g.totalBlocks; b += 256u) outRow[b] = int(b);
        if (tid == 0u) blockCount[g.qRow] = g.totalBlocks;
        atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
        if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
        return;
    }

    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    threadgroup uint keep[SORT_N];
    threadgroup uint keepCount;
    const uint sg = tid >> 5u, lane = tid & 31u;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
    if (tid == 0u) { greaterAccum = 0u; chosenBin = 0u; sgTotal = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_mh_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.kEff, false, tid, sg, lane);
    const uint topBin = chosenBin;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + 256u + tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lsa_mh_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.kEff, true, tid, sg, lane);
    const uint threshU = (topBin << 8u) | chosenBin;
    const uint rawCount = atomic_load_explicit(&candCount[unit], memory_order_relaxed);
    const bool fast = (rawCount <= p.candCap) && (sgTotal <= SORT_N);

    if (fast) {
        const uint total = sgTotal;
        const uint greater = greaterAccum;
        const uint needTies = g.kEff - greater;
        device const uint* cm = candM + uint64_t(unit) * p.candCap;
        device const uint* cu = candU + uint64_t(unit) * p.candCap;
        lsa_mh_fast_merge(keep, keepCount, cm, cu, rawCount, total, greater, needTies, threshU, outRow, 1u, tid);
        if (tid == 0u) outRow[0] = 0;
        const uint windowOutBase = 1u + g.kEff;
        for (uint w = tid; w < p.windowBlocks; w += 256u) outRow[windowOutBase + w] = int(g.middleEnd + w);
        if (tid == 0u) blockCount[g.qRow] = windowOutBase + p.windowBlocks;
    } else {
        const uint nItems = (p.stageMode == 2u) ? fineCandCount[unit] : g.numMiddle;
        const uint fineStride = p.coarseCap * 4u;
        const uint skipTies = sgTotal - g.kEff;
        if (tid == 0u) { outRow[0] = 0; chosenBin = 1u; greaterAccum = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint numChunks = (nItems + 255u) >> 8u;
        for (uint c = 0u; c < numChunks; c++) {
            const uint idx = (c << 8u) + tid;
            const bool valid = idx < nItems;
            const uint u = valid ? scoreScratch[uint64_t(unit) * p.scoreStride + idx] : 0u;
            const bool isGt = valid && (u > threshU);
            const bool isTie = valid && (u == threshU);
            const uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
            const uint tieSum = simd_sum(isTie ? 1u : 0u);
            if (lane == 0u) scanSum[sg] = tieSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieBase = greaterAccum;
            for (uint s = 0u; s < sg; s++) tieBase += scanSum[s];
            const bool keepTie = isTie && ((tieBase + tieRank) >= skipTies);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkTies = 0u;
                for (uint s = 0u; s < 8u; s++) chunkTies += scanSum[s];
                greaterAccum += chunkTies;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint keepTok = (isGt || keepTie) ? 1u : 0u;
            const uint wRank = simd_prefix_exclusive_sum(keepTok);
            const uint wSum = simd_sum(keepTok);
            if (lane == 0u) scanSum[sg] = wSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint wBase = chosenBin;
            for (uint s = 0u; s < sg; s++) wBase += scanSum[s];
            if (keepTok != 0u && (wBase + wRank) <= g.kEff) {
                const uint mId = (p.stageMode == 2u)
                    ? fineCandIdx[uint64_t(unit) * fineStride + idx]
                    : (idx + 1u);
                outRow[wBase + wRank] = int(mId);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkKept = 0u;
                for (uint s = 0u; s < 8u; s++) chunkKept += scanSum[s];
                chosenBin += chunkKept;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint windowOutBase = min(chosenBin, 1u + g.kEff);
        for (uint w = tid; w < p.windowBlocks; w += 256u) outRow[windowOutBase + w] = int(g.middleEnd + w);
        if (tid == 0u) blockCount[g.qRow] = windowOutBase + p.windowBlocks;
    }

    atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
}
