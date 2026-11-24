// qwgemm_moe_group_f32.metal — expert-major pair counting sort and tile builder.
// buffers (group pairs): 0 ids | 1 expertOff | 2 sortedPairIdx | 3 sortedTok | 4 experts | 5 totalPairs | 6 topK
#include <metal_stdlib>
#include <metal_simdgroup>
#include "../include/qw_moe_tile.metalh"
using namespace metal;

/// Expert-major pair order (MLX's _gather_sort): counting sort of pairs by expert id. expertOff[e]..expertOff[e+1]
/// is expert e's sorted range. Slot order within an expert is arbitrary: each pair's dot product is independent.
kernel void qwg_moe_group_pairs_f32(device const int* selected [[buffer(0)]],
                                     device int*       expertOff [[buffer(1)]],
                                     device int*       sortedPairIdx [[buffer(2)]],
                                     device int*       sortedTok [[buffer(3)]],
                                     constant uint&    experts [[buffer(4)]],
                                     constant uint&    totalPairs [[buffer(5)]],
                                     constant uint&    topK [[buffer(6)]],
                                     device int*       invPerm [[buffer(7)]],
                                     constant uint&    writeInv [[buffer(8)]],
                                     device int*       tileE [[buffer(9)]],
                                     device int*       tileR [[buffer(10)]],
                                     device int*       tileCount [[buffer(11)]],
                                     constant uint&    writeTiles [[buffer(12)]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int tgCount[1024];
    threadgroup int tgOff[1025];
    for (int e = int(tid); e < int(experts); e += 256)
        atomic_store_explicit(&tgCount[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < int(totalPairs); i += 256) {
        const int rawS = selected[i];
        if (rawS >= 0) {
            const uint e = uint(rawS) & 0xFFFFu;
            if (e < experts) atomic_fetch_add_explicit(&tgCount[e], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        const uint lane = tid;
        const uint chunk = (experts + 31u) / 32u;
        const uint e0 = min(lane * chunk, experts);
        const uint e1 = min(e0 + chunk, experts);
        int mySum = 0;
        for (uint e = e0; e < e1; ++e) {
            mySum += atomic_load_explicit(&tgCount[e], memory_order_relaxed);
        }
        int cursor = simd_prefix_exclusive_sum(mySum);
        for (uint e = e0; e < e1; ++e) {
            const int c = atomic_load_explicit(&tgCount[e], memory_order_relaxed);
            expertOff[e] = cursor;
            tgOff[e] = cursor;
            atomic_store_explicit(&tgCount[e], cursor, memory_order_relaxed);
            cursor += c;
        }
        if (lane == 31u) {
            expertOff[experts] = cursor;
            tgOff[experts] = cursor;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool directSlotTok = (writeInv & 2u) != 0u;
    for (int i = int(tid); i < int(totalPairs); i += 256) {
        const int rawS = selected[i];
        if (rawS < 0) continue;
        const uint raw = uint(rawS);
        const uint e = raw & 0xFFFFu;
        if (e >= experts) continue;
        const int at = atomic_fetch_add_explicit(&tgCount[e], 1, memory_order_relaxed);
        sortedPairIdx[at] = i;
        if (directSlotTok) {
            sortedTok[at] = i;
        } else {
            const uint gIdx = (raw >> 16u) & 0x7Fu;
            const uint gStride = (raw >> 23u) & 0x7Fu;
            const int t = i / int(topK);
            sortedTok[at] = (gStride > 0u) ? (t * int(gStride) + int(gIdx)) : t;
        }
        if ((writeInv & 1u) != 0u) invPerm[i] = at;
    }
    if ((writeTiles & 1u) != 0u && tid < 32u) {
        const uint lane = tid;
        const uint chunk = (experts + 31u) / 32u;
        const uint e0 = min(lane * chunk, experts);
        const uint e1 = min(e0 + chunk, experts);
        const bool compactTail = (writeInv & 4u) == 0u;
        int myTiles = 0;
        for (uint e = e0; e < e1; ++e) {
            const int c = tgOff[e + 1] - tgOff[e];
            if (c > 0) myTiles += (c + 63) / 64;
        }
        int t = simd_prefix_exclusive_sum(myTiles);
        for (uint e = e0; e < e1; ++e) {
            const int c = tgOff[e + 1] - tgOff[e];
            if (c > 0) {
                if (compactTail) {
                    const int q = c >> 6;
                    const int rem = c & 63;
                    for (int r = 0; r < q; ++r) {
                        tileE[t] = int(e);
                        tileR[t] = qw_moe_pack_tile_r(r, 64);
                        ++t;
                    }
                    if (rem > 0) {
                        const int tailBm = qw_moe_tile_block(rem, 0);
                        tileE[t] = int(e);
                        tileR[t] = qw_moe_pack_tile_r((q * 64) / tailBm, tailBm);
                        ++t;
                    }
                } else {
                    const int bm = qw_moe_tile_block(c, 0);
                    for (int r = 0; r * bm < c; ++r) {
                        tileE[t] = int(e);
                        tileR[t] = qw_moe_pack_tile_r(r, bm);
                        ++t;
                    }
                }
            }
        }
        const int totalTiles = simd_broadcast(t, 31);
        if (lane == 31u) tileCount[0] = totalTiles;
        const uint guCol = (writeTiles >> 1u) & 0x7FFFu;
        const uint dnCol = (writeTiles >> 16u) & 0xFFFFu;
        const uint maxTiles = min(totalPairs, min(experts, totalPairs) + (totalPairs + 63u) / 64u);
        if (guCol > 0u && maxTiles > 0u) {
            device uint* args = (device uint*)(tileCount + 4);
            const uint guChunk = max(1u, 2560u / guCol);
            const uint guSlices = (maxTiles + guChunk - 1u) / guChunk;
            for (uint s = lane; s < guSlices; s += 32u) {
                const int rem = totalTiles - int(s * guChunk);
                const uint n = rem > 0 ? min(uint(rem), guChunk) : 0u;
                args[s * 3u + 0u] = n > 0u ? guCol : 0u;
                args[s * 3u + 1u] = n;
                args[s * 3u + 2u] = n > 0u ? 1u : 0u;
            }
            if (dnCol > 0u) {
                device uint* dnArgs = args + guSlices * 3u;
                const uint dnChunk = max(1u, 2560u / dnCol);
                const uint dnSlices = (maxTiles + dnChunk - 1u) / dnChunk;
                for (uint s = lane; s < dnSlices; s += 32u) {
                    const int rem = totalTiles - int(s * dnChunk);
                    const uint n = rem > 0 ? min(uint(rem), dnChunk) : 0u;
                    dnArgs[s * 3u + 0u] = n > 0u ? dnCol : 0u;
                    dnArgs[s * 3u + 1u] = n;
                    dnArgs[s * 3u + 2u] = n > 0u ? 1u : 0u;
                }
            }
        }
    }
}

struct QWGemmMoEGroupBucketsParams {
    uint experts;
    uint totalPairs;
    uint topK;
    uint flags;
    uint guGridX;
    uint dnGridX;
};

/// Fuses pair counting sort (totalPairs <= 512, experts <= 1024) with active-bucket compaction.
kernel void qwg_moe_group_build_buckets_f32(
    device const int* selected [[buffer(0)]],
    device int* expertOff [[buffer(1)]],
    device int* sortedPairIdx [[buffer(2)]],
    device int* sortedTok [[buffer(3)]],
    device int* invPerm [[buffer(4)]],
    device int* activeExperts [[buffer(5)]],
    device uint* indirectArgs [[buffer(6)]],
    constant QWGemmMoEGroupBucketsParams& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (p.experts == 0u || p.experts > 1024u || p.totalPairs > 512u) return;
    threadgroup atomic_int tgCount[1024];
    threadgroup int sgSumC[8], sgOffC[8], chunkTotalC;
    threadgroup uint sgSumA[8], sgOffA[8], chunkTotalA;

    for (uint e = tid; e < p.experts; e += 256u) {
        atomic_store_explicit(&tgCount[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pairIdx[2] = { int(tid), int(tid) + 256 };
    int pairRaw[2] = { -1, -1 };
    for (uint k = 0u; k < 2u; ++k) {
        const int i = pairIdx[k];
        if (i < int(p.totalPairs)) {
            const int rawS = selected[i];
            pairRaw[k] = rawS;
            if (rawS >= 0) {
                const uint e = uint(rawS) & 0xFFFFu;
                if (e < p.experts) {
                    atomic_fetch_add_explicit(&tgCount[e], 1, memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int runningC = 0;
    uint runningA = 0u;
    for (uint base = 0u; base < p.experts; base += 256u) {
        const uint e = base + tid;
        const int c = (e < p.experts)
            ? atomic_load_explicit(&tgCount[e], memory_order_relaxed) : 0;
        const uint isAct = (c > 0) ? 1u : 0u;
        const int laneOffC = simd_prefix_exclusive_sum(c);
        const uint laneOffA = simd_prefix_exclusive_sum(isAct);
        if (lane == 31u) {
            sgSumC[sgi] = laneOffC + c;
            sgSumA[sgi] = laneOffA + isAct;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0u) {
            const int vc = (lane < 8u) ? sgSumC[lane] : 0;
            const int soc = simd_prefix_exclusive_sum(vc);
            if (lane < 8u) sgOffC[lane] = soc;
            if (lane == 7u) chunkTotalC = soc + vc;
            const uint va = (lane < 8u) ? sgSumA[lane] : 0u;
            const uint soa = simd_prefix_exclusive_sum(va);
            if (lane < 8u) sgOffA[lane] = soa;
            if (lane == 7u) chunkTotalA = soa + va;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e < p.experts) {
            const int off = runningC + sgOffC[sgi] + laneOffC;
            expertOff[e] = off;
            atomic_store_explicit(&tgCount[e], off, memory_order_relaxed);
            if (isAct != 0u) {
                activeExperts[runningA + sgOffA[sgi] + laneOffA] = int(e);
            }
        }
        runningC += chunkTotalC;
        runningA += chunkTotalA;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        expertOff[p.experts] = runningC;
        indirectArgs[0] = runningA;
        indirectArgs[4] = (runningA > 0u) ? p.guGridX : 0u;
        indirectArgs[5] = runningA;
        indirectArgs[6] = (runningA > 0u) ? 1u : 0u;
        indirectArgs[8] = (runningA > 0u) ? p.dnGridX : 0u;
        indirectArgs[9] = runningA;
        indirectArgs[10] = (runningA > 0u) ? 1u : 0u;
    }

    const bool writeInv = (p.flags & 1u) != 0u;
    const bool directSlotTok = (p.flags & 2u) != 0u;
    for (uint k = 0u; k < 2u; ++k) {
        const int rawS = pairRaw[k];
        if (rawS < 0) continue;
        const uint raw = uint(rawS);
        const uint e = raw & 0xFFFFu;
        if (e >= p.experts) continue;
        const int i = pairIdx[k];
        const int at = atomic_fetch_add_explicit(&tgCount[e], 1, memory_order_relaxed);
        sortedPairIdx[at] = i;
        if (directSlotTok) {
            sortedTok[at] = i;
        } else {
            const uint gIdx = (raw >> 16u) & 0x7Fu;
            const uint gStride = (raw >> 23u) & 0x7Fu;
            const int t = (p.topK > 0u) ? (i / int(p.topK)) : 0;
            sortedTok[at] = (gStride > 0u) ? (t * int(gStride) + int(gIdx)) : t;
        }
        if (writeInv) invPerm[i] = at;
    }
}

