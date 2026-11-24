// qwrouter.metal — Sparse-MoE router on GPU: precise f32 softmax and top-k selection.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
#include "include/qw_weight_basis.metalh"
#include "include/qw_fastmath_safe.metalh"
#include "include/qw_router_union.metalh"

// 32-point normalized WHT in f32 (8 elements per thread across 4 threads).
inline void qwg_route_fwht32_f32(thread float (&f)[8], uint lane_id) {
    // The STANDARD basis: the hidden is not rotated, so neither is the
    // router's copy of it.
    if (!QW_FWHT_ROTATED) return;
    for (int i = 0; i < 8; i += 2) { float a = f[i], b = f[i + 1]; f[i] = a + b; f[i + 1] = a - b; }
    for (int i = 0; i < 8; i += 4) {
        float a0 = f[i], a1 = f[i + 1], b0 = f[i + 2], b1 = f[i + 3];
        f[i] = a0 + b0; f[i + 1] = a1 + b1; f[i + 2] = a0 - b0; f[i + 3] = a1 - b1;
    }
    for (int i = 0; i < 4; ++i) { float a = f[i], b = f[i + 4]; f[i] = a + b; f[i + 4] = a - b; }
    bool neg1 = (lane_id & 1u) != 0;
    for (int i = 0; i < 8; ++i) { float p = simd_shuffle_xor(f[i], ushort(1)); f[i] = neg1 ? (p - f[i]) : (f[i] + p); }
    bool neg2 = (lane_id & 2u) != 0;
    constexpr float inv_sqrt32 = 0.17677669529663689f;
    for (int i = 0; i < 8; ++i) {
        float p = simd_shuffle_xor(f[i], ushort(2));
        f[i] = (neg2 ? (p - f[i]) : (f[i] + p)) * inv_sqrt32;
    }
}

// Batched router top-k for prefill (T threadgroups x 32 threads; 1 simdgroup per token row, zero barrier).
kernel void qwg_moe_route(device const float* logits     [[buffer(0)]],
                          device int*         idx        [[buffer(1)]],
                          device float*       wts        [[buffer(2)]],
                          constant uint&      E          [[buffer(3)]],
                          constant uint&      K          [[buffer(4)]],
                          constant uint&      normTopK   [[buffer(5)]],
                          device const float* bias       [[buffer(6)]],
                          device const float* scaleParam [[buffer(7)]],
                          device const int*   latentLabels [[buffer(8)]],
                          device int*         groupIds     [[buffer(9)]],
                          constant uint&      numGroups    [[buffer(10)]],
                          uint t    [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]]) {
    const bool doNorm = (normTopK & 1u) != 0u;
    const bool hasBias = (normTopK & 2u) != 0u;
    const bool splitFeature = (normTopK & 8u) != 0u;
    const bool hasScaleParam = (normTopK & 16u) != 0u;
    const bool useSigmoid = ((normTopK & 32u) != 0u) || hasBias || splitFeature;
    const uint nGroup = (normTopK >> 8u) & 0xFFu;
    const uint topkGroup = (normTopK >> 16u) & 0xFFu;
    const bool groupTop2 = (normTopK & (1u << 24u)) != 0u;
    const uint logitStride = splitFeature ? (2u * E) : E;
    const uint JN = min((E + 31u) >> 5u, 32u);
    float lv[32], ex[32], p[32];
    bool validScore[32];
    float mx = -3.4e38f;
    bool hasFiniteLogit = false;
    for (uint j = 0u; j < JN; j++) {
        const uint id = lane + 32u * j;
        if (id < E) {
            const float v = logits[(size_t)t * logitStride + id];
            lv[j] = v;
            if (qw_is_finite_bits(v)) {
                mx = hasFiniteLogit ? max(mx, v) : v;
                hasFiniteLogit = true;
            }
        }
    }
    if (useSigmoid) {
        for (uint j = 0u; j < JN; j++) {
            const uint id = lane + 32u * j;
            if (id < E) {
                float s = 1.0f / (1.0f + exp(-lv[j]));
                bool sfOk = true;
                if (splitFeature) {
                    const float lv1 = logits[(size_t)t * logitStride + E + id];
                    s += 1.0f / (1.0f + exp(-lv1));
                    sfOk = qw_is_finite_bits(lv1);
                }
                ex[j] = s;
                const float score = hasBias ? (s + bias[id]) : s;
                p[j] = score;
                validScore[j] = qw_is_finite_bits(lv[j]) && sfOk && qw_is_finite_bits(s) && qw_is_finite_bits(score);
            } else {
                validScore[j] = false;
            }
        }
    } else {
        const bool anyFinite = simd_any(hasFiniteLogit);
        const float maxLogit = anyFinite ? simd_max(hasFiniteLogit ? mx : -3.4e38f) : 0.0f;
        float total = 0.0f;
        for (uint j = 0u; j < JN; j++) {
            const uint id = lane + 32u * j;
            const float e = (id < E && qw_is_finite_bits(lv[j])) ? exp(lv[j] - maxLogit) : 0.0f;
            ex[j] = e;
            total += simd_sum(e);
        }
        for (uint j = 0u; j < JN; j++) {
            const uint id = lane + 32u * j;
            const bool ok = (id < E) && qw_is_finite_bits(lv[j]) && (total > 0.0f);
            p[j] = ok ? (ex[j] / total) : -1e30f;
            validScore[j] = ok;
        }
    }
    if (nGroup > 1u && topkGroup > 0u && topkGroup < nGroup && nGroup <= 64u && (E % nGroup) == 0u) {
        const uint epg = E / nGroup;
        float gScores[64];
        bool gHas[64];
        for (uint g = 0u; g < nGroup; ++g) {
            float m1 = -3.4e38f, m2 = -3.4e38f;
            uint i1 = 0xFFFFFFFFu;
            bool has1 = false, has2 = false;
            for (uint j = 0u; j < JN; ++j) {
                const uint id = lane + 32u * j;
                if (id < E && (id / epg) == g && validScore[j]) {
                    const float v = p[j];
                    if (!has1 || v > m1) {
                        if (has1) { m2 = m1; has2 = true; }
                        m1 = v; i1 = id; has1 = true;
                    } else if (!has2 || v > m2) {
                        m2 = v; has2 = true;
                    }
                }
            }
            const bool any1 = simd_any(has1);
            const float gMax1 = simd_max(has1 ? m1 : -3.4e38f);
            const uint gIdx1 = any1 ? simd_min((has1 && m1 == gMax1) ? i1 : 0xFFFFFFFFu) : 0xFFFFFFFFu;
            float score = -3.4e38f;
            if (any1) {
                if (groupTop2 && epg > 1u) {
                    const bool laneHas2 = (i1 == gIdx1) ? has2 : has1;
                    const float cand2 = (i1 == gIdx1) ? m2 : m1;
                    const bool any2 = simd_any(laneHas2);
                    const float gMax2 = simd_max(laneHas2 ? cand2 : -3.4e38f);
                    score = any2 ? (gMax1 + gMax2) : gMax1;
                } else {
                    score = gMax1;
                }
            }
            gScores[g] = score;
            gHas[g] = any1;
        }
        uint winMask[2] = {0u, 0u};
        bool gTaken[64];
        for (uint g = 0u; g < nGroup; ++g) gTaken[g] = false;
        for (uint r = 0u; r < topkGroup; ++r) {
            int bestG = -1;
            float bestS = -3.4e38f;
            for (uint g = 0u; g < nGroup; ++g) {
                if (!gTaken[g] && gHas[g] && (bestG < 0 || gScores[g] > bestS)) {
                    bestS = gScores[g];
                    bestG = int(g);
                }
            }
            if (bestG >= 0) {
                gTaken[uint(bestG)] = true;
                winMask[uint(bestG) >> 5u] |= (1u << (uint(bestG) & 31u));
            }
        }
        for (uint j = 0u; j < JN; ++j) {
            const uint id = lane + 32u * j;
            if (id < E) {
                const uint g = id / epg;
                if ((winMask[g >> 5u] & (1u << (g & 31u))) == 0u) validScore[j] = false;
            }
        }
    }
    bool taken[32];
    for (uint j = 0u; j < 32u; j++) taken[j] = false;
    float wsel[32];
    uint activeMask = 0u;
    for (uint r = 0u; r < K; r++) {
        float bv = -3.4e38f;
        uint bi = 0xFFFFFFFFu;
        bool hasCand = false;
        for (uint j = 0u; j < JN; j++) {
            const uint id = lane + 32u * j;
            if (id < E && !taken[j] && validScore[j]) {
                if (!hasCand || p[j] > bv || (p[j] == bv && id < bi)) {
                    bv = p[j];
                    bi = id;
                    hasCand = true;
                }
            }
        }
        const bool anyCand = simd_any(hasCand);
        const float vMax = simd_max(hasCand ? bv : -3.4e38f);
        const uint w = anyCand ? simd_min((hasCand && bv == vMax) ? bi : 0xFFFFFFFFu) : 0xFFFFFFFFu;
        if (w < E && (w & 31u) == lane) {
            const uint wj = w >> 5u;
            if (wj < 32u) taken[wj] = true;
        }
        const uint wj = (w < E) ? (w >> 5u) : 0u;
        float own = 0.0f;
        if (w < E) {
            for (uint j = 0u; j < JN; j++) if (wj == j) own = useSigmoid ? ex[j] : p[j];
        }
        const float pw = (w < E) ? simd_broadcast(own, ushort(w & 31u)) : 0.0f;
        if (lane == 0u) {
            const size_t slot = (size_t)t * K + r;
            if (w < E) {
                if (numGroups > 0u) {
                    const int g = latentLabels[w];
                    activeMask |= (1u << uint(g));
                    idx[slot] = (int(numGroups) << 23) | (g << 16) | int(w);
                } else {
                    idx[slot] = int(w);
                }
                wsel[r] = pw;
            } else {
                idx[slot] = -1;
                wsel[r] = 0.0f;
            }
        }
    }
    if (lane == 0u) {
        const float rScale = (QW_FC_ROUTED_SCALE > 0.0f) ? QW_FC_ROUTED_SCALE : 1.0f;
        float acc = 0.0f;
        if (doNorm) {
            for (uint r = 0u; r < K; r++) acc += wsel[r];
            if (useSigmoid) acc += 1e-20f;
        }
        for (uint r = 0u; r < K; r++) {
            const size_t slot = (size_t)t * K + r;
            const int rawId = idx[slot];
            const float scale = (hasScaleParam && rawId >= 0) ? scaleParam[uint(rawId) & 0xFFFFu] : rScale;
            const float normW = doNorm ? ((acc > 0.0f) ? (wsel[r] / acc) : 0.0f) : wsel[r];
            wts[slot] = normW * scale;
        }
        if (numGroups > 0u) {
            for (uint g = 0u; g < numGroups; ++g) {
                groupIds[(size_t)t * numGroups + g] = ((activeMask & (1u << g)) != 0u) ? int(g) : -1;
            }
        }
    }
}

// Single-token decode router top-k (tg 0, 512 threads = 16 simdgroups, 1 expert/thread) + optional fused
// shared-expert silu_mul + H_32 (tg 1 when IS > 0).
kernel void qwg_moe_route_decode(device const float* logits0    [[buffer(0)]],
                                 device int*         idx0       [[buffer(1)]],
                                 device float*       wts0       [[buffer(2)]],
                                 constant uint&      E          [[buffer(3)]],
                                 constant uint&      K          [[buffer(4)]],
                                 constant uint&      normTopK   [[buffer(5)]],
                                 device const float* guSh0      [[buffer(6)]],
                                 device float*       actSh0     [[buffer(7)]],
                                 constant uint&      IS         [[buffer(8)]],
                                 device const float* bias       [[buffer(9)]],
                                 device const float* scaleParam [[buffer(10)]],
                                 device const int*   latentLabels [[buffer(11)]],
                                 device int*         groupIds     [[buffer(12)]],
                                 constant uint&      numGroups    [[buffer(13)]],
                                 uint2 tgp  [[threadgroup_position_in_grid]],
                                 uint2 tpt  [[thread_position_in_threadgroup]],
                                 uint sgi  [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    // tgp.y = row (verify batches): each row is an independent T = 1 route on its own slices.
    const uint tg = tgp.x, row = tgp.y, tid = tpt.x;
    threadgroup QWRUnionScratch unionSc;
    if ((normTopK & 4u) != 0u) {
        // MoE-Spec (arXiv:2602.16052) union-aware budgeting across verify window (256-thread 1-TG):
        // E holds verify rows M (2..16), K holds topK (<= 16, M*K <= 256), IS holds expertBudget B.
        qwg_router_union_prune_tg(idx0, wts0, E, K, IS, (normTopK & 1u) != 0u, tid, sgi, lane, unionSc);
        return;
    }
    const bool splitFeature = (normTopK & 8u) != 0u;
    const bool hasScaleParam = (normTopK & 16u) != 0u;
    const uint logitStride = splitFeature ? (2u * E) : E;
    device const float* logits = logits0 + (size_t)row * logitStride;
    device int*         idx    = idx0 + (size_t)row * K;
    device float*       wts    = wts0 + (size_t)row * K;
    device const float* guSh   = guSh0 + (size_t)row * 2u * IS;
    device float*       actSh  = actSh0 + (size_t)row * IS;
    if (tg == 1u) {
        const uint mi = tid * 8u;
        if (mi >= IS) return;
        device const float* g = guSh + mi;
        device const float* u = g + IS;
        float ov[8];
        #pragma unroll
        for (uint k = 0; k < 8u; ++k) {
            ov[k] = qwg_swiglu_mul(g[k], u[k]);
        }
        if ((IS & 31u) == 0u) qwg_route_fwht32_f32(ov, tid & 31u);
        device float* o = actSh + mi;
        #pragma unroll
        for (uint k = 0; k < 8u; ++k) o[k] = ov[k];
        return;
    }

    const bool doNorm = (normTopK & 1u) != 0u;
    const bool hasBias = (normTopK & 2u) != 0u;
    const bool useSigmoid = ((normTopK & 32u) != 0u) || hasBias || splitFeature;
    const uint nGroup = (normTopK >> 8u) & 0xFFu;
    const uint topkGroup = (normTopK >> 16u) & 0xFFu;
    const bool groupTop2 = (normTopK & (1u << 24u)) != 0u;
    threadgroup float sMaxArr[16];
    threadgroup float sSumArr[16];
    threadgroup float sVal[2][16];
    threadgroup uint  sIdx[2][16];
    threadgroup float wsel[32];
    threadgroup float sGroupP[1024];
    threadgroup uchar sGroupValid[1024];
    threadgroup float sGroupScore[64];
    threadgroup uchar sGroupHas[64];
    threadgroup uint  sWinMask[2];

    const uint JN = min((E + 31u) >> 5u, 16u);
    const uint e[2] = { tid, tid + 512u };
    const bool valid[2] = { e[0] < E, e[1] < E };
    const float lv[2] = {
        valid[0] ? logits[e[0]] : -1e30f,
        valid[1] ? logits[e[1]] : -1e30f
    };
    const bool lvFinite[2] = {
        valid[0] && qw_is_finite_bits(lv[0]),
        valid[1] && qw_is_finite_bits(lv[1])
    };
    float p[2] = { -1e30f, -1e30f };
    bool cand[2] = { false, false };
    if (useSigmoid) {
        for (uint pass = 0u; pass < 2u; ++pass) {
            float sig = lvFinite[pass] ? (1.0f / (1.0f + exp(-lv[pass]))) : 0.0f;
            bool sfOk = true;
            if (splitFeature && valid[pass]) {
                const float lvSecond = logits[E + e[pass]];
                sig += 1.0f / (1.0f + exp(-lvSecond));
                sfOk = qw_is_finite_bits(lvSecond);
            }
            const float score = valid[pass] ? (hasBias ? (sig + bias[e[pass]]) : sig) : -1e30f;
            cand[pass] = lvFinite[pass] && sfOk && qw_is_finite_bits(sig) && qw_is_finite_bits(score);
            p[pass] = cand[pass] ? score : -1e30f;
        }
    } else {
        float myMax = -3.4e38f;
        bool myHas = false;
        if (lvFinite[0]) { myMax = lv[0]; myHas = true; }
        if (lvFinite[1]) { myMax = myHas ? max(myMax, lv[1]) : lv[1]; myHas = true; }
        const float sgMax = simd_max(myMax);
        const bool sgHas = simd_any(myHas);
        if (lane == 0u) sMaxArr[sgi] = sgHas ? sgMax : -3.4e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float maxLogit = simd_max((lane < JN) ? sMaxArr[lane] : -3.4e38f);
        const float ex0 = lvFinite[0] ? exp(lv[0] - maxLogit) : 0.0f;
        const float ex1 = lvFinite[1] ? exp(lv[1] - maxLogit) : 0.0f;
        const float sgSum = simd_sum(ex0 + ex1);
        if (lane == 0u) sSumArr[sgi] = sgSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total = 0.0f;
        for (uint j = 0u; j < JN; j++) total += sSumArr[j];
        cand[0] = lvFinite[0] && (total > 0.0f);
        cand[1] = lvFinite[1] && (total > 0.0f);
        p[0] = cand[0] ? (ex0 / total) : -1e30f;
        p[1] = cand[1] ? (ex1 / total) : -1e30f;
    }
    if (nGroup > 1u && topkGroup > 0u && topkGroup < nGroup && nGroup <= 64u && (E % nGroup) == 0u) {
        if (valid[0]) { sGroupP[e[0]] = p[0]; sGroupValid[e[0]] = cand[0] ? 1u : 0u; }
        if (valid[1]) { sGroupP[e[1]] = p[1]; sGroupValid[e[1]] = cand[1] ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint epg = E / nGroup;
        for (uint g = sgi; g < nGroup; g += 16u) {
            const uint gBase = g * epg;
            float m1 = -3.4e38f, m2 = -3.4e38f;
            uint i1 = 0xFFFFFFFFu;
            bool has1 = false, has2 = false;
            for (uint k = lane; k < epg; k += 32u) {
                const uint id = gBase + k;
                if (sGroupValid[id] != 0u) {
                    const float v = sGroupP[id];
                    if (!has1 || v > m1) {
                        if (has1) { m2 = m1; has2 = true; }
                        m1 = v; i1 = id; has1 = true;
                    } else if (!has2 || v > m2) {
                        m2 = v; has2 = true;
                    }
                }
            }
            const bool any1 = simd_any(has1);
            const float gMax1 = simd_max(has1 ? m1 : -3.4e38f);
            const uint gIdx1 = any1 ? simd_min((has1 && m1 == gMax1) ? i1 : 0xFFFFFFFFu) : 0xFFFFFFFFu;
            float score = -3.4e38f;
            if (any1) {
                if (groupTop2 && epg > 1u) {
                    const bool laneHas2 = (i1 == gIdx1) ? has2 : has1;
                    const float cand2 = (i1 == gIdx1) ? m2 : m1;
                    const bool any2 = simd_any(laneHas2);
                    const float gMax2 = simd_max(laneHas2 ? cand2 : -3.4e38f);
                    score = any2 ? (gMax1 + gMax2) : gMax1;
                } else {
                    score = gMax1;
                }
            }
            if (lane == 0u) { sGroupScore[g] = score; sGroupHas[g] = any1 ? 1u : 0u; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0u) {
            if (lane < 2u) sWinMask[lane] = 0u;
            bool gTaken[2] = { false, false };
            for (uint r = 0u; r < topkGroup; ++r) {
                float bv = -3.4e38f;
                uint bi = 0xFFFFFFFFu;
                bool bhas = false;
                for (uint pass = 0u; pass < 2u; ++pass) {
                    const uint g = lane + 32u * pass;
                    if (g < nGroup && !gTaken[pass] && sGroupHas[g] != 0u) {
                        const float v = sGroupScore[g];
                        if (!bhas || v > bv) { bv = v; bi = g; bhas = true; }
                    }
                }
                const bool anyG = simd_any(bhas);
                const float gMax = simd_max(bhas ? bv : -3.4e38f);
                const uint gw = anyG ? simd_min((bhas && bv == gMax) ? bi : 0xFFFFFFFFu) : 0xFFFFFFFFu;
                if (gw < nGroup) {
                    if ((gw & 31u) == lane) gTaken[gw >> 5u] = true;
                    if (lane == 0u) sWinMask[gw >> 5u] |= (1u << (gw & 31u));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid[0]) {
            const uint g0 = e[0] / epg;
            if ((sWinMask[g0 >> 5u] & (1u << (g0 & 31u))) == 0u) cand[0] = false;
        }
        if (valid[1]) {
            const uint g1 = e[1] / epg;
            if ((sWinMask[g1 >> 5u] & (1u << (g1 & 31u))) == 0u) cand[1] = false;
        }
    }
    {
        float myVal = -3.4e38f;
        uint myIdx = 0xFFFFFFFFu;
        bool myHas = false;
        if (cand[0]) { myVal = p[0]; myIdx = e[0]; myHas = true; }
        if (cand[1] && (!myHas || p[1] > myVal)) { myVal = p[1]; myIdx = e[1]; myHas = true; }
        const float bMax = simd_max(myHas ? myVal : -3.4e38f);
        const bool sgHas = simd_any(myHas);
        const uint bIdx = sgHas ? simd_min((myHas && myVal == bMax) ? myIdx : 0xFFFFFFFFu) : 0xFFFFFFFFu;
        if (lane == 0u) {
            sVal[0][sgi] = sgHas ? bMax : -1e30f;
            sIdx[0][sgi] = bIdx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool activeSG = (lane < JN);
    uint activeMask = 0u;
    for (uint r = 0u; r < K; r++) {
        const uint cur = r & 1u;
        const uint nxt = cur ^ 1u;
        const uint bi = activeSG ? sIdx[cur][lane] : 0xFFFFFFFFu;
        const bool hasSG = (bi != 0xFFFFFFFFu);
        const float bv = hasSG ? sVal[cur][lane] : -3.4e38f;
        const bool anyCand = simd_any(hasSG);
        const float vMax = simd_max(bv);
        const uint w = anyCand ? simd_min((hasSG && bv == vMax) ? bi : 0xFFFFFFFFu) : 0xFFFFFFFFu;
        if (tid == 0u) {
            if (w < E) {
                if (numGroups > 0u) {
                    const int g = latentLabels[w];
                    activeMask |= (1u << uint(g));
                    idx[r] = (int(numGroups) << 23) | (g << 16) | int(w);
                } else {
                    idx[r] = int(w);
                }
                if (useSigmoid) {
                    float s = 1.0f / (1.0f + exp(-logits[w]));
                    if (splitFeature) s += 1.0f / (1.0f + exp(-logits[E + w]));
                    wsel[r] = s;
                } else {
                    wsel[r] = vMax;
                }
            } else {
                idx[r] = -1;
                wsel[r] = 0.0f;
            }
        }
        if (r + 1u == K) break;
        const uint winSG = (w < E) ? ((w & 511u) >> 5u) : 0xFFFFFFFFu;
        if (sgi == winSG) {
            if (e[0] == w) cand[0] = false;
            if (e[1] == w) cand[1] = false;
            float myVal = -3.4e38f;
            uint myIdx = 0xFFFFFFFFu;
            bool myHas = false;
            if (cand[0]) { myVal = p[0]; myIdx = e[0]; myHas = true; }
            if (cand[1] && (!myHas || p[1] > myVal)) { myVal = p[1]; myIdx = e[1]; myHas = true; }
            const float bMax = simd_max(myHas ? myVal : -3.4e38f);
            const bool sgHas = simd_any(myHas);
            const uint bIdx = sgHas ? simd_min((myHas && myVal == bMax) ? myIdx : 0xFFFFFFFFu) : 0xFFFFFFFFu;
            if (lane == 0u) {
                sVal[nxt][sgi] = sgHas ? bMax : -1e30f;
                sIdx[nxt][sgi] = bIdx;
            }
        } else if (lane == 0u) {
            sVal[nxt][sgi] = sVal[cur][sgi];
            sIdx[nxt][sgi] = sIdx[cur][sgi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float rScale = (QW_FC_ROUTED_SCALE > 0.0f) ? QW_FC_ROUTED_SCALE : 1.0f;
        float acc = 0.0f;
        if (doNorm) {
            for (uint r = 0u; r < K; r++) acc += wsel[r];
            if (useSigmoid) acc += 1e-20f;
        }
        for (uint r = 0u; r < K; r++) {
            const int rawId = idx[r];
            const float scale = (hasScaleParam && rawId >= 0) ? scaleParam[uint(rawId) & 0xFFFFu] : rScale;
            const float normW = doNorm ? ((acc > 0.0f) ? (wsel[r] / acc) : 0.0f) : wsel[r];
            wts[r] = normW * scale;
        }
        if (numGroups > 0u) {
            device int* gOut = groupIds + (size_t)row * numGroups;
            for (uint g = 0u; g < numGroups; ++g) {
                gOut[g] = ((activeMask & (1u << g)) != 0u) ? int(g) : -1;
            }
        }
    }
}

// Decode router logits + shared-expert sigmoid gate from BF16 weights [E + 1, K] (K % 4 == 0).
// Dispatch: ((E + 1 + 3) / 4, T) threadgroups x 128 threads (1 simdgroup = 1 row, 4 rows per TG, zero barrier).
kernel void qwg_moe_router_bf16_f32(device const ushort* W      [[buffer(0)]],
                                    device const float*  X      [[buffer(1)]],
                                    device float*        logits [[buffer(2)]],
                                    device float*        sg     [[buffer(3)]],
                                    constant uint&       E      [[buffer(4)]],
                                    constant uint&       K      [[buffer(5)]],
                                    uint2 tg   [[threadgroup_position_in_grid]],
                                    uint  sgi  [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const uint n = tg.x * 4u + sgi, t = tg.y;
    if (n > E) return;
    const uint K4 = K >> 2u;
    device const ushort4* Wn4 = (device const ushort4*)(W + (size_t)n * K);
    device const float4* Xt4 = (device const float4*)(X + (size_t)t * K);
    float acc = 0.0f;
    for (uint k4 = lane; k4 < K4; k4 += 32u) {
        const ushort4 w16 = Wn4[k4];
        const float4 x4 = Xt4[k4];
        const float4 w4 = as_type<float4>(uint4(w16) << 16u);
        acc = fma(w4.x, x4.x, acc);
        acc = fma(w4.y, x4.y, acc);
        acc = fma(w4.z, x4.z, acc);
        acc = fma(w4.w, x4.w, acc);
    }
    const float y = simd_sum(acc);
    if (lane == 0u) {
        if (n < E) {
            logits[(size_t)t * E + n] = y;
        } else {
            sg[t] = 1.0f / (1.0f + exp(-y));
        }
    }
}

// Router logits for a batch of rows: Y[t, e] = sum_k X[t, k] * W[e, k] (f32, bf16, or f16 weights). Tiled 32 tokens x 32 experts per
// threadgroup, so each router weight tile is read once per 32 tokens instead of once per token. K % 32 == 0.
kernel void qwg_router_gemm(device const float* W [[buffer(0)]],
                            device const float* X [[buffer(1)]],
                            device float*       Y [[buffer(2)]],
                            constant uint&      M [[buffer(3)]],
                            constant uint&      N [[buffer(4)]],
                            constant uint&      K [[buffer(5)]],
                            constant uint&      weightMode [[buffer(6)]],
                            device const float* shGateW [[buffer(7)]],
                            device float*       shGateOut [[buffer(8)]],
                            constant uint&      hasShGate [[buffer(9)]],
                            uint2 tg  [[threadgroup_position_in_grid]],
                            uint  lid [[thread_index_in_threadgroup]],
                            uint2 tpg [[threads_per_threadgroup]]) {
    device const ushort* W16 = (device const ushort*)W;
    if (M <= 8u) {
        // Decode / MTP verify (M = 1..8): one expert per simdgroup (same lane-strided sum, so bit-identical to M=1)
        // reading W once across all M rows to fill every GPU core with zero threadgroup barriers.
        const uint sgi = lid >> 5u;
        const uint lane = lid & 31u;
        const uint e = tg.x * (tpg.x >> 5u) + sgi;
        const uint K4 = K >> 2u;
        device const float4* X4 = (device const float4*)X;
        const bool computeSh = (hasShGate != 0u && e == N);
        device const float4* shW4 = (device const float4*)shGateW;
        if (e < N) {
            float4 acc0 = float4(0.0f);
            float4 acc1 = float4(0.0f);
            if (weightMode == 1u) {
                device const ushort4* w4p = (device const ushort4*)(W16 + (size_t)e * K);
                for (uint k4 = lane; k4 < K4; k4 += 32u) {
                    const float4 wv = as_type<float4>(uint4(w4p[k4]) << 16u);
                    acc0.x += dot(wv, X4[k4]);
                    if (M > 1u) acc0.y += dot(wv, X4[(size_t)K4 + k4]);
                    if (M > 2u) acc0.z += dot(wv, X4[(size_t)2u * K4 + k4]);
                    if (M > 3u) acc0.w += dot(wv, X4[(size_t)3u * K4 + k4]);
                    if (M > 4u) {
                        acc1.x += dot(wv, X4[(size_t)4u * K4 + k4]);
                        if (M > 5u) acc1.y += dot(wv, X4[(size_t)5u * K4 + k4]);
                        if (M > 6u) acc1.z += dot(wv, X4[(size_t)6u * K4 + k4]);
                        if (M > 7u) acc1.w += dot(wv, X4[(size_t)7u * K4 + k4]);
                    }
                }
            } else if (weightMode == 2u) {
                device const ushort4* w4p = (device const ushort4*)(W16 + (size_t)e * K);
                for (uint k4 = lane; k4 < K4; k4 += 32u) {
                    const float4 wv = float4(as_type<half4>(w4p[k4]));
                    acc0.x += dot(wv, X4[k4]);
                    if (M > 1u) acc0.y += dot(wv, X4[(size_t)K4 + k4]);
                    if (M > 2u) acc0.z += dot(wv, X4[(size_t)2u * K4 + k4]);
                    if (M > 3u) acc0.w += dot(wv, X4[(size_t)3u * K4 + k4]);
                    if (M > 4u) {
                        acc1.x += dot(wv, X4[(size_t)4u * K4 + k4]);
                        if (M > 5u) acc1.y += dot(wv, X4[(size_t)5u * K4 + k4]);
                        if (M > 6u) acc1.z += dot(wv, X4[(size_t)6u * K4 + k4]);
                        if (M > 7u) acc1.w += dot(wv, X4[(size_t)7u * K4 + k4]);
                    }
                }
            } else {
                device const float4* w4p = (device const float4*)(W + (size_t)e * K);
                for (uint k4 = lane; k4 < K4; k4 += 32u) {
                    const float4 wv = w4p[k4];
                    acc0.x += dot(wv, X4[k4]);
                    if (M > 1u) acc0.y += dot(wv, X4[(size_t)K4 + k4]);
                    if (M > 2u) acc0.z += dot(wv, X4[(size_t)2u * K4 + k4]);
                    if (M > 3u) acc0.w += dot(wv, X4[(size_t)3u * K4 + k4]);
                    if (M > 4u) {
                        acc1.x += dot(wv, X4[(size_t)4u * K4 + k4]);
                        if (M > 5u) acc1.y += dot(wv, X4[(size_t)5u * K4 + k4]);
                        if (M > 6u) acc1.z += dot(wv, X4[(size_t)6u * K4 + k4]);
                        if (M > 7u) acc1.w += dot(wv, X4[(size_t)7u * K4 + k4]);
                    }
                }
            }
            acc0 = simd_sum(acc0);
            if (M > 4u) acc1 = simd_sum(acc1);
            if (lane == 0u) {
                Y[e] = acc0.x;
                if (M > 1u) Y[(size_t)N + e] = acc0.y;
                if (M > 2u) Y[(size_t)2u * N + e] = acc0.z;
                if (M > 3u) Y[(size_t)3u * N + e] = acc0.w;
                if (M > 4u) {
                    Y[(size_t)4u * N + e] = acc1.x;
                    if (M > 5u) Y[(size_t)5u * N + e] = acc1.y;
                    if (M > 6u) Y[(size_t)6u * N + e] = acc1.z;
                    if (M > 7u) Y[(size_t)7u * N + e] = acc1.w;
                }
            }
        } else if (computeSh) {
            float4 sh0 = float4(0.0f);
            float4 sh1 = float4(0.0f);
            for (uint k4 = lane; k4 < K4; k4 += 32u) {
                const float4 wv = shW4[k4];
                sh0.x += dot(wv, X4[k4]);
                if (M > 1u) sh0.y += dot(wv, X4[(size_t)K4 + k4]);
                if (M > 2u) sh0.z += dot(wv, X4[(size_t)2u * K4 + k4]);
                if (M > 3u) sh0.w += dot(wv, X4[(size_t)3u * K4 + k4]);
                if (M > 4u) {
                    sh1.x += dot(wv, X4[(size_t)4u * K4 + k4]);
                    if (M > 5u) sh1.y += dot(wv, X4[(size_t)5u * K4 + k4]);
                    if (M > 6u) sh1.z += dot(wv, X4[(size_t)6u * K4 + k4]);
                    if (M > 7u) sh1.w += dot(wv, X4[(size_t)7u * K4 + k4]);
                }
            }
            sh0 = simd_sum(sh0);
            if (M > 4u) sh1 = simd_sum(sh1);
            if (lane == 0u) {
                shGateOut[0] = 1.0f / (1.0f + exp(-sh0.x));
                if (M > 1u) shGateOut[1] = 1.0f / (1.0f + exp(-sh0.y));
                if (M > 2u) shGateOut[2] = 1.0f / (1.0f + exp(-sh0.z));
                if (M > 3u) shGateOut[3] = 1.0f / (1.0f + exp(-sh0.w));
                if (M > 4u) {
                    shGateOut[4] = 1.0f / (1.0f + exp(-sh1.x));
                    if (M > 5u) shGateOut[5] = 1.0f / (1.0f + exp(-sh1.y));
                    if (M > 6u) shGateOut[6] = 1.0f / (1.0f + exp(-sh1.z));
                    if (M > 7u) shGateOut[7] = 1.0f / (1.0f + exp(-sh1.w));
                }
            }
        }
        return;
    }
    threadgroup float As[32][36];
    threadgroup float Bs[32][36];
    threadgroup float shWs[32];
    const uint t0 = tg.y * 32u, e0 = tg.x * 32u;
    const uint tx = lid & 31u, ty = lid >> 5u;
    const uint sgM = ty >> 1u, sgN = ty & 1u;
    const uint r = lid >> 3u, c = (lid & 7u) << 2u;
    const bool computeSh = (hasShGate != 0u && tg.x == 0u);
    float shAcc = 0.0f;
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);
    for (uint k0 = 0u; k0 < K; k0 += 32u) {
        const float4 xv = (t0 + r < M) ? *(device const float4*)(X + (size_t)(t0 + r) * K + k0 + c) : float4(0.0f);
        float4 wv = float4(0.0f);
        if (e0 + r < N) {
            if (weightMode == 1u) {
                const ushort4 w16 = *(device const ushort4*)(W16 + (size_t)(e0 + r) * K + k0 + c);
                wv = as_type<float4>(uint4(w16) << 16u);
            } else if (weightMode == 2u) {
                const ushort4 w16 = *(device const ushort4*)(W16 + (size_t)(e0 + r) * K + k0 + c);
                wv = float4(as_type<half4>(w16));
            } else {
                wv = *(device const float4*)(W + (size_t)(e0 + r) * K + k0 + c);
            }
        }
        As[r][c + 0u] = xv.x; As[r][c + 1u] = xv.y; As[r][c + 2u] = xv.z; As[r][c + 3u] = xv.w;
        Bs[r][c + 0u] = wv.x; Bs[r][c + 1u] = wv.y; Bs[r][c + 2u] = wv.z; Bs[r][c + 3u] = wv.w;
        if (computeSh && lid < 32u) shWs[lid] = shGateW[k0 + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint k8 = 0u; k8 < 4u; k8++) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a,  &As[sgM * 8u][k8 * 8u], 36, ulong2(0, 0), false);
            simdgroup_load(b0, &Bs[sgN * 16u + 0u][k8 * 8u], 36, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_load(b1, &Bs[sgN * 16u + 8u][k8 * 8u], 36, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
        }
        if (computeSh && lid < 32u) {
            #pragma unroll
            for (uint kk = 0u; kk < 32u; ++kk) shAcc = fma(As[lid][kk], shWs[kk], shAcc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0, &As[sgM * 8u][sgN * 16u + 0u], 36, ulong2(0, 0), false);
    simdgroup_store(acc1, &As[sgM * 8u][sgN * 16u + 8u], 36, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; i++) {
        const uint m = ty * 4u + i;
        const uint row = t0 + m, col = e0 + tx;
        if (row < M && col < N) Y[(size_t)row * N + col] = As[m][tx];
    }
    if (computeSh && lid < 32u && t0 + lid < M) {
        shGateOut[t0 + lid] = 1.0f / (1.0f + exp(-shAcc));
    }
}
