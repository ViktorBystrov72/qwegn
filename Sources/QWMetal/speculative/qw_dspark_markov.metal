// qw_dspark_markov.metal — Target tap capture, SWA-128 context KV write,
// bidirectional block HCA attention, Markov HQ8/AQ8 logits/argmax, and confidence dot.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "include/qw_dequant.metalh"
#include "include/qw_fastmath_safe.metalh"
using namespace metal;

struct DSparkTapH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline DSparkTapH8 dspark_as_h8(uint4 w) { return *(thread DSparkTapH8*)&w; }
static inline float dspark_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16u); }

static inline void dspark_fwht32_f32_inplace(thread float (&f)[8], uint laneId) {
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float a = f[i], b = f[i + 1];
        f[i] = a + b;
        f[i + 1] = a - b;
    }
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const float a0 = f[i], a1 = f[i + 1], b0 = f[i + 2], b1 = f[i + 3];
        f[i] = a0 + b0;     f[i + 1] = a1 + b1;
        f[i + 2] = a0 - b0; f[i + 3] = a1 - b1;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = f[i], b = f[i + 4];
        f[i] = a + b;
        f[i + 4] = a - b;
    }
    const bool neg1 = (laneId & 1u) != 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float p = simd_shuffle_xor(f[i], ushort(1));
        f[i] = neg1 ? (p - f[i]) : (f[i] + p);
    }
    const bool neg2 = (laneId & 2u) != 0u;
    constexpr float kInvSqrt32 = 0.17677669529663689f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float p = simd_shuffle_xor(f[i], ushort(2));
        f[i] = (neg2 ? (p - f[i]) : (f[i] + p)) * kInvSqrt32;
    }
}

static inline float dspark_fwht32_lane_f32(float v, uint laneId) {
    #pragma unroll
    for (ushort s = 1u; s < 32u; s <<= 1u) {
        const float p = simd_shuffle_xor(v, s);
        v = ((laneId & s) == 0u) ? (v + p) : (p - v);
    }
    return v * 0.17677669529663689f;
}

// 1. Capture mHC stream mean (or 1-stream residual) into tapBuf [maxRows, totalSlots * H].
// flags: bit0=deferred f32 inject, bit1=deferred mHC f16 merge, bit2=1-stream f16 source, bit3=apply FWHT32.
kernel void qwg_dspark_mhc_tap_mean(
    device const float* base         [[buffer(0)]],
    device const void*  src          [[buffer(1)]],
    device const float* inj          [[buffer(2)]],
    device uint4*       dst          [[buffer(3)]],
    constant uint&      srcRowOffset [[buffer(4)]],
    constant uint&      dstRowOffset [[buffer(5)]],
    constant uint&      slot         [[buffer(6)]],
    constant uint&      hiddenU4     [[buffer(7)]],
    constant uint&      totalSlots   [[buffer(8)]],
    constant uint&      hc           [[buffer(9)]],
    constant uint&      flags        [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ti [[thread_index_in_threadgroup]]
) {
    const uint hidden = hiddenU4 << 3u;
    const uint sRow = srcRowOffset + tg;
    const uint dRow = dstRowOffset + tg;
    const bool applyFwht = (flags & 8u) != 0u;
    device uint4* yr = dst + (size_t)dRow * totalSlots * hiddenU4 + (size_t)slot * hiddenU4;

    if ((flags & 4u) != 0u) {
        device const uint4* xr = ((device const uint4*)base) + (size_t)sRow * hiddenU4;
        for (uint u = ti; u < hiddenU4; u += 32u) {
            DSparkTapH8 hv = dspark_as_h8(xr[u]);
            if (applyFwht) {
                float f[8];
                for (int h = 0; h < 8; ++h) f[h] = float(hv.v[h]);
                dspark_fwht32_f32_inplace(f, ti);
                for (int h = 0; h < 8; ++h) hv.v[h] = half(f[h]);
            }
            yr[u] = *(thread uint4*)&hv;
        }
        return;
    }

    const float invHC = 1.0f / float(max(1u, hc));
    device const float* rowBase = base + (size_t)sRow * hc * hidden;
    for (uint u = ti; u < hiddenU4; u += 32u) {
        float f[8];
        if ((flags & 2u) != 0u) {
            DSparkTapH8 bv = dspark_as_h8(*((device const uint4*)src + (size_t)sRow * hiddenU4 + u));
            device const float* ctrl = inj + (size_t)sRow * (hc + hc * hc);
            const uint hBase = u << 3u;
            for (int h = 0; h < 8; ++h) {
                const float b = float(bv.v[h]);
                float sumS = 0.0f;
                for (uint i = 0u; i < hc; ++i) {
                    float ns = ctrl[i] * b;
                    for (uint j = 0u; j < hc; ++j) {
                        ns = fma(ctrl[hc + i * hc + j], rowBase[(size_t)j * hidden + hBase + uint(h)], ns);
                    }
                    sumS += ns;
                }
                f[h] = sumS * invHC;
            }
        } else {
            float4 sum0 = float4(0.0f), sum1 = float4(0.0f);
            if ((flags & 1u) != 0u) {
                device const float4* sr = (device const float4*)((device const float*)src + (size_t)sRow * hidden) + u * 2u;
                const float4 sv0 = sr[0], sv1 = sr[1];
                device const float* injRow = inj + (size_t)sRow * hc;
                for (uint c = 0u; c < hc; ++c) {
                    const float sc = injRow[c];
                    device const float4* bc = (device const float4*)(rowBase + (size_t)c * hidden) + u * 2u;
                    sum0 += bc[0] + sv0 * sc;
                    sum1 += bc[1] + sv1 * sc;
                }
            } else {
                for (uint c = 0u; c < hc; ++c) {
                    device const float4* bc = (device const float4*)(rowBase + (size_t)c * hidden) + u * 2u;
                    sum0 += bc[0];
                    sum1 += bc[1];
                }
            }
            sum0 *= invHC;
            sum1 *= invHC;
            f[0] = sum0.x; f[1] = sum0.y; f[2] = sum0.z; f[3] = sum0.w;
            f[4] = sum1.x; f[5] = sum1.y; f[6] = sum1.z; f[7] = sum1.w;
        }
        if (applyFwht) {
            dspark_fwht32_f32_inplace(f, ti);
        }
        DSparkTapH8 rv;
        for (int h = 0; h < 8; ++h) rv.v[h] = half(f[h]);
        yr[u] = *(thread uint4*)&rv;
    }
}

struct DSparkCtxKVParams {
    uint T;
    uint dim;
    uint nopeDim;
    uint ropeDim;
    uint pos0;
    uint ringSize;
    uint flags;
    float rmsEps;
};

// 2. RMSNorm + optional plain GPT-J RoPE -> circular SWA ring or linear row buffer.
// flags: bit0=linear write (row t instead of (pos0+t)%ringSize), bit1=apply FWHT32 (when ropeDim==0).
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_dspark_swa_ctx_kv_write(
    device const half*          rawIn   [[buffer(0)]],
    device const half*          normW   [[buffer(1)]],
    device const float*         invFreq [[buffer(2)]],
    device half*                kvOut   [[buffer(3)]],
    constant DSparkCtxKVParams& p       [[buffer(4)]],
    uint t    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (t >= p.T || p.dim == 0u || p.dim > 8192u) return;
    threadgroup float sgScratch[8];
    threadgroup half  shOut[4096];

    device const half* xRow = rawIn + (size_t)t * p.dim;
    float sq = 0.0f;
    for (uint d = tid; d < p.dim; d += 256u) {
        const float v = float(xRow[d]);
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) sgScratch[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const float s = simd_sum((lane < 8u) ? sgScratch[lane] : 0.0f);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.dim) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgScratch[0];

    const uint pos = p.pos0 + t;
    const uint dstRow = ((p.flags & 1u) != 0u) ? t : (pos % max(1u, p.ringSize));
    device half* dst = kvOut + (size_t)dstRow * p.dim;

    if ((p.flags & 2u) != 0u && p.ropeDim == 0u) {
        for (uint base = sg * 32u; base < p.dim; base += 256u) {
            const uint d = base + lane;
            if (d < p.dim) {
                const float v = float(xRow[d]) * invRms * float(normW[d]);
                dst[d] = half(dspark_fwht32_lane_f32(v, lane));
            }
        }
        return;
    }

    if (p.ropeDim > 0u && p.dim <= 4096u) {
        for (uint d = tid; d < p.dim; d += 256u) {
            shOut[d] = half(float(xRow[d]) * invRms * float(normW[d]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint halfRope = p.ropeDim >> 1u;
        for (uint i = tid; i < halfRope; i += 256u) {
            const uint b = p.nopeDim + 2u * i;
            if (b + 1u < p.dim) {
                const float even = float(shOut[b]);
                const float odd  = float(shOut[b + 1u]);
                const float ang = float(pos) * invFreq[i];
                const float c = cos(ang), s = sin(ang);
                shOut[b]      = half(even * c - odd * s);
                shOut[b + 1u] = half(odd * c + even * s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < p.dim; d += 256u) {
            dst[d] = shOut[d];
        }
        return;
    }

    for (uint d = tid; d < p.dim; d += 256u) {
        dst[d] = half(float(xRow[d]) * invRms * float(normW[d]));
    }
}

struct DSparkBlockAttnParams {
    uint B;
    uint qHeads;
    uint headDim;
    uint qUpStride;
    uint pos0;
    uint ringSize;
    uint slidingWindow;
    uint hasSink;
    uint swaMaskMode;
};

#define DSPARK_ATTN_TK 8u
#define DSPARK_ATTN_KS 520u

// 3. Bidirectional-within-block + SWA-ring context MQA/HCA attention with optional sink.
// Grid: ((qHeads + 7) / 8, B, 1), Threadgroup: (256, 1, 1) -> 1 warp per Q head.
kernel void qwg_dspark_block_hca_attn(
    device const half*              qUpGate  [[buffer(0)]],
    device const half*              swaKV    [[buffer(1)]],
    device const half*              blkKV    [[buffer(2)]],
    device const float*             attnSink [[buffer(3)]],
    device half*                    attOut   [[buffer(4)]],
    constant DSparkBlockAttnParams& p        [[buffer(5)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint h0 = tg.x * 8u;
    const uint t = tg.y;
    const uint h = h0 + sg;
    const uint hd = p.headDim;
    if (t >= p.B || h0 >= p.qHeads || hd == 0u || hd > 512u || (hd & 31u) != 0u) return;

    threadgroup half ks[DSPARK_ATTN_TK * DSPARK_ATTN_KS];
    const uint nChunks = hd >> 5u;
    const float scale = rsqrt(float(hd));

    float qv[16], ov[16];
    for (uint c = 0u; c < 16u; ++c) {
        qv[c] = 0.0f;
        ov[c] = 0.0f;
    }
    if (h < p.qHeads) {
        device const half* qRow = qUpGate + (size_t)t * p.qUpStride + (size_t)h * hd;
        for (uint c = 0u; c < nChunks; ++c) {
            qv[c] = float(qRow[c * 32u + lane]) * scale;
        }
    }

    float mH = -1e30f, lH = 0.0f;
    if (h < p.qHeads && p.hasSink != 0u) {
        const float sk = attnSink[h];
        if (sk > -1e29f) { mH = sk; lH = 1.0f; }
    }

    const uint absQ = p.pos0 + t;
    const uint ctxRef = (p.swaMaskMode != 0u) ? p.pos0 : (absQ + 1u);
    const uint startK = (ctxRef > p.slidingWindow) ? (ctxRef - p.slidingWindow) : 0u;
    const uint nCtx = (p.pos0 > startK) ? (p.pos0 - startK) : 0u;
    const uint totalSlots = nCtx + p.B;
    const uint vecCols = hd >> 2u;
    const uint ringSz = max(1u, p.ringSize);

    for (uint s0 = 0u; s0 < totalSlots; s0 += DSPARK_ATTN_TK) {
        for (uint i = tid; i < DSPARK_ATTN_TK * vecCols; i += 256u) {
            const uint r = i / vecCols;
            const uint c4 = (i - r * vecCols) * 4u;
            const uint slot = s0 + r;
            half4 v = half4(0.0h);
            if (slot < totalSlots) {
                if (slot < nCtx) {
                    const uint kPos = startK + slot;
                    const uint ringIdx = kPos % ringSz;
                    v = *(device const half4*)(swaKV + (size_t)ringIdx * hd + c4);
                } else {
                    const uint bIdx = slot - nCtx;
                    v = *(device const half4*)(blkKV + (size_t)bIdx * hd + c4);
                }
            }
            *(threadgroup half4*)(ks + r * DSPARK_ATTN_KS + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tileRows = min(DSPARK_ATTN_TK, totalSlots - s0);
        if (h < p.qHeads) {
            for (uint r = 0u; r < tileRows; ++r) {
                threadgroup const half* kr = ks + r * DSPARK_ATTN_KS;
                float kv[16];
                float dot = 0.0f;
                for (uint c = 0u; c < nChunks; ++c) {
                    const float kVal = float(kr[c * 32u + lane]);
                    kv[c] = kVal;
                    dot = fma(qv[c], kVal, dot);
                }
                const float score = simd_sum(dot);
                const float mNew = max(mH, score);
                const float corr = (lH > 0.0f && mH > -1e29f) ? exp(mH - mNew) : 0.0f;
                const float w = exp(score - mNew);
                lH = lH * corr + w;
                for (uint c = 0u; c < nChunks; ++c) {
                    ov[c] = fma(w, kv[c], ov[c] * corr);
                }
                mH = mNew;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (h < p.qHeads) {
        const float invL = (lH > 0.0f) ? (1.0f / lH) : 0.0f;
        device half* dst = attOut + ((size_t)t * p.qHeads + h) * hd;
        for (uint c = 0u; c < nChunks; ++c) {
            dst[c * 32u + lane] = half(ov[c] * invL);
        }
    }
}

struct DSparkMarkovParams {
    uint stepIdx;
    uint V;
    uint rank;
    uint flags;
    uint partCount;
};

static inline float dspark_hq8_row_dot(
    device const char* rowPtr,
    uint rank,
    uint nGroups,
    uint g,
    uint sub,
    bool laneActive,
    thread const float4& lv0,
    thread const float4& lv1,
    float latSum,
    threadgroup const float* latSh
) {
    float acc = 0.0f;
    if (rank <= 256u && (rank & 31u) == 0u) {
        if (laneActive) {
            device const char* gp = rowPtr + (size_t)g * 36u;
            const half2 sb = *(device const half2*)gp;
            const packed_uint2 q2 = *(device const packed_uint2*)(gp + 4u + sub * 8u);
            const float4 f0 = float4(as_type<uchar4>(q2.x));
            const float4 f1 = float4(as_type<uchar4>(q2.y));
            acc = float(sb.x) * (dot(f0, lv0) + dot(f1, lv1));
            if (sub == 0u) acc += float(sb.y) * latSum;
        }
    } else {
        for (uint grp = 0u; grp < nGroups; ++grp) {
            device const char* gp = rowPtr + (size_t)grp * 36u;
            const half2 sb = *(device const half2*)gp;
            const uint code = ((device const uchar*)(gp + 4u))[g * 4u + sub];
            const float lat = latSh[grp * 32u + g * 4u + sub];
            acc = fma(float(sb.x) * float(code) + float(sb.y), lat, acc);
        }
    }
    return simd_sum(acc);
}

static inline void dspark_load_latent_sh(
    device const ushort* markovW1,
    device const int32_t* tokSeq,
    uint stepIdx,
    uint V,
    uint rank,
    bool rotateLatent,
    uint tid,
    uint lane,
    threadgroup float* latSh,
    threadgroup float* latSumSh
) {
    const int32_t tok = tokSeq[stepIdx];
    const bool validTok = (tok >= 0 && uint(tok) < V);
    device const ushort* w1Row = validTok ? (markovW1 + (size_t)uint(tok) * rank) : markovW1;
    for (uint c = tid; c < rank; c += 256u) {
        latSh[c] = validTok ? dspark_bf16_to_f32(w1Row[c]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rotateLatent) {
        for (uint base = (tid >> 5u) * 32u; base < rank; base += 256u) {
            const uint idx = base + lane;
            float v = (idx < rank) ? latSh[idx] : 0.0f;
            v = dspark_fwht32_lane_f32(v, lane);
            if (idx < rank) latSh[idx] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint sg = tid >> 5u;
    const uint nGroups = rank >> 5u;
    if (sg < min(nGroups, 8u)) {
        const float ls = simd_sum(latSh[sg * 32u + lane]);
        if (lane == 0u) latSumSh[sg] = ls;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 4. Fused Markov residual (w1[tok[step]] @ w2^T) + base_logits + partial sortable argmax keys.
// Grid: (partCount, 1, 1), Threadgroup: (256, 1, 1). Reduce keysHi/keysLo with qwg_head_argmax_fin.
kernel void qwg_dspark_markov_aq8_argmax(
    device const float*         baseLogits [[buffer(0)]],
    device const ushort*        markovW1   [[buffer(1)]],
    device const char*          markovW2   [[buffer(2)]],
    device const int32_t*       tokSeq     [[buffer(3)]],
    device uint*                keysHi     [[buffer(4)]],
    device uint*                keysLo     [[buffer(5)]],
    constant DSparkMarkovParams& p         [[buffer(6)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tg >= p.partCount || (p.rank & 31u) != 0u || p.rank > 1024u) return;
    threadgroup float latSh[1024];
    threadgroup float latSumSh[8];
    dspark_load_latent_sh(markovW1, tokSeq, p.stepIdx, p.V, p.rank, (p.flags & 1u) != 0u,
                          tid, lane, latSh, latSumSh);

    const uint nGroups = p.rank >> 5u;
    const size_t rowBytes = (size_t)nGroups * 36u;
    const size_t baseOff = (size_t)p.stepIdx * p.V;
    const uint rowsPerTG = (((p.V + p.partCount - 1u) / p.partCount) + 7u) & ~7u;
    const uint vStart = tg * rowsPerTG;
    const uint vEnd = min(p.V, vStart + rowsPerTG);

    const uint g = lane >> 2u;
    const uint sub = lane & 3u;
    const bool laneActive = (p.rank <= 256u) && (g < nGroups);
    const uint b0 = laneActive ? (g * 32u + sub * 8u) : 0u;
    const float4 lv0 = laneActive ? float4(latSh[b0], latSh[b0 + 1u], latSh[b0 + 2u], latSh[b0 + 3u]) : float4(0.0f);
    const float4 lv1 = laneActive ? float4(latSh[b0 + 4u], latSh[b0 + 5u], latSh[b0 + 6u], latSh[b0 + 7u]) : float4(0.0f);
    const float latSum = (laneActive && sub == 0u) ? latSumSh[g] : 0.0f;

    uint bh = 0u, bl = 0u;
    for (uint v = vStart + sg; v < vEnd; v += 8u) {
        const float dotVal = dspark_hq8_row_dot(
            markovW2 + (size_t)v * rowBytes, p.rank, nGroups, g, sub, laneActive,
            lv0, lv1, latSum, latSh);
        if (lane == 0u) {
            const float combined = baseLogits[baseOff + v] + dotVal;
            const uint kh = qw_sortable_f32_u32(combined);
            const uint kl = ~v;
            if (kh > bh || (kh == bh && kl > bl)) { bh = kh; bl = kl; }
        }
    }

    threadgroup uint sh[8], sl[8];
    if (lane == 0u) { sh[sg] = bh; sl[sg] = bl; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint winH = sh[0], winL = sl[0];
        for (uint s = 1u; s < 8u; ++s) {
            if (sh[s] > winH || (sh[s] == winH && sl[s] > winL)) { winH = sh[s]; winL = sl[s]; }
        }
        keysHi[tg] = winH;
        keysLo[tg] = winL;
    }
}

// 5. Full Markov logit evaluation for stochastic sampling: outLogits[v] = base_logits[j,v] + e_j @ w2[v]^T.
// Grid: ((V + 7) / 8, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_dspark_markov_aq8_logits(
    device const float*         baseLogits [[buffer(0)]],
    device const ushort*        markovW1   [[buffer(1)]],
    device const char*          markovW2   [[buffer(2)]],
    device const int32_t*       tokSeq     [[buffer(3)]],
    device float*               outLogits  [[buffer(4)]],
    constant DSparkMarkovParams& p         [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if ((p.rank & 31u) != 0u || p.rank > 1024u) return;
    const uint v = tg * 8u + sg;
    if (tg * 8u >= p.V) return;

    threadgroup float latSh[1024];
    threadgroup float latSumSh[8];
    dspark_load_latent_sh(markovW1, tokSeq, p.stepIdx, p.V, p.rank, (p.flags & 1u) != 0u,
                          tid, lane, latSh, latSumSh);
    if (v >= p.V) return;

    const uint nGroups = p.rank >> 5u;
    const size_t rowBytes = (size_t)nGroups * 36u;
    const uint g = lane >> 2u;
    const uint sub = lane & 3u;
    const bool laneActive = (p.rank <= 256u) && (g < nGroups);
    const uint b0 = laneActive ? (g * 32u + sub * 8u) : 0u;
    const float4 lv0 = laneActive ? float4(latSh[b0], latSh[b0 + 1u], latSh[b0 + 2u], latSh[b0 + 3u]) : float4(0.0f);
    const float4 lv1 = laneActive ? float4(latSh[b0 + 4u], latSh[b0 + 5u], latSh[b0 + 6u], latSh[b0 + 7u]) : float4(0.0f);
    const float latSum = (laneActive && sub == 0u) ? latSumSh[g] : 0.0f;

    const float dotVal = dspark_hq8_row_dot(
        markovW2 + (size_t)v * rowBytes, p.rank, nGroups, g, sub, laneActive,
        lv0, lv1, latSum, latSh);
    if (lane == 0u) {
        outLogits[v] = baseLogits[(size_t)p.stepIdx * p.V + v] + dotVal;
    }
}

struct DSparkConfParams {
    uint stepIdx;
    uint H;
    uint R;
    uint hiddenRowIdx;
    uint V;
    uint flags;
};

// 6. Confidence head dot: p_j = sigmoid(conf_w @ [head_hidden[j], e_j]).
// flags: bit0=un-rotate headHidden via FWHT32 in registers. Grid: (1, 1, 1), TG: (256, 1, 1).
kernel void qwg_dspark_confidence_dot(
    device const half*         headHidden [[buffer(0)]],
    device const ushort*       markovW1   [[buffer(1)]],
    device const ushort*       confW      [[buffer(2)]],
    device const int32_t*      tokSeq     [[buffer(3)]],
    device float*              confOut    [[buffer(4)]],
    constant DSparkConfParams& p          [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const half* hRow = headHidden + (size_t)p.hiddenRowIdx * p.H;
    const bool unrotate = (p.flags & 1u) != 0u;
    float acc = 0.0f;

    for (uint base = sg * 32u; base < p.H; base += 256u) {
        const uint d = base + lane;
        float hv = (d < p.H) ? float(hRow[d]) : 0.0f;
        if (unrotate) hv = dspark_fwht32_lane_f32(hv, lane);
        if (d < p.H) acc = fma(dspark_bf16_to_f32(confW[d]), hv, acc);
    }

    const int32_t tok = tokSeq[p.stepIdx];
    const bool validTok = (tok >= 0 && uint(tok) < p.V);
    device const ushort* eRow = validTok ? (markovW1 + (size_t)uint(tok) * p.R) : markovW1;
    for (uint r = tid; r < p.R; r += 256u) {
        const float ev = validTok ? dspark_bf16_to_f32(eRow[r]) : 0.0f;
        acc = fma(dspark_bf16_to_f32(confW[p.H + r]), ev, acc);
    }

    acc = simd_sum(acc);
    threadgroup float sgAcc[8];
    if (lane == 0u) sgAcc[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const float tot = simd_sum((lane < 8u) ? sgAcc[lane] : 0.0f);
        if (lane == 0u) confOut[p.stepIdx] = 1.0f / (1.0f + exp(-tot));
    }
}
