// qwattn_mla_dsa.metal — Decoupled-RoPE + Dynamic Sparse Attention (DSA) MLA kernels:
// latent norms + paged KV + k_pe/k_idx RoPE + H_128, in-place q_rope/q_idx prep,
// scalar DSA sparse attention, and 8x8 MMA VHA + sigmoid(o_gate) + H_32 exit.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../include/qw_mla_dequant.metalh"
#include "../include/qw_geometry_constants.metalh"
using namespace metal;

struct MLADSALatentPrepParams {
    uint T;
    uint pos0;
    uint qLoraRank;
    uint kvLoraRank;
    uint ropeDim;
    uint idxDim;
    float rmsEps;
    float kNormEps;
    uint kvMode;
    float mscale;
    uint useHadamard;
    uint ropeFlags;
};

struct MLAQRopeIdxPrepParams {
    uint T;
    uint pos0;
    uint qHeads;
    uint idxHeads;
    uint ropeDim;
    uint idxDim;
    uint qRopeOffset;
    uint qChannels;
    uint qFullStride;
    float mscale;
    uint useHadamard;
    uint ropeFlags;
};

struct MLAVHAGateParams {
    uint T;
    uint qHeads;
    uint vHeadDim;
    uint gateStride;
    uint oGateOffset;
    uint hasVHA;
    uint hasOGate;
};

struct MLAHeadGateProjParams {
    uint T;
    uint hidden;
    uint nGate;
};

// Architectural orthonormal 128-point Sylvester Walsh-Hadamard transform (H_128 / sqrt(128))
// across 32 lanes x float4; active regardless of QW_FWHT_ROTATED weight-basis compilation.
METAL_FUNC float4 qw_mla_fwht128_ortho(float4 v, uint lane) {
    const float a0 = v.x + v.y, a1 = v.x - v.y, a2 = v.z + v.w, a3 = v.z - v.w;
    v = float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);
    #pragma unroll
    for (ushort step = 1u; step < 32u; step <<= 1u) {
        const float4 mate = simd_shuffle_xor(v, step);
        v = ((lane & uint(step)) == 0u) ? (v + mate) : (mate - v);
    }
    return v * 0.08838834764831844f;
}

METAL_FUNC float qw_mla_read_kv_scalar(device const uchar* poolBytes,
                                       uint pageID, uint pageOff,
                                       uint halfRank, uint col, uint kvMode) {
    if (kvMode == 0u) {
        return float(((device const half*)poolBytes)[((size_t)pageID * 256u + pageOff) * halfRank + col]);
    } else if (kvMode == 1u) {
        const size_t dataBytes = 256ul * halfRank;
        const size_t pageBytes = dataBytes + 512ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        const half sc = *((device const half*)(pageBase + dataBytes + (size_t)pageOff * 2ul));
        return float(((device const char*)(pageBase + (size_t)pageOff * halfRank))[col]) * float(sc);
    } else {
        const uint groupsPerHalf = max(1u, halfRank >> 7u);
        const size_t dataBytes = 128ul * halfRank;
        const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
        device const uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
        const half2 sm = *((device const half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + (col >> 7u));
        const uchar b = (pageBase + (size_t)pageOff * (halfRank >> 1u))[col >> 1u];
        const uint nib = ((col & 1u) == 0u) ? uint(b & 0x0Fu) : uint(b >> 4u);
        return fma(float(nib), float(sm.x), float(sm.y));
    }
}

// 1. Fused q_a_layernorm + kv_a_layernorm + paged KV write + k_pe GPT-J RoPE + k_idx LN/NeoX/H_128.
// Grid: (T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_dsa_latent_prep(
    device const half*             mlaLatent   [[buffer(0)]],
    device const half*             qANormW     [[buffer(1)]],
    device const half*             kvANormW    [[buffer(2)]],
    device const half*             kNormW      [[buffer(3)]],
    device const half*             kNormB      [[buffer(4)]],
    device const float*            invFreq     [[buffer(5)]],
    device half*                   qResidOut   [[buffer(6)]],
    device half*                   kPool       [[buffer(7)]],
    device half*                   vPool       [[buffer(8)]],
    device const int*              kTable      [[buffer(9)]],
    device const int*              vTable      [[buffer(10)]],
    device half*                   kPeCache    [[buffer(11)]],
    device half*                   pooledKeys  [[buffer(12)]],
    constant MLADSALatentPrepParams& p         [[buffer(13)]],
    device const int*              mropePos    [[buffer(14)]],
    device const uint*             ropeAxisMap [[buffer(15)]],
    uint                           t           [[threadgroup_position_in_grid]],
    uint                           tid         [[thread_index_in_threadgroup]],
    uint                           sg          [[simdgroup_index_in_threadgroup]],
    uint                           lane        [[thread_index_in_simdgroup]]
) {
    if (t >= p.T) return;
    threadgroup float sgScratch[8];
    threadgroup float sgMinScratch[8];
    threadgroup half shX[512];

    const uint pos = p.pos0 + t;
    const uint latentStride = p.qLoraRank + p.kvLoraRank + p.ropeDim + p.idxDim;
    device const half* rowLat = mlaLatent + (size_t)t * latentStride;

    if (p.qLoraRank > 0u) {
        float sqQ = 0.0f;
        for (uint i = tid; i < p.qLoraRank; i += 256u) {
            const float v = float(rowLat[i]);
            sqQ = fma(v, v, sqQ);
        }
        sqQ = simd_sum(sqQ);
        if (lane == 0u) sgScratch[sg] = sqQ;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.qLoraRank) + p.rmsEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float invRmsQ = sgScratch[0];

        device half* dstQResid = qResidOut + (size_t)t * p.qLoraRank;
        for (uint base = sg * 32u; base < p.qLoraRank; base += 256u) {
            const uint idx = base + lane;
            float v = float(rowLat[idx]) * invRmsQ * float(qANormW[idx]);
            dstQResid[idx] = half(qwg_fwht32_lane_f32(v, lane));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const half* kvLat = rowLat + p.qLoraRank;
    float sqKV = 0.0f;
    for (uint i = tid; i < p.kvLoraRank; i += 256u) {
        const float v = float(kvLat[i]);
        sqKV = fma(v, v, sqKV);
    }
    sqKV = simd_sum(sqKV);
    if (lane == 0u) sgScratch[sg] = sqKV;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.kvLoraRank) + p.rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRmsKV = sgScratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pageSlot = pos >> 8u;
    const uint pageOff = pos & 255u;
    const uint pageK = uint(kTable[pageSlot]);
    const uint pageV = uint(vTable[pageSlot]);
    const uint halfRank = p.kvLoraRank >> 1u;

    for (uint base = 0u; base < p.kvLoraRank; base += 256u) {
        const uint idx = base + (sg * 32u + lane);
        float v = float(kvLat[idx]) * invRmsKV * float(kvANormW[idx]);
        v = qwg_fwht32_lane_f32(v, lane);
        const uint col = idx & (halfRank - 1u);
        device uchar* poolBytes = (base < halfRank) ? (device uchar*)kPool : (device uchar*)vPool;
        const uint pageID = (base < halfRank) ? pageK : pageV;

        if (p.kvMode == 0u) {
            device half* dst = ((device half*)poolBytes) + ((size_t)pageID * 256u + pageOff) * halfRank;
            dst[col] = half(v);
        } else if (p.kvMode == 1u) {
            float maxAbs = simd_max(fabs(v));
            if (lane == 0u) sgScratch[sg] = maxAbs;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float mVal = (lane < 8u) ? sgScratch[lane] : 0.0f;
                mVal = simd_max(mVal);
                if (lane == 0u) sgScratch[0] = float(half(max(mVal / 127.0f, 6.103515625e-5f)));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half scH = half(sgScratch[0]);
            const float invSc = (float(scH) > 0.0f) ? (1.0f / float(scH)) : 0.0f;
            const size_t dataBytes = 256ul * halfRank;
            const size_t pageBytes = dataBytes + 512ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            ((device char*)(pageBase + (size_t)pageOff * halfRank))[col] = char(clamp(round(v * invSc), -127.0f, 127.0f));
            if (col == 0u) *((device half*)(pageBase + dataBytes + (size_t)pageOff * 2ul)) = scH;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            const uint grp = sg >> 2u;
            float minV = simd_min(v), maxV = simd_max(v);
            if (lane == 0u) { sgMinScratch[sg] = minV; sgScratch[sg] = maxV; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg < 2u && lane < 4u) {
                const uint sgIdx = sg * 4u + lane;
                float gMin = sgMinScratch[sgIdx], gMax = sgScratch[sgIdx];
                for (uint off = 2u; off >= 1u; off >>= 1u) {
                    gMin = min(gMin, simd_shuffle_xor(gMin, ushort(off)));
                    gMax = max(gMax, simd_shuffle_xor(gMax, ushort(off)));
                }
                if (lane == 0u) {
                    sgScratch[sg * 2u + 0u] = float(half(max((gMax - gMin) / 15.0f, 6.103515625e-5f)));
                    sgScratch[sg * 2u + 1u] = float(half(gMin));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const half scH = half(sgScratch[grp * 2u + 0u]);
            const half mnH = half(sgScratch[grp * 2u + 1u]);
            const float invSc4 = (float(scH) > 0.0f) ? (1.0f / float(scH)) : 0.0f;
            shX[col] = half(uint(clamp(round((v - float(mnH)) * invSc4), 0.0f, 15.0f)));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint groupsPerHalf = max(1u, halfRank >> 7u);
            const size_t dataBytes = 128ul * halfRank;
            const size_t pageBytes = dataBytes + 256ul * groupsPerHalf * 4ul;
            device uchar* pageBase = poolBytes + (size_t)pageID * pageBytes;
            device uchar* qRow = pageBase + (size_t)pageOff * (halfRank >> 1u);
            if (tid < (halfRank >> 1u)) {
                const uint q0 = uint(shX[2u * tid]), q1 = uint(shX[2u * tid + 1u]);
                qRow[tid] = uchar((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
            }
            if (tid < groupsPerHalf) {
                *((device half2*)(pageBase + dataBytes) + (size_t)pageOff * groupsPerHalf + tid) =
                    half2(half(sgScratch[tid * 2u + 0u]), half(sgScratch[tid * 2u + 1u]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k_pe GPT-J or NeoX RoPE (1D or 3D MRoPE) -> kPeCache[pos]
    device const half* kPeRaw = rowLat + p.qLoraRank + p.kvLoraRank;
    const uint halfRope = p.ropeDim >> 1u;
    const bool interleaveRope = (p.ropeFlags & 1u) != 0u;
    const bool hasMRoPE = (p.ropeFlags & 2u) != 0u;
    for (uint pairIdx = tid; pairIdx < halfRope; pairIdx += 256u) {
        const uint ax = hasMRoPE ? ropeAxisMap[pairIdx] : 0u;
        const float rotPos = hasMRoPE ? float(mropePos[(size_t)ax * p.T + t]) : float(pos);
        const float ang = rotPos * invFreq[pairIdx];
        const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
        device half* dstPe = kPeCache + (size_t)pos * p.ropeDim;
        if (interleaveRope) {
            const float even = float(kPeRaw[2u * pairIdx]), odd = float(kPeRaw[2u * pairIdx + 1u]);
            dstPe[2u * pairIdx] = half(even * c - odd * s);
            dstPe[2u * pairIdx + 1u] = half(odd * c + even * s);
        } else {
            const float x0 = float(kPeRaw[pairIdx]), x1 = float(kPeRaw[pairIdx + halfRope]);
            dstPe[pairIdx] = half(x0 * c - x1 * s);
            dstPe[pairIdx + halfRope] = half(x1 * c + x0 * s);
        }
    }

    if (p.idxDim > 0u) {
        // k_idx LayerNorm + NeoX RoPE on [0..ropeDim-1] + H_128 -> pooledKeys[pos]
        device const half* kRaw = kPeRaw + p.ropeDim;
        const float xK = (tid < p.idxDim) ? float(kRaw[tid]) : 0.0f;
        float sumK = simd_sum(xK);
        if (lane == 0u) sgScratch[sg] = sumK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sgScratch[0] = s / float(p.idxDim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float meanK = sgScratch[0];

        const float diffK = (tid < p.idxDim) ? (xK - meanK) : 0.0f;
        float varK = simd_sum(diffK * diffK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) sgScratch[sg] = varK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float s = (lane < 8u) ? sgScratch[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sgScratch[0] = rsqrt(s / float(p.idxDim) + p.kNormEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float invStdK = sgScratch[0];

        if (tid < p.idxDim) {
            shX[tid] = half(diffK * invStdK * float(kNormW[tid]) + float(kNormB[tid]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < halfRope) {
            const float x0 = float(shX[tid]), x1 = float(shX[tid + halfRope]);
            const float ang = float(pos) * invFreq[tid];
            const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
            shX[tid] = half(x0 * c - x1 * s);
            shX[tid + halfRope] = half(x1 * c + x0 * s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u && p.idxDim == 128u) {
            float4 kv4 = float4(*(threadgroup const half4*)(shX + lane * 4u));
            if (p.useHadamard != 0u) kv4 = qw_mla_fwht128_ortho(kv4, lane);
            *(device half4*)(pooledKeys + (size_t)pos * 128u + lane * 4u) = half4(kv4);
        }
    }
}

// 2. In-place GPT-J/NeoX RoPE on q_rope + optional NeoX RoPE and H_128 on q_idx inside mlaQFull.
// Grid: (qHeads + idxHeads, T, 1), Threadgroup: (32, 1, 1).
kernel void qwg_mla_q_rope_idx_prep(
    device half*                 mlaQFull    [[buffer(0)]],
    device const float*          invFreq     [[buffer(1)]],
    constant MLAQRopeIdxPrepParams& p        [[buffer(2)]],
    device const int*            mropePos    [[buffer(3)]],
    device const uint*           ropeAxisMap [[buffer(4)]],
    uint2                        tg          [[threadgroup_position_in_grid]],
    uint                         lane        [[thread_index_in_simdgroup]]
) {
    const uint t = tg.y;
    if (t >= p.T) return;
    const uint pos = p.pos0 + t;
    const uint halfRope = p.ropeDim >> 1u;
    const bool interleaveRope = (p.ropeFlags & 1u) != 0u;
    const bool hasMRoPE = (p.ropeFlags & 2u) != 0u;
    device half* row = mlaQFull + (size_t)t * p.qFullStride;

    if (tg.x < p.qHeads) {
        device half* qr = row + p.qRopeOffset + tg.x * p.ropeDim;
        for (uint pairIdx = lane; pairIdx < halfRope; pairIdx += 32u) {
            const uint ax = hasMRoPE ? ropeAxisMap[pairIdx] : 0u;
            const float rotPos = hasMRoPE ? float(mropePos[(size_t)ax * p.T + t]) : float(pos);
            const float ang = rotPos * invFreq[pairIdx];
            const float c = cos(ang) * p.mscale, s = sin(ang) * p.mscale;
            if (interleaveRope) {
                const float even = float(qr[2u * pairIdx]), odd = float(qr[2u * pairIdx + 1u]);
                qr[2u * pairIdx] = half(even * c - odd * s);
                qr[2u * pairIdx + 1u] = half(odd * c + even * s);
            } else {
                const float x0 = float(qr[pairIdx]), x1 = float(qr[pairIdx + halfRope]);
                qr[pairIdx] = half(x0 * c - x1 * s);
                qr[pairIdx + halfRope] = half(x1 * c + x0 * s);
            }
        }
        return;
    }

    const uint idxH = tg.x - p.qHeads;
    if (idxH >= p.idxHeads || p.idxDim != 128u) return;
    device half* qi = row + p.qChannels + idxH * 128u;
    float4 v = float4(*(device const half4*)(qi + lane * 4u));
    // NeoX RoPE pairs [0..31] (lanes 0..7) with [32..63] (lanes 8..15).
    if (p.ropeDim == 64u) {
        const float4 mate = simd_shuffle_xor(v, ushort(8));
        if (lane < 16u) {
            const uint baseFreq = (lane & 7u) * 4u;
            const float4 ang4 = float(pos) * float4(invFreq[baseFreq + 0u], invFreq[baseFreq + 1u],
                                                    invFreq[baseFreq + 2u], invFreq[baseFreq + 3u]);
            const float4 c4 = cos(ang4) * p.mscale, s4 = sin(ang4) * p.mscale;
            v = (lane < 8u) ? (v * c4 - mate * s4) : (v * c4 + mate * s4);
        }
    }
    if (p.useHadamard != 0u) v = qw_mla_fwht128_ortho(v, lane);
    *(device half4*)(qi + lane * 4u) = half4(v);
}

// 3. O(1)-SRAM streaming scalar Decoupled-RoPE MLA attention (supports arbitrary seqLen & kvLoraRank <= 1024).
// Grid: (qHeads * T, 1, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_dsa_sparse_attn(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    uLatOut   [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device const half*              mlaQFull  [[buffer(8)]],
    device const half*              kPeCache  [[buffer(9)]],
    device const float*             attnSink  [[buffer(10)]],
    uint                            tg        [[threadgroup_position_in_grid]],
    uint                            d         [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;

    threadgroup float shQAbs[1024];
    threadgroup float shQRope[128];
    threadgroup float shScore[8];
    threadgroup uint  shTokJ[8];

    const uint seqLen = p.pos0 + t + 1u;
    const bool isDenseStream = (p.topK == 0u);
    const uint startTok = isDenseStream
        ? ((p.windowTokens > 0u && seqLen > p.windowTokens) ? (seqLen - p.windowTokens) : 0u)
        : 0u;
    const uint denseLim = p.topK + p.windowTokens;
    const uint slots = isDenseStream ? (seqLen - startTok) : min(min(seqLen, denseLim), 2560u);
    const float scale = rsqrt(float(p.headDim));
    const uint halfRank = p.kvLoraRank >> 1u;
    const uint kChunks = min(4u, (p.kvLoraRank + 255u) >> 8u);

    device const half* qRow = qAbs + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    for (uint k = d; k < p.kvLoraRank; k += 256u) shQAbs[k] = float(qRow[k]);
    device const half* qrRow = mlaQFull + (size_t)t * p.qFullStride + p.qRopeOffset + h * p.ropeDim;
    for (uint r = d; r < p.ropeDim; r += 256u) shQRope[r] = float(qrRow[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1e30f;
    float lsum = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const int* rowList = blockList + (size_t)t * p.listStride;

    for (uint blk = 0u; blk < slots; blk += 8u) {
        const uint nG = min(8u, slots - blk);
        const uint slot = blk + sg;
        if (slot < slots) {
            const uint j = isDenseStream
                ? (startTok + slot)
                : ((seqLen <= denseLim) ? slot : min(uint(max(rowList[slot], 0)), seqLen - 1u));
            const uint pageSlot = j >> 8u, pageOff = j & 255u;
            const uint pageK = uint(kTable[pageSlot]), pageV = uint(vTable[pageSlot]);
            float s0 = 0.0f;
            for (uint col = lane; col < halfRank; col += 32u) {
                s0 = fma(shQAbs[col], qw_mla_read_kv_scalar((device const uchar*)kPool, pageK, pageOff, halfRank, col, p.kvMode), s0);
                s0 = fma(shQAbs[halfRank + col], qw_mla_read_kv_scalar((device const uchar*)vPool, pageV, pageOff, halfRank, col, p.kvMode), s0);
            }
            if (p.ropeDim > 0u) {
                device const half* kp = kPeCache + (size_t)j * p.ropeDim;
                for (uint r = lane; r < p.ropeDim; r += 32u) {
                    s0 = fma(shQRope[r], float(kp[r]), s0);
                }
            }
            const float dotVal = simd_sum(s0) * scale;
            if (lane == 0u) {
                shScore[sg] = dotVal;
                shTokJ[sg] = j;
            }
        } else if (lane == 0u) {
            shScore[sg] = -1e30f;
            shTokJ[sg] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mNew = m;
        for (uint g = 0u; g < nG; ++g) mNew = max(mNew, shScore[g]);
        const float corr = (m > -1e29f && mNew > m) ? exp(m - mNew) : ((m > -1e29f) ? 1.0f : 0.0f);
        lsum *= corr;
        for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) acc[cChunk] *= corr;
        for (uint g = 0u; g < nG; ++g) {
            const float w = exp(shScore[g] - mNew);
            lsum += w;
            const uint j = shTokJ[g];
            const uint pageSlot = j >> 8u, pageOff = j & 255u;
            for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) {
                const uint col = cChunk * 256u + d;
                if (col < p.kvLoraRank) {
                    const uint pg = (col < halfRank) ? uint(kTable[pageSlot]) : uint(vTable[pageSlot]);
                    device const uchar* pl = (col < halfRank) ? (device const uchar*)kPool : (device const uchar*)vPool;
                    const uint pc = (col < halfRank) ? col : (col - halfRank);
                    acc[cChunk] = fma(w, qw_mla_read_kv_scalar(pl, pg, pageOff, halfRank, pc, p.kvMode), acc[cChunk]);
                }
            }
        }
        m = mNew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.hasSink != 0u) {
        const float sk = attnSink[h];
        if (sk > -1e29f) {
            const float mFinal = max(m, sk);
            const float corr = (m > -1e29f) ? exp(m - mFinal) : 0.0f;
            for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) acc[cChunk] *= corr;
            lsum = lsum * corr + exp(sk - mFinal);
        }
    }
    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device half* dstU = uLatOut + ((size_t)t * p.qHeads + h) * p.kvLoraRank;
    for (uint cChunk = 0u; cChunk < kChunks; ++cChunk) {
        const uint col = cChunk * 256u + d;
        if (col < p.kvLoraRank) dstU[col] = half(acc[cChunk] * invL);
    }
}

// 4a. Direct raw F16 gate projection for MLA g_proj (`[T, hidden] x [nGate, hidden]^T -> [T, nGate]`).
// Grid: (nGate, T, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_head_gate_proj(
    device const half*              x    [[buffer(0)]],
    device const half*              w    [[buffer(1)]],
    device half*                    out  [[buffer(2)]],
    constant MLAHeadGateProjParams& p    [[buffer(3)]],
    uint2                           tg   [[threadgroup_position_in_grid]],
    uint                            tid  [[thread_index_in_threadgroup]],
    uint                            sg   [[simdgroup_index_in_threadgroup]],
    uint                            lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x, t = tg.y;
    if (j >= p.nGate || t >= p.T) return;
    device const half* xRow = x + (size_t)t * p.hidden;
    device const half* wRow = w + (size_t)j * p.hidden;
    float acc = 0.0f;
    for (uint k = tid; k < p.hidden; k += 128u) {
        acc = fma(float(xRow[k]), float(wRow[k]), acc);
    }
    acc = simd_sum(acc);
    threadgroup float sgSum[4];
    if (lane == 0u) sgSum[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        out[(size_t)t * p.nGate + j] = half(sgSum[0] + sgSum[1] + sgSum[2] + sgSum[3]);
    }
}

// 4b. Fused 8x8 MMA VHA head mixing + sigmoid(o_gate) (element-wise or head-wise) + H_32 -> attOut.
// Grid: (vHeadDim / 32, T, 1), Threadgroup: (256, 1, 1).
kernel void qwg_mla_vha_gate_rot(
    device const half*       yIn      [[buffer(0)]],
    device const half*       gateSrc  [[buffer(1)]],
    device const half*       vhaMix   [[buffer(2)]],
    device half*             attOut   [[buffer(3)]],
    constant MLAVHAGateParams& p      [[buffer(4)]],
    uint2                    tg       [[threadgroup_position_in_grid]],
    uint                     tid      [[thread_index_in_threadgroup]],
    uint                     sg       [[simdgroup_index_in_threadgroup]],
    uint                     lane     [[thread_index_in_simdgroup]]
) {
    const uint dBase = tg.x * 32u;
    const uint t = tg.y;
    if (t >= p.T || dBase >= p.vHeadDim) return;

    constexpr uint YS = 40u;
    constexpr uint OS = 36u;
    threadgroup half  shY[64u * YS];
    threadgroup float shOut[64u * OS];

    const uint rowStride = p.qHeads * p.vHeadDim;
    device const half* yRow = yIn + (size_t)t * rowStride;
    device const half* gRow = gateSrc + (size_t)t * p.gateStride + p.oGateOffset;
    device half* dstRow = attOut + (size_t)t * rowStride;

    if (p.hasVHA != 0u && (p.qHeads & 7u) == 0u && p.qHeads <= 64u) {
        for (uint i = tid; i < p.qHeads * 8u; i += 256u) {
            const uint g = i >> 3u, c4 = (i & 7u) * 4u;
            *(threadgroup half4*)(shY + g * YS + c4) = *(device const half4*)(yRow + g * p.vHeadDim + dBase + c4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hBase = sg * 8u;
        if (hBase < p.qHeads) {
            simdgroup_float8x8 acc[4] = {
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f),
                simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f)
            };
            for (uint g0 = 0u; g0 < p.qHeads; g0 += 8u) {
                simdgroup_half8x8 mf;
                simdgroup_load(mf, vhaMix + hBase * p.qHeads + g0, p.qHeads);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_half8x8 yf;
                    simdgroup_load(yf, shY + g0 * YS + j * 8u, YS);
                    simdgroup_multiply_accumulate(acc[j], mf, yf, acc[j]);
                }
            }
            threadgroup float* sDst = shOut + hBase * OS;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) simdgroup_store(acc[j], sDst + j * 8u, OS);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hr = 0u; hr < 8u; ++hr) {
                const uint h = hBase + hr;
                const uint d = dBase + lane;
                float val = sDst[hr * OS + lane];
                if (p.hasOGate == 1u) {
                    const float g = float(gRow[h * p.vHeadDim + d]);
                    val *= 1.0f / (1.0f + exp(-g));
                } else if (p.hasOGate == 2u) {
                    const float g = float(gRow[h]);
                    val *= 1.0f / (1.0f + exp(-g));
                }
                dstRow[h * p.vHeadDim + d] = half(qwg_fwht32_lane_f32(val, lane));
            }
        }
        return;
    }

    for (uint h = sg; h < p.qHeads; h += 8u) {
        const uint d = dBase + lane;
        float val = 0.0f;
        if (p.hasVHA != 0u) {
            for (uint g = 0u; g < p.qHeads; ++g) {
                val = fma(float(vhaMix[h * p.qHeads + g]), float(yRow[g * p.vHeadDim + d]), val);
            }
        } else {
            val = float(yRow[h * p.vHeadDim + d]);
        }
        if (p.hasOGate == 1u) {
            const float g = float(gRow[h * p.vHeadDim + d]);
            val *= 1.0f / (1.0f + exp(-g));
        } else if (p.hasOGate == 2u) {
            const float g = float(gRow[h]);
            val *= 1.0f / (1.0f + exp(-g));
        }
        dstRow[h * p.vHeadDim + d] = half(qwg_fwht32_lane_f32(val, lane));
    }
}
