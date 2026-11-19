// qwple.metal — Per-layer embedding math after key/value projections.
// gate: per (token, stream), g = <keyNorm, queryNorm> / sqrt(H); gate = sign(g) * sqrt(max(|g|, 1e-6));
//       gv[t, c, :] = sigmoid(gate) * vals[t, :].
// conv: out[t, c] = gv[t, c] + SiLU(sum_j w[c, j] * gvNorm[t - (K - 1 - j) * dilation, c]), causal, zero history.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline void qwg_ple_gate_impl(device const float* keyNorm, device const float* queryNorm,
                               device const float* vals, device float* gv,
                               uint H, uint hc, uint valStride, uint tg, uint tid,
                               threadgroup float* partial, threadgroup float* sig) {
    const uint W = H * hc;
    const uint t = tg / hc, c = tg - t * hc;
    float g = 0.0f;
    for (uint d = tid; d < H; d += 128u) g += keyNorm[(size_t)t * W + (size_t)c * H + d] * queryNorm[(size_t)t * W + (size_t)c * H + d];
    g = simd_sum(g);
    if ((tid & 31u) == 0u) partial[tid >> 5u] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float total = (partial[0] + partial[1] + partial[2] + partial[3]) * rsqrt(float(H));
        const float mag = sqrt(max(fabs(total), 1e-6f));
        const float gate = (total > 0.0f) ? mag : ((total < 0.0f) ? -mag : 0.0f);
        sig[0] = 1.0f / (1.0f + exp(-gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < H; d += 128u) gv[(size_t)t * W + (size_t)c * H + d] = sig[0] * vals[(size_t)t * valStride + d];
}

kernel void qwg_ple_gate(device const float* keyNorm [[buffer(0)]], device const float* queryNorm [[buffer(1)]],
                         device const float* vals [[buffer(2)]], device float* gv [[buffer(3)]],
                         constant uint& H [[buffer(4)]], constant uint& hc [[buffer(5)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[4], sig[1];
    qwg_ple_gate_impl(keyNorm, queryNorm, vals, gv, H, hc, H, tg, tid, partial, sig);
}

kernel void qwg_ple_gate_strided(device const float* keyNorm [[buffer(0)]], device const float* queryNorm [[buffer(1)]],
                                 device const float* vals [[buffer(2)]], device float* gv [[buffer(3)]],
                                 constant uint& H [[buffer(4)]], constant uint& hc [[buffer(5)]],
                                 constant uint& valStride [[buffer(6)]],
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[4], sig[1];
    qwg_ple_gate_impl(keyNorm, queryNorm, vals, gv, H, hc, valStride, tg, tid, partial, sig);
}

// conv buffers: 0 gv f32 [T, W] | 1 gvNorm f32 [T, W] | 2 convWeight f32 [W, K] | 3 out f32 [T, W] | 4 W | 5 K | 6 dilation | 7 T
// dispatch: (W, T) threads.
// Causal dilated depthwise conv over normed values; rows before the chunk come from `hist` ([HL, W]), HL = 0 = none.
kernel void qwg_ple_fused_gate_norm(device const float* kv        [[buffer(0)]],
                                    device const float* queryNorm [[buffer(1)]],
                                    device const float* normKey   [[buffer(2)]],
                                    device const float* normConv  [[buffer(3)]],
                                    device float*       gv        [[buffer(4)]],
                                    device float*       gvNorm    [[buffer(5)]],
                                    constant uint&      H         [[buffer(6)]],
                                    constant uint&      hc        [[buffer(7)]],
                                    constant uint&      kvStride  [[buffer(8)]],
                                    constant float&     eps       [[buffer(9)]],
                                    uint2 tg  [[threadgroup_position_in_grid]],
                                    uint  tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x, t = tg.y;
    const uint W = hc * H;
    const size_t kBase = (size_t)t * kvStride + (size_t)c * H;
    const size_t vBase = (size_t)t * kvStride + W;
    const size_t wBase = (size_t)t * W + (size_t)c * H;
    const size_t wOff = (size_t)c * H;
    const uint sg = tid >> 5u, lane = tid & 31u;
    threadgroup float partialK[8], partialKQ[4], partialGV[8], tgSig;

    float kSq = 0.0f;
    for (uint d = tid; d < H; d += 256u) {
        const float v = kv[kBase + d];
        kSq += v * v;
    }
    kSq = simd_sum(kSq);
    if (lane == 0u) partialK[sg] = kSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float totalK = 0.0f;
    for (uint i = 0u; i < 8u; i++) totalK += partialK[i];
    const float invK = 1.0f / sqrt(totalK / float(H) + eps);

    if (tid < 128u) {
        float g = 0.0f;
        for (uint d = tid; d < H; d += 128u) {
            const float kn = kv[kBase + d] * invK * (1.0f + normKey[wOff + d]);
            g += kn * queryNorm[wBase + d];
        }
        g = simd_sum(g);
        if (lane == 0u) partialKQ[sg] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float total = (partialKQ[0] + partialKQ[1] + partialKQ[2] + partialKQ[3]) * rsqrt(float(H));
        const float mag = sqrt(max(fabs(total), 1e-6f));
        const float gate = (total > 0.0f) ? mag : ((total < 0.0f) ? -mag : 0.0f);
        tgSig = 1.0f / (1.0f + exp(-gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sig = tgSig;

    float gvSq = 0.0f;
    for (uint d = tid; d < H; d += 256u) {
        const float gvd = sig * kv[vBase + d];
        gv[wBase + d] = gvd;
        gvSq += gvd * gvd;
    }
    gvSq = simd_sum(gvSq);
    if (lane == 0u) partialGV[sg] = gvSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float totalGV = 0.0f;
    for (uint i = 0u; i < 8u; i++) totalGV += partialGV[i];
    const float invGV = 1.0f / sqrt(totalGV / float(H) + eps);
    for (uint d = tid; d < H; d += 256u) {
        gvNorm[wBase + d] = gv[wBase + d] * invGV * (1.0f + normConv[wOff + d]);
    }
}

kernel void qwg_ple_conv(device const float* gv         [[buffer(0)]],
                         device const float* gvNorm     [[buffer(1)]],
                         device const float* convWeight [[buffer(2)]],
                         device float*       out        [[buffer(3)]],
                         constant uint&      W          [[buffer(4)]],
                         constant uint&      K          [[buffer(5)]],
                         constant uint&      dilation   [[buffer(6)]],
                         constant uint&      T          [[buffer(7)]],
                         device const float* hist       [[buffer(8)]],
                         constant uint&      HL         [[buffer(9)]],
                         device const float* residual   [[buffer(10)]],
                         constant uint&      hasRes     [[buffer(11)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= W || t >= T) return;
    float acc = 0.0f;
    for (uint j = 0u; j < K; j++) {
        const int src = int(t) - int((K - 1u - j) * dilation);
        if (src >= 0) acc += convWeight[(size_t)c * K + j] * gvNorm[(size_t)uint(src) * W + c];
        else if (HL > 0u) acc += convWeight[(size_t)c * K + j] * hist[(size_t)(int(HL) + src) * W + c];
    }
    const float convOut = gv[(size_t)t * W + c] + acc / (1.0f + exp(-acc));
    out[(size_t)t * W + c] = hasRes ? (residual[(size_t)t * W + c] + convOut) : convOut;
}

// History roll: newHist[i] = the last HL rows of [oldHist (HL rows) | gvNorm (T rows)]. dispatch: (W, HL) threads.
kernel void qwg_ple_hist(device const float* oldHist [[buffer(0)]],
                         device const float* gvNorm  [[buffer(1)]],
                         device float*       newHist [[buffer(2)]],
                         constant uint&      W       [[buffer(3)]],
                         constant uint&      HL      [[buffer(4)]],
                         constant uint&      T       [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, i = gid.y;
    if (c >= W || i >= HL) return;
    const uint k = i + T;
    newHist[(size_t)i * W + c] = k < HL ? oldHist[(size_t)k * W + c] : gvNorm[(size_t)(k - HL) * W + c];
}

// inject buffers: 0 base f32 [T, C, H] | 1 src f32 [T, H] | 2 inj f32 [T, C] | 3 out f32 [T, C, H] | 4 H | 5 C
// Residual injection of a sub-layer output into the HC streams: out[t, c, :] = base[t, c, :] + src[t, :] * inj[t, c].
// dispatch: (C * H, T) threads.
kernel void qwg_inject(device const float* base [[buffer(0)]],
                       device const float* src  [[buffer(1)]],
                       device const float* inj  [[buffer(2)]],
                       device float*       out  [[buffer(3)]],
                       constant uint&      H    [[buffer(4)]],
                       constant uint&      C    [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x / H, d = gid.x - c * H, t = gid.y;
    if (c >= C) return;
    const size_t o = ((size_t)t * C + c) * H + d;
    out[o] = base[o] + src[(size_t)t * H + d] * inj[(size_t)t * C + c];
}

// VQ2_G32 zero-pread PLE gather: loads 256x2 f16 e-means codebook at ngramFile+1024 into threadgroup LUT,
// then decodes sector-packed 18 B/group rows into targetFloatOffsets within outBase.
kernel void qwg_ple_vq2_gather(device const uchar* ngramFile          [[buffer(0)]],
                               device const uint*  rowIds             [[buffer(1)]],
                               device const ulong* targetFloatOffsets [[buffer(2)]],
                               device float*       outBase            [[buffer(3)]],
                               constant uint&      headDim            [[buffer(4)]],
                               constant uint&      rowsPerTarget      [[buffer(5)]],
                               constant uint&      numTargets         [[buffer(6)]],
                               constant uint&      dataOffset         [[buffer(7)]],
                               constant uint&      rowsPerSector      [[buffer(8)]],
                               constant uint&      rowBytes           [[buffer(9)]],
                               uint tg   [[threadgroup_position_in_grid]],
                               uint nTGs [[threadgroups_per_grid]],
                               uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float2 lut[256];
    device const ushort* cb16 = reinterpret_cast<device const ushort*>(ngramFile + 1024u);
    lut[tid] = float2(as_type<half>(cb16[tid * 2u]), as_type<half>(cb16[tid * 2u + 1u]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint groupsPerRow = headDim >> 5u;
    const uint totalGroups = numTargets * rowsPerTarget * groupsPerRow;
    const uint team = tid >> 4u, lane = tid & 15u;
    const uint gridTeams = nTGs << 4u;
    for (uint item = (tg << 4u) + team; item < totalGroups; item += gridTeams) {
        const uint rLinear = item / groupsPerRow;
        const uint g = item - rLinear * groupsPerRow;
        const uint tIdx = rLinear / rowsPerTarget;
        const uint rIdx = rLinear - tIdx * rowsPerTarget;
        const ulong rowId = ulong(rowIds[rLinear]);
        const ulong sec = rowId / ulong(rowsPerSector);
        const ulong inSec = rowId - sec * ulong(rowsPerSector);
        const ulong byteOff = ulong(dataOffset) + sec * 4096ul + inSec * ulong(rowBytes) + ulong(g) * 18ul;
        device const uchar* grp = ngramFile + byteOff;
        const uint code = uint(grp[lane]);
        const ushort sBits = ushort(grp[16]) | (ushort(grp[17]) << 8u);
        const float scale = float(as_type<half>(sBits));
        const float2 val = lut[code] * scale;
        const ulong outIdx = targetFloatOffsets[tIdx] + ulong(rIdx) * ulong(headDim)
                           + ulong(g) * 32ul + ulong(lane) * 2ul;
        outBase[outIdx] = val.x;
        outBase[outIdx + 1ul] = val.y;
    }
}

// VQ4_G32 zero-pread PLE gather: loads 256x4 f16 e-means codebook at ngramFile+1024 into threadgroup LUT,
// then decodes sector-packed 10 B/group rows into targetFloatOffsets within outBase.
kernel void qwg_ple_vq4_gather(device const uchar* ngramFile          [[buffer(0)]],
                               device const uint*  rowIds             [[buffer(1)]],
                               device const ulong* targetFloatOffsets [[buffer(2)]],
                               device float*       outBase            [[buffer(3)]],
                               constant uint&      headDim            [[buffer(4)]],
                               constant uint&      rowsPerTarget      [[buffer(5)]],
                               constant uint&      numTargets         [[buffer(6)]],
                               constant uint&      dataOffset         [[buffer(7)]],
                               constant uint&      rowsPerSector      [[buffer(8)]],
                               constant uint&      rowBytes           [[buffer(9)]],
                               uint tg   [[threadgroup_position_in_grid]],
                               uint nTGs [[threadgroups_per_grid]],
                               uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float4 lut[256];
    device const ushort4* cb16 = reinterpret_cast<device const ushort4*>(ngramFile + 1024u);
    lut[tid] = float4(as_type<half4>(cb16[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint groupsPerRow = headDim >> 5u;
    const uint totalGroups = numTargets * rowsPerTarget * groupsPerRow;
    const uint team = tid >> 3u, lane = tid & 7u;
    const uint gridTeams = nTGs << 5u;
    for (uint item = (tg << 5u) + team; item < totalGroups; item += gridTeams) {
        const uint rLinear = item / groupsPerRow;
        const uint g = item - rLinear * groupsPerRow;
        const uint tIdx = rLinear / rowsPerTarget;
        const uint rIdx = rLinear - tIdx * rowsPerTarget;
        const ulong rowId = ulong(rowIds[rLinear]);
        const ulong sec = rowId / ulong(rowsPerSector);
        const ulong inSec = rowId - sec * ulong(rowsPerSector);
        const ulong byteOff = ulong(dataOffset) + sec * 4096ul + inSec * ulong(rowBytes) + ulong(g) * 10ul;
        device const uchar* grp = ngramFile + byteOff;
        const uint code = uint(grp[lane]);
        const ushort sBits = ushort(grp[8]) | (ushort(grp[9]) << 8u);
        const float scale = float(as_type<half>(sBits));
        const float4 val = lut[code] * scale;
        const ulong outIdx = targetFloatOffsets[tIdx] + ulong(rIdx) * ulong(headDim)
                           + ulong(g) * 32ul + ulong(lane) * 4ul;
        outBase[outIdx] = val.x;
        outBase[outIdx + 1ul] = val.y;
        outBase[outIdx + 2ul] = val.z;
        outBase[outIdx + 3ul] = val.w;
    }
}
