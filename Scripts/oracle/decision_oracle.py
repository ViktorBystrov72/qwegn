#!/usr/bin/env python3
"""PyTorch-to-Metal parity fixture generator for all 3 M3 decision archetypes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F


def _save_f16(path: Path, t: torch.Tensor) -> str:
    arr = t.detach().to(torch.float16).contiguous()
    path.write_bytes(arr.view(torch.uint16).cpu().numpy().tobytes())
    return path.name


def _save_bf16(path: Path, t: torch.Tensor) -> str:
    arr = t.detach().to(torch.bfloat16).contiguous()
    path.write_bytes(arr.view(torch.uint16).cpu().numpy().tobytes())
    return path.name


def _save_f32(path: Path, t: torch.Tensor) -> str:
    arr = t.detach().to(torch.float32).contiguous()
    path.write_bytes(arr.cpu().numpy().tobytes())
    return path.name


def _f16_seam(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).float()


def _apply_rope_zero_tail(x: torch.Tensor, theta: float, partial_factor: float, zero_tail: bool) -> torch.Tensor:
    # x: [T, H, D]
    seq_len, _, head_dim = x.shape
    raw_rot = int(round(head_dim * partial_factor))
    rot_dims = max(2, min(head_dim, raw_rot - (raw_rot % 2)))
    pairs = rot_dims // 2
    exponents = torch.arange(pairs, dtype=torch.float32) * 2.0 / float(rot_dims)
    inv_freq = 1.0 / (theta ** exponents)
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    angles = pos * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    xf = x.float()
    x0 = xf[..., :pairs]
    x1 = xf[..., pairs:rot_dims]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    tail = torch.zeros_like(xf[..., rot_dims:]) if zero_tail else xf[..., rot_dims:]
    return _f16_seam(torch.cat([y0, y1, tail], dim=-1))


def _topology_mask(
    seq_len: int,
    topology: str,
    group_ids: list[int],
    state_end: int,
    media_prefix_len: int,
    isolate_media: bool,
) -> torch.Tensor:
    allow = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for q in range(seq_len):
        qg = group_ids[q]
        if qg < 0:
            continue
        for k in range(seq_len):
            kg = group_ids[k]
            if kg < 0:
                continue
            if topology == "causal":
                allow[q, k] = k <= q
            elif topology == "prefixBidirectional":
                allow[q, k] = (q < state_end and k < state_end) or (k <= q)
            elif topology == "packedSet":
                allow[q, k] = (qg == 0) or (kg == 0) or (kg == qg)
            elif topology == "bidirectionalEncoder":
                allow[q, k] = ((q < media_prefix_len) == (k < media_prefix_len)) if (isolate_media and media_prefix_len > 0) else True
            else:
                allow[q, k] = True
    return allow


def _run_shortconv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, *, centered: bool, media_prefix_len: int) -> torch.Tensor:
    seq_len, channels = x.shape
    k = w.shape[1]
    out = torch.zeros(seq_len, channels, dtype=torch.float32)
    xf, wf = x.float(), w.float()
    half_k = (k - 1) // 2
    for t in range(seq_len):
        acc = b.float().clone() if b is not None else torch.zeros(channels, dtype=torch.float32)
        if centered:
            in_media = media_prefix_len > 0 and t < media_prefix_len
            win_lo = 0 if in_media else media_prefix_len
            win_hi = media_prefix_len if in_media else seq_len
            for j in range(k):
                src_t = t + j - half_k
                if win_lo <= src_t < win_hi:
                    acc += xf[src_t] * wf[:, j]
        else:
            for j in range(k):
                src_t = t - (k - 1 - j)
                if src_t >= 0:
                    acc += xf[src_t] * wf[:, j]
        out[t] = acc
    return _f16_seam(out)


def _rms_norm_f32(x: torch.Tensor, w: torch.Tensor, eps: float, unit_offset: bool = False) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    scale = w.float() + (1.0 if unit_offset else 0.0)
    return xf * rstd * scale


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, unit_offset: bool = False) -> torch.Tensor:
    return _f16_seam(_rms_norm_f32(x, w, eps, unit_offset=unit_offset))


def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, eps: float) -> torch.Tensor:
    return _f16_seam(F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float() if b is not None else None, eps=eps))


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None, act: str = "none") -> torch.Tensor:
    y = F.linear(x.float(), w.float(), b.float() if b is not None else None)
    if act == "gelu":
        return F.gelu(y)
    if act == "silu":
        return F.silu(y)
    return y


def _run_attention(
    normed: torch.Tensor,
    layer: dict,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    topology: str,
    group_ids: list[int],
    state_end: int,
    media_prefix_len: int,
    isolate_media: bool,
    rope_theta: float | None,
    partial_factor: float,
    zero_tail: bool,
    has_gate: bool,
) -> torch.Tensor:
    seq_len = normed.shape[0]
    q = _f16_seam(_linear(normed, layer["qW"], layer.get("qB"))).view(seq_len, q_heads, head_dim)
    k = _f16_seam(_linear(normed, layer["kW"], layer.get("kB"))).view(seq_len, kv_heads, head_dim)
    v = _f16_seam(_linear(normed, layer["vW"], layer.get("vB"))).view(seq_len, kv_heads, head_dim)
    if rope_theta is not None:
        q = _apply_rope_zero_tail(q, rope_theta, partial_factor, zero_tail)
        k = _apply_rope_zero_tail(k, rope_theta, partial_factor, zero_tail)
    gqa = q_heads // kv_heads
    if gqa > 1:
        k = k.repeat_interleave(gqa, dim=1)
        v = v.repeat_interleave(gqa, dim=1)
    mask = _topology_mask(seq_len, topology, group_ids, state_end, media_prefix_len, isolate_media)
    attn_out = torch.zeros(seq_len, q_heads, head_dim, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dim)
    for q_idx in range(seq_len):
        valid_k = mask[q_idx]
        if not valid_k.any():
            continue
        scores = torch.einsum("hd,khd->hk", q[q_idx], k) * scale
        scores = scores.masked_fill(~valid_k.unsqueeze(0), -1e9)
        probs = F.softmax(scores, dim=-1)
        attn_out[q_idx] = torch.einsum("hk,khd->hd", probs, v)
    attn_out = _f16_seam(attn_out)
    if has_gate:
        g = torch.sigmoid(_f16_seam(_linear(normed, layer["gateW"], layer.get("gateB")))).unsqueeze(-1)
        attn_out = _f16_seam(attn_out * g)
    return _linear(attn_out.reshape(seq_len, q_heads * head_dim), layer["oW"], layer.get("oB"))


def _save_layer_weights(out_dir: Path, prefix: str, layer: dict) -> dict:
    meta: dict = {}
    for k, val in layer.items():
        if val is None or isinstance(val, (str, int, float, bool)):
            meta[k] = val
        elif isinstance(val, torch.Tensor):
            meta[k] = _save_f16(out_dir / f"{prefix}_{k}.f16.bin", val)
    return meta


def _eval_hybrid_or_encoder(
    hidden_in: torch.Tensor,
    layers: list[dict],
    *,
    group_ids: list[int],
    state_end: int,
    media_prefix_len: int,
    final_norm: dict | None,
) -> torch.Tensor:
    x = hidden_in.float().clone()
    for layer in layers:
        x_in = _f16_seam(x)
        if layer["preBlockKind"] == "rms":
            normed = _rms_norm(x_in, layer["preBlockW"], layer["eps"])
        else:
            normed = _layer_norm(x_in, layer["preBlockW"], layer.get("preBlockB"), layer["eps"])
        if layer["blockKind"] in ("shortConvCausal", "shortConvCentered"):
            cin = _f16_seam(_linear(normed, layer["inProjW"], layer.get("inProjB"))) if "inProjW" in layer else normed
            conv_out = _run_shortconv(
                cin, layer["convW"], layer.get("convB"),
                centered=(layer["blockKind"] == "shortConvCentered"),
                media_prefix_len=media_prefix_len,
            )
            branch = _linear(conv_out, layer["outProjW"], layer.get("outProjB"))
        else:
            branch = _run_attention(
                normed, layer,
                q_heads=layer["qHeads"], kv_heads=layer["kvHeads"], head_dim=layer["headDim"],
                topology=layer["topology"], group_ids=group_ids, state_end=state_end,
                media_prefix_len=media_prefix_len, isolate_media=layer["isolateMedia"],
                rope_theta=layer.get("ropeTheta"), partial_factor=layer.get("partialFactor", 1.0),
                zero_tail=layer.get("zeroTail", False), has_gate=layer.get("hasGate", False),
            )
        if layer["postBlockKind"] == "rms":
            x = _f16_seam(x + _rms_norm_f32(_f16_seam(branch), layer["postBlockW"], layer["eps"], layer.get("unitOffset", False)))
        else:
            x = x + branch
            x16 = _f16_seam(x)
        if layer["postBlockKind"] == "rms":
            x16 = x

        if layer["preFFNKind"] == "rms":
            fnorm = _rms_norm(x16, layer["preFFNW"], layer["eps"])
        else:
            fnorm = _layer_norm(x16, layer["preFFNW"], layer.get("preFFNB"), layer["eps"])
        fmid = _linear(fnorm, layer["ffnUpW"], layer.get("ffnUpB"), act=layer["ffnAct"])
        fbranch = _linear(fmid, layer["ffnDownW"], layer.get("ffnDownB"))
        if layer["postFFNKind"] == "rms":
            x = _f16_seam(x + _rms_norm_f32(_f16_seam(fbranch), layer["postFFNW"], layer["eps"], layer.get("unitOffset", False)))
        else:
            x = x + fbranch

    if final_norm is not None:
        xf_in = _f16_seam(x)
        if final_norm["kind"] == "rms":
            x = _rms_norm(xf_in, final_norm["w"], final_norm["eps"])
        else:
            x = _layer_norm(xf_in, final_norm["w"], final_norm.get("b"), final_norm["eps"])
    return x


def _make_rms_weight(h: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.ones(h, dtype=torch.float16) * scale + torch.randn(h, dtype=torch.float16) * 0.05)


def _rand_w(out_f: int, in_f: int, scale: float = 0.08) -> torch.Tensor:
    return (torch.randn(out_f, in_f) * scale).to(torch.float16)


def _build_archetype1(out_dir: Path) -> dict:
    torch.manual_seed(101)
    h, ffn, qh, kvh, hd, seq_len, state_end = 64, 128, 4, 2, 16, 10, 4
    group_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    hidden_in = (torch.randn(seq_len, h) * 0.2).to(torch.float16)
    hidden_qb_pert = hidden_in.clone()
    hidden_qb_pert[7:] += 0.5
    hidden_qa_pert = hidden_in.clone()
    hidden_qa_pert[4:7] += 0.5

    l0 = {
        "blockKind": "shortConvCausal", "kernelSize": 7, "channels": h,
        "preBlockKind": "rms", "postBlockKind": "rms", "preFFNKind": "rms", "postFFNKind": "rms",
        "eps": 1e-6, "ffnAct": "silu", "unitOffset": False,
        "preBlockW": _make_rms_weight(h), "postBlockW": _make_rms_weight(h),
        "preFFNW": _make_rms_weight(h), "postFFNW": _make_rms_weight(h),
        "inProjW": _rand_w(h, h), "convW": (torch.randn(h, 7) * 0.15).to(torch.float16),
        "convB": (torch.randn(h) * 0.02).to(torch.float16), "outProjW": _rand_w(h, h),
        "ffnUpW": _rand_w(ffn, h), "ffnDownW": _rand_w(h, ffn),
    }
    l1 = {
        "blockKind": "attention", "topology": "packedSet", "isolateMedia": False,
        "qHeads": qh, "kvHeads": kvh, "headDim": hd,
        "ropeTheta": 1_000_000.0, "partialFactor": 0.5, "zeroTail": True, "hasGate": True,
        "preBlockKind": "rms", "postBlockKind": "rms", "preFFNKind": "rms", "postFFNKind": "rms",
        "eps": 1e-6, "ffnAct": "silu", "unitOffset": False,
        "preBlockW": _make_rms_weight(h), "postBlockW": _make_rms_weight(h),
        "preFFNW": _make_rms_weight(h), "postFFNW": _make_rms_weight(h),
        "qW": _rand_w(qh * hd, h), "kW": _rand_w(kvh * hd, h), "vW": _rand_w(kvh * hd, h),
        "oW": _rand_w(h, qh * hd), "gateW": _rand_w(qh, h), "gateB": (torch.randn(qh) * 0.1).to(torch.float16),
        "ffnUpW": _rand_w(ffn, h), "ffnDownW": _rand_w(h, ffn),
    }
    fn = {"kind": "rms", "w": _make_rms_weight(h), "eps": 1e-6}

    final_h = _eval_hybrid_or_encoder(hidden_in, [l0, l1], group_ids=group_ids, state_end=state_end, media_prefix_len=0, final_norm=fn)
    pert_qb_h = _eval_hybrid_or_encoder(hidden_qb_pert, [l0, l1], group_ids=group_ids, state_end=state_end, media_prefix_len=0, final_norm=fn)
    attn_only_h = _eval_hybrid_or_encoder(hidden_in, [l1], group_ids=group_ids, state_end=state_end, media_prefix_len=0, final_norm=None)
    attn_pert_qa_h = _eval_hybrid_or_encoder(hidden_qa_pert, [l1], group_ids=group_ids, state_end=state_end, media_prefix_len=0, final_norm=None)

    vocab_size, cand_ids = 512, [3, 17, 42, 128, 255, 400]
    lm_head_bf16 = (torch.randn(vocab_size, h) * 0.1).to(torch.bfloat16)
    slots_h = final_h[[6, 9]].to(torch.float16).float()
    full_logits = F.linear(slots_h, lm_head_bf16.float())
    sparse_logits = full_logits[:, cand_ids]

    q_start = (torch.randn(h) * 0.15).to(torch.float16)
    q_end = (torch.randn(h) * 0.15).to(torch.float16)
    state_h = final_h[:state_end].to(torch.float16).float()
    ptr_start = (state_h @ q_start.float()) / math.sqrt(h)
    ptr_end = (state_h @ q_end.float()) / math.sqrt(h)

    return {
        "hidden": h, "ffnDim": ffn, "seqLen": seq_len, "stateEndPos": state_end, "groupIds": group_ids,
        "vocabSize": vocab_size, "candidateIds": cand_ids, "slotRows": [6, 9],
        "hiddenIn": _save_f16(out_dir / "a1_hidden_in.f16.bin", hidden_in),
        "hiddenQBPert": _save_f16(out_dir / "a1_hidden_qb_pert.f16.bin", hidden_qb_pert),
        "hiddenQAPert": _save_f16(out_dir / "a1_hidden_qa_pert.f16.bin", hidden_qa_pert),
        "lmHeadBF16": _save_bf16(out_dir / "a1_lm_head.bf16.bin", lm_head_bf16),
        "qStart": _save_f16(out_dir / "a1_q_start.f16.bin", q_start),
        "qEnd": _save_f16(out_dir / "a1_q_end.f16.bin", q_end),
        "finalNormW": _save_f16(out_dir / "a1_fn_w.f16.bin", fn["w"]),
        "layers": [_save_layer_weights(out_dir, "a1_l0", l0), _save_layer_weights(out_dir, "a1_l1", l1)],
        "refFinalH": _save_f32(out_dir / "a1_ref_final_h.f32.bin", final_h),
        "refPertQB_Row6": _save_f32(out_dir / "a1_ref_pert_qb_r6.f32.bin", pert_qb_h[6]),
        "refAttnOnly": _save_f32(out_dir / "a1_ref_attn_only.f32.bin", attn_only_h),
        "refAttnPertQA_Row9": _save_f32(out_dir / "a1_ref_attn_pert_qa_r9.f32.bin", attn_pert_qa_h[9]),
        "refFullLogits": _save_f32(out_dir / "a1_ref_full_logits.f32.bin", full_logits),
        "refSparseLogits": _save_f32(out_dir / "a1_ref_sparse_logits.f32.bin", sparse_logits),
        "refPtrStart": _save_f32(out_dir / "a1_ref_ptr_start.f32.bin", ptr_start),
        "refPtrEnd": _save_f32(out_dir / "a1_ref_ptr_end.f32.bin", ptr_end),
    }


def _build_archetype2(out_dir: Path) -> dict:
    torch.manual_seed(202)
    h, ffn, qh, kvh, hd, seq_len, media_len = 64, 128, 4, 4, 16, 12, 5
    group_ids = [0] * seq_len
    hidden_in = (torch.randn(seq_len, h) * 0.2).to(torch.float16)
    hidden_media_pert = hidden_in.clone()
    hidden_media_pert[:media_len] += 0.5
    hidden_text_pert = hidden_in.clone()
    hidden_text_pert[media_len:] += 0.5

    def ln_pair():
        return _make_rms_weight(h), (torch.randn(h) * 0.02).to(torch.float16)

    l0_bw, l0_bb = ln_pair()
    l0_fw, l0_fb = ln_pair()
    l0 = {
        "blockKind": "shortConvCentered", "kernelSize": 3, "channels": h,
        "preBlockKind": "layerNorm", "postBlockKind": "none", "preFFNKind": "layerNorm", "postFFNKind": "none",
        "eps": 1e-5, "ffnAct": "gelu",
        "preBlockW": l0_bw, "preBlockB": l0_bb, "preFFNW": l0_fw, "preFFNB": l0_fb,
        "convW": (torch.randn(h, 3) * 0.15).to(torch.float16), "convB": (torch.randn(h) * 0.02).to(torch.float16),
        "outProjW": _rand_w(h, h), "outProjB": (torch.randn(h) * 0.01).to(torch.float16),
        "ffnUpW": _rand_w(ffn, h), "ffnUpB": (torch.randn(ffn) * 0.01).to(torch.float16),
        "ffnDownW": _rand_w(h, ffn), "ffnDownB": (torch.randn(h) * 0.01).to(torch.float16),
    }
    l1_bw, l1_bb = ln_pair()
    l1_fw, l1_fb = ln_pair()
    l1 = {
        "blockKind": "attention", "topology": "bidirectionalEncoder", "isolateMedia": True,
        "qHeads": qh, "kvHeads": kvh, "headDim": hd,
        "ropeTheta": 10_000.0, "partialFactor": 1.0, "zeroTail": False, "hasGate": False,
        "preBlockKind": "layerNorm", "postBlockKind": "none", "preFFNKind": "layerNorm", "postFFNKind": "none",
        "eps": 1e-5, "ffnAct": "gelu",
        "preBlockW": l1_bw, "preBlockB": l1_bb, "preFFNW": l1_fw, "preFFNB": l1_fb,
        "qW": _rand_w(qh * hd, h), "kW": _rand_w(kvh * hd, h), "vW": _rand_w(kvh * hd, h), "oW": _rand_w(h, qh * hd),
        "ffnUpW": _rand_w(ffn, h), "ffnDownW": _rand_w(h, ffn),
    }
    fn_w, fn_b = ln_pair()
    fn = {"kind": "layerNorm", "w": fn_w, "b": fn_b, "eps": 1e-5}

    final_h = _eval_hybrid_or_encoder(hidden_in, [l0, l1], group_ids=group_ids, state_end=0, media_prefix_len=media_len, final_norm=fn)
    pert_media_h = _eval_hybrid_or_encoder(hidden_media_pert, [l0, l1], group_ids=group_ids, state_end=0, media_prefix_len=media_len, final_norm=fn)
    pert_text_h = _eval_hybrid_or_encoder(hidden_text_pert, [l0, l1], group_ids=group_ids, state_end=0, media_prefix_len=media_len, final_norm=fn)

    marker_pos, mid_dim, out_dim = [7, 9, 11], 32, 1
    type_emb = (torch.randn(h) * 0.05).to(torch.float16)
    c1_w1, c1_b1 = _rand_w(mid_dim, h, 0.12), (torch.randn(mid_dim) * 0.02).to(torch.float16)
    c1_w2, c1_b2 = _rand_w(out_dim, mid_dim, 0.12), (torch.randn(out_dim) * 0.02).to(torch.float16)
    c2_w1, c2_b1 = _rand_w(mid_dim, h, 0.12), (torch.randn(mid_dim) * 0.02).to(torch.float16)
    c2_w2, c2_b2 = _rand_w(out_dim, mid_dim, 0.12), (torch.randn(out_dim) * 0.02).to(torch.float16)

    gathered = final_h[marker_pos].to(torch.float16).float() + type_emb.float().unsqueeze(0)
    chain1_ref = _linear(_linear(gathered, c1_w1, c1_b1, act="gelu"), c1_w2, c1_b2)
    chain2_ref = _linear(_linear(gathered, c2_w1, c2_b1, act="gelu"), c2_w2, c2_b2)

    return {
        "hidden": h, "ffnDim": ffn, "seqLen": seq_len, "mediaPrefixLen": media_len, "groupIds": group_ids,
        "markerPositions": marker_pos, "midDim": mid_dim, "outDim": out_dim,
        "hiddenIn": _save_f16(out_dir / "a2_hidden_in.f16.bin", hidden_in),
        "hiddenMediaPert": _save_f16(out_dir / "a2_hidden_media_pert.f16.bin", hidden_media_pert),
        "hiddenTextPert": _save_f16(out_dir / "a2_hidden_text_pert.f16.bin", hidden_text_pert),
        "finalNormW": _save_f16(out_dir / "a2_fn_w.f16.bin", fn_w),
        "finalNormB": _save_f16(out_dir / "a2_fn_b.f16.bin", fn_b),
        "typeEmb": _save_f16(out_dir / "a2_type_emb.f16.bin", type_emb),
        "c1W1": _save_f16(out_dir / "a2_c1_w1.f16.bin", c1_w1), "c1B1": _save_f16(out_dir / "a2_c1_b1.f16.bin", c1_b1),
        "c1W2": _save_f16(out_dir / "a2_c1_w2.f16.bin", c1_w2), "c1B2": _save_f16(out_dir / "a2_c1_b2.f16.bin", c1_b2),
        "c2W1": _save_f16(out_dir / "a2_c2_w1.f16.bin", c2_w1), "c2B1": _save_f16(out_dir / "a2_c2_b1.f16.bin", c2_b1),
        "c2W2": _save_f16(out_dir / "a2_c2_w2.f16.bin", c2_w2), "c2B2": _save_f16(out_dir / "a2_c2_b2.f16.bin", c2_b2),
        "layers": [_save_layer_weights(out_dir, "a2_l0", l0), _save_layer_weights(out_dir, "a2_l1", l1)],
        "refFinalH": _save_f32(out_dir / "a2_ref_final_h.f32.bin", final_h),
        "refPertMediaTextPart": _save_f32(out_dir / "a2_ref_pert_media_text.f32.bin", pert_media_h[media_len:]),
        "refPertTextMediaPart": _save_f32(out_dir / "a2_ref_pert_text_media.f32.bin", pert_text_h[:media_len]),
        "refChain1": _save_f32(out_dir / "a2_ref_chain1.f32.bin", chain1_ref.flatten()),
        "refChain2": _save_f32(out_dir / "a2_ref_chain2.f32.bin", chain2_ref.flatten()),
    }


def _build_archetype3(out_dir: Path) -> dict:
    torch.manual_seed(303)
    h, ffn, qh, kvh, hd, seq_len, valid_len = 64, 128, 4, 4, 16, 9, 7
    group_ids = [0] * valid_len + [-1] * (seq_len - valid_len)
    hidden_in = (torch.randn(seq_len, h) * 0.2).to(torch.float16)

    def make_enc_layer(prefix: str):
        bw, bb = _make_rms_weight(h), (torch.randn(h) * 0.02).to(torch.float16)
        fw, fb = _make_rms_weight(h), (torch.randn(h) * 0.02).to(torch.float16)
        layer = {
            "blockKind": "attention", "topology": "bidirectionalEncoder", "isolateMedia": False,
            "qHeads": qh, "kvHeads": kvh, "headDim": hd,
            "ropeTheta": 10_000.0, "partialFactor": 1.0, "zeroTail": False, "hasGate": False,
            "preBlockKind": "layerNorm", "postBlockKind": "none", "preFFNKind": "layerNorm", "postFFNKind": "none",
            "eps": 1e-5, "ffnAct": "gelu",
            "preBlockW": bw, "preBlockB": bb, "preFFNW": fw, "preFFNB": fb,
            "qW": _rand_w(qh * hd, h), "kW": _rand_w(kvh * hd, h), "vW": _rand_w(kvh * hd, h), "oW": _rand_w(h, qh * hd),
            "ffnUpW": _rand_w(ffn, h), "ffnDownW": _rand_w(h, ffn),
        }
        return layer, prefix

    l0, _ = make_enc_layer("a3_l0")
    l1, _ = make_enc_layer("a3_l1")
    fn_w, fn_b = _make_rms_weight(h), (torch.randn(h) * 0.02).to(torch.float16)
    fn = {"kind": "layerNorm", "w": fn_w, "b": fn_b, "eps": 1e-5}
    final_h = _eval_hybrid_or_encoder(hidden_in, [l0, l1], group_ids=group_ids, state_end=0, media_prefix_len=0, final_norm=fn)

    mid_dim, out_dim = 32, 4
    w1, b1 = _rand_w(mid_dim, h, 0.12), (torch.randn(mid_dim) * 0.02).to(torch.float16)
    w2, b2 = _rand_w(out_dim, mid_dim, 0.12), (torch.randn(out_dim) * 0.02).to(torch.float16)

    fh16 = final_h.to(torch.float16).float()
    pools = {
        "firstToken": fh16[0].unsqueeze(0),
        "lastToken": fh16[valid_len - 1].unsqueeze(0),
        "meanNonPad": fh16[:valid_len].mean(dim=0, keepdim=True),
    }
    pool_refs = {
        k: _save_f32(out_dir / f"a3_ref_{k}.f32.bin", _linear(_linear(p, w1, b1, act="gelu"), w2, b2).flatten())
        for k, p in pools.items()
    }

    return {
        "hidden": h, "ffnDim": ffn, "seqLen": seq_len, "validLen": valid_len, "groupIds": group_ids,
        "midDim": mid_dim, "outDim": out_dim,
        "hiddenIn": _save_f16(out_dir / "a3_hidden_in.f16.bin", hidden_in),
        "finalNormW": _save_f16(out_dir / "a3_fn_w.f16.bin", fn_w),
        "finalNormB": _save_f16(out_dir / "a3_fn_b.f16.bin", fn_b),
        "w1": _save_f16(out_dir / "a3_w1.f16.bin", w1), "b1": _save_f16(out_dir / "a3_b1.f16.bin", b1),
        "w2": _save_f16(out_dir / "a3_w2.f16.bin", w2), "b2": _save_f16(out_dir / "a3_b2.f16.bin", b2),
        "layers": [_save_layer_weights(out_dir, "a3_l0", l0), _save_layer_weights(out_dir, "a3_l1", l1)],
        "refPoolLogits": pool_refs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "archetype1_packed_set_lfm3": _build_archetype1(args.out),
        "archetype2_bidir_marker_omni": _build_archetype2(args.out),
        "archetype3_pooled_reranker": _build_archetype3(args.out),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"generated M3 decision oracle fixtures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
