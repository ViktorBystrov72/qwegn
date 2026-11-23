#!/usr/bin/env python3
"""Gate-0 HQ / SHVQ vs MLX KL + Top-1 Fidelity Oracle for Inkling-2-Mini.

Stage 1: Per-tensor error & cosine similarity (HQ4/6/8 vs SHVQ3/4 + FWHT_32 NVFP4).
Stage 2: Exact 48-visit float32 forward pass scoring original HF teacher logits
         against --mlx-candidate and all 5 Qwegn recipes via score_teacher_logits.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from hq4 import GROUP as HQ_GROUP
from hq_blocks import resolve_group_size
from inkling2_adapter import (
    RECIPES,
    Inkling2Recipe,
    Inkling2Source,
    Inkling2Spec,
    dequant_bf16,
    maybe_force_cpu_quant,
    open_inkling2_shards,
    parse_inkling2_config,
)
from pack import Q4G_ALIGNMENT
from score_teacher_logits import score as score_kl
from shvq import any_block_bytes, dequant_any_blocks, parse_any_quant_spec, quant_any_blocks

STAGE1_SPECS = ("hq4_g32_m5", "hq4_g64", "hq6_g32", "hq8_g32", "shvq3_g64", "shvq4_g32", "shvq4_g64")
CALIBRATION_TEXT = (
    "def hadamard_variance_ratio(x: list[float]) -> float:\n"
    "    n = len(x)\n    mean = sum(x) / n\n    return sum((v - mean) ** 2 for v in x) / max(1, n - 1)\n\n"
    "Theorem (Parseval invariance): For w in R^32 and orthonormal H_32, ||H_32 w||_2 = ||w||_2.\n\n"
    "In an autoregressive looped mixture-of-experts transformer, a prelude prepares local context "
    "before the recurrent core iterates across shared message banks, routing each token sparsely "
    "to top-k experts before the final coda projects normalized states onto unpadded vocabulary logits."
)


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _quant_blocks_compact(w: torch.Tensor, quant_str: str, *, device: str = "cpu"):
    if quant_str in ("bf16", "verbatim_nvfp4"):
        return None
    bits, group, is_m5, is_shvq = parse_any_quant_spec(quant_str)
    orig_shape = tuple(w.shape)
    w2 = w.reshape(-1, orig_shape[-1]) if w.ndim == 3 else w
    if w2.ndim != 2 or (w2.shape[1] % HQ_GROUP) != 0:
        return None
    k = w2.shape[1]
    eff_g = resolve_group_size(k, group) if not is_shvq else math.gcd(k, group)
    with maybe_force_cpu_quant(device):
        blks, eff_g = quant_any_blocks(w2, bits=bits, group=eff_g, is_m5=is_m5, is_shvq=is_shvq)
    use_m5 = (not is_shvq) and (bits == 4 and eff_g == 32 and is_m5)
    return blks, orig_shape, (bits, eff_g, use_m5, is_shvq)


def _dequant_compact(packed) -> torch.Tensor:
    blks, orig_shape, (bits, eff_g, use_m5, is_shvq) = packed
    return torch.from_numpy(dequant_any_blocks(blks, bits=bits, group=eff_g, is_m5=use_m5, is_shvq=is_shvq)).reshape(orig_shape).to(torch.float32)


def _quant_dequant_f32(w: torch.Tensor, quant_str: str, *, device: str = "cpu") -> torch.Tensor:
    packed = _quant_blocks_compact(w, quant_str, device=device)
    return w.to(torch.float32) if packed is None else _dequant_compact(packed)


def _tensor_error_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    diff, ref64 = ref.to(torch.float64) - cand.to(torch.float64), ref.to(torch.float64)
    return {
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "rel_l2": float(torch.linalg.norm(diff) / (torch.linalg.norm(ref64) + 1e-12)),
        "max_abs": float(diff.abs().max()),
        "cosine": float(F.cosine_similarity(ref64.reshape(-1), cand.to(torch.float64).reshape(-1), dim=0)),
    }


def eval_per_tensor_quant_fidelity(src: Inkling2Source, *, device: str = "cpu") -> dict:
    """Measure Stage-1 per-tensor error & cosine similarity including NVFP4 re-quant."""
    spec = src.spec
    max_e = min(8, spec.num_experts)

    def exp_slice(layer: int, proj: str) -> torch.Tensor:
        w = spec.hidden_size if proj != "down_proj" else spec.moe_intermediate_size
        return src.load_expert_proj_f32(layer, proj, as_torch=True)[:max_e].reshape(-1, w)

    lm_head = dequant_bf16(src.get_global("lm_head.weight"))
    probes = {
        "nvfp4_l0_gate_proj": exp_slice(0, "gate_proj"),
        "nvfp4_l0_down_proj": exp_slice(0, "down_proj"),
        "mxfp8_coda_gate_proj": exp_slice(spec.coda_start, "gate_proj"),
        "bf16_l0_q_proj": dequant_bf16(src.get_layer(0, "self_attn.q_proj.weight")),
        "bf16_lm_head": lm_head[: min(4096, lm_head.shape[0])],
    }
    per_tensor: dict[str, dict] = {}
    for tname, ref in probes.items():
        rows = {}
        for qspec in STAGE1_SPECS:
            bits, group, is_m5, is_shvq = parse_any_quant_spec(qspec)
            eff_g = math.gcd(ref.shape[1], group)
            use_m5 = (not is_shvq) and (bits == 4 and eff_g == 32 and is_m5)
            bpw = any_block_bytes(bits, eff_g, is_m5=use_m5, is_shvq=is_shvq) * 8.0 / eff_g
            rows[qspec] = {"bpw": bpw, "eff_group": eff_g, **_tensor_error_metrics(ref, _quant_dequant_f32(ref, qspec, device=device))}
        per_tensor[tname] = rows
    nv_g = per_tensor["nvfp4_l0_gate_proj"]
    return {
        "tensors": per_tensor,
        "nvfp4_fwht32_requant_summary": {
            "shvq4_g32_vs_hq4_g32_m5_rel_l2_ratio": nv_g["shvq4_g32"]["rel_l2"] / max(nv_g["hq4_g32_m5"]["rel_l2"], 1e-12),
            "shvq4_g64_vs_hq4_g64_rel_l2_ratio": nv_g["shvq4_g64"]["rel_l2"] / max(nv_g["hq4_g64"]["rel_l2"], 1e-12),
            "shvq4_g32_cosine": nv_g["shvq4_g32"]["cosine"],
            "hq4_g32_m5_cosine": nv_g["hq4_g32_m5"]["cosine"],
            "shvq4_g64_cosine": nv_g["shvq4_g64"]["cosine"],
            "hq4_g64_cosine": nv_g["hq4_g64"]["cosine"],
        },
    }


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * weight


def _tanh_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.clamp(x, -15.0, 15.0))


def _sconv_static(x: torch.Tensor, weight: torch.Tensor, *, denom: float = 5.0) -> torch.Tensor:
    w = weight.reshape(-1, 4)
    xp = F.pad(x, (0, 0, 3, 0))
    acc = xp[:-3] * w[:, 0] + xp[1:-2] * w[:, 1] + xp[2:-1] * w[:, 2] + x * w[:, 3]
    return (acc + x) / math.sqrt(denom)


def _sconv_dynamic(x: torch.Tensor, static_w: torch.Tensor, dyn_w: torch.Tensor, n_kv: int, head_dim: int) -> torch.Tensor:
    t_len, kv_dim = x.shape
    dw = dyn_w.reshape(t_len, n_kv, 4).unsqueeze(2).expand(t_len, n_kv, head_dim, 4).reshape(t_len, kv_dim, 4)
    w = static_w.t().unsqueeze(0) + dw
    xp = F.pad(x, (0, 0, 3, 0))
    return (w[..., 0] * x + w[..., 1] * xp[2:-1] + w[..., 2] * xp[1:-2] + w[..., 3] * xp[:-3] + x) / 3.0


def _attn_res(blocks: list[torch.Tensor], partial: torch.Tensor, queries: torch.Tensor, *, normalize_values: bool) -> torch.Tensor:
    cands = torch.stack(blocks + [partial], dim=0)
    h = partial.shape[-1]
    inv_rms = torch.rsqrt(cands.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    alpha = torch.einsum("cth,qh->qtc", cands, queries.reshape(-1, h)) * inv_rms.squeeze(-1).t().unsqueeze(0) / math.sqrt(h)
    vals = (cands * inv_rms) if normalize_values else cands
    return torch.einsum("qtc,cth->qth", torch.softmax(alpha, dim=-1), vals)


def _msg_controller(x: torch.Tensor, msg: torch.Tensor, down_w: torch.Tensor, forget_w: torch.Tensor, gain_w: torch.Tensor, eps: float) -> torch.Tensor:
    inv_x = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    inv_m = torch.rsqrt(msg.pow(2).mean(dim=-1, keepdim=True) + eps)
    hid = torch.cat([x * inv_x, msg * inv_m], dim=-1) @ down_w.t()
    return (1.0 - _tanh_safe(hid @ forget_w.t())) * x + _tanh_safe(hid @ gain_w.t()) * msg


class _LayerWeightProvider:
    """Streams Inkling-2 layers from Inkling2Source, optionally quantizing to a recipe."""

    def __init__(self, src: Inkling2Source, recipe: Inkling2Recipe | None, *, device: str = "cpu"):
        self.src = src
        self.spec = src.spec
        self.recipe = recipe
        self.device = device
        self._core_cache: dict[int, dict[str, object]] = {}

    def get_head(self) -> torch.Tensor:
        w = dequant_bf16(self.src.get_global("lm_head.weight"))
        if self.recipe is not None:
            w = _quant_dequant_f32(w, self.recipe.head_quant, device=self.device)
        return w[: self.spec.unpadded_vocab_size]

    def _build_layer_raw(self, layer: int) -> dict[str, object]:
        spec = self.spec
        is_loop = spec.loop_core_start <= layer < spec.loop_core_end
        dense_q = "bf16" if self.recipe is None else (self.recipe.coda_dense_quant if layer >= spec.coda_start else self.recipe.trunk_quant)
        gu_q, dn_q = ("bf16", "bf16") if self.recipe is None else (
            (self.recipe.moe_coda_gate_up, self.recipe.moe_coda_down) if layer >= spec.coda_start else (self.recipe.moe_main_gate_up, self.recipe.moe_main_down)
        )

        def load_lin(key: str, q_str: str):
            t = dequant_bf16(self.src.get_layer(layer, key))
            if is_loop and q_str not in ("bf16", "verbatim_nvfp4"):
                packed = _quant_blocks_compact(t, q_str, device=self.device)
                if packed is not None:
                    return ("packed", packed)
            return _quant_dequant_f32(t, q_str, device=self.device)

        def load_exp(proj: str, q_str: str):
            t = self.src.load_expert_proj_f32(layer, proj, as_torch=True)
            if q_str == "verbatim_nvfp4":
                return t
            if is_loop and q_str != "bf16":
                packed = _quant_blocks_compact(t, q_str, device=self.device)
                if packed is not None:
                    return ("packed", packed)
            return _quant_dequant_f32(t, q_str, device=self.device)

        d: dict[str, object] = {
            "mlp_input_query": self.src.get_layer(layer, "mlp_input.query").to(torch.float32),
            "post_attn_norm": dequant_bf16(self.src.get_layer(layer, "post_attention_layernorm.weight")),
            "attn_sconv": dequant_bf16(self.src.get_layer(layer, "attn_sconv.conv.weight")).reshape(-1, 4),
            "mlp_sconv": dequant_bf16(self.src.get_layer(layer, "mlp_sconv.conv.weight")).reshape(-1, 4),
            "q_scale": dequant_bf16(self.src.get_layer(layer, "self_attn.q_scale")),
            "rel_scale": dequant_bf16(self.src.get_layer(layer, "self_attn.rel_scale")),
            "rel_proj": dequant_bf16(self.src.get_layer(layer, "self_attn.rel_proj")),
            "router_gate": dequant_bf16(self.src.get_layer(layer, "mlp.gate_weight")),
            "router_bias": self.src.get_layer(layer, "mlp.e_score_correction_bias").to(torch.float32),
            "gate_proj_e": load_exp("gate_proj", gu_q),
            "up_proj_e": load_exp("up_proj", gu_q),
            "down_proj_e": load_exp("down_proj", dn_q),
        }
        if layer > 0:
            d["attn_input_query"] = self.src.get_layer(layer, "attention_input.query").to(torch.float32)
        if spec.is_sliding(layer):
            d["input_norm"] = dequant_bf16(self.src.get_layer(layer, "input_layernorm.weight"))
            d["k_sconv"] = dequant_bf16(self.src.get_layer(layer, "self_attn.k_sconv.conv.weight")).reshape(-1, 4)
            d["v_sconv"] = dequant_bf16(self.src.get_layer(layer, "self_attn.v_sconv.conv.weight")).reshape(-1, 4)
        else:
            d["q_norm"] = dequant_bf16(self.src.get_layer(layer, "q_input_layernorm.weight"))
            d["k_norm"] = dequant_bf16(self.src.get_layer(layer, "k_input_layernorm.weight"))
            d["v_norm"] = dequant_bf16(self.src.get_layer(layer, "v_input_layernorm.weight"))
            d["k_sconv_static"] = dequant_bf16(self.src.get_layer(layer, "self_attn.k_sconv.static_weight"))
            d["v_sconv_static"] = dequant_bf16(self.src.get_layer(layer, "self_attn.v_sconv.static_weight"))
            d["k_sconv_proj"] = load_lin("self_attn.k_sconv.weight_proj.weight", dense_q)
            d["v_sconv_proj"] = load_lin("self_attn.v_sconv.weight_proj.weight", dense_q)
        for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "r_proj", "sink_proj"):
            d[p] = load_lin(f"self_attn.{p}.weight", dense_q)
        if self.src.has_layer(layer, "self_attn.q_proj.bias"):
            d["q_bias"] = dequant_bf16(self.src.get_layer(layer, "self_attn.q_proj.bias"))
        if self.src.has_layer(layer, "self_attn.o_proj.bias"):
            d["o_bias"] = dequant_bf16(self.src.get_layer(layer, "self_attn.o_proj.bias"))
        if is_loop:
            d["controllers"] = {
                cn: (load_lin(f"controllers.{cn}.down_weight", "bf16"), load_lin(f"controllers.{cn}.forget_weight", "bf16"), load_lin(f"controllers.{cn}.gain_weight", "bf16"))
                for cn in ("attn", "k", "v", "mlp")
            }
        return d

    def get_layer(self, layer: int) -> dict[str, torch.Tensor]:
        if self.spec.loop_core_start <= layer < self.spec.loop_core_end:
            if layer not in self._core_cache:
                self._core_cache[layer] = self._build_layer_raw(layer)
            raw = self._core_cache[layer]
        else:
            raw = self._build_layer_raw(layer)

        def mat(v):
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "packed":
                return _dequant_compact(v[1])
            if isinstance(v, dict):
                return {k: mat(x) for k, x in v.items()}
            return tuple(mat(x) for x in v) if isinstance(v, tuple) else v

        return {k: mat(v) for k, v in raw.items()}


def forward_inkling2_48_visits(spec: Inkling2Spec, provider: _LayerWeightProvider, token_ids: torch.Tensor) -> torch.Tensor:
    """Exact 48-visit Inkling-2 PyTorch forward pass in float32."""
    with torch.no_grad():
        src = provider.src
        emb_tbl = dequant_bf16(src.get_global("embed_tokens.weight"))
        emb_norm = dequant_bf16(src.get_global("embed_norm.weight"))
        final_norm = dequant_bf16(src.get_global("norm.weight"))
        final_query = src.get_global("final_input.query").to(torch.float32)
        core_len = spec.loop_core_end - spec.loop_core_start
        msg_queries = src.get_global("message_queries").to(torch.float32).reshape(core_len, 4, spec.hidden_size)

        t_len, h = int(token_ids.shape[0]), spec.hidden_size
        n_h, n_kv, d, d_rel, eps = spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim, spec.d_rel, spec.rms_norm_eps
        x = _rms_norm(emb_tbl[token_ids], emb_norm, eps)
        blocks_slots: list[torch.Tensor] = []
        prelude_buf: torch.Tensor | None = None
        prelude_blocks_count, bank_f32, bank_valid = 0, None, False
        t_idx = torch.arange(t_len)[:, None]
        dist_mat = t_idx - torch.arange(t_len)[None, :]
        causal_mask = torch.triu(torch.full((t_len, t_len), float("-inf")), diagonal=1)
        pos_idx = torch.arange(t_len, dtype=torch.float32)

        for visit, l_idx in enumerate(spec.execution_plan):
            lw = provider.get_layer(l_idx)
            is_sliding = spec.is_sliding(l_idx)
            if l_idx == spec.loop_core_start:
                if visit == l_idx:
                    prelude_buf, prelude_blocks_count = x.clone(), len(blocks_slots)
                else:
                    states = torch.stack(blocks_slots[prelude_blocks_count + 1 :] + [x], dim=0)
                    bank_f32 = states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + eps)
                    bank_valid = True
                    assert prelude_buf is not None
                    x, blocks_slots = prelude_buf.clone(), blocks_slots[:prelude_blocks_count]

            use_bank = bank_valid and (spec.loop_core_start <= l_idx < spec.loop_core_end)
            msg_out = None
            if use_bank:
                assert bank_f32 is not None
                dots = torch.einsum("mth,qh->qtm", bank_f32, msg_queries[l_idx - spec.loop_core_start]) / math.sqrt(h)
                msg_out = torch.einsum("qtm,mth->qth", torch.softmax(dots, dim=-1), bank_f32)

            if l_idx == 0 and is_sliding:
                q_in = k_in = v_in = _rms_norm(x, lw["input_norm"], eps)
            else:
                ar_out = _attn_res(blocks_slots, x, lw["attn_input_query"], normalize_values=spec.attn_res_normalize_values)
                if is_sliding and not use_bank:
                    q_in = k_in = v_in = _rms_norm(ar_out[0], lw["input_norm"], eps)
                else:
                    nw = ([lw["input_norm"]] * 3) if is_sliding else [lw["q_norm"], lw["k_norm"], lw["v_norm"]]
                    ctrls, normed = lw.get("controllers"), []
                    for i, cn in enumerate(("attn", "k", "v")):
                        ar_i = ar_out[min(i, ar_out.shape[0] - 1)]
                        if use_bank and ctrls is not None and msg_out is not None:
                            ar_i = _msg_controller(ar_i, msg_out[i], *ctrls[cn], eps)
                        normed.append(_rms_norm(ar_i, nw[i], eps))
                    q_in, k_in, v_in = normed

            q_proj = q_in @ lw["q_proj"].t() + (lw["q_bias"] if "q_bias" in lw else 0.0)
            r_proj, sink_proj, gate_proj = q_in @ lw["r_proj"].t(), q_in @ lw["sink_proj"].t(), q_in @ lw["gate_proj"].t()
            k_proj, v_proj = k_in @ lw["k_proj"].t(), v_in @ lw["v_proj"].t()
            if is_sliding:
                k_conv, v_conv = _sconv_static(k_proj, lw["k_sconv"]), _sconv_static(v_proj, lw["v_sconv"])
            else:
                k_conv = _sconv_dynamic(k_proj, lw["k_sconv_static"], k_in @ lw["k_sconv_proj"].t(), n_kv, d)
                v_conv = _sconv_dynamic(v_proj, lw["v_sconv_static"], v_in @ lw["v_sconv_proj"].t(), n_kv, d)

            qv = q_proj.reshape(t_len, n_h, d) * lw["q_scale"].reshape(1, n_h, d)
            q_prep = qv * torch.clamp(spec.qk_l2_clip_threshold / (qv.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
            rv = r_proj.reshape(t_len, n_h, d_rel) * lw["rel_scale"].reshape(1, n_h, d_rel)
            r_prep = rv * torch.clamp(spec.rel_l2_clip_threshold / (rv.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
            if (not is_sliding) and spec.log_scaling_n_floor > 0:
                ratio = (pos_idx + 1.0) / float(spec.log_scaling_n_floor)
                tau = torch.where(ratio > 1.0, 1.0 + spec.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)), 1.0).reshape(t_len, 1, 1)
                q_prep, r_prep = q_prep * tau, r_prep * tau

            kv_r, vv_r = k_conv.reshape(t_len, n_kv, d), v_conv.reshape(t_len, n_kv, d)
            k_prep = kv_r * torch.clamp(spec.qk_l2_clip_threshold / (kv_r.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
            v_prep = vv_r * torch.rsqrt(vv_r.pow(2).mean(dim=-1, keepdim=True) + eps)

            rep = n_h // n_kv
            scores = torch.einsum("thd,shd->hts", q_prep, k_prep.repeat_interleave(rep, dim=1)) / math.sqrt(d)
            max_dist = min(spec.rel_extent, t_len)
            rel_bias = torch.einsum("thd,dr->htr", r_prep, lw["rel_proj"][:, :max_dist])[:, t_idx, torch.clamp(dist_mat, 0, max_dist - 1)]
            scores = scores + torch.where(((dist_mat >= 0) & (dist_mat < max_dist)).unsqueeze(0), rel_bias, 0.0)
            mask = causal_mask.masked_fill(dist_mat >= spec.sliding_window_size, float("-inf")) if (is_sliding and spec.sliding_window_size > 0) else causal_mask
            probs = torch.softmax(torch.cat([scores + mask.unsqueeze(0), sink_proj.t().unsqueeze(-1)], dim=-1), dim=-1)[..., :t_len]
            o_raw = torch.einsum("hts,shd->thd", probs, v_prep.repeat_interleave(rep, dim=1)).reshape(t_len, n_h * d) * torch.sigmoid(gate_proj)
            o_out = o_raw @ lw["o_proj"].t() + (lw["o_bias"] if "o_bias" in lw else 0.0)

            is_boundary = spec.is_boundary(l_idx)
            if is_boundary:
                blocks_slots.append(x.clone())
            res_f32 = _sconv_static(o_out, lw["attn_sconv"]) if is_boundary else (_sconv_static(o_out, lw["attn_sconv"]) + x)

            mlp_ar = _attn_res(blocks_slots, res_f32, lw["mlp_input_query"], normalize_values=spec.attn_res_normalize_values)[0]
            if use_bank and "controllers" in lw and msg_out is not None:
                mlp_ar = _msg_controller(mlp_ar, msg_out[3], *lw["controllers"]["mlp"], eps)
            mlp_in = _rms_norm(mlp_ar, lw["post_attn_norm"], eps)

            r_scores = torch.sigmoid(mlp_in @ lw["router_gate"].t())
            biased = r_scores + lw["router_bias"].unsqueeze(0) if spec.use_gate_bias else r_scores
            _, top_idx = torch.topk(biased, k=spec.experts_per_tok, dim=-1)
            top_w = torch.gather(r_scores, 1, top_idx)
            top_w = top_w * (spec.route_scale / (top_w.sum(dim=-1, keepdim=True) + 1e-20))

            g_bank, u_bank, d_bank = lw["gate_proj_e"], lw["up_proj_e"], lw["down_proj_e"]
            moe_out = torch.zeros((t_len, h), dtype=torch.float32)
            for e in sorted(set(top_idx.reshape(-1).tolist())):
                tok_i, slot_i = torch.where(top_idx == e)
                xe = mlp_in[tok_i]
                ge, ue = xe @ g_bank[e].t(), xe @ u_bank[e].t()
                act = (_tanh_safe(ge / spec.mlp_gate_softcap) * spec.mlp_gate_softcap * torch.sigmoid(ge)) * (_tanh_safe(ue / spec.mlp_up_softcap) * spec.mlp_up_softcap)
                moe_out.index_add_(0, tok_i, top_w[tok_i, slot_i].unsqueeze(-1) * (act @ d_bank[e].t()))
            x = res_f32 + _sconv_static(moe_out, lw["mlp_sconv"])

        final_h = _rms_norm(_attn_res(blocks_slots, x, final_query, normalize_values=spec.attn_res_normalize_values)[0], final_norm, eps)
        logits = final_h @ provider.get_head().t()
        return (_tanh_safe(logits / spec.final_logit_softcapping) * spec.final_logit_softcapping) if spec.final_logit_softcapping > 0 else logits


def _matrix_bytes(rows: int, cols: int, quant_str: str) -> int:
    if quant_str == "nvfp4" or quant_str == "verbatim_nvfp4":
        return _align_up(rows * (cols // 8) * 4, 256) + _align_up(rows * (cols // 16), 256) + 256
    if quant_str == "mxfp8":
        return _align_up(rows * cols, 256) + _align_up(rows * (cols // 32), 256)
    if quant_str == "bf16" or (cols % HQ_GROUP) != 0:
        return _align_up(rows * cols * 2, 256)
    bits, group, is_m5, is_shvq = parse_any_quant_spec(quant_str)
    eff_g = math.gcd(cols, group)
    bbytes = any_block_bytes(bits, eff_g, is_m5=(not is_shvq and bits == 4 and eff_g == 32 and is_m5), is_shvq=is_shvq)
    return _align_up(rows * (cols // eff_g) * bbytes, Q4G_ALIGNMENT)


def estimate_pack_metrics(spec: Inkling2Spec, recipe: Inkling2Recipe | None = None) -> dict[str, float]:
    """Estimate .qwp footprint (GiB) and B=1 active DRAM read per token (GB/tok)."""
    h, q_d, kv_d, r_d = spec.hidden_size, spec.num_attention_heads * spec.head_dim, spec.num_key_value_heads * spec.head_dim, spec.num_attention_heads * spec.d_rel
    moe_i, n_e, top_k = spec.moe_intermediate_size, spec.num_experts, spec.experts_per_tok
    embed_b = _align_up(spec.vocab_size * h * 2, 256)
    head_b = _matrix_bytes(spec.vocab_size, h, "bf16" if recipe is None else recipe.head_quant)
    head_act = int(head_b * (spec.unpadded_vocab_size / max(1, spec.vocab_size)))

    tot_l, act_l = [], []
    for l in range(spec.num_layers):
        dq = "bf16" if recipe is None else (recipe.coda_dense_quant if l >= spec.coda_start else recipe.trunk_quant)
        if recipe is None:
            gu_q = dn_q = spec.expert_mode(l, "gate_proj")
        elif l >= spec.coda_start:
            gu_q, dn_q = recipe.moe_coda_gate_up, recipe.moe_coda_down
        else:
            gu_q, dn_q = recipe.moe_main_gate_up, recipe.moe_main_down
        ab = _matrix_bytes(q_d, h, dq) * 2 + _matrix_bytes(kv_d, h, dq) * 2 + _matrix_bytes(h, q_d, dq) + _matrix_bytes(r_d, h, dq) + _matrix_bytes(spec.num_attention_heads, h, dq)
        if not spec.is_sliding(l):
            ab += _matrix_bytes(spec.num_key_value_heads * 4, h, dq) * 2
        if spec.loop_core_start <= l < spec.loop_core_end:
            ab += 4 * (_matrix_bytes(spec.loop_gate_rank, 2 * h, dq) + _matrix_bytes(h, spec.loop_gate_rank, dq) * 2)
        mb = _align_up(n_e * h * 2, 256) + _align_up(n_e * 4, 256) + 16384
        gu_b = _matrix_bytes(n_e * moe_i, h, gu_q) * 2 if gu_q in ("nvfp4", "verbatim_nvfp4", "mxfp8") else _matrix_bytes(n_e * 2 * moe_i, h, gu_q)
        dn_b = _matrix_bytes(n_e * h, moe_i, dn_q)
        tot_l.append(ab + mb + gu_b + dn_b)
        act_l.append(ab + mb + int((gu_b + dn_b) * min(1.0, float(top_k) / max(1, n_e))))

    return {"estimated_qwp_gib": (embed_b + head_b + sum(tot_l) + 65536) / float(1 << 30), "active_dram_gb_per_tok": (h * 2 + head_act + sum(act_l[l] for l in spec.execution_plan)) / 1e9}


def build_calibration_tokens(model_dir: Path | None, unpadded_vocab: int, num_tokens: int = 96) -> np.ndarray:
    if not (64 <= num_tokens <= 128):
        raise ValueError(f"num_tokens={num_tokens} must be in [64, 128]")
    if model_dir is not None and (model_dir / "tokenizer.json").is_file():
        try:
            from tokenizers import Tokenizer

            ids = [i for i in Tokenizer.from_file(str(model_dir / "tokenizer.json")).encode(CALIBRATION_TEXT).ids if 0 <= i < unpadded_vocab]
            if ids:
                while len(ids) < num_tokens:
                    ids.extend(ids)
                return np.asarray(ids[:num_tokens], dtype=np.int64)
        except Exception:
            pass
    return np.random.default_rng(20261121).integers(0, unpadded_vocab, size=(num_tokens,), dtype=np.int64)


def make_synthetic_inkling2_checkpoint(
    root: Path, *, layout: str = "mlx", num_layers: int = 24, hidden_size: int = 64, num_heads: int = 2,
    num_kv_heads: int = 1, head_dim: int = 32, d_rel: int = 4, moe_inter: int = 64, num_experts: int = 4,
    vocab_size: int = 128, unpadded_vocab_size: int = 120, gate_rank: int = 32, seed: int = 1337,
    with_vision: bool = True, with_dspark: bool = True, jitter_mlx_quant: bool = False,
) -> Path:
    """Create a synthetic 24-layer / 48-visit Inkling-2 checkpoint in HF or MLX layout."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    core_start, core_end, num_passes, sliding_window, rel_extent, top_k = 6, 18, 3, 16, 64, 2
    coda_start = max(core_end, num_layers - 4)
    local_layers = [l for l in range(num_layers) if (l + 1) % 6 != 0]
    quant_map: dict[str, object] = {"mode": "nvfp4", "bits": 4, "group_size": 16}
    hf_g2, hf_g3 = [], []
    for l in range(num_layers):
        mode = "nvfp4" if l < coda_start else "mxfp8"
        spec_l = {"mode": mode, "bits": 4 if mode == "nvfp4" else 8, "group_size": 16 if mode == "nvfp4" else 32}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quant_map[f"language_model.model.layers.{l}.mlp.switch_mlp.{proj}"] = spec_l
        (hf_g2 if l < coda_start else hf_g3).extend([f"model.llm.layers.{l}.mlp.experts.w13_weight", f"model.llm.layers.{l}.mlp.experts.w2_weight"])

    cfg: dict = {
        "architectures": ["Inkling2LoopForConditionalGeneration"], "eos_token_id": vocab_size - 2, "model_type": "inkling2_loop",
        "text_config": {
            "model_type": "inkling2_loop_text", "model_max_length": 1048576, "hidden_size": hidden_size,
            "num_hidden_layers": num_layers, "vocab_size": vocab_size, "unpadded_vocab_size": unpadded_vocab_size,
            "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads, "head_dim": head_dim, "d_rel": d_rel,
            "rel_extent": rel_extent, "q_bias": False, "o_bias": False, "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6, "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0, "local_layer_ids": local_layers, "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
            "sliding_window_size": sliding_window, "n_layers_per_attn_res_block": 6, "attn_res_normalize_values": True,
            "n_routed_experts": num_experts, "num_experts_per_tok": top_k, "n_shared_experts": 0, "intermediate_size": moe_inter,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True,
            "loop_config": {"core_start": core_start, "core_end": core_end, "num_passes": num_passes, "gate_rank": gate_rank, "is_varloop": False},
        },
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"} if layout == "hf" else quant_map,
    }
    if layout == "mlx":
        cfg["quantization"] = quant_map
        cfg["generation_config"] = {"eos_token_id": vocab_size - 2, "pad_token_id": vocab_size - 1}
    else:
        (root / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": [{"compute_dtype": "nvfp4", "tensors": hf_g2}, {"compute_dtype": "mxfp8", "tensors": hf_g3}]}}))
    if with_dspark:
        cfg["mtp_config"] = {"hidden_size": hidden_size, "num_hidden_layers": 3, "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads, "head_dim": head_dim, "intermediate_size": hidden_size * 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "q_bias": False, "o_bias": False, "local_layer_ids": [0, 2], "sliding_window_size": sliding_window, "block_size": 8, "markov_rank": gate_rank, "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]}
    if with_vision:
        cfg["vision_config"] = {"model_type": "inkling2_vision", "text_hidden_size": hidden_size, "temporal_patch_size": 2, "native_patch_size": 4, "adapter_merge_size": 2, "projection_hidden_dim": 32, "projector_bias": False, "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "mlp_ratio": 2.0, "posemb_grid_size": 4, "layer_norm_eps": 1e-5}
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "bos:{{ messages | length }}"}), encoding="utf-8")

    gen, rng = torch.Generator().manual_seed(seed), np.random.default_rng(seed)
    valid_fp8, valid_e8m0 = np.array([0x28, 0x30, 0x34, 0x38, 0x3C, 0x40, 0xB8, 0xBC], dtype=np.uint8), np.array([123, 124, 125, 126, 127], dtype=np.uint8)
    bf16 = lambda shape, sc=0.08: (torch.randn(shape, generator=gen, dtype=torch.float32) * sc).to(torch.bfloat16)
    f32 = lambda shape, sc=0.08: torch.randn(shape, generator=gen, dtype=torch.float32) * sc
    q_dim, kv_dim, tensors = num_heads * head_dim, num_kv_heads * head_dim, {}

    emb_w, emb_n, lm_h, f_norm, f_q = bf16((vocab_size, hidden_size), 0.25), torch.ones(hidden_size, dtype=torch.bfloat16), bf16((vocab_size, hidden_size), 0.15), torch.ones(hidden_size, dtype=torch.bfloat16), f32((hidden_size,))
    msg_q = f32((core_end - core_start, 4, hidden_size))
    if layout == "mlx":
        tensors.update({"language_model.model.embed_tokens.weight": emb_w, "language_model.model.embed_norm.weight": emb_n, "language_model.lm_head.weight": lm_h, "language_model.model.norm.weight": f_norm, "language_model.model.final_input.query": f_q, "language_model.model.message_queries": msg_q})
    else:
        cpq = torch.stack([msg_q[:, 0], msg_q[:, 3]], dim=1).reshape(-1, hidden_size).contiguous()
        cpkv = torch.stack([msg_q[:, 1], msg_q[:, 2]], dim=1).reshape(-1, hidden_size).contiguous()
        tensors.update({"model.llm.embed.weight": emb_w, "model.llm.embed_norm.weight": emb_n, "model.llm.unembed.weight": lm_h, "model.llm.norm.weight": f_norm, "model.llm.attn_res.output_query": f_q, "model.llm.middle_cycle_cross_pass_queries": cpq, "model.llm.middle_cycle_cross_pass_kv_queries": cpkv})

    local_set = set(local_layers)
    for l in range(num_layers):
        is_sl = l in local_set
        aq = f32((hidden_size,)) if (l > 0 and is_sl) else (f32((3, hidden_size)) if l > 0 else None)
        mq = f32((hidden_size,))
        a_sc, m_sc = bf16((hidden_size, 4), 0.1), bf16((hidden_size, 4), 0.1)
        if layout == "mlx":
            lp = f"language_model.model.layers.{l}."
            if aq is not None:
                tensors[f"{lp}attention_input.query"] = aq
            tensors[f"{lp}mlp_input.query"] = mq
            tensors[f"{lp}attn_sconv.conv.weight"], tensors[f"{lp}mlp_sconv.conv.weight"] = a_sc.unsqueeze(-1), m_sc.unsqueeze(-1)
        else:
            lp = f"model.llm.layers.{l}."
            if aq is not None:
                tensors[f"model.llm.attn_res.attn_queries.{l - 1}"] = aq
            tensors[f"model.llm.attn_res.mlp_queries.{l}"] = mq
            tensors[f"{lp}attn_sconv.weight"], tensors[f"{lp}mlp_sconv.weight"] = a_sc.unsqueeze(1), m_sc.unsqueeze(1)

        if is_sl:
            tensors[f"{lp}{'input_layernorm.weight' if layout == 'mlx' else 'attn_norm.weight'}"] = torch.ones(hidden_size, dtype=torch.bfloat16)
            kc, vc = bf16((kv_dim, 4), 0.1), bf16((kv_dim, 4), 0.1)
            sa = f"{lp}{'self_attn.' if layout == 'mlx' else 'attn.'}"
            tensors[f"{sa}{'k_sconv.conv.weight' if layout == 'mlx' else 'k_sconv.weight'}"] = kc.unsqueeze(-1) if layout == "mlx" else kc.unsqueeze(1)
            tensors[f"{sa}{'v_sconv.conv.weight' if layout == 'mlx' else 'v_sconv.weight'}"] = vc.unsqueeze(-1) if layout == "mlx" else vc.unsqueeze(1)
        else:
            for nm in ("q", "k", "v"):
                tensors[f"{lp}{nm}_{'input_layernorm' if layout == 'mlx' else 'input_norm'}.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)
            sa = f"{lp}{'self_attn.' if layout == 'mlx' else 'attn.'}"
            tensors[f"{sa}k_sconv.{'static_weight' if layout == 'mlx' else 'static_w'}"] = bf16((4, kv_dim), 0.1)
            tensors[f"{sa}v_sconv.{'static_weight' if layout == 'mlx' else 'static_w'}"] = bf16((4, kv_dim), 0.1)
            tensors[f"{sa}k_sconv.{'weight_proj.weight' if layout == 'mlx' else 'weight_proj'}"] = bf16((num_kv_heads * 4, hidden_size), 0.08)
            tensors[f"{sa}v_sconv.{'weight_proj.weight' if layout == 'mlx' else 'weight_proj'}"] = bf16((num_kv_heads * 4, hidden_size), 0.08)

        tensors[f"{lp}{'post_attention_layernorm.weight' if layout == 'mlx' else 'mlp_norm.weight'}"] = torch.ones(hidden_size, dtype=torch.bfloat16)
        proj_map = [("q_proj", "wq_du", (q_dim, hidden_size)), ("k_proj", "wk_dv", (kv_dim, hidden_size)), ("v_proj", "wv_dv", (kv_dim, hidden_size)), ("o_proj", "wo_ud", (hidden_size, q_dim)), ("gate_proj", "gate_attn", (q_dim, hidden_size)), ("r_proj", "wr_du", (num_heads * d_rel, hidden_size)), ("sink_proj", "attn_sink_proj", (num_heads, hidden_size))]
        for mk, hk, shp in proj_map:
            tensors[f"{sa}{mk if layout == 'mlx' else hk}.weight"] = bf16(shp)
        tensors[f"{sa}{'q_scale' if layout == 'mlx' else 'q_l2_clip.scale'}"] = torch.ones((num_heads, head_dim), dtype=torch.bfloat16)
        tensors[f"{sa}{'rel_scale' if layout == 'mlx' else 'rel_l2_clip.scale'}"] = torch.ones((num_heads, d_rel), dtype=torch.bfloat16)
        tensors[f"{sa}{'rel_proj' if layout == 'mlx' else 'rel_logits_proj.proj'}"] = bf16((d_rel, rel_extent), 0.05)

        if core_start <= l < core_end:
            for cn in ("attn", "k", "v", "mlp"):
                for sub, shp in (("down_weight", (gate_rank, 2 * hidden_size)), ("forget_weight", (hidden_size, gate_rank)), ("gain_weight", (hidden_size, gate_rank))):
                    tensors[f"{lp}controllers.{cn}.{sub}"] = bf16(shp, 0.05)

        tensors[f"{lp}mlp.{'gate_weight' if layout == 'mlx' else 'gate.weight'}"] = bf16((num_experts, hidden_size), 0.1)
        tensors[f"{lp}mlp.{'e_score_correction_bias' if layout == 'mlx' else 'gate.bias'}"] = f32((num_experts,), 0.02)

        ex_data = {}
        for proj, (n_r, k_c) in {"gate_proj": (moe_inter, hidden_size), "up_proj": (moe_inter, hidden_size), "down_proj": (hidden_size, moe_inter)}.items():
            if l < coda_start:
                w = rng.integers(0, 2**32, size=(num_experts, n_r, k_c // 8), dtype=np.uint32)
                if jitter_mlx_quant and proj == "gate_proj":
                    w = w ^ np.uint32(0x11111111)
                sc = rng.choice(valid_fp8, size=(num_experts, n_r, k_c // 16)).astype(np.uint8)
                amax = np.full((num_experts,), 134.4, dtype=np.float32)
                ex_data[proj] = (w, sc, amax, np.array([2688.0], dtype=np.float32))
            else:
                w = rng.choice(valid_fp8, size=(num_experts, n_r, k_c)).astype(np.uint8)
                sc = rng.choice(valid_e8m0, size=(num_experts, n_r, k_c // 32)).astype(np.uint8)
                ex_data[proj] = (w, sc)

        if layout == "mlx":
            for proj, tup in ex_data.items():
                sw = f"{lp}mlp.switch_mlp.{proj}."
                if l < coda_start:
                    tensors[sw + "weight"], tensors[sw + "scales"], tensors[sw + "weight_scale_2"] = torch.from_numpy(tup[0]), torch.from_numpy(tup[1]), torch.from_numpy(tup[2] / tup[3][0])
                else:
                    tensors[sw + "weight"], tensors[sw + "scales"] = torch.from_numpy(tup[0].view("<u4").copy()), torch.from_numpy(tup[1])
        else:
            ep = f"{lp}mlp.experts."
            for stem, p_pair in (("w13_weight", ("gate_proj", "up_proj")), ("w2_weight", ("down_proj",))):
                if l < coda_start:
                    w_list = [ex_data[p][0].view(np.uint8) for p in p_pair]
                    sc_list = [ex_data[p][1] for p in p_pair]
                    if len(w_list) == 2:
                        w_cat = np.empty((num_experts, w_list[0].shape[1] * 2, w_list[0].shape[2]), dtype=np.uint8)
                        sc_cat = np.empty((num_experts, sc_list[0].shape[1] * 2, sc_list[0].shape[2]), dtype=np.uint8)
                        w_cat[:, 0::2], w_cat[:, 1::2] = w_list[0], w_list[1]
                        sc_cat[:, 0::2], sc_cat[:, 1::2] = sc_list[0], sc_list[1]
                    else:
                        w_cat, sc_cat = w_list[0], sc_list[0]
                    tensors[ep + stem], tensors[ep + stem + "_scale"] = torch.from_numpy(w_cat), torch.from_numpy(sc_cat).view(torch.float8_e4m3fn)
                    tensors[ep + stem + "_amax"], tensors[ep + stem + "_outer_scale_denominator"] = torch.from_numpy(ex_data[p_pair[0]][2]), torch.from_numpy(ex_data[p_pair[0]][3])
                else:
                    w_list, sc_list = [ex_data[p][0] for p in p_pair], [ex_data[p][1] for p in p_pair]
                    if len(w_list) == 2:
                        w_cat = np.empty((num_experts, w_list[0].shape[1] * 2, w_list[0].shape[2]), dtype=np.uint8)
                        sc_cat = np.empty((num_experts, sc_list[0].shape[1] * 2, sc_list[0].shape[2]), dtype=np.uint8)
                        w_cat[:, 0::2], w_cat[:, 1::2] = w_list[0], w_list[1]
                        sc_cat[:, 0::2], sc_cat[:, 1::2] = sc_list[0], sc_list[1]
                    else:
                        w_cat, sc_cat = w_list[0], sc_list[0]
                    tensors[ep + stem], tensors[ep + stem + "_scale"] = torch.from_numpy(w_cat).view(torch.float8_e4m3fn), torch.from_numpy(sc_cat)

    if with_vision:
        pe, po = bf16((16, 48)), bf16((hidden_size, 32))
        if layout == "mlx":
            tensors["vision_tower.embeddings.patch_embedding.weight"], tensors["vision_tower.proj_out.weight"] = pe, po
        else:
            tensors["model.visual.backbone.embeddings.patch_embedding.weight"], tensors["model.visual.proj_out.weight"] = pe, po
    save_file(tensors, str(root / "model.safetensors"))

    if with_dspark:
        mtp = {
            "fuse_weight": bf16((4, 4), 0.2), "mask_embed": bf16((hidden_size,), 0.1), "norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
            "markov_head.markov_w1.weight": bf16((unpadded_vocab_size, gate_rank), 0.1),
            "markov_head.rnn_proj.weight": bf16((3 * gate_rank, 2 * gate_rank + 2 * hidden_size), 0.05), "markov_head.rnn_proj.bias": bf16((3 * gate_rank,), 0.02),
            "markov_head.dflare_norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
            "markov_head.hidden_out.weight": bf16((hidden_size, gate_rank), 0.08), "markov_head.hidden_out.bias": bf16((hidden_size,), 0.02),
        }
        for dl in range(3):
            dp = f"layers.{dl}."
            mtp.update({
                dp + "dflare_norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16), dp + "input_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
                dp + "post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16), dp + "self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                dp + "self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16), dp + "self_attn.k_norm_ctx.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                dp + "self_attn.q_proj.weight": bf16((q_dim, hidden_size)), dp + "self_attn.k_proj.weight": bf16((kv_dim, hidden_size)),
                dp + "self_attn.v_proj.weight": bf16((kv_dim, hidden_size)), dp + "self_attn.o_proj.weight": bf16((hidden_size, q_dim)),
                dp + "self_attn.k_proj_ctx.weight": bf16((kv_dim, hidden_size)), dp + "self_attn.v_proj_ctx.weight": bf16((kv_dim, hidden_size)),
                dp + "mlp.gate_proj.weight": bf16((hidden_size * 2, hidden_size)), dp + "mlp.up_proj.weight": bf16((hidden_size * 2, hidden_size)),
                dp + "mlp.down_proj.weight": bf16((hidden_size, hidden_size * 2)),
            })
        save_file(mtp, str(root / "mtp.safetensors"))
    return root


def _score_report_row(teacher_logits: np.ndarray, cand_logits: np.ndarray, contexts: np.ndarray, meta: dict) -> dict:
    sc = score_kl(teacher_logits, cand_logits, contexts, contexts)
    return {**meta, "positions": sc["positions"], "vocabulary": sc["vocabulary"], "mean_kl": sc["mean_kl"], "p95_kl": sc["p95_kl"], "max_kl": sc["max_kl"], "top1_agreement": sc["top1_agreement"]}


def run_eval(
    teacher_dir: Path | None, *, mlx_candidate_dir: Path | None = None, dry_run: bool = False, num_tokens: int = 96,
    candidates: tuple[str, ...] | None = None, out_dir: Path | None = None, device: str = "cpu", skip_stage1: bool = False,
) -> dict:
    """Run Stage-1 per-tensor error + Stage-2 48-visit teacher-KL evaluation."""
    cand_names = tuple(RECIPES.keys() if candidates is None else candidates)
    with tempfile.TemporaryDirectory(prefix="ink2_eval_") as tmp:
        tmp_path = Path(tmp)
        if dry_run:
            t_dir = make_synthetic_inkling2_checkpoint(tmp_path / "synth_hf_teacher", layout="hf")
            m_dir = make_synthetic_inkling2_checkpoint(tmp_path / "synth_mlx_cand", layout="mlx", jitter_mlx_quant=True) if mlx_candidate_dir is None else mlx_candidate_dir
        else:
            t_dir, m_dir = teacher_dir, mlx_candidate_dir
        assert t_dir is not None
        t_cfg = json.loads((t_dir / "config.json").read_text(encoding="utf-8"))
        t_reader, _ = open_inkling2_shards(t_dir)
        try:
            t_spec = parse_inkling2_config(t_cfg, source_dir=t_dir, reader=t_reader)
            t_src = Inkling2Source(t_reader, t_spec)
            stage1 = {} if skip_stage1 else eval_per_tensor_quant_fidelity(t_src, device=device)

            tok_np = build_calibration_tokens(t_dir, t_spec.unpadded_vocab_size, num_tokens=num_tokens)
            contexts = np.full((num_tokens, num_tokens), -1, dtype=np.int64)
            for p in range(num_tokens):
                contexts[p, : p + 1] = tok_np[: p + 1]
            tok_t = torch.from_numpy(tok_np)

            cap_dir = out_dir if out_dir is not None else (tmp_path / "captures")
            cap_dir.mkdir(parents=True, exist_ok=True)
            np.save(cap_dir / "contexts.npy", contexts)

            teacher_logits = forward_inkling2_48_visits(t_spec, _LayerWeightProvider(t_src, None, device=device), tok_t).cpu().numpy().astype(np.float32)
            np.save(cap_dir / "teacher_logits.npy", teacher_logits)

            cand_reports: dict[str, dict] = {}
            if m_dir is not None:
                m_cfg = json.loads((m_dir / "config.json").read_text(encoding="utf-8"))
                m_reader, _ = open_inkling2_shards(m_dir)
                try:
                    m_spec = parse_inkling2_config(m_cfg, source_dir=m_dir, reader=m_reader)
                    m_src = Inkling2Source(m_reader, m_spec)
                    m_logits = forward_inkling2_48_visits(m_spec, _LayerWeightProvider(m_src, None, device=device), tok_t).cpu().numpy().astype(np.float32)
                    np.save(cap_dir / "mlx_candidate_logits.npy", m_logits)
                    cand_reports["mlx_candidate"] = _score_report_row(teacher_logits, m_logits, contexts, {"recipe": {"name": "mlx_nvfp4_mxfp8"}, **estimate_pack_metrics(m_spec, None)})
                finally:
                    m_reader.close()

            for cname in cand_names:
                recipe = RECIPES[cname]
                c_logits = forward_inkling2_48_visits(t_spec, _LayerWeightProvider(t_src, recipe, device=device), tok_t).cpu().numpy().astype(np.float32)
                np.save(cap_dir / f"{cname}_logits.npy", c_logits)
                cand_reports[cname] = _score_report_row(teacher_logits, c_logits, contexts, {"recipe": asdict(recipe), **estimate_pack_metrics(t_spec, recipe)})
        finally:
            t_reader.close()

    return {"dry_run": dry_run, "teacher_layout": t_src.layout, "visits": len(t_spec.execution_plan), "num_layers": t_spec.num_layers, "num_tokens": num_tokens, "stage1": stage1, "candidates": cand_reports}


def _print_summary(report: dict) -> None:
    print(f"[eval_ink2_quant_kl] layout={report['teacher_layout']} visits={report['visits']} tokens={report['num_tokens']} dry_run={report['dry_run']}")
    if report.get("stage1"):
        print("Stage 1 NVFP4 FWHT_32 summary:", json.dumps(report["stage1"]["nvfp4_fwht32_requant_summary"]))
    hdr = f"{'Candidate':28s} | {'Mean KL':>9s} | {'P95 KL':>9s} | {'Max KL':>9s} | {'Top-1':>7s} | {'Est GiB':>7s} | {'GB/tok':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for cname, r in report["candidates"].items():
        print(f"{cname:28s} | {r['mean_kl']:9.5f} | {r['p95_kl']:9.5f} | {r['max_kl']:9.5f} | {r['top1_agreement'] * 100.0:6.2f}% | {r['estimated_qwp_gib']:7.3f} | {r['active_dram_gb_per_tok']:7.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", "--model", dest="teacher", type=Path, default=None, help="Teacher Inkling-2 dir (HF or MLX)")
    ap.add_argument("--mlx-candidate", type=Path, default=None, help="Optional MLX candidate dir to score against --teacher")
    ap.add_argument("--dry-run", action="store_true", help="Run synthetic HF-teacher vs MLX + Cand1..5 pipeline")
    ap.add_argument("--num-tokens", type=int, default=96, help="Calibration prompt length (64-128)")
    ap.add_argument("--candidates", nargs="*", default=list(RECIPES.keys()), choices=list(RECIPES.keys()))
    ap.add_argument("--out-dir", type=Path, default=None, help="Directory to store logit .npy captures")
    ap.add_argument("--output", type=Path, default=None, help="JSON report output path")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps"])
    ap.add_argument("--skip-stage1", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and args.teacher is None:
        ap.error("provide --teacher DIR or --dry-run")
    report = run_eval(args.teacher, mlx_candidate_dir=args.mlx_candidate, dry_run=args.dry_run, num_tokens=args.num_tokens, candidates=tuple(args.candidates), out_dir=args.out_dir, device=args.device, skip_stage1=args.skip_stage1)
    _print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
