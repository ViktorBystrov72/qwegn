"""Inkling-2 (inkling2_loop) QWP1 packer, HF/MLX adapter, and dequantizer.
Converts upstream HF (model.llm.*) or MLX (language_model.model.*) checkpoints,
vision tower, and DSpark MTP into 16 KiB-aligned QWP1 packs verified by verify.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterable

import numpy as np

from ernie55_adapter import ShardReader
from hq4 import GROUP as HQ_GROUP
import hq_blocks
from pack import Q4G_ALIGNMENT, pack_file, resolve_snapshot_dir, sha256, synthesize_base_chat_template
import shvq
from shvq import any_block_bytes, dequant_any_blocks, format_any_quant_spec, parse_any_quant_spec
from verify import audit_pack, read_header, verify_assets

E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _build_luts() -> tuple[np.ndarray, np.ndarray]:
    fp8 = np.empty(256, dtype=np.float32)
    e8m0 = np.empty(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp, mant = (b >> 3) & 0x0F, b & 0x07
        if exp == 15 and mant == 7:
            fp8[b] = np.nan
        elif exp == 0:
            fp8[b] = np.float32(sign * (2.0 ** -6) * (mant / 8.0))
        else:
            fp8[b] = np.float32(sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0))
        e8m0[b] = np.nan if b == 255 else np.float32(2.0 ** (-127 if b == 0 else (b - 127)))
    return fp8, e8m0


E4M3FN_LUT, UE8M0_LUT = _build_luts()
DSPARK_QUANT_CHOICES = ("bf16", "shvq4_g32", "hq8_g32", "hq4_g32_m5")
RECIPE_ALIASES = {
    "cand1": "cand1_hq8_shvq4", "cand2": "cand2_shvq4_uniform", "cand3": "cand3_maxwell_shvq3d4",
    "cand4": "cand4_hq4_scalar", "cand5": "cand5_hybrid_lossless_moe", "cand6": "cand6_shvq4_shvq2d3",
    "cand6b": "cand6b_shvq4_shvq3_moe", "cand7": "cand7_shvq3_shvq2_ultra", "cand8": "cand8_shvq3_shvq2g128",
}


@dataclass(frozen=True)
class Inkling2Recipe:
    name: str
    trunk_quant: str
    coda_dense_quant: str
    head_quant: str
    moe_main_gate_up: str
    moe_main_down: str
    moe_coda_gate_up: str
    moe_coda_down: str


RECIPES: dict[str, Inkling2Recipe] = {
    "cand1_hq8_shvq4": Inkling2Recipe("cand1_hq8_shvq4", "hq8_g32", "shvq4_g32", "shvq4_g32", "shvq4_g64", "shvq4_g64", "shvq4_g32", "shvq4_g32"),
    "cand2_shvq4_uniform": Inkling2Recipe("cand2_shvq4_uniform", "shvq4_g32", "shvq4_g32", "shvq4_g32", "shvq4_g64", "shvq4_g64", "shvq4_g64", "shvq4_g64"),
    "cand3_maxwell_shvq3d4": Inkling2Recipe("cand3_maxwell_shvq3d4", "hq8_g32", "shvq4_g32", "shvq4_g32", "shvq3_g64", "shvq4_g64", "shvq3_g64", "shvq4_g64"),
    "cand4_hq4_scalar": Inkling2Recipe("cand4_hq4_scalar", "hq4_g32_m5", "hq4_g32_m5", "hq4_g32_m5", "hq4_g64", "hq4_g64", "hq4_g64", "hq4_g64"),
    "cand5_hybrid_lossless_moe": Inkling2Recipe("cand5_hybrid_lossless_moe", "shvq4_g32", "shvq4_g32", "shvq4_g32", "verbatim_nvfp4", "verbatim_nvfp4", "shvq4_g32", "shvq4_g32"),
    "cand6_shvq4_shvq2d3": Inkling2Recipe("cand6_shvq4_shvq2d3", "shvq4_g32", "shvq4_g32", "shvq4_g32", "shvq2_g64", "shvq3_g64", "shvq3_g64", "shvq4_g64"),
    "cand6b_shvq4_shvq3_moe": Inkling2Recipe("cand6b_shvq4_shvq3_moe", "shvq4_g32", "shvq4_g32", "shvq4_g32", "shvq3_g64", "shvq3_g64", "shvq3_g64", "shvq4_g64"),
    "cand7_shvq3_shvq2_ultra": Inkling2Recipe("cand7_shvq3_shvq2_ultra", "shvq3_g32", "shvq3_g32", "shvq4_g32", "shvq2_g64", "shvq2_g64", "shvq2_g64", "shvq3_g64"),
    "cand8_shvq3_shvq2g128": Inkling2Recipe("cand8_shvq3_shvq2g128", "shvq3_g64", "shvq3_g64", "shvq3_g32", "shvq2_g128", "shvq2_g64", "shvq2_g64", "shvq3_g64"),
}


def resolve_recipe(name: str, *, cand5_trunk: str | None = None, cand5_coda: str | None = None) -> Inkling2Recipe:
    key = RECIPE_ALIASES.get(name, name)
    if key not in RECIPES:
        raise ValueError(f"Unknown Inkling-2 recipe {name!r}; expected {list(RECIPES)}")
    recipe = RECIPES[key]
    if key == "cand5_hybrid_lossless_moe" and (cand5_trunk is not None or cand5_coda is not None):
        trunk = cand5_trunk or recipe.trunk_quant
        coda = cand5_coda or recipe.moe_coda_gate_up
        return Inkling2Recipe(recipe.name, trunk, coda, recipe.head_quant, "verbatim_nvfp4", "verbatim_nvfp4", coda, coda)
    return recipe


@dataclass(frozen=True)
class Inkling2Spec:
    hidden_size: int
    num_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window_size: int
    local_layer_ids: frozenset[int]
    layers_per_attn_res_block: int
    attn_res_normalize_values: bool
    num_experts: int
    experts_per_tok: int
    moe_intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    use_gate_bias: bool
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    final_logit_softcapping: float
    loop_core_start: int
    loop_core_end: int
    loop_num_passes: int
    loop_gate_rank: int
    q_bias: bool
    o_bias: bool
    per_layer_expert_modes: dict[int, dict[str, str]]
    coda_start: int
    has_dspark: bool
    has_vision: bool

    @property
    def execution_plan(self) -> list[int]:
        core = list(range(self.loop_core_start, self.loop_core_end)) * self.loop_num_passes
        return list(range(0, self.loop_core_start)) + core + list(range(self.loop_core_end, self.num_layers))

    def is_sliding(self, layer: int) -> bool:
        return layer in self.local_layer_ids

    def is_boundary(self, layer: int) -> bool:
        return self.layers_per_attn_res_block > 0 and (layer % self.layers_per_attn_res_block == 0)

    def expert_mode(self, layer: int, proj: str = "gate_proj") -> str:
        return self.per_layer_expert_modes.get(layer, {}).get(proj, "bf16")


def _infer_hf_quant_modes(source_dir: Path | None, reader: ShardReader | None, num_layers: int) -> dict[int, dict[str, str]]:
    modes: dict[int, dict[str, str]] = {}
    if source_dir is not None and (source_dir / "hf_quant_config.json").is_file():
        hq = json.loads((source_dir / "hf_quant_config.json").read_text(encoding="utf-8"))
        for grp in (hq.get("model") or {}).get("parameter_groups") or []:
            comp = str(grp.get("compute_dtype", "")).lower()
            if comp not in ("nvfp4", "mxfp8"):
                continue
            for tname in grp.get("tensors") or []:
                if ".layers." not in tname:
                    continue
                head = tname.split(".layers.", 1)[1].split(".", 1)[0]
                if head.isdigit():
                    lidx = int(head)
                    if "w13_" in tname:
                        modes.setdefault(lidx, {})["gate_proj"] = comp
                        modes[lidx]["up_proj"] = comp
                    elif "w2_" in tname:
                        modes.setdefault(lidx, {})["down_proj"] = comp
    if reader is not None and reader.has("model.llm.embed.weight"):
        for lidx in range(num_layers):
            if lidx in modes:
                continue
            ep = f"model.llm.layers.{lidx}.mlp.experts."
            if reader.has(ep + "w13_weight_amax"):
                modes[lidx] = {"gate_proj": "nvfp4", "up_proj": "nvfp4", "down_proj": "nvfp4"}
            elif reader.has(ep + "w13_weight_scale"):
                modes[lidx] = {"gate_proj": "mxfp8", "up_proj": "mxfp8", "down_proj": "mxfp8"}
    return modes


def parse_inkling2_config(cfg: dict, *, source_dir: Path | None = None, reader: ShardReader | None = None) -> Inkling2Spec:
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    num_layers = int(tc["num_hidden_layers"])
    sw_size = int(tc.get("sliding_window_size") or tc.get("sliding_window") or 1024)
    loop_cfg = tc.get("loop_config") or {}
    core_start, core_end = int(loop_cfg.get("core_start", 6)), int(loop_cfg.get("core_end", 18))
    if bool(loop_cfg.get("is_varloop", False)):
        raise ValueError("inkling2 varloop is not supported")

    qroot = cfg.get("quantization") or cfg.get("quantization_config") or tc.get("quantization") or {}
    def_mode = str(qroot.get("mode", "bf16")).lower()
    per_layer: dict[int, dict[str, str]] = {}
    for k, v in qroot.items():
        if isinstance(v, dict) and "layers." in k:
            head = k.split("layers.", 1)[1].partition(".")[0]
            if head.isdigit():
                per_layer.setdefault(int(head), {})[k.rsplit(".", 1)[-1]] = str(v.get("mode", def_mode)).lower()
    for lidx, row in _infer_hf_quant_modes(source_dir, reader, num_layers).items():
        per_layer.setdefault(lidx, {}).update(row)
    for lidx in range(num_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            per_layer.setdefault(lidx, {}).setdefault(proj, def_mode)

    non_nv = [l for l in range(core_end, num_layers) if per_layer[l]["gate_proj"] != "nvfp4"]
    coda_start = min(non_nv) if non_nv else max(core_end, num_layers - max(1, num_layers // 6))
    n_h = int(tc["num_attention_heads"])
    h = int(tc["hidden_size"])
    return Inkling2Spec(
        h, num_layers, int(tc["vocab_size"]), int(tc.get("unpadded_vocab_size") or tc["vocab_size"]), n_h,
        int(tc.get("num_key_value_heads") or n_h), int(tc.get("head_dim") or (h // n_h)),
        int(tc.get("d_rel", 16)), int(tc.get("rel_extent", sw_size)), sw_size,
        frozenset(int(x) for x in (tc.get("local_layer_ids") or [])), int(tc.get("n_layers_per_attn_res_block", 6)),
        bool(tc.get("attn_res_normalize_values", True)), int(tc.get("n_routed_experts") or tc.get("num_experts") or 128),
        int(tc.get("num_experts_per_tok", 8)), int(tc.get("moe_intermediate_size") or tc["intermediate_size"]),
        float(tc.get("mlp_gate_softcap", 4.0)), float(tc.get("mlp_up_softcap", 25.0)), float(tc.get("route_scale", 8.0)),
        bool(tc.get("use_gate_bias", True)), float(tc.get("qk_l2_clip_threshold", 22.627416997969522)),
        float(tc.get("rel_l2_clip_threshold", 8.0)), int(tc.get("log_scaling_n_floor", 8192)),
        float(tc.get("log_scaling_alpha", 0.1)), float(tc.get("rms_norm_eps", 1e-6)), float(tc.get("final_logit_softcapping", 30.0)),
        core_start, core_end, int(loop_cfg.get("num_passes", 3)), int(loop_cfg.get("gate_rank", 64)),
        bool(tc.get("q_bias", False)), bool(tc.get("o_bias", False)), per_layer, coda_start,
        isinstance(cfg.get("mtp_config"), dict), isinstance(cfg.get("vision_config"), dict),
    )


def _as_raw_u8(arr) -> np.ndarray:
    """View uint8/float8/uint32/int32 weights as packed little-endian uint8 bytes."""
    import torch

    if isinstance(arr, torch.Tensor):
        t = arr.detach().cpu().contiguous()
        if t.dtype in (torch.uint8, torch.uint32, torch.int32, getattr(torch, "float8_e4m3fn", torch.uint8)):
            return t.view(torch.uint8).numpy()
        return t.to(torch.uint8).numpy()
    np_a = np.ascontiguousarray(arr)
    if np_a.dtype == np.uint8:
        return np_a
    if np_a.dtype in (np.uint32, np.int32):
        return np.ascontiguousarray(np_a, dtype="<u4").view(np.uint8)
    return np_a.astype(np.uint8)


def dequant_nvfp4(weight, scales, weight_scale_2, *, as_torch: bool = True):
    """Dequantize NVFP4_G16 (MLX u32 or HF u8 + E4M3FN scales + F32 s2) -> f32."""
    import torch

    w_u8 = _as_raw_u8(weight)
    if w_u8.ndim not in (2, 3):
        raise ValueError(f"NVFP4 weight must be 2D or 3D, got {w_u8.shape}")
    *prefix, rows, packed_bytes = w_u8.shape
    k = packed_bytes * 2
    if k % 16 != 0:
        raise ValueError(f"NVFP4 K={k} must be a multiple of 16")
    sc_u8 = _as_raw_u8(scales)
    if sc_u8.shape != (*prefix, rows, k // 16):
        raise ValueError(f"NVFP4 scales shape {sc_u8.shape} != expected {(*prefix, rows, k // 16)}")
    s2_np = weight_scale_2.detach().cpu().to(torch.float32).numpy() if isinstance(weight_scale_2, torch.Tensor) else np.asarray(weight_scale_2, dtype=np.float32)

    nibs = np.stack([w_u8 & 0x0F, (w_u8 >> 4) & 0x0F], axis=-1).reshape(*prefix, rows, k // 16, 16)
    sc_f32 = E4M3FN_LUT[sc_u8]
    if not np.isfinite(sc_f32).all():
        raise ValueError("NVFP4 scales contain NaN (0x7F/0xFF E4M3FN)")
    if s2_np.ndim == 0 or s2_np.size == 1:
        s2_b = np.float32(s2_np.reshape(-1)[0])
    elif prefix and s2_np.shape[0] == prefix[0]:
        s2_b = s2_np.reshape(prefix[0], 1, 1, 1)
    else:
        s2_b = s2_np
    if not np.isfinite(s2_b).all():
        raise ValueError("NVFP4 weight_scale_2 contains non-finite values")
    out = (E2M1_LUT[nibs] * (sc_f32[..., None] * s2_b)).reshape(*prefix, rows, k).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(out)) if as_torch else out


def dequant_mxfp8(weight, scales, *, as_torch: bool = True):
    """Dequantize MXFP8_G32 (HF f8_e4m3fn/u8 or MLX u32 + UE8M0 scales) -> f32."""
    import torch

    w_u8 = _as_raw_u8(weight)
    if w_u8.ndim not in (2, 3):
        raise ValueError(f"MXFP8 weight must be 2D or 3D, got {w_u8.shape}")
    *prefix, rows, k = w_u8.shape
    if k % 32 != 0:
        raise ValueError(f"MXFP8 K={k} must be a multiple of 32")
    sc_u8 = _as_raw_u8(scales)
    if sc_u8.shape != (*prefix, rows, k // 32):
        raise ValueError(f"MXFP8 scales shape {sc_u8.shape} != expected {(*prefix, rows, k // 32)}")
    w_f32 = E4M3FN_LUT[w_u8].reshape(*prefix, rows, k // 32, 32)
    sc_f32 = UE8M0_LUT[sc_u8][..., None]
    if not np.isfinite(w_f32).all() or not np.isfinite(sc_f32).all():
        raise ValueError("MXFP8 weights or UE8M0 scales contain NaN")
    out = (w_f32 * sc_f32).reshape(*prefix, rows, k).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(out)) if as_torch else out


def dequant_bf16(arr, *, as_torch: bool = True):
    import torch

    if isinstance(arr, torch.Tensor):
        t = arr.detach().cpu().to(torch.float32)
        return t if as_torch else t.numpy()
    np_arr = np.asarray(arr, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(np_arr)) if as_torch else np_arr


def _fhex(x: float, fmt: str) -> str:
    return "0x" + format(struct.unpack(">Q" if fmt == "d" else ">I", struct.pack(f">{fmt}", float(x)))[0], "x")


def inkling2_geo_v2(cfg: dict) -> str:
    """Compute canonical qwg-geo-v2 string matching Swift QWModelDescriptor.digestFields."""
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    spec = parse_inkling2_config(cfg)
    raw_types = tc.get("layer_types")
    if isinstance(raw_types, list) and len(raw_types) == spec.num_layers:
        kinds = "".join("F" if t in ("full_attention", "deepseek_sparse_attention") else ("C" if t == "compressed_attention" else ("S" if t == "sliding_attention" else "L")) for t in raw_types)
    else:
        kinds = "".join("S" if spec.is_sliding(l) else "F" for l in range(spec.num_layers))
    if tc.get("full_attention_interval") is not None:
        fa_int = int(tc["full_attention_interval"])
    elif tc.get("n_layers_per_attn_res_block") is not None:
        fa_int = int(tc["n_layers_per_attn_res_block"])
    elif "F" in kinds:
        cand = kinds.index("F") + 1
        fa_int = cand if all((i + 1) % cand == 0 for i, ch in enumerate(kinds) if ch == "F") else 0
    else:
        fa_int = 1
    rot_f = float(tc.get("partial_rotary_factor") or (tc.get("rope_parameters") or {}).get("partial_rotary_factor") or 1.0)
    theta = float(tc.get("rope_theta") or (tc.get("rope_parameters") or {}).get("rope_theta") or (cfg.get("mtp_config") or {}).get("rope_theta") or 10000.0)
    mtp_layers = int(tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers") or tc.get("mtp_num_layers") or (cfg.get("mtp_config") or {}).get("num_hidden_layers") or 0)
    eos_raw = tc.get("eos_token_id") if tc.get("eos_token_id") is not None else cfg.get("eos_token_id")
    if eos_raw is None:
        eos_raw = (cfg.get("generation_config") or {}).get("eos_token_id", 200006)
    eos = int(eos_raw[0] if isinstance(eos_raw, list) else eos_raw)
    maxpos = int(tc.get("max_position_embeddings") or tc.get("model_max_length") or cfg.get("max_position_embeddings") or cfg.get("model_max_length") or 1048576)
    qgate = 1 if tc.get("attn_output_gate", True) else 0
    fields = [
        f"model_type={tc['model_type']}", f"hidden={spec.hidden_size}", f"vocab={spec.vocab_size}",
        f"layers={spec.num_layers}", f"layer_kinds={kinds}", f"fa_interval={fa_int}",
        f"q={spec.num_attention_heads}", f"kv={spec.num_key_value_heads}", f"hd={spec.head_dim}", f"qgate={qgate}",
        f"rot_f={_fhex(rot_f, 'd')}", f"theta={_fhex(theta, 'd')}", "gdn=0,0,0,0",
        f"mtp={mtp_layers},{1 if tc.get('mtp_use_dedicated_embeddings') else 0}", f"eos={eos}",
        f"rms={_fhex(spec.rms_norm_eps, 'f')}", f"maxpos={maxpos}", "idx=0,0,0,0,0",
        f"moe={spec.num_experts},{spec.experts_per_tok},{spec.moe_intermediate_size},0,sigmoid",
        "hc=0,0", "ple=,0,0,0,0,0,0,0",
    ]
    return "qwg-geo-v2:" + "|".join(fields)


def inkling2_pack_identity(cfg: dict) -> dict[str, str]:
    return {"model_id": "inkling2_loop", "geometry_digest": hashlib.sha256(inkling2_geo_v2(cfg).encode()).hexdigest()}


def open_inkling2_shards(source: Path) -> tuple[ShardReader, list[Path]]:
    main_files = [p for p in sorted(source.glob("*.safetensors")) if p.name != "mtp.safetensors"]
    if not main_files:
        raise ValueError(f"no model .safetensors under {source}")
    return ShardReader(source, main_files), main_files


def detect_layout(reader: ShardReader) -> str:
    return "hf" if reader.has("model.llm.embed.weight") else "mlx"


def detect_prefix(reader: ShardReader) -> str:
    return "language_model." if (reader.has("language_model.model.embed_tokens.weight") or detect_layout(reader) == "hf") else ""


_HF_GLOBAL_MAP = {
    "embed_tokens.weight": "model.llm.embed.weight", "embed_norm.weight": "model.llm.embed_norm.weight",
    "norm.weight": "model.llm.norm.weight", "lm_head.weight": "model.llm.unembed.weight",
    "final_input.query": "model.llm.attn_res.output_query",
}
_HF_LAYER_MAP = {
    "mlp_input.query": "__mlp_query__", "attention_input.query": "__attn_query__",
    "input_layernorm.weight": "attn_norm.weight", "q_input_layernorm.weight": "q_input_norm.weight",
    "k_input_layernorm.weight": "k_input_norm.weight", "v_input_layernorm.weight": "v_input_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight", "attn_sconv.conv.weight": "attn_sconv.weight",
    "mlp_sconv.conv.weight": "mlp_sconv.weight", "self_attn.q_proj.weight": "attn.wq_du.weight",
    "self_attn.k_proj.weight": "attn.wk_dv.weight", "self_attn.v_proj.weight": "attn.wv_dv.weight",
    "self_attn.o_proj.weight": "attn.wo_ud.weight", "self_attn.gate_proj.weight": "attn.gate_attn.weight",
    "self_attn.r_proj.weight": "attn.wr_du.weight", "self_attn.sink_proj.weight": "attn.attn_sink_proj.weight",
    "self_attn.q_proj.bias": "attn.wq_du.bias", "self_attn.o_proj.bias": "attn.wo_ud.bias",
    "self_attn.q_scale": "attn.q_l2_clip.scale", "self_attn.rel_scale": "attn.rel_l2_clip.scale",
    "self_attn.rel_proj": "attn.rel_logits_proj.proj", "self_attn.k_sconv.conv.weight": "attn.k_sconv.weight",
    "self_attn.v_sconv.conv.weight": "attn.v_sconv.weight", "self_attn.k_sconv.static_weight": "attn.k_sconv.static_w",
    "self_attn.v_sconv.static_weight": "attn.v_sconv.static_w", "self_attn.k_sconv.weight_proj.weight": "attn.k_sconv.weight_proj",
    "self_attn.v_sconv.weight_proj.weight": "attn.v_sconv.weight_proj", "mlp.gate_weight": "mlp.gate.weight",
    "mlp.e_score_correction_bias": "mlp.gate.bias",
}


class Inkling2Source:
    """Unified tensor reader across upstream HF (model.llm.*) and MLX (language_model.model.*)."""

    def __init__(self, reader: ShardReader, spec: Inkling2Spec):
        self.reader, self.spec = reader, spec
        self.layout = detect_layout(reader)
        self.in_pfx = "language_model." if reader.has("language_model.model.embed_tokens.weight") else ""
        self.out_pfx = "language_model." if (self.in_pfx == "language_model." or self.layout == "hf") else ""
        self._msg_queries = None

    def _raw_layer_name(self, layer: int, key: str) -> str:
        if self.layout == "mlx":
            return f"{self.in_pfx}model.layers.{layer}.{key}"
        hf_sub = _HF_LAYER_MAP.get(key, key)
        if hf_sub == "__mlp_query__":
            return f"model.llm.attn_res.mlp_queries.{layer}"
        if hf_sub == "__attn_query__":
            return f"model.llm.attn_res.attn_queries.{layer - 1}"
        return f"model.llm.layers.{layer}.{hf_sub}"

    def get_global(self, key: str):
        import torch

        if self.layout == "mlx":
            return self.reader.get(f"{self.in_pfx}{key}" if key == "lm_head.weight" else f"{self.in_pfx}model.{key}")
        if key == "message_queries":
            if self._msg_queries is None:
                cpq, cpkv = self.reader.get("model.llm.middle_cycle_cross_pass_queries"), self.reader.get("model.llm.middle_cycle_cross_pass_kv_queries")
                self._msg_queries = torch.stack([cpq[0::2], cpkv[0::2], cpkv[1::2], cpq[1::2]], dim=1).contiguous()
            return self._msg_queries
        return self.reader.get(_HF_GLOBAL_MAP[key])

    def has_layer(self, layer: int, key: str) -> bool:
        return self.reader.has(self._raw_layer_name(layer, key))

    def get_layer(self, layer: int, key: str):
        t = self.reader.get(self._raw_layer_name(layer, key))
        return t.reshape(t.shape[0], t.shape[2], 1).contiguous() if (self.layout == "hf" and key.endswith("sconv.conv.weight") and t.ndim == 3 and t.shape[1] == 1) else t

    def _hf_expert_slice(self, layer: int, proj: str, with_scale: bool, with_s2: bool):
        ep, stem = f"model.llm.layers.{layer}.mlp.experts.", ("w2_weight" if proj == "down_proj" else "w13_weight")
        row_sl = slice(0, None, 2) if proj == "gate_proj" else (slice(1, None, 2) if proj == "up_proj" else slice(None))
        w = self.reader.get(ep + stem)[:, row_sl].contiguous()
        sc = self.reader.get(ep + stem + "_scale")[:, row_sl].contiguous() if with_scale else None
        s2 = (self.reader.get(ep + stem + "_amax").float() / self.reader.get(ep + stem + "_outer_scale_denominator").float()).contiguous() if with_s2 else None
        return w, sc, s2

    def get_nvfp4_raw_proj(self, layer: int, proj: str):
        import torch

        if self.layout == "mlx":
            sw = f"{self.in_pfx}model.layers.{layer}.mlp.switch_mlp."
            return self.reader.get(f"{sw}{proj}.weight"), self.reader.get(f"{sw}{proj}.scales"), self.reader.get(f"{sw}{proj}.weight_scale_2")
        w, sc, s2 = self._hf_expert_slice(layer, proj, with_scale=True, with_s2=True)
        return w.view(torch.uint32), sc, s2

    def load_expert_proj_f32(self, layer: int, proj: str, *, as_torch: bool = True):
        mode = self.spec.expert_mode(layer, proj)
        if self.layout == "mlx":
            return load_expert_proj_f32(self.reader, self.spec, f"{self.in_pfx}model.layers.{layer}.mlp.switch_mlp.", layer, proj, as_torch=as_torch)
        w, sc, s2 = self._hf_expert_slice(layer, proj, with_scale=(mode in ("nvfp4", "mxfp8")), with_s2=(mode == "nvfp4"))
        if mode == "nvfp4":
            return dequant_nvfp4(w, sc, s2, as_torch=as_torch)
        return dequant_mxfp8(w, sc, as_torch=as_torch) if mode == "mxfp8" else dequant_bf16(w, as_torch=as_torch)

    def iter_vision_canonical(self):
        for raw in sorted(self.reader.tensor_to_file):
            cname = _canonical_vision_name(raw)
            if cname is not None:
                yield cname, self.reader.get(raw)


def _canonical_vision_name(raw: str) -> str | None:
    if raw.startswith("vision_tower."):
        return "model.visual." + raw[len("vision_tower.") :]
    if not raw.startswith("model.visual."):
        return None
    tail = raw[len("model.visual.") :]
    if not tail.startswith("backbone."):
        return "model.visual." + tail
    sub = tail[len("backbone.") :]
    if sub.startswith("ln_pre."):
        return "model.visual.pre_layernorm." + sub[len("ln_pre.") :]
    if sub.startswith("layers."):
        for old, new in (
            (".ln_1.", ".layernorm_before."),
            (".ln_2.", ".layernorm_after."),
            (".ls_1.gamma", ".lambda_1"),
            (".ls_2.gamma", ".lambda_2"),
            (".mlp.c_fc.", ".mlp.fc1."),
            (".mlp.c_proj.", ".mlp.fc2."),
        ):
            sub = sub.replace(old, new)
    return "model.visual." + sub


def load_expert_proj_f32(reader: ShardReader, spec: Inkling2Spec, sw_pfx: str, layer: int, proj: str, *, as_torch: bool = True):
    mode = spec.expert_mode(layer, proj)
    w = reader.get(f"{sw_pfx}{proj}.weight")
    if mode == "nvfp4":
        return dequant_nvfp4(w, reader.get(f"{sw_pfx}{proj}.scales"), reader.get(f"{sw_pfx}{proj}.weight_scale_2"), as_torch=as_torch)
    if mode == "mxfp8":
        return dequant_mxfp8(w, reader.get(f"{sw_pfx}{proj}.scales"), as_torch=as_torch)
    return dequant_bf16(w, as_torch=as_torch)


@contextmanager
def maybe_force_cpu_quant(device: str):
    if device != "cpu":
        yield
        return
    import torch

    prev_metal, prev_mps, prev_avail = shvq._METAL_ENC_FAILED, hq_blocks._MPS_QUANT_FAILED, torch.backends.mps.is_available
    shvq._METAL_ENC_FAILED = True
    hq_blocks._MPS_QUANT_FAILED = True
    torch.backends.mps.is_available = lambda: False
    try:
        yield
    finally:
        shvq._METAL_ENC_FAILED, hq_blocks._MPS_QUANT_FAILED, torch.backends.mps.is_available = prev_metal, prev_mps, prev_avail


def _emit_2d_linear(name: str, arr, quant_str: str, expected_shapes: dict[str, list[int]], stats: dict[str, int]):
    shape = list(arr.shape)
    expected_shapes[name] = shape
    if len(shape) == 2 and shape[1] % HQ_GROUP == 0 and quant_str != "bf16":
        fmt = format_any_quant_spec(*parse_any_quant_spec(quant_str))
        stats[fmt] = stats.get(fmt, 0) + 1
        return name, arr, parse_any_quant_spec(quant_str)
    stats["bf16"] = stats.get("bf16", 0) + 1
    return name, arr, "bf16"


def _stream_inkling2_tensors(src: Inkling2Source, recipe: Inkling2Recipe, *, include_vision: bool, expected_shapes: dict[str, list[int]], stats: dict[str, int]):
    import torch

    spec, pfx = src.spec, src.out_pfx
    mdl = f"{pfx}model."

    def emit_raw(name: str, arr, kind: str | None = None):
        k = kind or ("f32" if isinstance(arr, torch.Tensor) and arr.dtype == torch.float32 else "bf16")
        expected_shapes[name] = list(arr.shape)
        stats[k] = stats.get(k, 0) + 1
        return name, arr, k

    def emit_quant(name: str, arr, q_str: str):
        expected_shapes[name] = list(arr.shape)
        fmt = format_any_quant_spec(*parse_any_quant_spec(q_str))
        stats[fmt] = stats.get(fmt, 0) + 1
        return name, arr, parse_any_quant_spec(q_str)

    for gkey, gkind in (("embed_tokens.weight", "bf16"), ("embed_norm.weight", "bf16"), ("norm.weight", "bf16"), ("final_input.query", "f32"), ("message_queries", "f32")):
        yield emit_raw(f"{mdl}{gkey}", src.get_global(gkey), gkind)
    yield _emit_2d_linear(f"{pfx}lm_head.weight", src.get_global("lm_head.weight"), recipe.head_quant, expected_shapes, stats)

    for l in range(spec.num_layers):
        lp, sa, sw = f"{mdl}layers.{l}.", f"{mdl}layers.{l}.self_attn.", f"{mdl}layers.{l}.mlp.switch_mlp."
        dense_q = recipe.coda_dense_quant if l >= spec.coda_start else recipe.trunk_quant
        if l > 0:
            yield emit_raw(f"{lp}attention_input.query", src.get_layer(l, "attention_input.query"), "f32")
        yield emit_raw(f"{lp}mlp_input.query", src.get_layer(l, "mlp_input.query"), "f32")
        norms = ("input_layernorm.weight",) if spec.is_sliding(l) else ("q_input_layernorm.weight", "k_input_layernorm.weight", "v_input_layernorm.weight")
        for nk in (*norms, "post_attention_layernorm.weight", "attn_sconv.conv.weight", "mlp_sconv.conv.weight"):
            yield emit_raw(f"{lp}{nk}", src.get_layer(l, nk), "bf16")
        for proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "r_proj.weight", "sink_proj.weight"):
            yield _emit_2d_linear(f"{sa}{proj}", src.get_layer(l, f"self_attn.{proj}"), dense_q, expected_shapes, stats)
        for sk in ("q_proj.bias", "o_proj.bias", "q_scale", "rel_scale", "rel_proj"):
            if sk.endswith(".bias") and not src.has_layer(l, f"self_attn.{sk}"):
                continue
            yield emit_raw(f"{sa}{sk}", src.get_layer(l, f"self_attn.{sk}"), "bf16")
        kv_raw = ("k_sconv.conv.weight", "v_sconv.conv.weight") if spec.is_sliding(l) else ("k_sconv.static_weight", "v_sconv.static_weight")
        for sk in kv_raw:
            yield emit_raw(f"{sa}{sk}", src.get_layer(l, f"self_attn.{sk}"), "bf16")
        if not spec.is_sliding(l):
            for sk in ("k_sconv.weight_proj.weight", "v_sconv.weight_proj.weight"):
                yield _emit_2d_linear(f"{sa}{sk}", src.get_layer(l, f"self_attn.{sk}"), dense_q, expected_shapes, stats)
        if spec.loop_core_start <= l < spec.loop_core_end:
            for cn in ("attn", "k", "v", "mlp"):
                for sub in ("down_weight", "forget_weight", "gain_weight"):
                    yield emit_raw(f"{lp}controllers.{cn}.{sub}", src.get_layer(l, f"controllers.{cn}.{sub}"), "bf16")
        yield emit_raw(f"{lp}mlp.gate_weight", src.get_layer(l, "mlp.gate_weight"), "bf16")
        yield emit_raw(f"{lp}mlp.e_score_correction_bias", src.get_layer(l, "mlp.e_score_correction_bias"), "f32")

        gu_q, dn_q = (recipe.moe_coda_gate_up, recipe.moe_coda_down) if l >= spec.coda_start else (recipe.moe_main_gate_up, recipe.moe_main_down)
        if gu_q == "verbatim_nvfp4" and spec.expert_mode(l, "gate_proj") == "nvfp4":
            for proj in ("gate_proj", "up_proj"):
                w_u32, sc_u8, s2 = src.get_nvfp4_raw_proj(l, proj)
                yield emit_raw(f"{sw}{proj}.weight", w_u32.contiguous().view(torch.float32), "f32")
                yield emit_raw(f"{sw}{proj}.scales", sc_u8.contiguous().view(torch.bfloat16), "bf16")
                yield emit_raw(f"{sw}{proj}.weight_scale_2", s2.contiguous().to(torch.float32), "f32")
                stats["verbatim_nvfp4"] = stats.get("verbatim_nvfp4", 0) + 1
        else:
            gu_stack = torch.cat([src.load_expert_proj_f32(l, "gate_proj"), src.load_expert_proj_f32(l, "up_proj")], dim=1).contiguous()
            yield emit_quant(f"{lp}mlp.experts.gate_up_proj", gu_stack, recipe.moe_coda_gate_up if gu_q == "verbatim_nvfp4" else gu_q)
        if dn_q == "verbatim_nvfp4" and spec.expert_mode(l, "down_proj") == "nvfp4":
            w_u32, sc_u8, s2 = src.get_nvfp4_raw_proj(l, "down_proj")
            yield emit_raw(f"{sw}down_proj.weight", w_u32.contiguous().view(torch.float32), "f32")
            yield emit_raw(f"{sw}down_proj.scales", sc_u8.contiguous().view(torch.bfloat16), "bf16")
            yield emit_raw(f"{sw}down_proj.weight_scale_2", s2.contiguous().to(torch.float32), "f32")
            stats["verbatim_nvfp4"] = stats.get("verbatim_nvfp4", 0) + 1
        else:
            yield emit_quant(f"{lp}mlp.experts.down_proj", src.load_expert_proj_f32(l, "down_proj").contiguous(), recipe.moe_coda_down if dn_q == "verbatim_nvfp4" else dn_q)

    if include_vision:
        for cname, t in src.iter_vision_canonical():
            yield emit_raw(cname, t, "bf16")


def _dspark_geometry_hash(cfg: dict) -> str:
    mc = cfg.get("mtp_config") or {}
    s = f"qwg-dspark-v1:{mc.get('hidden_size', 0)}|{mc.get('num_hidden_layers', 3)}|{mc.get('num_attention_heads', 0)}|{mc.get('num_key_value_heads', 0)}|{mc.get('head_dim', 128)}|{mc.get('intermediate_size', 0)}|{mc.get('block_size', 8)}|{mc.get('markov_rank', 256)}"
    return hashlib.sha256(s.encode()).hexdigest()


def pack_dspark_qwp(mtp_path: Path, out_qwp: Path, cfg: dict, *, quant_dspark: str = "bf16", assets_manifest_sha256: str, flat_root: bool = False) -> dict[str, int]:
    """Pack Inkling-2 DSpark (mtp.safetensors) into QWP1 dspark.qwp."""
    if quant_dspark not in DSPARK_QUANT_CHOICES:
        raise ValueError(f"Unsupported --quant-dspark {quant_dspark!r}; expected {DSPARK_QUANT_CHOICES}")
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(str(mtp_path))
    with safe_open(str(mtp_path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
    expected_shapes, stats = {}, {"bf16": 0}

    def is_quant_2d(name: str, t) -> bool:
        if quant_dspark == "bf16" or t.ndim != 2 or t.shape[1] % HQ_GROUP != 0:
            return False
        if name in ("fuse_weight", "markov_head.markov_w1.weight", "markov_head.rnn_proj.weight", "markov_head.hidden_out.weight"):
            return False
        return ".self_attn." in name or ".mlp." in name

    def stream():
        for name in keys:
            arr = tensors[name]
            expected_shapes[name] = list(arr.shape)
            if is_quant_2d(name, arr):
                fmt = format_any_quant_spec(*parse_any_quant_spec(quant_dspark))
                stats[fmt] = stats.get(fmt, 0) + 1
                yield name, arr, parse_any_quant_spec(quant_dspark)
            else:
                kind = "f32" if arr.dtype == torch.float32 else "bf16"
                stats[kind] = stats.get(kind, 0) + 1
                yield name, arr, kind

    pack_file(stream(), out_qwp, {"model_id": "dspark_inkling2_loop", "geometry_hash": _dspark_geometry_hash(cfg), "source": str(mtp_path), "sha256_source": sha256(mtp_path), "quant_recorded": quant_dspark, "assets_dir": "." if flat_root else "assets", "assets_manifest_sha256": assets_manifest_sha256})
    audit_pack(out_qwp, expected_shapes=expected_shapes)
    return stats


def _hash_files(files: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def pack_inkling2_model(
    source: Path, destination: Path, *, recipe_name: str = "cand1_hq8_shvq4", quant_dspark: str = "bf16",
    include_vision: bool = True, flat_root: bool = False, device: str = "auto", cand5_trunk: str | None = None, cand5_coda: str | None = None,
) -> dict:
    """Pack Inkling-2-Mini (HF/MLX safetensors + mtp.safetensors) into QWP1."""
    import torch

    if quant_dspark not in DSPARK_QUANT_CHOICES:
        raise ValueError(f"Unsupported --quant-dspark {quant_dspark!r}; expected {DSPARK_QUANT_CHOICES}")
    recipe = resolve_recipe(recipe_name, cand5_trunk=cand5_trunk, cand5_coda=cand5_coda)
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    source, destination = resolve_snapshot_dir(source).resolve(strict=True), destination.expanduser().absolute()
    if destination.exists() or destination.suffix == ".qwp":
        raise ValueError(f"invalid --out directory: {destination}")
    cfg_path = source / "config.json"
    if not cfg_path.is_file() or not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"missing config.json or tokenizer.json in {source}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    embedded_tpl = None
    if not (source / "chat_template.jinja").is_file():
        tok_cfg_p = source / "tokenizer_config.json"
        if tok_cfg_p.is_file():
            embedded_tpl = json.loads(tok_cfg_p.read_text(encoding="utf-8")).get("chat_template")
        if not isinstance(embedded_tpl, str) or not embedded_tpl:
            embedded_tpl = synthesize_base_chat_template(source)

    reader, main_files = open_inkling2_shards(source)
    mtp_path = source / "mtp.safetensors"
    if flat_root and mtp_path.is_file():
        reader.close()
        raise ValueError("--flat-root cannot be combined with sibling dspark.qwp")

    try:
        spec = parse_inkling2_config(cfg, source_dir=source, reader=reader)
        src = Inkling2Source(reader, spec)
        sha_src, identity, expected_shapes, stats = _hash_files(main_files), inkling2_pack_identity(cfg), {}, {"bf16": 0}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            assets = staging if flat_root else staging / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            manifest = {}
            for f in sorted(source.iterdir()):
                if f.is_file() and not f.name.startswith(".") and not f.name.endswith((".safetensors", ".index.json")) and f.name != "crc32.txt":
                    shutil.copyfile(f, assets / f.name)
                    manifest[f.name] = {"sha256": sha256(f), "bytes": f.stat().st_size}
            if embedded_tpl and not (assets / "chat_template.jinja").is_file():
                (assets / "chat_template.jinja").write_text(embedded_tpl, encoding="utf-8")
                manifest["chat_template.jinja"] = {"sha256": sha256(assets / "chat_template.jinja"), "bytes": (assets / "chat_template.jinja").stat().st_size}
            manifest_path = assets / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
            manifest_sha, weights, dspark_stats = sha256(manifest_path), staging / "model.qwp", None
            with maybe_force_cpu_quant(device):
                prov = {
                    **identity, "source": str(source), "source_layout": src.layout, "sha256_source": sha_src, "recipe": recipe.name,
                    "quant_recorded": f"{recipe.name}:trunk={recipe.trunk_quant}__head={recipe.head_quant}__moe={recipe.moe_main_gate_up}/{recipe.moe_main_down}",
                    "nvfp4_raw_packing": "u32_as_f32_bits__u8_pairs_as_bf16_bits" if recipe.moe_main_gate_up == "verbatim_nvfp4" else None,
                    "dspark_pack": "dspark.qwp" if mtp_path.is_file() else None, "dspark_quant": quant_dspark if mtp_path.is_file() else None,
                    "assets_dir": "." if flat_root else "assets", "assets_manifest_sha256": manifest_sha,
                }
                pack_file(_stream_inkling2_tensors(src, recipe, include_vision=include_vision, expected_shapes=expected_shapes, stats=stats), weights, prov)
                if mtp_path.is_file():
                    dspark_qwp = staging / "dspark.qwp"
                    dspark_stats = pack_dspark_qwp(mtp_path, dspark_qwp, cfg, quant_dspark=quant_dspark, assets_manifest_sha256=manifest_sha, flat_root=flat_root)
                    (staging / "dspark.qwp.sha256").write_text(f"{sha256(dspark_qwp)}  dspark.qwp\n")
            hdr, _ = audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            verify_assets(staging, hdr)
            staging.rename(destination)
    finally:
        reader.close()

    return {"source_layout": src.layout, "recipe": recipe.name, "quant_dspark": quant_dspark if mtp_path.is_file() else None, "stats": stats, "dspark_stats": dspark_stats, "model_qwp_bytes": (destination / "model.qwp").stat().st_size, "dspark_qwp_bytes": (destination / "dspark.qwp").stat().st_size if (destination / "dspark.qwp").is_file() else 0}


def reload_qwp_tensor(f, entry: dict) -> np.ndarray:
    """Read and dequantize any tensor entry from an open QWP1 file handle."""
    import torch

    shape = entry["shape"]
    if "raw_off" in entry:
        f.seek(entry["raw_off"])
        rkind = entry.get("kind", "bf16")
        if rkind == "f32":
            return np.frombuffer(f.read(math.prod(shape) * 4), dtype="<f4").reshape(shape).copy()
        if rkind == "bf16":
            return torch.from_numpy(np.frombuffer(f.read(math.prod(shape) * 2), dtype="<u2").reshape(shape).copy()).view(torch.bfloat16).to(torch.float32).numpy()
        raise ValueError(f"unsupported raw kind: {rkind}")
    hq = entry.get("hq")
    bits, group, is_m5, is_shvq = (4, 32, True, False) if hq is None else (int(hq["bits"]), int(hq["group"]), False, hq.get("vq") == "shvq1")
    bbytes = any_block_bytes(bits, group, is_m5=is_m5, is_shvq=is_shvq)
    f.seek(entry["hq4_off"])
    if "experts" in entry:
        blks = np.frombuffer(f.read(shape[0] * entry["e_hq4"]), dtype=np.uint8).reshape(shape[0] * shape[1], shape[2] // group, bbytes)
        return dequant_any_blocks(blks, bits=bits, group=group, is_m5=is_m5, is_shvq=is_shvq).reshape(shape)
    blks = np.frombuffer(f.read(shape[0] * (shape[1] // group) * bbytes), dtype=np.uint8).reshape(shape[0], shape[1] // group, bbytes)
    return dequant_any_blocks(blks, bits=bits, group=group, is_m5=is_m5, is_shvq=is_shvq)


def reload_nvfp4_proj_f32(f, tensors: dict[str, dict], sw_pfx: str, proj: str) -> np.ndarray:
    """Reload verbatim NVFP4 weight/scales/weight_scale_2 entries from QWP1 and dequantize."""
    we, se, s2e = tensors[f"{sw_pfx}{proj}.weight"], tensors[f"{sw_pfx}{proj}.scales"], tensors[f"{sw_pfx}{proj}.weight_scale_2"]
    f.seek(we["raw_off"])
    w_u32 = np.frombuffer(f.read(math.prod(we["shape"]) * 4), dtype="<u4").reshape(we["shape"]).copy()
    f.seek(se["raw_off"])
    sc_u8 = np.frombuffer(f.read(math.prod(se["shape"]) * 2), dtype=np.uint8).reshape(*se["shape"][:-1], se["shape"][-1] * 2).copy()
    f.seek(s2e["raw_off"])
    s2 = np.frombuffer(f.read(math.prod(s2e["shape"]) * 4), dtype="<f4").reshape(s2e["shape"]).copy()
    return dequant_nvfp4(w_u32, sc_u8, s2, as_torch=False)


def verify_16k_alignment(qwp_path: Path) -> list[str]:
    """Return all 2D/3D quantized hq4_off entries violating 16 KiB VM-page alignment."""
    hdr, _ = read_header(qwp_path)
    return [f"{n}: hq4_off={e['hq4_off']} not aligned to {Q4G_ALIGNMENT}" for n, e in hdr["tensors"].items() if "hq4_off" in e and (e["hq4_off"] % Q4G_ALIGNMENT) != 0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="source Inkling-2 checkpoint directory")
    ap.add_argument("--out", required=True, type=Path, help="output QWP1 directory")
    ap.add_argument("--recipe", default="cand1_hq8_shvq4", help="quant recipe (cand1..cand8)")
    ap.add_argument("--quant-dspark", default="bf16", choices=DSPARK_QUANT_CHOICES)
    ap.add_argument("--no-vision", action="store_true", help="exclude vision tensors")
    ap.add_argument("--flat-root", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--cand5-trunk", default=None, choices=["shvq4_g32", "hq8_g32"])
    ap.add_argument("--cand5-coda", default=None, choices=["shvq4_g32", "hq8_g32"])
    args = ap.parse_args(argv)
    print(json.dumps(pack_inkling2_model(args.model, args.out, recipe_name=args.recipe, quant_dspark=args.quant_dspark, include_vision=not args.no_vision, flat_root=args.flat_root, device=args.device, cand5_trunk=args.cand5_trunk, cand5_coda=args.cand5_coda), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
