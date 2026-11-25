#!/usr/bin/env python3
"""PyTorch FP32 oracle fixtures for M4 LFM3-VL and SigLIP2 decision vision towers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def _rand(shape: tuple[int, ...], gen: torch.Generator, scale: float = 0.08) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).contiguous()


def _merge_block_coords(h: int, w: int, merge: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for bh in range(h // merge):
        for bw in range(w // merge):
            for i in range(merge):
                for j in range(merge):
                    coords.append((bh * merge + i, bw * merge + j))
    return coords


def _layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return gamma * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_2d_axial_rope(
    x: torch.Tensor,
    coords: list[tuple[int, int]],
    head_dim: int,
    theta: float,
) -> torch.Tensor:
    # Matches Lfm3VlVisionRotaryEmbedding (`[freq_h, freq_w, freq_h, freq_w]` + `rotate_half`).
    spatial = head_dim // 2
    inv_count = max(1, spatial // 2)
    exponents = torch.arange(inv_count, dtype=torch.float32) * 2.0 / float(max(2, spatial))
    inv_freq = 1.0 / (theta ** exponents)
    pos_y = torch.tensor([c[0] for c in coords], dtype=torch.float32).unsqueeze(1)
    pos_x = torch.tensor([c[1] for c in coords], dtype=torch.float32).unsqueeze(1)
    ang_y = pos_y * inv_freq.unsqueeze(0)
    ang_x = pos_x * inv_freq.unsqueeze(0)
    hw_cos = torch.cat([torch.cos(ang_y), torch.cos(ang_x)], dim=-1)
    hw_sin = torch.cat([torch.sin(ang_y), torch.sin(ang_x)], dim=-1)
    c = torch.cat([hw_cos, hw_cos], dim=-1).unsqueeze(1)
    s = torch.cat([hw_sin, hw_sin], dim=-1).unsqueeze(1)
    return (x * c) + (_rotate_half(x) * s)


def _gqa_bidirectional(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    kv_heads: int,
    head_dim: int,
    coords: list[tuple[int, int]] | None = None,
    rope_theta: float = 10000.0,
    q_norm: torch.Tensor | None = None,
    k_norm: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    n_seq = q.shape[0]
    qh = q.view(n_seq, heads, head_dim)
    kh = k.view(n_seq, kv_heads, head_dim)
    vh = v.view(n_seq, kv_heads, head_dim)
    if q_norm is not None:
        qh = _rms_norm(qh, q_norm, eps)
    if k_norm is not None:
        kh = _rms_norm(kh, k_norm, eps)
    if coords is not None:
        qh = _apply_2d_axial_rope(qh, coords, head_dim, rope_theta)
        kh = _apply_2d_axial_rope(kh, coords, head_dim, rope_theta)
    kv_rep = max(1, heads // max(1, kv_heads))
    kh_exp = kh.repeat_interleave(kv_rep, dim=1)
    vh_exp = vh.repeat_interleave(kv_rep, dim=1)
    qt = qh.permute(1, 0, 2)
    kt = kh_exp.permute(1, 0, 2)
    vt = vh_exp.permute(1, 0, 2)
    attn = torch.softmax((qt @ kt.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
    out = (attn @ vt).permute(1, 0, 2).contiguous()
    return out.view(n_seq, heads * head_dim)


def build_lfm3_oracle(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(4041)

    dim, heads, kv_heads, mlp_inner = 32, 4, 2, 64
    head_dim = dim // heads
    patch, temporal, merge, conv_k = 4, 2, 2, 3
    h_grid, w_grid = 4, 6
    n_patch = h_grid * w_grid
    n_token = n_patch // (merge * merge)
    merged_dim = dim * merge * merge
    proj_mid, out_dim = 64, 48
    rope_theta, eps, proj_eps = 10000.0, 1e-5, 1e-6
    coords = _merge_block_coords(h_grid, w_grid, merge)
    patch_chan = 3 * temporal * patch * patch

    config = {
        "architectures": ["D1Model"],
        "model_type": "lfm3_vl",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 248056,
        "projector_bias": True,
        "projector_hidden_size": proj_mid,
        "projector_layer_norm_eps": proj_eps,
        "text_config": {
            "model_type": "lfm3",
            "eos_token_id": 2,
            "hidden_size": 128,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 4,
            "conv_bias": False,
            "max_position_embeddings": 2048,
            "norm_eps": 1e-5,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"}
            },
        },
        "vision_config": {
            "model_type": "lfm3_vl_vision",
            "hidden_size": dim,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "num_hidden_layers": 2,
            "intermediate_size": mlp_inner,
            "patch_size": patch,
            "temporal_patch_size": temporal,
            "spatial_merge_size": merge,
            "out_hidden_size": out_dim,
            "conv_kernel_size": conv_k,
            "layer_types": ["conv", "full_attention"],
            "rope_parameters": {"rope_theta": rope_theta, "rope_type": "axial"},
            "norm_eps": eps,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    pfx = "model.vision_tower."
    tensors: dict[str, torch.Tensor] = {
        f"{pfx}patchifier.weight": _rand((dim, patch_chan), gen, 0.12),
        f"{pfx}embedding_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}layers.0.operator_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}layers.0.conv.in_proj.weight": _rand((3 * dim, dim), gen, 0.08),
        f"{pfx}layers.0.conv.spatial_conv.weight": _rand((dim, 1, conv_k, conv_k), gen, 0.12),
        f"{pfx}layers.0.conv.out_proj.weight": _rand((dim, dim), gen, 0.08),
        f"{pfx}layers.0.ffn_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}layers.0.feed_forward.w1.weight": _rand((mlp_inner, dim), gen, 0.08),
        f"{pfx}layers.0.feed_forward.w2.weight": _rand((dim, mlp_inner), gen, 0.08),
        f"{pfx}layers.0.feed_forward.w3.weight": _rand((mlp_inner, dim), gen, 0.08),
        f"{pfx}layers.1.operator_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}layers.1.self_attn.q_proj.weight": _rand((dim, dim), gen, 0.08),
        f"{pfx}layers.1.self_attn.k_proj.weight": _rand((kv_heads * head_dim, dim), gen, 0.08),
        f"{pfx}layers.1.self_attn.v_proj.weight": _rand((kv_heads * head_dim, dim), gen, 0.08),
        f"{pfx}layers.1.self_attn.out_proj.weight": _rand((dim, dim), gen, 0.08),
        f"{pfx}layers.1.self_attn.q_layernorm.weight": torch.ones((head_dim,), dtype=torch.float32) + _rand((head_dim,), gen, 0.1),
        f"{pfx}layers.1.self_attn.k_layernorm.weight": torch.ones((head_dim,), dtype=torch.float32) + _rand((head_dim,), gen, 0.1),
        f"{pfx}layers.1.ffn_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}layers.1.feed_forward.w1.weight": _rand((mlp_inner, dim), gen, 0.08),
        f"{pfx}layers.1.feed_forward.w2.weight": _rand((dim, mlp_inner), gen, 0.08),
        f"{pfx}layers.1.feed_forward.w3.weight": _rand((mlp_inner, dim), gen, 0.08),
        "model.multi_modal_projector.layer_norm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        "model.multi_modal_projector.layer_norm.bias": _rand((dim,), gen, 0.04),
        "model.multi_modal_projector.linear_1.weight": _rand((proj_mid, merged_dim), gen, 0.06),
        "model.multi_modal_projector.linear_1.bias": _rand((proj_mid,), gen, 0.02),
        "model.multi_modal_projector.linear_2.weight": _rand((out_dim, proj_mid), gen, 0.06),
        "model.multi_modal_projector.linear_2.bias": _rand((out_dim,), gen, 0.02),
    }
    save_file(tensors, str(out_dir / "model.safetensors"))

    patches = _rand((n_patch, patch_chan), gen, 0.5)
    x = F.linear(patches, tensors[f"{pfx}patchifier.weight"])
    x = _rms_norm(x, tensors[f"{pfx}embedding_norm.weight"], eps)

    # Layer 0: gated spatial_conv in raster order + SwiGLU MLP.
    normed0 = _rms_norm(x, tensors[f"{pfx}layers.0.operator_norm.weight"], eps)
    b_g, c_g, x_g = F.linear(normed0, tensors[f"{pfx}layers.0.conv.in_proj.weight"]).chunk(3, dim=-1)
    gated = b_g * x_g
    raster = torch.zeros((h_grid, w_grid, dim), dtype=torch.float32)
    for p_idx, (ry, rx) in enumerate(coords):
        raster[ry, rx] = gated[p_idx]
    conv_out = F.conv2d(
        raster.permute(2, 0, 1).unsqueeze(0),
        tensors[f"{pfx}layers.0.conv.spatial_conv.weight"],
        padding=conv_k // 2,
        groups=dim,
    )[0].permute(1, 2, 0)
    gathered0 = torch.stack([conv_out[ry, rx] for ry, rx in coords], dim=0)
    x = x + F.linear(c_g * gathered0, tensors[f"{pfx}layers.0.conv.out_proj.weight"])
    n2_0 = _rms_norm(x, tensors[f"{pfx}layers.0.ffn_norm.weight"], eps)
    h_mlp0 = F.silu(F.linear(n2_0, tensors[f"{pfx}layers.0.feed_forward.w1.weight"])) * F.linear(
        n2_0, tensors[f"{pfx}layers.0.feed_forward.w3.weight"]
    )
    x = x + F.linear(h_mlp0, tensors[f"{pfx}layers.0.feed_forward.w2.weight"])

    # Layer 1: 2D Axial-RoPE bidirectional GQA with QK RMSNorm + SwiGLU MLP.
    normed1 = _rms_norm(x, tensors[f"{pfx}layers.1.operator_norm.weight"], eps)
    q = F.linear(normed1, tensors[f"{pfx}layers.1.self_attn.q_proj.weight"])
    k = F.linear(normed1, tensors[f"{pfx}layers.1.self_attn.k_proj.weight"])
    v = F.linear(normed1, tensors[f"{pfx}layers.1.self_attn.v_proj.weight"])
    attn1 = _gqa_bidirectional(
        q, k, v, heads, kv_heads, head_dim, coords=coords, rope_theta=rope_theta,
        q_norm=tensors[f"{pfx}layers.1.self_attn.q_layernorm.weight"],
        k_norm=tensors[f"{pfx}layers.1.self_attn.k_layernorm.weight"],
        eps=eps,
    )
    x = x + F.linear(attn1, tensors[f"{pfx}layers.1.self_attn.out_proj.weight"])
    n2_1 = _rms_norm(x, tensors[f"{pfx}layers.1.ffn_norm.weight"], eps)
    h_mlp1 = F.silu(F.linear(n2_1, tensors[f"{pfx}layers.1.feed_forward.w1.weight"])) * F.linear(
        n2_1, tensors[f"{pfx}layers.1.feed_forward.w3.weight"]
    )
    x = x + F.linear(h_mlp1, tensors[f"{pfx}layers.1.feed_forward.w2.weight"])

    x = _layer_norm(
        x,
        tensors["model.multi_modal_projector.layer_norm.weight"],
        tensors["model.multi_modal_projector.layer_norm.bias"],
        proj_eps,
    )
    merged = x.view(n_token, merged_dim)
    mid = F.gelu(
        F.linear(merged, tensors["model.multi_modal_projector.linear_1.weight"], tensors["model.multi_modal_projector.linear_1.bias"]),
        approximate="none",
    )
    out = F.linear(mid, tensors["model.multi_modal_projector.linear_2.weight"], tensors["model.multi_modal_projector.linear_2.bias"])

    (out_dir / "patches.f32").write_bytes(patches.numpy().tobytes())
    (out_dir / "expected.f32").write_bytes(out.contiguous().numpy().tobytes())
    meta = {"h": h_grid, "w": w_grid, "t": 1, "n_patch": n_patch, "n_token": n_token, "out_dim": out_dim}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def build_siglip2_oracle(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(5051)

    dim, heads, mlp_inner = 32, 4, 64
    head_dim = dim // heads
    patch, merge, pos_side = 4, 2, 4
    num_pos = pos_side * pos_side
    h_grid, w_grid = 4, 6
    n_patch = h_grid * w_grid
    n_token = n_patch // (merge * merge)
    merged_dim = dim * merge * merge
    proj_mid, out_dim = 64, 48
    eps = 1e-6
    coords = _merge_block_coords(h_grid, w_grid, merge)
    patch_chan = 3 * patch * patch

    config = {
        "architectures": ["Lfm2VlForConditionalGeneration"],
        "model_type": "lfm2_vl",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 248056,
        "text_config": {
            "model_type": "lfm2",
            "eos_token_id": 2,
            "hidden_size": 128,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 4,
            "conv_bias": False,
            "max_position_embeddings": 2048,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
        },
        "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_size": dim,
            "num_attention_heads": heads,
            "num_hidden_layers": 2,
            "intermediate_size": mlp_inner,
            "patch_size": patch,
            "num_position_embeddings": num_pos,
            "spatial_merge_size": merge,
            "out_dim": out_dim,
            "layer_norm_eps": eps,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    pfx = "vision_tower.vision_model."
    tensors: dict[str, torch.Tensor] = {
        f"{pfx}embeddings.patch_embedding.weight": _rand((dim, 3, patch, patch), gen, 0.12),
        f"{pfx}embeddings.patch_embedding.bias": _rand((dim,), gen, 0.03),
        f"{pfx}embeddings.position_embedding.weight": _rand((num_pos, dim), gen, 0.08),
        f"{pfx}post_layernorm.weight": torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1),
        f"{pfx}post_layernorm.bias": _rand((dim,), gen, 0.04),
        "multi_modal_projector.norm.weight": torch.ones((merged_dim,), dtype=torch.float32) + _rand((merged_dim,), gen, 0.1),
        "multi_modal_projector.norm.bias": _rand((merged_dim,), gen, 0.04),
        "multi_modal_projector.linear_1.weight": _rand((proj_mid, merged_dim), gen, 0.06),
        "multi_modal_projector.linear_1.bias": _rand((proj_mid,), gen, 0.02),
        "multi_modal_projector.linear_2.weight": _rand((out_dim, proj_mid), gen, 0.06),
        "multi_modal_projector.linear_2.bias": _rand((out_dim,), gen, 0.02),
    }
    for li in range(2):
        lp = f"{pfx}encoder.layers.{li}"
        tensors[f"{lp}.layer_norm1.weight"] = torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1)
        tensors[f"{lp}.layer_norm1.bias"] = _rand((dim,), gen, 0.05)
        tensors[f"{lp}.self_attn.q_proj.weight"] = _rand((dim, dim), gen, 0.08)
        tensors[f"{lp}.self_attn.q_proj.bias"] = _rand((dim,), gen, 0.02)
        tensors[f"{lp}.self_attn.k_proj.weight"] = _rand((dim, dim), gen, 0.08)
        tensors[f"{lp}.self_attn.k_proj.bias"] = _rand((dim,), gen, 0.02)
        tensors[f"{lp}.self_attn.v_proj.weight"] = _rand((dim, dim), gen, 0.08)
        tensors[f"{lp}.self_attn.v_proj.bias"] = _rand((dim,), gen, 0.02)
        tensors[f"{lp}.self_attn.out_proj.weight"] = _rand((dim, dim), gen, 0.08)
        tensors[f"{lp}.self_attn.out_proj.bias"] = _rand((dim,), gen, 0.02)
        tensors[f"{lp}.layer_norm2.weight"] = torch.ones((dim,), dtype=torch.float32) + _rand((dim,), gen, 0.1)
        tensors[f"{lp}.layer_norm2.bias"] = _rand((dim,), gen, 0.05)
        tensors[f"{lp}.mlp.fc1.weight"] = _rand((mlp_inner, dim), gen, 0.08)
        tensors[f"{lp}.mlp.fc1.bias"] = _rand((mlp_inner,), gen, 0.02)
        tensors[f"{lp}.mlp.fc2.weight"] = _rand((dim, mlp_inner), gen, 0.08)
        tensors[f"{lp}.mlp.fc2.bias"] = _rand((dim,), gen, 0.02)

    save_file(tensors, str(out_dir / "model.safetensors"))

    patches = _rand((n_patch, patch_chan), gen, 0.5)
    x = F.linear(
        patches,
        tensors[f"{pfx}embeddings.patch_embedding.weight"].view(dim, patch_chan),
        tensors[f"{pfx}embeddings.patch_embedding.bias"],
    )
    pos_grid = tensors[f"{pfx}embeddings.position_embedding.weight"].view(pos_side, pos_side, dim)
    pos_4d = pos_grid.permute(2, 0, 1).unsqueeze(0)
    pos_interp = F.interpolate(pos_4d, size=(h_grid, w_grid), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
    pos_mb = torch.stack([pos_interp[ry, rx] for ry, rx in coords], dim=0)
    x = x + pos_mb

    for li in range(2):
        lp = f"{pfx}encoder.layers.{li}"
        n1 = _layer_norm(x, tensors[f"{lp}.layer_norm1.weight"], tensors[f"{lp}.layer_norm1.bias"], eps)
        q = F.linear(n1, tensors[f"{lp}.self_attn.q_proj.weight"], tensors[f"{lp}.self_attn.q_proj.bias"])
        k = F.linear(n1, tensors[f"{lp}.self_attn.k_proj.weight"], tensors[f"{lp}.self_attn.k_proj.bias"])
        v = F.linear(n1, tensors[f"{lp}.self_attn.v_proj.weight"], tensors[f"{lp}.self_attn.v_proj.bias"])
        attn = _gqa_bidirectional(q, k, v, heads, heads, head_dim, coords=None)
        x = x + F.linear(attn, tensors[f"{lp}.self_attn.out_proj.weight"], tensors[f"{lp}.self_attn.out_proj.bias"])
        n2 = _layer_norm(x, tensors[f"{lp}.layer_norm2.weight"], tensors[f"{lp}.layer_norm2.bias"], eps)
        h_mlp = F.gelu(
            F.linear(n2, tensors[f"{lp}.mlp.fc1.weight"], tensors[f"{lp}.mlp.fc1.bias"]),
            approximate="tanh",
        )
        x = x + F.linear(h_mlp, tensors[f"{lp}.mlp.fc2.weight"], tensors[f"{lp}.mlp.fc2.bias"])

    x = _layer_norm(x, tensors[f"{pfx}post_layernorm.weight"], tensors[f"{pfx}post_layernorm.bias"], eps)
    merged = x.view(n_token, merged_dim)
    merged = _layer_norm(merged, tensors["multi_modal_projector.norm.weight"], tensors["multi_modal_projector.norm.bias"], eps)
    mid = F.gelu(
        F.linear(merged, tensors["multi_modal_projector.linear_1.weight"], tensors["multi_modal_projector.linear_1.bias"]),
        approximate="none",
    )
    out = F.linear(mid, tensors["multi_modal_projector.linear_2.weight"], tensors["multi_modal_projector.linear_2.bias"])

    (out_dir / "patches.f32").write_bytes(patches.numpy().tobytes())
    (out_dir / "expected.f32").write_bytes(out.contiguous().numpy().tobytes())
    meta = {"h": h_grid, "w": w_grid, "t": 1, "n_patch": n_patch, "n_token": n_token, "out_dim": out_dim}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out/vision_decision_oracle")
    args = parser.parse_args()
    root = Path(args.out)
    build_lfm3_oracle(root / "lfm3_vl_vision")
    build_siglip2_oracle(root / "siglip2_vision")
    print(f"Wrote M4 vision decision oracle fixtures to {root}")


if __name__ == "__main__":
    main()
