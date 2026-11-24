"""Unit and integration tests for the Bailing-MoE-v3 / Ling-3.2-air QWP1 packer adapter."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

QUANT_DIR = Path(__file__).resolve().parent
if str(QUANT_DIR) not in sys.path:
    sys.path.insert(0, str(QUANT_DIR))

from audit_release import audit_release  # noqa: E402
from bailing_adapter import quantize_bailing_model  # noqa: E402
from hq_blocks import dequant_fp8_blockwise  # noqa: E402
from pack import quantize_model  # noqa: E402
from shvq import dequant_any_blocks  # noqa: E402
from verify import audit_pack, read_header  # noqa: E402


def _make_fp8_pair(shape: tuple[int, int], seed: int, block: int = 128):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g, dtype=torch.float32) * 0.05
    n, k = shape
    sn, sk = (n + block - 1) // block, (k + block - 1) // block
    scales = torch.empty((sn, sk), dtype=torch.float32)
    q = torch.empty(shape, dtype=torch.float8_e4m3fn)
    for bi in range(sn):
        r0, r1 = bi * block, min(n, (bi + 1) * block)
        for bj in range(sk):
            c0, c1 = bj * block, min(k, (bj + 1) * block)
            tile = w[r0:r1, c0:c1]
            s = max(float(tile.abs().max().item()) / 448.0, 1e-6)
            scales[bi, bj] = s
            q[r0:r1, c0:c1] = (tile / s).to(torch.float8_e4m3fn)
    return q, scales


class BailingAdapterTest(unittest.TestCase):
    def test_quantize_synthetic_bailing_vl_kda_mla_moe_nextn_vision(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            src.mkdir()

            h = 128
            vocab = 256
            inter = 128
            q_heads = 2
            nope_d = 64
            rope_d = 32
            v_d = 64
            kv_rank = 64
            lin_v = 2
            lin_hd = 64
            n_exp = 4
            moe_inter = 64
            cfg = {
                "model_type": "bailing_moe_v3_vl",
                "tie_word_embeddings": False,
                "image_token_id": 250,
                "video_token_id": 251,
                "vision_start_token_id": 252,
                "vision_end_token_id": 253,
                "mrope_section": [4, 6, 6],
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 64,
                    "num_heads": 2,
                    "intermediate_size": 128,
                    "patch_size": 16,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "num_position_embeddings": 16,
                    "disable_merger_proj": True,
                },
                "text_config": {
                    "hidden_size": h,
                    "vocab_size": vocab,
                    "intermediate_size": inter,
                    "num_hidden_layers": 2,
                    "layer_group_size": 2,
                    "first_k_dense_replace": 1,
                    "num_attention_heads": q_heads,
                    "num_key_value_heads": q_heads,
                    "head_dim": lin_hd,
                    "q_lora_rank": None,
                    "kv_lora_rank": kv_rank,
                    "qk_nope_head_dim": nope_d,
                    "qk_rope_head_dim": rope_d,
                    "v_head_dim": v_d,
                    "use_mla_nope": False,
                    "gated_attention_proj_granularity_type": "head_wise",
                    "no_kda_lora": True,
                    "kda_safe_gate": True,
                    "kda_lower_bound": -5.0,
                    "linear_num_value_heads": lin_v,
                    "num_kv_heads_for_linear_attn": lin_v,
                    "short_conv_kernel_size": 4,
                    "num_experts": n_exp,
                    "num_experts_per_tok": 2,
                    "n_group": 2,
                    "topk_group": 1,
                    "moe_intermediate_size": moe_inter,
                    "moe_shared_expert_intermediate_size": moe_inter,
                    "routed_scaling_factor": 2.0,
                    "norm_topk_prob": True,
                    "score_function": "sigmoid",
                    "moe_router_enable_expert_bias": True,
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 100000.0,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 4096,
                    "eos_token_id": 2,
                    "num_nextn_predict_layers": 1,
                    "mtp_use_kda": False,
                },
            }
            (src / "config.json").write_text(json.dumps(cfg))
            (src / "tokenizer.json").write_text("{}")
            (src / "chat_template.jinja").write_text("{{ prompt }}")

            def bf(shape, val=0.02):
                return torch.full(shape, val, dtype=torch.bfloat16)

            tensors = {
                "model.word_embeddings.weight": bf((vocab, h), 0.01),
                "model.norm.weight": torch.ones(h, dtype=torch.bfloat16),
                "lm_head.weight": bf((vocab, h), -0.02),
                # Layer 0: KDA + Dense
                "model.layers.0.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
                "model.layers.0.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
                "model.layers.0.attention.q_proj.weight": bf((lin_v * lin_hd, h)),
                "model.layers.0.attention.k_proj.weight": bf((lin_v * lin_hd, h)),
                "model.layers.0.attention.v_proj.weight": bf((lin_v * lin_hd, h)),
                "model.layers.0.attention.f_proj.weight": bf((lin_v * lin_hd, h)),
                "model.layers.0.attention.g_proj.weight": bf((lin_v * lin_hd, h)),
                "model.layers.0.attention.b_proj.weight": bf((lin_v, h), 0.1),
                "model.layers.0.attention.q_conv1d.weight": bf((lin_v * lin_hd, 1, 4)),
                "model.layers.0.attention.k_conv1d.weight": bf((lin_v * lin_hd, 1, 4)),
                "model.layers.0.attention.v_conv1d.weight": bf((lin_v * lin_hd, 1, 4)),
                "model.layers.0.attention.o_norm.weight": torch.ones(lin_hd, dtype=torch.bfloat16),
                "model.layers.0.attention.A_log": torch.zeros(lin_v, dtype=torch.float32),
                "model.layers.0.attention.dt_bias": torch.zeros(lin_v * lin_hd, dtype=torch.float32),
                "model.layers.0.attention.o_proj.weight": bf((h, lin_v * lin_hd)),
                "model.layers.0.mlp.gate_proj.weight": bf((inter, h)),
                "model.layers.0.mlp.up_proj.weight": bf((inter, h)),
                "model.layers.0.mlp.down_proj.weight": bf((h, inter)),
            }

            idx_ramp = torch.arange(q_heads * (nope_d + rope_d) * h, dtype=torch.float32).reshape(
                q_heads, nope_d + rope_d, h)
            q1_raw = ((idx_ramp % 61) - 30.0) * 5e-4
            q1_raw[:, :nope_d, :] += 0.025
            q1_raw[:, nope_d:, :] -= 0.025

            def add_mla_and_moe(lidx: int, seed_base: int, fp8_gate: bool):
                p = f"model.layers.{lidx}."
                tensors[p + "input_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
                tensors[p + "post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
                tensors[p + "attention.q_proj.weight"] = q1_raw.reshape(q_heads * (nope_d + rope_d), h).to(torch.bfloat16)
                tensors[p + "attention.kv_a_proj_with_mqa.weight"] = bf((kv_rank + rope_d, h))
                tensors[p + "attention.kv_a_layernorm.weight"] = torch.ones(kv_rank, dtype=torch.bfloat16)
                tensors[p + "attention.kv_b_proj.weight"] = bf((q_heads * (nope_d + v_d), kv_rank))
                tensors[p + "attention.dense.weight"] = bf((h, q_heads * v_d))
                if fp8_gate:
                    gq, gs = _make_fp8_pair((q_heads, h), seed_base)
                    tensors[p + "attention.g_proj.weight"] = gq
                    tensors[p + "attention.g_proj.weight_scale_inv"] = gs
                else:
                    tensors[p + "attention.g_proj.weight"] = bf((q_heads, h), 0.25)
                tensors[p + "mlp.gate.weight"] = torch.ones((n_exp, h), dtype=torch.float32) * 0.01
                tensors[p + "mlp.gate.expert_bias"] = torch.full((n_exp,), 0.5, dtype=torch.float32)
                tensors[p + "mlp.shared_experts.gate_proj.weight"] = bf((moe_inter, h))
                tensors[p + "mlp.shared_experts.up_proj.weight"] = bf((moe_inter, h))
                tensors[p + "mlp.shared_experts.down_proj.weight"] = bf((h, moe_inter))
                for e in range(n_exp):
                    for proj, sh in (("gate_proj", (moe_inter, h)), ("up_proj", (moe_inter, h)), ("down_proj", (h, moe_inter))):
                        wq, ws = _make_fp8_pair(sh, seed_base + e * 10 + len(proj))
                        tensors[p + f"mlp.experts.{e}.{proj}.weight"] = wq
                        tensors[p + f"mlp.experts.{e}.{proj}.weight_scale_inv"] = ws

            add_mla_and_moe(1, 100, fp8_gate=False)
            tensors["model.layers.2.enorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            tensors["model.layers.2.hnorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            tensors["model.layers.2.eh_proj.weight"] = bf((h, 2 * h))
            tensors["model.layers.2.final_layernorm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            add_mla_and_moe(2, 500, fp8_gate=True)

            tensors["model.visual.patch_embed.proj.weight"] = bf((64, 3 * 2 * 16 * 16))
            tensors["linear_proj.0.weight"] = bf((h, 256), 0.03)
            tensors["linear_proj.0.bias"] = bf((h,), 0.0)
            tensors["linear_proj.2.weight"] = bf((h, h), 0.04)
            tensors["linear_proj.2.bias"] = bf((h,), 0.0)

            save_file(tensors, str(src / "model.safetensors"))

            quantize_model(
                source=src,
                destination=dst,
                dense_quant="hq8_g32",
                expert_quant="shvq3_g64",
                expert_down_quant="shvq3_g64",
            )

            info = audit_release(dst)
            self.assertTrue(info["ok"])
            qwp = dst / "model.qwp"
            header, data_start = read_header(qwp)
            audit_pack(qwp)
            ts = header["tensors"]
            self.assertEqual(header["provenance"]["model_id"], "bailing_moe_v3")
            self.assertIn("model.embed_tokens.weight", ts)
            self.assertEqual(ts["model.visual.merger.linear_fc1.weight"]["kind"], "bf16")
            self.assertEqual(ts["model.visual.merger.linear_fc2.bias"]["kind"], "bf16")
            self.assertEqual(ts["model.layers.2.attention.g_proj.weight"]["kind"], "bf16")
            self.assertEqual(ts["model.layers.1.mlp.gate.e_score_correction_bias"]["kind"], "f32")

            # Verify interleaved Q projection was reordered to planar [nope; rope]
            eq = ts["model.layers.1.attention.q_proj.weight"]
            with qwp.open("rb") as f:
                f.seek(eq["hq4_off"])
                n, k = eq["shape"]
                from shvq import any_block_bytes
                bb = any_block_bytes(8, 32, is_m5=False, is_shvq=False)
                raw = np.frombuffer(f.read(n * (k // 32) * bb), dtype=np.uint8).reshape(n, k // 32, bb)
                dq = dequant_any_blocks(raw, bits=8, group=32, is_m5=False, is_shvq=False)
            ref_nope = q1_raw[:, :nope_d, :].reshape(q_heads * nope_d, h).numpy()
            ref_rope = q1_raw[:, nope_d:, :].reshape(q_heads * rope_d, h).numpy()
            np.testing.assert_allclose(dq[:q_heads * nope_d], ref_nope, atol=2e-3, rtol=2e-2)
            np.testing.assert_allclose(dq[q_heads * nope_d:], ref_rope, atol=2e-3, rtol=2e-2)

            # Verify FP8 tile dequant -> BF16 roundtrip on layer 2 g_proj
            g2_ref = dequant_fp8_blockwise(
                tensors["model.layers.2.attention.g_proj.weight"],
                tensors["model.layers.2.attention.g_proj.weight_scale_inv"],
                block_size=128,
            ).to(torch.bfloat16).float().numpy()
            eg2 = ts["model.layers.2.attention.g_proj.weight"]
            with qwp.open("rb") as f:
                f.seek(eg2["raw_off"])
                u16 = np.frombuffer(f.read(q_heads * h * 2), dtype=np.uint16)
            g2_got = torch.from_numpy(u16.copy()).view(torch.bfloat16).float().numpy().reshape(q_heads, h)
            np.testing.assert_allclose(g2_got, g2_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
