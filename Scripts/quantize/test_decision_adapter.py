"""Unit tests for generalized Decision Model & LFM QWP1 packer adapters."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pack import quantize_model
from verify import read_header, verify_assets


def _write_minimal_tokenizer(src: Path) -> None:
    (src / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {"<unk>": 0, "yes": 1, "no": 2}, "merges": []},
        "added_tokens": [],
    }), encoding="utf-8")


def _build_lfm3_vl_checkpoint(src: Path) -> None:
    h, v, inter = 64, 128, 128
    cfg = {
        "architectures": ["Lfm3VlForConditionalGeneration"],
        "model_type": "lfm3_vl",
        "hidden_size": h,
        "vocab_size": v,
        "intermediate_size": inter,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "layer_types": ["conv", "full_attention"],
        "short_conv_kernel_size": 7,
        "decision_config": {
            "framing": "turnVerbalizer",
            "attentionTopology": "packedSet",
            "headKind": "vocabVerbalizer",
            "hasStatePointer": True,
        },
    }
    (src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _write_minimal_tokenizer(src)
    torch.manual_seed(11)
    tensors = {
        "model.embed_tokens.weight": torch.randn(v, h, dtype=torch.bfloat16) * 0.02,
        "model.norm.weight": torch.ones(h, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(v, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.0.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        "model.layers.0.shortconv.conv_L.weight": torch.randn(h, 1, 7, dtype=torch.bfloat16) * 0.1,
        "model.layers.0.shortconv.in_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.0.shortconv.out_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.1.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.1.self_attn.k_proj.weight": torch.randn(32, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.1.self_attn.v_proj.weight": torch.randn(32, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.1.self_attn.o_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.1.mlp.down_proj.weight": torch.randn(h, inter, dtype=torch.bfloat16) * 0.02,
        "model.state_pointer.q_start.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
        "model.state_pointer.q_end.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.02,
    }
    save_file(tensors, str(src / "model.safetensors"))


def _build_omni_encoder_checkpoint(src: Path) -> None:
    h, v = 64, 128
    cfg = {
        "architectures": ["D1OmniForDecision"],
        "model_type": "d1_omni",
        "hidden_size": h,
        "vocab_size": v,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "layer_norm_eps": 1e-5,
        "decision_config": {
            "framing": "optionMarkers",
            "attentionTopology": "bidirectionalEncoder",
            "headKind": "markerScorer",
        },
    }
    (src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _write_minimal_tokenizer(src)
    torch.manual_seed(23)
    tensors = {
        "model.embeddings.word_embeddings.weight": torch.randn(v, h, dtype=torch.bfloat16) * 0.02,
        "model.layers.0.norm1.weight": torch.ones(h, dtype=torch.bfloat16),
        "model.layers.0.norm1.bias": torch.zeros(h, dtype=torch.bfloat16),
        "model.layers.0.shortconv.conv.weight": torch.randn(h, 1, 3, dtype=torch.bfloat16) * 0.1,
        "marker_head.type_emb": torch.randn(h, dtype=torch.bfloat16) * 0.02,
        "marker_head.mlp.0.weight": torch.randn(32, h, dtype=torch.bfloat16) * 0.05,
        "marker_head.mlp.0.bias": torch.zeros(32, dtype=torch.bfloat16),
        "classifier.weight": torch.randn(4, h, dtype=torch.bfloat16) * 0.05,
    }
    save_file(tensors, str(src / "model.safetensors"))


class DecisionAdapterTests(unittest.TestCase):
    def test_lfm3_vl_preserves_small_decision_bf16_and_squeezes_conv1d(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "src", root / "pack"
            src.mkdir()
            _build_lfm3_vl_checkpoint(src)
            prev = os.environ.pop("QW_DECISION_FORCE_HQ4", None)
            try:
                quantize_model(src, out)
            finally:
                if prev is not None:
                    os.environ["QW_DECISION_FORCE_HQ4"] = prev

            weights = out / "model.qwp"
            self.assertEqual(weights.read_bytes()[:4], b"QWP1")
            header, _ = read_header(weights)
            self.assertEqual(verify_assets(out, header), 3)
            self.assertEqual(header["provenance"]["model_id"], "lfm3_vl")
            self.assertEqual(header["provenance"]["quant_recorded"], "bf16")
            conv_entry = header["tensors"]["model.layers.0.shortconv.conv_L.weight"]
            self.assertEqual(conv_entry["shape"], [64, 7])
            self.assertIn("raw_off", conv_entry)
            self.assertEqual(conv_entry["kind"], "bf16")
            self.assertIn("raw_off", header["tensors"]["model.state_pointer.q_start.weight"])
            self.assertIn("raw_off", header["tensors"]["model.layers.1.self_attn.q_proj.weight"])

    def test_force_hq4_quantizes_linear_while_keeping_conv_and_heads_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "src", root / "pack_hq4"
            src.mkdir()
            _build_lfm3_vl_checkpoint(src)
            prev = os.environ.get("QW_DECISION_FORCE_HQ4")
            os.environ["QW_DECISION_FORCE_HQ4"] = "1"
            try:
                quantize_model(src, out)
            finally:
                if prev is None:
                    os.environ.pop("QW_DECISION_FORCE_HQ4", None)
                else:
                    os.environ["QW_DECISION_FORCE_HQ4"] = prev

            header, _ = read_header(out / "model.qwp")
            self.assertIn("hq4_off", header["tensors"]["model.layers.1.self_attn.q_proj.weight"])
            self.assertIn("raw_off", header["tensors"]["model.layers.0.shortconv.conv_L.weight"])
            self.assertIn("raw_off", header["tensors"]["model.state_pointer.q_start.weight"])
            self.assertIn("raw_off", header["tensors"]["lm_head.weight"])

    def test_omni_encoder_checkpoint_packs_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "src", root / "pack_omni"
            src.mkdir()
            _build_omni_encoder_checkpoint(src)
            quantize_model(src, out)
            header, _ = read_header(out / "model.qwp")
            self.assertEqual(verify_assets(out, header), 3)
            self.assertEqual(header["provenance"]["model_id"], "d1_omni")
            conv_entry = header["tensors"]["model.layers.0.shortconv.conv.weight"]
            self.assertEqual(conv_entry["shape"], [64, 3])
            self.assertIn("raw_off", conv_entry)
            self.assertIn("raw_off", header["tensors"]["marker_head.mlp.0.weight"])

    def test_real_lfm3_vl_parent_d1_keys_and_calibration_preserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out_bf16, out_hq4 = root / "src", root / "pack_bf16", root / "pack_hq4"
            src.mkdir()
            import json
            import torch
            from safetensors.torch import save_file
            cfg = {
                "architectures": ["D1Model"],
                "auto_map": {"AutoModel": "modeling_d1_lfm3_vl.D1Model"},
                "model_type": "lfm3_vl",
                "d1_codes": 2,
                "d1_attention": "set",
                "text_config": {
                    "model_type": "lfm3",
                    "vocab_size": 256,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "layer_types": ["conv", "full_attention"],
                    "conv_L_cache": 7,
                    "rope_parameters": {
                        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5, "rope_type": "default"}
                    },
                    "norm_eps": 1e-5,
                    "use_sandwich_norm": True,
                },
            }
            (src / "config.json").write_text(json.dumps(cfg))
            (src / "tokenizer.json").write_text('{"model":{"vocab":{},"merges":[]},"added_tokens":[]}')
            (src / "calibration.json").write_text('{"temperatures":{"noul:2":1.0},"tables":{}}')
            tensors = {
                "model.language_model.embed_tokens.weight": torch.randn(256, 64, dtype=torch.bfloat16) * 0.02,
                "model.language_model.layers.0.conv.in_proj.weight": torch.randn(192, 64, dtype=torch.bfloat16) * 0.05,
                "model.language_model.layers.0.conv.conv.weight": torch.randn(64, 1, 7, dtype=torch.bfloat16) * 0.05,
                "model.language_model.layers.0.conv.out_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.05,
                "model.language_model.layers.1.self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16) * 0.05,
                "model.language_model.layers.1.self_attn.gate_proj.weight": torch.randn(4, 64, dtype=torch.bfloat16) * 0.05,
            }
            save_file(tensors, str(src / "model.safetensors"))
            quantize_model(src, out_bf16)
            h_bf16, _ = read_header(out_bf16 / "model.qwp")
            self.assertEqual(h_bf16["provenance"]["model_id"], "lfm3")
            self.assertEqual(h_bf16["provenance"]["quant_recorded"], "bf16")
            self.assertTrue((out_bf16 / "assets" / "calibration.json").is_file())
            self.assertEqual(h_bf16["tensors"]["model.language_model.layers.0.conv.conv.weight"]["shape"], [64, 7])
            quantize_model(src, out_hq4, dense_quant="hq4_g32_m5")
            h_hq4, _ = read_header(out_hq4 / "model.qwp")
            self.assertIn("hq4_off", h_hq4["tensors"]["model.language_model.layers.0.conv.in_proj.weight"])
            self.assertIn("raw_off", h_hq4["tensors"]["model.language_model.layers.0.conv.conv.weight"])
            self.assertIn("raw_off", h_hq4["tensors"]["model.language_model.layers.1.self_attn.gate_proj.weight"])


if __name__ == "__main__":
    unittest.main()
