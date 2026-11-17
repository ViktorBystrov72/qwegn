"""Unit tests for Scripts/quantize/pack_dflash.py (DFlash 2 & DSpine packer)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from pack_dflash import (
    derive_dflash_model_id,
    is_dflash_mlp_proj,
    is_dflash_raw_plane,
    quantize_dflash_model,
)
from shvq import dequant_shvq_blocks
from verify import audit_pack, read_header


def _make_dflash2_checkpoint(
    src: Path,
    *,
    hidden: int = 128,
    inter: int = 256,
    vocab: int = 64,
    rank: int = 32,
    extra_config: dict | None = None,
    dspine: bool = False,
    version_val: int = 1,
) -> Path:
    src.mkdir(parents=True, exist_ok=True)
    dcfg: dict = {
        "block_size": 16 if dspine else 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 15,
        "selector_rank": rank,
        "selector_top_k": 16,
        "target_layer_ids": [1, 3],
    }
    if dspine:
        dcfg["selected_prefix_code_bridge"] = {"code_size": 64, "rank": 64}
    cfg = {
        "architectures": ["DFlashDraftModel" if dspine else "DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "intermediate_size": inter,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "dflash_config": dcfg,
    }
    if extra_config:
        cfg.update(extra_config)
    (src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (src / "README.md").write_text("# Tiny DFlash2 Test\n", encoding="utf-8")

    g = torch.Generator().manual_seed(42)
    tensors = {
        "fc.weight": torch.randn(hidden, hidden * 2, generator=g, dtype=torch.bfloat16) * 0.02,
        "hidden_norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "layers.0.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "layers.0.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "layers.0.self_attn.q_norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "layers.0.self_attn.k_norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "layers.0.self_attn.q_proj.weight": torch.randn(128, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.self_attn.k_proj.weight": torch.randn(64, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.self_attn.v_proj.weight": torch.randn(64, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.self_attn.o_proj.weight": torch.randn(hidden, 128, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.mlp.gate_proj.weight": torch.randn(inter, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.mlp.up_proj.weight": torch.randn(inter, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.mlp.down_proj.weight": torch.randn(hidden, inter, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.attention_conv.base_kernel": torch.ones(2, 2, hidden, dtype=torch.bfloat16),
        "layers.0.attention_conv.kernel_projection.weight": torch.randn(32, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
        "layers.0.mlp_conv.base_kernel": torch.ones(2, 2, hidden, dtype=torch.bfloat16),
        "layers.0.mlp_conv.kernel_projection.weight": torch.randn(32, hidden, generator=g, dtype=torch.bfloat16) * 0.02,
    }
    if dspine:
        tensors["selected_prefix_bridge.codebook"] = torch.randn(64, 64, generator=g, dtype=torch.bfloat16)
        tensors["draft_block_causal_version"] = torch.tensor(version_val, dtype=torch.int64)
    else:
        tensors["candidate_selector.hidden_projection.weight"] = torch.randn(rank, hidden, generator=g, dtype=torch.bfloat16) * 0.02
        tensors["candidate_selector.predecessor_codebook"] = torch.randn(vocab, rank, generator=g, dtype=torch.bfloat16) * 0.02
        tensors["candidate_selector.successor_codebook"] = torch.randn(vocab, rank, generator=g, dtype=torch.bfloat16) * 0.02
    save_file(tensors, str(src / "model.safetensors"))
    return src


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_pack_integrity(test: unittest.TestCase, pack_dir: Path) -> tuple[dict, dict]:
    qwp = pack_dir / "model.qwp"
    test.assertTrue(qwp.is_file())
    hdr, counts = audit_pack(qwp)
    sha_line = (pack_dir / "model.qwp.sha256").read_text().strip().split()
    test.assertEqual(len(sha_line), 2)
    test.assertEqual(sha_line[1], "model.qwp")
    test.assertEqual(sha_line[0], _sha256_file(qwp))
    manifest_path = pack_dir / "assets" / "manifest.json"
    test.assertEqual(hdr["provenance"]["assets_manifest_sha256"], _sha256_file(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    test.assertIn("config.json", manifest)
    for fname, meta in manifest.items():
        fpath = pack_dir / "assets" / fname
        test.assertTrue(fpath.is_file())
        test.assertFalse(fpath.is_symlink())
        test.assertEqual(fpath.stat().st_size, meta["bytes"])
        test.assertEqual(_sha256_file(fpath), meta["sha256"])
    return hdr, counts


class TestPackDFlash(unittest.TestCase):
    def test_derive_model_id_from_hf_snapshot_and_config(self):
        snap = Path("/tmp/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/" + "a" * 40)
        self.assertEqual(
            derive_dflash_model_id({"model_type": "qwen3"}, snap, is_dspine=False),
            "dflash2_qwen3_8_flash_next",
        )
        self.assertEqual(
            derive_dflash_model_id({"model_type": "qwen3"}, snap, is_dspine=True),
            "dspine_qwen3_8_flash_next",
        )
        cfg_named = {"_name_or_path": "incoai/Qwen3.8-27B-DFlash2", "model_type": "qwen3"}
        self.assertEqual(
            derive_dflash_model_id(cfg_named, Path("/tmp/generic"), is_dspine=False),
            "dflash2_qwen3_8_27b",
        )
        self.assertEqual(
            derive_dflash_model_id({"model_type": "qwen3"}, Path("/tmp/DFlash2"), is_dspine=False),
            "dflash2_qwen3",
        )

    def test_plane_classifiers(self):
        for raw_name in (
            "hidden_norm.weight",
            "norm.weight",
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.attention_conv.base_kernel",
            "layers.0.attention_conv.kernel_projection.weight",
            "layers.0.mlp_conv.base_kernel",
            "layers.0.mlp_conv.kernel_projection.weight",
            "selected_prefix_bridge.codebook",
        ):
            self.assertTrue(is_dflash_raw_plane(raw_name), raw_name)
            self.assertFalse(is_dflash_mlp_proj(raw_name), raw_name)
        for mlp_name in (
            "layers.0.mlp.gate_proj.weight",
            "layers.4.mlp.up_proj.weight",
            "layers.2.mlp.down_proj.weight",
            "mlp.down_proj.weight",
        ):
            self.assertFalse(is_dflash_raw_plane(mlp_name), mlp_name)
            self.assertTrue(is_dflash_mlp_proj(mlp_name), mlp_name)
        for attn_name in (
            "fc.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
        ):
            self.assertFalse(is_dflash_raw_plane(attn_name), attn_name)
            self.assertFalse(is_dflash_mlp_proj(attn_name), attn_name)

    def test_pack_uniform_hq4_g32_m5(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _make_dflash2_checkpoint(root / "Qwen3.8-Flash-Next-DFlash2")
            dst = root / "out-hq4"
            stats = quantize_dflash_model(src, dst, quant="hq4_g32_m5")
            self.assertEqual(stats, {"hq4_g32_m5": 8, "bf16": 13})
            hdr, counts = _assert_pack_integrity(self, dst)
            self.assertEqual(counts, {"hq4_g32_m5": 8, "bf16": 13})
            self.assertEqual(hdr["provenance"]["model_id"], "dflash2_qwen3_8_flash_next")
            self.assertEqual(hdr["provenance"]["quant_recorded"], "hq4_g32_m5")
            self.assertNotIn("hq", hdr["tensors"]["fc.weight"])
            self.assertNotIn("hq", hdr["tensors"]["layers.0.mlp.gate_proj.weight"])

    def test_pack_mixed_shvq3_and_mlp_shvq15(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snap = root / "models--incoai--Qwen3.8-Flash-Next-DFlash2" / "snapshots" / ("b" * 40)
            src = _make_dflash2_checkpoint(snap)
            dst = root / "out-mixed"
            stats = quantize_dflash_model(
                src, dst, quant="shvq3_g64", mlp_quant="shvq15_g64"
            )
            self.assertEqual(stats, {"shvq3_g64": 5, "shvq15_g64": 3, "bf16": 13})
            hdr, counts = _assert_pack_integrity(self, dst)
            self.assertEqual(counts, {"hq4_g32_m5": 0, "bf16": 13, "shvq3_g64": 5, "shvq15_g64": 3})
            self.assertEqual(hdr["provenance"]["model_id"], "dflash2_qwen3_8_flash_next")
            self.assertEqual(
                hdr["provenance"]["quant_recorded"],
                "shvq3_g64__mlp_shvq15_g64",
            )
            for name in (
                "fc.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
            ):
                self.assertEqual(
                    hdr["tensors"][name]["hq"],
                    {"bits": 3, "group": 64, "vq": "shvq1"},
                )
            for name in (
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.down_proj.weight",
            ):
                self.assertEqual(
                    hdr["tensors"][name]["hq"],
                    {"bits": 15, "group": 64, "vq": "shvq1"},
                )
            # Verify SHVQ15 block payload length and dequantization roundtrip.
            qwp_bytes = (dst / "model.qwp").read_bytes()
            gate_e = hdr["tensors"]["layers.0.mlp.gate_proj.weight"]
            n_rows, k_cols = gate_e["shape"]
            bbytes = 2 + (64 * 3) // 16
            raw = np.frombuffer(
                qwp_bytes, np.uint8, count=n_rows * (k_cols // 64) * bbytes, offset=gate_e["hq4_off"]
            ).reshape(n_rows, k_cols // 64, bbytes)
            dq = dequant_shvq_blocks(raw, bits=15, group=64)
            self.assertEqual(dq.shape, (n_rows, k_cols))
            self.assertTrue(np.all(np.isfinite(dq)))

    def test_pack_same_mlp_quant_collapses_quant_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _make_dflash2_checkpoint(root / "Qwen3.8-Flash-Next-DFlash2")
            dst = root / "out-same"
            stats = quantize_dflash_model(
                src, dst, quant="shvq3_g64", mlp_quant="shvq3_g64"
            )
            self.assertEqual(stats, {"shvq3_g64": 8, "bf16": 13})
            hdr, _ = _assert_pack_integrity(self, dst)
            self.assertEqual(hdr["provenance"]["quant_recorded"], "shvq3_g64")

    def test_pack_dspine_and_version_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _make_dflash2_checkpoint(root / "Qwen3.8-27B-DSpine-b16", dspine=True)
            dst = root / "out-dspine"
            stats = quantize_dflash_model(src, dst, quant="shvq3_g64", mlp_quant="shvq15_g64")
            self.assertEqual(stats["shvq3_g64"], 5)
            self.assertEqual(stats["shvq15_g64"], 3)
            hdr, _ = _assert_pack_integrity(self, dst)
            self.assertEqual(hdr["provenance"]["model_id"], "dspine_qwen3_8_27b_dspine_b16")
            self.assertNotIn("draft_block_causal_version", hdr["tensors"])

            bad_src = _make_dflash2_checkpoint(root / "BadDSpine", dspine=True, version_val=2)
            with self.assertRaisesRegex(ValueError, "unsupported draft_block_causal_version"):
                quantize_dflash_model(bad_src, root / "out-bad", quant="hq4_g32_m5")

    def test_cli_mlp_quant_and_rejection_guards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _make_dflash2_checkpoint(root / "Qwen3.8-Flash-Next-DFlash2")
            dst = root / "cli-out"
            script = Path(__file__).resolve().parent / "pack_dflash.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--model",
                    str(src),
                    "--out",
                    str(dst),
                    "--quant",
                    "shvq3_g64",
                    "--mlp-quant",
                    "shvq15_g64",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("5 SHVQ3_G64 + 3 SHVQ15_G64 + 13 BF16", proc.stdout)
            hdr, _ = _assert_pack_integrity(self, dst)
            self.assertEqual(hdr["provenance"]["quant_recorded"], "shvq3_g64__mlp_shvq15_g64")

            with self.assertRaises(FileExistsError):
                quantize_dflash_model(src, dst, quant="hq4_g32_m5")
            with self.assertRaisesRegex(ValueError, "not a .qwp file"):
                quantize_dflash_model(src, root / "bad.qwp", quant="hq4_g32_m5")


if __name__ == "__main__":
    unittest.main()
