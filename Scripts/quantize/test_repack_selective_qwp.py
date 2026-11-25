"""CPU unit tests for selective QWP1 wide-dense repacker (`Scripts/quantize/repack_selective_qwp.py`)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import torch.nn.functional as F

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS / "quantize"))
sys.path.insert(0, str(_SCRIPTS / "oracle"))

from ernie55_adapter import quantize_ernie55_model  # noqa: E402
from ernie55_dspark_oracle import build_synthetic_dspark_hf_checkpoint  # noqa: E402
from inkling2_adapter import maybe_force_cpu_quant  # noqa: E402
from pack import sha256  # noqa: E402
from repack_selective_qwp import (  # noqa: E402
    is_wide_dense_projection,
    main as repack_selective_main,
    repack_selective_qwp,
)
from test_repack_dspark_qwp import _read_unrotated  # noqa: E402
from verify import audit_pack, compare_planes, hash_region, planes, read_header  # noqa: E402


class TestRepackSelectiveQWP(unittest.TestCase):
    def test_selective_repack_dir_default_hq4_g32_m5_preserves_protected_planes(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_dir = root / "hf", root / "src_pack"
            tc = build_synthetic_dspark_hf_checkpoint(hf_dir, seed=71, bfloat16_router=True)
            quantize_ernie55_model(hf_dir, src_dir, device="cpu")
            out_dir = root / "out_selective"

            rc = repack_selective_main(["--src", str(src_dir), "--out", str(out_dir), "--device", "cpu"])
            self.assertEqual(rc, 0)
            src_qwp, out_qwp = src_dir / "model.qwp", out_dir / "model.qwp"
            self.assertLess(out_qwp.stat().st_size, src_qwp.stat().st_size)
            self.assertEqual((out_dir / "model.qwp.sha256").read_text().split()[0], sha256(out_qwp))
            for sib in ("dspark.qwp", "dspark.qwp.sha256", "flashindexer.qwp", "flashindexer.qwp.sha256"):
                self.assertEqual(sha256(out_dir / sib), sha256(src_dir / sib), sib)
            self.assertEqual(sha256(out_dir / "assets" / "manifest.json"), sha256(src_dir / "assets" / "manifest.json"))

            src_hdr, _ = read_header(src_qwp)
            out_hdr, _ = audit_pack(out_qwp)
            t_out = out_hdr["tensors"]
            self.assertEqual(out_hdr["provenance"]["selective_wide_dense_quant"], "hq4_g32_m5")

            wide_names = (
                "model.layers.0.self_attn.q_up_gate_proj.weight",
                "model.layers.0.self_attn.wo_a.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.1.self_attn.q_b_gate_idx_proj.weight",
                "model.layers.1.self_attn.o_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "model.layers.1.mlp.shared_experts.down_proj.weight",
            )
            for nm in wide_names:
                self.assertNotIn("hq", t_out[nm], nm)
                self.assertIn("hq4_off", t_out[nm], nm)
                cos = float(F.cosine_similarity(
                    _read_unrotated(src_qwp, nm).reshape(1, -1),
                    _read_unrotated(out_qwp, nm).reshape(1, -1),
                ).item())
                self.assertGreater(cos, 0.96, nm)

            protected_names = [
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.mhc_contract.weight",
                "model.layers.0.self_attn.fused_in_proj.weight",
                "model.layers.1.self_attn.q_a_proj.weight",
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                "model.layers.1.self_attn.indexer.wk.weight",
                "model.layers.1.self_attn.kv_b_proj.weight",
                "model.layers.1.mlp.fc1.weight",
                "model.layers.1.mlp.fc2.weight",
                "model.layers.1.mlp.gate.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.0.down_proj.weight",
            ]
            self.assertEqual(t_out["lm_head.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(t_out["model.layers.0.self_attn.fused_in_proj.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(t_out["model.layers.1.mlp.fc1.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(t_out["model.layers.1.mlp.fc2.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(t_out["model.layers.1.mlp.gate.weight"]["kind"], "bf16")
            self.assertEqual(len([k for k in t_out if ".experts." in k]), 3 * tc["n_routed_experts"])

            with src_qwp.open("rb") as sf, out_qwp.open("rb") as of:
                for nm in protected_names:
                    se, oe = src_hdr["tensors"][nm], t_out[nm]
                    self.assertEqual(planes(oe), planes(se), nm)
                    for key, length, _ in planes(oe):
                        self.assertEqual(hash_region(of, oe[key], length), hash_region(sf, se[key], length), nm)
            compare_planes(out_qwp, src_qwp, allow_requantized=set(wide_names))

    def test_hq6_g32_standalone_qwp_and_classifier_coverage(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_dir = root / "hf", root / "src_pack"
            build_synthetic_dspark_hf_checkpoint(hf_dir, seed=82, bfloat16_router=True)
            quantize_ernie55_model(hf_dir, src_dir, device="cpu")
            out_qwp = root / "model_hq6.qwp"

            res = repack_selective_qwp(
                src_dir / "model.qwp", out_qwp, wide_dense_quant="hq6_g32", device="cpu",
            )
            self.assertLess(res["out_bytes"], res["src_bytes"])
            self.assertEqual(res["wide_dense_quant"], "hq6_g32")
            self.assertEqual(out_qwp.with_name(out_qwp.name + ".sha256").read_text().split()[0], sha256(out_qwp))
            hdr, _ = audit_pack(out_qwp)
            self.assertEqual(hdr["tensors"]["model.layers.0.self_attn.q_up_gate_proj.weight"]["hq"], {"bits": 6, "group": 32})
            self.assertEqual(hdr["tensors"]["model.layers.0.self_attn.fused_in_proj.weight"]["hq"], {"bits": 8, "group": 32})
            cos = float(F.cosine_similarity(
                _read_unrotated(src_dir / "model.qwp", "model.layers.1.self_attn.o_proj.weight").reshape(1, -1),
                _read_unrotated(out_qwp, "model.layers.1.self_attn.o_proj.weight").reshape(1, -1),
            ).item())
            self.assertGreater(cos, 0.995)

        q2d = {"shape": [128, 128], "hq4_off": 256}
        self.assertTrue(is_wide_dense_projection("model.layers.41.self_attn.q_b_proj.weight", q2d))
        self.assertTrue(is_wide_dense_projection("model.layers.1.mlp.shared_experts.gate_up_proj.weight", q2d))
        self.assertTrue(is_wide_dense_projection("model.layers.0.mlp.gate_up_proj.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.0.self_attn.in_proj.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.0.self_attn.linear_q_down_proj.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.0.self_attn.linear_kv_proj.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.1.mlp.fc1.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.1.mlp.fc2.weight", q2d))
        self.assertFalse(is_wide_dense_projection("dspark.layers.0.self_attn.o_proj.weight", q2d))
        self.assertFalse(is_wide_dense_projection("model.layers.1.mlp.experts.down_proj", {"shape": [4, 64, 64], "experts": 4}))

    def test_keep_mode_and_error_guards(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_dir = root / "hf", root / "src_pack"
            build_synthetic_dspark_hf_checkpoint(hf_dir, seed=93, bfloat16_router=True)
            quantize_ernie55_model(hf_dir, src_dir, device="cpu")
            kept_qwp = root / "model_kept.qwp"
            res_k = repack_selective_qwp(src_dir / "model.qwp", kept_qwp, wide_dense_quant="keep", device="cpu")
            self.assertEqual(res_k["requantized_wide_tensors"], 0)
            compare_planes(kept_qwp, src_dir / "model.qwp")

            with self.assertRaises(ValueError):
                repack_selective_qwp(src_dir, src_dir, device="cpu")
            with self.assertRaises(ValueError):
                repack_selective_qwp(src_dir / "model.qwp", src_dir / "model.qwp", device="cpu")
            with self.assertRaises(FileExistsError):
                repack_selective_qwp(src_dir, kept_qwp.parent, device="cpu")
            with self.assertRaises(FileNotFoundError):
                repack_selective_qwp(root / "missing", root / "out.qwp", device="cpu")


if __name__ == "__main__":
    unittest.main()
