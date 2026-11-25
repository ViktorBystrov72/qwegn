"""CPU unit tests for local DSpark QWP1 repacker (`Scripts/quantize/repack_dspark_qwp.py`)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS / "quantize"))
sys.path.insert(0, str(_SCRIPTS / "oracle"))

from ernie55_adapter import quantize_ernie55_model  # noqa: E402
from ernie55_dspark_oracle import build_synthetic_dspark_hf_checkpoint  # noqa: E402
from hq_blocks import dequant_hq8_g32_m5  # noqa: E402
from inkling2_adapter import maybe_force_cpu_quant  # noqa: E402
from pack import sha256  # noqa: E402
from repack_dspark_qwp import main as repack_dspark_main, repack_dspark_qwp  # noqa: E402
from shvq import any_block_bytes, dequant_any_blocks  # noqa: E402
from verify import audit_pack, compare_planes, hash_region, planes, read_header  # noqa: E402


def _read_unrotated(pack_path: Path, name: str) -> torch.Tensor:
    hdr, _ = read_header(pack_path)
    e = hdr["tensors"][name]
    shape = [int(d) for d in e["shape"]]
    k = shape[-1]
    n_rows = int(np.prod(shape[:-1]))
    with pack_path.open("rb") as fp:
        if "hq8_off" in e:
            fp.seek(int(e["hq8_off"]))
            qs = np.frombuffer(fp.read(n_rows * k), dtype=np.int8).reshape(n_rows, k).copy()
            fp.seek(int(e["hq8_s_off"]))
            sc = np.frombuffer(fp.read(n_rows * (k // 32) * 2), dtype=np.float16).reshape(n_rows, k // 32).copy()
            return torch.from_numpy(dequant_hq8_g32_m5(qs, sc)).reshape(shape).float()
        hq = e.get("hq")
        if hq is None:
            b, g, is_m5, is_shvq = 4, 32, True, False
        else:
            b, g, is_m5, is_shvq = int(hq["bits"]), int(hq["group"]), False, (hq.get("vq") == "shvq1")
        bbytes = any_block_bytes(b, g, is_m5=is_m5, is_shvq=is_shvq)
        nbytes = n_rows * (k // g) * bbytes
        fp.seek(int(e["hq4_off"]))
        blk = np.frombuffer(fp.read(nbytes), dtype=np.uint8).reshape(n_rows, k // g, bbytes).copy()
        w = dequant_any_blocks(blk, bits=b, group=g, is_m5=is_m5, is_shvq=is_shvq)
        return torch.from_numpy(w).reshape(shape).float()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


class TestRepackDSparkQWP(unittest.TestCase):
    def test_default_repack_bundles_draft_head_and_preserves_rotation(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_pack = root / "hf", root / "src_pack"
            tc = build_synthetic_dspark_hf_checkpoint(hf_dir, seed=41, bfloat16_router=True)
            quantize_ernie55_model(hf_dir, src_pack, device="cpu")
            src_ds = src_pack / "dspark.qwp"
            src_model = src_pack / "model.qwp"
            out_ds = root / "out" / "dspark_repacked.qwp"

            rc = repack_dspark_main([
                "--src", str(src_ds),
                "--out", str(out_ds),
                "--draft-head-from", str(src_model),
                "--draft-head-quant", "hq4_g32_m5",
                "--device", "cpu",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_ds.is_file())
            self.assertEqual(out_ds.with_name(out_ds.name + ".sha256").read_text().split()[0], sha256(out_ds))

            src_hdr, _ = read_header(src_ds)
            out_hdr, _ = audit_pack(out_ds)
            t_out = out_hdr["tensors"]

            # Provenance + draft head assertions (default keeps expert_down at shvq15_g64).
            prov = out_hdr["provenance"]
            self.assertEqual(prov["dspark_dense_quant"], "hq4_g32_m5")
            self.assertEqual(prov["dspark_expert_down_quant"], "shvq15_g64")
            self.assertEqual(prov["dspark_expert_quant"], "shvq15_g64")
            self.assertEqual(prov["dspark_markov_w2_quant"], "hq8_g32")
            self.assertEqual(prov["dspark_draft_head_quant"], "hq4_g32_m5")
            self.assertEqual(prov["dspark_reference"], str(src_ds.resolve()))
            self.assertEqual(t_out["draft_head.weight"]["shape"], [tc["vocab_size"], tc["hidden_size"]])
            self.assertNotIn("hq", t_out["draft_head.weight"])
            self.assertIn("hq4_off", t_out["draft_head.weight"])

            # Dense projections switched to hq4_g32_m5 (no hq metadata key).
            for dense_name in (
                "dspark.main_proj.weight",
                "dspark.layers.0.self_attn.linear_kv_proj.weight",
                "dspark.layers.0.self_attn.o_proj.weight",
                "dspark.layers.1.mlp.fc1.weight",
                "dspark.layers.1.mlp.shared_experts.down_proj.weight",
            ):
                self.assertNotIn("hq", t_out[dense_name], dense_name)
                self.assertIn("hq4_off", t_out[dense_name], dense_name)

            # 3D expert down_proj and gate_up_proj both stay shvq15_g64 verbatim by default.
            for lid in (0, 1):
                dn = t_out[f"dspark.layers.{lid}.mlp.experts.down_proj"]
                self.assertEqual(dn["experts"], tc["n_routed_experts"])
                self.assertEqual(dn["hq"], {"bits": 15, "group": 64, "vq": "shvq1"})
                self.assertEqual(t_out[f"dspark.layers.{lid}.mlp.experts.gate_up_proj"]["hq"],
                                 {"bits": 15, "group": 64, "vq": "shvq1"})

            # Markov, confidence_head, routers, norms, mhc, and both 3D expert stacks must be byte-exact.
            must_match = [
                "dspark.markov_w1.weight",
                "dspark.markov_w2.weight",
                "dspark.confidence_head.weight",
                "dspark.main_norm.weight",
                "dspark.norm.weight",
                "dspark.mhc_contract.weight",
                "dspark.layers.0.mlp.gate.weight",
                "dspark.layers.0.mlp.experts.gate_up_proj",
                "dspark.layers.0.mlp.experts.down_proj",
                "dspark.layers.1.mlp.experts.gate_up_proj",
                "dspark.layers.1.mlp.experts.down_proj",
            ]
            with src_ds.open("rb") as sf, out_ds.open("rb") as of:
                for nm in must_match:
                    se, oe = src_hdr["tensors"][nm], t_out[nm]
                    self.assertEqual(planes(oe), planes(se), nm)
                    for key, length, _ in planes(oe):
                        self.assertEqual(hash_region(of, oe[key], length), hash_region(sf, se[key], length), nm)

            # Rotation semantics (isRotated = true): unrotated dequant cosine stays high across 2D + draft_head.
            self.assertGreater(_cos(_read_unrotated(src_ds, "dspark.main_proj.weight"),
                                    _read_unrotated(out_ds, "dspark.main_proj.weight")), 0.96)
            self.assertGreater(_cos(_read_unrotated(src_model, "lm_head.weight"),
                                    _read_unrotated(out_ds, "draft_head.weight")), 0.96)

    def test_byte_reduction_and_keep_variants_compare_planes(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_pack = root / "hf", root / "src_pack"
            build_synthetic_dspark_hf_checkpoint(hf_dir, seed=52, bfloat16_router=True)
            quantize_ernie55_model(
                hf_dir, src_pack, device="cpu",
                dspark_dense_quant="shvq4_g32",
                dspark_expert_quant="shvq4_g64",
                dspark_expert_down_quant="shvq4_g64",
            )
            src_ds = src_pack / "dspark.qwp"
            src_model = src_pack / "model.qwp"

            # Compressing shvq4_g64 expert down_proj -> shvq2_g64 shrinks pack bytes and passes compare_planes.
            shrunk = root / "dspark_shrunk.qwp"
            res = repack_dspark_qwp(
                src_ds, shrunk, expert_down_quant="shvq2_g64", dense_quant="keep", device="cpu",
            )
            self.assertLess(res["out_bytes"], res["src_bytes"])
            self.assertGreater(res["saved_bytes"], 0)
            self.assertGreater(res["compared_bytes"], 0)
            compare_planes(shrunk, src_ds, allow_requantized=res["skipped_diff_planes"])
            self.assertGreater(_cos(_read_unrotated(src_ds, "dspark.layers.0.mlp.experts.down_proj"),
                                    _read_unrotated(shrunk, "dspark.layers.0.mlp.experts.down_proj")), 0.85)

            # Full-keep pass yields verbatim planes across every tensor.
            kept = root / "dspark_kept.qwp"
            res_keep = repack_dspark_qwp(
                src_ds, kept, expert_down_quant="keep", dense_quant="keep", device="cpu",
            )
            self.assertEqual(res_keep["requantized_tensors"], 0)
            compare_planes(kept, src_ds)

            # Replacing an existing bundled draft_head with shvq2_g64, hq4_g32_m5, keep (hq8_g32), or bf16 works cleanly.
            with_head = root / "dspark_head_shvq2.qwp"
            repack_dspark_qwp(
                shrunk, with_head, expert_down_quant="keep", dense_quant="keep",
                draft_head_from=src_model, draft_head_quant="shvq2_g64", device="cpu",
            )
            hdr_h2, _ = audit_pack(with_head)
            self.assertEqual(hdr_h2["tensors"]["draft_head.weight"]["hq"], {"bits": 2, "group": 64, "vq": "shvq1"})
            self.assertGreater(_cos(_read_unrotated(src_model, "lm_head.weight"),
                                    _read_unrotated(with_head, "draft_head.weight")), 0.85)

            replaced_bf16 = root / "dspark_head_bf16.qwp"
            repack_dspark_qwp(
                with_head, replaced_bf16, expert_down_quant="keep", dense_quant="keep",
                draft_head_from=src_model, draft_head_quant="bf16", device="cpu",
            )
            hdr_bf16, _ = audit_pack(replaced_bf16)
            self.assertEqual(hdr_bf16["tensors"]["draft_head.weight"]["kind"], "bf16")

            replaced_keep = root / "dspark_head_keep.qwp"
            repack_dspark_qwp(
                replaced_bf16, replaced_keep, expert_down_quant="keep", dense_quant="keep",
                draft_head_from=src_model, draft_head_quant="keep", device="cpu",
            )
            hdr_kp, _ = audit_pack(replaced_keep)
            self.assertEqual(hdr_kp["tensors"]["draft_head.weight"]["hq"], {"bits": 8, "group": 32})

    def test_guards_against_inplace_and_bad_args(self) -> None:
        with maybe_force_cpu_quant("cpu"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_pack = root / "hf", root / "src_pack"
            build_synthetic_dspark_hf_checkpoint(hf_dir, seed=63, bfloat16_router=True)
            quantize_ernie55_model(hf_dir, src_pack, device="cpu")
            src_ds = src_pack / "dspark.qwp"

            with self.assertRaises(ValueError):
                repack_dspark_qwp(src_ds, src_ds, device="cpu")
            with self.assertRaises(ValueError):
                repack_dspark_qwp(src_ds, root / "bad.qwp", expert_down_quant="bogus_quant", device="cpu")
            with self.assertRaises(FileNotFoundError):
                repack_dspark_qwp(root / "missing.qwp", root / "out.qwp", device="cpu")


if __name__ == "__main__":
    unittest.main()
