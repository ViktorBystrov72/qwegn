"""CPU unit tests for ERNIE-5.5 DSpark + FlashIndexer packer, BF16 routers, repacker & PyTorch oracle."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS / "quantize"))
sys.path.insert(0, str(_SCRIPTS / "oracle"))

from ernie55_adapter import main as adapter_main, quantize_ernie55_model  # noqa: E402
from ernie55_dspark_oracle import (  # noqa: E402
    build_synthetic_dspark_hf_checkpoint,
    dspark_draft_block,
    dspark_ingest_context,
    load_hf_canonical_weights,
    run_target_and_capture,
    verify_synthetic_oracle,
)
from inkling2_adapter import maybe_force_cpu_quant  # noqa: E402
from pack import sha256  # noqa: E402
from repack_dense_qwp import main as repack_main, repack_dense_qwp  # noqa: E402
from verify import audit_pack, read_header  # noqa: E402


class TestErnie55DSparkPack(unittest.TestCase):
    def setUp(self) -> None:
        maybe_force_cpu_quant("cpu")

    def test_native_bf16_router_and_f32_router_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_bf16, pk_bf16 = root / "hf_bf16", root / "pk_bf16"
            build_synthetic_dspark_hf_checkpoint(hf_bf16, seed=11, bfloat16_router=True)
            quantize_ernie55_model(hf_bf16, pk_bf16, device="cpu")
            hdr_m, _ = read_header(pk_bf16 / "model.qwp")
            hdr_d, _ = read_header(pk_bf16 / "dspark.qwp")
            self.assertEqual(hdr_m["tensors"]["model.layers.1.mlp.gate.weight"]["kind"], "bf16")
            self.assertEqual(hdr_d["tensors"]["dspark.layers.0.mlp.gate.weight"]["kind"], "bf16")
            self.assertEqual(hdr_d["tensors"]["dspark.layers.1.mlp.gate.weight"]["kind"], "bf16")

            hf_f32, pk_f32 = root / "hf_f32", root / "pk_f32"
            build_synthetic_dspark_hf_checkpoint(hf_f32, seed=11, bfloat16_router=False)
            quantize_ernie55_model(hf_f32, pk_f32, device="cpu")
            hdr_m32, _ = read_header(pk_f32 / "model.qwp")
            hdr_d32, _ = read_header(pk_f32 / "dspark.qwp")
            self.assertEqual(hdr_m32["tensors"]["model.layers.1.mlp.gate.weight"]["kind"], "f32")
            self.assertEqual(hdr_d32["tensors"]["dspark.layers.0.mlp.gate.weight"]["kind"], "f32")

    def test_dspark_only_partial_shards_and_into_existing_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_hf = root / "full_hf"
            tc = build_synthetic_dspark_hf_checkpoint(full_hf, seed=21, bfloat16_router=True)
            partial = root / "partial_hf"
            partial.mkdir()
            for fn in ("config.json", "model.safetensors.index.json", "model-00002-of-00002.safetensors"):
                shutil.copy2(full_hf / fn, partial / fn)
            (partial / "flashindexer").mkdir()
            shutil.copy2(full_hf / "flashindexer" / "flashindexer.safetensors", partial / "flashindexer" / "flashindexer.safetensors")

            with self.assertRaises(ValueError):
                adapter_main(["--model", str(partial), "--out", str(root / "bad"), "--dspark-only",
                              "--dspark-markov-w2-quant", "aq8g32", "--device", "cpu"])

            out_ds = root / "dspark_only_out"
            rc = adapter_main([
                "--model", str(partial), "--out", str(out_ds), "--dspark-only", "--device", "cpu",
                "--dspark-dense-quant", "shvq4g32", "--dspark-expert-quant", "shvq15g64",
                "--dspark-expert-down-quant", "shvq3g64", "--dspark-markov-w2-quant", "hq8g32",
            ])
            self.assertEqual(rc, 0)
            self.assertFalse((out_ds / "model.qwp").exists())
            self.assertTrue((out_ds / "dspark.qwp").is_file())
            self.assertTrue((out_ds / "flashindexer.qwp").is_file())
            self.assertFalse((out_ds / "assets" / "flashindexer.safetensors").exists())
            self.assertEqual((out_ds / "dspark.qwp.sha256").read_text().split()[0], sha256(out_ds / "dspark.qwp"))
            self.assertEqual((out_ds / "flashindexer.qwp.sha256").read_text().split()[0], sha256(out_ds / "flashindexer.qwp"))

            hdr_ds, _ = audit_pack(out_ds / "dspark.qwp")
            hdr_fi, _ = audit_pack(out_ds / "flashindexer.qwp")
            t_ds = hdr_ds["tensors"]
            self.assertEqual(t_ds["dspark.main_proj.weight"]["hq"], {"bits": 4, "group": 32, "vq": "shvq1"})
            self.assertEqual(t_ds["dspark.layers.0.self_attn.linear_kv_proj.weight"]["hq"], {"bits": 4, "group": 32, "vq": "shvq1"})
            self.assertEqual(t_ds["dspark.markov_w1.weight"]["shape"], [tc["vocab_size"], tc["dspark_markov_rank"]])
            self.assertEqual(t_ds["dspark.markov_w1.weight"]["kind"], "bf16")
            self.assertEqual(t_ds["dspark.markov_w2.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(t_ds["dspark.confidence_head.weight"]["shape"], [1, tc["hidden_size"] + tc["dspark_markov_rank"]])
            self.assertEqual(t_ds["dspark.layers.0.mlp.experts.gate_up_proj"]["experts"], tc["n_routed_experts"])
            self.assertEqual(t_ds["dspark.layers.0.mlp.experts.gate_up_proj"]["hq"], {"bits": 15, "group": 64, "vq": "shvq1"})
            self.assertEqual(t_ds["dspark.layers.0.mlp.experts.down_proj"]["hq"], {"bits": 3, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr_fi["tensors"]["model.layers.1.self_attn.indexer.linear_gate.weight"]["kind"], "bf16")
            self.assertEqual(hdr_fi["tensors"]["model.layers.1.self_attn.indexer.flashindexer_bias"]["kind"], "f32")

            # Verify --dspark-only updates an existing model.qwp pack without mutating model.qwp.
            existing = root / "existing"
            quantize_ernie55_model(full_hf, existing, device="cpu")
            for fn in ("dspark.qwp", "dspark.qwp.sha256", "flashindexer.qwp", "flashindexer.qwp.sha256"):
                (existing / fn).unlink()
            orig_model_sha = sha256(existing / "model.qwp")
            rc2 = adapter_main(["--model", str(partial), "--out", str(existing), "--dspark-only", "--device", "cpu"])
            self.assertEqual(rc2, 0)
            self.assertEqual(sha256(existing / "model.qwp"), orig_model_sha)
            hdr_ds2, _ = audit_pack(existing / "dspark.qwp")
            self.assertEqual(hdr_ds2["provenance"]["assets_manifest_sha256"], sha256(existing / "assets" / "manifest.json"))

    def test_repack_dense_qwp_keep_head_and_bf16_routers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_dir, src_pack = root / "hf", root / "src_pack"
            build_synthetic_dspark_hf_checkpoint(hf_dir, seed=31, bfloat16_router=False)
            quantize_ernie55_model(hf_dir, src_pack, device="cpu")

            src_hdr, _ = read_header(src_pack / "model.qwp")
            self.assertEqual(src_hdr["tensors"]["model.layers.1.mlp.gate.weight"]["kind"], "f32")
            self.assertEqual(src_hdr["tensors"]["lm_head.weight"]["hq"], {"bits": 8, "group": 32})

            # Case A: keep lm_head HQ8_G32 + cast F32 routers to BF16.
            rep_a = root / "repack_a"
            rc = repack_main([
                "--src", str(src_pack), "--out", str(rep_a),
                "--dense-quant", "shvq4_g32", "--keep-hq8-head", "--bf16-routers", "--device", "cpu",
            ])
            self.assertEqual(rc, 0)
            hdr_a, _ = audit_pack(rep_a / "model.qwp")
            self.assertEqual(hdr_a["tensors"]["lm_head.weight"]["hq"], {"bits": 8, "group": 32})
            self.assertEqual(hdr_a["tensors"]["model.layers.0.self_attn.o_proj.weight"]["hq"], {"bits": 4, "group": 32, "vq": "shvq1"})
            self.assertEqual(hdr_a["tensors"]["model.layers.1.mlp.experts.0.gate_proj.weight"]["hq"], {"bits": 15, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr_a["tensors"]["model.layers.1.mlp.experts.0.down_proj.weight"]["hq"], {"bits": 3, "group": 64, "vq": "shvq1"})
            self.assertEqual(hdr_a["tensors"]["model.layers.1.mlp.gate.weight"]["kind"], "bf16")
            self.assertEqual(hdr_a["provenance"]["quant_recorded"], "dense_shvq4_g32__expert_shvq15_g64__expdown_shvq3_g64")
            self.assertEqual(hdr_a["provenance"]["hq8_reference"], str((src_pack / "model.qwp").resolve()))
            self.assertTrue(hdr_a["provenance"]["bf16_routers"])
            self.assertEqual(hdr_a["provenance"]["dense_head_quant"], "hq8_g32")
            for sib in ("dspark.qwp", "dspark.qwp.sha256", "flashindexer.qwp", "flashindexer.qwp.sha256"):
                self.assertEqual(sha256(rep_a / sib), sha256(src_pack / sib))
            self.assertEqual(sha256(rep_a / "assets" / "manifest.json"), sha256(src_pack / "assets" / "manifest.json"))
            self.assertEqual((rep_a / "model.qwp.sha256").read_text().split()[0], sha256(rep_a / "model.qwp"))

            # Verify router cast values stay BF16-close to source F32 router.
            with (src_pack / "model.qwp").open("rb") as fs, (rep_a / "model.qwp").open("rb") as fr:
                es, er = src_hdr["tensors"]["model.layers.1.mlp.gate.weight"], hdr_a["tensors"]["model.layers.1.mlp.gate.weight"]
                fs.seek(es["raw_off"])
                f32_ref = torch.from_numpy(np.frombuffer(fs.read(4 * int(np.prod(es["shape"]))), dtype=np.float32).reshape(es["shape"]).copy())
                fr.seek(er["raw_off"])
                bf_got = torch.from_numpy(np.frombuffer(fr.read(2 * int(np.prod(er["shape"]))), dtype=np.uint16).reshape(er["shape"]).copy()).view(torch.bfloat16)
                self.assertTrue(torch.equal(bf_got, f32_ref.to(torch.bfloat16)))

            # Case B: repack lm_head too, leave F32 router unchanged.
            rep_b = root / "repack_b"
            res_b = repack_dense_qwp(src_pack, rep_b, dense_quant="shvq4_g32", keep_hq8_head=False, bf16_routers=False, device="cpu")
            hdr_b, _ = audit_pack(rep_b / "model.qwp")
            self.assertEqual(hdr_b["tensors"]["lm_head.weight"]["hq"], {"bits": 4, "group": 32, "vq": "shvq1"})
            self.assertEqual(hdr_b["tensors"]["model.layers.1.mlp.gate.weight"]["kind"], "f32")
            self.assertFalse(res_b["kept_hq8_head"])
            self.assertEqual(res_b["cast_bf16_routers"], 0)

    def test_dspark_oracle_swa_and_confidence_invariants(self) -> None:
        res = verify_synthetic_oracle()
        self.assertTrue(res["greedy_match"], res)
        self.assertTrue(res["fi_head_match"], res)
        self.assertLess(res["hq8_tap_err"], 0.15)
        self.assertLess(res["hq8_kv_err"], 0.25)
        self.assertLess(res["hq8_logit_err"], 0.35)
        self.assertLess(res["hq8_conf_err"], 0.08)

        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = Path(tmp) / "hf"
            tc = build_synthetic_dspark_hf_checkpoint(hf_dir, seed=7)
            w = load_hf_canonical_weights(hf_dir, tc)
            ctx = torch.tensor([3, 14, 29, 55, 67, 88], dtype=torch.long)  # len=6 > sliding_window=4
            tgt = run_target_and_capture(ctx, w, tc)
            kvs = dspark_ingest_context(tgt["concat_taps"], torch.arange(ctx.shape[0]), w, tc)
            ref = dspark_draft_block(int(ctx[-1].item()), kvs, int(ctx.shape[0]), w, tc)

            # Slots outside SWA window [pos0 - 4 ..< pos0) -> indices 0, 1 must have zero effect.
            kvs_outside = [kv.clone() for kv in kvs]
            kvs_outside[0][:2] += 10.0
            out_outside = dspark_draft_block(int(ctx[-1].item()), kvs_outside, int(ctx.shape[0]), w, tc)
            self.assertLess(float((ref["final_logits"] - out_outside["final_logits"]).abs().max().item()), 1e-6)

            # Slot inside SWA window (index 2 = pos0 - win) must change outputs.
            kvs_inside = [kv.clone() for kv in kvs]
            kvs_inside[0][2] += 10.0
            out_inside = dspark_draft_block(int(ctx[-1].item()), kvs_inside, int(ctx.shape[0]), w, tc)
            self.assertGreater(float((ref["final_logits"] - out_inside["final_logits"]).abs().max().item()), 1e-2)

            # confidence_head reads pre-norm head_hidden, invariant to scaling dspark.norm.weight.
            w_scaled = dict(w)
            w_scaled["dspark.norm.weight"] = w["dspark.norm.weight"] * 2.0
            out_scaled = dspark_draft_block(int(ctx[-1].item()), kvs, int(ctx.shape[0]), w_scaled, tc)
            self.assertTrue(torch.equal(ref["confidences"], out_scaled["confidences"]))
            self.assertGreater(float((ref["base_logits"] - out_scaled["base_logits"]).abs().max().item()), 1e-2)


if __name__ == "__main__":
    unittest.main()
