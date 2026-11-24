---
name: qwegine-model-onboarding
description: >-
  Use when onboarding a new HuggingFace model architecture or variant to Qwegn
  (config decoding, capability structs, QWP1 packer adapters, 2-layer PyTorch
  oracles, synthetic E2E tests, and pack qualification).
---

# Qwegn New Model Onboarding & Oracle Runbook

Follow these five steps when adding support for a new HuggingFace checkpoint architecture without model-coupled hacks:

1. **HF Config, Modeling & Synthetic Constraint Audit:**
   - Read `config.json`, `configuration_*.py` (`PretrainedConfig.__init__` defaults omitted from `config.json`), and `modeling_*.py` side by side before touching `Sources/`; never ingest `*.jinja` chat templates into context (Law 24).
   - Verify boolean/dimension flags verbatim against the real checkpoint: e.g., `Ling-3.2-air` sets `"use_mla_nope": false` (decoupled RoPE active), `"num_kv_heads_for_linear_attn": 0` (fallback to `num_attention_heads`), `vision_config` requires `"num_position_embeddings"` (not `num_pos_embeddings`), and `emitBailingCube` advances the 3D MRoPE cursor by `(lt - 1) * 2 + 1` (`mropeDelta = -3` for a single `2×2` merged frame inside 12 total tokens).
   - Keep synthetic MoE test fixtures at `E % 32 == 0` (`32 <= E <= 1024`, `K <= 32`) to satisfy `QWGemmMoERouter.swift:28` (`nExp = 8` traps).
   - Mirror runtime `QWMTPBackend` contracts in tests: pass `mtpBackend: nil` whenever `mtpDrafts == 0` (`"standard"` requires `1...15`).

2. **Capability-Seam Decoupling (`Scripts/lint_capability_seams.py`):**
   - Extend descriptor capability structs in `QWCore` (`KDAConfig`, `MLAConfig`, `MoERoutingConfig`, `SwiGLUClampConfig`, `VisionTowerConfig`, `MTPConfig`, `HCAConfig`, `BlockSparseGQAConfig`, `MRoPELayout`) and decode them in `QWModelDescriptor+Decode.swift`.
   - Dispatch in `Operators`, `Subsystems`, `Runtime`, and `Apps` exclusively via capability struct fields, never `descriptor.family` or `model_type` strings. Gate with `python3 Scripts/lint_capability_seams.py` (`0 violation(s)`).

3. **QWP1 Packer Adapter Invariants (`Scripts/quantize/`):**
   - Check per-tensor `reader.has(name + "_scale_inv")` rather than global `quantization_config.modules_to_not_convert`: mixed FP8/BF16 checkpoints can keep trunk `g_proj.weight` in BF16 while quantizing NextN `attention.g_proj.weight` in FP8.
   - De-interleave Direct-Q MLA projections from per-head `[nope, rope]` to planar `[nope; rope]` (`[qHeads * nopeD; qHeads * ropeD]`).
   - Guard cold-expert routing permutations with `(geometry.moeRoutingConfig?.nGroup ?? 1) <= 1` so grouped MoE routing preserves contiguous expert group boundaries.
   - Keep sub-32-row projections (`< 32` rows, e.g., KDA `b_proj` or head-wise `g_proj`) as raw `bf16` in `.qwp` and apply `fwht32OrthoInPlace` at driver load time when `hl.weightBasis.rotatesActivations` is true.

4. **2-Layer PyTorch-to-Metal Oracle (`max_abs_diff < 5e-3` + Argmax Identity):**
   - For ops requiring unavailable CUDA packages (e.g., `fla`, `einops`), register lightweight pure-PyTorch shims into `sys.modules` before loading `modeling_*.py` via a synthetic package namespace, and pass `_attn_implementation = "eager"`.
   - Separate Metal kernel fidelity from BF16 activation truncation (`~0.0137` diff vs FP16 in `RMSNorm`) and PTQ weight noise (`~0.0139` diff for `hq8_g32`): pack a tiny 2-layer synthetic checkpoint via `pack.quantize_model(..., dense_quant="hq8_g32", expert_quant="hq8_g32", expert_down_quant="hq8_g32")`, reload the exact dequantized QWP1 tensors (re-interleaving MLA `q_proj` if reordered) into PyTorch `torch.float16` (leaving FP32 router `gate.weight`/`expert_bias` and `A_log`/`dt_bias` in `float32`), and compare against Metal `QWEngineDriver.logitVector()` via an env-gated `XCTest` (`QW_*_ORACLE_DIR`) across both text and `QWVisualSpan` multimodal inputs.
   - Note `.venv-oracle` uses stdlib `unittest` (`pytest` is absent): run adapter tests via `.venv-oracle/bin/python3 -m unittest ...`, and symlink `.venv-oracle` + copy/link `Scripts/spike/out` only outside git staging (`rm -f .venv-oracle` before `git commit`).

5. **Pack & Full Qualification:**
   - Pre-flight free UMA memory (`vm_stat`) and zero GPU co-runners (`ps`), run `pack.py` + `pack.py --audit-release <out>`, verify greedy SHA-256 parity (`--mtp-drafts 0` vs `1 --mtp-backend standard --mtp-relaxed-margin 0 --mtp-draft-strategy neural`) across `f16`/`q8`/`q4` KV modes, and complete `QWServer` `S1–S9` + vision + `16K+` needle + `llmprobe` + paired A/B vs `master`.
