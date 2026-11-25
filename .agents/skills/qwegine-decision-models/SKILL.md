---
name: qwegine-decision-models
description: >-
  Use when implementing, optimizing, quantizing, or qualifying Qwegn System-1
  decision models (QWDecision, QWDecisionPackEngine, d1-2B and future decision
  architectures), decision HTTP/MCP APIs (/v1/decisions, /v1/systemone,
  /v1/laya, /mcp, RFC 7578 multipart), dual-slot sidecar UMA sizing, or
  external decision compliance benchmarks (jeval, typesafe-sdk, s1kit,
  JevBench, ImageJevBench-Box).
---

# Qwegn System-1 Decision Models, Compliance & SOTA Performance Runbook

Follow these four gates whenever modifying `QWDecision`, `QWDecisionPackEngine`, decision Metal kernels, or decision HTTP/MCP wire endpoints:

## 1. Zero-Shim External Wire Compliance Gate
Never certify decision APIs solely from internal `QWDecisionServerTests`. Run unmodified external clients directly against live `qwg serve` (zero translation shims):
- **External SDK & Harness Parity:** Verify `jeval check --model Kev` (`POST /v1/systemone`), `jeval check --model OpenAI-Decisions` (`POST /v1/decisions`), `typesafe-sdk` (`models.list()` returning `models[]`, `system_one` text, and RFC 7578 `multipart/form-data` `Image(Path(...))` uploads), `s1kit test --only qwegn` (`0` errors), Laya golden replay (`POST /v1/laya` `10/10` + `60,000`-char rail `HTTP 400`), and stateless `POST /mcp` (`qwg_decide`, `laya_ask`, `laya_guard`, `laya_status`).
- **Multibyte Offsets & Box Floor:** Verify UTF-8 multibyte text extraction (e.g., CJK + emoji) emits Unicode character indices in `char_span` (`charSpan`) rather than raw UTF-8 byte offsets, and verify `always_emit_box` (`--decision-always-emit-box`) returns best-effort box coordinates even below threshold so `ImageJevBench-Box` meets the `>= 90%` valid-box floor.
- **Auto Reasoning Routing:** Verify `reasoning.effort = "auto"` emits CORS-exposed `X-Qwegn-Decision-Route` (`Access-Control-Expose-Headers`), routes high-confidence trivial/deep prompts cleanly, and fails open below `0.70` confidence.

## 2. ADR-009 Calibration-Gated Quantization & Accuracy Benchmarks
- **Never Ship Quantized Decision Packs on Top-1 Alone:** Gate every quantized decision pack against BF16 + CPU PyTorch oracle on `ΔECE <= 0.005`, `ΔBrier <= 0.003`, and `Top-1 >= 99.0%` (`ADR-009`). Reject and delete candidates that preserve Top-1 while breaking calibration (e.g., `d1-2B` `hq4_g32_m5` hit `100%` Top-1 but failed at `ΔECE = 0.03172`; mixed `hq8_g32` + `bf16` passed at `ΔECE = 0.00065`).
- **Canonical Eval Suite:** Qualify directly against `/v1/systemone` on `JevBench` (`231` public questions, `0` failed, NLL/ECE parity vs BF16) and `ImageJevBench-Box` (`50` stratified items, `>= 90%` valid boxes, ranking floor PASS), plus 3× CLI `qwg decide` bit-exact determinism.

## 3. Standalone Boot, Honest UMA Sizing & Knob Parity
- **Standalone & Sidecar Boot:** Confirm `qwg serve <decision-pack> --no-lazy` auto-routes to the decision slot (`/health` `loaded: true`) instead of loading chat `QWEngine`, and confirm `--no-load-vision` refuses `--image` with non-zero exit while dropping vision RSS.
- **Live UMA Estimation Check (`<= 10%` Error):** Verify `QWDecisionWeightLoader.estimateResidentBytes(root:descriptor:loadVision:)` accounts for both backbone planes and resident vision buffer overhead by comparing `/v1/decisions/status` `estimated_bytes` against live server RSS deltas across `hq8` and `bf16` with `load_vision = true` and `false`.
- **6-Knob Parity:** Keep `--decision-model` (`QWG_DECISION_MODEL`), `--decision-compute-unit`, `--decision-load-vision`, `--decision-score-readout`, `--decision-truncation`, and `--decision-always-emit-box` synchronized across `QWKnobRegistry.swift`, `KnobOptions.swift`, `/v1/settings`, `Qwegn.app` `SettingsView`, and `/ui` Settings.

## 4. Metal 4 NAX Prefill, Coalesced Batching & Dual-Slot Preemption
- **Hot-Path Invariants:** Keep decision prefill on Apple10 NAX (`qwg_gemm_bf16_nax_coop_fast` F16+BF16, tiled `qwg_attn_topology_prefill_nax_d64`/`d128`, `qwg_cast_f16_to_f32` GPU residual cast), allocate reusable buffers/RoPE tables via `QWDecisionScratchArena`, and coalesce `/v1/decisions/batch` + `/v1/systemone/batch` on GPU.
- **Paired Interleaved A/B Gate:** Benchmark fresh `master` release vs candidate on a quiet GPU (`>= 2` alternating rounds) across `41`, `1035`, `4059`, `8091` tokens, `/v1/decisions/batch N=8`, and dual-slot co-residency (`Inkling-2-Mini-QWP-Fast` + `d1-2B-hq8`), verifying non-regression on short prompts and primary LLM decode retention (`QWGPUTurnArbitrator`).
