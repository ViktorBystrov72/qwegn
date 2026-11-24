---
name: qwegine-inference-qa-and-sampling
description: >-
  Use when modifying or debugging Qwegn runtime inference, sampling (QWSampler,
  repetition_penalty), multi-turn prefix caching (PrefixCache, QWFlashNext),
  speculative decoding across knobs (MTPVerification, DFlash 2, DSpine, .ngram,
  PLD), thinking budgets (QWServerSchema, reasoning.effort), constrained/tool
  decoding, or running multi-model QWServer coherence QA.
---

# Qwegn Inference Coherence, Sampling & Multi-Knob QA Runbook

## 1. Sampling, Repetition & Token-History Hygiene (`QWEngine` + `QWFlashNext`)

1. **Generated-Only Loop Repetition Window:** Separate `promptTokenCount` from generated output and scope `repetition_penalty` strictly to the trailing 256 generated tokens (`QWEngineSession.repetitionPenaltyWindow = 256`, `activeRepetitionHistory`, `FNGPUSessionData.recentGeneratedTokens`) using consecutive-cycle detection (`QWSampler.loopRepetitionTokens`: `4×` period-1, `3×` period-2, `2×` period-`3..96`, `≥6×` trailing-16 burst, plus 3rd-occurrence trailing `8`-gram detection). Keep prompt tokens and non-consecutive bigrams unpenalized so recurring digits, operators, and `</think>` close tags remain intact.
2. **Emitted-Token-Only History Updates:** During stochastic prefill and decode, write only the actually sampled/emitted token (never a parallel greedy `argmax`) into `tokenHistory`, `tokenRecycler`, and `seenTokens`. In `QWFlashNext`, record `firstToken` in `FNGPUSession.drawFirstToken` when `repetitionPenalty != 1.0`, and record newly sampled `next` / `nextTok` (not pre-step `token`) in speculative seam and batch loops (`FNGPUEngine+Seam`, `FNGPUBatchScheduler+Drive`, `FNGPUDFlash`).
3. **Fast-Accept Under Repetition Penalty:** Keep zero-copy greedy fast accept enabled when no repeat is active by gating on `!session.seenTokens.contains(draftTok)` rather than disabling whenever `repetitionPenalty != 1.0`.

## 2. Multi-Turn Cache, Prefill/Decode Sync & Speculation Parity

1. **Multi-Turn Prefix Continuity:** Always write live prefill head logits (`runsHead == true`) and preserve multi-turn prefix checkpoints across `QWEngineDriver` (including concurrent stochastic prompts in `DriverBatchDecode`) and `QWFlashNext`.
2. **Constraint & Streaming Speculation Parity:** Gate `tokenRecycler`, `.ngram`, `PLD`, and `DFlash 2` / `DSpine` speculation on `(constraint?.isUnconstrainedNow ?? true)` rather than `constraint == nil` so `QWThinkingBudgetConstraint` and `QWToolCallConstraint` speculate while unconstrained. Pass `streamsCommittedRounds: true, onStopToken:` in `generateExperimentalStochastic` so `cache_prompt: false` / `--no-prefix-cache` SSE streams keep speculating; note seeded stochastic runs (`seed >= 0`, `temperature > 0`) intentionally execute serial for determinism (`QWEngineDriverConformance`).
3. **Prefill & Async-Commit Isolation:** Suppress `activeConstraint` on intermediate `.naive` prefill tokens and restore `session.generated` / `tokenHistory` / `promptTokenCount` before the final token (`DriverPrefill`). Drain `session.lastCommitCB` at entry to `DriverDecode.forwardStep` before touching session slabs when falling back from async speculative commit to serial decode. Skip `mtpPromptEmbeddings` `memcpy` when `dflashDrafter != nil`.

## 3. Thinking Budgets & WebUI/API Knob Boundaries

1. **Graduated Reasoning Caps & Large-`max_tokens` Guard:** Map `reasoning.effort` (`low` → `256`, `medium` → `1024`, `high` → `4096`, `xhigh` → `8192`) in `QWServerSchema.resolvedThinkingPolicy`. When explicit `enable_thinking: true` omits `reasoning_effort` at `max_tokens > 32`, apply the `reserved` budget cap (preserving `thinkingBudget == nil` for `max_tokens <= 32`).
2. **WebUI Preset Hygiene:** Clear stale `session.params.maxTokens` overrides whenever switching chat presets (`chat-presets.js`).

## 4. Mandatory 9-Scenario Multi-Model Live `QWServer` QA Gate

Before marking a runtime, sampling, cache, speculation, or server change ready to merge, run sequential quiet-GPU `QWServer` QA across all installed flagship model families (`Dense/Looped`, `Dense`, `SHVQ MoE`, `Flash-Next`, `Hybrid MoE`), requiring `0` server `ERROR`/`FATAL`/`panic` log lines and clean `finish_reason == "stop"` on every scenario:

1. **`S1` Multi-turn WebUI stress:** 3 turns with warm-RAM prefix cache hits (`cached > 0` on T2/T3), `temperature: 0.4`, `top_p: 0.95`, `repetition_penalty: 1.5`, `tools`, `enable_thinking: false`.
2. **`S2` Thinking budgets:** `low`, `medium`, `high`, `xhigh`, and omitted `reasoning_effort` at large `max_tokens` under `repetition_penalty: 1.5`.
3. **`S3` Greedy + penalty sweep:** `temperature: 0.0` at `repetition_penalty` `1.0`, `1.05`, and `1.5`.
4. **`S4` Seeded determinism:** Paired runs with fixed `seed` matching exact length and text.
5. **`S5` Uncached streaming stochastic:** `stream: true`, `cache_prompt: false`, confirming `cached=0` and active `spec:` telemetry in the server log.
6. **`S6` Structured output, logprobs & stops:** JSON schema conformance, finite `logprobs`, and stripped `stop` sequences.
7. **`S7` Forced tool call:** `tool_choice` function call producing valid JSON arguments.
8. **`S8` Multi-protocol endpoints:** `/v1/responses` (`previous_response_id` chaining), `/v1/completions` (`echo` + finite `logprobs`), and `/v1/messages` (Anthropic `thinking` budget).
9. **`S9` Concurrent `B=4` streaming batch:** 4 parallel `cache_prompt: false` requests with distinct answers verifying zero cross-slot bleed.

**QA Harness Gotchas:**
- `POST /v1/messages` enforces `strictNotFound: true` (`QWHttpServer+Anthropic`): pass the loaded pack's exact `model` name rather than the `"qwegn"` alias.
- Give `S9` `max_tokens >= 160` so verbose/LaTeX models (`SenseNova-Loop-9B`) finish naturally instead of hitting `finish_reason == "length"`.
- Format assertion failures with `f"{type(e).__name__}: {e}"` so bare `assert` failures never log empty strings.
- **Long-Context NIAH / Recall Budget Calibration:** Looped/reasoning-capable models (e.g. `Inkling-2-Mini`) can emit `120–470` `<|content_thinking|>` recap tokens even under `--no-thinking`; calibrate prompts with an `8K` dry-run gate first, set `--max-tokens >= 1024`, grade extracted final-answer blocks, and require `hit_max_tokens == false` + `steady_allocs == 0` before launching multi-hour `256K–1M` sweeps.
