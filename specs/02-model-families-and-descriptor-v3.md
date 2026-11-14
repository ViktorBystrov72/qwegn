# 02 — Zero-Overload Model Families & `geometryDigestV3`

> Normative specification for `QWModelDescriptor`, `QWModelRole`, `QWModelFamilyRegistry`, config-sourced geometry, and digest versioning (`Sources/Foundation/QWCore/`).

## 1. Zero-Overload 11-Family Contract
Every HuggingFace `model_type` maps 1:1 through `QWModelFamilyRegistry` to an explicit `QWModelDescriptor.Family` case; no family case may be overloaded onto another architecture's traits or kernels. Unknown `model_type` strings throw `QWModelDescriptorError.unsupportedModelType`.

| `QWModelDescriptor.Family` | `model_type` keys in `builtinSpecs` | Role / Key Architectural Traits |
| :--- | :--- | :--- |
| `.qwen38Dense` | `qwen3_5`, `qwen3_5_text` | `defaultAttnOutputGate: true`, `+1` RMSNorm, Qwen chat template IDs |
| `.qwen38MoE` | `qwen3_5_moe`, `qwen3_5_moe_text` | Sparse MoE + attention output gate, `+1` RMSNorm |
| `.qwen38FlashNext` | `qwen4_exp`, `qwen4_exp_text` | Hybrid QSA + GDN + HyperPLE (`maxSpeculativeDrafts: 3`, sigmoid expert gate) |
| `.senseNovaLoop` | `sensenova_loop`, `sensenova_loop_text` | Looped middle blocks (`LoopSettings`) + `MHCConfig` Sinkhorn mixing |
| `.glm5Next` | `glm5_next`, `glm5_next_text` | Hybrid NoPE MLA (`HCAConfig`/`HybridMLAConfig`) + KDA, standard RMSNorm, sigmoid gate |
| `.ernie55MoE` | `ernie5_5_moe`, `ernie5_5_moe_text` | Latent-MoE + HCA/MLA-DSA hybrid, standard RMSNorm, sigmoid gate |
| `.qwen3Dense` | `qwen3` | Standard HF/MLX dense Qwen3 (`defaultAttnOutputGate: false`) |
| `.qwen3MoE` | `qwen3_moe` | Standard HF/MLX sparse MoE (`defaultAttnOutputGate: false`) |
| `.qwen2Dense` | `qwen2` | Standard dense Qwen2 (`defaultAttnOutputGate: false`) |
| `.llamaDense` | `llama` | Standard dense Llama (`defaultAttnOutputGate: false`) |
| `.mistralDense` | `mistral` | Registered descriptor (`isOfficialSupported: false` pending sliding window) |

## 2. Explicit Execution Roles (`QWModelRole`)
`QWModelDescriptor.role` (`typealias Role = QWModelRole`) separates execution semantics:
- `.causalLM`: text-only autoregressive decoder.
- `.multimodalConditionalLM`: vision-language model declaring `.vision` capability and verified `QWVisionConfig`.
- `.speculativeDrafter`: dedicated draft model (e.g., `DSpine` / `DFlash 2`); verified against target compatibility before attachment and rejected if invoked standalone as `--model` without target verifier context.

## 3. Config-Sourced Geometry & Linter Gate
- Geometry, layer cadence, RoPE/YaRN parameters, MoE/Latent/HCA/MLA fields, and `QWVisionConfig` come strictly from the model's `assets/config.json` (and companion pack files under `~/.qwg/models`), never hardcoded or synthesized when absent (`Scripts/lint_geometry_defaults.py` enforces zero hidden geometry fallbacks across `Sources/`).
- `QWModelDescriptor.verifyInternalConsistency()` re-derives `qProjOut`, `rotaryDims`, MRoPE sums, `gdnQKVChannels`, `gdnZChannels`, full-attention cadence, and PLE linear-layer IDs prior to runtime plan construction.
- Each family resolves explicitly via `QWModuleRegistry` (`DenseModule`, `FlashNextModule`, and dedicated hybrid modules in `QWHybridFamilyModules.swift`).

## 4. `geometryDigestV2` vs `geometryDigestV3`
- **`geometryDigestV2` (`qwg-geo-v2:...`)**: Preserved **byte-identical** for on-disk `.qwp` pack headers and `.qlsa` sidecar trailers so existing packs validate without re-packing.
- **`geometryDigestV3` (`qwg-geo-v3:...`)**: Extends `V2` with `family`, `role`, `env_type`, `kda`, `mla`, `loop`, `mhc`, and vision geometry (`vis=...` when `.vision` capability is present). Used for `QWEnginePlan`, compiled layer caches, and newly generated artifacts to eliminate cross-family/cross-role digest collisions.
