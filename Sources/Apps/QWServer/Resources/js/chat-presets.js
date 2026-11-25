/* Researched system role presets, .qwcache MemoryPack integration, and harnesses */

const SAMPLE_CHAT_TOOLS_JSON = JSON.stringify([{
  type: "function",
  function: {
    name: "read_workspace_file",
    description: "Read a UTF-8 file relative to the workspace root",
    parameters: { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" } }, required: ["path"] }
  }
}], null, 2);

const SAMPLE_EXTRACTION_SCHEMA_JSON = JSON.stringify({
  name: "structured_extraction",
  strict: true,
  schema: {
    type: "object",
    properties: {
      _reasoning: { type: "string" },
      summary: { type: "string" },
      entities: { type: "array", items: { type: "string" } },
      confidence: { type: "string", enum: ["high", "medium", "low"] }
    },
    required: ["_reasoning", "summary", "entities", "confidence"],
    additionalProperties: false
  }
}, null, 2);

const BUILT_IN_SYSTEM_PRESETS = [
  {
    id: "server-default",
    name: "Server Default",
    prompt: "",
    isCustom: false,
    harnessSummary: "Inherits ~/.qwg/settings.json defaults; keeps .qwcache context injection active.",
    harness: {
      temperature: null, topP: null, topK: null, repetitionPenalty: null,
      enableThinking: "inherit", reasoningEffort: "inherit", responseFormat: "text",
      exactSpeculation: "inherit"
    }
  },
  {
    id: "coding-agent",
    name: "Coding Agent",
    prompt: [
      "You are an autonomous senior software engineering agent executing tasks with architectural precision and zero regressions.",
      "",
      "<execution_protocol>",
      "1. ANALYZE: Trace symbols, callers, and edge cases before changing code. State your plan briefly when making non-trivial edits.",
      "2. EXECUTE: Produce clean, idiomatic, complete code with zero placeholders, stubs, or TODOs. Never fabricate APIs, file paths, or command outputs.",
      "3. VERIFY: Cite concrete verification checks (build, targeted tests, invariants) matching the change scope and self-correct root causes immediately on failure.",
      "</execution_protocol>",
      "",
      "<constraints_and_discipline>",
      "- Rely strictly on observed codebase context, active Memory Pack contents, or tool outputs.",
      "- Keep diffs minimal and scoped; do not rewrite unrelated modules or add unsolicited refactors.",
      "- Communicate in crisp technical prose; prioritize working code and exact diagnostics over narrative filler.",
      "</constraints_and_discipline>"
    ].join("\n"),
    isCustom: false,
    harnessSummary: "T=0.2 · top_p=0.95 · Thinking On · High effort · injects .qwcache, exact verify & tool discipline",
    harness: {
      temperature: 0.2, topP: 0.95, topK: null, repetitionPenalty: null,
      enableThinking: "true", reasoningEffort: "high", responseFormat: "text",
      injectMemoryPackNotice: true, injectToolHarness: true, exactSpeculation: "true"
    }
  },
  {
    id: "deep-reasoning",
    name: "Deep Reasoning",
    prompt: [
      "You are a rigorous first-principles reasoning analyst. Solve problems through explicit derivation, falsification, and verified evidence.",
      "",
      "<cognitive_protocol>",
      "1. FIRST PRINCIPLES: Strip superficial analogies and reduce the question to fundamental invariants, constraints, and mechanisms.",
      "2. ASSUMPTIONS: Explicitly list working assumptions whenever requirements are ambiguous; treat confirmed facts as settled.",
      "3. ADVERSARIAL CHECK: Test your hypothesis against material counterexamples, boundary conditions, and failure modes before concluding.",
      "4. SYNTHESIS: Match depth to difficulty—deliver a structured, unambiguous answer without circular reasoning.",
      "</cognitive_protocol>"
    ].join("\n"),
    isCustom: false,
    harnessSummary: "T=0.6 · top_p=0.95 · Thinking On · High effort · <think> trace verification harness",
    harness: {
      temperature: 0.6, topP: 0.95, topK: null, repetitionPenalty: null,
      enableThinking: "true", reasoningEffort: "high", responseFormat: "text",
      injectThinkingTagHint: true, exactSpeculation: "inherit"
    }
  },
  {
    id: "concise-technical",
    name: "Concise Technical",
    prompt: [
      "IDENTITY: Direct, grounded technical peer. Optimize strictly for correctness, utility, and signal density.",
      "",
      "CORE RULES:",
      "1. NO PREAMBLE OR FILLER: Lead at token zero with the direct answer, patch, or command. Omit greetings, apologies, and throat-clearing.",
      "2. ZERO SYCOPHANCY: Evaluate premises objectively. Never flatter or validate flawed assumptions; state corrections flatly once with technical evidence.",
      "3. NO REDUNDANCY: Do not restate the question or append trailing summary fluff.",
      "4. FORMAT PROPORTIONALITY: Use runnable code blocks and tight bullet points calibrated to technical complexity; state missing prerequisites plainly."
    ].join("\n"),
    isCustom: false,
    harnessSummary: "T=0.2 · top_p=0.9 · Thinking Off · Low effort · direct token-zero output",
    harness: {
      temperature: 0.2, topP: 0.9, topK: null, repetitionPenalty: null,
      enableThinking: "false", reasoningEffort: "low", responseFormat: "text",
      exactSpeculation: "inherit"
    }
  },
  {
    id: "json-extractor",
    name: "JSON / Structured Extractor",
    prompt: [
      "## Role",
      "You are a deterministic structured-data extraction engine. Convert unstructured input into strictly valid JSON without conversational text.",
      "",
      "## Extraction Rules",
      "1. GROUNDED FACTS ONLY: Extract only values explicitly supported by the input. Never invent, extrapolate, or guess missing fields.",
      "2. MISSING DATA: Emit `null` for absent scalar fields and `[]` for empty lists.",
      "3. TRACEABILITY: When using a free-form JSON object without a caller schema restricting keys, include a brief `\"_reasoning\"` key first to verify field grounding before populating outputs.",
      "",
      "## Output Contract",
      "Start your response directly with `{` and end with `}`. Do not wrap output in markdown fences or emit prose outside the JSON value."
    ].join("\n"),
    isCustom: false,
    harnessSummary: "T=0.0 · top_p=1.0 · Thinking Off · Native json_object/json_schema grammar + exact verify",
    harness: {
      temperature: 0.0, topP: 1.0, topK: null, repetitionPenalty: null,
      enableThinking: "false", reasoningEffort: "inherit", responseFormat: "json_object",
      injectJsonContract: true, exactSpeculation: "true"
    }
  }
];

function getPresetById(presetId) {
  const all = typeof getAllSystemPresets === "function" ? getAllSystemPresets() : BUILT_IN_SYSTEM_PRESETS;
  return all.find((p) => p.id === presetId) || null;
}

/* Matches QWMemoryPack.discoverManifestRoot(inSystemPrompt:) path regex exactly. */
function detectWorkspacePathInSystemPrompt(promptText) {
  if (typeof promptText !== "string" || !promptText) return "";
  const re = /((?:~|\/Users\/|\/home\/|\/opt\/|\/var\/|\/tmp\/|\/Volumes\/)[A-Za-z0-9._\-\/]+)/g;
  const match = re.exec(promptText);
  if (!match || !match[1]) return "";
  return match[1].replace(/[.,;:)\]'"]+$/, "");
}

function getActiveMemoryPackFacts() {
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  const facts = modelObj && modelObj.facts ? modelObj.facts : null;
  if (!facts || !facts.mountedMemoryPack) return null;
  return {
    summary: facts.mountedMemoryPack,
    rootPath: facts.mountedMemoryPackRoot || "",
    fileCount: typeof facts.mountedMemoryPackFiles === "number" ? facts.mountedMemoryPackFiles : null,
    tokenCount: typeof facts.mountedMemoryPackTokens === "number" ? facts.mountedMemoryPackTokens : null,
    deltaCount: typeof facts.mountedMemoryPackDelta === "number" ? facts.mountedMemoryPackDelta : null,
    kvMode: facts.mountedMemoryPackKV || ""
  };
}

function applyPresetHarnessToSession(session, presetId) {
  if (!session) return;
  const preset = BUILT_IN_SYSTEM_PRESETS.find((p) => p.id === presetId);
  if (!preset || !preset.harness || session.harnessEnabled === false) return;
  const h = preset.harness;
  if (!session.params) session.params = {};
  if ("temperature" in h) session.params.temperature = h.temperature;
  if ("topP" in h) session.params.topP = h.topP;
  if ("topK" in h) session.params.topK = h.topK;
  if ("repetitionPenalty" in h) session.params.repetitionPenalty = h.repetitionPenalty;
  session.params.maxTokens = ("maxTokens" in h) ? h.maxTokens : null;
  if ("enableThinking" in h) session.params.enableThinking = h.enableThinking;
  if ("reasoningEffort" in h) session.params.reasoningEffort = h.reasoningEffort;
  if ("responseFormat" in h) {
    if (h.responseFormat === "json_object" && (session.params.jsonSchema || "").trim()) {
      session.params.responseFormat = "json_schema";
    } else {
      session.params.responseFormat = h.responseFormat;
    }
  }
  if ("exactSpeculation" in h) session.params.exactSpeculation = h.exactSpeculation;
}

function buildHarnessedSystemPrompt(session) {
  if (!session) return "";
  const rawSys = (session.systemPrompt || "").trim();
  const srvDefault = typeof loadedSettingsObj !== "undefined" && loadedSettingsObj && typeof loadedSettingsObj.systemPrompt === "string"
    ? loadedSettingsObj.systemPrompt.trim()
    : "";
  const basePrompt = !rawSys && session.presetId === "server-default" ? srvDefault : rawSys;
  const explicitWs = (session.workspacePath || "").trim();
  const detectedInPrompt = detectWorkspacePathInSystemPrompt(basePrompt);
  const memFacts = getActiveMemoryPackFacts();
  const harnessOn = session.harnessEnabled !== false;
  const memHarnessOn = session.useMemoryPackHarness !== false;

  const sections = [];
  if (basePrompt) sections.push(basePrompt);

  if (explicitWs && !basePrompt.includes(explicitWs)) {
    sections.push(`<workspace_context>\nWorkspace directory: ${explicitWs}\n</workspace_context>`);
  }

  const effectiveRoot = explicitWs || detectedInPrompt || (memFacts ? memFacts.rootPath : "");
  if (memHarnessOn && (memFacts || effectiveRoot)) {
    const label = memFacts ? memFacts.summary : (effectiveRoot ? `workspace '${effectiveRoot}'` : "active repository");
    const deltaNote = memFacts && typeof memFacts.deltaCount === "number" && memFacts.deltaCount > 0
      ? ` Live working-tree delta overlays (${memFacts.deltaCount} changed file(s)) appear in <|memory_delta|> blocks.`
      : "";
    sections.push(
      `<memory_pack_harness>\n` +
      `Codebase Memory Pack (.qwcache / LSA) is configured for ${label}. When Resident Cortex files, [QWG REPOSITORY CORTEX MAP], or <|memory_vault_recall|> blocks appear in context, answer architecture, symbol, and cross-file questions directly from LSA memory before issuing redundant grep, find, or read_file tool calls.${deltaNote}\n` +
      `</memory_pack_harness>`
    );
  }

  if (harnessOn) {
    const toolsRaw = session.params && typeof session.params.toolsJson === "string" ? session.params.toolsJson.trim() : "";
    if (session.presetId === "coding-agent" && toolsRaw) {
      sections.push(
        "<tool_execution_harness>\n" +
        "Emit structured tool calls directly with exact schema arguments. Never narrate internal tool names to the user, and inspect files or verify outputs before claiming fixes.\n" +
        "</tool_execution_harness>"
      );
    }
    if (session.presetId === "deep-reasoning") {
      sections.push(
        "<reasoning_harness>\n" +
        "Execute first-principles derivation, explicit assumption checks, and counterexample tests inside <think>...</think> before presenting the clean final synthesis.\n" +
        "</reasoning_harness>"
      );
    }
    const fmt = session.params ? session.params.responseFormat : "text";
    if (session.presetId === "json-extractor" || fmt === "json_object" || fmt === "json_schema") {
      sections.push(
        "<json_harness>\n" +
        "Output a single syntactically valid JSON object conforming strictly to the requested format or schema. Never emit markdown code fences or trailing commentary.\n" +
        "</json_harness>"
      );
    }
  }

  if (!sections.length) return "";
  if (sections.length === 1 && !rawSys && session.presetId === "server-default" && !explicitWs && !memFacts) return "";
  return sections.join("\n\n");
}

function validateJsonTurnContent(content) {
  const raw = typeof content === "string" ? content.trim() : "";
  if (!raw) return { valid: false, reason: "Empty output" };
  try {
    const parsed = JSON.parse(raw);
    return { valid: true, parsed };
  } catch (err) {
    if (raw.includes("```")) return { valid: false, reason: "Contains markdown code fences around JSON" };
    return { valid: false, reason: err && err.message ? err.message : "Invalid JSON syntax" };
  }
}

function formatJsonChatTurn(msgId) {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  const msg = session.messages.find((m) => m.id === msgId);
  if (!msg || !msg.content) return;
  let candidate = msg.content.trim();
  const fenceMatch = candidate.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenceMatch && fenceMatch[1]) candidate = fenceMatch[1].trim();
  try {
    const parsed = JSON.parse(candidate);
    msg.content = JSON.stringify(parsed, null, 2);
    msg.isRaw = true;
    if (typeof commitChatWorkspace === "function") commitChatWorkspace(session);
    if (typeof notifyChat === "function") notifyChat("Formatted response as valid JSON.");
  } catch (err) {
    if (typeof notifyChat === "function") notifyChat("Cannot format invalid JSON: " + err.message, true);
  }
}

function updateActiveChatWorkspacePath(value) {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  session.workspacePath = value;
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  renderChatMemoryPackAndHarnessUI(session);
}

function useDetectedPromptWorkspacePath() {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  const detected = detectWorkspacePathInSystemPrompt(session.systemPrompt || "");
  if (!detected) return;
  session.workspacePath = detected;
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  mountChatMemoryPack(detected);
}

function toggleChatPresetHarness(enabled) {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  session.harnessEnabled = Boolean(enabled);
  if (session.harnessEnabled) applyPresetHarnessToSession(session, session.presetId);
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  if (typeof renderChatParamsDrawer === "function") renderChatParamsDrawer();
  renderChatMemoryPackAndHarnessUI(session);
}

function toggleChatMemoryPackHarness(enabled) {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  session.useMemoryPackHarness = Boolean(enabled);
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  renderChatMemoryPackAndHarnessUI(session);
}

function reapplyActivePresetHarness() {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  session.harnessEnabled = true;
  applyPresetHarnessToSession(session, session.presetId);
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  if (typeof renderChatParamsDrawer === "function") renderChatParamsDrawer();
  renderChatMemoryPackAndHarnessUI(session);
  if (typeof notifyChat === "function") notifyChat("Applied preset harness parameters.");
}

function loadSampleExtractionSchema() {
  if (typeof getActiveChatSession !== "function") return;
  const session = getActiveChatSession();
  session.params.responseFormat = "json_schema";
  session.params.jsonSchema = SAMPLE_EXTRACTION_SCHEMA_JSON;
  session.updatedAt = Date.now();
  if (typeof persistChatSessions === "function") persistChatSessions();
  if (typeof renderChatParamsDrawer === "function") renderChatParamsDrawer();
  if (typeof notifyChat === "function") notifyChat("Loaded sample JSON Schema for constrained decoding.");
}

function resolveChatWorkspaceCandidate(optionalPath) {
  const session = typeof getActiveChatSession === "function" ? getActiveChatSession() : null;
  const inputEl = typeof document !== "undefined" ? document.getElementById("chat-workspace-path") : null;
  const candidate = (typeof optionalPath === "string" ? optionalPath : (inputEl ? inputEl.value : (session ? session.workspacePath : "")))
    || detectWorkspacePathInSystemPrompt(session ? session.systemPrompt : "");
  return (candidate || "").trim();
}

async function mountChatMemoryPack(optionalPath) {
  if (typeof fetch === "undefined") return;
  const inputEl = typeof document !== "undefined" ? document.getElementById("chat-workspace-path") : null;
  const repoPath = resolveChatWorkspaceCandidate(optionalPath);
  if (!repoPath) {
    if (typeof notifyChat === "function") notifyChat("Enter a workspace directory path containing .qwg/manifest.json.", true);
    if (inputEl) inputEl.focus();
    return;
  }
  await inspectMemoryPackAndMount(repoPath, { forceModal: false, fromSettings: false });
}

async function openChatMemoryPackModal(optionalPath) {
  if (typeof fetch === "undefined") return;
  const inputEl = typeof document !== "undefined" ? document.getElementById("chat-workspace-path") : null;
  const repoPath = resolveChatWorkspaceCandidate(optionalPath);
  if (!repoPath) {
    if (typeof notifyChat === "function") notifyChat("Enter a workspace directory path first.", true);
    if (inputEl) inputEl.focus();
    return;
  }
  await inspectMemoryPackAndMount(repoPath, { forceModal: true, fromSettings: false });
}

async function unmountChatMemoryPack() {
  if (typeof fetch === "undefined") return;
  try {
    const res = await fetch(`${API_BASE}/v1/memory/unmount`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    if (typeof fetchDashboard === "function") fetchDashboard();
    if (typeof loadLiveSettings === "function" && typeof loadedSettingsObj !== "undefined" && loadedSettingsObj) loadLiveSettings();
    if (typeof notifyChat === "function") notifyChat("Unmounted active .qwcache Memory Pack.");
  } catch (err) {
    if (typeof notifyChat === "function") notifyChat("Memory Pack unmount failed: " + err.message, true);
  }
}

function renderChatMemoryPackAndHarnessUI(session) {
  if (typeof document === "undefined" || !session) return;
  const memFacts = getActiveMemoryPackFacts();
  const wsPath = (session.workspacePath || "").trim();
  const promptPath = detectWorkspacePathInSystemPrompt(session.systemPrompt || "");

  const badgeEl = document.getElementById("chat-mempack-badge");
  if (badgeEl) {
    if (memFacts) {
      badgeEl.textContent = `MemPack: ${memFacts.summary}`;
      badgeEl.classList.add("is-mounted");
      badgeEl.title = `${memFacts.rootPath || memFacts.summary} · Click to configure`;
    } else if (wsPath || promptPath) {
      badgeEl.textContent = "MemPack: Auto (request)";
      badgeEl.classList.remove("is-mounted");
      badgeEl.title = `Auto-discovers .qwg/manifest.json + .qwcache from '${wsPath || promptPath}' on send`;
    } else {
      badgeEl.textContent = "MemPack: —";
      badgeEl.classList.remove("is-mounted");
      badgeEl.title = "No .qwcache mounted; click to set workspace path or mount repository";
    }
  }

  const wsInput = document.getElementById("chat-workspace-path");
  if (wsInput && wsInput.value !== (session.workspacePath || "")) wsInput.value = session.workspacePath || "";
  const unmountBtn = document.getElementById("chat-mempack-unmount-btn");
  if (unmountBtn) unmountBtn.hidden = !memFacts;
  const memToggle = document.getElementById("chat-mempack-harness-toggle");
  if (memToggle) memToggle.checked = session.useMemoryPackHarness !== false;

  const detectEl = document.getElementById("chat-mempack-detect");
  if (detectEl) {
    if (promptPath && (!memFacts || memFacts.rootPath !== promptPath)) {
      detectEl.hidden = false;
      detectEl.innerHTML = `Detected path in system prompt: <code class="md-inline-code">${escapeHtml(promptPath)}</code> <button type="button" class="chat-turn-btn" onclick="useDetectedPromptWorkspacePath()">Mount Detected Path</button>`;
    } else {
      detectEl.hidden = true;
      detectEl.textContent = "";
    }
  }

  const statusEl = document.getElementById("chat-mempack-status");
  if (statusEl) {
    if (memFacts) {
      const filesText = typeof memFacts.fileCount === "number" ? `${memFacts.fileCount} files` : "indexed";
      const tokText = typeof memFacts.tokenCount === "number" && typeof formatTokenCount === "function" ? `${formatTokenCount(memFacts.tokenCount)} cortex tokens` : "";
      const deltaText = typeof memFacts.deltaCount === "number" ? `live delta: ${memFacts.deltaCount} file(s)` : "";
      statusEl.textContent = ["Mounted: " + memFacts.summary, memFacts.rootPath, filesText, tokText, deltaText].filter(Boolean).join(" · ");
    } else if (wsPath) {
      statusEl.textContent = `Workspace '${wsPath}' will be sent in system prompt for automatic .qwcache mounting, or click Mount .qwcache now.`;
    } else {
      statusEl.textContent = "Set a workspace path or include a repo path in System Prompt so the engine auto-mounts <root>/.qwg/*.qwcache.";
    }
  }

  const preset = BUILT_IN_SYSTEM_PRESETS.find((p) => p.id === session.presetId);
  const harnessToggle = document.getElementById("chat-harness-toggle");
  if (harnessToggle) harnessToggle.checked = session.harnessEnabled !== false;
  const summaryEl = document.getElementById("chat-harness-summary");
  if (summaryEl) {
    summaryEl.textContent = session.harnessEnabled === false
      ? "Preset harness paused (using manual sampling & format overrides)."
      : (preset && preset.harnessSummary ? preset.harnessSummary : "Custom prompt (workspace & .qwcache harness applies when configured).");
  }
}

function parseCustomMetadataInput(raw) {
  const trimmed = typeof raw === "string" ? raw.trim() : "";
  if (!trimmed) return {};
  if (trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Custom metadata JSON must be a flat object of scalar values.");
    }
    const out = {};
    Object.keys(parsed).forEach((key) => {
      const val = parsed[key];
      if (val === null || val === undefined) return;
      if (typeof val === "object") throw new Error(`Metadata key '${key}' must be a scalar string, number, or boolean.`);
      const k = String(key).trim();
      if (k) out[k] = String(val);
    });
    return out;
  }
  const out = {};
  trimmed.split(/[,;\n]+/).forEach((segment) => {
    const item = segment.trim();
    if (!item) return;
    let sepIdx = item.indexOf("=");
    if (sepIdx === -1) sepIdx = item.indexOf(":");
    if (sepIdx <= 0) throw new Error(`Invalid metadata entry '${item}'; expected key=value or JSON object.`);
    const k = item.slice(0, sepIdx).trim();
    const v = item.slice(sepIdx + 1).trim();
    if (k) out[k] = v;
  });
  return out;
}

function buildChatRequestMetadata(session) {
  const s = session || {};
  const p = s.params || {};
  const meta = {
    client: "qwegn-webui",
    harness_enabled: s.harnessEnabled === false ? "false" : "true",
    use_memory_pack: s.useMemoryPackHarness === false ? "false" : "true"
  };
  if (s.id) meta.session_id = String(s.id);
  if (s.presetId) meta.preset_id = String(s.presetId);
  const ws = typeof s.workspacePath === "string" ? s.workspacePath.trim() : "";
  if (ws) meta.workspace_path = ws;
  if (p.exactSpeculation === "true" || p.exactSpeculation === "false") meta.exact_speculation = p.exactSpeculation;
  Object.assign(meta, parseCustomMetadataInput(p.customMetadata || ""));
  return meta;
}

function applyChatRequestMetadataToPayload(payload, session) {
  const s = session || {};
  const p = s.params || {};
  if (s.id) payload.session_id = String(s.id);
  payload.cache_prompt = p.cachePrompt !== false;
  payload.store = p.store !== false;
  const cacheKey = typeof p.promptCacheKey === "string" ? p.promptCacheKey.trim() : "";
  if (cacheKey) {
    payload.prompt_cache_key = cacheKey;
    payload.cache_salt = cacheKey;
  }
  payload.metadata = buildChatRequestMetadata(s);
  return payload;
}

const CHAT_REASONING_QUIPS = [
  "Reticulating splines…",
  "Untangling the attention heads…",
  "Negotiating with the weights…",
  "Multiplying very serious matrices…",
  "Checking the math twice…",
  "Consulting the tensors…",
  "Pondering in high dimensions…",
  "Polishing the logits…",
  "Overthinking a simple prompt…",
  "Hallucinating responsibly…",
  "Keeping the 1s, sweeping up the 0s…",
  "Re-reading the prompt to look busy…",
  "Marinating the logic…",
  "Caramelizing thoughts…",
  "Percolating ideas…",
  "Brewing a fresh batch of logic…",
  "Steeping the context…",
  "Toasting the KV cache…",
  "Simmering a few semicolons…",
  "Kneading the syntax tree…",
  "Flibbertigibbeting…",
  "Noodling on edge cases…",
  "Bribing the compiler…",
  "Dusting off first principles…",
  "Chasing a stray parenthesis…",
  "Feeding the GPU hamsters…",
  "Aligning the flux capacitor…",
  "Counting backwards from infinity…"
];

function pickRandomReasoningQuip(previousIndex, randomFn) {
  const list = CHAT_REASONING_QUIPS;
  if (!list.length) return { index: -1, text: "" };
  const rng = typeof randomFn === "function" ? randomFn : Math.random;
  const hasPrev = Number.isInteger(previousIndex) && previousIndex >= 0 && previousIndex < list.length;
  if (list.length === 1 || !hasPrev) {
    const idx = Math.min(list.length - 1, Math.max(0, Math.floor(rng() * list.length)));
    return { index: idx, text: list[idx] };
  }
  const step = Math.min(list.length - 2, Math.max(0, Math.floor(rng() * (list.length - 1))));
  const idx = (previousIndex + 1 + step) % list.length;
  return { index: idx, text: list[idx] };
}

function isTranscriptNearBottom(metrics, thresholdPx) {
  if (!metrics) return true;
  const tol = typeof thresholdPx === "number" && Number.isFinite(thresholdPx) ? thresholdPx : 48;
  const scrollHeight = Number(metrics.scrollHeight) || 0;
  const scrollTop = Number(metrics.scrollTop) || 0;
  const clientHeight = Number(metrics.clientHeight) || 0;
  return (scrollHeight - scrollTop - clientHeight) <= tol;
}

function parseChatSSELine(rawLine) {
  const line = (rawLine || "").trim();
  if (!line || line.startsWith(":") || !line.startsWith("data:")) return null;
  const payload = line.slice(5).trim();
  if (!payload) return null;
  if (payload === "[DONE]") return { done: true };
  const parsed = JSON.parse(payload);
  if (parsed && (parsed.type === "server_error" || parsed.error)) {
    const errObj = parsed.error || parsed; return { done: false, error: errObj.message || JSON.stringify(errObj) };
  }
  return { done: false, data: parsed };
}

function mergeToolCallChunks(existingCalls, deltaCalls) {
  const out = Array.isArray(existingCalls) ? existingCalls.map((c) => Object.assign({}, c, { function: Object.assign({}, c.function || {}) })) : [];
  if (!Array.isArray(deltaCalls)) return out;
  deltaCalls.forEach((dc) => {
    if (!dc) return;
    const idx = typeof dc.index === "number" ? dc.index : out.length;
    while (out.length <= idx) out.push({ id: "", type: "function", function: { name: "", arguments: "" } });
    const target = out[idx];
    if (dc.id) target.id = dc.id; if (dc.type) target.type = dc.type;
    if (dc.function) {
      if (dc.function.name) target.function.name += dc.function.name;
      if (dc.function.arguments) target.function.arguments += dc.function.arguments;
    }
  });
  return out;
}

function splitInlineThinkTags(content, explicitReasoning, _) {
  const text = typeof content === "string" ? content : "";
  let extracted = typeof explicitReasoning === "string" ? explicitReasoning : "";
  let cleaned = ""; let cursor = 0;
  while (cursor < text.length) {
    const openIdx = text.indexOf("<think>", cursor);
    if (openIdx === -1) { cleaned += text.slice(cursor); break; }
    cleaned += text.slice(cursor, openIdx);
    const bodyStart = openIdx + 7; const closeIdx = text.indexOf("</think>", bodyStart);
    if (closeIdx === -1) { extracted += text.slice(bodyStart); break; }
    extracted += text.slice(bodyStart, closeIdx); cursor = closeIdx + 8;
  }
  return { reasoning: extracted.trim(), answer: cleaned.replace(/^\n+/, "") };
}

function exportChatSessionMarkdown(session) {
  const s = session || getActiveChatSession(); const lines = [`# ${s.title || "Untitled Chat"}`];
  if ((s.systemPrompt || "").trim()) lines.push("", "## System Prompt", "", "```text", s.systemPrompt.trim(), "```");
  s.messages.forEach((m) => {
    const label = m.role === "user" ? "User" : (m.role === "tool" ? `Tool (${m.toolName || m.toolCallId || "output"})` : "Assistant");
    lines.push("", `## ${label}`);
    if (m.reasoning) lines.push("", "<details><summary>Reasoning</summary>", "", "```text", m.reasoning, "```", "", "</details>");
    if (m.content) lines.push("", m.content);
    if (Array.isArray(m.toolCalls) && m.toolCalls.length > 0) lines.push("", "```json", JSON.stringify(m.toolCalls, null, 2), "```");
  });
  return lines.join("\n") + "\n";
}

function exportActiveChat(format) {
  const s = getActiveChatSession(); const slug = (s.title || "chat").toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40) || "chat";
  const isMd = format === "md";
  const blob = new Blob([isMd ? exportChatSessionMarkdown(s) : JSON.stringify(s, null, 2)], { type: isMd ? "text/markdown" : "application/json" });
  const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = `qwegn-${slug}.${isMd ? "md" : "json"}`; a.click(); URL.revokeObjectURL(url);
}
function exportActiveChatJson() { exportActiveChat("json"); }
function exportActiveChatMarkdown() { exportActiveChat("md"); }

const CHAT_NUMERIC_API_MAP = [
  ["temperature", "temperature"], ["topP", "top_p"], ["topK", "top_k"],
  ["maxTokens", "max_tokens"], ["repetitionPenalty", "repetition_penalty"], ["seed", "seed"]
];

function serializeChatTurnForApi(msg) {
  if (msg.role === "user") {
    const hasImgs = Array.isArray(msg.images) && msg.images.length > 0;
    const hasVids = Array.isArray(msg.videos) && msg.videos.length > 0;
    if (hasImgs || hasVids) {
      const parts = [{ type: "text", text: msg.content || "" }];
      if (hasImgs) msg.images.forEach((img) => { if (img && img.dataUrl) parts.push({ type: "image_url", image_url: { url: img.dataUrl } }); });
      if (hasVids) msg.videos.forEach((vid) => { if (vid && vid.dataUrl) parts.push({ type: "video_url", video_url: { url: vid.dataUrl } }); });
      return { role: "user", content: parts };
    }
    return { role: "user", content: msg.content || "" };
  }
  if (msg.role === "tool") return { role: "tool", tool_call_id: msg.toolCallId || "", name: msg.toolName || "", content: msg.content || "" };
  const out = { role: "assistant", content: msg.content || "" };
  if (Array.isArray(msg.toolCalls) && msg.toolCalls.length > 0) {
    out.tool_calls = msg.toolCalls.map((tc) => ({
      id: tc.id || "call_0", type: tc.type || "function",
      function: { name: tc.function && tc.function.name ? tc.function.name : "", arguments: tc.function && typeof tc.function.arguments === "string" ? tc.function.arguments : "{}" }
    }));
  }
  return out;
}

function buildChatCompletionPayload(session, options) {
  const opts = options || {}; const p = session.params || createDefaultChatParams(); const messages = [];
  const sysText = typeof buildHarnessedSystemPrompt === "function" ? buildHarnessedSystemPrompt(session) : (session.systemPrompt || "").trim();
  if (sysText) messages.push({ role: "system", content: sysText });
  (Array.isArray(opts.messages) ? opts.messages : (session.messages || [])).forEach((m) => messages.push(serializeChatTurnForApi(m)));
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  const payload = { model: modelObj && modelObj.modelId ? modelObj.modelId : "resident", messages, stream: opts.forceStream !== undefined ? opts.forceStream : p.stream !== false };
  CHAT_NUMERIC_API_MAP.forEach(([localKey, apiKey]) => { if (typeof p[localKey] === "number" && Number.isFinite(p[localKey])) payload[apiKey] = p[localKey]; });
  if (p.enableThinking === "true") payload.enable_thinking = true;
  else if (p.enableThinking === "false") payload.enable_thinking = false;
  if (p.reasoningEffort && p.reasoningEffort !== "inherit") payload.reasoning_effort = p.reasoningEffort;
  if (typeof p.stop === "string" && p.stop.trim()) { const stops = p.stop.split(",").map((s) => s.trim()).filter(Boolean); if (stops.length > 0) payload.stop = stops; }
  if (p.responseFormat === "json_object") payload.response_format = { type: "json_object" };
  else if (p.responseFormat === "json_schema") {
    const schemaText = (p.jsonSchema || "").trim();
    if (!schemaText) throw new Error("JSON Schema response_format requires a schema JSON object.");
    payload.response_format = { type: "json_schema", json_schema: JSON.parse(schemaText) };
  }
  if (typeof p.toolsJson === "string" && p.toolsJson.trim()) {
    const parsedTools = JSON.parse(p.toolsJson.trim());
    if (!Array.isArray(parsedTools)) throw new Error("Tools definition must be a JSON array.");
    payload.tools = parsedTools;
  }
  if (opts.continueLastAssistant === true) payload.continue_final_message = true;
  if (typeof applyChatRequestMetadataToPayload === "function") applyChatRequestMetadataToPayload(payload, session);
  return payload;
}

function formatChatTurnMetrics(msg) {
  const parts = [];
  if (typeof msg.elapsedMs === "number") parts.push(`browser round trip: ${msg.elapsedMs} ms`);
  if (msg.requestId) parts.push(`req: ${msg.requestId}`);
  if (msg.usage) {
    if (typeof msg.usage.prompt_tokens === "number") parts.push(`prompt: ${formatTokenCount(msg.usage.prompt_tokens)}`);
    if (typeof msg.usage.completion_tokens === "number") parts.push(`gen: ${formatTokenCount(msg.usage.completion_tokens)}`);
    const cached = msg.usage.prompt_tokens_details && msg.usage.prompt_tokens_details.cached_tokens;
    if (typeof cached === "number") parts.push(`cached: ${formatTokenCount(cached)}`);
    const rToks = msg.usage.completion_tokens_details && msg.usage.completion_tokens_details.reasoning_tokens;
    if (typeof rToks === "number" && rToks > 0) parts.push(`reasoning: ${formatTokenCount(rToks)}`);
  }
  return parts.length ? `<span class="chat-turn-metrics mono">${escapeHtml(parts.join(" · "))}</span>` : "";
}

function renderToolCallCardHtml(msgId, tc, answeredToolCallIds) {
  const fnName = tc.function && tc.function.name ? tc.function.name : "tool";
  let formattedArgs = tc.function && tc.function.arguments ? tc.function.arguments : "{}";
  try { formattedArgs = JSON.stringify(JSON.parse(formattedArgs), null, 2); } catch (_) {}
  const callId = tc.id || "call_0";
  const replyForm = answeredToolCallIds.has(callId)
    ? `<div class="panel-sub mono">Tool result recorded for ${escapeHtml(callId)}.</div>`
    : `<div class="chat-tool-reply-form"><label for="tool-out-${escapeHtml(callId)}" class="chat-field-label">Submit Tool Output (<code>role: "tool"</code>)</label>
        <textarea id="tool-out-${escapeHtml(callId)}" rows="2" class="input chat-textarea mono" placeholder='{"status":"ok"}…'></textarea>
        <div class="chat-tool-reply-actions">
          <button type="button" class="btn btn-primary btn-sm" data-chat-action="submit-tool-result" data-msg-id="${escapeHtml(msgId)}" data-call-id="${escapeHtml(callId)}" data-tool-name="${escapeHtml(fnName)}" data-run-continue="true">Send Tool Output &amp; Continue</button>
          <button type="button" class="btn btn-sm" data-chat-action="submit-tool-result" data-msg-id="${escapeHtml(msgId)}" data-call-id="${escapeHtml(callId)}" data-tool-name="${escapeHtml(fnName)}" data-run-continue="false">Append Only</button>
        </div></div>`;
  return `<div class="chat-tool-call-card"><div class="chat-tool-call-head"><strong class="mono">tool_call: ${escapeHtml(fnName)}</strong><span class="knob-pill mono">${escapeHtml(callId)}</span></div><pre class="chat-tool-args mono">${escapeHtml(formattedArgs)}</pre>${replyForm}</div>`;
}

function openSmokeProbeInChat() {
  initChatWorkspace();
  const promptEl = chatEl("probe-prompt"); const maxTokEl = chatEl("probe-max-tokens");
  const promptText = promptEl ? promptEl.value.trim() : ""; const maxTokVal = maxTokEl ? parseInt(maxTokEl.value, 10) : NaN;
  const probeRaw = typeof smokeProbeState !== "undefined" && smokeProbeState ? smokeProbeState.rawText || "" : "";
  const active = getActiveChatSession();
  const s = createBlankChatSession(promptText ? promptText.replace(/\s+/g, " ").slice(0, 42) : "Smoke Probe Chat", active ? active.presetId : "server-default", active ? active.systemPrompt : "", active ? active.workspacePath : "");
  if (typeof applyPresetHarnessToSession === "function") applyPresetHarnessToSession(s, s.presetId);
  if (Number.isFinite(maxTokVal) && maxTokVal > 0) s.params.maxTokens = maxTokVal;
  if (promptText && probeRaw && !probeRaw.startsWith("Error:")) {
    const parts = typeof extractSmokeProbeParts === "function" ? extractSmokeProbeParts(probeRaw) : { reasoning: "", answer: probeRaw };
    s.messages.push({ id: generateChatId("msg"), role: "user", content: promptText, images: [], createdAt: Date.now() });
    s.messages.push({ id: generateChatId("msg"), role: "assistant", content: parts.answer || probeRaw, reasoning: parts.reasoning || "", toolCalls: [], usage: null, isRaw: false, createdAt: Date.now() });
  }
  chatState.sessions.unshift(s); chatState.activeSessionId = s.id; persistChatSessions(); renderChatWorkspace();
  if (!probeRaw && promptText && chatEl("chat-input")) { chatEl("chat-input").value = promptText; autoResizeChatInput(chatEl("chat-input")); }
  if (typeof switchTab === "function") switchTab("chat");
  notifyChat("Opened Smoke Probe conversation in Chat.");
}
