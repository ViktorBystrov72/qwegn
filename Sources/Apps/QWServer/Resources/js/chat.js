/* Dedicated local Chat & Agent workbench controller */

const CHAT_SESSIONS_STORAGE_KEY = "qwg.webui.chats.v1";
const CHAT_PRESETS_STORAGE_KEY = "qwg.webui.presets.v1";
const CHAT_UI_STATE_STORAGE_KEY = "qwg.webui.chat_ui.v1";
const CHAT_PARAM_INPUT_BINDINGS = [
  ["chat-param-temp", "temperature"], ["chat-param-top-p", "topP"], ["chat-param-top-k", "topK"],
  ["chat-param-max-tokens", "maxTokens"], ["chat-param-rep-pen", "repetitionPenalty"], ["chat-param-seed", "seed"]
];
const CHAT_NUMERIC_API_MAP = [
  ["temperature", "temperature"], ["topP", "top_p"], ["topK", "top_k"],
  ["maxTokens", "max_tokens"], ["repetitionPenalty", "repetition_penalty"], ["seed", "seed"]
];
const chatState = {
  initialized: false, sessions: [], activeSessionId: null, customPresets: [],
  sidebarCollapsed: false, openDrawer: null, pendingImages: [], editingMsgId: null,
  clearAllArmedUntil: 0, isGenerating: false, activeAbortController: null, activeStatusLabel: "Ready"
};

function chatEl(id) { return typeof document !== "undefined" ? document.getElementById(id) : null; }
function generateChatId(prefix) { return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`; }
function notifyChat(msg, isError) { if (typeof showToast === "function") showToast(msg, isError); }
function createDefaultChatParams() {
  return {
    temperature: null, topP: null, topK: null, maxTokens: null, repetitionPenalty: null, seed: null,
    enableThinking: "inherit", reasoningEffort: "inherit", stream: true, responseFormat: "text",
    jsonSchema: "", stop: "", toolsJson: "",
    cachePrompt: true, promptCacheKey: "", store: true, exactSpeculation: "inherit", customMetadata: ""
  };
}
function createBlankChatSession(title, presetId, systemPrompt, workspacePath) {
  const now = Date.now();
  return {
    id: generateChatId("chat"), title: title || "New Chat", presetId: presetId || "server-default",
    systemPrompt: typeof systemPrompt === "string" ? systemPrompt : "",
    workspacePath: typeof workspacePath === "string" ? workspacePath : "",
    harnessEnabled: true, useMemoryPackHarness: true,
    params: createDefaultChatParams(),
    messages: [], createdAt: now, updatedAt: now
  };
}
function safeReadStorage(key) {
  if (typeof localStorage === "undefined") return null;
  try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; } catch (_) { return null; }
}
function safeWriteStorage(key, value) {
  if (typeof localStorage === "undefined") return;
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (err) { notifyChat("Storage quota warning: " + err.message, true); }
}
function loadChatPersistence() {
  const savedPresets = safeReadStorage(CHAT_PRESETS_STORAGE_KEY);
  chatState.customPresets = Array.isArray(savedPresets) ? savedPresets.filter((p) => p && p.id && p.name) : [];
  const savedSessions = safeReadStorage(CHAT_SESSIONS_STORAGE_KEY);
  chatState.sessions = Array.isArray(savedSessions) && savedSessions.length > 0
    ? savedSessions.map((s) => Object.assign(createBlankChatSession(s.title, s.presetId, s.systemPrompt), s, {
        params: Object.assign(createDefaultChatParams(), s.params || {}),
        messages: Array.isArray(s.messages) ? s.messages : []
      }))
    : [createBlankChatSession("New Chat", "server-default", "")];
  const uiPrefs = safeReadStorage(CHAT_UI_STATE_STORAGE_KEY) || {};
  chatState.sidebarCollapsed = uiPrefs.sidebarCollapsed === true;
  const matched = chatState.sessions.find((s) => s.id === uiPrefs.activeSessionId);
  chatState.activeSessionId = matched ? matched.id : chatState.sessions[0].id;
}
function persistChatSessions() {
  safeWriteStorage(CHAT_SESSIONS_STORAGE_KEY, chatState.sessions);
  safeWriteStorage(CHAT_UI_STATE_STORAGE_KEY, { activeSessionId: chatState.activeSessionId, sidebarCollapsed: chatState.sidebarCollapsed });
}
function persistCustomPresets() { safeWriteStorage(CHAT_PRESETS_STORAGE_KEY, chatState.customPresets); }
function getAllSystemPresets() { return BUILT_IN_SYSTEM_PRESETS.concat(chatState.customPresets || []); }
function getActiveChatSession() {
  if (!chatState.sessions.length) { const f = createBlankChatSession("New Chat", "server-default", ""); chatState.sessions.push(f); chatState.activeSessionId = f.id; }
  let s = chatState.sessions.find((x) => x.id === chatState.activeSessionId);
  if (!s) { s = chatState.sessions[0]; chatState.activeSessionId = s.id; }
  return s;
}
function commitChatWorkspace(session) {
  if (session) session.updatedAt = Date.now();
  persistChatSessions();
  renderChatWorkspace();
}
function initChatWorkspace() { if (!chatState.initialized) { loadChatPersistence(); chatState.initialized = true; } renderChatWorkspace(); }
function onChatTabActivated() {
  initChatWorkspace();
  if (typeof loadedSettingsObj !== "undefined" && !loadedSettingsObj && typeof loadLiveSettings === "function") loadLiveSettings();
  const input = chatEl("chat-input");
  if (input && typeof input.focus === "function") setTimeout(() => input.focus(), 40);
}
function renderChatWorkspace() {
  if (typeof document === "undefined") return;
  applyChatSidebarState();
  renderChatSessionList();
  renderActiveChatHeader();
  renderChatSystemDrawer();
  renderChatParamsDrawer();
  renderChatTranscript();
  renderChatAttachments();
  renderChatComposerState();
}
function applyChatSidebarState() {
  if (chatEl("chat-layout")) chatEl("chat-layout").classList.toggle("sidebar-collapsed", chatState.sidebarCollapsed);
  if (chatEl("chat-expand-sidebar-btn")) chatEl("chat-expand-sidebar-btn").hidden = !chatState.sidebarCollapsed;
}
function toggleChatSidebar() { chatState.sidebarCollapsed = !chatState.sidebarCollapsed; persistChatSessions(); applyChatSidebarState(); }
function toggleChatDrawer(drawerName) { chatState.openDrawer = chatState.openDrawer === drawerName ? null : drawerName; syncChatDrawerVisibility(); }
function syncChatDrawerVisibility() {
  const showSys = chatState.openDrawer === "system";
  const showParams = chatState.openDrawer === "params";
  if (chatEl("chat-system-drawer")) chatEl("chat-system-drawer").hidden = !showSys;
  if (chatEl("chat-params-drawer")) chatEl("chat-params-drawer").hidden = !showParams;
  if (chatEl("chat-toggle-system-btn")) chatEl("chat-toggle-system-btn").setAttribute("aria-expanded", showSys ? "true" : "false");
  if (chatEl("chat-toggle-params-btn")) chatEl("chat-toggle-params-btn").setAttribute("aria-expanded", showParams ? "true" : "false");
}
function renderChatSessionList() {
  const listEl = chatEl("chat-session-list");
  if (!listEl) return;
  const query = chatEl("chat-session-search") ? chatEl("chat-session-search").value.trim().toLowerCase() : "";
  const sorted = chatState.sessions.slice().sort((a, b) => (b.updatedAt || -1) - (a.updatedAt || -1));
  const filtered = query ? sorted.filter((s) => (s.title || "").toLowerCase().includes(query)
    || s.messages.some((m) => typeof m.content === "string" && m.content.toLowerCase().includes(query))) : sorted;
  if (!filtered.length) { listEl.innerHTML = `<div class="panel-sub" style="padding:10px 4px;">${query ? "No matching sessions." : "No saved sessions."}</div>`; return; }
  listEl.innerHTML = filtered.map((s) => {
    const turns = s.messages.length;
    const stamp = s.updatedAt ? new Date(s.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
    return `<div class="chat-session-item ${s.id === chatState.activeSessionId ? "is-active" : ""}" role="listitem">
      <button type="button" class="chat-session-select-btn" onclick="selectChatSession('${escapeHtml(s.id)}')">
        <div class="chat-session-title">${escapeHtml(s.title || "Untitled Chat")}</div>
        <div class="chat-session-meta mono">${turns} ${turns === 1 ? "turn" : "turns"} · ${escapeHtml(stamp)}</div>
      </button>
      <button type="button" class="chat-session-del" onclick="deleteChatSession('${escapeHtml(s.id)}')" aria-label="Delete session ${escapeHtml(s.title || "Untitled")}">×</button>
    </div>`;
  }).join("");
}
function activateFreshSessionState(session) {
  chatState.activeSessionId = session.id; chatState.pendingImages = []; chatState.editingMsgId = null; commitChatWorkspace(null);
}
function createNewChatSession() {
  const a = getActiveChatSession();
  const f = createBlankChatSession("New Chat", a ? a.presetId : "server-default", a ? a.systemPrompt : "", a ? a.workspacePath : "");
  if (a && typeof a.harnessEnabled === "boolean") f.harnessEnabled = a.harnessEnabled;
  if (a && typeof a.useMemoryPackHarness === "boolean") f.useMemoryPackHarness = a.useMemoryPackHarness;
  if (typeof applyPresetHarnessToSession === "function") applyPresetHarnessToSession(f, f.presetId);
  chatState.sessions.unshift(f);
  activateFreshSessionState(f);
}
function selectChatSession(sessionId) {
  if (chatState.isGenerating) { notifyChat("Cancel active generation before switching sessions.", true); return; }
  const found = chatState.sessions.find((s) => s.id === sessionId);
  if (found) activateFreshSessionState(found);
}
function deleteChatSession(sessionId) {
  if (chatState.isGenerating && chatState.activeSessionId === sessionId) stopActiveChatGeneration();
  chatState.sessions = chatState.sessions.filter((s) => s.id !== sessionId);
  if (!chatState.sessions.length) chatState.sessions = [createBlankChatSession("New Chat", "server-default", "")];
  if (!chatState.sessions.some((s) => s.id === chatState.activeSessionId)) chatState.activeSessionId = chatState.sessions[0].id;
  commitChatWorkspace(null);
}
function clearAllChatSessions() {
  const now = Date.now();
  const btn = document.querySelector(".chat-sidebar-foot .btn-danger");
  if (now > chatState.clearAllArmedUntil) {
    chatState.clearAllArmedUntil = now + 4000;
    if (btn) btn.textContent = "Confirm Clear?";
    setTimeout(() => { if (Date.now() >= chatState.clearAllArmedUntil && btn) btn.textContent = "Clear All"; }, 4100);
    return;
  }
  chatState.clearAllArmedUntil = 0;
  if (btn) btn.textContent = "Clear All";
  if (chatState.isGenerating) stopActiveChatGeneration();
  const f = createBlankChatSession("New Chat", "server-default", "");
  chatState.sessions = [f];
  activateFreshSessionState(f);
  notifyChat("Cleared all saved chat sessions.");
}
function renameActiveChatSession(nextTitle) {
  const s = getActiveChatSession(); s.title = (nextTitle || "").trim() || "Untitled Chat"; s.updatedAt = Date.now();
  persistChatSessions(); renderChatSessionList();
}
function resetActiveChatMessages() {
  if (chatState.isGenerating) stopActiveChatGeneration();
  const s = getActiveChatSession(); s.messages = []; chatState.editingMsgId = null;
  commitChatWorkspace(s); notifyChat("Cleared conversation turns.");
}
function renderActiveChatHeader() {
  const s = getActiveChatSession(); const el = chatEl("chat-title-input");
  if (el && el.value !== s.title) el.value = s.title;
  syncChatHeaderWithDashboard();
}
function syncChatHeaderWithDashboard() {
  if (typeof document === "undefined") return;
  const connected = typeof isConnected !== "undefined" && isConnected;
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  if (chatEl("chat-model-badge")) chatEl("chat-model-badge").textContent = connected && modelObj && modelObj.modelId ? modelObj.modelId : "—";
  if (chatEl("chat-vision-badge")) {
    const caps = modelObj && Array.isArray(modelObj.capabilities) ? modelObj.capabilities : [];
    chatEl("chat-vision-badge").textContent = (!connected || !modelObj) ? "Vision: —" : (caps.includes("vision") ? "Vision: Ready" : "Text-Only");
  }
  const session = getActiveChatSession();
  if (chatEl("chat-preset-badge")) {
    const matched = getAllSystemPresets().find((p) => p.id === session.presetId);
    chatEl("chat-preset-badge").textContent = matched ? matched.name : "Custom";
  }
  if (typeof renderChatMemoryPackAndHarnessUI === "function") renderChatMemoryPackAndHarnessUI(session);
  renderChatAttachments();
}
function syncChatWithServerSettings(_) { if (typeof document !== "undefined") renderChatSystemDrawer(); }
function renderChatSystemDrawer() {
  const session = getActiveChatSession();
  const selectEl = chatEl("chat-preset-select");
  const promptEl = chatEl("chat-system-prompt");
  const hintEl = chatEl("chat-system-hint");
  const presets = getAllSystemPresets();
  if (selectEl) {
    const known = presets.some((p) => p.id === session.presetId);
    const options = presets.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name + (p.isCustom ? " [Custom]" : ""))}</option>`);
    if (!known) options.push(`<option value="custom-inline">Custom (Unsaved)</option>`);
    selectEl.innerHTML = options.join("");
    selectEl.value = known ? session.presetId : "custom-inline";
  }
  const activePreset = presets.find((p) => p.id === session.presetId);
  if (chatEl("chat-delete-preset-btn")) chatEl("chat-delete-preset-btn").hidden = !(activePreset && activePreset.isCustom);
  if (promptEl && promptEl.value !== (session.systemPrompt || "")) promptEl.value = session.systemPrompt || "";
  if (hintEl) {
    const srv = typeof loadedSettingsObj !== "undefined" && loadedSettingsObj ? loadedSettingsObj.systemPrompt : "";
    hintEl.textContent = !(session.systemPrompt || "").trim() && srv
      ? `Server default active (${srv.slice(0, 60)}${srv.length > 60 ? "…" : ""}).`
      : "Prepended as the leading system message for Warm RAM prefix-cache reuse.";
  }
  if (chatEl("chat-preset-badge")) chatEl("chat-preset-badge").textContent = activePreset ? activePreset.name : "Custom";
  if (typeof renderChatMemoryPackAndHarnessUI === "function") renderChatMemoryPackAndHarnessUI(session);
  syncChatDrawerVisibility();
}
function selectChatSystemPreset(presetId) {
  const found = getAllSystemPresets().find((p) => p.id === presetId);
  if (!found) return;
  const s = getActiveChatSession(); s.presetId = found.id; s.systemPrompt = found.prompt || "";
  if (typeof applyPresetHarnessToSession === "function") applyPresetHarnessToSession(s, found.id);
  s.updatedAt = Date.now();
  persistChatSessions(); renderChatSystemDrawer(); renderChatParamsDrawer();
}
function updateActiveChatSystemPrompt(value) {
  const s = getActiveChatSession(); s.systemPrompt = value;
  const matched = getAllSystemPresets().find((p) => p.prompt === value);
  s.presetId = matched ? matched.id : "custom-inline"; s.updatedAt = Date.now();
  persistChatSessions(); renderChatSystemDrawer();
}
function saveCustomSystemPreset() {
  const drawer = chatEl("chat-system-drawer");
  if (!drawer) return;
  if (chatEl("chat-preset-inline-save")) { if (chatEl("chat-preset-name-input")) chatEl("chat-preset-name-input").focus(); return; }
  const box = document.createElement("div");
  box.id = "chat-preset-inline-save"; box.className = "chat-preset-inline-save";
  box.innerHTML = `<label for="chat-preset-name-input" class="chat-field-label">Preset Name</label>
    <input id="chat-preset-name-input" name="chat-preset-name-input" autocomplete="off" class="input select-sm" style="max-width:220px;" placeholder="e.g. Swift Reviewer…" />
    <button type="button" class="btn btn-primary btn-sm" onclick="confirmSaveCustomSystemPreset()">Save</button>
    <button type="button" class="btn btn-sm" onclick="cancelSaveCustomSystemPreset()">Cancel</button>`;
  drawer.insertBefore(box, chatEl("chat-system-prompt"));
  if (chatEl("chat-preset-name-input")) chatEl("chat-preset-name-input").focus();
}
function confirmSaveCustomSystemPreset() {
  const nameInput = chatEl("chat-preset-name-input");
  const rawName = nameInput ? nameInput.value.trim() : "";
  if (!rawName) { notifyChat("Enter a preset name.", true); if (nameInput) nameInput.focus(); return; }
  const session = getActiveChatSession();
  let saved = chatState.customPresets.find((p) => p.name.toLowerCase() === rawName.toLowerCase());
  if (saved) { saved.name = rawName; saved.prompt = session.systemPrompt || ""; }
  else { saved = { id: generateChatId("preset"), name: rawName, prompt: session.systemPrompt || "", isCustom: true }; chatState.customPresets.push(saved); }
  session.presetId = saved.id; session.updatedAt = Date.now();
  persistCustomPresets(); persistChatSessions(); cancelSaveCustomSystemPreset(); renderChatSystemDrawer(); notifyChat(`Saved preset "${rawName}".`);
}
function cancelSaveCustomSystemPreset() { const b = chatEl("chat-preset-inline-save"); if (b && b.parentNode) b.parentNode.removeChild(b); }
function deleteCustomSystemPreset() {
  const session = getActiveChatSession();
  const target = chatState.customPresets.find((p) => p.id === session.presetId);
  if (!target) return;
  chatState.customPresets = chatState.customPresets.filter((p) => p.id !== target.id);
  session.presetId = "custom-inline"; persistCustomPresets(); persistChatSessions(); renderChatSystemDrawer(); notifyChat(`Deleted preset "${target.name}".`);
}
async function saveSystemPromptToServerDefault() {
  const promptText = getActiveChatSession().systemPrompt || "";
  try {
    let base = typeof loadedSettingsObj !== "undefined" && loadedSettingsObj ? Object.assign({}, loadedSettingsObj) : null;
    if (!base) { const r = await fetch(`${API_BASE}/v1/settings`); if (!r.ok) throw new Error(`HTTP ${r.status}`); base = await r.json(); }
    base.systemPrompt = promptText;
    const postRes = await fetch(`${API_BASE}/v1/settings`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(base) });
    const data = await postRes.json().catch(() => ({}));
    if (!postRes.ok) throw new Error(data.message || `HTTP ${postRes.status}`);
    if (typeof loadedSettingsObj !== "undefined") loadedSettingsObj = base;
    if (typeof renderSettingsForm === "function") renderSettingsForm(base);
    if (typeof updateSettingsDirtyState === "function") updateSettingsDirtyState();
    renderChatSystemDrawer(); notifyChat("Updated server default systemPrompt in ~/.qwg/settings.json.");
  } catch (err) { notifyChat("Failed to save server default: " + err.message, true); }
}
function renderChatParamsDrawer() {
  const p = getActiveChatSession().params;
  const setVal = (id, val) => { const el = chatEl(id); if (el) { const s = val === null || val === undefined ? "" : String(val); if (el.value !== s) el.value = s; } };
  CHAT_PARAM_INPUT_BINDINGS.forEach(([id, key]) => setVal(id, p[key]));
  setVal("chat-param-thinking", p.enableThinking || "inherit"); setVal("chat-param-effort", p.reasoningEffort || "inherit");
  setVal("chat-param-stream", p.stream === false ? "false" : "true"); setVal("chat-param-format", p.responseFormat || "text");
  setVal("chat-param-stop", p.stop || ""); setVal("chat-json-schema", p.jsonSchema || ""); setVal("chat-tools-json", p.toolsJson || "");
  setVal("chat-param-cache-prompt", p.cachePrompt === false ? "false" : "true");
  setVal("chat-param-cache-key", p.promptCacheKey || "");
  setVal("chat-param-store", p.store === false ? "false" : "true");
  setVal("chat-param-exact-spec", p.exactSpeculation || "inherit");
  setVal("chat-param-metadata", p.customMetadata || "");
  if (chatEl("chat-json-schema-wrap")) chatEl("chat-json-schema-wrap").hidden = (p.responseFormat || "text") !== "json_schema";
  validateChatToolsJson(p.toolsJson || "");
}
function validateChatToolsJson(raw) {
  const errEl = chatEl("chat-tools-error");
  const trimmed = (raw || "").trim();
  if (!trimmed) { if (errEl) { errEl.hidden = true; errEl.textContent = ""; } return { valid: true, tools: null }; }
  try {
    const parsed = JSON.parse(trimmed);
    if (!Array.isArray(parsed)) throw new Error("Tools definition must be a JSON array.");
    if (errEl) { errEl.hidden = true; errEl.textContent = ""; }
    return { valid: true, tools: parsed };
  } catch (err) {
    if (errEl) { errEl.hidden = false; errEl.textContent = "Invalid tools JSON: " + err.message; }
    return { valid: false, error: err.message };
  }
}
function updateChatParam(key, rawValue, kind) {
  const session = getActiveChatSession();
  let parsed = rawValue;
  if (kind === "float") { parsed = rawValue === "" ? null : parseFloat(rawValue); if (parsed !== null && !Number.isFinite(parsed)) return; }
  else if (kind === "int") { parsed = rawValue === "" ? null : parseInt(rawValue, 10); if (parsed !== null && !Number.isFinite(parsed)) return; }
  else if (kind === "bool") { parsed = Boolean(rawValue); }
  session.params[key] = parsed; session.updatedAt = Date.now(); persistChatSessions();
  if (key === "responseFormat" && chatEl("chat-json-schema-wrap")) chatEl("chat-json-schema-wrap").hidden = parsed !== "json_schema";
  if (key === "toolsJson") validateChatToolsJson(parsed);
}
function resetChatParamsToDefault() {
  const s = getActiveChatSession(); s.params = createDefaultChatParams(); s.updatedAt = Date.now();
  persistChatSessions(); renderChatParamsDrawer(); notifyChat("Reset per-session parameter overrides.");
}
function loadSampleChatTool() { const s = getActiveChatSession(); s.params.toolsJson = SAMPLE_CHAT_TOOLS_JSON; s.updatedAt = Date.now(); persistChatSessions(); renderChatParamsDrawer(); }
function clearChatToolsJson() { const s = getActiveChatSession(); s.params.toolsJson = ""; s.updatedAt = Date.now(); persistChatSessions(); renderChatParamsDrawer(); }
function serializeChatTurnForApi(msg) {
  if (msg.role === "user") {
    if (Array.isArray(msg.images) && msg.images.length > 0) {
      const parts = [{ type: "text", text: msg.content || "" }];
      msg.images.forEach((img) => { if (img && img.dataUrl) parts.push({ type: "image_url", image_url: { url: img.dataUrl } }); });
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
function renderChatTranscript() {
  const host = chatEl("chat-transcript");
  if (!host) return;
  const msgs = getActiveChatSession().messages;
  if (!msgs.length) {
    host.innerHTML = `<div class="chat-empty-state"><h3>Local Agent &amp; Chat Workbench</h3>
      <p>Multi-turn sessions stay in your browser. System instructions condition Warm RAM prefix reuse cleanly across turns.</p>
      <div class="chat-starter-row">
        <button type="button" class="btn btn-sm" onclick="applyChatStarterPrompt('Explain the trade-offs of KV cache quantization in local unified-memory inference.')">Explain KV cache quant</button>
        <button type="button" class="btn btn-sm" onclick="applyChatStarterPrompt('Write a clean Swift function with unit tests to parse NDJSON safely.')">Swift NDJSON helper</button>
        <button type="button" class="btn btn-sm" onclick="toggleChatDrawer('system')">Choose System Prompt</button>
      </div></div>`;
    return;
  }
  const answeredIds = new Set();
  msgs.forEach((m) => { if (m.role === "tool" && m.toolCallId) answeredIds.add(m.toolCallId); });
  host.innerHTML = msgs.map((m, idx) => renderSingleChatTurnHtml(m, idx, msgs.length, answeredIds)).join("");
}
function renderSingleChatTurnHtml(msg, idx, totalCount, answeredToolCallIds) {
  const session = getActiveChatSession();
  const isAssistant = msg.role === "assistant"; const isUser = msg.role === "user"; const isTool = msg.role === "tool";
  const isJsonMode = session.presetId === "json-extractor" || (session.params && session.params.responseFormat !== "text");
  const roleLabel = isUser ? "You" : (isTool ? `Tool (${msg.toolName || msg.toolCallId || "output"})` : "Assistant");
  const btns = [];
  if (isUser) btns.push(`<button type="button" class="chat-turn-btn" onclick="beginEditChatTurn('${escapeHtml(msg.id)}')">Edit &amp; Fork</button>`);
  if (isAssistant) {
    btns.push(`<button type="button" class="chat-turn-btn" onclick="regenerateChatTurn('${escapeHtml(msg.id)}')">Regenerate</button>`);
    if (idx === totalCount - 1 && !chatState.isGenerating) btns.push(`<button type="button" class="chat-turn-btn" onclick="continueChatAssistantTurn('${escapeHtml(msg.id)}')">Continue</button>`);
    if (!msg.isStreaming && msg.content && isJsonMode) btns.push(`<button type="button" class="chat-turn-btn" onclick="formatJsonChatTurn('${escapeHtml(msg.id)}')">Format JSON</button>`);
    btns.push(`<button type="button" class="chat-turn-btn" onclick="toggleChatTurnRaw('${escapeHtml(msg.id)}')">${msg.isRaw ? "Rendered" : "Raw"}</button>`);
  }
  btns.push(`<button type="button" class="chat-turn-btn" onclick="copyChatTurnContent('${escapeHtml(msg.id)}')">Copy</button>`);
  btns.push(`<button type="button" class="chat-turn-btn" onclick="deleteChatTurn('${escapeHtml(msg.id)}')" aria-label="Delete turn">×</button>`);

  let jsonBadgeHtml = "";
  if (isAssistant && !msg.isStreaming && msg.content && isJsonMode && typeof validateJsonTurnContent === "function") {
    const jv = validateJsonTurnContent(msg.content);
    jsonBadgeHtml = `<span class="knob-pill chat-json-badge ${jv.valid ? "is-valid" : "is-invalid"}" title="${escapeHtml(jv.valid ? "Strict JSON verified" : (jv.reason || "Invalid JSON"))}">${jv.valid ? "JSON Valid" : "JSON Invalid"}</span>`;
  }

  let bodyHtml = "";
  if (chatState.editingMsgId === msg.id) {
    bodyHtml = `<div class="chat-edit-wrap"><textarea id="chat-edit-textarea-${escapeHtml(msg.id)}" rows="3" class="input chat-textarea mono">${escapeHtml(msg.content || "")}</textarea>
      <div class="chat-edit-actions"><button type="button" class="btn btn-primary btn-sm" onclick="saveAndResendChatTurn('${escapeHtml(msg.id)}')">Save &amp; Resend</button>
      <button type="button" class="btn btn-sm" onclick="cancelEditChatTurn()">Cancel</button></div></div>`;
  } else {
    if (isAssistant && msg.reasoning) {
      bodyHtml += `<details class="md-thinking"${msg.isStreaming && !msg.content ? " open" : ""}><summary>Reasoning trace (${msg.reasoning.length} chars)</summary><pre>${escapeHtml(msg.reasoning)}</pre></details>`;
    }
    if (isAssistant) {
      if (msg.isRaw) bodyHtml += `<pre class="probe-raw mono">${escapeHtml(msg.content || "")}</pre>`;
      else if (msg.content) bodyHtml += `<div class="probe-markdown">${renderSmokeProbeMarkdown(msg.content)}</div>`;
      else if (!msg.toolCalls || !msg.toolCalls.length) bodyHtml += `<div class="panel-sub mono">${msg.isStreaming ? "Streaming…" : "—"}</div>`;
    } else bodyHtml += `<div class="chat-user-text">${escapeHtml(msg.content || "")}</div>`;
    if (Array.isArray(msg.images) && msg.images.length > 0) {
      bodyHtml += `<div class="chat-turn-images">${msg.images.map((img) => `<img class="chat-turn-thumb" src="${escapeHtml(img.dataUrl)}" alt="${escapeHtml(img.name || "Attached image")}" />`).join("")}</div>`;
    }
    if (Array.isArray(msg.toolCalls) && msg.toolCalls.length > 0) bodyHtml += msg.toolCalls.map((tc) => renderToolCallCardHtml(msg.id, tc, answeredToolCallIds)).join("");
    if (msg.error) bodyHtml += `<div class="chat-turn-error">${escapeHtml(msg.error)}</div>`;
  }
  const roleClass = isUser ? "chat-turn-user" : (isTool ? "chat-turn-tool" : "chat-turn-assistant");
  return `<article id="chat-turn-${escapeHtml(msg.id)}" class="chat-turn ${roleClass}">
    <div class="chat-turn-head"><div class="chat-turn-role"><span>${escapeHtml(roleLabel)}</span>${jsonBadgeHtml}${formatChatTurnMetrics(msg)}</div><div class="chat-turn-actions">${btns.join("")}</div></div>
    <div class="chat-turn-body">${bodyHtml}</div></article>`;
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
          <button type="button" class="btn btn-primary btn-sm" onclick="submitToolResultForCall('${escapeHtml(msgId)}','${escapeHtml(callId)}','${escapeHtml(fnName)}',true)">Send Tool Output &amp; Continue</button>
          <button type="button" class="btn btn-sm" onclick="submitToolResultForCall('${escapeHtml(msgId)}','${escapeHtml(callId)}','${escapeHtml(fnName)}',false)">Append Only</button>
        </div></div>`;
  return `<div class="chat-tool-call-card"><div class="chat-tool-call-head"><strong class="mono">tool_call: ${escapeHtml(fnName)}</strong><span class="knob-pill mono">${escapeHtml(callId)}</span></div><pre class="chat-tool-args mono">${escapeHtml(formattedArgs)}</pre>${replyForm}</div>`;
}
function scrollChatTranscriptToBottom() { const h = chatEl("chat-transcript"); if (h) h.scrollTop = h.scrollHeight; }
function applyChatStarterPrompt(text) { const i = chatEl("chat-input"); if (i) { i.value = text; autoResizeChatInput(i); i.focus(); } }
function autoResizeChatInput(el) { if (el) { el.style.height = "auto"; el.style.height = `${Math.min(Math.max(el.scrollHeight, 68), 220)}px`; } }
function handleChatInputKeyDown(e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!chatState.isGenerating) sendChatTurn(); } }
function triggerChatImagePicker() { const f = chatEl("chat-file-input"); if (f) f.click(); }
function handleChatFileInputChange(e) { if (e && e.target && e.target.files && e.target.files.length) queueChatImageFiles(e.target.files); if (e && e.target) e.target.value = ""; }
function handleChatDragOver(e) { e.preventDefault(); if (chatEl("chat-composer")) chatEl("chat-composer").classList.add("is-dragover"); }
function handleChatDragLeave(_) { if (chatEl("chat-composer")) chatEl("chat-composer").classList.remove("is-dragover"); }
function handleChatDrop(e) { e.preventDefault(); handleChatDragLeave(e); if (e && e.dataTransfer && e.dataTransfer.files) queueChatImageFiles(e.dataTransfer.files); }
function handleChatPaste(event) {
  const items = event && event.clipboardData ? event.clipboardData.items : null;
  if (!items) return;
  const files = [];
  for (let i = 0; i < items.length; i += 1) {
    if (items[i] && items[i].kind === "file" && items[i].type && items[i].type.startsWith("image/")) { const f = items[i].getAsFile(); if (f) files.push(f); }
  }
  if (files.length > 0) queueChatImageFiles(files);
}
function queueChatImageFiles(fileList) {
  Array.from(fileList).forEach((file) => {
    if (!file || !file.type || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") return;
      chatState.pendingImages.push({ id: generateChatId("img"), name: file.name || "pasted-image.png", mimeType: file.type, dataUrl: reader.result });
      renderChatAttachments();
    };
    reader.readAsDataURL(file);
  });
}
function removePendingChatImage(imgId) { chatState.pendingImages = chatState.pendingImages.filter((img) => img.id !== imgId); renderChatAttachments(); }
function renderChatAttachments() {
  const strip = chatEl("chat-attachments-strip");
  if (!strip) return;
  if (!chatState.pendingImages.length) { strip.hidden = true; strip.innerHTML = ""; return; }
  strip.hidden = false;
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  const connected = typeof isConnected !== "undefined" && isConnected;
  const hasVision = connected && modelObj && Array.isArray(modelObj.capabilities) && modelObj.capabilities.includes("vision");
  const warnHtml = connected && modelObj && !hasVision ? `<div class="chat-attachment-warn">Note: active model capabilities do not include vision; image parts may be rejected.</div>` : "";
  strip.innerHTML = warnHtml + chatState.pendingImages.map((img) =>
    `<div class="chat-attachment-chip"><img src="${escapeHtml(img.dataUrl)}" alt="${escapeHtml(img.name)}" /><span class="mono">${escapeHtml(img.name)}</span>
      <button type="button" class="chat-session-del" onclick="removePendingChatImage('${escapeHtml(img.id)}')" aria-label="Remove image ${escapeHtml(img.name)}">×</button></div>`
  ).join("");
}
function renderChatComposerState() {
  if (chatEl("chat-send-btn")) chatEl("chat-send-btn").disabled = chatState.isGenerating;
  if (chatEl("chat-stop-btn")) chatEl("chat-stop-btn").hidden = !chatState.isGenerating;
  if (chatEl("chat-composer-status")) chatEl("chat-composer-status").textContent = chatState.activeStatusLabel || "Ready";
}
function stopActiveChatGeneration() { if (chatState.activeAbortController) { chatState.activeAbortController.abort(); chatState.activeAbortController = null; } }
async function sendChatTurn() {
  if (chatState.isGenerating) return;
  const inputEl = chatEl("chat-input"); const text = inputEl ? inputEl.value.trim() : ""; const images = chatState.pendingImages.slice();
  if (!text && images.length === 0) return;
  const session = getActiveChatSession();
  if (session.messages.length === 0 && (session.title === "New Chat" || session.title === "Untitled Chat") && text) session.title = text.replace(/\s+/g, " ").slice(0, 42);
  session.messages.push({ id: generateChatId("msg"), role: "user", content: text, images, createdAt: Date.now() });
  if (inputEl) { inputEl.value = ""; autoResizeChatInput(inputEl); }
  chatState.pendingImages = []; commitChatWorkspace(session); scrollChatTranscriptToBottom(); await executeChatCompletion(session, {});
}
function beginEditChatTurn(msgId) {
  if (chatState.isGenerating) return;
  chatState.editingMsgId = msgId; renderChatTranscript();
  if (chatEl(`chat-edit-textarea-${msgId}`)) chatEl(`chat-edit-textarea-${msgId}`).focus();
}
function cancelEditChatTurn() { chatState.editingMsgId = null; renderChatTranscript(); }
async function saveAndResendChatTurn(msgId) {
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx === -1) return;
  const area = chatEl(`chat-edit-textarea-${msgId}`); const nextText = area ? area.value.trim() : "";
  if (!nextText && (!session.messages[idx].images || !session.messages[idx].images.length)) { notifyChat("Message text cannot be empty.", true); return; }
  session.messages[idx].content = nextText; session.messages = session.messages.slice(0, idx + 1); chatState.editingMsgId = null;
  commitChatWorkspace(session); scrollChatTranscriptToBottom(); await executeChatCompletion(session, {});
}
async function regenerateChatTurn(msgId) {
  if (chatState.isGenerating) return;
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx <= 0) return;
  session.messages = session.messages.slice(0, idx); commitChatWorkspace(session); await executeChatCompletion(session, {});
}
async function continueChatAssistantTurn(msgId) {
  if (chatState.isGenerating) return;
  const session = getActiveChatSession(); const idx = session.messages.findIndex((m) => m.id === msgId);
  if (idx === -1 || idx !== session.messages.length - 1) return;
  await executeChatCompletion(session, { continueLastAssistant: true, targetAssistantMsg: session.messages[idx] });
}
function toggleChatTurnRaw(msgId) {
  const msg = getActiveChatSession().messages.find((m) => m.id === msgId);
  if (msg) { msg.isRaw = !msg.isRaw; persistChatSessions(); renderChatTranscript(); }
}
function copyChatTurnContent(msgId) {
  const msg = getActiveChatSession().messages.find((m) => m.id === msgId);
  if (!msg) return;
  const text = msg.content || (msg.toolCalls ? JSON.stringify(msg.toolCalls, null, 2) : "");
  if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => notifyChat("Turn copied.")).catch(() => notifyChat("Clipboard unavailable.", true));
  }
}
function deleteChatTurn(msgId) {
  if (chatState.isGenerating) return;
  const s = getActiveChatSession(); s.messages = s.messages.filter((m) => m.id !== msgId); commitChatWorkspace(s);
}
async function submitToolResultForCall(msgId, toolCallId, toolName, runContinue) {
  if (chatState.isGenerating) return;
  const input = chatEl(`tool-out-${toolCallId}`); const session = getActiveChatSession();
  const idx = session.messages.findIndex((m) => m.id === msgId);
  const toolMsg = { id: generateChatId("msg"), role: "tool", toolCallId, toolName, content: (input ? input.value : "") || "{}", createdAt: Date.now() };
  if (idx !== -1) session.messages.splice(idx + 1, 0, toolMsg); else session.messages.push(toolMsg);
  commitChatWorkspace(session); scrollChatTranscriptToBottom(); if (runContinue) await executeChatCompletion(session, {});
}
async function readChatSSEStream(res, assistantMsg, startedAt, acc) {
  const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let sawDone = false;
  while (!sawDone) {
    const chunk = await reader.read(); if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const lines = buffer.split("\n"); buffer = lines.pop() || "";
    for (let i = 0; i < lines.length; i += 1) {
      const parsed = parseChatSSELine(lines[i]); if (!parsed) continue;
      if (parsed.done) { sawDone = true; break; } if (parsed.error) throw new Error(parsed.error);
      if (parsed.data && parsed.data.usage) assistantMsg.usage = parsed.data.usage;
      const delta = parsed.data && Array.isArray(parsed.data.choices) && parsed.data.choices[0] ? parsed.data.choices[0].delta : null;
      if (delta) {
        if (typeof delta.reasoning_content === "string") acc.reasoning += delta.reasoning_content;
        if (typeof delta.content === "string") acc.content += delta.content;
        if (Array.isArray(delta.tool_calls)) acc.toolCalls = mergeToolCallChunks(acc.toolCalls, delta.tool_calls);
      }
    }
    const split = splitInlineThinkTags(acc.content, acc.reasoning, true);
    assistantMsg.content = split.answer; assistantMsg.reasoning = split.reasoning; assistantMsg.toolCalls = acc.toolCalls;
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    renderChatTranscript(); scrollChatTranscriptToBottom();
  }
  const finalSplit = splitInlineThinkTags(acc.content, acc.reasoning, false);
  assistantMsg.content = finalSplit.answer; assistantMsg.reasoning = finalSplit.reasoning;
}
async function executeChatCompletion(session, opts) {
  let payload;
  try { payload = buildChatCompletionPayload(session, { continueLastAssistant: Boolean(opts.continueLastAssistant) }); }
  catch (err) { notifyChat("Invalid request parameters: " + err.message, true); return; }
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  chatState.isGenerating = true; chatState.activeAbortController = controller; chatState.activeStatusLabel = payload.stream ? "Streaming…" : "Generating…";
  const isContinue = Boolean(opts.continueLastAssistant && opts.targetAssistantMsg);
  const basePrefix = isContinue ? (opts.targetAssistantMsg.content || "") : "";
  const assistantMsg = isContinue ? opts.targetAssistantMsg : {
    id: generateChatId("msg"), role: "assistant", content: "", reasoning: "", toolCalls: [], usage: null, isStreaming: true, isRaw: false, createdAt: Date.now()
  };
  if (!isContinue) session.messages.push(assistantMsg); else { assistantMsg.isStreaming = true; assistantMsg.error = ""; }
  renderChatTranscript(); renderChatComposerState(); scrollChatTranscriptToBottom();
  const startedAt = performance.now();
  const acc = { content: basePrefix, reasoning: isContinue ? (assistantMsg.reasoning || "") : "", toolCalls: isContinue ? (assistantMsg.toolCalls || []) : [] };
  const reqId = generateChatId("req");
  const reqHeaders = { "Content-Type": "application/json", "X-Request-ID": reqId };
  if (session && session.id) reqHeaders["X-Session-ID"] = session.id;
  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST", headers: reqHeaders, body: JSON.stringify(payload), signal: controller ? controller.signal : undefined
    });
    const echoedReqId = res.headers && typeof res.headers.get === "function" ? res.headers.get("x-request-id") : "";
    assistantMsg.requestId = echoedReqId || reqId;
    if (!res.ok) { const errBody = await res.json().catch(() => ({})); throw new Error((errBody.error && errBody.error.message) || errBody.message || `HTTP ${res.status}`); }
    if (payload.stream && res.body && typeof res.body.getReader === "function") await readChatSSEStream(res, assistantMsg, startedAt, acc);
    else {
      const json = await res.json(); if (json.usage) assistantMsg.usage = json.usage;
      const msg = Array.isArray(json.choices) && json.choices[0] ? json.choices[0].message : null;
      const split = splitInlineThinkTags(basePrefix + (msg && typeof msg.content === "string" ? msg.content : ""), acc.reasoning + (msg && typeof msg.reasoning_content === "string" ? msg.reasoning_content : ""), false);
      assistantMsg.content = split.answer; assistantMsg.reasoning = split.reasoning;
      if (msg && Array.isArray(msg.tool_calls)) assistantMsg.toolCalls = msg.tool_calls;
    }
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    chatState.activeStatusLabel = `browser round trip: ${assistantMsg.elapsedMs} ms`;
    if (typeof fetchDashboard === "function") fetchDashboard();
  } catch (err) {
    assistantMsg.elapsedMs = Math.round(performance.now() - startedAt);
    if (err && err.name === "AbortError") chatState.activeStatusLabel = "Cancelled by user";
    else { assistantMsg.error = err ? err.message : "Request failed"; chatState.activeStatusLabel = "browser round trip: failed"; notifyChat("Chat request failed: " + assistantMsg.error, true); }
  } finally {
    assistantMsg.isStreaming = false; chatState.isGenerating = false; chatState.activeAbortController = null;
    commitChatWorkspace(session); scrollChatTranscriptToBottom();
  }
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

if (typeof document !== "undefined") { document.addEventListener("DOMContentLoaded", () => { initChatWorkspace(); }); }
