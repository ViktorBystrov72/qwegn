async function loadDiscoveredModels() {
  try {
    const res = await fetch(`${API_BASE}/v1/models/discovered`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    discoveredModelsList = Array.isArray(data.models) ? data.models : [];
    discoveredDraftersList = Array.isArray(data.drafters) ? data.drafters : [];
    renderModelsCards();
  } catch (err) {
    showToast("Failed to list models: " + err.message, true);
  }
}

function renderModelsCards() {
  const container = document.getElementById("models-cards-grid");
  if (!container) return;
  const supported = discoveredModelsList.filter((m) => m.supported !== false);
  if (supported.length === 0) {
    container.innerHTML = `<div class="panel-sub" style="padding:24px;text-align:center;">No supported models found in <code>~/.qwg/models</code>.</div>`;
    return;
  }
  const activeName = publishedModel ? publishedModel.modelId : null;
  container.innerHTML = supported.map((m) => {
    const isActive = activeName && (m.name === activeName);
    const caps = (m.capabilities || []).map((c) => `<span class="knob-pill mono">${escapeHtml(c)}</span>`).join("");
    const modelName = String(m.name || "");
    const fullPath = String(m.packPath || "");
    return `
      <div class="model-card">
        <div style="display:flex;flex-direction:column;gap:10px;min-width:0;">
          <div class="model-card-head">
            <div style="min-width:0;flex:1;">
              <div class="model-card-title">${escapeHtml(m.name)}</div>
              <div class="model-card-path mono" title="${escapeHtml(fullPath)}">${escapeHtml(middleTruncate(fullPath, 46))}</div>
            </div>
            <span class="knob-pill mono ${isActive ? "ink-secondary" : ""}" style="width:auto;flex-shrink:0;border-radius:999px;">${isActive ? "Resident" : "Installed"}</span>
          </div>
          ${caps ? `<div class="model-card-caps">${caps}</div>` : ""}
        </div>
        <div class="model-card-foot">
          <button type="button" class="btn ${isActive ? "" : "btn-primary"} btn-sm" style="white-space:nowrap;" data-select-model-name="${escapeHtml(modelName)}" data-select-model-path="${escapeHtml(fullPath)}">
            ${isActive ? "Reload Model" : "Activate"}
          </button>
        </div>
      </div>`;
  }).join("");
}

async function selectAndApplyModel(name, packPath) {
  try {
    const res = await fetch(`${API_BASE}/v1/settings`);
    if (!res.ok) throw new Error(`settings read HTTP ${res.status}`);
    const currentSettings = await res.json();
    currentSettings.modelName = name;
    currentSettings.modelPackPath = packPath;
    const saveRes = await fetch(`${API_BASE}/v1/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentSettings)
    });
    if (!saveRes.ok) {
      const errData = await saveRes.json().catch(() => ({}));
      throw new Error(errData.message || `HTTP ${saveRes.status}`);
    }
    const loadRes = await fetch(`${API_BASE}/v1/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: packPath || name })
    });
    if (!loadRes.ok) {
      const errData = await loadRes.json().catch(() => ({}));
      throw new Error(errData.message || `load HTTP ${loadRes.status}`);
    }
    showToast(`Switched active model to ${name}.`);
    fetchDashboard();
    fetchEngineInfo();
    if (document.getElementById("settings-form-host")) loadLiveSettings();
  } catch (e) {
    showToast("Failed to switch model: " + e.message, true);
  }
}

function openSwitchModelModal() {
  const dialog = document.getElementById("modal-switch-model");
  const list = document.getElementById("modal-model-options");
  if (!dialog || !list) return;
  loadDiscoveredModels().then(() => {
    const supported = discoveredModelsList.filter((m) => m.supported !== false);
    list.innerHTML = supported.map((m) => {
      const modelName = String(m.name || "");
      const packPath = String(m.packPath || "");
      return `
        <button type="button" class="btn" style="justify-content:space-between;width:100%;" data-select-model-name="${escapeHtml(modelName)}" data-select-model-path="${escapeHtml(packPath)}" data-close-switch-modal="true">
          <span>${escapeHtml(m.name)}</span>
          <span class="ink-primary mono">Select</span>
        </button>`;
    }).join("");
    if (typeof dialog.showModal === "function") dialog.showModal();
  });
}

function closeSwitchModelModal() {
  const dialog = document.getElementById("modal-switch-model");
  if (dialog && dialog.open && typeof dialog.close === "function") dialog.close();
}

async function unloadModel() {
  if (!confirm("Unload active model weights from unified memory? The HTTP server will remain available.")) return;
  try {
    const res = await fetch(`${API_BASE}/v1/models/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const reclaimed = typeof data.memoryReclaimedMB === "number"
      ? ` Unified memory reclaimed: ${data.memoryReclaimedMB} MB.` : "";
    showToast(`Model unloaded.${reclaimed}`);
  } catch (e) {
    showToast("Unload failed: " + e.message, true);
  }
}

async function reloadOrLoadModel() {
  try {
    const res = await fetch(`${API_BASE}/v1/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("Model load triggered.");
  } catch (e) {
    showToast("Load failed: " + e.message, true);
  }
}

async function restartEngine() {
  if (!confirm("Restart the resident model? Active requests will be interrupted.")) return;
  try {
    const res = await fetch(`${API_BASE}/v1/engine/restart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("Model restart requested.");
  } catch (e) {
    showToast("Restart failed: " + e.message, true);
  }
}

async function enablePrefixCacheAction() {
  try {
    const res = await fetch(`${API_BASE}/v1/settings`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    s.warmRAMLimitGB = s.warmRAMLimitGB > 0 ? s.warmRAMLimitGB : 4.0;
    const saveRes = await fetch(`${API_BASE}/v1/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s)
    });
    if (!saveRes.ok) throw new Error(`HTTP ${saveRes.status}`);
    showToast("Saved prefix cache configuration.");
  } catch (e) {
    showToast("Enable prefix cache failed: " + e.message, true);
  }
}

// --- LIVE LOGS TAB (/v1/logs/stream) ---
function initLogStream() {
  if (logsEventSource) return;
  const badge = document.getElementById("logs-stream-status");
  if (badge) badge.textContent = "Connecting…";
  logsEventSource = new EventSource(`${API_BASE}/v1/logs/stream`);
  logsEventSource.onopen = () => {
    if (badge) badge.textContent = "SSE Connected";
  };
  logsEventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      const line = String(payload.line || "").trimEnd();
      if (!line) return;
      if (logBuffer.slice(-16).includes(line)) return;
      logBuffer.push(line);
      if (logBuffer.length > 2000) logBuffer.shift();
      renderLogsView();
    } catch (err) {
      console.warn("log parse error:", err);
    }
  };
  logsEventSource.onerror = () => {
    if (badge) badge.textContent = "Reconnecting…";
  };
}

function closeLogStream() {
  if (!logsEventSource) return;
  logsEventSource.close();
  logsEventSource = null;
  const badge = document.getElementById("logs-stream-status");
  if (badge) badge.textContent = "Paused";
}

function setLogFilter(filterName) {
  activeLogFilter = filterName;
  document.querySelectorAll("[data-log-filter]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.getAttribute("data-log-filter") === filterName);
  });
  renderLogsView();
}

function toggleLogWrap() {
  logWrap = !logWrap;
  const btn = document.getElementById("logs-wrap-btn");
  if (btn) btn.textContent = `Wrap: ${logWrap ? "On" : "Off"}`;
  renderLogsView();
}

function toggleLogFollow() {
  logFollow = !logFollow;
  const btn = document.getElementById("logs-follow-btn");
  if (btn) btn.textContent = `Follow: ${logFollow ? "On" : "Off"}`;
  renderLogsView();
}

function filteredLogLines() {
  const q = (document.getElementById("logs-search")?.value || "").trim().toLowerCase();
  return logBuffer.filter((line) => {
    const isHttp = line.includes(" -> ") || /\b(GET|POST|PUT|DELETE|PATCH)\s+\//.test(line);
    let levelMatch = true;
    if (activeLogFilter === "HTTP") levelMatch = isHttp;
    else if (activeLogFilter !== "ALL") levelMatch = line.includes(` ${activeLogFilter} `) || line.includes(`[${activeLogFilter}]`);
    if (!levelMatch) return false;
    return !q || line.toLowerCase().includes(q);
  });
}

function formatLogEntryHtml(line) {
  const m = line.match(/^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]?\s+(INFO|WARN|ERROR|ERR|DEBUG)\s*(?:\|\s*)?([\s\S]*)$/);
  const isHttp = line.includes(" -> ") || /\b(GET|POST|PUT|DELETE|PATCH)\s+\//.test(line);
  if (m) {
    const [, ts, rawLvl, rest] = m;
    const lvl = rawLvl === "ERR" ? "ERROR" : rawLvl;
    const badgeCls = lvl === "ERROR" ? "log-lvl-error"
      : lvl === "WARN" ? "log-lvl-warn"
      : lvl === "DEBUG" ? "log-lvl-debug"
      : isHttp ? "log-lvl-http" : "log-lvl-info";
    const bodyCls = lvl === "ERROR" ? "ink-error"
      : lvl === "WARN" ? "ink-tertiary"
      : lvl === "DEBUG" ? "ink-muted" : "";
    return `<span class="log-ts">${escapeHtml(ts)}</span><span class="log-badge ${badgeCls}">${escapeHtml(isHttp ? "HTTP" : lvl)}</span><span class="${bodyCls}">${escapeHtml(rest)}</span>`;
  }
  const httpMatch = line.match(/^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\]?\s+((?:GET|POST|PUT|DELETE|PATCH)\s+\/[\s\S]*)$/);
  if (httpMatch) {
    const [, ts, rest] = httpMatch;
    const isErrStatus = /\b[45]\d{2}\b/.test(rest);
    const badgeCls = isErrStatus ? "log-lvl-error" : "log-lvl-http";
    const bodyCls = isErrStatus ? "ink-error" : "";
    return `<span class="log-ts">${escapeHtml(ts)}</span><span class="log-badge ${badgeCls}">HTTP</span><span class="${bodyCls}">${escapeHtml(rest)}</span>`;
  }
  const fallbackCls = (line.includes(" ERROR ") || line.includes(" ERR |")) ? "ink-error"
    : line.includes(" WARN ") ? "ink-tertiary"
    : line.includes(" DEBUG ") ? "ink-muted"
    : line.includes(" -> 200 OK") ? "ink-secondary" : "";
  return `<span class="${fallbackCls}">${escapeHtml(line)}</span>`;
}

function renderLogsView() {
  const box = document.getElementById("logs-scroll");
  if (!box) return;
  const lines = filteredLogLines();
  if (lines.length === 0) {
    box.innerHTML = `<div class="log-empty">No log entries match the active filter.</div>`;
    return;
  }
  box.innerHTML = lines.map((line) => {
    return `<div class="log-entry ${logWrap ? "" : "nowrap"}">${formatLogEntryHtml(line)}</div>`;
  }).join("");
  if (logFollow) box.scrollTop = box.scrollHeight;
}

function copyVisibleLogs() {
  const text = filteredLogLines().join("\n");
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => showToast("Copied visible logs."));
  }
}

function clearVisibleLogs() {
  logBuffer = [];
  renderLogsView();
}

async function revealLogsInFinder() {
  try {
    const res = await fetch(`${API_BASE}/v1/logs/reveal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    showToast("Reveal logs failed: " + err.message, true);
  }
}

// --- SETTINGS TAB (/v1/settings) ---
async function loadLiveSettings() {
  const status = document.getElementById("settings-status-msg");
  if (status) status.textContent = "Loading…";
  try {
    if (!Array.isArray(discoveredModelsList) || discoveredModelsList.length === 0) {
      await loadDiscoveredModels();
    }
    const res = await fetch(`${API_BASE}/v1/settings`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadedSettingsObj = await res.json();
    renderSettingsForm(loadedSettingsObj);
    updateSettingsDirtyState();
    if (status) status.textContent = "Synced";
  } catch (err) {
    if (status) status.textContent = "Load failed";
    showToast("Failed to load settings: " + err.message, true);
  }
}

function collectSettingsDraftPayload() {
  if (!loadedSettingsObj) return null;
  const payload = Object.assign({}, loadedSettingsObj);
  document.querySelectorAll("[data-setting]").forEach((el) => {
    if (el.readOnly) return;
    const key = el.getAttribute("data-setting");
    if (el.hasAttribute("data-setting-nullable") && el.value === "") {
      if (loadedSettingsObj[key] === undefined) delete payload[key];
      else if (loadedSettingsObj[key] === null) payload[key] = null;
      else payload[key] = "";
    } else {
      payload[key] = el.value;
    }
  });
  document.querySelectorAll("[data-setting-int]").forEach((el) => {
    const val = parseInt(el.value, 10);
    if (Number.isFinite(val)) payload[el.getAttribute("data-setting-int")] = val;
  });
  document.querySelectorAll("[data-setting-float]").forEach((el) => {
    const val = parseFloat(el.value);
    if (Number.isFinite(val)) payload[el.getAttribute("data-setting-float")] = val;
  });
  document.querySelectorAll("[data-setting-bool]").forEach((el) => {
    const key = el.getAttribute("data-setting-bool");
    if (el.value === "") {
      if (loadedSettingsObj[key] === undefined) delete payload[key];
      else payload[key] = null;
    } else {
      payload[key] = el.value === "true";
    }
  });
  return payload;
}

function updateSettingsDirtyState() {
  const badge = document.getElementById("settings-dirty-badge");
  const discardBtn = document.getElementById("settings-discard-btn");
  if (!loadedSettingsObj) {
    if (badge) badge.hidden = true;
    if (discardBtn) discardBtn.disabled = true;
    return;
  }
  const draft = collectSettingsDraftPayload();
  const dirty = JSON.stringify(draft) !== JSON.stringify(loadedSettingsObj);
  if (badge) badge.hidden = !dirty;
  if (discardBtn) discardBtn.disabled = !dirty;
}

function discardLiveSettings() {
  if (!loadedSettingsObj) return;
  renderSettingsForm(loadedSettingsObj);
  updateSettingsDirtyState();
  const status = document.getElementById("settings-status-msg");
  if (status) status.textContent = "Synced";
  showToast("Reverted unsaved changes.");
}

function isExternalDrafterBackend(backend) {
  const b = String(backend || "").toLowerCase();
  return b === "dflash2" || b === "dflash" || b === "dspine";
}

function compatibleDraftersForModel(modelName) {
  const list = Array.isArray(discoveredDraftersList) ? discoveredDraftersList : [];
  if (!modelName) return list;
  return list.filter((d) => Array.isArray(d.compatibleModels) && d.compatibleModels.includes(modelName));
}

function nativeMaxDraftsForModel(modelName) {
  const match = (Array.isArray(discoveredModelsList) ? discoveredModelsList : []).find((m) => m.name === modelName);
  return (match && typeof match.maxSpeculativeDrafts === "number" && match.maxSpeculativeDrafts > 0)
    ? match.maxSpeculativeDrafts
    : 7;
}

function updateMtpDraftsHint() {
  const backendEl = document.getElementById("settings-mtp-backend");
  const draftsEl = document.getElementById("settings-mtp-drafts");
  const noteEl = document.getElementById("settings-mtp-drafts-note");
  if (!draftsEl || !noteEl) return;
  const isExt = backendEl ? isExternalDrafterBackend(backendEl.value) : false;
  const count = parseInt(draftsEl.value, 10);
  if (isExt && Number.isFinite(count) && count > 0) {
    noteEl.textContent = `Block size B = ${count + 1}; sparse-MoE adaptive window trims automatically on high entropy.`;
    noteEl.hidden = false;
  } else {
    noteEl.textContent = "";
    noteEl.hidden = true;
  }
}

function handleWebMTPDraftsInput() {
  updateMtpDraftsHint();
  updateSettingsDirtyState();
}

function handleWebMTPBackendChange(selectEl) {
  const newBackend = String(selectEl && selectEl.value ? selectEl.value : "").toLowerCase();
  const isExt = isExternalDrafterBackend(newBackend);
  document.querySelectorAll(".spec-native-only").forEach((el) => { el.hidden = isExt; });
  document.querySelectorAll(".spec-dflash-only").forEach((el) => { el.hidden = !isExt; });

  const modelInput = document.querySelector("[data-setting='modelName']");
  const modelName = modelInput ? modelInput.value : (loadedSettingsObj ? loadedSettingsObj.modelName : "");
  const draftsEl = document.getElementById("settings-mtp-drafts");
  const drafterSel = document.getElementById("settings-drafter-select");
  const drafterPathEl = document.getElementById("settings-drafter-pack-path");
  const compat = compatibleDraftersForModel(modelName);

  if (isExt) {
    const currentPath = drafterPathEl ? drafterPathEl.value.trim() : "";
    const currentMatch = compat.find((d) => d.packPath === currentPath && d.backend === newBackend);
    const targetDrafter = currentMatch || compat.find((d) => d.backend === newBackend) || compat[0] || null;
    if (targetDrafter) {
      if (drafterSel) drafterSel.value = targetDrafter.packPath;
      if (drafterPathEl) drafterPathEl.value = targetDrafter.packPath;
      if (draftsEl) {
        const maxD = typeof targetDrafter.maxDrafts === "number" && targetDrafter.maxDrafts > 0 ? targetDrafter.maxDrafts : 7;
        draftsEl.max = String(maxD);
        draftsEl.value = String(maxD);
      }
    }
  } else if (draftsEl) {
    const nativeMax = nativeMaxDraftsForModel(modelName);
    draftsEl.max = String(nativeMax);
    const cur = parseInt(draftsEl.value, 10);
    if (Number.isFinite(cur) && cur > nativeMax) {
      draftsEl.value = String(Math.min(3, nativeMax));
    }
  }
  updateMtpDraftsHint();
  updateSettingsDirtyState();
}

function handleWebDrafterSelectChange(selectEl) {
  const path = selectEl ? selectEl.value : "";
  const drafterPathEl = document.getElementById("settings-drafter-pack-path");
  const backendEl = document.getElementById("settings-mtp-backend");
  const draftsEl = document.getElementById("settings-mtp-drafts");
  if (drafterPathEl) drafterPathEl.value = path;
  const match = (Array.isArray(discoveredDraftersList) ? discoveredDraftersList : []).find((d) => d.packPath === path);
  if (match) {
    if (backendEl && isExternalDrafterBackend(backendEl.value)) {
      backendEl.value = match.backend;
    }
    if (draftsEl && typeof match.maxDrafts === "number" && match.maxDrafts > 0) {
      draftsEl.max = String(match.maxDrafts);
      draftsEl.value = String(match.maxDrafts);
    }
  }
  updateMtpDraftsHint();
  updateSettingsDirtyState();
}

function renderSettingsForm(s) {
  const host = document.getElementById("settings-form-host");
  if (!host || !s) return;
  const memFacts = typeof publishedModel !== "undefined" && publishedModel && publishedModel.facts ? publishedModel.facts : null;
  const mountedMem = memFacts && memFacts.mountedMemoryPack ? memFacts : null;
  const kvEnforced = Boolean(mountedMem && mountedMem.mountedMemoryPackKV);
  const activeKV = kvEnforced ? mountedMem.mountedMemoryPackKV : s.kvMode;
  const fanStatusText = memFacts && memFacts.fanControlStatus ? memFacts.fanControlStatus : "";
  const memStatusText = mountedMem
    ? `Mounted: ${mountedMem.mountedMemoryPack}${mountedMem.mountedMemoryPackRoot ? " · " + mountedMem.mountedMemoryPackRoot : ""}`
    : "No .qwcache mounted. Enter a repo root or auto-discover via agent system prompts.";
  const isExt = isExternalDrafterBackend(s.mtpBackend);
  const compatDrafters = compatibleDraftersForModel(s.modelName);
  const dflashAvail = compatDrafters.some((d) => d.backend === "dflash2");
  const dspineAvail = compatDrafters.some((d) => d.backend === "dspine");
  const curPack = typeof s.dflashDraftPackPath === "string" ? s.dflashDraftPackPath.trim() : "";
  const matchedDrafter = compatDrafters.find((d) => d.packPath === curPack) || null;
  const nativeCap = nativeMaxDraftsForModel(s.modelName);
  const maxDraftsLimit = isExt
    ? (matchedDrafter && typeof matchedDrafter.maxDrafts === "number" ? matchedDrafter.maxDrafts : (String(s.mtpBackend || "").toLowerCase() === "dspine" ? 15 : 7))
    : nativeCap;
  const draftTokensVal = (typeof s.mtpDrafts === "number" && Number.isFinite(s.mtpDrafts)) ? s.mtpDrafts : 3;
  const strategyVal = s.mtpDraftStrategy || "hybrid";
  const topKVal = (typeof s.dflashSelectorTopK === "number" && Number.isFinite(s.dflashSelectorTopK)) ? s.dflashSelectorTopK : 16;
  const drafterOptions = compatDrafters.map((d) =>
    `<option value="${escapeHtml(d.packPath)}" ${curPack === d.packPath ? "selected" : ""}>${escapeHtml(d.name)}</option>`
  );
  if (curPack && !compatDrafters.some((d) => d.packPath === curPack)) {
    const customLabel = curPack.split("/").filter(Boolean).pop() || curPack;
    drafterOptions.unshift(`<option value="${escapeHtml(curPack)}" selected>${escapeHtml(customLabel)} (Custom Path)</option>`);
  }
  if (drafterOptions.length === 0) {
    drafterOptions.push(`<option value="" selected>No compatible drafter pack installed</option>`);
  }
  const topKPresets = [1, 4, 8, 16];
  const customTopKOpt = !topKPresets.includes(topKVal)
    ? `<option value="${topKVal}" selected>${topKVal} (Custom)</option>`
    : "";
  host.innerHTML = `
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Model &amp; Context</h3>
      <div class="form-field"><label>Active Model</label><input class="input mono" data-setting="modelName" value="${escapeHtml(s.modelName)}" readonly /></div>
      <div class="form-field"><label>Max Context Tokens <span class="restart-tag">[restart]</span></label><input class="input mono" type="number" data-setting-int="maxContext" value="${s.maxContext}" /></div>
      <div class="form-field"><label>KV Mode <span class="restart-tag">${kvEnforced ? "[memory-pack]" : "[restart]"}</span></label>
        <select class="select" data-setting="kvMode" ${kvEnforced ? "disabled" : ""}>
          <option value="f16" ${activeKV === "f16" ? "selected" : ""}>FP16</option>
          <option value="q8" ${activeKV === "q8" ? "selected" : ""}>Q8_0</option>
          <option value="q4" ${activeKV === "q4" ? "selected" : ""}>Q4_0</option>
        </select>
      </div>
      <div class="form-field"><label>Load Vision Tower</label>
        <select class="select" data-setting-bool="loadVision">
          <option value="true" ${s.loadVision ? "selected" : ""}>Enabled</option>
          <option value="false" ${!s.loadVision ? "selected" : ""}>Disabled</option>
        </select>
      </div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Sampling &amp; Reasoning</h3>
      <div class="form-field"><label>Temperature</label><input class="input mono" type="number" step="0.05" data-setting-float="temperature" value="${s.temperature}" /></div>
      <div class="form-field"><label>Top-P</label><input class="input mono" type="number" step="0.05" data-setting-float="topP" value="${s.topP}" /></div>
      <div class="form-field"><label>Top-K</label><input class="input mono" type="number" data-setting-int="topK" value="${s.topK}" /></div>
      <div class="form-field"><label>Max Tokens</label><input class="input mono" type="number" data-setting-int="maxTokens" value="${s.maxTokens}" /></div>
      <div class="form-field"><label>Repetition Penalty</label><input class="input mono" type="number" step="0.01" min="1" data-setting-float="repetitionPenalty" value="${s.repetitionPenalty !== undefined && s.repetitionPenalty !== null ? s.repetitionPenalty : ""}" placeholder="1.0" /></div>
      <div class="form-field"><label>Seed</label><input class="input mono" type="number" step="1" data-setting-int="seed" value="${s.seed !== undefined && s.seed !== null ? s.seed : ""}" placeholder="-1" /></div>
      <div class="form-field"><label>Thinking Mode</label>
        <select class="select" data-setting-bool="enableThinking">
          <option value="" ${(s.enableThinking === null || s.enableThinking === undefined) ? "selected" : ""}>Server Default</option>
          <option value="true" ${s.enableThinking === true ? "selected" : ""}>Enabled</option>
          <option value="false" ${s.enableThinking === false ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Reasoning Effort</label>
        <select class="select" data-setting="reasoningEffort" data-setting-nullable="true">
          <option value="" ${!s.reasoningEffort ? "selected" : ""}>Server Default</option>
          <option value="xhigh" ${s.reasoningEffort === "xhigh" ? "selected" : ""}>Extra High (xhigh)</option>
          <option value="high" ${s.reasoningEffort === "high" ? "selected" : ""}>High</option>
          <option value="medium" ${s.reasoningEffort === "medium" ? "selected" : ""}>Medium</option>
          <option value="low" ${s.reasoningEffort === "low" ? "selected" : ""}>Low</option>
        </select>
      </div>
      <div class="form-field"><label>Default System Prompt</label><textarea class="input mono" rows="3" style="min-height:68px;resize:vertical;font-size:12px;" data-setting="systemPrompt" data-setting-nullable="true" placeholder="Default system prompt for requests without a system turn…">${escapeHtml(s.systemPrompt !== undefined && s.systemPrompt !== null ? s.systemPrompt : "")}</textarea></div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Speculative, LSA &amp; Cooling</h3>
      <div class="form-field"><label>Speculative Decoding <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="speculativeDecodingEnabled">
          <option value="true" ${s.speculativeDecodingEnabled ? "selected" : ""}>Enabled</option>
          <option value="false" ${!s.speculativeDecodingEnabled ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>MTP Backend <span class="restart-tag">[restart]</span></label>
        <select id="settings-mtp-backend" class="select" data-setting="mtpBackend" onchange="handleWebMTPBackendChange(this)">
          <option value="standard" ${s.mtpBackend === "standard" ? "selected" : ""}>Standard</option>
          <option value="variational" ${s.mtpBackend === "variational" ? "selected" : ""}>Variational</option>
          <option value="dflash2" ${s.mtpBackend === "dflash2" ? "selected" : ""} ${(s.mtpBackend !== "dflash2" && !dflashAvail) ? "disabled" : ""}>DFlash 2${dflashAvail ? "" : " (No Pack Installed)"}</option>
          <option value="dspine" ${s.mtpBackend === "dspine" ? "selected" : ""} ${(s.mtpBackend !== "dspine" && !dspineAvail) ? "disabled" : ""}>DSpine${dspineAvail ? "" : " (No Pack Installed)"}</option>
        </select>
      </div>
      <div class="form-field"><label>MTP Draft Tokens <span class="restart-tag">[restart]</span></label>
        <input id="settings-mtp-drafts" class="input mono" type="number" min="0" max="${maxDraftsLimit}" step="1" data-setting-int="mtpDrafts" value="${draftTokensVal}" oninput="handleWebMTPDraftsInput()" />
        <div id="settings-mtp-drafts-note" class="panel-sub" style="margin-top:2px;" ${(isExt && draftTokensVal > 0) ? "" : "hidden"}>${(isExt && draftTokensVal > 0) ? `Block size B = ${draftTokensVal + 1}; sparse-MoE adaptive window trims automatically on high entropy.` : ""}</div>
      </div>
      <div class="form-field spec-native-only" ${isExt ? "hidden" : ""}><label>Draft Strategy <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="mtpDraftStrategy">
          <option value="hybrid" ${strategyVal === "hybrid" ? "selected" : ""}>Hybrid (MTP + N-Gram)</option>
          <option value="mtp" ${strategyVal === "mtp" ? "selected" : ""}>MTP Only</option>
          <option value="ngram" ${strategyVal === "ngram" ? "selected" : ""}>N-Gram Only</option>
        </select>
      </div>
      <div class="form-field spec-dflash-only" ${isExt ? "" : "hidden"}><label>Drafter Model <span class="restart-tag">[restart]</span></label>
        <select id="settings-drafter-select" class="select" onchange="handleWebDrafterSelectChange(this)">
          ${drafterOptions.join("")}
        </select>
      </div>
      <div class="form-field spec-dflash-only" ${isExt ? "" : "hidden"}><label>Drafter Pack Path <span class="restart-tag">[restart]</span></label>
        <input id="settings-drafter-pack-path" class="input mono" data-setting="dflashDraftPackPath" value="${escapeHtml(curPack)}" placeholder="~/.qwg/models/&lt;drafter-pack&gt;" />
      </div>
      <div class="form-field spec-dflash-only" ${isExt ? "" : "hidden"}><label>Selector Top-K <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-int="dflashSelectorTopK">
          ${customTopKOpt}
          <option value="1" ${topKVal === 1 ? "selected" : ""}>1 (Argmax)</option>
          <option value="4" ${topKVal === 4 ? "selected" : ""}>4</option>
          <option value="8" ${topKVal === 8 ? "selected" : ""}>8</option>
          <option value="16" ${topKVal === 16 ? "selected" : ""}>16 (Default)</option>
        </select>
      </div>
      <div class="form-field"><label>LSA Mode <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="lsaMode">
          <option value="off" ${s.lsaMode === "off" ? "selected" : ""}>Off</option>
          <option value="auto" ${s.lsaMode === "auto" ? "selected" : ""}>Auto</option>
          <option value="sidecar" ${s.lsaMode === "sidecar" ? "selected" : ""}>Sidecar</option>
          <option value="svd" ${s.lsaMode === "svd" ? "selected" : ""}>Weight-SVD</option>
          <option value="qsa-cluster" ${s.lsaMode === "qsa-cluster" ? "selected" : ""}>QSA-Cluster</option>
        </select>
      </div>
      <div class="form-field"><label>Warm RAM Cache Limit (GB) <span class="restart-tag">[restart]</span></label><input class="input mono" type="number" step="0.5" data-setting-float="warmRAMLimitGB" value="${s.warmRAMLimitGB}" /></div>
      <div class="form-field"><label>Cooling &amp; Fan Policy</label>
        <select class="select" data-setting="coolingPolicy">
          <option value="auto" ${(s.coolingPolicy || "auto") === "auto" ? "selected" : ""}>System Auto (macOS SMC)</option>
          <option value="proactive" ${s.coolingPolicy === "proactive" ? "selected" : ""}>Proactive Pre-Spin (Inference Ramp)</option>
          <option value="max" ${s.coolingPolicy === "max" ? "selected" : ""}>Max Blast (100% RPM Ceiling)</option>
        </select>
        ${fanStatusText ? `<div class="panel-sub mono" style="margin-top:4px;">Status: ${escapeHtml(fanStatusText)}</div>` : ""}
      </div>
      <div class="form-field">
        <label>Codebase &amp; Knowledge Memory Pack (.qwg)</label>
        <div id="settings-mempack-status" class="panel-sub mono" style="margin-bottom:6px;">${escapeHtml(memStatusText)}</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
          <input id="settings-mempack-path" autocomplete="off" class="input mono select-sm" style="flex:1;min-width:160px;" value="${escapeHtml(mountedMem && mountedMem.mountedMemoryPackRoot ? mountedMem.mountedMemoryPackRoot : "")}" placeholder="/Users/.../repo" />
          <button type="button" class="btn btn-sm" onclick="mountMemoryPackFromSettings()">Mount .qwcache</button>
          <button type="button" class="btn btn-sm" onclick="openSettingsMemoryPackModal()">Index / Update…</button>
          <button type="button" class="btn btn-sm" onclick="unmountMemoryPackFromSettings()" ${mountedMem ? "" : "hidden"}>Unmount</button>
        </div>
      </div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Logging</h3>
      <div class="form-field"><label>Log Level</label>
        <select class="select" data-setting="logLevel">
          <option value="debug" ${s.logLevel === "debug" ? "selected" : ""}>Debug</option>
          <option value="info" ${s.logLevel === "info" ? "selected" : ""}>Info</option>
          <option value="warn" ${s.logLevel === "warn" ? "selected" : ""}>Warn</option>
          <option value="error" ${s.logLevel === "error" ? "selected" : ""}>Error</option>
        </select>
      </div>
      <div class="form-field"><label>Log Directory <span class="restart-tag">[restart]</span></label><input class="input mono" data-setting="logDir" value="${escapeHtml(s.logDir)}" /></div>
      <div class="form-field"><label>Log File Name <span class="restart-tag">[restart]</span></label><input class="input mono" data-setting="logFileName" value="${escapeHtml(s.logFileName)}" /></div>
    </div>`;
  host.oninput = updateSettingsDirtyState;
  host.onchange = updateSettingsDirtyState;
}

async function saveLiveSettings() {
  const payload = collectSettingsDraftPayload();
  if (!payload) return;

  const status = document.getElementById("settings-status-msg");
  if (status) status.textContent = "Saving…";
  try {
    const res = await fetch(`${API_BASE}/v1/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    const verifyRes = await fetch(`${API_BASE}/v1/settings`).catch(() => null);
    loadedSettingsObj = (verifyRes && verifyRes.ok) ? await verifyRes.json() : payload;
    renderSettingsForm(loadedSettingsObj);
    updateSettingsDirtyState();
    if (status) status.textContent = "Saved";
    showToast("Settings saved to ~/.qwg/settings.json.");
    if (typeof syncChatWithServerSettings === "function") syncChatWithServerSettings(loadedSettingsObj);
    fetchDashboard();
  } catch (err) {
    if (status) status.textContent = "Save failed";
    showToast("Failed to save settings: " + err.message, true);
  }
}

async function mountMemoryPackFromSettings() {
  const input = document.getElementById("settings-mempack-path");
  const repoPath = input ? input.value.trim() : "";
  if (!repoPath) {
    showToast("Enter a workspace directory containing .qwg/manifest.json.", true);
    return;
  }
  if (typeof inspectMemoryPackAndMount === "function") {
    await inspectMemoryPackAndMount(repoPath, { forceModal: false, fromSettings: true });
    return;
  }
  if (typeof performDirectMemoryPackMount === "function") {
    await performDirectMemoryPackMount(repoPath, true);
  }
}

async function openSettingsMemoryPackModal() {
  const input = document.getElementById("settings-mempack-path");
  const chatInput = document.getElementById("chat-workspace-path");
  const repoPath = (input ? input.value.trim() : "") || (chatInput ? chatInput.value.trim() : "");
  if (!repoPath) {
    showToast("Enter a workspace directory path first.", true);
    if (input) input.focus();
    return;
  }
  if (typeof inspectMemoryPackAndMount === "function") {
    await inspectMemoryPackAndMount(repoPath, { forceModal: true, fromSettings: true });
  }
}

async function unmountMemoryPackFromSettings() {
  try {
    const res = await fetch(`${API_BASE}/v1/memory/unmount`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    showToast("Unmounted active .qwcache Memory Pack.");
    await fetchDashboard();
    await loadLiveSettings();
  } catch (err) {
    showToast("Memory Pack unmount failed: " + err.message, true);
  }
}

async function revealSettingsInFinder() {
  try {
    const res = await fetch(`${API_BASE}/v1/settings/reveal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    showToast("Reveal settings failed: " + err.message, true);
  }
}

function handleDelegatedActionClick(event) {
  const btn = event.target && event.target.closest ? event.target.closest("[data-select-model-name]") : null;
  if (!btn) return;
  const name = btn.getAttribute("data-select-model-name") || "";
  const packPath = btn.getAttribute("data-select-model-path") || "";
  selectAndApplyModel(name, packPath);
  if (btn.getAttribute("data-close-switch-modal") === "true") closeSwitchModelModal();
}

if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    const port = new URL(API_BASE, window.location.href).port;
    const portEl = document.getElementById("header-port");
    if (portEl) portEl.textContent = port ? `Port ${port}` : "Port —";

    const tabParam = new URLSearchParams(window.location.search).get("tab");
    if (tabParam) switchTab(tabParam, false);

    if (typeof document !== "undefined") {
      document.addEventListener("click", handleDelegatedActionClick);
    }

    wireSparklineHover();
    window.addEventListener("resize", syncHeroTitleMarquee);
    fetchDashboard();
    initMetricsStream();
    fetchEngineInfo();
    setInterval(fetchEngineInfo, 3000);
    window.addEventListener("beforeunload", closeLogStream);
  });
}
