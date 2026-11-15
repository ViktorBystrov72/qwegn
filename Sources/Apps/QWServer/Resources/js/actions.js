async function loadDiscoveredModels() {
  try {
    const res = await fetch(`${API_BASE}/v1/models/discovered`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    discoveredModelsList = data.models || [];
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
    const encodedName = encodeURIComponent(m.name || "");
    const encodedPath = encodeURIComponent(m.packPath || "");
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
          <button type="button" class="btn ${isActive ? "" : "btn-primary"} btn-sm" style="white-space:nowrap;" onclick="selectAndApplyModel(decodeURIComponent('${encodedName}'), decodeURIComponent('${encodedPath}'))">
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
      const encodedName = encodeURIComponent(m.name || "");
      const encodedPath = encodeURIComponent(m.packPath || "");
      return `
        <button type="button" class="btn" style="justify-content:space-between;width:100%;" onclick="selectAndApplyModel(decodeURIComponent('${encodedName}'), decodeURIComponent('${encodedPath}')); closeSwitchModelModal();">
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
    const res = await fetch(`${API_BASE}/v1/models/unload`, { method: "POST" });
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
    const res = await fetch(`${API_BASE}/v1/models/load`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast("Model load triggered.");
  } catch (e) {
    showToast("Load failed: " + e.message, true);
  }
}

async function restartEngine() {
  if (!confirm("Restart the resident model? Active requests will be interrupted.")) return;
  try {
    const res = await fetch(`${API_BASE}/v1/engine/restart`, { method: "POST" });
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
    const res = await fetch(`${API_BASE}/v1/logs/reveal`, { method: "POST" });
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
    if (!el.readOnly) payload[el.getAttribute("data-setting")] = el.value;
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
    payload[el.getAttribute("data-setting-bool")] = el.value === "true";
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

function renderSettingsForm(s) {
  const host = document.getElementById("settings-form-host");
  if (!host || !s) return;
  host.innerHTML = `
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Model &amp; Context</h3>
      <div class="form-field"><label>Active Model</label><input class="input mono" data-setting="modelName" value="${escapeHtml(s.modelName)}" readonly /></div>
      <div class="form-field"><label>Max Context Tokens <span class="restart-tag">[restart]</span></label><input class="input mono" type="number" data-setting-int="maxContext" value="${s.maxContext}" /></div>
      <div class="form-field"><label>KV Mode <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="kvMode">
          <option value="f16" ${s.kvMode === "f16" ? "selected" : ""}>FP16</option>
          <option value="q8" ${s.kvMode === "q8" ? "selected" : ""}>Q8_0</option>
          <option value="q4" ${s.kvMode === "q4" ? "selected" : ""}>Q4_0</option>
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
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Speculative &amp; LSA</h3>
      <div class="form-field"><label>Speculative Decoding <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="speculativeDecodingEnabled">
          <option value="true" ${s.speculativeDecodingEnabled ? "selected" : ""}>Enabled</option>
          <option value="false" ${!s.speculativeDecodingEnabled ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>MTP Backend <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="mtpBackend">
          <option value="standard" ${s.mtpBackend === "standard" ? "selected" : ""}>Standard</option>
          <option value="variational" ${s.mtpBackend === "variational" ? "selected" : ""}>Variational</option>
          <option value="dflash2" ${s.mtpBackend === "dflash2" ? "selected" : ""}>DFlash 2</option>
          <option value="dspine" ${s.mtpBackend === "dspine" ? "selected" : ""}>DSpine</option>
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
    loadedSettingsObj = payload;
    updateSettingsDirtyState();
    if (status) status.textContent = "Saved";
    showToast("Settings saved to ~/.qwg/settings.json.");
    fetchDashboard();
  } catch (err) {
    if (status) status.textContent = "Save failed";
    showToast("Failed to save settings: " + err.message, true);
  }
}

async function revealSettingsInFinder() {
  try {
    const res = await fetch(`${API_BASE}/v1/settings/reveal`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    showToast("Reveal settings failed: " + err.message, true);
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    const port = new URL(API_BASE, window.location.href).port;
    const portEl = document.getElementById("header-port");
    if (portEl) portEl.textContent = port ? `Port ${port}` : "Port —";

    const tabParam = new URLSearchParams(window.location.search).get("tab");
    if (tabParam) switchTab(tabParam, false);

    wireSparklineHover();
    window.addEventListener("resize", syncHeroTitleMarquee);
    fetchDashboard();
    initMetricsStream();
    fetchEngineInfo();
    setInterval(fetchEngineInfo, 3000);
    window.addEventListener("beforeunload", closeLogStream);
  });
}
