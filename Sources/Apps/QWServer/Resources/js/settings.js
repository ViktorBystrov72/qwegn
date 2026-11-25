// settings.js — WebUI Settings tab rendering, knob parity, and /v1/settings persistence.

function readTuningVal(s, id, fallback) {
  const raw = s && s.engineTuning && typeof s.engineTuning === "object" ? s.engineTuning[id] : null;
  if (!raw || typeof raw !== "object") return fallback;
  if (raw.string !== undefined) return raw.string;
  if (raw.int !== undefined) return raw.int;
  if (raw.double !== undefined) return raw.double;
  if (raw.bool !== undefined) return raw.bool;
  return fallback;
}

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

  const existingTuning = loadedSettingsObj.engineTuning && typeof loadedSettingsObj.engineTuning === "object"
    ? loadedSettingsObj.engineTuning
    : {};
  const tuning = Object.assign({}, existingTuning);
  document.querySelectorAll("[data-tuning-string]").forEach((el) => {
    const id = el.getAttribute("data-tuning-string");
    const def = el.getAttribute("data-tuning-default");
    if (el.value === def) delete tuning[id];
    else if (!(existingTuning[id] && existingTuning[id].string === el.value)) tuning[id] = { string: el.value };
  });
  document.querySelectorAll("[data-tuning-int]").forEach((el) => {
    const id = el.getAttribute("data-tuning-int");
    const def = parseInt(el.getAttribute("data-tuning-default"), 10);
    const val = parseInt(el.value, 10);
    if (!Number.isFinite(val) || val === def) delete tuning[id];
    else if (!(existingTuning[id] && existingTuning[id].int === val)) tuning[id] = { int: val };
  });
  document.querySelectorAll("[data-tuning-double]").forEach((el) => {
    const id = el.getAttribute("data-tuning-double");
    const def = parseFloat(el.getAttribute("data-tuning-default"));
    const val = parseFloat(el.value);
    if (!Number.isFinite(val) || val === def) delete tuning[id];
    else if (!(existingTuning[id] && existingTuning[id].double === val)) tuning[id] = { double: val };
  });
  document.querySelectorAll("[data-tuning-bool]").forEach((el) => {
    const id = el.getAttribute("data-tuning-bool");
    const def = el.getAttribute("data-tuning-default") === "true";
    const val = el.value === "true";
    if (val === def) delete tuning[id];
    else if (!(existingTuning[id] && existingTuning[id].bool === val)) tuning[id] = { bool: val };
  });
  payload.engineTuning = tuning;
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
    : 15;
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
  const isVar = newBackend === "variational";
  document.querySelectorAll(".spec-native-only").forEach((el) => { el.hidden = isExt; });
  document.querySelectorAll(".spec-dflash-only").forEach((el) => { el.hidden = !isExt; });
  document.querySelectorAll(".mtp-variational-only").forEach((el) => { el.hidden = !isVar; });

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

function handleWebLsaModeChange(selectEl) {
  const isOff = String(selectEl && selectEl.value ? selectEl.value : "") === "off";
  document.querySelectorAll(".lsa-active-only").forEach((el) => { el.hidden = isOff; });
  updateSettingsDirtyState();
}

function handleWebMTPAdaptiveChange(selectEl) {
  const thrEl = document.getElementById("settings-mtp-entropy");
  if (thrEl) thrEl.disabled = !(selectEl && selectEl.value === "true");
  updateSettingsDirtyState();
}

function handleWebColdEventsChange(selectEl) {
  const overlapEl = document.getElementById("settings-cold-overlap");
  if (overlapEl) overlapEl.disabled = !(selectEl && selectEl.value === "true");
  updateSettingsDirtyState();
}

function handleWebSsdCacheToggle(selectEl) {
  const enabled = Boolean(selectEl && selectEl.value === "true");
  const limitEl = document.getElementById("settings-ssd-limit");
  const dirEl = document.getElementById("settings-cache-dir");
  if (limitEl) limitEl.disabled = !enabled;
  if (dirEl) dirEl.disabled = !enabled;
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
  const isVar = String(s.mtpBackend || "").toLowerCase() === "variational";
  const isLsaOff = String(s.lsaMode || "auto").toLowerCase() === "off";
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
  const relaxedMarginVal = (typeof s.mtpRelaxedMargin === "number" && Number.isFinite(s.mtpRelaxedMargin)) ? s.mtpRelaxedMargin : 1.5;
  const relaxedFloorVal = (typeof s.mtpRelaxedMinProb === "number" && Number.isFinite(s.mtpRelaxedMinProb)) ? s.mtpRelaxedMinProb : 0.2;
  const quadraticVal = s.mtpQuadraticSeeding !== false;
  const adaptiveDepthVal = Boolean(s.mtpAdaptiveDepth);
  const entropyThresholdVal = (typeof s.mtpEntropyThreshold === "number" && Number.isFinite(s.mtpEntropyThreshold)) ? s.mtpEntropyThreshold : 0.7;
  const lazyLoadVal = s.lazyLoad !== false;

  const lsaBudgetVal = (typeof s.lsaBudget === "number" && Number.isFinite(s.lsaBudget)) ? s.lsaBudget : 2048;
  const lsaBudgetPresets = [512, 1024, 2048, 4096, 8192, 16384];
  const customLsaBudgetOpt = !lsaBudgetPresets.includes(lsaBudgetVal)
    ? `<option value="${lsaBudgetVal}" selected>${lsaBudgetVal.toLocaleString()} tokens (Custom)</option>`
    : "";
  const lsaBlockVal = (typeof s.lsaClusterBlock === "number" && Number.isFinite(s.lsaClusterBlock)) ? s.lsaClusterBlock : 512;
  const lsaBlockPresets = [256, 512, 1024, 4096];
  const customLsaBlockOpt = !lsaBlockPresets.includes(lsaBlockVal)
    ? `<option value="${lsaBlockVal}" selected>${lsaBlockVal} (Custom)</option>`
    : "";
  const lsaCrossLoopVal = Boolean(s.lsaCrossLoopReuse);

  const memoryPolicyVal = readTuningVal(s, "memoryPolicy", "maxResident");
  const osReserveVal = readTuningVal(s, "osReserveGiB", 8);
  const residentExpertsVal = readTuningVal(s, "residentExpertsPerLayer", 0);
  const moeSpecBudgetVal = readTuningVal(s, "moeSpecBudget", 0);
  const streamNgramsVal = Boolean(readTuningVal(s, "streamNgrams", false));
  const expertCacheVal = readTuningVal(s, "expertCache", true) !== false;
  const coldEventsVal = readTuningVal(s, "coldEvents", true) !== false;
  const coldEventOverlapVal = readTuningVal(s, "coldEventOverlap", true) !== false;
  const prefillSpanMapVal = readTuningVal(s, "prefillSpanMap", true) !== false;
  const prefillChunkVal = (typeof s.prefillChunkSize === "number" && Number.isFinite(s.prefillChunkSize)) ? s.prefillChunkSize : 512;
  const maxSessionsVal = (typeof s.maxConcurrentSessions === "number" && Number.isFinite(s.maxConcurrentSessions)) ? s.maxConcurrentSessions : 8;
  const ssdEnabledVal = Boolean(s.ssdCacheEnabled);
  const ssdLimitVal = (typeof s.ssdCacheLimitGB === "number" && Number.isFinite(s.ssdCacheLimitGB)) ? s.ssdCacheLimitGB : 10.0;

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
      <h3 style="margin:0 0 10px;font-size:13px;">Model, Loading &amp; Context</h3>
      <div class="form-field"><label>Active Model</label><input class="input mono" data-setting="modelName" value="${escapeHtml(s.modelName)}" readonly /></div>
      <div class="form-field"><label>Lazy Load on First Request <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="lazyLoad">
          <option value="true" ${lazyLoadVal ? "selected" : ""}>Enabled (Load on demand)</option>
          <option value="false" ${!lazyLoadVal ? "selected" : ""}>Disabled (Eager startup)</option>
        </select>
      </div>
      <div class="form-field"><label>Load Vision Tower</label>
        <select class="select" data-setting-bool="loadVision">
          <option value="true" ${s.loadVision ? "selected" : ""}>Enabled</option>
          <option value="false" ${!s.loadVision ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Max Context Tokens <span class="restart-tag">[restart]</span></label><input class="input mono" type="number" min="64" max="1048576" data-setting-int="maxContext" value="${s.maxContext}" /></div>
      <div class="form-field"><label>KV Mode <span class="restart-tag">${kvEnforced ? "[memory-pack]" : "[restart]"}</span></label>
        <select class="select" data-setting="kvMode" ${kvEnforced ? "disabled" : ""}>
          <option value="f16" ${activeKV === "f16" ? "selected" : ""}>FP16</option>
          <option value="q8" ${activeKV === "q8" ? "selected" : ""}>Q8_0</option>
          <option value="q4" ${activeKV === "q4" ? "selected" : ""}>Q4_0</option>
        </select>
      </div>
      <div class="form-field"><label>Decision Sidecar Model Pack (Dual-Slot)</label>
        <input id="settings-decision-pack" class="input mono" data-setting="decisionModelPackPath" value="${escapeHtml(s.decisionModelPackPath || "")}" placeholder="~/.qwg/models/&lt;decision-pack&gt; or .aimodel (empty = disabled)" />
      </div>
      <div class="form-field"><label>Decision Sidecar Compute Unit</label>
        <select id="settings-decision-unit" class="select" data-setting="decisionComputeUnit">
          <option value="auto" ${(s.decisionComputeUnit || "auto") === "auto" ? "selected" : ""}>Auto (ANE / Metal 4)</option>
          <option value="metal" ${s.decisionComputeUnit === "metal" ? "selected" : ""}>Metal 4 GPU</option>
          <option value="ane" ${s.decisionComputeUnit === "ane" ? "selected" : ""}>Apple Neural Engine (ANE)</option>
        </select>
      </div>
      <div class="form-field"><label>Load Decision Vision Tower</label>
        <select id="settings-decision-vision" class="select" data-setting-bool="decisionLoadVision">
          <option value="true" ${s.decisionLoadVision !== false ? "selected" : ""}>Enabled</option>
          <option value="false" ${s.decisionLoadVision === false ? "selected" : ""}>Disabled</option>
        </select>
      </div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Sampling &amp; Reasoning</h3>
      <div class="form-field"><label>Temperature</label><input class="input mono" type="number" step="0.05" min="0" max="2" data-setting-float="temperature" value="${s.temperature}" /></div>
      <div class="form-field"><label>Top-P</label><input class="input mono" type="number" step="0.05" min="0" max="1" data-setting-float="topP" value="${s.topP}" /></div>
      <div class="form-field"><label>Top-K</label><input class="input mono" type="number" min="0" max="2048" data-setting-int="topK" value="${s.topK}" /></div>
      <div class="form-field"><label>Max Tokens</label><input class="input mono" type="number" min="1" max="1048576" data-setting-int="maxTokens" value="${s.maxTokens}" /></div>
      <div class="form-field"><label>Repetition Penalty</label><input class="input mono" type="number" step="0.01" min="1" max="2" data-setting-float="repetitionPenalty" value="${s.repetitionPenalty !== undefined && s.repetitionPenalty !== null ? s.repetitionPenalty : ""}" placeholder="1.0" /></div>
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
          <option value="auto" ${s.reasoningEffort === "auto" ? "selected" : ""}>Auto (Decision Sidecar)</option>
          <option value="xhigh" ${s.reasoningEffort === "xhigh" ? "selected" : ""}>Extra High (xhigh)</option>
          <option value="high" ${s.reasoningEffort === "high" ? "selected" : ""}>High</option>
          <option value="medium" ${s.reasoningEffort === "medium" ? "selected" : ""}>Medium</option>
          <option value="low" ${s.reasoningEffort === "low" ? "selected" : ""}>Low</option>
        </select>
      </div>
      <div class="form-field"><label>Default System Prompt</label><textarea class="input mono" rows="3" style="min-height:68px;resize:vertical;font-size:12px;" data-setting="systemPrompt" data-setting-nullable="true" placeholder="Default system prompt for requests without a system turn…">${escapeHtml(s.systemPrompt !== undefined && s.systemPrompt !== null ? s.systemPrompt : "")}</textarea></div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Speculative Decoding</h3>
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
          <option value="dspark" ${s.mtpBackend === "dspark" ? "selected" : ""}>DSpark</option>
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
      <div class="form-field mtp-variational-only" ${isVar ? "" : "hidden"}><label>Relaxed Margin (ε nats) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="0.1" min="0" max="5" data-setting-float="mtpRelaxedMargin" value="${relaxedMarginVal}" />
      </div>
      <div class="form-field mtp-variational-only" ${isVar ? "" : "hidden"}><label>Fidelity Floor (τ) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="0.05" min="0" max="0.5" data-setting-float="mtpRelaxedMinProb" value="${relaxedFloorVal}" />
      </div>
      <div class="form-field mtp-variational-only" ${isVar ? "" : "hidden"}><label>Quadratic Prefix Seeding <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="mtpQuadraticSeeding">
          <option value="true" ${quadraticVal ? "selected" : ""}>Enabled</option>
          <option value="false" ${!quadraticVal ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field spec-native-only" ${isExt ? "hidden" : ""}><label>Adaptive Draft Depth (Entropy Stop) <span class="restart-tag">[restart]</span></label>
        <select id="settings-mtp-adaptive" class="select" data-setting-bool="mtpAdaptiveDepth" onchange="handleWebMTPAdaptiveChange(this)">
          <option value="false" ${!adaptiveDepthVal ? "selected" : ""}>Disabled</option>
          <option value="true" ${adaptiveDepthVal ? "selected" : ""}>Enabled</option>
        </select>
      </div>
      <div class="form-field spec-native-only" ${isExt ? "hidden" : ""}><label>MTP Entropy Threshold <span class="restart-tag">[restart]</span></label>
        <input id="settings-mtp-entropy" class="input mono" type="number" step="0.05" min="0.05" max="1" data-setting-float="mtpEntropyThreshold" value="${entropyThresholdVal}" ${adaptiveDepthVal ? "" : "disabled"} />
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
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Latent Sparse Attention (LSA)</h3>
      <div class="form-field"><label>LSA Mode <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="lsaMode" onchange="handleWebLsaModeChange(this)">
          <option value="off" ${s.lsaMode === "off" ? "selected" : ""}>Off</option>
          <option value="auto" ${s.lsaMode === "auto" ? "selected" : ""}>Auto</option>
          <option value="sidecar" ${s.lsaMode === "sidecar" ? "selected" : ""}>Sidecar (.qlsa)</option>
          <option value="svd" ${s.lsaMode === "svd" ? "selected" : ""}>Weight-SVD</option>
          <option value="qsa-cluster" ${s.lsaMode === "qsa-cluster" ? "selected" : ""}>QSA-Cluster</option>
        </select>
      </div>
      <div class="form-field lsa-active-only" ${isLsaOff ? "hidden" : ""}><label>Key Budget (k) <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-int="lsaBudget">
          ${customLsaBudgetOpt}
          <option value="512" ${lsaBudgetVal === 512 ? "selected" : ""}>512 tokens</option>
          <option value="1024" ${lsaBudgetVal === 1024 ? "selected" : ""}>1,024 tokens</option>
          <option value="2048" ${lsaBudgetVal === 2048 ? "selected" : ""}>Auto / 2,048 (Adaptive ≤16K)</option>
          <option value="4096" ${lsaBudgetVal === 4096 ? "selected" : ""}>4,096 tokens</option>
          <option value="8192" ${lsaBudgetVal === 8192 ? "selected" : ""}>8,192 tokens</option>
          <option value="16384" ${lsaBudgetVal === 16384 ? "selected" : ""}>16,384 tokens</option>
        </select>
      </div>
      <div class="form-field lsa-active-only" ${isLsaOff ? "hidden" : ""}><label>Cluster Block (L) <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-int="lsaClusterBlock">
          ${customLsaBlockOpt}
          <option value="256" ${lsaBlockVal === 256 ? "selected" : ""}>256 tokens</option>
          <option value="512" ${lsaBlockVal === 512 ? "selected" : ""}>512 tokens (Default)</option>
          <option value="1024" ${lsaBlockVal === 1024 ? "selected" : ""}>1,024 tokens</option>
          <option value="4096" ${lsaBlockVal === 4096 ? "selected" : ""}>4,096 (e-means)</option>
        </select>
      </div>
      <div class="form-field lsa-active-only" ${isLsaOff ? "hidden" : ""}><label>Custom .qlsa Sidecar Path <span class="restart-tag">[restart]</span></label>
        <input class="input mono" data-setting="lsaSidecarPath" value="${escapeHtml(s.lsaSidecarPath || "")}" placeholder="Auto-detected from &lt;model_dir&gt;/model.qlsa" />
      </div>
      <div class="form-field lsa-active-only" ${isLsaOff ? "hidden" : ""}><label>Cross-Loop Block Reuse (Looped Attention) <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="lsaCrossLoopReuse">
          <option value="false" ${!lsaCrossLoopVal ? "selected" : ""}>Disabled (Full-Slice Recall)</option>
          <option value="true" ${lsaCrossLoopVal ? "selected" : ""}>Enabled (Cross-Loop Reuse)</option>
        </select>
      </div>
      <div class="panel-sub" style="margin-top:4px;line-height:1.45;">GQA ≤ k+516 runs dense; k=2,048 scales adaptively (4K at 98K+, 8K at 196K+, 16K at 786K+). Keep Cross-Loop Reuse off for full-slice recall.</div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Memory, Prefill &amp; MoE Tuning</h3>
      <div class="form-field"><label>Memory Policy <span class="restart-tag">[restart]</span></label>
        <select class="select" data-tuning-string="memoryPolicy" data-tuning-default="maxResident">
          <option value="maxResident" ${memoryPolicyVal === "maxResident" ? "selected" : ""}>Max Resident (Fastest)</option>
          <option value="safe" ${memoryPolicyVal === "safe" ? "selected" : ""}>Safe (No Swap)</option>
        </select>
      </div>
      <div class="form-field"><label>OS Reserve (GiB) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="1" min="0" max="256" data-tuning-double="osReserveGiB" data-tuning-default="8" value="${osReserveVal}" />
      </div>
      <div class="form-field"><label>Scratch Eviction <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="evictPrefillScratch">
          <option value="auto" ${(s.evictPrefillScratch || "auto") === "auto" ? "selected" : ""}>Auto</option>
          <option value="always" ${s.evictPrefillScratch === "always" ? "selected" : ""}>Always</option>
          <option value="never" ${s.evictPrefillScratch === "never" ? "selected" : ""}>Never</option>
        </select>
      </div>
      <div class="form-field"><label>Prefill Chunk Rows <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="32" min="32" max="8192" data-setting-int="prefillChunkSize" value="${prefillChunkVal}" />
      </div>
      <div class="form-field"><label>GDN Prefill <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="gdnPrefill">
          <option value="chunked" ${(s.gdnPrefill || "chunked") === "chunked" ? "selected" : ""}>Chunked (Default)</option>
          <option value="regres" ${s.gdnPrefill === "regres" ? "selected" : ""}>Regression (Parallel)</option>
          <option value="naive" ${s.gdnPrefill === "naive" ? "selected" : ""}>Naive (Stepwise)</option>
        </select>
      </div>
      <div class="form-field"><label>Stream PLE N-grams from SSD <span class="restart-tag">[restart]</span></label>
        <select class="select" data-tuning-bool="streamNgrams" data-tuning-default="false">
          <option value="false" ${!streamNgramsVal ? "selected" : ""}>Disabled (Resident RAM)</option>
          <option value="true" ${streamNgramsVal ? "selected" : ""}>Enabled (Stream ngram.bin)</option>
        </select>
      </div>
      <div class="form-field"><label>Resident Experts / Layer (0 = Auto) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="1" min="0" max="512" data-tuning-int="residentExpertsPerLayer" data-tuning-default="0" value="${residentExpertsVal}" />
      </div>
      <div class="form-field"><label>MoE-Spec Verify Budget (0 = Lossless) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="1" min="0" max="64" data-tuning-int="moeSpecBudget" data-tuning-default="0" value="${moeSpecBudgetVal}" />
      </div>
      <div class="form-field"><label>Cross-Token Expert Cache <span class="restart-tag">[restart]</span></label>
        <select class="select" data-tuning-bool="expertCache" data-tuning-default="true">
          <option value="true" ${expertCacheVal ? "selected" : ""}>Enabled</option>
          <option value="false" ${!expertCacheVal ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Event-Pipelined Cold Expert Staging <span class="restart-tag">[restart]</span></label>
        <select id="settings-cold-events" class="select" data-tuning-bool="coldEvents" data-tuning-default="true" onchange="handleWebColdEventsChange(this)">
          <option value="true" ${coldEventsVal ? "selected" : ""}>Enabled</option>
          <option value="false" ${!coldEventsVal ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Overlap Resident &amp; Cold Experts <span class="restart-tag">[restart]</span></label>
        <select id="settings-cold-overlap" class="select" data-tuning-bool="coldEventOverlap" data-tuning-default="true" ${coldEventsVal ? "" : "disabled"}>
          <option value="true" ${coldEventOverlapVal ? "selected" : ""}>Enabled</option>
          <option value="false" ${!coldEventOverlapVal ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Mapped Expert Spans per Prefill Chunk <span class="restart-tag">[restart]</span></label>
        <select class="select" data-tuning-bool="prefillSpanMap" data-tuning-default="true">
          <option value="true" ${prefillSpanMapVal ? "selected" : ""}>Enabled</option>
          <option value="false" ${!prefillSpanMapVal ? "selected" : ""}>Disabled</option>
        </select>
      </div>
      <div class="form-field"><label>Cooling &amp; Fan Policy</label>
        <select class="select" data-setting="coolingPolicy">
          <option value="auto" ${(s.coolingPolicy || "auto") === "auto" ? "selected" : ""}>System Auto (macOS SMC)</option>
          <option value="proactive" ${s.coolingPolicy === "proactive" ? "selected" : ""}>Proactive Pre-Spin (Inference Ramp)</option>
          <option value="max" ${s.coolingPolicy === "max" ? "selected" : ""}>Max Blast (100% RPM Ceiling)</option>
        </select>
        ${fanStatusText ? `<div class="panel-sub mono" style="margin-top:4px;">Status: ${escapeHtml(fanStatusText)}</div>` : ""}
      </div>
    </div>
    <div class="metric-cell" style="padding:14px;">
      <h3 style="margin:0 0 10px;font-size:13px;">Prefix Cache, Logging &amp; Updates</h3>
      <div class="form-field"><label>Warm RAM Cache Limit (GB; &gt;0 enables prefix cache) <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="0.5" min="0" data-setting-float="warmRAMLimitGB" value="${s.warmRAMLimitGB}" />
      </div>
      <div class="form-field"><label>Concurrent Session Slots <span class="restart-tag">[restart]</span></label>
        <input class="input mono" type="number" step="1" min="1" max="16" data-setting-int="maxConcurrentSessions" value="${maxSessionsVal}" />
      </div>
      <div class="form-field"><label>Persist Prefix Cache to Cold SSD <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting-bool="ssdCacheEnabled" onchange="handleWebSsdCacheToggle(this)">
          <option value="false" ${!ssdEnabledVal ? "selected" : ""}>Disabled</option>
          <option value="true" ${ssdEnabledVal ? "selected" : ""}>Enabled</option>
        </select>
      </div>
      <div class="form-field"><label>Cold SSD Limit (GB) <span class="restart-tag">[restart]</span></label>
        <input id="settings-ssd-limit" class="input mono" type="number" step="0.5" min="0" data-setting-float="ssdCacheLimitGB" value="${ssdLimitVal}" ${ssdEnabledVal ? "" : "disabled"} />
      </div>
      <div class="form-field"><label>Cache Directory <span class="restart-tag">[restart]</span></label>
        <input id="settings-cache-dir" class="input mono" data-setting="cacheDir" value="${escapeHtml(s.cacheDir || "~/.qwg/cache")}" placeholder="~/.qwg/cache" ${ssdEnabledVal ? "" : "disabled"} />
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
      <div class="form-field"><label>Log Level <span class="restart-tag">[restart]</span></label>
        <select class="select" data-setting="logLevel">
          <option value="debug" ${s.logLevel === "debug" ? "selected" : ""}>Debug</option>
          <option value="info" ${s.logLevel === "info" ? "selected" : ""}>Info</option>
          <option value="warn" ${s.logLevel === "warn" ? "selected" : ""}>Warn</option>
          <option value="error" ${s.logLevel === "error" ? "selected" : ""}>Error</option>
          <option value="none" ${s.logLevel === "none" ? "selected" : ""}>None</option>
        </select>
      </div>
      <div class="form-field"><label>Log Directory</label><input class="input mono" data-setting="logDir" value="${escapeHtml(s.logDir)}" /></div>
      <div class="form-field"><label>Log File Name</label><input class="input mono" data-setting="logFileName" value="${escapeHtml(s.logFileName)}" /></div>
      <div class="form-field"><label>Check Automatically</label>
        <select class="select" data-setting="updateCheckSchedule">
          <option value="onLaunchAndDaily" ${(s.updateCheckSchedule || "onLaunchAndDaily") === "onLaunchAndDaily" ? "selected" : ""}>On launch &amp; daily</option>
          <option value="daily" ${s.updateCheckSchedule === "daily" ? "selected" : ""}>Daily (every 24h)</option>
          <option value="onLaunch" ${s.updateCheckSchedule === "onLaunch" ? "selected" : ""}>On launch only</option>
          <option value="manual" ${s.updateCheckSchedule === "manual" ? "selected" : ""}>Manual only</option>
        </select>
      </div>
      <div id="settings-updates-host" class="update-settings-status"></div>
    </div>`;
  host.oninput = updateSettingsDirtyState;
  host.onchange = updateSettingsDirtyState;
  if (typeof renderWebUpdateSettingsCard === "function") renderWebUpdateSettingsCard();
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
    if (typeof fetchUpdateStatus === "function") fetchUpdateStatus();
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
