// decisions.js — Offline /ui?tab=decisions workbench for single-pass decision evaluation.

let decisionWorkbenchState = {
  initialized: false,
  sidecarStatus: null,
  stateMode: "text",
  stateText: "Patient reports mild fever (38.1 C) for 2 days without chest pain.",
  chain: "default",
  truncation: "auto_state",
  scoreReadout: "mean",
  images: [],
  questions: [
    {
      name: "triage_tier",
      kind: "choice",
      instructions: "Classify urgency tier",
      options: "routine,urgent,emergency",
      minVal: "",
      maxVal: ""
    }
  ],
  lastResponse: null,
  lastEndpoint: "/v1/decisions"
};

function formatDecisionBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GiB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  return `${bytes} B`;
}

function decisionConfidenceRailBadge(conf) {
  if (typeof conf !== "number" || !Number.isFinite(conf)) {
    return `<span class="dec-pill mono">Confidence —</span>`;
  }
  const pct = (conf * 100).toFixed(1);
  const acting = conf >= 0.7;
  const cls = acting ? "is-acting" : "is-failopen";
  const label = acting ? `Acting ≥ 0.70 (${pct}%)` : `Fail-Open < 0.70 (${pct}%)`;
  return `<span class="dec-pill mono ${cls}">${escapeHtml(label)}</span>`;
}

function extractDecisionSpanRange(ans) {
  if (!ans || !ans.span) return null;
  const sp = ans.span;
  let lo = null;
  let hi = null;
  if (Array.isArray(sp) && sp.length >= 2) {
    lo = Number(sp[0]);
    hi = Number(sp[1]);
  } else if (typeof sp === "object") {
    lo = typeof sp.lowerBound === "number" ? sp.lowerBound : (typeof sp.start === "number" ? sp.start : null);
    hi = typeof sp.upperBound === "number" ? sp.upperBound : (typeof sp.end === "number" ? sp.end : null);
  }
  if (lo === null || hi === null || !Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
  return { lo: Math.max(0, Math.floor(lo)), hi: Math.max(0, Math.floor(hi)) };
}

function renderDecisionSpanHighlightHTML(stateText, ans) {
  const text = String(stateText || "");
  const r = extractDecisionSpanRange(ans);
  if (!text || !r) return "";
  let before = "";
  let mid = "";
  let after = "";
  if (typeof TextEncoder === "function" && typeof TextDecoder === "function") {
    const enc = new TextEncoder().encode(text);
    const dec = new TextDecoder("utf-8", { fatal: false });
    const lo = Math.min(r.lo, enc.length);
    const hi = Math.min(r.hi, enc.length);
    before = dec.decode(enc.slice(0, lo));
    mid = dec.decode(enc.slice(lo, hi));
    after = dec.decode(enc.slice(hi));
  } else {
    before = text.slice(0, r.lo);
    mid = text.slice(r.lo, r.hi);
    after = text.slice(r.hi);
  }
  if (!mid) return "";
  return `<div class="dec-span-preview">${escapeHtml(before)}<mark class="dec-span-mark">${escapeHtml(mid)}</mark>${escapeHtml(after)}</div>`;
}

function renderDecisionBoxesSVGHTML(imgSrc, boxes) {
  if (!imgSrc || !Array.isArray(boxes) || boxes.length === 0) return "";
  const rects = boxes.map((b, idx) => {
    const left = Math.max(0, Math.min(1, Number(b.left || 0))) * 1000;
    const top = Math.max(0, Math.min(1, Number(b.top || 0))) * 1000;
    const right = Math.max(0, Math.min(1, Number(b.right || 0))) * 1000;
    const bottom = Math.max(0, Math.min(1, Number(b.bottom || 0))) * 1000;
    const w = Math.max(2, right - left);
    const h = Math.max(2, bottom - top);
    const conf = typeof b.confidence === "number" ? `${(b.confidence * 100).toFixed(0)}%` : "—";
    const ty = Math.max(18, top - 6);
    return `<g>
      <rect x="${left.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="rgba(56,139,253,0.14)" stroke="#56d4ff" stroke-width="5" />
      <text x="${left.toFixed(1)}" y="${ty.toFixed(1)}" fill="#56d4ff" font-size="26" font-family="monospace">#${idx + 1} (${escapeHtml(conf)})</text>
    </g>`;
  }).join("");
  return `<div class="dec-box-stage"><img src="${escapeHtml(imgSrc)}" alt="Decision vision preview" /><svg class="dec-box-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none">${rects}</svg></div>`;
}

function summarizeDecisionAnswerValue(ans) {
  if (!ans) return "—";
  if (ans.abstained) return "Abstained";
  if (ans.noul !== null && ans.noul !== undefined) return ans.noul ? "true (yes)" : "false (no)";
  if (typeof ans.choice === "string" && ans.choice) return ans.choice;
  if (Array.isArray(ans.selectedChoices) && ans.selectedChoices.length) return ans.selectedChoices.join(", ");
  if (typeof ans.score === "number") return ans.score.toFixed(3);
  if (typeof ans.estimate === "number") return ans.estimate.toFixed(3);
  if (typeof ans.number === "number") return String(ans.number);
  if (typeof ans.date === "string" && ans.date) return ans.date;
  if (typeof ans.text === "string" && ans.text) return ans.text;
  if (Array.isArray(ans.boxes) && ans.boxes.length) return `${ans.boxes.length} box(es)`;
  if (ans.found === false) return "not found";
  return "—";
}

function renderDecisionResultsHTML(resp, stateText, imagesList) {
  if (!resp || !Array.isArray(resp.answers)) {
    return `<div class="panel-sub">Submit a decision query to inspect calibrated distributions, spans, boxes, and confidence rails.</div>`;
  }
  const usage = resp.usage || {};
  const latText = typeof usage.latencyMs === "number" ? `${usage.latencyMs.toFixed(2)} ms` : (typeof usage.latency_ms === "number" ? `${Number(usage.latency_ms).toFixed(2)} ms` : "—");
  const tokText = typeof usage.promptTokens === "number" ? `${usage.promptTokens} tok` : (typeof usage.prompt_tokens === "number" ? `${usage.prompt_tokens} tok` : "—");
  const metaHtml = `<div class="dec-status-meta" style="margin-bottom:8px;">
    <span class="dec-pill mono">Backend: ${escapeHtml(resp.backend || "—")}</span>
    <span class="dec-pill mono">Model: ${escapeHtml(resp.model || "—")}</span>
    <span class="dec-pill mono">Latency: ${escapeHtml(latText)}</span>
    <span class="dec-pill mono">Prompt: ${escapeHtml(tokText)}</span>
  </div>`;

  const firstImg = Array.isArray(imagesList) && imagesList.length ? imagesList[0] : null;
  const answersHtml = resp.answers.map((ans) => {
    const valStr = summarizeDecisionAnswerValue(ans);
    const probs = Array.isArray(ans.probs) ? ans.probs.slice().sort((a, b) => (b.value || 0) - (a.value || 0)) : [];
    const probsHtml = probs.length
      ? `<div class="dec-prob-list">${probs.map((p) => {
          const v = Math.max(0, Math.min(1, Number(p.value || 0)));
          const pct = (v * 100).toFixed(1);
          return `<div class="dec-prob-row">
            <div class="dec-prob-head mono"><span>${escapeHtml(p.key || "")}</span><span>${pct}%</span></div>
            <div class="bar-bg"><div class="bar-fg" style="width:${pct}%;background:var(--accent);"></div></div>
          </div>`;
        }).join("")}</div>`
      : "";
    const spanHtml = renderDecisionSpanHighlightHTML(stateText, ans);
    const boxHtml = Array.isArray(ans.boxes) && ans.boxes.length && firstImg
      ? renderDecisionBoxesSVGHTML(firstImg, ans.boxes)
      : (Array.isArray(ans.boxes) && ans.boxes.length
          ? `<div class="panel-sub mono" style="margin-top:6px;">${ans.boxes.map((b, i) => `#${i + 1} [${Number(b.left).toFixed(3)}, ${Number(b.top).toFixed(3)}, ${Number(b.right).toFixed(3)}, ${Number(b.bottom).toFixed(3)}]`).join(" · ")}</div>`
          : "");
    return `<div class="dec-q-card">
      <div class="dec-status-strip">
        <strong class="mono">${escapeHtml(ans.name || "q")} · ${escapeHtml(ans.kind || "decision")}</strong>
        ${decisionConfidenceRailBadge(ans.confidence)}
      </div>
      <div class="mono" style="font-size:13px;color:var(--ink);">Answer: <strong>${escapeHtml(valStr)}</strong></div>
      ${probsHtml}
      ${spanHtml}
      ${boxHtml}
    </div>`;
  }).join("");
  return metaHtml + answersHtml;
}

function buildDecisionRequestPayload(st) {
  const s = st || decisionWorkbenchState;
  let parsedState = s.stateText || "";
  if (s.stateMode === "json") {
    try { parsedState = JSON.parse(parsedState || "{}"); } catch (_) { parsedState = s.stateText || ""; }
  }
  const questions = (s.questions || []).map((q, idx) => {
    const out = {
      name: (q.name || `q${idx}`).trim() || `q${idx}`,
      kind: q.kind || "noul",
      instructions: q.instructions || "",
      score_readout: s.scoreReadout || "mean",
      chain: s.chain || "default"
    };
    const rawOpts = String(q.options || "").split(",").map((x) => x.trim()).filter(Boolean);
    if (rawOpts.length > 0 && ["choice", "multiple", "score"].includes(out.kind)) {
      out.options = rawOpts;
    }
    if (out.kind === "estimate") {
      if (q.minVal !== "" && Number.isFinite(Number(q.minVal))) out.min = Number(q.minVal);
      if (q.maxVal !== "" && Number.isFinite(Number(q.maxVal))) out.max = Number(q.maxVal);
    }
    return out;
  });
  const body = {
    state: parsedState,
    questions: questions,
    truncation: s.truncation || "auto_state",
    chain: s.chain || "default"
  };
  if (Array.isArray(s.images) && s.images.length > 0) {
    body.images = s.images;
  }
  return body;
}

function buildDecisionSnippets(payload, endpointUrl) {
  const base = endpointUrl || API_BASE;
  const safePayload = Object.assign({}, payload);
  if (Array.isArray(safePayload.images) && safePayload.images.length > 0) {
    safePayload.images = safePayload.images.map((_, i) => `data:image/png;base64,<IMAGE_${i}>`);
  }
  const jsonStr = JSON.stringify(safePayload, null, 2);
  const curlDecisions = `curl -sS ${base}/v1/decisions \\\n  -H "Content-Type: application/json" \\\n  -d '${jsonStr}'`;
  const curlSystemOne = `curl -sS ${base}/v1/systemone \\\n  -H "Content-Type: application/json" \\\n  -d '${jsonStr}'`;
  const hermesMcp = `hermes mcp add qwegine --transport http --url ${base}/mcp`;
  return { curlDecisions, curlSystemOne, hermesMcp };
}

function updateDecisionSidecarPanel(model) {
  const strip = document.getElementById("dec-sidecar-banner");
  if (!strip) return;
  const facts = model && model.facts ? model.facts : null;
  const status = decisionWorkbenchState.sidecarStatus;
  const loaded = Boolean((facts && facts.decisionSidecarLoaded) || (status && status.loaded));
  const modelName = (facts && facts.decisionSidecarModel) || (status && status.model_name) || "No sidecar loaded";
  const backend = (facts && facts.decisionSidecarBackend) || (status && status.backend_kind) || "—";
  const bytesVal = (facts && facts.decisionSidecarBytes !== undefined) ? facts.decisionSidecarBytes : (status ? status.estimated_bytes : null);
  const latVal = (facts && facts.decisionSidecarMeanLatencyMs !== undefined) ? facts.decisionSidecarMeanLatencyMs : (status ? status.mean_latency_ms : null);
  const reqVal = (facts && facts.decisionSidecarTotalRequests !== undefined) ? facts.decisionSidecarTotalRequests : (status ? status.total_requests : null);
  const latStr = typeof latVal === "number" && Number.isFinite(latVal) ? `${latVal.toFixed(1)} ms` : "—";
  const reqStr = typeof reqVal === "number" && Number.isFinite(reqVal) ? String(reqVal) : "—";

  strip.innerHTML = `
    <div class="dec-status-meta">
      <span class="dot ${loaded ? "dot-active" : "dot-idle"}"></span>
      <strong class="mono">${escapeHtml(loaded ? modelName : "Decision Sidecar Unloaded (falls back to primary when compatible)")}</strong>
      ${loaded ? `<span class="dec-pill mono">${escapeHtml(backend)}</span>` : ""}
      ${loaded ? `<span class="dec-pill mono">Mem: ${escapeHtml(formatDecisionBytes(bytesVal))}</span>` : ""}
      ${loaded ? `<span class="dec-pill mono">Mean: ${escapeHtml(latStr)}</span>` : ""}
      ${loaded ? `<span class="dec-pill mono">Calls: ${escapeHtml(reqStr)}</span>` : ""}
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <button type="button" class="btn btn-sm" onclick="refreshDecisionSidecarStatus()">Refresh</button>
      ${loaded ? `<button type="button" class="btn btn-sm btn-danger" onclick="unloadDecisionSidecarFromWeb()">Unload Sidecar</button>` : ""}
    </div>`;
}

async function refreshDecisionSidecarStatus() {
  try {
    const res = await fetch(`${API_BASE}/v1/decisions/status`);
    if (res.ok) {
      decisionWorkbenchState.sidecarStatus = await res.json();
    }
  } catch (_) {}
  updateDecisionSidecarPanel(publishedModel);
}

async function unloadDecisionSidecarFromWeb() {
  try {
    const res = await fetch(`${API_BASE}/v1/decisions/model/unload`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    decisionWorkbenchState.sidecarStatus = await res.json();
    showToast("Decision sidecar unloaded.");
    updateDecisionSidecarPanel(publishedModel);
    if (typeof fetchDashboard === "function") fetchDashboard();
  } catch (err) {
    showToast("Unload failed: " + err.message, true);
  }
}

function addDecisionQuestionRow() {
  decisionWorkbenchState.questions.push({
    name: `q${decisionWorkbenchState.questions.length}`,
    kind: "noul",
    instructions: "Evaluate property",
    options: "",
    minVal: "0",
    maxVal: "100"
  });
  renderDecisionWorkbench();
}

function removeDecisionQuestionRow(idx) {
  if (decisionWorkbenchState.questions.length <= 1) return;
  decisionWorkbenchState.questions.splice(idx, 1);
  renderDecisionWorkbench();
}

function handleDecisionQuestionField(idx, field, val) {
  if (!decisionWorkbenchState.questions[idx]) return;
  decisionWorkbenchState.questions[idx][field] = val;
  if (field === "kind") renderDecisionWorkbench();
  else refreshDecisionSnippetsDOM();
}

function handleDecisionImagesChange(ev) {
  const files = Array.from((ev && ev.target && ev.target.files) || []).slice(0, 4);
  if (!files.length) return;
  Promise.all(files.map((file) => new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => resolve("");
    reader.readAsDataURL(file);
  }))).then((uris) => {
    decisionWorkbenchState.images = uris.filter(Boolean);
    renderDecisionWorkbench();
  });
}

function clearDecisionImages() {
  decisionWorkbenchState.images = [];
  renderDecisionWorkbench();
}

function refreshDecisionSnippetsDOM() {
  const host = document.getElementById("dec-snippets-host");
  if (!host) return;
  const payload = buildDecisionRequestPayload(decisionWorkbenchState);
  const sn = buildDecisionSnippets(payload, API_BASE);
  host.innerHTML = `
    <div class="panel-sub mono">POST /v1/decisions &amp; /v1/systemone</div>
    <pre class="dec-snippet-pre">${escapeHtml(sn.curlDecisions)}</pre>
    <div class="panel-sub mono" style="margin-top:8px;">OpenAI/System-1 Alias &amp; Hermes MCP</div>
    <pre class="dec-snippet-pre">${escapeHtml(sn.curlSystemOne)}\n\n# Register MCP server:\n${escapeHtml(sn.hermesMcp)}</pre>`;
}

async function runDecisionWorkbench(mode) {
  const endpoint = mode === "laya" ? "/v1/laya" : "/v1/decisions";
  decisionWorkbenchState.lastEndpoint = endpoint;
  const resHost = document.getElementById("dec-results-host");
  if (resHost) resHost.innerHTML = `<div class="panel-sub mono">Evaluating ${escapeHtml(endpoint)}…</div>`;
  try {
    let bodyObj;
    if (mode === "laya") {
      const q0 = decisionWorkbenchState.questions[0] || { kind: "noul", instructions: "", options: "" };
      const opts = String(q0.options || "").split(",").map((x) => x.trim()).filter(Boolean);
      bodyObj = {
        state: decisionWorkbenchState.stateText,
        question: q0.instructions,
        kind: q0.kind,
        chain: decisionWorkbenchState.chain
      };
      if (opts.length) bodyObj.options = opts;
      if (decisionWorkbenchState.images.length) bodyObj.images = decisionWorkbenchState.images;
    } else {
      bodyObj = buildDecisionRequestPayload(decisionWorkbenchState);
    }
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyObj)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = (data.error && data.error.message) || data.message || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    decisionWorkbenchState.lastResponse = data;
    if (resHost) {
      resHost.innerHTML = renderDecisionResultsHTML(data, decisionWorkbenchState.stateText, decisionWorkbenchState.images);
    }
    refreshDecisionSidecarStatus();
  } catch (err) {
    if (resHost) resHost.innerHTML = `<div class="dec-pill is-failopen mono">${escapeHtml(err.message)}</div>`;
    showToast("Decision error: " + err.message, true);
  }
}

function renderDecisionWorkbench() {
  const host = document.getElementById("decisions-workbench-host");
  if (!host) return;
  const st = decisionWorkbenchState;
  const qCards = st.questions.map((q, idx) => {
    const showOpts = ["choice", "multiple", "score"].includes(q.kind);
    const showEst = q.kind === "estimate";
    return `<div class="dec-q-card">
      <div class="dec-q-row">
        <input class="input mono select-sm" value="${escapeHtml(q.name)}" placeholder="name" oninput="handleDecisionQuestionField(${idx}, 'name', this.value)" />
        <select class="select select-sm" onchange="handleDecisionQuestionField(${idx}, 'kind', this.value)">
          ${["noul", "choice", "multiple", "score", "read", "estimate", "box", "boxes"].map((k) =>
            `<option value="${k}" ${q.kind === k ? "selected" : ""}>${k}</option>`
          ).join("")}
        </select>
        ${st.questions.length > 1 ? `<button type="button" class="btn btn-sm" onclick="removeDecisionQuestionRow(${idx})">Remove</button>` : ""}
      </div>
      <input class="input" value="${escapeHtml(q.instructions)}" placeholder="Decision question instructions…" oninput="handleDecisionQuestionField(${idx}, 'instructions', this.value)" />
      ${showOpts ? `<input class="input mono" value="${escapeHtml(q.options)}" placeholder="Comma-separated options/labels…" oninput="handleDecisionQuestionField(${idx}, 'options', this.value)" />` : ""}
      ${showEst ? `<div class="dec-q-row">
        <input class="input mono select-sm" type="number" value="${escapeHtml(q.minVal)}" placeholder="min" oninput="handleDecisionQuestionField(${idx}, 'minVal', this.value)" />
        <input class="input mono select-sm" type="number" value="${escapeHtml(q.maxVal)}" placeholder="max" oninput="handleDecisionQuestionField(${idx}, 'maxVal', this.value)" />
      </div>` : ""}
    </div>`;
  }).join("");

  const thumbs = st.images.map((uri) => `<img class="dec-thumb" src="${escapeHtml(uri)}" alt="Attached image" />`).join("");
  host.innerHTML = `<div class="dec-shell">
    <article class="card-panel"><div id="dec-sidecar-banner" class="dec-status-strip"></div></article>
    <div class="dec-grid">
      <article class="card-panel dec-col">
        <div class="panel-head"><h2 class="panel-title">State, Images &amp; Questions</h2>
          <button type="button" class="btn btn-sm" onclick="addDecisionQuestionRow()">+ Question</button>
        </div>
        <div class="dec-q-row">
          <select class="select select-sm" onchange="decisionWorkbenchState.stateMode=this.value;refreshDecisionSnippetsDOM()">
            <option value="text" ${st.stateMode === "text" ? "selected" : ""}>State: Text</option>
            <option value="json" ${st.stateMode === "json" ? "selected" : ""}>State: JSON</option>
          </select>
          <input class="input mono select-sm" value="${escapeHtml(st.chain)}" placeholder="chain (default)" oninput="decisionWorkbenchState.chain=this.value;refreshDecisionSnippetsDOM()" />
          <select class="select select-sm" onchange="decisionWorkbenchState.truncation=this.value;refreshDecisionSnippetsDOM()">
            ${["auto_state", "error", "head", "tail", "middle"].map((p) => `<option value="${p}" ${st.truncation === p ? "selected" : ""}>${p}</option>`).join("")}
          </select>
        </div>
        <textarea class="input mono" rows="4" oninput="decisionWorkbenchState.stateText=this.value;refreshDecisionSnippetsDOM()">${escapeHtml(st.stateText)}</textarea>
        <div class="dec-q-row">
          <input id="dec-image-input" type="file" accept="image/*" multiple hidden onchange="handleDecisionImagesChange(event)" />
          <button type="button" class="btn btn-sm" onclick="document.getElementById('dec-image-input')?.click()">Attach Image</button>
          ${st.images.length ? `<button type="button" class="btn btn-sm" onclick="clearDecisionImages()">Clear Images (${st.images.length})</button>` : `<span class="panel-sub">Optional vision input (LFM3-VL / SigLIP2 / Qwen-VL)</span>`}
        </div>
        ${thumbs ? `<div class="dec-img-thumbs">${thumbs}</div>` : ""}
        ${qCards}
        <div class="dec-q-row">
          <button type="button" class="btn btn-primary btn-sm" onclick="runDecisionWorkbench('decisions')">Evaluate (/v1/decisions)</button>
          <button type="button" class="btn btn-sm" onclick="runDecisionWorkbench('laya')">Laya Ask (/v1/laya)</button>
        </div>
      </article>
      <div class="dec-col">
        <article class="card-panel">
          <div class="panel-head"><h2 class="panel-title">Calibrated Output</h2></div>
          <div id="dec-results-host">${renderDecisionResultsHTML(st.lastResponse, st.stateText, st.images)}</div>
        </article>
        <article class="card-panel">
          <div class="panel-head"><h2 class="panel-title">API &amp; Hermes MCP Snippets</h2></div>
          <div id="dec-snippets-host"></div>
        </article>
      </div>
    </div>
  </div>`;
  updateDecisionSidecarPanel(publishedModel);
  refreshDecisionSnippetsDOM();
}

function onDecisionsTabActivated() {
  if (!decisionWorkbenchState.initialized) {
    decisionWorkbenchState.initialized = true;
    renderDecisionWorkbench();
  }
  refreshDecisionSidecarStatus();
}
