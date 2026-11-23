let webUpdateState = null;
let webUpdatePollTimer = null;

function isSafeExternalReleaseUrl(rawUrl) {
  const safe = typeof sanitizeMarkdownUrl === "function" ? sanitizeMarkdownUrl(rawUrl) : null;
  if (!safe) return false;
  try {
    const parsed = new URL(safe);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

function formatUpdateBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "—";
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function formatUpdateRelativeTime(iso) {
  if (!iso) return "Never";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "Unknown";
  const deltaSec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (deltaSec < 60) return "Just now";
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}m ago`;
  if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)}h ago`;
  return `${Math.floor(deltaSec / 86400)}d ago`;
}

function syncUpdatePollTimer(snapshot) {
  const active = snapshot && (snapshot.phase === "downloading" || snapshot.phase === "verifying" || snapshot.phase === "checking");
  if (active && !webUpdatePollTimer) {
    webUpdatePollTimer = setInterval(() => { fetchUpdateStatus(); }, 1000);
  } else if (!active && webUpdatePollTimer) {
    clearInterval(webUpdatePollTimer);
    webUpdatePollTimer = null;
  }
}

function applyWebUpdateState(payload) {
  if (!payload || !payload.snapshot) return;
  webUpdateState = payload;
  syncUpdatePollTimer(payload.snapshot);
  renderHeaderUpdatePill(payload.snapshot);
  renderWebUpdateSettingsCard();
  const modal = document.getElementById("modal-update");
  if (modal && modal.open) {
    renderUpdateModalContent();
  }
}

function renderHeaderUpdatePill(snapshot) {
  const pill = document.getElementById("header-update-pill");
  if (!pill) return;
  if (!snapshot) {
    pill.hidden = true;
    return;
  }
  const activeOp = snapshot.phase === "downloading" || snapshot.phase === "verifying" || snapshot.phase === "readyToInstall";
  const show = (snapshot.updateAvailable && !snapshot.isSkipped) || activeOp;
  if (!show || !snapshot.latestRelease) {
    pill.hidden = true;
    return;
  }
  const ver = snapshot.latestRelease.tagName || `v${snapshot.latestRelease.version}`;
  if (snapshot.phase === "downloading" && snapshot.totalBytes && snapshot.totalBytes > 0) {
    const pct = Math.min(100, Math.max(0, Math.round(((snapshot.downloadedBytes || 0) / snapshot.totalBytes) * 100)));
    pill.textContent = `↓ ${pct}%`;
  } else if (snapshot.phase === "verifying") {
    pill.textContent = "Verifying…";
  } else if (snapshot.phase === "readyToInstall") {
    pill.textContent = `✓ ${ver}`;
  } else {
    pill.textContent = `↑ ${ver}`;
  }
  pill.title = `Qwegn ${ver} available (installed v${snapshot.currentVersion}) — click for release notes`;
  pill.hidden = false;
}

function renderWebUpdateSettingsCard() {
  const host = document.getElementById("settings-updates-host");
  if (!host) return;
  if (!webUpdateState || !webUpdateState.snapshot) {
    host.innerHTML = `<div class="panel-sub">Update status unavailable.</div>
      <div class="update-settings-actions">
        <button type="button" class="btn btn-sm" onclick="checkForUpdatesFromWeb(true)">Check for Updates</button>
      </div>`;
    return;
  }
  const s = webUpdateState.snapshot;
  const rel = s.latestRelease;
  const checkedText = formatUpdateRelativeTime(s.lastCheckedAt);
  let statusLine = `Installed v${escapeHtml(s.currentVersion)} · Last checked: ${escapeHtml(checkedText)}`;
  if (s.phase === "checking") {
    statusLine = "Checking for updates…";
  } else if (s.updateAvailable && rel) {
    statusLine = `v${escapeHtml(rel.version)} available${s.isSkipped ? " (skipped)" : ""} · Last checked: ${escapeHtml(checkedText)}`;
  } else if (s.phase === "upToDate") {
    statusLine = `Up to date (v${escapeHtml(s.currentVersion)}) · Checked: ${escapeHtml(checkedText)}`;
  }
  const skipBtn = (s.isSkipped && rel)
    ? `<button type="button" class="btn btn-sm" onclick="setSkippedUpdateVersion(null)">Unskip ${escapeHtml(rel.tagName)}</button>`
    : "";
  host.innerHTML = `
    <div class="panel-sub mono">${statusLine}</div>
    <div class="update-settings-actions">
      <button type="button" class="btn btn-sm" onclick="checkForUpdatesFromWeb(true)">Check Now</button>
      <button type="button" class="btn btn-sm" onclick="openUpdateModal()">${s.updateAvailable ? "View Update…" : "Release Details…"}</button>
      ${skipBtn}
    </div>`;
}

function updateBadgeMetaForModal(s) {
  switch (s.phase) {
    case "readyToInstall": return ["Verified", "is-ready"];
    case "available": return ["Update Available", "is-stale"];
    case "downloading": return ["Downloading", "is-stale"];
    case "verifying": return ["Verifying SHA-256", "is-stale"];
    case "upToDate": return ["Up to Date", "is-ready"];
    case "error": return ["Error", "is-wrong"];
    default: return ["Checking…", "is-missing"];
  }
}

function renderUpdateModalContent() {
  const payload = webUpdateState;
  const badge = document.getElementById("update-status-badge");
  const headline = document.getElementById("update-version-headline");
  const summary = document.getElementById("update-status-summary");
  const metaRow = document.getElementById("update-meta-row");
  const shaBox = document.getElementById("update-sha-badge");
  const busyBox = document.getElementById("update-busy-warn");
  const notesSec = document.getElementById("update-notes-section");
  const notesBody = document.getElementById("update-notes-body");
  const progBox = document.getElementById("update-progress-box");
  const progPhase = document.getElementById("update-progress-phase");
  const progPct = document.getElementById("update-progress-pct");
  const progFill = document.getElementById("update-progress-fill");
  const progBytes = document.getElementById("update-progress-bytes");
  const actionsBar = document.getElementById("update-actions-bar");
  if (!payload || !payload.snapshot) return;

  const s = payload.snapshot;
  const rel = s.latestRelease;
  const [badgeLabel, badgeCls] = updateBadgeMetaForModal(s);
  if (badge) {
    badge.textContent = badgeLabel;
    badge.className = `mempack-badge ${badgeCls}`;
  }

  if (s.phase === "checking") {
    if (headline) headline.textContent = "Checking for Qwegn updates…";
    if (summary) summary.textContent = `Current installed version: v${s.currentVersion}.`;
  } else if (s.updateAvailable && rel) {
    if (headline) headline.textContent = `Qwegn ${rel.version} is available!`;
    const pkgSize = rel.pkgAsset && rel.pkgAsset.sizeBytes ? ` · ${formatUpdateBytes(rel.pkgAsset.sizeBytes)} .pkg` : "";
    if (summary) summary.textContent = `Installed: v${s.currentVersion} → Latest: v${rel.version}${pkgSize}`;
  } else if (s.phase === "error") {
    if (headline) headline.textContent = "Update check failed";
    if (summary) summary.textContent = s.errorMessage || "Unable to reach release feed.";
  } else {
    if (headline) headline.textContent = `Qwegn v${s.currentVersion} is up to date`;
    if (summary) summary.textContent = `Last checked: ${formatUpdateRelativeTime(s.lastCheckedAt)}.`;
  }

  if (metaRow) {
    const originLabel = s.installOrigin === "pkgReceipt"
      ? "macOS Installer (.pkg)"
      : s.installOrigin === "sourceCheckout" ? "Source checkout" : "Portable bundle";
    metaRow.textContent = `Install: ${originLabel} · Schedule: ${s.schedule}`;
    metaRow.hidden = false;
  }

  if (shaBox) {
    if (s.verifiedSHA256 && s.phase === "readyToInstall") {
      const sig = s.signatureSummary ? ` · ${s.signatureSummary}` : "";
      shaBox.textContent = `✓ SHA-256 verified: ${s.verifiedSHA256}${sig}`;
      shaBox.hidden = false;
    } else {
      shaBox.hidden = true;
    }
  }

  if (busyBox) {
    if (payload.inferenceActive && s.updateAvailable) {
      busyBox.textContent = "Active inference or model load in progress — finish current generation before launching Installer.app.";
      busyBox.hidden = false;
    } else if (!payload.loopbackDesktopActionsAllowed && s.updateAvailable) {
      busyBox.textContent = "Remote browser session: desktop .pkg download and Installer launch are restricted to loopback host.";
      busyBox.hidden = false;
    } else {
      busyBox.hidden = true;
    }
  }

  if (notesSec && notesBody) {
    if (rel && rel.notes) {
      notesBody.innerHTML = typeof renderSmokeProbeMarkdown === "function"
        ? renderSmokeProbeMarkdown(rel.notes)
        : `<pre class="mono">${escapeHtml(rel.notes)}</pre>`;
      notesSec.hidden = false;
    } else {
      notesSec.hidden = true;
    }
  }

  if (progBox) {
    if (s.phase === "downloading" || s.phase === "verifying") {
      const total = s.totalBytes || (rel && rel.pkgAsset ? rel.pkgAsset.sizeBytes : null) || 0;
      const done = s.downloadedBytes || 0;
      const pct = total > 0 ? Math.min(100, Math.max(0, Math.round((done / total) * 100))) : 0;
      if (progPhase) progPhase.textContent = s.phase === "verifying" ? "Verifying CryptoKit SHA-256…" : "Downloading verified .pkg…";
      if (progPct) progPct.textContent = `${pct}%`;
      if (progFill) progFill.style.width = `${pct}%`;
      if (progBytes) progBytes.textContent = total > 0 ? `${formatUpdateBytes(done)} / ${formatUpdateBytes(total)}` : formatUpdateBytes(done);
      progBox.hidden = false;
    } else {
      progBox.hidden = true;
    }
  }

  if (actionsBar) {
    const btns = [];
    const safeLink = rel && isSafeExternalReleaseUrl(rel.htmlURL) ? rel.htmlURL : null;
    if (s.phase === "downloading" || s.phase === "verifying") {
      btns.push(`<button type="button" class="btn btn-sm" onclick="cancelUpdateDownloadFromWeb()">Cancel Download</button>`);
    } else if (s.updateAvailable && rel) {
      if (s.isSkipped) {
        btns.push(`<button type="button" class="btn btn-sm" onclick="setSkippedUpdateVersion(null)">Unskip ${escapeHtml(rel.tagName)}</button>`);
      } else {
        btns.push(`<button type="button" class="btn btn-sm" onclick="setSkippedUpdateVersion('${escapeHtml(rel.version)}')">Skip Version</button>`);
      }
      if (safeLink) {
        btns.push(`<a class="btn btn-sm" href="${escapeHtml(safeLink)}" target="_blank" rel="noopener noreferrer">Release Page ↗</a>`);
      }
      if (payload.loopbackDesktopActionsAllowed) {
        if (s.phase === "readyToInstall") {
          const forceArg = payload.inferenceActive ? "true" : "false";
          const label = payload.inferenceActive ? "Force Open Installer…" : "Open Installer…";
          btns.push(`<button type="button" class="btn btn-primary btn-sm" onclick="installUpdateFromWeb(${forceArg})">${label}</button>`);
        } else {
          btns.push(`<button type="button" class="btn btn-primary btn-sm" onclick="downloadUpdateFromWeb()">${s.canInstallPkg ? "Download & Verify .pkg" : "Download .pkg…"}</button>`);
        }
      }
    } else {
      btns.push(`<button type="button" class="btn btn-primary btn-sm" onclick="checkForUpdatesFromWeb(true)">Check Again</button>`);
    }
    actionsBar.innerHTML = btns.join("");
  }
}

async function fetchUpdateStatus() {
  try {
    const res = await fetch(`${API_BASE}/v1/update`);
    if (!res.ok) return;
    const data = await res.json();
    applyWebUpdateState(data);
  } catch { /* Ignore background status poll errors when disconnected */ }
}

async function openUpdateModal() {
  const modal = document.getElementById("modal-update");
  if (!modal) return;
  if (!webUpdateState) await fetchUpdateStatus();
  renderUpdateModalContent();
  if (!modal.open && typeof modal.showModal === "function") modal.showModal();
}

function closeUpdateModal() {
  const modal = document.getElementById("modal-update");
  if (modal && modal.open && typeof modal.close === "function") modal.close();
}

async function checkForUpdatesFromWeb(openModalAfter) {
  try {
    const res = await fetch(`${API_BASE}/v1/update/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    applyWebUpdateState(data);
    if (openModalAfter) openUpdateModal();
  } catch (err) {
    showToast("Update check failed: " + err.message, true);
  }
}

async function setSkippedUpdateVersion(version) {
  try {
    const res = await fetch(`${API_BASE}/v1/update/skip`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    applyWebUpdateState(data);
  } catch (err) {
    showToast("Failed to update skip state: " + err.message, true);
  }
}

async function downloadUpdateFromWeb() {
  try {
    const res = await fetch(`${API_BASE}/v1/update/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    applyWebUpdateState(data);
  } catch (err) {
    showToast("Update download failed: " + err.message, true);
  }
}

async function cancelUpdateDownloadFromWeb() {
  try {
    const res = await fetch(`${API_BASE}/v1/update/download/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    applyWebUpdateState(data);
  } catch (err) {
    showToast("Cancel download failed: " + err.message, true);
  }
}

async function installUpdateFromWeb(force) {
  try {
    const res = await fetch(`${API_BASE}/v1/update/install`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: Boolean(force) })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    applyWebUpdateState(data);
    showToast("Opened macOS Installer.app with SHA-256 verified .pkg.");
  } catch (err) {
    showToast("Installer launch blocked: " + err.message, true);
  }
}

function initWebUpdates() {
  fetchUpdateStatus();
}
