const API_BASE = (typeof window !== "undefined" && window.location && window.location.origin) ? window.location.origin : "http://127.0.0.1:8080";
let isConnected = false;
let publishedModel = null;
let modelGeneration = 0;
let discoveredModelsList = [];
let metricsEventSource = null;
let logsEventSource = null;
let expandedRequestId = null;
let logBuffer = [];
let activeLogFilter = "ALL";
let logWrap = true;
let logFollow = true;
let loadedSettingsObj = null;
let wasPrefilling = false;
let prefillHoldTimer = null;
let lastPrefillTargetValue = null;
let lastPrefillTargetText = "";
let smokeProbeState = {
  status: "empty",
  content: "",
  reasoning: "",
  rawText: "",
  isJsonFallback: false,
  mode: "rendered"
};

function formatTokenCount(raw) {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return "—";
  const n = Math.round(raw);
  const sign = n < 0 ? "-" : "";
  const absVal = Math.abs(n);
  if (absVal >= 99995000) return `${sign}${(absVal / 1000000).toFixed(1)}M`;
  if (absVal >= 1000000) return `${sign}${(absVal / 1000000).toFixed(2)}M`;
  return `${sign}${absVal.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`;
}

function showToast(message, isError = false) {
  const host = document.getElementById("toast-region");
  if (!host) return;
  const node = document.createElement("div");
  node.className = `toast ${isError ? "toast-error" : ""}`;
  node.textContent = message;
  host.appendChild(node);
  setTimeout(() => { node.remove(); }, 3800);
}

function escapeHtml(raw) {
  return String(raw ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function switchTab(tabId, updateUrl = true) {
  const validTabs = ["monitor", "models", "logs", "settings"];
  const resolved = validTabs.includes(tabId) ? tabId : "monitor";
  document.querySelectorAll(".tab-pane").forEach((el) => el.classList.add("hidden"));
  const target = document.getElementById(`view-${resolved}`);
  if (target) target.classList.remove("hidden");

  validTabs.forEach((t) => {
    const btn = document.getElementById(`nav-tab-${t}`);
    if (btn) btn.classList.toggle("is-active", t === resolved);
  });

  if (updateUrl && window.history && typeof window.history.replaceState === "function") {
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("tab", resolved);
    window.history.replaceState(null, "", nextUrl.toString());
  }

  if (resolved === "models") loadDiscoveredModels();
  if (resolved === "logs") initLogStream();
  else closeLogStream();
  if (resolved === "settings") loadLiveSettings();
}

function updateSparkline(firstSeries, secondSeries) {
  const first = Array.isArray(firstSeries) ? firstSeries : [];
  const second = Array.isArray(secondSeries) ? secondSeries : [];
  const totalSlots = Math.max(first.length, second.length);
  const validPositive = (arr) => arr.filter((v) => v !== null && v !== undefined && v > 0);
  const posA = validPositive(first);
  const posB = validPositive(second);
  const peakA = posA.length ? Math.max(...posA) : 0;
  const peakB = posB.length ? Math.max(...posB) : 0;

  let ceilA = 0.80;
  let ceilB = 0.80;
  if (peakA > 0 && peakB > 0) {
    if (peakA > peakB * 1.5) {
      ceilA = 0.84;
      ceilB = 0.60;
    } else if (peakB > peakA * 1.5) {
      ceilA = 0.60;
      ceilB = 0.84;
    } else {
      ceilA = 0.84;
      ceilB = 0.70;
    }
  }

  const makePathGeo = (history, companion, peak, companionPeak, ceiling) => {
    const runs = [];
    let current = [];
    const maxRef = Math.max(1, peak > 0 ? peak / ceiling : 1);
    history.forEach((val, idx) => {
      if (val === null || val === undefined) {
        const comp = companion[idx];
        if (comp !== null && comp !== undefined && Number(comp) > 0) return;
        if (current.length > 0) runs.push(current);
        current = [];
        return;
      }
      const clamped = Math.max(0, Number(val));
      const x = totalSlots <= 1 ? 588 : 12 + (idx / (totalSlots - 1)) * 576;
      const norm = peak > 0 ? Math.min(1, Math.max(0, clamped / maxRef)) : 0;
      const y = 94 - norm * 84;
      current.push({ x, y, raw: clamped, idx });
    });
    if (current.length > 0) runs.push(current);

    const lineRuns = [];
    const areaRuns = [];
    const markerRuns = [];
    let guideD = "";
    let lastPoint = null;

    runs.forEach((pts) => {
      if (pts.length === 0) return;
      lastPoint = pts[pts.length - 1];
      if (pts.length >= 2) {
        let d = `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
        for (let i = 1; i < pts.length; i++) {
          const prev = pts[i - 1];
          const curr = pts[i];
          const cpX = (prev.x + curr.x) / 2;
          d += ` C ${cpX.toFixed(1)},${prev.y.toFixed(1)} ${cpX.toFixed(1)},${curr.y.toFixed(1)} ${curr.x.toFixed(1)},${curr.y.toFixed(1)}`;
        }
        lineRuns.push(d);
        const firstPt = pts[0];
        const lastPt = pts[pts.length - 1];
        areaRuns.push(`${d} L ${lastPt.x.toFixed(1)},96 L ${firstPt.x.toFixed(1)},96 Z`);
      } else {
        const pt = pts[0];
        markerRuns.push(`M ${(pt.x - 0.1).toFixed(1)},${pt.y.toFixed(1)} L ${(pt.x + 0.1).toFixed(1)},${pt.y.toFixed(1)}`);
      }
    });

    if (runs.length === 1 && runs[0].length === 1 && (totalSlots === 1 || companionPeak > 0)) {
      const pt = runs[0][0];
      const yStr = pt.y.toFixed(1);
      guideD = `M 12,${yStr} L 588,${yStr}`;
      areaRuns.push(`M 12,${yStr} L 588,${yStr} L 588,96 L 12,96 Z`);
    } else if (lastPoint) {
      const marker = `M ${(lastPoint.x - 0.1).toFixed(1)},${lastPoint.y.toFixed(1)} L ${(lastPoint.x + 0.1).toFixed(1)},${lastPoint.y.toFixed(1)}`;
      if (!markerRuns.includes(marker)) markerRuns.push(marker);
    }

    return {
      d: lineRuns.join(" "),
      areaD: areaRuns.join(" "),
      guideD,
      markersD: markerRuns.join(" "),
      lastPt: lastPoint
    };
  };

  const drawSeries = (geometry, ids) => {
    const path = document.getElementById(ids.path);
    const area = document.getElementById(ids.area);
    const guide = document.getElementById(ids.guide);
    const markers = document.getElementById(ids.markers);
    const dot = document.getElementById(ids.dot);
    if (path) path.setAttribute("d", geometry.d);
    if (area) area.setAttribute("d", geometry.areaD);
    if (guide) guide.setAttribute("d", geometry.guideD);
    if (markers) markers.setAttribute("d", geometry.markersD);
    if (dot) {
      if (geometry.lastPt) {
        dot.setAttribute("cx", geometry.lastPt.x.toFixed(1));
        dot.setAttribute("cy", geometry.lastPt.y.toFixed(1));
        dot.classList.remove("opacity-0");
      } else {
        dot.classList.add("opacity-0");
      }
    }
  };

  drawSeries(makePathGeo(first, second, peakA, peakB, ceilA), {
    path: "sparkline-series-a-path",
    area: "sparkline-series-a-area",
    guide: "sparkline-series-a-guide",
    markers: "sparkline-series-a-markers",
    dot: "sparkline-series-a-dot"
  });
  drawSeries(makePathGeo(second, first, peakB, peakA, ceilB), {
    path: "sparkline-series-b-path",
    area: "sparkline-series-b-area",
    guide: "sparkline-series-b-guide",
    markers: "sparkline-series-b-markers",
    dot: "sparkline-series-b-dot"
  });
}

function wireSparklineHover() {
  const svg = document.getElementById("sparkline-svg");
  const tip = document.getElementById("sparkline-tooltip");
  if (!svg || !tip) return;
  svg.addEventListener("mousemove", (evt) => {
    if (!publishedModel) return;
    const slots = groupCards(publishedModel, "throughput").slice(0, 2);
    const s0 = (slots[0] && Array.isArray(slots[0].series)) ? slots[0].series : [];
    const s1 = (slots[1] && Array.isArray(slots[1].series)) ? slots[1].series : [];
    const len = Math.max(s0.length, s1.length);
    if (len === 0) return;
    const rect = svg.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (evt.clientX - rect.left) / Math.max(1, rect.width)));
    const idx = Math.min(len - 1, Math.round(frac * Math.max(0, len - 1)));
    const v0 = s0[idx];
    const v1 = s1[idx];
    const fmt = (val, unit) => (val === null || val === undefined) ? "—" : `${Number(val).toFixed(1)}${unit ? ` ${unit}` : ""}`;
    const label0 = slots[0] ? slots[0].title : "A";
    const label1 = slots[1] ? slots[1].title : "B";
    tip.textContent = `#${idx + 1} · ${label0}: ${fmt(v0, slots[0]?.unit)} · ${label1}: ${fmt(v1, slots[1]?.unit)}`;
    tip.classList.remove("hidden");
  });
  svg.addEventListener("mouseleave", () => {
    tip.classList.add("hidden");
  });
}

function middleTruncate(str, maxLen) {
  const s = String(str || "");
  if (s.length <= maxLen || maxLen < 12) return s;
  const head = Math.ceil((maxLen - 1) * 0.45);
  const tail = Math.floor((maxLen - 1) * 0.55);
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}
