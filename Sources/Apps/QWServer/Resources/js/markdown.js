function sanitizeMarkdownUrl(rawUrl) {
  const trimmed = String(rawUrl ?? "").trim();
  if (!trimmed) return null;
  for (let i = 0; i < trimmed.length; i += 1) {
    const code = trimmed.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return null;
  }
  if (/^https?:\/\//i.test(trimmed) || /^mailto:/i.test(trimmed)) return trimmed;
  if (trimmed.startsWith("/") && !trimmed.startsWith("//") && !trimmed.startsWith("/\\")) return trimmed;
  if (trimmed.startsWith("#")) return trimmed;
  return null;
}

function splitMarkdownTableCells(line) {
  let s = String(line ?? "").trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) {
    let bs = 0;
    for (let k = s.length - 2; k >= 0 && s[k] === "\\"; k--) bs += 1;
    if (bs % 2 === 0) s = s.slice(0, -1);
  }
  const cells = [];
  let cur = "";
  let fence = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "\\" && fence === 0 && i + 1 < s.length && s[i + 1] === "|") {
      cur += "|";
      i += 1;
      continue;
    }
    if (ch === "`") {
      let run = 1;
      while (i + run < s.length && s[i + run] === "`") run += 1;
      if (fence === 0) fence = run;
      else if (fence === run) fence = 0;
      cur += s.slice(i, i + run);
      i += run - 1;
      continue;
    }
    if (ch === "|" && fence === 0) {
      cells.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }
  cells.push(cur.trim());
  return cells;
}

function isMarkdownTableDelimiter(line) {
  const s = String(line ?? "");
  if (!s.includes("|")) return null;
  const cells = splitMarkdownTableCells(s);
  if (cells.length === 0) return null;
  const aligns = [];
  for (const cell of cells) {
    if (!/^:?-{2,}:?$/.test(cell)) return null;
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    aligns.push(left && right ? "center" : right ? "right" : left ? "left" : "");
  }
  return aligns;
}

function applyInlineMarkdownEmphasis(escaped) {
  return escaped
    .replace(/\*\*\*(?=\S)([^\n]*?\S)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(?=\S)([^\n]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![A-Za-z0-9_])__(?=\S)([^\n]*?\S)__(?![A-Za-z0-9_])/g, "<strong>$1</strong>")
    .replace(/~~(?=\S)([^\n]*?\S)~~/g, "<del>$1</del>")
    .replace(/(?<!\*)\*(?!\*)(?=\S)([^\n*]*?\S)\*(?!\*)/g, "<em>$1</em>")
    .replace(/(?<![A-Za-z0-9_])_(?!_)(?=\S)([^\n_]*?\S)_(?!_)(?![A-Za-z0-9_])/g, "<em>$1</em>");
}

function renderInlineMarkdown(raw) {
  const clean = String(raw ?? "").replace(/\u0000/g, "");
  const codeSlots = [];
  const linkSlots = [];
  const withCode = clean.replace(/(`+)([^\n]*?)\1/g, (_, _ticks, inner) => {
    let content = inner;
    if (content.length >= 2 && content.startsWith(" ") && content.endsWith(" ") && /[^ ]/.test(content)) {
      content = content.slice(1, -1);
    }
    const idx = codeSlots.length;
    codeSlots.push(`<code class="md-inline-code">${escapeHtml(content)}</code>`);
    return `\u0000C${idx}\u0000`;
  });
  const withLinks = withCode.replace(/\[([^\]]+)\]\(([^()\s]*(?:\([^()\s]*\)[^()\s]*)*)\)/g, (_, label, href) => {
    const formattedLabel = applyInlineMarkdownEmphasis(escapeHtml(label));
    const safeHref = sanitizeMarkdownUrl(href);
    if (!safeHref) return formattedLabel;
    const idx = linkSlots.length;
    linkSlots.push(`<a href="${escapeHtml(safeHref)}" target="_blank" rel="noopener noreferrer">${formattedLabel}</a>`);
    return `\u0000L${idx}\u0000`;
  });
  let out = applyInlineMarkdownEmphasis(escapeHtml(withLinks));
  out = out.replace(/\u0000L(\d+)\u0000/g, (_, idx) => linkSlots[Number(idx)] ?? "");
  out = out.replace(/\u0000C(\d+)\u0000/g, (_, idx) => codeSlots[Number(idx)] ?? "");
  return out;
}

function renderSmokeProbeMarkdown(rawMarkdown) {
  const normalized = String(rawMarkdown ?? "").replace(/\r\n?/g, "\n");
  const lines = normalized.split("\n");
  const blocks = [];
  const isHeading = (line) => /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.test(line);
  const isFence = (line) => /^\s{0,3}(`{3,}|~{3,})\s*([^\s`~]*).*$/.test(line);
  const isHr = (line) => /^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$/.test(line);
  const isQuote = (line) => /^\s{0,3}>\s?(.*)$/.test(line);
  const matchList = (line) => /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line);
  const isBlockStart = (idx) => {
    const line = lines[idx];
    return isFence(line) || isHeading(line) || isHr(line) || isQuote(line) || Boolean(matchList(line))
      || (line.includes("|") && idx + 1 < lines.length && isMarkdownTableDelimiter(lines[idx + 1]) !== null);
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const fenceMatch = /^\s{0,3}(`{3,}|~{3,})\s*([^\s`~]*).*$/.exec(line);
    if (fenceMatch) {
      const marker = fenceMatch[1];
      const fenceCh = marker[0];
      const fenceLen = marker.length;
      const lang = fenceMatch[2] || "";
      const codeLines = [];
      i += 1;
      while (i < lines.length) {
        const closeMatch = new RegExp(`^\\s{0,3}${fenceCh}{${fenceLen},}\\s*$`);
        if (closeMatch.test(lines[i])) {
          i += 1;
          break;
        }
        codeLines.push(lines[i]);
        i += 1;
      }
      const langHeader = `<div class="md-code-lang mono"><span>${lang ? escapeHtml(lang) : "text"}</span><button type="button" class="md-code-copy-btn" aria-label="Copy code snippet">Copy</button></div>`;
      blocks.push(`<div class="md-code-block">${langHeader}<pre class="mono"><code>${escapeHtml(codeLines.join("\n"))}</code></pre></div>`);
      continue;
    }
    const headingMatch = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (headingMatch) {
      const level = headingMatch[1].length;
      blocks.push(`<h${level} class="md-h${level}">${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      i += 1;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length) {
      const aligns = isMarkdownTableDelimiter(lines[i + 1]);
      if (aligns !== null) {
        const headerCells = splitMarkdownTableCells(line);
        const colCount = Math.max(headerCells.length, aligns.length);
        const ths = Array.from({ length: colCount }, (_, col) => {
          const align = aligns[col] ? ` style="text-align:${aligns[col]};"` : "";
          return `<th${align}>${renderInlineMarkdown(headerCells[col] ?? "")}</th>`;
        }).join("");
        i += 2;
        const bodyRows = [];
        while (i < lines.length && lines[i].trim() && lines[i].includes("|") && !isFence(lines[i]) && !isHeading(lines[i])) {
          const rowCells = splitMarkdownTableCells(lines[i]);
          const tds = Array.from({ length: colCount }, (_, col) => {
            const align = aligns[col] ? ` style="text-align:${aligns[col]};"` : "";
            return `<td${align}>${renderInlineMarkdown(rowCells[col] ?? "")}</td>`;
          }).join("");
          bodyRows.push(`<tr>${tds}</tr>`);
          i += 1;
        }
        blocks.push(
          `<div class="table-scroll md-table-wrap"><table class="ledger md-table"><thead><tr>${ths}</tr></thead><tbody>${bodyRows.join("")}</tbody></table></div>`
        );
        continue;
      }
    }
    if (isHr(line)) {
      blocks.push("<hr class=\"md-hr\" />");
      i += 1;
      continue;
    }
    if (isQuote(line)) {
      const qLines = [];
      while (i < lines.length) {
        const qm = /^\s{0,3}>\s?(.*)$/.exec(lines[i]);
        if (!qm) break;
        qLines.push(qm[1]);
        i += 1;
      }
      blocks.push(`<blockquote class="md-quote">${renderSmokeProbeMarkdown(qLines.join("\n"))}</blockquote>`);
      continue;
    }
    if (matchList(line)) {
      const listEntries = [];
      while (i < lines.length) {
        const lm = matchList(lines[i]);
        if (lm) {
          const indent = lm[1].replace(/\t/g, "    ").length;
          const marker = lm[2];
          const isOrdered = /^\d/.test(marker);
          const startNum = isOrdered ? parseInt(marker, 10) : 1;
          let text = lm[3];
          i += 1;
          while (i < lines.length && lines[i].trim() && !isBlockStart(i)) {
            text += `\n${lines[i].trim()}`;
            i += 1;
          }
          listEntries.push({ indent, tag: isOrdered ? "ol" : "ul", startNum, text });
          continue;
        }
        if (!lines[i].trim() && i + 1 < lines.length && matchList(lines[i + 1])) {
          i += 1;
          continue;
        }
        break;
      }
      let html = "";
      const stack = [];
      const openList = (entry) => {
        const attrs = entry.tag === "ol" && entry.startNum !== 1 ? ` start="${entry.startNum}"` : "";
        html += `<${entry.tag}${attrs}>`;
        stack.push({ indent: entry.indent, tag: entry.tag, openLi: false });
      };
      const closeTopList = () => {
        const top = stack.pop();
        if (!top) return;
        if (top.openLi) html += "</li>";
        html += `</${top.tag}>`;
      };
      for (const entry of listEntries) {
        while (stack.length > 0 && entry.indent < stack[stack.length - 1].indent) {
          closeTopList();
        }
        if (stack.length === 0) {
          openList(entry);
        } else {
          const top = stack[stack.length - 1];
          if (entry.indent > top.indent) {
            if (!top.openLi) {
              html += "<li>";
              top.openLi = true;
            }
            openList(entry);
          } else if (entry.tag !== top.tag) {
            closeTopList();
            openList(entry);
          } else if (top.openLi) {
            html += "</li>";
            top.openLi = false;
          }
        }
        let itemHtml = entry.text;
        const taskMatch = /^\[( |x|X)\]\s+(.*)$/s.exec(itemHtml);
        if (taskMatch) {
          const checked = taskMatch[1].toLowerCase() === "x" ? " checked" : "";
          itemHtml = `<input type="checkbox" disabled${checked} style="margin-right:6px;vertical-align:middle;" />${renderInlineMarkdown(taskMatch[2])}`;
        } else {
          itemHtml = renderInlineMarkdown(itemHtml);
        }
        itemHtml = itemHtml.replace(/\n/g, "<br />");
        html += `<li>${itemHtml}`;
        stack[stack.length - 1].openLi = true;
      }
      while (stack.length > 0) closeTopList();
      blocks.push(html);
      continue;
    }
    const pLines = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(i)) {
      pLines.push(lines[i]);
      i += 1;
    }
    if (pLines.length > 0) {
      let joined = "";
      for (let k = 0; k < pLines.length; k++) {
        const seg = pLines[k];
        const hardBreak = / {2}$/.test(seg) || /\\$/.test(seg);
        const cleanedSeg = hardBreak ? seg.replace(/(?: {2}|\\)$/, "") : seg;
        if (k > 0) joined += "\n";
        joined += cleanedSeg + (hardBreak && k + 1 < pLines.length ? "\u0000BR\u0000" : "");
      }
      const renderedPara = renderInlineMarkdown(joined)
        .replace(/\u0000BR\u0000\n?/g, "<br />")
        .replace(/\n/g, " ");
      blocks.push(`<p>${renderedPara}</p>`);
    }
  }
  return blocks.join("");
}

function extractSmokeProbeParts(msg, body) {
  let rawContent = msg && typeof msg.content === "string" ? msg.content : "";
  let reasoning = msg && typeof msg.reasoning_content === "string" ? msg.reasoning_content.trim() : "";
  const thinkChunks = [];
  if (rawContent) {
    rawContent = rawContent.replace(/<think>([\s\S]*?)<\/think>/gi, (_, inner) => {
      const trimmed = inner.trim();
      if (trimmed) thinkChunks.push(trimmed);
      return "";
    }).trim();
  }
  if (!reasoning && thinkChunks.length > 0) {
    reasoning = thinkChunks.join("\n\n");
  }
  if (rawContent.length > 0 || reasoning.length > 0) {
    const rawCombined = reasoning && rawContent
      ? `[reasoning]\n${reasoning}\n\n[answer]\n${rawContent}`
      : (rawContent || `[reasoning]\n${reasoning}`);
    return { content: rawContent, reasoning, rawText: rawCombined, isJsonFallback: false };
  }
  return { content: "", reasoning: "", rawText: JSON.stringify(body, null, 2), isJsonFallback: true };
}

function renderSmokeProbeOutput() {
  const out = document.getElementById("probe-output");
  const modeBtn = document.getElementById("probe-mode-btn");
  const copyBtn = document.getElementById("probe-copy-btn");
  if (!out) return;
  if (smokeProbeState.status === "empty") {
    out.className = "probe-output";
    out.textContent = "";
    if (modeBtn) modeBtn.hidden = true;
    if (copyBtn) copyBtn.hidden = true;
    return;
  }
  if (smokeProbeState.status === "running" || smokeProbeState.status === "error") {
    out.className = smokeProbeState.status === "error" ? "probe-output is-raw is-error" : "probe-output is-raw";
    out.textContent = smokeProbeState.rawText;
    if (modeBtn) modeBtn.hidden = true;
    if (copyBtn) copyBtn.hidden = true;
    return;
  }
  if (modeBtn) {
    modeBtn.hidden = false;
    modeBtn.textContent = smokeProbeState.mode === "raw" ? "Rendered" : "Raw";
  }
  if (copyBtn) copyBtn.hidden = false;
  if (smokeProbeState.mode === "raw") {
    out.className = "probe-output is-raw";
    out.textContent = smokeProbeState.rawText;
    return;
  }
  out.className = "probe-output probe-markdown";
  if (smokeProbeState.isJsonFallback) {
    out.innerHTML = `<div class="md-code-block"><div class="md-code-lang mono">json</div><pre class="mono"><code>${escapeHtml(smokeProbeState.rawText)}</code></pre></div>`;
    return;
  }
  const sections = [];
  if (smokeProbeState.reasoning) {
    const openAttr = smokeProbeState.content ? "" : " open";
    sections.push(`<details class="md-thinking"${openAttr}><summary>Reasoning</summary><div class="md-thinking-body">${renderSmokeProbeMarkdown(smokeProbeState.reasoning)}</div></details>`);
  }
  if (smokeProbeState.content) {
    sections.push(renderSmokeProbeMarkdown(smokeProbeState.content));
  }
  out.innerHTML = sections.join("");
}

function toggleSmokeProbeMode() {
  smokeProbeState.mode = smokeProbeState.mode === "raw" ? "rendered" : "raw";
  renderSmokeProbeOutput();
}

function copySmokeProbeOutput() {
  const text = smokeProbeState.rawText || "";
  if (!text) return;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => showToast("Copied probe output."));
  }
}

async function runBrowserSmokeProbe() {
  const btn = document.getElementById("probe-run-btn");
  const timingEl = document.getElementById("probe-timing");
  const prompt = document.getElementById("probe-prompt")?.value || "Hello";
  const maxTokens = parseInt(document.getElementById("probe-max-tokens")?.value || "24", 10);
  const modelName = publishedModel && publishedModel.modelId ? publishedModel.modelId : "";
  if (btn) btn.disabled = true;
  smokeProbeState = {
    ...smokeProbeState,
    status: "running",
    content: "",
    reasoning: "",
    rawText: "Running probe…",
    isJsonFallback: false
  };
  renderSmokeProbeOutput();
  if (timingEl) timingEl.textContent = "Measuring browser round trip…";

  const started = performance.now();
  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelName,
        messages: [{ role: "user", content: prompt }],
        max_tokens: Number.isFinite(maxTokens) ? maxTokens : 24,
        stream: false
      })
    });
    const elapsedMs = Math.round(performance.now() - started);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(body.error?.message || body.message || `HTTP ${res.status}`);
    }
    const parts = extractSmokeProbeParts(body.choices?.[0]?.message, body);
    smokeProbeState = {
      ...smokeProbeState,
      status: "done",
      content: parts.content,
      reasoning: parts.reasoning,
      rawText: parts.rawText,
      isJsonFallback: parts.isJsonFallback
    };
    renderSmokeProbeOutput();
    if (timingEl) timingEl.textContent = `browser round trip: ${elapsedMs} ms`;
  } catch (err) {
    smokeProbeState = {
      ...smokeProbeState,
      status: "error",
      content: "",
      reasoning: "",
      rawText: `Error: ${err.message}`,
      isJsonFallback: false
    };
    renderSmokeProbeOutput();
    if (timingEl) timingEl.textContent = "browser round trip: failed";
    showToast("Smoke probe failed: " + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function copyMarkdownCodeBlock(btn) {
  if (!btn || !btn.closest) return;
  const block = btn.closest(".md-code-block");
  const code = block ? block.querySelector("pre code") : null;
  const text = code ? code.textContent || "" : "";
  if (!text) return;
  const restoreLabel = () => {
    btn.textContent = "Copy";
  };
  if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => {
      btn.textContent = "Copied";
      setTimeout(restoreLabel, 1400);
      showToast("Code snippet copied.");
    }).catch(() => {
      showToast("Clipboard unavailable.", true);
    });
  } else {
    showToast("Clipboard unavailable.", true);
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("click", (event) => {
    const btn = event.target && event.target.closest ? event.target.closest(".md-code-copy-btn") : null;
    if (btn) copyMarkdownCodeBlock(btn);
  });
}
