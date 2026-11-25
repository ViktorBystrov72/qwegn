/* Multimodal image, video, and live screen-snapshot helpers for /ui?tab=chat */

const CHAT_MAX_VIDEO_BYTES = 64 * 1024 * 1024;

function safeChatImageDataUrl(rawUrl) {
  const s = String(rawUrl ?? "").trim();
  return /^data:image\/[a-z0-9.+-]+;base64,/i.test(s) ? s : "";
}

function safeChatVideoDataUrl(rawUrl) {
  const s = String(rawUrl ?? "").trim();
  return /^data:video\/[a-z0-9.+-]+;base64,/i.test(s) ? s : "";
}

function triggerChatImagePicker() {
  const f = chatEl("chat-file-input");
  if (f) f.click();
}

function handleChatFileInputChange(e) {
  if (e && e.target && e.target.files && e.target.files.length) queueChatMediaFiles(e.target.files);
  if (e && e.target) e.target.value = "";
}

function handleChatDragOver(e) {
  e.preventDefault();
  if (chatEl("chat-composer")) chatEl("chat-composer").classList.add("is-dragover");
}

function handleChatDragLeave(_) {
  if (chatEl("chat-composer")) chatEl("chat-composer").classList.remove("is-dragover");
}

function handleChatDrop(e) {
  e.preventDefault();
  handleChatDragLeave(e);
  if (e && e.dataTransfer && e.dataTransfer.files) queueChatMediaFiles(e.dataTransfer.files);
}

function handleChatPaste(event) {
  const items = event && event.clipboardData ? event.clipboardData.items : null;
  if (!items) return;
  const files = [];
  for (let i = 0; i < items.length; i += 1) {
    const it = items[i];
    if (it && it.kind === "file" && it.type && (it.type.startsWith("image/") || it.type.startsWith("video/"))) {
      const f = it.getAsFile();
      if (f) files.push(f);
    }
  }
  if (files.length > 0) queueChatMediaFiles(files);
}

function queueChatImageFiles(fileList) {
  queueChatMediaFiles(fileList);
}

function queueChatMediaFiles(fileList) {
  Array.from(fileList || []).forEach((file) => {
    if (!file || !file.type) return;
    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");
    if (!isImage && !isVideo) return;
    if (isVideo && typeof file.size === "number" && file.size > CHAT_MAX_VIDEO_BYTES) {
      notifyChat(`Video "${file.name || "clip"}" exceeds 64 MiB WebUI limit (use qwg run --video for large files).`, true);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") return;
      if (isImage) {
        chatState.pendingImages.push({
          id: generateChatId("img"),
          name: file.name || "pasted-image.png",
          mimeType: file.type,
          dataUrl: reader.result
        });
      } else {
        if (!Array.isArray(chatState.pendingVideos)) chatState.pendingVideos = [];
        chatState.pendingVideos.push({
          id: generateChatId("vid"),
          name: file.name || "clip.mp4",
          mimeType: file.type,
          dataUrl: reader.result
        });
      }
      renderChatAttachments();
    };
    reader.readAsDataURL(file);
  });
}

function removePendingChatImage(imgId) {
  chatState.pendingImages = (chatState.pendingImages || []).filter((img) => img.id !== imgId);
  renderChatAttachments();
}

function removePendingChatVideo(vidId) {
  chatState.pendingVideos = (chatState.pendingVideos || []).filter((vid) => vid.id !== vidId);
  renderChatAttachments();
}

function renderChatAttachments() {
  const strip = chatEl("chat-attachments-strip");
  if (!strip) return;
  const imgs = Array.isArray(chatState.pendingImages) ? chatState.pendingImages : [];
  const vids = Array.isArray(chatState.pendingVideos) ? chatState.pendingVideos : [];
  if (!imgs.length && !vids.length) {
    strip.hidden = true;
    strip.innerHTML = "";
    return;
  }
  strip.hidden = false;
  const modelObj = typeof publishedModel !== "undefined" ? publishedModel : null;
  const connected = typeof isConnected !== "undefined" && isConnected;
  const hasVision = connected && modelObj && Array.isArray(modelObj.capabilities) && modelObj.capabilities.includes("vision");
  const warnHtml = connected && modelObj && !hasVision
    ? `<div class="chat-attachment-warn">Note: active model capabilities do not include vision; media parts may be rejected.</div>`
    : "";
  const imgChips = imgs.map((img) =>
    `<div class="chat-attachment-chip"><img src="${escapeHtml(safeChatImageDataUrl(img.dataUrl))}" alt="${escapeHtml(img.name)}" /><span class="mono">${escapeHtml(img.name)}</span>
      <button type="button" class="chat-session-del" data-chat-action="remove-image" data-img-id="${escapeHtml(img.id)}" aria-label="Remove image ${escapeHtml(img.name)}">×</button></div>`
  ).join("");
  const vidChips = vids.map((vid) =>
    `<div class="chat-attachment-chip"><span class="knob-pill mono">VIDEO</span><span class="mono">${escapeHtml(vid.name)}</span>
      <button type="button" class="chat-session-del" data-chat-action="remove-video" data-vid-id="${escapeHtml(vid.id)}" aria-label="Remove video ${escapeHtml(vid.name)}">×</button></div>`
  ).join("");
  strip.innerHTML = warnHtml + imgChips + vidChips;
}

async function captureChatScreenSnapshot() {
  if (typeof navigator !== "undefined" && navigator.mediaDevices && typeof navigator.mediaDevices.getDisplayMedia === "function") {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      const videoEl = document.createElement("video");
      videoEl.muted = true;
      videoEl.playsInline = true;
      videoEl.srcObject = stream;
      await videoEl.play();
      const rawW = Math.max(1, videoEl.videoWidth || 1280);
      const rawH = Math.max(1, videoEl.videoHeight || 720);
      const scale = Math.min(1, 1920 / Math.max(rawW, rawH));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(rawW * scale));
      canvas.height = Math.max(1, Math.round(rawH * scale));
      const ctx2d = canvas.getContext("2d");
      ctx2d.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/png");
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      chatState.pendingImages.push({
        id: generateChatId("screen"),
        name: `screen-${canvas.width}x${canvas.height}.png`,
        mimeType: "image/png",
        dataUrl
      });
      renderChatAttachments();
      notifyChat(`Captured ${canvas.width}×${canvas.height} screen snapshot.`);
      return;
    } catch (err) {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (err && (err.name === "NotAllowedError" || err.name === "AbortError")) return;
    }
  }
  try {
    const res = await fetch(`${API_BASE}/v1/vision/screen-capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: true, max_dimension: 1920 })
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error((json.error && json.error.message) || json.message || `HTTP ${res.status}`);
    }
    if (typeof json.data_url === "string" && safeChatImageDataUrl(json.data_url)) {
      chatState.pendingImages.push({
        id: generateChatId("screen"),
        name: `display-${json.width || 0}x${json.height || 0}.png`,
        mimeType: "image/png",
        dataUrl: json.data_url
      });
      renderChatAttachments();
      notifyChat(`Captured ${json.width}×${json.height} display snapshot.`);
    }
  } catch (err) {
    notifyChat("Screen capture failed: " + (err && err.message ? err.message : "denied"), true);
  }
}
