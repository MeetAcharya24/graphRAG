"use strict";

const WAKE_DELAY_MS = 3500;
const POLL_INTERVAL_MS = 900;
const AUTH_FLAG = "graphrag_authed";

const el = (id) => document.getElementById(id);

const loginView = el("login-view");
const appView = el("app-view");
const loginForm = el("login-form");
const loginError = el("login-error");
const logoutButton = el("logout-button");
const storageBanner = el("storage-banner");
const wakeBanner = el("wake-banner");
const uploadForm = el("upload-form");
const dropZone = el("drop-zone");
const fileInput = el("file-input");
const uploadList = el("upload-list");
const askForm = el("ask-form");
const questionInput = el("question");
const askButton = el("ask-button");
const log = el("log");
const entryTemplate = el("entry-template");
const sourceTemplate = el("source-template");

let pendingSlow = 0;
let wakeTimerHandle = null;

function showWakeBanner() {
  pendingSlow += 1;
  if (wakeTimerHandle === null) {
    wakeTimerHandle = setTimeout(() => {
      wakeBanner.hidden = false;
    }, WAKE_DELAY_MS);
  }
}

function hideWakeBanner() {
  pendingSlow = Math.max(0, pendingSlow - 1);
  if (pendingSlow === 0) {
    clearTimeout(wakeTimerHandle);
    wakeTimerHandle = null;
    wakeBanner.hidden = true;
  }
}

/** POSTs/GETs the API envelope, never throwing: callers branch on `.ok`. */
async function api(path, options = {}) {
  showWakeBanner();
  try {
    const response = await fetch(path, { credentials: "same-origin", ...options });
    const body = await response.json();
    if (!body.ok && body.error && body.error.code === "not_authenticated") {
      enterLoginView("Your session ended. Log in again.");
    }
    return { ...body, status: response.status, retryAfter: response.headers.get("retry-after") };
  } catch (_err) {
    return {
      ok: false,
      data: null,
      error: {
        code: "network_error",
        message: "Could not reach the server.",
        hint: "Check your connection and try again.",
      },
      status: 0,
      retryAfter: null,
    };
  } finally {
    hideWakeBanner();
  }
}

function describeError(error) {
  if (!error) return "Something went wrong.";
  return error.hint ? `${error.message} ${error.hint}` : error.message;
}

/** Disables `button` for `seconds`, re-enabling it with a countdown in its label. */
function disableFor(button, seconds, label) {
  const original = button.textContent;
  let remaining = Math.max(1, Math.ceil(seconds));
  button.disabled = true;
  const tick = () => {
    button.textContent = `${label} (${remaining}s)`;
    remaining -= 1;
    if (remaining < 0) {
      clearInterval(interval);
      button.disabled = false;
      button.textContent = original;
    }
  };
  const interval = setInterval(tick, 1000);
  tick();
}

// --- login / logout --------------------------------------------------------

function enterLoginView(message) {
  sessionStorage.removeItem(AUTH_FLAG);
  appView.hidden = true;
  loginView.hidden = false;
  if (message) {
    loginError.textContent = message;
    loginError.hidden = false;
  }
}

async function enterAppView() {
  sessionStorage.setItem(AUTH_FLAG, "1");
  loginView.hidden = true;
  appView.hidden = false;
  await refreshStorageBanner();
}

async function refreshStorageBanner() {
  const result = await api("/api/health");
  const storage = result.data && result.data.storage;
  if (storage && storage.backend !== "memory" && !storage.connected) {
    storageBanner.textContent = `The database is unreachable${
      storage.error ? `: ${storage.error}` : "."
    } Uploads and questions will fail until it recovers.`;
    storageBanner.hidden = false;
  } else {
    storageBanner.hidden = true;
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  const submitButton = loginForm.querySelector("button[type=submit]");
  submitButton.disabled = true;

  const result = await api("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password: el("password").value }),
  });

  submitButton.disabled = false;
  if (!result.ok) {
    if (result.status === 429) {
      disableFor(submitButton, Number(result.retryAfter) || 30, "Enter");
    }
    loginError.textContent = describeError(result.error);
    loginError.hidden = false;
    return;
  }
  el("password").value = "";
  await enterAppView();
});

logoutButton.addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  enterLoginView();
});

// --- upload ------------------------------------------------------------

["dragover", "dragleave", "drop"].forEach((type) => {
  dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.toggle("drag-over", type === "dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  uploadFiles(Array.from(event.dataTransfer.files));
});

fileInput.addEventListener("change", () => {
  uploadFiles(Array.from(fileInput.files));
  fileInput.value = "";
});

uploadForm.addEventListener("submit", (event) => event.preventDefault());

const SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"];
const MAX_UPLOAD_BYTES = 1024 * 1024;

function extensionOf(filename) {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? "" : filename.slice(dot).toLowerCase();
}

function uploadFiles(files) {
  files.forEach(uploadOneFile);
}

function uploadOneFile(file) {
  const row = document.createElement("li");
  row.className = "upload-row";
  row.innerHTML = `<div class="filename"></div><div class="status-line"></div>`;
  row.querySelector(".filename").textContent = file.name;
  const statusLine = row.querySelector(".status-line");
  uploadList.appendChild(row);

  if (!SUPPORTED_EXTENSIONS.includes(extensionOf(file.name))) {
    statusLine.textContent = `Not a supported type. Use ${SUPPORTED_EXTENSIONS.join(", ")}.`;
    row.classList.add("failed");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    statusLine.textContent = "Larger than the 1 MB limit.";
    row.classList.add("failed");
    return;
  }

  statusLine.textContent = "Uploading…";
  const formData = new FormData();
  formData.append("file", file);

  api("/api/documents", { method: "POST", body: formData }).then((result) => {
    if (!result.ok) {
      statusLine.textContent = describeError(result.error);
      row.classList.add("failed");
      return;
    }
    pollJob(result.data.job_id, statusLine, row);
  });
}

function pollJob(jobId, statusLine, row) {
  const tick = async () => {
    const result = await api(`/api/jobs/${jobId}`);
    if (!result.ok) {
      statusLine.textContent = describeError(result.error);
      row.classList.add("failed");
      return;
    }
    const job = result.data;
    if (job.state === "failed") {
      statusLine.textContent = job.error || "Ingestion failed.";
      row.classList.add("failed");
      return;
    }
    if (job.state === "succeeded") {
      statusLine.textContent = `Ready — ${job.chunks_total} chunk(s).`;
      return;
    }
    statusLine.textContent = `${job.stage} — ${job.chunks_done}/${job.chunks_total} chunks`;
    setTimeout(tick, POLL_INTERVAL_MS);
  };
  tick();
}

// --- ask -----------------------------------------------------------------

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  askButton.disabled = true;
  const result = await api("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  askButton.disabled = false;

  if (result.status === 429) {
    disableFor(askButton, Number(result.retryAfter) || 30, "Ask");
  }

  appendEntry(question, result);
  if (result.ok) questionInput.value = "";
});

function appendEntry(question, result) {
  const node = entryTemplate.content.cloneNode(true);
  const entry = node.querySelector(".entry");
  entry.querySelector(".entry-question").textContent = question;
  const body = entry.querySelector(".entry-body");

  if (!result.ok) {
    entry.classList.add("entry-error");
    body.appendChild(paragraph("entry-answer", describeError(result.error)));
    log.appendChild(entry);
    return;
  }

  const data = result.data;
  if (data.insufficient_evidence) entry.classList.add("no-evidence");
  body.appendChild(paragraph("entry-answer", data.answer));
  body.appendChild(
    paragraph("confidence-note", `Confidence ${Math.round(data.confidence * 100)}%`)
  );

  data.paths.forEach((path) => body.appendChild(renderPathChain(path)));
  if (data.sources.length) body.appendChild(renderSources(data.sources));

  log.appendChild(entry);
}

function paragraph(className, text) {
  const p = document.createElement("p");
  p.className = className;
  p.textContent = text;
  return p;
}

function renderPathChain(pathText) {
  const chain = document.createElement("div");
  chain.className = "path-chain";
  const parts = pathText.split(/\s—(.+?)→\s/);

  parts.forEach((part, index) => {
    if (index % 2 === 0) {
      const pill = document.createElement("span");
      pill.className = "path-pill";
      pill.textContent = part;
      chain.appendChild(pill);
    } else {
      const connector = document.createElement("span");
      connector.className = "path-connector";
      connector.textContent = part.replace(/_/g, " ");
      chain.appendChild(connector);
    }
  });
  return chain;
}

function renderSources(sources) {
  const list = document.createElement("ul");
  list.className = "source-list";

  sources.forEach((source) => {
    const node = sourceTemplate.content.cloneNode(true);
    const card = node.querySelector(".source-card");
    const toggle = card.querySelector(".source-toggle");
    const preview = card.querySelector(".source-excerpt-preview");
    const full = card.querySelector(".source-excerpt-full");

    card.querySelector(".source-file").textContent = `${source.doc_id} · ${source.chunk_id}`;
    preview.textContent = truncate(source.excerpt, 90);
    full.textContent = source.excerpt;

    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      full.hidden = expanded;
      preview.hidden = !expanded;
    });

    list.appendChild(card);
  });
  return list;
}

function truncate(text, length) {
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

// --- boot ------------------------------------------------------------------

if (sessionStorage.getItem(AUTH_FLAG)) {
  enterAppView();
} else {
  loginView.hidden = false;
}
