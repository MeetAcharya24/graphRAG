"use strict";

/* GraphRAG UI.
 *
 * Two things here are design, not plumbing, and are worth reading before
 * changing:
 *
 * 1. ASK_STAGES choreographs the wait. /api/ask is one blocking request with
 *    no streaming, so the stage names cannot arrive live - but the stages are
 *    real, and so are their durations, measured over HTTP (see STATUS.md:
 *    embed 0.7s, vector_search 1.2s, answer 1.7s, graph work the rest, 6.5s
 *    total). This is a timeline calibrated to those measurements, and it never
 *    claims completion the response has not confirmed: the last stage holds,
 *    and admits it when the request outruns the budget.
 *
 * 2. NODE_KINDS infers what a path node *is* from the relation vocabulary.
 *    The response carries path nodes as names only, so this is the only kind
 *    signal available without a backend change. Unambiguous edges only; an
 *    unrecognised or conflicting edge leaves its endpoints unmarked rather
 *    than guessing at them.
 */

const WAKE_DELAY_MS = 3500;
const POLL_INTERVAL_MS = 900;
const AUTH_FLAG = "graphrag_authed";
const PILL_STEP_MS = 120;
const DELETE_CONFIRM_TIMEOUT_MS = 3000;

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const el = (id) => document.getElementById(id);

const loginView = el("login-view");
const appView = el("app-view");
const loginForm = el("login-form");
const loginError = el("login-error");
const logoutButton = el("logout-button");
const notices = el("notices");
const deskEmpty = el("desk-empty");
const exampleQuestion = el("example-question");
const uploadForm = el("upload-form");
const dropZone = el("drop-zone");
const fileInput = el("file-input");
const uploadList = el("upload-list");
const askForm = el("ask-form");
const questionInput = el("question");
const askButton = el("ask-button");
const progress = el("progress");
const progressStage = el("progress-stage");
const wireFill = el("wire-fill");
const log = el("log");
const statusBar = el("status-bar");
const statusStats = el("status-stats");
const entryTemplate = el("entry-template");
const sourceTemplate = el("source-template");

// --- notices: one slot, calm copy, no stacked alarm bars -----------------

/** Adds or replaces the notice with `key`; `null` text removes it. */
function setNotice(key, text, tone = "lamp") {
  const existing = notices.querySelector(`[data-notice="${key}"]`);
  if (!text) {
    if (existing) existing.remove();
    return;
  }
  const node = existing || document.createElement("p");
  node.dataset.notice = key;
  node.className = tone === "signal" ? "note note-signal" : "note note-lamp";
  node.textContent = text;
  if (!existing) notices.appendChild(node);
}

let pendingSlow = 0;
let wakeTimerHandle = null;

function showWakeNotice() {
  pendingSlow += 1;
  if (wakeTimerHandle === null) {
    wakeTimerHandle = setTimeout(() => {
      setNotice("wake", "Waking the server. From cold the free tier takes up to a minute.");
    }, WAKE_DELAY_MS);
  }
}

function hideWakeNotice() {
  pendingSlow = Math.max(0, pendingSlow - 1);
  if (pendingSlow === 0) {
    clearTimeout(wakeTimerHandle);
    wakeTimerHandle = null;
    setNotice("wake", null);
  }
}

// --- the API envelope ----------------------------------------------------

/**
 * POSTs/GETs the API envelope, never throwing: callers branch on `.ok`.
 *
 * `trackWake` opts into the "waking the server" notice. It is only set for
 * the login and health calls that make up app *loading* - a slow in-app call
 * afterwards (asking a question, polling a job) is not the server waking up,
 * and labelling it that way would mislead, so those calls never touch the
 * notice no matter how long they take.
 */
async function api(path, options = {}, { trackWake = false } = {}) {
  if (trackWake) showWakeNotice();
  try {
    const response = await fetch(path, { credentials: "same-origin", ...options });
    const body = await response.json();
    if (!body.ok && body.error && body.error.code === "not_authenticated") {
      enterLoginView("That session has ended. Log in again.");
    }
    return { ...body, status: response.status, retryAfter: response.headers.get("retry-after") };
  } catch (_err) {
    return {
      ok: false,
      data: null,
      error: { code: "network_error", message: "", hint: "" },
      status: 0,
      retryAfter: null,
    };
  } finally {
    if (trackWake) hideWakeNotice();
  }
}

/* Every failure the user can reach, written out. Keyed on the stable error
 * code rather than the server's message, which is deliberately terse; where
 * there is no entry the server's own message and hint are shown instead, so a
 * new code is never swallowed. */
const ERROR_COPY = {
  network_error: "Can't reach the server. Check the connection and ask again.",
  too_many_requests: "That's a lot of questions at once. The next one can go in a moment.",
  llm_rate_limited: "The model is rate-limited right now. It clears in under a minute.",
  llm_quota_exceeded: "Today's model quota is spent. It resets tomorrow.",
  llm_unavailable: "The model is overloaded. It usually frees up within a minute.",
  llm_bad_response: "The model returned something unreadable. Asking again usually fixes it.",
  answer_budget_exhausted:
    "Finding the evidence ran past the time budget. A narrower question usually lands.",
  database_unavailable: "The database is unreachable, so there is nothing to search yet.",
  unsupported_file_type: "That file type can't be read. Use .pdf, .txt or .md.",
  upload_too_large: "That file is over the 1 MB limit.",
  empty_document: "That file has no readable text in it.",
  too_many_chunks: "That document is too long to ingest in one piece. Split it and retry.",
  job_not_found: "That upload's progress is no longer being tracked.",
};

function describeError(error) {
  if (!error) return "Something went wrong.";
  const copy = ERROR_COPY[error.code];
  if (copy) return copy;
  const message = error.message || "Something went wrong.";
  return error.hint ? `${message} ${error.hint}` : message;
}

/** Disables `button` for `seconds`, re-enabling it with a countdown in its label. */
function disableFor(button, seconds, label) {
  const original = button.textContent;
  let remaining = Math.max(1, Math.ceil(seconds));
  button.disabled = true;
  const tick = () => {
    button.textContent = `${label} in ${remaining}s`;
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

// --- login / logout ------------------------------------------------------

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
  await refreshDeskState();
}

/* The desk has two rest states. With nothing ingested there is nothing to
 * ask about, so the invitation to add a document becomes the page; once
 * there is, the drop zone demotes to a strip under the ask bar. A backend
 * that cannot report a count (the in-memory store) leaves the desk in its
 * working state rather than wrongly hiding the arrival copy. */
async function refreshDeskState() {
  const result = await api("/api/health", {}, { trackWake: true });
  const storage = (result.data && result.data.storage) || null;

  const unreachable = storage && storage.backend !== "memory" && !storage.connected;
  setNotice(
    "storage",
    unreachable
      ? `${ERROR_COPY.database_unavailable} Uploads and questions will fail until it is back.`
      : null,
    "signal"
  );

  const count = storage && typeof storage.chunk_count === "number" ? storage.chunk_count : null;
  setDeskEmpty(count === 0);
  renderStatusBar(storage);
}

function setDeskEmpty(isEmpty) {
  appView.dataset.desk = isEmpty ? "empty" : "ready";
  deskEmpty.hidden = !isEmpty;
}

function plural(count, one, many) {
  return count === 1 ? one : many;
}

/* The status footer, always present once logged in rather than appearing
 * and disappearing - a stable bottom edge for the page, and a place for
 * each storage state to say plainly where things stand. */
function renderStatusBar(storage) {
  if (!storage) {
    statusBar.hidden = true;
    return;
  }
  statusBar.hidden = false;

  if (storage.backend === "memory") {
    statusStats.textContent = "In-memory storage — nothing persists across a restart.";
    return;
  }
  if (!storage.connected) {
    statusStats.textContent = "Database unreachable — counts unavailable right now.";
    return;
  }

  const graph = storage.graph || {};
  const entities = typeof graph.entity_count === "number" ? graph.entity_count : 0;
  const relations = typeof graph.relation_count === "number" ? graph.relation_count : 0;
  const chunks = typeof storage.chunk_count === "number" ? storage.chunk_count : 0;
  statusStats.textContent =
    `${entities} ${plural(entities, "entity", "entities")} · ` +
    `${relations} ${plural(relations, "relation", "relations")} · ` +
    `${chunks} ${plural(chunks, "chunk", "chunks")} indexed`;
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  const submitButton = loginForm.querySelector("button[type=submit]");
  submitButton.disabled = true;

  const result = await api(
    "/api/login",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: el("password").value }),
    },
    { trackWake: true }
  );

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

// --- upload --------------------------------------------------------------

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

/* Named for what the person is waiting on, not for the pipeline step. */
const STAGE_COPY = {
  chunking: "splitting into chunks",
  embedding: "embedding",
  extracting: "reading entities",
  linking: "wiring the graph",
};

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
  row.innerHTML =
    '<div class="upload-info">' +
    '<div class="filename"></div><div class="status-line"></div>' +
    '<div class="meter"><span></span></div>' +
    "</div>" +
    '<button type="button" class="delete-doc" aria-label="Delete this document">Delete</button>';
  row.querySelector(".filename").textContent = file.name;
  const statusLine = row.querySelector(".status-line");
  const meter = row.querySelector(".meter span");
  uploadList.appendChild(row);

  const fail = (text) => {
    statusLine.textContent = text;
    row.classList.remove("working");
    row.classList.add("failed");
  };

  if (!SUPPORTED_EXTENSIONS.includes(extensionOf(file.name))) {
    fail(`That file type can't be read. Use ${SUPPORTED_EXTENSIONS.join(", ")}.`);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    fail("That file is over the 1 MB limit.");
    return;
  }

  row.classList.add("working");
  statusLine.textContent = "uploading";
  const formData = new FormData();
  formData.append("file", file);

  api("/api/documents", { method: "POST", body: formData }).then((result) => {
    if (!result.ok) {
      fail(describeError(result.error));
      return;
    }
    wireDeleteButton(row, result.data.doc_id, statusLine);
    pollJob(result.data.job_id, { row, statusLine, meter, fail });
  });
}

/* A document is only deletable once it is actually in the graph - a row
 * still uploading or failed has nothing server-side worth removing. The
 * button stays in the markup throughout (simpler than inserting it later)
 * but only `.upload-row.done` shows it, in CSS.
 *
 * Confirmation is inline, not a native confirm() dialog: a second click
 * within DELETE_CONFIRM_TIMEOUT_MS commits, otherwise the button reverts -
 * calmer than a popup, and the only way to style it to match the page. */
function wireDeleteButton(row, docId, statusLine) {
  const button = row.querySelector(".delete-doc");
  let revertTimer = null;

  const revert = () => {
    clearTimeout(revertTimer);
    button.classList.remove("confirming");
    button.textContent = "Delete";
  };

  button.addEventListener("click", () => {
    if (!button.classList.contains("confirming")) {
      button.classList.add("confirming");
      button.textContent = "Confirm delete?";
      revertTimer = setTimeout(revert, DELETE_CONFIRM_TIMEOUT_MS);
      return;
    }

    clearTimeout(revertTimer);
    button.disabled = true;
    button.textContent = "Deleting…";

    api(`/api/documents/${docId}`, { method: "DELETE" }).then((result) => {
      if (!result.ok) {
        button.disabled = false;
        revert();
        statusLine.textContent = describeError(result.error);
        return;
      }
      row.remove();
    });
  });
}

function pollJob(jobId, { row, statusLine, meter, fail }) {
  const tick = async () => {
    const result = await api(`/api/jobs/${jobId}`);
    if (!result.ok) {
      fail(describeError(result.error));
      return;
    }
    const job = result.data;

    if (job.state === "failed") {
      fail(job.error || "Ingesting that document failed.");
      return;
    }
    if (job.state === "succeeded") {
      row.classList.remove("working");
      row.classList.add("done");
      statusLine.textContent = `ready — ${job.chunks_total} chunk${
        job.chunks_total === 1 ? "" : "s"
      } in the graph`;
      // The desk now has work on it, so leave the empty state behind.
      setDeskEmpty(false);
      return;
    }
    if (job.state === "queued") {
      statusLine.textContent = "queued";
    } else {
      const stage = STAGE_COPY[job.stage] || job.stage;
      statusLine.textContent = job.chunks_total
        ? `${stage} — ${job.chunks_done}/${job.chunks_total} chunks`
        : stage;
    }
    if (job.chunks_total) {
      meter.style.width = `${Math.round((job.chunks_done / job.chunks_total) * 100)}%`;
    }
    setTimeout(tick, POLL_INTERVAL_MS);
  };
  tick();
}

// --- the choreographed wait ----------------------------------------------

/* The four stages a person can care about, each covering the real retrieval
 * stages behind it, with the durations measured over HTTP. */
const ASK_STAGES = [
  { label: "reading the question", seconds: 0.7 }, // embed_question
  { label: "searching documents", seconds: 1.8 }, // vector_search, seed_resolution
  { label: "walking the graph", seconds: 2.4 }, // expand_subgraph, shortest_paths, chunk_recovery
  { label: "writing the answer", seconds: 1.7 }, // answer
];
const ASK_TOTAL_SECONDS = ASK_STAGES.reduce((sum, stage) => sum + stage.seconds, 0);
const OVERRUN_GRACE_SECONDS = 2.5;

let progressTimer = null;

function startProgress() {
  stopProgress();
  progress.hidden = false;
  wireFill.style.transitionDuration = "0s";
  wireFill.style.width = "0%";
  // Reflow, so the reset to 0% is not coalesced with the first stage's width
  // and animated away from wherever the previous question left the wire.
  void wireFill.offsetWidth;

  let elapsed = 0;
  let index = 0;

  const advance = () => {
    const stage = ASK_STAGES[index];
    progressStage.textContent = stage.label;
    elapsed += stage.seconds;
    // The wire travels across the stage's own duration, so it moves with the
    // work rather than in jumps. Under reduced motion the CSS transition is
    // absent and this simply steps.
    wireFill.style.transitionDuration = `${stage.seconds}s`;
    wireFill.style.width = `${Math.min(100, (elapsed / ASK_TOTAL_SECONDS) * 100)}%`;
    index += 1;

    if (index < ASK_STAGES.length) {
      progressTimer = setTimeout(advance, stage.seconds * 1000);
      return;
    }
    // Past the measured budget the honest thing is to say so, not to sit on a
    // wire that looks finished.
    progressTimer = setTimeout(() => {
      progressStage.textContent = "still writing — this one is taking longer than usual";
    }, (stage.seconds + OVERRUN_GRACE_SECONDS) * 1000);
  };

  advance();
}

function stopProgress() {
  clearTimeout(progressTimer);
  progressTimer = null;
  progress.hidden = true;
  progressStage.textContent = "";
}

// --- ask -----------------------------------------------------------------

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) askQuestion(question);
});

// The empty desk's example is a real question, not a decoration: clicking it
// runs the exact same flow a typed question would, including the honest
// insufficient-evidence result it gets before any document is uploaded.
exampleQuestion.addEventListener("click", () => {
  const question = exampleQuestion.dataset.question;
  questionInput.value = question;
  askQuestion(question);
});

async function askQuestion(question) {
  askButton.disabled = true;
  startProgress();

  // Everything from here on must still leave the person with a visible
  // result: a thrown error inside fetch, inside rendering, anywhere. The
  // finally block is what guarantees the wait always ends and the button
  // always comes back, no matter which branch below runs or throws.
  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (result.status === 429) {
      disableFor(askButton, Number(result.retryAfter) || 30, "Ask");
    }

    appendEntry(question, result);
    if (result.ok) {
      questionInput.value = "";
    }
  } catch (err) {
    // appendEntry renders a rich result and can in principle throw on a
    // response shape it does not expect; a visible, plain-worded entry
    // beats losing the question into an unhandled rejection.
    appendPlainError(question, "Something went wrong showing that answer. Try asking again.");
    console.error("Rendering the answer failed:", err);
  } finally {
    stopProgress();
    askButton.disabled = false;
  }
}

/** The one thing every ask outcome is guaranteed to produce, in isolation
 * from the richer renderers below - used for a genuine fetch/HTTP failure
 * and as the last-resort fallback when rendering a success throws. */
function appendPlainError(question, message) {
  const node = entryTemplate.content.cloneNode(true);
  const entry = node.querySelector(".entry");
  entry.classList.add("entry-error");
  entry.querySelector(".entry-question").textContent = question;
  entry.querySelector(".confidence-fill").style.height = "100%";
  entry.querySelector(".entry-answer").appendChild(paragraph(message));
  log.appendChild(entry);
  revealNewest(entry);
}

function appendEntry(question, result) {
  if (!result.ok) {
    appendPlainError(question, describeError(result.error));
    return;
  }

  const node = entryTemplate.content.cloneNode(true);
  const entry = node.querySelector(".entry");
  entry.querySelector(".entry-question").textContent = question;

  const answerBlock = entry.querySelector(".entry-answer-block");
  const answer = entry.querySelector(".entry-answer");
  const fill = entry.querySelector(".confidence-fill");

  const data = result.data;
  const paths = Array.isArray(data.paths) ? data.paths : [];
  const sources = Array.isArray(data.sources) ? data.sources : [];

  // The path draws first, then the answer rises as it finishes.
  const pathsHost = entry.querySelector(".entry-paths");
  let pillIndex = 0;
  paths.forEach((pathText) => {
    const { chain, nextIndex } = renderPathChain(pathText, pillIndex);
    if (chain) {
      pathsHost.appendChild(chain);
      pillIndex = nextIndex;
    }
  });
  entry.style.setProperty("--after-path", `${pillIndex * PILL_STEP_MS + 200}ms`);

  if (data.insufficient_evidence) entry.classList.add("no-evidence");

  const percent = Math.round((Number(data.confidence) || 0) * 100);
  fill.style.height = `${percent}%`;
  answerBlock.title = `Confidence ${percent}%`;

  renderAnswer(answer, String(data.answer || ""));
  answer.appendChild(hiddenLine(`Confidence ${percent}%.`));

  if (sources.length) {
    entry.querySelector(".entry-sources").appendChild(renderSources(sources));
  }

  // Built off-document and appended only once everything above succeeded,
  // so a renderer that throws midway never leaves a half-built entry stuck
  // invisible in a detached fragment - the catch in the submit handler
  // falls back to appendPlainError instead.
  log.appendChild(entry);
  revealNewest(entry);
}

/* Brings the new finding into view without stealing focus from the field the
 * person may still be typing in. */
function revealNewest(entry) {
  entry.scrollIntoView({
    block: "nearest",
    behavior: reduceMotion.matches ? "auto" : "smooth",
  });
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function hiddenLine(text) {
  const p = document.createElement("p");
  p.className = "visually-hidden";
  p.textContent = text;
  return p;
}

// --- the answer ----------------------------------------------------------

/**
 * Renders the answer text, as a real list when it is one.
 *
 * A model asked for "concise and specific" often returns an enumeration; run
 * together into one paragraph it is noticeably harder to read than the thing
 * it is. Only an ascending 1, 2, 3 run counts, so a stray "2." mid-sentence
 * cannot turn prose into a list.
 */
function renderAnswer(host, text) {
  const numbered = splitNumbered(text);
  if (numbered) {
    if (numbered.lead) host.appendChild(paragraph(numbered.lead));
    const list = document.createElement("ol");
    numbered.items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
    host.appendChild(list);
    return;
  }
  text
    .split(/\n\s*\n|\n/)
    .map((block) => block.trim())
    .filter(Boolean)
    .forEach((block) => host.appendChild(paragraph(block)));
}

function splitNumbered(text) {
  const marker = /(?:^|\s)(\d{1,2})[.)]\s+/g;
  const run = [];
  let expected = 1;
  let match;
  while ((match = marker.exec(text)) !== null) {
    if (Number(match[1]) !== expected) continue;
    run.push({ at: match.index, from: match.index + match[0].length });
    expected += 1;
  }
  if (run.length < 2) return null;

  return {
    lead: text.slice(0, run[0].at).trim(),
    items: run.map((hit, i) =>
      text.slice(hit.from, i + 1 < run.length ? run[i + 1].at : text.length).trim()
    ),
  };
}

// --- the path chain ------------------------------------------------------

/* What each relation says about the things at its ends. Only the edges whose
 * endpoints the extraction vocabulary actually pins down are listed; see the
 * NODE_KINDS note at the top of the file. */
const NODE_KINDS = {
  built: ["actor", "work"],
  delivered_service_to: ["work", "actor"],
  used_technology: ["actor", "material"],
  produces: ["actor", "work"],
  works_at: ["actor", "actor"],
  founded: ["actor", "actor"],
  reports_to: ["actor", "actor"],
  acquired: ["actor", "actor"],
  acquired_by: ["actor", "actor"],
  collaborates_with: ["actor", "actor"],
  competes_with: ["actor", "actor"],
  located_in: [null, "place"],
};

const KIND_LABELS = {
  work: "service or product",
  material: "technology",
  place: "place",
  actor: "organisation or person",
};

/** One kind per node, from the edges on either side of it; conflicts go neutral. */
function inferKinds(names, relations) {
  return names.map((_name, i) => {
    const incoming = i > 0 ? (NODE_KINDS[relations[i - 1]] || [])[1] : undefined;
    const outgoing = i < relations.length ? (NODE_KINDS[relations[i]] || [])[0] : undefined;
    const candidates = [incoming, outgoing].filter(Boolean);
    if (!candidates.length) return "unknown";
    return candidates.every((kind) => kind === candidates[0]) ? candidates[0] : "unknown";
  });
}

/**
 * `A —built→ B —delivered_service_to→ C` as pills strung on a wire.
 *
 * `startIndex` continues the draw order from the previous chain, so several
 * paths read as one sweep across the entry rather than racing each other.
 */
function renderPathChain(pathText, startIndex) {
  const parts = String(pathText).split(/\s—(.+?)→\s/);
  if (parts.length < 3) return { chain: null, nextIndex: startIndex };

  const names = parts.filter((_part, i) => i % 2 === 0);
  const relations = parts.filter((_part, i) => i % 2 === 1);
  const kinds = inferKinds(names, relations);

  const chain = document.createElement("div");
  chain.className = "path-chain";
  chain.setAttribute("role", "group");
  chain.setAttribute("aria-label", "Graph path");
  // The chain scrolls sideways when it outruns the track, so it has to be
  // reachable by keyboard to be scrollable without a mouse.
  chain.tabIndex = 0;

  const run = document.createElement("div");
  run.className = "path-run";
  chain.appendChild(run);

  let index = startIndex;
  names.forEach((name, i) => {
    if (i > 0) {
      const link = document.createElement("span");
      link.className = "path-link";
      link.style.setProperty("--i", String(index + 1));

      const relation = document.createElement("span");
      relation.className = "path-relation";
      relation.textContent = relations[i - 1].replace(/_/g, " ");
      link.appendChild(relation);

      const wire = document.createElement("span");
      wire.className = "path-wire";
      link.appendChild(wire);
      run.appendChild(link);
      index += 1;
    }

    const pill = document.createElement("span");
    pill.className = "path-pill";
    pill.dataset.kind = kinds[i];
    pill.style.setProperty("--i", String(index));
    pill.textContent = name;
    if (KIND_LABELS[kinds[i]]) pill.title = KIND_LABELS[kinds[i]];
    run.appendChild(pill);
  });

  return { chain, nextIndex: index + 1 };
}

// --- sources -------------------------------------------------------------

function renderSources(sources) {
  const list = document.createElement("ul");
  list.className = "source-list";

  sources.forEach((source, i) => {
    const node = sourceTemplate.content.cloneNode(true);
    const row = node.querySelector(".source-row");
    const toggle = row.querySelector(".source-toggle");
    const drawer = row.querySelector(".source-drawer");
    const preview = row.querySelector(".source-excerpt-preview");
    const full = row.querySelector(".source-excerpt-full");

    const chunkIndex = Number(String(source.chunk_id || "").split(":").pop());
    const label = source.filename || source.doc_id || "unknown document";
    const heading = Number.isInteger(chunkIndex) ? `${label} · chunk ${chunkIndex + 1}` : label;

    row.querySelector(".source-number").textContent = String(i + 1);
    row.querySelector(".source-file").textContent = heading;
    preview.textContent = truncate(source.excerpt || "", 110);
    full.textContent = source.excerpt || "";
    toggle.setAttribute("aria-label", `Source ${i + 1}, ${heading}`);

    toggle.addEventListener("click", () => toggleDrawer(toggle, drawer));
    list.appendChild(row);
  });
  return list;
}

/* Animates to the drawer's measured height rather than a guessed max-height,
 * then releases to `auto` so a long chunk can still reflow afterwards. */
function toggleDrawer(toggle, drawer) {
  const open = toggle.getAttribute("aria-expanded") === "true";
  toggle.setAttribute("aria-expanded", String(!open));

  if (reduceMotion.matches) {
    drawer.hidden = open;
    drawer.classList.toggle("open", !open);
    return;
  }

  if (open) {
    drawer.style.height = `${drawer.scrollHeight}px`;
    drawer.classList.remove("open");
    requestAnimationFrame(() => {
      drawer.style.height = "0px";
    });
    drawer.addEventListener(
      "transitionend",
      () => {
        if (toggle.getAttribute("aria-expanded") === "false") drawer.hidden = true;
      },
      { once: true }
    );
    return;
  }

  drawer.hidden = false;
  drawer.style.height = "0px";
  requestAnimationFrame(() => {
    drawer.style.height = `${drawer.scrollHeight}px`;
  });
  drawer.addEventListener(
    "transitionend",
    () => {
      if (toggle.getAttribute("aria-expanded") === "true") {
        drawer.classList.add("open");
        drawer.style.height = "";
      }
    },
    { once: true }
  );
}

function truncate(text, length) {
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

// --- boot ----------------------------------------------------------------

if (sessionStorage.getItem(AUTH_FLAG)) {
  enterAppView();
} else {
  loginView.hidden = false;
}
