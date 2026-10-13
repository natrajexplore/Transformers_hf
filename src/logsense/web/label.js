// LogSense console — Label tab: paste raw logs -> anonymize -> label blind -> download .jsonl
// or evaluate the classifier on it. The draft (anonymized lines only) is kept in this browser.
"use strict";

const EXCLUDE = "__exclude";
const lab = store.get("labelDraft", null) || { rows: [], anon: null, focus: 0, evaluated: false };

function saveLab() {
  store.set("labelDraft", lab);
}
const labCounts = () => {
  const labeled = lab.rows.filter((r) => r.label && r.label !== EXCLUDE).length;
  const excluded = lab.rows.filter((r) => r.label === EXCLUDE).length;
  return { labeled, excluded, open: lab.rows.length - labeled - excluded };
};

function setSteps() {
  const { labeled, open } = labCounts();
  const stage = !lab.rows.length ? 1 : lab.evaluated ? 5 : open === 0 && labeled ? 4 : 3;
  document.querySelectorAll("#label .step").forEach((el) => {
    const n = Number(el.dataset.step);
    el.classList.toggle("done", n < stage || (n === 2 && lab.rows.length > 0));
    el.classList.toggle("current", n === stage);
  });
}

function renderLegend() {
  $("lab-legend").innerHTML =
    CONFIG.labels.map((l, i) => `<span class="legend-chip"><kbd>${i + 1}</kbd> ${catBadge(l)}</span>`).join("") +
    `<span class="legend-chip"><kbd>X</kbd> exclude</span>`;
}

function labRowHtml(r, i) {
  const options = [`<option value="">— choose —</option>`]
    .concat(CONFIG.labels.map((l, k) => `<option value="${l}"${r.label === l ? " selected" : ""}>${k + 1} · ${l}</option>`))
    .concat(`<option value="${EXCLUDE}"${r.label === EXCLUDE ? " selected" : ""}>✕ exclude (fits no category)</option>`)
    .join("");
  let pred = "";
  if (lab.evaluated && r.label && r.label !== EXCLUDE) {
    pred = r.predicted && r.predicted !== r.label
      ? `<div class="pred bad">✗ model said ${catBadge(r.predicted)}</div>`
      : `<div class="pred good">✓ model agrees</div>`;
  }
  return `<div class="lrow${i === lab.focus ? " focus" : ""}${r.label === EXCLUDE ? " excluded" : ""}" data-i="${i}">
    <span class="ln">${i + 1}</span><span class="ltext">${esc(r.text)}</span>
    <select aria-label="Category for line ${i + 1}">${options}</select>${pred}</div>`;
}

function renderLab() {
  $("lab-work").hidden = !lab.rows.length;
  setSteps();
  if (!lab.rows.length) return;
  renderLegend();
  $("lab-table").innerHTML = lab.rows.map(labRowHtml).join("");
  updateLabProgress();
}
function updateLabProgress() {
  const { labeled, excluded, open } = labCounts();
  const total = lab.rows.length;
  $("lab-progress-bar").style.width = `${((labeled + excluded) / total) * 100}%`;
  $("lab-progress").textContent = `${labeled} labeled · ${excluded} excluded · ${open} left of ${total}`;
  $("lab-download").disabled = $("lab-evaluate").disabled = labeled === 0;
  setSteps();
}

function setLabel(i, value) {
  const r = lab.rows[i];
  if (!r) return;
  r.label = value || null;
  lab.evaluated = false; // labels changed: old predictions no longer shown
  const el = $("lab-table").querySelector(`.lrow[data-i="${i}"]`);
  if (el) el.outerHTML = labRowHtml(r, i);
  saveLab();
  updateLabProgress();
}
function moveFocus(i, scroll = true) {
  if (!lab.rows.length) return;
  const prev = $("lab-table").querySelector(".lrow.focus");
  prev?.classList.remove("focus");
  lab.focus = Math.max(0, Math.min(lab.rows.length - 1, i));
  const el = $("lab-table").querySelector(`.lrow[data-i="${lab.focus}"]`);
  el?.classList.add("focus");
  if (scroll) el?.scrollIntoView({ block: "nearest" });
  saveLab();
}

$("lab-table").addEventListener("change", (e) => {
  const row = e.target.closest(".lrow");
  if (row && e.target.tagName === "SELECT") {
    const i = Number(row.dataset.i);
    setLabel(i, e.target.value);
    moveFocus(i);
  }
});
$("lab-table").addEventListener("click", (e) => {
  const row = e.target.closest(".lrow");
  if (row && e.target.tagName !== "SELECT") {
    moveFocus(Number(row.dataset.i), false);
    $("lab-table").focus({ preventScroll: true });
  }
});

document.addEventListener("keydown", (e) => {
  if (currentTab !== "label" || !lab.rows.length || e.altKey || e.ctrlKey || e.metaKey) return;
  const a = document.activeElement;
  if (a && (a.tagName === "TEXTAREA" || a.tagName === "INPUT" || a.tagName === "SELECT")) return;
  const k = e.key.toLowerCase();
  const n = Number(e.key);
  if (n >= 1 && n <= CONFIG.labels.length) {
    setLabel(lab.focus, CONFIG.labels[n - 1]);
    moveFocus(lab.focus + 1);
  } else if (k === "x") {
    setLabel(lab.focus, EXCLUDE);
    moveFocus(lab.focus + 1);
  } else if (k === "backspace" || k === "delete") {
    setLabel(lab.focus, "");
  } else if (k === "arrowdown" || k === "j") {
    moveFocus(lab.focus + 1);
  } else if (k === "arrowup" || k === "k") {
    moveFocus(lab.focus - 1);
  } else return;
  e.preventDefault();
});

function loadRows(lines, anonInfo) {
  lab.rows = lines.map((text) => ({ text, label: null, predicted: null }));
  lab.anon = anonInfo;
  lab.focus = 0;
  lab.evaluated = false;
  $("lab-results").innerHTML = "";
  saveLab();
  renderAnonNote();
  renderLab();
  $("lab-table").focus({ preventScroll: true });
}
function renderAnonNote() {
  const a = lab.anon;
  if (!a) return ($("lab-anon-note").innerHTML = "");
  if (a.skipped) {
    $("lab-anon-note").innerHTML = `<div class="card small"><b>Anonymization skipped.</b> <span class="muted">Make sure these lines
      contain no real IPs, users or hostnames before sharing the file.</span></div>`;
    return;
  }
  if (a.demo) {
    $("lab-anon-note").innerHTML = `<div class="card small"><b>Demo lines</b> <span class="muted">come from the synthetic feed and are
      already anonymized. Label them to try the workflow.</span></div>`;
    return;
  }
  const names = { ip: "IPs", user: "users", host: "hostnames", mac: "MACs", email: "emails", serial: "serials", ap: "AP names", fqdn: "domains" };
  const parts = Object.entries(a.replaced || {}).map(([k, v]) => `<span class="chip">${v} ${names[k] || k} (${a.distinct[k]} distinct)</span>`);
  $("lab-anon-note").innerHTML = `<div class="card small"><div class="row"><b>Anonymized.</b>${parts.join("") || '<span class="muted">nothing matched</span>'}</div>
    <p class="muted" style="margin:6px 0 0">The raw text was cleared from the box above and is not stored. Anonymization is
    pattern-based: scan the lines below for anything it missed before you share the file.</p></div>`;
}

$("lab-load").addEventListener("click", () =>
  busy($("lab-load"), async () => {
    const lines = $("lab-input").value.split(/\r?\n/).filter((l) => l.trim());
    if (!lines.length) return toast("Paste some log lines first", "error");
    if ($("lab-skip-anon").checked) {
      loadRows(lines.map((l) => l.trim()), { skipped: true });
    } else {
      const r = await postJSON("/api/anonymize", { lines });
      loadRows(r.lines, { replaced: r.replaced, distinct: r.distinct });
      toast(`Anonymized ${r.lines.length} lines`, "ok", 2500);
    }
    $("lab-input").value = ""; // don't leave raw logs on screen
  })
);
$("lab-demo").addEventListener("click", () =>
  busy($("lab-demo"), async () => {
    const f = await postJSON("/api/feed", { n: 20, seed: Math.floor(Math.random() * 1e6) });
    loadRows(f.events.map((e) => e.text), { demo: true });
  })
);
let resetArmed = null;
$("lab-reset").addEventListener("click", () => {
  if (!resetArmed) {
    $("lab-reset").textContent = "Click again to reset";
    resetArmed = setTimeout(() => { $("lab-reset").textContent = "Reset"; resetArmed = null; }, 3000);
    return;
  }
  clearTimeout(resetArmed);
  resetArmed = null;
  $("lab-reset").textContent = "Reset";
  Object.assign(lab, { rows: [], anon: null, focus: 0, evaluated: false });
  store.del("labelDraft");
  $("lab-results").innerHTML = "";
  renderAnonNote();
  renderLab();
});

const labeledJsonl = () =>
  lab.rows.filter((r) => r.label && r.label !== EXCLUDE).map((r) => JSON.stringify({ text: r.text, label: r.label })).join("\n") + "\n";

$("lab-download").addEventListener("click", () => {
  const day = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  download(`logsense-labeled-${day}.jsonl`, labeledJsonl(), "application/x-ndjson");
  const { labeled, excluded } = labCounts();
  toast(`Saved ${labeled} labeled lines${excluded ? ` (${excluded} excluded)` : ""}`, "ok");
});

async function runLabelEval() {
  if (!labCounts().labeled) return toast("Label at least one line first", "error");
  await busy([$("lab-evaluate"), $("lab-download")], async () => {
    const r = await evaluateBlob(new Blob([labeledJsonl()], { type: "application/x-ndjson" }), "labeled-in-browser.jsonl");
    const wrong = new Map(r.mistakes.map((m) => [m.text, m.predicted]));
    lab.rows.forEach((row) => (row.predicted = row.label && row.label !== EXCLUDE ? wrong.get(row.text) || row.label : null));
    lab.evaluated = true;
    saveLab();
    renderLab();
    const { excluded } = labCounts();
    $("lab-results").innerHTML = `<div class="card">${renderEvalResult(r)}
      ${excluded ? `<p class="muted small">${excluded} line(s) excluded as fitting no category: worth noting as a gap in the taxonomy.</p>` : ""}</div>`;
    $("lab-results").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}
$("lab-evaluate").addEventListener("click", runLabelEval);
primaryAction.label = () => (lab.rows.length ? runLabelEval() : $("lab-load").click());

loaders.label = () => { renderAnonNote(); renderLab(); };
