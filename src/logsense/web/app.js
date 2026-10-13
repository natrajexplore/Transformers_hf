// LogSense console — core: helpers, tabs, theme, toasts, drawer, shortcuts, and the
// Try it / Summarize / Performance / Evaluation / Models tabs. live.js and label.js build on it.
"use strict";

// ================================================================ helpers
const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const store = {
  // Browser storage can be blocked (private mode); never let that break the page.
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ } },
  del(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Backend not reachable. Is `python -m logsense.api` running?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error((Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d) || `${res.status} ${res.statusText}`);
  }
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function toast(message, kind = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// Run an async action with its buttons disabled; errors become a toast.
async function busy(buttons, fn) {
  const list = [buttons].flat().filter(Boolean);
  list.forEach((b) => (b.disabled = true));
  try {
    return await fn();
  } catch (err) {
    toast(err.message, "error", 7000);
  } finally {
    list.forEach((b) => (b.disabled = false));
  }
}

async function copyText(text, what = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${what} to clipboard`, "ok", 2000);
  } catch {
    toast("Clipboard not available in this browser", "error");
  }
}

function download(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function bar(value, max = 1, alt = false) {
  const w = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return `<div class="bar${alt ? " alt" : ""}"><span style="width:${w}%"></span></div>`;
}
const barRow = (label, value, max = 1, digits = 2, alt = false) =>
  `<div class="barrow"><span>${label}</span>${bar(value ?? 0, max, alt)}<span class="val">${fmt(value, digits)}</span></div>`;
const catBadge = (c, extra = "") => `<span class="badge cat-${esc(c)}">${esc(c)}${extra}</span>`;

function highlight(line, entities) {
  let out = "", pos = 0;
  for (const e of [...entities].sort((a, b) => a.start - b.start)) {
    if (e.start < pos) continue;
    out += esc(line.slice(pos, e.start));
    out += `<span class="ent ent-${esc(e.type)}" title="${esc(e.type)} · ${fmt(e.score, 2)}">${esc(line.slice(e.start, e.end))}<sup>${esc(e.type)}</sup></span>`;
    pos = e.end;
  }
  return out + esc(line.slice(pos));
}

function confusionTable(cm) {
  const max = Math.max(1, ...cm.matrix.flat());
  const head = cm.labels.map((l) => `<th>${esc(l)}</th>`).join("");
  const rows = cm.matrix
    .map((row, i) => {
      const cells = row
        .map((v, j) => {
          const a = v ? 0.15 + 0.85 * (v / max) : 0;
          const color = i === j ? `rgba(18,183,106,${a})` : `rgba(217,45,32,${a})`;
          return `<td style="background:${color}">${v || ""}</td>`;
        })
        .join("");
      return `<tr><th>${esc(cm.labels[i])}</th>${cells}</tr>`;
    })
    .join("");
  return `<table class="cm"><tr><th>true ↓ / predicted →</th>${head}</tr>${rows}</table>`;
}

// Shared by the Evaluation and Label tabs.
function renderEvalResult(r) {
  const perClass = Object.entries(r.per_class_f1).map(([k, v]) => barRow(catBadge(k), v)).join("");
  const mistakes = r.mistakes
    .map((x) => `<tr><td>${catBadge(x.label)}</td><td>${catBadge(x.predicted)}</td><td class="mono">${esc(x.text)}</td></tr>`)
    .join("");
  return `<p><b>${esc(r.filename || "labeled lines")}</b> · n=${r.n} · accuracy <b>${fmt(r.accuracy)}</b> ·
      macro-F1 <b>${fmt(r.macro_f1)}</b> <span class="muted small">(categories present in the file only)</span></p>
    <div class="grid2"><div>${perClass}</div><div>${confusionTable(r.confusion_matrix)}</div></div>
    ${mistakes
      ? `<details open><summary>${r.mistakes.length} mistake(s)${r.mistakes.length === 200 ? " (first 200)" : ""}</summary>
         <table><tr><th>your label</th><th>model said</th><th>line</th></tr>${mistakes}</table></details>`
      : `<p class="good">No mistakes.</p>`}`;
}

// ================================================================ config, health, theme
const CONFIG = { labels: [], entity_types: [], severity: {}, alerts: { window_s: 300, min_events: {} } };
const SEV_ORDER = ["info", "low", "medium", "high", "critical"];
const configReady = api("/api/config")
  .then((c) => Object.assign(CONFIG, c))
  .catch((e) => toast(e.message, "error", 8000));

async function pollHealth() {
  const el = $("health");
  try {
    const h = await api("/api/health");
    el.className = "health ok";
    el.lastElementChild.textContent = h.loaded.length ? `online · ${h.loaded.length} model(s) loaded` : "online";
  } catch {
    el.className = "health down";
    el.lastElementChild.textContent = "offline";
  }
}
pollHealth();
setInterval(pollHealth, 15000);

const THEMES = ["auto", "light", "dark"];
function applyTheme(t) {
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
  $("theme-btn").title = `Theme: ${t} (click to change)`;
  $("theme-btn").textContent = { auto: "◐", light: "☀", dark: "☾" }[t];
}
applyTheme(store.get("theme", "auto"));
$("theme-btn").addEventListener("click", () => {
  const next = THEMES[(THEMES.indexOf(store.get("theme", "auto")) + 1) % THEMES.length];
  store.set("theme", next);
  applyTheme(next);
});

// ================================================================ tabs
const loaders = {}; // tab name -> function run when the tab is shown
const primaryAction = {}; // tab name -> Ctrl+Enter action
const TAB_NAMES = [...document.querySelectorAll("#tabs button")].map((b) => b.dataset.tab);
let currentTab = "live";
function showTab(name) {
  if (!TAB_NAMES.includes(name)) name = "live";
  currentTab = name;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (loaders[name]) loaders[name]();
  history.replaceState(null, "", `#${name}`);
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll("[data-goto]").forEach((a) =>
  a.addEventListener("click", (e) => { e.preventDefault(); showTab(a.dataset.goto); })
);

// ================================================================ drawer (incident drill-down)
const drawer = {
  open(title, html) {
    $("drawer-title").innerHTML = title;
    $("drawer-body").innerHTML = html;
    $("drawer").classList.add("open");
    $("drawer").setAttribute("aria-hidden", "false");
    $("drawer-backdrop").hidden = false;
  },
  close() {
    $("drawer").classList.remove("open");
    $("drawer").setAttribute("aria-hidden", "true");
    $("drawer-backdrop").hidden = true;
  },
};
$("drawer-close").addEventListener("click", drawer.close);
$("drawer-backdrop").addEventListener("click", drawer.close);

/**
 * Open the incident view for one analyzed event.
 * ev: {line, category, confidence, entities, ts?, vendor?, label?}
 * related: other analyzed events of the same incident (for the LLM note), or [].
 */
function openIncident(ev, related = []) {
  const sev = CONFIG.severity[ev.category] || "info";
  const when = ev.ts ? new Date(ev.ts * 1000).toLocaleString() : "–";
  const ents = ev.entities.length
    ? `<table><tr><th>type</th><th>value</th><th class="num">score</th></tr>${ev.entities
        .map((e) => `<tr><td><span class="ent ent-${esc(e.type)}">${esc(e.type)}</span></td><td class="mono">${esc(e.text)}</td><td class="num">${fmt(e.score, 2)}</td></tr>`)
        .join("")}</table>`
    : `<p class="muted">No entities found.</p>`;
  const burst = related.length ? related : [ev];
  const relatedHtml = related.length > 1
    ? `<details><summary>${related.length} related ${esc(ev.category)} events in this incident window</summary>
        <table>${related.slice(-30).map((r) => `<tr><td class="mono">${esc(r.line)}</td></tr>`).join("")}</table></details>`
    : "";
  drawer.open(
    `<span class="sev sev-${sev}"></span> <span class="sevtext">${sev}</span> ${catBadge(ev.category)}`,
    `<dl class="kv"><dt>Time</dt><dd>${esc(when)}</dd>
       <dt>Vendor</dt><dd>${esc(ev.vendor || "–")}</dd>
       <dt>Confidence</dt><dd>${pct(ev.confidence)}</dd>
       ${ev.label ? `<dt>Ground truth</dt><dd>${catBadge(ev.label)} ${ev.label === ev.category ? '<span class="good">✓</span>' : '<span class="bad">✗ model mistake</span>'}</dd>` : ""}
     </dl>
     <div class="row"><b>Log line</b><span class="spacer"></span><button class="small-btn" data-copy-line>Copy</button></div>
     <div class="logline">${highlight(ev.line, ev.entities)}</div>
     <h3>Entities</h3>${ents}
     <h3 style="margin-top:14px">Similar past events</h3><div id="dr-similar"><div class="skeleton" style="height:60px"></div></div>
     <h3 style="margin-top:14px">Incident note</h3>
     ${relatedHtml}
     <p class="muted small">Summarizes ${burst.length} line(s) with the LoRA model and lists the classifier's own counts next to it.</p>
     <button id="dr-note" class="primary">Generate incident note</button>
     <div id="dr-note-out"></div>`
  );
  $("drawer-body").querySelector("[data-copy-line]").addEventListener("click", () => copyText(ev.line, "Log line copied"));
  postJSON("/api/analyze", { lines: [ev.line], k: 5 })
    .then((d) => {
      const sims = d.results[0].similar;
      $("dr-similar").innerHTML = `<table><tr><th class="num">cosine</th><th>category</th><th>past event</th></tr>${sims
        .map((s) => `<tr><td class="num">${fmt(s.score)}</td><td>${catBadge(s.category)}</td><td class="mono">${esc(s.text)}</td></tr>`)
        .join("")}</table>`;
    })
    .catch((e) => ($("dr-similar").innerHTML = `<p class="bad">${esc(e.message)}</p>`));
  $("dr-note").addEventListener("click", () =>
    busy($("dr-note"), async () => {
      $("dr-note-out").innerHTML = `<div class="skeleton" style="height:70px"></div><p class="muted small">Running the LLM… the first call loads it.</p>`;
      const lines = burst.map((r) => r.line).slice(-CONFIG.limits?.max_burst_lines || -30);
      const r = await postJSON("/api/summarize", { lines, mode: "lora" });
      $("dr-note-out").innerHTML = summaryCard(r, true);
      bindCopyButtons($("dr-note-out"));
    }).catch(() => {})
  );
}

function summaryCard(r, compact = false) {
  const checks = r.checks
    ? Object.entries(r.checks)
        .map(([k, v]) => `<span class="${v ? "good" : "bad"}">${v ? "✓" : "✗"} ${esc(k.replace("_ok", ""))}</span>`)
        .join(" · ")
    : compact ? "" : `<span class="muted small">No reference facts (pasted logs): compare with the classifier counts.</span>`;
  const counts = Object.entries(r.classifier_counts).map(([k, v]) => catBadge(k, ` ${v}`)).join(" ");
  return `<div class="card"><div class="card-title">${r.mode === "lora" ? "LoRA" : "Prompt-only"}
      <span class="muted small">${r.seconds} s</span><span class="spacer"></span>
      <button data-copy="${esc(r.summary)}">Copy</button></div>
    <pre class="note">${esc(r.summary)}</pre>
    <div>${checks}</div>
    <div class="row" style="margin-top:8px"><span class="muted small">Classifier counts:</span>${counts}</div></div>`;
}
function bindCopyButtons(root) {
  root.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => copyText(b.dataset.copy, "Note copied")));
}

// ================================================================ keyboard shortcuts
const typing = (el) => el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.tagName === "SELECT" || el.isContentEditable);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { drawer.close(); return; }
  if (e.altKey && /^[1-7]$/.test(e.key)) { e.preventDefault(); showTab(TAB_NAMES[Number(e.key) - 1]); return; }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && primaryAction[currentTab]) { e.preventDefault(); primaryAction[currentTab](); return; }
  if (typing(document.activeElement)) return;
  if (e.key === "?") { $("help-dialog").showModal(); }
});
$("help-btn").addEventListener("click", () => $("help-dialog").showModal());

// ================================================================ Try it
let lastTry = null;
$("try-samples").addEventListener("click", () =>
  busy($("try-samples"), async () => {
    const seed = Math.floor(Math.random() * 1e6);
    const d = await api(`/api/samples?split=${$("try-split").value}&n=5&seed=${seed}`);
    $("try-input").value = d.rows.map((r) => r.text).join("\n");
    $("try-input").dataset.gold = JSON.stringify(Object.fromEntries(d.rows.map((r) => [r.text, r.label])));
  })
);
$("try-input").addEventListener("input", () => delete $("try-input").dataset.gold);

async function runTry() {
  const lines = $("try-input").value.split("\n").filter((l) => l.trim());
  if (!lines.length) return toast("Paste some log lines first", "error");
  await busy($("try-run"), async () => {
    $("try-results").innerHTML = `<div class="skeleton"></div><div class="skeleton"></div>`;
    const d = await postJSON("/api/analyze", { lines, k: Number($("try-k").value) });
    lastTry = d;
    $("try-copy").disabled = false;
    const gold = JSON.parse($("try-input").dataset.gold || "{}");
    $("try-timing").innerHTML = Object.entries(d.timing_ms).map(([k, v]) => `<span class="chip">${k} ${v} ms</span>`).join("");
    const events = d.results.map((r) => ({ ...r, label: gold[r.line] }));
    $("try-results").innerHTML = events
      .map((r, i) => {
        const sev = CONFIG.severity[r.category] || "info";
        const verdict = r.label
          ? r.label === r.category ? `<span class="good">✓ matches label</span>` : `<span class="bad">✗ labeled ${esc(r.label)}</span>`
          : "";
        return `<div class="card result" data-i="${i}" title="Click for details">
          <div class="row"><span class="sev sev-${sev}"></span>${catBadge(r.category)}
            ${bar(r.confidence)}<span>${pct(r.confidence)}</span>${verdict}
            <span class="muted small">${r.entities.length} entities · ${r.similar.length} similar</span></div>
          <div class="logline">${highlight(r.line, r.entities)}</div></div>`;
      })
      .join("");
    $("try-results").querySelectorAll(".result").forEach((el) =>
      el.addEventListener("click", () => {
        const ev = events[Number(el.dataset.i)];
        openIncident(ev, events.filter((x) => x.category === ev.category));
      })
    );
  });
}
$("try-run").addEventListener("click", runTry);
primaryAction.try = runTry;
$("try-copy").addEventListener("click", () => lastTry && copyText(JSON.stringify(lastTry, null, 2), "Results JSON copied"));

// ================================================================ Summarize
let burstFacts = null;
$("sum-input").addEventListener("input", () => {
  burstFacts = null; // edited by hand: the generated facts no longer apply
  $("sum-reference").innerHTML = "";
});
$("sum-sample").addEventListener("click", () =>
  busy($("sum-sample"), async () => {
    const b = await api("/api/sample-burst");
    $("sum-input").value = b.lines.join("\n");
    burstFacts = b.facts;
    $("sum-reference").innerHTML = `<div class="card"><div class="card-title">Reference note
        <span class="muted small">computed from the generator's facts: ${b.facts.count} × ${esc(b.facts.label)}
        of ${b.facts.total} lines, ${b.facts.devices} device(s)</span></div><pre class="note">${esc(b.reference)}</pre></div>`;
    $("sum-results").innerHTML = "";
  })
);
async function runSummary(modes) {
  const lines = $("sum-input").value.split("\n").filter((l) => l.trim());
  if (!lines.length) return toast("Paste a burst or generate a sample first", "error");
  const buttons = ["sum-lora", "sum-prompt", "sum-both", "sum-sample"].map($);
  await busy(buttons, async () => {
    $("sum-results").innerHTML = "";
    try {
      for (const mode of modes) {
        $("sum-status").textContent = `Generating (${mode})… the first run loads the LLM`;
        $("sum-results").insertAdjacentHTML("beforeend", `<div class="skeleton" id="sum-wait"></div>`);
        const r = await postJSON("/api/summarize", { lines, mode, facts: burstFacts });
        $("sum-wait").remove();
        $("sum-results").insertAdjacentHTML("beforeend", summaryCard(r));
      }
      bindCopyButtons($("sum-results"));
    } finally {
      $("sum-status").textContent = "";
      $("sum-wait")?.remove();
    }
  });
}
$("sum-lora").addEventListener("click", () => runSummary(["lora"]));
$("sum-prompt").addEventListener("click", () => runSummary(["prompt"]));
$("sum-both").addEventListener("click", () => runSummary(["prompt", "lora"]));
primaryAction.summarize = () => runSummary(["lora"]);

// ================================================================ Performance
async function runBench() {
  const models = [...document.querySelectorAll(".bench-model:checked")].map((c) => c.value);
  if (!models.length) return toast("Pick at least one model", "error");
  await busy($("bench-run"), async () => {
    $("bench-status").textContent = "Running… NER is the slowest: it processes one line at a time.";
    const t0 = performance.now();
    try {
      const d = await postJSON("/api/benchmark", {
        split: $("bench-split").value,
        n_lines: Number($("bench-n").value),
        batch_size: Number($("bench-bs").value),
        models,
      });
      const maxRate = Math.max(...d.results.map((r) => r.lines_per_s || 0));
      const maxMs = Math.max(...d.results.map((r) => r.ms_per_line));
      const rows = d.results
        .map((r) => `<tr><td>${r.model}</td><td class="num">${r.lines}</td><td class="num">${r.ms_per_line}</td>
          <td class="num">${r.lines_per_s ?? "–"}</td><td class="num">${r.batch_ms.p50}</td><td class="num">${r.batch_ms.p95}</td>
          <td class="num">${r.batch_ms.max}</td><td class="num">${r.accuracy !== undefined ? `${fmt(r.accuracy)} / ${fmt(r.macro_f1)}` : ""}</td></tr>`)
        .join("");
      const stamp = new Date().toLocaleTimeString();
      $("bench-results").insertAdjacentHTML(
        "afterbegin",
        `<div class="card"><div class="card-title">${esc(d.split)} · batch ${d.batch_size}
          <span class="muted small">· ${stamp} · ${((performance.now() - t0) / 1000).toFixed(1)} s wall · process memory ${d.memory_mb ?? "?"} MB</span></div>
        <div class="grid2"><div><b class="small">Throughput: lines/s, higher is better</b>
          ${d.results.map((r) => barRow(r.model, r.lines_per_s, maxRate, 1)).join("")}</div>
          <div><b class="small">Latency: ms per line, lower is better</b>
          ${d.results.map((r) => barRow(r.model, r.ms_per_line, maxMs, 2, true)).join("")}</div></div>
        <table><tr><th>model</th><th class="num">lines</th><th class="num">ms/line</th><th class="num">lines/s</th>
          <th class="num">batch p50 ms</th><th class="num">batch p95 ms</th><th class="num">batch max ms</th>
          <th class="num">accuracy / macro-F1</th></tr>${rows}</table></div>`
      );
      toast("Benchmark finished", "ok", 2500);
    } finally {
      $("bench-status").textContent = "";
    }
  });
}
$("bench-run").addEventListener("click", runBench);
primaryAction.performance = runBench;

// ================================================================ Evaluation
function perClassBlock(title, series) {
  const labels = Object.keys(series[0].values);
  const rows = labels
    .map((l) => `<tr><td>${catBadge(l)}</td>${series
      .map((s) => `<td class="num">${fmt(s.values[l], 2)}</td><td style="min-width:80px">${bar(s.values[l] ?? 0)}</td>`)
      .join("")}</tr>`)
    .join("");
  const head = series.map((s) => `<th class="num">${esc(s.name)}</th><th></th>`).join("");
  return `<h3 style="margin-top:12px">${esc(title)}</h3><table><tr><th>class</th>${head}</tr>${rows}</table>`;
}

function renderMetrics(m) {
  const parts = [];
  const b = m.baseline, c = m.classifier;
  if (b || c) {
    const models = [
      ["Zero-shot (bart-large-mnli)", b && ((s) => b[s]?.zero_shot)],
      ["TF-IDF + LogReg", b && ((s) => b[s]?.tfidf_logreg)],
      ["DistilBERT fine-tuned", c && ((s) => c[s])],
    ].filter(([, get]) => get);
    const rows = models
      .map(([name, get]) => {
        const t = get("test"), u = get("test_unseen");
        return `<tr><td>${name}</td><td class="num">${fmt(t?.macro_f1)}</td><td class="num">${fmt(u?.macro_f1)}</td>
          <td style="min-width:140px">${bar(u?.macro_f1 ?? 0)}</td><td class="num">${u?.n ?? ""}</td></tr>`;
      })
      .join("");
    const series = models.map(([name, get]) => ({ name, values: get("test_unseen")?.per_class_f1 })).filter((s) => s.values);
    parts.push(`<div class="card"><div class="card-title">Classification · macro-F1</div>
      <table><tr><th>model</th><th class="num">test</th><th class="num">test_unseen</th><th></th><th class="num">n unseen</th></tr>${rows}</table>
      ${series.length ? perClassBlock("Per-class F1 on test_unseen", series) : ""}</div>`);
    if (c?.test_unseen?.confusion_matrix)
      parts.push(`<div class="card"><div class="card-title">DistilBERT confusion matrix · test_unseen</div>${confusionTable(c.test_unseen.confusion_matrix)}</div>`);
  }
  if (m.ner?.test_unseen) {
    const rows = Object.keys(m.ner.test_unseen)
      .map((t) => {
        const u = m.ner.test_unseen[t], s = m.ner.test?.[t];
        const name = t.includes("avg") ? `<i>${esc(t)}</i>` : `<span class="ent ent-${esc(t)}">${esc(t)}</span>`;
        return `<tr><td>${name}</td><td class="num">${fmt(s?.["f1-score"])}</td><td class="num">${fmt(u.precision)}</td>
          <td class="num">${fmt(u.recall)}</td><td class="num">${fmt(u["f1-score"])}</td>
          <td style="min-width:120px">${bar(u["f1-score"])}</td><td class="num">${u.support}</td></tr>`;
      })
      .join("");
    parts.push(`<div class="card"><div class="card-title">NER · entity-level (seqeval)</div>
      <table><tr><th>type</th><th class="num">test F1</th><th class="num">unseen P</th><th class="num">unseen R</th>
      <th class="num">unseen F1</th><th></th><th class="num">support</th></tr>${rows}</table></div>`);
  }
  if (m.search?.test_unseen) {
    const s = m.search.test_unseen;
    parts.push(`<div class="card"><div class="card-title">Similar-event search · top-1 neighbor has the same category</div>
      ${barRow("MiniLM + FAISS", s.embeddings?.top1_agreement)}${barRow("TF-IDF", s.tfidf?.top1_agreement, 1, 2, true)}
      ${perClassBlock("Per class (test_unseen)", [{ name: "MiniLM", values: s.embeddings.per_class }, { name: "TF-IDF", values: s.tfidf.per_class }])}</div>`);
  }
  if (m.summarizer?.prompt_only) {
    const p = m.summarizer.prompt_only, l = m.summarizer.lora;
    const rows = ["format_ok", "count_ok", "category_ok", "rougeL"]
      .map((k) => `<tr><td>${k}</td><td class="num">${fmt(p[k], 2)}</td><td>${bar(p[k], 1, true)}</td>
        <td class="num">${fmt(l?.[k], 2)}</td><td>${bar(l?.[k] ?? 0)}</td></tr>`)
      .join("");
    const ex = l?.examples?.[0];
    parts.push(`<div class="card"><div class="card-title">Burst summarizer · ${p.n} held-out bursts</div>
      <table><tr><th>check</th><th class="num">prompt-only</th><th></th><th class="num">LoRA</th><th></th></tr>${rows}
      <tr><td>seconds / burst</td><td class="num">${p.seconds_per_burst}</td><td></td><td class="num">${l?.seconds_per_burst ?? "–"}</td><td></td></tr></table>
      ${ex ? `<details><summary>Example: reference vs LoRA</summary><pre class="note">${esc(ex.reference)}</pre><pre class="note">${esc(ex.prediction)}</pre></details>` : ""}</div>`);
  }
  $("metrics").innerHTML = parts.join("") || `<p class="empty">No reports found. Train the models first.</p>`;
}
let metricsLoaded = false;
loaders.evaluation = () => {
  if (metricsLoaded) return;
  metricsLoaded = true;
  api("/api/metrics").then(renderMetrics).catch((e) => {
    metricsLoaded = false;
    $("metrics").innerHTML = `<p class="bad">${esc(e.message)}</p>`;
  });
};

async function evaluateBlob(blob, filename) {
  const form = new FormData();
  form.append("file", blob, filename);
  return api("/api/evaluate", { method: "POST", body: form });
}
async function runEvalUpload() {
  const file = $("eval-file").files[0];
  if (!file) return toast("Choose a .jsonl file first", "error");
  await busy($("eval-run"), async () => {
    $("eval-status").textContent = "Scoring…";
    try {
      $("eval-results").innerHTML = renderEvalResult(await evaluateBlob(file, file.name));
    } finally {
      $("eval-status").textContent = "";
    }
  });
}
$("eval-run").addEventListener("click", runEvalUpload);
primaryAction.evaluation = runEvalUpload;

// ================================================================ Models
loaders.models = () =>
  api("/api/models")
    .then((d) => {
      const rows = d.models
        .map((m) => `<tr><td><b>${m.name}</b></td><td>${esc(m.task)}</td><td><code>${esc(m.base_model)}</code></td>
          <td class="mono">${esc(m.path)}</td><td>${m.exists ? `<span class="good">yes</span>` : `<span class="bad">missing</span>`}</td>
          <td class="num">${m.local_size_mb ?? "–"}</td><td class="num">${m.first_call_s ?? '<span class="muted">not loaded</span>'}</td></tr>`)
        .join("");
      $("models-body").innerHTML = `<div class="card"><div class="card-title">Models <span class="muted small">· process memory ${d.memory_mb ?? "?"} MB</span></div>
        <table><tr><th>name</th><th>task</th><th>base model</th><th>local path</th><th>saved</th>
        <th class="num">local MB</th><th class="num">first call s (load + run)</th></tr>${rows}</table>
        <p class="hint" style="margin-top:8px">Search stores only the FAISS index locally; the MiniLM encoder and the Qwen base model
        come from the Hugging Face cache. The summarizer folder holds just the 4 MB LoRA adapter.</p></div>
        <div class="grid2"><div class="card"><div class="card-title">Categories &amp; severity</div><div class="chips">${d.labels
          .map((l) => `${catBadge(l)}<span class="sevtext muted">${esc(CONFIG.severity[l] || "")}</span>`).join(" ")}</div></div>
        <div class="card"><div class="card-title">Entity types</div><div class="chips">${d.entity_types
          .map((t) => `<span class="ent ent-${esc(t)}">${esc(t)}</span>`).join(" ")}</div></div></div>`;
    })
    .catch((e) => ($("models-body").innerHTML = `<p class="bad">${esc(e.message)}</p>`));

// ================================================================ start (after live.js / label.js register)
window.addEventListener("DOMContentLoaded", () => configReady.then(() => showTab((location.hash || "#live").slice(1))));
