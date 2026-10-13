// LogSense console: calls the FastAPI endpoints in logsense/api.py. No framework, no build step.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

// Run an async action with its button disabled and errors shown in `errorEl`.
async function busy(button, errorEl, fn) {
  const buttons = Array.isArray(button) ? button : [button];
  buttons.forEach((b) => (b.disabled = true));
  errorEl.textContent = "";
  try {
    await fn();
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

function bar(value, max = 1, alt = false) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return `<div class="bar${alt ? " alt" : ""}"><span style="width:${pct}%"></span></div>`;
}
const barRow = (label, value, max = 1, digits = 2, alt = false) =>
  `<div class="barrow"><span>${esc(label)}</span>${bar(value ?? 0, max, alt)}<span class="val">${fmt(value, digits)}</span></div>`;

function highlight(line, entities) {
  let out = "", pos = 0;
  for (const e of [...entities].sort((a, b) => a.start - b.start)) {
    if (e.start < pos) continue; // overlapping spans: keep the first
    out += esc(line.slice(pos, e.start));
    out += `<span class="ent ent-${esc(e.type)}" title="${esc(e.type)} ${fmt(e.score, 2)}">${esc(line.slice(e.start, e.end))}<sup>${esc(e.type)}</sup></span>`;
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
          const color = i === j ? `rgba(6,118,71,${a})` : `rgba(180,35,24,${a})`;
          return `<td style="background:${color}">${v || ""}</td>`;
        })
        .join("");
      return `<tr><th>${esc(cm.labels[i])}</th>${cells}</tr>`;
    })
    .join("");
  return `<table class="cm"><tr><th>true ↓ / predicted →</th>${head}</tr>${rows}</table>`;
}

// ---------------------------------------------------------------- tabs
const loaders = {};
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (loaders[name]) loaders[name]();
  history.replaceState(null, "", `#${name}`);
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------------------------------------------------------- Try it
$("try-samples").addEventListener("click", () =>
  busy($("try-samples"), $("try-error"), async () => {
    const seed = Math.floor(Math.random() * 1e6);
    const data = await api(`/api/samples?split=${$("try-split").value}&n=5&seed=${seed}`);
    $("try-input").value = data.rows.map((r) => r.text).join("\n");
    $("try-input").dataset.gold = JSON.stringify(Object.fromEntries(data.rows.map((r) => [r.text, r.label])));
  })
);

$("try-run").addEventListener("click", () =>
  busy($("try-run"), $("try-error"), async () => {
    const lines = $("try-input").value.split("\n").filter((l) => l.trim());
    $("try-results").innerHTML = `<p class="muted">Analyzing ${lines.length} line(s)… (the first call loads the models)</p>`;
    const data = await postJSON("/api/analyze", { lines, k: Number($("try-k").value) });
    const gold = JSON.parse($("try-input").dataset.gold || "{}");
    $("try-timing").innerHTML = Object.entries(data.timing_ms)
      .map(([k, v]) => `<span class="chip">${k} ${v} ms</span>`)
      .join("");
    $("try-results").innerHTML = data.results
      .map((r) => {
        const truth = gold[r.line];
        const verdict = truth
          ? truth === r.category
            ? `<span class="good">✓ matches label</span>`
            : `<span class="bad">✗ labeled ${esc(truth)}</span>`
          : "";
        const similar = r.similar
          .map((s) => `<tr><td class="num">${fmt(s.score)}</td><td>${esc(s.category)}</td><td class="mono">${esc(s.text)}</td></tr>`)
          .join("");
        return `<div class="card">
          <div class="row"><span class="badge">${esc(r.category)}</span>
            ${bar(r.confidence)}<span>${(r.confidence * 100).toFixed(1)}%</span>${verdict}</div>
          <div class="logline">${highlight(r.line, r.entities)}</div>
          <details><summary>${r.entities.length} entities · ${r.similar.length} similar past events</summary>
            <table><tr><th class="num">cosine</th><th>category</th><th>past event</th></tr>${similar}</table>
          </details></div>`;
      })
      .join("");
  })
);

// ---------------------------------------------------------------- Summarize
let burstFacts = null;
$("sum-input").addEventListener("input", () => {
  burstFacts = null; // edited by hand: the generated facts no longer apply
  $("sum-reference").innerHTML = "";
});

$("sum-sample").addEventListener("click", () =>
  busy($("sum-sample"), $("sum-error"), async () => {
    const b = await api("/api/sample-burst");
    $("sum-input").value = b.lines.join("\n");
    burstFacts = b.facts;
    $("sum-reference").innerHTML = `<div class="card"><h3>Reference note (computed from the generator's facts)</h3>
      <pre class="note">${esc(b.reference)}</pre></div>`;
    $("sum-results").innerHTML = "";
  })
);

function summaryCard(r) {
  const checks = r.checks
    ? Object.entries(r.checks)
        .map(([k, v]) => `<span class="${v ? "good" : "bad"}">${v ? "✓" : "✗"} ${esc(k.replace("_ok", ""))}</span>`)
        .join(" · ")
    : `<span class="muted">no reference facts (pasted logs)</span>`;
  const counts = Object.entries(r.classifier_counts)
    .map(([k, v]) => `<span class="chip">${esc(k)} ${v}</span>`)
    .join("");
  return `<div class="card"><h3>${r.mode === "lora" ? "LoRA" : "Prompt-only"} <small class="muted">${r.seconds} s</small></h3>
    <pre class="note">${esc(r.summary)}</pre>
    <div>${checks}</div>
    <div class="row" style="margin-top:6px"><span class="muted">classifier counts:</span><span class="chips">${counts}</span></div></div>`;
}

async function runSummary(modes) {
  const lines = $("sum-input").value.split("\n").filter((l) => l.trim());
  $("sum-results").innerHTML = "";
  for (const mode of modes) {
    $("sum-status").textContent = `Generating (${mode})… first run loads the LLM`;
    const r = await postJSON("/api/summarize", { lines, mode, facts: burstFacts });
    $("sum-results").insertAdjacentHTML("beforeend", summaryCard(r));
  }
  $("sum-status").textContent = "";
}
const sumButtons = ["sum-lora", "sum-prompt", "sum-both", "sum-sample"].map($);
const sumAction = (modes) => () =>
  busy(sumButtons, $("sum-error"), () => runSummary(modes).finally(() => ($("sum-status").textContent = "")));
$("sum-lora").addEventListener("click", sumAction(["lora"]));
$("sum-prompt").addEventListener("click", sumAction(["prompt"]));
$("sum-both").addEventListener("click", sumAction(["prompt", "lora"]));

// ---------------------------------------------------------------- Performance
$("bench-run").addEventListener("click", () =>
  busy($("bench-run"), $("bench-error"), async () => {
    const models = [...document.querySelectorAll(".bench-model:checked")].map((c) => c.value);
    if (!models.length) throw new Error("Pick at least one model");
    $("bench-status").textContent = "Running… (NER is the slowest: it runs one line at a time)";
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
        .map(
          (r) => `<tr><td>${r.model}</td><td class="num">${r.lines}</td><td class="num">${r.ms_per_line}</td>
          <td class="num">${r.lines_per_s ?? "–"}</td><td class="num">${r.batch_ms.p50}</td><td class="num">${r.batch_ms.p95}</td>
          <td class="num">${r.batch_ms.max}</td><td class="num">${r.accuracy !== undefined ? `${fmt(r.accuracy)} / ${fmt(r.macro_f1)}` : ""}</td></tr>`
        )
        .join("");
      const run = `<div class="card"><h3>${esc(d.split)} · batch ${d.batch_size}
          <small class="muted">· ${((performance.now() - t0) / 1000).toFixed(1)} s wall · process memory ${d.memory_mb ?? "?"} MB</small></h3>
        <div class="grid2"><div><b>Throughput (lines/s, higher is better)</b>
          ${d.results.map((r) => barRow(r.model, r.lines_per_s, maxRate, 1)).join("")}</div>
          <div><b>Latency (ms per line, lower is better)</b>
          ${d.results.map((r) => barRow(r.model, r.ms_per_line, maxMs, 2, true)).join("")}</div></div>
        <table><tr><th>model</th><th class="num">lines</th><th class="num">ms/line</th><th class="num">lines/s</th>
          <th class="num">batch p50 ms</th><th class="num">batch p95 ms</th><th class="num">batch max ms</th>
          <th class="num">accuracy / macro-F1</th></tr>${rows}</table></div>`;
      $("bench-results").insertAdjacentHTML("afterbegin", run); // newest run on top, keep history
    } finally {
      $("bench-status").textContent = "";
    }
  })
);

// ---------------------------------------------------------------- Evaluation
function perClassBlock(title, series) {
  // series: [{name, values: {label: f1}}]
  const labels = Object.keys(series[0].values);
  const rows = labels
    .map(
      (l) => `<tr><td>${esc(l)}</td>${series
        .map((s) => `<td class="num">${fmt(s.values[l], 2)}</td><td style="min-width:90px">${bar(s.values[l] ?? 0)}</td>`)
        .join("")}</tr>`
    )
    .join("");
  const head = series.map((s) => `<th class="num">${esc(s.name)}</th><th></th>`).join("");
  return `<h3>${esc(title)}</h3><table><tr><th>class</th>${head}</tr>${rows}</table>`;
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
    const series = models
      .map(([name, get]) => ({ name, values: get("test_unseen")?.per_class_f1 }))
      .filter((s) => s.values);
    parts.push(`<div class="card"><h3>Classification · macro-F1</h3>
      <table><tr><th>model</th><th class="num">test</th><th class="num">test_unseen</th><th></th><th class="num">n unseen</th></tr>${rows}</table>
      ${series.length ? perClassBlock("Per-class F1 on test_unseen", series) : ""}</div>`);
    if (c?.test_unseen?.confusion_matrix)
      parts.push(`<div class="card"><h3>DistilBERT confusion matrix · test_unseen</h3>${confusionTable(c.test_unseen.confusion_matrix)}</div>`);
  }
  if (m.ner) {
    const types = Object.keys(m.ner.test_unseen || {});
    const rows = types
      .map((t) => {
        const u = m.ner.test_unseen[t], s = m.ner.test?.[t];
        return `<tr><td>${esc(t)}</td><td class="num">${fmt(s?.["f1-score"])}</td><td class="num">${fmt(u["precision"])}</td>
          <td class="num">${fmt(u["recall"])}</td><td class="num">${fmt(u["f1-score"])}</td>
          <td style="min-width:120px">${bar(u["f1-score"])}</td><td class="num">${u.support}</td></tr>`;
      })
      .join("");
    parts.push(`<div class="card"><h3>NER · entity-level (seqeval)</h3>
      <table><tr><th>type</th><th class="num">test F1</th><th class="num">unseen P</th><th class="num">unseen R</th>
      <th class="num">unseen F1</th><th></th><th class="num">support</th></tr>${rows}</table></div>`);
  }
  if (m.search?.test_unseen) {
    const s = m.search.test_unseen;
    parts.push(`<div class="card"><h3>Similar-event search · top-1 neighbor has the same category</h3>
      ${barRow("MiniLM + FAISS", s.embeddings?.top1_agreement)}${barRow("TF-IDF", s.tfidf?.top1_agreement, 1, 2, true)}
      ${perClassBlock("Per class (test_unseen)", [
        { name: "MiniLM", values: s.embeddings.per_class },
        { name: "TF-IDF", values: s.tfidf.per_class },
      ])}</div>`);
  }
  if (m.summarizer?.prompt_only) {
    const p = m.summarizer.prompt_only, l = m.summarizer.lora;
    const keys = ["format_ok", "count_ok", "category_ok", "rougeL"];
    const rows = keys
      .map((k) => `<tr><td>${k}</td><td class="num">${fmt(p[k], 2)}</td><td>${bar(p[k], 1, true)}</td>
        <td class="num">${fmt(l?.[k], 2)}</td><td>${bar(l?.[k] ?? 0)}</td></tr>`)
      .join("");
    parts.push(`<div class="card"><h3>Burst summarizer · ${m.summarizer.prompt_only.n} held-out bursts</h3>
      <table><tr><th>check</th><th class="num">prompt-only</th><th></th><th class="num">LoRA</th><th></th></tr>${rows}
      <tr><td>seconds / burst</td><td class="num">${p.seconds_per_burst}</td><td></td><td class="num">${l?.seconds_per_burst ?? "–"}</td><td></td></tr></table></div>`);
  }
  $("metrics").innerHTML = parts.join("") || `<p class="muted">No reports found. Train the models first.</p>`;
}
let metricsLoaded = false;
loaders.evaluation = () => {
  if (metricsLoaded) return;
  metricsLoaded = true;
  api("/api/metrics").then(renderMetrics).catch((e) => ($("metrics").textContent = e.message));
};

$("eval-run").addEventListener("click", () =>
  busy($("eval-run"), $("eval-error"), async () => {
    const file = $("eval-file").files[0];
    if (!file) throw new Error("Choose a .jsonl file first");
    const form = new FormData();
    form.append("file", file);
    $("eval-status").textContent = "Scoring…";
    try {
      const r = await api("/api/evaluate", { method: "POST", body: form });
      const perClass = Object.entries(r.per_class_f1).map(([k, v]) => barRow(k, v)).join("");
      const mistakes = r.mistakes
        .map((x) => `<tr><td>${esc(x.label)}</td><td class="bad">${esc(x.predicted)}</td><td class="mono">${esc(x.text)}</td></tr>`)
        .join("");
      $("eval-results").innerHTML = `<p><b>${esc(r.filename)}</b> · n=${r.n} · accuracy <b>${fmt(r.accuracy)}</b> ·
          macro-F1 <b>${fmt(r.macro_f1)}</b></p>
        <div class="grid2"><div>${perClass}</div><div>${confusionTable(r.confusion_matrix)}</div></div>
        ${mistakes ? `<details><summary>${r.mistakes.length} mistakes${r.mistakes.length === 200 ? " (first 200)" : ""}</summary>
          <table><tr><th>label</th><th>predicted</th><th>text</th></tr>${mistakes}</table></details>` : `<p class="good">No mistakes.</p>`}`;
    } finally {
      $("eval-status").textContent = "";
    }
  })
);

// ---------------------------------------------------------------- Models
loaders.models = () =>
  api("/api/models")
    .then((d) => {
      const rows = d.models
        .map(
          (m) => `<tr><td><b>${m.name}</b></td><td>${esc(m.task)}</td><td><code>${esc(m.base_model)}</code></td>
          <td class="mono">${esc(m.path)}</td><td>${m.exists ? `<span class="good">yes</span>` : `<span class="bad">missing</span>`}</td>
          <td class="num">${m.local_size_mb ?? "–"}</td><td class="num">${m.first_call_s ?? "not loaded"}</td></tr>`
        )
        .join("");
      $("models-body").innerHTML = `<div class="card"><h3>Models <small class="muted">· process memory ${d.memory_mb ?? "?"} MB</small></h3>
        <table><tr><th>name</th><th>task</th><th>base model</th><th>local path</th><th>saved</th>
        <th class="num">local MB</th><th class="num">first call s (load + run)</th></tr>${rows}</table>
        <p class="hint">Search stores only the FAISS index locally; the MiniLM encoder and the Qwen base model
        come from the Hugging Face cache. The summarizer folder holds just the LoRA adapter.</p></div>
        <div class="grid2"><div class="card"><h3>Categories</h3><div class="chips">${d.labels.map((l) => `<span class="chip">${esc(l)}</span>`).join("")}</div></div>
        <div class="card"><h3>Entity types</h3><div class="chips">${d.entity_types
          .map((t) => `<span class="ent ent-${esc(t)}">${esc(t)}</span>`)
          .join(" ")}</div></div></div>`;
    })
    .catch((e) => ($("models-body").textContent = e.message));

// ---------------------------------------------------------------- start
const startTab = (location.hash || "#try").slice(1);
showTab(document.querySelector(`.panel#${CSS.escape(startTab)}`) ? startTab : "try");
