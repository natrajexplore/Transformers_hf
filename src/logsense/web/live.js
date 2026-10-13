// LogSense console — Live tab: simulated syslog stream, triaged in real time, with a
// dashboard (counters, events/min, top talkers) and burst alerts. Uses helpers from app.js.
"use strict";

const LIVE_MAX = 800; // events kept in memory
const LIVE_RENDER = 300; // rows kept in the DOM
const VENDOR_NAME = { cisco_ios: "IOS", cisco_ise: "ISE", cisco_wlc: "C9800", panos: "PAN-OS" };
const HEADER_TS = /^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+/;
const HEADER_HOST = /^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+([^\s:]+)/;

const live = {
  running: false,
  seed: Math.floor(Math.random() * 1e6),
  t: null, // simulated clock (epoch s), advanced by the server
  incident: null, // open burst, passed back to the server
  events: [],
  alerts: [],
  nextId: 1,
  inflight: false,
  timer: null,
};

const sevRank = (c) => SEV_ORDER.indexOf(CONFIG.severity[c] || "info");
const hostOf = (line) => (line.match(HEADER_HOST) || [])[1] || "?";
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });

// ---------------------------------------------------------------- stream loop
async function tick() {
  live.timer = null;
  if (!live.running) return;
  const speed = Number($("live-speed").value);
  if (document.hidden || live.inflight) return schedule(speed);
  live.inflight = true;
  try {
    const feed = await postJSON("/api/feed", { n: Math.max(2, Math.round(4 * speed)), t: live.t, seed: live.seed, incident: live.incident });
    live.t = feed.t;
    live.incident = feed.incident;
    const d = await postJSON("/api/analyze", { lines: feed.events.map((e) => e.text), k: 0 });
    const fresh = feed.events.map((e, i) => ({
      id: live.nextId++,
      ts: e.ts,
      line: e.text,
      vendor: VENDOR_NAME[e.vendor] || e.vendor,
      label: e.label,
      category: d.results[i].category,
      confidence: d.results[i].confidence,
      entities: d.results[i].entities,
    }));
    live.events.push(...fresh);
    if (live.events.length > LIVE_MAX) live.events.splice(0, live.events.length - LIVE_MAX);
    fresh.forEach(checkAlerts);
    renderRows(fresh);
    renderDashboard();
  } catch (err) {
    toast(`Stream paused: ${err.message}`, "error", 8000);
    setRunning(false);
  } finally {
    live.inflight = false;
  }
  schedule(speed);
}
function schedule(speed) {
  if (live.running && !live.timer) live.timer = setTimeout(tick, 2000 / Math.max(speed, 0.5));
}
function setRunning(on) {
  live.running = on;
  $("live-toggle").textContent = on ? "❚❚ Pause" : live.events.length ? "▶ Resume" : "▶ Start stream";
  $("live-toggle").classList.toggle("primary", !on);
  $("live-hint").hidden = live.events.length > 0 || on;
  if (on) tick();
  else if (live.timer) { clearTimeout(live.timer); live.timer = null; }
}
$("live-toggle").addEventListener("click", () => setRunning(!live.running));
primaryAction.live = () => setRunning(!live.running);

// ---------------------------------------------------------------- feed rendering
function visible(ev) {
  if (sevRank(ev.category) < SEV_ORDER.indexOf($("live-minsev").value)) return false;
  const q = $("live-filter").value.trim().toLowerCase();
  if (!q) return true;
  return ev.line.toLowerCase().includes(q) || ev.category.includes(q) || ev.vendor.toLowerCase().includes(q);
}
function rowHtml(ev) {
  const sev = CONFIG.severity[ev.category] || "info";
  const cut = (ev.line.match(HEADER_TS) || [""])[0].length; // the time column already shows it
  const ents = ev.entities.filter((e) => e.start >= cut).map((e) => ({ ...e, start: e.start - cut, end: e.end - cut }));
  const wrong = $("live-truth").checked && ev.label !== ev.category;
  return `<div class="ev${wrong ? " wrong" : ""}" data-id="${ev.id}">
    <span class="time">${clock(ev.ts)}</span><span class="sev sev-${sev}" title="${sev}"></span>
    <span class="catcell">${catBadge(ev.category)}</span>
    <span class="msg"><span class="vendor">${esc(ev.vendor)}</span>${highlight(ev.line.slice(cut), ents)}${wrong ? `<span class="truth">truth: ${esc(ev.label)}</span>` : ""}</span></div>`;
}
function renderRows(fresh) {
  const feed = $("live-feed");
  const html = fresh.filter(visible).reverse().map(rowHtml).join("");
  if (!html) return updateCount();
  feed.querySelector(".empty")?.remove();
  const atTop = feed.scrollTop < 30;
  const before = feed.scrollHeight;
  feed.insertAdjacentHTML("afterbegin", html);
  feed.querySelectorAll(".ev:not(.seen)").forEach((el) => el.classList.add("new", "seen"));
  while (feed.children.length > LIVE_RENDER) feed.lastElementChild.remove();
  if (!atTop) feed.scrollTop += feed.scrollHeight - before; // keep the user's place in older rows
  updateCount();
}
function rerenderFeed() {
  const rows = live.events.filter(visible).slice(-LIVE_RENDER).reverse();
  $("live-feed").innerHTML = rows.length
    ? rows.map(rowHtml).join("")
    : `<div class="empty">${live.events.length ? "No events match the filter." : "Press <b>Start stream</b> (or <kbd>Space</kbd>) to begin."}</div>`;
  $("live-feed").querySelectorAll(".ev").forEach((el) => el.classList.add("seen"));
  updateCount();
}
function updateCount() {
  const shown = $("live-feed").querySelectorAll(".ev").length;
  $("live-count").textContent = live.events.length ? `${live.events.length} events · ${shown} shown` : "";
}
["live-minsev", "live-truth"].forEach((id) => $(id).addEventListener("change", rerenderFeed));
$("live-filter").addEventListener("input", rerenderFeed);

$("live-feed").addEventListener("click", (e) => {
  const row = e.target.closest(".ev");
  if (!row) return;
  const ev = live.events.find((x) => x.id === Number(row.dataset.id));
  if (ev) openIncident(ev, relatedTo(ev));
});
function relatedTo(ev) {
  const w = CONFIG.alerts.window_s;
  return live.events.filter((x) => x.category === ev.category && Math.abs(x.ts - ev.ts) <= w).slice(-30);
}

// ---------------------------------------------------------------- alerts
function checkAlerts(ev) {
  const need = CONFIG.alerts.min_events[ev.category];
  if (!need) return; // noise never alerts
  const w = CONFIG.alerts.window_s;
  const open = live.alerts.find((a) => a.category === ev.category && ev.ts - a.lastTs <= w);
  if (open) {
    open.lastTs = ev.ts;
    open.count += 1;
    open.hosts.add(hostOf(ev.line));
    open.ids.push(ev.id);
    return;
  }
  const recent = live.events.filter((x) => x.category === ev.category && ev.ts - x.ts <= w && x.ts <= ev.ts);
  if (recent.length < need) return;
  const alert = {
    id: live.nextId++,
    category: ev.category,
    firstTs: recent[0].ts,
    lastTs: ev.ts,
    count: recent.length,
    hosts: new Set(recent.map((x) => hostOf(x.line))),
    ids: recent.map((x) => x.id),
  };
  live.alerts.unshift(alert);
  const sev = CONFIG.severity[ev.category];
  if (sev === "critical" || sev === "high") toast(`Alert: ${recent.length} × ${ev.category} within ${Math.round(w / 60)} min`, "error", 5000);
}
function renderAlerts() {
  const list = live.alerts.slice(0, 8);
  $("alert-count").textContent = live.alerts.length;
  $("alert-count").classList.toggle("hot", live.alerts.some((a) => CONFIG.severity[a.category] === "critical"));
  $("alerts").innerHTML = list.length
    ? list
        .map((a) => {
          const sev = CONFIG.severity[a.category] || "info";
          const hosts = [...a.hosts];
          return `<div class="alert${sev === "critical" ? " sev-crit" : ""}" data-alert="${a.id}">
            <div class="row"><span class="sev sev-${sev}"></span>${catBadge(a.category)}<b>${a.count} events</b></div>
            <div class="when">${clock(a.firstTs)}–${clock(a.lastTs)} · ${hosts.length === 1 ? esc(hosts[0]) : `${hosts.length} devices`}</div></div>`;
        })
        .join("")
    : `<p class="muted small">No alerts yet. An alert opens when a category repeats within ${Math.round(CONFIG.alerts.window_s / 60)} min.</p>`;
}
$("alerts").addEventListener("click", (e) => {
  const el = e.target.closest("[data-alert]");
  if (!el) return;
  const a = live.alerts.find((x) => x.id === Number(el.dataset.alert));
  const evs = live.events.filter((x) => a.ids.includes(x.id));
  if (evs.length) openIncident(evs[evs.length - 1], evs);
  else toast("Those events have scrolled out of memory", "error");
});

// ---------------------------------------------------------------- dashboard
function countBy(list, key) {
  const m = new Map();
  for (const x of list) m.set(key(x), (m.get(key(x)) || 0) + 1);
  return m;
}
function renderDashboard() {
  const evs = live.events;
  const byCat = countBy(evs, (e) => e.category);
  const maxCat = Math.max(1, ...byCat.values());
  $("by-category").innerHTML = CONFIG.labels.map((l) => barRow(catBadge(l), byCat.get(l) || 0, maxCat, 0)).join("");
  const byVendor = countBy(evs, (e) => e.vendor);
  const maxV = Math.max(1, ...byVendor.values());
  $("by-vendor").innerHTML = [...byVendor.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([v, n]) => barRow(esc(v), n, maxV, 0, true))
    .join("") || `<p class="muted small">–</p>`;

  // Top talkers: entities seen in non-noise events (what the incidents are about).
  const talkers = {};
  for (const e of evs) {
    if (e.category === "noise") continue;
    for (const ent of e.entities) {
      if (!["IP", "USER", "INTERFACE"].includes(ent.type)) continue;
      (talkers[ent.type] ||= new Map()).set(ent.text, ((talkers[ent.type].get(ent.text)) || 0) + 1);
    }
  }
  $("talkers").innerHTML = ["IP", "USER", "INTERFACE"]
    .map((t) => {
      const top = [...(talkers[t] || new Map()).entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
      return `<div><h4>${t}</h4>${top.length ? top.map(([v, n]) => `<div class="t"><span title="${esc(v)}">${esc(v)}</span><span class="muted">${n}</span></div>`).join("") : `<div class="muted small">–</div>`}</div>`;
    })
    .join("");

  renderSpark();
  renderAlerts();
}
function renderSpark() {
  const svg = $("spark");
  if (!live.events.length) { svg.innerHTML = ""; $("rate-now").textContent = ""; return; }
  const lastMin = Math.floor(live.events[live.events.length - 1].ts / 60);
  const mins = 20;
  const all = new Array(mins).fill(0), inc = new Array(mins).fill(0);
  for (const e of live.events) {
    const i = Math.floor(e.ts / 60) - (lastMin - mins + 1);
    if (i < 0 || i >= mins) continue;
    all[i] += 1;
    if (e.category !== "noise") inc[i] += 1;
  }
  const max = Math.max(4, ...all);
  const pts = (arr) => arr.map((v, i) => `${(i / (mins - 1)) * 300},${58 - (v / max) * 54}`).join(" ");
  svg.innerHTML = `<polygon points="0,60 ${pts(all)} 300,60" fill="var(--bar)" opacity=".18"/>
    <polyline points="${pts(all)}" fill="none" stroke="var(--bar)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <polyline points="${pts(inc)}" fill="none" stroke="var(--s-critical)" stroke-width="1.5" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>`;
  $("rate-now").textContent = `${all[mins - 2] ?? 0}/min · dashed = non-noise (simulated time)`;
}

// ---------------------------------------------------------------- export / clear / keys
$("live-export").addEventListener("click", () => {
  const rows = live.events.filter(visible);
  if (!rows.length) return toast("Nothing to export yet", "error");
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const head = "time,severity,category,confidence,vendor,ground_truth,entities,line";
  const body = rows.map((e) =>
    [new Date(e.ts * 1000).toISOString(), CONFIG.severity[e.category], e.category, e.confidence, e.vendor, e.label,
      e.entities.map((x) => `${x.type}=${x.text}`).join("; "), e.line].map(q).join(",")
  );
  download(`logsense-live-${new Date().toISOString().slice(0, 19).replace(/:/g, "")}.csv`, [head, ...body].join("\n"), "text/csv");
});
$("live-clear").addEventListener("click", () => {
  Object.assign(live, { events: [], alerts: [], incident: null });
  rerenderFeed();
  renderDashboard();
  setRunning(live.running);
});
document.addEventListener("keydown", (e) => {
  if (currentTab !== "live" || typing(document.activeElement) || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === " ") { e.preventDefault(); setRunning(!live.running); }
  if (e.key === "/") { e.preventDefault(); $("live-filter").focus(); }
});

loaders.live = () => { if (!live.events.length) { rerenderFeed(); renderDashboard(); } };
