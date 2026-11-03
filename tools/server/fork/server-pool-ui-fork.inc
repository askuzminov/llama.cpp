// fork: the dashboard of --pool, GET /pool: a page that reads /pool/status every second (server-pool-fork.cpp).
// self-contained, the nodes may have no internet. in parts: MSVC takes a string literal of up to 16 KB

static const char * const POOL_UI_PARTS[] = {
R"POOLUI(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama pool</title>
<style>
:root {
  --bg: #f5f6f8; --panel: #ffffff; --text: #1c2027; --muted: #687180; --line: #e2e5ea; --soft: #eef0f3;
  --accent: #2f6fdb; --ok: #1d9a52; --warn: #c98200; --bad: #d14343; --gen: #2f6fdb; --prompt: #8b4fd6; --idle: #c6ccd5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101317; --panel: #191d23; --text: #e5e8ed; --muted: #98a1ad; --line: #2a3038; --soft: #222831;
    --accent: #5d90f0; --ok: #3cbd74; --warn: #e2a43c; --bad: #ee6b6b; --gen: #5d90f0; --prompt: #a97cef; --idle: #3a414b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }
header { display: flex; align-items: baseline; gap: 12px; padding: 12px 20px; border-bottom: 1px solid var(--line);
  background: var(--panel); position: sticky; top: 0; z-index: 2; flex-wrap: wrap; }
header h1 { font-size: 16px; margin: 0; letter-spacing: .01em; }
header .sub { color: var(--muted); font-size: 13px; }
header .live { margin-left: auto; font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); display: inline-block; }
.dot.off { background: var(--bad); }
main { padding: 16px 20px 40px; max-width: 1560px; margin: 0 auto; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 11px 14px; min-width: 0; }
.tile .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
.tile .v { font-size: 24px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tile .v small { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 4px; }
.tile canvas { width: 100%; height: 30px; display: block; margin-top: 6px; }
.nodes { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; margin-bottom: 16px; }
@media (max-width: 520px) { .nodes { grid-template-columns: 1fr; } main { padding: 12px 16px 32px; } }
.node { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.node.off { opacity: .72; }
.nh { display: flex; align-items: center; gap: 8px; }
.nh .name { font-weight: 600; font-size: 15px; }
.nh .name small { font-weight: 500; font-size: 11px; color: var(--muted); margin-left: 4px; }
.nh .act { margin-left: auto; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: var(--soft); color: var(--muted); white-space: nowrap; }
.badge.up { background: color-mix(in srgb, var(--ok) 16%, transparent); color: var(--ok); }
.badge.down { background: color-mix(in srgb, var(--bad) 16%, transparent); color: var(--bad); }
.badge.loading, .badge.draining, .badge.unloaded { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.badge.mismatch { background: color-mix(in srgb, var(--bad) 16%, transparent); color: var(--bad); }
.lnk { display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; align-items: center; gap: 8px; margin: 3px 0; font-size: 12px; }
.lnk .u { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
.lnk .t { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.lnk .e { grid-column: 2 / 4; color: var(--bad); overflow-wrap: anywhere; }
.ld { width: 8px; height: 8px; border-radius: 50%; background: var(--idle); }
.ld.on { background: var(--ok); } .ld.err { background: var(--bad); }
button { font: inherit; font-size: 12px; padding: 3px 10px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--panel); color: var(--text); cursor: pointer; }
button:hover { border-color: var(--accent); }
.meta { color: var(--muted); font-size: 12px; margin: 4px 0 10px; overflow-wrap: anywhere; }
.sec { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin: 10px 0 4px; }
.slot { display: grid; grid-template-columns: 20px minmax(60px, 1fr) minmax(150px, auto); align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }
.slot .id { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.slot .t { font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
.bar { height: 9px; border-radius: 5px; background: var(--soft); overflow: hidden; position: relative; }
.bar i { display: block; height: 100%; }
.st-idle i { background: var(--idle); } .st-gen i { background: var(--gen); } .st-prompt i { background: var(--prompt); }
.st-prompt .t, .st-gen .t { color: var(--text); } .st-idle .t { color: var(--muted); }
.mem { display: grid; grid-template-columns: minmax(80px, 1fr) minmax(140px, auto); align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }
.mem .t { font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; color: var(--muted); }
.mem .bar i { background: var(--accent); }
.mem .bar.hi i { background: var(--warn); }
.nf { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-top: 10px;
  border-top: 1px solid var(--line); padding-top: 8px; }
.nf b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
.run { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font-size: 11px; padding: 1px 7px; border-radius: 999px; background: var(--soft); color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.errline { color: var(--bad); font-size: 12px; margin-top: 6px; overflow-wrap: anywhere; }
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
@media (max-width: 1000px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; overflow-x: auto; }
.panel h2 { font-size: 11px; margin: 0 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; color: var(--muted); font-weight: 500; padding: 4px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 4px 6px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td.d { white-space: normal; color: var(--muted); min-width: 220px; }
td.n { text-align: right; } th.n { text-align: right; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.k { font-weight: 600; }
.k-keep, .k-migrate { color: var(--ok); } .k-new { color: var(--accent); }
.k-move, .k-gone, .k-drain, .k-loading, .k-link { color: var(--warn); }
.k-error, .k-fail, .k-down, .k-mismatch { color: var(--bad); } .k-done, .k-undrain, .k-up, .k-match { color: var(--muted); }
.act-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--ok); margin-right: 5px; }
.idle-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--idle); margin-right: 5px; }
.empty { color: var(--muted); font-size: 12px; padding: 4px 2px; }
.legend { display: flex; gap: 12px; color: var(--muted); font-size: 11px; margin-left: 8px; }
.legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.legend .lg::before { background: var(--gen); } .legend .lp::before { background: var(--prompt); } .legend .li::before { background: var(--idle); }
</style>
</head>
)POOLUI",
R"POOLUI(<body>
<header>
  <h1>llama pool</h1>
  <span class="sub" id="pools"></span>
  <span class="sub" id="sub"></span>
  <span class="legend"><span class="lg">generating</span><span class="lp">prompt</span><span class="li">idle</span></span>
  <span class="live"><span class="dot" id="dot"></span><span id="live">connecting</span></span>
</header>
<main>
  <div class="tiles">
    <div class="tile"><div class="k">Nodes up</div><div class="v" id="t-nodes">-</div></div>
    <div class="tile"><div class="k">Slots busy</div><div class="v" id="t-slots">-</div></div>
    <div class="tile"><div class="k">Queue</div><div class="v" id="t-queue">-</div></div>
    <div class="tile"><div class="k">Generation</div><div class="v" id="t-gen">-</div><canvas id="c-gen"></canvas></div>
    <div class="tile"><div class="k">Prompt</div><div class="v" id="t-pp">-</div><canvas id="c-pp"></canvas></div>
    <div class="tile"><div class="k">Conversations</div><div class="v" id="t-conv">-</div></div>
  </div>
  <div class="nodes" id="nodes"></div>
  <div class="grid2">
    <div class="panel"><h2>Queue</h2><div id="queue"></div></div>
    <div class="panel"><h2>Conversations</h2><div id="convs"></div></div>
  </div>
  <div class="panel"><h2>Decisions</h2><div id="events"></div></div>
</main>
<script>
"use strict";
const HMAX = 300;
const hist = { gen: [], pp: [] };
const base = location.pathname.replace(/\/pool\/?$/, "");
const query = new URLSearchParams(location.search);
const key = query.get("key");
const model = query.get("model");
const H = key ? { "Authorization": "Bearer " + key } : {};

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function tok(n) {
  n = Number(n) || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(0) + "K";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}
function tps(x) { x = Number(x) || 0; return x >= 100 ? x.toFixed(0) : x.toFixed(1); }
function gib(b) { return ((Number(b) || 0) / 1073741824).toFixed(1); }
function age(ms) {
  const s = (Number(ms) || 0) / 1000;
  if (s < 1) return "now";
  if (s < 60) return s.toFixed(0) + " s";
  if (s < 3600) return (s / 60).toFixed(0) + " min";
  return (s / 3600).toFixed(1) + " h";
}
function clock(t) {
  const d = new Date(t);
  return d.toTimeString().slice(0, 8);
}
function cssv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function spark(id, data, color) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const max = Math.max(1, ...data);
  const step = (w - 2) / (HMAX - 1);
  const x0 = 1 + (HMAX - data.length) * step;
  c.beginPath();
  data.forEach((v, i) => {
    const x = x0 + i * step, y = h - 1 - (v / max) * (h - 3);
    if (i) c.lineTo(x, y); else c.moveTo(x, y);
  });
  c.lineWidth = 1.5; c.strokeStyle = color; c.stroke();
  c.lineTo(x0 + (data.length - 1) * step, h); c.lineTo(x0, h); c.closePath();
  c.globalAlpha = 0.12; c.fillStyle = color; c.fill(); c.globalAlpha = 1;
}

function slotRow(s) {
  const used = (s.n_prompt || 0) + (s.n_decoded || 0);
  const frac = s.n_ctx > 0 ? Math.min(1, used / s.n_ctx) : 0;
  let text;
  if (s.state === "gen") {
    text = "gen " + tps(s.gen_tps) + " t/s \u00b7 " + tok(used);
  } else if (s.state === "prompt") {
    const done = (s.n_processed || 0) + (s.n_cached || 0);
    text = "prompt " + tps(s.prompt_tps) + " t/s \u00b7 " + tok(done) + " / " + tok(s.n_prompt);
  } else {
    text = "idle" + (used > 0 ? " \u00b7 " + tok(used) : "");
  }
  return '<div class="slot st-' + esc(s.state) + '"><span class="id">' + esc(s.id) + '</span>' +
    '<div class="bar"><i style="width:' + (frac * 100).toFixed(1) + '%"></i></div>' +
    '<span class="t">' + esc(text) + '</span></div>';
}

function memRow(d) {
  const total = Number(d.total) || 0, free = Number(d.free) || 0;
  if (total <= 0) return "";
  const used = Math.max(0, total - free), frac = used / total;
  return '<div class="mem"><div><div class="sec" style="margin:0 0 2px;text-transform:none;letter-spacing:0">' +
    esc(d.name) + ' <span style="opacity:.75">' + esc(d.description) + '</span></div>' +
    '<div class="bar' + (frac > 0.92 ? " hi" : "") + '"><i style="width:' + (frac * 100).toFixed(1) + '%"></i></div></div>' +
    '<span class="t">' + gib(used) + " / " + gib(total) + " GiB</span></div>";
}
)POOLUI",
R"POOLUI(
function linkRow(l) {
  const mb = x => x > 0 ? (x / 8).toFixed(0) : "-";
  const sp = l.mbps_read > 0 || l.mbps_write > 0 ? "from " + mb(l.mbps_read) + " / to " + mb(l.mbps_write) + " MB/s measured" :
    "not measured yet";
  return '<div class="lnk"><span class="ld' + (l.error ? " err" : (l.active ? " on" : "")) + '"></span>' +
    '<span class="u">' + esc(l.url) + (l.active ? " \u00b7 in use" : "") + '</span><span class="t">' + esc(sp) + '</span>' +
    (l.error ? '<span class="e">' + esc(l.error) + '</span>' : "") + '</div>';
}

function nodeCard(n) {
  const state = n.mismatch ? "mismatch" : (n.draining ? "draining" : n.status);
  const off = n.status !== "up" || n.draining || n.mismatch;
  const mi = n.model_info || {};
  const meta = [n.url + (n.fallback ? " (fallback)" : ""), n.model || "", mi.ftype || "",
    n.n_ctx ? "ctx " + tok(n.n_ctx) : "", n.slots ? n.slots + " slots" : "",
    n.weight !== 1 ? "weight " + n.weight.toFixed(2) : "", n.signature ? "cache " + n.signature.slice(0, 8) : ""
  ].filter(Boolean).join(" \u00b7 ");
  let html = '<div class="node' + (off ? " off" : "") + '">' +
    '<div class="nh"><span class="name">' + esc(n.name) + (n.self ? ' <small>this router</small>' : '') + '</span><span class="badge ' + esc(state) + '">' + esc(state) + '</span>' +
    '<span class="act"><button data-node="' + esc(n.name) + '" data-drain="' + (n.draining ? "0" : "1") + '">' +
    (n.draining ? "Resume" : "Drain") + '</button></span></div>' +
    '<div class="meta">' + esc(meta) + '</div>';
  if (n.slot_list && n.slot_list.length) {
    html += '<div class="sec">Slots</div>' + n.slot_list.map(slotRow).join("");
  }
  if (n.links && n.links.length) html += '<div class="sec">Links</div>' + n.links.map(linkRow).join("");
  const devs = (n.devices || []).map(memRow).join("");
  if (devs) html += '<div class="sec">Memory</div>' + devs;
  const gpus = (n.gpus || []).map(g => '<span>' + esc(g.name) + ' <b>' + (g.temp_c != null ? g.temp_c.toFixed(0) + " \u00b0C" : "-") + '</b>' +
    (g.power_pct != null ? " \u00b7 power " + g.power_pct.toFixed(0) + "%" : "") + (g.power_w != null ? " \u00b7 " + g.power_w.toFixed(0) + " W" : "") +
    (g.fan_rpm != null ? " \u00b7 fan " + g.fan_rpm + " rpm" : "") + '</span>').join("");
  if (gpus) html += '<div class="sec">GPU</div><div class="nf">' + gpus + '</div>';
  const st = n.stats || {};
  if (st.uptime_s > 0) {
    const avg = (k, s) => st[s] > 0 ? " (" + tps(st[k] / st[s]) + " t/s)" : "";
    html += '<div class="sec">Work since the model loaded, ' + age(st.uptime_s * 1000) + ' ago</div><div class="nf">' +
      '<span title="prompt tokens it computed, with their average speed">prompt <b>' + tok(st.prompt_tokens) + '</b> tok' + avg("prompt_tokens", "prompt_s") + '</span>' +
      '<span title="tokens it generated, with the average speed of a slot">generated <b>' + tok(st.gen_tokens) + '</b> tok' + avg("gen_tokens", "gen_s") + '</span>' +
      '<span title="prompt tokens it took from its cache instead of computing them">from cache <b>' + tok(st.cached_tokens) + '</b> tok</span></div>';
  }
  html += '<div class="nf">' +
    '<span>in flight <b>' + n.inflight + " / " + n.slots + '</b></span>' +
    (n.external ? '<span title="requests of other routers of the pool and of clients that went to the node itself">other routers <b>' + n.external + '</b></span>' : "") +
    '<span>waiting for it <b>' + n.queued_for + '</b></span>' +
    (n.room ? '<span title="the cells of its unified KV cache that the running and paused slots hold with their answer reserve: a prompt that does not fit next to them waits in the pool for a node with room">' +
      'KV held <b>' + tok(n.room.held) + " / " + tok(n.room.n_ctx) + '</b>' + (n.room.deferred ? " (" + n.room.deferred + " waiting inside)" : "") + '</span>' : "") +
    '<span title="tokens its slots generate per second now (all of them, between two polls)">gen now <b>' + tps(n.gen_tps) + '</b> t/s</span>' +
    '<span title="prompt tokens its slots compute per second now (all of them, between two polls)">prompt now <b>' + tps(n.prompt_tps) + '</b> t/s</span>' +
    '<span title="the prefill speed of one request the decisions use (what computing a conversation there costs): the average of its prompts of move_tokens and more, else a default">' +
    'prefill <b>' + tps(n.prefill_tps > 0 ? n.prefill_tps : n.prefill_cfg) + '</b> t/s' + (n.prefill_tps > 0 ? "" : " (default)") + '</span>' +
    '<span title="the decode speed of one request: the average of its answers of 16 tokens and more, with the others it ran beside them. the new conversations are shared out by it">' +
    'decode <b>' + (n.decode_tps > 0 ? tps(n.decode_tps) : "-") + '</b> t/s</span>' +
    '<span title="of the prefix a new conversation shares with others it holds (the system prompt and tools of an agent), the part it took from its cache: the average of such requests. a hybrid model (gated delta net) reuses only up to a checkpoint inside that prefix">' +
    'shared prefix <b>' + (n.share >= 0 ? Math.round(n.share * 100) + "%" : "-") + '</b></span>' +
    '<span>served <b>' + n.requests + '</b></span>' +
    '<span>moves state <b>' + (n.migrate ? "yes" : "no") + '</b></span></div>';
  if (n.running && n.running.length) {
    html += '<div class="run">' + n.running.map(r => '<span class="chip" title="' + esc(r.path) + '">' +
      esc(r.conv || "#" + r.id) + " \u00b7 " + (r.stage ? esc(r.stage) + " \u00b7 " : "") + age(r.age_ms) + '</span>').join("") + '</div>';
  }
  if (n.status !== "up" && n.error) html += '<div class="errline">' + esc(n.error) + '</div>';
  if (n.mismatch) html += '<div class="errline">another model than the other nodes: ' + esc(JSON.stringify(mi)) + '</div>';
  return html + '</div>';
}

function table(head, rows, empty) {
  if (!rows.length) return '<div class="empty">' + esc(empty) + '</div>';
  return '<table><thead><tr>' + head.map(h => '<th' + (h.n ? ' class="n"' : '') + '>' + esc(h.t) + '</th>').join("") +
    '</tr></thead><tbody>' + rows.join("") + '</tbody></table>';
}

function render(s) {
  const p = s.pool;
  const c = s.config;
  // a router with several pools: a link to each, this one in bold
  const others = s.pools && s.pools.length > 1 ? s.pools : [];
  document.getElementById("pools").innerHTML = others.map(m => {
    const q = new URLSearchParams(location.search);
    q.set("model", m);
    return m === c.model ? "<b>" + esc(m) + "</b>" : '<a href="?' + esc(q.toString()) + '">' + esc(m) + "</a>";
  }).join(" \u00b7 ");
  document.getElementById("sub").textContent = [s.name, c.names && c.names.length ? "model " + c.names.join(", ") : "",
    "a conversation with ~" + tok(c.move_tokens) + " tok of its own on a busy node waits for it, up to " + c.wait_max + " s" +
    (c.migrate ? ", or moves with its state" : ""),
    s.state_kib_per_token > 0 ? "state ~" + s.state_kib_per_token.toFixed(0) + " KiB/tok" : ""].filter(Boolean).join(" \u00b7 ");
  document.getElementById("t-nodes").innerHTML = p.nodes_up + "<small>/ " + p.nodes + "</small>";
  document.getElementById("t-slots").innerHTML = p.slots_busy + "<small>/ " + p.slots + "</small>";
  document.getElementById("t-queue").textContent = p.queued;
  document.getElementById("t-gen").innerHTML = tps(p.gen_tps) + "<small>t/s</small>";
  document.getElementById("t-pp").innerHTML = tps(p.prompt_tps) + "<small>t/s</small>";
  document.getElementById("t-conv").textContent = p.conversations;
  hist.gen.push(p.gen_tps); hist.pp.push(p.prompt_tps);
  if (hist.gen.length > HMAX) { hist.gen.shift(); hist.pp.shift(); }
  spark("c-gen", hist.gen, cssv("--gen"));
  spark("c-pp", hist.pp, cssv("--prompt"));

  document.getElementById("nodes").innerHTML = s.nodes.map(nodeCard).join("");

  document.getElementById("queue").innerHTML = table(
    [{ t: "conversation" }, { t: "waits for" }, { t: "waiting", n: 1 }, { t: "cached there", n: 1 }, { t: "prompt", n: 1 }],
    s.queue.map(q => '<tr><td class="mono">' + esc(q.conv || "#" + q.id) + (q.short ? ' <small>short</small>' : '') + '</td><td>' + esc(q.want || "any node") +
      '</td><td class="n">' + age(q.age_ms) + '</td><td class="n">' + (q.want ? "~" + tok(q.est_tokens) : "-") +
      '</td><td class="n">~' + tok(q.prompt_tokens) + '</td></tr>'),
    "nothing waits");

  document.getElementById("convs").innerHTML = table(
    [{ t: "conversation" }, { t: "node" }, { t: "msgs", n: 1 }, { t: "prompt", n: 1 }, { t: "requests", n: 1 }, { t: "last", n: 1 }],
    s.conversations.map(c => '<tr><td class="mono">' + (c.active ? '<span class="act-dot"></span>' : '<span class="idle-dot"></span>') +
      esc(c.conv) + '</td><td>' + esc(c.node) + '</td><td class="n">' + c.messages + '</td><td class="n">~' + tok(c.prompt_tokens) +
      '</td><td class="n">' + c.requests + '</td><td class="n">' + age(c.age_ms) + '</td></tr>'),
    "no conversation yet");

  document.getElementById("events").innerHTML = table(
    [{ t: "time" }, { t: "what" }, { t: "conversation" }, { t: "node" }, { t: "detail" }],
    s.events.map(e => '<tr><td>' + clock(e.t) + '</td><td class="k k-' + esc(e.kind) + '">' + esc(e.kind) +
      '</td><td class="mono">' + esc(e.conv) + '</td><td>' + esc(e.node) + '</td><td class="d">' + esc(e.detail) + '</td></tr>'),
    "no decision yet");
}

document.addEventListener("click", ev => {
  const b = ev.target.closest("button[data-node]");
  if (!b) return;
  fetch(base + "/pool/drain", {
    method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, H),
    body: JSON.stringify({ node: b.dataset.node, drain: b.dataset.drain === "1" })
  }).then(tick);
});

let busy = false;
async function tick() {
  if (busy) return;
  busy = true;
  try {
    const r = await fetch(base + "/pool/status" + (model ? "?model=" + encodeURIComponent(model) : ""), { headers: H, cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    render(await r.json());
    document.getElementById("dot").className = "dot";
    document.getElementById("live").textContent = "live \u00b7 " + new Date().toTimeString().slice(0, 8);
  } catch (e) {
    document.getElementById("dot").className = "dot off";
    document.getElementById("live").textContent = "no answer from the pool: " + e.message;
  } finally {
    busy = false;
  }
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
)POOLUI",
};
