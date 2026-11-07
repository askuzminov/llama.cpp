// fork: the dashboard of --pool, GET /pool: a page that reads /pool/status every second (server-pool-fork.cpp).
// self-contained, the nodes may have no internet. in parts: MSVC takes a string literal of up to 16 KB
// pages by the hash of the address: #/ (overview), #/node/<name>, #/conversations, #/conv/<id>, #/decisions, #/config.
// edit the page here as one text, then scripts/fork/pool-ui-split.py splits it into parts again

static const char * const POOL_UI_PARTS[] = {
R"POOLUI(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama pool</title>
<style>
:root {
  color-scheme: light;
  --bg: #f3f5f8; --panel: #ffffff; --panel2: #f8f9fb; --text: #151a21; --muted: #697281; --faint: #9ba3ae;
  --line: #e3e7ec; --soft: #eef1f5; --hover: #f6f8fa;
  --accent: #2f6fdb; --ok: #1a9550; --warn: #b97700; --bad: #d03f3f; --teal: #0d8f84;
  --gen: #2f6fdb; --prompt: #8b4fd6; --idle: #cfd5dd;
  --shadow: 0 1px 2px rgba(16, 24, 40, .05);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #0e1116; --panel: #161a21; --panel2: #1a1f27; --text: #e5e8ee; --muted: #98a1ae; --faint: #6b7381;
    --line: #262c36; --soft: #212730; --hover: #1c2129;
    --accent: #6b9cf2; --ok: #3cbd74; --warn: #e2a43c; --bad: #ef6b6b; --teal: #2ec4b6;
    --gen: #6b9cf2; --prompt: #b088f2; --idle: #3a414c;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { font: 13.5px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text);
  -webkit-font-smoothing: antialiased; }
a { color: inherit; text-decoration: none; }
.lk { color: var(--accent); }
.lk:hover { text-decoration: underline; }
.nl { color: var(--text); }
.nl:hover { color: var(--accent); text-decoration: underline; }
.mono { font-family: var(--mono); font-size: 12px; }
.muted { color: var(--muted); }
.grow { flex: 1; }
button { font: inherit; font-size: 12px; padding: 4px 11px; border-radius: 7px; border: 1px solid var(--line);
  background: var(--panel); color: var(--text); cursor: pointer; white-space: nowrap; }
button:hover { border-color: var(--accent); }
input[type=search], select { font: inherit; font-size: 12.5px; padding: 5px 9px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text); }
input[type=search] { min-width: 240px; }
label.chk { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--muted); cursor: pointer; }

/* header */
.top { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 20px; padding: 0 20px; min-height: 52px;
  background: color-mix(in srgb, var(--panel) 94%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: 10px; white-space: nowrap; min-width: 0; }
.logo { width: 22px; height: 22px; border-radius: 7px; flex: none; background: linear-gradient(135deg, var(--gen), var(--prompt)); }
.brand b { font-size: 15px; }
.brand .model { color: var(--muted); font-size: 12.5px; max-width: 340px; overflow: hidden; text-overflow: ellipsis; }
.top nav { display: flex; gap: 2px; align-self: stretch; }
.top nav a { display: flex; align-items: center; gap: 6px; padding: 0 12px; color: var(--muted); font-weight: 550;
  border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap; }
.top nav a:hover { color: var(--text); }
.top nav a.on { color: var(--text); border-bottom-color: var(--accent); }
.cnt { font-size: 11px; padding: 0 6px; border-radius: 999px; background: var(--soft); color: var(--muted); font-weight: 600; }
.cnt.wa { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.live { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 7px; white-space: nowrap; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); flex: none;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
.dot.off { background: var(--bad); box-shadow: 0 0 0 3px color-mix(in srgb, var(--bad) 22%, transparent); }
.dot.paused { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in srgb, var(--warn) 22%, transparent); }
@media (max-width: 860px) {
  .top { flex-wrap: wrap; gap: 4px 14px; padding: 8px 16px 0; }
  .top nav { order: 3; width: 100%; overflow-x: auto; height: 38px; }
  .brand .model { max-width: 160px; }
}

main { max-width: 1600px; margin: 0 auto; padding: 18px 20px 48px; }
@media (max-width: 640px) { main { padding: 14px 16px 40px; } }
.mb { margin-bottom: 12px; }

/* tiles */
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
@media (max-width: 1240px) { .kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; box-shadow: var(--shadow);
  min-width: 0; display: flex; flex-direction: column; min-height: 108px; }
.kpi .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.kpi .v { font-size: 26px; font-weight: 650; letter-spacing: -.01em; margin-top: 2px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.kpi .v small { font-size: 13px; color: var(--muted); font-weight: 500; margin-left: 4px; letter-spacing: 0; }
.kpi .s { font-size: 12px; color: var(--muted); margin-top: auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .s b { color: var(--text); font-weight: 600; }
.kpi canvas { width: 100%; height: 30px; display: block; margin-top: auto; }
.pills { display: flex; flex-wrap: wrap; gap: 3px 10px; margin-top: auto; font-size: 12px; color: var(--muted); }
.pills span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; max-width: 100%; overflow: hidden; }
.pills span::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--ok); flex: none; }
.pills span.wa::before { background: var(--warn); } .pills span.ba::before { background: var(--bad); }
.sq { display: flex; flex-wrap: wrap; gap: 3px; margin-top: auto; align-items: center; }
.sq i { width: 13px; height: 13px; border-radius: 3px; background: var(--idle); }
.sq i.gen { background: var(--gen); } .sq i.prompt { background: var(--prompt); }
.sq i.off { background: transparent; border: 1.5px dashed var(--idle); }
.sq b { width: 7px; }

/* nodes */
.nodes { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; margin-bottom: 14px; align-items: start; }
@media (max-width: 520px) { .nodes { grid-template-columns: minmax(0, 1fr); } }
.node, .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
  padding: 14px 16px; min-width: 0; }
.node { position: relative; overflow: hidden; }
.node::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--ok); }
.node.s-draining::before, .node.s-loading::before, .node.s-unloaded::before { background: var(--warn); }
.node.s-down::before, .node.s-mismatch::before { background: var(--bad); }
.node.dim .body { opacity: .62; }
.nhead { display: flex; align-items: center; gap: 8px; min-width: 0; }
.nname { font-weight: 650; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
a.nname:hover { color: var(--accent); }
.tag { font-size: 11px; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 0 7px; white-space: nowrap; }
.badge { font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px; background: var(--soft); color: var(--muted); white-space: nowrap; }
.badge.ok { background: color-mix(in srgb, var(--ok) 15%, transparent); color: var(--ok); }
.badge.wa { background: color-mix(in srgb, var(--warn) 17%, transparent); color: var(--warn); }
.badge.ba { background: color-mix(in srgb, var(--bad) 15%, transparent); color: var(--bad); }
.nsub { color: var(--muted); font-size: 12px; margin: 3px 0 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.metric { background: var(--panel2); border: 1px solid var(--line); border-radius: 9px; padding: 6px 10px; min-width: 0; }
.metric .ml { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.metric .mv { font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; }
.metric .mv small { color: var(--muted); margin-left: 3px; font-size: 11.5px; font-weight: 500; }
.metric.gen .mv { color: var(--gen); } .metric.prompt .mv { color: var(--prompt); }
.metric.zero .mv { color: var(--text); }
.spark { width: 100%; height: 36px; display: block; margin: 8px 0 2px; }
.sec { color: var(--muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; margin: 12px 0 4px; }
.row { display: grid; grid-template-columns: 74px minmax(40px, 1fr) auto; align-items: center; gap: 10px; font-size: 12px; margin: 5px 0; }
.row.m { grid-template-columns: minmax(70px, 128px) minmax(40px, 1fr) auto; }
.row .l { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .l .sid { display: inline-block; min-width: 14px; color: var(--faint); font-variant-numeric: tabular-nums; }
.row .r { white-space: nowrap; font-variant-numeric: tabular-nums; color: var(--muted); text-align: right; }
.row .r b { color: var(--text); font-weight: 600; }
.bar { height: 8px; border-radius: 4px; background: var(--soft); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 4px; background: var(--accent); }
.bar.gen i { background: var(--gen); } .bar.prompt i { background: var(--prompt); } .bar.idle i { background: var(--idle); }
.bar.wa i { background: var(--warn); } .bar.ba i { background: var(--bad); }
.mrow { margin: 8px 0; font-size: 12px; }
.mrow .mh { display: flex; gap: 10px; margin-bottom: 3px; }
.mrow .mh .l { color: var(--muted); min-width: 0; overflow-wrap: anywhere; }
.mrow .mh .r { margin-left: auto; white-space: nowrap; color: var(--muted); font-variant-numeric: tabular-nums; }
.mrow .mh .r b { color: var(--text); font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip { font-size: 11.5px; padding: 2px 9px; border-radius: 999px; background: var(--soft); color: var(--text); white-space: nowrap;
  max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.chip .muted { margin-left: 4px; }
.chip.wa { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, transparent); }
.chip.ba { color: var(--bad); background: color-mix(in srgb, var(--bad) 13%, transparent); }
a.chip:hover { outline: 1px solid var(--accent); }
.foot { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line);
  margin-top: 12px; padding-top: 9px; }
.foot b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
.err { margin-top: 10px; font-size: 12px; color: var(--bad); background: color-mix(in srgb, var(--bad) 9%, transparent);
  border-radius: 8px; padding: 6px 10px; overflow-wrap: anywhere; }
.warnline { margin-top: 10px; font-size: 12px; color: var(--warn); background: color-mix(in srgb, var(--warn) 10%, transparent);
  border-radius: 8px; padding: 6px 10px; overflow-wrap: anywhere; }

/* panels and tables */
.ch { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; min-width: 0; }
.ch h2 { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0; font-weight: 650; white-space: nowrap; }
.ch .muted { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ch .more { margin-left: auto; font-size: 12px; color: var(--accent); white-space: nowrap; }
.ch .more:hover { text-decoration: underline; }
.cols2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; align-items: start; }
@media (max-width: 1080px) { .cols2 { grid-template-columns: minmax(0, 1fr); } }
.tw { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11.5px; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th.s { cursor: pointer; user-select: none; }
th.s:hover { color: var(--text); }
th.on { color: var(--text); }
th.on::after { content: ""; display: inline-block; margin-left: 5px; border: 4px solid transparent; vertical-align: 1px; }
th.on.up::after { border-bottom-color: currentColor; vertical-align: 3px; }
th.on.down::after { border-top-color: currentColor; vertical-align: -1px; }
td { padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; font-variant-numeric: tabular-nums; }
tbody tr:hover td { background: var(--hover); }
tr:last-child td { border-bottom: 0; }
td.n, th.n { text-align: right; }
td.w { white-space: normal; color: var(--muted); min-width: 260px; overflow-wrap: anywhere; }
td.cut { max-width: 1px; width: 100%; overflow: hidden; text-overflow: ellipsis; color: var(--muted); }
td .bar { min-width: 90px; }
.adot, .idot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
.adot { background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 20%, transparent); }
.idot { background: var(--idle); }
.empty { color: var(--muted); font-size: 12.5px; padding: 10px 2px; }
.note { color: var(--faint); font-size: 11.5px; margin-top: 8px; }
.kb { display: inline-block; font-size: 11px; font-weight: 650; padding: 1px 7px; border-radius: 6px; background: var(--soft);
  color: var(--muted); min-width: 58px; text-align: center; }
.kb.ok { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
.kb.ac { color: var(--accent); background: color-mix(in srgb, var(--accent) 13%, transparent); }
.kb.wa { color: var(--warn); background: color-mix(in srgb, var(--warn) 15%, transparent); }
.kb.ba { color: var(--bad); background: color-mix(in srgb, var(--bad) 14%, transparent); }
.kb.te { color: var(--teal); background: color-mix(in srgb, var(--teal) 14%, transparent); }
.feed .ev { display: grid; grid-template-columns: 58px 66px minmax(0, 1fr); gap: 10px; align-items: baseline; padding: 6px 0;
  border-bottom: 1px solid var(--line); font-size: 12.5px; }
.feed .ev:last-child { border-bottom: 0; }
.ev .t { color: var(--faint); font-variant-numeric: tabular-nums; font-size: 11.5px; }
.ev .d { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
.ev .d .who { color: var(--text); margin-right: 6px; }

/* toolbars, pages */
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-bottom: 12px; }
.seg { display: inline-flex; background: var(--soft); border-radius: 9px; padding: 2px; }
)POOLUI",
R"POOLUI(.seg button { border: 0; background: transparent; padding: 4px 12px; border-radius: 7px; color: var(--muted); font-weight: 600; }
.seg button.on { background: var(--panel); color: var(--text); box-shadow: 0 1px 2px rgba(0, 0, 0, .08); }
.ph { display: flex; align-items: center; gap: 10px; margin: 0 0 14px; flex-wrap: wrap; min-width: 0; }
.ph h1 { font-size: 20px; margin: 0; letter-spacing: -.01em; overflow-wrap: anywhere; }
.ph h1.mono { font-size: 17px; }
.crumb { color: var(--muted); font-size: 12.5px; margin-bottom: 6px; }
.crumb a:hover { color: var(--accent); }
.gn { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(0, 1fr); gap: 12px; align-items: start; }
@media (max-width: 1120px) { .gn { grid-template-columns: minmax(0, 1fr); } }
.stack { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.mgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
@media (max-width: 640px) { .mgrid, .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.charts { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
@media (max-width: 640px) { .charts { grid-template-columns: minmax(0, 1fr); } }
.chart { width: 100%; height: 86px; display: block; margin-top: 4px; }
.chl { display: flex; align-items: baseline; gap: 8px; font-size: 12px; color: var(--muted); }
.chl b { font-size: 15px; color: var(--text); font-variant-numeric: tabular-nums; }
.chl .r { margin-left: auto; font-size: 11.5px; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 16px; font-size: 12.5px; margin: 0; }
.kv dt { color: var(--muted); white-space: nowrap; }
.kv dd { margin: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.cfg td:first-child { color: var(--text); font-weight: 600; }
button.chip { border: 0; cursor: pointer; font-size: 11.5px; padding: 2px 8px 2px 2px; display: inline-flex; align-items: center; gap: 6px; }
button.chip b { font-variant-numeric: tabular-nums; }
button.chip.on { outline: 2px solid var(--accent); }
.cfg td.v { font-family: var(--mono); font-size: 12px; }
</style>
</head>
<body>
<header class="top">
  <div class="brand"><span class="logo"></span><b>llama pool</b><span class="model" id="h-model"></span><span id="h-pools"></span></div>
  <nav id="nav">
    <a href="#/" data-p="overview">Overview <span class="cnt" id="n-queue" hidden></span></a>
    <a href="#/conversations" data-p="conversations">Conversations <span class="cnt" id="n-conv"></span></a>
    <a href="#/decisions" data-p="decisions">Decisions</a>
    <a href="#/config" data-p="config">Config</a>
  </nav>
  <div class="right">
    <button id="pause" title="stop updating the page, to read or copy">Pause</button>
    <span class="live"><span class="dot" id="dot"></span><span id="live">connecting</span></span>
  </div>
</header>
<main id="view"></main>
<script>
"use strict";
const HMAX = 300;
const base = location.pathname.replace(/\/pool\/?$/, "");
const query = new URLSearchParams(location.search);
const key = query.get("key");
const model = query.get("model");
const H = key ? { "Authorization": "Bearer " + key } : {};
const DOT = " \u00b7 ";

let S = null;
let paused = false;
let busy = false;
const hist = { gen: [], pp: [] };
const nhist = {};
const ui = { cq: "", cnode: "", cactive: false, csort: "age_ms", cdir: 1, eg: "all", eq: "", enode: "", ek: "" };

function $(id) { return document.getElementById(id); }
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
function size(b) { b = Number(b) || 0; return b >= 1073741824 ? gib(b) + " GiB" : (b / 1048576).toFixed(b >= 104857600 ? 0 : 1) + " MiB"; }
function count(n) { n = Number(n) || 0; return n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : tok(n); }
function age(ms) {
  const s = (Number(ms) || 0) / 1000;
  if (s < 1) return "now";
  if (s < 60) return s.toFixed(0) + " s";
  if (s < 3600) return (s / 60).toFixed(0) + " min";
  if (s < 172800) return (s / 3600).toFixed(1) + " h";
  return (s / 86400).toFixed(1) + " d";
}
function clock(t) { return new Date(t).toTimeString().slice(0, 8); }
function cssv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function setHTML(el, html) {
  if (typeof el === "string") el = $(el);
  if (!el || el.__h === html) return;
  el.innerHTML = html;
  el.__h = html;
}
function push(a, v) { a.push(Number(v) || 0); if (a.length > HMAX) a.shift(); }
function peak(a) { return a.length ? Math.max(...a) : 0; }
function nodeHref(name) { return "#/node/" + encodeURIComponent(name); }
function convHref(id) { return "#/conv/" + encodeURIComponent(id); }
function nodeLink(name) { return name ? '<a class="nl" href="' + nodeHref(name) + '">' + esc(name) + '</a>' : '<span class="muted">-</span>'; }
function convLink(id) { return id ? '<a class="lk mono" href="' + convHref(id) + '">' + esc(id) + '</a>' : '<span class="muted">-</span>'; }
function yesno(b) { return b ? "yes" : "no"; }

// the state of a node, and its color: ok, wa (warn), ba (bad)
function nodeState(n) { return n.status !== "up" ? (n.status || "down") : n.mismatch ? "mismatch" : n.draining ? "draining" : "up"; }
function badges(n) {
  const st = n.status || "down";
  return '<span class="badge ' + stateCls(st) + '">' + esc(st) + '</span>' + (n.draining ? '<span class="badge wa" title="takes no new conversations">draining</span>' : '') +
    (n.mismatch ? '<span class="badge ba" title="another model than the other nodes">mismatch</span>' : '');
}
function stateCls(st) { return st === "up" ? "ok" : (st === "down" || st === "mismatch") ? "ba" : "wa"; }
function busyOf(n) { return (n.inflight || 0) + (n.external || 0); }

// GPU names without the vendor words: "AMD Radeon(TM) 8060S Graphics" -> "8060S", "NVIDIA GeForce RTX 3090" -> "RTX 3090"
function shortName(s) {
  return String(s || "").replace(/\((R|TM)\)/gi, "").replace(/\b(AMD|NVIDIA|Intel|GeForce|Radeon|Graphics|Corporation)\b/gi, "")
    .replace(/\s+/g, " ").trim();
}
function devLabel(d) {
  if (d.type === "cpu") return "RAM";
  const s = shortName(d.description);
  return (d.name || "GPU") + (s ? DOT + s : "");
}
// a GPU can come twice from the driver, once without a name ("GPU 0") with the same readings: the named one stays
function gpusOf(n) {
  const gs = n.gpus || [];
  const sig = g => [g.temp_c, g.power_pct, g.power_w, g.fan_rpm].join("|");
  const unnamed = g => /^GPU \d+$/.test(g.name || "");
  const named = new Set(gs.filter(g => !unnamed(g)).map(sig));
  return gs.filter(g => !unnamed(g) || !named.has(sig(g)));
}
function gpuText(g) {
  return [g.temp_c != null ? g.temp_c.toFixed(0) + " \u00b0C" : "", g.power_pct != null ? "power " + g.power_pct.toFixed(0) + "%" : "",
    g.power_w != null ? g.power_w.toFixed(0) + " W" : "", g.fan_rpm != null ? "fan " + g.fan_rpm + " rpm" : ""].filter(Boolean).join(DOT);
}
function gpuCls(g) { return g.temp_c >= 90 ? "ba" : g.temp_c >= 80 ? "wa" : ""; }

const KIND = {
  "new": "ac", cold: "ac", keep: "ok", "switch": "ok", migrate: "te", recompute: "wa",
  done: "", error: "ba", gone: "wa", fail: "ba",
  up: "ok", down: "ba", loading: "wa", unloaded: "wa", load: "ac", link: "wa", "link+": "ac", found: "ac",
  match: "", mismatch: "ba", conflict: "ba", drain: "wa", undrain: ""
};
const KIND_INFO = {
  "new": "a new conversation, to the node with the most free slots",
  cold: "no node holds the conversation: computed from the start",
  keep: "stays on the node that holds its cache",
  "switch": "goes to another node that holds an older copy of it: that node computes the difference",
  migrate: "goes to another node with its cached state",
  recompute: "goes to another node, which computes its prompt: no state could come along (the reason in the details)",
  done: "a request finished", error: "a request ended with an error",
  gone: "the client left while the request waited", fail: "no node could take the request",
  up: "a node is up", down: "a node does not answer", loading: "a node loads the model", unloaded: "the model of a node is unloaded",
  load: "the pool asked a router to load the model", link: "a node goes over another link", "link+": "another link to a router",
  found: "a router joined the pool", match: "the model of a node matches the others", mismatch: "a node has another model",
  conflict: "two routers have the same id", drain: "a node takes no new conversations", undrain: "a node takes conversations again"
};
const GROUPS = { routing: ["new", "cold", "keep", "switch", "migrate", "recompute"], requests: ["done", "error", "gone", "fail"] };
function groupOf(k) { return GROUPS.routing.includes(k) ? "routing" : GROUPS.requests.includes(k) ? "requests" : "nodes"; }
function kb(k) { return '<span class="kb ' + (KIND[k] || "") + '" title="' + esc(KIND_INFO[k] || k) + '">' + esc(k) + '</span>'; }

function table(head, rows, empty, sortable) {
  if (!rows.length) return '<div class="empty">' + esc(empty) + '</div>';
  return '<div class="tw"><table><thead><tr>' + head.map(h => {
    const cls = [h.n ? "n" : "", sortable && h.k ? "s" : "", sortable && h.k === ui.csort ? "on " + (ui.cdir > 0 ? "down" : "up") : ""]
      .filter(Boolean).join(" ");
    return '<th' + (cls ? ' class="' + cls + '"' : '') + (sortable && h.k ? ' data-sort="' + h.k + '"' : '') +
      (h.title ? ' title="' + esc(h.title) + '"' : '') + '>' + esc(h.t) + '</th>';
  }).join("") + '</tr></thead><tbody>' + rows.join("") + '</tbody></table></div>';
}
function metric(label, value, unit, cls, title) {
  return '<div class="metric' + (cls ? " " + cls : "") + '"' + (title ? ' title="' + esc(title) + '"' : '') + '><div class="ml">' + esc(label) +
    '</div><div class="mv">' + value + (unit ? '<small>' + esc(unit) + '</small>' : '') + '</div></div>';
}
function drainBtn(n) {
  return '<button data-node="' + esc(n.name) + '" data-drain="' + (n.draining ? "0" : "1") + '" title="' +
    (n.draining ? "take new conversations again" : "take no new conversations, the running ones finish") + '">' +
    (n.draining ? "Resume" : "Drain") + '</button>';
}
function slotRow(s) {
  const used = (s.n_prompt || 0) + (s.n_decoded || 0);
  let frac = s.n_ctx > 0 ? Math.min(1, used / s.n_ctx) : 0;
  let r, title = "context " + tok(used) + " of " + tok(s.n_ctx);
  const st = s.state === "gen" || s.state === "prompt" ? s.state : "idle";
  if (st === "gen") {
    r = '<b>' + tps(s.gen_tps) + '</b> t/s' + DOT + tok(used);
  } else if (st === "prompt") {
    const done = (s.n_processed || 0) + (s.n_cached || 0);
    frac = s.n_prompt > 0 ? Math.min(1, done / s.n_prompt) : 0;
    title = "prompt " + tok(done) + " of " + tok(s.n_prompt) + " read";
    r = '<b>' + tps(s.prompt_tps) + '</b> t/s' + DOT + tok(done) + " / " + tok(s.n_prompt);
  } else {
    r = used > 0 ? tok(used) + " kept" : "empty";
  }
  return '<div class="row"><span class="l"><span class="sid">' + esc(s.id) + '</span> ' + st + '</span>' +
    '<div class="bar ' + st + '" title="' + esc(title) + '"><i style="width:' + (frac * 100).toFixed(1) + '%"></i></div>' +
    '<span class="r">' + r + '</span></div>';
}
function memRow(d, full) {
  const total = Number(d.total) || 0, free = Number(d.free) || 0;
  if (total <= 0) return "";
  const used = Math.max(0, total - free), f = used / total;
  const label = full ? (d.type === "cpu" ? "RAM" + DOT + d.description : d.name + DOT + d.description) : devLabel(d);
  const bar = '<div class="bar' + (f > 0.995 ? " ba" : f > 0.98 ? " wa" : "") + '"><i style="width:' + (f * 100).toFixed(1) + '%"></i></div>';
  const val = '<span class="r"><b>' + gib(used) + '</b> / ' + gib(total) + ' GiB</span>';
  if (full) return '<div class="mrow"><div class="mh"><span class="l">' + esc(label) + '</span>' + val + '</div>' + bar + '</div>';
  return '<div class="row m" title="' + esc(d.name + ": " + (d.description || "")) + '"><span class="l">' + esc(label) + '</span>' + bar + val + '</div>';
}
function roomRow(n, full) {
  const r = n.room;
  if (!r || !(r.n_ctx > 0)) return "";
  const f = Math.min(1, (r.held || 0) / r.n_ctx);
  const tip = "the cells of its unified KV cache that the running and paused slots hold with their answer reserve: a prompt that does not fit next to them waits in the pool for a node with room";
  const bar = '<div class="bar' + (f > 0.92 ? " wa" : "") + '"><i style="width:' + (f * 100).toFixed(1) + '%"></i></div>';
  const val = '<span class="r"><b>' + tok(r.held) + '</b> / ' + tok(r.n_ctx) + (r.deferred ? DOT + r.deferred + " waiting inside" : "") + '</span>';
  if (full) return '<div class="mrow" title="' + tip + '"><div class="mh"><span class="l">KV cache held by the slots</span>' + val + '</div>' + bar + '</div>';
  return '<div class="row m" title="' + tip + '"><span class="l">KV held</span>' + bar + val + '</div>';
}
function runChips(n) {
  return (n.running || []).map(r => '<a class="chip" href="' + convHref(r.conv || "") + '" title="' + esc(r.path || "") + '">' +
    esc(r.conv || "#" + r.id) + '<span class="muted">' + (r.stage ? esc(r.stage) + DOT : "") + age(r.age_ms) + '</span></a>').join("");
}

function nodeCard(n) {
  const st = nodeState(n), mi = n.model_info || {};
  const sub = [String(n.url || "").replace(/^https?:\/\//, ""), n.n_ctx ? "ctx " + tok(n.n_ctx) : "", n.slots ? n.slots + (n.slots === 1 ? " slot" : " slots") : "",
    n.weight != null ? "weight " + n.weight.toFixed(2) : ""].filter(Boolean).join(DOT);
  const title = [n.model, mi.ftype, n.signature ? "cache " + n.signature : ""].filter(Boolean).join(DOT);
  let h = '<article class="node s-' + esc(st) + (st === "up" ? "" : " dim") + '">' +
    '<div class="nhead"><a class="nname" href="' + nodeHref(n.name) + '">' + esc(n.name) + '</a>' +
    (n.self ? '<span class="tag">this router</span>' : '') + badges(n) + '<span class="grow"></span>' + drainBtn(n) + '</div>' +
    '<div class="nsub" title="' + esc(title) + '">' + esc(sub || n.url) + '</div><div class="body">';
  const b = busyOf(n);
  h += '<div class="metrics">' +
    metric("gen", tps(n.gen_tps), "t/s", n.gen_tps > 0 ? "gen" : "zero", "tokens its slots generate per second now") +
    metric("prompt", tps(n.prompt_tps), "t/s", n.prompt_tps > 0 ? "prompt" : "zero", "prompt tokens its slots compute per second now, each over its last batch") +
    metric("busy", b, "/ " + (n.slots || 0), "", "requests in its slots: from this router" + (n.external ? ", " + n.external + " from other routers and clients" : "")) +
)POOLUI",
R"POOLUI(    metric("waiting", n.queued_for || 0, "", "", "requests in the queue of the pool that wait for this node") + '</div>';
  h += '<canvas class="spark" data-spark="node" data-name="' + esc(n.name) + '" title="the last 5 minutes: generation (fill) and prompt (line), each on its own scale"></canvas>';
  if (n.slot_list && n.slot_list.length) h += '<div class="sec">Slots</div>' + n.slot_list.map(slotRow).join("");
  const mem = (n.devices || []).map(d => memRow(d, false)).join("") + roomRow(n, false);
  if (mem) h += '<div class="sec">Memory</div>' + mem;
  const gpus = gpusOf(n).map(g => '<span class="chip ' + gpuCls(g) + '" title="' + esc(g.name) + '">' + esc(shortName(g.name) || g.name) +
    '<span class="muted">' + esc(gpuText(g)) + '</span></span>').join("");
  const run = runChips(n);
  if (gpus || run || n.fallback) {
    h += '<div class="chips">' + run + gpus + (n.fallback ? '<span class="chip wa" title="a faster link answered: the node goes back to it at the next probe">fallback link</span>' : '') + '</div>';
  }
  const st2 = n.stats || {};
  h += '<div class="foot">' +
    '<span title="the decode speed of one request: the average of its answers of 16 tokens and more">decode <b>' + (n.decode_tps > 0 ? tps(n.decode_tps) : "-") + '</b> t/s</span>' +
    '<span title="the prefill speed of one request the decisions use: the average of its prompts of move_tokens and more, else a default">prefill <b>' +
    tps(n.prefill_tps > 0 ? n.prefill_tps : n.prefill_cfg) + '</b>' + (n.prefill_tps > 0 ? "" : " (default)") + '</span>' +
    '<span title="of the prefix a new conversation shares with others it holds (the system prompt and tools of an agent), the part it took from its cache">shared prefix <b>' +
    (n.share >= 0 ? Math.round(n.share * 100) + "%" : "-") + '</b></span>' +
    '<span title="requests this router sent to it">served <b>' + (n.requests || 0) + '</b></span>' +
    (st2.uptime_s > 0 ? '<span title="since its model loaded">up <b>' + age(st2.uptime_s * 1000) + '</b></span>' : '') + '</div>';
  if (st !== "up" && n.error) h += '<div class="err">' + esc(n.error) + '</div>';
  if (n.mismatch) h += '<div class="err">another model than the other nodes (' + esc([n.model, mi.ftype].filter(Boolean).join(DOT)) + '): it takes no request, pool-same-model = 0 lets it</div>';
  if (st === "up" && n.seen_ms > 5000) h += '<div class="warnline">no answer to the polls for ' + age(n.seen_ms) + '</div>';
  return h + '</div></article>';
}

function queueTable(q, all) {
  return table([{ t: "conversation" }, { t: "waits for" }, { t: "waiting", n: 1 }, { t: "cached there", n: 1 }, { t: "prompt", n: 1 }],
    q.slice(0, all ? q.length : 6).map(x => '<tr><td>' + convLink(x.conv || "#" + x.id) + (x.short ? ' <span class="chip">short answer</span>' : '') +
      '</td><td>' + (x.want ? esc(x.want) : '<span class="muted">any node</span>') + '</td><td class="n">' + age(x.age_ms) +
      '</td><td class="n">' + (x.want ? "~" + tok(x.est_tokens) : "-") + '</td><td class="n">~' + tok(x.prompt_tokens) + '</td></tr>'),
    "nothing waits");
}
function feedRow(e, noNode) {
  const who = [e.conv ? convLink(e.conv) : "", e.node && !noNode ? nodeLink(e.node) : ""].filter(Boolean).join('<span class="muted">' + DOT + '</span>');
  return '<div class="ev"><span class="t">' + clock(e.t) + '</span>' + kb(e.kind) + '<span class="d" title="' + esc(e.detail) + '">' +
    (who ? '<span class="who">' + who + '</span>' : '') + esc(e.detail) + '</span></div>';
}
// overview
function kpis(s) {
  const p = s.pool;
  const pills = s.nodes.map(n => { const st = nodeState(n); return '<span class="' + stateCls(st) + '" title="' + esc(n.name + ": " + st) + '">' + esc(n.name) + '</span>'; }).join("");
  let sq = "";
  s.nodes.forEach((n, i) => {
    const st = nodeState(n);
    const list = n.slot_list && n.slot_list.length ? n.slot_list : Array.from({ length: n.slots || 0 }, (_, k) => ({ id: k, state: "idle" }));
    sq += (i ? '<b></b>' : '') + list.map(x => {
      const run = x.state === "gen" || x.state === "prompt";
      const c = n.status !== "up" || (n.draining && !run) ? "off" : run ? x.state : "";
      return '<i class="' + c + '" title="' + esc(n.name + ", slot " + x.id + ": " + (n.status !== "up" ? n.status : x.state + (n.draining ? ", draining" : ""))) + '"></i>';
    }).join("");
  });
  let ng = 0, np = 0;
  s.nodes.forEach(n => { if (n.status === "up") (n.slot_list || []).forEach(x => { ng += x.state === "gen" ? 1 : 0; np += x.state === "prompt" ? 1 : 0; }); });
  const oldest = s.queue.reduce((m, q) => Math.max(m, q.age_ms || 0), 0);
  const running = s.nodes.reduce((m, n) => m + (n.running || []).length, 0);
  const down = s.nodes.filter(n => n.status !== "up").length, draining = s.nodes.filter(n => n.draining && n.status === "up").length;
  setHTML("kpis",
    '<div class="kpi"><div class="k">Nodes up</div><div class="v">' + p.nodes_up + '<small>/ ' + p.nodes + '</small></div><div class="pills" title="' +
      esc((down ? down + " down" : "none down") + (draining ? DOT + draining + " draining" : "")) + '">' + pills + '</div></div>' +
    '<div class="kpi"><div class="k">Slots busy</div><div class="v">' + p.slots_busy + '<small>/ ' + p.slots + '</small></div><div class="sq">' + sq + '</div>' +
      '<div class="s" style="margin-top:6px"><b>' + ng + '</b> generating' + DOT + '<b>' + np + '</b> reading a prompt</div></div>' +
    '<div class="kpi"><div class="k">Queue</div><div class="v">' + p.queued + '</div><div class="s">' +
      (p.queued ? 'oldest waits <b>' + age(oldest) + '</b>' : 'nothing waits') + '</div></div>' +
    '<div class="kpi"><div class="k">Generation</div><div class="v">' + tps(p.gen_tps) + '<small>t/s</small></div><canvas data-spark="pool-gen"></canvas>' +
      '<div class="s">peak <b>' + tps(peak(hist.gen)) + '</b> t/s in ' + age(hist.gen.length * (s.config.poll_ms || 1000)) + '</div></div>' +
    '<div class="kpi"><div class="k">Prompt</div><div class="v">' + tps(p.prompt_tps) + '<small>t/s</small></div><canvas data-spark="pool-pp"></canvas>' +
      '<div class="s">peak <b>' + tps(peak(hist.pp)) + '</b> t/s in ' + age(hist.pp.length * (s.config.poll_ms || 1000)) + '</div></div>' +
    '<div class="kpi"><div class="k">Conversations</div><div class="v">' + p.conversations + '</div><div class="s"><b>' + running + '</b> running' + DOT +
      '<b>' + (p.prefixes || 0) + '</b> prefixes known</div></div>');
}
function pageOverview(s) {
  kpis(s);
  setHTML("ov-queue", s.queue.length ? '<div class="card mb"><div class="ch"><h2>Queue</h2><span class="muted">' + s.queue.length +
    ' waiting</span>' + (s.queue.length > 6 ? '<a class="more" href="#/conversations">all</a>' : '') + '</div>' + queueTable(s.queue, false) + '</div>' : "");
  setHTML("nodes", s.nodes.length ? s.nodes.map(nodeCard).join("") : '<div class="card empty">no node yet: the routers of the pool appear here once they answer</div>');
  const cs = s.conversations.slice().sort((a, b) => (b.active - a.active) || (a.age_ms - b.age_ms)).slice(0, 10);
  const act = s.conversations.filter(c => c.active).length;
  setHTML("ov-cn", act + " active" + DOT + s.pool.conversations + " known");
  setHTML("ov-convs", table([{ t: "conversation" }, { t: "node" }, { t: "msgs", n: 1 }, { t: "prompt", n: 1 }, { t: "last", n: 1 }],
    cs.map(c => '<tr><td>' + (c.active ? '<span class="adot"></span>' : '<span class="idot"></span>') + convLink(c.conv) + '</td><td>' + nodeLink(c.node) +
      '</td><td class="n">' + c.messages + '</td><td class="n">~' + tok(c.prompt_tokens) + '</td><td class="n">' + age(c.age_ms) + '</td></tr>'),
    "no conversation yet"));
  const ev = s.events.filter(e => e.kind !== "done" && e.kind !== "match").slice(0, 12);
  setHTML("ov-events", ev.length ? '<div class="feed">' + ev.map(e => feedRow(e, false)).join("") + '</div>' : '<div class="empty">no decision yet</div>');
}

// conversations
function nodeOptions(s, cur) {
  return '<option value="">all nodes</option>' + s.nodes.map(n => '<option value="' + esc(n.name) + '"' + (n.name === cur ? " selected" : "") + '>' +
    esc(n.name) + '</option>').join("");
}
function syncSelect(id, s, cur) {
  const el = $(id);
  if (!el) return;
  const names = s.nodes.map(n => n.name).join("\n");
  if (el.__names !== names) { el.innerHTML = nodeOptions(s, cur); el.__names = names; }
}
function pageConversations(s) {
  syncSelect("c-node", s, ui.cnode);
  setHTML("c-queue", s.queue.length ? '<div class="card mb"><div class="ch"><h2>Queue</h2><span class="muted">' + s.queue.length + ' waiting</span></div>' +
    queueTable(s.queue, true) + '</div>' : "");
  const q = ui.cq.trim().toLowerCase();
  let rows = s.conversations.filter(c => (!q || String(c.conv).toLowerCase().includes(q) || String(c.node).toLowerCase().includes(q)) &&
    (!ui.cnode || c.node === ui.cnode) && (!ui.cactive || c.active));
  const k = ui.csort, d = ui.cdir;
  rows.sort((a, b) => {
    const x = a[k], y = b[k];
    const r = typeof x === "string" || typeof y === "string" ? String(x).localeCompare(String(y)) : (Number(x) || 0) - (Number(y) || 0);
    return (r || (a.age_ms - b.age_ms)) * (k === "age_ms" ? d : -d);
  });
  setHTML("c-count", rows.length + " shown" + DOT + s.pool.conversations + " known" + (s.pool.conversations > s.conversations.length ?
    DOT + "the " + s.conversations.length + " most recent come to the page" : ""));
  setHTML("c-table", table([
    { t: "conversation", k: "conv" }, { t: "node", k: "node" }, { t: "msgs", k: "messages", n: 1, title: "messages in its last request" },
    { t: "prompt", k: "prompt_tokens", n: 1, title: "the tokens of its last prompt" },
    { t: "cached there", k: "cached_tokens", n: 1, title: "the tokens of it its node holds in the cache, as the pool estimates" },
    { t: "requests", k: "requests", n: 1 }, { t: "last", k: "age_ms", n: 1 }],
    rows.map(c => '<tr><td>' + (c.active ? '<span class="adot"></span>' : '<span class="idot"></span>') + convLink(c.conv) + '</td><td>' + nodeLink(c.node) +
      '</td><td class="n">' + c.messages + '</td><td class="n">~' + tok(c.prompt_tokens) + '</td><td class="n">' + (c.cached_tokens > 0 ? "~" + tok(c.cached_tokens) : "-") +
      '</td><td class="n">' + c.requests + '</td><td class="n">' + age(c.age_ms) + '</td></tr>'),
    s.conversations.length ? "no conversation matches" : "no conversation yet", true));
}
function pageConv(s, id) {
  const c = s.conversations.find(x => x.conv === id);
  const evs = s.events.filter(e => e.conv === id);
  const runs = [];
  s.nodes.forEach(n => (n.running || []).forEach(r => { if (r.conv === id) runs.push(Object.assign({ node: n.name }, r)); }));
  const qs = s.queue.filter(x => x.conv === id);
  let h = '<div class="crumb"><a href="#/conversations">Conversations</a></div><div class="ph"><h1 class="mono">' + esc(id) + '</h1>' +
    (c ? '<span class="badge ' + (c.active ? "ok" : "") + '">' + (c.active ? "running" : "idle") + '</span>' : '') +
    (qs.length ? '<span class="badge wa">waiting in the queue</span>' : '') + '</div>';
  if (c) {
    h += '<div class="mgrid mb">' + metric("node", c.node ? '<a class="lk" href="' + nodeHref(c.node) + '">' + esc(c.node) + '</a>' : "-", "") +
      metric("messages", c.messages, "") + metric("prompt", "~" + tok(c.prompt_tokens), "tok", "", "the tokens of its last prompt") +
      metric("cached there", c.cached_tokens > 0 ? "~" + tok(c.cached_tokens) : "-", c.cached_tokens > 0 ? "tok" : "", "", "the tokens of it its node holds, as the pool estimates") +
      metric("requests", c.requests, "") + metric("last request", age(c.age_ms), c.age_ms >= 1000 ? "ago" : "") + '</div>';
  } else {
    h += '<div class="card mb empty">not among the ' + s.conversations.length + ' most recent conversations the pool shows</div>';
  }
  if (runs.length) {
    h += '<div class="card mb"><div class="ch"><h2>Running</h2></div>' + table([{ t: "node" }, { t: "stage" }, { t: "for", n: 1 }, { t: "request" }],
      runs.map(r => '<tr><td>' + nodeLink(r.node) + '</td><td>' + esc(r.stage || "-") + '</td><td class="n">' + age(r.age_ms) + '</td><td class="mono">' + esc(r.path) + '</td></tr>'), "") + '</div>';
  }
  if (qs.length) h += '<div class="card mb"><div class="ch"><h2>Queue</h2></div>' + queueTable(qs, true) + '</div>';
  h += '<div class="card"><div class="ch"><h2>Decisions</h2><span class="muted">from the last ' + s.events.length + ' of the pool</span></div>' +
    table([{ t: "time" }, { t: "what" }, { t: "node" }, { t: "detail" }],
      evs.map(e => '<tr><td>' + clock(e.t) + '</td><td>' + kb(e.kind) + '</td><td>' + nodeLink(e.node) + '</td><td class="w">' + esc(e.detail) + '</td></tr>'),
      "no decision about it among them") + '</div>';
  setHTML("cv", h);
}

// decisions
function pageDecisions(s) {
  syncSelect("e-node", s, ui.enode);
  const q = ui.eq.trim().toLowerCase();
  const base_ = s.events.filter(e => (ui.eg === "all" || groupOf(e.kind) === ui.eg) && (!ui.enode || e.node === ui.enode) &&
    (!q || [e.kind, e.conv, e.node, e.detail].some(v => String(v || "").toLowerCase().includes(q))));
  const counts = {};
  base_.forEach(e => { counts[e.kind] = (counts[e.kind] || 0) + 1; });
  const evs = ui.ek ? base_.filter(e => e.kind === ui.ek) : base_;
  setHTML("e-sum", Object.keys(counts).sort((a, b) => counts[b] - counts[a]).map(k => '<button class="chip' + (ui.ek === k ? " on" : "") +
    '" data-ek="' + esc(k) + '" title="' + esc((KIND_INFO[k] || k) + (ui.ek === k ? "; click again for all kinds" : "; click to show only these")) + '">' + kb(k) +
    '<b>' + counts[k] + '</b></button>').join(""));
  setHTML("e-table", table([{ t: "time" }, { t: "what" }, { t: "conversation" }, { t: "node" }, { t: "detail" }],
    evs.map(e => '<tr><td>' + clock(e.t) + '</td><td>' + kb(e.kind) + '</td><td>' + (e.conv ? convLink(e.conv) : '<span class="muted">-</span>') + '</td><td>' +
      nodeLink(e.node) + '</td><td class="w">' + esc(e.detail) + '</td></tr>'),
    s.events.length ? "no decision matches" : "no decision yet"));
  setHTML("e-note", "the last " + s.events.length + " decisions of the pool, the newest first");
}
// a node
function pageNode(s, name) {
  const n = s.nodes.find(x => x.name === name);
  if (!n) { setHTML("nd", '<div class="crumb"><a href="#/">Overview</a></div><div class="card empty">no node ' + esc(name) + ' in this pool</div>'); return; }
  const st = nodeState(n), mi = n.model_info || {}, sx = n.stats || {}, gh = nhist[n.name] || { gen: [], pp: [] };
  let h = '<div class="crumb"><a href="#/">Overview</a></div><div class="ph"><h1>' + esc(n.name) + '</h1>' + (n.self ? '<span class="tag">this router</span>' : '') +
    badges(n) + '<span class="grow"></span>' + drainBtn(n) + '</div>';
  if (st !== "up" && n.error) h += '<div class="err mb">' + esc(n.error) + '</div>';
  h += '<div class="mgrid mb">' +
    metric("gen now", tps(n.gen_tps), "t/s", n.gen_tps > 0 ? "gen" : "zero") + metric("prompt now", tps(n.prompt_tps), "t/s", n.prompt_tps > 0 ? "prompt" : "zero") +
)POOLUI",
R"POOLUI(    metric("busy", busyOf(n), "/ " + (n.slots || 0), "", n.external ? n.external + " from other routers and clients" : "") +
    metric("waiting for it", n.queued_for || 0, "") +
    metric("decode", n.decode_tps > 0 ? tps(n.decode_tps) : "-", n.decode_tps > 0 ? "t/s" : "", "", "one request: the average of its answers of 16 tokens and more") +
    metric("prefill", tps(n.prefill_tps > 0 ? n.prefill_tps : n.prefill_cfg), n.prefill_tps > 0 ? "t/s" : "t/s default", "", "one request: the average of its prompts of move_tokens and more") +
    metric("shared prefix", n.share >= 0 ? Math.round(n.share * 100) + "%" : "-", "", "", "the part of a shared prefix it took from its cache") +
    metric("served", n.requests || 0, "", "", "requests this router sent to it") + '</div>';
  h += '<div class="gn"><div class="stack">';
  h += '<div class="card"><div class="charts">' +
    '<div><div class="chl">generation <b>' + tps(n.gen_tps) + '</b> t/s<span class="r">peak ' + tps(peak(gh.gen)) + '</span></div><canvas class="chart" data-spark="ngen" data-name="' + esc(n.name) + '"></canvas></div>' +
    '<div><div class="chl">prompt <b>' + tps(n.prompt_tps) + '</b> t/s<span class="r">peak ' + tps(peak(gh.pp)) + '</span></div><canvas class="chart" data-spark="npp" data-name="' + esc(n.name) + '"></canvas></div>' +
    '</div><div class="note">the last ' + age(gh.gen.length * (s.config.poll_ms || 1000)) + ' since this page opened</div></div>';
  h += '<div class="card"><div class="ch"><h2>Slots</h2></div>' + table([{ t: "slot" }, { t: "state" }, { t: "context" }, { t: "ctx", n: 1 }, { t: "prompt", n: 1 },
    { t: "computed", n: 1 }, { t: "from cache", n: 1 }, { t: "generated", n: 1 }, { t: "speed", n: 1 }],
    (n.slot_list || []).map(x => {
      const used = (x.n_prompt || 0) + (x.n_decoded || 0), f = x.n_ctx > 0 ? Math.min(1, used / x.n_ctx) : 0;
      const sp = x.state === "gen" ? tps(x.gen_tps) + " t/s" : x.state === "prompt" ? tps(x.prompt_tps) + " t/s" : "-";
      return '<tr><td>' + x.id + '</td><td>' + esc(x.state) + '</td><td><div class="bar ' + (x.state === "gen" || x.state === "prompt" ? x.state : "idle") +
        '" title="' + tok(used) + " of " + tok(x.n_ctx) + '"><i style="width:' + (f * 100).toFixed(1) + '%"></i></div></td><td class="n">' + tok(x.n_ctx) +
        '</td><td class="n">' + tok(x.n_prompt) + '</td><td class="n">' + tok(x.n_processed) + '</td><td class="n">' + tok(x.n_cached) + '</td><td class="n">' +
        tok(x.n_decoded) + '</td><td class="n">' + sp + '</td></tr>';
    }), "no slot information") + '</div>';
  if ((n.running || []).length) {
    h += '<div class="card"><div class="ch"><h2>Running</h2></div>' + table([{ t: "conversation" }, { t: "stage" }, { t: "for", n: 1 }, { t: "request" }],
      n.running.map(r => '<tr><td>' + convLink(r.conv || "#" + r.id) + '</td><td>' + esc(r.stage || "-") + '</td><td class="n">' + age(r.age_ms) +
        '</td><td class="mono">' + esc(r.path) + '</td></tr>'), "") + '</div>';
  }
  const cs = s.conversations.filter(c => c.node === n.name).slice(0, 15);
  h += '<div class="card"><div class="ch"><h2>Conversations on it</h2><span class="muted">' + s.conversations.filter(c => c.node === n.name).length +
    ' of the recent ones</span></div>' + table([{ t: "conversation" }, { t: "msgs", n: 1 }, { t: "prompt", n: 1 }, { t: "cached there", n: 1 }, { t: "last", n: 1 }],
      cs.map(c => '<tr><td>' + (c.active ? '<span class="adot"></span>' : '<span class="idot"></span>') + convLink(c.conv) + '</td><td class="n">' + c.messages +
        '</td><td class="n">~' + tok(c.prompt_tokens) + '</td><td class="n">' + (c.cached_tokens > 0 ? "~" + tok(c.cached_tokens) : "-") + '</td><td class="n">' +
        age(c.age_ms) + '</td></tr>'), "none") + '</div>';
  const evs = s.events.filter(e => e.node === n.name).slice(0, 25);
  h += '<div class="card"><div class="ch"><h2>Decisions about it</h2></div>' + (evs.length ? '<div class="feed">' + evs.map(e => feedRow(e, true)).join("") + '</div>' :
    '<div class="empty">none among the last ' + s.events.length + '</div>') + '</div>';
  h += '</div><div class="stack">';
  const kv = [
    ["model", esc(n.model || "-")], ["type", esc(mi.ftype || "-")], ["parameters", mi.n_params ? count(mi.n_params) : "-"],
    ["size", mi.size ? size(mi.size) : "-"], ["vocabulary", mi.n_vocab ? tok(mi.n_vocab) : "-"],
    ["context", (n.n_ctx ? tok(n.n_ctx) : "-") + (mi.n_ctx_train ? " (trained " + tok(mi.n_ctx_train) + ")" : "")],
    ["slots", String(n.slots || 0)], ["weight", n.weight != null ? n.weight.toFixed(2) + ' <span class="muted">share of new conversations against the fastest node</span>' : "-"],
    ["moves state", yesno(n.migrate)], ["cache", n.signature ? '<span class="mono">' + esc(n.signature) + '</span>' : "-"],
    ["address", '<span class="mono">' + esc(n.url) + '</span>' + (n.fallback ? ' <span class="badge wa">fallback</span>' : '')],
    ["last answer", n.seen_ms >= 0 ? age(n.seen_ms) + (n.seen_ms >= 1000 ? " ago" : "") : "never"]];
  h += '<div class="card"><div class="ch"><h2>Model and node</h2></div><dl class="kv">' + kv.map(r => '<dt>' + r[0] + '</dt><dd>' + r[1] + '</dd>').join("") + '</dl></div>';
  const mem = (n.devices || []).map(d => memRow(d, true)).join("") + roomRow(n, true);
  h += '<div class="card"><div class="ch"><h2>Memory</h2></div>' + (mem || '<div class="empty">not reported</div>') + '</div>';
  const gs = gpusOf(n);
  if (gs.length) {
    h += '<div class="card"><div class="ch"><h2>GPU</h2></div>' + table([{ t: "device" }, { t: "temperature", n: 1 }, { t: "power", n: 1 }, { t: "fan", n: 1 }],
      gs.map(g => '<tr><td>' + esc(g.name) + '</td><td class="n">' + (g.temp_c != null ? g.temp_c.toFixed(0) + " \u00b0C" : "-") + '</td><td class="n">' +
        (g.power_pct != null ? g.power_pct.toFixed(0) + "%" : g.power_w != null ? g.power_w.toFixed(0) + " W" : "-") + '</td><td class="n">' +
        (g.fan_rpm != null ? g.fan_rpm + " rpm" : "-") + '</td></tr>'), "") + '</div>';
  }
  const mb = x => x > 0 ? (x / 8).toFixed(0) + " MB/s" : "-";
  h += '<div class="card"><div class="ch"><h2>Links</h2></div>' + table([{ t: "address" }, { t: "" }, { t: "from it", n: 1 }, { t: "to it", n: 1 }],
    (n.links || []).map(l => '<tr><td class="mono">' + esc(l.url) + (l.error ? '<div class="err">' + esc(l.error) + '</div>' : '') + '</td><td>' +
      (l.active ? '<span class="badge ok">in use</span>' : l.error ? '<span class="badge ba">error</span>' : '<span class="badge">standby</span>') +
      '</td><td class="n">' + mb(l.mbps_read) + '</td><td class="n">' + mb(l.mbps_write) + '</td></tr>'), "no link") + '</div>';
  if (sx.uptime_s > 0) {
    const avg = (k, t) => sx[t] > 0 ? tps(sx[k] / sx[t]) + " t/s" : "-";
    const share = sx.cached_tokens + sx.prompt_tokens > 0 ? Math.round(100 * sx.cached_tokens / (sx.cached_tokens + sx.prompt_tokens)) + "%" : "-";
    h += '<div class="card"><div class="ch"><h2>Work since the model loaded</h2><span class="muted">' + age(sx.uptime_s * 1000) + ' ago</span></div><dl class="kv">' +
      '<dt>prompt computed</dt><dd><b>' + tok(sx.prompt_tokens) + '</b> tok' + DOT + avg("prompt_tokens", "prompt_s") + '</dd>' +
      '<dt>generated</dt><dd><b>' + tok(sx.gen_tokens) + '</b> tok' + DOT + avg("gen_tokens", "gen_s") + '</dd>' +
      '<dt>from cache</dt><dd><b>' + tok(sx.cached_tokens) + '</b> tok' + DOT + share + ' of the prompt tokens</dd></dl></div>';
  }
  h += '</div></div>';
  setHTML("nd", h);
}
// the configuration of the pool
function pageConfig(s) {
  const c = s.config;
  const rows = [
    ["router", esc(s.name) + ' <span class="muted mono">' + esc(s.id) + '</span>', "this router in the pool (pool-name, else the host name)"],
    ["model", esc(c.model) + (c.names && c.names.length ? ' <span class="muted">' + esc(c.names.join(", ")) + '</span>' : ''), "the model of this pool and the names it takes"],
    ["move-tokens", String(c.move_tokens), "a conversation with this many tokens of its own waits for its busy node, a shorter one moves at once"],
    ["wait-max", c.wait_max + " s", "the longest wait for the node of a conversation"],
    ["migrate", yesno(c.migrate), "a conversation moves with its cached state when the nodes can" + (s.state_kib_per_token > 0 ? " (~" + s.state_kib_per_token.toFixed(0) + " KiB per token)" : "")],
    ["prefill-tps", tps(c.prefill_tps) + " t/s", "the prefill speed a wait is weighed against until a node has measured its own"],
    ["link-mbps", c.link_mbps + " Mbit/s", "the speed of a link not measured yet"],
    ["rebalance-s", c.rebalance_s ? c.rebalance_s + " s" : "never", "a conversation leaves a crowded node when the move costs at most this"],
    ["quiet-wait-s", c.quiet_wait_s ? c.quiet_wait_s + " s" : "no wait", "the longest wait of a long prompt for a node where nothing runs"],
    ["short-tokens", c.short_tokens ? String(c.short_tokens) : "none", "a request for at most this many answer tokens goes ahead in the queue"],
    ["evict-idle-s", c.evict_idle_s ? c.evict_idle_s + " s" : "any idle", "a load the pool asks for unloads on that router only a model idle this long"],
    ["same-model", yesno(c.same_model), "a node of another model than the most of the others takes no request"],
    ["poll-ms", c.poll_ms + " ms", "how often each node is asked for its health and slots"],
    ["discover", yesno(c.discover) + (c.discover ? DOT + "UDP port " + c.port : ""), "the routers find each other by broadcasts"],
    ["peers", c.peers && c.peers.length ? '<span class="mono">' + esc(c.peers.join(", ")) + '</span>' : "none", "routers of the pool added without discovery"]];
  const pools = s.pools && s.pools.length > 1 ? '<div class="card mb"><div class="ch"><h2>Pools of this router</h2></div><div class="chips">' +
    s.pools.map(m => '<a class="chip" href="' + poolHref(m) + '">' + (m === c.model ? "<b>" + esc(m) + "</b>" : esc(m)) + '</a>').join("") + '</div></div>' : "";
  setHTML("cfg", pools + '<div class="card cfg"><div class="ch"><h2>Options</h2><span class="muted">pool-* keys in models.ini</span></div>' +
    table([{ t: "option" }, { t: "value" }, { t: "what it does" }],
      rows.map(r => '<tr><td>' + r[0] + '</td><td class="v">' + r[1] + '</td><td class="w">' + esc(r[2]) + '</td></tr>'), "") + '</div>');
}
function poolHref(m) {
  const q = new URLSearchParams(location.search);
  q.set("model", m);
  return "?" + q.toString() + location.hash;
}

// pages: the parts that do not change with the data (inputs) are built once per page
function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const arg = parts.length > 1 ? decodeURIComponent(parts.slice(1).join("/")) : "";
  if ((parts[0] === "node" || parts[0] === "conv") && arg) return { page: parts[0], arg };
  if (["conversations", "decisions", "config"].includes(parts[0])) return { page: parts[0], arg: "" };
  return { page: "overview", arg: "" };
}
let route = parseRoute();
let built = "";
function skeleton(r) {
  if (r.page === "overview") {
    return '<section class="kpis" id="kpis"></section><div id="ov-queue"></div><section class="nodes" id="nodes"></section><section class="cols2">' +
      '<div class="card"><div class="ch"><h2>Conversations</h2><span class="muted" id="ov-cn"></span><a class="more" href="#/conversations">all conversations</a></div><div id="ov-convs"></div></div>' +
      '<div class="card"><div class="ch"><h2>Decisions</h2><span class="muted">routing and nodes</span><a class="more" href="#/decisions">all decisions</a></div><div id="ov-events"></div></div></section>';
  }
  if (r.page === "conversations") {
    return '<div class="toolbar"><input type="search" placeholder="conversation or node" data-ui="cq" value="' + esc(ui.cq) + '">' +
      '<select id="c-node" data-ui="cnode"></select><label class="chk"><input type="checkbox" data-ui="cactive"' + (ui.cactive ? " checked" : "") +
      '> running only</label><span class="grow"></span><span class="muted" id="c-count"></span></div><div id="c-queue"></div><div class="card"><div id="c-table"></div></div>';
  }
  if (r.page === "decisions") {
    const seg = [["all", "All"], ["routing", "Routing"], ["requests", "Requests"], ["nodes", "Nodes"]];
    return '<div class="toolbar"><span class="seg">' + seg.map(g => '<button data-eg="' + g[0] + '"' + (ui.eg === g[0] ? ' class="on"' : '') + '>' + g[1] + '</button>').join("") +
      '</span><input type="search" placeholder="kind, conversation, node, detail" data-ui="eq" value="' + esc(ui.eq) + '"><select id="e-node" data-ui="enode"></select>' +
      '<span class="grow"></span><span class="muted" id="e-note"></span></div><div class="chips mb" id="e-sum"></div><div class="card"><div id="e-table"></div></div>';
  }
  if (r.page === "node") return '<div id="nd"></div>';
  if (r.page === "conv") return '<div id="cv"></div>';
  return '<div id="cfg"></div>';
}
function header(s) {
  const c = s.config;
  setHTML("h-model", esc(c.model || ""));
  const pools = s.pools && s.pools.length > 1 ? s.pools : [];
  setHTML("h-pools", pools.length ? '<select id="pool-sel" title="the pools of this router">' + pools.map(m => '<option value="' + esc(m) + '"' +
    (m === c.model ? " selected" : "") + '>' + esc(m) + '</option>').join("") + '</select>' : "");
  $("h-model").hidden = pools.length > 0;
  const nq = $("n-queue");
  nq.hidden = !s.pool.queued;
  nq.className = "cnt wa";
  nq.textContent = s.pool.queued;
  $("n-conv").textContent = s.pool.conversations;
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.p === (route.page === "node" ? "overview" : route.page === "conv" ? "conversations" : route.page)));
  document.title = (c.model ? c.model + " - " : "") + "llama pool";
}
const PAGES = { overview: pageOverview, conversations: pageConversations, decisions: pageDecisions, config: pageConfig,
  node: s => pageNode(s, route.arg), conv: s => pageConv(s, route.arg) };
function render() {
  if (!S) return;
  const sig = route.page + "/" + route.arg;
  if (built !== sig) {
    const v = $("view");
    v.innerHTML = skeleton(route);
    built = sig;
  }
  header(S);
  PAGES[route.page](S);
  drawAll();
}

// sparklines: each series on its own scale, the newest point at the right edge
function rgba(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return hex;
  const v = parseInt(m[1], 16);
  return "rgba(" + (v >> 16) + "," + ((v >> 8) & 255) + "," + (v & 255) + "," + a + ")";
}
function draw(cv, series) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  const span = Math.max(60, ...series.map(sr => (sr.data || []).length));
  const step = (w - 2) / (span - 1);
  for (const sr of series) {
    const d = sr.data;
    if (!d || d.length < 2) continue;
    const max = Math.max(1, ...d);
)POOLUI",
R"POOLUI(    const x0 = 1 + (span - d.length) * step;
    c.beginPath();
    d.forEach((v, i) => { const x = x0 + i * step, y = h - 1.5 - (v / max) * (h - 4); if (i) c.lineTo(x, y); else c.moveTo(x, y); });
    c.lineWidth = sr.fill ? 1.6 : 1.3; c.lineJoin = "round"; c.strokeStyle = sr.color; c.stroke();
    if (sr.fill) {
      c.lineTo(x0 + (d.length - 1) * step, h); c.lineTo(x0, h); c.closePath();
      const g = c.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, rgba(sr.color, 0.26)); g.addColorStop(1, rgba(sr.color, 0.02));
      c.fillStyle = g; c.fill();
    }
  }
}
function drawAll() {
  const gen = cssv("--gen"), pp = cssv("--prompt");
  document.querySelectorAll("canvas[data-spark]").forEach(cv => {
    const k = cv.dataset.spark, nh = nhist[cv.dataset.name] || { gen: [], pp: [] };
    if (k === "pool-gen") draw(cv, [{ data: hist.gen, color: gen, fill: true }]);
    else if (k === "pool-pp") draw(cv, [{ data: hist.pp, color: pp, fill: true }]);
    else if (k === "node") draw(cv, [{ data: nh.gen, color: gen, fill: true }, { data: nh.pp, color: pp }]);
    else if (k === "ngen") draw(cv, [{ data: nh.gen, color: gen, fill: true }]);
    else if (k === "npp") draw(cv, [{ data: nh.pp, color: pp, fill: true }]);
  });
}

// events
window.addEventListener("hashchange", () => { route = parseRoute(); render(); window.scrollTo(0, 0); });
window.addEventListener("resize", () => drawAll());
function onInput(ev) {
  const t = ev.target;
  if (t.id === "pool-sel") { location.href = poolHref(t.value); return; }
  if (!t.dataset || !t.dataset.ui) return;
  ui[t.dataset.ui] = t.type === "checkbox" ? t.checked : t.value;
  render();
}
document.addEventListener("input", onInput);
document.addEventListener("change", onInput);
document.addEventListener("click", ev => {
  const b = ev.target.closest("button[data-node]");
  if (b) {
    fetch(base + "/pool/drain", {
      method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, H),
      body: JSON.stringify({ node: b.dataset.node, drain: b.dataset.drain === "1" })
    }).then(() => tick(true));
    return;
  }
  const kc = ev.target.closest("button[data-ek]");
  if (kc) {
    ui.ek = ui.ek === kc.dataset.ek ? "" : kc.dataset.ek;
    render();
    return;
  }
  const g = ev.target.closest("button[data-eg]");
  if (g) {
    ui.eg = g.dataset.eg;
    g.parentNode.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === g));
    render();
    return;
  }
  const th = ev.target.closest("th[data-sort]");
  if (th) {
    const k = th.dataset.sort;
    if (ui.csort === k) ui.cdir = -ui.cdir; else { ui.csort = k; ui.cdir = 1; }
    render();
  }
});
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  $("dot").className = paused ? "dot paused" : "dot";
  $("live").textContent = paused ? "paused" : "resuming";
  if (!paused) tick(true);
});

async function tick(force) {
  if (busy || (paused && force !== true)) return;
  busy = true;
  try {
    const r = await fetch(base + "/pool/status" + (model ? "?model=" + encodeURIComponent(model) : ""), { headers: H, cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const s = await r.json();
    push(hist.gen, s.pool.gen_tps);
    push(hist.pp, s.pool.prompt_tps);
    for (const n of s.nodes) {
      const h = nhist[n.name] || (nhist[n.name] = { gen: [], pp: [] });
      push(h.gen, n.gen_tps);
      push(h.pp, n.prompt_tps);
    }
    S = s;
    if (!paused) {
      render();
      $("dot").className = "dot";
      $("live").textContent = "live" + DOT + new Date().toTimeString().slice(0, 8);
    }
  } catch (e) {
    $("dot").className = "dot off";
    $("live").textContent = "no answer from the pool: " + e.message;
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
