// mlws-ocr workbench: one page, every stage, correctable. Vanilla JS, no build step.
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
};

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

// ------------------------------------------------------------------ state
const S = {
  st: null,            // server state
  sel: null,           // selected stage index
  imgName: null,       // which image of the stage is shown
  img: null, imgScale: 1, imgKey: "",
  layout: null, layoutKey: "",
  finalLayout: null, finalKey: "",
  view: { z: 1, x: 0, y: 0 },
  tool: null,          // active tool id
  layers: { blocks: true, lines: true, words: true, tables: true, zones: true, rules: false },
  draft: null,         // box editor draft: {kind, items:[{box,...}], sel:Set, order:[]}
  drag: null,
  pts: [],             // deskew horizon points (page coords)
  selWord: null,
  hover: null,
  resultMode: "text",   // text | hocr
  result: null, resultKey: "",
};
const CLEANUP = new Set(["magnify", "deskew", "illumination", "binarize", "despeckle"]);
const LAYOUT = new Set(["imagezones", "rulings", "blocks", "tables", "lines", "components"]);
const READ = new Set(["recognize", "decode", "adapt", "chop", "correct", "output"]);
const RESULT = "result";   // the last tab: the extracted text, as plain text or hOCR

const canvas = $("canvas"), ctx = canvas.getContext("2d");

// ------------------------------------------------------------------ header
async function init() {
  const configs = await api("/api/configs");
  for (const c of configs) $("config").append(el("option", { value: "configs/" + c }, c.replace(".toml", "")));
  $("config").value = "configs/neural.toml";
  $("openBtn").onclick = () => openPath($("path").value.trim());
  $("path").addEventListener("keydown", (e) => { if (e.key === "Enter") openPath($("path").value.trim()); });
  $("path").addEventListener("input", () => { $("pdfPageBox").hidden = !/\.pdf$/i.test($("path").value.trim()); });
  $("upload").onchange = uploadFile;
  $("browseBtn").onclick = () => browse($("path").value ? $("path").value.replace(/\/[^/]*$/, "") : "");
  $("browserClose").onclick = () => $("browser").close();
  $("saveBtn").onclick = saveSession;
  $("loadBtn").onclick = loadSession;
  $("exportSel").onchange = (e) => { if (e.target.value) window.location = "/api/export/" + e.target.value; e.target.value = ""; };
  window.addEventListener("resize", draw);
  setupCanvas();
  await poll(true);
  setInterval(() => poll(false), 700);
}

async function openPath(p) {
  if (!p) return status("type or browse an image path");
  try {
    $("pdfPageBox").hidden = !/\.pdf$/i.test(p);
    await api("/api/open", { path: p, config: $("config").value, doc_type: $("doctype").value,
                             pdf_page: /\.pdf$/i.test(p) ? Math.max(0, Number($("pdfPage").value) - 1) : 0 });
    S.sel = null; S.imgKey = ""; S.layoutKey = ""; S.finalKey = ""; S.draft = null; fitNext = true;
    status("reading…");
    await poll(true);
  } catch (e) { status("open failed: " + e.message, true); }
}

async function uploadFile(e) {
  const f = e.target.files[0]; if (!f) return;
  status("uploading…");
  const r = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}&config=${encodeURIComponent($("config").value)}&doc_type=${$("doctype").value}`,
    { method: "POST", body: f });
  const j = await r.json();
  if (!r.ok) return status("upload failed: " + j.error, true);
  $("path").value = j.path; S.sel = null; S.imgKey = ""; fitNext = true; await poll(true);
}

async function browse(dir) {
  const d = await api("/api/browse?dir=" + encodeURIComponent(dir || ""));
  $("browserDir").textContent = d.dir;
  const list = $("browserList"); list.innerHTML = "";
  list.append(el("li", { class: "dir", onclick: () => browse(d.parent) }, ".."));
  for (const x of d.dirs) list.append(el("li", { class: "dir", onclick: () => browse(d.dir + "/" + x) }, x));
  for (const x of d.files) list.append(el("li", { onclick: () => { $("path").value = d.dir + "/" + x; $("browser").close(); openPath($("path").value); } }, x));
  if (!$("browser").open) $("browser").showModal();
}

async function saveSession() {
  if (!S.st || !S.st.open) return;
  const path = prompt("Save the session to (a .mlws.json file):", S.st.image.replace(/\.[^.]+$/, "") + ".mlws.json");
  if (!path) return;
  try { const r = await api("/api/save", { path }); status("saved " + r.path); } catch (e) { status(e.message, true); }
}

async function loadSession() {
  const path = prompt("Load a session file (.mlws.json):");
  if (!path) return;
  try {
    const r = await api("/api/load", { path });
    S.sel = null; S.imgKey = ""; fitNext = true;
    status(r.image_changed ? "loaded — note: the image file has changed since it was saved" : "loaded", r.image_changed);
    await poll(true);
  } catch (e) { status(e.message, true); }
}

function status(msg, bad) { $("status").textContent = msg; $("status").style.color = bad ? "var(--bad)" : ""; }

// ------------------------------------------------------------------ polling
let fitNext = true;
async function poll(force) {
  let st;
  try { st = await api("/api/state"); } catch { return; }
  if (!st.open) { S.st = st; return; }
  const busy = st.stages.some((s) => s.status === "running" || s.status === "pending");
  const changed = force || !S.st || JSON.stringify(S.st.stages.map((s) => [s.status, s.impl, s.ms])) !==
                  JSON.stringify(st.stages.map((s) => [s.status, s.impl, s.ms]));
  S.st = st;
  if (!changed) return;
  const total = st.stages.reduce((a, s) => a + (s.status === "done" ? s.ms : 0), 0);
  const running = st.stages.find((s) => s.status === "running");
  status(running ? `running ${running.slot}.${running.impl}…` :
         st.stages.some((s) => s.status === "error") ? "a stage failed" :
         busy ? "queued…" : `done in ${(total / 1000).toFixed(1)} s`);
  $("path").value ||= st.image;
  // the header shows the profile this page was READ with (it also picks the profile for the next Open)
  const cfg = "configs/" + st.config.split("/").pop();
  if ([...$("config").options].some((o) => o.value === cfg) && S.shownConfig !== cfg) { $("config").value = cfg; S.shownConfig = cfg; }
  renderPhases();
  if (S.sel == null) { S.sel = defaultStage(); }
  if (S.sel === RESULT) await showResult(); else await showStage(S.sel, false);
  await refreshText();
}

function defaultStage() {
  const st = S.st.stages;
  const out = st.length - 1;
  return st[out].status === "done" ? out : st.findIndex((s) => s.slot === "deskew");
}

// ------------------------------------------------------------------ phase list
function renderPhases() {
  const nav = $("phases"); nav.innerHTML = "";
  S.st.stages.forEach((s, k) => {
    const n = s.edits.length || (s.params.angle_deg != null ? 1 : 0);
    nav.append(el("div", { class: "phase" + (k === S.sel ? " sel" : ""), onclick: () => selectStage(k) },
      el("span", { class: "dot " + s.status }),
      el("span", {}, el("div", { class: "name" }, s.slot), el("div", { class: "impl" }, s.impl),
        n ? el("div", { class: "edited" }, "✎ corrected") : null),
      el("span", { class: "ms" }, s.status === "done" ? fmtMs(s.ms) : s.status === "error" ? "error" : "")));
  });
  const last = S.st.stages[S.st.stages.length - 1];
  nav.append(el("div", { class: "phase result" + (S.sel === RESULT ? " sel" : ""), onclick: () => selectStage(RESULT) },
    el("span", { class: "dot " + last.status }),
    el("span", {}, el("div", { class: "name" }, "result"), el("div", { class: "impl" }, "extracted text · hOCR")),
    el("span", { class: "ms" }, "")));
}
const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";

async function selectStage(k) {
  if (S.sel !== k) { S.tool = null; S.draft = null; S.pts = []; S.imgName = null; S.selWord = null; }
  S.sel = k; renderPhases();
  const isResult = k === RESULT;
  $("canvasWrap").hidden = isResult; $("resultPane").hidden = !isResult;
  $("textBox").hidden = isResult; $("params").parentElement.hidden = isResult;
  if (isResult) await showResult(); else { await showStage(k, true); draw(); }
}

// ------------------------------------------------------------------ the result tab
async function showResult() {
  const st = S.st.stages, last = st[st.length - 1];
  $("canvasWrap").hidden = true; $("resultPane").hidden = false; $("textBox").hidden = true;
  $("params").parentElement.hidden = true;
  const key = `${last.status}/${last.ms}/${last.edits.length}`;
  if (key !== S.resultKey) { S.result = await api("/api/result").catch(() => null); S.resultKey = key; }
  const R = S.result;
  // toolbar: text | hOCR, copy, download
  const tb = $("toolbar"); tb.innerHTML = "";
  const seg = el("span", { class: "seg" },
    el("button", { class: S.resultMode === "text" ? "on" : "", onclick: () => { S.resultMode = "text"; showResult(); } }, "Text"),
    el("button", { class: S.resultMode === "hocr" ? "on" : "", onclick: () => { S.resultMode = "hocr"; showResult(); } }, "hOCR"),
    el("button", { class: S.resultMode === "render" ? "on" : "", onclick: () => { S.resultMode = "render"; showResult(); } }, "Rendered"));
  tb.append(seg, el("span", { class: "sep" }),
    S.resultMode === "render" ? el("label", {}, el("input", { type: "checkbox", ...(S.renderScan ? { checked: "" } : {}),
      onchange: (e) => { S.renderScan = e.target.checked; showResult(); } }), "scan underneath") : null,
    S.resultMode === "render" ? el("label", {}, el("input", { type: "checkbox", ...(S.renderBoxes !== false ? { checked: "" } : {}),
      onchange: (e) => { S.renderBoxes = e.target.checked; showResult(); } }), "structure") : null,
    el("button", { onclick: async () => { await navigator.clipboard.writeText(S.resultMode === "text" ? R.text : R.hocr); status("copied"); } }, "Copy"),
    el("button", { onclick: () => { window.location = "/api/export/" + (S.resultMode === "text" ? "text" : "hocr"); } },
      S.resultMode === "text" ? "Download .txt" : "Download .hocr"));
  // side panel: a summary of the page
  $("stageHead").innerHTML = ""; $("tools").innerHTML = ""; $("params").innerHTML = "";
  $("stageHead").append(el("h2", {}, "Result"), el("div", {}, S.resultMode === "text"
    ? "The page's text, as the output stage wrote it (reading order, table rows aligned)."
    : S.resultMode === "hocr"
    ? "hOCR: the page's structure — blocks in reading order, lines, words with boxes and confidence, tables, images, rulings."
    : "The page redrawn from the hOCR file alone: every word at its box, blocks numbered in reading order, tables, images and rulings. Red words are low-confidence, green ones were corrected; hover for the confidence."));
  const pre = $("resultText");
  if (!R || !R.ready) { pre.className = ""; pre.textContent = "(the page is still being read)"; $("scalars").innerHTML = ""; return; }
  const sm = R.summary, t = el("table");
  for (const [k, v] of [["words", sm.words], ["lines", sm.lines], ["characters", sm.characters],
                        ["mean confidence", sm.mean_confidence], ["low-confidence words", sm.low_confidence_words],
                        ["corrected by the dictionary pass", sm.corrected_words], ["corrected by hand", sm.edited_words]])
    t.append(el("tr", {}, el("td", {}, k), el("td", {}, v == null ? "—" : String(v))));
  $("scalars").innerHTML = ""; $("scalars").append(t);
  $("resultRender").hidden = S.resultMode !== "render"; pre.hidden = S.resultMode === "render";
  if (S.resultMode === "text") { pre.className = ""; pre.textContent = R.text; return; }
  if (S.resultMode === "render") { renderHocr(R.hocr); return; }
  pre.className = "hocr"; pre.innerHTML = "";
  // light highlighting: tags muted/accent, word text bold
  for (const line of R.hocr.split("\n")) {
    const m = line.match(/^(<span class="ocrx_word"[^>]*>)(.*)(<\/span>)$/);
    if (m) pre.append(el("span", { class: "tag" }, m[1]), el("span", { class: "t" }, decodeEntities(m[2])), el("span", { class: "tag" }, m[3]), "\n");
    else pre.append(el("span", { class: line.startsWith("<") ? "tag" : "" }, line), "\n");
  }
}
function decodeEntities(s) { const d = document.createElement("textarea"); d.innerHTML = s; return d.value; }

// ------------------------------------------------------------------ one stage
function stageKind(slot) { return CLEANUP.has(slot) ? "cleanup" : LAYOUT.has(slot) ? "layout" : "read"; }
function defaultImage(s) {
  if (["magnify", "deskew", "illumination"].includes(s.slot)) return "gray";
  if (READ.has(s.slot)) return "gray";
  return "binary";
}

async function showStage(k, userAction) {
  const s = S.st.stages[k];
  renderHead(s, k);
  renderParams(s, k);
  renderScalars(s);
  if (!S.imgName) S.imgName = defaultImage(s);
  // the stage's own layout first: the box editor builds its draft from it
  const lkey = `${k}/${s.ms}`;
  if (s.status === "done" && lkey !== S.layoutKey) {
    S.layout = await api(`/api/layout/${k}`).catch(() => null); S.layoutKey = lkey;
    if (S.draft && !S.draft.dirty) S.draft = null;
  } else if (s.status !== "done") { S.layout = null; S.layoutKey = ""; }
  renderToolbar(s, k);
  renderTools(s, k);
  if (s.status !== "done") { draw(); return; }
  const want = S.tool === "horizon" ? "before" : S.imgName;
  const key = `${k}/${want}/${s.ms}`;
  if (key !== S.imgKey) {
    S.imgScale = Math.min(1, 2200 / Math.max(S.st.shape[0], S.st.shape[1]));
    const img = new Image();
    img.src = `/api/image/${k}/${want}.png?scale=${S.imgScale}&t=${s.ms}`;
    await img.decode().catch(() => null);
    S.img = img; S.imgKey = key;
    if (fitNext) { fit(); fitNext = false; }
  }
  draw();
}

function renderHead(s, k) {
  const h = $("stageHead"); h.innerHTML = "";
  h.append(el("h2", {}, `${k + 1}. ${s.slot}`));
  const sel = el("select", { title: "algorithm for this stage", onchange: async (e) => {
    await api(`/api/stage/${k}`, { impl: e.target.value }); S.imgName = null; await poll(true); } });
  for (const c of s.choices) sel.append(el("option", { value: c, ...(c === s.impl ? { selected: "" } : {}) }, c));
  h.append(el("label", {}, "Algorithm ", sel));
  h.append(el("div", { class: "row" },
    el("button", { onclick: async () => { await api("/api/run", { from: k }); await poll(true); } }, "Re-run from here")));
  if (s.error) h.append(el("div", { class: "err" }, s.error));
}

function renderParams(s, k) {
  const f = $("params"); f.innerHTML = "";
  const names = Object.keys(s.defaults);
  if (!names.length) { f.append(el("div", { class: "empty" }, "no parameters")); return; }
  for (const p of names) {
    const d = s.defaults[p], v = p in s.params ? s.params[p] : d;
    let input;
    if (typeof d === "boolean") input = el("input", { type: "checkbox", name: p, ...(v ? { checked: "" } : {}) });
    else if (typeof d === "number") input = el("input", { type: "number", name: p, step: "any", value: v ?? "" });
    else input = el("input", { type: "text", name: p, value: v == null ? "" : typeof v === "object" ? JSON.stringify(v) : v, placeholder: d == null ? "auto" : "" });
    f.append(el("label", { class: "row" + (JSON.stringify(v) !== JSON.stringify(d) ? " changed" : ""), title: `default: ${JSON.stringify(d)}` },
      el("span", {}, p), input));
  }
  f.append(el("div", { class: "actions" },
    el("button", { type: "submit", class: "primary" }, "Apply & re-run"),
    el("button", { type: "button", onclick: async () => {
      await api(`/api/stage/${k}`, { params: Object.fromEntries(names.map((p) => [p, s.defaults[p]])) }); await poll(true); } }, "Defaults")));
  f.onsubmit = async (e) => {
    e.preventDefault();
    const params = {};
    for (const p of names) {
      const i = f.elements[p], d = s.defaults[p];
      if (typeof d === "boolean") params[p] = i.checked;
      else if (typeof d === "number") params[p] = i.value === "" ? d : Number(i.value);
      else if (d == null) params[p] = i.value === "" ? null : (isNaN(Number(i.value)) ? i.value : Number(i.value));
      else { try { params[p] = typeof d === "object" ? JSON.parse(i.value) : i.value; } catch { params[p] = i.value; } }
    }
    try { await api(`/api/stage/${k}`, { params }); await poll(true); } catch (err) { status(err.message, true); }
  };
}

function renderScalars(s) {
  const d = $("scalars"); d.innerHTML = "";
  const t = el("table");
  t.append(el("tr", {}, el("td", {}, "time"), el("td", {}, s.status === "done" ? fmtMs(s.ms) : s.status)));
  for (const [k, v] of Object.entries(s.scalars || {}))
    t.append(el("tr", {}, el("td", {}, k), el("td", {}, typeof v === "number" ? +v.toFixed(4) : String(v))));
  d.append(t);
  if (s.notes && s.notes.length) d.append(el("div", { class: "notes" }, s.notes.join(" · ")));
}

// ------------------------------------------------------------------ toolbar & tools
function renderToolbar(s, k) {
  const tb = $("toolbar"); tb.innerHTML = "";
  const imgs = ["gray", "binary", "before", ...s.images];
  const sel = el("select", { onchange: (e) => { S.imgName = e.target.value; showStage(k); } });
  for (const n of imgs) sel.append(el("option", { value: n, ...(n === S.imgName ? { selected: "" } : {}) }, n === "before" ? "before this stage" : n));
  tb.append(el("label", {}, "Show", sel));
  tb.append(el("span", { class: "sep" }));
  for (const [layer, label] of [["blocks", "blocks"], ["lines", "lines"], ["words", "words"], ["tables", "tables"], ["zones", "images"], ["rules", "rulings"]]) {
    const cb = el("input", { type: "checkbox", ...(S.layers[layer] ? { checked: "" } : {}), onchange: (e) => { S.layers[layer] = e.target.checked; draw(); } });
    tb.append(el("label", {}, cb, label));
  }
  tb.append(el("span", { class: "sep" }));
  tb.append(el("button", { onclick: () => { fit(); draw(); } }, "Fit"));
  tb.append(el("button", { onclick: () => { zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1 / S.view.z / S.imgScale); } }, "1:1"));
}

function toolButton(id, label, title) {
  return el("button", { class: S.tool === id ? "on" : "", title, onclick: () => { S.tool = S.tool === id ? null : id; S.pts = [];
    canvas.classList.toggle("tool", !!S.tool); showStage(S.sel); } }, label);
}

function renderTools(s, k) {
  const t = $("tools"); t.innerHTML = "";
  $("hint").textContent = "";
  if (s.slot === "deskew") return deskewTools(t, s, k);
  if (s.slot === "despeckle") return noiseTools(t, s, k);
  if (s.slot === "blocks") return boxTools(t, s, k, "blocks");
  if (s.slot === "lines") return boxTools(t, s, k, "lines");
  if (s.slot === "correct") return correctTools(t, s, k);
  if (READ.has(s.slot)) return wordTools(t, s, k);
}

function deskewTools(t, s, k) {
  const applied = s.scalars.correction_deg, est = s.scalars.estimated_skew_deg;
  const inp = el("input", { type: "number", step: "0.05", value: applied ?? 0, style: "width:80px" });
  t.append(el("div", { class: "box" },
    el("div", {}, `Rotation applied: ${applied ?? "—"}°  ${s.scalars.manual ? "(manual)" : "(estimated)"}`),
    el("div", {}, `The stage's own estimate of the skew: ${est ?? "—"}°`),
    el("div", { class: "row" }, "Set angle", inp,
      el("button", { class: "primary", onclick: async () => { await api(`/api/stage/${k}`, { params: { angle_deg: Number(inp.value) } }); await poll(true); } }, "Apply"),
      el("button", { onclick: async () => { await api(`/api/stage/${k}`, { params: { angle_deg: null } }); await poll(true); } }, "Use estimate")),
    el("div", { class: "row" }, toolButton("horizon", "Draw a horizon", "click two points along a text line on the unrotated page"))));
  if (S.tool === "horizon") $("hint").textContent = "Click two points along one line of text; the page is rotated to level it.";
}

function noiseTools(t, s, k) {
  t.append(el("div", { class: "box" },
    el("div", {}, `Specks removed: ${s.scalars.components_removed ?? "—"} (min area ${s.scalars.min_area_px ?? "—"} px). Tune min_area_300dpi below, or pick by hand:`),
    el("div", { class: "row" },
      toolButton("erase_at", "Erase speck", "click a mark to remove it"),
      toolButton("restore_at", "Restore speck", "click where the stage removed a mark (show: removed_overlay)"),
      toolButton("erase", "Erase box", "drag a box; all ink in it is cleared")),
    editList(s, k)));
  if (S.tool === "restore_at" && S.imgName !== "removed_overlay" && s.images.includes("removed_overlay")) { S.imgName = "removed_overlay"; showStage(k); }
  const hints = { erase_at: "Click a mark to erase it.", restore_at: "Click a removed speck (shown in colour) to restore it.", erase: "Drag a box to erase everything inside." };
  $("hint").textContent = hints[S.tool] || "";
}

function editList(s, k) {
  const box = el("div", {});
  box.append(el("div", { class: "edits" }, s.edits.length ? s.edits.map((e, i) => el("div", {}, `${i + 1}. ${e.op} ${JSON.stringify(e.box || e.point || (e.text ? `"${e.text}"` : ""))}`)) : "no corrections yet"));
  box.append(el("div", { class: "row" },
    el("button", { onclick: () => pushEdits(k, s.edits.slice(0, -1)), ...(s.edits.length ? {} : { disabled: "" }) }, "Undo last"),
    el("button", { onclick: () => pushEdits(k, []), ...(s.edits.length ? {} : { disabled: "" }) }, "Clear all")));
  return box;
}

async function pushEdits(k, edits) {
  try { await api(`/api/edits/${k}`, { edits }); await poll(true); } catch (e) { status(e.message, true); }
}

// --- block / line box editor
function currentItems(kind) {
  if (!S.layout) return [];
  if (kind === "blocks") return (S.layout.blocks || []).map((b) => ({ box: b.slice() }));
  return (S.layout.lines || []).map((l) => ({ box: l.box.slice(), baseline: l.baseline, block: l.block }));
}

function boxTools(t, s, k, kind) {
  if (!S.layout) { t.append(el("div", { class: "box" }, "The stage has not run yet.")); S.draft = null; return; }
  if (!S.draft || S.draft.kind !== kind || S.draft.k !== k) S.draft = { kind, k, items: currentItems(kind), sel: new Set(), order: [], dirty: false };
  const D = S.draft;
  t.append(el("div", { class: "box" },
    el("div", {}, `${D.items.length} ${kind}${D.dirty ? " — unsaved changes" : ""}. Select to move or resize; the numbers are the reading order.`),
    el("div", { class: "row" },
      toolButton("select", "Select / move"), toolButton("add", "Add"), toolButton("split", "Split"), toolButton("order", "Set order")),
    el("div", { class: "row" },
      el("button", { onclick: () => { D.items = D.items.filter((_, i) => !D.sel.has(i)); D.sel.clear(); D.dirty = true; showStage(k); } }, "Delete"),
      el("button", { onclick: () => mergeSel(D, k) }, "Merge"),
      el("button", { class: "primary", onclick: () => applyDraft(D, s, k), ...(D.dirty ? {} : { disabled: "" }) }, "Apply & re-run"),
      el("button", { onclick: () => { S.draft = null; showStage(k); } }, "Revert")),
    editList(s, k)));
  const hints = { select: "Click to select (shift adds), drag to move, drag a corner to resize.", add: "Drag a new box.",
    split: "Click inside a box to split it at that height (alt-click: at that column).", order: "Click the boxes in reading order; Esc to finish." };
  $("hint").textContent = hints[S.tool] || "";
}

function mergeSel(D, k) {
  if (D.sel.size < 2) return;
  const idx = [...D.sel].sort((a, b) => a - b), bs = idx.map((i) => D.items[i].box);
  const u = [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))];
  const merged = { ...D.items[idx[0]], box: u };
  if ("baseline" in merged) merged.baseline = Math.max(...idx.map((i) => D.items[i].baseline ?? D.items[i].box[3]));
  D.items = D.items.filter((_, i) => !D.sel.has(i)); D.items.splice(idx[0], 0, merged);
  D.sel = new Set([idx[0]]); D.dirty = true; showStage(k);
}

async function applyDraft(D, s, k) {
  let edits;
  if (D.kind === "blocks") edits = [{ op: "set_blocks", blocks: D.items.map((i) => i.box.map(Math.round)) }];
  else {
    const blocks = (S.layout && S.layout.blocks) || [];
    edits = [{ op: "set_lines", lines: D.items.map((i) => {
      const b = i.box.map(Math.round), cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
      const blk = blocks.findIndex((q) => cx >= q[0] && cx <= q[2] && cy >= q[1] && cy <= q[3]);
      return { box: b, baseline: Math.round(i.baseline ?? (b[3] - 0.22 * (b[3] - b[1]))), block: blk >= 0 ? blk : (i.block ?? 0) };
    }) }];
  }
  S.draft = null; await pushEdits(k, edits);
}

// --- the dictionary post-processing pass
function correctTools(t, s, k) {
  const on = (s.params.enabled ?? s.defaults.enabled) !== false;
  const words = S.layout ? S.layout.lines.flatMap((l) => l.words).filter((w) => w.corrected_from) : [];
  const box = el("div", { class: "box" },
    el("div", {}, on ? `The learned-dictionary pass changed ${words.length} word${words.length === 1 ? "" : "s"}: ` +
      `unknown words fixed to dictionary words when the word's image agrees. Click one to find it.`
      : "The learned-dictionary pass is switched off for this profile."),
    el("div", { class: "row" },
      el("button", { class: on ? "" : "primary", onclick: async () => {
        await api(`/api/stage/${k}`, { params: { enabled: !on } }); await poll(true); } }, on ? "Switch off" : "Switch on")));
  for (const w of words)
    box.append(el("div", { class: "corr", onclick: () => { S.selWord = w; centerOn(w.box); draw(); } },
      el("s", {}, w.corrected_from), " → ", el("b", {}, w.text)));
  t.append(box);
}

// --- words
function wordTools(t, s, k) {
  const out = S.st.stages.length - 1;
  const w = S.selWord;
  t.append(el("div", { class: "box" },
    el("div", {}, "Click a word on the page or in the text to correct it. Corrections apply to the final text and hOCR."),
    w ? el("div", { class: "row" },
      el("input", { id: "wordEdit", type: "text", value: w.text, style: "flex:1", onkeydown: (e) => { if (e.key === "Enter") saveWord(); } }),
      el("button", { class: "primary", onclick: saveWord }, "Save")) : null,
    w ? el("div", {}, `confidence ${w.p_correct != null ? w.p_correct : w.confidence ?? "—"}`) : null,
    editList(S.st.stages[out], out)));
  if (w) setTimeout(() => $("wordEdit") && $("wordEdit").focus(), 0);
}

async function saveWord() {
  const out = S.st.stages.length - 1, w = S.selWord, v = $("wordEdit").value;
  if (!w || v === w.text) return;
  const edits = S.st.stages[out].edits.filter((e) => !(e.op === "word" && overlap(e.box, w.box) > 0.5));
  edits.push({ op: "word", box: w.box, text: v });
  S.selWord = null; await pushEdits(out, edits);
}

// ------------------------------------------------------------------ text panel
async function refreshText() {
  const st = S.st.stages, out = st.length - 1;
  if (st[out].status !== "done") { $("text").textContent = "(the page is still being read)"; return; }
  const key = `${out}/${st[out].ms}/${st[out].edits.length}`;
  if (key !== S.finalKey) { S.finalLayout = await api(`/api/layout/${out}`).catch(() => null); S.finalKey = key; }
  const L = S.finalLayout, box = $("text"); box.innerHTML = "";
  if (!L) return;
  let lastBlock = null;
  for (const ln of L.lines) {
    if (!ln.words.length) continue;
    if (lastBlock !== null && ln.block !== lastBlock) box.append("\n");
    lastBlock = ln.block;
    ln.words.forEach((w, i) => {
      const p = w.p_correct ?? w.confidence ?? 1;
      const span = el("span", { class: "w" + (p < 0.5 ? " low" : "") + (w.edited ? " edited" : ""),
        onclick: () => { S.selWord = w; if (!READ.has(st[S.sel].slot)) selectStage(out); else showStage(S.sel); centerOn(w.box); },
        onmouseenter: () => { S.hover = w.box; draw(); }, onmouseleave: () => { S.hover = null; draw(); } }, w.text);
      box.append(span, i < ln.words.length - 1 ? " " : "");
    });
    box.append("\n");
  }
}

// ------------------------------------------------------------------ canvas
function setupCanvas() {
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && S.draft && S.tool === "order") finishOrder();
    if ((e.key === "Delete" || e.key === "Backspace") && S.draft && S.draft.sel.size && document.activeElement === document.body) {
      S.draft.items = S.draft.items.filter((_, i) => !S.draft.sel.has(i)); S.draft.sel.clear(); S.draft.dirty = true; showStage(S.sel);
    }
  });
}

function fit() {
  if (!S.st || !S.st.shape) return;
  const [h, w] = S.st.shape, cw = canvas.clientWidth, ch = canvas.clientHeight;
  const z = Math.min(cw / w, ch / h) * 0.96;
  S.view = { z, x: (cw - w * z) / 2, y: (ch - h * z) / 2 };
}
function zoomAt(sx, sy, f) {
  const v = S.view, px = (sx - v.x) / v.z, py = (sy - v.y) / v.z;
  v.z = Math.max(0.02, Math.min(8, v.z * f)); v.x = sx - px * v.z; v.y = sy - py * v.z; draw();
}
function centerOn(b, zoom = true) {
  const v = S.view;
  if (zoom) {   // bring the word up to a readable size: about a sixth of the view's width
    const want = Math.min(4, canvas.clientWidth / 6 / Math.max(20, b[2] - b[0]));
    if (v.z < want) v.z = want;
  }
  v.x = canvas.clientWidth / 2 - (b[0] + b[2]) / 2 * v.z; v.y = canvas.clientHeight / 2 - (b[1] + b[3]) / 2 * v.z; draw();
}
const toPage = (e) => [(e.offsetX - S.view.x) / S.view.z, (e.offsetY - S.view.y) / S.view.z];
function evPage(e) { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left - S.view.x) / S.view.z, (e.clientY - r.top - S.view.y) / S.view.z]; }
const inBox = (p, b, pad = 0) => p[0] >= b[0] - pad && p[0] <= b[2] + pad && p[1] >= b[1] - pad && p[1] <= b[3] + pad;
function overlap(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const ar = (r) => Math.max(1, (r[2] - r[0]) * (r[3] - r[1]));
  return ix * iy / Math.min(ar(a), ar(b));
}

function onDown(e) {
  const p = toPage(e), s = S.st && S.st.stages[S.sel];
  if (!s) return;
  const D = S.draft;
  if (!S.tool) {
    if (READ.has(s.slot) && S.layout) {
      const w = S.layout.lines.flatMap((l) => l.words).find((w) => inBox(p, w.box));
      if (w) { S.selWord = w; showStage(S.sel); return; }
    }
    S.drag = { mode: "pan", sx: e.clientX, sy: e.clientY, vx: S.view.x, vy: S.view.y }; return;
  }
  if (S.tool === "horizon") {
    S.pts.push(p);
    if (S.pts.length === 2) {
      const [a, b] = S.pts[0][0] <= S.pts[1][0] ? S.pts : [S.pts[1], S.pts[0]];
      const ang = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI;
      S.tool = null; S.pts = []; canvas.classList.remove("tool");
      api(`/api/stage/${S.sel}`, { params: { angle_deg: +ang.toFixed(3) } }).then(() => poll(true));
    }
    draw(); return;
  }
  if (S.tool === "erase_at" || S.tool === "restore_at") {
    pushEdits(S.sel, [...s.edits, { op: S.tool, point: p.map(Math.round) }]); return;
  }
  if (S.tool === "erase" || S.tool === "add") { S.drag = { mode: "rect", a: p, b: p }; return; }
  if (!D) return;
  const hit = D.items.findIndex((it) => inBox(p, it.box));
  if (S.tool === "order") {
    if (hit >= 0 && !D.order.includes(hit)) { D.order.push(hit); if (D.order.length === D.items.length) finishOrder(); else draw(); }
    return;
  }
  if (S.tool === "split" && hit >= 0) {
    const it = D.items[hit], b = it.box;
    const [x, y] = p.map(Math.round);
    const A = { ...it, box: e.altKey ? [b[0], b[1], x, b[3]] : [b[0], b[1], b[2], y] };
    const B = { ...it, box: e.altKey ? [x, b[1], b[2], b[3]] : [b[0], y, b[2], b[3]] };
    if ("baseline" in it) { A.baseline = undefined; B.baseline = undefined; }
    D.items.splice(hit, 1, A, B); D.dirty = true; showStage(S.sel); return;
  }
  if (S.tool === "select") {
    const tol = 8 / S.view.z;
    for (const i of D.sel) {
      const b = D.items[i].box;
      for (const [cx, cy, c] of [[b[0], b[1], "tl"], [b[2], b[1], "tr"], [b[0], b[3], "bl"], [b[2], b[3], "br"]])
        if (Math.abs(p[0] - cx) < tol && Math.abs(p[1] - cy) < tol) { S.drag = { mode: "resize", i, c, orig: b.slice() }; return; }
    }
    if (hit < 0) { D.sel.clear(); draw(); return; }
    if (e.shiftKey) { D.sel.has(hit) ? D.sel.delete(hit) : D.sel.add(hit); }
    else if (!D.sel.has(hit)) D.sel = new Set([hit]);
    S.drag = { mode: "move", a: p, orig: [...D.sel].map((i) => [i, D.items[i].box.slice(), D.items[i].baseline]) };
    showStage(S.sel);
  }
}

function onMove(e) {
  const d = S.drag; if (!d) return;
  if (d.mode === "pan") { S.view.x = d.vx + e.clientX - d.sx; S.view.y = d.vy + e.clientY - d.sy; draw(); return; }
  const p = evPage(e), D = S.draft;
  if (d.mode === "rect") { d.b = p; draw(); return; }
  if (d.mode === "move") {
    const dx = p[0] - d.a[0], dy = p[1] - d.a[1];
    for (const [i, b, bl] of d.orig) { D.items[i].box = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]; if (bl != null) D.items[i].baseline = bl + dy; }
    D.dirty = true; draw(); return;
  }
  if (d.mode === "resize") {
    const b = d.orig.slice();
    if (d.c.includes("l")) b[0] = p[0]; else b[2] = p[0];
    if (d.c.includes("t")) b[1] = p[1]; else b[3] = p[1];
    D.items[d.i].box = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
    D.dirty = true; draw();
  }
}

function onUp() {
  const d = S.drag; S.drag = null; if (!d) return;
  if (d.mode === "rect") {
    const b = [Math.min(d.a[0], d.b[0]), Math.min(d.a[1], d.b[1]), Math.max(d.a[0], d.b[0]), Math.max(d.a[1], d.b[1])].map(Math.round);
    if (b[2] - b[0] < 3 || b[3] - b[1] < 3) { draw(); return; }
    const s = S.st.stages[S.sel];
    if (S.tool === "erase") pushEdits(S.sel, [...s.edits, { op: "erase", box: b }]);
    else if (S.tool === "add" && S.draft) { S.draft.items.push({ box: b }); S.draft.sel = new Set([S.draft.items.length - 1]); S.draft.dirty = true; showStage(S.sel); }
  } else if (d.mode === "move" || d.mode === "resize") showStage(S.sel);
}

function finishOrder() {
  const D = S.draft; if (!D) return;
  const rest = D.items.map((_, i) => i).filter((i) => !D.order.includes(i));
  D.items = [...D.order, ...rest].map((i) => D.items[i]); D.order = []; D.sel.clear(); D.dirty = true;
  S.tool = null; canvas.classList.remove("tool"); showStage(S.sel);
}

// ------------------------------------------------------------------ drawing
function draw() {
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) { canvas.width = cw * dpr; canvas.height = ch * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!S.st || !S.st.open) return;
  const v = S.view, s = S.st.stages[S.sel];
  ctx.save(); ctx.translate(v.x, v.y); ctx.scale(v.z, v.z);
  if (S.img && S.img.complete && S.img.naturalWidth) {
    ctx.imageSmoothingEnabled = v.z * S.imgScale < 2;
    ctx.drawImage(S.img, 0, 0, S.img.naturalWidth / S.imgScale, S.img.naturalHeight / S.imgScale);
  }
  const lw = 1.5 / v.z;
  const L = S.layout;
  if (L && s && s.status === "done" && S.tool !== "horizon") {
    if (S.layers.zones) for (const b of L.image_zones || []) rect(b, "#8e44ad", lw, [6 / v.z, 4 / v.z]);
    if (S.layers.rules) for (const r of [...(L.rules_h || []), ...(L.rules_v || [])]) if (Array.isArray(r) && r.length === 4) rect(r, "#c53030", lw);
    if (S.layers.tables) for (const t of L.tables || []) for (const c of t.cells || []) rect(c.box, "#0f9fb0", lw * 0.8);
    const D = S.draft;
    if (D && D.kind === "blocks") drawItems(D, "#2463eb", lw, true);
    else if (S.layers.blocks) (L.blocks || []).forEach((b, i) => { rect(b, "#2463eb", lw); label(b, i + 1, "#2463eb"); });
    if (D && D.kind === "lines") drawItems(D, "#1b8a4b", lw, true);
    else if (S.layers.lines && !READ.has(s.slot)) for (const l of L.lines || []) rect(l.box, "#1b8a4b", lw * 0.8);
    if (S.layers.words && READ.has(s.slot)) for (const l of L.lines || []) for (const w of l.words) {
      const p = w.p_correct ?? w.confidence ?? 1;
      rect(w.box, w.edited ? "#2463eb" : w.corrected_from ? "#1b8a4b" : p < 0.5 ? "#c53030" : "#b7791f",
           w.corrected_from ? lw * 2 : lw * 0.8);
    }
  }
  if (S.selWord) rect(S.selWord.box, "#c53030", lw * 2);
  if (S.hover) rect(S.hover, "#2463eb", lw * 2.5);
  if (S.drag && S.drag.mode === "rect") rect([S.drag.a[0], S.drag.a[1], S.drag.b[0], S.drag.b[1]], "#c53030", lw, [5 / v.z, 3 / v.z]);
  for (const p of S.pts) { ctx.fillStyle = "#c53030"; ctx.beginPath(); ctx.arc(p[0], p[1], 5 / v.z, 0, 7); ctx.fill(); }
  ctx.restore();
  if (s && s.status !== "done") {
    ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(0, 0, cw, 28); ctx.fillStyle = "#fff";
    ctx.fillText(s.status === "error" ? "this stage failed — see the error on the right" : "this stage has not run yet…", 10, 18);
  }
}

function rect(b, color, lw, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash || []);
  ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); ctx.setLineDash([]);
}
function label(b, text, color) {
  const v = S.view, fs = 14 / v.z;
  ctx.font = `bold ${fs}px system-ui`; ctx.fillStyle = color;
  const w = ctx.measureText(String(text)).width + 6 / v.z;
  ctx.fillRect(b[0], b[1] - fs - 2 / v.z, w, fs + 2 / v.z);
  ctx.fillStyle = "#fff"; ctx.fillText(String(text), b[0] + 3 / v.z, b[1] - 4 / v.z);
}
function drawItems(D, color, lw, numbers) {
  D.items.forEach((it, i) => {
    const sel = D.sel.has(i);
    rect(it.box, sel ? "#c53030" : color, sel ? lw * 2 : lw);
    if (numbers) {
      const ord = D.order.indexOf(i);
      label(it.box, S.tool === "order" ? (ord >= 0 ? ord + 1 : "·") : i + 1, sel ? "#c53030" : color);
    }
    if (sel) for (const [x, y] of [[it.box[0], it.box[1]], [it.box[2], it.box[1]], [it.box[0], it.box[3]], [it.box[2], it.box[3]]]) {
      ctx.fillStyle = "#c53030"; ctx.fillRect(x - 4 / S.view.z, y - 4 / S.view.z, 8 / S.view.z, 8 / S.view.z);
    }
  });
}

init();

// ------------------------------------------------------------------ hOCR, rendered from the file itself
function hocrBox(title) {
  const m = /bbox (-?\d+) (-?\d+) (-?\d+) (-?\d+)/.exec(title || "");
  return m ? m.slice(1, 5).map(Number) : null;
}
function hocrProp(title, key) {
  const m = new RegExp(key + " ([-\\d.]+)").exec(title || "");
  return m ? Number(m[1]) : null;
}
function renderHocr(hocr) {
  const host = $("resultRender"); host.innerHTML = "";
  const doc = new DOMParser().parseFromString(hocr, "application/xhtml+xml");
  if (doc.getElementsByTagName("parsererror").length) { host.textContent = "this hOCR does not parse as XHTML"; return; }
  const pageEl = [...doc.getElementsByTagName("*")].find((e) => e.getAttribute("class") === "ocr_page");
  const pb = hocrBox(pageEl && pageEl.getAttribute("title")) || [0, 0, 2550, 3300];
  const W = pb[2] - pb[0], H = pb[3] - pb[1];
  const avail = Math.max(300, host.clientWidth - 32);
  const sc = Math.min(1, avail / W);
  const page = el("div", { class: "hpage", style: `width:${W * sc}px;height:${H * sc}px` });
  if (S.renderScan) {
    const last = S.st.stages.length - 1;
    page.append(el("img", { src: `/api/image/${last}/gray.png?scale=${Math.min(1, 1600 / W)}`, class: "hscan" }));
  }
  const box = (b, cls, label, title) => {
    const d = el("div", { class: cls, title: title || "", style:
      `left:${(b[0] - pb[0]) * sc}px;top:${(b[1] - pb[1]) * sc}px;width:${(b[2] - b[0]) * sc}px;height:${(b[3] - b[1]) * sc}px` });
    if (label != null) d.append(el("span", { class: "hlabel" }, label));
    page.append(d); return d;
  };
  const all = [...doc.getElementsByTagName("*")];
  const words = new Map();   // hOCR word -> whether the layout says it was corrected
  if (S.finalLayout) for (const l of S.finalLayout.lines) for (const w of l.words) if (w.corrected_from) words.set(w.box.join(","), true);
  let area = 0;
  for (const e of all) {
    const cls = e.getAttribute("class"), b = hocrBox(e.getAttribute("title"));
    if (!b) continue;
    if (S.renderBoxes !== false) {
      if (cls === "ocr_carea") box(b, "hcarea", ++area);
      else if (cls === "ocr_table") box(b, "htable", "table");
      else if (e.tagName.toLowerCase() === "td") box(b, "hcell");
      else if (cls === "ocr_photo") box(b, "hphoto", "image");
      else if (cls === "ocr_separator") box(b, "hsep");
    }
    if (cls === "ocrx_word") {
      const conf = hocrProp(e.getAttribute("title"), "x_wconf");
      const h = (b[3] - b[1]) * sc, w = (b[2] - b[0]) * sc;
      const corrected = words.has(b.join(","));
      const span = el("span", { class: "hword" + (conf != null && conf < 50 ? " low" : "") + (corrected ? " fixed" : ""),
        title: `${e.textContent}  —  confidence ${conf ?? "?"}%${corrected ? " (dictionary-corrected)" : ""}`,
        style: `left:${(b[0] - pb[0]) * sc}px;top:${(b[1] - pb[1]) * sc}px;height:${h}px;font-size:${Math.max(4, h * 0.82)}px;line-height:${h}px` },
        e.textContent);
      page.append(span);
      // stretch or squeeze each word to its box width, as the scan printed it
      requestAnimationFrame(() => { const nat = span.scrollWidth || 1; span.style.transform = `scaleX(${Math.min(3, w / nat)})`; });
    }
  }
  host.append(page);
}
