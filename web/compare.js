// Base vs fine-tuned local Gemma repair panel. Owner: Claude (Gemma comparison).
// Loaded by app.js only if this file exists. Talks to /api/compare (compare_api.py) and
// replays attempts on the main stage through window.argonTwin.playAttempt.

const API = "/api/compare";
const DEFAULT_CASE = "scene-13-variant-2";
const VARIANT_LABEL = { base: "Base Gemma", tuned: "Fine-tuned Gemma" };
const STATUS_TEXT = {
  invalid_output: "Invalid output",
  timeout: "Timed out",
  service_unavailable: "Model server unavailable",
  http_error: "Model server error",
};
const FIELD_LABEL = {
  grasp_height_offset_m: "grasp height offset (m)",
  approach_yaw_deg: "approach yaw (deg)",
  pregrasp_clearance_m: "pre-grasp clearance (m)",
  lift_height_m: "lift height (m)",
  handover_point_m: "hand-over point (m)",
  handover_yaw_offset_deg: "hand-over yaw offset (deg)",
  gripper_close_delay_s: "gripper close delay (s)",
  release_delay_s: "release delay (s)",
  release_height_m: "release height (m)",
  max_speed_m_s: "max speed (m/s)",
};

const css = document.createElement("link");
css.rel = "stylesheet";
css.href = "compare.css?v=1";
document.head.append(css);

const ui = { record: null, polling: null };

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function getJSON(url, options = {}) {
  const res = await fetch(url, { cache: "no-store", ...options });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    throw new Error(detail ? (typeof detail === "string" ? detail : JSON.stringify(detail)) : `${res.status} ${url}`);
  }
  return body;
}

const fmtS = (s) => (s == null ? "n/a" : `${s.toFixed(2)} s`);
const fmtVal = (v) => (Array.isArray(v) ? `[${v.map((x) => (+x).toFixed(2)).join(", ")}]` : typeof v === "number" ? String(+v.toFixed(3)) : String(v));

function decisive(attempt) {
  const v = attempt.verdict;
  const m = attempt.sim.metrics;
  if (!v.passed) return v.failed_checks.join("; ") || "failed";
  return `can ${(m.can_final_xy_error_m * 1000).toFixed(1)} mm from target, at rest`;
}

function patchRows(patch, original) {
  return Object.entries(patch ?? {}).map(([k, v]) =>
    `<li><span class="field">${esc(FIELD_LABEL[k] ?? k)}</span> <s>${esc(fmtVal(original[k]))}</s> &rarr; <b>${esc(fmtVal(v))}</b></li>`).join("");
}

function chip(attempt) {
  return `<span class="cmp-chip ${attempt.verdict.passed ? "pass" : "fail"}">${attempt.verdict.passed ? "PASS" : "FAIL"}</span>`;
}

function step(n, title, body, replay) {
  return `<li class="cmp-step"><span class="cmp-n">${n}</span><div><h4>${title}</h4>${body}
    ${replay ? `<button type="button" class="cmp-replay" data-replay="${replay}">Replay on stage</button>` : ""}</div></li>`;
}

function variantStep(n, key, entry, original) {
  const scale = entry.adapter_scale ?? (key === "base" ? 0 : 1);
  const head = `${VARIANT_LABEL[key]} <span class="cmp-sub">adapter scale ${scale}</span>`;
  if (!entry.status || entry.status === "pending") return step(n, head, `<p class="cmp-muted">Waiting</p>`);
  if (entry.status === "requesting") return step(n, head, `<p class="cmp-muted">Asking the local model&hellip;</p>`);
  const timing = `<p class="cmp-time">Model response ${fmtS(entry.http_s)}${entry.attempt ? ` &middot; simulation ${fmtS(entry.attempt.sim_s)}` : ""}</p>`;
  if (STATUS_TEXT[entry.status]) {
    const failedAfter = `<p class="cmp-time">Request failed after ${fmtS(entry.http_s)}</p>`;
    return step(n, head, `<p><span class="cmp-chip err">${esc(STATUS_TEXT[entry.status])}</span></p>
      <p class="cmp-error">${esc(entry.error)}</p>
      ${entry.raw_content ? `<pre class="cmp-raw">${esc(entry.raw_content)}</pre>` : ""}
      <p class="cmp-muted">No patch applied, nothing simulated. No other model was used in its place.</p>${failedAfter}`);
  }
  const proposed = `<p class="cmp-label">Proposed patch</p><ul class="cmp-patch">${patchRows(entry.patch, original)}</ul>`;
  if (entry.status === "simulating" || !entry.attempt) return step(n, head, `${proposed}<p class="cmp-muted">MuJoCo is running it&hellip;</p>${timing}`);
  return step(n, head, `${proposed}<p class="cmp-verdict">${chip(entry.attempt)} <span>${esc(decisive(entry.attempt))}</span></p>${timing}`, key);
}

function renderRecord() {
  const r = ui.record;
  const out = document.getElementById("cmp-result");
  if (!r) { out.innerHTML = `<p class="cmp-muted">No live comparison yet.</p>`; return; }
  if (!r.original) {
    out.innerHTML = r.state === "error"
      ? `<p class="cmp-error" role="alert">Comparison stopped: ${esc(r.error)}</p>`
      : `<p class="cmp-muted">Re-running the failed attempt in MuJoCo&hellip;</p>`;
    return;
  }
  const o = r.original;
  const steps = [
    step(1, "Same failed attempt", `<p class="cmp-verdict">${chip(o)} <span>${esc(decisive(o))}</span></p>
      <p class="cmp-time">Simulation ${fmtS(o.sim_s)}</p>`, "original"),
    variantStep(2, "base", r.variants.base, r.original_plan),
    variantStep(3, "tuned", r.variants.tuned, r.original_plan),
  ];
  const diffs = r.payload_differences ?? [];
  const prov = [
    `Live run <code>${esc(r.compare_id)}</code>, ${esc(new Date(r.created_at).toLocaleString())}`,
    `${esc(r.case.label)}: <code>${esc(r.case.case_id)}</code>`,
    r.same_request_except_adapter === true ? "Both models got an identical request; only the adapter scale differed"
      : r.same_request_except_adapter === false ? "<b>Requests differed beyond the adapter scale</b>" : null,
    r.payload_matches_recorded_case === false
      ? `Failure re-measured by today's simulator: ${diffs.length} field(s) differ from the recorded case's input`
      : r.payload_matches_recorded_case ? "Input identical to the recorded case" : null,
    `Local server ${esc(r.provenance?.server?.base_url ?? "")}, model <code>argon-twin-repair</code>, 0 cloud calls`,
    "Same scene, same seed, same MuJoCo judge for both. Pass or fail comes only from measurements against fixed limits",
  ].filter(Boolean);
  out.innerHTML = `
    ${r.state === "error" ? `<p class="cmp-error" role="alert">Comparison stopped: ${esc(r.error)}</p>` : ""}
    <p class="cmp-label">Live run</p>
    <ol class="cmp-steps">${steps.join("")}</ol>
    <details class="cmp-prov"><summary>Provenance</summary><ul>${prov.map((p) => `<li>${p}</li>`).join("")}</ul></details>`;
}

async function replay(which) {
  const r = ui.record;
  if (!r || !window.argonTwin?.playAttempt) return;
  const attempt = which === "original" ? r.original : r.variants[which]?.attempt;
  if (attempt) await window.argonTwin.playAttempt(attempt, `/runs/${r.compare_id}/${which}/`, r.scene);
}

function setRunning(running) {
  const btn = document.getElementById("cmp-run");
  btn.disabled = running;
  btn.textContent = running ? "Running…" : "Run both models";
}

function showError(err) {
  const box = document.getElementById("cmp-msg");
  box.hidden = false;
  box.textContent = err?.message ?? String(err);
}

async function load(compareId) {
  ui.record = await getJSON(`${API}/${encodeURIComponent(compareId)}`);
  renderRecord();
  const running = ui.record.state === "running";
  setRunning(running);
  clearTimeout(ui.polling);
  if (running) ui.polling = setTimeout(() => load(compareId).catch(showError), 700);
}

function renderServer(status) {
  const el = document.getElementById("cmp-server");
  el.className = `cmp-server ${status.available ? "up" : "down"}`;
  el.textContent = status.available ? "Local model server running on this Mac" : "Local model server not answering";
}

async function start() {
  document.getElementById("cmp-msg").hidden = true;
  try {
    const status = await getJSON(`${API}/status`);
    renderServer(status);
    if (!status.available) {
      throw new Error(`The local Gemma server isn't answering at ${status.base_url}. Start it with service.py start. Nothing else is used in its place.`);
    }
    setRunning(true);
    const { compare_id } = await getJSON(API, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ case_id: document.getElementById("cmp-case").value, seed: 0 }),
    });
    await load(compare_id);
  } catch (err) {
    setRunning(false);
    showError(err);
  }
}

function renderRecorded(rec) {
  const row = (label, s) => (s ? `<tr><th>${label}</th><td>${s.physics_passed}/${s.cases}</td><td>${s.valid_patches}/${s.cases}</td><td>${s.median_latency_s.toFixed(2)} s</td></tr>` : "");
  document.getElementById("cmp-recorded").innerHTML = `
    <p class="cmp-label">${esc(rec.label)} &middot; recorded ${esc(new Date(rec.observed_at_utc).toLocaleString())}</p>
    <table><thead><tr><th></th><th>MuJoCo passes</th><th>Valid patches</th><th>Median response</th></tr></thead>
      <tbody>${row("Base", rec.base)}${row("Fine-tuned", rec.tuned)}</tbody></table>
    <p class="cmp-muted">Local ${esc(rec.base_quantization ?? "")} base, JSON schema required. Response times exclude simulation. Recorded earlier and separate from the live run above.</p>`;
}

function buildPanel() {
  const panel = document.createElement("aside");
  panel.id = "cmp-panel";
  panel.className = "cmp-panel";
  panel.hidden = true;
  panel.setAttribute("aria-label", "Local Gemma repair comparison");
  panel.innerHTML = `
    <header><h3>Local Gemma repair: base vs fine-tuned</h3>
      <button type="button" id="cmp-close" aria-label="Close">&times;</button></header>
    <p class="cmp-intro">Same failed attempt &rarr; each model proposes a fix &rarr; MuJoCo judges each fix.</p>
    <p id="cmp-server" class="cmp-server">Checking the local model server&hellip;</p>
    <div class="cmp-controls">
      <label>Validation case <select id="cmp-case"></select></label>
      <button type="button" id="cmp-run" class="primary">Run both models</button>
    </div>
    <p id="cmp-msg" class="cmp-error" role="alert" hidden></p>
    <div id="cmp-result"></div>
    <div id="cmp-recorded" class="cmp-recorded"></div>
    <details class="cmp-about"><summary>What this model is</summary><ul>
      <li>A small controller-repair pilot: Gemma 4 E4B with a LoRA adapter, served on this Mac.</li>
      <li>Trained on 36 synthetic, MuJoCo-verified repair pairs; tested on 12 held-out cases from four other scene variants.</li>
      <li>Examples cover hand-over-point and grasp-height corrections only. Labels are scripted, simulator-verified repairs, not written by Argon.</li>
      <li>It reads structured failure data. It doesn't see the desk video, control motors or decide success.</li>
      <li>Grasps use the disclosed weld assist.</li>
    </ul></details>`;
  document.querySelector("main.stage").append(panel);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.id = "cmp-toggle";
  toggle.textContent = "Gemma repair test";
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-controls", "cmp-panel");
  document.querySelector(".status")?.prepend(toggle);

  const setOpen = (open) => {
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
  };
  toggle.addEventListener("click", () => setOpen(panel.hidden));
  panel.querySelector("#cmp-close").addEventListener("click", () => setOpen(false));
  panel.querySelector("#cmp-run").addEventListener("click", start);
  panel.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-replay]");
    if (btn) replay(btn.dataset.replay);
  });
  if (new URLSearchParams(location.search).has("compare")) setOpen(true);
}

async function init() {
  if (!document.querySelector("main.stage")) return;
  buildPanel();
  try {
    const [status, recorded, cases, runs] = await Promise.all([
      getJSON(`${API}/status`), getJSON(`${API}/recorded`), getJSON(`${API}/cases`), getJSON(API),
    ]);
    renderServer(status);
    renderRecorded(recorded);
    const select = document.getElementById("cmp-case");
    select.innerHTML = cases.map((c) => {
      const rec = c.recorded_base_passed == null ? "" : ` (recorded: base ${c.recorded_base_passed ? "pass" : "fail"}, tuned ${c.recorded_tuned_passed ? "pass" : "fail"})`;
      return `<option value="${esc(c.case_id)}">${esc(c.case_id)}${esc(rec)}</option>`;
    }).join("");
    select.value = cases.some((c) => c.case_id === DEFAULT_CASE) ? DEFAULT_CASE : cases[0]?.case_id;
    const requested = new URLSearchParams(location.search).get("compare");
    const pick = runs.find((r) => r.compare_id === requested) ?? runs.find((r) => r.state === "done");
    if (pick) await load(pick.compare_id);
    else renderRecord();
  } catch (err) {
    // No comparison API here (static replay hosting, or a server started before compare_api.py).
    document.getElementById("cmp-server").textContent = "Comparison API not available on this server";
    document.getElementById("cmp-run").disabled = true;
    showError(err);
  }
}

init();
