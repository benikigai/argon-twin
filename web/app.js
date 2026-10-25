// ArgonTwin viewer. Owner: integrator (Claude).
// Live data comes from the local server (/api/runs). When that isn't reachable
// (e.g. on Vercel) the viewer plays recorded runs from replay/ and says so.
//
// Keys: ← → step through the loop, space replays, T cycles the view (gold, solid, photoreal when ready),
// H returns to the front hero camera, C toggles the gripper close-up, O shows collision surfaces.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const $ = (id) => document.getElementById(id);
const GOLD = new THREE.Color("#e3b341");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
// Live runs start from the profile's example scene and plan, plus whatever the plan editor
// changed (sent as initial_patch). Example files are never written.
//
// The default is the verified Celsius can collision repair (run 20261025-134505): the real can,
// segmented from the scan, rides on the simulated can; the right arm picks it up and the left
// places it. The demo example's plan puts the handover point inside the screens, so one click
// on Run the loop (or Reset, then Apply and run) reproduces failure -> Argon repair -> pass.
// Its own scene.handover_point_m is out of reach, so no passing point is in Argon's input.
// The safe fixture examples/desk_celsius.json is left untouched.
// ?variant=celsius is kept as an alias. Older runs, including the earlier collision proof
// 20261025-133036, stay reachable with ?run=<id>.
const CELSIUS = {
  label: null,
  base: "examples/desk_celsius_demo.json",
  breakHandover: [0.0, 0.21, 0.3],      // inside the screens
  pinnedRun: "20261025-134505",         // the recorded proof the viewer opens on
  background: "desk-without-celsius.splat",
};
const PROFILES = { default: CELSIUS, celsius: CELSIUS };
const VARIANT = new URLSearchParams(location.search).get("variant") === "celsius" ? "celsius" : "default";
const PROFILE = PROFILES[VARIANT];
const BASE_SCENE = PROFILE.base;
const BREAK_HANDOVER = PROFILE.breakHandover;
const PINNED_RUN = PROFILE.pinnedRun;
const ARM_REACH_M = 0.75;  // ViperX 300 nominal reach (about 750 mm); used only for on-screen notes

const state = {
  source: null,      // "api" | "replay"
  runId: null,
  run: null,         // { summary, manifest, log }
  base: "",
  steps: [],         // [{ kind: "attempt" | "critique", attempt }]
  current: 0,
  speed: 1,             // rollout playback speed (the guided tour plays attempts faster)
  splatStyle: "gold",  // view mode: "gold" | "solid" | "photo"
  rollout: null,
  rolloutKey: null,
  bodies: [],
  frame: 0,
  playing: false,
  lastTick: 0,
  pendingVerdict: null, // attempt whose verdict shows once its replay finishes
  baseScene: null,      // { scene, plan } from BASE_SCENE
  draft: null,          // { handover_point_m, lift_height_m } being edited
  closeup: false,
  showColliders: false,
  renderStyle: "scan",  // "scan" (gold dots) | "solid" | "photo" (trained splat, when present)
};

const scanMaterials = new Map();  // gold-scan point materials by role

// --- Three.js stage -------------------------------------------------------------

const canvas = $("stage");
// ?capture=1 keeps the last WebGL frame readable so a screenshot of the page includes the scene.
const CAPTURE = new URLSearchParams(location.search).get("capture") === "1";
// Opening view once the desk splat loads: Photoreal unless ?view=gold or ?view=solid.
const START_VIEW = new URLSearchParams(location.search).get("view") ?? "photo";
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: CAPTURE });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor("#000000");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.position.set(0.0, 1.05, 1.6);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.12, 0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight("#fff5dc", "#101014", 1.1));
const key = new THREE.DirectionalLight("#ffffff", 1.6);
key.position.set(0.8, 2, 1.2);
scene.add(key);

// MuJoCo is z-up; Three.js is y-up. Everything from the sim lives under `world`.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
const splatLayer = new THREE.Group();
const robotLayer = new THREE.Group();
const markerLayer = new THREE.Group();
world.add(splatLayer, robotLayer, markerLayer);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const [role, m] of scanMaterials) m.uniforms.uMinPx.value = minDotPx(role);
  if (typeof positionPhases === "function") positionPhases();
  heroCamera();
}

// Fixed front shot: both arm bases framed symmetrically, table centered, looking
// along +y in MuJoCo coordinates (from the camera's starting side of the table).
// Drag to look around; H or double-click returns here.
function heroCamera() {
  const sc = state.run?.manifest?.scene;
  const left = sc?.left_arm_base_m ?? [-0.45, 0, 0];
  const right = sc?.right_arm_base_m ?? [0.45, 0, 0];
  const cx = (left[0] + right[0]) / 2;
  const cy = (left[1] + right[1]) / 2;
  const halfSpan = Math.abs(right[0] - left[0]) / 2 + 0.22;        // arm bases plus their reach outward
  const hfov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect);
  const halfHeight = 0.38;                                            // arms reach about 0.6 m above the table
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const fitWidth = halfSpan / Math.tan(hfov / 2);
  const fitHeight = halfHeight / Math.tan(vfov / 2);
  const distance = Math.max(fitWidth, fitHeight) * 1.5;               // margin for the verdict and frames overlays
  const elevation = THREE.MathUtils.degToRad(24);
  const target = new THREE.Vector3(cx, 0.08, -cy);                    // MuJoCo (x, y, z) -> Three (x, z, -y)
  camera.position.set(cx, target.y + distance * Math.sin(elevation), -cy + distance * Math.cos(elevation));
  controls.target.copy(target);
  controls.update();
}
window.addEventListener("resize", resize);
resize();

// --- Data ------------------------------------------------------------------------

const rolloutCache = new Map();

async function getJSON(url, timeoutMs = 4000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctl.signal, cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function detectSource() {
  try { await getJSON("/api/health", 1500); return "api"; } catch { return "replay"; }
}

async function listRuns() {
  if (state.source === "api") return getJSON("/api/runs");
  const index = await getJSON("replay/index.json");
  return index.runs.map((r) => ({ ...r, source: "REPLAY", state: "done" }));
}

async function loadRun(runId) {
  state.runId = runId;
  if (state.source === "api") {
    state.run = await getJSON(`/api/runs/${runId}`);
    state.base = `/runs/${runId}/`;
  } else {
    const manifest = await getJSON(`replay/${runId}/manifest.json`);
    state.run = { summary: { run_id: runId, source: "REPLAY", state: "done", created_at: manifest.created_at }, manifest, log: [] };
    state.base = `replay/${runId}/`;
  }
  if (loadedScanUrl !== null) await useBackground(backgroundUrlFor(state.run));
}

function buildSteps() {
  const steps = [];
  for (const a of state.run?.manifest?.attempts ?? []) {
    steps.push({ kind: "attempt", attempt: a });
    if (a.critique) steps.push({ kind: "critique", attempt: a });
  }
  if (state.run?.summary?.state === "running") steps.push({ kind: "pending" });
  state.steps = steps;
}

// --- Copy ------------------------------------------------------------------------

const PLAIN = {
  "handover_completed is false": "The can never reached the second arm",
  "dropped": "The can fell",
  "off target": "The can missed the target",
  "not at rest": "The can was still moving at the end",
  "arm hit table": "An arm pressed into the table",
  "arm hit obstacle": "An arm hit the laptop or a screen",
  "arm inside obstacle": "An arm went inside the laptop or a screen",
  "object inside obstacle": "The can went through the laptop or a screen",
};

function plainReason(check) {
  for (const [prefix, text] of Object.entries(PLAIN)) if (check.startsWith(prefix)) return text;
  return check;
}

function attemptHeadline(a) {
  const m = a.sim.metrics;
  if (a.verdict.passed) return `Can placed ${(m.can_final_xy_error_m * 100).toFixed(1)} cm from target`;
  return plainReason(a.verdict.failed_checks[0] ?? "Failed");
}

const PATCH_NAMES = {
  grasp_height_offset_m: ["Grip height", (v) => `${(v * 100).toFixed(1)} cm`],
  approach_yaw_deg: ["Approach angle", (v) => `${v.toFixed(0)}°`],
  pregrasp_clearance_m: ["Approach clearance", (v) => `${(v * 100).toFixed(1)} cm`],
  lift_height_m: ["Lift height", (v) => `${(v * 100).toFixed(1)} cm`],
  handover_point_m: ["Hand-over point", (v) => v.map((x) => (x * 100).toFixed(0)).join(", ") + " cm"],
  handover_yaw_offset_deg: ["Hand-over twist", (v) => `${v.toFixed(0)}°`],
  gripper_close_delay_s: ["Grip delay", (v) => `${v.toFixed(2)} s`],
  release_delay_s: ["Release delay", (v) => `${v.toFixed(2)} s`],
  release_height_m: ["Release height", (v) => `${(v * 100).toFixed(1)} cm`],
  max_speed_m_s: ["Arm speed", (v) => `${v.toFixed(2)} m/s`],
};

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fmtClock(iso) {
  return iso ? new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
}

// --- Rendering ---------------------------------------------------------------------

function renderSource() {
  const el = $("source");
  const s = state.run?.summary;
  if (state.source === "api" && s?.source === "LIVE") {
    el.textContent = s.state === "running" ? "Live, running now" : "Live run";
    el.className = "source live";
  } else if (s) {
    el.textContent = `Replay of a run recorded at ${fmtClock(s.created_at)}`;
    el.className = "source replay";
  } else {
    el.textContent = state.source === "api" ? "Live, no runs yet" : "No recorded runs yet";
    el.className = "source";
  }
  $("run").hidden = state.source !== "api";
}

function renderRail() {
  const list = $("steps");
  list.replaceChildren();
  if (!state.steps.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = state.source === "api" ? "Run the loop to see the first attempt." : "No recorded run to replay yet.";
    list.append(li);
    return;
  }
  state.steps.forEach((step, i) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `step ${step.kind}`;
    if (i === state.current) { btn.classList.add("current"); btn.setAttribute("aria-current", "step"); }
    if (step.kind === "attempt") {
      const a = step.attempt;
      btn.classList.add(a.verdict.passed ? "pass" : "fail");
      btn.innerHTML = `<span class="title">Attempt ${a.attempt} ${a.verdict.passed ? "passed" : "failed"}</span>
        <span class="detail">${escapeHtml(attemptHeadline(a))}</span>`;
    } else if (step.kind === "critique") {
      const changes = Object.entries(step.attempt.critique.body.patch).filter(([, v]) => v !== null).length;
      btn.innerHTML = `<span class="title">Argon explains and fixes</span>
        <span class="detail">${changes} plan change${changes === 1 ? "" : "s"} for attempt ${step.attempt.attempt + 1}</span>`;
    } else {
      btn.classList.add("pending");
      btn.innerHTML = `<span class="title">Working</span><span class="detail">${escapeHtml((state.run.log ?? []).at(-1) ?? "")}</span>`;
    }
    btn.onclick = () => goTo(i);
    li.append(btn);
    list.append(li);
  });
}

function renderVerdict(a) {
  const box = $("verdict");
  if (!a) { box.hidden = true; return; }
  box.hidden = false;
  box.className = `verdict ${a.verdict.passed ? "pass" : "fail"}`;
  // Provenance: one short line, with the full sentences behind "Details".
  const flags = [];  // [short, long]
  if (a.mode === "MANUAL") flags.push(["hand-written start", "Scene and starting plan written by hand"]);
  if (a.mode === "LIVE_ARGON") flags.push(["plan from Argon", "Plan patched by Gemini 4 Argon in a live call"]);
  if (a.mode === "ARGON_INVALID") flags.push(["Argon output invalid", "Argon's fix failed validation, so the plan was kept"]);
  if (a.mode === "LOCAL_GEMMA_BASE") flags.push(["plan from local Gemma (base)", "Plan from the base Gemma 4 model running locally, for the comparison"]);
  if (a.mode === "LOCAL_GEMMA_TUNED") flags.push(["plan from local Gemma (fine-tuned)", "Plan from the fine-tuned Gemma 4 adapter running locally, for the comparison"]);
  const edited = state.run?.input?.initial_patch;
  if (a.attempt === 1 && edited) flags.push(["edited before run", `Starting plan edited before the run: ${describePatch(edited)}`]);
  const sc = state.run?.manifest?.scene;
  if (sc) {
    const dist = (b) => Math.hypot(...a.plan.handover_point_m.map((v, i) => v - b[i]));
    const nearest = Math.min(dist(sc.left_arm_base_m), dist(sc.right_arm_base_m));
    if (nearest > ARM_REACH_M) flags.push(["handover past reach", `Handover point ${nearest.toFixed(2)} m from the nearest arm base; the arm reaches about ${ARM_REACH_M} m`]);
  }
  if (a.sim.metrics.grasp_assist_used) flags.push(["weld grasp", "Grasp assist: the can is welded to the gripper once it closes"]);

  // Obstacle contact stays visible: it's a measurement, not provenance.
  const m = a.sim.metrics;
  const obstacles = a.verdict.obstacle_contact_measured
    ? `Obstacles: ${m.max_arm_obstacle_contact_force_n.toFixed(1)} N contact, arm ${(m.max_arm_obstacle_penetration_m * 1000).toFixed(1)} mm, can ${(m.max_object_obstacle_penetration_m * 1000).toFixed(1)} mm deep`
    : "Obstacle collisions not measured (earlier layout)";
  if (a.verdict.obstacle_contact_measured) flags.push(["new layout", "New workspace layout with screens and laptop fitted from the video"]);

  box.innerHTML = `<div class="word">${a.verdict.passed ? "Pass" : "Fail"}</div>
    <div class="reason">${escapeHtml(attemptHeadline(a))}</div>
    <div class="by">Measured by MuJoCo, attempt ${a.attempt}</div>
    <div class="obstacles ${a.verdict.obstacle_contact_measured ? "" : "unmeasured"}">${escapeHtml(obstacles)}</div>
    ${flags.length ? `<details class="provenance"><summary>${escapeHtml(flags.map((f) => f[0]).join(", "))}</summary>
      <ul>${flags.map((f) => `<li>${escapeHtml(f[1])}</li>`).join("")}</ul></details>` : ""}`;
}

function frameName(k) {
  return k.label.replace(new RegExp(`[_-]?${k.camera}$`), "").replace(/_/g, " ");
}

function keyframeURL(a, k) {
  return `${state.base}attempt_${a.attempt}/${k.path}`;
}

function richText(text) {
  // Escape, then show `code` spans from Argon's text as code.
  return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}

function readableDiagnosis(text) {
  const names = {
    ...Object.fromEntries(Object.entries(PATCH_NAMES).map(([key, [label]]) => [key, label.toLowerCase()])),
    upper_screen: "upper screen", lower_screen: "lower screen",
    max_arm_obstacle_contact_force_n: "peak arm contact force",
    max_object_obstacle_penetration_m: "can penetration",
    can_min_z_m: "lowest point of the can", left_grasped: "left arm grasped the can",
    right_grasped: "right arm grasped the can",
  };
  return richText(text.replace(/\s*\([^)]*\)/g, "").replace(/\b[a-z]+(?:_[a-z]+)+\b/g, (word) => names[word] ?? word));
}

function renderCritique(a) {
  const subs = $("subtitles");
  const ev = $("evidence");
  if (!a?.critique) { subs.hidden = true; ev.hidden = true; return; }
  const sameReview = subs.dataset.attempt === `${state.runId}/${a.attempt}`;
  const scrollTop = sameReview ? subs.scrollTop : 0;
  const moreOpen = sameReview && subs.querySelector("details.more")?.open;
  subs.dataset.attempt = `${state.runId}/${a.attempt}`;
  const c = a.critique;
  const changes = Object.entries(c.body.patch)
    .filter(([, v]) => v != null)
    .map(([k, v]) => {
      const [name, fmt] = PATCH_NAMES[k] ?? [k, String];
      const format = (value) => k === "handover_point_m"
        ? value.map((n, i) => `${["x", "y", "z"][i]} ${(n * 100).toFixed(0)}`).join(", ") + " cm"
        : fmt(value);
      return `<tr><th scope="row">${escapeHtml(name)}</th><td>${escapeHtml(format(a.plan[k]))}</td><td>${escapeHtml(format(v))}</td></tr>`;
    });
  const u = c.usage;
  const receipt = [
    u.input_tokens != null ? `${u.input_tokens.toLocaleString()} in` : null,
    u.output_tokens != null ? `${u.output_tokens.toLocaleString()} out` : null,
    u.thought_tokens != null ? `${u.thought_tokens.toLocaleString()} thinking` : "thinking tokens not reported",
    `${u.wall_time_s.toFixed(1)} s`,
  ].filter(Boolean).join(", ");
  const m = a.sim.metrics;
  const limits = a.verdict.thresholds;
  const measurements = [
    ["handover_completed", "Handover", m.handover_completed ? "Completed" : "Not completed", "Must complete"],
    ["dropped", "Lowest point of can", `${(m.can_min_z_m * 100).toFixed(1)} cm`, `At least -${(limits.drop_margin_m * 100).toFixed(0)} cm`],
    ["off target", "Distance from target", `${(m.can_final_xy_error_m * 100).toFixed(1)} cm`, `At most ${(limits.can_target_tolerance_m * 100).toFixed(0)} cm`],
    ["not at rest", "Final can speed", `${(m.can_final_speed_m_s * 100).toFixed(1)} cm/s`, `At most ${(limits.max_can_speed_at_end_m_s * 100).toFixed(0)} cm/s`],
    ["arm hit table", "Arm / table contact", `${m.max_arm_table_contact_force_n.toFixed(1)} N`, `At most ${limits.max_arm_table_contact_force_n} N`],
    ["arm hit obstacle", "Arm / obstacle contact", `${m.max_arm_obstacle_contact_force_n?.toFixed(1)} N`, `At most ${limits.max_arm_obstacle_contact_force_n} N`],
    ["arm inside obstacle", "Arm inside obstacle", `${(m.max_arm_obstacle_penetration_m * 1000).toFixed(1)} mm`, `At most ${limits.max_arm_obstacle_penetration_m * 1000} mm`],
    ["object inside obstacle", "Can inside obstacle", `${(m.max_object_obstacle_penetration_m * 1000).toFixed(1)} mm`, `At most ${limits.max_object_obstacle_penetration_m * 1000} mm`],
  ].filter(([prefix]) => a.verdict.failed_checks.some((check) => check.startsWith(prefix)));
  const priorities = ["arm hit obstacle", "object inside obstacle", "arm hit table", "arm inside obstacle", "handover_completed", "dropped", "off target", "not at rest"];
  measurements.sort((a, b) => priorities.indexOf(a[0]) - priorities.indexOf(b[0]));
  const measurementRows = (rows) => rows.map(([, name, value, limit]) => `<tr><th scope="row">${escapeHtml(name)}</th><td>${escapeHtml(value)}</td><td>${escapeHtml(limit)}</td></tr>`).join("");
  const steps = c.body.diagnosis;
  const frames = a.sim.keyframes;
  const cameras = new Set(frames.map((k) => k.camera)).size;
  const nextIndex = state.steps.findIndex((s) => s.kind === "attempt" && s.attempt.attempt === a.attempt + 1);
  const next = nextIndex >= 0 ? state.steps[nextIndex].attempt : null;
  const status = next ? `Attempt ${next.attempt}: ${next.verdict.passed ? "PASS" : "FAIL"}, measured by MuJoCo`
    : state.run?.summary?.state === "running" ? "The next simulation is running. This review will stay open."
    : "No next simulation is recorded.";
  subs.hidden = false;
  subs.innerHTML = `<div class="speaker">Gemini 4 Argon reviews attempt ${a.attempt}</div>
    <p class="review-input">${frames.length} rendered frames from ${cameras} cameras + scene geometry, executed plan, measurements and failed checks</p>
    <section class="review-section"><h2>1. Why MuJoCo failed it</h2>
      <div class="review-table-wrap"><table class="review-table"><thead><tr><th>Failed check</th><th>Measured</th><th>Required</th></tr></thead>
      <tbody>${measurementRows(measurements.slice(0, 3))}</tbody></table></div>
      ${measurements.length > 3 ? `<p class="review-note">${measurements.length} failed checks in total. All measurements below.</p>` : ""}</section>
    <section class="review-section"><h2>2. Argon's explanation</h2>
      <ol class="review-diagnosis">${steps.map((d) => `<li>${readableDiagnosis(d)}</li>`).join("")}</ol></section>
    <section class="review-section"><h2>3. Proposed changes</h2>
      <div class="review-table-wrap"><table class="review-table"><thead><tr><th>Plan parameter</th><th>Failed attempt</th><th>Proposed retry</th></tr></thead><tbody>${changes.join("")}</tbody></table></div>
      <p class="review-note">Positions are relative to the tabletop center: x across the arms, y away from the front camera, z up. Only listed values change.</p></section>
    <div class="review-next"><p>${escapeHtml(status)}</p><button type="button" id="review-next" ${next ? "" : "disabled"}>${next ? `Watch attempt ${next.attempt}` : "Waiting for retry"}</button></div>
    <details class="more" ${moreOpen ? "open" : ""}><summary>Full evidence, original response and call receipt</summary>
      <table class="review-table"><thead><tr><th>All failed checks</th><th>Measured</th><th>Required</th></tr></thead><tbody>${measurementRows(measurements)}</tbody></table>
      <ol>${steps.map((d) => `<li>${richText(d)}</li>`).join("")}</ol>
      <div class="review-frames">${c.body.cited_keyframes.map((label) => frames.find((k) => k.label === label)).filter(Boolean).map((k) => `<a href="${keyframeURL(a, k)}" target="_blank" rel="noopener"><img src="${keyframeURL(a, k)}" alt="${escapeHtml(k.label)}"><span>${escapeHtml(frameName(k))}, ${escapeHtml(k.camera.replace(/_/g, " "))}, ${k.t_s.toFixed(1)} s</span></a>`).join("")}</div>
      <p class="receipt">${state.run?.summary?.source === "REPLAY" ? "Recorded call" : "Call receipt"}: ${escapeHtml(receipt)}. Response ${escapeHtml(c.interaction.interaction_id)}</p>
    </details>`;
  $("review-next").onclick = () => { if (nextIndex >= 0) goTo(nextIndex); };
  subs.scrollTop = scrollTop;
  positionOverlays();

  const cited = c.body.cited_keyframes
    .map((label) => a.sim.keyframes.find((k) => k.label === label))
    .filter(Boolean)
    .slice(0, 2);
  ev.hidden = !cited.length || !$("planner").hidden;
  ev.innerHTML = `<div class="review-evidence-title">Frames Argon cited</div>` + cited.map((k) => `<figure><img alt="${escapeHtml(k.label)} frame from the ${escapeHtml(k.camera)} camera" src="${keyframeURL(a, k)}">
    <figcaption>Argon cited: ${escapeHtml(frameName(k))}, ${escapeHtml(k.camera.replace("_", " "))} camera, ${k.t_s.toFixed(1)} s</figcaption></figure>`).join("");
}

function renderAttemptEvidence(a) {
  const ev = $("evidence");
  // Gripper views first: the handover as the right wrist sees it, then the left wrist's grasp.
  const find = (phase, cam) => a.sim.keyframes.find((k) => k.phase === phase && k.camera === cam);
  const pick = [find("handover", "right_wrist"), find("grasp", "left_wrist"), find("handover", "overhead")]
    .filter(Boolean)
    .slice(0, 2);
  ev.hidden = !pick.length;
  ev.innerHTML = pick.map((k) => `<figure><img alt="${escapeHtml(k.label)} frame from the ${escapeHtml(k.camera)} camera" src="${keyframeURL(a, k)}">
    <figcaption>${escapeHtml(k.camera.replace("_", " "))} camera, ${escapeHtml(frameName(k))}</figcaption></figure>`).join("");
}

async function goTo(i, autoplay = true) {
  if (!state.steps.length) return;
  state.current = Math.max(0, Math.min(i, state.steps.length - 1));
  const step = state.steps[state.current];
  renderRail();
  if (step.kind === "attempt") {
    renderCritique(null);
    renderAttemptEvidence(step.attempt);
    renderMarkers(step.attempt);
    if (autoplay) holdVerdict(step.attempt);
    else { state.pendingVerdict = null; $("verdict").hidden = true; }
    await showRollout(step.attempt, autoplay);
    if (autoplay && !state.playing) releaseVerdict();
  } else if (step.kind === "critique") {
    $("verdict").hidden = true;
    state.pendingVerdict = null;
    renderCritique(step.attempt);
    renderMarkers(step.attempt, step.attempt.critique.body.patch.handover_point_m);
    await showRollout(step.attempt, false);
    seekToCited(step.attempt);
  } else {
    $("verdict").hidden = true;
    renderCritique(null);
    $("evidence").hidden = true;
  }
  $("replay").disabled = step.kind === "pending";
}

function render() {
  renderSource();
  buildSteps();
  if (state.current >= state.steps.length) state.current = Math.max(0, state.steps.length - 1);
  renderRail();
}

// --- Rollout playback ----------------------------------------------------------------

const stlLoader = new STLLoader();
const meshCache = new Map();

const ROLE_STYLE = {
  task:  { color: "#edeae2", opacity: 1, edges: null, emissive: "#3a352a" },
  prop:  { color: "#e3b341", opacity: 0.12, edges: 0.85 },
  arm:   { color: "#3b3f47", opacity: 1, edges: 0.85 },
  table: { color: "#101216", opacity: 1, edges: 0.75 },
};

function roleOf(bodyName) {
  const taskObject = state.run?.manifest?.scene?.task_object;
  if (bodyName === taskObject) return "task";
  if (bodyName === "world") return "table";
  if (bodyName.includes("/")) return "arm";
  return "prop";
}

// --- Gold scan style -----------------------------------------------------------------
// The physics twin drawn the way a LiDAR scan looks: surfaces sampled into dots,
// dotted outlines on simple shapes, a dot-grid floor, and one slow sweep of light.
// It is a rendering of the simulation's own geometry, not data captured from the video.

const SCAN_DENSITY = { arm: 110000, task: 90000, prop: 40000, table: 14000 };  // dots per m²
const SCAN_COLOR = { arm: "#ffd56b", task: "#fff6e3", prop: "#f0c45a", table: "#a47f2c" };
const sweep = { value: -10 };

const scanVertex = `
  uniform float uSize; uniform float uSweep; uniform float uMinPx;
  varying float vGlow; varying float vFade;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vec4 mv = viewMatrix * wp;
    gl_Position = projectionMatrix * mv;
    float d = wp.y - uSweep;
    vGlow = exp(-d * d / 0.0006);
    vFade = clamp(1.45 - (-mv.z) / 3.6, 0.45, 1.0);
    // Never smaller than uMinPx device pixels, so distant surfaces still read on a projector.
    gl_PointSize = max(uMinPx, uSize * (1.0 + 1.2 * vGlow) * (300.0 / -mv.z));
  }`;
const scanFragment = `
  uniform vec3 uColor; varying float vGlow; varying float vFade;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = dot(c, c);
    if (r > 0.25) discard;
    float a = smoothstep(0.25, 0.02, r);
    // The sweep brightens dots toward pale gold; everything else stays the base color.
    vec3 col = mix(uColor * (0.7 + 0.3 * vFade), vec3(1.0, 0.94, 0.78), vGlow * 0.75);
    gl_FragColor = vec4(col, a * (0.55 + 0.35 * vFade + 0.3 * vGlow));
  }`;

// Smallest dot in device pixels: about 2.2 px at 720 px tall, scaled with the canvas height.
// (scanMaterials is declared near the top so resize() can update it before the scene loads.)
function minDotPx(role) {
  const h = canvas.clientHeight || 720;
  const base = Math.min(2.6, Math.max(1.4, 2.2 * h / 720));
  return (role === "table" ? base * 0.75 : base) * Math.min(window.devicePixelRatio, 2);
}

function scanMaterial(role) {
  if (!scanMaterials.has(role)) {
    scanMaterials.set(role, new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(SCAN_COLOR[role]) },
        uSize: { value: { arm: 0.013, task: 0.016, prop: 0.011, table: 0.009 }[role] },
        uMinPx: { value: minDotPx(role) },
        uSweep: sweep,
      },
      vertexShader: scanVertex,
      fragmentShader: scanFragment,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,  // overlapping dots stay gold instead of adding up to white
    }));
  }
  return scanMaterials.get(role);
}

// Black stand-ins keep dots on the far side of a body from showing through it.
const occluderMaterial = new THREE.MeshBasicMaterial({ color: "#000000", polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 });

function sampleSurface(geometry, perM2, cap = 24000) {
  const geo = geometry.index ? geometry.toNonIndexed() : geometry;
  const pos = geo.attributes.position;
  const tris = pos.count / 3;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const areas = new Float32Array(tris);
  let total = 0;
  for (let t = 0; t < tris; t++) {
    a.fromBufferAttribute(pos, 3 * t); b.fromBufferAttribute(pos, 3 * t + 1); c.fromBufferAttribute(pos, 3 * t + 2);
    total += areas[t] = b.clone().sub(a).cross(c.clone().sub(a)).length() / 2;
  }
  const n = Math.max(40, Math.min(cap, Math.round(total * perM2)));
  const out = new Float32Array(n * 3);
  const cdf = new Float32Array(tris);
  let acc = 0;
  for (let t = 0; t < tris; t++) cdf[t] = (acc += areas[t] / total);
  for (let i = 0; i < n; i++) {
    const u = Math.random();
    let lo = 0, hi = tris - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    a.fromBufferAttribute(pos, 3 * lo); b.fromBufferAttribute(pos, 3 * lo + 1); c.fromBufferAttribute(pos, 3 * lo + 2);
    let r1 = Math.random(), r2 = Math.random();
    if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
    const p = a.clone().add(b.clone().sub(a).multiplyScalar(r1)).add(c.clone().sub(a).multiplyScalar(r2));
    out.set([p.x, p.y, p.z], 3 * i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(out, 3));
  return g;
}

function scanParts(geo, role, withOutline) {
  const parts = new THREE.Group();
  parts.userData.scan = true;
  parts.add(new THREE.Points(sampleSurface(geo, SCAN_DENSITY[role]), scanMaterial(role)));
  parts.add(new THREE.Mesh(geo, occluderMaterial));
  if (withOutline) {
    const line = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo, 25),
      new THREE.LineDashedMaterial({ color: SCAN_COLOR[role], dashSize: 0.006, gapSize: 0.006, transparent: true, opacity: 0.9 }),
    );
    line.computeLineDistances();
    parts.add(line);
  }
  return parts;
}

// Collision surfaces: the table and fitted props (laptop, screens) from this run's motion log,
// as MuJoCo placed them, drawn as bright outlines on top of any view so the splat-to-sim fit
// can be inspected. Arms are left out so the outlines stay readable.
const colliderMaterial = new THREE.LineBasicMaterial({ color: "#7fe0ff", transparent: true, opacity: 0.95, depthTest: false });

function applyColliders() {
  robotLayer.traverse((o) => { if (o.userData.collider) o.visible = state.showColliders; });
  $("colliders").setAttribute("aria-pressed", String(state.showColliders));
  setCaption();
}

// --- Object appearance from a segmented splat --------------------------------------------
// A scene object may name visual_splat_path: its own splat, with its origin at the object's
// base center in the table frame. In Photoreal it rides on the simulated body (so it moves
// with the physics) and the plain stand-in shape is hidden. Physics is unchanged.

function attachObjectAppearances() {
  const objects = state.run?.manifest?.scene?.objects ?? [];
  for (const o of objects) {
    if (!o.visual_splat_path) continue;
    const i = (state.bodyNames ?? []).indexOf(o.name);
    if (i < 0) continue;
    const body = state.bodies[i];
    body.traverse((c) => { if (c.userData.solid) c.userData.replacedInPhoto = true; });
    const url = `assets/${o.visual_splat_path}`;
    // The splat's origin is the object's base center in the table frame. Place it in the
    // body's local frame using the body's pose at the first frame, when the object is resting.
    const f0 = state.rollout.frames[0];
    const bodyPos = new THREE.Vector3(f0[i * 7], f0[i * 7 + 1], f0[i * 7 + 2]);
    const bodyQuatInv = new THREE.Quaternion(f0[i * 7 + 4], f0[i * 7 + 5], f0[i * 7 + 6], f0[i * 7 + 3]).invert();
    const baseLocal = new THREE.Vector3(o.pos_m[0], o.pos_m[1], o.pos_m[2]).sub(bodyPos).applyQuaternion(bodyQuatInv);
    // Render the object's own Gaussians (scale, rotation, opacity and color intact) with the same
    // splat renderer as the desk. Its viewer is a child of the moving body, so it follows the
    // physics; the library sorts and culls using the splat mesh's world matrix.
    const holder = new THREE.Group();
    holder.position.copy(baseLocal);
    holder.quaternion.copy(bodyQuatInv);
    holder.userData.photoAppearance = true;
    holder.visible = state.splatStyle === "photo";
    body.add(holder);
    loadObjectSplat(url).then((viewer) => {
      holder.add(viewer);
    }).catch((err) => {
      // Without its own splat the object keeps its plain shape in every view; no stand-in dots.
      body.remove(holder);
      body.traverse((c) => { if (c.userData.solid) c.userData.replacedInPhoto = false; });
      setCaption(`The ${o.name}'s own splat didn't load (${err.message}), so it shows as a plain shape.`);
      applyRenderStyle();
    });
  }
}

async function loadObjectSplat(url) {
  const GS = await import("@mkkellogg/gaussian-splats-3d");
  const viewer = new GS.DropInViewer({ dynamicScene: true, gpuAcceleratedSort: false, sharedMemoryForWorkers: false });
  await viewer.addSplatScenes([{ path: url }], false);
  return viewer;
}

function applyRenderStyle() {
  const scan = state.renderStyle === "scan";
  const photo = state.splatStyle === "photo";
  robotLayer.traverse((o) => {
    if (o.userData.scan) o.visible = scan;
    if (o.userData.solid) {
      // In Photoreal the trained splat is the table and the screens; the static proxy
      // boxes would cover its texture, so only the arms and the can stay solid.
      const staticProxy = o.userData.role === "prop" || o.userData.role === "table";
      o.visible = !scan && !(photo && (staticProxy || o.userData.replacedInPhoto));
    }
    if (o.userData.photoAppearance) o.visible = photo;
  });
  floorDots.visible = scan;
}

// Dot-grid floor under the table, fading with distance.
const floorDots = (() => {
  const pts = [], col = [];
  for (let x = -1.4; x <= 1.4001; x += 0.05) {
    for (let y = -1.4; y <= 1.4001; y += 0.05) {
      const r = Math.hypot(x, y);
      if (r > 1.4) continue;
      pts.push(x, y, -0.74);
      const k = 0.45 * (1 - r / 1.4);
      col.push(0.89 * k, 0.70 * k, 0.26 * k);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const dots = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.006, vertexColors: true, transparent: true, depthWrite: false }));
  dots.visible = false;
  return dots;
})();
world.add(floorDots);

function geomObject(g, role) {
  const style = ROLE_STYLE[role];
  const mat = new THREE.MeshStandardMaterial({
    color: style.color,
    emissive: style.emissive ?? "#000000",
    transparent: style.opacity < 1,
    opacity: style.opacity,
    roughness: 0.6,
    metalness: role === "arm" ? 0.35 : 0.05,
  });
  const s = g.size;
  const obj = new THREE.Group();
  obj.position.set(...g.pos);
  obj.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  const scanOn = state.renderStyle === "scan";

  if (g.type === "mesh") {
    const solid = new THREE.Mesh(new THREE.BufferGeometry(), mat);
    solid.userData.role = role;
    solid.userData.solid = true;
    solid.visible = !scanOn;
    obj.add(solid);
    if (g.mesh_file) {
      const url = `assets/meshes/${g.mesh_file}`;
      if (!meshCache.has(url)) meshCache.set(url, new Promise((ok, bad) => stlLoader.load(url, ok, undefined, bad)));
      meshCache.get(url).then((loaded) => {
        solid.geometry = loaded;
        const parts = scanParts(loaded, role, false);
        parts.visible = state.renderStyle === "scan";
        obj.add(parts);
      }).catch(() => setCaption(`Some arm meshes didn't load (${g.mesh_file}).`));
    }
    return obj;
  }

  let geo;
  switch (g.type) {
    case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case "sphere": geo = new THREE.SphereGeometry(s[0], 24, 16); break;
    case "ellipsoid": geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]); break;
    case "capsule": geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
    case "cylinder": geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 64).rotateX(Math.PI / 2); break;
    default:
      return null; // planes are not drawn; the floor is drawn separately
  }
  const solid = new THREE.Group();
  solid.userData.solid = true;
  solid.userData.role = role;
  solid.add(new THREE.Mesh(geo, mat));
  if (style.edges) {
    solid.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(geo, 25),
      new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: style.edges }),
    ));
  }
  solid.visible = !scanOn;
  const parts = scanParts(geo, role, role !== "task");
  parts.visible = scanOn;
  obj.add(solid, parts);
  // Collision outlines for the environment only (table, laptop, screens), not the arms.
  if (role === "prop" || role === "table") {
    const collider = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 1), colliderMaterial);
    collider.userData.collider = true;
    collider.visible = state.showColliders;
    obj.add(collider);
  }
  return obj;
}

async function showRollout(a, autoplay) {
  const keyId = `${state.runId}/${a.attempt}`;
  if (state.rolloutKey !== keyId) {
    state.playing = false;
    robotLayer.clear();
    state.bodies = [];
    state.rollout = null;
    state.rolloutKey = null;
    try {
      const url = `${state.base}attempt_${a.attempt}/${a.sim.rollout_path}`;
      let log = rolloutCache.get(url);
      if (!log) {
        // Public motion logs are large; retain successful downloads when stepping back.
        log = await getJSON(url, 60000);
        rolloutCache.set(url, log);
      }
      state.rolloutKey = keyId;
      state.rollout = log;
      state.bodyNames = log.bodies.map((b) => b.name);
      for (const body of log.bodies) {
        const group = new THREE.Group();
        const role = roleOf(body.name);
        for (const g of body.geoms) {
          const obj = geomObject(g, role);
          if (obj) group.add(obj);
        }
        robotLayer.add(group);
        state.bodies.push(group);
      }
      attachObjectAppearances();
      applyRenderStyle();
      const boxesOnly = log.bodies.every((b) => b.geoms.every((g) => g.type !== "mesh"));
      setCaption(boxesOnly ? "Arms drawn as simple boxes here; the camera frames show the real ALOHA meshes." : "");
    } catch (err) {
      state.pendingVerdict = null;
      $("verdict").hidden = true;
      setCaption(`This attempt's motion log isn't available (${err.message}). Select the attempt to retry.`);
      return;
    }
  }
  state.frame = 0;
  applyFrame(0);
  if (autoplay) play();
}

function seekToCited(a) {
  if (!state.rollout) return;
  const label = a.critique.body.cited_keyframes[0];
  const k = a.sim.keyframes.find((kf) => kf.label === label);
  if (!k) return;
  state.playing = false;
  state.frame = Math.min(state.rollout.frames.length - 1, Math.round(k.t_s * state.rollout.fps));
  applyFrame(state.frame);
}

function applyFrame(index) {
  const log = state.rollout;
  if (!log) return;
  const f = log.frames[Math.min(index, log.frames.length - 1)];
  state.bodies.forEach((group, b) => {
    const o = b * 7;
    group.position.set(f[o], f[o + 1], f[o + 2]);
    group.quaternion.set(f[o + 4], f[o + 5], f[o + 6], f[o + 3]);
  });
}

function play() {
  if (!state.rollout) return;
  state.frame = 0;
  state.playing = true;
  state.lastTick = performance.now();
}

// --- Desk scan ------------------------------------------------------------------------
// .splat layout: 32 bytes per splat = position f32x3, scale f32x3, rgba u8x4, rotation u8x4.

let splatBuffer = null;
let splatAlign = { scale: 1, position: [0, 0, 0], rotation_deg: [0, 0, 0] };
let captionText = "";
let realPoints = null;   // centers of the trained splat, drawn as reconstructed points
let photoViewer = null;  // Gaussian splatting renderer, loaded on first use
let photoNote = "";

const DEFAULT_SCAN_URL = "assets/desk.splat";  // default trained desk splat; a scene can name its own
let scanUrl = DEFAULT_SCAN_URL;   // background splat for the current run
let loadedScanUrl = null;         // what's actually loaded (null = nothing tried yet)

function backgroundUrlFor(run) {
  // A loaded run decides its own background; with no run, the profile's background is used.
  const path = run?.manifest ? run.manifest.scene?.background_splat_path : PROFILE.background;
  return path ? `assets/${path}` : DEFAULT_SCAN_URL;
}

// Switch the background splat when a run's scene names a different one. Old runs and the
// default demo name none, so they keep web/assets/desk.splat.
async function useBackground(url) {
  if (url === loadedScanUrl) return;
  if (realPoints) { splatLayer.remove(realPoints); realPoints.geometry.dispose(); realPoints = null; }
  if (photoViewer) { splatLayer.remove(photoViewer); photoViewer.dispose?.(); photoViewer = null; }
  splatBuffer = null;
  scanUrl = url;
  loadedScanUrl = url;
  await loadSplat();
}
const REAL_POINT_COLOR = new THREE.Color("#8fd0e3");  // reconstructed data: cool blue, never gold

const MODE_NOTES = {
  gold: "Gold: the simulation's own geometry drawn as dots and dotted outlines. Stylized, not captured from the video.",
  solid: "Solid: the simulation's geometry as plain surfaces.",
  photo: "Photoreal: the desk as a Gaussian splat trained from the 10-second phone video (soft, with some floaters), with the simulated arms on top.",
};

function setCaption(extra) {
  if (extra !== undefined) captionText = extra;
  const real = realPoints && state.splatStyle === "gold"
    ? " Blue dots are the desk reconstructed from the video, aligned to the sim table by fitting its plane and rim."
    : "";
  const colliders = state.showColliders ? " Blue outlines are the table and obstacle surfaces MuJoCo simulated in this run." : "";
  const segmented = state.splatStyle === "photo" && (state.run?.manifest?.scene?.objects ?? []).some((o) => o.visual_splat_path)
    ? " The can is its own splat, cut out of the scan, riding on the simulated can." : "";
  const base = MODE_NOTES[state.splatStyle] + real + segmented + colliders + (photoNote ? " " + photoNote : "");
  const text = [base, captionText].filter(Boolean).join(" ");
  $("caption").textContent = text;
  $("caption").title = text;  // full text on hover when narrow screens clamp it
}

function applyAlign(obj) {
  obj.scale.setScalar(splatAlign.scale);
  obj.position.set(...splatAlign.position);
  obj.rotation.set(...splatAlign.rotation_deg.map(THREE.MathUtils.degToRad));
}

function setPhotoAvailability(ready) {
  const btn = $("scan-modes").querySelector('[data-mode="photo"]');
  btn.disabled = !ready;
  btn.title = ready ? "Show the trained Gaussian splat of the desk" : "No trained desk splat loaded.";
  btn.textContent = ready ? "Photoreal" : "Photoreal (not ready)";
}

async function loadSplat() {
  setPhotoAvailability(false);
  try {
    const res = await fetch(scanUrl);
    if (!res.ok) throw new Error(String(res.status));
    splatBuffer = await res.arrayBuffer();
    try { splatAlign = { ...splatAlign, ...(await getJSON("assets/align.json")) }; } catch { /* identity until aligned */ }
    buildRealPoints();
    setPhotoAvailability(true);
  } catch {
    /* no trained splat yet: Photoreal stays disabled and says why */
  }
  await setScanMode(splatBuffer && SCAN_ORDER.includes(START_VIEW) ? START_VIEW : state.splatStyle);
}

function buildRealPoints() {
  const n = Math.floor(splatBuffer.byteLength / 32);
  const f32 = new Float32Array(splatBuffer);
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { pos[i * 3] = f32[i * 8]; pos[i * 3 + 1] = f32[i * 8 + 1]; pos[i * 3 + 2] = f32[i * 8 + 2]; }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  realPoints = new THREE.Points(geo, new THREE.PointsMaterial({ color: REAL_POINT_COLOR, size: 0.003, transparent: true, opacity: 0.8, depthWrite: false }));
  applyAlign(realPoints);
  splatLayer.add(realPoints);
}

async function ensurePhotoViewer() {
  if (photoViewer) return photoViewer;
  const GS = await import("@mkkellogg/gaussian-splats-3d");
  const viewer = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false });
  await viewer.addSplatScenes([{ path: scanUrl }], false);
  applyAlign(viewer);
  splatLayer.add(viewer);
  photoViewer = viewer;
  return viewer;
}

async function setScanMode(mode) {
  if (mode === "photo" && !splatBuffer) mode = state.splatStyle === "photo" ? "gold" : state.splatStyle;
  photoNote = "";
  if (mode === "photo") {
    try {
      await ensurePhotoViewer();
    } catch (err) {
      photoNote = `The photoreal renderer didn't load (${err.message}), so this is the gold view.`;
      mode = "gold";
    }
  }
  state.splatStyle = mode;
  state.renderStyle = mode === "gold" ? "scan" : "solid";
  applyRenderStyle();
  if (realPoints) realPoints.visible = mode === "gold";
  if (photoViewer) photoViewer.visible = mode === "photo";
  for (const b of $("scan-modes").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  setCaption();
}

// --- Controls ---------------------------------------------------------------------------

$("colliders").onclick = () => { state.showColliders = !state.showColliders; applyColliders(); };

$("scan-modes").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn && !btn.disabled) setScanMode(btn.dataset.mode);
});
const SCAN_ORDER = ["gold", "solid", "photo"];

$("replay").onclick = async () => {
  const step = state.steps[state.current];
  if (!step || step.kind === "pending") return;
  if (step.kind === "attempt") holdVerdict(step.attempt);
  await showRollout(step.attempt, true);
};

$("run").onclick = () => startRun();
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea")) return;
  if (e.key === "ArrowRight") goTo(state.current + 1);
  else if (e.key === "ArrowLeft") goTo(state.current - 1);
  else if (e.key === " ") { e.preventDefault(); $("replay").click(); }
  else if (e.key.toLowerCase() === "h") { setCloseup(false); }
  else if (e.key.toLowerCase() === "c") setCloseup(!state.closeup);
  else if (e.key.toLowerCase() === "o") { state.showColliders = !state.showColliders; applyColliders(); }
  else if (e.key.toLowerCase() === "t") {
    const next = SCAN_ORDER.slice(SCAN_ORDER.indexOf(state.splatStyle) + 1).concat(SCAN_ORDER)
      .find((m) => m !== state.splatStyle && (m !== "photo" || splatBuffer));
    setScanMode(next);
  }
});

async function pollLive() {
  while (state.source === "api" && state.run?.summary?.state === "running") {
    await new Promise((r) => setTimeout(r, 1000));
    const before = state.steps.filter((s) => s.kind !== "pending").length;
    state.run = await getJSON(`/api/runs/${state.runId}`);
    render();
    renderPhases();
    const now = state.steps.filter((s) => s.kind !== "pending").length;
    if (state.steps[state.current]?.kind === "critique") {
      // Keep the explanation readable while its retry finishes in the background.
      if (now !== before) renderCritique(state.steps[state.current].attempt);
    } else if (now !== before) {
      const review = state.steps.findIndex((s, i) => i >= before && s.kind === "critique");
      goTo(review >= 0 ? review : now - 1);
    }
  }
  render();
  renderPhases();
  renderPlanner();
}


// --- Verdict timing ------------------------------------------------------------------
// The verdict for an attempt appears after its replay finishes, so the audience watches
// the arms first and then reads what MuJoCo measured.

function holdVerdict(a) {
  state.pendingVerdict = reduceMotion ? null : a;
  if (reduceMotion) { renderVerdict(a); return; }
  $("verdict").hidden = true;
}

function releaseVerdict() {
  if (!state.pendingVerdict) return;
  renderVerdict(state.pendingVerdict);
  state.pendingVerdict = null;
}

// --- Markers -------------------------------------------------------------------------
// Target ring (green), the plan's handover point (gold), and the editor's draft (white).

const markerMat = {
  target: new THREE.MeshBasicMaterial({ color: "#7fd1a8", transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
  handover: new THREE.MeshBasicMaterial({ color: "#e3b341" }),
  patched: new THREE.MeshBasicMaterial({ color: "#7fd1a8" }),
  draft: new THREE.MeshBasicMaterial({ color: "#edeae2", wireframe: true }),
};

function pinMarker(point, mat, radius) {
  const g = new THREE.Group();
  const ball = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14), mat);
  ball.position.set(...point);
  const stem = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(point[0], point[1], 0), new THREE.Vector3(...point)]),
    new THREE.LineDashedMaterial({ color: mat.color, dashSize: 0.02, gapSize: 0.015 }),
  );
  stem.computeLineDistances();
  g.add(ball, stem);
  return g;
}

function renderMarkers(a, patchedPoint = null) {
  markerLayer.clear();
  const sc = state.run?.manifest?.scene ?? state.baseScene?.scene;
  if (sc) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.04, 48), markerMat.target);
    ring.position.set(sc.target_m[0], sc.target_m[1], 0.003);
    markerLayer.add(ring);
  }
  if (a) markerLayer.add(pinMarker(a.plan.handover_point_m, markerMat.handover, 0.018));
  if (patchedPoint) markerLayer.add(pinMarker(patchedPoint, markerMat.patched, 0.018));
  if (!$("planner").hidden && state.draft) markerLayer.add(pinMarker(state.draft.handover_point_m, markerMat.draft, 0.024));
}

function currentAttempt() {
  const step = state.steps[state.current];
  return step && step.kind !== "pending" ? step.attempt : null;
}

// --- Gripper close-up ----------------------------------------------------------------

// The close-up follows a real gripper body from the rollout: whichever gripper is
// closest to the task object in the current frame (the one holding or reaching for it).
function bodyWorld(name) {
  const i = (state.bodyNames ?? []).indexOf(name);
  return i < 0 ? null : state.bodies[i].getWorldPosition(new THREE.Vector3());
}

function closeupTarget() {
  const task = state.run?.manifest?.scene?.task_object ?? "can";
  const obj = bodyWorld(task);
  const grippers = (state.bodyNames ?? []).filter((n) => /gripper_base$/.test(n)).map(bodyWorld).filter(Boolean);
  if (!grippers.length) return obj;
  if (!obj) return grippers[0];
  grippers.sort((a, b) => a.distanceTo(obj) - b.distanceTo(obj));
  return grippers[0].clone().lerp(obj, 0.5);  // frame the fingers and the can together
}

function setCloseup(on) {
  state.closeup = on;
  $("closeup").setAttribute("aria-pressed", String(on));
  if (!on) { heroCamera(); return; }
  const target = closeupTarget();
  if (!target) { state.closeup = false; $("closeup").setAttribute("aria-pressed", "false"); return; }
  camera.position.set(target.x, target.y + 0.2, target.z + 0.48);
  controls.target.copy(target);
  controls.update();
}

function followCloseup() {
  if (!state.closeup) return;
  const target = closeupTarget();
  if (!target) return;
  const shift = target.clone().sub(controls.target);
  camera.position.add(shift);
  controls.target.copy(target);
}
$("closeup").onclick = () => setCloseup(!state.closeup);

// --- Live progress -------------------------------------------------------------------

const PHASE_TEXT = {
  running: (e) => `Attempt ${e.attempt} running`,
  judged: (e) => `Attempt ${e.attempt} ${e.passed ? "passed" : "failed"}`,
  diagnosing: () => "Argon diagnosing",
  patched: (e) => `Patch: ${Object.keys(e.changes ?? {}).map((k) => PATCH_NAMES[k]?.[0] ?? k).join(", ").toLowerCase() || "no change"}`,
  invalid: () => "Argon output invalid",
  done: (e) => (e.passed ? "Done: passed" : "Done: not passed"),
};

// The header wraps differently at every width, so measure it instead of guessing.
// Keep the verdict and critique card above the caption row, however tall it wraps.
function positionOverlays() {
  const stage = canvas.parentElement.getBoundingClientRect();
  const foot = document.querySelector(".stage-foot").getBoundingClientRect();
  const bottom = `${Math.ceil(stage.bottom - foot.top) + 12}px`;
  $("verdict").style.bottom = bottom;
  $("subtitles").style.bottom = bottom;
  const headerBottom = Math.max(...[".masthead", ".status"].map((selector) =>
    document.querySelector(selector).getBoundingClientRect().bottom));
  $("subtitles").style.maxHeight = `${Math.max(100, foot.top - headerBottom - 24)}px`;
}
new ResizeObserver(() => positionOverlays()).observe(document.querySelector(".stage-foot"));
new ResizeObserver(() => positionOverlays()).observe(document.querySelector(".status"));
new ResizeObserver(() => positionOverlays()).observe(document.querySelector(".masthead"));

function positionPhases() {
  const list = $("phases");
  if (list.hidden) return;
  const stage = canvas.parentElement.getBoundingClientRect();
  const bottoms = [".masthead", ".status"].map((sel) => {
    const el = document.querySelector(sel);
    return el && !el.hidden ? el.getBoundingClientRect().bottom - stage.top : 0;
  });
  list.style.top = `${Math.ceil(Math.max(...bottoms)) + 12}px`;
}

// Re-measure whenever the header changes size (buttons appear, fonts load, wrapping).
new ResizeObserver(() => positionPhases()).observe(document.querySelector(".status"));
new ResizeObserver(() => positionPhases()).observe(document.querySelector(".masthead"));

function renderPhases() {
  const list = $("phases");
  const events = state.run?.phases ?? [];
  if (state.source !== "api" || !events.length) { list.hidden = true; return; }
  list.hidden = false;
  const live = state.run.summary?.state === "running";
  list.innerHTML = events.slice(-6).map((e, i, arr) => {
    const cls = [i === arr.length - 1 && live ? "now" : "", e.phase === "judged" ? (e.passed ? "pass" : "fail") : "", e.phase === "done" ? (e.passed ? "pass" : "fail") : ""].join(" ");
    return `<li class="${cls}">${escapeHtml((PHASE_TEXT[e.phase] ?? (() => e.phase))(e))}</li>`;
  }).join("");
  positionPhases();
}

// --- Plan editor ---------------------------------------------------------------------

const HP_LIMIT = { x: [-1.5, 1.5], y: [-1.5, 1.5], z: [0, 0.8] };

async function loadBaseScene() {
  if (state.source !== "api") return;
  try {
    state.baseScene = await getJSON(`/api/scene?path=${encodeURIComponent(BASE_SCENE)}`);
    resetDraft();
    $("edit-plan").hidden = false;
    $("planner-base").textContent = `Starts from ${BASE_SCENE}. Edits apply to the next run only.`;
  } catch (err) {
    $("planner-base").textContent = `Couldn't load the base plan (${err.message}).`;
  }
}

function resetDraft() {
  const p = state.baseScene.plan;
  state.draft = { handover_point_m: [...p.handover_point_m], lift_height_m: p.lift_height_m };
  syncInputs();
}

function syncInputs() {
  const d = state.draft;
  ["x", "y", "z"].forEach((axis, i) => { $(`hp-${axis}`).value = d.handover_point_m[i].toFixed(2); });
  $("lift").value = d.lift_height_m.toFixed(2);
  renderPlanner();
}

function readInputs() {
  const errors = [];
  const num = (id, [lo, hi], label) => {
    const el = $(id);
    const v = Number(el.value);
    const ok = el.value.trim() !== "" && Number.isFinite(v) && v >= lo && v <= hi;
    el.setAttribute("aria-invalid", String(!ok));
    if (!ok) errors.push(`${label} must be a number from ${lo} to ${hi}`);
    return v;
  };
  const hp = ["x", "y", "z"].map((axis) => num(`hp-${axis}`, HP_LIMIT[axis], `Handover ${axis}`));
  const lift = num("lift", [0.03, 0.4], "Lift height");
  if (!errors.length) state.draft = { handover_point_m: hp, lift_height_m: lift };
  return errors;
}

function draftPatch() {
  if (!state.baseScene || !state.draft) return {};
  const base = state.baseScene.plan;
  const patch = {};
  if (state.draft.handover_point_m.some((v, i) => Math.abs(v - base.handover_point_m[i]) > 1e-9)) patch.handover_point_m = state.draft.handover_point_m;
  if (Math.abs(state.draft.lift_height_m - base.lift_height_m) > 1e-9) patch.lift_height_m = state.draft.lift_height_m;
  return patch;
}

function describePatch(patch) {
  const parts = Object.entries(patch).map(([k, v]) => {
    const [name, fmt] = PATCH_NAMES[k] ?? [k, String];
    return `${name} ${fmt(v)}`;
  });
  return parts.length ? parts.join(", ") : "No changes from the base plan";
}

function renderPlanner() {
  if (!state.baseScene || !state.draft) return;
  const patch = draftPatch();
  const sc = state.baseScene.scene;
  const dist = (b) => Math.hypot(...state.draft.handover_point_m.map((v, i) => v - b[i]));
  const nearest = Math.min(dist(sc.left_arm_base_m), dist(sc.right_arm_base_m));
  const reach = nearest > ARM_REACH_M ? ` Handover point is ${nearest.toFixed(2)} m from the nearest arm base; the arm reaches about ${ARM_REACH_M} m.` : "";
  $("planner-diff").textContent = describePatch(patch) + "." + reach;

  const running = state.source === "api" && state.run?.summary?.state === "running";
  const queue = $("planner-queue");
  queue.hidden = !(running && Object.keys(patch).length);
  if (!queue.hidden) queue.textContent = `Queued for the next run: ${describePatch(patch)}. The current run keeps its own plan.`;
  $("apply-run").disabled = running;
  $("apply-run").textContent = running ? "Apply when this run ends" : "Apply and run";
  renderMarkers(currentAttempt());
}

function onInput() {
  const errors = readInputs();
  const box = $("planner-error");
  box.hidden = !errors.length;
  box.textContent = errors.join(". ");
  renderPlanner();
}
["hp-x", "hp-y", "hp-z", "lift"].forEach((id) => $(id).addEventListener("input", onInput));

$("edit-plan").onclick = () => {
  const open = $("planner").hidden;
  $("planner").hidden = !open;
  $("edit-plan").setAttribute("aria-expanded", String(open));
  $("evidence").hidden = open || $("evidence").hidden;
  renderPlanner();
};
$("break-plan").onclick = () => {
  state.draft = { ...state.draft, handover_point_m: [...BREAK_HANDOVER] };
  syncInputs();
};
$("reset-plan").onclick = () => { resetDraft(); onInput(); };
$("apply-run").onclick = () => startRun();

async function startRun() {
  const errors = state.draft ? readInputs() : [];
  if (errors.length) { onInput(); return; }
  const patch = draftPatch();
  const body = { scene: BASE_SCENE };
  if (Object.keys(patch).length) body.initial_patch = patch;
  $("run").disabled = true;
  $("apply-run").disabled = true;
  try {
    const res = await fetch("/api/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const reply = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(reply.detail) ? reply.detail.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ") : reply.detail;
      throw new Error(detail ?? res.status);
    }
    await loadRun(reply.run_id);
    state.current = 0;
    render();
    renderPhases();
    setCloseup(false);
    pollLive();
  } catch (err) {
    $("planner-error").hidden = false;
    $("planner-error").textContent = `Couldn't start a run: ${err.message}`;
    $("source").textContent = `Couldn't start a run: ${err.message}`;
  } finally {
    $("run").disabled = false;
    renderPlanner();
  }
}

// --- Main ------------------------------------------------------------------------------

function tick(now) {
  if (state.playing && state.rollout) {
    const step = 1000 / state.rollout.fps / state.speed;  // speed > 1 only during the guided tour
    while (now - state.lastTick >= step) {
      state.lastTick += step;
      state.frame += 1;
      if (state.frame >= state.rollout.frames.length) {
        state.frame = state.rollout.frames.length - 1;
        state.playing = false;
        releaseVerdict();
        break;
      }
    }
    applyFrame(state.frame);
  }
  // One slow sweep of light up through the scene every 6 s (off for reduced motion).
  if (state.renderStyle === "scan" && !reduceMotion) {
    const t = (now / 1000) % 6;
    sweep.value = t < 3.2 ? -0.02 + (t / 3.2) * 0.75 : -10;
  } else {
    sweep.value = -10;
  }
  followCloseup();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

canvas.addEventListener("dblclick", () => setCloseup(false));

async function main() {
  if (PROFILE.label) {
    const tag = document.createElement("span");
    tag.className = "variant-tag";
    tag.textContent = PROFILE.label;
    document.querySelector(".masthead h1").after(tag);
  }
  state.source = await detectSource();
  if (state.source === "api") import("./compare.js").catch(() => {});
  let runs = [];
  try { runs = await listRuns(); } catch { /* no runs yet */ }
  const requested = new URLSearchParams(location.search).get("run");  // ?run=<id> picks a specific run
  // Default: a requested run, else a live one, else the pinned collision proof, else the
  // newest complete repair story (passed after more than one attempt), else the newest run.
  // Only runs on this profile's background count as its defaults, so an older layout is never
  // picked by accident; ?run=<id> still opens any run.
  const mine = runs.filter((r) => (r.scene_background ?? null) === PROFILE.background);
  const pick = runs.find((r) => r.run_id === requested)
    ?? runs.find((r) => r.state === "running")
    ?? mine.find((r) => r.run_id === PINNED_RUN)
    ?? mine.find((r) => r.passed && r.attempts > 1)
    ?? mine.find((r) => r.attempts > 0);
  if (pick) await loadRun(pick.run_id);
  render();
  renderPhases();
  heroCamera();
  loadBaseScene();
  await useBackground(backgroundUrlFor(state.run));
  requestAnimationFrame(tick);
  if (state.steps.length) await goTo(0);
  if (state.run?.summary?.state === "running") pollLive();
}

// --- Hooks for add-on panels (compare.js) ------------------------------------------------
// replay(runId, attempt): load a loop run (runs/<id>/manifest.json) and play that attempt.
// playAttempt(attemptResult, baseUrl, scene): play one AttemptResult from any folder laid out
//   like a loop run, i.e. `${baseUrl}attempt_<n>/<rollout and keyframes>`.
window.argonTwin = {
  async replay(runId, attempt = 1) {
    await loadRun(runId);
    render();
    renderPhases();
    const i = state.steps.findIndex((st) => st.kind === "attempt" && st.attempt.attempt === attempt);
    await goTo(i >= 0 ? i : 0);
  },
  async playAttempt(attemptResult, baseUrl, scene) {
    state.runId = `ext:${baseUrl}`;
    state.base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    state.run = {
      summary: { run_id: state.runId, source: "REPLAY", state: "done", created_at: null },
      manifest: { scene, attempts: [attemptResult] },
      log: [],
    };
    render();
    renderPhases();
    await goTo(0);
  },
};

const ready = main();

// Guided tour (tour.js): a scripted walk through the recorded Celsius run. It drives the
// viewer only through these hooks, so every live control keeps working as before.
import("./tour.js?v=4").then((m) => m.setup({
  ready, state, camera, controls, heroCamera, goTo, loadRun, render, renderPhases, renderMarkers, setScanMode, setCloseup,
  setColliders(on) { state.showColliders = on; applyColliders(); },
  splatReady: () => splatBuffer !== null,
})).catch((err) => console.warn("Tour unavailable:", err));

// The local comparison panel is loaded above only when the API is available.
