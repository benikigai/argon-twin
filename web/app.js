// ArgonTwin viewer. Owner: integrator (Claude).
// Live data comes from the local server (/api/runs). When that isn't reachable
// (e.g. on Vercel) the viewer plays recorded runs from replay/ and says so.
//
// Keys: ← → step through the loop, space replays, T cycles the desk scan (gold, photoreal, off),
// H returns to the front hero camera.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const $ = (id) => document.getElementById(id);
const GOLD = new THREE.Color("#e3b341");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
// The demo starts from a hand-written plan whose handover point is out of both arms' reach,
// so attempt 1 fails for a visible reason and the loop has something real to fix.
const DEMO_SCENE = "examples/desk_demo_bad_handover.json";
const ARM_REACH_M = 0.7;  // rough ViperX reach from the arm base; used only for the on-screen note

const state = {
  source: null,      // "api" | "replay"
  runId: null,
  run: null,         // { summary, manifest, log }
  base: "",
  steps: [],         // [{ kind: "attempt" | "critique", attempt }]
  current: 0,
  splatStyle: "gold",
  rollout: null,
  rolloutKey: null,
  bodies: [],
  frame: 0,
  playing: false,
  lastTick: 0,
};

// --- Three.js stage -------------------------------------------------------------

const canvas = $("stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
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
world.add(splatLayer, robotLayer);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
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
  const flags = [];
  if (a.sim.metrics.grasp_assist_used) flags.push("Grasp assist: the can is welded to the gripper once it closes");
  if (a.mode === "ARGON_INVALID") flags.push("Argon's fix failed validation, so the plan was kept");
  if (a.mode === "MANUAL") flags.push("Plan written by hand");
  const sc = state.run?.manifest?.scene;
  if (sc) {
    const dist = (b) => Math.hypot(...a.plan.handover_point_m.map((v, i) => v - b[i]));
    const nearest = Math.min(dist(sc.left_arm_base_m), dist(sc.right_arm_base_m));
    if (nearest > ARM_REACH_M) flags.push(`Handover point ${nearest.toFixed(2)} m from the nearest arm base, beyond its reach`);
  }
  box.innerHTML = `<div class="word">${a.verdict.passed ? "Pass" : "Fail"}</div>
    <div class="reason">${escapeHtml(attemptHeadline(a))}</div>
    <div class="by">Measured by MuJoCo, attempt ${a.attempt}</div>
    ${flags.length ? `<div class="flags">${flags.map((f) => `<span class="flag">${escapeHtml(f)}</span>`).join("")}</div>` : ""}`;
}

function frameName(k) {
  return k.label.replace(new RegExp(`[_-]?${k.camera}$`), "").replace(/_/g, " ");
}

function keyframeURL(a, k) {
  return `${state.base}attempt_${a.attempt}/${k.path}`;
}

function renderCritique(a) {
  const subs = $("subtitles");
  const ev = $("evidence");
  if (!a?.critique) { subs.hidden = true; ev.hidden = true; return; }
  const c = a.critique;
  const changes = Object.entries(c.body.patch)
    .filter(([, v]) => v !== null)
    .map(([k, v]) => {
      const [name, fmt] = PATCH_NAMES[k] ?? [k, String];
      return `<span>${escapeHtml(name)} <b>${escapeHtml(fmt(v))}</b></span>`;
    });
  subs.hidden = false;
  subs.innerHTML = `<div class="speaker">Gemini 4 Argon</div>
    <ol>${c.body.diagnosis.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ol>
    <div class="change">${changes.join("")}</div>`;

  const cited = c.body.cited_keyframes
    .map((label) => a.sim.keyframes.find((k) => k.label === label))
    .filter(Boolean)
    .slice(0, 2);
  ev.hidden = !cited.length;
  ev.innerHTML = cited.map((k) => `<figure><img alt="${escapeHtml(k.label)} frame from the ${escapeHtml(k.camera)} camera" src="${keyframeURL(a, k)}">
    <figcaption>What Argon looked at: ${escapeHtml(frameName(k))}, ${k.t_s.toFixed(1)} s</figcaption></figure>`).join("");
}

function renderAttemptEvidence(a) {
  const ev = $("evidence");
  const pick = ["overhead", "front", "left_wrist", "right_wrist"]
    .map((cam) => [...a.sim.keyframes].reverse().find((k) => k.camera === cam))
    .filter(Boolean)
    .slice(0, 2);
  ev.hidden = !pick.length;
  ev.innerHTML = pick.map((k) => `<figure><img alt="${escapeHtml(k.label)} frame from the ${escapeHtml(k.camera)} camera" src="${keyframeURL(a, k)}">
    <figcaption>${escapeHtml(k.camera.replace("_", " "))} camera, ${escapeHtml(frameName(k))}</figcaption></figure>`).join("");
}

async function goTo(i) {
  if (!state.steps.length) return;
  state.current = Math.max(0, Math.min(i, state.steps.length - 1));
  const step = state.steps[state.current];
  renderRail();
  if (step.kind === "attempt") {
    renderVerdict(step.attempt);
    renderCritique(null);
    renderAttemptEvidence(step.attempt);
    await showRollout(step.attempt, true);
  } else if (step.kind === "critique") {
    $("verdict").hidden = true;
    renderCritique(step.attempt);
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
  let geo;
  switch (g.type) {
    case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case "sphere": geo = new THREE.SphereGeometry(s[0], 24, 16); break;
    case "ellipsoid": geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]); break;
    case "capsule": geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
    case "cylinder": geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 64).rotateX(Math.PI / 2); break;
    case "mesh": {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
      if (g.mesh_file) {
        const url = `assets/meshes/${g.mesh_file}`;
        if (!meshCache.has(url)) meshCache.set(url, new Promise((ok, bad) => stlLoader.load(url, ok, undefined, bad)));
        meshCache.get(url).then((loaded) => { mesh.geometry = loaded; }).catch(() => {
          setCaption(`Some arm meshes didn't load (${g.mesh_file}).`);
        });
      }
      geo = null;
      mesh.position.set(...g.pos);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      return mesh;
    }
    default:
      return null; // planes are not drawn; the scanned desk is the floor
  }
  const obj = new THREE.Group();
  obj.add(new THREE.Mesh(geo, mat));
  if (style.edges) {
    obj.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(geo, 25),
      new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: style.edges }),
    ));
  }
  obj.position.set(...g.pos);
  obj.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  return obj;
}

async function showRollout(a, autoplay) {
  const keyId = `${state.runId}/${a.attempt}`;
  if (state.rolloutKey !== keyId) {
    state.playing = false;
    robotLayer.clear();
    state.bodies = [];
    state.rollout = null;
    state.rolloutKey = keyId;
    try {
      const log = await getJSON(`${state.base}attempt_${a.attempt}/${a.sim.rollout_path}`, 10000);
      state.rollout = log;
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
      const boxesOnly = log.bodies.every((b) => b.geoms.every((g) => g.type !== "mesh"));
      setCaption(boxesOnly ? "Arms drawn as simple boxes here; the camera frames show the real ALOHA meshes." : "");
    } catch (err) {
      setCaption(`This attempt's motion log isn't available (${err.message})`);
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
let goldPoints = null;   // THREE.Points built from the splat file
let photoViewer = null;  // Gaussian splatting renderer, loaded on first use
let photoNote = "";

const SCAN_URL = "assets/desk.splat";
const SCAN_NOTES = {
  gold: "Desk scanned from a 10-second phone video, drawn as gold points.",
  photo: "Desk scanned from a 10-second phone video, rendered as Gaussian splats.",
  off: "Desk scan hidden.",
};

function setCaption(extra) {
  if (extra !== undefined) captionText = extra;
  const scan = splatBuffer
    ? `${SCAN_NOTES[state.splatStyle]}${state.splatStyle === "off" ? "" : " Alignment set by hand."}${photoNote ? " " + photoNote : ""}`
    : "Desk scan still being reconstructed.";
  $("caption").textContent = [scan, captionText].filter(Boolean).join(" ");
}

function applyAlign(obj) {
  obj.scale.setScalar(splatAlign.scale);
  obj.position.set(...splatAlign.position);
  obj.rotation.set(...splatAlign.rotation_deg.map(THREE.MathUtils.degToRad));
}

async function loadSplat() {
  try {
    const res = await fetch(SCAN_URL);
    if (!res.ok) throw new Error(String(res.status));
    splatBuffer = await res.arrayBuffer();
    try { splatAlign = { ...splatAlign, ...(await getJSON("assets/align.json")) }; } catch { /* identity until aligned */ }
    $("scan-modes").hidden = false;
    await setScanMode("gold");
  } catch {
    /* no scan yet: the rollout's own table is drawn instead */
  }
  setCaption();
}

function buildGoldPoints() {
  const n = Math.floor(splatBuffer.byteLength / 32);
  const f32 = new Float32Array(splatBuffer);
  const u8 = new Uint8Array(splatBuffer);
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = f32[i * 8]; pos[i * 3 + 1] = f32[i * 8 + 1]; pos[i * 3 + 2] = f32[i * 8 + 2];
    const lum = 0.3 + 0.7 * (0.299 * u8[i * 32 + 24] + 0.587 * u8[i * 32 + 25] + 0.114 * u8[i * 32 + 26]) / 255;
    col[i * 3] = GOLD.r * lum; col[i * 3 + 1] = GOLD.g * lum; col[i * 3 + 2] = GOLD.b * lum;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  goldPoints = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.0035, vertexColors: true, transparent: true, opacity: 0.9, sizeAttenuation: true }));
  applyAlign(goldPoints);
  splatLayer.add(goldPoints);
}

async function ensurePhotoViewer() {
  if (photoViewer) return photoViewer;
  const GS = await import("@mkkellogg/gaussian-splats-3d");
  const viewer = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false });
  await viewer.addSplatScenes([{ path: SCAN_URL, showLoadingUI: false }]);
  applyAlign(viewer);
  splatLayer.add(viewer);
  photoViewer = viewer;
  return viewer;
}

async function setScanMode(mode) {
  if (!splatBuffer) return;
  state.splatStyle = mode;
  for (const b of $("scan-modes").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  if (mode === "gold" && !goldPoints) buildGoldPoints();
  if (mode === "photo") {
    try {
      await ensurePhotoViewer();
      photoNote = "";
    } catch (err) {
      // Say what happened instead of quietly showing something else.
      photoNote = `Photoreal renderer didn't load (${err.message}); showing gold points.`;
      state.splatStyle = "gold";
      if (!goldPoints) buildGoldPoints();
    }
  }
  if (goldPoints) goldPoints.visible = state.splatStyle === "gold";
  if (photoViewer) photoViewer.visible = state.splatStyle === "photo";
  for (const b of $("scan-modes").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === state.splatStyle));
  setCaption();
}

// --- Controls ---------------------------------------------------------------------------

$("scan-modes").addEventListener("click", (e) => {
  const mode = e.target.closest("button")?.dataset.mode;
  if (mode) setScanMode(mode);
});
const SCAN_ORDER = ["gold", "photo", "off"];

$("replay").onclick = () => {
  const step = state.steps[state.current];
  if (step && step.kind !== "pending") showRollout(step.attempt, true);
};

$("run").onclick = async () => {
  $("run").disabled = true;
  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ scene: DEMO_SCENE }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail ?? res.status);
    await loadRun(body.run_id);
    state.current = 0;
    render();
    heroCamera();
    pollLive();
  } catch (err) {
    $("source").textContent = `Couldn't start a run: ${err.message}`;
  } finally {
    $("run").disabled = false;
  }
};

window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea")) return;
  if (e.key === "ArrowRight") goTo(state.current + 1);
  else if (e.key === "ArrowLeft") goTo(state.current - 1);
  else if (e.key === " ") { e.preventDefault(); $("replay").click(); }
  else if (e.key.toLowerCase() === "h") heroCamera();
  else if (e.key.toLowerCase() === "t") setScanMode(SCAN_ORDER[(SCAN_ORDER.indexOf(state.splatStyle) + 1) % 3]);
});

async function pollLive() {
  while (state.source === "api" && state.run?.summary?.state === "running") {
    await new Promise((r) => setTimeout(r, 1000));
    const before = state.steps.filter((s) => s.kind !== "pending").length;
    state.run = await getJSON(`/api/runs/${state.runId}`);
    render();
    const now = state.steps.filter((s) => s.kind !== "pending").length;
    if (now !== before) goTo(now - 1);
  }
  render();
}

// --- Main ------------------------------------------------------------------------------

function tick(now) {
  if (state.playing && state.rollout) {
    const step = 1000 / state.rollout.fps;
    while (now - state.lastTick >= step) {
      state.lastTick += step;
      state.frame += 1;
      if (state.frame >= state.rollout.frames.length) {
        state.frame = state.rollout.frames.length - 1;
        state.playing = false;
        break;
      }
    }
    applyFrame(state.frame);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

canvas.addEventListener("dblclick", heroCamera);

async function main() {
  state.source = await detectSource();
  let runs = [];
  try { runs = await listRuns(); } catch { /* no runs yet */ }
  const requested = new URLSearchParams(location.search).get("run");  // ?run=<id> picks a specific run
  const pick = runs.find((r) => r.run_id === requested) ?? runs.find((r) => r.state === "running") ?? runs.find((r) => r.attempts > 0);
  if (pick) await loadRun(pick.run_id);
  render();
  heroCamera();
  await loadSplat();
  requestAnimationFrame(tick);
  if (state.steps.length) await goTo(0);
  if (state.run?.summary?.state === "running") pollLive();
}

main();
