// Guided tour of the recorded Celsius repair (run 20261025-134505).
//
// A "Tour" button in the status bar, or ?tour=1 to start it after load (for recording).
// Esc, the Tour button again, Run the loop or Edit plan stop it and hand control back.
// It only replays the recorded run: no API calls, and it never blocks a live run.
// ?tour=1&capture=1 is the recording mode: the largest 16:9 frame the window allows, no
// cursor, no tour chrome, and one second of still frame before the first move and after the verdict.

const RUN_ID = "20261025-134505";
const LEAD_S = 1;   // still frame before anything moves
const TAIL_S = 1;   // still frame after the final verdict

// Exactly 60 s from window.__tourStart to window.__tourDone: a 1 s still, this 58 s timeline,
// then a 1 s still on the PASS verdict. Times are seconds after the first still ends.
// Every number is checked: 10.4 s / 52 frames / 68,192 Gaussians / Colab T4 from the capture
// receipts; 1.6 to 3.7 mm from capture/fit_collision_scene.py; the rest from
// web/replay/20261025-134505/verification.json (216.19 N, 20 keyframes, a 12.22 s Argon call,
// 5.05 mm from target and 0 N obstacle contact on attempt 2).
const T = { phone: 0, gaussians: 6, splat: 10, colliders: 16, views: 22, attempt1: 28, argon: 36, attempt2: 49, end: 58 };
const CAPTIONS = {
  phone: "10.4 s of iPhone video is the whole input",
  gaussians: "52 frames to 68,192 Gaussians on a Colab T4, cropped to the desk",
  splat: "The real desk: a Gaussian splat trained from that video",
  colliders: "Screens and laptop fitted as collision boxes, 1.6 to 3.7 mm plane fit",
  views: "One physics twin, three views",
  attempt1: "Attempt 1: the arm drives the can into the screens. 216 N. MuJoCo: FAIL",
  argon: "Gemini 4 Argon reads 20 frames and the physics, finds the cause, moves the handover point (12.2 s call)",
  attempt2: "Attempt 2: clears the screens. 5 mm from target, 0 N contact. PASS",
};
const VIEW_NOTES = { gold: "Gold: the scan", solid: "Solid: what physics collides with", photo: "Photoreal: the real desk" };

const params = new URLSearchParams(location.search);
const RECORDING = params.get("tour") === "1" && params.get("capture") === "1";

let hooks = null;
let active = null;  // { token } while a tour runs
let band = null;
let button = null;
let phoneEl = null;

class Stopped extends Error {}

export function setup(h) {
  hooks = h;
  button = document.createElement("button");
  button.type = "button";
  button.id = "tour";
  button.textContent = "Tour";
  button.title = "A guided walk through the recorded Celsius repair. Esc stops it.";
  button.setAttribute("aria-pressed", "false");
  button.onclick = () => (active ? stop() : start());
  const status = document.querySelector(".status");
  status.insertBefore(button, document.getElementById("replay"));

  // Live controls always win: they stop the tour and then do their normal job.
  for (const id of ["run", "edit-plan", "apply-run"]) {
    document.getElementById(id)?.addEventListener("click", () => stop(), true);
  }
  window.addEventListener("keydown", (e) => { if (e.key === "Escape" && active) stop(); });

  if (RECORDING) {
    document.documentElement.classList.add("tour-recording");
    window.dispatchEvent(new Event("resize"));  // re-fit the renderer and camera to the 16:9 frame
  }
  if (params.get("tour") === "1") h.ready.then(() => start());
}

function liveRunning() {
  return hooks.state.run?.summary?.state === "running";
}

async function start() {
  if (active) return;
  window.__tourDone = false;
  if (liveRunning()) {
    // Starting the tour would take the screen away from a live run; offer it afterwards instead.
    button.title = "The tour is available once the live run finishes.";
    return;
  }
  const token = {};
  active = token;
  document.body.classList.add("touring");
  button.textContent = "Stop tour";
  button.setAttribute("aria-pressed", "true");
  showBand();
  try {
    await run(token);
    window.__tourDone = true;  // a recorder or test can stop on this
    if (!RECORDING) stop();
  } catch (err) {
    if (!(err instanceof Stopped)) { console.error(err); stop(); }
  }
}

function stop() {
  if (!active) return;
  active = null;
  const { state, setColliders, heroCamera } = hooks;
  state.speed = 1;
  setColliders(false);
  heroCamera();
  document.body.classList.remove("touring");
  band?.remove();
  band = null;
  phoneEl?.remove();
  phoneEl = null;
  button.textContent = "Tour";
  button.setAttribute("aria-pressed", "false");
}

// --- Timeline ----------------------------------------------------------------------------

async function run(token) {
  const h = hooks;
  const { state } = h;
  const check = () => { if (active !== token) throw new Stopped(); };

  // Always the recorded run, labeled as a replay, starting from attempt 1's first frame.
  h.setCloseup(false);
  h.setColliders(false);
  state.playing = false;
  if (state.runId !== RUN_ID) { await h.loadRun(RUN_ID); h.render(); h.renderPhases(); }
  check();
  const attemptIndex = (n) => state.steps.findIndex((s) => s.kind === "attempt" && s.attempt.attempt === n);
  const critiqueIndex = state.steps.findIndex((s) => s.kind === "critique" && s.attempt.attempt === 1);
  await h.goTo(attemptIndex(1), false);
  check();
  if (h.splatReady()) await h.setScanMode("photo");
  await sleep(1200, check);  // let the can's own splat attach before the clock starts
  h.heroCamera();
  const phone = await phoneClip(check);

  // The first still: the phone clip's first frame and the opening caption, nothing moving.
  window.__tourStart = performance.now();
  setCaption(CAPTIONS.phone);
  const t0 = window.__tourStart + LEAD_S * 1000;
  const at = (s) => sleepUntil(t0 + s * 1000, check);

  await at(T.phone);
  phone.play();
  await at(T.gaussians); setCaption(CAPTIONS.gaussians);
  phone.close();
  await h.setScanMode("gold");
  orbitCamera(t0 + T.gaussians * 1000, (T.attempt1 - T.gaussians) * 1000, token);  // beside the timeline; never holds up a step

  await at(T.splat); setCaption(CAPTIONS.splat);
  if (h.splatReady()) await h.setScanMode("photo");
  await at(T.colliders); setCaption(CAPTIONS.colliders); h.setColliders(true);
  await at(T.views); h.setColliders(false);
  for (const [i, mode] of ["gold", "solid", "photo"].entries()) {
    await at(T.views + i * 2);
    if (mode === "photo" && !h.splatReady()) continue;
    await h.setScanMode(mode);
    setCaption(CAPTIONS.views, VIEW_NOTES[mode]);
  }

  await at(T.attempt1); setCaption(CAPTIONS.attempt1);
  h.heroCamera();  // the orbit ends here; snap to the front shot in case a frame was missed
  state.speed = rolloutSpeed(state, 1, T.argon - T.attempt1 - 0.6);
  await h.goTo(attemptIndex(1), true);

  await at(T.argon); setCaption(CAPTIONS.argon);
  state.speed = 1;
  await h.goTo(critiqueIndex, false);
  const a1 = state.steps[critiqueIndex].attempt;
  moveHandover(a1, critiqueIndex, t0 + (T.argon + 2) * 1000, 5000, check).catch(() => {});

  await at(T.attempt2); setCaption(CAPTIONS.attempt2);
  state.speed = rolloutSpeed(state, 2, T.end - T.attempt2 - 1.2);
  await h.goTo(attemptIndex(2), true);

  await at(T.end + TAIL_S);
  state.speed = 1;
}

// The phone walkthrough over the stage, sped up so the whole 10.4 s clip fits its step.
async function phoneClip(check) {
  const wrap = document.createElement("div");
  wrap.className = "tour-phone";
  const video = document.createElement("video");
  Object.assign(video, { src: "assets/desk-scan-blurred.mp4", muted: true, playsInline: true, preload: "auto" });
  wrap.append(video);
  document.querySelector(".stage").append(wrap);
  await Promise.race([
    new Promise((r) => video.addEventListener("canplaythrough", r, { once: true })),
    new Promise((r) => setTimeout(r, 5000)),
  ]);
  check();
  const play = () => {
    video.playbackRate = Math.max(1, (video.duration || 10.4) / (T.gaussians - T.phone - 0.3));
    video.play().catch(() => {});
  };
  const close = () => { wrap.classList.add("out"); setTimeout(() => wrap.remove(), 600); };
  phoneEl = wrap;
  return { play, close };
}

// Play an attempt's whole rollout in about `seconds`.
function rolloutSpeed(state, attempt, seconds) {
  const a = state.run.manifest.attempts.find((x) => x.attempt === attempt);
  return Math.max(1, a.sim.metrics.sim_time_s / seconds);
}

// One slow orbit from the Gaussians beat to attempt 1, ending exactly on the front hero shot.
function orbitCamera(startAt, duration, token) {
  const { camera, controls, heroCamera } = hooks;
  heroCamera();
  const target = controls.target.clone();
  const offset = camera.position.clone().sub(target);
  return new Promise((resolve) => {
    const frame = (now) => {
      if (active !== token) return resolve();
      if (now - startAt > duration + 250) return resolve();  // late frame: the timeline already moved on
      const u = Math.min(1, Math.max(0, (now - startAt) / duration));
      const az = -0.5 * Math.sin(Math.PI * u);          // swing about 29 degrees and back
      const zoom = 1 - 0.14 * Math.sin(Math.PI * u);    // drift in a little at the middle
      const x = offset.x * Math.cos(az) - offset.z * Math.sin(az);
      const z = offset.x * Math.sin(az) + offset.z * Math.cos(az);
      camera.position.set(target.x + x * zoom, target.y + offset.y * zoom, target.z + z * zoom);
      controls.target.copy(target);
      if (u < 1) requestAnimationFrame(frame);
      else { heroCamera(); resolve(); }
    };
    requestAnimationFrame(frame);
  });
}

// Slide a marker from the failed handover point to Argon's patched one, in 3D.
async function moveHandover(a, stepIndex, startAt, ms, check) {
  const from = a.plan.handover_point_m;
  const to = a.critique.body.patch.handover_point_m;
  if (!to) return;
  hooks.renderMarkers(a, from);
  await sleepUntil(startAt, check);
  const t0 = performance.now();
  await new Promise((resolve) => {
    const frame = (now) => {
      if (!active || hooks.state.current !== stepIndex) return resolve();  // the tour moved on
      const u = Math.min(1, (now - t0) / ms);
      const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      hooks.renderMarkers(a, from.map((v, i) => v + (to[i] - v) * e));
      if (u < 1) requestAnimationFrame(frame); else resolve();
    };
    requestAnimationFrame(frame);
  });
}

// --- Lower third ---------------------------------------------------------------------------

function showBand() {
  band = document.createElement("div");
  band.className = "tour-band";
  band.setAttribute("aria-live", "polite");
  band.innerHTML = `<p class="tour-caption"></p>
    <p class="tour-badge">Recorded run ${RUN_ID}<br>Weld grasp assist</p>`;
  document.body.append(band);
}

function setCaption(text, note = "") {
  const el = band?.querySelector(".tour-caption");
  if (!el) return;
  el.textContent = text;
  if (note) {
    const span = document.createElement("span");
    span.className = "tour-note";
    span.textContent = note;
    el.append(span);
  }
}

// --- Clock --------------------------------------------------------------------------------

function sleep(ms, check) {
  return sleepUntil(performance.now() + ms, check);
}

function sleepUntil(when, check) {
  return new Promise((resolve, reject) => {
    const wait = () => {
      try { check(); } catch (err) { reject(err); return; }
      const left = when - performance.now();
      if (left <= 0) resolve(); else setTimeout(wait, Math.min(left, 100));
    };
    wait();
  });
}
