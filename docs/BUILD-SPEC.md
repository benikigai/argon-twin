# BUILD-SPEC — ArgonTwin: Workspace Video → Physics Twin → Bimanual Hand-Over, Fixed in a Loop

Canonical spec for the Gemini 4 Argon SF Hackathon (Sun Oct 25, 2026, SHACK15). First written 10:25 AM from decisions Ben made in chat; this repo copy is the canonical one from 11:40 AM. Everything below is built during the event in this repo (`argon-twin`).

## 1. Track, user, trigger

- **Track:** Problem Statement Two, *More Dimensions*: "Build a system that renders what it made, watches the result, critiques it against the goal, and fixes it in a loop."
- **User:** a robotics engineer who wants to try a manipulation task in *their own* workspace before touching hardware.
- **Painful trigger:** building a simulation twin of a real desk by hand takes days (measuring objects, writing scene files, tuning grasps), so people skip sim and break things on the real robot.
- **Changed state:** a 15-second phone video of the desk becomes a physics twin with two robot arms, and Argon iterates the task until MuJoCo proves it works.

## 2. Before / after

| Before | After |
|---|---|
| A phone video of Ben's desk (can, laptop or iPad, phone) | A 3D Gaussian splat of the desk with two ALOHA arms placed on it, and a MuJoCo twin built from Argon's reading of the video |
| No plan for the task | Argon's hand-over plan: left arm picks the can, passes it to the right arm, right arm places it on a target |
| Attempt 1 fails (shown, with the reason MuJoCo measured) | Attempt N succeeds, verified by MuJoCo, not by Argon |

## 3. Decisive demo moment

On the Three.js viewer: the splat of Ben's real desk, the two arms overlaid. Attempt 1 plays and fails (can slips or drops at the hand-over). Argon's critique appears next to the frames it looked at, with the plan change it made. Attempt 2 or 3 plays and the can lands on the target, with MuJoCo's pass check shown.

### Viewer design

- **One stage, not a control panel.** The 3D view fills the screen: the photoreal splat of the desk with the arms and objects replayed in it. No slider column, no KPI wall; the rules ban "a dashboard as the main feature".
- **Attempt filmstrip along the bottom:** `Attempt 1 ✗ dropped at hand-over` → `Argon critique` → `Attempt 2 ✓ on target`. Clicking a card replays that attempt on the stage.
- **Right rail, one card at a time:** Argon's critique for the selected attempt, with the 2–3 keyframes it cited and the plan change it made.
- **One verdict badge from MuJoCo** (PASS/FAIL plus the metric that decided it), and a small mode pill (`LIVE_ARGON`, `REPLAY`, `grasp assist: weld`).
- **Camera insets:** small MuJoCo renders (overhead and wrist) next to the stage, and the live interaction/environment ID shown as proof the managed agent ran.
- **Out of scope for the viewer:** robot pickers, extra mechanisms, voice, yield statistics. Only a real trained splat is called a splat.
- **Controls:** `Run loop` and `Replay`. Nothing else on screen during the demo.

### Using Argon's long decode (output past 65,536 tokens) honestly

The new capability is output, not input: with `background=True` and `max_output_tokens` above 65,536, Argon can write a large, multi-file artifact in one pass. Track 2's own text says this ("generate high-fidelity 3D games and procedural worlds in a single pass"). Use it where one consistent pass genuinely matters:

- **One-pass world build (`argon.build_world`):** Argon gets the full desk video, the ALOHA model description and our contracts in one request (long input), and returns in one background call a `WorldBuildBody`: the `SceneSpec` (every object's size and position with evidence), the initial `HandoverPlan`, and declarative `SupplementalCheck`s. `sim.py` turns the scene into MJCF deterministically; supplemental checks are evaluated on measured metrics and shown, but never change pass/fail and are never executed as code.
- **Show real numbers only.** The viewer shows the `usage` returned by that call (output and thinking tokens, wall time). If a run comes in under 65,536 tokens, show that number as it is; don't pad the prompt to cross the line.
- **Not live on stage.** A long decode likely takes minutes (the builder guide says to allow 15–20 minute timeouts for `high` thinking). Run it during the day and once more before judging; on stage, run the fast critique → patch loop live and show the world build's receipt.
- **Don't do:** have Argon type out training trajectories or collision meshes. Trajectories come from MuJoCo rollouts; geometry comes from primitives and menagerie meshes. Token volume is not the point; a working twin is.
- **Name models accurately.** Only claim a model was used or trained if that exact model ran.

## 4. Happy path and recovery path

**Happy path**
1. Ben records a 15 s video of the workspace on his phone.
2. Splat: frames → COLMAP → gsplat training on a Colab GPU → `.ply`/`.splat` file. Runs in the background; not part of the 3-minute live demo.
   - **Input (decided 11:40):** `capture/IMG_1272.MOV` (0.5× ultra-wide, 10.4 s, 1080p60, full orbit of the round table) as the primary. Use COLMAP's `OPENCV` camera model for the ultra-wide distortion. Add `IMG_1273.MOV` (1×, close-up of laptop/iPad/webcam, more motion blur) as a second camera only if the first reconstruction works and time allows.
   - **Crop to the table.** Both videos show other attendees' faces in the background. Crop the splat to a box around the table before it goes in the public repo or on Vercel.
   - **Gold style is a render mode, not a different asset:** the viewer draws the trained splat's points in a gold point-cloud style, with a toggle to the true-color splat.
3. `argon.build_world`: Argon watches the video and returns a `WorldBuild` (scene, initial plan, supplemental checks, usage).
4. `sim.validate_scene`: builds the MJCF (table + objects + ALOHA from `mujoco_menagerie`) and reports problems before any attempt.
5. `sim.run_attempt`: scripted hand-over with differential IK (`mink`); MuJoCo simulates it, writes keyframes, the rollout log and the MJCF, and returns measured `AttemptMetrics`.
6. `schemas.judge`: decides pass/fail from the metrics with thresholds from `config.py`.
7. `argon.critique`: Argon reads the keyframes, metrics and plan, explains the failure and returns a `PlanPatch`; `schemas.apply_patch` makes the next plan. Up to `config.MAX_ATTEMPTS` (4) attempts or until `judge` passes.
8. Viewer replays every attempt over the splat, with Argon's critique per attempt.

**Recovery path:** if Argon returns invalid JSON, retry once with the validation error; if still invalid, keep the previous plan and mark the attempt `ARGON_INVALID`. If the network or API is down on stage, the viewer plays today's recorded run, labeled `REPLAY (recorded <time>)`.

## 5. Architecture, ownership and contracts

```
capture/IMG_1272.MOV ──► capture/ pipeline (frames → COLMAP → gsplat on Colab GPU) ──► web/assets/desk.splat
        │
        └──► argon.build_world ──► WorldBuild(scene, initial_plan, supplemental_checks)
                                          │
                              sim.validate_scene ──► sim.run_attempt ──► SimAttempt(metrics, keyframes, rollout)
                                          ▲                                     │
                                          │                          schemas.judge ──► Verdict (deterministic)
                                          │                                     │
                     schemas.apply_patch ◄── argon.critique ◄───────────────────┘   (loop.py, ≤ 4 attempts)
                                          │
server.py ── /api/runs, /api/runs/{id}, /api/runs/{id}/attempts/{n}/rollout ──► web/ (Three.js + gaussian-splats-3d)
```

### Ownership (Ben, 11:50): one writer per file

| Owner | Files | Notes |
|---|---|---|
| **Claude (integrator, design owner)** | `schemas.py`, `config.py`, `loop.py`, `server.py`, `web/`, `examples/`, `README.md`, `docs/` | Owns the contracts and pass/fail thresholds |
| **Antigravity / Argon** | `argon.py`, `managed_agent.py` | All Gemini and Google API calls. Reads the key only through `config.gemini_api_key()` |
| **Codex** | `sim.py`, `tests/` (physics and contract verification), `capture/` | Splat pipeline for `capture/IMG_1272.MOV` on Colab; MuJoCo + `mink` scripted hand-over |

Rules:
- Nobody edits another owner's file. If you need a change there, ask its owner.
- `schemas.py` and `config.py` are frozen at `CONTRACT_VERSION = "1.0.0"` / `CONFIG_VERSION = "1.0.0"`. Only the integrator changes them, and bumps the version.
- Pass/fail is decided only by `schemas.judge(metrics)` with thresholds in `config.py`. Model output never changes a verdict. Argon's `SupplementalCheck`s are declarative (metric, operator, value), evaluated by `schemas.evaluate_supplemental`, and shown as extra information only.
- `loop.py` imports only the interface functions below. Each owner can build and test independently against `examples/desk_manual.json`.

### Interfaces (module-level functions, exact signatures)

`sim.py` (Codex):
```python
def validate_scene(scene: SceneSpec) -> list[str]
    # Builds the MJCF; returns problems (unreachable target, overlapping objects). [] = OK.
    # Raises SimError only if the scene can't be compiled.

def run_attempt(scene: SceneSpec, plan: HandoverPlan, *, attempt: int, out_dir: Path,
                render: bool = True, seed: int = 0) -> SimAttempt
    # One scripted hand-over. Deterministic for the same inputs and seed.
    # Writes into out_dir: keyframe PNGs (Keyframe.path is relative to out_dir),
    # rollout.json (a RolloutLog), scene.xml. Measures; does not judge.
```

`argon.py` (Antigravity):
```python
def build_world(video_path: Path, *, frames_dir: Path | None = None) -> WorldBuild
    # One background long-decode call (config.WORLD_BUILD_*). Model returns WorldBuildBody.
    # Retries once with the validation error, then raises ArgonInvalidOutput.

def critique(scene: SceneSpec, plan: HandoverPlan, attempt: AttemptResult, *,
             attempt_dir: Path, previous: InteractionRef | None = None) -> Critique
    # Reads keyframes under attempt_dir plus attempt.sim.metrics and attempt.verdict.
    # Model returns CritiqueBody (diagnosis, cited_keyframes, patch). Chains on previous.
    # Retries once, then raises ArgonInvalidOutput.
```

`managed_agent.py` (Antigravity):
```python
def robustness_sweep(scene: SceneSpec, plan: HandoverPlan, *, trials: int = 20,
                     previous: InteractionRef | None = None) -> SweepResult
    # Parameter sweep in the managed agent's sandbox; trial counts are measured there.
```

`schemas.py` (integrator, pure functions):
```python
def judge(metrics: AttemptMetrics) -> Verdict
def apply_patch(plan: HandoverPlan, patch: PlanPatch) -> HandoverPlan
def evaluate_supplemental(checks: list[SupplementalCheck], metrics: AttemptMetrics) -> list[SupplementalCheckResult]
```

`config.py` (integrator): model IDs, paths, timeouts, thresholds, and `gemini_api_key() -> str` (environment or the mounted `.env`; raises `MissingSecret`, no fallback key).

`loop.py` (integrator):
```python
def run_loop(*, video_path: Path | None, scene_path: Path | None = None,
             max_attempts: int = config.MAX_ATTEMPTS, sweep: bool = False) -> RunManifest
    # video_path → argon.build_world; scene_path → MANUAL mode from examples/*.json.
```

### Contract notes

- Frame: meters, origin at the table-top center, +z up, +x from the left arm toward the right arm. An object's `pos_m` is the center of its base (z = 0 when resting on the table). `AttemptMetrics` heights are relative to the table top.
- `Keyframe.label` values are what `CritiqueBody.cited_keyframes` refers to.
- `RolloutLog.frames`: 7 floats per body per frame (pos xyz, quat wxyz, MuJoCo order), bodies in `RolloutLog.bodies` order.
- `ArgonUsage` fields are copied from the API response; unknown stays `None`, never estimated.
- Pydantic tuples become `prefixItems` in JSON Schema. If the Interactions API rejects that as a response schema, ask for JSON in the prompt and validate with `WorldBuildBody.model_validate_json` / `CritiqueBody.model_validate_json` instead. Don't change the models.
- Reference inputs: `examples/desk_manual.json` holds a hand-written `SceneSpec` and `HandoverPlan` (placeholder estimates of the round table and Celsius can in `IMG_1272.MOV`).

## 6. Tool and access status

| Tool | ID / package | Status at 10:25 AM |
|---|---|---|
| Gemini 4 Argon | `gemini-4-argon`, Interactions API, `google-genai >= 2.3.0` | Credits claimed at kickoff. Video-input request shape on the Interactions API: UNKNOWN, check the builder guide / `gemini-api-dev` skill before writing `argon.py` |
| Managed agent | `antigravity-preview-09-2026` with `agent_config={"type": "antigravity", "model": "gemini-4-argon"}`, `environment="remote"` | Untested |
| Fast checks | `gemini-3.8-flash` | Untested; optional |
| Physics | `mujoco`, `mink`, ALOHA from `mujoco_menagerie` via `robot_descriptions` (module `aloha_mj_description`) | Not installed; this session has no network. `robot_descriptions` git-clones menagerie on first use, so load it in slice 0 before venue Wi-Fi gets busy |
| Splat | `ffmpeg`, COLMAP, `gsplat` on a Colab GPU (`colab` CLI is installed) | Untested. Mac has no CUDA, so training must run on Colab (or a CUDA box) |
| Viewer | Three.js + `@mkkellogg/gaussian-splats-3d` | Untested |
| Secrets | 1Password Environment set up by Codex, mounted into the repo root as a gitignored `.env` with `GEMINI_API_KEY` only (the event key, not Ben's personal key). The app reads the key from the environment and has no fallback. Exact vault and item reference: event folder `03_Sponsor-Toolsets/argon-api-verification-2026-10-25.md`, kept out of this public repo | Key verified with a live Argon call at ~11:15. The mount must point at `~/Code/argon-twin`. The `op` CLI has no account configured, so don't rely on `op read` |
| Repo | Public: github.com/benikigai/argon-twin | Created and pushed 11:45 |

## 7. Truth labels and approval points

- Every attempt shows its `mode` in the viewer.
- **Grasp assist:** if friction grasping is unreliable, the can is attached with a MuJoCo weld once the gripper closes on it. If used, the viewer says `grasp assist: weld` on screen.
- Splat-to-sim alignment is set by hand (scale, rotation, offset saved to `web/assets/align.json`). Say so in the pitch.
- ALOHA models, `mink` and the splat library are open-source; the README credits them and lists them under "not built today".
- Numbers in the pitch come only from today's runs.
- Ben approves: the public repo push, the submission, and any purchase of Colab compute.

## 8. Acceptance checks and demo script

1. `python sim.py examples/desk_manual.json` runs one hand-over attempt and prints a `SimAttempt`; `tests/` pass.
2. `python loop.py --video capture/IMG_1272.MOV` produces a `WorldBuild` from Argon and at least one failed then one passed attempt, written as a `RunManifest` under `runs/`.
3. `uvicorn server:app --port 8000` → `http://127.0.0.1:8000` shows the splat, both arms, and replays each attempt with Argon's critique beside it.
4. Pulling the network cable switches the viewer to `REPLAY` with the label visible.
5. README has the "built today" table matching the repo.

**3-minute demo (no slides):**
- 0:00–0:20 Who and why: "Setting up a sim of your own desk takes days. Here's mine from a 15-second video."
- 0:20–0:50 Splat + Argon's `SceneSpec` with its evidence lines.
- 0:50–1:40 Attempt 1 fails; Argon's critique with the frames it used; the patch.
- 1:40–2:20 Passing attempt; MuJoCo pass check on screen.
- 2:20–3:00 What's live vs replay, what's built today, what's next (real hardware).

## 9. Vertical slices (times PDT)

Re-planned at 11:25 after a ~50 min slip (Ben: keep everything, compress). Repo: `~/Code/argon-twin`. Key comes from a 1Password Environment mounted into the repo as a gitignored `.env`; model and agent IDs live in `config.py`. Done: a live `gemini-4-argon` Interactions call passed at ~11:15; `schemas.py` and `config.py` frozen at 1.0.0 (11:55).

Owners per section 5. Checkpoint at 1:15: if `sim.run_attempt` isn't producing a hand-over, Ben decides what to cut.

| # | When | Slice | Done when |
|---|---|---|---|
| 0 | 11:25–11:45 | `.env` mount + `config.py`; install `mujoco mink robot_descriptions fastapi uvicorn`; load ALOHA in MuJoCo; confirm `.env` is ignored by git | ALOHA loads and steps; `git status` doesn't show `.env` |
| 1 | 11:25–11:50 (Ben, in parallel) | Record the 15 s video; start the splat job on a Colab GPU | Video in `capture/`, job running |
| 2 | 11:45–1:00 | `sim.py`: table + can from a hand-written `SceneSpec`, scripted hand-over with `mink`, metrics, keyframe renders, pose log | Acceptance check 1 |
| 3 | 1:00–2:00 | `argon.py` + `loop.py`: video → `SceneSpec`; critique → `PlanPatch`; loop ≤ 4 | Acceptance check 2 |
| 4 | 2:00–3:15 | Viewer: splat + arm/object replay + attempt timeline + mode labels | Acceptance check 3 |
| 5 | 3:15–3:45 | Antigravity sandbox runs a robustness sweep of the passing plan and returns the best parameters (`LIVE_ARGON_MANAGED_AGENT`) | One sandbox run shown in the viewer |
| 6 | 3:45–4:45 | Freeze scope; recorded backup run; README; public repo; 60 s video; Ben submits before 5:00 | Acceptance checks 4–5 |

If slice 2 isn't passing by 1:15, cut slice 5 without asking again.

Cut order if behind: slice 5, then the splat (fall back to a plain 3D desk in Three.js), then the third object.

## 10. Non-goals and fallbacks

- No model training or distillation, no yield statistics, no multi-page app.
- No real-robot control.
- The LLM never grades itself; MuJoCo decides pass/fail.
- Fallbacks: Colab splat fails → phone app capture (Scaniverse/Polycam) → plain Three.js desk. Argon video input fails → Argon reads 3–5 extracted frames instead. Managed agent slow → skip slice 5.
- Everything in this repo is written during the event. The README's "built during the hackathon" table must match what actually ships.
