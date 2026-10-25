# BUILD-SPEC — ArgonTwin: Workspace Video → Physics Twin → Bimanual Hand-Over, Fixed in a Loop

Design spec for ArgonTwin, built at the Gemini 4 Argon SF Hackathon (Oct 25, 2026, SHACK15). Everything below was built during the event in this repo.

## 1. Problem, user, trigger

- **Track:** Problem Statement Two, *More Dimensions*: "Build a system that renders what it made, watches the result, critiques it against the goal, and fixes it in a loop."
- **User:** a robotics engineer bringing up a two-arm (ALOHA-style) manipulation station on a table they haven't used before, who has to try a task there before running it on hardware. The first real input is Ben's own desk at SHACK15, filmed on his phone.
- **Painful trigger:** every new table means a new simulation: measure the table and objects, write the scene file, tune the grasp and hand-over waypoints. That takes days, so the step gets skipped and untested motions run on the real arms, where an unreachable hand-over or a dropped object costs hardware time or damage.
- **Changed state:** a 10-second phone video of the table becomes a physics twin with two robot arms. A plan is tried in MuJoCo, and when it fails, Argon explains why and patches it, until MuJoCo's own measurements pass it.

### What one pass leaves out

Google's own showcase for Argon's 3D work is *Desk Racer* ([AI Studio guide](https://aistudio.google.com/learn/building-3d-games-and-worlds-with-gemini-4-argon), Oct 13, 2026): a go-kart game built on top of the author's desk from photos, checked with automated screenshots and geometric intersection checks. The guide's recommended loop is an "Adversarial Critic" subagent that rates rendered screenshots until the scene clears a bar such as 9/10. The track statement makes the same point from the other side: "A single pass is only the first draft."

ArgonTwin starts from the same kind of input, a real desk, and changes what the loop is checked against:

| | Desk Racer (Google showcase) | ArgonTwin |
|---|---|---|
| Output | A playable game scene | A two-arm hand-over plan for a real robot model (ALOHA, from MuJoCo Menagerie) |
| What the critic looks at | Rendered screenshots | Rendered keyframes from four cameras **and** measured physics metrics |
| Who decides it's done | A critic model's rating | `schemas.judge`: fixed thresholds on MuJoCo measurements. No model output changes the verdict |
| What a fix is | New scene code | A typed `PlanPatch` to bounded plan parameters, re-run in MuJoCo |
| Proof | Screenshots | A run manifest with metrics, keyframes, rollout, compiled scene and API receipts |

### Why this needs Argon

- **Video in, structured world out:** reading a phone walkaround of a cluttered table and returning every object's size and position with evidence, in one typed `WorldBuildBody`.
- **Multimodal critique:** reading 20 keyframes across four cameras alongside the measured metrics, and tying a failure to a specific plan parameter.
- **Managed Agents:** `antigravity-preview-09-2026` runs a screening sweep in a no-network Linux sandbox; the result is accepted only after proof-of-execution checks and a local MuJoCo re-run (section 5).

### People stay in control

- Nothing moves real hardware. The output is a plan a person reviews; running it on a robot is a separate, human decision (section 11).
- The model proposes; MuJoCo decides. Argon can't pass its own work.
- A person can edit or deliberately break the plan, and every attempt shows whether it came from a person (`MANUAL`) or from Argon.
- The API key stays on the local machine; the hosted page holds no secrets and makes no model calls. The sandbox has no network access.

### Runs live in under 3 minutes

The live demo starts from a failing plan and runs the loop: simulate, critique, patch, re-simulate, up to `config.MAX_ATTEMPTS` (4). In the primary proof run, `20261025-134505` (section 3), the two simulated attempts took 1.9 s and 1.1 s of wall time and the live Argon critique took 12.2 s. Earlier recorded loops: `20261025-133036` took 1.1 s and 0.8 s in simulation with a 23.7 s critique, `verified-no-hint-20261025-124355` took 14.5 s end to end with an 11.9 s critique, and `20261025-130553` spent 1.8 s in simulation and 34.1 s in the critique. One simulated attempt with 20 rendered keyframes takes about 0.7–1.2 s. Each critique is capped at `config.CRITIQUE_TIMEOUT_S` (180 s). The world build (minutes) is shown from a recording, not run live.

### Where this goes

ArgonTwin is a prototype of physics-refereed robot planning: capture a real space, build a physics twin, let Gemini Argon propose, watch, critique and fix, and let deterministic physics judge and score. Today that is two ALOHA arms, one hand-over task and one real desk. The same loop is where model-driven robots are heading:

- **A safety gate for model-driven robots.** As robots are driven by language and vision models, a model's plan goes through a physics referee before any motor moves. The model proposes; physics decides. This is the part the prototype already shows.
- **Rehearsal in your own space.** Before a home robot or humanoid tries a task in a specific kitchen, office or workcell, it rehearses on a twin of that space, and physics rejects moves that hit or drop things.
- **Verified training data.** Each passing rollout is a physics-checked trajectory on a real scanned scene, the kind of data robot policies need. The Gemma pilot (section 6) is a first step: trained on MuJoCo-verified repairs, it went from 4/12 to 11/12 in recorded local validation.
- **Scoring plans from policies and world models.** Learned world models predict plausible futures; a deterministic simulator measures them in newtons and millimeters. Argon sits between the two as the critic that reads frames and numbers and fixes the plan.

What doesn't carry over yet: the hand-over controller is scripted for ALOHA, grasps use the weld assist, and obstacles are fitted boxes. Other robot models can be swapped into the scene; a new robot or task still needs its own controller.

## 2. Before / after

| Before | After |
|---|---|
| A phone video of Ben's desk (can, laptop or iPad, phone) | A 3D Gaussian splat of the desk with two ALOHA arms placed on it, and a MuJoCo twin built from Argon's reading of the video |
| No plan for the task | A hand-over plan: one arm picks up the real can where it stands on the desk, passes it to the other arm, which places it on a target |
| Attempt 1 fails (shown, with the reason MuJoCo measured) | Attempt N succeeds, verified by MuJoCo, not by Argon |

## 3. Decisive demo moment

On the Three.js viewer: the splat of Ben's real desk, the two arms overlaid, and the real can, picked up from where it stands. Attempt 1 plays and fails because an arm hits a monitor. Argon's critique appears next to the frames it looked at, with the plan change it made. Attempt 2 plays, clears the monitors and puts the can on target, with MuJoCo's pass check shown.

### Primary proof run: `20261025-134505` (default)

The default demo: the viewer's root URL opens this run. Receipts: `runs/20261025-134505/verification.json` and `runs/celsius-variant/HANDOFF.md` on the demo machine; the run itself (manifest, input, receipt, keyframes, rollout, with local paths removed) ships in `web/replay/20261025-134505/`, which a fresh clone opens by default, labeled as a replay.

- **Start:** `examples/desk_celsius.json`, started live from the viewer (Apply and run) with the hand-over point edited to `[0.00, 0.21, 0.30]` m, inside the fitted screen region. The scene's own `handover_point_m` is deliberately out of both arms' reach, so the critique input contains no passing hand-over point.
- **The real can:** the task object is the Celsius can at its captured position `[0.34063638, 0.18702933, 0]` m. The right arm picks it up (`pickup_arm: "right"`, contract 1.0.3) and the left arm receives it and places it near `(-0.15, -0.28, 0)` m. In Photoreal, the can's own 943 captured Gaussians (segmented from the desk scan, with scale, rotation and opacity intact) move with the simulated can through pickup, hand-over and place; the background splat has the can removed (`desk-without-celsius.splat`, 976 Gaussians removed from the 17,157-Gaussian scan; `runs/celsius-variant/assets/extraction.json`).
- **Attempt 1 (`MANUAL`): FAIL.** The route runs through the screens: 216.194 N of arm–obstacle contact and 9.986 mm of can penetration. The left arm never grasps, and the can drops off the table.
- **Critique (`LIVE_ARGON`, `gemini-4-argon`):** 12.22 s, 28,025 input tokens and 469 output tokens; thinking tokens weren't reported. Argon sets `pregrasp_clearance_m` 0.06, `lift_height_m` 0.18 and `handover_point_m` `[0.05, -0.26, 0.26]`.
- **Attempt 2 (`LIVE_ARGON`): PASS.** The can lands 5.051 mm from target; all three obstacle metrics are zero. An independent MuJoCo repeat passes with the same error.
- **Checks:** 18 focused tests pass outside the sandbox. Gold, Solid and Photoreal were checked at 1280×720.
- **Limits:** the can's physics is an approximate cylinder (radius 0.029 m, height 0.155 m) with an assumed 0.35 kg mass; the obstacles are hand-fitted boxes (below); the grasp uses the weld assist, shown on screen; the splat is soft.

### Collision surfaces

The screens and the laptop are physics obstacles, but they're simple boxes, not a scan of the real shapes. Each box was fitted to COLMAP correspondences in one video frame (`0004.jpg`) in the trained splat's table frame, from hand-picked panel corners, inflated by 8 mm and sized to enclose the panel's tilt. They line up with the visual splat; they are not a dense physical mesh, and surfaces the video didn't see are uncertain. Runs from before these surfaces existed record their obstacle metrics as not measured rather than zero.

### Earlier proof run: `20261025-133036` (historical)

The previous default, still available at `?run=20261025-133036`. Receipts: `runs/20261025-133036/manifest.json` and `verification.json`. It uses a virtual can on the clear foreground side of the desk, with the left arm picking up.

- **Start:** `examples/desk_collision.json` plus an `initial_patch` putting the hand-over point at `[0.00, 0.21, 0.30]` m, inside the fitted screen region.
- **Attempt 1 (`MANUAL`): FAIL.** The can reaches the target (5.6 mm off), but the arm presses into a screen: 95.5166 N of arm–obstacle contact and 2.873 mm of arm penetration, both over the limits (5 N, 2 mm).
- **Critique (`LIVE_ARGON`):** 23.68 s, 27,319 input and 615 output tokens; thinking not reported. Patch: `handover_point_m` `[0.05, -0.26, 0.22]`, `lift_height_m` 0.14.
- **Attempt 2 (`LIVE_ARGON`): PASS** at 1.019 mm, all obstacle metrics zero; an independent re-run passes with the same metrics. Weld assist used.

### Viewer design

- **One stage, not a control panel.** The 3D view fills the screen: the photoreal splat of the desk with the arms and objects replayed in it. No slider column, no KPI wall.
- **Attempt filmstrip along the bottom:** `Attempt 1 ✗ arm hit screen` → `Argon critique` → `Attempt 2 ✓ on target, no contact`. Clicking a card replays that attempt on the stage.
- **Right rail, one card at a time:** Argon's critique for the selected attempt, with the 2–3 keyframes it cited and the plan change it made.
- **One verdict badge from MuJoCo** (PASS/FAIL plus the metric that decided it), and a small mode pill (`LIVE_ARGON`, `REPLAY`, `grasp assist: weld`).
- **Camera insets:** small MuJoCo renders (overhead and wrist) next to the stage, and the live interaction/environment ID shown as proof the managed agent ran.
- **Out of scope for the viewer:** robot pickers, extra mechanisms, voice, yield statistics. Only a real trained splat is called a splat.
- **Controls:** `Run loop` and `Replay`. Nothing else on screen during the demo.

### Using Argon's long decode (output past 65,536 tokens) honestly

The new capability is output, not input: with `background=True` and `max_output_tokens` above 65,536, Argon can write a large, multi-file artifact in one pass. Track 2's own text says this ("generate high-fidelity 3D games and procedural worlds in a single pass"). Use it where one consistent pass genuinely matters:

- **No answers in the prompt:** the critique prompt gives physical facts (each arm's conservative ~0.85 m reach envelope, the frame convention) but no passing plan values, and live runs start from scenes whose `handover_point_m` is out of reach. The world-build prompt still needs its own no-hint run before a world build is presented as Argon reading positions from the video.
- **One-pass world build (`argon.build_world`):** Argon gets the full desk video, the ALOHA model description and our contracts in one request (long input), and returns in one background call a `WorldBuildBody`: the `SceneSpec` (every object's size and position with evidence), the initial `HandoverPlan`, and declarative `SupplementalCheck`s. `sim.py` turns the scene into MJCF deterministically; supplemental checks are evaluated on measured metrics and shown, but never change pass/fail and are never executed as code.
- **Sequencing:** build this demonstration only after the core hand-over passes in `loop.py`. It must not change the frozen contracts (currently 1.0.3); any integration of a larger generated package is planned separately later. Report output tokens and thinking tokens as separate numbers (`ArgonUsage.output_tokens`, `ArgonUsage.thought_tokens`). No run has produced more than 65,536 output tokens yet.
- **Show real numbers only.** The viewer shows the `usage` returned by that call (output and thinking tokens, wall time). If a run comes in under 65,536 tokens, show that number as it is; don't pad the prompt to cross the line.
- **Not run live in the demo.** A long decode takes minutes (`config.WORLD_BUILD_TIMEOUT_S` is 20 minutes). Run it ahead of time; in the demo, run the fast critique → patch loop live and show the world build's recorded receipt.
- **Don't do:** have Argon type out training trajectories or collision meshes. Trajectories come from MuJoCo rollouts; geometry comes from primitives and menagerie meshes. Token volume is not the point; a working twin is.
- **Name models accurately.** Only claim a model was used or trained if that exact model ran.

## 4. Happy path and recovery path

**Happy path**
1. Record a 15 s video of the workspace on his phone.
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

**Recovery path:** see [Failure handling](#failure-handling). If the network or API is down, the viewer plays a recorded run, labeled `REPLAY (recorded <time>)`.

### Failure handling

No failure is silently replaced with a default. Every failure either stops the run or is labeled in the manifest and the viewer.

| Failure | Detected by | Response |
|---|---|---|
| Argon output fails schema validation | `WorldBuildBody` / `CritiqueBody` `model_validate_json` | Retry once with the exact validation error in the prompt (`config.ARGON_INVALID_RETRIES = 1`), then raise `ArgonInvalidOutput` |
| Critique cites a keyframe that doesn't exist in the attempt | `argon.critique` checks `cited_keyframes` against the attempt's keyframe labels | Treated as invalid output: same retry, then `ArgonInvalidOutput` |
| Critique returns a patch that changes nothing (all fields null) | `argon.critique` | Treated as invalid output: same retry, then `ArgonInvalidOutput` |
| `ArgonInvalidOutput` during the loop | `loop.py` | Keep the previous plan, run one more attempt marked `ARGON_INVALID`, then stop |
| Argon call too slow | Per-call timeouts: `WORLD_BUILD_TIMEOUT_S` (20 min), `CRITIQUE_TIMEOUT_S` (180 s) | Call fails; the run stops with the error |
| Scene can't be compiled | `sim.validate_scene` raises `SimError` | Run stops before any attempt |
| Scene compiles but is infeasible (unreachable target, overlapping objects) | `sim.validate_scene` returns problems | Reported before any attempt |
| Managed-agent result fails verification | `managed_agent.validate_sweep` raises `ManagedAgentError` | Sweep result rejected; see [Managed-agent sweep verification](#managed-agent-sweep-verification) |
| Missing API key | `config.gemini_api_key()` raises `MissingSecret` | No fallback key; the run stops |

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

### Module ownership: one writer per file

| Owner | Files | Notes |
|---|---|---|
| **Claude (integrator, design owner)** | `schemas.py`, `config.py`, `loop.py`, `server.py`, `export_replay.py`, `web/`, `examples/`, `README.md`, `docs/` except `docs/BUILD-SPEC.md` | Owns the contracts and pass/fail thresholds |
| **Claude (spec)** | `docs/BUILD-SPEC.md` | Owns this spec |
| **Antigravity / Argon** | `argon.py`, `managed_agent.py` | All Gemini and Google API calls. Reads the key only through `config.gemini_api_key()` |
| **Codex** | `sim.py`, `tests/` (physics and contract verification), `capture/`, `training/` | Splat pipeline for `capture/IMG_1272.MOV` on Colab; MuJoCo + `mink` scripted hand-over; Gemma 4 pilot |
| **Claude (spec)** | `gemma_local.py`, `compare.py`, `compare_api.py`, `web/compare.js`, `web/compare.css`, `tests/test_gemma_compare.py` | Base vs fine-tuned local Gemma comparison (section 6) |

Rules:
- Each file has one owner; changes to another owner's file go through that owner.
- `schemas.py` and `config.py` are frozen at `CONTRACT_VERSION = "1.0.3"` / `CONFIG_VERSION = "1.0.2"` (1.0.1 added finite-number validation, 1.0.2 optional obstacle contact metrics, 1.0.3 optional `SceneSpec.pickup_arm` (default `"left"`), `SceneSpec.background_splat_path` and `ObjectSpec.visual_splat_path`; splat paths are relative to `web/assets/`, and an object splat's origin is the object's base center). Only the integrator changes them, and bumps the version.
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
    # cited_keyframes must be labels from this attempt; the patch must change at least one field.
    # Retries once, then raises ArgonInvalidOutput.
```

`managed_agent.py` (Antigravity):
```python
def robustness_sweep(scene: SceneSpec, plan: HandoverPlan, *, trials: int = 20,
                     previous: InteractionRef | None = None) -> SweepResult
    # Screens `trials` perturbations of the plan in the managed agent's sandbox, verifies the
    # sandbox's proof of execution, then re-runs the chosen plan in local MuJoCo through judge.
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

### Hosting

- **Live backend:** `uvicorn server:app --port 8000` on a local machine (`http://127.0.0.1:8000`). Live runs start here; MuJoCo and Argon calls only ever run here.
- **Vercel now:** `https://argon-twin.vercel.app` serves the verified Celsius replay from `web/`, labeled with its recording time. The recorded failure, Argon critique and passing attempt are interactive.
- **Hosted boundary (`web/` only):** the public viewer never calls Gemini or runs MuJoCo and holds no secrets. New live runs and Gemma comparisons require the local server.
- `export_replay.py <run_id>` (integrator) copies a finished run's manifest, keyframes and rollout into `web/replay/<run_id>/` and updates `web/replay/index.json`. The viewer uses `/api/runs` when the local server answers and falls back to `web/replay/` otherwise.

### Managed-agent sweep verification

The sandbox runs a standard-library Python script with no network access (`network: {"allowlist": []}`). It screens plan perturbations against arm reach envelopes, obstacle bounds and tabletop clearance. **This is geometric screening, not MuJoCo physics**, and the receipt says so (`sandbox_evaluation: stdlib_geometric_screening_not_mujoco_physics`).

`validate_sweep` accepts the sandbox result only if all of these hold:

1. The returned `nonce` matches the one sent, and `input_sha256` matches the SHA-256 of the exact input file sent.
2. `platform` is `Linux`.
3. The trial count matches the request, and `passed_trials` is within `[0, trials]`.
4. The sandbox echoes the task object's position from the input scene (to 1e-6 m).
5. The returned `PlanPatch` applies cleanly to the input plan (`schemas.apply_patch` validates).
6. The interaction contains a non-error `code_execution_result`, tied to a `code_execution_call` in the same interaction, whose output contains `SANDBOX_PROOF <nonce> <input_sha256>`.

Then `robustness_sweep` runs the patched plan once in local MuJoCo (`sim.run_attempt` + `schemas.judge`). If it fails, the original passing plan is kept (`fallback_to_input_plan: true`). If the local check can't run, the receipt records `executed: false` and the sweep's plan counts as unverified.

### Run records

Every run writes `runs/<run_id>/`:

- `manifest.json`: a `RunManifest` with `contract_version`, `created_at`, the scene, every attempt's plan, mode, measured metrics, verdict, supplemental checks and critique, and the sweep result.
- Per attempt: keyframe PNGs, `rollout.json` (a `RolloutLog`), `scene.xml` (the compiled MJCF), `sim-attempt.json`.
- Per sweep: `receipt.json` (interaction ID, environment ID, code-execution call ID, agent and model IDs, usage, sandbox result and local MuJoCo check), `result.json`, the exact `sweep.py` sent, and the local check's rollout.

Rules:

- Model usage (tokens, thinking tokens, wall time) and interaction IDs are copied from the API response. Unknown values stay `None`; nothing is estimated.
- The manifest is rewritten after every attempt, so a crashed run still leaves a record up to its last attempt.
- `runs/` is gitignored because it holds large renders and frames from a video that shows bystanders. Any run quoted in the README or the demo is exported with `export_replay.py` into `web/replay/<run_id>/`, which is committed, so every public number points to a committed file. `export_replay.py` copies `manifest.json`, `input.json`, the run's verification receipt, keyframes and rollout, rewrites repo paths to repo-relative ones and drops other local paths, and records each run's `scene_background` in `web/replay/index.json`. When a run isn't in the local `runs/`, the server reads it from `web/replay/` and labels it REPLAY. The sweep's `sweep/receipt.json` isn't copied yet; until it is, the sweep's IDs come from the manifest's `sweep` field.

### Contract notes

- Frame: meters, origin at the table-top center, +z up, +x from the left arm toward the right arm. An object's `pos_m` is the center of its base (z = 0 when resting on the table). `AttemptMetrics` heights are relative to the table top.
- `Keyframe.label` values are what `CritiqueBody.cited_keyframes` refers to.
- `RolloutLog.frames`: 7 floats per body per frame (pos xyz, quat wxyz, MuJoCo order), bodies in `RolloutLog.bodies` order.
- `ArgonUsage` fields are copied from the API response; unknown stays `None`, never estimated.
- Pydantic tuples become `prefixItems` in JSON Schema. If the Interactions API rejects that as a response schema, ask for JSON in the prompt and validate with `WorldBuildBody.model_validate_json` / `CritiqueBody.model_validate_json` instead. Don't change the models.
- Reference inputs: `examples/desk_manual.json` holds a hand-written `SceneSpec` and `HandoverPlan` (placeholder estimates of the round table and Celsius can in `IMG_1272.MOV`).
- Demo start: one-click "Run the loop" uses `examples/desk_celsius_demo.json`: the same scene as `examples/desk_celsius.json` with the failing plan (hand-over point `[0.00, 0.21, 0.30]` m, inside the fitted screen region) baked in. The proof run `20261025-134505` was started from `desk_celsius.json` plus that hand-over point as an `initial_patch`; `desk_celsius.json` stays unchanged as the safe fixture. The earlier collision layout keeps its own pair: `examples/desk_collision_demo.json` (failing plan baked in) and `examples/desk_collision.json` (safe fixture). `examples/desk_demo_bad_handover.json` is legacy: its scene carries a reachable `handover_point_m`, so it isn't used for live runs.
- Obstacle metrics (contract 1.0.2): `AttemptMetrics.max_arm_obstacle_contact_force_n`, `max_arm_obstacle_penetration_m` and `max_object_obstacle_penetration_m`, judged against `config.MAX_ARM_OBSTACLE_CONTACT_FORCE_N` (5 N), `MAX_ARM_OBSTACLE_PENETRATION_M` (2 mm) and `MAX_OBJECT_OBSTACLE_PENETRATION_M` (2 mm). `None` means not measured (older runs); `Verdict.obstacle_contact_measured` says which.

## 6. Tools and status

| Tool | ID / package | Status |
|---|---|---|
| Gemini 4 Argon | `gemini-4-argon` via `google-genai` | Live critique repairs verified: primary `20261025-134505` (real can, collision), earlier `20261025-133036`. World build implemented, no no-hint run yet |
| Managed agent | `antigravity-preview-09-2026` with `agent_config={"type": "antigravity", "model": "gemini-4-argon"}`, `environment="remote"` | One verified sweep with receipt (sandbox screening, then local MuJoCo check) |
| Physics | `mujoco`, `mink`, ALOHA from `mujoco_menagerie` via `robot_descriptions` (module `aloha_mj_description`) | Obstacle contact measured against fitted screen and laptop boxes; 18 focused physics and API tests pass |
| Splat | `ffmpeg`, COLMAP, `gsplat` on a Colab GPU | Trained from 52 registered views and cropped to the desk: 17,157 Gaussians, committed as `web/assets/desk.splat` |
| Viewer | Three.js + `@mkkellogg/gaussian-splats-3d` | Real ALOHA meshes; Gold, Solid and Photoreal modes verified in the browser with the trained desk splat |
| Secrets | `GEMINI_API_KEY` from the environment or a gitignored `.env` mounted from a password manager | No fallback key; the key never enters git or the hosted page |
| Repo | Public: github.com/benikigai/argon-twin | |

### Local Gemma repair: base vs fine-tuned

A small controller-repair pilot, served on the local machine and compared live in the viewer (`Gemma repair test`).

- **Model:** Gemma 4 E4B (existing Q4_K_M base) with a LoRA adapter, served by `llama-server` at `http://127.0.0.1:8767/v1` as `argon-twin-repair`. Serving files and receipts live outside this repo (`~/Models/argon-twin/gemma-4-e4b/local-serving/`). No cloud inference.
- **Training:** 36 synthetic, MuJoCo-verified repair pairs; 12 held-out cases from four other scene variants. Examples cover hand-over-point and grasp-height corrections only. Labels are scripted, simulator-verified repairs, not Argon's. The model reads structured failure data (scene, plan, metrics, failed checks); it doesn't see the video, control motors or decide success.
- **Fair comparison (`compare.py`):** one held-out validation case's failed plan is re-run in the current simulator and judged. Both variants get the identical request (system prompt from the serving folder's `repair_client.py`, JSON schema `plan-patch.schema.json`, temperature 0, `max_tokens` 192, `cache_prompt: false`) except the per-request adapter scale: `"lora": [{"id": 0, "scale": 0.0}]` for base, `1.0` for fine-tuned. The server's global adapter setting is read, never changed. The scene and plan go to the model exactly as the validation case recorded them (re-serializing through a newer contract would add fields it never saw in training); only the metrics and failed checks come from the fresh measurement. Each returned patch is validated against the current `PlanPatch`, applied to its own copy of the original plan, and run on the same scene with the same seed through `sim.run_attempt` and `schemas.judge`. Model response time and simulation time are recorded separately.
- **Honest failures:** invalid output, timeout, server unavailable and HTTP errors are recorded and shown as such; no patch is applied and no other model is substituted. Physics failures show the judge's failed checks.
- **Records:** `runs/compare-<id>/compare.json` holds both requests and responses, patches, metrics, verdicts, timings and provenance (code hashes, adapter receipt, server state). Attempts live in `original/attempt_1/`, `base/attempt_2/` and `tuned/attempt_2/`, replayable on the main stage.
- **Live result (validation example `scene-13-variant-2`):** runs `compare-20261025-133604` and `compare-20261025-134202` gave identical results. The same failed attempt (FAIL: hand-over not completed, arm hit table 96.2 N, can inside obstacle 14.1 mm); base proposed `pregrasp_clearance_m` 0.08 → 0.12 and failed (arm hit table 442.0 N); fine-tuned proposed `grasp_height_offset_m` −0.05 → 0.0 and passed, can 4.0 mm from target. Model responses took 1.25–1.41 s each; each simulation about 0.8–1.2 s. The failure is re-measured by today's simulator, so its metrics differ from the recorded case's input; both patches match the recorded predictions exactly.
- **Recorded local validation (12 held-out cases, observed 2026-10-25 20:24 UTC):** base 4/12 MuJoCo passes, fine-tuned 11/12; both 12/12 valid patches; median response 0.69 s base, 0.81 s fine-tuned, excluding simulation. Shown separately from live runs. The Colab NF4 results (11/12 vs 12/12) use a different runtime and base quantization and aren't combined with these.

## 7. Dependencies and tests

### Dependencies

Python 3.12. Exact versions are pinned in `requirements.txt`, matching the environment the recorded runs used:

```
mujoco==3.17.0
mink==1.3.0
robot_descriptions==3.3.0
google-genai==2.33.0
pydantic==2.14.1
fastapi==0.162.0
uvicorn==0.54.0
numpy==2.5.4
pillow==13.0.0
httpx==0.28.1
pytest==9.1.1
```

`robot_descriptions` downloads MuJoCo Menagerie on first use, so the first run needs network access.

### Tests

All tests run offline with no API key. Argon and the managed agent are tested against recorded responses, never live calls.

| Area | Test | Status |
|---|---|---|
| Physics | Nominal hand-over passes and is bit-for-bit repeatable for the same seed | Done (`tests/test_sim.py`) |
| Physics | An unreachable hand-over point can't pass | Done |
| Physics | `validate_scene` reports an unreachable target | Done |
| Collision | A clear foreground hand-over passes the fitted surfaces; a screen collision fails even when the can reaches the target; older runs' obstacle metrics read as not measured, not zero | Done (`tests/test_collision_surfaces.py`) |
| API | Live-control `initial_patch` is snapshotted per run; invalid controls are rejected before running; `/api/health` never reads the credential pipe | Done (`tests/test_live_controls.py`) |
| Gemma comparison | Both variants' requests are identical except the adapter scale; out-of-bounds, unknown-field, empty, non-JSON and truncated outputs are rejected, not applied; refused connection, timeout and HTTP errors are reported with no fallback | Done (`tests/test_gemma_compare.py`, offline) |
| Contracts | `judge`: each threshold in `config.py` fails just past its limit and passes just inside it | To do |
| Contracts | `apply_patch`: null fields keep their values; the result revalidates against `HandoverPlan` bounds | To do |
| Contracts | `evaluate_supplemental` never changes the verdict | To do |
| Contracts | Models reject unknown fields and out-of-range values | To do |
| Sweep verification | `validate_sweep` rejects a wrong nonce, wrong hash, non-Linux platform, trial-count mismatch, wrong echoed position, invalid patch, and missing or errored proof | To do |
| Critique validation | Unknown keyframe label and all-null patch are rejected, then raise `ArgonInvalidOutput` after one retry | To do |
| Loop | `ArgonInvalidOutput` keeps the previous plan and marks the next attempt `ARGON_INVALID` | To do (stubs exist) |

CI: a GitHub Actions workflow runs `pytest` on every push (to do).

## 8. Truth labels

- Every attempt shows its `mode` in the viewer.
- **Grasp assist:** if friction grasping is unreliable, the can is attached with a MuJoCo weld once the gripper closes on it. If used, the viewer says `grasp assist: weld` on screen. Every verified run so far, including the primary proof run `20261025-134505`, uses it.
- **Collision boxes:** the screens and laptop are fitted, conservative boxes aligned to the splat, not a dense physical mesh (section 3).
- Splat-to-sim alignment comes from fitting the reconstructed table's plane and rim to the sim table; `desk.splat` is stored already in sim coordinates, so `web/assets/align.json` is the identity. Captions say how it was aligned, and that the photoreal splat is soft with some floaters.
- ALOHA models, `mink` and the splat library are open-source; the README credits them and lists them under "not built today".
- Numbers in the README and the demo come only from recorded runs, with their run IDs.

## 9. Acceptance checks and demo script

1. `python sim.py examples/desk_manual.json` runs one hand-over attempt and prints a `SimAttempt`; `pytest` passes offline.
2. `python loop.py --video capture/IMG_1272.MOV` produces a `WorldBuild` from Argon and at least one failed then one passed attempt, written as a `RunManifest` under `runs/`.
3. `uvicorn server:app --port 8000` → `http://127.0.0.1:8000` shows the splat, both arms, and replays each attempt with Argon's critique beside it.
4. Pulling the network cable switches the viewer to `REPLAY` with the label visible.
5. README has the "built today" table matching the repo.

**3-minute demo (no slides):**
- 0:00–0:20 Who and why: setting up a sim of your own desk takes days; here's one from a 10-second video.
- 0:20–0:50 Splat + Argon's `SceneSpec` with its evidence lines.
- 0:50–1:40 Attempt 1 fails; Argon's critique with the frames it used; the patch.
- 1:40–2:20 Passing attempt; MuJoCo pass check on screen.
- 2:20–3:00 What's live vs replay, what's built today, what's next (real hardware).

## 10. Build order

Built as vertical slices, each ending in a working check:

1. Load ALOHA in MuJoCo; key mounted and gitignored.
2. Record the desk video; start COLMAP and splat training on Colab.
3. `sim.py`: hand-written scene, scripted hand-over with `mink`, metrics, keyframes, rollout log (acceptance check 1).
4. `argon.py` + `loop.py`: video → `SceneSpec`, critique → `PlanPatch`, up to 4 attempts (acceptance check 2).
5. Viewer: arm and object replay, attempt filmstrip, mode labels (acceptance check 3).
6. Managed-agent robustness sweep of a passing plan.
7. Recorded backup run, README, replay export (acceptance checks 4–5).

Cut order if a slice slips: the sweep, then the splat (plain 3D desk instead), then the third object.

## 11. Non-goals and fallbacks

- No yield statistics, no multi-page app. The only training is the Gemma 4 pilot in `training/`, reported with measured MuJoCo pass counts on held-out cases.
- No real-robot control.
- The LLM never grades itself; MuJoCo decides pass/fail.
- Fallbacks: Colab splat fails → phone app capture (Scaniverse/Polycam) → plain Three.js desk. Argon video input fails → Argon reads 3–5 extracted frames instead. Managed agent slow → skip slice 5.
- Everything in this repo is written during the event. The README's "built during the hackathon" table must match what actually ships.
