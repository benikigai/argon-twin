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

### Viewer design (built fresh today; the Oct 21 cockpit is reference for lessons only)

- **One stage, not a control panel.** The 3D view fills the screen: the photoreal splat of the desk with the arms and objects replayed in it. No slider column, no KPI wall; the rules ban "a dashboard as the main feature".
- **Attempt filmstrip along the bottom:** `Attempt 1 ✗ dropped at hand-over` → `Argon critique` → `Attempt 2 ✓ on target`. Clicking a card replays that attempt on the stage.
- **Right rail, one card at a time:** Argon's critique for the selected attempt, with the 2–3 keyframes it cited and the plan change it made.
- **One verdict badge from MuJoCo** (PASS/FAIL plus the metric that decided it), and a small mode pill (`LIVE_ARGON`, `REPLAY`, `grasp assist: weld`).
- **Keep from the old cockpit:** small MuJoCo camera insets (overhead and wrist) next to the stage, and showing the live interaction/environment ID as proof the managed agent ran.
- **Drop from the old cockpit:** the three assembly mechanisms, robot picker, Gemma LoRA, Gemini Live Voice, embeddings, Monte Carlo yield numbers, and the gold wireframe "splat" style. Only a real trained splat may be called a splat.
- **Controls:** `Run loop` and `Replay`. Nothing else on screen during the demo.

### Using Argon's long decode (output past 65,536 tokens) honestly

The new capability is output, not input: with `background=True` and `max_output_tokens` above 65,536, Argon can write a large, multi-file artifact in one pass. Track 2's own text says this ("generate high-fidelity 3D games and procedural worlds in a single pass"). Use it where one consistent pass genuinely matters:

- **One-pass world build (`argon.build_world`):** Argon gets the full desk video, the ALOHA MJCF and our contracts in one request (long input), and returns in one background call the whole twin package: `scene.xml` (table, can, laptop with a hinged lid, phone), `task.json` (both arms' waypoints and gripper timing for the hand-over), `checks.py` (the pass/fail checks for this scene), and `viewer_scene.json` (object placements for Three.js). Our code then compiles the MJCF, runs `checks.py`, and the loop takes over.
- **Show real numbers only.** The viewer shows the `usage` returned by that call (output and thinking tokens, wall time). If a run comes in under 65,536 tokens, show that number as it is; don't pad the prompt to cross the line.
- **Not live on stage.** A long decode likely takes minutes (the builder guide says to allow 15–20 minute timeouts for `high` thinking). Run it during the day and once more before judging; on stage, run the fast critique → patch loop live and show the world build's receipt.
- **Don't do:** have Argon type out training trajectories or collision meshes. Trajectories come from MuJoCo rollouts; geometry comes from primitives and menagerie meshes. Token volume is not the point; a working twin is.
- **No Gemma "fine-tuning" claim** unless a real Gemma 4 checkpoint is loaded and trained. A small PyTorch network named after Gemma is not Gemma.

## 4. Happy path and recovery path

**Happy path**
1. Ben records a 15 s video of the workspace on his phone.
2. Splat: frames → COLMAP → gsplat training on a Colab GPU → `.ply`/`.splat` file. Runs in the background; not part of the 3-minute live demo.
3. `scene_from_video`: Argon watches the video and returns a validated `SceneSpec` (table size and height, each object's type, size and position, arm base positions, the hand-over point, the target spot).
4. `build_mjcf`: our code turns `SceneSpec` into MJCF that includes the ALOHA bimanual model from `mujoco_menagerie`.
5. `run_attempt`: scripted hand-over with differential IK (`mink`), MuJoCo simulates it, records body poses for the viewer, renders keyframes, and computes pass/fail metrics.
6. `critique_and_patch`: Argon gets the keyframes + metrics + current plan, explains the failure, and returns a validated `PlanPatch`. Loop up to 4 attempts or until MuJoCo passes.
7. Viewer replays every attempt over the splat, with Argon's critique per attempt.

**Recovery path:** if Argon returns invalid JSON, retry once with the validation error; if still invalid, keep the previous plan and mark the attempt `ARGON_INVALID`. If the network or API is down on stage, the viewer plays today's recorded run, labeled `REPLAY (recorded <time>)`.

## 5. Architecture and data contracts

```
phone video ──► capture/ (ffmpeg frames → COLMAP → gsplat on Colab GPU) ──► static/assets/desk.splat
     │
     └──► argon.scene_from_video ──► SceneSpec ──► sim.build_mjcf ──► MJCF (table + objects + ALOHA)
                                                           │
            PlanPatch ◄── argon.critique_and_patch ◄── sim.run_attempt (mink IK, keyframes, metrics, pose log)
                │                                          ▲
                └──────────────── loop (≤4) ───────────────┘
                                                           │
FastAPI server.py ── /api/run, /api/attempts/{n} ──► static/index.html (Three.js + gaussian-splats-3d)
```

Repo layout (`~/Code/argon-twin`):
`argon.py` (Gemini calls), `schemas.py`, `sim.py`, `loop.py`, `server.py`, `static/index.html`, `static/app.js`, `capture/` (splat notebook), `runs/` (attempt logs, gitignored except the demo run).

Contracts (Pydantic):

```python
class ObjectSpec(BaseModel):
    name: str                      # "can", "laptop", "phone"
    kind: Literal["cylinder", "box"]
    size_m: list[float]            # cylinder: [radius, height]; box: [x, y, z]
    pos_m: list[float]             # [x, y, z] in table frame, z = bottom on table
    yaw_deg: float = 0.0
    graspable: bool

class SceneSpec(BaseModel):
    table_size_m: list[float]      # [x, y]
    table_height_m: float
    objects: list[ObjectSpec]
    left_arm_base_m: list[float]
    right_arm_base_m: list[float]
    handover_point_m: list[float]
    target_m: list[float]
    evidence: list[str]            # what in the video each estimate came from

class PlanPatch(BaseModel):
    diagnosis: list[str]           # 2-4 steps, tied to frames/metrics
    grasp_height_offset_m: float
    approach_yaw_deg: float
    handover_point_m: list[float]
    gripper_close_delay_s: float
    lift_height_m: float
    release_height_m: float

class AttemptResult(BaseModel):
    attempt: int
    passed: bool
    can_final_error_m: float
    dropped: bool
    handover_completed: bool
    arm_table_collision: bool
    mode: Literal["LIVE_ARGON", "LIVE_ARGON_MANAGED_AGENT", "ARGON_INVALID", "REPLAY"]
```

Pass check (MuJoCo only): `handover_completed and not dropped and not arm_table_collision and can_final_error_m <= 0.03`.

## 6. Tool and access status

| Tool | ID / package | Status at 10:25 AM |
|---|---|---|
| Gemini 4 Argon | `gemini-4-argon`, Interactions API, `google-genai >= 2.3.0` | Credits claimed at kickoff. Video-input request shape on the Interactions API: UNKNOWN, check the builder guide / `gemini-api-dev` skill before writing `argon.py` |
| Managed agent | `antigravity-preview-09-2026` with `agent_config={"type": "antigravity", "model": "gemini-4-argon"}`, `environment="remote"` | Untested |
| Fast checks | `gemini-3.8-flash` | Untested; optional |
| Physics | `mujoco`, `mink`, ALOHA from `mujoco_menagerie` via `robot_descriptions` (module `aloha_mj_description`) | Not installed; this session has no network. `robot_descriptions` git-clones menagerie on first use, so load it in slice 0 before venue Wi-Fi gets busy |
| Splat | `ffmpeg`, COLMAP, `gsplat` on a Colab GPU (`colab` CLI is installed) | Untested. Mac has no CUDA, so training must run on Colab (or a CUDA box) |
| Viewer | Three.js + `@mkkellogg/gaussian-splats-3d` | Untested |
| Secrets | 1Password Environment set up by Codex, mounted into the repo root as a gitignored `.env` with `GEMINI_API_KEY` only (the event key, not Ben's personal key). The app reads the key from the environment and has no fallback. Exact vault and item reference: event folder `03_Sponsor-Toolsets/argon-api-verification-2026-10-25.md`, kept out of this public repo | Key verified with a live Argon call at ~11:15. Mount must point at `~/Code/argon-twin` (it was mounted into `~/Code/argon-hackathon`). The `op` CLI has no account configured, so don't rely on `op read` |
| Repo | Public GitHub, created at kickoff | `gh` token reported invalid; re-check with network |

## 7. Truth labels and approval points

- Every attempt shows its `mode` in the viewer.
- **Grasp assist:** if friction grasping is unreliable, the can is attached with a MuJoCo weld once the gripper closes on it. If used, the viewer says `grasp assist: weld` on screen.
- Splat-to-sim alignment is set by hand (scale, rotation, offset saved to `static/assets/align.json`). Say so in the pitch.
- ALOHA models, `mink` and the splat library are open-source; the README credits them and lists them under "not built today".
- Numbers in the pitch come only from today's runs.
- Ben approves: the public repo push, the submission, and any purchase of Colab compute.

## 8. Acceptance checks and demo script

1. `python sim.py --scene examples/desk_manual.json` runs one hand-over attempt and prints an `AttemptResult`.
2. `python loop.py --video capture/desk.mp4` produces a `SceneSpec` from Argon and at least one failed then one passed attempt, logged under `runs/`.
3. `uvicorn server:app` → `http://127.0.0.1:8088` shows the splat, both arms, and replays each attempt with Argon's critique beside it.
4. Pulling the network cable switches the viewer to `REPLAY` with the label visible.
5. README has the "built today" table matching the repo.

**3-minute demo (no slides):**
- 0:00–0:20 Who and why: "Setting up a sim of your own desk takes days. Here's mine from a 15-second video."
- 0:20–0:50 Splat + Argon's `SceneSpec` with its evidence lines.
- 0:50–1:40 Attempt 1 fails; Argon's critique with the frames it used; the patch.
- 1:40–2:20 Passing attempt; MuJoCo pass check on screen.
- 2:20–3:00 What's live vs replay, what's built today, what's next (real hardware).

## 9. Vertical slices (times PDT)

Re-planned at 11:25 after a ~50 min slip (Ben: keep everything, compress). Repo is `~/Code/argon-twin` (created 11:40; the earlier `~/Code/argon-hackathon` folder held pre-event prototype files and is not the submission). Key comes from a 1Password Environment mounted into the repo as a gitignored `.env`; model and agent IDs live in `config.py`. Done already: a live `gemini-4-argon` Interactions call passed at ~11:15 (receipt in `03_Sponsor-Toolsets/argon-api-verification-2026-10-25.md`).

**Builder: Antigravity (Gemini 4 Argon) is the only agent that edits the repo** (Ben, 11:27). Codex and Claude stay out of it, with two exceptions: the splat job, which writes only under `capture/`, and reviews at slice ends, which are read-only. Checkpoint at 1:15: if Antigravity is stuck on MuJoCo/IK, Ben decides whether Codex takes over.

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

- No PyTorch distillation, no Monte Carlo yield study, no peg-in-socket mechanisms, no multi-page app.
- No real-robot control.
- The LLM never grades itself; MuJoCo decides pass/fail.
- Fallbacks: Colab splat fails → phone app capture (Scaniverse/Polycam) → plain Three.js desk. Argon video input fails → Argon reads 3–5 extracted frames instead. Managed agent slow → skip slice 5.
- Do not copy anything from `07_Prototype-Code/`. The init script's README template lists components we are not building (distillation, spatial HUD); rewrite that table to match what is actually built.
