# ArgonTwin

**Your desk, rehearsed by robots before hardware.**

**Hosted page: [argon-twin.vercel.app](https://argon-twin.vercel.app)** is a labeled TARGET PREVIEW (concept), not the live app. Live runs happen on my Mac. Hosted replay of recorded runs comes later.

10:30 AM: hacking starts at SHACK15.\
11:24 AM: I walk a circle around my table with my phone. 10.4 seconds of video.\
12:43 PM: I give two ALOHA arms a handover point they can't reach. The can falls. Gemini 4 Argon reads the frames and moves the point. MuJoCo passes the next attempt. 1.49 cm from target.\
1:05 PM: same thing, live in the viewer. I press Break the plan. Argon fixes it in one call. 0.53 cm.\
1:10 PM: my real desk, trained into a Gaussian splat from that 10 second video, sits under the arms.

Built at the [Gemini 4 Argon SF Hackathon](https://cerebralvalley.ai/events/~/e/gemini-4-argon-sf), Google DeepMind × Cerebral Valley, Oct 25, 2026. Problem Statement Two, *More Dimensions*: render what it made, watch the result, critique it against the goal, fix it in a loop.

![TARGET EXPERIENCE: concept illustration, not a screenshot. A gold point cloud desk with two robot arms passing a can, a MuJoCo PASS badge, Argon's critique on the right, and an attempt filmstrip along the bottom.](docs/target-experience.svg)

*Target experience: a concept illustration of what we are building, not a screenshot. The critique text in it is an example, not model output.*

## Why

You're bringing up a two arm station on a table you've never used. Before anything runs on the real arms, you want to try the task in sim.

Every new table means a new sim. Measure the table. Measure the objects. Write the scene file. Tune the grasp and the handover. Days, every time.

So the step gets skipped. Untested motions run on real hardware. An unreachable handover or a dropped part costs arm time, or the arm.

ArgonTwin starts from a phone video of the table. MuJoCo runs two ALOHA arms through the task. When an attempt fails, Gemini 4 Argon looks at the rendered frames and the measured numbers, says why, and patches the plan. MuJoCo runs it again.

Google's own showcase for Argon's 3D work is [Desk Racer](https://aistudio.google.com/learn/building-3d-games-and-worlds-with-gemini-4-argon): a go kart game built on top of a real desk, refined by a critic model that rates screenshots. Same kind of input here. Different finish line. ArgonTwin is done when fixed thresholds on MuJoCo's measurements say so, not when a model likes the picture.

The model proposes. MuJoCo decides.

## The loop

![The loop: a 10 second phone video goes to Gemini 4 Argon, which builds the scene. Two ALOHA arms run the handover in MuJoCo. Argon watches keyframes and metrics, explains the failure, and patches the plan. Only MuJoCo's judge decides pass or fail. Passing runs are saved for the live viewer and, later, a recorded replay on Vercel.](docs/loop.svg)

- **Pass or fail is never the model's call.** `schemas.judge` checks measured MuJoCo numbers against fixed thresholds in `config.py`. The handover has to complete, the can can't drop, it has to land within 3 cm of the target and sit still, and no arm can press into the table. Argon can suggest extra checks. They show up on screen. They never change the verdict.
- **People stay in control.** Nothing here moves real hardware. A person can edit or break the plan before a run, and every attempt says whether its plan came from a person (`MANUAL`) or from Argon (`LIVE_ARGON`). The key stays on my Mac.
- **Live and replay stay separate.** MuJoCo and every Gemini call run on my Mac. The Vercel site will only play recorded runs, labeled with the time they were recorded. No keys there, no model calls.
- **Argon's long output is planned, not proven.** Argon can write past 65,536 output tokens in one call. The plan is a one pass world build that writes the scene, the controller settings and the tests together, with output tokens and thinking tokens reported separately. No run has crossed 65,536 output tokens yet.

## Proof

Two real repairs, both with a scene whose own handover point was also out of reach, so no passing answer was anywhere in what Argon saw.

| Run | How it started | Argon's patch | Result |
|---|---|---|---|
| `verified-no-hint-20261025-124355` | Handwritten plan, handover point (1.05, −0.75, 0.55). Can fell, 58.2 cm off target | Handover point → (0.00, −0.08, 0.26). 26,007 tokens in, 468 out, 11.9 s. Response `generateContent:_VveasKdJoaz1MkPwr_K0Qw` | Pass, 1.49 cm from target. Whole loop 14.5 s. Both plans rerun independently, same fail and pass |
| `20261025-130553` (live, through the viewer) | Base plan, then Break the plan: handover point (1.2, 0.8, 0.6). Can missed the second arm, 19.3 cm off target | Handover point → (0.00, −0.05, 0.22), lift height → 0.20 m. 26,030 tokens in, 515 out, 34.1 s. Response `generateContent:N2HeasDfEImsjrEPj6LciAk` | Pass, 0.53 cm from target |

Each simulated attempt renders 20 keyframes from four cameras and takes about 0.7 to 1.2 s on my Mac. Argon's thinking token count isn't reported by the API, so it isn't shown.

## The stack

![The stack, by layer, with status as of Oct 25. Capture: iPhone video, COLMAP with 52 views, a Gaussian splat trained on a Colab T4 and cropped to the table, all verified. Reasoning: gemini-4-argon live critique verified, antigravity-preview-09-2026 managed agent in progress, Gemma 4 E4B via Ollama with a measured 8 of 12 baseline and fine tuning in progress. Physics: MuJoCo 3.17 passing with Argon's patch and weld assist, two ALOHA arms with mink 1.3, pytest tests passing. Contracts: Pydantic schemas 1.0.1, deterministic judge, live repair loop verified. Show: FastAPI and uvicorn on localhost:8000, Three.js viewer with real ALOHA meshes, Gold, Solid and Photoreal views. Host: Vercel target preview live, recorded replay planned, 1Password Environment for the event key.](docs/stack.svg)

| Layer | What runs it | Why this one |
|---|---|---|
| Capture | iPhone 17 Pro Max, 0.5x ultra wide, 1080p60. COLMAP for camera poses, then a Gaussian splat trained on a Colab T4 (1,500 steps, 52 views) and cropped to the table | The ultra wide clip orbits the whole table in 10 seconds. The crop keeps everyone else at the event out of the public file |
| Reasoning | `gemini-4-argon` (multimodal `generateContent` for the critique). `antigravity-preview-09-2026` managed agent with Argon. Gemma 4 E4B (base) through Ollama for the pilot | Argon reads 20 frames and the metrics in one call and returns a typed patch |
| Physics | MuJoCo 3.17, `mink` 1.3 for inverse kinematics, the ALOHA bimanual model from MuJoCo Menagerie via `robot_descriptions` 3.3 | A deterministic referee. Same inputs, same result, every time |
| Contracts | Pydantic models frozen at 1.0.1 in `schemas.py` (every number must be finite), thresholds in `config.py` | Three builders wrote three parts at once. The contract is how they fit |
| Show | FastAPI + uvicorn, Three.js 0.170, `gaussian-splats-3d` 0.4.7, Archivo type | One screen. The scene, the verdict, Argon's critique, and the loop in order |
| Host | Vercel for the public page. 1Password Environment for the event key, mounted as a local `.env` | Public page holds no secrets. The key never touches git |

## Try it

Start the server, open `http://127.0.0.1:8000`, and the newest repair run plays.

- **Edit plan:** change the handover point or lift height. **Break the plan** moves the handover point out of reach. **Reset** goes back to the base plan. **Apply and run** starts a live run with your edits (only the changes are sent, as `initial_patch`). Every run saves its own starting point in `runs/<id>/input.json`; example files are never changed.
- **The loop rail** steps through attempt, critique, attempt. The progress strip shows each live phase: running, judged, Argon diagnosing, patch, done. The verdict appears after each replay finishes.
- **Gripper view** follows the real gripper body closest to the can.
- **Gold / Solid / Photoreal:** Gold draws the simulation's own geometry as dots (stylized, not captured). Blue dots are the real desk from the video. Photoreal shows the trained desk splat with the simulated arms on top.
- Keys: ← → step, space replays, C gripper view, H front view, T cycles the views.

## What's real right now

Oct 25, 2026. Every claim here has a receipt.

| Part | Owner | Status |
|---|---|---|
| `schemas.py`, `config.py` | Claude | Done. Contracts frozen at 1.0.1 |
| `loop.py`, `server.py`, `web/` | Claude | Live repair verified in the browser: run `20261025-130553` above. Plan editor, live progress, gripper view, three views |
| `sim.py`, `tests/` | Codex | Real ALOHA handover in MuJoCo with weld grasp assist. 20 keyframes per attempt from four cameras. 12 regression tests pass on Codex's machine |
| `argon.py` | Antigravity | Live critique verified in both proof runs. Prompts carry no scene positions or passing values. The world build from video isn't verified yet |
| `managed_agent.py` | Antigravity | Not verified in this README |
| Desk splat | Codex | Trained on a Colab T4, 17,157 Gaussians after the crop, aligned to the sim table by fitting the keyboard plane and table rim. Soft, with some floaters |
| `training/` Gemma 4 pilot | Codex | Baseline measured, fine tuning not finished (below) |
| Hosted page | | [argon-twin.vercel.app](https://argon-twin.vercel.app) is a TARGET PREVIEW. Recorded replay isn't hosted yet |

Everything in this repo was written during the event.

## Gemma 4 pilot

Can a small open model propose handover plans that pass MuJoCo? Does fine tuning help?

- 36 training cases. 12 test cases across 4 held out scene variants.
- Gemma 4 E4B, base weights, run locally through Ollama with a JSON schema: **8 of 12** test plans passed MuJoCo. **12 of 12** were valid JSON.
- Fine tuned adapter: no result yet.

## What's not working yet

Both repairs start from scenes I wrote by hand. Argon building the scene straight from the video isn't verified yet.

Grasping uses a weld. Once a gripper closes on the can, a MuJoCo weld holds it instead of fingertip friction. The viewer flags it on every attempt.

The desk splat is soft and has floaters on the tabletop. Its fit to the sim table assumes an 0.8 m table.

`sim.py`, `tests/` and `argon.py` work on my machine but aren't pushed yet, so a fresh clone can't run the loop today.

The world build is a long call, minutes not seconds. On stage I run the critique loop live.

## Run it

Python 3.12, `mujoco`, `mink`, `robot_descriptions`, `google-genai`, `pydantic`, `fastapi`, `uvicorn`. A `requirements.txt` lands with the sim commit.

```bash
uvicorn server:app --port 8000                       # live viewer at http://127.0.0.1:8000
python loop.py --scene examples/desk_live_base.json --patch '{"handover_point_m":[1.2,0.8,0.6]}'
python export_replay.py <run_id>                     # copy a finished run into web/replay/
```

`http://127.0.0.1:8000/?run=<run_id>` opens a specific run. `GEMINI_API_KEY` comes from the environment or a local `.env`, which is never committed.

## Built on

- [MuJoCo](https://github.com/google-deepmind/mujoco) and the ALOHA model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [mink](https://github.com/kevinzakka/mink) for inverse kinematics
- [COLMAP](https://colmap.github.io/) and [gsplat](https://github.com/nerfstudio-project/gsplat) for the desk scan
- [Three.js](https://threejs.org/) and [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) for the viewer
- Gemini 4 Argon, the Antigravity managed agent, and Gemma 4 from Google DeepMind

Full plan and contracts: [`docs/BUILD-SPEC.md`](docs/BUILD-SPEC.md).

Day one. Lets Build!
