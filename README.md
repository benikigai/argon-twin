# ArgonTwin

**Your desk, rehearsed by robots before hardware.**

**Hosted page: [argon-twin.vercel.app](https://argon-twin.vercel.app)** is a labeled TARGET PREVIEW (concept), not the live app. Live runs happen on my Mac. Hosted replay of recorded runs comes later.

10:30 AM: hacking starts at SHACK15.\
11:24 AM: I walk a circle around my table with my phone. 10.4 seconds of video.\
12:10 PM: two ALOHA arms pass a can across a MuJoCo version of that table. 0.4 cm from target. (Handwritten plan, weld grasp assist. Details below.)

Built at the [Gemini 4 Argon SF Hackathon](https://cerebralvalley.ai/events/~/e/gemini-4-argon-sf), Google DeepMind × Cerebral Valley, Oct 25, 2026. Problem Statement Two, *More Dimensions*: render what it made, watch the result, critique it against the goal, fix it in a loop.

![TARGET EXPERIENCE: concept illustration, not a screenshot. A gold point cloud desk with two robot arms passing a can, a MuJoCo PASS badge, Argon's critique on the right, and an attempt filmstrip along the bottom.](docs/target-experience.svg)

*Target experience: a concept illustration of what we are building, not a screenshot. The critique text in it is an example, not model output.*

## Why

Every robot task starts the same way. Measure the table. Measure the objects. Write the scene file. Tune the grasp. Days of setup before the first test.

So most people skip the sim and test on the real robot. Then they break things.

ArgonTwin collapses that setup into a phone video. Gemini 4 Argon reads the video and writes the scene. MuJoCo runs two arms through the task. When it fails, Argon looks at the frames and the numbers, says why, and changes the plan. Then MuJoCo runs it again.

The model proposes. MuJoCo decides.

## The loop

![The loop: a 10 second phone video goes to Gemini 4 Argon, which builds the scene. Two ALOHA arms run the handover in MuJoCo. Argon watches keyframes and metrics, explains the failure, and patches the plan. Only MuJoCo's judge decides pass or fail. Passing runs are saved for the live viewer and, later, a recorded replay on Vercel.](docs/loop.svg)

- **Pass or fail is never the model's call.** `schemas.judge` checks measured MuJoCo numbers against fixed thresholds in `config.py`. The handover has to complete, the can can't drop, it has to land within 3 cm of the target and sit still, and no arm can press into the table. Argon can suggest extra checks. They show up on screen. They never change the verdict.
- **Live and replay stay separate.** MuJoCo and every Gemini call run on my Mac. The Vercel site will only play recorded runs, labeled with the time they were recorded. No keys there, no model calls.
- **Argon's long output is planned, not proven.** Argon can write past 65,536 output tokens in one call. The plan is a one pass world build that writes the scene, the controller settings and the tests together, with output tokens and thinking tokens reported separately, straight from the API. No run has crossed 65,536 output tokens yet. That comes after the core handover works with Argon in the loop.

## The stack

![The stack, by layer. Capture: iPhone video (verified), COLMAP on a Colab T4 with 52 views registered (in progress), gsplat splat not exported yet. Reasoning: gemini-4-argon direct call verified, antigravity-preview-09-2026 managed agent in progress, Gemma 4 E4B via Ollama with a measured 8 of 12 baseline and fine tuning in progress. Physics: MuJoCo 3.17 with a passing handwritten plan using weld assist, two ALOHA arms with mink 1.3, three passing pytest physics tests. Contracts: Pydantic schemas 1.0.0, deterministic judge, loop.py tested with stubs. Show: FastAPI and uvicorn on localhost:8000, Three.js viewer with real ALOHA meshes, gold and photoreal scan toggle waiting on the splat. Host: Vercel target preview live, recorded replay planned, 1Password Environment for the event key.](docs/stack.svg)

| Layer | What runs it | Why this one |
|---|---|---|
| Capture | iPhone 17 Pro Max, 0.5x ultra wide, 1080p60. COLMAP for camera poses, `gsplat` for the splat, on a Colab T4 | The ultra wide clip orbits the whole table in 10 seconds. A Mac has no CUDA, so training runs on Colab |
| Reasoning | `gemini-4-argon` through the Interactions API. `antigravity-preview-09-2026` managed agent with Argon. Gemma 4 E4B (base) through Ollama for the pilot | Argon watches video and frames, and writes long structured output. The managed agent runs work in its own sandbox |
| Physics | MuJoCo 3.17, `mink` 1.3 for inverse kinematics, the ALOHA bimanual model from MuJoCo Menagerie via `robot_descriptions` 3.3 | A deterministic referee. Same inputs, same result, every time |
| Contracts | Pydantic models frozen at 1.0.0 in `schemas.py`, thresholds in `config.py` | Three builders wrote three parts at once. The contract is how they fit |
| Show | FastAPI + uvicorn, Three.js 0.170, `gaussian-splats-3d` 0.4.7 for the photoreal scan, Archivo type | One screen. The scene, the verdict, Argon's critique, and the loop in order |
| Host | Vercel for the public page. 1Password Environment for the event key, mounted as a local `.env` | Public page holds no secrets. The key never touches git |

## What's real right now

Oct 25, 2026. Every claim here has a receipt.

| Part | Owner | Status |
|---|---|---|
| `schemas.py`, `config.py` | Claude | Done. Contracts frozen at 1.0.0 |
| `sim.py`, `tests/test_sim.py` | Codex | A handwritten (MANUAL) plan passes the handover with weld grasp assist. Can lands 3.853 mm from target in the recorded run, 20 rendered keyframes in `runs/verified-manual/attempt-1`. 3 tests pass: nominal handover with exact repeatability, an unreachable handover that has to fail, target reach validation |
| `web/` viewer | Claude | Shows the real passing run with the real ALOHA meshes (12 STL files, 7.9 MB). The gold and photoreal scan modes are tested only on a synthetic splat, because no desk splat exists yet |
| `loop.py`, `server.py`, `export_replay.py` | Claude | Tested end to end with stub sim and Argon modules. Not yet run with the real sim and real Argon together |
| `argon.py`, `managed_agent.py` | Antigravity | Files exist. Loop integration not verified. A direct `gemini-4-argon` Interactions call was verified earlier today |
| `capture/` splat | Codex | COLMAP registered 52 views. The Colab runtime ended before a trained splat was exported. No usable splat yet |
| `training/` Gemma 4 pilot | Codex | Baseline measured, fine tuning in progress (below) |
| Hosted page | | [argon-twin.vercel.app](https://argon-twin.vercel.app) is a TARGET PREVIEW. Recorded replay isn't hosted yet |

Everything in this repo was written during the event.

## Gemma 4 pilot

Can a small open model propose handover plans that pass MuJoCo? Does fine tuning help?

- 36 training cases. 12 test cases across 4 held out scene variants. All in `runs/gemma-pilot`.
- Gemma 4 E4B, base weights, run locally through Ollama with a JSON schema: **8 of 12** test plans passed MuJoCo. **12 of 12** were valid JSON.
- Fine tuned adapter: training on a Colab T4. No result yet.

## What's not working yet

The only passing handover uses a plan I wrote by hand, and a weld that locks the can to the gripper once it closes. Fingertip friction alone isn't holding the can yet. The viewer flags the weld on every attempt that uses it.

Argon isn't in the loop yet. No verified run where Argon critiques a failure and the next attempt passes.

The desk splat died with the Colab runtime after 52 views registered. The viewer shows a plain table until it's back. Splat to sim alignment is set by hand.

The world build is a long call, minutes not seconds. On stage I run the critique loop live and show the world build's recorded result.

## Run it

Python 3.12, `mujoco`, `mink`, `robot_descriptions`, `google-genai`, `pydantic`, `fastapi`, `uvicorn`. A `requirements.txt` lands with the next sim commit.

```bash
uvicorn server:app --port 8000                       # live viewer at http://127.0.0.1:8000
python loop.py --scene examples/desk_manual.json     # handwritten scene, Argon critiques
python loop.py --video capture/IMG_1272.MOV          # Argon builds the scene from the video
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
