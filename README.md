# ArgonTwin

**Your desk, rehearsed by robots before hardware.**

A 15-second phone video of a real workspace becomes a 3D Gaussian splat and a MuJoCo physics twin. Two ALOHA arms are placed on the desk and asked to hand a can from one arm to the other. Gemini 4 Argon watches each simulated attempt, explains why it failed, and patches the plan until MuJoCo verifies the hand-over.

Built at the [Gemini 4 Argon SF Hackathon](https://cerebralvalley.ai/events/~/e/gemini-4-argon-sf) (Google DeepMind × Cerebral Valley, SHACK15, Oct 25, 2026) for Problem Statement Two, *More Dimensions*: render what it made, watch the result, critique it against the goal, fix it in a loop.

## Status

Repo created at 11:40 AM PDT on Oct 25, 2026. It holds the plan only; code lands slice by slice per [`docs/BUILD-SPEC.md`](docs/BUILD-SPEC.md).

## Built during the hackathon

Updated as each part ships. Nothing in this repo predates the event.

| Part | What it does | Status |
|---|---|---|
| `schemas.py`, `config.py` | Data contracts, module interfaces, deterministic pass/fail judge, model IDs and thresholds | Done (contract 1.0.0) |
| `capture/` | Desk video → frames → camera poses → trained Gaussian splat | Planned |
| `sim.py` | MuJoCo scene with two ALOHA arms, scripted hand-over, pass/fail checks | Planned |
| `argon.py`, `loop.py` | Argon builds the twin from the video, critiques each attempt, patches the plan | Planned |
| `server.py`, `web/` | 3D viewer: the splat with the arms replaying each attempt, Argon's critique beside it | Planned (placeholder page only) |

## Open-source parts we use (not built today)

- [MuJoCo](https://github.com/google-deepmind/mujoco) and the ALOHA model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [mink](https://github.com/kevinzakka/mink) for inverse kinematics
- [gsplat](https://github.com/nerfstudio-project/gsplat) for splat training, [COLMAP](https://colmap.github.io/) for camera poses
- [Three.js](https://threejs.org/) and [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) for the viewer

## Models

- `gemini-4-argon` through the Gemini Interactions API
- `antigravity-preview-09-2026` managed agent (Antigravity) with `gemini-4-argon`

## Running it

Not runnable yet. Setup instructions arrive with the first code slice.

Secrets are read from a local `.env` that is never committed.
