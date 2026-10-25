"""Render → watch → critique → fix loop.

Owner: integrator (Claude). Calls only the interfaces frozen in schemas.py:
sim.validate_scene / sim.run_attempt, argon.build_world / argon.critique and,
optionally, managed_agent.robustness_sweep. Pass/fail comes from schemas.judge.

    python loop.py --scene examples/desk_manual.json     # MANUAL start, Argon critiques
    python loop.py --video capture/IMG_1272.MOV          # Argon builds the world first
    python loop.py --scene examples/desk_manual.json --patch '{"handover_point_m": [1.2, 0.8, 0.6]}'

An optional initial PlanPatch is applied to the starting plan before attempt 1.
Every run writes its own input snapshot (runs/<id>/input.json); example files are
never written, and nothing changes the plan of a run once it has started except
Argon's critique between attempts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import config
from schemas import (
    AttemptResult,
    ArgonInvalidOutput,
    ExecutionMode,
    HandoverPlan,
    InteractionRef,
    PlanPatch,
    RunManifest,
    SceneSpec,
    SupplementalCheck,
    WorldBuild,
    apply_patch,
    evaluate_supplemental,
    judge,
)

Progress = Callable[[str], None]
# on_phase(phase, detail). Phases, in order per attempt:
#   "running"    {"attempt": n}
#   "judged"     {"attempt": n, "passed": bool, "failed_checks": [...]}
#   "diagnosing" {"attempt": n}
#   "patched"    {"attempt": n, "changes": {field: value}}  (plan for attempt n + 1)
#   "invalid"    {"attempt": n, "error": str}                (Argon output failed validation)
# and once at the end: "done" {"passed": bool, "attempts": n}
PhaseHook = Callable[[str, dict], None]


def _say(progress: Optional[Progress], message: str) -> None:
    if progress:
        progress(message)


def _phase(on_phase: Optional[PhaseHook], phase: str, **detail) -> None:
    if on_phase:
        on_phase(phase, detail)


def load_manual(scene_path: Path) -> tuple[SceneSpec, HandoverPlan]:
    data = json.loads(Path(scene_path).read_text())
    scene = SceneSpec.model_validate(data["scene"])
    plan = HandoverPlan.model_validate(data["plan"]) if "plan" in data else HandoverPlan(handover_point_m=scene.handover_point_m)
    return scene, plan


def new_run_dir(run_id: Optional[str] = None) -> tuple[str, Path]:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.model_dump_json(indent=2))
    tmp.replace(path)
    return path


def run_loop(
    *,
    video_path: Optional[Path] = None,
    scene_path: Optional[Path] = None,
    max_attempts: int = config.MAX_ATTEMPTS,
    initial_patch: Optional[PlanPatch] = None,
    sweep: bool = False,
    critique: bool = True,
    run_id: Optional[str] = None,
    progress: Optional[Progress] = None,
    on_phase: Optional[PhaseHook] = None,
) -> RunManifest:
    if (video_path is None) == (scene_path is None):
        raise ValueError("pass exactly one of video_path or scene_path")

    sim = importlib.import_module("sim")
    run_id, run_dir = new_run_dir(run_id)
    created_at = datetime.now(timezone.utc)

    world_build: Optional[WorldBuild] = None
    checks: list[SupplementalCheck] = []
    previous: Optional[InteractionRef] = None

    if scene_path is not None:
        scene, plan = load_manual(scene_path)
        mode: ExecutionMode = "MANUAL"
        _say(progress, f"Loaded manual scene from {scene_path}")
    else:
        argon = importlib.import_module("argon")
        _say(progress, "Argon is building the world from the video (long decode)…")
        world_build = argon.build_world(Path(video_path))
        scene, plan = world_build.body.scene, world_build.body.initial_plan
        checks = list(world_build.body.supplemental_checks)
        previous = world_build.interaction
        mode = world_build.mode
        (run_dir / "world_build.json").write_text(world_build.model_dump_json(indent=2))
        _say(progress, f"World built: {len(scene.objects)} objects, interaction {previous.interaction_id}")

    changed = {}
    if initial_patch is not None:
        changed = {k: v for k, v in initial_patch.model_dump().items() if v is not None}
        plan = apply_patch(plan, initial_patch)
        _say(progress, "Starting plan edited before the run: " + ", ".join(f"{k}={v}" for k, v in changed.items()))

    # Per-run snapshot of exactly what this run started from.
    (run_dir / "input.json").write_text(json.dumps({
        "source": {"scene_path": str(scene_path) if scene_path else None, "video_path": str(video_path) if video_path else None},
        "initial_patch": changed or None,
        "scene": scene.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
    }, indent=2))

    problems = sim.validate_scene(scene)
    if problems:
        _say(progress, "Scene problems: " + "; ".join(problems))

    attempts: list[AttemptResult] = []

    def manifest(passed: bool, sweep_result=None) -> RunManifest:
        return RunManifest(
            run_id=run_id,
            created_at=created_at,
            video_path=str(video_path) if video_path else None,
            scene=scene,
            world_build=world_build,
            attempts=attempts,
            sweep=sweep_result,
            passed=passed,
            splat_path=str(config.SPLAT_PATH.relative_to(config.REPO_ROOT)) if config.SPLAT_PATH.exists() else None,
        )

    passed = False
    for n in range(1, max_attempts + 1):
        attempt_dir = run_dir / f"attempt_{n}"
        attempt_dir.mkdir()
        _say(progress, f"Attempt {n}: simulating ({mode})…")
        _phase(on_phase, "running", attempt=n)
        sim_attempt = sim.run_attempt(scene, plan, attempt=n, out_dir=attempt_dir)
        verdict = judge(sim_attempt.metrics)
        result = AttemptResult(
            attempt=n,
            mode=mode,
            plan=plan,
            sim=sim_attempt,
            verdict=verdict,
            supplemental=evaluate_supplemental(checks, sim_attempt.metrics),
        )
        attempts.append(result)
        write_manifest(run_dir, manifest(False))
        _say(progress, f"Attempt {n}: {'PASS' if verdict.passed else 'FAIL: ' + '; '.join(verdict.failed_checks)}")
        _phase(on_phase, "judged", attempt=n, passed=verdict.passed, failed_checks=verdict.failed_checks)

        if verdict.passed:
            passed = True
            break
        if not critique or n == max_attempts or mode == "ARGON_INVALID":
            break

        argon = importlib.import_module("argon")
        _say(progress, f"Argon is critiquing attempt {n}…")
        _phase(on_phase, "diagnosing", attempt=n)
        try:
            crit = argon.critique(scene, plan, result, attempt_dir=attempt_dir, previous=previous)
        except ArgonInvalidOutput as exc:
            # Spec: keep the previous plan and mark the next attempt ARGON_INVALID, then stop.
            _say(progress, f"Argon output invalid after retry ({exc}); rerunning with the same plan, marked ARGON_INVALID.")
            _phase(on_phase, "invalid", attempt=n, error=str(exc))
            mode = "ARGON_INVALID"
            continue
        attempts[-1] = result.model_copy(update={"critique": crit})
        previous = crit.interaction
        plan = apply_patch(plan, crit.body.patch)
        mode = crit.mode
        write_manifest(run_dir, manifest(False))
        _say(progress, "Argon: " + " → ".join(crit.body.diagnosis))
        _phase(on_phase, "patched", attempt=n, changes={k: v for k, v in crit.body.patch.model_dump().items() if v is not None})

    sweep_result = None
    if passed and sweep:
        managed_agent = importlib.import_module("managed_agent")
        _say(progress, "Managed agent: robustness sweep of the passing plan…")
        sweep_result = managed_agent.robustness_sweep(scene, plan, previous=previous)
        _say(progress, f"Sweep: {sweep_result.passed_trials}/{sweep_result.trials} trials passed")

    final = manifest(passed, sweep_result)
    write_manifest(run_dir, final)
    (config.RUNS_DIR / "latest.txt").write_text(run_id)
    _say(progress, f"Run {run_id}: {'PASSED' if passed else 'NOT PASSED'} after {len(attempts)} attempt(s)")
    _phase(on_phase, "done", passed=passed, attempts=len(attempts))
    return final


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--scene", type=Path)
    parser.add_argument("--max-attempts", type=int, default=config.MAX_ATTEMPTS)
    parser.add_argument("--sweep", action="store_true", help="run the managed-agent robustness sweep after a pass")
    parser.add_argument("--no-critique", action="store_true", help="simulate attempt 1 only, no Argon calls")
    parser.add_argument("--patch", type=str, default=None, help="initial PlanPatch as JSON, applied before attempt 1")
    args = parser.parse_args(argv)
    patch = PlanPatch.model_validate_json(args.patch) if args.patch else None
    manifest = run_loop(
        video_path=args.video,
        scene_path=args.scene,
        initial_patch=patch,
        max_attempts=args.max_attempts,
        sweep=args.sweep,
        critique=not args.no_critique,
        progress=lambda m: print(m, flush=True),
    )
    return 0 if manifest.passed else 1


if __name__ == "__main__":
    sys.exit(main())
