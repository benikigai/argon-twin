"""Copy a finished run into web/replay/ so the hosted viewer can play it.

Owner: integrator (Claude).

    python export_replay.py 20261025-141500

    python export_replay.py 20261025-134505 --from /path/to/other/runs

The viewer labels everything under web/replay/ as REPLAY with the recorded time.
Only manifest.json, input.json, the run's verification receipt (if any), keyframe
PNGs and rollout JSON are copied. Absolute local paths in those JSON files are
rewritten to repo-relative paths, or removed when they point outside the repo.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import config
from schemas import RunManifest

REPLAY_DIR = config.WEB_DIR / "replay"
RECEIPTS = ("verification.json", "verification-receipt.json")
LOCAL_PREFIXES = ("/Users/", "/home/", "/private/", "/tmp/", "/var/")


def _sanitize(value: Any, repo_root: str) -> Any:
    """Repo paths become repo-relative; other local absolute paths are dropped."""
    if isinstance(value, dict):
        return {k: _sanitize(v, repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v, repo_root) for v in value]
    if isinstance(value, str):
        if value.startswith(repo_root):
            return value[len(repo_root):]
        if value.startswith(LOCAL_PREFIXES):
            return "<local path removed>"
    return value


def _copy_json(src: Path, dst: Path, repo_root: str) -> None:
    dst.write_text(json.dumps(_sanitize(json.loads(src.read_text()), repo_root), indent=2))


def export(run_id: str, runs_dir: Path = config.RUNS_DIR) -> None:
    src = runs_dir / run_id
    manifest = RunManifest.model_validate_json((src / "manifest.json").read_text())
    dst = REPLAY_DIR / run_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    # Paths recorded by the checkout that produced the run, which may not be this one.
    repo_root = str(runs_dir.resolve().parent) + "/"
    for name in ("manifest.json", "input.json", *RECEIPTS):
        if (src / name).exists():
            _copy_json(src / name, dst / name, repo_root)
    for attempt in manifest.attempts:
        a_src = src / f"attempt_{attempt.attempt}"
        a_dst = dst / f"attempt_{attempt.attempt}"
        a_dst.mkdir()
        for name in [k.path for k in attempt.sim.keyframes] + [attempt.sim.rollout_path]:
            target = a_dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a_src / name, target)

    index_path = REPLAY_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"runs": []}
    index["runs"] = [r for r in index["runs"] if r["run_id"] != run_id]
    index["runs"].insert(0, {
        "run_id": run_id,
        "created_at": manifest.created_at.isoformat(),
        "passed": manifest.passed,
        "attempts": len(manifest.attempts),
        "scene_background": manifest.scene.background_splat_path,
    })
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Exported {run_id} → {dst.relative_to(config.REPO_ROOT)} ({len(manifest.attempts)} attempts)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        export(args[0])
    elif len(args) == 3 and args[1] == "--from":
        export(args[0], Path(args[2]).resolve())
    else:
        sys.exit("usage: python export_replay.py <run_id> [--from <runs dir>]")
