"""Copy a finished run into web/replay/ so the hosted viewer can play it.

Owner: integrator (Claude).

    python export_replay.py 20261025-141500

The hosted viewer labels everything under web/replay/ as REPLAY with the
recorded time. Only manifest.json, keyframe PNGs and rollout JSON are copied.
"""

from __future__ import annotations

import json
import shutil
import sys

import config
from schemas import RunManifest

REPLAY_DIR = config.WEB_DIR / "replay"


def export(run_id: str) -> None:
    src = config.RUNS_DIR / run_id
    manifest = RunManifest.model_validate_json((src / "manifest.json").read_text())
    dst = REPLAY_DIR / run_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src / "manifest.json", dst / "manifest.json")
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
    })
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Exported {run_id} → {dst.relative_to(config.REPO_ROOT)} ({len(manifest.attempts)} attempts)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python export_replay.py <run_id>")
    export(sys.argv[1])
