"""Viewer API and static hosting.

Owner: integrator (Claude).

    uvicorn server:app --port 8000

Runs started through POST /api/runs in this process are LIVE. Any other run
directory under runs/ is served as REPLAY, with its recorded time.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import loop
from schemas import CONTRACT_VERSION, RunManifest

app = FastAPI(title="ArgonTwin", version=CONTRACT_VERSION)

_lock = threading.Lock()
_live: dict[str, dict] = {}  # run_id -> {"state", "log", "error"}


class StartRun(BaseModel):
    scene: Optional[str] = None  # repo-relative path, e.g. examples/desk_manual.json
    video: Optional[str] = None  # repo-relative path, e.g. capture/IMG_1272.MOV
    max_attempts: int = config.MAX_ATTEMPTS
    sweep: bool = False


class RunSummary(BaseModel):
    run_id: str
    created_at: Optional[datetime]
    passed: Optional[bool]
    attempts: int
    source: Literal["LIVE", "REPLAY"]
    state: Literal["running", "done", "error"]


def _repo_path(relative: str) -> Path:
    path = (config.REPO_ROOT / relative).resolve()
    if config.REPO_ROOT not in path.parents or not path.is_file():
        raise HTTPException(400, f"not a file inside the repo: {relative}")
    return path


def _run_dir(run_id: str) -> Path:
    path = (config.RUNS_DIR / run_id).resolve()
    if path.parent != config.RUNS_DIR.resolve() or not path.is_dir():
        raise HTTPException(404, f"no run {run_id}")
    return path


def _read_manifest(run_id: str) -> Optional[RunManifest]:
    path = _run_dir(run_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text())
    except ValueError:
        return None  # other tools keep their own manifest.json under runs/; not a loop run


def _summary(run_id: str) -> RunSummary:
    manifest = _read_manifest(run_id)
    live = _live.get(run_id)
    return RunSummary(
        run_id=run_id,
        created_at=manifest.created_at if manifest else None,
        passed=manifest.passed if manifest and (not live or live["state"] == "done") else None,
        attempts=len(manifest.attempts) if manifest else 0,
        source="LIVE" if live else "REPLAY",
        state=live["state"] if live else "done",
    )


@app.get("/api/health")
def health() -> dict:
    try:
        config.gemini_api_key()
        key = True
    except config.MissingSecret:
        key = False
    return {
        "contract_version": CONTRACT_VERSION,
        "gemini_key_configured": key,
        "splat_available": config.SPLAT_PATH.exists(),
        "model": config.GEMINI_MODEL,
        "agent": config.GEMINI_AGENT,
    }


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    if not config.RUNS_DIR.exists():
        return []
    ids = sorted((p.name for p in config.RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    return [_summary(run_id) for run_id in ids]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    summary = _summary(run_id)
    return {
        "summary": summary.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json") if manifest else None,
        "log": _live.get(run_id, {}).get("log", []),
        "error": _live.get(run_id, {}).get("error"),
    }


@app.post("/api/runs", response_model=RunSummary)
def start_run(request: StartRun) -> RunSummary:
    if (request.scene is None) == (request.video is None):
        raise HTTPException(400, "give exactly one of scene or video")
    scene_path = _repo_path(request.scene) if request.scene else None
    video_path = _repo_path(request.video) if request.video else None
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "a run is already in progress")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    state = {"state": "running", "log": [], "error": None}
    _live[run_id] = state

    def work() -> None:
        try:
            loop.run_loop(
                video_path=video_path,
                scene_path=scene_path,
                max_attempts=request.max_attempts,
                sweep=request.sweep,
                run_id=run_id,
                progress=state["log"].append,
            )
            state["state"] = "done"
        except Exception as exc:  # surfaced to the viewer, not swallowed
            state["state"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["log"].append(traceback.format_exc(limit=3))
        finally:
            _lock.release()

    threading.Thread(target=work, name=f"run-{run_id}", daemon=True).start()
    return RunSummary(run_id=run_id, created_at=None, passed=None, attempts=0, source="LIVE", state="running")


config.RUNS_DIR.mkdir(exist_ok=True)
app.mount("/runs", StaticFiles(directory=config.RUNS_DIR), name="runs")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT)
