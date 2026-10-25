"""Viewer API and static hosting.

Owner: integrator (Claude).

    uvicorn server:app --port 8000

Runs started through POST /api/runs in this process are LIVE. Any other run
directory under runs/ is served as REPLAY, with its recorded time.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import loop
from schemas import CONTRACT_VERSION, PlanPatch, RunManifest

app = FastAPI(title="ArgonTwin", version=CONTRACT_VERSION)

@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default handler echoes the rejected input, which can't be serialized when
    # it is NaN or infinity. Report where and why, without the value.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


_lock = threading.Lock()
_live: dict[str, dict] = {}  # run_id -> {"state", "log", "error", "phase", "phases"}


class StartRun(BaseModel):
    """POST /api/runs body. Exactly one of scene or video."""

    scene: Optional[str] = None  # repo-relative path, e.g. examples/desk_manual.json
    video: Optional[str] = None  # repo-relative path, e.g. capture/IMG_1272.MOV
    initial_patch: Optional[PlanPatch] = None  # applied to the starting plan before attempt 1
    max_attempts: int = Field(config.MAX_ATTEMPTS, ge=1, le=config.MAX_ATTEMPTS)
    sweep: bool = False


class RunSummary(BaseModel):
    run_id: str
    created_at: Optional[datetime]
    passed: Optional[bool]
    attempts: int
    source: Literal["LIVE", "REPLAY"]
    state: Literal["running", "done", "error"]
    scene_background: Optional[str] = None  # the run's scene.background_splat_path (None = default desk)


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
    if run_id in _live and not (config.RUNS_DIR / run_id).is_dir():
        return None  # started a moment ago; the worker hasn't created its folder yet
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
        scene_background=manifest.scene.background_splat_path if manifest else None,
    )


@app.get("/api/health")
def health() -> dict:
    # Liveness must not read the credential pipe or wait for 1Password approval.
    key = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "contract_version": CONTRACT_VERSION,
        "gemini_key_configured": key,
        "gemini_key_mount_present": (config.REPO_ROOT / ".env").exists(),
        "splat_available": config.SPLAT_PATH.exists(),
        "model": config.GEMINI_MODEL,
        "agent": config.GEMINI_AGENT,
    }


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    if not config.RUNS_DIR.exists():
        return []
    summaries = [_summary(p.name) for p in config.RUNS_DIR.iterdir() if p.is_dir()]
    # Newest first by the time the run recorded, not by directory name; live runs on top.
    earliest = datetime.min.replace(tzinfo=timezone.utc)
    summaries.sort(key=lambda r: (r.state == "running", r.created_at or earliest), reverse=True)
    return summaries


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    manifest = _read_manifest(run_id)
    summary = _summary(run_id)
    if run_id in _live and not (config.RUNS_DIR / run_id).is_dir():
        input_path = None  # in-memory state only until the worker creates the run folder
    else:
        input_path = _run_dir(run_id) / "input.json"
    live = _live.get(run_id, {})
    return {
        "summary": summary.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json") if manifest else None,
        "input": json.loads(input_path.read_text()) if input_path and input_path.exists() else None,
        "phase": live.get("phase"),        # latest {"phase", "attempt", ...} from loop.on_phase
        "phases": live.get("phases", []),  # every phase event so far, in order
        "log": live.get("log", []),
        "error": live.get("error"),
    }


@app.get("/api/scene")
def get_scene(path: str) -> dict:
    """Read-only: the scene and starting plan in a repo example file, for the plan editor."""
    scene, plan = loop.load_manual(_repo_path(path))
    return {"path": path, "scene": scene.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}


@app.post("/api/runs", response_model=RunSummary)
def start_run(request: StartRun) -> RunSummary:
    if (request.scene is None) == (request.video is None):
        raise HTTPException(400, "give exactly one of scene or video")
    patch = request.initial_patch
    if patch is not None and all(v is None for v in patch.model_dump().values()):
        patch = None
    scene_path = _repo_path(request.scene) if request.scene else None
    video_path = _repo_path(request.video) if request.video else None
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "a run is already in progress")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    state = {"state": "running", "log": [], "error": None, "phase": None, "phases": []}
    _live[run_id] = state

    def on_phase(phase: str, detail: dict) -> None:
        event = {"phase": phase, **detail}
        state["phase"] = event
        state["phases"].append(event)

    def work() -> None:
        try:
            loop.run_loop(
                video_path=video_path,
                scene_path=scene_path,
                initial_patch=patch,
                max_attempts=request.max_attempts,
                sweep=request.sweep,
                run_id=run_id,
                progress=state["log"].append,
                on_phase=on_phase,
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


# Optional Gemma base vs fine-tuned comparison API (compare_api.py, owned by the spec session).
# Guarded so the server still starts before that module exists.
try:
    import compare_api
except ImportError:
    compare_api = None
if compare_api is not None:
    app.include_router(compare_api.router)

config.RUNS_DIR.mkdir(exist_ok=True)
app.mount("/runs", StaticFiles(directory=config.RUNS_DIR), name="runs")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT)
