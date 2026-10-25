"""HTTP API for the base vs fine-tuned local Gemma comparison.

Owner: Claude (Gemma comparison). server.py includes this router at startup.

    GET  /api/compare/status      local model server health (read-only)
    GET  /api/compare/recorded    recorded local validation on 12 cases, from receipts
    GET  /api/compare/cases       held-out validation cases with recorded outcomes
    GET  /api/compare             comparison runs, newest first
    POST /api/compare             start one: {"case_id": "...", "seed": 0}
    GET  /api/compare/{id}        one comparison record (compare.json), live while running

Comparison files are served by the existing /runs mount under runs/compare-<id>/.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import compare
import config
import gemma_local

router = APIRouter(prefix="/api/compare", tags=["gemma-compare"])

_lock = threading.Lock()
_live: dict[str, dict] = {}  # compare_id -> latest record while running


class StartCompare(BaseModel):
    case_id: str = Field(..., pattern=r"^scene-\d+-variant-\d+$")
    seed: int = Field(0, ge=0, le=2**31 - 1)


@router.get("/status")
def status() -> dict:
    info = gemma_local.server_info()
    info["available"] = bool(info.get("health")) and info["health"].get("status") == "ok"
    info["model"] = gemma_local.MODEL_ID
    return info


@router.get("/recorded")
def recorded() -> dict:
    return compare.recorded_summary()


@router.get("/cases")
def cases() -> list[dict]:
    return compare.validation_cases()


@router.get("")
def list_compares() -> list[dict]:
    return compare.list_compares()


@router.get("/{compare_id}")
def get_compare(compare_id: str) -> dict:
    if compare_id in _live:
        return _live[compare_id]
    path = (config.RUNS_DIR / compare_id / "compare.json").resolve()
    if path.parent.parent != config.RUNS_DIR.resolve() or not compare_id.startswith(compare.COMPARE_PREFIX) or not path.exists():
        raise HTTPException(404, f"no comparison {compare_id}")
    return json.loads(path.read_text())


@router.post("")
def start(request: StartCompare) -> dict:
    if request.case_id not in {c["case_id"] for c in compare.validation_cases()}:
        raise HTTPException(400, f"unknown validation case {request.case_id}")
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "a comparison is already running")
    compare_id = compare.COMPARE_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S")
    _live[compare_id] = {"compare_id": compare_id, "state": "running", "phase": "starting"}

    def on_update(record: dict) -> None:
        _live[compare_id] = json.loads(json.dumps(record))  # snapshot, not a shared reference

    def work() -> None:
        try:
            compare.run_compare(request.case_id, seed=request.seed, compare_id=compare_id, on_update=on_update)
        except Exception as exc:  # e.g. the folder couldn't be created; shown, not swallowed
            _live[compare_id] = {"compare_id": compare_id, "state": "error", "phase": "error",
                                 "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}
            return
        finally:
            _lock.release()
        _live.pop(compare_id, None)  # finished: served from compare.json from now on

    threading.Thread(target=work, name=f"compare-{compare_id}", daemon=True).start()
    return {"compare_id": compare_id, "state": "running"}
