"""Base vs fine-tuned local Gemma repair, judged by the same MuJoCo referee.

Owner: Claude (Gemma comparison). Uses the frozen interfaces only: sim.run_attempt,
schemas.apply_patch and schemas.judge. Thresholds and judge logic are untouched.

    python compare.py --case scene-13-variant-2

One comparison:
  1. Load a held-out validation case (failed scene + plan) from the serving folder's eval.jsonl.
  2. Re-run that failed plan in the current simulator and judge it (attempt 1).
  3. Send both variants the same payload, built from that fresh measurement.
  4. Validate each patch, apply it to its own fresh copy of the original plan, and run it
     on the same scene with the same seed (attempt 2 in each branch). Judge each.
  5. Write everything to runs/compare-<id>/compare.json as it happens.

Layout, so the viewer can replay each branch with window.argonTwin.playAttempt:
  runs/compare-<id>/original/attempt_1/   failed starting attempt
  runs/compare-<id>/base/attempt_2/       base Gemma's repair
  runs/compare-<id>/tuned/attempt_2/      fine-tuned Gemma's repair
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import config
import gemma_local
import sim
from schemas import CONTRACT_VERSION, HandoverPlan, PlanPatch, SceneSpec, apply_patch, judge

COMPARE_PREFIX = "compare-"
COMPARE_VERSION = "1"
VARIANTS = ("base", "tuned")
EVAL_PATH = gemma_local.SERVING_DIR / "eval.jsonl"
SUMMARY_PATH = gemma_local.SERVING_DIR / "endpoint-evaluation/summary.json"
PREDICTIONS_PATH = gemma_local.SERVING_DIR / "endpoint-evaluation/predictions.json"
MANIFEST_PATH = gemma_local.SERVING_DIR / "serving-manifest.json"
CONVERSION_PATH = gemma_local.SERVING_DIR / "conversion-receipt.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _git_head() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --- Recorded validation (read from receipts, never recomputed) ------------------

def validation_cases() -> list[dict]:
    """The 12 held-out cases, with what the recorded local evaluation measured for each."""
    rows = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    recorded: dict[str, dict] = {}
    for p in _read_json(PREDICTIONS_PATH) or []:
        recorded.setdefault(p["case_id"], {})[p["model"]] = p
    cases = []
    for row in rows:
        rec = recorded.get(row["case_id"], {})
        cases.append({
            "case_id": row["case_id"],
            "scene_group": row["scene_group"],
            "recorded_base_passed": rec.get("base_Q4", {}).get("verdict", {}).get("passed"),
            "recorded_tuned_passed": rec.get("tuned_Q4", {}).get("verdict", {}).get("passed"),
            "recorded_base_patch": rec.get("base_Q4", {}).get("prediction"),
            "recorded_tuned_patch": rec.get("tuned_Q4", {}).get("prediction"),
        })
    return cases


def recorded_summary() -> dict:
    """The saved local endpoint evaluation, labeled as recorded. Not mixed with Colab results."""
    summary = _read_json(SUMMARY_PATH) or {}
    return {
        "label": "Recorded local validation, 12 held-out cases",
        "observed_at_utc": summary.get("observed_at_utc"),
        "runtime": (_read_json(MANIFEST_PATH) or {}).get("runtime"),
        "base_quantization": (_read_json(CONVERSION_PATH) or {}).get("serving_base_quantization"),
        "schema_constrained": summary.get("schema_constrained"),
        "dataset_sha256": summary.get("dataset_sha256"),
        "base": summary.get("summary", {}).get("base_Q4"),
        "tuned": summary.get("summary", {}).get("tuned_Q4"),
        "source": str(SUMMARY_PATH),
        "note": "Separate from any live comparison below. Colab NF4 results use a different "
                "runtime and base quantization and are not shown here.",
    }


def load_case(case_id: str) -> tuple[dict, SceneSpec, HandoverPlan]:
    for line in EVAL_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["case_id"] == case_id:
            user = next(m for m in row["messages"] if m["role"] == "user")
            payload = json.loads(user["content"])
            return row, SceneSpec.model_validate(payload["scene"]), HandoverPlan.model_validate(payload["plan"])
    raise KeyError(f"no validation case {case_id}")


def build_payload(scene: dict, plan: dict, metrics: dict, failed_checks: list[str]) -> dict:
    """scene and plan are the validation case's own JSON, sent as recorded: re-serializing them
    through a newer contract would add fields the model never saw in training."""
    return {
        "scene": scene,
        "plan": plan,
        "metrics": {k: metrics[k] for k in gemma_local.PAYLOAD_METRIC_FIELDS},
        "failed_checks": list(failed_checks),
    }


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Where two JSON values differ, for the 'same input as the recorded case' check."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, float) and isinstance(b, float):
        return [] if abs(a - b) <= 1e-9 else [path]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def _attempt_record(sim_attempt, verdict, plan: HandoverPlan, mode: str, sim_s: float) -> dict:
    """Shaped like schemas.AttemptResult so the viewer can play it, plus sim timing."""
    return {
        "attempt": sim_attempt.attempt,
        "mode": mode,
        "plan": plan.model_dump(mode="json"),
        "sim": sim_attempt.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json"),
        "supplemental": [],
        "critique": None,
        "sim_s": sim_s,
    }


def _simulate(scene: SceneSpec, plan: HandoverPlan, attempt: int, out_dir: Path, seed: int):
    t0 = time.perf_counter()
    result = sim.run_attempt(scene, plan, attempt=attempt, out_dir=out_dir, render=True, seed=seed)
    return result, judge(result.metrics), time.perf_counter() - t0


def run_compare(case_id: str, *, seed: int = 0, compare_id: Optional[str] = None,
                on_update: Optional[Callable[[dict], None]] = None) -> dict:
    compare_id = compare_id or COMPARE_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / compare_id
    run_dir.mkdir(parents=True, exist_ok=False)
    row, scene, original_plan = load_case(case_id)
    recorded_payload = json.loads(next(m for m in row["messages"] if m["role"] == "user")["content"])
    cases = {c["case_id"]: c for c in validation_cases()}

    record: dict[str, Any] = {
        "compare_version": COMPARE_VERSION,
        "compare_id": compare_id,
        "kind": "LIVE_LOCAL_GEMMA_COMPARISON",
        "created_at": _now(),
        "finished_at": None,
        "state": "running",
        "phase": "original_running",
        "error": None,
        "case": {
            "case_id": case_id,
            "label": "Validation example (held-out case from the recorded local evaluation)",
            "scene_group": row["scene_group"],
            "recorded": cases.get(case_id),
        },
        "seed": seed,
        "scene": scene.model_dump(mode="json"),
        "original_plan": original_plan.model_dump(mode="json"),
        "original": None,
        "payload": None,
        "payload_matches_recorded_case": None,
        "payload_differences": None,
        "variants": {v: {"status": "pending"} for v in VARIANTS},
        "same_request_except_adapter": None,
        "provenance": {
            "contract_version": CONTRACT_VERSION,
            "repo_head": _git_head(),
            "sim_py_sha256": _sha256_file(config.REPO_ROOT / "sim.py"),
            "schemas_py_sha256": _sha256_file(config.REPO_ROOT / "schemas.py"),
            "config_py_sha256": _sha256_file(config.REPO_ROOT / "config.py"),
            "eval_jsonl_sha256": _sha256_file(EVAL_PATH),
            "serving_manifest": _read_json(MANIFEST_PATH),
            "adapter": _read_json(CONVERSION_PATH),
            "server": gemma_local.server_info(),
            "request_settings": {
                "temperature": gemma_local.TEMPERATURE,
                "max_tokens": gemma_local.MAX_TOKENS,
                "cache_prompt": False,
                "response_format": "json_schema from plan-patch.schema.json",
                "system_prompt": "SYSTEM from repair_client.py",
                "timeout_s": gemma_local.TIMEOUT_S,
                "adapter_scale": dict(gemma_local.ADAPTER_SCALE),
            },
            "cloud_calls": 0,
        },
        "recorded_validation": recorded_summary(),
    }

    def save(phase: Optional[str] = None) -> None:
        if phase:
            record["phase"] = phase
        (run_dir / "compare.json").write_text(json.dumps(record, indent=2))
        if on_update:
            on_update(record)

    try:
        save()
        # 1. The same failed attempt both models will see, measured now.
        orig_sim, orig_verdict, orig_s = _simulate(scene, original_plan, 1, run_dir / "original/attempt_1", seed)
        record["original"] = _attempt_record(orig_sim, orig_verdict, original_plan, "MANUAL", orig_s)
        metrics = orig_sim.metrics.model_dump(mode="json")
        payload = build_payload(recorded_payload["scene"], recorded_payload["plan"], metrics, orig_verdict.failed_checks)
        record["payload"] = payload
        diffs = _diff(payload, recorded_payload)
        record["payload_matches_recorded_case"] = not diffs
        record["payload_differences"] = diffs
        save("original_judged")
        if orig_verdict.passed:
            raise RuntimeError("the starting plan passed in the current simulator, so there is nothing to repair")

        # 2. Same payload to each variant; each patch applied to its own copy of the original plan.
        fingerprints = []
        for variant in VARIANTS:
            entry = record["variants"][variant]
            entry["status"] = "requesting"
            save(f"{variant}_requesting")
            result = gemma_local.propose(payload, variant)
            fingerprints.append(result["request_fingerprint"])
            entry.update(result)
            entry["attempt"] = None
            if result["status"] != "ok":
                save(f"{variant}_failed")
                continue
            plan = apply_patch(original_plan.model_copy(deep=True), PlanPatch.model_validate(result["patch"]))
            entry["status"] = "simulating"
            save(f"{variant}_simulating")
            rep_sim, rep_verdict, rep_s = _simulate(scene, plan, 2, run_dir / variant / "attempt_2", seed)
            entry["attempt"] = _attempt_record(rep_sim, rep_verdict, plan, f"LOCAL_GEMMA_{variant.upper()}", rep_s)
            entry["status"] = "ok"
            save(f"{variant}_judged")
        record["same_request_except_adapter"] = len(set(fingerprints)) == 1
        record["state"] = "done"
    except Exception as exc:  # recorded and shown; nothing is substituted
        record["state"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["finished_at"] = _now()
    save("done" if record["state"] == "done" else "error")
    return record


def list_compares() -> list[dict]:
    out = []
    if not config.RUNS_DIR.exists():
        return out
    for path in config.RUNS_DIR.glob(f"{COMPARE_PREFIX}*/compare.json"):
        data = _read_json(path)
        if not data:
            continue
        out.append({
            "compare_id": data.get("compare_id"),
            "created_at": data.get("created_at"),
            "state": data.get("state"),
            "case_id": data.get("case", {}).get("case_id"),
            "base_passed": (data["variants"]["base"].get("attempt") or {}).get("verdict", {}).get("passed"),
            "tuned_passed": (data["variants"]["tuned"].get("attempt") or {}).get("verdict", {}).get("passed"),
        })
    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", required=True, help="validation case_id, e.g. scene-13-variant-2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    record = run_compare(args.case, seed=args.seed, on_update=lambda r: print(r["phase"], file=sys.stderr))
    print(json.dumps({
        "compare_id": record["compare_id"], "state": record["state"], "error": record["error"],
        "original_passed": (record["original"] or {}).get("verdict", {}).get("passed"),
        **{v: {"status": e.get("status"), "patch": e.get("patch"), "http_s": e.get("http_s"),
               "passed": (e.get("attempt") or {}).get("verdict", {}).get("passed"), "error": e.get("error")}
           for v, e in record["variants"].items()},
    }, indent=2))
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
