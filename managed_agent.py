"""Antigravity Managed Agent sandbox geometric screening sweep for ArgonTwin.

Executes a single bounded sandbox session on `antigravity-preview-09-2026` with
`gemini-4-argon`, mounting `/workspace/sweep_input.json` inline, running a pure
Python stdlib geometric/reachability screening script over `trials` candidate
perturbations of the bimanual hand-over plan, downloading the remote environment
snapshot tarball, and cryptographically verifying execution proof before
returning a `schemas.SweepResult`.

Truth boundary:
- Inside the remote sandbox (`network: {"allowlist": []}`, Python stdlib only),
  `passed_trials` measures **stdlib geometric/reachability screening** against
  the scene's arm base envelopes, obstacle bounds, and tabletop clearance — NOT
  MuJoCo physics simulation or verified task success.
- After the sandbox returns its candidate `PlanPatch`, if local `sim.py` is
  available, `robustness_sweep` verifies the patched `best_plan` once with the
  real local `sim.run_attempt(..., render=False)` + `schemas.judge` and records
  both the sandbox geometric screening label and local MuJoCo verdict in
  `receipt.json`.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config
import schemas

SNAPSHOT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/files/"
    "environment-{environment_id}:download?alt=media"
)


class ManagedAgentError(schemas.ArgonInvalidOutput, RuntimeError):
    """Raised when the Managed Agent sandbox session or its proof validation fails."""


class SandboxSweepResult(BaseModel):
    """Schema for `/workspace/result.json` produced inside the Linux sandbox.

    Note: `passed_trials` from the remote stdlib sandbox represents geometric
    reachability/clearance screening (`screening_mode="stdlib_geometric_screening"`),
    not MuJoCo physics rollouts.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    nonce: str
    input_sha256: str
    platform: str
    task_object_pos_m: tuple[float, float, float]
    trials: int = Field(..., ge=1)
    passed_trials: int = Field(..., ge=0)
    screening_mode: Literal["stdlib_geometric_screening"] = "stdlib_geometric_screening"
    reasoning: str = Field(..., min_length=1, max_length=2000)
    patch: schemas.PlanPatch


# Alias for proof-validation test compatibility.
SandboxAnalysis = SandboxSweepResult


def read_artifact(archive: bytes, filename: str) -> bytes:
    """Safely read a regular file `/workspace/{filename}` from the environment tarball in memory."""
    target = f"workspace/{filename}"
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            for member in tar.getmembers():
                if (
                    member.name.lstrip("./") == target
                    and member.isfile()
                    and member.size <= 256_000
                ):
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        return extracted.read()
    except (tarfile.TarError, OSError) as exc:
        raise ManagedAgentError(
            f"Corrupt environment snapshot tarball while reading {target}: {exc}"
        ) from exc
    raise ManagedAgentError(f"Missing regular file {target} in environment snapshot")


def _extract_task_object_pos(scene_dict: dict[str, Any]) -> tuple[float, float, float]:
    task_name = scene_dict.get("task_object")
    objects = scene_dict.get("objects") or []
    for obj in objects:
        if obj.get("name") == task_name or obj.get("role") == "task_object":
            pos = obj.get("pos_m")
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
    raise ManagedAgentError("Scene payload has no valid task_object pos_m")


def validate_sweep(
    result: SandboxSweepResult,
    payload: dict[str, Any],
    raw: bytes,
    steps: list[dict[str, Any]],
) -> str:
    """Verify cryptographic input hash, Linux sandbox execution proof, and PlanPatch validity."""
    expected_sha = hashlib.sha256(raw).hexdigest()
    if result.nonce != payload.get("nonce") or result.input_sha256 != expected_sha:
        raise ManagedAgentError("Sandbox result does not match current input hash/nonce")
    if result.platform != "Linux":
        raise ManagedAgentError(f"Unexpected sandbox platform: {result.platform}")

    expected_trials = int(payload.get("trials", 0))
    if result.trials != expected_trials:
        raise ManagedAgentError(
            f"Sandbox trial count mismatch: expected {expected_trials}, got {result.trials}"
        )
    if result.passed_trials < 0 or result.passed_trials > result.trials:
        raise ManagedAgentError(
            f"Sandbox passed_trials {result.passed_trials} out of range [0, {result.trials}]"
        )

    expected_pos = _extract_task_object_pos(payload["scene"])
    if any(abs(a - b) > 1e-6 for a, b in zip(result.task_object_pos_m, expected_pos)):
        raise ManagedAgentError(
            f"Sandbox did not echo task_object_pos_m {expected_pos}; got {result.task_object_pos_m}"
        )

    base_plan = schemas.HandoverPlan.model_validate(payload["plan"])
    try:
        schemas.apply_patch(base_plan, result.patch)
    except ValidationError as exc:
        raise ManagedAgentError(
            f"Sandbox PlanPatch produced an invalid HandoverPlan: {exc}"
        ) from exc

    proof = f"SANDBOX_PROOF {result.nonce} {expected_sha}"
    call_ids = {
        step.get("id")
        for step in steps
        if step.get("type") == "code_execution_call" and step.get("id")
    }
    for step in steps:
        if (
            step.get("type") == "code_execution_result"
            and not step.get("is_error")
            and step.get("call_id") in call_ids
            and proof in (step.get("result") or "")
        ):
            return str(step["call_id"])
    raise ManagedAgentError("Interaction output contains no executed sandbox proof")


# Alias for proof-validation test compatibility.
validate_analysis = validate_sweep


def _extract_interaction_steps(interaction: Any) -> list[dict[str, Any]]:
    """Extract flat step dicts from `interaction.outputs` and/or `interaction.steps`."""
    flat: list[dict[str, Any]] = []
    for attr in ("outputs", "steps"):
        items = getattr(interaction, attr, None)
        if not isinstance(items, list):
            continue
        for item in items:
            d = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else (dict(item) if isinstance(item, dict) else {})
            )
            if isinstance(d.get("raw"), dict):
                d = {**d, **d["raw"]}
            if d:
                flat.append(d)
                content = d.get("content")
                if isinstance(content, list):
                    for sub in content:
                        if isinstance(sub, dict):
                            if isinstance(sub.get("raw"), dict):
                                sub = {**sub, **sub["raw"]}
                            flat.append(sub)
    return flat


def _resolve_sweep_out_dir(nonce: str, explicit_out_dir: Optional[Path] = None) -> Path:
    """Resolve output directory inside an active passing run directory if recently modified, without polluting RUNS_DIR top-level."""
    if explicit_out_dir is not None:
        return Path(explicit_out_dir)
    if config.RUNS_DIR.exists():
        now = time.time()
        candidates: list[Path] = []
        for p in config.RUNS_DIR.iterdir():
            mpath = p / "manifest.json"
            if not (p.is_dir() and mpath.exists() and (now - mpath.stat().st_mtime) < 300.0):
                continue
            try:
                mdata = json.loads(mpath.read_text(encoding="utf-8"))
                if mdata.get("sweep") is None and any(
                    a.get("verdict", {}).get("passed") for a in mdata.get("attempts", [])
                ):
                    candidates.append(p)
            except Exception:
                continue
        if candidates:
            latest_run = max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime)
            return latest_run / "sweep"
    return Path("/tmp/argon-twin-managed-agent") / nonce


def _is_chainable_interaction_id(ref: Optional[schemas.InteractionRef]) -> bool:
    """Return True only if `ref` holds a real `/v1beta/interactions` ID (never a `generateContent` response_id)."""
    if ref is None or not ref.interaction_id:
        return False
    api_kind = getattr(ref, "api_kind", None) or getattr(ref, "endpoint", None)
    if api_kind and "generate" in str(api_kind).lower():
        return False
    if ref.interaction_id.startswith("generateContent:"):
        return False
    return ref.interaction_id.startswith("v1_")


def _extract_usage(interaction: Any, wall_time_s: float) -> schemas.ArgonUsage:
    usage_obj = getattr(interaction, "usage", None)
    if usage_obj is None:
        return schemas.ArgonUsage(wall_time_s=round(wall_time_s, 3))
    dump = usage_obj.model_dump(mode="json") if hasattr(usage_obj, "model_dump") else {}
    return schemas.ArgonUsage(
        input_tokens=dump.get("total_input_tokens") or dump.get("input_tokens"),
        output_tokens=dump.get("total_output_tokens") or dump.get("output_tokens"),
        thought_tokens=dump.get("total_thought_tokens") or dump.get("thought_tokens"),
        total_tokens=dump.get("total_tokens"),
        wall_time_s=round(wall_time_s, 3),
    )


def robustness_sweep(
    scene: schemas.SceneSpec,
    plan: schemas.HandoverPlan,
    *,
    trials: int = 20,
    previous: Optional[schemas.InteractionRef] = None,
    deadline_seconds: int = 150,
    out_dir: Optional[Path] = None,
) -> schemas.SweepResult:
    """Run a parameter & perturbation sweep in the managed agent's Linux sandbox.

    Conforms to `schemas.ManagedAgentInterface.robustness_sweep`.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    key = config.gemini_api_key()
    t0 = time.monotonic()

    payload: dict[str, Any] = {
        "nonce": uuid.uuid4().hex,
        "trials": trials,
        "scene": scene.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "thresholds": {
            "can_target_tolerance_m": config.CAN_TARGET_TOLERANCE_M,
            "drop_margin_m": config.DROP_MARGIN_M,
            "max_can_speed_at_end_m_s": config.MAX_CAN_SPEED_AT_END_M_S,
            "max_arm_table_contact_force_n": config.MAX_ARM_TABLE_CONTACT_FORCE_N,
        },
    }
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    out_dir = _resolve_sweep_out_dir(payload["nonce"], out_dir)

    prompt = (
        "Read /workspace/sweep_input.json. Write /workspace/sweep.py and run it with "
        "the code execution tool. Do not install packages or use the network; use Python "
        "standard library only (json, hashlib, math, platform, random).\n\n"
        "IMPORTANT TRUTH LABEL: Because this sandbox has Python stdlib only (no MuJoCo/mink), "
        "your script performs explicit GEOMETRIC REACHABILITY & CLEARANCE SCREENING "
        "(`screening_mode = \"stdlib_geometric_screening\"`), NOT MuJoCo physics simulation.\n\n"
        "The script /workspace/sweep.py must:\n"
        "1. Read raw bytes of /workspace/sweep_input.json, compute hashlib.sha256(raw).hexdigest(), "
        "and parse the JSON payload (nonce, trials, scene, plan, thresholds).\n"
        "2. Find the task object in scene['objects'] (where name == scene['task_object'] or "
        "role == 'task_object') and extract its 3-element pos_m as task_object_pos_m.\n"
        "3. Screen `trials` deterministic seeded perturbations of the bimanual hand-over geometry "
        "against the scene's arm base envelopes (`left_arm_base_m`, `right_arm_base_m` within "
        "conservative 0.85m reach and overlapping bilateral workspace above the table), "
        "tabletop height margin (`> -drop_margin_m`), and non-task obstacle clearances to select "
        "the safest `PlanPatch` and count how many of the `trials` pass geometric screening.\n"
        "4. Write /workspace/result.json matching this exact JSON structure (no extra keys):\n"
        "   {\n"
        '     "nonce": payload["nonce"],\n'
        '     "input_sha256": sha256_hex,\n'
        '     "platform": platform.system(),\n'
        '     "task_object_pos_m": [x, y, z],\n'
        '     "trials": payload["trials"],\n'
        '     "passed_trials": <int between 0 and trials>,\n'
        '     "screening_mode": "stdlib_geometric_screening",\n'
        '     "reasoning": "[stdlib_geometric_screening] <concise summary of geometric screening findings and chosen patch>",\n'
        '     "patch": {\n'
        "       // only valid PlanPatch fields (or null/omitted if unchanged):\n"
        "       // grasp_height_offset_m [-0.05..0.05], approach_yaw_deg [-90..90],\n"
        "       // pregrasp_clearance_m [0.02..0.20], lift_height_m [0.03..0.40],\n"
        "       // handover_point_m [x, y, z], handover_yaw_offset_deg [-90..90],\n"
        "       // gripper_close_delay_s [0..2.0], release_delay_s [0..2.0],\n"
        "       // release_height_m [0..0.15], max_speed_m_s (0.02..1.0]\n"
        "     }\n"
        "   }\n"
        "5. Print the exact line `SANDBOX_PROOF <nonce> <input_sha256>` to stdout."
    )

    client = genai.Client(api_key=key)
    create_kwargs: dict[str, Any] = {
        "agent": config.GEMINI_AGENT,
        "agent_config": {
            "type": "antigravity",
            "model": config.GEMINI_MODEL,
            "max_total_tokens": 32_000,
        },
        "extra_body": {
            "environment": {
                "type": "remote",
                "network": {"allowlist": []},
                "sources": [
                    {
                        "type": "inline",
                        "target": "/workspace/sweep_input.json",
                        "content": raw.decode("utf-8"),
                    }
                ],
            }
        },
        "input": prompt,
        "background": True,
        "store": True,
    }
    chainable_prev_id = (
        previous.interaction_id if _is_chainable_interaction_id(previous) else None
    )
    if chainable_prev_id is not None:
        create_kwargs["previous_interaction_id"] = chainable_prev_id

    try:
        interaction = client.interactions.create(**create_kwargs)
    except Exception as exc:
        raise ManagedAgentError(f"Failed to start managed agent interaction: {exc}") from exc

    poll_interval_s = float(getattr(config, "ARGON_POLL_INTERVAL_S", 2.0))
    deadline = time.monotonic() + deadline_seconds
    while interaction.status in {"in_progress", "requires_action", "queued"}:
        if time.monotonic() >= deadline:
            raise ManagedAgentError(
                f"Managed agent interaction {interaction.id} timed out after {deadline_seconds}s"
            )
        time.sleep(poll_interval_s)
        interaction = client.interactions.get(interaction.id)

    if interaction.status != "completed" or not getattr(interaction, "environment_id", None):
        raise ManagedAgentError(
            f"Managed agent ended with status={interaction.status}, "
            f"environment_id={getattr(interaction, 'environment_id', None)}"
        )

    try:
        resp = httpx.get(
            SNAPSHOT_URL.format(environment_id=interaction.environment_id),
            headers={"x-goog-api-key": key},
            follow_redirects=True,
            timeout=60.0,
        )
        resp.raise_for_status()
        archive = resp.content
    except Exception as exc:
        raise ManagedAgentError(
            f"Failed to download environment snapshot {interaction.environment_id}: {exc}"
        ) from exc

    try:
        result_bytes = read_artifact(archive, "result.json")
        sweep_py_bytes = read_artifact(archive, "sweep.py")
        analysis = SandboxSweepResult.model_validate_json(result_bytes)
    except ValidationError as exc:
        raise ManagedAgentError(f"Sandbox result.json failed schema validation: {exc}") from exc

    steps = _extract_interaction_steps(interaction)
    call_id = validate_sweep(analysis, payload, raw, steps)

    candidate_plan = schemas.apply_patch(plan, analysis.patch)
    best_plan = candidate_plan

    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify candidate_plan with real local MuJoCo sim.py + schemas.judge when available
    # so a geometric screening patch never replaces a physics-passing plan with a failing one.
    local_mujoco_verification: dict[str, Any] = {"executed": False}
    try:
        sim = importlib.import_module("sim")
        sim_check_dir = out_dir / "local_mujoco_check"
        sim_attempt = sim.run_attempt(
            scene, candidate_plan, attempt=1, out_dir=sim_check_dir, render=False
        )
        verdict = schemas.judge(sim_attempt.metrics)
        local_mujoco_verification = {
            "executed": True,
            "candidate_plan_passed": verdict.passed,
            "failed_checks": verdict.failed_checks,
            "can_final_xy_error_m": sim_attempt.metrics.can_final_xy_error_m,
            "handover_completed": sim_attempt.metrics.handover_completed,
            "grasp_assist_used": sim_attempt.metrics.grasp_assist_used,
        }
        if not verdict.passed:
            best_plan = plan
            local_mujoco_verification["fallback_to_input_plan"] = True
    except Exception as exc:
        local_mujoco_verification = {"executed": False, "reason": str(exc)}

    wall_time_s = time.monotonic() - t0
    usage = _extract_usage(interaction, wall_time_s)
    interaction_ref = schemas.InteractionRef(
        interaction_id=interaction.id,
        previous_interaction_id=chainable_prev_id,
        environment_id=interaction.environment_id,
        model=config.GEMINI_MODEL,
        agent=config.GEMINI_AGENT,
    )

    result_path = out_dir / "result.json"
    sweep_py_path = out_dir / "sweep.py"
    receipt_path = out_dir / "receipt.json"

    result_path.write_bytes(result_bytes)
    sweep_py_path.write_bytes(sweep_py_bytes)
    receipt = {
        "evidence": "LIVE_ARGON_MANAGED_AGENT",
        "sandbox_evaluation": "stdlib_geometric_screening_not_mujoco_physics",
        "local_mujoco_verification": local_mujoco_verification,
        "interaction_id": interaction.id,
        "previous_interaction_id": chainable_prev_id,
        "upstream_ref": previous.interaction_id if previous else None,
        "environment_id": interaction.environment_id,
        "code_execution_call_id": call_id,
        "agent": config.GEMINI_AGENT,
        "model": config.GEMINI_MODEL,
        "usage": usage.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json"),
        "best_plan": best_plan.model_dump(mode="json"),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    return schemas.SweepResult(
        best_plan=best_plan,
        trials=analysis.trials,
        passed_trials=analysis.passed_trials,
        artifacts=[
            str(result_path),
            str(sweep_py_path),
            str(receipt_path),
        ],
        interaction=interaction_ref,
        usage=usage,
        mode="LIVE_ARGON_MANAGED_AGENT",
    )
