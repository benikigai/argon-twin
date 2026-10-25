"""Frozen data contracts and module interfaces for ArgonTwin.

Owner: integrator (Claude). CONTRACT_VERSION 1.0.1 is frozen (1.0.1 only adds finite-number validation): other owners code
against it and don't edit this file. Changes go through the integrator and bump
the version.

Frame convention for every position in this file: meters, origin at the center
of the table top, +z up, +x from the left arm toward the right arm, +y away from
the camera's starting side. An object's `pos_m` is the center of its base, so
an object resting on the table has z = 0.

Who fills what:
- sim.py (Codex) measures: SimAttempt, AttemptMetrics, Keyframe, RolloutLog.
- argon.py / managed_agent.py (Antigravity) produce: WorldBuild, Critique, SweepResult.
- judge() here decides pass/fail from measured metrics only. Model output never
  decides pass/fail; SupplementalCheck results are shown but don't change it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config

CONTRACT_VERSION = "1.0.1"  # 1.0.1: reject NaN and infinity in every float field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo order

ExecutionMode = Literal[
    "MANUAL",                    # hand-written scene/plan, no model call
    "LIVE_ARGON",                # gemini-4-argon via the Interactions API
    "LIVE_ARGON_MANAGED_AGENT",  # antigravity managed agent with gemini-4-argon
    "ARGON_INVALID",             # model output failed validation after retry; previous plan kept
    "REPLAY",                    # recorded run played back, nothing live
]

Camera = Literal["overhead", "front", "left_wrist", "right_wrist"]
Phase = Literal["approach", "grasp", "lift", "handover", "place", "settle"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# --- Errors ------------------------------------------------------------------

class ArgonInvalidOutput(Exception):
    """Model output failed schema validation after the allowed retries."""

    def __init__(self, message: str, *, raw: str = "", errors: str = ""):
        super().__init__(message)
        self.raw = raw
        self.errors = errors


class SimError(Exception):
    """The scene could not be built or simulated (not a task failure)."""


# --- Scene -------------------------------------------------------------------

class ObjectSpec(Strict):
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{0,31}$")
    kind: Literal["cylinder", "box"]
    size_m: tuple[float, ...] = Field(..., description="cylinder: (radius, height); box: (x, y, z)")
    pos_m: Vec3
    yaw_deg: float = Field(0.0, ge=-180.0, le=180.0)
    mass_kg: float = Field(0.3, gt=0.0, le=5.0)
    graspable: bool = False

    @model_validator(mode="after")
    def _size_matches_kind(self) -> "ObjectSpec":
        expected = 2 if self.kind == "cylinder" else 3
        if len(self.size_m) != expected:
            raise ValueError(f"{self.kind} needs {expected} size values, got {len(self.size_m)}")
        if any(v <= 0 or v > 2.0 for v in self.size_m):
            raise ValueError("sizes must be in (0, 2] m")
        return self


class SceneSpec(Strict):
    table_shape: Literal["round", "rect"]
    table_size_m: tuple[float, float] = Field(..., description="rect: (x, y); round: (diameter, diameter)")
    table_height_m: float = Field(..., gt=0.3, lt=1.3)
    objects: list[ObjectSpec] = Field(..., min_length=1, max_length=8)
    task_object: str = Field(..., description="name of the object handed over, e.g. 'can'")
    left_arm_base_m: Vec3
    right_arm_base_m: Vec3
    handover_point_m: Vec3
    target_m: Vec3
    evidence: list[str] = Field(default_factory=list, description="what in the video each estimate came from")
    source: Literal["argon", "manual"]

    @model_validator(mode="after")
    def _task_object_is_graspable(self) -> "SceneSpec":
        names = [o.name for o in self.objects]
        if len(set(names)) != len(names):
            raise ValueError("object names must be unique")
        match = [o for o in self.objects if o.name == self.task_object]
        if not match:
            raise ValueError(f"task_object '{self.task_object}' is not in objects")
        if not match[0].graspable:
            raise ValueError(f"task_object '{self.task_object}' must be graspable")
        return self


# --- Plan --------------------------------------------------------------------

class HandoverPlan(Strict):
    """Parameters the scripted controller in sim.py executes.

    Sequence: left arm approaches and grasps the task object, lifts, moves to
    the hand-over point; right arm grasps; left releases; right moves to the
    target and releases at release_height_m above the table.
    """

    grasp_height_offset_m: float = Field(0.0, ge=-0.05, le=0.05, description="grasp point relative to object mid-height")
    approach_yaw_deg: float = Field(0.0, ge=-90.0, le=90.0)
    pregrasp_clearance_m: float = Field(0.08, ge=0.02, le=0.20)
    lift_height_m: float = Field(0.15, ge=0.03, le=0.40)
    handover_point_m: Vec3
    handover_yaw_offset_deg: float = Field(0.0, ge=-90.0, le=90.0, description="right gripper yaw relative to left at hand-over")
    gripper_close_delay_s: float = Field(0.25, ge=0.0, le=2.0)
    release_delay_s: float = Field(0.25, ge=0.0, le=2.0, description="left opens this long after right closes")
    release_height_m: float = Field(0.02, ge=0.0, le=0.15)
    max_speed_m_s: float = Field(0.25, gt=0.02, le=1.0)


class PlanPatch(Strict):
    """Partial update to a HandoverPlan. None means unchanged."""

    grasp_height_offset_m: Optional[float] = Field(None, ge=-0.05, le=0.05)
    approach_yaw_deg: Optional[float] = Field(None, ge=-90.0, le=90.0)
    pregrasp_clearance_m: Optional[float] = Field(None, ge=0.02, le=0.20)
    lift_height_m: Optional[float] = Field(None, ge=0.03, le=0.40)
    handover_point_m: Optional[Vec3] = None
    handover_yaw_offset_deg: Optional[float] = Field(None, ge=-90.0, le=90.0)
    gripper_close_delay_s: Optional[float] = Field(None, ge=0.0, le=2.0)
    release_delay_s: Optional[float] = Field(None, ge=0.0, le=2.0)
    release_height_m: Optional[float] = Field(None, ge=0.0, le=0.15)
    max_speed_m_s: Optional[float] = Field(None, gt=0.02, le=1.0)


def apply_patch(plan: HandoverPlan, patch: PlanPatch) -> HandoverPlan:
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    return plan.model_copy(update=changes)


# --- Model-call records ------------------------------------------------------

class InteractionRef(Strict):
    interaction_id: str
    previous_interaction_id: Optional[str] = None
    environment_id: Optional[str] = None
    model: str = config.GEMINI_MODEL
    agent: Optional[str] = None


class ArgonUsage(Strict):
    """Copied from the API response. Unknown values stay None; never estimated."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thought_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    wall_time_s: float


MetricName = Literal[
    "can_final_xy_error_m",
    "can_min_z_m",
    "can_final_speed_m_s",
    "max_arm_table_contact_force_n",
    "handover_completed",
    "left_grasped",
    "right_grasped",
]


class SupplementalCheck(Strict):
    """Declarative check proposed by Argon. Evaluated on measured metrics; never executed as code."""

    name: str
    metric: MetricName
    op: Literal["<=", "<", ">=", ">", "=="]
    value: float
    rationale: str


class SupplementalCheckResult(Strict):
    check: SupplementalCheck
    measured: float
    passed: bool


# Model-facing bodies: what argon.py asks Gemini to return (use as response schema).

class WorldBuildBody(Strict):
    scene: SceneSpec
    initial_plan: HandoverPlan
    supplemental_checks: list[SupplementalCheck] = Field(default_factory=list, max_length=12)
    notes: list[str] = Field(default_factory=list)


class CritiqueBody(Strict):
    diagnosis: list[str] = Field(..., min_length=2, max_length=4, description="cause chain tied to keyframes and metrics")
    cited_keyframes: list[str] = Field(..., min_length=1, max_length=4, description="Keyframe.label values used as evidence")
    patch: PlanPatch


# Full records: body plus provenance, filled by argon.py.

class WorldBuild(Strict):
    body: WorldBuildBody
    interaction: InteractionRef
    usage: ArgonUsage
    mode: Literal["LIVE_ARGON", "LIVE_ARGON_MANAGED_AGENT", "REPLAY"]


class Critique(Strict):
    body: CritiqueBody
    interaction: InteractionRef
    usage: ArgonUsage
    mode: Literal["LIVE_ARGON", "LIVE_ARGON_MANAGED_AGENT", "REPLAY"]


# --- Simulation outputs (sim.py) ---------------------------------------------

class Keyframe(Strict):
    label: str = Field(..., description="e.g. 'grasp', 'handover', 'drop', 'final'")
    phase: Phase
    t_s: float
    camera: Camera
    path: str = Field(..., description="PNG path relative to the attempt's out_dir")


class AttemptMetrics(Strict):
    """Measured by MuJoCo. Heights are relative to the table top."""

    can_final_pos_m: Vec3
    can_final_xy_error_m: float = Field(..., ge=0.0, description="xy distance from final can center to scene.target_m")
    can_min_z_m: float
    can_final_speed_m_s: float = Field(..., ge=0.0)
    left_grasped: bool
    right_grasped: bool
    handover_completed: bool = Field(..., description="right gripper held the object while the left gripper was open")
    max_arm_table_contact_force_n: float = Field(..., ge=0.0)
    grasp_assist_used: bool = Field(..., description="a weld attached the object; must be shown in the viewer")
    failure_phase: Optional[Phase] = Field(None, description="first phase where a measured condition failed")
    sim_time_s: float
    wall_time_ms: float


class GeomDesc(Strict):
    type: Literal["box", "sphere", "capsule", "cylinder", "ellipsoid", "mesh", "plane"]
    size: tuple[float, ...]
    pos: Vec3
    quat: Quat
    rgba: tuple[float, float, float, float]
    mesh_file: Optional[str] = Field(None, description="path under web/assets/meshes/ for mesh geoms")


class RolloutBody(Strict):
    name: str
    geoms: list[GeomDesc]


class RolloutLog(Strict):
    """Written by sim.py as JSON at SimAttempt.rollout_path for the viewer."""

    contract_version: str = CONTRACT_VERSION
    fps: float
    bodies: list[RolloutBody]
    frames: list[list[float]] = Field(..., description="per frame: 7 floats per body (pos xyz, quat wxyz), in body order")


class SimAttempt(Strict):
    attempt: int = Field(..., ge=1)
    metrics: AttemptMetrics
    keyframes: list[Keyframe]
    rollout_path: str
    mjcf_path: str
    warnings: list[str] = Field(default_factory=list)


# --- Deterministic judging ---------------------------------------------------

class Verdict(Strict):
    passed: bool
    failed_checks: list[str]
    thresholds: dict[str, float]


def judge(metrics: AttemptMetrics) -> Verdict:
    """The only pass/fail decision in the system. Thresholds come from config.py."""
    failed: list[str] = []
    if not metrics.handover_completed:
        failed.append("handover_completed is false")
    if metrics.can_min_z_m < -config.DROP_MARGIN_M:
        failed.append(f"dropped: can_min_z_m {metrics.can_min_z_m:.3f} < -{config.DROP_MARGIN_M}")
    if metrics.can_final_xy_error_m > config.CAN_TARGET_TOLERANCE_M:
        failed.append(f"off target: {metrics.can_final_xy_error_m:.3f} m > {config.CAN_TARGET_TOLERANCE_M} m")
    if metrics.can_final_speed_m_s > config.MAX_CAN_SPEED_AT_END_M_S:
        failed.append(f"not at rest: {metrics.can_final_speed_m_s:.3f} m/s")
    if metrics.max_arm_table_contact_force_n > config.MAX_ARM_TABLE_CONTACT_FORCE_N:
        failed.append(f"arm hit table: {metrics.max_arm_table_contact_force_n:.1f} N")
    return Verdict(
        passed=not failed,
        failed_checks=failed,
        thresholds={
            "can_target_tolerance_m": config.CAN_TARGET_TOLERANCE_M,
            "drop_margin_m": config.DROP_MARGIN_M,
            "max_can_speed_at_end_m_s": config.MAX_CAN_SPEED_AT_END_M_S,
            "max_arm_table_contact_force_n": config.MAX_ARM_TABLE_CONTACT_FORCE_N,
        },
    )


def evaluate_supplemental(checks: list[SupplementalCheck], metrics: AttemptMetrics) -> list[SupplementalCheckResult]:
    ops = {
        "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "==": lambda a, b: a == b,
    }
    results = []
    for check in checks:
        measured = float(getattr(metrics, check.metric))
        results.append(SupplementalCheckResult(check=check, measured=measured, passed=ops[check.op](measured, check.value)))
    return results


# --- Loop records (loop.py) --------------------------------------------------

class AttemptResult(Strict):
    attempt: int = Field(..., ge=1)
    mode: ExecutionMode
    plan: HandoverPlan
    sim: SimAttempt
    verdict: Verdict
    supplemental: list[SupplementalCheckResult] = Field(default_factory=list)
    critique: Optional[Critique] = Field(None, description="Argon's critique of this attempt; drives the next plan")


class SweepResult(Strict):
    """Managed-agent robustness sweep. pass_rate is measured in the sandbox, not estimated."""

    best_plan: HandoverPlan
    trials: int = Field(..., ge=1)
    passed_trials: int = Field(..., ge=0)
    artifacts: list[str] = Field(default_factory=list)
    interaction: InteractionRef
    usage: ArgonUsage
    mode: Literal["LIVE_ARGON_MANAGED_AGENT"] = "LIVE_ARGON_MANAGED_AGENT"


class RunManifest(Strict):
    contract_version: str = CONTRACT_VERSION
    run_id: str
    created_at: datetime
    video_path: Optional[str] = None
    scene: SceneSpec
    world_build: Optional[WorldBuild] = None
    attempts: list[AttemptResult]
    sweep: Optional[SweepResult] = None
    passed: bool
    splat_path: Optional[str] = None


# --- Module interfaces -------------------------------------------------------
# Each owner implements these as module-level functions with exactly these
# signatures. loop.py calls only these.

class SimInterface(Protocol):
    def validate_scene(self, scene: SceneSpec) -> list[str]:
        """Build the MJCF and return problems (unreachable target, overlaps). Empty list = OK.
        Raises SimError only if the scene can't be compiled at all."""
        ...

    def run_attempt(
        self, scene: SceneSpec, plan: HandoverPlan, *,
        attempt: int, out_dir: Path, render: bool = True, seed: int = 0,
    ) -> SimAttempt:
        """Run one scripted hand-over. Deterministic for the same inputs and seed.
        Writes keyframe PNGs, rollout JSON (RolloutLog) and the MJCF into out_dir.
        Measures; does not judge."""
        ...


class ArgonInterface(Protocol):
    def build_world(self, video_path: Path, *, frames_dir: Optional[Path] = None) -> WorldBuild:
        """One background long-decode call: video in, WorldBuildBody out.
        Retries once on invalid output, then raises ArgonInvalidOutput."""
        ...

    def critique(
        self, scene: SceneSpec, plan: HandoverPlan, attempt: AttemptResult, *,
        attempt_dir: Path, previous: Optional[InteractionRef] = None,
    ) -> Critique:
        """Read the attempt's keyframes (under attempt_dir) and metrics; return a CritiqueBody.
        Chains on previous.interaction_id when given. Retries once, then raises ArgonInvalidOutput."""
        ...


class ManagedAgentInterface(Protocol):
    def robustness_sweep(
        self, scene: SceneSpec, plan: HandoverPlan, *,
        trials: int = 20, previous: Optional[InteractionRef] = None,
    ) -> SweepResult:
        """Run a parameter sweep in the managed agent's sandbox and return the best plan
        with measured trial counts."""
        ...
