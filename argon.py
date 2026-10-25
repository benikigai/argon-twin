"""Gemini 4 Argon multimodal client for ArgonTwin.

Owner: Antigravity. Implements `schemas.ArgonInterface` (`build_world`, `critique`)
against frozen `schemas.CONTRACT_VERSION == "1.0.0"` and `config.CONFIG_VERSION == "1.0.0"`.

Routing note (verified live on google-genai 2.33.0 with event key):
- `/v1beta/interactions` routes `model="gemini-4-argon"` to `models/gemini-4-argon-agent`,
  which rejects image inputs with HTTP 400 (`Image input modality is not enabled for
  models/gemini-4-argon-agent`).
- `/v1beta/models/gemini-4-argon:generateContent` (`client.models.generate_content` with
  `model="gemini-4-argon"`) routes directly to multimodal `models/gemini-4-argon`,
  which natively accepts image/video `Part.from_bytes` inputs alongside `ThinkingConfig`
  and returns `response_id` + `usage_metadata` (including `IMAGE` prompt token counts).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

import config
import schemas

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_VIDEO_MIME_MAP = {
    ".mov": "video/mov",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".avi": "video/avi",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
}

_IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _extract_response_text(response: Any) -> str:
    """Extract text output from a GenerateContentResponse or Interaction object."""
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    chunks: list[str] = []
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None)
            if isinstance(parts, list):
                for part in parts:
                    if getattr(part, "thought", False):
                        continue
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        chunks.append(part_text)
    if chunks:
        return "\n".join(chunks)

    outputs = getattr(response, "outputs", None)
    if isinstance(outputs, list):
        for item in outputs:
            d = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else (item if isinstance(item, dict) else {})
            )
            item_type = d.get("type") or getattr(item, "type", None)
            item_text = d.get("text") or getattr(item, "text", None)
            if item_type == "text" and isinstance(item_text, str):
                chunks.append(item_text)
    if chunks:
        return "\n".join(chunks)

    steps = getattr(response, "steps", None)
    if isinstance(steps, list):
        for step in steps:
            d = (
                step.model_dump(mode="json")
                if hasattr(step, "model_dump")
                else (step if isinstance(step, dict) else {})
            )
            if d.get("type") == "text" and isinstance(d.get("text"), str):
                chunks.append(d["text"])
            content = d.get("content")
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        chunks.append(part["text"])
    return "\n".join(chunks)


def _extract_json_text(raw_text: str) -> str:
    """Strip markdown code fences or surrounding prose to isolate the JSON object."""
    text = (raw_text or "").strip()
    if not text:
        return ""
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _build_usage(
    responses: list[Any], wall_time_s: float
) -> schemas.ArgonUsage:
    """Copy token counts from the API response(s) without estimating missing fields."""
    in_tok: Optional[int] = None
    out_tok: Optional[int] = None
    thought_tok: Optional[int] = None
    total_tok: Optional[int] = None

    for resp in responses:
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            dump = (
                usage_meta.model_dump(mode="json")
                if hasattr(usage_meta, "model_dump")
                else {}
            )
            p_tok = dump.get("prompt_token_count")
            c_tok = dump.get("candidates_token_count")
            t_tok = dump.get("thoughts_token_count")
            tot_tok = dump.get("total_token_count")
            if p_tok is not None:
                in_tok = (in_tok or 0) + int(p_tok)
            if c_tok is not None:
                out_tok = (out_tok or 0) + int(c_tok)
            if t_tok is not None:
                thought_tok = (thought_tok or 0) + int(t_tok)
            if tot_tok is not None:
                total_tok = (total_tok or 0) + int(tot_tok)
            continue

        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            dump = (
                usage_obj.model_dump(mode="json")
                if hasattr(usage_obj, "model_dump")
                else {}
            )
            p_tok = dump.get("total_input_tokens") or dump.get("input_tokens")
            c_tok = dump.get("total_output_tokens") or dump.get("output_tokens")
            t_tok = dump.get("total_thought_tokens") or dump.get("thought_tokens")
            tot_tok = dump.get("total_tokens")
            if p_tok is not None:
                in_tok = (in_tok or 0) + int(p_tok)
            if c_tok is not None:
                out_tok = (out_tok or 0) + int(c_tok)
            if t_tok is not None:
                thought_tok = (thought_tok or 0) + int(t_tok)
            if tot_tok is not None:
                total_tok = (total_tok or 0) + int(tot_tok)

    return schemas.ArgonUsage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        thought_tokens=thought_tok,
        total_tokens=total_tok,
        wall_time_s=round(wall_time_s, 3),
    )


def _response_id(response: Any) -> str:
    """Return the actual server-assigned response_id from generateContent (never fabricated)."""
    rid = getattr(response, "response_id", None)
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    iid = getattr(response, "id", None)
    if isinstance(iid, str) and iid.strip():
        return iid.strip()
    raise RuntimeError("Gemini API response did not include a valid response_id or interaction id")


def _build_generate_content_ref(response: Any) -> schemas.InteractionRef:
    """Build an InteractionRef for `models.generate_content` with explicit endpoint provenance.

    Never presents a stateless `generateContent` response_id as a chainable `/v1beta/interactions`
    ID and never populates `previous_interaction_id`. Supports optional future schema extensions
    (`api_kind`, `endpoint`, `response_id`) if added to `schemas.InteractionRef`.
    """
    raw_id = _response_id(response)
    fields = schemas.InteractionRef.model_fields
    kwargs: dict[str, Any] = {
        "interaction_id": raw_id if ("api_kind" in fields or "endpoint" in fields) else f"generateContent:{raw_id}",
        "previous_interaction_id": None,
        "environment_id": None,
        "model": config.GEMINI_MODEL,
        "agent": None,
    }
    if "api_kind" in fields:
        kwargs["api_kind"] = "generateContent"
    if "endpoint" in fields:
        kwargs["endpoint"] = f"models/{config.GEMINI_MODEL}:generateContent"
    if "response_id" in fields:
        kwargs["response_id"] = raw_id
    return schemas.InteractionRef(**kwargs)


def _image_part_from_path(img_path: Path) -> dict[str, Any]:
    ext = img_path.suffix.lower()
    mime = _IMAGE_MIME_MAP.get(ext) or mimetypes.guess_type(img_path.name)[0] or "image/png"
    raw_bytes = img_path.read_bytes()
    return {
        "type": "image",
        "mime_type": mime,
        "data": base64.b64encode(raw_bytes).decode("ascii"),
        "raw_bytes": raw_bytes,
    }


def _extract_frames_from_video(video_path: Path, count: int = 5) -> list[dict[str, Any]]:
    """Extract evenly-spaced JPEG frames from video_path using cv2 or ffmpeg."""
    parts: list[dict[str, Any]] = []
    if not video_path.exists():
        return parts

    # Try OpenCV first if installed in the active environment
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            indices = (
                [int(round(i * (total - 1) / max(count - 1, 1))) for i in range(count)]
                if total > count
                else list(range(max(total, 1)))
            )
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024.0 / float(max(h, w))
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_enc:
                    raw_bytes = buf.tobytes()
                    t_s = idx / max(fps, 1.0)
                    parts.append({"type": "text", "text": f"Video frame index={idx} t={t_s:.2f}s"})
                    parts.append({
                        "type": "image",
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(raw_bytes).decode("ascii"),
                        "raw_bytes": raw_bytes,
                    })
            cap.release()
            if parts:
                return parts
    except Exception:
        pass

    # Fallback to ffmpeg CLI if available
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            duration_s = 10.4
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                duration_s = max(float(probe.stdout.strip()), 1.0)
            except Exception:
                pass
            sample_fps = max(count / duration_s, 0.1)
            out_pattern = str(Path(tmpdir) / "frame_%02d.jpg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"fps={sample_fps:.4f},scale=1024:1024:force_original_aspect_ratio=decrease",
                    "-frames:v",
                    str(count),
                    "-q:v",
                    "3",
                    out_pattern,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            extracted = sorted(Path(tmpdir).glob("frame_*.jpg"))
            for idx, fp in enumerate(extracted):
                parts.append({"type": "text", "text": f"Extracted video frame #{idx + 1} ({fp.name})"})
                parts.append(_image_part_from_path(fp))
    except Exception:
        pass

    return parts


def _collect_video_and_frame_inputs(
    video_path: Path, frames_dir: Optional[Path], *, include_inline_video_max_bytes: int = 10 * 1024 * 1024
) -> list[dict[str, Any]]:
    """Build multimodal input parts from frames_dir and/or video_path."""
    content: list[dict[str, Any]] = []

    if frames_dir is not None and frames_dir.exists():
        frame_files = sorted(
            p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_MIME_MAP
        )
        if frame_files:
            step = max(1, len(frame_files) // 6)
            selected = frame_files[::step][:6]
            for idx, fp in enumerate(selected):
                content.append({"type": "text", "text": f"Workspace frame #{idx + 1}: {fp.name}"})
                content.append(_image_part_from_path(fp))

    if not content and video_path.exists():
        content.extend(_extract_frames_from_video(video_path, count=5))

    if video_path.exists() and video_path.is_file():
        size_bytes = video_path.stat().st_size
        if size_bytes <= include_inline_video_max_bytes or not content:
            mime = _VIDEO_MIME_MAP.get(video_path.suffix.lower(), "video/mov")
            raw_bytes = video_path.read_bytes()
            content.append({"type": "text", "text": f"Workspace video: {video_path.name}"})
            content.append({
                "type": "video",
                "mime_type": mime,
                "data": base64.b64encode(raw_bytes).decode("ascii"),
                "raw_bytes": raw_bytes,
            })

    if not content:
        raise FileNotFoundError(
            f"Neither readable frames in {frames_dir} nor video at {video_path} could be loaded."
        )
    return content


def _to_genai_parts(content: list[dict[str, Any]]) -> list[types.Part]:
    """Convert internal multimodal dicts into `google.genai.types.Part` objects."""
    parts: list[types.Part] = []
    for item in content:
        kind = item.get("type")
        if kind == "text":
            parts.append(types.Part.from_text(text=str(item.get("text", ""))))
        elif kind in ("image", "video"):
            raw_bytes = item.get("raw_bytes")
            if raw_bytes is None:
                b64 = item.get("data", "")
                raw_bytes = base64.b64decode(b64) if b64 else b""
            mime = str(item.get("mime_type") or ("image/png" if kind == "image" else "video/mov"))
            parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
    return parts


def _run_model_interaction(
    client: genai.Client,
    *,
    content: list[dict[str, Any]],
    thinking_level: str,
    max_output_tokens: int,
    timeout_s: float,
) -> tuple[Any, float]:
    """Run a multimodal `gemini-4-argon` call via `models.generate_content`.

    Note: `/v1beta/interactions` routes `gemini-4-argon` to `models/gemini-4-argon-agent`
    which rejects image inputs (`Image input modality is not enabled for
    models/gemini-4-argon-agent`). `client.models.generate_content(model="gemini-4-argon")`
    routes directly to multimodal `models/gemini-4-argon` and preserves all image/video parts.
    Stateless `generateContent` calls do not take or emit `previous_interaction_id`.
    """
    t0 = time.monotonic()
    parts = _to_genai_parts(content)
    gen_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=str(thinking_level).upper()),
        max_output_tokens=max_output_tokens,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=parts,
        config=gen_config,
    )
    wall_time_s = time.monotonic() - t0
    return response, wall_time_s


def build_world(video_path: Path, *, frames_dir: Optional[Path] = None) -> schemas.WorldBuild:
    """One long-decode call on multimodal `gemini-4-argon`: video/frames in, WorldBuildBody out.

    Retries once on invalid output, then raises ArgonInvalidOutput.
    """
    key = config.gemini_api_key()
    client = genai.Client(api_key=key, http_options={"timeout": 120_000})

    media_parts = _collect_video_and_frame_inputs(video_path, frames_dir)
    schema_json = json.dumps(schemas.WorldBuildBody.model_json_schema(), indent=2)

    prompt = f"""You are building a deterministic MuJoCo physics twin for a bimanual ALOHA robot hand-over task from the workspace video/frames.

Coordinate frame and physical constraints:
- Units: meters.
- Origin (0, 0, 0) is at the center of the table top surface, +z is up, +x points from the left ALOHA arm toward the right ALOHA arm, +y points away from the camera's starting side.
- Every object's `pos_m` is [x, y, z] of the center of its base on the table (an object resting on the table has z = 0.0).
- Object names must match `^[a-z][a-z0-9_]{{0,31}}$` and be unique.
- `kind` is either `"cylinder"` with `size_m: [radius, height]` (2 values in (0, 2]) or `"box"` with `size_m: [x, y, z]` (3 values in (0, 2]).
- Estimate the table shape, dimensions, and visible objects on the table from the video frames, recording your visual basis for each estimate in `scene.evidence`.
- `task_object` must be the name of the graspable object to hand over (with `"graspable": true` in `objects`); non-task obstacles on the table have `"graspable": false`.
- Set `scene.pickup_arm` (`"left"` or `"right"`, default `"left"`) to the donor arm on the `task_object`'s starting side (`-x` for `"left"`, `+x` for `"right"`), so the donor arm picks up `task_object` and hands it to the opposite receiver arm (`"right"` when `pickup_arm == "left"`, `"left"` when `pickup_arm == "right"`).
- Set `scene.source` to `"argon"`.
- The two ALOHA arms are mounted on opposite sides of the table along the x-axis (left arm on `-x`, right arm on `+x`, each with a conservative 0.85 m reach envelope from its base). Choose `scene.handover_point_m` above the table within reach of both arms and `scene.target_m` on the receiver arm's side of the table clear of obstacles and within the receiver arm's reach.
- Provide `initial_plan` (`HandoverPlan`) and up to 6 `supplemental_checks` (`SupplementalCheck`) using only valid `MetricName` values:
  `"can_final_xy_error_m"`, `"can_min_z_m"`, `"can_final_speed_m_s"`, `"max_arm_table_contact_force_n"`, `"max_arm_obstacle_contact_force_n"`, `"max_arm_obstacle_penetration_m"`, `"max_object_obstacle_penetration_m"`, `"handover_completed"`, `"left_grasped"`, `"right_grasped"`.

Return ONLY a single valid JSON object conforming strictly to this `WorldBuildBody` JSON Schema (no markdown commentary outside JSON):
{schema_json}
"""

    base_content = [{"type": "text", "text": prompt}, *media_parts]
    content = list(base_content)
    total_wall_s = 0.0
    last_raw = ""
    last_err = ""
    responses: list[Any] = []

    for attempt_idx in range(config.ARGON_INVALID_RETRIES + 1):
        response, wall_s = _run_model_interaction(
            client,
            content=content,
            thinking_level=config.WORLD_BUILD_THINKING,
            max_output_tokens=config.WORLD_BUILD_MAX_OUTPUT_TOKENS,
            timeout_s=config.WORLD_BUILD_TIMEOUT_S,
        )
        responses.append(response)
        total_wall_s += wall_s
        last_raw = _extract_response_text(response)
        cleaned = _extract_json_text(last_raw)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and isinstance(parsed.get("scene"), dict):
                parsed["scene"]["source"] = "argon"
            body = schemas.WorldBuildBody.model_validate(parsed)
            return schemas.WorldBuild(
                body=body,
                interaction=_build_generate_content_ref(response),
                usage=_build_usage(responses, total_wall_s),
                mode="LIVE_ARGON",
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_err = str(exc)
            if attempt_idx < config.ARGON_INVALID_RETRIES:
                retry_prompt = (
                    f"Your previous response:\n{last_raw[:4000]}\n\n"
                    "failed `WorldBuildBody` validation with the following error:\n"
                    f"{last_err}\n\n"
                    'Return ONLY the corrected JSON object matching `WorldBuildBody` with `scene.source = "argon"`.'
                )
                content = [*base_content, {"type": "text", "text": retry_prompt}]

    raise schemas.ArgonInvalidOutput(
        f"WorldBuildBody validation failed after {config.ARGON_INVALID_RETRIES + 1} attempts: {last_err}",
        raw=last_raw,
        errors=last_err,
    )


def critique(
    scene: schemas.SceneSpec,
    plan: schemas.HandoverPlan,
    attempt: schemas.AttemptResult,
    *,
    attempt_dir: Path,
    previous: Optional[schemas.InteractionRef] = None,
) -> schemas.Critique:
    """Read the attempt's keyframes (under attempt_dir) and metrics; return a CritiqueBody.

    Preserves all rendered keyframe PNG images as multimodal inputs to `gemini-4-argon`
    via stateless `models.generate_content` (`previous_interaction_id` is not chained).
    Retries once, then raises ArgonInvalidOutput.
    """
    key = config.gemini_api_key()
    client = genai.Client(api_key=key, http_options={"timeout": 90_000})

    keyframe_parts: list[dict[str, Any]] = []
    available_labels: list[str] = []
    for kf in attempt.sim.keyframes:
        available_labels.append(kf.label)
        kf_path = attempt_dir / kf.path
        if kf_path.exists() and kf_path.is_file():
            keyframe_parts.append({
                "type": "text",
                "text": (
                    f'Keyframe label="{kf.label}" phase="{kf.phase}" '
                    f'camera="{kf.camera}" t_s={kf.t_s:.2f}'
                ),
            })
            keyframe_parts.append(_image_part_from_path(kf_path))

    if not keyframe_parts:
        raise FileNotFoundError(
            f"No rendered keyframe PNGs found under {attempt_dir} for visual critique."
        )

    schema_json = json.dumps(schemas.CritiqueBody.model_json_schema(), indent=2)
    scene_context_json = scene.model_dump_json(indent=2, exclude={"handover_point_m"})
    valid_labels_note = (
        f"Available keyframe labels (you MUST cite 1 to 4 of these exact strings in `cited_keyframes`): {json.dumps(available_labels)}"
    )

    donor = getattr(scene, "pickup_arm", "left")
    receiver = "right" if donor == "left" else "left"
    donor_sign_str = "-pregrasp_clearance_m" if donor == "left" else "+pregrasp_clearance_m"
    receiver_sign_str = "+pregrasp_clearance_m" if receiver == "right" else "-pregrasp_clearance_m"
    donor_retract_str = "-0.16" if donor == "left" else "+0.16"
    receiver_settle_str = "+0.12" if receiver == "right" else "-0.12"

    prompt = f"""You are critiquing a simulated bimanual ALOHA robot hand-over attempt in MuJoCo using the attached rendered keyframe images and measured physics metrics, and proposing a `PlanPatch` to fix the failure on the next attempt.

Scene specification (table, object geometry, arm bases, `pickup_arm`, and landing target):
{scene_context_json}

Executed plan (`HandoverPlan` for Attempt #{attempt.attempt} — the executed `handover_point_m` and motion parameters must be reconsidered for physical feasibility):
{plan.model_dump_json(indent=2)}

Measured MuJoCo metrics (`AttemptMetrics`):
{attempt.sim.metrics.model_dump_json(indent=2)}

Deterministic judge verdict (`Verdict`):
{attempt.verdict.model_dump_json(indent=2)}

Pass thresholds (`config`) & physical controller constraints:
- Coordinate frame: meters, origin at tabletop center, +z up, +x from left arm (`left_arm_base_m={list(scene.left_arm_base_m)}`) toward right arm (`right_arm_base_m={list(scene.right_arm_base_m)}`), +y away from the front camera. Each arm has a conservative reach envelope of ~0.85 m from its base, and bilateral hand-overs require `plan.handover_point_m` to lie comfortably within the overlapping workspace of both arms above the table and clear of obstacles.
- Donor & receiver arm roles (`scene.pickup_arm = "{donor}"`):
  - Donor arm is `{donor}` and opposite receiver arm is `{receiver}` (pregrasp sign in x is `+x` for `"right"` and `-x` for `"left"`).
  - Grasp & lift: donor (`{donor}`) approaches `task_object` (`"{scene.task_object}"`) from `{donor_sign_str}` in x (`grasp + [{donor_sign_str}, 0, 0.035]` where `grasp = obj_center + [0, 0, grasp_height_offset_m]`), moves to `grasp`, closes and attaches within 0.055 m of object center (`{donor}_grasped = true`), then lifts to `grasp + [0, 0, lift_height_m]`.
  - Hand-over & release: donor (`{donor}`) moves to `plan.handover_point_m + [0, 0, 0.02]` while receiver (`{receiver}`) pregrasps at `plan.handover_point_m + [{receiver_sign_str}, 0, -0.02]`, then moves to `obj_pos + [0, 0, -0.025]`, closes and attaches within 0.055 m (`{receiver}_grasped = true`). After `gripper_close_delay_s` and `release_delay_s`, donor (`{donor}`) detaches/releases while receiver (`{receiver}`) holds within 0.07 m (`handover_completed = true`).
  - Place & settle: donor (`{donor}`) retracts to `plan.handover_point_m + [{donor_retract_str}, 0, 0.08]` while receiver (`{receiver}`) moves above `scene.target_m` `{list(scene.target_m)}` (`goal_center = scene.target_m + [0, 0, object_height/2 + release_height_m]`, pre-place at `goal_center - offset + [0, 0, 0.12]`), lowers to `goal_center - offset`, releases (`detach`), and settles by `[{receiver_settle_str}, 0, 0.12]`.
- `can_min_z_m >= -{config.DROP_MARGIN_M}` (not dropped below tabletop)
- `can_final_xy_error_m <= {config.CAN_TARGET_TOLERANCE_M}` m (within {config.CAN_TARGET_TOLERANCE_M} m of `scene.target_m` `{list(scene.target_m)}`)
- `can_final_speed_m_s <= {config.MAX_CAN_SPEED_AT_END_M_S}` m/s
- `max_arm_table_contact_force_n <= {config.MAX_ARM_TABLE_CONTACT_FORCE_N}` N
- `max_arm_obstacle_contact_force_n <= {config.MAX_ARM_OBSTACLE_CONTACT_FORCE_N}` N (arm against fitted obstacles such as laptop/screens)
- `max_arm_obstacle_penetration_m <= {config.MAX_ARM_OBSTACLE_PENETRATION_M}` m ({config.MAX_ARM_OBSTACLE_PENETRATION_M * 1000:.1f} mm deepest arm-obstacle penetration)
- `max_object_obstacle_penetration_m <= {config.MAX_OBJECT_OBSTACLE_PENETRATION_M}` m ({config.MAX_OBJECT_OBSTACLE_PENETRATION_M * 1000:.1f} mm deepest held-object penetration into an obstacle)

{valid_labels_note}

Instructions:
1. Inspect the attached rendered keyframe images and the measured `AttemptMetrics` / `failed_checks`.
2. In `diagnosis` (2 to 4 concise strings), explain the physical cause chain tied to specific keyframes and measured metrics (e.g. why grasp missed, why hand-over missed/dropped, or why placement missed `target_m`).
3. In `cited_keyframes` (1 to 4 strings), list the exact keyframe `label` values from the available list above that visually show the failure.
4. In `patch` (`PlanPatch`), provide concrete updated parameter values (within schema bounds: `grasp_height_offset_m` in [-0.05, 0.05], `approach_yaw_deg` in [-90, 90], `pregrasp_clearance_m` in [0.02, 0.20], `lift_height_m` in [0.03, 0.40], `handover_point_m` [x, y, z], `handover_yaw_offset_deg` in [-90, 90], `gripper_close_delay_s` in [0.0, 2.0], `release_delay_s` in [0.0, 2.0], `release_height_m` in [0.0, 0.15], `max_speed_m_s` in (0.02, 1.0]) that resolve the failure. Leave unchanged fields as `null`.

Return ONLY a single valid JSON object conforming strictly to this `CritiqueBody` JSON Schema:
{schema_json}
"""

    base_content = [{"type": "text", "text": prompt}, *keyframe_parts]
    content = list(base_content)
    total_wall_s = 0.0
    last_raw = ""
    last_err = ""
    responses: list[Any] = []

    for attempt_idx in range(config.ARGON_INVALID_RETRIES + 1):
        response, wall_s = _run_model_interaction(
            client,
            content=content,
            thinking_level=config.CRITIQUE_THINKING,
            max_output_tokens=config.CRITIQUE_MAX_OUTPUT_TOKENS,
            timeout_s=config.CRITIQUE_TIMEOUT_S,
        )
        responses.append(response)
        total_wall_s += wall_s
        last_raw = _extract_response_text(response)
        cleaned = _extract_json_text(last_raw)
        try:
            body = schemas.CritiqueBody.model_validate_json(cleaned)
            if available_labels:
                invalid_cited = [lbl for lbl in body.cited_keyframes if lbl not in available_labels]
                if invalid_cited:
                    raise ValueError(
                        f"cited_keyframes {invalid_cited} must be chosen from available keyframe labels {available_labels}"
                    )
            if all(v is None for v in body.patch.model_dump().values()):
                raise ValueError("PlanPatch must modify at least one parameter (all fields were null)")

            return schemas.Critique(
                body=body,
                interaction=_build_generate_content_ref(response),
                usage=_build_usage(responses, total_wall_s),
                mode="LIVE_ARGON",
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_err = str(exc)
            if attempt_idx < config.ARGON_INVALID_RETRIES:
                retry_prompt = (
                    f"Your previous response:\n{last_raw[:4000]}\n\n"
                    "failed `CritiqueBody` validation with the following error:\n"
                    f"{last_err}\n\n"
                    "Return ONLY the corrected JSON object matching `CritiqueBody`."
                )
                content = [*base_content, {"type": "text", "text": retry_prompt}]

    raise schemas.ArgonInvalidOutput(
        f"CritiqueBody validation failed after {config.ARGON_INVALID_RETRIES + 1} attempts: {last_err}",
        raw=last_raw,
        errors=last_err,
    )


# Convenience aliases matching earlier spec prose if referenced
scene_from_video = build_world
critique_and_patch = critique
