"""Client for the locally served Gemma 4 E4B repair model (base or fine-tuned adapter).

Owner: Claude (Gemma comparison). Talks only to the OpenAI-compatible llama-server on
loopback; never calls a cloud model and never falls back to one.

The request matches the serving folder's repair_client.py: same system prompt (read from
that file), same JSON schema (plan-patch.schema.json), temperature 0, max_tokens 192.
The two variants differ only in the per-request adapter scale:

    base       "lora": [{"id": 0, "scale": 0.0}]
    fine-tuned "lora": [{"id": 0, "scale": 1.0}]

The server's global adapter configuration is read for provenance and never changed.
Both requests also send "cache_prompt": false, so the second request can't reuse the
first one's prompt cache and response times stay comparable.

Every call returns a GemmaResult with a status. Nothing is retried or substituted:
"ok", "invalid_output", "timeout", "service_unavailable" or "http_error".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import ValidationError

from schemas import PlanPatch

SERVING_DIR = Path(os.environ.get(
    "ARGON_GEMMA_SERVING_DIR", Path.home() / "Models/argon-twin/gemma-4-e4b/local-serving"))
BASE_URL = os.environ.get("ARGON_GEMMA_URL", "http://127.0.0.1:8767")
MODEL_ID = "argon-twin-repair"
TEMPERATURE = 0
MAX_TOKENS = 192
TIMEOUT_S = 60.0

Variant = Literal["base", "tuned"]
ADAPTER_SCALE: dict[str, float] = {"base": 0.0, "tuned": 1.0}
Status = Literal["ok", "invalid_output", "timeout", "service_unavailable", "http_error"]

# Fields the training and evaluation payloads carried in "metrics". Wall-clock time and
# metrics added to the simulator later are left out so the input matches that format.
PAYLOAD_METRIC_FIELDS = (
    "can_final_pos_m", "can_final_xy_error_m", "can_min_z_m", "can_final_speed_m_s",
    "left_grasped", "right_grasped", "handover_completed", "max_arm_table_contact_force_n",
    "grasp_assist_used", "failure_phase", "sim_time_s",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def system_prompt() -> str:
    """The exact SYSTEM string from the serving folder's repair_client.py."""
    path = SERVING_DIR / "repair_client.py"
    spec = importlib.util.spec_from_file_location("argon_gemma_repair_client", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEM


def response_schema() -> dict:
    return json.loads((SERVING_DIR / "plan-patch.schema.json").read_text())


def build_request(payload: dict, variant: Variant) -> dict:
    """The full request body. Only lora[0].scale differs between variants."""
    if set(payload) != {"scene", "plan", "metrics", "failed_checks"}:
        raise ValueError("payload must have exactly scene, plan, metrics and failed_checks")
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "cache_prompt": False,
        "lora": [{"id": 0, "scale": ADAPTER_SCALE[variant]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "plan_patch", "strict": True, "schema": response_schema()}},
    }


def request_fingerprint(body: dict) -> str:
    """Hash of everything except the adapter scale, to prove both variants got the same request."""
    shared = {k: v for k, v in body.items() if k != "lora"}
    return _sha256(json.dumps(shared, sort_keys=True, separators=(",", ":")).encode())


def _get(path: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as res:
        return json.load(res)


def server_info() -> dict:
    """Read-only provenance: health, served models and the global adapter list."""
    info: dict[str, Any] = {"base_url": BASE_URL}
    for key, path in (("health", "/health"), ("models", "/v1/models"), ("lora_adapters", "/lora-adapters")):
        try:
            info[key] = _get(path)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            info[key] = None
            info[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _classify_network_error(exc: BaseException) -> Status:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    return "service_unavailable"


def propose(payload: dict, variant: Variant, *, timeout_s: float = TIMEOUT_S) -> dict:
    """One request to the local endpoint. Returns a JSON-safe result record; never raises for
    model or network failures, so the caller can show them as they are."""
    body = build_request(payload, variant)
    record: dict[str, Any] = {
        "variant": variant,
        "adapter_scale": ADAPTER_SCALE[variant],
        "model": MODEL_ID,
        "endpoint": BASE_URL + "/v1/chat/completions",
        "request_fingerprint": request_fingerprint(body),
        "request": body,
        "status": None,
        "error": None,
        "http_s": None,
        "response": None,
        "raw_content": None,
        "finish_reason": None,
        "patch": None,
    }
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as res:
            response = json.load(res)
        record["http_s"] = time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        record["http_s"] = time.perf_counter() - t0
        record["status"] = "http_error"
        record["error"] = f"HTTP {exc.code}: {exc.read()[:500].decode(errors='replace')}"
        return record
    except (urllib.error.URLError, OSError) as exc:
        record["http_s"] = time.perf_counter() - t0
        record["status"] = _classify_network_error(exc)
        record["error"] = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
        return record

    record["response"] = {k: response.get(k) for k in ("id", "model", "created", "usage", "timings")}
    try:
        choice = response["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        record["raw_content"] = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        record["status"] = "invalid_output"
        record["error"] = f"malformed response: {type(exc).__name__}: {exc}"
        return record

    if record["finish_reason"] != "stop":
        record["status"] = "invalid_output"
        record["error"] = f"incomplete generation (finish_reason={record['finish_reason']})"
        return record
    try:
        data = json.loads(record["raw_content"])
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")
        patch = PlanPatch.model_validate(data)  # the application's current contract
        if all(v is None for v in patch.model_dump().values()):
            raise ValueError("patch changes nothing")
    except (ValueError, ValidationError) as exc:
        record["status"] = "invalid_output"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["status"] = "ok"
    record["patch"] = patch.model_dump(mode="json", exclude_none=True)
    return record
