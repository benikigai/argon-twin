"""Shared configuration for ArgonTwin.

Owner: integrator (Claude). Other modules import from here; they don't redefine
these values. Pass/fail thresholds live here and are applied only by
`schemas.judge`, never by model output.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_VERSION = "1.0.0"

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = REPO_ROOT / "capture"
RUNS_DIR = REPO_ROOT / "runs"
WEB_DIR = REPO_ROOT / "web"
WEB_ASSETS_DIR = WEB_DIR / "assets"
EXAMPLES_DIR = REPO_ROOT / "examples"

PRIMARY_VIDEO = CAPTURE_DIR / "IMG_1272.MOV"
SECONDARY_VIDEO = CAPTURE_DIR / "IMG_1273.MOV"
SPLAT_PATH = WEB_ASSETS_DIR / "desk.splat"
SPLAT_ALIGN_PATH = WEB_ASSETS_DIR / "align.json"

# --- Models (public IDs, not secrets) ----------------------------------------
GEMINI_MODEL = "gemini-4-argon"
GEMINI_AGENT = "antigravity-preview-09-2026"
GEMINI_FAST_MODEL = "gemini-3.8-flash"

# Thinking levels per call type.
WORLD_BUILD_THINKING = "high"
CRITIQUE_THINKING = "medium"

# Long decode for the one-pass world build (background call).
WORLD_BUILD_MAX_OUTPUT_TOKENS = 262_144
WORLD_BUILD_TIMEOUT_S = 20 * 60
CRITIQUE_MAX_OUTPUT_TOKENS = 32_768
CRITIQUE_TIMEOUT_S = 180
ARGON_INVALID_RETRIES = 1  # one retry with the validation error, then ARGON_INVALID

# --- Loop --------------------------------------------------------------------
MAX_ATTEMPTS = 4

# --- Deterministic pass/fail thresholds (applied by schemas.judge only) ------
CAN_TARGET_TOLERANCE_M = 0.03     # can's final center within 3 cm of target (x, y)
DROP_MARGIN_M = 0.01              # can lower than table top - margin at any time = dropped
MAX_CAN_SPEED_AT_END_M_S = 0.02   # can must be at rest at the end of the attempt
MAX_ARM_TABLE_CONTACT_FORCE_N = 5.0

# --- Server ------------------------------------------------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


class MissingSecret(RuntimeError):
    """Raised when a required secret is not available. There is no fallback key."""


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. The .env may be a 1Password-mounted pipe, so read it once."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def gemini_api_key() -> str:
    """Return the event Gemini key from the environment or the mounted .env.

    Never logs or prints the key. Raises MissingSecret instead of falling back.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = _read_dotenv(REPO_ROOT / ".env").get("GEMINI_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = key
    if not key:
        raise MissingSecret(
            "GEMINI_API_KEY is not set. Mount the 1Password Environment into the repo root as .env."
        )
    return key
