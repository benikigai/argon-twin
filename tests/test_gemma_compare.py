"""Local Gemma comparison: request contract and honest failure handling. Offline, no model server.

Owner: Claude (Gemma comparison).
"""

import io
import json
import socket
import urllib.error

import pytest

import gemma_local

PAYLOAD = {
    "scene": {"table_shape": "round"},
    "plan": {"handover_point_m": [1.2, 0.8, 0.6]},
    "metrics": {"handover_completed": False},
    "failed_checks": ["handover_completed is false"],
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def reply(content, finish_reason="stop"):
    body = {"id": "x", "model": gemma_local.MODEL_ID, "usage": {"prompt_tokens": 10},
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}
    return lambda req, timeout=None: FakeResponse(json.dumps(body).encode())


def test_variants_differ_only_in_adapter_scale():
    base = gemma_local.build_request(PAYLOAD, "base")
    tuned = gemma_local.build_request(PAYLOAD, "tuned")
    assert base["lora"] == [{"id": 0, "scale": 0.0}]
    assert tuned["lora"] == [{"id": 0, "scale": 1.0}]
    assert {k: v for k, v in base.items() if k != "lora"} == {k: v for k, v in tuned.items() if k != "lora"}
    assert gemma_local.request_fingerprint(base) == gemma_local.request_fingerprint(tuned)
    assert base["temperature"] == 0 and base["max_tokens"] == 192
    assert base["response_format"]["json_schema"]["schema"] == gemma_local.response_schema()
    assert base["messages"][0]["content"] == gemma_local.system_prompt()


def test_valid_patch_is_validated_against_current_contract(monkeypatch):
    monkeypatch.setattr(gemma_local.urllib.request, "urlopen", reply('{"grasp_height_offset_m": 0.0}'))
    result = gemma_local.propose(PAYLOAD, "tuned")
    assert result["status"] == "ok"
    assert result["patch"] == {"grasp_height_offset_m": 0.0}
    assert result["http_s"] is not None


@pytest.mark.parametrize("content,finish", [
    ('{"grasp_height_offset_m": 0.5}', "stop"),      # out of the contract's bounds
    ('{"not_a_field": 1}', "stop"),                  # unknown field
    ("{}", "stop"),                                  # changes nothing
    ('{"grasp_height_offset_m": null}', "stop"),     # changes nothing
    ("not json", "stop"),
    ('{"grasp_height_offset_m": 0.0', "length"),     # cut off at max_tokens
])
def test_invalid_output_is_reported_not_applied(monkeypatch, content, finish):
    monkeypatch.setattr(gemma_local.urllib.request, "urlopen", reply(content, finish))
    result = gemma_local.propose(PAYLOAD, "base")
    assert result["status"] == "invalid_output"
    assert result["patch"] is None
    assert result["raw_content"] == content


def _raise(exc):
    def fake(req, timeout=None):
        raise exc
    return fake


@pytest.mark.parametrize("exc,status", [
    (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), "service_unavailable"),
    (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
    (TimeoutError("timed out"), "timeout"),
    (urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"server error")), "http_error"),
])
def test_network_failures_are_reported_without_fallback(monkeypatch, exc, status):
    monkeypatch.setattr(gemma_local.urllib.request, "urlopen", _raise(exc))
    result = gemma_local.propose(PAYLOAD, "tuned")
    assert result["status"] == status
    assert result["patch"] is None
    assert result["endpoint"].startswith(gemma_local.BASE_URL)
