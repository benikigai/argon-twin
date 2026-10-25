"""Verify that a user-edited plan is isolated to its own measured run."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import config
import loop
from schemas import PlanPatch
from server import StartRun

ROOT = Path(__file__).resolve().parents[1]


def test_initial_patch_is_snapshotted_and_does_not_change_next_run(tmp_path, monkeypatch):
    source = ROOT / 'examples/desk_manual.json'
    original = source.read_bytes()
    monkeypatch.setattr(config, 'RUNS_DIR', tmp_path)
    phases = []
    broken = loop.run_loop(
        scene_path=source,
        initial_patch=PlanPatch(handover_point_m=(1.2, .8, .6)),
        critique=False,
        max_attempts=1,
        run_id='broken',
        on_phase=lambda phase, detail: phases.append((phase, detail)),
    )
    assert not broken.passed
    assert broken.attempts[0].plan.handover_point_m == (1.2, .8, .6)
    snapshot = json.loads((tmp_path / 'broken/input.json').read_text())
    assert snapshot['initial_patch'] == {'handover_point_m': [1.2, .8, .6]}
    assert snapshot['plan']['handover_point_m'] == [1.2, .8, .6]
    assert [phase for phase, _ in phases] == ['running', 'judged', 'done']
    assert phases[-1][1]['passed'] is False

    clean = loop.run_loop(scene_path=source, critique=False, max_attempts=1, run_id='clean')
    assert clean.passed
    assert clean.attempts[0].plan.handover_point_m == (0., -.05, .25)
    assert source.read_bytes() == original


@pytest.mark.parametrize('payload', [
    {'scene': 'examples/desk_manual.json', 'max_attempts': 0},
    {'scene': 'examples/desk_manual.json', 'max_attempts': 5},
    {'scene': 'examples/desk_manual.json', 'initial_patch': {'max_speed_m_s': 100}},
    {'scene': 'examples/desk_manual.json', 'initial_patch': {'handover_point_m': [0, 0]}},
    {'scene': 'examples/desk_manual.json', 'initial_patch': {'unknown_control': 1}},
    {'scene': 'examples/desk_manual.json', 'initial_patch': {'handover_point_m': [float('nan'), 0, .2]}},
    {'scene': 'examples/desk_manual.json', 'initial_patch': {'handover_point_m': [0, float('inf'), .2]}},
])
def test_invalid_live_controls_are_rejected_before_execution(payload):
    with pytest.raises(ValidationError):
        StartRun.model_validate(payload)


def test_known_live_run_is_readable_before_worker_writes_files(tmp_path, monkeypatch):
    from threading import Event
    import time
    from fastapi.testclient import TestClient
    import server

    entered, release = Event(), Event()

    def delayed_worker(**kwargs):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(config, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(server.loop, 'run_loop', delayed_worker)
    client = TestClient(server.app)
    response = client.post('/api/runs', json={'scene': 'examples/desk_manual.json'})
    assert response.status_code == 200
    run_id = response.json()['run_id']
    try:
        assert entered.wait(timeout=1)
        assert not (tmp_path / run_id).exists()
        response = client.get(f'/api/runs/{run_id}')
        assert response.status_code == 200
        assert response.json()['summary']['state'] == 'running'
        assert response.json()['manifest'] is None
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while server._lock.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        server._live.pop(run_id, None)


def test_health_never_reads_the_credential_pipe(monkeypatch):
    from fastapi.testclient import TestClient
    import server

    def forbidden_read():
        raise AssertionError('Liveness must not wait for 1Password')

    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(server.config, 'gemini_api_key', forbidden_read)
    response = TestClient(server.app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['gemini_key_configured'] is False
    assert isinstance(response.json()['gemini_key_mount_present'], bool)
