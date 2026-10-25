"""Collision checks must reject a handover even when the can lands on target."""
import json
from pathlib import Path

import pytest

from schemas import HandoverPlan, SceneSpec, judge
from sim import run_attempt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fitted_scene():
    body = json.loads((ROOT / 'examples/desk_collision.json').read_text())
    return SceneSpec.model_validate(body['scene']), HandoverPlan.model_validate(body['plan'])


def test_fitted_surfaces_allow_clear_foreground_handover(fitted_scene, tmp_path):
    scene, plan = fitted_scene
    result = run_attempt(scene, plan, attempt=1, out_dir=tmp_path, render=False)
    metrics = result.metrics
    assert judge(metrics).passed
    assert judge(metrics).obstacle_contact_measured
    assert metrics.max_arm_obstacle_contact_force_n == 0
    assert metrics.max_arm_obstacle_penetration_m == 0
    assert metrics.max_object_obstacle_penetration_m == 0
    evidence = json.loads((tmp_path / 'collision-evidence.json').read_text())
    assert {'upper_screen', 'lower_screen', 'laptop_base', 'left_screen', 'left_base'} <= set(evidence['obstacles'])
    assert evidence['min_arm_obstacle_distance_m'] >= .02


def test_screen_collision_fails_even_when_can_reaches_target(fitted_scene, tmp_path):
    scene, plan = fitted_scene
    bad = plan.model_copy(update={'handover_point_m': (0, .21, .30)})
    result = run_attempt(scene, bad, attempt=1, out_dir=tmp_path, render=False)
    assert result.metrics.handover_completed
    assert result.metrics.can_final_xy_error_m < .03
    assert result.metrics.max_arm_obstacle_contact_force_n > 5
    verdict = judge(result.metrics)
    assert not verdict.passed
    assert any('obstacle' in reason for reason in verdict.failed_checks)
    events = json.loads((tmp_path / 'collision-evidence.json').read_text())['events']
    assert any('upper_screen' in event['bodies'] for event in events)


def test_unknown_old_collision_measurements_are_not_zero(fitted_scene, tmp_path):
    scene, plan = fitted_scene
    result = run_attempt(scene, plan, attempt=1, out_dir=tmp_path, render=False)
    old = result.metrics.model_copy(update={
        'max_arm_obstacle_contact_force_n': None,
        'max_arm_obstacle_penetration_m': None,
        'max_object_obstacle_penetration_m': None,
    })
    assert not judge(old).obstacle_contact_measured
    object_hit = result.metrics.model_copy(update={'max_object_obstacle_penetration_m': .01})
    assert not judge(object_hit).passed
