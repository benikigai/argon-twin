import json
from pathlib import Path

import numpy as np

from schemas import HandoverPlan, RolloutLog, SceneSpec, judge
from sim import run_attempt, validate_scene

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    body = json.loads((ROOT / 'examples/desk_manual.json').read_text())
    return SceneSpec.model_validate(body['scene']), HandoverPlan.model_validate(body['plan'])


def test_real_aloha_handover_and_repeatability(tmp_path):
    scene, plan = inputs()
    assert validate_scene(scene) == []
    first = run_attempt(scene, plan, attempt=1, out_dir=tmp_path / 'first', render=False)
    second = run_attempt(scene, plan, attempt=2, out_dir=tmp_path / 'second', render=False)
    assert first.metrics.left_grasped and first.metrics.right_grasped
    assert first.metrics.handover_completed and first.metrics.grasp_assist_used
    assert judge(first.metrics).passed
    assert np.allclose(first.metrics.can_final_pos_m, second.metrics.can_final_pos_m, atol=1e-10)
    rollout = RolloutLog.model_validate_json((tmp_path / 'first/rollout.json').read_text())
    assert any(body.name == 'left/gripper_link' for body in rollout.bodies)
    assert any(body.name == 'right/gripper_link' for body in rollout.bodies)
    assert len(rollout.frames) > 100
    assert all(len(frame) == len(rollout.bodies) * 7 for frame in rollout.frames)
    assert rollout.frames[0] != rollout.frames[-1]


def test_unreachable_handover_cannot_pass(tmp_path):
    scene, plan = inputs()
    bad = plan.model_copy(update={'handover_point_m': (1.2, .8, .6)})
    result = run_attempt(scene, bad, attempt=1, out_dir=tmp_path, render=False)
    assert not result.metrics.handover_completed
    assert not judge(result.metrics).passed


def test_scene_reports_unreachable_target():
    scene, _ = inputs()
    scene = scene.model_copy(update={'target_m': (3., 0., 0.)})
    assert any('target' in issue for issue in validate_scene(scene))
