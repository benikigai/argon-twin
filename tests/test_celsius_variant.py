"""The photographed right-side can must be picked by the right arm safely."""
import json
from pathlib import Path

from schemas import HandoverPlan, SceneSpec, judge
from sim import _Attempt, run_attempt

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    data = json.loads((ROOT / 'examples/desk_celsius.json').read_text())
    return SceneSpec.model_validate(data['scene']), HandoverPlan.model_validate(data['plan'])


def test_right_pickup_and_left_placement_are_physical(tmp_path, monkeypatch):
    scene, plan = inputs()
    attachments = []
    attach = _Attempt.attach

    def record_attach(self, side):
        attachments.append(side)
        return attach(self, side)

    monkeypatch.setattr(_Attempt, 'attach', record_attach)
    result = run_attempt(scene, plan, attempt=1, out_dir=tmp_path, render=False)
    assert scene.pickup_arm == 'right'
    assert attachments == ['right', 'left']
    assert result.metrics.left_grasped and result.metrics.right_grasped
    assert judge(result.metrics).passed
    assert result.metrics.max_arm_obstacle_contact_force_n == 0
    assert result.metrics.max_arm_obstacle_penetration_m == 0
    assert result.metrics.max_object_obstacle_penetration_m == 0
    assert result.metrics.can_final_xy_error_m < .01


def test_celsius_cannot_pass_through_a_screen(tmp_path):
    scene, plan = inputs()
    bad = plan.model_copy(update={'handover_point_m': (0, .21, .30)})
    result = run_attempt(scene, bad, attempt=1, out_dir=tmp_path, render=False)
    assert not judge(result.metrics).passed
    assert result.metrics.max_arm_obstacle_contact_force_n > 5
    assert result.metrics.max_object_obstacle_penetration_m > .002
