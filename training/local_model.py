"""Local Gemma proposal transport. MuJoCo still decides whether a patch works."""
import json
import time

import httpx

from schemas import AttemptMetrics, HandoverPlan, PlanPatch, SceneSpec, judge


SYSTEM_PROMPT = 'Propose a minimal correction to the failed hand-over plan. Return only a flat JSON object matching PlanPatch, without any wrapper key or explanation. Allowed keys: grasp_height_offset_m, approach_yaw_deg, pregrasp_clearance_m, lift_height_m, handover_point_m, handover_yaw_offset_deg, gripper_close_delay_s, release_delay_s, release_height_m, max_speed_m_s. Include only changed fields. MuJoCo will independently verify the proposed plan.'


def propose_patch(scene: SceneSpec, plan: HandoverPlan, metrics: AttemptMetrics,
                  *, model: str = 'gemma4') -> tuple[PlanPatch, dict]:
    prompt = {'scene': scene.model_dump(mode='json'), 'plan': plan.model_dump(mode='json'),
              'metrics': metrics.model_dump(mode='json', exclude={'wall_time_ms'}),
              'failed_checks': judge(metrics).failed_checks}
    started = time.monotonic()
    response = httpx.post('http://127.0.0.1:11434/api/chat', timeout=60,
        json={'model': model, 'stream': False, 'think': False,
              'format': PlanPatch.model_json_schema(),
              'options': {'temperature': 0, 'num_ctx': 2048, 'num_predict': 256},
              'messages': [
                  {'role': 'system', 'content': SYSTEM_PROMPT},
                  {'role': 'user', 'content': json.dumps(prompt, separators=(',', ':'))}]})
    response.raise_for_status()
    data = response.json()
    patch = PlanPatch.model_validate_json(data['message']['content'])
    changes = patch.model_dump(exclude_none=True)
    if not changes:
        raise ValueError('Local model returned no proposed changes')
    HandoverPlan.model_validate({**plan.model_dump(), **changes})
    return patch, {'model': data['model'], 'provider': 'local_ollama',
                   'wall_time_s': time.monotonic() - started,
                   'eval_count': data.get('eval_count'),
                   'eval_duration_ns': data.get('eval_duration')}
