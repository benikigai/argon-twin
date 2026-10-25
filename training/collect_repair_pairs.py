"""Collect a bounded pilot of measured repair pairs; no LLM labels or invented trajectories."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from schemas import SceneSpec, HandoverPlan, judge
from sim import run_attempt


def main():
    out = ROOT / 'runs' / 'gemma-pilot'
    out.mkdir(parents=True, exist_ok=True)
    template = json.loads((ROOT / 'examples/desk_manual.json').read_text())
    source_hash = hashlib.sha256((ROOT / 'sim.py').read_bytes()).hexdigest()
    datasets = {'train': [], 'eval': []}
    source_dir = out / 'source'
    source_dir.mkdir(exist_ok=True)
    for name in ['sim.py', 'schemas.py', 'config.py']:
        (source_dir / name).write_bytes((ROOT / name).read_bytes())
    attempted = 0
    for group in range(16):
        rng = np.random.default_rng(group)
        body = json.loads(json.dumps(template))
        can = next(obj for obj in body['scene']['objects'] if obj['name'] == 'can')
        can['pos_m'][0] += float(rng.uniform(-.015, .015))
        can['pos_m'][1] += float(rng.uniform(-.015, .015))
        can['mass_kg'] = float(rng.uniform(.28, .4))
        good = HandoverPlan.model_validate(body['plan'])
        scene = SceneSpec.model_validate(body['scene'])
        baseline = run_attempt(scene, good, attempt=1, out_dir=out / f'scene-{group:02}/baseline', render=False)
        if not judge(baseline.metrics).passed:
            continue
        failures = [
            {'handover_point_m': (float(rng.uniform(.9, 1.3)), .7, .5)},
            {'handover_point_m': (-float(rng.uniform(.9, 1.3)), -.7, .5)},
            {'grasp_height_offset_m': -.05},
            {'grasp_height_offset_m': .05},
            {'handover_yaw_offset_deg': float(rng.choice([-85, 85]))},
            {'max_speed_m_s': .95},
        ]
        for variant, changes in enumerate(failures):
            attempted += 1
            bad = good.model_copy(update=changes)
            failed = run_attempt(scene, bad, attempt=1,
                out_dir=out / f'scene-{group:02}/failure-{variant}', render=False)
            verdict = judge(failed.metrics)
            if verdict.passed:
                continue
            repair = {key: getattr(good, key) for key in changes}
            prompt = {'scene': scene.model_dump(mode='json'), 'plan': bad.model_dump(mode='json'),
                      'metrics': failed.metrics.model_dump(mode='json', exclude={'wall_time_ms'}),
                      'failed_checks': verdict.failed_checks}
            row = {'case_id': f'scene-{group:02}-variant-{variant}', 'scene_group': group, 'source': 'measured_mujoco_repair', 'sim_sha256': source_hash,
                   'messages': [
                       {'role': 'system', 'content': 'Propose a minimal correction to the failed hand-over plan. Return only a flat JSON object matching PlanPatch, without any wrapper key or explanation. Allowed keys: grasp_height_offset_m, approach_yaw_deg, pregrasp_clearance_m, lift_height_m, handover_point_m, handover_yaw_offset_deg, gripper_close_delay_s, release_delay_s, release_height_m, max_speed_m_s. Include only changed fields. MuJoCo will independently verify the proposed plan.'},
                       {'role': 'user', 'content': json.dumps(prompt, separators=(',', ':'))},
                       {'role': 'assistant', 'content': json.dumps(repair, separators=(',', ':'))}],
                   'evidence': {'failed_artifact': f'scene-{group:02}/failure-{variant}/rollout.json',
                                'repaired_artifact': f'scene-{group:02}/baseline/rollout.json',
                                'failed_verdict': verdict.model_dump(),
                                'repaired_metrics': baseline.metrics.model_dump(mode='json'),
                                'repaired_passed': True}}
            datasets['eval' if group >= 12 else 'train'].append(row)
        print(f'scene={group} train={len(datasets["train"])} eval={len(datasets["eval"])}', flush=True)
    for split, rows in datasets.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'base_model': 'google/gemma-4-E4B-it', 'sim_sha256': source_hash,
              'attempted_corruptions': attempted, 'train_pairs': len(datasets['train']),
              'eval_pairs': len(datasets['eval']), 'split': 'held-out scene groups 12-15',
              'label_source': 'scripted correction with measured passing rollout; not Argon output',
              'scope': 'small synthetic-scene pilot, not a general robotics dataset',
              'fine_tuning_started': False}
    (out / 'dataset-receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
