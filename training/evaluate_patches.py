"""Evaluate downloaded base/tuned model predictions with the actual local simulator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schemas import SceneSpec, HandoverPlan, PlanPatch, judge
from sim import run_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / 'gemma-evaluation')
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for index, row in enumerate(records):
        stats = summary.setdefault(row['model'], {'examples': 0, 'valid_patches': 0, 'passed': 0})
        stats['examples'] += 1
        try:
            raw = row['prediction'].strip()
            if raw.startswith('```'):
                raw = '\n'.join(raw.splitlines()[1:-1])
            patch = PlanPatch.model_validate_json(raw)
            prompt = json.loads(row['input'])
            plan = HandoverPlan.model_validate({**prompt['plan'],
                **patch.model_dump(exclude_none=True)})
            stats['valid_patches'] += 1
            result = run_attempt(SceneSpec.model_validate(prompt['scene']), plan,
                attempt=1, out_dir=out / f'{index:03}', render=False)
            verdict = judge(result.metrics)
            stats['passed'] += int(verdict.passed)
            row['verdict'] = verdict.model_dump()
        except (ValueError, TypeError) as exc:
            row['error'] = str(exc)
    from statistics import median
    for model_name, stats in summary.items():
        times = sorted(float(r['latency_s']) for r in records if r['model'] == model_name and 'latency_s' in r)
        if times:
            stats['median_inference_s'] = median(times)
            stats['p95_inference_s'] = times[min(len(times) - 1, int(len(times) * .95))]
    (out / 'predictions-verified.json').write_text(json.dumps(records, indent=2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
