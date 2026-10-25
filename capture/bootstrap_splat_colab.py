"""Prepare a prebuilt CUDA runtime and train the already-calibrated desk capture.

Upload camera-data.tar.gz and train_colab.py to /content before running this.
This script allocates no runtime; the caller owns the approved Colab deadline.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tarfile
import time

STARTED = time.time()

ROOT = Path('/content/argon-twin')
VENV = Path('/content/argon-splat-venv')
WHEEL = 'https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu121-cp310-cp310-linux_x86_64.whl'


def run(args, timeout):
    print('SETUP', ' '.join(args), flush=True)
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    print((result.stdout + result.stderr)[-2400:], flush=True)
    result.check_returncode()


ROOT.mkdir(exist_ok=True)
with tarfile.open('/content/argon-camera-data.tar.gz') as archive:
    archive.extractall(ROOT, filter='data')
run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 120)
run([sys.executable, '-m', 'uv', 'venv', '--python', '3.10', str(VENV)], 180)
python = str(VENV / 'bin/python')
run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
     'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], 480)
run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
     'numpy==1.26.4', 'scipy', 'Pillow', 'ninja', 'jaxtyping', 'rich', 'packaging', 'setuptools==75.8.0', 'wheel'], 180)
run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
     '--no-deps', WHEEL], 180)
probe = "import torch,gsplat; print('CUDA_READY',torch.__version__,gsplat.__version__,torch.cuda.is_available()); assert torch.cuda.is_available()"
run([python, '-c', probe], 60)
(ROOT / 'runtime-plan.json').write_text(json.dumps({
    'python': '3.10', 'torch': '2.4.1+cu121', 'gsplat': '1.5.3+pt24cu121',
    'precompiled_wheel': WHEEL, 'input': '/content/argon-camera-data.tar.gz',
    'training_budget_s': 900,
}, indent=2))
environment = os.environ.copy()
deadline = float(environment.get('ARGON_EXPORT_DEADLINE_EPOCH', STARTED + 27 * 60))
remaining = deadline - time.time() - 180
if remaining < 60:
    raise RuntimeError('Not enough approved time remains to train and export')
environment['ARGON_TRAIN_BUDGET_S'] = str(min(900, remaining))
run_result = subprocess.run([python, '/content/train_colab.py'], env=environment, timeout=1100)
raise SystemExit(run_result.returncode)
