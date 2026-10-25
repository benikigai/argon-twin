"""Run with colab exec -s argon-twin-splat -f capture/train_colab.py.

Requires COLMAP, ffmpeg, torch, scipy, Pillow and gsplat==1.5.3.
Outputs remain private until the reconstructed table is cropped and reviewed.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/content/argon-twin')
VIDEO = Path('/content/IMG_1272.MOV')
START = time.monotonic()
ROOT.mkdir(exist_ok=True)


def run(args, timeout=300):
    print('RUN', ' '.join(map(str, args)), flush=True)
    log = ROOT / (str(args[1]).replace('/', '_') + '.log')
    with log.open('w') as out:
        result = subprocess.run(list(map(str, args)), stdout=out,
                                stderr=subprocess.STDOUT, timeout=timeout)
    print(log.read_text()[-2200:], flush=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.returncode}')


dataset = ROOT / 'undistorted'
txt = ROOT / 'colmap_txt'
if not (txt / 'images.txt').exists():
    frames = ROOT / 'frames'
    frames.mkdir(exist_ok=True)
    if not (frames / '0052.png').exists():
        run(['ffmpeg', '-y', '-i', VIDEO, '-vf', 'fps=5,scale=960:-2', '-pix_fmt', 'rgb24', frames / '%04d.png'])
    db = ROOT / 'database.db'
    if not db.exists():
        run(['colmap', 'feature_extractor', '--database_path', db, '--image_path', frames,
         '--ImageReader.camera_model', 'OPENCV', '--ImageReader.single_camera', '1',
         '--SiftExtraction.use_gpu', '0'])
    run(['colmap', 'sequential_matcher', '--database_path', db, '--SiftMatching.use_gpu', '0', '--SequentialMatching.overlap', '5'])
    sparse = ROOT / 'sparse'
    sparse.mkdir(exist_ok=True)
    run(['colmap', 'mapper', '--database_path', db, '--image_path', frames,
         '--output_path', sparse], timeout=420)
    models = [p for p in sparse.iterdir() if p.is_dir() and (p / 'images.bin').exists()]
    if not models:
        raise RuntimeError('COLMAP did not register a reconstruction')
    model = max(models, key=lambda p: (p / 'images.bin').stat().st_size)
    dataset = ROOT / 'undistorted'
    run(['colmap', 'image_undistorter', '--image_path', frames, '--input_path', model,
         '--output_path', dataset, '--output_type', 'COLMAP'])
    txt = ROOT / 'colmap_txt'
    txt.mkdir(exist_ok=True)
    run(['colmap', 'model_converter', '--input_path', dataset / 'sparse',
         '--output_path', txt, '--output_type', 'TXT'])

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
os.environ.setdefault('MAX_JOBS', '2')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda'
cameras = {}
for line in (txt / 'cameras.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    cid, kind, width, height, *params = line.split()
    if kind != 'PINHOLE':
        raise ValueError(f'Expected undistorted PINHOLE, got {kind}')
    fx, fy, cx, cy = map(float, params)
    cameras[int(cid)] = (int(width), int(height), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

lines = [line for line in (txt / 'images.txt').read_text().splitlines()
         if not line.startswith('#')]
views = []
for line in lines[::2]:
    if not line.strip():
        continue
    values = line.split()
    q = np.array(values[1:5], dtype=float)
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    w2c[:3, 3] = np.array(values[5:8], dtype=float)
    width, height, K = cameras[int(values[8])]
    ratio = 640 / max(width, height)
    size = (round(width * ratio), round(height * ratio))
    K = K.copy()
    K[0, :] *= size[0] / width
    K[1, :] *= size[1] / height
    pixels = np.asarray(Image.open(dataset / 'images' / values[9]).convert('RGB').resize(size), dtype=np.float32) / 255
    views.append((torch.tensor(w2c, dtype=torch.float32, device=device),
                  torch.tensor(K, dtype=torch.float32, device=device),
                  torch.tensor(pixels, device=device)))
if len(views) < 8:
    raise RuntimeError(f'Only {len(views)} registered views; reconstruction insufficient')

points, colors = [], []
for line in (txt / 'points3D.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    v = line.split()
    if float(v[7]) < 3.0 and len(v[8:]) >= 6:
        points.append(list(map(float, v[1:4])))
        colors.append(list(map(float, v[4:7])))
points = np.asarray(points, dtype=np.float32)
colors = np.asarray(colors, dtype=np.float32) / 255
if len(points) < 100:
    raise RuntimeError(f'Only {len(points)} reliable sparse points')
distances = cKDTree(points).query(points, k=4)[0][:, 1:]
scale = np.maximum(np.sqrt((distances ** 2).mean(axis=1)), 1e-5)
scene_scale = float(np.linalg.norm(np.std(points, axis=0)))
tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
params = torch.nn.ParameterDict({
    'means': torch.nn.Parameter(tensor(points)),
    'scales': torch.nn.Parameter(tensor(np.log(scale)[:, None].repeat(3, axis=1))),
    'quats': torch.nn.Parameter(tensor(np.tile([1, 0, 0, 0], (len(points), 1)))),
    'opacities': torch.nn.Parameter(torch.full((len(points),), -2.1972246, device=device)),
    'colors': torch.nn.Parameter(torch.logit(tensor(colors).clamp(.001, .999))),
})
lrs = {'means': 1.6e-4 * scene_scale, 'scales': .005, 'quats': .001, 'opacities': .05, 'colors': .01}
optimizers = {name: torch.optim.Adam([p], lr=lrs[name], eps=1e-15) for name, p in params.items()}
strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=1300, refine_every=100,
                           reset_every=3000, verbose=True)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state(scene_scale=scene_scale)
losses = []
train_start = time.monotonic()
print('TRAIN_START', json.dumps({'views': len(views), 'points': len(points), 'scene_scale': scene_scale}), flush=True)
for step in range(1500):
    view, K, pixels = views[np.random.randint(len(views))]
    height, width = pixels.shape[:2]
    render, alpha, info = rasterization(params['means'], params['quats'], params['scales'].exp(),
        params['opacities'].sigmoid(), params['colors'].sigmoid(), view[None], K[None],
        width, height, packed=False)
    strategy.step_pre_backward(params, optimizers, state, step, info)
    loss = (render[0] - pixels).abs().mean()
    loss.backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    strategy.step_post_backward(params, optimizers, state, step, info)
    losses.append(float(loss.detach()))
    if step % 100 == 0:
        print('TRAIN', step, losses[-1], len(params['means']), flush=True)
    if time.monotonic() - START > float(os.environ.get('ARGON_TRAIN_BUDGET_S', '1100')):
        print('TIME_BUDGET: exporting current trained result', flush=True)
        break

out = ROOT / 'output'
out.mkdir(exist_ok=True)
p = {name: value.detach().cpu().numpy() for name, value in params.items()}
rgb = 1 / (1 + np.exp(-p['colors']))
opacity = 1 / (1 + np.exp(-p['opacities']))
quat = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True).clip(1e-8)
with (out / 'desk-uncropped.splat').open('wb') as f:
    for i in range(len(p['means'])):
        f.write(np.asarray(p['means'][i], dtype='<f4').tobytes())
        f.write(np.asarray(np.exp(p['scales'][i]), dtype='<f4').tobytes())
        f.write(np.clip(np.r_[rgb[i], opacity[i]] * 255, 0, 255).astype('uint8').tobytes())
        f.write(np.clip(quat[i] * 128 + 128, 0, 255).astype('uint8').tobytes())

attrs = np.concatenate([p['means'], np.zeros_like(p['means']), (rgb - .5) / .28209479177387814,
                        p['opacities'][:, None], p['scales'], quat], axis=1).astype('<f4')
names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
         'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
header = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(attrs)) + '\n'
header += ''.join(f'property float {name}\n' for name in names) + 'end_header\n'
with (out / 'desk-uncropped.ply').open('wb') as f:
    f.write(header.encode())
    f.write(attrs.tobytes())
with torch.no_grad():
    view, K, pixels = views[0]
    result, _, _ = rasterization(params['means'], params['quats'], params['scales'].exp(),
        params['opacities'].sigmoid(), params['colors'].sigmoid(), view[None], K[None],
        pixels.shape[1], pixels.shape[0], packed=False)
    Image.fromarray((result[0].clamp(0, 1).cpu().numpy() * 255).astype('uint8')).save(out / 'training-view.png')
    Image.fromarray((pixels.cpu().numpy() * 255).astype('uint8')).save(out / 'reference-view.png')
report = {'input_sha256': hashlib.sha256(VIDEO.read_bytes()).hexdigest() if VIDEO.exists() else json.loads((ROOT / 'source.json').read_text())['video_sha256'],
          'gsplat_version': '1.5.3', 'registered_views': len(views), 'initial_points': len(points),
          'trained_gaussians': len(attrs), 'steps': len(losses), 'mean_initial_50_l1': float(np.mean(losses[:50])),
          'mean_final_50_l1': float(np.mean(losses[-50:])), 'training_seconds': time.monotonic() - train_start,
          'total_seconds': time.monotonic() - START, 'cropped_for_publication': False}
(out / 'report.json').write_text(json.dumps(report, indent=2))
run(['tar', '-czf', ROOT / 'splat-output.tar.gz', '-C', out, '.'])
print('COMPLETE', json.dumps(report), flush=True)
