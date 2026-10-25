"""Separate the photographed Celsius can from the calibrated desk Gaussian asset.

The can is selected geometrically around its COLMAP-fitted cylinder. Output
means are local to the object base center; the visual and collision body can
therefore move together. No image synthesis or new training is involved.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CENTER_XY = np.array([0.34063638, 0.18702933])
RADIUS_M = 0.029
HEIGHT_M = 0.155
MASS_KG = 0.35  # Assumed full drink mass; not measured from the video.


def extract(out_dir: Path):
    source = ROOT / 'web/assets/desk.splat'
    raw = source.read_bytes()
    if len(raw) % 32:
        raise ValueError('Invalid 32-byte Gaussian record file')
    rows = np.frombuffer(raw, np.uint8).reshape(-1, 32)
    xyz = np.frombuffer(raw, '<f4').reshape(-1, 8)[:, :3]
    radial = np.linalg.norm(xyz[:, :2] - CENTER_XY, axis=1)
    selected = (radial < .047) & (xyz[:, 2] > -.012) & (xyz[:, 2] < .170)
    # Include the rendered rim/halo in removal so a ghost can does not stay behind.
    remove = (radial < .051) & (xyz[:, 2] > -.018) & (xyz[:, 2] < .178)
    if not 50 < selected.sum() < len(rows) / 4:
        raise ValueError(f'Unexpected can selection: {selected.sum()} of {len(rows)}')
    body_origin = np.r_[CENTER_XY, 0.0]
    moving = rows[selected].copy()
    local_xyz = np.frombuffer(moving, '<f4').reshape(-1, 8)[:, :3]
    local_xyz -= body_origin
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {'celsius-can.splat': moving.tobytes(), 'desk-without-celsius.splat': rows[~remove].tobytes()}
    for name, data in outputs.items():
        (out_dir / name).write_bytes(data)
    receipt = {
        'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_gaussians': len(rows), 'can_gaussians': int(selected.sum()),
        'background_gaussians': int((~remove).sum()), 'removed_background_gaussians': int(remove.sum()),
        'can_visual_base_origin_m': body_origin.tolist(), 'cylinder_base_m': [*CENTER_XY.tolist(), 0.0],
        'cylinder_radius_m': RADIUS_M, 'cylinder_height_m': HEIGHT_M,
        'assumed_mass_kg': MASS_KG, 'selection_radius_m': .047, 'removal_radius_m': .051,
        'note': 'Can center fitted to tracked Celsius-label points in registered image 0004.jpg. Radius padded from a 25.6 mm circle fit. Height and mass are approximate. Geometry mask may contain reconstruction floaters; visually inspect before promotion.',
        'outputs': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name,data in outputs.items()},
    }
    (out_dir / 'extraction.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    print(json.dumps(extract(ROOT / 'runs/celsius-variant/assets'), indent=2))
