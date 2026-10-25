"""Prepare camera poses on the Mac; reserve Colab GPU time for Gaussian training."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import time

import pycolmap

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / 'capture/IMG_1272.MOV'
OUT = ROOT / 'runs/splat-local'


def main():
    started = time.monotonic()
    images = OUT / 'frames'
    images.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(VIDEO),
        '-vf', 'fps=5,scale=640:640:force_original_aspect_ratio=decrease',
        '-q:v', '2', str(images / '%04d.jpg')], check=True)
    database = OUT / 'database.db'
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 4
    extraction.sift.max_num_features = 4096
    reader = pycolmap.ImageReaderOptions(camera_model='OPENCV')
    print('EXTRACT_FEATURES', flush=True)
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 4
    pairing = pycolmap.SequentialPairingOptions(overlap=5)
    print('MATCH_VIDEO_FRAMES', flush=True)
    pycolmap.match_sequential(database, matching_options=matching,
        pairing_options=pairing, device=pycolmap.Device.cpu)
    sparse = OUT / 'sparse'
    sparse.mkdir(exist_ok=True)
    print('RECONSTRUCT_CAMERAS', flush=True)
    models = pycolmap.incremental_mapping(database, images, sparse)
    if not models:
        raise RuntimeError('No camera reconstruction')
    model_id, model = max(models.items(), key=lambda item: item[1].num_reg_images())
    if model.num_reg_images() < 8:
        raise RuntimeError('Fewer than eight registered views')
    data = OUT / 'undistorted'
    pycolmap.undistort_images(data, sparse / str(model_id), images,
        undistort_options=pycolmap.UndistortCameraOptions(max_image_size=640),
        num_threads=4, jpeg_quality=90)
    text = OUT / 'colmap_txt'
    text.mkdir(exist_ok=True)
    undistorted = pycolmap.Reconstruction(data / 'sparse')
    undistorted.write_text(text)
    report = {'video_sha256': hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
              'pycolmap': pycolmap.__version__, 'registered_views': model.num_reg_images(),
              'points': model.num_points3D(), 'elapsed_seconds': time.monotonic() - started,
              'trained_splat': False}
    (OUT / 'source.json').write_text(json.dumps(report, indent=2))
    with tarfile.open(OUT / 'camera-data.tar.gz', 'w:gz') as archive:
        archive.add(text, arcname='colmap_txt')
        archive.add(data / 'images', arcname='undistorted/images')
        archive.add(OUT / 'source.json', arcname='source.json')
    print('CAMERAS_READY', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
