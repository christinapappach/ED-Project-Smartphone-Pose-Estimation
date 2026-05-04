"""
MeTRAbs Pose Estimation Server — Hugging Face Space (BATCH-OF-9 variant)

Same model, same per-frame logic as app.py. The only thing this file changes
is the upload endpoint: it accepts up to 9 videos, 9 intrinsics JSON files,
and 9 LiDAR depth .bin files in a single request, processed sequentially on
the GPU. Outputs are written as output_video_{i}.mp4 and output_file_{i}.csv
for i = 1..9.

Inputs (multipart form):
    files:             1..9 .mp4 video files                      (required)
    intrinsics_files:  0..9 .json intrinsics files                (optional, paired by index)
    depth_files:       0..9 .bin LiDAR depth files                (optional, paired by index)
    ground_truth_files:0..9 .csv ground-truth files               (optional, paired by index)

Pairing is positional: intrinsics_files[i] / depth_files[i] / ground_truth_files[i]
go with files[i]. Missing optional entries fall back to "no intrinsics / no LiDAR / no GT"
for that video, matching the single-file behavior in app.py.
"""
import os
import ssl
import json
import csv
import shutil
import subprocess
import threading
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import time as _now
from typing import Optional

import certifi
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _Stub:
        def GPU(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _Stub()

# `@spaces.GPU(duration=...)` is meant for ZeroGPU Spaces — it allocates a
# time-limited GPU slice per call. On a *dedicated* GPU (t4-small etc.) the
# GPU is yours full-time and the decorator can actually slow you down, so we
# only enable it when we know we're on ZeroGPU. HF sets SPACES_ZERO_GPU=true
# in that case.
ON_ZERO_GPU = HAS_SPACES and os.environ.get('SPACES_ZERO_GPU', '').lower() == 'true'
if ON_ZERO_GPU:
    _gpu_call = spaces.GPU(duration=1500)
else:
    def _gpu_call(fn):
        return fn

# ─────────────────────────── Live log buffer ───────────────────────────
_LOG_LINES: deque[str] = deque(maxlen=1000)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    _LOG_LINES.append(line)


# ─────────────────────────── SSL ───────────────────────────
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

# ─────────────────────────── Config ───────────────────────────
MODEL_TYPE = 'metrabs_mob3l_y4t'
SKELETON = 'smpl_24'
RESIZE_FACTOR = 1.0
FRAME_SKIP = 1
CACHE_DIR = os.environ.get('METRABS_CACHE', './metrabs_models')

# Hard cap on the batch size. Keep it at 9 so a misconfigured client can't
# slurp arbitrary GPU time per request.
MAX_BATCH = 9

SMPL24_NAMES = [
    'Pelvis', 'Left_Hip', 'Right_Hip', 'Spine1', 'Left_Knee', 'Right_Knee',
    'Spine2', 'Left_Ankle', 'Right_Ankle', 'Spine3', 'Left_Foot', 'Right_Foot',
    'Neck', 'Left_Shoulder', 'Right_Shoulder', 'Head', 'Left_Upper_Arm',
    'Right_Upper_Arm', 'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
    'Left_Hand', 'Right_Hand',
]

_UNIT_FACTORS = (
    ('mm', 1.0),
    ('meters', 0.001),
    ('centimeters', 0.1),
    ('inches', 0.0393701),
    ('feet', 0.00328084),
)

_DEFAULT_ANKLE_JOINTS = ['Left_Ankle', 'Right_Ankle']

BASE_DIR = Path('/tmp/metrabs-server')
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'outputs'
ZIP_EXTRACT_FOLDER = BASE_DIR / 'zip_extracted'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
ZIP_EXTRACT_FOLDER.mkdir(parents=True, exist_ok=True)

# Path the zip-runner endpoint looks at by default. The HF Space puts files
# pushed to its repo into the working dir, so a zip uploaded via HF web UI
# (e.g. "HUGGING FACE VIDEOS.zip") will be sitting right there.
DEFAULT_ZIP_NAME = 'HUGGING FACE VIDEOS.zip'

# Shared state for the background batch runner. The /run_zip endpoint kicks
# off processing in a thread and returns immediately so the HTTP connection
# can close — running 9 videos sync would blow past every reverse-proxy
# timeout between us and the user's browser.
_BATCH_LOCK = threading.Lock()
_BATCH_STATE: dict = {
    'state': 'idle',          # idle | running | done | error
    'started_at': None,
    'finished_at': None,
    'zip': None,
    'total': 0,
    'completed': 0,
    'current': None,          # 'VIDn' the worker is on right now
    'results': [],            # one entry per processed VID
    'error': None,
}

# ─────────────────────────── Load model once at startup ───────────────────────────
from huggingface_hub import snapshot_download
MODEL_REPO = 'EngDesFAU26-SmartphonePose/metrabs-eff2l-y4-384px'
_log(f'Downloading MeTRAbs model from HF hub: {MODEL_REPO}')
_model_path = snapshot_download(repo_id=MODEL_REPO, cache_dir=CACHE_DIR)
if not os.path.exists(os.path.join(_model_path, 'saved_model.pb')):
    raise RuntimeError(f'saved_model.pb not found in {_model_path}')
_log(f'Loading model from {_model_path}')

# Diagnostic: log what TF sees for hardware. If this Space is on t4-small
# but TF reports zero GPUs, every frame will run on CPU and we'll see
# ~10× slowdowns — that's the symptom we're hunting.
_gpus = tf.config.list_physical_devices('GPU')
_log(f'TF physical GPUs: {_gpus} (count={len(_gpus)})')
_log(f'spaces module loaded: {HAS_SPACES}, on ZeroGPU: {ON_ZERO_GPU}, '
     f'SPACES_ZERO_GPU env: {os.environ.get("SPACES_ZERO_GPU", "<unset>")}')

MODEL = tf.saved_model.load(_model_path)
EDGES = MODEL.per_skeleton_joint_edges[SKELETON].numpy().tolist()
_log(f'Model loaded. Skeleton {SKELETON}, {len(EDGES)} edges')


# ─────────────────────────── Helpers ───────────────────────────
def _delete_all_files(folder: Path) -> None:
    for f in folder.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass


def _safe_name(filename: str, fallback: str) -> str:
    """Strip any directory parts from a client-supplied filename so it can't
    escape UPLOAD_FOLDER. Empty / suspicious names fall back to `fallback`."""
    if not filename:
        return fallback
    base = Path(filename).name
    if not base or base in ('.', '..'):
        return fallback
    return base


def _draw_skeletons(frame, poses2d, poses3d, edges, w, h) -> None:
    """Bones (white), joints (yellow), per-joint depth label in mm (green). Single-person."""
    if poses2d is None or len(poses2d) == 0:
        return
    pose2d = poses2d[0].astype(int)
    pose3d = poses3d[0] if poses3d is not None and len(poses3d) > 0 else None

    for i, j in edges:
        if i < len(pose2d) and j < len(pose2d):
            x1, y1 = pose2d[i]
            x2, y2 = pose2d[j]
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    for jid, (x, y) in enumerate(pose2d):
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)
        if pose3d is not None and jid < len(pose3d):
            z_mm = pose3d[jid, 2]
            cv2.putText(frame, f'{z_mm:.0f}mm', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


def analyze(output_csv, ground_truth_csv=None, used_intrinsics=False, joints=None):
    """Compare the model output CSV against a ground-truth CSV on ankle joints.
    Identical to app.py's analyze(): rewrites the err columns in place and
    returns a per-joint summary dict.
    """
    if joints is None:
        joints = _DEFAULT_ANKLE_JOINTS

    if not ground_truth_csv or not os.path.exists(ground_truth_csv):
        return {
            'status': 'skipped',
            'reason': 'no ground truth CSV provided',
            'used_intrinsics': used_intrinsics,
            'joints_analyzed': joints,
        }

    with open(output_csv, newline='') as f:
        all_rows = list(csv.reader(f))
    if len(all_rows) < 3:
        return {
            'status': 'no_data',
            'reason': 'output CSV has no data rows',
            'used_intrinsics': used_intrinsics,
            'joints_analyzed': joints,
        }
    header_row1, header_row2, data_rows = all_rows[0], all_rows[1], all_rows[2:]
    n_cols = len(header_row2)
    MM_X, MM_Y, MM_Z = 3, 4, 5
    ERR_X, ERR_Y, ERR_Z, EUCL = n_cols - 4, n_cols - 3, n_cols - 2, n_cols - 1

    df_gt = pd.read_csv(ground_truth_csv)
    df_gt = df_gt[df_gt['joint_name'].isin(joints)].copy()
    gt_lookup = {}
    for _, gt_row in df_gt.iterrows():
        try:
            key = (str(int(gt_row['frame'])), str(int(gt_row['joint_idx'])), str(gt_row['joint_name']))
            gt_lookup[key] = (float(gt_row['x_mm']), float(gt_row['y_mm']), float(gt_row['z_mm']))
        except (ValueError, KeyError):
            continue

    per_joint_errs = {j: {'x': [], 'y': [], 'z': [], 'eucl': []} for j in joints}
    rows_compared = 0
    for row in data_rows:
        if len(row) < n_cols:
            continue
        jname = row[2]
        if jname not in joints:
            continue
        key = (row[0], row[1], jname)
        if key not in gt_lookup:
            continue
        gx, gy, gz = gt_lookup[key]
        try:
            px = float(row[MM_X])
            py = float(row[MM_Y])
            pz = float(row[MM_Z])
        except (ValueError, IndexError):
            continue
        ex = abs(px - gx)
        ey = abs(py - gy)
        ez = abs(pz - gz)
        eucl = ((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2) ** 0.5
        row[ERR_X] = f'{ex:.4f}'
        row[ERR_Y] = f'{ey:.4f}'
        row[ERR_Z] = f'{ez:.4f}'
        row[EUCL] = f'{eucl:.4f}'
        per_joint_errs[jname]['x'].append(ex)
        per_joint_errs[jname]['y'].append(ey)
        per_joint_errs[jname]['z'].append(ez)
        per_joint_errs[jname]['eucl'].append(eucl)
        rows_compared += 1

    if rows_compared == 0:
        return {
            'status': 'no_match',
            'reason': 'no overlapping (frame, joint_name) rows between output and GT',
            'used_intrinsics': used_intrinsics,
            'joints_analyzed': joints,
        }

    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header_row1)
        w.writerow(header_row2)
        for row in data_rows:
            w.writerow(row)

    per_joint = {}
    all_eucl = []
    for jname in joints:
        e = per_joint_errs[jname]
        if not e['eucl']:
            continue
        per_joint[jname] = {
            'mean_abs_err_x_mm': round(sum(e['x']) / len(e['x']), 2),
            'mean_abs_err_y_mm': round(sum(e['y']) / len(e['y']), 2),
            'mean_abs_err_z_mm': round(sum(e['z']) / len(e['z']), 2),
            'mean_euclidean_distance_error_mm': round(sum(e['eucl']) / len(e['eucl']), 2),
        }
        all_eucl.extend(e['eucl'])

    return {
        'status': 'ok',
        'used_intrinsics': used_intrinsics,
        'joints_analyzed': joints,
        'rows_compared': rows_compared,
        'overall_mean_euclidean_distance_error_mm': round(sum(all_eucl) / len(all_eucl), 2) if all_eucl else 0,
        'per_joint': per_joint,
    }


def _read_depth_file(depth_path):
    """Parse the iOS-side depth.bin file into a [n_frames, height, width] float32
    array of metric depth in METERS. Identical to app.py.
    """
    if not depth_path or not os.path.exists(depth_path):
        return None
    try:
        with open(depth_path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                _log('Depth file: header too short')
                return None
            w = int.from_bytes(header[0:4], 'little')
            h = int.from_bytes(header[4:8], 'little')
            n = int.from_bytes(header[8:12], 'little')
            if w == 0 or h == 0 or n == 0:
                _log(f'Depth file: empty (w={w} h={h} n={n})')
                return None
            data = np.frombuffer(f.read(), dtype=np.float32)
            expected = n * h * w
            if data.size != expected:
                _log(f'Depth file size mismatch: got {data.size} floats, expected {expected}')
                return None
            arr = data.reshape(n, h, w)
            _log(f'Depth file loaded: {n} frames at {w}x{h} (≈{arr.nbytes / 1e6:.1f} MB)')
            return arr
    except Exception as e:
        _log(f'Depth file read error: {e}')
        return None


def _build_two_row_header() -> tuple[list[str], list[str]]:
    row1 = ['', '', '']
    for unit, _ in _UNIT_FACTORS:
        row1 += [unit, '', '']
    row1 += ['error_mm', '', '', '']

    row2 = ['frame', 'joint_idx', 'joint_name']
    for _ in _UNIT_FACTORS:
        row2 += ['x', 'y', 'z']
    row2 += ['err_x', 'err_y', 'err_z', 'euclidean_err']
    return row1, row2


def _parse_intrinsics_from_text(intrinsics_text: Optional[str]) -> Optional[np.ndarray]:
    """Parse a JSON intrinsics string of the iOS form
        {"camera_intrinsics": {"intrinsic_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]}}
    into a 3x3 float32 matrix. Returns None on any failure.
    """
    if not intrinsics_text:
        return None
    try:
        data = json.loads(intrinsics_text)
        matrix = data.get('camera_intrinsics', {}).get('intrinsic_matrix')
        if matrix:
            arr = np.array(matrix, dtype=np.float32)
            _log(f'Intrinsics loaded: fx={arr[0,0]:.1f}, fy={arr[1,1]:.1f}')
            return arr
    except Exception as e:
        _log(f'Intrinsics JSON parse error: {e}')
    return None


@_gpu_call
def _process_video_gpu(input_path: str,
                       video_out: str,
                       csv_out: str,
                       intrinsic_matrix_np: np.ndarray | None,
                       depth_data_path: str | None = None) -> dict:
    """Run MeTRAbs once per frame with intrinsics if provided (else default FOV).
    LiDAR depth fusion (when both depth.bin and intrinsics are provided) is the
    same 5x5-median + 1m-sanity-check approach as app.py.
    """
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(width * RESIZE_FACTOR)
    out_h = int(height * RESIZE_FACTOR)
    out_fps = fps / FRAME_SKIP if FRAME_SKIP > 0 else fps
    _log(f'Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height} → model input {out_w}x{out_h}')

    intrinsics_tensor = None
    if intrinsic_matrix_np is not None:
        scaled = intrinsic_matrix_np.copy()
        scaled[0, :] *= RESIZE_FACTOR
        scaled[1, :] *= RESIZE_FACTOR
        intrinsics_tensor = tf.constant(scaled, dtype=tf.float32)
        _log(f'Using intrinsics (scaled by {RESIZE_FACTOR}):\n{scaled}')
    else:
        _log('No intrinsics provided — falling back to MeTRAbs default FOV (depth scale will be off).')

    depth_data = _read_depth_file(depth_data_path) if depth_data_path else None
    use_lidar = depth_data is not None and intrinsics_tensor is not None
    if depth_data_path and not use_lidar:
        if depth_data is None:
            _log('Depth file present but failed to load — falling back to MeTRAbs Z.')
        elif intrinsics_tensor is None:
            _log('Depth file present but no intrinsics — LiDAR fusion needs both. Skipping.')

    # Each video gets its own scratch temp file so a failure on one doesn't
    # leave the others stuck on a half-written remux input.
    temp_out = str(Path(video_out).with_suffix('.tmp.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out, fourcc, out_fps, (out_w, out_h))

    header_row1, header_row2 = _build_two_row_header()

    detected = 0
    with open(csv_out, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header_row1)
        writer.writerow(header_row2)

        source_idx = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if FRAME_SKIP > 1 and source_idx % FRAME_SKIP != 0:
                source_idx += 1
                continue
            processed += 1

            if RESIZE_FACTOR != 1.0:
                frame_proc = cv2.resize(frame, (out_w, out_h))
            else:
                frame_proc = frame
            rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            image_tensor = tf.convert_to_tensor(rgb)

            if intrinsics_tensor is not None:
                pred = MODEL.detect_poses(
                    image_tensor, skeleton=SKELETON,
                    intrinsic_matrix=intrinsics_tensor,
                    detector_threshold=0.1,
                )
            else:
                pred = MODEL.detect_poses(
                    image_tensor, skeleton=SKELETON,
                    detector_threshold=0.1,
                )

            poses2d = pred.get('poses2d')
            poses3d = pred.get('poses3d')
            if hasattr(poses2d, 'numpy'):
                poses2d = poses2d.numpy()
            if hasattr(poses3d, 'numpy'):
                poses3d = poses3d.numpy()

            pose = poses3d[0] if poses3d is not None and len(poses3d) > 0 else None

            if pose is not None and use_lidar and source_idx < len(depth_data):
                pose2d_first = poses2d[0] if poses2d is not None and len(poses2d) > 0 else None
                if pose2d_first is not None:
                    depth_frame = depth_data[source_idx]
                    dh, dw = depth_frame.shape
                    pose = pose.copy()
                    HALF = 2
                    SANITY_MM = 1000.0
                    for jid in range(len(pose)):
                        px = float(pose2d_first[jid][0])
                        py = float(pose2d_first[jid][1])
                        dx = int(px * dw / out_w)
                        dy = int(py * dh / out_h)
                        if not (0 <= dx < dw and 0 <= dy < dh):
                            continue
                        y_lo = max(0, dy - HALF); y_hi = min(dh, dy + HALF + 1)
                        x_lo = max(0, dx - HALF); x_hi = min(dw, dx + HALF + 1)
                        patch = depth_frame[y_lo:y_hi, x_lo:x_hi]
                        valid = patch[patch > 0]
                        if valid.size == 0:
                            continue
                        z_mm = float(np.median(valid)) * 1000.0
                        if abs(z_mm - float(pose[jid][2])) > SANITY_MM:
                            continue
                        pose[jid][2] = z_mm

            if pose is not None:
                detected += 1
                for jid, (x, y, z) in enumerate(pose):
                    name = SMPL24_NAMES[jid] if jid < len(SMPL24_NAMES) else f'joint_{jid}'
                    fx, fy, fz = float(x), float(y), float(z)
                    row = [source_idx, jid, name]
                    for _, factor in _UNIT_FACTORS:
                        row += [fx * factor, fy * factor, fz * factor]
                    row += ['', '', '', '']
                    writer.writerow(row)

            _draw_skeletons(frame_proc, poses2d, poses3d, EDGES, out_w, out_h)
            out.write(frame_proc)

            if processed % 25 == 0:
                _log(f'Processed {processed}/{total_frames} frames ({detected} w/ detections)')

            source_idx += 1

    cap.release()
    out.release()

    subprocess.run(
        ['ffmpeg', '-i', temp_out, '-vcodec', 'libx264',
         '-crf', '30', '-preset', 'fast', '-y', str(video_out)],
        check=True,
    )
    try:
        os.remove(temp_out)
    except OSError:
        pass

    return {'detected': detected, 'used_lidar': bool(use_lidar)}


# ─────────────────────────── FastAPI app ───────────────────────────
api = FastAPI(title='MeTRAbs Pose Estimation Server (batch-of-9)')
api.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
api.mount('/outputs', StaticFiles(directory=str(OUTPUT_FOLDER)), name='outputs')


@api.get('/status')
def status():
    return {
        'message': 'MeTRAbs HF Space running (batch-of-9)',
        'endpoint': '/upload',
        'ui': '/',
        'skeleton': SKELETON,
        'resize_factor': RESIZE_FACTOR,
        'frame_skip': FRAME_SKIP,
        'max_batch': MAX_BATCH,
    }


@api.get('/logs')
def logs():
    return {'lines': list(_LOG_LINES)}


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, 'wb') as f:
        shutil.copyfileobj(upload.file, f)
    upload.file.close()


def _pair_at(seq, i):
    """Return seq[i] if it exists and has a non-empty filename, else None."""
    if i >= len(seq):
        return None
    item = seq[i]
    if item is None:
        return None
    if not getattr(item, 'filename', None):
        return None
    return item


@api.post('/upload')
async def upload_videos(
    files: list[UploadFile] = File(...),
    intrinsics_files: list[UploadFile] = File(default=[]),
    depth_files: list[UploadFile] = File(default=[]),
    ground_truth_files: list[UploadFile] = File(default=[]),
):
    _log('BATCH UPLOAD REQUEST RECEIVED')

    # Sanity-check the videos list. Optional sidecar lists can be shorter
    # (or empty) — they're paired by index, with missing entries treated as
    # "no intrinsics / no depth / no GT" for that video.
    if not files:
        raise HTTPException(400, 'No video files provided')
    files = [f for f in files if f and f.filename]
    if not files:
        raise HTTPException(400, 'No video files provided (all entries empty)')
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f'Too many videos: {len(files)} > {MAX_BATCH}')
    for vf in files:
        if Path(vf.filename).suffix.lower() != '.mp4':
            raise HTTPException(400, f'Only .mp4 files allowed, got {vf.filename}')

    _log(f'Batch size: {len(files)} videos, '
         f'{sum(1 for x in intrinsics_files if x and x.filename)} intrinsics, '
         f'{sum(1 for x in depth_files if x and x.filename)} depth, '
         f'{sum(1 for x in ground_truth_files if x and x.filename)} GT csvs')

    # Wipe last batch's artifacts before accepting the new one. Same policy as
    # app.py — fixed output URLs, each new request overwrites them.
    _delete_all_files(UPLOAD_FOLDER)
    _delete_all_files(OUTPUT_FOLDER)

    results = []
    for i, vf in enumerate(files):
        idx = i + 1
        _log(f'─── video {idx}/{len(files)}: {vf.filename} ───')

        # Per-video upload paths. Use indexed names so different uploads with
        # the same filename don't clobber each other inside UPLOAD_FOLDER.
        safe_video_name = _safe_name(vf.filename, f'input_{idx}.mp4')
        input_path = UPLOAD_FOLDER / f'{idx:02d}_{safe_video_name}'
        await _save_upload(vf, input_path)
        _log(f'Saved video: {input_path.name}')

        # Optional intrinsics JSON (file). Read its contents and parse via the
        # same helper as app.py uses for the inline JSON string.
        intrinsic_np = None
        intr = _pair_at(intrinsics_files, i)
        if intr is not None:
            intr_path = UPLOAD_FOLDER / f'{idx:02d}_intrinsics.json'
            await _save_upload(intr, intr_path)
            try:
                intrinsic_np = _parse_intrinsics_from_text(intr_path.read_text())
            except Exception as e:
                _log(f'  Intrinsics file read error: {e}')

        # Optional LiDAR depth.
        depth_path = None
        dep = _pair_at(depth_files, i)
        if dep is not None:
            depth_path = UPLOAD_FOLDER / f'{idx:02d}_depth.bin'
            await _save_upload(dep, depth_path)
            _log(f'  Saved depth: {depth_path.name} ({depth_path.stat().st_size} bytes)')

        # Optional ground truth.
        gt_path = None
        gt = _pair_at(ground_truth_files, i)
        if gt is not None:
            gt_path = UPLOAD_FOLDER / f'{idx:02d}_ground_truth.csv'
            await _save_upload(gt, gt_path)
            _log(f'  Saved GT: {gt_path.name}')

        output_video = OUTPUT_FOLDER / f'output_video_{idx}.mp4'
        output_csv = OUTPUT_FOLDER / f'output_file_{idx}.csv'

        per_video = {
            'index': idx,
            'input_file': input_path.name,
            'output_file': output_video.name,
            'output_url': f'/outputs/{output_video.name}',
            'csv_file': output_csv.name,
            'csv_url': f'/outputs/{output_csv.name}',
            'used_intrinsics': intrinsic_np is not None,
            'used_ground_truth': gt_path is not None,
        }

        try:
            t0 = _now()
            gpu_result = _process_video_gpu(
                str(input_path), str(output_video), str(output_csv),
                intrinsic_np,
                depth_data_path=str(depth_path) if depth_path else None,
            )
            elapsed = _now() - t0
            per_video.update({
                'status': 'ok',
                'frames_with_detections': gpu_result.get('detected', 0),
                'used_lidar_depth': gpu_result.get('used_lidar', False),
                'processing_time_seconds': round(elapsed, 1),
            })
            _log(f'Done video {idx} in {elapsed:.1f}s, '
                 f"{gpu_result.get('detected', 0)} frames with detections, "
                 f"lidar_used={gpu_result.get('used_lidar', False)}")
        except Exception as e:
            # Skip GT analysis on failure — there's nothing to compare against.
            per_video.update({'status': 'error', 'error': f'Processing failed: {e}'})
            _log(f'Video {idx} FAILED: {e}')
            results.append(per_video)
            continue

        if not output_video.exists() or not output_csv.exists():
            per_video.update({'status': 'error', 'error': 'Output files were not created'})
            results.append(per_video)
            continue

        try:
            per_video['error_analysis'] = analyze(
                str(output_csv),
                ground_truth_csv=str(gt_path) if gt_path else None,
                used_intrinsics=intrinsic_np is not None,
            )
        except Exception as e:
            per_video['error_analysis'] = {'error': f'analyze failed: {e}'}

        results.append(per_video)

    succeeded = sum(1 for r in results if r.get('status') == 'ok')
    return {
        'message': f'Batch processed: {succeeded}/{len(results)} succeeded',
        'batch_size': len(results),
        'succeeded': succeeded,
        'results': results,
    }


# ─────────────────────────── Zip-on-disk batch path ───────────────────────────
# This path exists because pushing 1.4 GB through a single multipart /upload
# request hits HF's reverse-proxy size cap and connection-write timeouts.
# Instead the user pushes the zip into the Space repo (HF Hub supports LFS
# transparently), and we extract + process from local disk — no HTTP body
# size involved.

def _resolve_zip_path(name: str) -> Path:
    """Find `name` in a few sensible places: as given (absolute), the cwd,
    and the parent of cwd. Returns the first that exists, else raises."""
    candidates = []
    p = Path(name)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / name)
        candidates.append(Path.cwd().parent / name)
        candidates.append(Path('/home/user/app') / name)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'Zip not found. Looked in: {[str(c) for c in candidates]}')


def _safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract zip_path into a fresh `dest/<stem>` subfolder, refusing any
    member whose resolved path escapes dest (zip-slip guard). Returns the
    extraction directory."""
    extract_dir = dest / zip_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    base = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise RuntimeError(f'Refusing unsafe zip member: {member.filename}')
            zf.extract(member, extract_dir)
    return extract_dir


def _find_vid_folders(root: Path) -> list[Path]:
    """Walk `root` and return every directory that contains at least one
    .mp4, one .json, and one .bin (excluding output_* files we wrote on a
    previous run). Sorted by name so VID1 < VID2 < ... < VID9."""
    found = []
    for d in sorted(p for p in root.rglob('*') if p.is_dir()):
        files = [f for f in d.iterdir() if f.is_file() and not f.name.startswith('output_')]
        suffixes = {f.suffix.lower() for f in files}
        if {'.mp4', '.json', '.bin'}.issubset(suffixes):
            found.append(d)
    return found


def _first_with_suffix(folder: Path, suffix: str) -> Path:
    matches = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == suffix.lower()
        and not f.name.startswith('output_')
    )
    if not matches:
        raise FileNotFoundError(f'No {suffix} file in {folder}')
    return matches[0]


def _batch_worker(zip_path: Path, start_index: int = 1) -> None:
    """Run on a background thread. Extracts the zip, processes every VID
    folder it finds (capped at MAX_BATCH), writes outputs as
    output_video_{idx}.mp4 / output_file_{idx}.csv inside OUTPUT_FOLDER, and
    updates _BATCH_STATE as it goes. Errors on one VID don't kill the rest.

    If `start_index > 1`, the first (start_index - 1) VIDs are skipped
    entirely — useful when the user already has those outputs locally and
    only wants to re-run the rest after a config change. The completed
    counter still reports against `total = len(vids)` so progress reads
    correctly relative to the whole batch."""
    try:
        _log(f'Batch worker: extracting {zip_path.name} (start_index={start_index})')
        extract_dir = _safe_extract_zip(zip_path, ZIP_EXTRACT_FOLDER)
        vids = _find_vid_folders(extract_dir)[:MAX_BATCH]
        if not vids:
            with _BATCH_LOCK:
                _BATCH_STATE.update(state='error',
                                    error=f'No VID folders (mp4+json+bin) found in {zip_path.name}',
                                    finished_at=datetime.now(timezone.utc).isoformat())
            _log(f'Batch worker: no VID folders found, aborting')
            return
        _log(f'Batch worker: found {len(vids)} VID folders → {[v.name for v in vids]}')

        # Clear ONLY the VID outputs we're about to overwrite. If the caller
        # passed start_index=2 because they already have output_*_1.{mp4,csv}
        # saved locally, we shouldn't touch any leftover output_*_1.* files
        # that might happen to be on disk.
        for f in OUTPUT_FOLDER.iterdir():
            if not f.is_file():
                continue
            for j in range(start_index, len(vids) + 1):
                if f.name == f'output_video_{j}.mp4' or f.name == f'output_file_{j}.csv':
                    try:
                        f.unlink()
                    except Exception:
                        pass
                    break

        with _BATCH_LOCK:
            _BATCH_STATE.update(total=len(vids),
                                completed=max(0, start_index - 1),
                                results=[])

        for i, vid_dir in enumerate(vids):
            idx = i + 1
            if idx < start_index:
                _log(f'Skipping VID{idx} ({vid_dir.name}) — before start_index={start_index}')
                continue
            with _BATCH_LOCK:
                _BATCH_STATE['current'] = vid_dir.name
            _log(f'─── batch {idx}/{len(vids)}: {vid_dir.name} ───')

            per_video = {
                'index': idx,
                'vid_folder': vid_dir.name,
                'output_video_url': f'/outputs/output_video_{idx}.mp4',
                'csv_url': f'/outputs/output_file_{idx}.csv',
            }

            try:
                video = _first_with_suffix(vid_dir, '.mp4')
                intrinsics_path = _first_with_suffix(vid_dir, '.json')
                depth = _first_with_suffix(vid_dir, '.bin')
            except FileNotFoundError as e:
                per_video.update(status='error', error=str(e))
                with _BATCH_LOCK:
                    _BATCH_STATE['results'].append(per_video)
                    _BATCH_STATE['completed'] = idx
                continue

            try:
                intrinsic_np = _parse_intrinsics_from_text(intrinsics_path.read_text())
            except Exception as e:
                _log(f'  intrinsics read failed ({e}) — falling back to default FOV')
                intrinsic_np = None

            output_video = OUTPUT_FOLDER / f'output_video_{idx}.mp4'
            output_csv = OUTPUT_FOLDER / f'output_file_{idx}.csv'

            try:
                t0 = _now()
                gpu_result = _process_video_gpu(
                    str(video), str(output_video), str(output_csv),
                    intrinsic_np, depth_data_path=str(depth),
                )
                elapsed = _now() - t0
                per_video.update(
                    status='ok',
                    frames_with_detections=gpu_result.get('detected', 0),
                    used_lidar=gpu_result.get('used_lidar', False),
                    used_intrinsics=intrinsic_np is not None,
                    processing_time_seconds=round(elapsed, 1),
                )
                _log(f'  done in {elapsed:.1f}s, '
                     f"detections={gpu_result.get('detected', 0)}, "
                     f"lidar={gpu_result.get('used_lidar', False)}")
            except Exception as e:
                per_video.update(status='error', error=f'Processing failed: {e}')
                _log(f'  FAILED: {e}')

            with _BATCH_LOCK:
                _BATCH_STATE['results'].append(per_video)
                _BATCH_STATE['completed'] = idx

        with _BATCH_LOCK:
            _BATCH_STATE.update(state='done',
                                current=None,
                                finished_at=datetime.now(timezone.utc).isoformat())
        _log(f'Batch worker: done ({_BATCH_STATE["completed"]}/{_BATCH_STATE["total"]})')
    except Exception as e:
        with _BATCH_LOCK:
            _BATCH_STATE.update(state='error',
                                error=f'{type(e).__name__}: {e}',
                                current=None,
                                finished_at=datetime.now(timezone.utc).isoformat())
        _log(f'Batch worker crashed: {e}')


@api.post('/run_zip')
def run_zip(name: str = DEFAULT_ZIP_NAME, start_index: int = 1):
    """Kick off background processing of the named zip (already on disk in
    the Space repo). Returns immediately; poll /run_zip/status for progress.
    Refuses to start a second batch while one is in flight.

    `start_index` (1-indexed) lets you skip the first N-1 VIDs when you
    already have those outputs locally and only want to (re)run the rest —
    useful after a redeploy that wiped /tmp."""
    with _BATCH_LOCK:
        if _BATCH_STATE['state'] == 'running':
            raise HTTPException(409, f'Batch already running ({_BATCH_STATE["completed"]}/{_BATCH_STATE["total"]} on {_BATCH_STATE["current"]}). Wait or restart Space.')

    if start_index < 1:
        raise HTTPException(400, 'start_index must be ≥ 1')

    try:
        zip_path = _resolve_zip_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    with _BATCH_LOCK:
        _BATCH_STATE.update(
            state='running',
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            zip=str(zip_path),
            total=0,
            completed=0,
            current=None,
            results=[],
            error=None,
        )

    t = threading.Thread(target=_batch_worker, args=(zip_path, start_index), daemon=True)
    t.start()
    return {
        'message': f'Started batch processing of {zip_path.name}',
        'zip': str(zip_path),
        'status_url': '/run_zip/status',
    }


@api.get('/run_zip/status')
def run_zip_status():
    """Return a snapshot of the batch worker's state. Safe to poll often."""
    with _BATCH_LOCK:
        return dict(_BATCH_STATE)


@api.get('/list_zip')
def list_zip(name: str = DEFAULT_ZIP_NAME):
    """Peek into a zip without processing — show how many VID folders it
    contains, so the user can sanity-check the upload before triggering."""
    try:
        zip_path = _resolve_zip_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    extract_dir = _safe_extract_zip(zip_path, ZIP_EXTRACT_FOLDER)
    vids = _find_vid_folders(extract_dir)
    return {
        'zip': str(zip_path),
        'vid_count': len(vids),
        'vid_folders': [str(v.relative_to(extract_dir)) for v in vids],
    }


# ─────────────────────────── Minimal HTML upload form ───────────────────────────
_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MeTRAbs (batch of 9)</title></head>
<body style="font-family:system-ui;max-width:820px;margin:2rem auto;padding:1rem">
<h1>MeTRAbs — batch of up to 9</h1>

<h3 style="margin-top:1rem">Run the zip already on the Space</h3>
<p style="color:#555">Whatever's at <code>HUGGING FACE VIDEOS.zip</code> in this Space's repo
gets unzipped, every VID folder inside (mp4 + json + bin) gets processed, and outputs
land at <code>/outputs/output_file_N.csv</code> + <code>/outputs/output_video_N.mp4</code>.</p>
<p>
  <button id="run-zip" type="button">▶ Run zip</button>
  <span id="run-zip-status" style="color:#666;margin-left:8px"></span>
</p>

<h3 style="margin-top:2rem">— or — upload up to 9 directly</h3>
<p style="color:#555">Only practical for small clips. For big LiDAR captures use the zip path above.</p>
<form id="f">
  <p><label><strong>Videos (.mp4, up to 9):</strong><br>
    <input type="file" name="files" accept="video/mp4" multiple required></label></p>
  <p><label><strong>Intrinsics (.json, optional, paired by index):</strong><br>
    <input type="file" name="intrinsics_files" accept="application/json,.json" multiple></label></p>
  <p><label><strong>LiDAR depth (.bin, optional, paired by index):</strong><br>
    <input type="file" name="depth_files" accept=".bin" multiple></label></p>
  <p><label><strong>Ground truth (.csv, optional, paired by index):</strong><br>
    <input type="file" name="ground_truth_files" accept=".csv" multiple></label></p>
  <button type="submit">Upload batch</button>
</form>
<p id="status"></p>
<h3 style="margin-top:2rem">Results <span id="result-status" style="font-weight:normal;color:#888;font-size:0.85em">(updated after each batch)</span></h3>
<div id="result-box" style="padding:12px;border:1px solid #ddd;border-radius:4px;background:#fafafa">
  <p style="margin:0 0 8px;color:#888;font-size:0.85em">Each new batch overwrites these URLs.</p>
  <div id="downloads"></div>
</div>
<h3 style="margin-top:2rem">Live server log <span style="font-weight:normal;color:#888;font-size:0.85em">(auto-refreshes every 2s)</span></h3>
<pre id="log" style="background:#111;color:#0f0;padding:10px;height:280px;overflow:auto;font-size:12px;line-height:1.35;margin:0;border-radius:4px;white-space:pre-wrap">connecting...</pre>
<script>
function renderDownloads(n) {
  let html = '';
  for (let i = 1; i <= n; i++) {
    html += '<p style="margin:4px 0">'
      + '<strong>#' + i + ':</strong> '
      + '<a href="/outputs/output_file_' + i + '.csv" download>📄 CSV</a> · '
      + '<a href="/outputs/output_file_' + i + '.csv" target="_blank">🔍 view</a> · '
      + '<a href="/outputs/output_video_' + i + '.mp4" download>🎞️ video</a>'
      + '</p>';
  }
  document.getElementById('downloads').innerHTML = html;
}
renderDownloads(0);

// ── Zip-on-disk runner ────────────────────────────────────────────────────
async function pollZipStatus() {
  try {
    const r = await fetch('/run_zip/status', {cache:'no-store'});
    const s = await r.json();
    const el = document.getElementById('run-zip-status');
    if (s.state === 'idle') {
      el.textContent = 'idle';
    } else if (s.state === 'running') {
      el.textContent = `running… ${s.completed}/${s.total || '?'} (on ${s.current || '...'})`;
      if (s.total) renderDownloads(s.total);
    } else if (s.state === 'done') {
      el.textContent = `✅ done: ${s.completed}/${s.total} processed`;
      renderDownloads(s.total);
    } else if (s.state === 'error') {
      el.textContent = `⚠ error: ${s.error || 'unknown'}`;
    }
  } catch { /* keep prior text */ }
}
document.getElementById('run-zip').addEventListener('click', async () => {
  const btn = document.getElementById('run-zip');
  btn.disabled = true;
  document.getElementById('run-zip-status').textContent = 'starting…';
  try {
    const r = await fetch('/run_zip', {method:'POST'});
    const j = await r.json();
    if (!r.ok) { document.getElementById('run-zip-status').textContent = 'error: ' + (j.detail || r.status); }
  } catch (e) {
    document.getElementById('run-zip-status').textContent = 'error: ' + e.message;
  } finally {
    setTimeout(() => { btn.disabled = false; }, 2000);
  }
});
setInterval(pollZipStatus, 2000);
pollZipStatus();
// ──────────────────────────────────────────────────────────────────────────

document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const nVideos = (fd.getAll('files') || []).filter(f => f && f.name).length;
  if (nVideos > 9) {
    document.getElementById('status').textContent = 'Error: pick 9 videos at most.';
    return;
  }
  document.getElementById('status').textContent =
    'Processing ' + nVideos + ' video(s) — watch the log below.';
  renderDownloads(nVideos);
  try {
    const r = await fetch('/upload', {method:'POST', body: fd});
    const j = await r.json();
    if (!r.ok) { document.getElementById('status').textContent = 'Error: ' + (j.detail || r.status); return; }
    document.getElementById('status').textContent =
      '✅ Batch done: ' + j.succeeded + '/' + j.batch_size + ' succeeded.';
    document.getElementById('result-status').textContent =
      'last batch: ' + j.succeeded + '/' + j.batch_size + ' succeeded';
  } catch (err) {
    document.getElementById('status').textContent =
      '⚠ Browser lost the upload connection, but the server may still be running. '
      + 'Watch the log; downloads above will be valid once you see "Done video N in Xs" lines.';
  }
});
async function pollLogs() {
  try {
    const r = await fetch('/logs', {cache:'no-store'});
    const j = await r.json();
    const lines = j.lines || [];
    const el = document.getElementById('log');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.textContent = lines.join('\\n') || '(no log output yet)';
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch { /* keep previous view */ }
}
setInterval(pollLogs, 2000);
pollLogs();
</script>
</body></html>
"""


@api.get('/', response_class=HTMLResponse)
def ui():
    return HTMLResponse(_UI_HTML)


# ─────────────────────────── Gradio shell (ZeroGPU plumbing) ───────────────────────────
import gradio as gr

with gr.Blocks(analytics_enabled=False) as _gradio_shell:
    gr.Markdown("MeTRAbs batch-of-9 server is running. Open `/` for the upload UI, or POST to `/upload`.")

app = gr.mount_gradio_app(api, _gradio_shell, path='/gradio')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
