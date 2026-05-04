"""
MeTRAbs Pose Estimation Server — Hugging Face Space
FastAPI + minimal Gradio shell (Gradio shell only exists so HF Spaces lets us pick ZeroGPU).
The real UI is the FastAPI HTML at "/". MeTRAbs runs under @spaces.GPU.
"""
import os
import ssl
import json
import csv
import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import time as _now

import certifi
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# ─────────────────────────── Live log buffer ───────────────────────────
_LOG_LINES: deque[str] = deque(maxlen=500)


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
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Load model once at startup ───────────────────────────
from huggingface_hub import snapshot_download
MODEL_REPO = 'EngDesFAU26-SmartphonePose/metrabs-eff2l-y4-384px'
_log(f'Downloading MeTRAbs model from HF hub: {MODEL_REPO}')
_model_path = snapshot_download(repo_id=MODEL_REPO, cache_dir=CACHE_DIR)
if not os.path.exists(os.path.join(_model_path, 'saved_model.pb')):
    raise RuntimeError(f'saved_model.pb not found in {_model_path}')
_log(f'Loading model from {_model_path}')
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

    The output CSV uses a 2-row header (row 1 = unit groups, row 2 = x/y/z labels).
    Reads it line-by-line so the structure is preserved, computes per-row error
    for ankle joints whose (frame, joint_idx, joint_name) match the GT, and
    writes the err values back into the last 4 columns of each row.

    GT CSV is expected in the simpler one-row-header format with columns
    `frame, joint_idx, joint_name, x_mm, y_mm, z_mm`.
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

    # Load the output CSV preserving its 2-row header structure.
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
    # mm columns are the first (x, y, z) triplet after the 3 id columns.
    MM_X, MM_Y, MM_Z = 3, 4, 5
    # Error columns are the last 4 columns of the row.
    ERR_X, ERR_Y, ERR_Z, EUCL = n_cols - 4, n_cols - 3, n_cols - 2, n_cols - 1

    # Load GT.
    df_gt = pd.read_csv(ground_truth_csv)
    df_gt = df_gt[df_gt['joint_name'].isin(joints)].copy()
    gt_lookup = {}
    for _, gt_row in df_gt.iterrows():
        try:
            key = (str(int(gt_row['frame'])), str(int(gt_row['joint_idx'])), str(gt_row['joint_name']))
            gt_lookup[key] = (float(gt_row['x_mm']), float(gt_row['y_mm']), float(gt_row['z_mm']))
        except (ValueError, KeyError):
            continue

    # Walk data rows, fill in errors where the joint is in our analyze set
    # AND its (frame, joint_idx, joint_name) is present in the GT.
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

    # Write back, preserving the 2-row header.
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header_row1)
        w.writerow(header_row2)
        for row in data_rows:
            w.writerow(row)

    # Build summary.
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
    array of metric depth in METERS.

    Header (12 bytes, little-endian uint32s): width, height, frame_count.
    Body: frame_count × height × width × float32 row-major.
    Returns None if the file is missing, malformed, or empty.
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
    """Build a two-row CSV header so unit names visually group their x/y/z cells in
    Excel/Numbers/Sheets.

    Layout:
      Row 1:  ,,,mm,,,meters,,,centimeters,,,inches,,,feet,,,error_mm,,,
      Row 2:  frame,joint_idx,joint_name,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,err_x,err_y,err_z,euclidean_err
    """
    row1 = ['', '', '']                                          # under id columns
    for unit, _ in _UNIT_FACTORS:
        row1 += [unit, '', '']                                   # group label spans the 3 cells
    row1 += ['error_mm', '', '', '']                             # error group spans 4 cells

    row2 = ['frame', 'joint_idx', 'joint_name']
    for _ in _UNIT_FACTORS:
        row2 += ['x', 'y', 'z']
    row2 += ['err_x', 'err_y', 'err_z', 'euclidean_err']
    return row1, row2


@spaces.GPU(duration=600)
def _process_video_gpu(input_path: str,
                       video_out: str,
                       csv_out: str,
                       intrinsic_matrix_np: np.ndarray | None,
                       depth_data_path: str | None = None) -> dict:
    """Run MeTRAbs once per frame with intrinsics if provided (else default FOV).
    When LiDAR depth is uploaded alongside intrinsics, each joint's Z is replaced
    by the measured depth at its 2D pixel — sub-inch absolute accuracy.

    Writes a CSV with a 2-row header (unit groups + x/y/z labels) so the columns
    are easy to read in Excel/Numbers. Single set of x/y/z per joint per frame
    (no separate "no intrinsics" mode — the no-intrinsics view was 50% off and
    unhelpful for downstream analysis).

    Returns dict with `detected` count and `used_lidar` flag.
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

    # LiDAR depth (optional). Used only when intrinsics are also present —
    # joint z values get replaced by measured depth at each joint pixel.
    depth_data = _read_depth_file(depth_data_path) if depth_data_path else None
    use_lidar = depth_data is not None and intrinsics_tensor is not None
    if depth_data_path and not use_lidar:
        if depth_data is None:
            _log('Depth file present but failed to load — falling back to MeTRAbs Z.')
        elif intrinsics_tensor is None:
            _log('Depth file present but no intrinsics — LiDAR fusion needs both. Skipping.')

    temp_out = str(OUTPUT_FOLDER / 'temp.mp4')
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

            # Single inference. Pass intrinsics if available — that's the only mode
            # we keep in the CSV. (Mode A "no intrinsics" was dropped: it was ~50% off
            # in absolute depth and never used downstream.)
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

            # LiDAR fusion: replace per-joint z with measured depth at the joint pixel,
            # but with two safeguards against the background-bleed pattern we saw
            # earlier (single-pixel mapping landing on wall/floor instead of body):
            #   1. Take the MEDIAN of a 5x5 patch around the joint's depth-space
            #      pixel — kills single-pixel outliers from depth-map noise.
            #   2. Reject the LiDAR reading if it disagrees with MeTRAbs's predicted
            #      z by more than 1 m. MeTRAbs (with intrinsics) has reasonable
            #      relative depth within a body; if our looked-up depth is wildly
            #      off, the 2D pixel almost certainly landed on the wrong object,
            #      so we keep MeTRAbs's z for that joint.
            if pose is not None and use_lidar and source_idx < len(depth_data):
                pose2d_first = poses2d[0] if poses2d is not None and len(poses2d) > 0 else None
                if pose2d_first is not None:
                    depth_frame = depth_data[source_idx]   # [dh, dw] float32 meters
                    dh, dw = depth_frame.shape
                    pose = pose.copy()
                    HALF = 2                # 5x5 patch (half-width of 2)
                    SANITY_MM = 1000.0      # discard if disagrees with MeTRAbs by >1 m
                    for jid in range(len(pose)):
                        px = float(pose2d_first[jid][0])
                        py = float(pose2d_first[jid][1])
                        dx = int(px * dw / out_w)
                        dy = int(py * dh / out_h)
                        if not (0 <= dx < dw and 0 <= dy < dh):
                            continue
                        # Median over a 5x5 patch, ignoring zero/invalid pixels.
                        y_lo = max(0, dy - HALF); y_hi = min(dh, dy + HALF + 1)
                        x_lo = max(0, dx - HALF); x_hi = min(dw, dx + HALF + 1)
                        patch = depth_frame[y_lo:y_hi, x_lo:x_hi]
                        valid = patch[patch > 0]
                        if valid.size == 0:
                            continue
                        z_mm = float(np.median(valid)) * 1000.0
                        # Sanity check against MeTRAbs's predicted z for this joint.
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
                    row += ['', '', '', '']   # error cols (filled by analyze() if GT given)
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
api = FastAPI(title='MeTRAbs Pose Estimation Server')
api.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
api.mount('/outputs', StaticFiles(directory=str(OUTPUT_FOLDER)), name='outputs')


@api.get('/status')
def status():
    return {
        'message': 'MeTRAbs HF Space running',
        'endpoint': '/upload',
        'ui': '/',
        'skeleton': SKELETON,
        'resize_factor': RESIZE_FACTOR,
        'frame_skip': FRAME_SKIP,
    }


@api.get('/logs')
def logs():
    return {'lines': list(_LOG_LINES)}


def _parse_intrinsics(intrinsics_json, fx, fy, cx, cy):
    if intrinsics_json:
        try:
            data = json.loads(intrinsics_json)
            matrix = data.get('camera_intrinsics', {}).get('intrinsic_matrix')
            if matrix:
                arr = np.array(matrix, dtype=np.float32)
                _log(f'Intrinsics loaded: fx={arr[0,0]:.1f}, fy={arr[1,1]:.1f}')
                return arr
        except Exception as e:
            _log(f'Intrinsics JSON parse error: {e}')
    if fx:
        _fx = float(fx)
        _fy = float(fy or fx)
        _cx = float(cx or 0)
        _cy = float(cy or 0)
        arr = np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]], dtype=np.float32)
        _log(f'Intrinsics from fields: fx={_fx}, fy={_fy}')
        return arr
    return None


@api.post('/upload')
async def upload_video(
    file: UploadFile = File(...),
    intrinsics_json: str = Form(None),
    focal_length_x: str = Form(None),
    focal_length_y: str = Form(None),
    principal_point_x: str = Form(None),
    principal_point_y: str = Form(None),
    ground_truth: UploadFile = File(None),
    depth_data: UploadFile = File(None),
):
    _log('UPLOAD REQUEST RECEIVED')
    _log(f'filename: {file.filename}')

    if not file.filename:
        raise HTTPException(400, 'No filename provided')
    ext = Path(file.filename).suffix.lower()
    if ext != '.mp4':
        raise HTTPException(400, f'Only .mp4 files allowed, got {file.filename}')

    _delete_all_files(UPLOAD_FOLDER)
    _delete_all_files(OUTPUT_FOLDER)

    input_path = UPLOAD_FOLDER / file.filename
    output_video = OUTPUT_FOLDER / 'output_video_1.mp4'
    output_csv = OUTPUT_FOLDER / 'output_file_1.csv'

    with open(input_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    _log(f'Saved upload: {input_path}')

    gt_path = None
    if ground_truth is not None and getattr(ground_truth, 'filename', None):
        gt_path = UPLOAD_FOLDER / 'ground_truth.csv'
        with open(gt_path, 'wb') as f_gt:
            shutil.copyfileobj(ground_truth.file, f_gt)
        ground_truth.file.close()
        _log(f'Saved ground truth CSV: {gt_path}')

    depth_path = None
    if depth_data is not None and getattr(depth_data, 'filename', None):
        depth_path = UPLOAD_FOLDER / 'depth.bin'
        with open(depth_path, 'wb') as f_d:
            shutil.copyfileobj(depth_data.file, f_d)
        depth_data.file.close()
        _log(f'Saved depth_data file: {depth_path} ({depth_path.stat().st_size} bytes)')

    intrinsic_np = _parse_intrinsics(
        intrinsics_json, focal_length_x, focal_length_y,
        principal_point_x, principal_point_y,
    )

    try:
        t0 = _now()
        gpu_result = _process_video_gpu(
            str(input_path), str(output_video), str(output_csv),
            intrinsic_np,
            depth_data_path=str(depth_path) if depth_path else None,
        )
        detected = gpu_result.get('detected', 0)
        used_lidar = gpu_result.get('used_lidar', False)
        elapsed = _now() - t0
        _log(f'Done in {elapsed:.1f}s, {detected} frames with detections, lidar_used={used_lidar}')
    except Exception as e:
        raise HTTPException(500, f'Processing failed: {e}')
    finally:
        file.file.close()

    if not output_video.exists():
        raise HTTPException(500, f'Output video not created: {output_video}')
    if not output_csv.exists():
        raise HTTPException(500, f'CSV not created: {output_csv}')

    try:
        analysis_result = analyze(
            str(output_csv),
            ground_truth_csv=str(gt_path) if gt_path else None,
            used_intrinsics=intrinsic_np is not None,
        )
    except Exception as e:
        analysis_result = {'error': f'analyze failed: {e}'}

    return {
        'message': 'Video processed successfully',
        'input_file': input_path.name,
        'output_file': output_video.name,
        'output_url': f'/outputs/{output_video.name}',
        'csv_file': output_csv.name,
        'csv_url': f'/outputs/{output_csv.name}',
        'used_intrinsics': intrinsic_np is not None,
        'used_ground_truth': gt_path is not None,
        'used_lidar_depth': used_lidar,
        'processing_time_seconds': round(elapsed, 1),
        'frames_with_detections': detected,
        'error_analysis': analysis_result,
    }


# ─────────────────────────── Minimal HTML upload form ───────────────────────────
_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MeTRAbs</title></head>
<body style="font-family:system-ui;max-width:760px;margin:2rem auto;padding:1rem">
<h1>MeTRAbs</h1>
<form id="f">
  <input type="file" name="file" accept="video/mp4" required><br><br>
  <textarea name="intrinsics_json" rows="4" cols="60" placeholder="intrinsics JSON (optional)"></textarea><br><br>
  <button type="submit">Upload</button>
</form>
<p id="status"></p>
<h3 style="margin-top:2rem">Latest result <span id="result-status" style="font-weight:normal;color:#888;font-size:0.85em">(updated after each upload)</span></h3>
<div id="result-box" style="padding:12px;border:1px solid #ddd;border-radius:4px;background:#fafafa">
  <p style="margin:0 0 8px"><strong>Downloads</strong> <span style="color:#888;font-size:0.85em">(fixed URLs — each new upload overwrites these)</span></p>
  <p style="margin:0 0 8px">
    <a href="/outputs/output_file_1.csv" download>📄 Download CSV</a>
    &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="/outputs/output_file_1.csv" target="_blank">🔍 View CSV in browser</a>
    &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="/outputs/output_video_1.mp4" download>🎞️ Download annotated video</a>
  </p>
  <p style="margin:0" id="csv-preview-status"><button id="preview-btn" type="button">Preview CSV (first 20 rows)</button></p>
  <pre id="csv-preview" style="display:none;background:#111;color:#eee;padding:10px;height:200px;overflow:auto;font-size:12px;margin:8px 0 0;border-radius:4px"></pre>
</div>
<h3 style="margin-top:2rem">Live server log <span style="font-weight:normal;color:#888;font-size:0.85em">(auto-refreshes every 2s)</span></h3>
<pre id="log" style="background:#111;color:#0f0;padding:10px;height:260px;overflow:auto;font-size:12px;line-height:1.35;margin:0;border-radius:4px;white-space:pre-wrap">connecting...</pre>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  document.getElementById('status').textContent =
    'Processing — watch the log below. Downloads above will update when the log says "Done in Xs".';
  try {
    const r = await fetch('/upload', {method:'POST', body:new FormData(e.target)});
    const j = await r.json();
    if (!r.ok) { document.getElementById('status').textContent = 'Error: ' + (j.detail || r.status); return; }
    document.getElementById('status').textContent =
      '✅ Done in ' + j.processing_time_seconds + 's. intrinsics used: ' + j.used_intrinsics
      + '. frames with detections: ' + j.frames_with_detections + '. Downloads below are ready.';
  } catch (err) {
    document.getElementById('status').textContent =
      '⚠ Browser lost the upload connection, but the server is probably still running. '
      + 'Watch the log; downloads above will be valid once you see "Done in Xs".';
  }
});
let _lastDoneSeen = '';
async function pollLogs() {
  try {
    const r = await fetch('/logs', {cache:'no-store'});
    const j = await r.json();
    const lines = j.lines || [];
    const el = document.getElementById('log');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.textContent = lines.join('\\n') || '(no log output yet)';
    if (atBottom) el.scrollTop = el.scrollHeight;
    let lastUpload = -1, lastDone = -1;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].indexOf('UPLOAD REQUEST RECEIVED') !== -1) lastUpload = i;
      if (lines[i].indexOf('Done in') !== -1) lastDone = i;
    }
    if (lastDone > lastUpload && lastDone !== -1) {
      const doneLine = lines[lastDone];
      if (doneLine !== _lastDoneSeen) {
        _lastDoneSeen = doneLine;
        const m = doneLine.match(/Done in ([\\d.]+)s, (\\d+) frames/);
        if (m) {
          document.getElementById('result-status').textContent =
            '✅ last run: ' + m[1] + 's, ' + m[2] + ' frames with detections';
        }
      }
    }
  } catch { /* keep previous view */ }
}
setInterval(pollLogs, 2000);
pollLogs();
document.getElementById('preview-btn').addEventListener('click', async () => {
  const pre = document.getElementById('csv-preview');
  const btn = document.getElementById('preview-btn');
  if (pre.style.display === 'block') {
    pre.style.display = 'none';
    btn.textContent = 'Preview CSV (first 20 rows)';
    return;
  }
  btn.textContent = 'Loading…';
  try {
    const r = await fetch('/outputs/output_file_1.csv', {cache:'no-store'});
    if (!r.ok) { pre.textContent = 'No CSV yet (HTTP ' + r.status + ')'; }
    else {
      const txt = await r.text();
      const rows = txt.split('\\n').slice(0, 21).join('\\n');
      pre.textContent = rows + (txt.split('\\n').length > 21 ? '\\n… (truncated)' : '');
    }
    pre.style.display = 'block';
    btn.textContent = 'Hide preview';
  } catch (e) {
    pre.textContent = 'Error: ' + e.message;
    pre.style.display = 'block';
    btn.textContent = 'Preview CSV (first 20 rows)';
  }
});
</script>
</body></html>
"""


@api.get('/', response_class=HTMLResponse)
def ui():
    return HTMLResponse(_UI_HTML)


# ─────────────────────────── Gradio shell (ZeroGPU plumbing) ───────────────────────────
import gradio as gr

with gr.Blocks(analytics_enabled=False) as _gradio_shell:
    gr.Markdown("MeTRAbs server is running. Open `/` for the upload UI, or POST to `/upload`.")

# Mount Gradio at /gradio so HF accepts the gradio SDK label.
# FastAPI is the parent app; / still serves your HTML upload form.
app = gr.mount_gradio_app(api, _gradio_shell, path='/gradio')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)