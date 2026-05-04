"""
MeTRAbs Pose Estimation Server — Hugging Face Space
Mirrors the teammate's FastAPI /upload endpoint + adds iPhone ARKit camera intrinsics support.

Runs on ZeroGPU (free shared H100). The @spaces.GPU decorator allocates the GPU
only during the processing call; the server is otherwise idle.
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
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import spaces  # HF ZeroGPU — no-op locally
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _Stub:
        def GPU(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _Stub()

# ─────────────────────────── Live log buffer (exposed via /logs for web UI) ──
_LOG_LINES: deque[str] = deque(maxlen=500)


def _log(msg: str) -> None:
    """Print and append to the ring buffer served at /logs."""
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    _LOG_LINES.append(line)


# ─────────────────────────── SSL / cert fix ───────────────────────────
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

# ─────────────────────────── Config ───────────────────────────
MODEL_TYPE = 'metrabs_mob3l_y4t'
SKELETON = 'smpl_24'
RESIZE_FACTOR = 0.25
FRAME_SKIP = 2
CACHE_DIR = os.environ.get('METRABS_CACHE', './metrabs_models')

BASE_DIR = Path('/tmp/metrabs-server')
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'outputs'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Load model once at startup ───────────────────────────
# Mirrored from RWTH Aachen's omnomnom server to our HF org for fast, reliable cold starts.
# HF CDN is ~10-50x faster than RWTH and gets cached by HF Spaces across restarts.
from huggingface_hub import snapshot_download
MODEL_REPO = 'EngDesFAU26-SmartphonePose/metrabs-mob3l-y4t'
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


def _draw_skeletons(frame, poses2d, edges, w, h) -> None:
    if poses2d is None:
        return
    for pose in poses2d:
        pts = pose.astype(int)
        for i, j in edges:
            if i < len(pts) and j < len(pts):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 2, (0, 128, 255), -1)


@spaces.GPU(duration=300)
def _process_video_gpu(input_path: str,
                       video_out: str,
                       csv_out: str,
                       intrinsic_matrix_np: np.ndarray | None) -> int:
    """Run MeTRAbs on the whole video. Decorated so ZeroGPU allocates a GPU for the call."""
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(width * RESIZE_FACTOR)
    out_h = int(height * RESIZE_FACTOR)
    out_fps = fps / FRAME_SKIP if FRAME_SKIP > 0 else fps

    intrinsics_tensor = None
    if intrinsic_matrix_np is not None:
        scaled = intrinsic_matrix_np.copy()
        scaled[0, :] *= RESIZE_FACTOR
        scaled[1, :] *= RESIZE_FACTOR
        intrinsics_tensor = tf.constant(scaled, dtype=tf.float32)
        _log(f'Using intrinsics (scaled):\n{scaled}')

    temp_out = str(OUTPUT_FOLDER / 'temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out, fourcc, out_fps, (out_w, out_h))

    detected = 0
    with open(csv_out, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame', 'person_id', 'joint_id', 'x', 'y', 'z'])

        source_idx = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            source_idx += 1
            if FRAME_SKIP > 1 and source_idx % FRAME_SKIP != 0:
                continue
            processed += 1

            frame_small = cv2.resize(frame, (out_w, out_h))
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            image_tensor = tf.convert_to_tensor(rgb)
            if intrinsics_tensor is not None:
                pred = MODEL.detect_poses(
                    image_tensor, skeleton=SKELETON,
                    intrinsic_matrix=intrinsics_tensor,
                )
            else:
                pred = MODEL.detect_poses(image_tensor, skeleton=SKELETON)

            poses2d = pred.get('poses2d')
            poses3d = pred.get('poses3d')
            if hasattr(poses2d, 'numpy'):
                poses2d = poses2d.numpy()
            if hasattr(poses3d, 'numpy'):
                poses3d = poses3d.numpy()

            if poses3d is not None and len(poses3d) > 0:
                detected += 1
                for pid, pose in enumerate(poses3d):
                    for jid, (x, y, z) in enumerate(pose):
                        writer.writerow([processed, pid, jid, float(x), float(y), float(z)])

            _draw_skeletons(frame_small, poses2d, EDGES, out_w, out_h)
            out.write(frame_small)

            if processed % 25 == 0:
                _log(f'Processed {processed} frames ({detected} w/ detections)')

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

    return detected


# ─────────────────────────── FastAPI app ───────────────────────────
api = FastAPI(title='MeTRAbs Pose Estimation Server')
api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
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
    """Recent server log lines (last ~500). Polled by the web UI for live progress."""
    return {'lines': list(_LOG_LINES)}


def _parse_intrinsics(intrinsics_json: str | None,
                      fx: str | None, fy: str | None,
                      cx: str | None, cy: str | None) -> np.ndarray | None:
    """Return a 3x3 intrinsics matrix or None."""
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

    intrinsic_np = _parse_intrinsics(
        intrinsics_json, focal_length_x, focal_length_y,
        principal_point_x, principal_point_y,
    )

    try:
        t0 = _now()
        detected = _process_video_gpu(str(input_path), str(output_video), str(output_csv), intrinsic_np)
        elapsed = _now() - t0
        _log(f'Done in {elapsed:.1f}s, {detected} frames with detections')
    except Exception as e:
        raise HTTPException(500, f'Processing failed: {e}')
    finally:
        file.file.close()

    if not output_video.exists():
        raise HTTPException(500, f'Output video not created: {output_video}')
    if not output_csv.exists():
        raise HTTPException(500, f'CSV not created: {output_csv}')

    # Run error analysis (stub by default; teammate fills in error_analysis.py)
    try:
        from error_analysis import analyze
        analysis_result = analyze(str(output_csv), used_intrinsics=intrinsic_np is not None)
    except Exception as e:
        analysis_result = {'error': f'error_analysis failed: {e}'}

    return {
        'message': 'Video processed successfully',
        'input_file': input_path.name,
        'output_file': output_video.name,
        'output_url': f'/outputs/{output_video.name}',
        'csv_file': output_csv.name,
        'csv_url': f'/outputs/{output_csv.name}',
        'used_intrinsics': intrinsic_np is not None,
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
// Upload handler — tab-safe: even if this fetch dies, server keeps going and
// the downloads above will still work once the log shows "Done in Xs".
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
      + '. frames with detections: ' + j.frames_with_detections
      + '. Downloads below are ready.';
  } catch (err) {
    // Browser lost the POST (tab idle, proxy timeout) — status banner falls back
    // to the log-based detection below, downloads still work.
    document.getElementById('status').textContent =
      '⚠ Browser lost the upload connection, but the server is probably still running. '
      + 'Watch the log; downloads above will be valid once you see "Done in Xs".';
  }
});

// Live log poller — also sniffs for "Done in Xs" to update the result banner
// in case the upload fetch died before returning.
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

    // Find the last "Done in" after the last "UPLOAD REQUEST RECEIVED"
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
  } catch { /* keep previous view on transient failure */ }
}
setInterval(pollLogs, 2000);
pollLogs();

// CSV preview — fetch and show first 20 rows
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


# FastAPI is the app (no Gradio mount — HF Docker SDK runs this directly)
app = api


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
