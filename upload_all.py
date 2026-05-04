"""
Sequentially upload VID1..VID9 from the iPhone capture folder to the running
MeTRAbs HF Space, wait for each run to finish, and save its output CSV +
annotated video back into the *same* VIDn folder it came from before
triggering the next one — so inputs and outputs stay paired up.

Why one-at-a-time: a single 1.4 GB multipart upload would 413 against the
Space's reverse-proxy size cap, time out the connection, and chew Space RAM.
~180 MB per request fits well within all three.

Why save outputs *between* runs: the Space writes to fixed paths
(/outputs/output_file_1.csv, /outputs/output_video_1.mp4) and wipes them on
the next upload — so we have to download before kicking off the next video.

Re-runnable: if `VIDn/output_file_n.csv` already exists, that VID is skipped.
Delete a result file to force that one to re-run.

Usage:
    1. Edit SPACE_URL below to point at your Space (or set HF_SPACE_URL env var).
    2. python upload_all.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

# ─── Edit these if your paths differ ─────────────────────────────────────────
SPACE_URL = os.environ.get(
    'HF_SPACE_URL',
    'https://engdesfau26-smartphonepose-metrabs-server.hf.space',
).rstrip('/')

SOURCE_DIR = Path('/Users/christinapappachan/Downloads/HUGGING FACE VIDEOS')
NUM_VIDEOS = 9

# How long to wait for the Space to finish processing one video. MeTRAbs on
# ZeroGPU usually finishes a 30-second clip in under 2 min, but cold starts
# (model download + GPU spin-up) can take several minutes. Give it room.
UPLOAD_TIMEOUT_SECS = (30, 1800)   # (connect, read) — 30 min read budget
DOWNLOAD_TIMEOUT_SECS = (30, 600)
# ─────────────────────────────────────────────────────────────────────────────


def _find_one(folder: Path, suffix: str) -> Path:
    """Return the single iPhone-side file with the given suffix in `folder`.
    Files we wrote ourselves on a previous run (`output_*`) are filtered out
    so a re-run still finds the original input mp4 instead of the annotated
    output mp4."""
    matches = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() == suffix.lower() and not p.name.startswith('output_')
    )
    if not matches:
        raise FileNotFoundError(f'No {suffix} file in {folder}')
    if len(matches) > 1:
        print(f'  ! multiple {suffix} files in {folder.name}, using {matches[0].name}')
    return matches[0]


def _human_size(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n_bytes < 1024:
            return f'{n_bytes:.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.1f} TB'


def _stream_download(url: str, dest: Path) -> None:
    """Stream the URL to `dest` so we don't load the whole annotated video into
    RAM. Writes to a .part file first and atomically renames on success."""
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECS) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def upload_one(idx: int, video: Path, intrinsics: Path, depth: Path) -> dict:
    """POST a single (video, intrinsics, depth) triple to /upload and return
    the JSON response. The Space's app.py expects intrinsics as a *form
    string*, not a file, so we read the JSON and pass it as text."""
    intrinsics_text = intrinsics.read_text()
    # Cheap sanity check — fail fast if the intrinsics JSON is the wrong shape,
    # before we waste a minute uploading a 180 MB body the server will then
    # silently fall back from.
    try:
        parsed = json.loads(intrinsics_text)
        if not parsed.get('camera_intrinsics', {}).get('intrinsic_matrix'):
            print(f'  ! intrinsics file has no camera_intrinsics.intrinsic_matrix '
                  f'— Space will fall back to default FOV')
    except json.JSONDecodeError as e:
        print(f'  ! intrinsics JSON parse error ({e}) — Space will fall back to default FOV')

    url = f'{SPACE_URL}/upload'
    print(f'  POST {url}')
    print(f'    video      {video.name}      ({_human_size(video.stat().st_size)})')
    print(f'    intrinsics {intrinsics.name} ({_human_size(intrinsics.stat().st_size)})')
    print(f'    depth      {depth.name}      ({_human_size(depth.stat().st_size)})')

    t0 = time.monotonic()
    with open(video, 'rb') as vf, open(depth, 'rb') as df:
        files = {
            'file': (f'video_{idx:02d}.mp4', vf, 'video/mp4'),
            'depth_data': (f'depth_{idx:02d}.bin', df, 'application/octet-stream'),
        }
        data = {'intrinsics_json': intrinsics_text}
        r = requests.post(url, files=files, data=data, timeout=UPLOAD_TIMEOUT_SECS)
    elapsed = time.monotonic() - t0

    if not r.ok:
        # Surface the server's own error so we can tell 413 (too big) from a
        # GPU/processing error.
        raise RuntimeError(f'HTTP {r.status_code} after {elapsed:.0f}s — {r.text[:500]}')

    payload = r.json()
    print(f'  ✓ server returned ok in {elapsed:.0f}s '
          f"(detected={payload.get('frames_with_detections', '?')}, "
          f"lidar={payload.get('used_lidar_depth', '?')}, "
          f"intrinsics={payload.get('used_intrinsics', '?')})")
    return payload


def fetch_outputs(idx: int, dest_folder: Path) -> None:
    """Download the Space's two fixed output URLs and save them inside
    `dest_folder` (which is the VIDn folder for this index). Has to happen
    *before* the next upload, since the Space overwrites those URLs on the
    next request."""
    csv_url = f'{SPACE_URL}/outputs/output_file_1.csv'
    vid_url = f'{SPACE_URL}/outputs/output_video_1.mp4'
    csv_dest = dest_folder / f'output_file_{idx}.csv'
    vid_dest = dest_folder / f'output_video_{idx}.mp4'
    print(f'  ↓ {csv_url}\n      -> {csv_dest}')
    _stream_download(csv_url, csv_dest)
    print(f'  ↓ {vid_url}\n      -> {vid_dest}')
    _stream_download(vid_url, vid_dest)


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f'Source dir does not exist: {SOURCE_DIR}', file=sys.stderr)
        return 2

    print(f'Space:  {SPACE_URL}')
    print(f'Source: {SOURCE_DIR}')
    print(f'Outputs go back into each VID{1}..VID{NUM_VIDEOS} folder.')
    print()

    # Quick reachability check so we don't try 9 huge uploads against a Space
    # that's asleep / 404 / wrong URL.
    try:
        s = requests.get(f'{SPACE_URL}/status', timeout=30)
        s.raise_for_status()
        print(f'Space /status: {s.json()}')
    except Exception as e:
        print(f'Could not reach {SPACE_URL}/status: {e}')
        print("If the Space is asleep, open it in a browser once to wake it, then re-run.")
        return 1
    print()

    summary: list[tuple[int, str]] = []
    for idx in range(1, NUM_VIDEOS + 1):
        folder = SOURCE_DIR / f'VID{idx}'
        csv_dest = folder / f'output_file_{idx}.csv'
        vid_dest = folder / f'output_video_{idx}.mp4'

        print(f'━━━ VID{idx} ━━━')
        if not folder.exists():
            print(f'  skipped — {folder} missing')
            summary.append((idx, 'missing source folder'))
            continue

        if csv_dest.exists() and vid_dest.exists():
            print(f'  skipped — results already in VID{idx}/ '
                  f'(delete output_file_{idx}.csv + output_video_{idx}.mp4 to re-run)')
            summary.append((idx, 'already done'))
            continue

        try:
            video = _find_one(folder, '.mp4')
            intrinsics = _find_one(folder, '.json')
            depth = _find_one(folder, '.bin')
        except FileNotFoundError as e:
            print(f'  skipped — {e}')
            summary.append((idx, f'missing input ({e})'))
            continue

        try:
            upload_one(idx, video, intrinsics, depth)
        except Exception as e:
            print(f'  ✗ upload failed: {e}')
            summary.append((idx, f'upload failed: {e}'))
            continue

        try:
            fetch_outputs(idx, folder)
        except Exception as e:
            # If the upload succeeded but the download didn't, the next upload
            # will overwrite the server's outputs and we lose this run. Bail
            # so the user can investigate before that happens.
            print(f'  ✗ download failed: {e}')
            print(f'    Stopping — re-run after fixing, or pull '
                  f'{SPACE_URL}/outputs/output_file_1.csv manually before next upload.')
            summary.append((idx, f'download failed: {e}'))
            break

        summary.append((idx, 'ok'))

    print()
    print('━━━ summary ━━━')
    for idx, status in summary:
        print(f'  VID{idx}: {status}')

    failed = [s for s in summary if s[1] not in ('ok', 'already done')]
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
