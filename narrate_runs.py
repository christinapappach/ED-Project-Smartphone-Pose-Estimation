"""
Narrate each VID by sampling the Pelvis trajectory at evenly-spaced
waypoints and reporting actual positions, the farthest / closest points
from the camera, and obvious direction reversals.

The previous version of this script reported tiny "net displacements" and
crazy peak speeds, both of which were misleading: net displacement is near
zero on an out-and-back walk (the subject ends near the start), and peak
speed gets blown up by single-frame jitter spikes.

This version:
  • smooths over 15 frames (0.25s @ 60 fps) — kills the jitter spikes
  • reports the *max excursion* from the start position, not just the net
  • prints waypoints at 0%, 25%, 50%, 75%, 100% of the active window so the
    user can see the actual path (which way the subject went, when they
    turned around)
  • uses median speed (not peak) for the headline number

MeTRAbs camera-frame coords, in mm:
  +X right, +Y down, +Z forward (away from camera, so smaller Z = closer)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/christinapappachan/Downloads/HUGGING FACE VIDEOS')
FPS = 60
SMOOTH_WIN = 15           # 0.25 s at 60 fps
PAUSE_SPEED_MM_S = 250
PAUSE_MIN_FRAMES = 20


def smooth(a: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    if len(a) < win:
        return a
    kernel = np.ones(win) / win
    return np.convolve(a, kernel, mode='same')


def _waypoint_label(x_mm: float, z_mm: float) -> str:
    """Human-readable position relative to the camera."""
    side = ('right' if x_mm > 100 else
            'left' if x_mm < -100 else
            'centered')
    if z_mm < 800:
        depth = 'right at camera'
    elif z_mm < 2000:
        depth = f'{z_mm/1000:.1f}m out'
    else:
        depth = f'{z_mm/1000:.1f}m back'
    return f'{side}, {depth}'


def narrate_one(idx: int, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f'\n━━━ VID{idx} ━━━ (no CSV)')
        return

    df = pd.read_csv(csv_path, skiprows=[0])
    df = df[df['joint_name'] == 'Pelvis'].sort_values('frame').reset_index(drop=True)
    if df.empty:
        print(f'\n━━━ VID{idx} ━━━ — no Pelvis rows')
        return

    px = df['x'].astype(float).to_numpy()
    py = df['y'].astype(float).to_numpy()
    pz = df['z'].astype(float).to_numpy()
    frames = df['frame'].astype(int).to_numpy()

    sx, sy, sz = smooth(px), smooth(py), smooth(pz)

    # Per-step ground-plane distance and speed.
    dt = np.diff(frames) / FPS
    dt = np.where(dt > 0, dt, 1 / FPS)
    dx = np.diff(sx); dz = np.diff(sz)
    step_dist = np.sqrt(dx ** 2 + dz ** 2)
    speed = step_dist / dt

    total_path_mm = float(step_dist.sum())
    duration_s = float((frames[-1] - frames[0]) / FPS)
    median_speed_mm_s = float(np.median(speed))
    p95_speed_mm_s = float(np.percentile(speed, 95))   # skip the spike outliers

    # Max excursion from start position — answers "how far out did they go?"
    start_x, start_z = sx[0], sz[0]
    excursion = np.sqrt((sx - start_x) ** 2 + (sz - start_z) ** 2)
    max_excursion_mm = float(excursion.max())
    max_excursion_idx = int(excursion.argmax())
    max_excursion_frame = int(frames[max_excursion_idx])

    # Z-axis extremes: closest to camera and farthest from camera in the run.
    nearest_z_mm = float(sz.min()); nearest_idx = int(sz.argmin())
    farthest_z_mm = float(sz.max()); farthest_idx = int(sz.argmax())

    # Waypoints at 0/25/50/75/100% of the active window.
    n = len(frames)
    waypoint_idx = [0, n // 4, n // 2, (3 * n) // 4, n - 1]

    # Pauses: ground-plane speed below threshold for at least N frames.
    pauses = []
    run_start = None
    for i, s in enumerate(speed):
        if s < PAUSE_SPEED_MM_S:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= PAUSE_MIN_FRAMES:
                pauses.append((int(frames[run_start]), int(frames[i]),
                               (i - run_start) / FPS))
            run_start = None
    if run_start is not None and (len(speed) - run_start) >= PAUSE_MIN_FRAMES:
        pauses.append((int(frames[run_start]), int(frames[-1]),
                       (len(speed) - run_start) / FPS))

    print(f'\n━━━ VID{idx} ━━━')
    print(f'  Active window: frame {frames[0]}–{frames[-1]} '
          f'(t = {frames[0]/FPS:.1f}s → {frames[-1]/FPS:.1f}s, {duration_s:.1f}s)')

    print(f'  Waypoints (smoothed Pelvis positions, in m):')
    for pct, i in zip([0, 25, 50, 75, 100], waypoint_idx):
        x_m, z_m = sx[i] / 1000, sz[i] / 1000
        f = frames[i]
        print(f'    {pct:>3}%  frame {f:>4} (t={f/FPS:5.1f}s)   '
              f'({x_m:+.2f}, {z_m:+.2f}) m   ← {_waypoint_label(sx[i], sz[i])}')

    print(f'  Closest to camera: Z={nearest_z_mm/1000:.2f}m at frame {frames[nearest_idx]} '
          f'(t={frames[nearest_idx]/FPS:.1f}s)')
    print(f'  Farthest from camera: Z={farthest_z_mm/1000:.2f}m at frame {frames[farthest_idx]} '
          f'(t={frames[farthest_idx]/FPS:.1f}s)')
    print(f'  Max distance from start position: {max_excursion_mm/1000:.2f}m '
          f'at frame {max_excursion_frame} (t={max_excursion_frame/FPS:.1f}s)')
    print(f'  Total path walked: {total_path_mm/1000:.2f}m')
    print(f'  Walking speed: median {median_speed_mm_s/1000:.2f} m/s, '
          f'95th-pctile {p95_speed_mm_s/1000:.2f} m/s')
    if pauses:
        print(f'  Pauses ({len(pauses)}):')
        for sf, ef, dur in pauses:
            print(f'    - frame {sf}–{ef} (t = {sf/FPS:.1f}s, {dur:.1f}s long)')

    # One-line story.
    z_range = farthest_z_mm - nearest_z_mm
    walks_back = z_range > 1500   # >1.5m of depth movement = significant out/in
    if walks_back:
        # Identify the order: did they walk away first, or come closer first?
        # Compare frame indices of extremes.
        if farthest_idx > nearest_idx:
            order = (f"started ~{nearest_z_mm/1000:.1f}m from camera, "
                     f"walked back to ~{farthest_z_mm/1000:.1f}m "
                     f"(at t={frames[farthest_idx]/FPS:.1f}s), "
                     f"then returned toward camera")
        else:
            order = (f"started ~{farthest_z_mm/1000:.1f}m back, "
                     f"walked forward to ~{nearest_z_mm/1000:.1f}m "
                     f"(at t={frames[nearest_idx]/FPS:.1f}s), "
                     f"then walked back")
    else:
        order = f'stayed within ~{z_range/1000:.1f}m depth range'
    pause_note = ''
    if pauses:
        longest = max(pauses, key=lambda p: p[2])
        pause_note = f'; paused {len(pauses)}× (longest {longest[2]:.1f}s at t={longest[0]/FPS:.1f}s)'
    print(f'  → Subject {order}, total path {total_path_mm/1000:.1f}m '
          f'in {duration_s:.1f}s at ~{median_speed_mm_s/1000:.1f} m/s{pause_note}.')


def main() -> int:
    print(f'Reading CSVs from {ROOT}')
    print(f'(camera frame: +X right, +Y down, +Z forward; smoothing 15 frames)')
    for idx in range(1, 10):
        csv_path = ROOT / f'VID{idx}' / f'output_file_{idx}.csv'
        narrate_one(idx, csv_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
