"""
Read every output_file_*.csv produced by the MeTRAbs batch run and report:
  • a running tally per VID — same shape as the server's "Processed X/Y
    frames (N w/ detections)" log lines
  • the frame the subject first enters (first detected frame)
  • the frame they leave (last detected frame)
  • detection rate during the active window vs. overall

The CSVs use a 2-row header (unit-group row + x/y/z row), so we skip the
first row and let the second be the column header.

A frame counts as "with detections" if any joint row exists for it. The
SMPL-24 skeleton has 24 joints, so each detected frame contributes 24 rows
to the CSV; no rows means no pose detected for that frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/Users/christinapappachan/Downloads/HUGGING FACE VIDEOS')
JOINTS_PER_FRAME = 24       # SMPL-24
BUCKET = 25                 # match server log granularity

# Total frame counts per video — pulled from the server logs we have.
# If a VID is missing here, the script falls back to "max detected frame + 1"
# which is correct except for tail-end frames with no detection.
KNOWN_TOTALS = {
    1: 834, 2: 749, 3: 805, 4: 758, 5: 757,
    6: 805, 7: 650, 8: 749, 9: 749,
}


def analyze_one(idx: int, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f'  {csv_path.name}: MISSING')
        return

    # First row is the unit-group row (mm,,,meters,,,...). Real header is
    # row 2 (frame, joint_idx, joint_name, x, y, z, ...). pandas sees both
    # as candidate headers; skiprows=[0] drops the unit-group row.
    df = pd.read_csv(csv_path, skiprows=[0])
    if df.empty:
        print(f'  VID{idx}: CSV has no data rows')
        return

    # `frame` column is the integer frame index from the source video. Same
    # frame number repeats up to 24 times (one per joint).
    frames = sorted(set(int(f) for f in df['frame']))
    detected_set = set(frames)

    total_frames = KNOWN_TOTALS.get(idx, max(frames) + 1)
    n_detected = len(detected_set)
    first = min(frames)
    last = max(frames)
    rate_overall = n_detected / total_frames * 100
    active_window = last - first + 1
    rate_active = n_detected / active_window * 100 if active_window else 0

    print(f'\n━━━ VID{idx} ━━━ ({n_detected}/{total_frames} frames detected, {rate_overall:.0f}% overall)')

    # Running tally — print only the buckets where the count actually
    # changes, so noise is filtered out. "Person enters" arrow on the first
    # bucket where count goes from 0 → positive, "leaves" arrow on the last
    # bucket where new detections stop arriving.
    running = 0
    last_count = 0
    last_arrow_frame = None
    for boundary in range(BUCKET, total_frames + BUCKET, BUCKET):
        boundary = min(boundary, total_frames)
        in_bucket = sum(1 for f in detected_set if (boundary - BUCKET) < f <= boundary)
        running += in_bucket
        arrow = ''
        if last_count == 0 and running > 0:
            arrow = '   ← person enters'
            last_arrow_frame = boundary
        elif in_bucket == 0 and last_count > 0 and running == last_count and running < n_detected:
            # Empty bucket after the person was visible — they may have left.
            # Only flag the *first* such gap; later gaps could be brief drops.
            pass
        print(f'  Processed {boundary:4d}/{total_frames} frames ({running:4d} w/ detections){arrow}')
        last_count = running
        if boundary >= total_frames:
            break

    print(f'  first detected frame: {first}  ({first/60:.1f}s @ 60fps)')
    print(f'  last  detected frame: {last}   ({last/60:.1f}s @ 60fps)')
    print(f'  active window: frames {first}–{last} = {active_window} frames; '
          f'detection rate inside that window = {rate_active:.1f}%')


def main() -> int:
    print(f'Reading from {ROOT}')
    print(f'Total frames per VID (from server logs): {KNOWN_TOTALS}\n')

    summary: list[tuple[int, int, int, int, float]] = []
    for idx in range(1, 10):
        csv_path = ROOT / f'VID{idx}' / f'output_file_{idx}.csv'
        if not csv_path.exists():
            continue
        analyze_one(idx, csv_path)

        df = pd.read_csv(csv_path, skiprows=[0])
        frames = sorted(set(int(f) for f in df['frame']))
        if frames:
            total = KNOWN_TOTALS.get(idx, max(frames) + 1)
            summary.append((idx, total, len(set(frames)), min(frames), max(frames)))

    # Side-by-side summary at the end so the user can eyeball all 9 at once.
    print('\n\n═══ SUMMARY ═══')
    print(f'{"VID":<5}{"frames":>8}{"detected":>10}{"%":>6}'
          f'{"first":>8}{"last":>8}{"window%":>10}')
    for idx, total, det, first, last in summary:
        win = last - first + 1
        win_pct = det / win * 100 if win else 0
        overall = det / total * 100
        print(f'{idx:<5}{total:>8}{det:>10}{overall:>6.0f}'
              f'{first:>8}{last:>8}{win_pct:>9.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
