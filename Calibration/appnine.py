#!/usr/bin/env python3
"""
appnine.py — Batch-upload N recording folders to the MeTRAbs HF Space and
download all their result CSVs in one go.

Use this for validation experiments: you have 5 ft / 7 ft / 10 ft × 3 trials
worth of recordings sitting in folders, and you want each one processed by the
server with a unique output filename (no overwrites), without manually hitting
Upload nine times in the iOS app.

Each input folder must contain at minimum:
    video.mp4

And optionally (for full ARKit + LiDAR processing):
    intrinsics.json     — ARKit camera metadata (full intrinsics blob)
    depth.bin           — per-frame LiDAR depth, iPhone-only

Usage
-----
    python appnine.py <input_root_dir> <output_dir>

Where <input_root_dir> contains one subfolder per recording, e.g.:

    recordings/
        5ft_trial1/   video.mp4   depth.bin   intrinsics.json
        5ft_trial2/   video.mp4   depth.bin   intrinsics.json
        5ft_trial3/   ...
        7ft_trial1/   ...
        ...
        10ft_trial3/  ...

Output:  <output_dir>/<folder_name>.csv  for each input folder
         <output_dir>/_summary.json      run stats (elapsed time, detection
                                         counts, lidar usage flag, etc.)

Sequential by design — the HF Space holds one GPU lock at a time, so we wait
for each clip to finish before starting the next.

Defaults to the public Space URL, override with --server if you point at a
different deployment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


SERVER_URL = "https://engdesfau26-smartphonepose-metrabs-server.hf.space"


def upload_one(folder: Path, server: str) -> tuple[str | None, dict | str]:
    """POST one recording to the server's /upload endpoint and pull the CSV.

    Returns (csv_text, info_dict) on success, or (None, error_string) on failure.
    """
    video = folder / "video.mp4"
    if not video.exists():
        # Some folders use the original timestamped name from the iOS app.
        candidates = list(folder.glob("metrabs_*.mp4")) + list(folder.glob("*.mp4"))
        if not candidates:
            return None, f"no .mp4 found in {folder}"
        video = candidates[0]

    depth = folder / "depth.bin"
    if not depth.exists():
        candidates = list(folder.glob("*.depth.bin")) + list(folder.glob("*depth*.bin"))
        depth = candidates[0] if candidates else None

    intrinsics_path = folder / "intrinsics.json"
    if not intrinsics_path.exists():
        candidates = list(folder.glob("*.json"))
        intrinsics_path = candidates[0] if candidates else None

    files = {"file": ("video.mp4", open(video, "rb"), "video/mp4")}
    data: dict[str, str] = {}

    if intrinsics_path and intrinsics_path.exists():
        try:
            blob = intrinsics_path.read_text()
            data["intrinsics_json"] = blob
            obj = json.loads(blob)
            fl = obj.get("camera_intrinsics", {}).get("focal_length", {})
            pp = obj.get("camera_intrinsics", {}).get("principal_point", {})
            for k_in, k_out in (("fx", "focal_length_x"), ("fy", "focal_length_y")):
                if k_in in fl:
                    data[k_out] = str(fl[k_in])
            for k_in, k_out in (("cx", "principal_point_x"), ("cy", "principal_point_y")):
                if k_in in pp:
                    data[k_out] = str(pp[k_in])
        except Exception as e:
            print(f"    (intrinsics parse warning: {e})")

    if depth and depth.exists():
        files["depth_data"] = ("depth.bin", open(depth, "rb"), "application/octet-stream")

    n_inputs = len(files) + (1 if "intrinsics_json" in data else 0)
    sizes_mb = sum(Path(p).stat().st_size for p in (video, depth) if p and p.exists()) // 1024 // 1024
    print(f"    sending {n_inputs} fields, {sizes_mb} MB total ...")

    t0 = time.time()
    try:
        resp = requests.post(f"{server.rstrip('/')}/upload", files=files, data=data, timeout=7200)
    finally:
        for tup in files.values():
            try:
                tup[1].close()
            except Exception:
                pass

    elapsed = time.time() - t0
    if not resp.ok:
        return None, f"HTTP {resp.status_code}: {resp.text[:400]}"

    try:
        j = resp.json()
    except Exception:
        return None, f"non-JSON response: {resp.text[:400]}"

    csv_url = j.get("csv_url")
    if not csv_url:
        return None, f"no csv_url in response: {j}"

    csv_resp = requests.get(f"{server.rstrip('/')}{csv_url}", timeout=180)
    if not csv_resp.ok:
        return None, f"CSV fetch HTTP {csv_resp.status_code}"

    info = {
        "elapsed_s": round(elapsed, 1),
        "frames_with_detections": j.get("frames_with_detections"),
        "used_intrinsics": j.get("used_intrinsics"),
        "used_lidar_depth": j.get("used_lidar_depth"),
        "server_processing_seconds": j.get("processing_time_seconds"),
    }
    return csv_resp.text, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_root", help="Directory containing per-recording subfolders")
    ap.add_argument("output_dir", help="Where to write the result CSVs")
    ap.add_argument("--server", default=SERVER_URL, help=f"HF Space URL (default: {SERVER_URL})")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        sys.exit(f"input_root not found: {in_root}")

    folders = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not folders:
        sys.exit(f"no subfolders found in {in_root}")

    print(f"Found {len(folders)} recording folder(s) in {in_root}:")
    for p in folders:
        print(f"  • {p.name}")
    print(f"Server: {args.server}")
    print(f"Output: {out_dir}/")
    print()

    summary: dict[str, dict] = {}
    started_at = time.time()
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder.name}")
        csv_text, info = upload_one(folder, args.server)
        if csv_text is None:
            print(f"    ❌ {info}")
            summary[folder.name] = {"status": "failed", "error": str(info)}
            continue
        out_csv = out_dir / f"{folder.name}.csv"
        out_csv.write_text(csv_text)
        n_rows = csv_text.count("\n") - 2  # subtract 2-row header
        print(
            f"    ✓ {info['elapsed_s']}s "
            f"({info['frames_with_detections']} detections, "
            f"lidar={info['used_lidar_depth']})"
        )
        summary[folder.name] = {
            "status": "ok",
            "csv": out_csv.name,
            "rows": n_rows,
            **info,
        }

    total_elapsed = time.time() - started_at
    summary["_total_elapsed_seconds"] = round(total_elapsed, 1)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done in {total_elapsed/60:.1f} min. Wrote {len(folders)} CSVs to {out_dir}/")
    print(f"Summary: {out_dir}/_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
