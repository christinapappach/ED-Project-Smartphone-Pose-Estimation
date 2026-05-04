"""
Minimal test client for the MeTRAbs HF Space.

Uploads a video (optionally with ARKit intrinsics JSON) and saves the
returned CSV locally so you can feed it into your error analysis pipeline.

Usage:
    python test_client.py path/to/video.mp4
    python test_client.py path/to/video.mp4 --intrinsics intrinsics.json
"""
import argparse
import json
import sys
from pathlib import Path

import requests

SERVER = "https://engdesfau26-smartphonepose-metrabs-server.hf.space"


def upload(video_path: Path, intrinsics_path: Path | None) -> dict:
    with open(video_path, "rb") as vf:
        files = {"file": (video_path.name, vf, "video/mp4")}
        data = {}
        if intrinsics_path and intrinsics_path.exists():
            data["intrinsics_json"] = intrinsics_path.read_text()
        print(f"[client] POST {SERVER}/upload  (this takes 3-5 min on CPU)")
        r = requests.post(f"{SERVER}/upload", files=files, data=data, timeout=900)
    r.raise_for_status()
    return r.json()


def download(url_path: str, out_path: Path) -> None:
    full_url = f"{SERVER}{url_path}"
    print(f"[client] GET  {full_url}")
    r = requests.get(full_url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"[client] saved -> {out_path}  ({len(r.content):,} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="path to .mp4")
    ap.add_argument("--intrinsics", type=Path, help="optional ARKit JSON")
    ap.add_argument("--out-dir", type=Path, default=Path("./results"))
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = upload(args.video, args.intrinsics)
    print(json.dumps(result, indent=2))

    stem = args.video.stem
    if result.get("csv_url"):
        download(result["csv_url"], args.out_dir / f"{stem}.csv")
    if result.get("output_url"):
        download(result["output_url"], args.out_dir / f"{stem}_annotated.mp4")

    print("\n[client] done. feed the CSV into your error analysis script.")


if __name__ == "__main__":
    main()
