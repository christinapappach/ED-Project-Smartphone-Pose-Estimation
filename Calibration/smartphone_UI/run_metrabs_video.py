#!/usr/bin/env python3
"""Run MeTRAbs model on a video and save an overlayed output video.

Creates a local cache folder for the downloaded model.

Usage:
  python run_metrabs_video.py --video input.mp4 --out out.mp4

Optional args:
  --model MODEL_TYPE   (default: metrabs_mob3l_y4t)
  --skeleton SKELETON  (default: smpl_24)
  --max-frames N       (process first N frames)

This script aims to be minimal and robust on macOS. It uses OpenCV for video I/O
and matplotlib/opencv drawing for simple overlays.
"""
import argparse
import os
import ssl
import certifi
import sys
from time import time
import csv

try:
    import tensorflow as tf
except Exception as e:
    print("Error importing TensorFlow:", e)
    print("Install required packages from requirements.txt")
    raise

import numpy as np
import cv2

# Fix SSL certificate issues (helps on some macOS setups)
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context


def download_and_load_model(model_type: str, cache_dir: str):
    """Download (if needed) and load a MeTRAbs saved_model.

    Returns a loaded SavedModel object.
    """
    server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
    os.makedirs(cache_dir, exist_ok=True)

    fname = f"{model_type}_20211019.zip"
    origin = f"{server_prefix}/{model_type}_20211019.zip"

    print(f"Downloading model {model_type} to cache dir {cache_dir} (if missing)...")
    try:
        model_zip_path = tf.keras.utils.get_file(
            fname=fname,
            origin=origin,
            extract=True,
            cache_subdir='.',
            cache_dir=cache_dir
        )
    except Exception as e:
        print("Failed to download model:", e)
        raise

    base_dir = os.path.dirname(model_zip_path)

    # find saved_model.pb
    model_path = None
    for root, dirs, files in os.walk(base_dir):
        if 'saved_model.pb' in files:
            model_path = root
            break
    if model_path is None:
        raise FileNotFoundError("Could not find saved_model.pb in extracted folder")

    print(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully")
    return model


def draw_overlay(frame_bgr, boxes, poses2d, edges):
    """Draw bounding boxes and 2D skeleton joints on frame."""
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    _draw_boxes(frame, boxes, w, h)
    _draw_skeletons(frame, poses2d, edges, w, h)
    
    return frame


def _draw_boxes(frame, boxes, w, h):
    """Draw bounding boxes on frame."""
    if boxes is None or len(boxes) == 0:
        return
    
    for box in boxes:
        try:
            if len(box) == 4:
                x, y, bw, bh = box
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            elif len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except (ValueError, TypeError):
            continue


def _draw_skeletons(frame, poses2d, edges, w, h):
    """Draw skeleton joints and edges on frame."""
    if poses2d is None or len(poses2d) == 0:
        return
    
    for pose in poses2d:
        pts = pose.astype(int)
        # Draw skeleton edges
        for i_start, i_end in edges:
            if i_start < len(pts) and i_end < len(pts):
                p1, p2 = tuple(pts[i_start]), tuple(pts[i_end])
                cv2.line(frame, p1, p2, (0, 200, 255), 2)
        # Draw joint circles
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 128, 255), -1)


def process_video(input_path, output_path, model, skeleton='smpl_24', max_frames=None, csv_path=None):
    """Process video and extract 3D poses with 2D skeleton visualization."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use H.264 codec (avc1) for better macOS compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    # get joint edges from model if available
    try:
        edges_t = model.per_skeleton_joint_edges[skeleton]
        edges = edges_t.numpy().tolist()
    except Exception:
        edges = []

    # Open CSV file for pose coordinates if requested
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'person_id', 'joint_id', 'x', 'y', 'z'])

    frame_idx = 0
    t0 = time()
    print(f"Processing video {input_path} -> {output_path} @ {fps} FPS, size={width}x{height}")
    if csv_path:
        print(f"Pose coordinates will be saved to {csv_path}")
    
    try:
        _process_frames(cap, out, model, skeleton, edges, csv_writer, frame_idx, t0, max_frames)
    finally:
        cap.release()
        out.release()
        if csv_file:
            csv_file.close()
        print(f"Output saved to {output_path}")
        if csv_path:
            print(f"Pose coordinates saved to {csv_path}")


def _process_frames(cap, out, model, skeleton, edges, csv_writer, frame_idx, t0, max_frames):
    """Process video frames and extract poses."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = tf.convert_to_tensor(rgb)

        try:
            pred = model.detect_poses(image_tensor, skeleton=skeleton)
        except Exception as e:
            print(f"Frame {frame_idx}: detection failed: {e}")
            out.write(frame)
            continue

        boxes, poses2d, poses3d = _extract_poses(pred)

        # Save 3D pose coordinates to CSV
        if csv_writer and poses3d is not None:
            for person_id, pose3d in enumerate(poses3d):
                for joint_id, (x, y, z) in enumerate(pose3d):
                    csv_writer.writerow([frame_idx, person_id, joint_id, float(x), float(y), float(z)])

        # Draw overlay and write frame
        try:
            frame_out = draw_overlay(frame, boxes, poses2d, edges)
        except Exception as e:
            print(f"Frame {frame_idx}: drawing failed: {e}")
            frame_out = frame

        out.write(frame_out)

        if frame_idx % 50 == 0:
            elapsed = time() - t0
            print(f"Processed {frame_idx} frames in {elapsed:.1f}s")
    
    print(f"Finished. Processed {frame_idx} frames.")


def _extract_poses(pred):
    """Extract boxes, poses2d, and poses3d from model prediction."""
    boxes = poses2d = poses3d = None
    
    try:
        if 'boxes' in pred:
            boxes = pred['boxes'].numpy() if hasattr(pred['boxes'], 'numpy') else np.array(pred['boxes'])
    except Exception:
        pass
    
    try:
        if 'poses2d' in pred:
            poses2d = pred['poses2d'].numpy() if hasattr(pred['poses2d'], 'numpy') else np.array(pred['poses2d'])
    except Exception:
        pass
    
    try:
        if 'poses3d' in pred:
            poses3d = pred['poses3d'].numpy() if hasattr(pred['poses3d'], 'numpy') else np.array(pred['poses3d'])
    except Exception:
        pass
    
    return boxes, poses2d, poses3d


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run MeTRAbs on a video file.')
    parser.add_argument('--video', '-v', required=True, help='Input video path')
    parser.add_argument('--out', '-o', default='out.mp4', help='Output video path')
    parser.add_argument('--csv', default=None, help='Output CSV path for pose coordinates (optional)')
    parser.add_argument('--model', default='metrabs_mob3l_y4t', help='Model type to download')
    parser.add_argument('--cache', default=os.path.join(os.path.expanduser('~'), 'Documents', 'metrabs_models'), help='Cache directory for model download')
    parser.add_argument('--skeleton', default='smpl_24', help='Skeleton key to use')
    parser.add_argument('--max-frames', type=int, default=None, help='Process only first N frames')
    args = parser.parse_args(argv)

    model = download_and_load_model(args.model, cache_dir=args.cache)

    # If no CSV path specified, use default based on output video name
    csv_path = args.csv
    if csv_path is None and args.out:
        csv_path = args.out.rsplit('.', 1)[0] + '_poses.csv'

    process_video(args.video, args.out, model, skeleton=args.skeleton, max_frames=args.max_frames, csv_path=csv_path)


if __name__ == '__main__':
    main()