# metrabs_server.py
# The ONLY thing that lives on your server.
# Receives the video + camera intrinsics from the iPhone, runs MeTRAbs, returns 3D poses.
#
# Install deps:
#   pip install flask tensorflow
#   (plus whatever MeTRAbs itself needs — see its README)
#
# Run:
#   python metrabs_server.py

import tensorflow as tf
import numpy as np
import cv2
import json
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

# ── Load MeTRAbs model once at startup (not per-request) ──────────────────────
print("Loading MeTRAbs model...")
model = tf.saved_model.load("path/to/metrabs_model")  # ⚠️ point this at your model folder
print("✅ Model ready")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── 1. Pull the camera intrinsics the iPhone sent as form fields ───────────
    # These are the 4 numbers that tell MeTRAbs about the camera lens.
    # The iPhone already calculated them — we just pass them straight through.
    try:
        fx = float(request.form["focal_length_x"])
        fy = float(request.form["focal_length_y"])
        cx = float(request.form["principal_point_x"])
        cy = float(request.form["principal_point_y"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or bad intrinsics: {e}"}), 400

    print(f"📐 Intrinsics received: fx={fx} fy={fy} cx={cx} cy={cy}")

    # ── 2. Save the video to a temp file ──────────────────────────────────────
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files["video"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_file.save(tmp.name)
    print(f"💾 Video saved to temp: {tmp.name}")

    # ── 3. Extract frames from the video ──────────────────────────────────────
    cap = cv2.VideoCapture(tmp.name)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MeTRAbs expects RGB, OpenCV gives BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    os.unlink(tmp.name)  # clean up temp file

    print(f"🎞️  Extracted {len(frames)} frames")

    if not frames:
        return jsonify({"error": "Could not read any frames from video"}), 400

    # ── 4. Build the intrinsics matrix MeTRAbs expects ────────────────────────
    # MeTRAbs wants a 3x3 camera matrix:
    #   [[fx,  0, cx],
    #    [ 0, fy, cy],
    #    [ 0,  0,  1]]
    intrinsic_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

    # ── 5. Run MeTRAbs frame by frame ─────────────────────────────────────────
    # (Adjust this call to match whichever MeTRAbs API version you're using)
    all_poses = []

    for i, frame in enumerate(frames):
        frame_tensor = tf.constant(frame[np.newaxis], dtype=tf.uint8)  # add batch dim

        # Pass intrinsics so MeTRAbs outputs real-world meters, not relative depth
        result = model.detect_poses(
            frame_tensor,
            intrinsic_matrix=intrinsic_matrix[np.newaxis],  # batch dim
            skeleton="coco_19"   # or "smpl_24", "h36m_17" — whatever your backend uses
        )

        # result["poses3d"] shape: (batch, num_people, num_joints, 3)
        poses = result["poses3d"].numpy().tolist()
        all_poses.append({
            "frame": i,
            "poses_3d": poses
        })

        if i % 60 == 0:
            print(f"  Processed frame {i}/{len(frames)}")

    print("✅ All frames processed")

    # ── 6. Return results ─────────────────────────────────────────────────────
    return jsonify({
        "frame_count": len(frames),
        "results": all_poses
    })


if __name__ == "__main__":
    # 0.0.0.0 makes it reachable from the iPhone on the same network
    app.run(host="0.0.0.0", port=8000, debug=False)
