import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ================= CONFIG =================
SKELETON = "smpl_24"

SMPL24_NAMES = [
    'Pelvis', 'Left_Hip', 'Right_Hip', 'Spine1', 'Left_Knee', 'Right_Knee',
    'Spine2', 'Left_Ankle', 'Right_Ankle', 'Spine3', 'Left_Foot', 'Right_Foot',
    'Neck', 'Left_Shoulder', 'Right_Shoulder', 'Head', 'Left_Upper_Arm',
    'Right_Upper_Arm', 'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
    'Left_Hand', 'Right_Hand'
]

SMPL24_EDGES = [
    [0,3],[3,6],[6,9],[9,12],[12,15],
    [0,1],[1,4],[4,7],[7,10],
    [0,2],[2,5],[5,8],[8,11],
    [12,13],[13,16],[16,18],[18,20],[20,22],
    [12,14],[14,17],[17,19],[19,21],[21,23],
]

# ================= MODEL LOADING =================
_model = None

def load_model(model_path: str):
    global _model
    if _model is None:
        print(f"[MeTRAbs] Loading model from {model_path}")
        _model = tf.saved_model.load(model_path)
    return _model

# ================= CORE PROCESSING =================
def process_video(video_path: str, model_path: str, output_dir: str):
    """
    Runs MeTRAbs on a video and returns pose data.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_data = []
    frame_idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timg = tf.convert_to_tensor(rgb)

        try:
            pred = model.detect_poses(timg, skeleton=SKELETON)
        except TypeError:
            pred = model.detect_poses(timg)

        poses3d = pred["poses3d"].numpy()
        poses2d = pred["poses2d"].numpy()

        if poses3d.size > 0:
            frames_data.append({
                "frame": frame_idx,
                "poses3d": poses3d[0].tolist(),  # JSON-safe
                "poses2d": poses2d[0].tolist()
            })

        frame_idx += 1

    cap.release()

    return {
        "fps": fps,
        "num_frames": len(frames_data),
        "frames": frames_data
    }