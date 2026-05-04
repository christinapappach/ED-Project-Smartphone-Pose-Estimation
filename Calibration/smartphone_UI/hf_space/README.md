---
title: MeTRAbs Pose Estimation Server
emoji: 🧍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MeTRAbs Pose Estimation Server

3D pose estimation server using [MeTRAbs](https://github.com/isarandi/metrabs) with iPhone ARKit camera intrinsics support.

## Usage

### REST API (used by the iOS app)

```
POST /upload
Content-Type: multipart/form-data

- file: <video.mp4>
- intrinsics_json: <ARKit JSON blob>  (optional)
- focal_length_x, focal_length_y, principal_point_x, principal_point_y: floats (optional, alternative to JSON)
```

Returns JSON with `csv_url`, `output_url`, `used_intrinsics`, `processing_time_seconds`.

### Gradio UI

Visit `/ui` for a web interface to test uploads.

### Health check

`GET /` returns `{"message": "MeTRAbs server running", ...}`

## Developer notes

- `app.py` — FastAPI + Gradio, MeTRAbs with `@spaces.GPU` for ZeroGPU allocation
- Model is downloaded on first boot (cached in container), `metrabs_mob3l_y4t` with `smpl_24` skeleton
- Processes video at 0.25x resolution, every 2nd frame (matches teammate's CPU server settings)
- Intrinsic matrix is scaled by resize factor before being passed to `detect_poses`

## Team

Part of the **Smartphone Pose Estimation — EngDesFAU26** project (FAU Emergency Department calibration research).

- Org: `EngDesFAU26-SmartphonePose` on Hugging Face
- Space: `EngDesFAU26-SmartphonePose/Metrabs_server`
- Public endpoint: `https://engdesfau26-smartphonepose-metrabs-server.hf.space`

All 5 devs on the org with `write` role can edit `app.py` via the web editor or `git clone` + push.
