# SmartPhone Pose Estimation

iOS app that records 60 fps ARKit video + camera intrinsics, uploads to a
MeTRAbs pose estimation backend, and saves 3D pose CSVs on device.

FAU Engineering Design research project (EngDesFAU26).

---

## Quick start (new teammate)

**You need:** a Mac with Xcode, an iPhone, and a free Apple ID.

```bash
git clone <this-repo-url>
cd smartphone_UI
python start.py
```

That's it. `start.py` will:
1. Check Xcode / Node 20+ / CocoaPods are installed
2. `npm install` dependencies
3. `pod install` (with the Ruby UTF-8 locale fix)
4. Find your connected iPhone
5. Build and install the app via `xcodebuild` + `devicectl`

First run takes ~5 min. Subsequent rebuilds: `python start.py --skip-pods --skip-npm`
(~1 min).

### Troubleshooting

| Error | Fix |
|---|---|
| `Node vXX is too old` | `source ~/.nvm/nvm.sh && nvm use 20` |
| `CocoaPods not found` | `sudo gem install cocoapods` |
| `No iPhone detected` | Plug in, unlock, tap "Trust this Mac" |
| `Provisioning profile...` fails | Open `ios/FirstApp.xcworkspace` in Xcode → select FirstApp target → Signing & Capabilities → sign in with your Apple ID |
| App won't launch on phone | Settings → General → VPN & Device Management → trust developer profile |

---

## Architecture

```
iPhone (ARKit capture)                HF Space (FastAPI + MeTRAbs)
  │                                     │
  ├── records 60fps HEVC video           ├── /upload        ← receives video + intrinsics
  ├── captures camera intrinsics JSON    ├── /               ← minimal web UI for testing
  │                                       ├── /status         ← health check
  │                                       └── /outputs/*      ← CSV + annotated video
  │
  └── POST /upload ─────────────────→
       ← returns csv_url, output_url, error_analysis
```

- **Backend**: `hf_space/` — Python FastAPI server, deployed to
  [Hugging Face Spaces](https://huggingface.co/spaces/EngDesFAU26-SmartphonePose/Metrabs_server)
- **MeTRAbs model**: mirrored to
  [huggingface.co/EngDesFAU26-SmartphonePose/metrabs-mob3l-y4t](https://huggingface.co/EngDesFAU26-SmartphonePose/metrabs-mob3l-y4t)
  for fast cold starts
- **Frontend**: React Native / Expo + custom Swift ARKit module (`modules/ARCapture/`)

---

## Project layout

```
smartphone_UI/
├── start.py              ← one-command setup (run this first)
├── app.json              ← Expo app config (name, icon, bundle id)
├── App.js                ← root component
├── screens/
│   └── CameraScreen.js   ← main UI, record + upload logic
├── modules/
│   └── ARCapture/        ← native Swift module for ARKit recording
├── assets/
│   └── icon.png          ← app icon
├── ios/
│   ├── FirstApp/         ← native project (Info.plist, icon, etc.)
│   └── FirstApp.xcworkspace
├── hf_space/             ← backend that gets deployed to HF Spaces
│   ├── app.py            ← FastAPI server
│   ├── error_analysis.py ← STUB — teammate fills in analyze()
│   ├── Dockerfile
│   └── requirements.txt
└── MeTRAbs_Colab_Server.ipynb   ← legacy Colab version (kept for reference)
```

---

## Backend development

The backend lives at `hf_space/`. To iterate on it:

### Test locally (no iPhone needed)
```bash
cd hf_space
pip install -r requirements.txt
python app.py
# open http://localhost:7860
```

### Deploy changes to HF
```bash
cd hf_space
# edit app.py / error_analysis.py
hf upload --repo-type=space EngDesFAU26-SmartphonePose/Metrabs_server \
    . . --commit-message "your message"
```

Or clone the Space as a git repo and push normally.

### Test with curl / Python
```bash
cd hf_space
python test_client.py some_video.mp4
```

---

## Team access

All 5 org members at
[huggingface.co/EngDesFAU26-SmartphonePose](https://huggingface.co/EngDesFAU26-SmartphonePose)
have write access to the Space and model repos.

For iOS distribution beyond dev-install: each teammate uses their free
Apple ID with `python start.py`. App signatures expire every 7 days on
free accounts — just run `start.py` again to refresh.

For longer signatures + TestFlight distribution, one team member would
need to join the Apple Developer Program ($99/yr).
