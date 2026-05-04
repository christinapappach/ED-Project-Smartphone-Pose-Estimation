# ED Project — Smartphone Pose Estimation

iPhone-based 3D human pose capture pipeline for the **FAU EngDesFAU26** Engineering Design research project. Records 60 fps ARKit video + camera intrinsics on iPhone, uploads each capture to a [MeTRAbs](https://github.com/isarandi/metrabs) backend running on a Hugging Face Space, and analyzes the resulting 3D pose CSVs locally.

This repo bundles three things:

| Folder / file | What it is |
|---|---|
| `Calibration/Calibration.xcodeproj` | Native Swift/ARKit capture app (`Calibration` target) |
| `Calibration/smartphone_UI/` | React Native variant of the capture app (slightly modified fork of [christinapappach/smartphone-pose-estimation](https://github.com/christinapappach/smartphone-pose-estimation)) |
| `nine.py`, `analyze_csvs.py`, `narrate_runs.py`, `upload_all.py` | MeTRAbs server + local analysis scripts |

---

## Prerequisites (macOS)

You need a Mac. Install once:

```bash
# 1. Xcode + command line tools (from the App Store, then:)
xcode-select --install

# 2. Homebrew  (https://brew.sh)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Python 3.10+, Node 20+, CocoaPods
brew install python@3.11 node@20 cocoapods
```

Verify:

```bash
xcodebuild -version          # Xcode 15+
python3 --version            # 3.10+
node --version               # v20+
pod --version                # 1.14+
```

---

## 1. Clone

```bash
git clone https://github.com/christinapappach/ED-Project-Smartphone-Pose-Estimation.git
cd ED-Project-Smartphone-Pose-Estimation
```

---

## 2. Build the native `Calibration` iOS app (no Xcode UI required)

The simpler ARKit capture app, located in `Calibration/`.

### Build for the iOS Simulator

```bash
cd Calibration
xcodebuild \
  -project Calibration.xcodeproj \
  -scheme Calibration \
  -configuration Debug \
  -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  build
```

### Run on the Simulator

```bash
# Boot a simulator (first time only)
xcrun simctl boot "iPhone 15 Pro" || true
open -a Simulator

# Install + launch the freshly-built app
APP_PATH=$(xcodebuild -project Calibration.xcodeproj -scheme Calibration -configuration Debug -sdk iphonesimulator -showBuildSettings | awk -F'= ' '/ TARGET_BUILD_DIR /{print $2}')
xcrun simctl install booted "$APP_PATH/Calibration.app"
xcrun simctl launch booted Calibration.Calibration   # replace with your bundle id if different
```

> **Note:** ARKit + LiDAR features only work on a **physical device** (LiDAR-equipped iPhone Pro / iPad Pro). The simulator build is for UI smoke-testing only.

### Build + install on a connected iPhone

```bash
# Plug in your iPhone, trust the Mac, then:
xcrun devicectl list devices                                    # find your device UDID
DEVICE_ID="<paste-udid-here>"

xcodebuild \
  -project Calibration.xcodeproj \
  -scheme Calibration \
  -configuration Debug \
  -destination "id=$DEVICE_ID" \
  -allowProvisioningUpdates \
  build

# Install the .app onto the phone
APP_PATH=$(xcodebuild -project Calibration.xcodeproj -scheme Calibration -configuration Debug -showBuildSettings -destination "id=$DEVICE_ID" | awk -F'= ' '/ TARGET_BUILD_DIR /{print $2}')
xcrun devicectl device install app --device "$DEVICE_ID" "$APP_PATH/Calibration.app"
```

> **First time only:** open `Calibration.xcodeproj` in Xcode once to pick a Team under **Signing & Capabilities**, then close it. After that, `-allowProvisioningUpdates` handles the rest from the terminal.

---

## 3. Build + run the React Native `smartphone_UI` app

This is the richer UI with the full HF/MeTRAbs upload flow. It has its own one-shot installer:

```bash
cd Calibration/smartphone_UI
python start.py
```

`start.py` will:
1. Verify Xcode / Node 20+ / CocoaPods are installed
2. `npm install`
3. `pod install` (with the Ruby UTF-8 locale fix)
4. Detect your connected iPhone
5. Build and install via `xcodebuild` + `devicectl`

First run takes ~5 min. Rebuild only:

```bash
python start.py --skip-pods --skip-npm   # ~1 min
```

See [`Calibration/smartphone_UI/README.md`](Calibration/smartphone_UI/README.md) for troubleshooting.

---

## 4. Run the Python MeTRAbs analysis scripts

These run locally and talk to the MeTRAbs Hugging Face Space.

```bash
# From the repo root:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Script | What it does |
|---|---|
| `upload_all.py` | Sequentially uploads `VID1..VID9` from a capture folder to the MeTRAbs HF Space and saves each output CSV |
| `analyze_csvs.py` | Reads every `output_file_*.csv` and reports per-VID tallies + summary stats |
| `narrate_runs.py` | Samples the Pelvis trajectory at evenly-spaced waypoints and narrates each run |
| `nine.py` | The MeTRAbs server itself (BATCH-OF-9 variant) — runs **on the Hugging Face Space**, not locally |

Point `upload_all.py` at your own Space if you have one:

```bash
export HF_SPACE_URL="https://<your-space>.hf.space"
python upload_all.py
```

The default Space is `https://engdesfau26-smartphonepose-metrabs-server.hf.space`.

---

## Repo layout

```
.
├── Calibration/
│   ├── Calibration.xcodeproj/        # Native iOS app (Swift + ARKit)
│   ├── Calibration/                  # Swift sources, Info.plist, assets
│   ├── ARCaptureManager.swift
│   ├── MetrabsUploader.swift
│   ├── metrabs_server.py             # Reference server (lighter than nine.py)
│   └── smartphone_UI/                # React Native variant — see its own README
├── nine.py                           # MeTRAbs HF Space server (BATCH-OF-9)
├── upload_all.py                     # Batch uploader for VID1..VID9
├── analyze_csvs.py                   # CSV analyzer
├── narrate_runs.py                   # Trajectory narrator
├── requirements.txt
└── README.md
```

---

## Credits

- Original React Native capture app: [christinapappach/smartphone-pose-estimation](https://github.com/christinapappach/smartphone-pose-estimation)
- MeTRAbs pose estimation: [isarandi/metrabs](https://github.com/isarandi/metrabs)
- FAU Engineering Design research project (EngDesFAU26)
