# Integration Setup

## What changed
Only `screens/CameraScreen.js` was modified. `App.js` and `screens/HomeScreen.js` are
identical to the original. All of your teammate's recording logic is preserved exactly.

## One package to add
The calibration HUD uses `expo-sensors` for device tilt data. Run this once inside
the project folder:

```
npx expo install expo-sensors
```

(If expo-sensors is already in your node_modules from another Expo dependency, this
is a no-op — it just confirms the version.)

## Drop-in instructions
1. Copy `App.js`, `screens/HomeScreen.js`, and `screens/CameraScreen.js` into
   the existing `smartphone_UI` repo, replacing the originals.
2. Run `npx expo install expo-sensors` in the project root.
3. Run `npx expo start` as normal.

## What the calibration HUD does
- **Tilt angle (°)** — live device pitch shown in the top-centre of the camera view,
  offset by your calibration baseline (same math as the Swift app).
- **LEVEL badge** — green pill that appears when the device is within ±1° of the
  calibrated position.
- **Angle dial** — horizontal bar with a green centre line and a white dot that
  slides left/right to show deviation from level (±30° range).
- **REC timer** — red dot + MM:SS counter that appears while recording is active.
- **Calibrate button** — orange button beside the existing Record button. Tap it to
  lock the current tilt angle as the "zero" reference, just like in the Swift app.
