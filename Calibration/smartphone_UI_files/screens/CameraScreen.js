// ─────────────────────────────────────────────────────────────────────────────
// CameraScreen.js
//
// Teammate's original camera/recording code is fully preserved below.
// Christina's calibration HUD (tilt angle, LEVEL badge, angle dial, REC timer,
// and Calibrate button) has been added as an overlay on the camera view using
// expo-sensors DeviceMotion — mirroring the Swift app's behaviour.
// ─────────────────────────────────────────────────────────────────────────────

import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import { shareAsync } from 'expo-sharing';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useVideoPlayer, VideoView } from 'expo-video';
import { DeviceMotion } from 'expo-sensors'; // <-- added for tilt calibration

// ── Calibration constants ─────────────────────────────────────────────────────
const LEVEL_THRESHOLD = 1.0;   // degrees within which device is considered "level"
const DIAL_HALF_WIDTH = 110;   // half the dial bar width in points
const DIAL_MAX_DEG    = 30;    // angle (°) that corresponds to the dial edge

export default function CameraScreen() {

  // ── Teammate's original state (unchanged) ──────────────────────────────────
  const [hasCameraPermission,    setHasCameraPermission]    = useState(null);
  const [hasMediaPermission,     setHasMediaPermission]     = useState(null);
  const [hasMicrophonePermission,setHasMicrophonePermission]= useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [video,       setVideo]       = useState(null);
  const [type,        setType]        = useState('back');
  const cameraReference = useRef(null);

  // Teammate's original permission + init effect (unchanged)
  useEffect(() => {
    (async () => {
      MediaLibrary.requestPermissionsAsync();
      const cameraStatus     = await Camera.requestCameraPermissionsAsync();
      const microphoneStatus = await Camera.requestMicrophonePermissionsAsync();
      const mediaPermission  = await MediaLibrary.requestPermissionsAsync();

      setHasCameraPermission(cameraStatus.status === 'granted');
      setHasMicrophonePermission(microphoneStatus.status === 'granted');
      setHasMediaPermission(mediaPermission.status === 'granted');
    })();
  }, []);

  // Teammate's original video player (unchanged)
  const player = useVideoPlayer(video, p => {
    p.loop = true;
    p.play();
  });

  // Teammate's original recording functions (unchanged)
  let recordVideo = async () => {
    setIsRecording(true);
    let options = { mute: false, maxDuration: 60 };
    cameraReference.current.recordAsync(options).then((recordedVideo) => {
      setVideo(recordedVideo);
      setIsRecording(false);
    });
  };

  let stopRecording = async () => {
    setIsRecording(false);
    cameraReference.current.stopRecording();
  };

  // ── Calibration state ──────────────────────────────────────────────────────
  const [rawPitch,          setRawPitch]          = useState(0);
  const [calibrationOffset, setCalibrationOffset] = useState(0);
  const [recordingSeconds,  setRecordingSeconds]  = useState(0);
  const timerRef = useRef(null);

  // Derived calibration values
  const correctedPitch = rawPitch - calibrationOffset;
  const isLevel        = Math.abs(correctedPitch) < LEVEL_THRESHOLD;

  // Subscribe to device motion for pitch angle (mirrors ARKit pitch in Swift app)
  useEffect(() => {
    DeviceMotion.setUpdateInterval(100); // 10 Hz — smooth enough for UI
    const sub = DeviceMotion.addListener((data) => {
      if (data.rotation) {
        // beta = pitch around X-axis, in radians → convert to degrees
        const pitchDeg = (data.rotation.beta * 180) / Math.PI;
        setRawPitch(pitchDeg);
      }
    });
    return () => sub.remove();
  }, []);

  // Recording elapsed-time counter (shown in HUD while recording)
  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => {
        setRecordingSeconds(s => s + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  // Press "Calibrate" → store current pitch as the zero reference
  const handleCalibrate = useCallback(() => {
    setCalibrationOffset(rawPitch);
  }, [rawPitch]);

  // Helpers
  const formatTime = (s) => {
    const mm = Math.floor(s / 60).toString().padStart(2, '0');
    const ss = (s % 60).toString().padStart(2, '0');
    return `${mm}:${ss}`;
  };

  // Angle dial: indicator X offset, clamped to ±DIAL_HALF_WIDTH
  const clamped        = Math.max(-DIAL_MAX_DEG, Math.min(DIAL_MAX_DEG, correctedPitch));
  const dialIndicatorX = (clamped / DIAL_MAX_DEG) * DIAL_HALF_WIDTH;

  // ── Teammate's original video-playback screen (unchanged) ─────────────────
  if (video) {
    let shareVideo = () => { shareAsync(video.uri).then(); };
    return (
      <SafeAreaView>
        <VideoView player={player} style={styles.video} nativeControls />
        <Button title="Share"   onPress={shareVideo} />
        <Button title="Discard" onPress={() => setVideo(undefined)} />
      </SafeAreaView>
    );
  }

  // ── Main camera view ───────────────────────────────────────────────────────
  return (
    <CameraView
      style={styles.container}
      type={type}
      ref={cameraReference}
      mode="video"
    >

      {/* ── Calibration HUD overlay (Christina's) ── */}
      <View style={styles.hud}>

        {/* Tilt angle + LEVEL badge */}
        <View style={styles.angleRow}>
          <Text style={styles.angleText}>{correctedPitch.toFixed(1)}°</Text>
          {isLevel && (
            <View style={styles.levelBadge}>
              <Text style={styles.levelText}>LEVEL</Text>
            </View>
          )}
        </View>

        {/* Angle dial */}
        <View style={styles.dialBar}>
          {/* Green centre reference line */}
          <View style={styles.dialCentreLine} />
          {/* White indicator dot that slides left/right */}
          <View style={[styles.dialIndicator, { transform: [{ translateX: dialIndicatorX }] }]} />
        </View>

        {/* REC timer — only visible while recording */}
        {isRecording && (
          <View style={styles.recRow}>
            <View style={styles.recDot} />
            <Text style={styles.recText}>REC  {formatTime(recordingSeconds)}</Text>
          </View>
        )}

      </View>
      {/* ── End calibration HUD ── */}

      {/* ── Bottom controls ───────────────────────────────────────────────── */}
      <View style={styles.buttonContainer}>

        {/* Calibrate button (Christina's) */}
        <TouchableOpacity style={styles.calibrateBtn} onPress={handleCalibrate}>
          <Text style={styles.calibrateBtnText}>Calibrate</Text>
        </TouchableOpacity>

        {/* Teammate's original record/stop button (unchanged) */}
        <Button
          title={isRecording ? "Stop recording" : "Record Video"}
          onPress={isRecording ? stopRecording : recordVideo}
        />

      </View>

    </CameraView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({

  // Teammate's original styles (preserved, minor layout tweak to buttonContainer
  // so the Calibrate button sits beside the existing Record button)
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  video: {
    width: '100%',
    height: 400,
  },

  // Calibration HUD
  hud: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  angleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 10,
  },
  angleText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '600',
    textShadowColor: 'rgba(0,0,0,0.85)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  levelBadge: {
    backgroundColor: '#34C759',
    borderRadius: 6,
    paddingHorizontal: 9,
    paddingVertical: 3,
  },
  levelText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  dialBar: {
    width: 220,
    height: 22,
    backgroundColor: 'rgba(255,255,255,0.18)',
    borderRadius: 11,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    marginBottom: 8,
  },
  dialCentreLine: {
    position: 'absolute',
    width: 2,
    height: 22,
    backgroundColor: '#34C759',
  },
  dialIndicator: {
    position: 'absolute',
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.45,
    shadowRadius: 3,
  },
  recRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  recDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: '#FF3B30',
  },
  recText: {
    color: '#FF3B30',
    fontSize: 15,
    fontWeight: '700',
  },

  // Bottom controls — row layout with Calibrate beside the record button
  buttonContainer: {
    position: 'absolute',
    bottom: 44,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 20,
    backgroundColor: 'rgba(0,0,0,0.35)',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 18,
  },
  calibrateBtn: {
    backgroundColor: '#FF9500',
    borderRadius: 10,
    paddingHorizontal: 20,
    paddingVertical: 11,
  },
  calibrateBtnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 15,
  },
});
