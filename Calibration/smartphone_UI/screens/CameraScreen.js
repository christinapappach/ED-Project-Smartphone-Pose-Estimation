// ─────────────────────────────────────────────────────────────────────────────
// CameraScreen.js
//
// Camera/recording UI with calibration HUD overlay.
// After recording, extracts camera intrinsics as JSON and displays them
// before allowing upload of video + intrinsics to the MeTRAbs server.
// ─────────────────────────────────────────────────────────────────────────────

import { ActivityIndicator, Alert, Button, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as MediaLibrary from 'expo-media-library';
import { File, Paths } from 'expo-file-system/next';
import * as LegacyFileSystem from 'expo-file-system/legacy';
import { shareAsync } from 'expo-sharing';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useVideoPlayer, VideoView } from 'expo-video';
import { Platform } from 'react-native';

// Native ARKit module
import {
  ARCameraView,
  isAvailable,
  startSession,
  stopSession,
  startRecording,
  stopRecording,
  calibrate,
  uploadToServer,
  addPitchListener,
  addFrameCountListener,
  addRecordingFinishedListener,
  addTrackingStateListener,
} from '../modules/ar-capture';
import { useUploadQueue } from '../contexts/UploadQueue';

// ── Constants ────────────────────────────────────────────────────────────────
const LEVEL_THRESHOLD = 1.0;
const DIAL_HALF_WIDTH = 110;
const DIAL_MAX_DEG    = 30;
const DEFAULT_URL     = "https://engdesfau26-smartphonepose-metrabs-server.hf.space";

export default function CameraScreen() {

  // ── Global upload queue (FIFO worker that survives navigation) ─────────────
  const queue = useUploadQueue();

  // ── State ──────────────────────────────────────────────────────────────────
  const [serverUrl,        setServerUrl]         = useState(DEFAULT_URL);
  const [arAvailable,      setArAvailable]      = useState(true);
  const [isRecordingState, setIsRecording]       = useState(false);
  const [video,            setVideo]             = useState(null);   // { uri, intrinsics, fx, fy, cx, cy, frameCount }
  const [trackingState,    setTrackingState]     = useState('normal');
  const [trackingMsg,      setTrackingMsg]       = useState('');

  // Post-recording state
  const [processing,       setProcessing]        = useState(false);
  const [cameraJson,       setCameraJson]         = useState(null);
  const [jsonSaved,        setJsonSaved]          = useState(false);
  const [jsonFilePath,     setJsonFilePath]       = useState(null);
  const [jsonError,        setJsonError]          = useState(null);
  const [uploadingState,   setUploadingState]     = useState(false);
  const [uploadResult,     setUploadResult]       = useState(null);
  const [csvResult,        setCsvResult]          = useState(null);  // { path, rows }
  const [pendingJson,      setPendingJson]        = useState(null);

  // Calibration state
  const [rawPitch,          setRawPitch]          = useState(0);
  const [calibrationOffset, setCalibrationOffset] = useState(0);
  const [recordingSeconds,  setRecordingSeconds]  = useState(0);
  const [nativeFrameCount,  setNativeFrameCount]  = useState(0);
  const timerRef = useRef(null);

  // Derived calibration values
  const correctedPitch = rawPitch - calibrationOffset;
  const isLevel        = Math.abs(correctedPitch) < LEVEL_THRESHOLD;

  // ── ARKit session lifecycle ────────────────────────────────────────────────
  useEffect(() => {
    const available = isAvailable();
    setArAvailable(available);
    if (available) startSession();
    return () => { if (available) stopSession(); };
  }, []);

  // ── Native event listeners ─────────────────────────────────────────────────
  useEffect(() => {
    const pitchSub  = addPitchListener(({ pitch }) => setRawPitch(pitch));
    const frameSub  = addFrameCountListener(({ count }) => setNativeFrameCount(count));
    const trackSub  = addTrackingStateListener(({ state, message }) => {
      setTrackingState(state);
      setTrackingMsg(message);
    });

    const recordSub = addRecordingFinishedListener((result) => {
      const fullData = {
        camera_intrinsics: {
          focal_length: { fx: result.fx || 0, fy: result.fy || 0 },
          principal_point: { cx: result.cx || 0, cy: result.cy || 0 },
          intrinsic_matrix: result.intrinsics || [],
        },
        image_resolution: {
          width: result.imageWidth || 0,
          height: result.imageHeight || 0,
        },
        recording_info: {
          frame_count: result.frameCount || 0,
          duration_seconds: result.durationSeconds || 0,
          target_fps: result.targetFPS || 60,
          actual_fps: result.actualFPS || 0,
          video_format_fps: result.videoFormatFPS || 0,
          file_size_bytes: result.fileSizeBytes || 0,
          timestamp: new Date().toISOString(),
          device: 'iPhone',
          ar_tracking: true,
        },
        depth_data: {
          has_lidar: result.hasLiDAR || false,
          has_depth: result.hasDepthData || false,
          depth_width: result.depthWidth || 0,
          depth_height: result.depthHeight || 0,
          depth_uri: result.depthUri || null,
          depth_frame_count: result.depthFrameCount || 0,
          depth_file_size_bytes: result.depthFileSizeBytes || 0,
        },
        audio: {
          has_audio: result.hasAudio || false,
        },
        camera_pose: {
          euler_angles: result.eulerAngles || [0, 0, 0],
          camera_transform_4x4: result.cameraTransform || [],
          projection_matrix_4x4: result.projectionMatrix || [],
        },
        exposure: {
          exposure_duration: result.exposureDuration || 0,
          exposure_offset: result.exposureOffset || 0,
          lens_position: result.lensPosition || 0,
        },
        lighting: {
          ambient_intensity: result.lightEstimateIntensity || 0,
          ambient_color_temperature: result.lightEstimateTemperature || 0,
        },
        frame_transforms: result.frameTransforms || [],
      };

      setCameraJson(fullData);
      setVideo({
        uri: result.videoUri,
        depthUri: result.depthUri || null,
        folderUri: result.recordingFolder || null,
      });
      setIsRecording(false);
      // Stash the folder URI alongside the JSON so the save-effect knows
      // where to write intrinsics.json (inside the recording folder).
      setPendingJson({ data: fullData, folderUri: result.recordingFolder || null });
    });

    return () => {
      pitchSub.remove();
      frameSub.remove();
      recordSub.remove();
      trackSub.remove();
    };
  }, []);

  // Recording elapsed-time counter
  useEffect(() => {
    if (isRecordingState) {
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => setRecordingSeconds(s => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecordingState]);

  // Save JSON to file when pendingJson is set. If a recording folder was
  // returned by the native module, we save inside it as intrinsics.json so
  // every artifact for one recording stays bundled together. Falls back to a
  // flat camera_intrinsics_<ts>.json if the native module didn't provide a
  // folder URI (older builds).
  useEffect(() => {
    if (!pendingJson) return;
    (async () => {
      try {
        const data = pendingJson.data || pendingJson;
        const folderUri = pendingJson.folderUri || null;
        const jsonString = JSON.stringify(data, null, 2);
        let savedUri;
        if (folderUri) {
          // Write into the recording folder.
          savedUri = `${folderUri}intrinsics.json`;
          await LegacyFileSystem.writeAsStringAsync(savedUri, jsonString);
        } else {
          const fileName = `camera_intrinsics_${Date.now()}.json`;
          const file = new File(Paths.document, fileName);
          file.create();
          file.write(jsonString);
          savedUri = file.uri;
        }
        console.log('JSON saved to:', savedUri);
        setJsonFilePath(savedUri);
        setJsonSaved(true);
        setJsonError(null);
      } catch (err) {
        console.log('JSON save error:', err);
        setJsonError(err.message || 'Failed to save');
        setJsonSaved(false);
      }
    })();
    setPendingJson(null);
  }, [pendingJson]);

  // Video player (for preview)
  const player = useVideoPlayer(video ? video.uri : null, p => {
    p.loop = true;
    p.play();
  });

  // ── Recording controls ─────────────────────────────────────────────────────
  const handleStartRecording = () => {
    setIsRecording(true);
    setNativeFrameCount(0);
    setUploadResult(null);
    startRecording(60, 60);
  };

  const handleStopRecording = () => {
    stopRecording();
  };

  // ── Calibrate ──────────────────────────────────────────────────────────────
  const handleCalibrate = useCallback(() => {
    calibrate();
    setCalibrationOffset(rawPitch);
  }, [rawPitch]);

  // ── Enqueue a recording for upload ─────────────────────────────────────────
  // The actual upload work happens in the global UploadQueue context's worker
  // — survives navigation and runs FIFO. This handler just packs the payload
  // and pushes it on the queue, so the user can immediately go back to the
  // camera screen and record more clips while previous ones are still uploading.
  const handleUpload = () => {
    if (!video || !video.uri) { console.log("No video to upload"); return; }

    const id = queue.enqueue({
      videoUri: video.uri,
      depthUri: video.depthUri || null,
      folderUri: video.folderUri || null,
      cameraJson: cameraJson || null,
      serverUrl,
      label: video.folderUri
        ? video.folderUri.replace(/\/$/, '').split('/').pop()
        : video.uri.split('/').pop(),
    });
    console.log("Enqueued upload job:", id);

    // Surface in the local screen state for the existing review-screen UI.
    setUploadingState(true);
    setUploadResult({ success: true, queued: true, jobId: id });
    setCsvResult(null);
  };

  // ── Discard and go back to camera ──────────────────────────────────────────
  const handleDiscard = () => {
    setVideo(null);
    setCameraJson(null);
    setJsonSaved(false);
    setJsonFilePath(null);
    setProcessing(false);
    setUploadResult(null);
    setCsvResult(null);
  };

  // Helpers
  const formatTime = (s) => {
    const mm = Math.floor(s / 60).toString().padStart(2, '0');
    const ss = (s % 60).toString().padStart(2, '0');
    return `${mm}:${ss}`;
  };

  const clamped        = Math.max(-DIAL_MAX_DEG, Math.min(DIAL_MAX_DEG, correctedPitch));
  const dialIndicatorX = (clamped / DIAL_MAX_DEG) * DIAL_HALF_WIDTH;

  // ── ARKit not available (simulator) ────────────────────────────────────────
  if (!arAvailable) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={{ color: '#fff', fontSize: 18 }}>ARKit requires a physical device</Text>
      </View>
    );
  }

  // ── Post-recording review screen (video + JSON) ────────────────────────────
  if (video && cameraJson) {
    return (
      <SafeAreaView style={styles.reviewContainer}>
        <ScrollView contentContainerStyle={styles.reviewScroll}>

          {/* Video preview */}
          <Text style={styles.sectionTitle}>Recorded Video</Text>
          <VideoView player={player} style={styles.videoPreview} nativeControls />

          {/* JSON data display */}
          <Text style={styles.sectionTitle}>Camera Intrinsics (JSON)</Text>
          <View style={styles.jsonContainer}>
            <ScrollView style={styles.jsonScroll} nestedScrollEnabled>
              <Text style={styles.jsonText}>
                {JSON.stringify(cameraJson, null, 2)}
              </Text>
            </ScrollView>
          </View>

          {/* Status indicators */}
          <View style={styles.statusBlock}>
            <View style={styles.statusRow}>
              <View style={[styles.statusDot, { backgroundColor: jsonSaved ? '#34C759' : jsonError ? '#FF3B30' : '#FF9500' }]} />
              <Text style={styles.statusText}>
                {jsonSaved ? 'JSON saved to device' : jsonError ? 'Save failed' : 'Saving JSON...'}
              </Text>
            </View>
            {jsonSaved && jsonFilePath && (
              <Text style={styles.filePathText} numberOfLines={2}>{jsonFilePath}</Text>
            )}
            {jsonError && (
              <Text style={styles.errorText}>{jsonError}</Text>
            )}
          </View>

          {/* Server URL input */}
          <Text style={styles.sectionTitle}>Server URL</Text>
          <TextInput
            style={styles.urlInput}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="https://xxxx.ngrok-free.app"
            placeholderTextColor="#666"
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
          <Text style={styles.urlHint}>
            Paste your Colab ngrok URL here, or use local server IP
          </Text>

          {/* Upload result */}
          {uploadResult && (
            <View style={[styles.resultBanner, { backgroundColor: uploadResult.success ? '#34C759' : '#FF3B30' }]}>
              <Text style={styles.resultText}>
                {uploadResult.success
                  ? `MeTRAbs done! ${uploadResult.frames} frames in ${uploadResult.time}s`
                  : `Upload failed: ${uploadResult.error}`}
              </Text>
              {uploadResult.success && (
                <Text style={styles.resultSubText}>
                  {uploadResult.detections} joint detections | Intrinsics: {uploadResult.usedIntrinsics ? 'YES' : 'NO'}
                </Text>
              )}
            </View>
          )}

          {/* CSV result */}
          {csvResult && (
            <View style={styles.csvBanner}>
              <Text style={styles.csvTitle}>3D Poses CSV</Text>
              <Text style={styles.csvInfo}>{csvResult.rows} rows saved</Text>
              <Text style={styles.filePathText} numberOfLines={2}>{csvResult.path}</Text>
              <TouchableOpacity
                style={[styles.shareBtn, { marginTop: 8 }]}
                onPress={() => shareAsync(csvResult.path)}
              >
                <Text style={styles.shareBtnText}>Share CSV</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Action buttons */}
          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={[styles.uploadBtn, uploadingState && styles.btnDisabled]}
              onPress={handleUpload}
              disabled={uploadingState}
            >
              {uploadingState ? (
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                  <ActivityIndicator color="#fff" size="small" />
                  <Text style={styles.uploadBtnText}>Processing on server...</Text>
                </View>
              ) : (
                <Text style={styles.uploadBtnText}>Upload & Run MeTRAbs</Text>
              )}
            </TouchableOpacity>

            <TouchableOpacity style={styles.shareBtn} onPress={() => shareAsync(video.uri)}>
              <Text style={styles.shareBtnText}>Share Video</Text>
            </TouchableOpacity>

            {jsonFilePath && (
              <TouchableOpacity style={styles.shareBtn} onPress={() => shareAsync(jsonFilePath)}>
                <Text style={styles.shareBtnText}>Share JSON</Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity style={styles.discardBtn} onPress={handleDiscard}>
              <Text style={styles.discardBtnText}>Discard & Re-record</Text>
            </TouchableOpacity>
          </View>

        </ScrollView>
      </SafeAreaView>
    );
  }

  // ── Main AR camera view ────────────────────────────────────────────────────
  return (
    <View style={styles.container}>
      <ARCameraView style={StyleSheet.absoluteFill} isActive={true} />

      {/* Tracking quality warning */}
      {trackingState !== 'normal' && trackingMsg !== '' && (
        <View style={styles.trackingWarning}>
          <Text style={styles.trackingText}>{trackingMsg}</Text>
        </View>
      )}

      {/* Calibration HUD overlay */}
      <View style={styles.hud}>
        <View style={styles.angleRow}>
          <Text style={styles.angleText}>{correctedPitch.toFixed(1)}°</Text>
          {isLevel && (
            <View style={styles.levelBadge}>
              <Text style={styles.levelText}>LEVEL</Text>
            </View>
          )}
        </View>

        <View style={styles.dialBar}>
          <View style={styles.dialCentreLine} />
          <View style={[styles.dialIndicator, { transform: [{ translateX: dialIndicatorX }] }]} />
        </View>

        {isRecordingState && (
          <View style={styles.recRow}>
            <View style={styles.recDot} />
            <Text style={styles.recText}>
              REC  {formatTime(recordingSeconds)}  |  {nativeFrameCount} frames
            </Text>
          </View>
        )}
      </View>

      {/* Bottom controls */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.calibrateBtn} onPress={handleCalibrate}>
          <Text style={styles.calibrateBtnText}>Calibrate</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.recordBtn, isRecordingState && styles.recordBtnActive]}
          onPress={isRecordingState ? handleStopRecording : handleStartRecording}
        >
          <Text style={[styles.recordBtnText, isRecordingState && styles.recordBtnTextActive]}>
            {isRecordingState ? "Stop" : "Record"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({

  container: { flex: 1, backgroundColor: '#000' },

  // Tracking warning
  trackingWarning: {
    position: 'absolute', top: 110, alignSelf: 'center',
    backgroundColor: 'rgba(255, 149, 0, 0.85)', borderRadius: 12,
    paddingHorizontal: 14, paddingVertical: 6,
  },
  trackingText: { color: '#fff', fontSize: 13, fontWeight: '600' },

  // Calibration HUD
  hud: {
    position: 'absolute', top: 60, left: 0, right: 0,
    alignItems: 'center', paddingHorizontal: 20,
  },
  angleRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 10 },
  angleText: {
    color: '#fff', fontSize: 24, fontWeight: '600',
    textShadowColor: 'rgba(0,0,0,0.85)', textShadowOffset: { width: 0, height: 1 }, textShadowRadius: 4,
  },
  levelBadge: { backgroundColor: '#34C759', borderRadius: 6, paddingHorizontal: 9, paddingVertical: 3 },
  levelText: { color: '#fff', fontSize: 13, fontWeight: '700', letterSpacing: 1.2 },
  dialBar: {
    width: 220, height: 22, backgroundColor: 'rgba(255,255,255,0.18)',
    borderRadius: 11, justifyContent: 'center', alignItems: 'center', overflow: 'hidden', marginBottom: 8,
  },
  dialCentreLine: { position: 'absolute', width: 2, height: 22, backgroundColor: '#34C759' },
  dialIndicator: {
    position: 'absolute', width: 16, height: 16, borderRadius: 8, backgroundColor: '#fff',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.45, shadowRadius: 3,
  },
  recRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  recDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#FF3B30' },
  recText: { color: '#FF3B30', fontSize: 15, fontWeight: '700' },

  // Bottom controls
  buttonContainer: {
    position: 'absolute', bottom: 44, left: 20, right: 20,
    flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 20,
    backgroundColor: 'rgba(0,0,0,0.35)', paddingVertical: 14, paddingHorizontal: 20, borderRadius: 18,
  },
  calibrateBtn: { backgroundColor: '#FF9500', borderRadius: 10, paddingHorizontal: 20, paddingVertical: 11 },
  calibrateBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  recordBtn: { backgroundColor: '#FF3B30', borderRadius: 10, paddingHorizontal: 28, paddingVertical: 11 },
  recordBtnActive: { backgroundColor: '#fff', borderWidth: 2, borderColor: '#FF3B30' },
  recordBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  recordBtnTextActive: { color: '#FF3B30' },

  // Processing screen
  processingContainer: {
    flex: 1, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center', gap: 16,
  },
  processingTitle: { color: '#fff', fontSize: 20, fontWeight: '700' },
  processingSubtitle: { color: '#999', fontSize: 14 },

  // Post-recording review screen
  reviewContainer: { flex: 1, backgroundColor: '#000' },
  reviewScroll: { padding: 20, paddingBottom: 40 },
  sectionTitle: { color: '#fff', fontSize: 18, fontWeight: '700', marginTop: 16, marginBottom: 8 },

  videoPreview: { width: '100%', height: 280, borderRadius: 12, overflow: 'hidden' },

  jsonContainer: {
    backgroundColor: '#1C1C1E', borderRadius: 12, padding: 14,
    borderWidth: 1, borderColor: '#333',
  },
  jsonScroll: { maxHeight: 220 },
  jsonText: { color: '#30D158', fontSize: 12, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },

  statusBlock: { marginTop: 12, gap: 4 },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  statusDot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { color: '#ccc', fontSize: 14, fontWeight: '600' },
  filePathText: { color: '#666', fontSize: 11, marginLeft: 18, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  errorText: { color: '#FF3B30', fontSize: 12, marginLeft: 18 },

  urlInput: {
    backgroundColor: '#1C1C1E', borderRadius: 10, padding: 14, color: '#fff',
    fontSize: 14, borderWidth: 1, borderColor: '#444',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  urlHint: { color: '#666', fontSize: 11, marginTop: 4 },

  resultBanner: { borderRadius: 10, padding: 12, marginTop: 12 },
  resultText: { color: '#fff', fontSize: 14, fontWeight: '600', textAlign: 'center' },
  resultSubText: { color: 'rgba(255,255,255,0.8)', fontSize: 12, textAlign: 'center', marginTop: 4 },

  csvBanner: {
    backgroundColor: '#1C1C1E', borderRadius: 12, padding: 14, marginTop: 12,
    borderWidth: 1, borderColor: '#30D158',
  },
  csvTitle: { color: '#30D158', fontSize: 16, fontWeight: '700' },
  csvInfo: { color: '#aaa', fontSize: 13, marginTop: 2 },

  actionButtons: { marginTop: 20, gap: 12 },
  uploadBtn: {
    backgroundColor: '#007AFF', borderRadius: 12, paddingVertical: 14, alignItems: 'center',
  },
  btnDisabled: { opacity: 0.5 },
  uploadBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  shareBtn: {
    backgroundColor: '#1C1C1E', borderRadius: 12, paddingVertical: 14, alignItems: 'center',
    borderWidth: 1, borderColor: '#444',
  },
  shareBtnText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  discardBtn: {
    backgroundColor: 'transparent', borderRadius: 12, paddingVertical: 14, alignItems: 'center',
    borderWidth: 1, borderColor: '#FF3B30',
  },
  discardBtnText: { color: '#FF3B30', fontSize: 15, fontWeight: '600' },
});
