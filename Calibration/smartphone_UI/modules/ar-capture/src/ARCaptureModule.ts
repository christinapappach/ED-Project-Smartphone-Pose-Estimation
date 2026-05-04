import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';
import { NativeModulesProxy, EventEmitter } from 'expo-modules-core';

const NativeModule = requireNativeModule('ARCapture');
const emitter = new EventEmitter(NativeModule);

export const ARCameraView = requireNativeViewManager('ARCapture');

export function isAvailable(): boolean {
  return NativeModule.isAvailable();
}

export function startSession(): void {
  NativeModule.startSession();
}

export function stopSession(): void {
  NativeModule.stopSession();
}

export function startRecording(fps = 30, maxDuration = 60): void {
  NativeModule.startRecording(fps, maxDuration);
}

export function stopRecording(): void {
  NativeModule.stopRecording();
}

export function calibrate(): void {
  NativeModule.calibrate();
}

export function uploadToServer(videoUri: string, serverUrl: string): void {
  NativeModule.uploadToServer(videoUri, serverUrl);
}

export function getTrackingState(): string {
  return NativeModule.getTrackingState();
}

// Event listeners
export function addPitchListener(listener: (event: { pitch: number; isLevel: boolean }) => void) {
  return emitter.addListener('onPitchUpdate', listener);
}

export function addFrameCountListener(listener: (event: { count: number }) => void) {
  return emitter.addListener('onFrameCountUpdate', listener);
}

export function addRecordingFinishedListener(
  listener: (event: {
    videoUri: string;
    // Camera intrinsics
    intrinsics: number[][];
    fx: number; fy: number; cx: number; cy: number;
    // Recording info
    frameCount: number;
    durationSeconds: number;
    targetFPS: number;
    actualFPS: number;
    videoFormatFPS: number;
    fileSizeBytes: number;
    // Resolution
    imageWidth: number;
    imageHeight: number;
    // Depth / LiDAR
    hasLiDAR: boolean;
    hasDepthData: boolean;
    depthWidth: number;
    depthHeight: number;
    // Camera pose
    eulerAngles: number[];
    cameraTransform: number[][];
    projectionMatrix: number[][];
    // Exposure
    exposureDuration: number;
    exposureOffset: number;
    lensPosition: number;
    // Lighting
    lightEstimateIntensity: number;
    lightEstimateTemperature: number;
    // Per-frame transforms
    frameTransforms: Array<{
      frame: number;
      timestamp: number;
      transform: number[][];
      euler: number[];
      intrinsics_fx: number;
      intrinsics_fy: number;
    }>;
  }) => void
) {
  return emitter.addListener('onRecordingFinished', listener);
}

export function addTrackingStateListener(
  listener: (event: { state: string; message: string }) => void
) {
  return emitter.addListener('onTrackingStateChange', listener);
}

export function addUploadProgressListener(
  listener: (event: { progress: number }) => void
) {
  return emitter.addListener('onUploadProgress', listener);
}
