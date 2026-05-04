// UploadQueue.js — global FIFO queue for video uploads.
// Survives screen navigation (lives at the App provider level), so the user
// can navigate back to the Camera screen mid-upload and record more clips.
// Each clip becomes a job; the worker processes them one at a time.

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { File, Paths } from 'expo-file-system';
// LegacyFileSystem.writeAsStringAsync supports writing to any file:// URI,
// which is how we drop the CSV into the per-recording folder when one exists.

const UploadQueueContext = createContext(null);

let _idCounter = 0;
const newId = () => `job-${Date.now()}-${++_idCounter}`;

// Status states a job moves through: pending → uploading → (done | failed)
const STATUS = {
  PENDING: 'pending',
  UPLOADING: 'uploading',
  DONE: 'done',
  FAILED: 'failed',
};

export function UploadQueueProvider({ children }) {
  // The full job list. Newest-first ordering for the UI; worker pulls oldest pending.
  const [jobs, setJobs] = useState([]);
  // Stable ref so the worker effect can read the latest jobs without re-running.
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  // Prevent two workers from running concurrently.
  const workerBusyRef = useRef(false);

  /**
   * Push a new upload onto the queue.
   *
   * @param {object} payload
   *   videoUri:    string         (required) file:// path to the recorded MP4
   *   depthUri?:   string|null    optional path to LiDAR .depth.bin
   *   cameraJson?: object|null    ARKit metadata (fx/fy/cx/cy etc.)
   *   serverUrl:   string         (required) HF Space base URL
   *   label?:      string         optional human-readable name (defaults to filename)
   * @returns {string} the job's id
   */
  const enqueue = useCallback((payload) => {
    const id = newId();
    const job = {
      id,
      ...payload,
      label: payload.label || (payload.videoUri ? payload.videoUri.split('/').pop() : 'recording'),
      status: STATUS.PENDING,
      createdAt: Date.now(),
    };
    setJobs((prev) => [job, ...prev]);
    return id;
  }, []);

  const removeJob = useCallback((id) => {
    setJobs((prev) => prev.filter((j) => j.id !== id));
  }, []);

  const retryJob = useCallback((id) => {
    setJobs((prev) =>
      prev.map((j) => (j.id === id ? { ...j, status: STATUS.PENDING, error: null } : j))
    );
  }, []);

  // Worker effect: whenever jobs change, if we're not already running and there's
  // a pending job, run it. Single-threaded.
  //
  // No cleanup function is used: this provider is mounted at the App level and
  // never unmounts, and a defensive `cancelled` flag in cleanup would falsely
  // fire on every re-render (jobs array changes when status flips), which
  // caused the worker to bail before marking jobs done — leaving the queue
  // stuck after the first upload.
  useEffect(() => {
    if (workerBusyRef.current) return;
    const next = jobs.find((j) => j.status === STATUS.PENDING);
    if (!next) return;

    workerBusyRef.current = true;

    (async () => {
      // Mark uploading.
      setJobs((prev) => prev.map((j) => (j.id === next.id ? { ...j, status: STATUS.UPLOADING, startedAt: Date.now() } : j)));

      try {
        const result = await runUploadWithFallback(next);
        setJobs((prev) => prev.map((j) => (j.id === next.id ? { ...j, status: STATUS.DONE, finishedAt: Date.now(), ...result } : j)));
      } catch (err) {
        console.log('Upload job failed:', err);
        setJobs((prev) => prev.map((j) => (j.id === next.id ? { ...j, status: STATUS.FAILED, finishedAt: Date.now(), error: err.message || String(err) } : j)));
      } finally {
        workerBusyRef.current = false;
      }
    })();
  }, [jobs]);

  // Convenience selectors for UI.
  const pending = jobs.filter((j) => j.status === STATUS.PENDING);
  const uploading = jobs.find((j) => j.status === STATUS.UPLOADING);
  const inFlight = pending.length + (uploading ? 1 : 0);

  const value = {
    jobs,
    pending,
    uploading,
    inFlight,
    enqueue,
    removeJob,
    retryJob,
    STATUS,
  };

  return <UploadQueueContext.Provider value={value}>{children}</UploadQueueContext.Provider>;
}

export function useUploadQueue() {
  const ctx = useContext(UploadQueueContext);
  if (!ctx) throw new Error('useUploadQueue() must be used inside <UploadQueueProvider>');
  return ctx;
}

// ──────────────────────────────────────────────────────────────────────────────
// Upload implementation: same fetch+poll race as CameraScreen used to do, but
// pulled out so it runs independent of any screen's mount state.

async function runUploadWithFallback(job) {
  const baseUrl = job.serverUrl.replace(/\/$/, '');
  const uploadUrl = `${baseUrl}/upload`;
  const logsUrl   = `${baseUrl}/logs`;
  const csvUrl    = `${baseUrl}/outputs/output_file_1.csv`;
  const startedAtUTC = new Date().toISOString().substring(11, 19); // "HH:MM:SS"

  const formData = new FormData();
  formData.append('file', {
    uri: Platform.OS === 'android' ? job.videoUri : job.videoUri.replace('file://', ''),
    name: 'video.mp4',
    type: 'video/mp4',
  });
  if (job.cameraJson) {
    formData.append('intrinsics_json', JSON.stringify(job.cameraJson));
    const fx = job.cameraJson.camera_intrinsics?.focal_length?.fx;
    const fy = job.cameraJson.camera_intrinsics?.focal_length?.fy;
    const cx = job.cameraJson.camera_intrinsics?.principal_point?.cx;
    const cy = job.cameraJson.camera_intrinsics?.principal_point?.cy;
    if (fx != null) formData.append('focal_length_x', String(fx));
    if (fy != null) formData.append('focal_length_y', String(fy));
    if (cx != null) formData.append('principal_point_x', String(cx));
    if (cy != null) formData.append('principal_point_y', String(cy));
  }
  if (job.depthUri) {
    formData.append('depth_data', {
      uri: Platform.OS === 'android' ? job.depthUri : job.depthUri.replace('file://', ''),
      name: 'depth.bin',
      type: 'application/octet-stream',
    });
  }

  const fetchPath = (async () => {
    try {
      const r = await fetch(uploadUrl, { method: 'POST', body: formData, headers: { Accept: 'application/json' } });
      if (!r.ok) {
        const text = await r.text();
        return { ok: false, error: `HTTP ${r.status}: ${text}` };
      }
      return { ok: true, source: 'fetch', data: await r.json() };
    } catch (err) {
      console.log('queue worker: fetch failed, polling will take over:', err.message);
      return new Promise(() => {}); // never resolves; let polling decide
    }
  })();

  const pollPath = (async () => {
    const maxMinutes = 45;
    const intervalMs = 5000;
    const maxAttempts = Math.ceil((maxMinutes * 60 * 1000) / intervalMs);
    const doneRegex = /^\[(\d\d:\d\d:\d\d)\] Done in ([\d.]+)s, (\d+) frames with detections/;
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((res) => setTimeout(res, intervalMs));
      try {
        const r = await fetch(logsUrl, { cache: 'no-store' });
        if (!r.ok) continue;
        const j = await r.json();
        const lines = j.lines || [];
        for (let idx = lines.length - 1; idx >= 0; idx--) {
          const m = lines[idx].match(doneRegex);
          if (m && m[1] >= startedAtUTC) {
            return {
              ok: true,
              source: 'poll',
              data: {
                csv_url: '/outputs/output_file_1.csv',
                output_url: '/outputs/output_video_1.mp4',
                processing_time_seconds: parseFloat(m[2]),
                frames_with_detections: parseInt(m[3], 10),
                used_intrinsics: !!job.cameraJson,
                used_lidar_depth: !!job.depthUri,
              },
            };
          }
        }
      } catch { /* keep polling */ }
    }
    return { ok: false, error: `Timed out after ${maxMinutes} min waiting for server` };
  })();

  const result = await Promise.race([fetchPath, pollPath]);
  if (!result.ok) throw new Error(result.error);

  const data = result.data;
  // Best-effort fetch of the CSV output to a local file. If this job has a
  // recording folder, we save inside it as output.csv so the recording stays
  // self-contained. Otherwise fall back to flat poses_3d_<ts>.csv in Documents.
  let csvLocalPath = null;
  let csvRows = 0;
  try {
    const csvFullUrl = data.csv_url ? `${baseUrl}${data.csv_url}` : csvUrl;
    const csvResp = await fetch(csvFullUrl);
    if (csvResp.ok) {
      const csvText = await csvResp.text();
      if (csvText) {
        if (job.folderUri) {
          csvLocalPath = `${job.folderUri}output.csv`;
          await FileSystem.writeAsStringAsync(csvLocalPath, csvText);
        } else {
          const csvFileName = `poses_3d_${Date.now()}.csv`;
          const csvFile = new File(Paths.document, csvFileName);
          csvFile.create();
          csvFile.write(csvText);
          csvLocalPath = csvFile.uri;
        }
        csvRows = csvText.trim().split('\n').length - 1;
      }
    }
  } catch (e) {
    console.log('queue worker: CSV save failed:', e.message);
  }

  return {
    source: result.source,
    response: data,
    csvLocalPath,
    csvRows,
    framesWithDetections: data.frames_with_detections || data.frames_processed,
    processingSeconds: data.processing_time_seconds,
    usedIntrinsics: data.used_intrinsics,
    usedLidar: data.used_lidar_depth,
  };
}
