from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import shutil
import time
from pathlib import Path
import cv2
from run_metrabs_video import _process_frames
#uvicorn server:app --reload --host 0.0.0.0 --port 8000
app = FastAPI(title="Pose Estimation Backend API")

# App dependencies

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs = {}


# Utility Functions

def extract_frames(video_path, max_frames=50):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()
    return frames



def metrabs_pose_inference(frames):
    print(f"[MeTRAbs] Running on {len(frames)} frames...")
    poses = _process_frames(frames)

    print("[MeTRAbs] Finished inference")
    return poses




def run_inference(job_id, video_path):
    try:
        jobs[job_id]["status"] = "processing"

        # Simulate load time
        time.sleep(1)

        frames = extract_frames(video_path)
        poses = metrabs_pose_inference(frames)


        jobs[job_id]["result"] = {
            "job_id": job_id,
            "num_frames": len(frames),
            "poses": poses
        }
        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)



# Status

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]






# API Endpoints

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    filename = file.filename.lower()

    if not filename.endswith((".mp4", ".mov", ".avi")):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file type"}
    )


    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued"}

    background_tasks.add_task(run_inference, job_id, video_path)

    return {
        "job_id": job_id,
        "status": "queued"
    }
