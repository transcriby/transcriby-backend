#!/usr/bin/env python3
"""
Transcriby API — offline speech-to-text, as a FastAPI web service.

No desktop window here — this is a headless HTTP API around the same
faster-whisper engine used by the desktop app, meant to be called from
curl, Postman, a script, or another app. Transcription jobs run in the
background; you poll for progress and pick up the result when it's done.

---------------------------------------------------------------------------
SETUP (one time)

    pip install fastapi "uvicorn[standard]" python-multipart faster-whisper numpy

RUN

    python transcriby_api.py
    (equivalently: uvicorn transcriby_api:app --host 0.0.0.0 --port 8000)

Then open http://localhost:8000/docs in a browser — FastAPI's built-in
Swagger UI lets you upload a file and try every endpoint by hand, with no
separate client needed.

The first time you use a given model size, it downloads once from Hugging
Face (needs internet) and is cached under ~/.cache/huggingface. Every run
after that works fully offline.

---------------------------------------------------------------------------
ENDPOINTS

  GET    /health                    liveness check
  GET    /models                    available model sizes and notes
  POST   /transcribe                upload a file, start a job -> {"job_id"}
  GET    /jobs/{job_id}             status + progress (+ result once done)
  POST   /jobs/{job_id}/cancel      cancel a running job
  DELETE /jobs/{job_id}             drop a finished/failed job from memory
  GET    /jobs/{job_id}/txt         plain-text transcript (once done)
  GET    /jobs/{job_id}/srt         SubRip subtitles (once done, needs timestamps)

POST /transcribe form fields:
  file          (required) the audio/video file
  model         one of tiny.en, base.en, small.en, tiny, base (default base.en)
  timestamps    "true"/"false" — segment timestamps + enables word_level (default true)
  word_level    "true"/"false" — word-level timestamps, a bit slower (default false)

Example:
    curl -F "file=@meeting.mp3" -F "model=base.en" http://localhost:8000/transcribe
    curl http://localhost:8000/jobs/<job_id>
---------------------------------------------------------------------------
"""

import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

SAMPLE_RATE = 16000

# How many transcription jobs run at once. Each loaded model stays resident
# in memory, so keep this low on a modest machine — jobs simply queue.
MAX_WORKERS = int(os.environ.get("TRANSCRIBY_WORKERS", "1"))

MODELS = {
    "tiny.en":  ("tiny.en",  "English-only. Fastest, ~75 MB download."),
    "base.en":  ("base.en",  "English-only. Good balance — recommended."),
    "small.en": ("small.en", "English-only. More accurate, slower, ~465 MB."),
    "tiny":     ("tiny",     "Multilingual, auto-detects language, ~75 MB."),
    "base":     ("base",     "Multilingual, auto-detects language, ~145 MB."),
}
DEFAULT_MODEL_KEY = "tiny.en"


def fmt_time(s):
    s = max(0, int(s or 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def srt_time(s):
    s = max(0.0, s or 0.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


class _Cancelled(Exception):
    pass


class Job:
    def __init__(self, job_id, filename, model_key, want_timestamps, word_level):
        self.id = job_id
        self.filename = filename
        self.model_key = model_key
        self.want_timestamps = want_timestamps
        self.word_level = word_level
        self.status = "queued"       # queued | processing | done | error | cancelled
        self.status_text = "Queued"
        self.progress_pct = 0.0
        self.error = None
        self.result = None           # {"text","segments","words","duration","language"}
        self.cancel_event = threading.Event()
        self.created_at = time.time()
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            d = {
                "job_id": self.id,
                "filename": self.filename,
                "model": self.model_key,
                "status": self.status,
                "status_text": self.status_text,
                "progress_pct": round(self.progress_pct, 1),
                "created_at": self.created_at,
            }
            if self.status == "error":
                d["error"] = self.error
            if self.status == "done":
                d["result"] = self.result
            return d


_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_models_cache = {}
_models_lock = threading.Lock()
_jobs = {}
_jobs_lock = threading.Lock()


def get_model(model_key):
    with _models_lock:
        if model_key not in _models_cache:
            from faster_whisper import WhisperModel
            model_name = MODELS[model_key][0]
            _models_cache[model_key] = WhisperModel(
                model_name, device="cpu", compute_type="int8"
            )
        return _models_cache[model_key]


def run_job(job: Job, file_path: str):
    try:
        with job.lock:
            job.status = "processing"
            job.status_text = "Loading model… (first use downloads it, needs internet)"
        model = get_model(job.model_key)
        if job.cancel_event.is_set():
            raise _Cancelled()

        with job.lock:
            job.status_text = "Decoding audio…"
        from faster_whisper.audio import decode_audio
        audio = decode_audio(file_path, sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        if job.cancel_event.is_set():
            raise _Cancelled()

        segments_gen, info = model.transcribe(
            audio,
            word_timestamps=job.word_level,
            without_timestamps=(not job.want_timestamps),
            vad_filter=True,
        )
        total_duration = info.duration or duration

        segments = []
        words = []
        for seg in segments_gen:
            if job.cancel_event.is_set():
                raise _Cancelled()
            segments.append({"start": seg.start or 0.0, "end": seg.end, "text": seg.text})
            if job.word_level and seg.words:
                for w in seg.words:
                    words.append({"start": w.start, "end": w.end, "text": w.word})
            pct = (seg.end / total_duration * 100) if total_duration else 0.0
            with job.lock:
                job.progress_pct = pct
                job.status_text = f"Transcribing… {fmt_time(seg.end)} / {fmt_time(total_duration)}"

        full_text = re.sub(r"[ \t]+", " ", "".join(s["text"] for s in segments)).strip()
        with job.lock:
            job.status = "done"
            job.status_text = "Done."
            job.progress_pct = 100.0
            job.result = {
                "text": full_text,
                "segments": segments,
                "words": words,
                "duration": duration,
                "language": getattr(info, "language", None),
            }
    except _Cancelled:
        with job.lock:
            job.status = "cancelled"
            job.status_text = "Cancelled."
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        msg = str(e)
        if "ModuleNotFoundError" in tb and "faster_whisper" in tb:
            msg = "faster-whisper isn't installed on the server. Run: pip install faster-whisper"
        with job.lock:
            job.status = "error"
            job.status_text = "Failed."
            job.error = msg
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


app = FastAPI(
    title="Transcriby API",
    description="Offline speech-to-text (faster-whisper) as a background-job HTTP API.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_job_or_404(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {key: {"model": name, "note": note} for key, (name, note) in MODELS.items()}


@app.post("/transcribe", status_code=202)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_KEY),
    timestamps: bool = Form(True),
    word_level: bool = Form(False),
):
    if model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'. Valid options: {list(MODELS)}")
    word_level = bool(word_level and timestamps)

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="transcriby_")
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    job_id = uuid.uuid4().hex
    job = Job(job_id, file.filename, model, bool(timestamps), word_level)
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(run_job, job, tmp_path)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _get_job_or_404(job_id).snapshot()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _get_job_or_404(job_id)
    job.cancel_event.set()
    return {"ok": True}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "No such job")
    return {"ok": True}


@app.get("/jobs/{job_id}/txt", response_class=PlainTextResponse)
def get_txt(job_id: str):
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not done yet (status: {job.status})")
    return job.result["text"]


@app.get("/jobs/{job_id}/srt", response_class=PlainTextResponse)
def get_srt(job_id: str):
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not done yet (status: {job.status})")
    segments = job.result["segments"]
    if not segments:
        raise HTTPException(409, "This job ran with timestamps off, so no .srt is available")
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg["start"] or 0.0
        end = seg["end"] if seg["end"] is not None else start + 3.0
        lines.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
