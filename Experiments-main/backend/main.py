# main.py
import os
import json
import uuid
import subprocess
import traceback
import time
from typing import Optional, List
from pathlib import Path

import pandas as pd
import requests
import openai
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Load environment
# -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
CSV_PATH = os.getenv("CSV_PATH", r"C:\Users\swapn\OneDrive\Desktop\MyProjects\ChemistryExp.csv")  # adjust or set in .env
STATIC_DIR = os.getenv("STATIC_DIR", "static_videos")


if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment or .env")

if not ELEVENLABS_API_KEY:
    print("Warning: ELEVENLABS_API_KEY not set. TTS will fallback to silent audio placeholder.")

# configure OpenAI
openai.api_key = OPENAI_API_KEY

# create storage dir
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------
# Utility: load CSV robustly
# -----------------------
def load_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "cp1252", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load CSV ({path}). Last error: {last_exc}")

if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV file not found at {CSV_PATH}. Put your CSV there or set CSV_PATH env var.")

df = load_csv(CSV_PATH)
# convert to list-of-dicts cache
experiments = df.to_dict(orient="records")

# -----------------------
# Helpers to find columns and rows
# -----------------------
POSSIBLE_TITLE_COLS = ["Experiment Title", "ExperimentName", "Experiment", "title", "Experiment Name"]
POSSIBLE_PROCEDURE_COLS = ["Procedure / Steps", "Procedure", "Procedure/Steps", "Procedure / Step", "Procedure Steps"]
POSSIBLE_DESC_COLS = ["Outcome / Observation", "Description", "Outcome", "Observation"]
POSSIBLE_SAFETY_COLS = ["Safety Notes / Cautions", "Safety Notes", "Safety", "Cautions"]

def get_first_col(cols: List[str], default=None):
    for c in cols:
        if c in df.columns:
            return c
    return default

COL_TITLE = get_first_col(POSSIBLE_TITLE_COLS, default=None)
COL_PROCEDURE = get_first_col(POSSIBLE_PROCEDURE_COLS, default=None)
COL_DESC = get_first_col(POSSIBLE_DESC_COLS, default=None)
COL_SAFETY = get_first_col(POSSIBLE_SAFETY_COLS, default=None)

def get_row_title(row: dict):
    if COL_TITLE and row.get(COL_TITLE):
        return str(row.get(COL_TITLE))
    # fallback to any column with 'Experiment' in name
    for k in row.keys():
        if "experiment" in k.lower():
            return str(row.get(k))
    return "Untitled Experiment"

def find_experiment_by_id(exp_id: int):
    # fallback to index (1-based)
    idx = int(exp_id) - 1
    if 0 <= idx < len(experiments):
        return experiments[idx]
    return None

def find_experiment_by_name(name: str):
    name_l = name.strip().lower()
    for row in experiments:
        title = get_row_title(row)
        if title.strip().lower() == name_l:
            return row
    # partial match
    for row in experiments:
        title = get_row_title(row)
        if name_l in title.strip().lower():
            return row
    return None

# -----------------------
# OpenAI: create structured video plan
# -----------------------
def generate_video_plan(title: str, description: str, procedure: str, safety: str = "", target_grade: str = "grade 9"):
    prompt = f'''
You are a science teacher and short-form educational video director for school students ({target_grade}).
Create a concise video plan for the experiment below. Return only valid JSON.

Title: {title}
Short description / expected outcome: {description}
Procedure / steps: {procedure}
Safety notes: {safety}

Return JSON with keys:
- short_description: string
- suggested_duration_seconds: integer
- segments: list of 4-8 objects with:
    - heading: string
    - timestamp_seconds: integer (start time relative)
    - narration: string (what to speak)
    - visual: string (what to show)
'''
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful teacher and concise video script writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        text = resp.choices[0].message["content"].strip()
        # try parse json from assistant output
        plan = json.loads(text)
        return plan
    except Exception:
        # fallback: simple single-segment plan
        return {
            "short_description": description or title,
            "suggested_duration_seconds": 90,
            "segments": [
                {"heading": title, "timestamp_seconds": 0, "narration": f"{title}. {description}. Procedure: {procedure}", "visual": "Show the experiment setup and close-up steps."}
            ]
        }

# -----------------------
# ElevenLabs TTS via HTTP
# -----------------------
def synthesize_elevenlabs(text: str, out_path: str, voice_id: Optional[str] = None, timeout: int = 60):
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set in environment; set it to enable ElevenLabs TTS.")

    vid = voice_id if voice_id else ELEVENLABS_VOICE_ID
    if not vid:
        raise RuntimeError("ELEVENLABS_VOICE_ID not set. Provide voice id in env.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"ElevenLabs TTS request failed {resp.status_code}: {resp.text[:400]}")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path

# -----------------------
# Image Generation with Pillow
# -----------------------
def create_slide(text: str, size: tuple = (1280, 720), bg_color: str = "black", text_color: str = "white"):
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), text, font=font, fill=text_color)
    return img

# -----------------------
# Assemble with ffmpeg
# -----------------------
def combine_images_and_audio(image_paths: list, audio_path: str, out_path: str, frame_rate: int = 1):
    """
    Combines a list of images and an audio file into a video using ffmpeg.
    """
    # Create a temporary file listing the images
    with open("image_list.txt", "w") as f:
        for img_path in image_paths:
            f.write(f"file '{img_path}'\n")
            f.write(f"duration 1\n") # Each image will be shown for 1 second

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "image_list.txt",
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-vf", f"fps={frame_rate},format=yuv420p",
        "-shortest",
        out_path
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed to combine images and audio: {e}")
    finally:
        os.remove("image_list.txt")


# -----------------------
# FastAPI app & endpoints
# -----------------------
app = FastAPI(title="AI Experiment Video Generator (CSV + OpenAI + Pillow + ElevenLabs)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development, allow all; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    experiment_id: Optional[int] = None
    experiment_name: Optional[str] = None
    target_grade: Optional[str] = "grade 9"

@app.get("/")
def root():
    return {"message": "AI Experiment Video Generator running", "csv_path": CSV_PATH, "count": len(experiments)}

@app.get("/experiments")
def list_experiments():
    # Return just the experiment titles and their index/ID for easy reference
    exp_list = []
    for i, row in enumerate(experiments, start=1):
        title = get_row_title(row)
        exp_list.append({"id": i, "title": title})
    return {"experiments": exp_list}

@app.get("/experiment/{exp_id}")
def get_experiment(exp_id: int):
    row = find_experiment_by_id(exp_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return row

@app.post("/generate_video")
def generate_video(payload: GenerateRequest):
    # find experiment
    row = None
    if payload.experiment_id:
        row = find_experiment_by_id(int(payload.experiment_id))
    elif payload.experiment_name:
        row = find_experiment_by_name(payload.experiment_name)
    else:
        raise HTTPException(status_code=400, detail="Provide experiment_id or experiment_name")

    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")

    title = get_row_title(row)
    description = str(row.get(COL_DESC) or "")
    procedure = str(row.get(COL_PROCEDURE) or "")
    safety = str(row.get(COL_SAFETY) or "")

    # 1) Build plan via OpenAI
    try:
        plan = generate_video_plan(title, description, procedure, safety, target_grade=payload.target_grade)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    segments = plan.get("segments") or []
    if not segments:
        raise HTTPException(status_code=500, detail="OpenAI returned no segments")

    # 2) Create job id and filenames
    job_id = str(uuid.uuid4())
    base = os.path.join(STATIC_DIR, job_id)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    audio_path = f"{base}.mp3"
    final_video_path = f"{base}.mp4"

    # 3) Generate images with Pillow
    image_paths = []
    for i, seg in enumerate(segments):
        img = create_slide(seg.get('visual', ''))
        img_path = os.path.join(STATIC_DIR, f"{job_id}_slide_{i:02d}.png")
        img.save(img_path)
        image_paths.append(img_path)

    # 4) TTS via ElevenLabs (or fallback silent audio if not configured)
    narration_full = "\n\n".join(seg.get("narration","") for seg in segments if seg.get("narration"))
    try:
        if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
            synthesize_elevenlabs(narration_full or title, audio_path)
        else:
            # create 1-second silent audio so ffmpeg won't fail (not ideal)
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-t", "1", audio_path]
            subprocess.check_call(cmd)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    # 5) Combine images and audio
    try:
        combine_images_and_audio(image_paths, audio_path, final_video_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to combine images and audio: {e}")


    # 6) Success - return job info
    return {"job_id": job_id, "title": title, "video_path": f"/video/{job_id}", "script": plan}

@app.get("/video/{job_id}")
def get_video(job_id: str):
    path = os.path.join(STATIC_DIR, f"{job_id}.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
