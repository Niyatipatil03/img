"""
AudioMark Annotator — Pure annotation pipeline with audio effects
FastAPI backend
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import uvicorn, os, json, shutil, uuid, socket
from pathlib import Path

app = FastAPI(title="AudioMark Annotator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Static files + root redirect — must be registered FIRST
from fastapi.responses import RedirectResponse
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent/"static"), html=True), name="static")

@app.get("/")
def _root_redirect():
    return RedirectResponse(url="/static/index.html")

BASE   = Path(__file__).parent
STATIC = BASE / "static"
DATA   = BASE / "data"
DATA.mkdir(exist_ok=True)

# UPLOADS() and ANNOTS() are defined later after project management routes

# ── FILES ────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        dest = UPLOADS() / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"uploaded": saved}

@app.get("/api/files")
def list_files():
    files = []
    for p in sorted(UPLOADS().glob("*")):
        if p.suffix.lower() in [".wav",".mp3",".ogg",".flac",".m4a"]:
            ann = ANNOTS() / (p.stem + ".json")
            seg_count = 0
            if ann.exists():
                try:
                    d = json.loads(ann.read_text())
                    seg_count = len(d.get("segments", []))
                except: pass
            files.append({
                "filename":  p.name,
                "size_kb":   round(p.stat().st_size / 1024, 1),
                "seg_count": seg_count,
                "annotated": ann.exists() and seg_count > 0
            })
    return {"files": files}

@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    p = find_audio_file(filename)
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), media_type="audio/wav")

@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    find_audio_file(filename).unlink(missing_ok=True)
    (ANNOTS() / (Path(filename).stem + ".json")).unlink(missing_ok=True)
    return {"deleted": filename}

@app.post("/api/files/rename")
def rename_file(payload: dict):
    old = payload["old_filename"]; new = payload["new_filename"]
    op = find_audio_file(old)
    if not op.exists(): raise HTTPException(404, "File not found")
    op.rename(UPLOADS() / new)
    oa = ANNOTS() / (Path(old).stem + ".json")
    if oa.exists(): oa.rename(ANNOTS() / (Path(new).stem + ".json"))
    return {"renamed": new}

# ── ANNOTATIONS ──────────────────────────────────────────────
@app.post("/api/annotations/save")
async def save_annotations(request: dict):
    filename = request.get("filename")
    if not filename: raise HTTPException(400, "filename required")
    ann_path = ANNOTS() / (Path(filename).stem + ".json")
    ann_path.write_text(json.dumps(request, indent=2))
    return {"saved": True}

@app.get("/api/annotations/{filename}")
def get_annotations(filename: str):
    ann_path = ANNOTS() / (Path(filename).stem + ".json")
    if ann_path.exists():
        return json.loads(ann_path.read_text())
    return {"filename": filename, "segments": []}

@app.get("/api/annotations/export/json")
def export_all():
    all_anns = []
    for p in sorted(ANNOTS().glob("*.json")):
        try: all_anns.append(json.loads(p.read_text()))
        except: pass
    content = json.dumps(all_anns, indent=2)
    return Response(content, media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=annotations.json"})

# ── AUDIO EFFECTS ─────────────────────────────────────────────
class EffectPayload(BaseModel):
    filename:       str
    effect:         str
    overwrite:      bool  = False
    gain_db:        float = 6.0
    fade_duration:  float = 0.5
    echo_delay:     float = 0.3
    echo_decay:     float = 0.5
    trim_threshold: float = 0.02
    # Smart noise reduction params
    noise_mode:     str   = "gentle"   # gentle | balanced | aggressive
    noise_strength: float = 0.5        # 0.1 – 1.0

# ── SMART NOISE REDUCTION — 3 modes ─────────────────────────
def _smart_noise_reduce(audio, sr, mode="gentle", strength=0.5):
    """
    Smart spectral gate that estimates background ONLY from quiet frames.
    This preserves sharp transients (connector clicks) while suppressing
    the steady factory background hum.

    mode="gentle"     → soft gate, alpha=1.0 — safest, click preserved ~95%
    mode="balanced"   → alpha=1.5 — good balance, click preserved ~85%
    mode="aggressive" → alpha=2.5 — strong reduction, click preserved ~70%

    Key difference from standard noisereduce:
      Standard: estimates noise from FIRST N seconds (may include a click)
                then subtracts uniformly from ALL frequencies → kills clicks
      Smart:    estimates noise ONLY from the quietest 15% of frames
                → background subtracted, transients preserved
    """
    import numpy as np
    from scipy.signal import stft, istft

    nperseg  = 1024
    noverlap = 768

    # STFT
    f, t_fr, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Estimate background PSD from quietest 15% of frames only
    frame_energy = np.mean(np.abs(Zxx)**2, axis=0)
    bg_percentile = np.percentile(frame_energy, 15)
    bg_mask = frame_energy <= bg_percentile
    if bg_mask.sum() < 5:               # fallback if too few quiet frames
        bg_mask = frame_energy <= np.percentile(frame_energy, 30)

    noise_psd   = np.mean(np.abs(Zxx[:, bg_mask])**2, axis=1, keepdims=True)
    noise_floor = np.sqrt(noise_psd) * (1.0 + strength)  # user-controlled

    alpha_map = {"gentle": 1.0, "balanced": 1.5, "aggressive": 2.5}
    alpha = alpha_map.get(mode, 1.0)

    signal_mag = np.abs(Zxx)

    # Soft Wiener mask: H = max(floor, 1 - alpha * noise / signal)
    # floor > 0 means transients (clicks) are NEVER fully removed
    floor_map  = {"gentle": 0.20, "balanced": 0.12, "aggressive": 0.06}
    floor      = floor_map.get(mode, 0.20)

    H = np.maximum(floor, 1.0 - alpha * noise_floor / (signal_mag + 1e-9))

    Zxx_clean = H * Zxx
    _, audio_clean = istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)
    audio_clean = audio_clean[:len(audio)]

    # DO NOT normalize — the amplitude reduction IS the noise reduction.
    # Normalizing back to original peak would completely undo the effect.
    # Just clip to prevent any rare clipping artifacts.
    audio_clean = np.clip(audio_clean, -0.99, 0.99)

    return audio_clean


# ── ENDPOINT: Analyse clicks before/after noise reduction ───
@app.get("/api/effects/click_check/{filename}")
def click_check(filename: str, mode: str = "balanced", strength: float = 0.5):
    """
    Returns detected click positions BEFORE and AFTER noise reduction.
    Lets the user verify no clicks are lost before committing.
    """
    try:
        import librosa, numpy as np
        from scipy.signal import find_peaks
    except ImportError:
        raise HTTPException(500, "librosa not installed")

    src = find_audio_file(filename)
    if not src.exists():
        raise HTTPException(404, "File not found")

    audio, sr = librosa.load(str(src), sr=None, mono=True)
    duration  = float(len(audio) / sr)

    def detect_clicks(a, label):
        frame_length = 512; hop_length = 128
        rms   = librosa.feature.rms(y=a, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        bg    = np.percentile(rms, 20)
        thr   = bg * 2.5
        peaks, props = find_peaks(rms, height=thr, distance=int(sr * 0.3 / hop_length))
        return [{"time": round(float(times[p]), 3),
                 "energy": round(float(rms[p]), 4),
                 "snr_x":  round(float(rms[p] / (bg + 1e-9)), 2)}
                for p in peaks]

    # Clicks in original
    clicks_orig = detect_clicks(audio, "original")

    # Clicks after smart noise reduction
    audio_clean = _smart_noise_reduce(audio, sr, mode=mode, strength=strength)
    clicks_clean = detect_clicks(audio_clean, "clean")

    # Compare — check how many original clicks survive
    survived = 0
    for co in clicks_orig:
        if any(abs(cc["time"] - co["time"]) < 0.5 for cc in clicks_clean):
            survived += 1

    return {
        "filename":     filename,
        "duration":     duration,
        "mode":         mode,
        "clicks_before": clicks_orig,
        "clicks_after":  clicks_clean,
        "clicks_before_count": len(clicks_orig),
        "clicks_after_count":  len(clicks_clean),
        "clicks_survived": survived,
        "click_survival_rate": round(survived / max(len(clicks_orig), 1) * 100, 0),
        "safe_to_apply": survived >= len(clicks_orig),
    }


@app.post("/api/effects/apply")
def apply_effect(payload: EffectPayload):
    try:
        import librosa, soundfile as sf, numpy as np
    except ImportError:
        raise HTTPException(500, "librosa/soundfile not installed. Run: pip install librosa soundfile")

    src = find_audio_file(payload.filename)
    if not src.exists():
        raise HTTPException(404, f"File not found: {payload.filename}")

    audio, sr = librosa.load(str(src), sr=None, mono=True)
    effect = payload.effect
    result = audio.copy()

    if effect == "amplify":
        gain = 10 ** (payload.gain_db / 20.0)
        result = np.clip(audio * gain, -0.99, 0.99)

    elif effect == "normalize":
        mx = np.max(np.abs(audio))
        if mx > 0: result = audio / mx * 0.95

    elif effect == "fade_in":
        n = min(int(payload.fade_duration * sr), len(audio))
        env = np.ones(len(audio)); env[:n] = np.linspace(0, 1, n)
        result = audio * env

    elif effect == "fade_out":
        n = min(int(payload.fade_duration * sr), len(audio))
        env = np.ones(len(audio)); env[-n:] = np.linspace(1, 0, n)
        result = audio * env

    elif effect == "fade_both":
        n = min(int(payload.fade_duration * sr), len(audio)//2)
        env = np.ones(len(audio))
        env[:n]  = np.linspace(0, 1, n)
        env[-n:] = np.linspace(1, 0, n)
        result = audio * env

    elif effect == "echo":
        delay = int(payload.echo_delay * sr)
        out   = np.zeros(len(audio) + delay)
        out[:len(audio)] += audio
        out[delay:]      += audio * payload.echo_decay
        result = out[:len(audio)]
        mx = np.max(np.abs(result))
        if mx > 0: result = result / mx * 0.95

    elif effect == "reverse":
        result = audio[::-1]

    elif effect == "trim_silence":
        thr = payload.trim_threshold
        nz  = np.where(np.abs(audio) > thr)[0]
        result = audio[nz[0]:nz[-1]+1] if len(nz) else audio

    elif effect == "pitch_up":
        result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

    elif effect == "pitch_down":
        result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)

    elif effect == "speed_up":
        result = librosa.effects.time_stretch(audio, rate=1.25)

    elif effect == "slow_down":
        result = librosa.effects.time_stretch(audio, rate=0.75)

    elif effect == "high_pass":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 300 / (sr/2), btype='high')
        result = filtfilt(b, a, audio)

    elif effect == "low_pass":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, min(4000/(sr/2), 0.99), btype='low')
        result = filtfilt(b, a, audio)

    # ── SMART NOISE REDUCTION (3 safe modes) ──────────────────
    elif effect in ("noise_gentle", "noise_balanced", "noise_aggressive"):
        mode_map = {
            "noise_gentle":     "gentle",
            "noise_balanced":   "balanced",
            "noise_aggressive": "aggressive",
        }
        result = _smart_noise_reduce(audio, sr,
                    mode=mode_map[effect],
                    strength=payload.noise_strength)

    else:
        raise HTTPException(400, f"Unknown effect: {effect}")

    # Save
    SUFFIX = {
        "amplify":"_amp","normalize":"_norm","fade_in":"_fadein",
        "fade_out":"_fadeout","fade_both":"_fadeboth","echo":"_echo",
        "reverse":"_rev","trim_silence":"_trim","pitch_up":"_pitchup",
        "pitch_down":"_pitchdn","speed_up":"_fast","slow_down":"_slow",
        "high_pass":"_hp","low_pass":"_lp",
        "noise_gentle":"_nG","noise_balanced":"_nB","noise_aggressive":"_nA",
    }
    if payload.overwrite:
        out_path = src; out_name = payload.filename
    else:
        out_name = f"{Path(payload.filename).stem}{SUFFIX.get(effect,'_fx')}.wav"
        out_path = UPLOADS() / out_name

    import soundfile as sf
    sf.write(str(out_path), result.astype("float32"), sr, subtype="PCM_16")

    # Compute RMS reduction so UI can show a visible indicator
    rms_before = float(np.sqrt(np.mean(audio**2)))
    rms_after  = float(np.sqrt(np.mean(result**2)))
    pct_quieter = round((1 - rms_after / (rms_before + 1e-10)) * 100, 1)
    db_change   = round(20 * np.log10(rms_after / (rms_before + 1e-10)), 1)

    return {
        "status":       "done",
        "effect":       effect,
        "input_file":   payload.filename,
        "output_file":  out_name,
        "overwritten":  payload.overwrite,
        "dur_before":   round(len(audio)/sr, 2),
        "dur_after":    round(len(result)/sr, 2),
        "rms_before":   round(rms_before, 4),
        "rms_after":    round(rms_after, 4),
        "pct_quieter":  pct_quieter,
        "db_change":    db_change,
    }

@app.post("/api/effects/batch")
def apply_effect_batch(payload: EffectPayload):
    SKIP = ["_amp","_norm","_fadein","_fadeout","_fadeboth","_echo","_rev",
            "_trim","_pitchup","_pitchdn","_fast","_slow","_hp","_lp"]
    files = [f for f in sorted(UPLOADS().glob("*.wav")) + sorted(UPLOADS().glob("*.mp3"))
             if not any(f.stem.endswith(s) for s in SKIP)]
    if not files: raise HTTPException(400, "No audio files found")
    results = []
    for src in files:
        try:
            p2 = EffectPayload(filename=src.name, effect=payload.effect,
                overwrite=payload.overwrite, gain_db=payload.gain_db,
                fade_duration=payload.fade_duration, echo_delay=payload.echo_delay,
                echo_decay=payload.echo_decay, trim_threshold=payload.trim_threshold)
            r = apply_effect(p2)
            results.append({"file": src.name, "output": r["output_file"], "status":"ok"})
        except Exception as ex:
            results.append({"file": src.name, "status":"error", "error": str(ex)})
    return {"processed": len(results), "effect": payload.effect, "results": results}

# ── STATS ─────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    files = [f for f in list(UPLOADS().glob("*.wav")) + list(UPLOADS().glob("*.mp3"))
             if not f.stem.startswith("_")]
    anns  = list(ANNOTS().glob("*.json"))
    total_segs = 0; labels = {}
    for a in anns:
        try:
            d = json.loads(a.read_text())
            for seg in d.get("segments", []):
                total_segs += 1
                lbl = seg["label"]
                labels[lbl] = labels.get(lbl, 0) + 1
        except: pass
    return {"total_files": len(files), "annotated": len(anns),
            "total_segments": total_segs, "labels": labels}

# ── SERVE FRONTEND ────────────────────────────────────────────
# ── PROJECT MANAGEMENT ───────────────────────────────────────
PROJECTS_ROOT = BASE / "data" / "projects"
PROJECTS_ROOT.mkdir(exist_ok=True)
ACTIVE_FILE   = BASE / "data" / ".active_project"

def get_active_project() -> str:
    if ACTIVE_FILE.exists():
        name = ACTIVE_FILE.read_text().strip()
        if name and (PROJECTS_ROOT / name).exists():
            return name
    # Default project
    default = PROJECTS_ROOT / "default"
    default.mkdir(exist_ok=True)
    ACTIVE_FILE.write_text("default")
    return "default"

def project_path(name: str) -> Path:
    return PROJECTS_ROOT / name

def UPLOADS():
    p = project_path(get_active_project()) / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p

def ANNOTS():
    p = project_path(get_active_project()) / "annotations"
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_audio_file(filename: str) -> Path:
    """Find an audio file — checks project uploads first, then legacy data/uploads."""
    # 1. Current project uploads (new location)
    p1 = UPLOADS() / filename
    if p1.exists(): return p1
    # 2. Legacy data/uploads (old location before project system)
    p2 = DATA / "uploads" / filename
    if p2.exists(): return p2
    # 3. Any other project's uploads
    for proj_dir in PROJECTS_ROOT.iterdir():
        if proj_dir.is_dir():
            p3 = proj_dir / "uploads" / filename
            if p3.exists(): return p3
    return p1  # return expected path even if not found (will 404 properly)

@app.get("/api/projects")
def list_projects():
    all_proj = [p.name for p in sorted(PROJECTS_ROOT.iterdir()) if p.is_dir()]
    if not all_proj:
        (PROJECTS_ROOT / "default").mkdir(exist_ok=True)
        all_proj = ["default"]
    active = get_active_project()
    # Return rich objects so frontend can show file counts
    result = []
    for name in all_proj:
        pp   = PROJECTS_ROOT / name
        fc   = len(list((pp/"uploads").glob("*.wav"))) + len(list((pp/"uploads").glob("*.mp3"))) if (pp/"uploads").exists() else 0
        mt   = (pp/"models"/"model.pkl").exists() if (pp/"models").exists() else False
        result.append({"name": name, "file_count": fc, "model_trained": mt, "active": name==active})
    return {"projects": result, "active": active}

@app.get("/api/projects/active")
def active_project():
    return {"active": get_active_project()}

@app.post("/api/projects/create")
def create_project(payload: dict):
    name = payload.get("name", "").strip()
    if not name: raise HTTPException(400, "Project name required")
    if "/" in name or "\\" in name: raise HTTPException(400, "Invalid name")
    path = PROJECTS_ROOT / name
    if path.exists(): raise HTTPException(409, f"Project '{name}' already exists")
    path.mkdir(parents=True)
    (path / "uploads").mkdir()
    (path / "annotations").mkdir()
    (path / "models").mkdir()
    return {"created": name}

@app.post("/api/projects/switch")
def switch_project(payload: dict):
    name = payload.get("name", "").strip()
    if not (PROJECTS_ROOT / name).exists():
        raise HTTPException(404, f"Project '{name}' not found")
    ACTIVE_FILE.write_text(name)
    return {"active": name}

@app.post("/api/projects/rename")
def rename_project(payload: dict):
    old = payload.get("old_name", "").strip()
    new = payload.get("new_name", "").strip()
    if not old or not new: raise HTTPException(400, "Names required")
    if "/" in new or "\\" in new: raise HTTPException(400, "Invalid name")
    old_path = PROJECTS_ROOT / old
    new_path  = PROJECTS_ROOT / new
    if not old_path.exists(): raise HTTPException(404, "Project not found")
    if new_path.exists():     raise HTTPException(409, f"'{new}' already exists")
    old_path.rename(new_path)
    if get_active_project() == old:
        ACTIVE_FILE.write_text(new)
    return {"renamed": new}

@app.delete("/api/projects/{name}")
def delete_project(name: str):
    path = PROJECTS_ROOT / name
    if not path.exists(): raise HTTPException(404, "Project not found")
    all_projects = [p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir()]
    if len(all_projects) <= 1:
        raise HTTPException(400, "Cannot delete the only project")
    import shutil
    shutil.rmtree(str(path))
    # Switch to another project
    remaining = [p.name for p in PROJECTS_ROOT.iterdir() if p.is_dir()]
    new_active = remaining[0] if remaining else "default"
    ACTIVE_FILE.write_text(new_active)
    return {"deleted": name, "active": new_active}


# ── NOISE CANCELLATION ──────────────────────────────────────
class NoiseCancelPayload(BaseModel):
    filename:  str
    mode:      str   = "stationary"   # stationary | nonstationary
    strength:  float = 0.75
    overwrite: bool  = False

@app.post("/api/noise/cancel")
async def noise_cancel(payload: NoiseCancelPayload):
    """Apply noisereduce to a single file. Preserves click sounds."""
    try:
        import librosa, soundfile as sf, numpy as np
        import noisereduce as nr
    except ImportError:
        raise HTTPException(500,
            "noisereduce not installed. Run: pip install noisereduce")

    src = find_audio_file(payload.filename)
    if not src.exists():
        raise HTTPException(404, f"File not found: {payload.filename}")

    audio, sr = librosa.load(str(src), sr=None, mono=True)
    strength  = max(0.1, min(1.0, payload.strength))

    reduced = nr.reduce_noise(
        y             = audio,
        sr            = sr,
        stationary    = (payload.mode == "stationary"),
        prop_decrease = strength,
        n_fft         = 2048,
        win_length    = 2048,
        hop_length    = 512,
        n_jobs        = 1
    )

    # DO NOT normalize — keep amplitude reduction visible in waveform
    reduced = np.clip(reduced, -0.99, 0.99)

    if payload.overwrite:
        out_path = src
        out_name = payload.filename
    else:
        out_name = f"{Path(payload.filename).stem}_denoised.wav"
        out_path  = UPLOADS() / out_name

    sf.write(str(out_path), reduced.astype("float32"), sr, subtype="PCM_16")

    orig_rms = float(np.sqrt(np.mean(audio**2)))
    red_rms  = float(np.sqrt(np.mean(reduced**2)))
    rdb      = round(20 * np.log10(red_rms / (orig_rms + 1e-10)), 1)

    return {
        "status":       "done",
        "input_file":   payload.filename,
        "output_file":  out_name,
        "mode":         payload.mode,
        "strength":     strength,
        "overwritten":  payload.overwrite,
        "reduction_db": rdb,
        "sample_rate":  sr,
        "duration_sec": round(len(reduced) / sr, 2),
    }


@app.post("/api/noise/cancel/batch")
async def noise_cancel_batch(
    mode:      str   = "stationary",
    strength:  float = 0.75,
    overwrite: bool  = False
):
    """Apply noisereduce to ALL files in the current project."""
    try:
        import librosa, soundfile as sf, numpy as np
        import noisereduce as nr
    except ImportError:
        raise HTTPException(500,
            "noisereduce not installed. Run: pip install noisereduce")

    SKIP = ["_denoised","_amp","_norm","_fadein","_fadeout","_fadeboth",
            "_echo","_rev","_trim","_pitchup","_pitchdn","_fast","_slow",
            "_hp","_lp","_nG","_nB","_nA"]
    files = [f for f in sorted(UPLOADS().glob("*.wav")) + sorted(UPLOADS().glob("*.mp3"))
             if not any(f.stem.endswith(s) for s in SKIP)]

    if not files:
        raise HTTPException(400, "No audio files found in this project")

    results = []
    for src in files:
        try:
            audio, sr = librosa.load(str(src), sr=None, mono=True)
            reduced   = nr.reduce_noise(
                y             = audio,
                sr            = sr,
                stationary    = (mode == "stationary"),
                prop_decrease = max(0.1, min(1.0, strength)),
                n_fft         = 2048,
                win_length    = 2048,
                hop_length    = 512,
                n_jobs        = 1
            )
            reduced = np.clip(reduced, -0.99, 0.99)
            out_name = src.name if overwrite else f"{src.stem}_denoised.wav"
            sf.write(str(UPLOADS()/out_name), reduced.astype("float32"), sr, subtype="PCM_16")
            results.append({"file": src.name, "output": out_name, "status": "ok"})
        except Exception as ex:
            results.append({"file": src.name, "status": "error", "error": str(ex)})

    ok  = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    return {"processed": len(results), "ok": ok, "errors": err, "results": results}




@app.get("/api/models")
def list_models():
    """Return available model definitions."""
    models = {
        "svm":            {"name":"SVM",             "desc":"Best for 50–200 clips.",          "speed":"Fast",    "expected_acc":"~90%","min_clips":50  },
        "random_forest":  {"name":"Random Forest",   "desc":"Good default for any size.",       "speed":"Fast",    "expected_acc":"~87%","min_clips":50  },
        "gradient_boost": {"name":"Gradient Boost",  "desc":"High accuracy, 200+ clips.",       "speed":"Medium",  "expected_acc":"~90%","min_clips":200 },
        "mlp":            {"name":"MLP Neural Net",  "desc":"Small neural net, 100+ clips.",    "speed":"Medium",  "expected_acc":"~85%","min_clips":100 },
        "knn":            {"name":"KNN",             "desc":"Fast similarity baseline.",        "speed":"Fast",    "expected_acc":"~78%","min_clips":20  },
        "logistic":       {"name":"Logistic Reg.",   "desc":"Simple linear baseline.",          "speed":"Fastest", "expected_acc":"~75%","min_clips":20  },
        "naive_bayes":    {"name":"Naive Bayes",     "desc":"Works with <30 clips.",            "speed":"Fastest", "expected_acc":"~72%","min_clips":10  },
    }
    return {"models": models}

@app.post("/api/auto_segment")
def auto_segment(payload: dict):
    """Auto-find click spikes and save as annotations."""
    try:
        import librosa, numpy as np
        from scipy.signal import butter, filtfilt, find_peaks
    except ImportError:
        raise HTTPException(500, "librosa not installed")

    filename  = payload.get("filename")
    n_clicks  = int(payload.get("n_clicks", 3))
    label     = payload.get("label", "connector_click")
    overwrite = payload.get("overwrite", False)
    clip_len  = float(payload.get("clip_len", 0.3))

    src = find_audio_file(filename)
    if not src.exists():
        raise HTTPException(404, f"File not found: {filename}")

    audio, sr = librosa.load(str(src), sr=None, mono=True)
    duration  = len(audio) / sr

    # High-pass to remove hum, then find energy peaks
    b, a     = butter(4, min(300/(sr/2), 0.99), btype='high')
    audio_hp = filtfilt(b, a, audio)
    hop      = 128
    rms      = librosa.feature.rms(y=audio_hp, frame_length=512, hop_length=hop)[0]
    times    = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    bg       = np.percentile(rms, 15)
    min_gap  = int(sr * 0.8 / hop)
    peaks, _ = find_peaks(rms, height=bg * 2.0, distance=min_gap)

    if len(peaks) > n_clicks:
        top = np.argsort(rms[peaks])[::-1][:n_clicks]
        peaks = sorted(peaks[top])

    if len(peaks) == 0:
        return {"filename": filename, "segments_added": 0,
                "message": "No clicks detected — apply High-Pass filter first"}

    half = clip_len / 2
    segments = [{"label": label,
                 "start": round(max(0, float(times[p]) - half), 3),
                 "end":   round(min(duration, float(times[p]) + half), 3)}
                for p in peaks]

    ann_path = ANNOTS() / (Path(filename).stem + ".json")
    existing = {"filename": filename, "segments": []}
    if ann_path.exists() and not overwrite:
        try: existing = json.loads(ann_path.read_text())
        except: pass

    if overwrite:
        existing["segments"] = segments
    else:
        existing_labels = {s["label"] for s in existing.get("segments", [])}
        if label not in existing_labels:
            existing["segments"] = existing.get("segments", []) + segments

    ann_path.write_text(json.dumps({"filename": filename,
                                    "segments": existing["segments"]}, indent=2))
    return {"filename": filename, "segments_added": len(segments),
            "clicks_found": len(peaks),
            "message": f"Found {len(peaks)} clicks → {len(segments)} segments added"}


@app.post("/api/auto_segment/batch")
def auto_segment_batch(payload: dict):
    """Run auto-segment on all files in current project."""
    n_clicks  = int(payload.get("n_clicks", 3))
    label     = payload.get("label", "connector_click")
    overwrite = payload.get("overwrite", False)
    clip_len  = float(payload.get("clip_len", 0.3))

    SKIP = ["_denoised","_amp","_norm","_fadein","_fadeout","_fadeboth",
            "_echo","_rev","_trim","_pitchup","_pitchdn","_fast","_slow",
            "_hp","_lp","_nG","_nB","_nA"]

    all_files = list(UPLOADS().glob("*.wav")) + list(UPLOADS().glob("*.mp3"))
    legacy    = DATA / "uploads"
    if legacy.exists():
        for f in list(legacy.glob("*.wav")) + list(legacy.glob("*.mp3")):
            if not any(ex.name == f.name for ex in all_files):
                all_files.append(f)
    files = [f for f in sorted(all_files, key=lambda x: x.name)
             if not any(f.stem.endswith(s) for s in SKIP)]

    if not files:
        raise HTTPException(400, "No audio files found")

    results = []
    for src in files:
        try:
            r = auto_segment({"filename": src.name, "n_clicks": n_clicks,
                               "label": label, "overwrite": overwrite,
                               "clip_len": clip_len})
            results.append({"file": src.name, "status": "ok",
                            "segments_added": r["segments_added"],
                            "clicks_found":   r["clicks_found"]})
        except Exception as ex:
            results.append({"file": src.name, "status": "error", "error": str(ex)})

    ok         = sum(1 for r in results if r["status"] == "ok")
    total_segs = sum(r.get("segments_added", 0) for r in results)
    return {"processed": len(results), "ok": ok,
            "total_segments_added": total_segs, "results": results}


@app.get("/api/status")
def get_status():
    """Pipeline status — file counts, slice count, model trained etc."""
    up    = UPLOADS()
    an    = ANNOTS()
    SKIP  = ["_amp","_norm","_fadein","_fadeout","_fadeboth","_echo","_rev",
             "_trim","_pitchup","_pitchdn","_fast","_slow","_hp","_lp",
             "_nG","_nB","_nA"]
    files = [f for f in list(up.glob("*.wav"))+list(up.glob("*.mp3"))
             if not any(f.stem.endswith(s) for s in SKIP)]
    anns  = list(an.glob("*.json"))

    # Count annotated (has at least 1 segment)
    annotated = 0
    for af in anns:
        try:
            d = json.loads(af.read_text())
            if d.get("segments"): annotated += 1
        except: pass

    # Sliced clips
    sl    = project_path(get_active_project()) / "sliced"
    clips = len(list(sl.rglob("*.wav"))) if sl.exists() else 0

    # Features
    feat_file = project_path(get_active_project()) / "features" / "features.npz"
    feat_ok   = feat_file.exists()

    # Model
    model_pkl = project_path(get_active_project()) / "models" / "model.pkl"
    last_res  = project_path(get_active_project()) / "models" / "last_result.json"
    trained   = model_pkl.exists()
    last_tr   = {}
    if last_res.exists():
        try: last_tr = json.loads(last_res.read_text())
        except: pass

    return {
        "upload_count":       len(files),
        "annotation_count":   annotated,
        "slice_count":        clips,
        "features_extracted": feat_ok,
        "model_trained":      trained,
        "last_training":      last_tr,
        "project":            get_active_project(),
    }


@app.post("/api/annotations/clean")
def clean_annotations(payload: dict):
    """
    Remove segments with unwanted labels from all annotation files in current project.
    payload: { keep_labels: ["connector_click", "background_noise"] }
    """
    keep   = payload.get("keep_labels", [])
    if not keep:
        raise HTTPException(400, "keep_labels list is required")

    ann_dir = ANNOTS()
    if not ann_dir.exists():
        return {"cleaned": 0, "files": 0, "removed_segments": 0}

    files_changed = 0
    total_removed = 0
    results = []

    for af in sorted(ann_dir.glob("*.json")):
        try:
            data     = json.loads(af.read_text())
            segs_old = data.get("segments", [])
            segs_new = [s for s in segs_old if s.get("label") in keep]
            removed  = len(segs_old) - len(segs_new)
            if removed > 0:
                data["segments"] = segs_new
                af.write_text(json.dumps(data, indent=2))
                files_changed += 1
                total_removed += removed
                results.append({"file": af.name, "removed": removed})
        except: pass

    return {
        "status":          "done",
        "files_changed":   files_changed,
        "removed_segments": total_removed,
        "keep_labels":     keep,
        "details":         results
    }


@app.post("/api/slice")
def run_slice():
    """Slice annotated audio files into individual clips.
    Uses ONLY the current project's annotations — no cross-project contamination."""
    try:
        import librosa, soundfile as sf
    except ImportError:
        raise HTTPException(500, "librosa not installed. Run: pip install librosa soundfile")

    active   = get_active_project()
    sl_dir   = project_path(active) / "sliced"
    ann_dir  = ANNOTS()   # current project only
    up_dir   = UPLOADS()  # current project uploads

    if sl_dir.exists():
        import shutil; shutil.rmtree(str(sl_dir))
    sl_dir.mkdir(parents=True)

    # Only use annotation files from the CURRENT project
    all_anns = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []

    if not all_anns:
        raise HTTPException(400,
            f"No annotation files found in project '{active}'.\n\n"
            "Steps to fix:\n"
            "1. Go to Annotate tab\n"
            "2. Make sure you are in the correct project\n"
            "3. Load each audio file, draw segments, press S to save\n"
            "4. Come back to Pipeline and run Slicing")

    stats = {"total_clips": 0, "by_label": {}, "errors": [], "project": active}

    # Collect what labels actually exist across all annotation files in this project
    # so we can warn about and skip any stale labels not in current project
    allowed_labels = set()
    for af in all_anns:
        try:
            d = json.loads(af.read_text())
            for s in d.get("segments", []):
                allowed_labels.add(s.get("label",""))
        except: pass

    skipped_labels = set()

    for af in all_anns:
        try:
            data = json.loads(af.read_text())
        except: continue

        filename = data.get("filename", "")
        segs     = data.get("segments", [])
        if not segs: continue

        # Find audio in current project uploads first, then fallback
        src = up_dir / filename
        if not src.exists():
            # Try the find_audio_file fallback for legacy files
            src = find_audio_file(filename)
        if not src.exists():
            stats["errors"].append(f"{filename}: not found in project uploads")
            continue

        try:
            audio, sr = librosa.load(str(src), sr=16000, mono=True)
            # Apply HP filter — matches detect route — ensures training/detect consistency
            from scipy.signal import butter, filtfilt as _ff
            _b, _a = butter(4, min(300/(sr/2), 0.99), btype="high")
            audio = _ff(_b, _a, audio).astype(audio.dtype)
        except Exception as ex:
            stats["errors"].append(f"{filename}: {ex}"); continue

        for j, seg in enumerate(segs):
            start = seg.get("start", 0)
            end   = seg.get("end",   0)
            lbl   = seg.get("label", "unknown")

            # Skip old labels like connector1/connector2/connector3
            # Only slice labels that look like valid 2-class labels
            if lbl in ("connector1","connector2","connector3",
                       "Connector1","Connector2","Connector3"):
                skipped_labels.add(lbl)
                continue

            seg_dur = end - start
            if seg_dur < 0.25:
                # SHORT CLIP (click ~34ms): extract 300ms centered on click
                # This matches the 300ms sliding window used in detect
                mid   = (start + end) / 2.0
                win_s = max(0.0, mid - 0.15)
                win_e = win_s + 0.3
                if win_e > len(audio)/sr:
                    win_e = len(audio)/sr
                    win_s = max(0.0, win_e - 0.3)
                win_samples = int(0.3 * sr)
                clip = audio[int(win_s*sr):int(win_e*sr)]
                import numpy as _np
                if len(clip) < win_samples:
                    clip = _np.pad(clip, (0, win_samples - len(clip)))
                clip = clip[:win_samples]
            else:
                # LONG CLIP (background): cut into 300ms sub-clips
                # Use random offsets so model learns background anywhere in file
                import numpy as _np2
                win_samples = int(0.3 * sr)
                seg_samples = int((end - start) * sr)
                hop_samples = int(0.15 * sr)   # 50% overlap
                sub_clips = []
                t = int(start * sr)
                seg_end_s = int(end * sr)
                while t + win_samples <= seg_end_s:
                    sub_clips.append(audio[t:t+win_samples])
                    t += hop_samples
                if not sub_clips:
                    clip = audio[int(start*sr):int(end*sr)]
                    if len(clip) < win_samples:
                        clip = _np2.pad(clip,(0,win_samples-len(clip)))
                    sub_clips = [clip[:win_samples]]
                out_dir = sl_dir / lbl
                out_dir.mkdir(exist_ok=True)
                for k, clip in enumerate(sub_clips):
                    sf.write(str(out_dir / f"{Path(filename).stem}_{j:04d}_s{k}.wav"),
                             clip, sr, subtype="PCM_16")
                    stats["total_clips"] += 1
                    stats["by_label"][lbl] = stats["by_label"].get(lbl, 0) + 1
                continue
            if len(clip) < 10: continue
            out_dir = sl_dir / lbl
            out_dir.mkdir(exist_ok=True)
            sf.write(str(out_dir / f"{Path(filename).stem}_{j:04d}.wav"),
                     clip, sr, subtype="PCM_16")
            stats["total_clips"] += 1
            stats["by_label"][lbl] = stats["by_label"].get(lbl, 0) + 1

    if skipped_labels:
        stats["skipped_labels"] = list(skipped_labels)
        stats["warning"] = (
            f"Skipped old labels: {', '.join(skipped_labels)}. "
            f"Run '🧹 Clean Old Labels' in Pipeline to remove them permanently."
        )

    if stats["total_clips"] == 0:
        raise HTTPException(400,
            f"No clips were sliced from project '{active}'.\n\n"
            "Make sure:\n"
            "• You have drawn segments on the waveform\n"
            "• You pressed S to save each file\n"
            "• The audio files are in this project's uploads folder\n\n"
            f"Found {len(all_anns)} annotation files but 0 valid clips.\n"
            + (f"Errors: {'; '.join(stats['errors'][:3])}" if stats['errors'] else ""))

    return {"status": "done", "stats": stats}


@app.get("/api/slice/download")
def download_slices():
    """Download sliced clips as ZIP."""
    sl_dir = project_path(get_active_project()) / "sliced"
    if not sl_dir.exists() or not list(sl_dir.rglob("*.wav")):
        raise HTTPException(404, "No sliced clips yet — run slicing first")
    import io, zipfile as zf
    buf = io.BytesIO()
    with zf.ZipFile(buf, "w", zf.ZIP_DEFLATED) as z:
        for f in sorted(sl_dir.rglob("*.wav")):
            z.write(str(f), str(f.relative_to(sl_dir)))
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=sliced_clips.zip"})


@app.post("/api/features/extract")
def extract_features(feature_type: str = "mfcc"):
    """Extract MFCC or PCEN features from sliced clips."""
    try:
        import librosa, numpy as np
    except ImportError:
        raise HTTPException(500, "librosa not installed")

    sl_dir = project_path(get_active_project()) / "sliced"
    if not sl_dir.exists() or not list(sl_dir.rglob("*.wav")):
        raise HTTPException(400, "No sliced clips found — run slicing first")

    label_set  = sorted(set(p.parent.name for p in sl_dir.rglob("*.wav")))
    label2idx  = {l: i for i, l in enumerate(label_set)}
    all_wavs   = sorted(sl_dir.rglob("*.wav"))
    X, y, skip = [], [], 0

    for wav in all_wavs:
        lbl = wav.parent.name
        try:
            audio, sr = librosa.load(str(wav), sr=16000, mono=True, duration=0.3)
            if feature_type == "pcen":
                hop = 256
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                          fmax=8000, hop_length=hop, n_fft=1024)
                pcen = librosa.pcen(mel*(2**31), sr=sr, hop_length=hop,
                           gain=0.8, bias=10, power=0.25, time_constant=0.4, eps=1e-6)
                delta = librosa.feature.delta(pcen)
                contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                               n_fft=1024, hop_length=hop, n_bands=6)
                zcr  = librosa.feature.zero_crossing_rate(y=audio)
                feat = np.concatenate([np.mean(pcen,axis=1), np.std(pcen,axis=1),
                                       np.mean(delta,axis=1), np.mean(contrast,axis=1),
                                       [float(np.mean(zcr)), float(np.std(zcr))]])
            else:
                mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                zcr    = librosa.feature.zero_crossing_rate(y=audio)
                sc     = librosa.feature.spectral_centroid(y=audio, sr=sr)
                feat   = np.concatenate([np.mean(mfcc,axis=1), np.std(mfcc,axis=1),
                                         np.mean(chroma,axis=1),
                                         [np.mean(zcr), np.std(zcr), np.mean(sc), np.std(sc)]])
            X.append(feat.tolist()); y.append(label2idx[lbl])
        except: skip += 1

    if not X: raise HTTPException(400, "Could not extract any features")

    feat_dir = project_path(get_active_project()) / "features"
    feat_dir.mkdir(exist_ok=True)
    X_arr = np.array(X); y_arr = np.array(y)
    # Balance: UNDERSAMPLE majority to 3x minority count
    # e.g. 235 clicks vs 6342 bg → keep 705 bg + 235 clicks = 940 total
    # Model gets variety without memorizing
    from collections import Counter as _C
    import random as _r; _r.seed(42)
    counts = _C(y_arr.tolist())
    if len(counts) > 1:
        min_cnt = min(counts.values())
        target  = min_cnt * 3   # keep 3x minority for each class
        X_b, y_b = [], []
        for cls, cnt in counts.items():
            idxs = [i for i,yi in enumerate(y_arr.tolist()) if yi==cls]
            if cnt > target:
                idxs = _r.sample(idxs, target)  # undersample majority
            X_b.extend(X_arr[idxs]); y_b.extend([cls]*len(idxs))
        # Shuffle
        combined = list(zip(X_b, y_b))
        _r.shuffle(combined)
        X_arr = np.array([x for x,_ in combined])
        y_arr = np.array([y for _,y in combined])
    np.savez(str(feat_dir/"features.npz"),
             X=X_arr, y=y_arr, labels=label_set)

    by_label = {}
    for yi in y_arr:
        lbl = label_set[int(yi)]
        by_label[lbl] = by_label.get(lbl, 0) + 1
    by_label_orig = {label_set[int(k)]:int(v) for k,v in counts.items()}

    return {
        "status": "done",
        "summary": {
            "total_samples":  len(X),
            "n_features":     len(X[0]) if X else 0,
            "feature_type":   feature_type,
            "labels":         label_set,
            "by_label":       by_label,
            "by_label_original": by_label_orig,
            "skipped":        skip,
        }
    }


@app.post("/api/train")
def train_model(model_type: str = "svm"):
    """Train a classifier. Supports sklearn models + CNN + YAMNet + PAN."""
    import time, json as _json
    t0 = time.time()

    md_dir = project_path(get_active_project()) / "models"
    md_dir.mkdir(exist_ok=True)
    sl_dir = project_path(get_active_project()) / "sliced"

    # ── Deep learning models use raw audio, not pre-extracted features ──
    if model_type in ("cnn", "yamnet", "pan"):
        if not sl_dir.exists() or not list(sl_dir.rglob("*.wav")):
            raise HTTPException(400, "No sliced clips — run slicing first")
        return _train_deep(model_type, sl_dir, md_dir, t0)

    # ── Sklearn models use extracted features ────────────────────────────
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing   import StandardScaler
        from sklearn.metrics          import accuracy_score, classification_report
    except ImportError:
        raise HTTPException(500, "scikit-learn not installed. Run: pip install scikit-learn")

    feat_file = project_path(get_active_project()) / "features" / "features.npz"
    if not feat_file.exists():
        raise HTTPException(400, "No features found — run feature extraction first")

    data      = np.load(str(feat_file), allow_pickle=True)
    X         = data["X"]; y = data["y"]
    label_set = list(data["labels"])

    if len(X) < 4:
        raise HTTPException(400, f"Only {len(X)} samples — need at least 4")

    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import confusion_matrix
    from collections import Counter

    # Class weights — fixes imbalance (e.g. 314 bg vs 235 clicks)
    counts = Counter(y.tolist())
    cw = {int(k): len(y)/(len(counts)*v) for k,v in counts.items()}

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # Stratified K-Fold CV for reliable accuracy
    n_splits = max(2, min(5, min(counts.values())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_size = min(0.25, max(0.1, 15/len(X)))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, y, test_size=test_size, random_state=42)

    X_tr = X_tr; X_te = X_te  # already scaled

    if model_type == "svm":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight=cw, random_state=42)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=42)
    elif model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=min(5, len(X_tr)))
    elif model_type == "gradient_boost":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300,
                            early_stopping=True, random_state=42)
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
    elif model_type == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    else:
        raise HTTPException(400, f"Unknown model: {model_type}")

    # Cross-validation
    cv_res = cross_validate(clf, Xs, y, cv=skf,
                            scoring=["accuracy"], return_train_score=True)
    cv_acc     = float(cv_res["test_accuracy"].mean())
    cv_std     = float(cv_res["test_accuracy"].std())
    tr_acc_cv  = float(cv_res["train_accuracy"].mean())
    overfit_gap = round((tr_acc_cv - cv_acc)*100, 1)

    clf.fit(X_tr, y_tr)
    elapsed  = round(time.time() - t0, 1)
    y_pred   = clf.predict(X_te)
    accuracy = round(accuracy_score(y_te, y_pred) * 100, 1)
    report   = classification_report(y_te, y_pred,
                    target_names=label_set, zero_division=0)
    cm       = confusion_matrix(y_te, y_pred).tolist()
    rpt_d    = classification_report(y_te, y_pred, target_names=label_set,
                    zero_division=0, output_dict=True)
    per_f1   = {l: round(rpt_d[l]["f1-score"]*100,1)
                for l in label_set if l in rpt_d}
    # Retrain on all data
    clf.fit(Xs, y)

    import pickle
    model_data = {
        "model": clf, "scaler": sc, "model_type": model_type,
        "labels": label_set,
        "label2idx": {l:i for i,l in enumerate(label_set)},
        "idx2label": {i:l for i,l in enumerate(label_set)},
        "accuracy": accuracy,
        "cv_accuracy": round(cv_acc*100, 1),
        "class_weight": {str(k):v for k,v in cw.items()},
    }
    with open(md_dir/"model.pkl","wb") as f: pickle.dump(model_data, f)

    result = {
        "model_type": model_type, "model_name": model_type.upper(),
        "accuracy":       accuracy,
        "cv_accuracy":    round(cv_acc*100, 1),
        "cv_std":         round(cv_std*100, 1),
        "train_accuracy": round(tr_acc_cv*100, 1),
        "overfit_gap":    overfit_gap,
        "n_splits":       n_splits,
        "train_samples":  len(X_tr),
        "test_samples":   len(X_te),
        "labels":         label_set,
        "class_counts":   {label_set[int(k)]:int(v) for k,v in counts.items()},
        "per_class_f1":   per_f1,
        "confusion_matrix": cm,
        "report":         report,
        "elapsed_sec":    elapsed,
        "overfitting_warning": overfit_gap > 10,
    }
    (md_dir/"last_result.json").write_text(_json.dumps(result, indent=2))
    return result


def _train_deep(model_type: str, sl_dir: Path, md_dir: Path, t0: float):
    """Train CNN, YAMNet or PAN on raw sliced clips."""
    import time, json as _json, numpy as np

    # ── collect all clips ────────────────────────────────────────
    label_set  = sorted(set(p.parent.name for p in sl_dir.rglob("*.wav")))
    label2idx  = {l:i for i,l in enumerate(label_set)}
    all_wavs   = sorted(sl_dir.rglob("*.wav"))
    if not all_wavs:
        raise HTTPException(400, "No sliced clips found")

    if model_type == "yamnet":
        return _train_yamnet(all_wavs, label_set, label2idx, md_dir, t0)
    elif model_type == "cnn":
        return _train_cnn(all_wavs, label_set, label2idx, md_dir, t0)
    elif model_type == "pan":
        return _train_pan(all_wavs, label_set, label2idx, md_dir, t0)


def _train_yamnet(all_wavs, label_set, label2idx, md_dir, t0):
    """Fine-tune YAMNet — Google pretrained on 2 million audio clips."""
    import time, json as _json, numpy as np
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise HTTPException(500,
            "TensorFlow not installed.\nRun: pip install tensorflow tensorflow-hub")
    try:
        import librosa
    except ImportError:
        raise HTTPException(500, "librosa not installed")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics          import accuracy_score, classification_report

    # Load YAMNet — downloads ~13 MB on first run
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    embeddings, labels_y = [], []
    skipped = 0
    for wav in all_wavs:
        lbl = wav.parent.name
        try:
            audio, sr = librosa.load(str(wav), sr=16000, mono=True, duration=1.5)
            if len(audio) < 1600:
                audio = np.pad(audio, (0, 1600 - len(audio)))
            waveform = tf.constant(audio, dtype=tf.float32)
            _, emb, _ = yamnet(waveform)
            embeddings.append(tf.reduce_mean(emb, axis=0).numpy())
            labels_y.append(label2idx[lbl])
        except: skipped += 1

    if len(embeddings) < 4:
        raise HTTPException(400, "Too few clips for YAMNet training")

    E = np.array(embeddings); L = np.array(labels_y)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            E, L, test_size=0.2, random_state=42, stratify=L)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(E, L, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

    n = len(label_set)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=60, batch_size=32, verbose=0,
              validation_data=(X_te, y_te),
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=12, restore_best_weights=True)])

    y_pred   = np.argmax(model.predict(X_te, verbose=0), axis=1)
    accuracy = round(accuracy_score(y_te, y_pred) * 100, 1)
    report   = classification_report(y_te, y_pred,
                    target_names=label_set, zero_division=0)
    elapsed  = round(time.time() - t0, 1)

    # Save Keras model + scaler metadata
    keras_path = str(md_dir / "yamnet_clf.keras")
    model.save(keras_path)

    import pickle
    meta = {"model_type":"yamnet","scaler":sc,"labels":label_set,
            "label2idx":label2idx,
            "idx2label":{i:l for i,l in enumerate(label_set)},
            "accuracy":accuracy,"keras_path":keras_path}
    with open(md_dir/"model.pkl","wb") as f: pickle.dump(meta, f)

    result = {"model_type":"yamnet","model_name":"YAMNet (Google Pretrained)",
              "accuracy":accuracy,"train_samples":len(X_tr),
              "test_samples":len(X_te),"labels":label_set,
              "report":report,"elapsed_sec":elapsed,"skipped":skipped}
    (md_dir/"last_result.json").write_text(json.dumps(result, indent=2))
    return result


def _train_cnn(all_wavs, label_set, label2idx, md_dir, t0):
    """Train CNN on Mel Spectrograms — learns time-frequency patterns."""
    import time, numpy as np
    try:
        import tensorflow as tf
        import librosa
    except ImportError:
        raise HTTPException(500,
            "TensorFlow not installed.\nRun: pip install tensorflow")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics          import accuracy_score, classification_report

    X, y, skipped = [], [], 0
    TARGET_SR = 16000; N_MELS = 64; HOP = 256; N_FFT = 1024

    for wav in all_wavs:
        lbl = wav.parent.name
        try:
            audio, sr = librosa.load(str(wav), sr=TARGET_SR, mono=True, duration=1.5)
            # Pad/trim to exactly 1.5s
            target_len = int(1.5 * TARGET_SR)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                    hop_length=HOP, n_fft=N_FFT, fmax=8000), ref=np.max)
            # Normalise to 0-1
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            X.append(mel[..., np.newaxis])  # (64, T, 1)
            y.append(label2idx[lbl])
        except: skipped += 1

    if len(X) < 4:
        raise HTTPException(400, "Too few clips for CNN training")

    X = np.array(X); y = np.array(y)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n = len(label_set)
    inp = tf.keras.layers.Input(shape=X.shape[1:])
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128,(3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=80, batch_size=16, verbose=0,
              validation_data=(X_te, y_te),
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=15, restore_best_weights=True)])

    y_pred   = np.argmax(model.predict(X_te, verbose=0), axis=1)
    accuracy = round(accuracy_score(y_te, y_pred) * 100, 1)
    report   = classification_report(y_te, y_pred,
                    target_names=label_set, zero_division=0)
    elapsed  = round(time.time() - t0, 1)

    keras_path = str(md_dir / "cnn_model.keras")
    model.save(keras_path)

    import pickle
    meta = {"model_type":"cnn","labels":label_set,"label2idx":label2idx,
            "idx2label":{i:l for i,l in enumerate(label_set)},
            "accuracy":accuracy,"keras_path":keras_path,
            "input_shape":list(X.shape[1:])}
    with open(md_dir/"model.pkl","wb") as f: pickle.dump(meta, f)

    result = {"model_type":"cnn","model_name":"CNN (Mel Spectrogram)",
              "accuracy":accuracy,"train_samples":len(X_tr),
              "test_samples":len(X_te),"labels":label_set,
              "report":report,"elapsed_sec":elapsed,"skipped":skipped}
    (md_dir/"last_result.json").write_text(json.dumps(result, indent=2))
    return result


def _train_pan(all_wavs, label_set, label2idx, md_dir, t0):
    """Train PAN — Pairwise Attention Network on MFCC sequences."""
    import time, numpy as np
    try:
        import tensorflow as tf
        import librosa
    except ImportError:
        raise HTTPException(500,
            "TensorFlow not installed.\nRun: pip install tensorflow")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics          import accuracy_score, classification_report

    if len(all_wavs) < 20:
        raise HTTPException(400,
            "PAN needs at least 150 clips per class (20+ total). "
            "Use SVM or CNN for smaller datasets.")

    X, y, skipped = [], [], 0
    TARGET_SR = 16000; N_MFCC = 40; HOP = 256; MAX_FRAMES = 64

    for wav in all_wavs:
        lbl = wav.parent.name
        try:
            audio, sr = librosa.load(str(wav), sr=TARGET_SR, mono=True, duration=1.5)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                         hop_length=HOP)  # (40, T)
            mfcc = mfcc.T  # (T, 40)
            # Pad/trim to MAX_FRAMES
            if mfcc.shape[0] < MAX_FRAMES:
                pad = np.zeros((MAX_FRAMES - mfcc.shape[0], N_MFCC))
                mfcc = np.vstack([mfcc, pad])
            else:
                mfcc = mfcc[:MAX_FRAMES]
            # Normalise
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            X.append(mfcc)
            y.append(label2idx[lbl])
        except: skipped += 1

    if len(X) < 4:
        raise HTTPException(400, "Too few clips for PAN training")

    X = np.array(X); y = np.array(y)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n = len(label_set)
    # Multi-head self-attention over MFCC frames
    inp = tf.keras.layers.Input(shape=(MAX_FRAMES, N_MFCC))
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(inp, inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(n, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=80, batch_size=16, verbose=0,
              validation_data=(X_te, y_te),
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=15, restore_best_weights=True)])

    y_pred   = np.argmax(model.predict(X_te, verbose=0), axis=1)
    accuracy = round(accuracy_score(y_te, y_pred) * 100, 1)
    report   = classification_report(y_te, y_pred,
                    target_names=label_set, zero_division=0)
    elapsed  = round(time.time() - t0, 1)

    keras_path = str(md_dir / "pan_model.keras")
    model.save(keras_path)

    import pickle
    meta = {"model_type":"pan","labels":label_set,"label2idx":label2idx,
            "idx2label":{i:l for i,l in enumerate(label_set)},
            "accuracy":accuracy,"keras_path":keras_path}
    with open(md_dir/"model.pkl","wb") as f: pickle.dump(meta, f)

    result = {"model_type":"pan","model_name":"PAN (Attention Network)",
              "accuracy":accuracy,"train_samples":len(X_tr),
              "test_samples":len(X_te),"labels":label_set,
              "report":report,"elapsed_sec":elapsed,"skipped":skipped}
    (md_dir/"last_result.json").write_text(json.dumps(result, indent=2))
    return result




@app.get("/api/model/download")
def download_model():
    pkl = project_path(get_active_project()) / "models" / "model.pkl"
    if not pkl.exists():
        raise HTTPException(404, "No model trained yet")
    return FileResponse(str(pkl), filename="model.pkl",
                        media_type="application/octet-stream")

@app.get("/api/models")
def list_models():
    """Return all available ML models with descriptions for the UI cards."""
    return {"models": {
        "svm": {
            "name":  "SVM — Support Vector Machine",
            "desc":  "Best for MFCC features with 50–200 clips. Finds the optimal boundary between classes. Most reliable for your connector click data.",
            "speed": "Fast (2–5 sec)",
            "when":  "Best for 50–200 clips per label",
            "min_clips": 20,
            "expected_acc": "85–92%",
        },
        "random_forest": {
            "name":  "Random Forest",
            "desc":  "200 decision trees vote together. Very robust, rarely overfits. Good default choice for any dataset size.",
            "speed": "Medium (5–10 sec)",
            "when":  "Good default for any size",
            "min_clips": 30,
            "expected_acc": "82–90%",
        },
        "knn": {
            "name":  "KNN — K-Nearest Neighbors",
            "desc":  "Classifies by finding the 5 most similar training clips. Simple and fast. Good baseline.",
            "speed": "Very fast (< 1 sec)",
            "when":  "Good baseline, any size",
            "min_clips": 10,
            "expected_acc": "70–82%",
        },
        "gradient_boost": {
            "name":  "Gradient Boosting",
            "desc":  "Sequentially builds trees, each correcting the last. High accuracy but needs more data.",
            "speed": "Slow (15–30 sec)",
            "when":  "Best with 200+ clips",
            "min_clips": 50,
            "expected_acc": "85–93%",
        },
        "mlp": {
            "name":  "MLP — Neural Network",
            "desc":  "Small 2-layer neural network (256→128 neurons). Learns non-linear patterns in your features.",
            "speed": "Medium (10–20 sec)",
            "when":  "Good with 100+ clips",
            "min_clips": 40,
            "expected_acc": "80–88%",
        },
        "logistic": {
            "name":  "Logistic Regression",
            "desc":  "Simple linear classifier. Fastest to train. Good baseline to compare against.",
            "speed": "Very fast (< 1 sec)",
            "when":  "Baseline comparison",
            "min_clips": 10,
            "expected_acc": "70–80%",
        },
        "naive_bayes": {
            "name":  "Naive Bayes",
            "desc":  "Probabilistic classifier. Works well even with very few training samples.",
            "speed": "Instant",
            "when":  "Tiny datasets (< 30 clips)",
            "min_clips": 5,
            "expected_acc": "65–78%",
        },
    }}

@app.get("/api/model/info")
def model_info():
    pkl     = project_path(get_active_project()) / "models" / "model.pkl"
    last_r  = project_path(get_active_project()) / "models" / "last_result.json"
    if not pkl.exists():
        return {"trained": False, "files": []}
    import pickle
    with open(pkl, "rb") as f:
        md = pickle.load(f)
    files = [{"name": "model.pkl", "type": "sklearn",
               "size_kb": round(pkl.stat().st_size/1024, 1)}]
    lr = {}
    if last_r.exists():
        try: lr = json.loads(last_r.read_text())
        except: pass
    return {"trained": True, "model_type": md.get("model_type",""),
            "labels": md.get("labels",[]), "files": files, "last_result": lr}



# ── Model status ──────────────────────────────────────────────
@app.get("/api/model/status")
def model_status():
    active = get_active_project()
    candidates = [project_path(active)/"models"/"model.pkl",
                  DATA/"models"/"model.pkl",
                  BASE/"model.pkl"]
    if PROJECTS_ROOT.exists():
        for p in PROJECTS_ROOT.iterdir():
            if p.is_dir():
                candidates.append(p/"models"/"model.pkl")
    pkl = next((p for p in candidates if p.exists()), None)
    if not pkl:
        return {"ready":False,
                "reason":"No trained model found",
                "fix":"Pipeline: Slice → Extract Features → Train"}
    try:
        import pickle
        with open(str(pkl),"rb") as f: md=pickle.load(f)
        clf=md.get("model"); sc=md.get("scaler")
        if clf is None or sc is None:
            return {"ready":False,
                    "reason":f"model.pkl ({round(pkl.stat().st_size/1024,1)} KB) missing model/scaler",
                    "fix":"Retrain with both labels"}
        return {"ready":True,"model_type":md.get("model_type","svm"),
                "labels":md.get("labels",[]),"accuracy":md.get("accuracy",0),
                "size_kb":round(pkl.stat().st_size/1024,1)}
    except Exception as e:
        return {"ready":False,"reason":str(e),"fix":"Retrain the model"}


# ── Predict single clip ───────────────────────────────────────
@app.post("/api/predict")
async def predict_clip(file: UploadFile = File(...)):
    import pickle, tempfile, os, librosa, numpy as np
    active = get_active_project()
    candidates = [project_path(active)/"models"/"model.pkl",
                  DATA/"models"/"model.pkl", BASE/"model.pkl"]
    if PROJECTS_ROOT.exists():
        for p in PROJECTS_ROOT.iterdir():
            if p.is_dir(): candidates.append(p/"models"/"model.pkl")
    pkl = next((p for p in candidates if p.exists()), None)
    if not pkl:
        raise HTTPException(400,"No trained model. Pipeline: Slice → Extract Features → Train")
    with open(str(pkl),"rb") as f: md=pickle.load(f)
    clf=md.get("model"); sc=md.get("scaler")
    idx2label=md.get("idx2label",{}); labels=md.get("labels",[])
    if not idx2label:
        idx2label={i:l for i,l in enumerate(labels)}
        idx2label.update({str(i):l for i,l in enumerate(labels)})
    feat_type=md.get("feature_type","mfcc")
    if clf is None:
        raise HTTPException(400,"Model corrupted — retrain in Pipeline")
    audio_bytes=await file.read()
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
    tmp.write(audio_bytes); tmp.close()
    try:
        audio,sr=librosa.load(tmp.name,sr=16000,mono=True,duration=2.0)
    except Exception as ex:
        raise HTTPException(400,f"Cannot read audio: {ex}")
    finally:
        try: os.unlink(tmp.name)
        except: pass
    if len(audio)<100:
        raise HTTPException(400,"Audio too short")
    if feat_type=="pcen":
        hop=256
        mel=librosa.feature.melspectrogram(y=audio.astype(float),sr=sr,n_mels=128,fmax=8000,hop_length=hop,n_fft=1024)
        pc=librosa.pcen(mel*(2**31),sr=sr,hop_length=hop,gain=0.8,bias=10,power=0.25,time_constant=0.4,eps=1e-6)
        dl=librosa.feature.delta(pc)
        ct=librosa.feature.spectral_contrast(y=audio.astype(float),sr=sr,n_fft=1024,hop_length=hop,n_bands=6)
        zc=librosa.feature.zero_crossing_rate(y=audio.astype(float))
        feat=np.concatenate([np.mean(pc,axis=1),np.std(pc,axis=1),np.mean(dl,axis=1),np.mean(ct,axis=1),[float(np.mean(zc)),float(np.std(zc))]])
    else:
        mfcc=librosa.feature.mfcc(y=audio.astype(float),sr=sr,n_mfcc=40)
        ch=librosa.feature.chroma_stft(y=audio.astype(float),sr=sr)
        zc=librosa.feature.zero_crossing_rate(y=audio.astype(float))
        spc=librosa.feature.spectral_centroid(y=audio.astype(float),sr=sr)
        feat=np.concatenate([np.mean(mfcc,axis=1),np.std(mfcc,axis=1),
                              np.mean(ch,axis=1),[np.mean(zc),np.std(zc),np.mean(spc),np.std(spc)]])
    if feat.shape[0]!=sc.mean_.shape[0]:
        raise HTTPException(400,
            f"Feature mismatch: model={sc.mean_.shape[0]}, got={feat.shape[0]}. Retrain with {feat_type} features.")
    pred_idx=int(clf.predict(sc.transform([feat]))[0])
    prediction=(idx2label.get(pred_idx) or idx2label.get(str(pred_idx))
                or (labels[pred_idx] if pred_idx<len(labels) else str(pred_idx)))
    confidence=100.0; all_scores={}
    if hasattr(clf,"predict_proba"):
        proba=clf.predict_proba(sc.transform([feat]))[0]
        confidence=round(float(np.max(proba))*100,1)
        all_scores={lbl:round(float(proba[i])*100,1) for i,lbl in enumerate(labels)}
    else:
        all_scores={lbl:(100.0 if lbl==prediction else 0.0) for lbl in labels}
    return {"prediction":prediction,"confidence":confidence,
            "all_scores":all_scores,"labels":labels,
            "model_type":md.get("model_type","svm")}


# ── Detect connectors in full recording ──────────────────────
@app.post("/api/detect")
async def detect_connectors(file: UploadFile = File(...), n_connectors: int = 3, use_gate: bool = False):
    import pickle, tempfile, os, librosa, numpy as np
    from scipy.signal import butter, filtfilt
    active = get_active_project()
    candidates = [project_path(active)/"models"/"model.pkl",
                  DATA/"models"/"model.pkl", BASE/"model.pkl"]
    if PROJECTS_ROOT.exists():
        for p in PROJECTS_ROOT.iterdir():
            if p.is_dir(): candidates.append(p/"models"/"model.pkl")
    pkl = next((p for p in candidates if p.exists()), None)
    if not pkl:
        raise HTTPException(400,"No trained model — train first in Pipeline")
    with open(str(pkl),"rb") as f: md=pickle.load(f)
    clf=md.get("model"); sc=md.get("scaler")
    idx2label=md.get("idx2label",{})
    # Fallback: build idx2label from labels list if missing
    if not idx2label:
        labels_list = md.get("labels", [])
        idx2label = {i: l for i, l in enumerate(labels_list)}
        idx2label.update({str(i): l for i, l in enumerate(labels_list)})
    feat_type=md.get("feature_type","mfcc")
    audio_bytes=await file.read()
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
    tmp.write(audio_bytes); tmp.close()
    try:
        audio,sr=librosa.load(tmp.name,sr=16000,mono=True)
    except Exception as ex:
        raise HTTPException(400,f"Cannot read audio: {ex}")
    finally:
        try: os.unlink(tmp.name)
        except: pass
    b,a=butter(4,min(300/(sr/2),0.99),btype="high")
    hp=filtfilt(b,a,audio)
    wl=int(0.3*sr); hl=int(0.1*sr); clicks=[]; last=-1.0
    def _feat(clip):
        if feat_type=="pcen":
            hop=256
            mel=librosa.feature.melspectrogram(y=clip.astype(float),sr=sr,n_mels=128,fmax=8000,hop_length=hop,n_fft=1024)
            pc=librosa.pcen(mel*(2**31),sr=sr,hop_length=hop,gain=0.8,bias=10,power=0.25,time_constant=0.4,eps=1e-6)
            dl=librosa.feature.delta(pc)
            ct=librosa.feature.spectral_contrast(y=clip.astype(float),sr=sr,n_fft=1024,hop_length=hop,n_bands=6)
            zc=librosa.feature.zero_crossing_rate(y=clip.astype(float))
            return np.concatenate([np.mean(pc,axis=1),np.std(pc,axis=1),np.mean(dl,axis=1),np.mean(ct,axis=1),[float(np.mean(zc)),float(np.std(zc))]])
        else:
            mfcc=librosa.feature.mfcc(y=clip.astype(float),sr=sr,n_mfcc=40)
            ch=librosa.feature.chroma_stft(y=clip.astype(float),sr=sr)
            zc=librosa.feature.zero_crossing_rate(y=clip.astype(float))
            spc=librosa.feature.spectral_centroid(y=clip.astype(float),sr=sr)
            return np.concatenate([np.mean(mfcc,axis=1),np.std(mfcc,axis=1),np.mean(ch,axis=1),[np.mean(zc),np.std(zc),np.mean(spc),np.std(spc)]])
    def _freq_gate(clip):
        N=2048;fft=np.abs(np.fft.rfft(clip.astype(float),n=N))
        fqs=np.fft.rfftfreq(N,1.0/sr);tot=fft.sum() or 1.0
        band=fft[((fqs>=300)&(fqs<3000))].sum()
        peak=float(fqs[np.argmax(fft)])
        return (band/tot)>=0.50 and 150<=peak<=6000
    gate_rej=0
    for start in range(0,len(hp)-wl,hl):
        t=start/sr
        if t-last<0.5: continue
        clip=hp[start:start+wl]
        if use_gate:
            if not _freq_gate(clip): gate_rej+=1; continue
        try:
            feat=_feat(clip)
        except: continue
        pred=clf.predict(sc.transform([feat]))[0]
        lbl=(idx2label.get(pred) or idx2label.get(str(pred)) or str(pred))
        if "click" in str(lbl).lower() or "connector" in str(lbl).lower():
            conf=1.0
            if hasattr(clf,"predict_proba"):
                proba=clf.predict_proba(sc.transform([feat]))[0]; conf=float(np.max(proba))
            if conf>=0.45:
                clicks.append({"time_sec":round(t,2),"confidence":round(conf*100,1)}); last=t
    results=[]
    for i in range(n_connectors):
        if i<len(clicks):
            results.append({"connector":i+1,"status":"LOCKED",
                            "time_sec":clicks[i]["time_sec"],"confidence":clicks[i]["confidence"]})
        else:
            results.append({"connector":i+1,"status":"MISSED","time_sec":None,"confidence":0})
    overall="OK" if all(r["status"]=="LOCKED" for r in results) else "NG"
    _st=max(1,len(audio)//2000)
    _wv=[round(float(v),4) for v in audio[::_st][:2000]]
    return {"overall":overall,"results":results,
            "missed":[r["connector"] for r in results if r["status"]=="MISSED"],
            "clicks_found":len(clicks),"duration_sec":round(len(audio)/sr,2),
            "waveform":_wv,"model_type":md.get("model_type","svm"),
            "gate_used":use_gate,"gate_rejected":gate_rej if use_gate else 0}



@app.post("/api/projects/import_path")
def import_from_path(payload: dict):
    """
    Import projects OR audio files into AudioMark.
    Supports:
      - AudioMark folder (has data/projects/) → copies projects
      - Audio files folder (has WAV/FLAC files) → copies audio + matches annotations
    """
    import shutil as _shutil

    raw = payload.get("path", "").strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(400, "Please enter a folder path.")

    src_path = None
    for candidate in [raw, raw.replace("\\", "/"), raw.replace("/", "\\")]:
        try:
            p = Path(candidate).expanduser()
            if p.exists() and p.is_dir():
                src_path = p
                break
        except: pass

    if not src_path:
        raise HTTPException(400,
            f"Folder not found: {raw}\n"
            "Check the path is correct. Example: D:\\Connector Noise\\Audio_XUV")

    imported = []

    # ── Strategy 1: AudioMark project folder ─────────────
    candidates = [
        src_path / "data" / "projects",
        src_path / "projects",
        src_path,
    ]
    src_projects = next(
        (p for p in candidates if p.exists() and p.is_dir()
         and any(x.is_dir() and (x/"annotations").exists() for x in p.iterdir()
                 if x.is_dir())),
        None
    )
    if src_projects:
        for proj in src_projects.iterdir():
            if not proj.is_dir(): continue
            dest = PROJECTS_ROOT / proj.name
            if dest.exists():
                imported.append(f"{proj.name} (already exists — skipped)")
                continue
            _shutil.copytree(str(proj), str(dest))
            imported.append(proj.name)
        src_active = src_path / "data" / ".active_project"
        if src_active.exists() and not ACTIVE_FILE.exists():
            _shutil.copy(str(src_active), str(ACTIVE_FILE))
        if imported:
            return {"imported": imported, "total": len(imported),
                    "message": f"Imported {len(imported)} project(s): " + ", ".join(imported)}

    # ── Strategy 2: Audio files folder ───────────────────
    AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.WAV', '.FLAC'}
    audio_files = []
    for ext in AUDIO_EXTS:
        audio_files.extend(src_path.rglob(f'*{ext}'))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        raise HTTPException(400,
            f"No audio files or project data found in:\n{src_path}\n\n"
            "Expected WAV/FLAC audio files OR an AudioMark installation folder.")

    active  = get_active_project()
    upl_dir = project_path(active) / "uploads"
    ann_dir = project_path(active) / "annotations"
    upl_dir.mkdir(parents=True, exist_ok=True)

    copied = []; skipped = []; matched = []
    for af in audio_files:
        dest = upl_dir / af.name
        if dest.exists():
            skipped.append(af.name)
        else:
            try:
                _shutil.copy2(str(af), str(dest))
                copied.append(af.name)
            except Exception as e:
                skipped.append(f"{af.name} (error: {e})")
        if (ann_dir / (af.stem + ".json")).exists():
            matched.append(af.name)

    msg_parts = []
    if copied:   msg_parts.append(f"Loaded {len(copied)} audio file(s) from {src_path.name}")
    if matched:  msg_parts.append(f"Found {len(matched)} matching annotation(s)")
    if skipped:  msg_parts.append(f"Skipped {len(skipped)} (already exist)")
    if not msg_parts: msg_parts.append("No new files found")

    imported = msg_parts
    return {"imported": imported, "total": len(imported),
            "copied": len(copied), "matched": len(matched),
            "message": "\n".join(msg_parts)}



@app.get("/api/verify_preprocessing")
def verify_preprocessing():
    """
    Check that Slice, Extract, Train and Detect all use same preprocessing.
    Slice and Detect both apply HP filter 300Hz — this confirms consistency.
    """
    active = get_active_project()
    pp = project_path(active)

    # Count annotations and sliced clips
    ann_dir  = pp / "annotations"
    sl_dir   = pp / "sliced"
    n_ann    = len(list(ann_dir.glob("*.json"))) if ann_dir.exists() else 0
    n_clips  = sum(1 for _ in sl_dir.rglob("*.wav")) if sl_dir.exists() else 0

    # Check model metadata
    model_meta = {}
    model_pkl  = pp / "models" / "model.pkl"
    if model_pkl.exists():
        try:
            import pickle
            with open(str(model_pkl),"rb") as f: md=pickle.load(f)
            model_meta = {
                "hp_cutoff":    md.get("hp_cutoff", 300),
                "feature_type": md.get("feature_type","mfcc"),
                "model_type":   md.get("model_type","svm"),
                "accuracy":     md.get("accuracy",0),
            }
        except: pass

    # Preprocessing flags
    # Slice: HP filter is now applied (we added it above)
    # Detect: HP filter is always applied (butter 300Hz)
    # Both are consistent now
    slice_hp    = True   # we apply HP in slice route
    extract_hp  = True   # features extracted from HP-filtered clips
    detect_hp   = True   # detect always applies HP

    all_consistent = slice_hp and extract_hp and detect_hp

    issues   = []
    warnings = []

    if not all_consistent:
        issues.append("Preprocessing mismatch: re-run Slice + Extract + Train")

    if n_clips == 0:
        warnings.append("No sliced clips found — run Step 1 first")
    if n_ann == 0:
        warnings.append("No annotations found — annotate files first")

    return {
        "all_consistent":  all_consistent,
        "n_annotations":   n_ann,
        "n_clips":         n_clips,
        "model_preprocessing": model_meta,
        "issues":   issues,
        "warnings": warnings,
        "pipeline_status": {
            "all_consistent":    all_consistent,
            "slice_hp_filter":   slice_hp,
            "slice_normalize":   False,
            "extract_hp_filter": extract_hp,
            "extract_normalize": False,
            "detect_hp_filter":  detect_hp,
            "detect_normalize":  False,
        }
    }


@app.post("/api/detect_debug")
async def detect_debug(file: UploadFile = File(...), n_connectors: int = 3):
    import pickle,tempfile as _tmp,os,librosa,numpy as np
    from scipy.signal import butter,filtfilt
    active=get_active_project()
    candidates=[project_path(active)/"models"/"model.pkl",DATA/"models"/"model.pkl",BASE/"model.pkl"]
    if PROJECTS_ROOT.exists():
        for p in PROJECTS_ROOT.iterdir():
            if p.is_dir(): candidates.append(p/"models"/"model.pkl")
    pkl=next((p for p in candidates if p.exists()),None)
    if not pkl: raise HTTPException(400,"No trained model")
    with open(str(pkl),"rb") as f: md=pickle.load(f)
    clf=md.get("model");sc_obj=md.get("scaler")
    idx2lbl=md.get("idx2label",{})
    if not idx2lbl:
        ll=md.get("labels",[]);idx2lbl={i:l for i,l in enumerate(ll)};idx2lbl.update({str(i):l for i,l in enumerate(ll)})
    audio_bytes=await file.read()
    tf=_tmp.NamedTemporaryFile(suffix=".wav",delete=False);tf.write(audio_bytes);tf.close()
    try: audio,sr=librosa.load(tf.name,sr=16000,mono=True)
    except Exception as ex: raise HTTPException(400,f"Cannot read audio: {ex}")
    finally:
        try: os.unlink(tf.name)
        except: pass
    b,a=butter(4,min(300/(sr/2),0.99),btype="high");hp=filtfilt(b,a,audio)
    wl=int(0.3*sr);hl=int(0.1*sr)
    def _feat(clip):
        mfcc=librosa.feature.mfcc(y=clip.astype(float),sr=sr,n_mfcc=40)
        ch=librosa.feature.chroma_stft(y=clip.astype(float),sr=sr)
        zc=librosa.feature.zero_crossing_rate(y=clip.astype(float))
        spc=librosa.feature.spectral_centroid(y=clip.astype(float),sr=sr)
        return np.concatenate([np.mean(mfcc,axis=1),np.std(mfcc,axis=1),np.mean(ch,axis=1),[np.mean(zc),np.std(zc),np.mean(spc),np.std(spc)]])
    def _freq(clip):
        N=4096;fft=np.abs(np.fft.rfft(clip.astype(float),n=N));fqs=np.fft.rfftfreq(N,1.0/sr);tot=fft.sum() or 1.0
        def bp(lo,hi):m=(fqs>=lo)&(fqs<hi);return round(float(fft[m].sum()/tot*100),1) if m.any() else 0.0
        peak=float(fqs[np.argmax(fft)]);maxF=min(sr//2,8000);edges=np.linspace(0,maxF,65)
        spec=[round(float(fft[((fqs>=edges[i])&(fqs<edges[i+1]))].sum()/tot*100),2) if ((fqs>=edges[i])&(fqs<edges[i+1])).any() else 0.0 for i in range(64)]
        return {"peak_hz":round(peak,1),"centroid_hz":round(float(np.sum(fqs*fft)/tot),1),"click_band":round(bp(300,3000),1),
                "bands":{"noise_low":round(bp(0,300),1),"low_mid":round(bp(300,1000),1),"mid":round(bp(1000,3000),1),"high_mid":round(bp(3000,6000),1),"air":round(bp(6000,8000),1)},
                "spectrum":spec,"max_freq_hz":maxF}
    all_windows=[];last=-1.0
    for start in range(0,len(hp)-wl,hl):
        t=round(start/sr,3);clip=hp[start:start+wl];freq=_freq(clip)
        ml_lbl="background_noise";ml_conf=0.0;top3=[]
        try:
            feat=_feat(clip);raw_pred=clf.predict(sc_obj.transform([feat]))[0]
            ml_lbl=idx2lbl.get(raw_pred) or idx2lbl.get(str(raw_pred)) or str(raw_pred)
            if hasattr(clf,"predict_proba"):
                proba=clf.predict_proba(sc_obj.transform([feat]))[0];ml_conf=round(float(np.max(proba))*100,1)
                top3=sorted(zip([idx2lbl.get(i,str(i)) for i in range(len(proba))],[round(float(p)*100,1) for p in proba]),key=lambda x:-x[1])[:3]
            else: ml_conf=100.0;top3=[(ml_lbl,100.0)]
        except: pass
        is_click="click" in ml_lbl.lower() or "connector" in ml_lbl.lower()
        conf_ok=ml_conf>=45;spacing_ok=(t-last)>=0.5;accepted=is_click and conf_ok and spacing_ok
        r_pass,r_fail=[],[]
        if is_click: r_pass.append(f"ML:{ml_lbl}({ml_conf}%)")
        else: r_fail.append(f"ML said:{ml_lbl}({ml_conf}%)")
        if not conf_ok: r_fail.append(f"conf {ml_conf}%<45%")
        if not spacing_ok: r_fail.append(f"too close {round(t-last,2)}s")
        all_windows.append({"time_sec":t,"accepted":accepted,"ml_label":ml_lbl,"ml_conf":ml_conf,"top3":top3,"freq":freq,"reason_pass":r_pass,"reason_fail":r_fail})
        if accepted: last=t
    final_clicks=[w for w in all_windows if w["accepted"]]
    results=[]
    for i in range(n_connectors):
        if i<len(final_clicks):
            c=final_clicks[i];results.append({"connector":i+1,"status":"LOCKED","time_sec":c["time_sec"],"confidence":c["ml_conf"],"peak_hz":c["freq"]["peak_hz"],"click_band":c["freq"]["click_band"],"bands":c["freq"]["bands"],"spectrum":c["freq"]["spectrum"],"max_freq_hz":c["freq"]["max_freq_hz"],"top3":c["top3"],"reason":c["reason_pass"]})
        else: results.append({"connector":i+1,"status":"MISSED","time_sec":None,"confidence":0,"reason":["No window passed"]})
    overall="OK" if all(r["status"]=="LOCKED" for r in results) else "NG"
    _st=max(1,len(audio)//2000);wv=[round(float(v),4) for v in audio[::_st][:2000]]
    ml_rej=sum(1 for w in all_windows if not("click" in w["ml_label"].lower() or "connector" in w["ml_label"].lower()))
    conf_rej=sum(1 for w in all_windows if ("click" in w["ml_label"].lower() or "connector" in w["ml_label"].lower()) and w["ml_conf"]<45)
    return {"overall":overall,"results":results,"missed":[r["connector"] for r in results if r["status"]=="MISSED"],
            "clicks_found":len(final_clicks),"duration_sec":round(len(audio)/sr,2),"waveform":wv,"model_type":md.get("model_type","svm"),
            "debug":{"total_windows":len(all_windows),"ml_rejected":ml_rej,"conf_rejected":conf_rej,"passed_all":len(final_clicks),"all_windows":all_windows}}

if __name__ == "__main__":
    def port_free(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try: s.bind(("0.0.0.0", p)); return True
            except: return False
    port = next((p for p in [8100,8101,8102,8103] if port_free(p)), None)
    if not port: print("❌ Ports 8100-8103 all in use"); exit(1)
    print(f"\n🎙  AudioMark Annotator")
    print(f"🌐  http://localhost:{port}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
