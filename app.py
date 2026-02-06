"""
Simple local web app to download YouTube videos as MP4 with custom start/end timestamps.
Requires: ffmpeg installed on your system (for merging/trimming).

Temp files: All downloads go to the system temp dir (see DOWNLOAD_DIR below).
They are deleted automatically after each request, so disk use is minimal.
When you set start/end times, only that segment is downloaded (no full-file download).
"""
import os
import re
import uuid
import threading
import time

# Use certifi's CA bundle so SSL works on macOS (avoids CERTIFICATE_VERIFY_FAILED)
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
import json
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Callable

from flask import Flask, render_template, request, send_file, jsonify, Response

app = Flask(__name__)
# Temp dir for this app; cleared after each download. Typically system temp (e.g. /var/folders/.../T on macOS).
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "yt-downloader"

# In-memory job progress; keyed by job_id. Values: status, progress_pct, downloaded_bytes, total_bytes, eta_sec, speed_bytes_per_sec, phase, error
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def parse_timestamp(ts: str) -> float:
    """Parse '1:30', '1:30:00', or '90' into seconds."""
    ts = (ts or "0").strip()
    if not ts:
        return 0.0
    # Already seconds
    if re.match(r"^\d+\.?\d*$", ts):
        return float(ts)
    parts = [float(p) for p in ts.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


# Download phase is 0-70%, convert phase is 70-100%
DOWNLOAD_PCT_MAX = 70


def get_video_info(url: str) -> dict | None:
    """Extract video duration and estimated size without downloading. Returns {'duration': sec, 'estimated_size': bytes} or None."""
    import yt_dlp
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return None
        duration = info.get("duration")
        if duration is None or duration <= 0:
            return None
        size = info.get("filesize") or info.get("filesize_approx")
        if size is None and info.get("formats"):
            for f in info["formats"]:
                fs = f.get("filesize") or f.get("filesize_approx")
                if fs and (size is None or fs > (size or 0)):
                    size = fs
            if size is None:
                for f in reversed(info["formats"]):
                    fs = f.get("filesize") or f.get("filesize_approx")
                    if fs:
                        size = fs
                        break
        if size is None or size <= 0:
            size = int(duration * 2.5 * 1024 * 1024 / 60)
        return {"duration": float(duration), "estimated_size": int(size)}
    except Exception:
        return None


def download_youtube_mp4(
    url: str,
    out_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    progress_callback: Callable[..., None] | None = None,
    estimated_clip_bytes: int | None = None,
) -> bool:
    """Download YouTube video as MKV using yt-dlp (convert to MP4 is done separately with progress).
    If estimated_clip_bytes is set (e.g. from get_video_info + clip ratio), progress uses it when yt-dlp doesn't report total."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    def progress_hook(d):
        if not progress_callback:
            return
        status = d.get("status", "")
        if status == "finished":
            progress_callback({"phase": "downloading", "progress_pct": DOWNLOAD_PCT_MAX})
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        down = d.get("downloaded_bytes") or 0
        elapsed = d.get("elapsed") or 0
        if total and total > 0:
            pct = min(DOWNLOAD_PCT_MAX, down / total * DOWNLOAD_PCT_MAX)
        elif estimated_clip_bytes and estimated_clip_bytes > 0 and down and down > 0:
            pct = min(DOWNLOAD_PCT_MAX, down / estimated_clip_bytes * DOWNLOAD_PCT_MAX)
        elif down and down > 0:
            pct = min(DOWNLOAD_PCT_MAX, (down / (150 * 1024 * 1024)) * DOWNLOAD_PCT_MAX)
        else:
            pct = min(50.0, elapsed * 2.0) if elapsed else 0
        progress_callback({
            "status": status,
            "downloaded_bytes": down,
            "total_bytes": total,
            "elapsed": elapsed,
            "eta": d.get("eta"),
            "speed": d.get("speed"),
            "progress_pct": round(pct, 1),
            "phase": "downloading",
        })

    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        "merge_output_format": "mkv",
        "outtmpl": str(out_path.with_suffix("")),
        "quiet": True,
        "noprogress": False,
        "progress_hooks": [progress_hook] if progress_callback else [],
    }
    if start_sec is not None and end_sec is not None:
        opts["download_ranges"] = download_range_func(None, [(start_sec, end_sec)])
        opts["force_keyframes_at_cuts"] = True
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return out_path.with_suffix(".mkv").exists() or out_path.exists()
    except Exception:
        return False


def _ffmpeg_convert_with_progress(
    input_path: Path,
    output_path: Path,
    progress_callback: Callable[..., None],
) -> bool:
    """Convert MKV to H.264/AAC MP4 with ffmpeg, reporting progress 70-100% via progress_callback."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_sec = float(result.stdout.strip())
    except Exception:
        duration_sec = 1.0
    if duration_sec <= 0:
        duration_sec = 1.0

    # Parse progress: -progress pipe:2 gives out_time_ms= (microseconds); fallback to time=HH:MM:SS.xx in stderr
    time_re_us = re.compile(r"out_time_ms=(\d+)")
    time_re_hr = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
            "-progress", "pipe:2",
            str(output_path),
        ],
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stderr is not None
    last_pct = float(DOWNLOAD_PCT_MAX)
    for line in proc.stderr:
        current_sec = None
        mo = time_re_us.search(line)
        if mo:
            current_sec = int(mo.group(1)) / 1_000_000.0
        else:
            mo2 = time_re_hr.search(line)
            if mo2:
                h, m, s, cs = int(mo2.group(1)), int(mo2.group(2)), int(mo2.group(3)), int(mo2.group(4))
                current_sec = h * 3600 + m * 60 + s + cs / 100.0
        if current_sec is not None:
            pct = DOWNLOAD_PCT_MAX + (100 - DOWNLOAD_PCT_MAX) * (current_sec / duration_sec)
            pct = min(100.0, max(DOWNLOAD_PCT_MAX, pct))
            if pct >= last_pct + 0.25 or pct >= 99.9:
                last_pct = pct
                progress_callback({
                    "phase": "converting",
                    "progress_pct": round(pct, 1),
                })
    proc.wait()
    if proc.returncode != 0:
        return False
    progress_callback({"phase": "converting", "progress_pct": 100})
    return output_path.exists()


def _run_download(job_id: str, url: str, start_sec: float, end_sec: float | None, use_range: bool, end_for_download: float):
    raw_path = DOWNLOAD_DIR / "raw" / job_id
    raw_path.mkdir(parents=True, exist_ok=True)
    video_path = raw_path / "video"
    final_path = raw_path / "clip.mp4"

    def on_progress(d: dict):
        with _jobs_lock:
            if job_id not in _jobs:
                return
            update = {"phase": d.get("phase", "downloading")}
            if "progress_pct" in d:
                update["progress_pct"] = d["progress_pct"]
            if "downloaded_bytes" in d:
                update["downloaded_bytes"] = d["downloaded_bytes"]
            if "total_bytes" in d:
                update["total_bytes"] = d["total_bytes"]
            if "eta" in d:
                update["eta_sec"] = d["eta"]
            if "speed" in d:
                update["speed_bytes_per_sec"] = d["speed"]
            _jobs[job_id].update(update)
            pct = update.get("progress_pct")
            phase = update.get("phase", "")
            if pct is not None:
                print(f"[yt-downloader] {phase} {pct:.1f}%")

    estimated_clip_bytes: int | None = None
    if use_range:
        info = get_video_info(url)
        if info:
            full_duration = info["duration"]
            full_size = info["estimated_size"]
            clip_end = end_for_download if end_for_download < 1e9 else full_duration
            clip_duration = max(0, min(clip_end, full_duration) - start_sec)
            if full_duration and full_duration > 0 and clip_duration > 0:
                estimated_clip_bytes = int(full_size * (clip_duration / full_duration))

    try:
        ok = download_youtube_mp4(
            url, video_path,
            start_sec=start_sec if use_range else None,
            end_sec=end_for_download if use_range else None,
            progress_callback=on_progress,
            estimated_clip_bytes=estimated_clip_bytes,
        )
        with _jobs_lock:
            if job_id not in _jobs:
                return
            if not ok:
                _jobs[job_id].update({"status": "error", "error": "Download failed"})
                return
        mkv_file = next(raw_path.glob("*.mkv"), None)
        if not mkv_file or not mkv_file.exists():
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id].update({"status": "error", "error": "No MKV produced"})
            return
        on_progress({"phase": "converting", "progress_pct": DOWNLOAD_PCT_MAX})
        if not _ffmpeg_convert_with_progress(mkv_file, final_path, on_progress):
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id].update({"status": "error", "error": "Convert failed"})
            return
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update({"status": "complete", "file_path": str(final_path), "progress_pct": 100})
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update({"status": "error", "error": str(e)})
    finally:
        # Clean up intermediate files but keep clip.mp4 until result is fetched
        for f in raw_path.glob("*"):
            if f.name != "clip.mp4" and f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download", methods=["POST"])
def download():
    if request.is_json:
        data = request.get_json() or {}
        url = (data.get("url") or "").strip()
        start_ts = (data.get("start") or "0").strip()
        end_ts = (data.get("end") or "").strip()
    else:
        url = (request.form.get("url") or "").strip()
        start_ts = (request.form.get("start") or "0").strip()
        end_ts = (request.form.get("end") or "").strip()

    if not url or "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Please enter a valid YouTube URL."}), 400

    start_sec = parse_timestamp(start_ts)
    end_sec = parse_timestamp(end_ts) if end_ts else None
    if end_sec is not None and end_sec <= start_sec:
        return jsonify({"error": "End time must be after start time."}), 400

    job_id = uuid.uuid4().hex
    use_range = start_sec > 0 or (end_sec is not None and end_sec < 1e9)
    end_for_download = end_sec if end_sec is not None else float("inf")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "progress_pct": None,
            "downloaded_bytes": None,
            "total_bytes": None,
            "eta_sec": None,
            "speed_bytes_per_sec": None,
            "phase": "starting",
            "error": None,
            "file_path": None,
            "started_at": time.time(),
        }

    thread = threading.Thread(
        target=_run_download,
        args=(job_id, url, start_sec, end_sec, use_range, end_for_download),
        daemon=True,
    )
    thread.start()

    def event_stream():
        while True:
            with _jobs_lock:
                j = _jobs.get(job_id)
            if not j:
                break
            st = j.get("status")
            if st == "complete":
                yield f"data: {json.dumps({'status': 'complete', 'job_id': job_id})}\n\n"
                break
            if st == "error":
                yield f"data: {json.dumps({'status': 'error', 'error': j.get('error') or 'Unknown error'})}\n\n"
                break
            pct = j.get("progress_pct")
            phase = j.get("phase") or "downloading"
            # When no progress from yt-dlp (common with clip/range downloads), show time-based 0→50% so the bar moves
            if pct is None and phase in ("starting", "downloading"):
                elapsed = time.time() - j.get("started_at", time.time())
                pct = min(50.0, elapsed * 2.0)
                phase = "downloading"
            payload = {
                "status": "progress",
                "progress_pct": pct,
                "eta_sec": j.get("eta_sec"),
                "speed": j.get("speed_bytes_per_sec"),
                "phase": phase,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.25)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/result/<job_id>")
def download_result(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j or j.get("status") != "complete":
            return jsonify({"error": "Not found or not ready"}), 404
        path = j.get("file_path")
        if not path:
            return jsonify({"error": "No file"}), 404
        file_path = Path(path)
        if not file_path.exists():
            return jsonify({"error": "File gone"}), 404
        # Remove job so we only allow one fetch
        del _jobs[job_id]

    def cleanup():
        try:
            raw_dir = file_path.parent
            for f in list(raw_dir.glob("*")) + [file_path]:
                if f.exists():
                    f.unlink()
            raw_dir.rmdir()
        except Exception:
            pass

    # Delete file after a delay so the response can be sent first
    threading.Timer(120, cleanup).start()
    return send_file(
        file_path,
        as_attachment=True,
        download_name="clip.mp4",
        mimetype="video/mp4",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
