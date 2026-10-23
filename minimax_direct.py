"""
MiniMax direct API for H3 video (cloud, billed per output second) + small ffmpeg helpers.

Official endpoints (https://platform.minimax.io/docs/guides/video-generation):
  POST /v2/video_generation              -> {"task_id": ...}   text/image/reference to video
  POST /v2/video_regeneration            -> {"task_id": ...}   a 768P H3 clip -> 2K
  GET  /v2/query/video_generation/{id}   -> {"task": {status, content.url, usage, error}}

The API has no seed parameter and no lip-sync (driving speech) input: reference_audio only
sets a voice timbre. Exact speech + lip-sync stays on the local GPU (MiniMaxH3AddGuide).
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
from pathlib import Path
from typing import Optional

import httpx

# MiniMax pay-as-you-go list prices in USD per output second (vendor pricing page, Oct 2026).
# Only used for estimates shown before spending money; MiniMax bills the real amount.
PRICE_PER_SECOND = {
    ("MiniMax-H3", "768P"): 0.08,
    ("MiniMax-H3", "2K"): 0.13,
    ("MiniMax-H3-Max", "480P"): 0.05,
    ("MiniMax-H3-Max", "768P"): 0.08,
}
REGEN_PRICE_PER_SECOND = 0.05      # 768P -> 2K regeneration, per second of output
# Input materials: (free images, price per extra image) and reference-video price per second.
# Regeneration bills the original task's input materials again at its own rates.
IMAGE_PRICING = {"MiniMax-H3": (5, 0.04), "MiniMax-H3-Max": (2, 0.074), "regeneration": (5, 0.025)}
REF_VIDEO_PER_SECOND = {
    ("MiniMax-H3", "768P"): 0.08,
    ("MiniMax-H3", "2K"): 0.13,
    ("MiniMax-H3-Max", "480P"): 0.0553,
    ("MiniMax-H3-Max", "768P"): 0.143,
    ("regeneration", "2K"): 0.05,
}
MODEL_RESOLUTIONS = {"MiniMax-H3": ("768P", "2K"), "MiniMax-H3-Max": ("480P", "768P")}
MODEL_DURATION = {"MiniMax-H3": (4, 15), "MiniMax-H3-Max": (5, 15)}

MAX_BODY_BYTES = 60 * 1024 * 1024  # API limit is 64 MB per request
INLINE_VIDEO_BYTES = 20 * 1024 * 1024
REF_AUDIO_MIN, REF_AUDIO_TOTAL = 2.0, 15.0
REF_VIDEO_MIN, REF_VIDEO_TOTAL = 2.0, 15.0

# Regeneration only accepts clips that match H3's own 768P output.
REGEN_FPS = 24
REGEN_MIN_AREA, REGEN_MAX_AREA = 768 * 768, 768 * 1344
REGEN_MIN_FRAMES, REGEN_MAX_FRAMES, REGEN_FRAME_STEP = 107, 362, 17

MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
        "mp3": "audio/mpeg", "wav": "audio/wav", "mp4": "video/mp4", "mov": "video/quicktime"}


class APIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Validation and cost estimates
# ---------------------------------------------------------------------------
def check_model(model: str, resolution: str, duration: int) -> Optional[str]:
    if model not in MODEL_RESOLUTIONS:
        return f"model must be one of {', '.join(MODEL_RESOLUTIONS)}"
    if resolution not in MODEL_RESOLUTIONS[model]:
        return f"{model} supports resolution {' / '.join(MODEL_RESOLUTIONS[model])}"
    lo, hi = MODEL_DURATION[model]
    if not lo <= duration <= hi:
        return f"{model} duration must be a whole number of seconds from {lo} to {hi}"
    return None


def _inputs(parts: dict, key: str, resolution: str, image_count: int, ref_video_seconds: float) -> None:
    free, each = IMAGE_PRICING[key]
    extra = max(0, image_count - free)
    if extra:
        parts[f"{extra} images beyond the first {free} x ${each}"] = extra * each
    if ref_video_seconds:
        rate = REF_VIDEO_PER_SECOND[(key, resolution)]
        parts[f"reference video {ref_video_seconds:.1f} s x ${rate}/s"] = rate * ref_video_seconds


def _result(parts: dict) -> dict:
    return {"estimated_cost_usd": round(sum(parts.values()), 3),
            "breakdown": {k: round(v, 3) for k, v in parts.items()}}


def estimate_generation(model: str, resolution: str, duration: int, image_count: int = 0,
                        reference_video_seconds: float = 0.0) -> dict:
    rate = PRICE_PER_SECOND[(model, resolution)]
    parts = {f"video {duration} s x ${rate}/s ({model} {resolution})": rate * duration}
    _inputs(parts, model, resolution, image_count, reference_video_seconds)
    return _result(parts)


def estimate_regeneration(seconds: float, image_count: int = 0, reference_video_seconds: float = 0.0) -> dict:
    parts = {f"2K regeneration {seconds:.2f} s x ${REGEN_PRICE_PER_SECOND}/s": REGEN_PRICE_PER_SECOND * seconds}
    _inputs(parts, "regeneration", "2K", image_count, reference_video_seconds)
    return _result(parts)


def cost_from_usage(kind: str, model: Optional[str], resolution: Optional[str], usage: dict) -> Optional[float]:
    """List-price cost of a finished task from the usage MiniMax reports (approximate)."""
    secs = usage.get("output_seconds") or usage.get("total_seconds")
    if not secs:
        return None
    imgs, vid = usage.get("input_image_count") or 0, usage.get("input_seconds") or 0
    if kind == "regeneration":
        return estimate_regeneration(secs, imgs, vid)["estimated_cost_usd"]
    if (model, resolution) not in PRICE_PER_SECOND:
        return None
    return estimate_generation(model, resolution, secs, imgs, vid)["estimated_cost_usd"]


def regen_spec_problems(info: dict) -> list[str]:
    """Why a clip cannot go to /v2/video_regeneration (empty list = OK)."""
    p = []
    if not info.get("has_audio"):
        p.append("the clip has no audio track")
    fps = info.get("fps")
    if not fps or abs(fps - REGEN_FPS) > 0.01:
        p.append(f"frame rate is {fps}, must be {REGEN_FPS} fps")
    w, h = info.get("width") or 0, info.get("height") or 0
    if w % 32 or h % 32:
        p.append(f"size {w}x{h}: width and height must be divisible by 32")
    if not REGEN_MIN_AREA <= w * h <= REGEN_MAX_AREA:
        p.append(f"size {w}x{h} is outside the 768P range (768x768 .. 768x1344 pixels); "
                 "render with quality 'standard' or 'high', not 'draft'")
    f = info.get("frames")
    if f is None or not REGEN_MIN_FRAMES <= f <= REGEN_MAX_FRAMES or (f - REGEN_MIN_FRAMES) % REGEN_FRAME_STEP:
        p.append(f"{f} frames: must be {REGEN_MIN_FRAMES}-{REGEN_MAX_FRAMES} in steps of {REGEN_FRAME_STEP} "
                 "(duration 4-15 s as rendered by this server)")
    return p


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------
def data_uri(data: bytes, ext: str) -> str:
    return f"data:{MIME.get(ext.lower(), 'application/octet-stream')};base64,{base64.b64encode(data).decode()}"


def media_item(kind: str, url: str, role: Optional[str] = None) -> dict:
    """kind: image | video | audio."""
    item = {"type": f"{kind}_url", f"{kind}_url": {"url": url}}
    if role:
        item["role"] = role
    return item


def text_item(text: str) -> dict:
    return {"type": "text", "text": text}


def body_size(body: dict) -> int:
    return len(json.dumps(body))


def redact(body: dict) -> dict:
    """Copy of a request body with inline base64 replaced by a short marker (for logs/records)."""
    def fix(v):
        if isinstance(v, str) and v.startswith("data:") and len(v) > 200:
            return v[:40] + f"...<{len(v)} chars>"
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    return fix(body)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
class VideoAPI:
    def __init__(self, api_key: str, base: str = "https://api.minimax.io",
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.api_key = api_key
        self.base = base.rstrip("/")
        self.transport = transport

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.transport, follow_redirects=True)

    async def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        async with self._client(180) as c:
            r = await c.request(method, f"{self.base}{path}", json=body,
                                headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            j = r.json()
        except Exception:
            raise APIError(f"MiniMax HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400 or j.get("type") == "error":
            err = j.get("error") if isinstance(j.get("error"), dict) else {}
            raise APIError(f"MiniMax HTTP {r.status_code}: {err.get('message') or json.dumps(j)[:300]}")
        base = j.get("base_resp") or {}
        if base.get("status_code", 0) != 0:
            raise APIError(f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}")
        return j

    async def create(self, body: dict) -> str:
        j = await self._call("POST", "/v2/video_generation", body)
        if not j.get("task_id"):
            raise APIError(f"no task_id in response: {json.dumps(j)[:300]}")
        return str(j["task_id"])

    async def regenerate(self, body: dict) -> str:
        j = await self._call("POST", "/v2/video_regeneration", body)
        if not j.get("task_id"):
            raise APIError(f"no task_id in response: {json.dumps(j)[:300]}")
        return str(j["task_id"])

    async def query(self, task_id: str) -> dict:
        j = await self._call("GET", f"/v2/query/video_generation/{task_id}")
        return j.get("task") or {}

    async def download(self, url: str, dest: Path) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        async with self._client(600) as c:
            async with c.stream("GET", url) as r:
                if r.status_code != 200:
                    raise APIError(f"download failed: HTTP {r.status_code}")
                with open(tmp, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
                        n += len(chunk)
        tmp.replace(dest)
        return n


# ---------------------------------------------------------------------------
# ffmpeg helpers (imageio-ffmpeg's bundled binary, else the system ffmpeg)
# ---------------------------------------------------------------------------
def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


async def run_ffmpeg(*args: str, timeout: float = 180, want_stdout: bool = False):
    """Returns (returncode, stderr) or, with want_stdout, (returncode, stderr, stdout)."""
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg is not available (install with: venv/bin/pip install imageio-ffmpeg)")
    proc = await asyncio.create_subprocess_exec(
        exe, "-hide_banner", "-nostdin", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("ffmpeg timed out")
    if want_stdout:
        return proc.returncode, err.decode(errors="replace"), out.decode(errors="replace")
    return proc.returncode, err.decode(errors="replace")


async def probe(path: Path) -> dict:
    """width, height, fps, duration, has_audio, frames (one framecrc line per video packet)."""
    rc, err, out = await run_ffmpeg("-i", str(path), "-map", "0:v:0?", "-c", "copy",
                                    "-f", "framecrc", "-", want_stdout=True)
    info: dict = {"has_audio": bool(re.search(r"Stream #\S+.*?: Audio:", err))}
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        info["duration"] = round(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), 3)
    m = re.search(r"Stream #\S+.*?: Video: .*?\b(\d{2,5})x(\d{2,5})\b", err)
    if m:
        info["width"], info["height"] = int(m[1]), int(m[2])
        fm = re.search(r"(\d+(?:\.\d+)?) fps", err[m.start():])
        if fm:
            info["fps"] = float(fm[1])
        if rc == 0:
            info["frames"] = sum(1 for line in out.splitlines() if line and not line.startswith("#"))
    if rc != 0 and "duration" not in info:
        raise RuntimeError(f"could not read media file: {err.strip().splitlines()[-1] if err.strip() else rc}")
    return info


async def extract_frame(video: Path, dest: Path, position: str = "last", time_seconds: float = 0.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if position == "last":
        args = ["-y", "-sseof", "-0.5", "-i", str(video), "-update", "1", str(dest)]
    else:
        t = 0.0 if position == "first" else max(0.0, float(time_seconds))
        args = ["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(dest)]
    rc, err = await run_ffmpeg(*args)
    if rc != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"frame extraction failed: {err.strip().splitlines()[-1] if err.strip() else rc}")


async def to_wav(src: Path, dest: Path, max_seconds: Optional[float] = None) -> None:
    args = ["-y", "-i", str(src), "-vn"]
    if max_seconds:
        args += ["-t", f"{max_seconds:.3f}"]
    args += ["-c:a", "pcm_s16le", str(dest)]
    rc, err = await run_ffmpeg(*args)
    if rc != 0 or not dest.is_file():
        raise RuntimeError(f"audio conversion failed: {err.strip().splitlines()[-1] if err.strip() else rc}")


async def replace_audio(video: Path, audio_from: Path, dest: Path) -> None:
    """Video stream of `video` + audio stream of `audio_from` (no re-encode)."""
    rc, err = await run_ffmpeg("-y", "-i", str(video), "-i", str(audio_from), "-map", "0:v:0",
                               "-map", "1:a:0", "-c", "copy", "-shortest", str(dest))
    if rc != 0 or not dest.is_file():
        raise RuntimeError(f"audio remux failed: {err.strip().splitlines()[-1] if err.strip() else rc}")
