"""
MiniMax Direct — a small local MCP server (stdio) for the Claude desktop app.
MiniMax H3 video straight from MiniMax's API on your own computer; no rented GPU needed.

Tools: generate_video, upscale_to_2k, get_job, list_jobs, extract_frame, prompt_guide
Inputs are files on this computer (full path, or a path relative to the output folder) or
http(s) URLs. Results are saved into MINIMAX_OUTPUT_DIR.

API key (never put it in this file): env MINIMAX_API_KEY, or the first line that does not start
with '#' in  %USERPROFILE%\\.minimax\\api_key.txt  (read on every call, no restart needed).
Env: MINIMAX_OUTPUT_DIR (results), MINIMAX_API_BASE (default https://api.minimax.io),
     MINIMAX_HOME (default %USERPROFILE%\\.minimax: key file, job records, temp files)
Needs: pip install "mcp>=1.12,<2" httpx imageio-ffmpeg     (+ minimax_direct.py next to this file)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

import minimax_direct as MD  # noqa: E402

HOME = Path(os.environ.get("MINIMAX_HOME") or Path.home() / ".minimax")
JOBS = HOME / "jobs"
TMP = HOME / "tmp"
OUT = Path(os.environ.get("MINIMAX_OUTPUT_DIR") or Path.home() / "Videos" / "MiniMax-Direct")
BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io")
MAX_WAIT = 50
MAX_FILE = 50 * 1024 * 1024
_downloads: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def api_key() -> str:
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        return key
    try:
        for line in (HOME / "api_key.txt").read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except Exception:
        pass
    return ""


def api() -> MD.VideoAPI:
    key = api_key()
    if not key:
        raise RuntimeError(f"no MiniMax API key yet: paste it as one line into {HOME / 'api_key.txt'}")
    return MD.VideoAPI(key, BASE)


def safe_name(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", s).strip(" ._")[:80]


def local_path(s: str) -> Optional[Path]:
    s = (s or "").strip().strip('"')
    if s.lower().startswith("file:///"):
        s = unquote(s[8:])
    if not s or s.startswith(("http://", "https://", "data:")) or len(s) > 1000:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(s)))
    if not p.is_absolute():
        p = OUT / p
    return p if p.is_file() else None


def ext_of(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[4:8] == b"ftyp":
        return "mov" if data[8:10] == b"qt" else "mp4"
    return "png"


async def load(src: str) -> tuple[bytes, Optional[Path]]:
    """(bytes, local path or None) of a file path, http(s) URL, data URI or base64."""
    p = local_path(src)
    if p:
        if p.stat().st_size > MAX_FILE:
            raise ValueError(f"{p.name} is larger than 50 MB")
        return p.read_bytes(), p
    s = (src or "").strip()
    if s.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            r = await c.get(s)
            r.raise_for_status()
        data = r.content
    elif s.startswith("data:") or len(s) > 200:
        data = base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)
    else:
        raise ValueError(f"file not found: {src}")
    if not data or len(data) > MAX_FILE:
        raise ValueError("file is empty or larger than 50 MB")
    return data, None


async def as_file(src: str) -> Path:
    data, p = await load(src)
    if p:
        return p
    TMP.mkdir(parents=True, exist_ok=True)
    p = TMP / f"in_{uuid.uuid4().hex[:10]}.{ext_of(data)}"
    p.write_bytes(data)
    return p


async def image_item(src: str, role: str) -> dict:
    s = (src or "").strip()
    if s.startswith(("http://", "https://")):
        return MD.media_item("image", s, role)
    data, _ = await load(s)
    return MD.media_item("image", MD.data_uri(data, ext_of(data)), role)


async def audio_items(sources: list[str]) -> list[dict]:
    """Reference audios, each trimmed to 15 s / n (the same rule as the GPU server)."""
    per = MD.REF_AUDIO_TOTAL / max(1, len(sources))
    items = []
    for s in sources:
        p = await as_file(s)
        d = (await MD.probe(p)).get("duration") or 0.0
        if d < MD.REF_AUDIO_MIN:
            raise ValueError(f"{p.name} is {d:.1f} s; reference audio must be at least 2 s")
        if d <= per and p.suffix.lower() in (".mp3", ".wav"):
            data, ext = p.read_bytes(), p.suffix[1:]
        else:
            TMP.mkdir(parents=True, exist_ok=True)
            out = TMP / f"aud_{uuid.uuid4().hex[:8]}.wav"
            await MD.to_wav(p, out, min(per, d))
            data, ext = out.read_bytes(), "wav"
        items.append(MD.media_item("audio", MD.data_uri(data, ext), "reference_audio"))
    return items


async def video_item(src: str, role: str) -> tuple[dict, float, Path]:
    p = await as_file(src)
    info = await MD.probe(p)
    if p.stat().st_size > MD.INLINE_VIDEO_BYTES:
        if src.strip().startswith(("http://", "https://")):
            return MD.media_item("video", src.strip(), role), info.get("duration") or 0.0, p
        raise ValueError(f"{p.name} is larger than 20 MB (send a smaller file or an http(s) link)")
    data = p.read_bytes()
    return MD.media_item("video", MD.data_uri(data, ext_of(data)), role), info.get("duration") or 0.0, p


def job_file(job_id: str) -> Path:
    return JOBS / f"{safe_name(job_id)}.json"


def load_job(job_id: str) -> dict:
    try:
        return json.loads(job_file(job_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_job(job_id: str, rec: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    tmp = job_file(job_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_file(job_id))


def make_code(kind: str, args: dict) -> str:
    raw = json.dumps({"kind": kind, **args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def needs_confirmation(code: str, summary: dict) -> dict:
    return {"status": "needs_confirmation", "confirm_code": code, **summary,
            "note": "Nothing was sent or billed. Tell the user the estimated cost and wait for their "
                    "OK, then call again with the same arguments plus "
                    f"confirm_code='{code}'."}


def usage_cost(rec: dict, usage: dict) -> Optional[float]:
    return MD.cost_from_usage(rec.get("kind"), rec.get("model"), rec.get("resolution"), usage)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("minimax-direct", instructions=(
    "MiniMax Direct: MiniMax H3 video from MiniMax's cloud API, run on the user's computer. "
    "Call prompt_guide() once before writing video prompts. generate_video and upscale_to_2k COST "
    "MONEY: the first call returns an estimated cost and a confirm_code and sends nothing; tell the "
    "user the cost and call again with confirm_code only after they agree. Then poll get_job(job_id) "
    "until done; results are saved as files on this computer (paths are returned). "
    "Inputs are file paths on this computer. Default is 768P. The API has no seed and no lip-sync "
    "to an exact speech track; reference audio only sets a voice timbre. "
    "upscale_to_2k turns a chosen 768P H3 clip into 2K with the same picture (~$0.05/s)."))


@mcp.tool()
async def generate_video(
    prompt: str,
    first_frame_image: Optional[str] = None,
    last_frame_image: Optional[str] = None,
    reference_images: Optional[list[str]] = None,
    reference_audios: Optional[list[str]] = None,
    reference_videos: Optional[list[str]] = None,
    model: Literal["MiniMax-H3", "MiniMax-H3-Max"] = "MiniMax-H3",
    resolution: Literal["480P", "768P", "2K"] = "768P",
    duration_seconds: int = 5,
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] = "16:9",
    prompt_expansion: Literal["disabled", "balanced", "quality"] = "disabled",
    name: Optional[str] = None,
    confirm_code: Optional[str] = None,
) -> dict:
    """Make a video with audio on MiniMax H3 (cloud API). COSTS MONEY (two-step confirm).
    Prices per second (Oct 2026 list): H3 768P $0.08, H3 2K $0.13, H3-Max 480P $0.05, 768P $0.08.

    Modes (cannot be mixed): text only; image-to-video (first_frame_image, optional
    last_frame_image; the clip follows the image's aspect ratio); reference (1-9
    reference_images, 0-3 reference_videos, 0-3 reference_audios; 2-15 s each, 15 s total).
    Files: full paths on this computer (e.g. C:\\...\\nori.png), paths relative to the output
    folder, or http(s) URLs.
    model / resolution: MiniMax-H3 = 768P or 2K, 4-15 s. MiniMax-H3-Max = faster, 480P or
    768P, 5-15 s. Default MiniMax-H3 768P (can be upscaled to 2K later with upscale_to_2k).
    prompt_expansion: H3-Max only; 'disabled' keeps your structured prompt exactly.
    name: optional file name prefix for the result, e.g. 'EP1_15B-1'.
    Returns needs_confirmation (+ cost) first; after the user agrees call again with
    confirm_code, then poll get_job(job_id)."""
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 7000:
        return {"status": "error", "error": "prompt must be 1-7000 characters"}
    imgs = [x for x in (reference_images or []) if x]
    auds = [x for x in (reference_audios or []) if x]
    vids = [x for x in (reference_videos or []) if x]
    i2v, r2v = bool(first_frame_image or last_frame_image), bool(imgs or auds or vids)
    if i2v and r2v:
        return {"status": "error", "error": "first/last frame and reference inputs cannot be mixed (API rule)"}
    if last_frame_image and not first_frame_image:
        return {"status": "error", "error": "last_frame_image needs first_frame_image too"}
    if len(imgs) > 9 or len(auds) > 3 or len(vids) > 3:
        return {"status": "error", "error": "at most 9 reference images, 3 audios and 3 videos"}
    if auds and not (imgs or vids):
        return {"status": "error", "error": "reference audio needs at least one reference image or video"}
    d = int(round(float(duration_seconds)))
    problem = MD.check_model(model, resolution, d)
    if problem:
        return {"status": "error", "error": problem}
    content, vid_secs = [MD.text_item(prompt)], 0.0
    try:
        if first_frame_image:
            content.append(await image_item(first_frame_image, "first_frame"))
        if last_frame_image:
            content.append(await image_item(last_frame_image, "last_frame"))
        for s in imgs:
            content.append(await image_item(s, "reference_image"))
        for s in vids:
            item, secs, _ = await video_item(s, "reference_video")
            if not MD.REF_VIDEO_MIN <= secs <= MD.REF_VIDEO_TOTAL:
                raise ValueError(f"reference video is {secs:.1f} s; each must be 2-15 s")
            vid_secs += secs
            content.append(item)
        if vid_secs > MD.REF_VIDEO_TOTAL:
            raise ValueError(f"reference videos total {vid_secs:.1f} s; the limit is 15 s")
        if auds:
            content.extend(await audio_items(auds))
    except Exception as e:
        return {"status": "error", "error": f"could not prepare inputs: {e}"}
    body = {"model": model, "content": content, "resolution": resolution, "duration": d,
            "ratio": "adaptive" if i2v else aspect_ratio}
    if model == "MiniMax-H3-Max":
        body["extra"] = {"prompt_expansion_mode": prompt_expansion}
    if MD.body_size(body) > MD.MAX_BODY_BYTES:
        return {"status": "error", "error": "inputs are larger than the API's 64 MB request limit"}
    mode = "image_to_video" if i2v else "reference_to_video" if r2v else "text_to_video"
    n_img = sum(1 for c in content if c.get("type") == "image_url")
    summary = {"mode": mode, "model": model, "resolution": resolution, "duration_seconds": d,
               "reference_video_seconds": round(vid_secs, 2),
               **MD.estimate_generation(model, resolution, d, n_img, round(vid_secs, 2))}
    code = make_code("generate_video", {
        "prompt": prompt, "first": first_frame_image, "last": last_frame_image, "imgs": imgs,
        "auds": auds, "vids": vids, "model": model, "resolution": resolution, "duration": d,
        "ratio": aspect_ratio, "expansion": prompt_expansion, "name": name})
    if confirm_code != code:
        return needs_confirmation(code, summary)
    try:
        task_id = await api().create(body)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    save_job(task_id, {"job_id": task_id, "kind": "generation", "created": time.time(),
                       "name": name, "request": body, **summary})
    return {"job_id": task_id, "status": "queued", **summary,
            "note": "Rendering on MiniMax's cloud (usually a few minutes). Poll get_job(job_id)."}



@mcp.tool()
async def upscale_to_2k(
    video: str,
    prompt: Optional[str] = None,
    first_frame_image: Optional[str] = None,
    last_frame_image: Optional[str] = None,
    reference_images: Optional[list[str]] = None,
    reference_audios: Optional[list[str]] = None,
    keep_original_audio: bool = False,
    name: Optional[str] = None,
    confirm_code: Optional[str] = None,
) -> dict:
    """Turn a 768P MiniMax H3 clip into 2K with the same picture and motion (MiniMax
    regeneration, ~$0.05 per second: 5 s = ~$0.26). COSTS MONEY (two-step confirm).

    video: the job_id of a 768P generate_video result (its prompt and inputs are reused
      automatically), or the path of a 768P H3 clip on this computer, e.g. one rendered on the
      rented GPU (quality standard/high, 4-15 s, 24 fps, with sound).
    For a clip given by path, MiniMax needs EXACTLY what it was rendered with: pass the same
      prompt and the same first_frame_image / last_frame_image or reference_images /
      reference_audios files.
    keep_original_audio: put the clip's own soundtrack on the 2K result (use it for clips whose
      mouths were synced to a speech track).
    name: optional file name prefix for the result."""
    rec = load_job(video.strip()) if video and not local_path(video) else {}
    try:
        if rec:
            if rec.get("resolution") != "768P" or rec.get("kind") != "generation":
                return {"status": "error", "error": "only 768P generate_video results can be upscaled"}
            if not rec.get("file") or not Path(rec["file"]).is_file():
                return {"status": "error", "error": "that job has no downloaded video yet (call get_job first)"}
            path = Path(rec["file"])
            content = list(rec["request"]["content"])
        else:
            path = local_path(video)
            if not path:
                return {"status": "error", "error": f"video not found: {video}"}
            if not (prompt or "").strip():
                return {"status": "error", "error": "pass the exact prompt this clip was rendered with"}
            if (first_frame_image or last_frame_image) and (reference_images or reference_audios):
                return {"status": "error", "error": "first/last frame and reference inputs cannot be mixed"}
            content = [MD.text_item(prompt.strip())]
            if first_frame_image:
                content.append(await image_item(first_frame_image, "first_frame"))
            if last_frame_image:
                content.append(await image_item(last_frame_image, "last_frame"))
            for s in reference_images or []:
                content.append(await image_item(s, "reference_image"))
            if reference_audios:
                content.extend(await audio_items([a for a in reference_audios if a]))
        info = await MD.probe(path)
        problems = MD.regen_spec_problems(info)
        if problems:
            return {"status": "error", "error": "this clip cannot be upscaled: " + "; ".join(problems)}
        if path.stat().st_size > MD.INLINE_VIDEO_BYTES:
            return {"status": "error", "error": "the clip is larger than 20 MB"}
        data = path.read_bytes()
        content.append(MD.media_item("video", MD.data_uri(data, ext_of(data)), "base_video"))
    except Exception as e:
        return {"status": "error", "error": f"could not prepare inputs: {e}"}
    body = {"model": "MiniMax-H3", "content": content, "resolution": "2K"}
    if MD.body_size(body) > MD.MAX_BODY_BYTES:
        return {"status": "error", "error": "inputs are larger than the API's 64 MB request limit"}
    seconds = round(info["frames"] / MD.REGEN_FPS, 2)
    n_img = sum(1 for c in content if c.get("type") == "image_url")
    summary = {"mode": "upscale_to_2k", "source": str(path), "source_size": f"{info['width']}x{info['height']}",
               "duration_seconds": seconds, "keep_original_audio": bool(keep_original_audio),
               **MD.estimate_regeneration(seconds, n_img, rec.get("reference_video_seconds") or 0.0)}
    code = make_code("upscale_to_2k", {
        "video": video, "prompt": prompt, "first": first_frame_image, "last": last_frame_image,
        "imgs": reference_images, "auds": reference_audios, "keep": bool(keep_original_audio), "name": name})
    if confirm_code != code:
        return needs_confirmation(code, summary)
    try:
        task_id = await api().regenerate(body)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    save_job(task_id, {"job_id": task_id, "kind": "regeneration", "created": time.time(), "resolution": "2K",
                       "name": name or rec.get("name") or path.stem, "source_file": str(path), **summary})
    return {"job_id": task_id, "status": "queued", **summary,
            "note": "Upscaling on MiniMax's cloud (usually a few minutes). Poll get_job(job_id)."}


async def _fetch(job_id: str, task: dict) -> None:
    rec = load_job(job_id)
    res = task.get("resolution") or rec.get("resolution") or ""
    stem = safe_name(rec.get("name") or rec.get("mode") or "MiniMax") or "MiniMax"
    dest = OUT / f"{stem}_{res}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    await api().download(task["content"]["url"], dest)
    if rec.get("keep_original_audio") and rec.get("source_file"):
        mixed = dest.with_name(dest.stem + "_mix.mp4")
        try:
            await MD.replace_audio(dest, Path(rec["source_file"]), mixed)
            mixed.replace(dest)
            rec["audio"] = "original soundtrack kept"
        except Exception as e:
            rec["audio"] = f"could not keep the original soundtrack ({e})"
    usage = task.get("usage") or {}
    rec.update(file=str(dest), usage=usage, finished=time.time(), status="done",
               cost_usd=usage_cost(rec, usage))
    save_job(job_id, rec)


def _done(job_id: str, rec: dict) -> dict:
    out = {"job_id": job_id, "status": "done", "file": rec["file"], "folder": str(Path(rec["file"]).parent)}
    for k in ("mode", "resolution", "duration_seconds", "cost_usd", "audio"):
        if rec.get(k) is not None:
            out[k] = rec[k]
    return out


async def _status(job_id: str) -> dict:
    rec = load_job(job_id)
    if rec.get("file") and Path(rec["file"]).is_file():
        return _done(job_id, rec)
    try:
        task = await api().query(job_id)
    except MD.APIError as e:
        return {"job_id": job_id, "status": "error", "error": str(e)}
    except (httpx.HTTPError, RuntimeError) as e:
        return {"job_id": job_id, "status": "running", "note": f"could not reach MiniMax right now ({e})"}
    s = task.get("status")
    if s in ("queued", "running"):
        return {"job_id": job_id, "status": s}
    if s in ("failed", "cancelled"):
        err = task.get("error") or {}
        if rec:
            rec["status"] = s
            save_job(job_id, rec)
        return {"job_id": job_id, "status": "error",
                "error": f"MiniMax task {s}: {err.get('message') or err.get('code') or 'no details'}"}
    if s != "succeeded" or not (task.get("content") or {}).get("url"):
        return {"job_id": job_id, "status": "unknown", "note": f"unexpected MiniMax status {s!r}"}
    t = _downloads.get(job_id)
    if t is not None and t.done():
        _downloads.pop(job_id, None)
        if t.exception():
            return {"job_id": job_id, "status": "error",
                    "error": f"download failed: {t.exception()} (call get_job again to retry)"}
        return _done(job_id, load_job(job_id))
    if t is None:
        _downloads[job_id] = asyncio.create_task(_fetch(job_id, task))
    return {"job_id": job_id, "status": "running", "note": "MiniMax finished; downloading the video"}


@mcp.tool()
async def get_job(job_id: str, wait_seconds: int = 45) -> dict:
    """Check a generate_video / upscale_to_2k job. Waits up to wait_seconds (max 50) and
    returns as soon as it finishes; call again while status is queued/running.
    When done, 'file' is the saved video on this computer."""
    job_id = (job_id or "").strip()
    deadline = time.monotonic() + max(0, min(MAX_WAIT, int(wait_seconds or 0)))
    while True:
        st = await _status(job_id)
        if st["status"] in ("done", "error", "unknown") or time.monotonic() >= deadline:
            return st
        await asyncio.sleep(2 if _downloads.get(job_id) else 5)


@mcp.tool()
async def list_jobs(limit: int = 10) -> dict:
    """Recent MiniMax Direct jobs (newest first) with their files, plus approximate total spend."""
    recs = []
    for p in JOBS.glob("*.json"):
        try:
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    recs.sort(key=lambda r: r.get("created") or 0, reverse=True)
    keys = ("job_id", "mode", "name", "resolution", "duration_seconds", "status", "file", "cost_usd",
            "estimated_cost_usd")
    items = [{k: r.get(k) for k in keys if r.get(k) is not None} for r in recs[: max(1, min(100, int(limit)))]]
    for r, it in zip(recs, items):
        it["created"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("created") or 0))
    return {"jobs": items, "total_spend_usd": round(sum(r.get("cost_usd") or 0 for r in recs), 3),
            "output_folder": str(OUT), "api_key_set": bool(api_key())}


@mcp.tool()
async def extract_frame(video: str, position: Literal["first", "last", "time"] = "last",
                        time_seconds: float = 0.0) -> dict:
    """Save one frame of a clip as a PNG (in the output folder's 'frames' subfolder) and return its
    path, e.g. the LAST frame of a shot to use as first_frame_image of the next shot.
    video: a path on this computer or the job_id of a finished job."""
    rec = load_job(video.strip()) if video and not local_path(video) else {}
    path = Path(rec["file"]) if rec.get("file") else local_path(video)
    if not path or not path.is_file():
        return {"status": "error", "error": f"video not found: {video}"}
    tag = f"t{time_seconds:g}" if position == "time" else position
    dest = OUT / "frames" / f"{path.stem[:60]}_{tag}.png"
    try:
        await MD.extract_frame(path, dest, position, time_seconds)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {"status": "done", "file": str(dest)}


GUIDE = """MiniMax H3 prompt guide (official structured format). The API sends your text to the model
as-is, so write the structure yourself. English everywhere except the exact spoken words.

== Base format ==
integrated_multimodal_description: [Shot 1] <style label>. <subjects + setting, 1-2 identity
details each>. <main action>. The camera <move> with <amplitude> at <speed>. <Character> (S1),
<age range, gender, pitch, timbre, speaking rate, native Central Thai accent>, <delivery> says
<d>[Thai] exact words</d>. <non-speaking characters: 'does not speak'>.

overall_soundscape: <1-4 sentences: ambience, physical sounds, non-verbal human sounds. No dialogue.>

non_diegetic_music: <score the characters cannot hear> or N/A

With first_frame_image, put this line FIRST, then a blank line, then the three fields:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

== Dialogue ==
- Inside <d>...</d> only the language tag and the exact words: <d>[Thai] รีบ หนี ไป ลูก!</d>.
- Never put dialogue in plain "quotes" (can become burned-in subtitles).
- Stable speaker IDs (S1), (S2) with voice attributes at first appearance.
- Thai is not one of H3's best-supported languages: short lines (about 3 Thai syllables per
  second), spaces between words, numbers as words, no rare words or English loanwords.
- duration_seconds = line + action; extra time makes the model invent filler speech.
- Off-screen voice: 'says in an off-screen voiceover <d>[Thai] ...</d>', lips of on-screen
  characters stay closed. Cut-off line: end the tag with <cutoff>.
- The API has no seed: each call is a new take. For exact words with lip-sync use the rented
  GPU's speech_audio instead.

== Reference mode ==
Start with: subject_definitions: <Picture 1> is Nori, a girl explorer (dark hair, teal jacket,
light-green scarf, boxy detector on her right hip). <Picture 2> is the market walkway.
<Audio 1> is the voice timbre reference for Nori. Reference the timbre only; do not reuse its words.
Then the three fields: '<Picture 1> (S1) says in the voice timbre of <Audio 1> <d>[Thai] ...</d>'.

== Visual recipe ==
Camera move, light/time of day, 3-5 concrete actions, style lock ('keep the exact art style and
colors of the first frame'), and the props each character carries in EVERY shot (continuity).
"""


@mcp.tool()
async def prompt_guide() -> dict:
    """Read before writing prompts: the official H3 prompt format, dialogue tags, Thai tips."""
    return {"guide": GUIDE}


if __name__ == "__main__":
    mcp.run()
