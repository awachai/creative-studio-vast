"""
Creative MCP server  (MiniMax H3 video + Z-Image + MiniMax Speech)
------------------------------------------------------------------
Runs next to ComfyUI on the GPU machine and exposes, over streamable HTTP
(protected by a bearer token):

  text_to_video, image_to_video, reference_to_video   -> MiniMax H3 (local GPU)
  text_to_image                                       -> Z-Image-Turbo (local GPU)
  text_to_speech, list_voices                         -> MiniMax Speech API (cloud)
  api_video, upscale_to_2k                            -> MiniMax H3 direct API (cloud, paid)
  extract_frame                                       -> one frame of a clip as an image_id
  get_job, list_recent_outputs, server_status

Env vars:
  MCP_TOKEN          required. Clients send "Authorization: Bearer <MCP_TOKEN>".
  COMFY_URL          ComfyUI inside the machine (default http://127.0.0.1:18188)
  COMFY_OUTPUT_DIR   ComfyUI output folder (default /workspace/ComfyUI/output)
  PUBLIC_URL         public base URL of this server, used for download links
  MCP_PORT           listen port on 127.0.0.1 (default 8765)
  MINIMAX_API_KEY    optional, enables text_to_speech / list_voices / api_video / upscale_to_2k
                     (the video tools need a pay-as-you-go MiniMax account with balance)
  MINIMAX_API_BASE   default https://api.minimax.io  (China keys: https://api.minimaxi.com)
  MINIMAX_VOICE_ID   default voice for text_to_speech
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import io
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote, unquote

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

import minimax_direct as MD
import workflows as W
from oauth import SCOPE, OwnerOAuthProvider

MCP_TOKEN = os.environ.get("MCP_TOKEN", "")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:18188").rstrip("/")
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output"))
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/workspace/ComfyUI/input"))
ASSET_PREFIX = "asset:"
# Cloudflare tunnel cuts a request at ~100 s and MCP clients time out around 60 s,
# so a single call never blocks longer than this; callers simply call get_job again.
MAX_WAIT = 45
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
MCP_PORT = int(os.environ.get("MCP_PORT", "8765"))
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io").rstrip("/")
MINIMAX_VOICE_ID = os.environ.get("MINIMAX_VOICE_ID", "English_expressive_narrator")
JOBS_DIR = Path(os.environ.get("STUDIO_JOBS_DIR", str(Path(__file__).with_name("jobs"))))
API_PREFIX = "api:"   # job_ids of MiniMax cloud tasks, e.g. "api:424010985738629"
API_SUB = "api"       # output subfolder for downloaded cloud results
video_api = MD.VideoAPI(MINIMAX_API_KEY, MINIMAX_API_BASE) if MINIMAX_API_KEY else None

if not MCP_TOKEN:
    raise SystemExit("MCP_TOKEN is not set")

CLIENT_ID = f"creative-mcp-{uuid.uuid4().hex[:8]}"
MAX_INPUT_BYTES = 25 * 1024 * 1024
Aspect = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]
Quality = Literal["draft", "standard", "high"]


# ---------------------------------------------------------------------------
# Download links (signed per file, so they can be shared without the token)
# ---------------------------------------------------------------------------
def _sign(subfolder: str, filename: str) -> str:
    return hmac.new(MCP_TOKEN.encode(), f"{subfolder}/{filename}".encode(),
                    hashlib.sha256).hexdigest()[:32]


def download_url(subfolder: str, filename: str) -> str:
    sub = subfolder or "_"
    path = f"/files/{quote(sub, safe='')}/{quote(filename, safe='')}?sig={_sign(sub, filename)}"
    return f"{PUBLIC_URL}{path}" if PUBLIC_URL else path


# ---------------------------------------------------------------------------
# ComfyUI helpers
# ---------------------------------------------------------------------------
async def comfy_get(path: str, **params):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{COMFY_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()


async def load_input_image(source: str) -> tuple[bytes, str]:
    """Accept an http(s) URL, a data: URI, or raw base64. Returns (bytes, ext)."""
    source = (source or "").strip()
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            r = await c.get(source)
            r.raise_for_status()
            data = r.content
    else:
        if source.startswith("data:"):
            source = source.split(",", 1)[1]
        try:
            data = base64.b64decode(source, validate=False)
        except binascii.Error as e:
            raise ValueError(f"image is neither a URL nor valid base64: {e}")
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("image is empty or larger than 25 MB")
    ext = "png"
    if data[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        ext = "webp"
    return data, ext


def image_size(data: bytes) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


async def upload_image_bytes(data: bytes, ext: str, label: Optional[str] = None) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", label or "").strip("_")[:40]
    name = f"asset_{slug + '_' if slug else ''}{uuid.uuid4().hex[:8]}.{ext}"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{COMFY_URL}/upload/image",
                         files={"image": (name, data, f"image/{'jpeg' if ext == 'jpg' else ext}")},
                         data={"type": "input", "overwrite": "true"})
        r.raise_for_status()
        return r.json()["name"]


def _ext_of(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


async def resolve_image(source: str) -> tuple[str, bytes]:
    """Turn an image reference into a ComfyUI input filename (uploading only if needed).
    Accepts: 'asset:<name>' from upload_image, a download URL produced by this server
    (read straight from local disk), any http(s) URL, a data URI, or base64."""
    s = (source or "").strip()
    if s.startswith(ASSET_PREFIX):
        name = Path(s[len(ASSET_PREFIX):]).name
        path = COMFY_INPUT_DIR / name
        if not path.is_file():
            raise ValueError(f"unknown image_id {s} (use list_assets to see available ids)")
        return name, path.read_bytes()
    if PUBLIC_URL and s.startswith(f"{PUBLIC_URL}/files/"):
        m = re.match(r"^/files/([^/]+)/([^?]+)", s[len(PUBLIC_URL):])
        if m:
            sub, fn = unquote(m.group(1)), Path(unquote(m.group(2))).name
            path = COMFY_OUTPUT_DIR / ("" if sub == "_" else Path(sub).name) / fn
            if path.is_file():
                data = path.read_bytes()
                return await upload_image_bytes(data, _ext_of(data)), data
    data, ext = await load_input_image(s)
    return await upload_image_bytes(data, ext), data


# ---------------------------------------------------------------------------
# Temporary upload links (signed, expiring) — files go straight to disk, not via the LLM
# ---------------------------------------------------------------------------
def _upload_ticket(exp: int) -> str:
    sig = hmac.new(MCP_TOKEN.encode(), f"upload:{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"{exp}.{sig}"


def _upload_url(exp: int) -> str:
    return f"{PUBLIC_URL}/upload/{_upload_ticket(exp)}"


def _ticket_ok(ticket: str) -> bool:
    try:
        exp_s, _ = ticket.split(".", 1)
        exp = int(exp_s)
    except Exception:
        return False
    return exp >= time.time() and hmac.compare_digest(ticket, _upload_ticket(exp))


def _audio_ext(data: bytes) -> Optional[str]:
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[4:8] == b"ftyp":
        return "m4a"
    return None


async def resolve_audio(source: str) -> str:
    """Turn an audio reference into a ComfyUI input filename.
    Accepts: a URL returned by this server (e.g. from text_to_speech; read from local disk),
    'asset:<name>' of an audio file already in the input folder, any http(s) URL, a data URI,
    or base64 (mp3/wav/flac/ogg/m4a, max 25 MB)."""
    s = (source or "").strip()
    if s.startswith(ASSET_PREFIX):
        name = Path(s[len(ASSET_PREFIX):]).name
        if not (COMFY_INPUT_DIR / name).is_file():
            raise ValueError(f"unknown audio id {s}")
        return name
    data = None
    if PUBLIC_URL and s.startswith(f"{PUBLIC_URL}/files/"):
        m = re.match(r"^/files/([^/]+)/([^?]+)", s[len(PUBLIC_URL):])
        if m:
            sub, fn = unquote(m.group(1)), Path(unquote(m.group(2))).name
            path = COMFY_OUTPUT_DIR / ("" if sub == "_" else Path(sub).name) / fn
            if path.is_file():
                data = path.read_bytes()
    if data is None:
        if s.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(s)
                r.raise_for_status()
                data = r.content
        else:
            try:
                data = base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s, validate=False)
            except binascii.Error as e:
                raise ValueError(f"audio is neither a URL nor valid base64: {e}")
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("audio is empty or larger than 25 MB")
    ext = _audio_ext(data)
    if not ext:
        raise ValueError("unrecognised audio format (use mp3, wav, flac, ogg or m4a)")
    name = f"audio_{uuid.uuid4().hex[:10]}.{ext}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (COMFY_INPUT_DIR / name).write_bytes(data)
    return name


_NODE_CACHE: dict[str, bool] = {}


async def comfy_has_node(name: str) -> bool:
    if name not in _NODE_CACHE:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{COMFY_URL}/object_info/{name}")
                _NODE_CACHE[name] = r.status_code == 200 and name in r.json()
        except Exception:
            return False
    return _NODE_CACHE[name]


async def prepare_speech(speech_audio: Optional[str]) -> Optional[str]:
    """Resolve speech_audio to a ComfyUI input file, or raise with a clear message."""
    if not speech_audio:
        return None
    if not await comfy_has_node("MiniMaxH3AddGuide"):
        raise ValueError("this ComfyUI version has no MiniMaxH3AddGuide node; update ComfyUI "
                         "or use reference_to_video(reference_audios=...) instead")
    return await resolve_audio(speech_audio)


async def submit(workflow: dict) -> str:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": CLIENT_ID})
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the job: {r.text[:800]}")
    return r.json()["prompt_id"]


def _collect_outputs(entry: dict) -> list[dict]:
    files = []
    for out in entry.get("outputs", {}).values():
        for item in out.get("images", []) + out.get("videos", []) + out.get("gifs", []) + out.get("audio", []):
            if item.get("type", "output") != "output":
                continue
            fn = item.get("filename", "")
            kind = ("video" if fn.endswith((".mp4", ".webm", ".mov")) else
                    "image" if fn.endswith((".png", ".jpg", ".jpeg", ".webp")) else
                    "audio" if fn.endswith((".mp3", ".wav", ".flac", ".opus")) else "file")
            meta = item.get("metadata") or {}
            rec = {"type": kind, "filename": fn, "url": download_url(item.get("subfolder", ""), fn)}
            for k in ("width", "height", "fps"):
                if meta.get(k):
                    rec[k] = meta[k]
            if meta.get("duration"):
                rec["duration_seconds"] = round(meta["duration"], 2)
            files.append(rec)
    return files


# ---------------------------------------------------------------------------
# Job records (jobs/<id>.json): the workflow/request behind each clip, so a clip can be
# upscaled to 2K later even after ComfyUI forgets its history.
# ---------------------------------------------------------------------------
def _job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', job_id)}.json"


def load_job(job_id: str) -> dict:
    try:
        return json.loads(_job_file(job_id).read_text())
    except Exception:
        return {}


def save_job(job_id: str, rec: dict) -> None:
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        p = _job_file(job_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(p)
    except Exception:
        pass


def _output_items(entry: dict) -> list[list[str]]:
    items = []
    for out in entry.get("outputs", {}).values():
        for item in out.get("images", []) + out.get("videos", []) + out.get("gifs", []) + out.get("audio", []):
            if item.get("type", "output") == "output" and item.get("filename"):
                items.append([item.get("subfolder", ""), item["filename"]])
    return items


def _remember_outputs(job_id: str, entry: dict) -> None:
    rec = load_job(job_id)
    if rec and not rec.get("outputs"):
        rec["outputs"] = _output_items(entry)
        save_job(job_id, rec)


async def job_status(job_id: str) -> dict:
    if job_id.startswith(API_PREFIX):
        return await api_job_status(job_id)
    hist = await comfy_get(f"/history/{job_id}")
    if job_id in hist:
        entry = hist[job_id]
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            err = next((m[1] for m in st.get("messages", []) if m[0] == "execution_error"), {})
            return {"job_id": job_id, "status": "error",
                    "error": err.get("exception_message", "unknown error"),
                    "node": err.get("node_type")}
        files = _collect_outputs(entry)
        if st.get("completed") or files:
            _remember_outputs(job_id, entry)
            return {"job_id": job_id, "status": "done", "outputs": files}
    q = await comfy_get("/queue")
    if any(item[1] == job_id for item in q.get("queue_running", [])):
        return {"job_id": job_id, "status": "running"}
    pending = [item[1] for item in q.get("queue_pending", [])]
    if job_id in pending:
        return {"job_id": job_id, "status": "queued", "queue_position": pending.index(job_id) + 1}
    return {"job_id": job_id, "status": "unknown", "note": "Job not found (ComfyUI may have restarted)."}


async def run_job(workflow: dict, info: dict, wait_seconds: int) -> dict:
    try:
        job_id = await submit(workflow)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    save_job(job_id, {"job_id": job_id, "tool": info.get("tool"), "created": time.time(),
                      "workflow": workflow})
    result = {"job_id": job_id, "status": "queued", **info}
    wait = max(0, min(MAX_WAIT, int(wait_seconds or 0)))
    if wait:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            st = await job_status(job_id)
            if st["status"] in ("done", "error"):
                return {**result, **st}
        result.update(status="running", note="Still rendering; poll get_job with this job_id.")
    return result


def _seed(seed: Optional[int]) -> int:
    return int(seed) if seed is not None else random.randint(1, 2**48)


def _clamp_duration(d: float) -> float:
    return max(1.0, min(15.0, float(d)))


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
_kwargs = dict(stateless_http=True, json_response=True)
try:  # newer SDKs validate the Host header; traffic arrives through a tunnel
    from mcp.server.transport_security import TransportSecuritySettings
    _kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
except Exception:
    pass

INSTRUCTIONS = (
        "Private creative studio on a GPU server. Video tools (MiniMax H3) create clips WITH "
        "synchronized audio, so describe sounds and dialogue in the prompt. Image tool uses "
        "Z-Image-Turbo. Speech uses MiniMax Speech. GPU jobs are asynchronous: call the tool, then "
        "call get_job(job_id) — it waits until the job finishes (up to 45 s per call; just call it again while queued/running — a 10 s high-quality clip takes several minutes). "
        "NEVER type base64 for image files: call get_upload_url and send the files with code "
        "(curl/python) or give the link to the user; reuse the returned image_ids in every shot. Use quality='draft' with 3-5 s "
        "clips for previews, standard/high only for final shots, and batch jobs that use the same "
        "tool together (switching between video tools and text_to_image reloads models, ~15 s). "
        "For a character saying exact lines with lip-sync: make the line with text_to_speech, then pass its URL as speech_audio to image_to_video or reference_to_video (duration = speech length + ~1 s, put the same line in a <d>[Thai] ...</d> tag in the prompt). reference_audios only sets the voice timbre. "
        "Results come back as download URLs. Typical times on draft quality with turbo: "
        "image ~5 s, 3-5 s video ~30-60 s; high quality / long clips take several minutes. "
        "Write prompts in English for best results; spoken dialogue may be in other languages. "
        "BEFORE WRITING ANY VIDEO PROMPT call prompt_guide() once: H3 expects its official structured "
        "prompt format (integrated_multimodal_description / overall_soundscape / non_diegetic_music, "
        "speaker IDs (S1), dialogue as <d>[Thai] exact words</d>). Plain quoted dialogue gives worse "
        "speech and can render as burned-in subtitles. Final shots: quality='high'; shots with spoken "
        "lines: turbo=False (speech is the first thing low step counts ruin). For a scene that already "
        "exists as an image, prefer image_to_video with that image as the first frame. "
        "HYBRID WORKFLOW: drafts and 768P finals run on this GPU (no per-clip cost; same seed + same "
        "quality/aspect/duration reproduces a draft's composition). When the user wants 2K, call "
        "upscale_to_2k(job_id of the chosen local clip, quality standard/high, 4-15 s): MiniMax's "
        "cloud turns that exact clip into 2K for ~$0.05/s. api_video renders on MiniMax's cloud "
        "instead (no seed, no lip-sync to a speech track; use it only when the user asks). Both cloud "
        "tools COST MONEY: the first call returns the estimated cost and a confirm_code; tell the user "
        "the cost and call again with confirm_code only after they agree. extract_frame turns any "
        "frame of a clip into an image_id (last frame of one shot -> first frame of the next)."
)
# OAuth (for ChatGPT & co.) + the static owner token as a bearer token (for API/test use).
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions

if not PUBLIC_URL.startswith("https://"):
    raise SystemExit("PUBLIC_URL must be the https tunnel URL (start the server with start.sh)")
oauth_provider = OwnerOAuthProvider(MCP_TOKEN, PUBLIC_URL, Path(__file__).with_name("oauth_state.json"))
_kwargs["auth_server_provider"] = oauth_provider
_kwargs["auth"] = AuthSettings(
    issuer_url=PUBLIC_URL,
    resource_server_url=f"{PUBLIC_URL}/mcp",
    client_registration_options=ClientRegistrationOptions(
        enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]),
    revocation_options=RevocationOptions(enabled=True),
    required_scopes=[SCOPE],
)
try:
    mcp = FastMCP("creative-studio", instructions=INSTRUCTIONS, **_kwargs)
except TypeError:  # older SDK without transport_security
    _kwargs.pop("transport_security", None)
    mcp = FastMCP("creative-studio", instructions=INSTRUCTIONS, **_kwargs)


@mcp.tool()
async def text_to_video(
    prompt: str,
    duration_seconds: float = 5.0,
    aspect_ratio: Aspect = "16:9",
    quality: Quality = "standard",
    seed: Optional[int] = None,
    turbo: bool = True,
    wait_seconds: int = 0,
) -> dict:
    """Generate a video with native stereo audio from text (MiniMax H3).

    prompt: scene, subject, action, lighting, camera movement, sounds/dialogue.
    duration_seconds: 1-15. quality: draft (~864x480) / standard / high (~1280x720).
    turbo: 8-step LoRA, much faster. wait_seconds: block up to N s (max 45) for the result.
    Returns job_id; poll get_job until status == "done"."""
    if not prompt.strip():
        return {"status": "error", "error": "prompt is empty"}
    d, s = _clamp_duration(duration_seconds), _seed(seed)
    wf = W.text_to_video(prompt.strip(), d, aspect_ratio, quality, s, turbo)
    return await run_job(wf, {"tool": "text_to_video", "seed": s, "duration_seconds": d}, wait_seconds)


@mcp.tool()
async def image_to_video(
    image: str,
    prompt: str,
    last_frame_image: Optional[str] = None,
    duration_seconds: float = 5.0,
    aspect_ratio: Literal["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"] = "auto",
    quality: Quality = "standard",
    seed: Optional[int] = None,
    turbo: bool = True,
    wait_seconds: int = 0,
    speech_audio: Optional[str] = None,
    speech_start_seconds: float = 0.0,
) -> dict:
    """Animate an image into a video with audio (MiniMax H3). The image is the first frame.

    image: an image_id from upload_image ("asset:..."; preferred, no re-upload), a URL
      returned by this server, any http(s) URL, a data URI, or base64 (max 25 MB).
    prompt: what happens next: motion, camera, sounds/dialogue.
    last_frame_image: optional image the clip should end on (first+last frame mode).
    aspect_ratio: "auto" matches the input image.
    speech_audio: optional finished speech track (best: the URL returned by
      text_to_speech). It is placed into the clip's soundtrack so the character's mouth
      moves to THESE exact words (lip-sync). Set duration_seconds to the speech length
      (text_to_speech returns duration_seconds) plus ~0.5-1 s; audio past the end is cut.
      In the prompt say who is speaking and put the same line in a dialogue tag, e.g.
      'The woman in the red apron (S1) looks at the camera and says <d>[Thai] ...</d>'
      (see prompt_guide).
    speech_start_seconds: when the speech starts in the clip (default 0).
    Returns job_id; poll get_job until status == "done"."""
    try:
        first, data = await resolve_image(image)
        last = None
        if last_frame_image:
            last, _ = await resolve_image(last_frame_image)
    except Exception as e:
        return {"status": "error", "error": f"could not load image: {e}"}
    aspect = aspect_ratio
    if aspect == "auto":
        size = image_size(data)
        aspect = W.closest_aspect(*size) if size else "16:9"
    d, s = _clamp_duration(duration_seconds), _seed(seed)
    try:
        speech = await prepare_speech(speech_audio)
    except Exception as e:
        return {"status": "error", "error": f"could not use speech_audio: {e}"}
    wf = W.image_to_video(prompt.strip(), first, d, aspect, quality, s, turbo, last)
    if speech:
        W.add_speech_guide(wf, speech, speech_start_seconds)
    return await run_job(wf, {"tool": "image_to_video", "seed": s, "duration_seconds": d,
                              "aspect_ratio": aspect, "lip_sync": bool(speech)}, wait_seconds)


@mcp.tool()
async def reference_to_video(
    reference_images: list[str],
    prompt: str,
    duration_seconds: float = 5.0,
    aspect_ratio: Aspect = "16:9",
    quality: Quality = "standard",
    seed: Optional[int] = None,
    turbo: bool = False,
    wait_seconds: int = 0,
    reference_audios: Optional[list[str]] = None,
    speech_audio: Optional[str] = None,
    speech_start_seconds: float = 0.0,
) -> dict:
    """Create a new video with audio that features the people/objects/styles from 1-9
    reference images (MiniMax H3 Ref2VA). Unlike image_to_video, the references are not the
    first frame; they define who/what appears.

    reference_images: list of image_ids from upload_image ("asset:...", preferred),
      URLs returned by this server, http(s) URLs, data URIs, or base64 images, in order.
    reference_audios: optional 1-3 voice samples that set HOW a character sounds (timbre,
      way of speaking). Best: the URL returned by text_to_speech (a clear 5-10 s line in the
      character's voice). Also accepts http(s) URLs, data URIs or base64 audio. Total is
      trimmed to 15 s. The model speaks the dialogue written in the prompt in that voice,
      with matching lip movement; it does not replay the sample word for word.
      For exact words + lip-sync use speech_audio instead (or as well).
    speech_audio: optional finished speech track (best: the URL returned by
      text_to_speech). It is placed into the clip's soundtrack so the character's mouth
      moves to THESE exact words (lip-sync). Set duration_seconds to the speech length
      (text_to_speech returns duration_seconds) plus ~0.5-1 s; audio past the end is cut.
      In the prompt say who is speaking and put the same line in a dialogue tag, e.g.
      'The woman in the red apron (S1) looks at the camera and says <d>[Thai] ...</d>'
      (see prompt_guide).
    speech_start_seconds: when the speech starts in the clip (default 0).
    turbo: default False (full 20 steps). The reference turbo LoRA is only 4 steps and gives
      blurry, off-model results; use turbo=True only for quick drafts.
    Put the location/background image in reference_images too and name it in the prompt,
      and describe each character's key look next to its tag (hair, clothes, props) so the
      model keeps them on-model.
    prompt: the scene. Refer to references with tags in order: <Picture 1>, <Picture 2> ...
      for images and <Audio 1> ... for audios, e.g.
      '<Picture 1> (S1) looks at the camera and says in the voice timbre of <Audio 1>
      <d>[Thai] สวัสดีครับ</d>'. The audio sample must say a DIFFERENT sentence; add
      'Reference the timbre only; do not reuse its words.' (see prompt_guide).
    Returns job_id; poll get_job until status == "done"."""
    if not 1 <= len(reference_images) <= 9:
        return {"status": "error", "error": "provide between 1 and 9 reference images"}
    reference_audios = [a for a in (reference_audios or []) if a]
    if len(reference_audios) > 3:
        return {"status": "error", "error": "provide at most 3 reference audios"}
    try:
        names = []
        for src in reference_images:
            name, _ = await resolve_image(src)
            names.append(name)
    except Exception as e:
        return {"status": "error", "error": f"could not load image: {e}"}
    try:
        audio_names = [await resolve_audio(a) for a in reference_audios]
    except Exception as e:
        return {"status": "error", "error": f"could not load audio: {e}"}
    d, s = _clamp_duration(duration_seconds), _seed(seed)
    try:
        speech = await prepare_speech(speech_audio)
    except Exception as e:
        return {"status": "error", "error": f"could not use speech_audio: {e}"}
    wf = W.reference_to_video(prompt.strip(), names, d, aspect_ratio, quality, s, turbo, audio_names)
    if speech:
        W.add_speech_guide(wf, speech, speech_start_seconds)
    return await run_job(wf, {"tool": "reference_to_video", "seed": s, "duration_seconds": d,
                              "references": len(names), "reference_audios": len(audio_names),
                              "lip_sync": bool(speech)},
                         wait_seconds)


@mcp.tool()
async def text_to_image(
    prompt: str,
    aspect_ratio: Aspect = "1:1",
    quality: Quality = "standard",
    seed: Optional[int] = None,
    wait_seconds: int = 60,
) -> dict:
    """Generate a still image from text (Z-Image-Turbo, 8 steps, a few seconds).
    quality: draft (~0.6 MP) / standard (~1 MP) / high (~2 MP).
    Tip: the resulting image URL can be passed to image_to_video or reference_to_video."""
    if not prompt.strip():
        return {"status": "error", "error": "prompt is empty"}
    s = _seed(seed)
    wf = W.text_to_image(prompt.strip(), aspect_ratio, quality, s)
    return await run_job(wf, {"tool": "text_to_image", "seed": s}, wait_seconds)


@mcp.tool()
async def text_to_speech(
    text: str,
    voice_id: Optional[str] = None,
    language: Optional[str] = "auto",
    model: Literal["speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo"] = "speech-2.8-hd",
    speed: float = 1.0,
    pitch: int = 0,
    emotion: Optional[Literal["happy", "sad", "angry", "fearful", "disgusted",
                              "surprised", "calm", "fluent"]] = None,
    audio_format: Literal["mp3", "wav", "flac"] = "mp3",
    volume: float = 1.0,
    deepen_lighten: int = 0,
    stronger_softer: int = 0,
    nasal_crisp: int = 0,
    sound_effect: Optional[Literal["spacious_echo", "auditorium_echo", "lofi_telephone", "robotic"]] = None,
    voice_mix: Optional[list[dict]] = None,
) -> dict:
    """Convert text to natural speech with MiniMax Speech (cloud API, billed per character).

    text: up to 10,000 characters. Pauses: <#0.5#> (seconds). speech-2.8 models also
      understand tags like (laughs), (sighs).
    voice_id: from list_voices; defaults to the server's configured voice.
    language: language boost, e.g. "Thai", "English", "Japanese", or "auto".
    speed 0.5-2, pitch -12..12 (semitones), volume 0.1-10.
    Voice modifier (same as the MiniMax web "Voice Modifier"), each -100..100, 0 = unchanged:
      deepen_lighten: negative = deeper, positive = lighter/brighter.
      stronger_softer: negative = stronger/more forceful, positive = softer.
      nasal_crisp: negative = more nasal, positive = crisper.
    sound_effect: spacious_echo, auditorium_echo, lofi_telephone (phone call), robotic.
    voice_mix: blend 2-4 voices into a new one instead of voice_id, e.g.
      [{"voice_id": "Thai_female_1_sample1", "weight": 70}, {"voice_id": "English_PlayfulGirl", "weight": 30}]
      (weights 1-100). Keep a character's exact settings to reuse the same voice.
    Returns a download URL for the audio file (synchronous)."""
    if not MINIMAX_API_KEY:
        return {"status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
    text = (text or "").strip()
    if not text:
        return {"status": "error", "error": "text is empty"}
    if len(text) > 10000:
        return {"status": "error", "error": "text exceeds 10,000 characters"}
    clamp = lambda v, lo, hi: max(lo, min(hi, v))
    mix = []
    for m in voice_mix or []:
        if isinstance(m, dict) and m.get("voice_id"):
            mix.append({"voice_id": str(m["voice_id"]), "weight": clamp(int(m.get("weight", 50)), 1, 100)})
    if voice_mix and not 2 <= len(mix) <= 4:
        return {"status": "error", "error": "voice_mix needs 2-4 entries like {voice_id, weight}"}
    voice_setting = {"voice_id": "" if mix else (voice_id or MINIMAX_VOICE_ID),
                     "speed": clamp(float(speed), 0.5, 2.0), "vol": clamp(float(volume), 0.1, 10.0),
                     "pitch": clamp(int(pitch), -12, 12)}
    if emotion:
        voice_setting["emotion"] = emotion
    body = {
        "model": model, "text": text, "stream": False,
        "voice_setting": voice_setting,
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": audio_format, "channel": 1},
        "output_format": "hex",
    }
    if mix:
        body["timbre_weights"] = mix
    modify = {k: clamp(int(v), -100, 100) for k, v in
              (("pitch", deepen_lighten), ("intensity", stronger_softer), ("timbre", nasal_crisp)) if v}
    if sound_effect:
        modify["sound_effects"] = sound_effect
    if modify:
        body["voice_modify"] = modify
    if language:
        body["language_boost"] = language
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{MINIMAX_API_BASE}/v1/t2a_v2", json=body,
                         headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"})
    try:
        j = r.json()
    except Exception:
        return {"status": "error", "error": f"MiniMax HTTP {r.status_code}: {r.text[:300]}"}
    base = j.get("base_resp") or {}
    if base.get("status_code", 0) != 0 or not (j.get("data") or {}).get("audio"):
        return {"status": "error", "error": f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}"}
    audio = bytes.fromhex(j["data"]["audio"])
    sub = "audio"
    fn = f"MiniMax_Speech_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{audio_format}"
    (COMFY_OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    (COMFY_OUTPUT_DIR / sub / fn).write_bytes(audio)
    extra = j.get("extra_info") or {}
    return {"status": "done", "tool": "text_to_speech",
            "outputs": [{"type": "audio", "filename": fn, "url": download_url(sub, fn),
                         "duration_seconds": round((extra.get("audio_length") or 0) / 1000, 2)}],
            "characters_billed": extra.get("usage_characters"),
            "voice_id": voice_setting["voice_id"] or None, **({"voice_mix": mix} if mix else {}),
            **({"voice_modify": modify} if modify else {})}


@mcp.tool()
async def list_voices(search: Optional[str] = None, limit: int = 50) -> dict:
    """List MiniMax system voices (and your cloned voices). search filters by id/name/
    description, e.g. "Thai", "female", "narrator"."""
    if not MINIMAX_API_KEY:
        return {"status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{MINIMAX_API_BASE}/v1/get_voice", json={"voice_type": "all"},
                         headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"})
    try:
        j = r.json()
    except Exception:
        return {"status": "error", "error": f"MiniMax HTTP {r.status_code}: {r.text[:300]}"}
    voices = []
    for group in ("system_voice", "voice_cloning", "voice_generation"):
        for v in j.get(group) or []:
            desc = v.get("description")
            if isinstance(desc, list):
                desc = " ".join(map(str, desc))
            voices.append({"voice_id": v.get("voice_id"), "name": v.get("voice_name"),
                           "description": desc, "kind": group})
    if search:
        s = search.lower()
        voices = [v for v in voices if s in " ".join(str(x) for x in v.values()).lower()]
    return {"count": len(voices), "voices": voices[: max(1, min(500, int(limit)))]}


@mcp.tool()
async def design_voice(
    description: str,
    preview_text: str,
    voice_id: Optional[str] = None,
) -> dict:
    """Create a brand-new custom voice from a text description (MiniMax Voice Design).
    Works for any language, e.g. Thai voices beyond the 4 built-in ones.

    description: who is speaking and how, e.g. "Thai woman in her 30s, warm and soft,
      gentle bedtime-story narrator, slow pace" (English descriptions work best).
    preview_text: a sample sentence (max 500 chars) in the target language, e.g. Thai.
    voice_id: optional custom id (letters, digits, underscore), e.g. "thai_storyteller_f".
    Returns the new voice_id (use it in text_to_speech) and a preview audio URL.
    The voice then also appears in list_voices under kind 'voice_generation'."""
    if not MINIMAX_API_KEY:
        return {"status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
    if not description.strip() or not preview_text.strip():
        return {"status": "error", "error": "description and preview_text are required"}
    body = {"prompt": description.strip(), "preview_text": preview_text.strip()[:500]}
    if voice_id:
        body["voice_id"] = voice_id
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{MINIMAX_API_BASE}/v1/voice_design", json=body,
                         headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"})
    try:
        j = r.json()
    except Exception:
        return {"status": "error", "error": f"MiniMax HTTP {r.status_code}: {r.text[:300]}"}
    base = j.get("base_resp") or {}
    if base.get("status_code", 0) != 0 or not j.get("voice_id"):
        return {"status": "error", "error": f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}"}
    out = {"status": "done", "tool": "design_voice", "voice_id": j["voice_id"]}
    if j.get("trial_audio"):
        sub = "audio"
        fn = f"VoiceDesign_{j['voice_id']}_{uuid.uuid4().hex[:6]}.mp3"
        (COMFY_OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
        (COMFY_OUTPUT_DIR / sub / fn).write_bytes(bytes.fromhex(j["trial_audio"]))
        out["outputs"] = [{"type": "audio", "filename": fn, "url": download_url(sub, fn)}]
    return out


@mcp.tool()
async def upload_image(image: str, label: Optional[str] = None) -> dict:
    """Upload an image ONCE and get a reusable image_id ("asset:...").
    Use it for characters, products, locations, first/last frames that appear in many
    shots: pass the image_id to image_to_video (image / last_frame_image) or
    reference_to_video (reference_images) instead of re-sending base64 every call.

    image: base64, data URI, or http(s) URL (max 25 MB). For files on disk prefer
      get_upload_url + curl/python — typing base64 by hand is slow and can corrupt the file.
    label: short name to recognise it later, e.g. "noori", "punti", "kitchen_scene"."""
    try:
        data, ext = await load_input_image(image)
        name = await upload_image_bytes(data, ext, label)
    except Exception as e:
        return {"status": "error", "error": f"could not load image: {e}"}
    size = image_size(data)
    out = {"status": "done", "image_id": f"{ASSET_PREFIX}{name}", "bytes": len(data)}
    if size:
        out["width"], out["height"] = size
    return out


@mcp.tool()
async def get_upload_url(minutes: int = 30) -> dict:
    """Get a temporary upload URL for sending image FILES straight to the server by code,
    instead of typing base64 into a tool call (much faster, never corrupts files).

    Use it whenever the images are files on the user's computer or in your sandbox:
      curl -s -F "file=@/path/nori.png" -F "label=nori" "<upload_url>"
    (repeat -F "file=@..." to send several files at once; label applies to all of them).
    Python: requests.post(upload_url, files={"file": open(path, "rb")}, data={"label": "nori"})
    The JSON response lists the new image_ids ("asset:..."). The same URL opened in a browser
    shows a drag-and-drop page, so you can also give it to the user. Valid for `minutes` (max 240)."""
    m = max(1, min(240, int(minutes)))
    url = _upload_url(int(time.time()) + m * 60)
    return {"status": "done", "upload_url": url, "expires_in_minutes": m,
            "curl_example": f'curl -s -F "file=@/path/to/image.png" -F "label=nori" "{url}"'}


@mcp.tool()
async def list_assets(search: Optional[str] = None, limit: int = 50) -> dict:
    """List images previously uploaded with upload_image (newest first), with their image_ids."""
    files = [p for p in COMFY_INPUT_DIR.glob("asset_*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if search:
        files = [p for p in files if search.lower() in p.name.lower()]
    items = [{"image_id": f"{ASSET_PREFIX}{p.name}", "bytes": p.stat().st_size,
              "uploaded": time.strftime("%Y-%m-%d %H:%M", time.gmtime(p.stat().st_mtime)) + " UTC"}
             for p in files[: max(1, min(500, int(limit)))]]
    return {"count": len(files), "assets": items}


@mcp.tool()
async def get_job(job_id: str, wait_seconds: int = 45) -> dict:
    """Check a job from text_to_video / image_to_video / reference_to_video /
    text_to_image, or a cloud job ("api:...") from api_video / upscale_to_2k. Waits up to wait_seconds (default 45, max 45; larger values are capped) and returns as soon as the
    job finishes, so call it again only if status is still queued/running.
    status: queued, running, done, error. When done, 'outputs' holds URLs."""
    deadline = time.monotonic() + max(0, min(MAX_WAIT, int(wait_seconds or 0)))
    job_id = (job_id or "").strip()
    while True:
        st = await job_status(job_id)
        if st["status"] in ("done", "error", "unknown") or time.monotonic() >= deadline:
            return st
        await asyncio.sleep(5 if job_id.startswith(API_PREFIX) else 2)


@mcp.tool()
async def list_recent_outputs(limit: int = 10) -> dict:
    """List the most recent finished GPU outputs (videos and images) with download URLs."""
    limit = max(1, min(50, int(limit)))
    hist = await comfy_get("/history", max_items=limit)
    items = []
    for job_id in reversed(list(hist.keys())):
        st = await job_status(job_id)
        for o in st.get("outputs", []):
            items.append({"job_id": job_id, **o})
    return {"outputs": items[:limit]}


# ---------------------------------------------------------------------------
# MiniMax direct API (cloud H3): api_video, upscale_to_2k, extract_frame
# ---------------------------------------------------------------------------
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MAX_VIDEO_BYTES = 50 * 1024 * 1024
_api_downloads: dict[str, asyncio.Task] = {}


def _output_path(sub: str, fn: str) -> Path:
    sub = "" if sub in ("", "_") else Path(sub).name
    return COMFY_OUTPUT_DIR / sub / Path(fn).name


def _parse_server_url(s: str) -> Optional[tuple[str, str]]:
    """(subfolder, filename) of a /files/ link made by this server, else None."""
    if PUBLIC_URL and s.startswith(f"{PUBLIC_URL}/files/"):
        s = s[len(PUBLIC_URL):]
    m = re.match(r"^/files/([^/]+)/([^?]+)", s)
    if not m:
        return None
    sub = unquote(m.group(1))
    return ("" if sub == "_" else sub), Path(unquote(m.group(2))).name


def _video_ext(data: bytes) -> Optional[str]:
    if data[4:8] == b"ftyp":
        return "mov" if data[8:10] == b"qt" else "mp4"
    return None


def _confirm_code(kind: str, args: dict) -> str:
    raw = json.dumps({"kind": kind, **args}, sort_keys=True, ensure_ascii=False)
    return hmac.new(MCP_TOKEN.encode(), raw.encode(), hashlib.sha256).hexdigest()[:10]


def _needs_confirmation(code: str, summary: dict) -> dict:
    return {"status": "needs_confirmation", "confirm_code": code, **summary,
            "note": "Nothing was sent or billed. Show the user the estimated cost and wait for "
                    "their OK, then call this tool again with the same arguments plus "
                    f"confirm_code='{code}'."}


async def find_video(ref: str) -> dict:
    """Locate a clip made by this server: a local job_id, an 'api:...' job_id, or a /files/ URL.
    Returns {path, sub, filename, job_id, workflow, record}."""
    ref = (ref or "").strip()
    if ref.startswith(API_PREFIX):
        rec = load_job(ref)
        if not rec.get("local_file"):
            raise ValueError(f"{ref} has no finished video yet (call get_job until it is done)")
        sub, fn = rec["local_file"]
        return {"path": _output_path(sub, fn), "sub": sub, "filename": fn, "job_id": ref,
                "workflow": None, "record": rec}
    if UUID_RE.match(ref):
        rec, workflow, items = load_job(ref), None, []
        try:
            hist = await comfy_get(f"/history/{ref}")
            entry = hist.get(ref) or {}
            items = _output_items(entry)
            p = entry.get("prompt")
            if isinstance(p, list) and len(p) > 2 and isinstance(p[2], dict):
                workflow = p[2]
        except Exception:
            pass
        items = items or rec.get("outputs") or []
        workflow = workflow or rec.get("workflow")
        vids = [it for it in items if it[1].lower().endswith((".mp4", ".mov", ".webm"))]
        if not vids:
            raise ValueError(f"job {ref} has no finished video (still rendering, or not a video job)")
        sub, fn = vids[0]
        return {"path": _output_path(sub, fn), "sub": sub, "filename": fn, "job_id": ref,
                "workflow": workflow, "record": rec}
    parsed = _parse_server_url(ref)
    if parsed:
        sub, fn = parsed
        found = {"path": _output_path(sub, fn), "sub": sub, "filename": fn, "job_id": None,
                 "workflow": None, "record": {}}
        for p in JOBS_DIR.glob("*.json"):
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if [sub, fn] in (rec.get("outputs") or []) or rec.get("local_file") == [sub, fn]:
                found.update(job_id=rec.get("job_id"), workflow=rec.get("workflow"), record=rec)
                break
        return found
    raise ValueError("video must be a job_id (local or 'api:...') or a video URL returned by this server")


async def _video_path_any(ref: str) -> Path:
    """Like find_video, but also downloads an outside http(s) video into the input folder."""
    try:
        return (await find_video(ref))["path"]
    except ValueError:
        if not ref.strip().startswith(("http://", "https://")):
            raise
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
        r = await c.get(ref.strip())
        r.raise_for_status()
    data = r.content
    if not data or len(data) > MAX_VIDEO_BYTES:
        raise ValueError("video is empty or larger than 50 MB")
    ext = _video_ext(data) or "mp4"
    path = COMFY_INPUT_DIR / f"video_{uuid.uuid4().hex[:10]}.{ext}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


async def _source_bytes(src: str) -> bytes:
    """Bytes of an asset:..., a /files/ URL of this server, an http(s) URL, a data URI or base64."""
    s = (src or "").strip()
    if s.startswith(ASSET_PREFIX):
        path = COMFY_INPUT_DIR / Path(s[len(ASSET_PREFIX):]).name
        if not path.is_file():
            raise ValueError(f"unknown id {s}")
        return path.read_bytes()
    parsed = _parse_server_url(s)
    if parsed and _output_path(*parsed).is_file():
        return _output_path(*parsed).read_bytes()
    data, _ = await load_input_image(s)   # URL / data URI / base64 (any file type)
    return data


async def _api_image(src: str) -> str:
    """Image reference -> URL string for the API (outside URLs pass through, the rest inline)."""
    s = (src or "").strip()
    if s.startswith(("http://", "https://")) and not _parse_server_url(s):
        return s
    data = await _source_bytes(s)
    return MD.data_uri(data, _ext_of(data))


async def _api_reference_audios(sources: list[str]) -> list[dict]:
    files = []
    for s in sources:
        p = COMFY_INPUT_DIR / await resolve_audio(s)
        d = (await MD.probe(p)).get("duration") or 0.0
        if d < MD.REF_AUDIO_MIN:
            raise ValueError(f"reference audio is {d:.1f} s; each must be at least {MD.REF_AUDIO_MIN:.0f} s")
        files.append((p, d))
    over = sum(d for _, d in files) > MD.REF_AUDIO_TOTAL
    per = MD.REF_AUDIO_TOTAL / max(1, len(files))
    items = []
    for p, d in files:
        if not over and p.suffix.lower() in (".mp3", ".wav"):
            data, ext = p.read_bytes(), p.suffix[1:]
        else:
            out = p.with_name(p.stem + "_api.wav")
            await MD.to_wav(p, out, per if over else None)
            data, ext = out.read_bytes(), "wav"
        items.append(MD.media_item("audio", MD.data_uri(data, ext), "reference_audio"))
    return items


async def _api_video_source(path: Path, public: Optional[tuple[str, str]] = None) -> str:
    """Inline small clips; larger ones go by this server's signed public URL."""
    size = path.stat().st_size
    if size <= MD.INLINE_VIDEO_BYTES:
        data = path.read_bytes()
        return MD.data_uri(data, _video_ext(data) or "mp4")
    if public:
        return download_url(*public)
    raise ValueError("video is larger than 20 MB and is not a file of this server")


async def _regen_inputs_from_workflow(wf: dict) -> tuple[str, list[dict], list[str]]:
    """Exact prompt + first/last/reference inputs a local H3 job was rendered with."""
    h3 = (wf.get("h3") or {}).get("inputs") or {}
    text = h3.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("could not read the prompt this clip was rendered with")

    def linked(link, key):
        node = wf.get(link[0]) if isinstance(link, list) and link else None
        return ((node or {}).get("inputs") or {}).get(key), node

    items, notes = [], []

    def image_item(link, role):
        name, _ = linked(link, "image")
        path = COMFY_INPUT_DIR / Path(str(name)).name
        if not path.is_file():
            raise ValueError(f"input image {name} is no longer on this server")
        data = path.read_bytes()
        items.append(MD.media_item("image", MD.data_uri(data, _ext_of(data)), role))

    if "first_frame" in h3:
        image_item(h3["first_frame"], "first_frame")
    if "last_frame" in h3:
        image_item(h3["last_frame"], "last_frame")
    idx = lambda k: int(k.rsplit("_", 1)[1])
    for k in sorted((k for k in h3 if k.startswith("ref_images.")), key=idx):
        image_item(h3[k], "reference_image")
    for k in sorted((k for k in h3 if k.startswith("ref_audios.")), key=idx):
        seconds, trim = linked(h3[k], "duration")
        name, _ = linked((trim or {}).get("inputs", {}).get("audio"), "audio")
        src = COMFY_INPUT_DIR / Path(str(name)).name
        if not src.is_file():
            raise ValueError(f"reference audio {name} is no longer on this server")
        out = src.with_name(src.stem + "_regen.wav")
        await MD.to_wav(src, out, float(seconds) if seconds else MD.REF_AUDIO_TOTAL)
        items.append(MD.media_item("audio", MD.data_uri(out.read_bytes(), "wav"), "reference_audio"))
    if "speech_guide" in wf:
        notes.append("This clip used a lip-sync speech track. The API has no input for it; the "
                      "speech is already in the clip's soundtrack.")
    return text.strip(), items, notes


def _usage_cost(rec: dict, usage: dict) -> Optional[float]:
    secs = usage.get("output_seconds") or usage.get("total_seconds")
    if not secs:
        return None
    if rec.get("kind") == "regeneration":
        return round(MD.REGEN_PRICE_PER_SECOND * secs, 3)
    rate = MD.PRICE_PER_SECOND.get((rec.get("model"), rec.get("resolution")))
    if rate is None:
        return None
    extra_img = max(0, (usage.get("input_image_count") or 0) - MD.FREE_IMAGES) * MD.EXTRA_IMAGE_PRICE
    return round(rate * (secs + (usage.get("input_seconds") or 0)) + extra_img, 3)


async def _api_fetch(job_id: str, task: dict) -> None:
    rec = load_job(job_id)
    task_id = job_id[len(API_PREFIX):]
    fn = f"MiniMax_API_{task.get('resolution') or rec.get('resolution') or ''}_{task_id}.mp4"
    dest = _output_path(API_SUB, fn)
    await video_api.download(task["content"]["url"], dest)
    if rec.get("keep_original_audio") and rec.get("source_file"):
        src = _output_path(*rec["source_file"])
        mixed = dest.with_name(dest.stem + "_mix.mp4")
        try:
            await MD.replace_audio(dest, src, mixed)
            mixed.replace(dest)
            rec["audio"] = "original soundtrack of the source clip kept"
        except Exception as e:
            rec["audio"] = f"could not keep the original soundtrack ({e}); MiniMax audio used"
    usage = task.get("usage") or {}
    rec.update(local_file=[API_SUB, fn], usage=usage, finished=time.time(),
               cost_usd=_usage_cost(rec, usage))
    save_job(job_id, rec)


def _api_done(job_id: str, rec: dict) -> dict:
    sub, fn = rec["local_file"]
    out = {"job_id": job_id, "status": "done", "tool": rec.get("mode"),
           "outputs": [{"type": "video", "filename": fn, "url": download_url(sub, fn)}]}
    for k in ("resolution", "duration_seconds", "cost_usd", "audio"):
        if rec.get(k) is not None:
            out[k] = rec[k]
    return out


async def api_job_status(job_id: str) -> dict:
    rec = load_job(job_id)
    if rec.get("local_file") and _output_path(*rec["local_file"]).is_file():
        return _api_done(job_id, rec)
    if not video_api:
        return {"job_id": job_id, "status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
    try:
        task = await video_api.query(job_id[len(API_PREFIX):])
    except MD.APIError as e:
        return {"job_id": job_id, "status": "error", "error": str(e)}
    except httpx.HTTPError as e:
        return {"job_id": job_id, "status": "running", "note": f"could not reach MiniMax right now ({e}); try again"}
    s = task.get("status")
    if s in ("queued", "running"):
        return {"job_id": job_id, "status": s, "tool": rec.get("mode")}
    if s in ("failed", "cancelled"):
        err = task.get("error") or {}
        return {"job_id": job_id, "status": "error",
                "error": f"MiniMax task {s}: {err.get('message') or err.get('code') or 'no details'}"}
    if s != "succeeded":
        return {"job_id": job_id, "status": "unknown", "note": f"unexpected MiniMax status {s!r}"}
    if not (task.get("content") or {}).get("url"):
        return {"job_id": job_id, "status": "error", "error": "MiniMax finished but returned no video URL"}
    t = _api_downloads.get(job_id)
    if t is not None and t.done():
        _api_downloads.pop(job_id, None)
        if t.exception():
            return {"job_id": job_id, "status": "error",
                    "error": f"download failed: {t.exception()} (call get_job again to retry)"}
        return _api_done(job_id, load_job(job_id))
    if t is None:
        _api_downloads[job_id] = asyncio.create_task(_api_fetch(job_id, task))
    return {"job_id": job_id, "status": "running", "note": "MiniMax finished; downloading the video to this server"}


@mcp.tool()
async def api_video(
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
    confirm_code: Optional[str] = None,
) -> dict:
    """Render a video with audio on MiniMax's cloud H3 (direct API). COSTS REAL MONEY per second.

    Two-step: the first call returns status 'needs_confirmation' with the estimated cost and a
    confirm_code (nothing is sent). Show the cost to the user, and only after they agree call
    again with the same arguments + confirm_code. Then poll get_job(job_id) ('api:...').
    Prices (Oct 2026 list): H3 768P $0.08/s, H3 2K $0.13/s, H3-Max 480P $0.05/s, 768P $0.08/s.

    Modes (cannot be mixed): text only; image-to-video (first_frame_image, optional
    last_frame_image); reference (1-9 reference_images, 0-3 reference_videos 2-15 s each,
    0-3 reference_audios 2-15 s each, 15 s total each kind; audio needs an image or video).
    Images/audio: image_id ("asset:..."), URL from this server, any http(s) URL, data URI, base64.
    reference_videos: job_id or URL of a clip from this server, or an http(s) URL.
    reference_audios only set a voice TIMBRE; the API cannot lip-sync to an exact speech track
    (do that on the local GPU with speech_audio). The API has no seed: every call is a new take.
    model: MiniMax-H3 (768P / 2K, 4-15 s) or MiniMax-H3-Max (faster, 480P / 768P, 5-15 s).
    aspect_ratio: used for text-only and reference mode; image mode follows the image.
    prompt_expansion: H3-Max only; 'disabled' keeps your structured prompt exactly.
    Same prompt format as the local tools (see prompt_guide)."""
    if not video_api:
        return {"status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
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
    mode = "image_to_video" if i2v else "reference_to_video" if r2v else "text_to_video"
    args = {"prompt": prompt, "first": first_frame_image, "last": last_frame_image, "imgs": imgs,
            "auds": auds, "vids": vids, "model": model, "resolution": resolution, "duration": d,
            "ratio": aspect_ratio, "expansion": prompt_expansion}
    code = _confirm_code("api_video", args)

    content, vid_seconds = [MD.text_item(prompt)], 0.0
    try:
        if first_frame_image:
            content.append(MD.media_item("image", await _api_image(first_frame_image), "first_frame"))
        if last_frame_image:
            content.append(MD.media_item("image", await _api_image(last_frame_image), "last_frame"))
        for s in imgs:
            content.append(MD.media_item("image", await _api_image(s), "reference_image"))
        for s in vids:
            path = await _video_path_any(s)
            info = await MD.probe(path)
            vs = info.get("duration") or 0.0
            if not MD.REF_VIDEO_MIN <= vs <= MD.REF_VIDEO_TOTAL:
                raise ValueError(f"reference video is {vs:.1f} s; each must be 2-15 s")
            vid_seconds += vs
            parsed = _parse_server_url(s) or None
            try:
                v = await find_video(s)
                parsed = (v["sub"], v["filename"])
            except Exception:
                pass
            content.append(MD.media_item("video", await _api_video_source(path, parsed), "reference_video"))
        if vid_seconds > MD.REF_VIDEO_TOTAL:
            raise ValueError(f"reference videos total {vid_seconds:.1f} s; the limit is 15 s")
        if auds:
            content.extend(await _api_reference_audios(auds))
    except Exception as e:
        return {"status": "error", "error": f"could not prepare inputs: {e}"}

    body = {"model": model, "content": content, "resolution": resolution, "duration": d,
            "ratio": "adaptive" if i2v else aspect_ratio}
    if model == "MiniMax-H3-Max":
        body["extra"] = {"prompt_expansion_mode": prompt_expansion}
    if MD.body_size(body) > MD.MAX_BODY_BYTES:
        return {"status": "error", "error": "inputs are larger than the API's 64 MB request limit; use fewer or smaller files"}
    n_images = sum(1 for c in content if c.get("type") == "image_url")
    summary = {"tool": "api_video", "mode": mode, "model": model, "resolution": resolution,
               "duration_seconds": d,
               **MD.estimate_generation(model, resolution, d, n_images, round(vid_seconds, 2))}
    if confirm_code != code:
        return _needs_confirmation(code, summary)
    try:
        task_id = await video_api.create(body)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    job_id = API_PREFIX + task_id
    save_job(job_id, {"job_id": job_id, "kind": "generation", "created": time.time(),
                      "request": body, **summary})
    return {"job_id": job_id, "status": "queued", **summary,
            "note": "Rendering on MiniMax's cloud (usually a few minutes). Poll get_job(job_id)."}


@mcp.tool()
async def upscale_to_2k(
    video: str,
    keep_original_audio: Optional[bool] = None,
    prompt: Optional[str] = None,
    confirm_code: Optional[str] = None,
) -> dict:
    """Turn a finished 768P H3 clip into 2K with MiniMax's regeneration API (~$0.05 per
    second, e.g. 5 s = $0.25). The picture and motion stay the clip you chose.
    COSTS REAL MONEY: the first call checks the clip and returns the estimated cost and a
    confirm_code (nothing is sent). Show it to the user; after they agree call again with the
    same arguments + confirm_code. Then poll get_job(job_id) ('api:...').

    video: job_id of a local H3 job (best), a video URL returned by this server, or the job_id
      of an api_video 768P render. The clip must be quality 'standard' or 'high' (not 'draft'),
      4-15 s long, made by this server (24 fps, with audio). The exact prompt and the
      first-frame/reference inputs it was rendered with are sent automatically.
    keep_original_audio: put the source clip's soundtrack on the 2K result (default: yes for
      lip-sync clips made with speech_audio, otherwise keep MiniMax's audio).
    prompt: only needed when the server no longer knows how the clip was made."""
    if not video_api:
        return {"status": "error", "error": "MINIMAX_API_KEY is not configured on the server"}
    try:
        v = await find_video(video)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    path = v["path"]
    if not path.is_file():
        return {"status": "error", "error": f"{v['filename']} is no longer on this server"}
    try:
        info = await MD.probe(path)
    except Exception as e:
        return {"status": "error", "error": f"could not read the clip: {e}"}
    problems = MD.regen_spec_problems(info)
    if problems:
        return {"status": "error", "error": "this clip cannot be upscaled: " + "; ".join(problems),
                "clip": {k: info.get(k) for k in ("width", "height", "fps", "frames", "has_audio")}}
    rec, wf, notes = v.get("record") or {}, v.get("workflow"), []
    try:
        if rec.get("kind") == "generation":
            if rec.get("resolution") != "768P":
                return {"status": "error", "error": "only 768P clips can be upscaled"}
            content = list(rec["request"]["content"])
        elif wf:
            text, items, notes = await _regen_inputs_from_workflow(wf)
            if prompt and prompt.strip() != text:
                notes.append("prompt argument ignored: the exact prompt the clip was rendered with is used")
            content = [MD.text_item(text), *items]
        elif prompt and prompt.strip():
            content = [MD.text_item(prompt.strip())]
            notes.append("The server does not know how this clip was made, so only the prompt is sent. "
                         "It must be the exact prompt, and clips made from a first frame or references "
                         "may come out worse.")
        else:
            return {"status": "error", "error": "the server does not know how this clip was made; pass its exact prompt"}
        content.append(MD.media_item("video", await _api_video_source(path, (v["sub"], v["filename"])),
                                     "base_video"))
    except Exception as e:
        return {"status": "error", "error": f"could not prepare inputs: {e}"}
    body = {"model": "MiniMax-H3", "content": content, "resolution": "2K"}
    if MD.body_size(body) > MD.MAX_BODY_BYTES:
        return {"status": "error", "error": "inputs are larger than the API's 64 MB request limit"}
    lip = bool(wf and "speech_guide" in wf)
    keep = lip if keep_original_audio is None else bool(keep_original_audio)
    seconds = round(info["frames"] / MD.REGEN_FPS, 2)
    summary = {"tool": "upscale_to_2k", "mode": "regenerate_2k", "source": v["filename"],
               "source_size": f"{info['width']}x{info['height']}", "duration_seconds": seconds,
               "keep_original_audio": keep, **MD.estimate_regeneration(seconds)}
    if notes:
        summary["notes"] = notes
    code = _confirm_code("upscale_to_2k", {"src": [v["sub"], v["filename"]], "keep": keep,
                                           "prompt": (prompt or "").strip()})
    if confirm_code != code:
        return _needs_confirmation(code, summary)
    try:
        task_id = await video_api.regenerate(body)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    job_id = API_PREFIX + task_id
    save_job(job_id, {"job_id": job_id, "kind": "regeneration", "created": time.time(),
                      "resolution": "2K", "source_file": [v["sub"], v["filename"]], **summary})
    return {"job_id": job_id, "status": "queued", **summary,
            "note": "Upscaling on MiniMax's cloud (usually a few minutes). Poll get_job(job_id)."}


@mcp.tool()
async def extract_frame(
    video: str,
    position: Literal["first", "last", "time"] = "last",
    time_seconds: float = 0.0,
) -> dict:
    """Grab one frame of a clip as a PNG and a reusable image_id ("asset:...").
    Continuity: the LAST frame of shot N as the image / first_frame_image of shot N+1. Also to
    pick a frame from a draft and use it as first_frame_image for api_video.
    video: job_id (local or 'api:...'), a video URL returned by this server, or any http(s) video.
    position: first, last, or time (then time_seconds from the start)."""
    try:
        path = await _video_path_any(video)
        if not path.is_file():
            return {"status": "error", "error": "the video file is no longer on this server"}
        tag = f"t{time_seconds:g}" if position == "time" else position
        fn = f"frame_{path.stem[:40]}_{tag}_{uuid.uuid4().hex[:4]}.png"
        dest = _output_path("frames", fn)
        await MD.extract_frame(path, dest, position, time_seconds)
        data = dest.read_bytes()
        name = await upload_image_bytes(data, "png", f"frame_{path.stem[:24]}_{tag}")
    except Exception as e:
        return {"status": "error", "error": f"could not extract the frame: {e}"}
    out = {"status": "done", "image_id": f"{ASSET_PREFIX}{name}", "url": download_url("frames", fn)}
    size = image_size(data)
    if size:
        out["width"], out["height"] = size
    return out


PROMPT_GUIDE = """MiniMax H3 prompt guide (official structured format; the hosted MiniMax
service rewrites prompts into this automatically, this server does not, so write it yourself).
Write everything in English except the exact spoken words.

== Base format (text_to_video, image_to_video) ==
Three fields separated by blank lines:

integrated_multimodal_description: [Shot 1] <style label>. <subjects + setting, 1-2 identity
details each>. <main action>. The camera <move> with <amplitude> at <speed>. <Character>
(S1), <age range, gender, pitch, timbre, speaking rate, native Central Thai accent>, <delivery:
e.g. shouts desperately, voice breaking> says <d>[Thai] exact words</d> <other non-speaking
characters: 'does not speak'>.

overall_soundscape: <1-4 sentences: ambience, physical sounds, non-verbal human sounds like
sobbing or screaming. No dialogue here. 'N/A' only for total silence.>

non_diegetic_music: <score the characters cannot hear> or N/A

image_to_video: put this line FIRST, then one blank line, then the three fields:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

== Dialogue rules ==
- Inside <d>...</d>: only the language tag and the exact words, verbatim, never translated:
  <d>[Thai] รีบ หนี ไป ลูก!</d>. Identity, emotion, accent, action go OUTSIDE the tag.
- Never put dialogue in plain "quotes" (quotes are for on-screen text and can become subtitles).
- Give every speaker a stable ID (S1), (S2)... with voice attributes at first appearance.
  Characters who never speak get no ID; say they do not speak.
- One sentence or one speaker turn per clip. Thai is NOT one of H3's 11 fully supported
  languages, so keep lines short (about 3 Thai syllables per second max), put spaces between
  words, write numbers as words, avoid rare words and English loanwords.
- Match duration_seconds to the line + action; extra time makes the model invent filler speech.
- Voiceover: 'says in an off-screen voiceover <d>[Thai] ...</d>', then state the on-screen
  character's lips stay closed.
- Dialogue cut off by the end of the clip: end the tag with <cutoff>.
- Speaking shots: turbo=False for finals. Draft 2-3 seeds and let the user pick, then render
  high with that seed. Stop after 3 failed tries; use text_to_speech + speech_audio instead.

== Reference mode (reference_to_video) ==
Same rules. Start the prompt with a short subject_definitions block, e.g.
subject_definitions: <Picture 1> is Nori, a girl explorer (dark hair, teal jacket, light-green
scarf, boxy detector on her right hip). <Picture 2> is the market walkway setting.
<Audio 1> is the voice timbre reference for Nori. Reference the timbre only; do not reuse its words.
Then the three fields, referring to subjects by tag: '<Picture 1> (S1) says in the voice timbre
of <Audio 1> <d>[Thai] ...</d>'. The audio sample must say a different sentence than the line.

== Visual quality recipe ==
Camera move, light/time of day, 3-5 concrete actions, style lock ('keep the exact art style,
colors and layout of the first frame' or 'keep the characters' designs exactly as in the
references'), list props each character carries in EVERY shot (continuity).

== Example (image_to_video) ==
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Nostalgic anime film style, keep the exact art
style of the first frame. Inside the ancient hall, blinding light bursts from the floor and
stones float upward. A man in his late thirties holding a lantern (S1), deep warm male voice,
native Central Thai accent, turns toward the little girl hiding behind the rocks and shouts
desperately, voice breaking <d>[Thai] โนริ! ลูก พ่อ! รีบ หนี ไป!</d> The little girl does not speak.

overall_soundscape: Roaring magical wind, cracking stone, a deep rising hum from the glowing ring gate.

non_diegetic_music: N/A
"""


@mcp.tool()
async def prompt_guide() -> dict:
    """Read this BEFORE writing prompts for text_to_video / image_to_video / reference_to_video:
    the official MiniMax H3 structured prompt format, dialogue tags <d>[Thai] ...</d>, speaker
    IDs, Thai-specific tips and examples. Returns the guide text."""
    return {"guide": PROMPT_GUIDE}


@mcp.tool()
async def server_status() -> dict:
    """GPU memory and queue status of the studio server."""
    stats = await comfy_get("/system_stats")
    q = await comfy_get("/queue")
    dev = (stats.get("devices") or [{}])[0]
    return {"gpu": dev.get("name"),
            "vram_free_gb": round((dev.get("vram_free") or 0) / 1e9, 1),
            "vram_total_gb": round((dev.get("vram_total") or 0) / 1e9, 1),
            "jobs_running": len(q.get("queue_running", [])),
            "jobs_queued": len(q.get("queue_pending", [])),
            "speech_enabled": bool(MINIMAX_API_KEY),
            "cloud_video_enabled": bool(video_api),
            "cloud_spend_usd": _cloud_spend(),
            "ffmpeg": bool(MD.ffmpeg_exe())}


def _cloud_spend() -> float:
    """Approximate MiniMax video spend on this machine (list price x reported usage)."""
    total = 0.0
    for p in JOBS_DIR.glob("api_*.json"):
        try:
            total += json.loads(p.read_text()).get("cost_usd") or 0.0
        except Exception:
            pass
    return round(total, 3)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request):
    return PlainTextResponse("ok")


@mcp.custom_route("/files/{subfolder}/{filename}", methods=["GET"])
async def files(request: Request):
    sub = request.path_params["subfolder"]
    fn = request.path_params["filename"]
    if not hmac.compare_digest(request.query_params.get("sig", ""), _sign(sub, fn)):
        return JSONResponse({"error": "invalid signature"}, status_code=403)
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("GET", f"{COMFY_URL}/view", params={
        "filename": fn, "subfolder": "" if sub == "_" else sub, "type": "output"})
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        await resp.aclose()
        await client.aclose()
        return JSONResponse({"error": "file not found"}, status_code=404)

    async def body():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    headers = {"Content-Disposition": f'inline; filename="{fn}"'}
    if "content-length" in resp.headers:
        headers["Content-Length"] = resp.headers["content-length"]
    return StreamingResponse(body(), media_type=resp.headers.get("content-type", "application/octet-stream"),
                             headers=headers)


UPLOAD_PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Creative Studio – Upload</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:640px;margin:40px auto;padding:0 16px}
#drop{border:2px dashed #555;border-radius:14px;padding:48px 16px;text-align:center;color:#aaa;cursor:pointer}
#drop.over{border-color:#4b6bff;color:#fff}input[type=text]{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;
border:1px solid #444;background:#000;color:#fff;margin:12px 0}pre{background:#1d1d1f;padding:12px;border-radius:8px;white-space:pre-wrap}</style>
</head><body><h2>Creative Studio — อัปโหลดภาพ</h2>
<input type="text" id="label" placeholder="ชื่อเรียก (ไม่บังคับ) เช่น nori">
<div id="drop">ลากไฟล์ภาพมาวางที่นี่ หรือคลิกเพื่อเลือก<input type="file" id="f" multiple accept="image/*" hidden></div>
<pre id="out"></pre>
<script>
const drop=document.getElementById('drop'),f=document.getElementById('f'),out=document.getElementById('out');
drop.onclick=()=>f.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');send(e.dataTransfer.files)};
f.onchange=()=>send(f.files);
async function send(files){const fd=new FormData();for(const x of files)fd.append('file',x);
fd.append('label',document.getElementById('label').value);out.textContent='กำลังอัปโหลด...';
const r=await fetch(location.href,{method:'POST',body:fd});const j=await r.json();
out.textContent=j.status==='done'?j.assets.map(a=>a.image_id+'   ('+a.original+')').join('\\n'):JSON.stringify(j);}
</script></body></html>"""


@mcp.custom_route("/upload/{ticket}", methods=["GET", "POST"])
async def upload_route(request: Request):
    ticket = request.path_params["ticket"]
    if not _ticket_ok(ticket):
        return JSONResponse({"status": "error", "error": "upload link expired or invalid; call get_upload_url again"},
                            status_code=403)
    if request.method == "GET":
        return HTMLResponse(UPLOAD_PAGE)
    form = await request.form()
    label = str(form.get("label") or "").strip() or None
    assets = []
    for item in form.getlist("file"):
        if not hasattr(item, "read"):
            continue
        data = await item.read()
        if not data or len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"status": "error", "error": f"{item.filename}: empty or larger than 25 MB"},
                                status_code=400)
        name = await upload_image_bytes(data, _ext_of(data), label or Path(item.filename or "").stem)
        rec = {"image_id": f"{ASSET_PREFIX}{name}", "original": item.filename, "bytes": len(data)}
        size = image_size(data)
        if size:
            rec["width"], rec["height"] = size
        assets.append(rec)
    if not assets:
        return JSONResponse({"status": "error", "error": "no file field in the request"}, status_code=400)
    return JSONResponse({"status": "done", "assets": assets})


@mcp.custom_route("/login", methods=["GET", "POST"])
async def login(request: Request):
    if request.method == "POST":
        return await oauth_provider.login_post(request)
    return await oauth_provider.login_get(request)


# /mcp is protected by the SDK's bearer-auth middleware (OAuth access token or owner token).
app = mcp.streamable_http_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=MCP_PORT, log_level="info", proxy_headers=True)
