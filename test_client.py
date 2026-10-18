"""End-to-end check: connect like a real MCP client and run a few tools.
Usage: venv/bin/python test_client.py [base_url] [--full] [--voice]"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DIR = Path(__file__).parent
BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8765").rstrip("/")
FULL = "--full" in sys.argv
VOICE = "--voice" in sys.argv
TOKEN = (DIR / "token").read_text().strip()


def show(label, result):
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False)
    except Exception:
        pass
    print(f"[{label}] {text[:600]}")
    try:
        return json.loads("".join(getattr(c, "text", "") for c in result.content))
    except Exception:
        return {}


async def run(s, label, tool, args):
    """Call a GPU tool and keep calling get_job until it finishes."""
    res = show(label, await s.call_tool(tool, args))
    while res.get("job_id") and res.get("status") in ("queued", "running"):
        res = show(f"{label}/get_job", await s.call_tool("get_job", {"job_id": res["job_id"]}))
    return res


async def main():
    async with streamablehttp_client(f"{BASE}/mcp", headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("[tools]", ", ".join(t.name for t in tools.tools))
            show("server_status", await s.call_tool("server_status", {}))
            img = await run(s, "text_to_image", "text_to_image", {
                "prompt": ("A friendly Thai woman street food cook in a red apron facing the camera, medium close-up, "
                           "night market stall, warm lanterns, steam from a wok, photorealistic") if VOICE else
                          "A cozy Thai street food stall at night, warm lanterns, steam rising from a wok, photorealistic",
                "aspect_ratio": "16:9", "quality": "standard" if VOICE else "draft", "wait_seconds": 45})
            if VOICE and img.get("outputs"):
                line = "วันนี้ผัดไทยร้อน ๆ พร้อมเสิร์ฟแล้วค่ะ ใครหิวมาทางนี้เลย"
                voices = show("list_voices", await s.call_tool("list_voices", {"search": "Thai", "limit": 5}))
                ids = [v["voice_id"] for v in voices.get("voices", []) if v.get("voice_id")]
                vid = next((i for i in ids if "female" in i.lower()), ids[0] if ids else None)
                tts = show("text_to_speech", await s.call_tool("text_to_speech", {
                    "text": line, "language": "Thai", **({"voice_id": vid} if vid else {})}))
                if tts.get("outputs"):
                    speech = tts["outputs"][0]
                    await run(s, "image_to_video+lipsync", "image_to_video", {
                        "image": img["outputs"][0]["url"], "speech_audio": speech["url"],
                        "duration_seconds": round(speech.get("duration_seconds", 4) + 1, 1),
                        "prompt": "Close-up of the street food cook looking at the camera, smiling and "
                                  f"speaking Thai: \"{line}\". Wok sizzling softly in the background.",
                        "quality": "standard"})
            if FULL and img.get("outputs"):
                url = img["outputs"][0]["url"]
                await run(s, "image_to_video", "image_to_video", {
                    "image": url, "prompt": "Steam rises, the cook tosses noodles in the wok, sizzling sounds, street chatter",
                    "duration_seconds": 3, "quality": "draft", })
                await run(s, "reference_to_video", "reference_to_video", {
                    "reference_images": [url], "prompt": "A slow cinematic dolly shot through this food stall, crowd ambience",
                    "duration_seconds": 3, "quality": "draft", })
                await run(s, "text_to_video", "text_to_video", {
                    "prompt": "Rain falling on a Bangkok street at night, neon reflections, tuk-tuk passing, rain sounds",
                    "duration_seconds": 3, "quality": "draft", })
                show("text_to_speech", await s.call_tool("text_to_speech", {
                    "text": "สวัสดีครับ ยินดีต้อนรับสู่สตูดิโอของเรา", "language": "Thai"}))


asyncio.run(main())
