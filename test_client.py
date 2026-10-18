"""End-to-end check: connect like a real MCP client and run a few tools.
Usage: venv/bin/python test_client.py [base_url] [--full]"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DIR = Path(__file__).parent
BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8765").rstrip("/")
FULL = "--full" in sys.argv
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


async def main():
    async with streamablehttp_client(f"{BASE}/mcp", headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("[tools]", ", ".join(t.name for t in tools.tools))
            show("server_status", await s.call_tool("server_status", {}))
            img = show("text_to_image", await s.call_tool("text_to_image", {
                "prompt": "A cozy Thai street food stall at night, warm lanterns, steam rising from a wok, photorealistic",
                "aspect_ratio": "16:9", "quality": "draft", "wait_seconds": 180}))
            if FULL and img.get("outputs"):
                url = img["outputs"][0]["url"]
                show("image_to_video", await s.call_tool("image_to_video", {
                    "image": url, "prompt": "Steam rises, the cook tosses noodles in the wok, sizzling sounds, street chatter",
                    "duration_seconds": 3, "quality": "draft", "wait_seconds": 300}))
                show("reference_to_video", await s.call_tool("reference_to_video", {
                    "reference_images": [url], "prompt": "A slow cinematic dolly shot through this food stall, crowd ambience",
                    "duration_seconds": 3, "quality": "draft", "wait_seconds": 300}))
                show("text_to_video", await s.call_tool("text_to_video", {
                    "prompt": "Rain falling on a Bangkok street at night, neon reflections, tuk-tuk passing, rain sounds",
                    "duration_seconds": 3, "quality": "draft", "wait_seconds": 300}))
                show("text_to_speech", await s.call_tool("text_to_speech", {
                    "text": "สวัสดีครับ ยินดีต้อนรับสู่สตูดิโอของเรา", "language": "Thai"}))


asyncio.run(main())
