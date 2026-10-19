"""Try MiniMax voices from the terminal (through the local MCP server).
Usage: venv/bin/python tts_try.py '<json args for text_to_speech>' ['<json args>' ...]
Example:
  venv/bin/python tts_try.py '{"text": "สวัสดีค่ะ", "voice_id": "Thai_female_1_sample1", "pitch": 3}'
Each argument is one voice to try; prints a link per voice."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TOKEN = (Path(__file__).parent / "token").read_text().strip()


async def main(jobs):
    async with streamablehttp_client("http://127.0.0.1:8765/mcp",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for i, args in enumerate(jobs, 1):
                args.setdefault("language", "Thai")
                res = await s.call_tool("text_to_speech", args)
                d = json.loads("".join(getattr(c, "text", "") for c in res.content))
                label = args.pop("label", None) or f"#{i}"
                print(label, "->", d["outputs"][0]["url"] if d.get("outputs") else d)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main([json.loads(a) for a in sys.argv[1:]]))
