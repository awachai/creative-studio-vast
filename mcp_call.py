"""Call any Creative Studio tool from the server terminal (through the local MCP server).
Usage: venv/bin/python mcp_call.py <tool> '<json args>'
GPU jobs are followed with get_job until they finish. Example:
  venv/bin/python mcp_call.py image_to_video '{"image": "asset:...", "prompt": "...", "quality": "high"}'"""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TOKEN = (Path(__file__).parent / "token").read_text().strip()


def parse(res):
    text = "".join(getattr(c, "text", "") for c in res.content)
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


async def main(tool, args):
    t0 = time.time()
    async with streamablehttp_client("http://127.0.0.1:8765/mcp",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            d = parse(await s.call_tool(tool, args))
            while d.get("job_id") and d.get("status") in ("queued", "running"):
                print(f"[{int(time.time() - t0)}s] {d.get('status')} {d.get('queue_position', '')}", flush=True)
                d = parse(await s.call_tool("get_job", {"job_id": d["job_id"]}))
            print(json.dumps(d, ensure_ascii=False, indent=1))
            print(f"[done in {int(time.time() - t0)}s]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
