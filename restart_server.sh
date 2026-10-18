#!/bin/bash
# Restart only the MCP server (keeps the current tunnel URL).
DIR=/workspace/h3-mcp
pkill -f "$DIR/server.py" 2>/dev/null || true
sleep 1
set -a
MCP_TOKEN=$(cat "$DIR/token")
PUBLIC_URL=$(cat "$DIR/public_url")
MCP_PORT=8765
[ -f "$DIR/minimax.env" ] && . "$DIR/minimax.env"
set +a
cd "$DIR"
nohup "$DIR/venv/bin/python" "$DIR/server.py" > "$DIR/server.log" 2>&1 &
for i in $(seq 1 20); do curl -fs http://127.0.0.1:8765/health >/dev/null 2>&1 && break; sleep 1; done
if curl -fs http://127.0.0.1:8765/health >/dev/null; then
  echo "server restarted OK"
  echo "MCP URL : $PUBLIC_URL/mcp"
else
  echo "[error] see $DIR/server.log"; tail -20 "$DIR/server.log"
fi
