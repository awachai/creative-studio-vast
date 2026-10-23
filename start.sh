#!/bin/bash
# Start (or restart) the Creative Studio MCP server + a public Cloudflare tunnel.
# Usage: bash /workspace/h3-mcp/start.sh
set -e
DIR=/workspace/h3-mcp
PORT=${MCP_PORT:-8765}
cd "$DIR"

# 1) Python environment (separate from ComfyUI's)
if [ ! -x "$DIR/venv/bin/python" ]; then
  echo "[setup] creating venv..."
  python3 -m venv "$DIR/venv"
  "$DIR/venv/bin/pip" install -q --upgrade pip
  "$DIR/venv/bin/pip" install -q "mcp>=1.12,<2" httpx pillow uvicorn imageio-ffmpeg
fi

# 2) Access token (created once, kept private)
if [ ! -s "$DIR/token" ]; then
  "$DIR/venv/bin/python" -c "import secrets; print(secrets.token_urlsafe(32))" > "$DIR/token"
  chmod 600 "$DIR/token"
  echo "[setup] new access token created at $DIR/token"
fi

# 3) cloudflared
CF=$(command -v cloudflared || true)
if [ -z "$CF" ]; then
  CF="$DIR/cloudflared"
  if [ ! -x "$CF" ]; then
    echo "[setup] downloading cloudflared..."
    curl -fsSL -o "$CF" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CF"
  fi
fi

# 4) Stop previous instances
pkill -f "$DIR/server.py" 2>/dev/null || true
pkill -f "tunnel --no-autoupdate --url http://127.0.0.1:$PORT" 2>/dev/null || true
sleep 1

# 5) Tunnel first, so we know the public URL
: > "$DIR/tunnel.log"
nohup "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$DIR/tunnel.log" 2>&1 &
URL=""
for i in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$DIR/tunnel.log" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
if [ -z "$URL" ]; then echo "[error] tunnel did not start; see $DIR/tunnel.log"; exit 1; fi
echo "$URL" > "$DIR/public_url"

# 6) MCP server
set -a
MCP_TOKEN=$(cat "$DIR/token")
PUBLIC_URL="$URL"
MCP_PORT="$PORT"
[ -f "$DIR/minimax.env" ] && . "$DIR/minimax.env"
set +a
nohup "$DIR/venv/bin/python" "$DIR/server.py" > "$DIR/server.log" 2>&1 &
for i in $(seq 1 20); do
  curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fs "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[error] server did not start; see $DIR/server.log"; tail -20 "$DIR/server.log"; exit 1; }

echo ""
echo "=================================================="
echo " MCP server is running"
echo " MCP URL : $URL/mcp"
echo " Token   : run   cat $DIR/token"
if [ -n "$MINIMAX_API_KEY" ]; then echo " Speech  : enabled"; else echo " Speech  : disabled (add MINIMAX_API_KEY to $DIR/minimax.env)"; fi
echo "=================================================="
