#!/bin/bash
# =============================================================================
#  Creative Studio — Vast.ai provisioning script
#  Runs automatically on first boot of the official Vast "ComfyUI" image when
#  this file's URL is set as PROVISIONING_SCRIPT in the template.
#
#  Template environment variables:
#    PROVISIONING_SCRIPT  URL of this file
#    STUDIO_CODE_URL      base URL that holds server.py, oauth.py, ... (same place as this file)
#    MINIMAX_API_KEY      (optional) enables text_to_speech / voice tools
#    MINIMAX_VOICE_ID     (optional) default voice, e.g. Thai_female_1_sample1
#    STUDIO_TOKEN         (optional) fixed access code; otherwise a random one is created
#    STUDIO_AUTOSTART     1 = start the MCP server + public tunnel when ready (default 1)
#
#  Log: /workspace/h3-mcp/provision.log
# =============================================================================
set -uo pipefail
D=/workspace/h3-mcp
M=/workspace/ComfyUI/models
mkdir -p "$D" "$M"/{diffusion_models,text_encoders,vae,loras}
exec > >(tee -a "$D/provision.log") 2>&1
echo "[provision] start $(date -u)"

STUDIO_CODE_URL="${STUDIO_CODE_URL:-}"
if [ -z "$STUDIO_CODE_URL" ] && [ -n "${PROVISIONING_SCRIPT:-}" ]; then
  STUDIO_CODE_URL="${PROVISIONING_SCRIPT%/*}"
fi

# ---------------------------------------------------------------------------
# 1) Models (parallel, resumable, skipped if already complete)
# ---------------------------------------------------------------------------
H=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main
Z=https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files
MODELS=(
  "$H/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|$M/diffusion_models"
  "$H/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors|$M/diffusion_models"
  "$H/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|$M/text_encoders"
  "$H/vae/minimax_h3_video_vae_fp16.safetensors|$M/vae"
  "$H/vae/minimax_h3_audio_vae_fp32.safetensors|$M/vae"
  "$H/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors|$M/loras"
  "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors|$M/loras"
  "$Z/diffusion_models/z_image_turbo_int8_convrot.safetensors|$M/diffusion_models"
  "$Z/text_encoders/qwen_3_4b_fp8_mixed.safetensors|$M/text_encoders"
  "$Z/vae/ae.safetensors|$M/vae"
)

fetch_model() {
  local url="${1%%|*}" dest="${1##*|}"
  local name; name="$(basename "$url")"
  local marker="$dest/.$name.done"
  [ -f "$marker" ] && { echo "[model] ok   $name"; return 0; }
  for attempt in 1 2 3; do
    if wget -q -c -O "$dest/$name" "$url"; then
      touch "$marker"; echo "[model] done $name"; return 0
    fi
    echo "[model] retry $attempt $name"; sleep 5
  done
  echo "[model] FAILED $name"; return 1
}

FAIL=0
pids=()
for m in "${MODELS[@]}"; do fetch_model "$m" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || FAIL=1; done
[ "$FAIL" = 1 ] && echo "[provision] WARNING: some models failed; re-run: bash $D/provision.sh"

# ---------------------------------------------------------------------------
# 2) MCP server code
# ---------------------------------------------------------------------------
if [ -n "$STUDIO_CODE_URL" ]; then
  for f in server.py oauth.py workflows.py start.sh restart_server.sh test_client.py provision.sh; do
    if curl -fsSL "$STUDIO_CODE_URL/$f" -o "$D/$f.new"; then mv "$D/$f.new" "$D/$f"; else
      echo "[code] could not download $f"; rm -f "$D/$f.new"; fi
  done
  chmod +x "$D"/*.sh 2>/dev/null || true
else
  echo "[code] STUDIO_CODE_URL not set; expecting code already in $D"
fi

# ---------------------------------------------------------------------------
# 3) Secrets from template env (never printed)
# ---------------------------------------------------------------------------
if [ -n "${MINIMAX_API_KEY:-}" ]; then
  { echo "MINIMAX_API_KEY=$MINIMAX_API_KEY"
    [ -n "${MINIMAX_VOICE_ID:-}" ] && echo "MINIMAX_VOICE_ID=$MINIMAX_VOICE_ID"
    [ -n "${MINIMAX_API_BASE:-}" ] && echo "MINIMAX_API_BASE=$MINIMAX_API_BASE"
  } > "$D/minimax.env"
  chmod 600 "$D/minimax.env"
fi
if [ -n "${STUDIO_TOKEN:-}" ]; then
  printf '%s\n' "$STUDIO_TOKEN" > "$D/token"; chmod 600 "$D/token"
fi

# ---------------------------------------------------------------------------
# 4) Python env for the MCP server
# ---------------------------------------------------------------------------
if [ ! -x "$D/venv/bin/python" ]; then
  python3 -m venv "$D/venv" && "$D/venv/bin/pip" install -q --upgrade pip &&
  "$D/venv/bin/pip" install -q "mcp>=1.12,<2" httpx pillow uvicorn
fi

# ---------------------------------------------------------------------------
# 5) Start (waits for ComfyUI to be up)
# ---------------------------------------------------------------------------
# Runs in the background so provisioning can finish and ComfyUI can start.
if [ "${STUDIO_AUTOSTART:-1}" = "1" ] && [ -f "$D/start.sh" ] && [ "$FAIL" = 0 ]; then
  setsid nohup bash -c "
    for i in \$(seq 1 180); do curl -fs http://127.0.0.1:18188/system_stats >/dev/null 2>&1 && break; sleep 5; done
    bash $D/start.sh
  " > "$D/start.out" 2>&1 < /dev/null &
  echo "[provision] MCP server starting in background — see $D/start.out"
fi
touch "$D/.provisioned"
echo "[provision] finished $(date -u)"
