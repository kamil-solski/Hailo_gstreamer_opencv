#!/usr/bin/env bash
set -euo pipefail

# Values come from .env via docker-compose.  The :-fallbacks are a last resort
# for running the container directly (without Compose).
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"

# ── Build argument list ────────────────────────────────────────────────────────
ARGS=(
    "Gstreamer/web_app.py"
    "--host" "${HOST}"
    "--port" "${PORT}"
    "--width" "${WIDTH}"
    "--height" "${HEIGHT}"
    "--fps" "${FPS}"
)

# Optional model path passed as first positional argument:
#   docker compose run web-app ./models/yolov8n_face.hef
#   docker compose run web-app ./models/yolov8n-face.onnx
MODEL="${1:-}"
if [[ "${MODEL}" == *.hef ]]; then
    ARGS+=("--hef" "${MODEL}")
elif [[ "${MODEL}" == *.onnx ]]; then
    ARGS+=("--onnx" "${MODEL}")
elif [[ -n "${MODEL}" ]]; then
    echo "ERROR: unrecognised model extension — expected .hef or .onnx (got: ${MODEL})" >&2
    exit 1
fi

# ── Hand off to Python (exec replaces shell so signals reach the app) ─────────
exec python3 "${ARGS[@]}"
