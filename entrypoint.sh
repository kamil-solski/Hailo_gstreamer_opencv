#!/usr/bin/env bash
set -euo pipefail

ARGS=("Gstreamer/web_app.py")

# Optional model path passed as first positional argument:
#   docker compose run web-app /models/yolov8n_face.hef
#   docker compose run web-app /models/yolov8n-face.onnx
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
