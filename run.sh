#!/usr/bin/env bash
# Start Trivexa: builds the UI if it's missing or stale, syncs Python deps, and serves everything on one port.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${TRIVEXA_HOST:-127.0.0.1}"
PORT="${TRIVEXA_PORT:-8000}"

if [ ! -d frontend/dist ] || [ -n "$(find frontend/src frontend/index.html -newer frontend/dist -print -quit)" ]; then
  echo "Building frontend…"
  (cd frontend && npm install --silent && npm run build)
fi

# The "texture" extra (MV-Adapter's texture baker) compiles nvdiffrast, which needs the CUDA toolkit.
# If that fails, run without it; Texture then uses the lean baker (see settings.toml).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
if ! (cd backend && uv sync --quiet --extra texture); then
  echo "Could not install the texture extra; continuing without it (lean texture baker only)." >&2
  (cd backend && uv sync --quiet)
fi

echo "Trivexa → http://$HOST:$PORT"
exec backend/.venv/bin/uvicorn trivexa.main:app --app-dir backend --host "$HOST" --port "$PORT"
