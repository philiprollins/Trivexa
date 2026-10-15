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

# The "texture" extra (MV-Adapter's texture baker) compiles nvdiffrast and the "pixal3d" extra compiles TRELLIS.2's
# CUDA extensions, which needs the CUDA toolkit. Those pin C++17, which torch's headers reject: CXX and
# NVCC_APPEND_FLAGS build them as C++20. Whatever fails to install is left out (lean texture baker, no Pixal3D).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX="$PWD/backend/tools/cxx20" NVCC_APPEND_FLAGS="-std=c++20"
if ! (cd backend && uv sync --quiet --extra texture --extra pixal3d); then
  echo "Could not install the texture + pixal3d extras; trying texture only." >&2
  if ! (cd backend && uv sync --quiet --extra texture); then
    echo "Could not install the texture extra; continuing without it (lean texture baker only)." >&2
    (cd backend && uv sync --quiet)
  fi
fi
unset CXX NVCC_APPEND_FLAGS

echo "Trivexa → http://$HOST:$PORT"
exec backend/.venv/bin/uvicorn trivexa.main:app --app-dir backend --host "$HOST" --port "$PORT"
