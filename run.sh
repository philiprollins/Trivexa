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

(cd backend && uv sync --quiet)

echo "Trivexa → http://$HOST:$PORT"
exec backend/.venv/bin/uvicorn trivexa.main:app --app-dir backend --host "$HOST" --port "$PORT"
