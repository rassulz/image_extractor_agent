#!/usr/bin/env bash
# One-command demo launcher: builds the CLIP index if needed, then starts
# the FastAPI backend (:8000) and the Vite React frontend (:5173).
set -e

ENV_NAME="${CONDA_ENV:-practical-ai-engineering}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> Building CLIP index if needed (env: $ENV_NAME)"
conda run -n "$ENV_NAME" python scripts/build_index.py

echo "==> Starting FastAPI backend on http://localhost:8000"
conda run -n "$ENV_NAME" uvicorn backend.main:app --port 8000 &
BACK_PID=$!

echo "==> Starting Vite frontend on http://localhost:5173"
( cd frontend && npm run dev ) &
FRONT_PID=$!

trap "kill $BACK_PID $FRONT_PID 2>/dev/null" EXIT
echo "==> Open http://localhost:5173  (Ctrl+C to stop)"
wait
