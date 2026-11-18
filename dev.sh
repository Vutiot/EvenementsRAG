#!/usr/bin/env bash
# Usage:
#   ./dev.sh backend    — start FastAPI on port 8000
#   ./dev.sh frontend   — start Vite on port 5173

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
  backend)
    exec "$ROOT/.venv/bin/uvicorn" src.api.main:app --reload --port 8000
    ;;
  frontend)
    cd "$ROOT/frontend" && exec npm run dev
    ;;
  *)
    echo "Usage: $0 {backend|frontend}"
    exit 1
    ;;
esac
