#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# IntuOne startup script (Linux / macOS)
# ─────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== IntuOne — ParallelLines Perception Engine ==="

# Python check
if ! command -v python3 &>/dev/null; then
  echo "ERROR: Python 3.11+ required. Install from https://python.org"
  exit 1
fi

# Virtual env (optional but recommended)
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate 2>/dev/null || true

# Install dependencies
echo "[2/4] Installing dependencies..."
pip install -e "." --quiet

# .env setup
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "  → Created .env from .env.example. Edit it to add your API keys."
fi

# Database choice
source .env 2>/dev/null || true
if [[ "${DATABASE_URL:-}" == *"sqlite"* ]]; then
  echo "[3/4] Using SQLite database (no PostgreSQL required)."
  pip install aiosqlite --quiet
else
  echo "[3/4] Using PostgreSQL — make sure it is running and DATABASE_URL is set in .env"
fi

# Run migrations
echo "[4/4] Running database migrations..."
alembic upgrade head

echo ""
echo "✓ IntuOne is starting..."
echo "  Chat UI  → http://localhost:8000/app/chat.html"
echo "  Dashboard → http://localhost:8000/app/dashboard.html"
echo "  API docs  → http://localhost:8000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
