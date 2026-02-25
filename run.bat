@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM IntuOne startup script — Windows
REM ─────────────────────────────────────────────────────────────────────────────
setlocal

echo === IntuOne — ParallelLines Perception Engine ===
cd /d "%~dp0"

REM Python check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11+ required.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Virtual environment
if not exist ".venv" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Install deps
echo [2/4] Installing dependencies...
pip install -e "." --quiet
pip install aiosqlite --quiet

REM .env setup
if not exist ".env" (
    copy .env.example .env >nul
    echo    Created .env — edit it to configure your database and API keys.
    echo    TIP: For quick start without PostgreSQL, set:
    echo         DATABASE_URL=sqlite+aiosqlite:///./parallellines.db
)

REM Migrations
echo [3/4] Running database migrations...
alembic upgrade head
if %errorlevel% neq 0 (
    echo.
    echo NOTE: If using PostgreSQL, make sure it is running and DATABASE_URL is correct.
    echo For a quick start without PostgreSQL, edit .env and set:
    echo   DATABASE_URL=sqlite+aiosqlite:///./parallellines.db
    echo Then run this script again.
    pause
    exit /b 1
)

echo [4/4] Starting IntuOne...
echo.
echo   Chat UI   : http://localhost:8000/app/chat.html
echo   Dashboard : http://localhost:8000/app/dashboard.html
echo   API docs  : http://localhost:8000/docs
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
