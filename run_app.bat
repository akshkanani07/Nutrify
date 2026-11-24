@echo off
echo ========================================================
echo   Starting Nutrify AI Full-Stack Application
echo ========================================================
echo.
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8001 ...
start "Nutrify AI Backend (FastAPI)" cmd /k "title Nutrify AI Backend && backend\.venv\Scripts\python.exe backend\run_server.py"

echo [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "Nutrify AI Frontend (Next.js)" cmd /k "title Nutrify AI Frontend && cd frontend && npm run dev"

echo.
echo ========================================================
echo   Services Launched!
echo   Frontend: http://localhost:3000/analyze
echo   Backend:  http://127.0.0.1:8001/docs
echo ========================================================

