Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Nutrify AI Full-Stack Application" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8001 ..." -ForegroundColor Green
Start-Process -FilePath "backend\.venv\Scripts\python.exe" -ArgumentList "backend\run_server.py" -WindowStyle Normal

Write-Host "[2/2] Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Yellow
Set-Location -Path "frontend"
npm run dev

