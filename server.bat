:: This file starts backend and frontend services for local development.
:: Run from project root with: .\server.bat

@echo off
setlocal

echo ===================================================
echo           MediFlow Local Startup
echo ===================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: 1. Start FastAPI backend
echo [1/2] Starting FastAPI Backend on port 8000...
if exist "%ROOT_DIR%venv\Scripts\python.exe" (
    start "MediFlow Backend" cmd /k "cd /d ""%ROOT_DIR%"" && set PYTHONPATH=. && venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
) else (
    start "MediFlow Backend" cmd /k "cd /d ""%ROOT_DIR%"" && set PYTHONPATH=. && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
)
timeout /t 3 /nobreak >nul
echo Backend started in a new window.
echo.

:: 2. Start Vite frontend
echo [2/2] Starting Vite Frontend on port 5173...
start "MediFlow Frontend" cmd /k "cd /d ""%ROOT_DIR%frontend"" && npm run dev"
echo Frontend started in a new window.
echo.

echo ===================================================
echo All services have been launched!
echo - Frontend: http://localhost:5173
echo - Backend API Docs: http://localhost:8000/docs
echo ===================================================
echo.
pause
