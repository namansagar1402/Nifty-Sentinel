@echo off
REM Double-click to open the Nifty Sentinel dashboard.
REM If it is already running it just opens the browser; otherwise it starts it first.
title Nifty Sentinel
cd /d "%~dp0"
set PORT=8510
set URL=http://localhost:%PORT%

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
    echo Nifty Sentinel is already running. Opening %URL% ...
    start "" "%URL%"
    exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
    set PY=python
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.10+ and run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    set PY=py
)

echo Starting Nifty Sentinel at %URL%
echo Keep this window open while you use the app. Close it to stop the app.
echo.
%PY% -m streamlit run ui/app.py --server.port %PORT% --server.headless false
if errorlevel 1 (
    echo.
    echo The app could not start. If a package is missing, run: pip install -r requirements.txt
    pause
)
