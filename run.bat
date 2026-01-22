@echo off
setlocal

echo Starting Gaussian Splat Application...

REM Check if .venv exists
if not exist ".venv" (
    echo [31mError: .venv directory not found.[0m
    echo [33mPlease run install.bat first to set up the project.[0m
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "node_modules" (
    echo [31mError: node_modules directory not found.[0m
    echo [33mPlease run install.bat first to install dependencies.[0m
    pause
    exit /b 1
)

REM Start the application
echo Launching development server...
call npm run dev

if %errorlevel% neq 0 (
    echo [31mApplication exited with an error.[0m
    pause
)
