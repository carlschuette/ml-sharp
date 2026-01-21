@echo off
setlocal
set "GREEN=[32m"
set "RED=[31m"
set "NC=[0m"

echo Starting installation...

REM 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed.
    echo Please download and install Python from https://www.python.org/downloads/
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo Found Python.

REM 2. Check for Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed.
    echo Please download and install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo Found Node.js.

REM 3. Set up Python Virtual Environment
if exist ".venv" (
    echo Virtual environment directory already exists.
) else (
    echo Creating virtual environment...
    python -m venv .venv
)

REM 4. Install Python dependencies
echo Installing Python dependencies...
REM Upgrade pip within the venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
)
echo Python dependencies installed successfully.

REM 5. Install Root Node dependencies
echo Installing root Node dependencies...
call npm install

REM 6. Install Frontend Node dependencies
echo Installing frontend Node dependencies...
cd app\frontend
call npm install
cd ..\..

echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo To start the application:
echo 1. Run the following command in the root directory:
echo    npm run dev
echo.
pause
