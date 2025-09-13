@echo off
REM AnComicsViewer Startup Script for Windows

echo 🚀 Starting AnComicsViewer...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements_standalone.txt

REM Launch application
echo ✅ Launching AnComicsViewer...
python main.py %*

pause
