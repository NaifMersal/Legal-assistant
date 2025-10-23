@echo off
REM Legal Assistant API - Startup Script (Windows)

echo 🏛️  Starting Legal Assistant API...
echo.

REM Check if .env file exists
if not exist .env (
    echo ⚠️  Warning: .env file not found!
    echo Creating .env from .env.example...
    copy .env.example .env
    echo ✓ .env file created. Please edit it and add your GOOGLE_API_KEY
    echo.
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo 📦 Virtual environment not found. Creating one...
    python -m venv venv
    echo ✓ Virtual environment created
    echo.
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt --upgrade --quiet

echo.
echo ✅ Setup complete!
echo.
echo 🚀 Starting FastAPI server...
echo 📍 API will be available at: http://localhost:8000
echo 📚 API docs will be available at: http://localhost:8000/docs
echo.

REM Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
