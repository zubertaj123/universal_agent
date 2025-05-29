#!/bin/bash
# PRODUCTION STARTUP SCRIPT - NO test interference

echo "🚀 Starting AI Insurance Voice Agent (PRODUCTION MODE)"
echo "========================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for API keys
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Please:"
    echo "   cp .env.production .env"
    echo "   Edit .env and add your API keys"
    exit 1
fi

# Kill any existing processes
echo "🛑 Stopping any existing processes..."
pkill -f "uvicorn" 2>/dev/null || true
sleep 2

# Clear cache directories
echo "🧹 Clearing caches..."
rm -rf data/cache/tts/* 2>/dev/null || true
rm -rf __pycache__ 2>/dev/null || true

# Create required directories
echo "📁 Creating directories..."
mkdir -p data/db data/cache/tts data/audio/recordings

# Start the application
echo "🎯 Starting PRODUCTION server..."
echo "Visit: http://localhost:8000"
echo "For HTTPS: https://localhost:8000 (if certificates exist)"
echo ""

python -m app.main
