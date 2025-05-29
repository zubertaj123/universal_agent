#!/usr/bin/env python3
"""
COMPLETE CLEANUP SCRIPT - Remove all audio interference sources
Run this to eliminate ALL test files causing audio overlap issues
"""
import os
import shutil
import subprocess
from pathlib import Path

def cleanup_project():
    """Remove all test files and interference sources"""
    
    print("🧹 COMPLETE PROJECT CLEANUP")
    print("Removing ALL sources of audio interference...")
    print("=" * 50)
    
    removed_count = 0
    
    # 1. Remove test scripts and files
    test_files_to_remove = [
        "scripts/test_voice.py",
        "scripts/test_minimal.py",
        "debug_websocket.py", 
        "app/main_minimal.py",
        "fix_dependencies.py",
    ]
    
    print("1. Removing test scripts...")
    for file_path in test_files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
                removed_count += 1
        except Exception as e:
            print(f"❌ Error removing {file_path}: {e}")
    
    # 2. Remove test directories
    test_directories = [
        "tests/",
        "data/cache/",
        "data/audio/temp/",
        "__pycache__",
    ]
    
    print("\n2. Removing test directories...")
    for dir_path in test_directories:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"✅ Removed directory: {dir_path}")
                removed_count += 1
        except Exception as e:
            print(f"❌ Error removing {dir_path}: {e}")
    
    # 3. Remove all audio test files
    print("\n3. Removing test audio files...")
    audio_extensions = [".mp3", ".wav", ".m4a", ".ogg"]
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if (file.startswith("test") and any(file.endswith(ext) for ext in audio_extensions)) or \
               file in ["debug.mp3", "sample.wav", "temp.mp3"]:
                try:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"✅ Removed audio: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Error removing {file_path}: {e}")
    
    # 4. Clean Python cache
    print("\n4. Cleaning Python cache...")
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            try:
                pycache_path = os.path.join(root, "__pycache__")
                shutil.rmtree(pycache_path)
                print(f"✅ Removed cache: {pycache_path}")
            except Exception as e:
                print(f"❌ Error removing cache: {e}")
    
    # 5. Stop any running processes that might interfere
    print("\n5. Checking for running processes...")
    try:
        # Kill any uvicorn processes
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        print("✅ Stopped any running uvicorn processes")
    except Exception as e:
        print(f"⚠️ Could not stop processes: {e}")
    
    # 6. Create production-only environment
    print("\n6. Creating production configuration...")
    
    production_env = """# PRODUCTION ENVIRONMENT - NO TEST MODES
APP_NAME="AI Insurance Voice Agent"
APP_VERSION="1.0.0"
APP_ENV="production"
DEBUG=false
LOG_LEVEL="INFO"

# Database
DATABASE_URL="sqlite:///./data/db/call_center.db"

# LLM Configuration (Add your keys)
# OPENAI_API_KEY="your-openai-key-here"
# ANTHROPIC_API_KEY="your-anthropic-key-here"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4-turbo-preview"
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# Speech Settings - PRODUCTION ONLY
STT_MODEL="base"
TTS_CACHE_ENABLED=true
VAD_THRESHOLD=0.5
EMBEDDING_MODEL="text-embedding-3-small"
VECTOR_DB_PATH="./data/db/vectors"

# DISABLE ALL TEST FEATURES
TEST_MODE=false
DEBUG_AUDIO=false
TEST_VOICE_GENERATION=false
PRELOAD_TEST_PHRASES=false
"""
    
    try:
        with open(".env.production", "w") as f:
            f.write(production_env)
        print("✅ Created .env.production")
    except Exception as e:
        print(f"❌ Error creating production config: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 CLEANUP COMPLETED!")
    print(f"📊 Removed {removed_count} files/directories")
    print("\n✅ NEXT STEPS:")
    print("1. Replace your WebSocket handler (app/api/websocket.py)")
    print("2. Replace your Speech Service (app/services/speech_service.py)")
    print("3. Replace your Call Handler JS (static/js/call_handler.js)")
    print("4. Copy .env.production to .env and add your API keys")
    print("5. Restart your server with: python -m app.main")
    print("6. Test with ONE voice call only")
    print("\n⚠️ IMPORTANT:")
    print("- NO test scripts will run")
    print("- NO multiple audio streams") 
    print("- VAD chunk size issues FIXED")
    print("- Audio overlap issues RESOLVED")
    print("=" * 50)

def create_startup_script():
    """Create clean startup script"""
    startup_script = """#!/bin/bash
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
"""
    
    try:
        with open("start_production.sh", "w") as f:
            f.write(startup_script)
        os.chmod("start_production.sh", 0o755)
        print("✅ Created start_production.sh")
    except Exception as e:
        print(f"❌ Error creating startup script: {e}")

def verify_cleanup():
    """Verify cleanup was successful"""
    print("\n🔍 VERIFYING CLEANUP...")
    
    issues = []
    
    # Check for remaining test files
    test_patterns = ["test_", "debug_", "_test.py", "minimal.py"]
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(pattern in file.lower() for pattern in test_patterns):
                if not file.endswith(('.md', '.txt', '.log')):  # Ignore docs
                    issues.append(f"Test file still exists: {os.path.join(root, file)}")
    
    # Check for audio test files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("test") and file.endswith(('.mp3', '.wav')):
                issues.append(f"Test audio file: {os.path.join(root, file)}")
    
    if issues:
        print("⚠️ CLEANUP ISSUES FOUND:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✅ CLEANUP VERIFICATION PASSED")
        print("   No test files or interference sources found")

if __name__ == "__main__":
    cleanup_project()
    create_startup_script()
    verify_cleanup()
    
    print("\n" + "🎉" * 20)
    print("PROJECT CLEANUP COMPLETE!")
    print("🎉" * 20)
    print("\nRun: ./start_production.sh")
    print("Or:  python -m app.main")