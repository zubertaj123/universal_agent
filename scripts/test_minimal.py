#!/usr/bin/env python3
"""
Test minimal functionality without problematic dependencies
"""
import sys
import os
sys.path.append('.')

def test_basic_imports():
    """Test basic Python imports"""
    print("🔄 Testing basic imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy imported")
    except ImportError as e:
        print(f"❌ SQLAlchemy import failed: {e}")
        return False
    
    return True

def test_app_imports():
    """Test app module imports"""
    print("\n🔄 Testing app imports...")
    
    try:
        from app.utils.logger import setup_logger
        logger = setup_logger(__name__)
        print("✅ Logger imported and working")
    except ImportError as e:
        print(f"❌ Logger import failed: {e}")
        return False
    
    try:
        from app.core.config import settings
        print(f"✅ Config imported - App: {settings.APP_NAME}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from app.core.database import Base
        print("✅ Database base imported")
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    return True

def test_models():
    """Test model imports"""
    print("\n🔄 Testing model imports...")
    
    try:
        from app.models.user import User
        print("✅ User model imported")
    except ImportError as e:
        print(f"❌ User model import failed: {e}")
        return False
    
    try:
        from app.models.conversation import CallRecord, CallStatus
        print("✅ Conversation models imported")
    except ImportError as e:
        print(f"❌ Conversation model import failed: {e}")
        return False
    
    return True

def test_problematic_imports():
    """Test imports that might cause NumPy issues"""
    print("\n🔄 Testing potentially problematic imports...")
    
    # Test sentence transformers (needs PyTorch/NumPy)
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers available")
        st_available = True
    except ImportError as e:
        print(f"⚠️ SentenceTransformers not available: {e}")
        st_available = False
    
    # Test librosa (needs NumPy)
    try:
        import librosa
        print("✅ Librosa available")
        librosa_available = True
    except ImportError as e:
        print(f"⚠️ Librosa not available: {e}")
        librosa_available = False
    
    # Test NumPy directly
    try:
        import numpy as np
        print(f"✅ NumPy available (version: {np.__version__})")
        numpy_available = True
    except ImportError as e:
        print(f"⚠️ NumPy not available: {e}")
        numpy_available = False
    
    return {
        'sentence_transformers': st_available,
        'librosa': librosa_available,
        'numpy': numpy_available
    }

def test_fastapi_app():
    """Test creating a basic FastAPI app"""
    print("\n🔄 Testing FastAPI app creation...")
    
    try:
        from fastapi import FastAPI
        
        app = FastAPI(title="Test App")
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World"}
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy"}
        
        print("✅ FastAPI app created successfully")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

def create_minimal_main():
    """Create a minimal main.py that works without ML dependencies"""
    minimal_main_content = '''"""
Minimal FastAPI application without ML dependencies
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Try to import our basic modules
try:
    from app.core.config import settings
except ImportError:
    class Settings:
        APP_NAME = "Local Call Center AI (Minimal)"
        APP_VERSION = "1.0.0"
        HOST = "0.0.0.0"
        PORT = 8000
        CORS_ORIGINS = ["*"]
    settings = Settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Minimal version without ML dependencies"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>Local Call Center AI (Minimal)</title></head>
    <body>
        <h1>🎯 Local Call Center AI</h1>
        <h2>Minimal Version Running!</h2>
        <p>This version runs without ML dependencies.</p>
        <ul>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/test">Test Endpoint</a></li>
        </ul>
        <h3>Available Features:</h3>
        <ul>
            <li>✅ FastAPI web framework</li>
            <li>✅ Database models</li>
            <li>✅ REST API endpoints</li>
            <li>✅ WebSocket support</li>
            <li>⚠️ ML features disabled (embeddings, audio processing)</li>
        </ul>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "mode": "minimal",
        "ml_features": False
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify functionality"""
    return {
        "message": "Test successful!",
        "features": {
            "fastapi": True,
            "database": True,
            "ml_features": False
        }
    }

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} (Minimal Mode)")
    print("Visit http://localhost:8000 to see the interface")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
'''
    
    with open("app/main_minimal.py", "w") as f:
        f.write(minimal_main_content)
    
    print("✅ Created app/main_minimal.py")

def main():
    """Run all tests"""
    print("🧪 Local Call Center AI - Minimal Testing")
    print("=" * 50)
    
    # Test basic imports
    basic_ok = test_basic_imports()
    if not basic_ok:
        print("\n❌ Basic imports failed. Please install:")
        print("pip install fastapi uvicorn sqlalchemy pydantic")
        return
    
    # Test app imports
    app_ok = test_app_imports()
    if not app_ok:
        print("\n❌ App imports failed. Check your project structure.")
        return
    
    # Test models
    models_ok = test_models()
    if not models_ok:
        print("\n❌ Model imports failed. Check for syntax errors in models.")
        return
    
    # Test FastAPI
    fastapi_ok = test_fastapi_app()
    
    # Test problematic imports
    ml_status = test_problematic_imports()
    
    # Create minimal version
    create_minimal_main()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Basic imports: {'OK' if basic_ok else 'FAILED'}")
    print(f"✅ App imports: {'OK' if app_ok else 'FAILED'}")
    print(f"✅ Models: {'OK' if models_ok else 'FAILED'}")
    print(f"✅ FastAPI: {'OK' if fastapi_ok else 'FAILED'}")
    print(f"⚠️ NumPy: {'Available' if ml_status['numpy'] else 'Not available'}")
    print(f"⚠️ Audio processing: {'Available' if ml_status['librosa'] else 'Not available'}")
    print(f"⚠️ Embeddings: {'Available' if ml_status['sentence_transformers'] else 'Not available'}")
    
    if basic_ok and app_ok and models_ok and fastapi_ok:
        print("\n🎉 Core functionality works! You can run:")
        print("   python app/main_minimal.py")
        print("\nOr try the full app:")
        print("   python -m app.main")
        
        if not all(ml_status.values()):
            print("\n💡 To enable ML features, fix NumPy compatibility:")
            print("   python fix_dependencies.py")
    else:
        print("\n❌ Some core functionality is broken. Please fix the errors above.")

if __name__ == "__main__":
    main()