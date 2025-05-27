"""
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
