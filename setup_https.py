#!/usr/bin/env python3
"""
Setup HTTPS for Local Call Center AI
"""
import subprocess
import os
from pathlib import Path

def create_self_signed_cert():
    """Create self-signed SSL certificate"""
    cert_dir = Path("certs")
    cert_dir.mkdir(exist_ok=True)
    
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    if cert_file.exists() and key_file.exists():
        print("✅ SSL certificates already exist")
        return str(cert_file), str(key_file)
    
    print("🔐 Creating self-signed SSL certificate...")
    
    # Create certificate
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", 
        "-keyout", str(key_file), "-out", str(cert_file),
        "-days", "365", "-nodes", "-subj", 
        "/C=US/ST=Local/L=Local/O=CallCenter/CN=localhost"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created certificate: {cert_file}")
        print(f"✅ Created private key: {key_file}")
        return str(cert_file), str(key_file)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create certificate: {e}")
        return None, None
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL:")
        print("  macOS: brew install openssl")
        print("  Ubuntu: sudo apt install openssl")
        return None, None

def update_main_for_https():
    """Update main.py to support HTTPS"""
    https_main_content = '''"""
Main FastAPI application with HTTPS support
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import ssl
from pathlib import Path

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import api_router
from app.api.websocket import websocket_router
from app.services.speech_service import SpeechService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    logger.info("Starting Call Center AI...")
    
    # Initialize database
    await init_db()
    
    # Initialize speech service
    speech_service = SpeechService()
    await speech_service.initialize()
    
    # Store in app state
    app.state.speech_service = speech_service
    
    yield
    
    # Cleanup
    logger.info("Shutting down Call Center AI...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:8443", "https://127.0.0.1:8443"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(api_router, prefix="/api")
app.include_router(websocket_router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Call Center AI"}
    )

@app.get("/call", response_class=HTMLResponse)
async def call_interface(request: Request):
    """Serve the call interface page"""
    return templates.TemplateResponse(
        "call_interface.html",
        {"request": request, "title": "Voice Call - Call Center AI"}
    )

@app.get("/chat", response_class=HTMLResponse) 
async def chat_interface(request: Request):
    """Serve the chat interface page"""
    return HTMLResponse(content="""
    <html>
    <head><title>Text Chat - Call Center AI</title></head>
    <body>
        <h1>💬 Text Chat Interface</h1>
        <p>Text chat functionality coming soon!</p>
        <a href="/">← Back to Home</a>
    </body>
    </html>
    """)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_interface(request: Request):
    """Serve the dashboard page"""
    return HTMLResponse(content="""
    <html>
    <head><title>Dashboard - Call Center AI</title></head>
    <body>
        <h1>📊 Call Center Dashboard</h1>
        <p>Dashboard functionality coming soon!</p>
        <ul>
            <li><a href="/api/calls">View API Calls</a></li>
            <li><a href="/docs">API Documentation</a></li>
        </ul>
        <a href="/">← Back to Home</a>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "https": True
    }

def run_https_server():
    """Run server with HTTPS on port 8000"""
    cert_file = Path("certs/cert.pem")
    key_file = Path("certs/key.pem")
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ SSL certificates not found. Run setup_https.py first.")
        return
    
    print("🚀 Starting HTTPS server on port 8000...")
    print("📱 Visit: https://localhost:8000")
    print("⚠️  You'll need to accept the self-signed certificate warning")
    
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(str(cert_file), str(key_file))
    
    uvicorn.run(
        "app.main:app",  # Use the original main.py
        host="0.0.0.0",
        port=8000,  # Use standard port 8000 for HTTPS
        ssl_keyfile=str(key_file),
        ssl_certfile=str(cert_file),
        reload=False,  # Disable reload with SSL
        log_level="info"
    )

if __name__ == "__main__":
    run_https_server()
'''
    
    # Save HTTPS version
    with open("app/main_https.py", "w") as f:
        f.write(https_main_content)
    
    print("✅ Created app/main_https.py")

def main():
    """Setup HTTPS for the application"""
    print("🔐 Setting up HTTPS for Local Call Center AI")
    print("=" * 50)
    
    # Create SSL certificate
    cert_file, key_file = create_self_signed_cert()
    
    if cert_file and key_file:
        # Update main.py for HTTPS
        update_main_for_https()
        
        print("\n🎉 HTTPS setup complete!")
        print("\nNext steps:")
        print("1. Run the HTTPS server:")
        print("   python app/main_https.py")
        print("\n2. Visit: https://localhost:8443")
        print("\n3. Accept the security warning (click 'Advanced' → 'Proceed to localhost')")
        print("\n4. Try the voice call - microphone should work!")
        
        print("\n💡 Alternative: Use regular HTTP with Chrome flag:")
        print("   chrome --unsafely-treat-insecure-origin-as-secure=http://localhost:8000")
        
    else:
        print("\n❌ Failed to setup HTTPS")
        print("Try the Chrome flag method instead:")
        print("chrome --unsafely-treat-insecure-origin-as-secure=http://localhost:8000 --user-data-dir=/tmp/chrome_dev")

if __name__ == "__main__":
    main()