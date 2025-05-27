"""
API module initialization
"""
from app.api.routes import api_router
from app.api.websocket import websocket_router

__all__ = ["api_router", "websocket_router"]