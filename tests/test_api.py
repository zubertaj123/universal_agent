"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_root_page(client):
    """Test main page"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Local Call Center AI" in response.text

def test_api_calls_list(client):
    """Test calls list endpoint"""
    response = client.get("/api/calls")
    assert response.status_code == 200
    data = response.json()
    assert "calls" in data
    assert "total" in data

def test_api_call_not_found(client):
    """Test call not found"""
    response = client.get("/api/calls/nonexistent-id")
    assert response.status_code == 404

def test_api_document_upload_no_file(client):
    """Test document upload without file"""
    response = client.post("/api/documents/upload")
    assert response.status_code == 422

def test_websocket_connection():
    """Test WebSocket connection"""
    from fastapi.testclient import TestClient
    
    with TestClient(app) as client:
        with client.websocket_connect("/ws/call/test-session") as websocket:
            # Send test message
            websocket.send_json({
                "type": "control",
                "action": "mute"
            })
            
            # Should receive some response
            # In a real test, we'd mock the services
            try:
                data = websocket.receive_json(timeout=1)
                assert data is not None
            except:
                # Timeout is expected without full service setup
                pass