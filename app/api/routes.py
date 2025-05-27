"""
REST API routes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime

from app.core.database import get_session
from app.models.conversation import CallRecord, CallStatus
from app.services.embedding_service import EmbeddingService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

api_router = APIRouter()

@api_router.get("/calls")
async def get_calls(
    skip: int = 0,
    limit: int = 100,
    status: Optional[CallStatus] = None,
    session: AsyncSession = Depends(get_session)
):
    """Get list of calls"""
    # Implement database query
    return {
        "calls": [],
        "total": 0,
        "skip": skip,
        "limit": limit
    }

@api_router.get("/calls/{call_id}")
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get specific call details"""
    # Implement database query
    return {
        "call_id": call_id,
        "status": "completed",
        "duration": 300,
        "transcript": []
    }

@api_router.post("/calls/{call_id}/transcript")
async def update_transcript(
    call_id: str,
    transcript: dict,
    session: AsyncSession = Depends(get_session)
):
    """Update call transcript"""
    # Implement database update
    return {"success": True}

@api_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    embedding_service: EmbeddingService = Depends()
):
    """Upload document to knowledge base"""
    try:
        content = await file.read()
        
        # Process and embed document
        result = await embedding_service.process_document(
            filename=file.filename,
            content=content
        )
        
        return {
            "success": True,
            "document_id": result["document_id"],
            "chunks": result["chunks"]
        }
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    return {
        "agents": [
            {
                "id": "agent-1",
                "status": "available",
                "current_calls": 0
            }
        ]
    }

@api_router.post("/test/call")
async def test_call(phone_number: str):
    """Initiate a test call"""
    # This would trigger a simulated call
    return {
        "call_id": f"test-{datetime.now().timestamp()}",
        "status": "initiated"
    }