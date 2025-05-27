"""
REST API routes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from app.core.database import get_session
from app.models.conversation import CallRecord, CallStatus, Claim
from app.models.user import User
from app.services.embedding_service import EmbeddingService
from app.services.cache_service import cache_service
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

api_router = APIRouter()

# Dependency for embedding service
async def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@api_router.get("/calls")
async def get_calls(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[CallStatus] = None,
    customer_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    """Get list of calls with filtering options"""
    try:
        # Build query
        query = select(CallRecord)
        conditions = []
        
        if status:
            conditions.append(CallRecord.status == status)
        if customer_id:
            conditions.append(CallRecord.customer_id == customer_id)
        if date_from:
            try:
                date_from_parsed = datetime.fromisoformat(date_from)
                conditions.append(CallRecord.started_at >= date_from_parsed)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")
        if date_to:
            try:
                date_to_parsed = datetime.fromisoformat(date_to)
                conditions.append(CallRecord.started_at <= date_to_parsed)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")
                
        if conditions:
            query = query.where(and_(*conditions))
            
        # Get total count
        count_query = select(func.count(CallRecord.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Get paginated results
        query = query.order_by(CallRecord.started_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        calls = result.scalars().all()
        
        # Format response
        formatted_calls = []
        for call in calls:
            formatted_calls.append({
                "id": call.id,
                "session_id": call.session_id,
                "status": call.status.value,
                "started_at": call.started_at.isoformat(),
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration": call.duration,
                "customer_phone": call.customer_phone,
                "customer_email": call.customer_email,
                "summary": call.summary,
                "sentiment": call.sentiment
            })
        
        return {
            "calls": formatted_calls,
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": total > skip + limit
        }
        
    except Exception as e:
        logger.error(f"Error getting calls: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/calls/{call_id}")
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get specific call details"""
    try:
        # Get call record
        query = select(CallRecord).where(CallRecord.id == call_id)
        result = await session.execute(query)
        call = result.scalar_one_or_none()
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
            
        # Get associated claims
        claims_query = select(Claim).where(Claim.call_id == call_id)
        claims_result = await session.execute(claims_query)
        claims = claims_result.scalars().all()
        
        # Get customer info if available
        customer = None
        if call.customer_id:
            customer_query = select(User).where(User.id == call.customer_id)
            customer_result = await session.execute(customer_query)
            customer = customer_result.scalar_one_or_none()
        
        return {
            "id": call.id,
            "session_id": call.session_id,
            "status": call.status.value,
            "started_at": call.started_at.isoformat(),
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            "duration": call.duration,
            "customer_phone": call.customer_phone,
            "customer_email": call.customer_email,
            "customer_info": customer.to_dict() if customer else None,
            "transcript": call.transcript,
            "summary": call.summary,
            "sentiment": call.sentiment,
            "metadata": call.metadata,
            "claims": [
                {
                    "id": claim.id,
                    "type": claim.claim_type,
                    "status": claim.status,
                    "created_at": claim.created_at.isoformat(),
                    "description": claim.description
                } for claim in claims
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call {call_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/calls/{call_id}/transcript")
async def update_transcript(
    call_id: str,
    transcript_data: dict,
    session: AsyncSession = Depends(get_session)
):
    """Update call transcript"""
    try:
        # Get call record
        query = select(CallRecord).where(CallRecord.id == call_id)
        result = await session.execute(query)
        call = result.scalar_one_or_none()
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
            
        # Update transcript
        new_transcript = transcript_data.get("transcript", [])
        if isinstance(new_transcript, list):
            call.transcript = new_transcript
        else:
            # Append single message
            if not call.transcript:
                call.transcript = []
            call.transcript.append(transcript_data)
            
        await session.commit()
        
        return {"success": True, "message": "Transcript updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating transcript for call {call_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/calls")
async def create_call(
    call_data: dict,
    session: AsyncSession = Depends(get_session)
):
    """Create a new call record"""
    try:
        call_id = str(uuid.uuid4())
        session_id = call_data.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
        
        new_call = CallRecord(
            id=call_id,
            session_id=session_id,
            status=CallStatus.INITIATED,
            started_at=datetime.utcnow(),
            customer_phone=call_data.get("customer_phone"),
            customer_email=call_data.get("customer_email"),
            customer_id=call_data.get("customer_id"),
            transcript=[],
            metadata=call_data.get("metadata", {})
        )
        
        session.add(new_call)
        await session.commit()
        await session.refresh(new_call)
        
        return {
            "success": True,
            "call_id": call_id,
            "session_id": session_id,
            "status": new_call.status.value
        }
        
    except Exception as e:
        logger.error(f"Error creating call: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.put("/calls/{call_id}/status")
async def update_call_status(
    call_id: str,
    status_data: dict,
    session: AsyncSession = Depends(get_session)
):
    """Update call status"""
    try:
        query = select(CallRecord).where(CallRecord.id == call_id)
        result = await session.execute(query)
        call = result.scalar_one_or_none()
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
            
        new_status = status_data.get("status")
        if new_status:
            try:
                call.status = CallStatus(new_status)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
                
        if new_status == "ended" and not call.ended_at:
            call.ended_at = datetime.utcnow()
            if call.started_at:
                call.duration = int((call.ended_at - call.started_at).total_seconds())
                
        # Update summary if provided
        if "summary" in status_data:
            call.summary = status_data["summary"]
            
        # Update sentiment if provided
        if "sentiment" in status_data:
            call.sentiment = status_data["sentiment"]
            
        await session.commit()
        
        return {"success": True, "status": call.status.value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating call status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = None,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Upload document to knowledge base"""
    try:
        # Validate file type
        allowed_types = ['text/plain', 'application/pdf', 'text/markdown']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_types}"
            )
            
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.content_type == 'text/plain':
            text_content = content.decode('utf-8')
        elif file.content_type == 'application/pdf':
            # For PDF files, you'd need to add PDF parsing
            raise HTTPException(status_code=400, detail="PDF processing not yet implemented")
        else:
            text_content = content.decode('utf-8')
            
        # Prepare metadata
        metadata = {
            "content_type": file.content_type,
            "file_size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }
        
        if category:
            metadata["category"] = category
            
        # Process and embed document
        result = await embedding_service.process_document(
            filename=file.filename,
            content=text_content,
            metadata=metadata
        )
        
        if result.get("status") == "success":
            return {
                "success": True,
                "document_id": result["document_id"],
                "chunks": result["chunks"],
                "filename": file.filename
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=result.get("error", "Document processing failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/documents")
async def list_documents(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """List all documents in knowledge base"""
    try:
        documents = await embedding_service.list_documents()
        stats = await embedding_service.get_collection_stats()
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": stats.get("total_chunks", 0),
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Delete document from knowledge base"""
    try:
        result = await embedding_service.delete_document(document_id)
        
        if result.get("status") == "success":
            return {
                "success": True,
                "message": f"Deleted {result.get('deleted', 0)} chunks",
                "document_id": document_id
            }
        else:
            raise HTTPException(
                status_code=404 if "not found" in result.get("message", "") else 400,
                detail=result.get("message", "Delete failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.post("/documents/search")
async def search_documents(
    search_data: dict,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Search documents in knowledge base"""
    try:
        query = search_data.get("query", "")
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        n_results = min(search_data.get("n_results", 5), 20)
        category = search_data.get("category")
        
        # Search
        if category:
            results = await embedding_service.search_by_category(
                query=query,
                category=category,
                n_results=n_results
            )
        else:
            results = await embedding_service.search(
                query=query,
                n_results=n_results
            )
            
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/customers")
async def list_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    """List customers with optional search"""
    try:
        query = select(User)
        
        if search:
            search_term = f"%{search}%"
            query = query.where(
                or_(
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term),
                    User.phone_number.ilike(search_term),
                    User.email.ilike(search_term),
                    User.account_number.ilike(search_term)
                )
            )
            
        # Get total count
        count_query = select(func.count(User.id))
        if search:
            search_term = f"%{search}%"
            count_query = count_query.where(
                or_(
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term),
                    User.phone_number.ilike(search_term),
                    User.email.ilike(search_term),
                    User.account_number.ilike(search_term)
                )
            )
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Get paginated results
        query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        customers = result.scalars().all()
        
        return {
            "customers": [customer.to_dict() for customer in customers],
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": total > skip + limit
        }
        
    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/customers/{customer_id}")
async def get_customer(
    customer_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get customer details with call history"""
    try:
        # Get customer
        customer_query = select(User).where(User.id == customer_id)
        customer_result = await session.execute(customer_query)
        customer = customer_result.scalar_one_or_none()
        
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        # Get recent calls
        calls_query = select(CallRecord).where(
            CallRecord.customer_id == customer_id
        ).order_by(CallRecord.started_at.desc()).limit(10)
        calls_result = await session.execute(calls_query)
        calls = calls_result.scalars().all()
        
        # Get claims
        claims_query = select(Claim).join(CallRecord).where(
            CallRecord.customer_id == customer_id
        ).order_by(Claim.created_at.desc())
        claims_result = await session.execute(claims_query)
        claims = claims_result.scalars().all()
        
        return {
            "customer": customer.to_dict(),
            "recent_calls": [
                {
                    "id": call.id,
                    "status": call.status.value,
                    "started_at": call.started_at.isoformat(),
                    "duration": call.duration,
                    "summary": call.summary
                } for call in calls
            ],
            "claims": [
                {
                    "id": claim.id,
                    "type": claim.claim_type,
                    "status": claim.status,
                    "created_at": claim.created_at.isoformat(),
                    "description": claim.description[:100] + "..." if len(claim.description) > 100 else claim.description
                } for claim in claims
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/claims")
async def list_claims(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    claim_type: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    """List claims with filtering"""
    try:
        query = select(Claim)
        conditions = []
        
        if status:
            conditions.append(Claim.status == status)
        if claim_type:
            conditions.append(Claim.claim_type == claim_type)
            
        if conditions:
            query = query.where(and_(*conditions))
            
        # Get total count
        count_query = select(func.count(Claim.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Get paginated results
        query = query.order_by(Claim.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        claims = result.scalars().all()
        
        return {
            "claims": [
                {
                    "id": claim.id,
                    "type": claim.claim_type,
                    "status": claim.status,
                    "created_at": claim.created_at.isoformat(),
                    "updated_at": claim.updated_at.isoformat(),
                    "description": claim.description,
                    "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                    "data": claim.data
                } for claim in claims
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": total > skip + limit
        }
        
    except Exception as e:
        logger.error(f"Error listing claims: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/claims/{claim_id}")
async def get_claim(
    claim_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get specific claim details"""
    try:
        query = select(Claim).where(Claim.id == claim_id)
        result = await session.execute(query)
        claim = result.scalar_one_or_none()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
            
        # Get associated call if available
        call_info = None
        if claim.call_id:
            call_query = select(CallRecord).where(CallRecord.id == claim.call_id)
            call_result = await session.execute(call_query)
            call = call_result.scalar_one_or_none()
            if call:
                call_info = {
                    "id": call.id,
                    "started_at": call.started_at.isoformat(),
                    "duration": call.duration,
                    "customer_phone": call.customer_phone
                }
                
        return {
            "id": claim.id,
            "type": claim.claim_type,
            "status": claim.status,
            "created_at": claim.created_at.isoformat(),
            "updated_at": claim.updated_at.isoformat(),
            "description": claim.description,
            "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
            "data": claim.data,
            "call_info": call_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting claim {claim_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/analytics/dashboard")
async def get_dashboard_analytics(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session)
):
    """Get dashboard analytics"""
    try:
        # Use cache for analytics data
        cache_key = f"analytics:dashboard:{days}"
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return cached_data
            
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Call statistics
        calls_query = select(func.count(CallRecord.id)).where(
            CallRecord.started_at >= start_date
        )
        calls_result = await session.execute(calls_query)
        total_calls = calls_result.scalar() or 0
        
        # Successful calls
        successful_calls_query = select(func.count(CallRecord.id)).where(
            and_(
                CallRecord.started_at >= start_date,
                CallRecord.status == CallStatus.ENDED
            )
        )
        successful_result = await session.execute(successful_calls_query)
        successful_calls = successful_result.scalar() or 0
        
        # Claims created
        claims_query = select(func.count(Claim.id)).where(
            Claim.created_at >= start_date
        )
        claims_result = await session.execute(claims_query)
        total_claims = claims_result.scalar() or 0
        
        # Average call duration
        duration_query = select(func.avg(CallRecord.duration)).where(
            and_(
                CallRecord.started_at >= start_date,
                CallRecord.duration.isnot(None)
            )
        )
        duration_result = await session.execute(duration_query)
        avg_duration = duration_result.scalar() or 0
        
        analytics_data = {
            "period_days": days,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            "total_claims": total_claims,
            "avg_call_duration": round(avg_duration, 2),
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache for 1 hour
        await cache_service.set(cache_key, analytics_data, ttl=3600)
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@api_router.post("/test/call")
async def test_call(test_data: dict):
    """Initiate a test call for development purposes"""
    try:
        phone_number = test_data.get("phone_number", "+1234567890")
        test_message = test_data.get("message", "This is a test call")
        
        # Create test call session
        session_id = f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # In a real implementation, this would trigger the actual call flow
        # For now, just return test data
        
        return {
            "success": True,
            "session_id": session_id,
            "phone_number": phone_number,
            "message": test_message,
            "status": "initiated",
            "test_mode": True
        }
        
    except Exception as e:
        logger.error(f"Test call error: {e}")
        raise HTTPException(status_code=500, detail="Test call failed")

@api_router.get("/agents/status")
async def get_agent_status():
    """Get status of AI agents"""
    try:
        # In a real implementation, this would check actual agent status
        # For now, return simulated status
        
        return {
            "agents": [
                {
                    "id": "agent-primary",
                    "name": "Primary Call Agent",
                    "status": "available",
                    "current_calls": 0,
                    "max_concurrent_calls": 10,
                    "uptime": "99.9%",
                    "last_health_check": datetime.now().isoformat()
                }
            ],
            "total_agents": 1,
            "available_agents": 1,
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")