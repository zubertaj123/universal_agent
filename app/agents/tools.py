"""
Tools available to the call center agent
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
import uuid
import asyncio

from app.core.database import get_session
from app.models.user import User
from app.models.conversation import CallRecord, Claim, CallStatus
from app.services.embedding_service import EmbeddingService
from app.utils.logger import setup_logger
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

logger = setup_logger(__name__)

class CustomerLookupInput(BaseModel):
    """Input for customer lookup"""
    phone_number: Optional[str] = Field(None, description="Customer phone number")
    account_number: Optional[str] = Field(None, description="Customer account number")
    email: Optional[str] = Field(None, description="Customer email")

class ClaimInput(BaseModel):
    """Input for creating a claim"""
    customer_id: str = Field(description="Customer ID")
    claim_type: str = Field(description="Type of claim (auto, home, health, life)")
    description: str = Field(description="Detailed claim description")
    incident_date: Optional[str] = Field(None, description="Date of incident (YYYY-MM-DD)")
    estimated_amount: Optional[float] = Field(None, description="Estimated claim amount")
    location: Optional[str] = Field(None, description="Location where incident occurred")
    
class AppointmentInput(BaseModel):
    """Input for scheduling appointment"""
    customer_id: str = Field(description="Customer ID")
    appointment_type: str = Field(description="Type of appointment (inspection, consultation, etc.)")
    preferred_date: str = Field(description="Preferred date (YYYY-MM-DD)")
    preferred_time: str = Field(description="Preferred time (HH:MM)")
    notes: Optional[str] = Field(None, description="Additional notes for appointment")

class ClaimStatusInput(BaseModel):
    """Input for checking claim status"""
    claim_id: str = Field(description="Claim ID to check")

class KnowledgeSearchInput(BaseModel):
    """Input for knowledge base search"""
    query: str = Field(description="Search query")
    category: Optional[str] = Field(None, description="Category filter (optional)")

async def lookup_customer(
    phone_number: Optional[str] = None,
    account_number: Optional[str] = None,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """Look up customer information in the database"""
    try:
        async for session in get_session():
            # Build query conditions
            conditions = []
            if phone_number:
                conditions.append(User.phone_number == phone_number)
            if account_number:
                conditions.append(User.account_number == account_number)
            if email:
                conditions.append(User.email == email)
            
            if not conditions:
                return {"found": False, "error": "No search criteria provided"}
            
            # Query database
            query = select(User).where(or_(*conditions))
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                # Get recent call history
                call_query = select(CallRecord).where(
                    CallRecord.customer_id == user.id
                ).order_by(CallRecord.started_at.desc()).limit(5)
                
                call_result = await session.execute(call_query)
                recent_calls = call_result.scalars().all()
                
                return {
                    "found": True,
                    "customer": {
                        "id": user.id,
                        "name": f"{user.first_name} {user.last_name}".strip(),
                        "phone": user.phone_number,
                        "email": user.email,
                        "account_number": user.account_number,
                        "preferred_language": user.preferred_language,
                        "is_active": user.is_active,
                        "call_count": user.call_count,
                        "last_call_date": user.last_call_date.isoformat() if user.last_call_date else None,
                        "created_at": user.created_at.isoformat()
                    },
                    "recent_calls": [
                        {
                            "id": call.id,
                            "status": call.status.value,
                            "started_at": call.started_at.isoformat(),
                            "duration": call.duration,
                            "summary": call.summary
                        } for call in recent_calls
                    ]
                }
            else:
                return {"found": False, "message": "Customer not found"}
                
    except Exception as e:
        logger.error(f"Customer lookup error: {e}")
        return {"found": False, "error": str(e)}

async def create_customer(
    phone_number: str,
    first_name: str,
    last_name: str,
    email: Optional[str] = None,
    preferred_language: str = "en"
) -> Dict[str, Any]:
    """Create a new customer record"""
    try:
        async for session in get_session():
            # Check if customer already exists
            existing = await session.execute(
                select(User).where(User.phone_number == phone_number)
            )
            if existing.scalar_one_or_none():
                return {"success": False, "error": "Customer with this phone number already exists"}
            
            # Generate account number
            account_number = f"ACC{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
            
            # Create new customer
            new_customer = User(
                id=str(uuid.uuid4()),
                phone_number=phone_number,
                email=email,
                first_name=first_name,
                last_name=last_name,
                account_number=account_number,
                preferred_language=preferred_language,
                is_active=True,
                call_count=0
            )
            
            session.add(new_customer)
            await session.commit()
            await session.refresh(new_customer)
            
            return {
                "success": True,
                "customer": {
                    "id": new_customer.id,
                    "name": f"{new_customer.first_name} {new_customer.last_name}",
                    "phone": new_customer.phone_number,
                    "email": new_customer.email,
                    "account_number": new_customer.account_number
                }
            }
            
    except Exception as e:
        logger.error(f"Customer creation error: {e}")
        return {"success": False, "error": str(e)}

async def create_claim(
    customer_id: str,
    claim_type: str,
    description: str,
    incident_date: Optional[str] = None,
    estimated_amount: Optional[float] = None,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new insurance claim"""
    try:
        async for session in get_session():
            # Verify customer exists
            customer_query = select(User).where(User.id == customer_id)
            customer_result = await session.execute(customer_query)
            customer = customer_result.scalar_one_or_none()
            
            if not customer:
                return {"success": False, "error": "Customer not found"}
            
            # Generate claim ID
            claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
            
            # Parse incident date
            parsed_incident_date = None
            if incident_date:
                try:
                    parsed_incident_date = datetime.strptime(incident_date, '%Y-%m-%d')
                except ValueError:
                    try:
                        parsed_incident_date = datetime.strptime(incident_date, '%m/%d/%Y')
                    except ValueError:
                        parsed_incident_date = datetime.now()
            
            # Create claim
            new_claim = Claim(
                id=claim_id,
                call_id=None,  # Will be set by the calling context
                claim_type=claim_type.lower(),
                status="open",
                description=description,
                incident_date=parsed_incident_date,
                data={
                    "customer_id": customer_id,
                    "estimated_amount": estimated_amount,
                    "location": location,
                    "priority": determine_claim_priority(claim_type, estimated_amount),
                    "created_by": "ai_agent",
                    "contact_info": {
                        "phone": customer.phone_number,
                        "email": customer.email
                    }
                }
            )
            
            session.add(new_claim)
            await session.commit()
            await session.refresh(new_claim)
            
            return {
                "success": True,
                "claim": {
                    "id": new_claim.id,
                    "type": new_claim.claim_type,
                    "status": new_claim.status,
                    "description": new_claim.description,
                    "incident_date": new_claim.incident_date.isoformat() if new_claim.incident_date else None,
                    "created_at": new_claim.created_at.isoformat(),
                    "estimated_amount": estimated_amount,
                    "priority": new_claim.data.get("priority", "standard")
                },
                "next_steps": generate_claim_next_steps(claim_type, estimated_amount)
            }
            
    except Exception as e:
        logger.error(f"Claim creation error: {e}")
        return {"success": False, "error": str(e)}

async def check_claim_status(claim_id: str) -> Dict[str, Any]:
    """Check the status of an existing claim"""
    try:
        async for session in get_session():
            # Query claim
            claim_query = select(Claim).where(Claim.id == claim_id)
            claim_result = await session.execute(claim_query)
            claim = claim_result.scalar_one_or_none()
            
            if not claim:
                return {"found": False, "error": f"Claim {claim_id} not found"}
            
            # Get associated call information if available
            call_info = None
            if claim.call_id:
                call_query = select(CallRecord).where(CallRecord.id == claim.call_id)
                call_result = await session.execute(call_query)
                call = call_result.scalar_one_or_none()
                if call:
                    call_info = {
                        "call_date": call.started_at.isoformat(),
                        "duration": call.duration
                    }
            
            # Calculate progress
            progress = calculate_claim_progress(claim.status, claim.created_at)
            
            return {
                "found": True,
                "claim": {
                    "id": claim.id,
                    "type": claim.claim_type,
                    "status": claim.status,
                    "description": claim.description,
                    "incident_date": claim.incident_date.isoformat() if claim.incident_date else None,
                    "created_at": claim.created_at.isoformat(),
                    "updated_at": claim.updated_at.isoformat(),
                    "data": claim.data,
                    "progress": progress,
                    "estimated_completion": estimate_completion_date(claim.status, claim.claim_type)
                },
                "call_info": call_info,
                "status_description": get_status_description(claim.status)
            }
            
    except Exception as e:
        logger.error(f"Claim status check error: {e}")
        return {"found": False, "error": str(e)}

async def update_claim_status(
    claim_id: str,
    new_status: str,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Update claim status"""
    try:
        valid_statuses = ["open", "in_progress", "pending_review", "approved", "denied", "closed"]
        if new_status not in valid_statuses:
            return {"success": False, "error": f"Invalid status. Valid options: {valid_statuses}"}
        
        async for session in get_session():
            claim_query = select(Claim).where(Claim.id == claim_id)
            claim_result = await session.execute(claim_query)
            claim = claim_result.scalar_one_or_none()
            
            if not claim:
                return {"success": False, "error": "Claim not found"}
            
            old_status = claim.status
            claim.status = new_status
            claim.updated_at = datetime.utcnow()
            
            if notes:
                if "status_history" not in claim.data:
                    claim.data["status_history"] = []
                claim.data["status_history"].append({
                    "old_status": old_status,
                    "new_status": new_status,
                    "updated_at": datetime.utcnow().isoformat(),
                    "notes": notes,
                    "updated_by": "ai_agent"
                })
            
            await session.commit()
            
            return {
                "success": True,
                "claim_id": claim_id,
                "old_status": old_status,
                "new_status": new_status,
                "updated_at": claim.updated_at.isoformat()
            }
            
    except Exception as e:
        logger.error(f"Claim update error: {e}")
        return {"success": False, "error": str(e)}

async def schedule_appointment(
    customer_id: str,
    appointment_type: str,
    preferred_date: str,
    preferred_time: str,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Schedule an appointment for a customer"""
    try:
        async for session in get_session():
            # Verify customer exists
            customer_query = select(User).where(User.id == customer_id)
            customer_result = await session.execute(customer_query)
            customer = customer_result.scalar_one_or_none()
            
            if not customer:
                return {"success": False, "error": "Customer not found"}
            
            # Parse datetime
            try:
                appointment_datetime = datetime.strptime(
                    f"{preferred_date} {preferred_time}", 
                    "%Y-%m-%d %H:%M"
                )
            except ValueError:
                try:
                    appointment_datetime = datetime.strptime(
                        f"{preferred_date} {preferred_time}", 
                        "%m/%d/%Y %I:%M %p"
                    )
                except ValueError:
                    return {"success": False, "error": "Invalid date/time format"}
            
            # Check if appointment is in the future
            if appointment_datetime <= datetime.now():
                return {"success": False, "error": "Appointment must be scheduled for a future date/time"}
            
            # Check business hours (9 AM to 5 PM weekdays)
            if appointment_datetime.weekday() > 4:  # Weekend
                return {"success": False, "error": "Appointments only available on weekdays"}
            if appointment_datetime.hour < 9 or appointment_datetime.hour >= 17:
                return {"success": False, "error": "Appointments only available between 9 AM and 5 PM"}
            
            # Generate appointment ID
            appointment_id = f"APT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            
            # Create appointment data (you might want to create an Appointment model)
            appointment_data = {
                "id": appointment_id,
                "customer_id": customer_id,
                "type": appointment_type,
                "scheduled_datetime": appointment_datetime.isoformat(),
                "status": "confirmed",
                "notes": notes,
                "created_at": datetime.now().isoformat(),
                "customer_info": {
                    "name": f"{customer.first_name} {customer.last_name}",
                    "phone": customer.phone_number,
                    "email": customer.email
                }
            }
            
            # Here you would typically save to an appointments table
            # For now, we'll simulate successful scheduling
            
            return {
                "success": True,
                "appointment": appointment_data,
                "confirmation_message": f"Appointment scheduled for {appointment_datetime.strftime('%B %d, %Y at %I:%M %p')}",
                "reminder": "You will receive a reminder 24 hours before your appointment"
            }
            
    except Exception as e:
        logger.error(f"Appointment scheduling error: {e}")
        return {"success": False, "error": str(e)}

async def transfer_to_human(reason: str, priority: str = "normal") -> Dict[str, Any]:
    """Transfer call to human agent"""
    transfer_reasons = {
        "complex_claim": "Complex claim requiring human expertise",
        "customer_complaint": "Customer complaint requiring manager attention",
        "technical_issue": "Technical issue beyond AI capabilities",
        "customer_request": "Customer specifically requested human agent",
        "escalation": "Issue escalated due to complexity",
        "fraud_suspicion": "Potential fraud detected",
        "legal_matter": "Legal matter requiring specialized attention"
    }
    
    queue_times = {
        "urgent": "2-3 minutes",
        "high": "5-7 minutes",
        "normal": "10-15 minutes",
        "low": "15-20 minutes"
    }
    
    return {
        "action": "transfer",
        "reason": reason,
        "reason_description": transfer_reasons.get(reason, reason),
        "priority": priority,
        "estimated_wait_time": queue_times.get(priority, "10-15 minutes"),
        "transfer_id": f"TXF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "message": f"I'm transferring you to a human agent who can better assist you. {transfer_reasons.get(reason, reason)}. Expected wait time: {queue_times.get(priority, '10-15 minutes')}.",
        "instructions": "Please stay on the line. Your call is important to us."
    }

async def search_knowledge_base(query: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Search internal knowledge base"""
    try:
        embedding_service = EmbeddingService()
        
        # Prepare search filters
        filter_metadata = {}
        if category:
            filter_metadata["category"] = category
        
        # Perform search
        results = await embedding_service.search(
            query=query,
            n_results=5,
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        if results:
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result["metadata"].get("filename", "Knowledge Base Entry"),
                    "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                    "relevance_score": 1 - result["distance"],  # Convert distance to similarity
                    "source": result["metadata"].get("source", "Internal KB"),
                    "category": result["metadata"].get("type", "general")
                })
            
            return {
                "found": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        else:
            return {
                "found": False,
                "query": query,
                "message": "No relevant information found in knowledge base",
                "suggestion": "You may want to transfer to a human agent for specialized assistance"
            }
            
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return {
            "found": False,
            "query": query,
            "error": str(e),
            "fallback": "Unable to search knowledge base at this time"
        }

async def get_policy_information(
    customer_id: str,
    policy_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get customer policy information"""
    try:
        # This would typically query a policy management system
        # For now, we'll return sample data structure
        
        async for session in get_session():
            customer_query = select(User).where(User.id == customer_id)
            customer_result = await session.execute(customer_query)
            customer = customer_result.scalar_one_or_none()
            
            if not customer:
                return {"found": False, "error": "Customer not found"}
            
            # Sample policy data (replace with actual policy system integration)
            sample_policies = [
                {
                    "policy_number": f"AUTO-{customer.account_number}-001",
                    "type": "auto",
                    "status": "active",
                    "premium": 1200.00,
                    "deductible": 500.00,
                    "coverage_limits": {
                        "liability": 100000,
                        "collision": 50000,
                        "comprehensive": 50000
                    },
                    "renewal_date": "2024-12-31"
                },
                {
                    "policy_number": f"HOME-{customer.account_number}-001",
                    "type": "home",
                    "status": "active",
                    "premium": 800.00,
                    "deductible": 1000.00,
                    "coverage_limits": {
                        "dwelling": 250000,
                        "personal_property": 125000,
                        "liability": 300000
                    },
                    "renewal_date": "2024-12-31"
                }
            ]
            
            if policy_type:
                policies = [p for p in sample_policies if p["type"] == policy_type.lower()]
            else:
                policies = sample_policies
            
            return {
                "found": True,
                "customer_id": customer_id,
                "policies": policies,
                "total_policies": len(policies)
            }
            
    except Exception as e:
        logger.error(f"Policy information error: {e}")
        return {"found": False, "error": str(e)}

# Helper functions
def determine_claim_priority(claim_type: str, estimated_amount: Optional[float]) -> str:
    """Determine claim priority based on type and amount"""
    if estimated_amount and estimated_amount > 50000:
        return "high"
    elif estimated_amount and estimated_amount > 10000:
        return "medium"
    elif claim_type.lower() in ["auto", "home"]:
        return "medium"
    else:
        return "standard"

def generate_claim_next_steps(claim_type: str, estimated_amount: Optional[float]) -> List[str]:
    """Generate next steps for a claim"""
    steps = [
        "Claim has been successfully created and assigned a unique ID",
        "You will receive a confirmation email within 15 minutes"
    ]
    
    if claim_type.lower() == "auto":
        steps.extend([
            "An adjuster will contact you within 24-48 hours to schedule vehicle inspection",
            "If your vehicle is undriveable, we can arrange towing services",
            "Keep all receipts for rental car expenses (if applicable)"
        ])
    elif claim_type.lower() == "home":
        steps.extend([
            "A claims adjuster will schedule an inspection within 2-3 business days",
            "Take photos of all damage for documentation",
            "Make temporary repairs to prevent further damage (keep receipts)"
        ])
    else:
        steps.append("A claims specialist will review your case and contact you within 2 business days")
    
    if estimated_amount and estimated_amount > 25000:
        steps.append("Due to the claim amount, a senior adjuster will be assigned to your case")
    
    return steps

def calculate_claim_progress(status: str, created_at: datetime) -> Dict[str, Any]:
    """Calculate claim progress percentage"""
    status_progress = {
        "open": 10,
        "in_progress": 40,
        "pending_review": 70,
        "approved": 90,
        "denied": 100,
        "closed": 100
    }
    
    days_since_created = (datetime.utcnow() - created_at).days
    
    return {
        "percentage": status_progress.get(status, 0),
        "days_since_created": days_since_created,
        "current_stage": status.replace("_", " ").title()
    }

def estimate_completion_date(status: str, claim_type: str) -> str:
    """Estimate claim completion date"""
    if status in ["closed", "denied", "approved"]:
        return "Completed"
    
    # Typical processing times by claim type
    processing_days = {
        "auto": 7,
        "home": 14,
        "health": 21,
        "life": 30
    }
    
    days = processing_days.get(claim_type.lower(), 14)
    estimated_date = datetime.now() + timedelta(days=days)
    
    return estimated_date.strftime("%Y-%m-%d")

def get_status_description(status: str) -> str:
    """Get human-readable status description"""
    descriptions = {
        "open": "Your claim has been received and is being reviewed",
        "in_progress": "We are actively processing your claim",
        "pending_review": "Your claim is under final review",
        "approved": "Your claim has been approved and payment is being processed",
        "denied": "Your claim has been denied. You will receive a detailed explanation",
        "closed": "Your claim has been completed and closed"
    }
    
    return descriptions.get(status, "Status information not available")

def get_call_center_tools() -> List[Tool]:
    """Get all available tools for the call center agent"""
    tools = [
        StructuredTool.from_function(
            func=lookup_customer,
            name="lookup_customer",
            description="Look up customer information by phone, email, or account number. Use this when you need to identify or find information about a customer.",
            args_schema=CustomerLookupInput
        ),
        StructuredTool.from_function(
            func=create_claim,
            name="create_claim",
            description="Create a new insurance claim for a customer. Use this when a customer wants to file a new claim for auto, home, health, or life insurance.",
            args_schema=ClaimInput
        ),
        StructuredTool.from_function(
            func=check_claim_status,
            name="check_claim_status",
            description="Check the status and details of an existing claim. Use this when a customer asks about their claim status or progress.",
            args_schema=ClaimStatusInput
        ),
        Tool.from_function(
            func=update_claim_status,
            name="update_claim_status",
            description="Update the status of an existing claim. Use this only when authorized to change claim status based on new information.",
        ),
        StructuredTool.from_function(
            func=schedule_appointment,
            name="schedule_appointment",
            description="Schedule an appointment for a customer (inspection, consultation, etc.). Use this when customers need to schedule meetings or inspections.",
            args_schema=AppointmentInput
        ),
        Tool.from_function(
            func=transfer_to_human,
            name="transfer_to_human",
            description="Transfer the call to a human agent when the AI cannot handle the request or when specifically requested by the customer.",
        ),
        StructuredTool.from_function(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description="Search the internal knowledge base for information about policies, procedures, and FAQ answers.",
            args_schema=KnowledgeSearchInput
        ),
        Tool.from_function(
            func=get_policy_information,
            name="get_policy_information",
            description="Retrieve policy information for a customer including coverage details, premiums, and deductibles.",
        ),
        Tool.from_function(
            func=create_customer,
            name="create_customer",
            description="Create a new customer record when dealing with a first-time caller who needs to be added to the system.",
        ),
    ]
    
    return tools