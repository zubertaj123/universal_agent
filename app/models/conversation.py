"""
Conversation and call models
"""
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, DateTime, JSON, Enum as SQLEnum, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class CallStatus(str, Enum):
    """Call status enum"""
    INITIATED = "initiated"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    TRANSFERRED = "transferred"
    ENDED = "ended"
    FAILED = "failed"

class ConversationState:
    """Manages conversation state"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.turn_count: int = 0
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": self.turn_count,
            **(metadata or {})
        }
        self.messages.append(message)
        
        if role == "user":
            self.turn_count += 1
            
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages"""
        return self.messages[-n:]
        
    def update_context(self, key: str, value: Any):
        """Update conversation context"""
        self.context[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "messages": self.messages,
            "context": self.context,
            "turn_count": self.turn_count
        }

class CallRecord(Base):
    """Database model for call records"""
    __tablename__ = "call_records"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, unique=True, index=True)
    status = Column(SQLEnum(CallStatus), default=CallStatus.INITIATED)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Integer, default=0)  # in seconds
    
    # Customer information
    customer_phone = Column(String, nullable=True)
    customer_email = Column(String, nullable=True)
    customer_id = Column(String, nullable=True)
    
    # Conversation data
    transcript = Column(JSON, default=list)
    summary = Column(JSON, nullable=True)
    sentiment = Column(String, nullable=True)
    
    # Claim data
    claims = relationship("Claim", back_populates="call")
    
    # Metadata
    metadata = Column(JSON, default=dict)

class Claim(Base):
    """Database model for claims"""
    __tablename__ = "claims"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("call_records.id"))
    claim_type = Column(String)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Claim details
    description = Column(String)
    incident_date = Column(DateTime, nullable=True)
    data = Column(JSON, default=dict)
    
    # Relationships
    call = relationship("CallRecord", back_populates="claims")

class ConversationManager:
    """Manages conversation flow and state"""
    
    def __init__(self):
        self.state = ConversationState()
        self.intents = []
        self.entities = {}
        
    def process_message(self, message: str, role: str = "user") -> Dict[str, Any]:
        """Process incoming message"""
        # Add to conversation
        self.state.add_message(role, message)
        
        # Extract intents and entities (placeholder)
        # In real implementation, this would use NLU
        
        return {
            "message_added": True,
            "turn": self.state.turn_count
        }
        
    def get_context_prompt(self) -> str:
        """Get context for LLM prompt"""
        recent_messages = self.state.get_recent_messages()
        
        prompt = "Conversation history:\n"
        for msg in recent_messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
            
        if self.state.context:
            prompt += f"\nContext: {self.state.context}\n"
            
        return prompt