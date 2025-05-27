"""
Conversation and call models
"""
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import Column, String, DateTime, JSON, Enum as SQLEnum, Integer, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
import json

class CallStatus(str, Enum):
    """Call status enum"""
    INITIATED = "initiated"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    TRANSFERRED = "transferred"
    ENDED = "ended"
    FAILED = "failed"

class ConversationState:
    """Manages conversation state and context"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.turn_count: int = 0
        self.customer_info: Dict[str, Any] = {}
        self.session_data: Dict[str, Any] = {}
        self.emotion_history: List[str] = []
        self.intent_history: List[str] = []
        self.tools_used: List[str] = []
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": self.turn_count,
            "message_id": str(uuid.uuid4()),
            **(metadata or {})
        }
        self.messages.append(message)
        
        if role == "user":
            self.turn_count += 1
            
    def add_system_message(self, content: str, message_type: str = "system"):
        """Add system message (for internal tracking)"""
        self.add_message("system", content, {"type": message_type})
            
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages"""
        return self.messages[-n:] if len(self.messages) > n else self.messages
        
    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get all messages by specific role"""
        return [msg for msg in self.messages if msg.get("role") == role]
        
    def update_context(self, key: str, value: Any):
        """Update conversation context"""
        self.context[key] = value
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value"""
        return self.context.get(key, default)
        
    def update_customer_info(self, info: Dict[str, Any]):
        """Update customer information"""
        self.customer_info.update(info)
        
    def add_emotion(self, emotion: str):
        """Track emotion changes"""
        if not self.emotion_history or self.emotion_history[-1] != emotion:
            self.emotion_history.append(emotion)
            
    def add_intent(self, intent: str):
        """Track intent history"""
        self.intent_history.append(intent)
        
    def add_tool_usage(self, tool_name: str):
        """Track tool usage"""
        self.tools_used.append(tool_name)
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Generate conversation summary"""
        user_messages = self.get_messages_by_role("user")
        agent_messages = self.get_messages_by_role("assistant")
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "agent_messages": len(agent_messages),
            "turns": self.turn_count,
            "duration_minutes": self._calculate_duration(),
            "emotions_detected": list(set(self.emotion_history)),
            "intents_identified": list(set(self.intent_history)),
            "tools_used": list(set(self.tools_used)),
            "customer_identified": bool(self.customer_info),
            "last_activity": self.messages[-1]["timestamp"] if self.messages else None
        }
        
    def _calculate_duration(self) -> float:
        """Calculate conversation duration in minutes"""
        if len(self.messages) < 2:
            return 0.0
            
        try:
            start_time = datetime.fromisoformat(self.messages[0]["timestamp"])
            end_time = datetime.fromisoformat(self.messages[-1]["timestamp"])
            return (end_time - start_time).total_seconds() / 60.0
        except:
            return 0.0
        
    def clear_session(self):
        """Clear session data while preserving conversation"""
        self.session_data.clear()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "messages": self.messages,
            "context": self.context,
            "turn_count": self.turn_count,
            "customer_info": self.customer_info,
            "session_data": self.session_data,
            "emotion_history": self.emotion_history,
            "intent_history": self.intent_history,
            "tools_used": self.tools_used,
            "summary": self.get_conversation_summary()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create instance from dictionary"""
        instance = cls()
        instance.messages = data.get("messages", [])
        instance.context = data.get("context", {})
        instance.turn_count = data.get("turn_count", 0)
        instance.customer_info = data.get("customer_info", {})
        instance.session_data = data.get("session_data", {})
        instance.emotion_history = data.get("emotion_history", [])
        instance.intent_history = data.get("intent_history", [])
        instance.tools_used = data.get("tools_used", [])
        return instance

class CallRecord(Base):
    """Database model for call records"""
    __tablename__ = "call_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, unique=True, index=True)
    status = Column(SQLEnum(CallStatus), default=CallStatus.INITIATED)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Integer, default=0)  # in seconds
    
    # Customer information
    customer_phone = Column(String, nullable=True)
    customer_email = Column(String, nullable=True)
    customer_id = Column(String, nullable=True)
    
    # Call details
    call_direction = Column(String, default="inbound")  # inbound, outbound
    call_source = Column(String, nullable=True)  # web, phone, api
    caller_id = Column(String, nullable=True)
    
    # Conversation data
    transcript = Column(JSON, default=list)
    summary = Column(JSON, nullable=True)
    sentiment = Column(String, nullable=True)
    
    # Quality metrics
    audio_quality = Column(String, default="good")
    connection_quality = Column(String, default="stable")
    customer_satisfaction = Column(Integer, nullable=True)  # 1-5 rating
    
    # Business outcomes
    resolution_achieved = Column(Boolean, default=False)
    follow_up_required = Column(Boolean, default=False)
    escalated = Column(Boolean, default=False)
    
    # Agent information
    primary_agent_type = Column(String, default="ai")  # ai, human
    primary_agent_id = Column(String, nullable=True)
    agent_performance_score = Column(Float, nullable=True)
    
    # Technical metadata
    call_metadata = Column(JSON, default=dict)
    recording_path = Column(String, nullable=True)
    
    # Relationships
    claims = relationship("Claim", back_populates="call")
    
    def __repr__(self):
        return f"<CallRecord(id='{self.id}', session='{self.session_id}', status='{self.status}')>"
    
    def add_transcript_entry(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add entry to transcript"""
        if not self.transcript:
            self.transcript = []
            
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.transcript.append(entry)
    
    def update_status(self, new_status: CallStatus, notes: str = None):
        """Update call status with optional notes"""
        old_status = self.status
        self.status = new_status
        
        if new_status == CallStatus.ENDED and not self.ended_at:
            self.ended_at = datetime.utcnow()
            if self.started_at:
                self.duration = int((self.ended_at - self.started_at).total_seconds())
        
        # Add status change to metadata
        if not self.call_metadata:
            self.call_metadata = {}
        if "status_history" not in self.call_metadata:
            self.call_metadata["status_history"] = []
            
        self.call_metadata["status_history"].append({
            "from": old_status.value if old_status else None,
            "to": new_status.value,
            "timestamp": datetime.now().isoformat(),
            "notes": notes
        })
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate call metrics"""
        metrics = {
            "duration_minutes": self.duration / 60.0 if self.duration else 0,
            "message_count": len(self.transcript) if self.transcript else 0,
            "customer_messages": 0,
            "agent_messages": 0,
            "avg_response_time": 0.0,
            "conversation_turns": 0
        }
        
        if self.transcript:
            customer_msgs = [m for m in self.transcript if m.get("role") == "user"]
            agent_msgs = [m for m in self.transcript if m.get("role") == "assistant"]
            
            metrics["customer_messages"] = len(customer_msgs)
            metrics["agent_messages"] = len(agent_msgs)
            metrics["conversation_turns"] = min(len(customer_msgs), len(agent_msgs))
            
            # Calculate average response time
            response_times = []
            for i in range(len(self.transcript) - 1):
                if (self.transcript[i].get("role") == "user" and 
                    self.transcript[i + 1].get("role") == "assistant"):
                    try:
                        user_time = datetime.fromisoformat(self.transcript[i]["timestamp"])
                        agent_time = datetime.fromisoformat(self.transcript[i + 1]["timestamp"])
                        response_time = (agent_time - user_time).total_seconds()
                        response_times.append(response_time)
                    except:
                        continue
            
            if response_times:
                metrics["avg_response_time"] = sum(response_times) / len(response_times)
        
        return metrics
    
    def to_dict(self, include_transcript: bool = True) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration,
            "customer_phone": self.customer_phone,
            "customer_email": self.customer_email,
            "customer_id": self.customer_id,
            "call_direction": self.call_direction,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "resolution_achieved": self.resolution_achieved,
            "follow_up_required": self.follow_up_required,
            "call_metadata": self.call_metadata
        }
        
        if include_transcript:
            data["transcript"] = self.transcript
            
        return data

class Claim(Base):
    """Database model for insurance claims"""
    __tablename__ = "claims"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("call_records.id"), nullable=True)
    
    # Basic claim information
    claim_type = Column(String, nullable=False)  # auto, home, health, life
    status = Column(String, default="open")
    priority = Column(String, default="normal")  # low, normal, high, urgent
    
    # Dates
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    incident_date = Column(DateTime, nullable=True)
    reported_date = Column(DateTime, default=datetime.utcnow)
    
    # Claim details
    description = Column(Text, nullable=False)
    cause_of_loss = Column(String, nullable=True)
    location_of_loss = Column(String, nullable=True)
    
    # Financial information
    estimated_amount = Column(Float, nullable=True)
    deductible_amount = Column(Float, nullable=True)
    settlement_amount = Column(Float, nullable=True)
    
    # Customer information
    customer_id = Column(String, nullable=True)
    policy_number = Column(String, nullable=True)
    
    # Processing information
    assigned_adjuster = Column(String, nullable=True)
    adjuster_contact = Column(String, nullable=True)
    
    # Documentation
    documents_required = Column(JSON, default=list)
    documents_received = Column(JSON, default=list)
    photos_required = Column(Boolean, default=False)
    photos_received = Column(Boolean, default=False)
    
    # Additional data
    data = Column(JSON, default=dict)
    
    # Relationships
    call = relationship("CallRecord", back_populates="claims")
    
    def __repr__(self):
        return f"<Claim(id='{self.id}', type='{self.claim_type}', status='{self.status}')>"
    
    def update_status(self, new_status: str, notes: str = None, updated_by: str = None):
        """Update claim status with audit trail"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        # Add to audit trail
        if not self.data:
            self.data = {}
        if "status_history" not in self.data:
            self.data["status_history"] = []
            
        self.data["status_history"].append({
            "from_status": old_status,
            "to_status": new_status,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": updated_by or "system",
            "notes": notes
        })
    
    def add_document(self, document_type: str, file_path: str, uploaded_by: str = None):
        """Add document to claim"""
        if not self.data:
            self.data = {}
        if "documents" not in self.data:
            self.data["documents"] = []
            
        document = {
            "type": document_type,
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat(),
            "uploaded_by": uploaded_by or "customer",
            "document_id": str(uuid.uuid4())
        }
        
        self.data["documents"].append(document)
        
        # Update received documents list
        if not self.documents_received:
            self.documents_received = []
        if document_type not in self.documents_received:
            self.documents_received.append(document_type)
    
    def calculate_settlement_estimate(self) -> Dict[str, Any]:
        """Calculate estimated settlement based on claim data"""
        estimate = {
            "estimated_gross": self.estimated_amount or 0,
            "deductible": self.deductible_amount or 0,
            "estimated_net": 0,
            "factors": []
        }
        
        if estimate["estimated_gross"] > 0 and estimate["deductible"] > 0:
            estimate["estimated_net"] = max(0, estimate["estimated_gross"] - estimate["deductible"])
        
        # Add factors that might affect settlement
        if self.claim_type == "auto":
            if self.data and self.data.get("fault_percentage"):
                fault_pct = self.data["fault_percentage"]
                if fault_pct < 100:
                    estimate["factors"].append(f"Fault percentage: {fault_pct}%")
                    estimate["estimated_net"] *= (fault_pct / 100)
        
        return estimate
    
    def get_required_actions(self) -> List[str]:
        """Get list of required actions to process claim"""
        actions = []
        
        # Document requirements
        if self.documents_required:
            missing_docs = set(self.documents_required) - set(self.documents_received or [])
            for doc in missing_docs:
                actions.append(f"Obtain {doc}")
        
        # Photo requirements
        if self.photos_required and not self.photos_received:
            actions.append("Obtain photos of damage")
        
        # Status-specific actions
        if self.status == "open":
            actions.append("Assign adjuster")
        elif self.status == "in_progress":
            if not self.assigned_adjuster:
                actions.append("Assign adjuster")
        elif self.status == "pending_review":
            actions.append("Complete claim review")
        
        return actions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "call_id": self.call_id,
            "claim_type": self.claim_type,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "incident_date": self.incident_date.isoformat() if self.incident_date else None,
            "description": self.description,
            "estimated_amount": self.estimated_amount,
            "settlement_amount": self.settlement_amount,
            "customer_id": self.customer_id,
            "policy_number": self.policy_number,
            "assigned_adjuster": self.assigned_adjuster,
            "data": self.data,
            "required_actions": self.get_required_actions(),
            "settlement_estimate": self.calculate_settlement_estimate()
        }

class ConversationManager:
    """Advanced conversation flow and state management"""
    
    def __init__(self):
        self.state = ConversationState()
        self.intents = []
        self.entities = {}
        self.context_memory = {}
        self.interaction_patterns = []
        
    def process_message(self, message: str, role: str = "user", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process incoming message with advanced analysis"""
        # Add to conversation state
        self.state.add_message(role, message, metadata)
        
        processing_result = {
            "message_added": True,
            "turn": self.state.turn_count,
            "analysis": {}
        }
        
        if role == "user":
            # Analyze user message
            analysis = self._analyze_user_message(message)
            processing_result["analysis"] = analysis
            
            # Update state based on analysis
            if analysis.get("intent"):
                self.state.add_intent(analysis["intent"])
            if analysis.get("emotion"):
                self.state.add_emotion(analysis["emotion"])
            if analysis.get("entities"):
                self.entities.update(analysis["entities"])
        
        return processing_result
    
    def _analyze_user_message(self, message: str) -> Dict[str, Any]:
        """Analyze user message for intent, entities, and sentiment"""
        from app.utils.text_processing import extract_intent, extract_phone_number, extract_email, extract_dates
        
        # Extract intent
        intent_info = extract_intent(message)
        
        # Extract entities
        entities = {
            "phones": extract_phone_number(message),
            "emails": extract_email(message),
            "dates": extract_dates(message)
        }
        
        # Simple sentiment analysis
        emotion = self._detect_emotion(message)
        
        # Extract keywords
        keywords = self._extract_keywords(message)
        
        return {
            "intent": intent_info.get("primary_intent"),
            "intent_confidence": intent_info.get("scores", {}),
            "emotion": emotion,
            "entities": entities,
            "keywords": keywords,
            "message_length": len(message.split()),
            "urgency_indicators": self._detect_urgency(message)
        }
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text"""
        text_lower = text.lower()
        
        # Emotion indicators
        emotions = {
            "frustrated": ["frustrated", "angry", "mad", "annoyed", "upset", "furious"],
            "confused": ["confused", "don't understand", "unclear", "lost", "puzzled"],
            "worried": ["worried", "concerned", "scared", "anxious", "nervous"],
            "happy": ["happy", "great", "excellent", "wonderful", "pleased", "satisfied"],
            "urgent": ["urgent", "asap", "immediately", "emergency", "right now"]
        }
        
        for emotion, indicators in emotions.items():
            if any(indicator in text_lower for indicator in indicators):
                return emotion
        
        return "neutral"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction
        keywords = []
        important_terms = [
            "claim", "policy", "accident", "damage", "insurance", "coverage",
            "deductible", "premium", "appointment", "agent", "cancel", "refund"
        ]
        
        text_lower = text.lower()
        for term in important_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords
    
    def _detect_urgency(self, text: str) -> List[str]:
        """Detect urgency indicators"""
        urgency_indicators = []
        text_lower = text.lower()
        
        urgent_phrases = [
            "emergency", "urgent", "asap", "immediately", "right now",
            "can't wait", "need help now", "very important"
        ]
        
        for phrase in urgent_phrases:
            if phrase in text_lower:
                urgency_indicators.append(phrase)
        
        return urgency_indicators
    
    def get_context_prompt(self, max_messages: int = 10) -> str:
        """Get formatted context for LLM prompt"""
        recent_messages = self.state.get_recent_messages(max_messages)
        
        prompt = "Recent conversation:\n"
        for msg in recent_messages:
            role = msg["role"].title()
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            prompt += f"{role}: {content}\n"
        
        # Add context information
        if self.state.customer_info:
            prompt += f"\nCustomer Information: {json.dumps(self.state.customer_info, indent=2)}\n"
        
        if self.state.context:
            prompt += f"\nConversation Context: {json.dumps(self.state.context, indent=2)}\n"
        
        # Add recent intents and emotions
        if self.state.intent_history:
            recent_intents = list(set(self.state.intent_history[-3:]))
            prompt += f"\nRecent Intents: {', '.join(recent_intents)}\n"
        
        if self.state.emotion_history:
            recent_emotions = list(set(self.state.emotion_history[-3:]))
            prompt += f"\nRecent Emotions: {', '.join(recent_emotions)}\n"
        
        return prompt
    
    def should_escalate(self) -> bool:
        """Determine if conversation should be escalated"""
        # Check for escalation conditions
        escalation_conditions = [
            len(self.state.emotion_history) > 0 and "frustrated" in self.state.emotion_history[-2:],
            self.state.turn_count > 15,  # Long conversation
            "transfer" in self.state.intent_history,
            any(indicator in ["emergency", "urgent"] for indicator in 
                [msg.get("content", "").lower() for msg in self.state.get_recent_messages(3)])
        ]
        
        return any(escalation_conditions)
    
    def get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on conversation state"""
        actions = []
        
        # Based on recent intents
        recent_intents = self.state.intent_history[-3:] if self.state.intent_history else []
        
        if "create_claim" in recent_intents:
            actions.append("Guide customer through claim creation process")
        if "check_status" in recent_intents:
            actions.append("Look up customer account and provide status update")
        if "schedule" in recent_intents:
            actions.append("Offer available appointment times")
        
        # Based on emotions
        recent_emotions = self.state.emotion_history[-2:] if self.state.emotion_history else []
        
        if "frustrated" in recent_emotions:
            actions.append("Use empathetic language and consider escalation")
        if "confused" in recent_emotions:
            actions.append("Provide clear, step-by-step explanations")
        
        # Based on conversation length
        if self.state.turn_count > 10:
            actions.append("Summarize progress and confirm next steps")
        
        # If customer not identified
        if not self.state.customer_info:
            actions.append("Gather customer identification information")
        
        return actions
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive conversation summary"""
        summary = self.state.get_conversation_summary()
        
        # Add manager-specific insights
        summary.update({
            "escalation_recommended": self.should_escalate(),
            "recommended_actions": self.get_recommended_actions(),
            "key_topics": list(set(self.state.intent_history)),
            "emotional_journey": self.state.emotion_history,
            "customer_satisfaction_indicators": self._assess_satisfaction(),
            "resolution_likelihood": self._assess_resolution_likelihood()
        })
        
        return summary
    
    def _assess_satisfaction(self) -> Dict[str, Any]:
        """Assess customer satisfaction indicators"""
        indicators = {
            "positive_signals": 0,
            "negative_signals": 0,
            "neutral_signals": 0,
            "overall_sentiment": "neutral"
        }
        
        # Analyze emotions
        for emotion in self.state.emotion_history:
            if emotion in ["happy", "satisfied"]:
                indicators["positive_signals"] += 1
            elif emotion in ["frustrated", "angry", "worried"]:
                indicators["negative_signals"] += 1
            else:
                indicators["neutral_signals"] += 1
        
        # Determine overall sentiment
        if indicators["positive_signals"] > indicators["negative_signals"]:
            indicators["overall_sentiment"] = "positive"
        elif indicators["negative_signals"] > indicators["positive_signals"]:
            indicators["overall_sentiment"] = "negative"
        
        return indicators
    
    def _assess_resolution_likelihood(self) -> float:
        """Assess likelihood of successful resolution (0-1)"""
        score = 0.5  # Base score
        
        # Positive factors
        if self.state.customer_info:
            score += 0.1  # Customer identified
        if "create_claim" in self.state.intent_history:
            score += 0.2  # Clear intent
        if len(self.state.tools_used) > 0:
            score += 0.1  # Tools being used
        
        # Negative factors
        if "frustrated" in self.state.emotion_history:
            score -= 0.2
        if self.state.turn_count > 15:
            score -= 0.1  # Long conversation without resolution
        
        return max(0.0, min(1.0, score))