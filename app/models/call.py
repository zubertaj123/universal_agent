"""
Call-related models and data structures
"""
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, JSON, Enum as SQLEnum, Integer, Float, Boolean, Text
from sqlalchemy.orm import relationship
from app.core.database import Base

class CallDirection(str, Enum):
    """Call direction enum"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"

class CallDisposition(str, Enum):
    """Call disposition/outcome enum"""
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"
    ESCALATED = "escalated"
    CALLBACK_REQUESTED = "callback_requested"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    FAILED = "failed"

class AudioQuality(str, Enum):
    """Audio quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class CallMetrics(Base):
    """Database model for detailed call metrics"""
    __tablename__ = "call_metrics"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, index=True)  # Foreign key to CallRecord
    
    # Audio metrics
    audio_quality = Column(SQLEnum(AudioQuality), default=AudioQuality.GOOD)
    silence_duration = Column(Float, default=0.0)  # Total silence in seconds
    speech_duration = Column(Float, default=0.0)   # Total speech in seconds
    avg_volume_level = Column(Float, default=0.0)
    peak_volume_level = Column(Float, default=0.0)
    noise_level = Column(Float, default=0.0)
    
    # Conversation metrics
    total_words_spoken = Column(Integer, default=0)
    customer_words = Column(Integer, default=0)
    agent_words = Column(Integer, default=0)
    speaking_rate_wpm = Column(Float, default=0.0)  # Words per minute
    
    # Response times
    avg_response_time = Column(Float, default=0.0)  # Average agent response time
    max_response_time = Column(Float, default=0.0)  # Maximum response time
    first_response_time = Column(Float, default=0.0)  # Time to first agent response
    
    # Interruptions and overlaps
    customer_interruptions = Column(Integer, default=0)
    agent_interruptions = Column(Integer, default=0)
    simultaneous_speech_duration = Column(Float, default=0.0)
    
    # Emotional analysis
    emotion_changes = Column(Integer, default=0)
    dominant_emotion = Column(String, default="neutral")
    emotion_timeline = Column(JSON, default=list)  # Time-series emotion data
    
    # Technical metrics
    latency_avg = Column(Float, default=0.0)  # Average processing latency
    latency_max = Column(Float, default=0.0)  # Maximum latency spike
    connection_quality = Column(String, default="stable")
    packet_loss = Column(Float, default=0.0)
    
    # AI performance
    transcription_accuracy = Column(Float, default=0.0)  # Estimated accuracy
    intent_recognition_confidence = Column(Float, default=0.0)
    tool_usage_count = Column(Integer, default=0)
    successful_tool_calls = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CallNote(Base):
    """Model for call notes and annotations"""
    __tablename__ = "call_notes"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, index=True)
    
    note_type = Column(String, default="general")  # general, issue, follow_up, etc.
    content = Column(Text, nullable=False)
    priority = Column(String, default="normal")  # low, normal, high, urgent
    
    # Categorization
    category = Column(String, nullable=True)  # billing, technical, complaint, etc.
    tags = Column(JSON, default=list)
    
    # Visibility and access
    is_internal = Column(Boolean, default=False)  # Internal note vs customer-visible
    created_by = Column(String, nullable=True)  # User ID who created the note
    
    # Follow-up tracking
    requires_follow_up = Column(Boolean, default=False)
    follow_up_date = Column(DateTime, nullable=True)
    follow_up_completed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CallTransfer(Base):
    """Model for call transfer records"""
    __tablename__ = "call_transfers"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, index=True)
    
    transfer_type = Column(String, nullable=False)  # warm, cold, conference
    transfer_reason = Column(String, nullable=False)
    transfer_category = Column(String, nullable=True)  # technical, billing, escalation
    
    # Source information
    from_agent_type = Column(String, default="ai")  # ai, human
    from_agent_id = Column(String, nullable=True)
    
    # Destination information
    to_agent_type = Column(String, default="human")  # ai, human, department
    to_agent_id = Column(String, nullable=True)
    to_department = Column(String, nullable=True)
    to_phone_number = Column(String, nullable=True)
    
    # Transfer execution
    transfer_initiated_at = Column(DateTime, default=datetime.utcnow)
    transfer_completed_at = Column(DateTime, nullable=True)
    transfer_successful = Column(Boolean, default=False)
    
    # Context preservation
    context_provided = Column(JSON, default=dict)  # Information passed to new agent
    customer_briefed = Column(Boolean, default=False)  # Was customer informed?
    
    # Quality metrics
    customer_satisfaction_impact = Column(String, nullable=True)  # positive, neutral, negative
    notes = Column(Text, nullable=True)

class CallRecording(Base):
    """Model for call recording metadata"""
    __tablename__ = "call_recordings"
    
    id = Column(String, primary_key=True)
    call_id = Column(String, index=True)
    
    # File information
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, default=0)  # Size in bytes
    file_format = Column(String, default="mp3")
    duration_seconds = Column(Float, default=0.0)
    
    # Recording metadata
    channels = Column(Integer, default=1)  # mono/stereo
    sample_rate = Column(Integer, default=16000)
    bit_rate = Column(Integer, default=128)  # kbps
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(String, nullable=True)
    
    # Transcription
    has_transcript = Column(Boolean, default=False)
    transcript_path = Column(String, nullable=True)
    transcript_confidence = Column(Float, default=0.0)
    
    # Privacy and compliance
    is_encrypted = Column(Boolean, default=True)
    retention_until = Column(DateTime, nullable=True)  # When to delete
    compliance_flags = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models for API responses and requests

class CallSummaryResponse(BaseModel):
    """Response model for call summary"""
    call_id: str
    session_id: str
    duration: int
    status: str
    customer_satisfaction: Optional[str] = None
    resolution_achieved: bool = False
    follow_up_required: bool = False
    
    # Key outcomes
    claims_created: List[str] = []
    appointments_scheduled: List[str] = []
    issues_resolved: List[str] = []
    
    # Metrics summary
    total_messages: int = 0
    customer_emotion_summary: str = "neutral"
    agent_performance_score: Optional[float] = None
    
    class Config:
        from_attributes = True

class CallMetricsResponse(BaseModel):
    """Response model for call metrics"""
    call_id: str
    audio_quality: str
    conversation_quality: Dict[str, Any]
    technical_metrics: Dict[str, Any]
    ai_performance: Dict[str, Any]
    
    class Config:
        from_attributes = True

class CallAnalyticsRequest(BaseModel):
    """Request model for call analytics"""
    start_date: datetime
    end_date: datetime
    metrics: List[str] = ["duration", "satisfaction", "resolution_rate"]
    group_by: Optional[str] = None  # hour, day, week, month
    filters: Dict[str, Any] = {}

class CallAnalyticsResponse(BaseModel):
    """Response model for call analytics"""
    period: Dict[str, str]
    total_calls: int
    metrics: Dict[str, Any]
    trends: List[Dict[str, Any]] = []
    top_issues: List[Dict[str, Any]] = []
    performance_indicators: Dict[str, float] = {}

class LiveCallStatus(BaseModel):
    """Model for live call status updates"""
    call_id: str
    session_id: str
    status: str
    current_speaker: str
    emotion: str
    confidence: float
    last_message_time: datetime
    duration: int
    
    # Real-time metrics
    audio_level: float = 0.0
    connection_quality: str = "good"
    processing_latency: float = 0.0
    
    # Current context
    current_intent: Optional[str] = None
    customer_identified: bool = False
    active_tools: List[str] = []

class CallSearchRequest(BaseModel):
    """Request model for searching calls"""
    query: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    status_filter: Optional[str] = None
    customer_filter: Optional[str] = None
    agent_filter: Optional[str] = None
    duration_range: Optional[Dict[str, int]] = None
    
    # Pagination
    page: int = 1
    page_size: int = 50
    
    # Sorting
    sort_by: str = "started_at"
    sort_order: str = "desc"

class CallSearchResponse(BaseModel):
    """Response model for call search results"""
    calls: List[Dict[str, Any]]
    total_results: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

# Utility functions for call data processing

def calculate_call_quality_score(metrics: CallMetrics) -> float:
    """Calculate overall call quality score (0-100)"""
    score = 0.0
    
    # Audio quality (25%)
    audio_scores = {
        AudioQuality.EXCELLENT: 25,
        AudioQuality.GOOD: 20,
        AudioQuality.FAIR: 12,
        AudioQuality.POOR: 5
    }
    score += audio_scores.get(metrics.audio_quality, 10)
    
    # Conversation flow (25%)
    if metrics.avg_response_time > 0:
        response_score = max(0, 25 - (metrics.avg_response_time - 2) * 5)  # Penalize >2s response
        score += min(25, response_score)
    
    # Technical performance (25%)
    if metrics.latency_avg < 0.5:
        score += 25
    elif metrics.latency_avg < 1.0:
        score += 20
    elif metrics.latency_avg < 2.0:
        score += 15
    else:
        score += 5
    
    # AI performance (25%)
    if metrics.transcription_accuracy > 0:
        score += metrics.transcription_accuracy * 25
    else:
        score += 15  # Default assumption
    
    return min(100.0, score)

def analyze_conversation_patterns(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze conversation patterns from transcript"""
    if not transcript:
        return {}
    
    user_messages = [msg for msg in transcript if msg.get("role") == "user"]
    agent_messages = [msg for msg in transcript if msg.get("role") == "assistant"]
    
    # Calculate basic metrics
    total_user_words = sum(len(msg.get("content", "").split()) for msg in user_messages)
    total_agent_words = sum(len(msg.get("content", "").split()) for msg in agent_messages)
    
    # Conversation balance
    if total_user_words + total_agent_words > 0:
        user_talk_ratio = total_user_words / (total_user_words + total_agent_words)
    else:
        user_talk_ratio = 0.5
    
    # Turn-taking analysis
    speaker_changes = 0
    for i in range(1, len(transcript)):
        if transcript[i].get("role") != transcript[i-1].get("role"):
            speaker_changes += 1
    
    return {
        "total_turns": len(transcript),
        "user_turns": len(user_messages),
        "agent_turns": len(agent_messages),
        "total_user_words": total_user_words,
        "total_agent_words": total_agent_words,
        "user_talk_ratio": user_talk_ratio,
        "speaker_changes": speaker_changes,
        "avg_user_message_length": total_user_words / len(user_messages) if user_messages else 0,
        "avg_agent_message_length": total_agent_words / len(agent_messages) if agent_messages else 0
    }

def detect_call_issues(call_record, metrics: Optional[CallMetrics] = None) -> List[str]:
    """Detect potential issues in a call"""
    issues = []
    
    # Duration-based issues
    if call_record.duration and call_record.duration < 30:
        issues.append("very_short_call")
    elif call_record.duration and call_record.duration > 1800:  # 30 minutes
        issues.append("very_long_call")
    
    # Status-based issues
    if call_record.status.value == "failed":
        issues.append("call_failed")
    elif call_record.status.value == "transferred":
        issues.append("required_transfer")
    
    # Sentiment-based issues
    if call_record.sentiment == "negative":
        issues.append("negative_sentiment")
    
    # Metrics-based issues
    if metrics:
        if metrics.avg_response_time > 5.0:
            issues.append("slow_response_times")
        if metrics.customer_interruptions > 5:
            issues.append("excessive_interruptions")
        if metrics.audio_quality == AudioQuality.POOR:
            issues.append("poor_audio_quality")
        if metrics.transcription_accuracy < 0.7:
            issues.append("poor_transcription")
    
    return issues

def generate_call_insights(call_record, metrics: Optional[CallMetrics] = None) -> Dict[str, Any]:
    """Generate insights and recommendations for a call"""
    insights = {
        "quality_score": 0.0,
        "strengths": [],
        "areas_for_improvement": [],
        "recommendations": [],
        "follow_up_actions": []
    }
    
    # Calculate quality score
    if metrics:
        insights["quality_score"] = calculate_call_quality_score(metrics)
    
    # Analyze transcript patterns
    if call_record.transcript:
        patterns = analyze_conversation_patterns(call_record.transcript)
        
        # Identify strengths
        if patterns.get("user_talk_ratio", 0) > 0.4:
            insights["strengths"].append("Good customer engagement")
        if patterns.get("avg_agent_message_length", 0) < 50:
            insights["strengths"].append("Concise agent responses")
    
    # Detect issues and provide recommendations
    issues = detect_call_issues(call_record, metrics)
    for issue in issues:
        if issue == "slow_response_times":
            insights["areas_for_improvement"].append("Agent response speed")
            insights["recommendations"].append("Consider system optimization for faster processing")
        elif issue == "negative_sentiment":
            insights["areas_for_improvement"].append("Customer satisfaction")
            insights["recommendations"].append("Review conversation for improvement opportunities")
            insights["follow_up_actions"].append("Consider customer satisfaction survey")
    
    # Success indicators
    if call_record.summary and call_record.summary.get("claim_created"):
        insights["strengths"].append("Successfully created customer claim")
    
    return insights