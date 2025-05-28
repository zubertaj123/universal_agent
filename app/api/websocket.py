"""
Complete WebSocket Handler with Audio Stream Management
Replace your entire app/api/websocket.py with this file
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import traceback
import threading

from app.services.speech_service import SpeechService
from app.services.llm_service import LLMService
from app.agents.call_agent import CallCenterAgent, AgentState
from app.models.conversation import ConversationManager
from app.utils.audio_processing import AudioProcessor
from app.utils.logger import setup_logger
from app.core.database import get_session
from app.models.conversation import CallRecord, CallStatus
from app.models.user import User
from sqlalchemy import select

logger = setup_logger(__name__)

# Create the router
websocket_router = APIRouter()

class AudioStreamManager:
    """Singleton manager to prevent multiple simultaneous audio streams"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.active_streams: Dict[str, Dict] = {}
            self.session_streams: Dict[str, set] = {}
            self.global_lock = asyncio.Lock()
            self._initialized = True
            logger.info("AudioStreamManager initialized as singleton")
    
    async def register_stream(self, session_id: str, stream_type: str = "tts") -> str:
        """Register a new audio stream and ensure no conflicts"""
        if not hasattr(self, 'global_lock'):
            self.global_lock = asyncio.Lock()
            
        async with self.global_lock:
            stream_id = str(uuid.uuid4())
            
            # Stop any existing streams for this session
            await self._stop_session_streams(session_id)
            
            # Register new stream
            self.active_streams[stream_id] = {
                "session_id": session_id,
                "type": stream_type,
                "created_at": datetime.now(),
                "status": "active"
            }
            
            if session_id not in self.session_streams:
                self.session_streams[session_id] = set()
            self.session_streams[session_id].add(stream_id)
            
            logger.info(f"Registered audio stream {stream_id} for session {session_id}")
            return stream_id
    
    async def _stop_session_streams(self, session_id: str):
        """Stop all active streams for a session"""
        if session_id in self.session_streams:
            stream_ids = list(self.session_streams[session_id])
            for stream_id in stream_ids:
                await self.unregister_stream(stream_id)
            if stream_ids:
                logger.info(f"Stopped {len(stream_ids)} streams for session {session_id}")
    
    async def unregister_stream(self, stream_id: str):
        """Unregister an audio stream"""
        if not hasattr(self, 'global_lock'):
            self.global_lock = asyncio.Lock()
            
        async with self.global_lock:
            if stream_id in self.active_streams:
                stream_info = self.active_streams[stream_id]
                session_id = stream_info["session_id"]
                
                # Remove from active streams
                del self.active_streams[stream_id]
                
                # Remove from session streams
                if session_id in self.session_streams:
                    self.session_streams[session_id].discard(stream_id)
                    if not self.session_streams[session_id]:
                        del self.session_streams[session_id]
                
                logger.debug(f"Unregistered audio stream {stream_id}")
    
    async def is_session_playing(self, session_id: str) -> bool:
        """Check if any audio is currently playing for a session"""
        return (session_id in self.session_streams and 
                len(self.session_streams[session_id]) > 0)
    
    async def stop_all_streams(self):
        """Emergency stop all audio streams"""
        if not hasattr(self, 'global_lock'):
            self.global_lock = asyncio.Lock()
            
        async with self.global_lock:
            stream_count = len(self.active_streams)
            self.active_streams.clear()
            self.session_streams.clear()
            if stream_count > 0:
                logger.warning(f"Emergency stopped {stream_count} audio streams")
    
    async def get_active_streams(self) -> Dict:
        """Get information about all active streams"""
        return {
            "total_streams": len(self.active_streams),
            "sessions_with_audio": len(self.session_streams),
            "streams": list(self.active_streams.keys()),
            "session_breakdown": {
                session_id: len(streams) 
                for session_id, streams in self.session_streams.items()
            }
        }

# Global singleton instance
audio_manager = AudioStreamManager()

class CallSession:
    """Manages a single call session with audio stream management"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.audio_processor = AudioProcessor()
        self.conversation_manager = ConversationManager()
        self.audio_buffer = []
        self.is_active = True
        self.call_record_id = None
        self.customer_id = None
        self.start_time = datetime.now()
        self.last_audio_time = datetime.now()
        self.audio_chunk_count = 0
        
    async def handle_audio_stream(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Process incoming audio from browser"""
        try:
            self.audio_chunk_count += 1
            self.last_audio_time = datetime.now()
            
            if self.audio_chunk_count % 100 == 0:
                logger.debug(f"Processing audio chunk #{self.audio_chunk_count}, size: {len(audio_bytes)} bytes")
            
            try:
                audio_pcm = self.audio_processor.decode_and_resample_browser_audio(audio_bytes)
            except Exception as e:
                logger.error(f"Audio decoding failed: {e}")
                return None

            if audio_pcm is None or audio_pcm.size == 0:
                logger.debug("No audio data returned from decoder")
                return None

            if audio_pcm.dtype != np.float32:
                audio_pcm = audio_pcm.astype(np.float32)

            self.audio_buffer.append(audio_pcm)
            
            try:
                buffered = np.concatenate(self.audio_buffer) if self.audio_buffer else audio_pcm
            except ValueError as e:
                logger.error(f"Audio buffer concatenation failed: {e}")
                self.audio_buffer = [audio_pcm]
                buffered = audio_pcm

            min_chunk_size = 512
            if len(buffered) >= min_chunk_size:
                try:
                    chunk_to_process = buffered[:min_chunk_size]
                    
                    if len(buffered) > min_chunk_size:
                        self.audio_buffer = [buffered[min_chunk_size:]]
                    else:
                        self.audio_buffer = []
                    
                    processed_audio = self.audio_processor.preprocess_audio(chunk_to_process)
                    return processed_audio
                    
                except Exception as e:
                    logger.error(f"Audio chunk processing failed: {e}")
                    self.audio_buffer = []
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")
            self.audio_buffer = []
            return None

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics for monitoring"""
        return {
            "session_id": self.session_id,
            "is_active": self.is_active,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "audio_chunks_processed": self.audio_chunk_count,
            "last_audio_time": self.last_audio_time.isoformat(),
            "buffer_size": len(self.audio_buffer),
            "call_record_id": self.call_record_id
        }

    async def create_call_record(self):
        """Create database record for this call"""
        try:
            async for session in get_session():
                self.call_record_id = str(uuid.uuid4())
                
                call_record = CallRecord(
                    id=self.call_record_id,
                    session_id=self.session_id,
                    status=CallStatus.ACTIVE,
                    started_at=self.start_time,
                    transcript=[],
                    call_metadata={"websocket_session": True}
                )
                
                session.add(call_record)
                await session.commit()
                
                logger.info(f"Created call record {self.call_record_id} for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Failed to create call record: {e}")
    
    async def update_call_record(self, **kwargs):
        """Update call record with new information"""
        if not self.call_record_id:
            return
            
        try:
            async for session in get_session():
                query = select(CallRecord).where(CallRecord.id == self.call_record_id)
                result = await session.execute(query)
                call_record = result.scalar_one_or_none()
                
                if call_record:
                    for key, value in kwargs.items():
                        if hasattr(call_record, key):
                            setattr(call_record, key, value)
                    
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update call record: {e}")

async def generate_and_send_natural_tts(
    websocket: WebSocket,
    speech_service: SpeechService,
    text: str,
    session_id: str,
    voice_style: str = "friendly"
) -> bool:
    """
    Generate and send TTS audio with proper stream management
    """
    try:
        if not text.strip():
            return False
            
        logger.info(f"Generating TTS for session {session_id}: '{text[:50]}...'")
        
        # Register and manage audio stream
        stream_id = await audio_manager.register_stream(session_id, "tts")
        
        # Check if another stream is already playing
        active_streams = await audio_manager.get_active_streams()
        if active_streams["total_streams"] > 1:
            logger.warning(f"Multiple streams detected: {active_streams}")
            # Stop other streams for this session
            await audio_manager._stop_session_streams(session_id)
            # Re-register this stream
            stream_id = await audio_manager.register_stream(session_id, "tts")
        
        try:
            # Set voice with controlled settings
            speech_service.set_voice(voice_style)
            
            # Send audio start notification
            await websocket.send_json({
                "type": "audio_start",
                "text": text,
                "voice": voice_style,
                "stream_id": stream_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate audio with controlled streaming
            chunk_count = 0
            total_bytes = 0
            
            async for audio_chunk in speech_service.synthesize_natural(text, stream=True):
                # Check if stream is still registered (not cancelled)
                if stream_id not in audio_manager.active_streams:
                    logger.info(f"Stream {stream_id} was cancelled, stopping generation")
                    break
                
                if audio_chunk and len(audio_chunk) > 0:
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": audio_chunk.hex(),
                        "chunk_number": chunk_count,
                        "stream_id": stream_id,
                        "session_id": session_id,
                        "format": "mp3",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Controlled timing to prevent audio overlap
                    await asyncio.sleep(0.05)
            
            # Send completion signal
            await websocket.send_json({
                "type": "audio_complete",
                "chunks_sent": chunk_count,
                "total_bytes": total_bytes,
                "stream_id": stream_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"TTS completed for session {session_id}: {chunk_count} chunks")
            return True
            
        finally:
            # Always unregister the stream when done
            await audio_manager.unregister_stream(stream_id)
            
    except Exception as e:
        logger.error(f"TTS generation failed for session {session_id}: {e}")
        
        # Clean up on error
        try:
            if 'stream_id' in locals():
                await audio_manager.unregister_stream(stream_id)
        except:
            pass
        
        # Send error notification
        await websocket.send_json({
            "type": "audio_error",
            "message": "Audio generation failed",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        return False

def generate_simple_response(text: str) -> str:
    """Generate contextual responses based on input"""
    text_lower = text.lower()
    
    if "claim" in text_lower:
        return "I'd be happy to help you with your insurance claim. Could you please provide me with your policy number and details about the incident?"
    elif any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "Hello! Thank you for calling. I'm here to help you with your insurance needs. What can I assist you with today?"
    elif "help" in text_lower:
        return "I'm here to help you with insurance claims, policy questions, and general support. What specific assistance do you need?"
    elif any(word in text_lower for word in ["policy", "coverage", "premium"]):
        return "I can help you with policy information. Could you please provide your policy number so I can look up your coverage details?"
    elif any(word in text_lower for word in ["appointment", "schedule", "meeting"]):
        return "I'd be happy to help you schedule an appointment. What type of appointment would you like to schedule, and what's your preferred date and time?"
    elif any(word in text_lower for word in ["transfer", "human", "agent", "person"]):
        return "I understand you'd like to speak with a human agent. Let me transfer you to one of our specialists who can assist you further."
    elif any(word in text_lower for word in ["thank", "thanks"]):
        return "You're very welcome! Is there anything else I can help you with today?"
    elif any(word in text_lower for word in ["bye", "goodbye", "end"]):
        return "Thank you for calling! Have a wonderful day. If you need any further assistance, please don't hesitate to call us again."
    else:
        return "I understand. Let me help you with that. Could you please provide me with more details so I can assist you better?"

async def handle_websocket_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Handle incoming WebSocket messages with audio stream management"""
    message_type = message.get("type")
    
    logger.debug(f"Received WebSocket message: {message_type}")
    
    if message_type == "audio":
        await handle_audio_message(message, session, agent_state, speech_service, websocket)
    elif message_type == "text":
        await handle_text_message(message, session, agent_state, speech_service, websocket)
    elif message_type == "control":
        await handle_control_message(message, session, websocket)
    elif message_type == "end_call":
        await handle_end_call(message, session, agent_state, websocket)
    elif message_type == "stop_audio":
        await handle_stop_audio(message, session, websocket)
    elif message_type == "connection":
        await handle_connection_message(message, session, websocket)
    elif message_type == "test":
        await handle_test_message(message, session, websocket)
    else:
        logger.warning(f"Unknown message type: {message_type}")
        await send_message_to_client(websocket, "error", {
            "message": f"Unknown message type: {message_type}"
        })

async def handle_text_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Handle text message with natural audio response"""
    try:
        text = message.get("text", "").strip()
        if not text:
            return
            
        logger.info(f"Processing text message for session {session.session_id}: '{text}'")
        
        # Stop any currently playing audio first
        await audio_manager._stop_session_streams(session.session_id)
        
        # Add to conversation
        agent_state["messages"].append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate contextual response
        response = generate_simple_response(text)
        
        # Send transcript
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": response,
            "session_id": session.session_id
        })
        
        # Add agent response to conversation
        agent_state["messages"].append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate TTS response with proper stream management
        success = await generate_and_send_natural_tts(
            websocket, 
            speech_service, 
            response, 
            session.session_id,
            "friendly"
        )
        
        if success:
            logger.info("Natural audio response sent successfully")
        else:
            logger.error("Failed to send natural audio response")
        
        # Update call record
        try:
            await session.update_call_record(transcript=agent_state["messages"])
        except Exception as e:
            logger.error(f"Failed to update call record: {e}")
        
    except Exception as e:
        logger.error(f"Error handling text message: {e}")

async def handle_audio_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Handle audio data from client"""
    try:
        audio_data = message.get("data", "")
        if not audio_data:
            logger.debug("Empty audio data received")
            return
            
        try:
            audio_bytes = bytes.fromhex(audio_data)
        except ValueError as e:
            logger.warning(f"Invalid hex audio data: {e}")
            return
        
        if len(audio_bytes) == 0:
            logger.debug("Zero-length audio data")
            return
        
        logger.debug(f"Processing audio chunk: {len(audio_bytes)} bytes")
            
        try:
            processed_audio = await session.handle_audio_stream(audio_bytes)
        except Exception as e:
            logger.error(f"Audio stream processing failed: {e}")
            processed_audio = None
        
        # Only proceed with transcription if we have valid audio
        if processed_audio is not None and len(processed_audio) > 0:
            try:
                if processed_audio.dtype != np.float32:
                    processed_audio = processed_audio.astype(np.float32)
                
                has_speech = False
                try:
                    has_speech = speech_service.detect_voice_activity(processed_audio)
                except Exception as e:
                    logger.warning(f"VAD failed: {e}, using basic energy detection")
                    rms_energy = np.sqrt(np.mean(processed_audio**2))
                    has_speech = rms_energy > 0.005
                
                if has_speech:
                    logger.info("Speech detected, attempting transcription")
                    
                    # Stop any currently playing audio
                    await audio_manager._stop_session_streams(session.session_id)
                    
                    transcription = None
                    try:
                        transcription = await asyncio.wait_for(
                            speech_service.transcribe(processed_audio),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Transcription timeout")
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                    
                    if transcription and transcription.strip():
                        logger.info(f"Transcribed: '{transcription}'")
                        
                        await send_message_to_client(websocket, "transcript", {
                            "speaker": "user",
                            "text": transcription,
                            "session_id": session.session_id
                        })
                        
                        agent_state["messages"].append({
                            "role": "user",
                            "content": transcription,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Generate and send response
                        response = generate_simple_response(transcription)
                        
                        await send_message_to_client(websocket, "transcript", {
                            "speaker": "agent",
                            "text": response,
                            "session_id": session.session_id
                        })
                        
                        agent_state["messages"].append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Generate TTS response
                        await generate_and_send_natural_tts(
                            websocket,
                            speech_service,
                            response,
                            session.session_id,
                            "friendly"
                        )
                        
                        # Update call record
                        try:
                            await session.update_call_record(transcript=agent_state["messages"])
                        except Exception as e:
                            logger.error(f"Failed to update call record: {e}")
                        
                    else:
                        logger.debug("No transcription result or empty text")
                else:
                    logger.debug("No speech detected in audio chunk")
            except Exception as e:
                logger.error(f"Speech processing failed: {e}")
        else:
            logger.debug("No valid audio data to process")
            
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")

async def handle_control_message(message: dict, session: CallSession, websocket: WebSocket):
    """Handle control messages (mute, pause, etc.)"""
    try:
        action = message.get("action")
        
        if action == "mute":
            session.audio_processor.mute()
            await send_message_to_client(websocket, "status", {
                "message": "Microphone muted",
                "session_id": session.session_id
            })
        elif action == "unmute":
            session.audio_processor.unmute()
            await send_message_to_client(websocket, "status", {
                "message": "Microphone unmuted",
                "session_id": session.session_id
            })
        elif action == "pause":
            session.is_active = False
            await send_message_to_client(websocket, "status", {
                "message": "Call paused",
                "session_id": session.session_id
            })
        elif action == "resume":
            session.is_active = True
            await send_message_to_client(websocket, "status", {
                "message": "Call resumed",
                "session_id": session.session_id
            })
        else:
            logger.warning(f"Unknown control action: {action}")
            
    except Exception as e:
        logger.error(f"Error handling control message: {e}")

async def handle_stop_audio(message: dict, session: CallSession, websocket: WebSocket):
    """Handle request to stop current audio"""
    try:
        logger.info(f"Stopping audio for session {session.session_id}")
        await audio_manager._stop_session_streams(session.session_id)
        
        await send_message_to_client(websocket, "audio_stopped", {
            "session_id": session.session_id,
            "message": "Audio stopped"
        })
        
    except Exception as e:
        logger.error(f"Error stopping audio: {e}")

async def handle_end_call(message: dict, session: CallSession, agent_state: AgentState, websocket: WebSocket):
    """Handle call end request"""
    try:
        logger.info(f"Call end requested for session {session.session_id}")
        
        # Stop all audio first
        await audio_manager._stop_session_streams(session.session_id)
        
        agent_state["call_status"] = "ending"
        session.is_active = False
        
        farewell = "Thank you for calling. Have a great day!"
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": farewell,
            "session_id": session.session_id
        })
        
        # Send farewell with TTS
        await generate_and_send_natural_tts(
            websocket,
            SpeechService(),  # Create new instance for farewell
            farewell,
            session.session_id,
            "friendly"
        )
        
        await send_message_to_client(websocket, "status", {
            "message": "Call ending...",
            "session_id": session.session_id
        })
        
    except Exception as e:
        logger.error(f"Error handling end call: {e}")

async def handle_connection_message(message: dict, session: CallSession, websocket: WebSocket):
    """Handle connection initialization message"""
    try:
        session_id = message.get("sessionId", session.session_id)
        timestamp = message.get("timestamp")
        
        logger.info(f"Connection message received from session: {session_id}")
        
        await send_message_to_client(websocket, "connected", {
            "sessionId": session_id,
            "status": "acknowledged",
            "server_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling connection message: {e}")

async def handle_test_message(message: dict, session: CallSession, websocket: WebSocket):
    """Handle test messages for debugging"""
    try:
        logger.info(f"Test message received: {message}")
        
        await send_message_to_client(websocket, "test_response", {
            "original_message": message,
            "server_time": datetime.now().isoformat(),
            "session_id": session.session_id,
            "audio_status": await audio_manager.get_active_streams()
        })
        
    except Exception as e:
        logger.error(f"Error handling test message: {e}")

async def send_message_to_client(websocket: WebSocket, message_type: str, data: dict):
    """Send message to WebSocket client"""
    try:
        message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        await websocket.send_json(message)
    except Exception as e:
        logger.error(f"Error sending message to client: {e}")

async def finalize_call_record(session: CallSession, agent_state: AgentState):
    """Finalize call record when call ends"""
    try:
        if not session.call_record_id:
            return
            
        end_time = datetime.now()
        duration = int((end_time - session.start_time).total_seconds())
        
        customer_info = agent_state.get("customer_info", {})
        claim_data = agent_state.get("claim_data", {})
        
        emotions = agent_state.get("intent_history", [])
        overall_sentiment = "neutral"
        if "frustrated" in emotions:
            overall_sentiment = "negative"
        elif "satisfied" in emotions:
            overall_sentiment = "positive"
        
        summary = {
            "total_messages": len(agent_state.get("messages", [])),
            "customer_identified": bool(customer_info),
            "claim_created": bool(claim_data),
            "tools_used": [t.get("tool") for t in agent_state.get("tools_output", [])],
            "emotions_detected": list(set(emotions)),
            "resolution_status": "completed" if claim_data else "information_provided",
            "audio_streams_used": len(audio_manager.session_streams.get(session.session_id, set()))
        }
        
        await session.update_call_record(
            status=CallStatus.ENDED,
            ended_at=end_time,
            duration=duration,
            transcript=agent_state.get("messages", []),
            summary=summary,
            sentiment=overall_sentiment,
            customer_phone=customer_info.get("phone"),
            customer_email=customer_info.get("email"),
            customer_id=customer_info.get("id")
        )
        
        logger.info(f"Finalized call record {session.call_record_id} - Duration: {duration}s")
        
    except Exception as e:
        logger.error(f"Error finalizing call record: {e}")

@websocket_router.websocket("/ws/call/{session_id}")
async def websocket_call(
    websocket: WebSocket,
    session_id: Optional[str] = None
):
    """Main WebSocket endpoint for call handling with audio stream management"""
    if not session_id:
        session_id = str(uuid.uuid4())
        
    logger.info(f"WebSocket connection for session {session_id}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket accepted for session {session_id}")
        
        # CRITICAL: Stop any existing streams for this session
        await audio_manager._stop_session_streams(session_id)
        
        # Initialize services
        speech_service = SpeechService()
        llm_service = LLMService()
        agent = CallCenterAgent(llm_service)
        
        # Create call session
        session = CallSession(session_id, websocket)
        
        # Initialize speech service
        try:
            logger.info("Initializing speech service...")
            await speech_service.initialize()
            logger.info("Speech service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize speech service: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "Speech service initialization failed"
            })
            await websocket.close()
            return
        
        # Create call record
        await session.create_call_record()
        
        # Initialize agent state
        agent_state: AgentState = {
            "messages": [],
            "current_speaker": "agent",
            "call_status": "active",
            "customer_info": {},
            "claim_data": {},
            "emotion": "neutral",  
            "tools_output": [],
            "next_action": "greet",
            "audio_queue": asyncio.Queue(),
            "context": {},
            "intent_history": []
        }
        
        # Send controlled initial greeting
        greeting = "Hello! Thank you for calling. I'm your AI assistant and I can help you with insurance claims, policy questions, or connect you with a human agent. What can I help you with today?"
        
        logger.info("Sending initial greeting...")
        
        # Send transcript first
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": greeting,
            "session_id": session_id
        })
        
        # Add to conversation
        agent_state["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate TTS greeting with proper stream management
        try:
            logger.info("Generating natural greeting audio...")
            success = await generate_and_send_natural_tts(
                websocket, 
                speech_service, 
                greeting, 
                session_id,
                "friendly"
            )
            
            if success:
                logger.info("✅ Greeting audio sent successfully")
            else:
                logger.error("❌ Failed to send greeting audio")
                # Send fallback message
                await websocket.send_json({
                    "type": "status",
                    "message": "Connected - audio system ready",
                    "session_id": session_id
                })
                
        except Exception as greeting_error:
            logger.error(f"Greeting audio generation failed: {greeting_error}")
            
            # Send text-only greeting as fallback
            await websocket.send_json({
                "type": "status", 
                "message": "Connected - please type your message or speak",
                "session_id": session_id
            })
        
        # Background tasks list
        tasks = []
        
        # Start background agent loop (simplified for this version)
        # agent_task = asyncio.create_task(
        #     run_agent_loop(agent, agent_state, session, speech_service, websocket)
        # )
        # tasks.append(agent_task)
        
        # Main message loop
        while session.is_active:
            try:
                message = await websocket.receive_json()
                await handle_websocket_message(
                    message, session, agent_state, speech_service, websocket
                )
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await send_message_to_client(websocket, "error", {
                    "message": "Invalid message format",
                    "session_id": session_id
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await send_message_to_client(websocket, "error", {
                    "message": "Error processing your request",
                    "session_id": session_id
                })
                
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        
    finally:
        # CRITICAL: Clean up audio streams
        logger.info(f"Cleaning up session {session_id}")
        session.is_active = False
        
        # Stop all audio streams for this session
        await audio_manager._stop_session_streams(session_id)
        
        # Cancel background tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update final call record
        await finalize_call_record(session, agent_state)
        
        # Close WebSocket
        try:
            await websocket.close()
        except:
            pass

# Additional WebSocket endpoints for testing and management

@websocket_router.websocket("/ws/test/{session_id}")
async def websocket_test(websocket: WebSocket, session_id: str):
    """Test WebSocket endpoint for development"""
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Test WebSocket connection established",
            "audio_streams": await audio_manager.get_active_streams()
        })
        
        while True:
            message = await websocket.receive_json()
            
            response = {
                "type": "echo",
                "original_message": message,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "audio_status": await audio_manager.get_active_streams()
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"Test WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")

@websocket_router.websocket("/ws/admin/audio-status")
async def websocket_audio_admin(websocket: WebSocket):
    """Admin WebSocket for monitoring audio streams"""
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "audio_status",
            "streams": await audio_manager.get_active_streams(),
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            try:
                # Wait for commands or send periodic updates
                message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
                
                if message.get("type") == "get_status":
                    await websocket.send_json({
                        "type": "audio_status",
                        "streams": await audio_manager.get_active_streams(),
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "stop_all":
                    await audio_manager.stop_all_streams()
                    await websocket.send_json({
                        "type": "all_stopped",
                        "message": "All audio streams stopped",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send periodic status update
                await websocket.send_json({
                    "type": "audio_status",
                    "streams": await audio_manager.get_active_streams(),
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("Audio admin WebSocket disconnected")
    except Exception as e:
        logger.error(f"Audio admin WebSocket error: {e}")

# Utility functions for testing

async def emergency_stop_all_audio():
    """Emergency function to stop all audio streams"""
    try:
        await audio_manager.stop_all_streams()
        logger.warning("Emergency audio stop executed")
        return True
    except Exception as e:
        logger.error(f"Emergency audio stop failed: {e}")
        return False

async def get_audio_debug_info():
    """Get debugging information about audio streams"""
    try:
        return {
            "active_streams": await audio_manager.get_active_streams(),
            "manager_status": {
                "initialized": hasattr(audio_manager, '_initialized'),
                "has_lock": hasattr(audio_manager, 'global_lock'),
                "total_sessions": len(audio_manager.session_streams),
                "total_streams": len(audio_manager.active_streams)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting audio debug info: {e}")
        return {"error": str(e)}

# Export everything needed
__all__ = [
    "websocket_router", 
    "audio_manager", 
    "emergency_stop_all_audio",
    "get_audio_debug_info"
]

logger.info("Complete WebSocket handler with audio stream management loaded")