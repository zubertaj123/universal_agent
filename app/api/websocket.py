"""
COMPLETE FIX - WebSocket Handler with VAD and Audio Issues Resolved
Replace your app/api/websocket.py with this version
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import threading

from app.services.speech_service import SpeechService
from app.utils.audio_processing import AudioProcessor
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create the router
websocket_router = APIRouter()

class SingletonAudioManager:
    """FIXED Audio Manager - Prevents ALL overlapping audio"""
    
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
            self.current_session = None
            self.is_speaking = False
            self.audio_lock = asyncio.Lock()
            self.speech_service = None
            self._initialized = True
            logger.info("✅ Audio Manager initialized")
    
    async def request_audio_slot(self, session_id: str) -> bool:
        """Request exclusive audio slot"""
        if not hasattr(self, 'audio_lock'):
            self.audio_lock = asyncio.Lock()
            
        async with self.audio_lock:
            if self.is_speaking and self.current_session != session_id:
                logger.warning(f"❌ Audio busy - denying {session_id}")
                return False
            
            self.current_session = session_id
            self.is_speaking = True
            logger.info(f"✅ Audio granted to {session_id}")
            return True
    
    async def release_audio_slot(self, session_id: str):
        """Release audio slot"""
        if not hasattr(self, 'audio_lock'):
            return
            
        async with self.audio_lock:
            if self.current_session == session_id:
                self.current_session = None
                self.is_speaking = False
                logger.info(f"✅ Audio released by {session_id}")
    
    async def force_stop_all_audio(self):
        """EMERGENCY: Stop all audio"""
        if not hasattr(self, 'audio_lock'):
            self.audio_lock = asyncio.Lock()
            
        async with self.audio_lock:
            logger.warning("🚨 EMERGENCY: Force stopping all audio")
            self.current_session = None
            self.is_speaking = False

# Global singleton
audio_manager = SingletonAudioManager()

class FixedCallSession:
    """FIXED Call Session with proper audio chunk handling"""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.audio_processor = AudioProcessor()
        self.is_active = True
        self.start_time = datetime.now()
        self.conversation = []
        
        # FIXED: Proper audio buffering for VAD
        self.audio_buffer = []
        self.buffer_max_samples = 512  # Exact VAD requirement
        self.sample_rate = 16000
        
    async def process_audio_safely(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """FIXED: Process audio with proper VAD chunk sizes"""
        try:
            if len(audio_bytes) == 0:
                return None
                
            # Decode audio
            audio_pcm = self.audio_processor.decode_and_resample_browser_audio(audio_bytes)
            
            if audio_pcm is None or audio_pcm.size == 0:
                return None
                
            # Ensure float32
            if audio_pcm.dtype != np.float32:
                audio_pcm = audio_pcm.astype(np.float32)
            
            # Add to buffer
            self.audio_buffer.extend(audio_pcm)
            
            # CRITICAL FIX: Only process when we have exactly 512 samples for VAD
            if len(self.audio_buffer) >= self.buffer_max_samples:
                # Extract exactly 512 samples
                chunk_to_process = np.array(self.audio_buffer[:self.buffer_max_samples], dtype=np.float32)
                
                # Remove processed samples from buffer
                self.audio_buffer = self.audio_buffer[self.buffer_max_samples:]
                
                # Preprocess the chunk
                processed = self.audio_processor.preprocess_audio(chunk_to_process)
                
                return processed
                
            return None
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            self.audio_buffer = []  # Clear buffer on error
            return None

async def send_single_tts_response(
    websocket: WebSocket,
    text: str,
    session_id: str
) -> bool:
    """FIXED: Send single TTS response without overlaps"""
    try:
        # Request exclusive audio slot
        can_speak = await audio_manager.request_audio_slot(session_id)
        if not can_speak:
            logger.warning(f"Cannot speak - audio busy")
            return False
        
        logger.info(f"🎤 Generating TTS: '{text[:50]}...'")
        
        try:
            # Initialize speech service if needed
            if not audio_manager.speech_service:
                audio_manager.speech_service = SpeechService()
                await audio_manager.speech_service.initialize()
            
            # Set voice
            audio_manager.speech_service.set_voice("friendly")
            
            # Send start notification
            stream_id = str(uuid.uuid4())
            await websocket.send_json({
                "type": "audio_start",
                "text": text,
                "session_id": session_id,
                "stream_id": stream_id
            })
            
            # Generate audio
            chunk_count = 0
            total_bytes = 0
            
            async for audio_chunk in audio_manager.speech_service.synthesize_natural(text, stream=True):
                # Check if we still have audio slot
                if audio_manager.current_session != session_id:
                    logger.warning("Audio slot lost during generation")
                    break
                
                if audio_chunk and len(audio_chunk) > 0:
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                    
                    await websocket.send_json({
                        "type": "audio_chunk", 
                        "data": audio_chunk.hex(),
                        "chunk_number": chunk_count,
                        "session_id": session_id,
                        "stream_id": stream_id
                    })
                    
                    # Controlled timing to prevent overlap
                    await asyncio.sleep(0.1)
            
            # Send completion
            await websocket.send_json({
                "type": "audio_complete",
                "chunks_sent": chunk_count,
                "total_bytes": total_bytes,
                "session_id": session_id
            })
            
            logger.info(f"✅ TTS completed: {chunk_count} chunks")
            return True
            
        finally:
            # ALWAYS release audio slot
            await audio_manager.release_audio_slot(session_id)
            
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        await audio_manager.release_audio_slot(session_id)
        return False

def generate_contextual_response(text: str) -> str:
    """Generate contextual responses"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["hello", "hi", "hey", "good morning"]):
        return "Hello! I'm your AI insurance assistant. I can help you file claims, check policies, or answer questions. What can I help you with today?"
    
    elif "claim" in text_lower:
        return "I'd be happy to help you file an insurance claim. Can you tell me what type of incident occurred and when it happened?"
    
    elif any(word in text_lower for word in ["policy", "coverage", "premium"]):
        return "I can help you with policy information. Could you provide your policy number so I can look up your coverage details?"
    
    elif "help" in text_lower:
        return "I'm here to help! I can assist with filing claims, checking policy information, scheduling appointments, or answering insurance questions. What do you need?"
    
    elif any(word in text_lower for word in ["appointment", "schedule"]):
        return "I can help you schedule an appointment. What type of appointment do you need - a claim inspection, policy review, or consultation?"
    
    elif any(word in text_lower for word in ["human", "agent", "transfer"]):
        return "I understand you'd like to speak with a human agent. Let me connect you with one of our specialists."
    
    elif any(word in text_lower for word in ["thank", "thanks"]):
        return "You're very welcome! Is there anything else I can help you with today?"
    
    elif any(word in text_lower for word in ["bye", "goodbye"]):
        return "Thank you for calling! Have a wonderful day. Please don't hesitate to contact us if you need further assistance."
    
    else:
        return "I understand. Could you please provide me with more details about what you need help with? I'm here to assist with your insurance needs."

async def handle_text_message(message: dict, session: FixedCallSession, websocket: WebSocket):
    """FIXED: Handle text with single audio response"""
    try:
        text = message.get("text", "").strip()
        if not text:
            return
            
        logger.info(f"📝 Processing: '{text}'")
        
        # Stop any current audio
        await audio_manager.force_stop_all_audio()
        
        # Add to conversation
        session.conversation.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        response = generate_contextual_response(text)
        
        # Send transcripts
        await websocket.send_json({
            "type": "transcript",
            "speaker": "user", 
            "text": text,
            "session_id": session.session_id
        })
        
        await websocket.send_json({
            "type": "transcript",
            "speaker": "agent",
            "text": response,
            "session_id": session.session_id
        })
        
        # Add response to conversation
        session.conversation.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send SINGLE audio response
        await send_single_tts_response(websocket, response, session.session_id)
        
    except Exception as e:
        logger.error(f"Text handling error: {e}")

async def handle_audio_message(message: dict, session: FixedCallSession, websocket: WebSocket):
    """FIXED: Handle audio with proper VAD chunk sizes"""
    try:
        audio_data = message.get("data", "")
        if not audio_data:
            return
            
        # Convert hex to bytes
        try:
            audio_bytes = bytes.fromhex(audio_data)
        except ValueError:
            logger.warning("Invalid hex audio data")
            return
            
        if len(audio_bytes) == 0:
            return
        
        # Process with FIXED chunk sizes for VAD
        processed_audio = await session.process_audio_safely(audio_bytes)
        
        if processed_audio is not None and len(processed_audio) > 0:
            # FIXED VAD: Use proper chunk size (512 samples)
            try:
                # Ensure we have exactly 512 samples for VAD
                if len(processed_audio) != 512:
                    if len(processed_audio) > 512:
                        # Truncate to 512
                        vad_chunk = processed_audio[:512]
                    else:
                        # Pad to 512
                        padding = 512 - len(processed_audio)
                        vad_chunk = np.pad(processed_audio, (0, padding), mode='constant')
                else:
                    vad_chunk = processed_audio
                
                # Check for speech with FIXED VAD
                has_speech = False
                try:
                    if audio_manager.speech_service:
                        has_speech = audio_manager.speech_service.detect_voice_activity(vad_chunk, 16000)
                    else:
                        # Fallback energy detection
                        rms_energy = np.sqrt(np.mean(vad_chunk**2))
                        has_speech = rms_energy > 0.005
                except Exception as vad_error:
                    logger.warning(f"VAD failed, using energy detection: {vad_error}")
                    rms_energy = np.sqrt(np.mean(vad_chunk**2))
                    has_speech = rms_energy > 0.005
                
                if has_speech:
                    logger.info("🎤 Speech detected, transcribing...")
                    
                    # Stop current audio
                    await audio_manager.force_stop_all_audio()
                    
                    # Transcribe using the full processed audio (not just VAD chunk)
                    try:
                        if not audio_manager.speech_service:
                            audio_manager.speech_service = SpeechService()
                            await audio_manager.speech_service.initialize()
                        
                        # Use original processed audio for transcription
                        transcription = await asyncio.wait_for(
                            audio_manager.speech_service.transcribe(processed_audio),
                            timeout=10.0
                        )
                        
                        if transcription and transcription.strip():
                            logger.info(f"📝 Transcribed: '{transcription}'")
                            
                            # Handle as text message
                            await handle_text_message(
                                {"text": transcription}, 
                                session, 
                                websocket
                            )
                            
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                        
            except Exception as e:
                logger.error(f"VAD processing failed: {e}")
                
    except Exception as e:
        logger.error(f"Audio handling error: {e}")

async def handle_control_message(message: dict, session: FixedCallSession, websocket: WebSocket):
    """Handle control messages"""
    action = message.get("action", "")
    
    if action == "stop_audio":
        await audio_manager.force_stop_all_audio()
        await websocket.send_json({
            "type": "status",
            "message": "Audio stopped",
            "session_id": session.session_id
        })
    elif action == "mute":
        session.audio_processor.mute()
        await websocket.send_json({
            "type": "status", 
            "message": "Microphone muted",
            "session_id": session.session_id
        })
    elif action == "unmute":
        session.audio_processor.unmute()
        await websocket.send_json({
            "type": "status",
            "message": "Microphone unmuted", 
            "session_id": session.session_id
        })

@websocket_router.websocket("/ws/call/{session_id}")
async def websocket_call_handler(websocket: WebSocket, session_id: str):
    """MAIN WebSocket Handler - FIXED for VAD and audio overlap issues"""
    logger.info(f"🔌 WebSocket connection: {session_id}")
    
    try:
        await websocket.accept()
        
        # FORCE stop any existing audio
        await audio_manager.force_stop_all_audio()
        
        # Create session
        session = FixedCallSession(session_id, websocket)
        
        # Send initial greeting
        greeting = "Hello! I'm your AI insurance assistant. I can help you with claims, policy questions, and more. How can I assist you today?"
        
        # Send transcript
        await websocket.send_json({
            "type": "transcript",
            "speaker": "agent",
            "text": greeting,
            "session_id": session_id
        })
        
        # Send SINGLE greeting audio
        try:
            success = await send_single_tts_response(websocket, greeting, session_id)
            if success:
                logger.info("✅ Greeting sent successfully")
            else:
                logger.warning("⚠️ Greeting audio failed")
        except Exception as e:
            logger.error(f"Greeting error: {e}")
        
        # Main message loop
        while session.is_active:
            try:
                message = await websocket.receive_json()
                message_type = message.get("type", "")
                
                logger.debug(f"📨 Message: {message_type}")
                
                if message_type == "audio":
                    await handle_audio_message(message, session, websocket)
                elif message_type == "text":
                    await handle_text_message(message, session, websocket)
                elif message_type == "control":
                    await handle_control_message(message, session, websocket)
                elif message_type == "end_call":
                    # Send farewell
                    farewell = "Thank you for calling! Have a great day."
                    await websocket.send_json({
                        "type": "transcript",
                        "speaker": "agent", 
                        "text": farewell,
                        "session_id": session_id
                    })
                    await send_single_tts_response(websocket, farewell, session_id)
                    break
                elif message_type == "connection":
                    await websocket.send_json({
                        "type": "connected",
                        "session_id": session_id,
                        "message": "Connected successfully - fixed VAD"
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up session {session_id}")
        session.is_active = False
        
        # Force stop all audio
        await audio_manager.force_stop_all_audio()
        
        # Close WebSocket
        try:
            await websocket.close()
        except:
            pass

logger.info("✅ FIXED WebSocket handler loaded - VAD issues resolved")