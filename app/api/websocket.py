"""
WebSocket endpoints for real-time communication
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import traceback

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

websocket_router = APIRouter()

class CallSession:
    """Manages a single call session - Updated version"""
    
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
        
    async def handle_audio_stream(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Process incoming audio from browser - Improved version"""
        try:
            self.audio_chunk_count += 1
            self.last_audio_time = datetime.now()
            
            # Log audio data info for debugging
            if self.audio_chunk_count % 100 == 0:  # Log every 100th chunk
                logger.debug(f"Processing audio chunk #{self.audio_chunk_count}, size: {len(audio_data)} bytes")
            
            # Decode and resample browser audio with improved error handling
            try:
                audio_pcm = self.audio_processor.decode_and_resample_browser_audio(audio_data)
            except Exception as e:
                logger.error(f"Audio decoding failed: {e}")
                return None

            if audio_pcm is None or audio_pcm.size == 0:
                logger.debug("No audio data returned from decoder")
                return None

            # Buffer audio until we have enough for processing
            self.audio_buffer.append(audio_pcm)
            
            # Combine buffered audio
            try:
                buffered = np.concatenate(self.audio_buffer) if self.audio_buffer else audio_pcm
            except ValueError as e:
                logger.error(f"Audio buffer concatenation failed: {e}")
                self.audio_buffer = [audio_pcm]  # Reset buffer
                buffered = audio_pcm

            # Process in chunks for better real-time performance
            min_chunk_size = 512  # Minimum samples to process
            if len(buffered) >= min_chunk_size:
                try:
                    # Take the first chunk for processing
                    chunk_to_process = buffered[:min_chunk_size]
                    
                    # Keep remaining audio in buffer
                    if len(buffered) > min_chunk_size:
                        self.audio_buffer = [buffered[min_chunk_size:]]
                    else:
                        self.audio_buffer = []
                    
                    # Preprocess the audio chunk
                    processed_audio = self.audio_processor.preprocess_audio(chunk_to_process)
                    
                    return processed_audio
                    
                except Exception as e:
                    logger.error(f"Audio chunk processing failed: {e}")
                    self.audio_buffer = []  # Reset buffer on error
                    return None
            else:
                # Not enough audio yet, wait for more
                return None

        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")
            # Reset buffer on any error to prevent corruption
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
                    call_metadata={"websocket_session": True}  # Fixed: use call_metadata instead of metadata
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



@websocket_router.websocket("/ws/call/{session_id}")
async def websocket_call(
    websocket: WebSocket,
    session_id: Optional[str] = None
):
    """WebSocket endpoint for call handling"""
    logger.info("WebSocket endpoint hit")
    if not session_id:
        session_id = str(uuid.uuid4())
    logger.info("About to accept WebSocket")
    await websocket.accept()
    logger.info("WebSocket accepted for session %s", session_id)
    
    # Initialize services
    speech_service = SpeechService()
    llm_service = LLMService()
    agent = CallCenterAgent(llm_service)
    
    # Create call session
    session = CallSession(session_id, websocket)
    
    # Initialize speech service
    try:
        await speech_service.initialize()
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
    
    # Background tasks
    tasks = []
    
    try:
        # Send initial greeting
        greeting = "Hello! Thank you for calling. How may I assist you today?"
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": greeting
        })
        
        # Generate and send TTS for greeting - Fixed: use proper voice style
        await generate_and_send_tts(websocket, speech_service, greeting, "en-US-AriaNeural")
        
        # Add greeting to conversation
        agent_state["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start background tasks
        agent_task = asyncio.create_task(
            run_agent_loop(agent, agent_state, session, speech_service, websocket)
        )
        tasks.append(agent_task)
        
        audio_output_task = asyncio.create_task(
            handle_audio_output(agent_state["audio_queue"], websocket, speech_service)
        )
        tasks.append(audio_output_task)
        
        # Main message loop
        while session.is_active:
            try:
                message = await websocket.receive_json()
                await handle_websocket_message(message, session, agent_state, speech_service, websocket)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await send_message_to_client(websocket, "error", {
                    "message": "Invalid message format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await send_message_to_client(websocket, "error", {
                    "message": "Error processing your request"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        
    finally:
        # Cleanup
        logger.info(f"Cleaning up session {session_id}")
        session.is_active = False
        
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

async def handle_websocket_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Handle incoming WebSocket messages - FIXED VERSION"""
    message_type = message.get("type")
    
    logger.debug(f"Received WebSocket message: {message_type}")
    
    if message_type == "audio":
        await handle_audio_message(message, session, agent_state, speech_service, websocket)
    elif message_type == "text":
        await handle_text_message(message, session, agent_state, websocket)
    elif message_type == "control":
        await handle_control_message(message, session, websocket)
    elif message_type == "end_call":
        await handle_end_call(message, session, agent_state, websocket)
    elif message_type == "connection":
        # FIXED: Handle connection message
        await handle_connection_message(message, session, websocket)
    elif message_type == "test":
        # Handle test messages for debugging
        await handle_test_message(message, session, websocket)
    else:
        logger.warning(f"Unknown message type: {message_type}")
        await send_message_to_client(websocket, "error", {
            "message": f"Unknown message type: {message_type}"
        })

async def handle_connection_message(
    message: dict,
    session: CallSession,
    websocket: WebSocket
):
    """Handle connection initialization message"""
    try:
        session_id = message.get("sessionId", "unknown")
        timestamp = message.get("timestamp")
        
        logger.info(f"Connection message received from session: {session_id}")
        
        # Send acknowledgment
        await send_message_to_client(websocket, "connected", {
            "sessionId": session_id,
            "status": "acknowledged",
            "server_time": datetime.now().isoformat()
        })
        
        # Send initial greeting after connection is established
        await asyncio.sleep(0.5)  # Small delay to ensure client is ready
        
        greeting = "Hello! Thank you for calling. I'm your AI assistant. How may I help you today?"
        
        # Send transcript
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": greeting
        })
        
        # Generate TTS for greeting
        try:
            await generate_and_send_tts(websocket, speech_service, greeting, "en-US-AriaNeural")
        except Exception as e:
            logger.error(f"TTS generation failed for greeting: {e}")
        
    except Exception as e:
        logger.error(f"Error handling connection message: {e}")

async def handle_test_message(
    message: dict,
    session: CallSession,
    websocket: WebSocket
):
    """Handle test messages for debugging"""
    try:
        logger.info(f"Test message received: {message}")
        
        # Echo back the test message
        await send_message_to_client(websocket, "test_response", {
            "original_message": message,
            "server_time": datetime.now().isoformat(),
            "session_id": session.session_id
        })
        
    except Exception as e:
        logger.error(f"Error handling test message: {e}")

async def handle_audio_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Handle audio data from client - IMPROVED VERSION"""
    try:
        audio_data = message.get("data", "")
        if not audio_data:
            logger.debug("Empty audio data received")
            return
            
        # Convert hex string to bytes with error handling
        try:
            audio_bytes = bytes.fromhex(audio_data)
        except ValueError as e:
            logger.warning(f"Invalid hex audio data: {e}")
            return
        
        if len(audio_bytes) == 0:
            logger.debug("Zero-length audio data")
            return
        
        logger.debug(f"Processing audio chunk: {len(audio_bytes)} bytes")
            
        # Process audio stream with improved error handling
        try:
            processed_audio = await session.handle_audio_stream(audio_bytes)
        except Exception as e:
            logger.error(f"Audio stream processing failed: {e}")
            processed_audio = None
        
        # Only proceed with transcription if we have valid audio
        if processed_audio is not None and len(processed_audio) > 0:
            try:
                # Check for voice activity with fallback
                has_speech = False
                try:
                    has_speech = speech_service.detect_voice_activity(processed_audio)
                except Exception as e:
                    logger.warning(f"VAD failed: {e}, using basic energy detection")
                    # Fallback to simple energy-based detection
                    rms_energy = np.sqrt(np.mean(processed_audio**2))
                    has_speech = rms_energy > 0.01  # Lowered threshold for testing
                
                if has_speech:
                    logger.info("Speech detected, attempting transcription")
                    
                    # Transcribe audio with timeout and error handling
                    transcription = None
                    try:
                        # Add timeout to prevent hanging
                        transcription = await asyncio.wait_for(
                            speech_service.transcribe(processed_audio),
                            timeout=10.0  # Increased timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Transcription timeout")
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                    
                    # Process transcription if successful
                    if transcription and transcription.strip():
                        logger.info(f"Transcribed: '{transcription}'")
                        
                        # Send transcript to client
                        await send_message_to_client(websocket, "transcript", {
                            "speaker": "user",
                            "text": transcription
                        })
                        
                        # Add to agent state
                        agent_state["messages"].append({
                            "role": "user",
                            "content": transcription,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update call record safely
                        try:
                            await session.update_call_record(
                                transcript=agent_state["messages"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to update call record: {e}")
                        
                        # Trigger agent processing by setting current speaker
                        agent_state["current_speaker"] = "user"
                        
                        logger.info("Agent processing triggered")
                        
                    else:
                        logger.debug("No transcription result or empty text")
                        
                        # For testing: simulate user input every 50 audio chunks
                        session.audio_chunk_count = getattr(session, 'audio_chunk_count', 0) + 1
                        if session.audio_chunk_count % 100 == 0:  # Every 100 chunks for testing
                            test_message = "I need help with my insurance claim"
                            logger.info(f"Simulating user input for testing: {test_message}")
                            
                            await send_message_to_client(websocket, "transcript", {
                                "speaker": "user", 
                                "text": test_message
                            })
                            
                            agent_state["messages"].append({
                                "role": "user",
                                "content": test_message,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            agent_state["current_speaker"] = "user"
                else:
                    logger.debug("No speech detected in audio chunk")
            except Exception as e:
                logger.error(f"Speech processing failed: {e}")
        else:
            logger.debug("No valid audio data to process")
            
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")
        await send_message_to_client(websocket, "error", {
            "message": "Error processing audio - continuing with demo mode"
        })

async def handle_text_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    websocket: WebSocket
):
    """Handle text message from client - IMPROVED"""
    try:
        text = message.get("text", "").strip()
        if not text:
            return
            
        logger.info(f"Received text message: '{text}'")
        
        # Send transcript confirmation
        await send_message_to_client(websocket, "transcript", {
            "speaker": "user",
            "text": text
        })
        
        # Add to agent state
        agent_state["messages"].append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update call record
        try:
            await session.update_call_record(
                transcript=agent_state["messages"]
            )
        except Exception as e:
            logger.error(f"Failed to update call record: {e}")
        
        # Trigger agent processing
        agent_state["current_speaker"] = "user"
        
        logger.info("Agent processing triggered from text message")
        
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await send_message_to_client(websocket, "error", {
            "message": "Error processing text message"
        })

# FIXED: Improved agent loop that actually processes messages
async def run_agent_loop(
    agent: CallCenterAgent,
    state: AgentState,
    session: CallSession,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Run the agent processing loop - IMPROVED VERSION"""
    logger.info("Starting improved agent processing loop")
    
    # Send initial greeting immediately
    try:
        initial_greeting = "Hello! I'm your AI assistant. I can help you with insurance claims, policy questions, or connect you with a human agent. What can I help you with today?"
        
        # Add to state
        state["messages"].append({
            "role": "assistant",
            "content": initial_greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to client
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": initial_greeting
        })
        
        # Generate TTS
        await generate_and_send_tts(websocket, speech_service, initial_greeting, "en-US-AriaNeural")
        
        logger.info("Initial greeting sent")
        
    except Exception as e:
        logger.error(f"Failed to send initial greeting: {e}")
    
    # Main processing loop
    last_message_count = 0
    
    while session.is_active and state["call_status"] not in ["ended", "ending"]:
        try:
            current_message_count = len(state["messages"])
            
            # Check if there are new messages to process
            if current_message_count > last_message_count:
                logger.debug(f"Message count changed: {last_message_count} -> {current_message_count}")
                
                # Check if the last message is from user and we haven't responded yet
                if (state["messages"] and 
                    state["messages"][-1].get("role") == "user"):
                    
                    # Check if we already have a response
                    user_message_time = state["messages"][-1].get("timestamp")
                    needs_response = True
                    
                    # Look for agent response after this user message
                    for i in range(len(state["messages"]) - 1, -1, -1):
                        msg = state["messages"][i]
                        if (msg.get("role") == "assistant" and 
                            msg.get("timestamp", "") > user_message_time):
                            needs_response = False
                            break
                        elif msg.get("role") == "user":
                            break
                    
                    if needs_response:
                        logger.info("Processing user message with agent...")
                        
                        user_message = state["messages"][-1].get("content", "")
                        logger.info(f"User said: '{user_message}'")
                        
                        # Set state for agent processing
                        state["current_speaker"] = "user"
                        
                        try:
                            # Run agent workflow
                            logger.debug("Running agent workflow...")
                            updated_state = await agent.run(state)
                            
                            # Update state
                            state.update(updated_state)
                            
                            logger.debug("Agent workflow completed")
                            
                        except Exception as e:
                            logger.error(f"Agent workflow error: {e}")
                            
                            # Fallback response
                            fallback_response = "I apologize, I'm having trouble processing your request right now. Let me try to help you anyway. Could you please tell me more about what you need assistance with?"
                            
                            # Add fallback response
                            state["messages"].append({
                                "role": "assistant", 
                                "content": fallback_response,
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        # Check if agent generated a response
                        if (len(state["messages"]) > current_message_count and
                            state["messages"][-1].get("role") == "assistant"):
                            
                            response_text = state["messages"][-1].get("content", "")
                            if response_text.strip():
                                logger.info(f"Agent response: '{response_text}'")
                                
                                # Send transcript
                                await send_message_to_client(websocket, "transcript", {
                                    "speaker": "agent", 
                                    "text": response_text
                                })
                                
                                # Generate and send TTS
                                try:
                                    await generate_and_send_tts(websocket, speech_service, response_text, "en-US-AriaNeural")
                                except Exception as e:
                                    logger.error(f"TTS generation failed: {e}")
                                
                                # Update call record
                                try:
                                    await session.update_call_record(
                                        transcript=state["messages"]
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update call record: {e}")
                                
                                # Reset speaker state
                                state["current_speaker"] = "waiting"
                                
                                logger.info("Agent response sent successfully")
                            else:
                                logger.warning("Agent generated empty response")
                        else:
                            logger.warning("Agent did not generate a response")
                
                last_message_count = current_message_count
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await asyncio.sleep(1)  # Longer delay on error
    
    logger.info("Agent processing loop ended")

# Helper function to send a test message (for debugging)
async def send_test_user_message(websocket: WebSocket, agent_state: AgentState):
    """Send a test user message for debugging"""
    test_message = "Hello, I need help with filing an insurance claim"
    
    await send_message_to_client(websocket, "transcript", {
        "speaker": "user",
        "text": test_message
    })
    
    agent_state["messages"].append({
        "role": "user",
        "content": test_message, 
        "timestamp": datetime.now().isoformat()
    })
    
    agent_state["current_speaker"] = "user"
    
    logger.info(f"Sent test message: {test_message}")
    
async def handle_control_message(
    message: dict,
    session: CallSession,
    websocket: WebSocket
):
    """Handle control messages (mute, pause, etc.)"""
    try:
        action = message.get("action")
        
        if action == "mute":
            session.audio_processor.mute()
            await send_message_to_client(websocket, "status", {
                "message": "Microphone muted"
            })
        elif action == "unmute":
            session.audio_processor.unmute()
            await send_message_to_client(websocket, "status", {
                "message": "Microphone unmuted"
            })
        elif action == "pause":
            session.is_active = False
            await send_message_to_client(websocket, "status", {
                "message": "Call paused"
            })
        elif action == "resume":
            session.is_active = True
            await send_message_to_client(websocket, "status", {
                "message": "Call resumed"
            })
        else:
            logger.warning(f"Unknown control action: {action}")
            
    except Exception as e:
        logger.error(f"Error handling control message: {e}")

async def handle_end_call(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    websocket: WebSocket
):
    """Handle call end request"""
    try:
        logger.info(f"Call end requested for session {session.session_id}")
        
        agent_state["call_status"] = "ending"
        session.is_active = False
        
        # Send farewell message
        farewell = "Thank you for calling. Have a great day!"
        await send_message_to_client(websocket, "transcript", {
            "speaker": "agent",
            "text": farewell
        })
        
        await send_message_to_client(websocket, "status", {
            "message": "Call ending..."
        })
        
    except Exception as e:
        logger.error(f"Error handling end call: {e}")

async def handle_audio_output(
    audio_queue: asyncio.Queue,
    websocket: WebSocket,
    speech_service: SpeechService
):
    """Handle audio output from agent"""
    logger.info("Starting audio output handler")
    
    while True:
        try:
            # Get audio request from queue
            audio_request = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            
            if audio_request["type"] == "speech":
                text = audio_request["content"]
                emotion = audio_request.get("emotion", "neutral")
                voice_style = audio_request.get("voice_style", "en-US-AriaNeural")  # Fixed: use actual voice name
                
                logger.debug(f"Generating TTS for: {text[:50]}...")
                
                # Set appropriate voice
                speech_service.set_voice(voice_style)
                
                # Generate and stream TTS
                async for audio_chunk in speech_service.synthesize(text, stream=True):
                    await websocket.send_json({
                        "type": "audio",
                        "data": audio_chunk.hex()
                    })
                    
                logger.debug("TTS audio sent to client")
                
        except asyncio.TimeoutError:
            # No audio requests, continue
            continue
        except Exception as e:
            logger.error(f"Audio output error: {e}")
            break
    
    logger.info("Audio output handler ended")

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

async def generate_and_send_tts(
    websocket: WebSocket,
    speech_service: SpeechService,
    text: str,
    voice_style: str = "en-US-AriaNeural"  # Fixed: use actual voice name
):
    """Generate TTS and send audio to client"""
    try:
        if not text.strip():
            return
            
        # Set voice style
        speech_service.set_voice(voice_style)
        
        # Generate and stream audio
        async for audio_chunk in speech_service.synthesize(text, stream=True):
            await websocket.send_json({
                "type": "audio",
                "data": audio_chunk.hex()
            })
            
    except Exception as e:
        logger.error(f"TTS generation error: {e}")

async def finalize_call_record(session: CallSession, agent_state: AgentState):
    """Finalize call record when call ends"""
    try:
        if not session.call_record_id:
            return
            
        end_time = datetime.now()
        duration = int((end_time - session.start_time).total_seconds())
        
        # Extract call summary information
        customer_info = agent_state.get("customer_info", {})
        claim_data = agent_state.get("claim_data", {})
        
        # Determine sentiment
        emotions = agent_state.get("intent_history", [])
        overall_sentiment = "neutral"
        if "frustrated" in emotions:
            overall_sentiment = "negative"
        elif "satisfied" in emotions:
            overall_sentiment = "positive"
        
        # Create summary
        summary = {
            "total_messages": len(agent_state.get("messages", [])),
            "customer_identified": bool(customer_info),
            "claim_created": bool(claim_data),
            "tools_used": [t.get("tool") for t in agent_state.get("tools_output", [])],
            "emotions_detected": list(set(emotions)),
            "resolution_status": "completed" if claim_data else "information_provided"
        }
        
        # Update call record
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

# Additional WebSocket endpoints for testing and monitoring

@websocket_router.websocket("/ws/test/{session_id}")
async def websocket_test(websocket: WebSocket, session_id: str):
    """Test WebSocket endpoint for development"""
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Test WebSocket connection established"
        })
        
        while True:
            message = await websocket.receive_json()
            
            # Echo back the message with timestamp
            response = {
                "type": "echo",
                "original_message": message,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"Test WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")

@websocket_router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for system monitoring"""
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Monitor connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send periodic system status updates
        while True:
            # In a real implementation, this would gather actual system metrics
            status_update = {
                "type": "system_status",
                "timestamp": datetime.now().isoformat(),
                "active_calls": 0,  # Would be actual count
                "system_load": "normal",
                "memory_usage": "45%",
                "speech_service_status": "operational"
            }
            
            await websocket.send_json(status_update)
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        logger.info("Monitor WebSocket disconnected")
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")