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
    """Manages a single call session"""
    
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
        
    async def handle_audio_stream(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Process incoming audio from browser"""
        try:
            # Decode and resample browser audio (e.g., webm/opus) to mono 16kHz PCM float32
            audio_pcm = self.audio_processor.decode_and_resample_browser_audio(audio_data, target_sr=16000)
            if audio_pcm.size == 0:
                return None

            # Buffer audio until at least 512 samples (16kHz = 32ms)
            self.audio_buffer.append(audio_pcm)
            buffered = np.concatenate(self.audio_buffer) if self.audio_buffer else audio_pcm

            # Only process in chunks of 512 samples
            frames = self.audio_processor.frame_audio(buffered, frame_size=512)
            if frames:
                # Remove used samples from buffer
                used_samples = len(frames) * 512
                self.audio_buffer = [buffered[used_samples:]] if buffered[used_samples:].size > 0 else []
                # Only process the first frame (for VAD), or process all as needed
                processed_audio = self.audio_processor.preprocess_audio(frames[0])
                return processed_audio

            return None

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None
    
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
                    metadata={"websocket_session": True}
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
        
        # Generate and send TTS for greeting
        await generate_and_send_tts(websocket, speech_service, greeting)
        
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
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "audio":
        await handle_audio_message(message, session, agent_state, speech_service, websocket)
    elif message_type == "text":
        await handle_text_message(message, session, agent_state, websocket)
    elif message_type == "control":
        await handle_control_message(message, session, websocket)
    elif message_type == "end_call":
        await handle_end_call(message, session, agent_state, websocket)
    else:
        logger.warning(f"Unknown message type: {message_type}")

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
            return
            
        # Convert hex string to bytes
        audio_bytes = bytes.fromhex(audio_data)
        
        # Process audio stream (now handles browser blobs, resampling, framing)
        processed_audio = await session.handle_audio_stream(audio_bytes)
        
        if processed_audio is not None:
            # Check for voice activity
            has_speech = speech_service.detect_voice_activity(processed_audio)
            
            if has_speech:
                # Transcribe audio
                transcription = await speech_service.transcribe(processed_audio)
                
                if transcription and transcription.strip():
                    logger.debug(f"Transcribed: {transcription}")
                    
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
                    
                    # Update call record
                    await session.update_call_record(
                        transcript=agent_state["messages"]
                    )
                    
                    # Trigger agent processing
                    agent_state["current_speaker"] = "user"
                    
    except Exception as e:
        logger.error(f"Error handling audio message: {e}")
        await send_message_to_client(websocket, "error", {
            "message": "Error processing audio"
        })

async def handle_text_message(
    message: dict,
    session: CallSession,
    agent_state: AgentState,
    websocket: WebSocket
):
    """Handle text message from client"""
    try:
        text = message.get("text", "").strip()
        if not text:
            return
            
        logger.debug(f"Received text: {text}")
        
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
        await session.update_call_record(
            transcript=agent_state["messages"]
        )
        
        # Trigger agent processing
        agent_state["current_speaker"] = "user"
        
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await send_message_to_client(websocket, "error", {
            "message": "Error processing text message"
        })

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

async def run_agent_loop(
    agent: CallCenterAgent,
    state: AgentState,
    session: CallSession,
    speech_service: SpeechService,
    websocket: WebSocket
):
    """Run the agent processing loop"""
    logger.info("Starting agent processing loop")
    
    while session.is_active and state["call_status"] not in ["ended", "ending"]:
        try:
            # Only process if there's new user input
            if (state["messages"] and 
                state["messages"][-1].get("role") == "user" and
                state["current_speaker"] == "user"):
                
                logger.debug("Processing agent workflow...")
                
                # Run agent workflow
                updated_state = await agent.run(state)
                
                # Update state
                state.update(updated_state)
                
                # Check if agent generated a response
                if (state["messages"] and 
                    state["messages"][-1].get("role") == "assistant" and
                    state["current_speaker"] == "agent"):
                    
                    response_text = state["messages"][-1].get("content", "")
                    if response_text.strip():
                        # Send transcript
                        await send_message_to_client(websocket, "transcript", {
                            "speaker": "agent", 
                            "text": response_text
                        })
                        
                        # Generate and send TTS
                        await generate_and_send_tts(websocket, speech_service, response_text)
                        
                        # Update call record
                        await session.update_call_record(
                            transcript=state["messages"]
                        )
                        
                        # Reset speaker to wait for user
                        state["current_speaker"] = "waiting"
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(1)  # Longer delay on error
    
    logger.info("Agent processing loop ended")

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
                voice_style = audio_request.get("voice_style", "professional")
                
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
    voice_style: str = "professional"
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