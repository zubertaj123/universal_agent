"""
WebSocket endpoints for real-time communication
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
import numpy as np
from typing import Optional
from datetime import datetime
import uuid

from app.services.speech_service import SpeechService
from app.services.llm_service import LLMService
from app.agents.call_agent import CallCenterAgent, AgentState
from app.models.conversation import ConversationManager
from app.utils.audio_processing import AudioProcessor
from app.utils.logger import setup_logger

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
        
    async def handle_audio_stream(self, audio_data: bytes):
        """Process incoming audio"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.append(audio_array)
        
        # Process when we have enough data (e.g., 1 second)
        if len(self.audio_buffer) >= 50:  # Assuming 20ms chunks
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            return full_audio
        return None

@websocket_router.websocket("/ws/call/{session_id}")
async def websocket_call(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    speech_service: SpeechService = Depends(lambda: SpeechService()),
    llm_service: LLMService = Depends(lambda: LLMService())
):
    """WebSocket endpoint for call handling"""
    await websocket.accept()
    
    if not session_id:
        session_id = str(uuid.uuid4())
        
    session = CallSession(session_id, websocket)
    agent = CallCenterAgent(llm_service)
    
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
        "audio_queue": asyncio.Queue()
    }
    
    # Tasks for concurrent processing
    tasks = []
    
    try:
        # Send initial greeting
        greeting = "Hello! Thank you for calling. How may I assist you today?"
        await send_tts_response(websocket, speech_service, greeting)
        
        # Start agent task
        agent_task = asyncio.create_task(
            run_agent_loop(agent, agent_state, session, speech_service)
        )
        tasks.append(agent_task)
        
        # Audio output task
        audio_output_task = asyncio.create_task(
            handle_audio_output(agent_state["audio_queue"], websocket, speech_service)
        )
        tasks.append(audio_output_task)
        
        # Main message loop
        while True:
            message = await websocket.receive_json()
            
            if message["type"] == "audio":
                # Handle audio data
                audio_data = message["data"]
                audio_bytes = bytes.fromhex(audio_data)
                
                full_audio = await session.handle_audio_stream(audio_bytes)
                if full_audio is not None:
                    # Transcribe audio
                    text = await speech_service.transcribe(full_audio)
                    
                    if text:
                        # Add to conversation
                        agent_state["messages"].append({
                            "role": "user",
                            "content": text,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Notify agent
                        await websocket.send_json({
                            "type": "transcript",
                            "speaker": "user",
                            "text": text
                        })
                        
            elif message["type"] == "end_call":
                agent_state["call_status"] = "ending"
                break
                
            elif message["type"] == "control":
                # Handle control messages (pause, resume, etc.)
                await handle_control_message(message, session)
                
    except WebSocketDisconnect:
        logger.info(f"Call {session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in call {session_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Cleanup
        session.is_active = False
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
            
        # Save call record
        await save_call_record(session_id, agent_state)
        
        await websocket.close()

async def run_agent_loop(
    agent: CallCenterAgent,
    state: AgentState,
    session: CallSession,
    speech_service: SpeechService
):
    """Run the agent processing loop"""
    while session.is_active and state["call_status"] != "ended":
        try:
            # Run agent workflow
            state = await agent.run(state)
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await asyncio.sleep(1)

async def handle_audio_output(
    audio_queue: asyncio.Queue,
    websocket: WebSocket,
    speech_service: SpeechService
):
    """Handle audio output from agent"""
    while True:
        try:
            # Get audio from queue
            audio_data = await audio_queue.get()
            
            if audio_data["type"] == "speech":
                # Generate TTS
                text = audio_data["content"]
                emotion = audio_data.get("emotion", "neutral")
                
                # Send transcript
                await websocket.send_json({
                    "type": "transcript",
                    "speaker": "agent",
                    "text": text
                })
                
                # Stream TTS audio
                async for audio_chunk in speech_service.synthesize(text, stream=True):
                    await websocket.send_json({
                        "type": "audio",
                        "data": audio_chunk.hex()
                    })
                    
        except Exception as e:
            logger.error(f"Audio output error: {e}")
            break

async def send_tts_response(
    websocket: WebSocket,
    speech_service: SpeechService,
    text: str,
    voice: Optional[str] = None
):
    """Send TTS response over websocket"""
    # Send transcript first
    await websocket.send_json({
        "type": "transcript",
        "speaker": "agent",
        "text": text
    })
    
    # Stream audio
    async for audio_chunk in speech_service.synthesize(text, voice=voice, stream=True):
        await websocket.send_json({
            "type": "audio",
            "data": audio_chunk.hex()
        })

async def handle_control_message(message: dict, session: CallSession):
    """Handle control messages"""
    action = message.get("action")
    
    if action == "pause":
        session.is_active = False
    elif action == "resume":
        session.is_active = True
    elif action == "mute":
        session.audio_processor.mute()
    elif action == "unmute":
        session.audio_processor.unmute()

async def save_call_record(session_id: str, state: AgentState):
    """Save call record to database"""
    try:
        # Extract call data
        call_data = {
            "session_id": session_id,
            "messages": state["messages"],
            "customer_info": state.get("customer_info", {}),
            "claim_data": state.get("claim_data", {}),
            "call_summary": state.get("call_summary", {}),
            "duration": len(state["messages"]) * 10,  # Rough estimate
            "status": state["call_status"]
        }
        
        # Save to database (implement actual saving)
        logger.info(f"Saved call record for {session_id}")
        
    except Exception as e:
        logger.error(f"Error saving call record: {e}")