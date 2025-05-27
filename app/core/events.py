"""
Event system for the application
"""
from typing import Dict, List, Callable, Any
import asyncio
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EventBus:
    """Simple event bus for application events"""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._async_handlers: Dict[str, List[Callable]] = {}
        
    def on(self, event_name: str, handler: Callable):
        """Register event handler"""
        if asyncio.iscoroutinefunction(handler):
            if event_name not in self._async_handlers:
                self._async_handlers[event_name] = []
            self._async_handlers[event_name].append(handler)
        else:
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(handler)
            
    def off(self, event_name: str, handler: Callable):
        """Unregister event handler"""
        if event_name in self._handlers and handler in self._handlers[event_name]:
            self._handlers[event_name].remove(handler)
        if event_name in self._async_handlers and handler in self._async_handlers[event_name]:
            self._async_handlers[event_name].remove(handler)
            
    async def emit(self, event_name: str, data: Any = None):
        """Emit event to all handlers"""
        # Call sync handlers
        if event_name in self._handlers:
            for handler in self._handlers[event_name]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
                    
        # Call async handlers
        if event_name in self._async_handlers:
            tasks = []
            for handler in self._async_handlers[event_name]:
                tasks.append(self._call_async_handler(handler, data, event_name))
            await asyncio.gather(*tasks)
            
    async def _call_async_handler(self, handler: Callable, data: Any, event_name: str):
        """Call async handler with error handling"""
        try:
            await handler(data)
        except Exception as e:
            logger.error(f"Error in async event handler for {event_name}: {e}")

# Global event bus instance
event_bus = EventBus()

# Common events
class Events:
    """Common application events"""
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    CALL_TRANSFERRED = "call.transferred"
    
    CLAIM_CREATED = "claim.created"
    CLAIM_UPDATED = "claim.updated"
    
    CUSTOMER_IDENTIFIED = "customer.identified"
    CUSTOMER_EMOTION_CHANGED = "customer.emotion.changed"
    
    AGENT_ACTION = "agent.action"
    AGENT_TOOL_USED = "agent.tool.used"
    
    AUDIO_SPEECH_DETECTED = "audio.speech.detected"
    AUDIO_SILENCE_DETECTED = "audio.silence.detected"
    
    ERROR_OCCURRED = "error.occurred"