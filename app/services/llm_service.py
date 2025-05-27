"""
LLM service for handling AI completions
"""
from typing import List, Dict, Any, Optional
import openai
from anthropic import Anthropic
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMService:
    """Service for interacting with LLMs"""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        
        if self.provider == "openai":
            openai.api_key = settings.OPENAI_API_KEY
            self.client = openai
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        tools: Optional[List] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send chat completion request"""
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        try:
            if self.provider == "openai":
                response = await self._openai_chat(
                    messages, temperature, max_tokens, tools, stream
                )
            elif self.provider == "anthropic":
                response = await self._anthropic_chat(
                    messages, temperature, max_tokens, stream
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            return response
            
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise
            
    async def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List],
        stream: bool
    ) -> Dict[str, Any]:
        """OpenAI chat completion"""
        params = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
            
        if stream:
            response = await openai.ChatCompletion.acreate(**params, stream=True)
            return {"stream": response}
        else:
            response = await openai.ChatCompletion.acreate(**params)
            return response.choices[0].message
            
    async def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """Anthropic chat completion"""
        # Convert messages to Anthropic format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
        response = await self.client.messages.create(
            model=settings.LLM_MODEL,
            messages=anthropic_messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return {"stream": response}
        else:
            return {"content": response.content[0].text}
            
    async def complete(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Simple completion"""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.get("content", "")