"""
Speech synthesis and recognition service
"""
import asyncio
import edge_tts
from faster_whisper import WhisperModel
import numpy as np
from pathlib import Path
import hashlib
from typing import Optional, AsyncGenerator, Dict, List
import torch
import aiofiles
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VoiceStyle:
    """Available voice styles"""
    PROFESSIONAL = "en-US-AriaNeural"
    FRIENDLY = "en-US-JennyNeural"
    TECHNICAL = "en-US-TonyNeural"
    EMPATHETIC = "en-US-SaraNeural"
    MULTILINGUAL = "en-US-JennyMultilingualNeural"

class SpeechService:
    """Combined STT and TTS service"""
    
    def __init__(self):
        self.tts_cache_dir = Path(settings.CACHE_DIR) / "tts"
        self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stt_model = None
        self.voices_cache = {}
        self.current_voice = VoiceStyle.PROFESSIONAL
        
        # VAD model
        self.vad_model = None
        
    async def initialize(self):
        """Initialize speech models"""
        logger.info("Initializing speech service...")
        
        # Initialize STT
        self._init_stt()
        
        # Initialize VAD
        self._init_vad()
        
        # Cache available voices
        await self._cache_voices()
        
        logger.info("Speech service initialized")
        
    def _init_stt(self):
        """Initialize Whisper model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        self.stt_model = WhisperModel(
            settings.STT_MODEL,
            device=device,
            compute_type=compute_type
        )
        
    def _init_vad(self):
        """Initialize Voice Activity Detection"""
        torch.set_num_threads(1)
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        
    async def _cache_voices(self):
        """Cache available TTS voices"""
        voices = await edge_tts.list_voices()
        
        # Organize by language
        for voice in voices:
            lang = voice['Locale'].split('-')[0]
            if lang not in self.voices_cache:
                self.voices_cache[lang] = []
            self.voices_cache[lang].append(voice)
            
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Detect if audio contains speech"""
        if self.vad_model is None:
            return True
            
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
            
        # Run VAD
        speech_prob = self.vad_model(audio_tensor, sample_rate).item()
        return speech_prob > settings.VAD_THRESHOLD
        
    async def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            # Run transcription
            segments, _ = self.stt_model.transcribe(
                audio,
                language=language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=settings.VAD_THRESHOLD
                )
            )
            
            # Combine segments
            text = " ".join([segment.text.strip() for segment in segments])
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
            
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        stream: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """Convert text to speech with improved audio quality"""
        
        # Fix voice style mapping
        voice_style = voice or self.current_voice
        
        # Map voice styles to actual Edge TTS voice names
        voice_mapping = {
            "professional": "en-US-AriaNeural",
            "friendly": "en-US-JennyNeural", 
            "technical": "en-US-TonyNeural",
            "empathetic": "en-US-SaraNeural",
            "multilingual": "en-US-JennyMultilingualNeural"
        }
        
        # Use mapping if it's a style name, otherwise use as-is
        actual_voice = voice_mapping.get(voice_style.lower(), voice_style)
        
        # FIXED: Adjust rate and pitch for more natural speech
        if rate == "+0%":
            rate = "-10%"  # Slightly slower for better comprehension
        if pitch == "+0Hz":
            pitch = "-2Hz"  # Slightly lower pitch for more natural sound
        
        logger.info(f"Generating TTS: voice={actual_voice}, rate={rate}, pitch={pitch}")
        
        # Check cache
        if settings.TTS_CACHE_ENABLED and not stream:
            cache_key = self._get_cache_key(text, actual_voice, rate, pitch)
            cached_file = self.tts_cache_dir / f"{cache_key}.mp3"
            
            if cached_file.exists():
                logger.debug(f"Using cached TTS: {cache_key}")
                async with aiofiles.open(cached_file, 'rb') as f:
                    data = await f.read()
                    yield data
                return
                
        # Generate speech with improved settings
        try:
            communicate = edge_tts.Communicate(
                text=text, 
                voice=actual_voice, 
                rate=rate, 
                pitch=pitch
            )
            
            logger.debug(f"Starting TTS generation for: '{text[:50]}...'")
            
            if stream:
                # Stream audio chunks
                chunk_count = 0
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunk_count += 1
                        yield chunk["data"]
                        
                logger.debug(f"TTS streaming completed: {chunk_count} chunks")
                
            else:
                # Save to cache and return
                cache_key = self._get_cache_key(text, actual_voice, rate, pitch)
                cached_file = self.tts_cache_dir / f"{cache_key}.mp3"
                
                # Generate to temp file first
                temp_file = cached_file.with_suffix('.tmp')
                await communicate.save(str(temp_file))
                
                # Move to final location
                temp_file.rename(cached_file)
                
                # Read and yield the generated audio
                async with aiofiles.open(cached_file, 'rb') as f:
                    data = await f.read()
                    yield data
                    
                logger.debug(f"TTS cached: {cache_key}")
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Generate a simple error tone instead of failing
            yield await self._generate_error_tone()
    
    async def _generate_error_tone(self) -> bytes:
        """Generate a simple error tone when TTS fails"""
        try:
            # Create a simple beep using Edge TTS
            error_communicate = edge_tts.Communicate(
                "Audio generation error", 
                "en-US-AriaNeural",
                rate="-20%",
                pitch="-5Hz"
            )
            
            temp_file = self.tts_cache_dir / "error_tone.mp3"
            await error_communicate.save(str(temp_file))
            
            async with aiofiles.open(temp_file, 'rb') as f:
                return await f.read()
        except:
            # Return empty bytes if even error tone fails
            return b""

    async def generate_and_send_tts(
        websocket: WebSocket,
        speech_service: SpeechService,
        text: str,
        voice_style: str = "en-US-AriaNeural"
    ):
        """Generate TTS and send audio to client - FIXED VERSION"""
        try:
            if not text.strip():
                return
                
            logger.info(f"Generating TTS for: '{text[:50]}...'")
            
            # Set voice style
            speech_service.set_voice(voice_style)
            
            # Generate audio in chunks for streaming
            chunk_count = 0
            total_bytes = 0
            
            async for audio_chunk in speech_service.synthesize(text, stream=True):
                if audio_chunk and len(audio_chunk) > 0:
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                    
                    # Convert to hex for transmission
                    hex_data = audio_chunk.hex()
                    
                    # Send audio chunk to client
                    await websocket.send_json({
                        "type": "audio",
                        "data": hex_data,
                        "chunk": chunk_count,
                        "format": "mp3",
                        "voice": voice_style
                    })
                    
                    # Small delay between chunks for smoother playback
                    await asyncio.sleep(0.01)
            
            logger.info(f"TTS sent: {chunk_count} chunks, {total_bytes} bytes total")
            
            # Send end marker
            await websocket.send_json({
                "type": "audio_complete",
                "chunks_sent": chunk_count,
                "total_bytes": total_bytes
            })
                
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            
            # Send error notification to client
            await websocket.send_json({
                "type": "audio_error",
                "message": "TTS generation failed"
            })

    def _get_cache_key(self, text: str, voice: str, rate: str, pitch: str) -> str:
        """Generate cache key for TTS"""
        key_string = f"{text}_{voice}_{rate}_{pitch}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def set_voice(self, voice: str):
        """Set current voice"""
        # Map voice styles to actual names
        voice_mapping = {
            "professional": "en-US-AriaNeural",
            "friendly": "en-US-JennyNeural",
            "technical": "en-US-TonyNeural", 
            "empathetic": "en-US-SaraNeural",
            "multilingual": "en-US-JennyMultilingualNeural"
        }
        
        self.current_voice = voice_mapping.get(voice.lower(), voice)
        
    async def get_voice_for_language(self, language: str, gender: str = "Female") -> Optional[str]:
        """Get appropriate voice for language"""
        lang_code = language.lower()[:2]
        
        if lang_code in self.voices_cache:
            voices = self.voices_cache[lang_code]
            gender_voices = [v for v in voices if v.get('Gender', '').lower() == gender.lower()]
            
            if gender_voices:
                # Prefer neural voices
                neural_voices = [v for v in gender_voices if 'Neural' in v['ShortName']]
                if neural_voices:
                    return neural_voices[0]['ShortName']
                return gender_voices[0]['ShortName']
                
        return None