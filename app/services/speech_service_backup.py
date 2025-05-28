"""
Speech synthesis and recognition service - Complete Fixed Version
"""
import asyncio
import edge_tts
from faster_whisper import WhisperModel
import numpy as np
from pathlib import Path
import hashlib
from typing import Optional, AsyncGenerator, Dict, List, Any
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
    """Combined STT and TTS service - Fixed Version"""
    
    def __init__(self):
        self.tts_cache_dir = Path(settings.CACHE_DIR) / "tts"
        self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stt_model = None
        self.voices_cache = {}
        self.current_voice = VoiceStyle.FRIENDLY  # Default to Jenny for natural sound
        
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
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            logger.info(f"Initializing Whisper model on {device} with {compute_type}")
            
            self.stt_model = WhisperModel(
                settings.STT_MODEL,
                device=device,
                compute_type=compute_type
            )
            
            logger.info("Whisper model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
        
    def _init_vad(self):
        """Initialize Voice Activity Detection"""
        try:
            torch.set_num_threads(1)
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            logger.info("VAD model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize VAD model: {e}")
            self.vad_model = None
        
    async def _cache_voices(self):
        """Cache available TTS voices"""
        try:
            voices = await edge_tts.list_voices()
            
            # Organize by language
            for voice in voices:
                lang = voice['Locale'].split('-')[0]
                if lang not in self.voices_cache:
                    self.voices_cache[lang] = []
                self.voices_cache[lang].append(voice)
                
            logger.info(f"Cached {len(voices)} TTS voices")
        except Exception as e:
            logger.error(f"Failed to cache voices: {e}")
            
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Detect if audio contains speech"""
        try:
            if self.vad_model is None:
                # Fallback to energy-based detection
                rms_energy = np.sqrt(np.mean(audio**2))
                return rms_energy > settings.VAD_THRESHOLD
                
            # Convert to tensor with proper type
            if isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
                
            # Run VAD
            speech_prob = self.vad_model(audio_tensor, sample_rate).item()
            return speech_prob > settings.VAD_THRESHOLD
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Fallback to energy-based detection
            rms_energy = np.sqrt(np.mean(audio**2))
            return rms_energy > settings.VAD_THRESHOLD
        
    async def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> Optional[str]:
        """Transcribe audio to text - FIXED tensor type error"""
        try:
            # CRITICAL FIX: Ensure audio is float32, not double
            if audio.dtype != np.float32:
                logger.debug(f"Converting audio from {audio.dtype} to float32")
                audio = audio.astype(np.float32)
            
            # Additional validation
            if len(audio) == 0:
                logger.debug("Empty audio array, skipping transcription")
                return None
            
            # Ensure audio values are in valid range [-1, 1]
            if np.max(np.abs(audio)) > 1.0:
                logger.debug("Normalizing audio values to [-1, 1] range")
                audio = audio / np.max(np.abs(audio))
            
            # Check if audio has sufficient content
            rms_energy = np.sqrt(np.mean(audio**2))
            if rms_energy < 0.001:  # Very quiet audio
                logger.debug(f"Audio too quiet (RMS: {rms_energy}), skipping transcription")
                return None
            
            logger.debug(f"Transcribing audio: shape={audio.shape}, dtype={audio.dtype}, RMS={rms_energy:.6f}")
            
            # Run transcription with fixed parameters
            try:
                segments, info = self.stt_model.transcribe(
                    audio,
                    language=language,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        threshold=0.5,
                        min_speech_duration_ms=250
                    ),
                    # IMPORTANT: Additional parameters to prevent tensor errors
                    beam_size=1,  # Reduce beam size for stability
                    temperature=0.0,  # Deterministic output
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
                
                # Process segments safely
                transcription_parts = []
                for segment in segments:
                    text = segment.text.strip()
                    if text and len(text) > 1:  # Ignore single characters
                        transcription_parts.append(text)
                
                if transcription_parts:
                    final_text = " ".join(transcription_parts)
                    logger.info(f"Transcription successful: '{final_text}'")
                    return final_text
                else:
                    logger.debug("No meaningful transcription segments found")
                    return None
                    
            except Exception as transcription_error:
                logger.error(f"Whisper transcription failed: {transcription_error}")
                
                # Try with more conservative settings
                logger.debug("Trying conservative transcription settings...")
                try:
                    segments, _ = self.stt_model.transcribe(
                        audio,
                        language=language,
                        vad_filter=False,  # Disable VAD
                        beam_size=1,
                        temperature=0.0
                    )
                    
                    text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
                    if text_parts:
                        result = " ".join(text_parts)
                        logger.info(f"Conservative transcription successful: '{result}'")
                        return result
                        
                except Exception as conservative_error:
                    logger.error(f"Conservative transcription also failed: {conservative_error}")
                
                return None
                
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
        """Convert text to speech with improved audio quality for natural speech"""
        
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
        
        # FIXED: Optimize rate and pitch for natural, clear speech
        if rate == "+0%":
            rate = "-8%"  # Slower for natural sound and clarity
        if pitch == "+0Hz":
            pitch = "+0Hz"  # Keep natural pitch
        
        # IMPORTANT: Ensure we're using a high-quality voice for consistency
        if actual_voice == "en-US-AriaNeural":
            actual_voice = "en-US-JennyNeural"  # Jenny often sounds more natural
        
        logger.info(f"Generating natural TTS: voice={actual_voice}, rate={rate}, pitch={pitch}")
        
        # Check cache first
        if settings.TTS_CACHE_ENABLED and not stream:
            cache_key = self._get_cache_key(text, actual_voice, rate, pitch)
            cached_file = self.tts_cache_dir / f"{cache_key}.mp3"
            
            if cached_file.exists():
                logger.debug(f"Using cached TTS: {cache_key}")
                async with aiofiles.open(cached_file, 'rb') as f:
                    data = await f.read()
                    yield data
                return
                
        # Generate speech with optimized settings for natural sound
        try:
            # FIXED: Add volume control for better audio quality
            communicate = edge_tts.Communicate(
                text=text, 
                voice=actual_voice, 
                rate=rate, 
                pitch=pitch,
                volume="+0%"  # Normal volume
            )
            
            logger.debug(f"Starting natural TTS generation for: '{text[:50]}...'")
            
            if stream:
                # Stream audio chunks with proper buffering
                chunk_count = 0
                total_size = 0
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunk_count += 1
                        audio_data = chunk["data"]
                        total_size += len(audio_data)
                        
                        # Yield properly sized chunks for smooth playback
                        yield audio_data
                        
                        # Small delay between chunks for smoother streaming
                        if chunk_count % 10 == 0:  # Every 10th chunk
                            await asyncio.sleep(0.001)  # 1ms delay
                            
                logger.debug(f"Natural TTS streaming completed: {chunk_count} chunks, {total_size} bytes")
                
            else:
                # Generate complete file
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
                    
                logger.debug(f"Natural TTS cached: {cache_key}")
                
        except Exception as e:
            logger.error(f"Natural TTS generation failed: {e}")
            # Generate a proper error message instead of a tone
            yield await self._generate_fallback_speech(text)

    async def _generate_fallback_speech(self, original_text: str) -> bytes:
        """Generate fallback speech when primary TTS fails"""
        try:
            fallback_text = "I apologize, there was an audio generation issue. Please let me try again."
            
            # Use the most reliable voice
            fallback_communicate = edge_tts.Communicate(
                fallback_text, 
                "en-US-AriaNeural",  # Most stable voice
                rate="-10%",  # Slower for clarity
                pitch="+0Hz"
            )
            
            temp_file = self.tts_cache_dir / "fallback_speech.mp3"
            await fallback_communicate.save(str(temp_file))
            
            async with aiofiles.open(temp_file, 'rb') as f:
                return await f.read()
        except:
            # Return empty bytes if even fallback fails
            return b""

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
        logger.debug(f"Voice set to: {self.current_voice}")
        
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

    # Add method to test voice quality
    async def test_voice_quality(self, test_text: str = "This is a test of natural sounding speech"):
        """Test voice generation for quality assurance"""
        logger.info("Testing voice quality...")
        
        try:
            chunk_count = 0
            total_size = 0
            
            async for chunk in self.synthesize(test_text, voice="en-US-JennyNeural", stream=True):
                chunk_count += 1
                total_size += len(chunk)
            
            logger.info(f"Voice quality test completed: {chunk_count} chunks, {total_size} bytes")
            return {"success": True, "chunks": chunk_count, "size": total_size}
            
        except Exception as e:
            logger.error(f"Voice quality test failed: {e}")
            return {"success": False, "error": str(e)}

    async def clear_cache(self):
        """Clear TTS cache"""
        try:
            import shutil
            if self.tts_cache_dir.exists():
                shutil.rmtree(self.tts_cache_dir)
                self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("TTS cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.tts_cache_dir.exists():
                return {"files": 0, "size_mb": 0}
            
            files = list(self.tts_cache_dir.glob("*.mp3"))
            total_size = sum(f.stat().st_size for f in files)
            
            return {
                "files": len(files),
                "size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.tts_cache_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    async def preload_common_phrases(self):
        """Preload common phrases for faster response"""
        common_phrases = [
            "Hello! How can I help you today?",
            "I understand your concern. Let me help you with that.",
            "Could you please provide more details?",
            "Thank you for calling. Have a great day!",
            "I'm transferring you to a human agent now.",
            "Your claim has been created successfully.",
            "Is there anything else I can help you with?"
        ]
        
        logger.info("Preloading common phrases...")
        
        for phrase in common_phrases:
            try:
                # Generate and cache each phrase
                chunks = []
                async for chunk in self.synthesize(phrase, voice="en-US-JennyNeural", stream=False):
                    chunks.append(chunk)
                logger.debug(f"Preloaded: '{phrase[:30]}...'")
            except Exception as e:
                logger.warning(f"Failed to preload phrase '{phrase[:30]}...': {e}")
        
        logger.info("Common phrases preloaded")

# Helper function for WebSocket integration
async def generate_and_send_tts_simple(
    websocket,
    speech_service: SpeechService,
    text: str,
    voice_style: str = "en-US-JennyNeural"
):
    """Simple TTS generation for consistent results"""
    try:
        logger.info(f"Generating consistent TTS: '{text[:50]}...'")
        
        # Force consistent settings
        speech_service.set_voice(voice_style)
        
        # Generate with consistent parameters
        chunk_count = 0
        async for audio_chunk in speech_service.synthesize(
            text, 
            voice=voice_style,
            rate="-8%",  # Consistent slower rate
            pitch="+0Hz",  # Natural pitch
            stream=True
        ):
            chunk_count += 1
            hex_data = audio_chunk.hex()
            
            await websocket.send_json({
                "type": "audio",
                "data": hex_data,
                "chunk": chunk_count,
                "voice": voice_style,
                "consistent": True
            })
            
            # Controlled timing
            await asyncio.sleep(0.02)
        
        logger.info(f"Consistent TTS completed: {chunk_count} chunks")
        
    except Exception as e:
        logger.error(f"Simple TTS generation failed: {e}")
        raise