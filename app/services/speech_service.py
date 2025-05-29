"""
FIXED Speech Service - VAD chunk size issue resolved
Replace your app/services/speech_service.py with this version
"""
import asyncio
import edge_tts
from faster_whisper import WhisperModel
import numpy as np
from pathlib import Path
import tempfile
from typing import Optional, AsyncGenerator
import torch
import aiofiles

from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VoiceStyle:
    """Production voice styles"""
    PROFESSIONAL = "en-US-AriaNeural"
    FRIENDLY = "en-US-JennyNeural" 
    EMPATHETIC = "en-US-SaraNeural"

class SpeechService:
    """FIXED Speech Service - VAD chunk size issues resolved"""
    
    def __init__(self):
        self.tts_cache_dir = Path(settings.CACHE_DIR) / "tts"
        self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stt_model = None
        self.vad_model = None
        self.current_voice = VoiceStyle.FRIENDLY
        
        # Production voice settings
        self.voice_settings = {
            "voice": "en-US-JennyNeural",
            "rate": "-12%",
            "pitch": "+0Hz", 
            "volume": "+0%"
        }
        
        # VAD settings - FIXED
        self.vad_sample_rate = 16000
        self.vad_chunk_size = 512  # CRITICAL: Exact requirement for Silero VAD
        
    async def initialize(self):
        """Initialize speech components"""
        logger.info("🎤 Initializing FIXED speech service...")
        
        try:
            await self._init_stt()
            await self._init_vad()
            logger.info("✅ FIXED speech service ready")
            
        except Exception as e:
            logger.error(f"Speech service initialization failed: {e}")
            raise
            
    async def _init_stt(self):
        """Initialize Whisper"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            logger.info(f"Loading Whisper: {settings.STT_MODEL} on {device}")
            
            loop = asyncio.get_event_loop()
            self.stt_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    settings.STT_MODEL,
                    device=device,
                    compute_type=compute_type
                )
            )
            
            logger.info("✅ Whisper loaded")
            
        except Exception as e:
            logger.warning(f"Whisper initialization failed: {e}")
            self.stt_model = None
        
    async def _init_vad(self):
        """Initialize VAD with FIXED settings"""
        try:
            torch.set_num_threads(1)
            
            loop = asyncio.get_event_loop()
            model_and_utils = await loop.run_in_executor(
                None,
                lambda: torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
            )
            
            self.vad_model = model_and_utils[0] if isinstance(model_and_utils, tuple) else model_and_utils
            logger.info("✅ VAD loaded with FIXED chunk size support")
            
        except Exception as e:
            logger.warning(f"VAD initialization failed: {e}")
            self.vad_model = None
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """FIXED: VAD with proper chunk size handling"""
        try:
            if self.vad_model is None:
                # Fallback energy detection
                rms_energy = np.sqrt(np.mean(audio**2))
                return rms_energy > 0.005
            
            # CRITICAL FIX: Ensure proper audio format and size
            if isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
            
            # CRITICAL FIX: Ensure exactly 512 samples for 16kHz
            required_samples = 512 if sample_rate == 16000 else 256
            
            if len(audio_tensor) != required_samples:
                if len(audio_tensor) > required_samples:
                    # Truncate to required size
                    audio_tensor = audio_tensor[:required_samples]
                    logger.debug(f"Truncated audio to {required_samples} samples")
                else:
                    # Pad to required size
                    padding = required_samples - len(audio_tensor)
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                    logger.debug(f"Padded audio to {required_samples} samples")
            
            # Validate tensor shape and content
            if len(audio_tensor.shape) != 1:
                logger.error(f"Invalid audio tensor shape: {audio_tensor.shape}")
                return False
            
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                logger.warning("Audio contains NaN or Inf values")
                return False
            
            # Run VAD with proper error handling
            try:
                speech_prob = self.vad_model(audio_tensor, sample_rate).item()
                is_speech = speech_prob > 0.5
                
                logger.debug(f"VAD result: {speech_prob:.3f} -> {'speech' if is_speech else 'silence'}")
                return is_speech
                
            except Exception as vad_error:
                logger.warning(f"VAD model failed: {vad_error}")
                # Fallback to energy detection
                rms_energy = np.sqrt(np.mean(audio_tensor.numpy()**2))
                return rms_energy > 0.005
                
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Safe fallback
            try:
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.numpy()
                else:
                    audio_np = audio
                rms_energy = np.sqrt(np.mean(audio_np**2))
                return rms_energy > 0.005
            except:
                return False
        
    async def transcribe(self, audio: np.ndarray, language: str = "en") -> Optional[str]:
        """FIXED: Transcribe with proper audio handling"""
        try:
            if self.stt_model is None:
                logger.warning("STT model not available")
                return None
                
            # Ensure proper format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if len(audio) == 0:
                return None
            
            # Normalize audio safely
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            # Check if audio has content
            rms_energy = np.sqrt(np.mean(audio**2))
            if rms_energy < 0.001:
                logger.debug("Audio too quiet for transcription")
                return None
            
            logger.debug(f"Transcribing: {audio.shape}, RMS: {rms_energy:.6f}")
            
            # Run transcription
            loop = asyncio.get_event_loop()
            
            try:
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.stt_model.transcribe(
                        audio,
                        language=language,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500,
                            threshold=0.5,
                            min_speech_duration_ms=250
                        ),
                        beam_size=1,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6
                    )
                )
                
                # Extract text
                text_parts = []
                for segment in segments:
                    text = segment.text.strip()
                    if text and len(text) > 1:
                        text_parts.append(text)
                
                if text_parts:
                    result = " ".join(text_parts)
                    logger.info(f"✅ Transcribed: '{result}'")
                    return result
                else:
                    logger.debug("No meaningful transcription")
                    return None
                    
            except Exception as transcription_error:
                logger.error(f"Transcription failed: {transcription_error}")
                
                # Try conservative settings
                try:
                    segments, _ = await loop.run_in_executor(
                        None,
                        lambda: self.stt_model.transcribe(
                            audio,
                            language=language,
                            vad_filter=False,
                            beam_size=1,
                            temperature=0.0
                        )
                    )
                    
                    text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
                    if text_parts:
                        result = " ".join(text_parts)
                        logger.info(f"✅ Conservative transcription: '{result}'")
                        return result
                        
                except Exception as conservative_error:
                    logger.error(f"Conservative transcription failed: {conservative_error}")
                
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def synthesize_natural(
        self,
        text: str,
        voice: Optional[str] = None,
        stream: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """Generate natural TTS audio"""
        try:
            if not text.strip():
                logger.warning("Empty text for TTS")
                return
                
            # Use production settings
            actual_voice = voice or self.voice_settings["voice"]
            rate = self.voice_settings["rate"]
            pitch = self.voice_settings["pitch"]
            volume = self.voice_settings["volume"]
            
            logger.info(f"🎤 Generating TTS: '{text[:50]}...'")
            
            if stream:
                # Streaming generation
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=actual_voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume
                )
                
                # Collect chunks for smooth delivery
                chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                
                # Send in optimal sizes
                if chunks:
                    complete_audio = b''.join(chunks)
                    
                    # Send in 8KB chunks
                    chunk_size = 8192
                    for i in range(0, len(complete_audio), chunk_size):
                        audio_chunk = complete_audio[i:i + chunk_size]
                        yield audio_chunk
                        
                        # Natural timing
                        await asyncio.sleep(0.08)
                
                logger.info("✅ TTS streaming completed")
                
            else:
                # Complete file generation
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=actual_voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume
                )
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    await communicate.save(temp_path)
                    
                    async with aiofiles.open(temp_path, 'rb') as f:
                        audio_data = await f.read()
                    
                    yield audio_data
                    logger.info(f"✅ TTS file generated: {len(audio_data)} bytes")
                    
                finally:
                    try:
                        Path(temp_path).unlink()
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")

    async def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> AsyncGenerator[bytes, None]:
        """Legacy method - redirects to natural synthesis"""
        async for chunk in self.synthesize_natural(text, voice, stream=True):
            yield chunk

    def set_voice(self, voice: str):
        """Set voice with production settings"""
        voice_presets = {
            "professional": {
                "voice": "en-US-AriaNeural",
                "rate": "-10%",
                "pitch": "+0Hz"
            },
            "friendly": {
                "voice": "en-US-JennyNeural", 
                "rate": "-12%",
                "pitch": "+0Hz"
            },
            "empathetic": {
                "voice": "en-US-SaraNeural",
                "rate": "-15%", 
                "pitch": "-1Hz"
            }
        }
        
        if voice.lower() in voice_presets:
            self.voice_settings.update(voice_presets[voice.lower()])
        else:
            self.voice_settings["voice"] = voice
            
        self.current_voice = self.voice_settings["voice"]
        logger.info(f"🎤 Voice set to: {self.current_voice}")

    def get_stats(self) -> dict:
        """Get service statistics"""
        return {
            "stt_available": self.stt_model is not None,
            "vad_available": self.vad_model is not None,
            "current_voice": self.current_voice,
            "voice_settings": self.voice_settings,
            "vad_chunk_size": self.vad_chunk_size,
            "vad_sample_rate": self.vad_sample_rate
        }

logger.info("✅ FIXED Speech Service loaded - VAD chunk size issues resolved")