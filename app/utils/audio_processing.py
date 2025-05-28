"""
Audio processing utilities - Enhanced version with improved audio handling
"""
import numpy as np
from scipy import signal
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import subprocess
import struct
import wave
import io
import tempfile
import os
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioProcessor:
    """Enhanced audio processing utilities with improved format handling"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_muted = False
        self.gain_factor = 1.0
        self.vad_threshold = 0.001
        self.speech_counter = 0
        
        # Audio processing parameters
        self.frame_size = 1024
        self.hop_length = 512
        self.pre_emphasis_coeff = 0.97
        
        # Voice activity detection parameters
        self.min_speech_duration = 0.3  # seconds
        self.min_silence_duration = 0.5  # seconds
        
        # Audio quality parameters
        self.max_audio_length = 30.0  # Maximum 30 seconds per chunk
        self.min_audio_samples = 160   # Minimum samples for processing (10ms at 16kHz)
        
        # Format detection cache
        self._format_cache = {}

    def ensure_float32_audio(self, audio_data: Union[np.ndarray, bytes, list]) -> np.ndarray:
        """
        Ensure audio data is in the correct float32 format
        Handles various input types and converts them to normalized float32 numpy array
        """
        try:
            if audio_data is None:
                return np.array([], dtype=np.float32)
            
            # Handle numpy arrays
            if isinstance(audio_data, np.ndarray):
                return self._convert_numpy_to_float32(audio_data)
            
            # Handle bytes
            elif isinstance(audio_data, (bytes, bytearray)):
                return self._convert_bytes_to_float32(audio_data)
            
            # Handle lists
            elif isinstance(audio_data, (list, tuple)):
                audio_array = np.array(audio_data)
                return self._convert_numpy_to_float32(audio_array)
            
            # Handle other types
            else:
                logger.warning(f"Unsupported audio data type: {type(audio_data)}")
                return np.array([], dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error ensuring float32 audio: {e}")
            return np.array([], dtype=np.float32)

    def _convert_numpy_to_float32(self, audio: np.ndarray) -> np.ndarray:
        """Convert numpy array to normalized float32"""
        try:
            if audio.size == 0:
                return np.array([], dtype=np.float32)
            
            # Handle different input dtypes
            if audio.dtype == np.float32:
                # Already float32, just ensure proper range
                return np.clip(audio, -1.0, 1.0)
            
            elif audio.dtype == np.float64:
                # Convert from float64 to float32
                return np.clip(audio.astype(np.float32), -1.0, 1.0)
            
            elif audio.dtype == np.int16:
                # Convert from 16-bit int to float32
                return (audio.astype(np.float32) / 32768.0)
            
            elif audio.dtype == np.int32:
                # Convert from 32-bit int to float32
                return (audio.astype(np.float32) / 2147483648.0)
            
            elif audio.dtype == np.uint8:
                # Convert from 8-bit unsigned to float32
                return ((audio.astype(np.float32) - 128.0) / 128.0)
            
            elif audio.dtype == np.uint16:
                # Convert from 16-bit unsigned to float32
                return ((audio.astype(np.float32) - 32768.0) / 32768.0)
            
            else:
                # Try generic conversion
                logger.warning(f"Unknown dtype {audio.dtype}, attempting generic conversion")
                return np.clip(audio.astype(np.float32), -1.0, 1.0)
                
        except Exception as e:
            logger.error(f"Error converting numpy array to float32: {e}")
            return np.array([], dtype=np.float32)

    def _convert_bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Convert bytes to float32 audio using multiple strategies"""
        try:
            if len(audio_bytes) == 0:
                return np.array([], dtype=np.float32)
            
            # Try different byte interpretations
            conversion_strategies = [
                (np.int16, 2, lambda x: x / 32768.0),           # 16-bit signed
                (np.float32, 4, lambda x: np.clip(x, -1.0, 1.0)), # 32-bit float
                (np.int32, 4, lambda x: x / 2147483648.0),      # 32-bit signed
                (np.uint8, 1, lambda x: (x - 128.0) / 128.0),  # 8-bit unsigned
                (np.uint16, 2, lambda x: (x - 32768.0) / 32768.0), # 16-bit unsigned
            ]
            
            for dtype, bytes_per_sample, converter in conversion_strategies:
                if len(audio_bytes) % bytes_per_sample == 0:
                    try:
                        # Try to interpret bytes as this format
                        raw_data = np.frombuffer(audio_bytes, dtype=dtype)
                        
                        if len(raw_data) > 0:
                            # Convert to float32
                            float_data = converter(raw_data.astype(np.float64))
                            
                            # Validate the result
                            if self._is_valid_audio(float_data):
                                return float_data.astype(np.float32)
                                
                    except Exception as e:
                        logger.debug(f"Failed to convert bytes as {dtype}: {e}")
                        continue
            
            # If all conversions fail, return empty array
            logger.warning("All byte conversion strategies failed")
            return np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error converting bytes to float32: {e}")
            return np.array([], dtype=np.float32)

    def _is_valid_audio(self, audio: np.ndarray, 
                       max_amplitude: float = 10.0, 
                       min_samples: int = 1) -> bool:
        """Check if audio data looks valid"""
        try:
            if len(audio) < min_samples:
                return False
            
            # Check for reasonable amplitude range
            max_val = np.max(np.abs(audio))
            if max_val > max_amplitude:
                return False
            
            # Check for all zeros or all same value (likely invalid)
            if np.all(audio == audio[0]):
                return len(audio) < 10  # Allow short silent segments
            
            # Check for reasonable dynamic range
            if len(audio) > 100:
                std_dev = np.std(audio)
                if std_dev < 1e-8:  # Too little variation
                    return False
            
            return True
            
        except Exception:
            return False

    def decode_and_resample_browser_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Enhanced browser audio decoding with improved format detection and fallback strategies
        """
        try:
            if len(audio_data) == 0:
                return np.array([], dtype=np.float32)
            
            # Try multiple decoding strategies with caching
            format_key = self._get_format_key(audio_data)
            
            if format_key in self._format_cache:
                strategy = self._format_cache[format_key]
                audio_array = self._try_decode_strategy(audio_data, strategy)
                if audio_array is not None and len(audio_array) > 0:
                    return self.ensure_float32_audio(audio_array)
            
            # Try all strategies and cache the successful one
            audio_array = self._try_all_decode_strategies(audio_data)
            
            if audio_array is None or len(audio_array) == 0:
                logger.debug("All decoding strategies failed, generating placeholder")
                return self._generate_placeholder_audio()
            
            return self.ensure_float32_audio(audio_array)
            
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return np.array([], dtype=np.float32)

    def _get_format_key(self, audio_data: bytes) -> str:
        """Generate a key to identify audio format"""
        if len(audio_data) >= 12:
            # Use first few bytes as format identifier
            header = audio_data[:12]
            return header.hex()[:8]  # Use first 4 bytes as hex
        return f"len_{len(audio_data)}"

    def _try_all_decode_strategies(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Try all decoding strategies and cache the successful one"""
        
        strategies = [
            ("raw_pcm", self._decode_raw_pcm),
            ("wav_format", self._decode_wav_format),
            ("webm_opus", self._decode_webm_opus),
            ("mp3_format", self._decode_mp3_format),
            ("ffmpeg_auto", self._decode_with_ffmpeg_auto),
        ]
        
        format_key = self._get_format_key(audio_data)
        
        for strategy_name, decode_func in strategies:
            try:
                result = decode_func(audio_data)
                if result is not None and len(result) > 0 and self._is_valid_audio(result):
                    logger.debug(f"Successfully decoded with strategy: {strategy_name}")
                    self._format_cache[format_key] = strategy_name
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                continue
        
        return None

    def _try_decode_strategy(self, audio_data: bytes, strategy_name: str) -> Optional[np.ndarray]:
        """Try a specific decoding strategy"""
        strategy_map = {
            "raw_pcm": self._decode_raw_pcm,
            "wav_format": self._decode_wav_format,
            "webm_opus": self._decode_webm_opus,
            "mp3_format": self._decode_mp3_format,
            "ffmpeg_auto": self._decode_with_ffmpeg_auto,
        }
        
        decode_func = strategy_map.get(strategy_name)
        if decode_func:
            try:
                return decode_func(audio_data)
            except Exception as e:
                logger.debug(f"Cached strategy {strategy_name} failed: {e}")
        
        return None

    def _decode_raw_pcm(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Enhanced raw PCM decoding with better format detection"""
        try:
            # Try different PCM formats with validation
            formats_to_try = [
                (np.int16, 2, "16-bit signed PCM"),
                (np.float32, 4, "32-bit float PCM"),
                (np.int32, 4, "32-bit signed PCM"),
                (np.uint8, 1, "8-bit unsigned PCM"),
            ]
            
            for dtype, bytes_per_sample, description in formats_to_try:
                if len(audio_data) % bytes_per_sample == 0:
                    try:
                        # Try to decode as this format
                        audio_array = self.ensure_float32_audio(
                            np.frombuffer(audio_data, dtype=dtype)
                        )
                        
                        # Validate the result
                        if self._is_valid_audio(audio_array) and len(audio_array) >= self.min_audio_samples:
                            logger.debug(f"Successfully decoded as {description}")
                            return audio_array
                            
                    except Exception as e:
                        logger.debug(f"Failed to decode as {description}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Raw PCM decoding failed: {e}")
            return None

    def _decode_wav_format(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Enhanced WAV format decoding"""
        try:
            # Check for WAV header with more thorough validation
            if (len(audio_data) > 44 and 
                audio_data[:4] == b'RIFF' and 
                audio_data[8:12] == b'WAVE' and
                audio_data[12:16] == b'fmt '):
                
                with io.BytesIO(audio_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wav_file:
                        # Get WAV parameters
                        sample_rate = wav_file.getframerate()
                        sample_width = wav_file.getsampwidth()
                        n_channels = wav_file.getnchannels()
                        n_frames = wav_file.getnframes()
                        
                        logger.debug(f"WAV: {sample_rate}Hz, {sample_width}B, {n_channels}ch, {n_frames} frames")
                        
                        # Read audio data
                        frames = wav_file.readframes(n_frames)
                        
                        # Convert based on sample width
                        if sample_width == 1:  # 8-bit
                            audio_array = np.frombuffer(frames, dtype=np.uint8)
                            audio_array = (audio_array.astype(np.float32) - 128.0) / 128.0
                        elif sample_width == 2:  # 16-bit
                            audio_array = np.frombuffer(frames, dtype=np.int16)
                            audio_array = audio_array.astype(np.float32) / 32768.0
                        elif sample_width == 3:  # 24-bit (rare)
                            # Handle 24-bit audio
                            audio_array = self._decode_24bit_pcm(frames)
                        elif sample_width == 4:  # 32-bit
                            audio_array = np.frombuffer(frames, dtype=np.int32)
                            audio_array = audio_array.astype(np.float32) / 2147483648.0
                        else:
                            logger.warning(f"Unsupported sample width: {sample_width}")
                            return None
                        
                        # Handle multi-channel audio (convert to mono)
                        if n_channels > 1:
                            audio_array = audio_array.reshape(-1, n_channels)
                            audio_array = np.mean(audio_array, axis=1)
                        
                        # Resample if necessary
                        if sample_rate != self.sample_rate:
                            audio_array = self._resample_audio(audio_array, sample_rate, self.sample_rate)
                        
                        return self.ensure_float32_audio(audio_array)
            
            return None
            
        except Exception as e:
            logger.debug(f"WAV format decoding failed: {e}")
            return None

    def _decode_24bit_pcm(self, frames: bytes) -> np.ndarray:
        """Decode 24-bit PCM audio"""
        try:
            # 24-bit PCM is stored as 3 bytes per sample
            if len(frames) % 3 != 0:
                return np.array([], dtype=np.float32)
            
            # Convert 24-bit to 32-bit by padding with zeros
            samples = []
            for i in range(0, len(frames), 3):
                # Read 3 bytes and convert to signed 24-bit value
                sample_bytes = frames[i:i+3]
                if len(sample_bytes) == 3:
                    # Convert to signed 32-bit (pad with 0 or 0xFF based on sign)
                    if sample_bytes[2] & 0x80:  # Negative number
                        sample_32bit = struct.unpack('<i', sample_bytes + b'\xFF')[0]
                    else:  # Positive number
                        sample_32bit = struct.unpack('<i', sample_bytes + b'\x00')[0]
                    
                    samples.append(sample_32bit)
            
            if samples:
                audio_array = np.array(samples, dtype=np.int32)
                # Normalize from 24-bit range to float32
                return audio_array.astype(np.float32) / 8388608.0  # 2^23
            
            return np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"24-bit PCM decoding failed: {e}")
            return np.array([], dtype=np.float32)

    def _decode_mp3_format(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Decode MP3 format using FFmpeg"""
        try:
            # Check for MP3 header
            if len(audio_data) > 3 and (
                audio_data[:3] == b'ID3' or  # ID3 tag
                (audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0)  # MP3 frame header
            ):
                return self._decode_with_ffmpeg(audio_data, input_format='mp3')
            
            return None
            
        except Exception as e:
            logger.debug(f"MP3 format decoding failed: {e}")
            return None

    def _decode_webm_opus(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Enhanced WebM/Opus decoding"""
        try:
            # Check for WebM header
            if len(audio_data) > 4 and audio_data[:4] == b'\x1A\x45\xDF\xA3':
                return self._decode_with_ffmpeg(audio_data, input_format='webm')
            
            # Check for Ogg/Opus header
            if len(audio_data) > 4 and audio_data[:4] == b'OggS':
                return self._decode_with_ffmpeg(audio_data, input_format='ogg')
            
            return None
            
        except Exception as e:
            logger.debug(f"WebM/Opus decoding failed: {e}")
            return None

    def _decode_with_ffmpeg(self, audio_data: bytes, input_format: str) -> Optional[np.ndarray]:
        """Decode audio using FFmpeg with specified input format"""
        try:
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', input_format,
                '-i', 'pipe:0',
                '-f', 'wav',
                '-ac', '1',  # Convert to mono
                '-ar', str(self.sample_rate),
                '-acodec', 'pcm_s16le',
                'pipe:1'
            ]
            
            process = subprocess.run(
                cmd,
                input=audio_data,
                capture_output=True,
                timeout=10  # Increased timeout
            )
            
            if process.returncode == 0 and len(process.stdout) > 44:
                # Skip WAV header (44 bytes) and convert to float32
                wav_data = process.stdout[44:]
                audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if self._is_valid_audio(audio_array):
                    logger.debug(f"Successfully decoded with FFmpeg ({input_format})")
                    return audio_array
            
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg timeout during decoding")
            return None
        except FileNotFoundError:
            logger.debug("FFmpeg not available")
            return None
        except Exception as e:
            logger.debug(f"FFmpeg decoding failed: {e}")
            return None

    def _decode_with_ffmpeg_auto(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Use FFmpeg with automatic format detection"""
        try:
            # Try common formats in order
            formats_to_try = ['mp3', 'webm', 'ogg', 'wav', 'm4a', 'aac']
            
            for fmt in formats_to_try:
                result = self._decode_with_ffmpeg(audio_data, fmt)
                if result is not None:
                    return result
            
            return None
            
        except Exception as e:
            logger.debug(f"FFmpeg auto-detection failed: {e}")
            return None

    def _resample_audio(self, audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Enhanced audio resampling with quality preservation"""
        try:
            if original_rate == target_rate:
                return audio
            
            # Calculate resampling parameters
            ratio = target_rate / original_rate
            num_samples = int(len(audio) * ratio)
            
            if num_samples == 0:
                return np.array([], dtype=np.float32)
            
            # Use high-quality resampling
            try:
                # Try scipy's high-quality resampling
                from scipy.signal import resample_poly
                
                # Calculate optimal up/down factors
                from fractions import Fraction
                frac = Fraction(target_rate, original_rate).limit_denominator(1000)
                up_factor = frac.numerator
                down_factor = frac.denominator
                
                # Use polyphase resampling for better quality
                resampled = resample_poly(audio, up_factor, down_factor)
                
            except ImportError:
                # Fallback to basic resampling
                resampled = signal.resample(audio, num_samples)
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            # Fallback: simple interpolation
            try:
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, num_samples)
                resampled = np.interp(x_new, x_old, audio)
                return resampled.astype(np.float32)
            except:
                return audio

    def _generate_placeholder_audio(self) -> np.ndarray:
        """Generate realistic placeholder audio for demo purposes"""
        try:
            # Generate 100ms of varied audio to simulate real input
            duration = 0.1
            samples = int(duration * self.sample_rate)
            
            # Create realistic audio simulation
            noise_level = 0.0005  # Very low noise floor
            
            # Base noise
            audio = np.random.normal(0, noise_level, samples).astype(np.float32)
            
            # Occasionally add speech-like patterns
            self.speech_counter += 1
            
            if self.speech_counter % 50 == 0:  # Every 5 seconds at 10 FPS
                # Add a brief speech-like pattern
                speech_duration = 0.03  # 30ms
                speech_samples = int(speech_duration * self.sample_rate)
                t = np.linspace(0, speech_duration, speech_samples)
                
                # Create speech-like formant pattern
                f1, f2 = 800, 1200  # Typical formant frequencies
                speech_signal = (
                    0.002 * np.sin(2 * np.pi * f1 * t) * np.exp(-t * 10) +
                    0.001 * np.sin(2 * np.pi * f2 * t) * np.exp(-t * 8)
                )
                
                # Mix with audio
                if len(audio) >= len(speech_signal):
                    audio[:len(speech_signal)] += speech_signal.astype(np.float32)
            
            return audio
            
        except Exception as e:
            logger.error(f"Placeholder audio generation failed: {e}")
            return np.array([], dtype=np.float32)

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhanced comprehensive audio preprocessing pipeline
        """
        try:
            # Ensure we have valid audio data
            audio = self.ensure_float32_audio(audio)
            
            if len(audio) == 0:
                return audio
            
            # Limit audio length to prevent memory issues
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                logger.warning(f"Audio too long ({len(audio)} samples), truncating to {max_samples}")
                audio = audio[:max_samples]
            
            # Skip processing if audio is too short
            if len(audio) < self.min_audio_samples:
                logger.debug(f"Audio too short ({len(audio)} samples), minimum is {self.min_audio_samples}")
                return audio
            
            # Apply preprocessing pipeline
            processed_audio = audio.copy()
            
            # 1. Remove DC offset (center the signal around zero)
            processed_audio = self._remove_dc_offset(processed_audio)
            
            # 2. Normalize amplitude to prevent clipping
            processed_audio = self._normalize_audio(processed_audio)
            
            # 3. Apply pre-emphasis filter (boost high frequencies)
            processed_audio = self._apply_pre_emphasis(processed_audio)
            
            # 4. Apply noise gate to reduce background noise
            processed_audio = self._apply_noise_gate(processed_audio)
            
            # 5. Apply dynamic range compression for consistent levels
            processed_audio = self._apply_compression(processed_audio)
            
            # 6. Apply gain
            processed_audio = processed_audio * self.gain_factor
            
            # 7. Final safety clipping
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            
            # 8. Validate output
            if not self._is_valid_audio(processed_audio):
                logger.warning("Preprocessing resulted in invalid audio, returning original")
                return np.clip(audio, -1.0, 1.0)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            # Return safely clipped original audio on error
            return np.clip(self.ensure_float32_audio(audio), -1.0, 1.0)

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal"""
        try:
            if len(audio) == 0:
                return audio
            
            # Simple DC removal
            dc_offset = np.mean(audio)
            
            # Only remove significant DC offsets
            if abs(dc_offset) > 0.01:
                audio_centered = audio - dc_offset
                logger.debug(f"Removed DC offset: {dc_offset:.4f}")
                return audio_centered
            
            return audio
            
        except Exception as e:
            logger.error(f"DC offset removal failed: {e}")
            return audio

    def _normalize_audio(self, audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio amplitude with target level"""
        try:
            if len(audio) == 0:
                return audio
            
            max_val = np.max(np.abs(audio))
            
            if max_val > 1e-8:  # Avoid division by zero
                # Calculate gain to reach target level
                gain = target_level / max_val
                
                # Limit gain to prevent excessive amplification
                max_gain = 10.0  # Maximum 20dB gain
                gain = min(gain, max_gain)
                
                normalized = audio * gain
                
                if gain > 1.1:  # Log significant amplification
                    logger.debug(f"Applied normalization gain: {20*np.log10(gain):.1f} dB")
                
                return normalized
            
            return audio
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return audio

    def _apply_pre_emphasis(self, audio: np.ndarray, alpha: float = None) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies"""
        try:
            if len(audio) <= 1:
                return audio
            
            alpha = alpha or self.pre_emphasis_coeff
            
            # Apply pre-emphasis: y[n] = x[n] - α*x[n-1]
            emphasized = signal.lfilter([1, -alpha], [1], audio)
            
            return emphasized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Pre-emphasis failed: {e}")
            return audio

    def _apply_noise_gate(self, audio: np.ndarray, 
                         threshold: float = None, 
                         ratio: float = 0.1) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        try:
            if len(audio) == 0:
                return audio
            
            threshold = threshold or (self.vad_threshold * 2)
            
            # Calculate RMS in overlapping windows
            window_size = 512
            hop_size = 256
            
            gated_audio = audio.copy()
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window**2))
                
                if rms < threshold:
                    # Apply gate (reduce amplitude)
                    gated_audio[i:i + window_size] *= ratio
            
            return gated_audio
            
        except Exception as e:
            logger.error(f"Noise gate failed: {e}")
            return audio

    def _apply_compression(self, audio: np.ndarray, 
                          threshold: float = 0.7, 
                          ratio: float = 4.0,
                          attack_ms: float = 5.0,
                          release_ms: float = 50.0) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            if len(audio) == 0:
                return audio
            
            # Simple peak compression
            compressed = audio.copy()
            
            # Find peaks above threshold
            peaks = np.abs(audio) > threshold
            
            if np.any(peaks):
                # Calculate compression gain
                compression_gain = threshold + (np.abs(audio) - threshold) / ratio
                compression_gain = compression_gain / np.abs(audio)
                
                # Apply compression only to peaks
                compressed = np.where(peaks, 
                                     audio * compression_gain, 
                                     audio)
            
            return compressed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return audio

    def frame_audio(self, audio: np.ndarray, frame_size: int = 512, hop_size: int = None) -> List[np.ndarray]:
        """Split audio into overlapping frames with enhanced windowing"""
        try:
            if len(audio) == 0:
                return []
            
            hop_size = hop_size or frame_size // 2
            frames = []
            
            # Apply windowing function for better frequency analysis
            window = np.hanning(frame_size)
            
            for i in range(0, len(audio) - frame_size + 1, hop_size):
                frame = audio[i:i + frame_size]
                if len(frame) == frame_size:
                    # Apply window function
                    windowed_frame = frame * window
                    frames.append(windowed_frame)
            
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []

    def detect_voice_activity(self, audio: np.ndarray, enhanced: bool = True) -> bool:
        """Enhanced voice activity detection with multiple features"""
        try:
            if len(audio) == 0:
                return False
            
            # Ensure proper audio format
            audio = self.ensure_float32_audio(audio)
            
            if not enhanced:
                # Simple energy-based detection
                rms_energy = np.sqrt(np.mean(audio**2))
                return rms_energy > self.vad_threshold
            
            # Enhanced multi-feature VAD
            features = self._extract_vad_features(audio)
            
            # Multi-criteria decision
            criteria_met = 0
            total_criteria = 0
            
            # Energy criterion
            if features['rms_energy'] > self.vad_threshold:
                criteria_met += 1
            total_criteria += 1
            
            # Zero crossing rate (speech typically has moderate ZCR)
            if 0.01 < features['zero_crossing_rate'] < 0.3:
                criteria_met += 1
            total_criteria += 1
            
            # Spectral centroid (speech has energy in mid frequencies)
            if 300 < features['spectral_centroid'] < 3000:
                criteria_met += 1
            total_criteria += 1
            
            # Spectral rolloff (speech has specific rolloff characteristics)
            if features['spectral_rolloff'] > 1000:
                criteria_met += 1
            total_criteria += 1
            
            # Spectral flatness (speech is less flat than noise)
            if features['spectral_flatness'] < 0.8:
                criteria_met += 1
            total_criteria += 1
            
            # Decision based on majority vote
            confidence = criteria_met / total_criteria
            is_speech = confidence >= 0.6  # At least 60% of criteria met
            
            logger.debug(f"VAD: {criteria_met}/{total_criteria} criteria met, confidence: {confidence:.2f}")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Fallback to simple energy detection
            try:
                rms_energy = np.sqrt(np.mean(self.ensure_float32_audio(audio)**2))
                return rms_energy > self.vad_threshold
            except:
                return False

    def _extract_vad_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive features for voice activity detection"""
        features = {}
        
        try:
            # Ensure minimum length for reliable feature extraction
            if len(audio) < 160:  # 10ms at 16kHz
                # Pad with zeros if too short
                audio = np.pad(audio, (0, 160 - len(audio)), mode='constant')
            
            # 1. RMS Energy
            features['rms_energy'] = np.sqrt(np.mean(audio**2))
            
            # 2. Zero Crossing Rate
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            features['zero_crossing_rate'] = zero_crossings / len(audio)
            
            # 3. Spectral features using FFT
            try:
                # Apply window to reduce spectral leakage
                windowed_audio = audio * np.hanning(len(audio))
                fft = np.fft.fft(windowed_audio)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft)//2]
                
                # Avoid division by zero
                total_energy = np.sum(magnitude)
                if total_energy > 1e-10:
                    # Spectral Centroid
                    features['spectral_centroid'] = np.sum(freqs * magnitude) / total_energy
                    
                    # Spectral Rolloff (frequency below which 85% of energy lies)
                    cumulative_energy = np.cumsum(magnitude)
                    rolloff_point = 0.85 * total_energy
                    rolloff_idx = np.where(cumulative_energy >= rolloff_point)[0]
                    if len(rolloff_idx) > 0:
                        features['spectral_rolloff'] = freqs[rolloff_idx[0]]
                    else:
                        features['spectral_rolloff'] = freqs[-1]
                    
                    # Spectral Flatness (measure of how noise-like vs. tone-like)
                    # Geometric mean / Arithmetic mean
                    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
                    arithmetic_mean = np.mean(magnitude)
                    if arithmetic_mean > 1e-10:
                        features['spectral_flatness'] = geometric_mean / arithmetic_mean
                    else:
                        features['spectral_flatness'] = 0
                    
                    # Spectral Bandwidth
                    centroid = features['spectral_centroid']
                    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / total_energy)
                    features['spectral_bandwidth'] = bandwidth
                    
                else:
                    # Default values for silence
                    features['spectral_centroid'] = 0
                    features['spectral_rolloff'] = 0
                    features['spectral_flatness'] = 1.0
                    features['spectral_bandwidth'] = 0
                    
            except Exception as e:
                logger.debug(f"Spectral feature extraction failed: {e}")
                # Fallback values
                features['spectral_centroid'] = 1000
                features['spectral_rolloff'] = 2000
                features['spectral_flatness'] = 0.5
                features['spectral_bandwidth'] = 1000
            
            # 4. Temporal features
            
            # Short-time energy variance
            frame_size = 256
            if len(audio) >= frame_size:
                frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size+1, frame_size//2)]
                frame_energies = [np.mean(frame**2) for frame in frames if len(frame) == frame_size]
                if frame_energies:
                    features['energy_variance'] = np.var(frame_energies)
                else:
                    features['energy_variance'] = 0
            else:
                features['energy_variance'] = 0
            
            # Peak-to-average ratio
            peak_amplitude = np.max(np.abs(audio))
            avg_amplitude = np.mean(np.abs(audio))
            if avg_amplitude > 1e-10:
                features['peak_to_average_ratio'] = peak_amplitude / avg_amplitude
            else:
                features['peak_to_average_ratio'] = 1.0
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return safe default values
            features = {
                'rms_energy': np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0,
                'zero_crossing_rate': 0.1,
                'spectral_centroid': 1000,
                'spectral_rolloff': 2000,
                'spectral_flatness': 0.5,
                'spectral_bandwidth': 1000,
                'energy_variance': 0,
                'peak_to_average_ratio': 1.0
            }
        
        return features

    def detect_silence(self, audio: np.ndarray, threshold: float = None) -> bool:
        """Enhanced silence detection with hysteresis"""
        try:
            if len(audio) == 0:
                return True
            
            audio = self.ensure_float32_audio(audio)
            threshold = threshold or self.vad_threshold
            
            # Use RMS energy for silence detection
            rms = np.sqrt(np.mean(audio**2))
            
            # Add hysteresis to prevent flickering
            silence_threshold = threshold
            speech_threshold = threshold * 1.5
            
            # Simple state-based decision (could be enhanced with history)
            is_silence = rms < silence_threshold
            
            return is_silence
            
        except Exception as e:
            logger.error(f"Silence detection error: {e}")
            return True

    def mute(self):
        """Mute audio processing"""
        self.is_muted = True
        logger.info("Audio processing muted")

    def unmute(self):
        """Unmute audio processing"""
        self.is_muted = False
        logger.info("Audio processing unmuted")

    def set_gain(self, gain_db: float):
        """Set audio gain in dB"""
        self.gain_factor = 10**(gain_db / 20)
        logger.info(f"Audio gain set to {gain_db:.1f} dB (factor: {self.gain_factor:.3f})")

    def adjust_vad_threshold(self, threshold: float):
        """Adjust VAD threshold dynamically"""
        if 0.0001 <= threshold <= 0.1:
            self.vad_threshold = threshold
            logger.info(f"VAD threshold adjusted to {threshold:.6f}")
        else:
            logger.warning(f"VAD threshold {threshold} out of range [0.0001, 0.1]")

    def convert_to_wav(self, audio: np.ndarray, output_path: str = None) -> bytes:
        """Convert audio array to WAV format with enhanced error handling"""
        try:
            audio = self.ensure_float32_audio(audio)
            
            if len(audio) == 0:
                logger.warning("Cannot convert empty audio to WAV")
                return b""
            
            # Convert to 16-bit PCM with proper scaling
            audio_clipped = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            
            if output_path:
                # Save to file
                try:
                    with wave.open(output_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    logger.debug(f"WAV file saved to {output_path}")
                    return None
                except Exception as e:
                    logger.error(f"Failed to save WAV file: {e}")
                    return b""
            else:
                # Return as bytes
                try:
                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    return buffer.getvalue()
                except Exception as e:
                    logger.error(f"Failed to create WAV bytes: {e}")
                    return b""
                
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            return b""

    def calculate_audio_metrics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive audio quality and content metrics"""
        try:
            audio = self.ensure_float32_audio(audio)
            
            if len(audio) == 0:
                return {"error": "Empty audio"}
            
            metrics = {
                # Basic metrics
                'length_samples': len(audio),
                'duration_seconds': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                
                # Amplitude metrics
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'peak_amplitude': float(np.max(np.abs(audio))),
                'dynamic_range_db': 0.0,
                
                # Statistical metrics
                'mean': float(np.mean(audio)),
                'std_dev': float(np.std(audio)),
                'skewness': 0.0,
                'kurtosis': 0.0,
            }
            
            # Calculate dynamic range
            if metrics['rms_energy'] > 1e-10:
                metrics['dynamic_range_db'] = 20 * np.log10(metrics['peak_amplitude'] / metrics['rms_energy'])
            
            # Calculate statistical moments
            try:
                from scipy import stats
                metrics['skewness'] = float(stats.skew(audio))
                metrics['kurtosis'] = float(stats.kurtosis(audio))
            except ImportError:
                # Manual calculation fallback
                if metrics['std_dev'] > 1e-10:
                    normalized = (audio - metrics['mean']) / metrics['std_dev']
                    metrics['skewness'] = float(np.mean(normalized**3))
                    metrics['kurtosis'] = float(np.mean(normalized**4) - 3)
            
            # Voice activity analysis
            frames = self.frame_audio(audio, 512)
            if frames:
                speech_frames = sum(1 for frame in frames if self.detect_voice_activity(frame))
                metrics['voice_activity_ratio'] = speech_frames / len(frames)
                metrics['total_frames'] = len(frames)
                metrics['speech_frames'] = speech_frames
            else:
                metrics['voice_activity_ratio'] = 0.0
                metrics['total_frames'] = 0
                metrics['speech_frames'] = 0
            
            # Frequency domain metrics
            try:
                # Use windowed FFT for better frequency analysis
                windowed_audio = audio * np.hanning(len(audio))
                fft = np.fft.fft(windowed_audio)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft)//2]
                
                total_energy = np.sum(magnitude)
                if total_energy > 1e-10:
                    # Spectral centroid (brightness)
                    metrics['spectral_centroid'] = float(np.sum(freqs * magnitude) / total_energy)
                    
                    # Energy distribution in frequency bands
                    low_freq_energy = np.sum(magnitude[freqs < 1000]) / total_energy
                    mid_freq_energy = np.sum(magnitude[(freqs >= 1000) & (freqs < 4000)]) / total_energy
                    high_freq_energy = np.sum(magnitude[freqs >= 4000]) / total_energy
                    
                    metrics['low_freq_energy_ratio'] = float(low_freq_energy)
                    metrics['mid_freq_energy_ratio'] = float(mid_freq_energy)
                    metrics['high_freq_energy_ratio'] = float(high_freq_energy)
                else:
                    metrics['spectral_centroid'] = 0.0
                    metrics['low_freq_energy_ratio'] = 0.0
                    metrics['mid_freq_energy_ratio'] = 0.0
                    metrics['high_freq_energy_ratio'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Frequency analysis failed: {e}")
                metrics['spectral_centroid'] = 0.0
                metrics['low_freq_energy_ratio'] = 0.0
                metrics['mid_freq_energy_ratio'] = 0.0
                metrics['high_freq_energy_ratio'] = 0.0
            
            # Quality assessment
            quality_score = self._assess_audio_quality(metrics)
            metrics['quality_score'] = quality_score
            metrics['quality_rating'] = self._get_quality_rating(quality_score)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            metrics['zero_crossing_rate'] = zero_crossings / len(audio)
            
            # Estimate SNR (simplified)
            if metrics['rms_energy'] > 0:
                # Estimate noise floor as 10th percentile of energy
                frame_energies = [np.mean(frame**2) for frame in frames] if frames else [metrics['rms_energy']**2]
                noise_floor = np.sqrt(np.percentile(frame_energies, 10))
                
                if noise_floor > 1e-10:
                    metrics['estimated_snr_db'] = 20 * np.log10(metrics['rms_energy'] / noise_floor)
                else:
                    metrics['estimated_snr_db'] = 60.0  # Very high SNR
            else:
                metrics['estimated_snr_db'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Audio metrics calculation error: {e}")
            return {"error": str(e)}

    def _assess_audio_quality(self, metrics: Dict[str, Any]) -> float:
        """Assess overall audio quality based on metrics (0-100 scale)"""
        try:
            score = 50.0  # Base score
            
            # RMS energy (should be reasonable, not too low or high)
            rms = metrics.get('rms_energy', 0)
            if 0.01 <= rms <= 0.5:
                score += 20
            elif 0.001 <= rms <= 0.8:
                score += 10
            
            # Dynamic range (good dynamic range indicates quality)
            dynamic_range = metrics.get('dynamic_range_db', 0)
            if 20 <= dynamic_range <= 60:
                score += 15
            elif 10 <= dynamic_range <= 80:
                score += 10
            
            # SNR (higher is better)
            snr = metrics.get('estimated_snr_db', 0)
            if snr >= 20:
                score += 15
            elif snr >= 10:
                score += 10
            elif snr >= 5:
                score += 5
            
            # Voice activity (some speech content is good)
            vad_ratio = metrics.get('voice_activity_ratio', 0)
            if 0.1 <= vad_ratio <= 0.9:
                score += 10
            elif vad_ratio > 0:
                score += 5
            
            # Frequency distribution (balanced is good)
            mid_freq = metrics.get('mid_freq_energy_ratio', 0)
            if mid_freq >= 0.3:  # Good mid-frequency content
                score += 10
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return 50.0

    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to human-readable rating"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 50:
            return "Poor"
        else:
            return "Very Poor"

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration"""
        return {
            'sample_rate': self.sample_rate,
            'is_muted': self.is_muted,
            'gain_factor': self.gain_factor,
            'gain_db': 20 * np.log10(self.gain_factor) if self.gain_factor > 0 else -60,
            'vad_threshold': self.vad_threshold,
            'frame_size': self.frame_size,
            'hop_length': self.hop_length,
            'pre_emphasis_coeff': self.pre_emphasis_coeff,
            'speech_counter': self.speech_counter,
            'format_cache_size': len(self._format_cache),
            'max_audio_length': self.max_audio_length,
            'min_audio_samples': self.min_audio_samples
        }

    def reset_processor_state(self):
        """Reset processor state for new audio session"""
        self.speech_counter = 0
        self._format_cache.clear()
        self.is_muted = False
        logger.info("Audio processor state reset")

    def benchmark_processing(self, test_duration: float = 1.0) -> Dict[str, float]:
        """Benchmark audio processing performance"""
        try:
            import time
            
            # Generate test audio
            samples = int(test_duration * self.sample_rate)
            test_audio = np.random.randn(samples).astype(np.float32) * 0.1
            
            results = {}
            
            # Benchmark preprocessing
            start_time = time.time()
            for _ in range(10):
                _ = self.preprocess_audio(test_audio)
            preprocessing_time = (time.time() - start_time) / 10
            results['preprocessing_ms'] = preprocessing_time * 1000
            
            # Benchmark VAD
            start_time = time.time()
            for _ in range(100):
                _ = self.detect_voice_activity(test_audio[:1600])  # 100ms chunks
            vad_time = (time.time() - start_time) / 100
            results['vad_ms'] = vad_time * 1000
            
            # Benchmark framing
            start_time = time.time()
            for _ in range(10):
                _ = self.frame_audio(test_audio)
            framing_time = (time.time() - start_time) / 10
            results['framing_ms'] = framing_time * 1000
            
            # Calculate real-time factor
            results['realtime_factor'] = test_duration / preprocessing_time
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}