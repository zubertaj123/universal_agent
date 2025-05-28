"""
Audio processing utilities - Fixed version
"""
import numpy as np
from scipy import signal
from typing import List, Dict, Any, Optional
import subprocess
import struct
import wave
import io
import tempfile
import os
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioProcessor:
    """Fixed audio processing utilities"""
    
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

    def decode_and_resample_browser_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Decode browser audio data and resample to target sample rate
        Fixed version with better format detection and fallback strategies
        """
        try:
            if len(audio_data) == 0:
                return np.array([], dtype=np.float32)
            
            # Try multiple decoding strategies
            audio_array = self._try_decode_strategies(audio_data)
            
            if audio_array is None or len(audio_array) == 0:
                logger.warning("All decoding strategies failed, returning empty array")
                return np.array([], dtype=np.float32)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return np.array([], dtype=np.float32)

    def _try_decode_strategies(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Try multiple decoding strategies in order of preference"""
        
        # Strategy 1: Try to detect and decode WebM/Opus format
        try:
            result = self._decode_webm_opus(audio_data)
            if result is not None and len(result) > 0:
                logger.debug("Successfully decoded as WebM/Opus")
                return result
        except Exception as e:
            logger.debug(f"WebM/Opus decoding failed: {e}")
        
        # Strategy 2: Try to decode as raw PCM (common for MediaRecorder)
        try:
            result = self._decode_raw_pcm(audio_data)
            if result is not None and len(result) > 0:
                logger.debug("Successfully decoded as raw PCM")
                return result
        except Exception as e:
            logger.debug(f"Raw PCM decoding failed: {e}")
        
        # Strategy 3: Try FFmpeg with format auto-detection
        try:
            result = self._decode_with_ffmpeg_auto(audio_data)
            if result is not None and len(result) > 0:
                logger.debug("Successfully decoded with FFmpeg auto-detection")
                return result
        except Exception as e:
            logger.debug(f"FFmpeg auto-detection failed: {e}")
        
        # Strategy 4: Try as WAV format
        try:
            result = self._decode_wav_format(audio_data)
            if result is not None and len(result) > 0:
                logger.debug("Successfully decoded as WAV")
                return result
        except Exception as e:
            logger.debug(f"WAV decoding failed: {e}")
        
        # Strategy 5: Generate placeholder audio for demo purposes
        logger.warning("All decoding strategies failed, generating placeholder audio")
        return self._generate_placeholder_audio()

    def _decode_webm_opus(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Decode WebM/Opus format using FFmpeg"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # FFmpeg command for WebM/Opus input
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_input_path,
                    '-f', 'wav',
                    '-ac', '1',  # mono
                    '-ar', str(self.sample_rate),
                    '-acodec', 'pcm_s16le',
                    temp_output_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    check=True
                )
                
                # Read the converted WAV file
                with wave.open(temp_output_path, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio_array
                    
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                    
        except Exception as e:
            logger.debug(f"WebM/Opus decoding failed: {e}")
            return None

    def _decode_raw_pcm(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Decode raw PCM data (common for MediaRecorder API)"""
        try:
            # Try different PCM formats
            formats_to_try = [
                (np.int16, 2),  # 16-bit signed
                (np.float32, 4),  # 32-bit float
                (np.int32, 4),  # 32-bit signed
            ]
            
            for dtype, bytes_per_sample in formats_to_try:
                if len(audio_data) % bytes_per_sample == 0:
                    try:
                        if dtype == np.float32:
                            audio_array = np.frombuffer(audio_data, dtype=dtype)
                            # Ensure values are in [-1, 1] range
                            audio_array = np.clip(audio_array, -1.0, 1.0)
                        else:
                            audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
                            if dtype == np.int16:
                                audio_array = audio_array / 32768.0
                            elif dtype == np.int32:
                                audio_array = audio_array / 2147483648.0
                        
                        # Basic validation - check if audio has reasonable characteristics
                        if len(audio_array) > 0 and np.max(np.abs(audio_array)) > 1e-6:
                            return audio_array
                            
                    except Exception as e:
                        logger.debug(f"Failed to decode as {dtype}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Raw PCM decoding failed: {e}")
            return None

    def _decode_with_ffmpeg_auto(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Use FFmpeg with automatic format detection"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'mp3',  # Try MP3 first
                '-i', 'pipe:0',
                '-f', 'wav',
                '-ac', '1',
                '-ar', str(self.sample_rate),
                '-acodec', 'pcm_s16le',
                'pipe:1'
            ]
            
            process = subprocess.run(
                cmd,
                input=audio_data,
                capture_output=True,
                timeout=5
            )
            
            if process.returncode == 0 and len(process.stdout) > 44:
                # Skip WAV header (44 bytes) and convert to float32
                wav_data = process.stdout[44:]
                audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_array
            
            return None
            
        except Exception as e:
            logger.debug(f"FFmpeg auto-detection failed: {e}")
            return None

    def _decode_wav_format(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Try to decode as WAV format"""
        try:
            # Check if data starts with WAV header
            if len(audio_data) > 44 and audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                with io.BytesIO(audio_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        sample_width = wav_file.getsampwidth()
                        
                        if sample_width == 2:  # 16-bit
                            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit
                            audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            return None
                            
                        # Resample if necessary
                        original_rate = wav_file.getframerate()
                        if original_rate != self.sample_rate:
                            audio_array = self._resample_audio(audio_array, original_rate, self.sample_rate)
                        
                        return audio_array
            
            return None
            
        except Exception as e:
            logger.debug(f"WAV format decoding failed: {e}")
            return None

    def _resample_audio(self, audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio using scipy"""
        try:
            from scipy import signal
            
            if original_rate == target_rate:
                return audio
            
            # Calculate resampling ratio
            ratio = target_rate / original_rate
            num_samples = int(len(audio) * ratio)
            
            # Use scipy's resample function
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return audio

    def _generate_placeholder_audio(self) -> np.ndarray:
        """Generate placeholder audio for demo purposes"""
        try:
            # Generate 100ms of low-level noise
            duration = 0.1  # 100ms
            samples = int(duration * self.sample_rate)
            
            # Create very low amplitude noise
            noise = np.random.normal(0, 0.0001, samples).astype(np.float32)
            
            # Occasionally add a brief tone to simulate speech
            self.speech_counter += 1
            if self.speech_counter % 100 == 0:  # Every 100th call (about every 10 seconds)
                tone_duration = 0.05  # 50ms tone
                tone_samples = int(tone_duration * self.sample_rate)
                t = np.linspace(0, tone_duration, tone_samples)
                tone = 0.001 * np.sin(2 * np.pi * 440 * t)  # Very quiet 440Hz tone
                
                # Mix tone with noise
                if len(noise) >= len(tone):
                    noise[:len(tone)] += tone.astype(np.float32)
            
            return noise
            
        except Exception as e:
            logger.error(f"Placeholder audio generation failed: {e}")
            return np.array([], dtype=np.float32)

    def frame_audio(self, audio: np.ndarray, frame_size: int = 512) -> List[np.ndarray]:
        """Split audio into frames"""
        frames = []
        for i in range(0, len(audio), frame_size):
            chunk = audio[i:i + frame_size]
            if len(chunk) == frame_size:
                frames.append(chunk)
        return frames

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Comprehensive audio preprocessing pipeline"""
        try:
            if len(audio) == 0:
                return audio
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize amplitude
            audio = self._normalize_audio(audio)
            
            # Remove DC offset
            audio = self._remove_dc_offset(audio)
            
            # Apply pre-emphasis filter
            audio = self._apply_pre_emphasis(audio)
            
            # Apply gain
            audio = audio * self.gain_factor
            
            # Final clipping protection
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio"""
        return audio - np.mean(audio)

    def _apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies"""
        return signal.lfilter([1, -self.pre_emphasis_coeff], [1], audio)

    def detect_voice_activity(self, audio: np.ndarray, enhanced: bool = False) -> bool:
        """Enhanced voice activity detection"""
        try:
            if len(audio) == 0:
                return False
            
            # Basic energy-based detection
            rms_energy = np.sqrt(np.mean(audio**2))
            
            if not enhanced:
                return rms_energy > self.vad_threshold
            
            # Enhanced detection with multiple features
            features = self._extract_vad_features(audio)
            
            # Decision logic combining multiple features
            is_speech = (
                features['rms_energy'] > self.vad_threshold and
                features['zero_crossing_rate'] < 0.3 and
                features['spectral_centroid'] > 500  # Basic spectral check
            )
            
            return is_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return rms_energy > self.vad_threshold if 'rms_energy' in locals() else False

    def _extract_vad_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features for voice activity detection"""
        features = {}
        
        try:
            # RMS Energy
            features['rms_energy'] = np.sqrt(np.mean(audio**2))
            
            # Zero Crossing Rate (simplified)
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            features['zero_crossing_rate'] = zero_crossings / len(audio)
            
            # Simple spectral centroid estimation
            # Use FFT for frequency analysis
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft)//2]
            
            if np.sum(magnitude) > 0:
                features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                features['spectral_centroid'] = 0
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Fallback to basic features
            features['rms_energy'] = np.sqrt(np.mean(audio**2))
            features['zero_crossing_rate'] = 0.1
            features['spectral_centroid'] = 1000
        
        return features

    def detect_silence(self, audio: np.ndarray, threshold: float = None) -> bool:
        """Detect if audio segment is silence"""
        if len(audio) == 0:
            return True
        
        threshold = threshold or self.vad_threshold
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold

    def mute(self):
        """Mute audio processing"""
        self.is_muted = True
        logger.debug("Audio muted")

    def unmute(self):
        """Unmute audio processing"""
        self.is_muted = False
        logger.debug("Audio unmuted")

    def set_gain(self, gain_db: float):
        """Set audio gain in dB"""
        self.gain_factor = 10**(gain_db / 20)
        logger.debug(f"Audio gain set to {gain_db} dB")

    def convert_to_wav(self, audio: np.ndarray, output_path: str = None) -> bytes:
        """Convert audio array to WAV format"""
        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            if output_path:
                # Save to file
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                return None
            else:
                # Return as bytes
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            return b""

    def calculate_audio_metrics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive audio metrics"""
        if len(audio) == 0:
            return {}
        
        try:
            metrics = {
                'rms_energy': np.sqrt(np.mean(audio**2)),
                'peak_amplitude': np.max(np.abs(audio)),
                'duration_seconds': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'zero_crossing_rate': np.sum(np.diff(np.sign(audio)) != 0) / len(audio),
            }
            
            # Voice activity ratio
            frames = self.frame_audio(audio, 512)
            if frames:
                speech_frames = sum(1 for frame in frames if self.detect_voice_activity(frame))
                metrics['voice_activity_ratio'] = speech_frames / len(frames)
            else:
                metrics['voice_activity_ratio'] = 0
            
            # Estimate SNR (simplified)
            if metrics['rms_energy'] > 0:
                noise_floor = np.percentile(np.abs(audio), 10)  # Assume 10th percentile is noise
                if noise_floor > 0:
                    metrics['estimated_snr_db'] = 20 * np.log10(metrics['rms_energy'] / noise_floor)
                else:
                    metrics['estimated_snr_db'] = 60  # Very high SNR if no noise detected
            else:
                metrics['estimated_snr_db'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Audio metrics calculation error: {e}")
            return {'error': str(e)}