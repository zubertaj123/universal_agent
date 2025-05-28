"""
Audio processing utilities
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List, Dict, Any
import librosa
import io
import subprocess
import wave
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioProcessor:
    """Comprehensive audio processing utilities"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_muted = False
        self.gain_factor = 1.0
        self.noise_profile = None
        
        # Audio processing parameters
        self.frame_size = 1024
        self.hop_length = 512
        self.pre_emphasis_coeff = 0.97
        
        # Voice activity detection parameters
        self.vad_threshold = 0.01
        self.min_speech_duration = 0.3  # seconds
        self.min_silence_duration = 0.5  # seconds

    def decode_and_resample_browser_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode browser audio format and resample to target sample rate
        Fixed method to handle FFmpeg properly
        """
        try:
            # Try using subprocess to call ffmpeg directly
            cmd = [
                'ffmpeg',
                '-f', 'webm',  # Input format
                '-i', 'pipe:0',  # Input from stdin
                '-f', 'wav',     # Output format
                '-ac', '1',      # Mono
                '-ar', str(self.sample_rate),  # Sample rate
                'pipe:1'         # Output to stdout
            ]
            
            process = subprocess.run(
                cmd,
                input=audio_bytes,
                capture_output=True,
                timeout=5  # 5 second timeout
            )
            
            if process.returncode == 0:
                # Convert WAV bytes to numpy array
                # Skip WAV header (44 bytes) and convert to float32
                wav_data = process.stdout[44:]  # Skip WAV header
                audio_data = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_data
            else:
                logger.warning(f"FFmpeg failed with return code {process.returncode}: {process.stderr.decode()}")
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout - audio processing took too long")
        except FileNotFoundError:
            logger.warning("FFmpeg not found in PATH - using fallback audio processing")
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
        
        # Fallback: try to interpret as raw audio data
        try:
            # Assume it's raw 16-bit PCM audio
            if len(audio_bytes) % 2 == 0:  # Even number of bytes for 16-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_data
            else:
                logger.warning("Odd number of bytes in audio data - padding")
                padded_bytes = audio_bytes + b'\x00'
                audio_data = np.frombuffer(padded_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_data
        except Exception as e:
            logger.error(f"Fallback audio processing failed: {e}")
            return np.array([], dtype=np.float32)

    def frame_audio(self, audio: np.ndarray, frame_size: int = 512) -> List[np.ndarray]:
        """
        Split audio into frames of frame_size samples.
        Returns a list of numpy arrays.
        """
        frames = []
        for i in range(0, len(audio), frame_size):
            chunk = audio[i:i + frame_size]
            if len(chunk) == frame_size:
                frames.append(chunk)
        return frames

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Comprehensive audio preprocessing pipeline"""
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Handle empty or very short audio
            if len(audio) < 100:
                return audio
            
            # Normalize amplitude
            audio = self._normalize_audio(audio)
            
            # Remove DC offset
            audio = self._remove_dc_offset(audio)
            
            # Apply pre-emphasis filter
            audio = self._apply_pre_emphasis(audio)
            
            # Noise reduction if profile available
            if self.noise_profile is not None:
                audio = self._reduce_noise(audio)
            
            # Apply gain
            audio = audio * self.gain_factor
            
            # Final clipping protection
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio  # Return original audio on error
    
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
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction using spectral subtraction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply spectral subtraction
            if self.noise_profile is not None:
                # Expand noise profile to match magnitude shape
                noise_mag = np.mean(self.noise_profile)
                alpha = 2.0  # Over-subtraction factor
                beta = 0.01  # Spectral floor
                
                # Spectral subtraction
                clean_magnitude = magnitude - alpha * noise_mag
                clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
            else:
                clean_magnitude = magnitude
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft, hop_length=self.hop_length)
            
            return clean_audio
            
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return audio
    
    def detect_silence(self, audio: np.ndarray, threshold: float = None) -> bool:
        """Detect if audio segment is silence"""
        if threshold is None:
            threshold = self.vad_threshold
            
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio**2))
        
        # Also check zero-crossing rate
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        # Combine RMS and ZCR for better silence detection
        is_silent = (rms < threshold) and (zcr < 0.1)
        
        return is_silent
    
    def detect_voice_activity(self, audio: np.ndarray, enhanced: bool = True) -> bool:
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
                features['spectral_centroid'] > 1000 and  # Typical for speech
                features['spectral_rolloff'] > 2000 and
                features['zero_crossing_rate'] < 0.3 and
                features['mfcc_variance'] > 0.1
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
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # Spectral features
            stft = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # MFCC variance (speech typically has higher variance)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_variance'] = np.var(mfccs)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Fallback to basic features
            features['rms_energy'] = np.sqrt(np.mean(audio**2))
            features['zero_crossing_rate'] = 0.1
            features['spectral_centroid'] = 1500
            features['spectral_rolloff'] = 3000
            features['mfcc_variance'] = 0.2
            features['spectral_bandwidth'] = 1000
        
        return features
    
    def split_audio_on_silence(
        self,
        audio: np.ndarray,
        min_silence_duration: float = None,
        silence_threshold: float = None
    ) -> List[Tuple[int, int]]:
        """Split audio on silence and return segment boundaries"""
        if min_silence_duration is None:
            min_silence_duration = self.min_silence_duration
        if silence_threshold is None:
            silence_threshold = self.vad_threshold
        
        # Calculate frame parameters
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        min_silence_frames = int(min_silence_duration * self.sample_rate / hop_length)
        
        # Analyze each frame
        speech_frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            is_speech = not self.detect_silence(frame, silence_threshold)
            speech_frames.append(is_speech)
        
        # Find speech segments
        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                if not in_speech:
                    # Start of new speech segment
                    in_speech = True
                    start_frame = i
                    silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # End of speech segment
                        end_frame = i - silence_count
                        start_sample = start_frame * hop_length
                        end_sample = end_frame * hop_length
                        
                        if end_sample > start_sample:
                            segments.append((start_sample, end_sample))
                        
                        in_speech = False
                        silence_count = 0
        
        # Handle case where audio ends during speech
        if in_speech:
            end_sample = len(audio)
            start_sample = start_frame * hop_length
            if end_sample > start_sample:
                segments.append((start_sample, end_sample))
        
        return segments
    
    def extract_speech_segments(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract speech segments from audio"""
        segments_boundaries = self.split_audio_on_silence(audio)
        segments = []
        
        for start, end in segments_boundaries:
            segment = audio[start:end]
            if len(segment) > int(0.1 * self.sample_rate):  # Minimum 100ms
                segments.append(segment)
        
        return segments
    
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
    
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio quality and provide recommendations"""
        metrics = self.calculate_audio_metrics(audio)
        
        analysis = {
            "overall_quality": "good",
            "issues": [],
            "recommendations": [],
            "metrics": metrics
        }
        
        try:
            # Check for common issues
            if metrics.get('rms_energy', 0) < 0.001:
                analysis["issues"].append("Very low volume")
                analysis["recommendations"].append("Increase microphone gain")
                analysis["overall_quality"] = "poor"
            
            if metrics.get('peak_amplitude', 0) > 0.95:
                analysis["issues"].append("Audio clipping detected")
                analysis["recommendations"].append("Reduce input volume")
                analysis["overall_quality"] = "poor"
            
            if metrics.get('voice_activity_ratio', 0) < 0.1:
                analysis["issues"].append("Very little speech detected")
                analysis["recommendations"].append("Check microphone placement")
            
            snr = metrics.get('estimated_snr_db')
            if snr is not None and snr < 10:
                analysis["issues"].append("High background noise")
                analysis["recommendations"].append("Use noise cancellation or move to quieter environment")
                if analysis["overall_quality"] == "good":
                    analysis["overall_quality"] = "fair"
            
            # Check spectral characteristics
            spectral_centroid = metrics.get('spectral_centroid_mean', 0)
            if spectral_centroid < 500 or spectral_centroid > 8000:
                analysis["issues"].append("Unusual spectral characteristics")
                analysis["recommendations"].append("Check audio equipment")
            
        except Exception as e:
            logger.error(f"Audio quality analysis error: {e}")
            analysis["issues"].append(f"Analysis error: {e}")
        
        return analysis