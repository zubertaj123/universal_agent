"""
Audio processing utilities
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List, Dict, Any
import librosa
import io
# from pydub import AudioSegment
import ffmpeg
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

    def decode_and_resample_browser_audio(audio_bytes: bytes):
        try:
            out, _ = (
                ffmpeg
                .input('pipe:0', format='webm')  # Replace with correct format
                .output('pipe:1', format='wav', ac=1, ar='16000')
                .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
            )
            return out
        except ffmpeg.Error as e:
            print("FFmpeg Error:", e.stderr.decode())
            raise

    def frame_audio(self, audio: np.ndarray, frame_size: int = 512) -> list:
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
    
    def calculate_audio_metrics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive audio quality metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['duration'] = len(audio) / self.sample_rate
            metrics['sample_rate'] = self.sample_rate
            metrics['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            metrics['peak_amplitude'] = float(np.max(np.abs(audio)))
            
            # Dynamic range
            metrics['dynamic_range_db'] = 20 * np.log10(metrics['peak_amplitude'] / (metrics['rms_energy'] + 1e-10))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            metrics['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Spectral features
            stft = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
            metrics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # MFCCs (perceptual features)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            metrics['mfcc_mean'] = mfccs.mean(axis=1).tolist()
            metrics['mfcc_std'] = mfccs.std(axis=1).tolist()
            
            # Harmonic-to-noise ratio estimation
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.mean(harmonic**2)
            percussive_energy = np.mean(percussive**2)
            metrics['harmonic_ratio'] = float(harmonic_energy / (harmonic_energy + percussive_energy + 1e-10))
            
            # Estimate SNR (rough approximation)
            speech_segments = self.extract_speech_segments(audio)
            if speech_segments:
                speech_power = np.mean([np.mean(seg**2) for seg in speech_segments])
                silence_segments = self._extract_silence_segments(audio)
                if silence_segments:
                    noise_power = np.mean([np.mean(seg**2) for seg in silence_segments])
                    metrics['estimated_snr_db'] = 10 * np.log10(speech_power / (noise_power + 1e-10))
                else:
                    metrics['estimated_snr_db'] = None
            else:
                metrics['estimated_snr_db'] = None
            
            # Voice activity statistics
            vad_results = []
            frame_size = int(0.025 * self.sample_rate)  # 25ms
            hop_size = int(0.010 * self.sample_rate)    # 10ms
            
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                vad_results.append(self.detect_voice_activity(frame))
            
            metrics['voice_activity_ratio'] = sum(vad_results) / len(vad_results) if vad_results else 0
            
        except Exception as e:
            logger.error(f"Audio metrics calculation error: {e}")
            # Return basic metrics on error
            metrics = {
                'duration': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'rms_energy': float(np.sqrt(np.mean(audio**2))),
                'peak_amplitude': float(np.max(np.abs(audio))),
                'error': str(e)
            }
        
        return metrics
    
    def _extract_silence_segments(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract silence segments for noise estimation"""
        segments_boundaries = self.split_audio_on_silence(audio)
        silence_segments = []
        
        last_end = 0
        for start, end in segments_boundaries:
            if start > last_end:
                # There's silence between last segment and current
                silence_segment = audio[last_end:start]
                if len(silence_segment) > int(0.1 * self.sample_rate):  # Min 100ms
                    silence_segments.append(silence_segment)
            last_end = end
        
        # Check for silence at the end
        if last_end < len(audio):
            silence_segment = audio[last_end:]
            if len(silence_segment) > int(0.1 * self.sample_rate):
                silence_segments.append(silence_segment)
        
        return silence_segments
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        try:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio
    
    def apply_noise_gate(self, audio: np.ndarray, threshold: float = None, 
                        attack_time: float = 0.001, release_time: float = 0.1) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        if threshold is None:
            threshold = self.vad_threshold
        
        try:
            # Calculate envelope
            frame_size = int(0.010 * self.sample_rate)  # 10ms frames
            envelope = []
            
            for i in range(0, len(audio), frame_size):
                frame = audio[i:i + frame_size]
                envelope.append(np.sqrt(np.mean(frame**2)))
            
            # Smooth envelope
            envelope = np.array(envelope)
            
            # Apply gate
            gate_open = envelope > threshold
            
            # Smooth gate transitions
            attack_frames = int(attack_time * self.sample_rate / frame_size)
            release_frames = int(release_time * self.sample_rate / frame_size)
            
            smoothed_gate = np.copy(gate_open).astype(float)
            
            # Apply attack/release
            for i in range(1, len(smoothed_gate)):
                if gate_open[i] and not gate_open[i-1]:
                    # Gate opening - apply attack
                    for j in range(min(attack_frames, len(smoothed_gate) - i)):
                        smoothed_gate[i + j] = min(1.0, smoothed_gate[i + j] + j / attack_frames)
                elif not gate_open[i] and gate_open[i-1]:
                    # Gate closing - apply release
                    for j in range(min(release_frames, len(smoothed_gate) - i)):
                        smoothed_gate[i + j] = max(0.0, smoothed_gate[i + j] - j / release_frames)
            
            # Apply gate to audio
            gated_audio = np.copy(audio)
            for i, gate_value in enumerate(smoothed_gate):
                start_idx = i * frame_size
                end_idx = min((i + 1) * frame_size, len(audio))
                gated_audio[start_idx:end_idx] *= gate_value
            
            return gated_audio
            
        except Exception as e:
            logger.error(f"Noise gate error: {e}")
            return audio
    
    def enhance_speech(self, audio: np.ndarray) -> np.ndarray:
        """Apply speech enhancement techniques"""
        try:
            # Apply noise gate
            enhanced = self.apply_noise_gate(audio)
            
            # Apply spectral subtraction if noise profile available
            if self.noise_profile is not None:
                enhanced = self._reduce_noise(enhanced)
            
            # Apply mild compression
            enhanced = self._apply_compression(enhanced)
            
            # High-pass filter to remove low-frequency noise
            enhanced = self._apply_highpass_filter(enhanced, cutoff=80)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Speech enhancement error: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, threshold: float = 0.5, 
                          ratio: float = 3.0) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple compression
            compressed = np.copy(audio)
            
            # Find samples above threshold
            above_threshold = np.abs(compressed) > threshold
            
            # Apply compression to samples above threshold
            compressed[above_threshold] = (
                np.sign(compressed[above_threshold]) * 
                (threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio)
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return audio
    
    def _apply_highpass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter"""
        try:
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design high-pass filter
            b, a = signal.butter(4, normalized_cutoff, btype='high')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            logger.error(f"High-pass filter error: {e}")
            return audio
    
    def calibrate_noise_profile(self, noise_audio: np.ndarray):
        """Calibrate noise profile from noise-only audio"""
        try:
            # Compute noise spectrum
            stft = librosa.stft(noise_audio, n_fft=self.frame_size, hop_length=self.hop_length)
            self.noise_profile = np.mean(np.abs(stft), axis=1)
            logger.info("Noise profile calibrated")
            
        except Exception as e:
            logger.error(f"Noise profile calibration error: {e}")
    
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