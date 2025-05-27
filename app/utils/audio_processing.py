"""
Audio processing utilities
"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional
import librosa

class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_muted = False
        
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for speech recognition"""
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Apply pre-emphasis
        audio = signal.lfilter([1, -0.97], 1, audio)
        
        return audio
        
    def apply_noise_reduction(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply noise reduction to audio"""
        # Simple spectral subtraction
        # In production, use more sophisticated methods
        return audio
        
    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Detect if audio is silence"""
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold
        
    def split_audio_on_silence(
        self,
        audio: np.ndarray,
        min_silence_duration: float = 0.5,
        silence_threshold: float = 0.01
    ) -> list:
        """Split audio on silence"""
        # Calculate frame length
        frame_length = int(min_silence_duration * self.sample_rate)
        
        # Find non-silent intervals
        non_silent_intervals = []
        start = None
        
        for i in range(0, len(audio), frame_length):
            chunk = audio[i:i + frame_length]
            is_silent = self.detect_silence(chunk, silence_threshold)
            
            if not is_silent and start is None:
                start = i
            elif is_silent and start is not None:
                non_silent_intervals.append((start, i))
                start = None
                
        if start is not None:
            non_silent_intervals.append((start, len(audio)))
            
        # Extract chunks
        chunks = []
        for start, end in non_silent_intervals:
            chunks.append(audio[start:end])
            
        return chunks
        
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        
    def mute(self):
        """Mute audio processing"""
        self.is_muted = True
        
    def unmute(self):
        """Unmute audio processing"""
        self.is_muted = False