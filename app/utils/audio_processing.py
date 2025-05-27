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
        
    def apply_noise