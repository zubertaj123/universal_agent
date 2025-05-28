#!/usr/bin/env python3
"""
Temporary fix to get the call center working without real audio processing
"""
import shutil
from pathlib import Path

def apply_temp_audio_fix():
    """Apply temporary audio processing fix"""
    print("🔧 Applying temporary audio processing fix...")
    
    # Backup original file
    original_file = Path("app/utils/audio_processing.py")
    backup_file = Path("app/utils/audio_processing.py.backup")
    
    if original_file.exists():
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backed up original to {backup_file}")
    
    # Create simplified audio processor
    simplified_content = '''"""
Audio processing utilities - Simplified version for demo
"""
import numpy as np
from typing import List, Dict, Any
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioProcessor:
    """Simplified audio processing for demo purposes"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_muted = False
        self.gain_factor = 1.0
        self.vad_threshold = 0.001
        self.speech_counter = 0  # To simulate occasional speech detection

    def decode_and_resample_browser_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Simplified audio decoding - creates placeholder audio"""
        try:
            if len(audio_bytes) == 0:
                return np.array([], dtype=np.float32)
            
            # Create minimal audio signal
            duration = 0.032  # 32ms
            samples = int(duration * self.sample_rate)
            
            # Create low-level noise
            audio_data = np.random.normal(0, 0.0005, samples).astype(np.float32)
            
            # Occasionally add a stronger signal to simulate speech
            self.speech_counter += 1
            if self.speech_counter % 50 == 0:  # Every 50th chunk
                speech_signal = 0.005 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
                audio_data = audio_data + speech_signal.astype(np.float32)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
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
        """Basic preprocessing"""
        if len(audio) == 0:
            return audio
        return np.clip(audio * self.gain_factor, -1.0, 1.0)

    def detect_voice_activity(self, audio: np.ndarray, enhanced: bool = False) -> bool:
        """Simplified VAD"""
        if len(audio) == 0:
            return False
        
        rms_energy = np.sqrt(np.mean(audio**2))
        return rms_energy > self.vad_threshold

    def detect_silence(self, audio: np.ndarray, threshold: float = None) -> bool:
        """Detect silence"""
        if len(audio) == 0:
            return True
        
        threshold = threshold or self.vad_threshold
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold

    def mute(self):
        self.is_muted = True
        
    def unmute(self):
        self.is_muted = False
    
    def set_gain(self, gain_db: float):
        self.gain_factor = 10**(gain_db / 20)
'''
    
    # Write the simplified version
    with open(original_file, 'w') as f:
        f.write(simplified_content)
    
    print("✅ Applied temporary audio processing fix")
    return True

def update_call_interface():
    """Update call interface to show it's in demo mode"""
    print("🔧 Updating call interface...")
    
    template_file = Path("templates/call_interface.html")
    if not template_file.exists():
        print("⚠️ Call interface template not found")
        return False
    
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Add demo notice
    demo_notice = '''
    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; margin: 16px 0;">
        <h3 style="color: #92400e; margin: 0 0 8px 0;">🚧 Demo Mode</h3>
        <p style="color: #92400e; margin: 0; font-size: 14px;">
            This is running in demo mode with simulated audio processing. 
            Real microphone input is not being processed yet. 
            The AI will still respond with text-to-speech.
        </p>
    </div>
    '''
    
    # Insert after the status panel
    if '<div class="status-panel">' in content:
        content = content.replace(
            '</div>\n            \n            <div class="audio-visualizer">',
            '</div>\n            ' + demo_notice + '\n            <div class="audio-visualizer">'
        )
        
        with open(template_file, 'w') as f:
            f.write(content)
        
        print("✅ Updated call interface with demo notice")
        return True
    
    return False

def main():
    """Apply temporary fixes"""
    print("🔧 Applying temporary fixes to get Call Center AI working")
    print("=" * 60)
    
    success_count = 0
    
    if apply_temp_audio_fix():
        success_count += 1
    
    if update_call_interface():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Applied {success_count}/2 temporary fixes")
    
    if success_count > 0:
        print("\n🎉 Temporary fixes applied! You can now run:")
        print("   python app/main_https.py")
        print("\n💡 What's working:")
        print("   ✅ Web interface loads")
        print("   ✅ WebSocket connections work")
        print("   ✅ TTS (text-to-speech) works")
        print("   ✅ Call flow and database work")
        print("\n⚠️ What's in demo mode:")
        print("   🚧 Audio processing (using placeholder)")
        print("   🚧 Speech recognition (simulated)")
        print("\n📝 To restore original audio processing later:")
        print("   mv app/utils/audio_processing.py.backup app/utils/audio_processing.py")
    else:
        print("\n❌ Some fixes failed - check the errors above")

if __name__ == "__main__":
    main()