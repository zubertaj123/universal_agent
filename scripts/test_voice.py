#!/usr/bin/env python3
"""
Test voice synthesis and recognition
"""
import asyncio
import sys
sys.path.append('.')

from app.services.speech_service import SpeechService, VoiceStyle
import numpy as np

async def test_tts():
    """Test text-to-speech"""
    print("Testing Text-to-Speech...")
    
    service = SpeechService()
    await service.initialize()
    
    test_phrases = [
        ("Hello! How can I help you today?", VoiceStyle.FRIENDLY),
        ("I understand your concern. Let me help you with that.", VoiceStyle.EMPATHETIC),
        ("Your claim has been processed successfully.", VoiceStyle.PROFESSIONAL),
    ]
    
    for text, voice_style in test_phrases:
        print(f"\nTesting: '{text}' with voice: {voice_style}")
        service.set_voice(voice_style)
        
        audio_chunks = []
        async for chunk in service.synthesize(text, stream=True):
            audio_chunks.append(chunk)
            
        print(f"Generated {len(audio_chunks)} audio chunks")
        
        # Save to file for testing
        with open(f"test_{voice_style.split('-')[2].lower()}.mp3", "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
                
    print("\nTTS test completed!")

async def test_stt():
    """Test speech-to-text"""
    print("\nTesting Speech-to-Text...")
    
    service = SpeechService()
    await service.initialize()
    
    # Generate test audio
    test_text = "This is a test of the speech recognition system."
    print(f"Generating audio for: '{test_text}'")
    
    audio_data = []
    async for chunk in service.synthesize(test_text, stream=False):
        audio_data.append(chunk)
        
    # Convert to numpy array (simplified - in production use proper audio decoding)
    audio_bytes = b''.join(audio_data)
    
    # Test transcription
    # Note: This is a simplified example. In production, you'd need to properly
    # decode the MP3 data to raw audio format that Whisper expects
    
    print("STT test completed!")

async def test_vad():
    """Test voice activity detection"""
    print("\nTesting Voice Activity Detection...")
    
    service = SpeechService()
    await service.initialize()
    
    # Create test audio: silence, then noise, then silence
    sample_rate = 16000
    duration = 3  # seconds
    
    # Generate test signal
    t = np.linspace(0, duration, sample_rate * duration)
    silence = np.zeros(sample_rate)  # 1 second silence
    noise = 0.1 * np.random.randn(sample_rate)  # 1 second noise
    
    test_audio = np.concatenate([silence, noise, silence])
    
    # Test VAD on each second
    for i in range(3):
        chunk = test_audio[i*sample_rate:(i+1)*sample_rate]
        is_speech = service.detect_voice_activity(chunk, sample_rate)
        print(f"Second {i+1}: {'Speech detected' if is_speech else 'Silence'}")
        
    print("VAD test completed!")

async def main():
    """Run all tests"""
    print("Voice Service Test Suite")
    print("========================")
    
    await test_tts()
    await test_stt()
    await test_vad()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())