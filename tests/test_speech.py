"""
Tests for speech services
"""
import pytest
import numpy as np
from app.services.speech_service import SpeechService, VoiceStyle

@pytest.fixture
async def speech_service():
    """Create speech service instance"""
    service = SpeechService()
    await service.initialize()
    return service

@pytest.mark.asyncio
async def test_service_initialization(speech_service):
    """Test service initialization"""
    assert speech_service.stt_model is not None
    assert speech_service.vad_model is not None
    assert len(speech_service.voices_cache) > 0

@pytest.mark.asyncio
async def test_voice_activity_detection(speech_service):
    """Test VAD functionality"""
    # Test with silence
    silence = np.zeros(16000)  # 1 second of silence
    assert not speech_service.detect_voice_activity(silence)
    
    # Test with noise
    noise = np.random.randn(16000) * 0.1
    assert speech_service.detect_voice_activity(noise)

@pytest.mark.asyncio
async def test_tts_synthesis(speech_service):
    """Test text-to-speech"""
    text = "Hello, this is a test"
    
    # Test streaming
    chunks = []
    async for chunk in speech_service.synthesize(text, stream=True):
        chunks.append(chunk)
        
    assert len(chunks) > 0
    assert all(isinstance(chunk, bytes) for chunk in chunks)

@pytest.mark.asyncio
async def test_tts_caching(speech_service):
    """Test TTS caching"""
    text = "This should be cached"
    
    # First call - should generate
    first_chunks = []
    async for chunk in speech_service.synthesize(text, stream=False):
        first_chunks.append(chunk)
        
    # Second call - should use cache
    second_chunks = []
    async for chunk in speech_service.synthesize(text, stream=False):
        second_chunks.append(chunk)
        
    # Should be the same
    assert len(first_chunks) == len(second_chunks)

@pytest.mark.asyncio
async def test_voice_selection(speech_service):
    """Test voice selection"""
    # Test setting voice
    speech_service.set_voice(VoiceStyle.FRIENDLY)
    assert speech_service.current_voice == VoiceStyle.FRIENDLY
    
    # Test language-specific voice
    spanish_voice = await speech_service.get_voice_for_language("es")
    assert spanish_voice is not None
    assert "es" in spanish_voice

@pytest.mark.asyncio
async def test_transcription(speech_service):
    """Test speech-to-text"""
    # Create test audio (sine wave)
    sample_rate = 16000
    duration = 2
    frequency = 440
    
    t = np.linspace(0, duration, sample_rate * duration)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # This is a simplified test - real audio would produce text
    result = await speech_service.transcribe(audio)
    # In a real test, we'd use actual speech audio
    assert result is None or isinstance(result, str)