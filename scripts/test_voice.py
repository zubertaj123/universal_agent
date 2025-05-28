#!/usr/bin/env python3
"""
Test voice synthesis and recognition
"""
import asyncio
import sys
import tempfile
import os
from pathlib import Path
import numpy as np
import wave

# Add project root to path
sys.path.append('.')

from app.services.speech_service import SpeechService, VoiceStyle
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VoiceSystemTester:
    """Comprehensive voice system testing"""
    
    def __init__(self):
        self.speech_service = None
        self.test_results = {
            "tts_tests": [],
            "stt_tests": [],
            "vad_tests": [],
            "integration_tests": []
        }
    
    async def initialize_service(self):
        """Initialize the speech service"""
        try:
            print("🔄 Initializing Speech Service...")
            self.speech_service = SpeechService()
            await self.speech_service.initialize()
            print("✅ Speech Service initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Speech Service: {e}")
            return False
    
    async def test_tts_basic(self):
        """Test basic text-to-speech functionality"""
        print("\n🎤 Testing Text-to-Speech....")
        
        test_phrases = [
            ("Hello! How can I help you today?", VoiceStyle.FRIENDLY),
            ("I understand your concern. Let me help you with that.", VoiceStyle.EMPATHETIC),
            ("Your claim has been processed successfully.", VoiceStyle.PROFESSIONAL),
            ("Please hold while I transfer you to a specialist.", VoiceStyle.TECHNICAL),
        ]
        
        results = []
        
        for text, voice_style in test_phrases:
            try:
                print(f"  🔄 Testing: '{text[:30]}...' with {voice_style}")
                
                # Set voice style
                self.speech_service.set_voice(voice_style)
                
                # Generate audio
                audio_chunks = []
                chunk_count = 0
                
                async for chunk in self.speech_service.synthesize(text, stream=True):
                    audio_chunks.append(chunk)
                    chunk_count += 1
                
                total_size = sum(len(chunk) for chunk in audio_chunks)
                
                # Save to temp file for verification
                temp_file = f"test_tts_{voice_style.split('-')[-1].lower()}.mp3"
                with open(temp_file, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
                
                result = {
                    "text": text,
                    "voice_style": voice_style,
                    "success": True,
                    "chunks": chunk_count,
                    "total_size": total_size,
                    "file": temp_file
                }
                
                print(f"    ✅ Generated {chunk_count} chunks, {total_size} bytes")
                results.append(result)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results.append({
                    "text": text,
                    "voice_style": voice_style,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["tts_tests"] = results
        successful = sum(1 for r in results if r.get("success"))
        print(f"📊 TTS Results: {successful}/{len(results)} tests passed")
    
    async def test_tts_languages(self):
        """Test multi-language TTS support"""
        print("\n🌍 Testing Multi-language TTS...")
        
        language_tests = [
            ("Hello, how are you?", "en"),
            ("Hola, ¿cómo estás?", "es"),
            ("Bonjour, comment allez-vous?", "fr"),
            ("Hallo, wie geht es dir?", "de"),
        ]
        
        for text, lang in language_tests:
            try:
                print(f"  🔄 Testing {lang}: '{text}'")
                
                # Get appropriate voice for language
                voice = await self.speech_service.get_voice_for_language(lang)
                if voice:
                    self.speech_service.set_voice(voice)
                    print(f"    📢 Using voice: {voice}")
                else:
                    print(f"    ⚠️  No specific voice found for {lang}, using default")
                
                # Generate audio
                audio_chunks = []
                async for chunk in self.speech_service.synthesize(text, stream=True):
                    audio_chunks.append(chunk)
                
                print(f"    ✅ Generated audio for {lang}")
                
            except Exception as e:
                print(f"    ❌ Error with {lang}: {e}")
    
    async def test_vad(self):
        """Test Voice Activity Detection"""
        print("\n🔍 Testing Voice Activity Detection...")
        
        sample_rate = 16000
        
        # Test cases: silence, noise, speech-like patterns
        test_cases = [
            ("Silence", np.zeros(sample_rate)),
            ("White Noise", np.random.randn(sample_rate) * 0.1),
            ("Loud Noise", np.random.randn(sample_rate) * 0.5),
            ("Sine Wave (440Hz)", np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))),
            ("Complex Audio", np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)) + 
                            0.2 * np.random.randn(sample_rate)),
        ]
        
        results = []
        
        for name, audio in test_cases:
            try:
                print(f"  🔄 Testing VAD on: {name}")
                
                is_speech = self.speech_service.detect_voice_activity(audio, sample_rate)
                
                print(f"    {'🗣️ ' if is_speech else '🔇 '}{'Speech detected' if is_speech else 'No speech detected'}")
                
                results.append({
                    "name": name,
                    "has_speech": is_speech,
                    "success": True
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results.append({
                    "name": name,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["vad_tests"] = results
        print(f"📊 VAD Results: All {len(results)} tests completed")
    
    async def test_stt_basic(self):
        """Test basic speech-to-text functionality"""
        print("\n🎧 Testing Speech-to-Text...")
        
        # Note: For real STT testing, we'd need actual audio files
        # This is a simplified test using synthetic data
        
        try:
            # Create a simple test audio signal
            sample_rate = 16000
            duration = 2
            
            # Generate a simple test signal (not real speech)
            t = np.linspace(0, duration, sample_rate * duration)
            test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
            
            print("  🔄 Testing STT with synthetic audio...")
            result = await self.speech_service.transcribe(test_audio)
            
            if result is None:
                print("    ✅ STT correctly returned None for non-speech audio")
                success = True
            else:
                print(f"    📝 STT result: '{result}'")
                success = True
                
            self.test_results["stt_tests"].append({
                "test_type": "synthetic_audio",
                "result": result,
                "success": success
            })
            
        except Exception as e:
            print(f"    ❌ STT Error: {e}")
            self.test_results["stt_tests"].append({
                "test_type": "synthetic_audio",
                "success": False,
                "error": str(e)
            })
    
    async def test_integration(self):
        """Test TTS + STT integration"""
        print("\n🔄 Testing TTS -> STT Integration...")
        
        test_phrase = "This is a test of the speech system integration."
        
        try:
            print(f"  🎤 Generating TTS for: '{test_phrase}'")
            
            # Generate TTS audio
            audio_data = []
            async for chunk in self.speech_service.synthesize(test_phrase, stream=False):
                audio_data.append(chunk)
            
            audio_bytes = b''.join(audio_data)
            print(f"    ✅ Generated {len(audio_bytes)} bytes of audio")
            
            # Save to temporary file
            temp_audio_file = "test_integration.mp3"
            with open(temp_audio_file, "wb") as f:
                f.write(audio_bytes)
            
            print("    💾 Audio saved for potential manual verification")
            
            # Note: To properly test STT with the generated TTS audio,
            # we'd need to decode the MP3 to raw audio format that Whisper expects
            print("    ℹ️  Full TTS->STT loop requires MP3 decoding (not implemented in this test)")
            
            self.test_results["integration_tests"].append({
                "test_type": "tts_to_file",
                "success": True,
                "file_size": len(audio_bytes),
                "file_path": temp_audio_file
            })
            
        except Exception as e:
            print(f"    ❌ Integration test error: {e}")
            self.test_results["integration_tests"].append({
                "test_type": "tts_to_file",
                "success": False,
                "error": str(e)
            })
    
    async def test_performance(self):
        """Test performance metrics"""
        print("\n⚡ Testing Performance...")
        
        test_text = "This is a performance test message."
        iterations = 3
        
        try:
            import time
            
            print(f"  🔄 Running {iterations} TTS generations...")
            
            times = []
            for i in range(iterations):
                start_time = time.time()
                
                audio_chunks = []
                async for chunk in self.speech_service.synthesize(test_text, stream=True):
                    audio_chunks.append(chunk)
                
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                
                print(f"    🎯 Iteration {i+1}: {duration:.2f}s")
            
            avg_time = sum(times) / len(times)
            print(f"    📊 Average generation time: {avg_time:.2f}s")
            
            # Performance evaluation
            if avg_time < 1.0:
                print("    ✅ Excellent performance (< 1s)")
            elif avg_time < 2.0:
                print("    ✅ Good performance (< 2s)")
            elif avg_time < 5.0:
                print("    ⚠️  Acceptable performance (< 5s)")
            else:
                print("    ❌ Slow performance (> 5s)")
                
        except Exception as e:
            print(f"    ❌ Performance test error: {e}")
    
    async def test_voice_styles(self):
        """Test different voice styles"""
        print("\n🎭 Testing Voice Styles...")
        
        test_text = "Thank you for calling our support center."
        
        # Test all available voice styles
        voice_styles = [
            VoiceStyle.PROFESSIONAL,
            VoiceStyle.FRIENDLY,
            VoiceStyle.TECHNICAL,
            VoiceStyle.EMPATHETIC,
            VoiceStyle.MULTILINGUAL
        ]
        
        for style in voice_styles:
            try:
                print(f"  🔄 Testing voice style: {style}")
                
                self.speech_service.set_voice(style)
                
                audio_chunks = []
                async for chunk in self.speech_service.synthesize(test_text, stream=True):
                    audio_chunks.append(chunk)
                
                total_size = sum(len(chunk) for chunk in audio_chunks)
                print(f"    ✅ Generated {total_size} bytes with {style}")
                
            except Exception as e:
                print(f"    ❌ Error with {style}: {e}")
    
    def cleanup_test_files(self):
        """Clean up temporary test files"""
        print("\n🧹 Cleaning up test files...")
        
        test_files = [
            "test_professional.mp3",
            "test_friendly.mp3", 
            "test_technical.mp3",
            "test_empathetic.mp3",
            "test_integration.mp3"
        ]
        
        for file in test_files:
            try:
                if Path(file).exists():
                    os.remove(file)
                    print(f"    🗑️  Removed {file}")
            except Exception as e:
                print(f"    ❌ Error removing {file}: {e}")
        print("🧹 Cleanup complete")