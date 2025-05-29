/**
 * FIXED AUDIO STREAMING - Replaces SingleAudioPlayer class
 * Solves: Robotic chunky voice by implementing proper audio buffering
 */

class StreamingAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.audioBuffer = [];
        this.isPlaying = false;
        this.volume = 0.8;
        this.activeStreamId = null;
        this.sessionId = null;
        
        // CRITICAL: Stream buffering for natural playback
        this.pendingChunks = [];
        this.currentPlayback = null;
        this.nextAudioReady = false;
        
        console.log('🎵 StreamingAudioPlayer initialized for natural voice');
    }
    
    async initializeAudioContext() {
        if (this.audioContext) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 44100,  // High quality for TTS
                latencyHint: 'playback'  // Optimized for playback
            });
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            console.log('🎵 Audio context ready for streaming:', this.audioContext.sampleRate + 'Hz');
        } catch (error) {
            console.error('❌ Audio context setup failed:', error);
        }
    }
    
    async startAudioSequence(streamId, text) {
        console.log('🎬 Starting SMOOTH audio sequence:', text?.substring(0, 50) + '...');
        
        // CRITICAL: Don't stop current audio abruptly - let it finish naturally
        this.activeStreamId = streamId;
        this.pendingChunks = [];
        this.nextAudioReady = false;
        
        // Initialize audio context if needed
        await this.initializeAudioContext();
    }
    
    async addAudioChunk(audioData, metadata = {}) {
        try {
            // Only accept chunks from the current stream
            if (metadata.streamId && metadata.streamId !== this.activeStreamId) {
                console.log('🚫 Ignoring chunk from old stream:', metadata.streamId);
                return false;
            }
            
            console.log('📦 Buffering audio chunk:', metadata.chunkNumber || 'unknown');
            
            // Convert bytes to AudioBuffer for smooth playback
            const audioBuffer = await this.createAudioBuffer(audioData);
            if (!audioBuffer) {
                console.warn('⚠️ Failed to create audio buffer');
                return false;
            }
            
            // Add to buffer
            this.pendingChunks.push({
                buffer: audioBuffer,
                chunkNumber: metadata.chunkNumber,
                metadata: metadata
            });
            
            // Start playback if not already playing
            if (!this.isPlaying) {
                await this.startContinuousPlayback();
            }
            
            return true;
            
        } catch (error) {
            console.error('❌ Audio chunk processing failed:', error);
            return false;
        }
    }
    
    async createAudioBuffer(audioData) {
        try {
            if (!this.audioContext) {
                await this.initializeAudioContext();
            }
            
            // Decode audio data to AudioBuffer
            const audioBuffer = await this.audioContext.decodeAudioData(audioData.buffer);
            return audioBuffer;
            
        } catch (error) {
            console.error('❌ Audio buffer creation failed:', error);
            
            // Fallback: try with different approach
            try {
                const blob = new Blob([audioData], { type: 'audio/mpeg' });
                const arrayBuffer = await blob.arrayBuffer();
                return await this.audioContext.decodeAudioData(arrayBuffer);
            } catch (fallbackError) {
                console.error('❌ Fallback audio decoding failed:', fallbackError);
                return null;
            }
        }
    }
    
    async startContinuousPlayback() {
        if (this.isPlaying) return;
        
        this.isPlaying = true;
        console.log('🔊 Starting CONTINUOUS playback...');
        
        const playNextChunk = async () => {
            // Check if we have chunks to play
            if (this.pendingChunks.length === 0) {
                // Wait a bit for more chunks
                setTimeout(() => {
                    if (this.pendingChunks.length > 0 && this.isPlaying) {
                        playNextChunk();
                    } else if (this.pendingChunks.length === 0) {
                        // No more chunks, end playback
                        this.isPlaying = false;
                        console.log('🔊 Continuous playback completed');
                    }
                }, 50); // Small delay to allow buffering
                return;
            }
            
            // Get next chunk
            const chunk = this.pendingChunks.shift();
            
            try {
                // Create source node
                const source = this.audioContext.createBufferSource();
                const gainNode = this.audioContext.createGain();
                
                source.buffer = chunk.buffer;
                gainNode.gain.value = this.volume;
                
                // Connect audio graph
                source.connect(gainNode);
                gainNode.connect(this.audioContext.destination);
                
                // Set up event handlers
                source.onended = () => {
                    // Immediately play next chunk for seamless playback
                    if (this.isPlaying) {
                        playNextChunk();
                    }
                };
                
                // Start playback
                source.start(0);
                this.currentPlayback = source;
                
                console.log('🔊 Playing chunk', chunk.chunkNumber, 'seamlessly');
                
            } catch (error) {
                console.error('❌ Chunk playback failed:', error);
                // Continue with next chunk
                if (this.isPlaying) {
                    playNextChunk();
                }
            }
        };
        
        // Start the playback chain
        playNextChunk();
    }
    
    completeAudioSequence(stats) {
        console.log('🎬 Audio sequence completed naturally:', stats);
        
        // Let current chunks finish playing
        setTimeout(() => {
            if (this.pendingChunks.length === 0) {
                this.isPlaying = false;
                this.activeStreamId = null;
                console.log('✅ Audio sequence fully completed');
            }
        }, 1000);
    }
    
    stopCurrentAudio() {
        console.log('🛑 Stopping audio stream');
        
        this.isPlaying = false;
        this.pendingChunks = [];
        this.activeStreamId = null;
        
        if (this.currentPlayback) {
            try {
                this.currentPlayback.stop();
                this.currentPlayback = null;
            } catch (error) {
                // Audio might already be stopped
            }
        }
    }
    
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        console.log('🔊 Volume set to:', this.volume);
    }
    
    getStatus() {
        return {
            isPlaying: this.isPlaying,
            volume: this.volume,
            activeStreamId: this.activeStreamId,
            sessionId: this.sessionId,
            pendingChunks: this.pendingChunks.length,
            bufferReady: this.pendingChunks.length > 0
        };
    }
}

/**
 * UPDATED CallHandler class with fixed audio streaming
 */
class CallHandler {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.isConnected = false;
        this.isMuted = false;
        this.isPaused = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.startTime = null;
        this.durationInterval = null;
        
        // FIXED: Streaming audio player for natural voice
        this.audioPlayer = new StreamingAudioPlayer();
        this.currentAudioSequence = null;
        
        // Audio processing parameters
        this.targetSampleRate = 16000;
        this.audioProcessor = null;
        this.audioSource = null;
        
        // FIXED: Audio buffering for proper VAD chunk sizes
        this.audioBuffer = [];
        this.vad_chunk_size = 512;  // Match server-side VAD requirements
        
        // Production mode
        this.productionMode = true;
        this.debugMode = false;
        
        console.log('🎯 CallHandler initialized with STREAMING audio');
    }

    async initialize() {
        console.log('🚀 Initializing STREAMING Call Handler...');
        
        this.sessionId = this.generateSessionId();
        this.audioPlayer.sessionId = this.sessionId;
        
        try {
            console.log('🎵 Setting up audio context...');
            await this.initializeAudioContext();
            
            console.log('🎤 Requesting microphone access...');
            await this.requestMicrophoneAccess();
            
            console.log('🔌 Connecting to WebSocket...');
            await this.connectWebSocket();
            
            console.log('🎙️ Setting up audio processing...');
            await this.setupAudioProcessing();
            
            this.startDurationTimer();
            this.updateStatus('Connected - Ready for conversation');
            
            console.log('✅ STREAMING Call Handler ready');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            throw error;
        }
    }

    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.targetSampleRate,
                latencyHint: 'interactive'
            });
            
            console.log('🎵 Audio context ready:', {
                sampleRate: this.audioContext.sampleRate,
                state: this.audioContext.state
            });
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
        } catch (error) {
            console.warn('⚠️ Using default audio context');
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
        }
    }

    async requestMicrophoneAccess() {
        try {
            const constraints = {
                audio: {
                    sampleRate: this.targetSampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    latency: 0.01
                }
            };
            
            this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            const track = this.mediaStream.getAudioTracks()[0];
            const settings = track.getSettings();
            
            console.log('🎤 Microphone ready:', {
                sampleRate: settings.sampleRate,
                channelCount: settings.channelCount
            });
            
        } catch (error) {
            console.error('❌ Microphone error:', error);
            
            let errorMessage = 'Microphone access failed';
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Microphone access denied. Please allow microphone access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone found. Please connect a microphone and try again.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Microphone is being used by another application.';
            } else {
                errorMessage = `Microphone error: ${error.message}`;
            }
            
            throw new Error(errorMessage);
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsUrl = this.buildWebSocketUrl();
            console.log('🔌 Connecting to WebSocket:', wsUrl);
            
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';
            
            const connectionTimeout = setTimeout(() => {
                if (!this.isConnected) {
                    console.error('❌ WebSocket connection timeout');
                    if (this.ws) {
                        this.ws.close();
                    }
                    reject(new Error('Connection timeout - server may be unreachable'));
                }
            }, 15000);
            
            this.ws.onopen = () => {
                clearTimeout(connectionTimeout);
                this.isConnected = true;
                console.log('✅ WebSocket connected');
                
                this.sendMessage({
                    type: 'connection',
                    sessionId: this.sessionId,
                    timestamp: new Date().toISOString(),
                    clientInfo: {
                        userAgent: navigator.userAgent,
                        sampleRate: this.audioContext.sampleRate
                    }
                });
                
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('❌ Message parsing error:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                clearTimeout(connectionTimeout);
                console.error('❌ WebSocket error:', error);
                this.updateStatus('Connection error');
                reject(new Error('WebSocket connection failed'));
            };
            
            this.ws.onclose = (event) => {
                clearTimeout(connectionTimeout);
                this.isConnected = false;
                console.log('🔌 WebSocket closed');
                
                this.updateStatus('Disconnected');
                this.stopDurationTimer();
                this.audioPlayer.stopCurrentAudio();
            };
        });
    }

    buildWebSocketUrl() {
        const isSecure = window.location.protocol === 'https:';
        const wsProtocol = isSecure ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${wsProtocol}//${host}/ws/call/${this.sessionId}`;
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('❌ Send message error:', error);
                return false;
            }
        } else {
            console.warn('⚠️ WebSocket not ready');
            return false;
        }
    }

    async setupAudioProcessing() {
        try {
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            const bufferSize = 4096;
            const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            processor.onaudioprocess = (event) => {
                if (!this.isRecording || this.isPaused || this.isMuted) {
                    return;
                }
                
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                this.processAudioChunkFixed(inputData);
            };
            
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            this.audioSource = source;
            this.audioProcessor = processor;
            this.isRecording = true;
            
            this.setupAudioVisualization(source);
            console.log('✅ Audio processing ready');
            
        } catch (error) {
            console.error('❌ Audio processing setup failed:', error);
            throw error;
        }
    }

    processAudioChunkFixed(audioData) {
        try {
            this.audioBuffer.push(...audioData);
            
            while (this.audioBuffer.length >= this.vad_chunk_size) {
                const chunk = this.audioBuffer.splice(0, this.vad_chunk_size);
                
                if (this.hasSignificantAudio(chunk)) {
                    const pcmData = this.convertToPCM16(chunk);
                    
                    if (pcmData.length > 0) {
                        const hexString = Array.from(pcmData)
                            .map(b => b.toString(16).padStart(2, '0'))
                            .join('');
                        
                        this.sendMessage({
                            type: 'audio',
                            data: hexString,
                            format: 'pcm16',
                            sampleRate: this.audioContext.sampleRate,
                            chunkSize: this.vad_chunk_size,
                            timestamp: Date.now()
                        });
                    }
                }
            }
            
        } catch (error) {
            console.error('❌ Audio chunk processing error:', error);
            this.audioBuffer = [];
        }
    }

    convertToPCM16(float32Array) {
        try {
            const buffer = new ArrayBuffer(float32Array.length * 2);
            const view = new DataView(buffer);
            
            for (let i = 0; i < float32Array.length; i++) {
                let sample = Math.max(-1, Math.min(1, float32Array[i]));
                sample = sample * 32767;
                view.setInt16(i * 2, sample, true);
            }
            
            return new Uint8Array(buffer);
        } catch (error) {
            console.error('❌ PCM16 conversion error:', error);
            return new Uint8Array(0);
        }
    }

    hasSignificantAudio(audioData) {
        try {
            const rms = Math.sqrt(
                audioData.reduce((sum, sample) => sum + sample * sample, 0) / audioData.length
            );
            return rms > 0.001;
        } catch (error) {
            console.error('❌ Audio analysis error:', error);
            return false;
        }
    }

    setupAudioVisualization(source) {
        try {
            const canvas = document.getElementById('audio-canvas');
            if (!canvas) return;
            
            const canvasCtx = canvas.getContext('2d');
            const analyser = this.audioContext.createAnalyser();
            
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            const draw = () => {
                if (!this.isRecording) return;
                
                requestAnimationFrame(draw);
                analyser.getByteFrequencyData(dataArray);
                
                canvasCtx.fillStyle = 'rgb(249, 250, 251)';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    const hue = (i / bufferLength) * 120;
                    const saturation = 70;
                    const lightness = 30 + (dataArray[i] / 255) * 40;
                    
                    canvasCtx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                    
                    x += barWidth + 1;
                }
            };
            
            draw();
            console.log('✅ Audio visualization ready');
            
        } catch (error) {
            console.error('❌ Visualization setup error:', error);
        }
    }

    /**
     * FIXED message handling with STREAMING audio
     */
    async handleMessage(data) {
        try {
            const messageType = data.type;
            console.log('📨 Received:', messageType);
            
            switch (messageType) {
                case 'audio_start':
                    await this.handleAudioStart(data);
                    break;
                    
                case 'audio_chunk':
                    await this.handleAudioChunk(data);
                    break;
                    
                case 'audio_complete':
                    await this.handleAudioComplete(data);
                    break;
                    
                case 'transcript':
                    this.addTranscript(data.speaker, data.text);
                    break;
                    
                case 'status':
                    this.updateStatus(data.message);
                    break;
                    
                case 'connected':
                    this.updateStatus('Connected and ready');
                    console.log('✅ Server connection confirmed');
                    break;
                    
                case 'error':
                case 'audio_error':
                    console.error('❌ Server error:', data.message);
                    this.updateStatus('Error: ' + data.message);
                    this.audioPlayer.stopCurrentAudio();
                    break;
                    
                default:
                    if (this.debugMode) {
                        console.log('❓ Unknown message type:', messageType);
                    }
            }
            
        } catch (error) {
            console.error('❌ Message handling error:', error);
        }
    }
    
    async handleAudioStart(data) {
        console.log('🎬 Starting STREAMING audio sequence');
        
        this.currentAudioSequence = data.stream_id;
        this.updateStatus('Agent speaking...');
        
        // Initialize streaming audio
        await this.audioPlayer.startAudioSequence(data.stream_id, data.text);
    }
    
    async handleAudioChunk(data) {
        try {
            if (data.stream_id && data.stream_id !== this.currentAudioSequence) {
                return; // Ignore old chunks
            }
            
            const hexData = data.data;
            if (!hexData || typeof hexData !== 'string') {
                console.warn('⚠️ Invalid audio data');
                return;
            }
            
            try {
                const bytes = new Uint8Array(
                    hexData.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
                );
                
                // Add to streaming buffer for natural playback
                const success = await this.audioPlayer.addAudioChunk(bytes, {
                    chunkNumber: data.chunk_number,
                    streamId: data.stream_id,
                    sessionId: data.session_id
                });
                
                if (success) {
                    console.log('📦 Chunk buffered for streaming:', data.chunk_number);
                }
                
            } catch (hexError) {
                console.error('❌ Hex conversion error:', hexError);
            }
            
        } catch (error) {
            console.error('❌ Audio chunk handling error:', error);
        }
    }
    
    async handleAudioComplete(data) {
        console.log('🎬 Audio sequence completed:', data.chunks_sent || 'unknown', 'chunks');
        
        this.currentAudioSequence = null;
        this.updateStatus('Listening...');
        
        // Let audio finish naturally
        this.audioPlayer.completeAudioSequence({
            chunks: data.chunks_sent,
            bytes: data.total_bytes
        });
    }

    addTranscript(speaker, text) {
        try {
            const transcriptDiv = document.getElementById('transcript');
            if (!transcriptDiv) return;
            
            const entry = document.createElement('div');
            entry.className = `transcript-entry ${speaker}`;
            
            const speakerDiv = document.createElement('div');
            speakerDiv.className = 'speaker';
            speakerDiv.textContent = speaker === 'user' ? 'You' : 'Agent';
            
            const textDiv = document.createElement('div');
            textDiv.className = 'text';
            textDiv.textContent = text;
            
            const timeDiv = document.createElement('div');
            timeDiv.className = 'time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            
            entry.appendChild(speakerDiv);
            entry.appendChild(textDiv);
            entry.appendChild(timeDiv);
            
            transcriptDiv.appendChild(entry);
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
            
            // Limit entries
            const entries = transcriptDiv.querySelectorAll('.transcript-entry');
            if (entries.length > 100) {
                entries[0].remove();
            }
            
        } catch (error) {
            console.error('❌ Transcript error:', error);
        }
    }

    updateStatus(status) {
        try {
            const statusElement = document.getElementById('call-status');
            if (statusElement) {
                statusElement.textContent = status;
            }
            console.log('📊 Status:', status);
        } catch (error) {
            console.error('❌ Status update error:', error);
        }
    }

    startDurationTimer() {
        this.startTime = Date.now();
        
        this.durationInterval = setInterval(() => {
            try {
                const elapsed = Date.now() - this.startTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                
                const durationElement = document.getElementById('call-duration');
                if (durationElement) {
                    durationElement.textContent = 
                        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                }
            } catch (error) {
                console.error('❌ Duration timer error:', error);
            }
        }, 1000);
    }

    stopDurationTimer() {
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }
    }

    generateSessionId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    // Control methods
    toggleMute() {
        this.isMuted = !this.isMuted;
        const muteBtn = document.getElementById('mute-btn');
        if (muteBtn) {
            muteBtn.textContent = this.isMuted ? '🔇 Unmute' : '🎤 Mute';
            muteBtn.className = this.isMuted ? 'btn btn-danger' : 'btn';
        }
        
        this.sendMessage({
            type: 'control',
            action: this.isMuted ? 'mute' : 'unmute'
        });
        
        console.log('🔇 Mute toggled:', this.isMuted);
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        const pauseBtn = document.getElementById('pause-btn');
        if (pauseBtn) {
            pauseBtn.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
            pauseBtn.className = this.isPaused ? 'btn btn-primary' : 'btn';
        }
        
        this.sendMessage({
            type: 'control',
            action: this.isPaused ? 'pause' : 'resume'
        });
        
        console.log('⏸️ Pause toggled:', this.isPaused);
    }

    transferToHuman() {
        console.log('👤 Requesting human transfer...');
        
        this.sendMessage({
            type: 'transfer',
            reason: 'User requested human agent',
            timestamp: new Date().toISOString()
        });
        
        this.updateStatus('Requesting human agent...');
    }

    endCall() {
        console.log('📞 Ending call...');
        
        try {
            this.isRecording = false;
            this.audioPlayer.stopCurrentAudio();
            this.audioBuffer = [];
            
            if (this.audioProcessor) {
                try {
                    this.audioProcessor.disconnect();
                    this.audioProcessor = null;
                } catch (error) {
                    console.error('❌ Error disconnecting processor:', error);
                }
            }
            
            if (this.audioSource) {
                try {
                    this.audioSource.disconnect();
                    this.audioSource = null;
                } catch (error) {
                    console.error('❌ Error disconnecting source:', error);
                }
            }
            
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => {
                    try {
                        track.stop();
                    } catch (error) {
                        console.error('❌ Error stopping track:', error);
                    }
                });
                this.mediaStream = null;
            }
            
            if (this.audioContext && this.audioContext.state !== 'closed') {
                try {
                    this.audioContext.close();
                } catch (error) {
                    console.error('❌ Error closing audio context:', error);
                }
            }
            
            this.sendMessage({
                type: 'end_call',
                timestamp: new Date().toISOString()
            });
            
            if (this.ws) {
                try {
                    this.ws.close(1000, 'Call ended by user');
                } catch (error) {
                    console.error('❌ Error closing WebSocket:', error);
                }
            }
            
            this.updateStatus('Call ended');
            this.stopDurationTimer();
            
            console.log('✅ Call cleanup completed');
            
            // Redirect to home after brief delay
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
            
        } catch (error) {
            console.error('❌ Error during call cleanup:', error);
            // Force redirect even if cleanup fails
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);
        }
    }

    // Utility methods
    getSessionStats() {
        return {
            sessionId: this.sessionId,
            isConnected: this.isConnected,
            isRecording: this.isRecording,
            isMuted: this.isMuted,
            isPaused: this.isPaused,
            audioStatus: this.audioPlayer.getStatus(),
            websocketState: this.ws?.readyState,
            audioContextState: this.audioContext?.state,
            bufferSize: this.audioBuffer.length,
            productionMode: this.productionMode,
            currentSequence: this.currentAudioSequence
        };
    }
    
    enableDebugMode() {
        this.debugMode = true;
        console.log('🐛 Debug mode enabled');
    }
    
    disableDebugMode() {
        this.debugMode = false;
        console.log('🐛 Debug mode disabled');
    }
}

// Global instance
let callHandler = null;

// PRODUCTION interface functions
async function startCall() {
    console.log('🎯 Starting STREAMING call...');
    
    if (!callHandler) {
        callHandler = new CallHandler();
    }
    
    // Update UI immediately
    document.getElementById('start-call-btn').style.display = 'none';
    document.getElementById('mute-btn').style.display = 'inline-block';
    document.getElementById('pause-btn').style.display = 'inline-block';
    document.getElementById('transfer-btn').style.display = 'inline-block';
    document.getElementById('end-call-btn').style.display = 'inline-block';
    
    document.getElementById('call-status').textContent = 'Starting call...';
    
    try {
        await callHandler.initialize();
        console.log('✅ STREAMING call started successfully');
        
    } catch (error) {
        console.error('❌ Failed to start call:', error);
        
        // Show specific error messages
        let errorMessage = 'Failed to start call';
        if (error.message.includes('denied')) {
            errorMessage = 'Microphone access denied. Please allow microphone access and try again.';
            showMicrophoneInstructions();
        } else if (error.message.includes('timeout')) {
            errorMessage = 'Connection timeout. Please check your internet connection.';
        } else if (error.message.includes('WebSocket')) {
            errorMessage = 'Connection failed. Please try again later.';
        } else {
            errorMessage = `Error: ${error.message}`;
        }
        
        document.getElementById('call-status').textContent = errorMessage;
        
        // Reset UI on error
        document.getElementById('start-call-btn').style.display = 'inline-block';
        document.getElementById('mute-btn').style.display = 'none';
        document.getElementById('pause-btn').style.display = 'none';  
        document.getElementById('transfer-btn').style.display = 'none';
        document.getElementById('end-call-btn').style.display = 'none';
    }
}

function toggleMute() {
    if (callHandler) {
        callHandler.toggleMute();
    } else {
        console.warn('⚠️ Call handler not initialized');
    }
}

function togglePause() {
    if (callHandler) {
        callHandler.togglePause();
    } else {
        console.warn('⚠️ Call handler not initialized');
    }
}

function transferToHuman() {
    if (callHandler) {
        callHandler.transferToHuman();
    } else {
        console.warn('⚠️ Call handler not initialized');
    }
}

function endCall() {
    if (callHandler) {
        callHandler.endCall();
    } else {
        console.warn('⚠️ Call handler not initialized');
        // Fallback - just redirect
        window.location.href = '/';
    }
}

function showMicrophoneInstructions() {
    const instructions = document.createElement('div');
    instructions.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #dc2626;
        color: white;
        padding: 20px;
        border-radius: 10px;
        z-index: 1000;
        max-width: 500px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    instructions.innerHTML = `
        <div style="font-size: 24px; margin-bottom: 10px;">🎤</div>
        <strong>Microphone Access Required</strong><br><br>
        To use voice calls, please:<br>
        1. Look for the microphone icon 🎤 in your browser's address bar<br>
        2. Click it and select "Allow"<br>
        3. Or click the 🔒 lock icon → Site Settings → Microphone → Allow<br>
        4. Refresh this page and try again<br><br>
        
        <strong>For HTTPS issues:</strong><br>
        • Make sure you're accessing via https://localhost:8000<br>
        • Accept the security certificate if prompted<br><br>
        
        <button onclick="this.parentElement.remove()" style="background: white; color: #dc2626; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; margin-top: 10px;">Got it!</button>
    `;
    document.body.appendChild(instructions);
    
    // Auto-remove after 20 seconds
    setTimeout(() => {
        if (instructions.parentElement) {
            instructions.remove();
        }
    }, 20000);
}

// PRODUCTION utility functions
function checkCallHandler() {
    if (callHandler) {
        const stats = callHandler.getSessionStats();
        console.log('📊 Call Handler Status:', {
            connected: stats.isConnected,
            recording: stats.isRecording,
            muted: stats.isMuted,
            audioPlaying: stats.audioStatus.isPlaying,
            sessionId: stats.sessionId,
            streamingBuffer: stats.audioStatus.pendingChunks
        });
        return stats;
    } else {
        console.log('❌ Call handler not initialized');
        return null;
    }
}

function setAudioVolume(volume) {
    if (callHandler && callHandler.audioPlayer) {
        callHandler.audioPlayer.setVolume(volume);
        console.log('🔊 Volume set to:', volume);
    } else {
        console.error('❌ Call handler or audio player not available');
    }
}

// Text message function for testing
function sendTestMessage(message) {
    if (callHandler && callHandler.isConnected) {
        console.log('📤 Sending text message:', message);
        
        const success = callHandler.sendMessage({
            type: 'text',
            text: message,
            timestamp: new Date().toISOString()
        });
        
        if (success) {
            console.log('✅ Message sent successfully');
        } else {
            console.error('❌ Failed to send message');
        }
    } else {
        console.error('❌ Call handler not connected');
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 STREAMING Call Handler page loaded');
    
    // Update initial status
    setTimeout(() => {
        const status = document.getElementById('call-status');
        if (status && status.textContent === 'Ready to start') {
            status.textContent = 'Click "Start Call" to begin';
        }
    }, 1000);
});

// Add keyboard shortcuts for accessibility
document.addEventListener('keydown', (event) => {
    if (event.altKey && callHandler) {
        switch(event.code) {
            case 'KeyM':
                event.preventDefault();
                toggleMute();
                break;
            case 'KeyP':
                event.preventDefault();
                togglePause();
                break;
            case 'KeyE':
                event.preventDefault();
                endCall();
                break;
            case 'KeyT':
                event.preventDefault();
                transferToHuman();
                break;
        }
    }
});

// Handle page unload cleanup
window.addEventListener('beforeunload', (event) => {
    if (callHandler && callHandler.isConnected) {
        try {
            callHandler.endCall();
        } catch (error) {
            console.error('❌ Cleanup error on page unload:', error);
        }
    }
});

// Export functions for console access
window.checkCallHandler = checkCallHandler;
window.setAudioVolume = setAudioVolume;
window.sendTestMessage = sendTestMessage;

console.log('✅ STREAMING AUDIO Call Handler loaded');
console.log('🎵 Natural voice streaming - NO MORE ROBOTIC CHUNKS!');
console.log('🔧 Available functions:');
console.log('  - checkCallHandler() - Check status');
console.log('  - setAudioVolume(0.8) - Set volume');
console.log('  - sendTestMessage("text") - Send text message');
console.log('⌨️  Keyboard shortcuts: Alt+M (mute), Alt+P (pause), Alt+E (end), Alt+T (transfer)');
console.log('🎯 Ready for NATURAL voice calls!');