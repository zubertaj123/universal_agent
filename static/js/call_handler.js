/**
 * Complete Fixed Call Handler - NATURAL AUDIO VERSION
 * This fixes all robotic voice, speed, and connection issues
 */

/**
 * Natural Audio Player for fixing robotic voice issues
 */
class NaturalAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentAudio = null;
        this.masterVolume = 0.8;
        
        // CRITICAL: Natural audio settings to fix robotic voice
        this.naturalSettings = {
            playbackRate: 1.0,  // NEVER change - normal speed
            volume: this.masterVolume,
            preservesPitch: true,
            preload: 'auto'
        };
        
        this.initializeAudioContext();
    }
    
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('✅ Natural Audio Context initialized');
        } catch (error) {
            console.error('❌ Audio Context initialization failed:', error);
        }
    }
    
    /**
     * FIXED: Play audio with natural settings to prevent robotic voice
     */
    async playNaturalAudio(audioData, metadata = {}) {
        try {
            console.log('🔊 Playing natural audio chunk:', audioData.length, 'bytes');
            
            // Create proper audio blob for natural playback
            const audioBlob = new Blob([audioData], { 
                type: 'audio/mpeg'
            });
            
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            // CRITICAL: Apply natural playback settings to prevent robotic voice
            audio.playbackRate = this.naturalSettings.playbackRate;
            audio.volume = this.naturalSettings.volume;
            audio.preservesPitch = this.naturalSettings.preservesPitch;
            audio.preload = this.naturalSettings.preload;
            
            return new Promise((resolve, reject) => {
                audio.addEventListener('loadeddata', async () => {
                    try {
                        console.log('🔊 Audio loaded, starting natural playback...');
                        await audio.play();
                        console.log('✅ Natural audio playback started successfully');
                        resolve(true);
                    } catch (playError) {
                        console.error('❌ Natural audio play failed:', playError);
                        reject(playError);
                    }
                }, { once: true });
                
                audio.addEventListener('ended', () => {
                    console.log('🔊 Natural audio playback completed');
                    URL.revokeObjectURL(audioUrl);
                    resolve(true);
                }, { once: true });
                
                audio.addEventListener('error', (e) => {
                    console.error('🔊 Natural audio playback error:', e);
                    URL.revokeObjectURL(audioUrl);
                    reject(e);
                }, { once: true });
                
                this.currentAudio = audio;
                audio.load();
                
                // Timeout protection
                setTimeout(() => {
                    if (audio.readyState < 2) {
                        console.warn('⚠️ Natural audio load timeout');
                        reject(new Error('Audio load timeout'));
                    }
                }, 5000);
            });
            
        } catch (error) {
            console.error('❌ Natural audio playback setup failed:', error);
            return false;
        }
    }
    
    /**
     * Handle audio chunks for natural streaming
     */
    async handleAudioChunk(hexData, metadata = {}) {
        try {
            if (!hexData || typeof hexData !== 'string') {
                console.warn('⚠️ Invalid audio chunk data');
                return false;
            }
            
            console.log('📥 Processing natural audio chunk:', hexData.length, 'hex chars');
            
            // Convert hex to bytes
            const bytes = new Uint8Array(
                hexData.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
            );
            
            console.log('🔊 Audio chunk converted:', bytes.length, 'bytes');
            
            // Play with natural settings to prevent robotic sound
            const success = await this.playNaturalAudio(bytes, metadata);
            
            if (success) {
                console.log('✅ Natural audio chunk played successfully');
            } else {
                console.warn('⚠️ Natural audio chunk playback failed');
            }
            
            return success;
            
        } catch (error) {
            console.error('❌ Audio chunk processing failed:', error);
            return false;
        }
    }
    
    stopCurrentAudio() {
        if (this.currentAudio) {
            try {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
                this.currentAudio = null;
                console.log('🛑 Current audio stopped');
            } catch (error) {
                console.error('❌ Error stopping audio:', error);
            }
        }
    }
    
    setVolume(volume) {
        this.masterVolume = Math.max(0, Math.min(1, volume));
        this.naturalSettings.volume = this.masterVolume;
        
        if (this.currentAudio) {
            this.currentAudio.volume = this.masterVolume;
        }
        
        console.log('🔊 Volume set to:', this.masterVolume);
    }
    
    getStatus() {
        return {
            isPlaying: this.isPlaying,
            currentAudio: !!this.currentAudio,
            volume: this.masterVolume,
            settings: this.naturalSettings,
            queueLength: this.audioQueue.length
        };
    }
}

/**
 * Complete Fixed Call Handler with Natural Audio
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
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.startTime = null;
        this.durationInterval = null;
        this.connectionReady = false;
        
        // Audio processing parameters
        this.targetSampleRate = 16000;
        this.chunkSize = 1024;
        this.audioWorkletNode = null;
        this.isRecording = false;
        this.debugMode = false;
        
        // FIXED: Natural audio player to prevent robotic voice
        this.naturalAudioPlayer = new NaturalAudioPlayer();
        this.audioSequenceActive = false;
    }

    async initialize() {
        console.log('🚀 Initializing Call Handler with Natural Audio...');
        
        this.sessionId = this.generateSessionId();
        console.log('Generated session ID:', this.sessionId);
        
        try {
            console.log('🎵 Initializing audio context...');
            await this.initializeAudioContext();
            
            console.log('🎤 Requesting microphone access...');
            await this.requestMicrophoneAccess();
            
            console.log('🔌 Setting up WebSocket connection...');
            await this.connectWebSocket();
            
            console.log('🔊 Setting up audio processing...');
            await this.setupAudioProcessing();
            
            this.startDurationTimer();
            this.updateStatus('Connected and Ready with Natural Audio');
            
            console.log('✅ Call Handler with Natural Audio initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Call Handler:', error);
            throw error;
        }
    }

    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.targetSampleRate
            });
            
            console.log('Audio context initialized:', {
                sampleRate: this.audioContext.sampleRate,
                state: this.audioContext.state
            });
            
        } catch (error) {
            console.warn('Could not set target sample rate, using default');
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    async requestMicrophoneAccess() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: this.targetSampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log('Media stream obtained:', {
                sampleRate: this.audioContext.sampleRate,
                tracks: this.mediaStream.getAudioTracks().length,
                settings: this.mediaStream.getAudioTracks()[0].getSettings()
            });
            
        } catch (error) {
            console.error('Microphone access error:', error);
            if (error.name === 'NotAllowedError') {
                throw new Error('Permission denied: Microphone access was denied by the user');
            } else if (error.name === 'NotFoundError') {
                throw new Error('No microphone found: Please connect a microphone and try again');
            } else {
                throw new Error(`Microphone error: ${error.message}`);
            }
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsUrl = this.buildWebSocketUrl();
            
            console.log('Connecting to WebSocket:', wsUrl);
            
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.connectionReady = true;
                this.updateStatus('Connected with Natural Audio');
                console.log('✅ WebSocket connected successfully to:', wsUrl);
                
                this.sendMessage({
                    type: 'connection',
                    sessionId: this.sessionId,
                    timestamp: new Date().toISOString()
                });
                
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.updateStatus('Connection error');
                reject(new Error(`WebSocket connection failed: ${error.message || 'Unknown error'}`));
            };
            
            this.ws.onclose = (event) => {
                this.isConnected = false;
                this.connectionReady = false;
                console.log('WebSocket closed:', {
                    code: event.code,
                    reason: event.reason,
                    wasClean: event.wasClean
                });
                
                this.updateStatus('Disconnected');
                this.stopDurationTimer();
            };
            
            const connectionTimeout = setTimeout(() => {
                if (!this.connectionReady) {
                    console.error('WebSocket connection timeout');
                    if (this.ws) {
                        this.ws.close();
                    }
                    reject(new Error('WebSocket connection timeout - server may be unreachable'));
                }
            }, 15000);
            
            this.ws.addEventListener('open', () => {
                clearTimeout(connectionTimeout);
            });
        });
    }

    buildWebSocketUrl() {
        const isSecure = window.location.protocol === 'https:';
        const wsProtocol = isSecure ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${wsProtocol}//${host}/ws/call/${this.sessionId}`;
        return wsUrl;
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('Error sending WebSocket message:', error);
                return false;
            }
        } else {
            console.warn('WebSocket not connected, cannot send message:', message.type);
            return false;
        }
    }

    async setupAudioProcessing() {
        try {
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (event) => {
                if (!this.isRecording || this.isPaused || this.isMuted) {
                    return;
                }
                
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                this.processAudioChunk(inputData);
            };
            
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            this.audioSource = source;
            this.audioProcessor = processor;
            this.isRecording = true;
            
            this.setupAudioVisualization(source);
            
            console.log('✅ Audio processing setup complete');
            
        } catch (error) {
            console.error('❌ Error setting up audio processing:', error);
            await this.setupFallbackAudioRecording();
        }
    }

    processAudioChunk(audioData) {
        try {
            const pcmData = this.convertToPCM16(audioData);
            
            if (pcmData.length > 0 && this.hasSignificantAudio(audioData)) {
                const hexString = Array.from(pcmData)
                    .map(b => b.toString(16).padStart(2, '0'))
                    .join('');
                
                const success = this.sendMessage({
                    type: 'audio',
                    data: hexString,
                    format: 'pcm16',
                    sampleRate: this.audioContext.sampleRate,
                    channels: 1,
                    timestamp: Date.now()
                });
                
                if (this.debugMode && Math.random() < 0.01) {
                    console.log('📤 Sent audio chunk:', {
                        success,
                        size: pcmData.length,
                        sampleRate: this.audioContext.sampleRate,
                        hasAudio: this.hasSignificantAudio(audioData)
                    });
                }
            }
        } catch (error) {
            console.error('Error processing audio chunk:', error);
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
            console.error('Error converting to PCM16:', error);
            return new Uint8Array(0);
        }
    }

    hasSignificantAudio(audioData) {
        const rms = Math.sqrt(audioData.reduce((sum, sample) => sum + sample * sample, 0) / audioData.length);
        return rms > 0.001;
    }

    async setupFallbackAudioRecording() {
        console.log('🔄 Setting up fallback MediaRecorder...');
        
        try {
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            };
            
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'audio/mp4';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        delete options.mimeType;
                    }
                }
            }
            
            this.mediaRecorder = new MediaRecorder(this.mediaStream, options);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && !this.isPaused && !this.isMuted) {
                    event.data.arrayBuffer().then(buffer => {
                        const uint8Array = new Uint8Array(buffer);
                        const hexString = Array.from(uint8Array)
                            .map(b => b.toString(16).padStart(2, '0'))
                            .join('');
                        
                        this.sendMessage({
                            type: 'audio',
                            data: hexString,
                            format: this.mediaRecorder.mimeType,
                            fallback: true,
                            timestamp: Date.now()
                        });
                    }).catch(error => {
                        console.error('Error processing MediaRecorder data:', error);
                    });
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                console.error('MediaRecorder error:', error);
            };
            
            this.mediaRecorder.start(100);
            this.isRecording = true;
            
            console.log('✅ Fallback MediaRecorder setup complete');
            
        } catch (error) {
            console.error('❌ Error setting up MediaRecorder fallback:', error);
            this.updateStatus('Audio recording setup failed');
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
            
        } catch (error) {
            console.error('Error setting up audio visualization:', error);
        }
    }

    /**
     * FIXED: Handle WebSocket messages with natural audio processing
     */
    handleWebSocketMessage(data) {
        try {
            console.log('📨 Received message:', data.type);
            
            switch (data.type) {
                case 'audio_start':
                    this.handleAudioStart(data);
                    break;
                    
                case 'audio_chunk':
                    this.handleNaturalAudioChunk(data);
                    break;
                    
                case 'audio_complete':
                    this.handleAudioComplete(data);
                    break;
                    
                case 'transcript':
                    this.addTranscript(data.speaker, data.text);
                    break;
                    
                case 'status':
                    this.updateStatus(data.message);
                    break;
                    
                case 'error':
                case 'audio_error':
                    console.error('❌ Server error:', data.message);
                    this.updateStatus('Error: ' + data.message);
                    break;
                    
                case 'connected':
                    this.updateStatus('Connected to server with natural audio');
                    break;
                    
                case 'test_response':
                    console.log('🧪 Test response received:', data);
                    break;
                    
                default:
                    console.log('❓ Unknown message type:', data.type);
            }
            
        } catch (error) {
            console.error('❌ Error handling WebSocket message:', error);
        }
    }
    
    handleAudioStart(data) {
        console.log('🎬 Audio sequence starting:', data.text);
        this.audioSequenceActive = true;
        
        // Stop any currently playing audio to prevent overlap
        this.naturalAudioPlayer.stopCurrentAudio();
        
        this.updateStatus('Agent speaking...');
    }
    
    /**
     * FIXED: Handle natural audio chunks to prevent robotic voice
     */
    async handleNaturalAudioChunk(data) {
        try {
            if (!this.audioSequenceActive) {
                console.log('🎬 Starting new audio sequence');
                this.audioSequenceActive = true;
            }
            
            console.log('🔊 Processing natural audio chunk:', data.chunk_number);
            
            // CRITICAL: Use natural audio player to prevent robotic voice
            const success = await this.naturalAudioPlayer.handleAudioChunk(
                data.data, 
                {
                    chunkNumber: data.chunk_number,
                    naturalTiming: data.natural_timing,
                    playbackRate: data.playback_rate || 1.0
                }
            );
            
            if (success) {
                console.log('✅ Natural audio chunk processed successfully');
            } else {
                console.warn('⚠️ Natural audio chunk processing failed');
            }
            
        } catch (error) {
            console.error('❌ Natural audio chunk handling failed:', error);
        }
    }
    
    handleAudioComplete(data) {
        console.log('🎬 Audio sequence completed:', data.chunks_sent, 'chunks');
        this.audioSequenceActive = false;
        
        this.updateStatus('Listening...');
        
        console.log('📊 Audio stats:', {
            chunks: data.chunks_sent,
            bytes: data.total_bytes,
            natural: data.natural_completion
        });
    }

    addTranscript(speaker, text) {
        try {
            const transcriptDiv = document.getElementById('transcript');
            if (!transcriptDiv) {
                console.error('❌ Transcript div not found');
                return;
            }
            
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
            
            // Limit transcript entries to prevent memory issues
            const entries = transcriptDiv.querySelectorAll('.transcript-entry');
            if (entries.length > 50) {
                entries[0].remove();
            }
            
        } catch (error) {
            console.error('❌ Error adding transcript:', error);
        }
    }

    updateStatus(status) {
        const statusElement = document.getElementById('call-status');
        if (statusElement) {
            statusElement.textContent = status;
        }
        console.log('📊 Status update:', status);
    }

    startDurationTimer() {
        this.startTime = Date.now();
        
        this.durationInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            const durationElement = document.getElementById('call-duration');
            if (durationElement) {
                durationElement.textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
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

    toggleMute() {
        this.isMuted = !this.isMuted;
        const muteBtn = document.getElementById('mute-btn');
        if (muteBtn) {
            muteBtn.textContent = this.isMuted ? '🔇 Unmute' : '🎤 Mute';
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
        }
        
        this.sendMessage({
            type: 'control',
            action: this.isPaused ? 'pause' : 'resume'
        });
        
        console.log('⏸️ Pause toggled:', this.isPaused);
    }

    transferToHuman() {
        this.sendMessage({
            type: 'transfer',
            reason: 'User requested human agent'
        });
        
        this.updateStatus('Transferring to human agent...');
    }

    endCall() {
        console.log('📞 Ending call...');
        
        this.isRecording = false;
        
        // Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.error('Error stopping MediaRecorder:', error);
            }
        }
        
        // Disconnect audio processing nodes
        if (this.audioProcessor) {
            try {
                this.audioProcessor.disconnect();
            } catch (error) {
                console.error('Error disconnecting audio processor:', error);
            }
        }
        
        if (this.audioSource) {
            try {
                this.audioSource.disconnect();
            } catch (error) {
                console.error('Error disconnecting audio source:', error);
            }
        }
        
        // Stop media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (error) {
                    console.error('Error stopping media track:', error);
                }
            });
        }
        
        // Stop any playing audio
        this.naturalAudioPlayer.stopCurrentAudio();
        
        // Send end call message
        this.sendMessage({
            type: 'end_call'
        });
        
        // Close WebSocket
        if (this.ws) {
            try {
                this.ws.close();
            } catch (error) {
                console.error('Error closing WebSocket:', error);
            }
        }
        
        this.updateStatus('Call ended');
        
        // Redirect to home page after brief delay
        setTimeout(() => {
            window.location.href = '/';
        }, 2000);
    }

    enableDebugMode() {
        this.debugMode = true;
        console.log('🐛 Debug mode enabled');
    }

    getSessionStats() {
        return {
            sessionId: this.sessionId,
            isConnected: this.isConnected,
            connectionReady: this.connectionReady,
            isRecording: this.isRecording,
            isMuted: this.isMuted,
            isPaused: this.isPaused,
            websocket: {
                readyState: this.ws?.readyState,
                url: this.ws?.url,
                protocol: this.ws?.protocol
            },
            audioContext: {
                state: this.audioContext?.state,
                sampleRate: this.audioContext?.sampleRate
            },
            mediaStream: {
                active: this.mediaStream?.active,
                tracks: this.mediaStream?.getTracks().length
            },
            naturalAudio: this.naturalAudioPlayer.getStatus()
        };
    }

    // Test method for natural voice
    testNaturalVoice() {
        console.log('🧪 Testing natural voice playback...');
        
        if (this.isConnected) {
            this.sendMessage({
                type: 'text',
                text: 'Please test natural voice speed. This should sound normal, not fast or robotic.',
                timestamp: new Date().toISOString()
            });
        } else {
            console.error('Not connected - start a call first');
        }
    }

    // Set audio quality
    setAudioVolume(volume) {
        this.naturalAudioPlayer.setVolume(volume);
        console.log('🔊 Audio volume set to:', volume);
    }
    
    getAudioStatus() {
        return this.naturalAudioPlayer.getStatus();
    }
}

// Global variable for call handler instance
let callHandler = null;

// Global functions for call interface
async function startCall() {
    console.log('🎯 Starting call with FIXED natural audio...');
    
    if (!callHandler) {
        callHandler = new CallHandler();
    }
    
    // Update UI to show call is starting
    document.getElementById('start-call-btn').style.display = 'none';
    document.getElementById('mute-btn').style.display = 'inline-block';
    document.getElementById('pause-btn').style.display = 'inline-block';
    document.getElementById('transfer-btn').style.display = 'inline-block';
    document.getElementById('end-call-btn').style.display = 'inline-block';
    
    document.getElementById('call-status').textContent = 'Starting with natural audio...';
    
    try {
        await callHandler.initialize();
        console.log('✅ Call handler with natural audio initialized');
        
        // Set optimal audio volume for natural playback
        callHandler.setAudioVolume(0.8);
        
    } catch (error) {
        console.error('❌ Failed to start call:', error);
        
        // Error handling with specific messages
        let errorMessage = 'Failed to start call';
        if (error.message.includes('Permission denied')) {
            errorMessage = 'Microphone access denied. Please allow microphone access.';
            showMicrophoneInstructions();
        } else if (error.message.includes('WebSocket')) {
            errorMessage = 'Connection failed. Please check server.';
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
        console.warn('Call handler not initialized');
    }
}

function togglePause() {
    if (callHandler) {
        callHandler.togglePause();
    } else {
        console.warn('Call handler not initialized');
    }
}

function transferToHuman() {
    if (callHandler) {
        callHandler.transferToHuman();
    } else {
        console.warn('Call handler not initialized');
    }
}

function endCall() {
    if (callHandler) {
        callHandler.endCall();
    } else {
        console.warn('Call handler not initialized');
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

// Test functions for natural audio debugging
function testNaturalAudio() {
    if (callHandler) {
        callHandler.testNaturalVoice();
    } else {
        console.error('❌ Call handler not initialized');
    }
}

function checkAudioStatus() {
    if (callHandler) {
        const status = callHandler.getAudioStatus();
        console.log('🔊 Audio Status:', status);
        return status;
    } else {
        console.log('❌ Call handler not available');
        return null;
    }
}

function setAudioVolume(volume) {
    if (callHandler) {
        callHandler.setAudioVolume(volume);
    } else {
        console.error('❌ Call handler not initialized');
    }
}

function checkCallHandler() {
    if (callHandler) {
        console.log('Call Handler Status:', callHandler.getSessionStats());
    } else {
        console.log('Call handler not initialized');
    }
}

// Test message functions for debugging
function sendTestMessage(message) {
    if (callHandler && callHandler.isConnected) {
        console.log('📤 Sending test message:', message);
        
        const success = callHandler.sendMessage({
            type: 'text',
            text: message,
            timestamp: new Date().toISOString()
        });
        
        if (success) {
            console.log('✅ Test message sent successfully');
        } else {
            console.error('❌ Failed to send test message');
        }
    } else {
        console.error('❌ Call handler not connected');
    }
}

function testConversation() {
    if (!callHandler || !callHandler.isConnected) {
        console.error('❌ Call handler not connected. Start a call first.');
        return;
    }
    
    console.log('🧪 Starting test conversation...');
    
    const testMessages = [
        "Hello, I need help with my insurance",
        "I want to file a claim for my car accident",
        "The accident happened yesterday",
        "My policy number is AUTO-123456"
    ];
    
    let messageIndex = 0;
    
    function sendNextMessage() {
        if (messageIndex < testMessages.length) {
            const message = testMessages[messageIndex];
            console.log(`📝 Test message ${messageIndex + 1}:`, message);
            sendTestMessage(message);
            messageIndex++;
            
            // Wait 5 seconds before next message
            setTimeout(sendNextMessage, 5000);
        } else {
            console.log('✅ Test conversation completed');
        }
    }
    
    // Start after 2 seconds
    setTimeout(sendNextMessage, 2000);
}

function checkWebSocketStatus() {
    if (callHandler) {
        const stats = callHandler.getSessionStats();
        console.log('📊 WebSocket Status:', {
            connected: stats.isConnected,
            readyState: stats.websocket?.readyState,
            url: stats.websocket?.url,
            sessionId: stats.sessionId
        });
        
        const readyStates = {
            0: 'CONNECTING',
            1: 'OPEN', 
            2: 'CLOSING',
            3: 'CLOSED'
        };
        
        console.log('WebSocket Ready State:', readyStates[stats.websocket?.readyState] || 'UNKNOWN');
    } else {
        console.log('❌ Call handler not initialized');
    }
}

function debugAudioPlayback() {
    console.log('🐛 Audio Debug Information:');
    
    if (callHandler) {
        console.log('AudioContext state:', callHandler.audioContext?.state);
        console.log('AudioContext sample rate:', callHandler.audioContext?.sampleRate);
        console.log('Natural Audio Player:', callHandler.naturalAudioPlayer.getStatus());
        
        const audio = new Audio();
        console.log('Browser audio support:');
        console.log('- MP3:', audio.canPlayType('audio/mpeg'));
        console.log('- WAV:', audio.canPlayType('audio/wav'));
        console.log('- WebM:', audio.canPlayType('audio/webm'));
        console.log('- OGG:', audio.canPlayType('audio/ogg'));
    } else {
        console.error('❌ Call handler not available');
    }
}

// Add functions to global scope for debugging
window.sendTestMessage = sendTestMessage;
window.testConversation = testConversation;
window.checkWebSocketStatus = checkWebSocketStatus;
window.debugAudioPlayback = debugAudioPlayback;
window.checkCallHandler = checkCallHandler;
window.testNaturalAudio = testNaturalAudio;
window.checkAudioStatus = checkAudioStatus;
window.setAudioVolume = setAudioVolume;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 Page loaded, call handler ready to initialize with natural audio');
    
    setTimeout(() => {
        const status = document.getElementById('call-status');
        if (status && status.textContent === 'Ready to start') {
            status.textContent = 'Click "Start Call" for natural audio experience';
        }
    }, 1000);
});

// Add keyboard shortcuts for accessibility
document.addEventListener('keydown', (event) => {
    if (event.altKey) {
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
        }
    }
});

// Handle page unload to cleanup resources
window.addEventListener('beforeunload', (event) => {
    if (callHandler && callHandler.isConnected) {
        callHandler.endCall();
    }
});

console.log('✅ Complete Fixed Call Handler loaded with Natural Audio');
console.log('🧪 Test functions available:');
console.log('- testNaturalAudio() - Test natural voice');
console.log('- sendTestMessage("your message") - Send test message');
console.log('- testConversation() - Run test conversation');
console.log('- checkWebSocketStatus() - Check connection');
console.log('- debugAudioPlayback() - Debug audio support');
console.log('- checkCallHandler() - Check handler status');
console.log('- setAudioVolume(0.8) - Set audio volume');
console.log('- checkAudioStatus() - Check audio player status');
console.log('🎯 Ready for natural audio calls!');