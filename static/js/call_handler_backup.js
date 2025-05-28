/**
 * Fixed Call Handler - Complete Version with Audio Overlap Fix
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
        this.shouldStopCurrentAudio = false;
        
        // Audio processing parameters
        this.targetSampleRate = 16000;
        this.chunkSize = 1024;
        this.audioWorkletNode = null;
        this.isRecording = false;
        this.debugMode = false;
        
        // FIXED: Audio playback tracking to prevent overlap
        this.currentAudioElement = null;
        this.currentAudioUrl = null;
        this.currentAudioSource = null;
        this.audioSettings = {
            volume: 0.9,
            playbackRate: 1.0,
            useCompressor: true
        };
    }

    async initialize() {
        console.log('🚀 Initializing Call Handler...');
        
        // Generate session ID
        this.sessionId = this.generateSessionId();
        console.log('Generated session ID:', this.sessionId);
        
        try {
            // Step 1: Initialize audio context
            console.log('🎵 Initializing audio context...');
            await this.initializeAudioContext();
            
            // Step 2: Request microphone access
            console.log('🎤 Requesting microphone access...');
            await this.requestMicrophoneAccess();
            
            // Step 3: Setup WebSocket connection
            console.log('🔌 Setting up WebSocket connection...');
            await this.connectWebSocket();
            
            // Step 4: Setup audio processing
            console.log('🔊 Setting up audio processing...');
            await this.setupAudioProcessing();
            
            // Step 5: Start duration timer
            this.startDurationTimer();
            
            // Step 6: Update UI
            this.updateStatus('Connected and Ready');
            
            console.log('✅ Call Handler initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Call Handler:', error);
            throw error;
        }
    }

    async initializeAudioContext() {
        try {
            // Initialize audio context with target sample rate
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
                this.updateStatus('Connected');
                console.log('✅ WebSocket connected successfully to:', wsUrl);
                
                // Send initial connection message
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
            
            // Limit transcript entries
            const entries = transcriptDiv.querySelectorAll('.transcript-entry');
            if (entries.length > 50) {
                entries[0].remove();
            }
            
        } catch (error) {
            console.error('❌ Error adding transcript:', error);
        }
    }

    async playAudio(hexData) {
        try {
            if (!hexData || typeof hexData !== 'string') {
                console.warn('⚠️ Invalid audio data received');
                return;
            }

            console.log('🔊 Processing TTS audio data:', hexData.length + ' chars');

            // Convert hex string back to array buffer
            const bytes = new Uint8Array(
                hexData.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
            );

            console.log('🔊 Audio bytes converted:', bytes.length);

            // FIXED: Only stop if we're starting a new sentence/response
            // Don't stop for every chunk of the same response
            if (this.shouldStopCurrentAudio) {
                await this.stopCurrentAudio();
                this.shouldStopCurrentAudio = false;
            }

            // Play audio chunk
            const success = await this.playAudioChunk(bytes);

            if (!success) {
                console.warn('⚠️ Audio chunk playback failed');
            }

        } catch (error) {
            console.error('❌ Audio playback error:', error);
        }
    }

    // NEW: Play individual audio chunks without stopping previous ones
    async playAudioChunk(bytes) {
        try {
            // Create blob with proper MIME type
            const audioBlob = new Blob([bytes], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(audioBlob);

            console.log('🔊 Created audio chunk URL:', audioUrl);

            // Create audio element
            const audio = new Audio(audioUrl);
            audio.volume = this.audioSettings.volume;
            audio.playbackRate = this.audioSettings.playbackRate;
            audio.preservesPitch = true;

            // Play immediately without waiting for canplaythrough
            return new Promise((resolve) => {
                audio.addEventListener('loadeddata', async () => {
                    try {
                        await audio.play();
                        console.log('✅ Audio chunk started playing');
                        resolve(true);
                    } catch (playError) {
                        console.error('❌ Play failed:', playError);
                        resolve(false);
                    }
                }, { once: true });

                audio.addEventListener('ended', () => {
                    URL.revokeObjectURL(audioUrl);
                    console.log('🔊 Audio chunk finished');
                }, { once: true });

                audio.addEventListener('error', (e) => {
                    console.error('🔊 Audio chunk error:', e);
                    URL.revokeObjectURL(audioUrl);
                    resolve(false);
                }, { once: true });

                // Start loading
                audio.load();

                // Timeout
                setTimeout(() => {
                    if (audio.readyState < 2) {
                        console.warn('Audio chunk load timeout');
                        resolve(false);
                    }
                }, 5000);
            });

        } catch (error) {
            console.error('❌ Audio chunk creation failed:', error);
            return false;
        }
    }

    // Updated handleWebSocketMessage to handle audio_complete
    handleWebSocketMessage(data) {
        try {
            console.log('📨 Received WebSocket message:', data.type, data);
            
            switch (data.type) {
                case 'transcript':
                    this.addTranscript(data.speaker, data.text);
                    break;
                    
                case 'audio':
                    this.playAudio(data.data);
                    break;
                    
                case 'audio_complete':
                    console.log('🔊 Audio sequence complete:', data);
                    this.shouldStopCurrentAudio = true; // Prepare for next response
                    break;
                    
                case 'status':
                    this.updateStatus(data.message);
                    break;
                    
                case 'error':
                    console.error('❌ Server error:', data.message);
                    this.updateStatus('Error: ' + data.message);
                    break;
                    
                case 'connected':
                    this.updateStatus('Connected to server');
                    break;
                    
                case 'test_response':
                    console.log('🧪 Test response received:', data);
                    break;
                    
                default:
                    console.log('❓ Unknown message type:', data.type, data);
            }
        } catch (error) {
            console.error('❌ Error handling WebSocket message:', error);
        }
    } 

    // NEW: Stop any currently playing audio
    async stopCurrentAudio() {
        // Stop any existing audio elements
        if (this.currentAudioElement) {
            try {
                this.currentAudioElement.pause();
                this.currentAudioElement.currentTime = 0;
                if (this.currentAudioUrl) {
                    URL.revokeObjectURL(this.currentAudioUrl);
                    this.currentAudioUrl = null;
                }
            } catch (e) {
                console.debug('Error stopping audio element:', e);
            }
            this.currentAudioElement = null;
        }
        
        // Stop any AudioContext sources
        if (this.currentAudioSource) {
            try {
                this.currentAudioSource.stop();
                this.currentAudioSource.disconnect();
            } catch (e) {
                console.debug('Error stopping audio source:', e);
            }
            this.currentAudioSource = null;
        }
    }

    // NEW: Single audio playback method (no overlap)
    async playAudioSingle(bytes) {
        try {
            // Create blob with proper MIME type
            const audioBlob = new Blob([bytes], { type: 'audio/mpeg' });
            this.currentAudioUrl = URL.createObjectURL(audioBlob);
            
            console.log('🔊 Created single audio URL:', this.currentAudioUrl);
            
            // Create audio element with optimal settings
            this.currentAudioElement = new Audio(this.currentAudioUrl);
            
            // CRITICAL: Set proper playback properties
            this.currentAudioElement.playbackRate = this.audioSettings.playbackRate;
            this.currentAudioElement.volume = this.audioSettings.volume;
            this.currentAudioElement.preservesPitch = true;
            this.currentAudioElement.preload = 'auto';
            
            // Add event listeners
            return new Promise((resolve, reject) => {
                this.currentAudioElement.addEventListener('canplaythrough', async () => {
                    try {
                        console.log('🔊 Audio ready, starting playback...');
                        await this.currentAudioElement.play();
                        console.log('✅ Single audio playback started successfully');
                        resolve(true);
                    } catch (playError) {
                        console.error('❌ Play failed:', playError);
                        reject(playError);
                    }
                }, { once: true });
                
                this.currentAudioElement.addEventListener('ended', () => {
                    console.log('🔊 Audio finished playing');
                    this.cleanupCurrentAudio();
                    resolve(true);
                }, { once: true });
                
                this.currentAudioElement.addEventListener('error', (e) => {
                    console.error('🔊 Audio error:', e);
                    this.cleanupCurrentAudio();
                    reject(e);
                }, { once: true });
                
                // Start loading
                this.currentAudioElement.load();
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    if (this.currentAudioElement && this.currentAudioElement.readyState < 4) {
                        console.warn('Audio load timeout');
                        reject(new Error('Audio load timeout'));
                    }
                }, 10000);
            });
            
        } catch (error) {
            console.error('❌ Single audio playback failed:', error);
            return false;
        }
    }

    // NEW: Fallback audio playback
    async playAudioFallback(bytes) {
        console.log('🔄 Using fallback audio playback...');
        
        try {
            const mimeTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav'];
            
            for (const mimeType of mimeTypes) {
                try {
                    const audioBlob = new Blob([bytes], { type: mimeType });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    const audio = new Audio(audioUrl);
                    audio.playbackRate = 1.0;
                    audio.volume = 0.9;
                    
                    await audio.play();
                    console.log(`✅ Fallback playback successful with ${mimeType}`);
                    
                    audio.addEventListener('ended', () => {
                        URL.revokeObjectURL(audioUrl);
                    });
                    
                    return true;
                    
                } catch (e) {
                    console.debug(`Fallback failed with ${mimeType}:`, e);
                    continue;
                }
            }
            
            console.error('❌ All fallback methods failed');
            return false;
            
        } catch (error) {
            console.error('❌ Fallback audio error:', error);
            return false;
        }
    }

    // NEW: Cleanup current audio
    cleanupCurrentAudio() {
        if (this.currentAudioUrl) {
            URL.revokeObjectURL(this.currentAudioUrl);
            this.currentAudioUrl = null;
        }
        this.currentAudioElement = null;
        this.currentAudioSource = null;
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
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.error('Error stopping MediaRecorder:', error);
            }
        }
        
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
        this.stopCurrentAudio();
        
        this.sendMessage({
            type: 'end_call'
        });
        
        if (this.ws) {
            try {
                this.ws.close();
            } catch (error) {
                console.error('Error closing WebSocket:', error);
            }
        }
        
        this.updateStatus('Call ended');
        
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
            }
        };
    }

    // Test method for natural voice
    testNaturalVoice() {
        console.log('🧪 Testing natural voice playback...');
        
        if (this.isConnected) {
            this.sendMessage({
                type: 'text',
                text: 'Testing natural voice speed. This should sound normal, not fast.',
                timestamp: new Date().toISOString()
            });
        } else {
            console.error('Not connected - start a call first');
        }
    }

    // Set audio quality
    setAudioQuality(quality = 'natural') {
        const settings = {
            'natural': { 
                volume: 0.9, 
                playbackRate: 1.0,
                useCompressor: true 
            },
            'clear': { 
                volume: 0.9, 
                playbackRate: 1.0,
                useCompressor: true 
            },
            'loud': { 
                volume: 1.0, 
                playbackRate: 1.0,
                useCompressor: false 
            }
        };
        
        this.audioSettings = settings[quality] || settings['natural'];
        console.log('🔊 Audio quality set to:', quality, this.audioSettings);
    }
}

// Global functions for call interface
async function startCall() {
    console.log('🎯 Start call clicked');
    
    if (!callHandler) {
        callHandler = new CallHandler();
    }
    
    document.getElementById('start-call-btn').style.display = 'none';
    document.getElementById('mute-btn').style.display = 'inline-block';
    document.getElementById('pause-btn').style.display = 'inline-block';
    document.getElementById('transfer-btn').style.display = 'inline-block';
    document.getElementById('end-call-btn').style.display = 'inline-block';
    
    document.getElementById('call-status').textContent = 'Starting...';
    
    try {
        await callHandler.initialize();
        console.log('✅ Call handler initialized successfully');
    } catch (error) {
        console.error('❌ Failed to start call:', error);
        
        let errorMessage = 'Failed to start call';
        if (error.message.includes('Permission denied')) {
            errorMessage = 'Microphone access denied. Please allow microphone access and try again.';
            showMicrophoneInstructions();
        } else if (error.message.includes('WebSocket') || error.message.includes('timeout')) {
            errorMessage = 'Connection failed. Please check your server is running and try again.';
        } else {
            errorMessage = `Error: ${error.message}`;
        }
        
        document.getElementById('call-status').textContent = errorMessage;
        
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
    
    setTimeout(() => {
        if (instructions.parentElement) {
            instructions.remove();
        }
    }, 20000);
}

function checkCallHandler() {
    if (callHandler) {
        console.log('Call Handler Status:', callHandler.getSessionStats());
    } else {
        console.log('Call handler not initialized');
    }
}

// Test functions
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
            
            setTimeout(sendNextMessage, 5000);
        } else {
            console.log('✅ Test conversation completed');
        }
    }
    
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

// Add functions to global scope
window.sendTestMessage = sendTestMessage;
window.testConversation = testConversation;
window.checkWebSocketStatus = checkWebSocketStatus;
window.debugAudioPlayback = debugAudioPlayback;
window.checkCallHandler = checkCallHandler;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 Page loaded, call handler ready to initialize');
    
    setTimeout(() => {
        const status = document.getElementById('call-status');
        if (status && status.textContent === 'Ready to start') {
            status.textContent = 'Click "Start Call" to begin';
        }
    }, 1000);
});

// Add keyboard shortcuts
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

// Handle page unload
window.addEventListener('beforeunload', (event) => {
    if (callHandler && callHandler.isConnected) {
        callHandler.endCall();
    }
});

console.log('🧪 Test functions loaded. Available commands:');
console.log('- sendTestMessage("your message here")');
console.log('- testConversation()');
console.log('- checkWebSocketStatus()'); 
console.log('- debugAudioPlayback()');
console.log('- checkCallHandler()');