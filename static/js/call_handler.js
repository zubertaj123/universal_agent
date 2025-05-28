/**
 * Fixed Call Handler - WebSocket Connection Issues Resolved
 */

// Global call handler instance
let callHandler = null;

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
            
            // Step 3: Setup WebSocket connection (FIXED)
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
            // FIXED: Simplified and more reliable WebSocket URL construction
            const wsUrl = this.buildWebSocketUrl();
            
            console.log('Connecting to WebSocket:', wsUrl);
            console.log('Location details:', {
                protocol: window.location.protocol,
                hostname: window.location.hostname, 
                host: window.location.host,
                port: window.location.port
            });
            
            this.ws = new WebSocket(wsUrl);
            
            // Set binary type for audio data
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
                    console.log('Raw message:', event.data);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                console.log('WebSocket state:', this.ws?.readyState);
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
                
                // Handle different close codes
                if (event.code === 1006) {
                    console.error('WebSocket closed abnormally - possible server issue');
                } else if (event.code === 1011) {
                    console.error('WebSocket closed due to server error');
                }
                
                this.updateStatus('Disconnected');
                this.stopDurationTimer();
            };
            
            // FIXED: Increased timeout and better error handling
            const connectionTimeout = setTimeout(() => {
                if (!this.connectionReady) {
                    console.error('WebSocket connection timeout');
                    if (this.ws) {
                        this.ws.close();
                    }
                    reject(new Error('WebSocket connection timeout - server may be unreachable'));
                }
            }, 15000); // Increased to 15 seconds
            
            // Clear timeout on successful connection
            this.ws.addEventListener('open', () => {
                clearTimeout(connectionTimeout);
            });
        });
    }

    // FIXED: More reliable WebSocket URL construction
    buildWebSocketUrl() {
        // Use the current page's protocol and host
        const isSecure = window.location.protocol === 'https:';
        const wsProtocol = isSecure ? 'wss:' : 'ws:';
        
        // Use window.location.host which includes port automatically
        const host = window.location.host;
        
        // Construct the full WebSocket URL
        const wsUrl = `${wsProtocol}//${host}/ws/call/${this.sessionId}`;
        
        return wsUrl;
    }

    // FIXED: Better message sending with error handling
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
            // Create audio source from media stream
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create script processor for real-time audio processing
            const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (event) => {
                if (!this.isRecording || this.isPaused || this.isMuted) {
                    return;
                }
                
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Process audio in chunks
                this.processAudioChunk(inputData);
            };
            
            // Connect audio nodes
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            // Store references
            this.audioSource = source;
            this.audioProcessor = processor;
            
            // Start recording
            this.isRecording = true;
            
            // Setup audio visualization
            this.setupAudioVisualization(source);
            
            console.log('✅ Audio processing setup complete');
            
        } catch (error) {
            console.error('❌ Error setting up audio processing:', error);
            // Fallback to MediaRecorder
            await this.setupFallbackAudioRecording();
        }
    }

    processAudioChunk(audioData) {
        try {
            // Convert Float32Array to 16-bit PCM
            const pcmData = this.convertToPCM16(audioData);
            
            // Only send if we have significant audio data
            if (pcmData.length > 0 && this.hasSignificantAudio(audioData)) {
                // Convert to hex string for transmission
                const hexString = Array.from(pcmData)
                    .map(b => b.toString(16).padStart(2, '0'))
                    .join('');
                
                // FIXED: Use the new sendMessage method
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
                // Convert float32 [-1, 1] to int16 [-32768, 32767]
                let sample = Math.max(-1, Math.min(1, float32Array[i]));
                sample = sample * 32767;
                view.setInt16(i * 2, sample, true); // true = little endian
            }
            
            return new Uint8Array(buffer);
        } catch (error) {
            console.error('Error converting to PCM16:', error);
            return new Uint8Array(0);
        }
    }

    hasSignificantAudio(audioData) {
        // Check if audio has significant energy (not just silence/noise)
        const rms = Math.sqrt(audioData.reduce((sum, sample) => sum + sample * sample, 0) / audioData.length);
        return rms > 0.001; // Threshold for significant audio
    }

    async setupFallbackAudioRecording() {
        console.log('🔄 Setting up fallback MediaRecorder...');
        
        try {
            // Configure MediaRecorder with optimal settings
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            };
            
            // Fallback mime types if opus not supported
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'audio/mp4';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        delete options.mimeType; // Use default
                    }
                }
            }
            
            this.mediaRecorder = new MediaRecorder(this.mediaStream, options);
            
            console.log('MediaRecorder configured with:', {
                mimeType: this.mediaRecorder.mimeType,
                state: this.mediaRecorder.state
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && !this.isPaused && !this.isMuted) {
                    // Convert blob to array buffer and then to hex
                    event.data.arrayBuffer().then(buffer => {
                        const uint8Array = new Uint8Array(buffer);
                        const hexString = Array.from(uint8Array)
                            .map(b => b.toString(16).padStart(2, '0'))
                            .join('');
                        
                        // FIXED: Use the new sendMessage method
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
            
            // Start recording with small time slices
            this.mediaRecorder.start(100); // 100ms chunks
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
            
            // Configure analyser
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            const draw = () => {
                if (!this.isRecording) return;
                
                requestAnimationFrame(draw);
                
                analyser.getByteFrequencyData(dataArray);
                
                // Clear canvas
                canvasCtx.fillStyle = 'rgb(249, 250, 251)';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw frequency bars
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    
                    // Color based on frequency and amplitude
                    const hue = (i / bufferLength) * 120; // Blue to green
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

    handleWebSocketMessage(data) {
        try {
            console.log('📨 Received WebSocket message:', data.type, data);
            
            switch (data.type) {
                case 'transcript':
                    console.log('📝 Adding transcript:', data.speaker, data.text);
                    this.addTranscript(data.speaker, data.text);
                    break;
                    
                case 'audio':
                    console.log('🔊 Playing audio:', data.data ? data.data.length + ' chars' : 'no data');
                    this.playAudio(data.data);
                    break;
                    
                case 'status':
                    console.log('📊 Status update:', data.message);
                    this.updateStatus(data.message);
                    break;
                    
                case 'error':
                    console.error('❌ Server error:', data.message);
                    this.updateStatus('Error: ' + data.message);
                    break;
                    
                case 'connected':
                    console.log('✅ Server confirmed connection:', data);
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

    addTranscript(speaker, text) {
        try {
            console.log('📝 Adding transcript entry:', speaker, text);
            
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
            
            console.log('✅ Transcript entry added successfully');
            
            // Limit transcript entries to prevent memory issues
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
            
            // FIXED: Use blob approach for Edge-TTS MP3 data
            try {
                // Create blob with proper MIME type for MP3
                const audioBlob = new Blob([bytes], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                console.log('🔊 Created audio blob URL:', audioUrl);
                
                // Create audio element for proper MP3 playback
                const audioElement = new Audio(audioUrl);
                
                // Set playback properties for natural speech
                audioElement.playbackRate = 1.0;  // Normal speed
                audioElement.volume = 1.0;        // Full volume
                audioElement.preservesPitch = true; // Maintain natural pitch
                
                // Add event listeners for debugging
                audioElement.addEventListener('loadstart', () => {
                    console.log('🔊 Audio loading started');
                });
                
                audioElement.addEventListener('canplay', () => {
                    console.log('🔊 Audio can start playing');
                });
                
                audioElement.addEventListener('play', () => {
                    console.log('🔊 Audio playback started');
                });
                
                audioElement.addEventListener('ended', () => {
                    console.log('🔊 Audio playback finished');
                    // Clean up URL
                    URL.revokeObjectURL(audioUrl);
                });
                
                audioElement.addEventListener('error', (e) => {
                    console.error('🔊 Audio playback error:', e);
                    URL.revokeObjectURL(audioUrl);
                });
                
                // Play the audio
                try {
                    await audioElement.play();
                    console.log('✅ Audio played successfully with natural speech');
                } catch (playError) {
                    console.error('❌ Audio play failed:', playError);
                    URL.revokeObjectURL(audioUrl);
                    
                    // Fallback: try with different MIME type
                    await this.tryAlternativeAudioPlayback(bytes);
                }
                
            } catch (error) {
                console.error('❌ Audio blob creation failed:', error);
                // Try alternative playback method
                await this.tryAlternativeAudioPlayback(bytes);
            }
            
        } catch (error) {
            console.error('❌ Audio playback error:', error);
        }
    }

    async tryAlternativeAudioPlayback(bytes) {
        console.log('🔄 Trying alternative audio playback methods...');
        
        // Method 1: Try as WebM
        try {
            const webmBlob = new Blob([bytes], { type: 'audio/webm' });
            const webmUrl = URL.createObjectURL(webmBlob);
            const webmAudio = new Audio(webmUrl);
            
            await webmAudio.play();
            console.log('✅ Alternative playback (WebM) successful');
            
            webmAudio.addEventListener('ended', () => {
                URL.revokeObjectURL(webmUrl);
            });
            
            return;
        } catch (webmError) {
            console.log('⚠️ WebM playback failed:', webmError.message);
        }
        
        // Method 2: Try as WAV
        try {
            const wavBlob = new Blob([bytes], { type: 'audio/wav' });
            const wavUrl = URL.createObjectURL(wavBlob);
            const wavAudio = new Audio(wavUrl);
            
            await wavAudio.play();
            console.log('✅ Alternative playback (WAV) successful');
            
            wavAudio.addEventListener('ended', () => {
                URL.revokeObjectURL(wavUrl);
            });
            
            return;
        } catch (wavError) {
            console.log('⚠️ WAV playback failed:', wavError.message);
        }
        
        // Method 3: Try with AudioContext (last resort)
        try {
            await this.playAudioWithContext(bytes);
        } catch (contextError) {
            console.error('❌ All audio playback methods failed:', contextError);
        }
    }

    async playAudioWithContext(bytes) {
        console.log('🔄 Trying AudioContext playback...');
        
        if (!this.audioContext) {
            console.error('❌ AudioContext not available');
            return;
        }
        
        try {
            const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer.slice());
            
            // Create source with proper settings
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            
            // Add gain node to control volume
            const gainNode = this.audioContext.createGain();
            gainNode.gain.value = 1.0;
            
            // Connect: source -> gain -> destination
            source.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            // Set playback rate to normal (not fast)
            source.playbackRate.value = 1.0;
            
            source.start();
            console.log('✅ AudioContext playback started');
            
        } catch (error) {
            console.error('❌ AudioContext decode failed:', error);
            throw error;
        }
    }

    // Add audio quality control methods
    setAudioQuality(quality = 'normal') {
        this.audioQuality = quality;
        
        const settings = {
            'low': { volume: 0.8, playbackRate: 1.0 },
            'normal': { volume: 1.0, playbackRate: 1.0 },
            'high': { volume: 1.0, playbackRate: 0.95 } // Slightly slower for clarity
        };
        
        this.audioSettings = settings[quality] || settings['normal'];
        console.log('🔊 Audio quality set to:', quality, this.audioSettings);
    }

    // Test audio playback with a known good file
    async testAudioPlayback() {
        console.log('🧪 Testing audio playback...');
        
        // Create a simple test tone
        const sampleRate = 44100;
        const duration = 1; // 1 second
        const frequency = 440; // A note
        
        const audioBuffer = this.audioContext.createBuffer(1, sampleRate * duration, sampleRate);
        const channelData = audioBuffer.getChannelData(0);
        
        // Generate sine wave
        for (let i = 0; i < channelData.length; i++) {
            channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.1;
        }
        
        // Play the test tone
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
        
        console.log('🔊 Test tone should be playing (440Hz for 1 second)');
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

    // FIXED: Simpler session ID generation
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
        
        // Stop recording
        this.isRecording = false;
        
        // Stop MediaRecorder if active
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.error('Error stopping MediaRecorder:', error);
            }
        }
        
        // Disconnect audio nodes
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
        
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (error) {
                    console.error('Error stopping media track:', error);
                }
            });
        }
        
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
        
        // Redirect to home after a delay
        setTimeout(() => {
            window.location.href = '/';
        }, 2000);
    }

    // Debug method to enable verbose logging
    enableDebugMode() {
        this.debugMode = true;
        console.log('🐛 Debug mode enabled');
    }

    // FIXED: More comprehensive session statistics
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
}

// Global functions for call interface
async function startCall() {
    console.log('🎯 Start call clicked');
    console.log('Current location:', window.location.href);
    
    // Initialize call handler if not already done
    if (!callHandler) {
        console.log('Creating new CallHandler instance...');
        callHandler = new CallHandler();
    }
    
    // Hide start button, show other controls
    document.getElementById('start-call-btn').style.display = 'none';
    document.getElementById('mute-btn').style.display = 'inline-block';
    document.getElementById('pause-btn').style.display = 'inline-block';
    document.getElementById('transfer-btn').style.display = 'inline-block';
    document.getElementById('end-call-btn').style.display = 'inline-block';
    
    // Update status
    document.getElementById('call-status').textContent = 'Starting...';
    
    try {
        console.log('Initializing call handler...');
        await callHandler.initialize();
        console.log('✅ Call handler initialized successfully');
    } catch (error) {
        console.error('❌ Failed to start call:', error);
        
        // Show detailed error message to user
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
        
        // Show start button again on error
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
        // Fallback - just redirect to home
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
        • Make sure you're accessing via https:// if available<br>
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

// Debug function to check call handler status
function checkCallHandler() {
    if (callHandler) {
        console.log('Call Handler Status:', callHandler.getSessionStats());
    } else {
        console.log('Call handler not initialized');
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 Page loaded, call handler ready to initialize');
    
    // Show initial instructions
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
        // Try to end call gracefully
        callHandler.endCall();
    }
});

// Test WebSocket connection function for debugging
async function testWebSocketConnection() {
    console.log('🧪 Testing WebSocket connection...');
    
    const wsUrl = new CallHandler().buildWebSocketUrl();
    console.log('Testing URL:', wsUrl);
    
    try {
        const testWs = new WebSocket(wsUrl);
        
        testWs.onopen = () => {
            console.log('✅ Test WebSocket connection successful');
            testWs.close();
        };
        
        testWs.onerror = (error) => {
            console.error('❌ Test WebSocket connection failed:', error);
        };
        
        testWs.onclose = (event) => {
            console.log('Test WebSocket closed:', event.code, event.reason);
        };
        
    } catch (error) {
        console.error('❌ Failed to create test WebSocket:', error);
    }
}


// Add these test functions to your call_handler.js or call them from browser console

// Test function to send a text message
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

// Test conversation flow
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
            
            // Send next message after 5 seconds
            setTimeout(sendNextMessage, 5000);
        } else {
            console.log('✅ Test conversation completed');
        }
    }
    
    // Start sending messages after 2 seconds
    setTimeout(sendNextMessage, 2000);
}

// Debug function to check WebSocket status
function checkWebSocketStatus() {
    if (callHandler) {
        const stats = callHandler.getSessionStats();
        console.log('📊 WebSocket Status:', {
            connected: stats.isConnected,
            readyState: stats.websocket?.readyState,
            url: stats.websocket?.url,
            sessionId: stats.sessionId
        });
        
        // WebSocket ready states
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

// Function to manually trigger agent response (for testing)
function triggerAgentResponse() {
    if (callHandler && callHandler.isConnected) {
        console.log('🤖 Triggering manual agent response...');
        
        callHandler.sendMessage({
            type: 'test',
            action: 'trigger_response',
            message: 'Manual trigger for testing'
        });
    }
}

// Add to window for easy access in console
window.sendTestMessage = sendTestMessage;
window.testConversation = testConversation;
window.checkWebSocketStatus = checkWebSocketStatus;
window.triggerAgentResponse = triggerAgentResponse;

console.log('🧪 Test functions loaded. Available commands:');
console.log('- sendTestMessage("your message here")');
console.log('- testConversation()');
console.log('- checkWebSocketStatus()'); 
console.log('- triggerAgentResponse()');


// Add these debug functions to your call_handler.js

// Debug function to inspect WebSocket messages
function enableWebSocketDebug() {
    if (callHandler && callHandler.ws) {
        const originalOnMessage = callHandler.ws.onmessage;
        
        callHandler.ws.onmessage = (event) => {
            console.log('🐛 Raw WebSocket message received:', event.data);
            
            try {
                const data = JSON.parse(event.data);
                console.log('🐛 Parsed WebSocket data:', data);
                
                // Check specifically for transcript messages
                if (data.type === 'transcript') {
                    console.log('🐛 Transcript message details:', {
                        speaker: data.speaker,
                        text: data.text,
                        timestamp: data.timestamp
                    });
                }
                
                // Check for audio messages  
                if (data.type === 'audio') {
                    console.log('🐛 Audio message details:', {
                        dataLength: data.data ? data.data.length : 'no data',
                        hasData: !!data.data
                    });
                }
                
            } catch (error) {
                console.error('🐛 Error parsing WebSocket message:', error);
            }
            
            // Call original handler
            if (originalOnMessage) {
                originalOnMessage.call(callHandler.ws, event);
            }
        };
        
        console.log('🐛 WebSocket debug mode enabled');
    } else {
        console.error('❌ Call handler or WebSocket not available');
    }
}

// Function to manually check transcript div
function checkTranscriptDiv() {
    const transcriptDiv = document.getElementById('transcript');
    console.log('🐛 Transcript div check:', {
        exists: !!transcriptDiv,
        children: transcriptDiv ? transcriptDiv.children.length : 'N/A',
        innerHTML: transcriptDiv ? transcriptDiv.innerHTML : 'N/A'
    });
    
    if (transcriptDiv) {
        const entries = transcriptDiv.querySelectorAll('.transcript-entry');
        console.log('🐛 Transcript entries:', entries.length);
        
        entries.forEach((entry, index) => {
            const speaker = entry.querySelector('.speaker');
            const text = entry.querySelector('.text');
            console.log(`🐛 Entry ${index}:`, {
                speaker: speaker ? speaker.textContent : 'no speaker',
                text: text ? text.textContent : 'no text'
            });
        });
    }
}

// Function to manually add a test transcript entry
function addTestTranscript() {
    if (callHandler) {
        callHandler.addTranscript('agent', 'This is a test response from the agent to verify transcript display is working.');
        console.log('✅ Test transcript entry added');
    } else {
        console.error('❌ Call handler not available');
    }
}

// Function to send a simple test and wait for response
async function sendTestAndWait() {
    console.log('🧪 Sending test message and waiting for response...');
    
    if (!callHandler || !callHandler.isConnected) {
        console.error('❌ Not connected');
        return;
    }
    
    // Clear console for clean output
    console.clear();
    console.log('🧪 Test started - sending message...');
    
    // Enable debug mode
    enableWebSocketDebug();
    
    // Send test message
    sendTestMessage("Hello, I need help with my insurance claim");
    
    // Wait and check results
    setTimeout(() => {
        console.log('🧪 Checking results after 3 seconds...');
        checkTranscriptDiv();
    }, 3000);
}

// Add functions to global scope
window.enableWebSocketDebug = enableWebSocketDebug;
window.checkTranscriptDiv = checkTranscriptDiv;
window.addTestTranscript = addTestTranscript;
window.sendTestAndWait = sendTestAndWait;

console.log('🐛 Debug functions loaded:');
console.log('- enableWebSocketDebug() - See raw WebSocket messages');
console.log('- checkTranscriptDiv() - Check transcript DOM state');
console.log('- addTestTranscript() - Add a test transcript entry');
console.log('- sendTestAndWait() - Send test message and debug response');


// Additional debugging functions for audio
function debugAudioPlayback() {
    console.log('🐛 Audio Debug Information:');
    
    if (callHandler) {
        console.log('AudioContext state:', callHandler.audioContext?.state);
        console.log('AudioContext sample rate:', callHandler.audioContext?.sampleRate);
        
        // Test if browser supports different audio formats
        const audio = new Audio();
        console.log('Browser audio support:');
        console.log('- MP3:', audio.canPlayType('audio/mpeg'));
        console.log('- WAV:', audio.canPlayType('audio/wav'));
        console.log('- WebM:', audio.canPlayType('audio/webm'));
        console.log('- OGG:', audio.canPlayType('audio/ogg'));
        
        // Test audio playback
        callHandler.testAudioPlayback();
    } else {
        console.error('❌ Call handler not available');
    }
}

// Function to adjust TTS playback speed
function setTTSSpeed(speed = 1.0) {
    if (callHandler) {
        callHandler.ttsPlaybackRate = speed;
        console.log('🔊 TTS playback rate set to:', speed);
        console.log('(1.0 = normal, 0.8 = slower, 1.2 = faster)');
    }
}

// Add to global scope
window.debugAudioPlayback = debugAudioPlayback;
window.setTTSSpeed = setTTSSpeed;

console.log('🔊 Audio fix loaded. New functions:');
console.log('- debugAudioPlayback() - Check audio capabilities');
console.log('- setTTSSpeed(0.8) - Slow down TTS (0.5-2.0)');