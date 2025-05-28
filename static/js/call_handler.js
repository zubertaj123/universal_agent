/**
 * Fixed Call Handler - Complete audio processing and WebSocket communication
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
            // Fix the WebSocket URL construction
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const host = window.location.hostname === "0.0.0.0" ? "localhost" : window.location.hostname;
            const port = window.location.port;
            const wsUrl = `${protocol}//${host}:${port}/ws/call/${this.sessionId}`;
            
            console.log('Connecting to WebSocket:', wsUrl);
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.connectionReady = true;
                this.updateStatus('Connected');
                console.log('✅ WebSocket connected successfully');
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
                reject(new Error('WebSocket connection failed'));
            };
            
            this.ws.onclose = (event) => {
                this.isConnected = false;
                this.connectionReady = false;
                console.log('WebSocket closed:', event.code, event.reason);
                this.updateStatus('Disconnected');
                this.stopDurationTimer();
            };
            
            // Set connection timeout
            setTimeout(() => {
                if (!this.connectionReady) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 10000);
        });
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
                
                // Send via WebSocket with error handling
                if (this.ws && this.ws.readyState === WebSocket.OPEN && this.connectionReady) {
                    try {
                        this.ws.send(JSON.stringify({
                            type: 'audio',
                            data: hexString,
                            format: 'pcm16',
                            sampleRate: this.audioContext.sampleRate,
                            channels: 1
                        }));
                        
                        if (this.debugMode && Math.random() < 0.01) {
                            console.log('📤 Sent audio chunk:', {
                                size: pcmData.length,
                                sampleRate: this.audioContext.sampleRate,
                                hasAudio: this.hasSignificantAudio(audioData)
                            });
                        }
                    } catch (error) {
                        console.error('Error sending audio data:', error);
                    }
                } else if (this.debugMode && Math.random() < 0.001) {
                    console.warn('WebSocket not ready, skipping audio chunk');
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
                        
                        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.connectionReady) {
                            this.ws.send(JSON.stringify({
                                type: 'audio',
                                data: hexString,
                                format: this.mediaRecorder.mimeType,
                                fallback: true
                            }));
                        }
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
            switch (data.type) {
                case 'transcript':
                    this.addTranscript(data.speaker, data.text);
                    break;
                    
                case 'audio':
                    this.playAudio(data.data);
                    break;
                    
                case 'status':
                    this.updateStatus(data.message);
                    break;
                    
                case 'error':
                    console.error('Server error:', data.message);
                    this.updateStatus('Error: ' + data.message);
                    break;
                    
                case 'connected':
                    console.log('Server confirmed connection:', data);
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type, data);
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
        }
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
            
            // Limit transcript entries to prevent memory issues
            const entries = transcriptDiv.querySelectorAll('.transcript-entry');
            if (entries.length > 50) {
                entries[0].remove();
            }
            
        } catch (error) {
            console.error('Error adding transcript:', error);
        }
    }

    async playAudio(hexData) {
        try {
            if (!hexData || typeof hexData !== 'string') {
                console.warn('Invalid audio data received');
                return;
            }
            
            // Convert hex string back to array buffer
            const bytes = new Uint8Array(
                hexData.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
            );
            
            // Decode audio data
            const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer.slice());
            
            // Create and play audio source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
        } catch (error) {
            console.warn('Audio playback error (this is normal for TTS):', error.message);
            // TTS audio playback errors are expected due to format differences
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
        return 'call-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        const muteBtn = document.getElementById('mute-btn');
        if (muteBtn) {
            muteBtn.textContent = this.isMuted ? '🔇 Unmute' : '🎤 Mute';
        }
        
        if (this.isConnected && this.connectionReady) {
            this.ws.send(JSON.stringify({
                type: 'control',
                action: this.isMuted ? 'mute' : 'unmute'
            }));
        }
        
        console.log('🔇 Mute toggled:', this.isMuted);
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        const pauseBtn = document.getElementById('pause-btn');
        if (pauseBtn) {
            pauseBtn.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
        }
        
        if (this.isConnected && this.connectionReady) {
            this.ws.send(JSON.stringify({
                type: 'control',
                action: this.isPaused ? 'pause' : 'resume'
            }));
        }
        
        console.log('⏸️ Pause toggled:', this.isPaused);
    }

    transferToHuman() {
        if (this.isConnected && this.connectionReady) {
            this.ws.send(JSON.stringify({
                type: 'transfer',
                reason: 'User requested human agent'
            }));
            
            this.updateStatus('Transferring to human agent...');
        }
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
        if (this.isConnected && this.connectionReady) {
            try {
                this.ws.send(JSON.stringify({
                    type: 'end_call'
                }));
            } catch (error) {
                console.error('Error sending end call message:', error);
            }
        }
        
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

    // Get current session statistics
    getSessionStats() {
        return {
            sessionId: this.sessionId,
            isConnected: this.isConnected,
            isRecording: this.isRecording,
            isMuted: this.isMuted,
            isPaused: this.isPaused,
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
    console.log('Protocol:', window.location.protocol);
    console.log('Host:', window.location.host);
    
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
        } else if (error.message.includes('WebSocket')) {
            errorMessage = 'Connection failed. Please check your internet connection and try again.';
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