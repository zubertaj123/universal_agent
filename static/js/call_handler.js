/**
 * Fixed Call Handler - Improved audio processing and WebSocket communication
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
        this.chunkSize = 1024; // Process in 1KB chunks
        this.audioWorkletNode = null;
        this.isRecording = false;
        this.debugMode = false;
    }

    async initialize() {
        // Generate session ID
        this.sessionId = this.generateSessionId();
        
        // Initialize audio context with target sample rate
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.targetSampleRate
            });
        } catch (error) {
            console.warn('Could not set target sample rate, using default');
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Request microphone access with specific constraints
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
                channels: this.mediaStream.getAudioTracks()[0].getSettings()
            });
            
            // Setup WebSocket connection first
            await this.connectWebSocket();
            
            // Setup improved audio processing
            await this.setupImprovedAudioProcessing();
            
            // Start duration timer
            this.startDurationTimer();
            
            // Update UI
            this.updateStatus('Connected');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.updateStatus('Microphone access denied');
            throw error;
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
            const wsUrl = `${wsScheme}://${window.location.host}/ws/call/${this.sessionId}`;
            
            console.log('Connecting to WebSocket:', wsUrl);
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.connectionReady = true;
                this.updateStatus('Connected');
                console.log('WebSocket connected successfully');
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
                console.error('WebSocket error:', error);
                this.updateStatus('Connection error');
                reject(error);
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

    async setupImprovedAudioProcessing() {
        try {
            // Create audio source from media stream
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create script processor for real-time audio processing
            // Note: ScriptProcessorNode is deprecated but widely supported
            // In production, consider using AudioWorklet
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
            
            console.log('Improved audio processing setup complete');
            
        } catch (error) {
            console.error('Error setting up audio processing:', error);
            // Fallback to MediaRecorder if AudioWorklet fails
            this.setupFallbackAudioRecording();
        }
    }

    processAudioChunk(audioData) {
        try {
            // Convert Float32Array to 16-bit PCM
            const pcmData = this.convertToPCM16(audioData);
            
            // Only send if we have a reasonable amount of audio data
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
                            console.log('Sent audio chunk:', {
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

    setupFallbackAudioRecording() {
        console.log('Setting up fallback MediaRecorder...');
        
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
            
            // Start recording with small time slices for better real-time performance
            this.mediaRecorder.start(100); // 100ms chunks
            this.isRecording = true;
            
        } catch (error) {
            console.error('Error setting up MediaRecorder fallback:', error);
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
        console.log('Status update:', status);
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
        
        console.log('Mute toggled:', this.isMuted);
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
        
        console.log('Pause toggled:', this.isPaused);
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
        console.log('Ending call...');
        
        // Stop recording
        this.isRecording = false;
        
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
        console.log('Debug mode enabled');
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