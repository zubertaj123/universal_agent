/**
 * Call Handler - WebSocket communication and UI management
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
    }

    async initialize() {
        // Generate session ID
        this.sessionId = this.generateSessionId();
        
        // Initialize audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Request microphone access
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Setup WebSocket connection
            this.connectWebSocket();
            
            // Setup audio recording
            this.setupAudioRecording();
            
            // Start duration timer
            this.startDurationTimer();
            
            // Update UI
            this.updateStatus('Connected');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.updateStatus('Microphone access denied');
        }
    }

    connectWebSocket() {
        const wsUrl = `ws://localhost:8000/ws/call/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateStatus('Connected');
            console.log('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error');
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateStatus('Disconnected');
            this.stopDurationTimer();
        };
    }

    setupAudioRecording() {
        this.mediaRecorder = new MediaRecorder(this.mediaStream);
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && !this.isPaused && !this.isMuted) {
                // Convert blob to array buffer
                event.data.arrayBuffer().then(buffer => {
                    const uint8Array = new Uint8Array(buffer);
                    const hexString = Array.from(uint8Array)
                        .map(b => b.toString(16).padStart(2, '0'))
                        .join('');
                    
                    // Send audio data over WebSocket
                    if (this.isConnected) {
                        this.ws.send(JSON.stringify({
                            type: 'audio',
                            data: hexString
                        }));
                    }
                });
            }
        };
        
        // Start recording in chunks
        this.mediaRecorder.start(100); // 100ms chunks
        
        // Setup audio visualization
        this.setupAudioVisualization();
    }

    setupAudioVisualization() {
        const canvas = document.getElementById('audio-canvas');
        const canvasCtx = canvas.getContext('2d');
        const analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        source.connect(analyser);
        analyser.fftSize = 256;
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            requestAnimationFrame(draw);
            
            analyser.getByteFrequencyData(dataArray);
            
            canvasCtx.fillStyle = 'rgb(249, 250, 251)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                
                canvasCtx.fillStyle = `rgb(37, 99, 235)`;
                canvasCtx.fillRect(x, canvas.height - barHeight / 2, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }

    handleWebSocketMessage(data) {
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
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    addTranscript(speaker, text) {
        const transcriptDiv = document.getElementById('transcript');
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
    }

    async playAudio(hexData) {
        // Convert hex string back to array buffer
        const bytes = new Uint8Array(hexData.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
        const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
        
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
    }

    updateStatus(status) {
        document.getElementById('call-status').textContent = status;
    }

    startDurationTimer() {
        this.startTime = Date.now();
        
        this.durationInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            document.getElementById('call-duration').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
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
        document.getElementById('mute-btn').textContent = this.isMuted ? '🔇 Unmute' : '🎤 Mute';
        
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'control',
                action: this.isMuted ? 'mute' : 'unmute'
            }));
        }
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        document.getElementById('pause-btn').textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
        
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'control',
                action: this.isPaused ? 'pause' : 'resume'
            }));
        }
    }

    transferToHuman() {
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'transfer',
                reason: 'User requested human agent'
            }));
            
            this.updateStatus('Transferring to human agent...');
        }
    }

    endCall() {
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'end_call'
            }));
        }
        
        // Stop recording
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
        }
        
        // Redirect to home
        setTimeout(() => {
            window.location.href = '/';
        }, 2000);
    }
}

// Initialize call handler when page loads
let callHandler;

document.addEventListener('DOMContentLoaded', () => {
    if (window.location.pathname === '/call') {
        callHandler = new CallHandler();
        callHandler.initialize();
    }
});

// Global functions for HTML onclick handlers
function toggleMute() {
    if (callHandler) {
        callHandler.toggleMute();
    }
}

function togglePause() {
    if (callHandler) {
        callHandler.togglePause();
    }
}

function transferToHuman() {
    if (callHandler) {
        callHandler.transferToHuman();
    }
}

function endCall() {
    if (callHandler) {
        callHandler.endCall();
    }
}