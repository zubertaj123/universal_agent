# Local Call Center AI

A fully functional, locally-running AI-powered call center solution with voice capabilities, built using modern Python frameworks and open-source models.

## Features

- 🎙️ **Real-time Voice Communication**: Speech-to-text and text-to-speech using Whisper and Edge-TTS
- 🤖 **Intelligent Agent**: LangGraph-based agent with tool calling capabilities
- 💬 **Multi-modal Interface**: Support for both voice calls and text chat
- 🔍 **RAG System**: Knowledge base with semantic search using ChromaDB
- 📊 **Call Analytics**: Track and analyze call metrics and transcripts
- 🌐 **Multi-language Support**: 40+ languages with automatic detection
- 💾 **Local First**: Everything runs on your machine, no cloud dependencies
- 🔒 **Privacy Focused**: Your data stays on your infrastructure

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (for audio processing)
- Redis (optional, for caching)
- CUDA-capable GPU (optional, for faster STT)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/local-call-center-ai.git
   cd local-call-center-ai

2. Create project structure
bashchmod +x create_project_structure.sh
./create_project_structure.sh

3. Set up environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

4. Configure environment variables
bashcp .env.example .env
# Edit .env with your API keys (OpenAI or Anthropic)

5. Initialize database
bashpython scripts/populate_db.py

6. Run the application
bashpython -m app.main

7. Access the interface

Web UI: http://localhost:8000
API Docs: http://localhost:8000/docs



Docker Deployment
bashdocker-compose up -d
Architecture
The system uses a modular architecture:

FastAPI: REST API and WebSocket endpoints
LangGraph: Agent orchestration and state management
Whisper: Speech-to-text (via faster-whisper)
Edge-TTS: Free, high-quality text-to-speech
ChromaDB: Vector database for RAG
SQLAlchemy: Relational database ORM
Redis: Caching layer (optional)

Usage
Voice Call

Click "Start Voice Call" on the main page
Allow microphone access when prompted
Speak naturally with the AI agent
The agent can:

Create and track claims
Schedule appointments
Answer questions
Transfer to human agents



Adding Knowledge
bashpython scripts/ingest_documents.py /path/to/documents
Testing Voice
bashpython scripts/test_voice.py
Configuration
Edit config/settings.yaml for:

Speech recognition settings
TTS voice selection
Agent behavior
Audio processing parameters

API Endpoints

GET /health: Health check
GET /api/calls: List all calls
POST /api/calls: Create new call
WS /ws/call/{session_id}: WebSocket for real-time communication

Performance Optimization

GPU Acceleration: Automatically uses CUDA if available
Caching: TTS responses are cached for common phrases
Streaming: Audio is streamed in real-time
VAD: Voice Activity Detection reduces processing

Contributing

Fork the repository
Create a feature branch
Make your changes
Submit a pull request

License
MIT License - see LICENSE file for details
Acknowledgments

OpenAI Whisper for speech recognition
Microsoft Edge-TTS for speech synthesis
LangChain/LangGraph for agent framework
The open-source community


### `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
data/
*.db
*.sqlite
*.log

# Audio files
*.mp3
*.wav
*.m4a

# Model files
*.bin
*.pt
*.onnx

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/_build/