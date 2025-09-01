# Zhara AI Assistant

Zhara is an advanced AI assistant that combines large language models, speech recognition, text-to-speech, and lip sync capabilities for a natural interactive experience.

## Features
- Speech-to-text (STT) using Whisper
- Text-to-speech (TTS) using Coqui TTS
- Large Language Model integration with Ollama
- Lip sync generation with Rhubarb
- Modern web interface

## Installation

### Prerequisites
- Python 3.10 or higher
- At least 4GB RAM available
- Internet connection for downloading models
- Ollama installed and running (for LLM functionality)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YashVinchhi/zhara.git
   cd zhara
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Rhubarb Lip Sync (for lip sync generation):
   - Download from [Rhubarb Lip Sync releases](https://github.com/DanielSWolf/rhubarb-lip-sync/releases)
   - Extract and add the `rhubarb` executable to your system PATH

4. Install and start Ollama:
   - Follow the [Ollama installation guide](https://ollama.ai/download)
   - Pull a model (e.g., `ollama pull qwen2.5-coder:14b`)
   - Ensure Ollama is running on `http://localhost:11434`

5. Run the application:
   ```bash
   python zhara.py
   ```

6. Access the web interface at:
   ```
   http://localhost:8000
   ```
   
Note: To run with Docker or Docker Compose, see [DOCKER.md](./DOCKER.md) for detailed instructions.

 

### Configuration

You can customize Zhara's behavior using environment variables:

```bash
export MAX_AUDIO_DURATION=600
export MAX_TEXT_LENGTH=2000
export MAX_FILE_AGE=24
export OLLAMA_HOST=http://localhost:11434
python zhara.py
```

Available environment variables:
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 300)
- `MAX_TEXT_LENGTH`: Maximum text length for TTS (default: 1000)
- `MAX_FILE_AGE`: Maximum age of files in hours (default: 24)
- `OLLAMA_HOST`: Ollama server host (default: http://localhost:11434)

## API Reference

### Endpoints

1. Text Chat Endpoint
   ```
   POST /ask
   Content-Type: application/json

   {
     "text": "your question",
     "model": "default"
   }
   ```
   Returns:
   ```json
   {
     "reply": "AI response text",
     "audio_url": "/audio/response_xyz.wav",
     "viseme_url": "/viseme/viseme_xyz.json"
   }
   ```

2. Speech-to-Text Endpoint
   ```
   POST /stt
   Content-Type: multipart/form-data

   file: <audio_file>
   ```
   Returns:
   ```json
   {
     "text": "transcribed text"
   }
   ```

3. Health Check
   ```
   GET /health
   ```
   Returns:
   ```json
   {
     "status": "healthy",
     "timestamp": "2025-08-12T14:30:00.000Z"
   }
   ```

### Example Usage

1. Using curl:
   ```bash
   # Send a text query
   curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, Zhara!"}'

   # Send audio for transcription
   curl -X POST "http://localhost:8000/stt" \
     -F "file=@your_audio.wav"

   # Check health
   curl "http://localhost:8000/health"
   ```

2. Using Python:
   ```python
   import requests

   # Send a text query
   response = requests.post(
       "http://localhost:8000/ask",
       json={"text": "Hello, Zhara!"}
   )
   print(response.json())

   # Send audio for transcription
   with open("your_audio.wav", "rb") as f:
       response = requests.post(
           "http://localhost:8000/stt",
           files={"file": f}
       )
   print(response.json())
   ```

## Troubleshooting

### Common Issues

1. "Connection refused" when accessing Ollama:
   - Check if Ollama is running: `ollama list` or `ps aux | grep ollama`
   - Verify Ollama is accessible at the configured host
   - Verify OLLAMA_HOST environment variable

2. Audio generation fails:
   - Check available disk space
   - Verify storage directory permissions
   - Check if TTS models are properly installed

3. Application won't start:
   - Check for port conflicts (port 8000)
   - Verify all Python dependencies are installed
   - Check system resources and available memory

4. Rhubarb lip sync errors:
   - Ensure Rhubarb is installed and in your PATH
   - Check audio file format compatibility

### Getting Help

If you encounter issues:
1. Check the application logs in your terminal
2. Verify all prerequisites are met
3. Create an issue on GitHub with:
   - Error messages
   - Application logs
   - System information

## Notes
- Make sure Ollama is running locally if you want to use the LLM endpoint.
- The static files are served from the project directory root.

## License
Copyright © 2025 Yash Vinchhi  
All rights reserved.  

This software is proprietary.  
Unauthorized copying, modification, distribution, or use of this code,  
via any medium, is strictly prohibited without explicit permission.
