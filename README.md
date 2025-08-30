# Zhara AI Assistant

Zhara is an advanced AI assistant that combines large language models, speech recognition, text-to-speech, and lip sync capabilities for a natural interactive experience.

## Features
- Speech-to-text (STT) using Whisper
- Text-to-speech (TTS) using Coqui TTS
- Large Language Model integration with Ollama
- Lip sync generation with Rhubarb
- Modern web interface
- Docker support for easy deployment

## Quick Start with Docker (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- Internet connection for downloading models

### Method 1: Using Docker Compose (Easiest)

1. Clone the repository:
   ```bash
   git clone https://github.com/YashVinchhi/zhara.git
   cd zhara
   ```

2. Start both Zhara and Ollama using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access the web interface at:
   ```
   http://localhost:8000
   ```

### Method 2: Using Docker Manually

1. Start Ollama with the required model:
   ```bash
   # Start Ollama container
   docker run -d --name ollama -p 11434:11434 ollama/ollama
   
   # Pull and run the model
   docker exec -it ollama ollama run qwen2:0.5b
   ```

2. Run Zhara container:
   ```bash
   docker run -d \
     --name zhara \
     -p 8000:8000 \
     --network host \
     ghcr.io/yashvinchhi/zhara:latest
   ```

3. Access the web interface at:
   ```
   http://localhost:8000
   ```

### Docker Configuration Options

You can customize Zhara's behavior using environment variables:

```bash
docker run -d \
  --name zhara \
  -p 8000:8000 \
  -e MAX_AUDIO_DURATION=600 \
  -e MAX_TEXT_LENGTH=2000 \
  -e MAX_FILE_AGE=24 \
  -e OLLAMA_HOST=http://localhost:11434 \
  ghcr.io/yashvinchhi/zhara:latest
```

Available environment variables:
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 300)
- `MAX_TEXT_LENGTH`: Maximum text length for TTS (default: 1000)
- `MAX_FILE_AGE`: Maximum age of files in hours (default: 24)
- `OLLAMA_HOST`: Ollama server host (default: http://localhost:11434)

### Docker Commands Reference

1. Container Management:
   ```bash
   # Stop containers
   docker stop zhara ollama

   # Start containers
   docker start zhara ollama

   # Restart containers
   docker restart zhara ollama

   # Remove containers
   docker rm -f zhara ollama
   ```

2. Viewing Logs:
   ```bash
   # View Zhara logs
   docker logs zhara

   # Follow Zhara logs
   docker logs -f zhara

   # View Ollama logs
   docker logs ollama
   ```

3. Container Information:
   ```bash
   # Check container status
   docker ps -a

   # View container details
   docker inspect zhara
   ```

4. Volume Management:
   ```bash
   # List volumes
   docker volume ls

   # Clean up unused volumes
   docker volume prune
   ```

### Local Installation (Development)

If you prefer to run without Docker:

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
   - Check if Ollama container is running: `docker ps`
   - Check Ollama logs: `docker logs ollama`
   - Verify OLLAMA_HOST environment variable

2. Audio generation fails:
   - Check available disk space
   - Verify storage directory permissions
   - Check TTS model logs in container

3. Container won't start:
   - Check for port conflicts
   - Verify Docker daemon is running
   - Check system resources

### Getting Help

If you encounter issues:
1. Check the container logs
2. Verify all prerequisites are met
3. Create an issue on GitHub with:
   - Error messages
   - Container logs
   - System information

## Notes
- Make sure Ollama is running locally if you want to use the LLM endpoint.
- The static files are served from the project directory root.

## License
MIT
