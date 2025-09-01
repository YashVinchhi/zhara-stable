# Zhara AI Assistant - Dockerfile
# Base image with Python
FROM python:3.11-slim

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: ffmpeg and minimal build tools
# Core system deps for audio/TTS: ffmpeg, libsndfile1 (for soundfile), espeak-ng (phonemizer), plus basic build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency manifests first for better caching
COPY requirements.txt ./

# Torch CPU wheels are heavy; allow override. Default to CPU wheels to avoid GPU drivers in container
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PIP_EXTRA_INDEX_URL=${TORCH_INDEX}

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required directories
RUN mkdir -p storage/audio storage/visemes logs

# Expose port
EXPOSE 8000

# Runtime environment defaults (can be overridden)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    MAX_AUDIO_DURATION=300 \
    MAX_TEXT_LENGTH=1000 \
    MAX_FILE_AGE=24

# Healthcheck hitting FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1

# Start the app with Uvicorn
CMD ["python", "zhara.py"]
