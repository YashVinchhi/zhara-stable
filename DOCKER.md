# Zhara AI Assistant — Docker Guide

This guide covers everything you need to build and run Zhara using Docker or Docker Compose.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Optional: Docker Compose v2 (bundled with Docker Desktop)
- Ollama running somewhere (local host or remote) for LLM features

Recommended: clone the repository so you have `docker/Dockerfile`, `docker-compose.yml`, `.env.example`.

## Quick Start (Docker Compose)

1) Copy the example environment file:
```powershell
copy .env.example .env
```

2) Start the app:
```powershell
docker compose up --build
```

3) Open the app:
```
http://localhost:8000
```

To stop:
```powershell
docker compose down
```

## Quick Start (Docker CLI)

Build the image:
```powershell
docker build -t zhara:latest -f docker/Dockerfile .
```

Run the container (persists storage and logs on the host):
```powershell
docker run --rm -p 8000:8000 `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v ${PWD}\storage:/app/storage `
  -v ${PWD}\logs:/app/logs `
  zhara:latest
```

Open the app:
```
http://localhost:8000
```

## Environment Variables

You can set these via `.env` (Compose) or `-e` flags (Docker):

- `HOST` (default: 0.0.0.0)
- `PORT` (default: 8000)
- `OLLAMA_BASE_URL` (default: http://host.docker.internal:11434)
- `MAX_AUDIO_DURATION` (default: 300)
- `MAX_TEXT_LENGTH` (default: 1000)
- `MAX_FILE_AGE` (default: 24)

Copy and edit `.env.example` as needed:
```powershell
copy .env.example .env
notepad .env
```

## Connecting to Ollama

- Windows/Mac with Docker Desktop: `host.docker.internal` resolves to your host, so if Ollama runs on your host at port 11434, the default works.
- Linux: modern Docker supports `host.docker.internal`; if not, you can add a host entry:
  ```yaml
  # docker-compose.yml example (service level)
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```
  Or run with:
  ```bash
  docker run --add-host=host.docker.internal:host-gateway ...
  ```
- Remote Ollama: set `OLLAMA_BASE_URL=http://<remote-ip>:11434`.

## Volumes and Persistence

The app stores generated audio and visemes in `storage/`, and logs in `logs/`.

- Compose mounts:
  ```yaml
  volumes:
    - ./storage:/app/storage
    - ./logs:/app/logs
  ```
- Docker run mounts:
  ```powershell
  -v ${PWD}\storage:/app/storage -v ${PWD}\logs:/app/logs
  ```

## Health Check and Logs

- Health endpoint: `GET http://localhost:8000/health`
- Container logs (Compose):
  ```powershell
  docker compose logs -f
  ```
- Container logs (Docker):
  ```powershell
  docker logs -f <container_id>
  ```

## Updating

- Pull latest code and rebuild:
  ```powershell
  git pull
  docker compose build --no-cache
  docker compose up -d
  ```

## GPU Notes

This image uses CPU-only PyTorch wheels for simplicity and portability. For GPU acceleration you’d need:

- A CUDA-enabled base image (e.g., `nvidia/cuda`),
- Matching `torch`/`torchaudio` CUDA wheels,
- NVIDIA Container Toolkit, and
- Compose or run flags to enable GPU (`--gpus all`).

This is out of scope for this guide.

## Troubleshooting

- Docker daemon not running
  - Start Docker Desktop; ensure `docker info` works before building.

- Port 8000 already in use
  - Change published port (Compose): `ports: ["8080:8000"]`
  - Docker: `-p 8080:8000`

- Slow first build
  - Large ML dependencies are installed; subsequent builds are cached.

- Cannot reach Ollama from container
  - Verify `OLLAMA_BASE_URL` and network path.
  - On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` if needed.

- Audio/TTS errors
  - Ensure `ffmpeg` is present (installed in the image).
  - Check `storage/` and `logs/` are writable.
