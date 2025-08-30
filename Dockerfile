# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Add metadata labels
LABEL maintainer="Yash Vinchhi <your-email@example.com>"
LABEL version="1.0.0"
LABEL description="Zhara AI Assistant - A conversational AI with speech and viseme generation capabilities"
LABEL org.opencontainers.image.source="https://github.com/YashVinchhi/zhara"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

    # Install Rhubarb Lip Sync
RUN apt-get update && apt-get install -y wget unzip \
    && wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    && unzip Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    && mv Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb /usr/local/bin/ \
    && rm Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    && rm -rf Rhubarb-Lip-Sync-1.13.0-Linux \
    && rm -rf /var/lib/apt/lists/*# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p storage/audio storage/visemes static/css static/js

# Expose port
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["uvicorn", "zhara:app", "--host", "0.0.0.0", "--port", "8000"]
