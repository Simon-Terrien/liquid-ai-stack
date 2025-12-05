# liquid-rag-runtime/Dockerfile
# Multi-stage build optimized for fast inference
# Build: docker build -t liquid-rag .
# Run CPU: docker run --rm -p 8000:8000 -v ./models:/models -v ./data:/data liquid-rag
# Run GPU: docker run --rm --gpus all -p 8000:8000 -v ./models:/models -v ./data:/data liquid-rag

ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash agent
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir .

# Copy application
COPY --chown=agent:agent . .

# Create data directories
RUN mkdir -p /data/vectordb && \
    chown -R agent:agent /data

# Volume mount points
VOLUME ["/models", "/data"]

# Environment
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/models/.cache
ENV TRANSFORMERS_CACHE=/models/.cache

# Expose API port
EXPOSE 8000

# Switch to non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
CMD ["python3", "-m", "rag_runtime.api_server", "--host", "0.0.0.0", "--port", "8000"]
