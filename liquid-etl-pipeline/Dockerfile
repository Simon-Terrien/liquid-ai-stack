# liquid-etl-pipeline/Dockerfile
# Multi-stage build with CPU/GPU support
# Build: docker build -t liquid-etl .
# Run CPU: docker run --rm -v ./models:/models -v ./data:/data liquid-etl
# Run GPU: docker run --rm --gpus all -v ./models:/models -v ./data:/data liquid-etl

ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04

FROM ${BASE_IMAGE}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash agent
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir .

# Install shared core (in production, this would come from PyPI or internal registry)
# For dev, mount as volume or copy
COPY --chown=agent:agent . .

# Create data directories
RUN mkdir -p /data/raw /data/processed /data/ft /data/vectordb && \
    chown -R agent:agent /data

# Volume mount points
VOLUME ["/models", "/data"]

# Environment variables for device selection
# Set CUDA_VISIBLE_DEVICES="" for CPU-only mode
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/models/.cache
ENV TRANSFORMERS_CACHE=/models/.cache

# Switch to non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "from liquid_shared import check_device_compatibility; print(check_device_compatibility())"

# Default command
CMD ["python3", "-m", "etl_pipeline.run_etl"]
