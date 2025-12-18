FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Create directories for data and checkpoints
RUN mkdir -p data checkpoints logs

# Set default command
CMD ["/bin/bash"]
