# Use standard Ubuntu base image (CUDA will be available from RunPod host)
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    mpg123 \
    ffmpeg \
    sox \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support separately
RUN pip3 install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create directory for temporary audio files
RUN mkdir -p /tmp/audio

# Expose port
EXPOSE 5050

# Command to run the application
CMD ["python3", "app.py"] 