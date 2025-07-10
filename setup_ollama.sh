#!/bin/bash

# Setup script for Ollama on RunPod

echo "🚀 Setting up Ollama..."

# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service in background
echo "🔄 Starting Ollama service..."
ollama serve &

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to initialize..."
sleep 10

# Pull the required model
echo "📥 Downloading gemma3:27b model..."
ollama pull gemma3:27b

echo "✅ Ollama setup complete!"
echo "🎯 Model gemma3:27b is ready to use" 