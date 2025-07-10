# Voice Agent - Twilio Integration

A real-time voice agent that integrates Twilio calling with AI-powered speech recognition, language processing, and text-to-speech.

## Features

- 🎙️ Real-time speech-to-text using faster-whisper
- 🧠 AI responses via Ollama + Langchain
- 🔊 High-quality text-to-speech via ElevenLabs
- 📞 Twilio WebSocket integration
- ⚡ FastAPI backend with WebSocket support

## Dependencies

- **Python 3.8+** with CUDA support
- **NVIDIA GPU** (required for faster-whisper)
- **System packages**: mpg123, ffmpeg, sox
- **External services**: ElevenLabs API, Twilio

## RunPod Deployment Guide

### Step 1: Create a RunPod Instance

1. Go to [RunPod.io](https://runpod.io) and create an account
2. Click "Deploy" > "GPU Pods"
3. Select a GPU instance (recommended: RTX 4090 or better)
4. Choose a template with CUDA support (PyTorch template recommended)
5. Set up SSH access for easier management

### Step 2: Environment Setup

1. **Upload your code** to the pod:
   ```bash
   # Via SSH or RunPod file manager
   git clone <your-repo> /workspace/voice-agent
   cd /workspace/voice-agent
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   nano .env
   ```

3. **Install system dependencies**:
   ```bash
   apt-get update
   apt-get install -y mpg123 ffmpeg sox curl wget
   ```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up Ollama

```bash
# Make setup script executable
chmod +x setup_ollama.sh

# Run Ollama setup (this will take several minutes)
./setup_ollama.sh
```

### Step 5: Configure Environment Variables

Edit your `.env` file with the following required values:

```bash
# ElevenLabs (required for TTS)
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Server configuration
PORT=5050
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gemma3:27b
```

### Step 6: Run the Application

```bash
python3 app.py
```

Your voice agent will be running on port 5050. You can access it at `http://your-pod-ip:5050`

### Step 7: Twilio Configuration

1. Configure your Twilio webhook URL to point to your RunPod instance:
   ```
   https://your-pod-ip:5050/incoming-call
   ```

2. Ensure your RunPod instance has a public IP and the port is accessible.

## Docker Deployment (Alternative)

If you prefer using Docker:

```bash
# Build the image
docker build -t voice-agent .

# Run with GPU support
docker run --gpus all -p 5050:5050 --env-file .env voice-agent
```

## Troubleshooting

### Common Issues:

1. **CUDA not available**: Ensure you selected a GPU-enabled RunPod instance
2. **Ollama connection failed**: Make sure Ollama service is running (`ollama serve`)
3. **Audio processing errors**: Verify mpg123 and ffmpeg are installed
4. **ElevenLabs API errors**: Check your API key and voice ID in `.env`

### Monitoring Logs:

```bash
# Check application logs
tail -f app.log

# Check Ollama status
ollama list
```

## Performance Tips

- Use RTX 4090 or better for optimal faster-whisper performance
- Monitor GPU memory usage during operation
- Consider using smaller Ollama models for faster responses (e.g., `gemma3:7b`)

## Security Notes

- Keep your API keys secure and never commit them to version control
- Use HTTPS for production deployments
- Consider implementing rate limiting for public endpoints 