import runpod
import os
from app import app
import asyncio
from threading import Thread
import uvicorn

# Set environment variables
os.environ["ELEVENLABS_API_KEY"] = "sk_86910b1ddb6b98f817323852eb68a6b6d6665f8f4ce51d2d"
os.environ["ELEVENLABS_VOICE_ID"] = "kdmDKE6EkgrWrrykO9Qt"

# Start FastAPI app in background thread


def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8888)


# Start the FastAPI server
server_thread = Thread(target=start_fastapi, daemon=True)
server_thread.start()


def handler(event):
    """
    RunPod serverless handler
    """
    # For voice agent, we just need to keep the server running
    # The actual webhook handling is done by FastAPI
    return {
        "statusCode": 200,
        "body": "Voice agent is running"
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
