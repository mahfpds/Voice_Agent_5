import os, httpx, subprocess, audioop
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

def stream(text: str):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        mp3_data = response.content

    # Save MP3 to temp file
    with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data)
        tmp_mp3.flush()
        mp3_path = tmp_mp3.name

    # Decode MP3 to raw PCM @ 8kHz mono
    decode_proc = subprocess.Popen(
        ["mpg123", "-q", "--rate", "8000", "--mono", "-w", "-", mp3_path],
        stdout=subprocess.PIPE
    )

    chunk_size = 160  # 20ms of μ-law at 8kHz = 160 samples = 320 bytes PCM
    while True:
        pcm_chunk = decode_proc.stdout.read(chunk_size * 2)  # 2 bytes/sample
        if not pcm_chunk:
            break
        ulaw_chunk = audioop.lin2ulaw(pcm_chunk, 2)
        yield ulaw_chunk

