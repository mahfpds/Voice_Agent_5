import os
import base64
import json
import asyncio
import audioop
import time
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from stt import new_stream
from tts import stream as tts_stream
from ollama import Client as OllamaClient
from dotenv import load_dotenv
import wave
from lang import get_llm_response


load_dotenv()

PORT = int(os.getenv("PORT", 5050))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("LLM_MODEL", "gemma3:27b")
def tnr(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Custom Twilio Voice Bot is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    # Use RunPod proxy URL for WebSocket connection
    host = "nwg343q6btw0c8-5050.proxy.runpod.net"
    response = f"""
    <Response>
        <Connect>
            <Stream url="wss://{host}/media-stream" />
        </Connect>
    </Response>
    """
    return HTMLResponse(content=response, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("[🎧] Caller connected via WebSocket")

    stream_stt = new_stream()
    buffer = bytearray()
    chunk_duration_sec = 2
    sample_rate = 16000
    bytes_per_sec = sample_rate * 2

    stream_sid = None  # will hold the real stream ID
    stream_cid = None  # will hold the real call ID

    async def receive_from_twilio():
        nonlocal stream_sid
        nonlocal stream_cid
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)

                if data['event'] == 'media':
                    ulaw = base64.b64decode(data['media']['payload'])
                    pcm = audioop.ulaw2lin(ulaw, 2)  # 8-bit μ-law → 16-bit PCM
                    print(f"[DEBUG] Received {len(pcm)} bytes of PCM audio")
                    buffer.extend(pcm)

                    while len(buffer) >= bytes_per_sec * chunk_duration_sec:
                        chunk = buffer[:bytes_per_sec * chunk_duration_sec]
                        buffer[:] = buffer[bytes_per_sec * chunk_duration_sec:]
                        print(f"[DEBUG] Feeding {len(chunk)} bytes to Whisper")
                        stream_stt.feed_audio(chunk)

                elif data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    stream_cid = data['start']['callSid']
                    print(f"[🔄] Started stream: {stream_sid}")
                    print(f"[🔄] Call ID: {stream_cid}")
                    await send_file(websocket, stream_sid, 'intro.wav')

        except WebSocketDisconnect:
            print("[❌] WebSocket disconnected")

    async def respond_to_user():
        nonlocal stream_sid
        nonlocal stream_cid

        async for seg in stream_stt:
            text = seg.text.strip()
            if not text:
                continue

            print(f"[{tnr()}] [user said] : {text}")

            response = get_llm_response(text, stream_cid)

            print(f"[{tnr()}] [assistant response] : {response}")

            await send_tts(response, websocket, stream_sid)

    await asyncio.gather(receive_from_twilio(), respond_to_user())


async def send_tts(text: str, websocket: WebSocket, stream_sid: str = None):
    pcm_data = bytearray()
    for ulaw_chunk in tts_stream(text):
        pcm_chunk = audioop.ulaw2lin(ulaw_chunk, 2)  # output is 16-bit PCM
        pcm_data.extend(pcm_chunk)

    pcm_data_copy = pcm_data[:]
    f_name = f"out{tnr()}.wav"

    with wave.open(f_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(8000)
        wf.writeframes(pcm_data_copy)

    await send_file(websocket, stream_sid, f_name)

    os.remove(f_name)


async def send_file(websocket: WebSocket, stream_sid: str, f_name: str):

    with wave.open(f_name, "rb") as wav:
        raw_wav = wav.readframes(wav.getnframes())
        raw_ulaw = audioop.lin2ulaw(raw_wav, wav.getsampwidth())
        encoded_audio = base64.b64encode(raw_ulaw).decode("utf-8")

    await websocket.send_json({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": encoded_audio},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
