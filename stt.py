import whisper
import numpy as np
import asyncio
import torch


class WhisperStream:
    def __init__(self, sample_rate=16000, chunk_duration=2):
        try:
            # Use OpenAI Whisper with GPU (better cuDNN compatibility)
            self.model = whisper.load_model("large-v3").cuda()
            print("[INFO] Using CUDA GPU for OpenAI Whisper")
        except Exception as e:
            print(f"[WARNING] CUDA failed: {e}, falling back to CPU")
            self.model = whisper.load_model("large-v3")

        self.sample_rate = sample_rate
        self.frames_per_chunk = int(sample_rate * chunk_duration)
        self.buffer = bytearray()
        self.queue = asyncio.Queue()

    def feed_audio(self, pcm16: bytes):
        self.buffer.extend(pcm16)
        while len(self.buffer) >= self.frames_per_chunk * 2:  # 2 bytes per frame (16-bit)
            chunk = self.buffer[:self.frames_per_chunk * 2]
            self.buffer = self.buffer[self.frames_per_chunk * 2:]
            self.queue.put_nowait(chunk)

    async def __aiter__(self):
        while True:
            chunk = await self.queue.get()
            audio_np = np.frombuffer(
                chunk, np.int16).astype(np.float32) / 32768.0

            # pad if shorter than expected (e.g. final buffer flush)
            if len(audio_np) < self.frames_per_chunk:
                pad = np.zeros(self.frames_per_chunk -
                               len(audio_np), dtype=np.float32)
                audio_np = np.concatenate([audio_np, pad])
            else:
                audio_np = audio_np[:self.frames_per_chunk]

            # Use OpenAI Whisper transcription
            result = self.model.transcribe(
                audio_np,
                language="en",
                verbose=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )

            if result.get("segments"):
                for segment in result["segments"]:
                    print(
                        f"[🧠 STT] {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
                    # Create a segment-like object for compatibility

                    class Segment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text

                    yield Segment(segment['start'], segment['end'], segment['text'])


def new_stream():
    return WhisperStream()
