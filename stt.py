from faster_whisper import WhisperModel
import numpy as np
import asyncio


class WhisperStream:
    def __init__(self, sample_rate=16000, chunk_duration=2):
        self.model = WhisperModel(
            "large-v3", device="cpu", compute_type="int8")
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

            segments, _ = self.model.transcribe(
                audio_np, language="en", beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            for seg in segments:
                print(f"[🧠 STT] {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
                yield seg


def new_stream():
    return WhisperStream()
