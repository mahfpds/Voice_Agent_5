import whisper
import numpy as np
import asyncio
import torch

# Global model variable - load once at startup
_whisper_model = None


def load_whisper_model():
    """Load Whisper model once at startup"""
    global _whisper_model
    if _whisper_model is None:
        try:
            print("[INFO] Loading Whisper model at startup...")
            _whisper_model = whisper.load_model("large-v3").cuda()
            print("[INFO] ✅ Whisper model loaded successfully on GPU")
        except Exception as e:
            print(f"[WARNING] CUDA failed: {e}, falling back to CPU")
            _whisper_model = whisper.load_model("large-v3")
            print("[INFO] ✅ Whisper model loaded successfully on CPU")
    return _whisper_model


class WhisperStream:
    def __init__(self, sample_rate=16000, chunk_duration=2):
        # Use the pre-loaded global model
        self.model = load_whisper_model()
        print("[INFO] Using pre-loaded Whisper model for new stream")

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

            # Use OpenAI Whisper transcription with better thresholds
            result = self.model.transcribe(
                audio_np,
                language="en",
                verbose=False,
                no_speech_threshold=0.8,  # Higher threshold to reduce false positives
                logprob_threshold=-0.5,   # More strict probability threshold
                condition_on_previous_text=False,  # Don't use context from previous text
                compression_ratio_threshold=2.4,   # Detect repetitive/nonsense audio
                temperature=0.0  # Use most confident predictions only
            )

            if result.get("segments"):
                for segment in result["segments"]:
                    # Additional filtering - only accept segments with reasonable content
                    text = segment['text'].strip()

                    # Skip very short segments (likely noise)
                    if len(text) < 3:
                        continue

                    # Skip segments that are just punctuation or single letters
                    if text.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '') == '':
                        continue

                    print(
                        f"[🧠 STT] {segment['start']:.2f}s - {segment['end']:.2f}s: {text}")
                    # Create a segment-like object for compatibility

                    class Segment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text

                    yield Segment(segment['start'], segment['end'], text)


def new_stream():
    return WhisperStream()
