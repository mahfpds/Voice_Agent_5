import whisper
import numpy as np
import asyncio
import torch

# Global model variable - load once at startup
_whisper_model = None

# Common hallucination phrases to filter out
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thank you.", "thanks.", "thank you very much",
    "you're welcome", "welcome", "hello", "hi", "goodbye", "bye",
    "okay", "ok", "alright", "right", "yes", "yeah", "no", "nope",
    "mm-hmm", "uh-huh", "mm", "uh", "um", "oh", "ah", "hmm"
}


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


def has_sufficient_energy(audio_np, threshold=0.01):
    """Check if audio has sufficient energy to contain speech"""
    rms = np.sqrt(np.mean(audio_np ** 2))
    return rms > threshold


def is_likely_hallucination(text):
    """Check if text is likely a hallucination"""
    text_lower = text.lower().strip()

    # Remove common punctuation
    cleaned = text_lower.replace('.', '').replace(
        ',', '').replace('!', '').replace('?', '').strip()

    # Check against known hallucination phrases
    if cleaned in HALLUCINATION_PHRASES:
        return True

    # Check if it's just repeated characters or very short
    if len(cleaned) < 4:
        return True

    return False


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

            # First check: Does the audio have sufficient energy?
            if not has_sufficient_energy(audio_np, threshold=0.015):
                print("[🔇 STT] Skipping low-energy audio (likely silence)")
                continue

            # Use OpenAI Whisper transcription with very aggressive thresholds
            result = self.model.transcribe(
                audio_np,
                language="en",
                verbose=False,
                no_speech_threshold=0.9,   # Much higher - almost certain speech needed
                logprob_threshold=-0.3,    # Much stricter probability threshold
                condition_on_previous_text=False,  # Don't use context from previous text
                compression_ratio_threshold=2.0,   # Detect repetitive/nonsense audio
                temperature=0.0,  # Use most confident predictions only
                initial_prompt=""  # No initial prompt to bias transcription
            )

            if result.get("segments"):
                for segment in result["segments"]:
                    # Additional filtering - only accept segments with reasonable content
                    text = segment['text'].strip()

                    # Skip very short segments (likely noise)
                    if len(text) < 5:
                        print(f"[❌ STT] Skipping too short: '{text}'")
                        continue

                    # Skip likely hallucinations
                    if is_likely_hallucination(text):
                        print(f"[❌ STT] Filtering hallucination: '{text}'")
                        continue

                    # Check minimum duration (avoid super quick "words")
                    duration = segment['end'] - segment['start']
                    if duration < 0.5:  # Less than half a second
                        print(
                            f"[❌ STT] Skipping too fast: '{text}' ({duration:.2f}s)")
                        continue

                    # Check average log probability for this segment
                    if 'avg_logprob' in segment and segment['avg_logprob'] < -0.4:
                        print(
                            f"[❌ STT] Low confidence: '{text}' (logprob: {segment['avg_logprob']:.3f})")
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
