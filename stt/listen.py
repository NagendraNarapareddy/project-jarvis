"""
JARVIS Speech-to-Text
Mic capture → whisper.cpp → text string.
Uses VAD (voice activity detection) to auto-stop recording.
"""

import asyncio
import io
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

logger = logging.getLogger("jarvis.stt")

# Defaults
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500       # RMS amplitude below this = silence
SILENCE_DURATION = 0.8        # Seconds of silence before auto-stop
MAX_RECORD_SECONDS = 10       # Hard cap on recording time
MIN_RECORD_SECONDS = 0.5      # Minimum audio length to process

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "ggml-tiny.en-q5_1.bin"

# Detect whisper binary (Windows vs Linux/Termux)
if sys.platform == "win32":
    WHISPER_BINARY = PROJECT_DIR / "whisper.cpp" / "whisper-cli.exe"
    if not WHISPER_BINARY.exists():
        WHISPER_BINARY = PROJECT_DIR / "whisper.cpp" / "main.exe"
else:
    # cmake build puts binary in build/bin/
    WHISPER_BINARY = PROJECT_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    if not WHISPER_BINARY.exists():
        WHISPER_BINARY = PROJECT_DIR / "whisper.cpp" / "main"


def _rms(data: bytes) -> float:
    """Calculate RMS (root mean square) amplitude of 16-bit PCM audio."""
    if len(data) < 2:
        return 0.0
    count = len(data) // 2
    shorts = struct.unpack(f"<{count}h", data[:count * 2])
    sum_sq = sum(s * s for s in shorts)
    return (sum_sq / count) ** 0.5


def record_audio(
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> bytes | None:
    """
    Record audio from microphone with VAD auto-stop.

    Returns:
        Raw WAV bytes, or None if no speech detected.
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("pyaudio not installed. Run: pip install pyaudio")
        return None

    pa = pyaudio.PyAudio()
    stream = None

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info("Listening... (speak now)")
        frames = []
        silent_chunks = 0
        has_speech = False
        max_chunks = int(max_seconds * sample_rate / CHUNK_SIZE)
        silence_chunks_limit = int(silence_duration * sample_rate / CHUNK_SIZE)

        for _ in range(max_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)

            rms = _rms(data)

            if rms > silence_threshold:
                has_speech = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            # Stop if we had speech and then silence
            if has_speech and silent_chunks >= silence_chunks_limit:
                logger.info("Silence detected, stopping recording.")
                break

        if not has_speech:
            logger.info("No speech detected.")
            return None

        # Check minimum length
        audio_duration = len(frames) * CHUNK_SIZE / sample_rate
        if audio_duration < MIN_RECORD_SECONDS:
            logger.info("Audio too short, ignoring.")
            return None

        # Convert to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

        return buf.getvalue()

    except Exception as e:
        logger.error(f"Recording error: {e}")
        return None
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        pa.terminate()


def transcribe(
    wav_data: bytes,
    model_path: str | Path = DEFAULT_MODEL,
    language: str = "en",
) -> str | None:
    """
    Transcribe WAV audio using whisper.cpp.

    Args:
        wav_data: Raw WAV file bytes
        model_path: Path to whisper GGML model
        language: Language code

    Returns:
        Transcribed text string, or None on failure.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Whisper model not found: {model_path}")
        logger.error("Run: bash models/download_stt.sh")
        return None

    if not WHISPER_BINARY.exists():
        logger.error(f"Whisper binary not found: {WHISPER_BINARY}")
        logger.error("Download whisper.cpp and place binary in whisper.cpp/ folder")
        return None

    # Write WAV to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(wav_data)
        tmp.close()

        cmd = [
            str(WHISPER_BINARY),
            "-m", str(model_path),
            "-f", tmp.name,
            "-l", language,
            "--no-timestamps",
            "-np",           # No prints except results
        ]

        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(f"Whisper error: {result.stderr}")
            return None

        # Clean up output
        text = result.stdout.strip()
        # Remove common whisper artifacts
        text = text.replace("[BLANK_AUDIO]", "").strip()
        text = text.replace("(silence)", "").strip()

        if not text or text in ("", " ", ".", ".."):
            return None

        logger.info(f"Transcription: {text}")
        return text

    except subprocess.TimeoutExpired:
        logger.error("Whisper transcription timed out")
        return None
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None
    finally:
        os.unlink(tmp.name)


def listen(
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
    model_path: str | Path = DEFAULT_MODEL,
) -> str | None:
    """
    Full pipeline: record from mic → transcribe → return text.

    Returns:
        Clean UTF-8 text string, or None if nothing was heard.
    """
    wav_data = record_audio(
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        max_seconds=max_seconds,
    )
    if wav_data is None:
        return None

    return transcribe(wav_data, model_path=model_path)


async def listen_async(**kwargs) -> str | None:
    """Async wrapper around listen() — runs in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: listen(**kwargs))


# ─── Standalone test mode ───────────────────────────────────
def _test():
    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("  JARVIS STT Test")
    print("=" * 50)
    print(f"  Model  : {DEFAULT_MODEL}")
    print(f"  Binary : {WHISPER_BINARY}")
    print()

    if not DEFAULT_MODEL.exists():
        print("[FAIL] Model not found. Run: bash models/download_stt.sh")
        return

    if not WHISPER_BINARY.exists():
        print("[FAIL] Whisper binary not found.")
        print("       Download from whisper.cpp releases and place in whisper.cpp/")
        return

    print("Recording 3 seconds of audio...")
    start = time.monotonic()
    wav_data = record_audio(max_seconds=3, silence_duration=3.5)

    if wav_data is None:
        print("[WARN] No audio captured. Check your microphone.")
        return

    print(f"Recorded {len(wav_data)} bytes. Transcribing...")
    text = transcribe(wav_data)
    elapsed = time.monotonic() - start

    if text:
        print(f"\n  Transcript: \"{text}\"")
        print(f"  Latency   : {elapsed:.3f}s")
        print(f"  Result    : PASS")
    else:
        print(f"\n  No speech recognized.")
        print(f"  Result    : FAIL")


if __name__ == "__main__":
    _test()
