"""
JARVIS Text-to-Speech
Text → Piper TTS → audio output.
Supports async non-blocking playback.
"""

import asyncio
import io
import logging
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

logger = logging.getLogger("jarvis.tts")

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Detect Piper binary
if sys.platform == "win32":
    PIPER_BINARY = PROJECT_DIR / "piper" / "piper.exe"
else:
    PIPER_BINARY = PROJECT_DIR / "piper" / "piper"

VOICE_MODEL = PROJECT_DIR / "models" / "en_US-lessac-medium.onnx"
VOICE_CONFIG = PROJECT_DIR / "models" / "en_US-lessac-medium.onnx.json"


def speak(
    text: str,
    piper_binary: str | Path = PIPER_BINARY,
    voice_model: str | Path = VOICE_MODEL,
    voice_config: str | Path = VOICE_CONFIG,
) -> bool:
    """
    Speak text using Piper TTS.

    Args:
        text: Text to speak
        piper_binary: Path to piper executable
        voice_model: Path to .onnx voice model
        voice_config: Path to .onnx.json config

    Returns:
        True if successful, False on error.
    """
    piper_binary = Path(piper_binary)
    voice_model = Path(voice_model)

    if not piper_binary.exists():
        logger.error(f"Piper binary not found: {piper_binary}")
        return False

    if not voice_model.exists():
        logger.error(f"Voice model not found: {voice_model}")
        return False

    text = text.strip()
    if not text:
        return True

    try:
        # Generate WAV to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            str(piper_binary),
            "--model", str(voice_model),
            "--output_file", tmp.name,
        ]

        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(f"Piper error: {result.stderr}")
            return False

        # Play the audio
        _play_wav(tmp.name)

        # Clean up
        Path(tmp.name).unlink(missing_ok=True)

        logger.info(f"Spoke: {text[:50]}...")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out")
        return False
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False


def _play_wav(wav_path: str):
    """Play a WAV file using the best available method."""
    if sys.platform == "win32":
        _play_wav_sounddevice(wav_path)
    else:
        # Linux/Termux: try play-audio (Termux), then aplay, then paplay
        for player in ["play-audio", "aplay -q", "paplay"]:
            cmd = player.split() + [wav_path]
            try:
                subprocess.run(cmd, timeout=30, check=True,
                               capture_output=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        logger.error("No audio player found. Install termux-api or pulseaudio.")


def _play_wav_sounddevice(wav_path: str):
    """Play WAV using sounddevice (cross-platform)."""
    try:
        import sounddevice as sd
        import numpy as np

        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert to numpy array
        audio = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    except ImportError:
        # Fallback: winsound for basic WAV playback
        import winsound
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)


async def speak_async(
    text: str,
    **kwargs,
) -> bool:
    """
    Async non-blocking TTS. Runs speak() in a thread pool
    so JARVIS can keep listening while audio plays.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: speak(text, **kwargs))


# ─── Standalone test ────────────────────────────────────────
def _test():
    logging.basicConfig(level=logging.DEBUG)
    print("=" * 50)
    print("  JARVIS TTS Test")
    print("=" * 50)
    print(f"  Binary : {PIPER_BINARY}")
    print(f"  Model  : {VOICE_MODEL}")
    print()

    if not PIPER_BINARY.exists():
        print(f"[FAIL] Piper binary not found: {PIPER_BINARY}")
        print("       Download from piper releases and extract to piper/")
        return

    if not VOICE_MODEL.exists():
        print(f"[FAIL] Voice model not found: {VOICE_MODEL}")
        print("       Run: bash models/download_tts.sh")
        return

    test_text = "JARVIS online and ready."
    print(f"  Speaking: \"{test_text}\"")

    start = time.monotonic()
    success = speak(test_text)
    elapsed = time.monotonic() - start

    if success:
        print(f"\n  Latency : {elapsed:.3f}s")
        print(f"  Result  : PASS")
    else:
        print(f"\n  Result  : FAIL")


if __name__ == "__main__":
    _test()
