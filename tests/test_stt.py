"""
JARVIS STT Test
Record 3 seconds of audio and print transcript.
"""

import sys
import time

sys.path.insert(0, ".")
from stt.listen import record_audio, transcribe, DEFAULT_MODEL, WHISPER_BINARY

import logging
logging.basicConfig(level=logging.DEBUG)


def main():
    print("=" * 50)
    print("  JARVIS STT Test")
    print("=" * 50)

    print(f"  Model : {DEFAULT_MODEL}")
    print(f"  Binary: {WHISPER_BINARY}")

    # Check dependencies
    if not DEFAULT_MODEL.exists():
        print(f"[FAIL] Model not found: {DEFAULT_MODEL}")
        print("       Run: bash models/download_stt.sh")
        return False

    if not WHISPER_BINARY.exists():
        print(f"[FAIL] Whisper binary not found: {WHISPER_BINARY}")
        return False

    try:
        import pyaudio
    except ImportError:
        print("[FAIL] pyaudio not installed. Run: pip install pyaudio")
        return False

    print("\nSpeak something (3 seconds)...")
    start = time.monotonic()

    wav_data = record_audio(max_seconds=3, silence_duration=3.5)
    if wav_data is None:
        print("[FAIL] No audio captured. Check microphone.")
        return False

    text = transcribe(wav_data)
    elapsed = time.monotonic() - start

    if text:
        print(f"\n  Transcript: \"{text}\"")
        print(f"  Latency   : {elapsed:.3f}s")
        print("  Result    : PASS")
        return True
    else:
        print("\n  No speech recognized.")
        print("  Result    : FAIL")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
