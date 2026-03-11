"""
JARVIS TTS Test
Speak "JARVIS online and ready" and report latency.
"""

import sys
import time

sys.path.insert(0, ".")
from tts.speak import speak, PIPER_BINARY, VOICE_MODEL


def main():
    print("=" * 50)
    print("  JARVIS TTS Test")
    print("=" * 50)
    print(f"  Binary : {PIPER_BINARY}")
    print(f"  Model  : {VOICE_MODEL}")

    if not PIPER_BINARY.exists():
        print("[FAIL] Piper binary not found.")
        print("       Download from piper releases and extract to piper/")
        return False

    if not VOICE_MODEL.exists():
        print("[FAIL] Voice model not found.")
        print("       Run: bash models/download_tts.sh")
        return False

    test_text = "Hi Nagendra how are you man."
    print(f"\n  Speaking: \"{test_text}\"")

    start = time.monotonic()
    success = speak(test_text)
    elapsed = time.monotonic() - start

    if success:
        print(f"  Latency : {elapsed:.3f}s")
        print(f"  Result  : PASS")
        return True
    else:
        print(f"  Result  : FAIL")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
