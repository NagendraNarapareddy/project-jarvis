"""
JARVIS Full Pipeline Test
Tests one complete voice round-trip: STT → LLM → TTS.
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO)

from jarvis import Jarvis


async def main():
    print("=" * 50)
    print("  JARVIS Pipeline Test")
    print("=" * 50)

    jarvis = Jarvis()
    jarvis.wake_word_enabled = False  # Skip wake word for testing

    # Check LLM
    healthy = await jarvis.llm.health_check()
    if not healthy:
        print("[FAIL] LLM server not running.")
        return False

    print("[OK] LLM server healthy.")
    print("\nSpeak something (will be sent to LLM, reply spoken back)...\n")

    start = time.monotonic()
    result = await jarvis.process_voice()
    elapsed = time.monotonic() - start

    await jarvis.llm.close()

    if result:
        print(f"\n  Round-trip: {elapsed:.3f}s")
        print(f"  Result   : PASS")
        return True
    else:
        print(f"\n  No voice detected or error occurred.")
        print(f"  Result   : FAIL")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
