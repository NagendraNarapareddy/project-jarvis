"""
JARVIS LLM Test
Sends 5 prompts to the LLM server and reports latency.
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")
from llm.client import LLMClient


async def main():
    client = LLMClient()

    print("=" * 50)
    print("  JARVIS LLM Test")
    print("=" * 50)

    # Health check
    healthy = await client.health_check()
    if not healthy:
        print("\n[FAIL] LLM server not reachable at localhost:8080")
        print("       Start it with: bash llm/server.sh")
        await client.close()
        return False

    print("[OK] Server is healthy.\n")

    prompts = [
        "Say hello in one sentence.",
        "What is 10 plus 15?",
        "Name a color.",
        "What day comes after Monday?",
        "Say goodbye in one sentence.",
    ]

    latencies = []
    passed = 0

    for i, prompt in enumerate(prompts, 1):
        try:
            start = time.monotonic()
            response = await client.generate(prompt, max_tokens=64)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            passed += 1
            print(f"  [{i}] {prompt}")
            print(f"      -> {response.strip()}")
            print(f"      Latency: {elapsed:.3f}s\n")
        except Exception as e:
            print(f"  [{i}] {prompt}")
            print(f"      [ERROR] {e}\n")

    await client.close()

    if latencies:
        avg = sum(latencies) / len(latencies)
        print("-" * 50)
        print(f"  Passed: {passed}/{len(prompts)}")
        print(f"  Avg latency: {avg:.3f}s")
        target_met = avg < 5.0  # Generous target for testing
        status = "PASS" if passed == len(prompts) else "FAIL"
        print(f"  Result: {status}")
        print("-" * 50)
        return passed == len(prompts)

    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
