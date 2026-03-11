"""
JARVIS LLM Client
Async HTTP wrapper for llama.cpp server with streaming support.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger("jarvis.llm")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are JARVIS, a fast on-device AI assistant running on Android. "
    "Keep all replies under 3 sentences. Be concise, direct, and helpful. "
    "Never mention that you are an AI."
)


class LLMClient:
    """Async client for llama.cpp HTTP server."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.base_url = f"http://{host}:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """Check if the LLM server is running."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError) as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the full response.

        Args:
            prompt: User message text
            conversation_history: Optional list of {"role": ..., "content": ...} dicts
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            The LLM response text

        Raises:
            ConnectionError: If the server is not reachable
            RuntimeError: If the server returns an error
        """
        messages = self._build_messages(prompt, conversation_history)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False,
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"LLM server error {resp.status}: {body}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                "Is llm/server.sh running?"
            )

    async def generate_stream(
        self,
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the LLM as they are generated.

        Yields:
            Individual text tokens/chunks as they arrive
        """
        messages = self._build_messages(prompt, conversation_history)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": True,
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"LLM server error {resp.status}: {body}")

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded or not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                "Is llm/server.sh running?"
            )

    def _build_messages(
        self, prompt: str, history: Optional[list[dict]] = None
    ) -> list[dict]:
        """Build the messages array with system prompt, history, and user prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages


async def _test():
    """Test the LLM client with 5 prompts and print average latency."""
    logging.basicConfig(level=logging.INFO)

    client = LLMClient()

    # Health check
    print("Checking LLM server health...")
    healthy = await client.health_check()
    if not healthy:
        print("[FAIL] LLM server is not running.")
        print("       Start it with: bash llm/server.sh")
        return

    print("[OK] LLM server is healthy.\n")

    test_prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "What is 25 times 4?",
        "Name three planets in our solar system.",
        "What is Python used for?",
    ]

    latencies = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"--- Prompt {i}: {prompt}")
        start = time.monotonic()
        try:
            response = await client.generate(prompt)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            print(f"    Response: {response}")
            print(f"    Latency : {elapsed:.3f}s\n")
        except (ConnectionError, RuntimeError) as e:
            print(f"    [ERROR] {e}\n")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print("============================================")
        print(f"  Prompts tested : {len(latencies)}/{len(test_prompts)}")
        print(f"  Avg latency    : {avg:.3f}s")
        print(f"  Min latency    : {min(latencies):.3f}s")
        print(f"  Max latency    : {max(latencies):.3f}s")
        print("============================================")

    await client.close()


if __name__ == "__main__":
    asyncio.run(_test())
