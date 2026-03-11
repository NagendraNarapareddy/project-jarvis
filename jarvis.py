"""
JARVIS — Main Daemon Entry Point
Unified pipeline: STT → LLM → TTS with wake word detection.
"""

import asyncio
import logging
import re
import signal
import sys
import time
from pathlib import Path

import yaml

from llm.client import LLMClient
from stt.listen import listen, listen_async
from tts.speak import speak, speak_async
from utils.logger import setup_logger

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"


class Jarvis:
    """Main JARVIS assistant loop."""

    def __init__(self, config_path: str | Path = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.logger = setup_logger(
            name="jarvis",
            log_file=self.config["jarvis"].get("log_file", "jarvis.log"),
        )

        # LLM client
        llm_cfg = self.config["llm"]
        self.llm = LLMClient(
            host=llm_cfg.get("host", "127.0.0.1"),
            port=llm_cfg.get("port", 8080),
            max_tokens=llm_cfg.get("max_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.7),
            system_prompt=llm_cfg.get("system_prompt", ""),
        )

        # Conversation history (last N turns)
        self.max_history = 10
        self.history: list[dict] = []

        # Wake word
        self.wake_word = self.config["jarvis"].get("wake_word", "hey jarvis").lower()
        self.wake_word_enabled = True

        # State
        self._running = False

    def _add_to_history(self, role: str, content: str):
        """Add a message to conversation history, trimming to max."""
        self.history.append({"role": role, "content": content})
        # Keep last N turns (each turn = user + assistant = 2 entries)
        max_entries = self.max_history * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def _check_wake_word(self, text: str) -> str | None:
        """
        Check if text contains wake word. Returns the command after
        the wake word, or None if wake word not found.
        """
        text_lower = text.lower().strip()

        # Check if text starts with or contains wake word
        if text_lower.startswith(self.wake_word):
            command = text[len(self.wake_word):].strip()
            # If they just said "Hey Jarvis" with nothing after, return empty
            return command if command else ""

        # Also accept without wake word if wake word detection is disabled
        if not self.wake_word_enabled:
            return text.strip()

        return None

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for incremental TTS."""
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    async def process_voice(self) -> bool:
        """
        One cycle: listen → stream LLM → speak sentence by sentence.
        Starts TTS as soon as the first sentence is ready.
        Returns True if a conversation happened, False otherwise.
        """
        # Listen for voice input
        stt_cfg = self.config.get("stt", {})
        text = await listen_async(
            silence_threshold=stt_cfg.get("silence_threshold", 500),
            silence_duration=0.8,
            max_seconds=stt_cfg.get("max_record_seconds", 10),
        )

        if text is None:
            return False

        self.logger.info(f"Heard: {text}")

        # Check wake word
        if self.wake_word_enabled:
            command = self._check_wake_word(text)
            if command is None:
                self.logger.debug(f"No wake word detected in: {text}")
                return False
            if command == "":
                await speak_async("Yes?")
                return True
            user_input = command
        else:
            user_input = text.strip()

        if not user_input:
            return False

        self.logger.info(f"User: {user_input}")

        # Stream LLM response and speak sentence by sentence
        try:
            start = time.monotonic()
            full_response = ""
            buffer = ""
            spoken_sentences = []

            async for token in self.llm.generate_stream(
                prompt=user_input,
                conversation_history=self.history,
            ):
                full_response += token
                buffer += token

                # Check if buffer contains a complete sentence
                sentences = self._split_sentences(buffer)
                if len(sentences) > 1:
                    # Speak all complete sentences, keep the last partial one
                    for sentence in sentences[:-1]:
                        if sentence and sentence not in spoken_sentences:
                            self.logger.info(f"Speaking: {sentence}")
                            await speak_async(sentence)
                            spoken_sentences.append(sentence)
                    buffer = sentences[-1]

            # Speak any remaining text
            remaining = buffer.strip()
            if remaining and remaining not in spoken_sentences:
                self.logger.info(f"Speaking: {remaining}")
                await speak_async(remaining)

            elapsed = time.monotonic() - start
            self.logger.info(f"Total round-trip: {elapsed:.2f}s")
            self.logger.info(f"Full response: {full_response}")

        except ConnectionError:
            self.logger.error("LLM server not available")
            await speak_async("Sorry, my brain is offline right now.")
            return False
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            await speak_async("Sorry, I encountered an error.")
            return False

        # Update history
        self._add_to_history("user", user_input)
        self._add_to_history("assistant", full_response)

        return True

    async def run(self):
        """Main loop — continuously listen and respond."""
        self._running = True

        # Check LLM server
        self.logger.info("Checking LLM server...")
        if not await self.llm.health_check():
            self.logger.error("LLM server not running. Start it with: bash llm/server.sh")
            await speak_async("LLM server is not running. Please start it first.")
            return

        self.logger.info("=" * 50)
        self.logger.info("  JARVIS is ONLINE")
        self.logger.info(f"  Wake word: \"{self.wake_word}\"")
        self.logger.info("=" * 50)

        await speak_async("JARVIS online and ready.")

        while self._running:
            try:
                await self.process_voice()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                # Brief pause before retrying
                await asyncio.sleep(1)

        await self.shutdown()

    async def run_no_wake_word(self):
        """Run without wake word — every voice input goes to LLM."""
        self.wake_word_enabled = False
        await self.run()

    async def shutdown(self):
        """Clean shutdown."""
        self._running = False
        self.logger.info("JARVIS shutting down...")
        await self.llm.close()
        self.logger.info("Goodbye.")

    def stop(self):
        """Signal the loop to stop."""
        self._running = False


async def main():
    jarvis = Jarvis()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def signal_handler():
        jarvis.stop()

    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    # Parse args
    no_wake = "--no-wake" in sys.argv

    if no_wake:
        await jarvis.run_no_wake_word()
    else:
        await jarvis.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJARVIS stopped.")
