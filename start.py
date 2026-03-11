"""
JARVIS Single Launcher
Starts LLM server + JARVIS pipeline in one script.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"

# Load config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Detect LLM server binary
if sys.platform == "win32":
    LLAMA_SERVER = PROJECT_DIR / "llama.cpp" / "llama-server.exe"
else:
    LLAMA_SERVER = PROJECT_DIR / "llama.cpp" / "llama-server"

MODEL_PATH = PROJECT_DIR / config["llm"]["model_path"]
LLM_HOST = config["llm"].get("host", "127.0.0.1")
LLM_PORT = config["llm"].get("port", 8080)
CTX_SIZE = config["llm"].get("context_size", 2048)
THREADS = config["llm"].get("threads", 4)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launcher")

llm_process = None


def start_llm_server() -> subprocess.Popen | None:
    """Start llama-server as a background process."""
    if not LLAMA_SERVER.exists():
        logger.error(f"llama-server not found: {LLAMA_SERVER}")
        return None

    if not MODEL_PATH.exists():
        logger.error(f"LLM model not found: {MODEL_PATH}")
        return None

    cmd = [
        str(LLAMA_SERVER),
        "--model", str(MODEL_PATH),
        "--host", LLM_HOST,
        "--port", str(LLM_PORT),
        "--ctx-size", str(CTX_SIZE),
        "--threads", str(THREADS),
        "--log-disable",
    ]

    logger.info(f"Starting LLM server on {LLM_HOST}:{LLM_PORT}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


async def wait_for_server(host: str, port: int, timeout: float = 30) -> bool:
    """Wait until the LLM server responds to health checks."""
    import aiohttp

    url = f"http://{host}:{port}/health"
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
        except (aiohttp.ClientError, OSError):
            pass
        await asyncio.sleep(0.5)

    return False


async def run():
    global llm_process

    print("=" * 50)
    print("  JARVIS Launcher")
    print("=" * 50)

    # Step 1: Start LLM server
    llm_process = start_llm_server()
    if llm_process is None:
        print("[FAIL] Could not start LLM server.")
        return

    print(f"  LLM server PID: {llm_process.pid}")
    print("  Waiting for server to be ready...")

    ready = await wait_for_server(LLM_HOST, LLM_PORT)
    if not ready:
        print("[FAIL] LLM server did not start within 30s.")
        llm_process.terminate()
        return

    print("  [OK] LLM server is ready.\n")

    # Step 2: Start JARVIS
    from jarvis import Jarvis

    jarvis = Jarvis()

    # Parse args
    no_wake = "--no-wake" in sys.argv
    if no_wake:
        jarvis.wake_word_enabled = False
        print("  Mode: No wake word (direct voice input)")
    else:
        print(f"  Mode: Wake word (\"{jarvis.wake_word}\")")

    print("  Press Ctrl+C to stop.\n")

    try:
        await jarvis.run()
    except KeyboardInterrupt:
        pass
    finally:
        await jarvis.shutdown()
        if llm_process and llm_process.poll() is None:
            logger.info("Stopping LLM server...")
            llm_process.terminate()
            try:
                llm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                llm_process.kill()
            logger.info("LLM server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nJARVIS stopped.")
        if llm_process and llm_process.poll() is None:
            llm_process.terminate()
