#!/bin/bash
# Download Piper TTS voice model (~63MB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VOICE_FILE="en_US-lessac-medium.onnx"
CONFIG_FILE="en_US-lessac-medium.onnx.json"

VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$VOICE_FILE"
CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$CONFIG_FILE"

echo "============================================"
echo "  JARVIS - TTS Voice Downloader"
echo "============================================"

download() {
    local url="$1" dest="$2"
    if command -v wget &> /dev/null; then
        wget -c -O "$dest" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$dest" "$url"
    else
        echo "[ERROR] Neither wget nor curl found."
        exit 1
    fi
}

if [ -f "$SCRIPT_DIR/$VOICE_FILE" ] && [ -f "$SCRIPT_DIR/$CONFIG_FILE" ]; then
    echo "[INFO] Voice model already exists. Skipping."
    exit 0
fi

echo "[INFO] Downloading voice model (~63MB)..."
download "$VOICE_URL" "$SCRIPT_DIR/$VOICE_FILE"

echo "[INFO] Downloading voice config..."
download "$CONFIG_URL" "$SCRIPT_DIR/$CONFIG_FILE"

echo "[SUCCESS] TTS voice downloaded."
