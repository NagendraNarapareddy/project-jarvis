#!/bin/bash
# Download whisper.cpp tiny.en q5_1 model (~32MB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_FILE="ggml-tiny.en-q5_1.bin"
MODEL_PATH="$SCRIPT_DIR/$MODEL_FILE"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin"

echo "============================================"
echo "  JARVIS - STT Model Downloader"
echo "============================================"

if [ -f "$MODEL_PATH" ]; then
    echo "[INFO] Model already exists: $MODEL_PATH"
    exit 0
fi

echo "[INFO] Downloading $MODEL_FILE (~32MB)..."

if command -v wget &> /dev/null; then
    wget -c -O "$MODEL_PATH" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -C - -o "$MODEL_PATH" "$MODEL_URL"
else
    echo "[ERROR] Neither wget nor curl found."
    exit 1
fi

echo "[SUCCESS] Model downloaded: $MODEL_PATH"
