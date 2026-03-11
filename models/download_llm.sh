#!/bin/bash
# Download Llama-3.2-1B-Instruct Q4_K_M GGUF model (~700MB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_FILE="llama-3.2-1b-instruct-q4_k_m.gguf"
MODEL_PATH="$SCRIPT_DIR/$MODEL_FILE"
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

echo "============================================"
echo "  JARVIS - LLM Model Downloader"
echo "============================================"

if [ -f "$MODEL_PATH" ]; then
    echo "[INFO] Model already exists: $MODEL_PATH"
    exit 0
fi

echo "[INFO] Downloading $MODEL_FILE (~700MB)..."

if command -v wget &> /dev/null; then
    wget -c -O "$MODEL_PATH" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -C - -o "$MODEL_PATH" "$MODEL_URL"
else
    echo "[ERROR] Neither wget nor curl found."
    exit 1
fi

echo "[SUCCESS] Model downloaded: $MODEL_PATH"
