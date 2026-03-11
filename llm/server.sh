#!/bin/bash
# Start llama-server for JARVIS LLM inference
# Serves on localhost:8080

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/llama-server"
MODEL_PATH="$PROJECT_DIR/models/llama-3.2-1b-instruct-q4_k_m.gguf"

# Server configuration
HOST="127.0.0.1"
PORT=8080
CTX_SIZE=2048
THREADS=4

echo "============================================"
echo "  JARVIS - LLM Server"
echo "============================================"

# Check if llama-server binary exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "[ERROR] llama-server not found at: $LLAMA_SERVER"
    echo "[INFO]  Run the following to compile:"
    echo "        cd $PROJECT_DIR/llama.cpp && make LLAMA_ANDROID=1 -j4"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at: $MODEL_PATH"
    echo "[INFO]  Run: bash $PROJECT_DIR/models/download_llm.sh"
    exit 1
fi

# Check if port is already in use
if command -v ss &> /dev/null; then
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo "[WARN] Port $PORT is already in use. Server may already be running."
        exit 1
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo "[WARN] Port $PORT is already in use."
        exit 1
    fi
fi

echo "[INFO] Model : $MODEL_PATH"
echo "[INFO] Host  : $HOST:$PORT"
echo "[INFO] Ctx   : $CTX_SIZE tokens"
echo "[INFO] Threads: $THREADS"
echo "[INFO] Starting server..."
echo ""

exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --log-disable
