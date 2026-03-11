#!/bin/bash
# ============================================
#  JARVIS - Full Termux Setup Script
#  Installs everything from scratch on a
#  fresh Termux install.
# ============================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  JARVIS - Termux Setup"
echo "============================================"
echo "  Project dir: $PROJECT_DIR"
echo ""

# ── Step 1: System packages ─────────────────
echo "[1/7] Installing system packages..."
pkg update -y && pkg upgrade -y
pkg install -y \
    python git clang cmake make \
    ffmpeg termux-api portaudio \
    wget pulseaudio

# ── Step 2: Python packages ─────────────────
echo "[2/7] Installing Python packages..."
pip install --break-system-packages \
    pyaudio requests pyyaml aiohttp numpy

# ── Step 3: Compile llama.cpp ────────────────
echo "[3/7] Setting up llama.cpp..."
if [ ! -f "$PROJECT_DIR/llama.cpp/llama-server" ]; then
    if [ ! -d "$PROJECT_DIR/llama.cpp" ]; then
        cd "$PROJECT_DIR"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    fi
    cd "$PROJECT_DIR/llama.cpp"
    make LLAMA_ANDROID=1 -j4
    echo "[OK] llama.cpp compiled."
else
    echo "[OK] llama.cpp already compiled."
fi

# ── Step 4: Compile whisper.cpp ──────────────
echo "[4/7] Setting up whisper.cpp..."
if [ ! -f "$PROJECT_DIR/whisper.cpp/main" ]; then
    if [ ! -d "$PROJECT_DIR/whisper.cpp" ]; then
        cd "$PROJECT_DIR"
        git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
    fi
    cd "$PROJECT_DIR/whisper.cpp"
    make -j4
    echo "[OK] whisper.cpp compiled."
else
    echo "[OK] whisper.cpp already compiled."
fi

# ── Step 5: Download Piper TTS ───────────────
echo "[5/7] Setting up Piper TTS..."
PIPER_DIR="$PROJECT_DIR/piper"
if [ ! -f "$PIPER_DIR/piper" ]; then
    mkdir -p "$PIPER_DIR"
    cd "$PIPER_DIR"
    # Download ARM64 Linux build
    PIPER_VERSION="2023.11.14-2"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
    echo "  Downloading Piper from: $PIPER_URL"
    wget -q -O piper.tar.gz "$PIPER_URL"
    tar xzf piper.tar.gz --strip-components=1
    rm -f piper.tar.gz
    chmod +x piper
    echo "[OK] Piper installed."
else
    echo "[OK] Piper already installed."
fi

# ── Step 6: Download models ──────────────────
echo "[6/7] Downloading models..."
cd "$PROJECT_DIR"
bash models/download_llm.sh
bash models/download_stt.sh
bash models/download_tts.sh

# ── Step 7: Setup boot script ────────────────
echo "[7/7] Setting up auto-start on boot..."
BOOT_DIR="$HOME/.termux/boot"
mkdir -p "$BOOT_DIR"

cat > "$BOOT_DIR/start-jarvis.sh" << 'BOOTEOF'
#!/bin/bash
# JARVIS Auto-Start on Boot

PROJECT_DIR="$HOME/project-jarvis"
LOG_FILE="$PROJECT_DIR/jarvis.log"
BACKOFF=5

# Acquire wake lock
termux-wake-lock

# Show status notification
termux-notification \
    --id jarvis-status \
    --title "JARVIS" \
    --content "ONLINE - Starting up..." \
    --ongoing \
    --priority high

cd "$PROJECT_DIR"

# Auto-restart loop with backoff
while true; do
    echo "$(date): Starting JARVIS..." >> "$LOG_FILE"

    termux-notification \
        --id jarvis-status \
        --title "JARVIS" \
        --content "ONLINE" \
        --ongoing \
        --priority high

    python start.py --no-wake >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): JARVIS exited normally." >> "$LOG_FILE"
        break
    fi

    echo "$(date): JARVIS crashed (exit $EXIT_CODE). Restarting in ${BACKOFF}s..." >> "$LOG_FILE"

    termux-notification \
        --id jarvis-status \
        --title "JARVIS" \
        --content "ERROR - Restarting in ${BACKOFF}s..." \
        --ongoing \
        --priority high

    sleep $BACKOFF
done

termux-notification \
    --id jarvis-status \
    --title "JARVIS" \
    --content "OFFLINE" \
    --priority low
BOOTEOF

chmod +x "$BOOT_DIR/start-jarvis.sh"
echo "[OK] Boot script installed."

echo ""
echo "============================================"
echo "  JARVIS Setup Complete!"
echo "============================================"
echo ""
echo "  To start manually:"
echo "    cd $PROJECT_DIR"
echo "    python start.py --no-wake"
echo ""
echo "  JARVIS will auto-start on next boot."
echo "============================================"
