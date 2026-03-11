#!/bin/bash
# Acquire Termux wake lock to keep JARVIS alive in background

set -euo pipefail

if command -v termux-wake-lock &> /dev/null; then
    termux-wake-lock
    echo "[INFO] Wake lock acquired."
else
    echo "[WARN] termux-wake-lock not available. Install termux-api package."
fi
