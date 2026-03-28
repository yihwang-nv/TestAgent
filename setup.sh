#!/usr/bin/env bash
# =============================================================================
# setup.sh — Full project setup
# Run once before anything else.
# Usage: bash setup.sh [--skip-download]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SKIP_DOWNLOAD=false
[[ "${1:-}" == "--skip-download" ]] && SKIP_DOWNLOAD=true

echo "======================================================"
echo " Qwen3.5-9B Reasoning Distilled — Project Setup"
echo " Workspace: $PROJECT_DIR"
echo "======================================================"

# ── 1. GPU check ──────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | awk '{print "  GPU: "$0}'

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Setting up Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  Already exists: $VENV_DIR (skipping creation)"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel --quiet

# ── 3. PyTorch (CUDA 12.8 — Blackwell SM_100) ────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch for CUDA 12.8 (Blackwell RTX 5070 / SM_100)..."
if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    echo "  PyTorch with CUDA already installed — skipping"
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 --quiet
    echo "  Done."
fi

# ── 4. Project dependencies ───────────────────────────────────────────────────
echo ""
echo "[4/5] Installing project dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "  Done."

# ── 5. Download model ─────────────────────────────────────────────────────────
echo ""
if $SKIP_DOWNLOAD; then
    echo "[5/5] Skipping model download (--skip-download passed)."
    echo "      Run manually: python download_model.py"
else
    echo "[5/5] Downloading model (~19 GB — Ctrl-C to skip and do it later)..."
    python "$PROJECT_DIR/download_model.py" || {
        echo ""
        echo "  Download interrupted or failed."
        echo "  Resume later with: python download_model.py"
    }
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo " Next:"
echo "   bash start_server.sh        # start inference server"
echo "   bash start_agent.sh         # start agent REPL (new terminal)"
echo ""
echo " Quantization (edit config.yaml → model.quantization):"
echo "   4bit  ~5  GB VRAM  ← default"
echo "   8bit  ~10 GB VRAM  ← better quality"
