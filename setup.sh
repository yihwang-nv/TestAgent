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
echo " Inference engine: vLLM"
echo " Workspace: $PROJECT_DIR"
echo "======================================================"

# ── 1. GPU check ──────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | awk '{print "  GPU: "$0}'

CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  Driver: $CUDA_VER"

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Setting up Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  Already exists: $VENV_DIR (skipping creation)"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel --quiet

# ── 3. Install vLLM + project dependencies ────────────────────────────────────
# NOTE: vLLM bundles its own torch, transformers, fastapi, uvicorn.
#       Do NOT install torch separately — it will cause version conflicts.
echo ""
echo "[3/4] Installing vLLM and dependencies..."
echo "  (vLLM is large — first install may take a few minutes)"

# Check if we're on a Blackwell GPU (sm_100) — needs vLLM >= 0.6
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qiE "RTX (50|PRO 6000 Blackwell)"; then
    echo "  Detected Blackwell GPU ($GPU_NAME) — ensuring vLLM >= 0.6.0"
fi

pip install -r "$PROJECT_DIR/requirements.txt" #--quiet
echo "  Done."

# Verify vllm installed
python -c "import vllm; print(f'  vLLM {vllm.__version__} installed')"

# ── 4. Download model ─────────────────────────────────────────────────────────
echo ""
if $SKIP_DOWNLOAD; then
    echo "[4/4] Skipping model download (--skip-download passed)."
    echo "      Run manually: python download_model.py"
else
    echo "[4/4] Downloading model (~19 GB — Ctrl-C to skip and do it later)..."
    python "$PROJECT_DIR/download_model.py" || {
        echo ""
        echo "  Download interrupted or failed."
        echo "  Resume later with: python download_model.py"
    }
fi

# ── 4b. Patch tokenizer_config.json ──────────────────────────────────────────
# The uploaded model has "tokenizer_class": "TokenizersBackend" which doesn't
# exist in transformers. Patch it to "PreTrainedTokenizerFast" (correct class).
echo ""
echo "[4b] Patching tokenizer_config.json..."
python "$PROJECT_DIR/fix_tokenizer.py" && echo "  Done." || echo "  Skipped (model not yet downloaded)."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo " Next:"
echo "   bash start_server.sh        # start vLLM inference server"
echo "   bash start_agent.sh         # start agent REPL (new terminal)"
echo ""
echo " Quantization options (edit config.yaml → model.quantization):"
echo "   none  ~19 GB VRAM  ← default (full bf16, best quality)"
echo "   8bit  ~10 GB VRAM  (bitsandbytes, --quant 8bit)"
echo "   awq   ~5  GB VRAM  (needs pre-quantized AWQ model variant)"
echo ""
