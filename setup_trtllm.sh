#!/usr/bin/env bash
# =============================================================================
# setup_trtllm.sh — Set up isolated TensorRT-LLM virtual environment
#
# Creates .venv-trtllm (separate from .venv which is used by vLLM).
# Keeping them separate avoids torch ABI conflicts between vLLM and TRT-LLM.
#
# Usage: bash setup_trtllm.sh [--skip-download]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-trtllm"
SKIP_DOWNLOAD=false
[[ "${1:-}" == "--skip-download" ]] && SKIP_DOWNLOAD=true

echo "======================================================"
echo " Qwen3.5-9B — TensorRT-LLM Environment Setup"
echo " Venv    : $VENV_DIR"
echo " (separate from .venv used by vLLM)"
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

# ── 2. Python venv ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Setting up .venv-trtllm..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  Already exists: $VENV_DIR (skipping creation)"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel --quiet

# ── 3. Install TensorRT-LLM + server deps ─────────────────────────────────────
# TRT-LLM bundles its own torch/transformers — do NOT install vLLM here.
echo ""
echo "[3/4] Installing TensorRT-LLM and dependencies..."
echo "  (TRT-LLM is large — first install may take several minutes)"
pip install -r "$PROJECT_DIR/requirements_trtllm.txt" --quiet
echo "  Done."

# Verify TRT-LLM loaded (use version string check — modelopt causes non-zero exit)
TRTLLM_VER=$(python -W ignore -c "import tensorrt_llm" 2>&1 \
    | grep -oP "TensorRT LLM version: \K[^\s]+" || true)
if [[ -n "$TRTLLM_VER" ]]; then
    echo "  TensorRT-LLM $TRTLLM_VER installed successfully."
else
    echo "  WARNING: Could not confirm TensorRT-LLM version — check install."
fi

# ── 4. Patch tokenizer (idempotent, safe to re-run) ───────────────────────────
echo ""
echo "[4/4] Patching tokenizer files..."
python "$PROJECT_DIR/fix_tokenizer.py" && echo "  Done." \
    || echo "  Skipped (model not yet downloaded — run python download_model.py)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " TensorRT-LLM setup complete!"
echo "======================================================"
echo ""
echo " Next steps:"
echo "   bash start_trtllm_server.sh            # build engine + start server"
echo "   bash start_trtllm_server.sh --build-only   # pre-build engine only"
echo "   bash start_agent.sh --port 8080        # agent REPL (new terminal)"
echo ""
echo " Quantization (override in config.yaml → tensorrt_llm.quantization):"
echo "   int8      ~10 GB VRAM  ← default (SmoothQuant)"
echo "   int4_awq  ~5  GB VRAM  (AWQ, fastest)"
echo "   none      ~19 GB VRAM  (full bf16)"
echo ""
