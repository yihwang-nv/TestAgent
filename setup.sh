#!/usr/bin/env bash
# =============================================================================
# setup.sh — Full project setup
# Usage: bash setup.sh [--skip-download] [--model-size auto|0.8b|2b|4b|9b|27b|35b-a3b|all]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SKIP_DOWNLOAD=false
MODEL_SIZE="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --model-size)    MODEL_SIZE="$2"; shift 2 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================"
echo " Qwen3.5 Reasoning Distilled — Project Setup"
echo " Inference: vLLM"
echo " Workspace: $PROJECT_DIR"
echo "======================================================"

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

echo ""
echo "[2/4] Setting up Python virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  Already exists: $VENV_DIR (skipping creation)"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

echo ""
echo "[3/4] Installing vLLM and dependencies..."
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || true)
if [[ -n "$TORCH_VER" && "$TORCH_VER" != "2.10."* ]]; then
    echo "  Found torch $TORCH_VER — uninstalling to let vLLM pin the correct version..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
fi
pip install -r "$PROJECT_DIR/requirements.txt" --extra-index-url https://pypi.nvidia.com
echo "  Done."
python -c "import vllm; print(f'  vLLM {vllm.__version__} installed')"

echo ""
if $SKIP_DOWNLOAD; then
    echo "[4/4] Skipping model download (--skip-download)."
    echo "      Run: python download_model.py --model-size auto|SIZE|all"
else
    echo "[4/4] Downloading model (Ctrl-C to skip)..."
    python "$PROJECT_DIR/download_model.py" --model-size "$MODEL_SIZE" || {
        echo ""
        echo "  Download interrupted or failed."
        echo "  Resume: python download_model.py --model-size $MODEL_SIZE"
    }
fi

echo ""
echo "[4b] Patching tokenizer_config.json under models/..."
python - "$PROJECT_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "models"
if not root.is_dir():
    print("  No models/ directory yet — skipped.")
    raise SystemExit(0)

patched = 0
for tok_cfg in root.glob("*/tokenizer_config.json"):
    data = json.loads(tok_cfg.read_text())
    old_cls = data.get("tokenizer_class", "")
    if old_cls in ("TokenizersBackend", ""):
        data["tokenizer_class"] = "Qwen2TokenizerFast"
        tok_cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Patched {tok_cfg.parent.name}: tokenizer_class -> Qwen2TokenizerFast")
        patched += 1
    else:
        print(f"  OK {tok_cfg.parent.name}: {old_cls!r}")
if patched == 0 and not list(root.glob("*/tokenizer_config.json")):
    print("  No tokenizer_config.json found (models not downloaded).")
PYEOF
echo "  Done."

echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo "  bash start_server.sh [--model-size auto] [--quant auto]"
echo "  python download_model.py --model-size all   # all sizes"
echo ""
