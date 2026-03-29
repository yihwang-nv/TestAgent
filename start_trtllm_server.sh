#!/usr/bin/env bash
# =============================================================================
# start_trtllm_server.sh — Start the Qwen3.5-9B TensorRT-LLM inference server
#
# Uses the TRT-LLM PyTorch backend — loads the HF model directly, no engine
# compilation required.  Start time is comparable to vLLM.
#
# Usage: bash start_trtllm_server.sh [--port 8080] [--quant none|int8|int4_awq]
#
# Endpoints (provided by tensorrt_llm_server.py):
#   GET  /health
#   GET  /v1/models
#   POST /v1/chat/completions   (streaming + non-streaming, OpenAI-compatible)
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv-trtllm/bin/activate"
PORT=8080
QUANT_OVERRIDE=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  PORT="$2";           shift 2 ;;
        --quant) QUANT_OVERRIDE="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Check venv ────────────────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: TRT-LLM virtual environment not found."
    echo "       Run: bash setup_trtllm.sh"
    exit 1
fi
source "$VENV"

# ── Check tensorrt_llm is installed ──────────────────────────────────────────
# Use version string detection: modelopt vllm-plugin import failure causes a
# non-zero exit even though tensorrt_llm itself imported successfully.
TRTLLM_CHECK=$(python -W ignore -c "import tensorrt_llm" 2>&1 || true)
TRTLLM_VER=$(echo "$TRTLLM_CHECK" | grep -oP "TensorRT LLM version: \K[^\s]+" || true)
if [[ -z "$TRTLLM_VER" ]]; then
    echo "ERROR: tensorrt_llm is not installed or failed to import."
    echo "       pip install tensorrt-llm"
    echo "       (requires CUDA >= 12.1, driver >= 525)"
    echo "       Import output: $TRTLLM_CHECK"
    exit 1
fi
echo "  TensorRT-LLM $TRTLLM_VER detected."

# ── Read config ───────────────────────────────────────────────────────────────
read_config() {
    python - <<EOF
import yaml
c = yaml.safe_load(open("$PROJECT_DIR/config.yaml"))
tc = c.get("tensorrt_llm", {})
# quant: normalise vLLM names to TRT-LLM names
raw_quant = tc.get("quantization", c["model"].get("quantization", "none"))
quant_map = {"8bit": "int8", "awq": "int4_awq", "4bit": "int4_awq"}
quant = quant_map.get(raw_quant, raw_quant)
model_dir = c["model"]["local_dir"]
print(model_dir)
print(quant)
EOF
}

mapfile -t CFG < <(read_config)
MODEL_DIR="${CFG[0]}"
QUANT="${QUANT_OVERRIDE:-${CFG[1]}}"

# ── Check model exists ────────────────────────────────────────────────────────
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "       Run: python download_model.py"
    exit 1
fi

# ── Print summary ─────────────────────────────────────────────────────────────
echo "======================================================"
echo " Qwen3.5-9B Inference Server (TensorRT-LLM PyTorch)"
echo " URL     : http://0.0.0.0:$PORT"
echo " Model   : $MODEL_DIR"
echo " Quant   : $QUANT"
echo " Backend : PyTorch (no engine compilation)"
echo "======================================================"
echo ""
echo " Endpoints:"
echo "   GET  /health"
echo "   GET  /v1/models"
echo "   POST /v1/chat/completions  (streaming + non-streaming)"
echo ""
echo " Press Ctrl-C to stop."
echo "======================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
exec python "$PROJECT_DIR/tensorrt_llm_server.py" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --quant "$QUANT"
