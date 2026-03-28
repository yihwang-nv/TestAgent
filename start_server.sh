#!/usr/bin/env bash
# =============================================================================
# start_server.sh — Start the Qwen3.5-9B vLLM inference server
# Exposes an OpenAI-compatible API on http://0.0.0.0:PORT
#
# Usage: bash start_server.sh [--port 8080] [--quant none|8bit|awq]
#
# Endpoints (provided by vllm serve):
#   GET  /health
#   GET  /v1/models
#   POST /v1/chat/completions   (streaming + non-streaming, OpenAI-compatible)
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"
PORT=8080
QUANT_OVERRIDE=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  PORT="$2";          shift 2 ;;
        --quant) QUANT_OVERRIDE="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Check venv ────────────────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source "$VENV"

# ── Read config ───────────────────────────────────────────────────────────────
read_config() {
    python - <<EOF
import yaml, sys
c = yaml.safe_load(open("$PROJECT_DIR/config.yaml"))
print(c['model']['local_dir'])
print(c['model'].get('quantization', 'none'))
print(c['model'].get('torch_dtype', 'bfloat16'))
print(str(c['vllm'].get('max_model_len', 8192)))
print(str(c['vllm'].get('gpu_memory_utilization', 0.90)))
print(str(c['vllm'].get('tensor_parallel_size', 1)))
print(c['vllm'].get('served_model_name', 'qwen3.5-9b'))
EOF
}

mapfile -t CFG < <(read_config)
MODEL_DIR="${CFG[0]}"
QUANT="${QUANT_OVERRIDE:-${CFG[1]}}"
DTYPE="${CFG[2]}"
MAX_MODEL_LEN="${CFG[3]}"
GPU_MEM_UTIL="${CFG[4]}"
TP_SIZE="${CFG[5]}"
MODEL_NAME="${CFG[6]}"

# ── Check model exists ────────────────────────────────────────────────────────
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "       Run: python download_model.py"
    exit 1
fi

# ── Build vllm serve args ─────────────────────────────────────────────────────
VLLM_ARGS=(
    "$MODEL_DIR"
    --dtype          "$DTYPE"
    --max-model-len  "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size   "$TP_SIZE"
    --served-model-name      "$MODEL_NAME"
    --port           "$PORT"
    --host           "0.0.0.0"
    --trust-remote-code
)

# Quantization
case "$QUANT" in
    8bit)
        VLLM_ARGS+=(--quantization bitsandbytes --load-format bitsandbytes)
        ;;
    awq)
        VLLM_ARGS+=(--quantization awq)
        ;;
    none|"")
        # full bf16, no extra flags
        ;;
    *)
        echo "WARNING: Unknown quantization '$QUANT', running without quant."
        ;;
esac

# ── Print summary ─────────────────────────────────────────────────────────────
echo "======================================================"
echo " Qwen3.5-9B Inference Server (vLLM)"
echo " URL     : http://0.0.0.0:$PORT"
echo " Model   : $MODEL_DIR"
echo " Quant   : $QUANT"
echo " dtype   : $DTYPE"
echo " Context : $MAX_MODEL_LEN tokens"
echo " GPU mem : ${GPU_MEM_UTIL} utilization"
echo "======================================================"
echo " Endpoints:"
echo "   GET  /health"
echo "   GET  /v1/models"
echo "   POST /v1/chat/completions  (streaming + non-streaming)"
echo ""
echo " Press Ctrl-C to stop."
echo "======================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
exec vllm serve "${VLLM_ARGS[@]}"
