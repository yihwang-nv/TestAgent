#!/usr/bin/env bash
# =============================================================================
# start_trtllm_server.sh — Start the Qwen3.5-9B TensorRT-LLM inference server
# Exposes an OpenAI-compatible API on http://0.0.0.0:PORT
#
# Usage: bash start_trtllm_server.sh [--port 8080] [--quant none|int8|int4_awq]
#        bash start_trtllm_server.sh --build-only [--quant int8]
#
# First run builds the TRT-LLM engine (~15–30 min) and caches it.
# Subsequent starts load the cached engine in seconds.
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
BUILD_ONLY=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       PORT="$2";           shift 2 ;;
        --quant)      QUANT_OVERRIDE="$2"; shift 2 ;;
        --build-only) BUILD_ONLY=true;     shift   ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
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
# Use version string detection instead of exit code: modelopt's failed vllm-plugin
# import emits a non-fatal UserWarning that causes a non-zero exit even though
# tensorrt_llm itself imported successfully.
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

dtype    = c["model"].get("torch_dtype", "bfloat16")
max_seq  = str(tc.get("max_seq_len", 8192))
tp_size  = str(tc.get("tensor_parallel_size", 1))
eng_dir  = tc.get("engine_dir", "")
model_dir = c["model"]["local_dir"]

# Resolve engine dir: use config value or derive default
if not eng_dir:
    import os
    eng_dir = os.path.join("$PROJECT_DIR", "engines", f"qwen3.5-9b-{quant}-{dtype}")

print(model_dir)
print(quant)
print(dtype)
print(max_seq)
print(tp_size)
print(eng_dir)
EOF
}

mapfile -t CFG < <(read_config)
MODEL_DIR="${CFG[0]}"
QUANT="${QUANT_OVERRIDE:-${CFG[1]}}"
DTYPE="${CFG[2]}"
MAX_SEQ="${CFG[3]}"
TP_SIZE="${CFG[4]}"
ENGINE_DIR="${CFG[5]}"

# Recompute engine dir if quant was overridden on CLI
if [[ -n "$QUANT_OVERRIDE" ]]; then
    ENGINE_DIR="$PROJECT_DIR/engines/qwen3.5-9b-${QUANT}-${DTYPE}"
fi

# ── Check model exists ────────────────────────────────────────────────────────
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "       Run: python download_model.py"
    exit 1
fi

# ── Print summary ─────────────────────────────────────────────────────────────
echo "======================================================"
echo " Qwen3.5-9B Inference Server (TensorRT-LLM)"
echo " URL     : http://0.0.0.0:$PORT"
echo " Model   : $MODEL_DIR"
echo " Engine  : $ENGINE_DIR"
echo " Quant   : $QUANT"
echo " dtype   : $DTYPE"
echo " Context : $MAX_SEQ tokens"
echo " TP size : $TP_SIZE"
echo "======================================================"

# ── Build engine if not cached ────────────────────────────────────────────────
if [[ ! -d "$ENGINE_DIR" ]]; then
    echo ""
    echo " No cached engine found at $ENGINE_DIR"
    echo " Building now — this takes 15–30 min on first run."
    echo " Subsequent starts will load the cached engine in seconds."
    echo "======================================================"
    echo ""
    python "$PROJECT_DIR/build_trtllm_engine.py" --quant "$QUANT"
else
    echo " Engine cache : FOUND (skipping build)"
    echo "======================================================"
fi

if $BUILD_ONLY; then
    echo ""
    echo "Engine built. Run without --build-only to start the server."
    exit 0
fi

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
