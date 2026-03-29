#!/usr/bin/env bash
# =============================================================================
# start_server.sh — OpenAI-compatible inference (vLLM or TensorRT-LLM)
#
# Usage:
#   bash start_server.sh [--port 8080] [--engine vllm|tensorrt_llm]
#          [--model-size auto|0.8b|2b|4b|9b|27b|35b-a3b] [--quant auto|none|8bit|4bit|awq]
#
# vLLM (.venv): none / 8bit / awq
# TensorRT-LLM (.venv_trtllm, setup_trtllm.sh): none / 4bit / 8bit / awq (+ tensorrt_llm.* in config.yaml)
# Optional: GPU_SELECT_MAX_MODEL_LEN=8192  覆盖 gpu_select 按显存推算的 --max-model-len（OOM 风险自负）
# Optional: VLLM_TOOL_CALL_PARSER=qwen3_xml  强制工具解析器（覆盖 config.yaml；日志若出现 hermes_tool_parser 说明曾用错解析器）
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_VLLM="$PROJECT_DIR/.venv/bin/activate"
PORT=8080
MODEL_SIZE="auto"
QUANT="auto"
ENGINE_CLI=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)        PORT="$2"; shift 2 ;;
        --model-size)  MODEL_SIZE="$2"; shift 2 ;;
        --quant)       QUANT="$2"; shift 2 ;;
        --engine)      ENGINE_CLI="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$VENV_VLLM" ]]; then
    echo "ERROR: vLLM venv not found (.venv). Run: bash setup.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_VLLM"

# Must match config.yaml vllm.tool_call_parser (server.py already reads YAML).
TOOL_CALL_PARSER="$(python -c "
import pathlib
try:
    import yaml
    cfg = yaml.safe_load((pathlib.Path('${PROJECT_DIR}') / 'config.yaml').read_text())
    p = str((cfg.get('vllm') or {}).get('tool_call_parser') or 'qwen3_xml').strip()
    print(p or 'qwen3_xml')
except Exception:
    print('qwen3_xml')
")"

if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER}"
    echo "NOTE: VLLM_TOOL_CALL_PARSER override → tool_call_parser=$TOOL_CALL_PARSER"
fi

GPU_CMD=(python "$PROJECT_DIR/gpu_select.py" --model-size "$MODEL_SIZE" --quant "$QUANT")
[[ -n "$ENGINE_CLI" ]] && GPU_CMD+=(--engine "$ENGINE_CLI")

export _GPU_SELECT_JSON
_GPU_SELECT_JSON=$("${GPU_CMD[@]}") || {
    echo "ERROR: gpu_select.py failed"
    exit 1
}

mapfile -t SEL < <(python -c "
import json, os
j = json.loads(os.environ['_GPU_SELECT_JSON'])
for k in (
    'engine', 'local_dir', 'quantization', 'torch_dtype', 'max_model_len',
    'gpu_memory_utilization', 'tensor_parallel_size', 'served_model_name',
    'hf_repo', 'model_size', 'vram_gb', 'gpu_name', 'trt_venv_python',
):
    print(j.get(k, ''))
")
unset _GPU_SELECT_JSON

ENGINE="${SEL[0]}"
MODEL_DIR="${SEL[1]}"
QUANT_RESOLVED="${SEL[2]}"
DTYPE="${SEL[3]}"
MAX_MODEL_LEN="${SEL[4]}"
GPU_MEM_UTIL="${SEL[5]}"
TP_SIZE="${SEL[6]}"
MODEL_NAME="${SEL[7]}"
HF_REPO="${SEL[8]}"
RESOLVED_SIZE="${SEL[9]}"
VRAM_GB="${SEL[10]}"
GPU_NAME="${SEL[11]}"
TRT_PY="${SEL[12]}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "       Run: python download_model.py --model-size ${RESOLVED_SIZE:-auto}"
    exit 1
fi

if [[ "$ENGINE" == "tensorrt_llm" ]]; then
    if [[ ! -x "$TRT_PY" ]]; then
        echo "ERROR: TensorRT-LLM interpreter not found: $TRT_PY"
        echo "       Run: bash setup_trtllm.sh"
        exit 1
    fi
    echo "======================================================"
    echo " Local LLM — TensorRT-LLM (trtllm-serve)"
    echo "------------------------------------------------------"
    echo " 启动参数 (CLI)"
    echo "   --engine      ${ENGINE_CLI:-<config default>}  →  $ENGINE"
    echo "   --model-size  $MODEL_SIZE  →  $RESOLVED_SIZE"
    echo "   --quant       $QUANT  →  $QUANT_RESOLVED"
    echo "   --port        $PORT"
    echo "------------------------------------------------------"
    echo " GPU"
    echo "   设备          : $GPU_NAME"
    echo "   显存          : ${VRAM_GB} GB"
    echo "------------------------------------------------------"
    echo " 模型"
    echo "   HuggingFace   : $HF_REPO"
    echo "   本地目录      : $MODEL_DIR"
    echo "   API 名称      : $MODEL_NAME"
    echo "   量化          : $QUANT_RESOLVED"
    echo "   max_model_len : $MAX_MODEL_LEN (see tensorrt_llm.max_num_tokens)"
    echo "------------------------------------------------------"
    echo " TRT venv       : $TRT_PY"
    echo " URL            : http://0.0.0.0:$PORT"
    echo "======================================================"
    echo ""
    exec "$TRT_PY" "$PROJECT_DIR/trtllm_server.py" \
        --model-size "$MODEL_SIZE" --quant "$QUANT" --port "$PORT" --host "0.0.0.0"
fi

# ── vLLM ─────────────────────────────────────────────────────────────────────
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
    --tokenizer-mode slow
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
)

case "$QUANT_RESOLVED" in
    8bit)
        VLLM_ARGS+=(--quantization bitsandbytes --load-format bitsandbytes)
        ;;
    awq)
        VLLM_ARGS+=(--quantization awq)
        ;;
    none|"")
        ;;
    4bit)
        echo "ERROR: vLLM path does not use 4bit here. Use: --engine tensorrt_llm --quant 4bit"
        exit 1
        ;;
    *)
        echo "WARNING: Unknown quantization '$QUANT_RESOLVED', running without quant."
        ;;
esac

echo "======================================================"
echo " Qwen3.5 — vLLM"
echo "------------------------------------------------------"
echo " 启动参数 (CLI)"
echo "   --engine      ${ENGINE_CLI:-<config default>}  →  $ENGINE"
echo "   --model-size  $MODEL_SIZE  →  $RESOLVED_SIZE"
echo "   --quant       $QUANT  →  $QUANT_RESOLVED"
echo "   --port        $PORT"
echo "------------------------------------------------------"
echo " GPU"
echo "   设备          : $GPU_NAME"
echo "   显存          : ${VRAM_GB} GB"
echo "------------------------------------------------------"
echo " 模型与 vLLM"
echo "   HuggingFace   : $HF_REPO"
echo "   本地目录      : $MODEL_DIR"
echo "   API 名称      : $MODEL_NAME"
echo "   dtype         : $DTYPE"
echo "   量化          : $QUANT_RESOLVED"
echo "   max_len       : $MAX_MODEL_LEN"
echo "   gpu_mem_util  : $GPU_MEM_UTIL"
echo "   tensor_para   : $TP_SIZE"
echo "   tool_parser   : $TOOL_CALL_PARSER  (config vllm.tool_call_parser)"
echo "------------------------------------------------------"
echo " URL            : http://0.0.0.0:$PORT"
echo "======================================================"
echo ""

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
exec vllm serve "${VLLM_ARGS[@]}"
