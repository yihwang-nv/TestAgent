#!/usr/bin/env bash
# =============================================================================
# setup_trtllm.sh — isolated venv for TensorRT-LLM (no vLLM in this env)
# Usage: bash setup_trtllm.sh
# Optional: TRTLLM_PIP_SPEC='tensorrt-llm==1.3.0rc9' bash setup_trtllm.sh
#           (use a 1.3 rc from PyPI + NVIDIA index for Qwen3-Next / newer Qwen3.5 — check NV docs)
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv_trtllm"

echo "======================================================"
echo " TensorRT-LLM venv: $VENV_DIR"
echo " (Separate from .venv used by vLLM)"
echo "======================================================"

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv: $VENV_DIR"
else
    echo "Venv already exists: $VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install -U pip wheel
NV_INDEX="--extra-index-url https://pypi.nvidia.com"
if [[ -n "${TRTLLM_PIP_SPEC:-}" ]]; then
    echo "Installing TensorRT-LLM from TRTLLM_PIP_SPEC: $TRTLLM_PIP_SPEC"
    pip install "$TRTLLM_PIP_SPEC" "pyyaml>=6.0" $NV_INDEX
else
    pip install -r "$PROJECT_DIR/requirements_trtllm.txt" $NV_INDEX
fi

echo ""
TRT_VER=$(python -c "import tensorrt_llm as t; print(getattr(t, '__version__', 'unknown'))" 2>/dev/null) || true
[[ -z "${TRT_VER:-}" ]] && TRT_VER="unknown"
echo "tensorrt_llm: $TRT_VER"
if command -v trtllm-serve &>/dev/null; then
    echo "trtllm-serve: $(command -v trtllm-serve)"
else
    echo "NOTE: trtllm-serve not on PATH — check TensorRT-LLM install / entry points."
fi
echo "======================================================"
echo " Done."
if [[ "$TRT_VER" == 1.2.* ]]; then
    echo " NOTE: Qwen3.5 (qwen3_5) 需要新版 transformers；当前 1.2 栈钉死 4.57.3 — 此类权重请用 vLLM。"
elif [[ "$TRT_VER" == 1.3.* ]]; then
    echo " NOTE: 1.3.x（含 rc9）PyPI 包仍依赖 transformers==4.57.3，qwen3_5 在 TRT 下通常仍不可用；请用 vLLM 跑 Qwen3.5。"
else
    echo " NOTE: 官方 tensorrt-llm wheel 多钉 transformers 4.57.3；qwen3_5 请优先 vLLM。qwen3_next 见 NVIDIA recipes。"
fi
echo "       若 import 时出现 parakeet / auto_docstring 提示，一般来自 TensorRT-LLM 自带脚本，可忽略。"
echo "======================================================"
