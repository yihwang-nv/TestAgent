#!/usr/bin/env bash
# =============================================================================
# start_server.sh — Start the Qwen3.5-9B inference server
# Exposes an OpenAI-compatible API on http://0.0.0.0:PORT
# Usage: bash start_server.sh [--port 8080] [--quant 4bit|8bit]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"
PORT=8080
QUANT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  PORT="$2";  shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check venv
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source "$VENV"

# Check model exists
MODEL_DIR=$(python -c "
import yaml
with open('$PROJECT_DIR/config.yaml') as f:
    c = yaml.safe_load(f)
print(c['model']['local_dir'])
")
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "       Run: python download_model.py"
    exit 1
fi

# Override quantization in config if --quant passed
if [[ -n "$QUANT" ]]; then
    python - <<EOF
import yaml, re
path = "$PROJECT_DIR/config.yaml"
with open(path) as f:
    text = f.read()
text = re.sub(r'(quantization:\s*")[^"]*(")', r'\g<1>$QUANT\g<2>', text)
with open(path, "w") as f:
    f.write(text)
print(f"  config.yaml → quantization: $QUANT")
EOF
fi

echo "======================================================"
echo " Qwen3.5-9B Inference Server"
echo " URL  : http://0.0.0.0:$PORT"
echo " Model: $MODEL_DIR"
echo " Quant: $(python -c "import yaml; c=yaml.safe_load(open('$PROJECT_DIR/config.yaml')); print(c['model']['quantization'])")"
echo "======================================================"
echo " Endpoints:"
echo "   GET  /health"
echo "   GET  /v1/models"
echo "   POST /v1/chat/completions  (streaming + non-streaming)"
echo ""
echo " Press Ctrl-C to stop."
echo "======================================================"
echo ""

python "$PROJECT_DIR/server.py" --host 0.0.0.0 --port "$PORT"
