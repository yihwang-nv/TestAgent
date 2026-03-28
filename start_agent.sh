#!/usr/bin/env bash
# =============================================================================
# start_agent.sh — Start the local agent REPL
# Connects to the inference server and starts an interactive session.
# Usage: bash start_agent.sh [--port 8080] [--hide-thinking] [--prompt "..."]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"
PORT=8080
EXTRA_ARGS=""
PROMPT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)          PORT="$2"; shift 2 ;;
        --hide-thinking) EXTRA_ARGS="$EXTRA_ARGS --hide-thinking"; shift ;;
        --prompt)        PROMPT="$2"; shift 2 ;;
        --temperature)   EXTRA_ARGS="$EXTRA_ARGS --temperature $2"; shift 2 ;;
        --max-tokens)    EXTRA_ARGS="$EXTRA_ARGS --max-tokens $2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SERVER_URL="http://localhost:$PORT"

# Check venv
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source "$VENV"

# Wait for server to be ready (up to 60 s)
echo "Waiting for server at $SERVER_URL ..."
for i in $(seq 1 30); do
    if curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
        MODEL=$(curl -s "$SERVER_URL/health" | python -c "import sys,json; print(json.load(sys.stdin).get('model','?'))")
        echo "Server ready — model: $MODEL"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo ""
        echo "ERROR: Server did not respond after 60 s."
        echo "       Start it first: bash start_server.sh"
        exit 1
    fi
    sleep 2
done

echo ""

# One-shot or REPL
if [[ -n "$PROMPT" ]]; then
    python "$PROJECT_DIR/agent.py" \
        --server "$SERVER_URL" \
        $EXTRA_ARGS \
        "$PROMPT"
else
    python "$PROJECT_DIR/agent.py" \
        --repl \
        --server "$SERVER_URL" \
        $EXTRA_ARGS
fi
