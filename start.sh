#!/usr/bin/env bash
# Start the local model inference server + agent REPL, or Claude agent.
# Usage:
#   bash start.sh                # launches vLLM server + local agent in tmux
#   bash start.sh --server       # start vLLM server only
#   bash start.sh --agent        # start local agent REPL (assumes server running)
#   bash start.sh --claude       # start Claude Code agent (Anthropic SDK, no vLLM needed)

set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$WORKSPACE/.venv/bin/activate"

[[ -f "$VENV" ]] || { echo "Run setup.sh first."; exit 1; }

MODE="${1:-both}"

start_server() {
    echo "[start.sh] Starting vLLM inference server on :8080 ..."
    source "$VENV"
    python "$WORKSPACE/server.py" --host 0.0.0.0 --port 8080
}

start_agent() {
    echo "[start.sh] Waiting for server to be ready ..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:8080/health > /dev/null 2>&1 && break
        sleep 2
    done
    echo "[start.sh] Starting local agent REPL ..."
    source "$VENV"
    python "$WORKSPACE/agent.py" --repl
}

start_claude() {
    echo "[start.sh] Starting Claude Code agent ..."
    bash "$WORKSPACE/start_claude_agent.sh"
}

case "$MODE" in
  --server)  start_server ;;
  --agent)   start_agent  ;;
  --claude)  start_claude ;;
  *)
    if command -v tmux &> /dev/null; then
        SESSION="qwen-agent"
        tmux new-session -d -s "$SESSION" -x 220 -y 50 || true
        tmux send-keys -t "$SESSION" "bash $0 --server" Enter
        tmux split-window -h -t "$SESSION"
        tmux send-keys -t "$SESSION" "bash $0 --agent" Enter
        tmux attach-session -t "$SESSION"
    else
        echo "tmux not found — run in two terminals:"
        echo "  Terminal 1: bash start.sh --server"
        echo "  Terminal 2: bash start.sh --agent"
        echo ""
        echo "Or for Claude Code agent (no vLLM needed):"
        echo "  bash start.sh --claude"
    fi
    ;;
esac
