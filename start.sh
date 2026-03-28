#!/usr/bin/env bash
# Start the local model inference server + open the agent REPL in tmux.
# Usage:
#   bash start.sh           # launches both in split tmux panes
#   bash start.sh --server  # start server only
#   bash start.sh --agent   # start agent REPL only (assumes server is running)

set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$WORKSPACE/.venv/bin/activate"

[[ -f "$VENV" ]] || { echo "Run setup.sh first."; exit 1; }

MODE="${1:-both}"

start_server() {
    echo "[start.sh] Starting inference server on :8080 ..."
    source "$VENV"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
    python "$WORKSPACE/server.py" --host 0.0.0.0 --port 8080
}

start_agent() {
    echo "[start.sh] Waiting for server to be ready ..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:8080/health > /dev/null 2>&1 && break
        sleep 2
    done
    echo "[start.sh] Starting agent REPL ..."
    source "$VENV"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?'Set ANTHROPIC_API_KEY env var'}"
    python "$WORKSPACE/agent.py" --repl
}

case "$MODE" in
  --server) start_server ;;
  --agent)  start_agent  ;;
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
    fi
    ;;
esac
