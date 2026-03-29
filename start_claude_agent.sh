#!/usr/bin/env bash
# =============================================================================
# start_claude_agent.sh — Start the Code Agent
#
# Usage:
#   bash start_claude_agent.sh                        # local vLLM (default)
#   bash start_claude_agent.sh --anthropic            # use Anthropic API
#   bash start_claude_agent.sh "explain server.py"    # one-shot prompt
#   bash start_claude_agent.sh --server http://host:8080  # custom server
#   bash start_claude_agent.sh --kb                       # + PDF/向量记忆（需 requirements 中 chromadb 等）
#   bash start_claude_agent.sh --kb --kb-auto             # 每轮自动拼接相关片段
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV" ]]; then
    echo "Virtual environment not found. Running setup..."
    bash "$PROJECT_DIR/setup.sh" --skip-download
fi
source "$VENV"

cd "$PROJECT_DIR"
exec python "$PROJECT_DIR/claude_agent.py" "$@"
