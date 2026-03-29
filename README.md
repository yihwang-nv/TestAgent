# small_model — local LLM stack & coding agent

OpenAI-compatible **vLLM** inference (default) plus optional **TensorRT-LLM**, a **terminal coding agent** (`claude_agent.py`) with tools (read/write files, shell, grep, glob, optional vector KB), and smaller utilities (`agent.py`, `chat.py`). Model paths and Hugging Face repos are configured in `config.yaml`.

## Requirements

- NVIDIA GPU with recent drivers (`nvidia-smi` works)
- Python 3.10+ recommended (match your vLLM wheel)
- Enough VRAM for the checkpoint you choose (see `gpu_select.py` heuristics and `config.yaml`)

## Quick start

```bash
# 1) Create .venv, install vLLM stack, download weights (GPU-aware default size)
bash setup.sh

# 2) In another terminal — serve (reads config.yaml + gpu_select)
bash start_server.sh

# 3) Agent REPL against http://localhost:8080
bash start_claude_agent.sh
```

One-shot:

```bash
bash start_claude_agent.sh "explain server.py"
```

## Main scripts

| Script | Role |
|--------|------|
| `setup.sh` | venv + `pip install -r requirements.txt` (+ optional model download) |
| `download_model.py` | Pull weights from `hf_repo` in `config.yaml` |
| `start_server.sh` | Launch vLLM or TensorRT-LLM using `gpu_select.py` JSON |
| `server.py` | Python entry for the same launcher logic |
| `start_claude_agent.sh` | Activates `.venv` and runs `claude_agent.py` |
| `setup_trtllm.sh` | Separate `.venv_trtllm` for TensorRT-LLM |
| `start_agent.sh` / `start.sh` | Thin wrappers (see file headers) |

## Configuration

- **`config.yaml`** — `models.*` (`hf_repo`, `local_dir`, `served_model_name`), `vllm.*` (`max_model_len`, `tool_call_parser`, …), `inference.engine`.
- **`gpu_select.py`** — VRAM-based `--model-size` / `--quant` / `max_model_len` cap; override with `GPU_SELECT_MAX_MODEL_LEN` if needed.
- **Gated HF models** — `HF_TOKEN` or `huggingface-cli login` when downloading.

## Agent features

- Local mode: OpenAI-compatible API (`--server URL`, default `http://localhost:8080`).
- Optional `--anthropic` with `ANTHROPIC_API_KEY`.
- `--kb` / `--kb-auto` — Chroma + embeddings + PDF/Markdown ingest (`agent_knowledge.py`).

## Optional environment variables

- `GPU_SELECT_MAX_MODEL_LEN` — force vLLM `--max-model-len` (OOM risk if too high).
- `VLLM_TOOL_CALL_PARSER` — override `config.yaml` `vllm.tool_call_parser` (e.g. `qwen3_xml` for Qwen3.5 XML tools).
- `AGENT_ASSUMED_MAX_MODEL_LEN`, `AGENT_MAX_TOOL_CHARS`, `AGENT_MIN_COMPLETION_TOKENS`, … — see `claude_agent.py` / comments there.

## Troubleshooting

- **vLLM logs `hermes_tool_parser.py` + `JSONDecodeError` on tool calls** — The running server is using the **Hermes** parser (expects JSON inside `<tool_call>`). For Qwen3.5 XML tools, stop the process and restart with **`bash start_server.sh`** so `--tool-call-parser qwen3_xml` is applied, or run `VLLM_TOOL_CALL_PARSER=qwen3_xml bash start_server.sh`. Confirm the banner line `tool_parser : qwen3_xml` before serving traffic.

## License

Weights are third-party (see each `hf_repo` on Hugging Face). This repository’s scripts are provided as-is; add a `LICENSE` file if you redistribute them.
