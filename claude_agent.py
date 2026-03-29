#!/usr/bin/env python3
"""
claude_agent.py — Claude Code-style coding agent.

Supports two backends:
  1. Local vLLM server (OpenAI-compatible API) — default, no API key needed
  2. Anthropic API (Claude) — requires ANTHROPIC_API_KEY

Usage:
  python claude_agent.py                              # local vLLM (default)
  python claude_agent.py --local                      # explicit local mode
  python claude_agent.py --anthropic                  # use Anthropic API
  python claude_agent.py "explain server.py"          # one-shot
  python claude_agent.py --server http://localhost:8080  # custom server URL
  python claude_agent.py --kb                         # + vector KB (PDF/memory); pip: chromadb sentence-transformers pymupdf
  python claude_agent.py --kb --kb-auto               # auto-inject top KB chunks each turn
"""

import os
import re
import sys
import json
import uuid
import tempfile
import threading
import subprocess
import typer
from itertools import islice
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

import agent_knowledge

PROJECT = Path(__file__).parent
console = Console()
cli = typer.Typer(add_completion=False)

# Prefix glyphs cycle for motion; the text after "·" is fixed per step (not a blind carousel).
_ACTIVITY_PREFIX_GLYPHS = ("✽", "✻", "✢", "✦", "✳", "✷", "*", "·")


def _tool_step_label(tool_name: str) -> str:
    labels = {
        "file_stat": "Inspecting path…",
        "read_file": "Reading file…",
        "write_file": "Writing file…",
        "str_replace": "Patching file…",
        "shell": "Running shell command…",
        "glob_search": "Glob search…",
        "grep_search": "Searching in files…",
        "list_directory": "Listing directory…",
        "kb_search": "Searching knowledge base…",
        "kb_ingest_file": "Ingesting into knowledge base…",
        "kb_remember": "Saving to memory…",
    }
    return labels.get(tool_name, f"Running {tool_name}…")


def _run_with_activity(console_: Console, step_label: str, blocking_call):
    """Run a blocking callable; prefix symbol cycles, message reflects the current step only."""
    if not (step_label and step_label.strip()):
        return blocking_call()
    glyphs = _ACTIVITY_PREFIX_GLYPHS
    stop = threading.Event()
    status_holder: list = [None]

    def _line(i: int) -> str:
        g = glyphs[i % len(glyphs)]
        return f"[dim]{g} {step_label}[/dim]"

    def _cycle():
        i = 0
        interval = 0.4
        while not stop.wait(interval):
            st = status_holder[0]
            if st is not None:
                try:
                    i += 1
                    st.update(_line(i))
                except Exception:
                    pass

    with console_.status(_line(0), spinner="dots2") as status:
        status_holder[0] = status
        t = threading.Thread(target=_cycle, daemon=True)
        t.start()
        try:
            return blocking_call()
        finally:
            stop.set()
            t.join(timeout=1.5)

HISTORY_FILE = PROJECT / ".agent_history"

SYSTEM_PROMPT = """\
You are an expert AI coding assistant running inside a terminal on the user's machine.
You have access to tools that let you read files, write files, execute shell commands,
and search through the codebase.

Guidelines:
- Be concise and direct. Show code changes, not lengthy explanations.
- **ReAct (Thought → Action → Observation)**: On multi-step tasks, **before** each batch of tool calls write a short
  block starting with `Thought:` (1–4 sentences): what you know, what is still missing, and why the next tool(s)
  are appropriate. Your `<tool_call>` JSON is **Action**. The following messages that carry tool results are
  **Observation** — read them, then either another `Thought:` and more tools, or a final answer to the user.
  For trivial one-shot replies **without** tools, omit `Thought:`.
- **Before read_file**: call **file_stat** first on paths you have not inspected yet. If the file is large,
  use read_file with `offset` and `limit` instead of loading the whole file. The tool will block huge
  full-file reads without chunking. Long outputs are truncated for the API — prefer **limit 80–200 lines** per call unless the server context is very large.
- **Paths (no hallucination)**: You **must not** invent paths from “typical” repo layouts (wrong subdirs
  like `pkg/a/foo` vs `pkg/b/foo`, wrong suffix `.cc` vs `.cpp`, or mirrored vendor trees).
  A path is valid **only** if the user pasted it verbatim, or it came back from **glob_search** / **list_directory** /
  **shell** `find` / **grep_search** hit list. Workflow: user names a repo root + a filename → **glob_search**
  under that root (e.g. `**/attentionOp.cpp`, `**/attentionOp.*`) → **file_stat** on a listed hit → **read_file**.
  If **file_stat** says not found, **do not** tweak the path by guess — widen **glob_search** or ask the user.
- **grep_search**: small/medium jobs return matches inline. If the searched file is large or there are
  many hits, results are written to a temp file; read that file in chunks with read_file(offset, limit).
- **Tool calls (local vLLM, Hermes / tool_call_parser=hermes)**: Each call is **one JSON object** inside
  `<tool_call>…</tool_call>`. Shape: `{{"name": "<tool_name>", "arguments": {{ ... }}}}` — **valid JSON only**
  (double-quoted keys/strings, no trailing commas, escape newlines inside strings as `\\n`). Wrong: qwen3_xml
  blocks like `<function=glob_search><parameter=…>` — do not use those here. Example (copy shape; use real names/args):
  `<tool_call>`
  `{{"name": "glob_search", "arguments": {{"pattern": "**/attentionOp.cpp", "directory": "/absolute/or/cwd-relative/root"}}}}`
  `</tool_call>`
  Close `</tool_call>` before normal user-facing text. Multiple tools → multiple `<tool_call>` blocks.
- When editing files, read the relevant region (with offset/limit) before write_file or str_replace.
- **Tool errors**: If a tool message starts with `Error:` or describes a missing/invalid argument, **you** fix it:
  read the error, adjust parameters (e.g. correct `path` from glob_search), and call the tool again in the
  same task — the runtime will give you another turn automatically after each tool result.
- **Mutating tools** (write_file, str_replace, and risky shell — git add/commit/push, rm, package install,
  systemctl, docker/kubectl mutations, sudo, etc.) **will pause for explicit user confirmation** in the terminal.
  Tell the user what you are about to do before calling the tool so they know what to approve.
- For shell, prefer read-only commands first (git status/diff/log, ls). Do not chain destructive commands
  without user intent.
- Explain what you changed and why, briefly.
- If a task is ambiguous, ask a clarifying question before proceeding.
- **Long inputs / context limits**: If the conversation or tool outputs would make the next request too large
  (server errors about max context / input tokens), **split the work across multiple turns**: finish one
  coherent slice (e.g. one file or one subsystem), give a **short summary + what is left to do**, and stop
  or ask the user to say "continue" for the next slice. Prefer smaller **read_file** windows and **grep**
  over loading huge regions in one turn.
- **Knowledge base** (when enabled with `--kb`): use **kb_search** to retrieve ingested PDFs/docs and saved memories before answering factual questions about them. Use **kb_ingest_file** to add PDF, Markdown, code, or text files. Use **kb_remember** to persist short facts the user wants recalled in later sessions.

Working directory: {cwd}
"""

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS_ANTHROPIC_BASE = [
    {
        "name": "file_stat",
        "description": (
            "Probe a path: size, type, line count (for smaller files). "
            "Use only paths from the user or from glob_search/list_dir/find — never guessed layouts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or cwd-relative path; must be discovered, not invented.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read file contents as numbered lines. For large files use offset+limit; call file_stat first. "
            "Path must be verified (glob_search → file_stat), not assumed from other codebases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path after glob/stat or user-provided exact path."},
                "offset": {"type": "integer", "description": "Start line (1-indexed). Optional."},
                "limit": {"type": "integer", "description": "Max lines to read. Optional."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating or overwriting it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "str_replace",
        "description": "Replace an exact string in a file. old_string must match exactly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_string": {"type": "string", "description": "Exact string to find."},
                "new_string": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "shell",
        "description": "Execute a shell command. Timeout: 120s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "working_directory": {"type": "string", "description": "Directory to run in."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "glob_search",
        "description": (
            "Find files matching a glob pattern. Prefer this before file_stat when the exact subdirectory is unknown. "
            "Recursive globs skip .git, .cache, node_modules, venv, __pycache__, and similar bulky dirs for speed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob, e.g. '**/attentionOp.cpp' or '**/FooBar.*' under the user's tree.",
                },
                "directory": {"type": "string", "description": "Base directory (e.g. user-given repo root)."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": (
            "Search for a regex pattern (ripgrep if available). "
            "Small targets / moderate output: results inline. "
            "Large single file or very large hit sets: full results written under the system temp dir; "
            "response includes a preview — use read_file on that path with offset/limit to analyze."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern."},
                "path": {"type": "string", "description": "File or directory to search (preferred if both set)."},
                "directory": {
                    "type": "string",
                    "description": "Base directory to search (same role as path when path omitted; like glob_search).",
                },
                "include": {"type": "string", "description": "Glob filter, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories at a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path."},
            },
            "required": [],
        },
    },
]

KB_TOOLS_ANTHROPIC = [
    {
        "name": "kb_search",
        "description": (
            "Semantic search over the local vector knowledge base: ingested files (PDF, text, code) "
            "and notes saved via kb_remember. Call this when the user asks about uploaded documents "
            "or prior saved facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Max chunks to return (default 8, max 32).",
                },
                "kind": {
                    "type": "string",
                    "description": "Optional filter: 'memory' (saved notes only), 'document' (files only), or omit for both.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "kb_ingest_file",
        "description": (
            "Ingest a file into the vector knowledge base for later kb_search. "
            "Supports .pdf (text layer; scanned PDFs use OCR if Tesseract is installed), .docx, "
            ".md, .txt, .py, and common text formats. Path relative to CWD OK."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to ingest."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "kb_remember",
        "description": (
            "Save a short note or fact to long-term vector memory (survives restarts). "
            "Use when the user explicitly asks to remember something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Content to remember."},
                "title": {"type": "string", "description": "Optional short label for this memory."},
            },
            "required": ["text"],
        },
    },
]


def build_tool_schemas(include_kb: bool) -> tuple[list[dict], list[dict]]:
    anth = TOOLS_ANTHROPIC_BASE + (KB_TOOLS_ANTHROPIC if include_kb else [])
    return anth, _to_openai_tools(anth)


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


# ── Tool implementations ─────────────────────────────────────────────────────

def _coerce_optional_int(val) -> int | None:
    """Tool / YAML may pass a lone-element list/tuple or str; avoid min(tuple, int) crashes."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        val = val[0]
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _normalize_tool_args(name: str, args: dict) -> dict:
    """Map common model alias keys to schema names (Hermes / small models often skip exact keys)."""
    out = dict(args) if isinstance(args, dict) else {}

    def _strip_str(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    if name in ("read_file", "file_stat", "write_file", "str_replace", "kb_ingest_file"):
        if not _strip_str(out.get("path")):
            for alt in ("file_path", "filepath", "target_file", "target_path", "filename"):
                s = _strip_str(out.get(alt))
                if s:
                    out["path"] = s
                    break
            if not _strip_str(out.get("path")):
                f = out.get("file")
                if isinstance(f, str) and f.strip():
                    out["path"] = f.strip()

    if name == "read_file":
        if out.get("offset") is None:
            for alt in ("start_line", "start", "line", "line_number"):
                c = _coerce_optional_int(out.get(alt))
                if c is not None:
                    out["offset"] = c
                    break
        if out.get("limit") is None:
            for alt in ("max_lines", "num_lines", "lines", "n_lines"):
                c = _coerce_optional_int(out.get(alt))
                if c is not None:
                    out["limit"] = c
                    break

    if name == "str_replace":
        if not _strip_str(out.get("old_string")):
            for alt in ("old_str", "from_string", "search"):
                s = out.get(alt)
                if isinstance(s, str) and s:
                    out["old_string"] = s
                    break
        if not _strip_str(out.get("new_string")):
            for alt in ("new_str", "to_string", "replace"):
                s = out.get(alt)
                if isinstance(s, str) and s:
                    out["new_string"] = s
                    break

    if name == "list_directory" and not _strip_str(out.get("path")):
        for alt in ("directory", "dir", "file_path"):
            s = _strip_str(out.get(alt))
            if s:
                out["path"] = s
                break

    if name == "shell" and not _strip_str(out.get("command")):
        for alt in ("cmd", "shell_command", "shell_cmd"):
            s = _strip_str(out.get(alt))
            if s:
                out["command"] = s
                break

    if name == "glob_search" and not _strip_str(out.get("directory")):
        for alt in ("dir", "path", "root", "base_dir", "base"):
            s = _strip_str(out.get(alt))
            if s:
                out["directory"] = s
                break

    if name == "kb_search" and not _strip_str(out.get("query")):
        for alt in ("q", "search", "text"):
            s = _strip_str(out.get(alt))
            if s:
                out["query"] = s
                break

    return out


def _assistant_message_text(content) -> str:
    """OpenAI/vLLM may return str or list[text blocks]; normalize for history + render."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


# Full-file read blocked above this size unless offset/limit are used.
READ_FILE_MAX_BYTES_FULL = 384 * 1024

# Grep: spill matches to a temp file when the search target file is large or output is huge.
GREP_SPILL_SOURCE_BYTES = READ_FILE_MAX_BYTES_FULL
GREP_MAX_INLINE_BYTES = 48 * 1024
GREP_INLINE_MAX_LINES = 120
GREP_PREVIEW_LINES = 45


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v > 0 else default
    except ValueError:
        return default


# 单次工具返回写入对话的最大字符数（避免 read_file 撑满上下文）
AGENT_MAX_TOOL_CHARS = _env_int("AGENT_MAX_TOOL_CHARS", 7_000)
# vLLM 把 tools schema 打进 prompt，约占大量 token；从假定上下文里扣掉后再换算字符预算
AGENT_ASSUMED_MAX_MODEL_LEN = _env_int("AGENT_ASSUMED_MAX_MODEL_LEN", 16384)
AGENT_PROMPT_OVERHEAD_TOKENS = _env_int("AGENT_PROMPT_OVERHEAD_TOKENS", 8500)
AGENT_PRUNE_TOOL_CAP = _env_int("AGENT_PRUNE_TOOL_CAP", 2_800)
AGENT_ANTHROPIC_PRUNE_CHARS = _env_int("AGENT_ANTHROPIC_PRUNE_CHARS", 52_000)
# Stay below vLLM input+output sum (code tokenizes denser than ~3 chars/token for JSON heuristics).
AGENT_CONTEXT_SAFETY_TOKENS = _env_int("AGENT_CONTEXT_SAFETY_TOKENS", 512)
# Chars of messages JSON allowed per token of input budget after reserves (lower = more aggressive prune).
AGENT_PRUNE_CHARS_PER_SLOT = max(1, _env_int("AGENT_PRUNE_CHARS_PER_SLOT", 2))
AGENT_COMPLETION_RESERVE_TOKENS = _env_int("AGENT_COMPLETION_RESERVE_TOKENS", 256)
# Never pass max_tokens below this when char-heuristic says "no room" (avoids vLLM getting 1).
AGENT_MIN_COMPLETION_TOKENS = _env_int("AGENT_MIN_COMPLETION_TOKENS", 512)
# Single completion ceiling (env); default large so --max-tokens can track model window after clamp.
AGENT_MAX_COMPLETION_TOKENS_CAP = _env_int("AGENT_MAX_COMPLETION_TOKENS", 131072)

_effective_ctx_len: int | None = None


def set_effective_context_len(n: int) -> None:
    """Call from main() for local vLLM: min(config vllm.max_model_len, gpu_select VRAM tier)."""
    global _effective_ctx_len
    try:
        if isinstance(n, (list, tuple)) and len(n) == 1:
            n = n[0]
        _effective_ctx_len = max(2048, int(n))
    except (TypeError, ValueError):
        _effective_ctx_len = max(2048, AGENT_ASSUMED_MAX_MODEL_LEN)


def _assumed_model_len_for_prune() -> int:
    if _effective_ctx_len is not None:
        return _effective_ctx_len
    return AGENT_ASSUMED_MAX_MODEL_LEN


# 终端里工具结果默认只显示前几行（类似 Claude Code）；Ctrl+O 切换后续是否全文打印；/peek 查看最近一次全文
TOOL_FOLD_PREVIEW_LINES = _env_int("AGENT_TOOL_FOLD_PREVIEW_LINES", 8)
_fold_tool_output_full: bool = False
_last_tool_result_full: str = ""
_last_tool_result_label: str = ""


def toggle_tool_output_fold_mode() -> None:
    global _fold_tool_output_full
    _fold_tool_output_full = not _fold_tool_output_full
    mode = "全文" if _fold_tool_output_full else f"预览（前 {TOOL_FOLD_PREVIEW_LINES} 行）"
    console.print(f"\n[dim]工具输出: {mode} · 再次 Ctrl+O 切换[/dim]")


def _truncate_tool_result(text: str, tool_name: str) -> str:
    if not text or len(text) <= AGENT_MAX_TOOL_CHARS:
        return text
    head = text[:AGENT_MAX_TOOL_CHARS]
    return (
        f"{head}\n\n"
        f"... [输出已截断: {len(text)} → {AGENT_MAX_TOOL_CHARS} 字符; "
        f"大文件请 read_file(path, offset, limit) 分段或 grep_search 定位]"
    )


def _openai_messages_json_size(messages: list[dict]) -> int:
    try:
        return len(json.dumps(messages, default=str))
    except Exception:
        return sum(len(str(m.get("content", ""))) for m in messages)


def prune_openai_messages(
    messages: list[dict],
    max_completion_tokens: int,
    *,
    pressure_tokens: int = 0,
) -> None:
    """
    Shrink tool outputs so prompt + tool schemas stay under vLLM max_input_tokens.
    Uses a conservative chars-per-slot budget (see AGENT_PRUNE_CHARS_PER_SLOT); overhead covers
    system + tool schemas + chat template; safety tokens avoid off-by-a-few-hundred 400s.
    ``pressure_tokens`` is subtracted from avail (used after server 400 to clamp harder).
    """
    avail = (
        _assumed_model_len_for_prune()
        - int(max_completion_tokens)
        - AGENT_PROMPT_OVERHEAD_TOKENS
        - AGENT_CONTEXT_SAFETY_TOKENS
        - max(0, int(pressure_tokens))
    )
    if avail < 2048:
        avail = 2048
    char_budget = max(10_000, avail * AGENT_PRUNE_CHARS_PER_SLOT)
    cap = AGENT_PRUNE_TOOL_CAP
    for _ in range(24):
        if _openai_messages_json_size(messages) <= char_budget:
            return
        shrunk = False
        for m in messages:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                c = m["content"]
                if len(c) > cap:
                    old_len = len(c)
                    m["content"] = (
                        c[:cap]
                        + f"\n\n... [历史压缩: {old_len} → {cap} 字符，避免超出上下文上限]"
                    )
                    shrunk = True
        if not shrunk:
            cap = max(800, cap * 3 // 4)


def _anthropic_tool_text_volume(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "tool_result" and isinstance(b.get("content"), str):
                        n += len(b["content"])
                    elif b.get("type") == "text" and isinstance(b.get("text"), str):
                        n += len(b["text"])
    return n


def prune_anthropic_messages(messages: list[dict]) -> None:
    if _anthropic_tool_text_volume(messages) <= AGENT_ANTHROPIC_PRUNE_CHARS:
        return
    cap = AGENT_PRUNE_TOOL_CAP
    for _ in range(10):
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                t = b.get("content")
                if isinstance(t, str) and len(t) > cap:
                    old_len = len(t)
                    b["content"] = (
                        t[:cap]
                        + f"\n\n... [历史压缩: {old_len} → {cap} 字符]"
                    )
        if _anthropic_tool_text_volume(messages) <= AGENT_ANTHROPIC_PRUNE_CHARS:
            return
        cap = max(1200, cap * 2 // 3)


def tool_file_stat(path: str) -> str:
    fpath = _resolve(path)
    if not fpath.exists():
        return f"Error: Path not found: {fpath}"
    try:
        st = fpath.stat()
        sz = st.st_size
        if fpath.is_dir():
            n = len(list(fpath.iterdir()))
            return (
                f"path: {fpath}\n"
                f"type: directory\n"
                f"entries: {n}\n"
                f"(use list_directory for names; file_stat is for one path)"
            )
        lines_note = ""
        if sz <= 2 * 1024 * 1024:
            try:
                with fpath.open("rb") as f:
                    raw = f.read()
                text = raw.decode("utf-8", errors="replace")
                line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
                lines_note = f"approx_lines: {line_count}\n"
            except Exception as e:
                lines_note = f"lines: (unavailable: {e})\n"
        else:
            lines_note = "approx_lines: (file > 2 MiB — use read_file with offset/limit)\n"
        hum = f"{sz} B" if sz < 1024 else (f"{sz / 1024:.1f} KiB" if sz < 1024**2 else f"{sz / 1024**2:.1f} MiB")
        return (
            f"path: {fpath}\n"
            f"type: file\n"
            f"size_bytes: {sz} ({hum})\n"
            f"{lines_note}"
            f"hint: if large, read_file(path, offset=1, limit=200)"
        )
    except Exception as e:
        return f"Error: {e}"


def tool_read_file(path: str, offset: int = None, limit: int = None) -> str:
    fpath = _resolve(path)
    if not fpath.exists():
        return f"Error: File not found: {fpath}"
    if not fpath.is_file():
        return f"Error: Not a file: {fpath}"
    offset = _coerce_optional_int(offset)
    limit = _coerce_optional_int(limit)
    try:
        sz = fpath.stat().st_size
        if offset is None and limit is None and sz > READ_FILE_MAX_BYTES_FULL:
            return (
                f"Error: File is {sz} bytes (> {READ_FILE_MAX_BYTES_FULL} bytes). "
                f"Call file_stat('{path}') first, then read_file with offset and limit "
                f"(e.g. offset=1, limit=120)."
            )
        lines = fpath.read_text(errors="replace").splitlines(keepends=True)
        if offset is not None:
            start = max(0, offset - 1)
            end = start + limit if limit else len(lines)
            lines = lines[start:end]
            numbered = [f"{start + i + 1:6d}|{line}" for i, line in enumerate(lines)]
        else:
            numbered = [f"{i + 1:6d}|{line}" for i, line in enumerate(lines)]
        return "".join(numbered)
    except Exception as e:
        return f"Error reading {fpath}: {e}"


def tool_write_file(path: str, content: str) -> str:
    fpath = _resolve(path)
    try:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
        return f"Wrote {len(content)} bytes to {fpath}"
    except Exception as e:
        return f"Error writing {fpath}: {e}"


def tool_str_replace(path: str, old_string: str, new_string: str) -> str:
    fpath = _resolve(path)
    if not fpath.exists():
        return f"Error: File not found: {fpath}"
    try:
        text = fpath.read_text()
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {fpath}"
        if count > 1:
            return f"Error: old_string found {count} times — make it more specific"
        new_text = text.replace(old_string, new_string, 1)
        fpath.write_text(new_text)
        return f"Replaced 1 occurrence in {fpath}"
    except Exception as e:
        return f"Error: {e}"


def tool_shell(command: str, working_directory: str = None) -> str:
    cwd = working_directory or str(Path.cwd())
    console.print(f"  [dim]$ {command}[/dim]")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=120,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (120s)"
    except Exception as e:
        return f"Error: {e}"


# Pruned during recursive glob — avoids crawling .git, package trees, caches (major speedup).
GLOB_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".cache",
        "node_modules",
        "venv",
        ".venv",
        ".virtualenv",
        ".yarn",
        ".pnpm-store",
        "bower_components",
    }
)


def _glob_collect(base: Path, pattern: str) -> list[Path]:
    """Paths under base matching pattern. ``**`` recursion skips GLOB_SKIP_DIR_NAMES."""
    base = base.resolve()
    if not base.is_dir():
        return []
    pat = pattern.replace("\\", "/")
    if "**" in pat:
        out: list[Path] = []
        for root, dirs, files in os.walk(base, topdown=True):
            dirs[:] = [d for d in dirs if d not in GLOB_SKIP_DIR_NAMES]
            rp = Path(root)
            for fn in files:
                full = rp / fn
                try:
                    rel = full.relative_to(base)
                except ValueError:
                    continue
                try:
                    if rel.match(pat):
                        out.append(full)
                except ValueError:
                    continue
        return sorted(out, key=lambda p: str(p))
    try:
        return sorted(base.glob(pat))
    except Exception:
        return []


def tool_glob_search(pattern: str, directory: str = None) -> str:
    base = _resolve(directory) if directory else Path.cwd()
    try:
        matches = _glob_collect(base, pattern)
        if not matches:
            return f"No files matched: {pattern}"
        lines = [str(m.relative_to(base.resolve())) for m in matches[:200]]
        result = "\n".join(lines)
        if len(matches) > 200:
            result += f"\n... and {len(matches) - 200} more"
        return result
    except Exception as e:
        return f"Error: {e}"


def _grep_temp_path() -> Path:
    base = Path(os.environ.get("AGENT_GREP_TMPDIR", tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="agent_grep_", suffix=".txt", dir=str(base))
    os.close(fd)
    return Path(name)


def _grep_spill_summary(out_path: Path, reason: str) -> str:
    try:
        nlines = int(
            subprocess.check_output(["wc", "-l", str(out_path)], text=True, timeout=60).split()[0]
        )
    except Exception:
        nlines = -1
    preview: list[str] = []
    try:
        with out_path.open(encoding="utf-8", errors="replace") as f:
            preview = [ln.rstrip("\n") for ln in islice(f, GREP_PREVIEW_LINES)]
    except Exception as e:
        preview = [f"(could not read preview: {e})"]
    prev_txt = "\n".join(preview) if preview else "(empty file)"
    sz = out_path.stat().st_size
    return (
        f"{reason}\n"
        f"Results file: {out_path}\n"
        f"size_bytes: {sz}"
        + (f"\nline_count: {nlines}" if nlines >= 0 else "")
        + f"\n\n--- preview (first {len(preview)} lines) ---\n{prev_txt}\n\n"
        f"Next: read_file(path={str(out_path)!r}, offset=1, limit=80) and increase offset to scan."
    )


def tool_grep_search(pattern: str, path: str = None, directory: str = None, include: str = None) -> str:
    # `directory` matches glob_search naming; many models pass directory= without path=.
    target = (path if path not in (None, "") else None) or (directory if directory not in (None, "") else None) or "."
    cwd = str(Path.cwd())
    search_base = _resolve(target) if target not in (".", "") else Path.cwd()
    force_spill = search_base.is_file() and search_base.stat().st_size >= GREP_SPILL_SOURCE_BYTES

    cmd_rg = ["rg", "--line-number", "--no-heading", "--color=never", "-e", pattern]
    if include:
        cmd_rg += ["--glob", include]
    cmd_rg.append(target)

    def run_rg_to_file(out: Path) -> tuple[int, str]:
        with out.open("w", encoding="utf-8", errors="replace") as wf:
            p = subprocess.run(
                cmd_rg,
                cwd=cwd,
                stdout=wf,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
        err = (p.stderr or "").strip()
        return p.returncode, err

    if force_spill:
        out_path = _grep_temp_path()
        try:
            code, err = run_rg_to_file(out_path)
        except FileNotFoundError:
            gcmd = ["grep", "-rHn"]
            if include:
                gcmd += ["--include", include]
            gcmd += ["-E", pattern, target]
            try:
                with out_path.open("w", encoding="utf-8", errors="replace") as wf:
                    p = subprocess.run(
                        gcmd, cwd=cwd, stdout=wf, stderr=subprocess.PIPE, text=True, timeout=120
                    )
                code, err = p.returncode, (p.stderr or "").strip()
            except Exception as e:
                out_path.unlink(missing_ok=True)
                return f"Error (grep fallback): {e}"
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return f"Error: {e}"
        if code not in (0, 1):
            out_path.unlink(missing_ok=True)
            return f"grep failed (code {code}): {err or '(no stderr)'}"
        if out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)
            return f"No matches for pattern: {pattern}"
        return _grep_spill_summary(
            out_path,
            reason=f"Spilled because search target file is >= {GREP_SPILL_SOURCE_BYTES} bytes.",
        )

    try:
        result = subprocess.run(
            cmd_rg, capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        output = result.stdout
        err = (result.stderr or "").strip()
        if result.returncode not in (0, 1):
            return f"rg failed (code {result.returncode}): {err or output[:500]}"
        output = output.strip()
        if not output:
            return f"No matches for pattern: {pattern}"
        if len(output.encode("utf-8", errors="replace")) > GREP_MAX_INLINE_BYTES:
            out_path = _grep_temp_path()
            out_path.write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
            return _grep_spill_summary(
                out_path,
                reason="Spilled because match output exceeded inline size limit.",
            )
        lines = output.splitlines()
        if len(lines) > GREP_INLINE_MAX_LINES:
            out_path = _grep_temp_path()
            out_path.write_text(output + "\n", encoding="utf-8")
            return _grep_spill_summary(
                out_path,
                reason=f"Spilled because there are {len(lines)} matching lines (> {GREP_INLINE_MAX_LINES}).",
            )
        return output
    except FileNotFoundError:
        gcmd = ["grep", "-rHn"]
        if include:
            gcmd += ["--include", include]
        gcmd += ["-E", pattern, target]
        try:
            result = subprocess.run(
                gcmd, capture_output=True, text=True, timeout=60, cwd=cwd,
            )
            output = result.stdout.strip()
            if not output:
                return f"No matches for: {pattern}"
            if len(output.encode("utf-8", errors="replace")) > GREP_MAX_INLINE_BYTES:
                out_path = _grep_temp_path()
                out_path.write_text(output + "\n", encoding="utf-8")
                return _grep_spill_summary(
                    out_path,
                    reason="Spilled because match output exceeded inline size limit (grep).",
                )
            lines = output.splitlines()
            if len(lines) > GREP_INLINE_MAX_LINES:
                out_path = _grep_temp_path()
                out_path.write_text(output + "\n", encoding="utf-8")
                return _grep_spill_summary(
                    out_path,
                    reason=f"Spilled: {len(lines)} lines (grep, > {GREP_INLINE_MAX_LINES}).",
                )
            return output
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def tool_list_directory(path: str = None) -> str:
    target = _resolve(path) if path else Path.cwd()
    if not target.exists():
        return f"Error: Directory not found: {target}"
    try:
        # Use 0/1 not bool so keys are always (int, str); avoids rare sort/compare quirks.
        entries = sorted(target.iterdir(), key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"  d {e.name}/")
            else:
                size = e.stat().st_size
                if size < 1024:
                    s = f"{size} B"
                elif size < 1024 * 1024:
                    s = f"{size / 1024:.1f} KB"
                else:
                    s = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  f {e.name}  ({s})")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


def tool_kb_search(query: str, top_k=None, kind: str = None) -> str:
    store = agent_knowledge.get_kb_store()
    if not store:
        return (
            "知识库未启用。使用: python claude_agent.py --kb\n"
            f"依赖: {agent_knowledge.kb_install_hint()}"
        )
    k = 8
    if top_k is not None:
        try:
            k = int(top_k)
        except (TypeError, ValueError):
            k = 8
    kd = (kind or "").strip().lower()
    if kd not in ("memory", "document"):
        kd = None
    return store.search(query or "", top_k=k, kind=kd)


def tool_kb_ingest_file(path: str) -> str:
    store = agent_knowledge.get_kb_store()
    if not store:
        return "知识库未启用。使用: python claude_agent.py --kb"
    if not (path or "").strip():
        return "Error: 空路径"
    return store.ingest_file(_resolve(path))


def tool_kb_remember(text: str, title: str = None) -> str:
    store = agent_knowledge.get_kb_store()
    if not store:
        return "知识库未启用。使用: python claude_agent.py --kb"
    return store.remember(text or "", title)


TOOL_DISPATCH = {
    "file_stat":      lambda args: tool_file_stat(**args),
    "read_file":      lambda args: tool_read_file(**args),
    "write_file":     lambda args: tool_write_file(**args),
    "str_replace":    lambda args: tool_str_replace(**args),
    "shell":          lambda args: tool_shell(**args),
    "glob_search":    lambda args: tool_glob_search(**args),
    "grep_search": lambda args: tool_grep_search(
        args.get("pattern") or "",
        path=args.get("path"),
        directory=args.get("directory"),
        include=args.get("include"),
    ),
    "list_directory": lambda args: tool_list_directory(**args),
    "kb_search":      lambda args: tool_kb_search(
        args.get("query") or "",
        args.get("top_k"),
        args.get("kind"),
    ),
    "kb_ingest_file": lambda args: tool_kb_ingest_file(args.get("path") or ""),
    "kb_remember":    lambda args: tool_kb_remember(args.get("text") or "", args.get("title")),
}


def _user_confirm(summary: str) -> bool:
    console.print(f"\n[bold yellow]Confirmation required[/bold yellow]\n{summary}")
    try:
        ans = input("Type 'yes' to run this tool, anything else to cancel: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]cancelled[/dim]")
        return False
    ok = ans in ("yes", "y")
    if not ok:
        console.print("[dim]Cancelled by user.[/dim]")
    return ok


def _shell_needs_confirm(command: str) -> bool:
    c = command.strip()
    cl = c.lower()
    if not cl:
        return False
    if cl.startswith("sudo ") or re.search(r"\bsudo\s+", cl):
        return True
    if re.search(r"\brm\s+", cl):
        return True
    if re.search(r"\bmv\s+", cl):
        return True
    if re.search(r"\bchmod\b", cl) or re.search(r"\bchown\b", cl):
        return True
    if re.search(r"\b(dd|mkfs|shutdown|reboot|halt|poweroff)\b", cl):
        return True
    if re.search(
        r"\b(apt|apt-get|yum|dnf|pacman|zypper|brew)\s+"
        r"(install|remove|purge|upgrade|update|full-upgrade)\b",
        cl,
    ):
        return True
    if re.search(
        r"\bsystemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask)\b",
        cl,
    ):
        return True
    if re.search(
        r"\bdocker\s+(rm|run|build|push|pull|stop|kill|restart|system\s+prune)\b",
        cl,
    ):
        return True
    if re.search(r"\bkubectl\s+(delete|apply|replace|patch|exec)\b", cl):
        return True
    if re.match(r"git\s+", cl):
        if re.match(
            r"git\s+(status|diff|log|show|grep|blame|help)\b",
            cl,
        ):
            return False
        if re.match(r"git\s+ls-files\b", cl):
            return False
        if re.match(r"git\s+rev-parse\b", cl):
            return False
        if re.match(r"git\s+describe\b", cl):
            return False
        if re.match(r"git\s+remote(\s+(-v|--verbose))?\s*$", cl):
            return False
        if re.match(r"git\s+branch\s*$", cl):
            return False
        if re.match(r"git\s+branch\s+(-a|-r|--all|--list)(\s|$)", cl):
            return False
        if re.match(r"git\s+stash\s+list\b", cl):
            return False
        if re.match(r"git\s+config\s+(--get|--list)\b", cl):
            return False
        return True
    return False


# ── Rendering ────────────────────────────────────────────────────────────────

# Qwen-style reasoning segments (matches vLLM chat template for many Qwen3 checkpoints)
_THINK_TAG_OPEN = "<" + "think" + ">"
_THINK_TAG_CLOSE = "<" + "/think" + ">"


def _print_markdown_or_plain(chunk: str) -> None:
    chunk = chunk or ""
    if not chunk.strip():
        return
    try:
        console.print(Markdown(chunk))
    except Exception:
        console.print(chunk)


def render_text(text: str) -> None:
    """Print assistant text; Qwen-style reasoning spans (think tags) use dim styling."""
    if not text or not text.strip():
        return
    if _THINK_TAG_OPEN not in text:
        _print_markdown_or_plain(text)
        return
    pos = 0
    while pos < len(text):
        start = text.find(_THINK_TAG_OPEN, pos)
        if start < 0:
            _print_markdown_or_plain(text[pos:])
            break
        if start > pos:
            _print_markdown_or_plain(text[pos:start])
        end = text.find(_THINK_TAG_CLOSE, start + len(_THINK_TAG_OPEN))
        if end < 0:
            console.print(Text(text[start:], style="dim"))
            break
        console.print(
            Text(
                text[start : end + len(_THINK_TAG_CLOSE)],
                style="dim",
            )
        )
        pos = end + len(_THINK_TAG_CLOSE)


def render_tool_use(name: str, args: dict):
    labels = {
        "shell": ("bold yellow", "shell", args.get("command", "")),
        "file_stat": ("bold cyan", "stat", args.get("path", "")),
        "read_file": ("bold cyan", "read", args.get("path", "")),
        "write_file": ("bold green", "write", args.get("path", "")),
        "str_replace": ("bold green", "edit", args.get("path", "")),
        "glob_search": ("bold magenta", "search", args.get("pattern", "")),
        "grep_search": ("bold magenta", "grep", args.get("pattern", "")),
        "list_directory": ("bold blue", "ls", args.get("path", ".")),
        "kb_search": ("bold green", "kb", args.get("query", "")[:60]),
        "kb_ingest_file": ("bold green", "kb+", args.get("path", "")),
        "kb_remember": ("bold green", "mem", (args.get("title") or args.get("text", ""))[:50]),
    }
    style, label, detail = labels.get(name, ("bold", name, ""))
    console.print(f"  [{style}]{label}:[/{style}] {detail}")


def render_tool_result(result: str, tool_label: str = ""):
    global _last_tool_result_full, _last_tool_result_label
    _last_tool_result_full = result or ""
    _last_tool_result_label = tool_label or "tool"
    if not result:
        return
    lines = result.splitlines()
    preview = max(1, TOOL_FOLD_PREVIEW_LINES)
    if _fold_tool_output_full or len(lines) <= preview:
        console.print(f"[dim]{result}[/dim]")
        return
    head = "\n".join(lines[:preview])
    console.print(f"[dim]{head}[/dim]")
    console.print(
        f"[dim]… ({len(lines)} 行，已折叠) · Ctrl+O 切换全文/预览 · /peek 查看本次完整输出[/dim]"
    )


def _exec_tool(name: str, args: dict) -> str:
    if not isinstance(args, dict):
        args = {}
    args = _normalize_tool_args(name, args)
    render_tool_use(name, args)

    if name == "read_file" and not (isinstance(args.get("path"), str) and args["path"].strip()):
        result = (
            "Error: read_file requires `path` (file to read). "
            "Use the exact path from glob_search/file_stat or the user; optional: `offset`, `limit` (line numbers)."
        )
        render_tool_result(result, tool_label=name)
        return result

    if name == "write_file":
        p = args.get("path", "")
        exists = _resolve(p).exists() if p else False
        summ = f"write_file → {p!r} ({'overwrite' if exists else 'create'})"
        if not _user_confirm(summ):
            result = "User declined: write_file was not executed."
            render_tool_result(result, tool_label=name)
            return result

    if name == "str_replace":
        p = args.get("path", "")
        summ = f"str_replace → {p!r} (in-place edit)"
        if not _user_confirm(summ):
            result = "User declined: str_replace was not executed."
            render_tool_result(result, tool_label=name)
            return result

    if name == "shell":
        cmd = args.get("command", "")
        if _shell_needs_confirm(cmd):
            summ = f"shell → {cmd!r}"
            if not _user_confirm(summ):
                result = "User declined: shell command was not executed."
                render_tool_result(result, tool_label=name)
                return result

    handler = TOOL_DISPATCH.get(name)
    if handler:
        result = _run_with_activity(
            console,
            _tool_step_label(name),
            lambda: handler(args),
        )
    else:
        result = f"Error: Unknown tool '{name}'"
    result = _truncate_tool_result(result, name)
    render_tool_result(result, tool_label=name)
    return result


# ── Backend: OpenAI-compatible (local vLLM) ──────────────────────────────────

def _clamp_openai_max_tokens(sys_msg: dict, messages: list[dict], requested_max: int) -> int:
    """Cap completion budget so input_est + max_tokens + fudge fits server context.

    Call only **after** prune_openai_messages: pre-prune JSON is huge while old tool blobs
    are still present; clamping first falsely drives max_tokens → 1.
    """
    ctx = int(_assumed_model_len_for_prune())
    raw = len(json.dumps([sys_msg] + messages, default=str))
    # Post-prune payload: mix of prose and code; ~2.5 chars/token is a middle ground vs vLLM.
    est_content_tokens = max(1, (raw * 2 + 4) // 5)
    est_prompt_tokens = est_content_tokens + AGENT_PROMPT_OVERHEAD_TOKENS
    slack = AGENT_COMPLETION_RESERVE_TOKENS + AGENT_CONTEXT_SAFETY_TOKENS
    room = ctx - est_prompt_tokens - slack
    try:
        req = int(requested_max)
    except (TypeError, ValueError):
        req = 8192
    req = max(1, min(req, AGENT_MAX_COMPLETION_TOKENS_CAP))
    floor = AGENT_MIN_COMPLETION_TOKENS
    if room >= floor:
        room_eff = room
    else:
        # Heuristic often over-counts prompt vs real BPE tokens; try a relaxed prompt estimate.
        relaxed_prompt = (est_content_tokens // 2) + AGENT_PROMPT_OVERHEAD_TOKENS
        room_relaxed = ctx - relaxed_prompt - slack
        room_eff = max(floor, room, room_relaxed)
    return max(1, min(req, room_eff))


def _parse_hermes_json_tool_calls(content: str) -> Optional[tuple[str, list[dict]]]:
    """
    When vLLM returns no tool_calls (e.g. hermes parse failed on malformed JSON), extract Hermes-style
    JSON inside <tool_call>…</tool_call>. Returns (prefix_text, [{id,name,arguments}, ...]).
    """
    if not content or "<tool_call>" not in content:
        return None
    prefix = content[: content.find("<tool_call>")].strip()
    specs: list[dict] = []
    i = 0
    while True:
        s = content.find("<tool_call>", i)
        if s < 0:
            break
        e = content.find("</tool_call>", s)
        if e < 0:
            inner = content[s + len("<tool_call>") :].strip()
            i = len(content)
        else:
            inner = content[s + len("<tool_call>") : e].strip()
            i = e + len("</tool_call>")
        if not inner.startswith("{"):
            if e < 0:
                break
            continue
        try:
            data = json.loads(inner)
        except json.JSONDecodeError:
            if e < 0:
                break
            continue
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            if e < 0:
                break
            continue
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        specs.append(
            {
                "id": f"fallback_{uuid.uuid4().hex[:20]}",
                "name": name.strip(),
                "arguments": args,
            }
        )
        if e < 0:
            break
    if not specs:
        return None
    return prefix, specs


def _openai_context_length_error(exc: BaseException) -> bool:
    el = str(exc).lower()
    code = getattr(exc, "status_code", None)
    if code == 400 and ("context" in el or "input_tokens" in el or "maximum context" in el):
        return True
    return (
        ("context" in el and "length" in el)
        or "maximum context" in el
        or "input_tokens" in el
    )


def run_turn_openai(
    client,
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str,
    tools_openai: list[dict],
) -> str:
    sys_msg = {"role": "system", "content": system}
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = min(8192, AGENT_MAX_COMPLETION_TOKENS_CAP)

    while True:
        # Prune first: clamp uses trimmed JSON — clamp-before-prune made max_tokens→1 spuriously.
        pressure = 0
        response = None
        mt = max_tokens
        for attempt in range(5):
            prune_openai_messages(messages, mt, pressure_tokens=pressure)
            mt = _clamp_openai_max_tokens(sys_msg, messages, max_tokens)
            if mt < max_tokens:
                prune_openai_messages(messages, mt, pressure_tokens=pressure)
                mt = _clamp_openai_max_tokens(sys_msg, messages, max_tokens)
            if attempt == 0 and mt < max_tokens:
                console.print(
                    f"[dim]本轮 max_tokens 收紧为 {mt}（修剪后估算仍接近上限，为输出预留 token）[/dim]"
                )
            try:
                response = _run_with_activity(
                    console,
                    "Waiting for model response…",
                    lambda mt_=mt: client.chat.completions.create(
                        model=model,
                        max_tokens=mt_,
                        messages=[sys_msg] + messages,
                        tools=tools_openai,
                        tool_choice="auto",
                    ),
                )
                break
            except Exception as e:
                if not _openai_context_length_error(e) or attempt >= 4:
                    if _openai_context_length_error(e):
                        console.print(
                            "[yellow]仍超出服务端上下文：试 /compact、缩短对话，或调低 AGENT_MAX_TOOL_CHARS / "
                            "提高 AGENT_PROMPT_OVERHEAD_TOKENS（使修剪更狠）。[/yellow]"
                        )
                    raise
                pressure += 6000
                mt = max(AGENT_MIN_COMPLETION_TOKENS, mt // 2)
                console.print(
                    "[dim]检测到上下文超限，自动加压修剪并降低 max_tokens 后重试…[/dim]"
                )
        assert response is not None

        choice = response.choices[0]
        msg = choice.message

        text_full = _assistant_message_text(msg.content)
        api_tcs = msg.tool_calls or []
        text_out = text_full
        fallback_specs: list[dict] | None = None
        if (
            not api_tcs
            and text_full
            and os.environ.get("AGENT_DISABLE_HERMES_FALLBACK", "").strip() != "1"
        ):
            parsed = _parse_hermes_json_tool_calls(text_full)
            if parsed:
                text_out, fallback_specs = parsed
                console.print(
                    "[dim]已用客户端回退解析 <tool_call> 内 JSON（服务端未返回 tool_calls）[/dim]"
                )

        assistant_msg: dict = {"role": "assistant", "content": text_out}
        if api_tcs:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in api_tcs
            ]
        elif fallback_specs:
            assistant_msg["tool_calls"] = [
                {
                    "id": s["id"],
                    "type": "function",
                    "function": {
                        "name": s["name"],
                        "arguments": json.dumps(s["arguments"], ensure_ascii=False),
                    },
                }
                for s in fallback_specs
            ]
        messages.append(assistant_msg)

        if text_out.strip():
            render_text(text_out)

        if not api_tcs and not fallback_specs:
            return text_out

        if api_tcs:
            for tc in api_tcs:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = _exec_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            for s in fallback_specs:
                result = _exec_tool(s["name"], s["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": s["id"],
                    "content": result,
                })

        # Must loop again so the model sees tool output (including Error: …) and can fix args / retry.
        # vLLM often sets finish_reason to "stop" even when the assistant message contained tool_calls.


# ── Backend: Anthropic ───────────────────────────────────────────────────────

def run_turn_anthropic(
    client,
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str,
    tools_anthropic: list[dict],
) -> str:
    while True:
        prune_anthropic_messages(messages)
        response = _run_with_activity(
            console,
            "Waiting for Claude API…",
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools_anthropic,
                messages=messages,
            ),
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        has_tool_use = any(b.type == "tool_use" for b in assistant_content)

        for block in assistant_content:
            if block.type == "text" and block.text.strip():
                render_text(block.text)

        if not has_tool_use:
            return "\n".join(b.text for b in assistant_content if b.type == "text")

        tool_results = []
        for block in assistant_content:
            if block.type != "tool_use":
                continue
            result = _exec_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        # Loop again so the model reads tool_result (errors included) and can correct tool calls.


# ── CLI ───────────────────────────────────────────────────────────────────────

@cli.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="One-shot prompt (skip REPL)"),
    local: bool = typer.Option(True, "--local/--anthropic",
                               help="Use local vLLM server (default) or Anthropic API"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
    server: str = typer.Option("http://localhost:8080", "--server", "-S",
                               help="Local server URL (for --local mode)"),
    max_tokens: int = typer.Option(
        65536,
        "--max-tokens",
        help="单次补全请求上限（默认较大；实际会按上下文 min 到可用窗口）。可设 AGENT_MAX_COMPLETION_TOKENS / AGENT_MIN_COMPLETION_TOKENS。",
    ),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="Override system prompt"),
    working_dir: Optional[str] = typer.Option(None, "--cwd", "-C", help="Working directory"),
    kb: bool = typer.Option(
        False,
        "--kb",
        help="向量知识库：PDF/文本入库、语义检索、长期记忆（需 chromadb sentence-transformers pymupdf）",
    ),
    kb_dir: Optional[str] = typer.Option(
        None,
        "--kb-dir",
        help="Chroma 持久化目录（默认: 项目下 .agent_kb）",
    ),
    kb_embed_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--kb-embed-model",
        help="sentence-transformers 模型名（首次会下载权重）",
    ),
    kb_auto: bool = typer.Option(
        False,
        "--kb-auto",
        help="每轮用户消息前自动拼接最相关的知识库片段（仍可用 kb_search 精查）",
    ),
):
    if working_dir:
        os.chdir(working_dir)

    sys_prompt = system or SYSTEM_PROMPT.format(cwd=Path.cwd())

    kb_store = None
    agent_knowledge.set_kb_store(None)
    if kb:
        if not agent_knowledge.kb_deps_available():
            console.print("[bold red]Error:[/bold red] 知识库依赖未安装。")
            console.print(f"  {agent_knowledge.kb_install_hint()}")
            raise typer.Exit(1)
        kdir = Path(kb_dir).expanduser() if kb_dir else PROJECT / ".agent_kb"
        try:
            kb_store = agent_knowledge.AgentKnowledgeBase(kdir, embed_model=kb_embed_model)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] 无法初始化知识库: {e}")
            raise typer.Exit(1)
        agent_knowledge.set_kb_store(kb_store)

    tools_anthropic, tools_openai = build_tool_schemas(kb_store is not None)

    if local:
        from openai import OpenAI
        client = OpenAI(base_url=f"{server}/v1", api_key="not-needed")
        if model:
            model_name = model
        else:
            try:
                models = client.models.list()
                model_name = models.data[0].id if models.data else "qwen3.5-9b"
            except Exception:
                model_name = "qwen3.5-9b"

        try:
            from gpu_select import get_gpu_vram_gb, load_config, max_model_len_for_vram

            cfg_y = load_config(PROJECT / "config.yaml")
            _raw_mlen = (cfg_y.get("vllm") or {}).get("max_model_len", 262_144)
            if isinstance(_raw_mlen, (list, tuple)) and _raw_mlen:
                _raw_mlen = _raw_mlen[0]
            cfg_m = int(_raw_mlen)
            cap = min(cfg_m, max_model_len_for_vram(get_gpu_vram_gb()))
            env_ov = os.environ.get("AGENT_ASSUMED_MAX_MODEL_LEN", "").strip()
            if env_ov.isdigit():
                cap = int(env_ov)
            set_effective_context_len(cap)
        except Exception:
            env_ov = os.environ.get("AGENT_ASSUMED_MAX_MODEL_LEN", "").strip()
            set_effective_context_len(
                int(env_ov) if env_ov.isdigit() else AGENT_ASSUMED_MAX_MODEL_LEN
            )

        def do_turn(msgs: list[dict]) -> str:
            return run_turn_openai(client, msgs, model_name, max_tokens, sys_prompt, tools_openai)

        backend_label = f"Local vLLM ({server})"
    else:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set.")
            console.print("  export ANTHROPIC_API_KEY='sk-ant-...'")
            raise typer.Exit(1)
        model_name = model or "claude-sonnet-4-20250514"
        client = anthropic.Anthropic(api_key=api_key)

        def do_turn(msgs: list[dict]) -> str:
            return run_turn_anthropic(client, msgs, model_name, max_tokens, sys_prompt, tools_anthropic)

        backend_label = "Anthropic API"

    tool_line = (
        "file_stat, read_file, write_file, str_replace, shell, glob_search, grep_search, list_directory"
    )
    if kb_store:
        tool_line += ", kb_search, kb_ingest_file, kb_remember"
    kb_panel = ""
    if kb_store:
        kb_panel = f"\nKB: [dim]{kb_store.persist_dir}[/dim] · embed [dim]{kb_embed_model}[/dim]"
        if kb_auto:
            kb_panel += " · [yellow]auto-RAG[/yellow]"

    ctx_hint = ""
    if local and _effective_ctx_len is not None:
        ctx_hint = f"\n上下文压缩参照: [dim]{_effective_ctx_len}[/dim] tokens（min(config vllm.max_model_len, 显存档位)；可 export AGENT_ASSUMED_MAX_MODEL_LEN=覆盖）"

    console.print(Panel(
        f"[bold cyan]Code Agent[/bold cyan]\n"
        f"Backend: [yellow]{backend_label}[/yellow] · Model: [yellow]{model_name}[/yellow]\n"
        f"CWD: [dim]{Path.cwd()}[/dim]{kb_panel}{ctx_hint}\n"
        f"Tools: {tool_line}\n"
        f"[dim]write/str_replace & risky shell need typing 'yes' to confirm[/dim]",
        title="[bold]Agent[/bold]",
        border_style="cyan",
    ))

    if prompt:
        user_content = prompt
        if kb_store and kb_auto:
            ctx = kb_store.context_for_prompt(prompt, top_k=5)
            if ctx:
                user_content = f"{ctx}\n\n【当前用户消息】\n{prompt}"
        messages = [{"role": "user", "content": user_content}]
        do_turn(messages)
        return

    console.print(
        "[dim]Commands: /quit  /clear  /model  /compact  /history  /peek · "
        "Ctrl+O: 切换工具输出 全文/折叠预览[/dim]"
    )
    console.print("[dim]Press Ctrl+C to interrupt, Ctrl+D to exit[/dim]\n")

    messages: list[dict] = []
    history = FileHistory(str(HISTORY_FILE))
    kb = KeyBindings()

    @kb.add("c-o")
    def _ctrl_o_toggle_fold(_event) -> None:
        toggle_tool_output_fold_mode()

    prompt_session = PromptSession(
        history=history,
        key_bindings=kb,
        multiline=False,
    )

    while True:
        try:
            user_input = prompt_session.prompt("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower().strip()
        if cmd_lower in ("/quit", "/exit", "/q"):
            console.print("[dim]Bye![/dim]")
            break
        elif cmd_lower == "/clear":
            messages.clear()
            console.print("[green]Context cleared.[/green]")
            continue
        elif cmd_lower.startswith("/model "):
            model_name = user_input.split(None, 1)[1].strip()
            console.print(f"[green]Model -> {model_name}[/green]")
            continue
        elif cmd_lower == "/compact":
            if len(messages) > 4:
                messages = messages[-4:]
                console.print("[green]Compacted to last 2 turns.[/green]")
            else:
                console.print("[dim]Already compact.[/dim]")
            continue
        elif cmd_lower == "/history":
            for i, m in enumerate(messages):
                role = m["role"].upper()
                if isinstance(m["content"], str):
                    snippet = m["content"][:80].replace("\n", " ")
                elif isinstance(m["content"], list):
                    snippet = f"[{len(m['content'])} blocks]"
                else:
                    snippet = str(m["content"])[:80]
                console.print(f"  [{i}] {role}: {snippet}")
            continue
        elif cmd_lower == "/peek":
            if not _last_tool_result_full.strip():
                console.print("[dim]尚无已缓存的工具输出。[/dim]")
            else:
                console.print(
                    Panel(
                        _last_tool_result_full,
                        title=f"上次工具输出 · {_last_tool_result_label}",
                        border_style="dim",
                        title_align="left",
                    )
                )
            continue

        user_content = user_input
        if kb_store and kb_auto:
            ctx = kb_store.context_for_prompt(user_input, top_k=5)
            if ctx:
                user_content = f"{ctx}\n\n【当前用户消息】\n{user_input}"
        messages.append({"role": "user", "content": user_content})

        console.print()
        try:
            do_turn(messages)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            messages.pop()

        console.print()


if __name__ == "__main__":
    cli()
