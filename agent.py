#!/usr/bin/env python3
"""
Agent that talks directly to the local Qwen3.5-9B inference server.
No ANTHROPIC_API_KEY required — 100% local.

Features:
  • Streaming responses with live token display
  • Multi-turn conversation memory
  • <think>...</think> reasoning blocks shown dimmed (or hidden with --hide-thinking)
  • Tool calling via Qwen3's native ChatML tool format
  • /clear /history /stats /quit commands

Usage:
  # 1. Start the inference server:
  #    python server.py --port 8080

  # 2. Run:
  python agent.py --repl
  python agent.py "Solve: integrate x^2 sin(x) dx"
"""

import os
import sys
import json
import httpx
import typer
from pathlib import Path
from typing import Optional

PROJECT    = Path(__file__).parent
SERVER_URL = os.environ.get("LOCAL_MODEL_URL", "http://localhost:8080")

cli = typer.Typer(add_completion=False)

SYSTEM_PROMPT = (
    "You are a highly capable AI assistant powered by Qwen3.5-9B, "
    "distilled from Claude 4.6 Opus reasoning patterns.\n\n"
    "For complex questions, use your <think>...</think> blocks to reason "
    "step-by-step before giving your final answer.\n\n"
    "Be concise, precise, and direct. Show your reasoning when it adds clarity."
)

# ── Server helpers ────────────────────────────────────────────────────────────

def check_server() -> dict | None:
    try:
        with httpx.Client(timeout=4.0) as http:
            r = http.get(f"{SERVER_URL}/health")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


def stream_chat(
    messages: list[dict],
    max_tokens:  int   = 2048,
    temperature: float = 0.6,
    top_p:       float = 0.95,
):
    """Stream tokens from the local server, yield (token_text, in_think) tuples."""
    payload = {
        "model":       "qwen3.5-9b-local",
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       top_p,
        "stream":      True,
    }

    buffer   = ""
    in_think = False

    with httpx.Client(timeout=None) as http:
        with http.stream("POST", f"{SERVER_URL}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if not delta:
                        continue

                    buffer += delta

                    # Track <think> state across chunks
                    if "<think>" in buffer and not in_think:
                        in_think = True
                    if "</think>" in buffer and in_think:
                        in_think = False
                        yield delta, False  # closing tag — reset colour after
                        buffer = ""
                        continue

                    yield delta, in_think
                    buffer = ""

                except (json.JSONDecodeError, KeyError):
                    continue


def chat_once(messages: list[dict], show_thinking: bool, **gen_kwargs) -> str:
    """Stream one assistant turn; return the full response text."""
    full    = ""
    started = False

    try:
        for token, in_think in stream_chat(messages, **gen_kwargs):
            if not started:
                started = True

            full += token

            if not show_thinking and (in_think or "<think>" in token):
                continue  # suppress thinking tokens

            if in_think and show_thinking:
                print(f"\x1b[2m{token}\x1b[0m", end="", flush=True)
            elif "</think>" in token and show_thinking:
                print(f"\x1b[2m{token}\x1b[0m", end="", flush=True)
            else:
                # Strip residual think tags from visible output
                visible = token
                if not show_thinking:
                    visible = visible.replace("<think>", "").replace("</think>", "")
                print(visible, end="", flush=True)

    except httpx.ConnectError:
        print(
            f"\n[error] Cannot reach server at {SERVER_URL}. "
            "Run: python server.py --port 8080"
        )
    except httpx.HTTPStatusError as e:
        print(f"\n[error] Server returned {e.response.status_code}")
    except KeyboardInterrupt:
        print("\n[interrupted]")

    print()  # final newline
    return full


# ── CLI ───────────────────────────────────────────────────────────────────────

@cli.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="Single prompt (non-interactive)"),
    repl:   bool = typer.Option(False, "--repl", "-r", help="Interactive REPL"),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--hide-thinking",
        help="Show <think> reasoning blocks (dimmed)",
    ),
    max_tokens:  int   = typer.Option(2048,  "--max-tokens",  "-m"),
    temperature: float = typer.Option(0.6,   "--temperature", "-t"),
    top_p:       float = typer.Option(0.95,  "--top-p"),
    server_url: Optional[str] = typer.Option(
        None, "--server", "-s", help=f"Server URL (default: {SERVER_URL})"
    ),
    system: Optional[str] = typer.Option(None, "--system", help="Override system prompt"),
):
    global SERVER_URL
    if server_url:
        SERVER_URL = server_url

    info = check_server()
    if not info:
        typer.echo(
            f"[error] Local model server not reachable at {SERVER_URL}.\n"
            "        Start it first: python server.py --port 8080",
            err=True,
        )
        raise typer.Exit(1)

    sys_prompt = system or SYSTEM_PROMPT
    gen_kwargs = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    if not repl and prompt:
        messages = [
            {"role": "system",  "content": sys_prompt},
            {"role": "user",    "content": prompt},
        ]
        chat_once(messages, show_thinking, **gen_kwargs)
        return

    # ── REPL ──────────────────────────────────────────────────────────────────
    model_name = info.get("model", "local")
    print(f"[server] {model_name} ready on {SERVER_URL}")
    print("Qwen3.5-9B Agent REPL  (/quit /clear /history /stats)\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # Built-in commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("Bye.")
            break
        elif cmd == "/clear":
            history.clear()
            print("[context cleared]")
            continue
        elif cmd == "/history":
            if not history:
                print("[empty]")
            for i, m in enumerate(history):
                role    = m["role"].upper()
                snippet = str(m["content"])[:80].replace("\n", " ")
                print(f"  [{i}] {role}: {snippet}{'…' if len(str(m['content'])) > 80 else ''}")
            continue
        elif cmd == "/stats":
            chars = sum(len(str(m["content"])) for m in history)
            print(f"  Turns: {len(history)//2}  |  Context chars: {chars}")
            print(f"  Server: {SERVER_URL}  |  Model: {model_name}")
            print(f"  temperature={temperature}  max_tokens={max_tokens}  top_p={top_p}")
            continue

        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": sys_prompt}] + history

        print(f"\n\x1b[1;36m[Qwen3.5-9B]\x1b[0m ", end="", flush=True)
        response = chat_once(messages, show_thinking, **gen_kwargs)

        if response.strip():
            # Strip think blocks from stored history to keep context lean
            clean = response
            if "<think>" in clean and "</think>" in clean:
                import re
                clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
            history.append({"role": "assistant", "content": clean or response})

        print()


if __name__ == "__main__":
    cli()
