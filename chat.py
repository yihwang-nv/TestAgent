#!/usr/bin/env python3
"""
Interactive CLI chat with a local causal LM from config.yaml (transformers + optional bitsandbytes).
Model may emit vendor-specific reasoning blocks before final answers, depending on the checkpoint.
"""

import sys
import yaml
import typer
import torch
from pathlib import Path
from typing import Optional
from threading import Thread
from rich.console import Console
from rich.panel import Panel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

app = Console()
cli = typer.Typer(add_completion=False)
PROJECT = Path(__file__).parent

QUANT_VRAM = {"4bit": "~5 GB", "8bit": "~10 GB", "none": "~19 GB"}


def load_config():
    with open(PROJECT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_bnb_config(quantization: str, torch_dtype) -> Optional[BitsAndBytesConfig]:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


@cli.command()
def main(
    quant: Optional[str] = typer.Option(None, "--quant", "-q",
        help="Override quantization: 4bit | 8bit | none"),
    system: str = typer.Option(
        "You are a helpful, precise, and concise AI assistant.",
        help="System prompt",
    ),
    show_thinking: bool = typer.Option(True, "--show-thinking/--hide-thinking",
        help="Show or hide <think> reasoning blocks"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    cfg         = load_config()
    size_key    = cfg.get("chat_model_size", "4b")
    models      = cfg.get("models") or {}
    if size_key not in models:
        app.print(f"[red]chat_model_size {size_key!r} not in config models[/red]")
        raise typer.Exit(1)
    model_entry = dict(models[size_key])
    defaults    = cfg.get("model_defaults") or {}
    gen_cfg     = cfg["generation"]

    ld = model_entry.get("local_dir", "")
    model_dir = PROJECT / ld if not Path(ld).is_absolute() else Path(ld)

    quantization = quant or model_entry.get("quantization", "4bit")
    torch_dtype  = getattr(torch, model_entry.get("torch_dtype", defaults.get("torch_dtype", "bfloat16")))

    if not model_dir.exists() or not any(model_dir.glob("*.safetensors")):
        app.print(f"[red]Model not found:[/red] {model_dir}")
        app.print("Run [bold]python download_model.py --model-size <size>[/bold] first.")
        raise typer.Exit(1)

    vram_hint = QUANT_VRAM.get(quantization, "?")
    app.print(Panel(
        f"[bold cyan]Qwen3.5 ({size_key}) · Reasoning Distilled[/bold cyan]\n"
        f"Quantization: [yellow]{quantization}[/yellow] ({vram_hint} VRAM) • "
        f"dtype: [yellow]{model_entry.get('torch_dtype', defaults.get('torch_dtype', 'bfloat16'))}[/yellow]\n"
        f"[dim]{model_dir.name}[/dim]",
        title="[bold]Local chat (transformers)[/bold]",
    ))

    app.print("[dim]Loading model...[/dim]")
    bnb_config = build_bnb_config(quantization, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    load_kwargs = dict(
        pretrained_model_name_or_path=str(model_dir),
        device_map=model_entry.get("device_map", "auto"),
        torch_dtype=torch_dtype if quantization == "none" else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    if verbose:
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()
    app.print("[green]Model loaded.[/green]")

    app.print("[dim]Commands: /quit  /clear  /stats  /quant <4bit|8bit>[/dim]")
    if show_thinking:
        app.print("[dim]Tip: <think> reasoning shown in dim — use --hide-thinking to suppress[/dim]")
    app.print("")

    history: list[dict] = []

    while True:
        try:
            user_input = input("[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            app.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd in ("/quit", "/exit", "/q"):
                app.print("[dim]Bye.[/dim]")
                break
            elif cmd == "/clear":
                history.clear()
                app.print("[green]Context cleared.[/green]")
                continue
            elif cmd == "/stats":
                app.print(f"[bold]Turns in context:[/bold] {len(history) // 2}")
                app.print(f"[bold]Quantization:[/bold] {quantization}")
                if torch.cuda.is_available():
                    used  = torch.cuda.memory_allocated() / 1e9
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    app.print(f"[bold]VRAM:[/bold] {used:.1f} GB / {total:.1f} GB")
                continue
            elif cmd.startswith("/quant "):
                app.print("[yellow]Changing quantization requires restarting the process.[/yellow]")
                app.print("Run: [bold]python chat.py --quant <4bit|8bit>[/bold]")
                continue
            else:
                app.print(f"[red]Unknown command:[/red] {user_input}")
                continue

        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system}] + history

        # Apply chat template — generation prompt seeds <|im_start|>assistant\n<think>\n
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
            do_sample=gen_cfg.get("do_sample", True),
            temperature=gen_cfg.get("temperature", 0.6),
            top_p=gen_cfg.get("top_p", 0.95),
            top_k=gen_cfg.get("top_k", 40),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.05),
            pad_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print(f"\n\x1b[1;36m[Qwen3.5-9B]\x1b[0m ", end="", flush=True)

        full_response = ""
        in_think      = False

        for token_text in streamer:
            full_response += token_text

            if not show_thinking:
                # Buffer and strip <think>...</think> blocks entirely
                continue

            # Render <think> blocks dimmed
            if "<think>" in token_text and not in_think:
                in_think = True
                token_text = token_text.replace("<think>", "\x1b[2m<think>")
            if "</think>" in token_text and in_think:
                in_think = False
                token_text = token_text.replace("</think>", "</think>\x1b[0m")

            print(token_text, end="", flush=True)

        if not show_thinking:
            # Print only the part after </think>
            if "</think>" in full_response:
                answer = full_response.split("</think>", 1)[-1].strip()
            else:
                answer = full_response.strip()
            print(answer, flush=True)

        if in_think:
            print("\x1b[0m", end="", flush=True)  # safety ANSI reset
        print()

        thread.join()
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    cli()
