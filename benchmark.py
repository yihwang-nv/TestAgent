#!/usr/bin/env python3
"""
Throughput benchmark for Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled on RTX 5070.
Reports prompt-processing (PP) and token-generation (TG) speed.
"""

import time
import yaml
import torch
import typer
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rich.console import Console
from rich.table import Table

console = Console()
PROJECT = Path(__file__).parent

TEST_PROMPTS = [
    "Explain the transformer attention mechanism in detail.",
    "Write a Python function that implements binary search with type hints.",
    "What are the key differences between MoE and dense language models?",
]


def load_config():
    with open(PROJECT / "config.yaml") as f:
        return yaml.safe_load(f)


def main(
    quant: str  = typer.Option(None, "--quant", "-q", help="4bit | 8bit | none"),
    runs:  int  = typer.Option(3, help="Benchmark runs per prompt"),
    n_gen: int  = typer.Option(64, help="Tokens to generate per run"),
):
    cfg       = load_config()
    model_cfg = cfg["model"]
    model_dir = PROJECT / model_cfg["local_dir"]

    quantization = quant or model_cfg.get("quantization", "4bit")
    torch_dtype  = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))

    if not model_dir.exists():
        console.print(f"[red]Model not found:[/red] {model_dir}")
        raise typer.Exit(1)

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    console.print(f"Loading model [{quantization}]...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype if quantization == "none" else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.eval()

    table = Table(
        title=f"Qwen3.5-9B Reasoning Distilled Benchmark — RTX 5070 (GPU layers: ALL)"
    )
    table.add_column("Prompt", style="cyan", max_width=40)
    table.add_column("Prompt tokens", justify="right")
    table.add_column("PP speed (t/s)", justify="right", style="green")
    table.add_column("TG speed (t/s)", justify="right", style="yellow")

    for prompt in TEST_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        n_prompt = input_ids.shape[-1]
        pp_speeds, tg_speeds = [], []

        for _ in range(runs):
            # Prompt processing speed
            with torch.no_grad():
                t0 = time.perf_counter()
                _ = model(input_ids)
                torch.cuda.synchronize()
                pp_time = time.perf_counter() - t0
            pp_speeds.append(n_prompt / pp_time)

            # Token generation speed
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=n_gen,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            tg_time = time.perf_counter() - t1
            generated = out.shape[-1] - n_prompt
            tg_speeds.append(generated / tg_time)

        avg_pp = sum(pp_speeds) / len(pp_speeds)
        avg_tg = sum(tg_speeds) / len(tg_speeds)

        table.add_row(
            prompt[:38] + ("…" if len(prompt) > 38 else ""),
            str(n_prompt),
            f"{avg_pp:.1f}",
            f"{avg_tg:.1f}",
        )

    console.print(table)
    console.print(
        "[dim]PP = prompt processing, TG = token generation. "
        "Expect ~60–120 TG t/s for 9B Q5_K_M fully on RTX 5070 VRAM.[/dim]"
    )


if __name__ == "__main__":
    typer.run(main)
