#!/usr/bin/env python3
"""
Download Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled from Hugging Face.
Downloads the full safetensors model (~19 GB).
"""

import sys
import yaml
import typer
from pathlib import Path
from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()
PROJECT     = Path(__file__).parent
CONFIG_PATH = PROJECT / "config.yaml"

app = typer.Typer(add_completion=False)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@app.command()
def main(
    token: str = typer.Option(None, "--token", "-t", envvar="HF_TOKEN",
                               help="HuggingFace token (needed for gated models)"),
):
    cfg       = load_config()
    repo      = cfg["model"]["hf_repo"]
    local_dir = PROJECT / cfg["model"]["local_dir"]

    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        console.print(f"[green]Model already downloaded:[/green] {local_dir}")
        console.print("Delete the folder to re-download.")
        raise typer.Exit(0)

    console.print(f"[bold cyan]Repo:[/bold cyan]  {repo}")
    console.print(f"[bold cyan]Dest:[/bold cyan]  {local_dir}")
    console.print("[yellow]Size: ~19 GB — ensure you have enough disk space.[/yellow]")
    console.print("[dim]Downloading all model files (safetensors + tokenizer)...[/dim]\n")

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token or None,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        console.print(f"\n[bold green]Download complete:[/bold green] {local_dir}")
        console.print("Run [bold]python chat.py[/bold] to start chatting.")
    except KeyboardInterrupt:
        console.print("\n[red]Download interrupted.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
