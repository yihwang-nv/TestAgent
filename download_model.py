#!/usr/bin/env python3
"""Download model weights listed in config.yaml from Hugging Face."""

import json
import sys
from pathlib import Path

import typer
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console

from gpu_select import MODEL_SIZE_KEYS, load_config, resolve_paths

console = Console()
PROJECT = Path(__file__).parent
CONFIG_PATH = PROJECT / "config.yaml"

app = typer.Typer(add_completion=False)


def _patch_tokenizer(local_dir: Path) -> None:
    tok = local_dir / "tokenizer_config.json"
    if not tok.exists():
        return
    data = json.loads(tok.read_text())
    cls = data.get("tokenizer_class", "")
    if cls in ("TokenizersBackend", ""):
        data["tokenizer_class"] = "Qwen2TokenizerFast"
        tok.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        console.print(f"  [yellow]Patched tokenizer_class → Qwen2TokenizerFast[/yellow]")


def _download_one(repo: str, local_dir: Path, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        ignore_patterns=["*.md", "*.gitattributes"],
    )
    _patch_tokenizer(local_dir)


@app.command()
def main(
    model_size: str = typer.Option(
        "auto",
        "--model-size",
        "-m",
        help="auto | all | 0.8b | 2b | 4b | 9b | 27b | 35b-a3b",
    ),
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        envvar="HF_TOKEN",
        help="HuggingFace token (gated models)",
    ),
):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if "models" not in cfg:
        console.print("[red]config.yaml missing models:[/red]")
        raise typer.Exit(1)

    ms = model_size.lower().strip()

    def do_one(key: str) -> None:
        info = resolve_paths(cfg, key)
        repo = info["hf_repo"]
        local_dir = Path(info["local_dir"])
        if local_dir.exists() and any(local_dir.glob("*.safetensors")):
            console.print(f"[green]Already present:[/green] {local_dir}")
            return
        console.print(f"[bold cyan]Repo:[/bold cyan]  {repo}")
        console.print(f"[bold cyan]Dest:[/bold cyan]  {local_dir}\n")
        try:
            _download_one(repo, local_dir, token)
            console.print(f"\n[bold green]Done:[/bold green] {local_dir}")
        except KeyboardInterrupt:
            console.print("\n[red]Interrupted.[/red]")
            raise typer.Exit(1)

    if ms == "auto":
        from gpu_select import resolve_model_config

        r = resolve_model_config("auto", "auto", cfg)
        key = r["model_size"]
        console.print(f"[dim]auto → model_size={key}[/dim]")
        do_one(key)
    elif ms == "all":
        for key in sorted(cfg["models"].keys(), key=lambda x: (len(x), x)):
            console.print(f"\n[bold]--- {key} ---[/bold]")
            do_one(key)
    elif ms in MODEL_SIZE_KEYS:
        do_one(ms)
    else:
        console.print(
            f"[red]Invalid --model-size {model_size!r}. Use auto, all, or {sorted(MODEL_SIZE_KEYS)}[/red]"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
