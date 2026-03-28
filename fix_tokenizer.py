#!/usr/bin/env python3
"""
fix_tokenizer.py — Patch tokenizer_config.json after model download.

The uploaded model has a broken tokenizer class name:
  "tokenizer_class": "TokenizersBackend"

Correct class for Qwen2/Qwen3 models is "Qwen2TokenizerFast".
Using the generic "PreTrainedTokenizerFast" causes a secondary error
because vLLM's processor checks for the Qwen2-specific class:
  TypeError: Received CachedPreTrainedTokenizerFast but ('Qwen2Tokenizer',
             'Qwen2TokenizerFast') was expected.

This script patches the file in-place. Safe to re-run — idempotent.
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT  = Path(__file__).parent
CONFIG   = PROJECT / "config.yaml"
GOOD_CLS = "Qwen2TokenizerFast"


def main():
    cfg       = yaml.safe_load(CONFIG.read_text())
    model_dir = Path(cfg["model"]["local_dir"])
    tok_cfg   = model_dir / "tokenizer_config.json"

    if not tok_cfg.exists():
        sys.exit(f"ERROR: {tok_cfg} not found. Run: python download_model.py")

    data    = json.loads(tok_cfg.read_text())
    current = data.get("tokenizer_class", "(not set)")

    if current == GOOD_CLS:
        print(f"tokenizer_config.json already correct ({current}). Nothing to do.")
        return

    data["tokenizer_class"] = GOOD_CLS
    tok_cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Patched tokenizer_class: {current!r} → {GOOD_CLS!r}")
    print(f"File: {tok_cfg}")


if __name__ == "__main__":
    main()
