#!/usr/bin/env python3
"""
fix_tokenizer.py — Patch model tokenizer files after download.

Two bugs in the uploaded model's tokenizer are fixed here:

1. tokenizer_config.json — wrong tokenizer class
   "tokenizer_class": "TokenizersBackend"  →  "Qwen2TokenizerFast"
   TokenizersBackend doesn't exist in transformers.
   Qwen2TokenizerFast is the correct class for Qwen2/Qwen3 models.

2. tokenizer.json — Mistral-copied pre-tokenizer regex (fix_mistral_regex)
   The regex was copied from Mistral models and differs from the correct
   Qwen2 tokenizer regex in two ways:
     [\\p{L}\\p{M}]+             ->  \\p{L}+      (drop \\p{M} mark category)
     \\p{N}                     ->  \\p{N}{1,3}  (add quantifier)
     [^\\s\\p{L}\\p{M}\\p{N}]+ ->  [^\\s\\p{L}\\p{N}]+  (drop \\p{M})
   vLLM warns: "incorrect regex pattern ... will lead to incorrect tokenization"

This script patches both files in-place. Safe to re-run — idempotent.
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT  = Path(__file__).parent
CONFIG   = PROJECT / "config.yaml"
GOOD_CLS = "Qwen2TokenizerFast"

# Mistral-copied bad regex → correct Qwen2 regex
BAD_REGEX  = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GOOD_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def fix_tokenizer_config(model_dir: Path) -> bool:
    """Patch tokenizer_config.json: fix tokenizer_class. Returns True if changed."""
    path    = model_dir / "tokenizer_config.json"
    data    = json.loads(path.read_text())
    current = data.get("tokenizer_class", "(not set)")

    if current == GOOD_CLS:
        print(f"  tokenizer_config.json: already correct ({GOOD_CLS})")
        return False

    data["tokenizer_class"] = GOOD_CLS
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  tokenizer_config.json: {current!r} → {GOOD_CLS!r}")
    return True


def fix_pretokenizer_regex(model_dir: Path) -> bool:
    """Patch tokenizer.json: fix Mistral-copied pre-tokenizer regex. Returns True if changed."""
    path = model_dir / "tokenizer.json"
    data = json.loads(path.read_text())

    changed = False
    pre = data.get("pre_tokenizer", {})

    def _fix_node(node):
        nonlocal changed
        if isinstance(node, dict):
            if node.get("type") == "Split":
                pattern = node.get("pattern", {})
                if isinstance(pattern, dict) and pattern.get("Regex") == BAD_REGEX:
                    pattern["Regex"] = GOOD_REGEX
                    changed = True
            for v in node.values():
                _fix_node(v)
        elif isinstance(node, list):
            for item in node:
                _fix_node(item)

    _fix_node(pre)

    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print("  tokenizer.json: Mistral regex → Qwen2 regex (fix_mistral_regex applied)")
    else:
        print("  tokenizer.json: regex already correct or pattern not found")

    return changed


def main():
    cfg       = yaml.safe_load(CONFIG.read_text())
    model_dir = Path(cfg["model"]["local_dir"])

    if not model_dir.exists():
        sys.exit(f"ERROR: Model not found at {model_dir}\n"
                 "       Run: python download_model.py")

    print(f"Patching tokenizer files in: {model_dir}")
    fix_tokenizer_config(model_dir)
    fix_pretokenizer_regex(model_dir)
    print("Done.")


if __name__ == "__main__":
    main()
