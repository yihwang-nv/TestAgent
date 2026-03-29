#!/usr/bin/env python3
"""
fix_config.py — Patch the model's config.json for TensorRT-LLM compatibility.

The uploaded checkpoint uses model_type "qwen3_5" (Qwen3.5 VLM wrapper) with
the real model config nested inside "text_config".  Neither transformers 4.57
nor TRT-LLM recognise "qwen3_5" directly.

The text backbone is architecturally identical to "qwen3_next" (same hybrid
linear/full-attention design, same field names).  TRT-LLM ships
modeling_qwen3_next.py which supports this architecture natively.

This script:
  1. Reads config.json
  2. Promotes "text_config" fields to the top level
  3. Sets model_type = "qwen3_next", architectures = ["Qwen3NextForCausalLM"]
  4. Extracts rope_theta from the nested rope_parameters dict
  5. Removes VLM-only fields (vision_config, image/video token ids, etc.)
  6. Writes the patched config back (idempotent — skips if already patched)

Run once after downloading the model, or re-run safely at any time.
"""

import json
import shutil
from pathlib import Path

import yaml

PROJECT = Path(__file__).parent
CONFIG  = PROJECT / "config.yaml"


def _model_dir() -> Path:
    c = yaml.safe_load(CONFIG.read_text())
    return Path(c["model"]["local_dir"])


# Fields to drop from the VLM wrapper (not meaningful for text-only inference)
_VLM_FIELDS = {
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "processor_config",
    "model_name",          # informal name field, can confuse loaders
}

# Fields to strip from text_config that we don't want to carry over
# (bos_token_id is None in text_config; we use the outer eos/pad)
_TEXT_STRIP = {"model_type"}


def patch_config(model_dir: Path) -> bool:
    """
    Patch config.json in-place.  Returns True if a patch was applied,
    False if it was already patched (idempotent).
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        print(f"  fix_config: config.json not found at {cfg_path} — skipping.")
        return False

    cfg = json.loads(cfg_path.read_text())

    if cfg.get("model_type") == "qwen3_next":
        print("  fix_config: config.json already patched (model_type=qwen3_next) — skipping.")
        return False

    if cfg.get("model_type") != "qwen3_5":
        mt = cfg.get("model_type", "<missing>")
        print(f"  fix_config: unexpected model_type '{mt}' — skipping (only handles qwen3_5).")
        return False

    text = cfg.get("text_config", {})
    if not text:
        print("  fix_config: no text_config found — skipping.")
        return False

    # ── Back up the original ──────────────────────────────────────────────────
    bak = cfg_path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"  fix_config: original backed up to {bak.name}")

    # ── Build the patched config ──────────────────────────────────────────────
    new_cfg: dict = {}

    # 1. Promote text_config fields (drop model_type — we set it explicitly)
    for k, v in text.items():
        if k not in _TEXT_STRIP:
            new_cfg[k] = v

    # 2. Extract rope_theta from nested rope_parameters dict
    rope_params = text.get("rope_parameters", {})
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        new_cfg["rope_theta"] = rope_params["rope_theta"]
    # partial_rotary_factor already promoted from text_config above

    # Remove the nested rope_parameters dict (qwen3_next uses flat fields)
    new_cfg.pop("rope_parameters", None)

    # 3. Merge outer token IDs (prefer outer eos/pad over text_config nulls)
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        outer_val = cfg.get(field)
        if outer_val is not None:
            new_cfg[field] = outer_val

    # 4. Carry over any remaining outer-level fields that aren't VLM-only
    for k, v in cfg.items():
        if k in _VLM_FIELDS or k == "text_config":
            continue
        if k not in new_cfg:          # don't overwrite text_config values
            new_cfg[k] = v

    # 5. Set the patched identifiers
    new_cfg["model_type"]    = "qwen3_next"
    new_cfg["architectures"] = ["Qwen3NextForCausalLM"]

    # ── Write ─────────────────────────────────────────────────────────────────
    cfg_path.write_text(json.dumps(new_cfg, indent=2) + "\n")
    print("  fix_config: config.json patched  "
          "(qwen3_5 → qwen3_next / Qwen3NextForCausalLM)")
    return True


def main():
    model_dir = _model_dir()
    if not model_dir.is_dir():
        print(f"  fix_config: model not yet downloaded at {model_dir} — skipping.")
        return
    patch_config(model_dir)


if __name__ == "__main__":
    main()
