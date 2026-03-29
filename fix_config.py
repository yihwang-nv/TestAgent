#!/usr/bin/env python3
"""
fix_config.py — Prepare a TensorRT-LLM-compatible model directory.

The original checkpoint uses model_type "qwen3_5" (Qwen3.5 VLM wrapper).
vLLM 0.18+ handles this natively — the original model dir is left untouched.

TensorRT-LLM needs two things:
  1. model_type = "qwen3_next" (the text backbone architecture)
  2. Weights remapped: model.language_model.X → model.X
     (drop vision weights model.visual.*, keep mtp.*, lm_head.*)

This script creates a NEW directory alongside the original:
  <original_dir>-trtllm/
    config.json          — patched (qwen3_next, flattened)
    tokenizer*.json      — symlinked from original
    model.safetensors.*  — remapped safetensors shards (text-only keys)

Run once before starting the TRT-LLM server.  Safe to re-run (idempotent).
Expected runtime: 5–15 min (IO-bound, depends on storage speed).
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).parent
CONFIG  = PROJECT / "config.yaml"


def _model_dirs() -> tuple[Path, Path]:
    c = yaml.safe_load(CONFIG.read_text())
    orig = Path(c["model"]["local_dir"])
    trtllm = orig.parent / (orig.name + "-trtllm")
    return orig, trtllm


# ── Config patch (same logic as before, now writes to trtllm_dir) ─────────────

_VLM_FIELDS = {
    "vision_config", "image_token_id", "video_token_id",
    "vision_start_token_id", "vision_end_token_id",
    "processor_config", "model_name",
}
_TEXT_STRIP = {"model_type"}


def _make_trtllm_config(orig_dir: Path, trtllm_dir: Path):
    cfg = json.loads((orig_dir / "config.json").read_text())

    if cfg.get("model_type") != "qwen3_5":
        mt = cfg.get("model_type", "<missing>")
        print(f"  fix_config: unexpected model_type '{mt}' — only handles qwen3_5.")
        return False

    text = cfg.get("text_config", {})
    if not text:
        print("  fix_config: no text_config in config.json — cannot patch.")
        return False

    new_cfg: dict = {}
    for k, v in text.items():
        if k not in _TEXT_STRIP:
            new_cfg[k] = v

    rope_params = text.get("rope_parameters", {})
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        new_cfg["rope_theta"] = rope_params["rope_theta"]
    new_cfg.pop("rope_parameters", None)

    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        outer_val = cfg.get(field)
        if outer_val is not None:
            new_cfg[field] = outer_val

    for k, v in cfg.items():
        if k in _VLM_FIELDS or k == "text_config":
            continue
        if k not in new_cfg:
            new_cfg[k] = v

    new_cfg["model_type"]    = "qwen3_next"
    new_cfg["architectures"] = ["Qwen3NextForCausalLM"]

    (trtllm_dir / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    print("  fix_config: config.json written (qwen3_5 → qwen3_next).")
    return True


# ── Weight remap ──────────────────────────────────────────────────────────────

def _remap_weights(orig_dir: Path, trtllm_dir: Path):
    """
    Load each safetensors shard, drop model.visual.* keys, remap
    model.language_model.X → model.X, write to trtllm_dir.
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("  fix_config: safetensors not installed — pip install safetensors")
        sys.exit(1)

    import torch

    idx_path = orig_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        print("  fix_config: model.safetensors.index.json not found.")
        sys.exit(1)

    idx = json.loads(idx_path.read_text())
    weight_map = idx["weight_map"]

    # Determine shard files
    shards = sorted(set(weight_map.values()))
    print(f"  fix_config: remapping {len(shards)} shard(s) …")

    new_weight_map = {}
    t0 = time.time()

    for shard_file in shards:
        src = orig_dir / shard_file
        dst = trtllm_dir / shard_file
        print(f"    {shard_file} …", flush=True)

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(src), framework="pt", device="cpu") as f:
            for key in f.keys():
                # Drop vision weights — not needed for text inference
                if key.startswith("model.visual."):
                    continue
                # Remap: model.language_model.X → model.X
                if key.startswith("model.language_model."):
                    new_key = "model." + key[len("model.language_model."):]
                else:
                    new_key = key
                tensors[new_key] = f.get_tensor(key)

        save_file(tensors, str(dst))
        for new_key in tensors:
            new_weight_map[new_key] = shard_file

    # Write updated index
    new_idx = {
        "metadata": idx.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    (trtllm_dir / "model.safetensors.index.json").write_text(
        json.dumps(new_idx, indent=2) + "\n"
    )

    elapsed = time.time() - t0
    print(f"  fix_config: weights remapped in {elapsed/60:.1f} min.")


# ── Symlink tokenizer files ───────────────────────────────────────────────────

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
]


def _link_tokenizer(orig_dir: Path, trtllm_dir: Path):
    for fname in _TOKENIZER_FILES:
        src = orig_dir / fname
        dst = trtllm_dir / fname
        if src.exists() and not dst.exists():
            os.symlink(src, dst)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    orig_dir, trtllm_dir = _model_dirs()

    if not orig_dir.is_dir():
        print(f"  fix_config: model not found at {orig_dir} — run download_model.py first.")
        return

    if trtllm_dir.exists() and (trtllm_dir / "config.json").exists():
        # Check if already the right type
        existing = json.loads((trtllm_dir / "config.json").read_text())
        if existing.get("model_type") == "qwen3_next":
            print(f"  fix_config: TRT-LLM model dir already prepared at {trtllm_dir} — skipping.")
            return

    print(f"  fix_config: preparing TRT-LLM model dir at {trtllm_dir} …")
    trtllm_dir.mkdir(parents=True, exist_ok=True)

    ok = _make_trtllm_config(orig_dir, trtllm_dir)
    if not ok:
        return

    _link_tokenizer(orig_dir, trtllm_dir)

    # Check if weights already remapped
    idx_dst = trtllm_dir / "model.safetensors.index.json"
    if not idx_dst.exists():
        print(f"  fix_config: remapping weights (one-time, ~5–15 min) …")
        _remap_weights(orig_dir, trtllm_dir)
    else:
        print("  fix_config: weights already remapped — skipping.")

    print(f"  fix_config: done. TRT-LLM model dir: {trtllm_dir}")

    # Print config.yaml update hint
    tc = yaml.safe_load(CONFIG.read_text()).get("tensorrt_llm", {})
    if not tc.get("model_dir"):
        print()
        print("  NOTE: add this to config.yaml → tensorrt_llm section to use the remapped dir:")
        print(f"    model_dir: \"{trtllm_dir}\"")


if __name__ == "__main__":
    main()
