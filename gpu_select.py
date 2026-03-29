#!/usr/bin/env python3
"""
GPU VRAM detection and automatic model + quantization selection.

Used by start_server.sh (CLI JSON output) and server.py (import).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent

# Auto tiers: thresholds are *descending*; first match wins.
# Budget beyond raw weights: vLLM KV cache, CUDA graphs, cudnn/workspace, fragmentation.
# (e.g. ~16 GB cards report ~15.5 GiB — 4B bf16 + max_len 8192 OOMs without 8bit.)
AUTO_TIERS: list[tuple[float, str, str]] = [
    (72.0, "27b", "none"),
    (40.0, "27b", "8bit"),
    (32.0, "9b", "none"),
    (22.0, "4b", "none"),
    (18.0, "9b", "8bit"),
    (8.0, "4b", "8bit"),
    (6.0, "2b", "none"),
    (4.0, "2b", "8bit"),
    (3.0, "0.8b", "none"),
    (0.0, "0.8b", "8bit"),
]

# Min total VRAM (GB) before recommending bf16 (none) for a fixed --model-size + quant=auto.
# Must stay consistent with AUTO_TIERS intent above.
MIN_VRAM_BF16_GB: dict[str, float] = {
    "27b": 72.0,
    "9b": 32.0,
    "4b": 22.0,
    "2b": 7.0,
    "0.8b": 3.5,
    "35b-a3b": 88.0,
}

MODEL_SIZE_KEYS = frozenset({"0.8b", "2b", "4b", "9b", "27b", "35b-a3b"})

VALID_ENGINES = frozenset({"vllm", "tensorrt_llm"})
QUANTS_VLLM = frozenset({"auto", "none", "8bit", "awq"})
QUANTS_TRT = frozenset({"auto", "none", "4bit", "8bit", "awq"})


def default_engine(cfg: dict[str, Any]) -> str:
    return str((cfg.get("inference") or {}).get("engine") or "vllm").lower().strip()


def trt_venv_python(cfg: dict[str, Any]) -> str:
    """Path to the TRT venv interpreter — do NOT resolve() symlinks: realpath skips pyvenv.cfg."""
    sub = (cfg.get("tensorrt_llm") or {}).get("venv_subdir", ".venv_trtllm")
    bindir = PROJECT_DIR / sub / "bin"
    for name in ("python", "python3"):
        p = bindir / name
        if p.is_file():
            return str(p)
    return str(bindir / "python")


def get_gpu_vram_gb() -> float:
    """Return total VRAM (GB) for GPU 0 via nvidia-smi. 0.0 if unavailable."""
    return get_gpu_info()[0]


def get_gpu_info() -> tuple[float, str]:
    """Return (vram_total_gb, gpu_name) for GPU 0."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mib = float(parts[0])
            name = ",".join(parts[1:]).strip()
            return mib / 1024.0, name
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0.0, "unknown"


def max_model_len_for_vram(vram_gb: float) -> int:
    # Upper bound from VRAM; final length is min(config vllm.max_model_len, this).
    # Cap at 262k to match common Qwen3.5 max_position_embeddings; vLLM rejects above without VLLM_ALLOW_LONG_MAX_MODEL_LEN.
    env = os.environ.get("GPU_SELECT_MAX_MODEL_LEN", "").strip()
    if env.isdigit():
        return max(512, min(int(env), 262_144))
    if vram_gb >= 96.0:
        return 262_144
    if vram_gb >= 40.0:
        return 65_536
    if vram_gb >= 24.0:
        return 16_384
    # 8–12 GB: tools + chat template alone can exceed 4k tokens; 8192 is a practical floor for agent use.
    if vram_gb >= 12.0:
        return 8192
    if vram_gb >= 6.0:
        return 8192
    if vram_gb >= 4.0:
        return 4096
    return 2048


def auto_select(
    vram_gb: float,
    model_size: str = "auto",
    quant: str = "auto",
    engine: str = "vllm",
) -> dict[str, Any]:
    """
    Resolve model_size, quant, max_model_len from VRAM and optional overrides.
    For tensorrt_llm + quant=auto, tier "8bit" maps to "4bit" (TRT weight-only / AWQ path).
    """
    _, gpu_name = get_gpu_info()
    auto_mlen = max_model_len_for_vram(vram_gb)

    if model_size == "auto":
        ms, tier_quant, _ = _tier_pick(vram_gb)
    else:
        ms = model_size
        tier_quant = None

    if quant == "auto":
        if model_size == "auto":
            q = tier_quant
            if engine == "tensorrt_llm" and q == "8bit":
                q = "4bit"
        else:
            q = _quant_for_fixed_model(vram_gb, ms, engine=engine)
    else:
        q = quant

    return {
        "model_size": ms,
        "quant": q,
        "max_model_len": auto_mlen,
        "vram_gb": vram_gb,
        "gpu_name": gpu_name,
    }


def _tier_pick(vram_gb: float) -> tuple[str, str, int]:
    mlen = max_model_len_for_vram(vram_gb)
    for threshold, size, q in AUTO_TIERS:
        if vram_gb >= threshold:
            return size, q, mlen
    return "0.8b", "8bit", mlen


def _estimate_vram_need_gb(model_size: str, quant: str, cfg: dict[str, Any]) -> float:
    models = cfg.get("models") or {}
    m = models.get(model_size, {})
    bf16 = float(m.get("size_gb_bf16", 10))
    # Extra slack for KV + CUDA graph capture (vLLM), not just weights.
    slack = 8.0 if quant == "none" else 5.0
    if quant == "8bit":
        return bf16 * 0.55 + slack
    return bf16 + slack


def _quant_for_fixed_model(vram_gb: float, model_size: str, engine: str = "vllm") -> str:
    floor = MIN_VRAM_BF16_GB.get(model_size)
    cfg = load_config()
    est = _estimate_vram_need_gb(model_size, "none", cfg)
    need_bf16 = max(floor or 0.0, est)
    if vram_gb >= need_bf16:
        return "none"
    if engine == "tensorrt_llm":
        return "4bit"
    return "8bit"


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    import yaml

    p = path or (PROJECT_DIR / "config.yaml")
    return yaml.safe_load(p.read_text())


def resolve_paths(cfg: dict[str, Any], model_size: str) -> dict[str, Any]:
    """Resolve local_dir to absolute path; merge per-model fields."""
    models = cfg.get("models") or {}
    if model_size not in models:
        raise KeyError(f"Unknown model_size '{model_size}'. Valid: {sorted(models.keys())}")

    m = dict(models[model_size])
    local = m.get("local_dir", "")
    lp = Path(local)
    if not lp.is_absolute():
        lp = PROJECT_DIR / lp
    m["local_dir"] = str(lp.resolve())

    defaults = cfg.get("model_defaults") or {}
    m.setdefault("torch_dtype", defaults.get("torch_dtype", "bfloat16"))
    if not m.get("served_model_name"):
        m["served_model_name"] = f"qwen3.5-{model_size.replace('.', '-')}"

    vllm = cfg.get("vllm") or {}
    return {
        "model_size": model_size,
        "hf_repo": m["hf_repo"],
        "local_dir": m["local_dir"],
        "served_model_name": m["served_model_name"],
        "torch_dtype": m["torch_dtype"],
        "max_model_len": vllm.get("max_model_len", 8192),
        "gpu_memory_utilization": vllm.get("gpu_memory_utilization", 0.95),
        "tensor_parallel_size": vllm.get("tensor_parallel_size", 1),
    }


def resolve_model_config(
    model_size: str,
    quant: str,
    config: Optional[dict[str, Any]] = None,
    engine: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full serve-ready config from config.yaml + auto or explicit model_size/quant/engine.
    4-bit quantization always uses the tensorrt_llm engine (vLLM path does not expose it here).
    """
    cfg = config if config is not None else load_config()
    eng = (engine or default_engine(cfg)).lower().strip()
    if eng not in VALID_ENGINES:
        raise ValueError(f"Invalid engine {eng!r}. Use: {sorted(VALID_ENGINES)}")

    vram_gb = get_gpu_vram_gb()
    picked = auto_select(vram_gb, model_size=model_size, quant=quant, engine=eng)
    if picked["quant"] == "4bit" and eng == "vllm":
        eng = "tensorrt_llm"

    base = resolve_paths(cfg, picked["model_size"])
    cap = min(base["max_model_len"], picked["max_model_len"])
    base["max_model_len"] = cap
    base["quantization"] = picked["quant"]
    base["vram_gb"] = picked["vram_gb"]
    base["gpu_name"] = picked["gpu_name"]
    base["engine"] = eng
    base["trt_venv_python"] = trt_venv_python(cfg)
    return base


def build_cli_json(
    cfg: dict[str, Any],
    model_size: str,
    quant: str,
    engine: Optional[str] = None,
) -> dict[str, Any]:
    r = resolve_model_config(model_size, quant, cfg, engine=engine)
    out: dict[str, Any] = {
        "engine": r["engine"],
        "model_size": r["model_size"],
        "quantization": r["quantization"],
        "local_dir": r["local_dir"],
        "hf_repo": r["hf_repo"],
        "served_model_name": r["served_model_name"],
        "torch_dtype": r["torch_dtype"],
        "max_model_len": r["max_model_len"],
        "gpu_memory_utilization": r["gpu_memory_utilization"],
        "tensor_parallel_size": r["tensor_parallel_size"],
        "vram_gb": round(float(r["vram_gb"]), 2),
        "gpu_name": r["gpu_name"],
        "trt_venv_python": r["trt_venv_python"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-aware model selection (JSON to stdout)")
    parser.add_argument("--model-size", default="auto", help="auto|0.8b|2b|4b|9b|27b|35b-a3b")
    parser.add_argument("--quant", default="auto", help="vLLM: auto|none|8bit|awq  TRT: +4bit")
    parser.add_argument(
        "--engine",
        default=None,
        help="vllm|tensorrt_llm (default: config inference.engine)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    ms = args.model_size.lower().strip()
    q = args.quant.lower().strip()
    if ms != "auto" and ms not in MODEL_SIZE_KEYS:
        print(json.dumps({"error": f"invalid --model-size {ms}"}), file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    if "models" not in cfg:
        print(json.dumps({"error": "config.yaml missing models: section"}), file=sys.stderr)
        sys.exit(1)

    eng_eff = (args.engine or default_engine(cfg)).lower().strip()
    if eng_eff not in VALID_ENGINES:
        print(json.dumps({"error": f"invalid --engine {eng_eff}"}), file=sys.stderr)
        sys.exit(1)
    # 4bit is accepted on paper with default vllm; resolve_model_config switches engine to tensorrt_llm.
    allowed = QUANTS_TRT if eng_eff == "tensorrt_llm" else (QUANTS_VLLM | frozenset({"4bit"}))
    if q not in allowed:
        print(
            json.dumps(
                {
                    "error": f"invalid --quant {q} for engine {eng_eff}. Allowed: {sorted(allowed)}",
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        out = build_cli_json(cfg, ms, q, engine=args.engine)
    except (KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
