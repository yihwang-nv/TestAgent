#!/usr/bin/env python3
"""
build_trtllm_engine.py — Pre-compile TensorRT-LLM engine from HF checkpoint.

Run this ONCE before starting tensorrt_llm_server.py.
Builds a GPU-specific engine and caches it to disk.

Usage:
  python build_trtllm_engine.py                     # uses config.yaml defaults
  python build_trtllm_engine.py --quant none        # bf16 full precision
  python build_trtllm_engine.py --quant int8        # SmoothQuant W8A8
  python build_trtllm_engine.py --quant int4_awq    # AWQ W4A16 (fastest)
  python build_trtllm_engine.py --force             # rebuild even if cached

Engine output: config.yaml → tensorrt_llm.engine_dir
Default path:  <project>/engines/qwen3.5-9b-<quant>-<dtype>/

Build time (first run): 15–30 min depending on GPU.
Subsequent server starts load the cached engine in seconds.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

try:
    from tensorrt_llm import LLM
    from tensorrt_llm.llm import BuildConfig
    from tensorrt_llm.quantization import QuantConfig, QuantAlgo
except ImportError:
    sys.exit(
        "ERROR: tensorrt_llm is not installed.\n"
        "       pip install tensorrt-llm\n"
        "       (requires CUDA ≥ 12.1, driver ≥ 525, Ampere/Hopper/Blackwell GPU)"
    )

PROJECT = Path(__file__).parent
CONFIG  = PROJECT / "config.yaml"


def _engine_dir(c: dict, quant: str) -> Path:
    base = c.get("tensorrt_llm", {}).get("engine_dir")
    if base:
        return Path(base)
    dtype = c["model"].get("torch_dtype", "bfloat16")
    return PROJECT / "engines" / f"qwen3.5-9b-{quant}-{dtype}"


def _quant_config(quant: str) -> QuantConfig | None:
    match quant:
        case "int8":
            return QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)
        case "int4_awq":
            return QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        case _:
            return None


def main():
    parser = argparse.ArgumentParser(description="Build TRT-LLM engine")
    parser.add_argument("--quant", default=None,
                        choices=["none", "int8", "int4_awq"],
                        help="Quantization scheme (overrides config.yaml)")
    parser.add_argument("--force", action="store_true",
                        help="Delete cached engine and rebuild from scratch")
    args = parser.parse_args()

    c         = yaml.safe_load(CONFIG.read_text())
    model_dir = Path(c["model"]["local_dir"])
    quant     = args.quant or c["model"].get("quantization", "none")
    # Normalise vLLM quant names
    quant     = {"8bit": "int8", "awq": "int4_awq", "4bit": "int4_awq"}.get(quant, quant)

    tc        = c.get("tensorrt_llm", {})
    dtype     = c["model"].get("torch_dtype", "bfloat16")
    max_seq   = tc.get("max_seq_len", 8192)
    max_batch = tc.get("max_batch_size", 8)
    tp_size   = tc.get("tensor_parallel_size", 1)

    engine_dir = _engine_dir(c, quant)

    if not model_dir.is_dir():
        sys.exit(f"ERROR: HF model not found at {model_dir}\n"
                 "       Run: python download_model.py")

    if engine_dir.exists():
        if args.force:
            print(f"[build] --force: removing existing engine at {engine_dir}")
            shutil.rmtree(engine_dir)
        else:
            print(f"[build] Engine already exists at {engine_dir}")
            print("[build] Use --force to rebuild. Exiting.")
            return

    print("=" * 56)
    print(" TensorRT-LLM Engine Builder")
    print(f"  Model   : {model_dir}")
    print(f"  Engine  : {engine_dir}")
    print(f"  dtype   : {dtype}")
    print(f"  quant   : {quant}")
    print(f"  seq_len : {max_seq}")
    print(f"  batch   : {max_batch}")
    print(f"  TP      : {tp_size}")
    print("=" * 56)
    print()

    t0 = time.time()
    engine_dir.mkdir(parents=True, exist_ok=True)

    build_cfg = BuildConfig(max_seq_len=max_seq, max_batch_size=max_batch)
    quant_cfg = _quant_config(quant)

    try:
        print("[build] Building engine (this may take 15–30 min)…")
        llm = LLM(
            model=str(model_dir),
            dtype=dtype,
            tensor_parallel_size=tp_size,
            build_config=build_cfg,
            **({"quant_config": quant_cfg} if quant_cfg else {}),
        )
        print(f"[build] Saving engine to {engine_dir} …")
        llm.save(str(engine_dir))
        del llm

    except Exception:
        shutil.rmtree(engine_dir, ignore_errors=True)
        raise

    elapsed = time.time() - t0
    print()
    print(f"[build] Done in {elapsed/60:.1f} min.")
    print(f"[build] Engine saved to: {engine_dir}")
    print()
    print(" Start server:")
    print("   python tensorrt_llm_server.py --port 8080")
    print(f"   python tensorrt_llm_server.py --port 8080 --quant {quant}")


if __name__ == "__main__":
    main()
