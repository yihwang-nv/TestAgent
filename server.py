#!/usr/bin/env python3
"""
server.py — Qwen3.5-9B inference server (vLLM backend)

This is a thin launcher that reads config.yaml and exec's `vllm serve`.
It exists for compatibility with tools that expect `python server.py`.

For direct use, prefer:  bash start_server.sh [--port 8080] [--quant 8bit]

Fallback (transformers-based, no vLLM):  python server_transformers_fallback.py
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--quant", default=None,
                        help="Override quantization: none|8bit|awq")
    args = parser.parse_args()

    cfg        = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())
    model_dir  = cfg["model"]["local_dir"]
    quant      = args.quant or cfg["model"].get("quantization", "none")
    dtype      = cfg["model"].get("torch_dtype", "bfloat16")
    vcfg       = cfg.get("vllm", {})
    max_len    = vcfg.get("max_model_len", 8192)
    gpu_util   = vcfg.get("gpu_memory_utilization", 0.90)
    tp_size    = vcfg.get("tensor_parallel_size", 1)
    model_name = vcfg.get("served_model_name", "qwen3.5-9b")

    if not Path(model_dir).is_dir():
        sys.exit(f"ERROR: Model not found at {model_dir}\n"
                 "       Run: python download_model.py")

    cmd = [
        "vllm", "serve", model_dir,
        "--dtype",                    dtype,
        "--max-model-len",            str(max_len),
        "--gpu-memory-utilization",   str(gpu_util),
        "--tensor-parallel-size",     str(tp_size),
        "--served-model-name",        model_name,
        "--host",                     args.host,
        "--port",                     str(args.port),
        "--trust-remote-code",
    ]

    if quant == "8bit":
        cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    elif quant == "awq":
        cmd += ["--quantization", "awq"]

    print(f"Launching: {' '.join(cmd)}")
    os.execvp("vllm", cmd)  # replace this process with vllm serve


if __name__ == "__main__":
    main()
