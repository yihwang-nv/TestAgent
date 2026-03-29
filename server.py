#!/usr/bin/env python3
"""
server.py — launcher: vLLM or TensorRT-LLM (reads config.yaml + gpu_select).

Prefer: bash start_server.sh [--engine vllm|tensorrt_llm] ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from gpu_select import resolve_model_config

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Start inference server (vLLM or TensorRT-LLM)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model-size",
        default="auto",
        help="auto | 0.8b | 2b | 4b | 9b | 27b | 35b-a3b",
    )
    parser.add_argument(
        "--quant",
        default="auto",
        help="auto | none | 8bit | 4bit | awq (4bit → TensorRT-LLM)",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="vllm | tensorrt_llm (default: config inference.engine)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())
    vllm_cfg = cfg.get("vllm") or {}
    tool_call_parser = str(vllm_cfg.get("tool_call_parser") or "qwen3_xml").strip()
    if os.environ.get("VLLM_TOOL_CALL_PARSER", "").strip():
        tool_call_parser = os.environ["VLLM_TOOL_CALL_PARSER"].strip()
        print(f"NOTE: VLLM_TOOL_CALL_PARSER override → tool_call_parser={tool_call_parser}")
    r = resolve_model_config(
        args.model_size.lower(),
        args.quant.lower(),
        cfg,
        engine=args.engine,
    )
    eng = r["engine"]

    if eng == "tensorrt_llm":
        trt_py = Path(r["trt_venv_python"])
        if not trt_py.is_file():
            sys.exit(
                f"ERROR: TensorRT-LLM venv missing: {trt_py}\n"
                "       Run: bash setup_trtllm.sh"
            )
        os.execv(
            str(trt_py),
            [
                str(trt_py),
                str(PROJECT_DIR / "trtllm_server.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model-size",
                args.model_size.lower(),
                "--quant",
                args.quant.lower(),
            ],
        )

    model_dir = r["local_dir"]
    quant = r["quantization"]
    if quant == "4bit":
        sys.exit("ERROR: Use --engine tensorrt_llm for 4-bit quantization.")

    dtype = r["torch_dtype"]
    max_len = r["max_model_len"]
    gpu_util = r["gpu_memory_utilization"]
    tp_size = r["tensor_parallel_size"]
    model_name = r["served_model_name"]

    if not Path(model_dir).is_dir():
        sys.exit(
            f"ERROR: Model not found at {model_dir}\n"
            "       Run: python download_model.py --model-size <size|auto|all>"
        )

    cmd = [
        "vllm",
        "serve",
        model_dir,
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--tensor-parallel-size",
        str(tp_size),
        "--served-model-name",
        model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--tokenizer-mode",
        "slow",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
    ]

    if quant == "8bit":
        cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    elif quant == "awq":
        cmd += ["--quantization", "awq"]

    req_ms = args.model_size.lower()
    req_q = args.quant.lower()
    req_eng = args.engine or "(config)"
    sep = "=" * 54
    print(sep)
    print(" Qwen3.5 vLLM — 已选模型与参数")
    print("-" * 54)
    print(" 启动参数 (CLI)")
    print(f"   --engine      {req_eng}  →  {eng}")
    print(f"   --model-size  {req_ms}  →  解析为: {r['model_size']}")
    print(f"   --quant       {req_q}  →  解析为: {quant}")
    print(f"   --host        {args.host}")
    print(f"   --port        {args.port}")
    print("-" * 54)
    print(" GPU")
    print(f"   设备        : {r['gpu_name']}")
    print(f"   显存        : {r['vram_gb']:.2f} GB (nvidia-smi)")
    print("-" * 54)
    print(" 模型与 vLLM")
    print(f"   HuggingFace : {r['hf_repo']}")
    print(f"   本地目录    : {model_dir}")
    print(f"   API 名称    : {model_name}  (/v1/models)")
    print(f"   dtype       : {dtype}")
    print(f"   量化        : {quant}")
    print(f"   max_len     : {max_len}")
    print(f"   gpu_mem_util: {gpu_util}")
    print(f"   tensor_para : {tp_size}")
    print(f"   tool_parser : {tool_call_parser}")
    print(sep)
    print(f"Launching: {' '.join(cmd)}")
    os.execvp("vllm", cmd)


if __name__ == "__main__":
    main()
