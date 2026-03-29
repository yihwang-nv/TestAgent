#!/usr/bin/env python3
"""
Launch TensorRT-LLM OpenAI-compatible server (trtllm-serve).
Run with the TensorRT-LLM venv: .venv_trtllm/bin/python trtllm_server.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from gpu_select import load_config, resolve_model_config

PROJECT_DIR = Path(__file__).resolve().parent


def _transformers_can_load_local_config(model_dir: Path) -> tuple[bool, str]:
    """Same check trtllm-serve uses early; fails on e.g. qwen3_5 + transformers 4.57."""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        return False, msg if len(msg) <= 900 else msg[:900] + "..."


def _venv_ml_versions_line() -> str:
    """For error messages; transformers import is enough to explain qwen3_5 pins."""
    try:
        import transformers

        line = f"transformers {transformers.__version__}"
    except Exception:
        return "transformers ?"
    try:
        import tensorrt_llm as trt

        line += f", tensorrt_llm {getattr(trt, '__version__', '?')}"
    except Exception:
        pass
    return line


def _trtllm_serve_executable(cfg: dict) -> str:
    """
    Path to trtllm-serve. Do not use sys.executable.resolve().parent — that often points at the
    base Python install, which has no trtllm-serve; scripts live in this repo's .venv_trtllm/bin.
    """
    trt = cfg.get("tensorrt_llm") or {}
    override = trt.get("trtllm_serve_path")
    if override:
        p = Path(str(override)).expanduser()
        if p.is_file():
            return str(p)
    sub = trt.get("venv_subdir", ".venv_trtllm")
    bindir = PROJECT_DIR / sub / "bin"
    for name in ("trtllm-serve", "trtllm-serve.exe"):
        p = bindir / name
        if p.is_file():
            return str(p)
    w = shutil.which("trtllm-serve")
    if w:
        return w
    return ""


def _build_cmd(
    cfg: dict,
    r: dict,
    host: str,
    port: int,
) -> list[str]:
    trt = cfg.get("tensorrt_llm") or {}
    quant = r["quantization"]
    model_path = r["local_dir"]
    tp = int(r.get("tensor_parallel_size", 1))

    serve = _trtllm_serve_executable(cfg)
    if not serve:
        sub = trt.get("venv_subdir", ".venv_trtllm")
        sys.exit(
            f"ERROR: trtllm-serve not found under {PROJECT_DIR / sub / 'bin'}\n"
            "       Run: bash setup_trtllm.sh"
        )

    cmd: list[str] = [serve, "serve", model_path, "--host", host, "--port", str(port)]

    if trt.get("backend"):
        cmd += ["--backend", str(trt["backend"])]

    cmd += ["--tp_size", str(tp)]

    if trt.get("pp_size"):
        cmd += ["--pp_size", str(int(trt["pp_size"]))]

    mbs = trt.get("max_batch_size")
    if mbs is not None:
        cmd += ["--max_batch_size", str(int(mbs))]

    mnt = trt.get("max_num_tokens")
    if mnt is None:
        mnt = int(r.get("max_model_len", 8192))
    cmd += ["--max_num_tokens", str(int(mnt))]

    frac = trt.get("kv_cache_free_gpu_memory_fraction")
    if frac is not None:
        cmd += ["--kv_cache_free_gpu_memory_fraction", str(float(frac))]

    sm = r.get("served_model_name") or trt.get("served_model_name")
    if sm and trt.get("pass_served_model_name", False):
        flag = str(trt.get("served_model_name_flag", "--served_model_name"))
        cmd += [flag, str(sm)]

    elmo = trt.get("extra_llm_api_options")
    if elmo:
        p = Path(str(elmo).strip()).expanduser()
        if not p.is_absolute():
            p = PROJECT_DIR / p
        if p.is_file():
            cmd += ["--extra_llm_api_options", str(p.resolve())]
        else:
            print(
                f"WARNING: tensorrt_llm.extra_llm_api_options not found: {p}",
                file=sys.stderr,
            )

    def _extend_args(key: str) -> None:
        extra = trt.get(key)
        if not extra:
            return
        if isinstance(extra, list):
            cmd.extend(str(x) for x in extra)
        elif isinstance(extra, str):
            cmd.append(extra)

    if quant == "4bit":
        q4 = trt.get("quant_4bit_args") or []
        if not q4:
            print(
                "NOTE: quant=4bit selected but quant_4bit_args is empty. "
                "TensorRT-LLM 1.2 `trtllm-serve serve` has no --quantization; "
                "PyTorch backend typically loads HF weights in bf16 unless you add "
                "ModelOpt/engine steps or extra_llm_api_options (see TRT-LLM docs).",
                file=sys.stderr,
            )
        _extend_args("quant_4bit_args")
    elif quant == "8bit":
        _extend_args("quant_8bit_args")

    _extend_args("extra_args")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT-LLM trtllm-serve launcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-size", default="auto")
    parser.add_argument("--quant", default="auto")
    args = parser.parse_args()

    cfg_path = PROJECT_DIR / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    r = resolve_model_config(
        args.model_size.lower(),
        args.quant.lower(),
        cfg,
        engine="tensorrt_llm",
    )

    if r["quantization"] not in ("none", "4bit", "8bit", "awq"):
        sys.exit(f"Unsupported TRT quantization: {r['quantization']!r}")

    model_dir = Path(r["local_dir"])
    if not model_dir.is_dir():
        sys.exit(
            f"ERROR: Model not found at {model_dir}\n"
            "       Run: python download_model.py --model-size <size|auto>"
        )

    hf_cfg = model_dir / "config.json"
    if hf_cfg.is_file():
        try:
            meta = json.loads(hf_cfg.read_text())
        except json.JSONDecodeError:
            meta = {}
        if meta.get("model_type") == "qwen3_5":
            ok, err = _transformers_can_load_local_config(model_dir)
            if not ok:
                sys.exit(
                    "ERROR: 当前 TRT venv 无法用 transformers 加载该 checkpoint（model_type qwen3_5）。\n"
                    "官方 PyPI 上的 tensorrt-llm（含 1.3.0rc9）在依赖里仍固定 transformers==4.57.3，\n"
                    "该版本未注册 qwen3_5，因此换 1.3 rc 也解决不了依赖 qwen3_5 的 Qwen3.5 系权重。\n"
                    f"当前解释器: {sys.executable}\n"
                    f"已加载: {_venv_ml_versions_line()}\n"
                    f"探测: {err}\n"
                    "可行方案:\n"
                    "  (1) 使用 vLLM: bash start_server.sh --engine vllm\n"
                    "  (2) 等待 NVIDIA 发布解除该 transformers 钉死的版本；勿自行 pip -U transformers，易破坏 tensorrt_llm。\n"
                )

    cmd = _build_cmd(cfg, r, args.host, args.port)
    sep = "=" * 54
    print(sep)
    print(" TensorRT-LLM — 已选模型与参数")
    print("-" * 54)
    print(f"   model_size  : {r['model_size']}")
    print(f"   quantization: {r['quantization']}")
    print(f"   HuggingFace : {r['hf_repo']}")
    print(f"   本地目录    : {r['local_dir']}")
    print(f"   GPU         : {r['gpu_name']}  (~{r['vram_gb']:.2f} GB)")
    print(f"   max_tokens  : {r['max_model_len']} (see tensorrt_llm.max_num_tokens)")
    print(sep)
    print(f"Launching: {' '.join(cmd)}")
    print(sep)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
