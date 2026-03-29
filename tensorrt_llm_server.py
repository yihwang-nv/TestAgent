#!/usr/bin/env python3
"""
tensorrt_llm_server.py — OpenAI-compatible inference server (TensorRT-LLM backend)

TRT-LLM compiles the HF model into a GPU-specific TensorRT engine for maximum
throughput. The engine is built once and cached to disk; subsequent starts load
it in seconds.

Workflow:
  1. (First run) Build engine from HF checkpoint — takes 15–30 min:
       python tensorrt_llm_server.py --build-only
     OR let the server auto-build on first start (same time cost).

  2. Start server:
       python tensorrt_llm_server.py [--port 8080] [--quant none|int8|int4_awq]

  3. Query exactly like the vLLM server — OpenAI-compatible:
       POST /v1/chat/completions  (streaming + non-streaming)
       GET  /health
       GET  /v1/models

Engine cache location:
  config.yaml → tensorrt_llm.engine_dir
  Default: <project>/engines/qwen3.5-9b-<quant>-<dtype>/

Quantization options:
  none     — bf16 full precision (~19 GB VRAM)
  int8     — SmoothQuant W8A8   (~10 GB VRAM)
  int4_awq — AWQ W4A16          (~5  GB VRAM)  ← fastest
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# ── TensorRT-LLM imports ──────────────────────────────────────────────────────
# These are only available if tensorrt_llm is installed.
# Install: pip install tensorrt-llm  (requires CUDA ≥ 12.1, driver ≥ 525)
try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llm import BuildConfig
    from tensorrt_llm.quantization import QuantConfig, QuantAlgo
    _TRTLLM_AVAILABLE = True
except (ImportError, Exception):
    # Catch broad Exception: modelopt's vllm-plugin import failure raises a
    # non-fatal UserWarning at import time but the actual TRT-LLM classes may
    # still be available. If the specific classes above fail, mark unavailable.
    _TRTLLM_AVAILABLE = False

PROJECT    = Path(__file__).parent
CONFIG     = PROJECT / "config.yaml"

logging.basicConfig(level=logging.INFO, format="[trtllm] %(message)s")
log = logging.getLogger("trtllm_server")

# ── Globals ───────────────────────────────────────────────────────────────────
llm:       Optional["LLM"]         = None
tokenizer: Optional[AutoTokenizer] = None
cfg:       Optional[dict]          = None


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _engine_dir(c: dict, quant: str) -> Path:
    """Return the engine directory for the given quantization setting."""
    base = c.get("tensorrt_llm", {}).get("engine_dir")
    if base:
        return Path(base)
    dtype = c["model"].get("torch_dtype", "bfloat16")
    return PROJECT / "engines" / f"qwen3.5-9b-{quant}-{dtype}"


def _quant_config(quant: str) -> Optional["QuantConfig"]:
    """Return TRT-LLM QuantConfig for the requested scheme, or None for bf16."""
    if not _TRTLLM_AVAILABLE:
        return None
    match quant:
        case "int8":
            # SmoothQuant — calibrated W8A8, ~10 GB VRAM
            return QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)
        case "int4_awq":
            # Activation-Aware Weight Quantization W4A16, ~5 GB VRAM
            return QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        case "none" | _:
            return None


# ── Engine build ──────────────────────────────────────────────────────────────

def build_engine(c: dict, quant: str) -> Path:
    """
    Build (or load) a TensorRT-LLM engine from the HF checkpoint.

    If an engine already exists at engine_dir, skip the build and return
    the cached path immediately.

    Returns the engine directory path.
    """
    if not _TRTLLM_AVAILABLE:
        sys.exit("ERROR: tensorrt_llm is not installed.\n"
                 "       pip install tensorrt-llm")

    model_dir  = Path(c["model"]["local_dir"])
    engine_dir = _engine_dir(c, quant)
    tc         = c.get("tensorrt_llm", {})
    dtype      = c["model"].get("torch_dtype", "bfloat16")
    max_seq    = tc.get("max_seq_len", 8192)
    max_batch  = tc.get("max_batch_size", 8)
    tp_size    = tc.get("tensor_parallel_size", 1)

    if not model_dir.is_dir():
        sys.exit(f"ERROR: HF model not found at {model_dir}\n"
                 "       Run: python download_model.py")

    if engine_dir.exists():
        log.info("Engine cache found: %s — skipping build.", engine_dir)
        return engine_dir

    log.info("Building TRT-LLM engine …")
    log.info("  Source   : %s", model_dir)
    log.info("  Engine   : %s", engine_dir)
    log.info("  dtype    : %s", dtype)
    log.info("  quant    : %s", quant)
    log.info("  seq_len  : %d", max_seq)
    log.info("  batch    : %d", max_batch)
    log.info("  TP       : %d", tp_size)
    log.info("This will take 15–30 minutes on first run.")

    build_cfg  = BuildConfig(max_seq_len=max_seq, max_batch_size=max_batch)
    quant_cfg  = _quant_config(quant)

    engine_dir.mkdir(parents=True, exist_ok=True)

    try:
        tmp_llm = LLM(
            model=str(model_dir),
            dtype=dtype,
            tensor_parallel_size=tp_size,
            build_config=build_cfg,
            **({"quant_config": quant_cfg} if quant_cfg else {}),
        )
        tmp_llm.save(str(engine_dir))
        del tmp_llm

    except Exception:
        # Clean up partial engine dir so next run retries the build
        shutil.rmtree(engine_dir, ignore_errors=True)
        raise

    log.info("Engine built and saved to %s", engine_dir)
    return engine_dir


# ── Model load ────────────────────────────────────────────────────────────────

def load_model(c: dict, quant: str) -> tuple["LLM", AutoTokenizer]:
    """Load a pre-built TRT-LLM engine and the HF tokenizer."""
    if not _TRTLLM_AVAILABLE:
        sys.exit("ERROR: tensorrt_llm is not installed.\n"
                 "       pip install tensorrt-llm")

    model_dir  = Path(c["model"]["local_dir"])
    engine_dir = _engine_dir(c, quant)
    tc         = c.get("tensorrt_llm", {})
    tp_size    = tc.get("tensor_parallel_size", 1)

    log.info("Loading engine from %s …", engine_dir)
    _llm = LLM(
        model=str(engine_dir),
        tensor_parallel_size=tp_size,
    )
    log.info("Engine loaded.")

    log.info("Loading tokenizer from %s …", model_dir)
    _tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    log.info("Tokenizer loaded.")

    return _llm, _tok


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, tokenizer, cfg
    cfg   = _load_cfg()
    quant = app.state.quant

    engine_dir = _engine_dir(cfg, quant)
    if not engine_dir.exists():
        log.info("No cached engine found — building now (15–30 min)…")
        await asyncio.get_event_loop().run_in_executor(
            None, build_engine, cfg, quant
        )

    llm, tokenizer = await asyncio.get_event_loop().run_in_executor(
        None, load_model, cfg, quant
    )
    yield
    log.info("Shutting down.")
    del llm


app = FastAPI(title="Qwen3.5-9B TRT-LLM Server")


# ── Pydantic models (OpenAI-compatible) ───────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str                       = "qwen3.5-9b"
    messages: list[ChatMessage]
    max_tokens:        Optional[int]   = None
    temperature:       Optional[float] = None
    top_p:             Optional[float] = None
    top_k:             Optional[int]   = None
    repetition_penalty: Optional[float] = None
    stream:            bool            = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_template(messages: list[ChatMessage]) -> str:
    """Apply Qwen3 ChatML template; return a plain text prompt string."""
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


def _sampling_params(req: ChatCompletionRequest) -> "SamplingParams":
    gen = cfg.get("generation", {})
    return SamplingParams(
        max_tokens=        req.max_tokens          or gen.get("max_new_tokens",     2048),
        temperature=       req.temperature         or gen.get("temperature",        0.6),
        top_p=             req.top_p               or gen.get("top_p",              0.95),
        top_k=             req.top_k               or gen.get("top_k",              40),
        repetition_penalty=req.repetition_penalty  or gen.get("repetition_penalty", 1.05),
    )


def _make_chunk(cid: str, content: str, finish: Optional[str] = None) -> str:
    """Format a single SSE delta chunk."""
    return "data: " + json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish,
        }],
    }) + "\n\n"


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream_tokens(req: ChatCompletionRequest) -> AsyncIterator[str]:
    cid    = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt = _apply_template(req.messages)
    sp     = _sampling_params(req)

    # TRT-LLM HLAPI: generate_async yields RequestOutput objects.
    # output.outputs[0].text_diff is the incremental token text.
    async for output in llm.generate_async(prompt, streaming=True, sampling_params=sp):
        delta = output.outputs[0].text_diff
        if delta:
            yield _make_chunk(cid, delta)

    yield _make_chunk(cid, "", finish="stop")
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "qwen3.5-9b",
        "backend": "tensorrt_llm",
        "engine": str(_engine_dir(cfg, app.state.quant)) if cfg else "loading",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3.5-9b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "backend": "tensorrt_llm",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.stream:
        return StreamingResponse(
            _stream_tokens(req),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full output
    prompt = _apply_template(req.messages)
    sp     = _sampling_params(req)

    outputs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: llm.generate([prompt], sampling_params=sp)
    )
    text          = outputs[0].outputs[0].text
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    prompt_tokens = len(outputs[0].prompt_token_ids or [])

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="TensorRT-LLM inference server")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--quant",      default=None,
                        choices=["none", "int8", "int4_awq"],
                        help="Quantization (overrides config.yaml)")
    parser.add_argument("--build-only", action="store_true",
                        help="Build engine and exit without starting server")
    args = parser.parse_args()

    if not _TRTLLM_AVAILABLE:
        sys.exit(
            "ERROR: tensorrt_llm is not installed.\n"
            "       pip install tensorrt-llm\n"
            "       (requires CUDA ≥ 12.1, driver ≥ 525, sm_80+)"
        )

    c     = _load_cfg()
    quant = args.quant or c["model"].get("quantization", "none")
    # Normalise: vLLM quant names → TRT-LLM names
    quant = {"8bit": "int8", "awq": "int4_awq", "4bit": "int4_awq"}.get(quant, quant)

    if args.build_only:
        build_engine(c, quant)
        return

    engine_dir = _engine_dir(c, quant)
    if not engine_dir.exists():
        print(f"[trtllm] No engine at {engine_dir}.")
        print("[trtllm] Building now — this takes 15–30 min on first run.")
        build_engine(c, quant)

    app.state.quant = quant
    app.router.lifespan_context = lifespan

    print("=" * 54)
    print(" Qwen3.5-9B Inference Server (TensorRT-LLM)")
    print(f" URL     : http://{args.host}:{args.port}")
    print(f" Engine  : {_engine_dir(c, quant)}")
    print(f" Quant   : {quant}")
    print("=" * 54)
    print(" Endpoints:")
    print("   GET  /health")
    print("   GET  /v1/models")
    print("   POST /v1/chat/completions  (streaming + non-streaming)")
    print()
    print(" Press Ctrl-C to stop.")
    print("=" * 54)
    print()

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
