#!/usr/bin/env python3
"""
tensorrt_llm_server.py — OpenAI-compatible inference server (TensorRT-LLM PyTorch backend)

Uses the TRT-LLM PyTorch backend, which loads the HF checkpoint directly —
no engine compilation step required.  Start time is comparable to vLLM.

Usage:
  python tensorrt_llm_server.py [--port 8080] [--quant none|int8|int4_awq]

Quantization options:
  none     — bf16 full precision (~19 GB VRAM)
  int8     — SmoothQuant W8A8   (~10 GB VRAM)
  int4_awq — AWQ W4A16          (~5  GB VRAM)  ← fastest

Endpoints (OpenAI-compatible):
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (streaming + non-streaming)
"""

import argparse
import asyncio
import json
import logging
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

# ── TRT-LLM imports — PyTorch backend (TRT-LLM 1.x) ─────────────────────────
# The root-level `LLM` is the PyTorch backend in TRT-LLM 1.x.
# Do NOT use tensorrt_llm._tensorrt_engine.LLM here — that is the TensorRT
# backend which requires a separate engine-build step and does not accept
# the model dir directly.
_trtllm_import_errors: list[str] = []

try:
    from tensorrt_llm import LLM, SamplingParams
except Exception as e:
    _trtllm_import_errors.append(f"  LLM/SamplingParams: {e}")
    LLM = SamplingParams = None  # type: ignore

# QuantConfig / QuantAlgo: .quantization in both versions; fallback to root
try:
    from tensorrt_llm.quantization import QuantConfig, QuantAlgo
except Exception:
    try:
        from tensorrt_llm import QuantConfig, QuantAlgo
    except Exception as e:
        _trtllm_import_errors.append(f"  QuantConfig/QuantAlgo: {e}")
        QuantConfig = QuantAlgo = None  # type: ignore

_TRTLLM_AVAILABLE = LLM is not None

PROJECT = Path(__file__).parent
CONFIG  = PROJECT / "config.yaml"

logging.basicConfig(level=logging.INFO, format="[trtllm] %(message)s")
log = logging.getLogger("trtllm_server")

# ── Globals ───────────────────────────────────────────────────────────────────
llm:       Optional["LLM"]         = None
tokenizer: Optional[AutoTokenizer] = None
cfg:       Optional[dict]          = None


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _quant_config(quant: str):
    """Return TRT-LLM QuantConfig for the requested scheme, or None for bf16."""
    if QuantConfig is None or QuantAlgo is None:
        if quant in ("int8", "int4_awq"):
            log.warning("QuantConfig unavailable — running without quantization "
                        "(requested: %s)", quant)
        return None
    match quant:
        case "int8":
            return QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)
        case "int4_awq":
            return QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        case _:
            return None


# ── Model load ────────────────────────────────────────────────────────────────

def load_model(c: dict, quant: str) -> tuple["LLM", AutoTokenizer]:
    """Load the HF model via TRT-LLM PyTorch backend + HF tokenizer."""
    if not _TRTLLM_AVAILABLE:
        sys.exit("ERROR: tensorrt_llm is not installed.\n"
                 "       pip install tensorrt-llm")

    model_dir = Path(c["model"]["local_dir"])
    tc        = c.get("tensorrt_llm", {})
    dtype     = c["model"].get("torch_dtype", "bfloat16")
    tp_size   = tc.get("tensor_parallel_size", 1)
    quant_cfg = _quant_config(quant)

    if not model_dir.is_dir():
        sys.exit(f"ERROR: HF model not found at {model_dir}\n"
                 "       Run: python download_model.py")

    log.info("Loading model (PyTorch backend): %s", model_dir)
    log.info("  dtype : %s  quant : %s  TP : %d", dtype, quant, tp_size)

    _llm = LLM(
        model=str(model_dir),
        dtype=dtype,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        **( {"quant_config": quant_cfg} if quant_cfg else {} ),
    )
    log.info("Model loaded.")

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

    llm, tokenizer = await asyncio.get_event_loop().run_in_executor(
        None, load_model, cfg, quant
    )
    yield
    log.info("Shutting down.")
    del llm


app = FastAPI(title="Qwen3.5-9B TRT-LLM Server (PyTorch backend)")


# ── Pydantic models (OpenAI-compatible) ───────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str                        = "qwen3.5-9b"
    messages: list[ChatMessage]
    max_tokens:         Optional[int]   = None
    temperature:        Optional[float] = None
    top_p:              Optional[float] = None
    top_k:              Optional[int]   = None
    repetition_penalty: Optional[float] = None
    stream:             bool            = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_template(messages: list[ChatMessage]) -> str:
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


def _sampling_params(req: ChatCompletionRequest) -> "SamplingParams":
    gen = cfg.get("generation", {})
    return SamplingParams(
        max_tokens=         req.max_tokens         or gen.get("max_new_tokens",     2048),
        temperature=        req.temperature        or gen.get("temperature",        0.6),
        top_p=              req.top_p              or gen.get("top_p",              0.95),
        top_k=              req.top_k              or gen.get("top_k",              40),
        repetition_penalty= req.repetition_penalty or gen.get("repetition_penalty", 1.05),
    )


def _make_chunk(cid: str, content: str, finish: Optional[str] = None) -> str:
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

    async for output in llm.generate_async(prompt, streaming=True, sampling_params=sp):
        delta = output.outputs[0].text_diff
        if delta:
            yield _make_chunk(cid, delta)

    yield _make_chunk(cid, "", finish="stop")
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    c = cfg or {}
    return {
        "status":  "ok",
        "model":   "qwen3.5-9b",
        "backend": "tensorrt_llm_pytorch",
        "model_dir": c.get("model", {}).get("local_dir", "loading"),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       "qwen3.5-9b",
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "local",
            "backend":  "tensorrt_llm_pytorch",
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
    prompt  = _apply_template(req.messages)
    sp      = _sampling_params(req)

    outputs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: llm.generate([prompt], sampling_params=sp)
    )
    text             = outputs[0].outputs[0].text
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    prompt_tokens    = len(outputs[0].prompt_token_ids or [])

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "qwen3.5-9b",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
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

    parser = argparse.ArgumentParser(description="TensorRT-LLM PyTorch-backend inference server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--quant", default=None,
                        choices=["none", "int8", "int4_awq"],
                        help="Quantization (overrides config.yaml)")
    args = parser.parse_args()

    if not _TRTLLM_AVAILABLE:
        print("ERROR: tensorrt_llm classes failed to import:", file=sys.stderr)
        for e in _trtllm_import_errors:
            print(e, file=sys.stderr)
        print("\nInstall: pip install tensorrt-llm", file=sys.stderr)
        sys.exit(1)

    c     = _load_cfg()
    quant = args.quant or c["model"].get("quantization", "none")
    # Normalise: vLLM quant names → TRT-LLM names
    quant = {"8bit": "int8", "awq": "int4_awq", "4bit": "int4_awq"}.get(quant, quant)

    app.state.quant = quant
    app.router.lifespan_context = lifespan

    model_dir = c["model"]["local_dir"]
    print("=" * 54)
    print(" Qwen3.5-9B Inference Server (TensorRT-LLM PyTorch)")
    print(f" URL     : http://{args.host}:{args.port}")
    print(f" Model   : {model_dir}")
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
