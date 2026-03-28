#!/usr/bin/env python3
"""
OpenAI-compatible inference server for Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled.
Exposes POST /v1/chat/completions with SSE streaming.
Used as the backend for the Claude Agent SDK agent in agent.py.
"""

import json
import time
import uuid
import yaml
import torch
import uvicorn
import asyncio
from pathlib import Path
from threading import Thread
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

PROJECT = Path(__file__).parent
CONFIG_PATH = PROJECT / "config.yaml"

# ── Globals ──────────────────────────────────────────────────────────────────
tokenizer = None
model      = None
cfg        = None

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, cfg

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model_dir    = PROJECT / cfg["model"]["local_dir"]
    quantization = cfg["model"].get("quantization", "4bit")
    torch_dtype  = getattr(torch, cfg["model"].get("torch_dtype", "bfloat16"))

    if not model_dir.exists():
        raise RuntimeError(f"Model not found at {model_dir}. Run: python download_model.py")

    print(f"[server] Loading model from {model_dir} ({quantization}) ...")

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # For non-quantized loads, skip device_map and move explicitly to CUDA.
    # device_map="auto" can silently fall back to CPU if accelerate's device
    # detection doesn't register the GPU before the model is dispatched.
    cuda_available = torch.cuda.is_available()
    device_target  = "cuda:0" if cuda_available else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if quantization == "none":
        # Load in bf16 then move directly to CUDA — guaranteed GPU placement
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device_target)
    else:
        # Quantized: let device_map handle placement (bitsandbytes manages this)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map=cfg["model"].get("device_map", "auto"),
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

    model.eval()
    actual_device = next(model.parameters()).device
    print(f"[server] Model ready on {actual_device}.")
    yield
    print("[server] Shutting down.")


app = FastAPI(title="Qwen3.5-9B Inference Server", lifespan=lifespan)

# ── Pydantic models (OpenAI-compatible) ───────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3.5-9b-local"
    messages: list[ChatMessage]
    max_tokens: Optional[int]    = None
    temperature: Optional[float] = None
    top_p: Optional[float]       = None
    top_k: Optional[int]         = None
    repetition_penalty: Optional[float] = None
    stream: bool = False

# ── Helpers ───────────────────────────────────────────────────────────────────
def _apply_template(messages: list[ChatMessage]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask) on the model device."""
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    encoded = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Newer transformers returns a BatchEncoding with input_ids + attention_mask.
    # Older versions return a bare tensor — build the mask manually in that case.
    if hasattr(encoded, "input_ids"):
        input_ids      = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device) \
                         if hasattr(encoded, "attention_mask") \
                         else torch.ones_like(input_ids)
    else:
        input_ids      = encoded.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _gen_kwargs(req: ChatCompletionRequest) -> dict:
    gen = cfg.get("generation", {})
    return dict(
        max_new_tokens=req.max_tokens   or gen.get("max_new_tokens", 2048),
        temperature=req.temperature     or gen.get("temperature", 0.6),
        top_p=req.top_p                 or gen.get("top_p", 0.95),
        top_k=req.top_k                 or gen.get("top_k", 40),
        repetition_penalty=req.repetition_penalty or gen.get("repetition_penalty", 1.05),
        do_sample=gen.get("do_sample", True),
        pad_token_id=tokenizer.eos_token_id,
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    device = str(next(model.parameters()).device) if model is not None else "not_loaded"
    return {"status": "ok", "model": "qwen3.5-9b-local", "device": device}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3.5-9b-local",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }


async def _stream_tokens(req: ChatCompletionRequest) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI delta format."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    input_ids, attention_mask = _apply_template(req.messages)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = _gen_kwargs(req)
    kwargs["input_ids"]      = input_ids
    kwargs["attention_mask"] = attention_mask
    kwargs["streamer"]       = streamer

    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    for token_text in streamer:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "qwen3.5-9b-local",
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # yield event loop

    # Final chunk
    final = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3.5-9b-local",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"
    thread.join()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.stream:
        return StreamingResponse(
            _stream_tokens(req),
            media_type="text/event-stream",
        )

    # Non-streaming
    input_ids, attention_mask = _apply_template(req.messages)
    kwargs = _gen_kwargs(req)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    n_input  = input_ids.shape[-1]
    new_ids  = output[0][n_input:]
    text     = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3.5-9b-local",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": n_input,
            "completion_tokens": len(new_ids),
            "total_tokens": n_input + len(new_ids),
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1,
                        help="Keep at 1 — GPU model is not fork-safe")
    args = parser.parse_args()

    print(f"[server] Starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
