#!/usr/bin/env python3
import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
STEPAUDIO2_SOURCE_DIR = os.environ.get(
    "STEPAUDIO2_SOURCE_DIR",
    os.path.join(PROJECT_DIR, "third_party", "Step-Audio2"),
)
DEFAULT_TOKEN2WAV_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "Step-Audio-2-mini", "token2wav")
for _p in (PROJECT_DIR, STEPAUDIO2_SOURCE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/stepaudio_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

logging.basicConfig(
    level=getattr(logging, os.environ.get("LYCHEEFD_T2W_SERVER_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lychee_fd.token2wav_server")

app = FastAPI()
_model = None
_model_path = None
_model_lock = threading.Lock()
_stream_states: Dict[str, dict] = {}


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _json_body(request: Request) -> dict:
    try:
        body = await request.body()
        if not body:
            return {}
        obj = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return obj


def _require_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Token2wav model is not loaded")
    return _model


def _load_model(model_path: str, float16: bool = False) -> None:
    global _model, _model_path
    from token2wav import Token2wav

    t0 = time.time()
    logger.info("Loading Token2wav model_path=%s CUDA_VISIBLE_DEVICES=%s", model_path, os.environ.get("CUDA_VISIBLE_DEVICES"))
    _model = Token2wav(model_path, float16=bool(float16))
    _model_path = str(model_path)
    logger.info("Token2wav loaded in %.2fs", time.time() - t0)


def _save_stream_state(stream_id: str, prompt_wav: str) -> None:
    _stream_states[str(stream_id)] = {
        "prompt_wav": str(prompt_wav),
        "stream_cache": getattr(_model, "stream_cache", None),
        "hift_cache_dict": getattr(_model, "hift_cache_dict", {}),
        "updated_epoch_ms": _now_ms(),
    }


def _restore_stream_state(stream_id: str, prompt_wav: str) -> bool:
    state = _stream_states.get(str(stream_id))
    if not isinstance(state, dict):
        return False
    if str(state.get("prompt_wav")) != str(prompt_wav):
        return False
    _model.stream_cache = state.get("stream_cache")
    _model.hift_cache_dict = state.get("hift_cache_dict", {})
    state["updated_epoch_ms"] = _now_ms()
    return True


@app.post("/v1/token2wav/health")
async def health(_request: Request):
    return JSONResponse(
        {
            "ok": _model is not None,
            "model_path": _model_path,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "stream_count": len(_stream_states),
        }
    )


@app.post("/v1/token2wav/start")
async def start_stream(request: Request):
    payload = await _json_body(request)
    stream_id = str(payload.get("stream_id") or "")
    prompt_wav = str(payload.get("prompt_wav") or "")
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    if not prompt_wav:
        raise HTTPException(status_code=400, detail="prompt_wav is required")
    model = _require_model()

    recv_ms = _now_ms()
    t_lock_start = time.perf_counter()
    with _model_lock:
        synth_start_ms = _now_ms()
        queue_wait_ms = max(0.0, (time.perf_counter() - t_lock_start) * 1000.0)
        try:
            model.set_stream_cache(prompt_wav)
            _save_stream_state(stream_id, prompt_wav)
        except Exception as exc:
            logger.exception("start_stream failed stream_id=%s", stream_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        synth_end_ms = _now_ms()
    return JSONResponse(
        {
            "ok": True,
            "stream_id": stream_id,
            "remote_recv_epoch_ms": recv_ms,
            "synth_start_epoch_ms": synth_start_ms,
            "synth_end_epoch_ms": synth_end_ms,
            "remote_queue_wait_ms": round(queue_wait_ms, 3),
            "remote_return_epoch_ms": _now_ms(),
        }
    )


@app.post("/v1/token2wav/stream")
async def stream_chunk(request: Request):
    payload = await _json_body(request)
    stream_id = str(payload.get("stream_id") or "")
    prompt_wav = str(payload.get("prompt_wav") or "")
    tokens = payload.get("tokens") or []
    last_chunk = bool(payload.get("last_chunk", False))
    advance_tokens = int(payload.get("advance_tokens") or 0)
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    if not prompt_wav:
        raise HTTPException(status_code=400, detail="prompt_wav is required")
    if not isinstance(tokens, list):
        raise HTTPException(status_code=400, detail="tokens must be a list")
    clean_tokens = [int(x) for x in tokens]
    model = _require_model()

    recv_ms = _now_ms()
    t_lock_start = time.perf_counter()
    with _model_lock:
        synth_start_ms = _now_ms()
        queue_wait_ms = max(0.0, (time.perf_counter() - t_lock_start) * 1000.0)
        try:
            if not _restore_stream_state(stream_id, prompt_wav):
                model.set_stream_cache(prompt_wav)
            t0 = time.perf_counter()
            pcm = model.stream(clean_tokens, prompt_wav=prompt_wav, last_chunk=last_chunk)
            synth_sec = time.perf_counter() - t0
            _save_stream_state(stream_id, prompt_wav)
        except Exception as exc:
            logger.exception("stream_chunk failed stream_id=%s tokens=%d", stream_id, len(clean_tokens))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        synth_end_ms = _now_ms()

    return_ms = _now_ms()
    return JSONResponse(
        {
            "ok": True,
            "stream_id": stream_id,
            "tokens": len(clean_tokens),
            "advance_tokens": advance_tokens,
            "last_chunk": last_chunk,
            "pcm_b64": base64.b64encode(bytes(pcm)).decode("ascii"),
            "pcm_bytes": len(pcm),
            "remote_recv_epoch_ms": recv_ms,
            "synth_start_epoch_ms": synth_start_ms,
            "synth_end_epoch_ms": synth_end_ms,
            "synth_sec": float(synth_sec),
            "synth_duration_ms": round(float(synth_sec) * 1000.0, 3),
            "remote_queue_wait_ms": round(queue_wait_ms, 3),
            "remote_return_epoch_ms": return_ms,
            "remote_roundtrip_ms": max(0, return_ms - recv_ms),
        }
    )


@app.post("/v1/token2wav/close")
async def close_stream(request: Request):
    payload = await _json_body(request)
    stream_id = str(payload.get("stream_id") or "")
    if stream_id:
        _stream_states.pop(stream_id, None)
    return JSONResponse({"ok": True, "stream_id": stream_id, "stream_count": len(_stream_states)})


def main():
    parser = argparse.ArgumentParser(description="Run Token2Wav as a small HTTP service.")
    parser.add_argument("--host", default=os.environ.get("LYCHEEFD_T2W_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LYCHEEFD_T2W_SERVER_PORT", "8091")))
    parser.add_argument("--model-path", default=os.environ.get("LYCHEEFD_T2W_MODEL_PATH", DEFAULT_TOKEN2WAV_MODEL_PATH))
    parser.add_argument("--float16", action="store_true", default=os.environ.get("LYCHEEFD_T2W_FLOAT16", "0") == "1")
    args = parser.parse_args()

    _load_model(args.model_path, float16=bool(args.float16))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
