"""Lychee-FD full-duplex realtime service and HTTP API.

This is the single realtime backend entrypoint. It loads the Lychee-FD
full-duplex model, manages online sessions, accepts audio chunks, emits
Server-Sent Events, and bridges to local or remote Token2Wav synthesis.
"""

import argparse
import base64
import datetime
import hashlib
import io
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)

# Add runtime dependency paths before lazy imports.
STEPAUDIO2_SOURCE_DIR = os.environ.get(
    "STEPAUDIO2_SOURCE_DIR",
    os.path.join(PROJECT_DIR, "third_party", "Step-Audio2"),
)
sys.path.append(STEPAUDIO2_SOURCE_DIR)

# ==================== Logging ====================
_log_level_name = os.environ.get("LYCHEEFD_LOG_LEVEL", "INFO").strip().upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("realtime_demo")
logger.info("Using Step-Audio2 source dir: %s", STEPAUDIO2_SOURCE_DIR)


def _import_token2wav_class():
    """Import Token2Wav lazily so API-only startup does not load the vocoder early."""
    from token2wav import Token2wav

    return Token2wav


def _import_vllm_generation_framework():
    """Import the vLLM realtime generation framework lazily."""
    from lychee_fd.runtime.vllm_generation import VLLMGenerationFramework

    return VLLMGenerationFramework


def _import_hf_v9_generation_framework():
    """Import the V9 HuggingFace realtime generation framework lazily."""
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework

    return HFRealtimeV9GenerationFramework


def _import_streaming_decoder():
    """Import the shared StreamingDecoder lazily."""
    from lychee_fd.runtime.vllm_generation import StreamingDecoder

    return StreamingDecoder

# ==================== Global Configuration ====================
DEFAULT_MODEL_PATH = os.environ.get("LYCHEEFD_MODEL_PATH", "")
DEFAULT_TOKEN2WAV_PATH = os.environ.get(
    "LYCHEEFD_TOKEN2WAV_PATH",
    os.path.join(PROJECT_DIR, "models", "Step-Audio-2-mini", "token2wav"),
)
ASSETS_DIR = os.environ.get("LYCHEEFD_ASSETS_DIR", os.path.join(PROJECT_DIR, "assets"))

# Clone prompt voice directory containing WAV files and matching prompt text.
CLONE_PROMPT_DIR = os.environ.get(
    "LYCHEEFD_CLONE_PROMPT_DIR",
    os.path.join(PROJECT_DIR, "frontend", "public", "clone_24k_mono"),
)

REALTIME_PROMPT_VOICE_OPTIONS = [
    {"id": "default_female", "label": "默认女声", "wav": "default_female.wav", "text": "text.txt"},
    {"id": "default_male", "label": "默认男声", "wav": "default_male.wav", "text": "text.txt"},
    {"id": "leijun", "label": "雷军", "wav": "leijun_voice.wav", "text": "text.txt"},
    {"id": "guodegang", "label": "郭德纲", "wav": "gudegang_voice.wav", "text": "guodegang.txt"},
    {"id": "jay", "label": "周杰伦", "wav": "jay.wav", "text": "jay.txt"},
    {"id": "huge", "label": "胡歌", "wav": "huge.wav", "text": "huge.txt"},
    {"id": "hanhong", "label": "韩红", "wav": "hanhong.wav", "text": "hanhong.txt"},
    {"id": "nailong", "label": "奶龙", "wav": "nailong.wav", "text": "nailong.txt"},
    {"id": "kenan", "label": "柯南", "wav": "kenan.wav", "text": "kenan.txt"},
    {"id": "haimian", "label": "海绵宝宝", "wav": "haimian.wav", "text": "haimian.txt"},
    {"id": "dengziqi", "label": "邓紫棋", "wav": "dengziqi.wav", "text": "dengziqi.txt"},
    {"id": "liyunlong", "label": "李云龙", "wav": "liyunlong.wav", "text": "liyunlong.txt"},
    {"id": "new_female", "label": "清纯女声", "wav": "new_female_voice.wav", "text": "text.txt"},
    {"id": "female", "label": "阳光女声", "wav": "female_voice.wav", "text": "text.txt"},
    {"id": "news_male", "label": "播音男声", "wav": "news_male_voice.wav", "text": "text.txt"},
    {"id": "user_voice", "label": "用户音色", "wav": "user_voice.wav", "text": "user_voice.txt"},
]
DEFAULT_REALTIME_PROMPT_VOICE = os.environ.get("LYCHEEFD_DEFAULT_PROMPT_VOICE", "default_female")

ALLOWING_BACKCHANNEL = False

# Temporary audio file directory.
TEMP_DIR = "/tmp/lychee_fd_realtime_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
REALTIME_ALIGNED_SAVE_DIR = os.getenv(
    "LYCHEEFD_REALTIME_ALIGNED_SAVE_DIR",
    os.path.join(PROJECT_DIR, "realtime_aligned_audio"),
)
os.makedirs(REALTIME_ALIGNED_SAVE_DIR, exist_ok=True)

# Token2Wav streaming parameters.
TTS_CHUNK_SIZE = 25
try:
    # Downstream token2wav hop size.
    TTS_VOCODER_HOP_SIZE = max(
        1, int(os.environ.get("LYCHEEFD_TTS_VOCODER_HOP_SIZE", "25")))
except ValueError:
    TTS_VOCODER_HOP_SIZE = 25
TTS_SAMPLE_RATE = 24000
INPUT_SAMPLE_RATE = 16000
TOKENS_PER_SECOND = 25
try:
    REALTIME_TTS_CHUNK_SIZE_DEFAULT = max(
        1, int(os.environ.get("LYCHEEFD_REALTIME_TTS_CHUNK_SIZE", "10")))
except ValueError:
    REALTIME_TTS_CHUNK_SIZE_DEFAULT = 10

# Process-wide model objects.
generator = None
token2wav_model = None

# Optional external token2wav service. Keep disabled by default so the legacy
# single-process path remains the fallback behavior.
REMOTE_TOKEN2WAV_ENABLED = str(
    os.environ.get("LYCHEEFD_T2W_REMOTE_ENABLED", "0")
).strip().lower() in {"1", "true", "yes", "on"}
REMOTE_TOKEN2WAV_URL = os.environ.get(
    "LYCHEEFD_T2W_REMOTE_URL", "http://127.0.0.1:8091"
).strip().rstrip("/")
REMOTE_TOKEN2WAV_FALLBACK = str(
    os.environ.get("LYCHEEFD_T2W_REMOTE_FALLBACK", "1")
).strip().lower() in {"1", "true", "yes", "on"}
CONTROL_EARLY_EXIT_ENABLED = str(
    os.environ.get("LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "on"}
CONTROL_EARLY_STATE_SSE = str(
    os.environ.get("LYCHEEFD_CONTROL_EARLY_STATE_SSE", "1")
).strip().lower() in {"1", "true", "yes", "on"}
CONTROL_EARLY_TTS_ABORT = str(
    os.environ.get("LYCHEEFD_CONTROL_EARLY_TTS_ABORT", "1")
).strip().lower() in {"1", "true", "yes", "on"}
CONTROL_EARLY_DEBUG = str(
    os.environ.get("LYCHEEFD_CONTROL_EARLY_DEBUG", "0")
).strip().lower() in {"1", "true", "yes", "on"}
try:
    REMOTE_TOKEN2WAV_TIMEOUT_SEC = max(
        0.1, float(os.environ.get("LYCHEEFD_T2W_REMOTE_TIMEOUT_SEC", "10.0"))
    )
except ValueError:
    REMOTE_TOKEN2WAV_TIMEOUT_SEC = 10.0
try:
    REMOTE_TOKEN2WAV_PRE_LOOKAHEAD_LEN = max(
        0,
        int(
            os.environ.get(
                "LYCHEEFD_T2W_REMOTE_PRE_LOOKAHEAD_LEN",
                os.environ.get("LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN", "3"),
            )
        ),
    )
except ValueError:
    REMOTE_TOKEN2WAV_PRE_LOOKAHEAD_LEN = 3

# Realtime session configuration.
REALTIME_INFER_WINDOW_MS = int(os.environ.get("LYCHEEFD_REALTIME_INFER_WINDOW_MS", "400"))
REALTIME_INFER_WINDOW_MIN_MS = int(os.environ.get("LYCHEEFD_REALTIME_INFER_WINDOW_MIN_MS", "160"))
REALTIME_SESSION_POLL_SEC = 0.05
REALTIME_SESSION_MAX_AUDIO_HASHES = 512
try:
    REALTIME_AUDIO_EVENT_QUEUE_MAX = max(
        64, int(os.environ.get("LYCHEEFD_REALTIME_AUDIO_EVENT_QUEUE_MAX", "4096"))
    )
except ValueError:
    REALTIME_AUDIO_EVENT_QUEUE_MAX = 1024
try:
    REALTIME_CONTROL_EVENT_QUEUE_MAX = max(
        64, int(os.environ.get("LYCHEEFD_REALTIME_CONTROL_EVENT_QUEUE_MAX", "4096"))
    )
except ValueError:
    REALTIME_CONTROL_EVENT_QUEUE_MAX = 1024
REALTIME_UNIFIED_PROTOCOL_VERSION = "realtime_unified_v1"
REALTIME_UNIFIED_TEXT_SNAPSHOT_ON_AUDIO = str(
    os.environ.get("LYCHEEFD_REALTIME_TEXT_SNAPSHOT_ON_AUDIO", "1")
).strip().lower() in {"1", "true", "yes", "on"}
REALTIME_INCREMENTAL_BACKEND = str(
    os.environ.get("LYCHEEFD_REALTIME_INCREMENTAL_BACKEND", "auto")
).strip().lower()
if REALTIME_INCREMENTAL_BACKEND not in {"auto", "hf"}:
    logger.warning(
        "Invalid LYCHEEFD_REALTIME_INCREMENTAL_BACKEND=%s, fallback to auto",
        REALTIME_INCREMENTAL_BACKEND,
    )
    REALTIME_INCREMENTAL_BACKEND = "auto"

REALTIME_STRICT_INFER_WINDOW = str(
    os.environ.get("LYCHEEFD_REALTIME_STRICT_INFER_WINDOW", "0")
).strip().lower() in {"1", "true", "yes", "on"}
INPUT_SILENCE_GATE_ENABLED = str(
    os.environ.get("LYCHEEFD_INPUT_SILENCE_GATE", "0")
).strip().lower() in {"1", "true", "yes", "on"}
try:
    INPUT_SILENCE_GATE_FRAME_MS = max(
        5, int(os.environ.get("LYCHEEFD_INPUT_GATE_FRAME_MS", "20"))
    )
except ValueError:
    INPUT_SILENCE_GATE_FRAME_MS = 20
try:
    INPUT_SILENCE_GATE_OPEN_DBFS = float(
        os.environ.get("LYCHEEFD_INPUT_GATE_OPEN_DBFS", "-40")
    )
except ValueError:
    INPUT_SILENCE_GATE_OPEN_DBFS = -40.0
try:
    INPUT_SILENCE_GATE_CLOSE_DBFS = float(
        os.environ.get("LYCHEEFD_INPUT_GATE_CLOSE_DBFS", "-46")
    )
except ValueError:
    INPUT_SILENCE_GATE_CLOSE_DBFS = -46.0
if INPUT_SILENCE_GATE_CLOSE_DBFS > INPUT_SILENCE_GATE_OPEN_DBFS:
    INPUT_SILENCE_GATE_CLOSE_DBFS = INPUT_SILENCE_GATE_OPEN_DBFS
try:
    INPUT_SILENCE_GATE_HANGOVER_MS = max(
        0, int(os.environ.get("LYCHEEFD_INPUT_GATE_HANGOVER_MS", "120"))
    )
except ValueError:
    INPUT_SILENCE_GATE_HANGOVER_MS = 120
try:
    INPUT_SILENCE_GATE_PREROLL_MS = max(
        0, int(os.environ.get("LYCHEEFD_INPUT_GATE_PREROLL_MS", "60"))
    )
except ValueError:
    INPUT_SILENCE_GATE_PREROLL_MS = 60
INPUT_SILENCE_GATE_FRAME_SAMPLES = max(
    1, int(INPUT_SAMPLE_RATE * INPUT_SILENCE_GATE_FRAME_MS / 1000)
)
INPUT_SILENCE_GATE_HANGOVER_FRAMES = max(
    0,
    int(math.ceil(INPUT_SILENCE_GATE_HANGOVER_MS / max(1, INPUT_SILENCE_GATE_FRAME_MS))),
)
INPUT_SILENCE_GATE_PREROLL_FRAMES = max(
    0,
    int(math.ceil(INPUT_SILENCE_GATE_PREROLL_MS / max(1, INPUT_SILENCE_GATE_FRAME_MS))),
)
REALTIME_STAGE_TIMING_LOG = str(
    os.environ.get("LYCHEEFD_REALTIME_STAGE_TIMING_LOG", "0")
).strip().lower() in {"1", "true", "yes", "on"}
REALTIME_STAGE_TIMING_LOG_DIR = os.environ.get(
    "LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR",
    os.path.join(PROJECT_DIR, "runtime_logs", "realtime_stage_timing"),
)
REALTIME_CONTROL_PROB_TRACE_LOG = str(
    os.environ.get("LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG", "0")
).strip().lower() in {"1", "true", "yes", "on"}
REALTIME_CONTROL_PROB_TRACE_LOG_DIR = os.environ.get(
    "LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG_DIR",
    os.path.join(PROJECT_DIR, "runtime_logs", "realtime_control_prob"),
)
STARTUP_TOKEN2WAV_FIRST = str(
    os.environ.get("LYCHEEFD_STARTUP_TOKEN2WAV_FIRST", "1")
).strip().lower() in {"1", "true", "yes", "on"}
STARTUP_WARMUP = str(
    os.environ.get("LYCHEEFD_STARTUP_WARMUP", "1")
).strip().lower() in {"1", "true", "yes", "on"}
STARTUP_WARMUP_PROMPT_VOICE = str(
    os.environ.get("LYCHEEFD_STARTUP_WARMUP_PROMPT_VOICE", "male")
).strip()
STARTUP_WARMUP_TOKEN = max(
    0,
    min(
        6560,
        int(os.environ.get("LYCHEEFD_STARTUP_WARMUP_TOKEN", "100")),
    ),
)
if INPUT_SILENCE_GATE_ENABLED:
    logger.info(
        "Input silence gate enabled: frame_ms=%d open_dbfs=%.1f close_dbfs=%.1f hangover_ms=%d preroll_ms=%d",
        INPUT_SILENCE_GATE_FRAME_MS,
        INPUT_SILENCE_GATE_OPEN_DBFS,
        INPUT_SILENCE_GATE_CLOSE_DBFS,
        INPUT_SILENCE_GATE_HANGOVER_MS,
        INPUT_SILENCE_GATE_PREROLL_MS,
    )


def _readable_time_tag(epoch_ms: Optional[int] = None) -> str:
    if epoch_ms is None:
        dt = datetime.datetime.now()
    else:
        dt = datetime.datetime.fromtimestamp(float(epoch_ms) / 1000.0)
    return dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _available_realtime_prompt_voices() -> List[dict]:
    voices = []
    clone_root = os.path.abspath(CLONE_PROMPT_DIR)
    for item in REALTIME_PROMPT_VOICE_OPTIONS:
        wav_name = str(item.get("wav", "")).strip()
        text_name = str(item.get("text", "")).strip()
        wav_path = os.path.abspath(os.path.join(clone_root, wav_name))
        text_path = os.path.abspath(os.path.join(clone_root, text_name)) if text_name else ""
        if not wav_path.startswith(clone_root + os.sep):
            continue
        if text_path and not text_path.startswith(clone_root + os.sep):
            continue
        if not os.path.isfile(wav_path):
            continue
        if text_path and not os.path.isfile(text_path):
            continue
        voices.append(
            {
                "id": str(item.get("id", "")).strip(),
                "label": str(item.get("label", "")).strip(),
                "wav": wav_name,
                "text": text_name,
            }
        )
    return voices


def _normalize_prompt_voice_id(prompt_voice: str) -> str:
    voice = str(prompt_voice or "").strip()
    if voice.lower().startswith("clone:"):
        voice = voice.split(":", 1)[1].strip()
    return voice


def _resolve_prompt_wav_path(prompt_voice: str) -> str:
    voice = _normalize_prompt_voice_id(prompt_voice)
    voice_lower = voice.lower()
    if not voice_lower:
        voice_lower = DEFAULT_REALTIME_PROMPT_VOICE.lower()

    for item in _available_realtime_prompt_voices():
        item_id = str(item.get("id", "")).strip()
        item_label = str(item.get("label", "")).strip()
        wav_name = str(item.get("wav", "")).strip()
        accepted = {
            item_id.lower(),
            item_label.lower(),
            os.path.splitext(wav_name)[0].lower(),
        }
        if voice_lower in accepted:
            return os.path.abspath(os.path.join(CLONE_PROMPT_DIR, wav_name))

    if voice_lower in {"男声", "male", "man", "m", "default_male"}:
        return os.path.join(ASSETS_DIR, "default_male.wav")
    if voice_lower in {"女声", "woman", "f", "default_female"}:
        return os.path.join(ASSETS_DIR, "default_female.wav")

    available_ids = ", ".join(item["id"] for item in _available_realtime_prompt_voices())
    raise ValueError(f"Unknown prompt voice: {prompt_voice!r}. Available clone voices: {available_ids}")


def warmup_token2wav(prompt_voice: str = STARTUP_WARMUP_PROMPT_VOICE) -> str:
    """
    Prime token2wav CUDA kernels/buffers before first realtime request.
    """
    global token2wav_model

    if token2wav_model is None:
        return "Token2Wav warmup skipped: model is not loaded."

    prompt_wav_path = _resolve_prompt_wav_path(prompt_voice)
    if not os.path.isfile(prompt_wav_path):
        return f"Token2Wav warmup skipped: prompt wav not found ({prompt_wav_path})."

    pre_lookahead_len = int(getattr(token2wav_model.flow, "pre_lookahead_len", 0))
    emit_window = int(max(1, TTS_VOCODER_HOP_SIZE + pre_lookahead_len))
    warmup_token = int(STARTUP_WARMUP_TOKEN)
    warmup_tokens = [warmup_token for _ in range(emit_window + 1)]

    t0 = time.perf_counter()
    try:
        token2wav_model.set_stream_cache(prompt_wav_path)
        token2wav_model.stream(
            warmup_tokens,
            prompt_wav=prompt_wav_path,
            last_chunk=False,
        )
        token2wav_model.stream(
            [warmup_token],
            prompt_wav=prompt_wav_path,
            last_chunk=True,
        )
    except Exception as exc:
        logger.warning("Token2Wav warmup failed: %s", exc)
        return f"Token2Wav warmup failed: {exc}"

    cost_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Token2Wav warmup done: token=%d, emit_window=%d, lookahead=%d, cost=%.1fms",
        warmup_token,
        emit_window,
        pre_lookahead_len,
        cost_ms,
    )
    return (
        "Token2Wav warmup done "
        f"(token={warmup_token}, emit_window={emit_window}, cost_ms={cost_ms:.1f})"
    )


class RemoteToken2WavClient:
    """HTTP client for the remote Token2Wav streaming service."""

    def __init__(self, base_url: str, timeout_sec: float):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote token2wav HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"remote token2wav request failed: {url}: {exc}") from exc
        try:
            obj = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"remote token2wav returned invalid JSON: {body[:200]!r}") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(f"remote token2wav returned non-object JSON: {type(obj).__name__}")
        if obj.get("ok") is False:
            raise RuntimeError(str(obj.get("error") or "remote token2wav error"))
        return obj

    def health(self) -> dict:
        """Check remote Token2Wav service health."""
        return self._post_json("/v1/token2wav/health", {})

    def start(self, *, stream_id: str, prompt_wav: str) -> dict:
        """Start one remote streaming synthesis session."""
        return self._post_json(
            "/v1/token2wav/start",
            {"stream_id": str(stream_id), "prompt_wav": str(prompt_wav)},
        )

    def stream(
        self,
        *,
        stream_id: str,
        prompt_wav: str,
        tokens: List[int],
        last_chunk: bool,
        advance_tokens: int,
    ) -> dict:
        """Submit one stoken chunk and return playable PCM bytes."""
        obj = self._post_json(
            "/v1/token2wav/stream",
            {
                "stream_id": str(stream_id),
                "prompt_wav": str(prompt_wav),
                "tokens": [int(x) for x in tokens],
                "last_chunk": bool(last_chunk),
                "advance_tokens": int(advance_tokens),
            },
        )
        pcm_b64 = obj.get("pcm_b64")
        if isinstance(pcm_b64, str) and pcm_b64:
            obj["pcm_bytes"] = base64.b64decode(pcm_b64.encode("ascii"))
        else:
            obj["pcm_bytes"] = b""
        return obj

    def close(self, *, stream_id: str) -> dict:
        """Close a remote streaming synthesis session."""
        return self._post_json("/v1/token2wav/close", {"stream_id": str(stream_id)})


_remote_token2wav_client: Optional[RemoteToken2WavClient] = None
_token2wav_load_lock = threading.Lock()


def get_remote_token2wav_client() -> RemoteToken2WavClient:
    """Return the process-wide remote Token2Wav client singleton."""
    global _remote_token2wav_client
    if _remote_token2wav_client is None:
        _remote_token2wav_client = RemoteToken2WavClient(
            REMOTE_TOKEN2WAV_URL,
            REMOTE_TOKEN2WAV_TIMEOUT_SEC,
        )
    return _remote_token2wav_client


def ensure_local_token2wav_loaded(token2wav_path: Optional[str] = None) -> None:
    """Ensure local Token2Wav is loaded as the fallback path for remote failures."""
    global token2wav_model
    if token2wav_model is not None:
        return
    with _token2wav_load_lock:
        if token2wav_model is not None:
            return
        load_path = token2wav_path or DEFAULT_TOKEN2WAV_PATH
        logger.info("Loading local Token2Wav fallback, path=%s", load_path)
        t0 = time.time()
        Token2wav = _import_token2wav_class()
        token2wav_model = Token2wav(load_path)
        logger.info("Local Token2Wav fallback loaded in %.2fs", time.time() - t0)


def get_token2wav_pre_lookahead_len() -> int:
    """Return the Token2Wav pre-lookahead length."""
    if token2wav_model is not None:
        try:
            return int(getattr(token2wav_model.flow, "pre_lookahead_len", 0))
        except Exception:
            pass
    return int(REMOTE_TOKEN2WAV_PRE_LOOKAHEAD_LEN)


def is_token2wav_available() -> bool:
    """Return whether local or remote Token2Wav is available."""
    return token2wav_model is not None or bool(REMOTE_TOKEN2WAV_ENABLED)


class RealtimeTTSPool:
    """Session-level Token2Wav bridge.

    It accepts stoken chunks asynchronously, emits PCM after the hop+lookahead
    threshold, and flushes tail audio on event_end/session_stop.
    """

    def __init__(
        self,
        *,
        prompt_wav_path: str,
        vocoder_hop_size: int,
        pre_lookahead_len: int,
        worker_name: str = "stepaudio_tts_worker",
        persistent_mode: bool = False,
        pcm_emit_callback: Optional[Callable[[dict], None]] = None,
        queue_pcm_messages: bool = True,
    ):
        self.prompt_wav_path = str(prompt_wav_path)
        self.vocoder_hop_size = int(vocoder_hop_size)
        self.pre_lookahead_len = int(pre_lookahead_len)
        self.persistent_mode = bool(persistent_mode)
        self._pcm_emit_callback = pcm_emit_callback if callable(pcm_emit_callback) else None
        self._queue_pcm_messages = bool(queue_pcm_messages)
        self._stream_id = uuid.uuid4().hex
        self._use_remote_t2w = bool(REMOTE_TOKEN2WAV_ENABLED)
        self._remote_t2w_disabled = False
        self._task_queue: "queue.Queue[object]" = queue.Queue(maxsize=1024)
        self._out_queue: "queue.Queue[dict]" = queue.Queue()
        self._stop_sentinel = object()
        self._error: Optional[str] = None
        self._state_lock = threading.Lock()
        self._event_active = False
        self._generation_id = 0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=str(worker_name),
            daemon=True,
        )
        self._worker.start()

    @property
    def event_active(self) -> bool:
        """Return whether a TTS event is currently active."""
        with self._state_lock:
            return bool(self._event_active)

    def _set_event_active(self, value: bool) -> None:
        with self._state_lock:
            self._event_active = bool(value)

    def _current_generation_id(self) -> int:
        with self._state_lock:
            return int(self._generation_id)

    def _advance_generation(self) -> int:
        with self._state_lock:
            self._generation_id += 1
            return int(self._generation_id)

    def _rotate_stream_id_for_abort(self) -> tuple[str, str, int]:
        with self._state_lock:
            old_stream_id = str(self._stream_id)
            self._stream_id = uuid.uuid4().hex
            self._generation_id += 1
            self._event_active = False
            return old_stream_id, str(self._stream_id), int(self._generation_id)

    def _get_stream_id(self) -> str:
        with self._state_lock:
            return str(self._stream_id)

    @property
    def pcm_callback_enabled(self) -> bool:
        """Return whether immediate PCM callback emission is enabled."""
        return self._pcm_emit_callback is not None

    def submit_event_start(self) -> None:
        """Notify the TTS worker that a new speech event starts."""
        generation_id = self._advance_generation()
        self._task_queue.put(
            {"type": "event_start", "generation_id": int(generation_id)}
        )

    def submit_audio_chunk(self, stoken_ids_raw: List[int]) -> None:
        """Submit one audio-token chunk to the TTS worker."""
        generation_id = self._current_generation_id()
        self._task_queue.put(
            {
                "type": "audio_chunk",
                "stoken_ids_raw": list(stoken_ids_raw),
                "generation_id": int(generation_id),
            }
        )

    def submit_event_end(self, *, force_flush: bool = True) -> None:
        """Notify the TTS worker to finish the current event and optionally flush."""
        generation_id = self._current_generation_id()
        self._task_queue.put(
            {
                "type": "event_end",
                "force_flush": bool(force_flush),
                "generation_id": int(generation_id),
            }
        )

    def submit_event_abort(self, *, reason: str = "interrupt") -> dict:
        """Abort the current TTS event and rotate the stream id."""
        old_stream_id, new_stream_id, generation_id = self._rotate_stream_id_for_abort()
        task = {
            "type": "event_abort",
            "reason": str(reason or "interrupt"),
            "old_stream_id": old_stream_id,
            "new_stream_id": new_stream_id,
            "generation_id": int(generation_id),
        }
        self._task_queue.put(task)
        return dict(task)

    def submit_session_stop(self, *, force_flush: bool = True) -> None:
        """Notify the TTS worker that the realtime session is stopping."""
        generation_id = self._current_generation_id()
        self._task_queue.put(
            {
                "type": "session_stop",
                "force_flush": bool(force_flush),
                "generation_id": int(generation_id),
            }
        )

    def drain(
        self,
        *,
        block: bool = False,
        timeout: float = 0.0,
        until_event_stats: bool = False,
    ) -> tuple[List[dict], Optional[dict]]:
        """Drain generated PCM/stat messages from the worker."""
        messages: List[dict] = []
        event_stats: Optional[dict] = None
        first_block = bool(block)
        while True:
            try:
                if first_block:
                    msg = self._out_queue.get(timeout=float(timeout))
                    first_block = False
                else:
                    msg = self._out_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, dict):
                messages.append(msg)
                if msg.get("type") == "event_stats":
                    event_stats = msg
                    if until_event_stats:
                        break
        return messages, event_stats

    def wait_event_stats(self, max_wait_sec: float = 30.0) -> tuple[List[dict], Optional[dict]]:
        """Wait for the current event stats message."""
        collected: List[dict] = []
        deadline = time.perf_counter() + max(0.0, float(max_wait_sec))
        while time.perf_counter() < deadline:
            msgs, event_stats = self.drain(block=True, timeout=0.1, until_event_stats=True)
            if msgs:
                collected.extend(msgs)
            if event_stats is not None:
                return collected, event_stats
        return collected, None

    def stop(self, *, force_flush: bool = False, flush_wait_sec: float = 5.0) -> List[dict]:
        """Stop the TTS worker and return messages collected before shutdown."""
        collected: List[dict] = []
        if force_flush:
            try:
                self.submit_session_stop(force_flush=True)
                msgs, _ = self.wait_event_stats(max_wait_sec=float(flush_wait_sec))
                if msgs:
                    collected.extend(msgs)
            except Exception:
                pass
        try:
            self._task_queue.put_nowait(self._stop_sentinel)
        except Exception:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        msgs, _ = self.drain(block=False)
        if msgs:
            collected.extend(msgs)
        return collected

    def _worker_loop(self) -> None:
        global token2wav_model

        tts_initialized = False
        tts_token_buf: List[int] = []
        worker_raw_tokens = 0
        worker_unique_tokens = 0
        worker_calls = 0
        worker_synth_total = 0.0

        def _reset_event_state(clear_cache: bool = True) -> None:
            nonlocal tts_initialized, tts_token_buf
            nonlocal worker_raw_tokens, worker_unique_tokens, worker_calls, worker_synth_total
            if clear_cache:
                tts_initialized = False
                tts_token_buf = []
            worker_raw_tokens = 0
            worker_unique_tokens = 0
            worker_calls = 0
            worker_synth_total = 0.0

        def _remote_available() -> bool:
            return bool(self._use_remote_t2w) and not bool(self._remote_t2w_disabled)

        def _disable_remote_after_error(exc: Exception) -> None:
            self._remote_t2w_disabled = True
            logger.warning(
                "Remote Token2Wav disabled for stream %s after error: %s",
                self._stream_id,
                exc,
            )
            if not REMOTE_TOKEN2WAV_FALLBACK:
                raise RuntimeError(f"Remote Token2Wav failed and fallback is disabled: {exc}") from exc
            ensure_local_token2wav_loaded()

        def _start_tts_stream() -> None:
            stream_id = self._get_stream_id()
            if _remote_available():
                try:
                    get_remote_token2wav_client().start(
                        stream_id=stream_id,
                        prompt_wav=self.prompt_wav_path,
                    )
                    return
                except Exception as exc:
                    _disable_remote_after_error(exc)
            ensure_local_token2wav_loaded()
            token2wav_model.set_stream_cache(self.prompt_wav_path)

        def _emit_tts(tokens: List[int], is_last: bool, advance_tokens: int) -> None:
            nonlocal worker_raw_tokens, worker_unique_tokens, worker_calls, worker_synth_total
            if not tokens:
                return
            emit_generation_id = self._current_generation_id()
            emit_stream_id = self._get_stream_id()
            worker_raw_tokens += int(len(tokens))
            worker_unique_tokens += max(0, int(advance_tokens))
            worker_calls += 1
            t_synth_start_epoch_ms = int(time.time() * 1000)
            t_synth_start = time.perf_counter()
            remote_meta = None
            if _remote_available():
                try:
                    remote_meta = get_remote_token2wav_client().stream(
                        stream_id=emit_stream_id,
                        prompt_wav=self.prompt_wav_path,
                        tokens=tokens,
                        last_chunk=bool(is_last),
                        advance_tokens=int(advance_tokens),
                    )
                    pcm = remote_meta.get("pcm_bytes", b"")
                except Exception as exc:
                    _disable_remote_after_error(exc)
                    token2wav_model.set_stream_cache(self.prompt_wav_path)
                    pcm = token2wav_model.stream(
                        tokens,
                        prompt_wav=self.prompt_wav_path,
                        last_chunk=bool(is_last),
                    )
            else:
                ensure_local_token2wav_loaded()
                pcm = token2wav_model.stream(
                    tokens,
                    prompt_wav=self.prompt_wav_path,
                    last_chunk=bool(is_last),
                )
            t_synth_cost = time.perf_counter() - t_synth_start
            t_synth_end_epoch_ms = int(time.time() * 1000)
            if isinstance(remote_meta, dict):
                try:
                    t_synth_start_epoch_ms = int(remote_meta.get("synth_start_epoch_ms", t_synth_start_epoch_ms))
                    t_synth_end_epoch_ms = int(remote_meta.get("synth_end_epoch_ms", t_synth_end_epoch_ms))
                    t_synth_cost = float(remote_meta.get("synth_sec", t_synth_cost))
                except Exception:
                    pass
            worker_synth_total += float(t_synth_cost)
            if emit_generation_id != self._current_generation_id():
                self._out_queue.put(
                    {
                        "type": "pcm_dropped",
                        "reason": "generation_aborted",
                        "tokens": int(len(tokens)),
                        "advance_tokens": int(advance_tokens),
                        "generation_id": int(emit_generation_id),
                        "stream_id": emit_stream_id,
                        "synth_sec": float(t_synth_cost),
                    }
                )
                return
            pcm_msg = {
                "type": "pcm",
                "pcm_bytes": bytes(pcm),
                "tokens": int(len(tokens)),
                "advance_tokens": int(advance_tokens),
                "is_last": bool(is_last),
                "synth_sec": float(t_synth_cost),
                "synth_start_epoch_ms": int(t_synth_start_epoch_ms),
                "synth_end_epoch_ms": int(t_synth_end_epoch_ms),
                "synth_duration_ms": round(float(t_synth_cost) * 1000.0, 3),
                "t2w_backend": "remote" if isinstance(remote_meta, dict) else "local",
                "generation_id": int(emit_generation_id),
                "stream_id": emit_stream_id,
            }
            if isinstance(remote_meta, dict):
                for key in (
                    "remote_recv_epoch_ms",
                    "remote_return_epoch_ms",
                    "remote_queue_wait_ms",
                    "remote_roundtrip_ms",
                ):
                    if key in remote_meta:
                        pcm_msg[key] = remote_meta[key]

            callback_ok = False
            if self._pcm_emit_callback is not None:
                try:
                    self._pcm_emit_callback(dict(pcm_msg))
                    callback_ok = True
                except Exception:
                    cb_err = traceback.format_exc()
                    logger.error("RealtimeTTSPool pcm_emit_callback failed: %s", cb_err)
                    self._out_queue.put(
                        {"type": "error", "detail": f"TTS pcm callback failed:\n{cb_err}"}
                    )

            should_queue_pcm = bool(self._queue_pcm_messages) or not bool(callback_ok)
            if should_queue_pcm:
                if self._pcm_emit_callback is not None and not callback_ok and not self._queue_pcm_messages:
                    pcm_msg["callback_fallback"] = True
                self._out_queue.put(pcm_msg)

        def _synth_stoken_chunk(stoken_ids_raw: List[int], is_last: bool = False) -> None:
            nonlocal tts_initialized, tts_token_buf
            t_chunk_start = time.perf_counter()
            clean = [int(x) for x in stoken_ids_raw if int(x) < 6561]
            if clean:
                tts_token_buf.extend(clean)

            if not tts_initialized:
                if not tts_token_buf:
                    self._out_queue.put(
                        {"type": "synth_chunk_done", "sec": float(time.perf_counter() - t_chunk_start)}
                    )
                    return
                _start_tts_stream()
                tts_initialized = True

            if not is_last:
                emit_window = int(self.vocoder_hop_size + self.pre_lookahead_len)
                while len(tts_token_buf) > emit_window:
                    emit_tokens = list(tts_token_buf[:emit_window])
                    tts_token_buf = tts_token_buf[int(self.vocoder_hop_size):]
                    _emit_tts(
                        emit_tokens,
                        is_last=False,
                        advance_tokens=int(self.vocoder_hop_size),
                    )
            else:
                if tts_token_buf:
                    flush_tokens = list(tts_token_buf)
                    _emit_tts(flush_tokens, is_last=True, advance_tokens=len(flush_tokens))
                    tts_token_buf = []
                tts_initialized = False

            self._out_queue.put(
                {"type": "synth_chunk_done", "sec": float(time.perf_counter() - t_chunk_start)}
            )

        try:
            while True:
                task = self._task_queue.get()
                if task is self._stop_sentinel:
                    break
                if not isinstance(task, dict):
                    continue
                task_type = task.get("type")
                if task_type == "event_start":
                    task_generation_id = int(task.get("generation_id") or -1)
                    if task_generation_id != self._current_generation_id():
                        continue
                    _reset_event_state(clear_cache=True)
                    self._set_event_active(True)
                    continue
                if task_type == "audio_chunk":
                    task_generation_id = int(task.get("generation_id") or -1)
                    if task_generation_id != self._current_generation_id():
                        self._out_queue.put(
                            {
                                "type": "stale_task_dropped",
                                "task_type": "audio_chunk",
                                "generation_id": int(task_generation_id),
                                "current_generation_id": int(self._current_generation_id()),
                                "tokens": int(len(task.get("stoken_ids_raw", []) or [])),
                            }
                        )
                        continue
                    _synth_stoken_chunk(task.get("stoken_ids_raw", []) or [], is_last=False)
                    continue
                if task_type == "event_abort":
                    old_stream_id = str(task.get("old_stream_id") or "")
                    reason = str(task.get("reason") or "interrupt")
                    _reset_event_state(clear_cache=True)
                    self._set_event_active(False)
                    if old_stream_id and self._use_remote_t2w and not self._remote_t2w_disabled:
                        try:
                            get_remote_token2wav_client().close(stream_id=old_stream_id)
                        except Exception:
                            logger.debug("Remote Token2Wav close on abort failed", exc_info=True)
                    self._out_queue.put(
                        {
                            "type": "event_aborted",
                            "reason": reason,
                            "old_stream_id": old_stream_id,
                            "new_stream_id": str(task.get("new_stream_id") or ""),
                            "generation_id": int(task.get("generation_id") or self._current_generation_id()),
                            "vocoder_raw_tokens": int(worker_raw_tokens),
                            "vocoder_unique_tokens": int(worker_unique_tokens),
                            "vocoder_calls": int(worker_calls),
                            "synth_total_time": float(worker_synth_total),
                        }
                    )
                    continue
                if task_type in {"event_end", "session_stop"}:
                    task_generation_id = int(task.get("generation_id") or -1)
                    if task_generation_id != self._current_generation_id():
                        self._out_queue.put(
                            {
                                "type": "stale_task_dropped",
                                "task_type": str(task_type),
                                "generation_id": int(task_generation_id),
                                "current_generation_id": int(self._current_generation_id()),
                            }
                        )
                        continue
                    do_flush = bool(task.get("force_flush", True))
                    if do_flush:
                        _synth_stoken_chunk([], is_last=True)
                    self._out_queue.put(
                        {
                            "type": "event_stats",
                            "vocoder_raw_tokens": int(worker_raw_tokens),
                            "vocoder_unique_tokens": int(worker_unique_tokens),
                            "vocoder_calls": int(worker_calls),
                            "synth_total_time": float(worker_synth_total),
                        }
                    )
                    self._set_event_active(False)
                    _reset_event_state(clear_cache=True)
                    continue
        except Exception:
            self._error = traceback.format_exc()
            self._out_queue.put({"type": "error", "detail": self._error})
        finally:
            if self._use_remote_t2w and not self._remote_t2w_disabled:
                try:
                    get_remote_token2wav_client().close(stream_id=self._get_stream_id())
                except Exception:
                    logger.debug("Remote Token2Wav close failed", exc_info=True)

# ==================== Model Loading ====================
USE_VLLM_BACKEND = os.environ.get("LYCHEEFD_USE_VLLM", "0") == "1"


def _env_optional_bool(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    logger.warning("Invalid boolean env %s=%s, fallback to %s", name, raw, default)
    return default


def _env_optional_int(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer env %s=%s, fallback to %s", name, raw, default)
        return default


def _env_optional_float(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%s, fallback to %s", name, raw, default)
        return default

def _infer_model_type(model_path: str) -> str:
    """Infer model_type from config metadata or path markers, then fallback to FD."""
    cfg = _load_model_config_dict(model_path)
    if _config_has_v9_merge(cfg):
        return "V9"

    norm_path = os.path.normpath(str(model_path or ""))
    candidates = [
        os.path.basename(norm_path),
        os.path.basename(os.path.dirname(norm_path)),
    ]
    for name in candidates:
        match = re.search(r"(?:^|_)v(\d+)(?:_|$)", str(name).lower())
        if match:
            return f"V{match.group(1)}"

    logger.warning(
        "Cannot infer model_type from path '%s'; fallback to 'FD'.",
        model_path,
    )
    return "FD"


def _infer_model_type_legacy(model_path: str) -> str:
    """Legacy path-based model_type inference used by the original vLLM branch."""
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(model_path)))
    parts = [p for p in parent_dir.split("_") if p]
    if len(parts) >= 3:
        return parts[2].upper()

    logger.warning(
        "Cannot infer legacy model_type from path '%s' (parent='%s'); fallback to 'FD'.",
        model_path,
        parent_dir,
    )
    return "FD"


def _load_model_config_dict(model_path: str) -> dict:
    config_path = os.path.join(os.path.normpath(str(model_path or "")), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to read model config %s: %s", config_path, exc)
        return {}


def _config_has_v9_merge(config: dict) -> bool:
    merge_cfg = config.get("merge_layer_config") if isinstance(config, dict) else None
    if not isinstance(merge_cfg, dict):
        return False
    try:
        return int(merge_cfg.get("num_hidden_layers", 0) or 0) > 0
    except (TypeError, ValueError):
        return True


def _should_use_hf_v9(model_path: str, model_type: str) -> bool:
    if str(model_type).upper() == "V9":
        return True
    return _config_has_v9_merge(_load_model_config_dict(model_path))


def load_models(model_path, token2wav_path, attn_impl="eager"):
    """Load the realtime inference framework and the Token2Wav vocoder."""
    global generator, token2wav_model

    status_msgs = []
    logger.info(f"Loading models, model_path={model_path}, token2wav_path={token2wav_path}")

    def _ensure_generator_loaded() -> None:
        global generator
        if generator is not None:
            status_msgs.append("Model already loaded (skipped)")
            return
        status_msgs.append("Loading duplex inference model...")
        t0 = time.time()

        if USE_VLLM_BACKEND:
            logger.info("Using vLLM backend (LYCHEEFD_USE_VLLM=1)")
            VLLMGenerationFramework = _import_vllm_generation_framework()
            model_type = _infer_model_type_legacy(model_path)
            vllm_enforce_eager = _env_optional_bool("LYCHEEFD_VLLM_ENFORCE_EAGER", True)
            vllm_enable_chunked_prefill = _env_optional_bool("LYCHEEFD_VLLM_ENABLE_CHUNKED_PREFILL", True)
            vllm_enable_prefix_caching = _env_optional_bool("LYCHEEFD_VLLM_ENABLE_PREFIX_CACHING", True)
            vllm_max_num_seqs = _env_optional_int("LYCHEEFD_VLLM_MAX_NUM_SEQS", None)
            vllm_max_num_batched_tokens = _env_optional_int("LYCHEEFD_VLLM_MAX_NUM_BATCHED_TOKENS", None)
            vllm_tensor_parallel_size = _env_optional_int("LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE", 1)
            vllm_pipeline_parallel_size = _env_optional_int("LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE", 1)
            vllm_gpu_memory_utilization = _env_optional_float("LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION", 0.85)
            vllm_max_model_len = _env_optional_int("LYCHEEFD_VLLM_MAX_MODEL_LEN", 8192)
            generator = VLLMGenerationFramework(
                model_type=model_type,
                model_path=model_path,
                device="cuda",
                torch_dtype=torch.bfloat16,
                align_audio_input=True,
                allowing_backchannel=ALLOWING_BACKCHANNEL,
                enforce_eager=vllm_enforce_eager,
                enable_chunked_prefill=vllm_enable_chunked_prefill,
                enable_prefix_caching=vllm_enable_prefix_caching,
                max_num_seqs=vllm_max_num_seqs,
                max_num_batched_tokens=vllm_max_num_batched_tokens,
                tensor_parallel_size=vllm_tensor_parallel_size,
                pipeline_parallel_size=vllm_pipeline_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=vllm_max_model_len,
            )
            status_msgs.append(
                "vLLM opts: "
                f"enforce_eager={vllm_enforce_eager}, "
                f"chunked_prefill={vllm_enable_chunked_prefill}, "
                f"prefix_caching={vllm_enable_prefix_caching}, "
                f"max_num_seqs={vllm_max_num_seqs}, "
                f"max_num_batched_tokens={vllm_max_num_batched_tokens}, "
                f"tp={vllm_tensor_parallel_size}, "
                f"pp={vllm_pipeline_parallel_size}, "
                f"gpu_mem_util={vllm_gpu_memory_utilization}, "
                f"max_model_len={vllm_max_model_len}"
            )
            status_msgs.append(f"vLLM model loaded ({time.time()-t0:.1f}s)")
        else:
            model_type = _infer_model_type(model_path)
            if _should_use_hf_v9(model_path, model_type):
                logger.info("Using HuggingFace V9 realtime backend (LYCHEEFD_USE_VLLM=0)")
                HFRealtimeV9GenerationFramework = _import_hf_v9_generation_framework()
                generator = HFRealtimeV9GenerationFramework(
                    model_type="V9",
                    model_path=model_path,
                    device="cuda",
                    attn_implementation=attn_impl,
                    torch_dtype=torch.bfloat16,
                    align_audio_input=True,
                    allowing_backchannel=ALLOWING_BACKCHANNEL,
                )
                status_msgs.append(f"HF V9 model loaded ({time.time()-t0:.1f}s)")
            else:
                raise RuntimeError(
                    "HF backend in the public Lychee-FD release supports V9 checkpoints only. "
                    f"Detected model_type={model_type!r}; please use a checkpoint with merge_layer_config "
                    "or run the vLLM backend."
                )

        logger.info("Model loaded in %.2fs", time.time() - t0)

    def _ensure_token2wav_loaded() -> None:
        global token2wav_model
        if REMOTE_TOKEN2WAV_ENABLED:
            status_msgs.append(
                "Token2Wav local load skipped; remote service enabled "
                f"({REMOTE_TOKEN2WAV_URL}, fallback={REMOTE_TOKEN2WAV_FALLBACK})"
            )
            try:
                health = get_remote_token2wav_client().health()
                logger.info("Remote Token2Wav health: %s", health)
                status_msgs.append("Remote Token2Wav health check OK")
            except Exception as exc:
                msg = f"Remote Token2Wav health check failed: {exc}"
                logger.warning(msg)
                if not REMOTE_TOKEN2WAV_FALLBACK:
                    raise RuntimeError(msg) from exc
                status_msgs.append(msg + " (local fallback will load lazily if needed)")
            return
        if token2wav_model is not None:
            status_msgs.append("Token2Wav already loaded (skipped)")
            return
        status_msgs.append("Loading Token2Wav vocoder...")
        t0 = time.time()
        ensure_local_token2wav_loaded(token2wav_path)
        logger.info("Token2Wav loaded in %.2fs", time.time() - t0)
        status_msgs.append(f"Token2Wav loaded ({time.time()-t0:.1f}s)")

    if STARTUP_TOKEN2WAV_FIRST:
        _ensure_token2wav_loaded()
        _ensure_generator_loaded()
    else:
        _ensure_generator_loaded()
        _ensure_token2wav_loaded()

    if STARTUP_WARMUP and not REMOTE_TOKEN2WAV_ENABLED:
        status_msgs.append(warmup_token2wav(prompt_voice=STARTUP_WARMUP_PROMPT_VOICE))
    elif STARTUP_WARMUP and REMOTE_TOKEN2WAV_ENABLED:
        status_msgs.append("Token2Wav warmup skipped in backend process because remote Token2Wav is enabled.")
    else:
        status_msgs.append("Token2Wav warmup skipped by env (LYCHEEFD_STARTUP_WARMUP=0).")

    return "\n".join(status_msgs)


# ==================== Utilities ====================
def pcm_to_numpy(pcm_bytes):
    """Convert vocoder PCM bytes into a numpy array that Gradio can play."""
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data


def _get_state_label(state_char):
    """Return a display label for an internal listening-state code."""
    labels = {"l": "listening", "s": "speaking", "b": "backchannel"}
    return labels.get(state_char, f"unknown({state_char})")


# ==================== Online Inference Path ====================

def run_chunk_dialogue_inference(
    audio_input,
    start_speak_factor,
    end_speak_factor,
    prompt_voice,
    tts_chunk_size,
    prefix=None,
    initial_listening_state="l",
    realtime_ctx=None,
    stream_session=None,
    stream_audio_incremental=False,
    stream_flush_audio_tail=False,
    start_listen_factor: float = 1.2,
    tts_bridge: Optional[RealtimeTTSPool] = None,
    flush_tts_on_round_end: bool = True,
    direct_text_callback=None,
    direct_state_callback=None,
):
    """Process audio in the online path and stream UI/realtime events.

    The audio is fed directly into the model without extra padding. Results are
    split by time window and yielded to either the Gradio compatibility API or
    the realtime HTTP session.

    Yields:
        chatbot_messages: Chat messages.
        output_audio_path: Synthesized response audio.
        status_text: Current status text.
        json_text: Raw JSON result.
        audio_debug_text: Audio-token diagnostics.
    """
    global generator, token2wav_model

    if generator is None or not is_token2wav_available():
        yield [], None, "Models are not loaded yet.", "", ""
        return

    if audio_input is None:
        yield [], None, "Please upload an audio file first.", "", ""
        return

    StreamingDecoder = _import_streaming_decoder()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"========== Streaming inference run_id={run_id} ==========")

    try:
        profile_latency = bool(_env_optional_bool("LYCHEEFD_PROFILE_LATENCY", False))
        profile_every = _env_optional_int("LYCHEEFD_PROFILE_EVERY", 20)
        try:
            profile_every = max(1, int(profile_every) if profile_every is not None else 20)
        except (TypeError, ValueError):
            profile_every = 20
        if profile_latency:
            logger.info(
                "[LAT] profiling enabled (LYCHEEFD_PROFILE_LATENCY=1, every=%d)",
                profile_every,
            )
        t_call_start_perf = time.perf_counter()

        # --- Audio preprocessing ---
        t_audio_pre_start = time.perf_counter()
        if isinstance(audio_input, tuple):
            sr, audio_np = audio_input
            audio_np = audio_np.astype(np.float32)
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
            max_val = max(abs(audio_np.max()), abs(audio_np.min()))
            if max_val > 1.0:
                audio_np = audio_np / max_val
            if sr != INPUT_SAMPLE_RATE:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=INPUT_SAMPLE_RATE)
        elif isinstance(audio_input, str):
            audio_np, _ = librosa.load(audio_input, sr=INPUT_SAMPLE_RATE)
        else:
            yield [], None, f"Unrecognized audio format: {type(audio_input)}", "", ""
            return

        total_duration = len(audio_np) / INPUT_SAMPLE_RATE
        logger.info(f"Input audio: {total_duration:.2f}s, samples={len(audio_np)}")

        input_path = os.path.join(TEMP_DIR, f"input_{run_id}.wav")
        sf.write(input_path, audio_np, INPUT_SAMPLE_RATE)
        audio_pre_sec = time.perf_counter() - t_audio_pre_start
        if profile_latency:
            logger.info(
                "[LAT] audio_preprocess=%.4fs duration=%.3fs samples=%d",
                audio_pre_sec,
                total_duration,
                len(audio_np),
            )

        prompt_wav_path = _resolve_prompt_wav_path(prompt_voice)

        if not os.path.isfile(prompt_wav_path):
            err_msg = (
                f"Prompt voice file not found: {prompt_wav_path}\n\n"
                f"Place 16 kHz mono WAV prompt files under {ASSETS_DIR}, for example:\n"
                f"  - default_male.wav\n"
                f"  - default_female.wav\n"
                "Token2Wav cannot synthesize speech without a prompt WAV."
            )
            logger.error(err_msg)
            yield [
                {"role": "user", "content": f"**Upload**: {total_duration:.2f}s audio"},
                {"role": "assistant", "content": f"**Error**\n\n{err_msg}"},
                "Missing prompt voice file",
                "",
                "",
            ]
            return

        # --- Build chatbot initial state ---
        chatbot_msgs = []
        chatbot_msgs.append({
            "role": "user",
            "content": f"**Upload**: {total_duration:.2f}s audio\n\nStreaming inference in progress..."
        })
        yield (
            chatbot_msgs,
            None,
            f"Streaming... (input: {total_duration:.2f}s)",
            "",
            "[Audio Token / RTF] waiting for the first audio token...",
        )

        t_start = time.time()
        t_stream_start_perf = time.perf_counter()
        round_trace = {
            "trace_version": 1,
            "run_id": run_id,
            "round_started_at_epoch_ms": int(time.time() * 1000),
            "input_duration_sec": float(total_duration),
            "state_changes": [],
            "state_changes_raw": [],
            "events": [],
            "token_timeline": [],
            "timeline_spans": [],
            "control_prob_points": [],
        }
        current_event_trace = None
        latency_stats = {
            "raw_wait_sec": 0.0,
            "raw_handle_sec": 0.0,
            "decoder_sec": 0.0,
            "audio_chunk_sec": 0.0,
            "emit_tts_sec": 0.0,
            "emit_tts_calls": 0,
            "synth_chunk_calls": 0,
            "raw_events": 0,
        }
        first_mark = {
            "raw_event": None,
            "control_decision": None,
            "state_s": None,
            "speaking_token": None,
            "audio_chunk": None,
            "t2w_submit": None,
            "pcm_out": None,
        }

        def _mark_first(name):
            if first_mark.get(name) is None:
                first_mark[name] = time.perf_counter()

        def _lat_value(name):
            ts = first_mark.get(name)
            if ts is None:
                return None
            return ts - t_stream_start_perf

        def _trace_now():
            now_perf = time.perf_counter()
            return {
                "timestamp_epoch_ms": int(time.time() * 1000),
                "rel_ms": round((now_perf - t_stream_start_perf) * 1000.0, 3),
            }

        def _normalize_reason(reason_value):
            if reason_value is None:
                return "unknown"
            reason_text = str(reason_value).strip()
            return reason_text if reason_text else "unknown"

        def _build_event_trace(kind_value, trace_ts, resumed=False):
            return {
                "event_index": int(len(round_trace["events"]) + 1),
                "kind": str(kind_value),
                "resumed": bool(resumed),
                "start_timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                "start_rel_ms": trace_ts["rel_ms"],
                "end_timestamp_epoch_ms": None,
                "end_rel_ms": None,
                "text": "",
                # Keep per-event first-token timestamps always present.
                # If not observed, they remain equal to event start.
                "first_text_token_timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                "first_text_token_rel_ms": trace_ts["rel_ms"],
                "first_stoken_timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                "first_stoken_rel_ms": trace_ts["rel_ms"],
                "first_t2w_submit_timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                "first_t2w_submit_rel_ms": trace_ts["rel_ms"],
                "first_pcm_out_timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                "first_pcm_out_rel_ms": trace_ts["rel_ms"],
                "first_text_token_latency_ms": None,
                "first_stoken_latency_ms": None,
                "first_t2w_submit_latency_ms": None,
                "first_pcm_out_latency_ms": None,
                "first_text_token_observed": False,
                "first_stoken_observed": False,
                "first_t2w_submit_observed": False,
                "first_pcm_out_observed": False,
                "audio_chunks": [],
                "state_changes": [],
            }

        # --- Set up streaming decoder ---
        decoder_chunk_size = int(tts_chunk_size) if tts_chunk_size is not None else TTS_CHUNK_SIZE
        vocoder_hop_size = TTS_VOCODER_HOP_SIZE
        decoder_end_on_generation_complete = stream_session is None
        decoder = StreamingDecoder(
            generator.tokenizer,
            generator,
            tts_chunk_size=decoder_chunk_size,
            end_event_on_generation_complete=decoder_end_on_generation_complete,
        )

        if stream_session is not None and hasattr(stream_session, "get_listening_state"):
            current_state = str(stream_session.get_listening_state()).lower()
        else:
            current_state = str(initial_listening_state).lower() if isinstance(initial_listening_state, str) else "l"
        if current_state not in {"l", "s", "b"}:
            current_state = "l"
        current_text = ""
        current_event_id = None
        current_text_seq = 0
        current_stoken_count = 0
        current_chunk_idx = 0
        latest_sl_prob = None
        latest_ss_prob = None
        latest_ks_prob = None
        latest_kl_prob = None
        latest_bc_prob = None
        finished_events = []
        all_audio_pcm = bytearray()
        pending_audio_np = []
        event_audio_tokens = []

        # Audio-token / RTF diagnostics.
        t_first_audio_token = None
        total_token2wav_sec = 0.0
        last_audio_debug = ""
        synth_stats = {"total_time": 0.0}

        pre_lookahead_len = get_token2wav_pre_lookahead_len()
        logger.info(
            "TTS chunk config: decoder_flush=%d, vocoder_hop=%d, lookahead=%d",
            int(decoder_chunk_size),
            int(vocoder_hop_size),
            int(pre_lookahead_len),
        )
        event_vocoder_raw_tokens = 0
        event_vocoder_unique_tokens = 0
        event_vocoder_calls = 0
        event_pcm_bytes_start = 0
        event_start_perf = None

        owns_tts_bridge = False
        active_tts_bridge = tts_bridge
        if active_tts_bridge is None:
            active_tts_bridge = RealtimeTTSPool(
                prompt_wav_path=prompt_wav_path,
                vocoder_hop_size=vocoder_hop_size,
                pre_lookahead_len=pre_lookahead_len,
                worker_name="stepaudio_tts_worker_round",
                persistent_mode=False,
            )
            owns_tts_bridge = True
        else:
            if str(getattr(active_tts_bridge, "prompt_wav_path", "")) != str(prompt_wav_path):
                raise RuntimeError(
                    "TTS bridge prompt voice mismatch: "
                    f"bridge={getattr(active_tts_bridge, 'prompt_wav_path', '')}, "
                    f"current={prompt_wav_path}"
                )

        def _drain_tts_output(block=False, timeout=0.0, until_event_stats=False):
            nonlocal all_audio_pcm, pending_audio_np
            nonlocal event_vocoder_raw_tokens, event_vocoder_unique_tokens, event_vocoder_calls
            drained_messages, event_stats_msg = active_tts_bridge.drain(
                block=bool(block),
                timeout=float(timeout),
                until_event_stats=bool(until_event_stats),
            )
            for msg in drained_messages:
                mtype = msg.get("type")
                if mtype == "error":
                    raise RuntimeError(f"TTS worker failed:\n{msg.get('detail')}")
                if mtype == "pcm":
                    pcm_bytes = msg.get("pcm_bytes", b"")
                    if pcm_bytes:
                        try:
                            synth_start_ms = msg.get("synth_start_epoch_ms")
                            synth_end_ms = msg.get("synth_end_epoch_ms")
                            if synth_start_ms is not None and synth_end_ms is not None:
                                round_trace.setdefault("timeline_spans", []).append({
                                    "name": "token2wav",
                                    "start_epoch_ms": int(float(synth_start_ms)),
                                    "end_epoch_ms": int(float(synth_end_ms)),
                                    "duration_ms": round(float(msg.get("synth_duration_ms", msg.get("synth_sec", 0.0) * 1000.0)), 3),
                                    "tokens": int(msg.get("tokens", 0)),
                                    "advance_tokens": int(msg.get("advance_tokens", 0)),
                                    "is_last": bool(msg.get("is_last", False)),
                                    "pcm_bytes": int(len(pcm_bytes)),
                                    "backend": str(msg.get("t2w_backend", "local")),
                                    "remote_recv_epoch_ms": msg.get("remote_recv_epoch_ms"),
                                    "remote_return_epoch_ms": msg.get("remote_return_epoch_ms"),
                                    "remote_queue_wait_ms": msg.get("remote_queue_wait_ms"),
                                    "remote_roundtrip_ms": msg.get("remote_roundtrip_ms"),
                                })
                        except Exception:
                            logger.debug("Failed to append token2wav timeline span", exc_info=True)
                        all_audio_pcm.extend(pcm_bytes)
                        # When realtime session uses direct PCM callback emit,
                        # do not re-buffer into local round audio path unless
                        # callback failed and this message is an explicit fallback.
                        should_buffer_local = (
                            not bool(getattr(active_tts_bridge, "pcm_callback_enabled", False))
                            or bool(msg.get("callback_fallback", False))
                        )
                        if should_buffer_local:
                            pending_audio_np.append(pcm_to_numpy(pcm_bytes))
                        _mark_first("pcm_out")
                        if isinstance(current_event_trace, dict) and not bool(current_event_trace.get("first_pcm_out_observed", False)):
                            trace_ts = _trace_now()
                            current_event_trace["first_pcm_out_timestamp_epoch_ms"] = trace_ts["timestamp_epoch_ms"]
                            current_event_trace["first_pcm_out_rel_ms"] = trace_ts["rel_ms"]
                            if current_event_trace.get("start_rel_ms") is not None:
                                current_event_trace["first_pcm_out_latency_ms"] = round(
                                    float(trace_ts["rel_ms"]) - float(current_event_trace["start_rel_ms"]),
                                    3,
                                )
                            current_event_trace["first_pcm_out_observed"] = True
                    latency_stats["emit_tts_sec"] += float(msg.get("synth_sec", 0.0))
                    latency_stats["emit_tts_calls"] += 1
                    if profile_latency and (
                        latency_stats["emit_tts_calls"] % profile_every == 0 or bool(msg.get("is_last", False))
                    ):
                        logger.info(
                            "[LAT][VOC] call=%d tokens=%d advance=%d is_last=%s synth=%.4fs pcm_bytes=%d pending=%d",
                            int(latency_stats["emit_tts_calls"]),
                            int(msg.get("tokens", 0)),
                            int(msg.get("advance_tokens", 0)),
                            bool(msg.get("is_last", False)),
                            float(msg.get("synth_sec", 0.0)),
                            int(len(pcm_bytes)),
                            int(len(pending_audio_np)),
                        )
                elif mtype == "synth_chunk_done":
                    latency_stats["audio_chunk_sec"] += float(msg.get("sec", 0.0))
                    latency_stats["synth_chunk_calls"] += 1
                elif mtype == "event_stats":
                    event_vocoder_raw_tokens = int(msg.get("vocoder_raw_tokens", 0))
                    event_vocoder_unique_tokens = int(msg.get("vocoder_unique_tokens", 0))
                    event_vocoder_calls = int(msg.get("vocoder_calls", 0))
                    synth_stats["total_time"] = float(msg.get("synth_total_time", 0.0))
                    event_stats_msg = msg
            return event_stats_msg

        def _wait_event_stats(max_wait_sec=30.0):
            deadline = time.perf_counter() + max_wait_sec
            while time.perf_counter() < deadline:
                msg = _drain_tts_output(block=True, timeout=0.1, until_event_stats=True)
                if msg is not None:
                    return msg
            return None

        # Resume decoding/synthesis context for cross-round realtime continuation.
        # Without this, rounds that start in speaking/backchannel may miss event_start.
        decoder._state = current_state
        if current_state in {"s", "b"}:
            resumed_kind = "response" if current_state == "s" else "backchannel"
            resumed_event = decoder._start_event(resumed_kind)
            current_event_id = resumed_event.get("event_id")
            trace_ts = _trace_now()
            current_event_trace = _build_event_trace(resumed_kind, trace_ts, resumed=True)
            current_event_trace["event_id"] = current_event_id
            round_trace["events"].append(current_event_trace)
            logger.info(f"  EVENT START: {resumed_kind} (resumed)")
            event_start_perf = time.perf_counter()
            current_text = ""
            current_stoken_count = 0
            event_audio_tokens = []
            event_vocoder_raw_tokens = 0
            event_vocoder_unique_tokens = 0
            event_vocoder_calls = 0
            event_pcm_bytes_start = len(all_audio_pcm)
            t_first_audio_token = None
            synth_stats["total_time"] = 0.0
            if not (bool(active_tts_bridge.persistent_mode) and bool(active_tts_bridge.event_active)):
                active_tts_bridge.submit_event_start()
            last_audio_debug = f"[Audio Token] resumed {resumed_kind} event; waiting for tokens..."
            chatbot_msgs.append({
                "role": "assistant",
                "content": f"**[{resumed_kind}]** generating...(resumed)"
            })

        def _emit_control_head_early_event(control_event: dict) -> None:
            if not isinstance(control_event, dict):
                return
            event_type = str(control_event.get("type") or "")
            if event_type == "control_head_pending":
                if CONTROL_EARLY_DEBUG:
                    logger.info(
                        "Control-head pending state_change %s->%s chunk=%s step=%s reason=%s",
                        control_event.get("from"),
                        control_event.get("to"),
                        control_event.get("chunk"),
                        control_event.get("step"),
                        control_event.get("reason"),
                    )
                return
            if event_type != "control_head_state_change":
                return
            if not (
                CONTROL_EARLY_EXIT_ENABLED
                and CONTROL_EARLY_STATE_SSE
                and callable(direct_state_callback)
            ):
                return
            trace_ts = _trace_now()
            payload = dict(control_event)
            payload.update(
                {
                    "type": "state_change",
                    "source": "model_control_head_early",
                    "trace_source": "control_head",
                    "server_state_commit_epoch_ms": trace_ts["timestamp_epoch_ms"],
                }
            )
            payload.setdefault("chunk_idx", payload.get("chunk", payload.get("pos")))
            payload.setdefault("chunk", payload.get("chunk_idx", payload.get("pos")))
            try:
                direct_state_callback(payload)
            except Exception:
                logger.exception("direct_state_callback failed on control-head early event")

        # --- Consume the streaming generator ---
        if stream_session is not None and hasattr(stream_session, "advance_stream"):
            raw_gen = stream_session.advance_stream(
                audio_np,
                emit_generation_complete=False,
                audio_is_incremental=bool(stream_audio_incremental),
                flush_audio_tail=bool(stream_flush_audio_tail),
                control_early_callback=_emit_control_head_early_event,
            )
        else:
            raw_gen = generator.full_chunk_stream_generation(
                audio_np,
                prefix=prefix,
                initial_listening_state=initial_listening_state,
                start_speak_token_factor=start_speak_factor,
                start_listen_token_factor=start_listen_factor,
                end_speak_token_factor=end_speak_factor,
            )

        generation_result = None
        yield_counter = 0
        t_wait_anchor = time.perf_counter()

        for raw_event in raw_gen:
            t_event_arrive = time.perf_counter()
            wait_sec = t_event_arrive - t_wait_anchor
            latency_stats["raw_wait_sec"] += wait_sec
            latency_stats["raw_events"] += 1
            _mark_first("raw_event")
            raw_type = raw_event.get("type", "?")
            if raw_type == "timeline_span":
                try:
                    span = dict(raw_event)
                    span.pop("type", None)
                    span.setdefault("raw_event_index", int(latency_stats["raw_events"]))
                    round_trace.setdefault("timeline_spans", []).append(span)
                except Exception:
                    logger.debug("Failed to append model timeline span", exc_info=True)
                _drain_tts_output()
                latency_stats["raw_handle_sec"] += time.perf_counter() - t_event_arrive
                t_wait_anchor = time.perf_counter()
                continue
            raw_sl = raw_event.get("sl_prob", None)
            raw_ss = raw_event.get("ss_prob", None)
            raw_ks = raw_event.get("ks_prob", None)
            raw_kl = raw_event.get("kl_prob", None)
            raw_bc = raw_event.get("bc_prob", None)
            if raw_sl is not None:
                try:
                    latest_sl_prob = float(raw_sl)
                except (TypeError, ValueError):
                    pass
            if raw_ss is not None:
                try:
                    latest_ss_prob = float(raw_ss)
                except (TypeError, ValueError):
                    pass
            if raw_ks is not None:
                try:
                    latest_ks_prob = float(raw_ks)
                except (TypeError, ValueError):
                    pass
            if raw_kl is not None:
                try:
                    latest_kl_prob = float(raw_kl)
                except (TypeError, ValueError):
                    pass
            if raw_bc is not None:
                try:
                    latest_bc_prob = float(raw_bc)
                except (TypeError, ValueError):
                    pass
            if raw_sl is not None or raw_ss is not None or raw_ks is not None or raw_kl is not None or raw_bc is not None:
                trace_ts = _trace_now()
                prob_space = "unknown"
                if raw_ks is not None:
                    prob_space = "speaking"
                elif raw_kl is not None or raw_bc is not None:
                    prob_space = "listening"
                elif str(raw_type) in {"speaking_token", "speaking_done"}:
                    prob_space = "speaking"
                elif str(current_state) in {"l"}:
                    prob_space = "listening"
                elif str(current_state) in {"s", "b"}:
                    prob_space = "speaking"
                control_prob_entry = {
                    "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                    "rel_ms": trace_ts["rel_ms"],
                    "raw_type": str(raw_type),
                    "chunk_idx": int(raw_event.get("chunk", raw_event.get("chunk_idx", current_chunk_idx)) or current_chunk_idx),
                    "control_token": raw_event.get("token"),
                    "sl_prob": latest_sl_prob if latest_sl_prob is not None else None,
                    "ss_prob": latest_ss_prob if latest_ss_prob is not None else None,
                    "ks_prob": latest_ks_prob if latest_ks_prob is not None else None,
                    "kl_prob": latest_kl_prob if latest_kl_prob is not None else None,
                    "bc_prob": latest_bc_prob if latest_bc_prob is not None else None,
                    "prob_space": prob_space,
                    "state_before_decode": str(current_state),
                }
                round_trace["control_prob_points"].append(control_prob_entry)
            if raw_type == "control_decision":
                logger.info(f"  [RAW] control_decision: token={raw_event.get('token')}")
            elif raw_type == "state_change":
                reason = _normalize_reason(raw_event.get("reason"))
                trace_ts = _trace_now()
                state_pos = _state_change_position_int(raw_event, current_chunk_idx)
                state_change_entry = {
                    "source": "raw",
                    "from": str(raw_event.get("from", "?")),
                    "to": str(raw_event.get("to", "?")),
                    "reason": reason,
                    "chunk": int(state_pos),
                    "chunk_idx": int(state_pos),
                    "pos": int(state_pos),
                    "decode_chunk_idx": int(current_chunk_idx),
                    "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                    "rel_ms": trace_ts["rel_ms"],
                    "early_exit": bool(raw_event.get("early_exit", False)),
                    "interrupt": bool(raw_event.get("interrupt", False)),
                    "interrupt_reason": raw_event.get("interrupt_reason"),
                    "sl_prob": raw_event.get("sl_prob"),
                    "ss_prob": raw_event.get("ss_prob"),
                    "ks_prob": raw_event.get("ks_prob"),
                }
                round_trace["state_changes_raw"].append(state_change_entry)
                if isinstance(current_event_trace, dict):
                    current_event_trace.setdefault("state_changes", []).append(dict(state_change_entry))
                if (
                    CONTROL_EARLY_EXIT_ENABLED
                    and CONTROL_EARLY_STATE_SSE
                    and bool(state_change_entry.get("early_exit", False))
                    and callable(direct_state_callback)
                ):
                    try:
                        direct_payload = dict(state_change_entry)
                        direct_payload.update(
                            {
                                "type": "state_change",
                                "source": "model_early_exit",
                                "trace_source": "raw",
                                "chunk": state_change_entry.get("chunk_idx"),
                                "chunk_idx": state_change_entry.get("chunk_idx"),
                                "server_state_commit_epoch_ms": trace_ts["timestamp_epoch_ms"],
                            }
                        )
                        direct_state_callback(direct_payload)
                    except Exception:
                        logger.exception("direct_state_callback failed on early state_change")
                logger.info(f"  [RAW] state_change: {raw_event.get('from')} -> {raw_event.get('to')} reason={reason}")
            elif raw_type == "speaking_done":
                logger.info(f"  [RAW] speaking_done: reason={raw_event.get('reason')}")
            elif raw_type == "speaking_token":
                _mark_first("speaking_token")
                pass
            elif raw_type in ("chunk_start", "chunk_end"):
                logger.info(f"  [RAW] {raw_type}: idx={raw_event.get('chunk_idx')}, pos={raw_event.get('chunk_pos', '?')}")
            else:
                logger.info(f"  [RAW] {raw_type}")

            t_decoder_start = time.perf_counter()
            decoded_events = decoder.feed(raw_event)
            decoder_sec = time.perf_counter() - t_decoder_start
            latency_stats["decoder_sec"] += decoder_sec

            for ev in decoded_events:
                evtype = ev.get("type")

                if evtype == "chunk_start":
                    current_chunk_idx = ev["chunk_idx"]

                elif evtype == "control_decision":
                    _mark_first("control_decision")
                    ctrl_token = ev.get("token", "?")
                    token_name = "KL" if ctrl_token == generator.kl_token_id else \
                                 "SS" if ctrl_token == generator.ss_token_id else \
                                 "BC" if ctrl_token == generator.bc_token_id else \
                                 f"unknown({ctrl_token})"
                    logger.info(f"  Chunk {current_chunk_idx}: control={token_name} ({ctrl_token})")

                elif evtype == "state_change":
                    from_s = ev.get("from", "?")
                    to_s = ev.get("to", "?")
                    reason = _normalize_reason(ev.get("reason"))
                    trace_ts = _trace_now()
                    state_pos = _state_change_position_int(ev, current_chunk_idx)
                    state_change_entry = {
                        "source": "decoded",
                        "from": str(from_s),
                        "to": str(to_s),
                        "reason": reason,
                        "chunk": int(state_pos),
                        "chunk_idx": int(state_pos),
                        "pos": int(state_pos),
                        "decode_chunk_idx": int(current_chunk_idx),
                        "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                        "rel_ms": trace_ts["rel_ms"],
                        "early_exit": bool(ev.get("early_exit", False)),
                        "interrupt": bool(ev.get("interrupt", False)),
                        "interrupt_reason": ev.get("interrupt_reason"),
                        "sl_prob": ev.get("sl_prob"),
                        "ss_prob": ev.get("ss_prob"),
                        "ks_prob": ev.get("ks_prob"),
                    }
                    round_trace["state_changes"].append(state_change_entry)
                    if isinstance(current_event_trace, dict):
                        current_event_trace.setdefault("state_changes", []).append(
                            {
                                "from": str(from_s),
                                "to": str(to_s),
                                "reason": reason,
                                "chunk": int(state_pos),
                                "chunk_idx": int(state_pos),
                                "pos": int(state_pos),
                                "decode_chunk_idx": int(current_chunk_idx),
                                "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                                "rel_ms": trace_ts["rel_ms"],
                                "early_exit": bool(ev.get("early_exit", False)),
                                "interrupt": bool(ev.get("interrupt", False)),
                                "interrupt_reason": ev.get("interrupt_reason"),
                                "sl_prob": ev.get("sl_prob"),
                                "ss_prob": ev.get("ss_prob"),
                                "ks_prob": ev.get("ks_prob"),
                            }
                        )
                    label = _get_state_label(to_s.lower())
                    if str(to_s).upper() == "S":
                        _mark_first("state_s")
                    logger.info(f"  STATE CHANGE: {from_s} -> {to_s} (reason={reason})")
                    chatbot_msgs.append({
                        "role": "assistant",
                        "content": (
                            f"State: {from_s} -> {to_s} {label} (chunk {current_chunk_idx})"
                            + f" reason={reason}"
                        )
                    })
                    current_state = to_s.lower()

                elif evtype == "event_start":
                    kind = ev["event_kind"]
                    current_event_id = ev.get("event_id")
                    trace_ts = _trace_now()
                    current_event_trace = _build_event_trace(kind, trace_ts, resumed=False)
                    current_event_trace["event_id"] = current_event_id
                    round_trace["events"].append(current_event_trace)
                    logger.info(f"  EVENT START: {kind}")
                    event_start_perf = time.perf_counter()
                    current_text = ""
                    current_text_seq = 0
                    current_stoken_count = 0
                    event_audio_tokens = []
                    event_vocoder_raw_tokens = 0
                    event_vocoder_unique_tokens = 0
                    event_vocoder_calls = 0
                    event_pcm_bytes_start = len(all_audio_pcm)
                    t_first_audio_token = None
                    synth_stats["total_time"] = 0.0
                    active_tts_bridge.submit_event_start()
                    last_audio_debug = f"[Audio Token] {kind} event started; waiting for tokens..."
                    chatbot_msgs.append({
                        "role": "assistant",
                        "content": f"**[{kind}]** generating..."
                    })

                elif evtype == "text_delta":
                    text_delta = ev.get("delta", ev.get("text", ""))
                    text_snapshot = ev.get("snapshot")
                    if isinstance(text_snapshot, str):
                        current_text = text_snapshot
                    else:
                        current_text += text_delta
                    try:
                        current_text_seq = int(ev.get("seq", current_text_seq + 1))
                    except (TypeError, ValueError):
                        current_text_seq += 1
                    if ev.get("event_id") is not None:
                        current_event_id = ev.get("event_id")
                    if callable(direct_text_callback):
                        try:
                            direct_text_callback({
                                "type": "text_delta",
                                "event_id": current_event_id,
                                "event_kind": ev.get("event_kind", current_event_trace.get("kind") if isinstance(current_event_trace, dict) else "response"),
                                "delta": text_delta if isinstance(text_delta, str) else "",
                                "snapshot": current_text,
                                "seq": int(current_text_seq),
                                "is_final": False,
                                "resumed": bool(current_event_trace.get("resumed", False)) if isinstance(current_event_trace, dict) else False,
                            })
                        except Exception:
                            logger.exception("direct_text_callback failed on text_delta")
                    if isinstance(current_event_trace, dict) and isinstance(text_delta, str) and text_delta:
                        if not bool(current_event_trace.get("first_text_token_observed", False)):
                            trace_ts = _trace_now()
                            current_event_trace["first_text_token_timestamp_epoch_ms"] = trace_ts["timestamp_epoch_ms"]
                            current_event_trace["first_text_token_rel_ms"] = trace_ts["rel_ms"]
                            if current_event_trace.get("start_rel_ms") is not None:
                                current_event_trace["first_text_token_latency_ms"] = round(
                                    float(trace_ts["rel_ms"]) - float(current_event_trace["start_rel_ms"]), 3
                                )
                            round_trace["token_timeline"].append(
                                {
                                    "type": "first_text_token",
                                    "event_index": current_event_trace.get("event_index"),
                                    "kind": current_event_trace.get("kind"),
                                    "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                                    "rel_ms": trace_ts["rel_ms"],
                                    "chunk_idx": int(current_chunk_idx),
                                    "text_delta": text_delta,
                                }
                            )
                        current_event_trace["first_text_token_observed"] = True
                        if (
                            current_event_trace.get("first_text_token_latency_ms") is None
                            and current_event_trace.get("start_rel_ms") is not None
                            and current_event_trace.get("first_text_token_rel_ms") is not None
                        ):
                            current_event_trace["first_text_token_latency_ms"] = round(
                                float(current_event_trace["first_text_token_rel_ms"]) - float(current_event_trace["start_rel_ms"]),
                                3,
                            )
                        current_event_trace["text"] = current_text
                        current_event_trace["event_id"] = current_event_id
                        current_event_trace["text_seq"] = int(current_text_seq)
                    if chatbot_msgs and chatbot_msgs[-1]["role"] == "assistant":
                        chatbot_msgs[-1]["content"] = f"**Text**: {current_text}"

                elif evtype == "audio_chunk":
                    _mark_first("audio_chunk")
                    stoken_ids_raw = ev["stoken_ids"]
                    batch_size = len(stoken_ids_raw)
                    trace_ts = _trace_now()
                    if t_first_audio_token is None:
                        t_first_audio_token = time.time()
                    if isinstance(current_event_trace, dict):
                        if not bool(current_event_trace.get("first_stoken_observed", False)):
                            current_event_trace["first_stoken_timestamp_epoch_ms"] = trace_ts["timestamp_epoch_ms"]
                            current_event_trace["first_stoken_rel_ms"] = trace_ts["rel_ms"]
                            if current_event_trace.get("start_rel_ms") is not None:
                                current_event_trace["first_stoken_latency_ms"] = round(
                                    float(trace_ts["rel_ms"]) - float(current_event_trace["start_rel_ms"]), 3
                                )
                            round_trace["token_timeline"].append(
                                {
                                    "type": "first_stoken",
                                    "event_index": current_event_trace.get("event_index"),
                                    "kind": current_event_trace.get("kind"),
                                    "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                                    "rel_ms": trace_ts["rel_ms"],
                                    "chunk_idx": int(current_chunk_idx),
                                    "valid_audio_tokens": int(batch_size),
                                }
                            )
                        current_event_trace["first_stoken_observed"] = True
                        if (
                            current_event_trace.get("first_stoken_latency_ms") is None
                            and current_event_trace.get("start_rel_ms") is not None
                            and current_event_trace.get("first_stoken_rel_ms") is not None
                        ):
                            current_event_trace["first_stoken_latency_ms"] = round(
                                float(current_event_trace["first_stoken_rel_ms"]) - float(current_event_trace["start_rel_ms"]),
                                3,
                            )
                        current_event_trace.setdefault("audio_chunks", []).append(
                            {
                                "timestamp_epoch_ms": trace_ts["timestamp_epoch_ms"],
                                "rel_ms": trace_ts["rel_ms"],
                                "chunk_idx": int(current_chunk_idx),
                                "valid_audio_tokens": int(batch_size),
                                "total_stream_audio_tokens_after_chunk": int(current_stoken_count + batch_size),
                            }
                        )
                    event_audio_tokens.extend(stoken_ids_raw)
                    current_stoken_count += batch_size
                    logger.info(f"  AUDIO CHUNK: {batch_size} stokens, total={current_stoken_count}")
                    _mark_first("t2w_submit")
                    if isinstance(current_event_trace, dict) and not bool(current_event_trace.get("first_t2w_submit_observed", False)):
                        current_event_trace["first_t2w_submit_timestamp_epoch_ms"] = trace_ts["timestamp_epoch_ms"]
                        current_event_trace["first_t2w_submit_rel_ms"] = trace_ts["rel_ms"]
                        if current_event_trace.get("start_rel_ms") is not None:
                            current_event_trace["first_t2w_submit_latency_ms"] = round(
                                float(trace_ts["rel_ms"]) - float(current_event_trace["start_rel_ms"]),
                                3,
                            )
                        current_event_trace["first_t2w_submit_observed"] = True
                    active_tts_bridge.submit_audio_chunk(list(stoken_ids_raw))
                    _drain_tts_output()
                    # Update RTF diagnostics.
                    total_token2wav_sec = synth_stats["total_time"]
                    elapsed = time.time() - t_start
                    token_dur_sec = current_stoken_count / TOKENS_PER_SECOND if current_stoken_count else 0
                    rft_token = (time.time() - t_first_audio_token) / (token_dur_sec or 1e-6) if current_stoken_count else 0
                    rft_wav = total_token2wav_sec / (token_dur_sec or 1e-6) if current_stoken_count else 0
                    last_audio_debug = (
                        f"[Audio Token] batch: +{batch_size} | total: {current_stoken_count} tokens\n"
                        f"[RTF] token generation: {rft_token:.3f} | Token2Wav: {rft_wav:.3f} "
                        "(RTF > 1 is slower than realtime)\n"
                        f"[Timing] inference: {elapsed:.2f}s | Token2Wav total: {total_token2wav_sec:.2f}s "
                        f"| equivalent audio: {token_dur_sec:.2f}s"
                    )

                elif evtype == "event_end":
                    kind = ev["event_kind"]
                    if ev.get("event_id") is not None:
                        current_event_id = ev.get("event_id")
                    full_text = ev.get("snapshot") if isinstance(ev.get("snapshot"), str) else ev.get("text", "")
                    if not full_text and current_text:
                        full_text = current_text
                    try:
                        current_text_seq = int(ev.get("seq", current_text_seq + 1))
                    except (TypeError, ValueError):
                        current_text_seq += 1
                    stoken_list = ev["stoken_ids"]
                    event_interrupted = bool(ev.get("interrupt", False))
                    event_interrupt_reason = ev.get("interrupt_reason")
                    trace_ts = _trace_now()
                    target_event_trace = current_event_trace
                    if not isinstance(target_event_trace, dict):
                        # Fallback: locate the latest unfinished event with same kind.
                        for _evt in reversed(round_trace.get("events", [])):
                            if _evt.get("kind") == str(kind) and _evt.get("end_timestamp_epoch_ms") is None:
                                target_event_trace = _evt
                                break
                    if isinstance(target_event_trace, dict):
                        target_event_trace["end_timestamp_epoch_ms"] = trace_ts["timestamp_epoch_ms"]
                        target_event_trace["end_rel_ms"] = trace_ts["rel_ms"]
                        target_event_trace["text"] = full_text
                        target_event_trace["event_id"] = current_event_id
                        target_event_trace["text_seq"] = int(current_text_seq)
                        target_event_trace["model_audio_tokens"] = int(len(stoken_list))
                        target_event_trace["stream_audio_tokens"] = int(current_stoken_count)
                        target_event_trace["effective_audio_tokens"] = int(event_vocoder_unique_tokens)
                        target_event_trace["vocoder_raw_tokens"] = int(event_vocoder_raw_tokens)
                        target_event_trace["vocoder_calls"] = int(event_vocoder_calls)
                        target_event_trace["interrupt"] = bool(event_interrupted)
                        target_event_trace["interrupt_reason"] = event_interrupt_reason
                        if target_event_trace.get("start_rel_ms") is not None:
                            target_event_trace["event_duration_ms"] = round(
                                float(trace_ts["rel_ms"]) - float(target_event_trace["start_rel_ms"]), 3
                            )
                    if callable(direct_text_callback):
                        try:
                            direct_text_callback({
                                "type": "event_end",
                                "event_id": current_event_id,
                                "event_kind": kind,
                                "delta": "",
                                "snapshot": full_text,
                                "seq": int(current_text_seq),
                                "is_final": True,
                                "interrupt": bool(event_interrupted),
                                "interrupt_reason": event_interrupt_reason,
                                "resumed": bool(target_event_trace.get("resumed", False)) if isinstance(target_event_trace, dict) else False,
                            })
                        except Exception:
                            logger.exception("direct_text_callback failed on event_end")
                    logger.info(
                        "  EVENT END: %s, text='%s', audio_tokens=%d, interrupt=%s",
                        kind,
                        full_text[:60],
                        len(stoken_list),
                        bool(event_interrupted),
                    )
                    if not event_interrupted:
                        # Flush remaining tokens at event end so tail audio is emitted.
                        active_tts_bridge.submit_event_end(force_flush=True)
                        if _wait_event_stats(max_wait_sec=30.0) is None:
                            raise RuntimeError("Timed out waiting for TTS worker event flush")
                    _drain_tts_output()
                    event_pcm_bytes = max(0, len(all_audio_pcm) - event_pcm_bytes_start)
                    event_pcm_sec = event_pcm_bytes / (TTS_SAMPLE_RATE * 2)
                    model_tokens = len(stoken_list)
                    stream_tokens = current_stoken_count
                    token_delta = model_tokens - event_vocoder_unique_tokens
                    logger.info(
                        "  [EVENT TTS STATS] kind=%s | model_stokens=%d | stream_stokens=%d | "
                        "vocoder_unique_tokens=%d | vocoder_raw_tokens=%d | vocoder_calls=%d | "
                        "delta(model-unique)=%d | pcm=%.3fs (%d bytes)",
                        kind,
                        model_tokens,
                        stream_tokens,
                        event_vocoder_unique_tokens,
                        event_vocoder_raw_tokens,
                        event_vocoder_calls,
                        token_delta,
                        event_pcm_sec,
                        event_pcm_bytes,
                    )
                    if isinstance(target_event_trace, dict):
                        target_event_trace["stream_audio_tokens"] = int(stream_tokens)
                        target_event_trace["effective_audio_tokens"] = int(event_vocoder_unique_tokens)
                        target_event_trace["vocoder_raw_tokens"] = int(event_vocoder_raw_tokens)
                        target_event_trace["vocoder_calls"] = int(event_vocoder_calls)
                        target_event_trace["vocoder_output_seconds"] = round(float(event_pcm_sec), 3)
                        target_event_trace["token_delta_model_vs_vocoder_unique"] = int(token_delta)
                    if profile_latency:
                        event_elapsed = (time.perf_counter() - event_start_perf) if event_start_perf is not None else None
                        logger.info(
                            "[LAT][EVENT] kind=%s elapsed=%s first_control=%s first_state_s=%s first_speaking=%s first_audio_chunk=%s first_pcm=%s",
                            kind,
                            f"{event_elapsed:.4f}s" if isinstance(event_elapsed, float) else "None",
                            f"{_lat_value('control_decision'):.4f}s" if _lat_value("control_decision") is not None else "None",
                            f"{_lat_value('state_s'):.4f}s" if _lat_value("state_s") is not None else "None",
                            f"{_lat_value('speaking_token'):.4f}s" if _lat_value("speaking_token") is not None else "None",
                            f"{_lat_value('audio_chunk'):.4f}s" if _lat_value("audio_chunk") is not None else "None",
                            f"{_lat_value('pcm_out'):.4f}s" if _lat_value("pcm_out") is not None else "None",
                        )
                    # Do not warn on token count mismatches here. In realtime
                    # persistent TTS mode, StreamingDecoder may report only the
                    # current resumed round while the Token2Wav worker keeps
                    # cumulative counters for the open speech event. Interrupted
                    # events also skip the final TTS flush, so this comparison is
                    # useful as a trace field but not as an error signal.
                    finished_events.append({
                        "type": kind,
                        "event_id": current_event_id,
                        "text": full_text,
                        "audio_tokens": len(stoken_list),
                        "stream_audio_tokens": stream_tokens,
                        "vocoder_unique_tokens": event_vocoder_unique_tokens,
                        "vocoder_raw_tokens": event_vocoder_raw_tokens,
                        "vocoder_calls": event_vocoder_calls,
                        "vocoder_output_seconds": round(event_pcm_sec, 3),
                        "token_delta_model_vs_vocoder_unique": token_delta,
                    })
                    if chatbot_msgs and chatbot_msgs[-1]["role"] == "assistant":
                        chatbot_msgs[-1]["content"] = (
                            f"**[{kind}]** {full_text if full_text else '(no text)'}\n"
                            f"Audio: {len(stoken_list)} tokens"
                        )
                    current_event_trace = None
                    current_event_id = None
                    current_text_seq = 0

                elif evtype == "generation_complete":
                    generation_result = ev
                    round_trace["generation_complete"] = _trace_now()
                    if isinstance(round_trace["generation_complete"], dict):
                        rel_ms = round_trace["generation_complete"].get("rel_ms")
                        try:
                            round_trace["generation_complete"]["rel_sec"] = round(float(rel_ms) / 1000.0, 6)
                        except (TypeError, ValueError):
                            round_trace["generation_complete"]["rel_sec"] = None
                    logger.info(f"  GENERATION COMPLETE")

            _drain_tts_output()
            handle_sec = time.perf_counter() - t_event_arrive
            latency_stats["raw_handle_sec"] += handle_sec
            if profile_latency and (
                latency_stats["raw_events"] == 1
                or (latency_stats["raw_events"] % profile_every == 0)
            ):
                logger.info(
                    "[LAT][RAW] idx=%d type=%s wait=%.4fs decode=%.4fs handle=%.4fs",
                    int(latency_stats["raw_events"]),
                    raw_type,
                    wait_sec,
                    decoder_sec,
                    handle_sec,
                )
            yield_counter += 1
            audio_output = None
            if pending_audio_np:
                audio_output = (TTS_SAMPLE_RATE, np.concatenate(pending_audio_np))
                pending_audio_np.clear()
            if audio_output is not None or yield_counter % 5 == 0:
                sl_disp = "None" if latest_sl_prob is None else f"{float(latest_sl_prob):.4f}"
                ss_disp = "None" if latest_ss_prob is None else f"{float(latest_ss_prob):.4f}"
                yield (
                    chatbot_msgs,
                    audio_output,
                    f"Streaming: chunk {current_chunk_idx}, state={current_state}, "
                    f"text_len={len(current_text)}, events={len(finished_events)}, "
                    f"S-L={sl_disp}, S-S={ss_disp}",
                    "",
                    last_audio_debug if last_audio_debug else "[Audio Token / RTF] no audio token for this event yet",
                )

            t_wait_anchor = time.perf_counter()

        if stream_session is not None and bool(flush_tts_on_round_end) and not bool(active_tts_bridge.persistent_mode):
            # Realtime incremental mode keeps events open across rounds; flush
            # vocoder buffers to avoid dropping tail tokens when this round ends.
            active_tts_bridge.submit_event_end(force_flush=True)
            _wait_event_stats(max_wait_sec=5.0)
            _drain_tts_output()

        if owns_tts_bridge:
            drained = active_tts_bridge.stop(force_flush=False)
            for msg in drained:
                if isinstance(msg, dict) and msg.get("type") == "error":
                    raise RuntimeError(f"TTS worker failed:\n{msg.get('detail')}")
            _drain_tts_output()

        if generation_result is None and stream_session is not None and hasattr(stream_session, "build_generation_complete_event"):
            generation_result = stream_session.build_generation_complete_event()

        logger.info(f"  === STATS: raw_events={yield_counter}, finished_events={len(finished_events)}, "
                     f"audio_pcm_bytes={len(all_audio_pcm)}, chatbot_msgs={len(chatbot_msgs)} ===")
        if profile_latency:
            stream_total_sec = time.perf_counter() - t_stream_start_perf
            total_with_pre_sec = time.perf_counter() - t_call_start_perf
            raw_events = max(1, int(latency_stats["raw_events"]))
            logger.info(
                "[LAT][TOTAL] pre=%.4fs stream=%.4fs total=%.4fs "
                "first(raw=%s control=%s S=%s speaking=%s audio_chunk=%s pcm=%s) "
                "raw_wait=%.4fs raw_handle=%.4fs decoder=%.4fs "
                "synth_chunk=%.4fs(calls=%d) vocoder=%.4fs(calls=%d) avg_wait=%.4fs avg_handle=%.4fs",
                audio_pre_sec,
                stream_total_sec,
                total_with_pre_sec,
                f"{_lat_value('raw_event'):.4f}s" if _lat_value("raw_event") is not None else "None",
                f"{_lat_value('control_decision'):.4f}s" if _lat_value("control_decision") is not None else "None",
                f"{_lat_value('state_s'):.4f}s" if _lat_value("state_s") is not None else "None",
                f"{_lat_value('speaking_token'):.4f}s" if _lat_value("speaking_token") is not None else "None",
                f"{_lat_value('audio_chunk'):.4f}s" if _lat_value("audio_chunk") is not None else "None",
                f"{_lat_value('pcm_out'):.4f}s" if _lat_value("pcm_out") is not None else "None",
                latency_stats["raw_wait_sec"],
                latency_stats["raw_handle_sec"],
                latency_stats["decoder_sec"],
                latency_stats["audio_chunk_sec"],
                int(latency_stats["synth_chunk_calls"]),
                latency_stats["emit_tts_sec"],
                int(latency_stats["emit_tts_calls"]),
                latency_stats["raw_wait_sec"] / raw_events,
                latency_stats["raw_handle_sec"] / raw_events,
            )

        t_infer = time.time() - t_start

        # --- Final audio output: flush remaining pending chunks ---
        final_audio = None
        if pending_audio_np:
            final_audio = (TTS_SAMPLE_RATE, np.concatenate(pending_audio_np))
            pending_audio_np.clear()

        if len(all_audio_pcm) > 0:
            merged_dur = len(all_audio_pcm) / (TTS_SAMPLE_RATE * 2)
            chatbot_msgs.append({
                "role": "assistant",
                "content": f"Audio synthesis complete: {merged_dur:.2f}s"
            })
        else:
            chatbot_msgs.append({
                "role": "assistant",
                "content": "No audio output (model stayed in listening state)"
            })

        # --- Summary ---
        summary_lines = [
            "---",
            f"### Summary",
            f"- Input: {total_duration:.2f}s | Inference: {t_infer:.2f}s",
            f"- Events: {len(finished_events)}",
            "",
        ]
        for ei, ev in enumerate(finished_events):
            summary_lines.append(
                f"- **Event {ei+1}** [{ev['type']}]: "
                f"{ev['text'][:80]}{'...' if len(ev['text'])>80 else ''} "
                f"({ev['audio_tokens']} audio tokens)"
            )

        chatbot_msgs.append({
            "role": "assistant",
            "content": "\n".join(summary_lines)
        })

        round_trace["finished_events_count"] = int(len(finished_events))
        round_trace["round_completed_at_epoch_ms"] = int(time.time() * 1000)
        round_trace["latency_summary"] = {
            "audio_preprocess_sec": round(float(audio_pre_sec), 6),
            "stream_infer_sec": round(float(time.perf_counter() - t_stream_start_perf), 6),
            "total_round_sec": round(float(time.perf_counter() - t_call_start_perf), 6),
            "raw_wait_sec": round(float(latency_stats["raw_wait_sec"]), 6),
            "raw_handle_sec": round(float(latency_stats["raw_handle_sec"]), 6),
            "decoder_sec": round(float(latency_stats["decoder_sec"]), 6),
            "audio_chunk_synth_sec": round(float(latency_stats["audio_chunk_sec"]), 6),
            "audio_emit_sec": round(float(latency_stats["emit_tts_sec"]), 6),
            "emit_tts_calls": int(latency_stats["emit_tts_calls"]),
            "synth_chunk_calls": int(latency_stats["synth_chunk_calls"]),
            "raw_events": int(latency_stats["raw_events"]),
            "first_raw_event_sec": round(float(_lat_value("raw_event")), 6) if _lat_value("raw_event") is not None else None,
            "first_control_sec": round(float(_lat_value("control_decision")), 6) if _lat_value("control_decision") is not None else None,
            "first_state_s_sec": round(float(_lat_value("state_s")), 6) if _lat_value("state_s") is not None else None,
            "first_speaking_token_sec": round(float(_lat_value("speaking_token")), 6) if _lat_value("speaking_token") is not None else None,
            "first_audio_chunk_sec": round(float(_lat_value("audio_chunk")), 6) if _lat_value("audio_chunk") is not None else None,
            "first_t2w_submit_sec": round(float(_lat_value("t2w_submit")), 6) if _lat_value("t2w_submit") is not None else None,
            "first_pcm_out_sec": round(float(_lat_value("pcm_out")), 6) if _lat_value("pcm_out") is not None else None,
            "t2w_submit_to_first_pcm_sec": (
                round(float(_lat_value("pcm_out")) - float(_lat_value("t2w_submit")), 6)
                if _lat_value("pcm_out") is not None and _lat_value("t2w_submit") is not None
                else None
            ),
        }
        if isinstance(current_event_trace, dict):
            round_trace["open_event_unfinished"] = True

        if isinstance(realtime_ctx, dict):
            next_prefix = None
            next_state_from_control = None
            if stream_session is not None and hasattr(stream_session, "get_prefix"):
                next_prefix = stream_session.get_prefix()
                if hasattr(stream_session, "get_listening_state"):
                    next_state_from_control = str(stream_session.get_listening_state()).lower()
            elif isinstance(generation_result, dict):
                text_ids = generation_result.get("text_ids")
                stoken_ids = generation_result.get("stoken_ids")
                control_ids = generation_result.get("control_ids")
                if (
                    isinstance(text_ids, list)
                    and isinstance(stoken_ids, list)
                    and isinstance(control_ids, list)
                ):
                    next_prefix = {
                        "input_ids": [int(x) for x in text_ids],
                        "stoken_ids": [int(x) for x in stoken_ids],
                        "control_input_ids": [int(x) for x in control_ids],
                    }
                next_state_from_control = _infer_listening_state_from_control_ids(control_ids)
            next_state = (
                next_state_from_control
                if next_state_from_control in {"l", "s", "b"}
                else (current_state if current_state in {"l", "s", "b"} else "l")
            )
            realtime_ctx["next_prefix"] = next_prefix
            realtime_ctx["next_listening_state"] = next_state
            realtime_ctx["round_trace"] = round_trace

        json_result = json.dumps(finished_events, ensure_ascii=False, indent=2, default=str)
        logger.info(f"========== Inference complete run_id={run_id} ==========")
        yield (
            chatbot_msgs,
            final_audio,
            "Inference complete",
            json_result,
            last_audio_debug if last_audio_debug else "[Audio Token / RTF] inference complete",
        )

    except Exception:
        try:
            if "owns_tts_bridge" in locals() and bool(owns_tts_bridge):
                if "active_tts_bridge" in locals() and active_tts_bridge is not None:
                    active_tts_bridge.stop(force_flush=False)
        except Exception:
            pass
        err = traceback.format_exc()
        logger.error(f"Inference error: {err}")
        chatbot_msgs = chatbot_msgs if 'chatbot_msgs' in dir() else []
        chatbot_msgs.append({
            "role": "assistant",
            "content": f"**Error**:\n```\n{err}\n```"
        })
        yield chatbot_msgs, None, "Inference error", "", ""


@dataclass
class InputSilenceGateState:
    """State for the input silence gate."""

    is_open: bool = False
    hangover_frames_left: int = 0


def _apply_input_silence_gate(
    chunk_np: np.ndarray,
    state: InputSilenceGateState,
) -> tuple[np.ndarray, Dict[str, int]]:
    """Apply energy gating while preserving the original sample length."""
    arr = np.asarray(chunk_np, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr, {"total_samples": 0, "muted_samples": 0, "voice_samples": 0}

    frame_size = int(INPUT_SILENCE_GATE_FRAME_SAMPLES)
    open_dbfs = float(INPUT_SILENCE_GATE_OPEN_DBFS)
    close_dbfs = float(INPUT_SILENCE_GATE_CLOSE_DBFS)
    hangover_frames = int(INPUT_SILENCE_GATE_HANGOVER_FRAMES)
    preroll_frames = int(INPUT_SILENCE_GATE_PREROLL_FRAMES)

    frames: List[np.ndarray] = []
    pos = 0
    while pos < arr.size:
        frame = arr[pos:pos + frame_size]
        if frame.size <= 0:
            break
        frames.append(frame.copy())
        pos += frame_size

    emit_mask: List[bool] = [False for _ in range(len(frames))]
    for idx, frame_copy in enumerate(frames):
        rms = float(np.sqrt(np.mean(np.square(frame_copy), dtype=np.float64)))
        dbfs = 20.0 * math.log10(max(rms, 1e-8))

        emit_original = False
        if state.is_open:
            if dbfs >= close_dbfs:
                state.hangover_frames_left = hangover_frames
                emit_original = True
            elif state.hangover_frames_left > 0:
                state.hangover_frames_left -= 1
                emit_original = True
            else:
                state.is_open = False
                emit_original = False
        else:
            if dbfs >= open_dbfs:
                state.is_open = True
                state.hangover_frames_left = hangover_frames
                emit_original = True
                if preroll_frames > 0 and idx > 0:
                    back_start = max(0, idx - preroll_frames)
                    for back_idx in range(back_start, idx):
                        emit_mask[back_idx] = True
            else:
                emit_original = False

        emit_mask[idx] = bool(emit_original)

    out_frames: List[np.ndarray] = []
    voice_samples = 0
    for idx, frame_copy in enumerate(frames):
        if emit_mask[idx]:
            out_frames.append(frame_copy)
            voice_samples += int(frame_copy.size)
        else:
            out_frames.append(np.zeros_like(frame_copy))
    gated = np.concatenate(out_frames).astype(np.float32) if out_frames else np.zeros(0, dtype=np.float32)
    muted_samples = int(max(0, arr.size - voice_samples))
    return gated, {
        "total_samples": int(arr.size),
        "muted_samples": int(muted_samples),
        "voice_samples": int(voice_samples),
    }


@dataclass
class RealtimeSessionState:
    """Complete runtime state for one realtime HTTP session."""

    session_id: str
    start_speak_factor: float
    start_listen_factor: float
    end_speak_factor: float
    prompt_voice: str
    tts_chunk_size: int
    infer_window_samples: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    all_audio_chunks: List[np.ndarray] = field(default_factory=list)
    pending_audio_chunks: List[np.ndarray] = field(default_factory=list)
    pending_samples: int = 0
    total_received_samples: int = 0
    stop_requested: bool = False
    finished: bool = False
    last_assistant_text: str = ""
    realtime_prefix: Optional[dict] = None
    realtime_listening_state: str = "l"
    infer_round: int = 0
    tts_bridge: Optional[RealtimeTTSPool] = None
    audio_hashes: set = field(default_factory=set)
    audio_hash_order: List[str] = field(default_factory=list)
    audio_event_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=REALTIME_AUDIO_EVENT_QUEUE_MAX)
    )
    control_event_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=REALTIME_CONTROL_EVENT_QUEUE_MAX)
    )
    worker: Optional[threading.Thread] = None
    incremental_stream_session: Optional[object] = None
    realtime_stream_audio_cache_snapshot: Optional[dict] = None
    true_incremental_audio: bool = True
    strict_infer_window: bool = False
    incremental_backend: str = "auto"
    stage_timing_log_path: Optional[str] = None
    control_prob_trace_log: bool = False
    control_prob_trace_path: Optional[str] = None
    control_prob_trace_records: List[dict] = field(default_factory=list)
    created_at_epoch_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    trace_rounds: List[dict] = field(default_factory=list)
    trace_state_changes: List[dict] = field(default_factory=list)
    trace_events: List[dict] = field(default_factory=list)
    pushed_state_change_keys: set = field(default_factory=set)
    last_client_chunk_sent_epoch_ms: Optional[int] = None
    last_server_chunk_recv_epoch_ms: Optional[int] = None
    round_timing: Dict[int, dict] = field(default_factory=dict)
    input_silence_gate_state: InputSilenceGateState = field(default_factory=InputSilenceGateState)
    input_silence_gate_total_samples: int = 0
    input_silence_gate_muted_samples: int = 0


_realtime_sessions: Dict[str, RealtimeSessionState] = {}
_realtime_sessions_lock = threading.Lock()


def _get_realtime_session(session_id: str) -> Optional[RealtimeSessionState]:
    with _realtime_sessions_lock:
        return _realtime_sessions.get(session_id)


def _is_audio_event_type(event_type: str) -> bool:
    return str(event_type) in {"audio_chunk_pcm", "audio_chunk"}


def _queue_put_nowait_with_drop(q: queue.Queue, event: dict) -> bool:
    try:
        q.put_nowait(event)
        return True
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(event)
        return True
    except queue.Full:
        return False


def _build_unified_audio_payload(event_payload: dict, event_type: str) -> Optional[dict]:
    if event_type == "audio_chunk_pcm":
        pcm_b64 = event_payload.get("pcm_b64")
        if not isinstance(pcm_b64, str) or not pcm_b64:
            return None
        return {
            "format": "pcm_s16le",
            "sample_rate": event_payload.get("sample_rate"),
            "num_channels": event_payload.get("num_channels", 1),
            "num_samples": event_payload.get("num_samples"),
            "pcm_b64": pcm_b64,
        }

    if event_type == "audio_chunk":
        wav_b64 = event_payload.get("wav_b64")
        if not isinstance(wav_b64, str) or not wav_b64:
            return None
        return {
            "format": "wav_b64",
            "sample_rate": event_payload.get("sample_rate"),
            "num_channels": event_payload.get("num_channels", 1),
            "num_samples": event_payload.get("num_samples"),
            "wav_b64": wav_b64,
        }

    return None


def _decorate_event_with_unified_frame(
    session: RealtimeSessionState,
    event_payload: dict,
    event_type: str,
) -> None:
    if not isinstance(event_payload, dict):
        return

    event_type = str(event_type or event_payload.get("type", "message"))
    text_delta = None
    if event_type == "assistant_text" and isinstance(event_payload.get("text"), str):
        text_delta = event_payload.get("text")

    text_snapshot = None
    if isinstance(text_delta, str) and text_delta:
        text_snapshot = text_delta
    elif REALTIME_UNIFIED_TEXT_SNAPSHOT_ON_AUDIO and _is_audio_event_type(event_type):
        with session.lock:
            cached_text = session.last_assistant_text
        if isinstance(cached_text, str) and cached_text:
            text_snapshot = cached_text

    event_payload["protocol_version"] = REALTIME_UNIFIED_PROTOCOL_VERSION
    event_payload["event_type"] = event_type
    if text_delta is not None or text_snapshot is not None:
        structured_snapshot = (
            event_payload.get("snapshot")
            if isinstance(event_payload.get("snapshot"), str)
            else None
        )
        event_payload["frame_text"] = {
            "delta": text_delta,
            "snapshot": structured_snapshot,
            "event_id": event_payload.get("event_id"),
            "event_kind": event_payload.get("event_kind"),
            "is_final": bool(event_payload.get("is_final", False)),
            "seq": event_payload.get("seq"),
        }
    else:
        event_payload["frame_text"] = None
    event_payload["frame_audio"] = _build_unified_audio_payload(event_payload, event_type)
    event_payload["frame_status"] = (
        event_payload.get("status")
        if event_type == "status" and isinstance(event_payload.get("status"), str)
        else None
    )
    event_payload["frame_error"] = (
        event_payload.get("error")
        if event_type == "error" and isinstance(event_payload.get("error"), str)
        else None
    )


def _push_session_event(session: RealtimeSessionState, event: dict) -> None:
    if not isinstance(event, dict):
        event = {"type": "message", "payload": event}

    event_type = str(event.get("type", "message"))
    primary_q = (
        session.audio_event_queue
        if _is_audio_event_type(event_type)
        else session.control_event_queue
    )
    secondary_q = (
        session.control_event_queue
        if primary_q is session.audio_event_queue
        else session.audio_event_queue
    )

    if _queue_put_nowait_with_drop(primary_q, event):
        return
    _queue_put_nowait_with_drop(secondary_q, event)


def _state_change_position_value(item: dict, default=None):
    if not isinstance(item, dict):
        return default
    for key in ("chunk", "pos", "chunk_pos", "chunk_idx"):
        value = item.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, str) and value.strip() == "":
                continue
        except Exception:
            pass
        return value
    return default


def _state_change_position_int(item: dict, default: int = 0) -> int:
    value = _state_change_position_value(item, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _state_change_audio_ms_from_item(item: dict) -> Optional[float]:
    if not isinstance(item, dict):
        return None

    for key in ("audio_ms", "audio_time_ms", "relative_audio_ms", "audio_rel_ms"):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value

    try:
        value = float(_state_change_position_value(item))
    except (TypeError, ValueError):
        value = None
    if value is not None and math.isfinite(value):
        return round(value * 1000.0 / 25.0, 3)

    return None


def _state_change_event_key(event: dict) -> Optional[tuple]:
    if not isinstance(event, dict):
        return None
    from_state = str(event.get("from", "")).strip().lower()
    to_state = str(event.get("to", "")).strip().lower()
    if not from_state or not to_state:
        return None
    reason = event.get("reason")
    if reason is None or str(reason).strip() == "":
        reason = "unknown"
    chunk_value = _state_change_position_value(event)
    round_id = event.get("round_id", event.get("infer_round"))
    return (
        str(round_id),
        from_state,
        to_state,
        str(reason),
        str(chunk_value),
    )


def _push_state_change_event_once(session: RealtimeSessionState, event: dict) -> bool:
    key = _state_change_event_key(event)
    if key is not None:
        with session.lock:
            if key in session.pushed_state_change_keys:
                return False
            session.pushed_state_change_keys.add(key)
            if len(session.pushed_state_change_keys) > 512:
                session.pushed_state_change_keys = set(list(session.pushed_state_change_keys)[-256:])
    _push_session_event(session, event)
    return True


def _push_state_change_events_from_trace(
    session: RealtimeSessionState,
    round_trace: dict,
    *,
    round_id: int,
    infer_round: int,
    round_audio_start_ms: float = 0.0,
) -> None:
    if not isinstance(round_trace, dict):
        return

    state_items = []
    for state_key in ("state_changes", "state_changes_raw"):
        items = round_trace.get(state_key)
        if isinstance(items, list):
            state_items.extend(items)

    seen = set()
    for item in state_items:
        if not isinstance(item, dict):
            continue
        from_state = str(item.get("from", "")).strip()
        to_state = str(item.get("to", "")).strip()
        if not from_state or not to_state:
            continue

        reason = item.get("reason")
        if reason is None or str(reason).strip() == "":
            reason = "unknown"
        reason = str(reason)

        chunk_value = _state_change_position_value(item)
        local_audio_ms = _state_change_audio_ms_from_item(item)
        audio_ms = None
        try:
            start_ms = float(round_audio_start_ms)
        except (TypeError, ValueError):
            start_ms = 0.0
        if not math.isfinite(start_ms):
            start_ms = 0.0
        if local_audio_ms is not None:
            audio_ms = round(max(0.0, start_ms + float(local_audio_ms)), 3)

        fingerprint = (
            from_state.lower(),
            to_state.lower(),
            str(reason),
            str(chunk_value),
            str(audio_ms),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        payload = dict(item)
        trace_source = payload.get("source")
        payload.update(
            {
                "type": "state_change",
                "source": "round_trace",
                "trace_source": trace_source,
                "from": from_state,
                "to": to_state,
                "reason": reason,
                "round_id": int(round_id),
                "infer_round": int(infer_round),
                "round_audio_start_ms": round(start_ms, 3),
            }
        )
        if chunk_value is not None:
            payload.setdefault("chunk", chunk_value)
            payload.setdefault("chunk_idx", chunk_value)
        if local_audio_ms is not None:
            payload["local_audio_ms"] = round(float(local_audio_ms), 3)
        if audio_ms is not None:
            payload["aligned_audio_ms"] = audio_ms
            payload["processed_audio_ms"] = audio_ms
            payload["audio_ms"] = audio_ms
            payload["audio_time_ms"] = audio_ms
        _push_state_change_event_once(session, payload)


def _pop_session_event(session: RealtimeSessionState, timeout_sec: float = 0.5) -> dict:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        try:
            return session.control_event_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            return session.audio_event_queue.get_nowait()
        except queue.Empty:
            pass

        remain = deadline - time.monotonic()
        if remain <= 0:
            raise queue.Empty()

        wait_sec = min(0.02, remain)
        try:
            return session.control_event_queue.get(timeout=wait_sec)
        except queue.Empty:
            pass
        try:
            return session.audio_event_queue.get_nowait()
        except queue.Empty:
            pass


def _extract_last_assistant_text(chatbot_msgs) -> str:
    if not isinstance(chatbot_msgs, list):
        return ""
    for item in reversed(chatbot_msgs):
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_val = part.get("text")
                    if isinstance(text_val, str) and text_val:
                        text_parts.append(text_val)
            if text_parts:
                return "\n".join(text_parts)
    return ""


def _strip_realtime_event_audio_suffix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s*Audio:\s*[0-9]+[\s\S]*$", "", text).strip()


def _infer_listening_state_from_control_ids(control_ids) -> Optional[str]:
    if not isinstance(control_ids, list) or not control_ids:
        return None
    if generator is None:
        return None

    ss_id = getattr(generator, "ss_token_id", None)
    ks_id = getattr(generator, "ks_token_id", None)
    sl_id = getattr(generator, "sl_token_id", None)
    bc_id = getattr(generator, "bc_token_id", None)
    kl_id = getattr(generator, "kl_token_id", None)

    # Reverse-scan to find the latest semantic control signal, ignoring trailing
    # non-semantic tokens (sleep/detect/pad). This matches realtime continuation.
    for raw_token in reversed(control_ids):
        try:
            tok = int(raw_token)
        except (TypeError, ValueError):
            continue
        if ss_id is not None and tok == int(ss_id):
            return "s"
        if ks_id is not None and tok == int(ks_id):
            return "s"
        if bc_id is not None and tok == int(bc_id):
            return "b"
        if sl_id is not None and tok == int(sl_id):
            return "l"
        if kl_id is not None and tok == int(kl_id):
            return "l"

    return None


def _audio_output_to_wav_event(audio_out):
    if not isinstance(audio_out, tuple) or len(audio_out) != 2:
        return None
    sample_rate, audio_np = audio_out
    if audio_np is None:
        return None
    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0:
        return None
    buf = io.BytesIO()
    sf.write(buf, audio_np, int(sample_rate), format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    wav_hash = hashlib.sha1(wav_bytes).hexdigest()
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {
        "type": "audio_chunk",
        "sample_rate": int(sample_rate),
        "num_samples": int(audio_np.size),
        "wav_b64": wav_b64,
        "_hash": wav_hash,
    }


def _close_incremental_stream_session(stream_session, session_id: str, reason: str) -> None:
    if stream_session is None:
        return
    close_fn = getattr(stream_session, "close", None)
    if not callable(close_fn):
        return
    try:
        result = close_fn()
        logger.info(
            "Realtime incremental stream session closed session_id=%s reason=%s result=%s",
            session_id,
            reason,
            result,
        )
    except Exception:
        logger.warning(
            "Failed to close realtime incremental stream session session_id=%s reason=%s",
            session_id,
            reason,
            exc_info=True,
        )


def _run_realtime_session_worker(session_id: str) -> None:
    with _realtime_sessions_lock:
        session = _realtime_sessions.get(session_id)
    if session is None:
        return

    _push_session_event(session, {"type": "status", "status": "Realtime session started; waiting for audio chunks..."})

    def _register_audio_hash(audio_hash: Optional[str]) -> bool:
        if not isinstance(audio_hash, str) or not audio_hash:
            return False
        with session.lock:
            if audio_hash in session.audio_hashes:
                return False
            session.audio_hashes.add(audio_hash)
            session.audio_hash_order.append(audio_hash)
            while len(session.audio_hash_order) > REALTIME_SESSION_MAX_AUDIO_HASHES:
                old_hash = session.audio_hash_order.pop(0)
                session.audio_hashes.discard(old_hash)
        return True

    def _attach_round_timing(event_payload: dict, round_id: int, emit_epoch_ms: int) -> None:
        if not isinstance(event_payload, dict):
            return
        timing = {}
        fallback_client_sent = None
        fallback_chunk_recv = None
        with session.lock:
            raw = session.round_timing.get(int(round_id))
            if isinstance(raw, dict):
                timing = dict(raw)
            if session.last_client_chunk_sent_epoch_ms is not None:
                fallback_client_sent = int(session.last_client_chunk_sent_epoch_ms)
            if session.last_server_chunk_recv_epoch_ms is not None:
                fallback_chunk_recv = int(session.last_server_chunk_recv_epoch_ms)
            round_timing_ref = session.round_timing.get(int(round_id))
            if isinstance(round_timing_ref, dict):
                round_timing_ref.setdefault("first_server_audio_emit_epoch_ms", int(emit_epoch_ms))
                round_timing_ref["last_server_audio_emit_epoch_ms"] = int(emit_epoch_ms)

        round_started = timing.get("round_started_epoch_ms")
        client_sent = timing.get("latest_client_chunk_sent_epoch_ms", fallback_client_sent)
        chunk_recv = timing.get("latest_server_chunk_recv_epoch_ms", fallback_chunk_recv)
        event_payload["server_round_started_epoch_ms"] = round_started
        event_payload["client_latest_chunk_sent_epoch_ms"] = client_sent
        event_payload["server_latest_chunk_recv_epoch_ms"] = chunk_recv
        try:
            if round_started is not None:
                event_payload["server_round_to_emit_ms"] = max(
                    0, int(emit_epoch_ms) - int(float(round_started))
                )
        except (TypeError, ValueError):
            pass
        try:
            if chunk_recv is not None:
                event_payload["server_chunk_recv_to_emit_ms"] = max(
                    0, int(emit_epoch_ms) - int(float(chunk_recv))
                )
        except (TypeError, ValueError):
            pass
        try:
            if client_sent is not None:
                event_payload["client_chunk_sent_to_emit_ms"] = max(
                    0, int(emit_epoch_ms) - int(float(client_sent))
                )
        except (TypeError, ValueError):
            pass

    def _emit_audio_event(audio_out, round_id: int) -> None:
        audio_event = _audio_output_to_wav_event(audio_out)
        if audio_event is None:
            return
        audio_hash = audio_event.pop("_hash", None)
        if not _register_audio_hash(audio_hash):
            return
        emit_epoch_ms = int(time.time() * 1000)
        audio_event["server_audio_emit_epoch_ms"] = emit_epoch_ms
        _attach_round_timing(audio_event, round_id=round_id, emit_epoch_ms=emit_epoch_ms)
        audio_event["round_id"] = int(round_id)
        _push_session_event(session, audio_event)

    def _emit_pcm_event(pcm_bytes: bytes, round_id: int, t2w_meta: Optional[dict] = None) -> None:
        if not pcm_bytes:
            return
        sample_count = int(len(pcm_bytes) // 2)
        if sample_count <= 0:
            return
        pcm_hash = hashlib.sha1(pcm_bytes).hexdigest()
        if not _register_audio_hash(pcm_hash):
            return
        emit_epoch_ms = int(time.time() * 1000)
        event_payload = {
            "type": "audio_chunk_pcm",
            "pcm_format": "s16le",
            "sample_rate": int(TTS_SAMPLE_RATE),
            "num_channels": 1,
            "num_samples": int(sample_count),
            "pcm_b64": base64.b64encode(pcm_bytes).decode("ascii"),
            "server_audio_emit_epoch_ms": emit_epoch_ms,
            "round_id": int(round_id),
        }
        # Token2Wav activity trace fields for correlating async synthesis with stalls.
        if isinstance(t2w_meta, dict):
            event_payload["t2w_synth_start_epoch_ms"] = t2w_meta.get("synth_start_epoch_ms")
            event_payload["t2w_synth_end_epoch_ms"] = t2w_meta.get("synth_end_epoch_ms")
            event_payload["t2w_synth_duration_ms"] = t2w_meta.get("synth_duration_ms")
            event_payload["t2w_tokens"] = t2w_meta.get("tokens")
            event_payload["t2w_advance_tokens"] = t2w_meta.get("advance_tokens")
            event_payload["t2w_remote_roundtrip_ms"] = t2w_meta.get("remote_roundtrip_ms")
            event_payload["t2w_backend"] = t2w_meta.get("t2w_backend")
        _attach_round_timing(event_payload, round_id=round_id, emit_epoch_ms=emit_epoch_ms)
        _push_session_event(
            session,
            event_payload,
        )

    def _emit_pcm_bytes_direct(pcm_bytes: bytes, t2w_meta: Optional[dict] = None) -> None:
        if not pcm_bytes:
            return
        with session.lock:
            current_round = int(session.infer_round)
        _emit_pcm_event(pcm_bytes, round_id=current_round, t2w_meta=t2w_meta)

    def _on_tts_pcm_message(msg: dict) -> None:
        if not isinstance(msg, dict):
            return
        if msg.get("type") != "pcm":
            return
        _emit_pcm_bytes_direct(msg.get("pcm_bytes", b""), t2w_meta=msg)

    def _peek_audio_prefix(chunks: List[np.ndarray], target_samples: int) -> tuple[List[np.ndarray], int]:
        remaining = max(0, int(target_samples))
        out: List[np.ndarray] = []
        for chunk in chunks:
            if remaining <= 0:
                break
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size <= 0:
                continue
            if arr.size <= remaining:
                out.append(arr)
                remaining -= int(arr.size)
            else:
                out.append(arr[:remaining].copy())
                remaining = 0
        consumed = max(0, int(target_samples) - remaining)
        return out, consumed

    def _drop_audio_prefix_inplace(chunks: List[np.ndarray], consumed_samples: int) -> int:
        remaining = max(0, int(consumed_samples))
        consumed = 0
        while remaining > 0 and chunks:
            head = np.asarray(chunks[0], dtype=np.float32).reshape(-1)
            if head.size <= 0:
                chunks.pop(0)
                continue
            if head.size <= remaining:
                remaining -= int(head.size)
                consumed += int(head.size)
                chunks.pop(0)
            else:
                chunks[0] = head[remaining:].copy()
                consumed += int(remaining)
                remaining = 0
        return consumed

    try:
        prompt_wav_path = _resolve_prompt_wav_path(session.prompt_voice)
        if not os.path.isfile(prompt_wav_path):
            raise FileNotFoundError(f"Prompt wav not found: {prompt_wav_path}")
        session_tts_bridge = RealtimeTTSPool(
            prompt_wav_path=prompt_wav_path,
            vocoder_hop_size=TTS_VOCODER_HOP_SIZE,
            pre_lookahead_len=int(get_token2wav_pre_lookahead_len()),
            worker_name=f"stepaudio_tts_{session_id[:8]}",
            persistent_mode=True,
            pcm_emit_callback=_on_tts_pcm_message,
            # Keep direct callback emit to frontend, but also retain pcm
            # messages in the local queue so round/event timing logic can
            # observe the first token2wav-produced audio packet.
            queue_pcm_messages=True,
        )
        with session.lock:
            session.tts_bridge = session_tts_bridge
    except Exception as exc:
        err = f"Failed to init session TTS bridge: {exc}"
        logger.error(err)
        _push_session_event(session, {"type": "error", "error": err})
        with session.lock:
            session.finished = True
        return

    while True:
        with session.lock:
            pending_samples = int(session.pending_samples)
            should_stop = bool(session.stop_requested)
            strict_infer_window = bool(session.strict_infer_window)

        if should_stop and pending_samples <= 0:
            break

        if pending_samples < session.infer_window_samples and not (should_stop and pending_samples > 0):
            time.sleep(REALTIME_SESSION_POLL_SEC)
            continue

        with session.lock:
            snapshot_total_chunk_count = int(len(session.all_audio_chunks))
            snapshot_pending_chunk_count = int(len(session.pending_audio_chunks))
            snapshot_pending_samples = int(session.pending_samples)
            snapshot_total_received = int(session.total_received_samples)
            session_incremental_backend = (
                session.incremental_backend
                if session.incremental_backend in {"auto", "hf"}
                else REALTIME_INCREMENTAL_BACKEND
            )
            round_prefix = session.realtime_prefix
            round_state = session.realtime_listening_state if session.realtime_listening_state in {"l", "s", "b"} else "l"
            round_stream_session = session.incremental_stream_session
            session.infer_round += 1
            round_id = int(session.infer_round)
            round_started_epoch_ms = int(time.time() * 1000)
            session.round_timing[int(round_id)] = {
                "round_started_epoch_ms": int(round_started_epoch_ms),
                "latest_client_chunk_sent_epoch_ms": (
                    int(session.last_client_chunk_sent_epoch_ms)
                    if session.last_client_chunk_sent_epoch_ms is not None
                    else None
                ),
                "latest_server_chunk_recv_epoch_ms": (
                    int(session.last_server_chunk_recv_epoch_ms)
                    if session.last_server_chunk_recv_epoch_ms is not None
                    else None
                ),
            }
            if len(session.round_timing) > 64:
                obsolete_rounds = sorted(session.round_timing.keys())[:-64]
                for old_round_id in obsolete_rounds:
                    session.round_timing.pop(old_round_id, None)
            round_consumed_samples = int(snapshot_pending_samples)
            if strict_infer_window and not (should_stop and snapshot_pending_samples <= session.infer_window_samples):
                round_consumed_samples = min(int(session.infer_window_samples), int(snapshot_pending_samples))
            if strict_infer_window:
                round_chunks, round_consumed_samples = _peek_audio_prefix(
                    session.pending_audio_chunks,
                    round_consumed_samples,
                )
            else:
                round_chunks = list(session.pending_audio_chunks[:snapshot_pending_chunk_count])
            round_audio_start_samples = max(0, int(snapshot_total_received) - int(snapshot_pending_samples))
            round_audio_start_ms = round(float(round_audio_start_samples) * 1000.0 / float(INPUT_SAMPLE_RATE), 3)
            round_prefix_len = (
                int(len(round_prefix.get("input_ids", [])))
                if isinstance(round_prefix, dict)
                else 0
            )

        if snapshot_total_chunk_count <= 0 or snapshot_pending_samples <= 0:
            if should_stop:
                break
            time.sleep(REALTIME_SESSION_POLL_SEC)
            continue

        audio_round = np.concatenate(round_chunks).astype(np.float32) if round_chunks else np.zeros(0, dtype=np.float32)
        if audio_round.size == 0:
            time.sleep(REALTIME_SESSION_POLL_SEC)
            continue

        status_msg = (
            f"Realtime inference: round={round_id}, "
            f"total_input={snapshot_total_received / INPUT_SAMPLE_RATE:.2f}s, "
            f"new_audio={round_consumed_samples / INPUT_SAMPLE_RATE:.2f}s, "
            f"audio_mode={'strict_incremental' if strict_infer_window else 'incremental'}"
        )
        _push_session_event(session, {"type": "status", "status": status_msg, "round_id": round_id})
        logger.info(
            "Realtime round=%d start prefix_len=%d state=%s total_sec=%.2f pending_sec=%.2f consume_sec=%.2f audio_mode=%s",
            round_id,
            round_prefix_len,
            round_state,
            snapshot_total_received / INPUT_SAMPLE_RATE,
            snapshot_pending_samples / INPUT_SAMPLE_RATE,
            round_consumed_samples / INPUT_SAMPLE_RATE,
            "strict_incremental" if strict_infer_window else "incremental",
        )

        try:
            if round_stream_session is None and hasattr(generator, "create_incremental_stream_session"):
                session_factory_name = "create_incremental_stream_session"
                if session_incremental_backend == "hf":
                    if hasattr(generator, "create_hf_incremental_stream_session"):
                        session_factory_name = "create_hf_incremental_stream_session"
                    else:
                        logger.warning(
                            "HF incremental session requested but generator has no "
                            "create_hf_incremental_stream_session; fallback to default."
                        )
                session_factory = getattr(generator, session_factory_name, None)
                if session_factory is None:
                    raise AttributeError(
                        f"Generator missing session factory: {session_factory_name}"
                    )
                round_stream_session = session_factory(
                    prefix=round_prefix,
                    initial_listening_state=round_state,
                    start_speak_token_factor=session.start_speak_factor,
                    start_listen_token_factor=session.start_listen_factor,
                    end_speak_token_factor=session.end_speak_factor,
                    audio_incremental_mode=True,
                )
                audio_cache_restore_stats = None
                audio_cache_snapshot = None
                with session.lock:
                    if (
                        session.incremental_stream_session is None
                        and isinstance(session.realtime_stream_audio_cache_snapshot, dict)
                    ):
                        audio_cache_snapshot = session.realtime_stream_audio_cache_snapshot
                if isinstance(audio_cache_snapshot, dict):
                    restore_audio_cache = getattr(
                        round_stream_session,
                        "restore_incremental_audio_cache",
                        None,
                    )
                    if not callable(restore_audio_cache):
                        raise AttributeError(
                            "Incremental stream session cannot restore audio cache."
                        )
                    audio_cache_restore_stats = restore_audio_cache(audio_cache_snapshot)
                keep_alive_for_speaking = getattr(
                    round_stream_session, "keep_alive_for_speaking", None
                )
                logger.info(
                    "Realtime incremental stream session created "
                    "(round=%d, prefix_len=%d, state=%s, factory=%s, keep_alive_for_speaking=%s, incremental_backend=%s)",
                    round_id,
                    round_prefix_len,
                    round_state,
                    session_factory_name,
                    keep_alive_for_speaking,
                    session_incremental_backend,
                )
                if isinstance(audio_cache_restore_stats, dict):
                    logger.info(
                        "Realtime incremental stream session restored audio cache "
                        "(round=%d, requested_tokens=%d, restored_tokens=%d, windows=%d)",
                        round_id,
                        int(audio_cache_restore_stats.get("requested_audio_input_token_count", 0)),
                        int(audio_cache_restore_stats.get("audio_input_token_count", 0)),
                        int(audio_cache_restore_stats.get("audio_window_count", 0)),
                    )
                with session.lock:
                    if session.incremental_stream_session is None:
                        session.incremental_stream_session = round_stream_session
                        session.realtime_stream_audio_cache_snapshot = None
                    else:
                        round_stream_session = session.incremental_stream_session

            round_ctx = {}
            round_flush_audio_tail = bool(should_stop)

            def _emit_direct_text_event(text_event: dict) -> None:
                if not isinstance(text_event, dict):
                    return
                snapshot = text_event.get("snapshot")
                delta = text_event.get("delta")
                if not isinstance(snapshot, str):
                    snapshot = ""
                if not isinstance(delta, str):
                    delta = ""
                if not snapshot and not delta:
                    return
                decoder_event_id = text_event.get("event_id")
                event_kind = str(text_event.get("event_kind") or "response").strip().lower() or "response"
                event_id = f"{round_id}:{decoder_event_id or event_kind}"
                try:
                    seq = int(text_event.get("seq", 0))
                except (TypeError, ValueError):
                    seq = 0
                payload = {
                    "type": "assistant_text",
                    "text": delta or snapshot,
                    "delta": delta,
                    "snapshot": snapshot,
                    "event_id": event_id,
                    "event_kind": event_kind,
                    "is_final": bool(text_event.get("is_final", False)),
                    "seq": seq,
                    "round_id": int(round_id),
                    "source": "decoder_direct",
                    "resumed": bool(text_event.get("resumed", False)),
                    "break_before": not bool(text_event.get("resumed", False)),
                }
                with session.lock:
                    session.last_assistant_text = snapshot or delta
                _push_session_event(session, payload)

            def _emit_direct_state_event(state_event: dict) -> None:
                if not isinstance(state_event, dict):
                    return
                if not CONTROL_EARLY_EXIT_ENABLED or not CONTROL_EARLY_STATE_SSE:
                    return
                from_state = str(state_event.get("from", "")).strip()
                to_state = str(state_event.get("to", "")).strip()
                if from_state.upper() != "L" and to_state.upper() != "L":
                    return
                if {from_state.lower(), to_state.lower()} - {"l", "s"}:
                    return
                now_ms = int(time.time() * 1000)
                payload = dict(state_event)
                payload.update(
                    {
                        "type": "state_change",
                        "source": "model_early_exit",
                        "round_id": int(round_id),
                        "infer_round": int(round_id),
                        "server_state_commit_epoch_ms": int(
                            payload.get("server_state_commit_epoch_ms") or now_ms
                        ),
                    }
                )
                payload.setdefault("chunk", payload.get("chunk_idx"))
                payload.setdefault("chunk_idx", payload.get("chunk"))
                pushed = _push_state_change_event_once(session, payload)
                if not pushed:
                    return
                if bool(payload.get("interrupt", False)) and CONTROL_EARLY_TTS_ABORT:
                    reason = str(payload.get("interrupt_reason") or "interrupt")
                    abort_meta = {}
                    try:
                        if session_tts_bridge is not None:
                            abort_meta = session_tts_bridge.submit_event_abort(reason=reason)
                    except Exception:
                        logger.exception("Failed to abort TTS bridge on early interrupt")
                    dropped_audio_events = 0
                    while True:
                        try:
                            session.audio_event_queue.get_nowait()
                            dropped_audio_events += 1
                        except queue.Empty:
                            break
                    interrupt_payload = {
                        "type": "audio_interrupt",
                        "reason": reason,
                        "round_id": int(round_id),
                        "source": "model_early_exit",
                        "server_interrupt_epoch_ms": int(time.time() * 1000),
                        "dropped_audio_events": int(dropped_audio_events),
                        "tts_abort_generation_id": abort_meta.get("generation_id") if isinstance(abort_meta, dict) else None,
                        "tts_old_stream_id": abort_meta.get("old_stream_id") if isinstance(abort_meta, dict) else None,
                        "tts_new_stream_id": abort_meta.get("new_stream_id") if isinstance(abort_meta, dict) else None,
                    }
                    _push_session_event(session, interrupt_payload)
                    if CONTROL_EARLY_DEBUG:
                        logger.info(
                            "Early interrupt emitted round=%d reason=%s abort_meta=%s",
                            int(round_id),
                            reason,
                            abort_meta,
                        )

            inference_gen = run_chunk_dialogue_inference(
                audio_input=(INPUT_SAMPLE_RATE, audio_round),
                start_speak_factor=session.start_speak_factor,
                end_speak_factor=session.end_speak_factor,
                prompt_voice=session.prompt_voice,
                tts_chunk_size=session.tts_chunk_size,
                start_listen_factor=session.start_listen_factor,
                prefix=round_prefix,
                initial_listening_state=round_state,
                realtime_ctx=round_ctx,
                stream_session=round_stream_session,
                stream_audio_incremental=True,
                stream_flush_audio_tail=round_flush_audio_tail,
                tts_bridge=session_tts_bridge,
                flush_tts_on_round_end=False,
                direct_text_callback=_emit_direct_text_event,
                direct_state_callback=_emit_direct_state_event,
            )

            text_event_kind = "response"
            text_event_seq = 0
            text_event_instance = 0
            text_event_id = f"{round_id}:{text_event_kind}:{text_event_instance}"

            for item in inference_gen:
                if not isinstance(item, tuple) or len(item) < 5:
                    continue
                chatbot_msgs, audio_out, status_text, _json_text, _audio_debug = item

                if isinstance(status_text, str) and status_text:
                    _push_session_event(session, {"type": "status", "status": status_text, "round_id": round_id})

                # Text is emitted directly from StreamingDecoder through
                # _emit_direct_text_event. Do not re-parse chatbot markdown here.

                _emit_audio_event(audio_out, round_id=round_id)

            round_trace = round_ctx.get("round_trace")
            round_trace_copy_for_sse = None
            with session.lock:
                if snapshot_pending_chunk_count > 0 and session.pending_audio_chunks:
                    if strict_infer_window:
                        _drop_audio_prefix_inplace(session.pending_audio_chunks, round_consumed_samples)
                    else:
                        del session.pending_audio_chunks[:min(snapshot_pending_chunk_count, len(session.pending_audio_chunks))]
                session.pending_samples = max(0, int(session.pending_samples) - int(round_consumed_samples))
                if isinstance(round_trace, dict):
                    round_trace.setdefault("round_id", int(round_id))
                    timing_for_round = session.round_timing.get(int(round_id))
                    if isinstance(timing_for_round, dict):
                        round_trace["round_timing"] = dict(timing_for_round)
                    try:
                        round_trace_copy = json.loads(json.dumps(round_trace, ensure_ascii=False, default=str))
                    except Exception:
                        round_trace_copy = dict(round_trace)
                    round_trace_copy_for_sse = round_trace_copy
                    # Push compact stage timings to the frontend every round,
                    # independent of whether stage-timing files are written
                    # (weight-test sends stage_timing_log=false).
                    try:
                        _stage_entry = _build_realtime_timeline_entry(
                            session_id=session.session_id,
                            round_id=int(round_id),
                            audio_mode=(
                                "strict_incremental"
                                if strict_infer_window
                                else "incremental"
                            ),
                            input_samples=int(round_consumed_samples),
                            input_duration_sec=float(round_consumed_samples) / float(INPUT_SAMPLE_RATE),
                            round_trace=round_trace_copy,
                        )
                        if _stage_entry is not None:
                            _queue_meta = {
                                "pending_before_ms": round(float(snapshot_pending_samples) / float(INPUT_SAMPLE_RATE) * 1000.0, 1),
                                "pending_after_ms": round(
                                    max(0, int(snapshot_pending_samples) - int(round_consumed_samples)) / float(INPUT_SAMPLE_RATE) * 1000.0, 1
                                ),
                                "consumed_ms": round(float(round_consumed_samples) / float(INPUT_SAMPLE_RATE) * 1000.0, 1),
                                "infer_window_ms": round(float(session.infer_window_samples) / float(INPUT_SAMPLE_RATE) * 1000.0, 1),
                            }
                            _push_session_event(
                                session,
                                _build_stage_timing_event(_stage_entry, int(round_id), queue_meta=_queue_meta),
                            )
                    except Exception:
                        logger.error(
                            "Failed to push stage_timing event: %s",
                            traceback.format_exc(),
                        )
                    if session.stage_timing_log_path:
                        try:
                            _append_realtime_stage_timing_log(
                                session.stage_timing_log_path,
                                session_id=session.session_id,
                                round_id=int(round_id),
                                audio_mode=(
                                    "strict_incremental"
                                    if strict_infer_window
                                    else "incremental"
                                ),
                                input_samples=int(round_consumed_samples),
                                input_duration_sec=float(round_consumed_samples) / float(INPUT_SAMPLE_RATE),
                                round_trace=round_trace_copy,
                            )
                        except Exception:
                            logger.error(
                                "Failed to append realtime stage timing log: %s",
                                traceback.format_exc(),
                            )
                        try:
                            _append_realtime_timeline_log(
                                session.stage_timing_log_path,
                                session_id=session.session_id,
                                round_id=int(round_id),
                                audio_mode=(
                                    "strict_incremental"
                                    if strict_infer_window
                                    else "incremental"
                                ),
                                input_samples=int(round_consumed_samples),
                                input_duration_sec=float(round_consumed_samples) / float(INPUT_SAMPLE_RATE),
                                round_trace=round_trace_copy,
                            )
                        except Exception:
                            logger.error(
                                "Failed to append realtime timeline log: %s",
                                traceback.format_exc(),
                            )
                    if session.control_prob_trace_log and session.control_prob_trace_path:
                        prob_points = round_trace_copy.get("control_prob_points")
                        if isinstance(prob_points, list):
                            for prob_item in prob_points:
                                if not isinstance(prob_item, dict):
                                    continue
                                enriched_prob = dict(prob_item)
                                enriched_prob.setdefault("session_id", str(session.session_id))
                                enriched_prob.setdefault("round_id", int(round_id))
                                enriched_prob.setdefault("infer_round", int(session.infer_round))
                                enriched_prob.setdefault("audio_mode", (
                                    "strict_incremental"
                                    if strict_infer_window
                                    else "incremental"
                                ))
                                session.control_prob_trace_records.append(enriched_prob)
                            try:
                                _flush_control_prob_trace_json(
                                    session.control_prob_trace_path,
                                    session_id=session.session_id,
                                    records=session.control_prob_trace_records,
                                )
                            except Exception:
                                logger.error(
                                    "Failed to flush control prob trace json: %s",
                                    traceback.format_exc(),
                                )
                    session.trace_rounds.append(round_trace_copy)
                    state_seen = set()
                    for state_key in ("state_changes", "state_changes_raw"):
                        state_changes = round_trace_copy.get(state_key)
                        if not isinstance(state_changes, list):
                            continue
                        for state_item in state_changes:
                            if isinstance(state_item, dict):
                                state_entry = dict(state_item)
                                state_entry.setdefault("round_id", int(round_id))
                                reason_val = state_entry.get("reason")
                                if reason_val is None or str(reason_val).strip() == "":
                                    state_entry["reason"] = "unknown"
                                try:
                                    fingerprint = json.dumps(state_entry, ensure_ascii=False, sort_keys=True, default=str)
                                except Exception:
                                    fingerprint = str(state_entry)
                                if fingerprint in state_seen:
                                    continue
                                state_seen.add(fingerprint)
                                session.trace_state_changes.append(state_entry)
                    events = round_trace_copy.get("events")
                    if isinstance(events, list):
                        for event_item in events:
                            if isinstance(event_item, dict):
                                event_entry = dict(event_item)
                                event_entry.setdefault("round_id", int(round_id))
                                session.trace_events.append(event_entry)
                next_prefix = round_ctx.get("next_prefix")
                if isinstance(next_prefix, dict):
                    session.realtime_prefix = next_prefix
                next_state = round_ctx.get("next_listening_state")
                if isinstance(next_state, str) and next_state in {"l", "s", "b"}:
                    session.realtime_listening_state = next_state
                logger.info(
                    "Realtime round=%d done next_prefix_len=%d next_state=%s",
                    round_id,
                    int(len(session.realtime_prefix.get("input_ids", []))) if isinstance(session.realtime_prefix, dict) else 0,
                    session.realtime_listening_state,
                )
            if isinstance(round_trace_copy_for_sse, dict):
                _push_state_change_events_from_trace(
                    session,
                    round_trace_copy_for_sse,
                    round_id=int(round_id),
                    infer_round=int(round_id),
                    round_audio_start_ms=float(round_audio_start_ms),
                )
        except Exception:
            err = traceback.format_exc()
            logger.error("Realtime session worker error (%s): %s", session_id, err)
            _push_session_event(session, {"type": "error", "error": err, "round_id": round_id})
            stream_session_to_close = None
            with session.lock:
                stream_session_to_close = session.incremental_stream_session
                session.incremental_stream_session = None
                session.finished = True
                session.tts_bridge = None
            _close_incremental_stream_session(
                stream_session_to_close,
                session_id,
                reason="worker_error",
            )
            try:
                if session_tts_bridge is not None:
                    session_tts_bridge.stop(force_flush=False)
            except Exception:
                pass
            return

    try:
        # Session stop: flush tail once (if any) and emit remaining audio.
        if session_tts_bridge is not None:
            session_tts_bridge.submit_session_stop(force_flush=True)
            flushed_msgs, _ = session_tts_bridge.wait_event_stats(max_wait_sec=5.0)
            flushed_msgs.extend(session_tts_bridge.drain(block=False)[0])
            for msg in flushed_msgs:
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "error":
                    logger.error("Session TTS worker error on stop: %s", msg.get("detail"))
                    continue
                if msg.get("type") != "pcm":
                    continue
                pcm_bytes = msg.get("pcm_bytes", b"")
                if not pcm_bytes:
                    continue
                _emit_pcm_event(pcm_bytes, round_id=int(session.infer_round))
    except Exception:
        logger.error("Failed to flush session TTS bridge on stop: %s", traceback.format_exc())
    finally:
        try:
            if session_tts_bridge is not None:
                session_tts_bridge.stop(force_flush=False)
        except Exception:
            pass

    stream_session_to_close = None
    with session.lock:
        stream_session_to_close = session.incremental_stream_session
        session.incremental_stream_session = None
        session.tts_bridge = None
        session.finished = True
    _close_incremental_stream_session(
        stream_session_to_close,
        session_id,
        reason="session_finished",
    )
    _push_session_event(session, {"type": "done", "session_id": session_id})


def _get_realtime_session_or_404(session_id: str) -> RealtimeSessionState:
    with _realtime_sessions_lock:
        session = _realtime_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session


def _fmt_diag_float(value, digits: int = 4) -> str:
    try:
        if value is None:
            return "None"
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return "None"


def _append_realtime_stage_timing_log(
    log_path: str,
    *,
    session_id: str,
    round_id: int,
    audio_mode: str,
    input_samples: int,
    input_duration_sec: float,
    round_trace: dict,
) -> None:
    if not log_path or not isinstance(round_trace, dict):
        return

    lat = round_trace.get("latency_summary", {})
    if not isinstance(lat, dict):
        lat = {}
    gen = round_trace.get("generation_complete", {})
    if not isinstance(gen, dict):
        gen = {}
    round_started_at = round_trace.get("round_started_at_epoch_ms")
    round_completed_at = round_trace.get("round_completed_at_epoch_ms")
    finished_events_count = round_trace.get("finished_events_count")

    lines = [
        f"[round {int(round_id)}] session={session_id} mode={audio_mode}",
        f"input_samples={int(input_samples)} input_duration_sec={_fmt_diag_float(input_duration_sec, 3)}",
        f"audio_preprocess_sec={_fmt_diag_float(lat.get('audio_preprocess_sec'))}",
        f"stream_infer_sec={_fmt_diag_float(lat.get('stream_infer_sec'))}",
        f"total_round_sec={_fmt_diag_float(lat.get('total_round_sec'))}",
        f"decoder_sec={_fmt_diag_float(lat.get('decoder_sec'))}",
        f"raw_wait_sec={_fmt_diag_float(lat.get('raw_wait_sec'))}",
        f"raw_handle_sec={_fmt_diag_float(lat.get('raw_handle_sec'))}",
        f"audio_chunk_synth_sec={_fmt_diag_float(lat.get('audio_chunk_synth_sec'))}",
        f"audio_emit_sec={_fmt_diag_float(lat.get('audio_emit_sec'))}",
        f"first_control_sec={_fmt_diag_float(lat.get('first_control_sec'))}",
        f"first_state_s_sec={_fmt_diag_float(lat.get('first_state_s_sec'))}",
        f"first_speaking_token_sec={_fmt_diag_float(lat.get('first_speaking_token_sec'))}",
        f"first_audio_chunk_sec={_fmt_diag_float(lat.get('first_audio_chunk_sec'))}",
        f"first_t2w_submit_sec={_fmt_diag_float(lat.get('first_t2w_submit_sec'))}",
        f"first_pcm_out_sec={_fmt_diag_float(lat.get('first_pcm_out_sec'))}",
        f"t2w_submit_to_first_pcm_sec={_fmt_diag_float(lat.get('t2w_submit_to_first_pcm_sec'))}",
        f"generation_complete_rel_sec={_fmt_diag_float(gen.get('rel_sec'))}",
        f"round_started_at_epoch_ms={round_started_at}",
        f"round_completed_at_epoch_ms={round_completed_at}",
        f"finished_events_count={finished_events_count}",
    ]

    events = round_trace.get("events", [])
    if isinstance(events, list) and events:
        for event in events:
            if not isinstance(event, dict):
                continue
            event_index = event.get("event_index")
            kind = event.get("kind")
            start_rel_ms = event.get("start_rel_ms")
            end_rel_ms = event.get("end_rel_ms")
            first_text_latency_ms = event.get("first_text_token_latency_ms")
            first_stoken_latency_ms = event.get("first_stoken_latency_ms")
            first_t2w_submit_latency_ms = event.get("first_t2w_submit_latency_ms")
            first_pcm_out_latency_ms = event.get("first_pcm_out_latency_ms")
            event_duration_ms = event.get("event_duration_ms")
            model_audio_tokens = event.get("model_audio_tokens")
            stream_audio_tokens = event.get("stream_audio_tokens")
            effective_audio_tokens = event.get("effective_audio_tokens")
            vocoder_calls = event.get("vocoder_calls")
            vocoder_output_seconds = event.get("vocoder_output_seconds")
            t2w_submit_to_first_pcm_sec = None
            try:
                if first_t2w_submit_latency_ms is not None and first_pcm_out_latency_ms is not None:
                    t2w_submit_to_first_pcm_sec = (
                        float(first_pcm_out_latency_ms) - float(first_t2w_submit_latency_ms)
                    ) / 1000.0
            except (TypeError, ValueError):
                t2w_submit_to_first_pcm_sec = None
            lines.extend(
                [
                    f"  [event {event_index}] kind={kind}",
                    f"  start_rel_sec={_fmt_diag_float(None if start_rel_ms is None else float(start_rel_ms) / 1000.0, 6)}",
                    f"  end_rel_sec={_fmt_diag_float(None if end_rel_ms is None else float(end_rel_ms) / 1000.0, 6)}",
                    f"  first_text_token_latency_sec={_fmt_diag_float(None if first_text_latency_ms is None else float(first_text_latency_ms) / 1000.0, 6)}",
                    f"  first_stoken_latency_sec={_fmt_diag_float(None if first_stoken_latency_ms is None else float(first_stoken_latency_ms) / 1000.0, 6)}",
                    f"  first_t2w_submit_latency_sec={_fmt_diag_float(None if first_t2w_submit_latency_ms is None else float(first_t2w_submit_latency_ms) / 1000.0, 6)}",
                    f"  first_pcm_out_latency_sec={_fmt_diag_float(None if first_pcm_out_latency_ms is None else float(first_pcm_out_latency_ms) / 1000.0, 6)}",
                    f"  t2w_submit_to_first_pcm_sec={_fmt_diag_float(t2w_submit_to_first_pcm_sec, 6)}",
                    f"  event_duration_sec={_fmt_diag_float(None if event_duration_ms is None else float(event_duration_ms) / 1000.0, 6)}",
                    f"  model_audio_tokens={model_audio_tokens}",
                    f"  stream_audio_tokens={stream_audio_tokens}",
                    f"  effective_audio_tokens={effective_audio_tokens}",
                    f"  vocoder_calls={vocoder_calls}",
                    f"  vocoder_output_seconds={_fmt_diag_float(vocoder_output_seconds, 6)}",
                ]
            )
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _realtime_timeline_log_path(stage_log_path: str) -> str:
    if not stage_log_path:
        return ""
    directory = os.path.dirname(stage_log_path)
    basename = os.path.basename(stage_log_path)
    if basename.startswith("realtime_stage_timing_") and basename.endswith(".txt"):
        suffix = basename[len("realtime_stage_timing_"):-len(".txt")]
        return os.path.join(directory, f"realtime_timeline_{suffix}.jsonl")
    return f"{stage_log_path}.timeline.jsonl"


def _build_realtime_timeline_entry(
    *,
    session_id: str,
    round_id: int,
    audio_mode: str,
    input_samples: int,
    input_duration_sec: float,
    round_trace: dict,
) -> Optional[dict]:
    """Aggregate one round's timeline spans into an entry dict.

    Pure computation (no file I/O). Reused by both the file writer
    (_append_realtime_timeline_log) and the SSE stage_timing push.
    Returns None if round_trace is not usable.
    """
    if not isinstance(round_trace, dict):
        return None
    spans = round_trace.get("timeline_spans", [])
    if not isinstance(spans, list):
        spans = []
    round_timing = round_trace.get("round_timing", {})
    if not isinstance(round_timing, dict):
        round_timing = {}
    synthetic_spans = []

    def _add_epoch_span(name: str, start_value, end_value) -> None:
        try:
            start_ms = int(float(start_value))
            end_ms = int(float(end_value))
        except (TypeError, ValueError):
            return
        if end_ms < start_ms:
            return
        synthetic_spans.append({
            "name": str(name),
            "start_epoch_ms": start_ms,
            "end_epoch_ms": end_ms,
            "duration_ms": round(float(end_ms - start_ms), 3),
            "source": "round_timing",
        })

    _add_epoch_span(
        "input_queue_wait",
        round_timing.get("latest_server_chunk_recv_epoch_ms"),
        round_timing.get("round_started_epoch_ms"),
    )
    _add_epoch_span(
        "round_to_first_audio_emit",
        round_timing.get("round_started_epoch_ms"),
        round_timing.get("first_server_audio_emit_epoch_ms"),
    )
    _add_epoch_span(
        "output_queue_send",
        round_timing.get("first_server_audio_emit_epoch_ms"),
        round_timing.get("first_server_sse_send_epoch_ms"),
    )
    spans = list(spans) + synthetic_spans
    clean_spans = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        item = dict(span)
        try:
            item["start_epoch_ms"] = int(float(item.get("start_epoch_ms")))
            item["end_epoch_ms"] = int(float(item.get("end_epoch_ms")))
            item["duration_ms"] = round(float(item.get("duration_ms", item["end_epoch_ms"] - item["start_epoch_ms"])), 3)
        except (TypeError, ValueError, KeyError):
            continue
        clean_spans.append(item)
    clean_spans.sort(key=lambda x: (x.get("start_epoch_ms", 0), x.get("end_epoch_ms", 0), x.get("name", "")))

    by_name = {}
    for span in clean_spans:
        name = str(span.get("name") or "unknown")
        stats = by_name.setdefault(name, {"count": 0, "duration_ms": 0.0})
        stats["count"] += 1
        stats["duration_ms"] = round(float(stats["duration_ms"]) + float(span.get("duration_ms", 0.0)), 3)

    entry = {
        "type": "realtime_timeline",
        "trace_version": 1,
        "written_epoch_ms": int(time.time() * 1000),
        "session_id": str(session_id),
        "round_id": int(round_id),
        "audio_mode": str(audio_mode),
        "input_samples": int(input_samples),
        "input_duration_sec": float(input_duration_sec),
        "round_started_at_epoch_ms": round_trace.get("round_started_at_epoch_ms"),
        "round_completed_at_epoch_ms": round_trace.get("round_completed_at_epoch_ms"),
        "round_timing": round_timing,
        "latency_summary": round_trace.get("latency_summary") if isinstance(round_trace.get("latency_summary"), dict) else {},
        "span_summary": by_name,
        "spans": clean_spans,
    }
    return entry


def _append_realtime_timeline_log(
    stage_log_path: str,
    *,
    session_id: str,
    round_id: int,
    audio_mode: str,
    input_samples: int,
    input_duration_sec: float,
    round_trace: dict,
) -> None:
    if not stage_log_path or not isinstance(round_trace, dict):
        return
    entry = _build_realtime_timeline_entry(
        session_id=session_id,
        round_id=round_id,
        audio_mode=audio_mode,
        input_samples=input_samples,
        input_duration_sec=input_duration_sec,
        round_trace=round_trace,
    )
    if entry is None:
        return
    log_path = _realtime_timeline_log_path(stage_log_path)
    if not log_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str))
        f.write("\n")


def _build_stage_timing_event(entry: dict, round_id: int, queue_meta: Optional[dict] = None) -> dict:
    """Build a compact SSE event carrying one round's stage timings (ms).

    span_summary durations are already in ms; latency_summary is in seconds,
    so latency fields are converted to ms here. token2wav and other spans may
    be absent depending on backend config -> the field is then None.
    """
    span_summary = entry.get("span_summary") or {}
    latency = entry.get("latency_summary") or {}

    def span_ms(name):
        slot = span_summary.get(name)
        if isinstance(slot, dict):
            v = slot.get("duration_ms")
            return round(float(v), 3) if v is not None else None
        return None

    def sec_ms(v):
        try:
            return round(float(v) * 1000.0, 1) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Dominant round state from transformer spans; any speaking span wins.
    _states = [
        str(sp.get("state"))
        for sp in (entry.get("spans") or [])
        if isinstance(sp, dict) and sp.get("name") == "transformer" and sp.get("state")
    ]
    if "speaking" in _states:
        round_state = "speaking"
    elif "backchannel" in _states:
        round_state = "backchannel"
    elif "listening" in _states:
        round_state = "listening"
    else:
        round_state = None

    return {
        "type": "stage_timing",
        "round_id": int(round_id),
        "state": round_state,
        "input_duration_sec": entry.get("input_duration_sec"),
        "audio_mode": entry.get("audio_mode"),
        "round_started_at_epoch_ms": entry.get("round_started_at_epoch_ms"),
        "round_completed_at_epoch_ms": entry.get("round_completed_at_epoch_ms"),
        "stages": {
            "input_queue_wait": span_ms("input_queue_wait"),
            "audio_preprocess": span_ms("audio_preprocess"),
            "audio_encoder": span_ms("audio_encoder"),
            "transformer": span_ms("transformer"),
            "token2wav": span_ms("token2wav"),
            "round_to_first_audio_emit": span_ms("round_to_first_audio_emit"),
            "output_queue_send": span_ms("output_queue_send"),
        },
        "latency": {
            "first_pcm_out_ms": sec_ms(latency.get("first_pcm_out_sec")),
            "first_t2w_submit_ms": sec_ms(latency.get("first_t2w_submit_sec")),
            "t2w_submit_to_first_pcm_ms": sec_ms(latency.get("t2w_submit_to_first_pcm_sec")),
            "stream_infer_ms": sec_ms(latency.get("stream_infer_sec")),
            "total_round_ms": sec_ms(latency.get("total_round_sec")),
        },
        "queue": queue_meta if isinstance(queue_meta, dict) else None,
    }


def _flush_control_prob_trace_json(
    log_path: str,
    *,
    session_id: str,
    records: List[dict],
) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    payload = {
        "schema_version": 1,
        "type": "realtime_control_prob_trace",
        "session_id": str(session_id),
        "record_count": int(len(records)),
        "records": records,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def register_realtime_session_routes(app) -> None:
    """Register HTTP API routes for realtime sessions."""
    @app.get("/api/realtime/voices")
    async def list_realtime_prompt_voices():
        """Return available prompt voices."""
        return JSONResponse(
            {
                "default_voice": DEFAULT_REALTIME_PROMPT_VOICE,
                "clone_prompt_dir": os.path.abspath(CLONE_PROMPT_DIR),
                "voices": _available_realtime_prompt_voices(),
            }
        )

    @app.post("/api/realtime/warmup")
    async def realtime_warmup(request: Request):
        """Warm up the local or remote Token2Wav service."""
        if not is_token2wav_available():
            raise HTTPException(status_code=503, detail="Token2Wav is not loaded yet.")
        payload = {}
        try:
            body = await request.body()
            if body:
                payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        prompt_voice = str(payload.get("prompt_voice", STARTUP_WARMUP_PROMPT_VOICE))
        if REMOTE_TOKEN2WAV_ENABLED and token2wav_model is None:
            try:
                health = get_remote_token2wav_client().health()
                summary = f"Remote Token2Wav health OK: {health}"
            except Exception as exc:
                if not REMOTE_TOKEN2WAV_FALLBACK:
                    raise HTTPException(status_code=503, detail=f"Remote Token2Wav warmup failed: {exc}") from exc
                ensure_local_token2wav_loaded()
                summary = warmup_token2wav(prompt_voice=prompt_voice)
        else:
            summary = warmup_token2wav(prompt_voice=prompt_voice)
        return JSONResponse(
            {
                "ok": "failed" not in summary.lower(),
                "summary": summary,
                "prompt_voice": prompt_voice,
            }
        )

    @app.post("/api/realtime/session/start")
    async def start_realtime_session(request: Request):
        """Create a new realtime inference session."""
        if generator is None or not is_token2wav_available():
            raise HTTPException(status_code=503, detail="Models are not loaded yet.")

        payload = {}
        try:
            body = await request.body()
            if body:
                payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        start_speak_factor = float(payload.get("start_speak_factor", 1.2))
        start_listen_factor = float(payload.get("start_listen_factor", 1.2))
        end_speak_factor = float(payload.get("end_speak_factor", 1.0))
        prompt_voice = _normalize_prompt_voice_id(
            str(payload.get("prompt_voice", DEFAULT_REALTIME_PROMPT_VOICE))
        )
        try:
            prompt_wav_path = _resolve_prompt_wav_path(prompt_voice)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not os.path.isfile(prompt_wav_path):
            raise HTTPException(status_code=400, detail=f"Prompt wav not found: {prompt_wav_path}")
        tts_chunk_size = int(payload.get("tts_chunk_size", REALTIME_TTS_CHUNK_SIZE_DEFAULT))
        tts_chunk_size = max(1, tts_chunk_size)
        true_incremental_audio = True
        strict_infer_window_raw = payload.get("strict_infer_window", REALTIME_STRICT_INFER_WINDOW)
        if isinstance(strict_infer_window_raw, str):
            strict_infer_window = strict_infer_window_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            strict_infer_window = bool(strict_infer_window_raw)
        stage_timing_log_raw = payload.get("stage_timing_log", REALTIME_STAGE_TIMING_LOG)
        if isinstance(stage_timing_log_raw, str):
            stage_timing_log = stage_timing_log_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            stage_timing_log = bool(stage_timing_log_raw)
        control_prob_trace_log_raw = payload.get("control_prob_trace_log", REALTIME_CONTROL_PROB_TRACE_LOG)
        if isinstance(control_prob_trace_log_raw, str):
            control_prob_trace_log = control_prob_trace_log_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            control_prob_trace_log = bool(control_prob_trace_log_raw)
        incremental_backend = str(
            payload.get("incremental_backend", REALTIME_INCREMENTAL_BACKEND)
        ).strip().lower()
        if incremental_backend not in {"auto", "hf"}:
            incremental_backend = REALTIME_INCREMENTAL_BACKEND
        infer_window_ms = int(payload.get("infer_window_ms", REALTIME_INFER_WINDOW_MS))
        infer_window_ms = max(REALTIME_INFER_WINDOW_MIN_MS, infer_window_ms)
        infer_window_samples = max(1, int(INPUT_SAMPLE_RATE * infer_window_ms / 1000))


        session_id = uuid.uuid4().hex
        session_time_tag = _readable_time_tag()
        stage_timing_log_path = None
        if stage_timing_log:
            os.makedirs(REALTIME_STAGE_TIMING_LOG_DIR, exist_ok=True)
            stage_timing_log_path = os.path.join(
                REALTIME_STAGE_TIMING_LOG_DIR,
                f"realtime_stage_timing_{session_time_tag}.txt",
            )
        control_prob_trace_path = None
        if control_prob_trace_log:
            os.makedirs(REALTIME_CONTROL_PROB_TRACE_LOG_DIR, exist_ok=True)
            control_prob_trace_path = os.path.join(
                REALTIME_CONTROL_PROB_TRACE_LOG_DIR,
                f"realtime_control_prob_{session_time_tag}.json",
            )
        session = RealtimeSessionState(
            session_id=session_id,
            start_speak_factor=start_speak_factor,
            start_listen_factor=start_listen_factor,
            end_speak_factor=end_speak_factor,
            prompt_voice=prompt_voice,
            tts_chunk_size=tts_chunk_size,
            infer_window_samples=infer_window_samples,
            true_incremental_audio=true_incremental_audio,
            strict_infer_window=strict_infer_window,
            incremental_backend=incremental_backend,
            stage_timing_log_path=stage_timing_log_path,
            control_prob_trace_log=control_prob_trace_log,
            control_prob_trace_path=control_prob_trace_path,
        )
        worker = threading.Thread(
            target=_run_realtime_session_worker,
            args=(session_id,),
            name=f"realtime_session_{session_id[:8]}",
            daemon=True,
        )
        session.worker = worker

        with _realtime_sessions_lock:
            _realtime_sessions[session_id] = session
        worker.start()

        return JSONResponse(
            {
                "session_id": session_id,
                "infer_window_ms": infer_window_ms,
                "chunk_size_ms": infer_window_ms,
                "start_listen_factor": start_listen_factor,
                "true_incremental_audio": bool(true_incremental_audio),
                "strict_infer_window": bool(strict_infer_window),
                "incremental_backend": incremental_backend,
                "stage_timing_log": bool(stage_timing_log),
                "stage_timing_log_path": stage_timing_log_path,
                "control_prob_trace_log": bool(control_prob_trace_log),
                "control_prob_trace_path": control_prob_trace_path,
                "input_silence_gate": {
                    "enabled": bool(INPUT_SILENCE_GATE_ENABLED),
                    "frame_ms": int(INPUT_SILENCE_GATE_FRAME_MS),
                    "open_dbfs": float(INPUT_SILENCE_GATE_OPEN_DBFS),
                    "close_dbfs": float(INPUT_SILENCE_GATE_CLOSE_DBFS),
                    "hangover_ms": int(INPUT_SILENCE_GATE_HANGOVER_MS),
                    "preroll_ms": int(INPUT_SILENCE_GATE_PREROLL_MS),
                },
            }
        )

    @app.post("/api/realtime/session/{session_id}/chunk")
    async def push_realtime_chunk(session_id: str, request: Request):
        """Receive a realtime audio chunk and append it to the session queue."""
        session = _get_realtime_session_or_404(session_id)
        chunk_recv_epoch_ms = int(time.time() * 1000)
        client_chunk_sent_epoch_ms = None
        client_chunk_sent_header = request.headers.get("x-client-chunk-sent-epoch-ms")
        if client_chunk_sent_header:
            try:
                client_chunk_sent_epoch_ms = int(float(client_chunk_sent_header))
            except (TypeError, ValueError):
                client_chunk_sent_epoch_ms = None
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty chunk body.")

        try:
            chunk_np, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid wav chunk: {exc}") from exc

        if chunk_np.ndim > 1:
            chunk_np = chunk_np.mean(axis=1)
        chunk_np = np.asarray(chunk_np, dtype=np.float32).reshape(-1)
        if chunk_np.size == 0:
            raise HTTPException(status_code=400, detail="Chunk has no samples.")

        sr = int(sr)
        if sr != INPUT_SAMPLE_RATE:
            chunk_np = librosa.resample(chunk_np, orig_sr=sr, target_sr=INPUT_SAMPLE_RATE)
        chunk_np = np.asarray(chunk_np, dtype=np.float32).reshape(-1)

        chunk_muted_samples = 0
        gate_ratio = None
        with session.lock:
            if session.stop_requested:
                raise HTTPException(status_code=409, detail="Session is stopping.")
            if INPUT_SILENCE_GATE_ENABLED:
                chunk_np, gate_stats = _apply_input_silence_gate(
                    chunk_np,
                    session.input_silence_gate_state,
                )
                chunk_muted_samples = int(gate_stats["muted_samples"])
                session.input_silence_gate_total_samples += int(gate_stats["total_samples"])
                session.input_silence_gate_muted_samples += int(gate_stats["muted_samples"])
                if session.input_silence_gate_total_samples > 0:
                    gate_ratio = float(
                        session.input_silence_gate_muted_samples
                        / max(1, session.input_silence_gate_total_samples)
                    )
            new_samples = int(chunk_np.shape[0])
            chunk_start_sample = int(session.total_received_samples)
            chunk_end_sample = int(chunk_start_sample + new_samples)
            session.all_audio_chunks.append(chunk_np)
            session.pending_audio_chunks.append(chunk_np)
            session.pending_samples += new_samples
            session.total_received_samples = int(chunk_end_sample)
            session.last_server_chunk_recv_epoch_ms = int(chunk_recv_epoch_ms)
            if client_chunk_sent_epoch_ms is not None:
                session.last_client_chunk_sent_epoch_ms = int(client_chunk_sent_epoch_ms)
            queued_samples = int(session.pending_samples)

        queued_ms = int(queued_samples * 1000 / INPUT_SAMPLE_RATE)
        return JSONResponse(
            {
                "ok": True,
                "session_id": session_id,
                "received_samples": int(chunk_np.shape[0]),
                "queued_ms": queued_ms,
                "input_silence_gate_enabled": bool(INPUT_SILENCE_GATE_ENABLED),
                "chunk_muted_samples": int(chunk_muted_samples),
                "session_muted_ratio": (float(gate_ratio) if gate_ratio is not None else None),
            }
        )

    @app.get("/api/realtime/session/{session_id}/events")
    async def stream_realtime_events(session_id: str):
        """Stream realtime text, state, and audio events over server-sent events."""
        session = _get_realtime_session_or_404(session_id)

        def event_stream():
            while True:
                try:
                    event = _pop_session_event(session, timeout_sec=0.5)
                except queue.Empty:
                    with session.lock:
                        finished = bool(session.finished)
                    yield "event: heartbeat\ndata: {}\n\n"
                    if finished:
                        break
                    continue

                event_type = str(event.get("type", "message"))
                event_payload = dict(event) if isinstance(event, dict) else {"type": event_type}
                send_epoch_ms = int(time.time() * 1000)
                event_payload["server_sse_send_epoch_ms"] = send_epoch_ms
                if _is_audio_event_type(event_type):
                    try:
                        event_round_id = int(event_payload.get("round_id"))
                        with session.lock:
                            timing_ref = session.round_timing.get(event_round_id)
                            if isinstance(timing_ref, dict):
                                timing_ref.setdefault("first_server_sse_send_epoch_ms", int(send_epoch_ms))
                                timing_ref["last_server_sse_send_epoch_ms"] = int(send_epoch_ms)
                    except (TypeError, ValueError):
                        pass
                emit_epoch_ms = event_payload.get("server_audio_emit_epoch_ms")
                try:
                    if emit_epoch_ms is not None:
                        event_payload["server_queue_delay_ms"] = max(
                            0, int(send_epoch_ms) - int(float(emit_epoch_ms))
                        )
                except (TypeError, ValueError):
                    pass
                _decorate_event_with_unified_frame(session, event_payload, event_type)
                payload = json.dumps(event_payload, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {payload}\n\n"

                if event_type in ("done", "error"):
                    with session.lock:
                        finished = bool(session.finished)
                    if finished:
                        break

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/realtime/session/{session_id}/stop")
    async def stop_realtime_session(session_id: str):
        """Request a realtime session shutdown."""
        session = _get_realtime_session_or_404(session_id)
        with session.lock:
            session.stop_requested = True
        _push_session_event(session, {"type": "status", "status": "Realtime session is stopping..."})
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.post("/api/realtime/session/{session_id}/save_aligned")
    async def save_realtime_aligned_audio(session_id: str, request: Request):
        """Save client-aligned input/output audio and metadata."""
        # Allow saving after stop; session may still be cleaning up. We keep route
        # idempotent and only use session_id for naming.
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

        input_file = form.get("input_wav")
        output_file = form.get("output_wav")
        if input_file is None or output_file is None:
            raise HTTPException(status_code=400, detail="Missing input_wav or output_wav file.")

        try:
            input_bytes = await input_file.read()
            output_bytes = await output_file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {exc}") from exc

        if not input_bytes or not output_bytes:
            raise HTTPException(status_code=400, detail="Uploaded aligned wav is empty.")

        req_sample_rate = form.get("sample_rate", INPUT_SAMPLE_RATE)
        req_input_samples = form.get("input_samples", "")
        req_output_samples = form.get("output_samples", "")
        req_started_at_ms = form.get("started_at_epoch_ms", "")
        req_duration_sec = form.get("aligned_duration_sec", "")

        def _as_int(value):
            try:
                if value is None:
                    return None
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, (int, np.integer)):
                    return int(value)
                text = str(value).strip()
                if text == "":
                    return None
                return int(text)
            except Exception:
                return None

        def _as_float(value):
            try:
                if value is None:
                    return None
                if isinstance(value, (float, int, np.floating, np.integer)):
                    return float(value)
                text = str(value).strip()
                if text == "":
                    return None
                return float(text)
            except Exception:
                return None

        def _ms_to_sec(value):
            number = _as_float(value)
            if number is None:
                return None
            return round(float(number) / 1000.0, 6)

        def _normalize_reason(reason_value):
            if reason_value is None:
                return "unknown"
            reason_text = str(reason_value).strip()
            return reason_text if reason_text else "unknown"

        def _normalize_state_change_item(item):
            if not isinstance(item, dict):
                return item
            out = dict(item)
            out["reason"] = _normalize_reason(out.get("reason"))
            return out

        def _collect_state_changes(round_items, state_items):
            merged = []
            seen = set()

            def _add(item):
                if not isinstance(item, dict):
                    return
                normalized = _normalize_state_change_item(item)
                try:
                    fp = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
                except Exception:
                    fp = str(normalized)
                if fp in seen:
                    return
                seen.add(fp)
                merged.append(normalized)

            if isinstance(state_items, list):
                for state_item in state_items:
                    _add(state_item)

            if isinstance(round_items, list):
                for round_item in round_items:
                    if not isinstance(round_item, dict):
                        continue
                    for state_key in ("state_changes", "state_changes_raw"):
                        state_list = round_item.get(state_key)
                        if not isinstance(state_list, list):
                            continue
                        for state_item in state_list:
                            _add(state_item)

            return merged

        def _ensure_event_trace_fields(event_item):
            if not isinstance(event_item, dict):
                return event_item
            out = dict(event_item)
            start_ts = _as_int(out.get("start_timestamp_epoch_ms"))
            if start_ts is None:
                start_ts = _as_int(out.get("end_timestamp_epoch_ms"))
            if start_ts is None:
                start_ts = int(time.time() * 1000)
            out["start_timestamp_epoch_ms"] = start_ts

            start_rel = _as_float(out.get("start_rel_ms"))
            if start_rel is None:
                start_rel = _as_float(out.get("end_rel_ms"))
            if start_rel is None:
                start_rel = 0.0
            out["start_rel_ms"] = start_rel

            first_text_ts = _as_int(out.get("first_text_token_timestamp_epoch_ms"))
            first_text_rel = _as_float(out.get("first_text_token_rel_ms"))
            first_stoken_ts = _as_int(out.get("first_stoken_timestamp_epoch_ms"))
            first_stoken_rel = _as_float(out.get("first_stoken_rel_ms"))

            if first_text_ts is None:
                out["first_text_token_timestamp_epoch_ms"] = start_ts
                first_text_ts = start_ts
            if first_text_rel is None:
                out["first_text_token_rel_ms"] = start_rel
                first_text_rel = start_rel
            if first_stoken_ts is None:
                out["first_stoken_timestamp_epoch_ms"] = start_ts
                first_stoken_ts = start_ts
            if first_stoken_rel is None:
                out["first_stoken_rel_ms"] = start_rel
                first_stoken_rel = start_rel

            if out.get("first_text_token_observed") is None:
                out["first_text_token_observed"] = False
            if out.get("first_stoken_observed") is None:
                out["first_stoken_observed"] = False

            if (
                _as_float(out.get("first_text_token_latency_ms")) is None
                and start_rel is not None
                and first_text_rel is not None
            ):
                out["first_text_token_latency_ms"] = round(float(first_text_rel) - float(start_rel), 3)
            if (
                _as_float(out.get("first_stoken_latency_ms")) is None
                and start_rel is not None
                and first_stoken_rel is not None
            ):
                out["first_stoken_latency_ms"] = round(float(first_stoken_rel) - float(start_rel), 3)

            state_changes = out.get("state_changes")
            if isinstance(state_changes, list):
                out["state_changes"] = [
                    _normalize_state_change_item(item) if isinstance(item, dict) else item
                    for item in state_changes
                ]
            return out

        def _is_ms_time_key(key):
            if key == "rel_ms":
                return True
            if not isinstance(key, str):
                return False
            return key.endswith("_epoch_ms") or key.endswith("_rel_ms") or key.endswith("_latency_ms") or key.endswith("_duration_ms")

        def _ms_key_to_sec_key(key):
            if key == "rel_ms":
                return "rel_sec"
            if isinstance(key, str) and key.endswith("_ms"):
                return f"{key[:-2]}sec"
            return key

        def _convert_trace_time_to_sec(payload):
            if isinstance(payload, list):
                return [_convert_trace_time_to_sec(item) for item in payload]
            if isinstance(payload, dict):
                out = {}
                for key, value in payload.items():
                    converted_val = _convert_trace_time_to_sec(value)
                    if _is_ms_time_key(key):
                        sec_val = _ms_to_sec(converted_val)
                        if sec_val is not None:
                            converted_val = sec_val
                    out[_ms_key_to_sec_key(key)] = converted_val
                return out
            return payload

        safe_session_id = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_"
            for ch in str(session_id)
        )[:64] or "session"
        save_day = datetime.datetime.now().strftime("%Y%m%d")
        save_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(REALTIME_ALIGNED_SAVE_DIR, save_day)
        os.makedirs(save_dir, exist_ok=True)

        prefix = f"realtime_{save_time}_{safe_session_id}"
        input_filename = f"{prefix}_input_aligned.wav"
        output_filename = f"{prefix}_output_aligned.wav"
        meta_filename = f"{prefix}_meta.json"
        input_path = os.path.join(save_dir, input_filename)
        output_path = os.path.join(save_dir, output_filename)
        meta_path = os.path.join(save_dir, meta_filename)

        try:
            with open(input_path, "wb") as f:
                f.write(input_bytes)
            with open(output_path, "wb") as f:
                f.write(output_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write aligned wav files: {exc}") from exc

        trace_rounds = []
        trace_state_changes = []
        trace_events = []
        session_created_at_ms = None
        session_infer_round = None
        session_total_received_samples = None
        session_final_state = None
        with _realtime_sessions_lock:
            session_obj = _realtime_sessions.get(session_id)
        if session_obj is not None:
            with session_obj.lock:
                session_created_at_ms = _as_int(session_obj.created_at_epoch_ms)
                session_infer_round = _as_int(session_obj.infer_round)
                session_total_received_samples = _as_int(session_obj.total_received_samples)
                session_final_state = str(session_obj.realtime_listening_state)
                try:
                    trace_rounds = json.loads(
                        json.dumps(session_obj.trace_rounds, ensure_ascii=False, default=str)
                    )
                except Exception:
                    trace_rounds = list(session_obj.trace_rounds)
                try:
                    trace_state_changes = json.loads(
                        json.dumps(session_obj.trace_state_changes, ensure_ascii=False, default=str)
                    )
                except Exception:
                    trace_state_changes = list(session_obj.trace_state_changes)
                try:
                    trace_events = json.loads(
                        json.dumps(session_obj.trace_events, ensure_ascii=False, default=str)
                    )
                except Exception:
                    trace_events = list(session_obj.trace_events)

        if isinstance(trace_rounds, list):
            for idx, round_item in enumerate(trace_rounds):
                if not isinstance(round_item, dict):
                    continue
                for state_key in ("state_changes", "state_changes_raw"):
                    state_list = round_item.get(state_key)
                    if isinstance(state_list, list):
                        trace_rounds[idx][state_key] = [
                            _normalize_state_change_item(item) if isinstance(item, dict) else item
                            for item in state_list
                        ]
                round_events = round_item.get("events")
                if isinstance(round_events, list):
                    trace_rounds[idx]["events"] = [
                        _ensure_event_trace_fields(item) if isinstance(item, dict) else item
                        for item in round_events
                    ]

        trace_state_changes = _collect_state_changes(trace_rounds, trace_state_changes)
        if isinstance(trace_events, list):
            trace_events = [
                _ensure_event_trace_fields(event_item) if isinstance(event_item, dict) else event_item
                for event_item in trace_events
            ]

        first_text_token_ts = None
        first_stoken_ts = None
        total_valid_audio_tokens_from_chunks = 0
        total_effective_audio_tokens_from_events = 0
        for event_item in trace_events:
            if not isinstance(event_item, dict):
                continue
            text_ts = _as_int(event_item.get("first_text_token_timestamp_epoch_ms"))
            stoken_ts = _as_int(event_item.get("first_stoken_timestamp_epoch_ms"))
            if text_ts is not None:
                if first_text_token_ts is None or text_ts < first_text_token_ts:
                    first_text_token_ts = text_ts
            if stoken_ts is not None:
                if first_stoken_ts is None or stoken_ts < first_stoken_ts:
                    first_stoken_ts = stoken_ts
            eff_tokens = _as_int(event_item.get("effective_audio_tokens"))
            if eff_tokens is not None and eff_tokens > 0:
                total_effective_audio_tokens_from_events += int(eff_tokens)
            audio_chunks = event_item.get("audio_chunks")
            if isinstance(audio_chunks, list):
                for chunk_item in audio_chunks:
                    if not isinstance(chunk_item, dict):
                        continue
                    valid_tokens = _as_int(chunk_item.get("valid_audio_tokens"))
                    if valid_tokens is not None and valid_tokens > 0:
                        total_valid_audio_tokens_from_chunks += int(valid_tokens)

        first_text_token_ts_sec = _ms_to_sec(first_text_token_ts)
        first_stoken_ts_sec = _ms_to_sec(first_stoken_ts)
        session_created_at_sec = _ms_to_sec(session_created_at_ms)
        started_at_epoch_sec = _ms_to_sec(req_started_at_ms)
        trace_rounds_sec = _convert_trace_time_to_sec(trace_rounds)
        trace_state_changes_sec = _convert_trace_time_to_sec(trace_state_changes)
        trace_events_sec = _convert_trace_time_to_sec(trace_events)

        meta_payload = {
            "session_id": str(session_id),
            "saved_at": datetime.datetime.now().isoformat(),
            "sample_rate": _as_int(req_sample_rate) if _as_int(req_sample_rate) is not None else req_sample_rate,
            "input_samples": _as_int(req_input_samples) if _as_int(req_input_samples) is not None else req_input_samples,
            "output_samples": _as_int(req_output_samples) if _as_int(req_output_samples) is not None else req_output_samples,
            "started_at_epoch_sec": started_at_epoch_sec if started_at_epoch_sec is not None else req_started_at_ms,
            "aligned_duration_sec": _as_float(req_duration_sec) if _as_float(req_duration_sec) is not None else req_duration_sec,
            "input_file": input_filename,
            "output_file": output_filename,
            "input_bytes": int(len(input_bytes)),
            "output_bytes": int(len(output_bytes)),
            "session_trace": {
                "session_created_at_epoch_sec": session_created_at_sec,
                "session_infer_round": session_infer_round,
                "session_total_received_samples": session_total_received_samples,
                "session_final_state": session_final_state,
                "state_change_count": int(len(trace_state_changes)),
                "event_count": int(len(trace_events)),
                "round_trace_count": int(len(trace_rounds)),
                "first_text_token_timestamp_epoch_sec": first_text_token_ts_sec,
                "first_stoken_timestamp_epoch_sec": first_stoken_ts_sec,
                "total_valid_audio_tokens_from_chunks": int(total_valid_audio_tokens_from_chunks),
                "total_effective_audio_tokens_from_events": int(total_effective_audio_tokens_from_events),
                "state_changes": trace_state_changes_sec,
                "events": trace_events_sec,
                "rounds": trace_rounds_sec,
            },
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to write aligned audio metadata json: %s", exc)

        logger.info(
            "Saved aligned realtime audio: session=%s input=%s output=%s",
            session_id,
            input_path,
            output_path,
        )
        return JSONResponse(
            {
                "ok": True,
                "session_id": session_id,
                "save_dir": save_dir,
                "input_path": input_path,
                "output_path": output_path,
                "meta_path": meta_path,
                "input_bytes": int(len(input_bytes)),
                "output_bytes": int(len(output_bytes)),
            }
        )


# ==================== Gradio Compatibility UI ====================

def build_demo():
    """Build the Gradio compatibility UI and mount realtime API routes."""
    with gr.Blocks(title="Lychee-FD Realtime Full-Duplex Demo") as demo:

        gr.HTML("""
        <div class="main-title">
            <h1>🎙️ Lychee-FD Realtime Full-Duplex Demo</h1>
        </div>
        <div class="subtitle">
            <p>The model continuously listens and decides when to speak in an online full-duplex loop.</p>
            <p>Upload audio to inspect chunk-level model state, text, and synthesized speech.</p>
        </div>
        """)

        with gr.Accordion("⚙️ Model Loading", open=True):
            with gr.Row():
                model_path_input = gr.Textbox(
                    label="Model path", value=DEFAULT_MODEL_PATH, scale=3,
                )
                token2wav_path_input = gr.Textbox(
                    label="Token2Wav path", value=DEFAULT_TOKEN2WAV_PATH, scale=3,
                )
                load_btn = gr.Button("🚀 Load Models", variant="primary", scale=1)
            load_status = gr.Textbox(label="Load status", interactive=False, lines=3)
            load_btn.click(
                fn=load_models,
                inputs=[model_path_input, token2wav_path_input],
                outputs=[load_status],
            )

        with gr.Accordion("🔧 Inference Parameters", open=False):
            with gr.Row():
                start_speak_factor = gr.Slider(
                    minimum=0.0, maximum=20.0, value=1.2, step=0.1,
                    label="Start-speak factor",
                    info="Added to the SS-token logit. Higher values make the model more likely to speak.",
                )
                end_speak_factor = gr.Slider(
                    minimum=0.1, maximum=10, value=1.0, step=0.1,
                    label="End-speak factor",
                    info="Higher values make the model more likely to stop speaking earlier.",
                )
                prompt_voice = gr.Radio(
                    choices=["male", "female"], value="male", label="Prompt voice",
                )
                tts_chunk_size = gr.Slider(
                    minimum=5, maximum=40, value=5, step=1,
                    label="Decoder chunk size (stokens)",
                    info="Controls upstream StreamingDecoder flush frequency only. Smaller values lower first-packet latency but increase scheduling overhead.",
                )

        gr.Markdown("### 📁 Upload Audio")
        gr.Markdown("""
        **Usage**:
        1. Upload an audio file.
        2. Click **Run Inference** to process audio in online chunks.
        3. Inspect the synthesized speech, chat view, and raw JSON result.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                upload_input = gr.Audio(
                    label="🎤 Audio input",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                infer_btn = gr.Button("▶️ Run Inference", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", value="Waiting for audio...", interactive=False)

                audio_debug_display = gr.Textbox(
                    label="🔬 Audio-token / RTF Diagnostics",
                    value="Diagnostics will appear during inference.",
                    interactive=False,
                    lines=5,
                )

                output_audio = gr.Audio(
                    label="🔊 Synthesized response",
                    type="numpy",
                    interactive=False,
                    autoplay=True,
                    streaming=True,
                )
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="📜 Chunk-level Dialogue",
                    height=600,
                )

        with gr.Accordion("📋 Raw JSON", open=False):
            json_output = gr.Code(label="Raw JSON", language="json", interactive=False)

        infer_btn.click(
            fn=lambda: ([], None, "Preparing inference...", "", "(preparing...)"),
            inputs=None,
            outputs=[chatbot, output_audio, status_text, json_output, audio_debug_display],
            queue=False,
            api_name="prepare_run_chunk_dialogue_inference",
        ).then(
            fn=run_chunk_dialogue_inference,
            inputs=[upload_input, start_speak_factor, end_speak_factor, prompt_voice, tts_chunk_size],
            outputs=[chatbot, output_audio, status_text, json_output, audio_debug_display],
            api_name="run_chunk_dialogue_inference",
        )

        def clear_reply():
            """Reset the compatibility UI outputs."""
            return [], None, "Waiting for audio...", "", "(cleared)"

        clear_btn.click(
            fn=clear_reply,
            inputs=[],
            outputs=[chatbot, output_audio, status_text, json_output, audio_debug_display],
            api_name="clear_run_chunk_dialogue_inference",
        )

        example_files = []
        if os.path.exists(ASSETS_DIR):
            for f in sorted(os.listdir(ASSETS_DIR)):
                if f.endswith(".wav") and "default" not in f:
                    example_files.append(os.path.join(ASSETS_DIR, f))
        if example_files:
            with gr.Accordion("🎵 Example Audio", open=False):
                gr.Examples(
                    examples=[[f] for f in example_files[:5]],
                    inputs=[upload_input],
                    label="Select an example audio file",
                )

        with gr.Accordion("📖 Notes", open=False):
            gr.Markdown("""
## Full-Duplex Dialogue Model

### Features
- **Full-duplex dialogue**: the model keeps listening and decides when to speak through the control channel.
- **Three-channel generation**: text, speech tokens, and control signals are generated together.
- **Online mode**: audio is fed directly without extra padding.

### Dialogue View

| Role | Meaning |
|------|------|
| User | Input-audio chunk metadata |
| Model | Model state changes and generated content |

### Control Signals
| Signal | Meaning |
|------|------|
| K-L (Keep Listening) | Continue listening |
| S-S (Start Speaking) | Start speaking |
| S-L (Start Listening) | Stop speaking and return to listening |
| K-S (Keep Speaking) | Continue speaking |
| B-C (Backchannel) | Backchannel response |
            """)

    return demo


# ==================== Entrypoint ====================
if __name__ == "__main__":
    # CLI entrypoint for local serving.
    parser = argparse.ArgumentParser(description="Lychee-FD full-duplex realtime service")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--token2wav_path", type=str, default=DEFAULT_TOKEN2WAV_PATH)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    # Keep share enabled by default for quick remote debugging.
    parser.add_argument("--share", dest="share", action="store_true", help="Enable temporary public Gradio URL")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable public Gradio URL")
    parser.set_defaults(share=True)
    parser.add_argument("--auto_load", default=True, help="Load models at startup")
    args = parser.parse_args()

    DEFAULT_MODEL_PATH = args.model_path
    DEFAULT_TOKEN2WAV_PATH = args.token2wav_path

    if args.auto_load:
        logger.info("Auto-loading models...")
        status = load_models(args.model_path, args.token2wav_path)
        logger.info(status)

    demo = build_demo()
    demo.queue()
    # Gradio 6.x rebuilds the app inside launch(), so register custom routes after launch().
    app, _local_url, _share_url = demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        strict_cors=False,
        allowed_paths=[TEMP_DIR, REALTIME_ALIGNED_SAVE_DIR],
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #666; font-size: 14px; margin-bottom: 15px; }
        """,
        prevent_thread_lock=True,
    )
    register_realtime_session_routes(app)
    demo.block_thread()
