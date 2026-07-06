#!/usr/bin/env python3
"""
Batch runner for the weight-test realtime route.

It mirrors unimoe_demo/src/App_weight_test.vue:
  input.wav -> mono 16 kHz -> fixed chunks -> realtime session/chunk/events.

The runner writes per-case logs after each case finishes, avoiding per-event
disk writes on the model path. It does not modify or instrument the backend.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

try:
    import scipy.signal
except Exception:  # noqa: BLE001
    scipy = None


TARGET_SAMPLE_RATE = 16000
TTS_TOKEN_HZ = 25
MIN_SEGMENT_SAMPLES = 800
DEFAULT_DATA_ROOT = (
    os.environ.get(
        "FULL_DUPLEX_USER_INTERRUPTION_DATA_ROOT",
        "datasets/SoulX-Duplug-Eval/Full-Duplex-Bench-zh/user_interruption",
    )
)
DEFAULT_OUTPUT_ROOT = (
    os.environ.get("STEPAUDIO_WEIGHT_BATCH_OUTPUT_ROOT", "batch_weight_runs")
)


def now_ms() -> int:
    return int(time.time() * 1000)


def perf_ms() -> float:
    return time.perf_counter() * 1000.0


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default))
            f.write("\n")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_audio_mono_16k(path: Path) -> tuple[np.ndarray, int, int]:
    samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.size <= 0:
        raise ValueError(f"empty audio: {path}")
    mono = samples.mean(axis=1).astype(np.float32, copy=False)
    source_rate = int(source_rate)
    if source_rate == TARGET_SAMPLE_RATE:
        return mono.astype(np.float32, copy=False), source_rate, int(samples.shape[1])
    # scipy.signal.resample_poly is deterministic and close enough for the
    # browser's AudioContext resampling used by weight-test.html.
    g = math.gcd(source_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = source_rate // g
    resampled = scipy.signal.resample_poly(mono, up, down).astype(np.float32, copy=False)
    return resampled, source_rate, int(samples.shape[1])


def append_tail_silence(samples: np.ndarray, silence_sec: float) -> np.ndarray:
    pad = max(0, int(round(float(silence_sec) * TARGET_SAMPLE_RATE)))
    if pad <= 0:
        return samples.astype(np.float32, copy=False)
    out = np.zeros(int(samples.shape[0]) + pad, dtype=np.float32)
    out[: samples.shape[0]] = samples
    return out


def split_samples(samples: np.ndarray, chunk_ms: int) -> list[np.ndarray]:
    seg_len = max(MIN_SEGMENT_SAMPLES, int(math.floor(TARGET_SAMPLE_RATE * chunk_ms / 1000.0)))
    out: list[np.ndarray] = []
    for start in range(0, int(samples.shape[0]), seg_len):
        piece = samples[start : start + seg_len].astype(np.float32, copy=True)
        if piece.shape[0] < MIN_SEGMENT_SAMPLES and out:
            prev = out.pop()
            out.append(np.concatenate([prev, piece]).astype(np.float32, copy=False))
        else:
            out.append(piece)
    return [x for x in out if x.shape[0] >= MIN_SEGMENT_SAMPLES]


def estimate_chunk_count(frames: int, sample_rate: int, tail_padding_sec: float, chunk_ms: int) -> int:
    samples_16k = max(0, int(round(float(frames) * TARGET_SAMPLE_RATE / float(sample_rate))))
    samples_16k += max(0, int(round(float(tail_padding_sec) * TARGET_SAMPLE_RATE)))
    if samples_16k <= 0:
        return 0
    seg_len = max(MIN_SEGMENT_SAMPLES, int(math.floor(TARGET_SAMPLE_RATE * chunk_ms / 1000.0)))
    full, rem = divmod(samples_16k, seg_len)
    if rem <= 0:
        return full
    if rem < MIN_SEGMENT_SAMPLES and full > 0:
        return full
    return full + 1


def pcm16_from_float32(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    out = np.empty(clipped.shape[0], dtype="<i2")
    neg = clipped < 0
    out[neg] = np.round(clipped[neg] * 32768.0).astype("<i2")
    out[~neg] = np.round(clipped[~neg] * 32767.0).astype("<i2")
    return out


def wav_bytes_mono(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    import io
    import wave

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16_from_float32(samples).tobytes())
    return bio.getvalue()


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32, copy=False), int(sample_rate), subtype="PCM_16")


def write_wav_stereo(path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(int(left.shape[0]), int(right.shape[0]))
    stereo = np.zeros((frame_count, 2), dtype=np.float32)
    stereo[: left.shape[0], 0] = left[:frame_count]
    stereo[: right.shape[0], 1] = right[:frame_count]
    sf.write(str(path), stereo, int(sample_rate), subtype="PCM_16")


def decode_pcm16_b64(pcm_b64: str, channels: int = 1) -> np.ndarray:
    raw = base64.b64decode(pcm_b64)
    usable = len(raw) - (len(raw) % 2)
    arr = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32)
    ch = max(1, int(channels or 1))
    if ch > 1:
        frames = arr.shape[0] // ch
        arr = arr[: frames * ch].reshape(frames, ch).mean(axis=1)
    return np.where(arr < 0, arr / 32768.0, arr / 32767.0).astype(np.float32, copy=False)


def decode_wav_b64(wav_b64: str) -> tuple[np.ndarray, int]:
    import io

    raw = base64.b64decode(wav_b64)
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32, copy=False), int(sr)


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if int(sample_rate) == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    g = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
    return scipy.signal.resample_poly(
        samples.astype(np.float32, copy=False),
        TARGET_SAMPLE_RATE // g,
        int(sample_rate) // g,
    ).astype(np.float32, copy=False)


def event_without_audio(payload: dict) -> dict:
    out = dict(payload)
    for key in ("pcm_b64", "wav_b64"):
        if isinstance(out.get(key), str):
            out[key] = f"<omitted:{len(out[key])} chars>"
    frame_audio = out.get("frame_audio")
    if isinstance(frame_audio, dict):
        frame_audio = dict(frame_audio)
        for key in ("pcm_b64", "wav_b64"):
            if isinstance(frame_audio.get(key), str):
                frame_audio[key] = f"<omitted:{len(frame_audio[key])} chars>"
        out["frame_audio"] = frame_audio
    return out


def normalize_event_payload(payload: dict, sse_event_type: str = "message") -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    raw_type = payload.get("type") if isinstance(payload.get("type"), str) else ""
    unified_type = payload.get("event_type") if isinstance(payload.get("event_type"), str) else ""
    event_type = unified_type or raw_type or sse_event_type or "message"
    normalized = dict(payload)
    normalized["type"] = event_type

    frame_text = payload.get("frame_text")
    if event_type == "assistant_text" and isinstance(frame_text, dict):
        normalized["text"] = normalized.get("text") or frame_text.get("delta") or frame_text.get("snapshot") or ""
        normalized["snapshot"] = normalized.get("snapshot") or frame_text.get("snapshot") or ""
        normalized["delta"] = normalized.get("delta") or frame_text.get("delta") or ""
        normalized["event_id"] = normalized.get("event_id") or frame_text.get("event_id") or ""
        if "is_final" not in normalized:
            normalized["is_final"] = frame_text.get("is_final")

    frame_audio = payload.get("frame_audio")
    if isinstance(frame_audio, dict) and event_type in {"audio_chunk_pcm", "audio_chunk"}:
        fmt = str(frame_audio.get("format") or "").lower()
        if fmt == "pcm_s16le" or frame_audio.get("pcm_b64"):
            normalized["type"] = "audio_chunk_pcm"
            normalized["pcm_b64"] = normalized.get("pcm_b64") or frame_audio.get("pcm_b64")
            normalized["sample_rate"] = normalized.get("sample_rate") or frame_audio.get("sample_rate")
            normalized["num_channels"] = normalized.get("num_channels") or frame_audio.get("num_channels")
        elif fmt == "wav_b64" or frame_audio.get("wav_b64"):
            normalized["type"] = "audio_chunk"
            normalized["wav_b64"] = normalized.get("wav_b64") or frame_audio.get("wav_b64")
            normalized["sample_rate"] = normalized.get("sample_rate") or frame_audio.get("sample_rate")
    return normalized


def apply_text_event(state: dict, event: dict) -> None:
    delta = event.get("delta") if isinstance(event.get("delta"), str) else event.get("text")
    if not isinstance(delta, str):
        delta = ""
    snapshot = event.get("snapshot") if isinstance(event.get("snapshot"), str) else ""
    event_id = event.get("event_id") if isinstance(event.get("event_id"), str) and event.get("event_id") else "default"
    current = str(state.get("response_text") or "").replace("\r", "")
    current_id = str(state.get("current_text_event_id") or "")
    current_snapshot = str(state.get("current_text_event_snapshot") or "")

    if snapshot:
        next_text = snapshot.replace("\r", "").strip()
        same_event = current_id == event_id
        prev = current_snapshot if same_event else ""
        if not next_text:
            return
        if same_event and prev and next_text.startswith(prev):
            suffix = next_text[len(prev) :]
            if suffix:
                state["response_text"] = f"{current}{suffix}"
            state["current_text_event_snapshot"] = next_text
            return
        if same_event and prev and prev.startswith(next_text):
            return
        state["response_text"] = f"{current}\n\n{next_text}" if current else next_text
        state["current_text_event_id"] = event_id
        state["current_text_event_snapshot"] = next_text
        return

    if delta:
        clean = delta.replace("\r", "")
        state["response_text"] = f"{current}{clean}"
        state["current_text_event_id"] = event_id
        state["current_text_event_snapshot"] = f"{current_snapshot}{clean}"


@dataclass
class CaseArchive:
    start_perf_ms: float = 0.0
    start_epoch_ms: int = 0
    input_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    output_chunks: list[np.ndarray] = field(default_factory=list)
    raw_output_chunks: list[np.ndarray] = field(default_factory=list)
    output_samples: int = 0
    raw_output_samples: int = 0
    aligned_samples: int = 0
    has_output: bool = False
    last_audio_arrival_perf_ms: Optional[float] = None
    last_audio_chunk_duration_ms: float = 0.0


class RealtimeBatchClient:
    def __init__(
        self,
        api_base: str,
        *,
        timeout_sec: float = 30.0,
        connect_timeout_sec: float = 10.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.http = requests.Session()

    def url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    def health(self) -> dict:
        resp = self.http.get(self.url("/api/realtime/voices"), timeout=(self.connect_timeout_sec, self.timeout_sec))
        resp.raise_for_status()
        return resp.json()

    def create_session(self, cfg: dict) -> dict:
        payload = {
            "start_speak_factor": cfg["start_speak_factor"],
            "start_listen_factor": cfg["start_listen_factor"],
            "end_speak_factor": cfg["end_speak_factor"],
            "prompt_voice": cfg["prompt_voice"],
            "tts_chunk_size": cfg["tts_chunk_size"],
            "infer_window_ms": cfg["infer_window_ms"],
            "stage_timing_log": False,
        }
        resp = self.http.post(
            self.url("/api/realtime/session/start"),
            json=payload,
            timeout=(self.connect_timeout_sec, self.timeout_sec),
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("session_id"):
            raise RuntimeError(f"session/start returned no session_id: {data}")
        return data

    def send_chunk(self, session_id: str, samples: np.ndarray, sent_epoch_ms: int) -> None:
        wav = wav_bytes_mono(samples, TARGET_SAMPLE_RATE)
        headers = {
            "Content-Type": "audio/wav",
            "X-Client-Chunk-Sent-Epoch-Ms": str(int(sent_epoch_ms)),
        }
        resp = self.http.post(
            self.url(f"/api/realtime/session/{session_id}/chunk"),
            headers=headers,
            data=wav,
            timeout=(self.connect_timeout_sec, self.timeout_sec),
        )
        if not resp.ok:
            raise RuntimeError(f"chunk upload failed {resp.status_code}: {resp.text[:500]}")

    def stop_session(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            self.http.post(
                self.url(f"/api/realtime/session/{session_id}/stop"),
                timeout=(self.connect_timeout_sec, self.timeout_sec),
            )
        except Exception:
            pass

    def stream_events(self, session_id: str):
        with self.http.get(
            self.url(f"/api/realtime/session/{session_id}/events"),
            stream=True,
            timeout=(self.connect_timeout_sec, None),
        ) as resp:
            resp.raise_for_status()
            event_type = "message"
            data_lines: list[str] = []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        yield event_type, raw_data
                    event_type = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())


def percentile(values: list[float], p: float) -> Optional[float]:
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, math.ceil((p / 100.0) * len(vals)) - 1))
    return vals[idx]


def average(values: list[float]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def stage_row(payload: dict) -> dict:
    stages = payload.get("stages") or {}
    latency = payload.get("latency") or {}
    queue_meta = payload.get("queue") or {}
    return {
        "round": payload.get("round_id"),
        "state": payload.get("state"),
        "input_sec": payload.get("input_duration_sec"),
        "started_at_ms": payload.get("round_started_at_epoch_ms"),
        "completed_at_ms": payload.get("round_completed_at_epoch_ms"),
        "queue_wait_ms": stages.get("input_queue_wait"),
        "preprocess_ms": stages.get("audio_preprocess"),
        "encoder_ms": stages.get("audio_encoder"),
        "transformer_ms": stages.get("transformer"),
        "token2wav_ms": stages.get("token2wav"),
        "round_to_first_audio_emit_ms": stages.get("round_to_first_audio_emit"),
        "output_queue_send_ms": stages.get("output_queue_send"),
        "first_pcm_ms": latency.get("first_pcm_out_ms"),
        "first_t2w_submit_ms": latency.get("first_t2w_submit_ms"),
        "t2w_submit_to_first_pcm_ms": latency.get("t2w_submit_to_first_pcm_ms"),
        "stream_infer_ms": latency.get("stream_infer_ms"),
        "total_round_ms": latency.get("total_round_ms"),
        "q_before_ms": queue_meta.get("pending_before_ms"),
        "q_consumed_ms": queue_meta.get("consumed_ms"),
        "q_after_ms": queue_meta.get("pending_after_ms"),
        "q_window_ms": queue_meta.get("infer_window_ms"),
        "raw": payload,
    }


def t2w_row(payload: dict, idx: int, recv_epoch_ms: int) -> Optional[dict]:
    if payload.get("t2w_synth_start_epoch_ms") is None and payload.get("t2w_synth_duration_ms") is None:
        return None
    advance = float(payload.get("t2w_advance_tokens") or 0)
    audio_ms = advance / TTS_TOKEN_HZ * 1000.0 if advance > 0 else None
    dur_ms = payload.get("t2w_synth_duration_ms")
    dur_ms_f = float(dur_ms) if dur_ms is not None else None
    return {
        "idx": idx,
        "start_ms": payload.get("t2w_synth_start_epoch_ms"),
        "end_ms": payload.get("t2w_synth_end_epoch_ms"),
        "dur_ms": dur_ms_f,
        "tokens": payload.get("t2w_tokens"),
        "advance": advance,
        "audio_ms": audio_ms,
        "rtf": (dur_ms_f / audio_ms) if dur_ms_f is not None and audio_ms and audio_ms > 0 else None,
        "roundtrip_ms": payload.get("t2w_remote_roundtrip_ms"),
        "backend": payload.get("t2w_backend"),
        "recv_epoch_ms": recv_epoch_ms,
    }


def record_output_chunk(
    archive: CaseArchive,
    samples: np.ndarray,
    sample_rate: int,
    *,
    stutter_threshold_ms: float,
    t2w_trace: list[dict],
    stutter_events: list[dict],
    first_pcm_state: dict,
) -> None:
    if samples.size <= 0:
        return
    if not first_pcm_state.get("recorded"):
        first_pcm_state["recorded"] = True
        first_sent = first_pcm_state.get("first_sent_epoch_ms")
        if first_sent is not None:
            first_pcm_state["first_pcm_e2e_ms"] = now_ms() - int(first_sent)

    arrival_perf = perf_ms()
    normalized = resample_to_16k(samples, int(sample_rate))
    chunk_duration_ms = normalized.shape[0] / TARGET_SAMPLE_RATE * 1000.0
    raw_chunk = normalized.astype(np.float32, copy=True)
    archive.raw_output_chunks.append(raw_chunk)
    archive.raw_output_samples += int(raw_chunk.shape[0])

    elapsed_samples = max(0, int(round(((arrival_perf - archive.start_perf_ms) / 1000.0) * TARGET_SAMPLE_RATE)))
    gap_samples = max(0, elapsed_samples - archive.output_samples)
    stutter_gap_ms = 0.0
    arrival_delta_ms = None
    if archive.has_output and archive.last_audio_arrival_perf_ms is not None:
        arrival_delta_ms = arrival_perf - archive.last_audio_arrival_perf_ms
        stutter_gap_ms = max(0.0, arrival_delta_ms - archive.last_audio_chunk_duration_ms)
    if gap_samples > 0:
        align_gap_ms = gap_samples / TARGET_SAMPLE_RATE * 1000.0
        if stutter_gap_ms >= float(stutter_threshold_ms):
            stutter_events.append(
                {
                    "index": len(stutter_events) + 1,
                    "gap_ms": stutter_gap_ms,
                    "align_gap_ms": align_gap_ms,
                    "arrival_delta_ms": arrival_delta_ms,
                    "prev_chunk_ms": archive.last_audio_chunk_duration_ms,
                    "at_ms": archive.output_samples / TARGET_SAMPLE_RATE * 1000.0,
                    "wall_rel_s": (now_ms() - (archive.start_epoch_ms or now_ms())) / 1000.0,
                    "near_t2w_dur_ms": t2w_trace[-1].get("dur_ms") if t2w_trace else None,
                    "near_t2w_rtf": t2w_trace[-1].get("rtf") if t2w_trace else None,
                }
            )
        archive.output_chunks.append(np.zeros(gap_samples, dtype=np.float32))
        archive.output_samples += int(gap_samples)

    archive.output_chunks.append(raw_chunk)
    archive.output_samples += int(raw_chunk.shape[0])
    archive.aligned_samples = max(archive.aligned_samples, archive.output_samples)
    archive.has_output = True
    archive.last_audio_arrival_perf_ms = arrival_perf
    archive.last_audio_chunk_duration_ms = chunk_duration_ms


def run_event_consumer(
    client: RealtimeBatchClient,
    session_id: str,
    result: dict,
    archive: CaseArchive,
    stutter_threshold_ms: float,
    first_pcm_state: dict,
    stop_event: threading.Event,
    errors: queue.Queue,
) -> None:
    try:
        for event_type, raw_data in client.stream_events(session_id):
            if stop_event.is_set():
                break
            if not raw_data or event_type == "heartbeat":
                continue
            try:
                raw_payload = json.loads(raw_data)
            except json.JSONDecodeError:
                result["events"].append({"type": "parse_error", "raw": raw_data[:500]})
                continue
            recv_epoch_ms = now_ms()

            if isinstance(raw_payload, dict) and raw_payload.get("type") == "stage_timing":
                result["stage_timing"].append(raw_payload)
                result["stage_rows"].append(stage_row(raw_payload))
                continue

            payload = normalize_event_payload(raw_payload, event_type)
            if not payload:
                continue
            result["events"].append(event_without_audio(payload))

            ptype = payload.get("type")
            if ptype == "error":
                raise RuntimeError(str(payload.get("error") or raw_data))
            if ptype == "assistant_text":
                apply_text_event(result["text_state"], payload)
            elif ptype == "state_change":
                result["state_changes"].append(payload)
            elif ptype == "audio_chunk_pcm" and payload.get("pcm_b64"):
                row = t2w_row(payload, len(result["t2w_trace"]) + 1, recv_epoch_ms)
                if row:
                    result["t2w_trace"].append(row)
                samples = decode_pcm16_b64(str(payload.get("pcm_b64")), int(payload.get("num_channels") or 1))
                record_output_chunk(
                    archive,
                    samples,
                    int(payload.get("sample_rate") or TARGET_SAMPLE_RATE),
                    stutter_threshold_ms=stutter_threshold_ms,
                    t2w_trace=result["t2w_trace"],
                    stutter_events=result["stutter_events"],
                    first_pcm_state=first_pcm_state,
                )
            elif ptype == "audio_chunk" and payload.get("wav_b64"):
                samples, sample_rate = decode_wav_b64(str(payload.get("wav_b64")))
                record_output_chunk(
                    archive,
                    samples,
                    sample_rate,
                    stutter_threshold_ms=stutter_threshold_ms,
                    t2w_trace=result["t2w_trace"],
                    stutter_events=result["stutter_events"],
                    first_pcm_state=first_pcm_state,
                )
            elif ptype == "done":
                result["done_seen"] = True
                break
    except Exception as exc:  # noqa: BLE001
        errors.put(exc)


def summarize_case(
    *,
    case_id: str,
    input_path: Path,
    output_dir: Path,
    session_id: str,
    input_samples: np.ndarray,
    sent_samples: np.ndarray,
    source_rate: int,
    source_channels: int,
    segments: list[np.ndarray],
    archive: CaseArchive,
    result: dict,
    cfg: dict,
    started_epoch_ms: int,
    finished_epoch_ms: int,
    input_sha1: str,
) -> dict:
    stage_rows = result["stage_rows"]
    speaking_rows = [r for r in stage_rows if r.get("state") == "speaking"]
    t2w_trace = result["t2w_trace"]
    stutters = result["stutter_events"]

    def vals(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]

    def stats(prefix: str, rows: list[dict], key: str, out: dict) -> None:
        arr = vals(rows, key)
        out[f"{prefix}_{key}_avg"] = average(arr)
        out[f"{prefix}_{key}_p95"] = percentile(arr, 95)
        out[f"{prefix}_{key}_max"] = max(arr) if arr else None

    summary: dict[str, Any] = {
        "case_id": case_id,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "session_id": session_id,
        "input_sha1": input_sha1,
        "started_epoch_ms": started_epoch_ms,
        "finished_epoch_ms": finished_epoch_ms,
        "wall_time_sec": round((finished_epoch_ms - started_epoch_ms) / 1000.0, 3),
        "source_rate": source_rate,
        "source_channels": source_channels,
        "input_sec": round(input_samples.shape[0] / TARGET_SAMPLE_RATE, 6),
        "sent_sec": round(sent_samples.shape[0] / TARGET_SAMPLE_RATE, 6),
        "tail_padding_sec": cfg["tail_padding_sec"],
        "chunk_ms": cfg["chunk_ms"],
        "infer_window_ms": cfg["infer_window_ms"],
        "chunk_count": len(segments),
        "response_text_len": len(result["text_state"].get("response_text") or ""),
        "stage_round_count": len(stage_rows),
        "speaking_round_count": len(speaking_rows),
        "t2w_count": len(t2w_trace),
        "stutter_count": len(stutters),
        "max_stutter_ms": max((float(x.get("gap_ms") or 0) for x in stutters), default=0.0),
        "first_pcm_e2e_ms": result["first_pcm_state"].get("first_pcm_e2e_ms"),
        "raw_output_sec": round(archive.raw_output_samples / TARGET_SAMPLE_RATE, 6),
        "aligned_sec": round(max(sent_samples.shape[0], archive.output_samples) / TARGET_SAMPLE_RATE, 6),
        "done_seen": bool(result.get("done_seen")),
    }

    for key in (
        "queue_wait_ms",
        "preprocess_ms",
        "encoder_ms",
        "transformer_ms",
        "token2wav_ms",
        "first_pcm_ms",
        "total_round_ms",
        "q_before_ms",
        "q_consumed_ms",
        "q_after_ms",
    ):
        stats("all", stage_rows, key, summary)
        stats("speaking", speaking_rows, key, summary)

    t2w_rtfs = [float(x["rtf"]) for x in t2w_trace if x.get("rtf") is not None and math.isfinite(float(x["rtf"]))]
    t2w_durs = [float(x["dur_ms"]) for x in t2w_trace if x.get("dur_ms") is not None and math.isfinite(float(x["dur_ms"]))]
    summary.update(
        {
            "t2w_rtf_avg": average(t2w_rtfs),
            "t2w_rtf_p95": percentile(t2w_rtfs, 95),
            "t2w_rtf_max": max(t2w_rtfs) if t2w_rtfs else None,
            "t2w_rtf_over_1_count": sum(1 for x in t2w_rtfs if x > 1.0),
            "t2w_dur_avg_ms": average(t2w_durs),
            "t2w_dur_p95_ms": percentile(t2w_durs, 95),
            "t2w_dur_max_ms": max(t2w_durs) if t2w_durs else None,
        }
    )
    return summary


def save_case_outputs(
    case_dir: Path,
    *,
    summary: dict,
    input_samples: np.ndarray,
    sent_samples: np.ndarray,
    archive: CaseArchive,
    result: dict,
    save_audio: bool,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "summary.json", summary)
    write_json(case_dir / "input_meta.json", {
        k: summary[k]
        for k in (
            "case_id",
            "input_path",
            "input_sha1",
            "source_rate",
            "source_channels",
            "input_sec",
            "sent_sec",
            "tail_padding_sec",
            "chunk_ms",
            "infer_window_ms",
            "chunk_count",
        )
    })
    write_jsonl(case_dir / "stage_timing.jsonl", result["stage_timing"])
    write_jsonl(case_dir / "stage_rows.jsonl", result["stage_rows"])
    write_jsonl(case_dir / "t2w_trace.jsonl", result["t2w_trace"])
    write_jsonl(case_dir / "state_changes.jsonl", [event_without_audio(x) for x in result["state_changes"]])
    write_jsonl(case_dir / "events.jsonl", result["events"])
    write_jsonl(case_dir / "stutter_events.jsonl", result["stutter_events"])
    (case_dir / "response.txt").write_text(result["text_state"].get("response_text") or "", encoding="utf-8")

    if not save_audio:
        return
    raw = np.concatenate(archive.raw_output_chunks).astype(np.float32, copy=False) if archive.raw_output_chunks else np.zeros(0, dtype=np.float32)
    aligned_len = max(int(sent_samples.shape[0]), int(archive.output_samples))
    right = np.concatenate(archive.output_chunks).astype(np.float32, copy=False) if archive.output_chunks else np.zeros(0, dtype=np.float32)
    left = np.zeros(aligned_len, dtype=np.float32)
    left[: min(aligned_len, sent_samples.shape[0])] = sent_samples[:aligned_len]
    right_aligned = np.zeros(aligned_len, dtype=np.float32)
    right_aligned[: min(aligned_len, right.shape[0])] = right[:aligned_len]
    write_wav_mono(case_dir / "input_sent.wav", sent_samples)
    write_wav_mono(case_dir / "raw_output.wav", raw)
    write_wav_stereo(case_dir / "aligned_stereo.wav", left, right_aligned)


def run_case(
    client: RealtimeBatchClient,
    input_path: Path,
    case_id: str,
    case_dir: Path,
    cfg: dict,
    *,
    realtime_send: bool,
    save_audio: bool,
) -> dict:
    started_epoch_ms = now_ms()
    input_samples, source_rate, source_channels = read_audio_mono_16k(input_path)
    if input_samples.shape[0] < MIN_SEGMENT_SAMPLES:
        raise RuntimeError(f"audio too short: {input_path}")
    sent_samples = append_tail_silence(input_samples, cfg["tail_padding_sec"])
    segments = split_samples(sent_samples, cfg["chunk_ms"])
    if not segments:
        raise RuntimeError(f"cannot split audio into chunks: {input_path}")

    input_wav_bytes = wav_bytes_mono(sent_samples, TARGET_SAMPLE_RATE)
    input_sha1 = sha1_bytes(input_wav_bytes)

    session_payload = client.create_session(cfg)
    session_id = str(session_payload["session_id"])
    archive = CaseArchive(input_samples=sent_samples, aligned_samples=int(sent_samples.shape[0]))
    archive.start_perf_ms = perf_ms()
    archive.start_epoch_ms = now_ms()

    result = {
        "stage_timing": [],
        "stage_rows": [],
        "t2w_trace": [],
        "state_changes": [],
        "events": [],
        "stutter_events": [],
        "text_state": {
            "response_text": "",
            "current_text_event_id": "",
            "current_text_event_snapshot": "",
        },
        "done_seen": False,
    }
    first_pcm_state = {
        "recorded": False,
        "first_sent_epoch_ms": None,
        "first_pcm_e2e_ms": None,
    }
    result["first_pcm_state"] = first_pcm_state

    stop_event = threading.Event()
    errors: queue.Queue = queue.Queue()
    consumer = threading.Thread(
        target=run_event_consumer,
        args=(
            client,
            session_id,
            result,
            archive,
            cfg["stutter_threshold_ms"],
            first_pcm_state,
            stop_event,
            errors,
        ),
        name=f"events_{case_id}",
        daemon=True,
    )
    consumer.start()

    try:
        for idx, segment in enumerate(segments, start=1):
            if not errors.empty():
                raise errors.get()
            if realtime_send:
                planned = archive.start_perf_ms + (idx - 1) * cfg["chunk_ms"]
                wait_ms = planned - perf_ms()
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000.0)
            sent_epoch_ms = now_ms()
            if idx == 1:
                first_pcm_state["first_sent_epoch_ms"] = sent_epoch_ms
            client.send_chunk(session_id, segment, sent_epoch_ms)

        client.stop_session(session_id)
        consumer.join(timeout=cfg["done_timeout_sec"])
        if consumer.is_alive():
            stop_event.set()
            raise TimeoutError(f"timed out waiting for done after {cfg['done_timeout_sec']}s")
        if not errors.empty():
            raise errors.get()
    except Exception:
        stop_event.set()
        client.stop_session(session_id)
        raise

    finished_epoch_ms = now_ms()
    summary = summarize_case(
        case_id=case_id,
        input_path=input_path,
        output_dir=case_dir,
        session_id=session_id,
        input_samples=input_samples,
        sent_samples=sent_samples,
        source_rate=source_rate,
        source_channels=source_channels,
        segments=segments,
        archive=archive,
        result=result,
        cfg=cfg,
        started_epoch_ms=started_epoch_ms,
        finished_epoch_ms=finished_epoch_ms,
        input_sha1=input_sha1,
    )
    save_case_outputs(
        case_dir,
        summary=summary,
        input_samples=input_samples,
        sent_samples=sent_samples,
        archive=archive,
        result=result,
        save_audio=save_audio,
    )
    return summary


def discover_cases(data_root: Path) -> list[tuple[str, Path]]:
    cases = []
    for wav in data_root.glob("*/input.wav"):
        case_id = wav.parent.name
        cases.append((case_id, wav))

    def sort_key(item: tuple[str, Path]):
        cid = item[0]
        return (0, int(cid)) if cid.isdigit() else (1, cid)

    return sorted(cases, key=sort_key)


def make_run_dir(output_root: Path, run_name: Optional[str]) -> Path:
    if not run_name:
        run_name = time.strftime("%Y%m%d_%H%M%S_user_interruption_weight_batch")
    return output_root / run_name


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run user_interruption inputs through realtime weight-test route.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Directory containing numeric subdirs with input.wav.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for run outputs.")
    parser.add_argument("--run-name", default="", help="Optional output run directory name.")
    parser.add_argument("--api-base", default="http://127.0.0.1:7860", help="Realtime backend API base.")
    parser.add_argument("--case", action="append", default=[], help="Run only specific case id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N cases after filtering.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip cases before this 0-based index after filtering.")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--infer-window-ms", type=int, default=400)
    parser.add_argument("--tail-padding-sec", type=float, default=30.0)
    parser.add_argument("--start-speak-factor", type=float, default=1.2)
    parser.add_argument("--start-listen-factor", type=float, default=1.2)
    parser.add_argument("--end-speak-factor", type=float, default=1.0)
    parser.add_argument("--prompt-voice", default="guodegang")
    parser.add_argument("--tts-chunk-size", type=int, default=1)
    parser.add_argument("--stutter-threshold-ms", type=float, default=160.0)
    parser.add_argument("--done-timeout-sec", type=float, default=180.0)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--no-realtime", action="store_true", help="Send chunks as fast as HTTP allows. Default preserves realtime pacing.")
    parser.add_argument("--no-audio-save", action="store_true", help="Do not save wav outputs, only JSON/CSV logs.")
    parser.add_argument("--dry-run", action="store_true", help="Only list cases and estimated duration; do not call backend.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    cases = discover_cases(data_root)
    if args.case:
        wanted = set(str(x) for x in args.case)
        cases = [item for item in cases if item[0] in wanted]
    if args.start_index > 0:
        cases = cases[args.start_index :]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print(f"No cases found under {data_root}", file=sys.stderr)
        return 2

    cfg = {
        "chunk_ms": int(clamp(args.chunk_ms, 20, 2000)),
        "infer_window_ms": int(clamp(args.infer_window_ms, 160, 2000)),
        "tail_padding_sec": max(0.0, float(args.tail_padding_sec)),
        "start_speak_factor": float(clamp(args.start_speak_factor, 0.1, 5.0)),
        "start_listen_factor": float(clamp(args.start_listen_factor, 0.1, 5.0)),
        "end_speak_factor": float(clamp(args.end_speak_factor, 0.1, 5.0)),
        "prompt_voice": str(args.prompt_voice),
        "tts_chunk_size": int(max(1, args.tts_chunk_size)),
        "stutter_threshold_ms": float(max(0.0, args.stutter_threshold_ms)),
        "done_timeout_sec": float(max(1.0, args.done_timeout_sec)),
    }

    # Estimate using file headers only.
    durations = []
    chunk_count = 0
    for _, path in cases:
        info = sf.info(str(path))
        durations.append(info.frames / info.samplerate)
        chunk_count += estimate_chunk_count(
            int(info.frames),
            int(info.samplerate),
            cfg["tail_padding_sec"],
            cfg["chunk_ms"],
        )
    sent_sec = sum(x + cfg["tail_padding_sec"] for x in durations)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "data_root": str(data_root),
                "api_base": args.api_base,
                "estimated_input_sec": round(sum(durations), 3),
                "estimated_sent_sec": round(sent_sec, 3),
                "estimated_min_hours_realtime": round(sent_sec / 3600.0, 3),
                "estimated_chunks": int(chunk_count),
                "config": cfg,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        for cid, path in cases:
            print(f"{cid}\t{path}")
        return 0

    run_dir = make_run_dir(output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_config.json",
        {
            "data_root": str(data_root),
            "api_base": args.api_base,
            "case_count": len(cases),
            "cases": [{"case_id": cid, "input_path": str(path)} for cid, path in cases],
            "estimated_input_sec": sum(durations),
            "estimated_sent_sec": sent_sec,
            "estimated_chunks": int(chunk_count),
            "config": cfg,
            "realtime_send": not args.no_realtime,
            "save_audio": not args.no_audio_save,
        },
    )
    client = RealtimeBatchClient(
        args.api_base,
        timeout_sec=args.timeout_sec,
        connect_timeout_sec=args.connect_timeout_sec,
    )
    client.health()

    manifest_path = run_dir / "manifest.jsonl"
    summaries: list[dict] = []
    failures: list[dict] = []
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for idx, (case_id, input_path) in enumerate(cases, start=1):
            case_dir = run_dir / f"{int(case_id):03d}" if case_id.isdigit() else run_dir / case_id
            print(f"[{idx}/{len(cases)}] case={case_id} input={input_path}", flush=True)
            try:
                summary = run_case(
                    client,
                    input_path,
                    case_id,
                    case_dir,
                    cfg,
                    realtime_send=not args.no_realtime,
                    save_audio=not args.no_audio_save,
                )
                summary["status"] = "ok"
                summaries.append(summary)
                manifest.write(json.dumps(summary, ensure_ascii=False, default=json_default) + "\n")
                manifest.flush()
                print(
                    f"  ok wall={summary['wall_time_sec']:.2f}s "
                    f"rounds={summary['stage_round_count']} "
                    f"first_pcm_e2e={summary.get('first_pcm_e2e_ms')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "case_id": case_id,
                    "input_path": str(input_path),
                    "output_dir": str(case_dir),
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_epoch_ms": now_ms(),
                }
                failures.append(failure)
                case_dir.mkdir(parents=True, exist_ok=True)
                write_json(case_dir / "error.json", failure)
                manifest.write(json.dumps(failure, ensure_ascii=False, default=json_default) + "\n")
                manifest.flush()
                print(f"  error {exc}", file=sys.stderr, flush=True)

    write_json(run_dir / "failures.json", failures)
    write_summary_csv(run_dir / "summary.csv", summaries)
    write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "ok_count": len(summaries),
            "failure_count": len(failures),
            "case_count": len(cases),
            "config": cfg,
        },
    )
    print(f"done run_dir={run_dir} ok={len(summaries)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
