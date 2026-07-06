#!/usr/bin/env python3
"""
Run Full-Duplex-Bench-zh/user_interruption through an already-started realtime
server and save outputs in the OfflineDuplexBenchDev-compatible layout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import queue
import threading
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

from user_interruption_bench_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DATA_ROOTS,
    BENCHMARK_NAME,
    LANGUAGE,
    SUBSET_NAME,
    TARGET_SAMPLE_RATE,
    append_tail_silence,
    audio_duration,
    compact_event,
    existing_ok,
    json_default,
    materialize_sample,
    merge_timeline_segments,
    part_root_for_save,
    read_audio_mono,
    sample_output_dir,
    scan_full_duplex_directory_samples,
    scan_user_interruption_samples,
    shard_items,
    split_samples,
    write_json,
    write_jsonl,
    write_wav_mono,
    write_wav_stereo,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def perf_ms() -> float:
    return time.perf_counter() * 1000.0


def pcm16_from_float32(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    out = np.empty(clipped.shape[0], dtype="<i2")
    neg = clipped < 0
    out[neg] = np.round(clipped[neg] * 32768.0).astype("<i2")
    out[~neg] = np.round(clipped[~neg] * 32767.0).astype("<i2")
    return out


def wav_bytes_mono(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16_from_float32(samples).tobytes())
    return bio.getvalue()


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
    raw = base64.b64decode(wav_b64)
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32, copy=False), int(sr)


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if int(sample_rate) == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    from user_interruption_bench_common import scipy

    if scipy is not None:
        gcd = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        return scipy.signal.resample_poly(
            samples.astype(np.float32, copy=False),
            TARGET_SAMPLE_RATE // gcd,
            int(sample_rate) // gcd,
        ).astype(np.float32, copy=False)
    import librosa

    return librosa.resample(samples.astype(np.float32, copy=False), orig_sr=int(sample_rate), target_sr=TARGET_SAMPLE_RATE)


def event_without_audio(payload: dict[str, Any]) -> dict[str, Any]:
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


def normalize_event_payload(payload: dict[str, Any], sse_event_type: str = "message") -> dict[str, Any] | None:
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


def apply_text_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    delta = event.get("delta") if isinstance(event.get("delta"), str) else event.get("text")
    delta = delta if isinstance(delta, str) else ""
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
class OnlineArchive:
    start_perf_ms: float = 0.0
    start_epoch_ms: int = 0
    aligned_chunks: list[np.ndarray] = field(default_factory=list)
    raw_chunks: list[np.ndarray] = field(default_factory=list)
    aligned_samples: int = 0
    raw_samples: int = 0
    has_output: bool = False
    last_audio_arrival_perf_ms: float | None = None
    last_audio_chunk_duration_ms: float = 0.0
    audio_segments: list[dict[str, Any]] = field(default_factory=list)


class RealtimeClient:
    def __init__(self, api_base: str, *, timeout_sec: float, connect_timeout_sec: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.http = requests.Session()

    def url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    def health(self) -> dict[str, Any]:
        resp = self.http.get(self.url("/api/realtime/voices"), timeout=(self.connect_timeout_sec, self.timeout_sec))
        resp.raise_for_status()
        return resp.json()

    def create_session(self, cfg: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "start_speak_factor": cfg["start_speak_factor"],
            "start_listen_factor": cfg["start_listen_factor"],
            "end_speak_factor": cfg["end_speak_factor"],
            "prompt_voice": cfg["prompt_voice"],
            "tts_chunk_size": cfg["tts_chunk_size"],
            "infer_window_ms": cfg["infer_window_ms"],
            "strict_infer_window": cfg["strict_infer_window"],
            "incremental_backend": cfg["incremental_backend"],
            "stage_timing_log": cfg["stage_timing_log"],
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
        headers = {
            "Content-Type": "audio/wav",
            "X-Client-Chunk-Sent-Epoch-Ms": str(int(sent_epoch_ms)),
        }
        resp = self.http.post(
            self.url(f"/api/realtime/session/{session_id}/chunk"),
            headers=headers,
            data=wav_bytes_mono(samples),
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
                        yield event_type, "\n".join(data_lines)
                    event_type = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())


def record_output_chunk(archive: OnlineArchive, samples: np.ndarray, sample_rate: int, event: dict[str, Any]) -> None:
    if samples.size <= 0:
        return
    normalized = resample_to_16k(samples, int(sample_rate))
    if normalized.size <= 0:
        return
    arrival_perf = perf_ms()
    raw_chunk = normalized.astype(np.float32, copy=True)
    archive.raw_chunks.append(raw_chunk)
    archive.raw_samples += int(raw_chunk.shape[0])

    elapsed_samples = max(0, int(round(((arrival_perf - archive.start_perf_ms) / 1000.0) * TARGET_SAMPLE_RATE)))
    gap_samples = max(0, elapsed_samples - archive.aligned_samples)
    if gap_samples > 0:
        archive.aligned_chunks.append(np.zeros(gap_samples, dtype=np.float32))
        archive.aligned_samples += int(gap_samples)

    start_sample = int(archive.aligned_samples)
    archive.aligned_chunks.append(raw_chunk)
    archive.aligned_samples += int(raw_chunk.shape[0])
    archive.has_output = True
    archive.last_audio_arrival_perf_ms = arrival_perf
    archive.last_audio_chunk_duration_ms = raw_chunk.shape[0] / TARGET_SAMPLE_RATE * 1000.0
    archive.audio_segments.append(
        {
            "index": len(archive.audio_segments),
            "status": "ok",
            "start_time": round(start_sample / TARGET_SAMPLE_RATE, 3),
            "end_time": round(archive.aligned_samples / TARGET_SAMPLE_RATE, 3),
            "timeline_start_s": round(start_sample / TARGET_SAMPLE_RATE, 3),
            "timeline_end_s": round(archive.aligned_samples / TARGET_SAMPLE_RATE, 3),
            "duration": round(raw_chunk.shape[0] / TARGET_SAMPLE_RATE, 3),
            "audio_samples": int(raw_chunk.shape[0]),
            "round_id": event.get("round_id"),
            "t2w_backend": event.get("t2w_backend"),
        }
    )


def run_event_consumer(
    client: RealtimeClient,
    session_id: str,
    result: dict[str, Any],
    archive: OnlineArchive,
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
            elif ptype == "stage_timing":
                result["stage_timing"].append(payload)
            elif ptype == "audio_chunk_pcm" and payload.get("pcm_b64"):
                audio = decode_pcm16_b64(str(payload["pcm_b64"]), int(payload.get("num_channels") or 1))
                record_output_chunk(archive, audio, int(payload.get("sample_rate") or TARGET_SAMPLE_RATE), payload)
            elif ptype == "audio_chunk" and payload.get("wav_b64"):
                audio, sr = decode_wav_b64(str(payload["wav_b64"]))
                record_output_chunk(archive, audio, sr, payload)
            elif ptype == "done":
                result["done_seen"] = True
                break
    except Exception as exc:  # noqa: BLE001
        errors.put(exc)


def build_aligned_output(archive: OnlineArchive, min_samples: int) -> np.ndarray:
    aligned = (
        np.concatenate(archive.aligned_chunks).astype(np.float32, copy=False)
        if archive.aligned_chunks
        else np.zeros(0, dtype=np.float32)
    )
    out_len = max(int(min_samples), int(aligned.shape[0]))
    out = np.zeros(out_len, dtype=np.float32)
    out[: min(out_len, aligned.shape[0])] = aligned[:out_len]
    return np.clip(out, -1.0, 1.0)


def run_one_sample(
    *,
    client: RealtimeClient,
    sample: dict[str, Any],
    sample_dir: Path,
    cfg: dict[str, Any],
    realtime_send: bool,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_audio, source_rate, source_channels = read_audio_mono(Path(str(sample["input_wav"])))
    sent_audio = append_tail_silence(input_audio, cfg["tail_padding_sec"])
    segments = split_samples(sent_audio, cfg["chunk_ms"])
    if not segments:
        raise RuntimeError(f"cannot split audio into chunks: {sample['input_wav']}")

    materialize_sample(sample, sample_dir, copy_input=True)
    write_wav_mono(sample_dir / "input.wav", input_audio)
    write_wav_mono(sample_dir / "input_sent.wav", sent_audio)

    started_epoch_ms = now_ms()
    session_payload = client.create_session(cfg)
    session_id = str(session_payload["session_id"])
    archive = OnlineArchive(start_perf_ms=perf_ms(), start_epoch_ms=now_ms())
    result: dict[str, Any] = {
        "events": [],
        "stage_timing": [],
        "state_changes": [],
        "text_state": {
            "response_text": "",
            "current_text_event_id": "",
            "current_text_event_snapshot": "",
        },
        "done_seen": False,
    }
    stop_event = threading.Event()
    errors: queue.Queue = queue.Queue()
    consumer = threading.Thread(
        target=run_event_consumer,
        args=(client, session_id, result, archive, stop_event, errors),
        name=f"ui_events_{str(sample['sample_id']).split('/')[-1]}",
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
            client.send_chunk(session_id, segment, now_ms())
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
    output_audio = build_aligned_output(archive, min_samples=input_audio.shape[0])
    raw_output = (
        np.concatenate(archive.raw_chunks).astype(np.float32, copy=False)
        if archive.raw_chunks
        else np.zeros(0, dtype=np.float32)
    )
    output_path = sample_dir / "output.wav"
    raw_output_path = sample_dir / "raw_output.wav"
    stereo_path = sample_dir / "stereo_input_model.wav"
    events_path = sample_dir / "events.json"
    model_run_path = sample_dir / "model_run.json"
    write_wav_mono(output_path, output_audio)
    write_wav_mono(raw_output_path, raw_output)
    write_wav_stereo(stereo_path, input_audio, output_audio)
    write_jsonl(sample_dir / "realtime_events.jsonl", result["events"])
    write_jsonl(sample_dir / "stage_timing.jsonl", result["stage_timing"])
    write_jsonl(sample_dir / "state_changes.jsonl", [event_without_audio(x) for x in result["state_changes"]])
    (sample_dir / "response.txt").write_text(result["text_state"].get("response_text") or "", encoding="utf-8")

    rendered_events = merge_timeline_segments(archive.audio_segments, gap_s=cfg["merge_gap_s"])
    run = {
        "status": "ok",
        "sample_id": sample["sample_id"],
        "variant": None,
        "model_type": cfg["run_label"],
        "model_path": None,
        "api_base": cfg["api_base"],
        "session_id": session_id,
        "session_start_response": session_payload,
        "input_duration_s": audio_duration(input_audio),
        "padding_duration_s": float(cfg["tail_padding_sec"]),
        "padded_input_duration_s": audio_duration(sent_audio),
        "stream_duration_s": round((finished_epoch_ms - started_epoch_ms) / 1000.0, 3),
        "raw_response_duration_s": audio_duration(raw_output),
        "timeline_response_duration_s": audio_duration(output_audio),
        "text": result["text_state"].get("response_text") or "",
        "pred_events": result["events"],
        "rendered_events": rendered_events,
        "delta_events": rendered_events,
        "response_turns": rendered_events,
        "control_events": {
            "state_changes": [compact_event(item, idx) for idx, item in enumerate(result["state_changes"])],
            "stage_timing_count": len(result["stage_timing"]),
            "done_seen": bool(result.get("done_seen")),
        },
        "online_runtime": {
            "source_rate": int(source_rate),
            "source_channels": int(source_channels),
            "chunk_ms": int(cfg["chunk_ms"]),
            "infer_window_ms": int(cfg["infer_window_ms"]),
            "chunk_count": len(segments),
            "realtime_send": bool(realtime_send),
            "strict_infer_window": bool(cfg["strict_infer_window"]),
            "incremental_backend": cfg["incremental_backend"],
        },
        "files": {
            "input": str(sample_dir / "input.wav"),
            "input_sent": str(sample_dir / "input_sent.wav"),
            "output": str(output_path),
            "raw_output": str(raw_output_path),
            "stereo_input_model": str(stereo_path),
            "events": str(events_path),
            "model_run": str(model_run_path),
            "realtime_events": str(sample_dir / "realtime_events.jsonl"),
        },
    }
    write_json(model_run_path, run)
    write_json(events_path, run)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Full-Duplex-Bench zh subset through realtime HTTP API.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark", default=BENCHMARK_NAME)
    parser.add_argument("--subset", default=SUBSET_NAME)
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:7860")
    parser.add_argument("--run-label", default="realtime_online")
    parser.add_argument("--incremental-backend", choices=["auto", "hf"], default="auto")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total-part", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--infer-window-ms", type=int, default=400)
    parser.add_argument("--tail-padding-sec", type=float, default=12.0)
    parser.add_argument("--start-speak-factor", type=float, default=1.2)
    parser.add_argument("--start-listen-factor", type=float, default=1.2)
    parser.add_argument("--end-speak-factor", type=float, default=1.0)
    parser.add_argument("--prompt-voice", default="snow")
    parser.add_argument("--tts-chunk-size", type=int, default=1)
    parser.add_argument("--strict-infer-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-timing-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--done-timeout-sec", type=float, default=180.0)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.5)
    parser.add_argument("--no-realtime", action="store_true", help="Send chunks as fast as possible; output timing will not be comparable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_root == DEFAULT_DATA_ROOT and args.subset in DEFAULT_DATA_ROOTS:
        args.data_root = DEFAULT_DATA_ROOTS[args.subset]
    if args.subset == SUBSET_NAME and args.data_root == DEFAULT_DATA_ROOT:
        samples = scan_user_interruption_samples(args.data_root)
    else:
        samples = scan_full_duplex_directory_samples(
            args.data_root,
            benchmark=args.benchmark,
            subset=args.subset,
            language=args.language,
        )
    if args.case:
        wanted = set(str(x) for x in args.case)
        samples = [item for item in samples if str(item["sample_id"]).split("/")[-1] in wanted]
    total_loaded = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    samples = shard_items(samples, args.part, args.total_part)
    if not samples:
        raise RuntimeError(f"no samples selected from {args.data_root}")

    save_path = args.save_path.expanduser()
    part_root = part_root_for_save(save_path, args.part, args.total_part)
    part_root.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "api_base": args.api_base.rstrip("/"),
        "run_label": args.run_label,
        "incremental_backend": args.incremental_backend,
        "chunk_ms": max(20, int(args.chunk_ms)),
        "infer_window_ms": max(160, int(args.infer_window_ms)),
        "tail_padding_sec": max(0.0, float(args.tail_padding_sec)),
        "start_speak_factor": float(args.start_speak_factor),
        "start_listen_factor": float(args.start_listen_factor),
        "end_speak_factor": float(args.end_speak_factor),
        "prompt_voice": str(args.prompt_voice),
        "tts_chunk_size": max(1, int(args.tts_chunk_size)),
        "strict_infer_window": bool(args.strict_infer_window),
        "stage_timing_log": bool(args.stage_timing_log),
        "done_timeout_sec": max(1.0, float(args.done_timeout_sec)),
        "merge_gap_s": max(0.0, float(args.merge_gap_s)),
    }
    write_json(
        part_root / "run_config.json",
        {
            "framework": "lychee-fd-demo realtime HTTP client",
            "benchmark": args.benchmark,
            "subset": args.subset,
            "language": args.language,
            "data_root": str(args.data_root.expanduser()),
            "save_path": str(save_path),
            "part_root": str(part_root),
            "part": args.part,
            "total_part": args.total_part,
            "total_loaded_samples": total_loaded,
            "samples_in_this_part": len(samples),
            "resume": bool(args.resume),
            "realtime_send": not args.no_realtime,
            "dry_run": bool(args.dry_run),
            "config": cfg,
        },
    )
    if args.dry_run:
        for sample in samples:
            print(f"{sample['sample_id']}\t{sample['input_wav']}")
        return 0

    client = RealtimeClient(
        args.api_base,
        timeout_sec=args.timeout_sec,
        connect_timeout_sec=args.connect_timeout_sec,
    )
    voices = client.health()
    write_json(part_root / "server_voices.json", voices)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        out_dir = sample_output_dir(part_root, sample)
        print(f"[{idx}/{len(samples)}] {sample['sample_id']} -> {out_dir}", flush=True)
        if args.resume and existing_ok(out_dir):
            run = {"status": "skipped", "sample_id": sample["sample_id"], "sample_dir": str(out_dir)}
            rows.append({"sample": sample, "run": run, "skipped": True})
            print("  skip existing ok", flush=True)
            continue
        try:
            run = run_one_sample(
                client=client,
                sample=sample,
                sample_dir=out_dir,
                cfg=cfg,
                realtime_send=not args.no_realtime,
            )
            rows.append({"sample": sample, "run": run})
            print(
                f"  ok text_len={len(run.get('text') or '')} "
                f"turns={len(run.get('response_turns') or [])} raw_sec={run.get('raw_response_duration_s'):.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "status": "error",
                "sample_id": sample["sample_id"],
                "sample_dir": str(out_dir),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            materialize_sample(sample, out_dir, copy_input=True)
            write_json(out_dir / "error.json", failure)
            write_json(out_dir / "events.json", failure)
            failures.append(failure)
            rows.append({"sample": sample, "run": failure})
            print(f"  error {exc}", flush=True)
            if args.stop_on_error:
                break
        write_jsonl(part_root / "summary.jsonl", rows)
    write_json(part_root / "failures.json", failures)
    write_json(
        part_root / "summary.json",
        {
            "part_root": str(part_root),
            "ok_count": sum(1 for row in rows if (row.get("run") or {}).get("status") == "ok"),
            "skipped_count": sum(1 for row in rows if (row.get("run") or {}).get("status") == "skipped"),
            "failure_count": len(failures),
            "sample_count": len(samples),
        },
    )
    print(f"done part_root={part_root} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
