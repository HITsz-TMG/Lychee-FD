#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    import scipy.signal
except Exception:  # noqa: BLE001
    scipy = None


TARGET_SAMPLE_RATE = 16000
MIN_SEGMENT_SAMPLES = 800
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "FULL_DUPLEX_USER_INTERRUPTION_DATA_ROOT",
        "datasets/SoulX-Duplug-Eval/Full-Duplex-Bench-zh/user_interruption",
    )
)
DEFAULT_TURN_TAKING_DATA_ROOT = Path(
    os.environ.get(
        "FULL_DUPLEX_TURN_TAKING_DATA_ROOT",
        "datasets/SoulX-Duplug-Eval/Full-Duplex-Bench-zh/turn_taking",
    )
)
DEFAULT_DATA_ROOTS = {
    "user_interruption": DEFAULT_DATA_ROOT,
    "turn_taking": DEFAULT_TURN_TAKING_DATA_ROOT,
}
DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get("OFFLINE_DUPLEX_BENCH_ROOT", "benchmarks/OfflineDuplexBenchDev")
)
BENCHMARK_NAME = "full_duplex_zh"
SUBSET_NAME = "user_interruption"
LANGUAGE = "zh"
METADATA_FILES = [
    "transcription.json",
    "interrupt.json",
    "label.json",
    "metadata.json",
    "input.json",
    "clean_input.json",
    "turn_taking.json",
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._ ") or "sample"


def numeric_key(path: Path) -> tuple[int, str]:
    name = path.stem if path.is_file() else path.name
    return (int(name), name) if name.isdigit() else (10**9, name)


def shard_items(items: list[Any], part: int, total_part: int) -> list[Any]:
    if total_part <= 1:
        return items
    if part < 0 or part >= total_part:
        raise ValueError(f"part must be in [0, total_part), got part={part}, total_part={total_part}")
    per_part = math.ceil(len(items) / total_part)
    start = part * per_part
    end = min((part + 1) * per_part, len(items))
    return items[start:end]


def scan_full_duplex_directory_samples(
    data_root: Path,
    *,
    benchmark: str = BENCHMARK_NAME,
    subset: str = SUBSET_NAME,
    language: str = LANGUAGE,
    metadata_files: list[str] | None = None,
) -> list[dict[str, Any]]:
    data_root = data_root.expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {data_root}")
    copy_metadata_files = metadata_files or METADATA_FILES

    samples: list[dict[str, Any]] = []
    for sample_dir in sorted([p for p in data_root.glob("*") if p.is_dir()], key=numeric_key):
        input_wav = sample_dir / "input.wav"
        if not input_wav.is_file():
            continue
        metadata: dict[str, Any] = {
            "source_sample_dir": str(sample_dir),
            "case_dir": str(sample_dir),
            "task": subset,
            "source_files": [str(input_wav)],
            "version": "v1.0",
        }
        for name in copy_metadata_files:
            path = sample_dir / name
            if not path.exists():
                continue
            metadata["source_files"].append(str(path))
            if path.suffix == ".json":
                try:
                    metadata[path.stem] = load_json(path)
                except Exception:  # noqa: BLE001
                    metadata[path.stem] = str(path)
        samples.append(
            {
                "sample_id": f"{benchmark}/{subset}/{sample_dir.name}",
                "benchmark": benchmark,
                "subset": subset,
                "language": language,
                "input_wav": str(input_wav),
                "metadata": metadata,
            }
        )
    return samples


def scan_user_interruption_samples(data_root: Path) -> list[dict[str, Any]]:
    return scan_full_duplex_directory_samples(
        data_root,
        benchmark=BENCHMARK_NAME,
        subset=SUBSET_NAME,
        language=LANGUAGE,
    )


def sample_output_dir(part_root: Path, sample: dict[str, Any]) -> Path:
    leaf = str(sample["sample_id"]).split("/")[-1]
    return part_root / safe_name(leaf)


def materialize_sample(sample: dict[str, Any], sample_dir: Path, *, copy_input: bool = True) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "sample.json", sample)
    if copy_input:
        copy_if_exists(Path(str(sample["input_wav"])), sample_dir / "input.wav")
    for source in (sample.get("metadata") or {}).get("source_files", []):
        src = Path(str(source))
        if src.exists() and src.suffix in {".json", ".timestamps"}:
            copy_if_exists(src, sample_dir / src.name)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def existing_ok(sample_dir: Path) -> bool:
    events = sample_dir / "events.json"
    if not events.exists():
        return False
    try:
        data = load_json(events)
    except Exception:  # noqa: BLE001
        return False
    return data.get("status") == "ok"


def read_audio_mono(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int, int]:
    samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.size <= 0:
        raise ValueError(f"empty audio: {path}")
    channels = int(samples.shape[1])
    mono = samples.mean(axis=1).astype(np.float32, copy=False)
    source_rate = int(source_rate)
    if source_rate == int(target_sr):
        return mono, source_rate, channels
    if scipy is not None:
        gcd = math.gcd(source_rate, int(target_sr))
        mono = scipy.signal.resample_poly(mono, int(target_sr) // gcd, source_rate // gcd)
    else:
        import librosa

        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=int(target_sr))
    return mono.astype(np.float32, copy=False), source_rate, channels


def append_tail_silence(samples: np.ndarray, silence_sec: float, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    pad = max(0, int(round(float(silence_sec) * int(sample_rate))))
    if pad <= 0:
        return samples.astype(np.float32, copy=False)
    out = np.zeros(int(samples.shape[0]) + pad, dtype=np.float32)
    out[: samples.shape[0]] = samples
    return out


def split_samples(samples: np.ndarray, chunk_ms: int, sample_rate: int = TARGET_SAMPLE_RATE) -> list[np.ndarray]:
    seg_len = max(MIN_SEGMENT_SAMPLES, int(math.floor(int(sample_rate) * int(chunk_ms) / 1000.0)))
    out: list[np.ndarray] = []
    for start in range(0, int(samples.shape[0]), seg_len):
        piece = samples[start : start + seg_len].astype(np.float32, copy=True)
        if piece.shape[0] < MIN_SEGMENT_SAMPLES and out:
            prev = out.pop()
            out.append(np.concatenate([prev, piece]).astype(np.float32, copy=False))
        else:
            out.append(piece)
    return [x for x in out if x.shape[0] >= MIN_SEGMENT_SAMPLES]


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32, copy=False), int(sample_rate), subtype="PCM_16")


def write_wav_stereo(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(int(left.shape[0]), int(right.shape[0]))
    stereo = np.zeros((frame_count, 2), dtype=np.float32)
    stereo[: min(frame_count, left.shape[0]), 0] = left[:frame_count]
    stereo[: min(frame_count, right.shape[0]), 1] = right[:frame_count]
    sf.write(str(path), stereo, int(sample_rate), subtype="PCM_16")


def audio_duration(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    return float(samples.shape[0]) / float(sample_rate) if sample_rate else 0.0


def compact_event(event: Any, index: int | None = None) -> dict[str, Any]:
    if not isinstance(event, dict):
        out = {"raw": str(event)}
        if index is not None:
            out["index"] = index
        return out
    keys = [
        "type",
        "text",
        "start_time",
        "end_time",
        "start_pos",
        "end_pos",
        "end_of_turn",
        "round_id",
        "state",
    ]
    result = {key: event[key] for key in keys if key in event}
    if index is not None:
        result.setdefault("index", index)
    if "audio" in event and isinstance(event["audio"], list):
        result["audio_token_count"] = len(event["audio"])
    if "tokens" in event and isinstance(event["tokens"], list):
        result["text_token_count"] = len(event["tokens"])
    return result


def merge_timeline_segments(segments: list[dict[str, Any]], gap_s: float = 0.5) -> list[dict[str, Any]]:
    valid = [
        {
            **seg,
            "timeline_start_s": float(seg.get("timeline_start_s", seg.get("start_time", 0.0)) or 0.0),
            "timeline_end_s": float(seg.get("timeline_end_s", seg.get("end_time", 0.0)) or 0.0),
        }
        for seg in segments
        if isinstance(seg, dict)
    ]
    valid = [seg for seg in valid if seg["timeline_end_s"] > seg["timeline_start_s"]]
    if not valid:
        return []
    valid.sort(key=lambda item: (item["timeline_start_s"], item["timeline_end_s"]))
    merged: list[dict[str, Any]] = []
    for seg in valid:
        if not merged or seg["timeline_start_s"] - merged[-1]["timeline_end_s"] > gap_s:
            merged.append(
                {
                    "index": len(merged),
                    "status": "ok",
                    "start_time": round(seg["timeline_start_s"], 3),
                    "end_time": round(seg["timeline_end_s"], 3),
                    "timeline_start_s": round(seg["timeline_start_s"], 3),
                    "timeline_end_s": round(seg["timeline_end_s"], 3),
                    "duration": round(seg["timeline_end_s"] - seg["timeline_start_s"], 3),
                    "audio_samples": int(seg.get("audio_samples") or 0),
                    "source_event_indices": [seg.get("index")],
                }
            )
            continue
        prev = merged[-1]
        prev["timeline_end_s"] = round(max(float(prev["timeline_end_s"]), seg["timeline_end_s"]), 3)
        prev["end_time"] = prev["timeline_end_s"]
        prev["duration"] = round(float(prev["timeline_end_s"]) - float(prev["timeline_start_s"]), 3)
        prev["audio_samples"] = int(prev.get("audio_samples") or 0) + int(seg.get("audio_samples") or 0)
        prev.setdefault("source_event_indices", []).append(seg.get("index"))
    return merged


def part_root_for_save(save_path: Path, part: int, total_part: int) -> Path:
    return save_path / f"part{part}of{total_part}" if total_part > 1 else save_path
