#!/usr/bin/env python3
"""Analyze batch weight-test runs.

This script reads the JSONL files produced by tools/run_weight_batch.py and
computes a speech-only playback-buffer gap metric from token2wav traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


STAGE_COLORS = {
    "queue_wait": "#f59e0b",
    "preprocess": "#38bdf8",
    "encoder": "#2563eb",
    "transformer": "#7c3aed",
    "token2wav": "#16a34a",
    "coverage": "#84cc16",
    "gap": "#dc2626",
}

STATE_BG = {
    "listening": "#dbeafe",
    "speaking": "#fee2e2",
    "backchannel": "#ede9fe",
    "unknown": "#f8fafc",
}


@dataclass
class SpeechSegment:
    segment_id: int
    start_ms: float
    end_ms: float
    first_round: int | None = None
    last_round: int | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        value_f = float(value)
        if math.isnan(value_f) or math.isinf(value_f):
            return default
        return value_f
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {
            "count": 0,
            "avg": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "sum": 0.0,
        }
    return {
        "count": len(clean),
        "avg": mean(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "max": max(clean),
        "sum": sum(clean),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def numeric_case_dirs(run_dir: Path) -> list[Path]:
    dirs = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            dirs.append(child)
    return sorted(dirs, key=lambda p: int(p.name))


def normalize_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"s", "speak", "speaking", "response"} or "speak" in state:
        return "speaking"
    if state in {"l", "listen", "listening"} or "listen" in state:
        return "listening"
    if state in {"b", "bc", "backchannel"} or "backchannel" in state:
        return "backchannel"
    return "unknown"


def build_speech_segments(stage_rows: list[dict[str, Any]], merge_gap_ms: float) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    current: SpeechSegment | None = None
    for row in sorted(stage_rows, key=lambda r: fnum(r.get("started_at_ms"), 0.0) or 0.0):
        if normalize_state(row.get("state")) != "speaking":
            current = None
            continue
        start = fnum(row.get("started_at_ms"))
        end = fnum(row.get("completed_at_ms"), start)
        if start is None:
            continue
        if end is None or end < start:
            end = start
        round_id = int(fnum(row.get("round"), 0) or 0) or None
        if current is None or start - current.end_ms > merge_gap_ms:
            current = SpeechSegment(
                segment_id=len(segments) + 1,
                start_ms=start,
                end_ms=end,
                first_round=round_id,
                last_round=round_id,
            )
            segments.append(current)
        else:
            current.end_ms = max(current.end_ms, end)
            if round_id is not None:
                current.last_round = round_id
    return segments


def assign_chunks_to_segments(
    t2w_rows: list[dict[str, Any]],
    segments: list[SpeechSegment],
    *,
    pre_tolerance_ms: float,
    post_tolerance_ms: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assigned: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    for row in sorted(t2w_rows, key=lambda r: fnum(r.get("start_ms"), fnum(r.get("recv_epoch_ms"), 0.0)) or 0.0):
        start = fnum(row.get("start_ms"), fnum(row.get("recv_epoch_ms")))
        end = fnum(row.get("end_ms"), start)
        recv = fnum(row.get("recv_epoch_ms"), end)
        points = [p for p in (start, end, recv) if p is not None]
        if not points:
            unassigned.append(row)
            continue
        center = sum(points) / len(points)
        best: SpeechSegment | None = None
        best_distance = float("inf")
        for segment in segments:
            lo = segment.start_ms - pre_tolerance_ms
            hi = segment.end_ms + post_tolerance_ms
            if lo <= center <= hi or any(lo <= p <= hi for p in points):
                if center < segment.start_ms:
                    distance = segment.start_ms - center
                elif center > segment.end_ms:
                    distance = center - segment.end_ms
                else:
                    distance = 0.0
                if distance < best_distance:
                    best = segment
                    best_distance = distance
        if best is None:
            unassigned.append(row)
            continue
        chunk = dict(row)
        chunk["speech_segment_id"] = best.segment_id
        best.chunks.append(chunk)
        assigned.append(chunk)
    return assigned, unassigned


def available_time(row: dict[str, Any], mode: str) -> float | None:
    if mode == "start":
        return fnum(row.get("start_ms"), fnum(row.get("recv_epoch_ms")))
    if mode == "end":
        return fnum(row.get("end_ms"), fnum(row.get("recv_epoch_ms")))
    return fnum(row.get("recv_epoch_ms"), fnum(row.get("end_ms"), fnum(row.get("start_ms"))))


def compute_speech_buffer_gaps(
    segments: list[SpeechSegment],
    *,
    availability_mode: str,
    min_gap_ms: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for segment in segments:
        chunks = sorted(segment.chunks, key=lambda r: available_time(r, availability_mode) or 0.0)
        coverage_end: float | None = None
        prev: dict[str, Any] | None = None
        for chunk in chunks:
            av = available_time(chunk, availability_mode)
            audio_ms = fnum(chunk.get("audio_ms"))
            if av is None or audio_ms is None or audio_ms <= 0:
                continue
            if coverage_end is None:
                coverage_end = av + audio_ms
                prev = chunk
                continue
            gap = av - coverage_end
            if gap >= min_gap_ms:
                events.append(
                    {
                        "case_id": None,
                        "segment_id": segment.segment_id,
                        "segment_first_round": segment.first_round,
                        "segment_last_round": segment.last_round,
                        "prev_t2w_idx": prev.get("idx") if prev else None,
                        "next_t2w_idx": chunk.get("idx"),
                        "gap_ms": gap,
                        "coverage_end_epoch_ms": coverage_end,
                        "next_available_epoch_ms": av,
                        "next_t2w_start_ms": chunk.get("start_ms"),
                        "next_t2w_end_ms": chunk.get("end_ms"),
                        "next_t2w_recv_ms": chunk.get("recv_epoch_ms"),
                        "prev_audio_ms": prev.get("audio_ms") if prev else None,
                        "next_audio_ms": chunk.get("audio_ms"),
                        "next_t2w_dur_ms": chunk.get("dur_ms"),
                        "next_t2w_rtf": chunk.get("rtf"),
                    }
                )
            coverage_end = max(coverage_end, av) + audio_ms
            prev = chunk
    return events


def case_analysis(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(case_dir / "summary.json")
    stage_rows = read_jsonl(case_dir / "stage_rows.jsonl")
    t2w_rows = read_jsonl(case_dir / "t2w_trace.jsonl")
    old_stutters = read_jsonl(case_dir / "stutter_events.jsonl")
    case_id = str(summary.get("case_id") or int(case_dir.name))

    segments = build_speech_segments(stage_rows, args.merge_speaking_gap_ms)
    assigned, unassigned = assign_chunks_to_segments(
        t2w_rows,
        segments,
        pre_tolerance_ms=args.segment_pre_tolerance_ms,
        post_tolerance_ms=args.segment_post_tolerance_ms,
    )
    gap_events = compute_speech_buffer_gaps(
        segments,
        availability_mode=args.availability_time,
        min_gap_ms=args.min_gap_ms,
    )
    for event in gap_events:
        event["case_id"] = case_id
        base = fnum(summary.get("started_epoch_ms"))
        if base is not None:
            av = fnum(event.get("next_available_epoch_ms"))
            cov = fnum(event.get("coverage_end_epoch_ms"))
            event["next_available_rel_s"] = (av - base) / 1000.0 if av is not None else None
            event["coverage_end_rel_s"] = (cov - base) / 1000.0 if cov is not None else None

    gaps = [float(e["gap_ms"]) for e in gap_events if fnum(e.get("gap_ms")) is not None]
    stage_values: dict[str, list[float]] = {
        "queue_wait_ms": [],
        "preprocess_ms": [],
        "encoder_ms": [],
        "transformer_ms": [],
        "token2wav_ms": [],
        "total_round_ms": [],
    }
    for row in stage_rows:
        for key in stage_values:
            value = fnum(row.get(key))
            if value is not None:
                stage_values[key].append(value)

    return {
        "case_dir": case_dir,
        "case_id": case_id,
        "summary": summary,
        "stage_rows": stage_rows,
        "t2w_rows": t2w_rows,
        "segments": segments,
        "assigned_t2w": assigned,
        "unassigned_t2w": unassigned,
        "gap_events": gap_events,
        "old_stutters": old_stutters,
        "case_row": {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "input_sec": summary.get("input_sec"),
            "wall_time_sec": summary.get("wall_time_sec"),
            "infer_window_ms": summary.get("infer_window_ms"),
            "chunk_ms": summary.get("chunk_ms"),
            "speaking_segments": len(segments),
            "speaking_round_count": summary.get("speaking_round_count"),
            "t2w_count": len(t2w_rows),
            "assigned_t2w_count": len(assigned),
            "unassigned_t2w_count": len(unassigned),
            "speech_gap_count": len(gaps),
            "speech_gap_over_40ms_count": sum(1 for x in gaps if x >= 40.0),
            "speech_gap_over_80ms_count": sum(1 for x in gaps if x >= 80.0),
            "speech_gap_over_160ms_count": sum(1 for x in gaps if x >= 160.0),
            "speech_gap_sum_ms": sum(gaps),
            "speech_gap_max_ms": max(gaps, default=0.0),
            "speech_gap_p50_ms": percentile(gaps, 50),
            "speech_gap_p95_ms": percentile(gaps, 95),
            "old_arrival_stutter_count": len(old_stutters),
            "old_arrival_max_stutter_ms": summary.get("max_stutter_ms"),
            "first_pcm_e2e_ms": summary.get("first_pcm_e2e_ms"),
            "transformer_p95_ms": percentile(stage_values["transformer_ms"], 95),
            "token2wav_p95_ms": percentile(stage_values["token2wav_ms"], 95),
            "total_round_p95_ms": percentile(stage_values["total_round_ms"], 95),
        },
        "stage_values": stage_values,
    }


def import_pyplot():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    return plt, Patch


def rel_ms(value: Any, origin_ms: float) -> float | None:
    v = fnum(value)
    if v is None:
        return None
    return (v - origin_ms) / 1000.0


def plot_gap_hist(gaps: list[float], out_path: Path, title: str) -> None:
    plt, _ = import_pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bins = [0, 20, 40, 80, 120, 160, 240, 400, 800, 1600, 3200, max(5000, max(gaps, default=0) + 1)]
    if gaps:
        ax.hist(gaps, bins=bins, color="#2563eb", alpha=0.8, edgecolor="#ffffff")
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel("speech buffer gap (ms)")
    ax.set_ylabel("event count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gap_ecdf(gaps: list[float], out_path: Path, title: str) -> None:
    plt, _ = import_pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    values = sorted(gaps)
    if values:
        y = [(idx + 1) / len(values) for idx in range(len(values))]
        ax.plot(values, y, color="#7c3aed", linewidth=2)
        for x in (40, 80, 160, 400):
            ax.axvline(x, color="#94a3b8", linewidth=0.8, linestyle="--")
            ax.text(x, 0.03, f"{x}ms", rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel("speech buffer gap (ms)")
    ax.set_ylabel("cumulative share")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_case_bars(case_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt, _ = import_pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = sorted(case_rows, key=lambda r: fnum(r.get("speech_gap_max_ms"), 0.0) or 0.0, reverse=True)[:30]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    labels = [str(r["case_id"]) for r in top]
    values = [float(r.get("speech_gap_max_ms") or 0.0) for r in top]
    ax.bar(labels, values, color="#dc2626", alpha=0.8)
    ax.set_xlabel("case id")
    ax.set_ylabel("max speech buffer gap (ms)")
    ax.set_title("Top cases by speech-only token2wav coverage gap")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_box(stage_values: dict[str, list[float]], out_path: Path) -> None:
    plt, _ = import_pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["queue_wait_ms", "preprocess_ms", "encoder_ms", "transformer_ms", "token2wav_ms", "total_round_ms"]
    data = [stage_values.get(k, []) for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.boxplot(data, tick_labels=[k.replace("_ms", "").replace("_", "\n") for k in keys], showfliers=False)
    ax.set_ylabel("duration (ms)")
    ax.set_title("Stage duration distribution (outliers hidden)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def nearest_row_for_time(stage_rows: list[dict[str, Any]], epoch_ms: float) -> int:
    best_idx = 0
    best_distance = float("inf")
    for idx, row in enumerate(stage_rows):
        start = fnum(row.get("started_at_ms"))
        end = fnum(row.get("completed_at_ms"), start)
        if start is None:
            continue
        if end is None:
            end = start
        if start <= epoch_ms <= end:
            return idx
        distance = min(abs(epoch_ms - start), abs(epoch_ms - end))
        if distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def plot_case_gantt(case: dict[str, Any], out_path: Path, availability_mode: str) -> None:
    plt, Patch = import_pyplot()
    stage_rows = case["stage_rows"]
    t2w_rows = case["assigned_t2w"]
    gap_events = case["gap_events"]
    summary = case["summary"]
    if not stage_rows:
        return
    origins = [fnum(summary.get("started_epoch_ms"))]
    for row in stage_rows:
        origins.append(fnum(row.get("started_at_ms")))
    for row in t2w_rows:
        origins.append(fnum(row.get("start_ms")))
    origin = min(v for v in origins if v is not None)

    n = len(stage_rows)
    height = min(24, max(8, n * 0.18 + 3.0))
    fig, ax = plt.subplots(figsize=(14, height), dpi=150)

    for idx, row in enumerate(stage_rows):
        y = n - idx
        start = fnum(row.get("started_at_ms"))
        end = fnum(row.get("completed_at_ms"), start)
        if start is None:
            continue
        if end is None or end < start:
            end = start
        state = normalize_state(row.get("state"))
        x = (start - origin) / 1000.0
        w = max(0.002, (end - start) / 1000.0)
        ax.broken_barh([(x, w)], (y - 0.42, 0.84), facecolors=STATE_BG.get(state, STATE_BG["unknown"]), alpha=0.45)

        queue_wait = fnum(row.get("queue_wait_ms"), 0.0) or 0.0
        if queue_wait > 0:
            ax.broken_barh(
                [((start - queue_wait - origin) / 1000.0, queue_wait / 1000.0)],
                (y - 0.36, 0.20),
                facecolors=STAGE_COLORS["queue_wait"],
                alpha=0.85,
            )
        cursor = start
        for key, color_key in [
            ("preprocess_ms", "preprocess"),
            ("encoder_ms", "encoder"),
            ("transformer_ms", "transformer"),
        ]:
            dur = fnum(row.get(key), 0.0) or 0.0
            if dur <= 0:
                continue
            ax.broken_barh(
                [((cursor - origin) / 1000.0, dur / 1000.0)],
                (y - 0.12, 0.24),
                facecolors=STAGE_COLORS[color_key],
                alpha=0.9,
            )
            cursor += dur

    for row in t2w_rows:
        start = fnum(row.get("start_ms"))
        end = fnum(row.get("end_ms"), start)
        if start is None or end is None:
            continue
        idx = nearest_row_for_time(stage_rows, start)
        y = n - idx
        ax.broken_barh(
            [((start - origin) / 1000.0, max(0.002, (end - start) / 1000.0))],
            (y + 0.14, 0.22),
            facecolors=STAGE_COLORS["token2wav"],
            alpha=0.95,
        )

    # A compact top lane shows playable coverage from token2wav availability.
    coverage_y = n + 1.2
    for row in t2w_rows:
        av = available_time(row, availability_mode)
        audio_ms = fnum(row.get("audio_ms"))
        if av is None or audio_ms is None or audio_ms <= 0:
            continue
        ax.broken_barh(
            [((av - origin) / 1000.0, audio_ms / 1000.0)],
            (coverage_y - 0.16, 0.32),
            facecolors=STAGE_COLORS["coverage"],
            alpha=0.55,
        )
    for event in gap_events:
        start = fnum(event.get("coverage_end_epoch_ms"))
        end = fnum(event.get("next_available_epoch_ms"))
        if start is None or end is None or end <= start:
            continue
        ax.axvspan((start - origin) / 1000.0, (end - origin) / 1000.0, color=STAGE_COLORS["gap"], alpha=0.18)
        ax.broken_barh(
            [((start - origin) / 1000.0, (end - start) / 1000.0)],
            (coverage_y + 0.22, 0.24),
            facecolors=STAGE_COLORS["gap"],
            alpha=0.9,
        )

    max_x = 0.0
    for row in stage_rows:
        end = fnum(row.get("completed_at_ms"))
        if end is not None:
            max_x = max(max_x, (end - origin) / 1000.0)
    for row in t2w_rows:
        av = available_time(row, availability_mode)
        audio_ms = fnum(row.get("audio_ms"), 0.0) or 0.0
        if av is not None:
            max_x = max(max_x, (av + audio_ms - origin) / 1000.0)

    tick_step = max(1, int(math.ceil(n / 25)))
    y_ticks = [n - idx for idx in range(0, n, tick_step)]
    y_labels = [str(stage_rows[idx].get("round") or idx + 1) for idx in range(0, n, tick_step)]
    ax.set_yticks([coverage_y] + y_ticks)
    ax.set_yticklabels(["T2W coverage"] + y_labels)
    ax.set_xlim(left=0, right=max_x + 0.5)
    ax.set_xlabel("elapsed wall time (s)")
    ax.set_ylabel("round")
    title = (
        f"Case {case['case_id']} stage gantt "
        f"(speech gaps={len(gap_events)}, max={max((e['gap_ms'] for e in gap_events), default=0):.1f}ms)"
    )
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    legend = [
        Patch(facecolor=STATE_BG["speaking"], label="speaking row", alpha=0.45),
        Patch(facecolor=STATE_BG["listening"], label="listening row", alpha=0.45),
        Patch(facecolor=STAGE_COLORS["queue_wait"], label="input queue wait"),
        Patch(facecolor=STAGE_COLORS["preprocess"], label="preprocess"),
        Patch(facecolor=STAGE_COLORS["encoder"], label="audio encoder"),
        Patch(facecolor=STAGE_COLORS["transformer"], label="transformer"),
        Patch(facecolor=STAGE_COLORS["token2wav"], label="token2wav actual"),
        Patch(facecolor=STAGE_COLORS["coverage"], label="T2W audio coverage"),
        Patch(facecolor=STAGE_COLORS["gap"], label="speech buffer gap"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="batch_weight_runs/<run_name> directory")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--availability-time", choices=["recv", "end", "start"], default="recv")
    parser.add_argument("--min-gap-ms", type=float, default=1.0, help="Minimum positive gap to record.")
    parser.add_argument("--merge-speaking-gap-ms", type=float, default=800.0)
    parser.add_argument("--segment-pre-tolerance-ms", type=float, default=800.0)
    parser.add_argument("--segment-post-tolerance-ms", type=float, default=1200.0)
    parser.add_argument("--top-gantt", type=int, default=20, help="Render top N cases by max speech gap.")
    parser.add_argument("--all-gantt", action="store_true", help="Render gantt charts for every case.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / f"analysis_speech_buffer_{args.availability_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [case_analysis(case_dir, args) for case_dir in numeric_case_dirs(run_dir)]
    case_rows = [case["case_row"] for case in cases]
    all_gap_events = []
    all_stage_values: dict[str, list[float]] = {
        "queue_wait_ms": [],
        "preprocess_ms": [],
        "encoder_ms": [],
        "transformer_ms": [],
        "token2wav_ms": [],
        "total_round_ms": [],
    }
    for case in cases:
        all_gap_events.extend(case["gap_events"])
        for key, values in case["stage_values"].items():
            all_stage_values.setdefault(key, []).extend(values)

    gap_values = [float(e["gap_ms"]) for e in all_gap_events]
    write_csv(out_dir / "case_speech_gap_summary.csv", case_rows)
    write_csv(out_dir / "speech_buffer_gap_events.csv", all_gap_events)
    stage_rows = []
    for key, values in all_stage_values.items():
        row = {"metric": key}
        row.update(stats(values))
        stage_rows.append(row)
    write_csv(out_dir / "stage_duration_summary.csv", stage_rows)
    top_rows = sorted(case_rows, key=lambda r: fnum(r.get("speech_gap_max_ms"), 0.0) or 0.0, reverse=True)
    write_csv(out_dir / "top_speech_gap_cases.csv", top_rows[: max(50, args.top_gantt)])

    summary_payload = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "availability_time": args.availability_time,
        "min_gap_ms": args.min_gap_ms,
        "case_count": len(cases),
        "total_t2w_chunks": sum(int(row.get("t2w_count") or 0) for row in case_rows),
        "assigned_t2w_chunks": sum(int(row.get("assigned_t2w_count") or 0) for row in case_rows),
        "unassigned_t2w_chunks": sum(int(row.get("unassigned_t2w_count") or 0) for row in case_rows),
        "speech_gap_stats_ms": stats(gap_values),
        "speech_gap_over_40ms_count": sum(1 for x in gap_values if x >= 40.0),
        "speech_gap_over_80ms_count": sum(1 for x in gap_values if x >= 80.0),
        "speech_gap_over_160ms_count": sum(1 for x in gap_values if x >= 160.0),
        "cases_with_speech_gap": sum(1 for row in case_rows if int(row.get("speech_gap_count") or 0) > 0),
        "cases_with_gap_over_160ms": sum(1 for row in case_rows if int(row.get("speech_gap_over_160ms_count") or 0) > 0),
        "top_cases_by_max_gap": [
            {
                "case_id": row.get("case_id"),
                "speech_gap_max_ms": row.get("speech_gap_max_ms"),
                "speech_gap_count": row.get("speech_gap_count"),
                "speech_gap_over_160ms_count": row.get("speech_gap_over_160ms_count"),
            }
            for row in top_rows[:10]
        ],
    }
    write_summary_json(out_dir / "analysis_summary.json", summary_payload)

    if not args.no_plots:
        plot_gap_hist(gap_values, out_dir / "speech_gap_hist.png", "Speech-only token2wav coverage gaps")
        plot_gap_ecdf(gap_values, out_dir / "speech_gap_ecdf.png", "Speech-only token2wav coverage gap ECDF")
        plot_case_bars(case_rows, out_dir / "top_case_max_speech_gap.png")
        plot_stage_box(all_stage_values, out_dir / "stage_duration_box.png")
        selected = cases if args.all_gantt else [
            next(case for case in cases if case["case_id"] == str(row["case_id"]))
            for row in top_rows[: max(0, args.top_gantt)]
        ]
        gantt_dir = out_dir / "gantt"
        for case in selected:
            plot_case_gantt(case, gantt_dir / f"case_{int(case['case_id']):03d}_gantt.png", args.availability_time)

    print(f"out_dir={out_dir}")
    print(f"cases={len(cases)}")
    print(f"speech_gap_events={len(gap_values)}")
    print(f"speech_gap_stats_ms={json.dumps(summary_payload['speech_gap_stats_ms'], ensure_ascii=False)}")
    print(f"cases_with_gap_over_160ms={summary_payload['cases_with_gap_over_160ms']}")
    print("top_cases=" + json.dumps(summary_payload["top_cases_by_max_gap"][:5], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
