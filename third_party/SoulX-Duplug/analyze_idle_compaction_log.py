#!/usr/bin/env python3
"""Analyze SoulX idle KV compaction events from backend logs.

The parser is intentionally log-format based so it can be run offline against
existing backend_dev_*.log files without importing the realtime server.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


COMPACT_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\[INFO\]\s+SoulX idle compaction "
    r"round=(?P<round>\d+)\s+"
    r"idle_ms=(?P<idle_ms>\d+)\s+"
    r"prefix_len=(?P<prefix_before>\d+)->(?P<prefix_after>\d+)\s+"
    r"dropped=(?P<dropped>\d+)\s+"
    r"audio_cache_tokens=(?P<audio_cache_tokens>\d+)\s+"
    r"audio_cache_windows=(?P<audio_cache_windows>\d+)\s+"
    r"compact_count=(?P<compact_count>\d+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"sync=(?P<sync>\S+)\s+"
    r"coverage=(?P<coverage_start>\d+)->(?P<coverage_end>\d+)\s+"
    r"processed=(?P<processed>\d+)\s+"
    r"kv_truncate=(?P<kv_truncate>\S+)\s+"
    r"kv_freed_blocks=(?P<kv_freed_blocks>\d+)\s+"
    r"(?:kv_kept_blocks=(?P<kv_kept_blocks>\d+)\s+)?"
    r"(?:kv_block_size=(?P<kv_block_size>\d+)\s+)?"
    r"kv_reason=(?P<kv_reason>\S+)"
)

SKIP_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\[INFO\]\s+SoulX idle compaction skipped "
    r"round=(?P<round>\d+)\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"idle_ms=(?P<idle_ms>\d+)\s+"
    r"prefix_len=(?P<prefix_before>\d+)->(?P<prefix_after>\d+)\s+"
    r"dropped=(?P<dropped>\d+)\s+"
    r"min_drop=(?P<min_drop>\d+)\s+"
    r"audio_cache_tokens=(?P<audio_cache_tokens>\d+)\s+"
    r"compact_count=(?P<compact_count>\d+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"sync=(?P<sync>\S+)\s+"
    r"coverage=(?P<coverage_start>\d+)->(?P<coverage_end>\d+)\s+"
    r"processed=(?P<processed>\d+)\s+"
    r"kv_reason=(?P<kv_reason>\S*)"
)

ROUND_START_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\[INFO\]\s+Realtime round=(?P<round>\d+) start "
    r"prefix_len=(?P<prefix_len>\d+)\s+"
    r"state=(?P<state>\S+)\s+"
    r"total_sec=(?P<total_sec>[0-9.]+)\s+"
    r"pending_sec=(?P<pending_sec>[0-9.]+)\s+"
    r"consume_sec=(?P<consume_sec>[0-9.]+)"
)

VLLM_DIAG_RE = re.compile(
    r"\[VLLM_DIAG\]\s+request=(?P<request>\S+)\s+tail_truncated "
    r"text_len=(?P<text_before>\d+)->(?P<text_after>\d+)\s+"
    r"freed_blocks=(?P<freed_blocks>\d+)"
)


@dataclass
class RoundStart:
    line_no: int
    ts: str
    round_id: int
    prefix_len: int
    state: str
    total_sec: float
    pending_sec: float
    consume_sec: float


@dataclass
class CompactEvent:
    line_no: int
    ts: str
    round_id: int
    idle_ms: int
    prefix_before: int
    prefix_after: int
    dropped_tokens: int
    audio_cache_tokens: int
    audio_cache_windows: int
    compact_count: int
    mode: str
    sync: str
    coverage_start: int
    coverage_end: int
    processed: int
    kv_truncate: bool
    kv_freed_blocks: int
    kv_kept_blocks: int
    kv_block_size: int
    kv_reason: str
    round_start: Optional[RoundStart] = None
    diag_text_before: Optional[int] = None
    diag_text_after: Optional[int] = None

    @property
    def coverage_samples(self) -> int:
        return max(0, self.coverage_end - self.coverage_start)

    def coverage_sec(self, sample_rate: int) -> float:
        return float(self.coverage_samples) / float(sample_rate) if sample_rate > 0 else 0.0

    def dropped_sec(self, tokens_per_sec: float) -> float:
        return float(self.dropped_tokens) / tokens_per_sec if tokens_per_sec > 0 else 0.0


@dataclass
class SkippedEvent:
    line_no: int
    ts: str
    round_id: int
    reason: str
    idle_ms: int
    prefix_before: int
    prefix_after: int
    dropped_tokens: int
    compact_count: int
    mode: str
    sync: str
    coverage_start: int
    coverage_end: int
    processed: int
    kv_reason: str


def _to_int(match: re.Match[str], name: str) -> int:
    return int(match.group(name))


def _to_float(match: re.Match[str], name: str) -> float:
    return float(match.group(name))


def iter_lf_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Yield LF-delimited lines so line numbers match rg/nl/sed.

    Python's default text iteration also treats bare carriage returns as line
    breaks. Some backend logs contain progress output with bare CR characters,
    so using normal text iteration can drift away from shell tool line numbers.
    """
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle.read().split(b"\n"), 1):
            yield line_no, raw_line.decode("utf-8", errors="replace").rstrip("\r")


def parse_log(path: Path) -> tuple[list[CompactEvent], list[SkippedEvent], list[RoundStart]]:
    compact_events: list[CompactEvent] = []
    skipped_events: list[SkippedEvent] = []
    round_starts: list[RoundStart] = []
    current_round_by_id: dict[int, RoundStart] = {}
    pending_diag: list[tuple[int, int, int, int]] = []

    for line_no, line in iter_lf_lines(path):
            start_match = ROUND_START_RE.search(line)
            if start_match:
                round_start = RoundStart(
                    line_no=line_no,
                    ts=start_match.group("ts"),
                    round_id=_to_int(start_match, "round"),
                    prefix_len=_to_int(start_match, "prefix_len"),
                    state=start_match.group("state"),
                    total_sec=_to_float(start_match, "total_sec"),
                    pending_sec=_to_float(start_match, "pending_sec"),
                    consume_sec=_to_float(start_match, "consume_sec"),
                )
                round_starts.append(round_start)
                current_round_by_id[round_start.round_id] = round_start
                continue

            diag_match = VLLM_DIAG_RE.search(line)
            if diag_match:
                pending_diag.append(
                    (
                        line_no,
                        _to_int(diag_match, "text_before"),
                        _to_int(diag_match, "text_after"),
                        _to_int(diag_match, "freed_blocks"),
                    )
                )
                continue

            compact_match = COMPACT_RE.search(line)
            if compact_match:
                event = CompactEvent(
                    line_no=line_no,
                    ts=compact_match.group("ts"),
                    round_id=_to_int(compact_match, "round"),
                    idle_ms=_to_int(compact_match, "idle_ms"),
                    prefix_before=_to_int(compact_match, "prefix_before"),
                    prefix_after=_to_int(compact_match, "prefix_after"),
                    dropped_tokens=_to_int(compact_match, "dropped"),
                    audio_cache_tokens=_to_int(compact_match, "audio_cache_tokens"),
                    audio_cache_windows=_to_int(compact_match, "audio_cache_windows"),
                    compact_count=_to_int(compact_match, "compact_count"),
                    mode=compact_match.group("mode"),
                    sync=compact_match.group("sync"),
                    coverage_start=_to_int(compact_match, "coverage_start"),
                    coverage_end=_to_int(compact_match, "coverage_end"),
                    processed=_to_int(compact_match, "processed"),
                    kv_truncate=compact_match.group("kv_truncate").lower() == "true",
                    kv_freed_blocks=_to_int(compact_match, "kv_freed_blocks"),
                    kv_kept_blocks=int(compact_match.group("kv_kept_blocks") or 0),
                    kv_block_size=int(compact_match.group("kv_block_size") or 0),
                    kv_reason=compact_match.group("kv_reason"),
                    round_start=current_round_by_id.get(_to_int(compact_match, "round")),
                )
                if pending_diag:
                    _, text_before, text_after, freed_blocks = pending_diag.pop(0)
                    if freed_blocks == event.kv_freed_blocks:
                        event.diag_text_before = text_before
                        event.diag_text_after = text_after
                compact_events.append(event)
                continue

            skip_match = SKIP_RE.search(line)
            if skip_match:
                skipped_events.append(
                    SkippedEvent(
                        line_no=line_no,
                        ts=skip_match.group("ts"),
                        round_id=_to_int(skip_match, "round"),
                        reason=skip_match.group("reason"),
                        idle_ms=_to_int(skip_match, "idle_ms"),
                        prefix_before=_to_int(skip_match, "prefix_before"),
                        prefix_after=_to_int(skip_match, "prefix_after"),
                        dropped_tokens=_to_int(skip_match, "dropped"),
                        compact_count=_to_int(skip_match, "compact_count"),
                        mode=skip_match.group("mode"),
                        sync=skip_match.group("sync"),
                        coverage_start=_to_int(skip_match, "coverage_start"),
                        coverage_end=_to_int(skip_match, "coverage_end"),
                        processed=_to_int(skip_match, "processed"),
                        kv_reason=skip_match.group("kv_reason"),
                    )
                )

    return compact_events, skipped_events, round_starts


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def infer_tokens_per_sec(rounds: list[RoundStart]) -> float:
    ratios: list[float] = []
    for prev, curr in zip(rounds, rounds[1:]):
        if curr.round_id != prev.round_id + 1:
            continue
        delta_prefix = curr.prefix_len - prev.prefix_len
        delta_sec = curr.total_sec - prev.total_sec
        if delta_prefix <= 0 or delta_sec <= 0:
            continue
        # Most normal rounds are 10 tokens / 0.4s. Compaction/session resets
        # are filtered above; this bound drops rare startup or flush outliers.
        ratio = float(delta_prefix) / float(delta_sec)
        if 5.0 <= ratio <= 80.0:
            ratios.append(ratio)
    if not ratios:
        return 25.0
    return median(ratios)


def pct(numerator: float, denominator: float) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def estimate_blocks(token_len: int, block_size: int, system_tokens: int = 0) -> int:
    logical_len = max(1, int(token_len) + max(0, int(system_tokens)))
    return int(math.ceil(float(logical_len) / float(block_size)))


def event_block_size(event: CompactEvent, fallback_block_size: int) -> int:
    return int(event.kv_block_size or fallback_block_size)


def event_kept_blocks(
    event: CompactEvent, fallback_block_size: int, system_tokens: int
) -> int:
    if event.kv_kept_blocks > 0:
        return int(event.kv_kept_blocks)
    return estimate_blocks(event.prefix_after, event_block_size(event, fallback_block_size), system_tokens)


def write_csv(
    csv_path: Path,
    events: list[CompactEvent],
    *,
    sample_rate: int,
    tokens_per_sec: float,
    block_size: int,
    system_tokens: int,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "line",
                "timestamp",
                "round",
                "compact_count",
                "mode",
                "sync",
                "prefix_before",
                "prefix_after",
                "dropped_tokens",
                "dropped_token_pct",
                "estimated_dropped_sec",
                "coverage_sec",
                "coverage_used_pct",
                "kept_blocks_est",
                "freed_blocks_log",
                "freed_token_capacity",
                "freed_capacity_pct_vs_before",
                "kv_truncate",
                "kv_reason",
                "total_sec_at_round",
                "round_state",
            ],
        )
        writer.writeheader()
        for event in events:
            effective_block_size = event_block_size(event, block_size)
            kept_blocks = event_kept_blocks(event, block_size, system_tokens)
            before_blocks = kept_blocks + event.kv_freed_blocks
            freed_capacity = event.kv_freed_blocks * effective_block_size
            writer.writerow(
                {
                    "line": event.line_no,
                    "timestamp": event.ts,
                    "round": event.round_id,
                    "compact_count": event.compact_count,
                    "mode": event.mode,
                    "sync": event.sync,
                    "prefix_before": event.prefix_before,
                    "prefix_after": event.prefix_after,
                    "dropped_tokens": event.dropped_tokens,
                    "dropped_token_pct": format_float(
                        pct(event.dropped_tokens, event.prefix_before), 2
                    ),
                    "estimated_dropped_sec": format_float(
                        event.dropped_sec(tokens_per_sec), 3
                    ),
                    "coverage_sec": format_float(event.coverage_sec(sample_rate), 3),
                    "coverage_used_pct": format_float(
                        pct(event.dropped_sec(tokens_per_sec), event.coverage_sec(sample_rate)), 2
                    ),
                    "kept_blocks_est": kept_blocks,
                    "freed_blocks_log": event.kv_freed_blocks,
                    "freed_token_capacity": freed_capacity,
                    "freed_capacity_pct_vs_before": format_float(
                        pct(event.kv_freed_blocks, before_blocks), 2
                    ),
                    "kv_truncate": event.kv_truncate,
                    "kv_reason": event.kv_reason,
                    "total_sec_at_round": (
                        format_float(event.round_start.total_sec, 3)
                        if event.round_start
                        else ""
                    ),
                    "round_state": event.round_start.state if event.round_start else "",
                }
            )


def render_report(
    path: Path,
    events: list[CompactEvent],
    skipped: list[SkippedEvent],
    rounds: list[RoundStart],
    *,
    sample_rate: int,
    tokens_per_sec: float,
    block_size: int,
    system_tokens: int,
    limit: Optional[int],
) -> str:
    total_dropped_tokens = sum(event.dropped_tokens for event in events)
    total_freed_blocks = sum(event.kv_freed_blocks for event in events)
    total_freed_capacity = sum(
        event.kv_freed_blocks * event_block_size(event, block_size) for event in events
    )
    total_idle_ms = sum(event.idle_ms for event in events)
    total_covered_sec = sum(event.coverage_sec(sample_rate) for event in events)
    total_dropped_sec = sum(event.dropped_sec(tokens_per_sec) for event in events)
    max_prefix = max((round_start.prefix_len for round_start in rounds), default=0)
    final_prefix = rounds[-1].prefix_len if rounds else 0
    max_total_sec = max((round_start.total_sec for round_start in rounds), default=0.0)

    lines: list[str] = []
    lines.append(f"# SoulX Idle KV Compaction Report")
    lines.append("")
    lines.append(f"- log: {path}")
    lines.append(f"- parsed_round_starts: {len(rounds)}")
    lines.append(f"- successful_compactions: {len(events)}")
    lines.append(f"- skipped_compactions: {len(skipped)}")
    lines.append(f"- sample_rate: {sample_rate} Hz")
    lines.append(f"- tokens_per_sec: {format_float(tokens_per_sec, 3)}")
    lines.append(f"- block_size: {block_size}")
    logged_block_sizes = sorted({event.kv_block_size for event in events if event.kv_block_size > 0})
    if logged_block_sizes:
        lines.append(f"- logged_block_sizes: {', '.join(str(x) for x in logged_block_sizes)}")
    lines.append(f"- system_tokens_for_block_estimate: {system_tokens}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- dropped_tokens_total: {total_dropped_tokens}")
    lines.append(f"- dropped_time_est_total: {format_float(total_dropped_sec, 3)} sec")
    lines.append(f"- idle_confirmed_time_sum: {format_float(total_idle_ms / 1000.0, 3)} sec")
    lines.append(f"- idle_coverage_time_sum: {format_float(total_covered_sec, 3)} sec")
    lines.append(f"- freed_blocks_total: {total_freed_blocks}")
    lines.append(f"- freed_token_capacity_total: {total_freed_capacity}")
    lines.append(f"- max_observed_prefix_len: {max_prefix}")
    lines.append(f"- final_observed_prefix_len: {final_prefix}")
    lines.append(f"- max_observed_session_audio_sec: {format_float(max_total_sec, 3)} sec")
    if max_prefix:
        lines.append(
            f"- dropped_vs_max_prefix: {format_float(pct(total_dropped_tokens, max_prefix), 2)}%"
        )
    if max_total_sec:
        lines.append(
            f"- dropped_time_vs_session_audio: {format_float(pct(total_dropped_sec, max_total_sec), 2)}%"
        )
    lines.append("")
    lines.append("## Events")
    lines.append("")
    lines.append(
        "| # | line | time | round | prefix | drop tok | drop % | est drop s | coverage s | freed blk | freed cap | cap % | kv | reason |"
    )
    lines.append(
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )

    display_events = events[-limit:] if limit and limit > 0 else events
    offset = len(events) - len(display_events)
    for idx, event in enumerate(display_events, offset + 1):
        effective_block_size = event_block_size(event, block_size)
        kept_blocks = event_kept_blocks(event, block_size, system_tokens)
        before_blocks = kept_blocks + event.kv_freed_blocks
        freed_capacity = event.kv_freed_blocks * effective_block_size
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(event.line_no),
                    event.ts,
                    str(event.round_id),
                    f"{event.prefix_before}->{event.prefix_after}",
                    str(event.dropped_tokens),
                    format_float(pct(event.dropped_tokens, event.prefix_before), 1),
                    format_float(event.dropped_sec(tokens_per_sec), 2),
                    format_float(event.coverage_sec(sample_rate), 2),
                    str(event.kv_freed_blocks),
                    str(freed_capacity),
                    format_float(pct(event.kv_freed_blocks, before_blocks), 1),
                    str(event.kv_truncate),
                    event.kv_reason,
                ]
            )
            + " |"
        )

    if skipped:
        reason_counts: dict[str, int] = {}
        for item in skipped:
            reason_counts[item.reason] = reason_counts.get(item.reason, 0) + 1
        lines.append("")
        lines.append("## Skipped")
        lines.append("")
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")

    lines.append("")
    lines.append("Notes:")
    lines.append("- `drop tok` uses the realtime prefix token length printed by the backend.")
    lines.append("- `freed blk` is the actual `kv_freed_blocks` value from the log.")
    lines.append("- `freed cap` is `freed blk * block_size`; it is KV page capacity, not necessarily exact dropped token count.")
    lines.append("- If the log does not print block size, pass `--block-size`; the default is 16.")
    return "\n".join(lines)


def latest_backend_log(root: Path) -> Path:
    candidates = sorted(
        root.glob("runtime_logs/controller_dev/backend_dev_*.log"),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No backend_dev_*.log found under {root}/runtime_logs/controller_dev")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse backend logs and summarize SoulX idle KV compaction."
    )
    parser.add_argument(
        "log",
        nargs="?",
        help="Backend log path. Defaults to the newest runtime_logs/controller_dev/backend_dev_*.log.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root used when log is omitted. Default: current directory.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Input audio sample rate used by coverage samples. Default: 16000.",
    )
    parser.add_argument(
        "--tokens-per-sec",
        type=float,
        default=0.0,
        help="Realtime prefix tokens per second. If omitted, inferred from round prefix/total_sec deltas.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="vLLM KV block size for capacity estimates. Default: 16.",
    )
    parser.add_argument(
        "--system-tokens",
        type=int,
        default=0,
        help="Optional system prompt token count added when estimating kept blocks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the last N compaction events in the markdown table. Default: all.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path for per-event metrics.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional markdown report output path. Default: print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    log_path = Path(args.log).resolve() if args.log else latest_backend_log(root)
    events, skipped, rounds = parse_log(log_path)

    tokens_per_sec = float(args.tokens_per_sec)
    if tokens_per_sec <= 0:
        tokens_per_sec = infer_tokens_per_sec(rounds)

    if args.csv:
        write_csv(
            Path(args.csv),
            events,
            sample_rate=int(args.sample_rate),
            tokens_per_sec=tokens_per_sec,
            block_size=int(args.block_size),
            system_tokens=int(args.system_tokens),
        )

    report = render_report(
        log_path,
        events,
        skipped,
        rounds,
        sample_rate=int(args.sample_rate),
        tokens_per_sec=tokens_per_sec,
        block_size=int(args.block_size),
        system_tokens=int(args.system_tokens),
        limit=int(args.limit) if args.limit else None,
    )

    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
