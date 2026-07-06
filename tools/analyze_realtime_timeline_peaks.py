#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def find_latest_jsonl(root: Path) -> Path:
    files = sorted(root.rglob("realtime_timeline_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No realtime_timeline_*.jsonl found under {root}")
    return files[-1]


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_start_ms(row):
    timing = row.get("round_timing") if isinstance(row.get("round_timing"), dict) else {}
    value = timing.get("round_started_epoch_ms")
    if value is None:
        value = row.get("round_started_at_epoch_ms")
    return as_float(value, None)


def span_summary_value(row, name, field="duration_ms"):
    summary = row.get("span_summary") if isinstance(row.get("span_summary"), dict) else {}
    item = summary.get(name) if isinstance(summary.get(name), dict) else {}
    return as_float(item.get(field), 0.0)


def span_summary_count(row, name):
    summary = row.get("span_summary") if isinstance(row.get("span_summary"), dict) else {}
    item = summary.get(name) if isinstance(summary.get(name), dict) else {}
    return int(as_float(item.get("count"), 0.0))


def load_items(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    starts = [round_start_ms(row) for row in rows]
    starts = [x for x in starts if x is not None]
    if not starts:
        raise ValueError(f"No round start timestamps found in {path}")
    t0 = min(starts)
    items = []
    for row in rows:
        start = round_start_ms(row)
        latency = row.get("latency_summary") if isinstance(row.get("latency_summary"), dict) else {}
        items.append({
            "round": int(row.get("round_id", len(items) + 1)),
            "sec": (start - t0) / 1000.0 if start is not None else 0.0,
            "total": as_float(latency.get("total_round_sec")) * 1000.0,
            "stream": as_float(latency.get("stream_infer_sec")) * 1000.0,
            "input": as_float(row.get("input_duration_sec")) * 1000.0,
            "transformer": span_summary_value(row, "transformer"),
            "token2wav": span_summary_value(row, "token2wav"),
            "encoder": span_summary_value(row, "audio_encoder"),
            "queue": span_summary_value(row, "input_queue_wait"),
            "transformer_calls": span_summary_count(row, "transformer"),
            "token2wav_calls": span_summary_count(row, "token2wav"),
        })
    return t0, items


def print_item(prefix, item):
    print(
        f"{prefix} t={item['sec']:.3f}s r{item['round']} "
        f"total={item['total']:.1f}ms input={item['input']:.1f}ms "
        f"transformer={item['transformer']:.1f}ms/{item['transformer_calls']} "
        f"token2wav={item['token2wav']:.1f}ms/{item['token2wav_calls']} "
        f"queue={item['queue']:.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Print blocking peaks from realtime_timeline JSONL.")
    parser.add_argument("jsonl", nargs="?", help="Path to realtime_timeline_*.jsonl. If omitted, use latest under runtime_logs.")
    parser.add_argument("--runtime-root", default="runtime_logs")
    parser.add_argument("--total-threshold-ms", type=float, default=700.0)
    parser.add_argument("--transformer-threshold-ms", type=float, default=700.0)
    parser.add_argument("--token2wav-threshold-ms", type=float, default=500.0)
    parser.add_argument("--input-threshold-ms", type=float, default=700.0)
    parser.add_argument("--cluster-total-ms", type=float, default=500.0)
    args = parser.parse_args()

    path = Path(args.jsonl) if args.jsonl else find_latest_jsonl(Path(args.runtime_root))
    session_start_ms, items = load_items(path)
    print(f"jsonl={path}")
    print(f"session_start_epoch_ms={int(session_start_ms)}")
    print(f"rounds={len(items)}")

    print(f"\nPeaks: total_round > {args.total_threshold_ms:.0f}ms")
    for item in items:
        if item["total"] > args.total_threshold_ms:
            print_item("TOTAL", item)

    print(f"\nPeaks: transformer > {args.transformer_threshold_ms:.0f}ms")
    for item in items:
        if item["transformer"] > args.transformer_threshold_ms:
            print_item("TRANS", item)

    print(f"\nPeaks: token2wav round-sum > {args.token2wav_threshold_ms:.0f}ms")
    for item in items:
        if item["token2wav"] > args.token2wav_threshold_ms:
            print_item("T2W  ", item)

    print(f"\nPeaks: actual input audio > {args.input_threshold_ms:.0f}ms")
    for item in items:
        if item["input"] > args.input_threshold_ms:
            print_item("INPUT", item)

    slow = [
        item for item in items
        if (
            item["total"] > args.cluster_total_ms
            or item["input"] > args.input_threshold_ms
            or item["transformer"] > args.transformer_threshold_ms
            or item["token2wav"] > args.token2wav_threshold_ms
        )
    ]
    clusters = []
    current = []
    for item in slow:
        if not current or item["round"] <= current[-1]["round"] + 1:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)

    print(
        "\nClusters: total>{:.0f}ms OR input>{:.0f}ms OR transformer>{:.0f}ms OR token2wav>{:.0f}ms".format(
            args.cluster_total_ms,
            args.input_threshold_ms,
            args.transformer_threshold_ms,
            args.token2wav_threshold_ms,
        )
    )
    for cluster in clusters:
        first = cluster[0]
        last = cluster[-1]
        round_text = f"r{first['round']}" if first["round"] == last["round"] else f"r{first['round']}-r{last['round']}"
        print(
            f"t={first['sec']:.3f}-{last['sec']:.3f}s {round_text} "
            f"max_total={max(x['total'] for x in cluster):.1f}ms "
            f"max_input={max(x['input'] for x in cluster):.1f}ms "
            f"max_transformer={max(x['transformer'] for x in cluster):.1f}ms "
            f"max_token2wav={max(x['token2wav'] for x in cluster):.1f}ms"
        )


if __name__ == "__main__":
    main()
