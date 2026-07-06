#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


STAGE_COLORS = {
    "input_audio_len": "#d9d9d9",
    "input_queue_wait": "#f2c94c",
    "round_to_first_audio_emit": "#bdbdbd",
    "audio_preprocess": "#56ccf2",
    "audio_encoder": "#2f80ed",
    "transformer": "#9b51e0",
    "token2wav": "#27ae60",
    "output_queue_send": "#eb5757",
}

STAGE_ORDER = [
    "input_audio_len",
    "input_queue_wait",
    "audio_preprocess",
    "audio_encoder",
    "transformer",
    "token2wav",
    "output_queue_send",
    "round_to_first_audio_emit",
]


def find_latest_jsonl(root: Path) -> Path:
    files = sorted(root.rglob("realtime_timeline_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No realtime_timeline_*.jsonl found under {root}")
    return files[-1]


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            rows.append(item)
    return rows


def span_duration_ms(span):
    try:
        return float(span.get("duration_ms"))
    except (TypeError, ValueError):
        pass
    try:
        return float(span.get("end_epoch_ms")) - float(span.get("start_epoch_ms"))
    except (TypeError, ValueError):
        return 0.0


def round_baseline_ms(row):
    timing = row.get("round_timing") if isinstance(row.get("round_timing"), dict) else {}
    for key in ("round_started_epoch_ms",):
        value = timing.get(key)
        if value is not None:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                pass
    value = row.get("round_started_at_epoch_ms")
    if value is not None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    starts = []
    for span in row.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        try:
            starts.append(int(float(span.get("start_epoch_ms"))))
        except (TypeError, ValueError):
            continue
    return min(starts) if starts else 0


def collect_stage_ms(row, name):
    total = 0.0
    count = 0
    summary = row.get("span_summary") if isinstance(row.get("span_summary"), dict) else {}
    item = summary.get(name) if isinstance(summary.get(name), dict) else None
    if item is not None:
        try:
            total = float(item.get("duration_ms") or 0.0)
            count = int(item.get("count") or 0)
            return total, count
        except (TypeError, ValueError):
            pass
    for span in row.get("spans", []) or []:
        if isinstance(span, dict) and span.get("name") == name:
            total += span_duration_ms(span)
            count += 1
    return total, count


def write_summary_csv(rows, output_csv: Path):
    fieldnames = [
        "round_id",
        "input_audio_ms",
        "total_round_ms",
        "stream_infer_ms",
        "audio_preprocess_ms",
        "audio_encoder_ms",
        "transformer_ms",
        "token2wav_ms",
        "input_queue_wait_ms",
        "output_queue_send_ms",
        "transformer_calls",
        "token2wav_calls",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            lat = row.get("latency_summary") if isinstance(row.get("latency_summary"), dict) else {}
            audio_pre, _ = collect_stage_ms(row, "audio_preprocess")
            audio_enc, _ = collect_stage_ms(row, "audio_encoder")
            transformer, transformer_calls = collect_stage_ms(row, "transformer")
            token2wav, token2wav_calls = collect_stage_ms(row, "token2wav")
            input_queue, _ = collect_stage_ms(row, "input_queue_wait")
            output_queue, _ = collect_stage_ms(row, "output_queue_send")
            writer.writerow({
                "round_id": row.get("round_id"),
                "input_audio_ms": round(float(row.get("input_duration_sec") or 0.0) * 1000.0, 3),
                "total_round_ms": round(float(lat.get("total_round_sec") or 0.0) * 1000.0, 3),
                "stream_infer_ms": round(float(lat.get("stream_infer_sec") or 0.0) * 1000.0, 3),
                "audio_preprocess_ms": round(audio_pre, 3),
                "audio_encoder_ms": round(audio_enc, 3),
                "transformer_ms": round(transformer, 3),
                "token2wav_ms": round(token2wav, 3),
                "input_queue_wait_ms": round(input_queue, 3),
                "output_queue_send_ms": round(output_queue, 3),
                "transformer_calls": transformer_calls,
                "token2wav_calls": token2wav_calls,
            })


def render_png(rows, output_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("No rows to render")

    n = len(rows)
    fig_height = max(8.0, min(48.0, n * 0.24 + 3.0))
    fig_width = 18.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    min_x = 0.0
    max_x = 400.0
    yticks = []
    yticklabels = []
    tick_stride = max(1, math.ceil(n / 40))

    for idx, row in enumerate(rows):
        y = idx
        baseline = round_baseline_ms(row)
        round_id = row.get("round_id", idx + 1)
        input_ms = float(row.get("input_duration_sec") or 0.0) * 1000.0
        max_x = max(max_x, input_ms)

        ax.barh(
            y,
            input_ms,
            left=0.0,
            height=0.82,
            color=STAGE_COLORS["input_audio_len"],
            alpha=0.35,
            edgecolor="none",
            zorder=0,
        )

        for span in row.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            name = str(span.get("name") or "unknown")
            try:
                start = float(span.get("start_epoch_ms")) - float(baseline)
            except (TypeError, ValueError):
                continue
            duration = span_duration_ms(span)
            if duration <= 0:
                duration = 0.8
            min_x = min(min_x, start)
            max_x = max(max_x, start + duration)
            color = STAGE_COLORS.get(name, "#828282")
            height = 0.50
            alpha = 0.90
            if name == "round_to_first_audio_emit":
                height = 0.18
                alpha = 0.40
            ax.barh(
                y,
                duration,
                left=start,
                height=height,
                color=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.15,
                zorder=2,
            )

        if idx % tick_stride == 0 or idx == n - 1:
            yticks.append(y)
            yticklabels.append(f"r{round_id} ({input_ms/1000.0:.2f}s)")

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axvline(400, color="#555555", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.text(402, -0.8, "400ms ref", fontsize=8, color="#555555", va="bottom")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ms relative to round_start; gray background = actual input audio length")
    ax.set_ylabel("round_id (actual processed audio seconds)")
    ax.set_title(title)
    ax.set_xlim(min_x - 30.0, max_x + 80.0)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)

    legend_items = [
        Patch(facecolor=STAGE_COLORS[name], label=name, alpha=0.75)
        for name in STAGE_ORDER
        if name in STAGE_COLORS
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render realtime timeline JSONL into a PNG chart and CSV summary.")
    parser.add_argument("jsonl", nargs="?", help="Path to realtime_timeline_*.jsonl. If omitted, use latest under runtime_logs.")
    parser.add_argument("--runtime-root", default="runtime_logs", help="Runtime logs root used when jsonl is omitted.")
    parser.add_argument("--output", help="Output PNG path. Defaults to <jsonl_stem>_timeline.png.")
    parser.add_argument("--csv", help="Output CSV path. Defaults to <jsonl_stem>_summary.csv.")
    parser.add_argument("--last", type=int, default=0, help="Only render the last N rounds. Default renders all rounds.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else find_latest_jsonl(Path(args.runtime_root))
    rows = load_rows(jsonl_path)
    if args.last and args.last > 0:
        rows = rows[-args.last:]

    output_png = Path(args.output) if args.output else jsonl_path.with_name(f"{jsonl_path.stem}_timeline.png")
    output_csv = Path(args.csv) if args.csv else jsonl_path.with_name(f"{jsonl_path.stem}_summary.csv")
    title = f"{jsonl_path.name} ({len(rows)} rounds)"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    render_png(rows, output_png, title)
    write_summary_csv(rows, output_csv)
    print(f"jsonl={jsonl_path}")
    print(f"png={output_png}")
    print(f"csv={output_csv}")
    print(f"rounds={len(rows)}")


if __name__ == "__main__":
    main()
