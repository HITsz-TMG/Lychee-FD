#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


STAGE_COLORS = {
    "input_audio_len": "#e5e7eb",
    "input_queue_wait": "#f59e0b",
    "audio_preprocess": "#38bdf8",
    "audio_encoder": "#2563eb",
    "transformer": "#7c3aed",
    "token2wav": "#16a34a",
    "output_queue_send": "#ef4444",
    "round_to_first_audio_emit": "#9ca3af",
}

STATE_BG_COLORS = {
    "listening": "#eff6ff",
    "speaking": "#fef2f2",
    "backchannel": "#f5f3ff",
    "mixed": "#f8fafc",
    "unknown": "#ffffff",
}

STATE_LABELS = {
    "listening": "listening",
    "speaking": "speaking",
    "backchannel": "backchannel",
    "mixed": "mixed state",
    "unknown": "unknown state",
}

STAGE_LABELS = {
    "input_audio_len": "actual audio length",
    "input_queue_wait": "input queue wait",
    "audio_preprocess": "audio preprocess",
    "audio_encoder": "audio encoder",
    "transformer": "transformer / vLLM",
    "token2wav": "token2wav",
    "output_queue_send": "SSE queue/send",
    "round_to_first_audio_emit": "round -> first audio emit",
}

VISIBLE_STAGE_ORDER = [
    "input_queue_wait",
    "audio_preprocess",
    "audio_encoder",
    "transformer",
    "token2wav",
    "output_queue_send",
]


def normalize_state(value):
    state = str(value or "").strip().lower()
    if state in {"s", "speak", "speaking", "response"}:
        return "speaking"
    if state in {"l", "listen", "listening"}:
        return "listening"
    if state in {"b", "bc", "backchannel", "backchannel_recheck"}:
        return "backchannel"
    if "backchannel" in state:
        return "backchannel"
    if "speak" in state:
        return "speaking"
    if "listen" in state:
        return "listening"
    return "unknown"


def dominant_round_state(row):
    totals = {}
    for span in row.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        if str(span.get("name") or "") != "transformer":
            continue
        state = normalize_state(span.get("state"))
        totals[state] = totals.get(state, 0.0) + span_duration_ms(span)
    if not totals:
        return "unknown"
    total = sum(totals.values())
    state, duration = max(totals.items(), key=lambda item: item[1])
    if len(totals) > 1 and total > 0 and duration / total < 0.60:
        return "mixed"
    return state


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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_start_epoch_ms(row):
    timing = row.get("round_timing") if isinstance(row.get("round_timing"), dict) else {}
    for value in (timing.get("round_started_epoch_ms"), row.get("round_started_at_epoch_ms")):
        v = as_float(value)
        if v is not None:
            return v
    starts = []
    for span in row.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        v = as_float(span.get("start_epoch_ms"))
        if v is not None:
            starts.append(v)
    return min(starts) if starts else None


def round_end_epoch_ms(row):
    value = as_float(row.get("round_completed_at_epoch_ms"))
    if value is not None:
        return value
    starts = []
    ends = []
    for span in row.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        start = as_float(span.get("start_epoch_ms"))
        end = as_float(span.get("end_epoch_ms"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if ends:
        return max(ends)
    start = round_start_epoch_ms(row)
    total_sec = as_float((row.get("latency_summary") or {}).get("total_round_sec"))
    if start is not None and total_sec is not None:
        return start + max(0.0, total_sec * 1000.0)
    return max(starts) if starts else None


def span_duration_ms(span):
    v = as_float(span.get("duration_ms"))
    if v is not None:
        return max(0.0, v)
    start = as_float(span.get("start_epoch_ms"))
    end = as_float(span.get("end_epoch_ms"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def build_round_intervals(rows):
    intervals = []
    for row_index, row in enumerate(rows):
        start = round_start_epoch_ms(row)
        end = round_end_epoch_ms(row)
        if start is None:
            continue
        if end is None or end < start:
            end = start
        intervals.append({
            "row_index": row_index,
            "round_id": row.get("round_id", row_index + 1),
            "start_epoch_ms": float(start),
            "end_epoch_ms": float(end),
        })
    return intervals


def choose_visual_round(span_start_ms, span_end_ms, original_row_index, round_intervals):
    """
    Assign a span to the round lane whose real processing interval overlaps it
    the most. This keeps async token2wav spans on the row that was active when
    the worker really ran, instead of the row that later drained the message.
    """
    if span_start_ms is None:
        return original_row_index
    if span_end_ms is None or span_end_ms < span_start_ms:
        span_end_ms = span_start_ms

    best_row = original_row_index
    best_overlap = -1.0
    span_mid = (float(span_start_ms) + float(span_end_ms)) / 2.0
    best_mid_distance = float("inf")
    for interval in round_intervals:
        overlap = min(float(span_end_ms), interval["end_epoch_ms"]) - max(float(span_start_ms), interval["start_epoch_ms"])
        overlap = max(0.0, overlap)
        interval_mid = (interval["start_epoch_ms"] + interval["end_epoch_ms"]) / 2.0
        mid_distance = abs(span_mid - interval_mid)
        if overlap > best_overlap or (overlap == best_overlap and mid_distance < best_mid_distance):
            best_overlap = overlap
            best_mid_distance = mid_distance
            best_row = int(interval["row_index"])

    if best_overlap > 0.0:
        return best_row

    # If the span lands in the gap between rounds, use the nearest active lane.
    nearest_row = original_row_index
    nearest_distance = float("inf")
    for interval in round_intervals:
        if span_mid < interval["start_epoch_ms"]:
            distance = interval["start_epoch_ms"] - span_mid
        elif span_mid > interval["end_epoch_ms"]:
            distance = span_mid - interval["end_epoch_ms"]
        else:
            distance = 0.0
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_row = int(interval["row_index"])
    return nearest_row


def collect_plot_items(rows):
    starts = []
    for row in rows:
        start = round_start_epoch_ms(row)
        if start is not None:
            starts.append(start)
        for span in row.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            span_start = as_float(span.get("start_epoch_ms"))
            if span_start is not None:
                starts.append(span_start)
    if not starts:
        raise ValueError("No epoch timestamps found in timeline rows")
    session_t0 = min(starts)

    items = []
    for row_index, row in enumerate(rows):
        round_id = row.get("round_id", row_index + 1)
        start = round_start_epoch_ms(row)
        if start is None:
            continue
        input_ms = max(0.0, float(row.get("input_duration_sec") or 0.0) * 1000.0)
        total_ms = max(0.0, float((row.get("latency_summary") or {}).get("total_round_sec") or 0.0) * 1000.0)
        for span in row.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            name = str(span.get("name") or "unknown")
            if name == "round_to_first_audio_emit":
                continue
            span_start = as_float(span.get("start_epoch_ms"))
            if span_start is None:
                continue
            duration = span_duration_ms(span)
            if duration <= 0.0:
                duration = 0.8
            state = span.get("state")
            token_count = span.get("target_new_tokens", span.get("tokens"))
            details = []
            if state is not None:
                details.append(f"state={state}")
            if token_count is not None:
                details.append(f"tokens={token_count}")
            detail_text = (" " + " ".join(details)) if details else ""
            items.append({
                "kind": name,
                "round_id": round_id,
                "row_index": row_index,
                "start_ms": span_start - session_t0,
                "duration_ms": duration,
                "label": f"r{round_id} {name} {duration:.1f}ms{detail_text}",
                "input_ms": input_ms,
                "total_ms": total_ms,
                "span": span,
            })
    return session_t0, items


def collect_workload_items(rows):
    starts = [round_start_epoch_ms(row) for row in rows]
    starts = [x for x in starts if x is not None]
    if not starts:
        raise ValueError("No round start timestamps found in timeline rows")
    session_t0 = min(starts)
    round_intervals = build_round_intervals(rows)

    items = []
    for row_index, row in enumerate(rows):
        round_id = row.get("round_id", row_index + 1)
        round_start = round_start_epoch_ms(row)
        if round_start is None:
            continue
        session_sec = (round_start - session_t0) / 1000.0
        input_ms = max(0.0, float(row.get("input_duration_sec") or 0.0) * 1000.0)
        total_ms = max(0.0, float((row.get("latency_summary") or {}).get("total_round_sec") or 0.0) * 1000.0)
        items.append({
            "kind": "input_audio_len",
            "round_id": round_id,
            "row_index": row_index,
            "start_ms": 0.0,
            "duration_ms": input_ms,
            "label": f"r{round_id} @ {session_sec:.3f}s audio workload {input_ms:.1f}ms total {total_ms:.1f}ms",
            "input_ms": input_ms,
            "total_ms": total_ms,
            "session_sec": session_sec,
        })
        for span in row.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            name = str(span.get("name") or "unknown")
            if name == "round_to_first_audio_emit":
                continue
            span_start_epoch = as_float(span.get("start_epoch_ms"))
            if span_start_epoch is None:
                continue
            span_end_epoch = as_float(span.get("end_epoch_ms"))
            duration = span_duration_ms(span)
            if duration <= 0.0:
                duration = 0.8
            visual_row_index = row_index
            visual_round_id = round_id
            if name == "token2wav":
                visual_row_index = choose_visual_round(
                    span_start_epoch,
                    span_end_epoch if span_end_epoch is not None else span_start_epoch + duration,
                    row_index,
                    round_intervals,
                )
                if 0 <= visual_row_index < len(rows):
                    visual_round_id = rows[visual_row_index].get("round_id", visual_row_index + 1)
            state = span.get("state")
            token_count = span.get("target_new_tokens", span.get("tokens"))
            details = []
            if state is not None:
                details.append(f"state={state}")
            if token_count is not None:
                details.append(f"tokens={token_count}")
            if visual_row_index != row_index:
                details.append(f"drained_in=r{round_id}")
            detail_text = (" " + " ".join(details)) if details else ""
            items.append({
                "kind": name,
                "round_id": visual_round_id,
                "source_round_id": round_id,
                "row_index": visual_row_index,
                "source_row_index": row_index,
                "start_ms": span_start_epoch - round_start,
                "absolute_start_ms": span_start_epoch - session_t0,
                "duration_ms": duration,
                "label": f"r{visual_round_id} {name} {duration:.1f}ms{detail_text}",
                "input_ms": input_ms,
                "total_ms": total_ms,
                "session_sec": session_sec,
                "span": span,
            })
    return session_t0, items


def render_png(rows, output_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _, items = collect_workload_items(rows)
    n = len(rows)
    row_states = [dominant_round_state(row) for row in rows]
    fig_height = max(7.0, min(54.0, n * 0.30 + 3.2))
    fig, ax = plt.subplots(figsize=(22.0, fig_height))

    y_lookup = {idx: idx for idx in range(n)}
    max_x = 400.0
    min_x = 0.0

    def aligned_start_ms(item):
        return float(item.get("session_sec", 0.0)) * 1000.0 + float(item.get("start_ms", 0.0))

    for row_index, state in enumerate(row_states):
        color = STATE_BG_COLORS.get(state, STATE_BG_COLORS["unknown"])
        ax.axhspan(row_index - 0.48, row_index + 0.48, color=color, alpha=0.72, zorder=-4)

    # Draw the actual audio workload first. It is anchored at the real
    # round_start on the session timeline, so adjacent rounds can show overlap.
    for item in items:
        if item["kind"] != "input_audio_len":
            continue
        y = y_lookup.get(item["row_index"])
        if y is None:
            continue
        start = aligned_start_ms(item)
        duration = float(item["duration_ms"])
        max_x = max(max_x, start + duration)
        min_x = min(min_x, start)
        ax.barh(
            y,
            duration,
            left=start,
            height=0.82,
            color=STAGE_COLORS["input_audio_len"],
            alpha=0.70,
            edgecolor="#9ca3af",
            linewidth=0.35,
            zorder=0,
        )
        if n <= 70 and duration > 0:
            ax.text(
                start + duration + max(10.0, max_x * 0.002),
                y,
                f"audio {duration/1000.0:.2f}s",
                fontsize=7,
                va="center",
                color="#4b5563",
            )

    for item in items:
        kind = item["kind"]
        if kind == "input_audio_len":
            continue
        y = y_lookup.get(item["row_index"])
        if y is None:
            continue
        start = aligned_start_ms(item)
        duration = float(item["duration_ms"])
        max_x = max(max_x, start + duration)
        min_x = min(min_x, start)
        ax.barh(
            y,
            duration,
            left=start,
            height=0.52,
            color=STAGE_COLORS.get(kind, "#6b7280"),
            alpha=0.94,
            edgecolor="#111827",
            linewidth=0.18,
            zorder=2,
        )

    tick_stride = max(1, math.ceil(n / 55))
    yticks = []
    ylabels = []
    session_by_row = {}
    for item in items:
        if item["kind"] == "input_audio_len":
            session_by_row[item["row_index"]] = float(item.get("session_sec", 0.0))
    for i, row in enumerate(rows):
        if i % tick_stride == 0 or i == n - 1:
            input_sec = float(row.get("input_duration_sec") or 0.0)
            round_id = row.get("round_id", i + 1)
            session_sec = session_by_row.get(i, 0.0)
            state = STATE_LABELS.get(row_states[i], row_states[i])
            yticks.append(i)
            ylabels.append(f"r{round_id} @ {session_sec:.1f}s | {state} | audio {input_sec:.2f}s")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("session elapsed time (ms); gray audio workload is anchored at each round_start")
    ax.set_ylabel("round start time in session | actual processed audio length")
    ax.set_title(title + " - session-aligned workload view")
    ax.set_xlim(min_x - 120.0, max_x + 320.0)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)

    legend_handles = [
        Patch(facecolor=STATE_BG_COLORS["listening"], edgecolor="none", label="row bg: listening", alpha=0.72),
        Patch(facecolor=STATE_BG_COLORS["speaking"], edgecolor="none", label="row bg: speaking", alpha=0.72),
        Patch(facecolor=STATE_BG_COLORS["backchannel"], edgecolor="none", label="row bg: backchannel", alpha=0.72),
        Patch(facecolor=STATE_BG_COLORS["mixed"], edgecolor="none", label="row bg: mixed state", alpha=0.72),
        Patch(facecolor=STAGE_COLORS["input_audio_len"], edgecolor="#9ca3af", label="audio workload length anchored at round_start", alpha=0.70),
    ]
    for name in VISIBLE_STAGE_ORDER:
        legend_handles.append(Patch(facecolor=STAGE_COLORS[name], label=STAGE_LABELS.get(name, name), alpha=0.94))
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        ncol=1,
        fontsize=9,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.16, right=0.82, top=0.92, bottom=0.12)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)

def html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(rows, output_html: Path, title: str):
    _, items = collect_workload_items(rows)
    n = len(rows)
    row_states = [dominant_round_state(row) for row in rows]
    row_h = 28
    left_pad = 230
    top_pad = 44
    right_pad = 80
    bottom_pad = 96

    def aligned_start_ms(item):
        return float(item.get("session_sec", 0.0)) * 1000.0 + float(item.get("start_ms", 0.0))

    max_x = max((aligned_start_ms(item) + item["duration_ms"] for item in items), default=400.0)
    min_x = min((aligned_start_ms(item) for item in items), default=0.0)
    min_x = min(min_x, 0.0)
    width = 1800
    height = top_pad + n * row_h + bottom_pad
    plot_w = width - left_pad - right_pad
    scale = plot_w / max(1.0, max_x - min_x)

    def sx(ms):
        return left_pad + (float(ms) - min_x) * scale

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html_escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#111827;background:#fff;}",
        ".hint{color:#4b5563;font-size:13px;margin:8px 0 16px;}",
        "svg{border:1px solid #e5e7eb;background:#ffffff;max-width:100%;height:auto;}",
        ".axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:11px;fill:#374151}.title{font-size:16px;font-weight:700;fill:#111827}.tick{font-size:10px;fill:#6b7280}",
        "</style></head><body>",
        f"<h2>{html_escape(title)} - session-aligned workload view</h2>",
        "<div class='hint'>每行是一个 round。横轴是整段会话时间；浅色行背景表示该轮 transformer 的主状态：蓝色 listening，红色 speaking，紫色 backchannel；灰色条表示该 round 在 round_start 时将要处理的真实音频长度；彩色条是后端真实阶段。竖向对齐表示不同 round 的阶段在同一段墙钟时间发生，可观察重叠/复用。</div>",
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<text class='tick' x='{left_pad}' y='{top_pad-20}'>session elapsed time</text>",
    ]

    span_ms = max_x - min_x
    tick_step = 1000.0
    if span_ms > 90000:
        tick_step = 10000.0
    elif span_ms > 45000:
        tick_step = 5000.0
    elif span_ms > 18000:
        tick_step = 2000.0
    first_tick = math.floor(min_x / tick_step) * tick_step
    tick = first_tick
    while tick <= max_x + tick_step:
        x = sx(tick)
        parts.append(f"<line class='grid' x1='{x:.1f}' y1='{top_pad-14}' x2='{x:.1f}' y2='{height-bottom_pad+12}'/>")
        parts.append(f"<text class='tick' x='{x+3:.1f}' y='{top_pad-20}'>{tick/1000.0:.1f}s</text>")
        tick += tick_step

    session_by_row = {}
    for item in items:
        if item["kind"] == "input_audio_len":
            session_by_row[item["row_index"]] = float(item.get("session_sec", 0.0))
    for i, row in enumerate(rows):
        y = top_pad + i * row_h
        input_sec = float(row.get("input_duration_sec") or 0.0)
        session_sec = session_by_row.get(i, 0.0)
        state = row_states[i]
        state_label = STATE_LABELS.get(state, state)
        label = f"r{row.get('round_id', i + 1)} @ {session_sec:.1f}s | {state_label} | audio {input_sec:.2f}s"
        parts.append(f"<text class='label' x='8' y='{y+16}'>{html_escape(label)}</text>")
        bg = STATE_BG_COLORS.get(state, STATE_BG_COLORS["unknown"])
        parts.append(
            f"<rect x='{left_pad}' y='{y+2}' width='{plot_w}' height='{row_h-3}' "
            f"fill='{bg}' fill-opacity='0.72'/>"
        )

    ordered = [item for item in items if item["kind"] == "input_audio_len"] + [item for item in items if item["kind"] != "input_audio_len"]
    for item in ordered:
        y = top_pad + item["row_index"] * row_h
        x = sx(aligned_start_ms(item))
        w = max(1.0, float(item["duration_ms"]) * scale)
        kind = item["kind"]
        if kind == "input_audio_len":
            h = 20
            yy = y + 4
            opacity = "0.70"
            stroke = "#9ca3af"
        else:
            h = 14
            yy = y + 7
            opacity = "0.94"
            stroke = "#111827"
        color = STAGE_COLORS.get(kind, "#6b7280")
        title_text = item.get("label", kind)
        parts.append(
            f"<rect x='{x:.1f}' y='{yy:.1f}' width='{w:.1f}' height='{h}' fill='{color}' "
            f"fill-opacity='{opacity}' stroke='{stroke}' stroke-width='0.4' rx='2'>"
            f"<title>{html_escape(title_text)}</title></rect>"
        )

    legend_x = left_pad
    legend_y = height - 42
    state_legend_items = ["listening", "speaking", "backchannel", "mixed"]
    for state in state_legend_items:
        color = STATE_BG_COLORS[state]
        label = f"row bg: {STATE_LABELS.get(state, state)}"
        parts.append(f"<rect x='{legend_x}' y='{legend_y}' width='14' height='10' fill='{color}' fill-opacity='0.72' stroke='#111827' stroke-width='0.2'/>")
        parts.append(f"<text class='label' x='{legend_x+19}' y='{legend_y+10}'>{html_escape(label)}</text>")
        legend_x += max(150, len(label) * 7 + 34)
        if legend_x > width - 260:
            legend_x = left_pad
            legend_y += 18

    legend_items = ["input_audio_len"] + VISIBLE_STAGE_ORDER
    for name in legend_items:
        color = STAGE_COLORS[name]
        label = "audio workload length anchored at round_start" if name == "input_audio_len" else STAGE_LABELS.get(name, name)
        parts.append(f"<rect x='{legend_x}' y='{legend_y}' width='14' height='10' fill='{color}' stroke='#111827' stroke-width='0.2'/>")
        parts.append(f"<text class='label' x='{legend_x+19}' y='{legend_y+10}'>{html_escape(label)}</text>")
        legend_x += max(150, len(label) * 7 + 34)
        if legend_x > width - 260:
            legend_x = left_pad
            legend_y += 18

    parts.append("</svg></body></html>")
    output_html.write_text("\n".join(parts), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Render realtime timeline JSONL as a plain Gantt chart.")
    parser.add_argument("jsonl", nargs="?", help="Path to realtime_timeline_*.jsonl. If omitted, use latest under runtime_logs.")
    parser.add_argument("--runtime-root", default="runtime_logs")
    parser.add_argument("--last", type=int, default=0, help="Only render last N rounds.")
    parser.add_argument("--batch-size", type=int, default=0, help="Render multiple charts, each containing N rounds.")
    parser.add_argument("--overlap", type=int, default=0, help="When --batch-size is used, overlap this many rounds between adjacent charts.")
    parser.add_argument("--output-prefix", help="Output prefix. Defaults to <jsonl_stem>_gantt or <jsonl_stem>_lastN_gantt.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl) if args.jsonl else find_latest_jsonl(Path(args.runtime_root))
    rows = load_rows(jsonl_path)
    if args.last and args.last > 0:
        rows = rows[-args.last:]
    suffix = f"_last{args.last}" if args.last and args.last > 0 else ""
    output_prefix = Path(args.output_prefix) if args.output_prefix else jsonl_path.with_name(f"{jsonl_path.stem}{suffix}_gantt")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    def render_one(batch_rows, prefix: Path, title_suffix: str):
        title = f"Realtime Gantt - {jsonl_path.name} {title_suffix} ({len(batch_rows)} rounds)"
        png_path = prefix.with_suffix(".png")
        html_path = prefix.with_suffix(".html")
        render_png(batch_rows, png_path, title)
        render_html(batch_rows, html_path, title)
        print(f"png={png_path}")
        print(f"html={html_path}")
        print(f"rounds={len(batch_rows)}")

    print(f"jsonl={jsonl_path}")
    if args.batch_size and args.batch_size > 0:
        batch_size = max(1, int(args.batch_size))
        overlap = max(0, min(int(args.overlap), batch_size - 1))
        step = max(1, batch_size - overlap)
        batch_idx = 1
        start = 0
        while start < len(rows):
            end = min(len(rows), start + batch_size)
            batch_rows = rows[start:end]
            if not batch_rows:
                break
            first_round = batch_rows[0].get("round_id", start + 1)
            last_round = batch_rows[-1].get("round_id", end)
            batch_prefix = output_prefix.with_name(
                f"{output_prefix.name}_batch{batch_idx:02d}_r{first_round}-{last_round}"
            )
            render_one(batch_rows, batch_prefix, f"r{first_round}-{last_round}")
            if end >= len(rows):
                break
            start += step
            batch_idx += 1
    else:
        render_one(rows, output_prefix, "")


if __name__ == "__main__":
    main()
