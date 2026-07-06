#!/usr/bin/env python3
"""Decode StepAudio vLLM token trace JSONL files into readable text.

Usage:
    python tools/decode_vllm_token_trace.py runtime_logs/.../vllm_token_trace_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = os.environ.get("STEPAUDIO_MODEL_PATH", "models/stepaudio_full_duplex")
TEXT_TOKEN_LIMIT = 151688
DEFAULT_TEXT_PAD_ID = 158358
DEFAULT_TTS_PAD_ID = 151695


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def _infer_model_path(trace_path: Path) -> str:
    startup_config = trace_path.parent / "startup_config.log"
    if startup_config.is_file():
        for line in startup_config.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model_path="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return os.environ.get("STEPAUDIO_MODEL_PATH", DEFAULT_MODEL_PATH)


def _append_token_list(out: list[int], value: Any) -> None:
    if isinstance(value, int):
        out.append(value)
        return
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, int):
            out.append(item)


def _extract_text_tokens_from_event(obj: dict[str, Any]) -> list[int]:
    tokens: list[int] = []

    # Current trace format: one text token per step event.
    if obj.get("type") == "step":
        _append_token_list(tokens, obj.get("text_token"))
        return tokens

    # Older / aggregated formats occasionally store a full token list at end.
    for key in (
        "generated_text_tokens",
        "generated_text_token_ids",
        "text_tokens",
        "text_token_ids",
        "valid_text_token_ids",
    ):
        _append_token_list(tokens, obj.get(key))

    return tokens


def _decode(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def decode_trace(trace_path: Path, model_path: str, output_path: Path, local_files_only: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
    special_text_ids: dict[str, Any] = {}
    line_count = 0

    for _lineno, obj in _read_jsonl(trace_path):
        line_count += 1
        if not isinstance(obj, dict):
            continue

        request_id = str(obj.get("request_id") or "(none)")
        rec = requests.setdefault(
            request_id,
            {
                "starts": 0,
                "ends": 0,
                "steps": 0,
                "raw": [],
                "step_raw": [],
                "aggregate_raw": [],
                "valid": [],
                "ts0": obj.get("timestamp_ms"),
                "ts1": obj.get("timestamp_ms"),
                "modes": Counter(),
                "terminations": Counter(),
                "reuse_true": 0,
                "reuse_false": 0,
            },
        )

        timestamp_ms = obj.get("timestamp_ms")
        if timestamp_ms is not None:
            if rec["ts0"] is None:
                rec["ts0"] = timestamp_ms
            rec["ts1"] = timestamp_ms

        event_type = obj.get("type")
        if event_type == "request_start":
            rec["starts"] += 1
            if obj.get("mode"):
                rec["modes"][obj.get("mode")] += 1
            if obj.get("reuse_request"):
                rec["reuse_true"] += 1
            else:
                rec["reuse_false"] += 1
            special = obj.get("special_token_ids") or {}
            if isinstance(special, dict) and isinstance(special.get("text"), dict):
                special_text_ids = special["text"]
        elif event_type == "request_end":
            rec["ends"] += 1
            if obj.get("termination_reason"):
                rec["terminations"][obj.get("termination_reason")] += 1

        if event_type == "step":
            rec["steps"] += 1

        event_tokens = _extract_text_tokens_from_event(obj)
        if event_type == "step":
            rec["step_raw"].extend(event_tokens)
        else:
            rec["aggregate_raw"].extend(event_tokens)

    text_pad_id = int(special_text_ids.get("text_pad_token_id", DEFAULT_TEXT_PAD_ID))
    tts_pad_id = int(special_text_ids.get("tts_pad_id", DEFAULT_TTS_PAD_ID))
    excluded = {text_pad_id, tts_pad_id}

    merged_valid_ids: list[int] = []
    for rec in requests.values():
        rec["raw"] = rec["step_raw"] if rec["step_raw"] else rec["aggregate_raw"]
        rec["valid"] = [
            tid for tid in rec["raw"]
            if isinstance(tid, int) and tid < TEXT_TOKEN_LIMIT and tid not in excluded
        ]
        merged_valid_ids.extend(rec["valid"])

    merged_text = _decode(tokenizer, merged_valid_ids)

    lines: list[str] = []
    lines.append("Decoded model text from vLLM token trace")
    lines.append(f"trace_file: {trace_path.resolve()}")
    lines.append(f"model_path: {model_path}")
    lines.append(f"line_count: {line_count}")
    lines.append(f"text_pad_id: {text_pad_id}")
    lines.append(f"tts_pad_id: {tts_pad_id}")
    lines.append(
        f"filter_rule: token_id < {TEXT_TOKEN_LIMIT} "
        f"and token_id not in {{{text_pad_id}, {tts_pad_id}}}"
    )
    lines.append(f"request_id_count: {len(requests)}")
    lines.append(f"raw_generated_text_token_count: {sum(len(r['raw']) for r in requests.values())}")
    lines.append(f"valid_text_token_count: {sum(len(r['valid']) for r in requests.values())}")
    lines.append("")
    lines.append("===== MERGED_VALID_TEXT =====")
    lines.append(merged_text or "(empty)")
    lines.append("")
    lines.append("===== BY_REQUEST =====")

    for idx, (request_id, rec) in enumerate(requests.items(), 1):
        decoded_text = _decode(tokenizer, rec["valid"])
        lines.append("")
        lines.append(f"--- request {idx}: {request_id} ---")
        lines.append(f"timestamp_ms: {rec['ts0']}..{rec['ts1']}")
        lines.append(
            "starts: {starts}, ends: {ends}, steps: {steps}, reuse_true: {reuse_true}, "
            "reuse_false: {reuse_false}, modes: {modes}, terminations: {terminations}".format(
                starts=rec["starts"],
                ends=rec["ends"],
                steps=rec["steps"],
                reuse_true=rec["reuse_true"],
                reuse_false=rec["reuse_false"],
                modes=dict(rec["modes"]),
                terminations=dict(rec["terminations"]),
            )
        )
        lines.append(f"valid_text_token_count: {len(rec['valid'])}")
        lines.append(f"valid_text_token_ids: {rec['valid']}")
        lines.append("decoded_text:")
        lines.append(decoded_text or "(empty)")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"decoded_txt={output_path.resolve()}")
    print(f"request_id_count={len(requests)}")
    print(f"valid_text_token_count={sum(len(r['valid']) for r in requests.values())}")
    print("MERGED_VALID_TEXT:")
    print(merged_text or "(empty)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode StepAudio vLLM token trace JSONL into readable text."
    )
    parser.add_argument("trace", type=Path, help="Path to vllm_token_trace_*.jsonl")
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Tokenizer/model path. Defaults to model_path in sibling startup_config.log, "
            "then STEPAUDIO_MODEL_PATH, then the v8_9_6 epoch-47260 path."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output txt path. Defaults to <trace_stem>_decoded_text.txt in the trace directory.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow transformers to fetch tokenizer files remotely. Default is local_files_only.",
    )
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    if not trace_path.is_file():
        raise SystemExit(f"trace file not found: {trace_path}")

    model_path = args.model_path or _infer_model_path(trace_path)
    output_path = args.output or trace_path.with_name(f"{trace_path.stem}_decoded_text.txt")

    decode_trace(
        trace_path=trace_path,
        model_path=model_path,
        output_path=output_path,
        local_files_only=not args.allow_remote,
    )


if __name__ == "__main__":
    main()
