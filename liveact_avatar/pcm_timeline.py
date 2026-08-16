from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


def output_chunks_for_samples(
    received_samples: int,
    *,
    sample_rate: int,
    fps: int,
    first_chunk_frames: int,
    steady_chunk_frames: int,
) -> int:
    """Return the fewest model chunks whose media time covers all PCM.

    LiveAct's first decoded chunk is shorter than every steady-state chunk.
    Counting only steady-state steps can therefore stop with real reply PCM
    still waiting behind the mux cursor.  Work in integer frame/sample units
    so the result agrees with the writer's cumulative media timeline.
    """

    received_samples = max(0, int(received_samples))
    if received_samples == 0:
        return 0
    sample_rate = max(1, int(sample_rate))
    fps = max(1, int(fps))
    first_chunk_frames = max(1, int(first_chunk_frames))
    steady_chunk_frames = max(1, int(steady_chunk_frames))

    required_frames = (
        received_samples * fps + sample_rate - 1
    ) // sample_rate
    if required_frames <= first_chunk_frames:
        return 1
    remaining_frames = required_frames - first_chunk_frames
    return 1 + (
        remaining_frames + steady_chunk_frames - 1
    ) // steady_chunk_frames


@dataclass(frozen=True)
class PcmWindow:
    """Immutable snapshot of one model input window."""

    start_sample: int
    end_sample: int
    real_samples: int
    padded_samples: int
    samples: np.ndarray


@dataclass(frozen=True)
class PcmTimelineStats:
    sample_rate: int
    received_samples: int
    retained_start_sample: int
    retained_samples: int
    muxed_samples: int
    queued_source_samples: int
    finished: bool
    version: int

    @property
    def received_sec(self) -> float:
        return self.received_samples / float(max(1, self.sample_rate))

    @property
    def retained_sec(self) -> float:
        return self.retained_samples / float(max(1, self.sample_rate))

    @property
    def muxed_sec(self) -> float:
        return self.muxed_samples / float(max(1, self.sample_rate))

    @property
    def queued_real_samples(self) -> int:
        return max(0, int(self.queued_source_samples))

    @property
    def queued_real_sec(self) -> float:
        return self.queued_real_samples / float(max(1, self.sample_rate))


class StreamingPcmTimeline:
    """Thread-safe, append-only PCM timeline with a separate mux cursor.

    In finite-file mode callers may append only source PCM and request tail
    padding after ``finish()``.  The continuous Avatar scheduler instead
    appends either real Token2Wav PCM or explicit zero PCM before freezing each
    LiveAct window.  Both modes use immutable overlapping model snapshots and
    one non-overlapping media cursor.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = max(1, int(sample_rate))
        self._condition = threading.Condition(threading.RLock())
        self._samples = np.zeros(0, dtype=np.float32)
        self._source_real = np.zeros(0, dtype=np.bool_)
        self._retained_start_sample = 0
        self._received_samples = 0
        self._muxed_samples = 0
        self._finished = False
        self._version = 0

    def append(
        self,
        samples: np.ndarray,
        *,
        is_last: bool = False,
        source_real: bool = True,
    ) -> int:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size:
            values = np.ascontiguousarray(values.copy())
        with self._condition:
            expected_end = self._retained_start_sample + int(self._samples.size)
            if expected_end != self._received_samples:
                raise RuntimeError(
                    "PCM timeline is not contiguous: "
                    f"retained_end={expected_end}, received={self._received_samples}"
                )
            if values.size:
                self._samples = np.concatenate([self._samples, values])
                self._source_real = np.concatenate(
                    [
                        self._source_real,
                        np.full(values.size, bool(source_real), dtype=np.bool_),
                    ]
                )
                self._received_samples += int(values.size)
            if is_last:
                self._finished = True
            if values.size or is_last:
                self._version += 1
                self._condition.notify_all()
            return int(values.size)

    def replace_earliest_silence(
        self,
        start_sample: int,
        samples: np.ndarray,
    ) -> tuple[int, Optional[int]]:
        """Replace the earliest mutable silence run with real source PCM.

        The continuous scheduler may need audio lookahead before Token2Wav has
        produced the next response, so it materializes zero PCM.  Real PCM that
        arrives later may replace only a contiguous silence run at or after the
        caller-provided frozen boundary.  The absolute replacement start is
        returned for diagnostics.
        """

        values = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        )
        if values.size <= 0:
            return 0, None
        with self._condition:
            absolute_start = min(
                max(int(start_sample), self._retained_start_sample),
                self._received_samples,
            )
            relative_start = absolute_start - self._retained_start_sample
            available_mask = self._source_real[relative_start:]
            silent_offsets = np.flatnonzero(~available_mask)
            if silent_offsets.size <= 0:
                return 0, None
            run_start_rel = relative_start + int(silent_offsets[0])
            run_mask = self._source_real[run_start_rel:]
            next_real = np.flatnonzero(run_mask)
            run_size = int(next_real[0]) if next_real.size else int(run_mask.size)
            take = min(int(values.size), run_size)
            if take <= 0:
                return 0, None
            run_end_rel = run_start_rel + take
            self._samples[run_start_rel:run_end_rel] = values[:take]
            self._source_real[run_start_rel:run_end_rel] = True
            self._version += 1
            self._condition.notify_all()
            return take, self._retained_start_sample + run_start_rel

    def replace_future_with_silence(self, start_sample: int) -> int:
        """Cancel real PCM at/after ``start_sample`` without moving time."""

        with self._condition:
            absolute_start = min(
                max(int(start_sample), self._retained_start_sample),
                self._received_samples,
            )
            relative_start = absolute_start - self._retained_start_sample
            cancelled_real = int(np.count_nonzero(self._source_real[relative_start:]))
            if relative_start < self._samples.size:
                self._samples[relative_start:] = 0.0
                self._source_real[relative_start:] = False
            if cancelled_real:
                self._version += 1
                self._condition.notify_all()
            return cancelled_real

    def reset(self) -> None:
        with self._condition:
            self._samples = np.zeros(0, dtype=np.float32)
            self._source_real = np.zeros(0, dtype=np.bool_)
            self._retained_start_sample = 0
            self._received_samples = 0
            self._muxed_samples = 0
            self._finished = False
            self._version += 1
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            if not self._finished:
                self._finished = True
                self._version += 1
                self._condition.notify_all()

    def stats(self) -> PcmTimelineStats:
        with self._condition:
            unmuxed_start = min(
                max(self._muxed_samples - self._retained_start_sample, 0),
                int(self._source_real.size),
            )
            return PcmTimelineStats(
                sample_rate=self.sample_rate,
                received_samples=int(self._received_samples),
                retained_start_sample=int(self._retained_start_sample),
                retained_samples=int(self._samples.size),
                muxed_samples=int(self._muxed_samples),
                queued_source_samples=int(
                    np.count_nonzero(self._source_real[unmuxed_start:])
                ),
                finished=bool(self._finished),
                version=int(self._version),
            )

    def wait_for_update(self, version: int, timeout: float) -> int:
        with self._condition:
            if self._version == int(version):
                self._condition.wait(timeout=max(0.0, float(timeout)))
            return int(self._version)

    def snapshot_window(
        self,
        start_sample: int,
        end_sample: int,
        *,
        allow_finished_padding: bool,
    ) -> Optional[PcmWindow]:
        start_sample = max(0, int(start_sample))
        end_sample = max(start_sample + 1, int(end_sample))
        with self._condition:
            if start_sample < self._retained_start_sample:
                raise RuntimeError(
                    "requested PCM window was already compacted: "
                    f"start={start_sample}, retained_start={self._retained_start_sample}"
                )
            if end_sample > self._received_samples:
                if not (allow_finished_padding and self._finished):
                    return None
                if start_sample >= self._received_samples:
                    return None

            real_end = min(end_sample, self._received_samples)
            relative_start = start_sample - self._retained_start_sample
            relative_end = real_end - self._retained_start_sample
            values = np.ascontiguousarray(
                self._samples[relative_start:relative_end].copy(),
                dtype=np.float32,
            )
            source_real = self._source_real[relative_start:relative_end]
            requested_samples = end_sample - start_sample
            available_samples = int(values.size)
            missing_samples = max(0, requested_samples - available_samples)
            if missing_samples:
                values = np.pad(values, (0, missing_samples), mode="constant")
            real_samples = int(np.count_nonzero(source_real))
            # Explicit scheduler silence and finite-stream tail padding are
            # both non-source samples from the model's point of view.
            padded_samples = max(0, requested_samples - real_samples)
            if values.size != requested_samples:
                raise RuntimeError(
                    f"invalid PCM snapshot size: got={values.size}, expected={requested_samples}"
                )
            return PcmWindow(
                start_sample=start_sample,
                end_sample=end_sample,
                real_samples=real_samples,
                padded_samples=padded_samples,
                samples=values,
            )

    def read_for_mux(
        self,
        sample_count: int,
        *,
        allow_finished_padding: bool,
    ) -> Optional[PcmWindow]:
        sample_count = max(0, int(sample_count))
        with self._condition:
            start_sample = int(self._muxed_samples)
        if sample_count == 0:
            return PcmWindow(
                start_sample=start_sample,
                end_sample=start_sample,
                real_samples=0,
                padded_samples=0,
                samples=np.zeros(0, dtype=np.float32),
            )
        return self.snapshot_window(
            start_sample,
            start_sample + sample_count,
            allow_finished_padding=allow_finished_padding,
        )

    def advance_muxed(self, target_sample: int) -> None:
        target_sample = max(0, int(target_sample))
        with self._condition:
            if target_sample < self._muxed_samples:
                raise RuntimeError(
                    f"PCM mux cursor cannot move backwards: {target_sample} < {self._muxed_samples}"
                )
            self._muxed_samples = target_sample
            self._version += 1
            self._condition.notify_all()

    def discard_before(self, sample_index: int) -> int:
        with self._condition:
            cutoff = min(
                max(int(sample_index), self._retained_start_sample),
                self._received_samples,
            )
            drop = cutoff - self._retained_start_sample
            if drop <= 0:
                return 0
            self._samples = np.ascontiguousarray(self._samples[drop:].copy())
            self._source_real = np.ascontiguousarray(
                self._source_real[drop:].copy(), dtype=np.bool_
            )
            self._retained_start_sample = cutoff
            self._version += 1
            return int(drop)
