"""Optional waveform augmentation and word-timestamp handling for E2E training."""

import math
from pathlib import Path
import warnings

import numpy as np


def word_intervals(source, user):
    """Return sorted, clipped word intervals in seconds relative to the user turn."""
    start, end = float(user["start_time"]), float(user["end_time"])
    local = user.get("force_align")
    alignment = local or source.get("user_audio_force_align")
    if not isinstance(alignment, dict) or "timestamps" not in alignment:
        raise ValueError("Pause supervision requires user.force_align.timestamps (local seconds) "
                         "or user_audio_force_align.timestamps (global seconds)")
    offset = 0.0 if local else start
    intervals = []
    for pair in alignment["timestamps"]:
        if len(pair) != 2:
            raise ValueError("Each force-align timestamp must be [start, end]")
        left, right = map(float, pair)
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise ValueError(f"Invalid force-align interval: {pair}")
        left, right = max(0.0, left - offset), min(end - start, right - offset)
        if right > left:
            intervals.append((left, right))
    return sorted(intervals)


def pause_control_positions(intervals, labels, keep_listening_id, slot_seconds,
                            threshold, harder_only=False):
    """Select K-L positions whose interval contains a word-gap end.

    Harder-only requires the gap end to fall in the final two audio slots.
    Positions are local to the current user segment, including interrupted turns.
    """
    pauses = [(a[1], b[0]) for a, b in zip(intervals, intervals[1:])
              if b[0] - a[1] > threshold]
    positions, previous_end = [], 0.0
    for pos, label in enumerate(labels):
        if label != keep_listening_id:
            continue
        end = (pos + 1) * slot_seconds
        if any(previous_end < gap_end <= end and
               (not harder_only or gap_end > end - 2 * slot_seconds)
               for _, gap_end in pauses):
            positions.append(pos)
        previous_end = end
    return positions


def mix_noise(audio, noise, speech_segments, sr, snr_db):
    """Scale noise using RMS over user speech segments; keep length and finite output."""
    if audio.shape != noise.shape or audio.ndim != 1:
        raise ValueError("Audio and noise must have the same one-dimensional shape")
    active = np.zeros(len(audio), dtype=bool)
    for start, end in speech_segments:
        active[max(0, int(start * sr)):min(len(audio), int(end * sr))] = True
    if not active.any():
        return audio
    signal_rms = np.sqrt(np.mean(audio[active].astype(np.float64) ** 2))
    noise_rms = np.sqrt(np.mean(noise[active].astype(np.float64) ** 2))
    if signal_rms <= 1e-12 or noise_rms <= 1e-12:
        return audio
    output = audio + noise * (signal_rms / noise_rms / 10 ** (snr_db / 20))
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * (0.99 / peak)
    return output.astype(np.float32)


def ambient_noise(length):
    """Generate the reference recipe's white/pink noise mixture."""
    if length < 2:
        return np.zeros(length, dtype=np.float32)
    white = np.random.randn(length)
    frequencies = np.fft.rfftfreq(length)
    frequencies[0] = frequencies[1]
    pink = np.fft.irfft(np.fft.rfft(np.random.randn(length)) / np.sqrt(frequencies), n=length)
    noise = (np.random.uniform(0.2, 0.4) * white / (white.std() + 1e-7) +
             np.random.uniform(0.6, 0.8) * pink / (pink.std() + 1e-7))
    return noise.astype(np.float32)


class DNSNoisePool:
    """Lazily sample local WAV files; never download or preload the noise corpus."""

    def __init__(self, directory, sr=16000):
        if not directory or not Path(directory).is_dir():
            raise ValueError("inject_dns_noise > 0 requires a valid dns_noise_dir")
        self.files = sorted(Path(directory).rglob("*.wav"))
        if not self.files:
            raise ValueError(f"No WAV noise files found in {directory}")
        self.sr = sr

    def sample_noise(self, length):
        import librosa

        output = np.zeros(length, dtype=np.float32)
        cursor = 0
        for _ in range(50):
            if cursor >= length:
                break
            path = self.files[np.random.randint(len(self.files))]
            try:
                clip, _ = librosa.load(path, sr=self.sr, mono=True)
            except (OSError, ValueError) as exc:
                warnings.warn(f"Skipping noise file {path}: {exc}")
                continue
            if not len(clip) or not np.isfinite(clip).all() or np.max(np.abs(clip)) > 0.99:
                continue
            count = min(len(clip), length - cursor)
            start = np.random.randint(len(clip) - count + 1)
            output[cursor:cursor + count] = clip[start:start + count]
            cursor += count
            cursor += min(length - cursor, int(np.random.uniform(0, 0.2) * self.sr))
        if cursor < length:
            warnings.warn("Noise sampling exhausted 50 attempts; remaining samples are silent")
        return output
