"""Streaming PCM derivative of the original ``demo.py`` inference service.

The model, distributed execution, warmup, cache, solver, latent blocks and
whole-chunk FFmpeg path intentionally follow the Demo. Only the complete-WAV
task input is replaced by a thread-safe PCM timeline and window scheduler.
"""

import base64
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from PIL import Image
from torchvision import transforms

from liveact_runtime.fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from liveact_avatar.defaults import DEFAULT_AVATAR_PROMPT
from liveact_avatar.pcm_timeline import StreamingPcmTimeline
from liveact_runtime.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from liveact_runtime.util_liveact import (
    center_rescale_crop_keep_ratio,
    get_audio_emb,
    get_embedding,
    get_msk,
)
from liveact_runtime.wan.modules.clip import CLIPModel
from liveact_runtime.wan.modules.t5 import T5EncoderModel


M3U8_NAME = "live.m3u8"

# Keep the derived streaming engine on the same CUDA execution settings as the
# original demo.py. These are process-wide performance settings, not changes to
# the LiveAct model architecture or sampling parameters.
# LiveAct serves only two fixed VAE decode shapes (6 latent frames for the
# first chunk and 11 latent frames for steady chunks).  cuDNN benchmarking
# spends roughly ten seconds searching Conv3d algorithms the first time a
# shape is seen on this deployment, while its steady-state gain is negligible.
# Prefer predictable first-request latency; startup warmup below still primes
# CUDA kernels and allocations for both shapes.
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class LiveActEngineConfig:
    ckpt_dir: str
    wav2vec_dir: str
    size: str = "416*720"
    hls_root: str = "liveact_avatar/hls_output"
    video_save_root: str = "liveact_avatar/generated_videos"
    t5_cpu: bool = True
    fp8_kv_cache: bool = False
    fp8_gemm: bool = True
    block_offload: bool = False
    compile_model: bool = False
    compile_vae_decode: bool = False
    seed: int = 42
    warmup: bool = False
    preload_image_path: Optional[str] = None
    preload_prompt: str = DEFAULT_AVATAR_PROMPT


@dataclass
class AvatarSessionConfig:
    session_id: str
    image_path: str
    prompt: str
    fps: int = 20
    input_sample_rate: int = 24000
    stream_video_only: bool = False
    # Bound how far a fast LiveAct worker may run ahead of wall clock.  The
    # current deployment is slightly slower than realtime, but keeping this
    # guard here prevents an optimized build from committing an unbounded
    # amount of silence before a later speech reply arrives.
    continuous_buffer_sec: float = 2.5
    edit_prompts: List[dict] = field(default_factory=list)


@dataclass
class AvatarChunkResult:
    iteration: int
    frames: int
    cost_sec: float
    audio_buffer_sec: float
    stream_ready: bool
    window_start_sec: float = 0.0
    window_end_sec: float = 0.0
    window_real_sec: float = 0.0
    window_padding_sec: float = 0.0
    model_backend: str = "eager"
    vae_backend: str = "eager"


@dataclass(frozen=True)
class PreparedAvatarAudioWindow:
    segment_id: int
    iteration: int
    start_sample: int
    end_sample: int
    real_samples: int
    padded_samples: int
    center_offset_frames: int
    pcm_s16le: bytes


@dataclass(frozen=True)
class StaticAvatarConditions:
    cache_key: tuple
    cond_image: torch.Tensor
    clip_context: torch.Tensor
    ref_target_masks: torch.Tensor
    y: torch.Tensor
    context: list
    edit_contexts: Dict[tuple, list]


class StaleAvatarAudioWindowError(RuntimeError):
    pass


def decode_pcm_s16le(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    pcm_i16 = np.frombuffer(pcm_bytes, dtype="<i2")
    if pcm_i16.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (pcm_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def resample_audio_for_liveact(audio: torch.Tensor, sr: int, fps: int, device) -> torch.Tensor:
    """Match the tempo logic used by generate.py/demo.py before Wav2Vec2."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    rate = 25 / float(fps)
    audio = audio.to(torch.float32).cpu()
    y, sr_out = torchaudio.sox_effects.apply_effects_tensor(audio, int(sr), [["tempo", f"{rate}"]])
    resampler = T.Resample(sr_out, 16000).to(device)
    return resampler(y.to(device)) * 3.0


class FfmpegHlsWriter:
    def __init__(
        self,
        *,
        hls_dir: str,
        width: int,
        height: int,
        fps: int,
        sample_rate: int,
        mp4_path: Optional[str] = None,
    ):
        self.hls_dir = os.path.abspath(hls_dir)
        self.mp4_path = os.path.abspath(mp4_path) if mp4_path else None
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.sample_rate = int(sample_rate)
        self.ffmpeg_bin = os.environ.get("LIVEACT_FFMPEG_BIN", "ffmpeg")
        self.process: Optional[subprocess.Popen] = None
        self.audio_fifo_path = os.path.join(self.hls_dir, ".audio_s16le.pipe")
        self.audio_fd: Optional[int] = None
        self._output_video_frames = 0
        self._output_audio_samples = 0
        # FFmpeg may temporarily consume one live input while not draining the
        # other.  Writing both complete chunk payloads from the inference
        # thread can therefore deadlock once either small OS pipe fills.  Keep
        # one long-lived feeder per input so future audio can keep arriving
        # while the video feeder is back-pressured (and vice versa).
        self._video_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._queue_sentinel = object()
        self._video_feeder: Optional[threading.Thread] = None
        self._audio_feeder: Optional[threading.Thread] = None
        self._feeder_errors: List[str] = []
        self._feeder_error_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._accepting_payloads = False

    @property
    def m3u8_path(self) -> str:
        return os.path.join(self.hls_dir, M3U8_NAME)

    def start(self) -> None:
        if os.path.exists(self.hls_dir):
            shutil.rmtree(self.hls_dir)
        os.makedirs(self.hls_dir, exist_ok=True)
        # A repeated/fixed session id must never expose an MP4 left by an
        # earlier call while the new HLS timeline is still being generated.
        if self.mp4_path:
            for stale_path in (self.mp4_path, self.mp4_path + ".part.mp4"):
                try:
                    os.remove(stale_path)
                except FileNotFoundError:
                    pass
        try:
            os.remove(self.audio_fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(self.audio_fifo_path, 0o600)
        # RDWR avoids a startup handshake deadlock: the parent can open the
        # FIFO before FFmpeg opens its read side. close_fds keeps this descriptor
        # out of the child, so closing it still produces a clean audio EOF.
        self.audio_fd = os.open(self.audio_fifo_path, os.O_RDWR)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-loglevel",
            "warning",
            "-thread_queue_size",
            "1024",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-thread_queue_size",
            "1024",
            # Unlike demo.py's seekable/finite WAV input, this FIFO remains
            # open for the whole call.  Disable the normal multi-second input
            # analysis or FFmpeg waits for future PCM before it starts draining
            # the video pipe, which deadlocks the first generated chunk.
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            self.audio_fifo_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            str(self.fps),
            "-keyint_min",
            str(self.fps),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1:first_pts=0",
            "-shortest",
            "-f",
            "hls",
            "-hls_time",
            "1",
            "-hls_list_size",
            "0",
            "-hls_segment_type",
            "mpegts",
            "-hls_flags",
            "append_list+independent_segments",
            "-hls_segment_filename",
            os.path.join(self.hls_dir, "live%06d.ts"),
            self.m3u8_path,
        ]
        print(f"[avatar][ffmpeg] {' '.join(command)}", flush=True)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
            )
        except Exception:
            os.close(self.audio_fd)
            self.audio_fd = None
            os.remove(self.audio_fifo_path)
            raise
        self._accepting_payloads = True
        self._video_feeder = threading.Thread(
            target=self._video_feeder_loop,
            name="avatar_ffmpeg_video_feeder",
            daemon=True,
        )
        self._audio_feeder = threading.Thread(
            target=self._audio_feeder_loop,
            name="avatar_ffmpeg_audio_feeder",
            daemon=True,
        )
        self._video_feeder.start()
        self._audio_feeder.start()

    @staticmethod
    def _video_tensor_to_bytes(video_tensor: torch.Tensor) -> tuple[bytes, int]:
        video_u8 = (
            ((video_tensor.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .contiguous()
            .cpu()
        )
        return video_u8.numpy().tobytes(), int(video_u8.shape[0])

    @staticmethod
    def _write_fd_all(fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise BrokenPipeError("FFmpeg audio FIFO stopped accepting PCM")
            view = view[written:]

    @staticmethod
    def _write_stream_all(stream, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = stream.write(view)
            if written is None:
                written = 0
            if written <= 0:
                raise BrokenPipeError("FFmpeg video pipe stopped accepting RGB frames")
            view = view[written:]
        stream.flush()

    def _record_feeder_error(self, source: str, exc: BaseException) -> None:
        message = f"FFmpeg {source} feeder failed: {exc}"
        with self._feeder_error_lock:
            self._feeder_errors.append(message)
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def _feeder_error(self) -> Optional[str]:
        with self._feeder_error_lock:
            return self._feeder_errors[0] if self._feeder_errors else None

    def _video_feeder_loop(self) -> None:
        process = self.process
        stream = process.stdin if process is not None else None
        try:
            if stream is None:
                raise RuntimeError("FFmpeg video stdin is unavailable")
            while True:
                payload = self._video_queue.get()
                if payload is self._queue_sentinel:
                    break
                self._write_stream_all(stream, payload)
        except BaseException as exc:
            self._record_feeder_error("video", exc)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, OSError):
                    pass

    def _audio_feeder_loop(self) -> None:
        fd = self.audio_fd
        try:
            if fd is None:
                raise RuntimeError("FFmpeg audio FIFO is unavailable")
            while True:
                payload = self._audio_queue.get()
                if payload is self._queue_sentinel:
                    break
                self._write_fd_all(fd, payload)
        except BaseException as exc:
            self._record_feeder_error("audio", exc)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                if self.audio_fd == fd:
                    self.audio_fd = None

    def _finish_feeders(self, process: subprocess.Popen, timeout: float = 120.0) -> Optional[str]:
        """Drain queued A/V, send EOF on both inputs, and join feeder threads."""

        with self._write_lock:
            if self._accepting_payloads:
                self._accepting_payloads = False
                # Both queues are FIFO: sentinels run only after every accepted
                # payload, then each feeder closes its own input to deliver EOF.
                self._audio_queue.put(self._queue_sentinel)
                self._video_queue.put(self._queue_sentinel)

        deadline = time.monotonic() + max(0.0, float(timeout))
        feeders = [thread for thread in (self._audio_feeder, self._video_feeder) if thread]
        for thread in feeders:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        alive = [thread.name for thread in feeders if thread.is_alive()]
        if alive:
            try:
                process.kill()
            except OSError:
                pass
            for thread in feeders:
                if thread.is_alive():
                    thread.join(timeout=5.0)
            message = f"FFmpeg input feeders did not drain within {timeout:.1f}s: {alive}"
            with self._feeder_error_lock:
                self._feeder_errors.append(message)

        # Defensive cleanup for a feeder that exited before taking ownership
        # of its descriptor. Normally both handles were closed by the threads.
        if self.audio_fd is not None:
            try:
                os.close(self.audio_fd)
            except OSError:
                pass
            self.audio_fd = None
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        return self._feeder_error()

    def write_av_tensor(self, video_tensor: torch.Tensor, pcm_s16le: bytes) -> int:
        """Queue model frames and their exactly corresponding mono PCM slice.

        Queueing is intentionally non-blocking with respect to FFmpeg pipe
        consumption. The two persistent feeders preserve per-stream ordering;
        ``close()`` drains both queues before finalizing HLS/MP4.
        """
        if self.process is None or self.process.stdin is None or self.audio_fd is None:
            raise RuntimeError("HLS writer is not started")
        if self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg exited early with code {self.process.returncode}")
        video_bytes, frame_count = self._video_tensor_to_bytes(video_tensor)
        target_video_frames = self._output_video_frames + int(frame_count)
        target_audio_samples = round(
            target_video_frames * self.sample_rate / float(self.fps)
        )
        required_audio_bytes = max(
            0, (target_audio_samples - self._output_audio_samples) * 2
        )
        pcm = np.frombuffer(pcm_s16le, dtype="<i2")
        required_audio_samples = required_audio_bytes // 2
        if pcm.size < required_audio_samples:
            pcm = np.pad(pcm, (0, required_audio_samples - pcm.size), mode="constant")
        elif pcm.size > required_audio_samples:
            pcm = pcm[:required_audio_samples]
        audio_bytes = np.ascontiguousarray(pcm, dtype="<i2").tobytes()
        with self._write_lock:
            if not self._accepting_payloads:
                raise RuntimeError("HLS writer is closing")
            feeder_error = self._feeder_error()
            if feeder_error:
                raise RuntimeError(feeder_error)
            if self.process.poll() is not None:
                raise RuntimeError(f"FFmpeg exited early with code {self.process.returncode}")
            # Enqueue audio first. If FFmpeg is currently waiting for future
            # audio timestamps, this lets its demuxer advance and releases any
            # video-pipe backpressure without blocking the inference thread.
            self._audio_queue.put(audio_bytes)
            self._video_queue.put(video_bytes)
            self._output_audio_samples = target_audio_samples
            self._output_video_frames = target_video_frames
        return frame_count

    def close(self) -> dict:
        result = {
            "video_path": self.mp4_path,
            "video_ready": False,
            "video_size_bytes": 0,
            "video_error": None,
        }
        if self.process is None:
            if self.mp4_path and os.path.isfile(self.mp4_path):
                result["video_ready"] = True
                result["video_size_bytes"] = int(os.path.getsize(self.mp4_path))
            return result
        process = self.process
        try:
            # FFmpeg can remain blocked probing two live pipes when a user
            # hangs up before the first model frame. There is no media to
            # finalize in that case, so terminate the empty encoder first and
            # close both feeder inputs with a short bounded join.
            empty_stream = self._output_video_frames <= 0
            if empty_stream and process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            feeder_error = self._finish_feeders(
                process,
                timeout=5.0 if empty_stream else 120.0,
            )
            if feeder_error:
                result["video_error"] = feeder_error
            self.process = None
            try:
                return_code = process.wait(timeout=10.0 if empty_stream else 120.0)
                if (
                    return_code != 0
                    and not empty_stream
                    and result["video_error"] is None
                ):
                    result["video_error"] = f"FFmpeg HLS encoder exited with code {return_code}"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
                result["video_error"] = "FFmpeg HLS encoder did not stop within 120 seconds"
        finally:
            self.process = None
            if self.audio_fd is not None:
                os.close(self.audio_fd)
                self.audio_fd = None
            try:
                os.remove(self.audio_fifo_path)
            except FileNotFoundError:
                pass

        if self.mp4_path and os.path.exists(self.m3u8_path):
            os.makedirs(os.path.dirname(self.mp4_path), exist_ok=True)
            temporary_path = self.mp4_path + ".part.mp4"
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            completed = subprocess.run(
                [
                    self.ffmpeg_bin, "-y", "-loglevel", "warning", "-i", self.m3u8_path,
                    "-c", "copy", "-movflags", "+faststart", temporary_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if (
                completed.returncode == 0
                and os.path.isfile(temporary_path)
                and os.path.getsize(temporary_path) > 0
            ):
                os.replace(temporary_path, self.mp4_path)
            else:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
                detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
                result["video_error"] = f"failed to save complete MP4: {detail}"

        if self.mp4_path and os.path.isfile(self.mp4_path):
            result["video_ready"] = True
            result["video_size_bytes"] = int(os.path.getsize(self.mp4_path))
        elif result["video_error"] is None:
            result["video_error"] = "no generated HLS frames were available to save"
        return result


class LiveActAvatarSession:
    def __init__(self, engine: "LiveActStreamingEngine", config: AvatarSessionConfig):
        session_started = time.perf_counter()
        self.session_started_at = time.monotonic()
        self.engine = engine
        self.config = config
        self.device = engine.device
        self.rank = engine.rank
        self.world_size = engine.world_size
        self.state_lock = threading.RLock()
        self.segment_id = 0
        self.iteration = 0
        self.pre_latent: Optional[torch.Tensor] = None
        self.input_sample_rate = int(config.input_sample_rate)
        self.audio_timeline = StreamingPcmTimeline(self.input_sample_rate)
        # Token2Wav writes real speech into this pending queue.  The LiveAct
        # window worker is the only writer to ``audio_timeline`` and commits
        # either pending real PCM or silence immediately before a model window
        # is frozen.  This keeps late speech from being placed behind a large,
        # eagerly-materialized silence buffer.
        self.audio_source_condition = threading.Condition(threading.RLock())
        self.pending_real_pcm = np.zeros(0, dtype=np.float32)
        self.speech_active = False
        self.continuous_started = False
        self.continuous_started_at: Optional[float] = None
        self.continuous_buffer_sec = max(
            0.5,
            float(config.continuous_buffer_sec),
        )
        # Samples below this cursor belong to already generated (or currently
        # generating) video frames.  Later speech may replace speculative idle
        # silence only after this boundary.
        self.protected_audio_until_sample = 0
        self.committed_real_samples = 0
        self.committed_silence_samples = 0
        self.interrupted_pending_samples = 0
        # Model input windows overlap, while mux output advances exactly by the
        # number of decoded frames.  This cursor spans the complete call: a
        # speech boundary or interruption changes only the future PCM source
        # and never resets media timestamps.
        self.segment_muxed_video_frames = 0
        self.total_muxed_video_frames = 0
        self.total_muxed_audio_samples = 0
        # LiveAct is single-active in the HTTP sidecar. Reuse the engine-owned
        # cache that is allocated once at service startup, just like demo.py,
        # instead of allocating tens of gigabytes of K/V tensors per call. The
        # first 6/8-latent iterations overwrite every cache range they read, so
        # a full multi-gigabyte zero-fill is neither needed nor done here.
        cache_reset_started = time.perf_counter()
        self.kv_cache = engine.reset_shared_kv_cache()
        self.stage_timings = {
            "kv_cache_reset_sec": time.perf_counter() - cache_reset_started,
        }
        self.context = None
        self.edit_contexts: Dict[tuple, list] = {}
        self.cond_image = None
        self.clip_context = None
        self.ref_target_masks = None
        self.y = None
        self.writer: Optional[FfmpegHlsWriter] = None
        self.received_pcm_samples = 0
        self.inference_active = False
        self.generated_video_frames = 0
        self.inference_cost_sec = 0.0
        self.first_chunk_frames = 0
        self.first_chunk_cost_sec = 0.0
        self.steady_video_frames = 0
        self.steady_inference_cost_sec = 0.0
        self.last_chunk_cost_sec = 0.0
        self.max_chunk_cost_sec = 0.0
        self.first_pcm_received_sec: Optional[float] = None
        self.first_window_ready_sec: Optional[float] = None
        self.first_inference_started_sec: Optional[float] = None
        self.first_chunk_completed_sec: Optional[float] = None
        self.first_stream_ready_sec: Optional[float] = None
        self.conditions_ready = False
        self.preparing_conditions = False
        self.preparation_error: Optional[str] = None
        self.conditions_event = threading.Event()
        self.cached_audio_window_key: Optional[tuple] = None
        self.cached_audio_embs: Optional[torch.Tensor] = None
        self.stage_timings["session_start_total_sec"] = (
            time.perf_counter() - session_started
        )

    @property
    def frame_num(self) -> int:
        return (sum(self.engine.blksz_lst) - 1) * self.engine.vae_stride[0] + 1

    @property
    def chunk_step_frames(self) -> int:
        return self.engine.blksz_lst[-1] * self.engine.vae_stride[0]

    @property
    def first_chunk_output_frames(self) -> int:
        return (
            (self.engine.blksz_lst[0] - 1) * self.engine.vae_stride[0] + 1
        )

    @property
    def stream_ready(self) -> bool:
        return bool(self.writer and os.path.exists(self.writer.m3u8_path))

    @property
    def hls_dir(self) -> str:
        return os.path.join(self.engine.config.hls_root, self.config.session_id)

    @property
    def final_video_path(self) -> str:
        return os.path.join(self.engine.config.video_save_root, f"{self.config.session_id}.mp4")

    def prepare_static_conditions(self) -> None:
        """Run Demo task preprocessing asynchronously from the HTTP request.

        ``demo.py`` returns from ``/start_stream`` after queueing the task. The
        streaming derivative mirrors that behavior: PCM may enter the source
        queue while CLIP/VAE/T5 conditions are prepared on both ranks. FFmpeg
        is deliberately deferred until the first inference-ready PCM window.
        """

        with self.state_lock:
            if self.conditions_ready:
                return
            if self.preparing_conditions:
                raise RuntimeError("LiveAct session conditions are already being prepared")
            self.preparing_conditions = True
            self.preparation_error = None
        prepare_started = time.perf_counter()
        try:
            self._prepare_static_conditions()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            with self.state_lock:
                self.conditions_ready = True
                self.stage_timings["condition_prepare_sec"] = (
                    time.perf_counter() - prepare_started
                )
                self.stage_timings["conditions_ready_sec"] = max(
                    0.0, time.monotonic() - self.session_started_at
                )
        except Exception as exc:
            with self.state_lock:
                self.preparation_error = str(exc)
            raise
        finally:
            with self.state_lock:
                self.preparing_conditions = False
                self.conditions_event.set()

    def _prepare_static_conditions(self) -> None:
        conditions, condition_timings, cache_hit = self.engine.get_static_conditions(
            image_path=self.config.image_path,
            prompt=self.config.prompt,
            edit_prompts=self.config.edit_prompts,
        )
        self.cond_image = conditions.cond_image
        self.clip_context = conditions.clip_context
        self.ref_target_masks = conditions.ref_target_masks
        self.y = conditions.y
        self.context = conditions.context
        self.edit_contexts = dict(conditions.edit_contexts)
        self.stage_timings.update(condition_timings)
        self.stage_timings["static_condition_cache_hit"] = bool(cache_hit)

        torch.manual_seed(int(self.engine.config.seed))

    def _ensure_writer_started(self) -> None:
        """Create the call media pipeline only when its first PCM is usable."""

        if self.rank != 0 or self.writer is not None:
            return
        stage_started = time.perf_counter()
        writer = FfmpegHlsWriter(
            hls_dir=self.hls_dir,
            width=self.engine.width,
            height=self.engine.height,
            fps=self.config.fps,
            sample_rate=self.input_sample_rate,
            mp4_path=self.final_video_path,
        )
        writer.start()
        self.writer = writer
        with self.state_lock:
            self.stage_timings["hls_writer_start_sec"] = (
                time.perf_counter() - stage_started
            )

    def append_pcm(self, pcm_bytes: bytes, sample_rate: int, is_last: bool = False) -> List[AvatarChunkResult]:
        pcm = decode_pcm_s16le(pcm_bytes)
        sample_rate = int(sample_rate or self.input_sample_rate)
        if pcm.size:
            if sample_rate != self.input_sample_rate:
                wav = torch.from_numpy(pcm).unsqueeze(0)
                wav = T.Resample(sample_rate, self.input_sample_rate)(wav).squeeze(0).numpy()
                pcm = wav.astype(np.float32)
            pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        appended = int(pcm.size)
        if appended or is_last:
            with self.audio_source_condition:
                if appended:
                    if self.pending_real_pcm.size:
                        self.pending_real_pcm = np.concatenate(
                            [self.pending_real_pcm, pcm]
                        )
                    else:
                        self.pending_real_pcm = np.ascontiguousarray(
                            pcm.copy(), dtype=np.float32
                        )
                    self.speech_active = True
                    if not self.continuous_started:
                        self.continuous_started = True
                        self.continuous_started_at = time.monotonic()
                    self._overlay_pending_real_on_silence_locked()
                if is_last:
                    # Natural <tts_end>: drain all pending real PCM, then let
                    # the window scheduler supply silence forever.  This is a
                    # speech boundary, not the end of the LiveAct session.
                    self.speech_active = False
                self.audio_source_condition.notify_all()
        if appended:
            with self.state_lock:
                self.received_pcm_samples += appended
                if self.first_pcm_received_sec is None:
                    self.first_pcm_received_sec = max(
                        0.0, time.monotonic() - self.session_started_at
                    )
        # GPU inference is driven by a separate audio-window worker. This method
        # only decodes/resamples and queues real PCM, so it never waits for the
        # distributed inference lock.
        return []

    def abort(self) -> None:
        """Interrupt only the current speech source, not the Avatar session.

        Full-duplex S->L transitions must stop queued reply audio quickly, but
        resetting iteration/KV/pre_latent would reintroduce the first-window
        delay and tear the continuous animation.  Already committed model
        windows are causal and cannot be rewritten; every uncommitted sample
        is dropped and future windows use silence until new Token2Wav PCM
        arrives.
        """

        with self.audio_source_condition:
            dropped = int(self.pending_real_pcm.size)
            if dropped:
                self.pending_real_pcm = np.zeros(0, dtype=np.float32)
            timeline_stats = self.audio_timeline.stats()
            cancel_from = max(
                int(timeline_stats.muxed_samples),
                int(self.protected_audio_until_sample),
            )
            cancelled_committed = self.audio_timeline.replace_future_with_silence(
                cancel_from
            )
            self.interrupted_pending_samples += dropped + cancelled_committed
            if cancelled_committed:
                with self.state_lock:
                    self.committed_real_samples = max(
                        0, self.committed_real_samples - cancelled_committed
                    )
                    self.committed_silence_samples += cancelled_committed
            self.speech_active = False
            self.audio_source_condition.notify_all()

    def close(self, *, discard_pending: bool = True) -> dict:
        save_result = {
            "video_path": self.final_video_path,
            "video_ready": False,
            "video_size_bytes": 0,
            "video_error": None,
        }
        if self.writer is not None:
            save_result = self.writer.close()
            self.writer = None
        if discard_pending:
            # A call hangup is a cancellation boundary, not an end-of-file
            # request.  Drop all session-owned audio/model state immediately;
            # the heavyweight engine weights remain resident on the GPU.
            with self.state_lock:
                self.segment_id += 1
                self.audio_timeline.reset()
                self.audio_timeline.finish()
                self.pre_latent = None
                self.kv_cache = None
                self.cached_audio_window_key = None
                self.cached_audio_embs = None
                self.context = None
                self.edit_contexts.clear()
                self.cond_image = None
                self.clip_context = None
                self.ref_target_masks = None
                self.y = None
                self.inference_active = False
            with self.audio_source_condition:
                self.pending_real_pcm = np.zeros(0, dtype=np.float32)
                self.speech_active = False
                self.continuous_started = False
                self.audio_source_condition.notify_all()
        return save_result

    def generate_ready_chunks(self, max_chunks: int = 8) -> List[AvatarChunkResult]:
        results: List[AvatarChunkResult] = []
        while len(results) < int(max_chunks):
            prepared = self.prepare_next_audio_window(timeout_sec=0.0)
            if prepared is None:
                break
            results.append(self.generate_prepared_audio_window(prepared))
        return results

    def _audio_start_end_frames(self, iteration: Optional[int] = None) -> tuple[int, int]:
        current_iteration = self.iteration if iteration is None else int(iteration)
        if current_iteration <= 1:
            audio_start_idx = 0
        else:
            audio_start_idx = (current_iteration - 1) * self.chunk_step_frames
        return audio_start_idx, audio_start_idx + self.frame_num

    def _within_continuous_generation_budget(self) -> bool:
        """Keep optimized workers from generating silence without bound."""

        with self.audio_source_condition:
            started_at = self.continuous_started_at
        if started_at is None:
            return False
        elapsed_sec = max(0.0, time.monotonic() - float(started_at))
        with self.state_lock:
            generated_sec = (
                self.total_muxed_video_frames / float(max(1, self.config.fps))
            )
        return generated_sec <= elapsed_sec + self.continuous_buffer_sec

    def _materialize_audio_through(
        self,
        end_sample: int,
        *,
        deadline: float,
    ) -> bool:
        """Commit real PCM first and deadline silence second.

        The append-only model timeline is written only here. During a speaking
        response we wait for Token2Wav instead of speculatively occupying its
        future slots. Once the response naturally ends or is interrupted, the
        exact missing suffix needed by the next attention window is committed
        as zero PCM so LiveAct never stops advancing.
        """

        target = max(1, int(end_sample))
        while True:
            with self.audio_source_condition:
                if not self.continuous_started:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self.audio_source_condition.wait(timeout=min(remaining, 0.25))
                    continue

                self._overlay_pending_real_on_silence_locked()
                stats = self.audio_timeline.stats()
                missing = target - int(stats.received_samples)
                if missing <= 0:
                    return True

                if self.pending_real_pcm.size:
                    take = min(missing, int(self.pending_real_pcm.size))
                    real_chunk = np.ascontiguousarray(
                        self.pending_real_pcm[:take].copy(), dtype=np.float32
                    )
                    self.pending_real_pcm = np.ascontiguousarray(
                        self.pending_real_pcm[take:].copy(), dtype=np.float32
                    )
                    # Keep source selection and timeline commit atomic with
                    # respect to abort/new PCM. Otherwise an interrupt could
                    # clear the queue and a worker that already copied it could
                    # append the cancelled speech afterwards.
                    self.audio_timeline.append(
                        real_chunk,
                        is_last=False,
                        source_real=True,
                    )
                    with self.state_lock:
                        self.committed_real_samples += int(real_chunk.size)
                    continue
                elif self.speech_active:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self.audio_source_condition.wait(timeout=min(remaining, 0.25))
                    continue
                else:
                    # Silence is materialized only for the exact window being
                    # scheduled. A later reply therefore waits behind at most
                    # the model's intrinsic lookahead, not an unbounded idle
                    # buffer.
                    self.audio_timeline.append(
                        np.zeros(missing, dtype=np.float32),
                        is_last=False,
                        source_real=False,
                    )
                    with self.state_lock:
                        self.committed_silence_samples += int(missing)
                    continue

    def _overlay_pending_real_on_silence_locked(self) -> int:
        """Move queued Token2Wav PCM into the earliest unfrozen idle slots.

        Caller must own ``audio_source_condition``.  This is what prevents a
        new response from being appended behind attention-lookahead silence.
        """

        replaced_total = 0
        while self.pending_real_pcm.size:
            timeline_stats = self.audio_timeline.stats()
            mutable_from = max(
                int(timeline_stats.muxed_samples),
                int(self.protected_audio_until_sample),
            )
            replaced, _replace_start = self.audio_timeline.replace_earliest_silence(
                mutable_from,
                self.pending_real_pcm,
            )
            if replaced <= 0:
                break
            self.pending_real_pcm = np.ascontiguousarray(
                self.pending_real_pcm[replaced:].copy(), dtype=np.float32
            )
            replaced_total += int(replaced)
        if replaced_total:
            with self.state_lock:
                self.committed_real_samples += replaced_total
                self.committed_silence_samples = max(
                    0, self.committed_silence_samples - replaced_total
                )
        return replaced_total

    def prepare_next_audio_window(
        self,
        *,
        timeout_sec: float = 0.0,
    ) -> Optional[PreparedAvatarAudioWindow]:
        """Wait for and snapshot the next continuous PCM model window."""

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self.state_lock:
            conditions_ready = bool(self.conditions_ready)
            preparation_error = self.preparation_error
        if preparation_error:
            raise RuntimeError(f"LiveAct condition preparation failed: {preparation_error}")
        if not conditions_ready:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.conditions_event.wait(remaining):
                return None
            with self.state_lock:
                if self.preparation_error:
                    raise RuntimeError(
                        f"LiveAct condition preparation failed: {self.preparation_error}"
                    )
                if not self.conditions_ready:
                    return None
        while True:
            with self.audio_source_condition:
                continuous_started = bool(self.continuous_started)
            if not continuous_started:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                with self.audio_source_condition:
                    self.audio_source_condition.wait(timeout=min(remaining, 0.25))
                continue

            if not self._within_continuous_generation_budget():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                with self.audio_source_condition:
                    self.audio_source_condition.wait(timeout=min(remaining, 0.05))
                continue

            with self.state_lock:
                segment_id = int(self.segment_id)
                iteration = int(self.iteration)

            audio_start_idx, audio_end_idx = self._audio_start_end_frames(iteration)
            # get_audio_emb() consumes a five-frame neighborhood [-2, +2]
            # around every center frame. The full-audio Demo gets both sides
            # naturally; the streaming version must explicitly retain the two
            # preceding frames and wait for two lookahead frames.
            context_start_idx = max(0, audio_start_idx - 2)
            context_end_idx = audio_end_idx + 2
            start_sample = round(
                self.input_sample_rate * (context_start_idx / float(self.config.fps))
            )
            end_sample = round(
                self.input_sample_rate * (context_end_idx / float(self.config.fps))
            )
            if not self._materialize_audio_through(end_sample, deadline=deadline):
                return None
            # Snapshot and freeze the media interval atomically with respect to
            # late PCM overlay. A reply arriving one instruction later must be
            # placed after this chunk, not muxed under frames inferred from the
            # already-captured silence snapshot.
            with self.audio_source_condition:
                window = self.audio_timeline.snapshot_window(
                    start_sample,
                    end_sample,
                    allow_finished_padding=False,
                )
                if window is not None:
                    expected_output_frames = (
                        self.first_chunk_output_frames
                        if iteration == 0
                        else self.chunk_step_frames
                    )
                    self.protected_audio_until_sample = max(
                        int(self.protected_audio_until_sample),
                        round(
                            (
                                self.segment_muxed_video_frames
                                + expected_output_frames
                            )
                            * self.input_sample_rate
                            / float(max(1, self.config.fps))
                        ),
                    )
            if window is not None:
                # Before the first reply the browser displays the configured
                # idle image/video, so opening two live FFmpeg inputs at call
                # creation has no benefit. Deferring this ~10 ms operation also
                # makes hangup-before-first-reply immediate and leak-free.
                self._ensure_writer_started()
                with self.state_lock:
                    if self.first_window_ready_sec is None:
                        self.first_window_ready_sec = max(
                            0.0, time.monotonic() - self.session_started_at
                        )
                pcm_s16le = (
                    np.clip(window.samples, -1.0, 1.0) * 32767.0
                ).round().astype("<i2", copy=False).tobytes()
                return PreparedAvatarAudioWindow(
                    segment_id=segment_id,
                    iteration=iteration,
                    start_sample=int(window.start_sample),
                    end_sample=int(window.end_sample),
                    real_samples=int(window.real_samples),
                    padded_samples=int(window.padded_samples),
                    center_offset_frames=int(audio_start_idx - context_start_idx),
                    pcm_s16le=pcm_s16le,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            stats = self.audio_timeline.stats()
            self.audio_timeline.wait_for_update(stats.version, timeout=min(remaining, 0.25))

    def _select_context(self) -> list:
        for (start, end), context in self.edit_contexts.items():
            if start <= self.iteration <= end:
                return context
        return self.context

    def _audio_embs_from_window(
        self,
        prepared: PreparedAvatarAudioWindow,
        audio_window: np.ndarray,
    ) -> torch.Tensor:
        cache_key = (
            int(prepared.segment_id),
            int(prepared.start_sample),
            int(prepared.end_sample),
            int(prepared.real_samples),
            int(prepared.padded_samples),
            int(prepared.center_offset_frames),
        )
        if self.cached_audio_window_key == cache_key and self.cached_audio_embs is not None:
            return self.cached_audio_embs
        audio_slice = torch.from_numpy(
            np.ascontiguousarray(audio_window, dtype=np.float32)
        ).unsqueeze(0)
        audio_resampled = resample_audio_for_liveact(
            audio_slice,
            self.input_sample_rate,
            self.config.fps,
            self.device,
        )
        audio_embedding = get_embedding(
            audio_resampled[0],
            self.engine.wav2vec_feature_extractor,
            self.engine.audio_encoder,
            device=self.device,
        )
        center_start = int(prepared.center_offset_frames)
        audio_embs = get_audio_emb(
            audio_embedding,
            center_start,
            center_start + self.frame_num,
            self.device,
        )
        self.cached_audio_window_key = cache_key
        self.cached_audio_embs = audio_embs
        return audio_embs

    def _prepare_mux_audio_for_frames(
        self,
        frame_count: int,
    ) -> tuple[bytes, int, int, int]:
        """Return the next contiguous PCM slice matching ``frame_count / fps``.

        This cursor is intentionally independent from the overlapping audio
        windows used for Wav2Vec/cross attention.  Cumulative rounding keeps
        long-running audio and video on the same rational media timeline.
        """
        frame_count = max(0, int(frame_count))
        target_video_frames = self.segment_muxed_video_frames + frame_count
        target_audio_samples = round(
            target_video_frames * self.input_sample_rate / float(self.config.fps)
        )
        timeline_stats = self.audio_timeline.stats()
        required_samples = max(0, target_audio_samples - timeline_stats.muxed_samples)
        window = self.audio_timeline.read_for_mux(
            required_samples,
            allow_finished_padding=False,
        )
        if window is None:
            raise RuntimeError(
                "LiveAct tried to mux audio before real PCM was ready: "
                f"required={required_samples}, received={timeline_stats.received_samples}, "
                f"muxed={timeline_stats.muxed_samples}"
            )
        pcm_s16le = (
            np.clip(window.samples, -1.0, 1.0) * 32767.0
        ).round().astype("<i2", copy=False).tobytes()
        return pcm_s16le, target_audio_samples, target_video_frames, required_samples

    def generate_prepared_audio_window(
        self,
        prepared: PreparedAvatarAudioWindow,
    ) -> AvatarChunkResult:
        with self.state_lock:
            if (
                int(prepared.segment_id) != int(self.segment_id)
                or int(prepared.iteration) != int(self.iteration)
            ):
                raise StaleAvatarAudioWindowError(
                    "stale LiveAct PCM window: "
                    f"prepared=segment{prepared.segment_id}/iter{prepared.iteration}, "
                    f"current=segment{self.segment_id}/iter{self.iteration}"
                )
            self.inference_active = True
            if self.first_inference_started_sec is None:
                self.first_inference_started_sec = max(
                    0.0, time.monotonic() - self.session_started_at
                )
        expected_samples = int(prepared.end_sample - prepared.start_sample)
        audio_window = decode_pcm_s16le(prepared.pcm_s16le)
        if audio_window.size != expected_samples:
            with self.state_lock:
                self.inference_active = False
            raise RuntimeError(
                f"invalid prepared PCM window: got={audio_window.size}, expected={expected_samples}"
            )
        try:
            return self._generate_one_chunk(prepared, audio_window)
        finally:
            with self.state_lock:
                self.inference_active = False

    def _generate_one_chunk(
        self,
        prepared: PreparedAvatarAudioWindow,
        audio_window: np.ndarray,
    ) -> AvatarChunkResult:
        start_time = time.perf_counter()
        iteration = int(self.iteration)
        f_idx = 0 if self.iteration == 0 else 1
        model_backend = self.engine.execution_backend_for_iteration(iteration, component="model")
        vae_backend = self.engine.execution_backend_for_iteration(iteration, component="vae")
        wan_i2v_model = self.engine.wan_model_for_iteration(iteration)
        vae_decode = self.engine.vae_decode_for_iteration(iteration)
        audio_embs = self._audio_embs_from_window(prepared, audio_window)
        y_cut = self.y[:, :, : self.frame_num // 4 + 1, ...]
        cached_context = self._select_context()
        latent = torch.randn(
            16,
            self.engine.blksz_lst[f_idx],
            self.engine.height // self.engine.vae_stride[1],
            self.engine.width // self.engine.vae_stride[2],
            dtype=torch.bfloat16,
            device=self.device,
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.engine.timesteps) - 1):
                timestep = self.engine.timesteps[i]
                arg_c = {
                    "context": cached_context,
                    "clip_fea": self.clip_context,
                    "ref_target_masks": self.ref_target_masks,
                    "audio": audio_embs,
                    "y": y_cut[
                        :,
                        :,
                        sum(self.engine.blksz_lst[:f_idx]) : sum(self.engine.blksz_lst[: f_idx + 1]),
                    ],
                    "start_idx": sum(self.engine.blksz_lst[:f_idx]) * self.engine.frame_len,
                    "end_idx": sum(self.engine.blksz_lst[: f_idx + 1]) * self.engine.frame_len,
                    "update_cache": self.iteration > 1,
                }
                noise_pred = wan_i2v_model(
                    [latent],
                    t=timestep,
                    kv_cache=self.kv_cache[i],
                    skip_audio=False if i in [1, 2] else True,
                    **arg_c,
                )[0]
                dt = (self.engine.timesteps[i] - self.engine.timesteps[i + 1]) / 1000
                latent = latent + (-noise_pred) * dt[0]

            if self.iteration == 0:
                videos = vae_decode(latent)
            else:
                combined_latent = torch.concat([self.pre_latent[:, -3:], latent], dim=1)
                videos = vae_decode(combined_latent)[:, :, 9:]
            self.pre_latent = latent

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        frame_count = 0
        if self.rank == 0 and self.writer is not None:
            decoded_frame_count = int(videos.shape[2])
            (
                pcm_s16le,
                target_audio_samples,
                target_video_frames,
                required_audio_samples,
            ) = self._prepare_mux_audio_for_frames(decoded_frame_count)
            frame_count = self.writer.write_av_tensor(videos, pcm_s16le)
            # Commit timeline cursors only after FFmpeg accepted both complete
            # payloads. A broken pipe must not make unplayed PCM look consumed.
            self.audio_timeline.advance_muxed(target_audio_samples)
            self.segment_muxed_video_frames = target_video_frames
            self.total_muxed_video_frames += int(frame_count)
            self.total_muxed_audio_samples += int(required_audio_samples)

        result = AvatarChunkResult(
            iteration=int(self.iteration),
            frames=int(frame_count),
            cost_sec=float(time.perf_counter() - start_time),
            audio_buffer_sec=float(self.audio_timeline.stats().retained_sec),
            stream_ready=self.stream_ready,
            window_start_sec=float(prepared.start_sample / max(1, self.input_sample_rate)),
            window_end_sec=float(prepared.end_sample / max(1, self.input_sample_rate)),
            window_real_sec=float(prepared.real_samples / max(1, self.input_sample_rate)),
            window_padding_sec=float(prepared.padded_samples / max(1, self.input_sample_rate)),
            model_backend=model_backend,
            vae_backend=vae_backend,
        )
        with self.state_lock:
            self.generated_video_frames += int(result.frames)
            self.inference_cost_sec += float(result.cost_sec)
            self.last_chunk_cost_sec = float(result.cost_sec)
            self.max_chunk_cost_sec = max(self.max_chunk_cost_sec, float(result.cost_sec))
            if self.iteration == 0:
                self.first_chunk_frames = int(result.frames)
                self.first_chunk_cost_sec = float(result.cost_sec)
                self.first_chunk_completed_sec = max(
                    0.0, time.monotonic() - self.session_started_at
                )
                if result.stream_ready:
                    self.first_stream_ready_sec = self.first_chunk_completed_sec
            else:
                self.steady_video_frames += int(result.frames)
                self.steady_inference_cost_sec += float(result.cost_sec)
            self.iteration += 1
            next_audio_start_idx, _ = self._audio_start_end_frames(self.iteration)
            next_context_start_idx = max(0, next_audio_start_idx - 2)
        timeline_stats = self.audio_timeline.stats()
        next_start_sample = round(
            self.input_sample_rate * (next_context_start_idx / float(self.config.fps))
        )
        self.audio_timeline.discard_before(
            min(next_start_sample, timeline_stats.muxed_samples)
        )
        return result


class LiveActStreamingEngine:
    def __init__(self, config: LiveActEngineConfig):
        self.config = config
        self.config.hls_root = os.path.abspath(self.config.hls_root)
        self.config.video_save_root = os.path.abspath(self.config.video_save_root)
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank
        self.width, self.height = [int(x) for x in config.size.split("*")]
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.blksz_lst = [6, 8]
        self.timesteps = [
            torch.tensor([_], device=self.device, dtype=torch.float32)
            for _ in [1000.0, 937.5, 833.33333333, 0.0]
        ]
        self.frame_len = (self.height // (self.patch_size[1] * self.vae_stride[1])) * (
            self.width // (self.patch_size[2] * self.vae_stride[2])
        )
        self.sessions: Dict[str, LiveActAvatarSession] = {}
        self._static_condition_cache: Dict[tuple, StaticAvatarConditions] = {}
        self._static_condition_lock = threading.RLock()
        self.preloaded_static_conditions = False
        self.preload_condition_timings: Dict[str, float] = {}
        os.makedirs(self.config.hls_root, exist_ok=True)
        os.makedirs(self.config.video_save_root, exist_ok=True)
        self._setup_distributed()
        self._load_models()
        cache_started = time.perf_counter()
        self.shared_kv_cache = self.new_kv_cache()
        if self.rank == 0:
            print(
                "[avatar] shared KV cache allocated once at startup "
                f"in {time.perf_counter() - cache_started:.3f}s",
                flush=True,
            )
        if self.config.preload_image_path:
            preload_started = time.perf_counter()
            _, preload_timings, _ = self.get_static_conditions(
                image_path=self.config.preload_image_path,
                prompt=self.config.preload_prompt,
                edit_prompts=[],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            self.preloaded_static_conditions = True
            self.preload_condition_timings = dict(preload_timings)
            self.preload_condition_timings["preload_total_sec"] = (
                time.perf_counter() - preload_started
            )
            if self.rank == 0:
                print(
                    "[avatar] fixed conditions preloaded "
                    f"image={os.path.abspath(self.config.preload_image_path)} "
                    f"prompt={self.config.preload_prompt!r} "
                    f"cost={self.preload_condition_timings['preload_total_sec']:.3f}s",
                    flush=True,
                )
        if self.config.warmup:
            self.warmup()

    def _setup_distributed(self) -> None:
        if self.world_size <= 1:
            return
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
            )
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=self.rank, world_size=self.world_size)
        initialize_model_parallel(
            sequence_parallel_degree=self.world_size,
            ring_degree=1,
            ulysses_degree=self.world_size,
        )

    def _load_models(self) -> None:
        if self.world_size > 1:
            from liveact_runtime.model_liveact.model_memory_sp import WanModel
        else:
            from liveact_runtime.model_liveact.model_memory import WanModel

        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (self.height, self.width))
                ),
                transforms.ToTensor(),
                transforms.Resize((self.height, self.width)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.wan_i2v_model = WanModel.from_pretrained(
            self.config.ckpt_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        ).to(dtype=torch.bfloat16)
        if self.config.fp8_gemm:
            enable_fp8_gemm(self.wan_i2v_model, options=FP8GemmOptions())
        if self.config.block_offload:
            for name, child in self.wan_i2v_model.named_children():
                if name != "blocks":
                    child.to(self.device)
            self.wan_i2v_model.enable_block_offload(onload_device=torch.device(f"cuda:{self.device}"))
        else:
            self.wan_i2v_model = self.wan_i2v_model.to(self.device)
        self.wan_i2v_model.freqs = self.wan_i2v_model.freqs.to(self.device)
        for n in range(40):
            self.wan_i2v_model.blocks[n].self_attn.init_kvidx(self.frame_len, self.world_size)
        self.wan_i2v_model.eval()
        self.model_execution_backend = "eager"
        if self.config.compile_model:
            self.wan_i2v_model = torch.compile(
                self.wan_i2v_model,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                dynamic=False,
            )
            self.model_execution_backend = "compiled"

        self.vae = LightVAE(
            vae_path=os.path.join(self.config.ckpt_dir, "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16,
            device=self.device,
            use_lightvae=False,
            parallel=(self.world_size > 1),
        )
        self.vae.model.eval()
        self.vae_decode = self.vae.decode
        self.vae_execution_backend = "eager"
        if self.config.compile_vae_decode:
            self.vae_decode = torch.compile(self.vae_decode)
            self.vae_execution_backend = "compiled"

        self.clip = CLIPModel(
            checkpoint_path=os.path.join(
                self.config.ckpt_dir,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            tokenizer_path=os.path.join(self.config.ckpt_dir, "xlm-roberta-large"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device="cpu" if self.config.t5_cpu else self.device,
            checkpoint_path=os.path.join(self.config.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.config.ckpt_dir, "google/umt5-xxl"),
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            self.config.wav2vec_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.wav2vec_dir,
            local_files_only=True,
        )
        self.audio_encoder.feature_extractor._freeze_parameters()
        for module in [self.wan_i2v_model, self.clip.model, self.audio_encoder, self.vae.model]:
            for param in module.parameters():
                param.requires_grad = False
        torch.cuda.empty_cache()

    def wan_model_for_iteration(self, iteration: int):
        del iteration
        return self.wan_i2v_model

    def vae_decode_for_iteration(self, iteration: int):
        del iteration
        return self.vae_decode

    def execution_backend_for_iteration(self, iteration: int, *, component: str) -> str:
        del iteration
        if component == "model":
            return self.model_execution_backend
        elif component == "vae":
            return self.vae_execution_backend
        raise ValueError(f"unknown LiveAct execution component: {component}")

    @property
    def condition_frame_num(self) -> int:
        return (sum(self.blksz_lst) - 1) * self.vae_stride[0] + 1

    @staticmethod
    def _normalize_edit_prompts(edit_prompts: List[dict]) -> tuple:
        normalized = []
        for item in edit_prompts or []:
            try:
                normalized.append(
                    (
                        int(item["start_chunk"]),
                        int(item["end_chunk"]),
                        str(item["prompt"]),
                    )
                )
            except Exception:
                continue
        return tuple(normalized)

    def _static_condition_key(
        self,
        *,
        image_path: str,
        prompt: str,
        edit_prompts: List[dict],
    ) -> tuple:
        resolved_image_path = os.path.abspath(os.path.realpath(str(image_path)))
        stat = os.stat(resolved_image_path)
        return (
            resolved_image_path,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            str(prompt),
            self._normalize_edit_prompts(edit_prompts),
        )

    def get_static_conditions(
        self,
        *,
        image_path: str,
        prompt: str,
        edit_prompts: List[dict],
    ) -> tuple[StaticAvatarConditions, Dict[str, float], bool]:
        """Return immutable image/text conditions shared by single-active calls."""

        lookup_started = time.perf_counter()
        cache_key = self._static_condition_key(
            image_path=image_path,
            prompt=prompt,
            edit_prompts=edit_prompts,
        )
        with self._static_condition_lock:
            cached = self._static_condition_cache.get(cache_key)
            if cached is not None:
                return (
                    cached,
                    {
                        "static_condition_lookup_sec": time.perf_counter() - lookup_started,
                        "image_prepare_sec": 0.0,
                        "clip_encode_sec": 0.0,
                        "condition_tensor_sec": 0.0,
                        "vae_encode_sec": 0.0,
                        "text_encode_sec": 0.0,
                    },
                    True,
                )

            timings: Dict[str, float] = {}
            stage_started = time.perf_counter()
            image = Image.open(cache_key[0]).convert("RGB")
            cond_image = self.transform(image).unsqueeze(1).unsqueeze(0).to(
                self.device, torch.bfloat16
            )
            timings["image_prepare_sec"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                clip_context = self.clip.visual(cond_image)
            timings["clip_encode_sec"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            ref_target_masks = torch.ones(
                3,
                self.height // self.vae_stride[1],
                self.width // self.vae_stride[2],
                device=self.device,
                dtype=torch.bfloat16,
            )
            msk = get_msk(
                self.condition_frame_num,
                cond_image,
                self.vae_stride,
                self.device,
            )
            video_frames = torch.zeros(
                1,
                cond_image.shape[1],
                self.condition_frame_num - cond_image.shape[2],
                self.height,
                self.width,
                device=self.device,
                dtype=torch.bfloat16,
            )
            padding_frames = torch.concat([cond_image, video_frames], dim=2)
            timings["condition_tensor_sec"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded_reference = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)
            y = torch.concat([msk, encoded_reference], dim=1)
            timings["vae_encode_sec"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            context = [
                self.text_encoder(
                    texts=str(prompt),
                    device="cpu" if self.config.t5_cpu else self.device,
                )[0].to(self.device, dtype=torch.bfloat16)
            ]
            edit_contexts: Dict[tuple, list] = {}
            for start, end, edit_text in cache_key[4]:
                edit_contexts[(start, end)] = [
                    self.text_encoder(
                        texts=edit_text,
                        device="cpu" if self.config.t5_cpu else self.device,
                    )[0].to(self.device, dtype=torch.bfloat16)
                ]
            timings["text_encode_sec"] = time.perf_counter() - stage_started
            timings["static_condition_lookup_sec"] = time.perf_counter() - lookup_started

            conditions = StaticAvatarConditions(
                cache_key=cache_key,
                cond_image=cond_image,
                clip_context=clip_context,
                ref_target_masks=ref_target_masks,
                y=y,
                context=context,
                edit_contexts=edit_contexts,
            )
            # LiveAct is single-active and the workbench uses one fixed avatar.
            # Bound retained GPU memory by keeping only the latest condition set.
            self._static_condition_cache.clear()
            self._static_condition_cache[cache_key] = conditions
            return conditions, timings, False

    def new_kv_cache(self) -> dict:
        kv_cache_tokens = self.frame_len * sum(self.blksz_lst) // self.world_size
        kv_cache_dtype = torch.float8_e4m3fn if self.config.fp8_kv_cache else torch.bfloat16
        kv_scale_shape = (1, kv_cache_tokens, 40, 1)
        return {
            i: {
                layer_id: {
                    "k": torch.zeros(
                        [1, kv_cache_tokens, 40, 128],
                        dtype=kv_cache_dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        [1, kv_cache_tokens, 40, 128],
                        dtype=kv_cache_dtype,
                        device=self.device,
                    ),
                    "k_scale": torch.ones(
                        kv_scale_shape,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if self.config.fp8_kv_cache
                    else None,
                    "v_scale": torch.ones(
                        kv_scale_shape,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if self.config.fp8_kv_cache
                    else None,
                    "mean_memory": False,
                    "offload_cache": False,
                    "fp8_kv_cache": self.config.fp8_kv_cache,
                }
                for layer_id in range(40)
            }
            for i in range(len(self.timesteps) - 1)
        }

    def reset_shared_kv_cache(self) -> dict:
        """Return the service-owned single-active cache without reallocating it.

        This intentionally follows ``demo.py``: a new sequence starts with the
        6-latent block and then the 8-latent block. Those calls overwrite the
        cache slices visible to attention before ``update_cache`` becomes true,
        so stale values are not consumed. Avoiding both allocation and a full
        zero-fill keeps session startup independent of cache size.
        """

        return self.shared_kv_cache

    def start_session(self, config: AvatarSessionConfig) -> dict:
        # The cache is deliberately shared, so enforce the same single-active
        # contract here as the HTTP lifecycle layer instead of relying only on
        # callers to serialize sessions.
        for active_session_id in list(self.sessions):
            self.stop_session(active_session_id)
        session = LiveActAvatarSession(self, config)
        self.sessions[config.session_id] = session
        return self.session_status(config.session_id)

    def prepare_session(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        session.prepare_static_conditions()
        return self.session_status(session_id)

    def push_pcm(
        self,
        session_id: str,
        pcm_bytes: bytes,
        sample_rate: int,
        is_last: bool = False,
    ) -> dict:
        session = self.sessions[session_id]
        chunks = session.append_pcm(pcm_bytes, sample_rate=sample_rate, is_last=is_last)
        return self.session_status(session_id, chunks=chunks)

    def prepare_next_audio_window(
        self,
        session_id: str,
        *,
        timeout_sec: float = 0.0,
    ) -> Optional[PreparedAvatarAudioWindow]:
        session = self.sessions[session_id]
        return session.prepare_next_audio_window(timeout_sec=timeout_sec)

    def generate_prepared_audio_window(
        self,
        session_id: str,
        prepared: PreparedAvatarAudioWindow,
    ) -> dict:
        session = self.sessions[session_id]
        chunk = session.generate_prepared_audio_window(prepared)
        return self.session_status(session_id, chunks=[chunk])

    def abort_session(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        session.abort()
        return self.session_status(session_id)

    def stop_session(self, session_id: str) -> dict:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return {"session_id": session_id, "found": False}
        # Do not drain buffered PCM here.  Stop means the user hung up, so
        # pending audio/video work must be discarded instead of inferred.
        save_result = session.close(discard_pending=True)
        return {
            "session_id": session_id,
            "found": True,
            "closed": True,
            "stream_path": f"/stream/{session_id}/{M3U8_NAME}",
            **save_result,
        }

    def video_status(self, session_id: str) -> dict:
        session_id = str(session_id)
        video_path = os.path.abspath(
            os.path.join(self.config.video_save_root, f"{session_id}.mp4")
        )
        temporary_path = video_path + ".part.mp4"
        video_ready = bool(os.path.isfile(video_path) and os.path.getsize(video_path) > 0)
        return {
            "session_id": session_id,
            "session_active": session_id in self.sessions,
            "video_path": video_path,
            "video_ready": video_ready,
            "video_saving": bool(os.path.isfile(temporary_path)),
            "video_size_bytes": int(os.path.getsize(video_path)) if video_ready else 0,
        }

    def session_status(
        self,
        session_id: str,
        chunks: Optional[List[AvatarChunkResult]] = None,
    ) -> dict:
        session = self.sessions[session_id]
        timeline_stats = session.audio_timeline.stats()
        with session.audio_source_condition:
            continuous_started = bool(session.continuous_started)
            speech_active = bool(session.speech_active)
            pending_real_samples = int(session.pending_real_pcm.size)
            continuous_started_at = session.continuous_started_at
        with session.state_lock:
            segment_id = int(session.segment_id)
            iteration = int(session.iteration)
            inference_active = bool(session.inference_active)
            conditions_ready = bool(session.conditions_ready)
            preparing_conditions = bool(session.preparing_conditions)
            preparation_error = session.preparation_error
            total_muxed_audio_samples = int(session.total_muxed_audio_samples)
            total_muxed_video_frames = int(session.total_muxed_video_frames)
            received_pcm_samples = int(session.received_pcm_samples)
            generated_video_frames = int(session.generated_video_frames)
            inference_cost_sec = float(session.inference_cost_sec)
            first_chunk_frames = int(session.first_chunk_frames)
            first_chunk_cost_sec = float(session.first_chunk_cost_sec)
            steady_video_frames = int(session.steady_video_frames)
            steady_inference_cost_sec = float(session.steady_inference_cost_sec)
            last_chunk_cost_sec = float(session.last_chunk_cost_sec)
            max_chunk_cost_sec = float(session.max_chunk_cost_sec)
            committed_real_samples = int(session.committed_real_samples)
            committed_silence_samples = int(session.committed_silence_samples)
            interrupted_pending_samples = int(session.interrupted_pending_samples)
            stage_timings = dict(session.stage_timings)
            stage_timings.update(
                {
                    "first_pcm_received_sec": session.first_pcm_received_sec,
                    "first_window_ready_sec": session.first_window_ready_sec,
                    "first_inference_started_sec": session.first_inference_started_sec,
                    "first_chunk_completed_sec": session.first_chunk_completed_sec,
                    "first_stream_ready_sec": session.first_stream_ready_sec,
                }
            )
            audio_start_idx, audio_end_idx = session._audio_start_end_frames(iteration)
            context_start_idx = max(0, audio_start_idx - 2)
        next_window_start_sample = round(
            session.input_sample_rate * (context_start_idx / float(session.config.fps))
        )
        next_window_center_sample = round(
            session.input_sample_rate * (audio_start_idx / float(session.config.fps))
        )
        next_window_end_sample = round(
            session.input_sample_rate * ((audio_end_idx + 2) / float(session.config.fps))
        )
        next_window_ready = bool(
            continuous_started
            and (
                timeline_stats.received_samples >= next_window_end_sample
                or pending_real_samples > 0
                or not speech_active
            )
        )
        stream_path = f"/stream/{session_id}/{M3U8_NAME}"
        generated_video_sec = (
            generated_video_frames / float(max(1, session.config.fps))
        )
        steady_video_sec = (
            steady_video_frames / float(max(1, session.config.fps))
        )
        return {
            "session_id": session_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "fps": int(session.config.fps),
            "size": f"{self.width}*{self.height}",
            "generation_mode": "continuous_speech_or_silence",
            "session_stage": (
                "preparation_error"
                if preparation_error
                else "preparing_conditions"
                if preparing_conditions or not conditions_ready
                else
                "inferencing"
                if inference_active
                else "streaming"
                if session.stream_ready
                else "buffering_pcm"
            ),
            "startup_timings": stage_timings,
            "segment_id": segment_id,
            "conditions_ready": conditions_ready,
            "preparation_error": preparation_error,
            "iteration": iteration,
            "inference_active": inference_active,
            # A response ending only changes audio_mode to silence. The input
            # timeline finishes exclusively when the complete call is stopped.
            "input_finished": False,
            "continuous_started": continuous_started,
            "continuous_elapsed_sec": (
                max(0.0, time.monotonic() - float(continuous_started_at))
                if continuous_started_at is not None
                else 0.0
            ),
            "audio_mode": (
                "standby"
                if not continuous_started
                else "speech"
                if speech_active or pending_real_samples > 0
                else "silence"
            ),
            "speech_active": speech_active,
            "received_pcm_sec": float(
                received_pcm_samples / max(1, session.input_sample_rate)
            ),
            "continuous_timeline_sec": float(timeline_stats.received_sec),
            "buffer_sec": float(timeline_stats.retained_sec),
            "buffer_start_sec": float(
                timeline_stats.retained_start_sample / max(1, session.input_sample_rate)
            ),
            "muxed_sec": float(
                total_muxed_audio_samples / max(1, session.input_sample_rate)
            ),
            "segment_muxed_sec": float(timeline_stats.muxed_sec),
            "pending_real_pcm_sec": float(
                pending_real_samples / max(1, session.input_sample_rate)
            ),
            # Backward-compatible alias consumed by the existing status/log UI.
            "queued_real_pcm_sec": float(
                pending_real_samples / max(1, session.input_sample_rate)
            ),
            "committed_real_pcm_sec": float(
                committed_real_samples / max(1, session.input_sample_rate)
            ),
            "committed_silence_sec": float(
                committed_silence_samples / max(1, session.input_sample_rate)
            ),
            "interrupted_pcm_dropped_sec": float(
                interrupted_pending_samples / max(1, session.input_sample_rate)
            ),
            "next_window_start_sec": float(
                next_window_start_sample / max(1, session.input_sample_rate)
            ),
            "next_window_center_sec": float(
                next_window_center_sample / max(1, session.input_sample_rate)
            ),
            "next_window_end_sec": float(
                next_window_end_sample / max(1, session.input_sample_rate)
            ),
            "next_window_ready": next_window_ready,
            "final_target_chunks": None,
            "waiting_for_pcm_sec": float(
                max(0, next_window_end_sample - timeline_stats.received_samples)
                / max(1, session.input_sample_rate)
            )
            if continuous_started and speech_active
            else 0.0,
            "pcm_queue_version": int(timeline_stats.version),
            "stream_ready": bool(session.stream_ready),
            "stream_path": stream_path,
            "video_path": session.final_video_path,
            "video_ready": bool(os.path.exists(session.final_video_path)),
            "generated_video_frames": generated_video_frames,
            "generated_video_sec": float(generated_video_sec),
            "realtime_margin_sec": float(
                generated_video_sec
                - (
                    max(0.0, time.monotonic() - float(continuous_started_at))
                    if continuous_started_at is not None
                    else 0.0
                )
            ),
            "muxed_video_frames": total_muxed_video_frames,
            "inference_cost_sec": inference_cost_sec,
            "rtf": (
                float(inference_cost_sec / generated_video_sec)
                if generated_video_sec > 0
                else None
            ),
            "first_chunk_frames": first_chunk_frames,
            "first_chunk_cost_sec": first_chunk_cost_sec,
            "first_chunk_rtf": (
                float(
                    first_chunk_cost_sec
                    / (first_chunk_frames / float(max(1, session.config.fps)))
                )
                if first_chunk_frames > 0
                else None
            ),
            "steady_video_frames": steady_video_frames,
            "steady_inference_cost_sec": steady_inference_cost_sec,
            "steady_rtf": (
                float(steady_inference_cost_sec / steady_video_sec)
                if steady_video_sec > 0
                else None
            ),
            "last_chunk_cost_sec": last_chunk_cost_sec,
            "max_chunk_cost_sec": max_chunk_cost_sec,
            "generated_chunks": [chunk.__dict__ for chunk in (chunks or [])],
        }

    def warmup(self) -> None:
        """Warm both demo-compatible LiveAct latent block shapes.

        This deliberately moves CUDA/JIT/compile cold cost to service startup.
        It does not change the model window, solver, timesteps, or output.
        """

        if self.rank == 0:
            print(
                "[avatar] warmup started (uniform backend, demo-compatible 6/8 latent blocks)",
                flush=True,
            )
        started = time.perf_counter()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        frame_num = (sum(self.blksz_lst) - 1) * self.vae_stride[0] + 1
        torch.manual_seed(int(self.config.seed))
        cache = self.reset_shared_kv_cache()
        with torch.no_grad():
            with self._static_condition_lock:
                fixed_conditions = next(iter(self._static_condition_cache.values()), None)
            if fixed_conditions is not None:
                # Use the real fixed image/prompt tensors for warmup so strides
                # and condition shapes match the first user task exactly.
                clip_context = fixed_conditions.clip_context
                ref_target_masks = fixed_conditions.ref_target_masks
                y = fixed_conditions.y
                context = fixed_conditions.context
            else:
                cond_image = torch.randn(
                    1,
                    3,
                    1,
                    self.height,
                    self.width,
                    device=self.device,
                    dtype=torch.bfloat16,
                ).clamp_(-1, 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clip_context = self.clip.visual(cond_image)

                ref_target_masks = torch.ones(
                    3,
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                video_frames = torch.zeros(
                    1,
                    cond_image.shape[1],
                    frame_num - cond_image.shape[2],
                    self.height,
                    self.width,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                padding_frames = torch.concat([cond_image, video_frames], dim=2)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)
                y = torch.concat(
                    [get_msk(frame_num, cond_image, self.vae_stride, self.device), y],
                    dim=1,
                )
                context = [
                    self.text_encoder(
                        texts=self.config.preload_prompt or DEFAULT_AVATAR_PROMPT,
                        device="cpu" if self.config.t5_cpu else self.device,
                    )[0].to(self.device, dtype=torch.bfloat16)
                ]

            dummy_audio = torch.randn(16000 * 6)
            audio_embedding = get_embedding(
                dummy_audio,
                self.wav2vec_feature_extractor,
                self.audio_encoder,
                device=self.device,
            )
            pre_latent = None
            # iteration=2 is materially different from iteration=1 even though
            # both use the 8-latent block: it enables the KV-cache compression
            # path (update_cache=True).  Warm all three execution states so the
            # first user session never compiles that steady-state branch.
            for iteration, f_idx in enumerate((0, 1, 1)):
                audio_start_idx = (
                    0
                    if iteration <= 1
                    else (iteration - 1) * self.blksz_lst[-1] * self.vae_stride[0]
                )
                audio_embs = get_audio_emb(
                    audio_embedding,
                    audio_start_idx,
                    audio_start_idx + frame_num,
                    self.device,
                )
                wan_i2v_model = self.wan_model_for_iteration(iteration)
                vae_decode = self.vae_decode_for_iteration(iteration)
                model_backend = self.execution_backend_for_iteration(
                    iteration, component="model"
                )
                vae_backend = self.execution_backend_for_iteration(
                    iteration, component="vae"
                )
                latent = torch.randn(
                    16,
                    self.blksz_lst[f_idx],
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                y_cut = y[:, :, : frame_num // 4 + 1, ...]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for index in range(len(self.timesteps) - 1):
                        timestep = self.timesteps[index]
                        noise_pred = wan_i2v_model(
                            [latent],
                            t=timestep,
                            kv_cache=cache[index],
                            context=context,
                            clip_fea=clip_context,
                            ref_target_masks=ref_target_masks,
                            audio=audio_embs,
                            y=y_cut[
                                :,
                                :,
                                sum(self.blksz_lst[:f_idx]) : sum(self.blksz_lst[: f_idx + 1]),
                            ],
                            start_idx=sum(self.blksz_lst[:f_idx]) * self.frame_len,
                            end_idx=sum(self.blksz_lst[: f_idx + 1]) * self.frame_len,
                            update_cache=iteration > 1,
                            skip_audio=False if index in (1, 2) else True,
                        )[0]
                        dt = (self.timesteps[index] - self.timesteps[index + 1]) / 1000
                        latent = latent + (-noise_pred) * dt[0]

                    if iteration == 0:
                        videos = vae_decode(latent)
                    else:
                        videos = vae_decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent
                torch.cuda.synchronize(self.device)
                if self.rank == 0:
                    print(
                        f"[avatar] warmup block={iteration} frames={int(videos.shape[2])} "
                        f"model_backend={model_backend} vae_backend={vae_backend}",
                        flush=True,
                    )

        # Some condition tensors come from the persistent fixed-condition cache,
        # while the uncached path creates local image/VAE temporaries. Let Python
        # release whichever locals actually exist when this method returns;
        # unconditional ``del`` is invalid for the cached branch.
        self.reset_shared_kv_cache()
        torch.manual_seed(int(self.config.seed))
        torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            print(
                f"[avatar] warmup completed in {time.perf_counter() - started:.3f}s",
                flush=True,
            )


def pcm_b64_to_bytes(value: str) -> bytes:
    if not value:
        return b""
    return base64.b64decode(value.encode("ascii"))
