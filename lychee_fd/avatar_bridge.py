"""Asynchronous HTTP bridge from Lychee-FD to the LiveAct avatar sidecar."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Optional


class RemoteAvatarClient:
    """Small dependency-free client for one LiveAct avatar sidecar."""

    def __init__(self, base_url: str, timeout_sec: float = 120.0):
        self.base_url = str(base_url).strip().rstrip("/")
        self.timeout_sec = max(0.1, float(timeout_sec))

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"avatar sidecar HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"avatar sidecar request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("avatar sidecar returned a non-object response")
        if result.get("ok") is False:
            raise RuntimeError(str(result.get("error") or "avatar sidecar request failed"))
        return result

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"avatar sidecar health check failed: {exc}") from exc
        if not isinstance(result, dict) or not bool(result.get("ok", True)):
            raise RuntimeError(f"avatar sidecar is unhealthy: {result!r}")
        return result

    def status(self, *, session_id: str) -> dict:
        path = "/v1/avatar/status/" + urllib.parse.quote(
            str(session_id),
            safe="",
        )
        try:
            with urllib.request.urlopen(
                self.base_url + path,
                timeout=min(self.timeout_sec, 5.0),
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"avatar sidecar HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"avatar sidecar status failed: {exc}") from exc
        if not isinstance(result, dict) or result.get("ok") is False:
            raise RuntimeError(f"invalid avatar sidecar status: {result!r}")
        return result

    def start(
        self,
        *,
        session_id: str,
        image_path: str,
        prompt: str,
        fps: int,
        sample_rate: int,
        continuous_buffer_sec: float = 2.5,
    ) -> dict:
        return self._post(
            "/v1/avatar/start",
            {
                "session_id": str(session_id),
                "image_path": str(image_path),
                "prompt": str(prompt),
                "fps": int(fps),
                "input_sample_rate": int(sample_rate),
                "continuous_buffer_sec": max(0.5, float(continuous_buffer_sec)),
                "stream_video_only": False,
            },
        )

    def push_pcm(
        self,
        *,
        session_id: str,
        pcm_bytes: bytes,
        sample_rate: int,
        is_last: bool = False,
    ) -> dict:
        return self._post(
            "/v1/avatar/push_pcm",
            {
                "session_id": str(session_id),
                "pcm_b64": base64.b64encode(bytes(pcm_bytes or b"")).decode("ascii"),
                "sample_rate": int(sample_rate),
                # This ends only the current speech source. LiveAct drains the
                # queued real PCM and then keeps the call-wide HLS timeline
                # advancing with inferred silence until more PCM or hangup.
                "is_last": bool(is_last),
            },
        )

    def abort(self, *, session_id: str, reason: str) -> dict:
        return self._post(
            "/v1/avatar/abort",
            {"session_id": str(session_id), "reason": str(reason or "interrupt")},
        )

    def stop(self, *, session_id: str) -> dict:
        return self._post("/v1/avatar/stop", {"session_id": str(session_id)})

    def heartbeat(self, *, session_id: str) -> dict:
        return self._post("/v1/avatar/heartbeat", {"session_id": str(session_id)})


class RealtimeAvatarBridge:
    """Serialize sidecar calls without blocking Lychee's inference/TTS threads."""

    def __init__(
        self,
        *,
        image_path: str,
        prompt: str,
        avatar_url: str,
        fps: int = 20,
        sample_rate: int = 24000,
        session_id: Optional[str] = None,
        timeout_sec: float = 120.0,
        event_callback: Optional[Callable[[dict], None]] = None,
        queue_size: int = 512,
        heartbeat_interval_sec: float = 5.0,
        activity_grace_sec: float = 15.0,
        segment_drain_timeout_sec: float = 180.0,
        status_poll_interval_sec: float = 0.2,
        playback_buffer_sec: float = 2.5,
        client: Optional[RemoteAvatarClient] = None,
    ):
        self.client = client or RemoteAvatarClient(avatar_url, timeout_sec=timeout_sec)
        self.session_id = str(session_id or uuid.uuid4().hex)
        self.image_path = str(image_path)
        self.prompt = str(prompt)
        self.fps = int(fps)
        self.sample_rate = int(sample_rate)
        self.event_callback = event_callback if callable(event_callback) else None
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_sentinel = object()
        self._state_lock = threading.Lock()
        self._started = False
        self._minimum_generation_id: Optional[int] = None
        self._abort_serial = 0
        self._active_generation_id: Optional[int] = None
        self._active_stream_id = None
        self._active_segment_id: Optional[int] = None
        self._active_segment_start_sec = 0.0
        self._active_generation_ready = False
        self._stream_ready_emitted = False
        self._closed = False
        self._last_activity_monotonic = time.monotonic()
        self._heartbeat_interval_sec = max(1.0, float(heartbeat_interval_sec))
        self._activity_grace_sec = max(
            self._heartbeat_interval_sec,
            float(activity_grace_sec),
        )
        self._segment_drain_timeout_sec = max(
            10.0,
            float(segment_drain_timeout_sec),
        )
        self._status_poll_interval_sec = max(
            0.05,
            float(status_poll_interval_sec),
        )
        # Do not expose a live HLS playlist containing only one short fragment.
        # Continuous silence inference will extend even a short first response
        # until this startup threshold is available.
        self._playback_buffer_sec = max(0.0, float(playback_buffer_sec))
        self._heartbeat_stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"avatar_bridge_{self.session_id[:8]}",
            daemon=True,
        )
        self._heartbeat_worker = threading.Thread(
            target=self._heartbeat_loop,
            name=f"avatar_heartbeat_{self.session_id[:8]}",
            daemon=True,
        )
        self._worker.start()
        self._heartbeat_worker.start()

    def _emit(self, payload: dict) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(dict(payload))
        except Exception:
            # Avatar callbacks must never terminate the bridge worker.
            pass

    def _enqueue(self, task: object) -> bool:
        with self._state_lock:
            if self._closed:
                return False
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            self._emit(
                {
                    "type": "avatar_error",
                    "error": "avatar bridge queue is full; PCM was not queued",
                }
            )
            return False

    def start(self) -> None:
        self.touch()
        self._enqueue({"type": "start"})

    def touch(self) -> None:
        """Record that the owning Lychee realtime session is still alive."""
        with self._state_lock:
            if not self._closed:
                self._last_activity_monotonic = time.monotonic()

    def submit_pcm(
        self,
        pcm_bytes: bytes,
        *,
        is_last: bool = False,
        generation_id=None,
        stream_id=None,
    ) -> bool:
        # A zero-byte final packet is a control boundary.  Token2Wav may have
        # emitted its last real PCM in the preceding packet, but LiveAct still
        # needs an explicit is_last marker before it may pad the short final
        # attention window.  Empty non-final packets remain meaningless.
        if not pcm_bytes and not is_last:
            return False
        self.touch()
        try:
            normalized_generation = int(generation_id) if generation_id is not None else None
        except (TypeError, ValueError):
            normalized_generation = None
        with self._state_lock:
            minimum_generation = self._minimum_generation_id
            abort_serial = int(self._abort_serial)
        if (
            normalized_generation is not None
            and minimum_generation is not None
            and normalized_generation < minimum_generation
        ):
            return False
        return self._enqueue(
            {
                "type": "pcm",
                "pcm_bytes": bytes(pcm_bytes),
                "is_last": bool(is_last),
                "generation_id": normalized_generation,
                "stream_id": stream_id,
                "abort_serial": abort_serial,
            }
        )

    def submit_abort(self, *, reason: str = "interrupt", generation_id=None) -> None:
        try:
            normalized_generation = int(generation_id) if generation_id is not None else None
        except (TypeError, ValueError):
            normalized_generation = None
        with self._state_lock:
            interrupted_generation = self._active_generation_id
            self._abort_serial += 1
            if normalized_generation is not None:
                self._minimum_generation_id = normalized_generation
            # Do not clear the session-wide stream owner here. S->L interrupts
            # speech, not LiveAct/HLS. Keeping an active owner also lets the
            # readiness poll expose a short first response whose remaining
            # model windows are completed with silence after the interrupt.

        # Remove queued PCM from the interrupted generation. The currently
        # executing HTTP request cannot be preempted, but all waiting work can.
        with self._queue.mutex:
            retained = []
            while self._queue.queue:
                item = self._queue.queue.popleft()
                if isinstance(item, dict) and item.get("type") == "pcm":
                    continue
                retained.append(item)
            self._queue.queue.extend(retained)
            self._queue.not_full.notify_all()

        self._enqueue(
            {
                "type": "abort",
                "reason": str(reason or "interrupt"),
                "generation_id": normalized_generation,
                "interrupted_generation_id": interrupted_generation,
            }
        )

    def _clear_active_generation(self) -> None:
        with self._state_lock:
            self._active_generation_id = None
            self._active_stream_id = None
            self._active_segment_id = None
            self._active_segment_start_sec = 0.0
            self._active_generation_ready = False

    def _activate_generation(self, task: dict, result: dict) -> None:
        generation_id = task.get("generation_id")
        stream_id = task.get("stream_id")
        should_emit = False
        with self._state_lock:
            if self._closed:
                return
            if (
                self._active_generation_id != generation_id
                or self._active_stream_id != stream_id
            ):
                muxed_sec = float(result.get("muxed_sec") or 0.0)
                segment_muxed_sec = float(result.get("segment_muxed_sec") or 0.0)
                self._active_generation_id = generation_id
                self._active_stream_id = stream_id
                self._active_segment_id = int(result.get("segment_id") or 0)
                self._active_segment_start_sec = max(0.0, muxed_sec - segment_muxed_sec)
                if not self._stream_ready_emitted:
                    self._active_generation_ready = False
                    should_emit = True
        if should_emit:
            self._emit(
                {
                    "type": "avatar_buffering",
                    "avatar_session_id": self.session_id,
                    "generation_id": generation_id,
                    "stream_id": stream_id,
                    "segment_id": result.get("segment_id"),
                    "playback_start_sec": max(
                        0.0,
                        float(result.get("muxed_sec") or 0.0)
                        - float(result.get("segment_muxed_sec") or 0.0),
                    ),
                }
            )

    def _maybe_emit_active_generation_ready(self, status: dict) -> None:
        with self._state_lock:
            if (
                self._closed
                or not self._started
                or self._active_generation_id is None
                or self._stream_ready_emitted
            ):
                return
            generation_id = self._active_generation_id
            stream_id = self._active_stream_id
            segment_id = self._active_segment_id
            segment_start_sec = float(self._active_segment_start_sec)
        iteration = int(status.get("iteration") or 0)
        segment_muxed_sec = float(status.get("segment_muxed_sec") or 0.0)
        muxed_sec = float(status.get("muxed_sec") or 0.0)
        playback_buffer_ready = bool(
            muxed_sec + 1e-3 >= self._playback_buffer_sec
        )
        ready = bool(
            status.get("stream_ready")
            and iteration > 0
            and playback_buffer_ready
        )
        if not ready:
            return
        with self._state_lock:
            if (
                self._closed
                or self._active_generation_id != generation_id
                or self._active_stream_id != stream_id
                or self._active_segment_id != segment_id
                or self._stream_ready_emitted
            ):
                return
            self._active_generation_ready = True
            self._stream_ready_emitted = True
        self._emit(
            {
                "type": "avatar_stream_ready",
                "avatar_session_id": self.session_id,
                "generation_id": generation_id,
                "stream_id": stream_id,
                "segment_id": segment_id,
                # One HLS timeline owns the complete call and is attached only
                # once. Later speech/silence transitions never seek to a new
                # response segment.
                "playback_start_sec": 0.0,
                "muxed_sec": muxed_sec,
                "segment_muxed_sec": segment_muxed_sec,
                "playback_buffer_sec": self._playback_buffer_sec,
                "playback_buffer_ready": playback_buffer_ready,
                "final_response_ready": False,
                "iteration": iteration,
                "stream_ready": True,
            }
        )

    def _poll_active_generation_status(self) -> None:
        with self._state_lock:
            should_poll = bool(
                not self._closed
                and self._started
                and self._active_generation_id is not None
                and not self._stream_ready_emitted
            )
        if not should_poll:
            return
        status = self.client.status(session_id=self.session_id)
        self._maybe_emit_active_generation_ready(status)

    def stop(self, *, join_timeout_sec: float = 130.0) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._heartbeat_stop.set()
        # Hangup has priority over every queued PCM/abort request.  The one
        # request currently in flight cannot be preempted, but no stale item
        # is allowed to run after it returns.
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.queue.append({"type": "stop"})
            self._queue.queue.append(self._stop_sentinel)
            self._queue.not_empty.notify_all()
            self._queue.not_full.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=max(0.0, float(join_timeout_sec)))
        if self._heartbeat_worker.is_alive():
            self._heartbeat_worker.join(timeout=self._heartbeat_interval_sec + 1.0)
        if self._worker.is_alive():
            self._emit(
                {
                    "type": "avatar_error",
                    "avatar_session_id": self.session_id,
                    "error": "avatar bridge stop timed out; sidecar TTL cleanup will take over",
                }
            )

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.client.health()
        result = self.client.start(
            session_id=self.session_id,
            image_path=self.image_path,
            prompt=self.prompt,
            fps=self.fps,
            sample_rate=self.sample_rate,
            continuous_buffer_sec=self._playback_buffer_sec,
        )
        self._started = True
        self.touch()
        self._emit({"type": "avatar_ready", "avatar_session_id": self.session_id, **result})

    def _heartbeat_loop(self) -> None:
        next_heartbeat = time.monotonic() + self._heartbeat_interval_sec
        while not self._heartbeat_stop.wait(self._status_poll_interval_sec):
            with self._state_lock:
                started = bool(self._started)
                closed = bool(self._closed)
                last_activity = float(self._last_activity_monotonic)
            if closed:
                return
            if not started:
                continue
            try:
                self._poll_active_generation_status()
            except Exception:
                # Readiness polling is best-effort. The PCM request path and
                # final drain path report actionable sidecar failures.
                pass
            now = time.monotonic()
            if now < next_heartbeat:
                continue
            next_heartbeat = now + self._heartbeat_interval_sec
            # Stop renewing the remote lease when the browser/backend session
            # has gone quiet.  The sidecar watchdog will then reclaim it.
            if now - last_activity > self._activity_grace_sec:
                continue
            try:
                self.client.heartbeat(session_id=self.session_id)
            except Exception:
                # The main bridge request path reports actionable failures.
                # Heartbeats are best-effort and must not flood SSE errors.
                pass

    def _is_stale_pcm(self, task: dict) -> bool:
        generation_id = task.get("generation_id")
        with self._state_lock:
            minimum_generation = self._minimum_generation_id
        return bool(
            generation_id is not None
            and minimum_generation is not None
            and int(generation_id) < int(minimum_generation)
        )

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is self._stop_sentinel:
                break
            if not isinstance(task, dict):
                continue
            try:
                task_type = str(task.get("type") or "")
                if task_type == "start":
                    self._ensure_started()
                    continue
                if task_type == "pcm":
                    if self._is_stale_pcm(task):
                        continue
                    self._ensure_started()
                    result = self.client.push_pcm(
                        session_id=self.session_id,
                        pcm_bytes=task.get("pcm_bytes", b""),
                        sample_rate=self.sample_rate,
                        is_last=bool(task.get("is_last", False)),
                    )
                    self._activate_generation(task, result)
                    self._emit(
                        {
                            "type": "avatar_chunk",
                            "avatar_session_id": self.session_id,
                            "generation_id": task.get("generation_id"),
                            "stream_id": task.get("stream_id"),
                            **result,
                        }
                    )
                    if bool(task.get("is_last", False)):
                        self._emit(
                            {
                                "type": "avatar_audio_mode",
                                "mode": "silence",
                                "reason": "tts_end",
                                "avatar_session_id": self.session_id,
                                "generation_id": task.get("generation_id"),
                                "stream_id": task.get("stream_id"),
                            }
                        )
                    continue
                if task_type == "abort":
                    if not self._started:
                        continue
                    reason = str(task.get("reason") or "interrupt")
                    result = self.client.abort(
                        session_id=self.session_id,
                        reason=reason,
                    )
                    self._emit(
                        {
                            **result,
                            # Full-duplex interruption changes the continuous
                            # audio source to silence. It never tears down the
                            # session-wide LiveAct cache or HLS player.
                            "type": "avatar_speech_interrupted",
                            "mode": "silence",
                            "avatar_session_id": self.session_id,
                            "generation_id": task.get("generation_id"),
                            "interrupted_generation_id": task.get(
                                "interrupted_generation_id"
                            ),
                            "reason": reason,
                            "playback_end_sec": float(result.get("muxed_sec") or 0.0),
                            "segment_muxed_sec": float(
                                result.get("segment_muxed_sec") or 0.0
                            ),
                        }
                    )
                    continue
                if task_type == "stop":
                    if self._started:
                        result = self.client.stop(session_id=self.session_id)
                        self._emit(
                            {
                                "type": "avatar_stopped",
                                "avatar_session_id": self.session_id,
                                **result,
                            }
                        )
                    self._started = False
                    continue
            except Exception as exc:
                self._emit(
                    {
                        "type": "avatar_error",
                        "avatar_session_id": self.session_id,
                        "error": str(exc),
                    }
                )
