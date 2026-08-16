import argparse
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import torch.distributed as dist

from liveact_avatar.avatar_engine import (
    AvatarSessionConfig,
    LiveActEngineConfig,
    LiveActStreamingEngine,
    M3U8_NAME,
    PreparedAvatarAudioWindow,
    StaleAvatarAudioWindowError,
    pcm_b64_to_bytes,
)
from liveact_avatar.defaults import (
    DEFAULT_AVATAR_IMAGE_PATH,
    DEFAULT_AVATAR_PROMPT,
)


engine: LiveActStreamingEngine
control_pg = None
dispatch_lock = threading.Lock()
session_lifecycle_lock = threading.RLock()
continuous_lock = threading.Lock()
continuous_threads = {}
continuous_stop_events = {}
session_activity_lock = threading.Lock()
session_last_seen = {}
session_ttl_sec = 45.0
stop_join_timeout_sec = 130.0
watchdog_stop_event = threading.Event()


def _truthy(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _handle_command(command: dict) -> dict:
    ctype = str(command.get("type") or "")
    if ctype == "start":
        session_id = str(command.get("session_id") or uuid.uuid4().hex)
        cfg = AvatarSessionConfig(
            session_id=session_id,
            image_path=str(command["image_path"]),
            prompt=str(command.get("prompt") or "A person is speaking naturally."),
            fps=int(command.get("fps") or 20),
            input_sample_rate=int(command.get("input_sample_rate") or 24000),
            stream_video_only=_truthy(command.get("stream_video_only"), False),
            continuous_buffer_sec=float(command.get("continuous_buffer_sec") or 2.5),
            edit_prompts=list(command.get("edit_prompts") or []),
        )
        return engine.start_session(cfg)

    if ctype == "prepare_session":
        return engine.prepare_session(str(command["session_id"]))

    if ctype == "push_pcm":
        return engine.push_pcm(
            str(command["session_id"]),
            pcm_b64_to_bytes(str(command.get("pcm_b64") or "")),
            sample_rate=int(command.get("sample_rate") or 24000),
            is_last=_truthy(command.get("is_last"), False),
        )

    if ctype == "generate_window":
        prepared = PreparedAvatarAudioWindow(
            segment_id=int(command["segment_id"]),
            iteration=int(command["iteration"]),
            start_sample=int(command["start_sample"]),
            end_sample=int(command["end_sample"]),
            real_samples=int(command["real_samples"]),
            padded_samples=int(command["padded_samples"]),
            center_offset_frames=int(command.get("center_offset_frames") or 0),
            pcm_s16le=bytes(command.get("pcm_s16le") or b""),
        )
        return engine.generate_prepared_audio_window(
            str(command["session_id"]),
            prepared,
        )

    if ctype == "abort":
        return engine.abort_session(str(command["session_id"]))

    if ctype == "stop":
        return engine.stop_session(str(command["session_id"]))

    if ctype == "status":
        return engine.session_status(str(command["session_id"]))

    raise ValueError(f"unknown avatar command: {ctype}")


def dispatch(command: dict) -> dict:
    with dispatch_lock:
        if engine.world_size > 1:
            payload = [dict(command)]
            dist.broadcast_object_list(payload, src=0, group=control_pg)
        return _handle_command(command)


def touch_session(session_id: str, *, create: bool = False) -> bool:
    session_id = str(session_id)
    with session_activity_lock:
        if not create and session_id not in session_last_seen:
            return False
        session_last_seen[session_id] = time.monotonic()
    return True


def forget_session(session_id: str) -> None:
    with session_activity_lock:
        session_last_seen.pop(str(session_id), None)


def session_activity_snapshot() -> dict:
    now = time.monotonic()
    with session_activity_lock:
        return {
            session_id: max(0.0, now - float(last_seen))
            for session_id, last_seen in session_last_seen.items()
        }


def _continuous_generation_loop(session_id: str, stop_event: threading.Event) -> None:
    try:
        if stop_event.is_set():
            return
        # Match demo.py's queued task lifecycle: /start creates a lightweight
        # session and returns, while static GPU conditions are prepared here.
        dispatch({"type": "prepare_session", "session_id": session_id})
        while not stop_event.is_set():
            try:
                prepared = engine.prepare_next_audio_window(
                    session_id,
                    timeout_sec=0.25,
                )
            except KeyError:
                break
            if prepared is None:
                continue
            if stop_event.is_set():
                break
            try:
                result = dispatch(
                    {
                        "type": "generate_window",
                        "session_id": session_id,
                        "segment_id": int(prepared.segment_id),
                        "iteration": int(prepared.iteration),
                        "start_sample": int(prepared.start_sample),
                        "end_sample": int(prepared.end_sample),
                        "real_samples": int(prepared.real_samples),
                        "padded_samples": int(prepared.padded_samples),
                        "center_offset_frames": int(prepared.center_offset_frames),
                        # Rank 0 ingests PCM without a distributed broadcast.
                        # Only immutable, inference-ready windows are sent to
                        # the second rank.
                        "pcm_s16le": bytes(prepared.pcm_s16le),
                    }
                )
            except StaleAvatarAudioWindowError:
                continue
            chunks = result.get("generated_chunks") or []
            frames = int(chunks[-1].get("frames") or 0) if chunks else 0
            if chunks:
                chunk = chunks[-1]
                print(
                    f"[avatar][audio-driven] session={session_id} "
                    f"iteration={chunk.get('iteration')} frames={frames} "
                    f"cost={float(chunk.get('cost_sec') or 0.0):.3f}s "
                    f"model_backend={chunk.get('model_backend') or 'unknown'} "
                    f"vae_backend={chunk.get('vae_backend') or 'unknown'} "
                    f"window={float(chunk.get('window_start_sec') or 0.0):.3f}-"
                    f"{float(chunk.get('window_end_sec') or 0.0):.3f}s "
                    f"padding={float(chunk.get('window_padding_sec') or 0.0):.3f}s "
                    f"audio_mode={result.get('audio_mode') or 'unknown'} "
                    f"buffer={float(result.get('buffer_sec') or 0.0):.3f}s "
                    f"queued_real={float(result.get('queued_real_pcm_sec') or 0.0):.3f}s",
                    flush=True,
                )
            # Lifecycle commands (abort/stop) still use dispatch_lock because
            # they mutate distributed model state. Yield briefly between GPU
            # windows so those commands are not starved by a deep PCM backlog.
            stop_event.wait(0.01)
    except Exception as exc:
        if not stop_event.is_set():
            print(f"[avatar][audio-driven][ERROR] session={session_id}: {exc}", flush=True)
    finally:
        with continuous_lock:
            if continuous_stop_events.get(session_id) is stop_event:
                continuous_stop_events.pop(session_id, None)
                continuous_threads.pop(session_id, None)


def start_continuous_generation(session_id: str) -> None:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_continuous_generation_loop,
        args=(session_id, stop_event),
        name=f"avatar_continuous_{session_id[:8]}",
        daemon=True,
    )
    with continuous_lock:
        continuous_stop_events[session_id] = stop_event
        continuous_threads[session_id] = thread
    thread.start()


def stop_continuous_generation(session_id: str, *, timeout_sec: float = None) -> bool:
    with continuous_lock:
        stop_event = continuous_stop_events.get(session_id)
        thread = continuous_threads.get(session_id)
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread is not threading.current_thread():
        timeout = stop_join_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        thread.join(timeout=timeout)
    alive = bool(thread is not None and thread.is_alive())
    if alive:
        print(
            f"[avatar][cleanup][ERROR] session={session_id} continuous thread "
            f"did not stop within {timeout_sec or stop_join_timeout_sec:.1f}s",
            flush=True,
        )
    return not alive


def _known_session_ids() -> list:
    with session_activity_lock:
        activity_ids = set(session_last_seen)
    with continuous_lock:
        continuous_ids = set(continuous_threads)
    return sorted(activity_ids | continuous_ids)


def _stop_avatar_session_locked(session_id: str, *, reason: str) -> dict:
    session_id = str(session_id)
    if not stop_continuous_generation(session_id):
        raise TimeoutError(
            f"avatar session {session_id} is still inside distributed inference; "
            "restart the sidecar if watchdog cleanup repeats"
        )
    result = dispatch({"type": "stop", "session_id": session_id})
    forget_session(session_id)
    print(f"[avatar][cleanup] session={session_id} reason={reason}", flush=True)
    return result


def _stop_all_sessions_locked(*, reason: str) -> list:
    results = []
    for session_id in _known_session_ids():
        results.append(_stop_avatar_session_locked(session_id, reason=reason))
    return results


def _session_watchdog_loop() -> None:
    while not watchdog_stop_event.wait(1.0):
        if session_ttl_sec <= 0:
            continue
        stale_ids = [
            session_id
            for session_id, age_sec in session_activity_snapshot().items()
            if age_sec > session_ttl_sec
        ]
        for session_id in stale_ids:
            try:
                with session_lifecycle_lock:
                    age_sec = session_activity_snapshot().get(session_id)
                    if age_sec is None or age_sec <= session_ttl_sec:
                        continue
                    _stop_avatar_session_locked(session_id, reason=f"heartbeat_timeout_{age_sec:.1f}s")
            except Exception as exc:
                print(
                    f"[avatar][watchdog][ERROR] session={session_id}: {exc}",
                    flush=True,
                )


def worker_loop_nonzero_rank() -> None:
    while True:
        payload = [None]
        dist.broadcast_object_list(payload, src=0, group=control_pg)
        command = payload[0]
        if not isinstance(command, dict):
            continue
        if command.get("type") == "__shutdown__":
            break
        try:
            _handle_command(command)
        except Exception as exc:
            # Rank 0 returns the request error to the caller.  A malformed
            # request must not terminate a nonzero torchrun worker and tear
            # down the whole distributed sidecar.
            print(
                f"[avatar][rank={engine.rank}][command-error] "
                f"type={command.get('type')}: {exc}",
                flush=True,
            )


def _viewer_html() -> bytes:
    return b"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LiveAct Avatar Stream Viewer</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
    main { max-width: 1120px; margin: 0 auto; padding: 20px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
    input { background: #1e1e1e; color: #eee; border: 1px solid #444; padding: 8px 10px; min-width: 260px; }
    button { background: #2f6feb; color: white; border: 0; padding: 9px 14px; cursor: pointer; }
    #audioBtn { background: #238636; }
    video { width: 100%; max-height: 72vh; background: #000; display: block; }
    pre { background: #1e1e1e; border: 1px solid #333; padding: 12px; overflow: auto; }
    .muted { color: #aaa; }
  </style>
</head>
<body>
  <main>
    <h2>LiveAct Avatar Stream Viewer</h2>
    <div class="row">
      <label>session_id</label>
      <input id="sessionId" value="test_liveact" />
      <button id="loadBtn">Load Stream</button>
      <button id="refreshBtn">Refresh Status</button>
      <button id="audioBtn" hidden>&#28857;&#20987;&#24320;&#21551;&#22768;&#38899;</button>
    </div>
    <p class="muted">HLS URL: <span id="hlsUrl"></span></p>
    <video id="video" controls autoplay playsinline></video>
    <h3>Status</h3>
    <pre id="status">waiting...</pre>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
  <script>
    let hls = null;
    let attachedSessionId = '';
    const video = document.getElementById('video');
    const sessionInput = document.getElementById('sessionId');
    const statusEl = document.getElementById('status');
    const hlsUrlEl = document.getElementById('hlsUrl');
    const audioBtn = document.getElementById('audioBtn');

    const params = new URLSearchParams(location.search);
    if (params.get('session_id')) sessionInput.value = params.get('session_id');

    function streamUrl() {
      return `/stream/${encodeURIComponent(sessionInput.value || 'test_liveact')}/live.m3u8`;
    }

    async function refreshStatus() {
      const sid = encodeURIComponent(sessionInput.value || 'test_liveact');
      try {
        const res = await fetch(`/v1/avatar/status/${sid}?t=${Date.now()}`);
        const text = await res.text();
        try {
          const payload = JSON.parse(text);
          statusEl.textContent = JSON.stringify(payload, null, 2);
          if (payload.stream_ready && attachedSessionId !== sessionInput.value) {
            attachStream();
          }
        } catch {
          statusEl.textContent = text;
        }
      } catch (err) {
        statusEl.textContent = String(err);
      }
    }

    function detachStream() {
      attachedSessionId = '';
      if (hls) {
        hls.destroy();
        hls = null;
      }
      video.pause();
      video.removeAttribute('src');
      video.load();
      audioBtn.hidden = true;
    }

    async function attemptPlayback() {
      video.volume = 1;
      video.muted = false;
      try {
        await video.play();
        audioBtn.hidden = true;
      } catch (_err) {
        // Chrome normally blocks unmuted autoplay because the HLS stream is
        // attached several seconds after the page was opened. Keep video
        // moving and expose an explicit user gesture to unlock its AAC track.
        video.muted = true;
        audioBtn.hidden = false;
        video.play().catch(() => {});
      }
    }

    function attachStream() {
      const url = streamUrl();
      hlsUrlEl.textContent = location.origin + url;
      if (hls) {
        hls.destroy();
        hls = null;
      }
      if (window.Hls && Hls.isSupported()) {
        hls = new Hls({ liveSyncDurationCount: 2, enableWorker: true });
        hls.on(Hls.Events.ERROR, (_event, data) => {
          if (data && data.fatal) {
            hls.destroy();
            hls = null;
            attachedSessionId = '';
          }
        });
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED, attemptPlayback);
        attachedSessionId = sessionInput.value;
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = url;
        video.addEventListener('loadedmetadata', attemptPlayback, { once: true });
        attachedSessionId = sessionInput.value;
      } else {
        statusEl.textContent = 'This browser cannot play HLS directly and hls.js failed to load. Use VLC/ffplay with the HLS URL.';
      }
    }

    function loadStream() {
      detachStream();
      hlsUrlEl.textContent = location.origin + streamUrl();
      refreshStatus();
    }

    document.getElementById('loadBtn').onclick = loadStream;
    document.getElementById('refreshBtn').onclick = refreshStatus;
    audioBtn.onclick = async () => {
      video.volume = 1;
      video.muted = false;
      try {
        await video.play();
        audioBtn.hidden = true;
      } catch (err) {
        statusEl.textContent = `The browser still blocked audio playback: ${String(err)}`;
      }
    };
    setInterval(refreshStatus, 2000);
    loadStream();
  </script>
</body>
</html>
"""


class AvatarRequestHandler(BaseHTTPRequestHandler):
    server_version = "LiveActAvatarHTTP/0.1"

    def log_message(self, fmt, *args):
        print(f"[avatar-http] {self.address_string()} - {fmt % args}", flush=True)

    def _base_url(self) -> str:
        host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_port}"
        return f"http://{host}"

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/viewer"}:
            self._send_bytes(200, _viewer_html(), "text/html; charset=utf-8")
            return
        if path == "/health":
            activity = session_activity_snapshot()
            self._send_json(
                200,
                {
                    "ok": True,
                    "rank": engine.rank,
                    "world_size": engine.world_size,
                    "size": f"{engine.width}*{engine.height}",
                    "sessions": list(engine.sessions.keys()),
                    "continuous_sessions": _known_session_ids(),
                    "session_idle_sec": {k: round(v, 3) for k, v in activity.items()},
                    "session_ttl_sec": float(session_ttl_sec),
                    "session_mode": "single_active_audio_driven",
                    "fixed_conditions_ready": bool(engine.preloaded_static_conditions),
                    "fixed_image_path": engine.config.preload_image_path,
                    "fixed_prompt": engine.config.preload_prompt,
                    "preload_condition_timings": engine.preload_condition_timings,
                    "viewer": self._base_url() + "/viewer",
                },
            )
            return
        if path.startswith("/v1/avatar/video_status/"):
            session_id = unquote(path.rsplit("/", 1)[-1])
            result = engine.video_status(session_id)
            result["ok"] = True
            result["video_url"] = self._base_url() + f"/video/{session_id}/output.mp4"
            self._send_json(200, result)
            return
        if path.startswith("/v1/avatar/status/"):
            session_id = unquote(path.rsplit("/", 1)[-1])
            try:
                # Status only reads thread-safe CPU-side counters. It must stay
                # responsive while the distributed GPU command owns dispatch_lock.
                result = engine.session_status(session_id)
                result["ok"] = True
                result["stream_url"] = self._base_url() + result["stream_path"]
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(404, {"ok": False, "error": str(exc)})
            return
        if path.startswith("/stream/"):
            self._serve_stream(path)
            return
        if path.startswith("/video/"):
            self._serve_video(path)
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json()
            if path == "/v1/avatar/start":
                requested_session_id = str(payload.get("session_id") or uuid.uuid4().hex)
                with session_lifecycle_lock:
                    # The distributed LiveAct engine is single-active.  A new
                    # call must never coexist with an orphaned inference loop.
                    _stop_all_sessions_locked(reason=f"superseded_by_{requested_session_id}")
                    result = dispatch(
                        {
                            "type": "start",
                            "session_id": requested_session_id,
                            "image_path": (
                                payload.get("image_path")
                                or engine.config.preload_image_path
                                or DEFAULT_AVATAR_IMAGE_PATH
                            ),
                            "prompt": (
                                payload.get("prompt")
                                or engine.config.preload_prompt
                                or DEFAULT_AVATAR_PROMPT
                            ),
                            "fps": int(payload.get("fps") or 20),
                            "input_sample_rate": int(payload.get("input_sample_rate") or 24000),
                            "stream_video_only": payload.get("stream_video_only", False),
                            "continuous_buffer_sec": float(
                                payload.get("continuous_buffer_sec") or 2.5
                            ),
                            "edit_prompts": payload.get("edit_prompts") or [],
                        }
                    )
                    touch_session(result["session_id"], create=True)
                    start_continuous_generation(result["session_id"])
                result["ok"] = True
                result["stream_url"] = self._base_url() + result["stream_path"]
                result["viewer_url"] = self._base_url() + f"/viewer?session_id={result['session_id']}"
                self._send_json(200, result)
                return
            if path == "/v1/avatar/push_pcm":
                session_id = str(payload.get("session_id") or "").strip()
                if not session_id:
                    raise ValueError("session_id is required")
                if not touch_session(session_id):
                    raise KeyError(f"avatar session not found or expired: {session_id}")
                # PCM ingestion is deliberately local to rank 0 and does not
                # acquire dispatch_lock. The audio-driven worker later
                # broadcasts an immutable model window to every GPU rank.
                result = engine.push_pcm(
                    session_id,
                    pcm_b64_to_bytes(str(payload.get("pcm_b64") or "")),
                    sample_rate=int(payload.get("sample_rate") or 24000),
                    is_last=_truthy(payload.get("is_last"), False),
                )
                result["ok"] = True
                result["stream_url"] = self._base_url() + result["stream_path"]
                result["viewer_url"] = self._base_url() + f"/viewer?session_id={result['session_id']}"
                self._send_json(200, result)
                return
            if path == "/v1/avatar/abort":
                session_id = str(payload.get("session_id") or "").strip()
                if not session_id:
                    raise ValueError("session_id is required")
                if not touch_session(session_id):
                    raise KeyError(f"avatar session not found or expired: {session_id}")
                # Continuous-mode interruption only clears rank-0's pending
                # real PCM and switches the source to silence. It does not
                # mutate distributed model/KV state, so it must not wait for
                # the current multi-GPU inference command to release
                # dispatch_lock.
                result = engine.abort_session(session_id)
                result["ok"] = True
                self._send_json(200, result)
                return
            if path == "/v1/avatar/stop":
                session_id = str(payload.get("session_id") or "").strip()
                if not session_id:
                    raise ValueError("session_id is required")
                with session_lifecycle_lock:
                    cleanup_results = _stop_all_sessions_locked(reason=f"hangup_{session_id}")
                    matching = [item for item in cleanup_results if item.get("session_id") == session_id]
                    result = matching[-1] if matching else dispatch({"type": "stop", "session_id": session_id})
                    forget_session(session_id)
                result["ok"] = True
                result["cleaned_sessions"] = [
                    item.get("session_id") for item in cleanup_results if item.get("session_id")
                ]
                if result.get("stream_path"):
                    result["stream_url"] = self._base_url() + result["stream_path"]
                if result.get("video_path"):
                    result["video_url"] = self._base_url() + f"/video/{session_id}/output.mp4"
                self._send_json(200, result)
                return
            if path == "/v1/avatar/heartbeat":
                session_id = str(payload.get("session_id") or "").strip()
                if not session_id:
                    raise ValueError("session_id is required")
                if not touch_session(session_id):
                    self._send_json(404, {"ok": False, "error": "avatar session not found"})
                    return
                self._send_json(200, {"ok": True, "session_id": session_id})
                return
            if path == "/v1/avatar/stop_all":
                with session_lifecycle_lock:
                    cleanup_results = _stop_all_sessions_locked(reason="explicit_stop_all")
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "cleaned_sessions": [
                            item.get("session_id") for item in cleanup_results if item.get("session_id")
                        ],
                    },
                )
                return
            self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _serve_stream(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            self._send_json(404, {"ok": False, "error": "invalid stream path"})
            return
        _, session_id, filename = parts
        if filename != M3U8_NAME and not filename.endswith(".ts"):
            self._send_json(400, {"ok": False, "error": "invalid stream file"})
            return
        stream_dir = os.path.abspath(os.path.join(engine.config.hls_root, session_id))
        file_path = os.path.abspath(os.path.join(stream_dir, filename))
        if not file_path.startswith(stream_dir + os.sep) or not os.path.isfile(file_path):
            self._send_json(404, {"ok": False, "error": "stream file not found"})
            return
        content_type = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t"
        with open(file_path, "rb") as f:
            self._send_bytes(200, f.read(), content_type)

    def _serve_video(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[2] != "output.mp4":
            self._send_json(404, {"ok": False, "error": "invalid video path"})
            return
        session_id = parts[1]
        video_dir = os.path.abspath(engine.config.video_save_root)
        file_path = os.path.abspath(os.path.join(video_dir, f"{session_id}.mp4"))
        if not file_path.startswith(video_dir + os.sep) or not os.path.isfile(file_path):
            self._send_json(404, {"ok": False, "error": "video file not found"})
            return
        file_size = int(os.path.getsize(file_path))
        start = 0
        end = max(0, file_size - 1)
        status = 200
        range_header = str(self.headers.get("Range") or "").strip()
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            raw_start, raw_end = match.groups()
            if raw_start:
                start = int(raw_start)
                end = min(end, int(raw_end)) if raw_end else end
            elif raw_end:
                suffix_length = min(file_size, int(raw_end))
                start = max(0, file_size - suffix_length)
            if start < 0 or start >= file_size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = 206

        content_length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(file_path)[0] or "video/mp4")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="liveact_{session_id}.mp4"',
        )
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        remaining = content_length
        with open(file_path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_args():
    parser = argparse.ArgumentParser(description="Dependency-free LiveAct avatar HTTP sidecar.")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--wav2vec_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--size", default="416*720")
    parser.add_argument("--hls_root", default="liveact_avatar/hls_output")
    parser.add_argument("--video_save_root", default="liveact_avatar/generated_videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--fp8_kv_cache", action="store_true")
    parser.add_argument("--no_fp8_gemm", action="store_true")
    parser.add_argument("--block_offload", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_vae_decode", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--preload_image_path",
        default=os.environ.get("LIVEACT_AVATAR_PRELOAD_IMAGE_PATH"),
    )
    parser.add_argument(
        "--preload_prompt",
        default=os.environ.get("LIVEACT_AVATAR_PRELOAD_PROMPT", DEFAULT_AVATAR_PROMPT),
    )
    parser.add_argument(
        "--session_ttl_sec",
        type=float,
        default=float(os.environ.get("LIVEACT_AVATAR_SESSION_TTL_SEC", "45")),
    )
    parser.add_argument(
        "--stop_join_timeout_sec",
        type=float,
        default=float(os.environ.get("LIVEACT_AVATAR_STOP_JOIN_TIMEOUT_SEC", "130")),
    )
    return parser.parse_args()


def main():
    global engine, control_pg, session_ttl_sec, stop_join_timeout_sec
    args = parse_args()
    config = LiveActEngineConfig(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        size=args.size,
        hls_root=args.hls_root,
        video_save_root=args.video_save_root,
        t5_cpu=bool(args.t5_cpu),
        fp8_kv_cache=bool(args.fp8_kv_cache),
        fp8_gemm=not bool(args.no_fp8_gemm),
        block_offload=bool(args.block_offload),
        compile_model=bool(args.compile_model),
        compile_vae_decode=bool(args.compile_vae_decode),
        seed=int(args.seed),
        warmup=bool(args.warmup),
        preload_image_path=(
            os.path.abspath(args.preload_image_path) if args.preload_image_path else None
        ),
        preload_prompt=str(args.preload_prompt or DEFAULT_AVATAR_PROMPT),
    )
    engine = LiveActStreamingEngine(config)
    session_ttl_sec = max(0.0, float(args.session_ttl_sec))
    stop_join_timeout_sec = max(1.0, float(args.stop_join_timeout_sec))
    if engine.world_size > 1:
        # Rank 1 normally waits in a blocking broadcast while no call is
        # active. PyTorch's default 30-minute collective timeout therefore
        # kills an otherwise healthy long-lived service after 30 idle minutes.
        # This control group carries tiny lifecycle/window objects, so a long
        # idle timeout is appropriate; GPU inference still has its own errors
        # and the HTTP session watchdog handles abandoned calls.
        control_timeout_sec = max(
            3600.0,
            float(os.environ.get("LIVEACT_AVATAR_CONTROL_TIMEOUT_SEC", "31536000")),
        )
        control_pg = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=control_timeout_sec),
        )

    if engine.rank == 0:
        watchdog = threading.Thread(
            target=_session_watchdog_loop,
            name="avatar_session_watchdog",
            daemon=True,
        )
        watchdog.start()
        httpd = ThreadingHTTPServer((args.host, int(args.port)), AvatarRequestHandler)
        print(
            f"[avatar] serving on {args.host}:{args.port}, "
            f"viewer=http://127.0.0.1:{args.port}/viewer, "
            f"world_size={engine.world_size}, size={args.size}, "
            f"session_ttl_sec={session_ttl_sec:.1f}, "
            "mode=single_active_audio_driven",
            flush=True,
        )
        httpd.serve_forever()
    else:
        print(f"[avatar] rank {engine.rank} waiting for rank0 commands", flush=True)
        worker_loop_nonzero_rank()


if __name__ == "__main__":
    main()
