import threading
import time
import unittest

from lychee_fd.avatar_bridge import RealtimeAvatarBridge


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class FakeAvatarClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = False
        self.push_calls = []
        self.abort_calls = []
        self.stop_calls = 0
        self.segment_id = 0
        self.iteration = 0
        self.stream_ready = False
        self.audio_mode = "standby"
        self.muxed_sec = 0.0

    def health(self):
        return {"ok": True}

    def start(self, **kwargs):
        with self.lock:
            self.started = True
        return {
            "ok": True,
            "session_id": kwargs["session_id"],
            "segment_id": self.segment_id,
            "stream_ready": False,
            "muxed_sec": self.muxed_sec,
            "segment_muxed_sec": 0.0,
        }

    def push_pcm(self, **kwargs):
        with self.lock:
            self.push_calls.append(dict(kwargs))
            if kwargs.get("pcm_bytes"):
                self.iteration = 1
                self.stream_ready = True
                self.audio_mode = "speech"
                self.muxed_sec = max(self.muxed_sec, 1.2)
            if kwargs.get("is_last"):
                self.audio_mode = "silence"
            return self._status_locked()

    def status(self, **_kwargs):
        with self.lock:
            return self._status_locked()

    def abort(self, **kwargs):
        with self.lock:
            self.abort_calls.append(dict(kwargs))
            self.audio_mode = "silence"
            return self._status_locked()

    def stop(self, **_kwargs):
        with self.lock:
            self.stop_calls += 1
        return {"ok": True, "closed": True}

    def heartbeat(self, **_kwargs):
        return {"ok": True}

    def _status_locked(self):
        return {
            "ok": True,
            "segment_id": self.segment_id,
            "iteration": self.iteration,
            "stream_ready": self.stream_ready,
            "input_finished": False,
            "audio_mode": self.audio_mode,
            "final_target_chunks": None,
            "inference_active": False,
            "queued_real_pcm_sec": 0.0,
            "muxed_sec": self.muxed_sec,
            "segment_muxed_sec": self.muxed_sec,
        }


class RealtimeAvatarBridgeTest(unittest.TestCase):
    def make_bridge(self, *, playback_buffer_sec=1.0):
        events = []
        client = FakeAvatarClient()
        bridge = RealtimeAvatarBridge(
            image_path="/tmp/avatar.png",
            prompt="conversation",
            avatar_url="http://unused",
            session_id="test-session",
            client=client,
            event_callback=events.append,
            status_poll_interval_sec=0.05,
            segment_drain_timeout_sec=10.0,
            playback_buffer_sec=playback_buffer_sec,
        )
        self.addCleanup(lambda: bridge.stop(join_timeout_sec=1.0))
        return bridge, client, events

    def test_pcm_is_streamed_and_empty_final_packet_switches_to_silence(self):
        bridge, client, events = self.make_bridge()
        bridge.start()
        self.assertTrue(wait_until(lambda: any(e.get("type") == "avatar_ready" for e in events)))

        self.assertTrue(
            bridge.submit_pcm(
                b"\x00\x00" * 240,
                generation_id=1,
                stream_id="tts-1",
            )
        )
        self.assertTrue(wait_until(lambda: len(client.push_calls) == 1))
        self.assertTrue(
            wait_until(lambda: any(e.get("type") == "avatar_stream_ready" for e in events))
        )

        self.assertTrue(
            bridge.submit_pcm(
                b"",
                is_last=True,
                generation_id=1,
                stream_id="tts-1",
            )
        )
        self.assertTrue(
            wait_until(lambda: any(e.get("type") == "avatar_audio_mode" for e in events))
        )
        self.assertEqual(len(client.push_calls), 2)
        self.assertEqual(client.push_calls[1]["pcm_bytes"], b"")
        self.assertTrue(client.push_calls[1]["is_last"])
        self.assertEqual(client.abort_calls, [])
        mode_event = next(e for e in events if e.get("type") == "avatar_audio_mode")
        self.assertEqual(mode_event["mode"], "silence")
        self.assertFalse(any(e.get("type") == "avatar_segment_complete" for e in events))

    def test_empty_non_final_packet_is_rejected(self):
        bridge, client, _events = self.make_bridge()
        self.assertFalse(bridge.submit_pcm(b"", is_last=False, generation_id=1))
        time.sleep(0.05)
        self.assertEqual(client.push_calls, [])

    def test_stream_ready_waits_for_playback_buffer(self):
        bridge, client, events = self.make_bridge(playback_buffer_sec=2.5)
        bridge.start()
        self.assertTrue(wait_until(lambda: client.started))
        self.assertTrue(
            bridge.submit_pcm(
                b"\x00\x00" * 240,
                generation_id=1,
                stream_id="tts-buffered",
            )
        )
        self.assertTrue(wait_until(lambda: len(client.push_calls) == 1))
        time.sleep(0.15)
        self.assertFalse(any(e.get("type") == "avatar_stream_ready" for e in events))

        with client.lock:
            client.muxed_sec = 2.6
        self.assertTrue(
            wait_until(lambda: any(e.get("type") == "avatar_stream_ready" for e in events))
        )
        ready = next(e for e in events if e.get("type") == "avatar_stream_ready")
        self.assertTrue(ready["playback_buffer_ready"])
        self.assertFalse(ready["final_response_ready"])

    def test_completed_short_response_waits_for_continuous_playback_buffer(self):
        bridge, client, events = self.make_bridge(playback_buffer_sec=2.5)
        bridge.start()
        self.assertTrue(wait_until(lambda: client.started))
        self.assertTrue(
            bridge.submit_pcm(
                b"\x00\x00" * 240,
                generation_id=1,
                stream_id="tts-short",
            )
        )
        self.assertTrue(wait_until(lambda: len(client.push_calls) == 1))
        self.assertTrue(
            bridge.submit_pcm(
                b"",
                is_last=True,
                generation_id=1,
                stream_id="tts-short",
            )
        )
        self.assertTrue(wait_until(lambda: len(client.push_calls) == 2))
        time.sleep(0.15)
        self.assertFalse(any(e.get("type") == "avatar_stream_ready" for e in events))
        with client.lock:
            # Continuous silence inference extends the same timeline after the
            # short reply until the player has a safe startup buffer.
            client.muxed_sec = 2.6
        self.assertTrue(
            wait_until(lambda: any(e.get("type") == "avatar_stream_ready" for e in events))
        )
        ready = next(e for e in events if e.get("type") == "avatar_stream_ready")
        self.assertTrue(ready["playback_buffer_ready"])
        self.assertFalse(ready["final_response_ready"])

    def test_abort_rejects_stale_generation_and_accepts_new_stream(self):
        bridge, client, events = self.make_bridge()
        bridge.start()
        self.assertTrue(wait_until(lambda: client.started))
        self.assertTrue(
            bridge.submit_pcm(b"\x01\x00", generation_id=1, stream_id="old")
        )
        self.assertTrue(wait_until(lambda: len(client.push_calls) == 1))
        bridge.submit_abort(reason="user_interrupt", generation_id=2)
        self.assertFalse(
            bridge.submit_pcm(b"\x01\x00", generation_id=1, stream_id="old")
        )
        self.assertTrue(
            bridge.submit_pcm(b"\x02\x00", generation_id=2, stream_id="new")
        )
        self.assertTrue(wait_until(lambda: len(client.abort_calls) >= 1))
        self.assertTrue(wait_until(lambda: len(client.push_calls) >= 2))
        self.assertEqual(client.push_calls[-1]["pcm_bytes"], b"\x02\x00")
        interrupt = next(e for e in events if e.get("type") == "avatar_speech_interrupted")
        self.assertEqual(interrupt["reason"], "user_interrupt")
        self.assertEqual(interrupt["interrupted_generation_id"], 1)
        self.assertEqual(interrupt["playback_end_sec"], 1.2)
        self.assertEqual(client.segment_id, 0)
        self.assertEqual(client.iteration, 1)


if __name__ == "__main__":
    unittest.main()
