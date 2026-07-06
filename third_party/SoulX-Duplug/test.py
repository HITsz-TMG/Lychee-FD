import os
import json
import time
import base64
import websocket
import threading

import soxr
import numpy as np
import soundfile as sf


def recv_loop(ws, stop_event):
    while not stop_event.is_set():
        try:
            msg = ws.recv()
            if msg is None:
                continue

            data = json.loads(msg)
            print("[RECV]", data)

        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            print("[RECV ERROR]", e)
            break


def test(wav_path, api_url="ws://localhost:8000/turn", session_id="test-session-001"):

    ws = websocket.create_connection(api_url)
    ws.settimeout(1.0)

    stop_event = threading.Event()
    recv_thread = threading.Thread(target=recv_loop, args=(ws, stop_event), daemon=True)
    recv_thread.start()

    input_audio, sr = sf.read(wav_path)

    if sr != 16000:
        input_audio = soxr.resample(input_audio, sr, 16000)

    start_time = time.time()
    try:
        for i in range(50):
            audio = input_audio[i * 2560 : (i + 1) * 2560]

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            assert audio.dtype == np.float32

            payload = {
                "type": "audio",
                "session_id": session_id,
                "audio": base64.b64encode(audio.tobytes()).decode(),
            }

            ws.send(json.dumps(payload))
            time.sleep(0.1)

    finally:
        stop_event.set()
        recv_thread.join(timeout=1.0)
        ws.close()


if __name__ == "__main__":
    test(os.path.join(os.path.dirname(__file__), "assets/tmp.wav"))
