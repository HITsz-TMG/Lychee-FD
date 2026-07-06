import time
import uuid
import base64

import json
import websocket
import numpy as np


class TurnTaking:
    def __init__(
        self,
        server_url="ws://localhost:8000/turn",
        client_id=None,
        timeout=1.0,
    ):
        self.client_id = client_id or uuid.uuid4().hex
        self.timeout = timeout
        self.server_url = server_url

    def connect(self):
        self.ws = websocket.create_connection(self.server_url)
        self.ws.settimeout(self.timeout)

    def process(self, audio_chunk: np.ndarray):
        if audio_chunk is None:
            return None
        payload = {
            "type": "audio",
            "session_id": self.client_id,
            "audio": base64.b64encode(
                np.asarray(audio_chunk, dtype=np.float32).tobytes()
            ).decode(),
        }

        try:
            self.ws.send(json.dumps(payload))
            response = self.ws.recv()
            # print(response)
            data = json.loads(response)
        except:
            self.connect()
            self.ws.send(json.dumps(payload))
            response = self.ws.recv()
            data = json.loads(response)

        assert isinstance(data, dict), ValueError(
            f"Unexpected response type: {type(data)}"
        )

        if data["state"]["state"] == "idle":
            print("Silence detected")
        elif data["state"]["state"] == "nonidle":
            print("Voice detected")
            print(f"[ASR of last 3.2s]: {data['state'].get('asr_buffer', '')}")
            print(f"[ASR of current chunk]: {data['state'].get('asr_segment', '')}")
        elif data["state"]["state"] == "speak":
            print(f"User finished speaking, taking turn")
            print(f"User transcription: {data['state'].get('text', '')}")
        elif data["state"]["state"] == "blank":
            print(
                "Accumulated unprocessed audio less than one chunk (160ms), wait for next request"
            )
