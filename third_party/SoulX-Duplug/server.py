import os
import base64
import json
import time
import asyncio
from typing import Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from service.engine import TurnTakingEngine
from service.session import TurnSession
from service.model import load_turn_model

SESSION_TTL_SEC = 60  # Recycle if no audio for 60 seconds
GC_INTERVAL_SEC = 10  # Session GC interval
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/config.yaml")

# Global state
sessions: Dict[str, TurnSession] = {}


# FastAPI Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("SOULX_DUPLUG_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    print(f"[TurnTaking] loading model config={config_path} ...")
    app.state.model = load_turn_model(config_path=config_path)
    print("[TurnTaking] model loaded")

    gc_task = asyncio.create_task(session_gc_loop())

    yield

    gc_task.cancel()
    print("[TurnTaking] shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "sessions": len(sessions),
        "config_path": os.environ.get("SOULX_DUPLUG_CONFIG_PATH", DEFAULT_CONFIG_PATH),
    }


async def session_gc_loop():
    while True:
        now = time.time()
        expired = []

        for sid, sess in sessions.items():
            if now - sess.last_active_ts > SESSION_TTL_SEC:
                expired.append(sid)

        for sid in expired:
            print(f"[TurnTaking] GC session {sid}")
            del sessions[sid]

        await asyncio.sleep(GC_INTERVAL_SEC)


@app.websocket("/turn")
async def turn_ws(ws: WebSocket):
    await ws.accept()
    print("[TurnTaking] websocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") != "audio":
                continue

            session_id = data["session_id"]

            if session_id not in sessions:
                engine = TurnTakingEngine(model=ws.app.state.model)
                sessions[session_id] = TurnSession(engine)
                print(f"[TurnTaking] new session {session_id}")

            session = sessions[session_id]
            session.touch()  # Update timestamp

            try:
                audio = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
            except Exception:
                continue

            state = session.feed_audio(audio)

            if state is not None:
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "turn_state",
                            "session_id": session_id,
                            "state": state,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )

    except WebSocketDisconnect:
        print("[TurnTaking] websocket disconnected")

    except Exception as e:
        print(f"[TurnTaking] websocket error: {e}")
