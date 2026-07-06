import argparse
import asyncio
import base64
import json
import uuid

import numpy as np


async def _run(args):
    try:
        import websockets
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'websockets'. Install it in the environment used "
            "for this test client."
        ) from exc

    audio = np.zeros(int(args.chunk_size), dtype=np.float32)
    session_id = args.session_id or f"test-{uuid.uuid4().hex[:8]}"

    async with websockets.connect(args.url, open_timeout=args.timeout) as ws:
        for idx in range(int(args.chunks)):
            payload = {
                "type": "audio",
                "session_id": session_id,
                "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
            }
            await ws.send(json.dumps(payload, ensure_ascii=False))
            raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            msg = json.loads(raw)
            state = msg.get("state", {})
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "session_id": msg.get("session_id"),
                        "state": state.get("state"),
                        "asr_segment": state.get("asr_segment"),
                        "asr_buffer": state.get("asr_buffer"),
                    },
                    ensure_ascii=False,
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Smoke test SoulX-Duplug sidecar websocket prediction.")
    parser.add_argument("--url", default="ws://127.0.0.1:18080/turn")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=2560)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
