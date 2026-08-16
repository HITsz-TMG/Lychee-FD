import argparse
import time

import torch
import torchaudio
import torchaudio.transforms as T

from lychee_fd.avatar_bridge import RemoteAvatarClient


def wav_to_pcm_s16le_chunks(path: str, target_sr: int, chunk_ms: int):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        wav = T.Resample(int(sr), int(target_sr))(wav)
    wav = wav.squeeze(0).clamp(-1.0, 1.0)
    samples_per_chunk = max(1, int(target_sr * chunk_ms / 1000))
    for start in range(0, wav.numel(), samples_per_chunk):
        chunk = wav[start : start + samples_per_chunk]
        pcm = (chunk * 32767.0).to(torch.int16).cpu().numpy().astype("<i2").tobytes()
        yield pcm


def main():
    parser = argparse.ArgumentParser(description="Push a wav file to the LiveAct avatar sidecar as PCM chunks.")
    parser.add_argument("--avatar_url", default="http://127.0.0.1:8092")
    parser.add_argument("--session_id", default="demo")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--prompt", default="A person is speaking naturally, with stable facial expression.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--chunk_ms", type=int, default=400)
    parser.add_argument("--sleep", action="store_true", help="Sleep between pushes to simulate realtime.")
    parser.add_argument("--keep_open", action="store_true", help="Do not stop/finalize the avatar session.")
    args = parser.parse_args()

    client = RemoteAvatarClient(args.avatar_url, timeout_sec=300)
    start = client.start(
        session_id=args.session_id,
        image_path=args.image_path,
        prompt=args.prompt,
        fps=args.fps,
        sample_rate=args.sample_rate,
    )
    print("started:", start)
    chunks = list(wav_to_pcm_s16le_chunks(args.audio_path, args.sample_rate, args.chunk_ms))
    for idx, pcm in enumerate(chunks):
        result = client.push_pcm(
            session_id=args.session_id,
            pcm_bytes=pcm,
            sample_rate=args.sample_rate,
            is_last=(idx == len(chunks) - 1),
        )
        print(f"push {idx + 1}/{len(chunks)}:", result)
        if args.sleep:
            time.sleep(args.chunk_ms / 1000.0)
    print("stream:", start.get("stream_url"))
    if not args.keep_open:
        stop = client.stop(session_id=args.session_id)
        print("stopped:", stop)
        if stop.get("video_path"):
            print("video:", stop.get("video_path"))
        if stop.get("video_url"):
            print("video_url:", stop.get("video_url"))


if __name__ == "__main__":
    main()
