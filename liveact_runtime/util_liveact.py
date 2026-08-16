import torch
import numpy as np
from PIL import Image
from einops import rearrange
import subprocess

def center_rescale_crop_keep_ratio(image, target_size):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    image = image.crop((left, top, right, bottom))
    return image


def get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device):
    indices = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_embedding.shape[0] - 1)
    audio_emb = audio_embedding[center_indices][None, ...].to(device)
    return audio_emb


def get_msk(frame_num, cond_image, vae_stride, device):
    h, w = cond_image.shape[-2], cond_image.shape[-1]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2).to(torch.bfloat16)  # B 4 T H W
    return msk

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu', fps=25):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * fps # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).to(device, audio_encoder.dtype)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    # audio_emb = audio_emb.cpu().detach()
    return audio_emb.detach()

def exec_cmd(cmd):
    return subprocess.run(cmd, shell=False, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def add_audio_to_video(silent_video_path: str, audio_video_path: str, output_video_path: str):
    cmd = [
        'ffmpeg',
        '-y',
        '-i', silent_video_path,
        '-i', audio_video_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        output_video_path
    ]

    try:
        exec_cmd(cmd)
        print(f"Video with audio generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")