import librosa
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from typing import List


def _mel_filters(n_mels: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram."""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if n_mels == 128:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    else:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))

def load_audio(file_path, target_rate=16000, max_length=None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    If max_length is provided, truncate the audio to that length
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
    audio = waveform[0]  # get the first channel

    # Truncate audio if it exceeds max_length
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]

    return audio

def log_mel_spectrogram(audio, n_mels=128, padding=479, device=None):
    """
    Compute the log-Mel spectrogram with specific padding for StepAudio
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def compute_token_num(max_feature_len):
    # First, audio goes through encoder:
    # 1. conv1: kernel=3, stride=1, padding=1 -> size unchanged
    # 2. conv2: kernel=3, stride=2, padding=1 -> size/2
    # 3. avg_pooler: kernel=2, stride=2 -> size/2
    max_feature_len = max_feature_len - 2  # remove padding
    encoder_output_dim = (max_feature_len + 1) // 2 // 2  # after conv2 and avg_pooler

    # Then through adaptor (parameters from config file):
    padding = 1
    kernel_size = 3  # from config: audio_encoder_config.kernel_size
    stride = 2      # from config: audio_encoder_config.adapter_stride
    adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
    return adapter_output_dim

def padding_mels(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats, feats lengths
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1)-2 for s in sample],
                                dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats,
                                batch_first=True,
                                padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths