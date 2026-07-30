from typing import Iterable, Optional, Tuple

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn
from transformers import PreTrainedModel, Qwen2Model
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_step_audio_2 import StepAudio2Config


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

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.
    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.
    1 for non-padded part and 0 for padded part.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks = [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
                    [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
                    [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask

class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).type(input.dtype)

class Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            None if self.bias is None else self.bias.to(input.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            input, weight.to(input.dtype), None if bias is None else bias.to(input.dtype)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x.contiguous()), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x.contiguous()))
        return x

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = LayerNorm(n_state)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2], x.dtype)  # (B, 1, T // 2)
        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask.unsqueeze(1))
            else:
                x = block(x, mask.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x.contiguous())
        return x, x_len

class Adaptor(nn.Module):
    def __init__(
        self,
        n_state: int = 1280,
        n_hidden: int = 3072,
        kernel_size: int = 7,
        stride: int = 4
    ):
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            # print("self.stride: {}".format(self.stride))
            self.conv = Conv1d(n_state, n_state, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(n_state, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, n_hidden)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        T = x.size(-1)
        if self.stride != -1:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.conv, x.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
            else:
                x = x.permute(0, 2, 1)
                x = F.gelu(self.conv(x))
                x = x.permute(0, 2, 1)
        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.linear1, x)
            x = torch.utils.checkpoint.checkpoint(self.relu, x)
            x = torch.utils.checkpoint.checkpoint(self.linear2, x)
        else:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
        return x

class StepAudio2ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StepAudio2Config
    main_input_name = "input_ids"
    # Important: Add this attribute to make HF recognize it as a model with generation capability
    # _keys_to_ignore_on_load_missing = ["lm_head.weight"]
    supports_gradient_checkpointing = True  # Declare gradient checkpointing support.
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: StepAudio2Config):
        super().__init__(config)
        if isinstance(config.torch_dtype, str):
            dtype = getattr(torch, config.torch_dtype)
        else:
            dtype = config.torch_dtype
        self.model = Qwen2Model(config.text_config)
        self.bf16 = dtype==torch.bfloat16
        self.encoder = AudioEncoder(
            config.audio_encoder_config.n_mels, config.audio_encoder_config.n_audio_ctx, config.audio_encoder_config.n_audio_state,
            config.audio_encoder_config.n_audio_head, config.audio_encoder_config.n_audio_layer
        )
        self.adapter = Adaptor(
            config.audio_encoder_config.n_audio_state, config.audio_encoder_config.llm_dim,
            config.audio_encoder_config.kernel_size, config.audio_encoder_config.adapter_stride
        )
        if self.bf16:
            self.encoder = self.encoder.bfloat16()
            self.adapter = self.adapter.bfloat16()
        self.lm_head = torch.nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=dtype
        )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        wavs=None,
        wav_lens=None,
        attention_mask=None,
        **kwargs
    ):
        hidden_states = self.model.embed_tokens(input_ids)
        if wavs is not None:
            if self.bf16:
                wavs = wavs.bfloat16()
            out, feat_lens = self.encoder(wavs, wav_lens)
            out = self.adapter(out)
            feat_lens = (feat_lens - 1) // 2 + 1
            insert_location = torch.nonzero(input_ids == 151688)
            insert_location[:,1] += 1
            for idx in range(len(insert_location)):
                i,s = insert_location[idx]
                hidden_states[i][s : s+feat_lens[idx]] = out[idx][:feat_lens[idx]]

        x = self.model(inputs_embeds=hidden_states, attention_mask=attention_mask)[0]
        logits = self.lm_head(x)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def get_input_embeddings(self):
        """Return the model's input embeddings - required for GenerationMixin"""
        return self.model.embed_tokens

    def get_output_embeddings(self):
        """Return the model's output embeddings (LM head) - required for GenerationMixin"""
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Prepare inputs for generation - required for GenerationMixin"""
        # Keep the wavs and wav_lens from the initial call
        wavs = kwargs.get("wavs", None)
        wav_lens = kwargs.get("wav_lens", None)

        # For generation steps after the first, we don't need to process audio again
        # because the audio tokens have already been replaced in the input sequence
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            # We're in a generation step, no need to process audio again
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": kwargs.get("past_key_values")
            }

        # First generation step, include audio processing
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "wavs": wavs,
            "wav_lens": wav_lens
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder the cache for beam search - required for GenerationMixin if using beam search"""
        # If you're not using past_key_values or beam search, this can be a simple pass-through
        # Otherwise implement according to your model's cache structure
        return past_key_values

    def _set_gradient_checkpointing(self, module, value=False):
        # For Qwen2Model
        if hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = value

            # Add the missing _gradient_checkpointing_func method to Qwen2Model
            # This is what Qwen2Model tries to use when gradient_checkpointing=True
            if value and not hasattr(self.model, '_gradient_checkpointing_func'):
                def _gradient_checkpointing_func(module_to_run, *args, **kwargs):
                    # This function wraps torch.utils.checkpoint.checkpoint
                    # and is used by Qwen2Model to perform checkpointing
                    return torch.utils.checkpoint.checkpoint(module_to_run, *args, **kwargs)

                self.model._gradient_checkpointing_func = _gradient_checkpointing_func

        # For custom encoder and adapter
        if hasattr(self.encoder, 'gradient_checkpointing'):
            self.encoder.gradient_checkpointing = value
        if hasattr(self.adapter, 'gradient_checkpointing'):
            self.adapter.gradient_checkpointing = value

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("step_audio_2", StepAudio2Config)
AutoModelForCausalLM.register(StepAudio2Config, StepAudio2ForCausalLM)
