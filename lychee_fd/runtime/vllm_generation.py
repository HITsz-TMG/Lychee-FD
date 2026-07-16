import argparse
import copy
import gc
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import datasets
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from pprint import pprint
from torch.utils.data import Dataset
from transformers import LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import AudioInput

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    IS_CUDA = False
except:
    IS_CUDA = True

SYSTEM_MESSAGE_PREFIX = "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"
SUPPORTED_FULL_DUPLEX_MODEL_TYPES = {
    "lychee_full_duplex",
    "step_audio_2_full_duplex",
}

# logits 处理器的快速路径:
# - 默认: 为降低时延，跳过逐 token 的整张量 sanitize 检查
# - 设置 LYCHEEFD_LOGITS_SANITIZE=1: 重新启用更安全的 sanitize 路径
_LYCHEEFD_LOGITS_SANITIZE = str(os.getenv("LYCHEEFD_LOGITS_SANITIZE", "0")).strip().lower() in {"1", "true", "yes", "on"}
_LYCHEEFD_DEBUG_CONTROL_LOGITS = str(
    os.getenv("LYCHEEFD_DEBUG_CONTROL_LOGITS", "0")
).strip().lower() in {"1", "true", "yes", "on"}


def _to_attr_config(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_attr_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_config(v) for v in value]
    return value


def _load_stepaudio_config(model_path: str):
    from transformers import AutoConfig

    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        if raw_config.get("model_type") in SUPPORTED_FULL_DUPLEX_MODEL_TYPES:
            return _to_attr_config(raw_config)
    except Exception:
        raw_config = None

    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        if raw_config is not None:
            return _to_attr_config(raw_config)
        with open(config_path, "r", encoding="utf-8") as f:
            return _to_attr_config(json.load(f))
_LYCHEEFD_VERBOSE_STREAM_LOG = str(
    os.getenv("LYCHEEFD_VERBOSE_STREAM_LOG", "0")
).strip().lower() in {"1", "true", "yes", "on"}
_LYCHEEFD_S2L_FILL_EOT_TTS_END = str(
    os.getenv("LYCHEEFD_S2L_FILL_EOT_TTS_END", "0")
).strip().lower() in {"1", "true", "yes", "on"}
_LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED = str(
    os.getenv("LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "on"}
_LYCHEEFD_CONTROL_EARLY_DEBUG = str(
    os.getenv("LYCHEEFD_CONTROL_EARLY_DEBUG", "0")
).strip().lower() in {"1", "true", "yes", "on"}
_LYCHEEFD_T2W_STRICT_STOKEN_RANGE = str(
    os.getenv("LYCHEEFD_T2W_STRICT_STOKEN_RANGE", "0")
).strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


_LYCHEEFD_T2W_CODEC_VOCAB_SIZE = max(
    1, _get_env_int("LYCHEEFD_T2W_CODEC_VOCAB_SIZE", 6561)
)


def _prepare_logits_scores(scores: torch.FloatTensor) -> torch.FloatTensor:
    if not _LYCHEEFD_LOGITS_SANITIZE:
        return scores
    scores = scores.float()
    if not torch.isfinite(scores).all():
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
    return scores

def compute_token_num(max_feature_len):
    # 先经过音频编码器:
    # 1. conv1: kernel=3, stride=1, padding=1 -> 尺寸不变
    # 2. conv2: kernel=3, stride=2, padding=1 -> 尺寸减半
    # 3. avg_pooler: kernel=2, stride=2 -> 尺寸再减半
    max_feature_len = max_feature_len - 2  # remove padding
    encoder_output_dim = (max_feature_len + 1) // 2 // 2  # after conv2 and avg_pooler
    
    # 再经过 adaptor（参数来自配置文件）:
    padding = 1
    kernel_size = 3  # from config: audio_encoder_config.kernel_size
    stride = 2      # from config: audio_encoder_config.adapter_stride
    adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
    return adapter_output_dim


def _mel_filters(n_mels: int) -> torch.Tensor:
    """加载 Mel 滤波器矩阵，用于将 STFT 投影为 Mel 频谱。"""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if n_mels == 128:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    else:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))


def log_mel_spectrogram(audio, n_mels=128, padding=479, device=None):
    """
    Compute the log-Mel spectrogram with Lychee-FD padding
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class ListeningControlLogitsProcessor(LogitsProcessor):
    _call_count = 0

    def __init__(
        self, 
        ss_token_id, 
        kl_token_id, 
        bc_token_id,
        vocab_size,
        start_speak_token_factor=1.2,
        bc_speak_token_factor=1.0,
        sleep_token_id=None,
        detect_token_id=None,
        prefix_input_len=None,
        control_token_chunk_size=None,
    ):
        self.start_speak_token_factor = start_speak_token_factor
        self.bc_speak_token_factor = bc_speak_token_factor
        self.ss_token_id = ss_token_id
        self.kl_token_id = kl_token_id
        self.bc_token_id = bc_token_id
        self.sleep_token_id = int(sleep_token_id) if sleep_token_id is not None else None
        self.detect_token_id = int(detect_token_id) if detect_token_id is not None else None
        self.prefix_input_len = (
            int(prefix_input_len) if prefix_input_len is not None else None
        )
        self.control_token_chunk_size = (
            int(control_token_chunk_size)
            if control_token_chunk_size is not None
            else None
        )
        self._allowed_tokens = [kl_token_id, ss_token_id]
        if bc_token_id is not None:
            self._allowed_tokens.append(bc_token_id)
        ListeningControlLogitsProcessor._call_count = 0
        self.last_probs = {
            "kl": None,
            "ss": None,
            "bc": None,
            "sl": None,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        scores = _prepare_logits_scores(scores)
        batch_size = int(scores.shape[0])
        next_call = ListeningControlLogitsProcessor._call_count + batch_size
        log_now = (next_call <= 5) or (next_call % 10 == 0)

        enforce_position_pattern = (
            self.control_token_chunk_size is not None
            and self.control_token_chunk_size > 0
            and self.sleep_token_id is not None
            and self.detect_token_id is not None
        )

        if enforce_position_pattern:
            # control processors receive generated_control_ids, which do not
            # include the text prefix, so do not subtract prefix_input_len.
            pos = input_ids.shape[1] % self.control_token_chunk_size
            if pos == self.control_token_chunk_size - 2:
                keep_val = scores[:, self.detect_token_id].clone()
                scores.fill_(float("-inf"))
                scores[:, self.detect_token_id] = keep_val
            elif pos == self.control_token_chunk_size - 1:
                keep_logits = scores[:, self._allowed_tokens].clone()
                scores.fill_(float("-inf"))
                scores[:, self._allowed_tokens] = keep_logits
                if self.start_speak_token_factor > 0:
                    scores[:, self.ss_token_id] *= self.start_speak_token_factor
                if self.bc_token_id is not None and self.bc_speak_token_factor > 0:
                    scores[:, self.bc_token_id] += self.bc_speak_token_factor
                try:
                    row = scores[0].float()
                    token_ids = [int(self.kl_token_id), int(self.ss_token_id)]
                    token_names = ["kl", "ss"]
                    if self.bc_token_id is not None:
                        token_ids.append(int(self.bc_token_id))
                        token_names.append("bc")
                    logits_vec = torch.stack([row[tid] for tid in token_ids], dim=0)
                    probs_vec = F.softmax(logits_vec, dim=0)
                    self.last_probs["kl"] = float(probs_vec[token_names.index("kl")].item())
                    self.last_probs["ss"] = float(probs_vec[token_names.index("ss")].item())
                    if "bc" in token_names:
                        self.last_probs["bc"] = float(probs_vec[token_names.index("bc")].item())
                    else:
                        self.last_probs["bc"] = None
                    self.last_probs["sl"] = None
                except Exception:
                    pass
            else:
                keep_val = scores[:, self.sleep_token_id].clone()
                scores.fill_(float("-inf"))
                scores[:, self.sleep_token_id] = keep_val
        else:
            keep_logits = scores[:, self._allowed_tokens].clone()
            scores.fill_(float("-inf"))
            scores[:, self._allowed_tokens] = keep_logits

            if self.start_speak_token_factor > 0:
                scores[:, self.ss_token_id] *= self.start_speak_token_factor
            if self.bc_token_id is not None and self.bc_speak_token_factor > 0:
                scores[:, self.bc_token_id] += self.bc_speak_token_factor

            # Persist latest control probabilities for downstream realtime UI/events.
            try:
                row = scores[0].float()
                token_ids = [int(self.kl_token_id), int(self.ss_token_id)]
                token_names = ["kl", "ss"]
                if self.bc_token_id is not None:
                    token_ids.append(int(self.bc_token_id))
                    token_names.append("bc")
                logits_vec = torch.stack([row[tid] for tid in token_ids], dim=0)
                probs_vec = F.softmax(logits_vec, dim=0)
                self.last_probs["kl"] = float(probs_vec[token_names.index("kl")].item())
                self.last_probs["ss"] = float(probs_vec[token_names.index("ss")].item())
                if "bc" in token_names:
                    self.last_probs["bc"] = float(probs_vec[token_names.index("bc")].item())
                else:
                    self.last_probs["bc"] = None
                self.last_probs["sl"] = None
            except Exception:
                # Keep previous values when current step is not numerically valid.
                pass

        ListeningControlLogitsProcessor._call_count = next_call
        if log_now and _LYCHEEFD_DEBUG_CONTROL_LOGITS:
            ss_raw = scores[0, self.ss_token_id].item()
            if self.start_speak_token_factor > 0:
                ss_raw = ss_raw / self.start_speak_token_factor
            kl_raw = scores[0, self.kl_token_id].item()
            if self.bc_token_id is not None:
                bc_raw = scores[0, self.bc_token_id].item() - (self.bc_speak_token_factor if self.bc_speak_token_factor > 0 else 0.0)
            else:
                bc_raw = float("-inf")
            ss_adj = scores[0, self.ss_token_id].item()
            kl_adj = scores[0, self.kl_token_id].item()
            valid_logits = torch.tensor([kl_adj, ss_adj])
            probs = F.softmax(valid_logits, dim=0)
            print(f"[LOGITS] call={ListeningControlLogitsProcessor._call_count} "
                    f"raw: SS={ss_raw:.3f} KL={kl_raw:.3f} BC={bc_raw:.3f} | "
                    f"adj: SS={ss_adj:.3f} KL={kl_adj:.3f} | "
                    f"prob: KL={probs[0]:.4f} SS={probs[1]:.4f} | "
                    f"ss_factor={self.start_speak_token_factor}")

        return scores

class SpeakingControlLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        sleep_token_id,
        detect_token_id,
        sl_token_id,
        ks_token_id,
        vocab_size,
        prefix_input_len,
        control_token_chunk_size,
        start_listen_token_factor=1.0,
    ):
        self.sleep_token_id = sleep_token_id
        self.detect_token_id = detect_token_id
        self.sl_token_id = sl_token_id
        self.ks_token_id = ks_token_id
        self.start_listen_token_factor = start_listen_token_factor

        self.prefix_input_len = prefix_input_len
        self.control_token_chunk_size = control_token_chunk_size
        self.last_probs = {
            "sl": None,
            "ss": None,
            "ks": None,
            "bc": None,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        scores = _prepare_logits_scores(scores)
        # control processors receive generated_control_ids, which do not
        # include the text prefix, so do not subtract prefix_input_len.
        pos = input_ids.shape[1] % self.control_token_chunk_size
        if pos == self.control_token_chunk_size - 2:
            keep_val = scores[:, self.detect_token_id].clone()
            scores.fill_(float("-inf"))
            scores[:, self.detect_token_id] = keep_val
        elif pos == self.control_token_chunk_size - 1:
            keep_vals = scores[:, [self.sl_token_id, self.ks_token_id]].clone()
            scores.fill_(float("-inf"))
            scores[:, self.sl_token_id] = keep_vals[:, 0]
            scores[:, self.ks_token_id] = keep_vals[:, 1]
            if self.start_listen_token_factor > 0:
                scores[:, self.sl_token_id] *= self.start_listen_token_factor
            try:
                row = scores[0].float()
                logits_vec = torch.stack(
                    [row[int(self.sl_token_id)], row[int(self.ks_token_id)]], dim=0
                )
                probs_vec = F.softmax(logits_vec, dim=0)
                self.last_probs["sl"] = float(probs_vec[0].item())
                self.last_probs["ks"] = float(probs_vec[1].item())
                self.last_probs["ss"] = None
                self.last_probs["bc"] = None
            except Exception:
                pass
        else:
            keep_val = scores[:, self.sleep_token_id].clone()
            scores.fill_(float("-inf"))
            scores[:, self.sleep_token_id] = keep_val

        return scores
class BackChannelLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        sleep_token_id,
        detect_token_id,
        sl_token_id,
        ss_token_id,
        bc_token_id,
        vocab_size,
        prefix_input_len,
        control_token_chunk_size,
        start_speak_token_factor=1.2,
    ):
        self.sleep_token_id = sleep_token_id
        self.detect_token_id = detect_token_id
        self.sl_token_id = sl_token_id
        self.ss_token_id = ss_token_id
        self.bc_token_id = bc_token_id
        self._speaking_allow_ids = [self.ss_token_id, self.sl_token_id]
        if self.bc_token_id is not None:
            self._speaking_allow_ids.append(self.bc_token_id)

        self.start_speak_token_factor = start_speak_token_factor
        self.prefix_input_len = prefix_input_len
        self.control_token_chunk_size = control_token_chunk_size
        self.last_probs = {
            "sl": None,
            "ss": None,
            "ks": None,
            "bc": None,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        scores = _prepare_logits_scores(scores)
        # control processors receive generated_control_ids, which do not
        # include the text prefix, so do not subtract prefix_input_len.
        pos = input_ids.shape[1] % self.control_token_chunk_size
        if pos == self.control_token_chunk_size - 2:
            keep_val = scores[:, self.detect_token_id].clone()
            scores.fill_(float("-inf"))
            scores[:, self.detect_token_id] = keep_val
        elif pos == self.control_token_chunk_size - 1:
            keep_vals = scores[:, self._speaking_allow_ids].clone()
            scores.fill_(float("-inf"))
            scores[:, self._speaking_allow_ids] = keep_vals
            scores[:, self.ss_token_id] *= self.start_speak_token_factor
            try:
                row = scores[0].float()
                token_ids = [int(x) for x in self._speaking_allow_ids]
                logits_vec = torch.stack([row[tid] for tid in token_ids], dim=0)
                probs_vec = F.softmax(logits_vec, dim=0)
                prob_map = {}
                for idx, tid in enumerate(token_ids):
                    prob_map[int(tid)] = float(probs_vec[idx].item())
                self.last_probs["sl"] = prob_map.get(int(self.sl_token_id))
                self.last_probs["ss"] = prob_map.get(int(self.ss_token_id))
                self.last_probs["bc"] = (
                    prob_map.get(int(self.bc_token_id))
                    if self.bc_token_id is not None
                    else None
                )
                self.last_probs["ks"] = None
            except Exception:
                pass
        else:
            keep_val = scores[:, self.sleep_token_id].clone()
            scores.fill_(float("-inf"))
            scores[:, self.sleep_token_id] = keep_val

        return scores


class FixedTokenLogitsProcessor(LogitsProcessor):
    """
    Force each decoding step to emit a single fixed token id.
    Used in listening/recheck one-step control calls so text/stoken heads
    stay deterministic (pad) and can safely reuse vLLM active requests.
    """

    def __init__(self, token_id: int):
        self.token_id = int(token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = _prepare_logits_scores(scores)
        keep_val = scores[:, self.token_id].clone()
        scores.fill_(float("-inf"))
        scores[:, self.token_id] = keep_val
        return scores


class SpeakingLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        text_start,
        text_end,
        eos_token_id,
        vocab_size,
        text_pad_token_id=None,
        has_eos=False,
        end_speak_token_factor=1,
        max_token_length=None,
        prefix_token_id=None,
        ngram_size=None,
    ):
        self.vocab_size = vocab_size
        self.text_start = text_start
        self.text_end = text_end
        self.text_pad_token_id = text_pad_token_id

        self.prefix_token_id = prefix_token_id
        self.prefix_cnt = 0

        self.has_eos = has_eos
        self.eos_token_id = eos_token_id
        self.end_speak_token_factor = end_speak_token_factor
        self.cnt = 0
        self.max_token_length = max_token_length

        if ngram_size is not None:
            self.ngram_size_processor = NoRepeatNGramLogitsProcessor(
                ngram_size=ngram_size,
            )
        else:
            self.ngram_size_processor = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        scores = _prepare_logits_scores(scores)
        last_token_ids = input_ids[:, -1]

        if self.prefix_token_id is not None:
            self.prefix_cnt += 1
            if self.prefix_cnt <= len(self.prefix_token_id):
                token_id = int(self.prefix_token_id[self.prefix_cnt - 1])
                keep_vals = scores[:, token_id].clone()
                scores.fill_(float("-inf"))
                scores[:, token_id] = keep_vals
                return scores

        if self.max_token_length is not None and not self.has_eos:
            self.cnt += 1
            if self.cnt >= self.max_token_length:
                keep_vals = scores[:, self.eos_token_id].clone()
                scores.fill_(float("-inf"))
                scores[:, self.eos_token_id] = keep_vals
                return scores

        if bool((last_token_ids == self.eos_token_id).any().item()):
            self.has_eos = True

        if self.has_eos and self.text_pad_token_id is not None:
            keep_val = scores[:, self.text_pad_token_id].clone()
            scores.fill_(float("-inf"))
            scores[:, self.text_pad_token_id] = keep_val
        else:
            if self.text_start is None or self.text_end is None:
                scores[:, self.eos_token_id] *= self.end_speak_token_factor
            else:
                keep_span = scores[:, self.text_start:self.text_end].clone()
                eos_val = scores[:, self.eos_token_id].clone()
                scores.fill_(float("-inf"))
                scores[:, self.text_start:self.text_end] = keep_span
                scores[:, self.eos_token_id] = eos_val * self.end_speak_token_factor

        if self.ngram_size_processor is not None:
            scores = self.ngram_size_processor(input_ids, scores)

        return scores
class SingleTurnGenerationFramework:
    """
    Shared realtime state-machine base for the vLLM runtime.

    This class intentionally does not load a HuggingFace model in the public
    release. HF realtime inference is implemented by
    lychee_fd.runtime.hf_v9_realtime.HFRealtimeV9GenerationFramework, while
    vLLM inference uses VLLMGenerationFramework below.
    """

    TOKENS_PER_SECOND = 25

    MAX_SPEECH_TOKEN_NUM = 1000

    MAX_TEXT_TOKEN_NUM = 128

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SingleTurnGenerationFramework in vllm_generation.py is a shared "
            "state-machine base and must not be instantiated directly. Use "
            "VLLMGenerationFramework for vLLM inference, or "
            "HFRealtimeV9GenerationFramework from hf_v9_realtime.py for HF "
            "realtime inference."
        )

    def init_speaking_processor(self, end_speak_token_factor):
        speaking_text_processor = LogitsProcessorList([
            SpeakingLogitsProcessor(
                text_start=None, 
                text_end=None, 
                text_pad_token_id=self.tts_pad_id,
                eos_token_id=self.eos_token_id,
                vocab_size=self.model.config.text_config.vocab_size,
                has_eos=False,
                end_speak_token_factor=end_speak_token_factor,
                max_token_length=self.MAX_TEXT_TOKEN_NUM
            )
        ])

        speaking_stoken_processor = LogitsProcessorList([
            SpeakingLogitsProcessor(
                text_start=self.stoken_audio_start_id,
                text_end=self.stoken_audio_end_id,
                text_pad_token_id=None,
                eos_token_id=self.tts_end_id,
                vocab_size=self.model.config.text_config.vocab_size,
                has_eos=False,
                prefix_token_id=[self.stoken_delay_token_id] * self.stoken_delay_num + [self.tts_start_id],
                ngram_size=self.stoken_no_repeat_n_gram,
                max_token_length=self.MAX_SPEECH_TOKEN_NUM
            ),
        ])

        return speaking_text_processor, speaking_stoken_processor

    def init_listening_pad_processor(self):
        fixed_text_processor = LogitsProcessorList([
            FixedTokenLogitsProcessor(self.text_pad_token_id),
        ])
        fixed_stoken_processor = LogitsProcessorList([
            FixedTokenLogitsProcessor(self.stoken_pad_token_id),
        ])
        return fixed_text_processor, fixed_stoken_processor

    @staticmethod
    def _set_last_token_safe(seq: torch.Tensor, token_id: int) -> torch.Tensor:
        """
        Avoid in-place writes on inference tensors returned by inference_mode paths
        (e.g. vLLM wrappers). Clone first, then mutate.
        """
        out = seq.clone()
        out[0, -1] = token_id
        return out

    def _apply_s2l_tail_fallback(
        self,
        generated_input_ids: torch.Tensor,
        generated_stoken_ids: torch.Tensor,
    ):
        """
        S->L fallback policy.
        - default: legacy behavior, both tails are padded
        - env on: conditionally close non-pad tails with EOT/TTS_END
        """
        if _LYCHEEFD_S2L_FILL_EOT_TTS_END:
            last_text_token = int(generated_input_ids[0, -1].item())
            last_stoken_token = int(generated_stoken_ids[0, -1].item())
            text_tail_token = (
                self.eos_token_id
                if last_text_token not in (self.tts_pad_id, self.text_pad_token_id)
                else self.text_pad_token_id
            )
            stoken_tail_token = (
                self.tts_end_id
                if last_stoken_token != self.stoken_pad_token_id
                else self.stoken_pad_token_id
            )
        else:
            text_tail_token = self.text_pad_token_id
            stoken_tail_token = self.stoken_pad_token_id

        new_text_ids = self._set_last_token_safe(generated_input_ids, text_tail_token)
        new_stoken_ids = self._set_last_token_safe(generated_stoken_ids, stoken_tail_token)
        return new_text_ids, new_stoken_ids

    def stepaudio_audio_prepreocess(
        self,
        audio = None,
        debug=False,
    ) -> BatchFeature:

        feats = []
        feats_lengths = []
        input_ids = []
        # for i in range(0, audio.shape[0], 16000 * self.window_second):
        window_samples = int(16000 * self.window_second)
        for i in range(0, audio.shape[0], window_samples):
            mel = log_mel_spectrogram(audio[i:i+window_samples], n_mels=128, padding=479, device=self.device if IS_CUDA else 'cpu')
            feats.append(mel.t())
            feats_lengths.append(mel.size(1)-2)
            # if debug:
            #     assert audio[i:i+16000*self.window_second].shape[0] % (self.AUDIO_TOKEN_N_SAMPLE // 2) == 0
            #     input_ids += [self.audio_pad_token_id] * (audio[i:i+16000*self.window_second].shape[0] // (self.AUDIO_TOKEN_N_SAMPLE // 2))
            # else:
            if self.align_audio_input:
                input_ids += [self.audio_token_id, self.audio_pad_token_id] * compute_token_num(mel.shape[1])
            else:
                input_ids += [self.audio_token_id] * compute_token_num(mel.shape[1])

        return {"input_ids": input_ids, "feats": feats, "feats_lengths": feats_lengths}

    def full_chunk_stream_offline_generation(
        self, 
        audio, 
        prefix=None, 
        initial_listening_state='l', 
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
    ):
        prefix_input_ids = self.tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
        system_input_length = len(prefix_input_ids)

        if prefix is not None:
            input_ids = torch.tensor(prefix_input_ids + prefix['input_ids'], dtype=torch.long, device=self.model.device).unsqueeze(0)
            control_input_ids = torch.tensor(prefix["control_input_ids"], dtype=torch.long, device=self.model.device).unsqueeze(0)
            stoken_ids = torch.tensor(prefix["stoken_ids"], dtype=torch.long, device=self.model.device).unsqueeze(0)
        else:   
            input_ids = torch.tensor(prefix_input_ids + [self.text_pad_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
            control_input_ids = torch.tensor([self.sleep_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
            stoken_ids = torch.tensor([self.stoken_pad_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
        prefix_input_ids = torch.tensor(prefix_input_ids, dtype=torch.long, device=self.model.device).unsqueeze(0)
        stoken_mapping = torch.tensor([-1] * (prefix_input_ids.shape[1] + stoken_ids.shape[1]), dtype=torch.long, device=self.model.device).unsqueeze(0)

        model_inputs = self.stepaudio_audio_prepreocess(audio)
        feats, feats_lengths = model_inputs["feats"], model_inputs["feats_lengths"]
        wavs = torch.nn.utils.rnn.pad_sequence(
            feats,
            batch_first=True,
            padding_value=0).transpose(1, 2).to(self.device)
        wav_lens = torch.tensor(feats_lengths, dtype=torch.int32).to(self.device)
        
        audio_input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0).to(self.device)

        past_key_values = None
        stoken_past_key_values = None
        control_past_key_values = None
        listening_state = initial_listening_state
        
        total_len = len(model_inputs["input_ids"])
        chunk_start = math.ceil((input_ids.shape[1] - system_input_length + 1) / self.control_token_chunk_size) * self.control_token_chunk_size
        chunk_end = math.ceil(total_len / self.control_token_chunk_size) * self.control_token_chunk_size

        generated_input_ids = input_ids
        generated_stoken_ids = stoken_ids
        generated_control_ids = control_input_ids
        speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
        profile_latency = str(os.getenv("LYCHEEFD_PROFILE_LATENCY", "0")).strip().lower() in {"1", "true", "yes", "on"}
        try:
            profile_every = max(1, int(os.getenv("LYCHEEFD_PROFILE_EVERY", "20")))
        except (TypeError, ValueError):
            profile_every = 20
        profile_model = {
            "l_calls": 0,
            "l_sec": 0.0,
            "b_calls": 0,
            "b_sec": 0.0,
            "s_calls": 0,
            "s_sec": 0.0,
            "bc_recheck_calls": 0,
            "bc_recheck_sec": 0.0,
            "s_tokens": 0,
            "b_tokens": 0,
        }
        if profile_latency:
            print(f"[LAT_MODEL] enabled (every={profile_every})")

        for cn_e in range(chunk_start, chunk_end + 1, self.control_token_chunk_size):
            
            target_length = min(cn_e, total_len)
            current_len = generated_input_ids.shape[1] - system_input_length
            target_new_length = target_length - current_len

            if listening_state == 'l':
                # Text: Pad, Stoken: Pad, Control: Sleep...Detect
                listening_control_processor = LogitsProcessorList([
                    ListeningControlLogitsProcessor(
                        ss_token_id=self.ss_token_id,
                        kl_token_id=self.kl_token_id,
                        bc_token_id=self.bc_token_id if self.allowing_backchannel else None,
                        vocab_size=self.model.config.text_config.vocab_size,
                        start_speak_token_factor=start_speak_token_factor,
                        bc_speak_token_factor=bc_speak_token_factor,
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                    )
                ])
                listening_text_processor, listening_stoken_processor = self.init_listening_pad_processor()
                
                stoken_mapping = torch.cat(
                    [stoken_mapping, torch.full((1, target_new_length), -1, dtype=torch.long, device=self.device)],
                    dim=1,
                )

                outputs = self.model.multi_head_generate(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=listening_control_processor,
                    logits_processor=listening_text_processor,
                    stoken_logits_processor=listening_stoken_processor,
                    temperature=1.0,
                    top_k=0,
                    eos_token_id=None,                 
                    pad_token_id=None,
                )
                past_key_values = outputs["past_key_values"]
                stoken_past_key_values = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                generated_input_ids = outputs["sequences"]
                generated_stoken_ids = outputs["stoken_ids"]
                generated_control_ids = outputs["control_ids"]
                
                pred_control_token = generated_control_ids[0, -1]

                if pred_control_token == self.ss_token_id:
                    print(f"[L]->[S] {cn_e}/{chunk_end}")
                    listening_state = "s"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                elif pred_control_token == self.bc_token_id:
                    print(f"[L]->[BC] {cn_e}/{chunk_end}")
                    listening_state = "b"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                else:
                    assert pred_control_token == self.kl_token_id or cn_e == chunk_end

            elif listening_state == 'b':
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        stoken_mapping = torch.cat([stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.device)], dim=1)
                if stoken_mapping.shape[1] - generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + target_new_length):
                        seq_l = p  - self.last_ss_pos - (self.stoken_delay_num + 1) 
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                    stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)
    
                speaking_control_processor = LogitsProcessorList([
                    BackChannelLogitsProcessor(
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        sl_token_id=self.sl_token_id, 
                        ss_token_id=self.ss_token_id, 
                        bc_token_id=self.bc_token_id,
                        vocab_size=self.model.config.text_config.vocab_size,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                    )
                ])

                outputs = self.model.multi_head_generate(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor, 
                    logits_processor=speaking_text_processor, 
                    stoken_logits_processor=speaking_stoken_processor, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id
                )

                past_key_values         = outputs["past_key_values"]
                stoken_past_key_values  = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                generated_input_ids     = outputs["sequences"]
                generated_stoken_ids    = outputs["stoken_ids"]
                generated_control_ids   = outputs["control_ids"]

                
                if generated_control_ids[0, -1] == self.sl_token_id:
                    print(f"[BC]->[L] {cn_e}/{chunk_end}")
                    listening_state = 'l'
                    self.last_ss_pos = None
                    generated_stoken_ids = self._set_last_token_safe(generated_stoken_ids, self.stoken_pad_token_id)
                    generated_input_ids = self._set_last_token_safe(generated_input_ids, self.text_pad_token_id)
                elif generated_control_ids[0, -1] == self.ss_token_id:
                    print(f"[BC]->[S] {cn_e}/{chunk_end}")
                    listening_state = 's'
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                    generated_stoken_ids = self._set_last_token_safe(generated_stoken_ids, self.stoken_pad_token_id)
                    generated_input_ids = self._set_last_token_safe(generated_input_ids, self.text_pad_token_id)
                elif generated_control_ids[0, -1] == self.bc_token_id or cn_e == chunk_end:
                    pass
                else:
                    assert generated_stoken_ids[0,-1] == self.tts_end_id
                    print(f"[BC]->[L] {cn_e}/{chunk_end}: stop talking...")
                    if target_length > generated_input_ids.shape[1] - system_input_length:
                        padding_len = target_length - generated_input_ids.shape[1] + system_input_length
                        padding_input_ids   = self.text_pad_token_id     * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids  = self.stoken_pad_token_id   * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_control_ids = self.sleep_token_id        * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.detect_token_id
                        padding_control_ids[:, -1] = self.sl_token_id

                        generated_input_ids     = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                        generated_stoken_ids    = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)
                        generated_control_ids   = torch.cat((generated_control_ids, padding_control_ids), dim=1)

                        listening_state = 'l'
                        self.last_ss_pos = None

                    else: 
                        print(f"[BC] {cn_e}/{chunk_end}: EOS at chunk end, starting a new chunk....", end='')
                        padding_len = self.control_token_chunk_size

                        stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.stoken_delay_num + 1:
                            stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(padding_len, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
                        if stoken_mapping.shape[1] - generated_input_ids.shape[1] < padding_len:
                            padding_stoken_mapping = []
                            for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + padding_len):
                                seq_l = p  - self.last_ss_pos - (self.stoken_delay_num + 1) 
                                padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                            padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                            stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)

                        padding_input_ids       = self.text_pad_token_id     * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids      = self.stoken_pad_token_id   * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_control_ids     = self.sleep_token_id        * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.detect_token_id

                        generated_input_ids     = torch.cat((generated_input_ids, padding_input_ids[:,:-1]), dim=1)
                        generated_stoken_ids    = torch.cat((generated_stoken_ids, padding_stoken_ids[:,:-1]), dim=1)
                        generated_control_ids   = torch.cat((generated_control_ids, padding_control_ids[:,:-1]), dim=1)

                        speaking_control_processor = LogitsProcessorList([
                            BackChannelLogitsProcessor(
                                sleep_token_id=self.sleep_token_id,
                                detect_token_id=self.detect_token_id,
                                sl_token_id=self.sl_token_id, 
                                ss_token_id=self.ss_token_id, 
                                bc_token_id=None,
                                vocab_size=self.model.config.text_config.vocab_size,
                                prefix_input_len=prefix_input_ids.shape[1],
                                control_token_chunk_size=self.control_token_chunk_size,
                                start_speak_token_factor=start_speak_token_factor,
                            )
                        ])

                        outputs = self.model.multi_head_generate(
                            input_ids=generated_input_ids,
                            stoken_ids=generated_stoken_ids,
                            control_input_ids=generated_control_ids,
                            prefix_input_ids=prefix_input_ids,
                            stoken_mapping=stoken_mapping,
                            audio_input_ids=audio_input_ids,
                            wavs=wavs,
                            wav_lens=wav_lens,
                            past_key_values=past_key_values,
                            stoken_past_key_values=stoken_past_key_values,
                            control_past_key_values=control_past_key_values,
                            use_cache=self.use_cache and not self.adding_text_hiddenstates,
                            max_new_tokens=1,
                            control_logits_processor=speaking_control_processor, 
                            logits_processor=None, 
                            stoken_logits_processor=None, 
                        )

                        past_key_values         = outputs["past_key_values"]
                        stoken_past_key_values  = outputs["stoken_past_key_values"]
                        control_past_key_values = outputs["control_past_key_values"]
                        
                        generated_control_ids   = outputs["control_ids"]

                        pred_control_token = generated_control_ids[0,-1]
                        if pred_control_token == self.sl_token_id:
                            print("[BC]->[L]")
                            listening_state = 'l'
                            self.last_ss_pos = None
                        else:
                            assert generated_control_ids[0, -1] == self.ss_token_id
                            print("[BC]->[S]")
                            listening_state = 's'
                            speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                            self.last_ss_pos = generated_input_ids.shape[1] + 1
                        padding_input_ids       = self.text_pad_token_id     * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids      = self.stoken_pad_token_id   * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        generated_input_ids     = torch.cat((generated_input_ids, padding_input_ids[:,:-1]), dim=1)
                        generated_stoken_ids    = torch.cat((generated_stoken_ids, padding_stoken_ids[:,:-1]), dim=1)

            else:
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        stoken_mapping = torch.cat([stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.device)], dim=1)
                if stoken_mapping.shape[1] - generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + target_new_length):
                        seq_l = p  - self.last_ss_pos - (self.stoken_delay_num + 1) 
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                    stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)
    
                speaking_control_processor = LogitsProcessorList([
                    SpeakingControlLogitsProcessor(
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        sl_token_id=self.sl_token_id, 
                        ks_token_id=self.ks_token_id, 
                        vocab_size=self.model.config.text_config.vocab_size,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                        start_listen_token_factor=start_listen_token_factor,
                    )
                ])

                outputs = self.model.multi_head_generate(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor, 
                    logits_processor=speaking_text_processor, 
                    stoken_logits_processor=speaking_stoken_processor, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id
                )

                past_key_values         = outputs["past_key_values"]
                stoken_past_key_values  = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                generated_input_ids     = outputs["sequences"]
                generated_stoken_ids    = outputs["stoken_ids"]
                generated_control_ids   = outputs["control_ids"]

                if generated_stoken_ids[0,-1] == self.tts_end_id:
                    print("stop talking...")
                    if target_length > generated_input_ids.shape[1] - system_input_length:
                        padding_len = target_length - generated_input_ids.shape[1] + system_input_length
                    else: 
                        print("EOS at chunk end, starting a new chunk....")
                        padding_len = self.control_token_chunk_size

                        stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.stoken_delay_num + 1:
                            stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(padding_len, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
                        if stoken_mapping.shape[1] - generated_input_ids.shape[1] < padding_len:
                            padding_stoken_mapping = []
                            for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + padding_len):
                                seq_l = p  - self.last_ss_pos - (self.stoken_delay_num + 1) 
                                padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                            padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                            stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)

                    padding_input_ids       = self.text_pad_token_id     * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    padding_stoken_ids      = self.stoken_pad_token_id   * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    padding_control_ids     = self.sleep_token_id        * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    
                    if padding_control_ids.shape[1] > 2:
                        padding_control_ids[:, -2] = self.detect_token_id
                    padding_control_ids[:, -1] = self.sl_token_id

                    generated_input_ids     = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                    generated_stoken_ids    = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)
                    generated_control_ids   = torch.cat((generated_control_ids, padding_control_ids), dim=1)

                    listening_state = 'l'
                    self.last_ss_pos = None
                else:
                    pred_control_token = generated_control_ids[0, -1]
                    
                    if pred_control_token == self.sl_token_id:
                        print(f"[S]->[L] {cn_e}/{chunk_end}")
                        listening_state = 'l'
                        self.last_ss_pos = None
                        generated_input_ids, generated_stoken_ids = self._apply_s2l_tail_fallback(
                            generated_input_ids,
                            generated_stoken_ids,
                        )
                    else:
                        assert pred_control_token == self.ks_token_id or cn_e == chunk_end, f"pred_control_token:{pred_control_token}, {self.tokenizer.decode(pred_control_token)}"

        generated_input_ids = generated_input_ids[:, system_input_length:][0].cpu().tolist()
        generated_control_ids = generated_control_ids[0].cpu().tolist()
        generated_stoken_ids = generated_stoken_ids[0].cpu().tolist()

        assert len(generated_input_ids) == len(generated_control_ids) == len(generated_stoken_ids)
        
        return self.decoder(generated_input_ids, generated_stoken_ids, generated_control_ids)


    def full_chunk_stream_generation(
        self,
        audio,
        prefix=None,
        initial_listening_state='l',
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
    ):
        """
        Streaming generator version of full_chunk_stream_offline_generation.
        Yields incremental events as they happen instead of blocking until the
        entire audio is processed.

        Event types yielded:
          - {"type": "control_decision", "state": str, "chunk": int, "token": int}
          - {"type": "state_change", "from": str, "to": str, "pos": int, "chunk": int}
          - {"type": "speaking_token", "text_token": int, "stoken": int, "control": int, "step": int}
          - {"type": "speaking_done", "reason": str, "chunk": int}
          - {"type": "chunk_start", "chunk_idx": int, "chunk_pos": int, "total_chunks": int}
          - {"type": "chunk_end", "chunk_idx": int}
          - {"type": "generation_complete", "text_ids": list, "stoken_ids": list, "control_ids": list}
        """
        prefix_input_ids = self.tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
        system_input_length = len(prefix_input_ids)

        if prefix is not None:
            input_ids = torch.tensor(prefix_input_ids + prefix['input_ids'], dtype=torch.long, device=self.model.device).unsqueeze(0)
            control_input_ids = torch.tensor(prefix["control_input_ids"], dtype=torch.long, device=self.model.device).unsqueeze(0)
            stoken_ids = torch.tensor(prefix["stoken_ids"], dtype=torch.long, device=self.model.device).unsqueeze(0)
        else:
            input_ids = torch.tensor(prefix_input_ids + [self.text_pad_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
            control_input_ids = torch.tensor([self.sleep_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
            stoken_ids = torch.tensor([self.stoken_pad_token_id], dtype=torch.long, device=self.model.device).unsqueeze(0)
        prefix_input_ids = torch.tensor(prefix_input_ids, dtype=torch.long, device=self.model.device).unsqueeze(0)
        stoken_mapping = torch.tensor([-1] * (prefix_input_ids.shape[1] + stoken_ids.shape[1]), dtype=torch.long, device=self.model.device).unsqueeze(0)

        model_inputs = self.stepaudio_audio_prepreocess(audio)
        feats, feats_lengths = model_inputs["feats"], model_inputs["feats_lengths"]
        wavs = torch.nn.utils.rnn.pad_sequence(
            feats, batch_first=True, padding_value=0).transpose(1, 2).to(self.device)
        wav_lens = torch.tensor(feats_lengths, dtype=torch.int32).to(self.device)

        audio_input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0).to(self.device)

        past_key_values = None
        stoken_past_key_values = None
        control_past_key_values = None
        listening_state = initial_listening_state

        total_len = len(model_inputs["input_ids"])
        chunk_start = math.ceil((input_ids.shape[1] - system_input_length + 1) / self.control_token_chunk_size) * self.control_token_chunk_size
        chunk_end = math.ceil(total_len / self.control_token_chunk_size) * self.control_token_chunk_size

        generated_input_ids = input_ids
        generated_stoken_ids = stoken_ids
        generated_control_ids = control_input_ids
        speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
        profile_latency = str(os.getenv("LYCHEEFD_PROFILE_LATENCY", "0")).strip().lower() in {"1", "true", "yes", "on"}
        try:
            profile_every = max(1, int(os.getenv("LYCHEEFD_PROFILE_EVERY", "20")))
        except (TypeError, ValueError):
            profile_every = 20
        profile_model = {
            "l_calls": 0,
            "l_sec": 0.0,
            "b_calls": 0,
            "b_sec": 0.0,
            "s_calls": 0,
            "s_sec": 0.0,
            "bc_recheck_calls": 0,
            "bc_recheck_sec": 0.0,
            "s_tokens": 0,
            "b_tokens": 0,
        }
        if profile_latency:
            print(f"[LAT_MODEL] enabled (every={profile_every})")
        is_vllm_backend = (
            self.model.__class__.__name__ == "_VLLMModelAdapter"
            and hasattr(self.model, "_engine")
        )
        keep_alive_env = str(os.getenv("LYCHEEFD_VLLM_KEEP_ALIVE_SPEAKING", "1")).strip().lower()
        keep_alive_listen_env = str(os.getenv("LYCHEEFD_VLLM_KEEP_ALIVE_LISTENING", "1")).strip().lower()
        keep_alive_for_speaking = (
            is_vllm_backend
            and keep_alive_env not in {"0", "false", "no", "off"}
        )
        keep_alive_for_listening = (
            is_vllm_backend
            and keep_alive_listen_env not in {"0", "false", "no", "off"}
        )

        chunk_idx = 0
        total_chunks = (chunk_end - chunk_start) // self.control_token_chunk_size + 1
        latest_sl_prob = 0.0
        latest_ss_prob = 0.0
        latest_ks_prob = 0.0

        def _safe_prob(proc_obj, key):
            nonlocal latest_sl_prob, latest_ss_prob, latest_ks_prob
            cached = None
            if key == "sl":
                cached = latest_sl_prob
            elif key == "ss":
                cached = latest_ss_prob
            elif key == "ks":
                cached = latest_ks_prob

            if proc_obj is None:
                return cached
            probs = getattr(proc_obj, "last_probs", None)
            if not isinstance(probs, dict):
                return cached
            value = probs.get(key, None)
            if value is None:
                return cached
            try:
                fv = float(value)
            except (TypeError, ValueError):
                return cached
            if not math.isfinite(fv):
                return cached
            if key == "sl":
                latest_sl_prob = fv
            elif key == "ss":
                latest_ss_prob = fv
            elif key == "ks":
                latest_ks_prob = fv
            return fv

        if _LYCHEEFD_VERBOSE_STREAM_LOG:
            print(f"[STREAM_GEN] total_audio_tokens={total_len}, chunk_start={chunk_start}, "
                  f"chunk_end={chunk_end}, chunk_size={self.control_token_chunk_size}, "
                  f"total_chunks={total_chunks}, system_len={system_input_length}, "
                  f"input_ids_shape={input_ids.shape}")

        for cn_e in range(chunk_start, chunk_end + 1, self.control_token_chunk_size):
            target_length = min(cn_e, total_len)
            current_len = generated_input_ids.shape[1] - system_input_length
            target_new_length = target_length - current_len
            if _LYCHEEFD_VERBOSE_STREAM_LOG:
                print(f"[STREAM_GEN] chunk cn_e={cn_e}, target_len={target_length}, "
                      f"current_len={current_len}, new_len={target_new_length}, state={listening_state}")

            yield {"type": "chunk_start", "chunk_idx": chunk_idx, "chunk_pos": cn_e, "total_chunks": total_chunks}

            if listening_state == 'l':
                listening_control_processor = LogitsProcessorList([
                    ListeningControlLogitsProcessor(
                        ss_token_id=self.ss_token_id,
                        kl_token_id=self.kl_token_id,
                        bc_token_id=self.bc_token_id if self.allowing_backchannel else None,
                        vocab_size=self.model.config.text_config.vocab_size,
                        start_speak_token_factor=start_speak_token_factor,
                        bc_speak_token_factor=bc_speak_token_factor,
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                    )
                ])
                listening_control_core = listening_control_processor[0] if len(listening_control_processor) > 0 else None
                listening_text_processor, listening_stoken_processor = self.init_listening_pad_processor()
                stoken_mapping = torch.cat(
                    [stoken_mapping, torch.full((1, target_new_length), -1, dtype=torch.long, device=self.device)],
                    dim=1,
                )

                # Use streaming generate for listening decision (1 token)
                last_result = None
                t_model_l_start = time.perf_counter()
                for token_result in self.model.multi_head_generate_stream(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=listening_control_processor,
                    logits_processor=listening_text_processor,
                    stoken_logits_processor=listening_stoken_processor,
                    temperature=1.0,
                    top_k=0,
                    eos_token_id=None,
                    keep_request_alive=keep_alive_for_listening,
                ):
                    last_result = token_result
                t_model_l_cost = time.perf_counter() - t_model_l_start
                profile_model["l_calls"] += 1
                profile_model["l_sec"] += t_model_l_cost
                if profile_latency and (
                    profile_model["l_calls"] == 1
                    or (profile_model["l_calls"] % profile_every == 0)
                ):
                    step_cnt = int(last_result["step"] + 1) if isinstance(last_result, dict) and "step" in last_result else 0
                    print(
                        f"[LAT_MODEL][L] chunk={cn_e} cost={t_model_l_cost:.4f}s steps={step_cnt} "
                        f"avg={profile_model['l_sec']/max(1, profile_model['l_calls']):.4f}s"
                    )

                if last_result is not None:
                    past_key_values = last_result["past_key_values"]
                    stoken_past_key_values = last_result["stoken_past_key_values"]
                    control_past_key_values = last_result["control_past_key_values"]
                    generated_input_ids = last_result["sequences"]
                    generated_stoken_ids = last_result["stoken_ids"]
                    generated_control_ids = last_result["control_ids"]
                    pred_control_token = generated_control_ids[0, -1]
                else:
                    padding_input_ids = self.text_pad_token_id * torch.ones(
                        (generated_input_ids.shape[0], 1),
                        dtype=generated_input_ids.dtype,
                        device=generated_input_ids.device,
                    )
                    padding_stoken_ids = self.stoken_pad_token_id * torch.ones(
                        (generated_input_ids.shape[0], 1),
                        dtype=generated_input_ids.dtype,
                        device=generated_input_ids.device,
                    )
                    padding_control_input_ids = self.kl_token_id * torch.ones(
                        (generated_input_ids.shape[0], 1),
                        dtype=generated_control_ids.dtype,
                        device=generated_control_ids.device,
                    )
                    generated_input_ids = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                    generated_stoken_ids = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)
                    generated_control_ids = torch.cat((generated_control_ids, padding_control_input_ids), dim=1)
                    pred_control_token = self.kl_token_id
                yield {
                    "type": "control_decision",
                    "state": "l",
                    "chunk": cn_e,
                    "token": int(pred_control_token),
                    "ss_prob": _safe_prob(listening_control_core, "ss"),
                    "sl_prob": _safe_prob(listening_control_core, "sl"),
                }

                if pred_control_token == self.ss_token_id:
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "early_exit": bool(_LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED),
                        "interrupt": False,
                        "ss_prob": _safe_prob(listening_control_core, "ss"),
                        "sl_prob": _safe_prob(listening_control_core, "sl"),
                    }
                    listening_state = "s"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                elif pred_control_token == self.bc_token_id:
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "B",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_bc",
                        "ss_prob": _safe_prob(listening_control_core, "ss"),
                        "sl_prob": _safe_prob(listening_control_core, "sl"),
                    }
                    listening_state = "b"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]

            elif listening_state == 'b':
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        stoken_mapping = torch.cat([stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.device)], dim=1)
                if stoken_mapping.shape[1] - generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + target_new_length):
                        seq_l = p - self.last_ss_pos - (self.stoken_delay_num + 1)
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                    stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)

                speaking_control_processor = LogitsProcessorList([
                    BackChannelLogitsProcessor(
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        sl_token_id=self.sl_token_id,
                        ss_token_id=self.ss_token_id,
                        bc_token_id=self.bc_token_id,
                        vocab_size=self.model.config.text_config.vocab_size,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None

                last_result = None
                t_model_b_start = time.perf_counter()
                for token_result in self.model.multi_head_generate_stream(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor,
                    logits_processor=speaking_text_processor,
                    stoken_logits_processor=speaking_stoken_processor,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id,
                    keep_request_alive=keep_alive_for_speaking,
                ):
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    last_result = token_result
                t_model_b_cost = time.perf_counter() - t_model_b_start
                profile_model["b_calls"] += 1
                profile_model["b_sec"] += t_model_b_cost

                if last_result is None:
                    last_result = {
                        "sequences": generated_input_ids,
                        "stoken_ids": generated_stoken_ids,
                        "control_ids": generated_control_ids,
                        "past_key_values": past_key_values,
                        "stoken_past_key_values": stoken_past_key_values,
                        "control_past_key_values": control_past_key_values,
                    }
                b_step_cnt = int(last_result["step"] + 1) if "step" in last_result else 0
                profile_model["b_tokens"] += max(0, b_step_cnt)
                if profile_latency and (
                    profile_model["b_calls"] == 1
                    or (profile_model["b_calls"] % profile_every == 0)
                ):
                    avg_b = profile_model["b_sec"] / max(1, profile_model["b_calls"])
                    avg_b_t = profile_model["b_sec"] / max(1, profile_model["b_tokens"])
                    print(
                        f"[LAT_MODEL][B] chunk={cn_e} cost={t_model_b_cost:.4f}s steps={b_step_cnt} "
                        f"avg_call={avg_b:.4f}s avg_token={avg_b_t:.4f}s"
                    )

                past_key_values = last_result["past_key_values"]
                stoken_past_key_values = last_result["stoken_past_key_values"]
                control_past_key_values = last_result["control_past_key_values"]
                generated_input_ids = last_result["sequences"]
                generated_stoken_ids = last_result["stoken_ids"]
                generated_control_ids = last_result["control_ids"]

                if generated_control_ids[0, -1] == self.sl_token_id:
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "L",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_sl",
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    listening_state = 'l'
                    self.last_ss_pos = None
                    generated_stoken_ids = self._set_last_token_safe(generated_stoken_ids, self.stoken_pad_token_id)
                    generated_input_ids = self._set_last_token_safe(generated_input_ids, self.text_pad_token_id)
                elif generated_control_ids[0, -1] == self.ss_token_id:
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    listening_state = 's'
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                    generated_stoken_ids = self._set_last_token_safe(generated_stoken_ids, self.stoken_pad_token_id)
                    generated_input_ids = self._set_last_token_safe(generated_input_ids, self.text_pad_token_id)
                elif generated_control_ids[0, -1] == self.bc_token_id or cn_e == chunk_end:
                    pass
                else:
                    assert generated_stoken_ids[0, -1] == self.tts_end_id
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > generated_input_ids.shape[1] - system_input_length:
                        padding_len = target_length - generated_input_ids.shape[1] + system_input_length
                        padding_input_ids = self.text_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids = self.stoken_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_control_ids = self.sleep_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.detect_token_id
                        padding_control_ids[:, -1] = self.sl_token_id
                        generated_input_ids = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                        generated_stoken_ids = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)
                        generated_control_ids = torch.cat((generated_control_ids, padding_control_ids), dim=1)
                        listening_state = 'l'
                        self.last_ss_pos = None
                    else:
                        padding_len = self.control_token_chunk_size
                        stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.stoken_delay_num + 1:
                            stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(padding_len, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
                        if stoken_mapping.shape[1] - generated_input_ids.shape[1] < padding_len:
                            _pad_map = []
                            for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + padding_len):
                                seq_l = p - self.last_ss_pos - (self.stoken_delay_num + 1)
                                _pad_map.append(seq_l // 4 + self.last_ss_pos)
                            _pad_map = torch.tensor(_pad_map, dtype=torch.long, device=self.model.device).unsqueeze(0)
                            stoken_mapping = torch.cat((stoken_mapping, _pad_map), dim=1)

                        padding_input_ids = self.text_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids = self.stoken_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_control_ids = self.sleep_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.detect_token_id

                        generated_input_ids = torch.cat((generated_input_ids, padding_input_ids[:, :-1]), dim=1)
                        generated_stoken_ids = torch.cat((generated_stoken_ids, padding_stoken_ids[:, :-1]), dim=1)
                        generated_control_ids = torch.cat((generated_control_ids, padding_control_ids[:, :-1]), dim=1)

                        bc_ctrl_proc = LogitsProcessorList([
                            BackChannelLogitsProcessor(
                                sleep_token_id=self.sleep_token_id,
                                detect_token_id=self.detect_token_id,
                                sl_token_id=self.sl_token_id,
                                ss_token_id=self.ss_token_id,
                                bc_token_id=None,
                                vocab_size=self.model.config.text_config.vocab_size,
                                prefix_input_len=prefix_input_ids.shape[1],
                                control_token_chunk_size=self.control_token_chunk_size,
                                start_speak_token_factor=start_speak_token_factor,
                            )
                        ])
                        bc_ctrl_core = bc_ctrl_proc[0] if len(bc_ctrl_proc) > 0 else None
                        bc_fixed_text_processor, bc_fixed_stoken_processor = self.init_listening_pad_processor()

                        last_result = None
                        t_model_bc_start = time.perf_counter()
                        for token_result in self.model.multi_head_generate_stream(
                            input_ids=generated_input_ids,
                            stoken_ids=generated_stoken_ids,
                            control_input_ids=generated_control_ids,
                            prefix_input_ids=prefix_input_ids,
                            stoken_mapping=stoken_mapping,
                            audio_input_ids=audio_input_ids,
                            wavs=wavs,
                            wav_lens=wav_lens,
                            past_key_values=past_key_values,
                            stoken_past_key_values=stoken_past_key_values,
                            control_past_key_values=control_past_key_values,
                            use_cache=self.use_cache and not self.adding_text_hiddenstates,
                            max_new_tokens=1,
                            control_logits_processor=bc_ctrl_proc,
                            logits_processor=bc_fixed_text_processor,
                            stoken_logits_processor=bc_fixed_stoken_processor,
                            keep_request_alive=keep_alive_for_listening,
                        ):
                            last_result = token_result
                        t_model_bc_cost = time.perf_counter() - t_model_bc_start
                        profile_model["bc_recheck_calls"] += 1
                        profile_model["bc_recheck_sec"] += t_model_bc_cost
                        if profile_latency and (
                            profile_model["bc_recheck_calls"] == 1
                            or (profile_model["bc_recheck_calls"] % profile_every == 0)
                        ):
                            print(
                                f"[LAT_MODEL][BC_RECHECK] chunk={cn_e} cost={t_model_bc_cost:.4f}s "
                                f"avg={profile_model['bc_recheck_sec']/max(1, profile_model['bc_recheck_calls']):.4f}s"
                            )

                        if last_result is None:
                            last_result = {
                                "sequences": generated_input_ids,
                                "stoken_ids": generated_stoken_ids,
                                "control_ids": generated_control_ids,
                                "past_key_values": past_key_values,
                                "stoken_past_key_values": stoken_past_key_values,
                                "control_past_key_values": control_past_key_values,
                            }

                        past_key_values = last_result["past_key_values"]
                        stoken_past_key_values = last_result["stoken_past_key_values"]
                        control_past_key_values = last_result["control_past_key_values"]
                        generated_input_ids = last_result["sequences"]
                        generated_stoken_ids = last_result["stoken_ids"]
                        generated_control_ids = last_result["control_ids"]

                        pred_control_token = generated_control_ids[0, -1]
                        if pred_control_token == self.sl_token_id:
                            yield {
                                "type": "state_change",
                                "from": "B",
                                "to": "L",
                                "pos": cn_e,
                                "chunk": cn_e,
                                "reason": "bc_recheck_sl",
                                "sl_prob": _safe_prob(bc_ctrl_core, "sl"),
                                "ss_prob": _safe_prob(bc_ctrl_core, "ss"),
                            }
                            listening_state = 'l'
                            self.last_ss_pos = None
                        else:
                            yield {
                                "type": "state_change",
                                "from": "B",
                                "to": "S",
                                "pos": cn_e,
                                "chunk": cn_e,
                                "reason": "bc_recheck_ss",
                                "sl_prob": _safe_prob(bc_ctrl_core, "sl"),
                                "ss_prob": _safe_prob(bc_ctrl_core, "ss"),
                            }
                            listening_state = 's'
                            speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                            self.last_ss_pos = generated_input_ids.shape[1]

            else:
                # Speaking state 's'
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        stoken_mapping = torch.cat([stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.device)], dim=1)
                if stoken_mapping.shape[1] - generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + target_new_length):
                        seq_l = p - self.last_ss_pos - (self.stoken_delay_num + 1)
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(padding_stoken_mapping, dtype=torch.long, device=self.model.device).unsqueeze(0)
                    stoken_mapping = torch.cat((stoken_mapping, padding_stoken_mapping), dim=1)

                speaking_control_processor = LogitsProcessorList([
                    SpeakingControlLogitsProcessor(
                        sleep_token_id=self.sleep_token_id,
                        detect_token_id=self.detect_token_id,
                        sl_token_id=self.sl_token_id,
                        ks_token_id=self.ks_token_id,
                        vocab_size=self.model.config.text_config.vocab_size,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                        start_listen_token_factor=start_listen_token_factor,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None

                last_result = None
                t_model_s_start = time.perf_counter()
                for token_result in self.model.multi_head_generate_stream(
                    input_ids=generated_input_ids,
                    stoken_ids=generated_stoken_ids,
                    control_input_ids=generated_control_ids,
                    prefix_input_ids=prefix_input_ids,
                    stoken_mapping=stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    past_key_values=past_key_values,
                    stoken_past_key_values=stoken_past_key_values,
                    control_past_key_values=control_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor,
                    logits_processor=speaking_text_processor,
                    stoken_logits_processor=speaking_stoken_processor,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id,
                    keep_request_alive=keep_alive_for_speaking,
                ):
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    last_result = token_result
                t_model_s_cost = time.perf_counter() - t_model_s_start
                profile_model["s_calls"] += 1
                profile_model["s_sec"] += t_model_s_cost

                if last_result is None:
                    last_result = {
                        "sequences": generated_input_ids,
                        "stoken_ids": generated_stoken_ids,
                        "control_ids": generated_control_ids,
                        "past_key_values": past_key_values,
                        "stoken_past_key_values": stoken_past_key_values,
                        "control_past_key_values": control_past_key_values,
                    }
                s_step_cnt = int(last_result["step"] + 1) if "step" in last_result else 0
                profile_model["s_tokens"] += max(0, s_step_cnt)
                if profile_latency and (
                    profile_model["s_calls"] == 1
                    or (profile_model["s_calls"] % profile_every == 0)
                ):
                    avg_s = profile_model["s_sec"] / max(1, profile_model["s_calls"])
                    avg_s_t = profile_model["s_sec"] / max(1, profile_model["s_tokens"])
                    print(
                        f"[LAT_MODEL][S] chunk={cn_e} cost={t_model_s_cost:.4f}s steps={s_step_cnt} "
                        f"avg_call={avg_s:.4f}s avg_token={avg_s_t:.4f}s"
                    )

                past_key_values = last_result["past_key_values"]
                stoken_past_key_values = last_result["stoken_past_key_values"]
                control_past_key_values = last_result["control_past_key_values"]
                generated_input_ids = last_result["sequences"]
                generated_stoken_ids = last_result["stoken_ids"]
                generated_control_ids = last_result["control_ids"]

                if generated_stoken_ids[0, -1] == self.tts_end_id:
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > generated_input_ids.shape[1] - system_input_length:
                        padding_len = target_length - generated_input_ids.shape[1] + system_input_length
                    else:
                        padding_len = self.control_token_chunk_size
                        stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.stoken_delay_num + 1:
                            stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(padding_len, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
                        if stoken_mapping.shape[1] - generated_input_ids.shape[1] < padding_len:
                            _pad_map = []
                            for p in range(stoken_mapping.shape[1], generated_input_ids.shape[1] + padding_len):
                                seq_l = p - self.last_ss_pos - (self.stoken_delay_num + 1)
                                _pad_map.append(seq_l // 4 + self.last_ss_pos)
                            _pad_map = torch.tensor(_pad_map, dtype=torch.long, device=self.model.device).unsqueeze(0)
                            stoken_mapping = torch.cat((stoken_mapping, _pad_map), dim=1)

                    padding_input_ids = self.text_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    padding_stoken_ids = self.stoken_pad_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    padding_control_ids = self.sleep_token_id * torch.ones((generated_input_ids.shape[0], padding_len), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    if padding_control_ids.shape[1] > 2:
                        padding_control_ids[:, -2] = self.detect_token_id
                    padding_control_ids[:, -1] = self.sl_token_id

                    generated_input_ids = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                    generated_stoken_ids = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)
                    generated_control_ids = torch.cat((generated_control_ids, padding_control_ids), dim=1)
                    listening_state = 'l'
                    self.last_ss_pos = None
                else:
                    pred_control_token = generated_control_ids[0, -1]
                    if pred_control_token == self.sl_token_id:
                        yield {
                            "type": "state_change",
                            "from": "S",
                            "to": "L",
                            "pos": cn_e,
                            "chunk": cn_e,
                            "reason": "control_sl",
                            "sl_prob": _safe_prob(speaking_control_core, "sl"),
                            "ss_prob": _safe_prob(speaking_control_core, "ss"),
                            "ks_prob": _safe_prob(speaking_control_core, "ks"),
                        }
                        listening_state = 'l'
                        self.last_ss_pos = None
                        generated_input_ids, generated_stoken_ids = self._apply_s2l_tail_fallback(
                            generated_input_ids,
                            generated_stoken_ids,
                        )

            yield {"type": "chunk_end", "chunk_idx": chunk_idx}
            chunk_idx += 1

        final_text = generated_input_ids[:, system_input_length:][0].cpu().tolist()
        final_control = generated_control_ids[0].cpu().tolist()
        final_stoken = generated_stoken_ids[0].cpu().tolist()
        if profile_latency:
            print(
                "[LAT_MODEL][TOTAL] "
                f"L={profile_model['l_sec']:.4f}s/{int(profile_model['l_calls'])}calls "
                f"B={profile_model['b_sec']:.4f}s/{int(profile_model['b_calls'])}calls/{int(profile_model['b_tokens'])}tokens "
                f"S={profile_model['s_sec']:.4f}s/{int(profile_model['s_calls'])}calls/{int(profile_model['s_tokens'])}tokens "
                f"BC_RECHECK={profile_model['bc_recheck_sec']:.4f}s/{int(profile_model['bc_recheck_calls'])}calls "
                f"avg_S_token={profile_model['s_sec']/max(1, profile_model['s_tokens']):.4f}s "
                f"avg_B_token={profile_model['b_sec']/max(1, profile_model['b_tokens']):.4f}s"
            )

        yield {
            "type": "generation_complete",
            "text_ids": final_text,
            "stoken_ids": final_stoken,
            "control_ids": final_control,
        }

    def create_incremental_stream_session(
        self,
        prefix=None,
        initial_listening_state='l',
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        """
        Create a persistent streaming session that can be advanced with
        progressively longer realtime audio snapshots without reinitializing
        model-side generation state (KV caches / generated ids / chunk cursor).
        """
        return IncrementalChunkStreamSession(
            framework=self,
            prefix=prefix,
            initial_listening_state=initial_listening_state,
            start_speak_token_factor=start_speak_token_factor,
            start_listen_token_factor=start_listen_token_factor,
            bc_speak_token_factor=bc_speak_token_factor,
            end_speak_token_factor=end_speak_token_factor,
            audio_incremental_mode=audio_incremental_mode,
        )

    def create_hf_incremental_stream_session(
        self,
        prefix=None,
        initial_listening_state='l',
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        """
        Create a persistent streaming session with HF-style semantics:
        no keep-alive request reuse across speaking decode calls.
        """
        return HFIncrementalChunkStreamSession(
            framework=self,
            prefix=prefix,
            initial_listening_state=initial_listening_state,
            start_speak_token_factor=start_speak_token_factor,
            start_listen_token_factor=start_listen_token_factor,
            bc_speak_token_factor=bc_speak_token_factor,
            end_speak_token_factor=end_speak_token_factor,
            audio_incremental_mode=audio_incremental_mode,
        )

    def decoder(self, text, stoken, control):
        all_events = []

        listening_state = "l"
        text_seq = []
        stoken_seq = []
        utterance_start_pos = None

        def collection(event_type='response'):
            output_text_tokens = [i for i in text_seq if i < 151688 and i not in [self.text_pad_token_id, self.tts_pad_id]]
            output_audio_tokens = [i - 151696 for i in stoken_seq if i > 151695 and i not in [self.stoken_delay_token_id, self.stoken_pad_token_id]]
            output_text = self.tokenizer.decode(output_text_tokens, skip_special_tokens=True)
            return {
                "start_time": round(utterance_start_pos / self.TOKENS_PER_SECOND, 3),
                "start_pos": utterance_start_pos,
                "end_time": round(utterance_end_pos / self.TOKENS_PER_SECOND, 3) if utterance_end_pos is not None else -1,
                "end_pos": utterance_end_pos  if utterance_end_pos is not None else -1,
                "text": output_text,
                "audio": output_audio_tokens,
                "tokens": output_text_tokens,
                "type": event_type
            }


        for pos, (token_id, stoken_id, control_token_id) in enumerate(zip(text, stoken, control)):
            if control_token_id == self.bc_token_id:
                if listening_state == 'l':
                    utterance_start_pos = pos
                    listening_state = 'b'
                else:
                    assert listening_state == 'b'
                    text_seq.append(token_id)
                    stoken_seq.append(stoken_id)
            elif control_token_id == self.ss_token_id:
                if listening_state == 'b':
                    utterance_end_pos = pos
                    all_events.append(collection(event_type='backchannel'))
                    text_seq = []
                    stoken_seq = []   
                else:
                    assert listening_state == 'l'
                listening_state = 's'
                utterance_start_pos = pos
            elif control_token_id == self.sl_token_id:
                assert listening_state == 'b' or listening_state == 's'
                listening_state = 'l'
                utterance_end_pos = pos
                all_events.append(collection(event_type='backchannel' if listening_state == 'b' else 'response'))
                text_seq = []
                stoken_seq = []                   
            elif listening_state == 's' or listening_state == 'b':
                text_seq.append(token_id)
                stoken_seq.append(stoken_id)
        
        if listening_state == 's' or listening_state == 'b':
            utterance_end_pos = None
            all_events.append(collection(event_type='backchannel' if listening_state == 'b' else 'response'))
    
        return all_events

class _VLLMModelAdapter:
    """
    Adapter that wraps the vLLM engine to expose the same interface as
    the HF full-duplex model for the inherited framework methods.

    Provides:
      - .device
      - .config  (the HF config)
      - .multi_head_generate_stream(**kwargs) -> generator
      - .multi_head_generate(**kwargs) -> dict
    """

    def __init__(self, vllm_engine, config, device):
        self._engine = vllm_engine
        self.config = config
        self.device = device
        self.stream_generation_flag = True
        self._cached_audio_key = None
        self._cached_audio_context = None
        # vLLM engine runtime and forward_context are not thread-safe across
        # concurrent streaming generators; serialize per adapter instance.
        self._stream_lock = threading.RLock()

    @staticmethod
    def _tensor_sig(tensor):
        return (
            int(id(tensor)),
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )

    def _resolve_audio_embeds(self, wavs, wav_lens):
        if wavs is None or wav_lens is None:
            return None

        audio_key = (self._tensor_sig(wavs), self._tensor_sig(wav_lens))
        if audio_key != self._cached_audio_key or self._cached_audio_context is None:
            audio_embeds, audio_feat_lens = self._engine.precompute_audio(wavs, wav_lens)
            self._cached_audio_key = audio_key
            self._cached_audio_context = (audio_embeds, audio_feat_lens)

        return self._cached_audio_context

    def truncate_active_request_to_sequences(self, **kwargs):
        truncate_fn = getattr(self._engine, "truncate_active_request_to_sequences", None)
        if not callable(truncate_fn):
            return {"ok": False, "reason": "vllm_truncate_api_unavailable"}
        return truncate_fn(**kwargs)

    def close_active_request(self):
        self._cached_audio_key = None
        self._cached_audio_context = None
        close_fn = getattr(self._engine, "close_active_request", None)
        if callable(close_fn):
            return close_fn()
        return {"ok": True, "request_id": None, "closed": False}

    def multi_head_generate_stream(self, **kwargs):
        wavs = kwargs.get("wavs", None)
        wav_lens = kwargs.get("wav_lens", None)
        step_kwargs = {
            "input_ids": kwargs["input_ids"],
            "stoken_ids": kwargs["stoken_ids"],
            "control_ids": kwargs["control_input_ids"],
            "prefix_input_ids": kwargs["prefix_input_ids"],
            "audio_input_ids": kwargs["audio_input_ids"],
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "sampling_params": kwargs.get("sampling_params", None),
            "text_processors": kwargs.get("logits_processor", None),
            "stoken_processors": kwargs.get("stoken_logits_processor", None),
            "control_processors": kwargs.get("control_logits_processor", None),
            "eos_token_id": kwargs.get("eos_token_id", None),
            "stoken_eos_token_id": kwargs.get("stoken_eos_token_id", None),
            "temperature": kwargs.get("temperature", 1.0),
            "top_k": kwargs.get("top_k", 0),
            "top_p": kwargs.get("top_p", 1.0),
            "keep_request_alive": kwargs.get("keep_request_alive", False),
            "control_early_callback": kwargs.get("control_early_callback", None),
            "wavs": wavs,
            "wav_lens": wav_lens,
            "audio_pad_token_id": getattr(self.config, "audio_pad_token_id", None),
        }

        def _locked_stream():
            with self._stream_lock:
                audio_embeds = kwargs.get("audio_embeds", None)
                if audio_embeds is None:
                    audio_embeds = self._resolve_audio_embeds(wavs, wav_lens)
                step_kwargs["audio_embeds"] = audio_embeds
                for result in self._engine.step_generate_stream(**step_kwargs):
                    yield result

        return _locked_stream()

    def multi_head_generate(self, **kwargs):
        result = None
        for result in self.multi_head_generate_stream(**kwargs):
            pass
        if result is None:
            return {
                "sequences": kwargs["input_ids"],
                "stoken_ids": kwargs["stoken_ids"],
                "control_ids": kwargs["control_input_ids"],
                "past_key_values": None,
                "stoken_past_key_values": None,
                "control_past_key_values": None,
            }
        return {
            "sequences": result["sequences"],
            "stoken_ids": result["stoken_ids"],
            "control_ids": result["control_ids"],
            "past_key_values": None,
            "stoken_past_key_values": None,
            "control_past_key_values": None,
        }


class VLLMGenerationFramework(SingleTurnGenerationFramework):
    """
    vLLM realtime inference framework.

    The state machine logic (listening/speaking/backchannel) is inherited
    from SingleTurnGenerationFramework, but model loading is handled here via
    LycheeVLLMEngine. Public HF realtime inference lives in hf_v9_realtime.py.

    Usage:
        framework = VLLMGenerationFramework(
            model_type="FD",
            model_path="/path/to/checkpoint",
            gpu_memory_utilization=0.85,
        )
        # Use exactly the same APIs:
        for event in framework.full_chunk_stream_generation(audio): ...
    """

    def __init__(
        self,
        model_type,
        model_path,
        device="cuda",
        attn_implementation="eager",
        torch_dtype=torch.float,
        align_audio_input=False,
        allowing_backchannel=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        disable_log_stats=True,
    ):
        self.device = torch.device(device)
        self._model_path = model_path

        try:
            from lychee_fd.vllm_integration.engine import LycheeVLLMEngine
        except ImportError:
            raise ImportError(
                "vLLM engine components not found. Ensure lychee_fd.vllm_integration "
                "is importable."
            )

        dtype_str = "bfloat16" if torch_dtype == torch.bfloat16 else "float16"
        self.vllm_engine = LycheeVLLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype_str,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            disable_log_stats=disable_log_stats,
        )

        config = _load_stepaudio_config(model_path)

        self.model = _VLLMModelAdapter(self.vllm_engine, config, self.device)

        self.tokenizer = self.vllm_engine.tokenizer

        self.kl_token_id = config.keep_listening_token_id
        self.ss_token_id = config.start_speaking_token_id
        self.sl_token_id = config.start_listening_token_id
        self.ks_token_id = config.keep_speaking_token_id
        self.bc_token_id = config.start_bc_token_id
        self.detect_token_id = config.detect_token_id
        self.sleep_token_id = config.sleep_token_id
        self.text_pad_token_id = config.text_pad_token_id
        self.stoken_pad_token_id = config.stoken_pad_token_id
        self.audio_pad_token_id = config.audio_pad_token_id
        self.stoken_delay_token_id = config.stoken_delay_token_id
        self.adding_text_hiddenstates = config.adding_text_hiddenstates

        self.align_audio_input = align_audio_input
        self.allowing_backchannel = allowing_backchannel

        self.audio_token_id = self.tokenizer(["<audio_patch>"]).input_ids[0][0]
        self.tts_pad_id = self.tokenizer(["<tts_pad>"]).input_ids[0][0]
        self.tts_end_id = self.tokenizer(["<tts_end>"]).input_ids[0][0]
        self.tts_start_id = self.tokenizer(["<tts_start>"]).input_ids[0][0]
        self.eos_token_id = self.tokenizer(["<|EOT|>"]).input_ids[0][0]
        self.control_token_chunk_size = config.control_token_chunk_size

        try:
            self.stoken_audio_start_id = self.tokenizer(["<audio_0>"]).input_ids[0][0]
            self.stoken_audio_tokenizer_end_id = self.tokenizer(["<audio_6655>"]).input_ids[0][0] + 1
        except Exception:
            self.stoken_audio_start_id = 151696
            self.stoken_audio_tokenizer_end_id = 158352
        self.stoken_audio_t2w_end_id = self.stoken_audio_start_id + _LYCHEEFD_T2W_CODEC_VOCAB_SIZE
        if _LYCHEEFD_T2W_STRICT_STOKEN_RANGE:
            self.stoken_audio_end_id = min(
                self.stoken_audio_tokenizer_end_id,
                self.stoken_audio_t2w_end_id,
            )
        else:
            self.stoken_audio_end_id = self.stoken_audio_tokenizer_end_id

        self.window_second = 0.4
        self.stoken_delay_num = 10
        self.stoken_no_repeat_n_gram = 4
        self.sampling_rate = 16000
        self.last_ss_pos = None
        self.use_cache = True

        print(f"[VLLMGenerationFramework] Initialized with vLLM backend")
        if _LYCHEEFD_VERBOSE_STREAM_LOG:
            print(f"[SETTING] stoken_delay_num={self.stoken_delay_num}")
            print(f"[SETTING] stoken_no_repeat_n_gram={self.stoken_no_repeat_n_gram}")
        if _LYCHEEFD_VERBOSE_STREAM_LOG or _LYCHEEFD_T2W_STRICT_STOKEN_RANGE:
            print(
                "[SETTING] t2w_strict_stoken_range="
                f"{int(_LYCHEEFD_T2W_STRICT_STOKEN_RANGE)} "
                f"audio_range=[{self.stoken_audio_start_id},{self.stoken_audio_end_id}) "
                f"tokenizer_end={self.stoken_audio_tokenizer_end_id} "
                f"t2w_vocab_size={_LYCHEEFD_T2W_CODEC_VOCAB_SIZE}"
            )


class IncrementalChunkStreamSession:
    """
    Persistent streaming session for realtime mode.

    Unlike full_chunk_stream_generation(), this session keeps model-side
    generation state (KV cache / generated sequences / state machine cursor)
    across calls and only advances newly available audio chunks.
    """

    def __init__(
        self,
        framework,
        prefix=None,
        initial_listening_state="l",
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        self.fw = framework
        self.start_speak_token_factor = start_speak_token_factor
        self.start_listen_token_factor = start_listen_token_factor
        self.bc_speak_token_factor = bc_speak_token_factor
        self.end_speak_token_factor = end_speak_token_factor

        prefix_ids = self.fw.tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
        self.system_input_length = len(prefix_ids)
        self.prefix_input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=self.fw.model.device).unsqueeze(0)

        if prefix is not None:
            self.generated_input_ids = torch.tensor(
                prefix_ids + prefix["input_ids"], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)
            self.generated_control_ids = torch.tensor(
                prefix["control_input_ids"], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)
            self.generated_stoken_ids = torch.tensor(
                prefix["stoken_ids"], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)
        else:
            self.generated_input_ids = torch.tensor(
                prefix_ids + [self.fw.text_pad_token_id], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)
            self.generated_control_ids = torch.tensor(
                [self.fw.sleep_token_id], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)
            self.generated_stoken_ids = torch.tensor(
                [self.fw.stoken_pad_token_id], dtype=torch.long, device=self.fw.model.device
            ).unsqueeze(0)

        self.stoken_mapping = torch.tensor(
            [-1] * (self.prefix_input_ids.shape[1] + self.generated_stoken_ids.shape[1]),
            dtype=torch.long,
            device=self.fw.model.device,
        ).unsqueeze(0)

        self.past_key_values = None
        self.stoken_past_key_values = None
        self.control_past_key_values = None

        state = str(initial_listening_state).lower() if isinstance(initial_listening_state, str) else "l"
        self.listening_state = state if state in {"l", "s", "b"} else "l"

        self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
            self.end_speak_token_factor
        )

        self.profile_latency = str(os.getenv("LYCHEEFD_PROFILE_LATENCY", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            self.profile_every = max(1, int(os.getenv("LYCHEEFD_PROFILE_EVERY", "20")))
        except (TypeError, ValueError):
            self.profile_every = 20
        self.profile_model = {
            "l_calls": 0,
            "l_sec": 0.0,
            "b_calls": 0,
            "b_sec": 0.0,
            "s_calls": 0,
            "s_sec": 0.0,
            "bc_recheck_calls": 0,
            "bc_recheck_sec": 0.0,
            "s_tokens": 0,
            "b_tokens": 0,
        }
        # Persist the last visible S-L / S-S probabilities for UI display.
        # Some control processors only produce one side per phase.
        self.latest_sl_prob = 0.0
        self.latest_ss_prob = 0.0
        self.latest_ks_prob = 0.0
        self.latest_kl_prob = 0.0
        self.latest_bc_prob = 0.0

        is_vllm_backend = (
            self.fw.model.__class__.__name__ == "_VLLMModelAdapter"
            and hasattr(self.fw.model, "_engine")
        )
        keep_alive_env = str(os.getenv("LYCHEEFD_VLLM_KEEP_ALIVE_SPEAKING", "1")).strip().lower()
        keep_alive_listen_env = str(os.getenv("LYCHEEFD_VLLM_KEEP_ALIVE_LISTENING", "1")).strip().lower()
        self.keep_alive_for_speaking = (
            is_vllm_backend
            and keep_alive_env not in {"0", "false", "no", "off"}
        )
        self.keep_alive_for_listening = (
            is_vllm_backend
            and keep_alive_listen_env not in {"0", "false", "no", "off"}
        )

        self.chunk_idx = 0
        self.next_chunk_pos = math.ceil(
            (self.generated_input_ids.shape[1] - self.system_input_length + 1) / self.fw.control_token_chunk_size
        ) * self.fw.control_token_chunk_size

        self.last_ss_pos = getattr(self.fw, "last_ss_pos", None)
        if self.last_ss_pos is None and self.listening_state in {"s", "b"}:
            self.last_ss_pos = max(self.system_input_length + 1, int(self.generated_input_ids.shape[1]))
        self.fw.last_ss_pos = self.last_ss_pos

        # True-incremental audio mode:
        # - consume only newly arrived waveform slices
        # - cache cumulative audio token ids
        # - cache cumulative precomputed audio embeds to avoid full re-encode
        self.audio_incremental_mode = bool(audio_incremental_mode)
        self._audio_window_samples = max(
            1,
            int(
                round(
                    float(getattr(self.fw, "window_second", 0.4))
                    * float(getattr(self.fw, "sampling_rate", 16000))
                )
            ),
        )
        self._audio_tail = np.zeros(0, dtype=np.float32)
        self._audio_input_id_list: List[int] = []
        self._audio_input_id_lens_cache: List[int] = []
        self._audio_feats_cache: List[torch.Tensor] = []
        self._audio_feat_lens_cache: List[int] = []
        self._audio_embed_seq_cache: List[torch.Tensor] = []
        self._audio_embed_lens_cache: List[int] = []
        self._audio_embed_packed_buffer: Optional[torch.Tensor] = None
        self._audio_embed_lens_buffer: Optional[torch.Tensor] = None
        self._audio_embed_packed_count = 0
        self._audio_embed_packed_max_len = 0
        self._timeline_spans: List[Dict[str, object]] = []
        self._closed = False

    def _cuda_stats_device(self):
        if not torch.cuda.is_available():
            return None
        fw = getattr(self, "fw", None)
        fw_device = getattr(fw, "device", None)
        if fw_device is None or not str(fw_device).startswith("cuda"):
            return None
        try:
            return torch.device(fw_device)
        except Exception:
            return None

    def _cuda_memory_stats(self, device=None) -> Dict[str, object]:
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        if device is None:
            device = self._cuda_stats_device()
        try:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            max_reserved = torch.cuda.max_memory_reserved(device)
            return {
                "cuda_available": True,
                "device": str(device) if device is not None else "current",
                "allocated_mib": round(float(allocated) / 1024.0 / 1024.0, 2),
                "reserved_mib": round(float(reserved) / 1024.0 / 1024.0, 2),
                "max_allocated_mib": round(float(max_allocated) / 1024.0 / 1024.0, 2),
                "max_reserved_mib": round(float(max_reserved) / 1024.0 / 1024.0, 2),
            }
        except Exception as exc:
            return {"cuda_available": True, "error": str(exc)}

    @staticmethod
    def _tensor_list_bytes(values) -> int:
        total = 0
        for value in list(values or []):
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
        return int(total)

    def _audio_cache_stats(self) -> Dict[str, object]:
        audio_feats = getattr(self, "_audio_feats_cache", []) or []
        audio_embeds = getattr(self, "_audio_embed_seq_cache", []) or []
        return {
            "audio_input_tokens": int(len(getattr(self, "_audio_input_id_list", []) or [])),
            "audio_windows": int(max(
                len(getattr(self, "_audio_input_id_lens_cache", []) or []),
                len(audio_feats),
                len(audio_embeds),
            )),
            "audio_feats_tensors": int(len(audio_feats)),
            "audio_embeds_tensors": int(len(audio_embeds)),
            "audio_feats_mib": round(float(self._tensor_list_bytes(audio_feats)) / 1024.0 / 1024.0, 2),
            "audio_embeds_mib": round(float(self._tensor_list_bytes(audio_embeds)) / 1024.0 / 1024.0, 2),
        }

    def _clear_packed_audio_embed_buffer(self) -> None:
        self._audio_embed_packed_buffer = None
        self._audio_embed_lens_buffer = None
        self._audio_embed_packed_count = 0
        self._audio_embed_packed_max_len = 0

    def _ensure_packed_audio_embed_capacity(
        self,
        *,
        required_count: int,
        required_max_len: int,
        prototype: torch.Tensor,
    ) -> None:
        required_count = max(1, int(required_count))
        required_max_len = max(1, int(required_max_len))
        hidden_size = int(prototype.shape[-1])
        current = self._audio_embed_packed_buffer
        current_lens = self._audio_embed_lens_buffer
        current_count = int(self._audio_embed_packed_count)
        need_new = (
            current is None
            or current_lens is None
            or current.device != prototype.device
            or current.dtype != prototype.dtype
            or int(current.shape[0]) < required_count
            or int(current.shape[1]) < required_max_len
            or int(current.shape[2]) != hidden_size
        )
        if not need_new:
            return

        old_capacity = int(current.shape[0]) if current is not None and current.ndim == 3 else 0
        old_max_len = int(current.shape[1]) if current is not None and current.ndim == 3 else 0
        new_capacity = max(required_count, 16 if old_capacity <= 0 else old_capacity * 2)
        new_max_len = max(required_max_len, old_max_len)
        new_buffer = torch.zeros(
            (new_capacity, new_max_len, hidden_size),
            dtype=prototype.dtype,
            device=prototype.device,
        )
        new_lens = torch.zeros((new_capacity,), dtype=torch.int32, device=prototype.device)
        can_copy_current = (
            current is not None
            and current_lens is not None
            and current_count > 0
            and current.ndim == 3
            and int(current.shape[2]) == int(new_buffer.shape[2])
        )
        if can_copy_current:
            kept = min(current_count, int(current.shape[0]), int(new_buffer.shape[0]))
            copy_len = min(int(current.shape[1]), int(new_buffer.shape[1]))
            new_buffer[:kept, :copy_len, :].copy_(current[:kept, :copy_len, :])
            new_lens[:kept].copy_(current_lens[:kept])

        self._audio_embed_packed_buffer = new_buffer
        self._audio_embed_lens_buffer = new_lens
        self._audio_embed_packed_max_len = int(new_max_len)

    def _append_packed_audio_embed(self, seq: torch.Tensor, length: int) -> None:
        if not torch.is_tensor(seq) or seq.ndim != 2:
            return
        seq = seq.detach()
        length = max(0, min(int(length), int(seq.shape[0])))
        required_count = int(self._audio_embed_packed_count) + 1
        required_max_len = max(int(self._audio_embed_packed_max_len), int(length), 1)
        self._ensure_packed_audio_embed_capacity(
            required_count=required_count,
            required_max_len=required_max_len,
            prototype=seq,
        )
        buffer = self._audio_embed_packed_buffer
        lens = self._audio_embed_lens_buffer
        if buffer is None or lens is None:
            return
        row = int(self._audio_embed_packed_count)
        buffer[row].zero_()
        if length > 0:
            buffer[row, :length, :].copy_(seq[:length, :])
        lens[row] = int(length)
        self._audio_embed_packed_count = row + 1

    def _rebuild_packed_audio_embed_buffer(self) -> None:
        self._clear_packed_audio_embed_buffer()
        for seq, length in zip(self._audio_embed_seq_cache, self._audio_embed_lens_cache):
            if torch.is_tensor(seq):
                self._append_packed_audio_embed(seq, int(length))

    def _packed_audio_embeds(self):
        if not self._audio_embed_seq_cache:
            return None
        if int(self._audio_embed_packed_count) != int(len(self._audio_embed_seq_cache)):
            self._rebuild_packed_audio_embed_buffer()
        buffer = self._audio_embed_packed_buffer
        lens = self._audio_embed_lens_buffer
        count = int(self._audio_embed_packed_count)
        if buffer is None or lens is None or count <= 0:
            return None
        return buffer[:count], lens[:count]

    def close(self) -> Dict[str, object]:
        """Release per-call streaming state held by this realtime session."""
        if getattr(self, "_closed", False):
            return {"ok": True, "closed": False, "reason": "already_closed"}
        self._closed = True
        stats_device = self._cuda_stats_device()
        memory_before = self._cuda_memory_stats(stats_device)
        cache_before = self._audio_cache_stats()

        close_result = None
        fw = getattr(self, "fw", None)
        model = getattr(fw, "model", None)
        close_active_request = getattr(model, "close_active_request", None)
        if callable(close_active_request):
            close_result = close_active_request()

        if fw is not None:
            try:
                fw.last_ss_pos = None
            except Exception:
                pass

        for attr in (
            "prefix_input_ids",
            "generated_input_ids",
            "generated_control_ids",
            "generated_stoken_ids",
            "stoken_mapping",
            "past_key_values",
            "stoken_past_key_values",
            "control_past_key_values",
            "speaking_text_processor",
            "speaking_stoken_processor",
        ):
            setattr(self, attr, None)

        self._audio_tail = np.zeros(0, dtype=np.float32)
        for attr in (
            "_audio_input_id_list",
            "_audio_input_id_lens_cache",
            "_audio_feats_cache",
            "_audio_feat_lens_cache",
            "_audio_embed_seq_cache",
            "_audio_embed_lens_cache",
            "_timeline_spans",
        ):
            value = getattr(self, attr, None)
            if isinstance(value, list):
                value.clear()
            else:
                setattr(self, attr, [])

        self._clear_packed_audio_embed_buffer()
        self.profile_model = {}
        self.fw = None

        gc.collect()
        memory_after_gc = self._cuda_memory_stats(stats_device)
        empty_cache_enabled = str(
            os.getenv("LYCHEEFD_CUDA_EMPTY_CACHE_ON_SESSION_CLOSE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        memory_after_empty_cache = None
        if empty_cache_enabled and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
            except Exception:
                pass
            memory_after_empty_cache = self._cuda_memory_stats(stats_device)

        return {
            "ok": True,
            "closed": True,
            "active_request": close_result,
            "cache_before": cache_before,
            "cuda_before": memory_before,
            "cuda_after_gc": memory_after_gc,
            "empty_cache_enabled": bool(empty_cache_enabled),
            "cuda_after_empty_cache": memory_after_empty_cache,
        }

    def _set_last_ss_pos(self, value):
        self.last_ss_pos = value
        self.fw.last_ss_pos = value

    def get_listening_state(self) -> str:
        return self.listening_state if self.listening_state in {"l", "s", "b"} else "l"

    def get_prefix(self) -> Dict[str, List[int]]:
        text_ids = self.generated_input_ids[:, self.system_input_length:][0].detach().cpu().tolist()
        stoken_ids = self.generated_stoken_ids[0].detach().cpu().tolist()
        control_ids = self.generated_control_ids[0].detach().cpu().tolist()
        return {
            "input_ids": [int(x) for x in text_ids],
            "stoken_ids": [int(x) for x in stoken_ids],
            "control_input_ids": [int(x) for x in control_ids],
        }

    def build_generation_complete_event(self) -> Dict[str, List[int]]:
        prefix = self.get_prefix()
        return {
            "type": "generation_complete",
            "text_ids": prefix["input_ids"],
            "stoken_ids": prefix["stoken_ids"],
            "control_ids": prefix["control_input_ids"],
        }

    def _is_vllm_backend(self) -> bool:
        return (
            self.fw.model.__class__.__name__ == "_VLLMModelAdapter"
            and hasattr(self.fw.model, "_engine")
        )

    @staticmethod
    def _concat_audio_chunks(chunks: List[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _record_timeline_span(
        self,
        name: str,
        start_perf: float,
        start_epoch_ms: int,
        end_perf: Optional[float] = None,
        end_epoch_ms: Optional[int] = None,
        **extra,
    ) -> None:
        if end_perf is None:
            end_perf = time.perf_counter()
        if end_epoch_ms is None:
            end_epoch_ms = int(time.time() * 1000)
        span = {
            "name": str(name),
            "start_epoch_ms": int(start_epoch_ms),
            "end_epoch_ms": int(end_epoch_ms),
            "duration_ms": round(max(0.0, float(end_perf) - float(start_perf)) * 1000.0, 3),
        }
        for key, value in extra.items():
            if value is not None:
                span[str(key)] = value
        self._timeline_spans.append(span)

    def _drain_timeline_span_events(self):
        spans = list(getattr(self, "_timeline_spans", []) or [])
        self._timeline_spans = []
        for span in spans:
            yield {"type": "timeline_span", **span}

    @staticmethod
    def _detach_cache_tensor(value):
        if torch.is_tensor(value):
            return value.detach()
        return value

    def _infer_audio_input_id_lens_cache(self) -> List[int]:
        lens = [
            max(0, int(x))
            for x in list(getattr(self, "_audio_input_id_lens_cache", []) or [])
        ]
        if lens:
            return lens

        token_count = int(len(getattr(self, "_audio_input_id_list", []) or []))
        if token_count <= 0:
            return []
        window_count = max(
            len(getattr(self, "_audio_feats_cache", []) or []),
            len(getattr(self, "_audio_embed_seq_cache", []) or []),
        )
        if window_count <= 0:
            return [token_count]

        default_len = max(1, int(getattr(self.fw, "control_token_chunk_size", 1)))
        inferred = []
        remaining = token_count
        for _ in range(window_count):
            if remaining <= 0:
                inferred.append(0)
                continue
            take = min(default_len, remaining)
            inferred.append(int(take))
            remaining -= int(take)
        if remaining > 0:
            inferred[-1] += int(remaining)
        return inferred

    def export_incremental_audio_cache(self, max_input_ids: Optional[int] = None) -> Dict[str, object]:
        """Export cached incremental audio context up to a token-aligned anchor."""
        total_tokens = int(len(getattr(self, "_audio_input_id_list", []) or []))
        if max_input_ids is None:
            requested_tokens = total_tokens
        else:
            requested_tokens = max(0, int(max_input_ids))

        lens = self._infer_audio_input_id_lens_cache()
        keep_windows = 0
        keep_tokens = 0
        for token_len in lens:
            if keep_tokens >= requested_tokens:
                break
            keep_tokens += max(0, int(token_len))
            keep_windows += 1
        if not lens and requested_tokens > 0:
            keep_tokens = min(requested_tokens, total_tokens)
            keep_windows = 0

        keep_tokens = max(0, min(int(keep_tokens), total_tokens))
        keep_lens = list(lens[:keep_windows])

        return {
            "requested_audio_input_token_count": int(requested_tokens),
            "audio_input_token_count": int(keep_tokens),
            "audio_window_count": int(keep_windows),
            "audio_input_ids": [
                int(x) for x in list(getattr(self, "_audio_input_id_list", []) or [])[:keep_tokens]
            ],
            "audio_input_id_lens": keep_lens,
            "audio_feats": [
                self._detach_cache_tensor(x)
                for x in list(getattr(self, "_audio_feats_cache", []) or [])[:keep_windows]
            ],
            "audio_feat_lens": [
                int(x) for x in list(getattr(self, "_audio_feat_lens_cache", []) or [])[:keep_windows]
            ],
            "audio_embed_seq": [
                self._detach_cache_tensor(x)
                for x in list(getattr(self, "_audio_embed_seq_cache", []) or [])[:keep_windows]
            ],
            "audio_embed_lens": [
                int(x) for x in list(getattr(self, "_audio_embed_lens_cache", []) or [])[:keep_windows]
            ],
        }

    def _apply_incremental_audio_cache_snapshot(self, snapshot: Dict[str, object]) -> Dict[str, int]:
        self._audio_input_id_list = [
            int(x) for x in list(snapshot.get("audio_input_ids") or [])
        ]
        self._audio_input_id_lens_cache = [
            max(0, int(x)) for x in list(snapshot.get("audio_input_id_lens") or [])
        ]
        self._audio_feats_cache = [
            x.to(self.fw.device) if torch.is_tensor(x) else x
            for x in list(snapshot.get("audio_feats") or [])
        ]
        self._audio_feat_lens_cache = [
            int(x) for x in list(snapshot.get("audio_feat_lens") or [])
        ]
        self._audio_embed_seq_cache = [
            x.to(self.fw.device) if torch.is_tensor(x) else x
            for x in list(snapshot.get("audio_embed_seq") or [])
        ]
        self._audio_embed_lens_cache = [
            int(x) for x in list(snapshot.get("audio_embed_lens") or [])
        ]
        self._clear_packed_audio_embed_buffer()
        self._audio_tail = np.zeros(0, dtype=np.float32)

        return {
            "requested_audio_input_token_count": int(
                snapshot.get("requested_audio_input_token_count") or 0
            ),
            "audio_input_token_count": int(len(self._audio_input_id_list)),
            "audio_window_count": int(max(
                len(self._audio_input_id_lens_cache),
                len(self._audio_embed_seq_cache),
                len(self._audio_feats_cache),
            )),
        }

    def restore_incremental_audio_cache(self, snapshot: Dict[str, object]) -> Dict[str, int]:
        if not isinstance(snapshot, dict):
            raise TypeError("audio cache snapshot must be a dict")

        return self._apply_incremental_audio_cache_snapshot(snapshot)

    @staticmethod
    def _list_is_prefix(prefix: List[int], full: List[int]) -> bool:
        if len(prefix) > len(full):
            return False
        return list(full[:len(prefix)]) == list(prefix)

    def truncate_active_request_to_prefix(self, prefix: Dict[str, List[int]]) -> Dict[str, object]:
        if not isinstance(prefix, dict):
            return {"ok": False, "reason": "invalid_prefix"}
        if not self._is_vllm_backend():
            return {"ok": False, "reason": "not_vllm_backend"}

        target_input_ids = [int(x) for x in list(prefix.get("input_ids") or [])]
        target_stoken_ids = [int(x) for x in list(prefix.get("stoken_ids") or [])]
        target_control_ids = [int(x) for x in list(prefix.get("control_input_ids") or [])]
        current_prefix = self.get_prefix()
        if not self._list_is_prefix(target_input_ids, current_prefix.get("input_ids", [])):
            return {"ok": False, "reason": "text_prefix_mismatch"}
        if not self._list_is_prefix(target_stoken_ids, current_prefix.get("stoken_ids", [])):
            return {"ok": False, "reason": "stoken_prefix_mismatch"}
        if not self._list_is_prefix(target_control_ids, current_prefix.get("control_input_ids", [])):
            return {"ok": False, "reason": "control_prefix_mismatch"}

        anchor_prefix_len = int(len(target_input_ids))
        if not (
            len(target_stoken_ids) == anchor_prefix_len
            and len(target_control_ids) == anchor_prefix_len
        ):
            return {
                "ok": False,
                "reason": "target_length_mismatch",
                "text_len": int(anchor_prefix_len),
                "stoken_len": int(len(target_stoken_ids)),
                "control_len": int(len(target_control_ids)),
            }

        audio_cache_snapshot = self.export_incremental_audio_cache(anchor_prefix_len)
        audio_cache_tokens = int(audio_cache_snapshot.get("audio_input_token_count") or 0)
        if audio_cache_tokens != anchor_prefix_len:
            return {
                "ok": False,
                "reason": "audio_cache_not_token_aligned",
                "requested_tokens": int(anchor_prefix_len),
                "audio_cache_tokens": int(audio_cache_tokens),
            }

        target_text_ids = torch.tensor(
            self.prefix_input_ids[0].detach().cpu().tolist() + target_input_ids,
            dtype=torch.long,
            device=self.fw.model.device,
        ).unsqueeze(0)
        target_stoken_tensor = torch.tensor(
            target_stoken_ids,
            dtype=torch.long,
            device=self.fw.model.device,
        ).unsqueeze(0)
        target_control_tensor = torch.tensor(
            target_control_ids,
            dtype=torch.long,
            device=self.fw.model.device,
        ).unsqueeze(0)
        target_audio_tensor = torch.tensor(
            [int(x) for x in list(audio_cache_snapshot.get("audio_input_ids") or [])],
            dtype=torch.long,
            device=self.fw.device,
        ).unsqueeze(0)

        truncate_fn = getattr(self.fw.model, "truncate_active_request_to_sequences", None)
        if not callable(truncate_fn):
            return {"ok": False, "reason": "vllm_truncate_api_unavailable"}
        truncate_result = truncate_fn(
            text_ids=target_text_ids,
            stoken_ids=target_stoken_tensor,
            control_ids=target_control_tensor,
            audio_ids=target_audio_tensor,
        )
        if not isinstance(truncate_result, dict) or not truncate_result.get("ok"):
            result = dict(truncate_result) if isinstance(truncate_result, dict) else {}
            result.setdefault("ok", False)
            result.setdefault("reason", "vllm_truncate_failed")
            result["audio_cache_tokens"] = int(audio_cache_tokens)
            result["audio_cache_windows"] = int(audio_cache_snapshot.get("audio_window_count") or 0)
            return result

        self.generated_input_ids = target_text_ids
        self.generated_stoken_ids = target_stoken_tensor
        self.generated_control_ids = target_control_tensor
        self.stoken_mapping = torch.full(
            (1, int(self.prefix_input_ids.shape[1]) + int(self.generated_stoken_ids.shape[1])),
            -1,
            dtype=torch.long,
            device=self.fw.device,
        )
        self.listening_state = "l"
        self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
            self.end_speak_token_factor
        )
        self.latest_sl_prob = 0.0
        self.latest_ss_prob = 0.0
        self.latest_ks_prob = 0.0
        self.latest_kl_prob = 0.0
        self.latest_bc_prob = 0.0
        self.chunk_idx = 0
        self.next_chunk_pos = math.ceil(
            (self.generated_input_ids.shape[1] - self.system_input_length + 1)
            / self.fw.control_token_chunk_size
        ) * self.fw.control_token_chunk_size
        restore_stats = self._apply_incremental_audio_cache_snapshot(audio_cache_snapshot)
        result = dict(truncate_result)
        result["audio_cache_tokens"] = int(restore_stats.get("audio_input_token_count", 0))
        result["audio_cache_windows"] = int(restore_stats.get("audio_window_count", 0))
        result["prefix_len"] = int(anchor_prefix_len)
        return result

    def _split_incremental_windows(self, audio: np.ndarray, flush_audio_tail: bool) -> List[np.ndarray]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self._audio_tail.size > 0:
            if audio.size > 0:
                audio = np.concatenate([self._audio_tail, audio]).astype(np.float32, copy=False)
            else:
                audio = self._audio_tail
        self._audio_tail = np.zeros(0, dtype=np.float32)

        if audio.size == 0:
            return []

        windows: List[np.ndarray] = []
        start = 0
        ws = int(self._audio_window_samples)
        while start + ws <= audio.shape[0]:
            windows.append(audio[start:start + ws])
            start += ws

        remain = audio[start:]
        if flush_audio_tail and remain.size > 0:
            windows.append(remain)
            remain = np.zeros(0, dtype=np.float32)
        self._audio_tail = remain.astype(np.float32, copy=False)
        return windows

    def _cache_precomputed_audio(self, feats: List[torch.Tensor], feat_lens: List[int]) -> None:
        if not feats:
            return

        wavs = torch.nn.utils.rnn.pad_sequence(
            feats, batch_first=True, padding_value=0
        ).transpose(1, 2).to(self.fw.device)
        wav_lens = torch.tensor(feat_lens, dtype=torch.int32).to(self.fw.device)

        t_encoder_start = time.perf_counter()
        encoder_start_epoch_ms = int(time.time() * 1000)
        if self._is_vllm_backend():
            out, out_lens = self.fw.model._engine.precompute_audio(wavs, wav_lens)
        else:
            _wavs = wavs.bfloat16() if getattr(self.fw.model, "bf16", False) else wavs
            out, out_lens = self.fw.model.encoder(_wavs, wav_lens)
            out = self.fw.model.adapter(out)
            out_lens = (out_lens - 1) // 2 + 1
        self._record_timeline_span(
            "audio_encoder",
            t_encoder_start,
            encoder_start_epoch_ms,
            backend=("vllm" if self._is_vllm_backend() else "hf"),
            batch_size=int(wavs.shape[0]),
            max_feat_len=int(wavs.shape[-1]) if hasattr(wavs, "shape") and len(wavs.shape) >= 3 else None,
        )

        for i in range(int(out.shape[0])):
            li = int(out_lens[i].item()) if torch.is_tensor(out_lens) else int(out_lens[i])
            li = max(0, min(li, int(out.shape[1])))
            seq = out[i, :li, :].detach()
            self._audio_embed_seq_cache.append(seq)
            self._audio_embed_lens_cache.append(li)
            self._append_packed_audio_embed(seq, li)

    def _append_incremental_audio(self, audio: np.ndarray, flush_audio_tail: bool) -> None:
        windows = self._split_incremental_windows(audio, flush_audio_tail=flush_audio_tail)
        if not windows:
            return

        for wav_chunk in windows:
            t_pre_start = time.perf_counter()
            pre_start_epoch_ms = int(time.time() * 1000)
            model_inputs = self.fw.stepaudio_audio_prepreocess(wav_chunk)
            self._record_timeline_span(
                "audio_preprocess",
                t_pre_start,
                pre_start_epoch_ms,
                input_samples=int(np.asarray(wav_chunk).size),
            )
            input_ids = model_inputs.get("input_ids", [])
            feats = model_inputs.get("feats", [])
            feat_lens = model_inputs.get("feats_lengths", [])
            input_id_len = int(len(input_ids)) if input_ids else 0
            self._audio_input_id_lens_cache.append(input_id_len)
            if input_id_len > 0:
                self._audio_input_id_list.extend([int(x) for x in input_ids])
            if feats:
                self._audio_feats_cache.extend(feats)
                self._audio_feat_lens_cache.extend([int(x) for x in feat_lens])
                self._cache_precomputed_audio(feats, [int(x) for x in feat_lens])

    def _build_cached_audio_context(self):
        total_len = int(len(self._audio_input_id_list))
        if total_len <= 0:
            return None, None, None, 0, None, None

        audio_input_ids = torch.tensor(
            self._audio_input_id_list,
            dtype=torch.long,
            device=self.fw.device,
        ).unsqueeze(0)

        wavs = None
        wav_lens = None
        build_raw_wavs = bool(self._audio_feats_cache) and not (
            self._is_vllm_backend() and bool(self._audio_embed_seq_cache)
        )
        if build_raw_wavs:
            wavs = torch.nn.utils.rnn.pad_sequence(
                self._audio_feats_cache, batch_first=True, padding_value=0
            ).transpose(1, 2).to(self.fw.device)
            wav_lens = torch.tensor(
                self._audio_feat_lens_cache, dtype=torch.int32
            ).to(self.fw.device)

        audio_embeds = None
        pre_computed_audio = None
        if self._audio_embed_seq_cache:
            if self._is_vllm_backend():
                audio_embeds = self._packed_audio_embeds()
            else:
                packed = torch.nn.utils.rnn.pad_sequence(
                    self._audio_embed_seq_cache,
                    batch_first=True,
                    padding_value=0.0,
                ).to(self.fw.device)
                packed_lens = torch.tensor(
                    self._audio_embed_lens_cache, dtype=torch.int32, device=self.fw.device
                )
                pre_computed_audio = (packed, packed_lens)

        return audio_input_ids, wavs, wav_lens, total_len, audio_embeds, pre_computed_audio

    def _prepare_audio(
        self,
        audio,
        *,
        audio_is_incremental=False,
        flush_audio_tail=False,
    ):
        if not audio_is_incremental:
            t_pre_start = time.perf_counter()
            pre_start_epoch_ms = int(time.time() * 1000)
            model_inputs = self.fw.stepaudio_audio_prepreocess(audio)
            self._record_timeline_span(
                "audio_preprocess",
                t_pre_start,
                pre_start_epoch_ms,
                input_samples=int(np.asarray(audio).size),
            )
            feats, feats_lengths = model_inputs["feats"], model_inputs["feats_lengths"]
            wavs = torch.nn.utils.rnn.pad_sequence(
                feats, batch_first=True, padding_value=0
            ).transpose(1, 2).to(self.fw.device)
            wav_lens = torch.tensor(feats_lengths, dtype=torch.int32).to(self.fw.device)
            audio_input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0).to(self.fw.device)
            total_len = len(model_inputs["input_ids"])
            return audio_input_ids, wavs, wav_lens, total_len, None, None

        self._append_incremental_audio(audio, flush_audio_tail=flush_audio_tail)
        return self._build_cached_audio_context()

    def advance_stream(
        self,
        audio,
        emit_generation_complete=True,
        audio_is_incremental=False,
        flush_audio_tail=False,
        control_early_callback=None,
    ):
        use_incremental_audio = bool(audio_is_incremental or self.audio_incremental_mode)
        self._timeline_spans = []
        audio_input_ids, wavs, wav_lens, total_len, audio_embeds, pre_computed_audio = self._prepare_audio(
            audio,
            audio_is_incremental=use_incremental_audio,
            flush_audio_tail=bool(flush_audio_tail),
        )
        for span_event in self._drain_timeline_span_events():
            yield span_event
        chunk_end = math.ceil(total_len / self.fw.control_token_chunk_size) * self.fw.control_token_chunk_size
        chunk_start = self.next_chunk_pos

        def _safe_prob(proc_obj, key):
            cache_attr = None
            if key == "sl":
                cache_attr = "latest_sl_prob"
            elif key == "ss":
                cache_attr = "latest_ss_prob"
            elif key == "ks":
                cache_attr = "latest_ks_prob"
            elif key == "kl":
                cache_attr = "latest_kl_prob"
            elif key == "bc":
                cache_attr = "latest_bc_prob"
            cached = getattr(self, cache_attr, None) if cache_attr is not None else None
            if proc_obj is None:
                return cached
            probs = getattr(proc_obj, "last_probs", None)
            if not isinstance(probs, dict):
                return cached
            value = probs.get(key, None)
            if value is None:
                return cached
            try:
                fv = float(value)
            except (TypeError, ValueError):
                return cached
            if not math.isfinite(fv):
                return cached
            if cache_attr is not None:
                setattr(self, cache_attr, fv)
            return fv

        def _emit_control_early(event):
            if not callable(control_early_callback) or not isinstance(event, dict):
                return
            try:
                control_early_callback(event)
            except Exception:
                pass

        if chunk_start > chunk_end:
            if emit_generation_complete:
                yield self.build_generation_complete_event()
            return

        total_chunks = (chunk_end - chunk_start) // self.fw.control_token_chunk_size + 1

        if _LYCHEEFD_VERBOSE_STREAM_LOG:
            print(
                f"[STREAM_SESSION] total_audio_tokens={total_len}, chunk_start={chunk_start}, "
                f"chunk_end={chunk_end}, chunk_size={self.fw.control_token_chunk_size}, total_chunks={total_chunks}, "
                f"state={self.listening_state}, chunk_idx={self.chunk_idx}"
            )

        for cn_e in range(chunk_start, chunk_end + 1, self.fw.control_token_chunk_size):
            target_length = min(cn_e, total_len)
            current_len = self.generated_input_ids.shape[1] - self.system_input_length
            target_new_length = target_length - current_len

            yield {"type": "chunk_start", "chunk_idx": self.chunk_idx, "chunk_pos": cn_e, "total_chunks": total_chunks}

            if target_new_length <= 0:
                yield {"type": "chunk_end", "chunk_idx": self.chunk_idx}
                self.chunk_idx += 1
                self.next_chunk_pos = cn_e + self.fw.control_token_chunk_size
                continue

            if self.listening_state == "l":
                listening_control_processor = LogitsProcessorList([
                    ListeningControlLogitsProcessor(
                        ss_token_id=self.fw.ss_token_id,
                        kl_token_id=self.fw.kl_token_id,
                        bc_token_id=self.fw.bc_token_id if self.fw.allowing_backchannel else None,
                        vocab_size=self.fw.model.config.text_config.vocab_size,
                        start_speak_token_factor=self.start_speak_token_factor,
                        bc_speak_token_factor=self.bc_speak_token_factor,
                        sleep_token_id=self.fw.sleep_token_id,
                        detect_token_id=self.fw.detect_token_id,
                        prefix_input_len=self.prefix_input_ids.shape[1],
                        control_token_chunk_size=self.fw.control_token_chunk_size,
                    )
                ])
                listening_control_core = listening_control_processor[0] if len(listening_control_processor) > 0 else None
                listening_text_processor, listening_stoken_processor = self.fw.init_listening_pad_processor()
                # Reserve mapping for newly generated listening tokens.
                self.stoken_mapping = torch.cat(
                    [self.stoken_mapping, torch.full((1, target_new_length), -1, dtype=torch.long, device=self.fw.device)],
                    dim=1,
                )

                last_result = None
                t_model_l_start = time.perf_counter()
                model_l_start_epoch_ms = int(time.time() * 1000)

                def _listening_control_early(control_event, *, chunk_pos=cn_e, target_steps=target_new_length):
                    try:
                        token_control = int(control_event.get("control_token"))
                        step = int(control_event.get("step", -1))
                    except (TypeError, ValueError):
                        return
                    if step != max(0, int(target_steps) - 1):
                        return
                    if token_control != int(self.fw.ss_token_id):
                        return
                    _emit_control_early(
                        {
                            "type": "control_head_state_change",
                            "from": "L",
                            "to": "S",
                            "pos": int(chunk_pos),
                            "chunk": int(chunk_pos),
                            "reason": "control_ss",
                            "early_exit": True,
                            "control_head_early": True,
                            "interrupt": False,
                            "control_token": int(token_control),
                            "step": int(step),
                            "ss_prob": _safe_prob(listening_control_core, "ss"),
                            "sl_prob": _safe_prob(listening_control_core, "sl"),
                            "kl_prob": _safe_prob(listening_control_core, "kl"),
                            "bc_prob": _safe_prob(listening_control_core, "bc"),
                            "model_control_epoch_ms": control_event.get("timestamp_epoch_ms"),
                        }
                    )

                for token_result in self.fw.model.multi_head_generate_stream(
                    input_ids=self.generated_input_ids,
                    stoken_ids=self.generated_stoken_ids,
                    control_input_ids=self.generated_control_ids,
                    prefix_input_ids=self.prefix_input_ids,
                    stoken_mapping=self.stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    audio_embeds=audio_embeds,
                    pre_computed_audio=pre_computed_audio,
                    past_key_values=self.past_key_values,
                    stoken_past_key_values=self.stoken_past_key_values,
                    control_past_key_values=self.control_past_key_values,
                    use_cache=self.fw.use_cache and not self.fw.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=listening_control_processor,
                    logits_processor=listening_text_processor,
                    stoken_logits_processor=listening_stoken_processor,
                    temperature=1.0,
                    top_k=0,
                    eos_token_id=None,
                    keep_request_alive=self.keep_alive_for_listening,
                    control_early_callback=_listening_control_early,
                ):
                    last_result = token_result
                t_model_l_cost = time.perf_counter() - t_model_l_start
                self._record_timeline_span(
                    "transformer",
                    t_model_l_start,
                    model_l_start_epoch_ms,
                    state="listening",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=int(target_new_length),
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event
                self.profile_model["l_calls"] += 1
                self.profile_model["l_sec"] += t_model_l_cost

                if last_result is not None:
                    self.past_key_values = last_result["past_key_values"]
                    self.stoken_past_key_values = last_result["stoken_past_key_values"]
                    self.control_past_key_values = last_result["control_past_key_values"]
                    self.generated_input_ids = last_result["sequences"]
                    self.generated_stoken_ids = last_result["stoken_ids"]
                    self.generated_control_ids = last_result["control_ids"]
                    pred_control_token = self.generated_control_ids[0, -1]
                else:
                    # Emergency fallback: append a keep-listening step.
                    padding_input_ids = self.fw.text_pad_token_id * torch.ones(
                        (self.generated_input_ids.shape[0], 1),
                        dtype=self.generated_input_ids.dtype,
                        device=self.generated_input_ids.device,
                    )
                    padding_stoken_ids = self.fw.stoken_pad_token_id * torch.ones(
                        (self.generated_input_ids.shape[0], 1),
                        dtype=self.generated_input_ids.dtype,
                        device=self.generated_input_ids.device,
                    )
                    padding_control_input_ids = self.fw.kl_token_id * torch.ones(
                        (self.generated_control_ids.shape[0], 1),
                        dtype=self.generated_control_ids.dtype,
                        device=self.generated_control_ids.device,
                    )
                    self.generated_input_ids = torch.cat((self.generated_input_ids, padding_input_ids), dim=1)
                    self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, padding_stoken_ids), dim=1)
                    self.generated_control_ids = torch.cat((self.generated_control_ids, padding_control_input_ids), dim=1)
                    pred_control_token = self.fw.kl_token_id
                yield {
                    "type": "control_decision",
                    "state": "l",
                    "chunk": cn_e,
                    "token": int(pred_control_token),
                    "ss_prob": _safe_prob(listening_control_core, "ss"),
                    "sl_prob": _safe_prob(listening_control_core, "sl"),
                    "kl_prob": _safe_prob(listening_control_core, "kl"),
                    "bc_prob": _safe_prob(listening_control_core, "bc"),
                }

                if pred_control_token == self.fw.ss_token_id:
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "ss_prob": _safe_prob(listening_control_core, "ss"),
                        "sl_prob": _safe_prob(listening_control_core, "sl"),
                    }
                    self.listening_state = "s"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])
                elif pred_control_token == self.fw.bc_token_id:
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "B",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_bc",
                        "ss_prob": _safe_prob(listening_control_core, "ss"),
                        "sl_prob": _safe_prob(listening_control_core, "sl"),
                    }
                    self.listening_state = "b"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])

            elif self.listening_state == "b":
                if self.last_ss_pos is None:
                    self._set_last_ss_pos(max(self.system_input_length + 1, int(self.generated_input_ids.shape[1])))
                stoken_mapping_len = self.stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.fw.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.fw.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        self.stoken_mapping = torch.cat(
                            [self.stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.fw.device)],
                            dim=1,
                        )
                if self.stoken_mapping.shape[1] - self.generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + target_new_length):
                        seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(
                        padding_stoken_mapping, dtype=torch.long, device=self.fw.model.device
                    ).unsqueeze(0)
                    self.stoken_mapping = torch.cat((self.stoken_mapping, padding_stoken_mapping), dim=1)

                speaking_control_processor = LogitsProcessorList([
                    BackChannelLogitsProcessor(
                        sleep_token_id=self.fw.sleep_token_id,
                        detect_token_id=self.fw.detect_token_id,
                        sl_token_id=self.fw.sl_token_id,
                        ss_token_id=self.fw.ss_token_id,
                        bc_token_id=self.fw.bc_token_id,
                        vocab_size=self.fw.model.config.text_config.vocab_size,
                        prefix_input_len=self.prefix_input_ids.shape[1],
                        control_token_chunk_size=self.fw.control_token_chunk_size,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None

                last_result = None
                t_model_b_start = time.perf_counter()
                model_b_start_epoch_ms = int(time.time() * 1000)
                for token_result in self.fw.model.multi_head_generate_stream(
                    input_ids=self.generated_input_ids,
                    stoken_ids=self.generated_stoken_ids,
                    control_input_ids=self.generated_control_ids,
                    prefix_input_ids=self.prefix_input_ids,
                    stoken_mapping=self.stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    audio_embeds=audio_embeds,
                    pre_computed_audio=pre_computed_audio,
                    past_key_values=self.past_key_values,
                    stoken_past_key_values=self.stoken_past_key_values,
                    control_past_key_values=self.control_past_key_values,
                    use_cache=self.fw.use_cache and not self.fw.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor,
                    logits_processor=self.speaking_text_processor,
                    stoken_logits_processor=self.speaking_stoken_processor,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.fw.tts_end_id,
                    keep_request_alive=self.keep_alive_for_speaking,
                ):
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    last_result = token_result
                t_model_b_cost = time.perf_counter() - t_model_b_start
                self._record_timeline_span(
                    "transformer",
                    t_model_b_start,
                    model_b_start_epoch_ms,
                    state="backchannel",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=int(target_new_length),
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event
                self.profile_model["b_calls"] += 1
                self.profile_model["b_sec"] += t_model_b_cost

                if last_result is None:
                    last_result = {
                        "sequences": self.generated_input_ids,
                        "stoken_ids": self.generated_stoken_ids,
                        "control_ids": self.generated_control_ids,
                        "past_key_values": self.past_key_values,
                        "stoken_past_key_values": self.stoken_past_key_values,
                        "control_past_key_values": self.control_past_key_values,
                    }
                b_step_cnt = int(last_result["step"] + 1) if "step" in last_result else 0
                self.profile_model["b_tokens"] += max(0, b_step_cnt)

                self.past_key_values = last_result["past_key_values"]
                self.stoken_past_key_values = last_result["stoken_past_key_values"]
                self.control_past_key_values = last_result["control_past_key_values"]
                self.generated_input_ids = last_result["sequences"]
                self.generated_stoken_ids = last_result["stoken_ids"]
                self.generated_control_ids = last_result["control_ids"]

                if self.generated_control_ids[0, -1] == self.fw.sl_token_id:
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "L",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_sl",
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    self.listening_state = "l"
                    self._set_last_ss_pos(None)
                    self.generated_stoken_ids = self.fw._set_last_token_safe(self.generated_stoken_ids, self.fw.stoken_pad_token_id)
                    self.generated_input_ids = self.fw._set_last_token_safe(self.generated_input_ids, self.fw.text_pad_token_id)
                elif self.generated_control_ids[0, -1] == self.fw.ss_token_id:
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    self.listening_state = "s"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])
                    self.generated_stoken_ids = self.fw._set_last_token_safe(self.generated_stoken_ids, self.fw.stoken_pad_token_id)
                    self.generated_input_ids = self.fw._set_last_token_safe(self.generated_input_ids, self.fw.text_pad_token_id)
                elif self.generated_control_ids[0, -1] == self.fw.bc_token_id or cn_e == chunk_end:
                    pass
                else:
                    assert self.generated_stoken_ids[0, -1] == self.fw.tts_end_id
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > self.generated_input_ids.shape[1] - self.system_input_length:
                        padding_len = target_length - self.generated_input_ids.shape[1] + self.system_input_length
                        padding_input_ids = self.fw.text_pad_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        padding_stoken_ids = self.fw.stoken_pad_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        padding_control_ids = self.fw.sleep_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.fw.detect_token_id
                        padding_control_ids[:, -1] = self.fw.sl_token_id
                        self.generated_input_ids = torch.cat((self.generated_input_ids, padding_input_ids), dim=1)
                        self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, padding_stoken_ids), dim=1)
                        self.generated_control_ids = torch.cat((self.generated_control_ids, padding_control_ids), dim=1)
                        self.listening_state = "l"
                        self._set_last_ss_pos(None)
                    else:
                        padding_len = self.fw.control_token_chunk_size
                        stoken_mapping_len = self.stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.fw.stoken_delay_num + 1:
                            self.stoken_mapping = torch.cat(
                                [
                                    self.stoken_mapping,
                                    torch.full(
                                        (1, min(padding_len, self.fw.stoken_delay_num + 1 - stoken_mapping_len)),
                                        -1,
                                        dtype=torch.long,
                                        device=self.fw.device,
                                    ),
                                ],
                                dim=1,
                            )
                        if self.stoken_mapping.shape[1] - self.generated_input_ids.shape[1] < padding_len:
                            _pad_map = []
                            for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + padding_len):
                                seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                                _pad_map.append(seq_l // 4 + self.last_ss_pos)
                            _pad_map = torch.tensor(_pad_map, dtype=torch.long, device=self.fw.model.device).unsqueeze(0)
                            self.stoken_mapping = torch.cat((self.stoken_mapping, _pad_map), dim=1)

                        padding_input_ids = self.fw.text_pad_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        padding_stoken_ids = self.fw.stoken_pad_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        padding_control_ids = self.fw.sleep_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        if padding_control_ids.shape[1] > 2:
                            padding_control_ids[:, -2] = self.fw.detect_token_id

                        self.generated_input_ids = torch.cat((self.generated_input_ids, padding_input_ids[:, :-1]), dim=1)
                        self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, padding_stoken_ids[:, :-1]), dim=1)
                        self.generated_control_ids = torch.cat((self.generated_control_ids, padding_control_ids[:, :-1]), dim=1)

                        bc_ctrl_proc = LogitsProcessorList([
                            BackChannelLogitsProcessor(
                                sleep_token_id=self.fw.sleep_token_id,
                                detect_token_id=self.fw.detect_token_id,
                                sl_token_id=self.fw.sl_token_id,
                                ss_token_id=self.fw.ss_token_id,
                                bc_token_id=None,
                                vocab_size=self.fw.model.config.text_config.vocab_size,
                                prefix_input_len=self.prefix_input_ids.shape[1],
                                control_token_chunk_size=self.fw.control_token_chunk_size,
                                start_speak_token_factor=self.start_speak_token_factor,
                            )
                        ])
                        bc_ctrl_core = bc_ctrl_proc[0] if len(bc_ctrl_proc) > 0 else None
                        bc_fixed_text_processor, bc_fixed_stoken_processor = self.fw.init_listening_pad_processor()

                        last_result = None
                        t_model_bc_start = time.perf_counter()
                        model_bc_start_epoch_ms = int(time.time() * 1000)
                        for token_result in self.fw.model.multi_head_generate_stream(
                            input_ids=self.generated_input_ids,
                            stoken_ids=self.generated_stoken_ids,
                            control_input_ids=self.generated_control_ids,
                            prefix_input_ids=self.prefix_input_ids,
                            stoken_mapping=self.stoken_mapping,
                            audio_input_ids=audio_input_ids,
                            wavs=wavs,
                            wav_lens=wav_lens,
                            audio_embeds=audio_embeds,
                            pre_computed_audio=pre_computed_audio,
                            past_key_values=self.past_key_values,
                            stoken_past_key_values=self.stoken_past_key_values,
                            control_past_key_values=self.control_past_key_values,
                            use_cache=self.fw.use_cache and not self.fw.adding_text_hiddenstates,
                            max_new_tokens=1,
                            control_logits_processor=bc_ctrl_proc,
                            logits_processor=bc_fixed_text_processor,
                            stoken_logits_processor=bc_fixed_stoken_processor,
                            keep_request_alive=self.keep_alive_for_listening,
                        ):
                            last_result = token_result
                        t_model_bc_cost = time.perf_counter() - t_model_bc_start
                        self._record_timeline_span(
                            "transformer",
                            t_model_bc_start,
                            model_bc_start_epoch_ms,
                            state="backchannel_recheck",
                            chunk_idx=int(self.chunk_idx),
                            target_new_tokens=1,
                        )
                        for span_event in self._drain_timeline_span_events():
                            yield span_event
                        self.profile_model["bc_recheck_calls"] += 1
                        self.profile_model["bc_recheck_sec"] += t_model_bc_cost

                        if last_result is None:
                            last_result = {
                                "sequences": self.generated_input_ids,
                                "stoken_ids": self.generated_stoken_ids,
                                "control_ids": self.generated_control_ids,
                                "past_key_values": self.past_key_values,
                                "stoken_past_key_values": self.stoken_past_key_values,
                                "control_past_key_values": self.control_past_key_values,
                            }

                        self.past_key_values = last_result["past_key_values"]
                        self.stoken_past_key_values = last_result["stoken_past_key_values"]
                        self.control_past_key_values = last_result["control_past_key_values"]
                        self.generated_input_ids = last_result["sequences"]
                        self.generated_stoken_ids = last_result["stoken_ids"]
                        self.generated_control_ids = last_result["control_ids"]

                        pred_control_token = self.generated_control_ids[0, -1]
                        if pred_control_token == self.fw.sl_token_id:
                            yield {
                                "type": "state_change",
                                "from": "B",
                                "to": "L",
                                "pos": cn_e,
                                "chunk": cn_e,
                                "reason": "bc_recheck_sl",
                                "sl_prob": _safe_prob(bc_ctrl_core, "sl"),
                                "ss_prob": _safe_prob(bc_ctrl_core, "ss"),
                                "ks_prob": _safe_prob(bc_ctrl_core, "ks"),
                            }
                            self.listening_state = "l"
                            self._set_last_ss_pos(None)
                        else:
                            yield {
                                "type": "state_change",
                                "from": "B",
                                "to": "S",
                                "pos": cn_e,
                                "chunk": cn_e,
                                "reason": "bc_recheck_ss",
                                "sl_prob": _safe_prob(bc_ctrl_core, "sl"),
                                "ss_prob": _safe_prob(bc_ctrl_core, "ss"),
                                "ks_prob": _safe_prob(bc_ctrl_core, "ks"),
                            }
                            self.listening_state = "s"
                            self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                                self.end_speak_token_factor
                            )
                            self._set_last_ss_pos(self.generated_input_ids.shape[1])

            else:
                # Speaking state 's'
                if self.last_ss_pos is None:
                    self._set_last_ss_pos(max(self.system_input_length + 1, int(self.generated_input_ids.shape[1])))
                stoken_mapping_len = self.stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.fw.stoken_delay_num + 1:
                    warmup_len = max(0, min(target_new_length, self.fw.stoken_delay_num + 1 - stoken_mapping_len))
                    if warmup_len > 0:
                        self.stoken_mapping = torch.cat(
                            [self.stoken_mapping, torch.full((1, warmup_len), -1, dtype=torch.long, device=self.fw.device)],
                            dim=1,
                        )
                if self.stoken_mapping.shape[1] - self.generated_input_ids.shape[1] < target_new_length:
                    padding_stoken_mapping = []
                    for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + target_new_length):
                        seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                        padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
                    padding_stoken_mapping = torch.tensor(
                        padding_stoken_mapping, dtype=torch.long, device=self.fw.model.device
                    ).unsqueeze(0)
                    self.stoken_mapping = torch.cat((self.stoken_mapping, padding_stoken_mapping), dim=1)

                speaking_control_processor = LogitsProcessorList([
                    SpeakingControlLogitsProcessor(
                        sleep_token_id=self.fw.sleep_token_id,
                        detect_token_id=self.fw.detect_token_id,
                        sl_token_id=self.fw.sl_token_id,
                        ks_token_id=self.fw.ks_token_id,
                        vocab_size=self.fw.model.config.text_config.vocab_size,
                        prefix_input_len=self.prefix_input_ids.shape[1],
                        control_token_chunk_size=self.fw.control_token_chunk_size,
                        start_listen_token_factor=self.start_listen_token_factor,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None

                last_result = None
                early_s2l_interrupt = False
                t_model_s_start = time.perf_counter()
                model_s_start_epoch_ms = int(time.time() * 1000)

                def _speaking_control_early(control_event, *, chunk_pos=cn_e):
                    try:
                        token_control = int(control_event.get("control_token"))
                        step = int(control_event.get("step", -1))
                    except (TypeError, ValueError):
                        return
                    if token_control != int(self.fw.sl_token_id):
                        return
                    _emit_control_early(
                        {
                            "type": "control_head_pending",
                            "from": "S",
                            "to": "L",
                            "pos": int(chunk_pos),
                            "chunk": int(chunk_pos),
                            "reason": "control_sl",
                            "early_exit": True,
                            "control_head_early": True,
                            "pending": True,
                            "requires_stoken_confirm": True,
                            "control_token": int(token_control),
                            "step": int(step),
                            "sl_prob": _safe_prob(speaking_control_core, "sl"),
                            "ss_prob": _safe_prob(speaking_control_core, "ss"),
                            "ks_prob": _safe_prob(speaking_control_core, "ks"),
                            "model_control_epoch_ms": control_event.get("timestamp_epoch_ms"),
                        }
                    )

                for token_result in self.fw.model.multi_head_generate_stream(
                    input_ids=self.generated_input_ids,
                    stoken_ids=self.generated_stoken_ids,
                    control_input_ids=self.generated_control_ids,
                    prefix_input_ids=self.prefix_input_ids,
                    stoken_mapping=self.stoken_mapping,
                    audio_input_ids=audio_input_ids,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    audio_embeds=audio_embeds,
                    pre_computed_audio=pre_computed_audio,
                    past_key_values=self.past_key_values,
                    stoken_past_key_values=self.stoken_past_key_values,
                    control_past_key_values=self.control_past_key_values,
                    use_cache=self.fw.use_cache and not self.fw.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor,
                    logits_processor=self.speaking_text_processor,
                    stoken_logits_processor=self.speaking_stoken_processor,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.fw.tts_end_id,
                    keep_request_alive=self.keep_alive_for_speaking,
                    control_early_callback=_speaking_control_early,
                ):
                    last_result = token_result
                    token_control = int(token_result["control_token"])
                    token_stoken = int(token_result["stoken_token"])
                    if (
                        _LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED
                        and token_control == int(self.fw.sl_token_id)
                        and token_stoken != int(self.fw.tts_end_id)
                    ):
                        early_s2l_interrupt = True
                        if _LYCHEEFD_CONTROL_EARLY_DEBUG:
                            print(
                                "[CONTROL_EARLY_EXIT] S->L interrupt "
                                f"chunk={cn_e} step={token_result.get('step')} "
                                f"control={token_control} stoken={token_stoken}"
                            )
                        break
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                t_model_s_cost = time.perf_counter() - t_model_s_start
                self._record_timeline_span(
                    "transformer",
                    t_model_s_start,
                    model_s_start_epoch_ms,
                    state="speaking",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=int(target_new_length),
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event
                self.profile_model["s_calls"] += 1
                self.profile_model["s_sec"] += t_model_s_cost

                if last_result is None:
                    last_result = {
                        "sequences": self.generated_input_ids,
                        "stoken_ids": self.generated_stoken_ids,
                        "control_ids": self.generated_control_ids,
                        "past_key_values": self.past_key_values,
                        "stoken_past_key_values": self.stoken_past_key_values,
                        "control_past_key_values": self.control_past_key_values,
                    }
                s_step_cnt = int(last_result["step"] + 1) if "step" in last_result else 0
                self.profile_model["s_tokens"] += max(0, s_step_cnt)

                self.past_key_values = last_result["past_key_values"]
                self.stoken_past_key_values = last_result["stoken_past_key_values"]
                self.control_past_key_values = last_result["control_past_key_values"]
                self.generated_input_ids = last_result["sequences"]
                self.generated_stoken_ids = last_result["stoken_ids"]
                self.generated_control_ids = last_result["control_ids"]

                if early_s2l_interrupt:
                    yield {
                        "type": "state_change",
                        "from": "S",
                        "to": "L",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_sl",
                        "early_exit": True,
                        "interrupt": True,
                        "interrupt_reason": "control_sl_without_tts_end",
                        "sl_prob": _safe_prob(speaking_control_core, "sl"),
                        "ss_prob": _safe_prob(speaking_control_core, "ss"),
                        "ks_prob": _safe_prob(speaking_control_core, "ks"),
                    }
                    self.listening_state = "l"
                    self._set_last_ss_pos(None)
                    self.generated_input_ids, self.generated_stoken_ids = self.fw._apply_s2l_tail_fallback(
                        self.generated_input_ids,
                        self.generated_stoken_ids,
                    )
                elif self.generated_stoken_ids[0, -1] == self.fw.tts_end_id:
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > self.generated_input_ids.shape[1] - self.system_input_length:
                        padding_len = target_length - self.generated_input_ids.shape[1] + self.system_input_length
                    else:
                        padding_len = self.fw.control_token_chunk_size
                        stoken_mapping_len = self.stoken_mapping.shape[1] - self.last_ss_pos
                        if stoken_mapping_len <= self.fw.stoken_delay_num + 1:
                            self.stoken_mapping = torch.cat(
                                [
                                    self.stoken_mapping,
                                    torch.full(
                                        (1, min(padding_len, self.fw.stoken_delay_num + 1 - stoken_mapping_len)),
                                        -1,
                                        dtype=torch.long,
                                        device=self.fw.device,
                                    ),
                                ],
                                dim=1,
                            )
                        if self.stoken_mapping.shape[1] - self.generated_input_ids.shape[1] < padding_len:
                            _pad_map = []
                            for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + padding_len):
                                seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                                _pad_map.append(seq_l // 4 + self.last_ss_pos)
                            _pad_map = torch.tensor(_pad_map, dtype=torch.long, device=self.fw.model.device).unsqueeze(0)
                            self.stoken_mapping = torch.cat((self.stoken_mapping, _pad_map), dim=1)

                    padding_input_ids = self.fw.text_pad_token_id * torch.ones(
                        (self.generated_input_ids.shape[0], padding_len),
                        dtype=self.generated_input_ids.dtype,
                        device=self.generated_input_ids.device,
                    )
                    padding_stoken_ids = self.fw.stoken_pad_token_id * torch.ones(
                        (self.generated_input_ids.shape[0], padding_len),
                        dtype=self.generated_input_ids.dtype,
                        device=self.generated_input_ids.device,
                    )
                    padding_control_ids = self.fw.sleep_token_id * torch.ones(
                        (self.generated_input_ids.shape[0], padding_len),
                        dtype=self.generated_input_ids.dtype,
                        device=self.generated_input_ids.device,
                    )
                    if padding_control_ids.shape[1] > 2:
                        padding_control_ids[:, -2] = self.fw.detect_token_id
                    padding_control_ids[:, -1] = self.fw.sl_token_id

                    self.generated_input_ids = torch.cat((self.generated_input_ids, padding_input_ids), dim=1)
                    self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, padding_stoken_ids), dim=1)
                    self.generated_control_ids = torch.cat((self.generated_control_ids, padding_control_ids), dim=1)
                    self.listening_state = "l"
                    self._set_last_ss_pos(None)
                else:
                    pred_control_token = self.generated_control_ids[0, -1]
                    if pred_control_token == self.fw.sl_token_id:
                        yield {
                            "type": "state_change",
                            "from": "S",
                            "to": "L",
                            "pos": cn_e,
                            "chunk": cn_e,
                            "reason": "control_sl",
                            "early_exit": False,
                            "interrupt": False,
                            "sl_prob": _safe_prob(speaking_control_core, "sl"),
                            "ss_prob": _safe_prob(speaking_control_core, "ss"),
                            "ks_prob": _safe_prob(speaking_control_core, "ks"),
                        }
                        self.listening_state = "l"
                        self._set_last_ss_pos(None)
                        self.generated_input_ids, self.generated_stoken_ids = self.fw._apply_s2l_tail_fallback(
                            self.generated_input_ids,
                            self.generated_stoken_ids,
                        )

            yield {"type": "chunk_end", "chunk_idx": self.chunk_idx}
            self.chunk_idx += 1
            self.next_chunk_pos = cn_e + self.fw.control_token_chunk_size

        if emit_generation_complete:
            yield self.build_generation_complete_event()


class HFIncrementalChunkStreamSession(IncrementalChunkStreamSession):
    """
    Incremental realtime session variant for HF-style debugging.

    This explicitly disables request keep-alive during speaking so each
    speaking decode call behaves like a standalone HF forward/generate step.
    """

    def __init__(
        self,
        framework,
        prefix=None,
        initial_listening_state="l",
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        super().__init__(
            framework=framework,
            prefix=prefix,
            initial_listening_state=initial_listening_state,
            start_speak_token_factor=start_speak_token_factor,
            start_listen_token_factor=start_listen_token_factor,
            bc_speak_token_factor=bc_speak_token_factor,
            end_speak_token_factor=end_speak_token_factor,
            audio_incremental_mode=audio_incremental_mode,
        )
        self.keep_alive_for_speaking = False
        self.keep_alive_for_listening = False


class StreamingDecoder:
    """
    Incremental decoder that consumes token-level streaming events from
    full_chunk_stream_generation() and emits higher-level events suitable
    for driving a real-time UI / audio pipeline.

    Emitted event types:
      - text_delta:   {"type": "text_delta", "text": str, "delta": str,
                       "snapshot": str, "event_id": str, "seq": int, "token_id": int}
      - audio_chunk:  {"type": "audio_chunk", "stoken_ids": list[int]}
                      (flushed every `tts_chunk_size` valid speech tokens)
      - state_change: {"type": "state_change", ...}  (passthrough)
      - event_start:  {"type": "event_start", "event_kind": "response"|"backchannel",
                       "event_id": str}
      - event_end:    {"type": "event_end", "event_kind": "response"|"backchannel",
                       "event_id": str, "text": str, "snapshot": str, "seq": int,
                       "stoken_ids": list[int]}
    """

    def __init__(self, tokenizer, framework, tts_chunk_size=25, end_event_on_generation_complete=True):
        self.tokenizer = tokenizer
        self.fw = framework
        self.tts_chunk_size = tts_chunk_size
        self._end_event_on_generation_complete = bool(end_event_on_generation_complete)

        self._text_buf: List[int] = []
        self._stoken_buf: List[int] = []
        self._stoken_flush_buf: List[int] = []
        self._state = "l"
        self._event_kind = None
        self._event_id = None
        self._event_counter = 0
        self._text_seq = 0
        self._prev_decoded_len = 0

    def _is_valid_text_token(self, tid):
        return (tid < 151688
                and tid != self.fw.text_pad_token_id
                and tid != self.fw.tts_pad_id)

    def _is_valid_stoken(self, sid):
        return (sid > 151695
                and sid != self.fw.stoken_delay_token_id
                and sid != self.fw.stoken_pad_token_id)

    def _decode_valid_text_snapshot(self):
        valid_text = [t for t in self._text_buf if self._is_valid_text_token(t)]
        if not valid_text:
            return ""
        return self.tokenizer.decode(valid_text, skip_special_tokens=True)

    def _flush_text_delta(self):
        if not self._text_buf:
            return None
        decoded = self._decode_valid_text_snapshot()
        if len(decoded) > self._prev_decoded_len:
            delta = decoded[self._prev_decoded_len:]
            self._prev_decoded_len = len(decoded)
            self._text_seq += 1
            return {
                "type": "text_delta",
                "text": delta,
                "delta": delta,
                "snapshot": decoded,
                "event_id": self._event_id,
                "event_kind": self._event_kind or "response",
                "seq": int(self._text_seq),
                "token_id": self._text_buf[-1],
            }
        return None

    def _flush_stoken_chunk(self, force=False):
        if not self._stoken_flush_buf:
            return None
        if force or len(self._stoken_flush_buf) >= self.tts_chunk_size:
            chunk = list(self._stoken_flush_buf[:self.tts_chunk_size])
            self._stoken_flush_buf = self._stoken_flush_buf[self.tts_chunk_size:]
            return {"type": "audio_chunk", "stoken_ids": chunk}
        return None

    def _start_event(self, kind):
        self._event_kind = kind
        self._event_counter += 1
        self._event_id = f"evt-{self._event_counter}"
        self._text_buf = []
        self._stoken_buf = []
        self._stoken_flush_buf = []
        self._text_seq = 0
        self._prev_decoded_len = 0
        return {"type": "event_start", "event_kind": kind, "event_id": self._event_id}

    def _end_event(self, *, interrupt=False, interrupt_reason=None):
        events = []
        if interrupt:
            self._stoken_flush_buf = []
        else:
            remaining = self._flush_stoken_chunk(force=True)
            while remaining:
                events.append(remaining)
                remaining = self._flush_stoken_chunk(force=True)

        full_text = self._decode_valid_text_snapshot()

        events.append({
            "type": "event_end",
            "event_kind": self._event_kind or "response",
            "event_id": self._event_id,
            "text": full_text,
            "snapshot": full_text,
            "seq": int(self._text_seq) + 1,
            "stoken_ids": [s - 151696 for s in self._stoken_buf if self._is_valid_stoken(s)],
            "interrupt": bool(interrupt),
            "interrupt_reason": interrupt_reason,
        })
        self._event_kind = None
        self._event_id = None
        return events

    def feed(self, stream_event):
        """
        Feed a raw event from full_chunk_stream_generation() and return a
        list of decoded events (may be empty).
        """
        results = []
        etype = stream_event.get("type")

        if etype == "state_change":
            from_st = stream_event["from"]
            to_st = stream_event["to"]
            is_interrupt = bool(stream_event.get("interrupt", False))
            interrupt_reason = stream_event.get("interrupt_reason")

            if self._state in ("s", "b") and to_st == "L":
                results.extend(
                    self._end_event(
                        interrupt=is_interrupt,
                        interrupt_reason=interrupt_reason,
                    )
                )
            if from_st == "L" and to_st in ("S", "B"):
                kind = "response" if to_st == "S" else "backchannel"
                results.append(self._start_event(kind))
            elif from_st == "B" and to_st == "S":
                results.extend(self._end_event())
                results.append(self._start_event("response"))

            self._state = to_st.lower()
            results.append(stream_event)

        elif etype == "speaking_token":
            tid = stream_event["text_token"]
            sid = stream_event["stoken"]

            self._text_buf.append(tid)
            self._stoken_buf.append(sid)

            td = self._flush_text_delta()
            if td:
                results.append(td)

            if self._is_valid_stoken(sid):
                self._stoken_flush_buf.append(sid - 151696)
            sc = self._flush_stoken_chunk()
            if sc:
                results.append(sc)

        elif etype == "speaking_done":
            results.extend(self._end_event())
            self._state = "l"

        elif etype == "generation_complete":
            if (
                self._end_event_on_generation_complete
                and self._state in ("s", "b")
                and self._event_kind is not None
            ):
                results.extend(self._end_event())
            results.append(stream_event)

        elif etype in ("chunk_start", "chunk_end", "control_decision"):
            results.append(stream_event)

        return results
