import os
import torch
import torch.nn.functional as F
import argparse
import librosa
import datasets
import numpy as np
import math
import json
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
from peft import PeftModel
import copy
import torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer, NoRepeatNGramLogitsProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import AudioInput
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    IS_CUDA = False
except:
    IS_CUDA = True

SYSTEM_MESSAGE_PREFIX = "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


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


def _mel_filters(n_mels: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram."""
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
    def __init__(
        self, 
        ss_token_id, 
        kl_token_id, 
        bc_token_id,
        vocab_size,
        start_speak_token_factor=1.0,
        bc_speak_token_factor=1.0,
    ):
        if bc_token_id is not None:
            self.not_listening_tokens = list(set(range(vocab_size)) - set([ss_token_id, kl_token_id, bc_token_id])) 
        else:
            self.not_listening_tokens = list(set(range(vocab_size)) - set([ss_token_id, kl_token_id])) 
        self.start_speak_token_factor = start_speak_token_factor
        self.bc_speak_token_factor = bc_speak_token_factor
        self.ss_token_id = ss_token_id
        self.bc_token_id = bc_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        # 取每个 batch 的最后一个 token
        last_token_ids = input_ids[:, -1]

        for batch_idx, last_token in enumerate(last_token_ids):
            scores[batch_idx, self.not_listening_tokens] = -float("inf")
            scores[batch_idx, self.ss_token_id] *= self.start_speak_token_factor
            if self.bc_token_id is not None:
                scores[batch_idx, self.bc_token_id] *= self.bc_speak_token_factor

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
    ):

        self.not_sleep_tokens = list(set(range(vocab_size)) - set([sleep_token_id])) 
        self.not_detect_tokens = list(set(range(vocab_size)) - set([detect_token_id])) 
        self.not_speaking_tokens = list(set(range(vocab_size)) - set([sl_token_id, ks_token_id])) 
        self.sl_token_id = sl_token_id
        self.ks_token_id = ks_token_id

        self.prefix_input_len = prefix_input_len
        self.control_token_chunk_size = control_token_chunk_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        # 取每个 batch 的最后一个 token
        last_token_ids = input_ids[:, -1]
        # 注意: multi_head_generate 给 control_logits_processor 传入的 input_ids 实际是
        # generated_control_ids, 它从初始化起就不含 prefix(初始仅 1 个 sleep_token), 因此
        # 这里直接对 shape[1] 取模即可, 不需要再减 prefix_input_len.
        # pos = (input_ids.shape[1] - prefix_input_len) % self.control_token_chunk_size
        pos = input_ids.shape[1] % self.control_token_chunk_size
        for batch_idx, last_token in enumerate(last_token_ids):
            if pos == self.control_token_chunk_size - 2:
                scores[batch_idx, self.not_detect_tokens] = -float("inf")
            elif pos == self.control_token_chunk_size - 1:
                scores[batch_idx, self.not_speaking_tokens] = -float("inf")
            else:
                scores[batch_idx, self.not_sleep_tokens] = -float("inf")

        return scores

class BackChannelLogitsProcessor(LogitsProcessor):
    def __init__(
        self, 
        sleep_token_id,
        detect_token_id,
        end_bc_token_id, 
        ss_token_id, 
        keep_bc_token_id,
        vocab_size,
        prefix_input_len,
        control_token_chunk_size,   
        start_speak_token_factor=1, 
    ):
        # BC 状态 chunk 末位的合法 control token 是「点对点」集合:
        #   - ss(start_speaking) : BC -> S(打断)
        #   - end_bc             : BC -> L(正常结束)
        #   - keep_bc            : 维持 BC
        # keep_bc=None 时(强制重开 chunk 的二次决策)只在 {ss, end_bc} 中选。
        self.not_sleep_tokens       = list(set(range(vocab_size)) - set([sleep_token_id])) 
        self.not_detect_tokens      = list(set(range(vocab_size)) - set([detect_token_id])) 
        if keep_bc_token_id is not None:
            self.not_speaking_tokens = list(set(range(vocab_size)) - set([ss_token_id, end_bc_token_id, keep_bc_token_id]))
        else:
            self.not_speaking_tokens = list(set(range(vocab_size)) - set([ss_token_id, end_bc_token_id]))
        
        self.ss_token_id = ss_token_id
        self.end_bc_token_id = end_bc_token_id
        self.keep_bc_token_id = keep_bc_token_id
        self.start_speak_token_factor = start_speak_token_factor
        self.prefix_input_len = prefix_input_len
        self.control_token_chunk_size = control_token_chunk_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, seq_len]
        scores: [batch_size, vocab_size]
        """
        # 取每个 batch 的最后一个 token
        last_token_ids = input_ids[:, -1]
        pos = input_ids.shape[1] % self.control_token_chunk_size
        for batch_idx, last_token in enumerate(last_token_ids):
            if pos == self.control_token_chunk_size - 2:
                scores[batch_idx, self.not_detect_tokens] = -float("inf")
            elif pos == self.control_token_chunk_size - 1:
                scores[batch_idx, self.not_speaking_tokens] = -float("inf")
                scores[batch_idx, self.ss_token_id] *= self.start_speak_token_factor
            else:
                scores[batch_idx, self.not_sleep_tokens] = -float("inf")

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
        self.not_speaking_token = [i for i in range(vocab_size) if ((text_start is not None and i < text_start) or (text_end is not None and i >= text_end)) and i != eos_token_id]
        self.not_eos_tokens = list(set(range(vocab_size)) - set([eos_token_id]))
        if text_pad_token_id is not None:
            self.not_text_pad_tokens =  list(set(range(vocab_size)) - set([text_pad_token_id]))
        else:
            self.not_text_pad_tokens = None

        self.prefix_token_id = prefix_token_id
        self.prefix_cnt =0

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
        # 取每个 batch 的最后一个 token
        last_token_ids = input_ids[:, -1]

        if self.prefix_token_id is not None:
            self.prefix_cnt += 1
            if self.prefix_cnt <= len(self.prefix_token_id):
                not_prefix_tokens = list(set(range(self.vocab_size)) - set([self.prefix_token_id[self.prefix_cnt - 1]]))
                scores[:, not_prefix_tokens] = -float("inf") 
                return scores
        
        if self.max_token_length is not None and not self.has_eos:
            self.cnt += 1
            if self.cnt >= self.max_token_length:
                scores[:, self.not_eos_tokens] = -float("inf") 
                return scores

        # if self.not_text_pad_tokens is None:
        #     for batch_idx, last_token in enumerate(last_token_ids):
        #         if last_token == self.eos_token_id:
        #            self.has_eos = True
        #         scores[batch_idx, self.not_speaking_token] = -float("inf") 
        #         scores[batch_idx, self.eos_token_id] *= self.end_speak_token_factor      
        # else:
        #     for batch_idx, last_token in enumerate(last_token_ids):
        #         if last_token == self.eos_token_id:
        #             self.has_eos = True
        #         if not self.has_eos:
        #             scores[batch_idx, self.not_speaking_token] = -float("inf")
        #             scores[batch_idx, self.eos_token_id] *= self.end_speak_token_factor
        #         else:
        #             scores[batch_idx, self.not_text_pad_tokens] = -float("inf")
        
        for batch_idx, last_token in enumerate(last_token_ids):
            if last_token == self.eos_token_id:
               self.has_eos = True
            if self.has_eos and self.not_text_pad_tokens is not None:
                scores[batch_idx, self.not_text_pad_tokens] = -float("inf")
            else:
                scores[batch_idx, self.not_speaking_token] = -float("inf") 
                scores[batch_idx, self.eos_token_id] *= self.end_speak_token_factor  
        
        if self.ngram_size_processor is not None:
            scores = self.ngram_size_processor(input_ids, scores)

        return scores


class SingleTurnGenerationFramework:

    TOKENS_PER_SECOND = 25

    MAX_SPEECH_TOKEN_NUM = 1000

    MAX_TEXT_TOKEN_NUM = 128

    def __init__(
        self,
        model_type,
        model_path,
        device = "cuda",
        attn_implementation= "eager",
        torch_dtype=torch.float,
        align_audio_input=False,
        allowing_backchannel=True,
        is_old_sl_strategy=True,
        zero_checkpoint_dir=None,
    ):
        self.device = torch.device(device)

        # 根据 model_type 选择模型实现:
        #   V9: 含 merge_model, 需要维护 merge_past_key_values
        #   V8: 不含 merge_model (旧逻辑)
        assert model_type == "V9"
        from lychee_fd.models.lychee_duplex import LycheeDuplexConfig, LycheeDuplexForCausalLM

        if zero_checkpoint_dir is None:
            self.model = LycheeDuplexForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
            self.model.stream_generation_flag = True
        else:
            # === DeepSpeed ZeRO-3 加载路径 ===
            # 训练阶段保存的 .safetensors shape 全为 0, 真正参数在
            # {zero_checkpoint_dir}/global_step*/zero_pp_rank_*_mp_rank_*_model_states.pt
            # 中按 rank 切片. 这里:
            #   1) 用 model_path 下的 config.json 建一个空壳模型 (随机初始化, 但权重 shape 正确)
            #   2) 调用 deepspeed 官方 API 把所有 rank 的切片合并成完整 fp32 state_dict
            #   3) load_state_dict(strict=True) 灌入权重, 再转目标 dtype / device
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

            print(f"[ZeRO-3] building empty model from config: {model_path}")
            cfg = LycheeDuplexConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            # 设置 attn_implementation, 让 _from_config 走对应的 attn 分支
            cfg._attn_implementation = attn_implementation
            # 注意: 不要传 torch_dtype 给 _from_config, 否则会按 cfg.torch_dtype 走;
            # 我们先在 fp32 下建壳, 加载完整权重后再统一 .to(torch_dtype).
            self.model = LycheeDuplexForCausalLM._from_config(
                cfg,
                attn_implementation=attn_implementation,
            )

            print(f"[ZeRO-3] merging fp32 state_dict from: {zero_checkpoint_dir}")
            fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(zero_checkpoint_dir)
            # DeepSpeed 已自动剥离 'module.' 前缀, key 与 model.state_dict() 对齐
            missing, unexpected = self.model.load_state_dict(fp32_state_dict, strict=False)
            # 硬性校验: 对于纯 ZeRO-3 (无 PEFT/LoRA), 不应该存在缺失或多余的 key
            assert len(missing) == 0, f"[ZeRO-3] missing keys when loading: {missing[:10]} ... (total {len(missing)})"
            assert len(unexpected) == 0, f"[ZeRO-3] unexpected keys when loading: {unexpected[:10]} ... (total {len(unexpected)})"
            print(f"[ZeRO-3] load_state_dict done. params loaded: {len(fp32_state_dict)}")
            del fp32_state_dict

            # 转目标 dtype 与 device
            self.model = self.model.to(dtype=torch_dtype, device=self.device).eval()
            self.model.stream_generation_flag = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        self.kl_token_id            = self.model.config.keep_listening_token_id
        self.ss_token_id            = self.model.config.start_speaking_token_id
        self.sl_token_id            = self.model.config.start_listening_token_id
        self.ks_token_id            = self.model.config.keep_speaking_token_id
        self.start_bc_token_id      = self.model.config.start_bc_token_id
        self.keep_bc_token_id       = getattr(self.model.config, "keep_bc_token_id", None) or self.model.config.start_bc_token_id
        self.end_bc_token_id        = getattr(self.model.config, "end_bc_token_id", None) or self.model.config.start_listening_token_id
        # bc_token_id 专指 L->BC 入口 token(== start_bc), 仅用于 listening 状态
        self.bc_token_id            = self.start_bc_token_id
        self.detect_token_id        = self.model.config.detect_token_id
        self.sleep_token_id         = self.model.config.sleep_token_id
        self.text_pad_token_id      = self.model.config.text_pad_token_id
        self.stoken_pad_token_id    = self.model.config.stoken_pad_token_id
        self.audio_pad_token_id     = self.model.config.audio_pad_token_id
        self.stoken_delay_token_id  = self.model.config.stoken_delay_token_id
        self.adding_text_hiddenstates = self.model.config.adding_text_hiddenstates

        self.align_audio_input      = align_audio_input
        self.allowing_backchannel   = allowing_backchannel    

        self.audio_token_id     = self.tokenizer(["<audio_patch>"]).input_ids[0][0]
        self.tts_pad_id         = self.tokenizer(["<tts_pad>"]).input_ids[0][0]
        self.tts_end_id         = self.tokenizer(["<tts_end>"]).input_ids[0][0]
        self.tts_start_id       = self.tokenizer(["<tts_start>"]).input_ids[0][0]
        self.eos_token_id       = self.tokenizer(["<|EOT|>"]).input_ids[0][0]
        self.control_token_chunk_size = self.model.config.control_token_chunk_size

        self.window_second = self.control_token_chunk_size * 0.04
        self.stoken_delay_num = self.model.config.stoken_delay_num
        self.stoken_no_repeat_n_gram = 4
        print(f"[SETTING] control_token_chunk_size={self.control_token_chunk_size}")
        print(f"[SETTING] window_second={self.window_second}")
        print(f"[SETTING] stoken_delay_num={self.stoken_delay_num}")
        print(f"[SETTING] stoken_no_repeat_n_gram={self.stoken_no_repeat_n_gram}")

        self.sampling_rate = 16000

        self.last_ss_pos = None

        self.is_old_sl_strategy = is_old_sl_strategy

        self.use_cache = True

    def init_speaking_processor(self, end_speak_token_factor):
        # 说的时候，限制 Text 通道 (处理 EOS 等)
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

        # 说的时候，限制 Stoken 通道 (处理 EOS 等)
        speaking_stoken_processor = LogitsProcessorList([
            SpeakingLogitsProcessor(
                text_start=151696, 
                text_end=158352, 
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

    def stepaudio_audio_prepreocess(
        self,
        audio = None,
        debug=False,
    ) -> BatchFeature:

        feats = []
        feats_lengths = []
        input_ids = []
        # V8: 使用 int() 避免 float range 问题
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
        start_speak_token_factor=1,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        return_raw_trace=False,
    ):
        # 1. 基础输入准备
        prefix_input_ids = self.tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
        system_input_length = len(prefix_input_ids)

        # 构造初始的 Prompt
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

        # 2. 音频处理
        model_inputs = self.stepaudio_audio_prepreocess(audio)
        feats, feats_lengths = model_inputs["feats"], model_inputs["feats_lengths"]
        wavs = torch.nn.utils.rnn.pad_sequence(
            feats,
            batch_first=True,
            padding_value=0).transpose(1, 2).to(self.device)
        wav_lens = torch.tensor(feats_lengths, dtype=torch.int32).to(self.device)
        
        # 音频总长度对应的 token 数 (用于计算 chunk)
        audio_input_ids = torch.tensor(model_inputs["input_ids"]).unsqueeze(0).to(self.device) # 注意：这里假设 preprocess 返回的是 token 长度
        
        # 3. 初始化状态
        past_key_values = None # 统一的 KV Cache
        stoken_past_key_values = None # 统一的 KV Cache
        control_past_key_values = None # 统一的 KV Cache
        merge_past_key_values = None # merge_model 的 KV Cache (V9 含 merge_model 时使用, 否则保持 None)
        listening_state = initial_listening_state
        

        # 计算 Chunk 范围
        # 注意：model_inputs["input_ids"] 是音频对应的文本长度模拟
        total_len = len(model_inputs["input_ids"])
        chunk_start = math.ceil((input_ids.shape[1] - system_input_length + 1) / self.control_token_chunk_size) * self.control_token_chunk_size
        chunk_end = math.ceil(total_len / self.control_token_chunk_size) * self.control_token_chunk_size

        # 维护生成的历史序列
        generated_input_ids = input_ids
        generated_stoken_ids = stoken_ids
        generated_control_ids = control_input_ids

        for cn_e in range(chunk_start, chunk_end + 1, self.control_token_chunk_size):
            
            target_length = min(cn_e, total_len)
            current_len = generated_input_ids.shape[1] - system_input_length
            target_new_length = target_length - current_len

            if listening_state == 'l':
                # === 听的状态 ===
                # 构造 Padding 输入
                # Text: Pad, Stoken: Pad, Control: Sleep...Detect
                
                if target_new_length > 1:
                    next_input_ids = torch.full((1, target_new_length-1), self.text_pad_token_id, dtype=torch.long, device=self.device)
                    next_stoken_ids = torch.full((1, target_new_length-1), self.stoken_pad_token_id, dtype=torch.long, device=self.device)
                    next_control_ids = torch.full((1, target_new_length-1), self.sleep_token_id, dtype=torch.long, device=self.device)
                
                    # 在 Padding 的最后一位放入 Detect Token (除非是完全结束)
                    if target_length % self.control_token_chunk_size == 0:
                        next_control_ids[0, -1] = self.detect_token_id

                    generated_input_ids = torch.cat([generated_input_ids, next_input_ids], dim=1)
                    generated_stoken_ids = torch.cat([generated_stoken_ids, next_stoken_ids], dim=1)
                    generated_control_ids = torch.cat([generated_control_ids, next_control_ids], dim=1)
                    
                    stoken_mapping = torch.cat([stoken_mapping, torch.full((1, target_new_length), -1, dtype=torch.long, device=self.device)], dim=1)
                    
                    listening_control_processor = LogitsProcessorList([
                        ListeningControlLogitsProcessor(
                            ss_token_id=self.ss_token_id, 
                            kl_token_id=self.kl_token_id, 
                            bc_token_id=self.bc_token_id if self.allowing_backchannel else None,
                            vocab_size=self.model.config.text_config.vocab_size,
                            start_speak_token_factor=start_speak_token_factor,
                            bc_speak_token_factor=bc_speak_token_factor,
                        )
                    ])

                # 调用 Generate
                # 目标：把 Padding 喂进去更新 Cache，并生成 **1个** 新的 Control Token 来决定状态
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
                    merge_past_key_values=merge_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=1, # 只生成 1 步决策
                    control_logits_processor=listening_control_processor, # 限制 Control 输出
                    temperature=1.0, # 听的时候通常贪婪采样
                    top_k=0,
                    eos_token_id=None,                 
                    pad_token_id=None,
                )

                # 更新状态
                past_key_values = outputs["past_key_values"]
                stoken_past_key_values = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                merge_past_key_values = outputs["merge_past_key_values"]
                
                # 获取生成的 1 个 token
                pred_control_token = outputs["control_ids"][0, -1]

                padding_input_ids = self.text_pad_token_id * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                padding_stoken_ids = self.stoken_pad_token_id * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)

                generated_input_ids = torch.cat((generated_input_ids, padding_input_ids), dim=1)
                generated_stoken_ids = torch.cat((generated_stoken_ids, padding_stoken_ids), dim=1)

                if pred_control_token == self.ss_token_id:
                    print(f"[L]->[S] {cn_e}/{chunk_end}")
                    padding_control_input_ids = self.ss_token_id * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    listening_state = "s"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                elif pred_control_token == self.bc_token_id:
                    print(f"[L]->[BC] {cn_e}/{chunk_end}")
                    padding_control_input_ids = self.bc_token_id * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                    listening_state = "b"
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                else:
                    assert pred_control_token == self.kl_token_id or cn_e == chunk_end
                    padding_control_input_ids = self.kl_token_id * torch.ones((input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                generated_control_ids = torch.cat((generated_control_ids, padding_control_input_ids), dim=1)

            elif listening_state == 'b':
                # 计算stoken_mapping
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
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
                        end_bc_token_id=self.end_bc_token_id, 
                        ss_token_id=self.ss_token_id, 
                        keep_bc_token_id=self.keep_bc_token_id,
                        vocab_size=self.model.config.text_config.vocab_size,
                        prefix_input_len=prefix_input_ids.shape[1],
                        control_token_chunk_size=self.control_token_chunk_size,
                    )
                ])

                # 调用 Generate
                # 目标：生成 target_new_length 个 token
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
                    merge_past_key_values=merge_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor, 
                    logits_processor=speaking_text_processor, 
                    stoken_logits_processor=speaking_stoken_processor, 
                    do_sample=True,
                    temperature=0.7, # 说话时需要采样
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id
                )

                past_key_values         = outputs["past_key_values"]
                stoken_past_key_values  = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                merge_past_key_values   = outputs["merge_past_key_values"]
                generated_input_ids     = outputs["sequences"]
                generated_stoken_ids    = outputs["stoken_ids"]
                generated_control_ids   = outputs["control_ids"]

                
                if generated_control_ids[0, -1] == self.end_bc_token_id:
                    print(f"[BC]->[L] {cn_e}/{chunk_end}")
                    listening_state = 'l'
                    self.last_ss_pos = None
                    if self.is_old_sl_strategy:
                        generated_stoken_ids[0, -1] = self.stoken_pad_token_id
                        generated_input_ids[0, -1]  = self.text_pad_token_id
                    else:
                        assert generated_stoken_ids[0, -1] != self.stoken_pad_token_id
                        generated_stoken_ids[0, -1] = self.stoken_pad_token_id if generated_stoken_ids[0, -1] == self.stoken_pad_token_id else self.tts_end_id
                        if generated_input_ids[0, -1] not in [self.text_pad_token_id, self.tts_pad_id]:
                            generated_input_ids[0, -1] = self.eos_token_id
                elif generated_control_ids[0, -1] == self.ss_token_id:
                    print(f"[BC]->[S] {cn_e}/{chunk_end}")
                    listening_state = 's'
                    speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                    self.last_ss_pos = generated_input_ids.shape[1]
                    if self.is_old_sl_strategy:
                        generated_stoken_ids[0, -1] = self.stoken_pad_token_id
                        generated_input_ids[0, -1]  = self.text_pad_token_id
                    else:
                        assert generated_stoken_ids[0, -1] != self.stoken_pad_token_id
                        generated_stoken_ids[0, -1] = self.stoken_pad_token_id if generated_stoken_ids[0, -1] == self.stoken_pad_token_id else self.tts_end_id
                        if generated_input_ids[0, -1] not in [self.text_pad_token_id, self.tts_pad_id]:
                            generated_input_ids[0, -1] = self.eos_token_id
                elif generated_control_ids[0, -1] == self.keep_bc_token_id or cn_e == chunk_end:
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
                        padding_control_ids[:, -1] = self.end_bc_token_id

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
                                end_bc_token_id=self.end_bc_token_id, 
                                ss_token_id=self.ss_token_id, 
                                keep_bc_token_id=None,
                                vocab_size=self.model.config.text_config.vocab_size,
                                prefix_input_len=prefix_input_ids.shape[1],
                                control_token_chunk_size=self.control_token_chunk_size,
                                start_speak_token_factor=start_speak_token_factor,
                            )
                        ])

                        # 调用 Generate
                        # 目标：生成 target_new_length 个 token
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
                            merge_past_key_values=merge_past_key_values,
                            use_cache=self.use_cache and not self.adding_text_hiddenstates,
                            max_new_tokens=1,
                            control_logits_processor=speaking_control_processor, 
                            logits_processor=None, 
                            stoken_logits_processor=None, 
                        )

                        past_key_values         = outputs["past_key_values"]
                        stoken_past_key_values  = outputs["stoken_past_key_values"]
                        control_past_key_values = outputs["control_past_key_values"]
                        merge_past_key_values   = outputs["merge_past_key_values"]
                        
                        generated_control_ids   = outputs["control_ids"]

                        pred_control_token = generated_control_ids[0,-1]
                        if pred_control_token == self.end_bc_token_id:
                            print("[BC]->[L]")
                            listening_state = 'l'
                            self.last_ss_pos = None
                        else:
                            assert generated_control_ids[0, -1] == self.ss_token_id
                            print("[BC]->[S]")
                            listening_state = 's'
                            speaking_text_processor, speaking_stoken_processor = self.init_speaking_processor(end_speak_token_factor)
                            self.last_ss_pos = generated_input_ids.shape[1] + 1 # 这里要算后面cat的

                        padding_input_ids       = self.text_pad_token_id     * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        padding_stoken_ids      = self.stoken_pad_token_id   * torch.ones((generated_input_ids.shape[0], 1), dtype=generated_input_ids.dtype, device=generated_input_ids.device)
                        generated_input_ids     = torch.cat((generated_input_ids, padding_input_ids[:,:-1]), dim=1)
                        generated_stoken_ids    = torch.cat((generated_stoken_ids, padding_stoken_ids[:,:-1]), dim=1)

            else:
                # 计算stoken_mapping
                stoken_mapping_len = stoken_mapping.shape[1] - self.last_ss_pos
                if stoken_mapping_len <= self.stoken_delay_num + 1:
                    stoken_mapping = torch.cat([stoken_mapping, torch.full((1, min(target_new_length, self.stoken_delay_num + 1 - stoken_mapping_len)), -1, dtype=torch.long, device=self.device)], dim=1)
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
                    )
                ])
                # 调用 Generate
                # 目标：生成 target_new_length 个 token
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
                    merge_past_key_values=merge_past_key_values,
                    use_cache=self.use_cache and not self.adding_text_hiddenstates,
                    max_new_tokens=target_new_length,
                    control_logits_processor=speaking_control_processor, 
                    logits_processor=speaking_text_processor, 
                    stoken_logits_processor=speaking_stoken_processor, 
                    do_sample=True,
                    temperature=0.7, # 说话时需要采样
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.tts_end_id
                )

                past_key_values         = outputs["past_key_values"]
                stoken_past_key_values  = outputs["stoken_past_key_values"]
                control_past_key_values = outputs["control_past_key_values"]
                merge_past_key_values   = outputs["merge_past_key_values"]
                generated_input_ids     = outputs["sequences"]
                generated_stoken_ids    = outputs["stoken_ids"]
                generated_control_ids   = outputs["control_ids"]

                # 提前生成结束, 强制加S-L
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
                        if self.is_old_sl_strategy:
                            generated_stoken_ids[0, -1] = self.stoken_pad_token_id
                            generated_input_ids[0, -1]  = self.text_pad_token_id
                        else:
                            assert generated_stoken_ids[0, -1] != self.stoken_pad_token_id
                            generated_stoken_ids[0, -1] = self.stoken_pad_token_id if generated_stoken_ids[0, -1] == self.stoken_pad_token_id else self.tts_end_id
                            if generated_input_ids[0, -1] not in [self.text_pad_token_id, self.tts_pad_id]:
                                generated_input_ids[0, -1] = self.eos_token_id
                    else:
                        assert pred_control_token == self.ks_token_id or cn_e == chunk_end, f"pred_control_token:{pred_control_token}, {self.tokenizer.decode(generated_control_ids[0])}"

        # 5. 后处理与解码
        # 提取 System Prompt 之后的内容
        generated_input_ids = generated_input_ids[:, system_input_length:][0].cpu().tolist()
        generated_control_ids = generated_control_ids[0].cpu().tolist()
        generated_stoken_ids = generated_stoken_ids[0].cpu().tolist()

        assert len(generated_input_ids) == len(generated_control_ids) == len(generated_stoken_ids)
        
        events = self.decoder(generated_input_ids, generated_stoken_ids, generated_control_ids)
        if return_raw_trace:
            return {
                "events": events,
                "trace": {
                    "generated_input_ids": generated_input_ids,
                    "generated_stoken_ids": generated_stoken_ids,
                    "generated_control_ids": generated_control_ids,
                },
            }

        return events

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


        # 处理模型生成的 B_model 事件
        for pos, (token_id, stoken_id, control_token_id) in enumerate(zip(text, stoken, control)):
            if control_token_id == self.start_bc_token_id and listening_state == 'l':
                # L -> BC 入口(该位 payload 为 idle, 不计入内容)
                utterance_start_pos = pos
                listening_state = 'b'
            elif control_token_id == self.keep_bc_token_id and listening_state == 'b':
                # BC 维持: chunk 末位仍带内容
                text_seq.append(token_id)
                stoken_seq.append(stoken_id)
            elif control_token_id == self.ss_token_id:
                if listening_state == 'b':
                    # BC -> S(打断)
                    utterance_end_pos = pos
                    all_events.append(collection(event_type='backchannel'))
                    text_seq = []
                    stoken_seq = []   
                else:
                    assert listening_state == 'l'
                listening_state = 's'
                utterance_start_pos = pos
            elif control_token_id == self.end_bc_token_id and listening_state == 'b':
                # BC -> L(正常结束)
                utterance_end_pos = pos
                all_events.append(collection(event_type='backchannel'))
                listening_state = 'l'
                text_seq = []
                stoken_seq = []
            elif control_token_id == self.sl_token_id:
                # S -> L
                assert listening_state == 's'
                utterance_end_pos = pos
                all_events.append(collection(event_type='response'))
                listening_state = 'l'
                text_seq = []
                stoken_seq = []                   
            elif listening_state == 's' or listening_state == 'b':
                text_seq.append(token_id)
                stoken_seq.append(stoken_id)
        
        if listening_state == 's' or listening_state == 'b':
            utterance_end_pos = None
            all_events.append(collection(event_type='backchannel' if listening_state == 'b' else 'response'))
    
        return all_events

    # def model_forward(self, audio, golden_ss_token_pos, answer_text=None, answer_ids=None):
    #     model_inputs = qwen_omni_audio_prepreocess(self.processor, audio, return_tensors="pt").to(self.model.device)
    #     input_features, feature_attention_mask = model_inputs["input_features"], model_inputs["feature_attention_mask"]

    #     if answer_text is not None:
    #         input_ids = self.processor.tokenizer([answer_text], add_special_tokens=False).input_ids[0]
    #         input_ids = [kl_token_id] * golden_ss_token_pos + [ss_token_id] + input_ids + [sl_token_id]

    #         input_ids = input_ids + [kl_token_id] * (model_inputs["input_ids"].shape[1] - len(input_ids))
    #         input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
    #     else:
    #         input_ids = answer_ids.to(self.model.device).long().unsqueeze(0)

    #     with torch.no_grad():
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=self.model.device),
    #             input_features=input_features,
    #             feature_attention_mask=feature_attention_mask,
    #             cache_position=torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.model.device),
    #             return_dict=True,                  
    #         )

    #     logits = outputs.logits.cpu()

    #     return logits


"""outputs = self.model(
    input_ids=y["input_ids"].unsqueeze(0).to(self.model.device),
    audio_input_ids=y["audio_input_ids"].unsqueeze(0).to(self.model.device),
    prefix_input_ids=prefix_input_ids,
    attention_mask=torch.ones_like(y["input_ids"].unsqueeze(0), dtype=torch.long, device=self.model.device), # batch size == 1
    input_features=y["input_features"].to(self.model.device),
    feature_attention_mask=y["feature_attention_mask"].to(self.model.device), 
    control_token_mode=True,
)
"""
