import copy
import torch
import torch.nn.functional as F
import argparse
import librosa
import datasets
import numpy as np
import math
import json
import sys
import re
import torchaudio
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from pprint import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence
from peft import PeftModel
import torch
from dataclasses import dataclass, field
import transformers
from transformers.feature_extraction_utils import BatchFeature

from training_utils import rank0_print
from .datasets_utils import debug_print
from .noise_utils import inject_ambient_noise

IGNORE_INDEX = -100

SYSTEM_MESSAGE_PREFIX = "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"

"""
相比V3版本, 将输入的audio token id分为间隔插入
相比V4, TTS END 改为了EOT, 加入tts start 和 tts end, 并且文本的EOT改为文本结尾而非tts_end部分
相比V5, 加入ai bc
"""

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
    Compute the log-Mel spectrogram with specific padding for StepAudio
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
    filters = _mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec




def extract_tagged_text(text, tag):
    """
    提取指定标签中的内容，以及标签前后的文本
    :param text: 原始文本
    :param tag: 标签名，例如 'interruption' 或 'backchannel'
    :return: (before_text, inside_text, after_text)
    """
    # 构造正则模式
    pattern = fr"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)  # DOTALL 让 . 匹配换行
    
    if match:
        inside_text = match.group(1).strip()
        before_text = text[:match.start()].strip()
        after_text = text[match.end():].strip()
        return before_text, inside_text, after_text
    else:
        return text.strip(), None, None  # 没有匹配到标签


class LazySupervisedDataset(Dataset):

    SAMPLE_RATE = 16000

    AUDIO_TOKEN_N_SAMPLE = 1280 # 16000 * 1 / 12.5

    AUDIO_TOKEN_OFFSET = 151696 

    def __init__(
        self,
        tokenizer,
        data_args,
    ):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.data = [datasets.load_from_disk(p) for p in data_args.data_path.split(';')]

        self.data_index = []
        for i, d in enumerate(self.data):
            self.data_index += [(i, x) for x in range(len(d))]
            
        self.control_token_chunk_size = data_args.control_token_chunk_size

        self.start_speaking_token_id = data_args.start_speaking_token_id 
        self.keep_listening_token_id = data_args.keep_listening_token_id 
        self.start_listening_token_id = data_args.start_listening_token_id
        self.keep_speaking_token_id = data_args.keep_speaking_token_id
        self.start_bc_token_id = data_args.start_bc_token_id
        self.detect_token_id = data_args.detect_token_id
        self.sleep_token_id = data_args.sleep_token_id

        # listening状态的
        self.text_pad_token_id = data_args.text_pad_token_id
        self.stoken_pad_token_id = data_args.stoken_pad_token_id

        self.stoken_delay_token_id = data_args.stoken_delay_token_id
        self.stoken_delay_num = data_args.stoken_delay_num

        # 弥补text 和 stoken差异的
        self.tts_pad_id = data_args.tts_pad_id # tokenizer.convert_tokens_to_ids("<tts_pad>")
        self.tts_start_id = data_args.tts_start_id # tokenizer.convert_tokens_to_ids("<tts_start>")
        self.tts_end_id = data_args.tts_end_id # tokenizer.convert_tokens_to_ids("<tts_end>")
        self.eot_id = data_args.eot_id

        self.audio_pad_token_id = data_args.audio_pad_token_id
        self.audio_token_id = data_args.audio_token_id

        self.window_second = data_args.window_second

        self.ignore_backchannel = data_args.ignore_backchannel
        self.adding_text_hiddenstates = data_args.adding_text_hiddenstates
        self.align_audio_input = data_args.align_audio_input
        self.max_data_length = data_args.max_data_length

        self.filtered_backchannel_set = set(json.load(open(data_args.filtered_backchannel_path))) if data_args.filtered_backchannel_path is not None else None

        if self.align_audio_input:
            self.AUDIO_TOKEN_N_SAMPLE_ALIGN = self.AUDIO_TOKEN_N_SAMPLE // 2
        else:
            self.AUDIO_TOKEN_N_SAMPLE_ALIGN = self.AUDIO_TOKEN_N_SAMPLE

        self.no_stoken_label = data_args.no_stoken_label

        self.inject_noise = data_args.inject_noise
        self.inject_noise_cnt = 0

        self._resamplers = dict()

    def __len__(self):
        return len(self.data_index)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = IMG_TOKEN_LENGTH if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list
    #

    # @property
    # def modality_lengths_type(self):
    #     length_list = []
    #     for sample in self.data["data_type"]:
    #         # if "data_len" in sample:
    #         #     cur_len = sample["data_len"]
    #         # else:
    #         #     cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
    #         cur_len = 1  # 这里不对长度划分
    #         cur_type = self.DATA_TYPE_TO_IDS[sample]

    #         length_list.append((cur_len, cur_type))
    #     return length_list

    def get_ai_bc_timestep(self, user_text, user_audio_token_num, ai_response_codec, input_len):
        before_bc, bc_text, after_bc = extract_tagged_text(user_text, tag="backchannel")
        assert after_bc is not None
        before_bc_token_ids = self.tokenizer([before_bc], add_special_tokens=False).input_ids[0]
        total_bc_token_ids = self.tokenizer([before_bc + ' ' + after_bc], add_special_tokens=False).input_ids[0]
        radio = len(before_bc_token_ids) / len(total_bc_token_ids)
        start_time = round(radio * user_audio_token_num / self.control_token_chunk_size)
        start_time = max(0, min(user_audio_token_num // self.control_token_chunk_size, start_time))
        start_time *= self.control_token_chunk_size
        token_ids, stoken, mapping = self.interleaved_tokenizer(
            bc_text, 
            self.load_stoken(ai_response_codec["Backchannel"]), 
            stoken_mapping_start=input_len + start_time
        )
        return start_time, token_ids, stoken, mapping

    def stepaudio_audio_prepreocess(
        self,
        audio = None,
        debug=False,
    ) -> BatchFeature:

        feats = []
        feats_lengths = []
        input_ids = []
        for i in range(0, audio.shape[0], int(16000 * self.window_second)):
            mel = log_mel_spectrogram(audio[i:i+int(16000 * self.window_second)], n_mels=128, padding=479)
            feats.append(mel.t())
            feats_lengths.append(mel.size(1)-2)
            if debug and self.align_audio_input:
                assert audio[i:i+int(16000 * self.window_second)].shape[0] % self.AUDIO_TOKEN_N_SAMPLE_ALIGN == 0
                input_ids += [self.audio_pad_token_id] * (audio[i:i+int(16000 * self.window_second)].shape[0] // self.AUDIO_TOKEN_N_SAMPLE_ALIGN)
            elif self.align_audio_input:
                input_ids += [self.audio_token_id, self.audio_pad_token_id] * compute_token_num(mel.shape[1])
            else:
                input_ids += [self.audio_token_id] * compute_token_num(mel.shape[1])

        return {"input_ids": input_ids, "feats": feats, "feats_lengths": feats_lengths}

    def user_bc_start(self, before_backchannel, after_backchannel, stoken):
        """
        这里为了方便, 假设stoken是和文本token id线性相关的
        """
        before_len = len(self.tokenizer([before_backchannel], add_special_tokens=False).input_ids[0])
        after_len = len(self.tokenizer([after_backchannel], add_special_tokens=False).input_ids[0])
        return int(len(stoken) * before_len / (before_len + after_len))

    def tokens2silence(self, token_num, dtype):
        return np.zeros(token_num * self.AUDIO_TOKEN_N_SAMPLE_ALIGN, dtype=dtype)

    def load_audio(self, path):
        audio = librosa.load(path, sr=self.SAMPLE_RATE)[0]
        original_len = audio.shape[0] / self.SAMPLE_RATE
        align_len = int(math.ceil(audio.shape[0] / self.AUDIO_TOKEN_N_SAMPLE)) * self.AUDIO_TOKEN_N_SAMPLE
        padding_len = align_len - len(audio)
        silence = np.zeros(padding_len, dtype=audio.dtype)
        audio = np.concatenate([audio, silence])
        token_num = audio.shape[0] // self.AUDIO_TOKEN_N_SAMPLE_ALIGN
        return audio, token_num, original_len

    def adding_noise(self, audio, speech_segments):
        if self.inject_noise_cnt < 10:
            rank0_print("adding noise !!!!!!!!")
            self.inject_noise_cnt += 1
        speech_segments = [(st / 25, et / 25) for st, et in speech_segments]
        return inject_ambient_noise(
            audio, 
            sr=self.SAMPLE_RATE,
            target_dbfs=np.random.uniform(-26, -20),
            snr_db_range=(40, 60),
            noise_mix={
                        'white': np.random.uniform(0.2, 0.4),
                        'pink':  np.random.uniform(0.6, 0.8),
                    },
            speech_segments=speech_segments,
            normalize=False,
        )
    
    def load_stoken(self, stoken):
        if isinstance(stoken, str):
            return eval(stoken)
        return stoken

    def interleaved_tokenizer(self, text, stokens, padding=True, adding_eos=True, stoken_mapping_start=0):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.adding_text_hiddenstates:
            stokens = stokens[:len(stokens) // 4 * 4]
        stoken_ids = [self.stoken_delay_token_id] * self.stoken_delay_num + [self.tts_start_id] + [s + self.AUDIO_TOKEN_OFFSET for s in stokens]
        stoken_mapping = [-1] * self.stoken_delay_num + [-1]
        for i in range(len(stokens) // 4):
            stoken_mapping += [i + stoken_mapping_start] * 4
        for i in range(len(stokens) % 4):
            stoken_mapping += [len(stokens) // 4 + stoken_mapping_start]
        
        if adding_eos:
            text_ids += [self.eot_id]
            stoken_ids += [self.tts_end_id]
            stoken_mapping += [-1]
        try:
            assert len(stoken_ids) >= len(text_ids)
        except:
            print(f"[Data] len(stoken_ids) < len(text_ids) :\n\t{text}\n\t{str(stokens)}")
            text_ids = text_ids[:len(stoken_ids)]

        if padding:
            text_ids += [self.tts_pad_id] * (len(stoken_ids) - len(text_ids))
        
        return text_ids, stoken_ids, stoken_mapping

    def text2stoken_mapping(self, input_ids, stoken_ids, stoken_mapping):
        """
        A: (batch_size, seq_len, hidden_size)
        B: (batch_size, seq_len, hidden_size)
        M: (batch_size, seq_len)
        """
        # 1. 创建掩码 (Mask)
        # mask 的形状为 (batch_size, seq_len)
        # valid_mask[i, j] 为 True 表示 M[i, j] != -1
        valid_mask = (stoken_mapping != -1)

        # 2. 准备 Gather 用的索引
        # 复制一份 M，避免修改原数据
        # 将 -1 的位置替换为 0 (或者任何合法的索引)，防止 gather 越界报错
        # 我们稍后会用 mask 把这些位置的数据过滤掉，所以填 0 没关系
        gather_indices = stoken_mapping.clone()
        gather_indices[~valid_mask] = 0
        
        # 3. 扩展索引维度以匹配 A 的 hidden_size
        # gather_indices 变成 (batch_size, seq_len, hidden_size)
        # 我们在 dim=1 (seq_len) 上进行索引，但需要保留 dim=2 (hidden_size) 的所有数据
        # gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, A.size(-1))

        # 4. 执行 Gather 操作
        # dim=1 表示我们在 seq_len 维度上根据索引取值
        # A_selected[i, j, k] = A[i, gather_indices[i, j, k], k]
        # 也就是 A[i, M[i,j], :]
        input_ids_selected = torch.gather(input_ids, dim=0, index=gather_indices)

        # 5. 将数据加到 B 中
        # 需要将 mask 扩展一维以便广播 (Broadcasting)
        # mask_expanded shape: (batch_size, seq_len, 1)
        # mask_expanded = valid_mask.unsqueeze(-1)
        
        return input_ids_selected * valid_mask
     
    def __getitem__(self, i):
        item_index = index = self.data_index[i]
        source_data = self.data[index[0]][index[1]]
        conversation = source_data["conversation"]["conversation_history"]
        
        user_audio = []
        user_audio_time = []
        input_ids = []
        stoken_ids = []
        stoken_mapping = []
        control_input_ids = []
        text_label_ids = []
        stoken_label_ids = []
        control_label_ids = []

        last_ai_resposne_ids = None
        last_control_label = None
        last_control_input_token = None
        last_ai_stoken = None
        last_ai_stoken_mapping = None

        for conv_id, conv in enumerate(conversation):
            user_speech_path = conv["speech"]["user_speech"]["Main"] # 当前不考虑backchannel
            user_speech_wave, user_speech_token_num, user_speech_original_len = self.load_audio(user_speech_path)

            # Todo: 这里可以在user_speech_wave前面拼上静音, 模拟随时开始说话延迟, 注意interruption的不能加
       
            # user speech
            if last_ai_resposne_ids is None:
                chunk_num = math.ceil((user_speech_token_num + 1) / self.control_token_chunk_size) # 这里+1保证 <s-s>在音频之后出现
                padding_user_speech_token_num = chunk_num * self.control_token_chunk_size
                silence = self.tokens2silence(padding_user_speech_token_num - user_speech_token_num, dtype=user_speech_wave.dtype)
                user_speech_wave = np.concatenate([user_speech_wave, silence])

                # 给AI BC用
                input_len = sum([len(t) for t in input_ids])

                input_token             = []
                stoken_token            = []
                control_input_token     = []
                stoken_mapping_token    = []
                contorl_label           = []
                stoken_label_token      = []
                text_label              = []
                for c in range(chunk_num):
                    input_token             += [self.text_pad_token_id]     * self.control_token_chunk_size
                    stoken_token            += [self.stoken_pad_token_id]   * self.control_token_chunk_size  
                    stoken_mapping_token    += [-1]                         * self.control_token_chunk_size  
                    text_label              += [IGNORE_INDEX]               * self.control_token_chunk_size
                    stoken_label_token      += [IGNORE_INDEX]               * self.control_token_chunk_size
                    if c != chunk_num - 1:
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.keep_listening_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.keep_listening_token_id]
                    else:
                        # 选定下一个chunk为s-s
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_speaking_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_speaking_token_id]
                
                # AI BC
                if not self.ignore_backchannel:
                    try:
                        ori_input_token             = copy.deepcopy(input_token)
                        ori_stoken_token            = copy.deepcopy(stoken_token)
                        ori_stoken_mapping_token    = copy.deepcopy(stoken_mapping_token)
                        ori_control_input_token     = copy.deepcopy(control_input_token)
                        ori_contorl_label           = copy.deepcopy(contorl_label)
                        ori_text_label              = copy.deepcopy(text_label)
                        ori_stoken_label_token      = copy.deepcopy(stoken_label_token)

                        if "AI_Backchannel" in conv["event"]:
                            print("[Datasets] AI_Backchannel")
                            assert self.filtered_backchannel_set is None or conv["speech"]['ai_speech']['Backchannel'] in self.filtered_backchannel_set, "Low quality BC"
                            start_timestep, bc_token_ids, bc_stoken_ids, bc_mapping_ids = self.get_ai_bc_timestep(conv["user_utterance"], user_speech_token_num, conv["ai_response_token"], input_len)
                            if start_timestep >= padding_user_speech_token_num:
                                print("[Datasets] bad start_timestep for AI_Backchannel in Interruption")
                            else:
                                start_chunk = start_timestep // self.control_token_chunk_size
                                end_chunk = math.ceil((start_timestep + len(bc_token_ids) + 1) / self.control_token_chunk_size)
                                assert  control_input_token[start_timestep-1] == self.keep_listening_token_id
                                control_input_token[start_timestep-1] = self.start_bc_token_id
                                contorl_label[start_timestep-1] = self.start_bc_token_id
                                for c in range(start_chunk + 1, min(end_chunk + 1, chunk_num)):
                                    pos = c * self.control_token_chunk_size - 1
                                    assert control_input_token[pos] == self.keep_listening_token_id
                                    if c != end_chunk:
                                        control_input_token[pos]    = self.start_bc_token_id # keep and start backchaneel are same
                                        contorl_label[pos]          = self.start_bc_token_id
                                    else:
                                        control_input_token[pos]    = self.start_listening_token_id
                                        contorl_label[pos]          = self.start_listening_token_id
                                bc_token_ids    = bc_token_ids[:len(input_token) - start_timestep]
                                bc_stoken_ids   = bc_stoken_ids[:len(input_token) - start_timestep]
                                bc_mapping_ids  = bc_mapping_ids[:len(input_token) - start_timestep]
                                for x,i in enumerate(range(start_timestep, len(bc_token_ids) + start_timestep)):
                                    assert input_token[i] == self.text_pad_token_id and text_label[i] == IGNORE_INDEX and stoken_token[i] == self.stoken_pad_token_id and stoken_label_token[i] == IGNORE_INDEX and stoken_mapping_token[i] == -1
                                    input_token[i]          = bc_token_ids[x]
                                    stoken_token[i]         = bc_stoken_ids[x]
                                    stoken_mapping_token[i] = bc_mapping_ids[x]
                                    text_label[i]           = bc_token_ids[x] if bc_token_ids[x] != self.tts_pad_id else IGNORE_INDEX
                                    stoken_label_token[i]   = bc_stoken_ids[x] if bc_stoken_ids[x] != self.tts_start_id and bc_stoken_ids[x] != self.stoken_delay_token_id else IGNORE_INDEX
                    except Exception as e:
                        print(f"bad datasets index: {item_index}")
                        print(e)
                        input_token             = ori_input_token         
                        stoken_token            = ori_stoken_token        
                        stoken_mapping_token    = ori_stoken_mapping_token
                        control_input_token     = ori_control_input_token 
                        contorl_label           = ori_contorl_label       
                        text_label              = ori_text_label          
                        stoken_label_token      = ori_stoken_label_token  

                cur_audio_time = np.concatenate(user_audio).shape[0] / self.SAMPLE_RATE if len(user_audio) else 0.0
                user_audio_time.append((cur_audio_time, cur_audio_time + user_speech_original_len))
                user_audio.append(user_speech_wave)
                input_ids.append(input_token)
                stoken_ids.append(stoken_token)
                stoken_mapping.append(stoken_mapping_token)
                control_input_ids.append(control_input_token)
                text_label_ids.append(text_label)
                stoken_label_ids.append(stoken_label_token)
                control_label_ids.append(contorl_label)

                assert "User_Interruption" not in conv["event"]
            
            else: 
                assert last_control_label is not None
                chunk_num = max(1, math.ceil((user_speech_token_num - len(last_control_label)) / self.control_token_chunk_size))
                padding_user_speech_token_num = chunk_num * self.control_token_chunk_size + len(last_control_label)
                silence = self.tokens2silence(padding_user_speech_token_num - user_speech_token_num, dtype=user_speech_wave.dtype)
                user_speech_wave = np.concatenate([user_speech_wave, silence])
                
                input_token = []
                stoken_token = []
                stoken_mapping_token = []
                control_input_token = []
                contorl_label = []
                stoken_label_token = []
                text_label = []

                # 上一轮残存的输入
                contorl_label       += copy.deepcopy(last_control_label)
                control_input_token += copy.deepcopy(last_control_input_token)
                input_token         += copy.deepcopy(last_ai_resposne_ids[:len(last_control_label)])
                stoken_token        += copy.deepcopy(last_ai_stoken[:len(last_control_label)])
                stoken_mapping_token+= last_ai_stoken_mapping[:len(last_control_label)]
                text_label          += [t if t != self.tts_pad_id else IGNORE_INDEX for t in copy.deepcopy(last_ai_resposne_ids[:len(last_control_label)])]
                stoken_label_token  += [t if t != self.tts_start_id and t != self.stoken_delay_token_id else IGNORE_INDEX for t in copy.deepcopy(last_ai_stoken[:len(last_control_label)])]

                if len(input_token) < len(contorl_label):
                    padding_len = len(contorl_label) - len(input_token)
                    text_label          += [IGNORE_INDEX]  * padding_len
                    stoken_label_token  += [IGNORE_INDEX]  * padding_len
                    input_token         += [self.text_pad_token_id] * padding_len
                    stoken_token        += [self.stoken_pad_token_id]  * padding_len
                    stoken_mapping_token+= [-1] * padding_len
                else: # 此时control 已经是S-L, 这里默认pad
                    input_token         = input_token[:-1]          + [self.text_pad_token_id]
                    text_label          = text_label[:-1]           + [IGNORE_INDEX]
                    stoken_token        = stoken_token[:-1]         + [self.stoken_pad_token_id]
                    stoken_label_token  = stoken_label_token[:-1]   + [IGNORE_INDEX]
                    stoken_mapping_token= stoken_mapping_token[:-1] + [-1]
                    

                for c in range(chunk_num):
                    input_token         += [self.text_pad_token_id]     * self.control_token_chunk_size
                    stoken_token        += [self.stoken_pad_token_id]   * self.control_token_chunk_size
                    stoken_mapping_token+= [-1]                         * self.control_token_chunk_size
                    text_label          += [IGNORE_INDEX]               * self.control_token_chunk_size
                    stoken_label_token  += [IGNORE_INDEX]               * self.control_token_chunk_size
                    if c != chunk_num - 1:
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.keep_listening_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.keep_listening_token_id]
                    else:
                        # 选定下一个chunk为s-s
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_speaking_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_speaking_token_id]
                
                # AI BC
                if not self.ignore_backchannel:
                    try:
                        ori_input_token             = copy.deepcopy(input_token)
                        ori_stoken_token            = copy.deepcopy(stoken_token)
                        ori_stoken_mapping_token    = copy.deepcopy(stoken_mapping_token)
                        ori_control_input_token     = copy.deepcopy(control_input_token)
                        ori_contorl_label           = copy.deepcopy(contorl_label)
                        ori_text_label              = copy.deepcopy(text_label)
                        ori_stoken_label_token      = copy.deepcopy(stoken_label_token)

                        if "AI_Backchannel" in conv["event"]:
                            print("[Datasets] AI_Backchannel")
                            assert self.filtered_backchannel_set is None or conv["speech"]['ai_speech']['Backchannel'] in self.filtered_backchannel_set, "Low quality BC"
                            start_timestep, bc_token_ids, bc_stoken_ids, bc_mapping_ids = self.get_ai_bc_timestep(conv["user_utterance"], user_speech_token_num, conv["ai_response_token"], input_len)
                            if start_timestep <= last_control_input_token or start_timestep >= padding_user_speech_token_num:
                                print("[Datasets] bad start_timestep for AI_Backchannel in Interruption")
                            else:
                                start_chunk = start_timestep // self.control_token_chunk_size
                                end_chunk = math.ceil((start_timestep + len(bc_token_ids) + 1) / self.control_token_chunk_size)
                                assert  control_input_token[start_timestep-1] == self.keep_listening_token_id
                                control_input_token[start_timestep-1] = self.start_bc_token_id
                                contorl_label[start_timestep-1] = self.start_bc_token_id
                                for c in range(start_chunk + 1, min(end_chunk + 1, chunk_num)):
                                    pos = c * self.control_token_chunk_size - 1
                                    assert control_input_token[pos] == self.keep_listening_token_id
                                    if c != end_chunk:
                                        control_input_token[pos]    = self.start_bc_token_id # keep and start backchaneel are same
                                        contorl_label[pos]          = self.start_bc_token_id
                                    else:
                                        control_input_token[pos]    = self.start_listening_token_id
                                        contorl_label[pos]          = self.start_listening_token_id
                                bc_token_ids    = bc_token_ids[:len(input_token) - start_timestep]
                                bc_stoken_ids   = bc_stoken_ids[:len(input_token) - start_timestep]
                                bc_mapping_ids  = bc_mapping_ids[:len(input_token) - start_timestep]
                                for x,i in enumerate(range(start_timestep, len(bc_token_ids) + start_timestep)):
                                    assert input_token[i] == self.text_pad_token_id and text_label[i] == IGNORE_INDEX and stoken_token[i] == self.stoken_pad_token_id and stoken_label_token[i] == IGNORE_INDEX and stoken_mapping_token[i] == -1
                                    input_token[i]          = bc_token_ids[x]
                                    stoken_token[i]         = bc_stoken_ids[x]
                                    stoken_mapping_token[i] = bc_mapping_ids[x]
                                    text_label[i]           = bc_token_ids[x] if bc_token_ids[x] != self.tts_pad_id else IGNORE_INDEX
                                    stoken_label_token[i]   = bc_stoken_ids[x] if bc_stoken_ids[x] != self.tts_start_id and bc_stoken_ids[x] != self.stoken_delay_token_id else IGNORE_INDEX
                    except Exception as e:
                        print(f"bad datasets index: {item_index}")
                        print(e)
                        input_token             = ori_input_token         
                        stoken_token            = ori_stoken_token        
                        stoken_mapping_token    = ori_stoken_mapping_token
                        control_input_token     = ori_control_input_token 
                        contorl_label           = ori_contorl_label       
                        text_label              = ori_text_label          
                        stoken_label_token      = ori_stoken_label_token  

                cur_audio_time = np.concatenate(user_audio).shape[0] / self.SAMPLE_RATE  if len(user_audio) else 0.0
                user_audio_time.append((cur_audio_time, cur_audio_time + user_speech_original_len))
                user_audio.append(user_speech_wave)
                input_ids.append(input_token)
                stoken_ids.append(stoken_token)
                stoken_mapping.append(stoken_mapping_token)
                control_input_ids.append(control_input_token)
                text_label_ids.append(text_label)
                stoken_label_ids.append(stoken_label_token)
                control_label_ids.append(contorl_label)

                assert not any([t in [self.tts_pad_id] for t in text_label])
                assert not any([t in [self.stoken_pad_token_id, self.tts_start_id, self.stoken_delay_token_id] for t in stoken_label_token])
                assert "User_Interruption" in conv["event"]
                last_ai_resposne_ids = None
                last_ai_stoken = None
                last_ai_stoken_mapping = None
                last_control_label = None
                last_control_input_token = None

            ai_response = conv["ai_response"]
            ai_response_codec = conv["ai_response_token"]

            # 检测是否为interruption
            before_interruption, interruption, after_interruption = extract_tagged_text(ai_response, tag="interruption")
            if after_interruption is not None:
                assert "User_Interruption" in conversation[conv_id + 1]["event"]
                assert "User_Backchannel" not in conv["event"]
                _, _, after_backchannel = extract_tagged_text(conv["ai_response"], tag="backchannel")
                assert after_backchannel is None

                full_response = before_interruption + ' ' + after_interruption
                _, before_interruption_stoken, _ = self.interleaved_tokenizer(before_interruption, self.load_stoken(ai_response_codec["BeforeInterruption"]), padding=False, adding_eos=False)
                input_len = sum([len(t) for t in input_ids])
                full_label, full_stoken, full_mapping = self.interleaved_tokenizer(full_response, self.load_stoken(ai_response_codec["BeforeInterruption"]) + self.load_stoken(ai_response_codec["AfterInterruption"]), stoken_mapping_start=input_len)
                
                ai_response_input_ids   = full_label[:len(before_interruption_stoken)]
                ai_response_stoken      = before_interruption_stoken
                ai_response_mapping     = full_mapping[:len(before_interruption_stoken)]
                last_ai_resposne_ids    = full_label[len(ai_response_stoken):]
                last_ai_stoken          = full_stoken[len(ai_response_stoken):]
                last_ai_stoken_mapping  = full_mapping[len(before_interruption_stoken):]

                # ai response
                user_audio.append(self.tokens2silence(token_num=len(ai_response_input_ids), dtype=user_speech_wave.dtype))
                input_ids.append(ai_response_input_ids)
                stoken_ids.append(ai_response_stoken)
                stoken_mapping.append(ai_response_mapping)
                text_label_ids.append(copy.deepcopy([t if t != self.tts_pad_id else IGNORE_INDEX for t in ai_response_input_ids]))
                stoken_label_ids.append(copy.deepcopy([t if t != self.tts_start_id and t != self.stoken_delay_token_id else IGNORE_INDEX for t in ai_response_stoken]))

                chunk_num = int(len(ai_response_input_ids) / self.control_token_chunk_size) 
                contorl_label = []
                control_input_token = []
                for c in range(chunk_num):
                    control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.keep_speaking_token_id]
                    contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.keep_speaking_token_id]

                last_control_input_token = [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_listening_token_id]
                last_control_label = [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_listening_token_id]

                rest_len = len(ai_response_input_ids) - len(contorl_label)
                control_input_token += last_control_input_token[:rest_len]
                contorl_label += last_control_label[:rest_len]

                last_control_input_token = last_control_input_token[rest_len:]
                last_control_label = last_control_label[rest_len:]

                control_input_ids.append(control_input_token)
                control_label_ids.append(contorl_label)

            else:
                assert conv_id == len(conversation) - 1 or "User_Interruption" not in conversation[conv_id + 1]["event"]
                
                if "<user_backchannel>" in before_interruption:
                    before_backchannel, backchannel, after_backchannel = extract_tagged_text(before_interruption, tag="user_backchannel")
                else:
                    before_backchannel, backchannel, after_backchannel = extract_tagged_text(before_interruption, tag="backchannel")
                user_backchannel_wave = None
                user_backchannel_start = None
                if after_backchannel is not None:
                    if self.ignore_backchannel:
                        ai_response = before_backchannel + after_backchannel
                    else:
                        assert "User_Backchannel" in conv["event"]
                        ai_response = before_backchannel + ' ' + after_backchannel
                        user_backchannel_start = self.user_bc_start(before_backchannel, after_backchannel, eval(ai_response_codec["Main"]))
                        try:
                            user_backchannel_wave, user_backchannel_num, user_backchannel_original_len = self.load_audio(conv["speech"]["user_speech"]["Backchannel"])
                        except:
                            print("[Datasets] bad User_Backchannel")
                            user_backchannel_wave = None
                            user_backchannel_start = None
                else:
                    ai_response = before_backchannel

                input_len = sum([len(t) for t in input_ids])
                ai_response_input_ids, ai_response_stoken, ai_response_mapping = self.interleaved_tokenizer(ai_response, self.load_stoken(ai_response_codec["Main"]), stoken_mapping_start=input_len)

                # 这里+1 是为了防止len(ai_response_input_ids)整除于self.control_token_chunk_size, 覆盖了s-l token
                chunk_num = math.ceil((len(ai_response_input_ids) + 1) / self.control_token_chunk_size) 
                padding_ai_response = [self.text_pad_token_id] * (chunk_num * self.control_token_chunk_size - len(ai_response_input_ids))
                padding_ai_stoken = [self.stoken_pad_token_id] * len(padding_ai_response)
                padding_ai_mapping = [-1] * len(padding_ai_response)
                padding_ai_response_input_ids       = ai_response_input_ids + padding_ai_response
                padding_ai_response_stoken          = ai_response_stoken + padding_ai_stoken
                padding_ai_response_mapping         = ai_response_mapping + padding_ai_mapping
                padding_ai_response_label_ids       = copy.deepcopy(copy.deepcopy([t if t != self.tts_pad_id else IGNORE_INDEX for t in ai_response_input_ids])) + [IGNORE_INDEX] * len(padding_ai_response)
                padding_ai_response_stoken_label    = copy.deepcopy(copy.deepcopy([t if t != self.tts_start_id and t != self.stoken_delay_token_id else IGNORE_INDEX for t in ai_response_stoken])) + [IGNORE_INDEX] * len(padding_ai_response)

                contorl_label = []
                control_input_token = []
                for c in range(chunk_num):
                    if c != chunk_num - 1:
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.keep_speaking_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.keep_speaking_token_id]
                    else:
                        control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_listening_token_id]
                        contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_listening_token_id]

                # ai response
                user_speech_wave = self.tokens2silence(token_num=len(padding_ai_response_input_ids), dtype=user_speech_wave.dtype)
                if user_backchannel_start is not None:
                    replace_start = user_backchannel_start * self.AUDIO_TOKEN_N_SAMPLE_ALIGN
                    user_backchannel_wave = user_backchannel_wave[:min(len(user_speech_wave)-replace_start, len(user_backchannel_wave))]
                    user_backchannel_original_len = min(user_backchannel_original_len, user_backchannel_wave.shape[0] / self.SAMPLE_RATE)
                    user_speech_wave[user_backchannel_start:user_backchannel_start+len(user_backchannel_wave)] = user_backchannel_wave
                cur_audio_time = np.concatenate(user_audio).shape[0] / self.SAMPLE_RATE if len(user_audio) else 0.0
                user_audio_time.append((cur_audio_time, cur_audio_time + user_speech_original_len))
                user_audio.append(user_speech_wave)

                input_ids.append(padding_ai_response_input_ids)
                text_label_ids.append(padding_ai_response_label_ids)
                stoken_ids.append(padding_ai_response_stoken)
                stoken_mapping.append(padding_ai_response_mapping)
                stoken_label_ids.append(padding_ai_response_stoken_label)
                control_input_ids.append(control_input_token)
                control_label_ids.append(contorl_label)

            tmp_input_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in input_ids], dim=0)
            if len(tmp_input_ids) > self.max_data_length:
                print(f"data length > {self.max_data_length}, cutting....")
                break

        # ------------------- debug -------------------

        for a, i, c, t, cl, si, sli, sm in list(zip(user_audio, input_ids, control_input_ids, text_label_ids, control_label_ids, stoken_ids, stoken_label_ids, stoken_mapping)):
            q = self.stepaudio_audio_prepreocess(a, debug=True)
            assert len(q['input_ids']) == len(t) == len(i) == len(c) == len(cl) == len(si) == len(sli) == len(sm)

        user_audio = np.concatenate(user_audio)
        if self.inject_noise:
            user_audio = self.adding_noise(user_audio, user_audio_time)
        user_audio_input = self.stepaudio_audio_prepreocess(user_audio)
        input_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in input_ids], dim=0)
        stoken_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_ids], dim=0)
        stoken_mapping = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_mapping], dim=0)
        control_input_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in control_input_ids], dim=0)
        text_label_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in text_label_ids], dim=0)
        stoken_label_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_label_ids], dim=0)
        control_label_ids = torch.cat([torch.tensor(t, dtype=torch.long) for t in control_label_ids], dim=0)

        if self.no_stoken_label:
            input_ids[stoken_ids==self.tts_end_id] = self.tts_end_id

        try:
            assert len(user_audio_input['input_ids']) == len(input_ids) == len(stoken_ids) == len(control_input_ids) == len(text_label_ids) == len(stoken_label_ids) == len(control_label_ids)  == len(stoken_mapping)
        except:
            print(len(user_audio_input['input_ids']),len(input_ids), len(stoken_ids) , len(control_input_ids) , len(text_label_ids), len(stoken_label_ids),len(control_label_ids))
            assert len(user_audio_input['input_ids']) == len(input_ids) + 1 and self.align_audio_input
            user_audio_input['input_ids'] = user_audio_input['input_ids'][:-1]
            assert len(user_audio_input['input_ids']) == len(input_ids) == len(stoken_ids) == len(control_input_ids) == len(text_label_ids) == len(stoken_label_ids) == len(control_label_ids)  == len(stoken_mapping)
            
        assert (text_label_ids == self.text_pad_token_id).sum() == 0
        assert (text_label_ids == self.tts_pad_id).sum() == 0, f"{text_label_ids.tolist()}"
        assert (stoken_label_ids == self.stoken_pad_token_id).sum() == 0
        stoken_label_ids[stoken_label_ids==self.stoken_delay_token_id] = IGNORE_INDEX

        feats = user_audio_input["feats"]
        feats_lengths = torch.tensor(user_audio_input["feats_lengths"], dtype=torch.torch.int32)
        audio_input_ids = user_audio_input["input_ids"]

        # # debug
        # listen_state = True
        # for s in range(0, len(input_ids), self.control_token_chunk_size):
        #     e = min(len(input_ids), s + self.control_token_chunk_size)
        #     if listen_state:
        #         assert (text_label_ids[s:e] == IGNORE_INDEX).all()
        #         assert (control_label_ids[s:e - 1] == IGNORE_INDEX).all()
        #         assert control_label_ids[e - 1] in [self.keep_listening_token_id, self.start_speaking_token_id] 
        #         assert (input_ids[s:e] == self.text_pad_token_id).all()
        #         assert (control_input_ids[s:e - 2] == self.sleep_token_id).all()
        #         assert control_input_ids[e - 2] == self.detect_token_id and control_input_ids[e - 1] == control_label_ids[e - 1]
        #         if control_label_ids[e - 1] == self.start_speaking_token_id:
        #             listen_state = False
        #     else:
        #         assert control_label_ids[e - 1] in [self.start_listening_token_id, self.keep_speaking_token_id] 
        #         assert (control_label_ids[s:e - 1] == IGNORE_INDEX).all()
        #         assert (control_input_ids[s:e - 2] == self.sleep_token_id).all()
        #         assert control_input_ids[e - 2] == self.detect_token_id and control_input_ids[e - 1] == control_label_ids[e - 1]
                
        #         if control_label_ids[e - 1] == self.keep_speaking_token_id:
        #             assert (text_label_ids[s:e] != IGNORE_INDEX).all()
        #             assert (text_label_ids[s:e] == input_ids[s:e]).all()
        #         else:
        #             assert self.eos_token_id in text_label_ids[s-1:e].tolist() or (text_label_ids[s:e-1] != IGNORE_INDEX).all()
        #             if (text_label_ids[s:e] != IGNORE_INDEX).all():
        #                 assert (text_label_ids[s:e] == input_ids[s:e]).all()
        #             else:
        #                 sp = (text_label_ids[s:e] != IGNORE_INDEX).sum()
        #                 assert (text_label_ids[s:s + sp] == input_ids[s:s + sp]).all()
        #                 assert (input_ids[s + sp:e] == self.text_pad_token_id).all()
        #             listen_state = True

        prefix_input_ids = self.tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False, return_tensors='pt').input_ids[0]
        input_ids = torch.cat((prefix_input_ids, input_ids), dim=0)
        stoken_mapping[stoken_mapping != -1] += len(prefix_input_ids)
        stoken_mapping = torch.cat((torch.ones_like(prefix_input_ids, dtype=torch.long, device=stoken_mapping.device) * -1, stoken_mapping), dim=0)
        labels = torch.cat((torch.ones_like(prefix_input_ids, dtype=torch.long, device=text_label_ids.device) * IGNORE_INDEX, text_label_ids), dim=0)
        stoken_label_ids = torch.cat((torch.ones_like(prefix_input_ids, dtype=torch.long, device=text_label_ids.device) * IGNORE_INDEX, stoken_label_ids), dim=0)
        control_label_ids = torch.cat((torch.ones_like(prefix_input_ids, dtype=torch.long, device=control_label_ids.device) * IGNORE_INDEX, control_label_ids), dim=0)
        audio_input_ids = torch.tensor(audio_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        data_dict = {
            "input_ids": input_ids,
            "stoken_ids": stoken_ids,
            "stoken_mapping": stoken_mapping,
            "control_input_ids": control_input_ids,
            "labels": labels,
            "stoken_label_ids": stoken_label_ids,
            "control_label_ids": control_label_ids,
            "wavs": feats,
            "wav_lens": feats_lengths,
            "attention_mask": attention_mask,
            "audio_input_ids": audio_input_ids,
            "prefix_input_ids": prefix_input_ids
        }

        return data_dict

DEBUG = 5

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, control_input_ids, labels, control_label_ids, wavs, wav_lens, attention_mask, audio_input_ids, prefix_input_ids, stoken_ids, stoken_label_ids, stoken_mapping = tuple(
            [instance[key] for instance in instances] for key in (
                "input_ids", "control_input_ids", "labels", "control_label_ids", "wavs", "wav_lens", "attention_mask", "audio_input_ids", "prefix_input_ids", "stoken_ids", "stoken_label_ids", "stoken_mapping"
                )
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        stoken_ids = torch.nn.utils.rnn.pad_sequence(stoken_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        stoken_mapping = torch.nn.utils.rnn.pad_sequence(stoken_mapping, batch_first=True, padding_value=-1)
        control_input_ids = torch.nn.utils.rnn.pad_sequence(control_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        audio_input_ids = torch.nn.utils.rnn.pad_sequence(audio_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        stoken_label_ids = torch.nn.utils.rnn.pad_sequence(stoken_label_ids, batch_first=True, padding_value=IGNORE_INDEX)
        control_label_ids = torch.nn.utils.rnn.pad_sequence(control_label_ids, batch_first=True, padding_value=IGNORE_INDEX)
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        prefix_input_ids = torch.stack(prefix_input_ids, dim=0)

        total_wavs = []
        for w in wavs:
            total_wavs += w
        wavs = torch.nn.utils.rnn.pad_sequence(
            total_wavs,
            batch_first=True,
            padding_value=0).transpose(1, 2)
        wav_lens = torch.cat(wav_lens, dim=0)

        batch = dict(
            input_ids=input_ids,
            stoken_ids=stoken_ids,
            stoken_mapping=stoken_mapping,
            control_input_ids=control_input_ids,
            audio_input_ids=audio_input_ids,
            labels=labels,
            stoken_label_ids=stoken_label_ids,
            control_label_ids=control_label_ids,
            attention_mask=attention_mask,
            wavs=wavs,
            wav_lens=wav_lens,
            prefix_input_ids=prefix_input_ids,
        )

        # if DEBUG > 0:
        #     debug_print(self.tokenizer, input_ids[0], labels[0])
        #     rank0_print("codec_labels", codec_labels.shape)
        #     rank0_print("codec_input_ids", codec_input_ids.shape)
        #     DEBUG -= 1

        return batch
