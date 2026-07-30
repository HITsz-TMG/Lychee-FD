import copy
import gc
import os
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

IGNORE_INDEX = -100

SYSTEM_MESSAGE_PREFIX = "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"

"""
Compared with V3, input audio token IDs are inserted with spacing.
Compared with V4, TTS END was changed to EOT; tts_start and tts_end were added;
text EOT now marks the end of text instead of the tts_end section.
Compared with V5, AI backchannel support was added.
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

        # Parse data_path. Each path may be followed by "@X" to set a sampling
        # ratio: X > 1 upsamples, X < 1 downsamples, and omitted means 1.0.
        # Example: "path1@2.0;path2@0.5;path3".
        self.data = []
        self.data_sample_rates = []
        for entry in data_args.data_path.split(';'):
            entry = entry.strip()
            if not entry:
                continue
            if '@' in entry:
                path, rate_str = entry.rsplit('@', 1)
                rate = float(rate_str)
            else:
                path, rate = entry, 1.0
            self.data.append(datasets.load_from_disk(path))
            self.data_sample_rates.append(rate)

        self.data_index = []
        for i, d in enumerate(self.data):
            rate = self.data_sample_rates[i]
            indices = [(i, x) for x in range(len(d))]
            if rate >= 1.0:
                # Upsample by repeating int(rate) times.
                repeat = int(rate)
                self.data_index += indices * repeat
            else:
                # Downsample by randomly keeping the requested ratio.
                k = max(1, int(round(len(indices) * rate)))
                self.data_index += random.sample(indices, k)
            rank0_print(f"[Dataset] loaded {path} | size={len(d)} | sample_rate={rate} | effective={len([x for x in self.data_index if x[0]==i])}")
            
        self.control_token_chunk_size = data_args.control_token_chunk_size

        self.start_speaking_token_id = data_args.start_speaking_token_id 
        self.keep_listening_token_id = data_args.keep_listening_token_id 
        self.start_listening_token_id = data_args.start_listening_token_id
        self.keep_speaking_token_id = data_args.keep_speaking_token_id
        self.detect_token_id = data_args.detect_token_id
        self.sleep_token_id = data_args.sleep_token_id

        # AI BC control token
        self.start_bc_token_id  = data_args.start_bc_token_id
        self.keep_bc_token_id   = data_args.start_bc_token_id
        self.end_bc_token_id    = data_args.start_listening_token_id

        # Tokens used in listening state.
        self.text_pad_token_id = data_args.text_pad_token_id
        self.stoken_pad_token_id = data_args.stoken_pad_token_id

        self.stoken_delay_token_id = data_args.stoken_delay_token_id
        self.stoken_delay_num = data_args.stoken_delay_num

        # Tokens used to bridge text and stoken differences.
        self.tts_pad_id = data_args.tts_pad_id # tokenizer.convert_tokens_to_ids("<tts_pad>")
        self.tts_start_id = data_args.tts_start_id # tokenizer.convert_tokens_to_ids("<tts_start>")
        self.tts_end_id = data_args.tts_end_id # tokenizer.convert_tokens_to_ids("<tts_end>")
        self.eot_id = data_args.eot_id

        self.audio_pad_token_id = data_args.audio_pad_token_id
        self.audio_token_id = data_args.audio_token_id

        self.window_second = data_args.window_second

        self.enable_user_bc = data_args.enable_user_bc
        self.enable_ai_bc = data_args.enable_ai_bc
        self.user_bc_lead_silence_sec = getattr(data_args, "user_bc_lead_silence_sec", 6.0)
        self.user_bc_min_gap_sec = getattr(data_args, "user_bc_min_gap_sec", 5.0)
        self.user_bc_max_num = getattr(data_args, "user_bc_max_num", 3)
        self.ai_bc_lead_silence_sec = getattr(data_args, "ai_bc_lead_silence_sec", 6.0)
        self.ai_bc_min_gap_sec = getattr(data_args, "ai_bc_min_gap_sec", 5.0)
        self.ai_bc_max_num = getattr(data_args, "ai_bc_max_num", 3)
        assert self.user_bc_lead_silence_sec >= 0 and self.user_bc_min_gap_sec >= 0 and self.user_bc_max_num >= 0
        assert self.ai_bc_lead_silence_sec >= 0 and self.ai_bc_min_gap_sec >= 0 and self.ai_bc_max_num >= 0

        self.adding_text_hiddenstates = data_args.adding_text_hiddenstates
        self.align_audio_input = data_args.align_audio_input
        self.max_data_length = data_args.max_data_length

        if self.align_audio_input:
            self.AUDIO_TOKEN_N_SAMPLE_ALIGN = self.AUDIO_TOKEN_N_SAMPLE // 2
        else:
            self.AUDIO_TOKEN_N_SAMPLE_ALIGN = self.AUDIO_TOKEN_N_SAMPLE

        self.no_stoken_label = data_args.no_stoken_label

        # Control-label mask stats: aggregate over _mask_log_interval samples,
        # print from worker 0 only, then reset.
        self._mask_log_interval = 250
        self._mask_log_cnt = 0
        self._mask_stat = self._new_mask_stat()

        # Backchannel filtering stats: aggregate over _bc_filter_log_interval
        # filter calls, print from worker 0 only, then reset.
        self._bc_filter_log_interval = 250
        self._bc_filter_log_cnt = 0
        self._bc_filter_stat = self._new_bc_filter_stat()

        self._resamplers = dict() 

    @staticmethod
    def _new_mask_stat():
        # Count special tokens by priority: S-* / K-* boundary / AI start_bc /
        # AI keep_bc / user bc. Also track normal tokens and sampled normals.
        return {"S": 0, "K_boundary": 0, "start_bc": 0, "keep_bc": 0, "bc": 0, "normal": 0, "normal_kept": 0}

    @staticmethod
    def _new_bc_filter_stat():
        return {
            "user_bc": {"accepted": 0, "deleted": 0},
            "ai_bc": {"accepted": 0, "deleted": 0},
        }

    @staticmethod
    def _is_mask_log_worker():
        """Only let dataloader worker 0, or a single-process loader, print logs."""
        try:
            info = torch.utils.data.get_worker_info()
        except Exception:
            info = None
        return info is None or info.id == 0

    def _accumulate_mask_stat(self, start_positions, keep_boundary_idx, start_bc_positions, keep_bc_positions,
                              stat_bc, n_normal, n_normal_kept, T):
        """Deduplicate and count special tokens by priority.

        AI BC local priority is K-boundary > start_bc > keep_bc > forced_bc.
        Normal-token totals and sampled-normal totals are aggregated and printed
        every _mask_log_interval samples.
        """
        def _valid(ps):
            if ps is None:
                return set()
            return set(int(p) for p in ps if 0 <= int(p) < T)

        S = _valid(start_positions)
        KB = _valid(keep_boundary_idx)
        SBC = _valid(start_bc_positions)
        KBC = _valid(keep_bc_positions)
        bc = _valid(stat_bc)

        # Priority deduplication: assign each position to its highest-priority class.
        seen = set(S)
        kb_only = KB - seen; seen |= kb_only
        start_bc_only = SBC - seen; seen |= start_bc_only
        keep_bc_only = KBC - seen; seen |= keep_bc_only
        bc_only = bc - seen; seen |= bc_only

        st = self._mask_stat
        st["S"] += len(S)
        st["K_boundary"] += len(kb_only)
        st["start_bc"] += len(start_bc_only)
        st["keep_bc"] += len(keep_bc_only)
        st["bc"] += len(bc_only)
        st["normal"] += n_normal
        st["normal_kept"] += n_normal_kept

        self._mask_log_cnt += 1
        if self._mask_log_cnt >= self._mask_log_interval:
            if self._is_mask_log_worker():
                total_special = st["S"] + st["K_boundary"] + st["start_bc"] + st["keep_bc"] + st["bc"]
                kept_ratio = (st["normal_kept"] / st["normal"] * 100.0) if st["normal"] > 0 else 0.0
                print(
                    f"[MaskStat] over {self._mask_log_cnt} samples | "
                    f"special(total={total_special}): S={st['S']} K_boundary={st['K_boundary']} "
                    f"start_bc={st['start_bc']} keep_bc={st['keep_bc']} "
                    f"bc={st['bc']} | "
                    f"normal={st['normal']} normal_kept={st['normal_kept']} (kept_ratio={kept_ratio:.1f}%)"
                )
            self._mask_stat = self._new_mask_stat()
            self._mask_log_cnt = 0

    def _record_backchannel_filter_stat(self, stat_name, accepted, deleted):
        stat_name = stat_name or "unknown_bc"
        if stat_name not in self._bc_filter_stat:
            self._bc_filter_stat[stat_name] = {"accepted": 0, "deleted": 0}
        self._bc_filter_stat[stat_name]["accepted"] += int(accepted)
        self._bc_filter_stat[stat_name]["deleted"] += int(deleted)

        self._bc_filter_log_cnt += 1
        if self._bc_filter_log_cnt >= self._bc_filter_log_interval:
            if self._is_mask_log_worker():
                parts = []
                ordered_names = ["user_bc", "ai_bc"] + [n for n in self._bc_filter_stat.keys() if n not in ("user_bc", "ai_bc")]
                for name in ordered_names:
                    s = self._bc_filter_stat[name]
                    total = s["accepted"] + s["deleted"]
                    accept_ratio = (s["accepted"] / total * 100.0) if total > 0 else 0.0
                    delete_ratio = (s["deleted"] / total * 100.0) if total > 0 else 0.0
                    parts.append(
                        f"{name}: accepted={s['accepted']} deleted={s['deleted']} "
                        f"accept_ratio={accept_ratio:.1f}% delete_ratio={delete_ratio:.1f}%"
                    )
                print(f"[BCFilterStat] over {self._bc_filter_log_cnt} filter calls | " + " | ".join(parts))
            self._bc_filter_stat = self._new_bc_filter_stat()
            self._bc_filter_log_cnt = 0

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
    #         cur_len = 1  # Length bucketing is not used here.
    #         cur_type = self.DATA_TYPE_TO_IDS[sample]

    #         length_list.append((cur_len, cur_type))
    #     return length_list

    def mask_control_label(
        self,
        control_label,
        keep_token_ids,
        start_token_ids,
        normal_keep_ratio: float = 0.2,
        special_multiple: int = 4,
        rng: random.Random = None,
        forced_keep_positions = None,
        stat_bc = None,
    ):
        """
        Apply a sampled mask over two control-label classes: special and normal.

        Special tokens are always kept:
            * The first/last endpoint of every keep_*(K-*) run. A run of length
              one contributes that position only once.
            * Every start_*(S-*) token, since S-* tokens are single-point runs.
            * start_bc / keep_bc inside AI BC runs. The two currently share one
              token id, so the first token in a run is start_bc and later tokens
              are keep_bc.
            * forced_keep_positions, such as hard tokens covered by user BC.

        Normal tokens are interior positions of keep_* runs. Real labels that
        are neither keep/start/AI BC remain unchanged and do not participate in
        balancing. Existing IGNORE_INDEX positions remain unchanged.

        Keep rule:
            * Keep all special tokens.
            * Randomly keep min(round(normal_keep_ratio * N_normal),
              special_multiple * N_special) normal tokens.
            * Set the remaining normal tokens to IGNORE_INDEX.

        Args:
            control_label: 1D torch.LongTensor of length T.
            keep_token_ids: keep-class token ids, e.g. {keep_listening, keep_speaking}.
            start_token_ids: transition token ids, e.g. {start_speaking, start_listening, start_bc}.
            normal_keep_ratio: upper bound for the random keep ratio of normal tokens.
            special_multiple: upper bound for normal keeps as a multiple of special count.
            rng: optional random.Random instance for reproducible sampling.
            forced_keep_positions: extra positions to keep as special.

        Returns:
            masked_control_label: 1D LongTensor with the same shape and dtype as control_label.
        """
        assert control_label.dim() == 1, f"expect 1D tensor, got shape {tuple(control_label.shape)}"
        assert 0.0 <= normal_keep_ratio <= 1.0 and special_multiple >= 0, \
            f"bad ratio: normal_keep_ratio={normal_keep_ratio}, special_multiple={special_multiple}"

        keep_set = set(int(x) for x in keep_token_ids)
        start_set = set(int(x) for x in start_token_ids)
        assert keep_set.isdisjoint(start_set), "keep_token_ids and start_token_ids must be disjoint"
        bc_token_id = int(self.start_bc_token_id)
        generic_start_set = set(start_set)
        generic_start_set.discard(bc_token_id)

        _rand = rng if rng is not None else random

        masked = control_label.clone()
        seq = control_label.tolist()
        T = len(seq)

        # ---- Step 1: Scan positions ----
        # Consecutive keep_* labels, optionally separated by IGNORE, form a run.
        # start_* / AI BC / other semantic tokens terminate keep runs.
        # start_bc and keep_bc currently share the same token id, so the first
        # token in a contiguous BC run is start_bc and subsequent tokens are keep_bc.
        # IGNORE is skipped and does not terminate runs because each chunk is
        # [IGNORE] * (C - 1) + [keep].
        keep_runs = []          # List[List[int]], each run contains keep positions; IGNORE may appear between positions.
        bc_runs = []            # List[List[int]], each run contains AI BC positions; IGNORE may appear between positions.
        cur_run = []
        cur_bc_run = []
        start_positions = []    # Regular S-* positions, excluding start_bc.

        def _flush_keep_run():
            nonlocal cur_run
            if cur_run:
                keep_runs.append(cur_run)
                cur_run = []

        def _flush_bc_run():
            nonlocal cur_bc_run
            if cur_bc_run:
                bc_runs.append(cur_bc_run)
                cur_bc_run = []

        for i, v in enumerate(seq):
            if v == IGNORE_INDEX:
                continue
            if v == bc_token_id:
                _flush_keep_run()
                cur_bc_run.append(i)
            elif v in keep_set:
                _flush_bc_run()
                cur_run.append(i)
            elif v in generic_start_set:
                _flush_keep_run()
                _flush_bc_run()
                start_positions.append(i)
            else:
                # Other real tokens are kept as-is and terminate keep / AI BC runs.
                _flush_keep_run()
                _flush_bc_run()
        _flush_keep_run()
        _flush_bc_run()

        start_bc_positions = []
        keep_bc_positions = []
        for run in bc_runs:
            start_bc_positions.append(run[0])
            keep_bc_positions.extend(run[1:])

        # ---- Step 2: Split special (always kept) and normal (sampled) positions ----
        special_idx = set(start_positions) | set(start_bc_positions) | set(keep_bc_positions)
        keep_boundary_idx = set()               # First/last endpoints of K-* runs for stats.
        normal_idx = []                         # Interior positions of keep runs.
        for run in keep_runs:
            if len(run) == 1:
                keep_boundary_idx.add(run[0])
            else:
                keep_boundary_idx.add(run[0])
                keep_boundary_idx.add(run[-1])
                normal_idx.extend(run[1:-1])
        special_idx |= keep_boundary_idx
        if forced_keep_positions is not None:
            special_idx.update([int(p) for p in forced_keep_positions if 0 <= int(p) < T])
        # forced_keep_positions may be inside keep runs; remove upgraded
        # special positions from the normal candidates.
        normal_idx = [i for i in normal_idx if i not in special_idx]

        # ---- Step 3: Randomly keep min(20%, 4 * special_count) normal tokens ----
        n_special = len(special_idx)
        n_normal = len(normal_idx)
        keep_normal_num = min(int(round(normal_keep_ratio * n_normal)), special_multiple * n_special)
        keep_normal_num = max(0, min(keep_normal_num, n_normal))

        sampled_normal = set()
        if keep_normal_num > 0:
            sampled_normal = set(_rand.sample(normal_idx, keep_normal_num))

        # ---- Step 4: Mask unselected normal tokens; leave special / other semantic / IGNORE untouched ----
        for i in normal_idx:
            if i not in sampled_normal:
                masked[i] = IGNORE_INDEX

        # ---- Step 5: Count special-token categories by priority for periodic stats ----
        self._accumulate_mask_stat(
            start_positions=start_positions,
            keep_boundary_idx=keep_boundary_idx,
            start_bc_positions=start_bc_positions,
            keep_bc_positions=keep_bc_positions,
            stat_bc=stat_bc,
            n_normal=n_normal, n_normal_kept=keep_normal_num, T=T,
        )

        return masked



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

    def _filter_backchannels_by_time(self, backchannels, base_start_time, lead_silence_sec, min_gap_sec, max_num, stat_name=None):
        """Filter BCs by time relative to the other speaker clip.

        Applies leading-silence filtering, random sparse sampling, and max_num.
        """
        def is_close(bc1, bc2):
            if abs(bc1["start_time"] - bc2["start_time"]) < min_gap_sec:
                return True
            if abs(bc1["end_time"] - bc2["end_time"]) < min_gap_sec:
                return True
            return False

        total = len(backchannels) if backchannels is not None else 0
        if max_num <= 0 or total == 0:
            self._record_backchannel_filter_stat(stat_name, accepted=0, deleted=total)
            return []

        candidates = []
        for bc in backchannels:
            rel_start = float(bc["start_time"]) - float(base_start_time)
            if rel_start >= lead_silence_sec:
                candidates.append((rel_start, bc))

        selected = []
        while candidates and len(selected) < max_num:
            rel_start, bc = random.choice(candidates)
            selected.append((rel_start, bc))
            candidates = [(t, item) for t, item in candidates if not is_close(bc, item)]

        selected_backchannels = [bc for _, bc in sorted(selected, key=lambda x: x[0])]
        self._record_backchannel_filter_stat(
            stat_name,
            accepted=len(selected_backchannels),
            deleted=max(0, total - len(selected_backchannels)),
        )
        return selected_backchannels

    def apply_ai_backchannel(self, conv, user_info, input_len, input_token, stoken_token, stoken_mapping_token, control_input_token, contorl_label, text_label, stoken_label_token, overlap_token_num=0):
        if not self.enable_ai_bc:
            return

        ai_backchannels = self._filter_backchannels_by_time(
            conv["ai_backchannel"],
            user_info["start_time"],
            self.ai_bc_lead_silence_sec,
            self.ai_bc_min_gap_sec,
            self.ai_bc_max_num,
            stat_name="ai_bc",
        )

        for ai_backchannel in ai_backchannels:
            bc_clip_time = ai_backchannel["start_time"] - user_info["start_time"]
            raw_start_timestep = int(bc_clip_time / 0.04)

            start_chunk = math.ceil((raw_start_timestep - overlap_token_num) / self.control_token_chunk_size)
            start_timestep = overlap_token_num + start_chunk * self.control_token_chunk_size

            if raw_start_timestep <= overlap_token_num:
                print(f"[Datasets] bad AI_Backchannel: raw_start_timestep: {raw_start_timestep}, overlap_token_num: {overlap_token_num}, AI BC happened during user interruption")
                continue 
            if start_timestep >= len(input_token) - 1:
                print(f"[Datasets] bad AI_Backchannel: start_timestep: {start_timestep}, len(input_token): {len(input_token)}, bad start time, check your data!")
                continue 
                
            bc_token_ids, bc_stoken_ids, bc_mapping_ids = self.interleaved_tokenizer(
                ai_backchannel["text"],
                self.load_stoken(ai_backchannel["stoken"]),
                stoken_mapping_start=input_len + start_timestep,
            )

            max_bc_len = len(input_token) - 1 - start_timestep
            bc_token_ids = bc_token_ids[:max_bc_len]
            bc_stoken_ids = bc_stoken_ids[:max_bc_len]
            bc_mapping_ids = bc_mapping_ids[:max_bc_len]
            bc_end = start_timestep + len(bc_token_ids)

            end_chunk = math.ceil((bc_end - overlap_token_num) / self.control_token_chunk_size)
            total_chunk_num = (len(input_token) - overlap_token_num) // self.control_token_chunk_size
            bc_been_interruption = end_chunk >= total_chunk_num
            end_chunk = min(end_chunk, total_chunk_num) 
           
            end_timestep = end_chunk * self.control_token_chunk_size + overlap_token_num

            # Usually caused by overlap with another BC.
            if any(t != self.text_pad_token_id for t in input_token[start_timestep:end_timestep]):
                print(f"[Datasets] bad AI_Backchannel: ai bc has overlap, skip...")
                print(f"[Datasets] bad AI_Backchannel: input_token[start_timestep:end_timestep]: {self.tokenizer.decode(input_token[start_timestep:end_timestep])}")
                continue   
            if any(control_input_token[c * self.control_token_chunk_size + overlap_token_num - 1] != self.keep_listening_token_id for c in range(start_chunk, end_chunk)): # Excludes end_chunk, which may be start_speaking.
                print(f"[Datasets] bad AI_Backchannel: ai bc has overlap (control), skip...")
                print(f"[Datasets] bad AI_Backchannel: control_input_token[start_timestep:end_timestep]: {self.tokenizer.decode(control_input_token[start_timestep:end_timestep])}")
                continue

            bc_control_positions = []
            for c in range(start_chunk, end_chunk + 1):
                pos = c * self.control_token_chunk_size + overlap_token_num - 1
                if c != end_chunk:
                    assert control_input_token[pos] == self.keep_listening_token_id
                else:
                    if bc_been_interruption:
                        assert control_input_token[pos] == self.start_speaking_token_id
                    else:
                        assert control_input_token[pos] == self.keep_listening_token_id
                bc_control_positions.append(pos)

            for pos_id, pos in enumerate(bc_control_positions):
                if pos_id == 0:
                    control_input_token[pos] = self.start_bc_token_id
                    contorl_label[pos] = self.start_bc_token_id
                elif pos_id == len(bc_control_positions) - 1:
                    if not bc_been_interruption:
                        control_input_token[pos] = self.end_bc_token_id
                        contorl_label[pos] = self.end_bc_token_id
                else:
                    control_input_token[pos] = self.keep_bc_token_id
                    contorl_label[pos] = self.keep_bc_token_id

            for x, j in enumerate(range(start_timestep, bc_end)):
                assert input_token[j] == self.text_pad_token_id and text_label[j] == IGNORE_INDEX
                assert stoken_token[j] == self.stoken_pad_token_id and stoken_label_token[j] == IGNORE_INDEX
                assert stoken_mapping_token[j] == -1
                input_token[j] = bc_token_ids[x]
                stoken_token[j] = bc_stoken_ids[x]
                stoken_mapping_token[j] = bc_mapping_ids[x]
                text_label[j] = bc_token_ids[x] if bc_token_ids[x] != self.tts_pad_id else IGNORE_INDEX
                stoken_label_token[j] = bc_stoken_ids[x] if bc_stoken_ids[x] != self.tts_start_id and bc_stoken_ids[x] != self.stoken_delay_token_id else IGNORE_INDEX

            for j in range(bc_end, end_timestep):
                input_token[j] = self.text_pad_token_id
                stoken_token[j] = self.stoken_pad_token_id
            input_token[end_timestep - 1] = self.text_pad_token_id
            stoken_token[end_timestep - 1] = self.stoken_pad_token_id

    def tokens2silence(self, token_num, dtype):
        return np.zeros(token_num * self.AUDIO_TOKEN_N_SAMPLE_ALIGN, dtype=dtype)

    def load_audio(self, path=None, audio=None):
        if audio is None:
            audio = librosa.load(path, sr=self.SAMPLE_RATE)[0]
        original_len = audio.shape[0] / self.SAMPLE_RATE
        align_len = int(math.ceil(audio.shape[0] / self.AUDIO_TOKEN_N_SAMPLE)) * self.AUDIO_TOKEN_N_SAMPLE
        padding_len = align_len - len(audio)
        silence = np.zeros(padding_len, dtype=audio.dtype)
        audio = np.concatenate([audio, silence])
        token_num = audio.shape[0] // self.AUDIO_TOKEN_N_SAMPLE_ALIGN
        return audio, token_num, original_len

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
        # 1. Create the mask.
        # Shape: (batch_size, seq_len).
        # valid_mask[i, j] is True when M[i, j] != -1.
        valid_mask = (stoken_mapping != -1)

        # 2. Prepare gather indices.
        # Clone M to avoid modifying the original data. Replace -1 positions
        # with 0, or any legal index, to avoid gather out-of-bounds errors.
        # Those positions are masked out later, so using 0 is harmless.
        gather_indices = stoken_mapping.clone()
        gather_indices[~valid_mask] = 0
        
        # 3. Expand index dimensions to match A's hidden size.
        # gather_indices becomes (batch_size, seq_len, hidden_size).
        # We index on dim=1 while keeping all data on dim=2.
        # gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, A.size(-1))

        # 4. Gather along the sequence dimension.
        # A_selected[i, j, k] = A[i, gather_indices[i, j, k], k]
        # That is A[i, M[i,j], :].
        input_ids_selected = torch.gather(input_ids, dim=0, index=gather_indices)

        # 5. Add selected data to B.
        # Expand the mask by one dimension for broadcasting.
        # mask_expanded shape: (batch_size, seq_len, 1)
        # mask_expanded = valid_mask.unsqueeze(-1)
        
        return input_ids_selected * valid_mask
     
    def _load_user_segment(self, global_user_speech_wave, user_info):
        """Slice the current user speech segment and align/pad it with load_audio."""
        assert user_info["end_time"] > user_info["start_time"], "bad user speech segment"
        seg_start = int(round(user_info["start_time"] * self.SAMPLE_RATE))
        seg_end = int(round(user_info["end_time"] * self.SAMPLE_RATE))
        user_speech_wave = np.ascontiguousarray(global_user_speech_wave[seg_start:seg_end])
        user_speech_wave, user_speech_token_num, _ = self.load_audio(path=None, audio=user_speech_wave)
        return user_speech_wave, user_speech_token_num

    def _postprocess_user_audio(self, user_audio):
        """Concatenate the full user-audio track without noise or background-speech augmentation."""
        return np.concatenate(user_audio)

    def apply_mask_control_label(self, control_label_ids, forced_bc):
        """Control-label mask entrypoint.

        Keep-speaking tokens covered by user backchannel are always kept.
        """
        return self.mask_control_label(
            control_label_ids,
            keep_token_ids={self.keep_listening_token_id, self.keep_speaking_token_id},
            start_token_ids={self.start_speaking_token_id, self.start_listening_token_id, self.start_bc_token_id},
            normal_keep_ratio=0.2,
            special_multiple=3,
            forced_keep_positions=list(forced_bc),
            stat_bc=forced_bc,
        )

    def _append_initial_user_turn(self, acc, conv, user_info, user_speech_wave, user_speech_token_num):
        """Build a pure listening-state user segment for the first turn or after a completed AI turn."""
        chunk_num = math.ceil((user_speech_token_num + 1) / self.control_token_chunk_size) # +1 ensures <s-s> appears after the audio.
        padding_user_speech_token_num = chunk_num * self.control_token_chunk_size
        silence = self.tokens2silence(padding_user_speech_token_num - user_speech_token_num, dtype=user_speech_wave.dtype)
        user_speech_wave = np.concatenate([user_speech_wave, silence])

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
                # Select the next chunk as S-S.
                control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_speaking_token_id]
                contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_speaking_token_id]

        # AI BC
        input_len = sum([len(t) for t in acc["input_ids"]]) 
        self.apply_ai_backchannel(conv, user_info, input_len, input_token, stoken_token, stoken_mapping_token, control_input_token, contorl_label, text_label, stoken_label_token)

        acc["user_audio"].append(user_speech_wave)
        acc["input_ids"].append(input_token)
        acc["stoken_ids"].append(stoken_token)
        acc["stoken_mapping"].append(stoken_mapping_token)
        acc["control_input_ids"].append(control_input_token)
        acc["text_label_ids"].append(text_label)
        acc["stoken_label_ids"].append(stoken_label_token)
        acc["control_label_ids"].append(contorl_label)

    def _append_continued_user_turn(self, acc, conv, user_info, user_speech_wave, user_speech_token_num, state):
        """Previous AI turn was interrupted: append remaining AI tokens, then this user segment."""
        last_control_label       = state["last_control_label"]
        last_control_input_token = state["last_control_input_token"]
        last_ai_resposne_ids     = state["last_ai_resposne_ids"]
        last_ai_stoken           = state["last_ai_stoken"]
        last_ai_stoken_mapping   = state["last_ai_stoken_mapping"]

        assert last_control_label is not None
        chunk_num = max(1, math.ceil((user_speech_token_num - len(last_control_label)) / self.control_token_chunk_size))
        padding_user_speech_token_num = chunk_num * self.control_token_chunk_size + len(last_control_label)
        silence = self.tokens2silence(padding_user_speech_token_num - user_speech_token_num, dtype=user_speech_wave.dtype)
        user_speech_wave = np.concatenate([user_speech_wave, silence])
        
        input_token             = []
        stoken_token            = []
        stoken_mapping_token    = []
        control_input_token     = []
        contorl_label           = []
        stoken_label_token      = []
        text_label              = []

        # Inputs remaining from the previous turn.
        contorl_label       += copy.deepcopy(last_control_label)
        control_input_token += copy.deepcopy(last_control_input_token)
        input_token         += copy.deepcopy(last_ai_resposne_ids[:len(last_control_label)])
        stoken_token        += copy.deepcopy(last_ai_stoken[:len(last_control_label)])
        stoken_mapping_token+= last_ai_stoken_mapping[:len(last_control_label)]
        text_label          += [t if t != self.tts_pad_id else IGNORE_INDEX for t in copy.deepcopy(last_ai_resposne_ids[:len(last_control_label)])]
        stoken_label_token  += [t if t != self.tts_start_id and t != self.stoken_delay_token_id else IGNORE_INDEX for t in copy.deepcopy(last_ai_stoken[:len(last_control_label)])]

        if len(input_token) < len(contorl_label):
            padding_len = len(contorl_label) - len(input_token)
            text_label              += [IGNORE_INDEX]               * padding_len
            stoken_label_token      += [IGNORE_INDEX]               * padding_len
            input_token             += [self.text_pad_token_id]     * padding_len
            stoken_token            += [self.stoken_pad_token_id]   * padding_len
            stoken_mapping_token    += [-1]                         * padding_len
        else: # Control is already S-L here, so pad by default.
            input_token             = input_token[:-1]          + [self.text_pad_token_id]
            text_label              = text_label[:-1]           + [IGNORE_INDEX]
            stoken_token            = stoken_token[:-1]         + [self.stoken_pad_token_id]
            stoken_label_token      = stoken_label_token[:-1]   + [IGNORE_INDEX]
            stoken_mapping_token    = stoken_mapping_token[:-1] + [-1]
        
        overlap_token_num = len(input_token)

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
                # Select the next chunk as S-S.
                control_input_token += [self.sleep_token_id] * (self.control_token_chunk_size - 2) + [self.detect_token_id] + [self.start_speaking_token_id]
                contorl_label += [IGNORE_INDEX] * (self.control_token_chunk_size - 1) + [self.start_speaking_token_id]
        
        input_len = sum([len(t) for t in acc["input_ids"]]) 
        self.apply_ai_backchannel(conv, user_info, input_len, input_token, stoken_token, stoken_mapping_token, control_input_token, contorl_label, text_label, stoken_label_token, overlap_token_num=overlap_token_num)

        acc["user_audio"].append(user_speech_wave)
        acc["input_ids"].append(input_token)
        acc["stoken_ids"].append(stoken_token)
        acc["stoken_mapping"].append(stoken_mapping_token)
        acc["control_input_ids"].append(control_input_token)
        acc["text_label_ids"].append(text_label)
        acc["stoken_label_ids"].append(stoken_label_token)
        acc["control_label_ids"].append(contorl_label)

        assert not any([t in [self.tts_pad_id] for t in text_label])
        assert not any([t in [self.stoken_pad_token_id, self.tts_start_id, self.stoken_delay_token_id] for t in stoken_label_token])
        state["last_ai_resposne_ids"] = None
        state["last_ai_stoken"] = None
        state["last_ai_stoken_mapping"] = None
        state["last_control_label"] = None
        state["last_control_input_token"] = None

    def _append_ai_response_turn(self, acc, conv, conv_id, conversation, ai_info, user_speech_wave, state):
        """Process this AI response: truncate interrupted turns and carry leftovers; otherwise append the full segment with user BC."""
        ai_response = ai_info["text"]
        ai_response_codec = self.load_stoken(ai_info["stoken"])
        input_len = sum([len(t) for t in acc["input_ids"]])
        full_label, full_stoken, full_mapping = self.interleaved_tokenizer(ai_response, ai_response_codec, stoken_mapping_start=input_len)

        # Detect interruption. Timestamps are used only to compute the relative interruption point.
        is_interruption = conv["is_ai_been_interrupted"]
        if is_interruption:
            assert conv_id + 1 < len(conversation)
            ai_start_time = ai_info["start_time"]
            next_user_start_time = conversation[conv_id + 1]["user"]["start_time"]
            before_interruption_stoken_num = max(0, int((next_user_start_time - ai_start_time) / 0.04))
            before_interruption_stoken_num = min(before_interruption_stoken_num, len(ai_response_codec))
            before_interruption_len = self.stoken_delay_num + 1 + before_interruption_stoken_num
            before_interruption_len = min(before_interruption_len, len(full_stoken) - 1)

            ai_response_input_ids   = full_label[:before_interruption_len]
            ai_response_stoken      = full_stoken[:before_interruption_len]
            ai_response_mapping     = full_mapping[:before_interruption_len]
            last_ai_resposne_ids    = full_label[before_interruption_len:]
            last_ai_stoken          = full_stoken[before_interruption_len:]
            last_ai_stoken_mapping  = full_mapping[before_interruption_len:]

            # ai response
            acc["user_audio"].append(self.tokens2silence(token_num=len(ai_response_input_ids), dtype=user_speech_wave.dtype))
            acc["input_ids"].append(ai_response_input_ids)
            acc["stoken_ids"].append(ai_response_stoken)
            acc["stoken_mapping"].append(ai_response_mapping)
            acc["text_label_ids"].append(copy.deepcopy([t if t != self.tts_pad_id else IGNORE_INDEX for t in ai_response_input_ids]))
            acc["stoken_label_ids"].append(copy.deepcopy([t if t != self.tts_start_id and t != self.stoken_delay_token_id else IGNORE_INDEX for t in ai_response_stoken]))

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

            acc["control_input_ids"].append(control_input_token)
            acc["control_label_ids"].append(contorl_label)

            state["last_ai_resposne_ids"]     = last_ai_resposne_ids
            state["last_ai_stoken"]           = last_ai_stoken
            state["last_ai_stoken_mapping"]   = last_ai_stoken_mapping
            state["last_control_input_token"] = last_control_input_token
            state["last_control_label"]       = last_control_label

        else:
            ai_response_input_ids, ai_response_stoken, ai_response_mapping = full_label, full_stoken, full_mapping
            

            # +1 prevents exact divisibility by control_token_chunk_size from overwriting the S-L token.
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
            
            # User BC
            self._apply_user_backchannel(acc, conv, ai_info, user_speech_wave, input_len, chunk_num)

            acc["user_audio"].append(user_speech_wave)

            acc["input_ids"].append(padding_ai_response_input_ids)
            acc["text_label_ids"].append(padding_ai_response_label_ids)
            acc["stoken_ids"].append(padding_ai_response_stoken)
            acc["stoken_mapping"].append(padding_ai_response_mapping)
            acc["stoken_label_ids"].append(padding_ai_response_stoken_label)
            acc["control_input_ids"].append(control_input_token)
            acc["control_label_ids"].append(contorl_label)

    def _control_chunk_audio_second(self):
        """Return the user-audio duration covered by one control chunk, in seconds.

        User audio is arranged in the sequence as alternating [audio(80 ms), pad(0 ms)]
        entries because self.align_audio_input is always True. The overall grid is aligned
        to AUDIO_TOKEN_N_SAMPLE_ALIGN (=40 ms) slots. One control chunk has
        control_token_chunk_size slots, so it covers control_token_chunk_size * 40 ms of
        user audio.
        """
        slot_second = self.AUDIO_TOKEN_N_SAMPLE_ALIGN / self.SAMPLE_RATE  # One slot equals one AI-channel token duration, 40 ms.
        return self.control_token_chunk_size * slot_second

    def time2control_chunk_index(self, t):
        """Return the control-chunk index for time t relative to this user-audio segment.

        control_token_chunk_size is always even, and the detect token is at slot
        (size - 2) inside the chunk. Although detect is not at the chunk end, that slot
        corresponds to a real 80 ms audio token, so it can observe the whole chunk
        (size * 40 ms) of user audio. Therefore control ownership over user audio is a
        uniform split, so duration-based integer division is sufficient.
        """
        return int(math.floor(t / self._control_chunk_audio_second()))

    def control_chunk_index2time(self, chunk_idx):
        """Return the [start, end) user-audio time span controlled by a control chunk."""
        chunk_second = self._control_chunk_audio_second()
        start_time = chunk_idx * chunk_second
        end_time = (chunk_idx + 1) * chunk_second
        return start_time, end_time

    def control_token_index2time(self, token_idx):
        """Return the [start, end) user-audio time span controlled by a control token."""
        assert (token_idx + 1) % self.control_token_chunk_size == 0, f"token_idx={token_idx} is not the final control token in a chunk"
        return self.control_chunk_index2time(int((token_idx + 1) / self.control_token_chunk_size) - 1)

    def _apply_user_backchannel(self, acc, conv, ai_info, user_speech_wave, input_len, chunk_num):
        """Overlay user backchannel during a normal AI response and record covered control-label positions."""
        if not self.enable_user_bc:
            return
        user_backchannels = self._filter_backchannels_by_time(
            conv["user_backchannel"],
            ai_info["start_time"],
            self.user_bc_lead_silence_sec,
            self.user_bc_min_gap_sec,
            self.user_bc_max_num,
            stat_name="user_bc",
        )
        for user_backchannel in user_backchannels:
            try:
                user_backchannel_wave, user_backchannel_num, user_backchannel_original_len = self.load_audio(user_backchannel["clip_path"])
                # Align to user_speech_wave by samples instead of quantizing continuous time with int(t / 0.04).
                replace_start = int(round((user_backchannel["start_time"] - ai_info["start_time"]) * self.SAMPLE_RATE))
                if replace_start < 0:
                    print("[Datasets] bad User_Backchannel")
                    continue
                if replace_start >= len(user_speech_wave):
                    print("[Datasets] bad User_Backchannel")
                    continue
                user_backchannel_wave = user_backchannel_wave[:min(len(user_speech_wave)-replace_start, len(user_backchannel_wave))]
                user_backchannel_original_len = min(user_backchannel_original_len, user_backchannel_wave.shape[0] / self.SAMPLE_RATE)
                user_speech_wave[replace_start:replace_start+len(user_backchannel_wave)] = user_backchannel_wave

                # Actual user-audio span occupied by user BC, relative to this AI response start
                # and therefore to the start of control chunk 0.
                bc_start_time = replace_start / self.SAMPLE_RATE
                bc_end_time = (replace_start + len(user_backchannel_wave)) / self.SAMPLE_RATE

                # Use the time<->control-chunk mapping to locate chunks covered by user BC.
                # These labels are normally keep-speaking, but the user is speaking at this
                # moment, so they are harder tokens and must be kept.
                start_chunk = self.time2control_chunk_index(bc_start_time)
                end_chunk = self.time2control_chunk_index(bc_end_time)
                for c in range(start_chunk, min(end_chunk + 1, chunk_num)):
                    chunk_start_time, chunk_end_time = self.control_chunk_index2time(c)
                    # Count BC coverage only for intersecting half-open intervals to avoid
                    # false positives exactly on chunk boundaries.
                    if bc_start_time < chunk_end_time and bc_end_time > chunk_start_time:
                        label_pos = input_len + c * self.control_token_chunk_size + self.control_token_chunk_size - 1
                        acc["user_bc_control_label_positions"].append(label_pos)
            except Exception as e:
                print(f"[Datasets] bad User_Backchannel: {e}")

    def _assert_duplex_format_without_ai_bc(self, input_ids, stoken_ids, stoken_mapping, control_input_ids,
                                            text_label_ids, control_label_ids, stoken_label_ids):
        """Validate the global chunk state machine without AI backchannel."""
        C = self.control_token_chunk_size
        N = len(input_ids)
        assert C >= 2, f"bad control_token_chunk_size={C}"
        assert N % C == 0, f"sequence length {N} is not divisible by chunk size {C}"
        assert len(stoken_ids) == len(stoken_mapping) == len(control_input_ids) == len(text_label_ids) == len(control_label_ids) == len(stoken_label_ids) == N

        def _all_eq(x, value):
            return bool((x == value).all().item())

        def _first_text_pad_offset(s, e):
            pad_pos = (input_ids[s:e] == self.text_pad_token_id).nonzero(as_tuple=False)
            return int(pad_pos[0].item()) if pad_pos.numel() else e - s

        def _assert_control_chunk(s, e):
            assert _all_eq(control_label_ids[s:e - 1], IGNORE_INDEX), f"non-IGNORE control label inside chunk [{s}, {e})"
            assert _all_eq(control_input_ids[s:e - 2], self.sleep_token_id), f"bad sleep span in chunk [{s}, {e})"
            assert int(control_input_ids[e - 2].item()) == self.detect_token_id, f"missing detect token at {e - 2}"
            assert int(control_input_ids[e - 1].item()) == int(control_label_ids[e - 1].item()), f"control input/label mismatch at {e - 1}"

        def _assert_idle_payload(s, e):
            assert _all_eq(input_ids[s:e], self.text_pad_token_id), f"listening chunk [{s}, {e}) has non-pad text input"
            assert _all_eq(text_label_ids[s:e], IGNORE_INDEX), f"listening chunk [{s}, {e}) has text labels"
            assert _all_eq(stoken_ids[s:e], self.stoken_pad_token_id), f"listening chunk [{s}, {e}) has non-pad stoken input"
            assert _all_eq(stoken_label_ids[s:e], IGNORE_INDEX), f"listening chunk [{s}, {e}) has stoken labels"
            assert _all_eq(stoken_mapping[s:e], -1), f"listening chunk [{s}, {e}) has stoken mapping"

        def _assert_generated_payload(s, e, require_full, tag):
            active_len = _first_text_pad_offset(s, e)
            if require_full:
                assert active_len == e - s, f"{tag} chunk [{s}, {e}) is unexpectedly padded at offset {active_len}"
            else:
                assert active_len < e - s, f"{tag} transition chunk [{s}, {e}) should contain a padding suffix"

            active_s = s
            active_e = s + active_len
            if active_len > 0:
                active_input = input_ids[active_s:active_e]
                expected_text_label = torch.where(
                    active_input == self.tts_pad_id,
                    torch.full_like(active_input, IGNORE_INDEX),
                    active_input,
                )
                assert bool((text_label_ids[active_s:active_e] == expected_text_label).all().item()), f"bad text labels in {tag} active span [{active_s}, {active_e})"

                active_stoken = stoken_ids[active_s:active_e]
                assert bool((active_stoken != self.stoken_pad_token_id).all().item()), f"{tag} active span [{active_s}, {active_e}) has stoken pad"
                stoken_ignore = (active_stoken == self.tts_start_id) | (active_stoken == self.stoken_delay_token_id)
                expected_stoken_label = torch.where(
                    stoken_ignore,
                    torch.full_like(active_stoken, IGNORE_INDEX),
                    active_stoken,
                )
                assert bool((stoken_label_ids[active_s:active_e] == expected_stoken_label).all().item()), f"bad stoken labels in {tag} active span [{active_s}, {active_e})"

            if active_e < e:
                assert _all_eq(input_ids[active_e:e], self.text_pad_token_id), f"bad text padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(text_label_ids[active_e:e], IGNORE_INDEX), f"bad text label padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_ids[active_e:e], self.stoken_pad_token_id), f"bad stoken padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_label_ids[active_e:e], IGNORE_INDEX), f"bad stoken label padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_mapping[active_e:e], -1), f"bad stoken mapping padding suffix in {tag} chunk [{active_e}, {e})"

        listen_state = True
        for s in range(0, N, C):
            e = s + C
            _assert_control_chunk(s, e)
            last_label = int(control_label_ids[e - 1].item())

            if listen_state:
                assert last_label in [self.keep_listening_token_id, self.start_speaking_token_id], f"bad listening control label {last_label} at {e - 1}"
                _assert_idle_payload(s, e)
                if last_label == self.start_speaking_token_id:
                    listen_state = False
            else:
                assert last_label in [self.keep_speaking_token_id, self.start_listening_token_id], f"bad speaking control label {last_label} at {e - 1}"
                if last_label == self.keep_speaking_token_id:
                    _assert_generated_payload(s, e, require_full=True, tag="speaking")
                else:
                    _assert_generated_payload(s, e, require_full=False, tag="speaking-to-listening")
                    listen_state = True

        # assert listen_state, "sample ends before returning to listening state"

    def _assert_duplex_format_with_ai_bc(self, input_ids, stoken_ids, stoken_mapping, control_input_ids,
                                         text_label_ids, control_label_ids, stoken_label_ids):
        """Validate the global chunk state machine with AI backchannel."""
        C = self.control_token_chunk_size
        N = len(input_ids)
        assert C >= 2, f"bad control_token_chunk_size={C}"
        assert N % C == 0, f"sequence length {N} is not divisible by chunk size {C}"
        assert len(stoken_ids) == len(stoken_mapping) == len(control_input_ids) == len(text_label_ids) == len(control_label_ids) == len(stoken_label_ids) == N

        def _all_eq(x, value):
            return bool((x == value).all().item())

        def _first_text_pad_offset(s, e):
            pad_pos = (input_ids[s:e] == self.text_pad_token_id).nonzero(as_tuple=False)
            return int(pad_pos[0].item()) if pad_pos.numel() else e - s

        def _assert_control_chunk(s, e):
            assert _all_eq(control_label_ids[s:e - 1], IGNORE_INDEX), f"non-IGNORE control label inside chunk [{s}, {e})"
            assert _all_eq(control_input_ids[s:e - 2], self.sleep_token_id), f"bad sleep span in chunk [{s}, {e})"
            assert int(control_input_ids[e - 2].item()) == self.detect_token_id, f"missing detect token at {e - 2}"
            assert int(control_input_ids[e - 1].item()) == int(control_label_ids[e - 1].item()), f"control input/label mismatch at {e - 1}"

        def _assert_idle_payload(s, e):
            assert _all_eq(input_ids[s:e], self.text_pad_token_id), f"idle chunk [{s}, {e}) has non-pad text input"
            assert _all_eq(text_label_ids[s:e], IGNORE_INDEX), f"idle chunk [{s}, {e}) has text labels"
            assert _all_eq(stoken_ids[s:e], self.stoken_pad_token_id), f"idle chunk [{s}, {e}) has non-pad stoken input"
            assert _all_eq(stoken_label_ids[s:e], IGNORE_INDEX), f"idle chunk [{s}, {e}) has stoken labels"
            assert _all_eq(stoken_mapping[s:e], -1), f"idle chunk [{s}, {e}) has stoken mapping"

        def _assert_generated_payload(s, e, require_full, tag):
            active_len = _first_text_pad_offset(s, e)
            if require_full:
                assert active_len == e - s, f"{tag} chunk [{s}, {e}) is unexpectedly padded at offset {active_len}"
            else:
                assert active_len < e - s, f"{tag} transition chunk [{s}, {e}) should contain a padding suffix"

            active_s = s
            active_e = s + active_len
            if active_len > 0:
                active_input = input_ids[active_s:active_e]
                expected_text_label = torch.where(
                    active_input == self.tts_pad_id,
                    torch.full_like(active_input, IGNORE_INDEX),
                    active_input,
                )
                assert bool((text_label_ids[active_s:active_e] == expected_text_label).all().item()), f"bad text labels in {tag} active span [{active_s}, {active_e})"

                active_stoken = stoken_ids[active_s:active_e]
                assert bool((active_stoken != self.stoken_pad_token_id).all().item()), f"{tag} active span [{active_s}, {active_e}) has stoken pad"
                stoken_ignore = (active_stoken == self.tts_start_id) | (active_stoken == self.stoken_delay_token_id)
                expected_stoken_label = torch.where(
                    stoken_ignore,
                    torch.full_like(active_stoken, IGNORE_INDEX),
                    active_stoken,
                )
                assert bool((stoken_label_ids[active_s:active_e] == expected_stoken_label).all().item()), f"bad stoken labels in {tag} active span [{active_s}, {active_e})"

            if active_e < e:
                assert _all_eq(input_ids[active_e:e], self.text_pad_token_id), f"bad text padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(text_label_ids[active_e:e], IGNORE_INDEX), f"bad text label padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_ids[active_e:e], self.stoken_pad_token_id), f"bad stoken padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_label_ids[active_e:e], IGNORE_INDEX), f"bad stoken label padding suffix in {tag} chunk [{active_e}, {e})"
                assert _all_eq(stoken_mapping[active_e:e], -1), f"bad stoken mapping padding suffix in {tag} chunk [{active_e}, {e})"

        listen_state = True
        ai_bc_state = False
        for s in range(0, N, C):
            e = s + C
            _assert_control_chunk(s, e)
            last_label = int(control_label_ids[e - 1].item())

            if listen_state:
                if ai_bc_state:
                    if last_label == self.keep_bc_token_id:
                        _assert_generated_payload(s, e, require_full=True, tag="ai-bc")
                    elif last_label == self.end_bc_token_id:
                        _assert_generated_payload(s, e, require_full=False, tag="ai-bc-to-listening")
                        ai_bc_state = False
                    elif last_label == self.start_speaking_token_id:
                        _assert_generated_payload(s, e, require_full=False, tag="ai-bc-to-speaking")
                        ai_bc_state = False
                        listen_state = False
                    else:
                        assert False, f"bad AI-BC control label {last_label} at {e - 1}"
                else:
                    assert last_label in [self.keep_listening_token_id, self.start_speaking_token_id, self.start_bc_token_id], f"bad listening control label {last_label} at {e - 1}"
                    _assert_idle_payload(s, e)
                    if last_label == self.start_bc_token_id:
                        ai_bc_state = True
                    elif last_label == self.start_speaking_token_id:
                        listen_state = False
            else:
                assert not ai_bc_state, "AI-BC state leaked into speaking state"
                assert last_label in [self.keep_speaking_token_id, self.start_listening_token_id], f"bad speaking control label {last_label} at {e - 1}"
                if last_label == self.keep_speaking_token_id:
                    _assert_generated_payload(s, e, require_full=True, tag="speaking")
                else:
                    _assert_generated_payload(s, e, require_full=False, tag="speaking-to-listening")
                    listen_state = True

        # assert listen_state and not ai_bc_state, "sample ends in unfinished speaking or AI-BC state"

    def __getitem__(self, i):
        item_index = index = self.data_index[i]
        source_data = self.data[index[0]][index[1]]
        conversation = source_data["turns"]
        
        # Accumulator for tokens and audio segments assembled from all turns.
        acc = {
            "user_audio": [],
            "input_ids": [],
            "stoken_ids": [],
            "stoken_mapping": [],
            "control_input_ids": [],
            "text_label_ids": [],
            "stoken_label_ids": [],
            "control_label_ids": [],
            "user_bc_control_label_positions": [],
        }

        # Cross-turn leftover state used when the previous AI response was interrupted.
        state = {
            "last_ai_resposne_ids": None,
            "last_control_label": None,
            "last_control_input_token": None,
            "last_ai_stoken": None,
            "last_ai_stoken_mapping": None,
        }

        # Load the full conversation audio once and slice by timestamps inside the loop to
        # avoid memory growth from repeated per-turn librosa.load calls.
        global_user_speech_wave = librosa.load(
            source_data["user_audio_path"],
            sr=self.SAMPLE_RATE,
        )[0]

        for conv_id, conv in enumerate(conversation):
            # Snapshot accumulator lengths before this turn so an overlong turn can be rolled back.
            _snap_len = {name: len(lst) for name, lst in acc.items()}

            user_info = conv["user"]
            ai_info = conv["assistant"]

            # 1. Slice and preprocess this turn's user-audio segment.
            user_speech_wave, user_speech_token_num = self._load_user_segment(
                global_user_speech_wave, user_info
            )

            # 2. Handle this user segment, either initial/normal or continuing an interrupted AI turn.
            if state["last_ai_resposne_ids"] is None:
                self._append_initial_user_turn(
                    acc, conv, user_info, user_speech_wave, user_speech_token_num
                )
            else:
                self._append_continued_user_turn(
                    acc, conv, user_info, user_speech_wave, user_speech_token_num, state
                )

            # 3. Handle this AI response, either interrupted or normally completed.
            self._append_ai_response_turn(
                acc, conv, conv_id, conversation, ai_info, user_speech_wave, state
            )

            # 4. After appending this turn, check total length. If it exceeds the limit,
            # discard this turn by rolling back to the snapshot and then stop.
            # This keeps the final sample strictly within max_data_length and avoids OOM
            # when downstream lm_head materializes full-vocabulary logits.
            cur_total_len = sum(len(t) for t in acc["input_ids"])
            if cur_total_len > self.max_data_length:
                if _snap_len["input_ids"] > 0:
                    # At least one turn was already appended, so roll back this turn and stop.
                    print(f"[Data] length {cur_total_len} > {self.max_data_length}, drop turn {conv_id} (rollback) and stop.")
                    for _name, _lst in acc.items():
                        del _lst[_snap_len[_name]:]
                    break
                else:
                    # The first turn alone is overlong, which is occasional dirty data.
                    # Drop this sample and randomly resample another one.
                    new_index = random.randint(0, len(self.data_index) - 1)
                    print(f"[Data] WARNING: first turn alone length {cur_total_len} > {self.max_data_length}, "
                          f"skip and resample. idx={i}, path={source_data['user_audio_path']}, new_index={new_index}")
                    del global_user_speech_wave
                    return self.__getitem__(new_index)

        # Release the full conversation audio once the loop is done to avoid resident memory buildup.
        del global_user_speech_wave

        # Unpack the accumulator and keep the original local variable names below.
        user_audio                      = acc["user_audio"]
        input_ids                       = acc["input_ids"]
        stoken_ids                      = acc["stoken_ids"]
        stoken_mapping                  = acc["stoken_mapping"]
        control_input_ids               = acc["control_input_ids"]
        text_label_ids                  = acc["text_label_ids"]
        stoken_label_ids                = acc["stoken_label_ids"]
        control_label_ids               = acc["control_label_ids"]
        forced_bc_positions             = acc["user_bc_control_label_positions"]

        # ------------------- debug -------------------

        for a, i, c, t, cl, si, sli, sm in list(zip(user_audio, input_ids, control_input_ids, text_label_ids, control_label_ids, stoken_ids, stoken_label_ids, stoken_mapping)):
            q = self.stepaudio_audio_prepreocess(a, debug=True)
            assert len(q['input_ids']) == len(t) == len(i) == len(c) == len(cl) == len(si) == len(sli) == len(sm)

        user_audio = self._postprocess_user_audio(user_audio)
        
        user_audio_input = self.stepaudio_audio_prepreocess(user_audio) 
        input_ids           = torch.cat([torch.tensor(t, dtype=torch.long) for t in input_ids], dim=0)
        stoken_ids          = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_ids], dim=0)
        stoken_mapping      = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_mapping], dim=0)
        control_input_ids   = torch.cat([torch.tensor(t, dtype=torch.long) for t in control_input_ids], dim=0)
        text_label_ids      = torch.cat([torch.tensor(t, dtype=torch.long) for t in text_label_ids], dim=0)
        stoken_label_ids    = torch.cat([torch.tensor(t, dtype=torch.long) for t in stoken_label_ids], dim=0)
        control_label_ids   = torch.cat([torch.tensor(t, dtype=torch.long) for t in control_label_ids], dim=0)

        try:
            assert len(user_audio_input['input_ids']) == len(input_ids) == len(stoken_ids) == len(control_input_ids) == len(text_label_ids) == len(stoken_label_ids) == len(control_label_ids)  == len(stoken_mapping)
        except:
            print(len(user_audio_input['input_ids']),len(input_ids), len(stoken_ids) , len(control_input_ids) , len(text_label_ids), len(stoken_label_ids),len(control_label_ids))
            assert len(user_audio_input['input_ids']) == len(input_ids) + 1 and self.align_audio_input
            user_audio_input['input_ids'] = user_audio_input['input_ids'][:-1]
            assert len(user_audio_input['input_ids']) == len(input_ids) == len(stoken_ids) == len(control_input_ids) == len(text_label_ids) == len(stoken_label_ids) == len(control_label_ids)  == len(stoken_mapping)

        # debug: before control-label sampling mask and no_stoken_label rewriting, validate
        # the global state-machine simulation with the complete labels.
        # if self.enable_ai_bc:
        #     self._assert_duplex_format_with_ai_bc(
        #         input_ids, stoken_ids, stoken_mapping, control_input_ids,
        #         text_label_ids, control_label_ids, stoken_label_ids,
        #     )
        # else:
        #     self._assert_duplex_format_without_ai_bc(
        #         input_ids, stoken_ids, stoken_mapping, control_input_ids,
        #         text_label_ids, control_label_ids, stoken_label_ids,
        #     )

        if self.no_stoken_label:
            input_ids[stoken_ids==self.tts_end_id] = self.tts_end_id

        assert (text_label_ids == self.text_pad_token_id).sum() == 0
        assert (text_label_ids == self.tts_pad_id).sum() == 0, f"{text_label_ids.tolist()}"
        assert (stoken_label_ids == self.stoken_pad_token_id).sum() == 0
        stoken_label_ids[stoken_label_ids==self.stoken_delay_token_id] = IGNORE_INDEX

        feats = user_audio_input["feats"]
        feats_lengths = torch.tensor(user_audio_input["feats_lengths"], dtype=torch.torch.int32)
        audio_input_ids = user_audio_input["input_ids"]
        del user_audio_input

        control_label_ids = self.apply_mask_control_label(
            control_label_ids,
            forced_bc_positions,
        )

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
            "user_wav": user_audio,
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
            "prefix_input_ids": prefix_input_ids,
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
