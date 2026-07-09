import os
import math
import time
from typing import Optional

import torch
from transformers import LogitsProcessorList

from lychee_fd.runtime.hf_v9_generation import (
    BackChannelLogitsProcessor,
    ListeningControlLogitsProcessor,
    SingleTurnGenerationFramework as _BaseV9Framework,
    SpeakingControlLogitsProcessor,
)
from lychee_fd.runtime.vllm_generation import (
    FixedTokenLogitsProcessor,
    HFIncrementalChunkStreamSession as _V6HFIncrementalChunkStreamSession,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


class _HFV9MultiHeadStreamAdapter:
    """
    Adapter for HF V9 model:
    - keeps existing .multi_head_generate behavior
    - exposes .multi_head_generate_stream for the realtime incremental session
    - carries merge_past_key_values across chunk calls keyed by the main KV cache
    """

    def __init__(self, model):
        self._model = model
        self.config = model.config
        self.device = getattr(model, "device", None)
        self.stream_generation_flag = getattr(model, "stream_generation_flag", False)
        self._merge_cache_by_past_id = {}

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _inject_merge_cache(self, kwargs):
        if kwargs.get("merge_past_key_values", None) is not None:
            return kwargs
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            return kwargs
        cached = self._merge_cache_by_past_id.get(id(past_key_values))
        if cached is not None:
            kwargs = dict(kwargs)
            kwargs["merge_past_key_values"] = cached
        return kwargs

    def _remember_merge_cache(self, result):
        if not isinstance(result, dict):
            return
        past_key_values = result.get("past_key_values", None)
        merge_past_key_values = result.get("merge_past_key_values", None)
        if past_key_values is not None and merge_past_key_values is not None:
            self._merge_cache_by_past_id[id(past_key_values)] = merge_past_key_values

    def multi_head_generate(self, **kwargs):
        kwargs = self._inject_merge_cache(kwargs)
        result = self._model.multi_head_generate(**kwargs)
        self._remember_merge_cache(result)
        return result

    def multi_head_generate_stream(self, **kwargs):
        kwargs = self._inject_merge_cache(kwargs)
        stream_fn = getattr(self._model, "multi_head_generate_stream", None)
        if callable(stream_fn):
            for result in stream_fn(**kwargs):
                self._remember_merge_cache(result)
                yield result
            return

        base_input_ids = kwargs.get("input_ids", None)
        base_stoken_ids = kwargs.get("stoken_ids", None)
        base_control_ids = kwargs.get("control_input_ids", None)

        result = self._model.multi_head_generate(**kwargs)
        self._remember_merge_cache(result)
        if not isinstance(result, dict):
            return

        seq = result.get("sequences", base_input_ids)
        stoken_seq = result.get("stoken_ids", base_stoken_ids)
        control_seq = result.get("control_ids", base_control_ids)
        if not (torch.is_tensor(seq) and torch.is_tensor(stoken_seq) and torch.is_tensor(control_seq)):
            return

        prev_text_len = int(base_input_ids.shape[1]) if torch.is_tensor(base_input_ids) else 0
        prev_stoken_len = int(base_stoken_ids.shape[1]) if torch.is_tensor(base_stoken_ids) else 0
        prev_control_len = int(base_control_ids.shape[1]) if torch.is_tensor(base_control_ids) else 0

        text_new = int(seq.shape[1]) - prev_text_len
        stoken_new = int(stoken_seq.shape[1]) - prev_stoken_len
        control_new = int(control_seq.shape[1]) - prev_control_len
        if not (text_new == stoken_new == control_new):
            raise RuntimeError(
                "HF V9 stream adapter channel delta mismatch: "
                f"text_new={text_new}, stoken_new={stoken_new}, control_new={control_new}, "
                f"base_lens=({prev_text_len},{prev_stoken_len},{prev_control_len}), "
                f"new_lens=({int(seq.shape[1])},{int(stoken_seq.shape[1])},{int(control_seq.shape[1])})"
            )

        for step_idx in range(int(text_new)):
            text_pos = prev_text_len + step_idx
            stoken_pos = prev_stoken_len + step_idx
            control_pos = prev_control_len + step_idx
            yield {
                "sequences": seq[:, : text_pos + 1],
                "stoken_ids": stoken_seq[:, : stoken_pos + 1],
                "control_ids": control_seq[:, : control_pos + 1],
                "past_key_values": result.get("past_key_values", None),
                "stoken_past_key_values": result.get("stoken_past_key_values", None),
                "control_past_key_values": result.get("control_past_key_values", None),
                "merge_past_key_values": result.get("merge_past_key_values", None),
                "text_token": int(seq[0, text_pos].item()),
                "stoken_token": int(stoken_seq[0, stoken_pos].item()),
                "control_token": int(control_seq[0, control_pos].item()),
                "step": int(step_idx),
            }


class HFRealtimeV9IncrementalChunkStreamSession(_V6HFIncrementalChunkStreamSession):
    """
    V9-specific HF realtime session.

    The V6 session generates all listening pads token-by-token. V9 offline
    instead prefills the new audio-aligned pad positions first, then samples
    one control token. This class follows the V9 offline state machine while
    keeping the shared realtime audio cache/timeline helpers from V6.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_past_key_values = None

    def _empty_result(self):
        return {
            "sequences": self.generated_input_ids,
            "stoken_ids": self.generated_stoken_ids,
            "control_ids": self.generated_control_ids,
            "past_key_values": self.past_key_values,
            "stoken_past_key_values": self.stoken_past_key_values,
            "control_past_key_values": self.control_past_key_values,
            "merge_past_key_values": self.merge_past_key_values,
        }

    def _update_caches_from_result(self, result, *, update_sequences: bool = True) -> None:
        if not isinstance(result, dict):
            return
        self.past_key_values = result.get("past_key_values", self.past_key_values)
        self.stoken_past_key_values = result.get(
            "stoken_past_key_values", self.stoken_past_key_values
        )
        self.control_past_key_values = result.get(
            "control_past_key_values", self.control_past_key_values
        )
        self.merge_past_key_values = result.get(
            "merge_past_key_values", self.merge_past_key_values
        )
        if update_sequences:
            self.generated_input_ids = result.get("sequences", self.generated_input_ids)
            self.generated_stoken_ids = result.get("stoken_ids", self.generated_stoken_ids)
            self.generated_control_ids = result.get("control_ids", self.generated_control_ids)

    @staticmethod
    def _processor_prob(proc_obj, key, fallback):
        probs = getattr(proc_obj, "last_probs", None)
        if not isinstance(probs, dict):
            return fallback
        value = probs.get(key)
        if value is None:
            return fallback
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(value):
            return fallback
        return value

    def _safe_prob(self, proc_obj, key):
        cache_attr = {
            "sl": "latest_sl_prob",
            "ss": "latest_ss_prob",
            "ks": "latest_ks_prob",
            "kl": "latest_kl_prob",
            "bc": "latest_bc_prob",
        }.get(key)
        cached = getattr(self, cache_attr, None) if cache_attr else None
        value = self._processor_prob(proc_obj, key, cached)
        if cache_attr is not None and value is not None:
            setattr(self, cache_attr, value)
        return value

    def _append_tokens(self, text_token: int, stoken_token: int, control_token: int) -> None:
        text_ids = int(text_token) * torch.ones(
            (self.generated_input_ids.shape[0], 1),
            dtype=self.generated_input_ids.dtype,
            device=self.generated_input_ids.device,
        )
        stoken_ids = int(stoken_token) * torch.ones(
            (self.generated_stoken_ids.shape[0], 1),
            dtype=self.generated_stoken_ids.dtype,
            device=self.generated_stoken_ids.device,
        )
        control_ids = int(control_token) * torch.ones(
            (self.generated_control_ids.shape[0], 1),
            dtype=self.generated_control_ids.dtype,
            device=self.generated_control_ids.device,
        )
        self.generated_input_ids = torch.cat((self.generated_input_ids, text_ids), dim=1)
        self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, stoken_ids), dim=1)
        self.generated_control_ids = torch.cat((self.generated_control_ids, control_ids), dim=1)

    def _pad_to_target_after_eos(self, padding_len: int, control_tail_token: int) -> None:
        padding_len = max(0, int(padding_len))
        if padding_len <= 0:
            return
        padding_input_ids = self.fw.text_pad_token_id * torch.ones(
            (self.generated_input_ids.shape[0], padding_len),
            dtype=self.generated_input_ids.dtype,
            device=self.generated_input_ids.device,
        )
        padding_stoken_ids = self.fw.stoken_pad_token_id * torch.ones(
            (self.generated_stoken_ids.shape[0], padding_len),
            dtype=self.generated_stoken_ids.dtype,
            device=self.generated_stoken_ids.device,
        )
        padding_control_ids = self.fw.sleep_token_id * torch.ones(
            (self.generated_control_ids.shape[0], padding_len),
            dtype=self.generated_control_ids.dtype,
            device=self.generated_control_ids.device,
        )
        if padding_control_ids.shape[1] > 2:
            padding_control_ids[:, -2] = self.fw.detect_token_id
        padding_control_ids[:, -1] = int(control_tail_token)
        self.generated_input_ids = torch.cat((self.generated_input_ids, padding_input_ids), dim=1)
        self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, padding_stoken_ids), dim=1)
        self.generated_control_ids = torch.cat((self.generated_control_ids, padding_control_ids), dim=1)

    def _extend_speaking_stoken_mapping(self, target_new_length: int) -> None:
        if self.last_ss_pos is None:
            self._set_last_ss_pos(max(self.system_input_length + 1, int(self.generated_input_ids.shape[1])))
        stoken_mapping_len = self.stoken_mapping.shape[1] - self.last_ss_pos
        if stoken_mapping_len <= self.fw.stoken_delay_num + 1:
            warmup_len = max(0, min(target_new_length, self.fw.stoken_delay_num + 1 - stoken_mapping_len))
            if warmup_len > 0:
                self.stoken_mapping = torch.cat(
                    [
                        self.stoken_mapping,
                        torch.full((1, warmup_len), -1, dtype=torch.long, device=self.fw.device),
                    ],
                    dim=1,
                )
        if self.stoken_mapping.shape[1] - self.generated_input_ids.shape[1] < target_new_length:
            padding_stoken_mapping = []
            for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + target_new_length):
                seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
            padding_stoken_mapping = torch.tensor(
                padding_stoken_mapping,
                dtype=torch.long,
                device=self.fw.model.device,
            ).unsqueeze(0)
            self.stoken_mapping = torch.cat((self.stoken_mapping, padding_stoken_mapping), dim=1)

    def _extend_mapping_for_padding_len(self, padding_len: int) -> None:
        if self.last_ss_pos is None:
            return
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
            padding_stoken_mapping = []
            for p in range(self.stoken_mapping.shape[1], self.generated_input_ids.shape[1] + padding_len):
                seq_l = p - self.last_ss_pos - (self.fw.stoken_delay_num + 1)
                padding_stoken_mapping.append(seq_l // 4 + self.last_ss_pos)
            padding_stoken_mapping = torch.tensor(
                padding_stoken_mapping,
                dtype=torch.long,
                device=self.fw.model.device,
            ).unsqueeze(0)
            self.stoken_mapping = torch.cat((self.stoken_mapping, padding_stoken_mapping), dim=1)

    def _stream_model(self, **kwargs):
        kwargs.setdefault("merge_past_key_values", self.merge_past_key_values)
        for token_result in self.fw.model.multi_head_generate_stream(**kwargs):
            yield token_result

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
        if audio_input_ids is None or total_len <= 0:
            if emit_generation_complete:
                yield self.build_generation_complete_event()
            return

        chunk_end = math.ceil(total_len / self.fw.control_token_chunk_size) * self.fw.control_token_chunk_size
        chunk_start = self.next_chunk_pos
        if chunk_start > chunk_end:
            if emit_generation_complete:
                yield self.build_generation_complete_event()
            return

        total_chunks = (chunk_end - chunk_start) // self.fw.control_token_chunk_size + 1

        for cn_e in range(chunk_start, chunk_end + 1, self.fw.control_token_chunk_size):
            target_length = min(cn_e, total_len)
            current_len = self.generated_input_ids.shape[1] - self.system_input_length
            target_new_length = target_length - current_len

            yield {
                "type": "chunk_start",
                "chunk_idx": self.chunk_idx,
                "chunk_pos": cn_e,
                "total_chunks": total_chunks,
            }

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
                    )
                ])
                listening_control_core = listening_control_processor[0] if len(listening_control_processor) > 0 else None

                if target_new_length > 1:
                    next_input_ids = torch.full(
                        (1, target_new_length - 1),
                        self.fw.text_pad_token_id,
                        dtype=torch.long,
                        device=self.fw.device,
                    )
                    next_stoken_ids = torch.full(
                        (1, target_new_length - 1),
                        self.fw.stoken_pad_token_id,
                        dtype=torch.long,
                        device=self.fw.device,
                    )
                    next_control_ids = torch.full(
                        (1, target_new_length - 1),
                        self.fw.sleep_token_id,
                        dtype=torch.long,
                        device=self.fw.device,
                    )
                    if target_length % self.fw.control_token_chunk_size == 0:
                        next_control_ids[0, -1] = self.fw.detect_token_id
                    self.generated_input_ids = torch.cat([self.generated_input_ids, next_input_ids], dim=1)
                    self.generated_stoken_ids = torch.cat([self.generated_stoken_ids, next_stoken_ids], dim=1)
                    self.generated_control_ids = torch.cat([self.generated_control_ids, next_control_ids], dim=1)

                self.stoken_mapping = torch.cat(
                    [
                        self.stoken_mapping,
                        torch.full((1, target_new_length), -1, dtype=torch.long, device=self.fw.device),
                    ],
                    dim=1,
                )

                last_result = None
                t_model_start = time.perf_counter()
                model_start_epoch_ms = int(time.time() * 1000)
                for token_result in self._stream_model(
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
                    control_logits_processor=listening_control_processor,
                    temperature=1.0,
                    top_k=0,
                    eos_token_id=None,
                ):
                    last_result = token_result
                self._record_timeline_span(
                    "transformer",
                    t_model_start,
                    model_start_epoch_ms,
                    state="listening",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=1,
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event

                if last_result is not None:
                    self._update_caches_from_result(last_result, update_sequences=False)
                    pred_control_token = int(last_result["control_ids"][0, -1].item())
                else:
                    pred_control_token = int(self.fw.kl_token_id)
                self._append_tokens(
                    self.fw.text_pad_token_id,
                    self.fw.stoken_pad_token_id,
                    pred_control_token,
                )

                yield {
                    "type": "control_decision",
                    "state": "l",
                    "chunk": cn_e,
                    "token": int(pred_control_token),
                    "ss_prob": self._safe_prob(listening_control_core, "ss"),
                    "sl_prob": self._safe_prob(listening_control_core, "sl"),
                    "kl_prob": self._safe_prob(listening_control_core, "kl"),
                    "bc_prob": self._safe_prob(listening_control_core, "bc"),
                }

                if pred_control_token == int(self.fw.ss_token_id):
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "ss_prob": self._safe_prob(listening_control_core, "ss"),
                        "sl_prob": self._safe_prob(listening_control_core, "sl"),
                    }
                    self.listening_state = "s"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])
                elif pred_control_token == int(self.fw.bc_token_id):
                    yield {
                        "type": "state_change",
                        "from": "L",
                        "to": "B",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_bc",
                        "ss_prob": self._safe_prob(listening_control_core, "ss"),
                        "sl_prob": self._safe_prob(listening_control_core, "sl"),
                    }
                    self.listening_state = "b"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])

            elif self.listening_state == "b":
                self._extend_speaking_stoken_mapping(target_new_length)
                speaking_control_processor = LogitsProcessorList([
                    BackChannelLogitsProcessor(
                        sleep_token_id=self.fw.sleep_token_id,
                        detect_token_id=self.fw.detect_token_id,
                        end_bc_token_id=self.fw.end_bc_token_id,
                        ss_token_id=self.fw.ss_token_id,
                        keep_bc_token_id=self.fw.keep_bc_token_id,
                        vocab_size=self.fw.model.config.text_config.vocab_size,
                        prefix_input_len=self.prefix_input_ids.shape[1],
                        control_token_chunk_size=self.fw.control_token_chunk_size,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None
                last_result = None
                t_model_start = time.perf_counter()
                model_start_epoch_ms = int(time.time() * 1000)
                for token_result in self._stream_model(
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
                    do_sample=True,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.fw.tts_end_id,
                ):
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": self._safe_prob(speaking_control_core, "sl"),
                        "ss_prob": self._safe_prob(speaking_control_core, "ss"),
                        "ks_prob": self._safe_prob(speaking_control_core, "ks"),
                    }
                    last_result = token_result
                self._record_timeline_span(
                    "transformer",
                    t_model_start,
                    model_start_epoch_ms,
                    state="backchannel",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=int(target_new_length),
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event
                if last_result is None:
                    last_result = self._empty_result()
                self._update_caches_from_result(last_result, update_sequences=True)

                last_control = int(self.generated_control_ids[0, -1].item())
                if last_control == int(self.fw.end_bc_token_id):
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "L",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_end_bc",
                        "sl_prob": self._safe_prob(speaking_control_core, "sl"),
                        "ss_prob": self._safe_prob(speaking_control_core, "ss"),
                    }
                    self.listening_state = "l"
                    self._set_last_ss_pos(None)
                    self.generated_stoken_ids = self.fw._set_last_token_safe(
                        self.generated_stoken_ids, self.fw.stoken_pad_token_id
                    )
                    self.generated_input_ids = self.fw._set_last_token_safe(
                        self.generated_input_ids, self.fw.text_pad_token_id
                    )
                elif last_control == int(self.fw.ss_token_id):
                    yield {
                        "type": "state_change",
                        "from": "B",
                        "to": "S",
                        "pos": cn_e,
                        "chunk": cn_e,
                        "reason": "control_ss",
                        "sl_prob": self._safe_prob(speaking_control_core, "sl"),
                        "ss_prob": self._safe_prob(speaking_control_core, "ss"),
                    }
                    self.listening_state = "s"
                    self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                        self.end_speak_token_factor
                    )
                    self._set_last_ss_pos(self.generated_input_ids.shape[1])
                    self.generated_stoken_ids = self.fw._set_last_token_safe(
                        self.generated_stoken_ids, self.fw.stoken_pad_token_id
                    )
                    self.generated_input_ids = self.fw._set_last_token_safe(
                        self.generated_input_ids, self.fw.text_pad_token_id
                    )
                elif last_control == int(self.fw.keep_bc_token_id) or cn_e == chunk_end:
                    pass
                else:
                    assert int(self.generated_stoken_ids[0, -1].item()) == int(self.fw.tts_end_id)
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > self.generated_input_ids.shape[1] - self.system_input_length:
                        padding_len = target_length - self.generated_input_ids.shape[1] + self.system_input_length
                        self._pad_to_target_after_eos(padding_len, self.fw.end_bc_token_id)
                        self.listening_state = "l"
                        self._set_last_ss_pos(None)
                    else:
                        padding_len = self.fw.control_token_chunk_size
                        self._extend_mapping_for_padding_len(padding_len)
                        pad_input = self.fw.text_pad_token_id * torch.ones(
                            (self.generated_input_ids.shape[0], padding_len),
                            dtype=self.generated_input_ids.dtype,
                            device=self.generated_input_ids.device,
                        )
                        pad_stoken = self.fw.stoken_pad_token_id * torch.ones(
                            (self.generated_stoken_ids.shape[0], padding_len),
                            dtype=self.generated_stoken_ids.dtype,
                            device=self.generated_stoken_ids.device,
                        )
                        pad_control = self.fw.sleep_token_id * torch.ones(
                            (self.generated_control_ids.shape[0], padding_len),
                            dtype=self.generated_control_ids.dtype,
                            device=self.generated_control_ids.device,
                        )
                        if pad_control.shape[1] > 2:
                            pad_control[:, -2] = self.fw.detect_token_id
                        self.generated_input_ids = torch.cat((self.generated_input_ids, pad_input[:, :-1]), dim=1)
                        self.generated_stoken_ids = torch.cat((self.generated_stoken_ids, pad_stoken[:, :-1]), dim=1)
                        self.generated_control_ids = torch.cat((self.generated_control_ids, pad_control[:, :-1]), dim=1)
                        recheck_proc = LogitsProcessorList([
                            BackChannelLogitsProcessor(
                                sleep_token_id=self.fw.sleep_token_id,
                                detect_token_id=self.fw.detect_token_id,
                                end_bc_token_id=self.fw.end_bc_token_id,
                                ss_token_id=self.fw.ss_token_id,
                                keep_bc_token_id=None,
                                vocab_size=self.fw.model.config.text_config.vocab_size,
                                prefix_input_len=self.prefix_input_ids.shape[1],
                                control_token_chunk_size=self.fw.control_token_chunk_size,
                                start_speak_token_factor=self.start_speak_token_factor,
                            )
                        ])
                        recheck_core = recheck_proc[0] if len(recheck_proc) > 0 else None
                        last_result = None
                        for token_result in self._stream_model(
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
                            control_logits_processor=recheck_proc,
                        ):
                            last_result = token_result
                        if last_result is not None:
                            self._update_caches_from_result(last_result, update_sequences=True)
                        pred = int(self.generated_control_ids[0, -1].item())
                        if pred == int(self.fw.end_bc_token_id):
                            yield {
                                "type": "state_change",
                                "from": "B",
                                "to": "L",
                                "pos": cn_e,
                                "chunk": cn_e,
                                "reason": "bc_recheck_end_bc",
                                "ss_prob": self._safe_prob(recheck_core, "ss"),
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
                                "ss_prob": self._safe_prob(recheck_core, "ss"),
                            }
                            self.listening_state = "s"
                            self.speaking_text_processor, self.speaking_stoken_processor = self.fw.init_speaking_processor(
                                self.end_speak_token_factor
                            )
                            self._set_last_ss_pos(self.generated_input_ids.shape[1])

            else:
                self._extend_speaking_stoken_mapping(target_new_length)
                speaking_control_processor = LogitsProcessorList([
                    SpeakingControlLogitsProcessor(
                        sleep_token_id=self.fw.sleep_token_id,
                        detect_token_id=self.fw.detect_token_id,
                        sl_token_id=self.fw.sl_token_id,
                        ks_token_id=self.fw.ks_token_id,
                        vocab_size=self.fw.model.config.text_config.vocab_size,
                        prefix_input_len=self.prefix_input_ids.shape[1],
                        control_token_chunk_size=self.fw.control_token_chunk_size,
                    )
                ])
                speaking_control_core = speaking_control_processor[0] if len(speaking_control_processor) > 0 else None
                last_result = None
                t_model_start = time.perf_counter()
                model_start_epoch_ms = int(time.time() * 1000)
                for token_result in self._stream_model(
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
                    do_sample=True,
                    temperature=0.7,
                    top_p=1,
                    eos_token_id=None,
                    stoken_eos_token_id=self.fw.tts_end_id,
                ):
                    yield {
                        "type": "speaking_token",
                        "text_token": token_result["text_token"],
                        "stoken": token_result["stoken_token"],
                        "control": token_result["control_token"],
                        "step": token_result["step"],
                        "sl_prob": self._safe_prob(speaking_control_core, "sl"),
                        "ss_prob": self._safe_prob(speaking_control_core, "ss"),
                        "ks_prob": self._safe_prob(speaking_control_core, "ks"),
                    }
                    last_result = token_result
                self._record_timeline_span(
                    "transformer",
                    t_model_start,
                    model_start_epoch_ms,
                    state="speaking",
                    chunk_idx=int(self.chunk_idx),
                    target_new_tokens=int(target_new_length),
                )
                for span_event in self._drain_timeline_span_events():
                    yield span_event
                if last_result is None:
                    last_result = self._empty_result()
                self._update_caches_from_result(last_result, update_sequences=True)

                if int(self.generated_stoken_ids[0, -1].item()) == int(self.fw.tts_end_id):
                    yield {"type": "speaking_done", "reason": "eos", "chunk": cn_e}
                    if target_length > self.generated_input_ids.shape[1] - self.system_input_length:
                        padding_len = target_length - self.generated_input_ids.shape[1] + self.system_input_length
                    else:
                        padding_len = self.fw.control_token_chunk_size
                        self._extend_mapping_for_padding_len(padding_len)
                    self._pad_to_target_after_eos(padding_len, self.fw.sl_token_id)
                    self.listening_state = "l"
                    self._set_last_ss_pos(None)
                else:
                    pred_control_token = int(self.generated_control_ids[0, -1].item())
                    if pred_control_token == int(self.fw.sl_token_id):
                        yield {
                            "type": "state_change",
                            "from": "S",
                            "to": "L",
                            "pos": cn_e,
                            "chunk": cn_e,
                            "reason": "control_sl",
                            "early_exit": False,
                            "interrupt": False,
                            "sl_prob": self._safe_prob(speaking_control_core, "sl"),
                            "ss_prob": self._safe_prob(speaking_control_core, "ss"),
                            "ks_prob": self._safe_prob(speaking_control_core, "ks"),
                        }
                        self.listening_state = "l"
                        self._set_last_ss_pos(None)
                        self.generated_input_ids, self.generated_stoken_ids = self.fw._apply_s2l_tail_fallback(
                            self.generated_input_ids,
                            self.generated_stoken_ids,
                        )
                    else:
                        assert (
                            pred_control_token == int(self.fw.ks_token_id)
                            or cn_e == chunk_end
                        ), f"pred_control_token:{pred_control_token}"

            yield {"type": "chunk_end", "chunk_idx": self.chunk_idx}
            self.chunk_idx += 1
            self.next_chunk_pos = cn_e + self.fw.control_token_chunk_size

        if emit_generation_complete:
            yield self.build_generation_complete_event()


class HFRealtimeV9GenerationFramework(_BaseV9Framework):
    """
    HF-only realtime adapter for V9 merge models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = _env_bool("LYCHEEFD_HF_USE_CACHE", True)
        self.model = _HFV9MultiHeadStreamAdapter(self.model)

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
        out = seq.clone()
        out[0, -1] = int(token_id)
        return out

    def _apply_s2l_tail_fallback(
        self,
        generated_input_ids: torch.Tensor,
        generated_stoken_ids: torch.Tensor,
    ):
        if bool(getattr(self, "is_old_sl_strategy", True)):
            text_tail_token = int(self.text_pad_token_id)
            stoken_tail_token = int(self.stoken_pad_token_id)
        else:
            last_text_token = int(generated_input_ids[0, -1].item())
            last_stoken_token = int(generated_stoken_ids[0, -1].item())
            text_tail_token = (
                int(self.eos_token_id)
                if last_text_token not in (int(self.tts_pad_id), int(self.text_pad_token_id))
                else int(self.text_pad_token_id)
            )
            stoken_tail_token = (
                int(self.tts_end_id)
                if last_stoken_token != int(self.stoken_pad_token_id)
                else int(self.stoken_pad_token_id)
            )
        return (
            self._set_last_token_safe(generated_input_ids, text_tail_token),
            self._set_last_token_safe(generated_stoken_ids, stoken_tail_token),
        )

    def create_incremental_stream_session(
        self,
        prefix=None,
        initial_listening_state="l",
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        return HFRealtimeV9IncrementalChunkStreamSession(
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
        initial_listening_state="l",
        start_speak_token_factor=1.2,
        start_listen_token_factor=1.0,
        bc_speak_token_factor=1,
        end_speak_token_factor=1,
        audio_incremental_mode=False,
    ):
        return self.create_incremental_stream_session(
            prefix=prefix,
            initial_listening_state=initial_listening_state,
            start_speak_token_factor=start_speak_token_factor,
            start_listen_token_factor=start_listen_token_factor,
            bc_speak_token_factor=bc_speak_token_factor,
            end_speak_token_factor=end_speak_token_factor,
            audio_incremental_mode=audio_incremental_mode,
        )
