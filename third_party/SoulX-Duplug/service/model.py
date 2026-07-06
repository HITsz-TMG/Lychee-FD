import os, sys
import torch
import random
import numpy as np
import math
import time
import re
from omegaconf import OmegaConf
import pytorch_lightning as pl

from utils.MyTn.textnorm import zh_norm, zh_remove_punc
from utils.text_utils import split_cn_en, check_en, get_lcs_substrings
from utils.backchannel_utils import check_backchannel, remove_leading_backchannel

from config.config import RunConfig
from model.model import State_Prediction_Model
from transformers import WhisperFeatureExtractor


class TurnModel:
    def __init__(
        self,
        config_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        ),
    ):
        self._init_load_model(config_path)

        self._init_hyperparameters()

        self._init_prompt_embeds()

        self.reset()

    def _init_load_model(self, config_path):
        cfg = OmegaConf.load(config_path)
        default_config = RunConfig()
        config = OmegaConf.merge(default_config, cfg)

        pl.seed_everything(config.infer_config.seed)
        torch.manual_seed(config.infer_config.seed)
        np.random.seed(config.infer_config.seed)
        random.seed(config.infer_config.seed)

        model = State_Prediction_Model(config)

        model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.model_config.glm_tokenizer_path
        )

        model.eval().to(config.infer_config.device)
        self.model = model
        self.config = config
        self.device = config.infer_config.device

        if hasattr(self.model.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.embed_tokens
        elif hasattr(self.model.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.model.llm.model.model.model.embed_tokens

        # if self.config.model_config.enable_cascade_asr:
        from model.asr import ParaformerASR, SensevoiceASR

        if config.infer_config.asr.model_name == "paraformer":
            self.cascade_asr = ParaformerASR()
        else:
            self.cascade_asr = SensevoiceASR(
                language=config.infer_config.asr.get("language", "auto")
            )

        print("[DuplexServer] Model loaded successfully.")

    def _init_hyperparameters(self):
        self.sampling_rate = self.config.infer_config.input.sample_rate
        self.device = self.config.infer_config.device
        self.chunk_token_len_small = (
            self.config.infer_config.input.chunk_token_len_small
        )
        self.developer_mode = self.config.infer_config.developer_mode

    def _init_prompt_embeds(self):
        self.input_text_tokens = torch.tensor(
            self.model.tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
        ).to(self.device)
        self.audio_pad_token = torch.tensor(
            self.model.tokenizer.encode("<|padding|>")
        ).to(self.device)
        self.audio_bos_token = torch.tensor(
            self.model.tokenizer.encode("<|begin_of_sentence|>")
        ).to(self.device)
        self.audio_eos_token = torch.tensor(
            self.model.tokenizer.encode("<|end_of_sentence|>")
        ).to(self.device)

        self.action_speak_token = torch.tensor(
            self.model.tokenizer.encode("<|user_complete|>")
        ).to(self.device)
        self.action_wait_token = torch.tensor(
            self.model.tokenizer.encode("<|user_incomplete|>")
        ).to(self.device)
        self.non_idle_token = torch.tensor(
            self.model.tokenizer.encode("<|user_nonidle|>")
        ).to(self.device)

        self.text_embeds = self.embed_tokens_func(self.input_text_tokens)
        self.audio_pad_embeds = self.embed_tokens_func(self.audio_pad_token)
        self.audio_bos_embeds = self.embed_tokens_func(self.audio_bos_token)
        self.audio_eos_embeds = self.embed_tokens_func(self.audio_eos_token)
        self.action_speak_embeds = self.embed_tokens_func(self.action_speak_token)
        self.action_wait_embeds = self.embed_tokens_func(self.action_wait_token)
        self.non_idle_embeds = self.embed_tokens_func(self.non_idle_token)

    def reset(self):
        self.buffer = (
            np.random.randn(
                self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.audio_ahead_size
            )
            * 0.0001
        )
        self.buffer_for_asr = np.random.randn(int(1.6 * self.sampling_rate)) * 0.00001
        self.cascade_buffer = np.random.randn(int(3.2 * self.sampling_rate)) * 0.00001
        self.speech_detected = False
        self.past_state = None
        self.history_chunks = []
        self.wait_idle_cnt = 0
        self.monitoring_wait_silence = False

    def clear_turn(self):
        self.buffer_for_asr = np.random.randn(int(1.6 * self.sampling_rate)) * 0.00001
        self.speech_detected = False
        self.wait_idle_cnt = 0
        self.monitoring_wait_silence = False

    def _log(self, *args, **kwargs):
        # only for debugging, too much logging will slow down the inference
        if self.developer_mode:
            print(*args, **kwargs)

    def get_rms(self, audio_chunk):
        if audio_chunk.dtype == np.int16:
            samples = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype == np.uint8:
            samples = (audio_chunk.astype(np.float32) - 128.0) / 128.0
        elif audio_chunk.dtype == np.float32:
            samples = np.clip(audio_chunk, -1.0, 1.0)
        else:
            print(audio_chunk.dtype)
            raise ValueError("Unsupported PCM format")
        return np.sqrt(np.mean(samples**2))

    def rms_db(self, y):
        """Calculate RMS dBFS of the audio"""
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-9:  # Avoid log(0)
            return -np.inf
        return 20 * np.log10(rms)

    def snapshot_runtime(self):
        return {
            "buffer": self.buffer,
            "buffer_for_asr": self.buffer_for_asr,
            "cascade_buffer": self.cascade_buffer,
            "speech_detected": self.speech_detected,
            "past_state": self.past_state,
            "history_chunks": self.history_chunks,
            "wait_idle_cnt": self.wait_idle_cnt,
            "monitoring_wait_silence": self.monitoring_wait_silence,
        }

    def restore_runtime(self, state):
        if not state:
            self.reset()
            return
        self.buffer = state.get("buffer", self.buffer)
        self.buffer_for_asr = state.get("buffer_for_asr", self.buffer_for_asr)
        self.cascade_buffer = state.get("cascade_buffer", self.cascade_buffer)
        self.speech_detected = state.get("speech_detected", False)
        self.past_state = state.get("past_state")
        self.history_chunks = state.get("history_chunks", [])
        self.wait_idle_cnt = state.get("wait_idle_cnt", 0)
        self.monitoring_wait_silence = state.get("monitoring_wait_silence", False)

    def get_chunk(self):
        if (
            len(self.buffer)
            >= self.config.infer_config.input.chunk_size
            + self.config.infer_config.input.audio_back_size
            + self.config.infer_config.input.audio_ahead_size
        ):
            audio_back = self.buffer[
                : self.config.infer_config.input.audio_back_size
            ].astype(np.float32)

            process_chunk = self.buffer[
                self.config.infer_config.input.audio_back_size : self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size
            ].astype(np.float32)

            audio_ahead = self.buffer[
                self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size : self.config.infer_config.input.audio_back_size
                + self.config.infer_config.input.chunk_size
                + self.config.infer_config.input.audio_ahead_size
            ].astype(np.float32)

            self.buffer = self.buffer[self.config.infer_config.input.chunk_size :]

            # print(process_chunk.dtype)
            return True, process_chunk, audio_back, audio_ahead

        return False, None, None, None

    def process(self, audio_chunk):
        """Process audio chunk, return predict state."""
        assert audio_chunk.dtype == np.float32
        # print(audio_chunk.shape)
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        delta_text = ""
        asr_buffer = ""
        predicted_state = {
            "state": "blank",
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

        start_prediction, process_chunk, audio_back, audio_ahead = self.get_chunk()

        if start_prediction:
            t_start = time.time()
            predicted_state = self.state_predict(process_chunk, audio_back, audio_ahead)
            self._log(f"[Timing] Total chunk: {time.time() - t_start:.4f}s\n\n")

        return predicted_state

    def state_predict(self, process_chunk, audio_back, audio_ahead):
        state, delta_text, asr_buffer = self.infer(
            process_chunk, audio_back, audio_ahead
        )

        if (
            self.get_rms(process_chunk) < self.config.infer_config.far_field_threshold
            and not self.speech_detected
            and state == "<|user_nonidle|>"
        ):
            self._log(f"Far-field speech detected: {self.get_rms(process_chunk)}")
            # process_chunk = np.zeros_like(process_chunk)
            self.reset()
            return {
                "state": "idle",
                "asr_segment": "",
                "asr_buffer": "",
            }

        self.past_state["history_len"] += 1

        self.history_chunks.append((process_chunk, state))
        if len(self.history_chunks) > 5:
            self.history_chunks.pop(0)

        if state == "<|user_idle|>":
            self._log("Silence detected")

            if self.monitoring_wait_silence:
                self.wait_idle_cnt += 1
                if self.wait_idle_cnt >= self.config.infer_config["max_wait_num"]:
                    self._log("Continuous silence after wait, triggering reply")
                    if self.speech_detected:
                        segment = self.cascade_asr.recognize(
                            self.buffer_for_asr, self.sampling_rate
                        )
                        # self.clear_turn()
                        self.reset()
                        return {
                            "state": "speak",
                            "text": segment,
                            "asr_segment": delta_text,
                            "asr_buffer": asr_buffer,
                        }

            if (
                self.past_state["history_len"] > 200
                and not self.monitoring_wait_silence
                and not self.speech_detected
            ):
                self.reset()

        elif state == "<|user_nonidle|>":
            self._log("Speech detected", self.get_rms(process_chunk.astype(np.float32)))
            self.speech_detected = True

            if self.monitoring_wait_silence:
                self.monitoring_wait_silence = False
                self.wait_idle_cnt = 0

            # semantic vad, yield nonidle after the first word or character finished
            # in case it's cut off in previous chunks
            to_concat = []
            if len(self.history_chunks) >= 2:
                prev_chunk, prev_state = self.history_chunks[-2]
                if prev_state in ["<|user_idle|>", "<|user_backchannel|>"]:
                    candidates = []
                    for i in range(len(self.history_chunks) - 2, -1, -1):
                        c, s = self.history_chunks[i]
                        if s in ["<|user_idle|>", "<|user_backchannel|>"]:
                            candidates.append(c)
                            if len(candidates) >= 5:
                                break
                        else:
                            break
                    to_concat = candidates[::-1]

            if to_concat:
                self.buffer_for_asr = np.concatenate(
                    [self.buffer_for_asr, *to_concat, process_chunk]
                )
            else:
                self.buffer_for_asr = np.concatenate(
                    [self.buffer_for_asr, process_chunk]
                )
            return {
                "state": "nonidle",
                "asr_segment": delta_text,
                "asr_buffer": asr_buffer,
            }

        elif state == "<|user_backchannel|>":
            # self._log("Backchannel detected")
            if not self.speech_detected:
                self.reset()
            if self.monitoring_wait_silence:
                self.wait_idle_cnt = 1

        elif state == "<|user_complete|>":
            self._log("User finished speaking, should reply now")
            if self.speech_detected:
                self.buffer_for_asr = np.concatenate(
                    [self.buffer_for_asr, process_chunk]
                )
                segment = self.cascade_asr.recognize(
                    self.buffer_for_asr, self.sampling_rate
                )
                # self.clear_turn()
                self.reset()
                return {
                    "state": "speak",
                    "text": segment,
                    "asr_segment": delta_text,
                    "asr_buffer": asr_buffer,
                }
            # self.clear_turn()
            self.reset()

        elif state == "<|user_incomplete|>":
            self._log("User hasn't finished speaking, wait")
            # self.reset()
            self.monitoring_wait_silence = True
            self.wait_idle_cnt = 0
            self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])

        else:
            self._log("Unknown state")
            self.reset()

        return {
            "state": "idle",
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

    @torch.no_grad()
    def infer(self, audio_chunk, audio_back, audio_ahead):
        self.cascade_buffer = np.concatenate([self.cascade_buffer, audio_chunk])

        # init
        if self.past_state is None:
            self.past_state = {
                "input_embeds": self.text_embeds,
                "past_key_values": None,
                "delta_text": [],
                "cascade_text": "",
                "state": "",
                "history_len": 0,
                "mistake_len": 0,
                "checkpoint": None,
            }

        audio_tokens = self._audio_to_tokens(audio_back, audio_chunk, audio_ahead)

        audio_embeds = self._tokens_to_embeds(audio_tokens)

        self.past_state["input_embeds"] = torch.cat(
            (self.past_state["input_embeds"], audio_embeds), dim=0
        ).unsqueeze(0)

        delta_text = self._asr(audio_embeds)

        state = self._state_predict(delta_text)

        del audio_embeds
        torch.cuda.empty_cache()
        return state, delta_text, self.past_state["cascade_text"]

    def _audio_to_tokens(
        self, audio_back: np.ndarray, audio_chunk: np.ndarray, audio_ahead: np.ndarray
    ):
        """
        audio waveform → discrete speech tokens
        """
        audio_segment = np.concatenate([audio_back, audio_chunk, audio_ahead], axis=0)
        # print(audio_segment.shape)

        start_index = len(audio_back) // self.model.token_samples
        end_index = min(
            start_index + 2,
            math.ceil(audio_segment.shape[0] / self.model.token_samples),
        )

        valid_range = (start_index, end_index)

        pooling_kernel_size = self.model.glm_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.model.glm_tokenizer.conv1.stride[0]
            * self.model.glm_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.model.feature_extractor.hop_length
        )

        features = self.model.feature_extractor(
            [audio_segment],
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
            device=self.device,
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(self.device)

        outputs = self.model.glm_tokenizer(**features)
        speech_tokens = outputs.quantized_token_ids

        attention_mask = features.attention_mask[
            :,
            :: self.model.glm_tokenizer.conv1.stride[0]
            * self.model.glm_tokenizer.conv2.stride[0],
        ]
        attention_mask = attention_mask[
            :, :: self.model.glm_tokenizer.config.pooling_kernel_size
        ]

        assert attention_mask.shape == speech_tokens.shape
        assert len(speech_tokens) == 1

        speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
        audio_tokens = speech_token[valid_range[0] : valid_range[1]]

        return audio_tokens

    def _tokens_to_embeds(self, tokens):
        tokens = torch.tensor(tokens, device=self.device)
        embeds = self.model.glm_tokenizer.codebook(tokens)
        embeds = self.model.audio_projector(embeds)

        if embeds.shape[0] != self.chunk_token_len_small:
            embeds = torch.cat(
                (
                    embeds,
                    self.audio_pad_embeds.expand(
                        self.chunk_token_len_small - embeds.shape[0], -1
                    ),
                ),
                dim=0,
            )

        return embeds

    def _asr(self, audio_embeds):
        # 1. Run model to see if it predicts speech (1 step)
        t_llm_check = time.time()
        with torch.no_grad():
            outputs = self.model.llm(
                inputs_embeds=self.past_state["input_embeds"],
                past_key_values=self.past_state["past_key_values"],
                use_cache=True,
            )
            logits = outputs.logits[0]
            current_kv = outputs.past_key_values
            pred = torch.argmax(logits, -1)[-1]
            self._log(f"[Timing] LLM check: {time.time() - t_llm_check:.4f}s")

            delta_text = ""
            need_correction = False
            corrected_prev_delta = ""

            if pred != self.model.asr_eos_token_id:
                # Speech detected -> Cascade ASR
                t_asr = time.time()
                full_text = remove_leading_backchannel(
                    self.cascade_asr.recognize(self.cascade_buffer, self.sampling_rate)
                )
                self._log(f"[Timing] Cascade ASR: {time.time() - t_asr:.4f}s")

                # Process Text Delta
                history_text = self.past_state.get("cascade_text", "")
                norm_full_text = split_cn_en(zh_norm(zh_remove_punc(full_text.strip())))
                norm_history_text = split_cn_en(
                    zh_norm(zh_remove_punc(history_text.strip()))
                )

                if len(norm_full_text) >= 5 and len(norm_history_text) >= 5:
                    backup_norm_full_text = norm_full_text.copy()
                    backup_norm_history_text = norm_history_text.copy()
                    norm_full_text, norm_history_text = get_lcs_substrings(
                        norm_full_text, norm_history_text
                    )

                prev_delta = (
                    self.past_state["delta_text"][-1]
                    if self.past_state["delta_text"]
                    else ""
                )
                prev_delta_split = split_cn_en(prev_delta)
                len_prev = len(prev_delta_split)

                if len_prev > len(norm_history_text):
                    norm_full_text = backup_norm_full_text
                    norm_history_text = backup_norm_history_text

                history_base = (
                    norm_history_text[:-len_prev] if len_prev > 0 else norm_history_text
                )

                if len(norm_full_text) > len(norm_history_text):
                    current_segment_in_full = norm_full_text[
                        len(history_base) : len(history_base) + len_prev
                    ]
                    if current_segment_in_full == prev_delta_split:
                        # 3.1 Consistent
                        delta_text = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in norm_full_text[len(norm_history_text) :]
                            ]
                        ).strip()
                    else:
                        # 3.2 Inconsistent
                        need_correction = True
                        corrected_prev_delta = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in current_segment_in_full
                            ]
                        ).strip()
                        delta_text = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in norm_full_text[len(history_base) + len_prev :]
                            ]
                        ).strip()

                elif len(norm_full_text) == len(norm_history_text):
                    current_segment_in_full = norm_full_text[len(history_base) :]
                    if current_segment_in_full == prev_delta_split:
                        # 4.1 Consistent
                        delta_text = ""
                    else:
                        # 4.2 Inconsistent
                        need_correction = True
                        corrected_prev_delta = "".join(
                            [
                                (s + " ") if check_en(s) else s
                                for s in current_segment_in_full
                            ]
                        ).strip()
                        delta_text = ""

                else:  # len(norm_full) < len(norm_history)
                    # 5
                    need_correction = True
                    remainder = norm_full_text[len(history_base) :]
                    corrected_prev_delta = ""
                    delta_text = "".join(
                        [(s + " ") if check_en(s) else s for s in remainder]
                    ).strip()

                self.past_state["cascade_text"] = "".join(
                    [(s + " ") if check_en(s) else s for s in norm_full_text]
                ).strip()

                self._log(
                    f"[History]: {''.join([(s + ' ') if check_en(s) else s for s in norm_history_text]).strip()}"
                )
                self._log(f"[ASR] Full: {self.past_state['cascade_text']}")
                self._log(f"[Need Correction]: {need_correction}")
                self._log(f"[Prev Delta]: {corrected_prev_delta}")
                self._log(f"[Delta]: {delta_text}")

            if need_correction and self.past_state["checkpoint"] is not None:
                self._log("--- Correction Triggered ---")
                self.past_state["past_key_values"] = self.past_state["checkpoint"]

                embeds_list = []
                # 1. Corrected Prev Text
                if corrected_prev_delta:
                    ids = self.model.tokenizer.encode(
                        corrected_prev_delta, add_special_tokens=False
                    )
                    if ids:
                        t_ids = torch.tensor(ids, dtype=torch.long).to(self.device)
                        embeds_list.append(self.embed_tokens_func(t_ids))

                # 2. EOS
                embeds_list.append(self.audio_eos_embeds)
                # 3. NonIdle
                embeds_list.append(self.non_idle_embeds)
                # 4. Audio (Current Chunk)
                embeds_list.append(audio_embeds)

                correction_input = torch.cat(embeds_list, dim=0).unsqueeze(0)

                outputs = self.model.llm(
                    inputs_embeds=correction_input,
                    past_key_values=self.past_state["past_key_values"],
                    use_cache=True,
                )
                self.past_state["past_key_values"] = outputs.past_key_values

                if self.past_state["delta_text"]:
                    self.past_state["delta_text"][-1] = corrected_prev_delta
            else:
                self.past_state["past_key_values"] = current_kv

            # Save Checkpoint for NEXT chunk (KV after Audio, before Text)
            self.past_state["checkpoint"] = self.past_state["past_key_values"]

            # cache 3.2s
            self.past_state["delta_text"].append(delta_text)
            max_len = int(3.2 * self.sampling_rate)
            if len(self.cascade_buffer) > max_len:
                self.cascade_buffer = self.cascade_buffer[-max_len:]

                delta_list = self.past_state["delta_text"][-20:]
                total_tokens = 0
                for t in delta_list:
                    total_tokens += len(split_cn_en(t))

                cascade_tokens = split_cn_en(self.past_state["cascade_text"])
                if total_tokens > 0:
                    keep_tokens = cascade_tokens[-total_tokens:]
                else:
                    keep_tokens = []

                update_text = "".join(
                    [(s + " ") if check_en(s) else s for s in keep_tokens]
                ).strip()
                self.past_state["cascade_text"] = update_text

            self._log(
                f"[Concat]: {''.join([(s + ' ') if check_en(s) else s for s in self.past_state['delta_text']]).strip()}"
            )

            # Prepare input for next step (State Prediction)
            # If delta_text exists, we concat text embeds + eos embeds to save one forward pass
            input_embeds_next = self.audio_eos_embeds.unsqueeze(0)

            if delta_text:
                ids = self.model.tokenizer.encode(delta_text, add_special_tokens=False)
                input_ids = torch.tensor(ids, dtype=torch.long).to(self.device)

                if len(input_ids) > 0:
                    embeds = self.embed_tokens_func(input_ids)
                    embeds = embeds.unsqueeze(0)
                    # Concat text embeds and EOS
                    input_embeds_next = torch.cat((embeds, input_embeds_next), dim=1)

            self.past_state["input_embeds"] = input_embeds_next

            return delta_text

    def _state_predict(self, delta_text):
        # 4. State Prediction (Feed EOS or Text+EOS)
        t_state = time.time()
        with torch.no_grad():
            outputs = self.model.llm(
                inputs_embeds=self.past_state["input_embeds"],
                past_key_values=self.past_state["past_key_values"],
                use_cache=True,
            )
            logits = outputs.logits[0]
            pred = torch.argmax(logits, -1)[-1]
            state = self.model.tokenizer.decode(pred)

        if state == "<|user_nonidle|>" and not delta_text:
            self.past_state["mistake_len"] += 1
        else:
            self.past_state["mistake_len"] = 0

        if (
            self.past_state["state"] == "<|user_nonidle|>" and state == "<|user_idle|>"
        ) or self.past_state["mistake_len"] >= self.config.infer_config.max_mistake_num:
            if (
                logits[-1, self.config.model_config.user_complete_token_id]
                > logits[-1, self.config.model_config.user_incomplete_token_id]
            ):
                state = "<|user_complete|>"
                self.past_state["input_embeds"] = self.action_speak_embeds
            else:
                state = "<|user_incomplete|>"
                self.past_state["input_embeds"] = self.action_wait_embeds
        else:
            # Update input_embeds for next turn (state token embedding)
            self.past_state["input_embeds"] = self.embed_tokens_func(pred.unsqueeze(0))

        self.past_state["past_key_values"] = outputs.past_key_values
        self.past_state["state"] = state
        self._log(f"[Timing] State pred: {time.time() - t_state:.4f}s")
        return state


def load_turn_model(config_path: str = "config/config.yaml"):
    """
    Called upon process startup
    """
    turn_model = TurnModel(config_path=config_path)

    print(f"[TurnModel] loaded on {turn_model.device}")
    return turn_model


if __name__ == "__main__":
    model = load_turn_model(config_path="config/config.yaml")

    print("model loaded OK")
