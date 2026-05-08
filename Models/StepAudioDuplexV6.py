from typing import Iterable, Optional, Tuple, Union
import librosa
import torch
import torch.nn.functional as F
from torch import nn
import torchaudio
from torch import Tensor, nn
from transformers import PreTrainedModel, Qwen2Model
from transformers.generation.utils import GenerationMixin
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Optional, Union
from dataclasses import dataclass
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.nn import Parameter, CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, check_torch_load_is_safe, logging
from transformers.utils.hub import cached_file
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
)
try:
    from modeling_step_audio_2 import AudioEncoder, Adaptor
    from configuration_step_audio_2 import StepAudio2TextConfig, StepAudio2EncoderConfig
except:
    from .modeling_step_audio_2 import AudioEncoder, Adaptor
    from .configuration_step_audio_2 import StepAudio2TextConfig, StepAudio2EncoderConfig

class StepAudio2FullDuplexConfig(PretrainedConfig):
    model_type = "step_audio_2_full_duplex"
    architectures = ["StepAudio2ForCausalLM"]

    def __init__(
        self,
        # full duplex control token
        start_speaking_token_id=None,
        keep_listening_token_id=None,
        start_listening_token_id=None,
        keep_speaking_token_id=None,
        detect_token_id=None,
        sleep_token_id=None,
        # fd text token
        text_pad_token_id=None,
        text_sleep_token_id=None,
        # fd stoken token
        stoken_pad_token_id=None,
        stoken_delay_token_id=None,
        # fd audio token
        audio_pad_token_id=None,
        # bc
        start_bc_token_id=None,
        #
        control_token_chunk_size=None,
        stoken_delay_num=0,
        #
        adding_text_hiddenstates=False,
        #
        stoken_token_ids_min=151694,
        stoken_token_ids_max=158352,
        control_token_ids_min=158352,
        control_token_ids_max=158356,
        #
        stoken_layer_config=None,
        control_layer_config=None,
        # loss
        no_text_label=False,
        no_stoken_label=False,
        # original
        audio_encoder_config :Optional[Union[dict, StepAudio2EncoderConfig]] = None,
        text_config: Optional[Union[dict, StepAudio2TextConfig]] = None,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = 2048,
        max_window_layers: Optional[int] = None,
        **kwargs
    ):
        # full duplex
        self.start_speaking_token_id = start_speaking_token_id
        self.keep_listening_token_id = keep_listening_token_id
        self.start_listening_token_id = start_listening_token_id
        self.keep_speaking_token_id = keep_speaking_token_id
        self.detect_token_id = detect_token_id
        self.sleep_token_id = sleep_token_id
        self.text_pad_token_id = text_pad_token_id
        self.text_sleep_token_id = text_sleep_token_id
        self.stoken_pad_token_id = stoken_pad_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.stoken_delay_token_id = stoken_delay_token_id
        self.start_bc_token_id = start_bc_token_id
        self.control_token_chunk_size = control_token_chunk_size
        self.adding_text_hiddenstates = adding_text_hiddenstates
        self.stoken_token_ids_min = stoken_token_ids_min
        self.stoken_token_ids_max = stoken_token_ids_max
        self.control_token_ids_min = control_token_ids_min
        self.control_token_ids_max = control_token_ids_max
        self.no_text_label = no_text_label
        self.no_stoken_label = no_stoken_label
        self.stoken_delay_num = stoken_delay_num
        
        if isinstance(stoken_layer_config, dict):
            stoken_layer_config = StepAudio2TextConfig(**stoken_layer_config).text_config
        self.stoken_layer_config = stoken_layer_config

        if isinstance(control_layer_config, dict):
            control_layer_config = StepAudio2TextConfig(**control_layer_config).text_config
        self.control_layer_config = control_layer_config

        kwargs.setdefault("use_sliding_window", use_sliding_window)
        kwargs.setdefault("sliding_window", sliding_window)
        if max_window_layers is None:
            max_window_layers = kwargs.get("num_hidden_layers", None)
        kwargs.setdefault("max_window_layers", max_window_layers)
        super().__init__(**kwargs)

        if text_config is None:
            text_config = StepAudio2TextConfig().text_config
        elif isinstance(text_config, dict):
            text_config = StepAudio2TextConfig(**text_config).text_config

        self.text_config = text_config

        if audio_encoder_config is None:
            self.audio_encoder_config = StepAudio2EncoderConfig()
        elif isinstance(audio_encoder_config, dict):
            self.audio_encoder_config = StepAudio2EncoderConfig(**audio_encoder_config)

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_stoken: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None
    loss_control: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    stoken_logits: Optional[torch.FloatTensor] = None
    control_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    stoken_past_key_values: Optional[Cache] = None
    control_past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    stoken_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    control_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None



class StepAudio2FullDuplex(PreTrainedModel, GenerationMixin):
    config_class = StepAudio2FullDuplexConfig
    main_input_name = "input_ids"
    # Important: Add this attribute to make HF recognize it as a model with generation capability
    # _keys_to_ignore_on_load_missing = ["lm_head.weight"]
    supports_gradient_checkpointing = True  # 新增，声明支持gradient checkpointing
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: StepAudio2TextConfig):
        super().__init__(config)
        if isinstance(config.torch_dtype, str):
            dtype = getattr(torch, config.torch_dtype)
        else:
            dtype = config.torch_dtype
        self.model = Qwen2Model(config.text_config)
        self.stoken_model = Qwen2Model(config.stoken_layer_config) if config.stoken_layer_config.num_hidden_layers > 0 else None
        self.control_model = Qwen2Model(config.control_layer_config) if config.control_layer_config.num_hidden_layers > 0 else None

        self.stoken_model_index = config.text_config.num_hidden_layers - config.stoken_layer_config.num_hidden_layers
        self.control_model_index = config.text_config.num_hidden_layers - config.control_layer_config.num_hidden_layers

        self.adding_text_hiddenstates = config.adding_text_hiddenstates
        
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
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=dtype
        )
        # self.stoken_head = nn.Linear(
        #     config.text_config.hidden_size, 
        #     config.text_config.vocab_size, 
        #     bias=False,
        #     dtype=dtype
        # )
        
        # self.control_head = nn.Linear(
        #     config.text_config.hidden_size, 
        #     config.text_config.vocab_size, 
        #     bias=False,
        #     dtype=dtype
        # )
        self.post_init()

        self.no_text_label = self.config.no_text_label
        self.no_stoken_label = self.config.no_stoken_label

        self.stream_generation_flag = False
        self.max_input_length = 0

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def text2stoken_mapping(self, input_ids, stoken_ids, stoken_mapping):
        valid_mask = (stoken_mapping != -1)

        gather_indices = stoken_mapping.clone()
        gather_indices[~valid_mask] = 0

        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, input_ids.size(-1))

        input_ids_selected = torch.gather(input_ids, dim=1, index=gather_indices)

        # 5. 将数据加到 B 中
        # 需要将 mask 扩展一维以便广播 (Broadcasting)
        # mask_expanded shape: (batch_size, seq_len, 1)
        mask_expanded = valid_mask.unsqueeze(-1)
        
        # 只有 mask 为 True 的地方才加上 A_selected，否则加 0
        # 使用 += 进行原位操作 (In-place operation) 以节省显存
        stoken_ids = stoken_ids + input_ids_selected * mask_expanded
        
        return stoken_ids

    def clamp_loss_fn(self, logits, labels, min_clamp=None, max_clamp=None):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
        shift_labels = shift_labels.view(-1)

        if min_clamp is None:
            min_clamp = 0
        if max_clamp is None:
            max_clamp = self.config.text_config.vocab_size
        shift_logits = shift_logits[:, min_clamp: max_clamp]
        shift_label_mask = shift_labels == -100
        shift_labels = shift_labels - min_clamp
        shift_labels[shift_label_mask] = -100
        
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def forward(
        self,
        input_ids=None,
        stoken_ids=None,
        stoken_mapping=None,
        control_input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids: Optional[torch.LongTensor] = None,
        prefix_input_ids: Optional[torch.LongTensor] = None,
        wavs=None,
        wav_lens=None,
        attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        stoken_past_key_values: Optional[list[torch.FloatTensor]] = None,
        control_past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        stoken_label_ids: Optional[torch.LongTensor] = None,
        control_label_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if self.no_text_label:
                inputs_embeds[:, prefix_input_ids.shape[1]:] = 0
            if not self.stream_generation_flag:
                inputs_embeds_prefix = inputs_embeds[:, :prefix_input_ids.shape[1]]
                inputs_embeds_suffix = inputs_embeds[:, prefix_input_ids.shape[1]:]

                if not self.no_stoken_label:
                    stoken_embeds = self.get_input_embeddings()(stoken_ids)
                    inputs_embeds_suffix = inputs_embeds_suffix + stoken_embeds

                control_embeds = self.get_input_embeddings()(control_input_ids)
                inputs_embeds_suffix = inputs_embeds_suffix + control_embeds

                inputs_embeds = torch.cat([inputs_embeds_prefix, inputs_embeds_suffix], dim=1)
            else:
                if past_key_values is None or past_key_values.get_seq_length() == 0:
                    control_embeds = self.get_input_embeddings()(control_input_ids)
                    control_embeds_prefix = torch.zeros((control_embeds.shape[0], prefix_input_ids.shape[1], control_embeds.shape[2]), dtype=control_embeds.dtype, device=control_embeds.device)
                    control_embeds = torch.cat((control_embeds_prefix, control_embeds), dim=1)
                    inputs_embeds = inputs_embeds + control_embeds

                    stoken_embeds = self.get_input_embeddings()(stoken_ids)
                    stoken_embeds_prefix = torch.zeros((stoken_embeds.shape[0], prefix_input_ids.shape[1], stoken_embeds.shape[2]), dtype=stoken_embeds.dtype, device=stoken_embeds.device)
                    stoken_embeds = torch.cat((stoken_embeds_prefix, stoken_embeds), dim=1)
                    inputs_embeds = inputs_embeds + stoken_embeds
                else:
                    inputs_embeds = inputs_embeds + self.get_input_embeddings()(control_input_ids) + self.get_input_embeddings()(stoken_ids)
         
            if wavs is not None:
                assert audio_input_ids is not None
                audio_inputs_embeds = self.get_input_embeddings()(audio_input_ids)
                if self.bf16:
                    wavs = wavs.bfloat16()
                out, feat_lens = self.encoder(wavs, wav_lens)
                out = self.adapter(out)
                feat_lens = (feat_lens - 1) // 2 + 1
                # insert_location = torch.nonzero(audio_input_ids == 151688)
                # insert_location[:,1] += 1
                # for idx in range(len(insert_location)):
                #     i,s = insert_location[idx]
                #     audio_inputs_embeds[i][s : s+feat_lens[idx]] = out[idx][:feat_lens[idx]]
                # inputs_embeds[:, prefix_input_ids.shape[1]:] = inputs_embeds[:, prefix_input_ids.shape[1]:] + audio_inputs_embeds
                
                mask = torch.arange(out.shape[1], device=out.device)[None, :] < feat_lens[:, None]
                audio_features = out[mask]
                audio_mask = (
                    (audio_input_ids == 151690)
                    .unsqueeze(-1)
                    .expand_as(audio_inputs_embeds)
                    .to(audio_inputs_embeds.device)
                )
                audio_features = audio_features.to(audio_inputs_embeds.device, audio_inputs_embeds.dtype)
                audio_inputs_embeds = audio_inputs_embeds.masked_scatter(audio_mask, audio_features)
                if not self.stream_generation_flag:
                    inputs_embeds[:, prefix_input_ids.shape[1]:] = inputs_embeds[:, prefix_input_ids.shape[1]:] + audio_inputs_embeds
                else:
                    audio_feature_prefix = torch.zeros((inputs_embeds.shape[0], prefix_input_ids.shape[1], audio_inputs_embeds.shape[2]), dtype=audio_inputs_embeds.dtype, device=audio_inputs_embeds.device)
                    audio_inputs_embeds = torch.cat((audio_feature_prefix, audio_inputs_embeds), dim=1)
                    inputs_embeds = inputs_embeds + audio_inputs_embeds[:, cache_position]

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
            **kwargs
        )
        
        logits = self.lm_head(outputs[0])

        if self.stoken_model is not None:
            stoken_inputs_embeds = outputs.hidden_states[self.stoken_model_index]
            if self.adding_text_hiddenstates:
                stoken_inputs_embeds = self.text2stoken_mapping(outputs.hidden_states[self.stoken_model_index], stoken_inputs_embeds, stoken_mapping)
            
            stoken_outputs = self.stoken_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=stoken_inputs_embeds,
                use_cache=use_cache,
                past_key_values=stoken_past_key_values,
                cache_position=cache_position,
                **kwargs
            )
        else:
            stoken_outputs = outputs

        stoken_logits = self.lm_head(stoken_outputs[0])

        if self.control_model is not None:
            control_inputs_embeds = outputs.hidden_states[self.control_model_index]

            control_outputs = self.control_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=control_inputs_embeds,
                use_cache=use_cache,
                past_key_values=control_past_key_values,
                cache_position=cache_position,
                **kwargs
            )
        else:
            control_outputs = outputs
        
        control_logits = self.lm_head(control_outputs[0])
        

        loss = loss_text = loss_stoken = loss_control = None
        if labels is not None:

            if not self.no_text_label:
                loss_text = self.clamp_loss_fn(logits, labels)
            else:
                loss_text = torch.tensor(0)

            if not self.no_stoken_label:
                loss_stoken = self.clamp_loss_fn(stoken_logits, stoken_label_ids, self.config.stoken_token_ids_min, self.config.stoken_token_ids_max)
            else:
                loss_stoken = torch.tensor(0)
            
            if self.lm_head.weight.requires_grad:
                loss_control = self.clamp_loss_fn(control_logits, control_label_ids, self.config.control_token_ids_min, self.config.start_bc_token_id + 1)
            else:
                loss_control = torch.tensor(0)
            
            loss = loss_text + loss_control + loss_stoken

            import wandb
            if wandb.run is not None:
                self.max_input_length = max(self.max_input_length, logits.shape[1])
                log_data = {
                    "total_loss": loss.item(),
                    "text_loss": loss_text.item(),
                    "stoken_loss": loss_stoken.item(),
                    "control_tokens_loss": loss_control.item(),
                }
                wandb.log(log_data)
                
                print(
                    f"total_loss: {loss.item():.4f}, "
                    f"text_loss: {loss_text.item():.4f}, "
                    f"control_tokens_loss: {loss_control.item():.4f}, "
                    f"stoken_loss: {loss_stoken.item():.4f}, "
                    f"max token length: {self.max_input_length}"
                )
        return CausalLMOutputWithPast(
            loss=loss,
            loss_stoken=loss_stoken,
            loss_text=loss_text,
            loss_control=loss_control,
            logits=logits,
            stoken_logits=stoken_logits,
            control_logits=control_logits,
            past_key_values=outputs.past_key_values,
            stoken_past_key_values=stoken_outputs.past_key_values,
            control_past_key_values=control_outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            stoken_hidden_states=stoken_outputs.last_hidden_state,
            control_hidden_states=control_outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    # def get_input_embeddings(self):
    #     """Return the model's input embeddings - required for GenerationMixin"""
    #     return self.model.embed_tokens

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Prepare inputs for generation - required for GenerationMixin"""
        # # Keep the wavs and wav_lens from the initial call
        # wavs = kwargs.get("wavs", None)
        # wav_lens = kwargs.get("wav_lens", None)

        # # For generation steps after the first, we don't need to process audio again
        # # because the audio tokens have already been replaced in the input sequence
        # if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
        #     # We're in a generation step, no need to process audio again
        #     return {
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "past_key_values": kwargs.get("past_key_values")
        #     }

        # # First generation step, include audio processing
        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "wavs": wavs,
        #     "wav_lens": wav_lens
        # }
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder the cache for beam search - required for GenerationMixin if using beam search"""
        # If you're not using past_key_values or beam search, this can be a simple pass-through
        # Otherwise implement according to your model's cache structure
        return past_key_values

    def _set_gradient_checkpointing(self, module, value=False):
        # For Qwen2Model
        if hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = value
            if self.stoken_model is not None:
                self.stoken_model.gradient_checkpointing = value
            if self.control_model is not None:
                self.control_model.gradient_checkpointing = value

            # Add the missing _gradient_checkpointing_func method to Qwen2Model
            # This is what Qwen2Model tries to use when gradient_checkpointing=True
            if value and not hasattr(self.model, '_gradient_checkpointing_func'):
                def _gradient_checkpointing_func(module_to_run, *args, **kwargs):
                    # This function wraps torch.utils.checkpoint.checkpoint
                    # and is used by Qwen2Model to perform checkpointing
                    return torch.utils.checkpoint.checkpoint(module_to_run, *args, **kwargs)

                self.model._gradient_checkpointing_func = _gradient_checkpointing_func
                if self.stoken_model is not None:
                    self.stoken_model._gradient_checkpointing_func = _gradient_checkpointing_func
                if self.control_model is not None:
                    self.control_model._gradient_checkpointing_func = _gradient_checkpointing_func

        # For custom encoder and adapter
        if hasattr(self.encoder, 'gradient_checkpointing'):
            self.encoder.gradient_checkpointing = value
        if hasattr(self.adapter, 'gradient_checkpointing'):
            self.adapter.gradient_checkpointing = value

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor, 
        input_ids_seq: torch.Tensor, 
        processors: Optional[LogitsProcessorList],
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        对 Logits 进行处理并采样。
        步骤: LogitsProcessor -> Temperature -> Top-K -> Top-P -> Softmax -> Multinomial
        """
        # A. 应用自定义 LogitsProcessor (如 ListeningLogitsProcessor)
        if processors is not None:
            logits = processors(input_ids_seq, logits)

        # --- 分支：贪婪解码 ---
        if not do_sample or (temperature is not None and temperature == 0):
            return torch.argmax(logits, dim=-1, keepdim=True)

        # B. 应用 Temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # C. 应用 Top-K 截断
        if top_k is not None and top_k > 0:
            # 获取 top_k 的值
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将小于 top_k 最小值的 logits 设为 -inf
            pivot = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < pivot, torch.tensor(float('-inf'), device=logits.device), logits)

        # D. 应用 Top-P (Nucleus Sampling) 截断
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过 top_p 的 token (保留第一个超过的)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 恢复原索引顺序并应用 mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # E. 采样
        probs = F.softmax(logits, dim=-1)
        
        # 如果全是 -inf (极端情况)，回退到均匀分布或 argmax
        if torch.isnan(probs).any() or torch.sum(probs) == 0:
            probs = torch.ones_like(probs) / probs.shape[-1]
            
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    @torch.no_grad()
    def multi_head_generate(
        self,
        input_ids: torch.LongTensor,
        stoken_ids: torch.LongTensor,
        control_input_ids: torch.LongTensor,
        audio_input_ids: torch.LongTensor,
        prefix_input_ids: torch.LongTensor,
        stoken_mapping: Optional[torch.Tensor] = None,
        wavs: Optional[torch.Tensor] = None,
        wav_lens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        # --- 采样参数 ---
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        # --- 处理器 ---
        logits_processor: Optional[LogitsProcessorList] = None, # 用于文本生成的处理器
        stoken_logits_processor: Optional[LogitsProcessorList] = None, # 用于语音生成的处理器
        control_logits_processor: Optional[LogitsProcessorList] = None, # 用于控制生成的处理器
        # --- 其他 ---
        eos_token_id: Optional[int] = None,
        stoken_eos_token_id: Optional[int] = None,
        past_key_values=None,
        stoken_past_key_values=None,
        control_past_key_values=None,
        use_cache=True,
        # 
        logits_dict=None,
        **kwargs
    ):
        """
        自定义的三通道生成函数。
        支持同时生成 Text, Stoken (语音token) 和 Control token。
        """
        assert input_ids.shape[0] == 1, "目前只支持单挑推理"

        # 1. 初始化输入参数
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # 如果没有提供 stoken_ids 或 control_input_ids，则用 padding 填充初始化
        # 注意：这里假设输入的长度是一致的，如果不一致需要你在外部做好 padding
        # if stoken_ids is None:
        #     stoken_ids = torch.full_like(input_ids, self.config.stoken_pad_token_id)
        # if control_input_ids is None:
        #     control_input_ids = torch.full_like(input_ids, self.config.sleep_token_id) # 假设默认是 sleep

        # 2. 初始化 KV Cache (关键：用于加速推理，避免重复计算)
        # 使用 DynamicCache，这是新版 Transformers 推荐的方式
        if past_key_values is None and use_cache:
            stoken_past_key_values = DynamicCache()
            control_past_key_values = DynamicCache()
            past_key_values = DynamicCache()
        elif not use_cache:
            stoken_past_key_values = None
            control_past_key_values = None
            past_key_values = None

        # 3. 准备输出容器
        # 我们会把生成的 token 追加到这里面
        generated_input_ids = input_ids.clone()
        generated_stoken_ids = stoken_ids.clone()
        generated_control_ids = control_input_ids.clone()

        # 4. 标记是否是第一步 (Prefill 阶段)
        is_prefill = True
        
        # 5. 生成循环
        for step in range(max_new_tokens):
            
            # --- A. 准备当前步的输入 ---
            if not use_cache:
                curr_input_ids = generated_input_ids
                curr_stoken_ids = generated_stoken_ids
                curr_control_ids = generated_control_ids
                cur_stoken_mapping = stoken_mapping[:,:generated_input_ids.shape[1]] if stoken_mapping is not None else None
                cache_position = torch.arange(curr_input_ids.shape[1], device=device)
            elif is_prefill:
                # 第一步：输入完整的 Prompt 序列
                curr_input_ids = generated_input_ids[:, past_key_values.get_seq_length():]
                curr_stoken_ids = generated_stoken_ids[:, past_key_values.get_seq_length()-prefix_input_ids.shape[1]:]
                curr_control_ids = generated_control_ids[:, past_key_values.get_seq_length()-prefix_input_ids.shape[1]:]
                cur_stoken_mapping = None
                cache_position = torch.arange(past_key_values.get_seq_length(), past_key_values.get_seq_length() + curr_input_ids.shape[1], device=device)
            else:
                # 后续步骤：只输入上一步生成的最后一个 token (Decode 阶段)
                curr_input_ids = generated_input_ids[:, -1:]
                curr_stoken_ids = generated_stoken_ids[:, -1:]
                curr_control_ids = generated_control_ids[:, -1:]
                cur_stoken_mapping = None
                cache_position = torch.tensor([past_key_values.get_seq_length()], device=device)

            # --- B. 模型前向传播 ---
            outputs = self.forward(
                input_ids=curr_input_ids,
                stoken_ids=curr_stoken_ids,
                control_input_ids=curr_control_ids,
                prefix_input_ids=prefix_input_ids, # 仅用于兼容接口
                audio_input_ids=audio_input_ids,
                stoken_mapping=cur_stoken_mapping,
                wavs=wavs, # 通常只在 Prefill 阶段处理音频输入
                wav_lens=wav_lens,
                past_key_values=past_key_values,
                stoken_past_key_values=stoken_past_key_values,
                control_past_key_values=control_past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            
            # --- C. 获取 Logits ---
            # outputs.logits 是 lm_head 的输出 (Text)
            # 你的 forward 返回的是 CausalLMOutputWithPast，里面只有 logits。
            # 但你的 forward 代码中计算了 control_logits 和 stoken_logits，
            # 只是最后 return 的时候根据 control_token_mode 选了一个。
            # **重要修改建议**：你需要修改一下 forward 的返回值，或者在这里手动重新计算 head。
            # 由于不能改你的 forward，我们利用 hidden_states 手动过一遍 head。

            # --- 获取 Logits ---
            text_logits = outputs.logits[:, -1, :]
            stoken_logits = outputs.stoken_logits[:, -1, :]
            control_logits = outputs.control_logits[:, -1, :]

            if logits_dict is not None and curr_input_ids.shape[1] > 1:
                logits_dict['logits'].append(outputs.logits[:, -curr_input_ids.shape[1]:].clone().cpu())
                logits_dict['stoken_logits'].append(outputs.stoken_logits[:, -curr_input_ids.shape[1]:].clone().cpu())
                logits_dict['control_logits'].append(outputs.control_logits[:, -curr_input_ids.shape[1]:].clone().cpu())

            # --- 采样 (分别对三个通道应用采样逻辑) ---
            
            # 1. 文本通道 (应用主要的 logits_processor)
            next_text = self._sample_from_logits(
                text_logits, 
                generated_input_ids, # 传入完整的历史序列供 processor 参考
                logits_processor,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=False,
            )
            
            # 2. 语音通道 (通常不需要复杂的 processor，或者传入特定的)
            next_stoken = self._sample_from_logits(
                stoken_logits, 
                generated_stoken_ids, 
                stoken_logits_processor,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
            
            # 3. 控制通道 (通常控制 token 需要比较确定性，可以不传 processor 或传特定的)
            next_control = self._sample_from_logits(
                control_logits, 
                generated_control_ids, 
                control_logits_processor,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,        
                do_sample=False,       
            )

            # --- E. 更新序列 ---
            generated_input_ids = torch.cat([generated_input_ids, next_text], dim=1)
            generated_stoken_ids = torch.cat([generated_stoken_ids, next_stoken], dim=1)
            generated_control_ids = torch.cat([generated_control_ids, next_control], dim=1)

            # --- F. 结束条件判断 ---
            if eos_token_id is not None:
                if (next_text == eos_token_id).all():
                    break

            if stoken_eos_token_id is not None:
                if (next_stoken == stoken_eos_token_id).all():
                    break
            
            # 标记 Prefill 结束，进入 Decode 模式
            is_prefill = False

        return {
            "sequences": generated_input_ids, 
            "stoken_ids": generated_stoken_ids,
            "control_ids": generated_control_ids,
            "past_key_values": past_key_values,
            "stoken_past_key_values": stoken_past_key_values,
            "control_past_key_values": control_past_key_values,
        }