# Copyright (c) 2025 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
from transformers import AutoConfig

from flashcosyvoice.config import CosyVoice2LLMConfig
from flashcosyvoice.modules.qwen2_components.layers import (
    ParallelLMHead, Qwen2DecoderLayer, RMSNorm, VocabParallelEmbedding)


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: CosyVoice2LLMConfig,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: CosyVoice2LLMConfig | AutoConfig
    ):
        super().__init__()
        self.model = Qwen2Model(config)
        if hasattr(config, "speech_vocab_size"):
            self.lm_head = ParallelLMHead(config.speech_vocab_size, config.hidden_size, bias=getattr(config, "lm_head_bias", True))
            self.model_type = "speech_llm"
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)
            self.model_type = "text_llm"
        self.tie_word_embeddings = config.tie_word_embeddings
        if self.tie_word_embeddings:
            if self.model_type == "speech_llm":
                assert config.vocab_size == config.speech_vocab_size, "vocab_size and speech_vocab_size must be the same when tie_word_embeddings is True"
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
