import torch, torchaudio
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import re
import pytorch_lightning as pl
import peft
from peft import LoraConfig, get_peft_model

from model.glm_4_voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperFeatureExtractor


class EncoderProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_embed_dim = config.audio_embed_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.audio_embed_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, self.llm_dim)

    def forward(self, x):
        x = x.contiguous()  # (batch, seq_len, dim)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class State_Prediction_Model(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model_config = config.model_config
        self.train_config = config.train_config
        self.asr_eos_token_id = self.model_config.asr_eos_token_id
        self.lm_vocab_size = self.model_config.lm_vocab_size
        self.best_val_acc = 0.0
        self.save_hyperparameters(self.config)

        self.sampling_rate = self.model_config.sampling_rate  # 16000
        self.token_samples = int(0.08 * self.sampling_rate)
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

        self.glm_tokenizer = WhisperVQEncoder.from_pretrained(
            config.model_config.glm_tokenizer_path
        )
        for name, param in self.glm_tokenizer.named_parameters():
            param.requires_grad = False
        self.glm_tokenizer.eval()

        if self.model_config.enable_projector:
            if self.global_rank == 0:
                print(f"setting up audio projector...")
            self.audio_projector = EncoderProjector(self.model_config)
            if self.model_config.freeze_projector:
                if self.global_rank == 0:
                    print(f"freeze audio projector...")
                for name, param in self.audio_projector.named_parameters():
                    param.requires_grad = False
                self.audio_projector.eval()
        else:
            self.audio_projector = None

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(config.model_config.model_name)

        for name, param in self.llm.named_parameters():
            param.requires_grad = True
        self.llm.train()

        if self.model_config.init_ckpt_path:
            print(f"loading state dict from {self.model_config.init_ckpt_path}...")
            checkpoint = torch.load(
                self.model_config.init_ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )
            try:
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"load_state_dict failed: {e}. Retry with strict=False")
                self.load_state_dict(state_dict, strict=False)

            del checkpoint

        if self.model_config.embed_only:
            if self.global_rank == 0:
                print(f"only train partial embedding weights...")
            self.partial_freeze_weights(
                self.model_config.original_vocab_size, self.model_config.lm_vocab_size
            )

        if self.model_config.enable_lora:
            if self.global_rank == 0:
                print(f"setting up lora model...")
            peft_config = LoraConfig(
                task_type=self.model_config.lora_task_type,
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)

            if self.model_config.init_ckpt_path_lora:
                print(
                    f"loading state dict from {self.model_config.init_ckpt_path_lora}..."
                )
                checkpoint = torch.load(
                    self.model_config.init_ckpt_path_lora,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
                state_dict = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )
                try:
                    self.load_state_dict(state_dict)
                except Exception as e:
                    print(f"load_state_dict failed: {e}. Retry with strict=False")
                    self.load_state_dict(state_dict, strict=False)

                del checkpoint

        if hasattr(self.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.embed_tokens
        elif hasattr(self.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.llm.model.model.model.embed_tokens

    def forward(self, batch):
        sequences, audio_masks, labels = batch

        if self.audio_projector:
            audio_tokens = sequences.clone()
            audio_tokens[audio_masks] -= self.model_config.added_audio_token_start
            audio_tokens[~audio_masks] = 0
            audio_embeds = self.glm_tokenizer.codebook(audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            sequences[audio_masks] = 0

            text_embeds = self.embed_tokens_func(sequences)

            audio_masks = audio_masks.unsqueeze(-1)
            inputs_embeds = audio_embeds * audio_masks + text_embeds * (~audio_masks)

            model_outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        else:
            model_outputs = self.llm(input_ids=sequences, labels=labels)

        return model_outputs

    def compute_accuracy(self, pad_outputs, pad_targets, ignore_label):
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return numerator.float() / denominator.float()

    def partial_freeze_weights(self, original_vocabsize, total_vocabsize):
        self.hook_handles = []

        if self.global_rank == 0:
            print(
                f"Only training partial embedding layer, from {original_vocabsize} to {total_vocabsize}"
            )

        trainable_range = (original_vocabsize, total_vocabsize)

        # Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
        def zero_out_gradient(grad):
            grad[: trainable_range[0], :] = 0
            grad[trainable_range[1] + 1 :, :] = 0
            return grad

        # Freeze all layers first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Assuming the output layer is `lm_head`
        for param in self.llm.lm_head.parameters():
            # Compute the standard deviation for He initialization
            std_dev = (2.0 / param.size(1)) ** 0.5

            # Initialize the specific rows with He initialization
            param[original_vocabsize:total_vocabsize] = (
                torch.randn((trainable_range[1] - trainable_range[0], param.size(1)))
                * std_dev
            )
            param.requires_grad = True
            # Register the hook on the weight tensor
            handle = param.register_hook(zero_out_gradient)
            self.hook_handles.append(handle)

        if hasattr(self.llm.model, "model") and hasattr(
            self.llm.model.model, "embed_tokens"
        ):
            embed_tokens_module = self.llm.model.model.embed_tokens
        elif hasattr(self.llm.model, "embed_tokens"):
            embed_tokens_module = self.llm.model.embed_tokens
        else:
            raise AttributeError("Cannot find embed_tokens in self.llm.model")

        # For non-tied embedding layers, both the two embedding layers need to be hooked
        if self.llm.lm_head.weight.data_ptr() != embed_tokens_module.weight.data_ptr():
            for param in embed_tokens_module.parameters():
                std_dev = (2.0 / param.size(1)) ** 0.5
                param[original_vocabsize:total_vocabsize] = (
                    torch.randn(
                        (trainable_range[1] - trainable_range[0], param.size(1))
                    )
                    * std_dev
                )
                param.requires_grad = True
                handle = param.register_hook(zero_out_gradient)
                self.hook_handles.append(handle)

    def check_en(self, text):
        symbol_pattern = re.compile(
            r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
            r"\u2000-\u206F"
            r"\u3000-\u303F"
            r"\uFF00-\uFFEF]"
        )

        for char in reversed(text):
            if char.isdigit() or symbol_pattern.match(char):
                continue
            if char >= "\u4e00" and char <= "\u9fff":  # is chinese
                return False
            else:
                return True

        return True

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
