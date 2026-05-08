
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import logging
import os
import pathlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence
import types
import datasets
import deepspeed
import tokenizers
import torch
from transformers import TrainerCallback, TrainingArguments, HfArgumentParser, Trainer
from transformers.integrations import WandbCallback
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms
from training_utils import rank0_print, rank0_pprint, MYEpochSaveCallback, set_trainable, compress_strings_set
from transformers.integrations import WandbCallback

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)

from DataLoaders.StepAudioDuplexDatasetV6 import DataCollatorForSupervisedDataset, LazySupervisedDataset

from Models.StepAudioDuplexV6 import (
    StepAudio2FullDuplexConfig, 
    StepAudio2FullDuplex
)
 
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    initialize: Optional[bool] = field(default=False)
    control_token_chunk_size: int = field(default=None)
    stoken_layer_num: int = field(default=0)
    control_layer_num: int = field(default=0)
    adding_text_hiddenstates: bool = field(default=False)
    no_text_label: bool = field(default=False)
    no_stoken_label: bool = field(default=False)
    stoken_delay_num: int = field(default=0)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    ignore_backchannel: bool = field(default=False, metadata={"help": "Path to the training data."})
    window_second: float = field(default=24)
    align_audio_input: bool = field(default=False)
    filtered_backchannel_path: str = field(default=None)
    max_data_length: int = field(default=10000)
    inject_noise: bool = field(default=False) 

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    attn_implementation: str = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    only_training_backbone: bool = field(default=False)
    only_training_stoken: bool = field(default=False)

def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def initial_model(model_path, training_args, model_args):
    rank0_print("________ initial start ________")
    model_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    rank0_print("________ initial tokenizer ________")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    rank0_print("________ initial model ________")

    stepaudio_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir, 
        torch_dtype=model_dtype,
        attn_implementation=training_args.attn_implementation,
        trust_remote_code=True
    )

    stepaudio_model.get_output_embeddings = types.MethodType(lambda self: self.lm_head, stepaudio_model)
    stepaudio_model.set_output_embeddings = types.MethodType(lambda self, x: setattr(self, 'lm_head', x), stepaudio_model)
    stepaudio_model.get_input_embeddings = stepaudio_model.model.get_input_embeddings
    stepaudio_model.set_input_embeddings = stepaudio_model.model.set_input_embeddings
    
    rank0_print("________ resize_token_embeddings ________")
 
    special_tokens = ["<|S-S|>", "<|S-L|>", "<|K-L|>", "<|K-S|>", "<|Detect|>", "<|Sleep|>", "<|TextPad|>", "<|StokenPad|>", "<|AudioPad|>", "<|StokenDelay|>", "<|BackChannel|>"]
    num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
    assert num_new_tokens == len(special_tokens)
    
    stepaudio_model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = stepaudio_model.get_input_embeddings().weight.data
        output_embeddings = stepaudio_model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    rank0_print("num_new_tokens", num_new_tokens)

    rank0_print("________ initial config ________")

    model_config = stepaudio_model.config.to_dict()
    model_config['text_config']['vocab_size'] = len(tokenizer)

    stoken_config = copy.deepcopy(model_config['text_config'])
    control_config = copy.deepcopy(model_config['text_config'])
    stoken_config["num_hidden_layers"] = model_args.stoken_layer_num
    control_config["num_hidden_layers"] = model_args.control_layer_num

    new_model_config = dict(
        start_speaking_token_id=tokenizer(["<|S-S|>"]).input_ids[0][0],
        keep_listening_token_id=tokenizer(["<|K-L|>"]).input_ids[0][0],
        start_listening_token_id=tokenizer(["<|S-L|>"]).input_ids[0][0],
        keep_speaking_token_id=tokenizer(["<|K-S|>"]).input_ids[0][0],
        detect_token_id=tokenizer(["<|Detect|>"]).input_ids[0][0],
        sleep_token_id=tokenizer(["<|Sleep|>"]).input_ids[0][0],
        text_pad_token_id=tokenizer(["<|TextPad|>"]).input_ids[0][0],
        stoken_pad_token_id=tokenizer(["<|StokenPad|>"]).input_ids[0][0],
        audio_pad_token_id=tokenizer(["<|AudioPad|>"]).input_ids[0][0],
        stoken_delay_token_id=tokenizer(["<|StokenDelay|>"]).input_ids[0][0],
        start_bc_token_id=tokenizer(["<|BackChannel|>"]).input_ids[0][0],
        control_token_chunk_size=model_args.control_token_chunk_size,
        adding_text_hiddenstates=model_args.adding_text_hiddenstates,
        stoken_layer_config=stoken_config,
        control_layer_config=control_config,
        no_text_label=model_args.no_text_label,
        no_stoken_label=model_args.no_stoken_label,
        stoken_delay_num=model_args.stoken_delay_num,
        **model_config
    )
    new_model_config = StepAudio2FullDuplexConfig(**new_model_config)

    rank0_print(new_model_config)

    model = StepAudio2FullDuplex._from_config(
        new_model_config, 
        torch_dtype=model_dtype,
        attn_implementation=training_args.attn_implementation,
    )

    rank0_print("________ load model ________")

    state_dict = stepaudio_model.state_dict()
    # state_dict["control_head.weight"] = state_dict["lm_head.weight"]
    # state_dict["stoken_head.weight"] = state_dict["lm_head.weight"]

    pattern = r"model\.layers\.(\d+)"
    for k in list(state_dict.keys()):
        match = re.match(pattern, k)
        if match:
            layer = int(match.group(1))
            rest = k[len(match.group(0)) :]
            if layer >= model.config.text_config.num_hidden_layers - model_args.stoken_layer_num:
                target_layer = layer - (model.config.text_config.num_hidden_layers - model_args.stoken_layer_num)
                state_dict[f"stoken_model.layers.{target_layer}{rest}"] = state_dict[k]
            if layer >= model.config.text_config.num_hidden_layers - model_args.control_layer_num:
                target_layer = layer - (model.config.text_config.num_hidden_layers - model_args.control_layer_num)
                state_dict[f"control_model.layers.{target_layer}{rest}"] = state_dict[k]
    
    pattern = r"model\."
    for k in list(state_dict.keys()):
        match = re.match(pattern, k)
        if match:
            rest = k[len(match.group(0)) :]
            if not rest.startswith("layers."):
                if model_args.stoken_layer_num > 0:
                    assert f"stoken_model.{rest}" not in state_dict
                    state_dict[f"stoken_model.{rest}"] = state_dict[k]
                if model_args.control_layer_num > 0:
                    assert f"control_model.{rest}" not in state_dict
                    state_dict[f"control_model.{rest}"] = state_dict[k]

    message = model.load_state_dict(state_dict, strict=True)
    rank0_print(f"loading message: {message}")
    # print("debug !!!!!!!!!!")

    rank0_print("________ checking model dtype ________")

    # 参数类型检查
    for name, param in model.named_parameters():
        assert param.dtype == model_dtype, f"{name} dtype is {param.dtype} but not {model_dtype}"

    rank0_print("________ initial done ________")

    return model, tokenizer


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("model_args")
    rank0_print(model_args.__dict__)
    rank0_print("data_args")
    rank0_print(data_args.__dict__)
    rank0_print("training_args")
    rank0_print(training_args.__dict__)

    if model_args.initialize:
        model, tokenizer = initial_model(
            model_path=model_args.model_name_or_path,
            training_args=training_args,
            model_args=model_args,
        )
        
    else:
        model = StepAudio2FullDuplex.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float32),
            trust_remote_code=True,
        )
    
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

    data_args.start_speaking_token_id  = model.config.start_speaking_token_id
    data_args.keep_listening_token_id  = model.config.keep_listening_token_id
    data_args.start_listening_token_id = model.config.start_listening_token_id
    data_args.keep_speaking_token_id = model.config.keep_speaking_token_id
    data_args.detect_token_id = model.config.detect_token_id
    data_args.sleep_token_id = model.config.sleep_token_id
    data_args.text_pad_token_id = model.config.text_pad_token_id
    data_args.stoken_pad_token_id = model.config.stoken_pad_token_id
    data_args.audio_pad_token_id = model.config.audio_pad_token_id
    data_args.stoken_delay_token_id = model.config.stoken_delay_token_id
    data_args.start_bc_token_id = model.config.start_bc_token_id
    data_args.control_token_chunk_size = model.config.control_token_chunk_size
    data_args.adding_text_hiddenstates = model.config.adding_text_hiddenstates
    data_args.stoken_delay_num = model.config.stoken_delay_num
    
    data_args.audio_token_id = tokenizer(["<audio_patch>"]).input_ids[0][0]
    data_args.tts_start_id = tokenizer(["<tts_start>"]).input_ids[0][0]
    data_args.tts_pad_id = tokenizer(["<tts_pad>"]).input_ids[0][0]
    data_args.tts_end_id = tokenizer(["<tts_end>"]).input_ids[0][0]
    data_args.eot_id = tokenizer(["<|EOT|>"]).input_ids[0][0]
    
    data_args.no_stoken_label = model.config.no_stoken_label

    # # for deepspeed zero-3 auto calculate
    # model.config.hidden_size = model.config.llm_config.hidden_size
    
    training_module_pattern = None
    # set_trainable(model, training_module_pattern=training_module_pattern, log=True, log_all=True) 

    if training_args.gradient_checkpointing:
        rank0_print(f"[Code] if training_args.gradient_checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if training_args.only_training_backbone:
        assert False
        model = set_trainable(
            model, 
            training_module_pattern=[
                "model",
                "adapter",
                "lm_head",
                "stoken_head",
                "control_head",
            ], 
            log=True, 
            log_all=True
        )
    elif training_args.only_training_stoken:
        model = set_trainable(
            model, 
            training_module_pattern=[
                "stoken_model",
                "stoken_head",
            ], 
            log=True, 
            log_all=True
        )
    else:
        model = set_trainable(
            model, 
            log=True, 
            log_all=True
        )

    rank0_print("--------------- Trainer Arguement ---------------")
    rank0_print(f"train_batch_size: {training_args.train_batch_size}")
    rank0_print(f"world_size: {training_args.world_size}")
    rank0_print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    rank0_print("------------------------------------------------")

    torch.cuda.empty_cache()
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        callbacks=[
            MYEpochSaveCallback(
                save_dir=training_args.output_dir,
                save_tokenizer=tokenizer
            ), 
            WandbCallback()
        ], 
        **data_module
        )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()


if __name__ == "__main__":
    train()
