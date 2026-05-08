"""
反正都是抄的llama训练代码, 这里就放一些一般不会改的training代码使用的工具函数
"""

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

import datasets
import deepspeed
import tokenizers
import torch
import transformers
from transformers import AutoProcessor, TrainerCallback, TrainingArguments

def rank0_pprint(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            pprint(*args)
    else:
        pprint(*args)


def rank0_print(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class MYEpochSaveCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    def __init__(
        self, 
        save_model=None, 
        save_dir=None, 
        save_processor=None, 
        save_tokenizer=None,
        skip_save_model=False,
        save_lora_base_model=False,
        save_sub_model_fn=None
        ):
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_processor = save_processor
        self.save_tokenizer = save_tokenizer
        self.skip_save_model = skip_save_model

        if save_sub_model_fn is None:
            save_sub_model_fn = []
        elif not isinstance(save_sub_model_fn, List):
            if not isinstance(save_sub_model_fn, Tuple):
                save_sub_model_fn = (save_sub_model_fn, "", {})
            save_sub_model_fn = [save_sub_model_fn]
        assert all(isinstance(fn, tuple) and len(fn) == 3 and isinstance(fn[1], str) and isinstance(fn[2], Dict) for fn in save_sub_model_fn)

        if save_lora_base_model:
            save_sub_model_fn.append((lambda model: model.base_model, {})) 
            
        self.save_sub_model_fn = save_sub_model_fn

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        # Save
        control.should_save = True
        self.__custom_save_model__(args, state, control, prefix='epoch', **kwargs)
        return control

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        self.__custom_save_model__(args, state, control, prefix='checkpoint', **kwargs)
        return control

    def __custom_save_model__(self, args: TrainingArguments, state, control, prefix='checkpoint', **kwargs):
        if self.save_dir is not None and torch.distributed.get_rank() == 0:
            save_dir = os.path.join(self.save_dir, f"{prefix}-{state.global_step}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            if self.save_model is not None and not self.skip_save_model:
                self.save_model.save_pretrained(save_dir)
            
            if self.save_sub_model_fn is not None:
                for fn, name, save_kwargs in self.save_sub_model_fn:
                    if name:
                        sub_save_dir = os.path.join(save_dir, name)
                    else:
                        sub_save_dir = save_dir
                    fn(self.save_model).save_pretrained(sub_save_dir, **save_kwargs)

            if self.save_processor is not None:
                self.save_processor.save_pretrained(save_dir)

            if self.save_tokenizer is not None:
                self.save_tokenizer.save_pretrained(save_dir)


def set_trainable(model, training_module_pattern=None, log=True, log_all=False):
    if training_module_pattern is None:
        model.requires_grad_(True)
    else:
        if isinstance(training_module_pattern, str):
            training_module_pattern = [training_module_pattern]
        assert isinstance(training_module_pattern, List)

        model.requires_grad_(False)
        for n, m in model.named_modules():
            if any([re.match(p, n) for p in training_module_pattern]):
                m.requires_grad_(True)

    if log:
        all_param = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            if param.requires_grad:
                trainable_params += num_params
                if log_all:
                    rank0_print(name, num_params, 'True')
                else:
                    rank0_print(name, num_params)
            else:
                if log_all:
                    rank0_print(name, num_params, "False")
                    
            all_param += num_params

        rank0_print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    return model
        

def get_peft_config(model_args):
    def get_attr(self, att, default=None):
        return getattr(self, att) if hasattr(self, att) else default

    peft_mode = model_args.peft_mode
    if peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=get_attr(model_args, "lora_target_modules", ['q_proj', 'v_proj']),
            r=get_attr(model_args, "lora_r", 16),
            lora_alpha=get_attr(model_args,"lora_alpha", 32),
            lora_dropout=get_attr(model_args,"lora_dropout", 0.05),
        )
    # Todo: 下面peft的加上参数控制
    elif peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
    elif peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
            encoder_hidden_size=512,
        )
    elif peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
        )
    else:
        raise KeyError(peft_mode)
    return peft_config


def prepare_peft_model(model, model_args, training_args, log=True):
    config = get_peft_config(model_args, training_args)
    model = get_peft_model(model, config)
    if log: model.print_trainable_parameters()
    return model


def prepare_model_for_gradient_checkpointing(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def compress_strings_set(strings):
    """
    将类似 model.layers.27.mlp.deepspeed_moe.experts.deepspeed_experts.0.up_proj.weight 的字符串按照数字压缩
    Args:
        strings:

    Returns:

    """
    # 解析字符串，按"."分割
    holder_str = "<number_holder>"

    def split_and_classify(s):
        parts = s.split(".")
        # 将每个部分分类，数字部分和非数字部分分开
        value = None
        key_parts = []
        find_digital = False
        for part in parts:
            if part.isdigit() and not find_digital:
                find_digital = True
                value = int(part)
                key_parts.append(holder_str)
            else:
                key_parts.append(part)
        key = ".".join(key_parts)
        return value, key

    # 压缩数字部分，找出连续的数字区间
    def compress_numeric_parts(numeric_parts):
        numeric_parts.sort()
        ranges = []

        if not numeric_parts:
            return numeric_parts
        # 遍历数字并找到连续数字的范围
        start = end = numeric_parts[0]
        for num in numeric_parts[1:]:
            if num == end + 1:
                end = num
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = num
        # 最后一个区间
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        return ranges

    while True:
        # 对所有字符串进行分类
        grouped = {}
        for s in strings:
            value, key = split_and_classify(s)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)

        result = []
        # 重新组合压缩后的数字部分和非数字部分
        for key, values in grouped.items():
            numeric_ranges = compress_numeric_parts(values)
            # 将数字部分用"[min-max]"或单一数字表示
            if numeric_ranges:
                numeric_str = f"[{','.join(numeric_ranges)}]"
            else:
                numeric_str = ""
            # 在非数字部分合适的位置插入压缩后的数字部分
            result.append(key.replace(holder_str, numeric_str))

        if len(result) == len(strings):
            break
        else:
            strings = result

    return set(result)
