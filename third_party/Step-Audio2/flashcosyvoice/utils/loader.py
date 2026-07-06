import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from flashcosyvoice.config import CosyVoice2LLMConfig


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_text_llm(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = param.weight_loader
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_speech_llm(model: nn.Module, path: str, hf_config: CosyVoice2LLMConfig):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # NOTE(xcsong): 1. load speech embedding + sos/taskid embedding + lm head
    embedding_weights = {}
    tmp_weights = torch.load(f"{path}/llm.pt", map_location="cpu", weights_only=True)
    missed, missed_names = 0, []
    for k, v in tmp_weights.items():
        if k == "speech_embedding.weight":  # torch.Size([6564, 896])
            speech_embedding_size = hf_config.speech_vocab_size  # 6562
            # NOTE(xcsong): padding to 6592 for vllm tensor parallel
            if speech_embedding_size != v.shape[0]:  # [6564, 896] -> [6562, 896]
                assert speech_embedding_size <= v.shape[0], f"speech_embedding_size should be less than or equal to {v.shape[0]}, but got {speech_embedding_size}"
                v = v[:speech_embedding_size, :]
            embedding_weights["speech_embedding.weight"] = v
        elif k == "llm_embedding.weight":  # torch.Size([2, 896]), eos and task_id
            assert v.shape[0] == 2, f"llm_embedding.weight should be of shape [2, 896], but got {v.shape}"
            embedding_weights["llm_embedding.weight"] = v
        elif k == "llm.model.model.embed_tokens.weight":  # torch.Size([151936, 896])
            embedding_weights["model.embed_tokens.weight"] = v
        elif k == "llm_decoder.weight":  # torch.Size([6564, 896])
            lm_head_size = hf_config.speech_vocab_size  # 6562
            if lm_head_size != v.shape[0]:  # [6564, 896] -> [6562, 896]
                assert lm_head_size <= v.shape[0], f"lm_head_size should be less than or equal to {v.shape[0]}, but got {lm_head_size}"
                v = v[:lm_head_size, :]
            param = model.get_parameter("lm_head.weight")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        elif k == "llm_decoder.bias":  # torch.Size([6564])
            lm_head_size = hf_config.speech_vocab_size  # 6562
            if lm_head_size != v.shape[0]:  # [6564] -> [6562]
                assert lm_head_size <= v.shape[0], f"lm_head_size should be less than or equal to {v.shape[0]}, but got {lm_head_size}"
                v = v[:lm_head_size]
            param = model.get_parameter("lm_head.bias")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        elif "llm.model." in k:
            weight_name = k.replace("llm.model.", "")
            for kk in packed_modules_mapping:
                if kk in weight_name:
                    vv, shard_id = packed_modules_mapping[kk]
                    param_name = weight_name.replace(kk, vv)
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = param.weight_loader
                        weight_loader(param, v, shard_id)
                        break
                    except Exception as e:
                        print(e)
                        print(f"skip parameter (1): {weight_name}")
                        continue
            else:
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, v)
                except Exception as e:
                    print(e)
                    print(f"skip parameter (2): {weight_name}")
                    continue
        else:
            missed += 1
            missed_names.append(weight_name)
            continue
    print(f"missed {missed} parameters: {missed_names}")

    # NOTE(xcsong): 2. merge text embedding, sos/taskid embedding, and speech embedding
    text_embedding_weight = embedding_weights["model.embed_tokens.weight"].cpu()  # [151936, 896]
    sos_taskid_embedding_weight = embedding_weights["llm_embedding.weight"].cpu()  # [2, 896]
    speech_embedding_weight = embedding_weights["speech_embedding.weight"].cpu()  # [6562, 896]
    final_embedding_weight = torch.cat([speech_embedding_weight, sos_taskid_embedding_weight, text_embedding_weight], dim=0)  # [158500, 896]
    param = model.get_parameter("model.embed_tokens.weight")
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, final_embedding_weight)


def load_model(model: nn.Module, path: str, hf_config: CosyVoice2LLMConfig | None = None):
    if model.model_type == "speech_llm":
        load_speech_llm(model, path, hf_config)
    elif model.model_type == "text_llm":
        load_text_llm(model, path)
    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")
