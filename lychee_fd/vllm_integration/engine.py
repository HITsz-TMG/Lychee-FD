
"""
vLLM inference engine for Lychee-FD full-duplex serving.

This module wraps the custom LycheeDuplexForVLLM model and the multi-head
sampler behind a step_generate() interface. The realtime state machine calls
that interface instead of the original multi_head_generate /
multi_head_generate_stream methods.

Design:
  - Keep listening/speaking state transitions outside vLLM.
  - Call step_generate() once per decode step.
  - Run audio encoding as preprocessing and pass features through side state.
  - Use PagedAttention to manage KV cache for all flattened branch layers.
"""

import logging
import time
import copy
import os
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from vllm import LLM, SamplingParams
    try:
        from vllm import MultiHeadRequestState
    except Exception:  # pragma: no cover - custom branch dependent
        try:
            from vllm.sequence import MultiHeadRequestState
        except Exception:
            MultiHeadRequestState = None
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from vllm.model_executor.models import ModelRegistry
    try:
        # Some vLLM versions resolve tasks from this registry module path.
        from vllm.model_executor.models.registry import ModelRegistry as _ModelRegistryV1
    except Exception:  # pragma: no cover - version dependent
        _ModelRegistryV1 = None

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    _ModelRegistryV1 = None
    MultiHeadRequestState = None

from .model_lychee import LycheeDuplexForVLLM, LycheeDuplexState
from .sampler import (
    MultiHeadSampler,
    MultiHeadSamplerOutput,
    MultiHeadSamplingParams,
)

logger = logging.getLogger(__name__)

LYCHEE_FULL_DUPLEX_ARCH = "LycheeFullDuplex"
LEGACY_FULL_DUPLEX_ARCH = "StepAudio2FullDuplex"
SUPPORTED_FULL_DUPLEX_ARCHS = (
    LYCHEE_FULL_DUPLEX_ARCH,
    LEGACY_FULL_DUPLEX_ARCH,
)
LYCHEE_FULL_DUPLEX_MODEL_TYPE = "lychee_full_duplex"
LEGACY_FULL_DUPLEX_MODEL_TYPE = "step_audio_2_full_duplex"
SUPPORTED_FULL_DUPLEX_MODEL_TYPES = (
    LYCHEE_FULL_DUPLEX_MODEL_TYPE,
    LEGACY_FULL_DUPLEX_MODEL_TYPE,
)


def _to_attr_config(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_attr_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_config(v) for v in value]
    return value


def _load_config_json(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_duplex_auto_config_registered():
    from transformers import AutoConfig, PretrainedConfig

    class LycheeFullDuplexConfig(PretrainedConfig):
        model_type = LYCHEE_FULL_DUPLEX_MODEL_TYPE

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class LegacyStepAudio2FullDuplexConfig(PretrainedConfig):
        model_type = LEGACY_FULL_DUPLEX_MODEL_TYPE

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    for config_cls in (LycheeFullDuplexConfig, LegacyStepAudio2FullDuplexConfig):
        try:
            AutoConfig.register(config_cls.model_type, config_cls)
        except ValueError:
            # Already registered in this process.
            pass


def _load_model_config(model_path: str):
    from transformers import AutoConfig

    raw_config = None
    try:
        raw_config = _load_config_json(model_path)
    except Exception:
        raw_config = None
    if (
        isinstance(raw_config, dict)
        and raw_config.get("model_type") in SUPPORTED_FULL_DUPLEX_MODEL_TYPES
    ):
        return _to_attr_config(raw_config)

    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning(
            "AutoConfig could not load custom full-duplex config from %s (%s); "
            "falling back to raw config.json.",
            model_path,
            exc,
        )
        return _to_attr_config(raw_config if raw_config is not None else _load_config_json(model_path))


@dataclass
class StepGenerateOutput:
    """Single-step generation result from the vLLM engine."""
    text_token: int
    stoken_token: int
    control_token: int
    text_logprobs: Optional[Dict[int, float]] = None
    stoken_logprobs: Optional[Dict[int, float]] = None
    control_logprobs: Optional[Dict[int, float]] = None


class AudioPreprocessor:
    """
    Handles audio encoding outside of vLLM. The encoder and adapter are
    loaded from the HF checkpoint but run independently of vLLM's model
    forward pass.
    """

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype

        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        try:
            AudioEncoder = get_class_from_dynamic_module("modeling_step_audio_2.AudioEncoder", model_path)
            Adaptor = get_class_from_dynamic_module("modeling_step_audio_2.Adaptor", model_path)
        except Exception as e:
            raise ImportError(f"Failed to dynamically load AudioEncoder/Adaptor from the HF checkpoint: {e}")

        config = _load_model_config(model_path)

        enc_cfg = config.audio_encoder_config
        self.encoder = AudioEncoder(
            enc_cfg.n_mels, enc_cfg.n_audio_ctx, enc_cfg.n_audio_state,
            enc_cfg.n_audio_head, enc_cfg.n_audio_layer,
        ).to(device=self.device, dtype=self.dtype).eval()

        self.adapter = Adaptor(
            enc_cfg.n_audio_state, enc_cfg.llm_dim,
            enc_cfg.kernel_size, enc_cfg.adapter_stride,
        ).to(device=self.device, dtype=self.dtype).eval()

        self._load_weights(model_path)

    def _load_weights(self, model_path: str):
        import os
        import glob
        from safetensors.torch import load_file

        shard_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
        if not shard_files:
            shard_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))

        encoder_state = {}
        adapter_state = {}

        for shard in shard_files:
            if shard.endswith(".safetensors"):
                weights = load_file(shard, device="cpu")
            else:
                weights = torch.load(shard, map_location="cpu")

            for key, val in weights.items():
                if key.startswith("model.encoder.") or key.startswith("encoder."):
                    clean_key = key.split("encoder.", 1)[1]
                    encoder_state[clean_key] = val
                elif key.startswith("model.adapter.") or key.startswith("adapter."):
                    clean_key = key.split("adapter.", 1)[1]
                    adapter_state[clean_key] = val

        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(f"Loaded {len(encoder_state)} audio encoder weights")
        if adapter_state:
            self.adapter.load_state_dict(adapter_state, strict=False)
            logger.info(f"Loaded {len(adapter_state)} audio adapter weights")

    @torch.inference_mode()
    def encode(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        """
        Encode audio waveforms into embeddings.
        Returns (features, feature_lengths) ready for the model forward pass.
        """
        wavs = wavs.to(device=self.device, dtype=self.dtype)
        wav_lens = wav_lens.to(device=self.device)
        out, feat_lens = self.encoder(wavs, wav_lens)
        out = self.adapter(out)
        feat_lens = (feat_lens - 1) // 2 + 1
        return out, feat_lens


class LycheeVLLMEngine:
    """
    High-level vLLM engine for the Lychee-FD full-duplex model.

    The state machine calls step_generate() for each decode step.
    This gives us PagedAttention benefits while keeping the complex
    state machine logic outside of vLLM.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        enforce_eager: bool = True,
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")

        # Register both the current Lychee name and the legacy checkpoint alias.
        for arch_name in SUPPORTED_FULL_DUPLEX_ARCHS:
            try:
                ModelRegistry.register_model(
                    arch_name,
                    "lychee_fd.vllm_integration.model_lychee:LycheeDuplexForVLLM",
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "already" not in msg or "register" not in msg:
                    raise

        self.model_path = model_path
        self.device = device

        # Build engine
        engine_args = EngineArgs(
            model=model_path,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            task="generate",
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

        # Audio preprocessor runs outside vLLM
        self.audio_preprocessor = AudioPreprocessor(
            model_path, device=device,
            dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        )

        self.sampler = MultiHeadSampler()

        self._model_ref = None
        self._tokenizer = None

        logger.info(f"LycheeVLLMEngine initialized with model_path={model_path}")

    @property
    def model(self):
        """Access the underlying vLLM model for direct forward calls."""
        if self._model_ref is None:
            worker = self.engine.model_executor.driver_worker
            self._model_ref = worker.model_runner.model
        return self._model_ref

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.engine.tokenizer.tokenizer
        return self._tokenizer

    def precompute_audio(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        """Pre-encode audio features for use in generation."""
        return self.audio_preprocessor.encode(wavs, wav_lens)

    @torch.inference_mode()
    def step_generate(
        self,
        input_ids: torch.LongTensor,
        stoken_ids: torch.LongTensor,
        control_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        audio_embeds: Optional[torch.Tensor] = None,
        sampling_params: Optional[MultiHeadSamplingParams] = None,
        text_processors: Optional[List] = None,
        stoken_processors: Optional[List] = None,
        control_processors: Optional[List] = None,
        text_input_ids_full: Optional[torch.Tensor] = None,
        stoken_input_ids_full: Optional[torch.Tensor] = None,
        control_input_ids_full: Optional[torch.Tensor] = None,
    ) -> StepGenerateOutput:
        """
        Single-step generation for the state machine.

        This is the core API that replaces multi_head_generate() when using
        vLLM. The state machine calls this once per decode step and receives
        the sampled token triple back.

        Args:
            input_ids:     Current step text token(s)  [1, seq_len_or_1]
            stoken_ids:    Current step stoken(s)       [1, seq_len_or_1]
            control_ids:   Current step control token(s) [1, seq_len_or_1]
            positions:     Position IDs for the current tokens
            kv_caches:     List of KV cache tensors (N+M+K entries)
            attn_metadata: vLLM attention metadata
            audio_embeds:  Pre-computed audio embeddings (only for prefill)
            sampling_params: Per-head sampling configuration
            *_processors:  LogitsProcessor lists for each head
            *_input_ids_full: Full historical sequences for processors

        Returns:
            StepGenerateOutput with the 3 sampled tokens
        """
        text_logits, stoken_logits, control_logits = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            stoken_ids=stoken_ids,
            control_ids=control_ids,
            audio_embeds=audio_embeds,
        )

        text_logits = text_logits[:, -1, :]
        stoken_logits = stoken_logits[:, -1, :]
        control_logits = control_logits[:, -1, :]

        sampler_output = self.sampler.sample(
            text_logits=text_logits,
            stoken_logits=stoken_logits,
            control_logits=control_logits,
            params=sampling_params,
            text_processors=text_processors,
            stoken_processors=stoken_processors,
            control_processors=control_processors,
            text_input_ids=text_input_ids_full,
            stoken_input_ids=stoken_input_ids_full,
            control_input_ids=control_input_ids_full,
        )

        return StepGenerateOutput(
            text_token=sampler_output.text_token,
            stoken_token=sampler_output.stoken_token,
            control_token=sampler_output.control_token,
            text_logprobs=sampler_output.text_logprobs,
            stoken_logprobs=sampler_output.stoken_logprobs,
            control_logprobs=sampler_output.control_logprobs,
        )


    @torch.inference_mode()
    def step_generate_stream(
        self,
        input_ids: torch.LongTensor,
        stoken_ids: torch.LongTensor,
        control_ids: torch.LongTensor,
        prefix_input_ids: torch.LongTensor,
        audio_input_ids: torch.LongTensor,
        audio_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        sampling_params: Optional[MultiHeadSamplingParams] = None,
        text_processors: Optional[List] = None,
        stoken_processors: Optional[List] = None,
        control_processors: Optional[List] = None,
        eos_token_id: Optional[int] = None,
        stoken_eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Generator[Dict, None, None]:
        
        from vllm import SamplingParams
        import time

        device = input_ids.device
        request_id = f"audio_stream_{time.time()}"

        # Prefill: publish the initial side-channel tensors for model.forward().
        LycheeDuplexState.stoken_ids = stoken_ids
        LycheeDuplexState.control_ids = control_ids
        LycheeDuplexState.audio_embeds = audio_embeds

        # Convert multi-head sampling settings to vLLM's text sampling params.
        vllm_sampling = SamplingParams(
            temperature=sampling_params.text_temperature if sampling_params else 1.0,
            top_p=sampling_params.text_top_p if sampling_params else 1.0,
            max_tokens=max_new_tokens,
        )

        # Register the text prompt as a normal vLLM request.
        prompt_list = input_ids[0].tolist()
        # self.engine.add_request(request_id, prompt=None, prompt_token_ids=prompt_list, sampling_params=vllm_sampling)
        self.engine.add_request(
            request_id=request_id, 
            inputs={"prompt_token_ids": prompt_list}, 
            params=vllm_sampling
        )

        step = 0
        generated_input_ids = input_ids.clone()
        generated_stoken_ids = stoken_ids.clone()
        generated_control_ids = control_ids.clone()

        # Poll vLLM instead of calling model.forward() manually.
        while self.engine.has_unfinished_requests():
            # vLLM allocates KV cache and calls forward internally.
            step_outputs = self.engine.step()

            for request_output in step_outputs:
                if request_output.request_id != request_id:
                    continue

                # Read the generated text token from vLLM.
                text_token = request_output.outputs[0].token_ids[-1]
                
                # Read side-channel tokens emitted by model.forward().
                stoken_token = LycheeDuplexState.out_stoken
                control_token = LycheeDuplexState.out_control

                is_prefill = (step == 0)

                next_text = torch.tensor([[text_token]], device=device)
                next_stoken = torch.tensor([[stoken_token]], device=device)
                next_control = torch.tensor([[control_token]], device=device)

                generated_input_ids = torch.cat([generated_input_ids, next_text], dim=1)
                generated_stoken_ids = torch.cat([generated_stoken_ids, next_stoken], dim=1)
                generated_control_ids = torch.cat([generated_control_ids, next_control], dim=1)

                yield {
                    "text_token": text_token,
                    "stoken_token": stoken_token,
                    "control_token": control_token,
                    "step": step,
                    "is_prefill_done": is_prefill,
                    "sequences": generated_input_ids,
                    "stoken_ids": generated_stoken_ids,
                    "control_ids": generated_control_ids,
                }

                # Publish side-channel tensors for the next decode step.
                LycheeDuplexState.stoken_ids = next_stoken
                LycheeDuplexState.control_ids = next_control
                LycheeDuplexState.audio_embeds = None  # Only needed during prefill.

                step += 1

import inspect as _inspect


class _PatchedLycheeVLLMEngine:
    """Patched vLLM engine wrapper used by VLLMGenerationFramework."""

    @staticmethod
    def _all_model_registries():
        registries = []
        for reg in (ModelRegistry, _ModelRegistryV1):
            if reg is None:
                continue
            if any(reg is existed for existed in registries):
                continue
            registries.append(reg)
        return registries

    @classmethod
    def _register_duplex_model(cls):
        model_cls = LycheeDuplexForVLLM
        try:
            model_src = _inspect.getsourcefile(model_cls) or _inspect.getfile(model_cls)
        except Exception:
            model_src = "<unknown>"
        logger.info("Lychee-FD model class source: %s", model_src)

        try:
            from vllm.model_executor.models.registry import ModelRegistry
            for arch_name in SUPPORTED_FULL_DUPLEX_ARCHS:
                if arch_name not in ModelRegistry.get_supported_archs():
                    ModelRegistry.register_model(arch_name, model_cls)
            import vllm.model_executor.models.registry as vllm_reg
            if hasattr(vllm_reg, "_MODEL_PRIORITY"):
                for arch_name in SUPPORTED_FULL_DUPLEX_ARCHS:
                    vllm_reg._MODEL_PRIORITY[arch_name] = ["generate"]
            model_cls.supported_tasks = {"generate"}
            
        except Exception as e:
            logger.warning(f"Manual task injection failed: {e}")

        model_entry = f"{model_cls.__module__}:LycheeDuplexForVLLM"
            
        # Class-level hints for registries that inspect model attributes.
        setattr(model_cls, "is_text_generation_model", True)
        setattr(model_cls, "is_pooling_model", False)
        setattr(model_cls, "supported_tasks", {"generate"})

        registered_archs = set()
        last_error = None
        for registry in cls._all_model_registries():
            sig_params = _inspect.signature(registry.register_model).parameters
            logger.info(
                "Registering Lychee-FD architectures %s via %s.%s (supports keys: %s)",
                ",".join(SUPPORTED_FULL_DUPLEX_ARCHS),
                registry.__module__,
                registry.__name__ if hasattr(registry, "__name__") else type(registry).__name__,
                ",".join(sorted(sig_params.keys())),
            )
            extra_kwargs = {}
            if "is_text_generation_model" in sig_params:
                extra_kwargs["is_text_generation_model"] = True
            if "is_pooling_model" in sig_params:
                extra_kwargs["is_pooling_model"] = False
            if "supported_tasks" in sig_params:
                extra_kwargs["supported_tasks"] = {"generate"}
            if "task" in sig_params:
                extra_kwargs["task"] = "generate"
            if "runner" in sig_params:
                extra_kwargs["runner"] = "generate"
            if "runner_type" in sig_params:
                extra_kwargs["runner_type"] = "generate"

            unregister = getattr(registry, "unregister_model", None)
            for arch_name in SUPPORTED_FULL_DUPLEX_ARCHS:
                if callable(unregister):
                    try:
                        unregister(arch_name)
                    except Exception:
                        pass

                for entry in (model_cls, model_entry):
                    try:
                        registry.register_model(arch_name, entry, **extra_kwargs)
                        registered_archs.add(arch_name)
                        break
                    except TypeError:
                        try:
                            registry.register_model(arch_name, entry)
                            registered_archs.add(arch_name)
                            break
                        except Exception as exc:
                            last_error = exc
                    except Exception as exc:
                        # "already registered" may be raised by some versions; treat as usable.
                        msg = str(exc).lower()
                        if "already" in msg and "register" in msg:
                            registered_archs.add(arch_name)
                            break
                        last_error = exc

        if not registered_archs and last_error is not None:
            raise RuntimeError(
                "Failed to register custom Lychee-FD vLLM model architectures "
                f"{SUPPORTED_FULL_DUPLEX_ARCHS}: {last_error}"
            )

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _cfg_set(obj, key, value):
        if obj is None:
            return
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)

    @classmethod
    def _clone_text_config_object(cls, text_cfg):
        if text_cfg is None:
            return None
        if not isinstance(text_cfg, dict):
            try:
                return copy.deepcopy(text_cfg)
            except Exception:
                pass

        # Fallback when text_cfg is dict-like: rebuild a config object.
        from transformers import AutoConfig as _HFAutoConfig
        text_dict = dict(text_cfg) if isinstance(text_cfg, dict) else {}
        model_type = text_dict.get("model_type", "qwen2")
        text_dict_no_type = dict(text_dict)
        text_dict_no_type.pop("model_type", None)
        try:
            return _HFAutoConfig.for_model(model_type, **text_dict_no_type)
        except Exception:
            # Last-resort: keep dict, but this path may not satisfy vLLM checks.
            return text_dict

    @classmethod
    def _build_hf_overrides(cls, model_path: str) -> Dict:
        cfg = _load_model_config(model_path)
        text_cfg = cls._cfg_get(cfg, "text_config")
        stoken_cfg = cls._cfg_get(cfg, "stoken_layer_config")
        control_cfg = cls._cfg_get(cfg, "control_layer_config")
        merge_cfg = cls._cfg_get(cfg, "merge_layer_config")

        n_stoken = int(cls._cfg_get(stoken_cfg, "num_hidden_layers", 0) or 0)
        n_control = int(cls._cfg_get(control_cfg, "num_hidden_layers", 0) or 0)
        n_merge = int(cls._cfg_get(merge_cfg, "num_hidden_layers", 0) or 0)
        control_branch_layer = cls._cfg_get(cfg, "control_branch_layer")
        if control_branch_layer is not None:
            try:
                control_branch_layer = int(control_branch_layer)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid control_branch_layer=%s in config; fallback to control layers count.",
                    control_branch_layer,
                )
                control_branch_layer = None
        if control_branch_layer is None and n_control > 0:
            # Backward-compatible fallback when checkpoint does not carry
            # explicit control branch split metadata.
            control_branch_layer = int(n_control)

        layer_types = cls._cfg_get(text_cfg, "layer_types")
        if isinstance(layer_types, (list, tuple)) and len(layer_types) > 0:
            n_main = int(len(layer_types))
        else:
            n_main = int(cls._cfg_get(text_cfg, "num_hidden_layers", 0) or 0)
        total_layers = int(n_main + n_stoken + n_control + n_merge)

        # Keep text_config as an object (not dict), otherwise some vLLM versions
        # assert on hasattr(config.text_config, "num_attention_heads").
        text_override = cls._clone_text_config_object(text_cfg)
        cls._cfg_set(text_override, "num_hidden_layers", total_layers)

        lt = cls._cfg_get(text_override, "layer_types")
        if isinstance(lt, (list, tuple)) and len(lt) > 0:
            padded = list(lt)
            if len(padded) < total_layers:
                pad_value = padded[-1] if padded else "full_attention"
                padded.extend([pad_value] * (total_layers - len(padded)))
            elif len(padded) > total_layers:
                padded = padded[:total_layers]
            cls._cfg_set(text_override, "layer_types", padded)

        logger.info(
            "Lychee-FD layer plan: text_cfg_type=%s, main=%d, stoken=%d, control=%d, merge=%d, control_branch_layer=%s, total=%d",
            type(text_override).__name__ if text_override is not None else "None",
            n_main,
            n_stoken,
            n_control,
            n_merge,
            control_branch_layer,
            total_layers,
        )

        overrides = {
            "architectures": [LYCHEE_FULL_DUPLEX_ARCH],
            "num_hidden_layers": total_layers,
            "text_config": text_override,
            "stepaudio_main_num_hidden_layers": int(n_main),
            "stepaudio_total_num_hidden_layers": int(total_layers),
        }
        if control_branch_layer is not None:
            overrides["control_branch_layer"] = int(control_branch_layer)
        return overrides



    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        enforce_eager: Optional[bool] = True,
        enable_chunked_prefill: Optional[bool] = True,
        enable_prefix_caching: Optional[bool] = True,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        disable_log_stats: bool = True,
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")

        self.model_path = model_path
        self.device = device
        self._request_counter = 0
        self._model_ref = None
        self._tokenizer = None
        self.sampler = MultiHeadSampler()
        self._max_model_len = int(max_model_len) if max_model_len is not None else 8192
        self._active_request_id: Optional[str] = None
        self._active_text_ids: Optional[torch.Tensor] = None
        self._active_stoken_ids: Optional[torch.Tensor] = None
        self._active_control_ids: Optional[torch.Tensor] = None
        self._active_audio_ids: Optional[torch.Tensor] = None
        self._active_text_sampling_key: Optional[Tuple[bool, float, int, float]] = None
        self._active_text_processor_key = None
        self._text_after_eos_seen = False
        self._text_after_eos_token_id: Optional[int] = None
        self._text_after_eos_pad_token_id: Optional[int] = None
        self._tail_truncate_enabled = True
        self._last_tail_truncate_result: Optional[Dict[str, Any]] = None
        self._unsafe_prefix_caching = str(
            os.getenv("LYCHEEFD_VLLM_UNSAFE_PREFIX_CACHING", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._ignore_text_eos = str(
            os.getenv("LYCHEEFD_VLLM_IGNORE_TEXT_EOS", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Realtime StepAudio always keeps audio side-state across decode steps.
        # Clearing per-step breaks 400ms incremental chunk consistency.
        self._clear_audio_state_each_step = False
        strict_native_side_env = str(
            os.getenv("LYCHEEFD_VLLM_STRICT_NATIVE_SIDE_TOKENS", "0")
        ).strip().lower()
        self._strict_native_side_tokens = strict_native_side_env in {
            "1", "true", "yes", "on"
        }
        token_trace_env = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE", "")
        ).strip().lower()
        if token_trace_env:
            self._token_trace_enabled = token_trace_env in {
                "1", "true", "yes", "on"
            }
        else:
            # Native mode is typically used for branch-token debugging; trace by default.
            self._token_trace_enabled = bool(self._strict_native_side_tokens)
        self._token_trace_decode_all = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_DECODE_ALL", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._token_trace_path = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_PATH", "")
        ).strip()
        self._token_trace_dir = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_DIR", "runtime_logs")
        ).strip() or "runtime_logs"
        self._token_trace_keep_raw_final_ids = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_KEEP_RAW_FINAL_IDS", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._token_trace_keep_raw_special_hits = str(
            os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_KEEP_RAW_SPECIAL_HITS", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self._token_trace_compact_chunk_size = max(
                4, int(os.getenv("LYCHEEFD_VLLM_TOKEN_TRACE_COMPACT_CHUNK", "24"))
            )
        except (TypeError, ValueError):
            self._token_trace_compact_chunk_size = 24
        self._token_trace_fp = None
        self._token_trace_runtime_path: Optional[str] = None
        self._token_special_ids: Dict[str, Dict[str, int]] = {
            "text": {},
            "stoken": {},
            "control": {},
        }
        diag_env = str(os.getenv("LYCHEEFD_VLLM_DIAG", "0")).strip().lower()
        self._diag_enabled = diag_env in {"1", "true", "yes", "on"}
        try:
            self._diag_every = max(1, int(os.getenv("LYCHEEFD_VLLM_DIAG_EVERY", "1")))
        except (TypeError, ValueError):
            self._diag_every = 1
        if self._diag_enabled:
            logger.info(
                "Lychee-FD vLLM diagnostics enabled (LYCHEEFD_VLLM_DIAG=1, every=%d)",
                self._diag_every,
            )
            logger.info(
                "Lychee-FD side-token mode: %s (set LYCHEEFD_VLLM_STRICT_NATIVE_SIDE_TOKENS=1 to force native outputs)",
                "strict_native" if self._strict_native_side_tokens else "constrained_preferred",
            )
            logger.info(
                "Lychee-FD audio-side state: clear_each_step=%s (fixed)",
                bool(self._clear_audio_state_each_step),
            )

        self._register_duplex_model()
        _ensure_duplex_auto_config_registered()
        hf_overrides = self._build_hf_overrides(model_path)

        sig_params = _inspect.signature(EngineArgs.__init__).parameters
        engine_kwargs = {}
        if "model" in sig_params:
            engine_kwargs["model"] = model_path
        if "trust_remote_code" in sig_params:
            engine_kwargs["trust_remote_code"] = True
        if "dtype" in sig_params:
            engine_kwargs["dtype"] = dtype
        if "gpu_memory_utilization" in sig_params:
            engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        if "max_model_len" in sig_params:
            engine_kwargs["max_model_len"] = max_model_len
        if enforce_eager is False:
            logger.warning(
                "enforce_eager=False requested, but Lychee-FD duplex uses Python-side "
                "stoken/control sampling that is incompatible with vLLM CUDA graph capture. "
                "Set LYCHEEFD_VLLM_ENFORCE_EAGER=1 to avoid capture-time failures."
            )
        if "enforce_eager" in sig_params and enforce_eager is not None:
            engine_kwargs["enforce_eager"] = bool(enforce_eager)
        if "enable_chunked_prefill" in sig_params and enable_chunked_prefill is not None:
            engine_kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        requested_prefix_caching = bool(enable_prefix_caching) if enable_prefix_caching is not None else False
        self._effective_prefix_caching = requested_prefix_caching
        if requested_prefix_caching and not self._unsafe_prefix_caching:
            # Lychee-FD duplex forward depends on side-channel states (stoken/control/audio)
            # that are not part of vLLM text prefix cache keys. Prefix cache reuse can
            # return stale KV for mismatched side-channel context.
            logger.warning(
                "Disabling vLLM prefix caching for Lychee-FD duplex. "
                "This model's forward depends on stoken/control/audio side channels "
                "outside text prompt ids, so prefix cache can reuse stale KV and "
                "corrupt generation. Set LYCHEEFD_VLLM_UNSAFE_PREFIX_CACHING=1 to override."
            )
            self._effective_prefix_caching = False
        if "enable_prefix_caching" in sig_params:
            engine_kwargs["enable_prefix_caching"] = bool(self._effective_prefix_caching)
        if "max_num_seqs" in sig_params and max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = int(max_num_seqs)
        if "max_num_batched_tokens" in sig_params and max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if "tensor_parallel_size" in sig_params and tensor_parallel_size is not None:
            engine_kwargs["tensor_parallel_size"] = max(1, int(tensor_parallel_size))
        if "pipeline_parallel_size" in sig_params and pipeline_parallel_size is not None:
            engine_kwargs["pipeline_parallel_size"] = max(1, int(pipeline_parallel_size))
        if "disable_log_stats" in sig_params:
            engine_kwargs["disable_log_stats"] = bool(disable_log_stats)
        if "hf_overrides" in sig_params:
            engine_kwargs["hf_overrides"] = hf_overrides
        if "task" in sig_params:
            engine_kwargs["task"] = "generate"
        if "runner" in sig_params:
            engine_kwargs["runner"] = "generate"

        logger.info(
            "Lychee-FD vLLM engine args: tp=%s pp=%s gpu_mem_util=%s max_model_len=%s",
            engine_kwargs.get("tensor_parallel_size", "<default>"),
            engine_kwargs.get("pipeline_parallel_size", "<default>"),
            engine_kwargs.get("gpu_memory_utilization", "<default>"),
            engine_kwargs.get("max_model_len", "<default>"),
        )

        self.engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
        model_obj = self.model
        runtime_cfg = getattr(model_obj, "config", None)
        logger.info(
            "Lychee-FD model runtime layers: main=%s stoken=%s control=%s merge=%s total=%s stoken_branch_idx=%s control_branch_idx=%s control_branch_layer=%s",
            getattr(model_obj, "num_main_layers", None),
            getattr(model_obj, "num_stoken_layers", None),
            getattr(model_obj, "num_control_layers", None),
            getattr(model_obj, "num_merge_layers", None),
            getattr(model_obj, "total_num_layers", None),
            getattr(model_obj, "stoken_branch_idx", None),
            getattr(model_obj, "control_branch_idx", None),
            self._cfg_get(runtime_cfg, "control_branch_layer", None),
        )
        missing_model_ifaces = [
            name for name in ("compute_logits", "sample")
            if not hasattr(model_obj, name)
        ]
        if missing_model_ifaces:
            model_cls = model_obj.__class__.__name__
            missing = ", ".join(missing_model_ifaces)
            raise RuntimeError(
                f"Loaded Lychee-FD vLLM model class '{model_cls}' is missing required "
                f"generation interface(s): {missing}. Ensure the runtime is loading the "
                "latest lychee_fd.vllm_integration.model_lychee module."
            )
        try:
            model_runner = self.engine.model_executor.driver_worker.model_runner
            runner_name = model_runner.__class__.__name__.lower()
            if "pooling" in runner_name:
                raise RuntimeError(
                    "vLLM selected pooling runner for Lychee-FD duplex. "
                    "This engine requires generation runner. "
                    "Please use a vLLM version that supports EngineArgs task/runner "
                    "selection and force task='generate'."
                )
        except AttributeError:
            logger.debug("Could not introspect vLLM model_runner type for sanity check.")
        self.audio_preprocessor = AudioPreprocessor(
            model_path=model_path,
            device=device,
            dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        )
        self._init_token_trace_metadata()
        if self._token_trace_enabled:
            logger.info(
                "Lychee-FD token trace enabled: %s",
                self._get_token_trace_path(),
            )
            logger.info(
                "Lychee-FD token trace format: compact_chunk=%d keep_raw_final_ids=%s keep_raw_special_hits=%s",
                int(self._token_trace_compact_chunk_size),
                bool(self._token_trace_keep_raw_final_ids),
                bool(self._token_trace_keep_raw_special_hits),
            )

    @property
    def model(self):
        if self._model_ref is None:
            worker = self.engine.model_executor.driver_worker
            self._model_ref = worker.model_runner.model
        return self._model_ref

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tok = self.engine.tokenizer
            self._tokenizer = tok.tokenizer if hasattr(tok, "tokenizer") else tok
        return self._tokenizer

    def _get_token_trace_path(self) -> str:
        if self._token_trace_runtime_path:
            return self._token_trace_runtime_path
        if self._token_trace_path:
            self._token_trace_runtime_path = self._token_trace_path
            return self._token_trace_runtime_path
        mode = "native" if self._strict_native_side_tokens else "constrained"
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"vllm_token_trace_{mode}_{ts}.jsonl"
        self._token_trace_runtime_path = os.path.join(self._token_trace_dir, filename)
        return self._token_trace_runtime_path

    def _ensure_token_trace_file(self):
        if not self._token_trace_enabled:
            return
        if self._token_trace_fp is not None:
            return
        path = self._get_token_trace_path()
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._token_trace_fp = open(path, "a", encoding="utf-8")
        except Exception as exc:
            self._token_trace_enabled = False
            self._token_trace_fp = None
            logger.warning("Disable token trace: cannot open %s (%s)", path, exc)

    def _write_token_trace_record(self, payload: Dict[str, Any]):
        if not self._token_trace_enabled:
            return
        self._ensure_token_trace_file()
        if self._token_trace_fp is None:
            return
        try:
            self._token_trace_fp.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._token_trace_fp.flush()
        except Exception as exc:
            logger.warning("Failed to write token trace record: %s", exc)

    def _safe_decode_single_token(self, token_id: int) -> Optional[str]:
        try:
            return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except Exception:
            return None

    @staticmethod
    def _cfg_token_id(cfg: Any, key: str) -> Optional[int]:
        value = None
        if cfg is not None:
            if isinstance(cfg, dict):
                value = cfg.get(key)
            else:
                value = getattr(cfg, key, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _lookup_token_id_by_string(self, token_text: str) -> Optional[int]:
        tok = self.tokenizer
        try:
            encoded = tok([token_text], add_special_tokens=False).input_ids
            if encoded and encoded[0] and len(encoded[0]) == 1:
                return int(encoded[0][0])
        except Exception:
            pass
        try:
            token_id = tok.convert_tokens_to_ids(token_text)
            if token_id is None:
                return None
            token_id = int(token_id)
            unk_id = getattr(tok, "unk_token_id", None)
            if unk_id is not None and token_id == int(unk_id):
                return None
            return token_id
        except Exception:
            return None

    def _register_special_token_id(self, branch: str, token_name: str, token_id: Optional[int]):
        if token_id is None:
            return
        self._token_special_ids.setdefault(branch, {})[token_name] = int(token_id)

    def _init_token_trace_metadata(self):
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            self._register_special_token_id(
                "text", "text_pad_token_id", self._cfg_token_id(cfg, "text_pad_token_id"))
            self._register_special_token_id(
                "text", "eos_token_id", self._cfg_token_id(cfg, "eos_token_id"))
            self._register_special_token_id(
                "text", "bos_token_id", self._cfg_token_id(cfg, "bos_token_id"))

            self._register_special_token_id(
                "stoken", "stoken_pad_token_id", self._cfg_token_id(cfg, "stoken_pad_token_id"))
            self._register_special_token_id(
                "stoken", "stoken_delay_token_id", self._cfg_token_id(cfg, "stoken_delay_token_id"))

            self._register_special_token_id(
                "control", "start_speaking_token_id",
                self._cfg_token_id(cfg, "start_speaking_token_id"))
            self._register_special_token_id(
                "control", "start_listening_token_id",
                self._cfg_token_id(cfg, "start_listening_token_id"))
            self._register_special_token_id(
                "control", "keep_speaking_token_id",
                self._cfg_token_id(cfg, "keep_speaking_token_id"))
            self._register_special_token_id(
                "control", "sleep_token_id", self._cfg_token_id(cfg, "sleep_token_id"))
            self._register_special_token_id(
                "control", "detect_token_id", self._cfg_token_id(cfg, "detect_token_id"))
            self._register_special_token_id(
                "control", "start_bc_token_id", self._cfg_token_id(cfg, "start_bc_token_id"))

        for token_name, token_text in (
            ("tts_pad_id", "<tts_pad>"),
            ("tts_start_id", "<tts_start>"),
            ("tts_end_id", "<tts_end>"),
            ("tts_bos_id", "<tts_bos>"),
            ("tts_eos_id", "<tts_eos>"),
        ):
            token_id = self._lookup_token_id_by_string(token_text)
            self._register_special_token_id("text", token_name, token_id)
            self._register_special_token_id("stoken", token_name, token_id)

    def _token_special_labels(self, branch: str, token_id: int) -> List[str]:
        labels: List[str] = []
        for token_name, special_id in self._token_special_ids.get(branch, {}).items():
            if int(token_id) == int(special_id):
                labels.append(token_name)
        if branch == "stoken" and int(token_id) <= 151695:
            labels.append("stoken_not_audio_domain")
        return labels

    def _collect_special_hits(self, branch: str, token_ids: List[int]) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for idx, token_id in enumerate(token_ids):
            labels = self._token_special_labels(branch, int(token_id))
            if not labels:
                continue
            item = {
                "idx": int(idx),
                "token_id": int(token_id),
                "labels": labels,
            }
            decoded = self._safe_decode_single_token(int(token_id))
            if decoded is not None:
                item["decoded"] = decoded
            hits.append(item)
        return hits

    def _collect_special_summary(self, branch: str, token_ids: List[int]) -> Dict[str, Any]:
        total_hits = 0
        label_stats: Dict[str, Dict[str, int]] = {}
        token_stats: Dict[int, Dict[str, Any]] = {}
        for idx, token_id in enumerate(token_ids):
            token_id_int = int(token_id)
            labels = self._token_special_labels(branch, token_id_int)
            if not labels:
                continue
            total_hits += 1

            token_item = token_stats.get(token_id_int)
            if token_item is None:
                token_item = {
                    "token_id": token_id_int,
                    "count": 0,
                    "first_idx": int(idx),
                    "last_idx": int(idx),
                    "labels": [],
                }
                token_stats[token_id_int] = token_item
            token_item["count"] = int(token_item["count"]) + 1
            token_item["last_idx"] = int(idx)
            known_labels = token_item["labels"]
            for label in labels:
                if label not in known_labels:
                    known_labels.append(label)

                label_item = label_stats.get(label)
                if label_item is None:
                    label_item = {
                        "label": label,
                        "count": 0,
                        "first_idx": int(idx),
                        "last_idx": int(idx),
                    }
                    label_stats[label] = label_item
                label_item["count"] = int(label_item["count"]) + 1
                label_item["last_idx"] = int(idx)

        by_label = sorted(
            label_stats.values(),
            key=lambda item: (-int(item["count"]), str(item["label"])),
        )
        by_token = sorted(
            token_stats.values(),
            key=lambda item: (-int(item["count"]), int(item["token_id"])),
        )
        for item in by_token:
            decoded = self._safe_decode_single_token(int(item["token_id"]))
            if decoded is not None:
                item["decoded"] = decoded

        return {
            "total_hits": int(total_hits),
            "by_label": by_label,
            "by_token": by_token,
        }

    def _build_compact_sequence(self, branch: str, token_ids: List[int]) -> Dict[str, Any]:
        total = int(len(token_ids))
        if total == 0:
            return {
                "length": 0,
                "segment_count": 0,
                "segments": [],
                "top_tokens": [],
                "special_summary": {
                    "total_hits": 0,
                    "by_label": [],
                    "by_token": [],
                },
            }

        chunk_size = max(4, int(self._token_trace_compact_chunk_size))
        segments: List[Dict[str, Any]] = []
        token_counts: Dict[int, int] = {}
        raw_chunk_tokens: List[int] = []
        raw_chunk_start: Optional[int] = None

        def flush_raw_chunk():
            nonlocal raw_chunk_tokens, raw_chunk_start
            if raw_chunk_start is None or not raw_chunk_tokens:
                raw_chunk_tokens = []
                raw_chunk_start = None
                return
            segments.append({
                "type": "raw_chunk",
                "start": int(raw_chunk_start),
                "end": int(raw_chunk_start + len(raw_chunk_tokens) - 1),
                "tokens": [int(x) for x in raw_chunk_tokens],
            })
            raw_chunk_tokens = []
            raw_chunk_start = None

        i = 0
        while i < total:
            token_id = int(token_ids[i])
            j = i + 1
            while j < total and int(token_ids[j]) == token_id:
                j += 1
            run_len = int(j - i)
            token_counts[token_id] = int(token_counts.get(token_id, 0)) + run_len

            labels = self._token_special_labels(branch, token_id)
            if labels or run_len > 1:
                flush_raw_chunk()
                segment: Dict[str, Any] = {
                    "type": "repeat",
                    "start": int(i),
                    "end": int(j - 1),
                    "count": int(run_len),
                    "token_id": int(token_id),
                }
                if labels:
                    segment["labels"] = labels
                decoded = self._safe_decode_single_token(token_id)
                if decoded is not None and (labels or run_len >= 4):
                    segment["decoded"] = decoded
                segments.append(segment)
            else:
                if raw_chunk_start is None:
                    raw_chunk_start = int(i)
                raw_chunk_tokens.append(int(token_id))
                if len(raw_chunk_tokens) >= chunk_size:
                    flush_raw_chunk()
            i = j
        flush_raw_chunk()

        top_tokens: List[Dict[str, Any]] = []
        for token_id, count in sorted(
            token_counts.items(),
            key=lambda item: (-int(item[1]), int(item[0])),
        )[:12]:
            entry: Dict[str, Any] = {
                "token_id": int(token_id),
                "count": int(count),
                "ratio": float(count) / float(total),
            }
            labels = self._token_special_labels(branch, int(token_id))
            if labels:
                entry["labels"] = labels
                decoded = self._safe_decode_single_token(int(token_id))
                if decoded is not None:
                    entry["decoded"] = decoded
            top_tokens.append(entry)

        preview_len = min(chunk_size, total)
        head_tokens = [int(x) for x in token_ids[:preview_len]]
        tail_tokens = [int(x) for x in token_ids[-preview_len:]] if total > preview_len else []

        return {
            "length": int(total),
            "segment_count": int(len(segments)),
            "head": {
                "start": 0,
                "end": int(len(head_tokens) - 1),
                "tokens": head_tokens,
            },
            "tail": {
                "start": int(total - len(tail_tokens)),
                "end": int(total - 1),
                "tokens": tail_tokens,
            } if tail_tokens else None,
            "top_tokens": top_tokens,
            "special_summary": self._collect_special_summary(branch, token_ids),
            "segments": segments,
        }

    def precompute_audio(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        return self.audio_preprocessor.encode(wavs, wav_lens)

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"stepaudio_vllm_{int(time.time() * 1000)}_{self._request_counter}"

    @staticmethod
    def _normalize_processors(processors):
        if processors is None:
            return None
        if isinstance(processors, (list, tuple)):
            return list(processors)
        try:
            return list(processors)
        except TypeError:
            return [processors]

    @classmethod
    def _processor_key(cls, processors):
        normalized = cls._normalize_processors(processors)
        if normalized is None:
            return None
        return tuple(id(proc) for proc in normalized)

    @classmethod
    def _extract_text_after_eos_ids(cls, processors) -> Tuple[Optional[int], Optional[int]]:
        normalized = cls._normalize_processors(processors)
        if normalized is None:
            return None, None
        for proc in normalized:
            eos_id = getattr(proc, "eos_token_id", None)
            pad_id = getattr(proc, "text_pad_token_id", None)
            if eos_id is None or pad_id is None:
                continue
            try:
                return int(eos_id), int(pad_id)
            except (TypeError, ValueError):
                continue
        return None, None

    def _build_multihead_sampling_payload(
        self,
        params: MultiHeadSamplingParams,
    ) -> Dict[str, Any]:
        payload = {
            "text": {
                "do_sample": bool(params.text_do_sample),
                "temperature": float(params.text_temperature),
                "top_k": int(params.text_top_k),
                "top_p": float(params.text_top_p),
            },
            "stoken": {
                "do_sample": bool(params.stoken_do_sample),
                "temperature": float(params.stoken_temperature),
                "top_k": int(params.stoken_top_k),
                "top_p": float(params.stoken_top_p),
            },
            "control": {
                "do_sample": bool(params.control_do_sample),
                "temperature": float(params.control_temperature),
                "top_k": int(params.control_top_k),
                "top_p": float(params.control_top_p),
            },
        }
        if self._strict_native_side_tokens:
            # Native side-token mode consumes model-side out_stoken/out_control
            # directly inside the patched vLLM sampler.
            payload["stepaudio_require_model_side_tokens"] = True
        return payload

    @staticmethod
    def _tensor_to_int_list(tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
        if tensor is None:
            return None
        if tensor.numel() == 0:
            return []
        return [int(x) for x in tensor.detach().reshape(-1).tolist()]

    def _build_multihead_request_state(
        self,
        *,
        request_id: str,
        text_ids: torch.Tensor,
        stoken_ids: torch.Tensor,
        control_ids: torch.Tensor,
        keep_alive: bool,
        sampling_params: MultiHeadSamplingParams,
    ):
        if MultiHeadRequestState is None:
            raise RuntimeError(
                "Current vLLM runtime does not expose MultiHeadRequestState. "
                "Please use the patched vLLM branch with native multi-head protocol support."
            )
        payload = self._build_multihead_sampling_payload(sampling_params)
        return MultiHeadRequestState(
            text_input_ids=self._tensor_to_int_list(text_ids),
            stoken_input_ids=self._tensor_to_int_list(stoken_ids),
            control_input_ids=self._tensor_to_int_list(control_ids),
            keep_alive=bool(keep_alive),
            session_id=str(request_id),
            text_sampling=payload["text"],
            stoken_sampling=payload["stoken"],
            control_sampling=payload["control"],
        )

    def _build_vllm_sampling(self, params: MultiHeadSamplingParams, max_new_tokens: int):
        # Do not gate on inspect.signature here: msgspec-backed constructors
        # often expose `(*args, **kwargs)` and hide real fields (including
        # max_tokens), which can silently fall back to runtime defaults.
        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "multihead_sampling": self._build_multihead_sampling_payload(params),
            "detokenize": False,
            "skip_special_tokens": False,
        }
        if bool(getattr(self, "_ignore_text_eos", False)):
            kwargs["ignore_eos"] = True
        if params.text_do_sample:
            kwargs["temperature"] = max(float(params.text_temperature), 1e-5)
            kwargs["top_p"] = float(params.text_top_p)
            kwargs["top_k"] = int(params.text_top_k)
        else:
            kwargs["temperature"] = 0.0

        candidate = dict(kwargs)
        last_error: Optional[Exception] = None
        need_budget_alias = False

        def _validate_sampling_budget(sp_obj):
            actual_budget = getattr(sp_obj, "max_tokens", None)
            if actual_budget is None:
                raise RuntimeError(
                    "SamplingParams object has no max_tokens field; cannot "
                    "guarantee Lychee-FD streaming budget correctness."
                )
            if int(actual_budget) != int(max_new_tokens):
                raise RuntimeError(
                    "SamplingParams max_tokens mismatch: "
                    f"expected={int(max_new_tokens)} actual={int(actual_budget)}. "
                    "Refusing to continue with implicit runtime defaults."
                )
            return sp_obj

        while True:
            try:
                return _validate_sampling_budget(SamplingParams(**candidate))
            except TypeError as exc:
                last_error = exc
                msg = str(exc)
                unsupported_key = None
                if ("unexpected keyword argument" in msg.lower()
                        or "unexpected keyword" in msg.lower()):
                    for key in list(candidate.keys()):
                        if f"'{key}'" in msg or f'"{key}"' in msg:
                            unsupported_key = key
                            break
                if unsupported_key is None:
                    break
                if unsupported_key == "multihead_sampling":
                    raise RuntimeError(
                        "SamplingParams does not accept multihead_sampling in this runtime. "
                        "Please use patched vLLM with native Lychee-FD multi-head support."
                    ) from exc
                if unsupported_key in {"max_tokens", "max_new_tokens"}:
                    # Budget key must never be silently dropped, otherwise
                    # vLLM may fall back to a default max_tokens (e.g. 16).
                    need_budget_alias = True
                    break
                candidate.pop(unsupported_key, None)
                continue
            except Exception as exc:
                last_error = exc
                break

        # Some runtimes may use alternative naming for generation budget.
        if need_budget_alias:
            alias_candidate = dict(candidate)
            alias_candidate.pop("max_tokens", None)
            alias_candidate["max_new_tokens"] = int(max_new_tokens)
            try:
                return _validate_sampling_budget(SamplingParams(**alias_candidate))
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to construct SamplingParams for Lychee-FD multi-head decoding: {last_error}"
        )

    def _add_request_compat(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params,
        multihead_request_state=None,
    ):
        attempts: List[Tuple[Tuple, Dict[str, object]]] = [
            (
                tuple(),
                {
                    "request_id": request_id,
                    "inputs": {
                        "prompt_token_ids": prompt_token_ids
                    },
                    "params": sampling_params,
                },
            ),
            (
                tuple(),
                {
                    "request_id": request_id,
                    "prompt": None,
                    "prompt_token_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                },
            ),
            (
                (request_id, ),
                {
                    "prompt_token_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                },
            ),
            (
                (request_id, ),
                {
                    "inputs": {
                        "prompt_token_ids": prompt_token_ids
                    },
                    "params": sampling_params,
                },
            ),
        ]
        last_error = None
        for args, kwargs in attempts:
            call_kwargs = dict(kwargs)
            if multihead_request_state is not None:
                call_kwargs["multihead_request_state"] = multihead_request_state
            try:
                self.engine.add_request(*args, **call_kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
        if multihead_request_state is not None:
            raise RuntimeError(
                "vLLM add_request does not accept multihead_request_state in this runtime. "
                "Please ensure patched vLLM branch is installed and imported."
            ) from last_error
        raise RuntimeError(f"Unsupported vLLM add_request signature: {last_error}")

    def _abort_request_compat(self, request_id: str):
        for fn_name in ("abort_request", "abort", "remove_request"):
            fn = getattr(self.engine, fn_name, None)
            if fn is None:
                continue
            try:
                fn(request_id)
                return
            except Exception:
                continue

    @staticmethod
    def _extract_text_token(request_output) -> Optional[int]:
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return None
        first = outputs[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids:
            return int(token_ids[-1])
        token_ids = getattr(first, "output_token_ids", None)
        if token_ids:
            return int(token_ids[-1])
        token_id = getattr(first, "token_id", None)
        if token_id is not None:
            return int(token_id)
        return None

    @staticmethod
    def _extract_optional_token_list(
        request_output,
        field_name: str,
    ) -> Optional[List[int]]:
        value = getattr(request_output, field_name, None)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            token_ids: List[int] = []
            for token in value:
                if token is None:
                    continue
                try:
                    token_ids.append(int(token))
                except (TypeError, ValueError):
                    continue
            return token_ids if token_ids else None
        try:
            return [int(value)]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_multihead_tokens(cls, request_output
                                  ) -> Tuple[Optional[int], Optional[int]]:
        stoken_ids = cls._extract_optional_token_list(request_output,
                                                      "stoken_token_ids")
        control_ids = cls._extract_optional_token_list(request_output,
                                                       "control_token_ids")
        stoken_token = int(stoken_ids[-1]) if stoken_ids else None
        control_token = int(control_ids[-1]) if control_ids else None
        return stoken_token, control_token

    @staticmethod
    def _extract_model_side_tokens() -> Tuple[Optional[int], Optional[int]]:
        stoken_token = getattr(LycheeDuplexState, "out_stoken", None)
        control_token = getattr(LycheeDuplexState, "out_control", None)
        try:
            stoken_token = int(stoken_token) if stoken_token is not None else None
        except (TypeError, ValueError):
            stoken_token = None
        try:
            control_token = int(control_token) if control_token is not None else None
        except (TypeError, ValueError):
            control_token = None
        return stoken_token, control_token

    @staticmethod
    def _is_request_finished(request_output) -> bool:
        for key in ("finished", "is_finished", "request_finished", "is_request_finished"):
            value = getattr(request_output, key, None)
            if isinstance(value, bool):
                return value
        outputs = getattr(request_output, "outputs", None)
        if outputs:
            first = outputs[0]
            finish_reason = getattr(first, "finish_reason", None)
            if finish_reason is not None:
                return True
        return False

    @staticmethod
    def _extract_finish_reason(request_output) -> Optional[str]:
        outputs = getattr(request_output, "outputs", None)
        if outputs:
            first = outputs[0]
            finish_reason = getattr(first, "finish_reason", None)
            if finish_reason is not None:
                return str(finish_reason)
        for key in ("finish_reason", "stop_reason", "reason"):
            value = getattr(request_output, key, None)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _text_sampling_key(params: MultiHeadSamplingParams) -> Tuple[bool, float, int, float]:
        return (
            bool(params.text_do_sample),
            float(params.text_temperature),
            int(params.text_top_k),
            float(params.text_top_p),
        )

    @staticmethod
    def _same_tensor(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
        if a is None or b is None:
            return False
        if a.shape != b.shape:
            return False
        return torch.equal(a, b)

    @staticmethod
    def _is_prefix_tensor(prefix: Optional[torch.Tensor], full: Optional[torch.Tensor]) -> bool:
        if prefix is None or full is None:
            return False
        if prefix.dim() != full.dim():
            return False
        if prefix.dim() == 1:
            if int(full.shape[0]) < int(prefix.shape[0]):
                return False
            return torch.equal(prefix, full[: int(prefix.shape[0])])
        if prefix.dim() == 2:
            if int(prefix.shape[0]) != int(full.shape[0]):
                return False
            if int(full.shape[1]) < int(prefix.shape[1]):
                return False
            return torch.equal(prefix, full[:, : int(prefix.shape[1])])
        return False

    @staticmethod
    def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
        if tensor is None:
            return None
        return [int(x) for x in tensor.shape]

    @staticmethod
    def _init_growth_buffer(base: torch.Tensor,
                            reserve_tokens: int) -> Tuple[torch.Tensor, int]:
        current_len = int(base.shape[1])
        reserve = max(0, int(reserve_tokens))
        if reserve == 0:
            return base, current_len
        total_len = current_len + reserve
        buffer = torch.empty((base.shape[0], total_len),
                             dtype=base.dtype,
                             device=base.device)
        if current_len > 0:
            buffer[:, :current_len] = base
        return buffer, current_len

    @staticmethod
    def _ensure_growth_capacity(buffer: torch.Tensor,
                                current_len: int,
                                min_extra: int = 1) -> torch.Tensor:
        required = int(current_len) + max(1, int(min_extra))
        if required <= int(buffer.shape[1]):
            return buffer
        grow_by = max(max(16, int(buffer.shape[1]) // 2), int(min_extra))
        new_total = required + grow_by
        new_buffer = torch.empty((buffer.shape[0], new_total),
                                 dtype=buffer.dtype,
                                 device=buffer.device)
        if current_len > 0:
            new_buffer[:, :current_len] = buffer[:, :current_len]
        return new_buffer

    @staticmethod
    def _active_prefix_view(buffer: torch.Tensor, length: int) -> torch.Tensor:
        return buffer[:, :int(length)]

    def _clear_active_request_state(self):
        self._active_request_id = None
        self._active_text_ids = None
        self._active_stoken_ids = None
        self._active_control_ids = None
        self._active_audio_ids = None
        self._active_text_sampling_key = None

    def close_active_request(self) -> Dict[str, Any]:
        """Abort and forget the current kept-alive request, if one exists."""
        request_id = self._active_request_id
        if request_id is None:
            self._clear_active_request_state()
            return {"ok": True, "request_id": None, "closed": False}

        aborted = False
        try:
            if self.engine.has_unfinished_requests():
                self._abort_request_compat(request_id)
                aborted = True
        finally:
            self._clear_active_request_state()
        return {"ok": True, "request_id": request_id, "closed": True, "aborted": aborted}

    @staticmethod
    def _tail_truncate_fail(reason: str, **extra) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "ok": False,
            "reason": str(reason),
        }
        result.update(extra)
        return result

    def _find_active_sequence_group(self, request_id: str):
        schedulers = getattr(self.engine, "scheduler", None)
        if schedulers is None:
            return None, None, None
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for scheduler_index, scheduler in enumerate(schedulers):
            for queue_name in ("running",):
                queue = getattr(scheduler, queue_name, None)
                if queue is None:
                    continue
                for seq_group in list(queue):
                    if getattr(seq_group, "request_id", None) == request_id:
                        return scheduler, seq_group, {
                            "scheduler_index": int(scheduler_index),
                            "queue": queue_name,
                        }
        return None, None, None

    @staticmethod
    def _set_sequence_stage_decode(seq_data) -> None:
        try:
            from vllm.sequence import SequenceStage
            seq_data._stage = SequenceStage.DECODE
        except Exception:
            pass

    def _truncate_block_table_tail(self, block_table, seq, target_len: int) -> Dict[str, Any]:
        block_size = int(getattr(block_table, "_block_size", 0) or getattr(seq, "block_size", 0) or 0)
        if block_size <= 0:
            return self._tail_truncate_fail("invalid_block_size")
        blocks_obj = getattr(block_table, "_blocks", None)
        blocks = getattr(blocks_obj, "_blocks", None)
        if not isinstance(blocks, list) or blocks_obj is None:
            return self._tail_truncate_fail("unsupported_block_table")
        if int(target_len) <= 0:
            return self._tail_truncate_fail("invalid_target_len")
        required_blocks = max(1, int(math.ceil(float(target_len) / float(block_size))))
        if required_blocks > len(blocks):
            return self._tail_truncate_fail(
                "target_beyond_block_table",
                target_len=int(target_len),
                block_size=int(block_size),
                block_count=int(len(blocks)),
            )

        new_blocks = list(blocks[:required_blocks])
        tail_blocks = list(blocks[required_blocks:])
        last_block = new_blocks[-1]
        keep_in_last = int(target_len) - (required_blocks - 1) * int(block_size)
        if keep_in_last <= 0:
            keep_in_last = int(block_size)
        token_ids = getattr(last_block, "token_ids", None)
        if not isinstance(token_ids, list):
            return self._tail_truncate_fail("unsupported_block_token_storage")

        trimmed_partial_block = False
        if len(token_ids) > keep_in_last:
            content_hash = None
            computed = False
            try:
                content_hash = getattr(last_block, "content_hash", None)
            except Exception:
                content_hash = None
            try:
                computed = bool(getattr(last_block, "computed", False))
            except Exception:
                computed = False
            if content_hash is not None or computed:
                return self._tail_truncate_fail(
                    "cannot_trim_immutable_partial_block",
                    target_len=int(target_len),
                    block_size=int(block_size),
                    keep_in_last=int(keep_in_last),
                    token_ids_in_last=int(len(token_ids)),
                )
            del token_ids[keep_in_last:]
            trimmed_partial_block = True
            update_total = getattr(last_block, "_update_num_tokens_total", None)
            if callable(update_total):
                try:
                    update_total()
                except Exception:
                    pass

        allocator = getattr(block_table, "_allocator", None)
        if tail_blocks and allocator is None:
            return self._tail_truncate_fail("missing_block_allocator")
        freed_blocks = 0
        for block in tail_blocks:
            try:
                allocator.free(block)
                freed_blocks += 1
            except Exception as exc:
                return self._tail_truncate_fail(
                    "free_tail_block_failed",
                    error=str(exc),
                    freed_blocks=int(freed_blocks),
                )

        try:
            blocks_obj.update(new_blocks)
        except Exception as exc:
            return self._tail_truncate_fail("block_table_update_failed", error=str(exc))
        try:
            block_table._num_full_slots = int(target_len)
        except Exception as exc:
            return self._tail_truncate_fail("block_table_slots_update_failed", error=str(exc))

        return {
            "ok": True,
            "block_size": int(block_size),
            "kept_blocks": int(required_blocks),
            "freed_blocks": int(freed_blocks),
            "trimmed_partial_block": bool(trimmed_partial_block),
        }

    @staticmethod
    def _truncate_computed_tracker(block_manager, seq_id: int, target_len: int, block_size: int) -> None:
        tracker = getattr(block_manager, "_computed_blocks_tracker", None)
        if tracker is None:
            return
        token_map = getattr(tracker, "_seq_id_to_num_tokens_computed", None)
        if isinstance(token_map, dict) and int(seq_id) in token_map:
            token_map[int(seq_id)] = min(int(token_map[int(seq_id)]), int(target_len))
        hashes_map = getattr(tracker, "_seq_id_to_blocks_hashes", None)
        if isinstance(hashes_map, dict) and int(seq_id) in hashes_map:
            keep_hashes = max(0, int(target_len) // max(1, int(block_size)))
            hashes_map[int(seq_id)] = list(hashes_map[int(seq_id)][:keep_hashes])

    def truncate_active_request_to_sequences(
        self,
        *,
        text_ids: torch.Tensor,
        stoken_ids: torch.Tensor,
        control_ids: torch.Tensor,
        audio_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Best-effort tail truncation for the current kept-alive vLLM request."""
        if not self._tail_truncate_enabled:
            result = self._tail_truncate_fail("disabled")
            self._last_tail_truncate_result = result
            return result
        if self._active_request_id is None:
            result = self._tail_truncate_fail("no_active_request")
            self._last_tail_truncate_result = result
            return result
        if not self.engine.has_unfinished_requests():
            result = self._tail_truncate_fail("engine_has_no_unfinished_requests")
            self._last_tail_truncate_result = result
            return result
        if not self._is_prefix_tensor(text_ids, self._active_text_ids):
            result = self._tail_truncate_fail("text_not_active_prefix")
            self._last_tail_truncate_result = result
            return result
        if not self._is_prefix_tensor(stoken_ids, self._active_stoken_ids):
            result = self._tail_truncate_fail("stoken_not_active_prefix")
            self._last_tail_truncate_result = result
            return result
        if not self._is_prefix_tensor(control_ids, self._active_control_ids):
            result = self._tail_truncate_fail("control_not_active_prefix")
            self._last_tail_truncate_result = result
            return result
        if self._active_audio_ids is not None:
            if audio_ids is None:
                result = self._tail_truncate_fail("missing_audio_ids")
                self._last_tail_truncate_result = result
                return result
            if not self._is_prefix_tensor(audio_ids, self._active_audio_ids):
                result = self._tail_truncate_fail("audio_not_active_prefix")
                self._last_tail_truncate_result = result
                return result

        target_text_len = int(text_ids.shape[1])
        active_text_len = int(self._active_text_ids.shape[1]) if self._active_text_ids is not None else 0
        if target_text_len >= active_text_len:
            result = self._tail_truncate_fail(
                "not_a_tail_shrink",
                target_text_len=int(target_text_len),
                active_text_len=int(active_text_len),
            )
            self._last_tail_truncate_result = result
            return result

        scheduler, seq_group, location = self._find_active_sequence_group(self._active_request_id)
        if scheduler is None or seq_group is None:
            result = self._tail_truncate_fail("active_seq_group_not_running")
            self._last_tail_truncate_result = result
            return result
        seqs = list(seq_group.get_seqs()) if hasattr(seq_group, "get_seqs") else []
        if len(seqs) != 1:
            result = self._tail_truncate_fail("unsupported_sequence_count", sequence_count=int(len(seqs)))
            self._last_tail_truncate_result = result
            return result
        seq = seqs[0]
        prompt_len = int(seq.get_prompt_len())
        old_output_len = int(seq.get_output_len())
        old_seq_len = int(seq.get_len())
        if old_seq_len != active_text_len:
            result = self._tail_truncate_fail(
                "active_len_mismatch",
                seq_len=int(old_seq_len),
                active_text_len=int(active_text_len),
            )
            self._last_tail_truncate_result = result
            return result
        if target_text_len < prompt_len:
            result = self._tail_truncate_fail(
                "target_before_request_prompt",
                target_text_len=int(target_text_len),
                prompt_len=int(prompt_len),
            )
            self._last_tail_truncate_result = result
            return result

        block_manager = getattr(scheduler, "block_manager", None)
        block_tables = getattr(block_manager, "block_tables", None)
        if not isinstance(block_tables, dict) or int(seq.seq_id) not in block_tables:
            result = self._tail_truncate_fail("missing_block_table")
            self._last_tail_truncate_result = result
            return result
        block_result = self._truncate_block_table_tail(
            block_tables[int(seq.seq_id)],
            seq,
            int(target_text_len),
        )
        if not block_result.get("ok"):
            self._last_tail_truncate_result = block_result
            return block_result

        target_output_len = int(target_text_len) - int(prompt_len)
        old_output_ids = list(seq.data.output_token_ids)
        seq.data.output_token_ids = old_output_ids[:target_output_len]
        if hasattr(seq, "output_logprobs") and isinstance(seq.output_logprobs, list):
            seq.output_logprobs = list(seq.output_logprobs[:target_output_len])
        try:
            # In vLLM decode state, the sequence length includes the last
            # token that should be fed as the next decode input. Marking all
            # tokens as computed leaves an empty input-position list in
            # ModelRunner after tail truncation.
            target_computed_len = max(0, int(target_text_len) - 1)
            seq.data._num_computed_tokens = int(target_computed_len)
            seq.data._num_cached_tokens = min(
                int(getattr(seq.data, "_num_cached_tokens", 0) or 0),
                int(target_computed_len),
            )
            seq.data._new_appended_tokens = []
            self._set_sequence_stage_decode(seq.data)
        except Exception:
            pass
        if hasattr(seq, "_last_output_token_ids_offset"):
            seq._last_output_token_ids_offset = min(
                int(getattr(seq, "_last_output_token_ids_offset", 0) or 0),
                int(target_output_len),
            )
        if hasattr(seq, "_last_output_text_offset"):
            seq._last_output_text_offset = min(
                int(getattr(seq, "_last_output_text_offset", 0) or 0),
                len(str(getattr(seq, "output_text", "") or "")),
            )
        if hasattr(seq_group, "cached_request_output"):
            seq_group.cached_request_output = None
        self._truncate_computed_tracker(
            block_manager,
            int(seq.seq_id),
            max(0, int(target_text_len) - 1),
            int(block_result.get("block_size", getattr(seq, "block_size", 1) or 1)),
        )

        self._update_active_sequences(
            text_ids.detach().clone(),
            stoken_ids.detach().clone(),
            control_ids.detach().clone(),
            audio_ids=audio_ids.detach().clone() if audio_ids is not None else None,
        )
        result = {
            "ok": True,
            "reason": "tail_truncated",
            "request_id": self._active_request_id,
            "old_text_len": int(active_text_len),
            "new_text_len": int(target_text_len),
            "dropped_text_tokens": int(active_text_len - target_text_len),
            "prompt_len": int(prompt_len),
            "old_output_len": int(old_output_len),
            "new_output_len": int(target_output_len),
            **block_result,
        }
        if isinstance(location, dict):
            result.update(location)
        self._last_tail_truncate_result = result
        if self._diag_enabled:
            logger.info(
                "[VLLM_DIAG] request=%s tail_truncated text_len=%d->%d freed_blocks=%d",
                self._active_request_id,
                int(active_text_len),
                int(target_text_len),
                int(block_result.get("freed_blocks", 0)),
            )
        return result

    def _update_active_sequences(
        self,
        text_ids: torch.Tensor,
        stoken_ids: torch.Tensor,
        control_ids: torch.Tensor,
        audio_ids: Optional[torch.Tensor] = None,
    ):
        self._active_text_ids = text_ids
        self._active_stoken_ids = stoken_ids
        self._active_control_ids = control_ids
        if audio_ids is not None:
            self._active_audio_ids = audio_ids

    def _can_resume_active_request(
        self,
        *,
        input_ids: torch.Tensor,
        stoken_ids: torch.Tensor,
        control_ids: torch.Tensor,
        audio_ids: Optional[torch.Tensor],
        sampling_params: MultiHeadSamplingParams,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        debug_info: Dict[str, Any] = {
            "active_shapes": {
                "text": self._tensor_shape(self._active_text_ids),
                "stoken": self._tensor_shape(self._active_stoken_ids),
                "control": self._tensor_shape(self._active_control_ids),
                "audio": self._tensor_shape(self._active_audio_ids),
            },
            "incoming_shapes": {
                "text": self._tensor_shape(input_ids),
                "stoken": self._tensor_shape(stoken_ids),
                "control": self._tensor_shape(control_ids),
                "audio": self._tensor_shape(audio_ids),
            },
        }
        if self._active_request_id is None:
            return False, "no_active_request", debug_info
        if self._active_text_sampling_key != self._text_sampling_key(sampling_params):
            debug_info["active_text_sampling_key"] = self._active_text_sampling_key
            debug_info["incoming_text_sampling_key"] = self._text_sampling_key(sampling_params)
            return False, "text_sampling_mismatch", debug_info
        if not self._same_tensor(self._active_text_ids, input_ids):
            return False, "text_mismatch", debug_info
        if not self._same_tensor(self._active_stoken_ids, stoken_ids):
            return False, "stoken_mismatch", debug_info
        if not self._same_tensor(self._active_control_ids, control_ids):
            return False, "control_mismatch", debug_info

        # Audio path supports true-incremental continuation: allow resume when
        # incoming audio ids extend the active request audio ids by suffix.
        if self._active_audio_ids is None and audio_ids is None:
            debug_info["audio_match"] = "none"
            return True, "ok", debug_info
        if self._same_tensor(self._active_audio_ids, audio_ids):
            debug_info["audio_match"] = "equal"
            return True, "ok", debug_info
        if self._is_prefix_tensor(self._active_audio_ids, audio_ids):
            debug_info["audio_match"] = "prefix_extend"
            return True, "ok", debug_info
        return False, "audio_mismatch", debug_info

    def _init_duplex_state(
        self,
        *,
        text_ids: torch.Tensor,
        text_processors,
        stoken_ids: torch.Tensor,
        control_ids: torch.Tensor,
        audio_embeds: Optional[torch.Tensor],
        audio_feat_lens: Optional[torch.Tensor],
        audio_input_ids: Optional[torch.Tensor],
        prefix_input_len: int,
        audio_patch_token_id: Optional[int],
        stoken_processors,
        control_processors,
        sampling_params: MultiHeadSamplingParams,
        request_id: Optional[str] = None,
        control_early_callback=None,
    ):
        LycheeDuplexState.reset()
        LycheeDuplexState.text_input_ids = text_ids.clone()
        LycheeDuplexState.text_processors = self._normalize_processors(text_processors)
        LycheeDuplexState.text_sampling = {
            "do_sample": bool(sampling_params.text_do_sample),
            "temperature": float(sampling_params.text_temperature),
            "top_k": int(sampling_params.text_top_k),
            "top_p": float(sampling_params.text_top_p),
        }
        LycheeDuplexState.text_after_eos_seen = bool(self._text_after_eos_seen)
        LycheeDuplexState.text_after_eos_token_id = self._text_after_eos_token_id
        LycheeDuplexState.text_after_eos_pad_token_id = self._text_after_eos_pad_token_id
        LycheeDuplexState.stoken_ids = stoken_ids
        LycheeDuplexState.control_ids = control_ids
        LycheeDuplexState.audio_embeds = audio_embeds
        LycheeDuplexState.audio_feat_lens = audio_feat_lens
        LycheeDuplexState.audio_input_ids = audio_input_ids
        LycheeDuplexState.prefix_input_len = int(prefix_input_len)
        LycheeDuplexState.audio_patch_token_id = audio_patch_token_id
        LycheeDuplexState.stoken_input_ids = stoken_ids.clone()
        LycheeDuplexState.control_input_ids = control_ids.clone()
        LycheeDuplexState.stoken_processors = self._normalize_processors(stoken_processors)
        LycheeDuplexState.control_processors = self._normalize_processors(control_processors)
        LycheeDuplexState.stoken_sampling = {
            "do_sample": bool(sampling_params.stoken_do_sample),
            "temperature": float(sampling_params.stoken_temperature),
            "top_k": int(sampling_params.stoken_top_k),
            "top_p": float(sampling_params.stoken_top_p),
        }
        LycheeDuplexState.control_sampling = {
            "do_sample": bool(sampling_params.control_do_sample),
            "temperature": float(sampling_params.control_temperature),
            "top_k": int(sampling_params.control_top_k),
            "top_p": float(sampling_params.control_top_p),
        }
        LycheeDuplexState.control_early_callback = control_early_callback if callable(control_early_callback) else None
        LycheeDuplexState.control_early_initial_text_len = int(text_ids.shape[-1])
        LycheeDuplexState.diag_enabled = bool(self._diag_enabled)
        LycheeDuplexState.diag_request_id = request_id

    @torch.inference_mode()
    def step_generate_stream(
        self,
        input_ids: torch.LongTensor,
        stoken_ids: torch.LongTensor,
        control_ids: torch.LongTensor,
        prefix_input_ids: torch.LongTensor,
        audio_input_ids: torch.LongTensor,
        audio_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        sampling_params: Optional[MultiHeadSamplingParams] = None,
        text_processors: Optional[List] = None,
        stoken_processors: Optional[List] = None,
        control_processors: Optional[List] = None,
        eos_token_id: Optional[int] = None,
        stoken_eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Generator[Dict, None, None]:
        if sampling_params is None:
            raw_temperature = kwargs.get("temperature", 1.0)
            raw_top_k = kwargs.get("top_k", 0)
            raw_top_p = kwargs.get("top_p", 1.0)
            sampling_params = MultiHeadSamplingParams.from_generation_kwargs(
                temperature=float(raw_temperature if raw_temperature is not None else 1.0),
                top_k=int(raw_top_k if raw_top_k is not None else 0),
                top_p=float(raw_top_p if raw_top_p is not None else 1.0),
                text_do_sample=False,
                stoken_do_sample=True,
                control_do_sample=False,
            )

        audio_feat_lens = None
        if audio_embeds is None and kwargs.get("wavs") is not None and kwargs.get("wav_lens") is not None:
            audio_embeds, audio_feat_lens = self.precompute_audio(kwargs["wavs"], kwargs["wav_lens"])
        elif isinstance(audio_embeds, (tuple, list)) and len(audio_embeds) == 2:
            audio_embeds, audio_feat_lens = audio_embeds

        audio_patch_token_id = kwargs.get("audio_patch_token_id", None)
        if audio_patch_token_id is None:
            audio_pad_token_id = kwargs.get("audio_pad_token_id", None)
            if audio_input_ids is not None:
                flat_audio = audio_input_ids.reshape(-1)
                if audio_pad_token_id is not None:
                    non_pad = flat_audio[flat_audio != int(audio_pad_token_id)]
                else:
                    non_pad = flat_audio
                if non_pad.numel() > 0:
                    audio_patch_token_id = int(non_pad[0].item())
        if audio_patch_token_id is None:
            audio_patch_token_id = 151690

        prefix_input_len = int(prefix_input_ids.shape[1]) if prefix_input_ids is not None else 0

        device = input_ids.device
        generated_input_ids = input_ids.clone()
        generated_stoken_ids = stoken_ids.clone()
        generated_control_ids = control_ids.clone()
        reserve_tokens = max(0, int(max_new_tokens))
        text_buffer, text_len = self._init_growth_buffer(generated_input_ids,
                                                         reserve_tokens)
        stoken_buffer, stoken_len = self._init_growth_buffer(
            generated_stoken_ids, reserve_tokens)
        control_buffer, control_len = self._init_growth_buffer(
            generated_control_ids, reserve_tokens)
        generated_input_ids = self._active_prefix_view(text_buffer, text_len)
        generated_stoken_ids = self._active_prefix_view(stoken_buffer,
                                                        stoken_len)
        generated_control_ids = self._active_prefix_view(control_buffer,
                                                         control_len)
        # Default to call-scoped requests so behavior matches HF's per-call
        # multi_head_generate(_stream) semantics.
        keep_request_alive = bool(kwargs.get("keep_request_alive", False))
        reuse_check_debug: Dict[str, Any] = {}
        reuse_block_reason = "keep_alive_disabled"
        reuse_request = False
        if self._active_request_id is not None and not self.engine.has_unfinished_requests():
            reuse_block_reason = "active_request_finished_in_engine"
            self._clear_active_request_state()

        if keep_request_alive:
            reuse_request, check_reason, check_debug = self._can_resume_active_request(
                input_ids=generated_input_ids,
                stoken_ids=generated_stoken_ids,
                control_ids=generated_control_ids,
                audio_ids=audio_input_ids,
                sampling_params=sampling_params,
            )
            reuse_check_debug = check_debug
            if not reuse_request:
                reuse_block_reason = str(check_reason)

        if self._active_request_id is not None and not reuse_request:
            if self._diag_enabled:
                logger.info(
                    "[VLLM_DIAG] abort stale request=%s before new generation call (reason=%s)",
                    self._active_request_id,
                    reuse_block_reason,
                )
            self._abort_request_compat(self._active_request_id)
            self._clear_active_request_state()

        # Keep a defined budget for diagnostics/token-trace on both
        # new-request and reuse-request paths.
        text_budget = int(max_new_tokens)
        if keep_request_alive:
            remaining = max(
                1, self._max_model_len - int(generated_input_ids.shape[1]) - 1)
            text_budget = max(text_budget, remaining)

        if reuse_request:
            request_id = self._active_request_id
            audio_match = str(reuse_check_debug.get("audio_match", "none"))
            # If audio ids did not change, avoid resending audio features on
            # resume. If audio ids were extended, keep fresh audio context.
            if audio_match in {"none", "equal"}:
                audio_embeds = None
                audio_feat_lens = None
                audio_input_ids = None
            self._update_active_sequences(
                generated_input_ids,
                generated_stoken_ids,
                generated_control_ids,
                audio_ids=audio_input_ids,
            )
            if self._diag_enabled:
                logger.info(
                    "[VLLM_DIAG] request=%s reuse text_len=%d stoken_len=%d control_len=%d audio_match=%s",
                    request_id,
                    int(generated_input_ids.shape[1]),
                    int(generated_stoken_ids.shape[1]),
                    int(generated_control_ids.shape[1]),
                    audio_match,
                )
        else:
            request_id = self._next_request_id()
            sampling = self._build_vllm_sampling(sampling_params, max_new_tokens=text_budget)
            multihead_request_state = self._build_multihead_request_state(
                request_id=request_id,
                text_ids=generated_input_ids,
                stoken_ids=generated_stoken_ids,
                control_ids=generated_control_ids,
                keep_alive=keep_request_alive,
                sampling_params=sampling_params,
            )
            self._add_request_compat(
                request_id,
                input_ids[0].tolist(),
                sampling,
                multihead_request_state=multihead_request_state,
            )
            self._active_request_id = request_id
            self._active_text_sampling_key = self._text_sampling_key(sampling_params)
            self._update_active_sequences(
                generated_input_ids,
                generated_stoken_ids,
                generated_control_ids,
                audio_ids=audio_input_ids,
            )
            if self._diag_enabled:
                if keep_request_alive:
                    logger.info(
                        "[VLLM_DIAG] request=%s reuse miss reason=%s",
                        request_id,
                        reuse_block_reason,
                    )
                logger.info(
                    "[VLLM_DIAG] request=%s new keep_alive=%s text_len=%d stoken_len=%d control_len=%d budget=%d",
                    request_id,
                    bool(keep_request_alive),
                    int(generated_input_ids.shape[1]),
                    int(generated_stoken_ids.shape[1]),
                    int(generated_control_ids.shape[1]),
                    int(text_budget),
                )

        text_processor_key = self._processor_key(text_processors)
        if text_processor_key != self._active_text_processor_key:
            self._active_text_processor_key = text_processor_key
            self._text_after_eos_seen = False
        self._text_after_eos_token_id, self._text_after_eos_pad_token_id = (
            self._extract_text_after_eos_ids(text_processors)
        )
        if self._text_after_eos_token_id is None or self._text_after_eos_pad_token_id is None:
            self._text_after_eos_seen = False

        self._init_duplex_state(
            text_ids=generated_input_ids,
            text_processors=text_processors,
            stoken_ids=generated_stoken_ids,
            control_ids=generated_control_ids,
            audio_embeds=audio_embeds,
            audio_feat_lens=audio_feat_lens,
            audio_input_ids=audio_input_ids,
            prefix_input_len=prefix_input_len,
            audio_patch_token_id=audio_patch_token_id,
            stoken_processors=stoken_processors,
            control_processors=control_processors,
            sampling_params=sampling_params,
            request_id=request_id,
            control_early_callback=kwargs.get("control_early_callback"),
        )

        trace_step_text_tokens: List[int] = []
        trace_step_stoken_tokens: List[int] = []
        trace_step_control_tokens: List[int] = []
        if self._token_trace_enabled:
            self._write_token_trace_record({
                "type": "request_start",
                "timestamp_ms": int(time.time() * 1000),
                "request_id": request_id,
                "reuse_request": bool(reuse_request),
                "keep_request_alive": bool(keep_request_alive),
                "reuse_block_reason": None if reuse_request else str(reuse_block_reason),
                "reuse_debug": reuse_check_debug if isinstance(reuse_check_debug, dict) else {},
                "mode": "strict_native" if self._strict_native_side_tokens else "constrained_preferred",
                "max_new_tokens": int(max_new_tokens),
                "text_budget": int(text_budget),
                "initial_lengths": {
                    "text": int(generated_input_ids.shape[1]),
                    "stoken": int(generated_stoken_ids.shape[1]),
                    "control": int(generated_control_ids.shape[1]),
                },
                "initial_text_ids": self._tensor_to_int_list(generated_input_ids) or [],
                "initial_stoken_ids": self._tensor_to_int_list(generated_stoken_ids) or [],
                "initial_control_ids": self._tensor_to_int_list(generated_control_ids) or [],
                "special_token_ids": self._token_special_ids,
            })

        try:
            step = 0
            stop_requested = False
            quota_reached = False
            request_finished = False
            termination_reason = "loop_end"
            request_finished_reason: Optional[str] = None
            while self.engine.has_unfinished_requests():
                step_outputs = self.engine.step()
                for request_output in step_outputs:
                    if getattr(request_output, "request_id", None) != request_id:
                        continue

                    text_token = self._extract_text_token(request_output)
                    if text_token is None:
                        continue

                    native_stoken_token, native_control_token = \
                        self._extract_multihead_tokens(request_output)
                    constrained_stoken_token, constrained_control_token = \
                        self._extract_model_side_tokens()

                    if self._strict_native_side_tokens:
                        stoken_token = native_stoken_token
                        control_token = native_control_token
                        stoken_source = "native"
                        control_source = "native"
                        if stoken_token is None and constrained_stoken_token is not None:
                            stoken_token = constrained_stoken_token
                            stoken_source = "constrained_fallback"
                        if control_token is None and constrained_control_token is not None:
                            control_token = constrained_control_token
                            control_source = "constrained_fallback"
                    else:
                        # Default path: prefer constrained tokens sampled in
                        # Lychee-FD model forward (matches historical behavior).
                        stoken_token = constrained_stoken_token
                        control_token = constrained_control_token
                        stoken_source = "constrained"
                        control_source = "constrained"
                        if stoken_token is None and native_stoken_token is not None:
                            stoken_token = native_stoken_token
                            stoken_source = "native_fallback"
                        if control_token is None and native_control_token is not None:
                            control_token = native_control_token
                            control_source = "native_fallback"

                    if stoken_token is None:
                        diag_last = getattr(LycheeDuplexState, "diag_last", None)
                        diag_reason = (
                            diag_last.get("reason")
                            if isinstance(diag_last, dict)
                            else None
                        )
                        raise AssertionError(
                            "Lychee-FD vLLM missing stoken token at decode step; "
                            "refusing last-token fallback to avoid stale stoken reuse. "
                            f"request_id={request_id}, step={int(step)}, "
                            f"native_stoken={native_stoken_token}, "
                            f"constrained_stoken={constrained_stoken_token}, "
                            f"stoken_source={stoken_source}, "
                            f"last_stoken={int(generated_stoken_ids[0, -1].item())}, "
                            f"diag_reason={diag_reason}"
                        )
                    if control_token is None:
                        diag_last = getattr(LycheeDuplexState, "diag_last", None)
                        diag_reason = (
                            diag_last.get("reason")
                            if isinstance(diag_last, dict)
                            else None
                        )
                        raise AssertionError(
                            "Lychee-FD vLLM missing control token at decode step; "
                            "refusing last-token fallback to avoid stale control reuse. "
                            f"request_id={request_id}, step={int(step)}, "
                            f"native_control={native_control_token}, "
                            f"constrained_control={constrained_control_token}, "
                            f"control_source={control_source}, "
                            f"last_control={int(generated_control_ids[0, -1].item())}, "
                            f"diag_reason={diag_reason}"
                        )

                    text_token_int = int(text_token)
                    stoken_token_int = int(stoken_token)
                    control_token_int = int(control_token)
                    if (
                        self._text_after_eos_token_id is not None
                        and text_token_int == int(self._text_after_eos_token_id)
                    ):
                        self._text_after_eos_seen = True
                    trace_step_text_tokens.append(text_token_int)
                    trace_step_stoken_tokens.append(stoken_token_int)
                    trace_step_control_tokens.append(control_token_int)

                    text_labels = self._token_special_labels("text", text_token_int)
                    stoken_labels = self._token_special_labels("stoken", stoken_token_int)
                    control_labels = self._token_special_labels("control", control_token_int)
                    if self._token_trace_enabled:
                        record: Dict[str, Any] = {
                            "type": "step",
                            "timestamp_ms": int(time.time() * 1000),
                            "request_id": request_id,
                            "step": int(step),
                            "text_token": text_token_int,
                            "stoken_token": stoken_token_int,
                            "control_token": control_token_int,
                            "text_labels": text_labels,
                            "stoken_labels": stoken_labels,
                            "control_labels": control_labels,
                            "text_after_eos_seen": bool(self._text_after_eos_seen),
                            "text_after_eos_pad_token": self._text_after_eos_pad_token_id,
                            "stoken_source": stoken_source,
                            "control_source": control_source,
                            "native_stoken": (
                                int(native_stoken_token)
                                if native_stoken_token is not None else None
                            ),
                            "native_control": (
                                int(native_control_token)
                                if native_control_token is not None else None
                            ),
                            "constrained_stoken": (
                                int(constrained_stoken_token)
                                if constrained_stoken_token is not None else None
                            ),
                            "constrained_control": (
                                int(constrained_control_token)
                                if constrained_control_token is not None else None
                            ),
                        }
                        if self._token_trace_decode_all or text_labels:
                            record["text_decoded"] = self._safe_decode_single_token(text_token_int)
                        if self._token_trace_decode_all or stoken_labels:
                            record["stoken_decoded"] = self._safe_decode_single_token(stoken_token_int)
                        if self._token_trace_decode_all or control_labels:
                            record["control_decoded"] = self._safe_decode_single_token(control_token_int)
                        self._write_token_trace_record(record)

                    if self._diag_enabled and (
                        step == 0 or (step % self._diag_every == 0)
                    ):
                        diag_last = getattr(LycheeDuplexState, "diag_last",
                                            None)
                        diag_reason = (diag_last.get("reason")
                                       if isinstance(diag_last, dict) else
                                       None)
                        forward_calls = (diag_last.get("forward_calls")
                                         if isinstance(diag_last, dict) else
                                         None)
                        logger.info(
                            "[VLLM_DIAG] request=%s step=%d text=%d "
                            "stoken=%d[%s] control=%d[%s] "
                            "native_stoken=%s native_control=%s "
                            "constrained_stoken=%s constrained_control=%s "
                            "fwd=%s reason=%s",
                            request_id,
                            int(step),
                            int(text_token),
                            int(stoken_token),
                            stoken_source,
                            int(control_token),
                            control_source,
                            (str(native_stoken_token)
                             if native_stoken_token is not None else "None"),
                            (str(native_control_token)
                             if native_control_token is not None else "None"),
                            (str(constrained_stoken_token)
                             if constrained_stoken_token is not None else "None"),
                            (str(constrained_control_token)
                             if constrained_control_token is not None else "None"),
                            forward_calls,
                            diag_reason,
                        )

                    text_buffer = self._ensure_growth_capacity(
                        text_buffer, text_len, 1)
                    stoken_buffer = self._ensure_growth_capacity(
                        stoken_buffer, stoken_len, 1)
                    control_buffer = self._ensure_growth_capacity(
                        control_buffer, control_len, 1)

                    text_buffer[:, text_len] = int(text_token)
                    stoken_buffer[:, stoken_len] = int(stoken_token)
                    control_buffer[:, control_len] = int(control_token)
                    text_len += 1
                    stoken_len += 1
                    control_len += 1

                    generated_input_ids = self._active_prefix_view(
                        text_buffer, text_len)
                    generated_stoken_ids = self._active_prefix_view(
                        stoken_buffer, stoken_len)
                    generated_control_ids = self._active_prefix_view(
                        control_buffer, control_len)
                    next_stoken = generated_stoken_ids[:, -1:]
                    next_control = generated_control_ids[:, -1:]
                    self._update_active_sequences(
                        generated_input_ids,
                        generated_stoken_ids,
                        generated_control_ids,
                    )

                    LycheeDuplexState.text_input_ids = generated_input_ids
                    LycheeDuplexState.text_after_eos_seen = bool(self._text_after_eos_seen)
                    LycheeDuplexState.text_after_eos_token_id = self._text_after_eos_token_id
                    LycheeDuplexState.text_after_eos_pad_token_id = self._text_after_eos_pad_token_id
                    LycheeDuplexState.stoken_ids = next_stoken
                    LycheeDuplexState.control_ids = next_control
                    LycheeDuplexState.stoken_input_ids = generated_stoken_ids
                    LycheeDuplexState.control_input_ids = generated_control_ids

                    yield {
                        "text_token": text_token_int,
                        "stoken_token": stoken_token_int,
                        "control_token": control_token_int,
                        "step": step,
                        "is_prefill_done": step == 0,
                        "sequences": generated_input_ids,
                        "stoken_ids": generated_stoken_ids,
                        "control_ids": generated_control_ids,
                        "past_key_values": None,
                        "stoken_past_key_values": None,
                        "control_past_key_values": None,
                        "merge_past_key_values": None,
                    }
                    step += 1
                    if step >= int(max_new_tokens):
                        quota_reached = True
                        termination_reason = "quota_reached"
                        break

                    if eos_token_id is not None and int(text_token) == int(eos_token_id):
                        stop_requested = True
                        termination_reason = "text_eos_stop"
                        break
                    if stoken_eos_token_id is not None and int(stoken_token) == int(stoken_eos_token_id):
                        stop_requested = True
                        termination_reason = "stoken_eos_stop"
                        break
                    if self._is_request_finished(request_output):
                        request_finished = True
                        request_finished_reason = self._extract_finish_reason(request_output)
                        termination_reason = "request_finished"
                        break

                if stop_requested:
                    self._abort_request_compat(request_id)
                    self._clear_active_request_state()
                    break
                if request_finished:
                    self._clear_active_request_state()
                    if self._diag_enabled:
                        logger.info(
                            "[VLLM_DIAG] request=%s finished by runtime reason=%s",
                            request_id,
                            request_finished_reason,
                        )
                    break
                if quota_reached:
                    if not keep_request_alive and self._active_request_id == request_id:
                        self._abort_request_compat(request_id)
                        self._clear_active_request_state()
                    break
            if self._active_request_id == request_id and not self.engine.has_unfinished_requests():
                self._clear_active_request_state()
        finally:
            if self._token_trace_enabled:
                final_text_ids = self._tensor_to_int_list(generated_input_ids) or []
                final_stoken_ids = self._tensor_to_int_list(generated_stoken_ids) or []
                final_control_ids = self._tensor_to_int_list(generated_control_ids) or []
                end_payload: Dict[str, Any] = {
                    "type": "request_end",
                    "timestamp_ms": int(time.time() * 1000),
                    "request_id": request_id,
                    "reuse_request": bool(reuse_request),
                    "keep_request_alive": bool(keep_request_alive),
                    "reuse_block_reason": None if reuse_request else str(reuse_block_reason),
                    "steps": int(len(trace_step_text_tokens)),
                    "termination_reason": str(termination_reason),
                    "request_finished_reason": request_finished_reason,
                    "generated_text_tokens": trace_step_text_tokens,
                    "generated_stoken_tokens": trace_step_stoken_tokens,
                    "generated_control_tokens": trace_step_control_tokens,
                    "final_lengths": {
                        "text": int(len(final_text_ids)),
                        "stoken": int(len(final_stoken_ids)),
                        "control": int(len(final_control_ids)),
                    },
                    "final_text_compact": self._build_compact_sequence("text", final_text_ids),
                    "final_stoken_compact": self._build_compact_sequence("stoken", final_stoken_ids),
                    "final_control_compact": self._build_compact_sequence("control", final_control_ids),
                }
                if self._token_trace_keep_raw_final_ids:
                    end_payload["final_text_ids"] = final_text_ids
                    end_payload["final_stoken_ids"] = final_stoken_ids
                    end_payload["final_control_ids"] = final_control_ids
                if self._token_trace_keep_raw_special_hits:
                    end_payload["final_text_special_hits"] = self._collect_special_hits(
                        "text", final_text_ids
                    )
                    end_payload["final_stoken_special_hits"] = self._collect_special_hits(
                        "stoken", final_stoken_ids
                    )
                    end_payload["final_control_special_hits"] = self._collect_special_hits(
                        "control", final_control_ids
                    )
                self._write_token_trace_record(end_payload)
            # Emit forward-internal profiler summary if enabled (default off).
            _fwd_prof = getattr(LycheeDuplexState, "fwd_profile", None)
            if _fwd_prof:
                try:
                    parts = []
                    for _name, _slot in _fwd_prof.items():
                        _tot, _cnt = float(_slot[0]), int(_slot[1])
                        _avg_ms = (_tot / _cnt * 1000.0) if _cnt > 0 else 0.0
                        parts.append(
                            f"{_name}: total={_tot*1000.0:.2f}ms calls={_cnt} avg={_avg_ms:.3f}ms"
                        )
                    logger.info(
                        "[VLLM_FWD_PROFILE] request=%s %s",
                        request_id,
                        " | ".join(parts),
                    )
                except Exception as _exc:  # never let profiling break inference
                    logger.warning("[VLLM_FWD_PROFILE] emit failed: %s", _exc)
            LycheeDuplexState.reset()


LycheeVLLMEngine = _PatchedLycheeVLLMEngine
