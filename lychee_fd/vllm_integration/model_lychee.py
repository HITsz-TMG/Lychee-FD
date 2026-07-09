"""
Custom vLLM model implementation for the Lychee-FD full-duplex architecture.

The checkpoint contains main, speech-token, control, and merge branches. This
module flattens those branches into a single vLLM model so PagedAttention can
manage KV cache across all transformer layers.
"""

import math
import inspect as _inspect
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
VLLM_AVAILABLE = True
logger = logging.getLogger(__name__)


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_required(obj, key, *, context: str = "config"):
    value = _cfg_get(obj, key, None)
    if value is None:
        raise ValueError(f"Missing required {context}.{key}")
    return value


def _cfg_set(obj, key, value):
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _infer_main_num_layers(config, text_cfg, n_stoken: int, n_control: int) -> int:
    explicit_main = _cfg_get(config, "stepaudio_main_num_hidden_layers")
    if explicit_main is not None:
        try:
            return int(explicit_main)
        except (TypeError, ValueError):
            pass

    layer_types = _cfg_get(text_cfg, "layer_types")
    if isinstance(layer_types, (list, tuple)) and len(layer_types) > 0:
        return int(len(layer_types))

    total_hint = _cfg_get(config, "stepaudio_total_num_hidden_layers")
    if total_hint is not None:
        try:
            merge_cfg = _cfg_get(config, "merge_layer_config")
            n_merge = int(_cfg_get(merge_cfg, "num_hidden_layers", 0) or 0)
            total_hint = int(total_hint)
            candidate_main = total_hint - int(n_stoken) - int(n_control) - n_merge
            if candidate_main > 0:
                return int(candidate_main)
        except (TypeError, ValueError):
            pass

    return int(_cfg_get(text_cfg, "num_hidden_layers", 0) or 0)


def _infer_control_branch_idx(config, n_main: int, n_control: int) -> int:
    control_branch_layer = _cfg_get(config, "control_branch_layer")
    if control_branch_layer is not None:
        try:
            idx = int(n_main) - int(control_branch_layer)
            if 0 <= idx <= int(n_main):
                return idx
            logger.warning(
                "Invalid control_branch_layer=%s for n_main=%s; fallback to n_main-n_control.",
                control_branch_layer,
                n_main,
            )
        except (TypeError, ValueError):
            logger.warning(
                "Invalid control_branch_layer=%s; fallback to n_main-n_control.",
                control_branch_layer,
            )
    return max(0, int(n_main) - int(n_control))


def _qwen2_layer_kwargs(cfg, layer_prefix: str, *, cache_config, quant_config) -> Dict[str, object]:
    hidden_size = int(_cfg_required(cfg, "hidden_size", context=layer_prefix))
    num_heads = int(_cfg_required(cfg, "num_attention_heads", context=layer_prefix))
    return {
        "hidden_size": hidden_size,
        "intermediate_size": int(_cfg_required(cfg, "intermediate_size", context=layer_prefix)),
        "num_heads": num_heads,
        "num_kv_heads": int(_cfg_get(cfg, "num_key_value_heads", num_heads)),
        "head_dim": hidden_size // num_heads,
        "max_position_embeddings": int(_cfg_get(cfg, "max_position_embeddings", 32768)),
        "rope_theta": float(_cfg_get(cfg, "rope_theta", 1000000.0)),
        "rms_norm_eps": float(_cfg_get(cfg, "rms_norm_eps", 1e-6)),
        "layer_idx": 0,
        "cache_config": cache_config,
        "quant_config": quant_config,
        "prefix": layer_prefix,
    }


class Qwen2MLPForVLLM(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_out, _ = self.gate_proj(x)
        up_out, _ = self.up_proj(x)
        x = self.act_fn(gate_out) * up_out
        x, _ = self.down_proj(x)
        return x


class Qwen2AttentionForVLLM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_idx: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False,
        )

        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.total_num_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayerForVLLM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        layer_idx: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Qwen2AttentionForVLLM(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen2MLPForVLLM(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# Runtime state and model classes registered with vLLM.
import torch.nn.functional as _F


class LycheeDuplexState:
    text_input_ids = None
    text_processors = None
    text_sampling = None
    out_text = None
    text_after_eos_seen = False
    text_after_eos_token_id = None
    text_after_eos_pad_token_id = None
    stoken_ids = None
    control_ids = None
    audio_embeds = None
    audio_feat_lens = None
    audio_input_ids = None
    prefix_input_len = None
    audio_patch_token_id = None
    stoken_input_ids = None
    control_input_ids = None
    stoken_processors = None
    control_processors = None
    stoken_sampling = None
    control_sampling = None
    control_early_callback = None
    control_early_initial_text_len = None
    out_stoken = None
    out_control = None
    diag_enabled = False
    diag_request_id = None
    diag_forward_calls = 0
    diag_last = None
    # Forward-internal profiler accumulators (only touched when profiling on).
    fwd_profile = None
    @classmethod
    def reset(cls):
        cls.text_input_ids = None
        cls.text_processors = None
        cls.text_sampling = None
        cls.out_text = None
        cls.text_after_eos_seen = False
        cls.text_after_eos_token_id = None
        cls.text_after_eos_pad_token_id = None
        cls.stoken_ids = None
        cls.control_ids = None
        cls.audio_embeds = None
        cls.audio_feat_lens = None
        cls.audio_input_ids = None
        cls.prefix_input_len = None
        cls.audio_patch_token_id = None
        cls.stoken_input_ids = None
        cls.control_input_ids = None
        cls.stoken_processors = None
        cls.control_processors = None
        cls.stoken_sampling = None
        cls.control_sampling = None
        cls.control_early_callback = None
        cls.control_early_initial_text_len = None
        cls.out_stoken = None
        cls.out_control = None
        cls.diag_enabled = False
        cls.diag_request_id = None
        cls.diag_forward_calls = 0
        cls.diag_last = None
        cls.fwd_profile = None


class LycheeDuplexForVLLM(nn.Module):
    # Hint vLLM task resolver that this custom architecture is for generation.
    is_text_generation_model = True
    is_pooling_model = False
    supported_tasks = {"generate"}

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        text_cfg = _cfg_required(config, "text_config")
        stoken_cfg = _cfg_required(config, "stoken_layer_config")
        control_cfg = _cfg_required(config, "control_layer_config")
        merge_cfg = _cfg_required(config, "merge_layer_config")

        self.hidden_size = int(_cfg_required(text_cfg, "hidden_size", context="text_config"))
        self.vocab_size = int(_cfg_required(text_cfg, "vocab_size", context="text_config"))

        n_stoken = int(_cfg_get(stoken_cfg, "num_hidden_layers", 0) or 0)
        n_control = int(_cfg_get(control_cfg, "num_hidden_layers", 0) or 0)
        n_merge = int(_cfg_get(merge_cfg, "num_hidden_layers", 0) or 0)
        n_main = int(_infer_main_num_layers(config, text_cfg, n_stoken, n_control))

        self.num_main_layers = n_main
        self.num_stoken_layers = n_stoken
        self.num_control_layers = n_control
        self.num_merge_layers = n_merge
        self.stoken_branch_idx = max(0, n_main - n_stoken)
        self.control_branch_idx = _infer_control_branch_idx(config, n_main, n_control)
        self.total_num_layers = n_main + n_stoken + n_control + n_merge

        _cfg_set(config, "stepaudio_main_num_hidden_layers", n_main)
        _cfg_set(config, "stepaudio_total_num_hidden_layers", self.total_num_layers)
        _cfg_set(config, "num_hidden_layers", self.total_num_layers)
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            _cfg_set(getattr(model_config, "hf_config", None), "num_hidden_layers", self.total_num_layers)

        cache_config = getattr(vllm_config, "cache_config", None)
        quant_config = getattr(vllm_config, "quant_config", None)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        # Side-channel embeddings may be checkpoint-specific; keep them optional.
        self.stoken_embed_tokens = (
            VocabParallelEmbedding(self.vocab_size, self.hidden_size) if n_stoken > 0 else None
        )
        self.control_embed_tokens = (
            VocabParallelEmbedding(self.vocab_size, self.hidden_size) if n_control > 0 else None
        )
        self.main_layers = nn.ModuleList(
            [
                Qwen2DecoderLayerForVLLM(
                    **_qwen2_layer_kwargs(
                        text_cfg,
                        f"{prefix}.model.layers.{i}",
                        cache_config=cache_config,
                        quant_config=quant_config,
                    )
                )
                for i in range(n_main)
            ]
        )
        self.main_norm = nn.RMSNorm(self.hidden_size, eps=float(_cfg_get(text_cfg, "rms_norm_eps", 1e-6)))

        self.stoken_layers = nn.ModuleList(
            [
                Qwen2DecoderLayerForVLLM(
                    **_qwen2_layer_kwargs(
                        stoken_cfg,
                        f"{prefix}.stoken_model.layers.{j}",
                        cache_config=cache_config,
                        quant_config=quant_config,
                    )
                )
                for j in range(n_stoken)
            ]
        )
        self.stoken_norm = (
            nn.RMSNorm(
                int(_cfg_get(stoken_cfg, "hidden_size", self.hidden_size)),
                eps=float(_cfg_get(stoken_cfg, "rms_norm_eps", 1e-6)),
            )
            if n_stoken > 0
            else None
        )

        self.control_layers = nn.ModuleList(
            [
                Qwen2DecoderLayerForVLLM(
                    **_qwen2_layer_kwargs(
                        control_cfg,
                        f"{prefix}.control_model.layers.{k}",
                        cache_config=cache_config,
                        quant_config=quant_config,
                    )
                )
                for k in range(n_control)
            ]
        )
        self.control_norm = (
            nn.RMSNorm(
                int(_cfg_get(control_cfg, "hidden_size", self.hidden_size)),
                eps=float(_cfg_get(control_cfg, "rms_norm_eps", 1e-6)),
            )
            if n_control > 0
            else None
        )

        self.merge_layers = nn.ModuleList(
            [
                Qwen2DecoderLayerForVLLM(
                    **_qwen2_layer_kwargs(
                        merge_cfg,
                        f"{prefix}.merge_model.layers.{r}",
                        cache_config=cache_config,
                        quant_config=quant_config,
                    )
                )
                for r in range(n_merge)
            ]
        )
        self.merge_norm = (
            nn.RMSNorm(
                int(_cfg_get(merge_cfg, "hidden_size", self.hidden_size)),
                eps=float(_cfg_get(merge_cfg, "rms_norm_eps", 1e-6)),
            )
            if n_merge > 0
            else None
        )

        self.lm_head = ParallelLMHead(self.vocab_size, self.hidden_size, bias=False)
        # Side projection head for stoken/control branches.
        # We avoid calling ParallelLMHead.forward in model forward, and we also
        # avoid directly relying on its internal weight layout for custom branch
        # sampling. This mirrors HF's plain nn.Linear projection semantics.
        self.side_lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self._side_subset_enabled = str(
            os.getenv("LYCHEEFD_VLLM_SIDE_SUBSET_PROJ", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._side_subset_require_processor = str(
            os.getenv("LYCHEEFD_VLLM_SIDE_SUBSET_REQUIRE_PROCESSOR", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._side_parallel_enabled = str(
            os.getenv("LYCHEEFD_VLLM_SIDE_PARALLEL_BRANCH", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # --- perf guards (default = optimized/off) ---
        # Per-layer torch.isfinite(...).all() forces a GPU->CPU sync every layer.
        # Default skip for speed; set =1 to restore the NaN/Inf safety net.
        self._forward_finite_check = str(
            os.getenv("LYCHEEFD_VLLM_FORWARD_FINITE_CHECK", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # NaN/posinf pre-scan inside side _sample_token (two .any() syncs/step/branch).
        # Default skip; set =1 to restore.
        self._sample_nan_guard = str(
            os.getenv("LYCHEEFD_VLLM_SAMPLE_NAN_GUARD", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # nan_to_num over the full vocab in compute_logits (one full-vocab kernel/step).
        # Default skip; set =1 to restore.
        self._logits_nan_guard = str(
            os.getenv("LYCHEEFD_VLLM_LOGITS_NAN_GUARD", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Lightweight forward-internal profiler. Default off. When on, accumulates
        # per-section wall time into LycheeDuplexState.fwd_profile and the
        # engine can emit it. Uses torch.cuda.synchronize so numbers are real.
        self._forward_profile = str(
            os.getenv("LYCHEEFD_VLLM_FORWARD_PROFILE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        stoken_min = int(_cfg_get(config, "stoken_token_ids_min", 151696) or 151696)
        stoken_max = int(_cfg_get(config, "stoken_token_ids_max", self.vocab_size) or self.vocab_size)
        control_min = int(_cfg_get(config, "control_token_ids_min", stoken_max) or stoken_max)
        control_max = int(_cfg_get(config, "control_token_ids_max", self.vocab_size) or self.vocab_size)
        stoken_extra = [
            _cfg_get(config, "stoken_delay_token_id"),
            _cfg_get(config, "stoken_pad_token_id"),
            _cfg_get(config, "start_speaking_token_id"),
            _cfg_get(config, "start_listening_token_id"),
            _cfg_get(config, "keep_speaking_token_id"),
            _cfg_get(config, "keep_listening_token_id"),
            _cfg_get(config, "sleep_token_id"),
            _cfg_get(config, "detect_token_id"),
        ]
        control_extra = [
            _cfg_get(config, "start_speaking_token_id"),
            _cfg_get(config, "keep_listening_token_id"),
            _cfg_get(config, "start_listening_token_id"),
            _cfg_get(config, "keep_speaking_token_id"),
            _cfg_get(config, "start_bc_token_id"),
            _cfg_get(config, "sleep_token_id"),
            _cfg_get(config, "detect_token_id"),
        ]
        self._stoken_subset_ids = self._build_subset_token_ids(
            ranges=[(stoken_min, stoken_max)],
            extras=stoken_extra,
        )
        self._control_subset_ids = self._build_subset_token_ids(
            ranges=[(control_min, control_max)],
            extras=control_extra,
        )
        self._stoken_subset_id_set = set(
            int(x) for x in self._stoken_subset_ids.tolist()
        )
        self._control_subset_id_set = set(
            int(x) for x in self._control_subset_ids.tolist()
        )
        self._subset_device_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self._subset_weight_cache: Dict[Tuple[str, str, int, str], torch.Tensor] = {}
        self._side_parallel_stream_device: Optional[int] = None
        self._stoken_branch_stream = None
        self._control_branch_stream = None
        self.audio_patch_token_id = 151690
        self.logits_processor = self._build_logits_processor(
            self.vocab_size,
            float(_cfg_get(config, "logit_scale", 1.0)),
        )
        self.sampler = Sampler()
        profile_latency_env = str(os.getenv("LYCHEEFD_PROFILE_LATENCY", "0")).strip().lower()
        profile_side_env = str(os.getenv("LYCHEEFD_PROFILE_SIDE", "")).strip().lower()
        if profile_side_env:
            self._side_profile_enabled = profile_side_env in {"1", "true", "yes", "on"}
        else:
            self._side_profile_enabled = profile_latency_env in {"1", "true", "yes", "on"}
        try:
            profile_side_every_env = os.getenv("LYCHEEFD_PROFILE_SIDE_EVERY", os.getenv("LYCHEEFD_PROFILE_EVERY", "20"))
            self._side_profile_every = max(1, int(profile_side_every_env))
        except (TypeError, ValueError):
            self._side_profile_every = 20
        self._side_profile_calls = 0
        self._side_profile_total_sec = 0.0
        self._side_profile_proj_sec = 0.0
        self._side_profile_proc_sec = 0.0
        self._side_profile_proc_stoken_sec = 0.0
        self._side_profile_proc_control_sec = 0.0
        self._side_profile_sample_sec = 0.0
        if self._side_profile_enabled:
            logger.info(
                "Lychee-FD side-channel profiling enabled (LYCHEEFD_PROFILE_SIDE=1, every=%d)",
                self._side_profile_every,
            )
        logger.info(
            "Lychee-FD side opts: subset_proj=%s require_proc=%s parallel_branch=%s "
            "stoken_subset=%d control_subset=%d",
            bool(self._side_subset_enabled),
            bool(self._side_subset_require_processor),
            bool(self._side_parallel_enabled),
            int(self._stoken_subset_ids.numel()),
            int(self._control_subset_ids.numel()),
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def _get_channel_embed_module(self, channel: str):
        # Keep channel API stable but force shared embedding with main text path.
        return self.embed_tokens

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Project stoken/control branches with a dedicated dense head so behavior
        # matches HF (plain linear) and remains independent from vLLM sampler path.
        return torch.nn.functional.linear(hidden_states, self.side_lm_head.weight)

    def _build_subset_token_ids(
        self,
        *,
        ranges: List[Tuple[int, int]],
        extras: List[Optional[int]],
    ) -> torch.Tensor:
        token_ids: Set[int] = set()
        for start, end in ranges:
            if start is None or end is None:
                continue
            lo = max(0, int(start))
            hi = min(int(self.vocab_size), int(end))
            if hi > lo:
                token_ids.update(range(lo, hi))
        for token_id in extras:
            if token_id is None:
                continue
            tid = int(token_id)
            if 0 <= tid < int(self.vocab_size):
                token_ids.add(tid)
        if not token_ids:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(sorted(token_ids), dtype=torch.long)

    @classmethod
    def _extract_processor_token_hints(cls, processors) -> Set[int]:
        hints: Set[int] = set()
        scalar_attrs = (
            "eos_token_id",
            "text_pad_token_id",
            "ss_token_id",
            "kl_token_id",
            "bc_token_id",
            "sl_token_id",
            "ks_token_id",
            "sleep_token_id",
            "detect_token_id",
        )
        list_attrs = ("_allowed_tokens", "_speaking_allow_ids", "prefix_token_id")
        for proc in cls._iter_processors(processors):
            for attr in scalar_attrs:
                value = getattr(proc, attr, None)
                if value is None:
                    continue
                try:
                    hints.add(int(value))
                except (TypeError, ValueError):
                    continue
            for attr in list_attrs:
                values = getattr(proc, attr, None)
                if values is None:
                    continue
                if torch.is_tensor(values):
                    values = values.reshape(-1).tolist()
                elif not isinstance(values, (list, tuple, set)):
                    values = [values]
                for value in values:
                    try:
                        hints.add(int(value))
                    except (TypeError, ValueError):
                        continue
        return hints

    def _compose_side_token_ids(self, branch: str, processors) -> torch.Tensor:
        if branch == "stoken":
            base_ids = self._stoken_subset_ids
            base_set = self._stoken_subset_id_set
        elif branch == "control":
            base_ids = self._control_subset_ids
            base_set = self._control_subset_id_set
        else:
            raise ValueError(f"Unsupported side branch: {branch}")
        hints = self._extract_processor_token_hints(processors)
        if not hints:
            return base_ids
        missing = [
            tid for tid in sorted(hints)
            if (0 <= int(tid) < int(self.vocab_size)) and (int(tid) not in base_set)
        ]
        if not missing:
            return base_ids
        # Keep extra ids on the same device as base subset ids to avoid
        # cross-device cat failures during realtime generation.
        extra_ids = torch.tensor(missing, dtype=torch.long, device=base_ids.device)
        return torch.unique(torch.cat([base_ids, extra_ids], dim=0), sorted=True)

    def _get_side_token_ids_on_device(
        self,
        branch: str,
        token_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        key = (branch, device.type, int(device.index) if device.index is not None else -1)
        base_ids = self._stoken_subset_ids if branch == "stoken" else self._control_subset_ids
        if token_ids.data_ptr() == base_ids.data_ptr():
            cached = self._subset_device_cache.get(key)
            if cached is None or cached.device != device:
                cached = token_ids.to(device=device, non_blocking=True)
                self._subset_device_cache[key] = cached
            return cached
        return token_ids.to(device=device, non_blocking=True)

    def _get_side_projection_weight(
        self,
        branch: str,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        base_ids = self._stoken_subset_ids if branch == "stoken" else self._control_subset_ids
        weight = self.side_lm_head.weight
        if token_ids.data_ptr() != base_ids.data_ptr():
            return weight.index_select(0, token_ids).to(dtype=hidden_states.dtype)

        key = (
            branch,
            hidden_states.device.type,
            int(hidden_states.device.index) if hidden_states.device.index is not None else -1,
            str(hidden_states.dtype),
        )
        cached = self._subset_weight_cache.get(key)
        if cached is None or cached.device != hidden_states.device or cached.dtype != hidden_states.dtype:
            cached = weight.index_select(0, token_ids).to(dtype=hidden_states.dtype)
            self._subset_weight_cache[key] = cached
        return cached

    @staticmethod
    def _select_last_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            return hidden_states[-1:, :]
        if hidden_states.dim() == 3:
            return hidden_states[:, -1, :]
        raise ValueError(f"Unsupported hidden shape for side projection: {tuple(hidden_states.shape)}")

    def _project_side_last_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        branch: str,
        processors,
    ) -> torch.Tensor:
        last_hidden = self._select_last_hidden(hidden_states)
        if not self._side_subset_enabled:
            return self._project_logits(last_hidden)
        if self._side_subset_require_processor and processors is None:
            return self._project_logits(last_hidden)
        token_ids = self._compose_side_token_ids(branch, processors)
        if token_ids.numel() == 0:
            return self._project_logits(last_hidden)
        token_ids = self._get_side_token_ids_on_device(branch, token_ids, last_hidden.device)
        side_weight = self._get_side_projection_weight(branch, token_ids, last_hidden)
        subset_logits = torch.nn.functional.linear(last_hidden, side_weight)
        full_logits = torch.full(
            (subset_logits.shape[0], int(self.vocab_size)),
            float("-inf"),
            dtype=subset_logits.dtype,
            device=subset_logits.device,
        )
        full_logits.index_copy_(1, token_ids, subset_logits)
        return full_logits

    def _ensure_side_parallel_streams(self, device: torch.device) -> bool:
        if not self._side_parallel_enabled:
            return False
        if device.type != "cuda":
            return False
        device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        if (
            self._stoken_branch_stream is None
            or self._control_branch_stream is None
            or self._side_parallel_stream_device != device_index
        ):
            with torch.cuda.device(device_index):
                self._stoken_branch_stream = torch.cuda.Stream()
                self._control_branch_stream = torch.cuda.Stream()
            self._side_parallel_stream_device = device_index
        return True

    def _run_side_branch(
        self,
        *,
        branch_input: torch.Tensor,
        layers: nn.ModuleList,
        norm: Optional[nn.Module],
        kv_caches: List[torch.Tensor],
        positions: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        diag_enabled: bool,
    ) -> Tuple[torch.Tensor, List[Dict[str, int]], Optional[Dict[str, float]]]:
        branch_hidden = branch_input
        bad_layers: List[Dict[str, int]] = []
        pre_norm_stats = None
        for idx, layer in enumerate(layers):
            branch_hidden = layer(
                positions,
                branch_hidden,
                kv_caches[idx],
                attn_metadata,
            )
            if self._forward_finite_check and not torch.isfinite(branch_hidden).all():
                counts = self._count_non_finite(branch_hidden)
                bad_layers.append(
                    {"layer": int(idx), "nan": counts["nan"], "inf": counts["inf"]}
                )
                branch_hidden = self._sanitize_tensor_finite(branch_hidden)
        if diag_enabled:
            pre_norm_stats = self._diag_tensor_stats(branch_hidden)
        if norm is not None:
            branch_hidden = norm(branch_hidden)
            if self._forward_finite_check and not torch.isfinite(branch_hidden).all():
                counts = self._count_non_finite(branch_hidden)
                bad_layers.append(
                    {"layer": int(len(layers)), "nan": counts["nan"], "inf": counts["inf"]}
                )
                branch_hidden = self._sanitize_tensor_finite(branch_hidden)
        return branch_hidden, bad_layers, pre_norm_stats

    @staticmethod
    def _build_logits_processor(vocab_size: int, logit_scale: float):
        params = list(_inspect.signature(LogitsProcessor.__init__).parameters.keys())
        argc = max(0, len(params) - 1)  # drop `self`
        attempts = []
        if argc >= 3:
            attempts.append((int(vocab_size), int(vocab_size), float(logit_scale)))
        if argc >= 2:
            attempts.append((int(vocab_size), int(vocab_size)))
        if argc >= 1:
            attempts.append((int(vocab_size),))
        for args in attempts:
            try:
                return LogitsProcessor(*args)
            except TypeError:
                continue
        return None

    @staticmethod
    def _iter_processors(processors):
        if processors is None:
            return []
        if isinstance(processors, (list, tuple)):
            return processors
        try:
            return list(processors)
        except TypeError:
            return [processors]

    @classmethod
    def _apply_processors(cls, logits, input_ids, processors):
        if input_ids is None or processors is None:
            return logits
        updated = logits
        for proc in cls._iter_processors(processors):
            updated = proc(input_ids, updated)
        return updated

    @staticmethod
    def _sample_token(logits, *, do_sample, temperature, top_k, top_p, nan_guard=True):
        if logits.dim() == 2:
            logits = logits[0]
        # Keep -inf masks intact (critical for constrained control decoding).
        logits = logits.float()
        if nan_guard:
            if torch.isnan(logits).any():
                logits = torch.where(torch.isnan(logits), torch.full_like(logits, -1e4), logits)
            if torch.isposinf(logits).any():
                logits = torch.where(torch.isposinf(logits), torch.full_like(logits, 1e4), logits)
        if not do_sample or temperature == 0:
            return int(torch.argmax(logits).item())

        work = logits
        if temperature > 0 and temperature != 1.0:
            work = work / temperature

        if top_k and top_k > 0:
            safe_k = min(int(top_k), work.shape[-1])
            top_vals, _ = torch.topk(work, safe_k)
            threshold = top_vals[-1]
            work = torch.where(work < threshold, torch.full_like(work, float("-inf")), work)

        if top_p is not None and 0 < float(top_p) < 1.0:
            sorted_logits, sorted_indices = torch.sort(work, descending=True)
            sorted_probs = _F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > float(top_p)
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            remove_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
            remove_mask.scatter_(0, sorted_indices, sorted_mask)
            work = work.masked_fill(remove_mask, float("-inf"))

        probs = _F.softmax(work, dim=-1)
        if torch.isnan(probs).any() or probs.sum() <= 0:
            return int(torch.argmax(logits).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _force_single_token_logits(logits: torch.Tensor, token_id: int) -> torch.Tensor:
        forced = torch.full_like(logits, -1e4)
        tid = int(token_id)
        if 0 <= tid < int(forced.shape[-1]):
            forced[..., tid] = 0.0
        return forced

    def _emit_control_early_callback(self, control_token: int, positions: torch.Tensor) -> None:
        callback = LycheeDuplexState.control_early_callback
        if not callable(callback):
            return
        text_ids = LycheeDuplexState.text_input_ids
        initial_len = LycheeDuplexState.control_early_initial_text_len
        try:
            current_len = int(text_ids.shape[-1]) if text_ids is not None else 0
        except Exception:
            current_len = 0
        try:
            base_len = int(initial_len)
        except (TypeError, ValueError):
            base_len = current_len
        step = max(0, int(current_len) - int(base_len))
        try:
            pos_flat = positions.reshape(-1).to(dtype=torch.long)
            position = int(pos_flat[-1].item()) if pos_flat.numel() > 0 else None
        except Exception:
            position = None
        payload = {
            "type": "control_head_early",
            "request_id": LycheeDuplexState.diag_request_id,
            "step": int(step),
            "position": position,
            "control_token": int(control_token),
            "timestamp_epoch_ms": int(time.time() * 1000),
        }
        try:
            callback(payload)
        except Exception:
            logger.debug("control_early_callback failed", exc_info=True)

    def _sample_text_token_for_merge(self, text_hidden: torch.Tensor) -> int:
        text_last = self._project_logits(self._select_last_hidden(text_hidden))
        text_last = self._apply_processors(
            text_last,
            LycheeDuplexState.text_input_ids,
            LycheeDuplexState.text_processors,
        )

        text_ids = LycheeDuplexState.text_input_ids
        text_pad_id = LycheeDuplexState.text_after_eos_pad_token_id
        text_eos_id = LycheeDuplexState.text_after_eos_token_id
        force_text_pad = bool(LycheeDuplexState.text_after_eos_seen)
        if (
            not force_text_pad
            and text_ids is not None
            and text_eos_id is not None
        ):
            try:
                force_text_pad = bool(
                    (text_ids.reshape(-1)[-1] == int(text_eos_id)).item()
                )
            except Exception:
                force_text_pad = False
        if force_text_pad and text_pad_id is not None:
            return int(text_pad_id)

        text_cfg = LycheeDuplexState.text_sampling or {}
        return self._sample_token(
            text_last,
            do_sample=bool(text_cfg.get("do_sample", False)),
            temperature=float(text_cfg.get("temperature", 1.0)),
            top_k=int(text_cfg.get("top_k", 0)),
            top_p=float(text_cfg.get("top_p", 1.0)),
            nan_guard=self._sample_nan_guard,
        )

    @classmethod
    def _set_last_logits(cls, logits, last_logits):
        if logits.dim() == 2:
            logits[-1:, :] = last_logits
        elif logits.dim() == 3:
            logits[:, -1, :] = last_logits
        else:
            raise ValueError(f"Unsupported logits shape for last-token update: {tuple(logits.shape)}")
        return logits

    @staticmethod
    def _diag_tensor_stats(logits: torch.Tensor) -> Dict[str, float]:
        work = logits.float()
        finite = work[torch.isfinite(work)]
        return {
            "nan": int(torch.isnan(work).sum().item()),
            "posinf": int(torch.isposinf(work).sum().item()),
            "neginf": int(torch.isneginf(work).sum().item()),
            "finite_min": float(finite.min().item()) if finite.numel() > 0 else float("nan"),
            "finite_max": float(finite.max().item()) if finite.numel() > 0 else float("nan"),
        }

    @staticmethod
    def _sanitize_tensor_finite(t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _count_non_finite(t: torch.Tensor) -> Dict[str, int]:
        return {
            "nan": int(torch.isnan(t).sum().item()),
            "inf": int(torch.isinf(t).sum().item()),
        }

    @staticmethod
    def _prof_accum(name: str, dt: float) -> None:
        """Accumulate a forward-section wall time into the global profiler.

        Only called when LYCHEEFD_VLLM_FORWARD_PROFILE=1. Stores
        {section: [total_sec, count]} on the global state for the engine to log.
        """
        store = LycheeDuplexState.fwd_profile
        if store is None:
            store = {}
            LycheeDuplexState.fwd_profile = store
        slot = store.get(name)
        if slot is None:
            store[name] = [float(dt), 1]
        else:
            slot[0] += float(dt)
            slot[1] += 1

    def _diag_control_tokens(self, logits: torch.Tensor) -> Dict[str, Optional[float]]:
        flat = logits.float()
        if flat.dim() == 2:
            flat = flat[0]
        token_map = {
            "ss": _cfg_get(self.config, "start_speaking_token_id"),
            "kl": _cfg_get(self.config, "keep_listening_token_id"),
            "bc": _cfg_get(self.config, "start_bc_token_id"),
            "sl": _cfg_get(self.config, "start_listening_token_id"),
            "ks": _cfg_get(self.config, "keep_speaking_token_id"),
            "sleep": _cfg_get(self.config, "sleep_token_id"),
            "detect": _cfg_get(self.config, "detect_token_id"),
        }
        out = {}
        vocab_size = int(flat.shape[-1])
        for name, token_id in token_map.items():
            if token_id is None:
                out[name] = None
                continue
            tid = int(token_id)
            if 0 <= tid < vocab_size:
                out[name] = float(flat[tid].item())
            else:
                out[name] = None
        top_k = min(5, vocab_size)
        top_vals, top_idx = torch.topk(flat, k=top_k)
        out["top_ids"] = [int(i.item()) for i in top_idx]
        out["top_vals"] = [float(v.item()) for v in top_vals]
        return out

    @staticmethod
    def _flatten_positions(positions, device):
        if positions is None:
            return None
        if positions.dim() == 0:
            pos = positions.view(1)
        else:
            pos = positions.reshape(-1)
        return pos.to(device=device, dtype=torch.long)

    @staticmethod
    def _should_apply_last_token_logic(positions, full_input_ids):
        if positions is None or full_input_ids is None:
            return True
        if full_input_ids.dim() == 0:
            return True
        target_pos = int(full_input_ids.shape[-1] - 1)
        flat_pos = positions.reshape(-1).to(dtype=torch.long, device=positions.device)
        return bool((flat_pos == target_pos).any().item())

    @staticmethod
    def _is_current_stream_capturing() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _project_sequence_embeds(self, seq_embeds, positions, hidden_states):
        seq_flat = seq_embeds.reshape(-1, seq_embeds.shape[-1])
        flat_pos = self._flatten_positions(positions, device=seq_flat.device)
        if flat_pos is None:
            return None

        selected = torch.zeros(
            (flat_pos.shape[0], seq_flat.shape[-1]),
            dtype=seq_flat.dtype,
            device=seq_flat.device,
        )
        valid_pos = (flat_pos >= 0) & (flat_pos < seq_flat.shape[0])
        if valid_pos.any():
            selected[valid_pos] = seq_flat.index_select(0, flat_pos[valid_pos])

        if hidden_states.dim() == 3:
            bsz, seq_len, hidden = hidden_states.shape
            return selected.view(bsz, seq_len, hidden)
        if hidden_states.dim() == 2:
            return selected
        raise ValueError(f"Unsupported hidden shape for sequence projection: {tuple(hidden_states.shape)}")

    def _build_channel_embeds_for_positions(
        self, channel_ids, positions, hidden_states, channel_type: str = "main"
    ):
        if channel_ids is None:
            return None
        if channel_ids.dim() == 1:
            channel_ids = channel_ids.unsqueeze(0)

        channel_embeds = self._get_channel_embed_module(channel_type)(channel_ids)
        # Side-channel position projection should depend on side-channel sequence
        # shape only. In incremental decode, stoken/control are usually length-1;
        # forcing projection due audio presence can zero out side embeds when
        # absolute positions exceed (prefix_len + 1).
        needs_prefix_projection = channel_ids.shape[-1] > 1
        if needs_prefix_projection:
            prefix_len = int(LycheeDuplexState.prefix_input_len or 0)
            if prefix_len > 0:
                prefix_zeros = torch.zeros(
                    (channel_embeds.shape[0], prefix_len, channel_embeds.shape[-1]),
                    dtype=channel_embeds.dtype,
                    device=channel_embeds.device,
                )
                channel_embeds = torch.cat([prefix_zeros, channel_embeds], dim=1)
            return self._project_sequence_embeds(channel_embeds, positions, hidden_states)

        return self._align_and_add(torch.zeros_like(hidden_states), channel_embeds)

    def _build_audio_embeds_for_positions(self, positions, hidden_states):
        audio_embeds = LycheeDuplexState.audio_embeds
        if audio_embeds is None:
            return None

        # Fallback: if no audio token layout is available, keep legacy align-add behavior.
        audio_input_ids = LycheeDuplexState.audio_input_ids
        if audio_input_ids is None:
            return self._align_and_add(torch.zeros_like(hidden_states), audio_embeds)

        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)
        if audio_input_ids.dim() == 1:
            audio_input_ids = audio_input_ids.unsqueeze(0)

        feat_lens = LycheeDuplexState.audio_feat_lens
        if feat_lens is None:
            feat_lens = torch.full(
                (audio_embeds.shape[0],),
                audio_embeds.shape[1],
                dtype=torch.long,
                device=audio_embeds.device,
            )
        else:
            feat_lens = feat_lens.to(device=audio_embeds.device, dtype=torch.long)
            feat_lens = torch.clamp(feat_lens, min=0, max=audio_embeds.shape[1])

        audio_inputs_embeds = self.embed_tokens(audio_input_ids)
        valid_mask = torch.arange(audio_embeds.shape[1], device=audio_embeds.device)[None, :] < feat_lens[:, None]
        audio_features = audio_embeds[valid_mask]

        patch_token_id = LycheeDuplexState.audio_patch_token_id
        if patch_token_id is None:
            patch_token_id = self.audio_patch_token_id
        patch_token_id = int(patch_token_id)

        token_mask = (audio_input_ids == patch_token_id)
        patch_token_count = int(token_mask.sum().item())
        feature_token_count = int(audio_features.shape[0])
        if feature_token_count > patch_token_count:
            audio_features = audio_features[:patch_token_count]
        elif feature_token_count < patch_token_count:
            if feature_token_count == 0:
                audio_features = torch.zeros(
                    (patch_token_count, audio_inputs_embeds.shape[-1]),
                    dtype=audio_inputs_embeds.dtype,
                    device=audio_inputs_embeds.device,
                )
            else:
                pad = torch.zeros(
                    (patch_token_count - feature_token_count, audio_features.shape[-1]),
                    dtype=audio_features.dtype,
                    device=audio_features.device,
                )
                audio_features = torch.cat([audio_features, pad], dim=0)

        if patch_token_count > 0:
            expanded_mask = token_mask.unsqueeze(-1).expand_as(audio_inputs_embeds)
            audio_features = audio_features.to(audio_inputs_embeds.device, audio_inputs_embeds.dtype)
            audio_inputs_embeds = audio_inputs_embeds.masked_scatter(expanded_mask, audio_features)

        prefix_len = int(LycheeDuplexState.prefix_input_len or 0)
        if prefix_len > 0:
            prefix_zeros = torch.zeros(
                (audio_inputs_embeds.shape[0], prefix_len, audio_inputs_embeds.shape[-1]),
                dtype=audio_inputs_embeds.dtype,
                device=audio_inputs_embeds.device,
            )
            audio_inputs_embeds = torch.cat([prefix_zeros, audio_inputs_embeds], dim=1)

        return self._project_sequence_embeds(audio_inputs_embeds, positions, hidden_states)

    def _build_merge_text_embeds_for_positions(
        self,
        positions,
        hidden_states: torch.Tensor,
        out_text: Optional[int],
    ):
        text_ids = LycheeDuplexState.text_input_ids
        if text_ids is None:
            return None
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        flat_pos = self._flatten_positions(positions, device=hidden_states.device)
        if flat_pos is None:
            return None

        flat_text = text_ids.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
        if flat_text.numel() == 0:
            return None

        max_idx = int(flat_text.shape[0] - 1)
        next_pos = flat_pos + 1
        safe_next_pos = torch.clamp(next_pos, min=0, max=max_idx)
        merge_text_ids = flat_text.index_select(0, safe_next_pos)
        needs_sampled_text = next_pos > max_idx
        if bool(needs_sampled_text.any().item()) and out_text is not None:
            sampled = torch.full_like(merge_text_ids, int(out_text))
            merge_text_ids = torch.where(needs_sampled_text, sampled, merge_text_ids)

        merge_text_embeds = self.embed_tokens(merge_text_ids)
        if hidden_states.dim() == 3:
            bsz, seq_len, hidden = hidden_states.shape
            return merge_text_embeds.view(bsz, seq_len, hidden)
        if hidden_states.dim() == 2:
            return merge_text_embeds
        raise ValueError(f"Unsupported hidden shape for merge text embeds: {tuple(hidden_states.shape)}")

    @staticmethod
    def _align_and_add(base_states, new_embeds):
        if new_embeds is None:
            return base_states

        base_dim = base_states.dim()
        if base_dim == 3:
            bsz, seq_len, hidden = base_states.shape
            base_flat = base_states.reshape(-1, hidden)
        elif base_dim == 2:
            seq_len, hidden = base_states.shape
            bsz = 1
            base_flat = base_states
        else:
            raise ValueError(f"Unsupported base hidden shape: {tuple(base_states.shape)}")

        if new_embeds.dim() == 3:
            new_flat = new_embeds.reshape(-1, new_embeds.shape[-1])
        elif new_embeds.dim() == 2:
            new_flat = new_embeds
        else:
            raise ValueError(f"Unsupported new embed shape: {tuple(new_embeds.shape)}")

        target_len = base_flat.shape[0]
        curr_len = new_flat.shape[0]
        if curr_len < target_len:
            pad_len = target_len - curr_len
            zero_pad = torch.zeros(
                (pad_len, new_flat.shape[-1]),
                dtype=new_flat.dtype,
                device=new_flat.device,
            )
            new_flat = torch.cat([zero_pad, new_flat], dim=0)
        elif curr_len > target_len:
            new_flat = new_flat[-target_len:, :]

        out_flat = base_flat + new_flat
        if base_dim == 3:
            return out_flat.view(bsz, seq_len, hidden)
        return out_flat

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: Optional["IntermediateTensors"] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del intermediate_tensors, kwargs
        diag_enabled = bool(LycheeDuplexState.diag_enabled)
        if diag_enabled:
            LycheeDuplexState.diag_forward_calls += 1

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        if LycheeDuplexState.stoken_ids is not None:
            stoken_step_embeds = self._build_channel_embeds_for_positions(
                LycheeDuplexState.stoken_ids,
                positions,
                hidden_states,
                channel_type="stoken",
            )
            if stoken_step_embeds is not None:
                hidden_states = hidden_states + stoken_step_embeds
        if LycheeDuplexState.control_ids is not None:
            control_step_embeds = self._build_channel_embeds_for_positions(
                LycheeDuplexState.control_ids,
                positions,
                hidden_states,
                channel_type="control",
            )
            if control_step_embeds is not None:
                hidden_states = hidden_states + control_step_embeds
        if LycheeDuplexState.audio_embeds is not None:
            audio_step_embeds = self._build_audio_embeds_for_positions(positions, hidden_states)
            if audio_step_embeds is not None:
                hidden_states = hidden_states + audio_step_embeds

        n_main = self.num_main_layers
        n_stoken = self.num_stoken_layers
        n_control = self.num_control_layers
        n_merge = self.num_merge_layers
        expected_cache_layers = self.total_num_layers
        cache_layers = len(kv_caches) if kv_caches is not None else 0
        if cache_layers < expected_cache_layers:
            raise RuntimeError(
                f"KV cache layer count mismatch: expected {expected_cache_layers}, got {cache_layers}. "
                "This usually means vLLM selected a pooling runner/task (embed/reward/classify/score) "
                "instead of generation. Force task='generate' for Lychee-FD duplex."
            )

        # HF uses outputs.hidden_states[idx], where hidden_states[0] is embeddings
        # and hidden_states[i+1] is after layer i. So branch index `idx` means:
        # - idx == 0: branch from embeddings (before any main layer)
        # - idx > 0: branch after layer (idx - 1)
        stoken_branch_input = hidden_states.clone() if (n_stoken > 0 and self.stoken_branch_idx == 0) else None
        control_branch_input = hidden_states.clone() if (n_control > 0 and self.control_branch_idx == 0) else None
        main_bad_layers: List[Dict[str, int]] = []
        _prof = self._forward_profile and hidden_states.is_cuda
        if _prof:
            torch.cuda.synchronize()
            _t_main = time.perf_counter()
        for i, layer in enumerate(self.main_layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i], attn_metadata)
            if self._forward_finite_check and not torch.isfinite(hidden_states).all():
                counts = self._count_non_finite(hidden_states)
                main_bad_layers.append(
                    {"layer": int(i), "nan": counts["nan"], "inf": counts["inf"]}
                )
                hidden_states = self._sanitize_tensor_finite(hidden_states)
            if n_stoken > 0 and (i + 1) == self.stoken_branch_idx:
                stoken_branch_input = hidden_states.clone()
            if n_control > 0 and (i + 1) == self.control_branch_idx:
                control_branch_input = hidden_states.clone()

        hidden_states = self.main_norm(hidden_states)
        if self._forward_finite_check and not torch.isfinite(hidden_states).all():
            counts = self._count_non_finite(hidden_states)
            main_bad_layers.append(
                {"layer": int(n_main), "nan": counts["nan"], "inf": counts["inf"]}
            )
            hidden_states = self._sanitize_tensor_finite(hidden_states)
        if _prof:
            torch.cuda.synchronize()
            self._prof_accum("main_layers", time.perf_counter() - _t_main)
        if diag_enabled:
            main_non_finite = self._count_non_finite(hidden_states)
            main_hidden_stats = self._diag_tensor_stats(hidden_states)
        else:
            main_non_finite = None
            main_hidden_stats = None

        stoken_bad_layers: List[Dict[str, int]] = []
        stoken_pre_norm_stats = None
        control_bad_layers: List[Dict[str, int]] = []
        control_pre_norm_stats = None
        run_stoken = bool(n_stoken > 0 and stoken_branch_input is not None)
        run_control = bool(n_control > 0 and control_branch_input is not None)
        use_parallel_side = (
            run_stoken
            and run_control
            and hidden_states.is_cuda
            and not self._is_current_stream_capturing()
            and self._ensure_side_parallel_streams(hidden_states.device)
        )
        stoken_hidden = hidden_states
        control_hidden = hidden_states
        stoken_result = None
        if use_parallel_side:
            stoken_result = {}
            control_result: Dict[
                str, Tuple[torch.Tensor, List[Dict[str, int]], Optional[Dict[str, float]]]
            ] = {}
            current_stream = torch.cuda.current_stream(device=hidden_states.device)
            with torch.cuda.stream(self._stoken_branch_stream):
                stoken_result["out"] = self._run_side_branch(
                    branch_input=stoken_branch_input,
                    layers=self.stoken_layers,
                    norm=self.stoken_norm,
                    kv_caches=kv_caches[n_main:n_main + n_stoken],
                    positions=positions,
                    attn_metadata=attn_metadata,
                    diag_enabled=diag_enabled,
                )
            with torch.cuda.stream(self._control_branch_stream):
                control_result["out"] = self._run_side_branch(
                    branch_input=control_branch_input,
                    layers=self.control_layers,
                    norm=self.control_norm,
                    kv_caches=kv_caches[n_main + n_stoken:n_main + n_stoken + n_control],
                    positions=positions,
                    attn_metadata=attn_metadata,
                    diag_enabled=diag_enabled,
                )
            current_stream.wait_stream(self._control_branch_stream)
            control_hidden, control_bad_layers, control_pre_norm_stats = control_result["out"]
        else:
            if run_control:
                control_hidden, control_bad_layers, control_pre_norm_stats = self._run_side_branch(
                    branch_input=control_branch_input,
                    layers=self.control_layers,
                    norm=self.control_norm,
                    kv_caches=kv_caches[n_main + n_stoken:n_main + n_stoken + n_control],
                    positions=positions,
                    attn_metadata=attn_metadata,
                    diag_enabled=diag_enabled,
                )
            else:
                control_hidden = hidden_states

        # If a side branch collapses to an all-zero tensor, fall back to the
        # main hidden state to keep duplex generation functional.
        branch_fallbacks: List[str] = []
        if n_control > 0 and control_hidden is not None:
            try:
                if bool(torch.max(torch.abs(control_hidden)).item() == 0.0):
                    control_hidden = hidden_states
                    branch_fallbacks.append("control_from_main")
            except Exception:
                pass

        control_hidden_before = control_hidden
        control_hidden = self._sanitize_tensor_finite(control_hidden)

        target_pos = None
        if LycheeDuplexState.text_input_ids is not None:
            target_pos = int(LycheeDuplexState.text_input_ids.shape[-1] - 1)
        apply_last_token_logic = self._should_apply_last_token_logic(
            positions,
            LycheeDuplexState.text_input_ids,
        )
        is_capturing = self._is_current_stream_capturing()

        control_last = None
        control_last_before = None
        control_proc_sec = 0.0
        control_sample_sec = 0.0
        control_proj_sec = 0.0
        if apply_last_token_logic and not is_capturing:
            control_proj_t0 = time.perf_counter()
            control_last = self._project_side_last_logits(
                control_hidden,
                branch="control",
                processors=LycheeDuplexState.control_processors,
            )
            control_proj_sec = time.perf_counter() - control_proj_t0
            if diag_enabled:
                control_last_before = control_last.detach().float().clone()
            control_proc_t0 = time.perf_counter()
            control_last = self._apply_processors(
                control_last,
                LycheeDuplexState.control_input_ids,
                LycheeDuplexState.control_processors,
            )
            control_proc_sec = time.perf_counter() - control_proc_t0
            control_cfg = LycheeDuplexState.control_sampling or {}
            control_sample_t0 = time.perf_counter()
            LycheeDuplexState.out_control = self._sample_token(
                control_last,
                do_sample=bool(control_cfg.get("do_sample", False)),
                temperature=float(control_cfg.get("temperature", 1.0)),
                top_k=int(control_cfg.get("top_k", 0)),
                top_p=float(control_cfg.get("top_p", 1.0)),
                nan_guard=self._sample_nan_guard,
            )
            control_sample_sec = time.perf_counter() - control_sample_t0
            self._emit_control_early_callback(
                int(LycheeDuplexState.out_control),
                positions,
            )

        if use_parallel_side:
            current_stream.wait_stream(self._stoken_branch_stream)
            stoken_hidden, stoken_bad_layers, stoken_pre_norm_stats = stoken_result["out"]
        elif run_stoken:
            stoken_hidden, stoken_bad_layers, stoken_pre_norm_stats = self._run_side_branch(
                branch_input=stoken_branch_input,
                layers=self.stoken_layers,
                norm=self.stoken_norm,
                kv_caches=kv_caches[n_main:n_main + n_stoken],
                positions=positions,
                attn_metadata=attn_metadata,
                diag_enabled=diag_enabled,
            )
        else:
            stoken_hidden = hidden_states

        if n_stoken > 0 and stoken_hidden is not None:
            try:
                if bool(torch.max(torch.abs(stoken_hidden)).item() == 0.0):
                    stoken_hidden = hidden_states
                    branch_fallbacks.append("stoken_from_main")
            except Exception:
                pass

        stoken_hidden_before = stoken_hidden
        stoken_hidden = self._sanitize_tensor_finite(stoken_hidden)

        merge_bad_layers: List[Dict[str, int]] = []
        merge_pre_norm_stats = None
        merge_hidden_before = None
        merge_hidden_after = None
        LycheeDuplexState.out_text = None
        if n_merge > 0 and apply_last_token_logic and not is_capturing:
            LycheeDuplexState.out_text = self._sample_text_token_for_merge(
                hidden_states
            )

        if n_merge > 0 and not is_capturing:
            merge_text_embeds = self._build_merge_text_embeds_for_positions(
                positions,
                stoken_hidden,
                LycheeDuplexState.out_text,
            )
            if merge_text_embeds is not None:
                merge_inputs = stoken_hidden + merge_text_embeds
                merge_hidden, merge_bad_layers, merge_pre_norm_stats = self._run_side_branch(
                    branch_input=merge_inputs,
                    layers=self.merge_layers,
                    norm=self.merge_norm,
                    kv_caches=kv_caches[
                        n_main + n_stoken + n_control:
                        n_main + n_stoken + n_control + n_merge
                    ],
                    positions=positions,
                    attn_metadata=attn_metadata,
                    diag_enabled=diag_enabled,
                )
                merge_hidden_before = merge_hidden
                if self._forward_finite_check and not torch.isfinite(merge_hidden).all():
                    counts = self._count_non_finite(merge_hidden)
                    merge_bad_layers.append(
                        {"layer": int(n_merge), "nan": counts["nan"], "inf": counts["inf"]}
                    )
                merge_hidden = self._sanitize_tensor_finite(merge_hidden)
                merge_hidden_after = merge_hidden
                stoken_hidden = merge_hidden
            else:
                branch_fallbacks.append("merge_missing_text")

        if not apply_last_token_logic:
            if diag_enabled:
                pos_flat = positions.reshape(-1).to(dtype=torch.long)
                LycheeDuplexState.diag_last = {
                    "request_id": LycheeDuplexState.diag_request_id,
                    "forward_calls": int(LycheeDuplexState.diag_forward_calls),
                    "reason": "skip_last_token_logic",
                    "positions_min": int(pos_flat.min().item()) if pos_flat.numel() > 0 else None,
                    "positions_max": int(pos_flat.max().item()) if pos_flat.numel() > 0 else None,
                    "positions_len": int(pos_flat.numel()),
                    "target_pos": target_pos,
                }
            return hidden_states

        # This model performs Python-side side-channel sampling for stoken/control.
        # CUDA graph capture cannot safely execute this path.
        if is_capturing:
            LycheeDuplexState.out_text = None
            LycheeDuplexState.out_stoken = None
            LycheeDuplexState.out_control = None
            if diag_enabled:
                pos_flat = positions.reshape(-1).to(dtype=torch.long)
                LycheeDuplexState.diag_last = {
                    "request_id": LycheeDuplexState.diag_request_id,
                    "forward_calls": int(LycheeDuplexState.diag_forward_calls),
                    "reason": "cuda_stream_capturing",
                    "positions_min": int(pos_flat.min().item()) if pos_flat.numel() > 0 else None,
                    "positions_max": int(pos_flat.max().item()) if pos_flat.numel() > 0 else None,
                    "positions_len": int(pos_flat.numel()),
                    "target_pos": target_pos,
                }
            return hidden_states

        side_t0 = time.perf_counter()
        side_proj_t0 = time.perf_counter()
        stoken_last = self._project_side_last_logits(
            stoken_hidden,
            branch="stoken",
            processors=LycheeDuplexState.stoken_processors,
        )
        side_proj_sec = (time.perf_counter() - side_proj_t0) + float(control_proj_sec)
        if diag_enabled:
            stoken_last_before = stoken_last.detach().float().clone()
        else:
            stoken_last_before = None
        side_proc_t0 = time.perf_counter()
        stoken_proc_t0 = time.perf_counter()
        stoken_last = self._apply_processors(
            stoken_last,
            LycheeDuplexState.stoken_input_ids,
            LycheeDuplexState.stoken_processors,
        )
        stoken_proc_sec = time.perf_counter() - stoken_proc_t0
        side_proc_sec = (time.perf_counter() - side_proc_t0) + float(control_proc_sec)

        stoken_cfg = LycheeDuplexState.stoken_sampling or {}

        side_sample_t0 = time.perf_counter()
        LycheeDuplexState.out_stoken = self._sample_token(
            stoken_last,
            do_sample=bool(stoken_cfg.get("do_sample", True)),
            temperature=float(stoken_cfg.get("temperature", 0.7)),
            top_k=int(stoken_cfg.get("top_k", 0)),
            top_p=float(stoken_cfg.get("top_p", 1.0)),
            nan_guard=self._sample_nan_guard,
        )
        side_sample_sec = (time.perf_counter() - side_sample_t0) + float(control_sample_sec)
        side_total_sec = (
            time.perf_counter() - side_t0
            + float(control_proj_sec)
            + float(control_proc_sec)
            + float(control_sample_sec)
        )
        if self._side_profile_enabled:
            self._side_profile_calls += 1
            self._side_profile_total_sec += side_total_sec
            self._side_profile_proj_sec += side_proj_sec
            self._side_profile_proc_sec += side_proc_sec
            self._side_profile_proc_stoken_sec += stoken_proc_sec
            self._side_profile_proc_control_sec += control_proc_sec
            self._side_profile_sample_sec += side_sample_sec
            if (
                self._side_profile_calls == 1
                or (self._side_profile_calls % self._side_profile_every == 0)
            ):
                avg_total = self._side_profile_total_sec / max(1, self._side_profile_calls)
                avg_proj = self._side_profile_proj_sec / max(1, self._side_profile_calls)
                avg_proc = self._side_profile_proc_sec / max(1, self._side_profile_calls)
                avg_proc_stoken = self._side_profile_proc_stoken_sec / max(1, self._side_profile_calls)
                avg_proc_control = self._side_profile_proc_control_sec / max(1, self._side_profile_calls)
                avg_sample = self._side_profile_sample_sec / max(1, self._side_profile_calls)
                logger.info(
                    "[LAT_SIDE] request=%s fwd=%s call=%d total=%.4fms proj=%.4fms proc=%.4fms "
                    "(stoken=%.4fms control=%.4fms) sample=%.4fms "
                    "avg_total=%.4fms avg_proj=%.4fms avg_proc=%.4fms "
                    "(avg_stoken=%.4fms avg_control=%.4fms) avg_sample=%.4fms",
                    LycheeDuplexState.diag_request_id,
                    LycheeDuplexState.diag_forward_calls,
                    int(self._side_profile_calls),
                    side_total_sec * 1000.0,
                    side_proj_sec * 1000.0,
                    side_proc_sec * 1000.0,
                    stoken_proc_sec * 1000.0,
                    control_proc_sec * 1000.0,
                    side_sample_sec * 1000.0,
                    avg_total * 1000.0,
                    avg_proj * 1000.0,
                    avg_proc * 1000.0,
                    avg_proc_stoken * 1000.0,
                    avg_proc_control * 1000.0,
                    avg_sample * 1000.0,
                )
        if diag_enabled:
            pos_flat = positions.reshape(-1).to(dtype=torch.long)
            control_diag = self._diag_control_tokens(control_last)
            LycheeDuplexState.diag_last = {
                "request_id": LycheeDuplexState.diag_request_id,
                "forward_calls": int(LycheeDuplexState.diag_forward_calls),
                "reason": "sampled",
                "positions_min": int(pos_flat.min().item()) if pos_flat.numel() > 0 else None,
                "positions_max": int(pos_flat.max().item()) if pos_flat.numel() > 0 else None,
                "positions_len": int(pos_flat.numel()),
                "target_pos": target_pos,
                "stoken_before": self._diag_tensor_stats(stoken_last_before),
                "stoken_after": self._diag_tensor_stats(stoken_last),
                "control_before": self._diag_tensor_stats(control_last_before),
                "control_after": self._diag_tensor_stats(control_last),
                "stoken_hidden_before": self._diag_tensor_stats(stoken_hidden_before),
                "stoken_hidden_after": self._diag_tensor_stats(stoken_hidden),
                "control_hidden_before": self._diag_tensor_stats(control_hidden_before),
                "control_hidden_after": self._diag_tensor_stats(control_hidden),
                "merge_hidden_before": (
                    self._diag_tensor_stats(merge_hidden_before)
                    if merge_hidden_before is not None else None
                ),
                "merge_hidden_after": (
                    self._diag_tensor_stats(merge_hidden_after)
                    if merge_hidden_after is not None else None
                ),
                "main_non_finite": main_non_finite,
                "main_hidden": main_hidden_stats,
                "main_bad_layers": main_bad_layers,
                "stoken_bad_layers": stoken_bad_layers,
                "control_bad_layers": control_bad_layers,
                "merge_bad_layers": merge_bad_layers,
                "stoken_pre_norm": stoken_pre_norm_stats,
                "control_pre_norm": control_pre_norm_stats,
                "merge_pre_norm": merge_pre_norm_stats,
                "branch_fallbacks": branch_fallbacks,
                "side_subset_enabled": bool(self._side_subset_enabled),
                "side_parallel_enabled": bool(use_parallel_side),
                "control_tokens": control_diag,
                "side_timing_ms": {
                    "total": float(side_total_sec * 1000.0),
                    "proj": float(side_proj_sec * 1000.0),
                    "proc": float(side_proc_sec * 1000.0),
                    "proc_stoken": float(stoken_proc_sec * 1000.0),
                    "proc_control": float(control_proc_sec * 1000.0),
                    "sample": float(side_sample_sec * 1000.0),
                },
                "sampled_text": (
                    int(LycheeDuplexState.out_text)
                    if LycheeDuplexState.out_text is not None else None
                ),
                "sampled_stoken": int(LycheeDuplexState.out_stoken),
                "sampled_control": int(LycheeDuplexState.out_control),
            }

        # vLLM generation runner expects hidden states here and calls
        # `compute_logits()` afterwards.
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None

        # Compatibility: some legacy paths may already return logits from forward.
        if hidden_states.shape[-1] == self.vocab_size:
            logits = hidden_states
        else:
            if self.logits_processor is not None:
                try:
                    logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
                except Exception:
                    logits = self._project_logits(hidden_states)
            else:
                logits = self._project_logits(hidden_states)

        text_ids = LycheeDuplexState.text_input_ids
        text_processors = LycheeDuplexState.text_processors
        if text_ids is not None and text_processors is not None:
            text_last = logits[-1:, :] if logits.dim() == 2 else logits[:, -1, :]
            text_last = self._apply_processors(text_last, text_ids, text_processors)
            logits = self._set_last_logits(logits, text_last)
        text_pad_id = LycheeDuplexState.text_after_eos_pad_token_id
        text_eos_id = LycheeDuplexState.text_after_eos_token_id
        force_text_pad = bool(LycheeDuplexState.text_after_eos_seen)
        if (
            not force_text_pad
            and text_ids is not None
            and text_eos_id is not None
        ):
            try:
                force_text_pad = bool(
                    (text_ids.reshape(-1)[-1] == int(text_eos_id)).item()
                )
            except Exception:
                force_text_pad = False
        if force_text_pad and text_pad_id is not None:
            text_last = logits[-1:, :] if logits.dim() == 2 else logits[:, -1, :]
            pad_id = int(text_pad_id)
            if 0 <= pad_id < int(text_last.shape[-1]):
                text_last.fill_(-1e4)
                text_last[:, pad_id] = 0.0
                logits = self._set_last_logits(logits, text_last)
                LycheeDuplexState.text_after_eos_seen = True
        out_text = LycheeDuplexState.out_text
        if out_text is not None:
            text_last = logits[-1:, :] if logits.dim() == 2 else logits[:, -1, :]
            forced_last = self._force_single_token_logits(text_last, int(out_text))
            logits = self._set_last_logits(logits, forced_last)
        if self._logits_nan_guard:
            return torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=float("-inf"))
        return logits

    def sample(self, logits: torch.Tensor, sampling_metadata):
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        side_lm_head_loaded = False
        total_considered = 0
        loaded_params = 0
        skipped_aux = 0
        unmatched = 0
        unmatched_examples: List[str] = []
        branch_loaded = {
            "main": 0,
            "stoken": 0,
            "control": 0,
            "merge": 0,
            "lm_head": 0,
            "embed_norm": 0,
        }
        for name, loaded_weight in weights:
            original_name = name
            remap_prefixes = [
                ("model.model.model.layers.", "main_layers."),
                ("model.model.layers.", "main_layers."),
                ("model.layers.", "main_layers."),
                ("model.stoken_model.model.layers.", "stoken_layers."),
                ("model.stoken_model.layers.", "stoken_layers."),
                ("stoken_model.model.layers.", "stoken_layers."),
                ("stoken_model.layers.", "stoken_layers."),
                ("model.control_model.model.layers.", "control_layers."),
                ("model.control_model.layers.", "control_layers."),
                ("control_model.model.layers.", "control_layers."),
                ("control_model.layers.", "control_layers."),
                ("model.merge_model.model.layers.", "merge_layers."),
                ("model.merge_model.layers.", "merge_layers."),
                ("merge_model.model.layers.", "merge_layers."),
                ("merge_model.layers.", "merge_layers."),
                ("model.model.model.embed_tokens.", "embed_tokens."),
                ("model.model.embed_tokens.", "embed_tokens."),
                ("model.embed_tokens.", "embed_tokens."),
                ("embed_tokens.", "embed_tokens."),
                ("model.stoken_model.model.embed_tokens.", "stoken_embed_tokens."),
                ("model.stoken_model.embed_tokens.", "stoken_embed_tokens."),
                ("stoken_model.model.embed_tokens.", "stoken_embed_tokens."),
                ("stoken_model.embed_tokens.", "stoken_embed_tokens."),
                ("model.control_model.model.embed_tokens.", "control_embed_tokens."),
                ("model.control_model.embed_tokens.", "control_embed_tokens."),
                ("control_model.model.embed_tokens.", "control_embed_tokens."),
                ("control_model.embed_tokens.", "control_embed_tokens."),
                ("model.model.model.norm.", "main_norm."),
                ("model.model.norm.", "main_norm."),
                ("model.norm.", "main_norm."),
                ("norm.", "main_norm."),
                ("model.stoken_model.model.norm.", "stoken_norm."),
                ("model.stoken_model.norm.", "stoken_norm."),
                ("stoken_model.model.norm.", "stoken_norm."),
                ("stoken_model.norm.", "stoken_norm."),
                ("model.control_model.model.norm.", "control_norm."),
                ("model.control_model.norm.", "control_norm."),
                ("control_model.model.norm.", "control_norm."),
                ("control_model.norm.", "control_norm."),
                ("model.merge_model.model.norm.", "merge_norm."),
                ("model.merge_model.norm.", "merge_norm."),
                ("merge_model.model.norm.", "merge_norm."),
                ("merge_model.norm.", "merge_norm."),
            ]
            for src, dst in remap_prefixes:
                if name.startswith(src):
                    name = name.replace(src, dst, 1)
                    break

            if name.startswith("model.lm_head."):
                side_name = name.replace("model.lm_head.", "side_lm_head.", 1)
                if side_name in params_dict:
                    side_param = params_dict[side_name]
                    side_loader = getattr(side_param, "weight_loader", default_weight_loader)
                    side_loader(side_param, loaded_weight)
                    side_lm_head_loaded = True
                    loaded_params += 1
                    branch_loaded["lm_head"] += 1
                name = name.replace("model.lm_head.", "lm_head.", 1)
            elif name.startswith("lm_head."):
                side_name = name.replace("lm_head.", "side_lm_head.", 1)
                if side_name in params_dict:
                    side_param = params_dict[side_name]
                    side_loader = getattr(side_param, "weight_loader", default_weight_loader)
                    side_loader(side_param, loaded_weight)
                    side_lm_head_loaded = True
                    loaded_params += 1
                    branch_loaded["lm_head"] += 1
            elif (
                name.startswith("model.encoder.")
                or name.startswith("encoder.")
                or name.startswith("model.adapter.")
                or name.startswith("adapter.")
                or name.startswith("model.merge_model.model.embed_tokens.")
                or name.startswith("model.merge_model.embed_tokens.")
                or name.startswith("merge_model.model.embed_tokens.")
                or name.startswith("merge_model.embed_tokens.")
            ):
                skipped_aux += 1
                continue

            total_considered += 1
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name in params_dict:
                    param = params_dict[stacked_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    is_stacked = True
                    loaded_params += 1
                    if stacked_name.startswith("main_layers."):
                        branch_loaded["main"] += 1
                    elif stacked_name.startswith("stoken_layers."):
                        branch_loaded["stoken"] += 1
                    elif stacked_name.startswith("control_layers."):
                        branch_loaded["control"] += 1
                    elif stacked_name.startswith("merge_layers."):
                        branch_loaded["merge"] += 1
                    elif stacked_name.startswith(("lm_head.", "side_lm_head.")):
                        branch_loaded["lm_head"] += 1
                    else:
                        branch_loaded["embed_norm"] += 1
                    break

            if not is_stacked and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params += 1
                if name.startswith("main_layers."):
                    branch_loaded["main"] += 1
                elif name.startswith("stoken_layers."):
                    branch_loaded["stoken"] += 1
                elif name.startswith("control_layers."):
                    branch_loaded["control"] += 1
                elif name.startswith("merge_layers."):
                    branch_loaded["merge"] += 1
                elif name.startswith(("lm_head.", "side_lm_head.")):
                    branch_loaded["lm_head"] += 1
                else:
                    branch_loaded["embed_norm"] += 1
            elif not is_stacked:
                unmatched += 1
                if len(unmatched_examples) < 20:
                    unmatched_examples.append(original_name)

        # Safety fallback: if side lm_head did not match checkpoint key patterns,
        # mirror lm_head weights so stoken/control branch projection stays aligned.
        if not side_lm_head_loaded:
            side_key = "side_lm_head.weight"
            main_key = "lm_head.weight"
            if side_key in params_dict and main_key in params_dict:
                with torch.no_grad():
                    params_dict[side_key].copy_(params_dict[main_key])
                logger.warning(
                    "side_lm_head weight not found in checkpoint keys; mirrored from lm_head."
                )

        logger.info(
            "Lychee-FD load_weights summary: considered=%d loaded=%d unmatched=%d skipped_aux=%d "
            "(main=%d stoken=%d control=%d merge=%d lm_head=%d embed_norm=%d)",
            int(total_considered),
            int(loaded_params),
            int(unmatched),
            int(skipped_aux),
            int(branch_loaded["main"]),
            int(branch_loaded["stoken"]),
            int(branch_loaded["control"]),
            int(branch_loaded["merge"]),
            int(branch_loaded["lm_head"]),
            int(branch_loaded["embed_norm"]),
        )
        missing_branches: List[str] = []
        if self.num_main_layers > 0 and branch_loaded["main"] == 0:
            missing_branches.append("main")
        if self.num_stoken_layers > 0 and branch_loaded["stoken"] == 0:
            missing_branches.append("stoken")
        if self.num_control_layers > 0 and branch_loaded["control"] == 0:
            missing_branches.append("control")
        if self.num_merge_layers > 0 and branch_loaded["merge"] == 0:
            missing_branches.append("merge")
        if missing_branches:
            raise RuntimeError(
                "Lychee-FD load_weights detected zero loaded params for branches: "
                f"{missing_branches}. This indicates checkpoint key remapping mismatch. "
                f"Unmatched={unmatched}, examples={unmatched_examples[:10]}"
            )
        if total_considered > 0 and unmatched > (total_considered // 2):
            logger.warning(
                "Lychee-FD load_weights has high unmatched ratio: unmatched=%d considered=%d",
                int(unmatched),
                int(total_considered),
            )
        if unmatched_examples:
            logger.warning(
                "Lychee-FD unmatched checkpoint keys (showing up to 20): %s",
                unmatched_examples,
            )
