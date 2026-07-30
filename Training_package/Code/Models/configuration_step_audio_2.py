from typing import Optional, Union

from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig


class StepAudio2EncoderConfig(PretrainedConfig):
    model_type = "step_audio_2_encoder"

    def __init__(
        self,
        n_mels=128,
        n_audio_ctx=1500,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=6,
        llm_dim=4096,
        kernel_size=3,
        adapter_stride=2,
        **kwargs,
    ):
        self.n_mels      = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.llm_dim     = llm_dim
        self.kernel_size = kernel_size
        self.adapter_stride = adapter_stride
        super().__init__(**kwargs)

class StepAudio2TextConfig(PretrainedConfig):
    model_type = "step_audio_2_text"

    def __init__(
        self,
        vocab_size=64012,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_attention_groups=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        eos_token_id=None,
        **kwargs
    ):

        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                eos_token_id = list(set([151643, 151645, 151665] + eos_token_id))
            else:
                eos_token_id = [151643, 151645, 151665, eos_token_id]
        else:
            eos_token_id = [151643, 151645, 151665]

        super().__init__(
            eos_token_id=eos_token_id,
            **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_key_value_heads = num_key_value_heads
        assert self.num_attention_groups == self.num_key_value_heads, f"num_attention_groups: {self.num_attention_groups} must be equal to num_key_value_heads: {self.num_key_value_heads}"
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Get torch_dtype from kwargs if provided
        torch_dtype = kwargs.get("torch_dtype", getattr(self, "torch_dtype", "bfloat16"))
        self.text_config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_attention_groups=num_attention_groups,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            architectures=["Qwen2ForCausalLM"],
            torch_dtype=torch_dtype,
        )

class StepAudio2Config(PretrainedConfig):
    model_type = "step_audio_2"
    architectures = ["StepAudio2ForCausalLM"]
    # Support alternative model types and architectures for 32B model
    # This allows the config to work with both "step_audio_2" and "step_audio_qwen2" model types

    def __init__(
        self,
        audio_encoder_config :Optional[Union[dict, StepAudio2EncoderConfig]] = None,
        text_config: Optional[Union[dict, StepAudio2TextConfig]] = None,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = 2048,
        max_window_layers: Optional[int] = None,
        **kwargs
    ):
        kwargs.setdefault("use_sliding_window", use_sliding_window)
        kwargs.setdefault("sliding_window", sliding_window)
        if max_window_layers is None:
            max_window_layers = kwargs.get("num_hidden_layers", None)
        kwargs.setdefault("max_window_layers", max_window_layers)
        # Save torch_dtype if provided (for 32B model flat config)
        if 'torch_dtype' in kwargs:
            self.torch_dtype = kwargs['torch_dtype']
        
        super().__init__(**kwargs)

        # Support for flat config structure (32B model format)
        # If text_config is None and we have flat config parameters, extract them
        if text_config is None:
            # Check if we have flat config parameters (32B model format)
            flat_text_params = {}
            text_param_names = [
                'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_attention_groups', 'num_key_value_heads',
                'hidden_act', 'max_position_embeddings', 'initializer_range',
                'rms_norm_eps', 'rope_theta', 'rope_scaling', 'eos_token_id', 'pad_token_id'
            ]
            
            for param_name in text_param_names:
                if param_name in kwargs:
                    flat_text_params[param_name] = kwargs[param_name]
            
            # Set default hidden_act if not present (32B model config doesn't have it)
            if 'hidden_act' not in flat_text_params:
                flat_text_params['hidden_act'] = 'silu'
            
            # Set default initializer_range if not present
            if 'initializer_range' not in flat_text_params:
                flat_text_params['initializer_range'] = 0.02
            
            # Also check for torch_dtype
            if 'torch_dtype' in kwargs:
                flat_text_params['torch_dtype'] = kwargs['torch_dtype']
            
            if flat_text_params:
                # We have flat config, use it to build text_config
                text_config = StepAudio2TextConfig(**flat_text_params).text_config
            else:
                # No flat config, use defaults
                text_config = StepAudio2TextConfig().text_config
        elif isinstance(text_config, dict):
            text_config = StepAudio2TextConfig(**text_config).text_config

        self.text_config = text_config

        if audio_encoder_config is None:
            # Check if we have flat audio_encoder_config in kwargs
            if 'audio_encoder_config' in kwargs and isinstance(kwargs['audio_encoder_config'], dict):
                self.audio_encoder_config = StepAudio2EncoderConfig(**kwargs['audio_encoder_config'])
            else:
                self.audio_encoder_config = StepAudio2EncoderConfig()
        elif isinstance(audio_encoder_config, dict):
            self.audio_encoder_config = StepAudio2EncoderConfig(**audio_encoder_config)
