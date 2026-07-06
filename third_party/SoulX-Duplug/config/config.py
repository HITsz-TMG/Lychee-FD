from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    task: str = "state_prediction"

    # vocab config
    text_vocab_size: int = 151643
    original_vocab_size: int = 151669
    lm_vocab_size: int = 151936
    tokenizer_vocab_size: int = 203566
    added_audio_token_size: int = 51866
    added_special_token_size: int = 31
    special_token_start: int = 151643
    added_token_start: int = 151669
    added_audio_token_start: int = 151700
    total_vocab_size: int = field(init=False)

    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    audio_pad_token_id: int = 151673
    asr_eos_token_id: int = 151674
    asr_bos_token_id: int = 151675

    user_complete_token_id: int = 151676
    user_backchannel_token_id: int = 151677
    user_incomplete_token_id: int = 151678
    assistant_backchannel_token_id: int = 151679
    user_idle_token_id: int = 151680
    user_nonidle_token_id: int = 151681
    assistant_interrupt_token_id: int = 151682

    # model config
    audio_embed_dim: int = 1280
    llm_dim: int = 2048
    glm_tokenizer_path: str = "pretrained_models/glm-4-voice-tokenizer"
    model_name: str = "pretrained_models/Qwen3-1.7B-expand_vocab_v2"
    init_ckpt_path: str = ""
    init_ckpt_path_lora: str = ""

    sampling_rate: int = 16000
    chunk_size: int = 960
    extract_token_batch: int = 256
    enable_audio_mask: bool = False
    asr_repetition_penalty: float = 1.0
    num_beam: int = 3
    punctuation: bool = False
    max_chunk_token_length: int = 50
    max_token_length: int = 1500
    enable_cascade_asr: bool = True

    # lora
    enable_lora: bool = False
    lora_task_type: str = "CAUSAL_LM"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # projector
    enable_projector: bool = True
    freeze_projector: bool = False
    embed_only: bool = False

    def __post_init__(self):
        self.total_vocab_size = (
            self.original_vocab_size
            + self.added_audio_token_size
            + self.added_special_token_size
        )


@dataclass
class DataConfig:
    # training set
    batch_size: int = 64
    num_workers: int = 12
    split_size: float = 0.05
    max_token_length: int = 1500
    avg_token_length: int = 150
    train_data_path: str = ""

    # dynamic batch size
    enable_dynamic_batch: bool = False
    max_batch_size: int = 512
    max_token_per_batch: int = 8192
    stream_dataset_buffer_size: int = 5000

    # test set
    test_data_name: str = ""
    test_lang: str = ""
    test_result_dir: str = ""
    test_result_file: str = ""
    test_data_path: str = ""
    input_wav_path: str = ""


@dataclass
class TrainConfig:
    # optimizer
    linear_lr: bool = False
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 5000
    anneal_steps: int = 400000
    anneal_rate: float = 0.5
    weight_decay: float = 1e-2
    betas: List[float] = field(init=False)
    eps: float = 1e-8

    enable_switch_loss_rate: bool = False
    switch_loss_rate_label: str = ""

    text_loss_rate: float = 1.0
    eos_loss_rate: float = 1.0
    idle_loss_rate: float = 1.0
    nonidle_loss_rate: float = 1.0
    user_complete_loss_rate: float = 1.0
    user_incomplete_loss_rate: float = 1.0
    user_backchannel_loss_rate: float = 1.0

    text_loss_rate_switched: float = 1.0
    eos_loss_rate_switched: float = 1.0
    idle_loss_rate_switched: float = 1.0
    nonidle_loss_rate_switched: float = 1.0
    user_complete_loss_rate_switched: float = 1.0
    user_incomplete_loss_rate_switched: float = 1.0
    user_backchannel_loss_rate_switched: float = 1.0

    # tricks
    # EMA
    enable_ema: bool = False
    ema_dacay: float = 0.9
    ema_every_n_steps: int = 1
    ema_start_step: int = 0

    # train adapter first
    adapter_first: bool = False
    unfreeze_step: int = 10000

    # trainer
    seed: int = 42
    stage: str = "train"
    total_steps: int = 1000000
    total_epochs: int = 0
    val_check_interval: int = 10000
    log_every_n_steps: int = 100
    accumulate_grad_batches: int = 1
    num_gpu_per_node: int = 8
    num_node: int = 1
    accelerator: str = "gpu"
    strategy: str = "ddp"
    precision: str = "16-mixed"
    sync_batchnorm: bool = True
    ckpt_path: str = ""
    default_root_dir: str = ""
    debug_log_dir: str = "debug_logs/"
    wandb_run_name: str = ""
    wandb_save_dir: str = ""

    def __post_init__(self):
        self.betas = [0.9, 0.999]


@dataclass
class InferConfig:
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42
    precision: str = "bf16"
    sample_rate: int = 16000
    max_wait_num: int = 10
    max_mistake_num: int = 5
    far_field_threshold: float = 0.02
    developer_mode: bool = False
    single_round: bool = False
    config_path: str = "config/config.yaml"

    input: dict = field(
        default_factory=lambda: {
            "chunk_size": 2560,
            "audio_back_size": 15360,
            "audio_ahead_size": 640,
            "sample_rate": 16000,
            "chunk_token_len_small": 2,
        }
    )
    asr: dict = field(
        default_factory=lambda: {
            "model_name": "paraformer",
            "language": "auto",
            "max_chunk_token_length": 256,
        }
    )


@dataclass
class RunConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DataConfig = field(default_factory=DataConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    infer_config: InferConfig = field(default_factory=InferConfig)
