# Full-Duplex Speech Language Model Training

This repository contains the training code for our full-duplex speech language model, which enables simultaneous listening and speaking capabilities with turn-taking control.

## Overview

Our model extends a speech-language backbone (StepAudio2) with:
- **Control Head**: Predicts turn-taking decisions (start-speaking, keep-listening, start-listening, keep-speaking, backchannel)
- **Stoken Head**: Generates speech tokens in parallel with text tokens
- **Multi-stream Architecture**: Backbone + lightweight branch decoders for control and speech token prediction

## Environment Setup

### Requirements

```bash
# Core dependencies
torch>=2.1.0
transformers==4.53.1
deepspeed>=0.12.0
datasets
tokenizers
peft
librosa
torchaudio
soundfile
pyloudnorm
safetensors
wandb
numpy
Pillow
torchvision
```

### Installation

```bash
pip install -r requirements.txt
```

## Data Format

Training data should be prepared as a HuggingFace `datasets` disk format (via `datasets.save_to_disk()`). Multiple datasets can be specified by separating paths with `;`.

### Expected Data Schema

Each sample should contain:

```python
{
    "conversation": {
        "conversation_history": [
            {
                "user_utterance": str,           # User's text (may contain <backchannel>...</backchannel> tags)
                "ai_response": str,              # AI response text (may contain <interruption>...</interruption> tags)
                "ai_response_token": {
                    "Main": str or list,         # Speech token codec for main response
                    "BeforeInterruption": str or list,  # (optional) codec before interruption
                    "AfterInterruption": str or list,   # (optional) codec after interruption
                    "Backchannel": str or list,         # (optional) codec for AI backchannel
                },
                "speech": {
                    "user_speech": {
                        "Main": str,             # Path to user's main speech audio file
                        "Backchannel": str,      # (optional) Path to user's backchannel audio
                    },
                    "ai_speech": {
                        "Backchannel": str,      # (optional) Path to AI backchannel audio
                    }
                },
                "event": list,                   # Event tags: ["User_Interruption", "AI_Backchannel", "User_Backchannel"]
            },
            ...
        ]
    }
}
```

### Audio Requirements
- Sample rate: 16000 Hz
- Format: WAV (mono)
- Speech tokens: Integer codec IDs (offset by 151696 in vocabulary)


## Training

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name_or_path` | Path to pretrained StepAudio2 model | Required |
| `--initialize` | Whether to initialize from base model (True) or load full-duplex checkpoint (False) | `False` |
| `--data_path` | Path(s) to training data, separated by `;` | Required |
| `--control_token_chunk_size` | Number of tokens per control decision chunk | Required |
| `--stoken_layer_num` | Number of layers for stoken branch decoder | `0` |
| `--control_layer_num` | Number of layers for control branch decoder | `0` |
| `--adding_text_hiddenstates` | Add text hidden states to stoken branch | `False` |
| `--no_text_label` | Disable text loss (train control/stoken only) | `False` |
| `--no_stoken_label` | Disable stoken loss | `False` |
| `--stoken_delay_num` | Number of delay tokens for stoken | `0` |
| `--window_second` | Audio window size in seconds | `24` |
| `--ignore_backchannel` | Ignore backchannel events in data | `False` |
| `--only_training_stoken` | Only train stoken branch parameters | `False` |
| `--max_data_length` | Maximum sequence length for data truncation | `10000` |

### Launch Training (GPU)

```bash
deepspeed --num_gpus=8 train.py \
    --model_name_or_path /path/to/stepaudio2_base \
    --data_path /path/to/dataset1;/path/to/dataset2 \
    --initialize True \
    --control_token_chunk_size 25 \
    --stoken_layer_num 4 \
    --control_layer_num 4 \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --model_max_length 12000 \
    --deepspeed ds_config.json \
    --save_steps 500 \
    --logging_steps 1 \
    --report_to wandb
```


### DeepSpeed Configuration Example

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true
    },
    "bf16": {
        "enabled": "auto"
    },
    "checkpoint": {
        "tag_validation": "ignore"
    }
}
```

## Special Tokens

The model uses the following special tokens for full-duplex control:

| Token | Meaning |
|-------|---------|
| `<\|S-S\|>` | Start Speaking |
| `<\|S-L\|>` | Start Listening |
| `<\|K-L\|>` | Keep Listening |
| `<\|K-S\|>` | Keep Speaking |
| `<\|Detect\|>` | Detection point (precedes control decision) |
| `<\|Sleep\|>` | Sleep/idle state |
| `<\|TextPad\|>` | Text padding during listening |
| `<\|StokenPad\|>` | Speech token padding |
| `<\|AudioPad\|>` | Audio input padding |
| `<\|StokenDelay\|>` | Speech token delay |
| `<\|BackChannel\|>` | Backchannel signal |
