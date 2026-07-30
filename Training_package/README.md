# StepAudio Full-Duplex Training

This package contains the training code for the StepAudio full-duplex model,
including supervised text, speech-token, control-token, and backchannel training.

## Contents

```
Training_package/
├── README.md
├── package_training_code.sh
├── Script/
│   ├── StepAudioFullDuplexV9_9.sh
│   └── deepspeed_zero2_high.conf
└── Code/
    ├── Train_StepAudioFullDuplexV9.py
    ├── training_utils.py
    ├── Models/
    │   ├── StepAudioDuplexV9.py
    │   ├── modeling_step_audio_2.py
    │   └── configuration_step_audio_2.py
    └── DataLoaders/
        ├── StepAudioDuplexDatasetV8.py
        └── datasets_utils.py
```

## Features

- Full-duplex text / speech-token / control-token training.
- User backchannel insertion during assistant speaking.
- Assistant backchannel insertion during user speaking.
- Control-label masking that keeps transition tokens and backchannel control
  tokens.
- Multi-dataset loading from HuggingFace `load_from_disk`, with optional
  `@ratio` sampling in `--data_path`.

## Data

Datasets are loaded from local HuggingFace `Dataset` directories. Each dataset
entry should provide conversation turns, user audio paths, assistant text, and
speech-token paths. Backchannel fields are used when `--enable_user_bc` or
`--enable_ai_bc` is enabled.

## Run

Edit paths in `Script/StepAudioFullDuplexV9_9.sh` first:

- `SAVE_PATH`
- `MODEL_PATH`
- `--data_path`
- `--deepspeed`
- Python or conda environment setup

Then launch:

```bash
bash Script/StepAudioFullDuplexV9_9.sh
```

The script automatically resumes from the latest `checkpoint-<step>` under
`SAVE_PATH`. If no checkpoint is found, it initializes from the base
StepAudio-2 model and adds the full-duplex special tokens.

## Main Files

- `Code/Train_StepAudioFullDuplexV9.py`: argument parsing, model
  initialization, tokenizer setup, and HuggingFace `Trainer` startup.
- `Code/DataLoaders/StepAudioDuplexDatasetV8.py`: dataset loading, turn
  assembly, backchannel insertion, audio feature extraction, and data collator.
- `Code/Models/StepAudioDuplexV9.py`: full-duplex model and losses.
- `Script/StepAudioFullDuplexV9_9.sh`: example DeepSpeed launch command.
