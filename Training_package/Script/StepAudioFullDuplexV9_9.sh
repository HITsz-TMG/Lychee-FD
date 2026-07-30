
#!/bin/bash

# oripath=$PWD
# echo $oripath

# echo "----------------------- INITIALIZE -----------------------"

# wget <PROXY_SETUP_URL> -O enable_internet_proxy.sh
# bash enable_internet_proxy.sh
# source ~/.bashrc

# sudo ln -s /path/to/shared_storage /path/to/workspace

# source /path/to/miniconda3/etc/profile.d/conda.sh

# conda activate full_duplex

# echo "conda activate full_duplex"


# echo "----------------------- PIP --------------------------"


# pip install pip==24.0

# pip install triton==3.1.0
# pip install -r /path/to/requirements.txt
# pip install qwen-omni-utils
# pip install -U venus-api-base -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com
# pip install -r /path/to/optional_requirements.txt
# pip install flash-attn --no-build-isolation

# pip list

# echo "----------------------- ARGS --------------------------"

# for arg in "$@"; do
#   case $arg in
#     --nproc_per_node=*)
#       export NPROC_PER_NODE="${arg#*=}"
#       ;;
#     --nnodes=*)
#       export NNODES="${arg#*=}"
#       ;;
#     --node_rank=*)
#       export NODE_RANK="${arg#*=}" # accelerate launch checks NODE_RANK or MACHINE_RANK
#       ;;
#     --master_addr=*)
#       export MASTER_ADDR="${arg#*=}"
#       ;;
#     --master_port=*)
#       export MASTER_PORT="${arg#*=}"
#       ;;
#     *)
#       # Collect remaining args and pass them to the Python script.
#       SCRIPT_ARGS+=" $arg"
#       ;;
#   esac
# done

# # Check whether all required environment variables are set.
# if [ -z "$NPROC_PER_NODE" ] || [ -z "$NNODES" ] || [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
#     echo "Error: Missing required parameters (--nproc_per_node, --nnodes, --node_rank, --master_addr, --master_port)"
#     exit 1
# fi

# echo "Setting environment variables:"
# echo "  NODE_RANK=$NODE_RANK"
# echo "  NNODES=$NNODES"
# echo "  NPROC_PER_NODE=$NPROC_PER_NODE"
# echo "  MASTER_ADDR=$MASTER_ADDR"
# echo "  MASTER_PORT=$MASTER_PORT"
# echo "  SCRIPT_ARGS=$SCRIPT_ARGS"

echo "----------------------- WANDB --------------------------"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

export WANDB_API_KEY=''
export WANDB_PROJECT='full_duplex'

# **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
# Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
# parameters.
export WANDB_WATCH='false'

# WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
# Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
# to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
# will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
# with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.
export WANDB_LOG_MODEL='false'

time=$(date "+%m-%d-%H-%M")

export NAME="stepaudio_full_duplex_v9_9"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/huggingface_cache}"

export SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/Outputs/${WANDB_PROJECT}/${NAME}}" 

echo "----------------------- RESUME --------------------------"


# Initialize resume state.
INITIALIZE=False
CHECKPOINT_PATH=""
max_step=-1 # Used for step comparison; any non-negative step is larger.

# Check whether SAVE_PATH exists and is a directory.
if [ ! -d "$SAVE_PATH" ]; then
  echo "Error: SAVE_PATH '$SAVE_PATH' does not exist or is not a directory." >&2
  INITIALIZE=True
  CHECKPOINT_PATH=""
  # Exit here if your environment requires SAVE_PATH to already exist.
else
  # Use find to locate directories named like "checkpoint-<step>".
  # -maxdepth 1 searches only directly under SAVE_PATH.
  # -type d restricts matches to directories.
  # -name "checkpoint-*" matches checkpoint directory names.
  # -print0 plus read -d $'\0' safely handles paths with spaces or special chars.
  found_one=0
  while IFS= read -r -d $'\0' dir_path; do
    # Extract the directory name, for example "checkpoint-16350".
    dir_name=$(basename "$dir_path")

    # Extract the numeric step by removing the "checkpoint-" prefix.
    step_str="${dir_name##checkpoint-}"

    # Validate that the extracted step is numeric.
    if [[ "$step_str" =~ ^[0-9]+$ ]]; then
    current_step=$((step_str)) # Convert to integer.
    found_one=1 # Mark that at least one valid checkpoint was found.

    if (( current_step > max_step )); then
        max_step=$current_step
        CHECKPOINT_PATH="$dir_path" # Track the checkpoint with the largest step.
    fi
    else
    echo "Warning: Found directory '$dir_name' that looks like a checkpoint but has an invalid step number." >&2
    fi
  done < <(find "$SAVE_PATH" -maxdepth 1 -type d -name "checkpoint-*" -print0 2>/dev/null)
  # 2>/dev/null suppresses find errors; SAVE_PATH was already checked above.

  # If no valid checkpoint directory was found, start from the base model.
  if [ "$found_one" -eq 0 ]; then
      INITIALIZE=True
      CHECKPOINT_PATH=""
  fi
fi

# Set MODEL_PATH.
if [ -z "$CHECKPOINT_PATH" ]; then # Check whether CHECKPOINT_PATH is empty.
  MODEL_PATH="${MODEL_PATH:-/path/to/models/stepfun-ai__Step-Audio-2-mini}" # Change this default path as needed.
  RESUME_FROM_CHECKPOINT_ARG="" # Do not pass --resume_from_checkpoint when empty.
else
  MODEL_PATH="$CHECKPOINT_PATH"
  RESUME_FROM_CHECKPOINT_ARG="--resume_from_checkpoint $CHECKPOINT_PATH" # Resume when a checkpoint exists.
fi

# Print the resolved resume settings.
echo "INITIALIZE=${INITIALIZE}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "MODEL_PATH=${MODEL_PATH}"

echo "----------------------- RUN --------------------------"

DATA_PATH="${DATA_PATH:-\
/path/to/datasets/conversation_1@1.0;\
/path/to/datasets/conversation_2@1.0;\
/path/to/datasets/conversation_3@1.0;\
/path/to/datasets/conversation_4@1.0;\
/path/to/datasets/conversation_5@1.0;\
}"

export DS_SKIP_CUDA_CHECK=1 
export NCCL_TIMEOUT=1200          # Seconds; read by some NCCL versions.
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Assumes deepspeed is available in the active environment.
deepspeed \
    --master_addr "localhost" \
    --master_port 9269 \
    "${PROJECT_ROOT}/Code/Train_StepAudioFullDuplexV9.py" \
    --attn_implementation flash_attention_2 \
    --deepspeed "${PROJECT_ROOT}/Script/deepspeed_zero2_high.conf" \
    --initialize $INITIALIZE \
    $RESUME_FROM_CHECKPOINT_ARG \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --max_data_length 6000 \
    --control_token_chunk_size 10 \
    --window_second 0.4 \
    --control_token_use_focal_loss False \
    --stoken_layer_num 4 \
    --merge_layer_num 4 \
    --control_layer_num 6 \
    --control_branch_layer 12 \
    --stoken_delay_num 0 \
    --enable_user_bc True \
    --enable_ai_bc False \
    --ai_bc_lead_silence_sec 1.0 \
    --ai_bc_min_gap_sec 3.0 \
    --ai_bc_max_num 5 \
    --adding_text_hiddenstates False \
    --align_audio_input True \
    --output_dir "$SAVE_PATH" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate 3e-6 \
    --weight_decay 0.00 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 64096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to "none" \
    --run_name "${NAME}_${time}"

#  nohup bash /path/to/project/Script/StepAudioFullDuplexV9_9.sh > /path/to/logs/StepAudioFullDuplexV9_9.txt 2>&1 &

