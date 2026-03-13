#!/bin/bash
#
# SLURM batch script for TDC GRPO training on NeMo RL.
#
# Mirrors the directory structure and feature flags from the OpenRLHF
# train_grpo_tdc_gpt_oss_slurm.sh script, but targets the NeMo RL codebase.
#
# Features ported from OpenRLHF:
#   - Local tracing directory (traces/, eval_metrics/, vllm_stats/, etc.)
#   - WandB panel routing (train/, parse/, vllm/, system/)
#   - Optional Liger Triton GRPO loss
#   - TIS/ICEPOP importance sampling correction
#   - Tool-calling environment with GPT-OSS protocol
#
# Usage:
#   # Single-turn TDC with Qwen 1.5B (1 GPU, quick test):
#   sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
#
#   # Multi-turn tool calling with Liger loss on 2 GPUs:
#   LIGER_GRPO_LOSS=1 NUM_GPUS=2 MULTI_TURN=1 \
#       sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
#
#   # Custom model + TIS:
#   MODEL=unsloth/gpt-oss-20b-BF16 NUM_GPUS=2 TIS=1 \
#       sbatch scripts/train_grpo_tdc_nemorl_slurm.sh
#
# Feature flags (set via env before sbatch):
#   MODEL                 Model name/path (default: Qwen/Qwen2.5-1.5B)
#   NUM_GPUS              GPUs per node (default: 1)
#   NUM_NODES             Nodes (default: 1)
#   MULTI_TURN            Enable multi-turn tool calling (default: 0)
#   MAX_TURNS             Max tool-calling turns (default: 10)
#   CHAT_PROTOCOL         Chat protocol for tool calling (default: gpt_oss)
#   LIGER_GRPO_LOSS       Enable Liger Triton GRPO loss (default: 0)
#   TIS                   Enable truncated importance sampling (default: 0)
#   TIS_TYPE              TIS variant: tis|icepop|seq-mask-tis (default: tis)
#   TIS_THRESHOLDS        Low/high thresholds (default: "0.5 5.0")
#   LEARNING_RATE         Optimizer LR (default: 1e-6)
#   MAX_STEPS             Max training steps (default: 500)
#   MAX_EPOCHS            Max epochs (default: 1)
#   TRAIN_DATA            Path to training JSONL
#   VAL_DATA              Path to validation JSONL
#   TASK_NAMES            Space-separated task names (default: BBB_Martins)
#   WANDB_PROJECT         WandB project (default: nemorl-tdc-grpo)
#   EXTRA_OVERRIDES       Additional Hydra overrides
#

### SLURM PARAMETERS ###
#SBATCH --job-name=nemorl-tdc-grpo
#SBATCH --output=logs/nemorl-tdc-grpo_%j.out
#SBATCH --error=logs/nemorl-tdc-grpo_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=384G
#SBATCH --cpus-per-task=32
#SBATCH --time=00-4:00:00

### PARCC PARAMETERS ###
export OMP_NUM_THREADS=16
export NCCL_NVLS_ENABLE=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_SL=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bond0
export UCX_TLS=rc
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

### ENVIRONMENT SETUP ###
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/vast/projects/myatskar/design-documents/conda_env/openrlhf}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.1}"
module load MAMBA
module load "$CUDA_MODULE"
export CONDA_ENV_PATH

############################
#        TASK SCRIPT       #
############################
run_task() {
    set -euo pipefail

    ### PROJECT ROOT ###
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [ "$PROJECT_ROOT" != "/" ] && [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; do
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    done
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo "Error: Cannot find RL-Tools project root" >&2
        exit 1
    fi
    cd "$PROJECT_ROOT"

    # OpenRLHF-Tools repo (for data)
    OPENRLHF_ROOT="${OPENRLHF_ROOT:-$(realpath ../OpenRLHF-Tools)}"

    ### ARGS ###
    MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
    NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
    NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
    MAX_STEPS="${MAX_STEPS:-500}"
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
    NUM_PROMPTS="${NUM_PROMPTS:-8}"
    NUM_GENS="${NUM_GENS:-8}"
    TRAIN_GLOBAL_BS="${TRAIN_GLOBAL_BS:-64}"
    VAL_PERIOD="${VAL_PERIOD:-32}"

    ### FEATURE FLAGS ###
    MULTI_TURN="${MULTI_TURN:-0}"
    MAX_TURNS="${MAX_TURNS:-10}"
    CHAT_PROTOCOL="${CHAT_PROTOCOL:-gpt_oss}"
    LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
    TIS="${TIS:-0}"
    TIS_TYPE="${TIS_TYPE:-tis}"
    TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

    ### TASK DATA ###
    TASK_NAMES=(${TASK_NAMES:-BBB_Martins})
    DATA_DIR="$OPENRLHF_ROOT/data/tdc/openai_format_gpt_oss"

    # Build training data — concatenate if multiple tasks
    TRAIN_PARTS=()
    VAL_PARTS=()
    for t in "${TASK_NAMES[@]}"; do
        f="$DATA_DIR/${t}_train.jsonl"
        if [ ! -f "$f" ]; then
            echo "Error: Training data not found: $f" >&2
            echo "Available:" >&2
            ls "$DATA_DIR" 2>/dev/null | grep "_train.jsonl" | sed 's/_train.jsonl//' | sort >&2
            exit 1
        fi
        TRAIN_PARTS+=("$f")
        vf="$DATA_DIR/${t}_val.jsonl"
        [ -f "$vf" ] && VAL_PARTS+=("$vf")
    done

    if [ ${#TASK_NAMES[@]} -eq 1 ]; then
        # Single task — use file directly
        TRAIN_DATA="${TRAIN_DATA:-${TRAIN_PARTS[0]}}"
        VAL_DATA="${VAL_DATA:-${VAL_PARTS[0]:-}}"
    else
        # Multiple tasks — concatenate into a combined file
        COMBINED_DIR="$LOG_DIR/combined_data"
        mkdir -p "$COMBINED_DIR"
        TRAIN_DATA="${TRAIN_DATA:-$COMBINED_DIR/multi_task_train.jsonl}"
        if [ ! -f "$TRAIN_DATA" ] || [ "$TRAIN_DATA" = "$COMBINED_DIR/multi_task_train.jsonl" ]; then
            cat "${TRAIN_PARTS[@]}" > "$COMBINED_DIR/multi_task_train.jsonl"
            TRAIN_DATA="$COMBINED_DIR/multi_task_train.jsonl"
            echo "Concatenated ${#TRAIN_PARTS[@]} train files -> $TRAIN_DATA"
        fi
        if [ -z "${VAL_DATA:-}" ] && [ ${#VAL_PARTS[@]} -gt 0 ]; then
            cat "${VAL_PARTS[@]}" > "$COMBINED_DIR/multi_task_val.jsonl"
            VAL_DATA="$COMBINED_DIR/multi_task_val.jsonl"
            echo "Concatenated ${#VAL_PARTS[@]} val files -> $VAL_DATA"
        fi
    fi

    if [ -z "${VAL_DATA:-}" ] || [ ! -f "$VAL_DATA" ]; then
        echo "Warning: Val data not found, disabling validation"
        VAL_DATA=""
    fi

    ### WANDB ###
    WANDB_PROJECT="${WANDB_PROJECT:-nemorl-tdc-grpo}"
    DATE_TAG=$(date +%m%d_%H%M)
    N_TASKS=${#TASK_NAMES[@]}
    RUN_NAME="nemorl-tdc-${N_TASKS}t-${MODEL##*/}-${DATE_TAG}"

    ### LOG DIRS ###
    LOG_DIR="logs/grpo_tdc/${RUN_NAME}"
    CKPT_DIR="results/grpo_tdc/${RUN_NAME}"
    mkdir -p "$LOG_DIR" logs

    ### RAY SETUP ###
    export RAY_TMPDIR="/tmp/ray_${USER}/${SLURM_JOB_ID:-$$}"
    mkdir -p "$RAY_TMPDIR"
    PERSIST_RAY_DIR="$PROJECT_ROOT/logs/ray_logs/${SLURM_JOB_ID:-$$}"

    # Copy Ray logs on exit
    copy_ray_logs() {
        set +e
        local real="$RAY_TMPDIR/ray/session_latest"
        [ -L "$real" ] && real="$(readlink -f "$real")"
        mkdir -p "$PERSIST_RAY_DIR"
        cp -a "$real" "$PERSIST_RAY_DIR/session_latest" 2>/dev/null || true
        echo "Ray logs copied to $PERSIST_RAY_DIR"
    }
    trap copy_ray_logs EXIT

    # Prevent stale torch/vLLM caches
    rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
    mkdir -p "$TRITON_CACHE_DIR"
    export VLLM_NO_USAGE_STATS=1
    export VLLM_DISABLE_TELEMETRY=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1

    ### START RAY ###
    export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
    ulimit -n 65535 2>/dev/null || true

    CONDA_RAY="$(which python) -m ray.scripts.scripts"
    echo "Using python: $(which python)"
    $CONDA_RAY stop --force 2>/dev/null || true
    rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

    echo "Starting Ray head at $RAY_NODE_IP_ADDRESS (GPUs=$NUM_GPUS)"
    $CONDA_RAY start --head \
        --node-ip-address "$RAY_NODE_IP_ADDRESS" \
        --num-gpus "$NUM_GPUS" \
        --temp-dir "$RAY_TMPDIR"
    export RAY_ADDRESS="$RAY_NODE_IP_ADDRESS:6379"

    echo "Waiting for Ray..."
    for i in $(seq 1 60); do
        $CONDA_RAY status >/dev/null 2>&1 && break
        sleep 1
    done
    $CONDA_RAY status || { echo "Error: Ray never came up" >&2; exit 1; }
    echo "Ray is ready."

    ### BUILD HYDRA OVERRIDES ###
    OVERRIDES=(
        "policy.model_name=$MODEL"
        "policy.optimizer.kwargs.lr=$LEARNING_RATE"
        "policy.max_total_sequence_length=$MAX_SEQ_LEN"
        "policy.train_global_batch_size=$TRAIN_GLOBAL_BS"
        "grpo.num_prompts_per_step=$NUM_PROMPTS"
        "grpo.num_generations_per_prompt=$NUM_GENS"
        "grpo.max_num_steps=$MAX_STEPS"
        "grpo.max_num_epochs=$MAX_EPOCHS"
        "grpo.val_period=$VAL_PERIOD"
        "cluster.gpus_per_node=$NUM_GPUS"
        "cluster.num_nodes=$NUM_NODES"
        "logger.log_dir=$LOG_DIR"
        "logger.wandb.project=$WANDB_PROJECT"
        "logger.wandb.name=$RUN_NAME"
        "checkpointing.checkpoint_dir=$CKPT_DIR"
        "data.train.data_path=$TRAIN_DATA"
    )

    # Validation data
    if [ -n "$VAL_DATA" ]; then
        OVERRIDES+=("data.validation.data_path=$VAL_DATA"
                     "data.validation.input_key=messages"
                     "data.validation.output_key=answer")
    else
        OVERRIDES+=("grpo.val_period=0" "grpo.val_at_start=false" "grpo.val_at_end=false")
    fi

    # Multi-turn tool calling
    if [ "$MULTI_TURN" = "1" ]; then
        OVERRIDES+=(
            "grpo.max_rollout_turns=$MAX_TURNS"
            "data.default.env_name=tool_calling"
            "env.tool_calling.chat_protocol=$CHAT_PROTOCOL"
            "env.tool_calling.max_turns=$MAX_TURNS"
        )
        echo "Multi-turn tool calling enabled (max_turns=$MAX_TURNS, protocol=$CHAT_PROTOCOL)"
    fi

    # Liger Triton GRPO loss
    if [ "$LIGER_GRPO_LOSS" = "1" ]; then
        OVERRIDES+=(
            "loss_fn.use_liger_grpo_loss=true"
            "loss_fn.clip_eps_low=0.2"
            "loss_fn.clip_eps_high=0.2"
            "loss_fn.beta=0.01"
            "loss_fn.temperature=1.0"
            "loss_fn.loss_type=grpo"
        )
        echo "Liger Triton GRPO loss enabled"
    fi

    # Truncated Importance Sampling
    if [ "$TIS" = "1" ]; then
        read -r TIS_LOW TIS_HIGH <<< "$TIS_THRESHOLDS"
        OVERRIDES+=(
            "loss_fn.use_importance_sampling_correction=true"
            "loss_fn.truncated_importance_sampling_ratio=$TIS_HIGH"
            "loss_fn.truncated_importance_sampling_ratio_min=$TIS_LOW"
            "loss_fn.truncated_importance_sampling_type=$TIS_TYPE"
        )
        echo "TIS enabled (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
    fi

    # Extra overrides
    if [ -n "$EXTRA_OVERRIDES" ]; then
        # shellcheck disable=SC2206
        OVERRIDES+=($EXTRA_OVERRIDES)
    fi

    ### PRINT CONFIG ###
    echo "========================================"
    echo "NeMo RL — TDC GRPO Training"
    echo "========================================"
    echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
    echo "Model: $MODEL"
    echo "Tasks: ${TASK_NAMES[*]}"
    echo "GPUs: $NUM_GPUS x $NUM_NODES nodes"
    echo "Train data: $TRAIN_DATA"
    echo "Val data: ${VAL_DATA:-disabled}"
    echo "Log dir: $LOG_DIR"
    echo "Checkpoint dir: $CKPT_DIR"
    echo "----------------------------------------"
    echo "LR: $LEARNING_RATE"
    echo "Max steps: $MAX_STEPS"
    echo "Prompts/step: $NUM_PROMPTS  Gens/prompt: $NUM_GENS"
    echo "Multi-turn: $MULTI_TURN  Liger: $LIGER_GRPO_LOSS  TIS: $TIS"
    echo "----------------------------------------"
    echo "W&B: project=$WANDB_PROJECT run=$RUN_NAME"
    echo "========================================"
    echo ""
    echo "Overrides:"
    printf "  %s\n" "${OVERRIDES[@]}"
    echo ""

    ### RUN TRAINING ###
    python examples/run_grpo.py \
        --config examples/configs/grpo_tdc_tool_calling.yaml \
        "${OVERRIDES[@]}"

    ### CLEANUP ###
    echo "Training complete! Stopping Ray..."
    $CONDA_RAY stop --force || true
}
############################
export -f run_task

mkdir -p logs
srun micromamba run -p "$CONDA_ENV_PATH" bash -c "run_task"
